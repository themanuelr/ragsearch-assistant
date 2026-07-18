"""
Note generation pipeline: analysis fill (7 LLM calls) + deterministic render + vault write.

Reads a PaperJSON v2 dict (in-memory or from cache file), runs 7 gemma4:e4b
analysis calls (summary, claims, methods_overview, results, limitations, topics,
open_questions), renders a deterministic Obsidian note with YAML frontmatter and
callouts, and writes it to Papers/<sanitized title>.md via the obsidian_cli
chokepoint.

Run standalone:  python scripts/note.py --paperjson <cache>.json [--force]
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pydantic import BaseModel, Field

from scripts.ingest import (
    _ollama_extraction_call,
    _parse_extraction_response,
    _estimate_num_ctx,
    _warmup_ollama,
    _load_config,
    _write_paperjson_cache,
    OLLAMA_MODEL,
)
from scripts.obsidian_cli import preflight, note_exists, create_note, _validate_vault_path

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

DEFAULT_SECTION_TIMEOUT = 300  # seconds per analysis call
# Mutable module global; set by generate_note() from config.json key
# "ollama_section_timeout" (mirrors ingest.py).  Threaded into _analysis_call.
_SECTION_TIMEOUT: int = DEFAULT_SECTION_TIMEOUT
_WINDOWS_ILLEGAL = re.compile(r'[/\\:*?"<>|]')

# Windows reserved device names — illegal as a file's base name regardless of
# extension or parent directory.  A paper titled "NUL"/"Aux"/"COM1" would target
# the device, not a file, so the note write is silently lost or errors (WR-04).
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

# Obsidian tag slug: collapse any maximal run of characters NOT in the
# allowed set (Unicode word chars, forward slash, hyphen) to a single hyphen.
# Forward slash is preserved so "input/output" becomes a nested Obsidian tag.
# Precompiled for repeated use in _slugify_tag (GAP-1 / NOTE-02).
_TAG_INVALID_RUN = re.compile(r'[^\w/-]+')

# Math-span regex: matches $$...$$ (display) and $...$ (inline) spans, including
# multi-line content (re.S so . and [^$] span newlines).  Used by the LaTeX
# escape repair to scope substitutions to math only (Task 2b).
_MATH_SPAN = re.compile(r'\$\$.*?\$\$|\$[^$]*?\$', re.S)

# Mapping of control chars the JSON decoder silently inserts for single-backslash
# LaTeX sequences emitted by gemma4:e4b → their intended backslash-escape forms.
_CTRL_TO_ESCAPE: dict[str, str] = {
    '\t': '\\t',   # \text, \to, \theta …
    '\n': '\\n',   # \nabla, \nu …
    '\r': '\\r',   # \rightarrow, \rightleftharpoons …
    '\x08': '\\b', # \beta (rare)
    '\x0c': '\\f', # \frac (rare)
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Emit a timestamped progress line to stderr."""
    print(
        f"[note {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Pydantic models — one per analysis call
# ---------------------------------------------------------------------------

class AnalysisSummary(BaseModel):
    summary: str = Field(..., description="2-3 paragraph summary of the paper")

class AnalysisClaims(BaseModel):
    claims: list[str] = Field(..., description="Key findings/contributions as a list")

class AnalysisMethods(BaseModel):
    methods_overview: str = Field(..., description="Overview of methodology")

class AnalysisResults(BaseModel):
    results: str = Field(..., description="Key results and outcomes")

class AnalysisLimitations(BaseModel):
    limitations: list[str] = Field(..., description="Limitations and caveats")

class AnalysisTopics(BaseModel):
    topics: list[str] = Field(..., description="Research topics/tags")

class AnalysisOpenQuestions(BaseModel):
    open_questions: list[str] = Field(..., description="Open questions raised by the paper")


# ---------------------------------------------------------------------------
# Paper text assembly
# ---------------------------------------------------------------------------

def _assemble_paper_text(paperjson: dict) -> str:
    """Build the cleaned paper text for analysis calls from PaperJSON sections.

    Reads the post-fill ``section.get("body")`` shape.  Falls back to joining
    text-block ``plain``/``display`` if ``body`` is absent (forward-compat).
    """
    parts = []
    for section in paperjson["extraction"]["sections"]:
        heading = section.get("heading", "")
        body = section.get("body", "")
        if not body:
            # Forward-compat: join text blocks if body is absent
            blocks = section.get("blocks", [])
            body = "\n".join(
                b.get("plain", b.get("display", ""))
                for b in blocks if isinstance(b, dict)
            )
        if heading:
            parts.append(f"## {heading}")
        if body:
            parts.append(body)
        parts.append("")  # blank line between sections
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Generic two-strike analysis caller
# ---------------------------------------------------------------------------

def _analysis_call(prompt: str, system: str, model_cls: type[BaseModel],
                   paper_text: str) -> BaseModel | None:
    """Two-strike analysis caller mirroring _fill_section pattern.

    Uses ``output_ratio=0.0`` (analysis output is far shorter than input).
    On ``[Ollama error:]`` raises RuntimeError (fail-fast, D-13).
    On ``[Ollama timeout:]`` consumes a strike.
    After both strikes returns None.
    """
    schema = model_cls.model_json_schema()
    num_ctx = _estimate_num_ctx(paper_text, overhead=2048, output_ratio=0.0)

    for attempt in range(2):
        raw = _ollama_extraction_call(
            prompt, system, schema,
            num_ctx=num_ctx,
            timeout=_SECTION_TIMEOUT,
        )
        if raw.startswith("[Ollama error:"):
            raise RuntimeError(raw)  # Ollama unreachable -> abort pipeline
        if raw.startswith("[Ollama timeout:"):
            continue  # timeout is a strike
        result = _parse_extraction_response(raw, model_cls)
        if result is not None:
            return result
        if attempt == 0:
            prompt += "\n\nIMPORTANT: Return ONLY raw JSON. Do not wrap in code fences."

    return None  # fill_failed for this field


# ---------------------------------------------------------------------------
# Analysis generation — 7 LLM calls
# ---------------------------------------------------------------------------

# Per-field call specs: (field_name, model_class, system_prompt, user_prompt_template)
# user_prompt_template is an f-string fragment receiving {paper_text}

_ANALYSIS_SPECS = [
    (
        "summary",
        AnalysisSummary,
        (
            "You are a scientific paper analyst. "
            "Write a concise summary of this paper's main contribution, methods, and findings. "
            "Ground every claim in the provided text -- do not invent. "
            "Return ONLY valid JSON matching the schema."
        ),
        "Summarize the following research paper:\n\n{paper_text}",
    ),
    (
        "claims",
        AnalysisClaims,
        (
            "You are a scientific paper analyst. "
            "Extract the key findings and contributions of this paper as a list. "
            "Each claim must be grounded in the text -- do not invent. "
            "Return ONLY valid JSON matching the schema."
        ),
        "List the key findings and contributions of the following research paper:\n\n{paper_text}",
    ),
    (
        "methods_overview",
        AnalysisMethods,
        (
            "You are a scientific paper analyst. "
            "Provide an overview of the methodology used in this paper. "
            "Ground every claim in the provided text -- do not invent. "
            "Return ONLY valid JSON matching the schema."
        ),
        "Describe the methodology of the following research paper:\n\n{paper_text}",
    ),
    (
        "results",
        AnalysisResults,
        (
            "You are a scientific paper analyst. "
            "Describe the key results and outcomes of this paper. "
            "Ground every claim in the provided text -- do not invent. "
            "Return ONLY valid JSON matching the schema."
        ),
        "Describe the key results of the following research paper:\n\n{paper_text}",
    ),
    (
        "limitations",
        AnalysisLimitations,
        (
            "You are a scientific paper analyst. "
            "List the limitations, caveats, and weaknesses acknowledged or evident in this paper. "
            "Ground every claim in the provided text -- do not invent. "
            "Return ONLY valid JSON matching the schema."
        ),
        "List the limitations of the following research paper:\n\n{paper_text}",
    ),
    (
        "topics",
        AnalysisTopics,
        (
            "You are a scientific paper analyst. "
            "Extract the research topics and tags relevant to this paper. "
            "Use concise topic names suitable for Obsidian tags. "
            "Return ONLY valid JSON matching the schema."
        ),
        "List the research topics covered in the following paper:\n\n{paper_text}",
    ),
    (
        "open_questions",
        AnalysisOpenQuestions,
        (
            "You are a scientific paper analyst. "
            "List the open questions raised by this paper or left unanswered. "
            "Ground every claim in the provided text -- do not invent. "
            "Return ONLY valid JSON matching the schema."
        ),
        "List the open questions raised by the following research paper:\n\n{paper_text}",
    ),
]


def _repair_math_escapes(text: str) -> str:
    """Within $$...$$ and $...$ math spans only, map decoded control chars back to
    their backslash-escape forms.

    gemma4:e4b emits LaTeX with single backslashes in its analysis JSON output
    (e.g. ``"\\text{Na}^+"``).  The JSON decoder silently converts these to
    control characters (``\\t`` → TAB, ``\\r`` → CR, ``\\n`` → LF).  Evidence
    across 3 real papers showed all 72 such control chars fell inside ``$...$``
    math spans; 0 were in prose; 266 genuine paragraph newlines were all
    outside math.  Scoping the repair to math spans is therefore safe (zero
    false positives) and complete.

    Text outside math spans — including genuine paragraph-break newlines — is
    untouched.  The function is idempotent and a no-op on text without ``$``.
    """
    if '$' not in text:
        return text

    def _fix_span(m: re.Match) -> str:
        span = m.group(0)
        for ctrl, escaped in _CTRL_TO_ESCAPE.items():
            span = span.replace(ctrl, escaped)
        return span

    return _MATH_SPAN.sub(_fix_span, text)


def _repair_analysis_value(value):
    """Apply _repair_math_escapes to str or list[str] analysis values; pass others through.

    Args:
        value: the raw value extracted from an analysis model field — may be
               ``str`` (summary, methods_overview, results) or ``list[str]``
               (claims, limitations, topics, open_questions), or ``None``.

    Returns:
        Repaired str / list[str], or the original value unchanged for other types.
    """
    if isinstance(value, str):
        return _repair_math_escapes(value)
    elif isinstance(value, list):
        return [_repair_math_escapes(v) if isinstance(v, str) else v for v in value]
    return value


def _generate_analysis(paperjson: dict) -> None:
    """Run the 7 analysis calls and populate paperjson["analysis"] in place.

    Calls ``_warmup_ollama()`` once (pin model, keep_alive=-1; Pitfall 5),
    then runs each call in order.  Per-field None is logged as a warning and
    the run continues.  For ``claims``/``limitations``, if the field came back
    None or empty, substitutes a single fallback element so D-14 callouts
    always have >= 1 entry.

    Raises RuntimeError on Ollama unreachable (D-13).
    """
    _warmup_ollama()

    paper_text = _assemble_paper_text(paperjson)
    analysis = paperjson["analysis"]

    for field_name, model_cls, system, prompt_template in _ANALYSIS_SPECS:
        prompt = prompt_template.format(paper_text=paper_text)
        result = _analysis_call(prompt, system, model_cls, paper_text)

        if result is not None:
            value = getattr(result, field_name)
            analysis[field_name] = _repair_analysis_value(value)
        else:
            _log(f"[note warning: {field_name} fill_failed]")
            # Leave as skeleton default (None for scalars, [] for lists)

    # D-14: claims/limitations fallback — ensure at least one entry for callouts
    if not analysis.get("claims"):
        analysis["claims"] = ["(generation failed -- see raw extract)"]
    if not analysis.get("limitations"):
        analysis["limitations"] = ["(generation failed -- see raw extract)"]

    analysis["generated_by"] = OLLAMA_MODEL


# ---------------------------------------------------------------------------
# Analysis-populated predicate (T-08-GAP-18..22) — the single definition
# ---------------------------------------------------------------------------
#
# Analysis-namespace fields that indicate the PaperJSON's analysis has been
# filled by the LLM cascade above (as opposed to the empty skeleton written
# at ingest time — see scripts/ingest.py::_assemble_paperjson's `analysis`
# dict). Ported verbatim from gui/scan.py's former `_ANALYSIS_POPULATED_KEYS`
# (constant name kept so the move is traceable); gui/scan.py now imports
# `_analysis_is_populated` from here instead of re-deriving this list.
#
# `entities` and `connections` are excluded: neither is ever written by any
# code path. Entity extraction and connection inference were deferred to
# Phase 6 by 02-CONTEXT.md D-10; Phase 6 was rescoped to the topic graph and
# never picked them up. Both fields were formally descoped by user decision
# on 2026-07-18 during Phase 8 Round-2 gap closure (D-GAP4, 08-17-PLAN.md) —
# `connections` was already absent from this key tuple before this change
# (and is now also removed from the skeleton itself), `entities` is removed
# from this key tuple by this change, and both removals now read as one
# deliberate act rather than an inherited quirk.
_ANALYSIS_POPULATED_KEYS = (
    "generated_by",
    "summary",
    "claims",
    "methods_overview",
    "results",
    "limitations",
    "open_questions",
    "topics",
)


def _analysis_is_populated(paperjson: dict) -> bool:
    """True when ``paperjson["analysis"]`` has been filled by the LLM cascade.

    An empty analysis skeleton (every value falsy — the shape
    ``scripts/ingest.py::_assemble_paperjson`` writes at ingest time) means
    "not yet filled", NEVER "corrupt". This is the back-compat contract for
    the pre-existing cache files that predate cache persistence: they are
    treated as unfilled and simply fill + persist on their first
    ``generate_note`` run, with no migration required.

    Takes the full PaperJSON dict (not just the ``analysis`` namespace) so
    both call sites — ``generate_note``'s self-skip and gui/scan.py's
    PaperJSON Fill stage light — share one shape and one definition.
    """
    analysis = paperjson.get("analysis") or {}
    return any(analysis.get(key) for key in _ANALYSIS_POPULATED_KEYS)


# ---------------------------------------------------------------------------
# Filename sanitization (T-02-03, D-17)
# ---------------------------------------------------------------------------

def _sanitize_filename(title: str, max_length: int = 200) -> str:
    """Sanitize a paper title for use as a Windows-safe filename.

    Strips all Windows-illegal characters (``/ \\ : * ? " < > |``), collapses
    whitespace, and caps length.  Returns ``Untitled`` when the result is empty.
    """
    sanitized = _WINDOWS_ILLEGAL.sub("", title)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip()
    if not sanitized:
        sanitized = "Untitled"
    # Windows reserved device-name guard (WR-04): the base name before the first
    # dot is what Windows resolves, so "NUL", "Aux", and even "NUL.foo" collide
    # with a device.  Prefix an underscore to disarm the collision.
    if sanitized.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        sanitized = "_" + sanitized
    return sanitized


# ---------------------------------------------------------------------------
# Tag slug helper (GAP-1 / NOTE-02, D-11)
# ---------------------------------------------------------------------------

def _slugify_tag(topic) -> str | None:
    """Convert an LLM-generated topic string to an Obsidian-valid tag slug.

    Obsidian forbids parentheses, commas, spaces, and most punctuation in tags
    (only Unicode word characters, hyphens, and forward slashes are safe).
    This helper:
      1. Lowercases the topic.
      2. Collapses every maximal run of forbidden characters to a single ``-``.
      3. Strips leading/trailing ``-``.
      4. Returns None if the result is empty (topic had no valid characters).

    Forward slashes are preserved as Obsidian nested-tag separators
    (``input/output gating`` -> ``input/output-gating``).

    Args:
        topic: A raw topic string from analysis.topics (may contain any chars).

    Returns:
        A non-empty slug string, or None if no valid characters remain.
    """
    slug = _TAG_INVALID_RUN.sub("-", str(topic).lower()).strip("-")
    return slug if slug else None


# ---------------------------------------------------------------------------
# Tag canonicalization at ingest (260714-dpl)
# ---------------------------------------------------------------------------
#
# Immediately after topics are generated and before rendering, each proposed
# tag slug is canonicalized against the LIVE vault tag vocabulary via a local
# LLM judgment call (gemma4:e4b, think:false). A true synonym is replaced by
# the existing canonical slug (the pre-existing vault tag always wins);
# distinct-but-related topics are left unchanged. The whole step is
# fail-open: any error, empty vocabulary, or parse failure returns the
# proposed slugs unchanged so ingest never breaks.

class _TagMap(BaseModel):
    new: str = Field(..., description="The proposed (new) tag slug")
    canonical: str = Field(
        ...,
        description=(
            "The existing canonical slug this proposed slug should map to when it "
            "is a true synonym/duplicate; otherwise the proposed slug itself."
        ),
    )

class _TagCanon(BaseModel):
    mappings: list[_TagMap] = Field(
        ..., description="One mapping entry per proposed slug"
    )


def _existing_tag_vocabulary(config: dict) -> set[str]:
    """Return the set of existing tag slugs from the LIVE vault.

    Reuses ``scripts.link._build_topic_membership`` (Papers/*.md live
    frontmatter) as the vocabulary source -- NOT the registry/
    ``.paperjson_cache`` (D-02 source-of-truth). Imported lazily to avoid an
    import cycle / keep the dependency optional. Fail-open: returns an empty
    set on ANY exception (missing vault_path, unreadable vault, etc.).
    """
    try:
        vault_path = config.get("vault_path")
        if not vault_path:
            return set()
        import pathlib as _pathlib
        from scripts.link import _build_topic_membership
        membership = _build_topic_membership(_pathlib.Path(vault_path))
        return set(membership.keys())
    except Exception:
        return set()


def _tag_canonicalization_call(proposed: list[str], existing: set[str]) -> dict[str, str] | None:
    """Ask the local LLM to map each proposed slug to an existing canonical slug.

    Mirrors ``_analysis_call``'s Ollama usage (``_ollama_extraction_call`` +
    ``_parse_extraction_response``, think:false) but MUST fail open: any
    Ollama-unreachable/timeout/parse-failure returns None instead of raising
    (unlike ``_analysis_call``, which raises RuntimeError on unreachable --
    D-13 -- for the main analysis pipeline).

    Returns:
        ``{proposed_slug: canonical_slug}`` mapping, or None on any failure.
    """
    system = (
        "You are a tag deduplication assistant for a research paper vault. "
        "You are given the EXISTING tag vocabulary already used in the vault and "
        "a list of NEWLY PROPOSED tag slugs for a paper just ingested. "
        "For each proposed slug, decide whether it is a TRUE SYNONYM or DUPLICATE "
        "of an existing tag (same concept, just phrased/abbreviated differently). "
        "If it is a true synonym/duplicate, map it to that EXISTING slug exactly "
        "as given. If it is a genuinely distinct topic -- even if related to an "
        "existing tag -- map it to itself unchanged. Do NOT merge distinct-but-"
        "related topics (e.g. a general topic and a more specific sub-topic are "
        "DISTINCT, not synonyms). Only ever output existing vocabulary slugs or "
        "the proposed slug itself -- never invent a new slug. "
        "Return ONLY valid JSON matching the schema."
    )
    prompt = (
        f"Existing tag vocabulary:\n{sorted(existing)}\n\n"
        f"Newly proposed tag slugs:\n{proposed}\n\n"
        "For each proposed slug, return a mapping entry with 'new' set to the "
        "proposed slug and 'canonical' set to the existing slug it duplicates, "
        "or to itself if it is distinct."
    )
    schema = _TagCanon.model_json_schema()
    num_ctx = _estimate_num_ctx(prompt, overhead=2048, output_ratio=0.5)

    try:
        raw = _ollama_extraction_call(
            prompt, system, schema,
            num_ctx=num_ctx,
            timeout=_SECTION_TIMEOUT,
        )
    except Exception:
        return None

    if raw.startswith("[Ollama error:") or raw.startswith("[Ollama timeout:"):
        return None

    result = _parse_extraction_response(raw, _TagCanon)
    if result is None:
        return None

    return {m.new: m.canonical for m in result.mappings}


def _canonicalize_tags(topics: list[str], config: dict) -> list[str]:
    """Canonicalize proposed topic slugs against the live vault tag vocabulary.

    Slugifies ``topics`` (dropping None), then -- if there is an existing
    vocabulary to compare against -- asks the local LLM which proposed slugs
    are true synonyms of existing slugs. Every applied replacement is logged
    (``old -> canonical``). A replacement is applied ONLY when the target is
    actually present in the existing vocabulary (safety guard against a
    hallucinated/invented target). Result is deduped, order-preserving.

    Fails open at every stage: empty proposed list, empty vocabulary, a
    failed/None LLM call, or ANY unexpected exception all return the
    slugified proposed topics unchanged. Never raises.
    """
    proposed = [s for s in (_slugify_tag(t) for t in (topics or [])) if s is not None]
    try:
        if not proposed:
            return proposed

        existing = _existing_tag_vocabulary(config)
        if not existing:
            return proposed

        mapping = _tag_canonicalization_call(proposed, existing)
        if mapping is None:
            return proposed

        result: list[str] = []
        for slug in proposed:
            target = mapping.get(slug, slug)
            if target != slug and target in existing:
                _log(f"[note tag-canonicalized: {slug} -> {target}]")
            else:
                target = slug
            if target not in result:
                result.append(target)
        return result
    except Exception:
        return proposed


# ---------------------------------------------------------------------------
# YAML frontmatter rendering (D-17, D-18, SC2, Pitfall 6)
# ---------------------------------------------------------------------------

def _render_frontmatter(meta: dict, analysis: dict) -> str:
    """Render YAML frontmatter between ``---`` fences.

    Every string value is double-quoted with internal ``"`` escaped, so
    ``yaml.safe_load()`` survives colons and special characters (SC2, Pitfall 6).
    Nullable doi/arxiv_id are omitted when None (Discretion).
    """
    title = meta.get("title", "Untitled").replace('"', '\\"')
    lines = ["---"]
    lines.append(f'title: "{title}"')

    # Authors as YAML list, each double-quoted (Pitfall 6)
    authors = meta.get("authors") or []
    if authors:
        lines.append("authors:")
        for a in authors:
            escaped = str(a).replace('"', '\\"')
            lines.append(f'  - "{escaped}"')

    # Simple scalar fields
    year = meta.get("year")
    if year is not None:
        lines.append(f"year: {year}")

    journal = meta.get("journal")
    if journal:
        escaped = str(journal).replace('"', '\\"')
        lines.append(f'journal: "{escaped}"')

    # Nullable fields — omit when None (Discretion)
    doi = meta.get("doi")
    if doi:
        lines.append(f'doi: "{doi}"')

    arxiv_id = meta.get("arxiv_id")
    if arxiv_id:
        lines.append(f'arxiv_id: "{arxiv_id}"')

    # Tags from analysis.topics (D-11, GAP-1) — Obsidian-valid slugs only.
    # _slugify_tag restricts to [\w/-]+ (Unicode word chars + slash + hyphen),
    # collapsing forbidden runs (parentheses, commas, spaces, …) to a single
    # hyphen and returning None for topics that reduce to empty.  Only emit
    # the tags: block when at least one slug survives (NOTE-02, CR-03).
    topics = analysis.get("topics") or []
    slugs = [s for s in (_slugify_tag(t) for t in topics) if s is not None]
    if slugs:
        lines.append("tags:")
        for slug in slugs:
            # Double-quote each slug (CR-03 / NOTE-02 yaml.safe_load round-trip).
            # The charset restriction means quotes/spaces can no longer appear in
            # a slug, but double-quoting is retained so the YAML invariant holds
            # even if a future slug source changes.
            lines.append(f'  - "{slug}"')

    lines.append("status: ingested")
    lines.append(f"date_ingested: {datetime.date.today().isoformat()}")
    lines.append("---")
    lines.append("")  # blank line after frontmatter
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic note rendering (Pattern 3, D-08, D-14, D-15)
# ---------------------------------------------------------------------------

def _render_note(paperjson: dict) -> str:
    """Render a complete Obsidian note from PaperJSON v2.

    Deterministic assembly from structured analysis + extraction data.
    No LLM call here — callouts are derived from claims[0] / limitations[0].
    """
    meta = paperjson["extraction"]["metadata"]
    analysis = paperjson["analysis"]

    # YAML frontmatter
    frontmatter = _render_frontmatter(meta, analysis)

    parts = []
    parts.append(f"# {meta.get('title', 'Untitled')}")
    parts.append("")

    # Summary section
    summary = analysis.get("summary") or "(generation failed -- raw extract below)"
    parts.append("## Summary")
    parts.append("")
    parts.append(summary)
    parts.append("")

    # [!important] callout after Summary (D-15)
    claims = analysis.get("claims") or []
    if claims:
        # Prefix every line so a multi-line LLM value stays inside the callout
        # block; otherwise lines after the first fall outside the ``>`` and break
        # the Obsidian callout (WR-02).
        first_claim = str(claims[0]).replace("\n", "\n> ")
        parts.append(f"> [!important] Key Contribution")
        parts.append(f"> {first_claim}")
        parts.append("")

    # Key Findings section — bullet list of all claims
    parts.append("## Key Findings")
    parts.append("")
    for claim in claims:
        parts.append(f"- {claim}")
    parts.append("")

    # Methodology
    methods = analysis.get("methods_overview") or "(generation failed)"
    parts.append("## Methodology")
    parts.append("")
    parts.append(methods)
    parts.append("")

    # Results
    results = analysis.get("results") or "(generation failed)"
    parts.append("## Results")
    parts.append("")
    parts.append(results)
    parts.append("")

    # Limitations section with [!warning] callout (D-15)
    parts.append("## Limitations")
    parts.append("")
    limitations = analysis.get("limitations") or []
    if limitations:
        # Prefix every line so a multi-line value stays inside the callout (WR-02).
        first_lim = str(limitations[0]).replace("\n", "\n> ")
        parts.append(f"> [!warning] Limitation")
        parts.append(f"> {first_lim}")
        parts.append("")
    for lim in limitations:
        parts.append(f"- {lim}")
    parts.append("")

    # My Notes section (user-owned, protected by skip-by-default)
    parts.append("## My Notes")
    parts.append("")

    return frontmatter + "\n".join(parts)


# ---------------------------------------------------------------------------
# Orchestrator: analysis -> render -> vault write
# ---------------------------------------------------------------------------

def generate_note(
    paperjson: dict,
    config: dict,
    force: bool = False,
    paperjson_path: str | None = None,
    force_analysis: bool = False,
) -> str:
    """Generate a note from PaperJSON and write it to the vault.

    Orchestrates: preflight -> analysis (self-skip if cached) -> canonicalize
    tags -> persist analysis to cache -> render -> vault write.

    Returns the vault-relative path on success, or a ``[note error: ...]``
    string on failure.  On an existing note with ``force=False``, skips
    (D-16) and returns the path without running any LLM calls.

    ``force`` vs. ``force_analysis`` — kept as two separate flags rather than
    overloading one, on purpose: ``force`` (D-16) means "overwrite an
    existing note on disk" and is what the GUI's Re-run Note button passes.
    Overloading it to also mean "re-run analysis" would make every note
    re-run pay the 7 analysis LLM calls this plan exists to eliminate —
    reintroducing the exact cost the gap identifies. ``force_analysis`` is
    the distinct, deliberate re-analysis escape hatch (e.g. after a model
    upgrade), keeping D-16's note-overwrite contract untouched.

    ``paperjson_path``, when given, is the cache file this PaperJSON was (or
    will be) persisted to. After analysis + tag canonicalization, the
    enriched PaperJSON is written back to that file via
    ``scripts.ingest._write_paperjson_cache`` (same filelock + temp-file +
    ``os.replace`` atomicity as the rest of the pipeline), so a second run
    against the same cache makes zero analysis LLM calls. When
    ``paperjson_path`` is None (the default), no cache file is read or
    written by this function — behaviour is unchanged from before this
    persistence was added.
    """
    # Fail-fast: vault_path required
    vault_path = config.get("vault_path")
    if not vault_path:
        return "[note error: vault_path not set in config.json]"

    # Preflight: vault root exists as a directory (filesystem check, no Obsidian dependency)
    if not preflight(config):
        return "[note error: vault root does not exist -- check vault_path in config.json]"

    # Determine filename and path (uses extraction metadata only — no LLM)
    meta = paperjson["extraction"]["metadata"]
    title = meta.get("title", "Untitled")
    filename = _sanitize_filename(title)
    path = f"Papers/{filename}.md"

    # Security: validate through the single chokepoint guard (WR-03) — fail fast
    # before any LLM analysis runs.  create_note re-validates at write time, but
    # checking here avoids 7 wasted analysis calls on a rejected path.  The shared
    # guard checks path *components* (not a raw substring), so legitimate titles
    # containing ".." (e.g. "Deep Learning... A Survey") are not falsely rejected.
    try:
        _validate_vault_path(path, config)
    except ValueError as e:
        return f"[note error: {e}]"

    # Skip-by-default (D-16): existing note + no force -> skip (zero LLM calls)
    if note_exists(path, config) and not force:
        _log(f"note already exists at {path} -- skipping (use --force to overwrite)")
        return path

    # Honour the per-clone analysis timeout (config.json "ollama_section_timeout",
    # default 300s) — slow hardware can raise it (WR-01; mirrors ingest.py).
    global _SECTION_TIMEOUT
    _SECTION_TIMEOUT = int(config.get("ollama_section_timeout", DEFAULT_SECTION_TIMEOUT))

    # Run analysis generation (7 LLM calls) — self-skip when the cached
    # analysis is already populated, unless a deliberate re-analysis was
    # requested (T-08-GAP-18/21: the cost win this plan exists to deliver).
    if _analysis_is_populated(paperjson) and not force_analysis:
        _log(
            "cached analysis already populated -- reusing (zero analysis "
            "calls; pass --force-analysis / force_analysis=True to re-run)"
        )
    else:
        _generate_analysis(paperjson)

    # Tag canonicalization (260714-dpl): merge true-synonym proposed tag slugs
    # into the existing live-vault canonical slug before rendering (fail-open).
    # Runs on BOTH paths (fresh analysis and reused cached analysis) so the
    # rendered note's tags are canonical either way.
    analysis = paperjson["analysis"]
    analysis["topics"] = _canonicalize_tags(analysis.get("topics") or [], config)

    # Persist the filled, canonicalized analysis back to the cache
    # (T-08-GAP-18). Sits AFTER canonicalization so the cache and the
    # rendered note never disagree on tags (T-08-GAP-20), and BEFORE the
    # render/vault-write so the 7 LLM calls are banked even if the vault
    # write later fails. A cache-write failure must never cost the user
    # their note (T-08-GAP-19) — non-fatal, logged as a warning.
    if paperjson_path:
        try:
            _cache_file = pathlib.Path(paperjson_path)
            _write_paperjson_cache(
                paperjson, _cache_file.stem, cache_dir=str(_cache_file.parent)
            )
        except Exception as e:
            _log(
                f"[note warning: failed to persist analysis to cache "
                f"{paperjson_path}: {e}]"
            )

    # Render the note
    rendered = _render_note(paperjson)

    # Write to vault via obsidian_cli chokepoint (direct atomic write)
    create_note(path=path, content=rendered, config=config, overwrite=force)
    _log(f"wrote note to {path}")
    return path


# ---------------------------------------------------------------------------
# Standalone cache resolution (D-07)
# ---------------------------------------------------------------------------

def _resolve_paperjson_path(stem: str, cache_dir: str = ".paperjson_cache") -> str:
    """Resolve the default PaperJSON cache file path from a PDF stem.

    Args:
        stem:       PDF filename stem (e.g. ``my_paper``).
        cache_dir:  Directory containing cached PaperJSON files.

    Returns:
        Absolute path to ``<cache_dir>/<stem>.json``.
    """
    import pathlib as _pathlib
    return str((_pathlib.Path(cache_dir) / f"{stem}.json").resolve())


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Windows cp1252 guard: wrap stdout in UTF-8 before any print (D-PATTERNS)
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Generate an Obsidian note from a PaperJSON v2 cache file."
    )
    parser.add_argument(
        "--paperjson", default=None,
        help="Path to the PaperJSON v2 cache file (JSON). When omitted, resolves .paperjson_cache/<stem>.json.",
    )
    parser.add_argument(
        "--stem", default=None,
        help="PDF filename stem (without extension). Used to resolve cache when --paperjson is omitted.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing note instead of skipping (D-16).",
    )
    parser.add_argument(
        "--force-analysis", action="store_true",
        help="Re-run the 7 analysis calls even when the cached analysis is populated.",
    )
    args = parser.parse_args()

    try:
        config = _load_config()

        # Resolve PaperJSON path: explicit --paperjson, or default from --stem
        if args.paperjson:
            pj_path = args.paperjson
        elif args.stem:
            pj_path = _resolve_paperjson_path(
                args.stem, config.get("paperjson_cache_dir", ".paperjson_cache")
            )
        else:
            parser.error("either --paperjson or --stem is required")

        with open(pj_path, encoding="utf-8") as f:
            paperjson = json.load(f)
        result = generate_note(
            paperjson, config, force=args.force,
            paperjson_path=pj_path, force_analysis=args.force_analysis,
        )
        print(result)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[note error: {e}]", file=sys.stderr)
        sys.exit(1)
