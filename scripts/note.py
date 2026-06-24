"""
Note generation pipeline: analysis fill (7 LLM calls) + deterministic render + vault write.

Reads a PaperJSON v2 dict (in-memory or from cache file), runs 7 gemma4:e4b
analysis calls (summary, claims, methods_overview, results, limitations, topics,
open_questions), renders a deterministic Obsidian note with YAML frontmatter and
callouts, and writes it to Papers/<sanitized title>.md via the obsidian_cli
chokepoint.

Run standalone:  python scripts/note.py --paperjson <cache>.json [--force]
"""

import datetime
import json
import re
import sys

from pydantic import BaseModel, Field

from scripts.ingest import (
    _ollama_extraction_call,
    _parse_extraction_response,
    _estimate_num_ctx,
    _warmup_ollama,
    _load_config,
    OLLAMA_MODEL,
)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_SECTION_TIMEOUT = 300  # seconds per analysis call (configurable via config.json)

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
            analysis[field_name] = value
        else:
            _log(f"[note warning: {field_name} fill_failed]")
            # Leave as skeleton default (None for scalars, [] for lists)

    # D-14: claims/limitations fallback — ensure at least one entry for callouts
    if not analysis.get("claims"):
        analysis["claims"] = ["(generation failed -- see raw extract)"]
    if not analysis.get("limitations"):
        analysis["limitations"] = ["(generation failed -- see raw extract)"]

    analysis["generated_by"] = OLLAMA_MODEL
