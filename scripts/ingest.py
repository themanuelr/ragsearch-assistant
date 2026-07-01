"""
MinerU-based PDF ingestion pipeline.

Invokes MinerU (hybrid_auto backend), parses the resulting content_list.json by block
type, and assembles a PaperJSON v2 document (extraction/analysis/provenance namespaces,
schema_version 2).

By default a short confirmation (cache path + written note path) is printed to stdout.
Pass --print / --stdout to emit the full PaperJSON as UTF-8 JSON to stdout instead.
Use --output / -o to write the PaperJSON to a file (bypasses PowerShell mojibake on Windows).

This module is extraction-only: no Ollama or LLM calls.
The analysis namespace ships as an empty skeleton; Phase 2 populates it.
"""

import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import filelock
import urllib.request
import urllib.error
import urllib.parse
from pydantic import BaseModel, Field, ValidationError, field_validator

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2
DEFAULT_TIMEOUT = 1800          # 30 minutes — covers long papers + first model download
MINERU_BACKEND = "hybrid_auto"
NOISE_BLOCK_TYPES = {"footer", "page_number", "aside_text", "header"}
OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:e4b"
DOI_RE = re.compile(r"10\.\d{4,}/\S+")
_ARXIV_URL_RE = re.compile(
    r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf|html)/(.+?)(?:\?.*)?$"
)

# ---------------------------------------------------------------------------
# num_ctx ladder + cap (Phase 1.3 Plan 05 — RTX 4060 benchmark 2026-06-12)
#
# gemma4:e4b benchmarks on RTX 4060 (8GB VRAM):
#   num_ctx 16384/32768/65536 → 100% GPU, ~3.1-3.2GB footprint, ~55 tok/s
#   num_ctx 131072             → 9.8GB → 70% CPU offload → 32.9 tok/s (worse with full contexts)
#
# DEFAULT_NUM_CTX_CAP = 65536 is the benchmark-validated sweet spot: full GPU,
# acceptable throughput, headroom for long sections. 131072 would CPU-offload
# on the RTX 4060 and is NOT the default. Per-clone configurable via config.json
# key "ollama_num_ctx_cap" for users on different hardware.
# ---------------------------------------------------------------------------
NUM_CTX_LADDER = (2048, 4096, 8192, 16384, 32768, 65536)
DEFAULT_NUM_CTX_CAP = 65536
_NUM_CTX_CAP: int = DEFAULT_NUM_CTX_CAP  # mutable module global; set by ingest() from config

# ---------------------------------------------------------------------------
# Phase 1.3 Plan 06 — GAP E: configurable section-fill timeout
#
# gemma4:e4b benchmarks on RTX 4060 (8GB VRAM):
#   ~55 tok/s at 16K-65K num_ctx (100% GPU)
# A long RESULTS section repaired at ~55 tok/s can easily exceed 180s of generation
# (observed live-UAT: test_manuel1 RESULTS → fill_failed under gap-05's 180s budget).
# 300s gives substantial headroom while still bounding a genuine hang.
# Per-clone configurable via config.json key "ollama_section_timeout".
# Also applied to the full-document DOI probe (GAP C) — large input on same tier.
# ---------------------------------------------------------------------------
DEFAULT_SECTION_TIMEOUT = 300
_SECTION_TIMEOUT: int = DEFAULT_SECTION_TIMEOUT  # mutable module global; set by ingest() from config


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 06 — GAP A: timestamped progress helper
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Emit a timestamped progress line to stderr.

    Format: [ingest YYYY-MM-DD HH:MM:SS] <msg>
    stdout is used for the short confirmation by default; pass --print/--stdout to
    emit the full PaperJSON there, or use --output/-o to write it to a file.
    """
    print(f"[ingest {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 06 — GAP B: UTF-8 file-write helper
# ---------------------------------------------------------------------------

def _emit_result(
    result: dict,
    output_path: "str | None",
    print_json: bool = False,
    confirmation: "str | None" = None,
) -> None:
    """Emit the final PaperJSON result to a file or stdout.

    Branch logic:
    - output_path truthy: write UTF-8 JSON directly to the file (bypasses PowerShell
      '>' redirection which re-decodes correct UTF-8 through the OEM console code page
      and writes UTF-16LE with mojibake — observed on Windows: José→"Jos├⌐", U+2019→"ΓÇÖ").
    - print_json True: print the full PaperJSON to stdout (opt-in, --print/--stdout).
    - default: print the short confirmation string (cache path + note path). If confirmation
      is None, a minimal fallback line is printed instead.

    output_path takes precedence over print_json — if both are given the file is written.
    """
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        _log(f"wrote {output_path}")
    elif print_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(confirmation if confirmation is not None else "[ingest] done.")


# ---------------------------------------------------------------------------
# Pydantic output models (Phase 1.3 LLM fill layer)
# ---------------------------------------------------------------------------

class DoiProbeResult(BaseModel):
    doi: str | None = None
    arxiv_id: str | None = None
    title: str | None = None

    @field_validator("doi")
    @classmethod
    def validate_doi_syntax(cls, v: str | None) -> str | None:
        """Reject syntactically invalid DOIs at the model level (D-00c)."""
        if v is not None and not DOI_RE.fullmatch(v.strip()):
            return None
        return v.strip() if v else v


class PaperMetadata(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None          # journal abbreviation, e.g. "J. Am. Chem. Soc."
    journal_full: str | None = None     # full journal title verbatim from document text; null if only abbreviation
                                        # additive optional field (Plan 06 GAP D); no SCHEMA_VERSION bump
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None


class SectionFillResult(BaseModel):
    heading: str
    body: str
    fill_failed: bool = False
    keep: bool = True  # LLM substantive-content verdict; default True so any parse/fill failure or
    # missing verdict CONSERVATIVELY RETAINS the section — never silently drop on uncertainty (Plan 08)


class RefEntry(BaseModel):
    number: int | None = None
    raw: str
    doi: str | None = None
    title: str | None = None
    year: int | None = None
    fill_failed: bool = False
    is_reference: bool = True  # LLM's "is this a genuine bibliographic reference?" verdict; default True
    # so any parse/fill failure or missing verdict CONSERVATIVELY RETAINS the entry — never silently
    # drop a reference on uncertainty (mirrors Plan 08's SectionFillResult.keep default)


class RefBatchResult(BaseModel):
    refs: list[RefEntry]


# ---------------------------------------------------------------------------
# Ollama LLM client + response helpers (Phase 1.3)
# ---------------------------------------------------------------------------

def _ollama_extraction_call(
    prompt: str,
    system: str,
    schema: dict,
    num_ctx: int = 4096,
    timeout: int = 120,
) -> str:
    """
    POST to /api/chat with format=<json_schema>. Returns raw content string.

    Mirrors mcp-ollama/server.py _ollama_chat but adds structured-output params:
    format, options.temperature=0, options.num_ctx, and sets think=false to suppress
    the reasoning preamble on Ollama 0.30.6 (verified empirically 2026-06-23: think=false
    + format coexist correctly, ~2.6x faster structured calls). Ollama #15260 was the
    historical catch-22 (think=false silently dropped format) and no longer reproduces
    on 0.30.6.
    """
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
        "format": schema,
        "think": False,
        "options": {
            "temperature": 0,
            "num_ctx": num_ctx,
        },
        # Ollama 0.30.6: think=false + format coexist correctly (verified 2026-06-23).
        # Historical #15260 catch-22 (think=false dropped format) no longer reproduces.
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except TimeoutError:
        # Python 3.10+ raises bare TimeoutError (alias of socket.timeout) when the
        # read deadline fires during resp.read() while the model is still generating.
        # Without this branch it escapes to main()'s catch-all and aborts the whole
        # run with empty stdout and no registry write (observed live-UAT defect, Plan 05).
        # The prefix MUST be distinct from [Ollama error:] — every two-strike consumer
        # raises RuntimeError on that exact prefix (D-00d unreachable abort), so reusing
        # it would still abort; [Ollama timeout:] is what lets two-strike fill_failed
        # degradation absorb timeouts (D-05/D-06).
        return f"[Ollama timeout: no response within {timeout}s]"
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            # connect-phase timeouts wrapped by urllib are also degradable
            return f"[Ollama timeout: no response within {timeout}s]"
        return f"[Ollama error: {e}]"
    try:
        return data["message"]["content"]
    except (KeyError, TypeError):
        return f"[Ollama error: unexpected response envelope: {data}]"


def _parse_extraction_response(raw: str, model_cls: type[BaseModel]) -> BaseModel | None:
    """
    Strip-then-parse-then-pydantic fallback for gemma4:e4b structured output.

    Ollama bug #15416: even with format=<schema>, gemma4:e4b in thinking mode
    wraps valid JSON in markdown code fences (```json ... ```).
    Strip the fence, then validate. Returns None on all failures.
    """
    stripped = raw.strip()
    candidates = [
        stripped,
        re.sub(r"^```[a-z]*\n?", "", stripped).rstrip("`").strip(),
    ]
    for candidate in candidates:
        try:
            return model_cls.model_validate_json(candidate)
        except (ValidationError, ValueError):
            continue
    return None


def _estimate_num_ctx(text: str, overhead: int = 2048, cap: int | None = None,
                      output_ratio: float = 0.0) -> int:
    """
    Approximate num_ctx from input length (~4 chars/token).

    Rounds up to the nearest rung in NUM_CTX_LADDER that fits the estimated token count.
    The cap is configurable per clone via config.json "ollama_num_ctx_cap" (default 65536,
    benchmark-validated for 100% GPU on RTX 4060 at ~55 tok/s with ~3.2GB footprint).

    Args:
        text:         Input text whose token count is estimated.
        overhead:     Token overhead added to the raw estimate (default 2048).
        cap:          Override the cap explicitly. When None, reads the module-level
                      _NUM_CTX_CAP (set once by ingest() from config). Pass a value
                      to override — e.g. for tests or probe calls with known small inputs.
        output_ratio: Fraction of the input size to reserve for ECHO output (the model
                      regenerates its input as output). 0.0 (default) = size for input
                      only (probe, metadata — backward compatible). 1.0 = size for input
                      + an equal-size output (section/ref-batch echo calls, which regenerate
                      the full body/refs as JSON). The result is still clamped to the
                      configurable cap, so a 2x budget exceeding the cap falls back to cap.

    Returns:
        The smallest NUM_CTX_LADDER rung >= ceil(estimated_tokens * (1 + output_ratio) + overhead)
        that does not exceed the active cap; or the cap itself when no rung fits.
    """
    active_cap = cap if cap is not None else _NUM_CTX_CAP
    estimated_tokens = len(text) // 4
    raw = math.ceil(estimated_tokens * (1 + output_ratio)) + overhead
    for ctx in NUM_CTX_LADDER:
        if ctx > active_cap:
            break
        if raw <= ctx:
            return ctx
    return active_cap


def _warmup_ollama(model: str = OLLAMA_MODEL) -> None:
    """
    Pin model in VRAM for pipeline run. Best-effort — preflight catches real outages.

    Sets keep_alive="-1" so the model stays loaded for the full ingest run.
    """
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "keep_alive": "-1",
        "options": {"num_ctx": 512},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
    except (urllib.error.URLError, TimeoutError):
        pass  # warm-up is best-effort


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Read config.json from the repo root (parent of scripts/) with UTF-8 encoding."""
    cfg_path = pathlib.Path(__file__).parent.parent / "config.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    # Expand ~ on path fields
    for key in ("registry_path", "vault_path", "mineru_path"):
        if key in cfg and cfg[key]:
            cfg[key] = str(pathlib.Path(cfg[key]).expanduser())
    return cfg


def _resolve_mineru(config: dict) -> str | None:
    """
    Resolve the MinerU executable path.

    Returns config['mineru_path'] if truthy and the file exists, else falls back
    to shutil.which('mineru'). Returns None if neither resolves.
    """
    configured = config.get("mineru_path")
    if configured and pathlib.Path(configured).exists():
        return configured
    return shutil.which("mineru")


# ---------------------------------------------------------------------------
# Phase 3 — web ingestion helpers: URL rewrite / PubMed resolution
# ---------------------------------------------------------------------------

def _resolve_defuddle(config: dict) -> str | None:
    """
    Resolve the defuddle executable path.

    Returns config['defuddle_path'] if truthy and the file exists, else falls back
    to shutil.which('defuddle'). Returns None if neither resolves.
    """
    configured = config.get("defuddle_path")
    if configured and pathlib.Path(configured).exists():
        return configured
    return shutil.which("defuddle")


def _resolve_pubmed_to_pmc(pmid: str, config: dict) -> str | None:
    """
    Query the NCBI E-utilities elink endpoint to resolve a PubMed ID to a PMC URL.

    Sends only the integer PMID to the fixed NCBI host. Fails open on any
    network or parse error (D-05 / T-03-05).
    """
    try:
        endpoint = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
            f"?dbfrom=pubmed&db=pmc&id={pmid}&retmode=json"
        )
        with urllib.request.urlopen(endpoint, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pmc_id = data["linksets"][0]["linksetdbs"][0]["links"][0]
        return f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmc_id}/"
    except Exception:
        return None


def _rewrite_url(url: str, config: dict) -> str:
    """
    Rewrite a user-supplied URL to the preferred full-text form (D-04/D-05).

    - arXiv /abs/<id> or /pdf/<id> → /html/<id>  (full-text HTML, defuddle-friendly)
    - PubMed URL → look up PMC via NCBI elink; return PMC URL or original on failure
    - All other URLs → returned unchanged (journal URLs defuddled as-given)

    Pure string matching; no try/except (rewriting cannot fail; network calls are
    encapsulated in _resolve_pubmed_to_pmc which fails open).
    """
    m = _ARXIV_URL_RE.match(url)
    if m:
        arxiv_id = m.group(1)
        return f"https://arxiv.org/html/{arxiv_id}"
    pubmed_m = re.match(r"https?://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
    if pubmed_m:
        pmid = pubmed_m.group(1)
        pmc_url = _resolve_pubmed_to_pmc(pmid, config)
        if pmc_url:
            return pmc_url
        return url
    return url


def _is_arxiv_url(url: str) -> bool:
    """Return True for arxiv.org or ar5iv.labs.arxiv.org URLs (D-04 / 03-03)."""
    return bool(_ARXIV_URL_RE.match(url)) or "ar5iv.labs.arxiv.org" in url


def _extract_arxiv_id(url: str) -> str | None:
    """
    Extract the arXiv paper id from an arxiv.org or ar5iv URL.

    Returns the id string (e.g. "1706.03762" or "1706.03762v7"), or None when
    the URL does not match either pattern (D-04 / 03-03).
    """
    m = _ARXIV_URL_RE.match(url)
    if m:
        return m.group(1)
    m2 = re.match(r"https?://ar5iv\.labs\.arxiv\.org/html/(.+?)(?:\?.*)?$", url)
    if m2:
        return m2.group(1)
    return None


def _web_body_too_thin(skeleton: dict, min_chars: int) -> bool:
    """
    Return True when the total plain text in the skeleton is below min_chars (D-09).

    Counts only type=="text" blocks' plain field — web pages produced by defuddle
    have no figures/tables for MVP, so only text content is relevant.
    Mirrors the block-iteration shape of _quality_gate (lines 1119-1128) but
    counts text-only plain for the web paywall check.
    """
    total = 0
    for section in skeleton.get("extraction", {}).get("sections", []):
        for block in section.get("blocks", []):
            if block.get("type") == "text":
                total += len(block.get("plain", "") or "")
    return total < min_chars


def _normalize_text(text: str) -> str:
    """
    Apply P0 normalization fix (MINERU.md §3, D-09).

    P0 (mandatory): fi/fl/ff/ffi/ffl ligature superscript misread.
    U+FFFD and charge-sign replacements removed (D-09); LLM fill layer handles these.
    """
    # P0 — ligature superscript fix (95+ occurrences per paper)
    text = re.sub(r"<sup>\s*(fi|fl|ff|ffi|ffl)\s*</sup>", r"\1", text)
    return text


def _build_display(text: str) -> str:
    """Return the display rendition of a text block (markdown with LaTeX, sup/sub)."""
    return _normalize_text(text)


def _despace_math(m: "re.Match") -> str:
    """
    Collapse intra-math whitespace and map trivially-mappable LaTeX symbols.

    Used as a re.sub callback for inline $...$ spans (D-19).
    """
    inner = re.sub(r"\s+", "", m.group(1))
    inner = inner.replace(r"\pm", "±").replace(r"\times", "×")
    return inner


def _build_plain(text: str) -> str:
    """
    Return the plain rendition of a text block (normalized for embedding).

    Flattens <sup>/<sub> tags to their content. Strips inline LaTeX delimiters
    and collapses intra-math whitespace. D-17/D-18/D-19.
    """
    normalized = _normalize_text(text)
    # D-18: flatten <sup>/<sub> tags
    plain = re.sub(r"<su[pb]>(.*?)</su[pb]>", r"\1", normalized)
    # D-19: strip inline $...$ LaTeX delimiters and collapse spaces
    plain = re.sub(r"\$([^$]*)\$", _despace_math, plain)
    return plain


def _quarantine_figure(block: dict) -> dict:
    """
    Build a quarantined figure block from a MinerU image block (D-03).

    The image.content field (VLM-generated, hallucination risk per MINERU.md §2)
    goes ONLY into figure_vlm_description. Only image_caption and img_path are
    treated as trusted figure metadata.

    Returns a figure dict: {type, img_path, caption, figure_vlm_description}.
    """
    caption = block.get("image_caption", [])
    if isinstance(caption, list):
        caption = " ".join(caption)
    caption_normalized = _normalize_text(caption)
    return {
        "type": "figure",
        "img_path": block.get("img_path", ""),
        "caption": caption_normalized,
        "figure_vlm_description": block.get("content", ""),  # quarantined — never embed
    }


# ---------------------------------------------------------------------------
# Phase 1.3 fill helpers
# ---------------------------------------------------------------------------

def _extract_first_page_and_footers(content_list: list) -> str:
    """
    Extract first-page text blocks + all footer blocks as a single text string.

    This is the DOI probe input (D-04) — deterministic, no LLM.
    Footer blocks are included because DOIs frequently appear in journal footers
    (confirmed from MINERU.md empirical findings for JACS and PNAS papers).

    Args:
        content_list: List of MinerU content_list.json block dicts.

    Returns:
        Newline-joined string of qualifying block texts.
    """
    lines = []
    for block in content_list:
        btype = block.get("type", "")
        page_idx = block.get("page_idx", 99)
        text = block.get("text", "")
        if not text.strip():
            continue
        # Include: first-page text blocks + all footer blocks (any page)
        if (btype == "text" and page_idx == 0) or btype == "footer":
            lines.append(text.strip())
    return "\n".join(lines)


def _extract_full_text(content_list: list) -> str:
    """
    Extract ALL text blocks (any page) + all footer blocks as a single text string.

    This is the full-document DOI probe input (Plan 06 GAP C). Unlike
    _extract_first_page_and_footers, this function does NOT restrict to page_idx == 0.
    The DOI may appear in Supporting Information on a later page (confirmed on test_manuel1:
    10.1021/jacs.3c10258 appears in SI, not the cover page).

    LLM-only probe input — no regex content mining introduced (LOCKED no-regex-prefill
    architecture preserved; GAP C only widens the LLM probe's input).

    Args:
        content_list: List of MinerU content_list.json block dicts.

    Returns:
        Newline-joined string of all text + footer block texts (any page, non-empty).
    """
    lines = []
    for block in content_list:
        btype = block.get("type", "")
        text = block.get("text", "")
        if not text.strip():
            continue
        # Include: ALL text blocks (any page) + all footer blocks (any page)
        if btype in ("text", "footer"):
            lines.append(text.strip())
    return "\n".join(lines)


def _syntactic_doi_valid(doi: str | None) -> bool:
    """
    Deterministic DOI syntax check (D-00c).

    Returns True if doi is a non-empty string matching 10.<digits>/<suffix>.
    NOTE: suffix truncation can still pass this (Pitfall 3 — a truncated DOI
    like '10.1073/pnas' is syntactically valid). The probe SYSTEM prompt carries
    the suffix-preservation burden; this function only rejects obviously malformed
    strings (None, empty, no DOI prefix).
    """
    return bool(doi and DOI_RE.fullmatch(doi.strip()))


def _web_doi_key_fallback(
    url: str,
    probe: "DoiProbeResult | None",
) -> "tuple[str | None, str | None]":
    """
    Extract a DOI or arXiv ID directly from the URL as a last-resort key (D-07).

    Only invoked when _doi_probe returned neither a DOI nor an arXiv ID.  The
    probe argument is accepted for signature symmetry but is not consulted —
    the function reads only the URL.

    Match order:
      1. New-style arXiv ID:  arxiv.org/{abs|html|pdf}/<NNNN.NNNNN>[vN]
         (a trailing vN version suffix is matched but stripped from the
         returned id so the key converges with the PDF probe — WR-01 / SC3)
      2. Old-style arXiv ID:  arxiv.org/{abs|html|pdf}/<archive[.SS]/NNNNNNN>
      3. doi.org/<DOI> path:  extracted DOI accepted only when _syntactic_doi_valid.

    Returns:
        (doi_or_None, arxiv_id_or_None) — both str | None.
        A URL-mined DOI that fails _syntactic_doi_valid is NOT returned
        (guards against a malformed string becoming the registry key, T-03-08).
    """
    # New-style arXiv ID — e.g. 1706.03762.  A trailing version suffix (vN) is
    # matched but kept OUT of the capture group, so a versioned URL yields the bare
    # id ("1706.03762"), converging with the PDF probe's key (WR-01 / SC3).
    m = re.search(r"arxiv\.org/(?:abs|html|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", url)
    if m:
        return None, m.group(1)

    # Old-style arXiv ID — e.g. hep-th/9712052 or math.AG/0601001
    m = re.search(r"arxiv\.org/(?:abs|html|pdf)/([a-z-]+(?:\.[A-Z]{2})?/\d{7})", url)
    if m:
        return None, m.group(1)

    # DOI from a doi.org redirect URL — strip trailing slash and any query string
    m = re.search(r"doi\.org/(10\.\d{4,}/.+?)(?:\?|$)", url)
    if m:
        doi = m.group(1).rstrip("/")
        if _syntactic_doi_valid(doi):
            return doi, None

    return None, None


def _doi_probe(full_text: str) -> "DoiProbeResult | None":
    """
    Pre-gate call: extract DOI, arXiv ID, and title from the FULL document text.

    The probe receives the full MinerU document text (all text + footer blocks, every
    page) so a DOI in Supporting Information is found. The title hint derives from the
    start of full_text (the cover page is at the beginning of the document).

    Raises RuntimeError on Ollama error (D-00d fail-fast — a probe failure means
    the server is unhealthy; proceeding probe-less would derive a garbage title-hash
    registry key, which is the unsafe disposition per T-01.3-14).
    Returns None if the LLM response is unparseable after strip+fence fallback.

    Args:
        full_text: Full document text (_extract_full_text output — all pages). The title
                   is at the start of this text (cover page); the DOI may be anywhere
                   including Supporting Information sections on later pages.
    """
    system = (
        "You are a metadata extractor for scientific papers. "
        "Return ONLY valid JSON matching the schema. Set a field to null if not found. "
        "Preserve the DOI exactly as printed — do not truncate or modify it. "
        "The DOI may appear anywhere in the document, including a Supporting Information "
        "section on a later page. The title is on the cover page."
    )
    prompt = (
        "Extract the DOI, arXiv ID, and title from the following paper text:\n\n"
        + full_text
    )
    raw = _ollama_extraction_call(
        prompt, system, DoiProbeResult.model_json_schema(),
        num_ctx=_estimate_num_ctx(full_text),  # dynamic sizing under configurable cap
        timeout=_SECTION_TIMEOUT,             # section tier — full-doc probe is a large call
    )
    if raw.startswith(("[Ollama error:", "[Ollama timeout:")):
        raise RuntimeError(raw)
    return _parse_extraction_response(raw, DoiProbeResult)


def _fill_metadata(first_page_text: str, probe: "DoiProbeResult | None") -> "PaperMetadata":
    """
    Full metadata fill call (post-cache-miss).

    Uses first-page text plus the probe's title/doi/arxiv hints. Two-strike retry.
    On total failure falls back to a PaperMetadata built from probe hints rather
    than crashing (metadata fill failure is degradable).

    Args:
        first_page_text: Output of _extract_first_page_and_footers().
        probe:           DoiProbeResult from _doi_probe(), or None.

    Returns:
        Filled PaperMetadata (may be minimal fallback on total failure).
    """
    doi_hint = probe.doi if probe else None
    arxiv_hint = probe.arxiv_id if probe else None
    title_hint = probe.title if probe else None

    system = (
        "You are a metadata extractor for scientific papers. "
        "Return ONLY valid JSON matching the schema. "
        "Preserve all values exactly as printed — do not rewrite or infer. "
        "Set a field to null if not found. "
        "Return the journal abbreviation in `journal` (exactly as printed, e.g. 'J. Am. Chem. Soc.'). "
        "Set `journal_full` to the full journal name ONLY if it appears verbatim in the text. "
        "If only an abbreviation is present, set `journal_full` to null. "
        "Never expand or guess the full name from the abbreviation. "
        "The DOI is an opaque identifier — NEVER derive the journal name or abbreviation from the "
        "DOI string or its prefix/suffix (e.g. '10.1016/j.trechm' must NOT produce journal='Trechm'). "
        "Set `journal` ONLY to the journal abbreviation actually printed on the page; "
        "if no journal abbreviation is printed, set `journal` to null."
    )
    prompt = (
        "Extract the metadata from the following paper first page.\n"
        + (f"Hint — DOI from probe: {doi_hint}\n" if doi_hint else "")
        + (f"Hint — arXiv ID from probe: {arxiv_hint}\n" if arxiv_hint else "")
        + (f"Hint — title from probe: {title_hint}\n" if title_hint else "")
        + "\n"
        + first_page_text[:3000]
    )
    schema = PaperMetadata.model_json_schema()

    for attempt in range(2):
        raw = _ollama_extraction_call(prompt, system, schema, num_ctx=4096, timeout=120)
        if raw.startswith("[Ollama error:"):
            raise RuntimeError(raw)
        if raw.startswith("[Ollama timeout:"):
            continue  # timeout is a strike — skip JSON-reminder, go to next attempt
        result = _parse_extraction_response(raw, PaperMetadata)
        if result is not None:
            return result
        if attempt == 0:
            prompt += "\n\nIMPORTANT: Return ONLY raw JSON. Do not wrap in code fences."

    # Both attempts failed (parse failures or timeouts) — fall back to probe hints rather than crashing
    # journal_full is not set here (probe has no journal info) — defaults to None gracefully (GAP D)
    print("[ingest warning: metadata fill_failed — using probe hints as fallback]", file=sys.stderr)
    return PaperMetadata(
        title=title_hint or "",
        doi=doi_hint,
        arxiv_id=arxiv_hint,
    )


def _fill_section(section_text: str, heading: str) -> "SectionFillResult":
    """
    One LLM call per section body (D-01).

    Two-strike retry: on first parse failure (or timeout), retry once; on second
    failure, return fill_failed=True with raw MinerU text carried (D-05/D-06).

    Over-size guard: if _estimate_num_ctx(section_text) >= the active num_ctx cap
    (default 65536 — configurable via config.json "ollama_num_ctx_cap") AND
    len(section_text) > cap * 4, skip the LLM call entirely and return fill_failed=True
    immediately (T-01.3-05). The guard uses the module-level _NUM_CTX_CAP, not a
    hardcoded value, so the cap source is shared with the sizer.

    SYSTEM prompt carries D-07 faithfulness burden AND U+FFFD/hyphenation repair.
    Raises RuntimeError on Ollama unreachable (D-00d).
    """
    # Over-size guard: section larger than the active num_ctx cap → skip LLM, return fill_failed
    # Uses the SAME echo-aware sizing (output_ratio=1.0) as the actual call below so the guard and
    # the call share one sizing basis (T-01.3-05 carried). A section whose 2x budget exceeds the cap
    # still falls back to the cap via the ladder, and a genuinely-too-big section short-circuits here.
    cap = _NUM_CTX_CAP
    estimated_ctx = _estimate_num_ctx(section_text, output_ratio=1.0)
    if estimated_ctx >= cap and len(section_text) > cap * 4:
        print(
            f"[ingest warning: section '{heading}' oversize — skipping LLM fill]",
            file=sys.stderr,
        )
        return SectionFillResult(heading=heading, body=section_text, fill_failed=True)

    system = (
        "You are a scientific text cleaner. "
        "Repair encoding artifacts (U+FFFD, ligature runs) and restore correct "
        "hyphenation for compound terms. "
        "Preserve the author's wording exactly — do not rewrite, summarise, or expand. "
        "Return ONLY valid JSON matching the schema.\n\n"
        # LLM-judge: fold a substantive-content relevance verdict into the same per-section call
        # (no extra LLM calls — Plan 08 GAP A). Figures/tables/boxes stay inside parent section
        # bodies; a caption with descriptive text makes its parent keep=true (deferred: surfacing
        # each as a separate captioned output entry is a future schema change, out of scope here).
        "Also judge whether this section is substantive scientific paper content.\n"
        "Set keep=true when the section is: abstract, introduction, results (and results "
        "subsections), discussion, conclusions, methods, experimental section, or "
        "figure/table/box captions that carry descriptive scientific text.\n"
        "Set keep=false when the section is: author or affiliation lists, acknowledgments, "
        "funding or grant statements, data-availability statements, conflict-of-interest "
        "or 'Notes' blurbs, supporting-information pointers, page headers or footers, "
        "journal banners or advertisements (e.g. promotional taglines), navigation text, "
        "or tiny fragments with no scientific content.\n"
        "If a KEPT section's heading is empty, missing, or non-descriptive, "
        "override it and infer the section's true heading from its body text "
        "(use 'Abstract' or 'Introduction' for the paper's lead block). "
        "Non-descriptive headings that MUST be overridden include: "
        "a bare numbered placeholder ('Section N' where N is a number), "
        "an 'N/A' or similar placeholder, "
        "an article-type label (a single generic word describing the article category rather than its content), "
        "or text that reproduces the paper's full title verbatim rather than naming a section. "
        "Do NOT rename a section that already has a genuine descriptive section name — "
        "names like Introduction, Results, Methods, Discussion, Conclusions, Abstract, "
        "Experimental, References, Background, or their common variants are genuine section names "
        "and must be preserved exactly as given."
    )
    prompt = (
        f"Clean the following section text from a research paper PDF.\n"
        f"Section heading: {heading}\n\n"
        f"{section_text}"
    )
    schema = SectionFillResult.model_json_schema()

    for attempt in range(2):                             # two-strike: try, retry, flag
        raw = _ollama_extraction_call(
            prompt, system, schema,
            # Echo-aware sizing: _fill_section regenerates the full section body as output (≈ input size),
            # so budget ≈ 2x input + overhead. Sized for 1x (prior behaviour), the output JSON truncated
            # mid-generation → unparseable → both strikes failed (observed: test_manuel1 RESULTS 118s,
            # test_manuel3 Experimental Procedure 149s — both completed under the 300s budget but returned
            # invalid JSON). The 300s timeout was CONFIRMED sufficient; truncation was the limiter.
            num_ctx=_estimate_num_ctx(section_text, output_ratio=1.0),
            timeout=_SECTION_TIMEOUT,  # configurable via config.json "ollama_section_timeout" (default 300s,
            # confirmed sufficient — RESULTS 118s, Experimental Procedure 149s under live run; do NOT raise)
        )
        if raw.startswith("[Ollama error:"):
            raise RuntimeError(raw)                      # Ollama unreachable → abort pipeline
        if raw.startswith("[Ollama timeout:"):
            continue  # timeout is a strike — skip JSON-reminder, go to next attempt
        result = _parse_extraction_response(raw, SectionFillResult)
        if result is not None:
            return result
        if attempt == 0:
            # First attempt failed — retry with explicit JSON reminder
            prompt += "\n\nIMPORTANT: Return ONLY raw JSON. Do not wrap in code fences."

    # Both attempts failed — flag the section, carry raw MinerU text (D-06)
    print(f"[ingest warning: section '{heading}' fill_failed]", file=sys.stderr)
    return SectionFillResult(heading=heading, body=section_text, fill_failed=True)


def _fill_references_batched(raw_refs: list) -> "tuple[list, int]":
    """
    Fill reference list in batches of BATCH_SIZE (~10 refs per call, D-02).

    Returns (filled_refs, failure_count).
    A failed batch increments failure_count by 1 and flags each ref in that
    batch with fill_failed=True, carrying the raw string. Raw loss is bounded
    to at most 10 refs per failed batch.

    Raises RuntimeError on Ollama unreachable (D-00d).
    """
    BATCH_SIZE = 10
    system = (
        "You are a reference parser for scientific papers. "
        "For each reference string, extract the number, DOI, title, and year if present. "
        "Set fields to null if not found. Do not invent values. "
        "Return ONLY valid JSON matching the schema.\n\n"
        # Format-agnostic numbering: the reference ordinal may appear in any of these styles at the
        # START of the string: (1) parenthesized, 1. period-delimited, [1] bracket, or a bare leading
        # '1 ' followed by the author. Read whichever leading list marker / ordinal is present and
        # return that integer in `number`. NEVER use an inline parenthetical number — a publication
        # year like (2005), a volume/issue like 12 (12), or a page range — as the `number` value.
        # If no leading ordinal is present at the start of the string, set `number` to null.\n\n
        # Reference-vs-footnote judgment: also set `is_reference` to decide whether each string is
        # a genuine bibliographic reference (is_reference=true) or a non-reference footnote that
        # leaked into the list — an author/affiliation block, a corresponding-author line, an email
        # address, a lab or department postal address, an ORCID or funding fragment, or any line
        # that is clearly not a citation (is_reference=false). Set is_reference=true for real
        # bibliography entries; set is_reference=false for affiliation/footnote lines.
        "The reference number (`number`) is the ordinal at the START of the string. "
        "It may be written as (1), 1., [1], or a bare leading '1 '. "
        "Read the leading list marker and return that integer. "
        "NEVER use an inline parenthetical — a year like (2005) or volume like 12 (12) — as `number`. "
        "If no leading ordinal is present, set `number` to null.\n"
        "Also set `is_reference`: true for genuine bibliography entries; false for "
        "affiliation/corresponding-author blocks, email lines, lab/department addresses, "
        "ORCID/funding fragments, or any non-citation line."
    )
    filled: list = []
    failures = 0
    kept_count = 0
    dropped_count = 0
    total_batches = (len(raw_refs) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(raw_refs), BATCH_SIZE):
        batch_no = i // BATCH_SIZE + 1
        _log(f"reference batch {batch_no}/{total_batches}")
        batch = raw_refs[i:i + BATCH_SIZE]
        batch_text = "\n".join(
            ref.get("raw", str(ref)) if isinstance(ref, dict) else str(ref)
            for ref in batch
        )
        prompt = f"Parse the following references:\n\n{batch_text}"
        schema = RefBatchResult.model_json_schema()
        result = None

        for attempt in range(2):
            # Echo-aware sizing: the ref-batch call regenerates all parsed references as JSON output
            # (≈ input size). Fixed num_ctx=4096 truncated large batches; output_ratio=1.0 allocates
            # budget for input + equal-size output. timeout=120 is unchanged (ref batches are small;
            # no timeout observed in live runs).
            raw = _ollama_extraction_call(prompt, system, schema,
                                          num_ctx=_estimate_num_ctx(batch_text, output_ratio=1.0),
                                          timeout=120)
            if raw.startswith("[Ollama error:"):
                raise RuntimeError(raw)
            if raw.startswith("[Ollama timeout:"):
                continue  # timeout is a strike — skip JSON-reminder, go to next attempt
            result = _parse_extraction_response(raw, RefBatchResult)
            if result is not None:
                break
            if attempt == 0:
                prompt += "\n\nIMPORTANT: Return ONLY raw JSON. Do not wrap in code fences."

        if result is not None:
            # Filter: drop ONLY entries with an explicit is_reference=False from a SUCCESSFUL parse.
            # Conservative default (mirrors Plan 08's keep=True): fill_failed/uncertain always retained.
            # is_reference key MUST NOT reach the output — excluded via model_dump(exclude=...).
            for r in result.refs:
                if r.is_reference is False and r.fill_failed is False:
                    # Explicit is_reference=false from a successful parse → drop (non-reference footnote)
                    dropped_count += 1
                else:
                    # Keep: is_reference=true, or uncertain/missing verdict (defaults True) — always retain
                    filled.append(r.model_dump(exclude={"is_reference"}))
                    kept_count += 1
        else:
            # Both attempts failed — flag each ref in batch as fill_failed (no verdict → always retain)
            failures += 1
            for ref in batch:
                raw_str = ref.get("raw", str(ref)) if isinstance(ref, dict) else str(ref)
                filled.append(RefEntry(raw=raw_str, fill_failed=True).model_dump(exclude={"is_reference"}))
                kept_count += 1

    _log(f"reference filter: kept {kept_count}, dropped {dropped_count}")

    # Targeted U+FFFD repair on the reference path only (Plan 10 Item 1d, INGEST-01).
    # Replaces the lone replacement character (observed at em-dash positions in out1 ref45)
    # with an em-dash. Scoped to ref raw/title strings only — no blanket replacement elsewhere.
    _UFFFD = "�"
    for ref in filled:
        for field in ("raw", "title"):
            val = ref.get(field)
            if isinstance(val, str) and _UFFFD in val:
                ref[field] = val.replace(_UFFFD, "—")  # em-dash at observed position

    return filled, failures


# ---------------------------------------------------------------------------
# Crossref optional same-paper validator (Plan 03)
# ---------------------------------------------------------------------------

# WR-03 placeholder-email guard (Plan 11): predicate consulted by ALL Crossref
# helpers to ensure the shipped placeholder mailto is never sent to api.crossref.org.
_CROSSREF_PLACEHOLDER_EMAILS = {"your-email@example.com", "unknown@example.com"}

# Track whether the placeholder warning has been emitted this process (one per run)
_crossref_placeholder_warned = False


def _crossref_contact_ok(config: dict) -> bool:
    """Return True if crossref_contact_email is a real (non-placeholder) email.

    When the email is missing, empty, or the shipped placeholder, returns False
    and emits ONE stderr warning per process. All Crossref helpers consult this
    before making any HTTPS request so the placeholder mailto is never transmitted.
    """
    global _crossref_placeholder_warned
    email = (config.get("crossref_contact_email") or "").strip()
    if not email or email in _CROSSREF_PLACEHOLDER_EMAILS:
        if not _crossref_placeholder_warned:
            print(
                "[ingest warning: crossref skipped — set a real crossref_contact_email "
                "in config.json (placeholder mailto not sent)]",
                file=sys.stderr,
            )
            _crossref_placeholder_warned = True
        return False
    return True


def _crossref_validate(doi: str, title_hint: str | None, config: dict) -> None:
    """
    Resolve DOI at Crossref over HTTPS and run a local-LLM same-paper check (D-13..D-16).

    Only called when config["crossref_validate"] is truthy and a syntactically-valid
    DOI is available. Outbound data: DOI string in URL only — no paper content leaves
    the machine (V9 privacy constraint).

    Behavior:
      - Network failure (URLError, JSONDecodeError): print [ingest warning: crossref unreachable
        ...] to stderr and return (D-16 fail-open).
      - Crossref returns no title (KeyError/IndexError/TypeError): return (nothing to compare).
      - title_hint is falsy: return (nothing to compare).
      - LLM check returns [Ollama error: ...]: print [ingest warning: crossref LLM check
        unavailable ...] and return (fail-open — LLM check is best-effort).
      - LLM verdict same_paper is explicitly False: raise RuntimeError([ingest error: ...])
        naming both titles (D-15 — hard abort before registry write).
      - Unparseable LLM verdict: return (fail-open).

    Args:
        doi:        Syntactically-valid DOI string (already validated by _syntactic_doi_valid).
        title_hint: Title from DOI probe (may be None).
        config:     Loaded config dict; reads crossref_contact_email for User-Agent (V14).
    """
    # WR-03: skip when email is placeholder/empty (no bogus mailto sent)
    if not _crossref_contact_ok(config):
        return

    url = f"https://api.crossref.org/works/{doi}"
    contact = config.get("crossref_contact_email", "unknown@example.com")
    req = urllib.request.Request(url, method="GET")
    req.add_header(
        "User-Agent",
        f"ragsearch-assistant/1.3 (mailto:{contact})",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        print(
            "[ingest warning: crossref unreachable — proceeding with syntactic validation only]",
            file=sys.stderr,
        )
        return  # D-16: fail-open on network / parse error (includes TimeoutError)

    # Extract Crossref title; missing key means nothing to compare — fail-open
    try:
        crossref_title = data["message"]["title"][0]
    except (KeyError, IndexError, TypeError):
        return  # no title in Crossref response — proceed

    # Nothing to compare if probe produced no title
    if not title_hint:
        return

    # One local-LLM same-paper check (D-14)
    schema = {
        "type": "object",
        "properties": {"same_paper": {"type": "boolean"}},
        "required": ["same_paper"],
    }
    system = "You are a bibliographic metadata checker."
    prompt = (
        f"DOI: {doi}\n"
        f"Title A (from paper): {title_hint}\n"
        f"Title B (from Crossref): {crossref_title}\n\n"
        "Do titles A and B refer to the same paper? "
        "Return JSON with a single boolean field: same_paper."
    )
    raw = _ollama_extraction_call(prompt, system, schema, num_ctx=2048, timeout=30)
    if raw.startswith(("[Ollama error:", "[Ollama timeout:")):
        # LLM check unavailable — fail-open (best-effort; DOI is already syntactically valid)
        print(
            "[ingest warning: crossref LLM check unavailable — proceeding]",
            file=sys.stderr,
        )
        return

    # Parse verdict; fail-open on unparseable response
    try:
        verdict = json.loads(raw)
        same = verdict.get("same_paper")
    except (json.JSONDecodeError, AttributeError):
        return  # fail-open on parse error

    if same is False:  # explicitly False — confirmed mismatch
        raise RuntimeError(
            f"[ingest error: Crossref DOI {doi!r} resolves to a different paper "
            f"({crossref_title!r} vs {title_hint!r}). "
            "Please supply the correct DOI and retry.]"
        )
    # same is True or None (None = model returned unexpected shape) — proceed


def _crossref_journal_full(doi: str, config: dict) -> str | None:
    """
    Resolve DOI at Crossref over HTTPS and return the full journal name.

    Returns ``message['container-title'][0]`` verbatim from the Crossref API response,
    or ``None`` on any network / parse / missing-field error (fail-open contract).

    Outbound data: the DOI string in the URL only — no paper content leaves the machine
    (V9 privacy constraint preserved; identical trust level to _crossref_validate).

    Sits alongside _crossref_validate, sharing its HTTPS GET + User-Agent (V14) +
    DOI-only + fail-open conventions. Do NOT refactor or call _crossref_validate.

    Note: config.json crossref_validate=true enables BOTH the existing same-paper guard
    (_crossref_validate) AND this journal enrichment helper. The pipeline is intentionally
    no longer fully-offline by default for the Crossref DOI-only path (Plan 08 user decision);
    only the DOI string leaves the machine. CLAUDE.md's offline-by-default line is superseded
    for this DOI-only path but CLAUDE.md is NOT edited here.
    """
    # WR-03: skip when email is placeholder/empty (no bogus mailto sent)
    if not _crossref_contact_ok(config):
        return None

    url = f"https://api.crossref.org/works/{doi}"
    contact = config.get("crossref_contact_email", "unknown@example.com")
    req = urllib.request.Request(url, method="GET")
    req.add_header(
        "User-Agent",
        f"ragsearch-assistant/1.3 (mailto:{contact})",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        _log("crossref journal lookup unavailable")
        return None  # fail-open — one quiet log line; ingest proceeds

    # Extract container-title[0]; fail-open on missing/empty/wrong-type
    try:
        full = data["message"]["container-title"][0]
    except (KeyError, IndexError, TypeError):
        return None
    if not full:
        return None
    return full  # verbatim from Crossref — extract-never-infer (Plan 08)


def _crossref_published_year(doi: str, config: dict) -> int | None:
    """
    Resolve DOI at Crossref over HTTPS and return the published year.

    Returns ``message['published']['date-parts'][0][0]`` verbatim from the Crossref
    API response, or ``None`` on any network / parse / missing-field error (fail-open
    contract). Falls back to ``published-print`` or ``published-online`` when the
    ``published`` key is absent.

    Outbound data: the DOI string in the URL only — no paper content leaves the machine
    (V9 privacy constraint preserved; identical trust level to _crossref_journal_full).

    Mirrors _crossref_journal_full conventions: HTTPS GET + User-Agent (V14) +
    DOI-only + fail-open. INGEST-01.
    """
    # WR-03: skip when email is placeholder/empty (no bogus mailto sent)
    if not _crossref_contact_ok(config):
        return None

    url = f"https://api.crossref.org/works/{doi}"
    contact = config.get("crossref_contact_email", "unknown@example.com")
    req = urllib.request.Request(url, method="GET")
    req.add_header(
        "User-Agent",
        f"ragsearch-assistant/1.3 (mailto:{contact})",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        _log("crossref year lookup unavailable")
        return None  # fail-open — one quiet log line; ingest proceeds

    # Extract published year from date-parts[0][0]; fall back to published-print / published-online
    msg = data.get("message", {})
    for field in ("published", "published-print", "published-online"):
        try:
            year = msg[field]["date-parts"][0][0]
            if isinstance(year, int):
                return year
        except (KeyError, IndexError, TypeError):
            continue
    return None


# Crossref title->DOI search constants (Plan 05)
# ---------------------------------------------------------------------------
# Minimum relevance score from api.crossref.org/works?query.bibliographic that
# must be met (strictly >=) before a returned DOI is considered a candidate.
# Crossref scores are unbounded floats; empirically, good matches land well
# above 50. A threshold of 70 rejects weak/partial matches while accepting
# strong bibliographic hits.
_CROSSREF_TITLE_MATCH_MIN_SCORE: float = 70.0

# Maximum result rows to request from the Crossref works-query endpoint.
# Set to 1 because only the top item (items[0]) is ever consulted; requesting
# one row keeps the response payload tiny and latency low (T-04-05-03 DoS mitigation).
_CROSSREF_TITLE_SEARCH_ROWS: int = 1

# Crossref OR-combines repeated filter names, so this single-string value keeps
# only publication-like record types and prevents image-data deposits, decision
# letters, and PDB components from outranking the real article at items[0].
_CROSSREF_TITLE_TYPE_FILTER: str = (
    "type:journal-article,type:posted-content,type:proceedings-article,type:book-chapter"
)

# Backoff slept between bounded retry attempts on a transient transport error
# (URLError / TimeoutError) in _crossref_title_to_doi. Bounded by crossref_retries
# (default 1 = no retry, byte-identical to prior single-attempt behavior).
_CROSSREF_RETRY_BACKOFF_SECONDS: float = 0.5


def _crossref_title_to_doi(title: str, config: dict) -> str | None:
    """Search Crossref by bibliographic title and return the top DOI if above threshold.

    Outbound data: the reference title in the URL query string and the contact
    email in the User-Agent header (T-04-05-01 — opt-in only; gated by
    _crossref_contact_ok + crossref_validate flag).

    Behavior:
      - Returns None immediately when _crossref_contact_ok is False (no request sent).
      - URL-encodes the title via urllib.parse.urlencode (T-04-05-04 injection defence).
      - Requests at most _CROSSREF_TITLE_SEARCH_ROWS results; considers only the
        highest-scoring item (items[0] from the already-sorted Crossref response).
      - Returns the item's DOI when its score >= _CROSSREF_TITLE_MATCH_MIN_SCORE,
        else returns None (T-04-05-02 false-match gate, first layer).
      - Retries a bounded number of times (config["crossref_retries"], default 1 =
        single attempt, unchanged) ONLY on a transient transport error
        (URLError / TimeoutError), sleeping _CROSSREF_RETRY_BACKOFF_SECONDS between
        attempts (Gap E). A malformed JSON body, empty items, sub-threshold score, or
        KeyError/IndexError/TypeError/ValueError are real "no match" answers and are
        NOT retried.
      - Returns None on empty items list, sub-threshold score, malformed JSON body, or
        exhausted transient-error retries — fail-open, one quiet stderr line
        (T-04-05-03 / mirrors _crossref_journal_full).

    Args:
        title:  Reference title string (LLM-filled; may contain any unicode).
        config: Loaded config dict; reads crossref_contact_email for User-Agent and
                crossref_retries for the bounded attempt count (default 1).

    Returns:
        DOI string (e.g. "10.1073/pnas.x") or None.
    """
    # WR-03: skip when email is placeholder/empty (no bogus mailto sent)
    if not _crossref_contact_ok(config):
        return None

    contact = config.get("crossref_contact_email", "unknown@example.com")
    params = urllib.parse.urlencode({
        "query.bibliographic": title,
        "rows": _CROSSREF_TITLE_SEARCH_ROWS,
        "select": "DOI,score,title",
        "filter": _CROSSREF_TITLE_TYPE_FILTER
    })
    url = f"https://api.crossref.org/works?{params}"
    req = urllib.request.Request(url, method="GET")
    req.add_header(
        "User-Agent",
        f"ragsearch-assistant/1.3 (mailto:{contact})",
    )

    attempts = max(1, int(config.get("crossref_retries", 1)))
    data = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=config.get("crossref_timeout", 30)) as resp:
                data = json.loads(resp.read())
            break  # success — stop retrying
        except (urllib.error.URLError, TimeoutError):
            if attempt < attempts - 1:
                time.sleep(_CROSSREF_RETRY_BACKOFF_SECONDS)
                continue
            _log("crossref title->doi lookup unavailable")
            return None  # fail-open after exhausting bounded retries
        except json.JSONDecodeError:
            _log("crossref title->doi lookup unavailable")
            return None  # malformed body is a real answer — not retried

    # Extract top item (Crossref returns items ranked by relevance score)
    try:
        items = data["message"]["items"]
        if not items:
            return None
        top = items[0]
        score = float(top["score"])
        doi = top["DOI"]
    except (KeyError, IndexError, TypeError, ValueError):
        return None

    if score < _CROSSREF_TITLE_MATCH_MIN_SCORE:
        return None

    return doi


def _enrich_references_with_crossref(skeleton: dict, config: dict) -> None:
    """Assign DOIs to doiless references via opt-in Crossref title search (Plan 05 Layer 3).

    Runs in-place on skeleton["extraction"]["references"]. For each reference
    that has a truthy title and a falsy doi:
      1. Calls _crossref_title_to_doi to get a candidate DOI (score-gated).
      2. If a DOI is returned, calls the existing _crossref_validate LLM
         same-paper confirmation:
           - Returns normally -> assigns ref["doi"] = doi.
           - Raises RuntimeError (confirmed mismatch, D-15) -> leaves doi unset,
             logs, continues (fail-open per ref).
    Each reference's enrichment is wrapped in its own try/except so one failure
    never aborts the cascade (T-04-05-03).

    Gate: returns immediately (no network, no mutation) unless
    config["crossref_validate"] is truthy AND _crossref_contact_ok(config).

    Args:
        skeleton: Mutable PaperJSON dict (extraction.references mutated in-place).
        config:   Loaded config dict; reads crossref_validate and crossref_contact_email.
    """
    if not config.get("crossref_validate", False):
        return
    if not _crossref_contact_ok(config):
        return

    refs = skeleton.get("extraction", {}).get("references", [])
    for ref in refs:
        if not ref.get("title") or ref.get("doi"):
            continue  # skip refs with no title or already-resolved DOI
        try:
            candidate_doi = _crossref_title_to_doi(ref["title"], config)
            if not candidate_doi:
                continue
            # Second gate: LLM same-paper confirmation (T-04-05-02, D-15)
            try:
                _crossref_validate(candidate_doi, ref["title"], config)
            except RuntimeError:
                # Confirmed mismatch — do not assign DOI; continue with next ref
                _log(f"crossref title enrichment: confirmed mismatch for ref title {ref['title']!r}")
                continue
            ref["doi"] = candidate_doi
        except Exception as exc:  # noqa: BLE001 — fail-open per ref
            _log(f"crossref title enrichment: error on ref {ref.get('title')!r}: {exc}")


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def _quality_gate(paperjson: dict, source: str = "pdf") -> str | None:
    """
    Validate extraction output quality (INGEST-03 redefined per D-11).

    Returns an [ingest error: ...]-prefixed message when the result is garbage:
      - no title detected (metadata.title is None or empty), OR
      - total non-noise text is near-empty (below threshold).

    Returns None when output passes quality checks.

    source: "pdf" (default) uses PDF-specific wording; "web" uses web-appropriate wording.
    """
    extraction = paperjson.get("extraction", {})
    metadata = extraction.get("metadata", {})
    title = metadata.get("title")

    if not title:
        if source == "web":
            return (
                "[ingest error: extraction produced no usable content — "
                "no title found on web page]"
            )
        return (
            "[ingest error: extraction produced no usable content — "
            "possible scanned/garbage PDF (no title detected)]"
        )

    # Count total content length across ALL block types (WR-05, Plan 11).
    # Text blocks: count plain or display.
    # Table/equation blocks: count plain placeholder (e.g. "[Table: ...]").
    # Figure blocks: count caption text.
    # This ensures figure/table-heavy papers are not spuriously rejected.
    total_text_len = 0
    for section in extraction.get("sections", []):
        for block in section.get("blocks", []):
            btype = block.get("type", "")
            if btype == "text":
                total_text_len += len(block.get("plain", "") or block.get("display", ""))
            elif btype in ("table", "equation"):
                total_text_len += len(block.get("plain", "") or block.get("display", ""))
            elif btype == "figure":
                total_text_len += len(block.get("caption", ""))

    # Near-empty threshold: 100 chars of non-noise content is a reasonable minimum
    # for a real paper (even an abstract is ~500 chars)
    NEAR_EMPTY_THRESHOLD = 100
    if total_text_len < NEAR_EMPTY_THRESHOLD:
        if source == "web":
            return (
                f"[ingest error: extraction produced no usable content — "
                f"web page body near-empty (total content {total_text_len} chars "
                f"below threshold {NEAR_EMPTY_THRESHOLD})]"
            )
        return (
            f"[ingest error: extraction produced no usable content — "
            f"possible scanned/garbage PDF (total content {total_text_len} chars "
            f"below threshold {NEAR_EMPTY_THRESHOLD})]"
        )

    return None


# ---------------------------------------------------------------------------
# MinerU invocation
# ---------------------------------------------------------------------------

def _run_mineru(
    pdf_path: str,
    out_dir: str,
    mineru_exe: str,
    timeout: int,
    force_extract: bool,
) -> None:
    """
    Run MinerU on the given PDF into out_dir, or reuse existing output (D-15).

    If content_list.json already exists in out_dir and force_extract is False,
    the GPU step is skipped. Otherwise MinerU is invoked via subprocess.

    Raises subprocess.TimeoutExpired if MinerU exceeds the timeout.
    Raises RuntimeError if MinerU exits non-zero.
    """
    # D-15: reuse check — look for any existing content_list.json under out_dir
    existing = _find_content_list_path(out_dir)
    if existing and not force_extract:
        return  # reuse existing output

    # D-13: MinerU writes into out_dir; we pass it as the -o argument
    # T-01.2-01: pass path as a separate argv element (list, no shell=True)
    result = subprocess.run(
        [mineru_exe, "-p", pdf_path, "-o", out_dir],
        timeout=timeout,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "(no stderr)"
        raise RuntimeError(f"MinerU exited with code {result.returncode}: {stderr}")


def _run_defuddle(url: str, defuddle_exe: str, timeout: int = 60) -> str:
    """
    Run defuddle on a URL and return the extracted markdown text (D-08 / T-03-02).

    Invokes defuddle as an argv list (never shell=True) so the URL is passed as a
    single literal token without shell-metacharacter expansion.

    Raises RuntimeError on non-zero exit (content gated or unreachable) or on empty
    stdout (D-08 Pitfall 2: defuddle exits 0 but produced nothing).
    Lets subprocess.TimeoutExpired propagate to the caller for clean error handling.
    """
    result = subprocess.run(
        [defuddle_exe, "parse", url, "--md", "--frontmatter"],
        timeout=timeout,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "(no stderr)"
        raise RuntimeError(
            f"web extraction failed — content gated or unreachable: {url} "
            f"(defuddle exit {result.returncode}: {stderr})"
        )
    md = result.stdout
    if not md.strip():
        raise RuntimeError(f"web extraction failed — empty output: {url}")
    return md


def _find_content_list_path(out_dir: str) -> str | None:
    """
    Locate the *_content_list.json produced by MinerU under out_dir.

    Searches hybrid_auto/*_content_list.json first (canonical MinerU path per §1),
    then falls back to a recursive glob for *_content_list.json.
    Returns the first match or None.
    """
    # Canonical path: <out_dir>/hybrid_auto/<doc>_content_list.json
    canonical = glob.glob(str(pathlib.Path(out_dir) / "hybrid_auto" / "*_content_list.json"))
    if canonical:
        return canonical[0]
    # Fallback: recursive search
    fallback = glob.glob(str(pathlib.Path(out_dir) / "**" / "*_content_list.json"), recursive=True)
    if fallback:
        return fallback[0]
    return None


def _find_content_list(out_dir: str) -> str:
    """
    Return the path to the content_list.json under out_dir, or raise on miss.

    Raises RuntimeError with a bracketed error message if no file is found.
    """
    path = _find_content_list_path(out_dir)
    if not path:
        raise RuntimeError(f"content_list.json not found in {out_dir}")
    return path


# ---------------------------------------------------------------------------
# content_list.json parser
# ---------------------------------------------------------------------------

def _route_block(block: dict, current_section: dict, references: list) -> None:
    """
    Route a single content_list block to the appropriate output bucket.

    Noise blocks (NOISE_BLOCK_TYPES) are silently dropped.
    list/ref_text blocks go to references.
    table/equation/image blocks become typed blocks in the current section.
    text blocks become text blocks in the current section.
    """
    btype = block.get("type", "")

    if btype in NOISE_BLOCK_TYPES:
        return  # D-02: drop noise

    if btype == "list":
        sub_type = block.get("sub_type", "")
        if sub_type == "ref_text":
            # D-01: references arrive as list/ref_text; collect raw items
            for item in block.get("list_items", []):
                references.append({"raw": item})
        # Non-ref_text list blocks treated as text (body list content)
        else:
            text = block.get("text", "")
            if text.strip():
                display = _build_display(text)
                plain = _build_plain(text)
                current_section["blocks"].append({
                    "type": "text",
                    "display": display,
                    "plain": plain,
                })
        return

    if btype == "table":
        # D-06: keep genuine table blocks as typed blocks
        content = block.get("text", "")
        # D-20: caption-derived plain placeholder (first words of caption if present)
        table_caption = block.get("table_caption", [])
        if isinstance(table_caption, list):
            table_caption = " ".join(table_caption)
        if table_caption.strip():
            words = table_caption.split()
            snippet = " ".join(words[:6])
            if len(words) > 6:
                snippet += "…"
            plain_placeholder = f"[Table: {snippet}]"
        else:
            plain_placeholder = "[Table]"
        current_section["blocks"].append({
            "type": "table",
            "display": content,
            "plain": plain_placeholder,
        })
        return

    if btype == "equation":
        # D-06: keep genuine equation blocks as typed blocks
        content = block.get("text", "")
        # D-20: caption-derived plain placeholder
        eq_caption = block.get("equation_caption", [])
        if isinstance(eq_caption, list):
            eq_caption = " ".join(eq_caption)
        if eq_caption.strip():
            words = eq_caption.split()
            snippet = " ".join(words[:6])
            if len(words) > 6:
                snippet += "…"
            plain_placeholder = f"[Equation: {snippet}]"
        else:
            plain_placeholder = "[Equation]"
        current_section["blocks"].append({
            "type": "equation",
            "display": content,
            "plain": plain_placeholder,
        })
        return

    if btype == "image":
        # D-03: quarantine VLM content via _quarantine_figure; keep caption + img_path
        current_section["blocks"].append(_quarantine_figure(block))
        return

    if btype == "text":
        text = block.get("text", "")
        if not text.strip():
            return
        display = _build_display(text)
        plain = _build_plain(text)
        current_section["blocks"].append({
            "type": "text",
            "display": display,
            "plain": plain,
        })
        return


def _parse_content_list(blocks: list) -> dict:
    """
    Parse a content_list.json block list into structured sections, references, and metadata.

    Routes each block by type (D-01/D-02/D-06). Derives a minimal title from the
    first text_level-1 block on page 0 (the only metadata field needed pre-fill).
    Reference strings are collected as raw items; Plan 02 LLM fill will structure them.

    Returns a dict with:
      - sections: list of {heading, level, blocks[]}
      - references: list of raw {raw} objects (structured by LLM fill in Plan 02)
      - title: first text_level-1 block text on page 0 (or None)
      - metadata: minimal metadata dict with title only (LLM fill populates remaining fields)
    """
    sections = []
    raw_ref_items = []
    title = None

    # Start with a default section to hold content before the first heading
    current_section = {"heading": "", "level": 0, "blocks": []}

    for block in blocks:
        btype = block.get("type", "")
        text_level = block.get("text_level")
        page_idx = block.get("page_idx", -1)

        # Collect raw reference strings before routing (so we can parse them later)
        if btype == "list" and block.get("sub_type") == "ref_text":
            for item in block.get("list_items", []):
                raw_ref_items.append(item)
            continue

        # Inline title detection: first text_level-1 block on page 0 (registry key + quality gate)
        if title is None and btype == "text" and text_level == 1 and page_idx == 0:
            title = _build_plain(block.get("text", "").strip())

        # Section boundary: text_level 2 starts a new section
        if btype == "text" and text_level == 2:
            # Flush current section if it has any blocks
            if current_section["blocks"]:
                sections.append(current_section)
            heading_text = block.get("text", "").strip()
            current_section = {
                "heading": _build_plain(heading_text),
                "level": 2,
                "blocks": [],
            }
            continue

        # Route all other blocks (text, table, equation, image, noise-drop)
        _route_block(block, current_section, raw_ref_items)

    # Flush final section
    if current_section["blocks"]:
        sections.append(current_section)

    # Minimal metadata: title only (LLM fill in Plan 02 populates remaining fields)
    metadata = {
        "title": title,
        "authors": None,
        "year": None,
        "journal": None,
        "doi": None,
        "arxiv_id": None,
        "accession_codes": [],
    }

    # References as raw items; LLM fill in Plan 02 structures them
    references = [{"raw": item} for item in raw_ref_items]

    return {
        "title": title,
        "sections": sections,
        "references": references,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Phase 3 — defuddle markdown parser
# ---------------------------------------------------------------------------

def _parse_defuddle_markdown(md_text: str) -> dict:
    """
    Parse defuddle-produced markdown into the same section/block/metadata shape
    as _parse_content_list (D-02 / byte-compatible so _assemble_paperjson consumes
    it unchanged).

    Line-by-line rules:
    - First `# ` line → title (H1 text, without the `# ` prefix).
    - `## ` line     → flush current section, start a level-2 section.
    - `### ` line    → flush current section, start a level-3 section.
    - Blank line     → flush the current paragraph into a text block.
    - Other lines    → accumulate into the current paragraph.

    Text blocks: {type:"text", display:_build_display(text), plain:_build_plain(text)}.
    Citation markers [^N] and inline $…$ survive in `display`; _build_plain strips `$…$`.

    Fallback: if no ## / ### headings were produced but text exists, a single fallback
    section {heading:"", level:0} is emitted (same as the default section).

    References stay empty (Phase 4). The `## References` block is left as ordinary
    text blocks.
    """
    lines = md_text.split("\n")

    # Detect and strip a leading YAML frontmatter block (---\n…\n---).
    # Extract title: from it as a fallback when no # H1 is present in the body.
    fm_title: str | None = None
    if lines and lines[0].strip() == "---":
        closing_idx = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                closing_idx = i
                break
        if closing_idx is not None:
            for fm_line in lines[1:closing_idx]:
                stripped = fm_line.strip()
                if stripped.startswith("title:"):
                    raw_value = stripped[len("title:"):].strip()
                    # Strip a single matching pair of surrounding quotes (" or ')
                    if (
                        len(raw_value) >= 2
                        and raw_value[0] in ('"', "'")
                        and raw_value[-1] == raw_value[0]
                    ):
                        raw_value = raw_value[1:-1]
                    candidate = _build_plain(raw_value)
                    fm_title = candidate if candidate else None
                    break
            # Remove frontmatter lines so they never enter paragraph accumulation
            lines = lines[closing_idx + 1:]

    title = None
    sections: list[dict] = []
    current_section: dict = {"heading": "", "level": 0, "blocks": []}
    current_para_lines: list[str] = []

    def _flush_paragraph() -> None:
        if current_para_lines:
            text = "\n".join(current_para_lines).strip()
            if text:
                current_section["blocks"].append({
                    "type": "text",
                    "display": _build_display(text),
                    "plain": _build_plain(text),
                })
            current_para_lines.clear()

    for line in lines:
        if line.startswith("# ") and title is None:
            # H1 — first only; flush any accumulated para then extract title
            _flush_paragraph()
            title = _build_plain(line[2:].strip())
        elif line.startswith("## ") or line.startswith("### "):
            # Section boundary — flush para, flush section, start new section
            _flush_paragraph()
            if current_section["blocks"]:
                sections.append(current_section)
            if line.startswith("## "):
                heading_text = line[3:].strip()
                level = 2
            else:
                heading_text = line[4:].strip()
                level = 3
            current_section = {
                "heading": _build_plain(heading_text),
                "level": level,
                "blocks": [],
            }
        elif not line.strip():
            # Blank line — flush current paragraph
            _flush_paragraph()
        else:
            # Non-blank, non-heading — accumulate into current paragraph
            current_para_lines.append(line)

    # Flush final paragraph and section
    _flush_paragraph()
    if current_section["blocks"]:
        sections.append(current_section)

    # Frontmatter title fallback: use when no # H1 was found in the body
    if title is None and fm_title:
        title = fm_title

    metadata = {
        "title": title,
        "authors": None,
        "year": None,
        "journal": None,
        "doi": None,
        "arxiv_id": None,
        "accession_codes": [],
    }

    return {
        "title": title,
        "sections": sections,
        "references": [],
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# PaperJSON assembly
# ---------------------------------------------------------------------------

def _assemble_paperjson(parsed: dict, provenance: dict) -> dict:
    """
    Assemble the locked PaperJSON v2 shape from parsed content and provenance info.

    Produces three top-level namespaces (D-08/D-22):
      extraction: ground truth (metadata, sections, references)
      analysis:   empty skeleton (Phase 2 fills)
      provenance: pdf_sha256, source info, schema_version, backend, etc.
    """
    # Use metadata from parsed["metadata"]; fall back gracefully when key is absent.
    # Plan 01: metadata has title only; Plan 02 LLM fill populates remaining fields.
    mined = parsed.get("metadata") or {}
    extraction = {
        "metadata": {
            "title": mined.get("title") or parsed.get("title"),
            "authors": mined.get("authors"),
            "year": mined.get("year"),
            "journal": mined.get("journal"),
            "doi": mined.get("doi"),
            "arxiv_id": mined.get("arxiv_id"),
            "accession_codes": mined.get("accession_codes", []),
        },
        "sections": parsed.get("sections", []),
        "references": parsed.get("references", []),
    }

    # D-22: analysis namespace as empty skeleton; Phase 2 fills
    analysis = {
        "generated_by": None,
        "summary": None,
        "claims": [],
        "methods_overview": None,
        "results": None,  # D-09: Phase 2 fills Results section
        "limitations": [],
        "open_questions": [],
        "entities": [],
        "topics": [],
        "connections": {
            "builds_on": [],
            "contradicts": [],
            "same_domain": [],
        },
    }

    return {
        "extraction": extraction,
        "analysis": analysis,
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# SHA-256 prefix length for title-hash registry keys (16 hex chars = 64-bit prefix).
# Collision probability is negligible (birthday bound ~1 in 2^32 for up to ~100k papers).
_TITLE_HASH_PREFIX_LEN = 16


def _normalize_title(title: str) -> str:
    """
    Normalize a paper title for stable SHA-256 hashing.

    Lowercases, strips leading/trailing whitespace, and collapses all
    whitespace and punctuation runs to a single space so that minor
    typography differences (dashes, extra spaces) do not produce
    different hash keys for the same paper.
    """
    title = title.lower().strip()
    # Collapse runs of whitespace and punctuation to a single space
    title = re.sub(r"[\s\W]+", " ", title).strip()
    return title


def _registry_key(metadata: dict) -> str:
    """
    Derive the registry key for a paper from its metadata.

    Priority: DOI → arXiv ID → SHA-256 prefix of normalized title (D-07).

    Args:
        metadata: Dict with optional keys doi, arxiv_id, title.

    Returns:
        DOI string if truthy; arXiv ID if truthy and DOI absent; else
        "sha256:<16-hex-chars>" derived from the normalized title.
    """
    doi = metadata.get("doi")
    if doi:
        return doi

    arxiv_id = metadata.get("arxiv_id")
    if arxiv_id:
        return arxiv_id

    title = metadata.get("title") or ""
    normalized = _normalize_title(title)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return "sha256:" + digest[:_TITLE_HASH_PREFIX_LEN]


def _read_registry(registry_path: str) -> dict:
    """
    Read the papers registry JSON file.

    Returns {} if the file does not exist or is empty/whitespace-only (treats these
    as an initialised-but-empty registry, e.g. a 0-byte file created by a prior
    interrupted write). Raises ValueError on corrupt non-empty files to avoid
    silently clobbering user data.

    Args:
        registry_path: Absolute path to the registry JSON file.

    Returns:
        Parsed registry dict, or empty dict if file absent or empty.

    Raises:
        ValueError: When the file exists, is non-empty, but is not valid JSON.
    """
    path = pathlib.Path(registry_path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    # Treat missing, empty, or whitespace-only file as an empty registry
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"corrupt registry file at {registry_path}: {e}"
        ) from e


def _write_registry(entry: dict, registry_path: str, key: str) -> None:
    """
    Write an entry to the papers registry using filelock + atomic os.replace() (D-07, REG-04).

    Acquires a .lock file to serialize concurrent writers, reads the current
    registry (or starts from {}), sets registry[key] = entry, writes to a .tmp
    sibling, then atomically replaces the registry file. The .tmp is always
    cleaned up (whether or not the replace succeeds).

    Args:
        entry:         The registry entry dict to write.
        registry_path: Absolute path to the registry JSON file.
        key:           The registry key (DOI, arXiv ID, or sha256:<prefix>).
    """
    registry_path = os.path.expanduser(registry_path)
    path = pathlib.Path(registry_path)
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = registry_path + ".lock"
    tmp_path = registry_path + ".tmp"

    with filelock.FileLock(lock_path):
        # Read current registry (or start empty); _read_registry handles missing/empty files
        registry = _read_registry(registry_path)

        # Union-merge projects[] from existing entry before overwrite (CR-01, REG-04).
        # A force re-ingest or cross-project re-registration must preserve prior projects.
        existing = registry.get(key)
        if existing and isinstance(existing.get("projects"), list):
            merged = list(dict.fromkeys(
                existing["projects"] + entry.get("projects", [])
            ))
            entry["projects"] = merged

        registry[key] = entry

        # Write to tmp then atomically replace
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, registry_path)


def _write_paperjson_cache(paperjson: dict, stem: str, cache_dir: str = ".paperjson_cache") -> str:
    """
    Write the full PaperJSON to a cache file using atomic os.replace (D-06).

    Mirrors _write_registry's durability pattern: ensure dir exists, write to a
    temp file, then atomically replace. The cache file survives across runs
    (not auto-deleted) for crash recovery and Phase 5 consumption.

    Args:
        paperjson:  Full PaperJSON v2 dict.
        stem:       PDF filename stem (pathlib.Path(pdf).stem).
        cache_dir:  Directory for cache files (default: .paperjson_cache).

    Returns:
        Absolute path to the written cache file.
    """
    cache_path = pathlib.Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    target = cache_path / f"{stem}.json"
    tmp_file = str(target) + ".tmp"

    lock_path = str(target) + ".lock"
    with filelock.FileLock(lock_path):
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(paperjson, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, str(target))

    return str(target.resolve())


def _check_registry(key: str, registry_path: str) -> dict | None:
    """
    Return the cached registry entry for key, or None if not present.

    Args:
        key:           Registry key (DOI, arXiv ID, or sha256:<prefix>).
        registry_path: Absolute path to the registry JSON file.

    Returns:
        The cached entry dict, or None if not found.
    """
    registry = _read_registry(registry_path)
    return registry.get(key)


def _registry_entry(
    paperjson: dict,
    source_path: str,
    paperjson_path: str,
    project_name: str,
) -> dict:
    """
    Build an extraction-only registry entry from a PaperJSON v2 document (D-23).

    Copies title/authors/year/journal/doi/arxiv_id from the extraction metadata.
    Sets summary and key_findings to None (Phase 2 backfills these).
    Sets projects to [project_name].

    Args:
        paperjson:      Full PaperJSON v2 dict.
        source_path:    Absolute path to the source PDF.
        paperjson_path: Absolute path to the stored PaperJSON file.
        project_name:   Project identifier from config.json.

    Returns:
        Registry entry dict with the D-23 key set.
    """
    meta = paperjson.get("extraction", {}).get("metadata", {})
    return {
        "title": meta.get("title"),
        "authors": meta.get("authors"),
        "year": meta.get("year"),
        "journal": meta.get("journal"),
        "doi": meta.get("doi"),
        "arxiv_id": meta.get("arxiv_id"),
        "projects": [project_name],
        "source_path": source_path,
        "paperjson_path": paperjson_path,
        "summary": None,
        "key_findings": None,
    }


# ---------------------------------------------------------------------------
# Phase 3 — web cache stem + web provenance
# ---------------------------------------------------------------------------

def _web_cache_stem(url: str) -> str:
    """
    Derive a cache-file stem for a web-ingested paper (D-03).

    Priority: arXiv ID from URL (human-readable, stable) → SHA-256 of normalized
    URL (generic fallback). The stem is passed to _write_paperjson_cache as `stem`
    and must not contain path separators or a .json suffix.
    """
    m = re.search(r"arxiv\.org/(?:abs|html|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", url)
    if m:
        return f"arxiv-{m.group(1)}"
    m = re.search(r"ar5iv\.labs\.arxiv\.org/html/(\d{4}\.\d{4,5}(?:v\d+)?)", url)
    if m:
        return f"arxiv-{m.group(1)}"
    normalized = url.lower().rstrip("/")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"url-{digest[:16]}"


def _web_provenance(url: str, fetched_url: str, md_text: str) -> dict:
    """
    Build the provenance dict for a web-ingested paper (analog of the PDF provenance dict).

    Omits pdf_sha256 / source_filename / mineru_version — nothing downstream reads
    those provenance keys at runtime (verified in 03-RESEARCH.md).
    """
    return {
        "source_url": url,
        "fetched_url": fetched_url,
        "content_sha256": hashlib.sha256(md_text.encode("utf-8")).hexdigest(),
        "backend": "defuddle",
        "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "normalizations_applied": ["llm_fill"],
        "schema_version": SCHEMA_VERSION,
    }


# ---------------------------------------------------------------------------
# Phase 3 — shared fill cascade (PDF + web converge here)
# ---------------------------------------------------------------------------

def _run_fill_cascade(
    skeleton: dict,
    parsed: dict,
    config: dict,
    *,
    full_text: str,
    first_page_text: str,
    source_path: str,
    cache_stem: str,
    registry_path: str,
    project_name: str,
    force_extract: bool = False,
    refill: bool = False,
    source_url: str | None = None,
) -> dict:
    """
    Run the shared fill-cascade tail (steps 5–13) for both PDF and web ingestion.

    This helper encapsulates the warmup → DOI probe → registry gate → metadata fill
    → section fill → ref fill → cache write → registry write → note generation path.
    Both ingest() and ingest_url() call it after producing their respective skeleton
    and parsed objects.

    Parameters that differ between PDF and web:
      full_text:       Full document text for DOI probe (blocks-extracted for PDF,
                       md_text for web).
      first_page_text: Cover-page text for metadata fill (first-page blocks for PDF,
                       md_text[:3000] for web).
      source_path:     Absolute PDF path string (PDF) or original URL (web) — used
                       as the informational source_path field in the registry entry.
      cache_stem:      pdf.stem for PDF; _web_cache_stem(url) for web.
      registry_path:   Expanded registry JSON path.
      project_name:    Project identifier from config.json.
      force_extract:   PDF only — bypass registry cache even on a hit.
      refill:          PDF only — bypass registry cache for re-fill runs.
      source_url:      Web only — original URL passed to _web_doi_key_fallback (D-07)
                       when the DOI probe yields neither a DOI nor an arXiv ID.
                       Defaults to None; PDF callers omit it, preserving identical behaviour.
    """
    # Step 5: Warm up Ollama to pin gemma4:e4b in VRAM for the full run (D-00d / Pattern 4)
    _log(f"warming up {OLLAMA_MODEL}")
    _warmup_ollama()

    # Step 6: DOI probe — LLM call on FULL document text; title hint is at the start.
    # full_text feeds the DOI probe (may include Supporting Information DOI on later pages);
    # first_page_text feeds _fill_metadata (cover-page metadata is more reliable for title/authors).
    _log("doi probe")
    try:
        probe = _doi_probe(full_text)
    except RuntimeError as e:
        print(f"[ingest error: {e}]", file=sys.stderr)
        sys.exit(1)

    doi = probe.doi if probe else None
    arxiv_id = probe.arxiv_id if probe else None
    title_hint = probe.title if probe else None

    # Step 7: Syntactic DOI validation — never use a malformed DOI as registry key (D-00c)
    if doi and not _syntactic_doi_valid(doi):
        doi = None  # fall back to arXiv ID or title-hash key chain

    # Step 7b: D-07 URL-pattern last-resort key (web path only)
    # Fires when the probe returned neither a DOI nor an arXiv ID AND source_url was provided.
    # Parses the URL itself for a recognisable arXiv id or doi.org DOI so the chain never
    # collapses straight to a title-hash key for a URL whose page did not print an id in-text.
    # PDF callers pass source_url=None (default) → this block is completely skipped.
    if not doi and not arxiv_id and source_url:
        doi, arxiv_id = _web_doi_key_fallback(source_url, probe)
        # Re-apply syntactic guard: a URL-mined DOI that fails the check must not become the key
        if doi and not _syntactic_doi_valid(doi):
            doi = None
        _log(
            f"url-pattern key fallback (D-07): doi={doi!r} arxiv_id={arxiv_id!r}"
        )

    # Step 8: Crossref validation hook (Plan 03)
    if config.get("crossref_validate", False) and doi:
        _crossref_validate(doi, title_hint, config)

    # Step 9: Registry check — HARD GATE (D-00b / REG-02)
    # No fill helpers run if the paper is already in the registry.
    # BYPASSED when refill=True — the whole point of --refill is to re-run the fill
    # even on a cache hit (Plan 11, INGEST-01).
    # CR-01 (SC3): when the probe yielded no title (a web probe failure on a generic
    # journal URL with neither a DOI nor an arXiv id), fall back to the parsed H1 title in
    # the skeleton so the key does not collapse to the constant empty-string hash
    # (sha256:e3b0c44298fc1c14) shared by every such paper. source_url-gated → web only;
    # the PDF path (source_url=None) keeps title_hint and stays byte-identical.
    key_title = title_hint
    if source_url and not key_title:
        key_title = skeleton.get("extraction", {}).get("metadata", {}).get("title")
    registry_key = _registry_key({"doi": doi, "arxiv_id": arxiv_id, "title": key_title})
    cached = _check_registry(registry_key, registry_path)
    if cached is not None and not force_extract and not refill:
        # Cache-hit append: add current project to cached entry if absent (CR-01, REG-02).
        # Persists via _write_registry (which union-merges) under the lock.
        if project_name and project_name not in cached.get("projects", []):
            cached.setdefault("projects", []).append(project_name)
            if registry_path:
                try:
                    _write_registry(cached, registry_path, registry_key)
                except Exception as e:
                    _log(f"registry cache-hit update failed: {e}")
        # GAP-CLOSURE (02-04): note-aware registry hit. If a surviving PaperJSON cache
        # exists for this already-ingested paper, generate the note from it best-effort.
        # generate_note self-skips when the note already exists (force=False, D-16), so this
        # is idempotent and runs zero LLM fill calls. A note failure never fails the ingest.
        pj_path = cached.get("paperjson_path") or ""
        if pj_path and pathlib.Path(pj_path).exists():
            try:
                from scripts import note as note
                with open(pj_path, "r", encoding="utf-8") as _pjf:
                    _cached_paperjson = json.load(_pjf)
                _note_result = note.generate_note(_cached_paperjson, config)
                if isinstance(_note_result, str) and _note_result.startswith("[note error:"):
                    _log(f"registry-hit note generation failed: {_note_result}")

                # Step 9b: Bibliography linking on the cache-hit path (D-09 — non-fatal
                # tail stage; gap-closure 04-06 / closes Gap B). Mirrors the Step 12c
                # miss-path hook so a registry-known paper whose note is (re)generated
                # here also gets its ## References section linked and stub-upgrade
                # detection run. Reuses _cached_paperjson (already loaded for note
                # regen) as the skeleton arg — never re-reads the cache file.
                try:
                    from scripts import biblio as biblio_mod
                    _biblio_result = biblio_mod.run_biblio(_cached_paperjson, config)
                    if isinstance(_biblio_result, str) and _biblio_result.startswith("[biblio warning:"):
                        _log(f"registry-hit bibliography linking: {_biblio_result}")
                except Exception as e:
                    print(
                        f"[biblio warning: registry-hit bibliography linking failed: {e}]",
                        file=sys.stderr,
                    )
            except Exception as e:
                print(f"[ingest warning: registry-hit note generation failed: {e}]", file=sys.stderr)
        return cached  # cache hit: return immediately, zero fill calls

    # Steps 10–12 (miss path): fill → renditions → registry write
    failed_count = 0

    # Step 10a: Metadata fill (one call post-miss)
    _log("metadata fill")
    try:
        metadata = _fill_metadata(first_page_text, probe)
    except RuntimeError as e:
        print(f"[ingest error: {e}]", file=sys.stderr)
        sys.exit(1)

    # Step 10a+: Crossref journal_full enrichment (Plan 08 GAP B).
    if config.get("crossref_validate", False) and doi and metadata.journal_full is None:
        full = _crossref_journal_full(doi, config)
        if full is not None:
            metadata.journal_full = full

    # Step 10a++: Crossref published-year enrichment (Plan 10 Item 1b, INGEST-01).
    if config.get("crossref_validate", False) and doi and metadata.year is None:
        crossref_year = _crossref_published_year(doi, config)
        if crossref_year is not None:
            metadata.year = crossref_year

    # Step 10a+++: journal-abbreviation fallback (Plan 09 GAP C).
    if not metadata.journal and metadata.journal_full:
        metadata.journal = metadata.journal_full

    skeleton["extraction"]["metadata"] = metadata.model_dump()

    # Step 10b: Per-section fill (one call per section, two-strike, D-01)
    total = len(skeleton["extraction"]["sections"])
    kept_sections: list[dict] = []
    kept_count = 0
    dropped_count = 0
    try:
        for i, section in enumerate(skeleton["extraction"]["sections"]):
            raw_parts = []
            for blk in section.get("blocks", []):
                if blk.get("type") == "text":
                    raw_parts.append(blk.get("display") or blk.get("plain") or "")
            raw_section_text = "\n".join(raw_parts)
            log_label = section.get("heading") or f"Section {i}"
            raw_heading = section.get("heading") or ""
            _log(f"section fill {i + 1}/{total}: '{log_label}'")
            fill_result = _fill_section(raw_section_text, raw_heading)
            out_heading = fill_result.heading or f"Section {i}"
            if fill_result.fill_failed:
                failed_count += 1
                kept_count += 1
                kept_sections.append({
                    "heading": out_heading,
                    "body": fill_result.body,
                    "fill_failed": fill_result.fill_failed,
                })
            elif fill_result.keep:
                kept_count += 1
                kept_sections.append({
                    "heading": out_heading,
                    "body": fill_result.body,
                    "fill_failed": fill_result.fill_failed,
                })
            else:
                dropped_count += 1
    except RuntimeError as e:
        print(f"[ingest error: {e}]", file=sys.stderr)
        sys.exit(1)
    # WR-01 body-less-fill floor (Plan 11)
    if total > 0 and not kept_sections:
        print(
            f"[ingest warning: all {total} sections judged non-substantive — "
            f"retaining unfiltered (body-less floor)]",
            file=sys.stderr,
        )
        kept_sections = []
        for i, section in enumerate(skeleton["extraction"]["sections"]):
            raw_parts = []
            for blk in section.get("blocks", []):
                if blk.get("type") == "text":
                    raw_parts.append(blk.get("display") or blk.get("plain") or "")
            raw_section_text = "\n".join(raw_parts)
            kept_sections.append({
                "heading": section.get("heading") or f"Section {i}",
                "body": raw_section_text,
                "fill_failed": True,
            })
        kept_count = len(kept_sections)
        dropped_count = 0

    skeleton["extraction"]["sections"] = kept_sections
    _log(f"section filter: kept {kept_count}, dropped {dropped_count}")

    # Step 10c: Ref batch fill (~10 per batch, D-02)
    try:
        refs, ref_failures = _fill_references_batched(parsed.get("references", []))
    except RuntimeError as e:
        print(f"[ingest error: {e}]", file=sys.stderr)
        sys.exit(1)
    skeleton["extraction"]["references"] = refs
    failed_count += ref_failures

    # Step 10d: Opt-in Crossref title->DOI enrichment (Plan 05 Layer 3).
    # Runs AFTER ref fill so LLM-filled titles are available, and BEFORE the
    # cache write so enriched DOIs are persisted — a later cache-hit re-run
    # reads the cached DOIs and never re-hits the network.
    # No-op when crossref_validate=False (offline default) or email is placeholder.
    _enrich_references_with_crossref(skeleton, config)

    # Step 11: Renditions — body IS the cleaned text from LLM fill (D-10)
    _log("renditions")

    # Step 11b: PaperJSON cache write (D-06 — surviving, gitignored)
    _log("cache write")
    cache_path = _write_paperjson_cache(skeleton, cache_stem)

    # Step 12: Registry write (filelock + atomic, REG-01 / REG-04)
    if registry_path:
        _log("registry write")
        entry = _registry_entry(skeleton, source_path, cache_path, project_name)
        try:
            _write_registry(entry, registry_path, registry_key)
        except Exception as e:
            print(f"[ingest warning: registry write failed: {e}]", file=sys.stderr)

    # Step 12b: Auto-invoke note generation (D-05/D-07 — best-effort)
    try:
        from scripts import note as note
        note_result = note.generate_note(skeleton, config)
        if isinstance(note_result, str) and (
            note_result.startswith("[note error:") or note_result.startswith("[note warning:")
        ):
            _log(f"note generation failed: {note_result}")
    except Exception as e:
        print(f"[ingest warning: note generation failed: {e}]", file=sys.stderr)

    # Step 12c: Bibliography linking (D-09 — non-fatal tail stage; Phase 4)
    try:
        from scripts import biblio as biblio_mod
        biblio_result = biblio_mod.run_biblio(skeleton, config)
        if isinstance(biblio_result, str) and biblio_result.startswith("[biblio warning:"):
            _log(f"bibliography linking: {biblio_result}")
    except Exception as e:
        print(f"[biblio warning: bibliography linking failed: {e}]", file=sys.stderr)

    # Step 13: Warn on partial fill, return skeleton
    if failed_count > 0:
        print(f"[ingest warning: {failed_count} sections/batches unfilled]", file=sys.stderr)

    return skeleton


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest(pdf_path: str, config: dict, force_extract: bool = False, refill: bool = False) -> dict:
    """
    Run the Phase 1.3 probe → quality-gate → registry-gate → fill cascade for one PDF.

    Ordering (D-12, D-00b, D-03, D-04):
      1. Preflight: PDF exists + MinerU resolves (fail-fast)
      1a. Refill guard: if refill=True, verify existing MinerU output (error if absent)
      2. Run-or-reuse MinerU → find + load content_list.json
      3. Parse content_list → build provenance → assemble minimal skeleton
      4. Quality gate on skeleton (D-12: garbage PDFs fail before any LLM call)
      5. Warm-up Ollama (keep_alive="-1" to pin model in VRAM)
      6. DOI probe on first-page + footer blocks (cheap pre-gate LLM call)
      7. Syntactic DOI validation (D-00c)
      8. Crossref validation hook (Plan 03)
      9. Registry check — HARD GATE: cache hit returns immediately, no fill calls
         (BYPASSED when refill=True so the fill re-runs)
     10. Fill cascade (miss path): metadata → per-section → ref batches
     11. Renditions on LLM-cleaned text (D-10)
     12. Registry write (atomic, filelock) — refill uses merge-write (Plan 10 union-merge)
     13. Warn on partial fill; return skeleton

    Args:
        pdf_path:      Absolute or relative path to the input PDF.
        config:        Loaded config.json dict (use _load_config()).
        force_extract: If True, re-run MinerU and re-register even if cached.
        refill:        If True, re-run LLM fill reusing existing MinerU output.
                       Errors if no prior content_list.json exists (no silent GPU fallback).
                       Bypasses registry cache return so fill re-runs. (Plan 11, INGEST-01)

    Returns:
        PaperJSON v2 dict (new ingest) OR cached registry entry dict (cache hit).

    Raises:
        SystemExit: On preflight failure or quality-gate failure.
    """
    pdf = pathlib.Path(pdf_path).resolve()

    # Step 1: Preflight — PDF exists + MinerU resolves
    if not pdf.exists():
        print(f"[ingest error: file not found: {pdf_path}]", file=sys.stderr)
        sys.exit(1)

    mineru_exe = _resolve_mineru(config)
    if mineru_exe is None:
        print(
            "[ingest error: mineru not found — set mineru_path in config.json or add to PATH]",
            file=sys.stderr,
        )
        sys.exit(1)

    global _NUM_CTX_CAP
    _NUM_CTX_CAP = int(config.get("ollama_num_ctx_cap", DEFAULT_NUM_CTX_CAP))
    global _SECTION_TIMEOUT
    _SECTION_TIMEOUT = int(config.get("ollama_section_timeout", DEFAULT_SECTION_TIMEOUT))

    out_dir = str(pathlib.Path(".mineru_output") / pdf.stem)
    timeout = int(config.get("mineru_timeout", DEFAULT_TIMEOUT))
    registry_path = os.path.expanduser(config.get("registry_path", ""))
    project_name = config.get("project_name", "")

    # Step 1a: Refill guard — verify existing MinerU output BEFORE running (Plan 11).
    # --refill reuses existing content_list.json; if none exists, error clearly
    # instead of silently falling back to a full MinerU GPU run.
    if refill:
        existing_cl = _find_content_list_path(out_dir)
        if existing_cl is None:
            print(
                f"[ingest error: --refill requires existing MinerU output for {pdf.name} "
                f"— none found at {out_dir}; run without --refill first]",
                file=sys.stderr,
            )
            sys.exit(1)

    # Step 2: Run-or-reuse MinerU → find + load content_list.json
    # When refill=True, force_extract is always False (reuse MinerU output, no GPU step).
    _log("mineru extraction starting")
    effective_force_extract = False if refill else force_extract
    try:
        _run_mineru(str(pdf), out_dir, mineru_exe, timeout, effective_force_extract)
    except subprocess.TimeoutExpired:
        print(f"[ingest error: mineru timed out after {timeout}s]", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"[ingest error: {e}]", file=sys.stderr)
        sys.exit(1)
    _log("mineru extraction finished")

    try:
        cl_path = _find_content_list(out_dir)
    except RuntimeError as e:
        print(f"[ingest error: {e}]", file=sys.stderr)
        sys.exit(1)

    try:
        with open(cl_path, encoding="utf-8") as f:
            blocks = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[ingest error: failed to load content_list.json: {e}]", file=sys.stderr)
        sys.exit(1)

    # Step 3: Parse → provenance → minimal skeleton
    parsed = _parse_content_list(blocks)

    pdf_sha256 = hashlib.sha256(pdf.read_bytes()).hexdigest()
    provenance = {
        "pdf_sha256": pdf_sha256,
        "source_filename": pdf.name,
        "mineru_version": None,
        "backend": MINERU_BACKEND,
        "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "normalizations_applied": ["ligature_fix", "llm_fill"],
        "schema_version": SCHEMA_VERSION,
    }

    skeleton = _assemble_paperjson(parsed, provenance)

    # Step 4: Quality gate — runs on MinerU output BEFORE any LLM call (D-12)
    gate_error = _quality_gate(skeleton)
    if gate_error:
        print(gate_error, file=sys.stderr)
        sys.exit(1)

    # Steps 5–13: Shared fill cascade (PDF path — delegates to shared helper).
    # Compute full_text and first_page_text from blocks here; the cascade helper
    # accepts them as parameters so the web path can pass md_text instead.
    full_text = _extract_full_text(blocks)
    first_page_text = _extract_first_page_and_footers(blocks)

    return _run_fill_cascade(
        skeleton,
        parsed,
        config,
        full_text=full_text,
        first_page_text=first_page_text,
        source_path=str(pdf),
        cache_stem=pathlib.Path(pdf).stem,
        registry_path=registry_path,
        project_name=project_name,
        force_extract=force_extract,
        refill=refill,
    )


def ingest_url(url: str, config: dict) -> dict:
    """
    Ingest a paper by URL (arXiv, PubMed, or journal) and return a PaperJSON v2 dict.

    Sibling entry point to ingest() (D-01): swaps MinerU steps 1–3 for a defuddle
    subprocess + heading-split markdown parser, then joins the same fill cascade via
    _run_fill_cascade (the PDF and web paths converge there).

    Step ordering:
      1. Preflight — reject non-http(s) URLs; resolve defuddle; set module globals.
      2. URL rewrite — arXiv /abs→/html; PubMed→PMC (fail-open).
      3. Run defuddle (argv list, timeout-bounded — D-08 / T-03-02 / T-03-04).
      4. Parse markdown → PaperJSON sections skeleton.
      5. Build web provenance dict.
      6. Assemble PaperJSON skeleton via _assemble_paperjson.
      7. Quality gate on skeleton (D-12).
      8–16. Shared fill cascade via _run_fill_cascade.

    D-09 thin-content gate and ar5iv retry land in 03-03.
    _web_doi_key_fallback lands in 03-04.

    Args:
        url:    HTTP(S) URL of the paper to ingest.
        config: Loaded config.json dict (use _load_config()).

    Returns:
        PaperJSON v2 dict (new ingest) OR cached registry entry dict (cache hit).

    Raises:
        SystemExit: On preflight failure or quality-gate failure.
    """
    # Step 1: Preflight
    # Scheme validation (T-03-03: reject file:// / data: / etc.)
    if not url.startswith(("http://", "https://")):
        print(
            f"[ingest error: invalid URL scheme — only http/https allowed: {url}]",
            file=sys.stderr,
        )
        sys.exit(1)

    defuddle_exe = _resolve_defuddle(config)
    if defuddle_exe is None:
        print(
            "[ingest error: defuddle not found — install with: npm install -g defuddle]",
            file=sys.stderr,
        )
        sys.exit(1)

    # Set module globals from config (mirror ingest() lines 1877-1880)
    global _NUM_CTX_CAP
    _NUM_CTX_CAP = int(config.get("ollama_num_ctx_cap", DEFAULT_NUM_CTX_CAP))
    global _SECTION_TIMEOUT
    _SECTION_TIMEOUT = int(config.get("ollama_section_timeout", DEFAULT_SECTION_TIMEOUT))

    registry_path = os.path.expanduser(config.get("registry_path", ""))
    project_name = config.get("project_name", "")
    defuddle_timeout = int(config.get("defuddle_timeout", 60))

    # Step 2: URL rewrite (arXiv /abs→/html; PubMed→PMC fail-open)
    _log(f"url rewrite: {url}")
    rewritten_url = _rewrite_url(url, config)
    if rewritten_url != url:
        _log(f"url rewritten to: {rewritten_url}")

    # Step 3: Run defuddle (T-03-02: argv list; T-03-04: timeout-bounded; D-08: empty gate)
    _log("defuddle extraction starting")
    try:
        md_text = _run_defuddle(rewritten_url, defuddle_exe, timeout=defuddle_timeout)
    except subprocess.TimeoutExpired:
        print(
            f"[ingest error: defuddle timed out after {defuddle_timeout}s]",
            file=sys.stderr,
        )
        sys.exit(1)
    except RuntimeError as e:
        print(f"[ingest error: {e}]", file=sys.stderr)
        sys.exit(1)
    _log("defuddle extraction finished")

    # Step 4: Parse markdown → sections skeleton
    _log("markdown parse")
    parsed = _parse_defuddle_markdown(md_text)

    # Step 5: Build web provenance dict
    provenance = _web_provenance(url, rewritten_url, md_text)

    # Step 6: Assemble PaperJSON skeleton
    skeleton = _assemble_paperjson(parsed, provenance)

    # Step 6b: D-04 ar5iv retry + D-09 body-size gate (03-03)
    min_chars = int(config.get("web_min_body_chars", 2000))

    # D-04: ar5iv retry — fires once when arXiv HTML is thin (older papers lack native HTML)
    if (
        _is_arxiv_url(url)
        and _web_body_too_thin(skeleton, min_chars)
        and "ar5iv.labs.arxiv.org" not in rewritten_url
    ):
        arxiv_id = _extract_arxiv_id(url)
        if arxiv_id:
            ar5iv_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
            _log(f"arxiv html thin — retrying with ar5iv: {ar5iv_url}")
            try:
                md_text = _run_defuddle(ar5iv_url, defuddle_exe, timeout=defuddle_timeout)
            except subprocess.TimeoutExpired:
                print(
                    f"[ingest error: defuddle timed out after {defuddle_timeout}s]",
                    file=sys.stderr,
                )
                sys.exit(1)
            except RuntimeError as e:
                print(f"[ingest error: {e}]", file=sys.stderr)
                sys.exit(1)
            parsed = _parse_defuddle_markdown(md_text)
            provenance = _web_provenance(url, ar5iv_url, md_text)
            skeleton = _assemble_paperjson(parsed, provenance)

    # D-09: Body-size gate — reject paywall / abstract-only pages pre-LLM (T-03-06)
    if _web_body_too_thin(skeleton, min_chars):
        total = sum(
            len(block.get("plain", "") or "")
            for section in skeleton.get("extraction", {}).get("sections", [])
            for block in section.get("blocks", [])
            if block.get("type") == "text"
        )
        print(
            f"[ingest error: web content too short — likely paywalled or abstract-only: "
            f"{url} (body chars: {total} < {min_chars})]",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 7: Quality gate — before any LLM call (D-12)
    gate_error = _quality_gate(skeleton, source="web")
    if gate_error:
        print(gate_error, file=sys.stderr)
        sys.exit(1)

    # Steps 8–16: Shared fill cascade (converges with PDF path here)
    return _run_fill_cascade(
        skeleton,
        parsed,
        config,
        full_text=md_text,
        first_page_text=md_text[:3000],
        source_path=url,
        cache_stem=_web_cache_stem(url),
        registry_path=registry_path,
        project_name=project_name,
        source_url=url,  # D-07: URL-pattern key fallback when probe finds no DOI/arXiv ID
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Windows cp1252 guard: wrap stdout in UTF-8 before any print (D-PATTERNS)
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=(
            "Ingest a research paper PDF or URL via MinerU/defuddle and emit a short "
            "confirmation (cache path + note path) to stdout. "
            "Use --print/--stdout to emit the full PaperJSON v2 to stdout instead, "
            "or use --output/-o to write it to a file."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--pdf",
        help="Path to the input PDF file.",
    )
    source_group.add_argument(
        "--url",
        help="URL of the paper to ingest (arXiv, PubMed, or journal).",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-run MinerU even if output already exists (D-15).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"MinerU subprocess timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Write the final PaperJSON to this path as UTF-8. "
            "Preferred over '>' on Windows — PowerShell '>' re-decodes correct UTF-8 "
            "through the OEM console code page and rewrites as UTF-16LE with mojibake "
            "(José → 'Jos├⌐', U+2019 → 'ΓÇÖ'). Use -o to bypass redirection entirely."
        ),
    )
    parser.add_argument(
        "--refill",
        action="store_true",
        help=(
            "Re-run the LLM fill reusing existing MinerU output (no GPU extraction step). "
            "Errors if no prior content_list.json exists for this PDF — run without --refill first. "
            "Bypasses the registry cache-hit gate so the fill cascade re-runs. "
            "Registry write uses merge (Plan 10 union-merge) — does not clobber other projects."
        ),
    )
    parser.add_argument(
        "--print", "--stdout",
        dest="print_json",
        action="store_true",
        help=(
            "Emit the full PaperJSON v2 to stdout as UTF-8 JSON. "
            "By default only a short confirmation (cache path + note path) is printed; "
            "use this flag to restore the full JSON dump."
        ),
    )

    args = parser.parse_args()

    try:
        config = _load_config()

        # Dispatch: --pdf or --url (mutually exclusive, one is required)
        if args.pdf:
            # Allow CLI --timeout to override config (PDF only)
            if args.timeout != DEFAULT_TIMEOUT:
                config["mineru_timeout"] = args.timeout
            result = ingest(args.pdf, config, force_extract=args.force_extract, refill=args.refill)
        elif args.url:
            result = ingest_url(args.url, config)

        # Build a short confirmation: cache path + written note path.
        # Used by the default (non --print) stdout branch; never aborts the run on failure.
        confirmation = None
        try:
            from scripts import note as _note_mod
            title = (
                result.get("extraction", {}).get("metadata", {}).get("title")
                or result.get("title")
                or "Untitled"
            )
            safe_name = _note_mod._sanitize_filename(title)
            note_path = f"Papers/{safe_name}.md"
            # Cache path: derive from pdf stem (PDF) or web cache stem (URL)
            if args.pdf:
                cache_path = result.get("paperjson_path") or str(
                    pathlib.Path(".paperjson_cache", pathlib.Path(args.pdf).stem + ".json").resolve()
                )
            else:
                cache_path = result.get("paperjson_path") or str(
                    pathlib.Path(".paperjson_cache", _web_cache_stem(args.url) + ".json").resolve()
                )
            confirmation = f"Cache: {cache_path}\nNote:  {note_path}"
        except Exception:
            pass  # fallback to default "[ingest] done." line

        _emit_result(result, args.output, print_json=args.print_json, confirmation=confirmation)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[ingest error: {e}]", file=sys.stderr)
        sys.exit(1)
