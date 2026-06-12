"""
MinerU-based PDF ingestion pipeline.

Invokes MinerU (hybrid_auto backend), parses the resulting content_list.json by block
type, and assembles a PaperJSON v2 document (extraction/analysis/provenance namespaces,
schema_version 2) that is printed to stdout as UTF-8 JSON.

This module is extraction-only: no Ollama or LLM calls.
The analysis namespace ships as an empty skeleton; Phase 2 populates it.
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import filelock
import urllib.request
import urllib.error
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
    journal: str | None = None
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None


class SectionFillResult(BaseModel):
    heading: str
    body: str
    fill_failed: bool = False


class RefEntry(BaseModel):
    number: int | None = None
    raw: str
    doi: str | None = None
    title: str | None = None
    year: int | None = None
    fill_failed: bool = False


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
    format, options.temperature=0, options.num_ctx, and CRITICALLY omits think key
    (Ollama #15260 drops format if think=false).
    """
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
        "format": schema,
        "options": {
            "temperature": 0,
            "num_ctx": num_ctx,
        },
        # CRITICAL: omit "think" key entirely (Ollama #15260 drops format if think=false)
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
    except urllib.error.URLError as e:
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


def _estimate_num_ctx(text: str, overhead: int = 2048) -> int:
    """
    Approximate num_ctx from input length (~4 chars/token).

    Rounds up to nearest power of two; hard cap 16384 (RTX 4060 8GB VRAM safety).
    """
    estimated_tokens = len(text) // 4
    raw = estimated_tokens + overhead
    for ctx in (2048, 4096, 8192, 16384):
        if raw <= ctx:
            return ctx
    return 16384


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
    except urllib.error.URLError:
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
# Quality gate
# ---------------------------------------------------------------------------

def _quality_gate(paperjson: dict) -> str | None:
    """
    Validate extraction output quality (INGEST-03 redefined per D-11).

    Returns an [ingest error: ...]-prefixed message when the result is garbage:
      - no title detected (metadata.title is None or empty), OR
      - total non-noise text is near-empty (below threshold).

    Returns None when output passes quality checks.
    """
    extraction = paperjson.get("extraction", {})
    metadata = extraction.get("metadata", {})
    title = metadata.get("title")

    if not title:
        return (
            "[ingest error: extraction produced no usable content — "
            "possible scanned/garbage PDF (no title detected)]"
        )

    # Count total text length across all text blocks
    total_text_len = 0
    for section in extraction.get("sections", []):
        for block in section.get("blocks", []):
            if block.get("type") == "text":
                total_text_len += len(block.get("plain", "") or block.get("display", ""))

    # Near-empty threshold: 100 chars of non-noise text is a reasonable minimum
    # for a real paper (even an abstract is ~500 chars)
    NEAR_EMPTY_THRESHOLD = 100
    if total_text_len < NEAR_EMPTY_THRESHOLD:
        return (
            f"[ingest error: extraction produced no usable content — "
            f"possible scanned/garbage PDF (total text {total_text_len} chars "
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

        registry[key] = entry

        # Write to tmp then atomically replace
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, registry_path)


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
# Public API
# ---------------------------------------------------------------------------

def ingest(pdf_path: str, config: dict, force_extract: bool = False) -> dict:
    """
    Run the full MinerU → content_list.json → PaperJSON v2 pipeline for one PDF.

    Fail-fast preflight (D-10): verifies PDF exists and MinerU resolves before
    launching the subprocess. Returns the assembled PaperJSON v2 dict on success.

    Registry dedup (REG-02): derives the registry key (DOI → arXiv → title-hash)
    from parsed metadata and returns the cached entry without running MinerU when
    the paper is already registered (unless force_extract=True).

    Args:
        pdf_path:      Absolute or relative path to the input PDF.
        config:        Loaded config.json dict (use _load_config()).
        force_extract: If True, re-run MinerU and re-register even if cached (D-15).

    Returns:
        PaperJSON v2 dict (new ingest) OR cached registry entry dict (cache hit).

    Raises:
        SystemExit: On preflight failure, emits [ingest error: ...] and exits non-zero.
    """
    pdf = pathlib.Path(pdf_path).resolve()

    # D-10: fail-fast preflight
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

    # D-13: persistent output dir per document stem
    out_dir = str(pathlib.Path(".mineru_output") / pdf.stem)
    timeout = int(config.get("mineru_timeout", DEFAULT_TIMEOUT))
    registry_path = os.path.expanduser(config.get("registry_path", ""))
    project_name = config.get("project_name", "")

    # ---------------------------------------------------------------------------
    # REG-02 dedup: if existing MinerU output is present and force_extract is off,
    # parse it first to derive the registry key without re-running the GPU step.
    # On a cache hit, return the cached entry immediately (skips _run_mineru).
    # ---------------------------------------------------------------------------
    if not force_extract:
        try:
            existing_cl = _find_content_list(out_dir)
            # Fast path: parse existing output to derive registry key (no GPU re-run)
            with open(existing_cl, encoding="utf-8") as f:
                existing_blocks = json.load(f)
            fast_parsed = _parse_content_list(existing_blocks)
            fast_metadata = fast_parsed.get("metadata", {})
            reg_key = _registry_key(fast_metadata)
            cached = _check_registry(reg_key, registry_path)
            if cached is not None:
                # Cache hit: return the cached registry entry (REG-02 satisfied)
                return cached
            # Cache miss: fall through to _run_mineru (which will reuse existing output)
        except RuntimeError:
            pass  # No existing output — proceed with _run_mineru (GPU step required)
        except (OSError, json.JSONDecodeError):
            pass  # Corrupt output — fall through to full re-run
    # ---------------------------------------------------------------------------

    # Run or reuse MinerU (GPU step — skipped on cache hit above)
    try:
        _run_mineru(str(pdf), out_dir, mineru_exe, timeout, force_extract)
    except subprocess.TimeoutExpired:
        # T-01.2-02: emit bracketed error on timeout
        print(f"[ingest error: mineru timed out after {timeout}s]", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"[ingest error: {e}]", file=sys.stderr)
        sys.exit(1)

    # Find and load content_list.json
    try:
        cl_path = _find_content_list(out_dir)
    except RuntimeError as e:
        print(f"[ingest error: {e}]", file=sys.stderr)
        sys.exit(1)

    try:
        with open(cl_path, encoding="utf-8") as f:
            blocks = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        # T-01.2-04: bracketed error on parse failure
        print(f"[ingest error: failed to load content_list.json: {e}]", file=sys.stderr)
        sys.exit(1)

    # Parse and assemble
    parsed = _parse_content_list(blocks)

    # Build provenance
    pdf_sha256 = hashlib.sha256(pdf.read_bytes()).hexdigest()
    provenance = {
        "pdf_sha256": pdf_sha256,
        "source_filename": pdf.name,
        "mineru_version": None,     # resolved from MinerU output in Plan 02
        "backend": MINERU_BACKEND,
        "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "normalizations_applied": ["ligature_fix", "llm_fill"],
        "schema_version": SCHEMA_VERSION,
    }

    result = _assemble_paperjson(parsed, provenance)

    # D-11: INGEST-03 garbage-output quality gate
    gate_error = _quality_gate(result)
    if gate_error:
        print(gate_error, file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # REG-01: write extraction-only registry entry for new (or force-re-ingested) paper
    # ---------------------------------------------------------------------------
    if registry_path:
        meta = result.get("extraction", {}).get("metadata", {})
        reg_key = _registry_key(meta)
        entry = _registry_entry(result, str(pdf), "", project_name)
        try:
            _write_registry(entry, registry_path, reg_key)
        except Exception as e:
            # Registry write failure is non-fatal — log and continue
            print(f"[ingest warning: registry write failed: {e}]", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Windows cp1252 guard: wrap stdout in UTF-8 before any print (D-PATTERNS)
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Ingest a research paper PDF via MinerU and emit PaperJSON v2 to stdout."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file.")
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

    args = parser.parse_args()

    try:
        config = _load_config()
        # Allow CLI --timeout to override config
        if args.timeout != DEFAULT_TIMEOUT:
            config["mineru_timeout"] = args.timeout
        result = ingest(args.pdf, config, force_extract=args.force_extract)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except SystemExit:
        raise
    except Exception as e:
        print(f"[ingest error: {e}]", file=sys.stderr)
        sys.exit(1)
