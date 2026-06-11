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
import pathlib
import re
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2
DEFAULT_TIMEOUT = 1800          # 30 minutes — covers long papers + first model download
MINERU_BACKEND = "hybrid_auto"
NOISE_BLOCK_TYPES = {"footer", "page_number", "aside_text", "header"}


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
    Apply P0/P1 normalization fixes (MINERU.md §3).

    P0 (mandatory): fi/fl/ff/ffi/ffl ligature superscript misread.
    P1: U+FFFD replacement character → dash.
    P1: charge sign (halide superscript-2 → superscript-minus).
    """
    # P0 — ligature superscript fix (95+ occurrences per paper)
    text = re.sub(r"<sup>\s*(fi|fl|ff|ffi|ffl)\s*</sup>", r"\1", text)
    # P1 — U+FFFD replacement character (em-dash/corruption)
    text = text.replace("�", "-")
    # P1 — charge sign misread (halide token + superscript-2 → superscript-minus)
    text = re.sub(r"\b(Cl|Br|I|F)<sup>2</sup>", r"\1<sup>−</sup>", text)
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
    Parse a content_list.json block list into structured sections and references.

    Routes each block by type (D-01/D-02/D-06). Returns a dict with:
      - sections: list of {heading, level, blocks[]}
      - references: list of {raw, ...}
      - title: first text_level-1 block text on page 0 (or None)
    """
    sections = []
    references = []
    title = None

    # Start with a default section to hold content before the first heading
    current_section = {"heading": "", "level": 0, "blocks": []}

    for block in blocks:
        btype = block.get("type", "")
        text_level = block.get("text_level")
        page_idx = block.get("page_idx", -1)

        # Extract title: first text_level=1 block on page 0
        if btype == "text" and text_level == 1 and page_idx == 0 and title is None:
            raw_title = block.get("text", "")
            title = _build_plain(raw_title)
            # Also add as a block in the section for completeness
            _route_block(block, current_section, references)
            continue

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

        # Route all other blocks
        _route_block(block, current_section, references)

    # Flush final section
    if current_section["blocks"]:
        sections.append(current_section)

    return {
        "title": title,
        "sections": sections,
        "references": references,
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
    extraction = {
        "metadata": {
            "title": parsed.get("title"),
            "authors": None,            # full mining in Plan 02
            "year": None,
            "journal": None,
            "doi": None,
            "arxiv_id": None,
            "accession_codes": [],
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
# Public API
# ---------------------------------------------------------------------------

def ingest(pdf_path: str, config: dict, force_extract: bool = False) -> dict:
    """
    Run the full MinerU → content_list.json → PaperJSON v2 pipeline for one PDF.

    Fail-fast preflight (D-10): verifies PDF exists and MinerU resolves before
    launching the subprocess. Returns the assembled PaperJSON v2 dict on success.

    Args:
        pdf_path:      Absolute or relative path to the input PDF.
        config:        Loaded config.json dict (use _load_config()).
        force_extract: If True, re-run MinerU even if output already exists (D-15).

    Returns:
        PaperJSON v2 dict with extraction, analysis, and provenance namespaces.

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

    # Run or reuse MinerU
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
        "normalizations_applied": ["ligature_fix", "ufffd_replacement", "charge_sign_fix"],
        "schema_version": SCHEMA_VERSION,
    }

    return _assemble_paperjson(parsed, provenance)


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
