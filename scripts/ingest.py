"""
CLI script: extract structured PaperJSON from a PDF and write to the global registry.
Usage: python scripts/ingest.py --pdf <absolute_path_to_pdf>
"""

# ---------------------------------------------------------------------------
# stdlib
# ---------------------------------------------------------------------------
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter

# ---------------------------------------------------------------------------
# third-party
# ---------------------------------------------------------------------------
import filelock
import pdfplumber
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
DEFAULT_REGISTRY_KEY_PREFIX_LEN = 16  # hex chars of SHA-256 title hash
CONFIG_FILENAME = "config.json"
SCANNED_CHAR_THRESHOLD = 100
TWO_COL_RIGHT_RATIO = 0.30
_TITLE_SENTINELS = frozenset({"untitled", "unknown", "no title", ""})

# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _fail(msg: str) -> None:
    """Print error to stderr and exit non-zero."""
    print(f"[ingest error: {msg}]", file=sys.stderr)
    sys.exit(1)


def _load_config(config_path: str) -> dict:
    """Load config.json from the given path.

    Expands ~ in registry_path and vault_path. Defaults project_name to the
    basename of the directory containing config.json (Open Question 3 resolution).
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        _fail(f"cannot read config.json: {e}")

    # Expand ~ in path fields
    if "registry_path" in config:
        config["registry_path"] = os.path.expanduser(config["registry_path"])
    if "vault_path" in config:
        config["vault_path"] = os.path.expanduser(config["vault_path"])

    # Default project_name to the config's parent-dir basename
    if not config.get("project_name"):
        config["project_name"] = os.path.basename(os.path.dirname(os.path.abspath(config_path)))

    return config


def _find_config() -> str:
    """Walk up directory tree from this script to find config.json at repo root.

    Starts at the directory containing this script and walks up parent directories
    until CONFIG_FILENAME is found. Calls _fail if no config found by filesystem root.
    """
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        candidate = os.path.join(current, CONFIG_FILENAME)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            # Reached filesystem root without finding config.json
            _fail("config.json not found in repo tree")
        current = parent


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for registry key generation."""
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def _compute_registry_key(paper: dict) -> str:
    """Return DOI, arXiv ID, or SHA-256 title hash as the registry key (D-12).

    Priority: DOI -> arXiv ID -> SHA-256 of normalized title (first 16 hex chars).
    """
    if paper.get("doi"):
        return paper["doi"]
    if paper.get("arxiv_id"):
        return f"arxiv:{paper['arxiv_id']}"
    title = paper.get("title") or ""
    hex_prefix = hashlib.sha256(
        _normalize_title(title).encode()
    ).hexdigest()[:DEFAULT_REGISTRY_KEY_PREFIX_LEN]
    return f"sha256:{hex_prefix}"


def _is_scanned(pdf_path: str, threshold: int = SCANNED_CHAR_THRESHOLD) -> bool:
    """Return True if PDF appears to be image-only (total chars < threshold).

    Wraps pdfplumber.open in try/except; on failure calls _fail with a clear
    error message (satisfies T-01-malformed-pdf threat mitigation).
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_chars = 0
            for page in pdf.pages:
                total_chars += len(page.chars)
                if total_chars >= threshold:
                    return False  # early exit: enough text found
    except Exception as e:
        _fail(f"cannot open PDF: {e}")
    return True  # less than threshold chars across entire document


def _detect_layout(page) -> bool:
    """Return True if page appears to be two-column (D-06 heuristic).

    Crops to the bottom 70% of the page (Pitfall 2 mitigation: the single-column
    title block at the top of two-column papers would otherwise suppress detection).
    Counts words whose x0 exceeds the page midpoint. Returns True when the right-half
    word ratio exceeds TWO_COL_RIGHT_RATIO (0.30).
    """
    region = page.crop((0, page.height * 0.3, page.width, page.height))
    words = region.extract_words()
    if not words:
        return False
    mid = page.width / 2
    right_count = sum(1 for w in words if w["x0"] > mid)
    return (right_count / len(words)) > TWO_COL_RIGHT_RATIO


def _extract_text_pages(pdf, is_two_col: bool) -> list[str]:
    """Extract text from all pages in reading order; crop columns if two-column."""
    texts = []
    for page in pdf.pages:
        if is_two_col:
            # Two-column: crop at midpoint and concatenate left then right
            left = page.crop((0, 0, page.width / 2, page.height))
            right = page.crop((page.width / 2, 0, page.width, page.height))
            texts.append(
                (left.extract_text(x_tolerance=3) or "") +
                "\n" +
                (right.extract_text(x_tolerance=3) or "")
            )
        else:
            texts.append(page.extract_text(x_tolerance=3) or "")
    return texts


def _extract_metadata(pdf_path: str) -> dict:
    """Extract title, authors, year from PDF metadata dict and pypdf fallback.

    Primary: pdfplumber pdf.metadata dict (Title, Author keys).
    Secondary: pypdf for typed creation_date.year.
    Fallback (Pitfall 1): page-1 font-size heuristic for title; ["Unknown"] for authors.
    All font-size comparisons use round(size, 1) and > body_size + 0.5 threshold.
    """
    meta = {"title": None, "authors": None, "year": None}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Primary: pdfplumber metadata dict
            raw = pdf.metadata or {}
            if raw.get("Title"):
                candidate = str(raw["Title"]).strip()
                if candidate.lower() not in _TITLE_SENTINELS:
                    meta["title"] = candidate or None
            if raw.get("Author"):
                meta["authors"] = [a.strip() for a in str(raw["Author"]).split(";") if a.strip()]

            # Fallback (Pitfall 1): extract title from page-1 font heuristic
            if not meta["title"] and pdf.pages:
                page0 = pdf.pages[0]
                words = page0.extract_words(extra_attrs=["fontname", "size"])
                if words:
                    # Find modal body font size
                    all_sizes = [w["size"] for w in words if w["size"] > 0]
                    if all_sizes:
                        body_size = Counter(round(s, 1) for s in all_sizes).most_common(1)[0][0]

                        # Group words into lines by approximate top position
                        lines: dict[float, list] = {}
                        for w in words:
                            line_top = round(w["top"], 0)
                            lines.setdefault(line_top, []).append(w)

                        # Find first line whose size exceeds body_size + 0.5
                        for top, line_words in sorted(lines.items()):
                            line_size = round(line_words[0]["size"], 1)
                            if line_size > body_size + 0.5:
                                meta["title"] = " ".join(w["text"] for w in line_words).strip()
                                break

                # Last resort: first non-empty line of page 1 text
                if not meta["title"]:
                    page_text = page0.extract_text(x_tolerance=3) or ""
                    for line in page_text.splitlines():
                        line = line.strip()
                        if line:
                            meta["title"] = line
                            break
    except Exception as e:
        _fail(f"cannot open PDF: {e}")

    # Secondary: pypdf for typed creation_date.year
    if not meta["year"]:
        try:
            reader = PdfReader(pdf_path)
            if reader.metadata and reader.metadata.creation_date:
                meta["year"] = reader.metadata.creation_date.year
        except Exception:
            pass  # metadata missing or malformed — proceed without year

    # D-02 fallback: authors must always be a list
    if not meta["authors"]:
        meta["authors"] = ["Unknown"]

    return meta


def _extract_arxiv_id(page1_text: str) -> str | None:
    """Extract arXiv ID from page 1 text using regex (e.g., arXiv:1706.03762v5).

    Per Pitfall 7: arXiv preprints include their ID in page content, not metadata.
    """
    match = re.search(r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)', page1_text)
    return match.group(1) if match else None


# Intentionally not implemented in Phase 1 — Phase 2 will wire this.
# Use _find_sections_from_pdf directly until then.
def _find_sections(pages_text: list[str], body_size: float) -> list[dict]:
    """Identify section headers via font heuristic and return list of {title, body} dicts.

    Note: This public function accepts pages_text strings (no font data).
    For full font-heuristic section detection, use _find_sections_from_pdf directly.
    This wrapper satisfies the Plan 01 contract signature.

    TODO (Phase 2): implement body once pages_text carries font metadata.
    """
    raise NotImplementedError(
        "_find_sections is not implemented. Call _find_sections_from_pdf instead."
    )


def _find_sections_from_pdf(pdf_path: str) -> tuple[list[dict], float]:
    """Extract sections from a PDF using font heuristic. Returns (sections, body_size).

    This is the internal implementation that accesses word-level font data by
    reopening the PDF. The public _find_sections signature only accepts pages_text
    strings (which have no font info), so extract_paper calls this directly.

    Section headers are detected when:
      - round(line_size, 1) > body_size + 0.5  (size-based — per Pitfall 3)
      - OR "bold" in fontname.lower()           (bold-based — catches numbered headers)

    Fallback: if no headers detected, returns a single {"title": "Body", "body": full_text}.
    """
    sections = []
    body_size = 10.0  # default fallback

    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Pass 1: collect all word sizes to find modal body font size
            all_sizes = []
            for page in pdf.pages:
                words = page.extract_words(extra_attrs=["fontname", "size"])
                all_sizes.extend(w["size"] for w in words if w["size"] > 0)

            if not all_sizes:
                # No text at all — return empty sections (caller will apply fallback)
                return [], body_size

            body_size = Counter(round(s, 1) for s in all_sizes).most_common(1)[0][0]

            # Pass 2: collect all lines across pages with their font info
            doc_lines = []  # list of {text, size, fontname, page_num, top}
            for page_num, page in enumerate(pdf.pages, 1):
                words = page.extract_words(extra_attrs=["fontname", "size"])
                # Group words by approximate top position (same line = within 0.5 pt)
                lines: dict[float, list] = {}
                for w in words:
                    line_top = round(w["top"], 0)
                    lines.setdefault(line_top, []).append(w)

                for top, line_words in sorted(lines.items()):
                    line_text = " ".join(w["text"] for w in line_words)
                    line_size = round(line_words[0]["size"], 1)
                    line_font = line_words[0]["fontname"]
                    doc_lines.append({
                        "text": line_text,
                        "size": line_size,
                        "fontname": line_font,
                        "page_num": page_num,
                        "top": top,
                    })

            # Pass 3: identify header lines and build sections
            def _is_header(line: dict) -> bool:
                is_larger = line["size"] > body_size + 0.5
                is_bold = "bold" in line["fontname"].lower()
                # Guard: very long lines are body text, not section headers
                is_short_enough = len(line["text"]) < 120
                return (is_larger or is_bold) and is_short_enough

            current_header = None
            current_body_lines = []

            for line in doc_lines:
                if _is_header(line):
                    # Save previous section
                    if current_header is not None:
                        body_text = "\n".join(current_body_lines).strip()
                        sections.append({"title": current_header, "body": body_text})
                    elif current_body_lines:
                        # Text before first header — treat as preamble
                        body_text = "\n".join(current_body_lines).strip()
                        if body_text:
                            sections.append({"title": "Preamble", "body": body_text})
                    current_header = line["text"]
                    current_body_lines = []
                else:
                    current_body_lines.append(line["text"])

            # Flush last section
            if current_header is not None:
                body_text = "\n".join(current_body_lines).strip()
                sections.append({"title": current_header, "body": body_text})
            elif current_body_lines:
                # No headers at all — entire document is body
                body_text = "\n".join(current_body_lines).strip()
                if body_text:
                    sections.append({"title": "Body", "body": body_text})

    except Exception as e:
        _fail(f"cannot open PDF: {e}")

    # D-02 fallback: at least one section
    if not sections:
        sections = [{"title": "Body", "body": ""}]

    return sections, body_size


def _read_registry(registry_path: str) -> dict:
    """Read the JSON registry file; return empty dict if missing or corrupted.

    Treats a missing file and a corrupted (non-JSON) file identically — both
    return {} so the caller can safely proceed without crashing (T-01-05 mitigation).
    No file lock needed for reads (RESEARCH.md Pattern 5 note).
    """
    if not os.path.exists(registry_path):
        return {}
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}  # corrupted registry treated as empty — never crash on read


def _write_registry_entry(registry_path: str, key: str, entry: dict) -> None:
    """Atomically add or update one entry in the registry JSON using filelock + os.replace.

    Each call creates its own FileLock instance (thread_local=True default means
    each thread must own its lock object — never share across threads). Uses
    tempfile.mkstemp(dir=registry-dir) so the temp file is on the same filesystem
    as the registry, making os.replace atomic (Pitfall 6 mitigation).
    """
    lock_path = registry_path + ".lock"
    # New FileLock instance per call — required for thread safety (RESEARCH.md critical note)
    lock = filelock.FileLock(lock_path, timeout=30)

    # Ensure parent directory exists before acquiring the lock
    dir_path = os.path.dirname(os.path.abspath(registry_path))
    os.makedirs(dir_path, exist_ok=True)

    with lock:
        # Read current state under lock
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    registry = json.load(f)
            except json.JSONDecodeError:
                registry = {}  # corrupted — start fresh
        else:
            registry = {}

        registry[key] = entry

        # Atomic write: temp file in same dir ensures same-filesystem atomic replace (Pitfall 6)
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, registry_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def extract_paper(pdf_path: str) -> dict:
    """
    Extract structured PaperJSON from a PDF file and write to the global registry.

    Checks the registry for a previously ingested entry (REG-02 dedup check) before
    running the full extraction pipeline. On a cache miss, runs full extraction and
    writes the registry entry (REG-01). On a cache hit, returns the cached entry
    directly without re-extracting.

    The registry entry stores the full PaperJSON plus D-14 metadata fields
    (summary, key_findings, projects, vault_note) so that a cache hit returns
    the same PaperJSON schema as a fresh extraction.

    Args:
        pdf_path: Absolute path to the PDF file to ingest.

    Returns:
        PaperJSON dict with required fields: title, authors, abstract, sections,
        source_path, doi, arxiv_id, year, journal, references.
    """
    # Step 1: scanned PDF guard (INGEST-03, T-01-03)
    if _is_scanned(pdf_path):
        _fail("scanned/image-only PDF — no extractable text (INGEST-03)")

    # Step 2: load config for registry path and project name
    config_path = _find_config()
    config = _load_config(config_path)
    registry_path = config["registry_path"]
    project_name = config["project_name"]

    # Step 3: extract metadata (title, authors, year) — lightweight; needed for key computation
    meta = _extract_metadata(pdf_path)
    title = meta["title"] or "Unknown"
    authors = meta["authors"] or ["Unknown"]
    year = meta["year"]

    # Step 4: layout detection and text extraction (needed for arXiv ID)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page0 = pdf.pages[0] if pdf.pages else None
            is_two_col = _detect_layout(page0) if page0 else False
            pages_text = _extract_text_pages(pdf, is_two_col)
    except Exception as e:
        _fail(f"cannot open PDF: {e}")

    # Step 5: extract arXiv ID from page 1 text
    page1_text = pages_text[0] if pages_text else ""
    arxiv_id = _extract_arxiv_id(page1_text)

    # Step 6: compute registry key from lightweight paper identity (D-12)
    # DOI is not extracted in Phase 1 (Phase 3 web path fills it); key falls to
    # arXiv ID (if present) or SHA-256 title hash.
    minimal_paper = {"doi": None, "arxiv_id": arxiv_id, "title": title}
    key = _compute_registry_key(minimal_paper)

    # Step 7: DEDUP CHECK (REG-02) — return cached entry if already ingested
    registry = _read_registry(registry_path)
    if key in registry:
        return registry[key]

    # Step 8: section detection via font heuristic (cache miss — full extraction)
    sections, body_size = _find_sections_from_pdf(pdf_path)

    # Step 9: extract abstract from sections
    abstract = ""
    for section in sections:
        if "abstract" in section["title"].lower():
            abstract = section["body"]
            break

    # Step 10: extract references from sections
    references = None
    for section in sections:
        if section["title"].lower() in ("references", "bibliography"):
            ref_text = section["body"]
            # Split on newline + leading number pattern
            raw_refs = re.split(r'\n\s*\[?\d+\]?\.?\s+', ref_text)
            references = [r.strip() for r in raw_refs if r.strip()]
            break

    # Step 11: assemble PaperJSON (D-01, D-02, D-03, D-04)
    paper_json = {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "sections": sections if sections else [{"title": "Body", "body": ""}],
        "doi": None,  # Phase 3 web path fills DOI
        "arxiv_id": arxiv_id,
        "year": year,
        "journal": None,
        "references": references,
        "source_path": os.path.abspath(pdf_path),
    }

    # Step 12: build D-14 registry entry and write (REG-01).
    # The registry entry stores the full PaperJSON plus D-14 metadata fields so that
    # a future cache hit (REG-02) returns the same PaperJSON schema.
    # summary and key_findings are null in Phase 1 (Open Question 2 — Phase 2 fills these).
    # projects = [project_name] (Open Question 3 resolution).
    registry_entry = dict(paper_json)  # start with full PaperJSON
    registry_entry.update({
        "summary": None,
        "key_findings": None,
        "projects": [project_name],
        "vault_note": None,
    })
    _write_registry_entry(registry_path, key, registry_entry)

    # Return full PaperJSON (D-10 — 10-key schema returned to Claude)
    return paper_json


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into PaperJSON + registry"
    )
    parser.add_argument("--pdf", required=True, help="Absolute path to the PDF file")
    args = parser.parse_args()
    paper_json = extract_paper(args.pdf)
    print(json.dumps(paper_json, ensure_ascii=False))
