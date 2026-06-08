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
import urllib.error
import urllib.request
from typing import NoReturn
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
_AUTHOR_SENTINELS = frozenset({"unknown", "anonymous", "n/a", "none", ""})
_AUTHOR_HEADER_KEYWORDS = frozenset({
    "abstract", "introduction", "keywords", "doi", "arxiv", "copyright",
    "received", "accepted", "published", "university", "department",
    "institute", "school", "lab",
})
REQUIRED_CONFIG_KEYS = ("registry_path", "vault_path")

PAPER_JSON_KEYS = frozenset({
    "title", "authors", "abstract", "sections", "doi",
    "arxiv_id", "year", "journal", "references", "figures", "source_path",
})

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:e4b"
LLM_TEXT_TRUNCATE = 80_000  # max chars sent to Ollama — context overflow guard
MAX_SECTIONS = 12  # max per-section LLM calls in the extraction pipeline

# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _fail(msg: str) -> NoReturn:
    """Print error to stderr and exit non-zero."""
    print(f"[ingest error: {msg}]", file=sys.stderr)
    sys.exit(1)


def _load_config(config_path: str) -> dict:
    """Load config.json from the given path.

    Validates that required keys (registry_path, vault_path) are present.
    Expands ~ in registry_path and vault_path. Defaults project_name to the
    basename of the directory containing config.json (Open Question 3 resolution).
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        _fail(f"cannot read config.json: {e}")

    # Validate required keys before use
    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            _fail(f"config.json is missing required key: '{key}'")

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
                text = page.extract_text() or ""
                total_chars += len(text)
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


def _extract_arxiv_id(page1_text: str) -> str | None:
    """Extract arXiv ID from page 1 text using regex (e.g., arXiv:1706.03762v5).

    Per Pitfall 7: arXiv preprints include their ID in page content, not metadata.
    """
    match = re.search(r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)', page1_text)
    return match.group(1) if match else None


def _extract_authors_from_text(page1_text: str) -> list[str]:
    """Scan page 1 text for author names when the LLM returns the 'Unknown' sentinel.

    Heuristic: checks the first 15 lines for candidate author lines. A candidate line
    must be non-empty, not a section-header keyword, 5-200 chars, and must split into
    tokens that look like personal names (5-200 chars, at least one space, first char
    uppercase, not all-uppercase). Splits on comma, ' and ', ' & '.

    Returns a list of cleaned name strings, or [] if no candidates found.
    No new imports — uses only re (already imported at module level).
    """

    lines = page1_text.split("\n")[:15]
    candidates = []
    for line in lines:
        line = line.strip()
        # Skip empty, too short, too long
        if not line or len(line) < 5 or len(line) > 200:
            continue
        # Skip lines containing section-header keywords
        line_lower = line.lower()
        if any(kw in line_lower for kw in _AUTHOR_HEADER_KEYWORDS):
            continue
        # Split by comma, ' and ', ' & ' to get individual name tokens
        tokens = re.split(r",\s*| and | & ", line)
        names = []
        for token in tokens:
            token = token.strip()
            if len(token) < 5 or len(token) > 200:
                continue
            # Must contain at least one space (first + last name minimum)
            if " " not in token:
                continue
            # First char must be uppercase
            if not token[0].isupper():
                continue
            # Must not be all-uppercase (avoids title-case section headers)
            if token.isupper():
                continue
            names.append(token)
        if names:
            candidates.extend(names)

    return candidates


def _extract_with_llm(pages_text: list[str]) -> dict:
    """Multi-call pipeline: section discovery → metadata → per-section bodies.

    Issues at least 2 separate HTTP calls to Ollama /api/chat for any non-empty
    pages_text. Each call produces a small, bounded JSON response that cannot
    overflow the Ollama JSON formatter (root cause fix for monolithic call failures).

    Raises urllib.error.URLError or TimeoutError when Ollama is unreachable.
    Returns fallback defaults on non-network exceptions.

    Returns a dict with keys: title, authors, abstract, sections, bibliography, figures.
    """
    from pydantic import BaseModel, ValidationError

    class _LLMSectionList(BaseModel):
        sections: list[str] = []

    class _LLMMetadata(BaseModel):
        title: str = ""
        authors: list[str] = []
        abstract: str = ""

    class _LLMSection(BaseModel):
        title: str = ""
        body: str = ""

    def _fallback() -> dict:
        return {
            "title": "",
            "authors": ["Unknown"],
            "abstract": "",
            "sections": [{"title": "Body", "body": ""}],
            "bibliography": None,
            "figures": None,
        }

    combined = "\n\n".join(pages_text)

    def _llm_post(prompt: str, timeout: int = 120) -> str:
        """POST prompt to Ollama /api/chat; return stripped content string."""
        payload = json.dumps({
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
        }).encode()
        req = urllib.request.Request(
            OLLAMA_BASE + "/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        raw = body["message"]["content"]
        return re.sub(r"^```(?:json)?\s*\n?|\n?```\s*$", "", raw.strip(), flags=re.MULTILINE)

    # Call 1: section discovery
    section_titles: list[str] = []
    try:
        raw_discovery = _llm_post(
            "You are a scientific paper parser. Read the paper text below and return ONLY "
            "a JSON object with one key 'sections' whose value is an array of section title "
            "strings in reading order. Example: {\"sections\": [\"Introduction\", \"Methods\", "
            "\"Results\", \"Conclusion\"]}. Return ONLY the JSON object, nothing else.\n\n"
            "Paper text:\n" + combined[:LLM_TEXT_TRUNCATE],
            timeout=120,
        )
        try:
            parsed = json.loads(raw_discovery)
            if isinstance(parsed, list):
                section_titles = [t for t in parsed if isinstance(t, str) and t.strip()]
            else:
                model = _LLMSectionList(**parsed)
                section_titles = [t for t in model.sections if isinstance(t, str) and t.strip()]
        except (json.JSONDecodeError, ValueError, ValidationError, TypeError):
            section_titles = []
    except (urllib.error.URLError, TimeoutError):
        raise
    except Exception:
        return _fallback()

    # Call 2: metadata (always runs — even when section discovery returned empty)
    page1_text = pages_text[0] if pages_text else ""
    title = ""
    authors: list[str] = ["Unknown"]
    abstract = ""
    try:
        raw_meta = _llm_post(
            "You are a scientific paper parser. Read the paper text below and return ONLY "
            "a JSON object with keys: 'title' (string), 'authors' (array of strings), "
            "'abstract' (string). Return ONLY the JSON object, nothing else.\n\n"
            "Paper text:\n" + page1_text[:8000],
            timeout=120,
        )
        try:
            parsed = json.loads(raw_meta)
            meta = _LLMMetadata(**parsed)
            title = meta.title
            authors = meta.authors or ["Unknown"]
            abstract = meta.abstract
        except (json.JSONDecodeError, ValueError, ValidationError, TypeError):
            pass
    except (urllib.error.URLError, TimeoutError):
        raise
    except Exception:
        pass

    # Section discovery returned empty — use fallback sections, skip per-section calls
    if not section_titles:
        fig_matches = re.findall(
            r'((?:Figure|Fig\.|Table)\s+\d+)',
            combined[:LLM_TEXT_TRUNCATE],
            flags=re.IGNORECASE,
        )
        return {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "sections": [{"title": "Body", "body": combined[:10000]}],
            "bibliography": None,
            "figures": [{"label": m, "caption": ""} for m in fig_matches[:20]] if fig_matches else None,
        }

    # Calls 3..N: per-section bodies (capped at MAX_SECTIONS)
    sections: list[dict] = []
    capped_titles = section_titles[:MAX_SECTIONS]
    for i, sec_title in enumerate(capped_titles):
        m = re.search(r'(?m)^\s*' + re.escape(sec_title), combined)
        if not m:
            chunk = ""
        else:
            start = m.start()
            end = len(combined)
            if i + 1 < len(capped_titles):
                m_next = re.search(r'(?m)^\s*' + re.escape(capped_titles[i + 1]), combined)
                if m_next and m_next.start() > start:
                    end = m_next.start()
            chunk = combined[start:end]

        body_text = ""
        try:
            raw_sec = _llm_post(
                "You are a scientific paper parser. Read the following section text and return "
                "ONLY a JSON object with keys: 'title' (section title as string) and 'body' "
                "(section content as string). Return ONLY the JSON object, nothing else.\n\n"
                f"Section title: {sec_title}\n\nSection text:\n" + chunk,
                timeout=120,
            )
            try:
                parsed = json.loads(raw_sec)
                sec_model = _LLMSection(**parsed)
                body_text = sec_model.body
            except (json.JSONDecodeError, ValueError, ValidationError, TypeError):
                body_text = ""
        except (urllib.error.URLError, TimeoutError):
            raise
        except Exception:
            body_text = ""
        sections.append({"title": sec_title, "body": body_text})

    if not sections:
        sections = [{"title": "Body", "body": combined[:10000]}]

    # Bibliography: check last section title for reference/bibliography markers
    bibliography = None
    if sections:
        last_title = sections[-1]["title"].lower()
        if "reference" in last_title or "bibliograph" in last_title:
            lines = [ln.strip() for ln in sections[-1]["body"].split("\n") if ln.strip()]
            bibliography = lines if lines else None

    # Figures: regex scan for Figure/Table labels — no extra LLM call
    fig_matches = re.findall(
        r'((?:Figure|Fig\.|Table)\s+\d+)',
        combined[:LLM_TEXT_TRUNCATE],
        flags=re.IGNORECASE,
    )
    figures = [{"label": m, "caption": ""} for m in fig_matches[:20]] if fig_matches else None

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "sections": sections,
        "bibliography": bibliography,
        "figures": figures,
    }


def _read_registry(registry_path: str) -> dict:
    """Read the JSON registry file; return empty dict if missing or unreadable.

    Treats any read failure (missing file, encoding error, corrupted JSON, OS error)
    as empty — always returns {} rather than raising. This covers registry files
    written by PowerShell (UTF-16 BOM), files with OS-level permission errors, and
    files with corrupted JSON content (T-01-05, T-01-06 mitigations).
    No file lock needed for reads (RESEARCH.md Pattern 5 note).
    """
    if not os.path.exists(registry_path):
        return {}
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}  # any read failure treated as empty — never crash on read


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
        source_path, doi, arxiv_id, year, journal, references, figures.
    """
    # Step 1: scanned PDF guard (INGEST-03, T-01-03)
    if _is_scanned(pdf_path):
        _fail("scanned/image-only PDF — no extractable text (INGEST-03)")

    # Step 2: load config for registry path and project name
    config_path = _find_config()
    config = _load_config(config_path)
    registry_path = config["registry_path"]
    project_name = config["project_name"]

    # Step 3: layout detection and text extraction
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page0 = pdf.pages[0] if pdf.pages else None
            is_two_col = _detect_layout(page0) if page0 else False
            pages_text = _extract_text_pages(pdf, is_two_col)
    except Exception as e:
        _fail(f"cannot open PDF: {e}")

    # Step 4: extract arXiv ID from page 1 text
    page1_text = pages_text[0] if pages_text else ""
    arxiv_id = _extract_arxiv_id(page1_text)

    # Step 5: LLM-based extraction (required — fails fast if Ollama unreachable)
    try:
        llm_result = _extract_with_llm(pages_text)
    except (urllib.error.URLError, TimeoutError) as e:
        _fail(f"Ollama unreachable — LLM is required for extraction: {e}")

    title = llm_result["title"] or "Unknown"
    authors = llm_result["authors"] or ["Unknown"]
    abstract = llm_result["abstract"]
    sections = llm_result["sections"]
    references = llm_result.get("bibliography") or None
    figures = llm_result.get("figures")

    # Offline fallback: recover authors from body text when LLM returns sentinel
    if authors == ["Unknown"] and page1_text:
        _heuristic = _extract_authors_from_text(page1_text)
        if _heuristic:
            authors = _heuristic

    # Step 6: inline pypdf year extraction (LLM does not return year)
    year = None
    try:
        reader = PdfReader(pdf_path)
        if reader.metadata and reader.metadata.creation_date:
            year = reader.metadata.creation_date.year
    except Exception:
        pass  # metadata missing or malformed — proceed without year

    # Step 7: compute registry key from paper identity (D-12)
    # DOI is not extracted in Phase 1 (Phase 3 web path fills it); key falls to
    # arXiv ID (if present) or SHA-256 title hash.
    # When LLM extraction fails (empty title), use a path-based hash to ensure
    # per-document uniqueness — prevents all failed-LLM papers from colliding on
    # the same sha256 key for normalized "unknown".
    _llm_title = llm_result["title"] or ""
    if not _llm_title:
        # LLM extraction failed — use PDF path hash to ensure per-document uniqueness
        _path_hash = hashlib.sha256(
            os.path.abspath(pdf_path).encode()
        ).hexdigest()[:DEFAULT_REGISTRY_KEY_PREFIX_LEN]
        minimal_paper = {"doi": None, "arxiv_id": arxiv_id, "title": f"__path:{_path_hash}"}
    else:
        minimal_paper = {"doi": None, "arxiv_id": arxiv_id, "title": _llm_title}
    key = _compute_registry_key(minimal_paper)

    # Step 8: DEDUP CHECK (REG-02) — return cached entry if already ingested
    registry = _read_registry(registry_path)
    if key in registry:
        return {k: registry[key][k] for k in PAPER_JSON_KEYS if k in registry[key]}

    # Step 9: assemble PaperJSON (D-01, D-02, D-03, D-04)
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
        "figures": figures,
        "source_path": os.path.abspath(pdf_path),
    }

    # Step 10: build D-14 registry entry and write (REG-01).
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
    # Step 10: write registry entry with consistent error reporting
    try:
        _write_registry_entry(registry_path, key, registry_entry)
    except Exception as e:
        _fail(f"cannot write registry entry '{key}': {e}")

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
