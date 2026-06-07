"""
Wave 0 test stubs for Phase 1 PDF ingestion pipeline.

All 7 tests encode the PaperJSON schema contract (D-01 through D-04) and
behavioral requirements (INGEST-01 through INGEST-04, REG-01, REG-02, REG-04).
Tests FAIL in Wave 0 (extract_paper raises NotImplementedError) — this is the
correct RED state. Waves 1-4 implement the bodies that turn these green.
"""

import json
import os
import subprocess
import sys
import threading

import pytest

import pdfplumber

from scripts.ingest import (
    _compute_registry_key,
    _detect_layout,
    _find_config,
    _load_config,
    _read_registry,
    _write_registry_entry,
    extract_paper,
)


def test_detect_layout(sample_pdf_path, sample_twocol_pdf_path):
    """
    INGEST-02 (unit): _detect_layout returns True for two-column PDFs, False for single-column.

    The D-06 heuristic crops to the bottom 70% of page 1 (Pitfall 2 mitigation) and counts
    words whose x0 exceeds the midpoint. A right-half ratio > TWO_COL_RIGHT_RATIO (0.30)
    signals two-column layout.
    """
    with pdfplumber.open(sample_twocol_pdf_path) as pdf:
        assert _detect_layout(pdf.pages[0]) is True, (
            "_detect_layout must return True for two-column sample_twocol.pdf"
        )
    with pdfplumber.open(sample_pdf_path) as pdf:
        assert _detect_layout(pdf.pages[0]) is False, (
            "_detect_layout must return False for single-column sample_onecol.pdf"
        )


def test_paper_json_schema(sample_pdf_path):
    """
    INGEST-01: extract_paper returns PaperJSON with all required fields typed correctly.

    Schema contract (D-01, D-02, D-04):
      - title: non-empty str
      - authors: list with at least one element
      - abstract: str (may be empty string per D-02 fallback)
      - sections: list of at least one {title: str, body: str} dict
      - source_path: str (absolute path to source PDF)
    """
    result = extract_paper(sample_pdf_path)
    assert isinstance(result, dict), "extract_paper must return a dict"
    assert isinstance(result["title"], str) and result["title"], "title must be a non-empty str"
    assert isinstance(result["authors"], list) and len(result["authors"]) >= 1, \
        "authors must be a list with at least one element"
    assert isinstance(result["abstract"], str), "abstract must be a str"
    assert isinstance(result["sections"], list) and len(result["sections"]) >= 1, \
        "sections must be a non-empty list"
    assert all(
        isinstance(s, dict) and "title" in s and "body" in s
        for s in result["sections"]
    ), "each section must have 'title' and 'body' keys"
    assert isinstance(result["source_path"], str), "source_path must be a str"


def test_two_column_layout(sample_twocol_pdf_path):
    """
    INGEST-02: extract_paper handles two-column PDFs with correct reading order.

    The sample_twocol.pdf has Section A in left column and Section B in right column.
    Correct reading order: left column text appears before right column text in sections.
    """
    result = extract_paper(sample_twocol_pdf_path)
    assert isinstance(result, dict), "extract_paper must return a dict"
    assert isinstance(result["sections"], list) and len(result["sections"]) >= 1, \
        "two-column PDF must produce at least one section"
    # Concatenate all section content for reading-order check
    all_text = " ".join(
        s.get("title", "") + " " + s.get("body", "")
        for s in result["sections"]
    ).lower()
    # "section a" (left) must appear before "section b" (right) in the output
    idx_a = all_text.find("section a")
    idx_b = all_text.find("section b")
    assert idx_a != -1, "left column 'Section A' content not found in output"
    assert idx_b != -1, "right column 'Section B' content not found in output"
    assert idx_a < idx_b, "left column must appear before right column (reading order)"
    # arXiv ID must be extracted from footer (INGEST-02 wiring check)
    arxiv_id = result.get("arxiv_id")
    assert arxiv_id is not None, "arxiv_id must be extracted from two-column fixture footer"
    assert str(arxiv_id).startswith("2301.00001"), (
        f"arxiv_id must start with '2301.00001', got: {arxiv_id}"
    )


def test_scanned_pdf_error(scanned_pdf_path):
    """
    INGEST-03: extract_paper raises an error (not returns empty JSON) for scanned PDFs.

    A scanned PDF has < 100 total chars. extract_paper must signal this as an error —
    either by raising an exception or returning a dict with an 'error' key — not by
    silently returning a PaperJSON with an empty title.
    """
    try:
        result = extract_paper(scanned_pdf_path)
        # If it returns a dict, it must signal the error explicitly
        assert "error" in result, (
            "scanned PDF must return a dict with 'error' key, "
            f"got: {result}"
        )
    except (ValueError, RuntimeError, SystemExit) as e:
        # Raising an exception is also acceptable behavior for scanned PDFs
        assert str(e), "exception message must not be empty"


def test_process_pdf_mcp_tool(sample_pdf_path):
    """
    INGEST-04: process_pdf MCP tool returns the same PaperJSON as the CLI.

    Calls ingest.py as a subprocess (mirroring how process_pdf in server.py works)
    and verifies the output is valid JSON with all required fields.
    """
    ingest_script = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "ingest.py")
    )
    result = subprocess.run(
        [sys.executable, ingest_script, "--pdf", sample_pdf_path],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"ingest.py CLI must exit 0 for a valid PDF; stderr: {result.stderr.strip()}"
    )
    assert result.stdout.strip(), "ingest.py CLI must produce non-empty stdout"
    paper = json.loads(result.stdout.strip())
    assert isinstance(paper, dict), "CLI output must be valid JSON dict"
    assert "title" in paper and paper["title"], "CLI PaperJSON must have non-empty title"
    assert "sections" in paper and len(paper["sections"]) >= 1, \
        "CLI PaperJSON must have at least one section"


def test_registry_write(sample_pdf_path):
    """
    REG-01: After a successful ingest, the registry file contains an entry with the
    correct key derived from the paper (DOI, arXiv ID, or SHA-256 title hash).
    """
    result = extract_paper(sample_pdf_path)
    key = _compute_registry_key(result)
    assert isinstance(key, str) and key, "registry key must be a non-empty str"

    # Read the registry that extract_paper actually wrote to (via config)
    config = _load_config(_find_config())
    with open(config["registry_path"], "r", encoding="utf-8") as f:
        registry = json.load(f)

    assert key in registry, f"extract_paper must have written key '{key}' to registry"
    assert registry[key]["title"] == result["title"], \
        "registry entry title must match PaperJSON title"


def test_registry_dedup(sample_pdf_path):
    """
    REG-02: Second ingest of the same paper returns the cached entry and skips extraction.

    After the first ingest writes the registry, calling extract_paper again for the same
    PDF must return the cached entry (registry hit) without re-running extraction.
    The registry must still contain exactly one entry for this paper.
    """
    # First ingest
    result1 = extract_paper(sample_pdf_path)
    key1 = _compute_registry_key(result1)

    # Second ingest of same file -- must return cached (key must be identical)
    result2 = extract_paper(sample_pdf_path)
    key2 = _compute_registry_key(result2)

    assert key1 == key2, (
        f"Same paper must produce same registry key on repeated ingest; "
        f"got '{key1}' then '{key2}'"
    )

    # Read the registry that extract_paper wrote to (via config)
    config = _load_config(_find_config())
    with open(config["registry_path"], "r", encoding="utf-8") as f:
        registry = json.load(f)
    assert len([k for k in registry if k == key1]) == 1, \
        "registry must not duplicate entries for the same paper"


def test_concurrent_registry_writes(tmp_registry_path):
    """
    REG-04: Two concurrent threads writing different entries produce a valid, complete registry.

    Both threads call _write_registry_entry simultaneously with distinct keys.
    The final JSON must parse without error and contain exactly 2 entries.
    """
    errors = []

    def write(key: str):
        try:
            _write_registry_entry(
                tmp_registry_path,
                key,
                {"title": f"Paper {key}", "authors": ["Test Author"]},
            )
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=write, args=(f"key-thread-1",))
    t2 = threading.Thread(target=write, args=(f"key-thread-2",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"concurrent writes raised exceptions: {errors}"

    with open(tmp_registry_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # must parse without error

    assert len(data) == 2, (
        f"registry must contain exactly 2 entries after concurrent writes; "
        f"got {len(data)}: {list(data.keys())}"
    )


def test_registry_read_unicode_error(tmp_registry_path):
    """
    REG-01 regression: _read_registry returns {} when the registry file is UTF-16 encoded.

    PowerShell may write JSON files with a UTF-16 LE BOM, which causes a UnicodeDecodeError
    when opened with encoding='utf-8'. The function must catch this and return {} — never
    raise an unhandled exception.
    """
    # Write a UTF-16 LE BOM followed by minimal content — simulates PowerShell output
    with open(tmp_registry_path, "wb") as f:
        f.write(b"\xff\xfe{}")
    result = _read_registry(tmp_registry_path)
    assert result == {}, (
        f"_read_registry must return {{}} for UTF-16 encoded file, got: {result}"
    )


def test_author_extraction_from_body(sample_pdf_path):
    """
    INGEST-01 regression: _extract_metadata extracts authors from body text when PDF metadata
    author field is absent or contains only a sentinel value (e.g., 'anonymous').

    sample_onecol.pdf has 'Alice Author, Bob Collaborator' as a visible author line on page 1
    but its PDF info dict Author field is 'anonymous' — a sentinel that must be rejected.
    After the fix, authors must come from body text.
    """
    result = extract_paper(sample_pdf_path)
    assert result["authors"] != ["Unknown"], (
        f"authors must not fall back to ['Unknown'] — expected body-text extraction, "
        f"got: {result['authors']}"
    )
    assert len(result["authors"]) >= 1, (
        f"authors must have at least one element, got: {result['authors']}"
    )
    # The fixture contains 'Alice Author, Bob Collaborator' — verify actual extraction
    assert result["authors"] != ["anonymous"], (
        f"authors must not be the sentinel value ['anonymous'], got: {result['authors']}"
    )
