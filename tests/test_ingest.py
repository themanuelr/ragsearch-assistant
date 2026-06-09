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
import urllib.error
import urllib.request
from unittest.mock import patch, MagicMock

import pytest

from scripts.ingest import (
    PAPER_JSON_KEYS,
    _compute_registry_key,
    _doi_from_text,
    _extract_with_llm,
    _find_config,
    _load_config,
    _read_registry,
    _write_registry_entry,
    extract_paper,
)


def _make_mock_response(inner_payload):
    """Build a mocked Ollama /api/chat response context manager.

    Wraps inner_payload (the model's content) in the Ollama envelope
    {"message": {"content": "<json string>"}} and returns a context-manager
    MagicMock suitable for mock_urlopen.side_effect entries.
    """
    response_bytes = json.dumps(
        {"message": {"content": json.dumps(inner_payload)}}
    ).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = response_bytes
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_resp
    mock_ctx.__exit__.return_value = None
    return mock_ctx


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


def test_two_column_arxiv_id_extraction(sample_twocol_pdf_path):
    """
    INGEST-02: extract_paper extracts the arXiv ID from a two-column PDF fixture.

    Text extraction uses extract_text(x_tolerance=3) per page (no layout detection —
    the column-reordering logic was removed by gap-closure plan 01-10). Sections are
    supplied by the autouse _mock_llm fixture, so this test only verifies real wiring
    that is exercised by production code: per-page extraction produces sections, and
    _extract_arxiv_id pulls the ID from the fixture footer.

    Reading-order assertions were removed (WR-02): the previous idx_a < idx_b check only
    verified the order of constants baked into the mock, not any code under test.
    """
    result = extract_paper(sample_twocol_pdf_path)
    assert isinstance(result, dict), "extract_paper must return a dict"
    assert isinstance(result["sections"], list) and len(result["sections"]) >= 1, \
        "two-column PDF must produce at least one section"
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



def _ollama_available() -> bool:
    """Return True if Ollama is reachable at localhost:11434."""
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _ollama_available(),
    reason="requires live Ollama server"
)
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
        timeout=360,
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


def test_registry_write(sample_pdf_path, _tmp_config):
    """
    REG-01: After a successful ingest, the registry file contains an entry with the
    correct key derived from the paper (DOI, arXiv ID, or SHA-256 title hash).
    """
    result = extract_paper(sample_pdf_path)

    # Use the temp config injected by _tmp_config fixture (bypasses monkeypatch binding issue)
    from scripts.ingest import _load_config as _lc
    config = _lc(_tmp_config)
    with open(config["registry_path"], "r", encoding="utf-8") as f:
        registry = json.load(f)

    assert len(registry) == 1, "registry must contain exactly one entry"
    actual_key, entry = next(iter(registry.items()))
    assert isinstance(actual_key, str) and actual_key, "registry key must be a non-empty str"
    assert entry["title"] == result["title"], \
        "registry entry title must match PaperJSON title"

def test_registry_dedup(sample_pdf_path, _tmp_config):
    """
    REG-02: Second ingest of the same paper returns the cached entry and skips extraction.

    After the first ingest writes the registry, calling extract_paper again for the same
    PDF must return the cached entry (registry hit) without re-running extraction.
    The registry must still contain exactly one entry for this paper.
    """
    # First ingest
    result1 = extract_paper(sample_pdf_path)

    # Second ingest of same file -- must return cached result unchanged
    result2 = extract_paper(sample_pdf_path)

    assert result1 == result2, (
        "Second ingest must return the cached registry entry unchanged — REG-02"
    )

    # Read the registry via the fixture config path (not unpatched _find_config) so we
    # always inspect the temp registry created by the _tmp_config autouse fixture,
    # not the real project registry.
    config = _load_config(_tmp_config)
    with open(config["registry_path"], "r", encoding="utf-8") as f:
        registry = json.load(f)
    assert len(registry) == 1, (
        f"registry must contain exactly 1 entry after two ingests of the same paper; "
        f"got {len(registry)}: {list(registry.keys())}"
    )


def test_registry_dedup_schema_match(sample_pdf_path):
    """
    REG-02 schema: Second ingest returns a result whose keys equal exactly PAPER_JSON_KEYS.

    Guards against D-14 metadata fields (summary, key_findings, projects, vault_note)
    leaking through the cache-hit path. The cache-hit return must filter the registry
    entry to only the 11 canonical PaperJSON keys.
    """
    # First ingest — populates registry
    result1 = extract_paper(sample_pdf_path)
    # Second ingest — must return from cache
    result2 = extract_paper(sample_pdf_path)

    assert result1 == result2, (
        "Second ingest must return cached entry unchanged — REG-02"
    )
    assert set(result2.keys()) == set(PAPER_JSON_KEYS), (
        f"Cache-hit return must contain exactly PAPER_JSON_KEYS keys; "
        f"extra keys: {set(result2.keys()) - set(PAPER_JSON_KEYS)}, "
        f"missing keys: {set(PAPER_JSON_KEYS) - set(result2.keys())}"
    )


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
    INGEST-01 regression: extract_paper returns authors from the LLM result, not from a
    deleted body-text heuristic.

    Under the autouse _mock_llm fixture, _extract_with_llm returns
    authors=["Alice Author", "Bob Collaborator"]. This test verifies that the
    extract_paper contract holds: authors must not be ["Unknown"], ["anonymous"], or empty.
    Authors flow exclusively from the LLM result (no offline heuristic path remains).
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


@patch("scripts.ingest.urllib.request.urlopen")
def test_extract_with_llm_section_discovery_fallback(mock_urlopen):
    """When section discovery returns malformed JSON, _extract_with_llm falls back to
    a single Body section without raising."""
    # Ollama /api/chat response envelope wraps content in {"message": {"content": "..."}}
    # The content itself is malformed JSON wrapped in markdown fences (Ollama bug #15260)
    ollama_response = json.dumps({"message": {"content": "```json\n{\"bad: json\"\n```"}}).encode()
    mock_response = MagicMock()
    mock_response.read.return_value = ollama_response
    mock_urlopen.return_value.__enter__.return_value = mock_response

    result = _extract_with_llm(["Sample paper text"])

    assert isinstance(result, dict), "result must be a dict — no exception raised"
    assert result["title"] == "", f"title must be empty string fallback, got: {result['title']}"
    assert len(result["sections"]) == 1
    assert result["sections"][0]["title"] == "Body"
    assert isinstance(result["sections"][0]["body"], str), (
        f"sections[0]['body'] must be a str, got: {type(result['sections'][0]['body'])}"
    )
    assert result["authors"] == ["Unknown"], (
        f"authors must be D-02 fallback, got: {result['authors']}"
    )


@patch("scripts.ingest.urllib.request.urlopen")
def test_extract_with_llm_multi_call_returns_fields(mock_urlopen):
    """Multi-call pipeline: section discovery + metadata + per-section calls return
    structured fields. bibliography is detected from References section; figures is
    None when no Figure/Table captions are present in the input text."""

    def _make_mock_response(inner_payload):
        response_bytes = json.dumps(
            {"message": {"content": json.dumps(inner_payload)}}
        ).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_bytes
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_resp
        mock_ctx.__exit__.return_value = None
        return mock_ctx

    mock_urlopen.side_effect = [
        _make_mock_response({                                  # Call 1: metadata
            "title": "Test Paper",
            "authors": ["A. Author"],
            "abstract": "An abstract.",
        }),
        _make_mock_response({                                  # Call 2: whole-document sections
            "sections": [
                {"title": "Introduction", "body": "Intro text."},
                {"title": "References", "body": "Smith et al. 2020."},
            ]
        }),
        _make_mock_response({                                  # Call 3: references
            "references": ["1. Smith et al. 2020.", "2. Doe et al. 2021."]
        }),
        _make_mock_response({                                  # Call 4: figures
            "figures": [{"label": "Fig. 1", "caption": "A figure."}]
        }),
    ]

    result = _extract_with_llm([
        "Introduction\nSome intro body.\n\nReferences\nSmith et al. 2020."
    ])

    assert result["title"] == "Test Paper", f"title must be 'Test Paper', got: {result['title']}"
    assert result["authors"] == ["A. Author"], f"authors must be ['A. Author'], got: {result['authors']}"
    assert result["abstract"] == "An abstract.", f"abstract mismatch: {result['abstract']}"
    assert len(result["sections"]) == 2, f"sections must have 2 entries, got: {len(result['sections'])}"
    assert result["sections"][0] == {"title": "Introduction", "body": "Intro text."}, (
        f"sections[0] mismatch: {result['sections'][0]}"
    )
    # references/figures now come from dedicated calls (plan 03); bibliography key is gone
    assert "bibliography" not in result, "bibliography key must be removed (renamed to references)"
    assert result["references"] == ["1. Smith et al. 2020.", "2. Doe et al. 2021."], (
        f"references must come from the dedicated call, got: {result.get('references')}"
    )
    assert result["figures"] == [{"label": "Fig. 1", "caption": "A figure."}], (
        f"figures must come from the dedicated call, got: {result.get('figures')}"
    )


@patch("scripts.ingest.urllib.request.urlopen")
def test_extract_with_llm_timeout(mock_urlopen):
    """INGEST-01 LLM path: _extract_with_llm re-raises urllib.error.URLError when Ollama
    is unreachable — fires on first call (section discovery) — fail-fast propagates immediately."""
    mock_urlopen.side_effect = urllib.error.URLError("connection timeout")

    with pytest.raises(urllib.error.URLError):
        _extract_with_llm(["Sample paper text"])


@patch("scripts.ingest._extract_with_llm")
def test_extract_paper_fails_on_ollama_unreachable(mock_llm, sample_pdf_path):
    """extract_paper must exit(1) when Ollama is unreachable — LLM is required."""
    import urllib.error
    mock_llm.side_effect = urllib.error.URLError("connection refused")

    with pytest.raises(SystemExit) as exc_info:
        extract_paper(sample_pdf_path)
    assert exc_info.value.code == 1


@patch("scripts.ingest._extract_with_llm")
def test_extract_paper_fails_on_timeout(mock_llm, sample_pdf_path):
    """extract_paper must exit(1) when _extract_with_llm raises TimeoutError."""
    mock_llm.side_effect = TimeoutError("timed out waiting for Ollama")

    with pytest.raises(SystemExit) as exc_info:
        extract_paper(sample_pdf_path)
    assert exc_info.value.code == 1, (
        f"extract_paper must exit with code 1 on TimeoutError, got: {exc_info.value.code}"
    )

@patch("scripts.ingest.urllib.request.urlopen")
def test_extract_with_llm_section_discovery_object_form(mock_urlopen):
    """Whole-document sections call: when Ollama returns {"sections": [{title, body}]}
    (object form), the _LLMSection validation branch is exercised correctly."""

    mock_urlopen.side_effect = [
        _make_mock_response({                                                    # Call 1: metadata
            "title": "Test Paper",
            "authors": ["A. Author"],
            "abstract": "An abstract.",
        }),
        _make_mock_response({"sections": [                                        # Call 2: sections
            {"title": "Introduction", "body": "Intro text."},
            {"title": "Conclusion", "body": "Conclusion text."},
        ]}),
    ]

    result = _extract_with_llm(["Sample paper text"])

    assert result["title"] == "Test Paper", f"title mismatch: {result['title']}"
    assert len(result["sections"]) == 2, (
        f"sections must have 2 entries when sections call returns object form; got: {len(result['sections'])}"
    )
    assert result["sections"][0]["title"] == "Introduction", (
        f"sections[0] title mismatch: {result['sections'][0]['title']}"
    )


# ---------------------------------------------------------------------------
# Plan 01.1-02: whole-document section extraction (heading-less Introduction)
# ---------------------------------------------------------------------------

@patch("scripts.ingest.urllib.request.urlopen")
def test_sections_introduction_captured_without_heading(mock_urlopen):
    """The whole-document sections call captures the Introduction with a non-empty
    body even when the input text has NO literal 'Introduction' heading."""
    mock_urlopen.side_effect = [
        _make_mock_response({                                   # Call 1: metadata
            "title": "Test Paper",
            "authors": ["A. Author"],
            "abstract": "An abstract.",
        }),
        _make_mock_response({"sections": [                       # Call 2: sections
            {"title": "Introduction", "body": "Sepsis poses a significant global health challenge."},
            {"title": "Materials and methods", "body": "Approved by the review board."},
        ]}),
    ]
    # Input text deliberately has NO "Introduction" heading — just abstract then body.
    result = _extract_with_llm([
        "Abstract\nThe long-term consequences of sepsis...\n\n"
        "Sepsis poses a significant global health challenge.\n\n"
        "Materials and methods\nApproved by the review board."
    ])
    intro = [s for s in result["sections"] if s["title"] == "Introduction"]
    assert intro, f"an 'Introduction' section must be present, got titles: {[s['title'] for s in result['sections']]}"
    assert intro[0]["body"].strip(), "the Introduction body must be non-empty despite no heading"


@patch("scripts.ingest.urllib.request.urlopen")
def test_sections_all_bodies_non_empty(mock_urlopen):
    """Every section returned by extraction has a non-empty body (no WR-03 empty-body path)."""
    mock_urlopen.side_effect = [
        _make_mock_response({                                   # Call 1: metadata
            "title": "Test Paper",
            "authors": ["A. Author"],
            "abstract": "An abstract.",
        }),
        _make_mock_response({"sections": [                       # Call 2: sections
            {"title": "Introduction", "body": "Intro body."},
            {"title": "Results", "body": "Results body."},
            {"title": "Conclusions", "body": "Conclusion body."},
        ]}),
    ]
    result = _extract_with_llm(["Some paper text"])
    assert len(result["sections"]) == 3
    assert all(s["body"].strip() for s in result["sections"]), (
        f"every section body must be non-empty, got: {result['sections']}"
    )


# ---------------------------------------------------------------------------
# Plan 01.1-03: dedicated references + deduplicated figures calls
# ---------------------------------------------------------------------------

def _meta_and_sections_responses():
    """Return the first two canned responses (metadata, sections) shared by the
    references/figures tests. Call sequence is metadata, sections, references, figures."""
    return [
        _make_mock_response({                       # Call 1: metadata
            "title": "Test Paper",
            "authors": ["A. Author"],
            "abstract": "An abstract.",
        }),
        _make_mock_response({"sections": [          # Call 2: sections (NO References title)
            {"title": "Introduction", "body": "Intro."},
            {"title": "Results", "body": "Results."},
        ]}),
    ]


@patch("scripts.ingest.urllib.request.urlopen")
def test_references_extracted_independent_of_sections(mock_urlopen):
    """The dedicated references call populates references even when section
    discovery returns NO 'References' title (decoupled from discovery)."""
    mock_urlopen.side_effect = _meta_and_sections_responses() + [
        _make_mock_response({"references": ["1. Reinhart, K. et al. 2017.", "2. Rudd, K. E. 2020."]}),  # Call 3
        _make_mock_response({"figures": []}),                                                          # Call 4
    ]
    result = _extract_with_llm(["paper text with no References heading in sections"])
    assert result["references"], "references must be populated independent of section discovery"
    assert result["references"][0].startswith("1."), (
        f"references[0] must start with '1.', got: {result['references'][0]}"
    )


@patch("scripts.ingest.urllib.request.urlopen")
def test_references_threshold_and_shape(mock_urlopen):
    """A references call returning 43 strings yields a list of >=40 non-empty
    strings whose first entry starts with '1.'."""
    refs = [f"{i}. Author {i} et al. (20{i:02d})." for i in range(1, 44)]
    mock_urlopen.side_effect = _meta_and_sections_responses() + [
        _make_mock_response({"references": refs}),   # Call 3
        _make_mock_response({"figures": []}),         # Call 4
    ]
    result = _extract_with_llm(["paper text"])
    assert len(result["references"]) >= 40, f"expected >=40 refs, got: {len(result['references'])}"
    assert all(isinstance(r, str) and r.strip() for r in result["references"]), "all refs non-empty str"
    assert result["references"][0].startswith("1."), "references[0] must start with '1.'"


@patch("scripts.ingest.urllib.request.urlopen")
def test_figures_dedup_no_duplicate_labels(mock_urlopen):
    """A figures call returning duplicate labels is deduplicated by label."""
    mock_urlopen.side_effect = _meta_and_sections_responses() + [
        _make_mock_response({"references": ["1. Ref."]}),                       # Call 3
        _make_mock_response({"figures": [                                       # Call 4: dup labels
            {"label": "Table 1", "caption": "short"},
            {"label": "Table 1", "caption": "a longer caption that wins"},
            {"label": "Fig. 1", "caption": "fig caption"},
        ]}),
    ]
    result = _extract_with_llm(["paper text"])
    labels = [f["label"] for f in result["figures"]]
    assert len(labels) == len(set(labels)), f"figure labels must be unique, got: {labels}"
    assert "Table 1" in labels and "Fig. 1" in labels


@patch("scripts.ingest.urllib.request.urlopen")
def test_figures_captions_non_empty(mock_urlopen):
    """All figures in a populated result have non-empty captions (empty ones dropped)."""
    mock_urlopen.side_effect = _meta_and_sections_responses() + [
        _make_mock_response({"references": ["1. Ref."]}),                       # Call 3
        _make_mock_response({"figures": [                                       # Call 4
            {"label": "Fig. 1", "caption": "real caption"},
            {"label": "", "caption": "no label — dropped"},
        ]}),
    ]
    result = _extract_with_llm(["paper text"])
    assert result["figures"], "figures must be populated"
    assert all(f["caption"].strip() for f in result["figures"]), (
        f"all figure captions must be non-empty, got: {result['figures']}"
    )
    assert all(f["label"].strip() for f in result["figures"]), "empty-label entries must be dropped"


# ---------------------------------------------------------------------------
# Plan 01.1-01 Task 2: doi + journal extraction (metadata schema call)
# ---------------------------------------------------------------------------

def test_doi_from_text_anchors_on_doi_org():
    """_doi_from_text prefers the doi.org-anchored front-matter DOI over a bare
    reference DOI elsewhere in the text (Pitfall 4)."""
    text = (
        "Scientific Reports | https://doi.org/10.1038/s41598-026-53619-9\n"
        "...\n"
        "35. Kim, S. M. et al. ... https://doi.org/10.1007/s11739-021-02847-0 (2021)."
    )
    assert _doi_from_text(text) == "10.1038/s41598-026-53619-9"


@patch("scripts.ingest.urllib.request.urlopen")
def test_doi_extracted_from_metadata_call(mock_urlopen):
    """The metadata LLM call returns a doi which flows into _extract_with_llm result.

    Call order (plan 02): metadata first, then ONE whole-document sections call.
    """
    mock_urlopen.side_effect = [
        _make_mock_response({                        # Call 1: metadata
            "title": "Test Paper",
            "authors": ["A. Author"],
            "abstract": "An abstract.",
            "year": 2026,
            "doi": "10.1038/s41598-026-53619-9",
            "journal": "Scientific Reports",
        }),
        _make_mock_response({"sections": []}),       # Call 2: sections (empty -> Body fallback)
    ]
    result = _extract_with_llm(["Some paper text https://doi.org/10.1038/s41598-026-53619-9"])
    assert result["doi"] == "10.1038/s41598-026-53619-9", (
        f"doi must flow from metadata call, got: {result.get('doi')}"
    )


@patch("scripts.ingest.urllib.request.urlopen")
def test_journal_extracted_from_metadata_call(mock_urlopen):
    """The metadata LLM call returns a journal which flows into the result."""
    mock_urlopen.side_effect = [
        _make_mock_response({                        # Call 1: metadata
            "title": "Test Paper",
            "authors": ["A. Author"],
            "abstract": "An abstract.",
            "year": 2026,
            "doi": "10.1038/s41598-026-53619-9",
            "journal": "Scientific Reports",
        }),
        _make_mock_response({"sections": []}),       # Call 2: sections (empty -> Body fallback)
    ]
    result = _extract_with_llm(["Some paper text"])
    assert result["journal"] == "Scientific Reports", (
        f"journal must flow from metadata call, got: {result.get('journal')}"
    )


@patch("scripts.ingest.urllib.request.urlopen")
def test_doi_reconcile_prefers_anchored(mock_urlopen, test_manuel2_pdf_path, monkeypatch):
    """When the LLM doi disagrees with the doi.org-anchored regex match,
    extract_paper prefers the anchored regex DOI (Pitfall 4 reconciliation)."""
    # Patch _extract_with_llm to return a WRONG doi (a reference DOI), and feed a
    # combined text whose front matter anchors the real one. extract_paper must
    # reconcile to the anchored DOI from the document text.
    def _stub_llm(pages_text):
        return {
            "title": "Test Paper",
            "authors": ["A. Author"],
            "abstract": "An abstract.",
            "sections": [{"title": "Introduction", "body": "Intro."}],
            "bibliography": None,
            "figures": None,
            "year": 2026,
            "doi": "10.1007/s11739-021-02847-0",   # a reference DOI — wrong paper DOI
            "journal": "Scientific Reports",
        }

    monkeypatch.setattr("scripts.ingest._extract_with_llm", _stub_llm)
    # Force the PDF text path to carry the anchored front-matter DOI.
    monkeypatch.setattr(
        "scripts.ingest._is_scanned", lambda *a, **k: False
    )

    class _Page:
        def extract_text(self, **kw):
            return (
                "Scientific Reports https://doi.org/10.1038/s41598-026-53619-9\n"
                "Body text.\n"
                "35. ref ... https://doi.org/10.1007/s11739-021-02847-0 (2021)."
            )

    class _PDF:
        pages = [_Page()]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("scripts.ingest.pdfplumber.open", lambda *a, **k: _PDF())

    result = extract_paper(test_manuel2_pdf_path)
    assert result["doi"] == "10.1038/s41598-026-53619-9", (
        f"reconciliation must prefer the doi.org-anchored DOI, got: {result['doi']}"
    )


# ---------------------------------------------------------------------------
# Plan 01.1-04: CLI unicode fix + DOI registry key + recorded/live scoring
# ---------------------------------------------------------------------------

def test_registry_doi_key(sample_pdf_path, _tmp_config, monkeypatch):
    """REG-01 key derivation: a paper whose extraction yields a DOI is written
    under that DOI key in an ISOLATED empty tmp registry.

    Scope note (Warning 3): this runs against the _tmp_config empty registry, so
    it proves DOI-key DERIVATION only — NOT real-registry sha256->DOI re-key,
    which is the 04-T2 human gate.
    """
    def _stub_llm(pages_text):
        return {
            "title": "A DOI Bearing Paper",
            "authors": ["A. Author"],
            "abstract": "ab",
            "sections": [{"title": "Introduction", "body": "Intro."}],
            "references": None,
            "figures": None,
            "year": 2026,
            "doi": "10.1038/s41598-026-53619-9",
            "journal": "Scientific Reports",
        }
    monkeypatch.setattr("scripts.ingest._extract_with_llm", _stub_llm)

    extract_paper(sample_pdf_path)

    config = _load_config(_tmp_config)
    with open(config["registry_path"], "r", encoding="utf-8") as f:
        registry = json.load(f)
    assert len(registry) == 1, f"expected exactly one entry, got: {list(registry.keys())}"
    key = next(iter(registry))
    assert key == "10.1038/s41598-026-53619-9", (
        f"DOI-bearing paper must be keyed by its DOI, got key: {key}"
    )


def test_recorded_output_scores_against_fixture(recorded_llm_output, expected_fixture,
                                                sample_pdf_path, _tmp_config, monkeypatch):
    """Deterministic recorded-replay: replay one captured real _extract_with_llm
    output through extract_paper and score the structural/threshold fields against
    the ground-truth fixture — no live Ollama needed.

    Skips until the recorded fixture is captured at the plan-04 checkpoint.
    """
    if recorded_llm_output is None:
        pytest.skip("recorded fixture not yet captured (plan 04 checkpoint)")

    from tests.conftest import score_against_expected
    monkeypatch.setattr("scripts.ingest._extract_with_llm", lambda pages_text: recorded_llm_output)

    result = extract_paper(sample_pdf_path)
    scores = score_against_expected(result, expected_fixture)
    # Structural/threshold fields must pass on the recorded output.
    for field in ("sections", "references", "figures"):
        assert scores.get(field) is True, (
            f"recorded-replay structural field '{field}' failed scoring: {scores}"
        )


@pytest.mark.skipif(not _ollama_available(), reason="requires live Ollama")
def test_cli_windows_unicode_round_trips(test_manuel2_pdf_path):
    """Criterion 6 / Pitfall 5: the CLI subprocess exits 0 and its stdout round-trips
    through json.loads even with non-cp1252 content (accented author 'Mårtensson')."""
    ingest_script = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "ingest.py")
    )
    result = subprocess.run(
        [sys.executable, ingest_script, "--pdf", test_manuel2_pdf_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
    )
    assert result.returncode == 0, (
        f"CLI must exit 0 on a cp1252 console; stderr: {result.stderr.strip()}"
    )
    paper = json.loads(result.stdout.strip())
    assert isinstance(paper, dict) and paper.get("title"), "CLI output must be valid PaperJSON"
    assert any("Mårtensson" in a for a in paper.get("authors", [])), (
        f"accented author must round-trip, got authors: {paper.get('authors')}"
    )


@pytest.mark.skipif(not _ollama_available(), reason="requires live Ollama")
def test_live_end_to_end_scores_against_fixture(test_manuel2_pdf_path, expected_fixture, _tmp_config):
    """Phase acceptance gate (opt-in): real extract_paper scored against the
    ground-truth fixture — exact-match metadata (incl. doi/journal) + structural
    sections/references/figures."""
    from tests.conftest import score_against_expected
    result = extract_paper(test_manuel2_pdf_path)
    scores = score_against_expected(result, expected_fixture)
    failed = [k for k, v in scores.items() if v is not True]
    assert not failed, f"live extraction failed scoring on fields: {failed} (full: {scores})"