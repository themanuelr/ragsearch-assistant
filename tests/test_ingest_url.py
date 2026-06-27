"""
Unit tests for scripts/ingest.py — Phase 3 web ingestion contract (RED baseline).

All 12 tests reference functions not yet implemented in scripts/ingest.py; they are
RED by design and turn GREEN incrementally as each Phase 3 wave lands its slice.

Import strategy: `import scripts.ingest as ing` at module level, then reference
new functions as `ing._function_name(...)` so that a missing function raises
AttributeError only inside its own test, not at collection time.
Top-level from-imports are limited to symbols that already exist.

Run with:  python -m pytest tests/test_ingest_url.py -x
"""

import pathlib
import pytest

import scripts.ingest as ing

# These symbols already exist in scripts.ingest — safe to from-import.
from scripts.ingest import (
    _assemble_paperjson,
    _registry_key,
    _syntactic_doi_valid,
    SCHEMA_VERSION,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "sample_defuddle.md"
THIN_FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "sample_defuddle_thin.md"


@pytest.fixture(scope="module")
def defuddle_md():
    return FIXTURE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def thin_md():
    return THIN_FIXTURE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def parsed(defuddle_md):
    return ing._parse_defuddle_markdown(defuddle_md)


# ---------------------------------------------------------------------------
# 12 contract tests
# ---------------------------------------------------------------------------

def test_parse_defuddle_markdown(parsed):
    """Parser yields non-empty sections list; every block has type==text with non-empty
    display and plain; title is the H1 text (without the leading '# ')."""
    assert parsed["sections"], "Expected at least one section"
    assert parsed["title"], "Expected a non-empty title extracted from H1"
    assert "# " not in parsed["title"], "title should not include the '# ' prefix"

    for section in parsed["sections"]:
        for block in section.get("blocks", []):
            assert block["type"] == "text", (
                f"Expected block type 'text', got {block['type']!r}"
            )
            assert block.get("display"), "Expected non-empty 'display' in block"
            assert block.get("plain"), "Expected non-empty 'plain' in block"


def test_parse_no_headings():
    """No-heading markdown returns exactly one section (unlabeled fallback) of text blocks."""
    result = ing._parse_defuddle_markdown("Just one paragraph of text with no headings at all.")
    assert len(result["sections"]) == 1, (
        f"Expected exactly 1 section for heading-free input, got {len(result['sections'])}"
    )
    for block in result["sections"][0].get("blocks", []):
        assert block["type"] == "text", (
            f"Expected text block in fallback section, got {block['type']!r}"
        )


def test_parse_preserves_markers(parsed):
    """Citation marker [^ and dollar-delimited LaTeX survive in block display fields.

    Note: _build_plain strips $...$ delimiters by design, so we assert against
    'display', not 'plain'.
    """
    all_display = " ".join(
        block.get("display", "")
        for section in parsed["sections"]
        for block in section.get("blocks", [])
    )
    assert "[^" in all_display, "Expected [^ footnote marker to survive in block display"
    assert "$" in all_display, "Expected $ inline LaTeX delimiter to survive in block display"


def test_rewrite_url_arxiv():
    """_rewrite_url maps arxiv.org/abs/<id> to arxiv.org/html/<id>."""
    assert ing._rewrite_url("https://arxiv.org/abs/1706.03762", {}) == (
        "https://arxiv.org/html/1706.03762"
    ), "Plain arXiv /abs/ URL should map to /html/"

    versioned = ing._rewrite_url("https://arxiv.org/abs/1706.03762v7", {})
    assert versioned == "https://arxiv.org/html/1706.03762v7", (
        "Versioned arXiv /abs/<id>v7 URL should map to /html/<id>v7"
    )


def test_rewrite_url_journal():
    """_rewrite_url leaves journal (nature.com) URLs unchanged."""
    url = "https://www.nature.com/articles/nature12373"
    assert ing._rewrite_url(url, {}) == url, (
        "Non-arXiv journal URL should be returned unchanged"
    )


def test_rewrite_url_pubmed(monkeypatch):
    """_rewrite_url returns PubMed URL unchanged when PMC lookup returns None (fail-open)."""
    monkeypatch.setattr(ing, "_resolve_pubmed_to_pmc", lambda pmid, config: None)
    url = "https://pubmed.ncbi.nlm.nih.gov/12345"
    result = ing._rewrite_url(url, {})
    assert result == url, (
        f"Expected PubMed URL returned unchanged when PMC lookup fails, got {result!r}"
    )


def test_run_defuddle_empty(monkeypatch):
    """_run_defuddle raises RuntimeError when subprocess returns empty/whitespace stdout."""
    class _FakeResult:
        returncode = 0
        stdout = "   \n  "

    monkeypatch.setattr(ing.subprocess, "run", lambda *a, **kw: _FakeResult())
    with pytest.raises(RuntimeError):
        ing._run_defuddle("https://x.example.com", "defuddle", timeout=5)


def test_body_too_thin(defuddle_md, thin_md):
    """_web_body_too_thin is True for the paywall stub and False for the full-text fixture."""
    thin_parsed = ing._parse_defuddle_markdown(thin_md)
    thin_provenance: dict = {}
    thin_skeleton = _assemble_paperjson(thin_parsed, thin_provenance)
    assert ing._web_body_too_thin(thin_skeleton, 2000) is True, (
        "Paywall-stub skeleton should be too thin at threshold 2000"
    )

    full_parsed = ing._parse_defuddle_markdown(defuddle_md)
    full_provenance: dict = {}
    full_skeleton = _assemble_paperjson(full_parsed, full_provenance)
    assert ing._web_body_too_thin(full_skeleton, 2000) is False, (
        "Full-text skeleton should NOT be too thin at threshold 2000"
    )


def test_web_cache_stem():
    """_web_cache_stem returns 'arxiv-<id>' for arXiv URLs and 'url-...' for others."""
    stem = ing._web_cache_stem("https://arxiv.org/abs/1706.03762")
    assert stem == "arxiv-1706.03762", (
        f"Expected 'arxiv-1706.03762', got {stem!r}"
    )

    other_stem = ing._web_cache_stem("https://www.nature.com/articles/nature12373")
    assert other_stem.startswith("url-"), (
        f"Expected stem starting with 'url-' for non-arXiv URL, got {other_stem!r}"
    )


def test_web_provenance_shape():
    """_web_provenance returns a dict with the required keys and correct constant values."""
    prov = ing._web_provenance(
        "https://arxiv.org/abs/x",
        "https://arxiv.org/html/x",
        "body text",
    )
    assert "source_url" in prov, "Expected 'source_url' key in provenance"
    assert "fetched_url" in prov, "Expected 'fetched_url' key in provenance"
    assert "content_sha256" in prov, "Expected 'content_sha256' key in provenance"
    assert len(prov["content_sha256"]) == 64, "Expected 64-hex-char SHA-256 digest"
    assert all(c in "0123456789abcdef" for c in prov["content_sha256"]), (
        "content_sha256 must be lowercase hex"
    )
    assert prov.get("backend") == "defuddle", (
        f"Expected backend 'defuddle', got {prov.get('backend')!r}"
    )
    assert prov.get("schema_version") == SCHEMA_VERSION, (
        f"Expected schema_version {SCHEMA_VERSION}, got {prov.get('schema_version')!r}"
    )
    assert prov.get("normalizations_applied") == ["llm_fill"], (
        f"Expected normalizations_applied == ['llm_fill'], got {prov.get('normalizations_applied')!r}"
    )


def test_web_doi_key_fallback():
    """_web_doi_key_fallback extracts (doi, arxiv_id) from URL patterns."""
    doi, arxiv_id = ing._web_doi_key_fallback("https://arxiv.org/abs/1706.03762", None)
    assert doi is None, f"Expected doi=None for arXiv URL, got {doi!r}"
    assert arxiv_id == "1706.03762", (
        f"Expected arxiv_id='1706.03762', got {arxiv_id!r}"
    )

    doi2, arxiv_id2 = ing._web_doi_key_fallback("https://doi.org/10.1038/nature12373", None)
    assert doi2 == "10.1038/nature12373", (
        f"Expected doi='10.1038/nature12373', got {doi2!r}"
    )
    assert arxiv_id2 is None, f"Expected arxiv_id=None for DOI URL, got {arxiv_id2!r}"


def test_registry_key_convergence():
    """_registry_key on web-derived metadata equals _registry_key on PDF-derived metadata
    for the same paper: arXiv id wins when no DOI; DOI wins when present."""
    # Case 1: arXiv URL, no DOI probe — fallback gives (None, "1706.03762")
    doi, arxiv_id = ing._web_doi_key_fallback("https://arxiv.org/abs/1706.03762", None)
    web_key = _registry_key({"doi": doi, "arxiv_id": arxiv_id, "title": None})
    pdf_key = _registry_key({"doi": None, "arxiv_id": "1706.03762", "title": None})
    assert web_key == pdf_key == "1706.03762", (
        f"Expected convergence on '1706.03762': web_key={web_key!r}, pdf_key={pdf_key!r}"
    )

    # Case 2: DOI URL — DOI becomes the registry key
    doi2, arxiv_id2 = ing._web_doi_key_fallback("https://doi.org/10.1038/nature12373", None)
    doi_key = _registry_key({"doi": doi2, "arxiv_id": arxiv_id2, "title": None})
    assert doi_key == "10.1038/nature12373", (
        f"Expected DOI key '10.1038/nature12373', got {doi_key!r}"
    )
