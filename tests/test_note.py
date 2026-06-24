"""
Wave-0 test scaffold for note generation (NOTE-01, NOTE-02, NOTE-04).

These stubs import from scripts.note (which does not exist yet); collection-time
ImportError is the intended RED signal that Plan 02 resolves.  The analysis-skeleton
regression test (test_results_field_in_skeleton) imports from scripts.ingest and is
expected to pass immediately.

Run with:  python -m pytest tests/test_note.py -x
"""

import pytest

from scripts.ingest import _assemble_paperjson, SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def skeleton():
    """Minimal PaperJSON assembled from empty parsed + provenance dicts."""
    parsed = {
        "title": "Test Paper",
        "sections": [],
        "references": [],
        "metadata": {"title": "Test Paper"},
    }
    provenance = {
        "pdf_sha256": "abc123",
        "source_filename": "test.pdf",
        "mineru_version": "2.5",
        "backend": "hybrid_auto",
        "extracted_at": "2026-06-24T00:00:00Z",
        "normalizations_applied": [],
        "schema_version": SCHEMA_VERSION,
    }
    return _assemble_paperjson(parsed, provenance)


# ---------------------------------------------------------------------------
# D-09: analysis skeleton carries results field (regression lock)
# ---------------------------------------------------------------------------

def test_results_field_in_skeleton(skeleton):
    """Analysis skeleton must include a 'results' key with value None (D-09)."""
    assert "results" in skeleton["analysis"], (
        "analysis skeleton missing 'results' key after D-09 addition"
    )
    assert skeleton["analysis"]["results"] is None, (
        "analysis['results'] should be None in the empty skeleton"
    )


# ---------------------------------------------------------------------------
# NOTE-01: Analysis generation (Plan 02 turns this green)
# ---------------------------------------------------------------------------

def test_analysis_generation():
    """LLM generates summary, claims, methods_overview, results, limitations."""
    from scripts.note import generate_analysis  # noqa: F811
    pytest.fail("Wave-0 stub: scripts.note not yet implemented (Plan 02)")


# ---------------------------------------------------------------------------
# NOTE-02: Frontmatter YAML safety (Plan 02 turns this green)
# ---------------------------------------------------------------------------

def test_frontmatter_yaml_safe():
    """YAML frontmatter double-quotes string values; colons round-trip safely."""
    from scripts.note import render_frontmatter  # noqa: F811
    pytest.fail("Wave-0 stub: scripts.note not yet implemented (Plan 02)")


# ---------------------------------------------------------------------------
# NOTE-04: Fence stripping (Plan 02 turns this green)
# ---------------------------------------------------------------------------

def test_fence_stripping():
    """Strip markdown code fences before JSON parse (reuses _parse_extraction_response)."""
    from scripts.note import strip_fences  # noqa: F811
    pytest.fail("Wave-0 stub: scripts.note not yet implemented (Plan 02)")
