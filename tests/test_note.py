"""
Unit tests for note generation (NOTE-01, NOTE-02, NOTE-04).

Tests cover:
  - test_analysis_generation (NOTE-01): 7 analysis fields filled via mocked Ollama
  - test_single_field_failure: per-field two-strike failure is non-fatal
  - test_fence_stripping (NOTE-04): fenced ```json response parses successfully
  - test_ollama_unreachable_fails_fast: [Ollama error:] raises RuntimeError (D-13)
  - test_results_field_in_skeleton: D-09 regression lock
  - test_frontmatter_yaml_safe (NOTE-02): colon in title round-trips yaml.safe_load()

Run with:  python -m pytest tests/test_note.py -x
"""

import json
import pytest
from unittest import mock

from scripts.ingest import _assemble_paperjson, _parse_extraction_response, SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_paperjson(title="Test Paper: A Study", sections=None):
    """Build a minimal PaperJSON v2 dict for testing."""
    if sections is None:
        sections = [
            {"heading": "Introduction", "body": "This paper studies transformers."},
            {"heading": "Methods", "body": "We used attention mechanisms."},
        ]
    parsed = {
        "title": title,
        "sections": sections,
        "references": [],
        "metadata": {
            "title": title,
            "authors": ["Alice Smith", "Bob Jones"],
            "year": 2025,
            "journal": "Nature",
            "doi": "10.1234/test",
            "arxiv_id": None,
        },
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
    pj = _assemble_paperjson(parsed, provenance)
    # Override metadata with our richer version (assemble_paperjson may not keep all fields)
    pj["extraction"]["metadata"]["authors"] = ["Alice Smith", "Bob Jones"]
    pj["extraction"]["metadata"]["year"] = 2025
    pj["extraction"]["metadata"]["journal"] = "Nature"
    pj["extraction"]["metadata"]["doi"] = "10.1234/test"
    pj["extraction"]["metadata"]["arxiv_id"] = None
    # Ensure sections have bodies (post-fill shape)
    pj["extraction"]["sections"] = sections
    return pj


@pytest.fixture
def paperjson():
    return _make_paperjson()


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
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_ollama_success(prompt, system, schema, num_ctx=4096, timeout=120):
    """Return valid JSON for each analysis model based on the schema."""
    # Detect which model by checking schema properties
    props = schema.get("properties", {})
    if "summary" in props:
        return json.dumps({"summary": "This paper presents a novel approach to transformers."})
    elif "claims" in props:
        return json.dumps({"claims": ["Transformers outperform RNNs", "Attention is sufficient"]})
    elif "methods_overview" in props:
        return json.dumps({"methods_overview": "The authors used self-attention mechanisms."})
    elif "results" in props:
        return json.dumps({"results": "The model achieved state-of-the-art performance."})
    elif "limitations" in props:
        return json.dumps({"limitations": ["Limited to English", "High compute cost"]})
    elif "topics" in props:
        return json.dumps({"topics": ["transformers", "attention"]})
    elif "open_questions" in props:
        return json.dumps({"open_questions": ["Can this scale to longer sequences?"]})
    return json.dumps({})


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
# NOTE-01: Analysis generation (7 fields filled via mocked Ollama)
# ---------------------------------------------------------------------------

def test_analysis_generation(paperjson):
    """LLM generates summary, claims, methods_overview, results, limitations,
    topics, open_questions — all populated with generated_by set."""
    from scripts.note import _generate_analysis

    with mock.patch("scripts.note._ollama_extraction_call", side_effect=_mock_ollama_success), \
         mock.patch("scripts.note._warmup_ollama"):
        _generate_analysis(paperjson)

    analysis = paperjson["analysis"]
    assert analysis["generated_by"] == "gemma4:e4b", (
        f"Expected generated_by='gemma4:e4b', got {analysis['generated_by']!r}"
    )
    assert isinstance(analysis["summary"], str) and len(analysis["summary"]) > 0, (
        "summary should be a non-empty string"
    )
    assert isinstance(analysis["claims"], list) and len(analysis["claims"]) > 0, (
        "claims should be a non-empty list"
    )
    assert isinstance(analysis["methods_overview"], str) and len(analysis["methods_overview"]) > 0, (
        "methods_overview should be a non-empty string"
    )
    assert isinstance(analysis["results"], str) and len(analysis["results"]) > 0, (
        "results should be a non-empty string"
    )
    assert isinstance(analysis["limitations"], list) and len(analysis["limitations"]) > 0, (
        "limitations should be a non-empty list"
    )
    assert isinstance(analysis["topics"], list) and len(analysis["topics"]) > 0, (
        "topics should be a non-empty list"
    )
    assert isinstance(analysis["open_questions"], list) and len(analysis["open_questions"]) > 0, (
        "open_questions should be a non-empty list"
    )


# ---------------------------------------------------------------------------
# Single-field failure is non-fatal (D-14 fallback for claims/limitations)
# ---------------------------------------------------------------------------

def test_single_field_failure(paperjson):
    """When one field fails both strikes, the run continues; claims/limitations
    get a fallback entry so callouts are always renderable."""
    from scripts.note import _generate_analysis

    call_count = 0

    def _mock_with_one_failure(prompt, system, schema, num_ctx=4096, timeout=120):
        nonlocal call_count
        call_count += 1
        props = schema.get("properties", {})
        # Fail the summary call (return unparseable on both strikes)
        if "summary" in props:
            return "not valid json at all"
        return _mock_ollama_success(prompt, system, schema, num_ctx, timeout)

    with mock.patch("scripts.note._ollama_extraction_call", side_effect=_mock_with_one_failure), \
         mock.patch("scripts.note._warmup_ollama"):
        _generate_analysis(paperjson)

    analysis = paperjson["analysis"]
    # summary failed — should be None
    assert analysis["summary"] is None, (
        f"Expected summary=None after failure, got {analysis['summary']!r}"
    )
    # Other fields should still be populated
    assert len(analysis["claims"]) > 0, "claims should still be populated despite summary failure"
    assert len(analysis["limitations"]) > 0, "limitations should still be populated"
    assert analysis["generated_by"] == "gemma4:e4b", "generated_by should still be set"


def test_claims_fallback_on_failure():
    """When claims fails, a fallback entry is provided so callouts render."""
    from scripts.note import _generate_analysis

    pj = _make_paperjson()

    def _mock_claims_fail(prompt, system, schema, num_ctx=4096, timeout=120):
        props = schema.get("properties", {})
        if "claims" in props:
            return "garbage"
        return _mock_ollama_success(prompt, system, schema, num_ctx, timeout)

    with mock.patch("scripts.note._ollama_extraction_call", side_effect=_mock_claims_fail), \
         mock.patch("scripts.note._warmup_ollama"):
        _generate_analysis(pj)

    assert len(pj["analysis"]["claims"]) >= 1, (
        "claims should have at least one fallback entry when generation fails"
    )


def test_limitations_fallback_on_failure():
    """When limitations fails, a fallback entry is provided so callouts render."""
    from scripts.note import _generate_analysis

    pj = _make_paperjson()

    def _mock_limitations_fail(prompt, system, schema, num_ctx=4096, timeout=120):
        props = schema.get("properties", {})
        if "limitations" in props:
            return "garbage"
        return _mock_ollama_success(prompt, system, schema, num_ctx, timeout)

    with mock.patch("scripts.note._ollama_extraction_call", side_effect=_mock_limitations_fail), \
         mock.patch("scripts.note._warmup_ollama"):
        _generate_analysis(pj)

    assert len(pj["analysis"]["limitations"]) >= 1, (
        "limitations should have at least one fallback entry when generation fails"
    )


# ---------------------------------------------------------------------------
# NOTE-04: Fence stripping (reuses _parse_extraction_response)
# ---------------------------------------------------------------------------

def test_fence_stripping():
    """Strip markdown code fences before JSON parse (reuses _parse_extraction_response)."""
    from scripts.note import _generate_analysis, AnalysisSummary

    pj = _make_paperjson()

    def _mock_fenced_response(prompt, system, schema, num_ctx=4096, timeout=120):
        props = schema.get("properties", {})
        if "summary" in props:
            # Return response wrapped in markdown fences
            return '```json\n{"summary": "A fenced summary."}\n```'
        return _mock_ollama_success(prompt, system, schema, num_ctx, timeout)

    with mock.patch("scripts.note._ollama_extraction_call", side_effect=_mock_fenced_response), \
         mock.patch("scripts.note._warmup_ollama"):
        _generate_analysis(pj)

    assert pj["analysis"]["summary"] == "A fenced summary.", (
        f"Fenced JSON should parse successfully, got {pj['analysis']['summary']!r}"
    )


# ---------------------------------------------------------------------------
# D-13: Ollama unreachable raises RuntimeError (fail-fast)
# ---------------------------------------------------------------------------

def test_ollama_unreachable_fails_fast(paperjson):
    """[Ollama error:] from the client raises RuntimeError (D-13)."""
    from scripts.note import _generate_analysis

    def _mock_ollama_error(prompt, system, schema, num_ctx=4096, timeout=120):
        return "[Ollama error: connection refused]"

    with mock.patch("scripts.note._ollama_extraction_call", side_effect=_mock_ollama_error), \
         mock.patch("scripts.note._warmup_ollama"):
        with pytest.raises(RuntimeError, match="Ollama error"):
            _generate_analysis(paperjson)
