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


# ---------------------------------------------------------------------------
# NOTE-02: Frontmatter YAML safety — colon in title round-trips (SC2)
# ---------------------------------------------------------------------------

def test_frontmatter_yaml_safe():
    """YAML frontmatter double-quotes string values; colons round-trip safely."""
    import yaml
    from scripts.note import _render_frontmatter

    meta = {
        "title": "Attention: All You Need",
        "authors": ["Alice O'Brien", "Bob Jones-Smith"],
        "year": 2025,
        "journal": "Nature: Methods",
        "doi": "10.1234/test",
        "arxiv_id": None,
    }
    analysis = {
        "topics": ["transformers", "attention mechanisms"],
    }

    fm = _render_frontmatter(meta, analysis)

    # Must start and end with ---
    assert fm.startswith("---"), "frontmatter must start with ---"
    assert "---" in fm[3:], "frontmatter must end with ---"

    # Extract YAML content between fences
    lines = fm.strip().split("\n")
    assert lines[0] == "---", "first line must be ---"
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i] == "---":
            end_idx = i
            break
    assert end_idx is not None, "closing --- not found"
    yaml_text = "\n".join(lines[1:end_idx])

    # Must round-trip through yaml.safe_load
    loaded = yaml.safe_load(yaml_text)
    assert loaded is not None, "yaml.safe_load returned None"
    assert loaded["title"] == "Attention: All You Need", (
        f"title mismatch: {loaded['title']!r}"
    )
    assert loaded["year"] == 2025, f"year mismatch: {loaded['year']!r}"


# ---------------------------------------------------------------------------
# Render note: section headers and callouts (NOTE-02 / SC3)
# ---------------------------------------------------------------------------

def test_render_note_sections_and_callouts():
    """_render_note contains required sections and exactly one of each callout."""
    from scripts.note import _render_note

    pj = _make_paperjson()
    pj["analysis"]["summary"] = "A summary of the paper."
    pj["analysis"]["claims"] = ["Key contribution 1", "Key contribution 2"]
    pj["analysis"]["methods_overview"] = "Used attention."
    pj["analysis"]["results"] = "State of the art."
    pj["analysis"]["limitations"] = ["English only", "Compute heavy"]
    pj["analysis"]["topics"] = ["transformers"]
    pj["analysis"]["open_questions"] = ["Scalability?"]
    pj["analysis"]["generated_by"] = "gemma4:e4b"

    rendered = _render_note(pj)

    # Required section headers
    for header in ["## Summary", "## Key Findings", "## Methodology",
                   "## Results", "## Limitations", "## My Notes"]:
        assert header in rendered, f"Missing section header: {header}"

    # Exactly one [!important] and one [!warning] callout
    assert rendered.count("[!important]") == 1, (
        f"Expected exactly 1 [!important] callout, got {rendered.count('[!important]')}"
    )
    assert rendered.count("[!warning]") == 1, (
        f"Expected exactly 1 [!warning] callout, got {rendered.count('[!warning]')}"
    )


# ---------------------------------------------------------------------------
# Sanitize filename — Windows-illegal chars stripped (T-02-03)
# ---------------------------------------------------------------------------

def test_sanitize_filename():
    """_sanitize_filename strips all Windows-illegal chars and caps length."""
    from scripts.note import _sanitize_filename

    result = _sanitize_filename('A/B:C*?"<>|D')
    illegal = set('/\\:*?"<>|')
    for ch in result:
        assert ch not in illegal, f"Illegal char {ch!r} found in sanitized: {result!r}"
    assert len(result) > 0, "sanitized filename should not be empty"

    # Empty input -> Untitled
    assert _sanitize_filename("") == "Untitled", "empty title should become Untitled"
    assert _sanitize_filename("???") == "Untitled", "all-illegal title should become Untitled"

    # Length cap
    long_title = "A" * 300
    assert len(_sanitize_filename(long_title)) <= 200, "filename exceeds max_length"


# ---------------------------------------------------------------------------
# generate_note: vault write via obsidian-cli (mocked)
# ---------------------------------------------------------------------------

def test_generate_note_creates_note():
    """generate_note calls create_note with correct args on a new note."""
    from scripts.note import generate_note

    pj = _make_paperjson()
    config = {"vault_name": "test-vault"}

    with mock.patch("scripts.note._generate_analysis") as mock_analysis, \
         mock.patch("scripts.note._render_note", return_value="# Test Note\n") as mock_render, \
         mock.patch("scripts.note.preflight", return_value=True), \
         mock.patch("scripts.note.note_exists", return_value=False), \
         mock.patch("scripts.note.create_note", return_value="Created: Papers/Test Paper A Study.md") as mock_create:

        result = generate_note(pj, config, force=False)

    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args
    # path must start with Papers/ and contain no ..
    path_arg = call_kwargs[1].get("path", call_kwargs[0][0] if call_kwargs[0] else "")
    assert path_arg.startswith("Papers/"), f"path should start with Papers/, got {path_arg!r}"
    assert ".." not in path_arg, f"path should not contain '..', got {path_arg!r}"
    # overwrite should be False (default)
    overwrite_arg = call_kwargs[1].get("overwrite", call_kwargs[0][3] if len(call_kwargs[0]) > 3 else False)
    assert overwrite_arg is False, "overwrite should be False when force=False"


def test_generate_note_skips_existing():
    """generate_note skips without calling create_note or _generate_analysis
    when note exists and force=False (zero LLM calls on existing note)."""
    from scripts.note import generate_note

    pj = _make_paperjson()
    config = {"vault_name": "test-vault"}

    with mock.patch("scripts.note._generate_analysis") as mock_analysis, \
         mock.patch("scripts.note._render_note", return_value="# Test\n"), \
         mock.patch("scripts.note.preflight", return_value=True), \
         mock.patch("scripts.note.note_exists", return_value=True), \
         mock.patch("scripts.note.create_note") as mock_create:

        result = generate_note(pj, config, force=False)

    mock_create.assert_not_called()
    mock_analysis.assert_not_called()
    # Must return the path, not an error
    assert result.startswith("Papers/"), (
        f"Expected path starting with Papers/, got {result!r}"
    )


def test_generate_note_force_overwrites():
    """generate_note calls create_note with overwrite=True when force=True."""
    from scripts.note import generate_note

    pj = _make_paperjson()
    config = {"vault_name": "test-vault"}

    with mock.patch("scripts.note._generate_analysis"), \
         mock.patch("scripts.note._render_note", return_value="# Test\n"), \
         mock.patch("scripts.note.preflight", return_value=True), \
         mock.patch("scripts.note.note_exists", return_value=True), \
         mock.patch("scripts.note.create_note", return_value="Created") as mock_create:

        result = generate_note(pj, config, force=True)

    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args
    overwrite_arg = call_kwargs[1].get("overwrite", False)
    assert overwrite_arg is True, "overwrite should be True when force=True"


def test_generate_note_missing_vault_name():
    """generate_note returns error when vault_name is missing from config."""
    from scripts.note import generate_note

    pj = _make_paperjson()
    config = {}  # no vault_name

    result = generate_note(pj, config)
    assert result.startswith("[note error:"), (
        f"Expected [note error: prefix, got {result!r}"
    )


def test_generate_note_preflight_failure():
    """generate_note returns error when preflight fails."""
    from scripts.note import generate_note

    pj = _make_paperjson()
    config = {"vault_name": "test-vault"}

    with mock.patch("scripts.note.preflight", return_value=False):
        result = generate_note(pj, config)

    assert result.startswith("[note error:"), (
        f"Expected [note error: prefix, got {result!r}"
    )


# ---------------------------------------------------------------------------
# Plan 03: Standalone note.py default cache resolution (D-07)
# ---------------------------------------------------------------------------


def test_standalone_default_cache_resolution(tmp_path):
    """When --paperjson is omitted and --stem is given, note.py resolves .paperjson_cache/<stem>.json."""
    import pathlib

    # Create a cache file at .paperjson_cache/<stem>.json
    cache_dir = tmp_path / ".paperjson_cache"
    cache_dir.mkdir()
    pj = _make_paperjson()
    cache_file = cache_dir / "my_paper.json"
    cache_file.write_text(json.dumps(pj), encoding="utf-8")

    # Import _resolve_paperjson_path (to be added in GREEN phase)
    from scripts.note import _resolve_paperjson_path

    resolved = _resolve_paperjson_path(stem="my_paper", cache_dir=str(cache_dir))
    assert resolved == str(cache_file), (
        f"Expected resolved path to be {cache_file}, got {resolved!r}"
    )


# ---------------------------------------------------------------------------
# Plan 05: Subprocess regression test — direct `python scripts/note.py` import
# ---------------------------------------------------------------------------


def test_direct_invocation_import_resolves():
    """Running `python scripts/note.py --help` must exit 0 with no ImportError (Defect 1 guard)."""
    import pathlib
    import subprocess
    import sys

    repo_root = str(pathlib.Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "scripts/note.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"scripts/note.py --help exited {result.returncode}; stderr:\n{result.stderr}"
    )
    assert "ImportError" not in result.stderr, (
        f"ImportError in stderr:\n{result.stderr}"
    )
    assert "cannot import name" not in result.stderr, (
        f"'cannot import name' in stderr:\n{result.stderr}"
    )
