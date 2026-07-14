"""
Unit tests for note generation (NOTE-01, NOTE-02, NOTE-04).

Tests cover:
  - test_analysis_generation (NOTE-01): 7 analysis fields filled via mocked Ollama
  - test_single_field_failure: per-field two-strike failure is non-fatal
  - test_fence_stripping (NOTE-04): fenced ```json response parses successfully
  - test_ollama_unreachable_fails_fast: [Ollama error:] raises RuntimeError (D-13)
  - test_results_field_in_skeleton: D-09 regression lock
  - test_frontmatter_yaml_safe (NOTE-02): colon in title round-trips yaml.safe_load()
  - TestSlugifyTag (GAP-1 / NOTE-02): _slugify_tag restricts to Obsidian tag charset

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
    config = {"vault_path": "./.local/test-vault", "vault_name": "test-vault"}

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
    overwrite_arg = call_kwargs[1].get("overwrite", False)
    assert overwrite_arg is False, "overwrite should be False when force=False"


def test_generate_note_skips_existing():
    """generate_note skips without calling create_note or _generate_analysis
    when note exists and force=False (zero LLM calls on existing note)."""
    from scripts.note import generate_note

    pj = _make_paperjson()
    config = {"vault_path": "./.local/test-vault", "vault_name": "test-vault"}

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
    config = {"vault_path": "./.local/test-vault", "vault_name": "test-vault"}

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


def test_generate_note_missing_vault_path():
    """generate_note returns error when vault_path is missing from config."""
    from scripts.note import generate_note

    pj = _make_paperjson()
    config = {}  # no vault_path

    result = generate_note(pj, config)
    assert result.startswith("[note error:"), (
        f"Expected [note error: prefix, got {result!r}"
    )


def test_generate_note_preflight_failure():
    """generate_note returns error when preflight fails (vault root missing)."""
    from scripts.note import generate_note

    pj = _make_paperjson()
    config = {"vault_path": "./.local/test-vault", "vault_name": "test-vault"}

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


# ---------------------------------------------------------------------------
# Task 2b: Math-scoped LaTeX escape repair (_repair_math_escapes)
# ---------------------------------------------------------------------------

class TestRepairMathEscapes:
    """Tests for _repair_math_escapes and _repair_analysis_value (Task 2b).

    gemma4:e4b emits LaTeX with single backslashes in JSON values; the JSON
    decoder silently turns \\text → TAB+"ext", \\r → CR, \\n → LF.
    These tests verify that the repair reconstructs the intended backslash
    sequences within math spans only, leaving prose text unchanged.
    """

    def test_tab_in_math_span(self):
        """$<TAB>ext{Na}^+$ repaired to $\\text{Na}^+$."""
        from scripts.note import _repair_math_escapes
        # Real TAB character inside $...$: simulates \text decoded by JSON
        inp = "$" + "\t" + "ext{Na}^+$"
        assert _repair_math_escapes(inp) == "$\\text{Na}^+$"

    def test_cr_in_math_span(self):
        """$<CR>ightleftharpoons$ repaired to $\\rightleftharpoons$."""
        from scripts.note import _repair_math_escapes
        inp = "$" + "\r" + "ightleftharpoons$"
        assert _repair_math_escapes(inp) == "$\\rightleftharpoons$"

    def test_compound_tabs_in_math_span(self):
        """$<TAB>ext{Tl}_2<TAB>ext{SO}_4$ repaired to $\\text{Tl}_2\\text{SO}_4$."""
        from scripts.note import _repair_math_escapes
        inp = "$" + "\t" + "ext{Tl}_2" + "\t" + "ext{SO}_4$"
        assert _repair_math_escapes(inp) == "$\\text{Tl}_2\\text{SO}_4$"

    def test_lf_in_math_span(self):
        """$<LF>abla$ repaired to $\\nabla$."""
        from scripts.note import _repair_math_escapes
        inp = "$" + "\n" + "abla$"
        assert _repair_math_escapes(inp) == "$\\nabla$"

    def test_prose_newlines_outside_math_unchanged(self):
        """Real paragraph-break newlines OUTSIDE math spans are preserved unchanged."""
        from scripts.note import _repair_math_escapes
        inp = "First paragraph.\n\nSecond paragraph."
        assert _repair_math_escapes(inp) == inp

    def test_prose_newlines_with_math_mixed(self):
        """Prose newlines outside math are unchanged; control chars inside math are repaired."""
        from scripts.note import _repair_math_escapes
        inp = "Some text.\n\nThe ion $" + "\t" + "ext{Na}^+$ is reactive.\n\nMore text."
        expected = "Some text.\n\nThe ion $\\text{Na}^+$ is reactive.\n\nMore text."
        assert _repair_math_escapes(inp) == expected

    def test_no_dollar_sign_noop(self):
        """Text without any $ is returned unchanged (fast-path)."""
        from scripts.note import _repair_math_escapes
        inp = "No math here: just text with " + "\t" + " tabs and " + "\n" + " newlines."
        assert _repair_math_escapes(inp) == inp

    def test_repair_analysis_value_str(self):
        """_repair_analysis_value repairs a str value."""
        from scripts.note import _repair_analysis_value
        inp = "$" + "\t" + "ext{Na}^+$"
        assert _repair_analysis_value(inp) == "$\\text{Na}^+$"

    def test_repair_analysis_value_list(self):
        """_repair_analysis_value repairs each str element in a list."""
        from scripts.note import _repair_analysis_value
        inp = [
            "$" + "\t" + "ext{Na}^+$",
            "plain text (no dollar sign)",
            "$" + "\r" + "ightarrow$",
        ]
        result = _repair_analysis_value(inp)
        assert result[0] == "$\\text{Na}^+$"
        assert result[1] == "plain text (no dollar sign)"
        assert result[2] == "$\\rightarrow$"

    def test_repair_analysis_value_passthrough(self):
        """_repair_analysis_value passes through non-str/non-list values unchanged."""
        from scripts.note import _repair_analysis_value
        assert _repair_analysis_value(None) is None
        assert _repair_analysis_value(42) == 42

    def test_generate_analysis_wiring(self):
        """_generate_analysis repairs control chars inside $...$ in analysis fields.

        Mocks _ollama_extraction_call to return JSON with a literal TAB inside a
        math span for the summary field; asserts the stored analysis value has the
        TAB replaced by backslash+t (i.e. the repair is wired in _generate_analysis).
        """
        from scripts.note import _generate_analysis

        pj = _make_paperjson()

        def _mock_mangled_summary(prompt, system, schema, num_ctx=4096, timeout=120):
            props = schema.get("properties", {})
            if "summary" in props:
                # Literal TAB inside $...$: simulates gemma4:e4b \text decoded by JSON
                mangled = "The ion $" + "\t" + "ext{Na}^+$ is notable."
                return json.dumps({"summary": mangled})
            return _mock_ollama_success(prompt, system, schema, num_ctx, timeout)

        with mock.patch("scripts.note._ollama_extraction_call",
                        side_effect=_mock_mangled_summary), \
             mock.patch("scripts.note._warmup_ollama"):
            _generate_analysis(pj)

        summary = pj["analysis"]["summary"]
        # TAB inside $...$ must be repaired to backslash+t
        assert "\t" not in summary, (
            f"TAB char should be repaired in analysis summary; got: {summary!r}"
        )
        assert "\\text" in summary, (
            f"Expected '\\\\text' (backslash+text) in repaired summary; got: {summary!r}"
        )


# ---------------------------------------------------------------------------
# GAP-1 / NOTE-02: Obsidian tag charset restriction (_slugify_tag)
# ---------------------------------------------------------------------------

class TestSlugifyTag:
    """Tests for _slugify_tag helper (GAP-1 fix, closing 02-UAT.md GAP 1 severity major).

    Obsidian forbids parentheses, commas, spaces, and other non-word characters in
    tags.  _slugify_tag must collapse forbidden runs to a single hyphen, preserve
    forward slashes (nested-tag separator), and return None for topics that reduce
    to an empty string after stripping.
    """

    def test_parentheses_collapsed(self):
        """Parentheses and contained text run collapsed to hyphen; trailing hyphen stripped."""
        from scripts.note import _slugify_tag
        result = _slugify_tag("Cation-Chloride Cotransporter (CCC)")
        assert result == "cation-chloride-cotransporter-ccc", (
            f"Expected 'cation-chloride-cotransporter-ccc', got {result!r}"
        )

    def test_commas_collapsed(self):
        """Commas (and surrounding spaces) collapsed to a single hyphen."""
        from scripts.note import _slugify_tag
        result = _slugify_tag("Calcium, Sodium, Chloride")
        assert result == "calcium-sodium-chloride", (
            f"Expected 'calcium-sodium-chloride', got {result!r}"
        )

    def test_forward_slash_preserved(self):
        """Forward slash is kept as a nested-tag separator; spaces collapsed."""
        from scripts.note import _slugify_tag
        result = _slugify_tag("input/output gating")
        assert result == "input/output-gating", (
            f"Expected 'input/output-gating', got {result!r}"
        )

    def test_all_forbidden_returns_none(self):
        """A topic consisting only of forbidden characters returns None."""
        from scripts.note import _slugify_tag
        assert _slugify_tag("()") is None, "Expected None for '()'"
        assert _slugify_tag("!!!") is None, "Expected None for '!!!'"
        assert _slugify_tag("---") is None, "Expected None for '---'"

    def test_empty_string_returns_none(self):
        """Empty string input returns None."""
        from scripts.note import _slugify_tag
        assert _slugify_tag("") is None, "Expected None for empty string"

    def test_plain_topic_lowercased(self):
        """Plain word topic is lowercased with no changes."""
        from scripts.note import _slugify_tag
        assert _slugify_tag("Transformers") == "transformers"

    def test_spaces_become_hyphens(self):
        """Spaces are replaced by a single hyphen."""
        from scripts.note import _slugify_tag
        assert _slugify_tag("attention mechanisms") == "attention-mechanisms"

    def test_multiple_spaces_one_hyphen(self):
        """Multiple consecutive spaces collapse to a single hyphen."""
        from scripts.note import _slugify_tag
        result = _slugify_tag("ion   transport")
        assert result == "ion-transport", (
            f"Expected 'ion-transport', got {result!r}"
        )

    def test_leading_trailing_hyphens_stripped(self):
        """Leading/trailing hyphens that arise from stripping are removed."""
        from scripts.note import _slugify_tag
        result = _slugify_tag("(leading) word (trailing)")
        # After collapse: "-leading--word--trailing-" then strip leading/trailing '-'
        assert not result.startswith("-"), f"Result starts with '-': {result!r}"
        assert not result.endswith("-"), f"Result ends with '-': {result!r}"


class TestRenderFrontmatterTagSlugification:
    """Integration tests: _render_frontmatter emits only valid Obsidian tag slugs."""

    def test_forbidden_chars_removed_from_tags(self):
        """Topics with parentheses/commas produce clean Obsidian tag slugs."""
        import yaml
        from scripts.note import _render_frontmatter

        meta = {
            "title": "Ion Transport",
            "authors": ["Alice Smith"],
            "year": 2024,
        }
        analysis = {
            "topics": ["Cation-Chloride Cotransporter (CCC)", "Calcium, Sodium, Chloride"],
        }
        fm = _render_frontmatter(meta, analysis)

        # No parentheses or commas should appear in the tags block
        assert "(" not in fm, f"Parenthesis found in frontmatter: {fm!r}"
        assert "," not in fm, f"Comma found in frontmatter (outside authors): {fm!r}"

        # Must still be valid YAML
        lines = fm.strip().split("\n")
        end_idx = next(i for i in range(1, len(lines)) if lines[i] == "---")
        yaml_text = "\n".join(lines[1:end_idx])
        loaded = yaml.safe_load(yaml_text)
        tags = loaded.get("tags", [])
        assert "cation-chloride-cotransporter-ccc" in tags, (
            f"Expected slug not in tags: {tags!r}"
        )
        assert "calcium-sodium-chloride" in tags, (
            f"Expected slug not in tags: {tags!r}"
        )

    def test_slash_preserved_as_nested_tag(self):
        """Forward slashes in topics are preserved in the emitted tag slug."""
        from scripts.note import _render_frontmatter

        meta = {"title": "Test", "authors": [], "year": 2024}
        analysis = {"topics": ["input/output gating"]}
        fm = _render_frontmatter(meta, analysis)
        assert "input/output-gating" in fm, (
            f"Expected 'input/output-gating' in frontmatter, got: {fm!r}"
        )

    def test_all_empty_topics_omits_tags_key(self):
        """When every topic slugifies to empty/None, the tags: key is omitted entirely."""
        import yaml
        from scripts.note import _render_frontmatter

        meta = {"title": "Test", "authors": [], "year": 2024}
        analysis = {"topics": ["()", "!!!", "---"]}
        fm = _render_frontmatter(meta, analysis)

        # The word 'tags' should not appear in frontmatter at all
        lines = fm.strip().split("\n")
        end_idx = next(i for i in range(1, len(lines)) if lines[i] == "---")
        yaml_text = "\n".join(lines[1:end_idx])
        loaded = yaml.safe_load(yaml_text)
        assert "tags" not in loaded, (
            f"Expected no 'tags' key when all topics are empty, got: {loaded!r}"
        )

    def test_empty_topics_list_omits_tags_key(self):
        """Empty topics list -> no tags: key (pre-existing behaviour preserved)."""
        import yaml
        from scripts.note import _render_frontmatter

        meta = {"title": "Test", "authors": [], "year": 2024}
        analysis = {"topics": []}
        fm = _render_frontmatter(meta, analysis)

        lines = fm.strip().split("\n")
        end_idx = next(i for i in range(1, len(lines)) if lines[i] == "---")
        yaml_text = "\n".join(lines[1:end_idx])
        loaded = yaml.safe_load(yaml_text)
        assert "tags" not in loaded, (
            f"Expected no 'tags' key for empty topics, got: {loaded!r}"
        )

    def test_yaml_safe_load_roundtrip_with_forbidden_chars(self):
        """Topics with forbidden chars still produce yaml.safe_load-safe frontmatter."""
        import yaml
        from scripts.note import _render_frontmatter

        meta = {
            "title": "Test: Colon Safety",
            "authors": ["Author One"],
            "year": 2023,
        }
        analysis = {
            "topics": [
                "Cation-Chloride Cotransporter (CCC)",
                "Calcium, Sodium, Chloride",
                "input/output gating",
            ]
        }
        fm = _render_frontmatter(meta, analysis)

        lines = fm.strip().split("\n")
        end_idx = next(i for i in range(1, len(lines)) if lines[i] == "---")
        yaml_text = "\n".join(lines[1:end_idx])
        loaded = yaml.safe_load(yaml_text)

        assert loaded is not None, "yaml.safe_load returned None"
        tags = loaded.get("tags", [])
        assert len(tags) == 3, f"Expected 3 tags, got {tags!r}"
        # Verify cleaned slugs
        assert "cation-chloride-cotransporter-ccc" in tags
        assert "calcium-sodium-chloride" in tags
        assert "input/output-gating" in tags


# ---------------------------------------------------------------------------
# 260714-dpl: Tag canonicalization at ingest (dedupe near-duplicate tags)
# ---------------------------------------------------------------------------
#
# _canonicalize_tags(topics, config) canonicalizes proposed topic slugs
# against the LIVE vault tag vocabulary (scripts.link._build_topic_membership)
# via a local-LLM judgment call (_tag_canonicalization_call), auto-replacing
# true synonyms with the existing canonical slug, logging every merge, and
# failing open (returning proposed slugs unchanged) on any error / empty
# vocab / parse failure / non-vocab target.


class TestExistingTagVocabulary:
    """_existing_tag_vocabulary(config) — live-vault vocab source (D-02)."""

    def test_uses_build_topic_membership_keys(self, monkeypatch):
        """Returns the set of keys from scripts.link._build_topic_membership."""
        from scripts.note import _existing_tag_vocabulary

        def _fake_build_topic_membership(vault_root):
            return {"topic-a": ["paper1"], "topic-b": ["paper2"]}

        monkeypatch.setattr(
            "scripts.link._build_topic_membership", _fake_build_topic_membership
        )
        result = _existing_tag_vocabulary({"vault_path": "./.local/test-vault"})
        assert result == {"topic-a", "topic-b"}

    def test_fail_open_no_vault_path(self):
        """Missing vault_path -> empty set (fail-open), no raise."""
        from scripts.note import _existing_tag_vocabulary

        assert _existing_tag_vocabulary({}) == set()

    def test_fail_open_on_exception(self, monkeypatch):
        """Any exception from _build_topic_membership -> empty set (fail-open)."""
        from scripts.note import _existing_tag_vocabulary

        def _raise(vault_root):
            raise OSError("boom")

        monkeypatch.setattr("scripts.link._build_topic_membership", _raise)
        result = _existing_tag_vocabulary({"vault_path": "./.local/test-vault"})
        assert result == set()


class TestCanonicalizeTags:
    """_canonicalize_tags(topics, config) — merge synonyms, preserve distinct,
    safety guard, fail-open, dedupe (260714-dpl)."""

    def test_merges_synonym(self, monkeypatch, capsys):
        """A true synonym proposed slug is replaced by the existing canonical slug,
        and the merge is logged (old -> canonical)."""
        from scripts.note import _canonicalize_tags

        monkeypatch.setattr(
            "scripts.note._existing_tag_vocabulary",
            lambda config: {
                "cryo-electron-microscopy",
                "ion-transporters",
                "structural-biology",
            },
        )
        monkeypatch.setattr(
            "scripts.note._tag_canonicalization_call",
            lambda proposed, existing: {
                "cryo-electron-microscopy-cryo-em": "cryo-electron-microscopy"
            },
        )

        result = _canonicalize_tags(
            ["cryo-electron-microscopy-cryo-em"], {"vault_path": "x"}
        )

        assert result == ["cryo-electron-microscopy"]
        captured = capsys.readouterr()
        assert "tag-canonicalized" in captured.err, (
            f"Expected a merge log line on stderr, got: {captured.err!r}"
        )
        assert "cryo-electron-microscopy-cryo-em" in captured.err
        assert "cryo-electron-microscopy" in captured.err

    def test_preserves_distinct_topic(self, monkeypatch):
        """A genuinely distinct-but-related topic is NOT merged (LLM maps it to itself)."""
        from scripts.note import _canonicalize_tags

        monkeypatch.setattr(
            "scripts.note._existing_tag_vocabulary",
            lambda config: {"structural-biology"},
        )
        monkeypatch.setattr(
            "scripts.note._tag_canonicalization_call",
            lambda proposed, existing: {
                "structural-biology-of-transporters": "structural-biology-of-transporters"
            },
        )

        result = _canonicalize_tags(
            ["structural-biology-of-transporters"], {"vault_path": "x"}
        )

        assert result == ["structural-biology-of-transporters"]

    def test_safety_guard_rejects_non_vocab_target(self, monkeypatch):
        """Even if the LLM returns a target NOT in the existing vocab, the proposed
        slug is kept unchanged (never apply a hallucinated/invented target)."""
        from scripts.note import _canonicalize_tags

        monkeypatch.setattr(
            "scripts.note._existing_tag_vocabulary",
            lambda config: {"ion-transporters"},
        )
        monkeypatch.setattr(
            "scripts.note._tag_canonicalization_call",
            lambda proposed, existing: {"ion-cotransporters": "hallucinated-tag"},
        )

        result = _canonicalize_tags(["ion-cotransporters"], {"vault_path": "x"})

        assert result == ["ion-cotransporters"]

    def test_fail_open_on_llm_call_error(self, monkeypatch):
        """When the canonicalization call raises (simulated Ollama down), the
        proposed slugs are returned unchanged and no exception propagates."""
        from scripts.note import _canonicalize_tags

        monkeypatch.setattr(
            "scripts.note._existing_tag_vocabulary",
            lambda config: {"cryo-electron-microscopy"},
        )

        def _raise(proposed, existing):
            raise RuntimeError("[Ollama error: connection refused]")

        monkeypatch.setattr("scripts.note._tag_canonicalization_call", _raise)

        result = _canonicalize_tags(
            ["cryo-electron-microscopy-cryo-em"], {"vault_path": "x"}
        )

        assert result == ["cryo-electron-microscopy-cryo-em"]

    def test_fail_open_on_none_from_llm_call(self, monkeypatch):
        """When the canonicalization call returns None (parse failure), the
        proposed slugs are returned unchanged."""
        from scripts.note import _canonicalize_tags

        monkeypatch.setattr(
            "scripts.note._existing_tag_vocabulary",
            lambda config: {"cryo-electron-microscopy"},
        )
        monkeypatch.setattr(
            "scripts.note._tag_canonicalization_call",
            lambda proposed, existing: None,
        )

        result = _canonicalize_tags(
            ["cryo-electron-microscopy-cryo-em"], {"vault_path": "x"}
        )

        assert result == ["cryo-electron-microscopy-cryo-em"]

    def test_fail_open_on_empty_vocab(self, monkeypatch):
        """When the existing vocabulary is empty, nothing to canonicalize against —
        proposed slugs returned unchanged (and the LLM call is never made)."""
        from scripts.note import _canonicalize_tags

        monkeypatch.setattr(
            "scripts.note._existing_tag_vocabulary", lambda config: set()
        )

        def _fail_if_called(proposed, existing):
            raise AssertionError("LLM call should not be made with empty vocab")

        monkeypatch.setattr(
            "scripts.note._tag_canonicalization_call", _fail_if_called
        )

        result = _canonicalize_tags(["some-topic"], {"vault_path": "x"})

        assert result == ["some-topic"]

    def test_dedupes_order_preserving(self, monkeypatch):
        """Two proposed slugs canonicalizing to the same existing slug collapse to
        one entry, preserving first-seen order."""
        from scripts.note import _canonicalize_tags

        monkeypatch.setattr(
            "scripts.note._existing_tag_vocabulary",
            lambda config: {"cryo-electron-microscopy"},
        )
        monkeypatch.setattr(
            "scripts.note._tag_canonicalization_call",
            lambda proposed, existing: {
                "cryo-em": "cryo-electron-microscopy",
                "cryo-electron-microscopy-cryo-em": "cryo-electron-microscopy",
                "other-topic": "other-topic",
            },
        )

        result = _canonicalize_tags(
            ["cryo-em", "other-topic", "cryo-electron-microscopy-cryo-em"],
            {"vault_path": "x"},
        )

        assert result == ["cryo-electron-microscopy", "other-topic"]

    def test_empty_topics_returns_empty(self, monkeypatch):
        """Empty topics list returns an empty list without calling anything else."""
        from scripts.note import _canonicalize_tags

        result = _canonicalize_tags([], {"vault_path": "x"})

        assert result == []

    def test_slugifies_proposed_topics(self, monkeypatch):
        """Raw (non-slug) proposed topics are slugified before canonicalization."""
        from scripts.note import _canonicalize_tags

        monkeypatch.setattr(
            "scripts.note._existing_tag_vocabulary",
            lambda config: {"structural-biology"},
        )
        monkeypatch.setattr(
            "scripts.note._tag_canonicalization_call",
            lambda proposed, existing: {"structural-biology-topic": "structural-biology"},
        )

        result = _canonicalize_tags(["Structural Biology Topic"], {"vault_path": "x"})

        assert result == ["structural-biology"]


class TestGenerateNoteCanonicalizesTags:
    """generate_note wires _canonicalize_tags between analysis and render."""

    def test_generate_note_calls_canonicalize_tags(self):
        from scripts.note import generate_note

        pj = _make_paperjson()
        config = {"vault_path": "./.local/test-vault", "vault_name": "test-vault"}

        with mock.patch("scripts.note._generate_analysis") as mock_analysis, \
             mock.patch(
                 "scripts.note._canonicalize_tags", return_value=["canonical-topic"]
             ) as mock_canon, \
             mock.patch("scripts.note._render_note", return_value="# Test\n"), \
             mock.patch("scripts.note.preflight", return_value=True), \
             mock.patch("scripts.note.note_exists", return_value=False), \
             mock.patch("scripts.note.create_note", return_value="Created"):

            generate_note(pj, config, force=False)

        mock_canon.assert_called_once()
        assert pj["analysis"]["topics"] == ["canonical-topic"], (
            "generate_note must overwrite analysis['topics'] with the "
            "canonicalization result before rendering"
        )
