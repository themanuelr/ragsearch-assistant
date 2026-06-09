"""
pytest fixtures shared across all test modules.
"""

import json
import os
import re
from unittest.mock import patch

import pytest


@pytest.fixture
def tmp_registry_path(tmp_path):
    """Temporary registry JSON path — cleaned up after each test, including .lock sibling."""
    path = str(tmp_path / "papers_registry.json")
    yield path
    lock_path = path + ".lock"
    if os.path.exists(lock_path):
        try:
            os.unlink(lock_path)
        except OSError:
            pass


@pytest.fixture
def sample_pdf_path():
    """Path to a real single-column PDF in tests/fixtures/."""
    return os.path.join(os.path.dirname(__file__), "fixtures", "sample_onecol.pdf")


@pytest.fixture
def sample_twocol_pdf_path():
    """Path to a real two-column PDF in tests/fixtures/."""
    return os.path.join(os.path.dirname(__file__), "fixtures", "sample_twocol.pdf")


@pytest.fixture
def scanned_pdf_path():
    """Path to an image-only PDF in tests/fixtures/."""
    return os.path.join(os.path.dirname(__file__), "fixtures", "sample_scanned.pdf")


@pytest.fixture
def test_manuel2_pdf_path():
    """Path to the real Scientific Reports PDF used for end-to-end scoring."""
    return os.path.join(os.path.dirname(__file__), "fixtures", "test_manuel2.pdf")


@pytest.fixture
def expected_fixture():
    """Load the hand-authored ground-truth PaperJSON for test_manuel2.pdf."""
    path = os.path.join(os.path.dirname(__file__), "fixtures", "test_manuel2.expected.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _norm_str(s: object) -> str:
    """Whitespace-normalize a string for exact-match comparison."""
    return re.sub(r"\s+", " ", str(s)).strip()


def score_against_expected(actual: dict, expected: dict) -> dict:
    """Score an extraction output against the ground-truth fixture.

    Returns a flat dict of per-field {field: True/False} results. Exact-match
    fields (per expected["_scoring_targets"]["exact_match"]) must match after
    whitespace-normalization; authors compares as an order-sensitive list.
    Structural fields use the threshold/shape rules documented in the fixture's
    _scoring_targets and the phase VALIDATION strategy (section bodies are anchor
    excerpts, NOT verbatim — assert presence, non-empty, and anchor containment).
    """
    targets = expected.get("_scoring_targets", {})
    exact_fields = targets.get("exact_match", [])
    results: dict = {}

    # --- exact-match scalar/list fields ---
    for field in exact_fields:
        exp_val = expected.get(field)
        act_val = actual.get(field)
        if field == "authors":
            results[field] = (
                isinstance(act_val, list)
                and [_norm_str(a) for a in act_val] == [_norm_str(a) for a in (exp_val or [])]
            )
        elif isinstance(exp_val, str):
            results[field] = isinstance(act_val, str) and _norm_str(act_val) == _norm_str(exp_val)
        else:
            results[field] = act_val == exp_val

    # --- structural: sections ---
    exp_sections = expected.get("sections", [])
    act_sections = actual.get("sections", [])
    sections_ok = (
        isinstance(act_sections, list)
        and len(act_sections) >= 1
        and all(
            isinstance(s, dict) and _norm_str(s.get("body", "")) != ""
            for s in act_sections
        )
    )
    # Introduction must be present with a non-empty body containing the anchor opening.
    intro_anchor = ""
    for s in exp_sections:
        if _norm_str(s.get("title", "")).lower() == "introduction":
            # anchor = opening substring before the first ellipsis marker
            body = s.get("body", "")
            intro_anchor = _norm_str(body.split("...")[0]) if body else ""
            break
    if intro_anchor:
        intro_match = any(
            _norm_str(s.get("title", "")).lower() == "introduction"
            and _norm_str(s.get("body", "")) != ""
            and intro_anchor.lower() in _norm_str(s.get("body", "")).lower()
            for s in act_sections
        )
        sections_ok = sections_ok and intro_match
    results["sections"] = sections_ok

    # --- structural: references (threshold) ---
    act_refs = actual.get("references")
    results["references"] = (
        isinstance(act_refs, list)
        and len(act_refs) >= 40
        and all(isinstance(r, str) and r.strip() for r in act_refs)
        and act_refs[0].lstrip().startswith("1.")
    )

    # --- structural: figures (dedup + count + captions) ---
    act_figs = actual.get("figures")
    if isinstance(act_figs, list) and act_figs:
        labels = [_norm_str(f.get("label", "")) for f in act_figs if isinstance(f, dict)]
        no_dupes = len(labels) == len(set(labels))
        count_ok = 6 <= len(act_figs) <= 8  # 7 +/- 1
        captions_ok = all(
            isinstance(f, dict) and _norm_str(f.get("caption", "")) != ""
            for f in act_figs
        )
        results["figures"] = no_dupes and count_ok and captions_ok
    else:
        results["figures"] = False

    return results


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    """Stub _extract_with_llm so tests run without a live Ollama server.

    Prevents each integration test from hanging for 300 s waiting for a
    timeout when Ollama is not running on a CI runner (WR-03 fix).

    Tests that specifically exercise the LLM network path
    (test_extract_with_llm_*) call _extract_with_llm directly via their own
    imported reference and additionally patch urlopen themselves, so this
    autouse stub has no practical effect on them.
    """
    def _stub(pages_text):
        return {
            "title": "Sample Single Column Paper",
            "authors": ["Alice Author", "Bob Collaborator"],
            "abstract": "A test abstract.",
            "sections": [
                {"title": "Section A", "body": "Left column content."},
                {"title": "Section B", "body": "Right column content."},
            ],
            "references": None,
            "figures": None,
            "year": None,
            "doi": None,
            "journal": None,
        }
    monkeypatch.setattr("scripts.ingest._extract_with_llm", _stub)
    yield


@pytest.fixture(autouse=True)
def _tmp_config(tmp_path):
    """Write a temp config.json so extract_paper does not require a real one.

    Patches scripts.ingest._find_config to return the temp config path for the
    duration of each test, preventing tests from walking the filesystem for a
    real config.json (which is gitignored and may not exist on a clean clone).
    """
    registry = tmp_path / "papers_registry.json"
    cfg = {
        "registry_path": str(registry),
        "vault_path": str(tmp_path / "vault"),
        "project_name": "test-project",
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    with patch("scripts.ingest._find_config", return_value=str(cfg_path)):
        yield str(cfg_path)
