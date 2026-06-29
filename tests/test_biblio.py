"""
Unit tests for bibliography linking (BIBLIO-01 through BIBLIO-04, SC4).

Tests cover:
  - test_refs_loaded_from_paperjson (BIBLIO-01): run_biblio consumes extraction.references
  - test_ref_doi_match_produces_wikilink (BIBLIO-02a): DOI registry hit → [[wikilink]]
  - test_ref_title_match_produces_wikilink (BIBLIO-02b): title-hash registry hit → [[wikilink]]
  - test_references_section_injected_before_my_notes (BIBLIO-02c): injection order
  - test_relink_idempotent (BIBLIO-02d): second run replaces, never duplicates References section
  - test_unmatched_ref_creates_stub (BIBLIO-03a): unmatched ref → Stubs/<title>.md
  - test_cited_by_accumulates (BIBLIO-03b): second citing paper appends to cited_by list
  - test_stub_dedup_by_key (BIBLIO-03c): two title variants with same key → one stub
  - test_stub_upgrade_moves_to_papers (BIBLIO-04a): stub deleted, full note in Papers/
  - test_stub_upgrade_rewrites_backlinks (BIBLIO-04b): [[stub title]] → [[full title]]
  - test_stubs_not_registered (BIBLIO-04c): stubs never written to registry
  - test_malformed_ref_does_not_abort (BIBLIO-04d / SC4): fill_failed ref is non-fatal
  - test_biblio_failure_does_not_abort_ingest (BIBLIO-04e / SC4): internal fail → [biblio warning:]

Run with:  python -m pytest tests/test_biblio.py -x
"""

import json
import pytest
from unittest import mock
from pathlib import Path

from scripts.ingest import (
    _assemble_paperjson,
    SCHEMA_VERSION,
    _registry_key,
    _check_registry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paperjson_with_refs(refs=None):
    """Build a minimal PaperJSON v2 dict with RefEntry-shaped references."""
    if refs is None:
        refs = [
            {"number": 1, "raw": "Vaswani et al., NeurIPS 2017", "doi": "10.1234/attn",
             "title": "Attention Is All You Need", "year": 2017, "fill_failed": False},
            {"number": 2, "raw": "LeCun et al., Nature 2015",   "doi": None,
             "title": "Deep Learning", "year": 2015, "fill_failed": False},
            {"number": 3, "raw": "",                              "doi": None,
             "title": None, "year": None, "fill_failed": True},
        ]
    parsed = {
        "title": "Test Citing Paper",
        "sections": [],
        "references": refs,
        "metadata": {"title": "Test Citing Paper", "doi": "10.9999/citing"},
    }
    provenance = {
        "pdf_sha256": "abc123", "source_filename": "test.pdf",
        "mineru_version": "2.5", "backend": "hybrid_auto",
        "extracted_at": "2026-06-28T00:00:00Z",
        "normalizations_applied": [], "schema_version": SCHEMA_VERSION,
    }
    pj = _assemble_paperjson(parsed, provenance)
    pj["extraction"]["references"] = refs
    return pj


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_vault(tmp_path):
    """Create a minimal vault structure under tmp_path."""
    (tmp_path / "Papers").mkdir()
    (tmp_path / "Stubs").mkdir()
    return tmp_path


@pytest.fixture
def config(tmp_vault):
    return {"vault_path": str(tmp_vault), "registry_path": str(tmp_vault / "registry.json")}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_refs_loaded_from_paperjson(tmp_vault, config):
    """BIBLIO-01: run_biblio consumes extraction.references without error."""
    from scripts import biblio  # noqa: PLC0415
    pj = _make_paperjson_with_refs()
    # Pre-write citing note so biblio can locate and rewrite it
    (tmp_vault / "Papers" / "Test Citing Paper.md").write_text(
        "# Test Citing Paper\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    result = biblio.run_biblio(pj, config)
    assert isinstance(result, str), "run_biblio must return a string"


def test_ref_doi_match_produces_wikilink(tmp_vault, config):
    """BIBLIO-02a: registry hit on DOI match yields [[sanitized title]] in References section."""
    from scripts import biblio  # noqa: PLC0415
    # Seed registry with DOI-keyed entry; registry has no vault_note field (Pitfall 1)
    doi_key = _registry_key({"doi": "10.1234/attn"})  # == "10.1234/attn"
    registry = {doi_key: {"title": "Attention Is All You Need", "authors": None, "year": 2017}}
    Path(config["registry_path"]).write_text(json.dumps(registry), encoding="utf-8")
    (tmp_vault / "Papers" / "Test Citing Paper.md").write_text(
        "# Test Citing Paper\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    pj = _make_paperjson_with_refs()
    biblio.run_biblio(pj, config)
    note_content = (tmp_vault / "Papers" / "Test Citing Paper.md").read_text(encoding="utf-8")
    assert "## References" in note_content, "References section not injected"
    assert "[[Attention Is All You Need]]" in note_content, "DOI-matched wikilink missing"


def test_ref_title_match_produces_wikilink(tmp_vault, config):
    """BIBLIO-02b: registry hit on normalized-title hash key yields [[wikilink]]."""
    from scripts import biblio  # noqa: PLC0415
    title_key = _registry_key({"title": "Deep Learning"})  # "sha256:505bd12938815abc"
    registry = {title_key: {"title": "Deep Learning", "authors": None, "year": 2015}}
    Path(config["registry_path"]).write_text(json.dumps(registry), encoding="utf-8")
    (tmp_vault / "Papers" / "Test Citing Paper.md").write_text(
        "# Test Citing Paper\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    pj = _make_paperjson_with_refs()
    biblio.run_biblio(pj, config)
    note_content = (tmp_vault / "Papers" / "Test Citing Paper.md").read_text(encoding="utf-8")
    assert "## References" in note_content, "References section not injected"
    assert "[[Deep Learning]]" in note_content, "Title-matched wikilink missing"


def test_references_section_injected_before_my_notes(tmp_vault, config):
    """BIBLIO-02c: ## References heading appears before ## My Notes in the rewritten note."""
    from scripts import biblio  # noqa: PLC0415
    (tmp_vault / "Papers" / "Test Citing Paper.md").write_text(
        "# Test Citing Paper\n\n## My Notes\n\nSome personal notes.\n",
        encoding="utf-8",
    )
    pj = _make_paperjson_with_refs()
    biblio.run_biblio(pj, config)
    note_content = (tmp_vault / "Papers" / "Test Citing Paper.md").read_text(encoding="utf-8")
    refs_pos = note_content.find("## References")
    notes_pos = note_content.find("## My Notes")
    assert refs_pos != -1, "## References section not found in rewritten note"
    assert notes_pos != -1, "## My Notes section disappeared from rewritten note"
    assert refs_pos < notes_pos, "## References must appear before ## My Notes"


def test_relink_idempotent(tmp_vault, config):
    """BIBLIO-02d: running run_biblio twice produces exactly one ## References section."""
    from scripts import biblio  # noqa: PLC0415
    (tmp_vault / "Papers" / "Test Citing Paper.md").write_text(
        "# Test Citing Paper\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    pj = _make_paperjson_with_refs()
    biblio.run_biblio(pj, config)
    biblio.run_biblio(pj, config)
    note_content = (tmp_vault / "Papers" / "Test Citing Paper.md").read_text(encoding="utf-8")
    count = note_content.count("## References")
    assert count == 1, f"Expected exactly 1 '## References' section after two runs, got {count}"


def test_unmatched_ref_creates_stub(tmp_vault, config):
    """BIBLIO-03a: an unmatched ref creates Stubs/<sanitized title>.md containing status: stub."""
    from scripts import biblio  # noqa: PLC0415
    # Only a title-only ref; no registry entry seeded → unmatched → stub
    refs = [
        {"number": 1, "raw": "LeCun et al., Nature 2015", "doi": None,
         "title": "Deep Learning", "year": 2015, "fill_failed": False},
    ]
    pj = _make_paperjson_with_refs(refs=refs)
    (tmp_vault / "Papers" / "Test Citing Paper.md").write_text(
        "# Test Citing Paper\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    biblio.run_biblio(pj, config)
    stubs = list((tmp_vault / "Stubs").iterdir())
    assert len(stubs) >= 1, "Expected at least one stub file created for unmatched ref"
    stub_content = stubs[0].read_text(encoding="utf-8")
    assert "status: stub" in stub_content, "Stub file must contain 'status: stub'"


def test_cited_by_accumulates(tmp_vault, config):
    """BIBLIO-03b: a second citing paper appends to the stub's cited_by list."""
    from scripts import biblio  # noqa: PLC0415
    refs = [
        {"number": 1, "raw": "LeCun et al., Nature 2015", "doi": None,
         "title": "Deep Learning", "year": 2015, "fill_failed": False},
    ]
    # First citing paper run
    pj1 = _make_paperjson_with_refs(refs=refs)
    (tmp_vault / "Papers" / "Test Citing Paper.md").write_text(
        "# Test Citing Paper\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    biblio.run_biblio(pj1, config)
    # Second citing paper
    second_title = "Second Citing Paper"
    parsed2 = {
        "title": second_title,
        "sections": [],
        "references": refs,
        "metadata": {"title": second_title, "doi": "10.9999/second"},
    }
    provenance2 = {
        "pdf_sha256": "xyz789", "source_filename": "second.pdf",
        "mineru_version": "2.5", "backend": "hybrid_auto",
        "extracted_at": "2026-06-28T00:00:00Z",
        "normalizations_applied": [], "schema_version": SCHEMA_VERSION,
    }
    pj2 = _assemble_paperjson(parsed2, provenance2)
    pj2["extraction"]["references"] = refs
    (tmp_vault / "Papers" / f"{second_title}.md").write_text(
        f"# {second_title}\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    biblio.run_biblio(pj2, config)
    stubs = list((tmp_vault / "Stubs").iterdir())
    assert stubs, "Expected stub file to exist"
    stub_content = stubs[0].read_text(encoding="utf-8")
    assert "Test Citing Paper" in stub_content, "First citing paper not in cited_by"
    assert second_title in stub_content, "Second citing paper not appended to cited_by"


def test_stub_dedup_by_key(tmp_vault, config):
    """BIBLIO-03c: two title variants with same registry key → exactly one stub."""
    from scripts import biblio  # noqa: PLC0415
    # "Deep Learning" and "Deep  Learning" (extra space) normalize to the same hash
    refs = [
        {"number": 1, "raw": "LeCun, 2015", "doi": None,
         "title": "Deep Learning", "year": 2015, "fill_failed": False},
        {"number": 2, "raw": "LeCun, 2015 alt", "doi": None,
         "title": "Deep  Learning", "year": 2015, "fill_failed": False},
    ]
    pj = _make_paperjson_with_refs(refs=refs)
    (tmp_vault / "Papers" / "Test Citing Paper.md").write_text(
        "# Test Citing Paper\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    biblio.run_biblio(pj, config)
    stubs = list((tmp_vault / "Stubs").iterdir())
    assert len(stubs) == 1, f"Expected exactly 1 stub (dedup by stub_key), got {len(stubs)}"


def test_stub_upgrade_moves_to_papers(tmp_vault, config):
    """BIBLIO-04a: after upgrade_stub, stub file is gone and Papers/<full title>.md exists."""
    from scripts import biblio  # noqa: PLC0415
    stub_key = _registry_key({"title": "Deep Learning"})  # "sha256:505bd12938815abc"
    stub_path = tmp_vault / "Stubs" / "Deep Learning.md"
    stub_path.write_text(
        f"---\nstatus: stub\nstub_key: \"{stub_key}\"\ncited_by:\n  - \"Test Paper\"\n---\n\n"
        "Raw citation: LeCun et al., 2015\n",
        encoding="utf-8",
    )
    biblio.upgrade_stub(
        stub_key=stub_key,
        full_title="Deep Learning",
        vault_path=config["vault_path"],
    )
    assert not stub_path.exists(), "Stub file must be deleted after upgrade"
    assert (tmp_vault / "Papers" / "Deep Learning.md").exists(), \
        "Full note must exist in Papers/ after upgrade"


def test_stub_upgrade_rewrites_backlinks(tmp_vault, config):
    """BIBLIO-04b: a citing note's [[stub title]] becomes [[full title]] after upgrade."""
    from scripts import biblio  # noqa: PLC0415
    stub_path = tmp_vault / "Stubs" / "Dl Placeholder.md"
    stub_path.write_text(
        "---\nstatus: stub\nstub_key: \"sha256:testkey0001xxxx\"\ncited_by:\n  - \"Citing Paper A\"\n---\n",
        encoding="utf-8",
    )
    citing_note = tmp_vault / "Papers" / "Citing Paper A.md"
    citing_note.write_text(
        "## References\n\n- [[Dl Placeholder]]\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    biblio.upgrade_stub(
        stub_key="sha256:testkey0001xxxx",
        full_title="Deep Learning",
        vault_path=config["vault_path"],
    )
    updated = citing_note.read_text(encoding="utf-8")
    assert "[[Deep Learning]]" in updated, "Backlink must be rewritten to [[full title]]"
    assert "[[Dl Placeholder]]" not in updated, "Old stub wikilink must be removed"


def test_stubs_not_registered(tmp_vault, config):
    """BIBLIO-04c: run_biblio never writes a registry entry for a stub."""
    from scripts import biblio  # noqa: PLC0415
    refs = [
        {"number": 1, "raw": "LeCun et al., Nature 2015", "doi": None,
         "title": "Deep Learning", "year": 2015, "fill_failed": False},
    ]
    pj = _make_paperjson_with_refs(refs=refs)
    (tmp_vault / "Papers" / "Test Citing Paper.md").write_text(
        "# Test Citing Paper\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    biblio.run_biblio(pj, config)
    stub_key = _registry_key({"title": "Deep Learning"})
    entry = _check_registry(stub_key, config["registry_path"])
    assert entry is None, f"Stub must NOT be registered (would skip future full ingest); got {entry}"


def test_malformed_ref_does_not_abort(tmp_vault, config):
    """BIBLIO-04d / SC4: malformed ref (fill_failed True, empty raw) is non-fatal."""
    from scripts import biblio  # noqa: PLC0415
    # Default refs include the malformed entry (number 3, fill_failed True, empty raw)
    pj = _make_paperjson_with_refs()
    (tmp_vault / "Papers" / "Test Citing Paper.md").write_text(
        "# Test Citing Paper\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    result = biblio.run_biblio(pj, config)
    assert isinstance(result, str), "run_biblio must return a string even with malformed ref"
    assert not result.startswith("[biblio warning:"), \
        "Malformed ref must not cause run_biblio to return a warning string"


def test_biblio_failure_does_not_abort_ingest(tmp_vault, config):
    """BIBLIO-04e / SC4: internal biblio failure returns [biblio warning:...] not raises."""
    from scripts import biblio  # noqa: PLC0415
    pj = _make_paperjson_with_refs()
    (tmp_vault / "Papers" / "Test Citing Paper.md").write_text(
        "# Test Citing Paper\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    # Force an internal helper to raise so run_biblio's own error handler is exercised
    with mock.patch.object(biblio, "_process_refs", side_effect=RuntimeError("simulated failure")):
        result = biblio.run_biblio(pj, config)
    assert result.startswith("[biblio warning:"), \
        f"Expected '[biblio warning:...]' but got: {result!r}"
