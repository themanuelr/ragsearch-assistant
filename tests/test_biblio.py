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
  - test_doiless_ref_to_in_vault_paper_renders_wikilink (BIBLIO-02/04-04): normalized-title hit
  - test_cross_citer_dedup_with_title_variation_one_stub (BIBLIO-03/04-04): accent-fold dedup
  - test_match_normalization_folds_accents (04-04): _normalize_title_for_match unit test

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
    """BIBLIO-04b: a citing note's real miss-branch line becomes [[full title]] after upgrade.

    Updated for WR-02 (code review): upgrade_stub is now a thin wrapper over
    _upgrade_stub — the single implementation used by run_biblio — so the citing
    note fixture uses the ACTUAL miss-branch render form (raw text + marker +
    <!--stub:{key}--> anchor), never an idealized [[stub title]] wikilink the
    miss branch never produces (04-09/Gap F bug 2).
    """
    from scripts import biblio  # noqa: PLC0415
    stub_key = "sha256:testkey0001xxxx"
    stub_path = tmp_vault / "Stubs" / "Dl Placeholder.md"
    stub_path.write_text(
        f"---\ntitle: \"Dl Placeholder\"\nstatus: stub\nstub_key: \"{stub_key}\"\n"
        "cited_by:\n  - \"Citing Paper A\"\n---\n",
        encoding="utf-8",
    )
    citing_note = tmp_vault / "Papers" / "Citing Paper A.md"
    citing_note.write_text(
        "# Citing Paper A\n\n## References\n\n"
        f"1. LeCun et al., 2015{biblio._STUB_MISS_MARKER}{biblio._stub_anchor(stub_key)}\n"
        "\n## My Notes\n\n",
        encoding="utf-8",
    )
    biblio.upgrade_stub(
        stub_key=stub_key,
        full_title="Deep Learning",
        vault_path=config["vault_path"],
    )
    updated = citing_note.read_text(encoding="utf-8")
    assert "1. [[Deep Learning]]" in updated, "Backlink must be rewritten to [[full title]]"
    assert "(not yet in vault)" not in updated, "Miss marker must be gone after rewrite"
    assert biblio._stub_anchor(stub_key) not in updated, "Stub anchor must be gone after rewrite"


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


def test_upgrade_merges_cited_by_into_full_note(tmp_vault, config):
    """BIBLIO-04f / SC3: after upgrade via run_biblio, full note frontmatter has cited_by
    and the citing note's REAL miss-branch line is relinked (closes Gap F bug 2).

    Rewritten from the masking version (04-09): the citing-note fixture is now produced
    by an actual run_biblio miss render — never a hand-seeded "[[wikilink]]" — so the
    backlink-rewrite assertion exercises the real on-disk text a stub-matched reference
    is rendered as.
    """
    from scripts import biblio  # noqa: PLC0415

    paper_b_title = "Deep Learning"
    citer_a_path = "Papers/Citer A.md"

    # Step 1: run a REAL run_biblio pass for Citer A with a single doiless ref to
    # paper B. This creates a title-hash-keyed Stubs/Deep Learning.md AND writes
    # Citer A's ## References section using the actual miss-branch render — the
    # exact real-world text bug 2's masking fixture bypassed.
    refs_a = [
        {
            "number": 1,
            "raw": "LeCun et al., Deep Learning, Nature 2015",
            "doi": None,
            "title": paper_b_title,
            "year": 2015,
            "fill_failed": False,
        }
    ]
    pj_a = _make_paperjson_with_refs(refs=refs_a)
    pj_a["extraction"]["metadata"]["title"] = "Citer A"
    pj_a["extraction"]["metadata"]["doi"] = "10.9999/citer-a"

    (tmp_vault / "Papers" / "Citer A.md").write_text(
        "# Citer A\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    result_a = biblio.run_biblio(pj_a, config)
    assert not result_a.startswith("[biblio warning:"), f"run_biblio (citer A) failed: {result_a}"

    citer_a_content_before = (tmp_vault / citer_a_path).read_text(encoding="utf-8")
    assert "*(not yet in vault)*" in citer_a_content_before, (
        "Setup precondition: Citer A's real miss-branch render must contain the "
        "'(not yet in vault)' marker before upgrade"
    )

    # Step 2: pre-write Papers/Deep Learning.md (simulating step 12b — note.generate_note
    # having already run for the newly-ingested full note). The full note must exist
    # or _inject_references_section raises and run_biblio returns a warning string.
    full_note_file = tmp_vault / "Papers" / f"{paper_b_title}.md"
    full_note_file.write_text(
        f'---\ntitle: "{paper_b_title}"\nauthor: "LeCun"\n---\n\n'
        f"# {paper_b_title}\n\n## My Notes\n\n",
        encoding="utf-8",
    )

    # Step 3: ingest paper B itself. doi=None isolates bug 2 — self_key already
    # matches the title-hash stub (both were derived from the same doiless title),
    # so the upgrade block fires; only the backlink rewrite is under test here.
    pj_b = _make_paperjson_with_refs(refs=[])
    pj_b["extraction"]["metadata"]["title"] = paper_b_title
    pj_b["extraction"]["metadata"]["doi"] = None

    result_b = biblio.run_biblio(pj_b, config)
    assert not result_b.startswith("[biblio warning:"), f"run_biblio (paper B) failed: {result_b}"

    assert not (tmp_vault / "Stubs" / f"{paper_b_title}.md").exists(), (
        "Stub must be deleted after upgrade"
    )
    full_note_content = full_note_file.read_text(encoding="utf-8")
    assert "cited_by:" in full_note_content, "Full note must have cited_by in frontmatter after upgrade"
    assert citer_a_path in full_note_content, "cited_by must include the citing note path"

    citer_a_content_after = (tmp_vault / citer_a_path).read_text(encoding="utf-8")
    assert f"[[{paper_b_title}]]" in citer_a_content_after, (
        "Citer A's real miss-branch line must be relinked to [[full title]] "
        "(bug 2: rewrite must target the actual rendered text, not an idealized wikilink)"
    )
    assert "(not yet in vault)" not in citer_a_content_after, (
        "Citer A's '(not yet in vault)' marker must be gone after backlink rewrite"
    )


def test_stub_upgrade_fires_when_new_ingest_carries_doi(tmp_vault, config):
    """BIBLIO-04g / SC3: a DOI-bearing new ingest still finds and upgrades a
    pre-existing title-hash-keyed stub (closes Gap F bug 1 — the DOI-vs-title-hash
    self_key mismatch on the UPGRADE axis, symmetric to the 04-07 DEDUP fix)."""
    from scripts import biblio  # noqa: PLC0415

    paper_b_title = "Deep Learning"
    paper_b_doi = "10.1038/s41586-019-1438-2"
    citer_a_path = "Papers/Citer A.md"

    # Step 1: same real-fixture setup — Citer A cites paper B doiless, producing a
    # title-hash-keyed Stubs/Deep Learning.md and a real miss-branch citing line.
    refs_a = [
        {
            "number": 1,
            "raw": "LeCun et al., Deep Learning, Nature 2015",
            "doi": None,
            "title": paper_b_title,
            "year": 2015,
            "fill_failed": False,
        }
    ]
    pj_a = _make_paperjson_with_refs(refs=refs_a)
    pj_a["extraction"]["metadata"]["title"] = "Citer A"
    pj_a["extraction"]["metadata"]["doi"] = "10.9999/citer-a"

    (tmp_vault / "Papers" / "Citer A.md").write_text(
        "# Citer A\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    result_a = biblio.run_biblio(pj_a, config)
    assert not result_a.startswith("[biblio warning:"), f"run_biblio (citer A) failed: {result_a}"

    stub_file = tmp_vault / "Stubs" / f"{paper_b_title}.md"
    assert stub_file.exists(), "Setup precondition: title-hash stub must exist before upgrade"
    stub_content_before = stub_file.read_text(encoding="utf-8")
    assert "sha256:" in stub_content_before, (
        "Setup precondition: stub must be keyed by a title-hash (doiless ref), not a DOI"
    )

    # Step 2: pre-write the full note (simulating step 12b).
    full_note_file = tmp_vault / "Papers" / f"{paper_b_title}.md"
    full_note_file.write_text(
        f'---\ntitle: "{paper_b_title}"\nauthor: "LeCun"\n---\n\n'
        f"# {paper_b_title}\n\n## My Notes\n\n",
        encoding="utf-8",
    )

    # Step 3: ingest paper B itself — THIS TIME its own metadata carries a real DOI
    # the original title-hash stub never had (pins bug 1).
    pj_b = _make_paperjson_with_refs(refs=[])
    pj_b["extraction"]["metadata"]["title"] = paper_b_title
    pj_b["extraction"]["metadata"]["doi"] = paper_b_doi

    result_b = biblio.run_biblio(pj_b, config)
    assert not result_b.startswith("[biblio warning:"), f"run_biblio (paper B) failed: {result_b}"

    assert not stub_file.exists(), "Stub must be deleted after upgrade (bug 1 fix)"

    full_note_content = full_note_file.read_text(encoding="utf-8")
    assert "cited_by:" in full_note_content, "Full note must have cited_by in frontmatter after upgrade"
    assert citer_a_path in full_note_content, "cited_by must include the citing note path"

    citer_a_content_after = (tmp_vault / citer_a_path).read_text(encoding="utf-8")
    assert f"[[{paper_b_title}]]" in citer_a_content_after, (
        "Citer A's real miss-branch line must be relinked to [[full title]] "
        "(bug 1: a DOI-bearing new ingest must still find the title-hash-keyed stub)"
    )
    assert "(not yet in vault)" not in citer_a_content_after, (
        "Citer A's '(not yet in vault)' marker must be gone after backlink rewrite"
    )


# ---------------------------------------------------------------------------
# Gap-closure tests (04-04): normalized-title registry resolution + accent-fold dedup
# ---------------------------------------------------------------------------

def test_doiless_ref_to_in_vault_paper_renders_wikilink(tmp_vault, config):
    """BIBLIO-02/04-04: a doiless ref whose title matches a DOI-keyed registry entry renders
    as [[wikilink]] and creates NO stub (Layer 2 of the locked resolution chain)."""
    from scripts import biblio  # noqa: PLC0415

    # Seed registry with a DOI-keyed entry (paper already in vault)
    registry = {
        "10.1073/pnas.2020": {
            "title": "Cryo EM Structures of KCC1",
            "authors": None,
            "year": 2020,
        }
    }
    Path(config["registry_path"]).write_text(json.dumps(registry), encoding="utf-8")

    # Build a ref with NO doi but a title that normalizes to the same string as the
    # registry entry (hyphen and mixed-case differences only — accent-free so the only
    # failure mode is the DOI-vs-title-hash key mismatch, not accent folding).
    refs = [
        {
            "number": 1,
            "raw": "Bhatt et al., PNAS 2020",
            "doi": None,
            "title": "Cryo-EM structures of KCC1",
            "year": 2020,
            "fill_failed": False,
        }
    ]
    pj = _make_paperjson_with_refs(refs=refs)

    (tmp_vault / "Papers" / "Test Citing Paper.md").write_text(
        "# Test Citing Paper\n\n## My Notes\n\n",
        encoding="utf-8",
    )

    result = biblio.run_biblio(pj, config)
    assert not result.startswith("[biblio warning:"), f"run_biblio failed: {result}"

    note_content = (tmp_vault / "Papers" / "Test Citing Paper.md").read_text(encoding="utf-8")
    assert "## References" in note_content, "References section not injected"
    assert "[[Cryo EM Structures of KCC1]]" in note_content, (
        "Doiless ref matching DOI-keyed registry entry must render as [[wikilink]]"
    )

    stubs = list((tmp_vault / "Stubs").iterdir())
    assert len(stubs) == 0, (
        f"No stub should be created for a ref that resolves to an in-vault paper; got {[s.name for s in stubs]}"
    )


def test_cross_citer_dedup_with_title_variation_one_stub(tmp_vault, config):
    """BIBLIO-03/04-04: two citers with accent-variant titles for the same not-in-vault paper
    produce exactly ONE stub accumulating both citers in cited_by (accent-fold dedup via _match_key)."""
    from scripts import biblio  # noqa: PLC0415
    from scripts.ingest import _assemble_paperjson, SCHEMA_VERSION  # noqa: PLC0415

    # Citer A references the paper with an umlaut: "Structures of Müller Cells"
    refs_a = [
        {
            "number": 1,
            "raw": "Müller et al., 2019",
            "doi": None,
            "title": "Structures of Müller Cells",
            "year": 2019,
            "fill_failed": False,
        }
    ]
    pj_a = _make_paperjson_with_refs(refs=refs_a)
    pj_a["extraction"]["metadata"]["title"] = "Citer Paper A"
    pj_a["extraction"]["metadata"]["doi"] = "10.9999/citer-a"

    (tmp_vault / "Papers" / "Citer Paper A.md").write_text(
        "# Citer Paper A\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    result_a = biblio.run_biblio(pj_a, config)
    assert not result_a.startswith("[biblio warning:"), f"run_biblio (citer A) failed: {result_a}"

    # Citer B references the same paper without the umlaut: "Structures of Muller Cells"
    refs_b = [
        {
            "number": 1,
            "raw": "Muller et al., 2019",
            "doi": None,
            "title": "Structures of Muller Cells",
            "year": 2019,
            "fill_failed": False,
        }
    ]
    parsed_b = {
        "title": "Citer Paper B",
        "sections": [],
        "references": refs_b,
        "metadata": {"title": "Citer Paper B", "doi": "10.9999/citer-b"},
    }
    provenance_b = {
        "pdf_sha256": "bbb222", "source_filename": "citer_b.pdf",
        "mineru_version": "2.5", "backend": "hybrid_auto",
        "extracted_at": "2026-06-30T00:00:00Z",
        "normalizations_applied": [], "schema_version": SCHEMA_VERSION,
    }
    pj_b = _assemble_paperjson(parsed_b, provenance_b)
    pj_b["extraction"]["references"] = refs_b

    (tmp_vault / "Papers" / "Citer Paper B.md").write_text(
        "# Citer Paper B\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    result_b = biblio.run_biblio(pj_b, config)
    assert not result_b.startswith("[biblio warning:"), f"run_biblio (citer B) failed: {result_b}"

    stubs = list((tmp_vault / "Stubs").iterdir())
    assert len(stubs) == 1, (
        f"Expected exactly ONE stub for the accent-variant titles (dedup by _match_key); "
        f"got {len(stubs)}: {[s.name for s in stubs]}"
    )

    stub_content = stubs[0].read_text(encoding="utf-8")
    assert "Citer Paper A" in stub_content, "First citer (A) must appear in stub cited_by"
    assert "Citer Paper B" in stub_content, "Second citer (B) must appear in stub cited_by"


def test_match_normalization_folds_accents(tmp_vault, config):
    """04-04 unit: _normalize_title_for_match folds NFKD accents and collapses case/punctuation."""
    from scripts import biblio  # noqa: PLC0415

    # Accent folding: ü -> u, accent diacritics stripped
    assert biblio._normalize_title_for_match("Müller") == biblio._normalize_title_for_match(
        "Muller"
    ), "_normalize_title_for_match must fold 'ü' to 'u'"

    # Case collapsing
    assert biblio._normalize_title_for_match("MÜLLER") == biblio._normalize_title_for_match(
        "muller"
    ), "_normalize_title_for_match must be case-insensitive after accent folding"

    # Punctuation collapsing (hyphen treated same as space)
    assert biblio._normalize_title_for_match("Cryo-EM") == biblio._normalize_title_for_match(
        "Cryo EM"
    ), "_normalize_title_for_match must collapse punctuation and whitespace runs"


# ---------------------------------------------------------------------------
# Gap-closure tests (04-07): stub-title-index dedup for DOI-vs-doiless refs
# and hyphen/space/joined title variants (Gap C, 04-UAT Test 1 dedup half)
# ---------------------------------------------------------------------------

def test_doiless_and_doi_ref_dedup_to_one_stub_keyed_by_doi(tmp_vault, config):
    """BIBLIO-03/04-07: DOI-bearing citer linked first, doiless citer second, same title
    for a not-in-vault paper -> exactly ONE stub keyed by the DOI, both citers in cited_by
    (closes Gap C mode 1: DOI-vs-title-hash key mismatch)."""
    from scripts import biblio  # noqa: PLC0415
    from scripts.ingest import _assemble_paperjson, SCHEMA_VERSION  # noqa: PLC0415

    shared_title = "Structure and mechanism of the cation-chloride cotransporter NKCC1"
    shared_doi = "10.1038/s41586-019-1438-2"

    # Citer A references the paper WITH the DOI (linked first)
    refs_a = [
        {
            "number": 1,
            "raw": "Chew et al., Nature 2019",
            "doi": shared_doi,
            "title": shared_title,
            "year": 2019,
            "fill_failed": False,
        }
    ]
    pj_a = _make_paperjson_with_refs(refs=refs_a)
    pj_a["extraction"]["metadata"]["title"] = "Citer Paper A"
    pj_a["extraction"]["metadata"]["doi"] = "10.9999/citer-a"

    (tmp_vault / "Papers" / "Citer Paper A.md").write_text(
        "# Citer Paper A\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    result_a = biblio.run_biblio(pj_a, config)
    assert not result_a.startswith("[biblio warning:"), f"run_biblio (citer A) failed: {result_a}"

    # Citer B references the SAME paper doiless with the same title text
    refs_b = [
        {
            "number": 1,
            "raw": "Chew et al., 2019 (doiless cite)",
            "doi": None,
            "title": shared_title,
            "year": 2019,
            "fill_failed": False,
        }
    ]
    parsed_b = {
        "title": "Citer Paper B",
        "sections": [],
        "references": refs_b,
        "metadata": {"title": "Citer Paper B", "doi": "10.9999/citer-b"},
    }
    provenance_b = {
        "pdf_sha256": "doib222", "source_filename": "citer_b_doiless.pdf",
        "mineru_version": "2.5", "backend": "hybrid_auto",
        "extracted_at": "2026-07-01T00:00:00Z",
        "normalizations_applied": [], "schema_version": SCHEMA_VERSION,
    }
    pj_b = _assemble_paperjson(parsed_b, provenance_b)
    pj_b["extraction"]["references"] = refs_b

    (tmp_vault / "Papers" / "Citer Paper B.md").write_text(
        "# Citer Paper B\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    result_b = biblio.run_biblio(pj_b, config)
    assert not result_b.startswith("[biblio warning:"), f"run_biblio (citer B) failed: {result_b}"

    stubs = list((tmp_vault / "Stubs").iterdir())
    assert len(stubs) == 1, (
        f"Expected exactly ONE stub for the DOI-vs-doiless refs (dedup by stub-title-index); "
        f"got {len(stubs)}: {[s.name for s in stubs]}"
    )

    stub_content = stubs[0].read_text(encoding="utf-8")
    assert f'stub_key: "{shared_doi}"' in stub_content, (
        f"Surviving stub must be keyed by the DOI ({shared_doi}); got: {stub_content}"
    )
    assert "Citer Paper A" in stub_content, "First citer (A, DOI-bearing) must appear in cited_by"
    assert "Citer Paper B" in stub_content, "Second citer (B, doiless) must appear in cited_by"


def test_hyphen_variant_titles_dedup_to_one_stub(tmp_vault, config):
    """BIBLIO-03/04-07: two doiless citers referencing the same not-in-vault paper with
    hyphen-vs-joined title variants -> exactly ONE stub (closes Gap C mode 2: punctuation
    collapsed to a space, not deleted)."""
    from scripts import biblio  # noqa: PLC0415
    from scripts.ingest import _assemble_paperjson, SCHEMA_VERSION  # noqa: PLC0415

    # Citer A: hyphenated title
    refs_a = [
        {
            "number": 1,
            "raw": "Chew et al., 2019 (hyphenated)",
            "doi": None,
            "title": "Cryo-EM structures of the human cation-chloride cotransporter KCC1",
            "year": 2019,
            "fill_failed": False,
        }
    ]
    pj_a = _make_paperjson_with_refs(refs=refs_a)
    pj_a["extraction"]["metadata"]["title"] = "Citer Paper A"
    pj_a["extraction"]["metadata"]["doi"] = "10.9999/citer-a-hyphen"

    (tmp_vault / "Papers" / "Citer Paper A.md").write_text(
        "# Citer Paper A\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    result_a = biblio.run_biblio(pj_a, config)
    assert not result_a.startswith("[biblio warning:"), f"run_biblio (citer A) failed: {result_a}"

    # Citer B: joined (no hyphen) title variant of the same paper
    refs_b = [
        {
            "number": 1,
            "raw": "Chew et al., 2019 (joined)",
            "doi": None,
            "title": "Cryo-EM structures of the human cationchloride cotransporter KCC1",
            "year": 2019,
            "fill_failed": False,
        }
    ]
    parsed_b = {
        "title": "Citer Paper B",
        "sections": [],
        "references": refs_b,
        "metadata": {"title": "Citer Paper B", "doi": "10.9999/citer-b-hyphen"},
    }
    provenance_b = {
        "pdf_sha256": "hyph222", "source_filename": "citer_b_hyphen.pdf",
        "mineru_version": "2.5", "backend": "hybrid_auto",
        "extracted_at": "2026-07-01T00:00:00Z",
        "normalizations_applied": [], "schema_version": SCHEMA_VERSION,
    }
    pj_b = _assemble_paperjson(parsed_b, provenance_b)
    pj_b["extraction"]["references"] = refs_b

    (tmp_vault / "Papers" / "Citer Paper B.md").write_text(
        "# Citer Paper B\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    result_b = biblio.run_biblio(pj_b, config)
    assert not result_b.startswith("[biblio warning:"), f"run_biblio (citer B) failed: {result_b}"

    stubs = list((tmp_vault / "Stubs").iterdir())
    assert len(stubs) == 1, (
        f"Expected exactly ONE stub for the hyphen/joined title variants (dedup by "
        f"stub-title-index); got {len(stubs)}: {[s.name for s in stubs]}"
    )

    stub_content = stubs[0].read_text(encoding="utf-8")
    assert "Citer Paper A" in stub_content, "First citer (A, hyphenated) must appear in cited_by"
    assert "Citer Paper B" in stub_content, "Second citer (B, joined) must appear in cited_by"


def test_dedup_normalization_strips_separators(tmp_vault, config):
    """04-07 unit: _normalize_title_for_dedup collapses hyphen/space/joined title variants
    to the SAME string, and still folds accents/case (inherited from _normalize_title_for_match)."""
    from scripts import biblio  # noqa: PLC0415

    hyphenated = biblio._normalize_title_for_dedup("cation-chloride")
    spaced = biblio._normalize_title_for_dedup("cation chloride")
    joined = biblio._normalize_title_for_dedup("cationchloride")
    assert hyphenated == spaced == joined, (
        f"_normalize_title_for_dedup must converge hyphen/space/joined variants; "
        f"got hyphenated={hyphenated!r} spaced={spaced!r} joined={joined!r}"
    )

    # Accent folding + case still hold (inherited from _normalize_title_for_match)
    assert biblio._normalize_title_for_dedup("Müller") == biblio._normalize_title_for_dedup(
        "Muller"
    ), "_normalize_title_for_dedup must still fold accents"


def test_upgrade_does_not_relink_unrelated_reference(tmp_vault, config):
    """CR-02 (code review): upgrading a stub whose normalized title is a character
    substring of ANOTHER unresolved reference's text must NOT rewrite that other
    reference's line. Identity comes from the exact <!--stub:{key}--> anchor
    rendered by the miss branch, never from fuzzy substring matching
    (e.g. 'learning' in 'lecunetaldeeplearningnature2015')."""
    from scripts import biblio  # noqa: PLC0415

    # Citer A cites TWO different not-in-vault papers: the short/generic
    # "Learning" and "Deep Learning" (whose rendered raw text contains
    # "Learning" as a substring after normalization).
    refs = [
        {"number": 1, "raw": "Short, 2020", "doi": None,
         "title": "Learning", "year": 2020, "fill_failed": False},
        {"number": 2, "raw": "LeCun et al., Deep Learning, Nature 2015", "doi": None,
         "title": "Deep Learning", "year": 2015, "fill_failed": False},
    ]
    pj_a = _make_paperjson_with_refs(refs=refs)
    pj_a["extraction"]["metadata"]["title"] = "Citer A"
    pj_a["extraction"]["metadata"]["doi"] = "10.9999/citer-a"

    (tmp_vault / "Papers" / "Citer A.md").write_text(
        "# Citer A\n\n## My Notes\n\n",
        encoding="utf-8",
    )
    result_a = biblio.run_biblio(pj_a, config)
    assert not result_a.startswith("[biblio warning:"), f"run_biblio (citer A) failed: {result_a}"
    stubs = sorted(p.name for p in (tmp_vault / "Stubs").iterdir())
    assert stubs == ["Deep Learning.md", "Learning.md"], f"Setup precondition: two stubs, got {stubs}"

    # Ingest "Learning" itself — its stub upgrades and Citer A's ref 1 relinks.
    full_note_file = tmp_vault / "Papers" / "Learning.md"
    full_note_file.write_text(
        '---\ntitle: "Learning"\n---\n\n# Learning\n\n## My Notes\n\n',
        encoding="utf-8",
    )
    pj_b = _make_paperjson_with_refs(refs=[])
    pj_b["extraction"]["metadata"]["title"] = "Learning"
    pj_b["extraction"]["metadata"]["doi"] = None
    result_b = biblio.run_biblio(pj_b, config)
    assert not result_b.startswith("[biblio warning:"), f"run_biblio (paper B) failed: {result_b}"

    citer_content = (tmp_vault / "Papers" / "Citer A.md").read_text(encoding="utf-8")
    assert "1. [[Learning]]" in citer_content, (
        "Ref 1 ('Learning') must be relinked to [[Learning]] after upgrade"
    )
    assert "LeCun et al., Deep Learning, Nature 2015 *(not yet in vault)*" in citer_content, (
        "Ref 2 ('Deep Learning') is UNRELATED and must remain an unresolved "
        "miss-branch line — substring matching must not relink it (CR-02)"
    )
    assert citer_content.count("[[Learning]]") == 1, (
        "Exactly one line must be relinked — no cross-reference contamination"
    )
    # The upgraded stub is gone; the unrelated stub survives
    assert not (tmp_vault / "Stubs" / "Learning.md").exists()
    assert (tmp_vault / "Stubs" / "Deep Learning.md").exists()


def test_cited_by_dedup_scoped_to_frontmatter_block(tmp_vault, config):
    """WR-03 (code review): the 'already listed' check in _append_cited_by must be
    scoped to the parsed cited_by block — a quoted occurrence of the citing path
    in the stub BODY (e.g. inside the raw citation text) must not suppress the
    append and silently lose the backlink (BIBLIO-03b accumulation guarantee)."""
    from scripts import biblio  # noqa: PLC0415

    stub_rel = "Stubs/Poisoned Stub.md"
    (tmp_vault / "Stubs" / "Poisoned Stub.md").write_text(
        '---\ntitle: "Poisoned Stub"\nauthors: []\nstatus: stub\n'
        'stub_key: "sha256:poison0000000000"\ncited_by:\n  - "Papers/First Citer.md"\n---\n\n'
        "**Raw citation:**\n"
        'See also "Papers/Second Citer.md" in the appendix.\n',  # body collision
        encoding="utf-8",
    )

    biblio._append_cited_by(stub_rel, "Papers/Second Citer.md", config)

    content = (tmp_vault / "Stubs" / "Poisoned Stub.md").read_text(encoding="utf-8")
    assert biblio._parse_cited_by(content) == [
        "Papers/First Citer.md",
        "Papers/Second Citer.md",
    ], "Second citer must be appended despite a quoted body occurrence of its path"

    # Idempotency: a second call with the same path must NOT duplicate the entry
    biblio._append_cited_by(stub_rel, "Papers/Second Citer.md", config)
    content = (tmp_vault / "Stubs" / "Poisoned Stub.md").read_text(encoding="utf-8")
    assert biblio._parse_cited_by(content).count("Papers/Second Citer.md") == 1


def test_match_key_rejects_malformed_doi(tmp_vault, config):
    """WR-04 (code review): _match_key must not accept a syntactically invalid
    'DOI' as a stable identity key — it falls back to the title-hash chain so
    the key converges with a later valid-DOI cite or real ingest."""
    from scripts import biblio  # noqa: PLC0415

    title = "Deep Learning"
    title_hash_key = biblio._match_key({"doi": None, "title": title})
    assert title_hash_key.startswith("sha256:"), "Precondition: doiless key is a title hash"

    # Malformed DOIs (no 10.<digits>/ prefix, truncated, junk) → title-hash fallback
    for bad_doi in ["doi:pending", "10./truncated", "not-a-doi", "10.12/short"]:
        assert biblio._match_key({"doi": bad_doi, "title": title}) == title_hash_key, (
            f"Malformed DOI {bad_doi!r} must fall back to the title-hash key"
        )

    # A syntactically valid DOI is still used verbatim
    assert biblio._match_key({"doi": "10.1038/s41586-019-1438-2", "title": title}) == (
        "10.1038/s41586-019-1438-2"
    )


def test_missing_citing_note_short_circuits_before_stub_creation(tmp_vault, config):
    """WR-05 (code review): if the citing note does not exist (e.g. step 12b's
    note.generate_note failed non-fatally), run_biblio must return a
    [biblio warning:] BEFORE creating any stubs or mutating cited_by lists —
    otherwise stubs are left with dangling backlinks a later successful re-run
    will not repair (the 'already listed' dedup treats them as recorded)."""
    from scripts import biblio  # noqa: PLC0415

    refs = [
        {"number": 1, "raw": "LeCun et al., Nature 2015", "doi": None,
         "title": "Deep Learning", "year": 2015, "fill_failed": False},
    ]
    pj = _make_paperjson_with_refs(refs=refs)
    # Deliberately do NOT pre-write Papers/Test Citing Paper.md

    result = biblio.run_biblio(pj, config)

    assert result.startswith("[biblio warning:"), (
        f"run_biblio must warn when the citing note is missing; got {result!r}"
    )
    stubs = list((tmp_vault / "Stubs").iterdir())
    assert stubs == [], (
        f"No stubs may be created when the citing note is missing (non-atomic "
        f"side effects); got {[s.name for s in stubs]}"
    )


def test_stub_frontmatter_escapes_backslash_in_title(tmp_vault, config):
    """CR-01 (code review): a title with a trailing backslash must not break the
    stub's YAML frontmatter — backslashes are escaped BEFORE quotes so the closing
    quote survives and subsequent keys (cited_by:) are not swallowed."""
    from scripts import biblio  # noqa: PLC0415

    ref = {
        "number": 1,
        "raw": "Evil et al., 2020",
        "doi": None,
        "title": "Structure of X\\",  # literal trailing backslash (LaTeX-ish)
        "year": 2020,
        "fill_failed": False,
    }
    rendered = biblio._render_stub(ref, "sha256:deadbeefdeadbeef", "Papers/Citer.md")

    # The escaped form must double the backslash so it cannot escape the closing quote
    assert 'title: "Structure of X\\\\"' in rendered, (
        f"Backslash must be escaped as \\\\ inside the double-quoted title; got:\n{rendered}"
    )
    # The frontmatter structure must survive: cited_by block still parseable
    assert biblio._parse_cited_by(rendered) == ["Papers/Citer.md"], (
        "cited_by must remain a parseable frontmatter block after backslash escaping"
    )
    # And the helper itself escapes in the correct order (backslash first, then quote)
    assert biblio._yaml_escape('a\\"b') == 'a\\\\\\"b'


def test_layer2_wikilink_unregressed_by_dedup(tmp_vault, config):
    """GUARD (04-07): a doiless ref to an in-vault DOI-keyed paper still renders a
    [[wikilink]] and creates NO stub after the dedup change (Layer-2 unregressed)."""
    from scripts import biblio  # noqa: PLC0415

    # Seed registry with a DOI-keyed entry (paper already in vault)
    registry = {
        "10.1073/pnas.2020": {
            "title": "Cryo EM Structures of KCC1",
            "authors": None,
            "year": 2020,
        }
    }
    Path(config["registry_path"]).write_text(json.dumps(registry), encoding="utf-8")

    refs = [
        {
            "number": 1,
            "raw": "Bhatt et al., PNAS 2020",
            "doi": None,
            "title": "Cryo-EM structures of KCC1",
            "year": 2020,
            "fill_failed": False,
        }
    ]
    pj = _make_paperjson_with_refs(refs=refs)

    (tmp_vault / "Papers" / "Test Citing Paper.md").write_text(
        "# Test Citing Paper\n\n## My Notes\n\n",
        encoding="utf-8",
    )

    result = biblio.run_biblio(pj, config)
    assert not result.startswith("[biblio warning:"), f"run_biblio failed: {result}"

    note_content = (tmp_vault / "Papers" / "Test Citing Paper.md").read_text(encoding="utf-8")
    assert "[[Cryo EM Structures of KCC1]]" in note_content, (
        "Doiless ref matching DOI-keyed registry entry must still render as [[wikilink]]"
    )

    stubs = list((tmp_vault / "Stubs").iterdir())
    assert len(stubs) == 0, (
        f"No stub should be created for a ref that resolves to an in-vault paper; "
        f"got {[s.name for s in stubs]}"
    )
