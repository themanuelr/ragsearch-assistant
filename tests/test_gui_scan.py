"""Tests for gui/scan.py — live pipeline-state scan (D-09, Phase 8 Plan 02).

Every test uses the ``gui_config`` tmp-scoped fixture (tests/conftest.py) so
none of the session-scoped ``_guard_real_*`` pollution guards ever trip.
scripts.embed/biblio/ingest/link/note are imported directly (never re-derived)
to build fixtures identical in shape to what the real write-side pipeline
produces.

Run with: python -m pytest tests/test_gui_scan.py -x
"""

import json
import pathlib
import re

from gui import scan
from scripts.note import _sanitize_filename

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMPTY_ANALYSIS = {
    "generated_by": None,
    "summary": None,
    "claims": [],
    "methods_overview": None,
    "results": None,
    "limitations": [],
    "open_questions": [],
    "entities": [],
    "topics": [],
    "connections": {"builds_on": [], "contradicts": [], "same_domain": []},
}


def _write_json(path, data):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _write_registry(gui_config, registry):
    _write_json(gui_config["registry_path"], registry)


def _registry_entry(title, paperjson_path, projects, **extra):
    entry = {
        "title": title,
        "authors": ["A. Author"],
        "year": 2020,
        "journal": "Some Journal",
        "doi": "10.1234/" + _sanitize_filename(title).lower().replace(" ", "-"),
        "arxiv_id": None,
        "projects": projects,
        "source_path": "/x/source.pdf",
        "paperjson_path": paperjson_path,
        "summary": None,
        "key_findings": None,
    }
    entry.update(extra)
    return entry


def _write_paperjson(gui_config, stem, analysis):
    path = pathlib.Path(gui_config["paperjson_cache_dir"]) / f"{stem}.json"
    data = {"extraction": {"metadata": {}}, "analysis": analysis, "provenance": {}}
    _write_json(path, data)
    return path


def _write_note(gui_config, title, refs=False, topics=False, tags=None):
    filename = _sanitize_filename(title)
    papers_dir = pathlib.Path(gui_config["vault_path"]) / "Papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    tags = tags or []
    tags_block = "".join(f'  - "{t}"\n' for t in tags)
    content = (
        "---\n"
        f'title: "{title}"\n'
        + ("tags:\n" + tags_block if tags else "")
        + "---\n\n"
        f"# {title}\n\n"
        "## Summary\n\nSome summary text.\n\n"
    )
    if refs:
        content += "## References\n\n- Some Ref\n\n"
    if topics:
        content += "## Related Topics\n\n[[Some Topic]]\n\n"
    content += "## My Notes\n"
    target = papers_dir / f"{filename}.md"
    target.write_text(content, encoding="utf-8")
    return target


def _get_chroma_collection(chroma_db_path):
    import chromadb  # noqa: PLC0415

    client = chromadb.PersistentClient(path=str(chroma_db_path))
    return client.get_or_create_collection(
        name="papers", configuration={"hnsw": {"space": "cosine"}}
    )


def _snapshot_files(root):
    root = pathlib.Path(root)
    if not root.exists():
        return set()
    return {p.relative_to(root).as_posix() for p in root.rglob("*")}


# ---------------------------------------------------------------------------
# Test 1: mineru stage
# ---------------------------------------------------------------------------

def test_mineru_stage_detected_from_content_list_json(gui_config):
    paperjson_path = str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "paper-one.json")
    entry_with = _registry_entry("Paper One", paperjson_path, ["proj-a"])
    entry_without = _registry_entry(
        "Paper Two",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "paper-two.json"),
        ["proj-a"],
    )
    _write_registry(gui_config, {"key-one": entry_with, "key-two": entry_without})

    mineru_dir = pathlib.Path(gui_config["mineru_output_dir"]) / "paper-one" / "auto"
    mineru_dir.mkdir(parents=True, exist_ok=True)
    (mineru_dir / "content_list.json").write_text("[]", encoding="utf-8")

    paper_with = scan.scan_paper(gui_config, "key-one")
    paper_without = scan.scan_paper(gui_config, "key-two")

    assert paper_with["stages"]["mineru"] is True
    assert paper_without["stages"]["mineru"] is False


# ---------------------------------------------------------------------------
# Test 2: registry + paperjson_fill stages
# ---------------------------------------------------------------------------

def test_registry_and_paperjson_fill_stages(gui_config):
    paperjson_path = str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "filled.json")
    entry = _registry_entry("Filled Paper", paperjson_path, ["proj-a"])
    _write_registry(gui_config, {"key-filled": entry})

    populated = dict(_EMPTY_ANALYSIS)
    populated["summary"] = "A real summary."
    populated["generated_by"] = "gemma4:e4b"
    _write_paperjson(gui_config, "filled", populated)

    paper = scan.scan_paper(gui_config, "key-filled")
    assert paper["stages"]["registry"] is True
    assert paper["stages"]["paperjson_fill"] is True

    # Empty-skeleton analysis makes paperjson_fill False
    entry2 = _registry_entry(
        "Empty Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "empty.json"),
        ["proj-a"],
    )
    _write_registry(gui_config, {"key-filled": entry, "key-empty": entry2})
    _write_paperjson(gui_config, "empty", _EMPTY_ANALYSIS)

    paper_empty = scan.scan_paper(gui_config, "key-empty")
    assert paper_empty["stages"]["paperjson_fill"] is False


# ---------------------------------------------------------------------------
# Test 3: note / biblio / topics stages
# ---------------------------------------------------------------------------

def test_note_biblio_topics_stages(gui_config):
    title = "Note Paper"
    entry = _registry_entry(
        title, str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "note-paper.json"), ["proj-a"]
    )
    _write_registry(gui_config, {"key-note": entry})
    _write_note(gui_config, title, refs=True, topics=True, tags=["transformers"])

    paper = scan.scan_paper(gui_config, "key-note")
    assert paper["stages"]["note"] is True
    assert paper["stages"]["biblio"] is True
    assert paper["stages"]["topics"] is True
    assert paper["metadata"]["tags"] == ["transformers"]

    # No note on disk -> all three False
    entry2 = _registry_entry(
        "No Note Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "no-note.json"),
        ["proj-a"],
    )
    _write_registry(gui_config, {"key-note": entry, "key-nonote": entry2})
    paper_nonote = scan.scan_paper(gui_config, "key-nonote")
    assert paper_nonote["stages"]["note"] is False
    assert paper_nonote["stages"]["biblio"] is False
    assert paper_nonote["stages"]["topics"] is False


# ---------------------------------------------------------------------------
# Test 4: embed stage
# ---------------------------------------------------------------------------

def test_embed_stage_true_when_chroma_entry_present(gui_config):
    entry = _registry_entry(
        "Embedded Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "embedded.json"),
        ["proj-a"],
    )
    _write_registry(gui_config, {"key-embed": entry})

    collection = _get_chroma_collection(gui_config["chroma_db_path"])
    collection.add(
        ids=["key-embed::0-abstract::0"],
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        documents=["search_document: some text"],
        metadatas=[{"registry_key": "key-embed", "status": "paper"}],
    )

    paper = scan.scan_paper(gui_config, "key-embed")
    assert paper["stages"]["embed"] is True


def test_embed_stage_unknown_when_chroma_dir_absent(gui_config):
    entry = _registry_entry(
        "No Chroma Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "no-chroma.json"),
        ["proj-a"],
    )
    _write_registry(gui_config, {"key-nochroma": entry})

    paper = scan.scan_paper(gui_config, "key-nochroma")
    assert paper["stages"]["embed"] is None
    assert not pathlib.Path(gui_config["chroma_db_path"]).exists(), (
        "embed stage check must never create the chroma_db_path directory"
    )


# ---------------------------------------------------------------------------
# Test 5: project filter
# ---------------------------------------------------------------------------

def test_scan_project_papers_filters_by_project_membership(gui_config):
    gui_config["project_name"] = "proj-a"
    entry_a = _registry_entry(
        "Project A Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "proj-a-paper.json"),
        ["proj-a"],
    )
    entry_b = _registry_entry(
        "Project B Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "proj-b-paper.json"),
        ["proj-b"],
    )
    entry_both = _registry_entry(
        "Shared Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "shared-paper.json"),
        ["proj-a", "proj-b"],
    )
    _write_registry(
        gui_config,
        {"key-a": entry_a, "key-b": entry_b, "key-both": entry_both},
    )

    papers = scan.scan_project_papers(gui_config)
    keys = {p["registry_key"] for p in papers}
    assert keys == {"key-a", "key-both"}


# ---------------------------------------------------------------------------
# Test 6: uningested drop-folder scan
# ---------------------------------------------------------------------------

def test_scan_uningested_lists_only_direct_pdfs_sorted(gui_config):
    uningested_dir = pathlib.Path(gui_config["uningested_dir"])
    uningested_dir.mkdir(parents=True, exist_ok=True)
    (uningested_dir / "b.pdf").write_bytes(b"%PDF-1.4")
    (uningested_dir / "a.pdf").write_bytes(b"%PDF-1.4")
    (uningested_dir / "notes.txt").write_text("not a pdf", encoding="utf-8")
    nested = uningested_dir / "subdir"
    nested.mkdir()
    (nested / "c.pdf").write_bytes(b"%PDF-1.4")

    pending = scan.scan_uningested(gui_config)
    names = [p["name"] for p in pending]
    assert names == ["a.pdf", "b.pdf"], "must be non-recursive and sorted by name"


def test_scan_uningested_missing_dir_returns_empty_without_creating(gui_config):
    uningested_dir = pathlib.Path(gui_config["uningested_dir"])
    assert not uningested_dir.exists()

    pending = scan.scan_uningested(gui_config)

    assert pending == []
    assert not uningested_dir.exists(), "scan_uningested must never create the drop folder"


# ---------------------------------------------------------------------------
# Test 7: read-only guarantee
# ---------------------------------------------------------------------------

def test_scan_is_read_only(gui_config, tmp_path):
    title = "Readonly Paper"
    entry = _registry_entry(
        title, str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "readonly.json"), ["proj-a"]
    )
    gui_config["project_name"] = "proj-a"
    _write_registry(gui_config, {"key-readonly": entry})
    _write_paperjson(gui_config, "readonly", _EMPTY_ANALYSIS)
    _write_note(gui_config, title, refs=True, topics=True)

    mineru_dir = pathlib.Path(gui_config["mineru_output_dir"]) / "readonly" / "auto"
    mineru_dir.mkdir(parents=True, exist_ok=True)
    (mineru_dir / "content_list.json").write_text("[]", encoding="utf-8")

    (pathlib.Path(gui_config["uningested_dir"])).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(gui_config["uningested_dir"]) / "waiting.pdf").write_bytes(b"%PDF-1.4")

    before = _snapshot_files(tmp_path)

    scan.scan_project_papers(gui_config)
    scan.scan_paper(gui_config, "key-readonly")
    scan.scan_uningested(gui_config)

    after = _snapshot_files(tmp_path)
    assert after == before, f"scan created new paths: {after - before}"


# ---------------------------------------------------------------------------
# Test 8: write-side contract reuse (identity, not equality)
# ---------------------------------------------------------------------------

def test_scan_reuses_write_side_contracts():
    from scripts import biblio, link

    assert scan._REFS_SECTION_RE is biblio._REFS_SECTION_RE
    assert scan._RELATED_TOPICS_RE is link._RELATED_TOPICS_RE


# ---------------------------------------------------------------------------
# Phase 8 Plan 06, Task 1: scan_universal (D-11) + Overview (Universal) page
# ---------------------------------------------------------------------------

def _patch_overview_config(monkeypatch, gui_config):
    import gui.routes.overview as overview_routes

    monkeypatch.setattr(overview_routes, "load_gui_config", lambda: gui_config)


def _capturing_enqueue(monkeypatch):
    """Monkeypatch gui.jobs.enqueue to capture (action, argv) without spawning
    a real subprocess; still registers a real JobState so the route's
    gui_jobs.get(job_id) call and job_status.html render normally (mirrors
    tests/test_gui_processing.py's pattern)."""
    import gui.jobs as gui_jobs

    captured = {"calls": []}

    def _fake_enqueue(action, argv, on_success=None):
        captured["calls"].append({"action": action, "argv": argv})
        job_id = f"fake-{len(captured['calls'])}-{action}"
        gui_jobs.JOBS[job_id] = gui_jobs.JobState(
            id=job_id, action=action, argv=argv, status="queued",
            created_at="2026-01-01T00:00:00",
        )
        return job_id

    monkeypatch.setattr(gui_jobs, "enqueue", _fake_enqueue)
    monkeypatch.setattr(gui_jobs, "JOBS", {})
    return captured


def test_scan_universal_returns_all_entries_with_membership_and_cache_presence(gui_config):
    gui_config["project_name"] = "proj-a"
    entry_a = _registry_entry(
        "Project A Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "paper-a.json"),
        ["proj-a"],
    )
    entry_b = _registry_entry(
        "Project B Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "paper-b.json"),
        ["proj-b"],
    )
    _write_registry(gui_config, {"key-a": entry_a, "key-b": entry_b})

    # key-a gets a MinerU output + PaperJSON cache; key-b gets neither.
    mineru_dir = pathlib.Path(gui_config["mineru_output_dir"]) / "paper-a" / "auto"
    mineru_dir.mkdir(parents=True, exist_ok=True)
    (mineru_dir / "content_list.json").write_text("[]", encoding="utf-8")
    _write_paperjson(gui_config, "paper-a", _EMPTY_ANALYSIS)

    entries = scan.scan_universal(gui_config)
    by_key = {e["registry_key"]: e for e in entries}

    assert set(by_key) == {"key-a", "key-b"}

    assert by_key["key-a"]["projects"] == ["proj-a"]
    assert by_key["key-a"]["in_this_project"] is True
    assert by_key["key-a"]["mineru_present"] is True
    assert by_key["key-a"]["paperjson_present"] is True

    assert by_key["key-b"]["projects"] == ["proj-b"]
    assert by_key["key-b"]["in_this_project"] is False
    assert by_key["key-b"]["mineru_present"] is False
    assert by_key["key-b"]["paperjson_present"] is False


def test_universal_readonly(gui_client, gui_config, monkeypatch, tmp_path):
    """scan_universal + a full page render must never create a new path under
    tmp_path, and the page's only POST target must be /jobs/enqueue with
    action=ingest_doi (D-11 locked rejection: no other action exists)."""
    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    entry = _registry_entry(
        "Readonly Universal Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "readonly-universal.json"),
        ["proj-b"],
    )
    _write_registry(gui_config, {"key-ru": entry})

    before = _snapshot_files(tmp_path)
    scan.scan_universal(gui_config)
    resp = gui_client.get("/overview/universal")
    after = _snapshot_files(tmp_path)

    assert resp.status_code == 200
    assert after == before, f"scan_universal / page render created new paths: {after - before}"

    assert 'hx-post="/jobs/enqueue"' in resp.text
    assert 'name="action" value="ingest_doi"' in resp.text
    assert "/processing/" not in resp.text


def test_overview_universal_page_shows_import_cta_only_for_out_of_project_rows(
    gui_client, gui_config, monkeypatch
):
    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    entry_in = _registry_entry(
        "In Project Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "in-project.json"),
        ["proj-a"],
    )
    entry_out = _registry_entry(
        "Out Of Project Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "out-of-project.json"),
        ["proj-b"],
    )
    _write_registry(gui_config, {"key-in": entry_in, "key-out": entry_out})

    resp = gui_client.get("/overview/universal")

    assert resp.status_code == 200
    assert "In Project Paper" in resp.text
    assert "Out Of Project Paper" in resp.text
    # Only the out-of-project row carries the CTA.
    assert resp.text.count("Import into This Project") == 1


def test_overview_universal_import_enqueues_doi_argv(gui_client, gui_config, monkeypatch):
    captured = _capturing_enqueue(monkeypatch)

    resp = gui_client.post(
        "/jobs/enqueue", data={"action": "ingest_doi", "doi": "10.1234/out-of-project"}
    )

    assert resp.status_code == 200
    assert len(captured["calls"]) == 1
    argv = captured["calls"][0]["argv"]
    assert "--doi" in argv
    assert "10.1234/out-of-project" in argv


# ---------------------------------------------------------------------------
# Phase 8 Plan 06, Task 2: per-stage re-run buttons on the drill-down (D-12)
# ---------------------------------------------------------------------------

def test_paper_row_rerun_buttons_present_when_paperjson_cache_present(
    gui_client, gui_config, monkeypatch
):
    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    title = "Rerun Ready Paper"
    paperjson_path = str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "rerun-ready.json")
    entry = _registry_entry(title, paperjson_path, ["proj-a"])
    _write_registry(gui_config, {"key-rerun": entry})
    _write_paperjson(gui_config, "rerun-ready", _EMPTY_ANALYSIS)

    resp = gui_client.get("/overview/project/paper/key-rerun")

    assert resp.status_code == 200
    for label in ("Re-run Note", "Re-run Biblio", "Re-run Embed", "Re-run Link"):
        assert label in resp.text


def test_paper_row_rerun_targets_valid_css_id_for_doi_key(gui_client, gui_config, monkeypatch):
    """08-08 Gap 2 (D-12 / GUI-08): a real DOI-shaped registry key (contains
    '.' and '/', both invalid unescaped in a CSS id selector) must still
    produce a matching, syntactically valid hx-target/id pair for every
    rerun stage."""
    import re

    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    doi_key = "10.1021/jacs.3c10258"
    title = "DOI Keyed Paper"
    paperjson_path = str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "doi-keyed.json")
    entry = _registry_entry(title, paperjson_path, ["proj-a"])
    _write_registry(gui_config, {doi_key: entry})
    _write_paperjson(gui_config, "doi-keyed", _EMPTY_ANALYSIS)

    resp = gui_client.get(f"/overview/project/paper/{doi_key}")

    assert resp.status_code == 200
    expected_slug = "10-1021-jacs-3c10258"
    assert re.match(r"^[A-Za-z0-9_-]+$", expected_slug)
    for stage in ("note", "biblio", "embed", "link"):
        expected_id = f"rerun-status-{stage}-{expected_slug}"
        assert f'hx-target="#{expected_id}"' in resp.text
        assert f'id="{expected_id}"' in resp.text
    assert "rerun-status-note-10.1021" not in resp.text


def test_paper_row_rerun_note_omitted_without_paperjson_cache(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    title = "No Cache Paper"
    paperjson_path = str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "no-cache.json")
    entry = _registry_entry(title, paperjson_path, ["proj-a"])
    _write_registry(gui_config, {"key-nocache": entry})
    # Deliberately never write the PaperJSON cache file for this entry.

    resp = gui_client.get("/overview/project/paper/key-nocache")

    assert resp.status_code == 200
    assert "Re-run Note" not in resp.text
    assert "Re-run Biblio" not in resp.text
    assert "Re-run Embed" not in resp.text
    assert "Re-run Link" not in resp.text
    assert "PaperJSON cache" in resp.text


def test_paper_row_rerun_biblio_enqueues_expected_argv(gui_client, gui_config, monkeypatch):
    import sys

    captured = _capturing_enqueue(monkeypatch)

    resp = gui_client.post("/jobs/enqueue", data={"action": "biblio", "stem": "rerun-ready"})

    assert resp.status_code == 200
    assert len(captured["calls"]) == 1
    argv = captured["calls"][0]["argv"]
    assert argv[0] == sys.executable
    assert argv[1].endswith("biblio.py")
    assert argv[-2:] == ["--stem", "rerun-ready"]


# ---------------------------------------------------------------------------
# 08-12 Task 2 (Gap B, GUI-03/D-12): two-way paper-row drill-down toggle
#
# The expand/collapse behaviour itself is browser-only (a click handler
# clearing or fetching into the cell) and isn't unit-testable in this stack.
# What CAN regress silently -- and 08-08 closed a bug of exactly this shape,
# an hx-target that didn't resolve to its container -- is the row/cell id
# correspondence and the toggle wiring: the row's detail-cell data attribute
# must equal the sibling detail cell's real id, and the drill-down URL data
# attribute must address that paper's own route, or a click expands (or
# collapses) the wrong row.
# ---------------------------------------------------------------------------

def test_overview_project_row_toggle_wiring_matches_detail_cell_and_url(
    gui_client, gui_config, monkeypatch
):
    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    entry_a = _registry_entry(
        "Toggle Paper A",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "toggle-a.json"),
        ["proj-a"],
    )
    entry_b = _registry_entry(
        "Toggle Paper B",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "toggle-b.json"),
        ["proj-a"],
    )
    _write_registry(gui_config, {"key-toggle-a": entry_a, "key-toggle-b": entry_b})

    resp = gui_client.get("/overview/project")

    assert resp.status_code == 200
    # No declarative one-way fill left on the ROW itself -- scoped to each
    # <tr class="paper-row" ...> opening tag (not the whole page) since
    # Phase 08.1 Plan 01 (D-01..D-05) legitimately added page-level hx-get/
    # hx-target search+pagination controls (the shared picker) elsewhere on
    # this same page; the row's own toggle mechanism must still be pure JS.
    row_tags = re.findall(r'<tr class="paper-row"[^>]*>', resp.text)
    assert len(row_tags) == 2
    for row_tag in row_tags:
        assert "hx-get=" not in row_tag
        assert "hx-target=" not in row_tag
    # Row carries the click handler.
    assert 'onclick="paperRowToggle(this)"' in resp.text
    # Each row's detail-cell data attribute matches its sibling detail cell's
    # real id, and its URL data attribute addresses that paper's own route.
    for index, key in enumerate(("key-toggle-a", "key-toggle-b")):
        expected_cell_id = f"paper-detail-{index}"
        assert f'data-detail-cell="{expected_cell_id}"' in resp.text
        assert f'id="{expected_cell_id}"' in resp.text
        assert f'data-drilldown-url="/overview/project/paper/{key}"' in resp.text
    # The toggle function is defined exactly once regardless of paper count.
    assert resp.text.count("window.paperRowToggle = function") == 1


def test_overview_project_route_and_drilldown_partial_unchanged(gui_client, gui_config, monkeypatch):
    """gui/routes/overview.py, the {registry_key:path} route contract, and
    the drill-down partial itself are out of scope for this gap -- only the
    row's toggle wiring changed. A drill-down request still resolves the
    real paper partial."""
    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    entry = _registry_entry(
        "Untouched Route Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "untouched.json"),
        ["proj-a"],
    )
    _write_registry(gui_config, {"key-untouched": entry})

    resp = gui_client.get("/overview/project/paper/key-untouched")

    assert resp.status_code == 200
    assert "Untouched Route Paper" in resp.text


# ---------------------------------------------------------------------------
# 08-13 gap closure: one predicate definition, two distinct column labels
# ---------------------------------------------------------------------------

def test_stage_paperjson_fill_delegates_to_note_predicate():
    """Test K: gui.scan no longer defines its own populated-keys tuple;
    _stage_paperjson_fill delegates to scripts.note._analysis_is_populated
    -- one shared predicate, imported not re-derived (Don't Hand-Roll)."""
    from scripts import note

    assert not hasattr(scan, "_ANALYSIS_POPULATED_KEYS"), (
        "gui/scan.py must not define its own populated-keys tuple anymore"
    )
    assert scan._analysis_is_populated is note._analysis_is_populated


def test_stage_paperjson_fill_follows_note_predicate_behaviorally(gui_config):
    """Test K (behavioral): a cache file whose analysis note.py deems
    populated reads True through _stage_paperjson_fill, and one with the
    empty skeleton reads False -- asserted via behavior, not by patching
    internals, so the test survives refactors."""
    paperjson_path = pathlib.Path(gui_config["paperjson_cache_dir"]) / "k-behavior.json"

    populated = dict(_EMPTY_ANALYSIS)
    populated["summary"] = "A real summary."
    populated["generated_by"] = "gemma4:e4b"
    _write_json(paperjson_path, {"extraction": {"metadata": {}}, "analysis": populated, "provenance": {}})
    assert scan._stage_paperjson_fill(paperjson_path) is True

    _write_json(
        paperjson_path,
        {"extraction": {"metadata": {}}, "analysis": _EMPTY_ANALYSIS, "provenance": {}},
    )
    assert scan._stage_paperjson_fill(paperjson_path) is False


def test_stage_paperjson_fill_tristate_preserved(gui_config):
    """Test L: absent -> False, unparseable -> None, neither raises. The
    tri-state contract and the read-only never-create guarantee (D-09) are
    unchanged by the predicate delegation."""
    absent_path = pathlib.Path(gui_config["paperjson_cache_dir"]) / "does-not-exist.json"
    assert scan._stage_paperjson_fill(absent_path) is False
    assert not absent_path.exists(), "must never create the path (D-09)"

    unparseable_path = pathlib.Path(gui_config["paperjson_cache_dir"]) / "unparseable.json"
    unparseable_path.parent.mkdir(parents=True, exist_ok=True)
    unparseable_path.write_text("not valid json {{{", encoding="utf-8")
    assert scan._stage_paperjson_fill(unparseable_path) is None


def test_project_and_universal_paperjson_headers_are_distinct(gui_client, gui_config, monkeypatch):
    """Test M: the Project and Universal PaperJSON column headers render
    distinct text, so the two predicates are not presented as the same
    claim under near-identical labels."""
    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    entry = _registry_entry(
        "Header Test Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "header-test.json"),
        ["proj-a"],
    )
    _write_registry(gui_config, {"key-header": entry})

    project_resp = gui_client.get("/overview/project")
    universal_resp = gui_client.get("/overview/universal")

    assert project_resp.status_code == 200
    assert universal_resp.status_code == 200

    project_header_match = re.search(r"<th[^>]*>([^<]*Filled[^<]*)</th>", project_resp.text)
    universal_header_match = re.search(r"<th[^>]*>([^<]*Cached[^<]*)</th>", universal_resp.text)

    assert project_header_match is not None, "Project page must have a header naming what it measures"
    assert universal_header_match is not None, "Universal page must have a header naming what it measures"
    assert project_header_match.group(1) != universal_header_match.group(1), (
        "the two PaperJSON column headers must not read identically"
    )


def test_project_and_universal_consistent_for_filled_paper(gui_config):
    """Test N (the user's actual complaint): for a paper whose cache exists
    AND whose analysis is populated, Project and Universal both report their
    PaperJSON indicator as true -- the contradiction the user saw is gone
    once the analysis is actually persisted."""
    gui_config["project_name"] = "proj-a"
    title = "Consistent Paper"
    paperjson_path = str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "consistent.json")
    entry = _registry_entry(title, paperjson_path, ["proj-a"])
    _write_registry(gui_config, {"key-consistent": entry})

    populated = dict(_EMPTY_ANALYSIS)
    populated["summary"] = "A real, persisted summary."
    populated["generated_by"] = "gemma4:e4b"
    _write_paperjson(gui_config, "consistent", populated)

    project_papers = scan.scan_project_papers(gui_config)
    project_paper = next(p for p in project_papers if p["registry_key"] == "key-consistent")
    assert project_paper["stages"]["paperjson_fill"] is True

    universal_entries = scan.scan_universal(gui_config)
    universal_entry = next(e for e in universal_entries if e["registry_key"] == "key-consistent")
    assert universal_entry["paperjson_present"] is True
