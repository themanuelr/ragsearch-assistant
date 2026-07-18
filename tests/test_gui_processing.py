"""Tests for the Processing page's Drop Folder / URL ingest / Per-Paper
Orchestration flows (Phase 8 Plan 05, D-10 + ROADMAP orchestration mandate).

Every POST test monkeypatches ``gui.jobs.enqueue`` to a capturing fake (mirrors
tests/test_gui_jobs.py's pattern) so no real subprocess (MinerU, note.py,
biblio.py, embed.py, link.py) is ever spawned. The fake still registers a real
``JobState`` in ``gui.jobs.JOBS`` so the route's ``gui_jobs.get(job_id)`` call
and the ``job_status.html``/``job_status_list.html`` partials render for real.

Cleanup-hook tests (``gui.jobs.make_drop_cleanup``) call the returned callable
directly -- no job engine involved.

Run with: python -m pytest tests/test_gui_processing.py -x
"""

import json
import pathlib
import queue as queue_module

import pytest

import gui.jobs as gui_jobs
import gui.routes.processing as processing_routes

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_jobs_state(tmp_path, monkeypatch):
    """Fresh, tmp-scoped jobs engine for every test (mirrors
    tests/test_gui_jobs.py -- gui.jobs is a session-wide singleton module, so
    every test file touching it needs its own reset)."""
    monkeypatch.setattr(gui_jobs, "JOBS_DIR", tmp_path / "gui_jobs")
    monkeypatch.setattr(gui_jobs, "JOBS", {})
    monkeypatch.setattr(gui_jobs, "_QUEUE", queue_module.Queue())
    monkeypatch.setattr(gui_jobs, "_PROC_BY_JOB", {})
    monkeypatch.setattr(gui_jobs, "_CANCELLED_QUEUED", set())
    monkeypatch.setattr(gui_jobs, "_ON_SUCCESS", {})
    monkeypatch.setattr(gui_jobs, "_WORKER_THREAD", None)
    yield


def _patch_processing_config(monkeypatch, gui_config):
    monkeypatch.setattr(processing_routes, "load_gui_config", lambda: gui_config)


def _capturing_enqueue(monkeypatch):
    """Monkeypatch gui.jobs.enqueue to capture (action, argv, on_success)
    without spawning a real subprocess; still registers a real JobState so
    gui_jobs.get() and the job_status partials render normally."""
    captured = {"calls": []}

    def _fake_enqueue(action, argv, on_success=None):
        captured["calls"].append({"action": action, "argv": argv, "on_success": on_success})
        job_id = f"fake-{len(captured['calls'])}-{action}"
        gui_jobs.JOBS[job_id] = gui_jobs.JobState(
            id=job_id, action=action, argv=argv, status="queued",
            created_at="2026-01-01T00:00:00",
        )
        return job_id

    monkeypatch.setattr(gui_jobs, "enqueue", _fake_enqueue)
    return captured


def _write_registry_entry(gui_config, registry_key, title, stem, projects=("proj-a",)):
    registry_path = pathlib.Path(gui_config["registry_path"])
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = {}
    if registry_path.exists():
        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)
    registry[registry_key] = {
        "title": title,
        "authors": ["A. Author"],
        "year": 2021,
        "journal": "J",
        "doi": registry_key,
        "arxiv_id": None,
        "projects": list(projects),
        "source_path": "/x/source.pdf",
        "paperjson_path": str(pathlib.Path(gui_config["paperjson_cache_dir"]) / f"{stem}.json"),
        "summary": None,
        "key_findings": None,
    }
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f)


# ---------------------------------------------------------------------------
# Task 1, Test 1: GET /processing drop-folder rows + empty state
# ---------------------------------------------------------------------------

def test_processing_page_renders_drop_rows_and_empty_state(gui_client, gui_config, monkeypatch):
    _patch_processing_config(monkeypatch, gui_config)

    uningested_dir = pathlib.Path(gui_config["uningested_dir"])
    uningested_dir.mkdir(parents=True, exist_ok=True)
    (uningested_dir / "paper-a.pdf").write_bytes(b"%PDF-1.4")
    (uningested_dir / "paper-b.pdf").write_bytes(b"%PDF-1.4")

    resp = gui_client.get("/processing")
    assert resp.status_code == 200
    assert resp.text.count("Ingest PDF") == 2
    assert "paper-a.pdf" in resp.text
    assert "paper-b.pdf" in resp.text

    for f in uningested_dir.glob("*.pdf"):
        f.unlink()

    resp2 = gui_client.get("/processing")
    assert resp2.status_code == 200
    assert "No PDFs waiting" in resp2.text
    assert "Drop PDF files into" in resp2.text


# ---------------------------------------------------------------------------
# Task 1, Test 2: POST /processing/ingest argv + checkboxes
# ---------------------------------------------------------------------------

def test_ingest_enqueues_pdf_argv_with_force_extract_and_refill(gui_client, gui_config, monkeypatch):
    _patch_processing_config(monkeypatch, gui_config)
    uningested_dir = pathlib.Path(gui_config["uningested_dir"])
    uningested_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = uningested_dir / "paper-a.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    captured = _capturing_enqueue(monkeypatch)

    resp = gui_client.post(
        "/processing/ingest",
        data={"filename": "paper-a.pdf", "force_extract": "true", "refill": "true"},
    )

    assert resp.status_code == 200
    assert len(captured["calls"]) == 1
    call = captured["calls"][0]
    argv = call["argv"]
    assert "--pdf" in argv
    resolved = str(pdf_path.resolve())
    assert resolved in argv
    assert argv.index(resolved) == argv.index("--pdf") + 1
    assert "--force-extract" in argv
    assert "--refill" in argv
    assert call["on_success"] is not None


def test_ingest_rejects_path_traversal_filename(gui_client, gui_config, monkeypatch):
    _patch_processing_config(monkeypatch, gui_config)
    uningested_dir = pathlib.Path(gui_config["uningested_dir"])
    uningested_dir.mkdir(parents=True, exist_ok=True)

    captured = _capturing_enqueue(monkeypatch)

    resp1 = gui_client.post("/processing/ingest", data={"filename": "..\\evil.pdf"})
    assert resp1.status_code in (400, 404)

    resp2 = gui_client.post("/processing/ingest", data={"filename": "a/b.pdf"})
    assert resp2.status_code in (400, 404)

    assert captured["calls"] == []


# ---------------------------------------------------------------------------
# Task 1, Tests 3/4/5: make_drop_cleanup hook (success / gated / containment)
# ---------------------------------------------------------------------------

def test_cleanup_hook_deletes_after_mineru_success(gui_config):
    uningested_dir = pathlib.Path(gui_config["uningested_dir"])
    uningested_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = uningested_dir / "paper-a.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    mineru_dir = pathlib.Path(gui_config["mineru_output_dir"]) / "paper-a" / "hybrid_auto"
    mineru_dir.mkdir(parents=True, exist_ok=True)
    (mineru_dir / "content_list.json").write_text("[]", encoding="utf-8")

    cleanup = gui_jobs.make_drop_cleanup(gui_config, str(pdf_path))
    cleanup()

    assert not pdf_path.exists()


def test_cleanup_requires_mineru_success(gui_config):
    uningested_dir = pathlib.Path(gui_config["uningested_dir"])
    uningested_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = uningested_dir / "paper-a.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    # No content_list.json for this stem anywhere under mineru_output_dir.
    cleanup = gui_jobs.make_drop_cleanup(gui_config, str(pdf_path))
    cleanup()

    assert pdf_path.exists()


def test_cleanup_containment(gui_config, tmp_path):
    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = outside_dir / "paper-a.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    # Even with a matching content_list.json, containment must refuse first.
    mineru_dir = pathlib.Path(gui_config["mineru_output_dir"]) / "paper-a"
    mineru_dir.mkdir(parents=True, exist_ok=True)
    (mineru_dir / "content_list.json").write_text("[]", encoding="utf-8")

    cleanup = gui_jobs.make_drop_cleanup(gui_config, str(pdf_path))
    cleanup()

    assert pdf_path.exists()


# ---------------------------------------------------------------------------
# Task 1, Test 6: POST /processing/ingest-all
# ---------------------------------------------------------------------------

def test_ingest_all_enqueues_one_job_per_pending_pdf_sorted(gui_client, gui_config, monkeypatch):
    _patch_processing_config(monkeypatch, gui_config)
    uningested_dir = pathlib.Path(gui_config["uningested_dir"])
    uningested_dir.mkdir(parents=True, exist_ok=True)
    (uningested_dir / "b-paper.pdf").write_bytes(b"%PDF-1.4")
    (uningested_dir / "a-paper.pdf").write_bytes(b"%PDF-1.4")

    captured = _capturing_enqueue(monkeypatch)

    resp = gui_client.post("/processing/ingest-all")

    assert resp.status_code == 200
    calls = captured["calls"]
    assert len(calls) == 2
    names = [pathlib.Path(c["argv"][c["argv"].index("--pdf") + 1]).name for c in calls]
    assert names == ["a-paper.pdf", "b-paper.pdf"]
    assert all(c["on_success"] is not None for c in calls)


# ---------------------------------------------------------------------------
# Task 1, Test 7: POST /processing/ingest-url
# ---------------------------------------------------------------------------

def test_ingest_url_enqueues_and_blank_url_errors(gui_client, gui_config, monkeypatch):
    _patch_processing_config(monkeypatch, gui_config)
    captured = _capturing_enqueue(monkeypatch)

    resp = gui_client.post(
        "/processing/ingest-url", data={"url": "https://arxiv.org/abs/1234.5678"}
    )
    assert resp.status_code == 200
    assert len(captured["calls"]) == 1
    argv = captured["calls"][0]["argv"]
    assert "--url" in argv
    assert "https://arxiv.org/abs/1234.5678" in argv

    resp2 = gui_client.post("/processing/ingest-url", data={"url": ""})
    assert resp2.status_code == 200
    assert len(captured["calls"]) == 1  # unchanged -- nothing new enqueued
    assert "url" in resp2.text.lower()


# ---------------------------------------------------------------------------
# No file-watcher daemon (INBOX-01 stays v2)
# ---------------------------------------------------------------------------

def test_no_watcher():
    gui_dir = _REPO_ROOT / "gui"
    for py_file in gui_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert "watchdog" not in text.lower(), f"{py_file} references watchdog"
        assert "Observer(" not in text, f"{py_file} instantiates an Observer"

    processing_source = (gui_dir / "routes" / "processing.py").read_text(encoding="utf-8")
    assert "scan_uningested" in processing_source


# ---------------------------------------------------------------------------
# Task 2: Per-paper orchestration
# ---------------------------------------------------------------------------

def test_orchestrate_enqueues_in_chosen_order(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_processing_config(monkeypatch, gui_config)
    _write_registry_entry(gui_config, "10.1234/known", "A Known Paper", "known-stem")

    captured = _capturing_enqueue(monkeypatch)

    resp = gui_client.post(
        "/processing/orchestrate",
        data={
            "registry_key": "10.1234/known",
            "biblio_include": "true", "biblio_order": "2",
            "link_include": "true", "link_order": "1",
        },
    )

    assert resp.status_code == 200
    calls = captured["calls"]
    assert len(calls) == 2
    assert calls[0]["action"] == "link_paper"
    assert calls[1]["action"] == "biblio"
    for call in calls:
        assert "--stem" in call["argv"]
        assert "known-stem" in call["argv"]


def test_orchestrate_unknown_registry_key_404(gui_client, gui_config, monkeypatch):
    _patch_processing_config(monkeypatch, gui_config)
    captured = _capturing_enqueue(monkeypatch)

    resp = gui_client.post(
        "/processing/orchestrate",
        data={"registry_key": "does-not-exist", "note_include": "true", "note_order": "1"},
    )

    assert resp.status_code == 404
    assert captured["calls"] == []


def test_orchestrate_empty_selection_enqueues_nothing(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_processing_config(monkeypatch, gui_config)
    _write_registry_entry(gui_config, "10.1234/known", "A Known Paper", "known-stem")

    captured = _capturing_enqueue(monkeypatch)

    resp = gui_client.post("/processing/orchestrate", data={"registry_key": "10.1234/known"})

    assert resp.status_code == 200
    assert captured["calls"] == []
    assert "select" in resp.text.lower()


def test_orchestrate_embed_argv_contains_paperjson_cache_path(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_processing_config(monkeypatch, gui_config)
    _write_registry_entry(gui_config, "10.1234/known", "A Known Paper", "known-stem")

    captured = _capturing_enqueue(monkeypatch)

    resp = gui_client.post(
        "/processing/orchestrate",
        data={"registry_key": "10.1234/known", "embed_include": "true", "embed_order": "1"},
    )

    assert resp.status_code == 200
    argv = captured["calls"][0]["argv"]
    expected_path = str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "known-stem.json")
    assert argv[-1] == expected_path


# ---------------------------------------------------------------------------
# 08-19 Gap 3 (UAT gap 3 / GUI-03/GUI-07): note force / force-analysis flags
# reachable through POST /processing/orchestrate.
# ---------------------------------------------------------------------------

def test_orchestrate_note_force_flag_reaches_argv(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_processing_config(monkeypatch, gui_config)
    _write_registry_entry(gui_config, "10.1234/known", "A Known Paper", "known-stem")

    captured = _capturing_enqueue(monkeypatch)

    resp = gui_client.post(
        "/processing/orchestrate",
        data={
            "registry_key": "10.1234/known",
            "note_include": "true", "note_order": "1",
            "note_force": "true",
        },
    )

    assert resp.status_code == 200
    argv = captured["calls"][0]["argv"]
    assert "--force" in argv
    assert "--force-analysis" not in argv


def test_orchestrate_note_force_analysis_flag_reaches_argv(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_processing_config(monkeypatch, gui_config)
    _write_registry_entry(gui_config, "10.1234/known", "A Known Paper", "known-stem")

    captured = _capturing_enqueue(monkeypatch)

    resp = gui_client.post(
        "/processing/orchestrate",
        data={
            "registry_key": "10.1234/known",
            "note_include": "true", "note_order": "1",
            "note_force_analysis": "true",
        },
    )

    assert resp.status_code == 200
    argv = captured["calls"][0]["argv"]
    assert "--force-analysis" in argv
    assert argv.count("--force") == 0


def test_orchestrate_note_both_flags_reach_argv(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_processing_config(monkeypatch, gui_config)
    _write_registry_entry(gui_config, "10.1234/known", "A Known Paper", "known-stem")

    captured = _capturing_enqueue(monkeypatch)

    resp = gui_client.post(
        "/processing/orchestrate",
        data={
            "registry_key": "10.1234/known",
            "note_include": "true", "note_order": "1",
            "note_force": "true", "note_force_analysis": "true",
        },
    )

    assert resp.status_code == 200
    argv = captured["calls"][0]["argv"]
    assert "--force" in argv
    assert "--force-analysis" in argv


def test_orchestrate_note_flags_absent_matches_todays_argv(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_processing_config(monkeypatch, gui_config)
    _write_registry_entry(gui_config, "10.1234/known", "A Known Paper", "known-stem")

    captured = _capturing_enqueue(monkeypatch)

    resp = gui_client.post(
        "/processing/orchestrate",
        data={
            "registry_key": "10.1234/known",
            "note_include": "true", "note_order": "1",
        },
    )

    assert resp.status_code == 200
    argv = captured["calls"][0]["argv"]
    assert "--force" not in argv
    assert "--force-analysis" not in argv


def test_orchestrate_note_flags_ignored_when_note_not_selected(gui_client, gui_config, monkeypatch):
    """A crafted POST sending note flags while note is unselected must have
    no effect (T-08-GAP-49) -- only the biblio job is enqueued, and nothing
    about it carries a note-only flag."""
    gui_config["project_name"] = "proj-a"
    _patch_processing_config(monkeypatch, gui_config)
    _write_registry_entry(gui_config, "10.1234/known", "A Known Paper", "known-stem")

    captured = _capturing_enqueue(monkeypatch)

    resp = gui_client.post(
        "/processing/orchestrate",
        data={
            "registry_key": "10.1234/known",
            "biblio_include": "true", "biblio_order": "1",
            "note_force": "true", "note_force_analysis": "true",
        },
    )

    assert resp.status_code == 200
    calls = captured["calls"]
    assert len(calls) == 1
    assert calls[0]["action"] == "biblio"
    assert "--force" not in calls[0]["argv"]
    assert "--force-analysis" not in calls[0]["argv"]


# ---------------------------------------------------------------------------
# 08-19 Task 2: flag tickboxes rendered in the orchestration table, unticked
# ---------------------------------------------------------------------------

def test_processing_page_renders_note_flag_checkboxes_unticked(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_processing_config(monkeypatch, gui_config)
    _write_registry_entry(gui_config, "10.1234/known", "A Known Paper", "known-stem")

    resp = gui_client.get("/processing")

    assert resp.status_code == 200
    assert 'name="note_force"' in resp.text
    assert 'name="note_force_analysis"' in resp.text
    # Neither checkbox carries `checked` -- unlike the include checkboxes,
    # which are checked by default (processing.html:80/91/102/113).
    import re
    for field in ("note_force", "note_force_analysis"):
        match = re.search(rf'<input[^>]*name="{field}"[^>]*>', resp.text)
        assert match is not None, f"{field} checkbox not found"
        assert "checked" not in match.group(0), f"{field} must default unticked"


def test_processing_page_link_row_and_overview_topics_column_use_connected_wording(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_processing_config(monkeypatch, gui_config)

    resp = gui_client.get("/processing")
    assert resp.status_code == 200
    assert "Link — Link by Topics" in resp.text


def test_processing_page_has_paper_select_and_script_rows(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_processing_config(monkeypatch, gui_config)
    _write_registry_entry(gui_config, "10.1234/known", "A Known Paper", "known-stem")

    resp = gui_client.get("/processing")

    assert resp.status_code == 200
    assert 'name="registry_key"' in resp.text
    assert "A Known Paper" in resp.text
    for field in ("note_include", "biblio_include", "embed_include", "link_include"):
        assert field in resp.text
    for field in ("note_order", "biblio_order", "embed_order", "link_order"):
        assert field in resp.text


# ---------------------------------------------------------------------------
# 08-12 Task 1 (Gap A, GUI-04/D-07/D-08): scroll-stable ~2s log tail
#
# The tail-follow behaviour itself (stick to bottom / hold position while
# reading) is browser-only DOM/scroll behaviour and cannot be driven from a
# server-rendered TestClient response without a headless browser (none is in
# this project's stack). What CAN regress silently -- and is pinned here --
# is the markup contract: the ~2s polling attributes and the self-terminating
# gate must survive byte-for-byte, and the tail element/script wiring that
# the scroll-restore logic depends on must be present and correctly keyed
# per job id.
# ---------------------------------------------------------------------------

def test_running_job_status_keeps_polling_and_carries_scroll_restore_wiring(gui_client):
    job_id = "20260101-000000-note-abc"
    gui_jobs.JOBS[job_id] = gui_jobs.JobState(
        id=job_id, action="note", argv=["python", "scripts/note.py"], status="running",
        created_at="2026-01-01T00:00:00", tail=["line one", "line two"],
    )

    resp = gui_client.get(f"/jobs/{job_id}/status")

    assert resp.status_code == 200
    # Preserved polling contract (D-07/D-08 self-terminating poll).
    assert f'hx-get="/jobs/{job_id}/status"' in resp.text
    assert 'hx-trigger="every 2s"' in resp.text
    assert 'hx-swap="outerHTML"' in resp.text
    # Per-job tail id + scroll-restore wiring referencing that same id.
    tail_id = f"job-tail-{job_id}"
    assert f'id="{tail_id}"' in resp.text
    assert f'"{tail_id}"' in resp.text  # referenced inside the inline script
    assert "getElementById" in resp.text
    assert "scrollTop" in resp.text


def test_terminal_job_status_omits_polling_attributes(gui_client):
    job_id = "20260101-000000-note-xyz"
    gui_jobs.JOBS[job_id] = gui_jobs.JobState(
        id=job_id, action="note", argv=["python", "scripts/note.py"], status="done",
        created_at="2026-01-01T00:00:00", tail=["done."],
    )

    resp = gui_client.get(f"/jobs/{job_id}/status")

    assert resp.status_code == 200
    assert "hx-get" not in resp.text
    assert "hx-trigger" not in resp.text
    # The tail id + scroll-restore wiring are unconditional (not gated on
    # queued/running), so a terminal render still carries them -- only the
    # polling attributes are gated.
    assert f'id="job-tail-{job_id}"' in resp.text
