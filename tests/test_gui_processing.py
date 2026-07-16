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
