"""Tests for gui/jobs.py (Phase 8 Plan 03): the FIFO subprocess job engine
(D-05/D-06/D-07/D-08), its argv builder, and the /jobs/* + /processing routes
that expose it.

Engine tests spawn harmless ``[sys.executable, "-c", ...]`` subprocesses
against a tmp-scoped jobs dir -- never the real ``.local/gui_jobs/``. An
autouse fixture reassigns every piece of ``gui.jobs``' module-level engine
state (JOBS dict, work queue, running-process map, cancelled-queued set,
on-success callbacks, worker-thread handle) before each test so tests never
leak jobs or worker threads across test functions (``gui.jobs._worker_loop``
captures its queue object at thread-start time specifically so a stale
worker thread from a prior test can never steal jobs off a freshly
monkeypatched queue).

Run with:  python -m pytest tests/test_gui_jobs.py -x
"""

import json
import pathlib
import queue as queue_module
import sys
import time

import pytest

import gui.jobs as jobs


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_jobs_state(tmp_path, monkeypatch):
    """Fresh, tmp-scoped jobs engine for every test."""
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "gui_jobs")
    monkeypatch.setattr(jobs, "JOBS", {})
    monkeypatch.setattr(jobs, "_QUEUE", queue_module.Queue())
    monkeypatch.setattr(jobs, "_PROC_BY_JOB", {})
    monkeypatch.setattr(jobs, "_CANCELLED_QUEUED", set())
    monkeypatch.setattr(jobs, "_ON_SUCCESS", {})
    monkeypatch.setattr(jobs, "_WORKER_THREAD", None)
    yield


def _wait_until(predicate, timeout=5.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _wait_terminal(job_id, timeout=5.0):
    ok = _wait_until(
        lambda: jobs.get(job_id).status in ("done", "error", "cancelled"),
        timeout=timeout,
    )
    job = jobs.get(job_id)
    assert ok, f"job {job_id} never reached a terminal state (status={job.status})"
    return job


def _patch_jobs_config(monkeypatch, gui_config):
    import gui.routes.logs as logs_routes

    monkeypatch.setattr(logs_routes, "load_gui_config", lambda: gui_config)


# ---------------------------------------------------------------------------
# Task 1, Test 1: build_action_argv
# ---------------------------------------------------------------------------

def test_build_action_argv_all_actions_return_list_starting_with_python():
    config = {"paperjson_cache_dir": "/cache"}
    cases = [
        ("ingest_pdf", {"path": "/x/paper.pdf"}, "ingest.py"),
        ("ingest_url", {"url": "https://example.org/paper"}, "ingest.py"),
        ("ingest_doi", {"doi": "10.1/xyz"}, "ingest.py"),
        ("note", {"stem": "my-stem"}, "note.py"),
        ("biblio", {"stem": "my-stem"}, "biblio.py"),
        ("embed", {"stem": "my-stem"}, "embed.py"),
        ("link_paper", {"stem": "my-stem"}, "link.py"),
        ("embed_all", {}, "embed.py"),
        ("link_all", {}, "link.py"),
        ("setup", {}, "setup.py"),
    ]
    for action, params, script_name in cases:
        argv = jobs.build_action_argv(action, config, **params)
        assert isinstance(argv, list), f"{action}: argv must be a list"
        assert argv[0] == sys.executable, f"{action}: argv[0] must be sys.executable"
        assert argv[1].endswith(script_name), (
            f"{action}: argv[1] must target {script_name}, got {argv[1]}"
        )
        assert all(isinstance(a, str) for a in argv), f"{action}: every argv element must be a str"


def test_build_action_argv_ingest_pdf_path_is_standalone_element():
    argv = jobs.build_action_argv("ingest_pdf", {}, path="/x/paper.pdf")
    assert "--pdf" in argv
    assert "/x/paper.pdf" in argv
    assert argv.index("/x/paper.pdf") == argv.index("--pdf") + 1


def test_build_action_argv_embed_all_and_link_all_target_correct_scripts():
    embed_argv = jobs.build_action_argv("embed_all", {})
    assert embed_argv[0] == sys.executable
    assert embed_argv[1].endswith("embed.py")
    assert embed_argv[-1] == "--all"

    link_argv = jobs.build_action_argv("link_all", {})
    assert link_argv[0] == sys.executable
    assert link_argv[1].endswith("link.py")
    assert link_argv[-1] == "--all"


def test_build_action_argv_embed_resolves_stem_to_cache_path():
    config = {"paperjson_cache_dir": "/cache/dir"}
    argv = jobs.build_action_argv("embed", config, stem="my-paper")
    assert argv[-1] == str(pathlib.Path("/cache/dir") / "my-paper.json")


def test_build_action_argv_unknown_action_raises():
    with pytest.raises(ValueError):
        jobs.build_action_argv("nope", {})


# ---------------------------------------------------------------------------
# 08-19 Gap 3 (UAT gap 3 / GUI-03/GUI-07): note branch force/force-analysis
# flags. build_action_argv is the single argv-building chokepoint (T-08-02);
# these tests pin its note-branch flag shape so a future edit here can't
# quietly reintroduce always-skip or always-force behavior.
# ---------------------------------------------------------------------------

def test_build_action_argv_note_force_flag_appends_force():
    argv = jobs.build_action_argv("note", {}, stem="my-stem", force=True)
    assert "--force" in argv


def test_build_action_argv_note_force_analysis_flag_appends_force_analysis():
    argv = jobs.build_action_argv("note", {}, stem="my-stem", force_analysis=True)
    assert "--force-analysis" in argv
    assert "--force" not in argv


def test_build_action_argv_note_both_flags_appends_both():
    argv = jobs.build_action_argv("note", {}, stem="my-stem", force=True, force_analysis=True)
    assert "--force" in argv
    assert "--force-analysis" in argv


def test_build_action_argv_note_no_flags_matches_todays_argv_exactly():
    argv = jobs.build_action_argv("note", {}, stem="my-stem")
    assert "--force" not in argv
    assert "--force-analysis" not in argv
    assert argv[0] == sys.executable
    assert argv[1].endswith("note.py")
    assert argv[2:] == ["--stem", "my-stem"]


# ---------------------------------------------------------------------------
# Task 1, Test 2/6: FIFO serial execution + queue position + is_busy()
# ---------------------------------------------------------------------------

def test_fifo_serial_execution_and_queue_position():
    job1 = jobs.enqueue("noop", [sys.executable, "-c", "import time; time.sleep(0.3)"])
    job2 = jobs.enqueue("noop", [sys.executable, "-c", "pass"])

    assert _wait_until(lambda: jobs.get(job1).status == "running", timeout=5)
    assert jobs.get(job2).status == "queued"
    assert jobs.get(job2).queue_position == 1

    job1_final = _wait_terminal(job1)
    job2_final = _wait_terminal(job2)

    assert job1_final.finished_at is not None
    assert job2_final.started_at is not None
    assert job2_final.started_at >= job1_final.finished_at, (
        "job2 must not start before job1 finishes (strict serial FIFO, D-06)"
    )


def test_is_busy_flips_true_then_false():
    assert jobs.is_busy() is False
    job_id = jobs.enqueue("noop", [sys.executable, "-c", "pass"])
    assert jobs.is_busy() is True
    _wait_terminal(job_id)
    assert jobs.is_busy() is False


# ---------------------------------------------------------------------------
# Task 1, Test 3: log file + JSON index (atomic write)
# ---------------------------------------------------------------------------

def test_job_writes_log_file_and_index():
    job_id = jobs.enqueue("noop", [sys.executable, "-c", "print('hello from job')"])
    job = _wait_terminal(job_id)

    log_path = pathlib.Path(job.log_path)
    assert log_path.exists()
    assert "hello from job" in log_path.read_text(encoding="utf-8")

    index_path = jobs.JOBS_DIR / "index.json"
    assert index_path.exists()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert job_id in index
    entry = index[job_id]
    assert entry["action"] == "noop"
    assert entry["status"] == "done"
    assert entry["returncode"] == 0
    assert entry["created_at"] and entry["started_at"] and entry["finished_at"]


def test_index_write_uses_temp_file_and_replace(monkeypatch):
    calls = []
    real_replace = jobs.os.replace

    def _spy_replace(src, dst):
        calls.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(jobs.os, "replace", _spy_replace)

    job_id = jobs.enqueue("noop", [sys.executable, "-c", "pass"])
    _wait_terminal(job_id)

    assert calls, "expected index.json to be written via os.replace at least once"
    for src, dst in calls:
        assert src.endswith(".tmp"), f"expected a temp-file source, got {src}"
        assert dst.endswith("index.json"), f"expected index.json as the replace target, got {dst}"


def test_list_jobs_survives_simulated_restart(monkeypatch):
    """Job history persists to index.json; a fresh in-memory JOBS dict (as if
    the server restarted) still finds it via list_jobs()/get()."""
    job_id = jobs.enqueue("noop", [sys.executable, "-c", "pass"])
    _wait_terminal(job_id)

    monkeypatch.setattr(jobs, "JOBS", {})  # simulate a fresh process

    ids = [job.id for job in jobs.list_jobs()]
    assert job_id in ids
    assert jobs.get(job_id) is not None
    assert jobs.get(job_id).status == "done"


# ---------------------------------------------------------------------------
# Task 1, Test 4: status transitions + tail cap
# ---------------------------------------------------------------------------

def test_status_transitions_done_and_error():
    ok_job = _wait_terminal(
        jobs.enqueue("noop", [sys.executable, "-c", "import sys; sys.exit(0)"])
    )
    assert ok_job.status == "done"
    assert ok_job.returncode == 0

    err_job = _wait_terminal(
        jobs.enqueue("noop", [sys.executable, "-c", "import sys; sys.exit(3)"])
    )
    assert err_job.status == "error"
    assert err_job.returncode == 3


def test_tail_holds_at_most_last_50_lines():
    job_id = jobs.enqueue("noop", [sys.executable, "-c", "for i in range(80): print(i)"])
    job = _wait_terminal(job_id)
    assert len(job.tail) <= 50
    assert job.tail[-1] == "79"


# ---------------------------------------------------------------------------
# Task 1, Test 5: cancel
# ---------------------------------------------------------------------------

def test_cancel_running_job_kills_process():
    job_id = jobs.enqueue("noop", [sys.executable, "-c", "import time; time.sleep(30)"])
    assert _wait_until(lambda: jobs.get(job_id).status == "running", timeout=5)

    assert jobs.cancel(job_id) is True

    job = _wait_terminal(job_id)
    assert job.status == "cancelled"


def test_cancel_queued_job_removes_it_from_the_queue():
    blocker_id = jobs.enqueue("noop", [sys.executable, "-c", "import time; time.sleep(0.4)"])
    queued_id = jobs.enqueue("noop", [sys.executable, "-c", "pass"])

    assert jobs.get(queued_id).status == "queued"
    assert jobs.cancel(queued_id) is True
    assert jobs.get(queued_id).status == "cancelled"

    _wait_terminal(blocker_id)
    time.sleep(0.2)  # give the worker a chance to dequeue-and-skip
    assert jobs.get(queued_id).status == "cancelled", "a cancelled queued job must never run"


# ---------------------------------------------------------------------------
# 08-08 Gap 1 (D-06 / GUI-04): worker survives non-UTF-8 child output and any
# _run_job exception -- the sole worker thread must never die mid-session.
# ---------------------------------------------------------------------------

def test_worker_survives_non_utf8_child_output():
    # Writes a lone invalid-UTF-8 byte (cp1252 'e-acute') directly to the raw
    # stdout buffer, bypassing any child-side text encoding, so it lands on
    # the merged pipe exactly as a real accented-author-name subprocess would.
    non_utf8_snippet = (
        "import sys; "
        "sys.stdout.buffer.write(b'caf\\xe9\\n'); "
        "sys.stdout.buffer.flush(); "
        "print('sentinel-ok')"
    )
    job1 = jobs.enqueue("noop", [sys.executable, "-c", non_utf8_snippet])
    job1_final = _wait_terminal(job1)
    assert job1_final.status == "done"
    assert job1_final.returncode == 0
    assert pathlib.Path(job1_final.log_path).exists()

    # The single worker thread must still be alive to serve the next job.
    job2 = jobs.enqueue("noop", [sys.executable, "-c", "print('second')"])
    job2_final = _wait_terminal(job2)
    assert job2_final.status == "done"


def test_worker_survives_run_job_exception(monkeypatch):
    real_run_job = jobs._run_job

    def _boom_or_real(job):
        if job.action == "boom":
            raise RuntimeError("boom")
        return real_run_job(job)

    monkeypatch.setattr(jobs, "_run_job", _boom_or_real)

    boom_id = jobs.enqueue("boom", [sys.executable, "-c", "pass"])
    noop_id = jobs.enqueue("noop", [sys.executable, "-c", "print('noop-ok')"])

    boom_final = _wait_terminal(boom_id)
    assert boom_final.status == "error"
    assert any("worker exception" in line for line in boom_final.tail)

    noop_final = _wait_terminal(noop_id)
    assert noop_final.status == "done"


# ---------------------------------------------------------------------------
# T-08-02: no shell spawn, ever
# ---------------------------------------------------------------------------

def test_enqueue_builds_argv_list():
    argv = jobs.build_action_argv("ingest_pdf", {}, path="/x/paper.pdf")
    assert isinstance(argv, list)
    assert argv[0] == sys.executable


def test_no_shell_spawn(monkeypatch):
    captured = {}
    real_popen = jobs.subprocess.Popen

    def _spy_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(jobs.subprocess, "Popen", _spy_popen)

    job_id = jobs.enqueue("noop", [sys.executable, "-c", "pass"])
    _wait_terminal(job_id)

    assert isinstance(captured["argv"], list)
    assert "shell" not in captured["kwargs"] or captured["kwargs"]["shell"] is False


# ---------------------------------------------------------------------------
# Task 2: /jobs/* routes + Logs & Activity page
# ---------------------------------------------------------------------------

def test_logs_page_empty_state(gui_client):
    resp = gui_client.get("/logs")
    assert resp.status_code == 200
    assert "No jobs yet" in resp.text


def test_enqueue_route_returns_polling_partial(gui_client, gui_config, monkeypatch):
    _patch_jobs_config(monkeypatch, gui_config)
    # Never let the route's real build_action_argv spawn the real, long-running
    # scripts/link.py --all -- swap in a harmless sleeping command instead.
    monkeypatch.setattr(
        jobs, "build_action_argv",
        lambda action, config, **params: [sys.executable, "-c", "import time; time.sleep(1)"],
    )

    resp = gui_client.post("/jobs/enqueue", data={"action": "link_all"})

    assert resp.status_code == 200
    assert 'hx-trigger="every 2s"' in resp.text


def test_job_status_route_finished_job_has_no_polling_trigger(gui_client, gui_config, monkeypatch):
    _patch_jobs_config(monkeypatch, gui_config)
    job_id = jobs.enqueue("noop", [sys.executable, "-c", "pass"])
    _wait_terminal(job_id)

    resp = gui_client.get(f"/jobs/{job_id}/status")

    assert resp.status_code == 200
    assert 'hx-trigger="every 2s"' not in resp.text


def test_job_status_route_path_separator_id_returns_404(gui_client):
    resp = gui_client.get("/jobs/..%2F..%2Fetc/status")
    assert resp.status_code == 404


def test_cancel_route_flips_queued_job_to_cancelled(gui_client, gui_config, monkeypatch):
    _patch_jobs_config(monkeypatch, gui_config)
    blocker_id = jobs.enqueue("noop", [sys.executable, "-c", "import time; time.sleep(0.4)"])
    queued_id = jobs.enqueue("noop", [sys.executable, "-c", "pass"])
    assert jobs.get(queued_id).status == "queued"

    resp = gui_client.post(f"/jobs/{queued_id}/cancel")

    assert resp.status_code == 200
    assert jobs.get(queued_id).status == "cancelled"
    _wait_terminal(blocker_id)


def test_job_log_route_renders_full_log(gui_client, gui_config, monkeypatch):
    _patch_jobs_config(monkeypatch, gui_config)
    job_id = jobs.enqueue("noop", [sys.executable, "-c", "print('full log line')"])
    _wait_terminal(job_id)

    resp = gui_client.get(f"/jobs/{job_id}/log")

    assert resp.status_code == 200
    assert "full log line" in resp.text


# ---------------------------------------------------------------------------
# Task 3: Processing page bulk actions
# ---------------------------------------------------------------------------

def test_processing_page_has_bulk_action_ctas(gui_client):
    resp = gui_client.get("/processing")
    assert resp.status_code == 200
    assert "Backfill Embeddings" in resp.text
    assert "Rebuild Topic Graph" in resp.text


def test_embed_all_and_link_all_argv_target_correct_scripts(gui_client, gui_config, monkeypatch):
    """Capture the real build_action_argv output via a faked enqueue() so the
    real backfill/link subprocess is never spawned (route-level assertion)."""
    import gui.routes.logs as logs_routes

    _patch_jobs_config(monkeypatch, gui_config)

    captured = {}

    def _fake_enqueue(action, argv, on_success=None):
        captured[action] = argv
        job_id = f"fake-{action}"
        jobs.JOBS[job_id] = jobs.JobState(
            id=job_id, action=action, argv=argv, status="queued",
            created_at="2026-01-01T00:00:00",
        )
        return job_id

    monkeypatch.setattr(logs_routes.gui_jobs, "enqueue", _fake_enqueue)

    gui_client.post("/jobs/enqueue", data={"action": "embed_all"})
    gui_client.post("/jobs/enqueue", data={"action": "link_all"})

    assert captured["embed_all"][0] == sys.executable
    assert captured["embed_all"][1].endswith("embed.py")
    assert captured["embed_all"][-1] == "--all"

    assert captured["link_all"][0] == sys.executable
    assert captured["link_all"][1].endswith("link.py")
    assert captured["link_all"][-1] == "--all"
