"""gui/jobs.py -- Subprocess-per-job FIFO engine (Phase 8 Plan 03, D-05/D-06/D-07/D-08).

Every pipeline action the GUI can trigger runs as a real subprocess -- never
an in-process import of a ``run_*``/``ingest*`` core function (D-05; those
modules carry mutable module-level globals that would leak across
"concurrent" requests in a long-lived FastAPI process, and a crashing LLM
call must never take the GUI server down). Exactly one background worker
thread drains a FIFO queue, so at most one pipeline subprocess ever runs at a
time (D-06 -- VRAM + registry-filelock contention, Phase 5 D-11). Progress is
observed by polling (D-07); state persists to per-job log files plus a JSON
index under ``.local/gui_jobs/`` so job history survives a server restart
(D-08).

Public API:
  build_action_argv(action, config, **params) -> list[str]
      Builds a subprocess argv list ([sys.executable, script_path, ...]) for
      one of the fixed set of known pipeline actions. Never a shell string
      (T-08-02). Raises ValueError for an unknown action.
  enqueue(action, argv, on_success=None) -> job_id
      Queues a job for the single serial worker; returns its id immediately.
  get(job_id) -> JobState | None
      In-memory lookup, falling back to the on-disk index (server-restart
      history).
  list_jobs() -> list[JobState]
      In-memory jobs merged with the on-disk index, newest first.
  cancel(job_id) -> bool
      Kills a running job's process, or marks a still-queued job cancelled
      so the worker skips it when dequeued.
  is_busy() -> bool
      True while any job is queued or running (08-07's chat-blocked state,
      D-16, reads this).
"""

import dataclasses
import datetime
import itertools
import json
import os
import pathlib
import queue
import re
import subprocess
import sys
import threading
import uuid

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

# Jobs directory -- monkeypatched by tests to a tmp_path subdirectory so test
# runs never write into the real .local/gui_jobs/.
JOBS_DIR = _REPO_ROOT / ".local" / "gui_jobs"

_TAIL_MAX_LINES = 50
_TERMINAL_STATUSES = ("done", "error", "cancelled")

_SEQ_COUNTER = itertools.count()


@dataclasses.dataclass
class JobState:
    id: str
    action: str
    argv: list
    status: str = "queued"
    returncode: int | None = None
    tail: list = dataclasses.field(default_factory=list)
    log_path: str = ""
    queue_position: int | None = None
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    pid: int | None = None
    seq: int = 0


# Module-level engine state. Every one of these is reassigned wholesale by
# tests/test_gui_jobs.py's autouse `_reset_jobs_state` fixture (monkeypatch
# .setattr on this module) so each test gets a fresh, isolated engine -- never
# a worker thread/queue/job dict leaked from a prior test.
JOBS: dict = {}
_QUEUE: "queue.Queue" = queue.Queue()
_PROC_BY_JOB: dict = {}
_CANCELLED_QUEUED: set = set()
_ON_SUCCESS: dict = {}
_WORKER_THREAD = None
_WORKER_LOCK = threading.Lock()
_INDEX_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Jobs dir / on-disk index (D-08) -- atomic temp-file + os.replace, matching
# scripts/obsidian_cli.py::create_note's write shape.
# ---------------------------------------------------------------------------

def _jobs_dir() -> pathlib.Path:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    return JOBS_DIR


def _index_path() -> pathlib.Path:
    return _jobs_dir() / "index.json"


def _job_to_dict(job: "JobState") -> dict:
    return dataclasses.asdict(job)


def _write_index() -> None:
    """Atomic temp-file + os.replace write. Guarded by ``_INDEX_LOCK``: both
    the enqueueing thread (a request handler, adding a new queued job) and
    the single worker thread (transitioning a job through
    queued->running->done/error) call this, and Windows' stricter file
    locking (unlike POSIX) raises ``PermissionError`` if two threads'
    temp-file writes/replaces to the same path ever interleave."""
    with _INDEX_LOCK:
        data = {jid: _job_to_dict(job) for jid, job in JOBS.items()}
        index_path = _index_path()
        tmp_path = index_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, index_path)


def _load_index() -> dict:
    index_path = _index_path()
    if not index_path.exists():
        return {}
    try:
        with open(index_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _new_job_id(action: str) -> str:
    """``YYYYMMDD-HHMMSS-<action>-<3-char suffix>``. The action segment is
    slugified to the same letters/digits/hyphens charset as the timestamp and
    suffix (some action names contain underscores, e.g. ``embed_all``) so the
    whole id can be validated with one strict pattern before any log-path
    construction (T-08-03)."""
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    action_slug = re.sub(r"[^A-Za-z0-9]+", "-", action).strip("-") or "job"
    suffix = uuid.uuid4().hex[:3]
    return f"{ts}-{action_slug}-{suffix}"


# ---------------------------------------------------------------------------
# Argv builder (T-08-02) -- one branch per parity-map action, argv as a
# Python list always, never a shell string.
# ---------------------------------------------------------------------------

def build_action_argv(action: str, config: dict, **params) -> list:
    """Build a subprocess argv list for a known pipeline action.

    Always ``[sys.executable, str(REPO_ROOT/scripts/<script>.py), ...]``.
    Raises ``ValueError`` for an unrecognized action -- callers (the
    ``/jobs/enqueue`` route) turn that into a 400, never a 500.
    """
    py = sys.executable

    def script(name: str) -> str:
        return str(_SCRIPTS_DIR / name)

    if action == "ingest_pdf":
        argv = [py, script("ingest.py"), "--pdf", params["path"]]
        if params.get("force_extract"):
            argv.append("--force-extract")
        if params.get("refill"):
            argv.append("--refill")
        return argv

    if action == "ingest_url":
        return [py, script("ingest.py"), "--url", params["url"]]

    if action == "ingest_doi":
        return [py, script("ingest.py"), "--doi", params["doi"]]

    if action == "note":
        argv = [py, script("note.py")]
        if params.get("paperjson"):
            argv += ["--paperjson", params["paperjson"]]
        else:
            argv += ["--stem", params["stem"]]
        if params.get("force"):
            argv.append("--force")
        return argv

    if action == "biblio":
        argv = [py, script("biblio.py")]
        if params.get("paperjson"):
            argv += ["--paperjson", params["paperjson"]]
        else:
            argv += ["--stem", params["stem"]]
        return argv

    if action == "embed":
        if params.get("paperjson"):
            pj_path = params["paperjson"]
        else:
            cache_dir = config.get("paperjson_cache_dir") or ".paperjson_cache"
            pj_path = str(pathlib.Path(cache_dir) / f"{params['stem']}.json")
        return [py, script("embed.py"), pj_path]

    if action == "link_paper":
        argv = [py, script("link.py")]
        if params.get("paperjson"):
            argv += ["--paperjson", params["paperjson"]]
        else:
            argv += ["--stem", params["stem"]]
        return argv

    if action == "embed_all":
        return [py, script("embed.py"), "--all"]

    if action == "link_all":
        return [py, script("link.py"), "--all"]

    if action == "setup":
        argv = [py, script("setup.py"), "--non-interactive"]
        if params.get("vault_path"):
            argv += ["--vault-path", params["vault_path"]]
        if params.get("registry_path"):
            argv += ["--registry-path", params["registry_path"]]
        if params.get("project_name"):
            argv += ["--project-name", params["project_name"]]
        return argv

    raise ValueError(f"unknown action: {action!r}")


# ---------------------------------------------------------------------------
# Queue position bookkeeping
# ---------------------------------------------------------------------------

def _update_queue_positions() -> None:
    queued = sorted(
        (job for job in JOBS.values() if job.status == "queued"),
        key=lambda job: job.seq,
    )
    for pos, job in enumerate(queued, start=1):
        job.queue_position = pos


# ---------------------------------------------------------------------------
# Worker thread (D-05/D-06/D-08) -- Pattern 1 from 08-RESEARCH.md, verbatim
# shape: merge stdout+stderr is safe here because every scripts/*.py prints
# exactly one final stdout line on success (verified per-script at plan time).
# ---------------------------------------------------------------------------

def _run_job(job: "JobState") -> None:
    job.status = "running"
    job.queue_position = None
    job.started_at = _now()
    _update_queue_positions()
    _write_index()

    log_path = pathlib.Path(job.log_path)

    try:
        proc = subprocess.Popen(
            job.argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
    except Exception as e:  # noqa: BLE001 - a failed spawn is a job error, not a server crash
        job.status = "error"
        job.returncode = -1
        job.tail.append(f"[jobs error: failed to start process: {e}]")
        job.finished_at = _now()
        _write_index()
        return

    job.pid = proc.pid
    _PROC_BY_JOB[job.id] = proc

    try:
        with open(log_path, "a", encoding="utf-8") as logf:
            for line in iter(proc.stdout.readline, ""):
                logf.write(line)
                logf.flush()
                job.tail.append(line.rstrip("\n"))
                job.tail = job.tail[-_TAIL_MAX_LINES:]
            proc.wait()
    finally:
        _PROC_BY_JOB.pop(job.id, None)

    if job.status == "cancelled":
        # cancel() already set status/finished_at; only fill returncode if
        # cancel() didn't already know it (queued-cancel path never runs).
        if job.returncode is None:
            job.returncode = proc.returncode
    else:
        job.returncode = proc.returncode
        job.status = "done" if proc.returncode == 0 else "error"
        job.finished_at = _now()

    on_success = _ON_SUCCESS.pop(job.id, None)
    if job.status == "done" and on_success is not None:
        try:
            on_success()
        except Exception as e:  # noqa: BLE001 - never crash the worker
            job.tail.append(f"[jobs warning: on_success callback failed: {e}]")

    _write_index()


def _worker_loop(work_queue) -> None:
    """Loop bound to the ``work_queue`` object captured at thread-start time
    (not a live lookup of the module-level ``_QUEUE`` name) so a test that
    monkeypatches ``_QUEUE`` to a fresh ``queue.Queue()`` between tests can
    never have a stale, still-running worker thread from a prior test steal
    jobs off the new queue."""
    while True:
        job_id = work_queue.get()
        if job_id in _CANCELLED_QUEUED:
            _CANCELLED_QUEUED.discard(job_id)
            continue
        job = JOBS.get(job_id)
        if job is None:
            continue
        _run_job(job)


def _ensure_worker() -> None:
    global _WORKER_THREAD
    with _WORKER_LOCK:
        if _WORKER_THREAD is None or not _WORKER_THREAD.is_alive():
            _WORKER_THREAD = threading.Thread(
                target=_worker_loop, args=(_QUEUE,), daemon=True
            )
            _WORKER_THREAD.start()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enqueue(action: str, argv: list, on_success=None) -> str:
    """Queue a job for the single serial worker; returns its job_id
    immediately (the job may still be queued when this returns)."""
    job_id = _new_job_id(action)
    job = JobState(
        id=job_id,
        action=action,
        argv=argv,
        status="queued",
        created_at=_now(),
        log_path=str(_jobs_dir() / f"{job_id}.log"),
        seq=next(_SEQ_COUNTER),
    )
    JOBS[job_id] = job
    if on_success is not None:
        _ON_SUCCESS[job_id] = on_success
    _update_queue_positions()
    _write_index()
    _ensure_worker()
    _QUEUE.put(job_id)
    return job_id


def get(job_id: str):
    """In-memory lookup, falling back to the on-disk index (job history
    surviving a server restart, D-08)."""
    job = JOBS.get(job_id)
    if job is not None:
        return job
    data = _load_index().get(job_id)
    if data is None:
        return None
    return JobState(**data)


def list_jobs() -> list:
    """In-memory jobs merged with the on-disk index, newest first."""
    merged = _load_index()
    for jid, job in JOBS.items():
        merged[jid] = _job_to_dict(job)
    jobs = [JobState(**data) for data in merged.values()]
    jobs.sort(key=lambda job: (job.created_at, job.seq), reverse=True)
    return jobs


def cancel(job_id: str) -> bool:
    """Kill a running job's process, or mark a still-queued job cancelled so
    the worker skips it (never runs it) when it's dequeued."""
    job = JOBS.get(job_id)
    if job is None:
        return False

    if job.status == "queued":
        job.status = "cancelled"
        job.queue_position = None
        job.finished_at = _now()
        _CANCELLED_QUEUED.add(job_id)
        _update_queue_positions()
        _write_index()
        return True

    if job.status == "running":
        proc = _PROC_BY_JOB.get(job_id)
        if proc is None:
            return False
        job.status = "cancelled"
        job.finished_at = _now()
        proc.kill()
        _write_index()
        return True

    return False


def is_busy() -> bool:
    """True while any job is queued or running (08-07's chat-blocked state,
    D-16, reads this)."""
    return any(job.status in ("queued", "running") for job in JOBS.values())


# ---------------------------------------------------------------------------
# Drop-folder cleanup hook (D-10, T-08-14/T-08-15) -- Phase 8 Plan 05
# ---------------------------------------------------------------------------

def make_drop_cleanup(config: dict, pdf_path):
    """Build an ``on_success`` callable for a drop-folder PDF ingest job.

    Deletes ``pdf_path`` only when BOTH gates pass:
      (a) the resolved ``pdf_path`` is inside the resolved
          ``config["uningested_dir"]`` (containment gate -- T-08-14/T-08-15).
      (b) a ``content_list.json`` exists under
          ``mineru_output_dir/<pdf stem>/`` -- i.e. MinerU extraction
          succeeded for this PDF. This is the locked D-10 semantics:
          deletion is gated on *MinerU* success, not full-pipeline success
          (``mineru_output_dir`` already retains the original copy
          regardless). Reuses ``gui.scan._stage_mineru``'s exact
          recursive-glob signal (imported locally to keep this module's
          import surface unchanged at load time) so the write-side scan and
          this cleanup gate can never disagree about what "MinerU succeeded"
          means (Don't Hand-Roll).

    A refusal is logged to stderr and the callable simply returns -- never
    raises. ``_run_job`` already wraps ``on_success`` calls in a try/except
    so a bug here can't crash the worker thread, but these two checks are
    the actual security boundary. A missing PDF (already deleted, e.g. a
    re-run) is a silent no-op.
    """
    def _cleanup() -> None:
        from gui.scan import _stage_mineru  # local import: keep module load order unchanged

        uningested_dir = config.get("uningested_dir")
        if not uningested_dir:
            print(
                "[jobs warning: drop cleanup refused -- no uningested_dir configured]",
                file=sys.stderr,
            )
            return

        try:
            resolved_pdf = pathlib.Path(pdf_path).resolve()
            resolved_dir = pathlib.Path(uningested_dir).resolve()
        except OSError as e:
            print(
                f"[jobs warning: drop cleanup refused -- could not resolve paths: {e}]",
                file=sys.stderr,
            )
            return

        try:
            resolved_pdf.relative_to(resolved_dir)
        except ValueError:
            print(
                f"[jobs warning: drop cleanup refused -- {resolved_pdf} is outside {resolved_dir}]",
                file=sys.stderr,
            )
            return

        stem = resolved_pdf.stem
        if not _stage_mineru(config, stem):
            print(
                f"[jobs warning: drop cleanup skipped -- no content_list.json found for stem {stem!r}]",
                file=sys.stderr,
            )
            return

        try:
            resolved_pdf.unlink()
        except FileNotFoundError:
            pass  # already gone -- no-op

    return _cleanup
