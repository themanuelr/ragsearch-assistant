"""Processing page (Phase 8 Plans 03 + 05).

Plan 03 built the Bulk Actions section (Backfill Embeddings / Rebuild Topic
Graph) end-to-end. This plan (08-05) fills the two remaining marked
sections:

  - Drop Folder + Ingest from URL: list ``scan_uningested()`` PDFs, ingest
    one/all, self-cleaning after MinerU success (D-10 --
    ``gui.jobs.make_drop_cleanup``, T-08-14/T-08-15); plus URL ingestion
    parity for ``ingest.py --url`` (WEB ingestion row).
  - Per-Paper Orchestration: pick which of note/biblio/embed/link run, and
    in what order, against any project paper (ROADMAP Processing mandate).

Detection is a scan on page load only (``scan_uningested``/
``scan_project_papers`` inside the GET handler below) -- no file-watcher
daemon is ever introduced (INBOX-01 stays v2).

Every state-changing route here builds its argv exclusively via
``gui.jobs.build_action_argv`` and enqueues through ``gui.jobs.enqueue`` --
never a second job-spawning path (T-08-02).
"""

import pathlib
from typing import Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import PlainTextResponse

from gui import jobs as gui_jobs
from gui.config import load_gui_config
from gui.scan import scan_uningested

router = APIRouter()


def _job_status_response(request: Request, job, status_code: int = 200):
    from gui.app import templates

    return templates.TemplateResponse(
        request, "partials/job_status.html", {"job": job}, status_code=status_code,
    )


def _job_status_list_response(request: Request, job_list, status_code: int = 200):
    from gui.app import templates

    return templates.TemplateResponse(
        request, "partials/job_status_list.html", {"jobs": job_list}, status_code=status_code,
    )


def _form_error_response(request: Request, message: str, status_code: int = 200):
    from gui.app import templates

    return templates.TemplateResponse(
        request, "partials/form_error.html", {"message": message}, status_code=status_code,
    )


def _is_safe_drop_filename(filename: str) -> bool:
    """Reject any path separator or parent-reference. ``filename`` arrives
    straight from the browser (T-08-15) -- check both slash styles
    regardless of host OS, since a crafted request can send either."""
    if not filename:
        return False
    if "/" in filename or "\\" in filename:
        return False
    if filename in (".", ".."):
        return False
    return True


@router.get("/processing")
def processing_page(request: Request):
    """Plain ``def`` (not ``async def``): scan_uningested/scan_project_papers
    do blocking filesystem/registry I/O (RESEARCH.md Pitfall 5). Detection is
    this scan call only -- never a background watcher."""
    from gui.app import templates

    config = load_gui_config()
    return templates.TemplateResponse(
        request,
        "processing.html",
        {
            "active_page": "processing",
            "page_title": "Processing",
            "pending": scan_uningested(config),
        },
    )


@router.post("/processing/ingest")
def processing_ingest(
    request: Request,
    filename: str = Form(...),
    force_extract: Optional[str] = Form(None),
    refill: Optional[str] = Form(None),
):
    """Ingest one dropped PDF. ``filename`` is resolved strictly against
    ``uningested_dir`` with a basename-only check (T-08-15) -- it never
    becomes a path component beyond that single join."""
    config = load_gui_config()
    uningested_dir = config.get("uningested_dir")

    if not _is_safe_drop_filename(filename):
        return PlainTextResponse("Invalid filename", status_code=400)
    if not uningested_dir:
        return PlainTextResponse("No drop folder configured", status_code=404)

    abs_path = (pathlib.Path(uningested_dir) / filename).resolve()
    if not abs_path.is_file():
        return PlainTextResponse("Not Found", status_code=404)

    argv = gui_jobs.build_action_argv(
        "ingest_pdf",
        config,
        path=str(abs_path),
        force_extract=bool(force_extract),
        refill=bool(refill),
    )
    on_success = gui_jobs.make_drop_cleanup(config, str(abs_path))
    job_id = gui_jobs.enqueue("ingest_pdf", argv, on_success=on_success)
    return _job_status_response(request, gui_jobs.get(job_id))


@router.post("/processing/ingest-all")
def processing_ingest_all(request: Request):
    """Enqueue one ingest_pdf job per pending drop-folder PDF, in the same
    name-sorted order ``scan_uningested`` already returns; the strict FIFO
    queue (D-06) serializes them."""
    config = load_gui_config()
    uningested_dir = config.get("uningested_dir")
    pending = scan_uningested(config)

    job_list = []
    for pdf in pending:
        abs_path = (pathlib.Path(uningested_dir) / pdf["name"]).resolve()
        argv = gui_jobs.build_action_argv("ingest_pdf", config, path=str(abs_path))
        on_success = gui_jobs.make_drop_cleanup(config, str(abs_path))
        job_id = gui_jobs.enqueue("ingest_pdf", argv, on_success=on_success)
        job_list.append(gui_jobs.get(job_id))

    return _job_status_list_response(request, job_list)


@router.post("/processing/ingest-url")
def processing_ingest_url(request: Request, url: Optional[str] = Form(None)):
    """Ingest a paper by URL (WEB ingestion parity). The URL is one argv
    element, never a shell string -- defuddle handles fetch/parse."""
    config = load_gui_config()
    value = (url or "").strip()
    if not value or not (value.startswith("http://") or value.startswith("https://")):
        return _form_error_response(
            request, "Enter a valid http:// or https:// URL to ingest."
        )

    argv = gui_jobs.build_action_argv("ingest_url", config, url=value)
    job_id = gui_jobs.enqueue("ingest_url", argv)
    return _job_status_response(request, gui_jobs.get(job_id))
