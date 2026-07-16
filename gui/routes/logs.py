"""Logs & Activity page + the /jobs/* job-control API (Phase 8 Plan 03).

Owns every ``/jobs/*`` endpoint (enqueue/status/cancel/log) so ``gui/app.py``
is never edited again after 08-01. Every handler is a plain ``def`` (not
``async def`` -- RESEARCH.md Pitfall 5): FastAPI runs sync routes in a thread
pool, so the blocking ``gui.jobs``/filesystem calls here never stall the
event loop that job-status polling and the Origin-check middleware share.

``job_id`` path params are validated against a strict charset (letters,
digits, hyphens -- the exact shape ``gui.jobs._new_job_id`` produces) before
any filesystem path is built from them (T-08-03) -- a crafted id can never
traverse outside ``.local/gui_jobs/``.
"""

import pathlib
import re
from typing import Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import PlainTextResponse

from gui import jobs as gui_jobs
from gui.config import load_gui_config

router = APIRouter()

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9-]+$")


def _job_status_response(request: Request, job, status_code: int = 200):
    from gui.app import templates

    return templates.TemplateResponse(
        request, "partials/job_status.html", {"job": job}, status_code=status_code,
    )


@router.get("/logs")
def logs_page(request: Request):
    from gui.app import templates

    history = gui_jobs.list_jobs()
    live = [job for job in history if job.status in ("queued", "running")]
    return templates.TemplateResponse(
        request,
        "logs.html",
        {
            "active_page": "logs",
            "page_title": "Logs & Activity",
            "history": history,
            "live": live,
        },
    )


@router.post("/jobs/enqueue")
def jobs_enqueue(
    request: Request,
    action: str = Form(...),
    path: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    doi: Optional[str] = Form(None),
    stem: Optional[str] = Form(None),
    paperjson: Optional[str] = Form(None),
    force: Optional[str] = Form(None),
    force_extract: Optional[str] = Form(None),
    refill: Optional[str] = Form(None),
    vault_path: Optional[str] = Form(None),
    registry_path: Optional[str] = Form(None),
    project_name: Optional[str] = Form(None),
):
    """Generic job-enqueue endpoint every action across Phase 8 reuses: build
    argv, enqueue, return the job_status partial (HTMX swap target). This
    plan wires the Processing page's two bulk actions (embed_all/link_all,
    which need no extra params); 08-04..08-07's per-paper/URL/DOI/setup
    actions post the same field names already accepted here.
    """
    config = load_gui_config()
    params = {}
    if path:
        params["path"] = path
    if url:
        params["url"] = url
    if doi:
        params["doi"] = doi
    if stem:
        params["stem"] = stem
    if paperjson:
        params["paperjson"] = paperjson
    if force:
        params["force"] = True
    if force_extract:
        params["force_extract"] = True
    if refill:
        params["refill"] = True
    if vault_path:
        params["vault_path"] = vault_path
    if registry_path:
        params["registry_path"] = registry_path
    if project_name:
        params["project_name"] = project_name

    try:
        argv = gui_jobs.build_action_argv(action, config, **params)
    except ValueError as e:
        return PlainTextResponse(str(e), status_code=400)

    job_id = gui_jobs.enqueue(action, argv)
    job = gui_jobs.get(job_id)
    return _job_status_response(request, job)


@router.get("/jobs/{job_id}/status")
def job_status(request: Request, job_id: str):
    if not _JOB_ID_RE.match(job_id):
        return PlainTextResponse("Not Found", status_code=404)
    job = gui_jobs.get(job_id)
    if job is None:
        return PlainTextResponse("Not Found", status_code=404)
    return _job_status_response(request, job)


@router.post("/jobs/{job_id}/cancel")
def job_cancel(request: Request, job_id: str):
    if not _JOB_ID_RE.match(job_id):
        return PlainTextResponse("Not Found", status_code=404)
    gui_jobs.cancel(job_id)
    job = gui_jobs.get(job_id)
    if job is None:
        return PlainTextResponse("Not Found", status_code=404)
    return _job_status_response(request, job)


@router.get("/jobs/{job_id}/log")
def job_log(request: Request, job_id: str):
    from gui.app import templates

    if not _JOB_ID_RE.match(job_id):
        return PlainTextResponse("Not Found", status_code=404)
    job = gui_jobs.get(job_id)
    if job is None:
        return PlainTextResponse("Not Found", status_code=404)

    log_text = ""
    if job.log_path:
        log_file = pathlib.Path(job.log_path)
        if log_file.exists():
            try:
                log_text = log_file.read_text(encoding="utf-8")
            except OSError:
                log_text = "[log unavailable]"

    return templates.TemplateResponse(
        request,
        "job_log.html",
        {
            "active_page": "logs",
            "page_title": f"Job Log — {job.id}",
            "job": job,
            "log_text": log_text,
        },
    )
