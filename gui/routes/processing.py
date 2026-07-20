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
from gui.scan import filter_and_paginate, scan_paper, scan_project_papers, scan_uningested

router = APIRouter()

# Canonical per-paper script order (note -> biblio -> embed -> link); used
# only to break tied order values in POST /processing/orchestrate.
_CANONICAL_ORDER = {"note": 1, "biblio": 2, "embed": 3, "link": 4}
_ORCHESTRATE_ACTIONS = {"note": "note", "biblio": "biblio", "embed": "embed", "link": "link_paper"}


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
    this scan call only -- never a background watcher.

    Per-paper orchestration's paper selector is the shared picker (D-16,
    Phase 08.1 Plan 01/04) in radio mode -- the same ``filter_and_paginate``
    shape ``GET /overview/project`` feeds into its own first-page render, so
    a bare ``GET /processing`` already contains the first page of
    radio-selectable papers without a round-trip (see
    ``partials/paper_picker.html``'s own initial-render contract)."""
    from gui.app import templates

    config = load_gui_config()
    papers = scan_project_papers(config)
    result = filter_and_paginate(papers, "", 1, 25)
    return templates.TemplateResponse(
        request,
        "processing.html",
        {
            "active_page": "processing",
            "page_title": "Processing",
            "pending": scan_uningested(config),
            "result": result,
            "scope": "project",
            "q": "",
            "select_mode": "radio",
            "page_size": 25,
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


@router.post("/processing/orchestrate")
def processing_orchestrate(
    request: Request,
    registry_key: str = Form(...),
    note_include: Optional[str] = Form(None),
    note_order: int = Form(1),
    note_force: Optional[str] = Form(None),
    note_force_analysis: Optional[str] = Form(None),
    biblio_include: Optional[str] = Form(None),
    biblio_order: int = Form(2),
    embed_include: Optional[str] = Form(None),
    embed_order: int = Form(3),
    link_include: Optional[str] = Form(None),
    link_order: int = Form(4),
):
    """Enqueue any chosen subset/order of note/biblio/embed/link for one
    project paper (ROADMAP: "pick which scripts run on which paper, and in
    what order"). ``registry_key`` is looked up as an opaque registry dict
    key via ``scan_paper`` -- never a filesystem path. The strict FIFO queue
    (D-06) guarantees the enqueued order is the execution order.

    ``note_force``/``note_force_analysis`` are the note row's only behavior
    flags (see 08-19-PLAN.md flag_scope table -- the other three scripts take
    no per-paper behavior flags). They exist because ``note.py``'s D-16
    skip-by-default (note.py:798) made this route's own note action a no-op
    against an already-noted paper -- there was no way to force regeneration
    from the GUI at all (Round-2 UAT gap 3). Likewise, the analysis-reuse
    self-skip (note.py:810, the 08-13 cost win) is only escapable via
    ``--force-analysis``; without a route parameter for it, force-analysis
    was equally unreachable from the GUI. An unticked checkbox submits no
    form field at all, so ``None`` is the correct unticked representation."""
    config = load_gui_config()

    paper = scan_paper(config, registry_key)
    if paper is None:
        return PlainTextResponse("Not Found", status_code=404)

    selected = []
    for name, include, order in (
        ("note", note_include, note_order),
        ("biblio", biblio_include, biblio_order),
        ("embed", embed_include, embed_order),
        ("link", link_include, link_order),
    ):
        if include:
            selected.append((name, order))

    if not selected:
        return _form_error_response(request, "Select at least one script to run.")

    selected.sort(key=lambda item: (item[1], _CANONICAL_ORDER[item[0]]))

    job_list = []
    for name, _order in selected:
        action = _ORCHESTRATE_ACTIONS[name]
        extra_params = {}
        if name == "note":
            extra_params["force"] = bool(note_force)
            extra_params["force_analysis"] = bool(note_force_analysis)
        argv = gui_jobs.build_action_argv(action, config, stem=paper["stem"], **extra_params)
        job_id = gui_jobs.enqueue(action, argv)
        job_list.append(gui_jobs.get(job_id))

    return _job_status_list_response(request, job_list)
