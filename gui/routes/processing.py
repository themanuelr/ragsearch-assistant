"""Processing page (Phase 8 Plan 03: Bulk Actions section).

The 08-01 stub's placeholder ``POST /processing`` route (added purely to give
the Origin-check CSRF middleware a state-changing route to exercise before
this plan existed) is retired here -- ``POST /jobs/enqueue``
(gui/routes/logs.py) is now the real state-changing route every action on
this page posts to, and it existed before this file was ever reached by a
request (the middleware short-circuits on Origin mismatch before routing, so
tests/test_gui_app.py::test_csrf_origin_rejected still passes unchanged).

08-05 inserts the Drop Folder and Per-Paper Orchestration sections into
processing.html (marked with Jinja comments below) on top of this file
without restructuring it.
"""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/processing")
def processing_page(request: Request):
    from gui.app import templates

    return templates.TemplateResponse(
        request,
        "processing.html",
        {
            "active_page": "processing",
            "page_title": "Processing",
        },
    )
