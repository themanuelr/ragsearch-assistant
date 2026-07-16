"""Logs & Activity page -- stub router for the Phase 8 walking skeleton
(08-01). Filled in by 08-03.
"""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/logs")
def logs_page(request: Request):
    from gui.app import templates

    return templates.TemplateResponse(
        request,
        "stub.html",
        {
            "active_page": "logs",
            "page_title": "Logs & Activity",
            "coming_soon": "coming soon — filled by plan 08-03",
        },
    )
