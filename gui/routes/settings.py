"""Settings page -- stub router for the Phase 8 walking skeleton (08-01).
Filled in by 08-04.
"""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/settings")
def settings_page(request: Request):
    from gui.app import templates

    return templates.TemplateResponse(
        request,
        "stub.html",
        {
            "active_page": "settings",
            "page_title": "Settings",
            "coming_soon": "coming soon — filled by plan 08-04",
        },
    )
