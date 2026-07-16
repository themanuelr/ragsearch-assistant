"""Overview pages (Project + Universal) -- stub router for the Phase 8 walking
skeleton (08-01). Filled in by 08-02 (project) and 08-06 (universal).
"""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/overview/project")
def overview_project(request: Request):
    from gui.app import templates

    return templates.TemplateResponse(
        request,
        "stub.html",
        {
            "active_page": "overview_project",
            "page_title": "Overview (Project)",
            "coming_soon": "coming soon — filled by plan 08-02",
        },
    )


@router.get("/overview/universal")
def overview_universal(request: Request):
    from gui.app import templates

    return templates.TemplateResponse(
        request,
        "stub.html",
        {
            "active_page": "overview_universal",
            "page_title": "Overview (Universal)",
            "coming_soon": "coming soon — filled by plan 08-06",
        },
    )
