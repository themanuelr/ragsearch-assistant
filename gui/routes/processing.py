"""Processing page -- stub router for the Phase 8 walking skeleton (08-01).
Filled in by 08-05.
"""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/processing")
def processing_page(request: Request):
    from gui.app import templates

    return templates.TemplateResponse(
        request,
        "stub.html",
        {
            "active_page": "processing",
            "page_title": "Processing",
            "coming_soon": "coming soon — filled by plan 08-05",
        },
    )


@router.post("/processing")
def processing_post(request: Request):
    """Stub POST target so the Origin-check CSRF middleware (T-08-01) has a
    real state-changing route to exercise before 08-05 fills in job enqueue.
    """
    from gui.app import templates

    return templates.TemplateResponse(
        request,
        "stub.html",
        {
            "active_page": "processing",
            "page_title": "Processing",
            "coming_soon": "coming soon — filled by plan 08-05",
        },
    )
