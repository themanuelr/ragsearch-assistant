"""Chat page -- stub router for the Phase 8 walking skeleton (08-01).
Filled in by 08-07.
"""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/chat")
def chat_page(request: Request):
    from gui.app import templates

    return templates.TemplateResponse(
        request,
        "stub.html",
        {
            "active_page": "chat",
            "page_title": "Chat",
            "coming_soon": "coming soon — filled by plan 08-07",
        },
    )
