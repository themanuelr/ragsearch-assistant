"""Overview pages (Project + Universal). Project page + drill-down partial
filled in by 08-02 (D-09 live scan). Universal page remains a stub, filled
in by 08-06.
"""

from fastapi import APIRouter, Request

from gui.config import load_gui_config
from gui.scan import scan_paper, scan_project_papers, scan_uningested

router = APIRouter()


@router.get("/overview/project")
def overview_project(request: Request):
    """Render the live-scanned per-paper stage table + drop-folder pending list.

    Plain ``def`` (not ``async def``, RESEARCH.md Pitfall 5): scan_project_papers
    and scan_uningested do blocking file/Chroma I/O, and FastAPI runs sync
    routes in its thread pool, avoiding an event-loop stall.
    """
    from gui.app import templates

    config = load_gui_config()
    papers = scan_project_papers(config)
    pending = scan_uningested(config)
    return templates.TemplateResponse(
        request,
        "overview_project.html",
        {
            "active_page": "overview_project",
            "page_title": "Overview (Project)",
            "papers": papers,
            "pending": pending,
        },
    )


@router.get("/overview/project/paper/{registry_key:path}")
def overview_project_paper(request: Request, registry_key: str):
    """HTMX drill-down partial for one paper (D-12, read-only this plan).

    ``registry_key`` arrives URL-encoded (DOIs contain slashes) via a
    ``:path`` converter and is used strictly as an opaque dict lookup key
    into the registry -- never joined into a filesystem path (T-08-07).
    """
    from gui.app import templates

    config = load_gui_config()
    paper = scan_paper(config, registry_key)
    if paper is None:
        return templates.TemplateResponse(
            request,
            "partials/paper_row.html",
            {"paper": None, "registry_key": registry_key},
            status_code=404,
        )
    return templates.TemplateResponse(
        request,
        "partials/paper_row.html",
        {"paper": paper},
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
