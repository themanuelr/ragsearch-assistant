"""Overview pages (Project + Universal). Project page + drill-down partial
filled in by 08-02 (D-09 live scan). Universal page (D-11: read-only global
table + the single 'Import into This Project' action, REG-03) and the
drill-down's per-stage re-run buttons (D-12) filled in by 08-06.
"""

import pathlib
import re

from fastapi import APIRouter, Request

from gui.config import load_gui_config
from gui.scan import scan_paper, scan_project_papers, scan_uningested, scan_universal

router = APIRouter()


def _rerun_status_id_slug(registry_key: str) -> str:
    """Whitelist-slug a registry key into a syntactically valid, escape-free
    CSS id fragment. Mirrors gui/jobs.py::_new_job_id's established slug
    idiom (Don't Hand-Roll a second slug convention). Any character outside
    ``[A-Za-z0-9]`` (DOI slashes/dots, arXiv colons, sha256's own colon,
    even '#'/'?'/'%'/'('/')' ) collapses to a single '-', so the result is
    always a valid unescaped CSS id selector regardless of the key's shape.
    """
    return re.sub(r"[^A-Za-z0-9]+", "-", registry_key).strip("-") or "paper"


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
    """HTMX drill-down partial for one paper (D-12).

    ``registry_key`` arrives URL-encoded (DOIs contain slashes) via a
    ``:path`` converter and is used strictly as an opaque dict lookup key
    into the registry -- never joined into a filesystem path (T-08-07).

    ``paperjson_cache_present`` is computed here (a plain ``Path.is_file()``
    check on the already-resolved cache path scan_paper returned) rather than
    in gui/scan.py -- every re-run action (note/biblio/embed/link_paper)
    resolves the same PaperJSON cache file via --stem, so this one flag gates
    all four re-run buttons in the partial.
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
    paperjson_cache_present = pathlib.Path(paper["cache_paths"]["paperjson_path"]).is_file()
    safe_key = _rerun_status_id_slug(registry_key)
    return templates.TemplateResponse(
        request,
        "partials/paper_row.html",
        {
            "paper": paper,
            "paperjson_cache_present": paperjson_cache_present,
            "safe_key": safe_key,
        },
    )


@router.get("/overview/universal")
def overview_universal(request: Request):
    """Render the read-only universal registry table (D-11).

    Plain ``def`` (not ``async def``): scan_universal does blocking
    file/registry I/O, mirroring overview_project's Pitfall 5 rationale.
    """
    from gui.app import templates

    config = load_gui_config()
    entries = scan_universal(config)
    return templates.TemplateResponse(
        request,
        "overview_universal.html",
        {
            "active_page": "overview_universal",
            "page_title": "Overview (Universal)",
            "entries": entries,
        },
    )
