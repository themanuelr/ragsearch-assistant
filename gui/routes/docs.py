"""Docs & Help page -- server-rendered markdown from a fixed whitelist (D-21).

GET /docs renders README.md by default; GET /docs/{slug} renders any other
whitelisted doc. The ``slug`` path segment is looked up as a ``DOCS`` dict
key ONLY -- it never becomes a filesystem path component, so a
non-whitelisted or traversal-shaped slug always 404s before any ``open()``
call happens (T-08-10).

Markdown -> HTML conversion happens server-side via ``markdown.markdown()``
(python-markdown, D-21) with the ``fenced_code``/``tables`` extensions. The
converted HTML is repo-authored content (README/SKILL.md/config.example.json,
never user/LLM-derived text) and is injected via a single explicitly-marked
trusted template variable (``| safe``) -- everything else on the page stays
autoescaped.
"""

import pathlib

import markdown
from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

router = APIRouter()

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# Fixed whitelist: slug -> (title, absolute path under the repo root). The
# slug is a dict key lookup only -- request-derived text never reaches
# open()/read_text() unless it is already a key in this module-level dict.
DOCS = {
    "readme": ("README", _REPO_ROOT / "README.md"),
    "obsidian-note-format": (
        "Obsidian Note Format",
        _REPO_ROOT / ".claude" / "skills" / "obsidian-note-format" / "SKILL.md",
    ),
    "config-reference": ("Config Reference", _REPO_ROOT / "config.example.json"),
}

_DEFAULT_SLUG = "readme"


def _render_doc(slug: str):
    """Returns (title, html) for a whitelisted slug, or None for an unknown
    slug (caller returns 404 -- no filesystem access happens for an unknown
    slug at all)."""
    entry = DOCS.get(slug)
    if entry is None:
        return None
    title, path = entry
    if not path.exists():
        return title, "<p>doc not found on disk</p>"
    text = path.read_text(encoding="utf-8")
    if slug == "config-reference":
        text = (
            "Every key `config.json` accepts, shown here as the example "
            "template (`config.example.json`):\n\n```json\n" + text + "\n```"
        )
    html = markdown.markdown(text, extensions=["fenced_code", "tables"])
    return title, html


def _docs_response(request: Request, slug: str):
    from gui.app import templates

    result = _render_doc(slug)
    if result is None:
        return PlainTextResponse("Not Found", status_code=404)
    title, html = result
    picker = [{"slug": s, "title": t} for s, (t, _p) in DOCS.items()]
    return templates.TemplateResponse(
        request,
        "docs.html",
        {
            "active_page": "docs",
            "page_title": "Docs & Help",
            "current_slug": slug,
            "doc_title": title,
            "doc_html": html,
            "picker": picker,
        },
    )


@router.get("/docs")
def docs_page(request: Request):
    return _docs_response(request, _DEFAULT_SLUG)


@router.get("/docs/{slug}")
def docs_page_slug(request: Request, slug: str):
    return _docs_response(request, slug)
