"""FastAPI app core for the Phase 8 local web GUI (D-01/D-02/D-03).

This module owns exactly two things every downstream Phase 8 plan depends on
without ever needing to touch this file again:

1. The shared ``app`` (FastAPI instance) + ``templates`` (Jinja2Templates)
   symbols, static-file mount, and the Origin-check CSRF middleware
   (T-08-01 / RESEARCH.md Pitfall 3).
2. The ``app.include_router(...)`` wiring for the six page routers defined
   in ``gui/routes/*.py`` -- each of those modules exposes a module-level
   ``router = APIRouter()`` symbol that downstream plans fill in, so route
   plans never need to edit this file.
"""

import pathlib

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from gui.chat_markdown import render_chat_markdown
from gui.format import format_file_size

_GUI_DIR = pathlib.Path(__file__).resolve().parent

# FastAPI auto-registers Swagger UI at "/docs" (plus "/redoc"/"/openapi.json")
# by default, which collides directly with this app's own server-rendered
# Docs & Help page at GET /docs (D-21, gui/routes/docs.py). This is a
# Jinja2/HTMX page-rendering app, not a consumed HTTP API, so the interactive
# API docs have no purpose here -- disable all three.
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=str(_GUI_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(_GUI_DIR / "templates"))
# The untrusted-output chokepoint (T-08-GAP-31/32, GUI-09): the ONLY filter
# permitted to turn model-derived chat text into HTML -- see
# gui/chat_markdown.py's module docstring for the full trust rationale.
templates.env.filters["chat_markdown"] = render_chat_markdown
# A plain display formatter, unlike the chokepoint above -- it handles no
# untrusted content and emits no HTML, just a human-readable size string.
templates.env.filters["format_file_size"] = format_file_size


@app.middleware("http")
async def reject_cross_origin_state_changes(request: Request, call_next):
    """Reject cross-origin state-changing requests (T-08-01, ASVS V13).

    A localhost bind is not automatically CSRF-safe: any page open in another
    browser tab can issue a fetch()/form POST to this server and the browser
    will deliver it. If an ``Origin`` header is present on a state-changing
    request (POST/PUT/DELETE), it must match this server's own host:port
    exactly, or the request is rejected with 403. An absent ``Origin`` is
    allowed (same-origin plain form submissions and non-browser/test clients
    often omit it) -- never add permissive CORS headers, which would defeat
    this check entirely.
    """
    if request.method in ("POST", "PUT", "DELETE"):
        origin = request.headers.get("origin")
        if origin:
            expected = f"http://{request.url.hostname}:{request.url.port}"
            if origin != expected:
                return PlainTextResponse(
                    "Forbidden: cross-origin request rejected", status_code=403
                )
    return await call_next(request)


@app.get("/")
def root():
    return RedirectResponse(url="/overview/project")


# Six stub page routers (gui/routes/*.py) -- each exposes a module-level
# `router = APIRouter()` symbol filled in by downstream Phase 8 plans.
from gui.routes import chat, docs, logs, overview, processing, settings  # noqa: E402

app.include_router(overview.router)
app.include_router(processing.router)
app.include_router(chat.router)
app.include_router(logs.router)
app.include_router(docs.router)
app.include_router(settings.router)
