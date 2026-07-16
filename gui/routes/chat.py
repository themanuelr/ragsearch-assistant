"""Chat page -- streaming Ollama chat with model picker, job-blocked state,
and persistent conversations (Phase 8 Plan 07, D-13/D-16/D-17/D-18).
Retrieval scopes (D-14/D-15) are wired by Task 3 into ``_resolve_scope``.

Every handler is a plain ``def`` (not ``async def`` -- RESEARCH.md Pitfall 5):
``chat_store``, ``gui.jobs``, and ``scan_project_papers`` all do blocking
file I/O, and FastAPI runs sync routes in its thread pool, so job-status
polling and the CSRF middleware never stall behind a chat request.

Retrieval-scope context assembly (D-14/D-15, Task 3) must happen
synchronously inside ``POST /chat/send`` -- it has to complete before the LLM
call, and ``_search``/vault-note reads are themselves blocking I/O. The
assembled LLM-ready message list and any Sources/banner/truncation metadata
are handed off to ``GET /chat/stream`` through ``_PENDING``, an in-process
dict keyed by conversation id -- deliberately NOT part of the on-disk
conversation JSON schema (chat_store's persisted shape stays exactly
``{id, title, model, created, updated, messages}``); this is ephemeral,
single-process, single-user hand-off state, not a second source of truth for
anything chat_store already owns.
"""

import json

from fastapi import APIRouter, Form, Request
from fastapi.responses import PlainTextResponse, RedirectResponse, StreamingResponse

from gui import chat_store
from gui import jobs as gui_jobs
from gui.config import load_gui_config
from gui.ollama_chat import _stream_ollama_chat, list_models
from gui.scan import scan_project_papers
from scripts.ingest import OLLAMA_MODEL as _DEFAULT_PIPELINE_MODEL

router = APIRouter()

# D-16 verbatim copy (08-UI-SPEC.md Copywriting Contract).
_BLOCKED_COPY_TEMPLATE = (
    "Chat is paused while a pipeline job runs (queue position {n}). "
    "It'll unlock automatically when the job finishes."
)

# conv_id -> {"llm_messages": [...], "sources": [...] | None,
#             "banner": str | None, "truncation_notes": [str] | None}
# Ephemeral hand-off from POST /chat/send to GET /chat/stream -- never
# persisted to disk. Monkeypatchable by tests.
_PENDING: dict = {}


def _active_job_count() -> int:
    """Count of jobs currently queued or running -- the D-16 'queue position'
    shown in the blocked copy."""
    return sum(1 for job in gui_jobs.list_jobs() if job.status in ("queued", "running"))


# ---------------------------------------------------------------------------
# GET /chat -- sidebar + active conversation + model picker + blocked state
# ---------------------------------------------------------------------------

@router.get("/chat")
def chat_page(request: Request):
    from gui.app import templates

    config = load_gui_config()
    conversations = chat_store.list_conversations()

    conv_id = request.query_params.get("conv")
    active_conv = chat_store.load(conv_id) if conv_id else None
    if active_conv is None and conversations:
        active_conv = conversations[0]

    model_names, model_error = list_models()
    default_model = config.get("model_name", _DEFAULT_PIPELINE_MODEL)
    selected_model = (active_conv or {}).get("model") or default_model

    busy = gui_jobs.is_busy()
    queue_position = _active_job_count() if busy else 0

    papers = scan_project_papers(config)

    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "active_page": "chat",
            "page_title": "Chat",
            "conversations": conversations,
            "active_conv": active_conv,
            "model_names": model_names,
            "model_error": model_error,
            "selected_model": selected_model,
            "busy": busy,
            "blocked_copy": _BLOCKED_COPY_TEMPLATE.format(n=queue_position),
            "papers": papers,
        },
    )


# ---------------------------------------------------------------------------
# Sidebar actions: new / rename / delete
# ---------------------------------------------------------------------------

@router.post("/chat/new")
def chat_new(request: Request):
    conv = chat_store.create_conversation()
    return RedirectResponse(url=f"/chat?conv={conv['id']}", status_code=303)


@router.post("/chat/{conv_id}/rename")
def chat_rename(request: Request, conv_id: str, title: str = Form(...)):
    chat_store.rename(conv_id, title)
    return RedirectResponse(url=f"/chat?conv={conv_id}", status_code=303)


@router.post("/chat/{conv_id}/delete")
def chat_delete(request: Request, conv_id: str):
    chat_store.delete(conv_id)
    return RedirectResponse(url="/chat", status_code=303)


# ---------------------------------------------------------------------------
# POST /chat/send -- append user message, resolve retrieval scope, hand off
# the assembled LLM messages to /chat/stream (D-16 hard block re-checked here)
# ---------------------------------------------------------------------------

def _resolve_scope(scope: str, message: str, notes: list, config: dict) -> tuple:
    """Resolve the D-14/D-15 retrieval scope into
    ``(context_messages, sources, banner, truncation_notes)``.

    Task 2 (this commit): ``scope`` is accepted from the form but not yet
    wired -- every scope behaves as "none" (no retrieval, no context
    injection). Task 3 fills in the ChromaDB (D-14) and paper-vault (D-15)
    bodies here.
    """
    return [], None, None, None


@router.post("/chat/send")
def chat_send(
    request: Request,
    conv: str = Form(...),
    message: str = Form(...),
    model: str = Form(...),
    scope: str = Form("none"),
    notes: list = Form([]),
):
    from gui.app import templates

    # D-16: hard block, re-checked server-side -- never trust the disabled
    # input alone (T-08-21). No Ollama/_search call happens below this line
    # when busy.
    if gui_jobs.is_busy():
        queue_position = _active_job_count()
        return templates.TemplateResponse(
            request,
            "partials/chat_blocked.html",
            {"blocked_copy": _BLOCKED_COPY_TEMPLATE.format(n=queue_position)},
        )

    conversation = chat_store.load(conv)
    if conversation is None:
        return PlainTextResponse("Not Found", status_code=404)

    config = load_gui_config()

    context_messages, sources, banner, truncation_notes = _resolve_scope(
        scope, message, notes, config
    )

    conversation["messages"].append({"role": "user", "content": message})
    conversation["model"] = model
    chat_store.save(conversation)

    llm_messages = context_messages + [
        {"role": m["role"], "content": m["content"]} for m in conversation["messages"]
    ]

    _PENDING[conv] = {
        "llm_messages": llm_messages,
        "sources": sources,
        "banner": banner,
        "truncation_notes": truncation_notes,
    }

    return templates.TemplateResponse(
        request,
        "partials/chat_pending.html",
        {
            "conv_id": conv,
            "model": model,
            "message": message,
            "banner": banner,
            "truncation_notes": truncation_notes,
        },
    )


# ---------------------------------------------------------------------------
# GET /chat/stream -- SSE bridge; appends the finished assistant message
# ---------------------------------------------------------------------------

def _sse_stream(conv_id: str, model: str):
    config = load_gui_config()
    timeout = config.get("ollama_section_timeout", 300)

    pending = _PENDING.pop(conv_id, None)
    conversation = chat_store.load(conv_id)
    if conversation is None:
        return

    if pending is not None:
        llm_messages = pending["llm_messages"]
        sources = pending["sources"]
        truncation_notes = pending["truncation_notes"]
    else:
        # Defensive fallback (e.g. a stream reopened without a fresh send):
        # no retrieval context, just the raw transcript.
        llm_messages = [
            {"role": m["role"], "content": m["content"]} for m in conversation["messages"]
        ]
        sources = None
        truncation_notes = None

    accumulated: list = []
    for frame in _stream_ollama_chat(llm_messages, model, timeout=timeout):
        payload = json.loads(frame[len("data: "):-2])
        if "token" in payload:
            accumulated.append(payload["token"])
        yield frame
        if payload.get("done") or "error" in payload:
            break

    conversation = chat_store.load(conv_id)
    if conversation is None:
        return
    assistant_msg = {"role": "assistant", "content": "".join(accumulated)}
    if sources:
        assistant_msg["sources"] = sources
    if truncation_notes:
        assistant_msg["truncation_notes"] = truncation_notes
    conversation["messages"].append(assistant_msg)
    conversation["model"] = model
    chat_store.save(conversation)


@router.get("/chat/stream")
def chat_stream(request: Request, conv: str, model: str = ""):
    config = load_gui_config()
    conversation = chat_store.load(conv)
    if conversation is None:
        def _missing():
            yield 'data: {"error": "[chat error: conversation not found]"}\n\n'

        return StreamingResponse(_missing(), media_type="text/event-stream")

    chosen_model = model or conversation.get("model") or config.get(
        "model_name", _DEFAULT_PIPELINE_MODEL
    )
    return StreamingResponse(
        _sse_stream(conv, chosen_model), media_type="text/event-stream"
    )
