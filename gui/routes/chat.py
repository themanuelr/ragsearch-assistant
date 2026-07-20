"""Chat page -- streaming Ollama chat with retrieval scopes, model picker,
job-blocked state, and persistent conversations (Phase 8 Plan 07, D-13..D-18).

Every handler is a plain ``def`` (not ``async def`` -- RESEARCH.md Pitfall 5):
``chat_store``, ``gui.jobs``, ``scan_project_papers``, and
``scripts.embed._search`` all do blocking file/Chroma I/O, and FastAPI runs
sync routes in its thread pool, so job-status polling and the CSRF
middleware never stall behind a chat request.

Retrieval-scope context assembly (D-14/D-15) happens synchronously inside
``POST /chat/send`` (``_resolve_scope``) -- it must complete before the LLM
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
import pathlib

from fastapi import APIRouter, Form, Request
from fastapi.responses import PlainTextResponse, RedirectResponse, StreamingResponse

from gui import chat_store
from gui import jobs as gui_jobs
from gui.config import load_gui_config
from gui.ollama_chat import _stream_ollama_chat, list_models
from gui.scan import scan_project_papers
from scripts.embed import _search
from scripts.ingest import (
    DEFAULT_NUM_CTX_CAP,
    OLLAMA_MODEL as _DEFAULT_PIPELINE_MODEL,
    _estimate_num_ctx,
    _read_registry,
)

router = APIRouter()

# D-16 verbatim copy (08-UI-SPEC.md Copywriting Contract).
_BLOCKED_COPY_TEMPLATE = (
    "Chat is paused while a pipeline job runs (queue position {n}). "
    "It'll unlock automatically when the job finishes."
)

# Gap C (08-UAT.md T-08-GAP-13): Paper Vault scope with nothing resolvable
# is a HARD block -- see _resolve_scope's docstring for why this is
# deliberately different from D-14's chroma fail-open.
_VAULT_EMPTY_SCOPE_BANNER = (
    "Select at least one paper in the Papers list to use Paper Vault scope."
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
# GET /chat/{conv_id}/messages -- rendered message-history partial, the
# race-free re-render target for _sse_stream's terminal saved-frame
# (Task 2, T-08-GAP-33). Renders the SAME partial GET /chat renders, so the
# page and the post-turn re-render can never drift (Test I).
# ---------------------------------------------------------------------------

@router.get("/chat/{conv_id}/messages")
def chat_messages(request: Request, conv_id: str):
    from gui.app import templates

    # chat_store.load validates conv_id against its strict charset BEFORE
    # constructing any path (T-08-18) and returns None for an invalid/
    # absent id -- no filesystem exception ever reaches this handler.
    active_conv = chat_store.load(conv_id)
    if active_conv is None:
        return PlainTextResponse("Not Found", status_code=404)

    return templates.TemplateResponse(
        request,
        "partials/chat_messages.html",
        {"active_conv": active_conv},
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

def _format_registry_metadata_line(entry: dict) -> str:
    """Build a readable 'Authors (Year) — Journal' line from a registry
    entry for the chroma excerpt header (Gap B). Omits any falsy field
    rather than emitting an empty or "None" label -- a header claiming an
    unknown author is worse than one that stays silent. Returns "" when the
    entry carries no usable metadata at all (unknown registry_key, or every
    field falsy); the caller must treat that as "no metadata line"."""
    authors = entry.get("authors")
    if isinstance(authors, list):
        authors_str = ", ".join(a for a in authors if a)
    else:
        authors_str = authors or ""

    lead_parts = []
    if authors_str:
        lead_parts.append(authors_str)
    year = entry.get("year")
    if year:
        lead_parts.append(f"({year})")
    lead = " ".join(lead_parts)

    journal = entry.get("journal")
    if journal and lead:
        return f"{lead} — {journal}"
    if journal:
        return str(journal)
    return lead


def _resolve_scope(
    scope: str,
    message: str,
    notes: list,
    config: dict,
    chroma_query: str | None = None,
    n_results: int = 5,
) -> tuple:
    """Resolve the D-14/D-15 retrieval scope into
    ``(context_messages, sources, banner, truncation_notes, blocked)``.

    scope == "chroma" (D-14): runs ``scripts.embed._search`` in-process
    (the one sanctioned in-process pipeline call) and frames every returned
    section excerpt into a system-role preamble. Every row of the returned
    ``sources`` list renders as one "Sources" entry -- retrieved context is
    never injected silently. A ``_search`` error result falls back to scope
    none for this message (fail-open, matching pipeline conventions) and is
    surfaced as a visible banner instead of raising. ``blocked`` is always
    False on this path -- see the fail-open-vs-hard-block note below.

    ``chroma_query`` (08.1-03, D-10) is an OPTIONAL retrieval-only query,
    separate from the chat ``message`` -- a blank/whitespace value falls
    back to ``message`` (today's behavior byte-unchanged). ``n_results``
    (D-13) threads straight to ``_search``'s second arg; the caller
    (``chat_send``) is responsible for clamping it to 1..20 before this
    function is called (Security Domain, T-08.1-03-01) -- this function
    does not re-clamp so a directly-called out-of-range value is visible to
    a caller that deliberately wants it (e.g. a future CLI/test harness),
    while the one HTTP-facing entry point always clamps first.

    scope == "vault" (D-15): ``notes`` are vault-relative paths the client
    submitted, but only accepted when they match an already-scanned paper's
    own ``vault_note`` path (server-side paths from the scan -- never an
    arbitrary client-supplied filesystem path, T-08-05-adjacent hardening).
    The assembled prompt is sized against ``ollama_num_ctx_cap`` via
    ``_estimate_num_ctx`` (Pitfall 6); while genuinely oversized, the
    last-selected note is dropped and named in ``truncation_notes`` -- never
    a silent oversized request. When nothing resolves (empty selection, or
    every submitted path fails the ``allowed_notes`` check), ``blocked`` is
    True and ``banner`` carries actionable copy (Gap C, 08-UAT.md
    T-08-GAP-13).

    scope == "none" (or anything else): pass-through, no retrieval call.
    ``blocked`` is always False.

    Fail-open (chroma) vs hard-block (vault) is a DELIBERATE distinction,
    not an oversight to "unify": D-14's chroma fail-open is correct because
    the model can still answer usefully without retrieval, and the banner
    tells the user why sources are missing. Vault scope is different -- the
    user explicitly asked to ground on papers they did not select, so
    answering anyway produces a confident non-answer (the exact failure Gap
    C reports). Blocking before any persist/Ollama call is the honest
    outcome there.
    """
    if scope == "chroma":
        # D-10: blank/whitespace-only chroma_query falls back to the chat
        # message itself -- today's behavior, byte-unchanged when the box
        # is left empty.
        query = (chroma_query or "").strip() or message
        result = _search(query, n_results, config)
        if result.get("error"):
            # D-14 fail-open: a _search error falls back to scope none for
            # this message, surfaced as a visible banner -- never a silent
            # swallow and never a raised exception. Not a hard block: the
            # model still answers, just without retrieved context.
            return [], None, result["error"], None, False

        # Gap B (08-UAT.md): Chroma's stored metadata carries only
        # title/heading/registry_key/status/vault_note -- authors/year/
        # journal live in the registry, so a metadata question ("who is the
        # first author") is unanswerable from context unless it's looked up
        # here. Read once per call (never once per paper/section -- the loop
        # below is over ~25 rows), matching gui/scan.py's established
        # `_read_registry` idiom (Don't Hand-Roll).
        registry = _read_registry(config.get("registry_path") or "")

        rows = []
        excerpt_blocks = []
        for paper in result.get("papers") or []:
            registry_entry = registry.get(paper.get("registry_key")) or {}
            metadata_line = _format_registry_metadata_line(registry_entry)
            for section in paper.get("sections", []):
                rows.append({
                    "title": paper["title"],
                    "section": section["heading"],
                    "score": section["score"],
                })
                header = f"[{paper['title']} — {section['heading']}]"
                if metadata_line:
                    header += f"\n{metadata_line}"
                excerpt_blocks.append(f"{header}\n{section['excerpt']}")
        if not rows:
            return [], None, None, None, False

        context_messages = [{
            "role": "system",
            "content": (
                "Answer the user's question using the retrieved excerpts "
                "below when they are relevant. If the excerpts don't cover "
                "the question, say so explicitly rather than guessing.\n\n"
                + "\n\n".join(excerpt_blocks)
            ),
        }]
        return context_messages, rows, None, None, False

    if scope == "vault":
        # Server-side paths only: the client submits a vault-relative note
        # path, but it is accepted only if it matches an already-scanned
        # paper's own vault_note path -- never an arbitrary client-supplied
        # path read directly off disk.
        allowed_notes = {
            paper["cache_paths"]["vault_note"]: paper
            for paper in scan_project_papers(config)
        }
        vault_root = pathlib.Path(config.get("vault_path") or "")
        cap = config.get("ollama_num_ctx_cap", DEFAULT_NUM_CTX_CAP)

        note_texts = []
        for rel_path in notes or []:
            paper = allowed_notes.get(rel_path)
            if paper is None:
                continue
            try:
                text = (vault_root / rel_path).read_text(encoding="utf-8")
            except OSError:
                continue
            note_texts.append({"title": paper["title"], "text": text})

        dropped_titles: list = []
        # Pitfall 6 guard: drop the last-selected note while the assembled
        # prompt is genuinely oversized -- same double-check idiom as
        # scripts/ingest.py::_fill_section's over-size guard (ladder-maxed
        # AND raw length confirms it's not just an equal-to-a-rung fit).
        while len(note_texts) > 1:
            combined = "\n\n".join(n["text"] for n in note_texts)
            estimated_ctx = _estimate_num_ctx(combined, cap=cap)
            if estimated_ctx >= cap and len(combined) > cap * 4:
                dropped = note_texts.pop()
                dropped_titles.append(dropped["title"])
                continue
            break

        if not note_texts:
            # Gap C (08-UAT.md T-08-GAP-13): a vault selection that resolves
            # to nothing is a HARD block, not a fail-open -- the user
            # explicitly asked to ground on papers they did not select, and
            # answering anyway produces a confident non-answer.
            return [], None, _VAULT_EMPTY_SCOPE_BANNER, (dropped_titles or None), True

        sources = [
            {"title": n["title"], "section": "Full Note", "score": None}
            for n in note_texts
        ]
        context_messages = [{
            "role": "system",
            "content": (
                "Answer using the following full paper notes as context.\n\n"
                + "\n\n".join(f"[{n['title']}]\n{n['text']}" for n in note_texts)
            ),
        }]
        return context_messages, sources, None, (dropped_titles or None), False

    # scope == "none" (or unrecognized): pass-through, no retrieval call.
    return [], None, None, None, False


@router.post("/chat/send")
def chat_send(
    request: Request,
    conv: str = Form(...),
    message: str = Form(...),
    model: str = Form(...),
    scope: str = Form("none"),
    notes: list = Form([]),
    chroma_query: str = Form(""),
    n_results: int = Form(5),
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

    # D-13/T-08.1-03-01: HTML min/max is advisory only -- clamp server-side
    # before n_results ever reaches _resolve_scope/_search, mirroring the
    # is_busy re-check above (never trust a client-supplied constraint
    # alone).
    n_results = min(max(int(n_results), 1), 20)

    context_messages, sources, banner, truncation_notes, blocked = _resolve_scope(
        scope, message, notes, config, chroma_query=chroma_query, n_results=n_results
    )

    if blocked:
        # Gap C (08-UAT.md T-08-GAP-13): returned BEFORE the user-message
        # append/save and before any _PENDING write, mirroring the D-16
        # busy guard's early-return shape -- nothing is persisted, no
        # _PENDING entry is written, no stream is opened, and therefore no
        # Ollama call happens. A blocked turn can never leave an orphaned
        # user message.
        return templates.TemplateResponse(
            request,
            "partials/chat_scope_banner.html",
            {"banner": banner},
        )

    conversation["messages"].append({"role": "user", "content": message})
    # Gap A (08-15 Task 1, T-08-GAP-27): the assistant reply that will land
    # in _sse_stream's finally block occupies the NEXT slot in
    # conversation["messages"] -- computed here, right after the user
    # message append, so the index is exact even though the assistant
    # entry itself does not exist yet. Passed into chat_pending.html so the
    # pending container's DOM id is keyed per MESSAGE, not per conversation
    # (see that template's own comment for why a conversation-keyed id
    # duplicates on every send and breaks getElementById).
    assistant_index = len(conversation["messages"])
    # Gap B (08-14 Task 2): this maintains the picker's CURRENT DEFAULT --
    # consumed only by chat_page's selected_model pre-selection -- not a
    # claim about which model produced which reply. That fact lives on
    # each assistant message's own "model" field (_sse_stream's finally
    # block). Do not "fix" the apparent duplication by deleting either one.
    conversation["model"] = model
    chat_store.save(conversation)

    transcript = [
        {"role": m["role"], "content": m["content"]} for m in conversation["messages"]
    ]
    # Gap A (08-UAT.md T-08-GAP-10): context placed at position 0 is
    # maximally distant from the question it grounds and gets outweighed by
    # the accumulated transcript -- including the model's own prior
    # refusals, which it then pattern-continues instead of reading the
    # excerpts. Splicing context immediately before the final (just-sent)
    # user turn is what made the SAME retrieved context produce a correct,
    # grounded answer in the replay used to diagnose this gap. When
    # context_messages is empty this is exactly `transcript` -- the
    # scope-none path stays byte-identical and no empty system message is
    # ever injected.
    llm_messages = transcript[:-1] + context_messages + transcript[-1:]

    # D-12: the effective retrieval query (same blank-fallback rule as
    # _resolve_scope's chroma branch) is stashed here so _sse_stream can
    # persist it on the assistant message -- None for any non-chroma scope,
    # never overloaded into `sources` (a list of per-excerpt dicts).
    effective_chroma_query = (
        (chroma_query or "").strip() or message if scope == "chroma" else None
    )

    _PENDING[conv] = {
        "llm_messages": llm_messages,
        "sources": sources,
        "banner": banner,
        "truncation_notes": truncation_notes,
        "chroma_query": effective_chroma_query,
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
            "assistant_index": assistant_index,
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
        chroma_query = pending.get("chroma_query")
    else:
        # Defensive fallback (e.g. a stream reopened without a fresh send):
        # no retrieval context, just the raw transcript.
        llm_messages = [
            {"role": m["role"], "content": m["content"]} for m in conversation["messages"]
        ]
        sources = None
        truncation_notes = None
        chroma_query = None

    # Gap A (08-14, T-08-GAP-23): the append below MUST run on every exit
    # path -- normal completion, an inline error frame, an unhandled
    # exception, AND client abandonment (a closed StreamingResponse
    # generator raises GeneratorExit inside this loop). A try/finally
    # around the accumulation loop is what makes the append unconditional
    # by construction rather than reachable only on the happy path; the
    # missing-conversation early return above stays OUTSIDE this block so
    # it remains a no-op instead of persisting to a deleted conversation.
    #
    # Decision (answering 08-11's deferred "should a failed/ungrounded turn
    # be persisted at all?"): persist an ERROR-MARKED assistant entry
    # rather than rolling back the user message. Rollback is rejected
    # because (1) it isn't implementable for the abandonment case -- the
    # user message was saved in a different, already-returned request, and
    # a closed generator cannot reliably undo it; (2) an error-marked entry
    # is honest history -- the user asked, the system failed, and silently
    # deleting their question hides that; (3) the poisoning risk that
    # motivated the question is about the model's own refusal PROSE being
    # pattern-continued, not a short bracketed error marker, and 08-11's
    # context-adjacency fix already places grounding next to the live
    # question. A future reader should not reopen this without re-reading
    # this reasoning.
    accumulated: list = []
    error_text = None
    try:
        for frame in _stream_ollama_chat(llm_messages, model, timeout=timeout):
            # WR-01: the fixed-offset slice assumes every frame is exactly
            # ``data: {json}\n\n``. A frame that ever drifts from that shape
            # (a keepalive/comment line, a differently terminated frame) must
            # not crash the whole stream -- this loop is wrapped in only a
            # try/finally, so an unguarded json.loads here would abort the
            # generator mid-turn. Pass an unparseable frame through untouched
            # and skip accumulation/terminal detection for it instead.
            try:
                payload = json.loads(frame[len("data: "):-2])
            except (ValueError, IndexError):
                yield frame
                continue
            if "token" in payload:
                accumulated.append(payload["token"])
            if "error" in payload:
                error_text = payload["error"]
            yield frame
            if payload.get("done") or "error" in payload:
                break
    finally:
        conversation = chat_store.load(conv_id)
        if conversation is not None:
            if accumulated:
                assistant_msg = {"role": "assistant", "content": "".join(accumulated)}
            else:
                # No tokens accumulated (an error frame, a done-with-nothing
                # stream, or an abandonment before any token arrived): an
                # honest error marker, never a bare/missing assistant turn.
                # Bracketed prefix matches the project's established
                # error-string convention (CONVENTIONS.md; see
                # chat_stream's existing "[chat error: conversation not
                # found]"). ``error_text`` (when captured) is folded in
                # rather than discarded -- it is the actual diagnostic.
                content = (
                    f"[chat error: {error_text}]"
                    if error_text
                    else "[chat error: no response received]"
                )
                assistant_msg = {
                    "role": "assistant",
                    "content": content,
                    "error": True,
                }
            # Gap B (08-14 Task 2): every assistant entry -- including this
            # error-marked one -- records the model that produced (or
            # failed to produce) it, set unconditionally (unlike sources).
            # Knowing which model failed is exactly the diagnostic value
            # the UAT lost when the model was only tracked per-conversation.
            assistant_msg["model"] = model
            # D-12 (08.1-03): recorded unconditionally, like "model" above --
            # a new top-level key, NOT folded into `sources` (a list of
            # per-excerpt dicts). None/absent for every non-chroma turn.
            assistant_msg["chroma_query"] = chroma_query if chroma_query else None
            if sources:
                assistant_msg["sources"] = sources
            if truncation_notes:
                assistant_msg["truncation_notes"] = truncation_notes
            conversation["messages"].append(assistant_msg)
            # The picker's current default only (see chat_send's identical
            # comment) -- not a per-reply claim; that lives on the message.
            conversation["model"] = model
            chat_store.save(conversation)

    # Race-free saved-frame (Task 2, T-08-GAP-33): the client receives
    # `done` (yielded inside the loop above, BEFORE this generator reaches
    # this line) while the persist above is still in flight -- re-rendering
    # on `done` would race that persist and could render a history missing
    # the very reply that just streamed. This trailing yield sits AFTER the
    # try/finally above, so by construction it is only reached once the
    # persist has completed, on the NORMAL completion path -- a closed
    # generator (client abandonment, GeneratorExit thrown at the yield
    # inside the loop) unwinds straight through `finally` and out of the
    # function without ever reaching this line, so an abandoned stream
    # never emits it (Test M). This ordering is the whole point and will
    # look like redundancy to a future reader; it is not -- do not move
    # this yield inside the try/finally or hang it off `done`.
    yield f'data: {json.dumps({"saved": True})}\n\n'


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
