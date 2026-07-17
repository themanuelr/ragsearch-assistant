"""Tests for gui/ollama_chat.py (Phase 8 Plan 07, Task 1): the NDJSON->SSE
streaming bridge (D-13) and model listing (D-17). Also covers the Chat page
routes in gui/routes/chat.py (Task 2: model picker/blocked state/persistence,
Task 3: retrieval scopes) per the plan's <action> instructions.

Mocks ``urllib.request.urlopen`` per the existing Ollama-call mocking
convention (see ``tests/test_ingest.py``'s ``_FakeHttpResponse``) — no live
Ollama server is contacted. The streaming variant needs a fake response that
is *iterable* line-by-line (mirroring ``http.client.HTTPResponse``'s
iteration behavior), which is why this file defines its own fake rather than
reusing ``test_ingest.py``'s read()-only fake.

Run with:  python -m pytest tests/test_gui_ollama_chat.py -x
"""

import json
import pathlib
import re
import urllib.error
from unittest import mock

import pytest

import gui.chat_store as chat_store_module
import gui.jobs as gui_jobs_module
import gui.ollama_chat as ollama_chat
import gui.routes.chat as chat_module
from gui.ollama_chat import _stream_ollama_chat, list_models

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _tmp_chat_store(tmp_path, monkeypatch):
    """Every test in this file gets a fresh, tmp-scoped chat store dir --
    never the real ``.local/gui_chats/``."""
    monkeypatch.setattr(chat_store_module, "STORE_DIR", tmp_path / "gui_chats")
    yield


class _FakeStreamResponse:
    """Fake context-manager urlopen result, iterable line-by-line (bytes),
    mirroring ``http.client.HTTPResponse``'s NDJSON streaming shape."""

    def __init__(self, lines: list):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self._lines)


class _FakeTagsResponse:
    """Fake context-manager urlopen result for a one-shot GET (read())."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


def _messages():
    return [{"role": "user", "content": "hi"}]


def _payload(frame: str) -> dict:
    assert frame.startswith("data: "), f"frame missing 'data: ' prefix: {frame!r}"
    assert frame.endswith("\n\n"), f"frame missing double-newline suffix: {frame!r}"
    return json.loads(frame[len("data: "):-2])


# ---------------------------------------------------------------------------
# Test 1: NDJSON tokens -> SSE frames, terminated by a done frame
# ---------------------------------------------------------------------------

def test_ndjson_to_sse_frames_tokens_and_done():
    lines = [
        json.dumps({"message": {"content": "Hel"}, "done": False}).encode(),
        json.dumps({"message": {"content": "lo"}, "done": False}).encode(),
        json.dumps({"done": True}).encode(),
    ]
    with mock.patch(
        "gui.ollama_chat.urllib.request.urlopen",
        return_value=_FakeStreamResponse(lines),
    ):
        frames = list(_stream_ollama_chat(_messages(), "gemma4:e4b"))

    assert len(frames) == 3
    assert _payload(frames[0]) == {"token": "Hel"}
    assert _payload(frames[1]) == {"token": "lo"}
    assert _payload(frames[2]) == {"done": True}


# ---------------------------------------------------------------------------
# Test 2: blank lines skipped; malformed JSON line -> inline error, continues
# ---------------------------------------------------------------------------

def test_blank_lines_skipped_and_malformed_line_yields_inline_error_and_continues():
    lines = [
        b"",
        b"   ",
        b"not-valid-json{{{",
        json.dumps({"message": {"content": "ok"}, "done": False}).encode(),
        json.dumps({"done": True}).encode(),
    ]
    with mock.patch(
        "gui.ollama_chat.urllib.request.urlopen",
        return_value=_FakeStreamResponse(lines),
    ):
        frames = list(_stream_ollama_chat(_messages(), "gemma4:e4b"))

    # Blank lines produced nothing; malformed line produced one non-fatal
    # inline error event; the stream continued to the token + done frames.
    assert len(frames) == 3
    error_payload = _payload(frames[0])
    assert "error" in error_payload
    assert error_payload["error"].startswith("[Ollama error"), error_payload
    assert _payload(frames[1]) == {"token": "ok"}
    assert _payload(frames[2]) == {"done": True}


# ---------------------------------------------------------------------------
# Test 3: connection-level failures -> single terminal error event
# ---------------------------------------------------------------------------

def test_url_error_yields_single_terminal_error_event():
    with mock.patch(
        "gui.ollama_chat.urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        frames = list(_stream_ollama_chat(_messages(), "gemma4:e4b"))

    assert len(frames) == 1
    payload = _payload(frames[0])
    assert payload["error"].startswith("[Ollama error"), payload


def test_timeout_yields_single_terminal_error_event_with_timeout_prefix():
    with mock.patch(
        "gui.ollama_chat.urllib.request.urlopen",
        side_effect=TimeoutError("timed out"),
    ):
        frames = list(_stream_ollama_chat(_messages(), "gemma4:e4b"))

    assert len(frames) == 1
    payload = _payload(frames[0])
    assert payload["error"].startswith("[Ollama timeout"), payload


# ---------------------------------------------------------------------------
# Test 4: list_models
# ---------------------------------------------------------------------------

def test_list_models_parses_tags_into_name_strings():
    body = json.dumps(
        {"models": [{"name": "gemma4:e4b"}, {"name": "llama3:8b"}]}
    ).encode()
    with mock.patch(
        "gui.ollama_chat.urllib.request.urlopen",
        return_value=_FakeTagsResponse(body),
    ):
        names, error = list_models()

    assert names == ["gemma4:e4b", "llama3:8b"]
    assert error is None


def test_list_models_unreachable_returns_empty_list_and_error_never_raises():
    with mock.patch(
        "gui.ollama_chat.urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        names, error = list_models()

    assert names == []
    assert error is not None
    assert error.startswith("[Ollama error"), error


def test_list_models_timeout_returns_empty_list_and_timeout_error():
    with mock.patch(
        "gui.ollama_chat.urllib.request.urlopen",
        side_effect=TimeoutError("timed out"),
    ):
        names, error = list_models()

    assert names == []
    assert error is not None
    assert error.startswith("[Ollama timeout"), error


# ---------------------------------------------------------------------------
# Acceptance criteria: no new HTTP client dependency
# ---------------------------------------------------------------------------

def test_no_new_http_client_dependency():
    src = (_REPO_ROOT / "gui" / "ollama_chat.py").read_text(encoding="utf-8")
    assert "import urllib.request" in src
    assert "import httpx" not in src
    assert "import requests" not in src


def test_module_defines_ollama_base_constant():
    assert ollama_chat.OLLAMA_BASE == "http://localhost:11434"


# ---------------------------------------------------------------------------
# gui/routes/chat.py -- Task 2: model picker, D-16 blocked state, persistence
# ---------------------------------------------------------------------------

def test_chat_page_enabled_when_not_busy(gui_client, gui_config, monkeypatch):
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr("gui.routes.chat.list_models", lambda: (["gemma4:e4b"], None))
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)

    resp = gui_client.get("/chat")

    assert resp.status_code == 200
    assert 'hx-post="/chat/send"' in resp.text
    assert "Chat is paused while a pipeline job runs" not in resp.text
    assert "Start a conversation" in resp.text


def test_chat_page_disabled_and_blocked_copy_when_busy(gui_client, gui_config, monkeypatch):
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr("gui.routes.chat.list_models", lambda: (["gemma4:e4b"], None))
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: True)
    monkeypatch.setattr(gui_jobs_module, "list_jobs", lambda: [])

    resp = gui_client.get("/chat")

    assert resp.status_code == 200
    assert "Chat is paused while a pipeline job runs" in resp.text
    assert 'hx-post="/chat/send"' not in resp.text
    assert "<textarea" in resp.text and "disabled" in resp.text


def test_chat_page_model_dropdown_lists_tags_and_preselects_config_model(
    gui_client, gui_config, monkeypatch
):
    gui_config["model_name"] = "gemma4:e4b"
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr(
        "gui.routes.chat.list_models",
        lambda: (["gemma4:e4b", "llama3:8b"], None),
    )

    resp = gui_client.get("/chat")

    assert resp.status_code == 200
    assert 'value="gemma4:e4b" selected' in resp.text
    assert "llama3:8b" in resp.text


# ---------------------------------------------------------------------------
# gui/templates/chat.html -- 08-15 Task 2 (Gap B, D-15/T-08-GAP-30): the
# note-selector block is visible only under Paper Vault scope. The reveal
# behaviour itself is browser-only (no headless browser in this stack), so
# these tests pin the wiring and the default-hidden state on GET /chat --
# what can regress silently without a browser -- not the click-driven
# toggle itself.
# ---------------------------------------------------------------------------

def test_chat_page_note_selector_hidden_by_default_with_scope_wiring(
    gui_client, gui_config, monkeypatch
):
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr("gui.routes.chat.list_models", lambda: (["gemma4:e4b"], None))
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr("gui.routes.chat.scan_project_papers", lambda config: [])

    resp = gui_client.get("/chat")

    assert resp.status_code == 200
    # Default scope is "none" -- the block must render hidden.
    assert re.search(
        r'<div class="chat-vault-picker" id="chat-vault-picker" hidden>', resp.text
    ), resp.text
    # The block's id and the scope radios must both be present so a change
    # handler wired to it can find both ends.
    assert 'id="chat-vault-picker"' in resp.text
    assert 'name="scope" value="vault"' in resp.text
    assert 'name="notes"' in resp.text
    assert 'id="chat-vault-notes"' in resp.text
    # A toggle handler must be present, keyed off the picker's id.
    assert 'getElementById("chat-vault-picker")' in resp.text


def test_chat_page_note_selector_renders_one_option_per_scanned_paper(
    gui_client, gui_config, monkeypatch
):
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr("gui.routes.chat.list_models", lambda: (["gemma4:e4b"], None))
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    fake_papers = [
        {"title": "Paper A", "cache_paths": {"vault_note": "Papers/Paper A.md"}},
        {"title": "Paper B", "cache_paths": {"vault_note": "Papers/Paper B.md"}},
    ]
    monkeypatch.setattr("gui.routes.chat.scan_project_papers", lambda config: fake_papers)

    resp = gui_client.get("/chat")

    assert resp.status_code == 200
    assert '<option value="Papers/Paper A.md">Paper A</option>' in resp.text
    assert '<option value="Papers/Paper B.md">Paper B</option>' in resp.text


def test_chat_send_appends_user_message_and_returns_stream_markup(
    gui_client, gui_config, monkeypatch
):
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)

    conv = chat_store_module.create_conversation()

    resp = gui_client.post(
        "/chat/send",
        data={
            "conv": conv["id"],
            "message": "Hello there",
            "model": "llama3:8b",
            "scope": "none",
        },
    )

    assert resp.status_code == 200
    assert "Hello there" in resp.text
    assert "/chat/stream" in resp.text

    saved = chat_store_module.load(conv["id"])
    assert saved["messages"][0] == {"role": "user", "content": "Hello there"}
    assert saved["model"] == "llama3:8b"


# ---------------------------------------------------------------------------
# gui/routes/chat.py + chat_pending.html -- 08-15 Task 1 (Gap A,
# T-08-GAP-27): the pending assistant container id is keyed per MESSAGE, not
# per conversation, so a second send in the same conversation cannot
# duplicate the first send's id and hijack getElementById.
# ---------------------------------------------------------------------------

_CHAT_ASSISTANT_CONTENT_ID_RE = re.compile(
    r'id="(chat-assistant-content-[^"]+)"'
)
_CHAT_ASSISTANT_TARGET_ID_RE = re.compile(
    r'getElementById\("(chat-assistant-content-[^"]+)"\)'
)
_VALID_DOM_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _send_with_canned_scope(gui_client, gui_config, monkeypatch, conv_id, message):
    """Shared harness for the Task 1 tests: not-busy, and _resolve_scope
    monkeypatched to a canned no-context tuple (per the plan's <action>) so
    these tests pin only the pending-partial markup contract, not retrieval
    behaviour already covered elsewhere in this file."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr(
        "gui.routes.chat._resolve_scope",
        lambda *a, **k: ([], None, None, None, False),
    )
    return gui_client.post(
        "/chat/send",
        data={
            "conv": conv_id,
            "message": message,
            "model": "gemma4:e4b",
            "scope": "none",
        },
    )


def test_chat_send_two_consecutive_sends_yield_distinct_assistant_container_ids(
    gui_client, gui_config, monkeypatch
):
    """Test A: the gap's named regression -- two consecutive POSTs to
    /chat/send in the SAME conversation must return pending partials whose
    assistant container ids differ, so #chat-messages (hx-swap="beforeend")
    can never end up holding two elements sharing one id."""
    conv = chat_store_module.create_conversation()

    resp1 = _send_with_canned_scope(gui_client, gui_config, monkeypatch, conv["id"], "Question 1")
    resp2 = _send_with_canned_scope(gui_client, gui_config, monkeypatch, conv["id"], "Question 2")

    assert resp1.status_code == 200
    assert resp2.status_code == 200

    id1 = _CHAT_ASSISTANT_CONTENT_ID_RE.search(resp1.text).group(1)
    id2 = _CHAT_ASSISTANT_CONTENT_ID_RE.search(resp2.text).group(1)
    assert id1 != id2, "duplicate assistant container id across two sends (T-08-GAP-27)"


def test_chat_pending_script_target_id_matches_rendered_container_id(
    gui_client, gui_config, monkeypatch
):
    """Test B: within one pending partial, the id the inline script resolves
    via getElementById must be exactly the id rendered on the assistant
    content element -- target and container must correspond, the same class
    of bug 08-08 closed for the drill-down re-run buttons."""
    conv = chat_store_module.create_conversation()

    resp = _send_with_canned_scope(gui_client, gui_config, monkeypatch, conv["id"], "hi")

    rendered_id = _CHAT_ASSISTANT_CONTENT_ID_RE.search(resp.text).group(1)
    target_id = _CHAT_ASSISTANT_TARGET_ID_RE.search(resp.text).group(1)
    assert rendered_id == target_id


def test_chat_pending_assistant_container_id_is_a_valid_dom_id(
    gui_client, gui_config, monkeypatch
):
    """Test C: the composed id (conv id + message index) is a syntactically
    valid, escape-free DOM id -- conversation ids are already validated
    against a strict charset (chat_store._CONV_ID_RE) and the index is a
    plain integer, so the composed id cannot carry an attribute- or
    selector-breaking character (T-08-GAP-28, accepted-not-mitigated)."""
    conv = chat_store_module.create_conversation()

    resp = _send_with_canned_scope(gui_client, gui_config, monkeypatch, conv["id"], "hi")

    rendered_id = _CHAT_ASSISTANT_CONTENT_ID_RE.search(resp.text).group(1)
    assert _VALID_DOM_ID_RE.match(rendered_id), rendered_id


def test_chat_pending_partial_other_contracts_unchanged(
    gui_client, gui_config, monkeypatch
):
    """Test D: the pending partial still renders the user message, the
    optional banner/truncation notes, and still opens the SSE stream for the
    right conversation and model -- this plan changes WHICH element is
    resolved, nothing else about the partial's contract."""
    conv = chat_store_module.create_conversation()
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr(
        "gui.routes.chat._resolve_scope",
        lambda *a, **k: ([], None, "a banner", ["dropped note"], False),
    )

    resp = gui_client.post(
        "/chat/send",
        data={
            "conv": conv["id"],
            "message": "Hello there",
            "model": "gemma4:e4b",
            "scope": "chroma",
        },
    )

    assert resp.status_code == 200
    assert "Hello there" in resp.text
    assert "a banner" in resp.text
    assert "dropped note" in resp.text
    assert f"/chat/stream?conv={conv['id']}" in resp.text
    assert "model=gemma4" in resp.text


def test_chat_blocked_while_busy(gui_client, gui_config, monkeypatch):
    """D-16 prohibition: while busy, POST /chat/send returns the blocked
    partial without attempting any retrieval/Ollama call, and GET /chat
    renders the disabled input with the D-16 copy."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: True)
    monkeypatch.setattr(gui_jobs_module, "list_jobs", lambda: [])

    conv = chat_store_module.create_conversation()

    search_calls = []
    monkeypatch.setattr(
        "gui.routes.chat._resolve_scope",
        lambda *a, **k: (search_calls.append(1) or ([], None, None, None, False)),
    )
    stream_calls = []
    monkeypatch.setattr(
        "gui.routes.chat._stream_ollama_chat",
        lambda *a, **k: (stream_calls.append(1) or iter([])),
    )

    resp = gui_client.post(
        "/chat/send",
        data={"conv": conv["id"], "message": "hi", "model": "gemma4:e4b", "scope": "none"},
    )

    assert resp.status_code == 200
    assert "Chat is paused while a pipeline job runs" in resp.text
    assert search_calls == [], "no retrieval scope should be resolved while busy"
    assert stream_calls == [], "no Ollama call should be attempted while busy"

    get_resp = gui_client.get("/chat")
    assert "Chat is paused while a pipeline job runs" in get_resp.text
    assert 'hx-post="/chat/send"' not in get_resp.text


# ---------------------------------------------------------------------------
# gui/routes/chat.py -- Task 3: retrieval scopes (D-14/D-15)
# ---------------------------------------------------------------------------

def _fake_search_result():
    return {
        "papers": [
            {
                "title": "A Great Paper",
                "registry_key": "10.1/x",
                "status": "paper",
                "vault_note": "Papers/A Great Paper.md",
                "score": 0.9,
                "sections": [
                    {"heading": "Results", "score": 0.9, "excerpt": "some excerpt text"},
                ],
            }
        ],
        "stubs": [],
        "error": None,
    }


def test_chat_sources_block_rendered_for_chroma_scope(gui_client, gui_config, monkeypatch):
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr("gui.routes.chat.list_models", lambda: (["gemma4:e4b"], None))

    conv = chat_store_module.create_conversation()

    search_calls = []

    def fake_search(query, n_results, config):
        search_calls.append((query, n_results))
        return _fake_search_result()

    monkeypatch.setattr("gui.routes.chat._search", fake_search)

    lines = [
        json.dumps({"message": {"content": "Answer text"}, "done": False}).encode(),
        json.dumps({"done": True}).encode(),
    ]
    monkeypatch.setattr(
        "gui.ollama_chat.urllib.request.urlopen",
        lambda *a, **k: _FakeStreamResponse(lines),
    )

    send_resp = gui_client.post(
        "/chat/send",
        data={
            "conv": conv["id"],
            "message": "what does the paper say",
            "model": "gemma4:e4b",
            "scope": "chroma",
        },
    )
    assert send_resp.status_code == 200
    assert len(search_calls) == 1, "chroma scope must call _search exactly once"

    stream_resp = gui_client.get(f"/chat/stream?conv={conv['id']}&model=gemma4:e4b")
    assert stream_resp.status_code == 200

    saved = chat_store_module.load(conv["id"])
    assistant_msg = saved["messages"][-1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == "Answer text"
    assert assistant_msg["sources"] == [
        {"title": "A Great Paper", "section": "Results", "score": 0.9}
    ]

    page_resp = gui_client.get(f"/chat?conv={conv['id']}")
    assert "Sources (1)" in page_resp.text
    assert "A Great Paper" in page_resp.text
    assert "Results" in page_resp.text


def test_chat_chroma_scope_error_falls_back_to_none_with_banner(gui_client, gui_config, monkeypatch):
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)

    conv = chat_store_module.create_conversation()

    monkeypatch.setattr(
        "gui.routes.chat._search",
        lambda *a, **k: {"papers": [], "stubs": [], "error": "[Ollama error: embedding unreachable]"},
    )

    resp = gui_client.post(
        "/chat/send",
        data={"conv": conv["id"], "message": "hello", "model": "gemma4:e4b", "scope": "chroma"},
    )

    assert resp.status_code == 200
    assert "[Ollama error: embedding unreachable]" in resp.text

    pending = chat_module._PENDING[conv["id"]]
    assert pending["llm_messages"] == [{"role": "user", "content": "hello"}]
    assert pending["sources"] is None


# ---------------------------------------------------------------------------
# gui/routes/chat.py -- 08-11 Task 2 (Gap B): chroma excerpt headers carry
# authors/year/journal read from the registry.
# ---------------------------------------------------------------------------

def _write_registry(gui_config, registry: dict):
    pathlib.Path(gui_config["registry_path"]).write_text(
        json.dumps(registry), encoding="utf-8"
    )


def test_chat_chroma_context_carries_authors_year_journal(gui_client, gui_config, monkeypatch):
    """Test D: the assembled chroma context contains the paper's authors,
    year and journal values, so a metadata question is answerable from
    context (08-UAT.md Gap B)."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)

    _write_registry(gui_config, {
        "10.1/x": {
            "title": "A Great Paper",
            "authors": ["Jane Doe", "John Smith"],
            "year": 2023,
            "journal": "Nature",
        }
    })
    monkeypatch.setattr("gui.routes.chat._search", lambda *a, **k: _fake_search_result())

    conv = chat_store_module.create_conversation()

    resp = gui_client.post(
        "/chat/send",
        data={
            "conv": conv["id"],
            "message": "who is the first author",
            "model": "gemma4:e4b",
            "scope": "chroma",
        },
    )
    assert resp.status_code == 200

    context_content = chat_module._PENDING[conv["id"]]["llm_messages"][0]["content"]
    assert "Jane Doe, John Smith" in context_content
    assert "2023" in context_content
    assert "Nature" in context_content


def test_chat_chroma_context_missing_metadata_degrades_cleanly(gui_client, gui_config, monkeypatch):
    """Test E: an unknown registry_key, or an entry whose authors/year/
    journal are None, still produces a valid excerpt block (title + heading
    + excerpt) and raises nothing -- a paper with no recorded metadata must
    never break retrieval. No empty or "None" label is ever emitted."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)

    # Registry file exists but has no entry for "10.1/x" at all -- and, as a
    # second scenario, an entry whose fields are explicitly None.
    _write_registry(gui_config, {
        "10.1/x": {"title": "A Great Paper", "authors": None, "year": None, "journal": None},
    })
    monkeypatch.setattr("gui.routes.chat._search", lambda *a, **k: _fake_search_result())

    conv = chat_store_module.create_conversation()

    resp = gui_client.post(
        "/chat/send",
        data={
            "conv": conv["id"],
            "message": "what does the paper say",
            "model": "gemma4:e4b",
            "scope": "chroma",
        },
    )
    assert resp.status_code == 200

    context_content = chat_module._PENDING[conv["id"]]["llm_messages"][0]["content"]
    assert "[A Great Paper — Results]" in context_content
    assert "some excerpt text" in context_content
    assert "None" not in context_content


def test_chat_chroma_missing_registry_key_degrades_cleanly(gui_client, gui_config, monkeypatch):
    """Test E (continued): a registry_key absent from the registry entirely
    (no registry file even written) must not raise."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr("gui.routes.chat._search", lambda *a, **k: _fake_search_result())

    conv = chat_store_module.create_conversation()

    resp = gui_client.post(
        "/chat/send",
        data={
            "conv": conv["id"],
            "message": "what does the paper say",
            "model": "gemma4:e4b",
            "scope": "chroma",
        },
    )
    assert resp.status_code == 200
    context_content = chat_module._PENDING[conv["id"]]["llm_messages"][0]["content"]
    assert "[A Great Paper — Results]" in context_content


def test_chat_chroma_sources_shape_unchanged_by_metadata_enrichment(gui_client, gui_config, monkeypatch):
    """Test F: the sources rows still carry exactly title/section/score --
    the Sources block the user expands is not restructured by this change."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)

    _write_registry(gui_config, {
        "10.1/x": {"title": "A Great Paper", "authors": ["Jane Doe"], "year": 2023, "journal": "Nature"},
    })
    monkeypatch.setattr("gui.routes.chat._search", lambda *a, **k: _fake_search_result())

    conv = chat_store_module.create_conversation()

    resp = gui_client.post(
        "/chat/send",
        data={
            "conv": conv["id"],
            "message": "what does the paper say",
            "model": "gemma4:e4b",
            "scope": "chroma",
        },
    )
    assert resp.status_code == 200

    sources = chat_module._PENDING[conv["id"]]["sources"]
    assert sources == [{"title": "A Great Paper", "section": "Results", "score": 0.9}]


def test_chat_vault_scope_truncates_over_cap_and_flags_warning(
    gui_client, gui_config, monkeypatch
):
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)

    vault_path = pathlib.Path(gui_config["vault_path"])
    (vault_path / "Papers").mkdir(parents=True, exist_ok=True)
    (vault_path / "Papers" / "Paper A.md").write_text("A" * 20000, encoding="utf-8")
    (vault_path / "Papers" / "Paper B.md").write_text("B" * 20000, encoding="utf-8")

    fake_papers = [
        {"title": "Paper A", "cache_paths": {"vault_note": "Papers/Paper A.md"}},
        {"title": "Paper B", "cache_paths": {"vault_note": "Papers/Paper B.md"}},
    ]
    monkeypatch.setattr("gui.routes.chat.scan_project_papers", lambda config: fake_papers)
    gui_config["ollama_num_ctx_cap"] = 2048  # tiny cap: forces a drop

    conv = chat_store_module.create_conversation()

    resp = gui_client.post(
        "/chat/send",
        data={
            "conv": conv["id"],
            "message": "summarize",
            "model": "gemma4:e4b",
            "scope": "vault",
            "notes": ["Papers/Paper A.md", "Papers/Paper B.md"],
        },
    )

    assert resp.status_code == 200
    assert "Context truncated" in resp.text
    assert "Paper B" in resp.text  # the dropped note is named in the warning

    pending = chat_module._PENDING[conv["id"]]
    combined_context = pending["llm_messages"][0]["content"]
    assert "Paper A" in combined_context
    assert "B" * 100 not in combined_context  # dropped note's content excluded


# ---------------------------------------------------------------------------
# gui/routes/chat.py -- 08-11 Task 3 (Gap C / T-08-GAP-13): Paper Vault
# scope with nothing resolvable is a hard block, not a silent ungrounded turn.
# ---------------------------------------------------------------------------

def _fail_if_called(*_a, **_k):
    raise AssertionError("blocked turn must not reach _stream_ollama_chat")


def test_chat_vault_empty_selection_blocks_with_banner_and_no_ollama_call(
    gui_client, gui_config, monkeypatch
):
    """Test G: scope='vault' with no notes submitted returns the banner
    partial, directs the user to select at least one paper, and makes no
    Ollama call."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr("gui.routes.chat._stream_ollama_chat", _fail_if_called)

    conv = chat_store_module.create_conversation()

    resp = gui_client.post(
        "/chat/send",
        data={"conv": conv["id"], "message": "summarize", "model": "gemma4:e4b", "scope": "vault"},
    )

    assert resp.status_code == 200
    assert "select at least one paper" in resp.text.lower()
    assert conv["id"] not in chat_module._PENDING


def test_chat_vault_empty_selection_persists_nothing(gui_client, gui_config, monkeypatch):
    """Test H: a blocked turn leaves the stored conversation's message list
    unchanged -- no orphaned user message -- and writes no _PENDING entry."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)

    conv = chat_store_module.create_conversation()
    assert conv["messages"] == []

    gui_client.post(
        "/chat/send",
        data={"conv": conv["id"], "message": "summarize", "model": "gemma4:e4b", "scope": "vault"},
    )

    saved = chat_store_module.load(conv["id"])
    assert saved["messages"] == []
    assert conv["id"] not in chat_module._PENDING


def test_chat_vault_unresolvable_selection_blocks_identically(gui_client, gui_config, monkeypatch):
    """Test I: a notes value that does not match any scanned paper's
    vault_note path is treated identically to an empty selection (blocked +
    banner) -- server-side validation drops it and the turn would otherwise
    be silently ungrounded."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr("gui.routes.chat.scan_project_papers", lambda config: [])
    monkeypatch.setattr("gui.routes.chat._stream_ollama_chat", _fail_if_called)

    conv = chat_store_module.create_conversation()

    resp = gui_client.post(
        "/chat/send",
        data={
            "conv": conv["id"],
            "message": "summarize",
            "model": "gemma4:e4b",
            "scope": "vault",
            "notes": ["Papers/Nonexistent.md"],
        },
    )

    assert resp.status_code == 200
    assert "select at least one paper" in resp.text.lower()
    assert conv["id"] not in chat_module._PENDING


def test_chat_vault_valid_selection_unaffected_by_guard(gui_client, gui_config, monkeypatch):
    """Test J: a resolvable vault selection still assembles context, still
    returns the pending partial, and still streams -- the guard fires only
    when the scope would otherwise be empty."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)

    vault_path = pathlib.Path(gui_config["vault_path"])
    (vault_path / "Papers").mkdir(parents=True, exist_ok=True)
    (vault_path / "Papers" / "Paper A.md").write_text("Paper A content", encoding="utf-8")

    fake_papers = [
        {"title": "Paper A", "cache_paths": {"vault_note": "Papers/Paper A.md"}},
    ]
    monkeypatch.setattr("gui.routes.chat.scan_project_papers", lambda config: fake_papers)

    conv = chat_store_module.create_conversation()

    resp = gui_client.post(
        "/chat/send",
        data={
            "conv": conv["id"],
            "message": "summarize",
            "model": "gemma4:e4b",
            "scope": "vault",
            "notes": ["Papers/Paper A.md"],
        },
    )

    assert resp.status_code == 200
    assert "/chat/stream" in resp.text
    pending = chat_module._PENDING[conv["id"]]
    assert "Paper A content" in pending["llm_messages"][0]["content"]


def test_chat_none_and_chroma_scopes_never_block(gui_client, gui_config, monkeypatch):
    """Test K: scope='none' and scope='chroma' never take the blocked path;
    the D-14 chroma fail-open still returns its banner AND still lets the
    model answer (fail-open is deliberately different from the vault hard
    block)."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)

    conv_none = chat_store_module.create_conversation()
    resp_none = gui_client.post(
        "/chat/send",
        data={"conv": conv_none["id"], "message": "hi", "model": "gemma4:e4b", "scope": "none"},
    )
    assert resp_none.status_code == 200
    assert conv_none["id"] in chat_module._PENDING

    monkeypatch.setattr(
        "gui.routes.chat._search",
        lambda *a, **k: {"papers": [], "stubs": [], "error": "[Ollama error: embedding unreachable]"},
    )
    conv_chroma = chat_store_module.create_conversation()
    resp_chroma = gui_client.post(
        "/chat/send",
        data={"conv": conv_chroma["id"], "message": "hi", "model": "gemma4:e4b", "scope": "chroma"},
    )
    assert resp_chroma.status_code == 200
    assert "[Ollama error: embedding unreachable]" in resp_chroma.text
    # D-14 fail-open: still a normal pending turn (still lets the model
    # answer), NOT the blocked banner partial.
    assert conv_chroma["id"] in chat_module._PENDING


def test_chat_none_scope_calls_no_retrieval(gui_client, gui_config, monkeypatch):
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)

    search_calls = []
    monkeypatch.setattr(
        "gui.routes.chat._search", lambda *a, **k: search_calls.append(1)
    )

    conv = chat_store_module.create_conversation()

    resp = gui_client.post(
        "/chat/send",
        data={"conv": conv["id"], "message": "hi", "model": "gemma4:e4b", "scope": "none"},
    )

    assert resp.status_code == 200
    assert search_calls == []


# ---------------------------------------------------------------------------
# gui/routes/chat.py -- 08-11 Task 1 (Gap A / T-08-GAP-10): context spliced
# adjacent to the latest user message, not at position 0.
# ---------------------------------------------------------------------------

def _seed_conversation_with_refusal():
    """Conversation whose history already contains a prior user turn AND a
    prior assistant refusal -- the exact shape from Gap A's diagnosis
    (08-UAT.md T-08-GAP-10): a ChromaDB-error window produced a refusal that
    then re-enters the next turn's prompt and gets pattern-continued."""
    conv = chat_store_module.create_conversation()
    conv["messages"] = [
        {"role": "user", "content": "who wrote the NKCC1 paper"},
        {
            "role": "assistant",
            "content": "I'm sorry for any confusion earlier; I must clarify that "
            "as of my last update at beginning 2023-4, I'm not capable to browse "
            "real-time databases",
        },
    ]
    chat_store_module.save(conv)
    return conv


def test_chat_context_spliced_before_final_user_turn_not_position_zero(
    gui_client, gui_config, monkeypatch
):
    """Test A: pins the ASSEMBLY position, not whether a live model then
    answers correctly -- that is a live-LLM property no hermetic test can
    assert, and is exactly what the confirmed diagnosis (replaying the SAME
    question with the SAME context but a CLEAN transcript) identified as the
    whole difference between a correct grounded answer and a confident
    non-answer (08-UAT.md T-08-GAP-10). This is not a weaker substitute
    chosen by accident."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)

    conv = _seed_conversation_with_refusal()

    canned_context = [{"role": "system", "content": "retrieved excerpts here"}]
    monkeypatch.setattr(
        "gui.routes.chat._resolve_scope",
        lambda *a, **k: (
            canned_context,
            [{"title": "t", "section": "s", "score": 0.9}],
            None,
            None,
            False,
        ),
    )

    resp = gui_client.post(
        "/chat/send",
        data={
            "conv": conv["id"],
            "message": "who is the first author",
            "model": "gemma4:e4b",
            "scope": "chroma",
        },
    )
    assert resp.status_code == 200

    llm_messages = chat_module._PENDING[conv["id"]]["llm_messages"]
    assert llm_messages[-1] == {"role": "user", "content": "who is the first author"}
    assert llm_messages[len(llm_messages) - 2] == canned_context[0]


def test_chat_context_splice_preserves_prior_transcript_order(
    gui_client, gui_config, monkeypatch
):
    """Test B: the fix reorders nothing except where the context is spliced
    in -- every prior transcript message stays present in its original
    relative order."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)

    conv = _seed_conversation_with_refusal()

    canned_context = [{"role": "system", "content": "retrieved excerpts here"}]
    monkeypatch.setattr(
        "gui.routes.chat._resolve_scope",
        lambda *a, **k: (canned_context, None, None, None, False),
    )

    resp = gui_client.post(
        "/chat/send",
        data={
            "conv": conv["id"],
            "message": "fresh question",
            "model": "gemma4:e4b",
            "scope": "chroma",
        },
    )
    assert resp.status_code == 200

    llm_messages = chat_module._PENDING[conv["id"]]["llm_messages"]
    assert llm_messages[:2] == [
        {"role": "user", "content": "who wrote the NKCC1 paper"},
        {
            "role": "assistant",
            "content": "I'm sorry for any confusion earlier; I must clarify that "
            "as of my last update at beginning 2023-4, I'm not capable to browse "
            "real-time databases",
        },
    ]


def test_chat_none_scope_context_no_splice_equals_plain_transcript(
    gui_client, gui_config, monkeypatch
):
    """Test C: an empty context_messages yields llm_messages equal to the
    plain transcript -- no injected system message, scope-none path stays
    byte-identical."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)

    conv = chat_store_module.create_conversation()

    resp = gui_client.post(
        "/chat/send",
        data={"conv": conv["id"], "message": "hi", "model": "gemma4:e4b", "scope": "none"},
    )
    assert resp.status_code == 200

    llm_messages = chat_module._PENDING[conv["id"]]["llm_messages"]
    assert llm_messages == [{"role": "user", "content": "hi"}]
    assert not any(m["role"] == "system" for m in llm_messages)


# ---------------------------------------------------------------------------
# gui/routes/chat.py -- 08-14 Task 1 (Gap A / D-18): _sse_stream guarantees
# exactly one assistant entry per turn, even on error/empty/abandoned
# streams, so no turn can ever leave an orphaned user message.
# ---------------------------------------------------------------------------

def _assert_alternation(messages: list) -> None:
    """Shared invariant: strict user/assistant alternation, no two
    consecutive messages of the same role -- the actual truth Task 1
    defends. Reused across every test in this section."""
    for i in range(1, len(messages)):
        assert messages[i]["role"] != messages[i - 1]["role"], (
            f"consecutive same-role messages at indices {i - 1},{i}: {messages}"
        )


def _send_turn(gui_client, gui_config, monkeypatch, conv_id, message="hi", model="gemma4:e4b"):
    """Drive a real POST /chat/send so the user message is appended/saved
    and a _PENDING entry is written exactly like the browser flow."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    resp = gui_client.post(
        "/chat/send",
        data={"conv": conv_id, "message": message, "model": model, "scope": "none"},
    )
    assert resp.status_code == 200


def _frame(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def test_errored_stream_leaves_no_orphaned_user_message(gui_client, gui_config, monkeypatch):
    """Test A: a turn whose stream yields an error frame must not leave a
    user message without a corresponding assistant entry. The transcript
    alternates and the final message is an assistant entry carrying the
    error marker."""
    conv = chat_store_module.create_conversation()
    _send_turn(gui_client, gui_config, monkeypatch, conv["id"], message="will error")

    monkeypatch.setattr(
        "gui.routes.chat._stream_ollama_chat",
        lambda *a, **k: iter([_frame({"error": "[Ollama error: connection refused]"})]),
    )

    list(chat_module._sse_stream(conv["id"], "gemma4:e4b"))

    saved = chat_store_module.load(conv["id"])
    _assert_alternation(saved["messages"])
    last = saved["messages"][-1]
    assert last["role"] == "assistant"
    assert last.get("error") is True
    assert last["content"].startswith("[chat error"), last["content"]


def test_done_with_no_tokens_still_yields_exactly_one_error_marked_assistant_entry(
    gui_client, gui_config, monkeypatch
):
    """Test B: a stream that yields 'done' with no tokens still results in
    exactly one assistant entry (error-marked, since the model produced
    nothing) -- never a bare user message."""
    conv = chat_store_module.create_conversation()
    _send_turn(gui_client, gui_config, monkeypatch, conv["id"], message="empty reply")

    monkeypatch.setattr(
        "gui.routes.chat._stream_ollama_chat",
        lambda *a, **k: iter([_frame({"done": True})]),
    )

    list(chat_module._sse_stream(conv["id"], "gemma4:e4b"))

    saved = chat_store_module.load(conv["id"])
    _assert_alternation(saved["messages"])
    last = saved["messages"][-1]
    assert last["role"] == "assistant"
    assert last.get("error") is True
    assert last["content"].startswith("[chat error"), last["content"]


def test_abandoned_stream_yields_exactly_one_assistant_entry_with_partial_text(
    gui_client, gui_config, monkeypatch
):
    """Test C: a stream abandoned mid-flight (generator closed after a
    partial token, as when the browser disconnects) still results in
    exactly one assistant entry, carrying the partial text accumulated so
    far."""
    conv = chat_store_module.create_conversation()
    _send_turn(gui_client, gui_config, monkeypatch, conv["id"], message="partial")

    def _fake_stream(*a, **k):
        yield _frame({"token": "partial answer"})
        yield _frame({"token": "-- never reached"})

    monkeypatch.setattr("gui.routes.chat._stream_ollama_chat", _fake_stream)

    gen = chat_module._sse_stream(conv["id"], "gemma4:e4b")
    next(gen)  # consume exactly one frame, mirroring a mid-flight disconnect
    gen.close()

    saved = chat_store_module.load(conv["id"])
    _assert_alternation(saved["messages"])
    last = saved["messages"][-1]
    assert last["role"] == "assistant"
    assert last["content"] == "partial answer"


def test_successful_stream_appends_exactly_one_assistant_message_unchanged(
    gui_client, gui_config, monkeypatch
):
    """Test D: a normal completed stream still appends exactly one
    assistant message whose content is the joined tokens, with sources and
    truncation_notes attached exactly as today."""
    conv = chat_store_module.create_conversation()
    _send_turn(gui_client, gui_config, monkeypatch, conv["id"], message="normal")
    chat_module._PENDING[conv["id"]] = {
        "llm_messages": [{"role": "user", "content": "normal"}],
        "sources": [{"title": "t", "section": "s", "score": 0.9}],
        "banner": None,
        "truncation_notes": ["dropped note"],
    }

    monkeypatch.setattr(
        "gui.routes.chat._stream_ollama_chat",
        lambda *a, **k: iter([
            _frame({"token": "Hel"}),
            _frame({"token": "lo"}),
            _frame({"done": True}),
        ]),
    )

    list(chat_module._sse_stream(conv["id"], "gemma4:e4b"))

    saved = chat_store_module.load(conv["id"])
    _assert_alternation(saved["messages"])
    last = saved["messages"][-1]
    assert last["role"] == "assistant"
    assert last["content"] == "Hello"
    assert last.get("error") is not True
    assert last["sources"] == [{"title": "t", "section": "s", "score": 0.9}]
    assert last["truncation_notes"] == ["dropped note"]


@pytest.mark.parametrize(
    "frames",
    [
        [{"error": "[Ollama error: boom]"}],
        [{"done": True}],
        [{"token": "hi"}, {"done": True}],
    ],
)
def test_turn_adds_exactly_two_messages_never_more(
    gui_client, gui_config, monkeypatch, frames
):
    """Test E: no path appends two assistant messages for one turn --
    the message count delta for a turn is exactly 2 (one user, one
    assistant), across the error, empty, and success paths."""
    conv = chat_store_module.create_conversation()
    before_count = len(conv["messages"])
    _send_turn(gui_client, gui_config, monkeypatch, conv["id"], message="count me")

    monkeypatch.setattr(
        "gui.routes.chat._stream_ollama_chat",
        lambda *a, **k: iter([_frame(p) for p in frames]),
    )

    list(chat_module._sse_stream(conv["id"], "gemma4:e4b"))

    saved = chat_store_module.load(conv["id"])
    assert len(saved["messages"]) - before_count == 2


# ---------------------------------------------------------------------------
# gui/routes/chat.py -- 08-14 Task 2 (Gap B): the model is recorded per
# assistant message, not only as the picker's last-written default.
# ---------------------------------------------------------------------------

def test_two_replies_with_different_models_each_retain_their_own_model(
    gui_client, gui_config, monkeypatch
):
    """Test F: two replies produced by different models in one conversation
    each retain their own model -- the second send does not retroactively
    relabel the first."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)

    conv = chat_store_module.create_conversation()

    def _stream_for(token):
        return lambda *a, **k: iter([_frame({"token": token}), _frame({"done": True})])

    monkeypatch.setattr("gui.routes.chat._stream_ollama_chat", _stream_for("reply one"))
    resp1 = gui_client.post(
        "/chat/send",
        data={"conv": conv["id"], "message": "q1", "model": "model-x", "scope": "none"},
    )
    assert resp1.status_code == 200
    stream_resp1 = gui_client.get(f"/chat/stream?conv={conv['id']}&model=model-x")
    assert stream_resp1.status_code == 200

    monkeypatch.setattr("gui.routes.chat._stream_ollama_chat", _stream_for("reply two"))
    resp2 = gui_client.post(
        "/chat/send",
        data={"conv": conv["id"], "message": "q2", "model": "model-y", "scope": "none"},
    )
    assert resp2.status_code == 200
    stream_resp2 = gui_client.get(f"/chat/stream?conv={conv['id']}&model=model-y")
    assert stream_resp2.status_code == 200

    saved = chat_store_module.load(conv["id"])
    assistant_msgs = [m for m in saved["messages"] if m["role"] == "assistant"]
    assert len(assistant_msgs) == 2
    assert assistant_msgs[0]["model"] == "model-x"
    assert assistant_msgs[1]["model"] == "model-y"

    # Test G: conversation["model"] holds the last pick (the picker's
    # current default), while per-message values stay distinct.
    assert saved["model"] == "model-y"


def test_chat_history_renders_each_assistant_messages_own_model(
    gui_client, gui_config, monkeypatch
):
    """Test H: the chat history renders each assistant message's own
    model."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr("gui.routes.chat.list_models", lambda: (["model-x", "model-y"], None))

    conv = chat_store_module.create_conversation()

    def _stream_for(token):
        return lambda *a, **k: iter([_frame({"token": token}), _frame({"done": True})])

    monkeypatch.setattr("gui.routes.chat._stream_ollama_chat", _stream_for("reply one"))
    gui_client.post(
        "/chat/send",
        data={"conv": conv["id"], "message": "q1", "model": "model-x", "scope": "none"},
    )
    gui_client.get(f"/chat/stream?conv={conv['id']}&model=model-x")

    monkeypatch.setattr("gui.routes.chat._stream_ollama_chat", _stream_for("reply two"))
    gui_client.post(
        "/chat/send",
        data={"conv": conv["id"], "message": "q2", "model": "model-y", "scope": "none"},
    )
    gui_client.get(f"/chat/stream?conv={conv['id']}&model=model-y")

    page_resp = gui_client.get(f"/chat?conv={conv['id']}")
    assert page_resp.status_code == 200
    assert "model-x" in page_resp.text
    assert "model-y" in page_resp.text


def test_prechange_conversation_without_model_field_renders_without_error(
    gui_client, gui_config, monkeypatch
):
    """Test I: a stored conversation whose assistant messages predate this
    change (no model field) renders without error and without inventing a
    model label."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr("gui.routes.chat.list_models", lambda: (["model-x"], None))

    conv = chat_store_module.create_conversation()
    conv["messages"] = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old reply"},
    ]
    chat_store_module.save(conv)

    resp = gui_client.get(f"/chat?conv={conv['id']}")

    assert resp.status_code == 200
    assert "old reply" in resp.text


def test_recorded_model_is_the_one_chat_stream_resolved_via_fallback_chain(
    gui_client, gui_config, monkeypatch
):
    """Test J: when chat_stream resolves the model through its fallback
    chain (conversation['model']) rather than an explicit query param, the
    recorded value is the resolved one."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)

    conv = chat_store_module.create_conversation()
    _send_turn(gui_client, gui_config, monkeypatch, conv["id"], message="q", model="model-z")

    monkeypatch.setattr(
        "gui.routes.chat._stream_ollama_chat",
        lambda *a, **k: iter([_frame({"token": "ans"}), _frame({"done": True})]),
    )

    # No `model=` query param -- chat_stream must fall back to
    # conversation.get("model"), which chat_send already set to "model-z".
    resp = gui_client.get(f"/chat/stream?conv={conv['id']}")
    assert resp.status_code == 200

    saved = chat_store_module.load(conv["id"])
    assistant_msg = saved["messages"][-1]
    assert assistant_msg["model"] == "model-z"


# ---------------------------------------------------------------------------
# T-08-05 (threat register), narrowed by Phase 8 Plan 16 (GUI-09): the DOM
# inner-markup setter API (`.innerHTML = ...`, `insertAdjacentHTML(...)`)
# must never be called directly in any chat template -- tokens must reach
# the DOM via textContent/createTextNode only. This no longer forbids the
# bare word "innerHTML": chat_pending.html's saved-frame branch passes
# htmx's public `swap: "innerHTML"` CONFIG STRING to `htmx.ajax(...)`, which
# is htmx swapping a SERVER-RENDERED partial (already passed through
# gui/chat_markdown.py's chokepoint) -- not this file calling the DOM API
# itself. The regex below still catches an actual direct-API violation.
# ---------------------------------------------------------------------------

_INNERHTML_DOM_API_RE = re.compile(r"\.innerHTML\s*=|insertAdjacentHTML\s*\(")


def test_no_innerhtml_dom_api_in_chat_templates():
    chat_templates_dir = _REPO_ROOT / "gui" / "templates"
    candidates = [
        chat_templates_dir / "chat.html",
        chat_templates_dir / "partials" / "chat_pending.html",
        chat_templates_dir / "partials" / "chat_messages.html",
        chat_templates_dir / "partials" / "chat_sidebar.html",
        chat_templates_dir / "partials" / "chat_sources.html",
        chat_templates_dir / "partials" / "chat_blocked.html",
        chat_templates_dir / "partials" / "chat_scope_banner.html",
    ]
    for tpl in candidates:
        text = tpl.read_text(encoding="utf-8")
        assert not _INNERHTML_DOM_API_RE.search(text), (
            f"{tpl} must never call the DOM innerHTML/insertAdjacentHTML API directly (T-08-05)"
        )


def test_chat_pending_token_path_still_createtextnode_only():
    """The LIVE streaming token path (data.token) must remain
    createTextNode-only -- unchanged by the saved-frame's htmx re-render."""
    text = (
        _REPO_ROOT / "gui" / "templates" / "partials" / "chat_pending.html"
    ).read_text(encoding="utf-8")
    assert "document.createTextNode(data.token)" in text


# ---------------------------------------------------------------------------
# gui/routes/chat.py + gui/templates/partials/chat_messages.html -- Phase 8
# Plan 16 Task 2 (GUI-09): one shared history render, a race-free post-turn
# re-render, and the sources-without-a-refresh gap closure (Tests I-N).
# ---------------------------------------------------------------------------

def test_history_partial_and_chat_page_render_the_same_history(
    gui_client, gui_config, monkeypatch
):
    """Test I: GET /chat and the new history endpoint render the SAME
    formatted markdown for the same conversation -- one render path, so the
    page and the partial cannot drift."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr("gui.routes.chat.list_models", lambda: (["gemma4:e4b"], None))

    conv = chat_store_module.create_conversation()
    conv["messages"] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "**bold reply**", "model": "gemma4:e4b"},
    ]
    chat_store_module.save(conv)

    page_resp = gui_client.get(f"/chat?conv={conv['id']}")
    history_resp = gui_client.get(f"/chat/{conv['id']}/messages")

    assert page_resp.status_code == 200
    assert history_resp.status_code == 200
    assert "<strong>bold reply</strong>" in page_resp.text
    assert "<strong>bold reply</strong>" in history_resp.text


def test_markdown_renders_in_history_user_content_excluded(
    gui_client, gui_config, monkeypatch
):
    """Test J: an assistant message's stored markdown renders as formatted
    HTML; a user message's content is NOT passed through the renderer and
    stays literal."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr("gui.routes.chat.list_models", lambda: (["gemma4:e4b"], None))

    conv = chat_store_module.create_conversation()
    conv["messages"] = [
        {"role": "user", "content": "**not bold** please"},
        {"role": "assistant", "content": "**bold reply**", "model": "gemma4:e4b"},
    ]
    chat_store_module.save(conv)

    resp = gui_client.get(f"/chat/{conv['id']}/messages")

    assert resp.status_code == 200
    assert "<strong>bold reply</strong>" in resp.text
    assert "**not bold** please" in resp.text
    assert "<strong>not bold</strong>" not in resp.text


def test_history_endpoint_serves_valid_conversation_id(gui_client, gui_config, monkeypatch):
    """Test K (valid id): the history endpoint returns the rendered
    history for a valid conversation id."""
    conv = chat_store_module.create_conversation()
    conv["messages"] = [{"role": "user", "content": "hi there"}]
    chat_store_module.save(conv)

    resp = gui_client.get(f"/chat/{conv['id']}/messages")

    assert resp.status_code == 200
    assert "hi there" in resp.text


def test_history_endpoint_404s_for_invalid_or_absent_id_without_filesystem_exception(
    gui_client, gui_config, monkeypatch
):
    """Test K (invalid/absent id): an id failing chat_store's strict
    charset, and a well-formed but non-existent id, both 404 -- never a
    filesystem exception (mirroring chat_store's own id-validation
    contract, T-08-18)."""
    resp_invalid_charset = gui_client.get("/chat/bad_id_here/messages")
    assert resp_invalid_charset.status_code == 404

    resp_absent = gui_client.get("/chat/20260101-000000-deadbeef/messages")
    assert resp_absent.status_code == 404


def test_saved_frame_emitted_only_after_persist_completes(
    gui_client, gui_config, monkeypatch
):
    """Test L: the terminal saved-frame is emitted only after the assistant
    message is persisted -- assert the persisted state at the MOMENT the
    saved-frame is observed (consumed frame by frame), not only at the end,
    which would not distinguish the race (T-08-GAP-33)."""
    conv = chat_store_module.create_conversation()
    _send_turn(gui_client, gui_config, monkeypatch, conv["id"], message="race check")

    monkeypatch.setattr(
        "gui.routes.chat._stream_ollama_chat",
        lambda *a, **k: iter([_frame({"token": "answer"}), _frame({"done": True})]),
    )

    gen = chat_module._sse_stream(conv["id"], "gemma4:e4b")
    saved_frame_seen = False
    persisted_at_saved_frame = None
    for raw in gen:
        payload = json.loads(raw[len("data: "):-2])
        if payload.get("saved"):
            saved_frame_seen = True
            saved_conv = chat_store_module.load(conv["id"])
            persisted_at_saved_frame = saved_conv["messages"][-1]
            break

    assert saved_frame_seen, "no saved-frame was emitted on the normal completion path"
    assert persisted_at_saved_frame["role"] == "assistant"
    assert persisted_at_saved_frame["content"] == "answer"


def test_no_saved_frame_on_abandoned_stream(gui_client, gui_config, monkeypatch):
    """Test M: a stream abandoned mid-flight (generator closed after one
    partial token) emits no saved-frame, and 08-14's exactly-one-assistant-
    entry guarantee still holds."""
    conv = chat_store_module.create_conversation()
    _send_turn(gui_client, gui_config, monkeypatch, conv["id"], message="abandon me")

    def _fake_stream(*a, **k):
        yield _frame({"token": "partial"})
        yield _frame({"token": "-- never reached"})

    monkeypatch.setattr("gui.routes.chat._stream_ollama_chat", _fake_stream)

    gen = chat_module._sse_stream(conv["id"], "gemma4:e4b")
    frame = next(gen)  # consume exactly one frame, mirroring a mid-flight disconnect
    payload = json.loads(frame[len("data: "):-2])
    assert payload == {"token": "partial"}
    gen.close()

    # A closed generator yields nothing further -- no saved-frame reaches
    # the client after abandonment.
    with pytest.raises(StopIteration):
        next(gen)

    saved = chat_store_module.load(conv["id"])
    assistant_msgs = [m for m in saved["messages"] if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["content"] == "partial"


def test_history_endpoint_includes_sources_after_turn_with_sources(
    gui_client, gui_config, monkeypatch
):
    """Test N: after a turn with sources, the history endpoint's rendered
    output includes the sources block for that turn -- the property the
    user could previously only get by leaving the page and coming back."""
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr("gui.routes.chat._search", lambda *a, **k: _fake_search_result())

    conv = chat_store_module.create_conversation()

    lines = [
        json.dumps({"message": {"content": "Answer text"}, "done": False}).encode(),
        json.dumps({"done": True}).encode(),
    ]
    monkeypatch.setattr(
        "gui.ollama_chat.urllib.request.urlopen",
        lambda *a, **k: _FakeStreamResponse(lines),
    )

    send_resp = gui_client.post(
        "/chat/send",
        data={
            "conv": conv["id"],
            "message": "what does the paper say",
            "model": "gemma4:e4b",
            "scope": "chroma",
        },
    )
    assert send_resp.status_code == 200

    stream_resp = gui_client.get(f"/chat/stream?conv={conv['id']}&model=gemma4:e4b")
    assert stream_resp.status_code == 200
    assert '"saved": true' in stream_resp.text

    history_resp = gui_client.get(f"/chat/{conv['id']}/messages")
    assert history_resp.status_code == 200
    assert "Sources (1)" in history_resp.text
    assert "A Great Paper" in history_resp.text


# gui/routes/chat.py + chat_pending.html -- 08-REVIEW CR-01 / WR-01
# (post-turn re-render was unreachable; frame re-parse was crash-fragile).


def _split_pending_script_blocks(pending_html: str):
    """Return (done_block, saved_block): the inline-script bodies of the
    `if (data.done) { ... }` and `if (data.saved) { ... }` branches in a
    rendered chat_pending partial. Brace-matched so a nested block cannot be
    truncated. Used to pin the CR-01 client contract without a JS runtime."""
    def _body(marker: str) -> str:
        start = pending_html.index(marker)
        brace = pending_html.index("{", start)
        depth = 0
        for i in range(brace, len(pending_html)):
            ch = pending_html[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return pending_html[brace + 1:i]
        raise AssertionError(f"unbalanced braces after {marker!r}")

    return _body("if (data.done)"), _body("if (data.saved)")


def test_chat_pending_done_branch_does_not_close_stream(
    gui_client, gui_config, monkeypatch
):
    """CR-01: the finished-reply re-render is driven ONLY by the trailing
    `saved` frame, which the server emits AFTER persisting the turn. If the
    `done` branch closes the EventSource, the connection drops before `saved`
    arrives and the re-render (markdown formatting, Sources, model label)
    never fires. So the `done` branch must NOT close the stream, and the
    `saved` branch must be the sole close-and-re-render trigger."""
    conv = chat_store_module.create_conversation()

    resp = _send_with_canned_scope(gui_client, gui_config, monkeypatch, conv["id"], "hi")

    done_block, saved_block = _split_pending_script_blocks(resp.text)
    assert "es.close()" not in done_block, (
        "done branch must not close the stream before the `saved` frame "
        f"arrives; got: {done_block!r}"
    )
    assert "es.close()" in saved_block, "saved branch must close the stream"
    assert "htmx.ajax" in saved_block, "saved branch must re-render the history"
    assert "/messages" in saved_block, "saved branch must GET the history partial"


def test_sse_stream_emits_saved_frame_last_after_done(
    gui_client, gui_config, monkeypatch
):
    """CR-01 (server side): on a normal completed turn the `saved` frame is
    emitted AFTER the `done` frame and is the final frame of the stream, so a
    client that waits for `saved` (rather than closing on `done`) reliably
    receives the terminal re-render trigger."""
    conv = chat_store_module.create_conversation()
    _send_turn(gui_client, gui_config, monkeypatch, conv["id"], message="normal")

    monkeypatch.setattr(
        "gui.routes.chat._stream_ollama_chat",
        lambda *a, **k: iter([
            _frame({"token": "Hi"}),
            _frame({"done": True}),
        ]),
    )

    frames = list(chat_module._sse_stream(conv["id"], "gemma4:e4b"))
    payloads = [_payload(f) for f in frames]

    done_idx = next(i for i, p in enumerate(payloads) if p.get("done"))
    saved_idx = next(i for i, p in enumerate(payloads) if p.get("saved"))
    assert saved_idx > done_idx, "saved frame must come after done"
    assert saved_idx == len(payloads) - 1, "saved frame must be the last frame"


def test_sse_stream_survives_malformed_frame(gui_client, gui_config, monkeypatch):
    """WR-01: a frame that does not match the ``data: {json}\\n\\n`` shape must
    not crash the stream. The malformed frame is passed through untouched and
    the generator still completes -- persisting the assistant turn and
    emitting the trailing `saved` frame."""
    conv = chat_store_module.create_conversation()
    _send_turn(gui_client, gui_config, monkeypatch, conv["id"], message="malformed")

    def _fake_stream(*a, **k):
        yield _frame({"token": "Hel"})
        yield ": keepalive\n\n"  # a valid SSE comment line, not a data frame
        yield _frame({"token": "lo"})
        yield _frame({"done": True})

    monkeypatch.setattr("gui.routes.chat._stream_ollama_chat", _fake_stream)

    frames = list(chat_module._sse_stream(conv["id"], "gemma4:e4b"))

    assert ": keepalive\n\n" in frames, "malformed frame must pass through untouched"
    assert any(_frame_is_saved(f) for f in frames), "stream must still complete"

    saved = chat_store_module.load(conv["id"])
    last = saved["messages"][-1]
    assert last["role"] == "assistant"
    assert last["content"] == "Hello", "tokens around the malformed frame still accumulate"


def _frame_is_saved(frame: str) -> bool:
    if not frame.startswith("data: "):
        return False
    try:
        return bool(_payload(frame).get("saved"))
    except (ValueError, IndexError):
        return False
