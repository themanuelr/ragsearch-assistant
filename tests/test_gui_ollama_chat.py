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
        lambda *a, **k: (search_calls.append(1) or ([], None, None, None)),
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
