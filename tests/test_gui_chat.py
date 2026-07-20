"""Tests for the chat ChromaDB retrieval-query box (Phase 08.1 Plan 03,
D-10..D-13).

There is no existing dedicated test_gui_chat.py -- chat route coverage lives
in tests/test_gui_ollama_chat.py (see its TestClient/config-patch/
_stream_ollama_chat-monkeypatch idioms, mirrored here). This file is scoped
to the NEW optional retrieval-query box only:

- D-10: blank ``chroma_query`` falls back to the message as the retrieval
  query; an explicit ``chroma_query`` overrides it.
- D-13: ``n_results`` threads through to ``embed._search``'s second arg,
  clamped server-side to 1..20 regardless of client input (Security Domain,
  T-08.1-03-01).
- D-12: the effective retrieval query is recorded as a top-level
  ``chroma_query`` key on the stored assistant message (not folded into
  ``sources``), present only for scope == "chroma".
- chat_sources.html renders the recorded query when present, HTML-escaped
  (T-08.1-03-02).

RED at authoring time (Task 1): ``_resolve_scope`` does not yet accept
``chroma_query``/``n_results``, ``chat_send`` does not yet clamp or persist
either, and chat_sources.html does not yet render a query line. Tasks 2-4
turn this green without changing this file.

Run with:  python -m pytest tests/test_gui_chat.py -x
"""

import json

import pytest

import gui.chat_store as chat_store_module
import gui.jobs as gui_jobs_module
import gui.routes.chat as chat_module


@pytest.fixture(autouse=True)
def _tmp_chat_store(tmp_path, monkeypatch):
    """Every test in this file gets a fresh, tmp-scoped chat store dir --
    never the real ``.local/gui_chats/`` (mirrors test_gui_ollama_chat.py)."""
    monkeypatch.setattr(chat_store_module, "STORE_DIR", tmp_path / "gui_chats")
    yield


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


def _frame(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# _resolve_scope: chroma_query blank-fallback, explicit-override, n_results
# threading (D-10, D-13)
# ---------------------------------------------------------------------------

def test_resolve_scope_chroma_blank_query_falls_back_to_message(gui_config, monkeypatch):
    search_calls = []

    def fake_search(query, n_results, config):
        search_calls.append((query, n_results))
        return _fake_search_result()

    monkeypatch.setattr("gui.routes.chat._search", fake_search)

    chat_module._resolve_scope(
        "chroma", "Q", [], gui_config, chroma_query="", n_results=5
    )

    assert search_calls == [("Q", 5)]


def test_resolve_scope_chroma_explicit_query_overrides_message(gui_config, monkeypatch):
    search_calls = []

    def fake_search(query, n_results, config):
        search_calls.append((query, n_results))
        return _fake_search_result()

    monkeypatch.setattr("gui.routes.chat._search", fake_search)

    chat_module._resolve_scope(
        "chroma", "Q", [], gui_config, chroma_query="topic X", n_results=5
    )

    assert search_calls == [("topic X", 5)]


def test_resolve_scope_chroma_n_results_threads_to_search(gui_config, monkeypatch):
    search_calls = []

    def fake_search(query, n_results, config):
        search_calls.append((query, n_results))
        return _fake_search_result()

    monkeypatch.setattr("gui.routes.chat._search", fake_search)

    chat_module._resolve_scope(
        "chroma", "Q", [], gui_config, chroma_query=None, n_results=12
    )

    assert search_calls == [("Q", 12)]


# ---------------------------------------------------------------------------
# chat_send: server-side n_results clamp to 1..20 (D-13, T-08.1-03-01)
# ---------------------------------------------------------------------------

def test_chat_send_clamps_out_of_range_n_results_before_search(
    gui_client, gui_config, monkeypatch
):
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)

    search_calls = []

    def fake_search(query, n_results, config):
        search_calls.append((query, n_results))
        return _fake_search_result()

    monkeypatch.setattr("gui.routes.chat._search", fake_search)

    conv_high = chat_store_module.create_conversation()
    resp_high = gui_client.post(
        "/chat/send",
        data={
            "conv": conv_high["id"],
            "message": "hi",
            "model": "gemma4:e4b",
            "scope": "chroma",
            "n_results": "999",
        },
    )
    assert resp_high.status_code == 200
    assert search_calls[-1][1] == 20, "n_results above range must clamp to 20"

    conv_low = chat_store_module.create_conversation()
    resp_low = gui_client.post(
        "/chat/send",
        data={
            "conv": conv_low["id"],
            "message": "hi",
            "model": "gemma4:e4b",
            "scope": "chroma",
            "n_results": "0",
        },
    )
    assert resp_low.status_code == 200
    assert search_calls[-1][1] == 1, "n_results below range must clamp to 1"


# ---------------------------------------------------------------------------
# chat_send: route accepts the new form fields without error
# ---------------------------------------------------------------------------

def test_chat_send_accepts_chroma_query_and_n_results_form_fields(
    gui_client, gui_config, monkeypatch
):
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr("gui.routes.chat._search", lambda *a, **k: _fake_search_result())

    conv = chat_store_module.create_conversation()
    resp = gui_client.post(
        "/chat/send",
        data={
            "conv": conv["id"],
            "message": "hi",
            "model": "gemma4:e4b",
            "scope": "chroma",
            "chroma_query": "topic X",
            "n_results": "12",
        },
    )

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Persistence (D-12): a chroma-scope turn's stored assistant message carries
# the effective retrieval query; a non-chroma turn's does not.
# ---------------------------------------------------------------------------

def test_chat_send_persists_chroma_query_on_assistant_message(
    gui_client, gui_config, monkeypatch
):
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr("gui.routes.chat._search", lambda *a, **k: _fake_search_result())
    monkeypatch.setattr(
        "gui.routes.chat._stream_ollama_chat",
        lambda *a, **k: iter([_frame({"token": "Answer"}), _frame({"done": True})]),
    )

    conv = chat_store_module.create_conversation()
    resp = gui_client.post(
        "/chat/send",
        data={
            "conv": conv["id"],
            "message": "what does the paper say",
            "model": "gemma4:e4b",
            "scope": "chroma",
            "chroma_query": "topic X",
            "n_results": "5",
        },
    )
    assert resp.status_code == 200

    list(chat_module._sse_stream(conv["id"], "gemma4:e4b"))

    saved = chat_store_module.load(conv["id"])
    assistant_msg = saved["messages"][-1]
    assert assistant_msg["chroma_query"] == "topic X"


def test_chat_send_blank_chroma_query_persists_message_as_effective_query(
    gui_client, gui_config, monkeypatch
):
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr("gui.routes.chat._search", lambda *a, **k: _fake_search_result())
    monkeypatch.setattr(
        "gui.routes.chat._stream_ollama_chat",
        lambda *a, **k: iter([_frame({"token": "Answer"}), _frame({"done": True})]),
    )

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

    list(chat_module._sse_stream(conv["id"], "gemma4:e4b"))

    saved = chat_store_module.load(conv["id"])
    assistant_msg = saved["messages"][-1]
    assert assistant_msg["chroma_query"] == "what does the paper say"


def test_chat_send_non_chroma_scope_records_no_chroma_query(
    gui_client, gui_config, monkeypatch
):
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr(
        "gui.routes.chat._stream_ollama_chat",
        lambda *a, **k: iter([_frame({"token": "Answer"}), _frame({"done": True})]),
    )

    conv = chat_store_module.create_conversation()
    resp = gui_client.post(
        "/chat/send",
        data={"conv": conv["id"], "message": "hi", "model": "gemma4:e4b", "scope": "none"},
    )
    assert resp.status_code == 200

    list(chat_module._sse_stream(conv["id"], "gemma4:e4b"))

    saved = chat_store_module.load(conv["id"])
    assistant_msg = saved["messages"][-1]
    assert assistant_msg.get("chroma_query") is None


# ---------------------------------------------------------------------------
# chat_sources.html: renders the recorded query when present, HTML-escaped
# ---------------------------------------------------------------------------

def test_chat_sources_html_renders_recorded_chroma_query(
    gui_client, gui_config, monkeypatch
):
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr("gui.routes.chat.list_models", lambda: (["gemma4:e4b"], None))
    monkeypatch.setattr("gui.routes.chat.scan_project_papers", lambda config: [])

    conv = chat_store_module.create_conversation()
    conv["messages"] = [
        {"role": "user", "content": "what does the paper say"},
        {
            "role": "assistant",
            "content": "Answer",
            "model": "gemma4:e4b",
            "sources": [{"title": "A Great Paper", "section": "Results", "score": 0.9}],
            "chroma_query": "topic X",
        },
    ]
    chat_store_module.save(conv)

    resp = gui_client.get(f"/chat?conv={conv['id']}")

    assert resp.status_code == 200
    assert "Retrieved for" in resp.text
    assert "topic X" in resp.text


def test_chat_sources_html_omits_query_line_when_absent(gui_client, gui_config, monkeypatch):
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr("gui.routes.chat.list_models", lambda: (["gemma4:e4b"], None))
    monkeypatch.setattr("gui.routes.chat.scan_project_papers", lambda config: [])

    conv = chat_store_module.create_conversation()
    conv["messages"] = [
        {"role": "user", "content": "what does the paper say"},
        {
            "role": "assistant",
            "content": "Answer",
            "model": "gemma4:e4b",
            "sources": [{"title": "A Great Paper", "section": "Results", "score": 0.9}],
        },
    ]
    chat_store_module.save(conv)

    resp = gui_client.get(f"/chat?conv={conv['id']}")

    assert resp.status_code == 200
    assert "Retrieved for" not in resp.text


def test_chat_sources_html_escapes_chroma_query_html(gui_client, gui_config, monkeypatch):
    monkeypatch.setattr("gui.routes.chat.load_gui_config", lambda: gui_config)
    monkeypatch.setattr(gui_jobs_module, "is_busy", lambda: False)
    monkeypatch.setattr("gui.routes.chat.list_models", lambda: (["gemma4:e4b"], None))
    monkeypatch.setattr("gui.routes.chat.scan_project_papers", lambda config: [])

    conv = chat_store_module.create_conversation()
    conv["messages"] = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "Answer",
            "model": "gemma4:e4b",
            "sources": [{"title": "A Great Paper", "section": "Results", "score": 0.9}],
            "chroma_query": "<script>alert(1)</script>",
        },
    ]
    chat_store_module.save(conv)

    resp = gui_client.get(f"/chat?conv={conv['id']}")

    assert resp.status_code == 200
    assert "<script>alert(1)</script>" not in resp.text
    assert "&lt;script&gt;" in resp.text
