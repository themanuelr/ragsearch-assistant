"""Tests for gui/ollama_chat.py (Phase 8 Plan 07, Task 1): the NDJSON->SSE
streaming bridge (D-13) and model listing (D-17).

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

import gui.ollama_chat as ollama_chat
from gui.ollama_chat import _stream_ollama_chat, list_models

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


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
