"""gui/ollama_chat.py -- Streaming Ollama client: NDJSON->SSE bridge (D-13)
+ model list (D-17), Phase 8 Plan 07.

Ollama's ``/api/chat`` streaming response (``stream: true``) is
newline-delimited JSON (NDJSON) -- NOT Server-Sent Events. Browsers'
``EventSource`` requires ``data: <payload>\\n\\n`` framing, so every NDJSON
line read here is parsed and re-emitted as one SSE frame server-side
(08-RESEARCH.md Pattern 3 / Pitfall 1). Piping Ollama's raw stream straight
through would produce malformed events that never fire ``onmessage``.

Mirrors ``mcp-ollama/server.py:_ollama_chat``'s urllib.request conventions
(timeouts, ``[Ollama error: ...]``/``[Ollama timeout: ...]`` string-prefix
errors) -- no new HTTP client dependency (Don't Hand-Roll: urllib only,
never httpx/requests).

Public API:
  _stream_ollama_chat(messages, model, timeout=300) -> generator[str]
      Yields SSE-framed events (``"data: {...}\\n\\n"``). Never raises past
      the caller: a connection-level failure yields exactly one terminal
      ``{"error": ...}`` event; a malformed individual NDJSON line yields a
      non-fatal inline ``{"error": ...}`` event and the stream continues.
  list_models(timeout=5) -> (list[str], str | None)
      Model names from GET /api/tags, plus an error string (or None). Never
      raises -- an unreachable Ollama returns ([], "[Ollama error: ...]").
"""

import json
import urllib.error
import urllib.request

OLLAMA_BASE = "http://localhost:11434"

_DEFAULT_TIMEOUT = 300


def _sse(payload: dict) -> str:
    """Frame one JSON-serializable payload as a browser-ready SSE event."""
    return f"data: {json.dumps(payload)}\n\n"


def _stream_ollama_chat(messages: list, model: str, timeout: int = _DEFAULT_TIMEOUT):
    """Generator bridging one Ollama ``/api/chat`` NDJSON stream to SSE.

    Each yielded frame's JSON payload is one of:
      {"token": "..."}   -- one content chunk
      {"done": true}      -- terminal event; generator stops right after
      {"error": "[Ollama error: ...]" | "[Ollama timeout: ...]"}
          -- either a non-fatal inline event for one malformed NDJSON line
             (streaming continues), or the single terminal event on a
             connection-level failure (URLError/timeout).
    """
    payload = json.dumps({"model": model, "messages": messages, "stream": True}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except (ValueError, UnicodeDecodeError) as e:
                    yield _sse({"error": f"[Ollama error: malformed response line: {e}]"})
                    continue
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield _sse({"token": content})
                if chunk.get("done"):
                    yield _sse({"done": True})
                    break
    except TimeoutError as e:
        # socket.timeout is an alias of TimeoutError since Python 3.10; must
        # be caught before URLError below (TimeoutError is not always a
        # URLError subclass depending on where the timeout fires).
        yield _sse({"error": f"[Ollama timeout: {e}]"})
    except urllib.error.URLError as e:
        yield _sse({"error": f"[Ollama error: {e}]"})


def list_models(timeout: int = 5):
    """Return ``(model_names, error_or_none)`` from GET /api/tags.

    Never raises -- an unreachable Ollama returns ``([], "[Ollama error: ...]")``
    so callers can render the page banner ("Can't reach Ollama at
    localhost:11434. Start Ollama and try again." per 08-UI-SPEC.md).
    """
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read())
        names = [m["name"] for m in data.get("models", [])]
        return names, None
    except TimeoutError as e:
        return [], f"[Ollama timeout: {e}]"
    except urllib.error.URLError as e:
        return [], f"[Ollama error: {e}]"
    except (ValueError, KeyError, TypeError) as e:
        return [], f"[Ollama error: unexpected response envelope: {e}]"
