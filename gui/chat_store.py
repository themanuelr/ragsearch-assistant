"""gui/chat_store.py -- JSON conversation persistence (Phase 8 Plan 07, D-18).

Conversations persist as ``.local/gui_chats/<id>.json`` -- one JSON file per
conversation, written atomically (temp file + ``os.replace``, matching
``scripts/obsidian_cli.py::create_note``'s shape).

Ids are generated **server-side only** (T-08-18 -- path traversal via a
conversation id). Every lookup validates the id against a strict
filename-safe charset *before* any path is constructed from it, mirroring
``gui/jobs.py``'s ``_JOB_ID_RE`` precedent for the identical threat class --
a crafted id (path separators, dot-only segments) can never traverse outside
``STORE_DIR`` because no path is ever built from an id that fails the check.

Conversation JSON shape::

    {
      "id": str, "title": str, "model": str | None,
      "created": str, "updated": str,
      "messages": [{"role": str, "content": str, "sources"?: list, ...}],
    }

Public API:
  create_conversation(title=None) -> dict
      New conversation dict, already written to disk.
  load(conv_id) -> dict | None
      None on an invalid/absent id -- never a filesystem exception.
  save(conv) -> None
      Atomic temp+os.replace write; bumps 'updated'.
  list_conversations() -> list[dict]
      Every conversation, newest-first (by 'updated').
  rename(conv_id, title) -> bool
  delete(conv_id) -> bool
"""

import datetime
import json
import os
import pathlib
import re
import uuid

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Monkeypatched by tests to a tmp_path subdirectory -- test runs never write
# into the real .local/gui_chats/.
STORE_DIR = _REPO_ROOT / ".local" / "gui_chats"

# Server-generated ids only: YYYYMMDD-HHMMSS-<8 hex chars>. Strict charset
# validated on every lookup before path construction (T-08-18) -- mirrors
# gui/jobs.py's _JOB_ID_RE precedent for the same threat class.
_CONV_ID_RE = re.compile(r"^[A-Za-z0-9-]+$")


def _store_dir() -> pathlib.Path:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    return STORE_DIR


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _new_conv_id() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{ts}-{suffix}"


def _is_valid_id(conv_id) -> bool:
    return bool(conv_id) and bool(_CONV_ID_RE.match(conv_id))


def _path_for(conv_id: str) -> pathlib.Path:
    return _store_dir() / f"{conv_id}.json"


def _write_atomic(conv: dict) -> None:
    path = _path_for(conv["id"])
    tmp_path = str(path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(conv, f, indent=2)
    os.replace(tmp_path, str(path))


def create_conversation(title: str | None = None) -> dict:
    """Create and persist a new conversation, returning its dict."""
    now = _now()
    conv = {
        "id": _new_conv_id(),
        "title": title or "New Chat",
        "model": None,
        "created": now,
        "updated": now,
        "messages": [],
    }
    _write_atomic(conv)
    return conv


def load(conv_id: str):
    """Return the conversation dict, or None for an invalid/absent id.

    The id is validated against ``_CONV_ID_RE`` *before* any path is built --
    a traversal-shaped id (``../secrets``, ``..``, path separators) never
    reaches the filesystem at all (T-08-18).
    """
    if not _is_valid_id(conv_id):
        return None
    path = _path_for(conv_id)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save(conv: dict) -> None:
    """Atomic write; bumps 'updated' to now."""
    conv["updated"] = _now()
    _write_atomic(conv)


def list_conversations() -> list:
    """Every conversation on disk, newest-first by 'updated'."""
    convs = []
    for path in _store_dir().glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                convs.append(json.load(f))
        except (OSError, ValueError):
            continue
    convs.sort(key=lambda c: c.get("updated", ""), reverse=True)
    return convs


def rename(conv_id: str, title: str) -> bool:
    """Rename a conversation's title. Returns False for an invalid/absent id."""
    conv = load(conv_id)
    if conv is None:
        return False
    conv["title"] = title
    save(conv)
    return True


def delete(conv_id: str) -> bool:
    """Delete a conversation's JSON file. Returns False for an invalid/absent id."""
    if not _is_valid_id(conv_id):
        return False
    path = _path_for(conv_id)
    if not path.is_file():
        return False
    path.unlink()
    return True
