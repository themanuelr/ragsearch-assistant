"""Tests for gui/chat_store.py (Phase 8 Plan 07, Task 2): JSON conversation
persistence (D-18) -- server-generated ids, atomic writes, and the
traversal-proof lookup contract (T-08-18).

Run with:  python -m pytest tests/test_gui_chat_store.py -x
"""

import json
import re

import pytest

import gui.chat_store as chat_store


@pytest.fixture(autouse=True)
def _tmp_store_dir(tmp_path, monkeypatch):
    """Every test gets a fresh, tmp-scoped store dir -- never the real
    ``.local/gui_chats/``."""
    monkeypatch.setattr(chat_store, "STORE_DIR", tmp_path / "gui_chats")
    yield


# ---------------------------------------------------------------------------
# Test 1: create_conversation id shape + save/load round-trip + atomic write
# ---------------------------------------------------------------------------

def test_create_conversation_returns_server_generated_filename_safe_id():
    conv = chat_store.create_conversation()

    assert conv["id"]
    assert re.match(r"^[A-Za-z0-9-]+$", conv["id"]), conv["id"]
    assert conv["title"] == "New Chat"
    assert conv["model"] is None
    assert conv["messages"] == []

    # File exists on disk under the (tmp-scoped) store dir.
    path = chat_store.STORE_DIR / f"{conv['id']}.json"
    assert path.is_file()


def test_save_load_round_trips_messages_title_and_model():
    conv = chat_store.create_conversation(title="My Conversation")
    conv["title"] = "Renamed"
    conv["model"] = "llama3:8b"
    conv["messages"].append({"role": "user", "content": "hello"})
    conv["messages"].append({"role": "assistant", "content": "hi there"})
    chat_store.save(conv)

    loaded = chat_store.load(conv["id"])

    assert loaded is not None
    assert loaded["title"] == "Renamed"
    assert loaded["model"] == "llama3:8b"
    assert loaded["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_files_live_under_store_dir_as_id_json_written_atomically(monkeypatch):
    """Writes go through temp-file + os.replace (never a direct open(path, 'w'))."""
    calls = {"replace": 0}
    real_replace = __import__("os").replace

    def _tracking_replace(src, dst):
        calls["replace"] += 1
        return real_replace(src, dst)

    monkeypatch.setattr(chat_store.os, "replace", _tracking_replace)

    conv = chat_store.create_conversation()
    path = chat_store.STORE_DIR / f"{conv['id']}.json"

    assert path.is_file()
    assert not path.with_suffix(".json.tmp").exists()  # tmp file cleaned up by replace
    assert calls["replace"] == 1

    with open(path, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["id"] == conv["id"]


# ---------------------------------------------------------------------------
# Test 2: list_conversations ordering, rename, delete, traversal-proof lookup
# ---------------------------------------------------------------------------

def test_list_conversations_orders_newest_first():
    first = chat_store.create_conversation(title="First")
    second = chat_store.create_conversation(title="Second")
    # Force distinct 'updated' timestamps regardless of clock resolution.
    first["updated"] = "2026-01-01T00:00:00"
    chat_store._write_atomic(first)
    second["updated"] = "2026-01-02T00:00:00"
    chat_store._write_atomic(second)

    listed = chat_store.list_conversations()

    assert [c["id"] for c in listed] == [second["id"], first["id"]]


def test_rename_updates_title():
    conv = chat_store.create_conversation(title="Original")

    ok = chat_store.rename(conv["id"], "Updated Title")

    assert ok is True
    assert chat_store.load(conv["id"])["title"] == "Updated Title"


def test_delete_removes_the_file():
    conv = chat_store.create_conversation()
    path = chat_store.STORE_DIR / f"{conv['id']}.json"
    assert path.is_file()

    ok = chat_store.delete(conv["id"])

    assert ok is True
    assert not path.is_file()
    assert chat_store.load(conv["id"]) is None


@pytest.mark.parametrize(
    "bad_id",
    [
        "../../etc/passwd",
        "..",
        "...",
        "foo/bar",
        "foo\\bar",
        "",
        None,
    ],
)
def test_load_rejects_traversal_shaped_or_dots_only_ids_without_touching_filesystem(
    bad_id, monkeypatch
):
    """A traversal-shaped or invalid id is rejected before any path is built --
    assert no filesystem read is attempted at all."""
    calls = []
    real_open = chat_store._path_for

    def _tracking_path_for(conv_id):
        calls.append(conv_id)
        return real_open(conv_id)

    monkeypatch.setattr(chat_store, "_path_for", _tracking_path_for)

    result = chat_store.load(bad_id)

    assert result is None
    assert calls == [], f"_path_for should never be called for an invalid id, got: {calls}"


def test_rename_and_delete_reject_traversal_shaped_ids():
    assert chat_store.rename("../../etc/passwd", "x") is False
    assert chat_store.delete("..") is False
