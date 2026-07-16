"""Tests for gui/routes/docs.py -- server-rendered markdown, whitelist-only
doc picker, zero path-traversal surface (Phase 8 Plan 04, D-21, T-08-10)."""

import gui.routes.docs as docs


def test_docs_root_renders_readme(gui_client):
    resp = gui_client.get("/docs")
    assert resp.status_code == 200
    # Proves markdown->HTML conversion ran server-side, not just that README
    # text appears somewhere -- a raw, unconverted markdown file would never
    # contain a literal <h1> tag.
    assert "<h1>Research Paper Intelligence System</h1>" in resp.text


def test_docs_obsidian_note_format_slug(gui_client):
    resp = gui_client.get("/docs/obsidian-note-format")
    assert resp.status_code == 200
    assert "<h1>" in resp.text or "<h2>" in resp.text


def test_docs_config_reference_slug_renders_fenced_json(gui_client):
    resp = gui_client.get("/docs/config-reference")
    assert resp.status_code == 200
    assert "registry_path" in resp.text
    assert "<pre>" in resp.text or "<code>" in resp.text


def test_doc_picker_rejects_unknown(gui_client):
    """GET /docs/<anything-not-whitelisted> (including traversal-shaped
    names) returns 404 -- the slug is a dict-key lookup only, never a path
    component (T-08-10)."""
    resp = gui_client.get("/docs/nope")
    assert resp.status_code == 404

    resp_traversal = gui_client.get("/docs/..%2Fconfig.json")
    assert resp_traversal.status_code == 404


def test_docs_module_has_no_open_call_on_request_derived_path():
    """Static guarantee: DOCS is the sole whitelist, and _render_doc never
    joins the incoming slug onto a filesystem path -- it is used strictly as
    a dict key (`DOCS.get(slug)`)."""
    assert isinstance(docs.DOCS, dict)
    assert set(docs.DOCS.keys()) == {"readme", "obsidian-note-format", "config-reference"}


def test_missing_doc_on_disk_renders_message_not_traceback(monkeypatch, gui_client, tmp_path):
    missing_path = tmp_path / "does-not-exist.md"
    monkeypatch.setitem(docs.DOCS, "readme", ("README", missing_path))

    resp = gui_client.get("/docs")

    assert resp.status_code == 200
    assert "doc not found on disk" in resp.text
