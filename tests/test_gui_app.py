"""Tests for the Phase 8 GUI walking skeleton (gui/app.py).

RED at Wave 0 authoring time (08-01 Task 1): this file imports ``gui.app``,
which does not exist until Task 2. Task 3 wires the six stub routers and
this file goes green (08-01 acceptance criteria).

Covers:
- root path renders the sidebar shell with all seven nav labels, locked order
- every one of the seven page routes returns 200
- the Origin-check CSRF middleware rejects a cross-origin state-changing POST
- scripts/gui.py binds uvicorn to 127.0.0.1 only (never 0.0.0.0)
- no template references a CDN / external asset URL -- vendored htmx + one
  local stylesheet only
"""

import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Locked sidebar order per 08-01-PLAN.md must_haves.truths and 08-UI-SPEC.md
# Layout & Navigation Contract.
_SIDEBAR_LABELS = [
    "Overview (Project)",
    "Overview (Universal)",
    "Processing",
    "Chat",
    "Logs & Activity",
    "Docs & Help",
    "Settings",
]

_PAGE_PATHS = [
    "/overview/project",
    "/overview/universal",
    "/processing",
    "/chat",
    "/logs",
    "/docs",
    "/settings",
]


def test_root_redirects_and_lists_sidebar_in_locked_order(gui_client):
    resp = gui_client.get("/", follow_redirects=True)
    assert resp.status_code == 200
    html = resp.text
    # Every label must appear, in the locked order (each subsequent label's
    # first occurrence must come after the previous one's).
    positions = [html.index(label) for label in _SIDEBAR_LABELS]
    assert positions == sorted(positions), (
        f"Sidebar labels not in locked order: {list(zip(_SIDEBAR_LABELS, positions))}"
    )


@pytest.mark.parametrize("path", _PAGE_PATHS)
def test_each_nav_page_returns_200(gui_client, path):
    resp = gui_client.get(path)
    assert resp.status_code == 200, f"{path} returned {resp.status_code}"


def test_csrf_origin_rejected(gui_client):
    """A cross-origin state-changing POST must be rejected with 403 (T-08-01)."""
    resp = gui_client.post(
        "/processing",
        data={"dummy": "1"},
        headers={"Origin": "http://evil.example"},
    )
    assert resp.status_code == 403


def test_launcher_binds_loopback_only(monkeypatch):
    """scripts/gui.py must call uvicorn.run with host='127.0.0.1' (T-08-06)."""
    import scripts.gui as gui_launcher

    calls = {}

    def _fake_run(app, host=None, port=None, **kwargs):
        calls["host"] = host
        calls["port"] = port

    monkeypatch.setattr(gui_launcher, "uvicorn", type("U", (), {"run": staticmethod(_fake_run)}))
    monkeypatch.setattr(gui_launcher, "webbrowser", type("W", (), {"open": staticmethod(lambda *a, **k: None)}))
    monkeypatch.setattr(
        gui_launcher.sys, "argv", ["gui.py", "--no-browser", "--port", "9999"]
    )

    gui_launcher.main()

    assert calls["host"] == "127.0.0.1"
    assert calls["host"] != "0.0.0.0"
    assert calls["port"] == 9999


def test_no_external_asset_urls():
    """No template may reference a CDN or remote script/style URL (zero-cloud)."""
    base_html = (_REPO_ROOT / "gui" / "templates" / "base.html").read_text(encoding="utf-8")
    assert "/static/htmx.min.js" in base_html
    assert "/static/style.css" in base_html
    assert "http://" not in base_html
    assert "https://" not in base_html

    templates_dir = _REPO_ROOT / "gui" / "templates"
    for tpl in templates_dir.rglob("*.html"):
        text = tpl.read_text(encoding="utf-8")
        assert "http://" not in text, f"{tpl} references an external http:// URL"
        assert "https://" not in text, f"{tpl} references an external https:// URL"
