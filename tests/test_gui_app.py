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

08-02 additions: Overview (Project) page must render the live-scanned stage
table + drop-folder pending list (D-09), and the per-paper drill-down
partial (D-12) must return 200 for a known key / 404 for an unknown one.
These tests monkeypatch ``gui.routes.overview.load_gui_config`` (the name
bound inside that route module) so the route reads the tmp-scoped
``gui_config`` fixture instead of the real repo-root config.json.
"""

import json
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


# ---------------------------------------------------------------------------
# 08-02: Overview (Project) page + per-paper drill-down partial (D-09/D-12)
# ---------------------------------------------------------------------------

def _patch_overview_config(monkeypatch, gui_config):
    import gui.routes.overview as overview_routes

    monkeypatch.setattr(overview_routes, "load_gui_config", lambda: gui_config)


def test_overview_project_empty_states(gui_client, gui_config, monkeypatch):
    """With an empty tmp project the page shows both locked empty states."""
    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    resp = gui_client.get("/overview/project")

    assert resp.status_code == 200
    assert "Papers in This Project" in resp.text
    assert "Drop Folder" in resp.text
    assert "No papers yet" in resp.text
    assert "No PDFs waiting" in resp.text


def test_overview_project_renders_paper_and_pending_pdf(gui_client, gui_config, monkeypatch):
    """A registry paper renders its title + seven stage-glyph cells; a
    drop-folder PDF renders in the pending list."""
    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    registry_path = pathlib.Path(gui_config["registry_path"])
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = {
        "10.1234/known": {
            "title": "A Known Paper",
            "authors": ["A. Author"],
            "year": 2021,
            "journal": "J",
            "doi": "10.1234/known",
            "arxiv_id": None,
            "projects": ["proj-a"],
            "source_path": "/x/known.pdf",
            "paperjson_path": str(
                pathlib.Path(gui_config["paperjson_cache_dir"]) / "known.json"
            ),
            "summary": None,
            "key_findings": None,
        }
    }
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f)

    uningested_dir = pathlib.Path(gui_config["uningested_dir"])
    uningested_dir.mkdir(parents=True, exist_ok=True)
    (uningested_dir / "waiting.pdf").write_bytes(b"%PDF-1.4")

    resp = gui_client.get("/overview/project")

    assert resp.status_code == 200
    assert "A Known Paper" in resp.text
    assert resp.text.count("stage-glyph") == 7
    assert "waiting.pdf" in resp.text


def test_overview_project_paper_drilldown_known_and_unknown(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    registry_path = pathlib.Path(gui_config["registry_path"])
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = {
        "10.1234/known": {
            "title": "A Known Paper",
            "authors": ["A. Author"],
            "year": 2021,
            "journal": "J",
            "doi": "10.1234/known",
            "arxiv_id": None,
            "projects": ["proj-a"],
            "source_path": "/x/known.pdf",
            "paperjson_path": str(
                pathlib.Path(gui_config["paperjson_cache_dir"]) / "known.json"
            ),
            "summary": None,
            "key_findings": None,
        }
    }
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f)

    known_resp = gui_client.get("/overview/project/paper/10.1234/known")
    assert known_resp.status_code == 200
    assert "A Known Paper" in known_resp.text
    assert str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "known.json") in known_resp.text

    unknown_resp = gui_client.get("/overview/project/paper/does-not-exist")
    assert unknown_resp.status_code == 404
