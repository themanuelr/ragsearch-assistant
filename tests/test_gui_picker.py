"""Tests for the shared paper-picker (Phase 08.1 Plan 01, D-01..D-05):
``gui.scan.filter_and_paginate``, ``gui.scan.scan_universal``'s new
``authors`` field (Pitfall 1), and the ``GET /papers/picker`` route + shared
``partials/paper_picker.html`` partial.

RED at authoring time (Task 1): ``gui.scan.filter_and_paginate`` and the
``/papers/picker`` route do not exist yet. Task 2 adds ``filter_and_paginate``
+ the ``scan_universal`` authors field; Task 3 adds the route + partial +
Overview wiring. Function-level imports of ``filter_and_paginate`` (mirroring
the Phase-4 ``from scripts import biblio`` attribute-access precedent) let
this file collect cleanly before the symbol exists -- only the test BODIES
fail until the implementation lands.

Run with: python -m pytest tests/test_gui_picker.py -x
"""

import json
import pathlib

from scripts.note import _sanitize_filename

# ---------------------------------------------------------------------------
# Helpers (mirror tests/test_gui_scan.py's established fixture idioms)
# ---------------------------------------------------------------------------


def _registry_entry(title, paperjson_path, projects, **extra):
    entry = {
        "title": title,
        "authors": ["A. Author", "B. Coauthor"],
        "year": 2020,
        "journal": "Some Journal",
        "doi": "10.1234/" + _sanitize_filename(title).lower().replace(" ", "-"),
        "arxiv_id": None,
        "projects": projects,
        "source_path": "/x/source.pdf",
        "paperjson_path": paperjson_path,
        "summary": None,
        "key_findings": None,
    }
    entry.update(extra)
    return entry


def _write_registry(gui_config, registry):
    registry_path = pathlib.Path(gui_config["registry_path"])
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f)


def _patch_overview_config(monkeypatch, gui_config):
    import gui.routes.overview as overview_routes

    monkeypatch.setattr(overview_routes, "load_gui_config", lambda: gui_config)


def _make_entries(n):
    return [
        {"title": f"Paper {i:02d}", "authors": ["X. Author"], "journal": "J", "year": 2000 + i}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Unit tests: gui.scan.filter_and_paginate (D-01, D-02, D-03)
# ---------------------------------------------------------------------------


def test_filter_and_paginate_empty_q_returns_page_one_slice():
    from gui.scan import filter_and_paginate

    entries = _make_entries(30)
    result = filter_and_paginate(entries, "", 1, 25)

    assert result["total"] == 30
    assert len(result["entries"]) == 25
    assert result["entries"] == entries[:25]
    assert result["page"] == 1
    assert result["page_size"] == 25


def test_filter_and_paginate_title_substring_filters():
    from gui.scan import filter_and_paginate

    entries = [
        {"title": "Deep Learning Basics", "authors": [], "journal": "J", "year": 2020},
        {"title": "Unrelated Paper", "authors": [], "journal": "J", "year": 2020},
    ]
    result = filter_and_paginate(entries, "deep learning", 1, 25)

    assert result["total"] == 1
    assert result["entries"][0]["title"] == "Deep Learning Basics"


def test_filter_and_paginate_matches_author_surname_both_shapes():
    """Feeds both the flat universal shape and the nested project/metadata
    shape (D-02) -- both must search by author."""
    from gui.scan import filter_and_paginate

    universal_shape = {"title": "Universal Paper", "authors": ["Jane Smith"], "journal": "J", "year": 2020}
    project_shape = {
        "title": "Project Paper",
        "metadata": {"authors": ["John Doe"], "title": "Project Paper", "journal": "J", "year": 2021},
    }
    entries = [universal_shape, project_shape]

    result_universal = filter_and_paginate(entries, "smith", 1, 25)
    assert result_universal["total"] == 1
    assert result_universal["entries"][0]["title"] == "Universal Paper"

    result_project = filter_and_paginate(entries, "doe", 1, 25)
    assert result_project["total"] == 1
    assert result_project["entries"][0]["title"] == "Project Paper"


def test_filter_and_paginate_matches_journal_and_year():
    from gui.scan import filter_and_paginate

    entries = [
        {"title": "A", "authors": [], "journal": "Nature", "year": 2019},
        {"title": "B", "authors": [], "journal": "Science", "year": 2021},
    ]

    by_journal = filter_and_paginate(entries, "nature", 1, 25)
    assert by_journal["total"] == 1
    assert by_journal["entries"][0]["title"] == "A"

    by_year = filter_and_paginate(entries, "2021", 1, 25)
    assert by_year["total"] == 1
    assert by_year["entries"][0]["title"] == "B"


def test_filter_and_paginate_page_two_and_has_next_flips():
    from gui.scan import filter_and_paginate

    entries = _make_entries(30)

    page1 = filter_and_paginate(entries, "", 1, 25)
    assert page1["has_next"] is True

    page2 = filter_and_paginate(entries, "", 2, 25)
    assert len(page2["entries"]) == 5
    assert page2["entries"] == entries[25:30]
    assert page2["has_next"] is False


def test_filter_and_paginate_case_insensitive():
    from gui.scan import filter_and_paginate

    entries = [{"title": "Quantum Computing", "authors": [], "journal": "J", "year": 2020}]

    result = filter_and_paginate(entries, "QUANTUM", 1, 25)
    assert result["total"] == 1


# ---------------------------------------------------------------------------
# Unit test: scan_universal exposes authors (Pitfall 1)
# ---------------------------------------------------------------------------


def test_scan_universal_entries_carry_authors_field(gui_config):
    from gui.scan import scan_universal

    entry = _registry_entry(
        "Authored Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "authored.json"),
        ["proj-a"],
        authors=["Ada Lovelace", "Charles Babbage"],
    )
    _write_registry(gui_config, {"key-authored": entry})

    entries = scan_universal(gui_config)
    by_key = {e["registry_key"]: e for e in entries}

    assert by_key["key-authored"]["authors"] == ["Ada Lovelace", "Charles Babbage"]


# ---------------------------------------------------------------------------
# Route tests: GET /papers/picker (D-04, D-05)
# ---------------------------------------------------------------------------


def test_picker_route_returns_200_with_rows(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    entry = _registry_entry(
        "Picker Route Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "picker-route.json"),
        ["proj-a"],
    )
    _write_registry(gui_config, {"key-picker": entry})

    resp = gui_client.get(
        "/papers/picker", params={"scope": "universal", "q": "", "page": 1, "page_size": 25}
    )

    assert resp.status_code == 200
    assert "Picker Route Paper" in resp.text


def test_picker_route_checkbox_mode_renders_registry_keys_inputs(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    entry = _registry_entry(
        "Checkbox Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "checkbox.json"),
        ["proj-a"],
    )
    _write_registry(gui_config, {"key-checkbox": entry})

    resp = gui_client.get("/papers/picker", params={"scope": "universal", "select_mode": "checkbox"})

    assert resp.status_code == 200
    assert 'name="registry_keys"' in resp.text
    assert 'name="registry_key"' not in resp.text


def test_picker_route_radio_mode_renders_registry_key_input(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    entry = _registry_entry(
        "Radio Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "radio.json"),
        ["proj-a"],
    )
    _write_registry(gui_config, {"key-radio": entry})

    resp = gui_client.get("/papers/picker", params={"scope": "universal", "select_mode": "radio"})

    assert resp.status_code == 200
    assert 'name="registry_key"' in resp.text
    assert 'name="registry_keys"' not in resp.text


def test_picker_route_default_display_mode_renders_no_select_inputs(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    entry = _registry_entry(
        "Display Paper",
        str(pathlib.Path(gui_config["paperjson_cache_dir"]) / "display.json"),
        ["proj-a"],
    )
    _write_registry(gui_config, {"key-display": entry})

    resp = gui_client.get("/papers/picker", params={"scope": "universal"})

    assert resp.status_code == 200
    assert 'name="registry_key"' not in resp.text
    assert 'name="registry_keys"' not in resp.text


def test_picker_route_clamps_oversized_page_size(gui_client, gui_config, monkeypatch):
    """A page_size outside {25,50,100,200} must clamp to an allowed value
    (25) server-side -- the HTML <select> options are client-side only and
    cannot be trusted (T-08.1-01-02)."""
    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    registry = {
        f"key-{i}": _registry_entry(
            f"Paper {i:02d}",
            str(pathlib.Path(gui_config["paperjson_cache_dir"]) / f"paper-{i}.json"),
            ["proj-a"],
        )
        for i in range(30)
    }
    _write_registry(gui_config, registry)

    resp = gui_client.get(
        "/papers/picker",
        params={"scope": "universal", "select_mode": "checkbox", "page_size": 9999},
    )

    assert resp.status_code == 200
    # 30 entries exist; an unclamped page_size=9999 would render all 30.
    # A correctly clamped page_size (25) renders exactly 25 rows.
    assert resp.text.count('name="registry_keys"') == 25


def test_picker_route_escapes_script_in_q(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    resp = gui_client.get(
        "/papers/picker", params={"scope": "universal", "q": "<script>alert(1)</script>"}
    )

    assert resp.status_code == 200
    assert "<script>alert(1)</script>" not in resp.text


# ---------------------------------------------------------------------------
# Overview wiring: both pages mount the shared picker (D-04)
# ---------------------------------------------------------------------------


def test_overview_pages_wire_the_shared_picker(gui_client, gui_config, monkeypatch):
    gui_config["project_name"] = "proj-a"
    _patch_overview_config(monkeypatch, gui_config)

    project_resp = gui_client.get("/overview/project")
    universal_resp = gui_client.get("/overview/universal")

    assert project_resp.status_code == 200
    assert universal_resp.status_code == 200
    for resp in (project_resp, universal_resp):
        assert 'id="picker-results"' in resp.text
        assert 'hx-get="/papers/picker"' in resp.text
