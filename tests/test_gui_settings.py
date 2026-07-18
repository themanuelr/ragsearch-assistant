"""Tests for gui/routes/settings.py -- typed config form, atomic save,
D-20 data-path warn+confirm flow (Phase 8 Plan 04, D-19/D-20).

Every test monkeypatches ``settings.CONFIG_PATH`` to a ``tmp_path`` file (the
``_patch_config_path`` autouse fixture below) -- the session-scoped
``_guard_real_config_json`` fixture in tests/conftest.py is the safety net if
any test here forgets to do so.
"""

import inspect
import json

import pytest

from gui.routes import settings


def _config_dict(gui_config, **overrides):
    """Full 23-key config dict: config.example.json defaults, tmp-scoped
    gui_config paths, plus test-specific overrides -- mirrors what a real
    <form> submission carries (every field, not just the one being changed)."""
    values = json.loads(settings._EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))
    values.update(gui_config)
    values.update(overrides)
    return values


def _form_data(values: dict) -> dict:
    """Convert a full config dict into TestClient-postable form data,
    matching how a real browser <form> submission encodes each FIELDS type
    (checkbox: omitted when falsy, present -> 'on' when truthy)."""
    payload = {}
    for field in settings.FIELDS:
        key = field["key"]
        val = values.get(key, "")
        if field["type"] == "bool":
            if val:
                payload[key] = "on"
        else:
            payload[key] = str(val)
    return payload


@pytest.fixture(autouse=True)
def _patch_config_path(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "gui_config" / "config.json")


# ---------------------------------------------------------------------------
# FIELDS table shape (acceptance criteria)
# ---------------------------------------------------------------------------

def test_fields_table_has_23_entries_and_six_datapaths():
    assert len(settings.FIELDS) == 23
    datapath_keys = {f["key"] for f in settings.FIELDS if f["is_data_path"]}
    assert datapath_keys == {
        "registry_path", "vault_path", "chroma_db_path",
        "mineru_output_dir", "paperjson_cache_dir", "uningested_dir",
    }


def test_save_writes_via_tmp_file_and_os_replace():
    src = inspect.getsource(settings)
    assert ".tmp" in src
    assert "os.replace" in src


# ---------------------------------------------------------------------------
# Test 1: GET /settings renders 23 typed fields
# ---------------------------------------------------------------------------

def test_get_settings_renders_typed_inputs(gui_client):
    resp = gui_client.get("/settings")
    assert resp.status_code == 200
    assert 'name="embed_batch_size"' in resp.text
    assert 'type="number"' in resp.text
    assert 'name="crossref_validate"' in resp.text
    assert 'type="checkbox"' in resp.text
    assert "requires GUI restart" in resp.text
    assert "gui_port" in resp.text


def test_get_settings_has_no_dead_obsidian_exe_field(gui_client):
    """obsidian_exe was removed as dead config (round-2 gap closure,
    GUI-07) -- the field must no longer be rendered as an editable
    Settings input."""
    resp = gui_client.get("/settings")
    assert resp.status_code == 200
    assert 'name="obsidian_exe"' not in resp.text
    assert "obsidian_exe" not in resp.text


def test_legacy_config_still_carrying_obsidian_exe_loads_and_renders(gui_client, gui_config):
    """T-08-GAP-51 back-compat case: an existing clone's on-disk config.json
    still carries the removed obsidian_exe key after upgrade. The GUI must
    ignore the extra key, not reject it -- both _load_config (plain
    json.load, no schema validation) and this route's dict-merge loader
    tolerate unknown keys."""
    base = _config_dict(gui_config, obsidian_exe="/some/old/path/to/Obsidian.exe")
    settings.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.CONFIG_PATH.write_text(json.dumps(base), encoding="utf-8")

    resp = gui_client.get("/settings")

    assert resp.status_code == 200
    assert 'name="obsidian_exe"' not in resp.text


# ---------------------------------------------------------------------------
# Test 2: POST a changed non-path value writes atomically, others preserved
# ---------------------------------------------------------------------------

def test_post_changed_nonpath_value_writes_config(gui_client, gui_config):
    base = _config_dict(gui_config)
    settings.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.CONFIG_PATH.write_text(json.dumps(base), encoding="utf-8")

    values = dict(base, embed_batch_size=32)

    resp = gui_client.post("/settings", data=_form_data(values))

    assert resp.status_code == 200
    saved = json.loads(settings.CONFIG_PATH.read_text(encoding="utf-8"))
    assert saved["embed_batch_size"] == 32
    assert isinstance(saved["embed_batch_size"], int)
    assert saved["embed_model"] == values["embed_model"]
    assert saved["ollama_num_ctx_cap"] == values["ollama_num_ctx_cap"]


# ---------------------------------------------------------------------------
# Test 3 + threat_model prohibition verification (exact test name pinned by
# 08-04-PLAN.md's threat_model table): registry_path change with a
# non-empty old registry requires confirm; confirming applies the write;
# neither location is ever migrated/copied.
# ---------------------------------------------------------------------------

def test_datapath_change_never_touches_data(gui_client, gui_config, tmp_path):
    old_registry = tmp_path / "old_registry.json"
    old_bytes = json.dumps({"k1": {"title": "a"}, "k2": {"title": "b"}}).encode("utf-8")
    old_registry.write_bytes(old_bytes)

    base = _config_dict(gui_config, registry_path=str(old_registry))
    settings.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.CONFIG_PATH.write_text(json.dumps(base), encoding="utf-8")

    new_registry = tmp_path / "new_registry.json"
    changed = dict(base, registry_path=str(new_registry))

    resp = gui_client.post("/settings", data=_form_data(changed))

    assert resp.status_code == 200
    assert "Change Anyway" in resp.text
    assert "Keep Current Path" in resp.text
    assert "2 item(s)" in resp.text
    assert "the GUI only repoints the config value" in resp.text

    # No write happened yet -- config.json still points at the old registry.
    saved_before = json.loads(settings.CONFIG_PATH.read_text(encoding="utf-8"))
    assert saved_before["registry_path"] == str(old_registry)

    # Confirm -- resubmit the same values plus confirmed=true.
    confirm_data = _form_data(changed)
    confirm_data["confirmed"] = "true"
    resp2 = gui_client.post("/settings", data=confirm_data)

    assert resp2.status_code == 200
    saved_after = json.loads(settings.CONFIG_PATH.read_text(encoding="utf-8"))
    assert saved_after["registry_path"] == str(new_registry)

    # D-20: the GUI never migrates data -- old file byte-identical, new
    # location was never created by the GUI itself.
    assert old_registry.read_bytes() == old_bytes
    assert not new_registry.exists()


# ---------------------------------------------------------------------------
# Test 5: data-path change whose OLD location does not exist writes
# immediately -- no confirm needed.
# ---------------------------------------------------------------------------

def test_datapath_change_with_empty_old_location_writes_immediately(gui_client, gui_config, tmp_path):
    base = _config_dict(gui_config, vault_path=str(tmp_path / "old_vault_never_created"))
    settings.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.CONFIG_PATH.write_text(json.dumps(base), encoding="utf-8")

    new_vault = tmp_path / "new_vault"
    changed = dict(base, vault_path=str(new_vault))

    resp = gui_client.post("/settings", data=_form_data(changed))

    assert resp.status_code == 200
    assert "Change Anyway" not in resp.text
    saved = json.loads(settings.CONFIG_PATH.read_text(encoding="utf-8"))
    assert saved["vault_path"] == str(new_vault)
    assert not new_vault.exists()  # D-20: repoint only, never create/migrate


# ---------------------------------------------------------------------------
# Test 6: type coercion failure re-renders with an error, writes nothing.
# ---------------------------------------------------------------------------

def test_post_invalid_number_shows_error_and_writes_nothing(gui_client, gui_config):
    base = _config_dict(gui_config)
    settings.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.CONFIG_PATH.write_text(json.dumps(base), encoding="utf-8")

    changed = dict(base, embed_batch_size="not-a-number")

    resp = gui_client.post("/settings", data=_form_data(changed))

    assert resp.status_code == 200
    assert "must be a whole number" in resp.text
    saved = json.loads(settings.CONFIG_PATH.read_text(encoding="utf-8"))
    assert saved["embed_batch_size"] == base["embed_batch_size"]


# ---------------------------------------------------------------------------
# Setup section (D-05 reuse of /jobs/enqueue, no second execution path)
# ---------------------------------------------------------------------------

def test_settings_page_has_run_full_setup_action(gui_client):
    resp = gui_client.get("/settings")
    assert resp.status_code == 200
    assert "Run Full Setup" in resp.text
    assert 'hx-post="/jobs/enqueue"' in resp.text
    assert '"action": "setup"' in resp.text
