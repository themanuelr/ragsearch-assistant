"""
Wave-0 FAILING test contract for Phase 7 clone-and-go bootstrap (SETUP-01/02/04).

scripts/setup.py and obsidian-vault-template/ do not exist yet -- these tests
pin the exact behavior Plans 02-04 must implement (see 07-01-PLAN.md,
07-RESEARCH.md "Code Examples", 07-VALIDATION.md "Per-Task Verification Map"):
  - test_bootstrap_vault_creates_folders (SETUP-01/04): Papers/Stubs/Topics +
    _Papers.base created, Reviews/ dropped (D-06)
  - test_bootstrap_vault_renames_dot_obsidian (SETUP-04): dot_obsidian/ ->
    .obsidian/ rename-on-copy (D-04, Pitfall 6)
  - test_bootstrap_vault_idempotent (SETUP-01, D-13): second run never
    clobbers existing vault content (Pitfall 2)
  - test_resolve_safe_target_rejects_sensitive (T-07-01, V5): repo root /
    $HOME rejected as a bootstrap target
  - test_init_chroma_reuses_get_collection (D-11): delegates to
    embed._get_collection, no second PersistentClient
  - test_model_present_tag_normalization (Pitfall 1): ":latest" suffix
    normalization
  - test_pull_models_fail_open_unreachable (D-14): fail-open, never raises
  - test_write_config_never_overwrites (D-13/T-07-03): never clobbers an
    existing config.json
  - test_ensure_empty_registry_never_overwrites (D-12/D-13): never clobbers
    an existing papers_registry.json; creates "{}" when absent
  - test_check_external_tools_warns_not_fails (D-10): MinerU/defuddle absence
    is a warning, never a hard failure
  - test_config_example_matches_write_config_keys (SETUP-02): config.example.json
    key-set stays in sync with what write_config actually writes (will only go
    GREEN once Plan 03 drops the legacy vault_name key per D-15)
  - test_template_core_plugins_enables_bases (SETUP-04, Pitfall 5): shipped
    core-plugins.json has "bases": true
  - test_template_papers_base_matches_link_content (D-07/D-08): shipped
    _Papers.base is byte-identical to scripts.link._PAPERS_BASE_CONTENT
  - test_template_excludes_dev_session_state (Pitfall 4, D-06/D-07): no
    workspace.json/workspace-mobile.json/graph.json, no Reviews/, no _Index.md

`scripts.setup` (and any other scripts.* module) is imported at FUNCTION level
(never module top) so pytest collection succeeds cleanly before the module
exists -- mirrors the Phase 1.3 / 04-01 attribute-access / function-level-import
precedent (tests/test_ingest.py, tests/test_biblio.py: `from scripts import
ingest  # noqa: PLC0415`).

Every test injects vault_path/registry_path/chroma_db_path under tmp_path (via
the shared `setup_config` fixture) so no test ever writes to the real repo-root
config.json/papers_registry.json/vault -- guarded at session scope by
tests/conftest.py::_guard_real_config_json (this plan's Task 2) plus the
pre-existing _guard_real_vault_root/_guard_real_chroma_db/_guard_real_paperjson_cache.

Run with: python -m pytest tests/test_setup.py -x
"""

import json
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def setup_config(tmp_path):
    """Config dict whose vault_path/registry_path/chroma_db_path all point
    under tmp_path, so no setup.py test can touch the real repo root."""
    return {
        "vault_path": str(tmp_path / "vault"),
        "registry_path": str(tmp_path / "registry.json"),
        "chroma_db_path": str(tmp_path / "chroma_db"),
        "project_name": "test-project",
    }


@pytest.fixture
def template_dir():
    """The repo-root obsidian-vault-template/ path (committed asset, SETUP-04)."""
    return _REPO_ROOT / "obsidian-vault-template"


# ---------------------------------------------------------------------------
# bootstrap_vault (SETUP-01 / SETUP-04)
# ---------------------------------------------------------------------------

def test_bootstrap_vault_creates_folders(setup_config):
    """After bootstrap_vault, the tmp vault has Papers/, Stubs/, Topics/ and
    _Papers.base -- and NO Reviews/ directory (D-06 reconciliation)."""
    from scripts import setup  # noqa: PLC0415

    setup.bootstrap_vault(setup_config)

    vault = pathlib.Path(setup_config["vault_path"])
    assert (vault / "Papers").is_dir(), "Expected Papers/ to be created"
    assert (vault / "Stubs").is_dir(), "Expected Stubs/ to be created"
    assert (vault / "Topics").is_dir(), "Expected Topics/ to be created"
    assert (vault / "_Papers.base").exists(), "Expected _Papers.base to be created"
    assert not (vault / "Reviews").exists(), (
        "Reviews/ must NOT be created (D-06 -- dropped, aspirational v2 REVIEW-01 folder)"
    )


def test_bootstrap_vault_renames_dot_obsidian(setup_config):
    """After bootstrap, the tmp vault has .obsidian/ (dotted) with
    core-plugins.json inside, and NO dot_obsidian/ directory (D-04 rename-on-copy,
    Pitfall 6)."""
    from scripts import setup  # noqa: PLC0415

    setup.bootstrap_vault(setup_config)

    vault = pathlib.Path(setup_config["vault_path"])
    assert (vault / ".obsidian").is_dir(), "Expected .obsidian/ (dotted) to exist after copy"
    assert (vault / ".obsidian" / "core-plugins.json").exists(), (
        "Expected core-plugins.json inside the renamed .obsidian/"
    )
    assert not (vault / "dot_obsidian").exists(), (
        "dot_obsidian/ must be renamed to .obsidian/ on copy, never left as-is"
    )


def test_bootstrap_vault_idempotent(setup_config):
    """Writing a sentinel into <vault>/.obsidian/appearance.json and into
    _Papers.base, then running bootstrap_vault a SECOND time, must preserve
    both sentinels unchanged (D-13 skip-existing-never-overwrite, Pitfall 2)."""
    from scripts import setup  # noqa: PLC0415

    setup.bootstrap_vault(setup_config)

    vault = pathlib.Path(setup_config["vault_path"])
    appearance_path = vault / ".obsidian" / "appearance.json"
    papers_base_path = vault / "_Papers.base"

    appearance_sentinel = '{"sentinel": "user-edited-appearance-do-not-touch"}'
    papers_base_sentinel = "# sentinel user-edited papers base -- do not touch\n"
    appearance_path.write_text(appearance_sentinel, encoding="utf-8")
    papers_base_path.write_text(papers_base_sentinel, encoding="utf-8")

    setup.bootstrap_vault(setup_config)

    assert appearance_path.read_text(encoding="utf-8") == appearance_sentinel, (
        "Second bootstrap_vault run must not overwrite an existing .obsidian/appearance.json"
    )
    assert papers_base_path.read_text(encoding="utf-8") == papers_base_sentinel, (
        "Second bootstrap_vault run must not overwrite an existing _Papers.base"
    )


# ---------------------------------------------------------------------------
# _resolve_safe_target (T-07-01, V5 path-safety guard)
# ---------------------------------------------------------------------------

def test_resolve_safe_target_rejects_sensitive(tmp_path):
    """_resolve_safe_target rejects the repo root and $HOME as bootstrap
    targets (safe=False, non-empty reason); a normal tmp_path target is
    accepted (safe=True)."""
    from scripts import setup  # noqa: PLC0415

    repo_result = setup._resolve_safe_target(str(_REPO_ROOT))
    assert repo_result["safe"] is False, "The repo root itself must never be a safe bootstrap target"
    assert repo_result["reason"], "Expected a non-empty reason when rejecting the repo root"

    home_result = setup._resolve_safe_target(str(pathlib.Path.home()))
    assert home_result["safe"] is False, "$HOME must never be a safe bootstrap target"
    assert home_result["reason"], "Expected a non-empty reason when rejecting $HOME"

    safe_target = tmp_path / "my-new-vault"
    safe_result = setup._resolve_safe_target(str(safe_target))
    assert safe_result["safe"] is True, "A normal tmp_path target must be accepted as safe"


# ---------------------------------------------------------------------------
# init_chroma (D-11 delegation, no second PersistentClient)
# ---------------------------------------------------------------------------

def test_init_chroma_reuses_get_collection(monkeypatch, setup_config):
    """init_chroma delegates to scripts.embed._get_collection(config) rather
    than constructing its own chromadb.PersistentClient (D-11)."""
    from scripts import setup  # noqa: PLC0415
    import scripts.embed as embed_mod  # noqa: PLC0415

    sentinel_collection = object()
    calls = []

    def fake_get_collection(config):
        calls.append(config)
        return sentinel_collection

    monkeypatch.setattr(embed_mod, "_get_collection", fake_get_collection)

    result = setup.init_chroma(setup_config)

    assert calls == [setup_config], "Expected embed._get_collection to be invoked with the config"
    assert result["status"] == "ok", f"Expected status 'ok', got {result!r}"


# ---------------------------------------------------------------------------
# Ollama model presence / pull (Pitfall 1, D-09, D-13, D-14)
# ---------------------------------------------------------------------------

def test_model_present_tag_normalization():
    """_model_present normalizes the ':tag' suffix so a bare configured name
    correctly matches a tagged installed name (Pitfall 1)."""
    from scripts import setup  # noqa: PLC0415

    assert setup._model_present("nomic-embed-text", ["nomic-embed-text:latest"]) is True
    assert setup._model_present("gemma4:e4b", ["gemma4:e4b"]) is True
    assert setup._model_present("llama3", ["nomic-embed-text:latest", "gemma4:e4b"]) is False


def test_pull_models_fail_open_unreachable(monkeypatch, setup_config):
    """When Ollama is unreachable (_ollama_model_names returns None),
    pull_models returns a structured error result and does NOT raise
    (D-14 fail-open)."""
    from scripts import setup  # noqa: PLC0415

    monkeypatch.setattr(setup, "_ollama_model_names", lambda: None)

    result = setup.pull_models(setup_config, ["nomic-embed-text", "gemma4:e4b"])

    assert result["status"] == "error", f"Expected status 'error' on unreachable Ollama, got {result!r}"


# ---------------------------------------------------------------------------
# write_config / ensure_empty_registry (D-12/D-13, T-07-01/T-07-03)
# ---------------------------------------------------------------------------

def test_write_config_never_overwrites(tmp_path):
    """write_config never overwrites an existing config.json; returns status
    'skipped' (D-13 security invariant, T-07-01/T-07-03)."""
    from scripts import setup  # noqa: PLC0415

    config_path = tmp_path / "config.json"
    sentinel_payload = '{"sentinel": "do-not-touch"}'
    config_path.write_text(sentinel_payload, encoding="utf-8")

    result = setup.write_config({"vault_path": str(tmp_path / "vault")}, config_path)

    assert config_path.read_text(encoding="utf-8") == sentinel_payload, (
        "write_config must never overwrite an existing config.json"
    )
    assert result["status"] == "skipped", f"Expected status 'skipped', got {result!r}"


def test_ensure_empty_registry_never_overwrites(tmp_path):
    """ensure_empty_registry never overwrites an existing papers_registry.json
    (status 'skipped'); on a fresh path it creates a file whose content parses
    as an empty dict (D-12/D-13)."""
    from scripts import setup  # noqa: PLC0415

    registry_path = tmp_path / "papers_registry.json"
    sentinel_payload = '{"sentinel-key": {"title": "do not touch"}}'
    registry_path.write_text(sentinel_payload, encoding="utf-8")

    result = setup.ensure_empty_registry(registry_path)

    assert registry_path.read_text(encoding="utf-8") == sentinel_payload, (
        "ensure_empty_registry must never overwrite an existing papers_registry.json"
    )
    assert result["status"] == "skipped", f"Expected status 'skipped', got {result!r}"

    fresh_path = tmp_path / "fresh_registry.json"
    fresh_result = setup.ensure_empty_registry(fresh_path)

    assert fresh_path.exists(), "Expected a fresh papers_registry.json to be created"
    assert json.loads(fresh_path.read_text(encoding="utf-8")) == {}, (
        "A freshly created papers_registry.json must parse as an empty dict"
    )
    assert fresh_result["status"] == "ok", f"Expected status 'ok' on fresh creation, got {fresh_result!r}"


# ---------------------------------------------------------------------------
# check_external_tools (D-10 warn-only)
# ---------------------------------------------------------------------------

def test_check_external_tools_warns_not_fails(monkeypatch, setup_config):
    """When MinerU and defuddle are both unresolvable, check_external_tools
    returns status 'warning' with a non-empty detail list and does NOT raise
    (D-10 warn-only)."""
    from scripts import setup  # noqa: PLC0415
    import scripts.ingest as ingest_mod  # noqa: PLC0415

    monkeypatch.setattr(ingest_mod, "_resolve_mineru", lambda config: None)
    monkeypatch.setattr(ingest_mod, "_resolve_defuddle", lambda config: None)

    result = setup.check_external_tools(setup_config)

    assert result["status"] == "warning", f"Expected status 'warning', got {result!r}"
    assert result["detail"], "Expected a non-empty detail list when both tools are absent"


# ---------------------------------------------------------------------------
# config.example.json <-> write_config key-set sync (SETUP-02)
# ---------------------------------------------------------------------------

def test_config_example_matches_write_config_keys(tmp_path):
    """config.example.json's documented key-set must match the key-set
    write_config actually writes to config.json (SETUP-02 sync). The
    representative values dict below excludes the legacy 'vault_name' key
    per D-15 -- this test only goes GREEN once Plan 03 drops 'vault_name'
    from config.example.json to match."""
    from scripts import setup  # noqa: PLC0415

    config_example_path = _REPO_ROOT / "config.example.json"
    example_keys = set(json.loads(config_example_path.read_text(encoding="utf-8")).keys())

    representative_values = {
        "registry_path": str(tmp_path / "registry.json"),
        "vault_path": str(tmp_path / "vault"),
        "project_name": "test-project",
        "mineru_path": "",
        "mineru_timeout": 1800,
        "defuddle_path": "",
        "web_min_body_chars": 2000,
        "defuddle_timeout": 60,
        "chroma_db_path": str(tmp_path / "chroma_db"),
        "mineru_output_dir": str(tmp_path / "mineru_output"),
        "paperjson_cache_dir": str(tmp_path / "paperjson_cache"),
        "embed_model": "nomic-embed-text",
        "embed_timeout": 60,
        "embed_batch_size": 16,
        "embed_section_max_tokens": 2000,
        "crossref_validate": False,
        "crossref_contact_email": "",
        "crossref_timeout": 30,
        "crossref_retries": 1,
        "ollama_num_ctx_cap": 65536,
        "ollama_section_timeout": 300,
        "gui_port": 8765,
        "uningested_dir": "./uningestedPDFs",
    }

    target = tmp_path / "config.json"
    setup.write_config(representative_values, target)
    written_keys = set(json.loads(target.read_text(encoding="utf-8")).keys())

    assert written_keys == example_keys, (
        f"config.example.json keys {sorted(example_keys)} do not match "
        f"write_config's written keys {sorted(written_keys)} "
        "(a new config key must be added to both config.example.json and "
        "this representative_values dict)"
    )


# ---------------------------------------------------------------------------
# obsidian-vault-template/ static content (SETUP-04)
# ---------------------------------------------------------------------------

def test_template_core_plugins_enables_bases(template_dir):
    """The shipped core-plugins.json has 'bases': true -- Bases is a core
    plugin NOT enabled by default (Pitfall 5, SC4 load-bearing)."""
    core_plugins_path = template_dir / "dot_obsidian" / "core-plugins.json"
    data = json.loads(core_plugins_path.read_text(encoding="utf-8"))

    assert data.get("bases") is True, (
        "Shipped core-plugins.json must set \"bases\": true or _Papers.base "
        "will not render as a table on a fresh clone (Pitfall 5)"
    )


def test_template_papers_base_matches_link_content(template_dir):
    """The shipped _Papers.base is byte-identical to scripts.link's
    _PAPERS_BASE_CONTENT, so the template never drifts from what
    ensure_papers_base would write at runtime (D-07/D-08)."""
    from scripts.link import _PAPERS_BASE_CONTENT  # noqa: PLC0415

    template_base_path = template_dir / "_Papers.base"
    assert template_base_path.read_text(encoding="utf-8") == _PAPERS_BASE_CONTENT, (
        "Shipped _Papers.base must be byte-identical to scripts.link._PAPERS_BASE_CONTENT"
    )


def test_template_excludes_dev_session_state(template_dir):
    """The shipped template excludes dev-session-specific Obsidian UI state
    (workspace.json, workspace-mobile.json, graph.json -- Pitfall 4) and the
    deferred Reviews/ folder and unused _Index.md landing file (D-06/D-07)."""
    dot_obsidian = template_dir / "dot_obsidian"

    assert dot_obsidian.is_dir(), (
        "Expected obsidian-vault-template/dot_obsidian/ to exist -- without it "
        "the exclusion checks below are vacuously true and prove nothing"
    )
    assert not (dot_obsidian / "workspace.json").exists(), (
        "workspace.json leaks dev-session tabs/lastOpenFiles -- must not be shipped (Pitfall 4)"
    )
    assert not (dot_obsidian / "workspace-mobile.json").exists(), (
        "workspace-mobile.json leaks dev-session state -- must not be shipped (Pitfall 4)"
    )
    assert not (dot_obsidian / "graph.json").exists(), (
        "graph.json is dev-session UI state -- must not be shipped (Pitfall 4)"
    )
    assert not (template_dir / "Reviews").exists(), (
        "Reviews/ is a deferred v2 folder -- must not be shipped (D-06)"
    )
    assert not (template_dir / "_Index.md").exists(), (
        "_Index.md was never built and is not the landing file -- _Papers.base is (D-07)"
    )
