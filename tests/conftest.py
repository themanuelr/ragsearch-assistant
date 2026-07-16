"""Shared pytest fixtures for the ingest test suite.

Todo T-rie-03 (2026-07-05-tests-pollute-real-paperjson-cache): guard against test
runs writing stray files into the real repo-root ``.paperjson_cache/`` dev cache.
Every ingest test config should point ``paperjson_cache_dir`` at ``tmp_path`` (see
``_make_ingest_config`` / ``_make_ingest_config_with_probe`` in tests/test_ingest.py);
this session-scoped fixture is the safety net that fails loudly if a future test
forgets to do so.

Phase 5 (Common Pitfall #8, todo 2026-06-25 / quick-task 260705-rie precedent):
the same class of bug applies to the real repo-root ``chroma_db/`` ChromaDB store.
Every tests/test_embed.py test should point ``chroma_db_path`` at ``tmp_path`` (see
``_make_embed_config``); ``_guard_real_chroma_db`` below is the safety net.

Phase 6 (06-03, GRAPH-03): the same class of bug applies to the vault root itself.
``scripts/link.py::_vault_root`` resolves a missing/empty ``vault_path`` to the
repo root, so any ingest test whose config omits ``vault_path`` and whose paper
note write is real (not mocked) risks writing a stray ``_Papers.base`` (or
``Papers/``/``Topics/`` note) straight into the repo root. Every test config that
reaches the link tail stage should point ``vault_path`` at ``tmp_path``;
``_guard_real_vault_root`` below is the safety net.

Phase 7 (07-01, Wave 0 Requirements): the same class of bug applies to the
repo-root ``config.json`` / ``papers_registry.json`` that ``scripts/setup.py``'s
tests exercise directly. Every ``tests/test_setup.py`` test should point
``config_path`` / ``registry_path`` at ``tmp_path``; ``_guard_real_config_json``
below is the safety net -- this is the highest-risk area in the suite for a
stray real config/registry write, since ``setup.py``'s whole job is creating
these two files.
"""

import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_REAL_CACHE_DIR = _REPO_ROOT / ".paperjson_cache"
_REAL_CHROMA_DIR = _REPO_ROOT / "chroma_db"
_REAL_PAPERS_BASE = _REPO_ROOT / "_Papers.base"
_REAL_PAPERS_DIR = _REPO_ROOT / "Papers"
_REAL_TOPICS_DIR = _REPO_ROOT / "Topics"
_REAL_CONFIG_JSON = _REPO_ROOT / "config.json"
_REAL_PAPERS_REGISTRY_JSON = _REPO_ROOT / "papers_registry.json"


def _snapshot_cache_files():
    """Return the set of *.json filenames currently in the real repo-root cache dir."""
    if not _REAL_CACHE_DIR.is_dir():
        return set()
    return {p.name for p in _REAL_CACHE_DIR.glob("*.json")}


def _snapshot_chroma_paths():
    """Return the set of all paths (files + dirs) currently under the real chroma_db/.

    Chroma writes a `chroma.sqlite3` file plus per-collection subdirectories (unlike
    the flat `*.json` glob above), so this snapshot walks recursively rather than
    globbing a single extension.
    """
    if not _REAL_CHROMA_DIR.is_dir():
        return set()
    return {
        p.relative_to(_REAL_CHROMA_DIR).as_posix()
        for p in _REAL_CHROMA_DIR.rglob("*")
    }


@pytest.fixture(scope="session", autouse=True)
def _guard_real_paperjson_cache():
    """Fail the test session if any NEW file lands in the real .paperjson_cache/.

    Pre-existing files (e.g. legit test_manuel*.json dev caches) are captured in the
    `before` snapshot and never trip this guard -- only files created during the run
    (test-pollution regressions) do.
    """
    before = _snapshot_cache_files()
    yield
    after = _snapshot_cache_files()
    new_files = after - before
    assert not new_files, (
        "Test run polluted the real repo-root .paperjson_cache/ with new file(s): "
        f"{sorted(new_files)}. Point the offending test's config at "
        "'paperjson_cache_dir': str(tmp_path / '.paperjson_cache') instead."
    )


@pytest.fixture(scope="session", autouse=True)
def _guard_real_chroma_db():
    """Fail the test session if any NEW path lands in the real repo-root chroma_db/.

    Pre-existing paths (a dev-machine chroma_db/ from real usage, if any) are
    captured in the `before` snapshot and never trip this guard -- only paths
    created during the run (test-pollution regressions) do.
    """
    before = _snapshot_chroma_paths()
    yield
    after = _snapshot_chroma_paths()
    new_paths = after - before
    assert not new_paths, (
        "Test run polluted the real repo-root chroma_db/ with new path(s): "
        f"{sorted(new_paths)}. Point the offending test's config at "
        "'chroma_db_path': str(tmp_path / 'chroma_db') instead."
    )


@pytest.fixture(scope="session", autouse=True)
def _guard_real_vault_root():
    """Fail the test session if a stray _Papers.base/Papers//Topics/ lands at repo root.

    A config dict missing ``vault_path`` resolves to the repo root itself
    (``scripts/obsidian_cli.py::_vault_root``'s empty-string fallback), so any
    ingest test reaching ``scripts/link.py``'s tail stage with a real (unmocked)
    paper-note write and no ``vault_path`` override risks writing vault
    artifacts directly into the repo working tree (06-03 regression class).
    """
    before_base = _REAL_PAPERS_BASE.exists()
    before_papers = _REAL_PAPERS_DIR.exists()
    before_topics = _REAL_TOPICS_DIR.exists()
    yield
    assert before_base or not _REAL_PAPERS_BASE.exists(), (
        "Test run polluted the real repo root with a stray _Papers.base file. "
        "Point the offending test's config at 'vault_path': str(tmp_path / 'vault') instead."
    )
    assert before_papers or not _REAL_PAPERS_DIR.exists(), (
        "Test run polluted the real repo root with a stray Papers/ directory. "
        "Point the offending test's config at 'vault_path': str(tmp_path / 'vault') instead."
    )
    assert before_topics or not _REAL_TOPICS_DIR.exists(), (
        "Test run polluted the real repo root with a stray Topics/ directory. "
        "Point the offending test's config at 'vault_path': str(tmp_path / 'vault') instead."
    )


@pytest.fixture(scope="session", autouse=True)
def _guard_real_config_json():
    """Fail the test session if a stray config.json/papers_registry.json lands
    at repo root.

    ``scripts/setup.py``'s whole job is creating ``config.json`` and
    ``papers_registry.json`` (SETUP-01/D-12/D-13), which makes its tests
    (``tests/test_setup.py``) the highest-risk area in the suite for a real,
    stray write at repo root if a test forgets to redirect ``config_path`` /
    ``registry_path`` at ``tmp_path``. A pre-existing real config/registry
    (e.g. a developer's own clone-local config.json) is captured in the
    ``before`` snapshot and never trips this guard -- only files created
    during the run (test-pollution regressions) do.
    """
    before_config = _REAL_CONFIG_JSON.exists()
    before_registry = _REAL_PAPERS_REGISTRY_JSON.exists()
    yield
    assert before_config or not _REAL_CONFIG_JSON.exists(), (
        "Test run polluted the real repo root with a stray config.json. "
        "Point the offending test's config_path at tmp_path instead."
    )
    assert before_registry or not _REAL_PAPERS_REGISTRY_JSON.exists(), (
        "Test run polluted the real repo root with a stray papers_registry.json. "
        "Point the offending test's registry_path at tmp_path instead."
    )


@pytest.fixture
def gui_config(tmp_path):
    """Tmp-scoped config dict for GUI tests (Phase 8).

    Mirrors the tmp-scoping discipline established for ingest tests (see
    ``_make_ingest_config`` in tests/test_ingest.py) so the session-scoped
    ``_guard_real_*`` fixtures above never trip: every path-bearing key here
    points at a subdirectory of ``tmp_path``, never the real repo root.
    """
    return {
        "registry_path": str(tmp_path / "papers_registry.json"),
        "vault_path": str(tmp_path / "vault"),
        "chroma_db_path": str(tmp_path / "chroma_db"),
        "paperjson_cache_dir": str(tmp_path / ".paperjson_cache"),
        "mineru_output_dir": str(tmp_path / "mineru_output"),
        "uningested_dir": str(tmp_path / "uningestedPDFs"),
        "gui_port": 8765,
    }


@pytest.fixture
def gui_client():
    """FastAPI TestClient over the gui.app ASGI app (Phase 8).

    Imported lazily inside the fixture body (not at module scope) so that
    collecting this conftest.py never fails before gui/app.py exists (Wave 0
    RED task) or for any other test file that doesn't request this fixture.
    """
    from fastapi.testclient import TestClient

    import gui.app

    return TestClient(gui.app.app)
