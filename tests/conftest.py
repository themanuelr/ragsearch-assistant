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
"""

import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_REAL_CACHE_DIR = _REPO_ROOT / ".paperjson_cache"
_REAL_CHROMA_DIR = _REPO_ROOT / "chroma_db"


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
