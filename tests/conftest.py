"""Shared pytest fixtures for the ingest test suite.

Todo T-rie-03 (2026-07-05-tests-pollute-real-paperjson-cache): guard against test
runs writing stray files into the real repo-root ``.paperjson_cache/`` dev cache.
Every ingest test config should point ``paperjson_cache_dir`` at ``tmp_path`` (see
``_make_ingest_config`` / ``_make_ingest_config_with_probe`` in tests/test_ingest.py);
this session-scoped fixture is the safety net that fails loudly if a future test
forgets to do so.
"""

import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_REAL_CACHE_DIR = _REPO_ROOT / ".paperjson_cache"


def _snapshot_cache_files():
    """Return the set of *.json filenames currently in the real repo-root cache dir."""
    if not _REAL_CACHE_DIR.is_dir():
        return set()
    return {p.name for p in _REAL_CACHE_DIR.glob("*.json")}


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
