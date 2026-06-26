"""
Unit tests for scripts/obsidian_cli.py — NOTE-03 direct atomic vault write.

Tests exercise preflight, note_exists, create_note, and _vault_root against
a real temp-directory vault (tmp_path).  No subprocess, no live Obsidian,
no GPU required.

Design rationale: The old obsidian.COM argv path corrupted ~6.7 KB notes
(newlines + LaTeX backslashes → invalid JSON → crash).  The new path writes
byte-exact UTF-8 .md files directly under vault_path via atomic
temp-file + os.replace.  These tests verify the critical properties:
  - Byte-exactness (LaTeX characters survive unmangled)
  - Overwrite semantics (False = skip, True = replace)
  - Filesystem-level existence checks
  - Preflight without running-Obsidian dependency
  - Path-traversal guard (T-02-03)

Run with:  python -m pytest tests/test_obsidian_cli.py -x
"""

import pathlib

import pytest

from scripts.obsidian_cli import (
    _vault_root,
    create_note,
    note_exists,
    preflight,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vault_dir(tmp_path):
    """A real temporary vault root directory (Papers/ not yet created)."""
    vault = tmp_path / "test_vault"
    vault.mkdir()
    return vault


@pytest.fixture
def vault_config(vault_dir):
    """Config dict pointing to the real temp vault root (absolute path)."""
    return {"vault_path": str(vault_dir)}


# ---------------------------------------------------------------------------
# NOTE-03: _vault_root resolves relative and absolute paths
# ---------------------------------------------------------------------------

def test_vault_root_resolves_absolute(vault_config, vault_dir):
    """_vault_root returns the absolute vault dir when vault_path is absolute."""
    root = _vault_root(vault_config)
    assert root == vault_dir.resolve(), (
        f"Expected {vault_dir.resolve()}, got {root}"
    )


def test_vault_root_resolves_relative():
    """_vault_root resolves a relative vault_path against the repo root."""
    # Use "./" (current dir relative) — should resolve to an absolute path
    config = {"vault_path": "./"}
    root = _vault_root(config)
    assert root.is_absolute(), f"_vault_root should return an absolute path, got {root}"


# ---------------------------------------------------------------------------
# NOTE-03: preflight — filesystem-based, no Obsidian dependency
# ---------------------------------------------------------------------------

def test_preflight_true_for_existing_dir(vault_config):
    """preflight returns True when vault_path exists as a directory."""
    assert preflight(vault_config) is True, (
        "preflight should return True for an existing vault directory"
    )


def test_preflight_false_for_nonexistent(tmp_path):
    """preflight returns False when vault_path does not exist."""
    config = {"vault_path": str(tmp_path / "does_not_exist")}
    assert preflight(config) is False, (
        "preflight should return False when the vault directory does not exist"
    )


def test_preflight_false_for_empty_vault_path():
    """preflight returns False when vault_path is absent from config."""
    assert preflight({}) is False, (
        "preflight should return False when vault_path is missing from config"
    )


# ---------------------------------------------------------------------------
# NOTE-03: note_exists — filesystem existence check
# ---------------------------------------------------------------------------

def test_note_exists_false_before_write(vault_config):
    """note_exists returns False before the note is written."""
    assert note_exists("Papers/new_note.md", vault_config) is False, (
        "note_exists should be False for a file that hasn't been created yet"
    )


def test_note_exists_true_after_write(vault_config, vault_dir):
    """note_exists returns True after the note is written."""
    create_note("Papers/my_note.md", "# Hello", vault_config)
    assert note_exists("Papers/my_note.md", vault_config) is True, (
        "note_exists should be True after create_note writes the file"
    )


# ---------------------------------------------------------------------------
# NOTE-03: create_note — byte-exact UTF-8 + LF roundtrip
# ---------------------------------------------------------------------------

def test_create_note_byte_exact_latex(vault_config, vault_dir):
    """Content with LaTeX characters roundtrips byte-exact (no mangling)."""
    content = (
        "# Attention Is All You Need\n"
        "\n"
        "The model complexity is $\\mathcal{O}(n^2 \\cdot d)$.\n"
        "\n"
        "Key equation: $\\Delta x = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V$\n"
        "\n"
        "> [!important] Key Contribution\n"
        "> Self-attention replaces recurrence entirely.\n"
    )
    result_path = create_note("Papers/attention.md", content, vault_config)
    written = pathlib.Path(result_path).read_text(encoding="utf-8")
    assert written == content, (
        f"Content roundtrip failed — got:\n{written!r}\n\nexpected:\n{content!r}"
    )


def test_create_note_creates_papers_subdir(vault_config, vault_dir):
    """create_note creates the Papers/ subdirectory automatically."""
    papers_dir = vault_dir / "Papers"
    assert not papers_dir.exists(), "Papers/ should not pre-exist"
    create_note("Papers/foo.md", "hello", vault_config)
    assert papers_dir.is_dir(), "create_note should create Papers/ if missing"


def test_create_note_returns_absolute_path(vault_config, vault_dir):
    """create_note returns an absolute path string to the written file."""
    result = create_note("Papers/bar.md", "# Bar", vault_config)
    assert pathlib.Path(result).is_absolute(), (
        f"create_note should return an absolute path, got {result!r}"
    )
    assert pathlib.Path(result).exists(), (
        "create_note return value should point to the existing file"
    )


# ---------------------------------------------------------------------------
# NOTE-03: overwrite semantics
# ---------------------------------------------------------------------------

def test_create_note_overwrite_false_does_not_clobber(vault_config, vault_dir):
    """overwrite=False: existing note is NOT modified."""
    create_note("Papers/existing.md", "original content", vault_config)
    create_note("Papers/existing.md", "new content", vault_config, overwrite=False)
    written = (vault_dir / "Papers" / "existing.md").read_text(encoding="utf-8")
    assert written == "original content", (
        "overwrite=False should not clobber an existing note"
    )


def test_create_note_overwrite_true_replaces(vault_config, vault_dir):
    """overwrite=True: existing note is atomically replaced."""
    create_note("Papers/existing.md", "original content", vault_config)
    create_note("Papers/existing.md", "replaced content", vault_config, overwrite=True)
    written = (vault_dir / "Papers" / "existing.md").read_text(encoding="utf-8")
    assert written == "replaced content", (
        "overwrite=True should replace an existing note"
    )


def test_create_note_overwrite_false_on_new_writes(vault_config, vault_dir):
    """overwrite=False: a new (non-existing) note is still written."""
    create_note("Papers/brand_new.md", "fresh content", vault_config, overwrite=False)
    written = (vault_dir / "Papers" / "brand_new.md").read_text(encoding="utf-8")
    assert written == "fresh content", (
        "overwrite=False should write a new note when it does not yet exist"
    )


# ---------------------------------------------------------------------------
# NOTE-03: path-traversal guard (T-02-03, T-02-06a)
# ---------------------------------------------------------------------------

def test_create_note_path_traversal_dotdot_rejected(vault_config):
    """create_note raises ValueError when path contains '..'."""
    with pytest.raises(ValueError, match="path traversal"):
        create_note("../outside.md", "evil", vault_config)


def test_create_note_path_traversal_leading_slash_rejected(vault_config):
    """create_note raises ValueError when path starts with '/'."""
    with pytest.raises(ValueError, match="path traversal"):
        create_note("/etc/passwd", "evil", vault_config)


def test_note_exists_traversal_dotdot_returns_false(vault_config):
    """note_exists with '..' in path returns False (does not error or escape)."""
    # note_exists is best-effort; it should not raise, just return False
    result = note_exists("../outside.md", vault_config)
    assert result is False, (
        "note_exists should return False for traversal paths, not raise"
    )
