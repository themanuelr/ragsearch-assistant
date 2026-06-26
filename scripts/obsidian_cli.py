"""
Obsidian vault I/O chokepoint — direct atomic file write (D-01, D-02, 02-06).

All vault writes in this project go through this module.  Notes are written
as byte-exact UTF-8 .md files directly under vault_root (resolved from
config["vault_path"]).  Obsidian's file watcher auto-indexes new or changed
files instantly, so a direct atomic write is functionally equivalent to an
obsidian-cli ``create`` command — without any running-instance dependency,
subprocess invocation, or argv serialization overhead.

This module remains the single sanctioned vault-I/O chokepoint (D-01, D-02);
only the mechanism changes: direct file I/O instead of shelling out to
obsidian.COM.  (obsidian.COM passed content as an argv token, which Obsidian
serialized to JSON; a ~6.7 KB note body with ~80 newlines + LaTeX backslashes
corrupted that JSON → main-process crash → 30 s timeout, note never written.)

Functions:
  preflight     — filesystem check: does the vault root exist as a directory?
  note_exists   — does a note file exist at the given vault-relative path?
  create_note   — write a note to the vault (atomic temp-file + os.replace,
                  UTF-8, LF, byte-exact — no escaping, no subprocess)
"""

import datetime
import os
import pathlib
import sys


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Emit a timestamped progress line to stderr."""
    print(
        f"[note {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _vault_root(config: dict) -> pathlib.Path:
    """Resolve the absolute vault root from config["vault_path"].

    vault_path may be relative (e.g. "./.local/ragsearch_dev_vault") —
    resolved relative to the repo root (parent of this module's directory).
    Absolute paths are used as-is.
    """
    vault_path = config.get("vault_path", "")
    p = pathlib.Path(vault_path)
    if not p.is_absolute():
        repo_root = pathlib.Path(__file__).parent.parent
        p = repo_root / p
    return p.resolve()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preflight(config: dict) -> bool:
    """
    Filesystem check: does the vault root exist and is it a directory? (D-03)

    Returns True only when vault_path is set in config AND vault_root is a real
    directory on disk.  No running-Obsidian dependency, no subprocess, no
    30-second timeout.  On any error (including missing vault_path) returns False
    — the caller should emit a ``[note error: vault root does not exist ...]``
    message and abort.
    """
    if not config.get("vault_path"):
        return False
    try:
        return _vault_root(config).is_dir()
    except Exception:
        return False


def note_exists(path: str, config: dict) -> bool:
    """
    Check whether a note exists at the given vault-relative path (filesystem).

    Returns True when <vault_root>/<path> exists on disk, False otherwise
    (including on any exception).
    """
    try:
        return (_vault_root(config) / path).exists()
    except Exception:
        return False


def create_note(
    path: str,
    content: str,
    config: dict,
    overwrite: bool = False,
) -> str:
    """
    Write a note to the vault at the given vault-relative path (D-04, T-02-06a/b).

    Content is written byte-exact as UTF-8 with LF newlines via a temp file +
    os.replace (atomic).  No subprocess, no shell, no argv escaping — LaTeX
    backslashes (``\\text``, ``\\Delta``) and raw newlines are preserved exactly
    as-is.  This eliminates the JSON-corruption crash that the obsidian.COM argv
    path caused on notes exceeding ~30 KB or containing special characters.

    When overwrite=False and the target already exists, the file is not modified
    and the existing absolute path is returned.  When overwrite=True the target
    is replaced atomically.

    Security:
        - Rejects path traversal: ``..`` segments, leading ``/`` or ``\\``
          (T-02-03, T-02-06a).
        - Write is constrained under the resolved vault_root.

    Args:
        path:      Vault-relative path (e.g. ``"Papers/My Paper.md"``).
        content:   Note content as a Python str.
        config:    Config dict containing at minimum ``"vault_path"``.
        overwrite: Whether to replace an existing note (default False).

    Returns:
        Absolute path string to the written (or already-existing) note file.

    Raises:
        ValueError: When path contains traversal sequences.
    """
    # Security: path-traversal guard (T-02-03, T-02-06a)
    path_obj = pathlib.Path(path)
    if ".." in path_obj.parts or path.startswith("/") or path.startswith("\\"):
        raise ValueError(
            f"[obsidian_cli error: invalid path '{path}' -- path traversal rejected]"
        )

    root = _vault_root(config)
    target = root / path

    # Honour overwrite=False: skip without clobbering
    if target.exists() and not overwrite:
        _log(f"note already exists at {target} -- skipping (overwrite=False)")
        return str(target)

    # Ensure parent directory exists (e.g. Papers/)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Normalise line endings to LF (byte-exact; no character escaping of any kind)
    content_lf = content.replace("\r\n", "\n").replace("\r", "\n")

    # Atomic write: temp file beside target, then os.replace (D-04, T-02-06a)
    tmp_file = str(target) + ".tmp"
    with open(tmp_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(content_lf)
    os.replace(tmp_file, str(target))

    _log(f"wrote note to {target}")
    return str(target)
