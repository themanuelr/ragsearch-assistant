"""
Obsidian desktop CLI wrapper — single sanctioned vault-I/O chokepoint (D-01, D-02).

All vault writes in this project go through this module.  The functions build
subprocess argv as a Python list (never shell=True) and detect errors by parsing
stdout for an ``Error:`` prefix (CLI exit code is always 0).

Functions:
  preflight     — fail-fast probe: is Obsidian running / vault reachable?
  note_exists   — does a note exist at the given vault-relative path?
  create_note   — write a note to the vault (create or overwrite)
"""

import datetime
import json
import pathlib
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

DEFAULT_OBSIDIAN_EXE = r"C:\Users\manue\AppData\Local\Programs\Obsidian\obsidian"

# Maximum safe argv-join length before the Windows CreateProcess limit (~32,600)
_MAX_ARGV_LENGTH = 30000


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

def _load_config() -> dict:
    """Read config.json from the repo root (parent of scripts/) with UTF-8 encoding."""
    cfg_path = pathlib.Path(__file__).parent.parent / "config.json"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _resolve_obsidian_exe(config: dict | None = None) -> str:
    """
    Resolve the Obsidian executable path.

    Returns config['obsidian_exe'] if truthy and the path exists, else falls back
    to shutil.which('obsidian'), else DEFAULT_OBSIDIAN_EXE.
    """
    if config is None:
        config = _load_config()
    configured = config.get("obsidian_exe")
    if configured and pathlib.Path(configured).exists():
        return configured
    found = shutil.which("obsidian")
    if found:
        return found
    return DEFAULT_OBSIDIAN_EXE


# ---------------------------------------------------------------------------
# Core subprocess wrapper
# ---------------------------------------------------------------------------

def _obsidian_cli(
    args: list[str],
    vault_name: str | None = None,
    timeout: int = 30,
    config: dict | None = None,
) -> str:
    """
    Run an Obsidian CLI command and return stdout.

    Builds argv as a Python list — never uses shell=True (T-02-02).
    Content is passed only inside the content= token.

    Args:
        args:       CLI subcommand tokens (e.g. ["create", "path=Papers/x.md"]).
        vault_name: Obsidian vault name for vault=<name> targeting.
        timeout:    Subprocess timeout in seconds (default 30).
        config:     Optional config dict; loaded from config.json if None.

    Returns:
        Stripped stdout string.

    Raises:
        subprocess.TimeoutExpired: When the command exceeds the timeout.
        FileNotFoundError: When the Obsidian executable is not found.
    """
    exe = _resolve_obsidian_exe(config)
    argv: list[str] = [exe]
    if vault_name:
        argv.append(f"vault={vault_name}")
    argv.extend(args)

    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preflight(vault_name: str | None = None, config: dict | None = None) -> bool:
    """
    Fail-fast probe: is Obsidian running and the vault reachable? (D-03)

    Runs ``files total`` and returns True only when stdout is all digits
    (indicating a count of files).  On TimeoutExpired / FileNotFoundError
    returns False — caller should emit ``[note error: Obsidian not running /
    vault unreachable ...]`` and exit.
    """
    try:
        out = _obsidian_cli(["files", "total"], vault_name=vault_name, config=config)
        return out.strip().isdigit()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def note_exists(
    path: str,
    vault_name: str | None = None,
    config: dict | None = None,
) -> bool:
    """
    Check whether a note exists at the given vault-relative path.

    Returns False when stdout starts with ``Error:`` (Pitfall 3: CLI exit code
    is always 0, so we parse stdout instead).  Returns True for any non-error
    stdout.
    """
    try:
        out = _obsidian_cli(["file", f"path={path}"], vault_name=vault_name, config=config)
        return not out.startswith("Error:")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def create_note(
    path: str,
    content: str,
    vault_name: str | None = None,
    overwrite: bool = False,
    config: dict | None = None,
) -> str:
    """
    Write a note to the vault at the given path (D-04).

    Content is passed inside the ``content=`` token — never interpolated into a
    shell string (T-02-02).  When ``overwrite=True`` the ``overwrite`` token is
    appended so the CLI replaces an existing note.

    Emits a length warning to stderr if the joined argv exceeds ~30KB (Pitfall 1:
    Windows CreateProcess ~32,600-char limit).  Full chunking is deferred to
    Plan 02 only if a real paper trips it.

    Returns:
        Stripped stdout string from the CLI.
    """
    args = ["create", f"path={path}", f"content={content}"]
    if overwrite:
        args.append("overwrite")

    # Length guard (Pitfall 1)
    total_len = sum(len(a) for a in args)
    if total_len > _MAX_ARGV_LENGTH:
        _log(
            f"[note warning: note exceeds ~30KB ({total_len} chars) "
            "-- chunked write may be required]"
        )

    return _obsidian_cli(args, vault_name=vault_name, config=config)
