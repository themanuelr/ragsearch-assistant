"""
Unit tests for scripts/obsidian_cli.py — NOTE-03 Obsidian-CLI wrapper.

Tests exercise _obsidian_cli, preflight, note_exists, and create_note with
subprocess.run mocked.  No live Obsidian instance or GPU required.

Run with:  python -m pytest tests/test_obsidian_cli.py -x
"""

import subprocess
from unittest import mock

import pytest

from scripts.obsidian_cli import (
    _obsidian_cli,
    create_note,
    note_exists,
    preflight,
)


# ---------------------------------------------------------------------------
# NOTE-03: args passed as list — no shell=True (T-02-02)
# ---------------------------------------------------------------------------

def test_args_passed_as_list():
    """subprocess.run receives a list argv; shell is NOT True."""
    mock_result = mock.MagicMock()
    mock_result.stdout = "ok"
    with mock.patch("scripts.obsidian_cli.subprocess.run", return_value=mock_result) as m:
        _obsidian_cli(["create", "path=Papers/x.md", "content=hello"], vault_name="v")

    call_args = m.call_args
    argv = call_args[0][0]

    # argv must be a list, not a string
    assert isinstance(argv, list), f"Expected list argv, got {type(argv)}"
    # shell must not be True
    shell_kwarg = call_args[1].get("shell", False)
    assert shell_kwarg is not True, "subprocess.run called with shell=True"


# ---------------------------------------------------------------------------
# NOTE-03: error detected via stdout prefix (Pitfall 3 — exit code always 0)
# ---------------------------------------------------------------------------

def test_error_detected_via_stdout_prefix():
    """note_exists returns False when stdout starts with 'Error:' (returncode=0)."""
    mock_result = mock.MagicMock()
    mock_result.stdout = 'Error: File "Papers/missing.md" not found.'
    mock_result.returncode = 0
    with mock.patch("scripts.obsidian_cli.subprocess.run", return_value=mock_result):
        result = note_exists("Papers/missing.md", vault_name="v")
    assert result is False, "note_exists should return False on 'Error:' stdout"

    # Non-error stdout → True
    mock_result2 = mock.MagicMock()
    mock_result2.stdout = "Papers/missing.md content here"
    mock_result2.returncode = 0
    with mock.patch("scripts.obsidian_cli.subprocess.run", return_value=mock_result2):
        result2 = note_exists("Papers/missing.md", vault_name="v")
    assert result2 is True, "note_exists should return True on non-error stdout"


# ---------------------------------------------------------------------------
# NOTE-03: preflight returns False when Obsidian unreachable
# ---------------------------------------------------------------------------

def test_preflight_false_when_unreachable():
    """preflight returns False on TimeoutExpired or FileNotFoundError."""
    with mock.patch(
        "scripts.obsidian_cli.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="obsidian", timeout=30),
    ):
        assert preflight(vault_name="v") is False, (
            "preflight should return False on TimeoutExpired"
        )

    with mock.patch(
        "scripts.obsidian_cli.subprocess.run",
        side_effect=FileNotFoundError("obsidian not found"),
    ):
        assert preflight(vault_name="v") is False, (
            "preflight should return False on FileNotFoundError"
        )


# ---------------------------------------------------------------------------
# NOTE-03: create_note overwrite flag appends 'overwrite' token to argv
# ---------------------------------------------------------------------------

def test_create_note_overwrite_flag():
    """When overwrite=True, the argv contains the 'overwrite' token."""
    mock_result = mock.MagicMock()
    mock_result.stdout = "Created Papers/x.md"
    with mock.patch("scripts.obsidian_cli.subprocess.run", return_value=mock_result) as m:
        create_note("Papers/x.md", "body text", vault_name="v", overwrite=True)

    argv = m.call_args[0][0]
    assert "overwrite" in argv, "argv should contain 'overwrite' when overwrite=True"

    # Without overwrite, 'overwrite' should NOT be in argv
    mock_result2 = mock.MagicMock()
    mock_result2.stdout = "Created Papers/x.md"
    with mock.patch("scripts.obsidian_cli.subprocess.run", return_value=mock_result2) as m2:
        create_note("Papers/x.md", "body text", vault_name="v", overwrite=False)

    argv2 = m2.call_args[0][0]
    assert "overwrite" not in argv2, "argv should NOT contain 'overwrite' when overwrite=False"
