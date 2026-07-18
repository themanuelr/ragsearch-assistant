"""Tests for gui/format.py (Phase 8 round-2 gap closure, GUI-06):
the human-readable file-size Jinja filter used on the Processing page's
drop-folder rows.

format_file_size is a pure function (no fixtures needed). Registration on
the shared templates environment (gui/app.py) is exercised via the
gui_client fixture against the real Processing page render.
"""

from gui.format import format_file_size


# ---------------------------------------------------------------------------
# Unit magnitude selection
# ---------------------------------------------------------------------------

def test_bytes_under_1kb_formats_with_byte_unit():
    assert format_file_size(512) == "512 B"


def test_kilobyte_range_formats_with_one_decimal():
    assert format_file_size(1536) == "1.5 KB"  # 1.5 KiB


def test_megabyte_range_formats_with_one_decimal():
    assert format_file_size(2_400_000) == "2.3 MB"  # ~2.29 MiB


def test_zero_bytes_formats_without_error():
    assert format_file_size(0) == "0 B"


def test_boundary_exactly_1kb_formats_as_kilobytes():
    assert format_file_size(1024) == "1.0 KB"


def test_boundary_exactly_1mb_formats_as_megabytes():
    assert format_file_size(1024 * 1024) == "1.0 MB"


# ---------------------------------------------------------------------------
# Degrade-not-raise contract (a filter that raises 500s the whole page)
# ---------------------------------------------------------------------------

def test_none_input_degrades_to_placeholder():
    assert format_file_size(None) == "--"


def test_non_numeric_string_degrades_to_placeholder():
    assert format_file_size("not-a-number") == "--"


def test_negative_input_degrades_to_placeholder():
    assert format_file_size(-5) == "--"


# ---------------------------------------------------------------------------
# Pure function -- no filesystem access, deterministic on repeated calls
# ---------------------------------------------------------------------------

def test_pure_function_same_input_same_output():
    assert format_file_size(4096) == format_file_size(4096)
