"""
Tests for the process_pdf MCP tool in mcp-ollama/server.py.

All tests mock subprocess.run so no GPU or MinerU installation is required.
"""

import sys
import os
import pathlib
import unittest
import unittest.mock as mock

# Add the mcp-ollama directory to sys.path so we can import server
_SERVER_DIR = pathlib.Path(__file__).parent.parent / "mcp-ollama"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from server import process_pdf, INGEST_SCRIPT


# ---------------------------------------------------------------------------
# Task 1 (Plan 04) — process_pdf wrapper tests
# ---------------------------------------------------------------------------

class TestProcessPdfMissingFile(unittest.TestCase):
    """process_pdf on a non-existent path returns a [process_pdf error:] string
    without invoking subprocess."""

    def test_process_pdf_missing_file(self):
        with mock.patch("subprocess.run") as mock_run:
            result = process_pdf("/no/such.pdf")
        self.assertTrue(
            result.startswith("[process_pdf error:"),
            f"Expected '[process_pdf error:' prefix, got: {result!r}"
        )
        mock_run.assert_not_called()


class TestProcessPdfSuccess(unittest.TestCase):
    """process_pdf returns the JSON stdout string when subprocess exits 0."""

    def test_process_pdf_success(self):
        sample_json = '{"extraction": {}, "analysis": {}, "provenance": {"schema_version": 2}}'
        # Use a real existing path so the os.path.exists check passes
        existing_path = str(pathlib.Path(__file__).parent / "test_process_pdf.py")
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = sample_json + "\n"
        mock_result.stderr = ""
        with mock.patch("subprocess.run", return_value=mock_result):
            result = process_pdf(existing_path)
        self.assertEqual(result, sample_json)


class TestProcessPdfFailure(unittest.TestCase):
    """process_pdf returns [process_pdf error: <stderr>] on non-zero returncode."""

    def test_process_pdf_failure(self):
        existing_path = str(pathlib.Path(__file__).parent / "test_process_pdf.py")
        mock_result = mock.MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "[ingest error: MinerU not found]"
        with mock.patch("subprocess.run", return_value=mock_result):
            result = process_pdf(existing_path)
        self.assertTrue(
            result.startswith("[process_pdf error:"),
            f"Expected '[process_pdf error:' prefix, got: {result!r}"
        )
        self.assertIn("MinerU not found", result)


class TestProcessPdfInvokesIngestScript(unittest.TestCase):
    """subprocess.run is called with a list argv containing INGEST_SCRIPT, --pdf, and path."""

    def test_process_pdf_invokes_ingest_script(self):
        existing_path = str(pathlib.Path(__file__).parent / "test_process_pdf.py")
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "{}"
        mock_result.stderr = ""
        with mock.patch("subprocess.run", return_value=mock_result) as mock_run:
            process_pdf(existing_path)
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        # First positional arg is the command list
        cmd = call_args[0][0]
        self.assertIsInstance(cmd, list, "subprocess.run must be called with a list, not shell=True")
        # Must not use shell=True
        call_kwargs = call_args[1] if call_args[1] else {}
        self.assertFalse(call_kwargs.get("shell", False), "shell=True is forbidden (T-01.2-13)")
        # Must contain the INGEST_SCRIPT path
        str_cmd = [str(x) for x in cmd]
        self.assertIn(str(INGEST_SCRIPT), str_cmd, f"INGEST_SCRIPT not in command: {str_cmd}")
        # Must contain --pdf and the path
        self.assertIn("--pdf", str_cmd)
        self.assertIn(existing_path, str_cmd)


if __name__ == "__main__":
    unittest.main()
