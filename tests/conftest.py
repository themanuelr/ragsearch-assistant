"""
pytest fixtures shared across all test modules.
"""

import os
import pytest


@pytest.fixture
def tmp_registry_path(tmp_path):
    """Temporary registry JSON path — cleaned up after each test."""
    return str(tmp_path / "papers_registry.json")


@pytest.fixture
def sample_pdf_path():
    """Path to a real single-column PDF in tests/fixtures/."""
    return os.path.join(os.path.dirname(__file__), "fixtures", "sample_onecol.pdf")


@pytest.fixture
def sample_twocol_pdf_path():
    """Path to a real two-column PDF in tests/fixtures/."""
    return os.path.join(os.path.dirname(__file__), "fixtures", "sample_twocol.pdf")


@pytest.fixture
def scanned_pdf_path():
    """Path to an image-only PDF in tests/fixtures/."""
    return os.path.join(os.path.dirname(__file__), "fixtures", "sample_scanned.pdf")
