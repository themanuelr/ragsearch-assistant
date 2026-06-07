"""
pytest fixtures shared across all test modules.
"""

import json
import os
from unittest.mock import patch

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


@pytest.fixture(autouse=True)
def _tmp_config(tmp_path):
    """Write a temp config.json so extract_paper does not require a real one.

    Patches scripts.ingest._find_config to return the temp config path for the
    duration of each test, preventing tests from walking the filesystem for a
    real config.json (which is gitignored and may not exist on a clean clone).
    """
    registry = tmp_path / "papers_registry.json"
    cfg = {
        "registry_path": str(registry),
        "vault_path": str(tmp_path / "vault"),
        "project_name": "test-project",
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    with patch("scripts.ingest._find_config", return_value=str(cfg_path)):
        yield str(cfg_path)
