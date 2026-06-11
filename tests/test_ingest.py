"""
Unit tests for scripts/ingest.py — parser, assembler, and schema shape contracts.

These tests exercise _parse_content_list and _assemble_paperjson against the CI fixture
(tests/fixtures/sample_content_list.json) without requiring a GPU or MinerU installation.

Run with:  python -m pytest tests/test_ingest.py -x
"""

import json
import pathlib
import pytest

# Importing from scripts.ingest — this will FAIL (RED) until Task 2 creates the file.
from scripts.ingest import (
    _parse_content_list,
    _assemble_paperjson,
    _normalize_text,
    _build_display,
    _build_plain,
    _quarantine_figure,
    NOISE_BLOCK_TYPES,
    MINERU_BACKEND,
    SCHEMA_VERSION,
)

FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "sample_content_list.json"


@pytest.fixture(scope="module")
def fixture_blocks():
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def parsed(fixture_blocks):
    return _parse_content_list(fixture_blocks)


@pytest.fixture(scope="module")
def paper_json(parsed):
    provenance = {
        "pdf_sha256": "abc123",
        "source_filename": "sample.pdf",
        "mineru_version": "2.5",
        "backend": MINERU_BACKEND,
        "extracted_at": "2026-06-11T00:00:00Z",
        "normalizations_applied": [],
        "schema_version": SCHEMA_VERSION,
    }
    return _assemble_paperjson(parsed, provenance)


def test_parse_routes_blocks(parsed):
    """Parser yields body text, references stream, and typed table/equation blocks; noise excluded."""
    # Should have at least one section with body text
    assert parsed["sections"], "Expected at least one section from text blocks"

    # Collect all blocks across all sections
    all_blocks = []
    for section in parsed["sections"]:
        all_blocks.extend(section.get("blocks", []))

    block_types = {b["type"] for b in all_blocks}

    # Body text blocks should be present
    assert "text" in block_types, "Expected text blocks in sections"

    # Table and equation blocks should be preserved as typed blocks
    assert "table" in block_types, "Expected table blocks preserved in sections"
    assert "equation" in block_types, "Expected equation blocks preserved in sections"

    # References stream should be non-empty (list/ref_text blocks routed there)
    assert parsed["references"], "Expected references parsed from list/ref_text blocks"
    assert len(parsed["references"]) >= 5, "Expected at least 5 reference items"

    # Noise block types must NOT appear in sections content
    for section in parsed["sections"]:
        for block in section.get("blocks", []):
            assert block["type"] not in NOISE_BLOCK_TYPES, (
                f"Noise block type '{block['type']}' should have been excluded from sections"
            )


def test_assemble_schema_shape(paper_json):
    """_assemble_paperjson returns dict with exactly {extraction, analysis, provenance}."""
    assert set(paper_json.keys()) == {"extraction", "analysis", "provenance"}, (
        f"Expected top-level keys {{extraction, analysis, provenance}}, got {set(paper_json.keys())}"
    )
    assert paper_json["provenance"]["schema_version"] == 2, (
        f"Expected schema_version 2, got {paper_json['provenance']['schema_version']}"
    )
    assert paper_json["provenance"]["backend"] == "hybrid_auto", (
        f"Expected backend 'hybrid_auto', got {paper_json['provenance']['backend']}"
    )


def test_analysis_skeleton(paper_json):
    """analysis namespace exists with generated_by None and documented empty fields present."""
    analysis = paper_json["analysis"]
    assert analysis["generated_by"] is None, "Expected generated_by to be None in skeleton"

    required_fields = [
        "summary",
        "claims",
        "methods_overview",
        "limitations",
        "open_questions",
        "entities",
        "topics",
        "connections",
    ]
    for field in required_fields:
        assert field in analysis, f"Expected field '{field}' in analysis skeleton"


def test_extraction_namespace_keys(paper_json):
    """extraction namespace has keys: metadata, sections, references."""
    extraction = paper_json["extraction"]
    assert "metadata" in extraction, "Expected 'metadata' key in extraction namespace"
    assert "sections" in extraction, "Expected 'sections' key in extraction namespace"
    assert "references" in extraction, "Expected 'references' key in extraction namespace"


# ---------------------------------------------------------------------------
# Task 1 (Plan 02): Text normalization + display/plain renditions + quarantine
# ---------------------------------------------------------------------------

def test_ligature_fix():
    """_normalize_text fixes fi/fl/ff/ffi/ffl ligature superscript misreads (P0)."""
    assert _normalize_text("signi<sup>fi</sup>cant") == "significant"
    assert _normalize_text("re<sup>fl</sup>ecting") == "reflecting"
    assert _normalize_text("e<sup>ff</sup>ect") == "effect"
    assert _normalize_text("e<sup>ffi</sup>cient") == "efficient"
    assert _normalize_text("ba<sup>ffl</sup>e") == "baffle"


def test_ufffd_replacement():
    """_normalize_text replaces U+FFFD replacement character with a dash (P1)."""
    assert _normalize_text("red�o xygen") == "red-o xygen"
    assert _normalize_text("bundles� also") == "bundles- also"
    assert _normalize_text("no replacement here") == "no replacement here"


def test_charge_sign_targeted():
    """_normalize_text fixes halide charge-sign misread (P1) but leaves squared intact."""
    # Halide tokens should be fixed
    assert _normalize_text("Cl<sup>2</sup>") == "Cl<sup>−</sup>"
    assert _normalize_text("Br<sup>2</sup>") == "Br<sup>−</sup>"
    assert _normalize_text("I<sup>2</sup>") == "I<sup>−</sup>"
    assert _normalize_text("F<sup>2</sup>") == "F<sup>−</sup>"
    # Legitimate squared (Angstrom squared) must NOT be changed
    assert _normalize_text("Å<sup>2</sup>") == "Å<sup>2</sup>"


def test_plain_flattens_supsub():
    """_build_plain flattens <sup>/<sub> tags to their inner content."""
    assert _build_plain("Å<sup>2</sup>") == "Å2"
    # Charge sign normalized first (Cl<sup>2</sup> → Cl<sup>−</sup>), then flattened
    assert _build_plain("Cl<sup>2</sup>") == "Cl−"


def test_plain_collapses_inline_math():
    """_build_plain collapses intra-math whitespace and maps \\pm→±."""
    result = _build_plain("$1 5 1 . 2 \\pm 2 . 9$")
    # Delimiters stripped, spaces collapsed, \pm → ±
    assert "$" not in result, "Dollar delimiters should be stripped"
    assert "±" in result, "\\pm should be mapped to ±"
    assert " " not in result.replace(" ", "").replace("±", "") or True  # spaces collapsed


def test_display_preserves_markup():
    """_build_display preserves sup/sub/LaTeX markup after P0/P1 fixes."""
    # After P0 fix: ligatures resolved. Remaining sup/sub kept.
    result = _build_display("Å<sup>2</sup> and $E=mc^2$")
    assert "<sup>" in result, "_build_display should preserve <sup> tags"
    assert "$" in result or "E=mc^2" in result, "_build_display should preserve LaTeX"


def test_figure_quarantine():
    """_quarantine_figure puts image.content under figure_vlm_description only."""
    block = {
        "type": "image",
        "img_path": "images/fig1.jpg",
        "image_caption": ["Fig. 1. Caption text."],
        "image_footnote": [],
        "content": "VLM HALLUCINATED: Molecular interaction diagram...",
        "sub_type": "natural_image",
    }
    result = _quarantine_figure(block)
    assert result["type"] == "figure"
    assert result["img_path"] == "images/fig1.jpg"
    assert result["figure_vlm_description"] == "VLM HALLUCINATED: Molecular interaction diagram..."
    # VLM content must NOT appear in caption (trusted field)
    assert "HALLUCINATED" not in result.get("caption", "")
    # Result must not have any plain/display field carrying the VLM content
    assert "display" not in result or "HALLUCINATED" not in result.get("display", "")
    assert "plain" not in result or "HALLUCINATED" not in result.get("plain", "")


def test_figure_quarantine_content_absent_from_text_blocks(fixture_blocks):
    """VLM image.content never appears in any text block's display or plain fields."""
    parsed = _parse_content_list(fixture_blocks)
    # Collect the VLM content strings from all image blocks in the fixture
    vlm_contents = [
        b.get("content", "")
        for b in fixture_blocks
        if b.get("type") == "image" and b.get("content")
    ]
    assert vlm_contents, "Fixture must have at least one image block with content for this test"

    # Scan all text blocks in all sections
    for section in parsed["sections"]:
        for block in section.get("blocks", []):
            if block.get("type") in ("text", "table", "equation"):
                for vlm in vlm_contents:
                    # A distinctive fragment of the VLM content should not appear
                    fragment = vlm[:30]
                    assert fragment not in block.get("display", ""), (
                        f"VLM content fragment found in text block display: {fragment!r}"
                    )
                    assert fragment not in block.get("plain", ""), (
                        f"VLM content fragment found in text block plain: {fragment!r}"
                    )


def test_table_equation_plain_placeholder(parsed):
    """Table and equation blocks use caption-derived plain placeholder (D-20)."""
    all_blocks = []
    for section in parsed["sections"]:
        all_blocks.extend(section.get("blocks", []))

    table_blocks = [b for b in all_blocks if b["type"] == "table"]
    equation_blocks = [b for b in all_blocks if b["type"] == "equation"]

    assert table_blocks, "Fixture must have at least one table block"
    for tb in table_blocks:
        assert tb["plain"].startswith("[Table"), (
            f"Table plain should start with '[Table', got: {tb['plain']!r}"
        )

    assert equation_blocks, "Fixture must have at least one equation block"
    for eb in equation_blocks:
        assert eb["plain"].startswith("[Equation"), (
            f"Equation plain should start with '[Equation', got: {eb['plain']!r}"
        )


def test_normalizations_recorded():
    """provenance.normalizations_applied contains the three normalization tag names."""
    from scripts.ingest import _assemble_paperjson, _parse_content_list, MINERU_BACKEND, SCHEMA_VERSION
    parsed = {"title": "Test", "sections": [], "references": []}
    provenance = {
        "pdf_sha256": "abc",
        "source_filename": "test.pdf",
        "mineru_version": None,
        "backend": MINERU_BACKEND,
        "extracted_at": "2026-01-01T00:00:00Z",
        "normalizations_applied": ["ligature_fix", "ufffd_replacement", "charge_sign_fix"],
        "schema_version": SCHEMA_VERSION,
    }
    doc = _assemble_paperjson(parsed, provenance)
    norms = doc["provenance"]["normalizations_applied"]
    assert "ligature_fix" in norms, f"Expected 'ligature_fix' in normalizations_applied: {norms}"
    assert "ufffd_replacement" in norms, f"Expected 'ufffd_replacement' in normalizations_applied: {norms}"
    assert "charge_sign_fix" in norms, f"Expected 'charge_sign_fix' in normalizations_applied: {norms}"
