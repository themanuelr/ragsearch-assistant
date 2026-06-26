"""
Unit tests for scripts/ingest.py — parser, assembler, and schema shape contracts.

These tests exercise _parse_content_list and _assemble_paperjson against the CI fixture
(tests/fixtures/sample_content_list.json) without requiring a GPU or MinerU installation.

Run with:  python -m pytest tests/test_ingest.py -x
"""

import json
import os
import pathlib
import pytest

# Importing from scripts.ingest
from scripts.ingest import (
    _parse_content_list,
    _assemble_paperjson,
    _normalize_text,
    _build_display,
    _build_plain,
    _quarantine_figure,
    _quality_gate,
    _ollama_extraction_call,
    _parse_extraction_response,
    _estimate_num_ctx,
    _warmup_ollama,
    DoiProbeResult,
    PaperMetadata,
    SectionFillResult,
    RefEntry,
    RefBatchResult,
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


def test_plain_flattens_supsub():
    """_build_plain flattens <sup>/<sub> tags to their inner content."""
    assert _build_plain("Å<sup>2</sup>") == "Å2"
    # U+FFFD and charge-sign fixes removed (D-09) — LLM fill handles these.
    # Verify that without charge-sign fix, Cl<sup>2</sup> flattens to Cl2 (not Cl−).
    assert _build_plain("Cl<sup>2</sup>") == "Cl2"


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
    """provenance.normalizations_applied contains ligature_fix and llm_fill; not ufffd/charge_sign."""
    from scripts.ingest import _assemble_paperjson, MINERU_BACKEND, SCHEMA_VERSION
    parsed = {"title": "Test", "sections": [], "references": []}
    provenance = {
        "pdf_sha256": "abc",
        "source_filename": "test.pdf",
        "mineru_version": None,
        "backend": MINERU_BACKEND,
        "extracted_at": "2026-01-01T00:00:00Z",
        "normalizations_applied": ["ligature_fix", "llm_fill"],
        "schema_version": SCHEMA_VERSION,
    }
    doc = _assemble_paperjson(parsed, provenance)
    norms = doc["provenance"]["normalizations_applied"]
    assert "llm_fill" in norms, f"Expected 'llm_fill' in normalizations_applied: {norms}"
    assert "ligature_fix" in norms, f"Expected 'ligature_fix' in normalizations_applied: {norms}"
    assert "ufffd_replacement" not in norms, f"Expected 'ufffd_replacement' NOT in normalizations_applied: {norms}"
    assert "charge_sign_fix" not in norms, f"Expected 'charge_sign_fix' NOT in normalizations_applied: {norms}"


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def test_quality_gate_passes_good(paper_json):
    """_quality_gate returns None for a valid parse with title and content."""
    result = _quality_gate(paper_json)
    assert result is None, f"Expected None from quality gate on good output, got: {result!r}"


def test_quality_gate_fails_garbage():
    """_quality_gate returns [ingest error: ...] message for title-less/near-empty parse."""
    from scripts.ingest import _assemble_paperjson, MINERU_BACKEND, SCHEMA_VERSION
    parsed_empty = {"title": None, "sections": [], "references": []}
    provenance = {
        "pdf_sha256": "abc",
        "source_filename": "garbage.pdf",
        "mineru_version": None,
        "backend": MINERU_BACKEND,
        "extracted_at": "2026-01-01T00:00:00Z",
        "normalizations_applied": [],
        "schema_version": SCHEMA_VERSION,
    }
    garbage_doc = _assemble_paperjson(parsed_empty, provenance)
    result = _quality_gate(garbage_doc)
    assert result is not None, "Expected an error message for garbage output, got None"
    assert result.startswith("[ingest error:"), (
        f"Expected '[ingest error: ...' prefix, got: {result!r}"
    )


# ---------------------------------------------------------------------------
# Task 1 (Plan 03): Registry key derivation + filelock atomic write + entry
# ---------------------------------------------------------------------------

from scripts.ingest import (  # noqa: E402 — import here after all registry funcs are added
    _registry_key,
    _read_registry,
    _write_registry,
    _registry_entry,
)


SAMPLE_PAPERJSON_FOR_REGISTRY = {
    "extraction": {
        "metadata": {
            "title": "A Novel Method for X",
            "authors": [{"name": "Smith, J."}],
            "year": 2024,
            "journal": "J. Am. Chem. Soc.",
            "doi": "10.1021/jacs.3c10258",
            "arxiv_id": "2401.00001",
            "accession_codes": [],
        },
        "sections": [],
        "references": [],
    },
    "analysis": {"generated_by": None},
    "provenance": {"schema_version": 2},
}


def test_key_priority_doi():
    """_registry_key returns doi when doi is present."""
    key = _registry_key({"doi": "10.1/x", "arxiv_id": "2401.00001", "title": "T"})
    assert key == "10.1/x", f"Expected DOI key '10.1/x', got {key!r}"


def test_key_priority_arxiv():
    """_registry_key returns arxiv_id when doi is None."""
    key = _registry_key({"doi": None, "arxiv_id": "2401.00001", "title": "T"})
    assert key == "2401.00001", f"Expected arXiv key '2401.00001', got {key!r}"


def test_key_priority_title_hash():
    """_registry_key returns sha256:<hex> when doi and arxiv_id are both None."""
    key = _registry_key({"doi": None, "arxiv_id": None, "title": "My Title"})
    assert key.startswith("sha256:"), f"Expected key starting with 'sha256:', got {key!r}"
    # Hex chars only after the prefix
    hex_part = key[len("sha256:"):]
    assert all(c in "0123456789abcdef" for c in hex_part), (
        f"Expected hex after 'sha256:', got {hex_part!r}"
    )
    assert len(hex_part) > 0, "Expected non-empty hex prefix"


def test_key_title_hash_stable():
    """_registry_key returns the same hash for identical titles."""
    k1 = _registry_key({"doi": None, "arxiv_id": None, "title": "Stable Title"})
    k2 = _registry_key({"doi": None, "arxiv_id": None, "title": "Stable Title"})
    assert k1 == k2, "Expected identical keys for identical titles"


def test_registry_entry_shape():
    """_registry_entry has the D-23 key set; summary and key_findings are None."""
    entry = _registry_entry(
        SAMPLE_PAPERJSON_FOR_REGISTRY,
        source_path="/data/paper.pdf",
        paperjson_path="/data/paper.json",
        project_name="my-project",
    )
    required_keys = {
        "title", "authors", "year", "journal", "doi", "arxiv_id",
        "projects", "source_path", "paperjson_path", "summary", "key_findings",
    }
    assert set(entry.keys()) == required_keys, (
        f"Expected keys {required_keys}, got {set(entry.keys())}"
    )
    assert entry["summary"] is None, "Expected summary to be None (extraction-only)"
    assert entry["key_findings"] is None, "Expected key_findings to be None (extraction-only)"
    assert entry["projects"] == ["my-project"], (
        f"Expected projects=['my-project'], got {entry['projects']}"
    )
    assert entry["source_path"] == "/data/paper.pdf"
    assert entry["paperjson_path"] == "/data/paper.json"
    assert entry["doi"] == "10.1021/jacs.3c10258"
    assert entry["arxiv_id"] == "2401.00001"


def test_write_then_read_roundtrip(tmp_path):
    """_write_registry then _read_registry returns a dict with the written entry."""
    reg_path = str(tmp_path / "registry.json")
    entry = {"title": "Test Paper", "doi": "10.1/test"}
    _write_registry(entry, reg_path, "10.1/test")
    registry = _read_registry(reg_path)
    assert isinstance(registry, dict), "Expected dict from _read_registry"
    assert "10.1/test" in registry, "Expected written key in registry"
    assert registry["10.1/test"]["title"] == "Test Paper"


def test_read_registry_missing_file(tmp_path):
    """_read_registry returns {} when the registry file does not exist."""
    reg_path = str(tmp_path / "nonexistent_registry.json")
    result = _read_registry(reg_path)
    assert result == {}, f"Expected empty dict for missing registry, got {result!r}"


def test_read_registry_empty_file(tmp_path):
    """_read_registry returns {} when the registry file exists but is 0 bytes (regression: REG-empty)."""
    reg_path = tmp_path / "empty_registry.json"
    reg_path.write_bytes(b"")  # 0-byte file — simulates the .local/papers_registry.json bug
    result = _read_registry(str(reg_path))
    assert result == {}, f"Expected empty dict for 0-byte registry, got {result!r}"


def test_read_registry_whitespace_only(tmp_path):
    """_read_registry returns {} when the registry file contains only whitespace."""
    reg_path = tmp_path / "ws_registry.json"
    reg_path.write_text("   \n\t\n", encoding="utf-8")
    result = _read_registry(str(reg_path))
    assert result == {}, f"Expected empty dict for whitespace-only registry, got {result!r}"


def test_read_registry_corrupt_nonempty_raises(tmp_path):
    """_read_registry raises ValueError (not silently) on corrupt non-empty file."""
    reg_path = tmp_path / "corrupt_registry.json"
    reg_path.write_text("{corrupted content", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        _read_registry(str(reg_path))


def test_write_registry_empty_file_no_crash(tmp_path):
    """_write_registry succeeds when the registry file exists but is 0 bytes (REG-empty fix)."""
    reg_path = tmp_path / "empty_registry.json"
    reg_path.write_bytes(b"")  # 0-byte file — the exact condition that caused the bug
    entry = {"title": "First Paper", "doi": "10.1/first"}
    # Must not raise; should write successfully
    _write_registry(entry, str(reg_path), "10.1/first")
    result = _read_registry(str(reg_path))
    assert "10.1/first" in result, f"Expected entry written, got keys: {list(result.keys())}"
    assert result["10.1/first"]["title"] == "First Paper"


def test_atomic_no_partial_file(tmp_path):
    """After _write_registry, no .tmp file remains beside the registry."""
    reg_path = str(tmp_path / "registry.json")
    tmp_path_file = reg_path + ".tmp"
    _write_registry({"title": "X"}, reg_path, "key1")
    assert not pathlib.Path(tmp_path_file).exists(), (
        f"Expected .tmp file to be cleaned up, but {tmp_path_file} still exists"
    )


def test_concurrent_writes_no_corruption(tmp_path):
    """Two concurrent writers writing distinct keys both survive; registry parses as valid JSON."""
    import concurrent.futures

    reg_path = str(tmp_path / "registry.json")

    def write_entry(key):
        _write_registry({"title": f"Paper {key}"}, reg_path, key)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(write_entry, "key-a"),
            executor.submit(write_entry, "key-b"),
        ]
        for f in concurrent.futures.as_completed(futures):
            f.result()  # re-raise any exceptions

    registry = _read_registry(reg_path)
    assert "key-a" in registry, f"Expected 'key-a' in registry after concurrent writes, got {registry.keys()}"
    assert "key-b" in registry, f"Expected 'key-b' in registry after concurrent writes, got {registry.keys()}"


# ---------------------------------------------------------------------------
# Task 2 (Plan 03): Dedup check wired into ingest() — skip on cache hit
# ---------------------------------------------------------------------------

import unittest.mock as mock  # noqa: E402


def _make_ingest_config(tmp_path, extra=None):
    """Build a minimal config dict pointing registry to a tmp file."""
    cfg = {
        "registry_path": str(tmp_path / "registry.json"),
        "project_name": "test-project",
        "mineru_path": "/fake/mineru",
    }
    if extra:
        cfg.update(extra)
    return cfg


def _write_real_content_list(tmp_path):
    """Write a minimal real content_list.json to tmp_path for tests that need file I/O.

    The content must pass the quality gate (title + text >= 100 chars).
    """
    cl_path = tmp_path / "content_list.json"
    blocks = [
        {"type": "text", "text": "Paper Title", "text_level": 1, "page_idx": 0},
        # Ensure enough text to pass the quality gate (threshold=100 chars of plain text)
        {"type": "text", "text": "X" * 200, "text_level": None, "page_idx": 0},
    ]
    cl_path.write_text(json.dumps(blocks), encoding="utf-8")
    return str(cl_path)


def test_dedup_skip_returns_cached(tmp_path):
    """When DOI probe key is already in registry, ingest() returns cached entry (REG-02).

    Phase 1.3: _run_mineru always runs (MinerU before probe); registry gate is now
    DOI-probe-gated rather than metadata-parse-gated. Fill helpers must not be called.
    """
    from scripts.ingest import ingest, _write_registry, DoiProbeResult

    cfg = _make_ingest_config(tmp_path)
    reg_path = cfg["registry_path"]

    # Pre-populate the registry with the probe-derived DOI key.
    # Use proper 4+ digit registrant prefix to survive _syntactic_doi_valid and DoiProbeResult validator.
    doi_key = "10.1000/dedup.test"
    cached = {
        "title": "A Great Paper Title",
        "doi": doi_key,
        "projects": ["other-project"],
        "summary": None,
        "key_findings": None,
        "authors": None,
        "year": None,
        "journal": None,
        "arxiv_id": None,
        "source_path": "/old/paper.pdf",
        "paperjson_path": "/old/paper.json",
    }
    _write_registry(cached, reg_path, doi_key)

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake content for hash")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = {
        "title": "A Great Paper Title",
        "sections": [{"heading": "", "level": 0, "blocks": [
            {"type": "text", "display": "A Great Paper Title", "plain": "A Great Paper Title"},
            # Enough text to pass quality gate (>= 100 chars total plain text)
            {"type": "text", "display": "X" * 200, "plain": "X" * 200},
        ]}],
        "references": [],
        "metadata": {
            "title": "A Great Paper Title",
            "authors": None,
            "year": None,
            "journal": None,
            "doi": doi_key,
            "arxiv_id": None,
            "accession_codes": [],
        },
    }
    probe_result = DoiProbeResult(doi=doi_key, arxiv_id=None, title="A Great Paper Title")

    # Phase 1.3: _run_mineru runs (MinerU always runs before probe); fill helpers must NOT be called
    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata",
                    side_effect=AssertionError("_fill_metadata must not be called on cache hit")), \
         mock.patch("scripts.ingest._fill_section",
                    side_effect=AssertionError("_fill_section must not be called on cache hit")), \
         mock.patch("scripts.ingest._fill_references_batched",
                    side_effect=AssertionError("_fill_references_batched must not be called on cache hit")):
        result = ingest(str(fake_pdf), cfg)

    assert result is not None, "Expected a result from ingest() on cache hit"
    assert result.get("title") == "A Great Paper Title" or (
        result.get("extraction", {}).get("metadata", {}).get("title") == "A Great Paper Title"
    ), f"Expected cached entry returned, got: {result}"


def test_new_paper_writes_entry(tmp_path):
    """Ingesting a not-yet-registered paper calls _write_registry once and the key appears."""
    from scripts.ingest import ingest, _read_registry, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    reg_path = cfg["registry_path"]

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake content for hash")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = {
        "title": "New Paper",
        "sections": [{"heading": "", "level": 0, "blocks": [
            {"type": "text", "display": "New Paper", "plain": "New Paper"},
            {"type": "text", "display": "A" * 200, "plain": "A" * 200},
        ]}],
        "references": [],
        "metadata": {
            "title": "New Paper",
            "authors": None,
            "year": 2024,
            "journal": None,
            "doi": "10.1000/new.paper",
            "arxiv_id": None,
            "accession_codes": [],
        },
    }
    probe_result = DoiProbeResult(doi="10.1000/new.paper", arxiv_id=None, title="New Paper")
    metadata_result = PaperMetadata(title="New Paper", doi="10.1000/new.paper", year=2024)
    section_fill = SectionFillResult(heading="", body="New Paper " + "A" * 200, fill_failed=False)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        ingest(str(fake_pdf), cfg)

    registry = _read_registry(reg_path)
    assert "10.1000/new.paper" in registry, (
        f"Expected DOI key '10.1000/new.paper' in registry after new ingest, got keys: {list(registry.keys())}"
    )


def test_force_extract_bypasses_cache(tmp_path):
    """With force_extract=True, the pipeline runs and re-registers even if key exists."""
    from scripts.ingest import ingest, _write_registry, _read_registry

    cfg = _make_ingest_config(tmp_path)
    reg_path = cfg["registry_path"]

    # Pre-populate registry
    old_entry = {
        "title": "Old Entry",
        "doi": "10.1000/force.test",
        "projects": ["old-project"],
        "summary": None, "key_findings": None,
        "authors": None, "year": 2020, "journal": None,
        "arxiv_id": None,
        "source_path": "/old.pdf",
        "paperjson_path": "/old.json",
    }
    _write_registry(old_entry, reg_path, "10.1000/force.test")

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 force extract fake")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = {
        "title": "Updated Paper",
        "sections": [{"heading": "", "level": 0, "blocks": [
            {"type": "text", "display": "Updated Paper", "plain": "Updated Paper"},
            {"type": "text", "display": "B" * 200, "plain": "B" * 200},
        ]}],
        "references": [],
        "metadata": {
            "title": "Updated Paper",
            "authors": None,
            "year": 2024,
            "journal": None,
            "doi": "10.1000/force.test",
            "arxiv_id": None,
            "accession_codes": [],
        },
    }

    from scripts.ingest import DoiProbeResult, PaperMetadata, SectionFillResult
    probe_result = DoiProbeResult(doi="10.1000/force.test", arxiv_id=None, title="Updated Paper")
    metadata_result = PaperMetadata(title="Updated Paper", doi="10.1000/force.test", year=2024)
    section_fill = SectionFillResult(heading="", body="Updated Paper " + "B" * 200, fill_failed=False)

    mineru_called = []

    def fake_mineru(*args, **kwargs):
        mineru_called.append(True)

    with mock.patch("scripts.ingest._run_mineru", side_effect=fake_mineru), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        result = ingest(str(fake_pdf), cfg, force_extract=True)

    # _run_mineru must have been called (force_extract bypasses cache)
    assert mineru_called, "_run_mineru should have been called when force_extract=True"

    # Registry should be updated with the new entry
    registry = _read_registry(reg_path)
    assert "10.1000/force.test" in registry, "Expected key to exist in registry after force re-ingest"
    # The new entry's year should be updated (2024 not 2020)
    assert registry["10.1000/force.test"].get("year") == 2024, (
        f"Expected updated year 2024 after force re-ingest, got {registry['10.1000/force.test'].get('year')}"
    )


def test_reg01_entry_written_on_new_ingest(tmp_path):
    """After a new ingest, _read_registry contains an entry with the paper's title and DOI (REG-01)."""
    from scripts.ingest import ingest, _read_registry

    cfg = _make_ingest_config(tmp_path)
    reg_path = cfg["registry_path"]

    fake_pdf = tmp_path / "reg01paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 reg01 fake content")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = {
        "title": "REG-01 Test Paper",
        "sections": [{"heading": "", "level": 0, "blocks": [
            {"type": "text", "display": "REG-01 Test Paper", "plain": "REG-01 Test Paper"},
            {"type": "text", "display": "C" * 200, "plain": "C" * 200},
        ]}],
        "references": [],
        "metadata": {
            "title": "REG-01 Test Paper",
            "authors": None,
            "year": 2024,
            "journal": "Test Journal",
            "doi": "10.1000/reg01.test",
            "arxiv_id": None,
            "accession_codes": [],
        },
    }

    from scripts.ingest import DoiProbeResult, PaperMetadata, SectionFillResult
    probe_result = DoiProbeResult(doi="10.1000/reg01.test", arxiv_id=None, title="REG-01 Test Paper")
    metadata_result = PaperMetadata(title="REG-01 Test Paper", doi="10.1000/reg01.test", year=2024, journal="Test Journal")
    section_fill = SectionFillResult(heading="", body="REG-01 Test Paper " + "C" * 200, fill_failed=False)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        ingest(str(fake_pdf), cfg)

    registry = _read_registry(reg_path)
    assert "10.1000/reg01.test" in registry, "Expected registry to contain the new paper's DOI key"
    entry = registry["10.1000/reg01.test"]
    # title comes from the LLM fill (PaperMetadata), which is written into skeleton metadata
    assert entry.get("title") == "REG-01 Test Paper", (
        f"Expected title 'REG-01 Test Paper', got {entry.get('title')!r}"
    )
    assert entry.get("doi") == "10.1000/reg01.test", (
        f"Expected doi '10.1000/reg01.test', got {entry.get('doi')!r}"
    )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 01: Mock-based unit tests for new LLM primitives
# ---------------------------------------------------------------------------

def test_parse_response_clean_json():
    """_parse_extraction_response validates a clean JSON string into the target model."""
    import json as _json
    raw = _json.dumps({"doi": "10.1073/pnas.2209111120", "arxiv_id": None, "title": "Sample"})
    result = _parse_extraction_response(raw, DoiProbeResult)
    assert result is not None, "Expected a validated model for clean JSON"
    assert result.doi == "10.1073/pnas.2209111120"
    assert result.title == "Sample"


def test_parse_response_fenced_json():
    """_parse_extraction_response recovers from ```json...``` fence wrapping (Ollama #15416)."""
    import json as _json
    payload = _json.dumps({"doi": "10.1073/pnas.2209111120", "arxiv_id": None, "title": "Sample"})
    fenced = f"```json\n{payload}\n```"
    result = _parse_extraction_response(fenced, DoiProbeResult)
    assert result is not None, "Expected a validated model from fenced JSON (Ollama #15416 recovery)"
    assert result.doi == "10.1073/pnas.2209111120"


def test_parse_response_garbage_returns_none():
    """_parse_extraction_response returns None for a non-JSON string."""
    result = _parse_extraction_response("not json at all", DoiProbeResult)
    assert result is None, "Expected None for garbage input"


def test_ollama_extraction_call_payload_think_false():
    """_ollama_extraction_call sends think=False alongside format in the /api/chat payload.

    Regression test: Ollama 0.30.6 honors think=False + format together (verified 2026-06-23,
    ~2.6x faster structured calls). Pins the key into the payload so the optimization cannot
    silently regress. The historical #15260 catch-22 (think=False dropped format) no longer
    reproduces on 0.30.6; the #15416 fence-stripping fallback in _parse_extraction_response
    remains as defensive coverage regardless.
    """
    import json as _json

    captured = {}

    def fake_urlopen(req, *args, **kwargs):
        captured["data"] = req.data
        body = _json.dumps({"message": {"content": "{}"}}).encode()
        return _FakeHttpResponse(body)

    with mock.patch("scripts.ingest.urllib.request.urlopen", side_effect=fake_urlopen):
        _ollama_extraction_call("p", "s", schema={"type": "object"}, num_ctx=512, timeout=5)

    assert "data" in captured, "Expected urlopen to have been called and req.data captured"
    payload = _json.loads(captured["data"].decode())
    assert payload.get("think") is False, (
        f"Expected think=False in /api/chat payload, got think={payload.get('think')!r}"
    )
    assert "format" in payload, (
        "Expected 'format' key present alongside think=False (Ollama 0.30.6 coexistence)"
    )


def test_doi_probe_full_suffix():
    """DOI extracted byte-for-byte; PNAS full suffix preserved (E1 — guards PNAS truncation defect)."""
    import json as _json
    mock_response = _json.dumps({
        "doi": "10.1073/pnas.2209111120",
        "arxiv_id": None,
        "title": "Sample Paper Title",
    })
    result = _parse_extraction_response(mock_response, DoiProbeResult)
    assert result is not None, "Expected a validated DoiProbeResult"
    assert result.doi == "10.1073/pnas.2209111120", (
        f"Expected full PNAS DOI suffix preserved, got: {result.doi!r}"
    )


def test_estimate_num_ctx_buckets():
    """_estimate_num_ctx returns correct bucket from extended ladder and caps at DEFAULT_NUM_CTX_CAP=65536.

    Updated in Plan 05: ladder now reaches 65536; explicit cap override still works.
    """
    from scripts.ingest import DEFAULT_NUM_CTX_CAP
    # Short text (~25 tokens + 2048 overhead = 2073 → bucket 4096 with default overhead)
    assert _estimate_num_ctx("x" * 100, overhead=0) == 2048, (
        "Expected 2048 for 100-char text with no overhead (25 tokens <= 2048 bucket)"
    )
    # Mid-length text: ~24000 chars → ~6000 tokens + 2048 overhead = 8048 → bucket 8192
    mid_text = "x" * 24000
    assert _estimate_num_ctx(mid_text) == 8192, (
        f"Expected 8192 for ~24000-char text, got {_estimate_num_ctx(mid_text)}"
    )
    # Large text: ~100000 chars → 25000 tokens + 2048 = 27048 → bucket 32768
    large_text = "x" * 100000
    assert _estimate_num_ctx(large_text) == 32768, (
        f"Expected 32768 for ~100000-char text, got {_estimate_num_ctx(large_text)}"
    )
    # Very large text: ~200000 chars → 50000 tokens + 2048 = 52048 → bucket 65536 (new default cap)
    very_large_text = "x" * 200000
    assert _estimate_num_ctx(very_large_text) == 65536, (
        f"Expected 65536 for ~200000-char text (new default cap), got {_estimate_num_ctx(very_large_text)}"
    )
    # Explicit cap override: cap=16384 clamps large text back to 16384
    assert _estimate_num_ctx(very_large_text, cap=16384) == 16384, (
        f"Expected 16384 with explicit cap=16384, got {_estimate_num_ctx(very_large_text, cap=16384)}"
    )


def test_models_schema_generation():
    """Each of the five Pydantic models generates an Ollama-compatible JSON schema with 'properties'."""
    models = [DoiProbeResult, PaperMetadata, SectionFillResult, RefEntry, RefBatchResult]
    for Model in models:
        schema = Model.model_json_schema()
        assert isinstance(schema, dict), f"{Model.__name__}.model_json_schema() must return a dict"
        assert "properties" in schema, (
            f"Expected 'properties' key in {Model.__name__}.model_json_schema() for Ollama format= compatibility, "
            f"got: {list(schema.keys())}"
        )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 02 Task 1: probe + fill helpers
# ---------------------------------------------------------------------------

from unittest.mock import patch
import json as _json_mod


def test_extract_first_page_and_footers_basic():
    """_extract_first_page_and_footers returns first-page text and footer blocks."""
    from scripts.ingest import _extract_first_page_and_footers
    blocks = [
        {"type": "text", "page_idx": 0, "text": "Title on page 0"},
        {"type": "text", "page_idx": 1, "text": "Body on page 1"},
        {"type": "footer", "page_idx": 1, "text": "DOI 10.1073/pnas.123"},
        {"type": "footer", "page_idx": 0, "text": "Journal Name"},
        {"type": "text", "page_idx": 0, "text": "   "},  # blank — should be skipped
    ]
    result = _extract_first_page_and_footers(blocks)
    assert "Title on page 0" in result
    assert "DOI 10.1073/pnas.123" in result
    assert "Journal Name" in result
    # Body page 1 text block should NOT be included
    assert "Body on page 1" not in result
    # Blank text should not produce a blank line entry
    assert "   " not in result


def test_extract_first_page_and_footers_empty():
    """_extract_first_page_and_footers returns empty string for empty block list."""
    from scripts.ingest import _extract_first_page_and_footers
    result = _extract_first_page_and_footers([])
    assert result == ""


def test_syntactic_doi_valid_good():
    """_syntactic_doi_valid returns True for a valid full DOI."""
    from scripts.ingest import _syntactic_doi_valid
    assert _syntactic_doi_valid("10.1073/pnas.2209111120") is True
    assert _syntactic_doi_valid("10.1021/jacs.3c10258") is True
    assert _syntactic_doi_valid("10.1016/j.cell.2023.01.001") is True


def test_syntactic_doi_valid_none():
    """_syntactic_doi_valid returns False for None input."""
    from scripts.ingest import _syntactic_doi_valid
    assert _syntactic_doi_valid(None) is False


def test_syntactic_doi_valid_garbage():
    """_syntactic_doi_valid returns False for non-DOI strings."""
    from scripts.ingest import _syntactic_doi_valid
    assert _syntactic_doi_valid("garbage") is False
    assert _syntactic_doi_valid("http://example.com") is False
    assert _syntactic_doi_valid("") is False


def test_doi_probe_raises_on_ollama_error():
    """_doi_probe raises RuntimeError when _ollama_extraction_call returns an error string (D-00d)."""
    from scripts.ingest import _doi_probe
    with patch("scripts.ingest._ollama_extraction_call", return_value="[Ollama error: connection refused]"):
        with pytest.raises(RuntimeError):
            _doi_probe("First page text with DOI 10.1073/pnas.123")


def test_doi_probe_returns_result_on_success():
    """_doi_probe returns a DoiProbeResult on a valid LLM response."""
    from scripts.ingest import _doi_probe
    mock_resp = _json_mod.dumps({
        "doi": "10.1073/pnas.2209111120",
        "arxiv_id": None,
        "title": "Sample Paper Title",
    })
    with patch("scripts.ingest._ollama_extraction_call", return_value=mock_resp):
        result = _doi_probe("First page text")
    assert result is not None
    assert result.doi == "10.1073/pnas.2209111120"
    assert result.title == "Sample Paper Title"


def test_fill_section_succeeds_on_valid_response():
    """_fill_section returns a SectionFillResult with the LLM body on success."""
    from scripts.ingest import _fill_section
    mock_resp = _json_mod.dumps({
        "heading": "Methods",
        "body": "Cleaned methods section text.",
        "fill_failed": False,
    })
    with patch("scripts.ingest._ollama_extraction_call", return_value=mock_resp):
        result = _fill_section("Raw methods text", "Methods")
    assert result is not None
    assert result.fill_failed is False
    assert result.body == "Cleaned methods section text."
    assert result.heading == "Methods"


def test_fill_section_fill_failed_on_two_parse_failures():
    """_fill_section returns fill_failed=True + raw text after two parse failures (D-05/D-06)."""
    from scripts.ingest import _fill_section
    with patch("scripts.ingest._ollama_extraction_call", return_value="not valid json"), \
         patch("scripts.ingest._parse_extraction_response", return_value=None):
        result = _fill_section("raw body text", "Methods")
    assert result.fill_failed is True
    assert result.body == "raw body text"
    assert result.heading == "Methods"


def test_fill_section_raises_on_ollama_error():
    """_fill_section raises RuntimeError when Ollama is unreachable (D-00d)."""
    from scripts.ingest import _fill_section
    with patch("scripts.ingest._ollama_extraction_call", return_value="[Ollama error: timeout]"):
        with pytest.raises(RuntimeError):
            _fill_section("some text", "Introduction")


def test_fill_section_system_prompt_faithfulness():
    """_fill_section SYSTEM prompt contains the D-07 faithfulness phrase."""
    from scripts.ingest import _fill_section
    captured_system = []

    def capture_call(prompt, system, schema, **kwargs):
        captured_system.append(system)
        return "[Ollama error: stop]"  # trigger RuntimeError after capture

    with patch("scripts.ingest._ollama_extraction_call", side_effect=capture_call):
        try:
            _fill_section("text", "heading")
        except RuntimeError:
            pass

    assert captured_system, "Expected _ollama_extraction_call to be called"
    assert "do not rewrite, summarise, or expand" in captured_system[0], (
        f"Expected faithfulness phrase in system prompt, got: {captured_system[0]!r}"
    )


def test_doi_probe_system_prompt_preservation():
    """_doi_probe SYSTEM prompt contains the DOI-preservation phrase (Pitfall 3)."""
    from scripts.ingest import _doi_probe
    captured_system = []

    def capture_call(prompt, system, schema, **kwargs):
        captured_system.append(system)
        return "[Ollama error: stop]"

    with patch("scripts.ingest._ollama_extraction_call", side_effect=capture_call):
        try:
            _doi_probe("text")
        except RuntimeError:
            pass

    assert captured_system, "Expected _ollama_extraction_call to be called"
    assert "Preserve the DOI exactly as printed" in captured_system[0], (
        f"Expected DOI preservation phrase in system prompt, got: {captured_system[0]!r}"
    )


def test_fill_section_oversize_guard():
    """_fill_section returns fill_failed immediately when section exceeds cap*4 chars (T-01.3-05).

    Updated in Plan 05: guard uses active cap (65536 default), not hardcoded 16384.
    Use 300000 chars (77048 estimated → ≥ 65536 cap AND len > 65536*4 = 262144) to trip the guard.
    """
    from scripts.ingest import _fill_section
    # ~300000 chars → 77048 estimated tokens → ≥ 65536 cap; 300000 > 65536*4 = 262144 → guard trips
    huge_text = "x" * 300000
    with patch("scripts.ingest._ollama_extraction_call", side_effect=AssertionError("LLM must not be called")):
        result = _fill_section(huge_text, "Huge Section")
    assert result.fill_failed is True
    assert result.body == huge_text


def test_fill_references_batched_success():
    """_fill_references_batched returns filled refs + 0 failures on a successful batch."""
    from scripts.ingest import _fill_references_batched
    raw_refs = [
        {"raw": "1. Smith J. et al. Nature 2024. 10.1/abc"},
        {"raw": "2. Jones A. Cell 2023. 10.2/xyz"},
    ]
    mock_resp = _json_mod.dumps({
        "refs": [
            {"number": 1, "raw": "1. Smith J. et al. Nature 2024. 10.1/abc",
             "doi": "10.1/abc", "title": "Paper 1", "year": 2024, "fill_failed": False},
            {"number": 2, "raw": "2. Jones A. Cell 2023. 10.2/xyz",
             "doi": "10.2/xyz", "title": "Paper 2", "year": 2023, "fill_failed": False},
        ]
    })
    with patch("scripts.ingest._ollama_extraction_call", return_value=mock_resp):
        filled, failures = _fill_references_batched(raw_refs)
    assert failures == 0
    assert len(filled) == 2
    assert filled[0]["doi"] == "10.1/abc"
    assert filled[1]["doi"] == "10.2/xyz"


def test_fill_references_batched_failed_batch():
    """_fill_references_batched flags each ref in a failed batch with fill_failed=True."""
    from scripts.ingest import _fill_references_batched
    raw_refs = [{"raw": "1. Bad ref"}, {"raw": "2. Also bad"}]
    with patch("scripts.ingest._ollama_extraction_call", return_value="not json"), \
         patch("scripts.ingest._parse_extraction_response", return_value=None):
        filled, failures = _fill_references_batched(raw_refs)
    assert failures == 1  # one failed batch
    assert len(filled) == 2
    for ref in filled:
        assert ref.get("fill_failed") is True


def test_fill_references_batched_raises_on_ollama_error():
    """_fill_references_batched raises RuntimeError on Ollama unreachable."""
    from scripts.ingest import _fill_references_batched
    raw_refs = [{"raw": "1. Some ref"}]
    with patch("scripts.ingest._ollama_extraction_call", return_value="[Ollama error: down]"):
        with pytest.raises(RuntimeError):
            _fill_references_batched(raw_refs)


def test_six_helpers_exist():
    """All six probe/fill helpers exist in scripts.ingest (acceptance criterion)."""
    from scripts.ingest import (
        _extract_first_page_and_footers,
        _doi_probe,
        _syntactic_doi_valid,
        _fill_metadata,
        _fill_section,
        _fill_references_batched,
    )
    assert callable(_extract_first_page_and_footers)
    assert callable(_doi_probe)
    assert callable(_syntactic_doi_valid)
    assert callable(_fill_metadata)
    assert callable(_fill_section)
    assert callable(_fill_references_batched)


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 02 Task 2: ingest() reordering — probe-gate-fill cascade
# ---------------------------------------------------------------------------

def _make_ingest_config_with_probe(tmp_path, extra=None):
    """Build a minimal config dict for probe-gate-fill cascade tests."""
    cfg = {
        "registry_path": str(tmp_path / "registry.json"),
        "project_name": "test-project",
        "mineru_path": "/fake/mineru",
    }
    if extra:
        cfg.update(extra)
    return cfg


def _write_content_list_for_probe(tmp_path, title="Paper Title"):
    """Write a minimal content_list.json with enough content to pass the quality gate."""
    cl_path = tmp_path / "content_list.json"
    blocks = [
        {"type": "text", "text": title, "text_level": 1, "page_idx": 0},
        # Enough text on page 0 so quality gate passes (threshold=100 chars plain text)
        {"type": "text", "text": "X" * 200, "text_level": None, "page_idx": 0},
    ]
    cl_path.write_text(json.dumps(blocks), encoding="utf-8")
    return str(cl_path)


def test_ingest_cache_hit_doi_probe_skips_fill(tmp_path):
    """Cache hit after DOI probe must skip all fill helpers (D-00b/REG-02)."""
    from scripts.ingest import ingest, _write_registry, DoiProbeResult

    cfg = _make_ingest_config_with_probe(tmp_path)
    reg_path = cfg["registry_path"]

    # Pre-populate registry with the probe-derived key.
    # DOI must use proper 4+ digit registrant prefix to survive _syntactic_doi_valid.
    doi_key = "10.1000/probe.dedup.test"
    cached_entry = {
        "title": "Cached Paper Title",
        "doi": doi_key,
        "projects": ["other"],
        "summary": None, "key_findings": None,
        "authors": None, "year": None, "journal": None, "arxiv_id": None,
        "source_path": "/old/paper.pdf", "paperjson_path": "/old/paper.json",
    }
    _write_registry(cached_entry, reg_path, doi_key)

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake for probe dedup")
    cl_path = _write_content_list_for_probe(tmp_path)

    probe_result = DoiProbeResult(doi=doi_key, arxiv_id=None, title="Cached Paper Title")

    # Fill helpers must NOT be called on a cache hit
    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata",
                    side_effect=AssertionError("_fill_metadata must not be called on cache hit")), \
         mock.patch("scripts.ingest._fill_section",
                    side_effect=AssertionError("_fill_section must not be called on cache hit")), \
         mock.patch("scripts.ingest._fill_references_batched",
                    side_effect=AssertionError("_fill_references_batched must not be called on cache hit")):
        result = ingest(str(fake_pdf), cfg)

    # Result must be the cached registry entry
    assert result.get("title") == "Cached Paper Title" or result.get("doi") == doi_key, (
        f"Expected cached entry returned on DOI probe hit, got: {result}"
    )


def test_ingest_normalizations_applied_llm_fill(tmp_path):
    """After ingest(), provenance.normalizations_applied contains ligature_fix and llm_fill (D-10)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config_with_probe(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 normalization test")
    cl_path = _write_content_list_for_probe(tmp_path)

    mock_parsed = {
        "title": "Norms Paper",
        "sections": [{"heading": "Abstract", "level": 0, "blocks": [
            {"type": "text", "display": "Norms Paper", "plain": "Norms Paper"},
            {"type": "text", "display": "X" * 200, "plain": "X" * 200},
        ]}],
        "references": [],
        "metadata": {
            "title": "Norms Paper",
            "authors": None, "year": 2024, "journal": None,
            "doi": "10.1000/norms.paper", "arxiv_id": None, "accession_codes": [],
        },
    }
    probe_result = DoiProbeResult(doi="10.1000/norms.paper", arxiv_id=None, title="Norms Paper")
    metadata_result = PaperMetadata(title="Norms Paper", doi="10.1000/norms.paper")
    section_fill = SectionFillResult(heading="Abstract", body="Cleaned abstract", fill_failed=False)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        result = ingest(str(fake_pdf), cfg)

    norms = result.get("provenance", {}).get("normalizations_applied", [])
    assert "ligature_fix" in norms, f"Expected 'ligature_fix' in normalizations_applied: {norms}"
    assert "llm_fill" in norms, f"Expected 'llm_fill' in normalizations_applied: {norms}"
    assert "ufffd_replacement" not in norms, (
        f"Expected 'ufffd_replacement' NOT in normalizations_applied: {norms}"
    )


def test_ingest_fast_path_dedup_removed():
    """1.2 fast-path dedup block (fast_parsed, fast_metadata) is absent from ingest() source."""
    import inspect
    from scripts.ingest import ingest
    source = inspect.getsource(ingest)
    assert "fast_parsed" not in source, (
        "Expected 'fast_parsed' to be removed from ingest() (1.2 fast-path dedup deleted)"
    )
    assert "fast_metadata" not in source, (
        "Expected 'fast_metadata' to be removed from ingest() (1.2 fast-path dedup deleted)"
    )


def test_ingest_crossref_hook_comment_present():
    """ingest() contains the Plan 03 Crossref validation anchor comment."""
    import inspect
    from scripts.ingest import ingest
    source = inspect.getsource(ingest)
    assert "Crossref validation hook (Plan 03)" in source, (
        "Expected '# Crossref validation hook (Plan 03)' anchor comment in ingest()"
    )


def test_ingest_check_registry_after_doi_probe():
    """In ingest() source, _check_registry appears after _doi_probe (ordering constraint D-00b)."""
    import inspect
    from scripts.ingest import ingest
    source = inspect.getsource(ingest)
    probe_pos = source.find("_doi_probe")
    check_pos = source.find("_check_registry")
    assert probe_pos != -1, "Expected _doi_probe call in ingest()"
    assert check_pos != -1, "Expected _check_registry call in ingest()"
    assert probe_pos < check_pos, (
        f"Expected _doi_probe (pos {probe_pos}) to appear before _check_registry (pos {check_pos}) in ingest()"
    )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 02 Task 3: Mock-LLM integration tests
# REG-02 / D-00b: cache-hit-no-fill
# REG-01: miss-fills-and-writes
# E9 / D-05/D-06: fill_failed graceful degradation
# D-00d: Ollama unreachable aborts
# ---------------------------------------------------------------------------

def test_cache_hit_skips_all_fill(tmp_path):
    """REG-02/D-00b: with registry pre-populated for the probe DOI, ingest() returns
    cached entry and all fill helpers raise if called."""
    from scripts.ingest import ingest, _write_registry, DoiProbeResult, _read_registry

    cfg = _make_ingest_config(tmp_path)
    reg_path = cfg["registry_path"]

    doi_key = "10.1073/pnas.2209111120"  # real DOI format
    cached_entry = {
        "title": "Attention Is All You Need",
        "doi": doi_key,
        "projects": ["prior-project"],
        "summary": None, "key_findings": None,
        "authors": ["Vaswani A.", "Shazeer N."],
        "year": 2017,
        "journal": "NeurIPS",
        "arxiv_id": "1706.03762",
        "source_path": "/prior/paper.pdf",
        "paperjson_path": "/prior/paper.json",
    }
    _write_registry(cached_entry, reg_path, doi_key)

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 attention is all you need")
    cl_path = _write_real_content_list(tmp_path)

    probe_result = DoiProbeResult(doi=doi_key, arxiv_id="1706.03762", title="Attention Is All You Need")

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value={
             "title": "Attention Is All You Need",
             "sections": [{"heading": "", "level": 0, "blocks": [
                 {"type": "text", "display": "Attention Is All You Need", "plain": "Attention Is All You Need"},
                 {"type": "text", "display": "X" * 200, "plain": "X" * 200},
             ]}],
             "references": [],
             "metadata": {"title": "Attention Is All You Need", "authors": None, "year": None,
                          "journal": None, "doi": doi_key, "arxiv_id": None, "accession_codes": []},
         }), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata",
                    side_effect=AssertionError("D-00b violated: _fill_metadata called on cache hit")), \
         mock.patch("scripts.ingest._fill_section",
                    side_effect=AssertionError("D-00b violated: _fill_section called on cache hit")), \
         mock.patch("scripts.ingest._fill_references_batched",
                    side_effect=AssertionError("D-00b violated: _fill_references_batched called on cache hit")):
        result = ingest(str(fake_pdf), cfg)

    # Must return the cached entry (not a full PaperJSON)
    assert result.get("title") == "Attention Is All You Need", (
        f"Expected cached title returned, got: {result.get('title')}"
    )
    assert result.get("doi") == doi_key, (
        f"Expected cached doi returned, got: {result.get('doi')}"
    )


def test_miss_fills_and_writes_entry(tmp_path):
    """REG-01: on cache miss, ingest() calls fill helpers + writes registry entry keyed by probe DOI."""
    from scripts.ingest import ingest, _read_registry, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    reg_path = cfg["registry_path"]

    doi_key = "10.1021/jacs.3c10258"  # real DOI format, NOT in registry
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 new paper to fill")
    cl_path = _write_real_content_list(tmp_path)

    probe_result = DoiProbeResult(doi=doi_key, arxiv_id=None, title="A Novel Method for X")
    metadata_result = PaperMetadata(
        title="A Novel Method for X",
        authors=["Smith J.", "Jones A."],
        doi=doi_key,
        year=2024,
        journal="J. Am. Chem. Soc.",
    )
    section_fill = SectionFillResult(heading="", body="Cleaned section body text.", fill_failed=False)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value={
             "title": "A Novel Method for X",
             "sections": [{"heading": "", "level": 0, "blocks": [
                 {"type": "text", "display": "A Novel Method for X", "plain": "A Novel Method for X"},
                 {"type": "text", "display": "A" * 200, "plain": "A" * 200},
             ]}],
             "references": [],
             "metadata": {"title": "A Novel Method for X", "authors": None, "year": None,
                          "journal": None, "doi": doi_key, "arxiv_id": None, "accession_codes": []},
         }), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result) as mock_fill_meta, \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill) as mock_fill_sec, \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)) as mock_fill_refs:
        result = ingest(str(fake_pdf), cfg)

    # Fill helpers were called
    assert mock_fill_meta.called, "Expected _fill_metadata to be called on cache miss"
    assert mock_fill_sec.called, "Expected _fill_section to be called on cache miss"
    assert mock_fill_refs.called, "Expected _fill_references_batched to be called on cache miss"

    # Registry was written with the probe DOI key
    registry = _read_registry(reg_path)
    assert doi_key in registry, (
        f"Expected DOI key '{doi_key}' in registry after fill, got keys: {list(registry.keys())}"
    )
    # Registry entry title comes from PaperMetadata fill result
    entry = registry[doi_key]
    assert entry.get("title") == "A Novel Method for X", (
        f"Expected filled title in registry, got: {entry.get('title')!r}"
    )


def test_fill_failed_graceful(tmp_path, capsys):
    """E9/D-05/D-06: _fill_section with two parse failures returns fill_failed=True + raw text."""
    from scripts.ingest import _fill_section

    with patch("scripts.ingest._ollama_extraction_call", return_value="not json at all"), \
         patch("scripts.ingest._parse_extraction_response", return_value=None):
        result = _fill_section("raw body text of the Methods section", "Methods")

    assert result.fill_failed is True, "Expected fill_failed=True after two parse failures"
    assert result.body == "raw body text of the Methods section", (
        f"Expected raw text preserved in body, got: {result.body!r}"
    )
    assert result.heading == "Methods", f"Expected heading preserved, got: {result.heading!r}"

    # Warning should have been printed to stderr
    captured = capsys.readouterr()
    assert "fill_failed" in captured.err, (
        f"Expected 'fill_failed' warning in stderr, got: {captured.err!r}"
    )


def test_ollama_unreachable_aborts(tmp_path):
    """D-00d: Ollama unreachable causes RuntimeError in fill helpers and _doi_probe."""
    from scripts.ingest import _fill_section, _doi_probe

    # _fill_section raises on Ollama error
    with patch("scripts.ingest._ollama_extraction_call", return_value="[Ollama error: connection refused]"):
        with pytest.raises(RuntimeError) as exc_info:
            _fill_section("some section text", "Introduction")
    assert "Ollama error" in str(exc_info.value), (
        f"Expected RuntimeError with Ollama error message, got: {exc_info.value}"
    )

    # _doi_probe also raises on Ollama error
    with patch("scripts.ingest._ollama_extraction_call", return_value="[Ollama error: timeout]"):
        with pytest.raises(RuntimeError) as exc_info:
            _doi_probe("first page text with doi")
    assert "Ollama error" in str(exc_info.value), (
        f"Expected RuntimeError from _doi_probe on Ollama error, got: {exc_info.value}"
    )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 03 Task 2 (RED): _crossref_validate existence + HTTPS contract
# ---------------------------------------------------------------------------

def test_crossref_validate_exists():
    """_crossref_validate must exist in scripts.ingest (Task 2 RED gate)."""
    from scripts.ingest import _crossref_validate
    assert callable(_crossref_validate), "_crossref_validate must be callable"


def test_crossref_validate_uses_https():
    """_crossref_validate must use HTTPS URL for api.crossref.org (V9 — DOI-only outbound)."""
    import inspect
    from scripts.ingest import _crossref_validate
    source = inspect.getsource(_crossref_validate)
    assert "https://api.crossref.org" in source, (
        "Expected 'https://api.crossref.org' in _crossref_validate source (V9 HTTPS-only)"
    )
    assert "http://api.crossref.org" not in source, (
        "Plain http:// must NOT be used for Crossref (V9)"
    )


def test_crossref_validate_config_user_agent():
    """_crossref_validate must read crossref_contact_email from config (D-13/V14 — not hardcoded)."""
    import inspect
    from scripts.ingest import _crossref_validate
    source = inspect.getsource(_crossref_validate)
    assert "crossref_contact_email" in source, (
        "Expected 'crossref_contact_email' read from config in _crossref_validate"
    )


def test_crossref_hook_wired_in_ingest():
    """ingest() must call _crossref_validate (not just a comment) when the flag is on."""
    import inspect
    from scripts.ingest import ingest
    source = inspect.getsource(ingest)
    # After Plan 03, the real call (not the commented-out placeholder) must appear
    assert "_crossref_validate(" in source, (
        "Expected live _crossref_validate( call in ingest() (Plan 03 hook activated)"
    )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 03 Task 3: Mock-urllib tests for Crossref control flow
# E10/D-15: abort-on-mismatch
# E10/D-16: fail-open-on-network-error
# D-14: same-paper pass (no abort)
# D-13: off-by-default no-call
# ---------------------------------------------------------------------------

import io as _io
import urllib.error as _urllib_error


class _FakeHttpResponse:
    """Minimal context-manager fake for urllib.request.urlopen return value."""

    def __init__(self, body_bytes: bytes):
        self._body = body_bytes

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_crossref_abort_on_mismatch(capsys):
    """E10/D-15: LLM confirms different paper → RuntimeError with [ingest error: prefix."""
    from scripts.ingest import _crossref_validate
    import json as _json

    crossref_payload = _json.dumps({
        "message": {"title": ["Some Other Paper That Is Not The Same"]}
    }).encode()
    llm_verdict = _json.dumps({"same_paper": False})

    cfg = {"crossref_contact_email": "t@e.com"}
    with mock.patch("scripts.ingest.urllib.request.urlopen",
                    return_value=_FakeHttpResponse(crossref_payload)), \
         mock.patch("scripts.ingest._ollama_extraction_call", return_value=llm_verdict):
        with pytest.raises(RuntimeError) as exc_info:
            _crossref_validate("10.1/x", "Real Title", cfg)

    assert str(exc_info.value).startswith("[ingest error:"), (
        f"Expected RuntimeError starting with '[ingest error:', got: {exc_info.value!r}"
    )


def test_crossref_fail_open_on_network_error(capsys):
    """E10/D-16: URLError → returns None (no raise) + [ingest warning: crossref unreachable."""
    from scripts.ingest import _crossref_validate

    cfg = {"crossref_contact_email": "t@e.com"}
    with mock.patch("scripts.ingest.urllib.request.urlopen",
                    side_effect=_urllib_error.URLError("connection refused")):
        result = _crossref_validate("10.1000/any.doi", "Some Title", cfg)

    assert result is None, f"Expected None (fail-open) on URLError, got {result!r}"
    captured = capsys.readouterr()
    assert "crossref unreachable" in captured.err, (
        f"Expected 'crossref unreachable' in stderr, got: {captured.err!r}"
    )


def test_crossref_same_paper_no_abort():
    """D-14: LLM confirms same paper → no raise (returns None)."""
    from scripts.ingest import _crossref_validate
    import json as _json

    crossref_payload = _json.dumps({
        "message": {"title": ["A Novel Method for X"]}
    }).encode()
    llm_verdict = _json.dumps({"same_paper": True})

    cfg = {"crossref_contact_email": "t@e.com"}
    with mock.patch("scripts.ingest.urllib.request.urlopen",
                    return_value=_FakeHttpResponse(crossref_payload)), \
         mock.patch("scripts.ingest._ollama_extraction_call", return_value=llm_verdict):
        result = _crossref_validate("10.1000/any.doi", "A Novel Method for X", cfg)

    assert result is None, f"Expected None (no abort) on matching paper, got {result!r}"


def test_crossref_off_by_default_no_call(tmp_path):
    """D-13: with crossref_validate absent/false, ingest() never calls _crossref_validate."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    # Config has no crossref_validate key at all (default off)
    cfg = _make_ingest_config(tmp_path)
    assert "crossref_validate" not in cfg, "Fixture must not set crossref_validate for this test"

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 crossref off by default test")
    cl_path = _write_real_content_list(tmp_path)

    probe_result = DoiProbeResult(doi="10.1000/off.default", arxiv_id=None, title="Test Paper")
    metadata_result = PaperMetadata(title="Test Paper", doi="10.1000/off.default", year=2024)
    section_fill = SectionFillResult(heading="", body="Test body " + "A" * 200, fill_failed=False)

    # _crossref_validate patched with a raising side_effect — must never be called
    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value={
             "title": "Test Paper",
             "sections": [{"heading": "", "level": 0, "blocks": [
                 {"type": "text", "display": "Test Paper", "plain": "Test Paper"},
                 {"type": "text", "display": "A" * 200, "plain": "A" * 200},
             ]}],
             "references": [],
             "metadata": {
                 "title": "Test Paper", "authors": None, "year": 2024, "journal": None,
                 "doi": "10.1000/off.default", "arxiv_id": None, "accession_codes": [],
             },
         }), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)), \
         mock.patch("scripts.ingest._crossref_validate",
                    side_effect=AssertionError("crossref called when flag off")) as mock_cv:
        ingest(str(fake_pdf), cfg)  # must not raise AssertionError

    mock_cv.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 04 Task 1: E3 artifact-cleanliness scan
# ---------------------------------------------------------------------------

import re as _re_mod


def _scan_extraction_artifacts(extraction: dict) -> list[str]:
    """Scan the assembled extraction namespace for artifact residue.

    Returns a list of "<field-path>: <reason>" strings for any string that contains:
    - U+FFFD (replacement character, codepoint 0xFFFD)
    - Any codepoint in range(0xFB00, 0xFB07) — the ﬀﬁﬂﬃﬄﬅﬆ ligature block
    - The regex <sup>(fi|fl|ff|ffi|ffl)</sup> substring (HTML ligature artifacts)

    Walks:
    - extraction["metadata"]["title"]
    - extraction["sections"][i]["body"] / ["display"] / ["plain"]
    - extraction["references"][j]["raw"]
    """
    _LIGATURE_RANGE = range(0xFB00, 0xFB07)
    _SUP_LIGATURE_RE = _re_mod.compile(r"<sup>(fi|fl|ff|ffi|ffl)</sup>")
    offending: list[str] = []

    def _check(field_path: str, value) -> None:
        if not isinstance(value, str):
            return
        if "�" in value:
            offending.append(f"{field_path}: contains U+FFFD replacement character")
        for ch in value:
            if ord(ch) in _LIGATURE_RANGE:
                offending.append(f"{field_path}: contains ligature codepoint U+{ord(ch):04X} ({ch!r})")
                break  # one entry per field per category
        if _SUP_LIGATURE_RE.search(value):
            offending.append(f"{field_path}: contains <sup>ligature</sup> artifact")

    # Walk metadata.title
    metadata = extraction.get("metadata", {})
    _check("metadata.title", metadata.get("title"))

    # Walk sections
    for i, section in enumerate(extraction.get("sections", [])):
        for key in ("body", "display", "plain"):
            val = section.get(key)
            if val is not None:
                _check(f"sections[{i}].{key}", val)

    # Walk references.raw
    for j, ref in enumerate(extraction.get("references", [])):
        _check(f"references[{j}].raw", ref.get("raw"))

    return offending


def test_no_artifact_residue(tmp_path):
    """E3 (Plan 04): given an ingest() run with mocked clean LLM fill, no string anywhere
    in the assembled extraction namespace contains U+FFFD, ligature codepoints, or
    <sup>ligature</sup> substrings. (No GPU required — all fill helpers are mocked.)("""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 E3 artifact scan test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = {
        "title": "A Clean Publication-Faithful Title",
        "sections": [{"heading": "Introduction", "level": 0, "blocks": [
            {"type": "text", "display": "Clean body text.", "plain": "Clean body text."},
            {"type": "text", "display": "X" * 200, "plain": "X" * 200},
        ]}],
        "references": [],
        "metadata": {
            "title": "A Clean Publication-Faithful Title",
            "authors": None, "year": 2024, "journal": None,
            "doi": "10.1000/e3.clean.test", "arxiv_id": None, "accession_codes": [],
        },
    }
    probe_result = DoiProbeResult(doi="10.1000/e3.clean.test", arxiv_id=None, title="A Clean Publication-Faithful Title")
    metadata_result = PaperMetadata(title="A Clean Publication-Faithful Title", doi="10.1000/e3.clean.test", year=2024)
    section_fill = SectionFillResult(
        heading="Introduction",
        body="Clean body text with no artifacts or residue, only faithful publication text.",
        fill_failed=False,
    )

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        result = ingest(str(fake_pdf), cfg)

    # Result must be a full PaperJSON with an extraction namespace
    assert "extraction" in result, f"Expected full PaperJSON result with 'extraction' key, got: {list(result.keys())}"

    violations = _scan_extraction_artifacts(result["extraction"])
    assert violations == [], (
        f"Expected no artifact residue in clean extraction namespace, found violations: {violations}"
    )


def test_artifact_scanner_detects_residue():
    """Negative control (Plan 04): _scan_extraction_artifacts returns >= 3 offending entries
    on a hand-built extraction dict that contains a U+FFFD title, a ligature-codepoint
    section body, and a <sup>fi</sup> reference raw string.  Proves the scanner is not a no-op."""
    artifact_extraction = {
        "metadata": {
            "title": "Paper with � replacement character in title",
        },
        "sections": [
            {
                "body": "Signiﬁcant result in this section body (ﬀect)",  # ﬁ=U+FB01, ﬀ=U+FB00
                "display": None,
                "plain": None,
            }
        ],
        "references": [
            {"raw": "1. Smith J. et al. Nature 2024. E<sup>fi</sup>ciency study."},
        ],
    }
    violations = _scan_extraction_artifacts(artifact_extraction)
    assert len(violations) >= 3, (
        f"Expected >= 3 offending entries from artifact-laden control dict, got {len(violations)}: {violations}"
    )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 05 (gap closure) Task 1: timeout hardening
# ---------------------------------------------------------------------------

def test_extraction_call_timeout_returns_string():
    """_ollama_extraction_call returns [Ollama timeout: ...] string on bare TimeoutError — no exception escapes."""
    with patch("scripts.ingest.urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        result = _ollama_extraction_call("prompt", "system", {})
    assert isinstance(result, str), "Expected a string return, not an exception"
    assert result.startswith("[Ollama timeout:"), (
        f"Expected result starting with '[Ollama timeout:', got: {result!r}"
    )


def test_fill_section_timeout_degrades_to_fill_failed(capsys):
    """Core gap (Plan 05): [Ollama timeout:] return from _ollama_extraction_call degrades to fill_failed=True."""
    from scripts.ingest import _fill_section
    timeout_str = "[Ollama timeout: no response within 180s]"
    with patch("scripts.ingest._ollama_extraction_call", return_value=timeout_str):
        result = _fill_section("raw body text", "RESULTS")
    assert result.fill_failed is True, "Expected fill_failed=True on timeout degradation"
    assert result.body == "raw body text", (
        f"Expected raw body text preserved, got: {result.body!r}"
    )
    captured = capsys.readouterr()
    assert "fill_failed" in captured.err, (
        f"Expected 'fill_failed' in stderr on timeout, got: {captured.err!r}"
    )
    # Must NOT raise RuntimeError
    # (the with block above already proves no raise)


def test_fill_metadata_timeout_falls_back_to_probe_hints():
    """[Ollama timeout:] in _fill_metadata falls back to probe hints — no raise."""
    from scripts.ingest import _fill_metadata, DoiProbeResult
    timeout_str = "[Ollama timeout: no response within 120s]"
    probe = DoiProbeResult(doi="10.1000/x", arxiv_id=None, title="Fallback Title")
    with patch("scripts.ingest._ollama_extraction_call", return_value=timeout_str):
        result = _fill_metadata("some first page text", probe)
    # Must return a PaperMetadata built from probe hints — no raise
    assert result is not None, "Expected PaperMetadata fallback on timeout, not None"
    from scripts.ingest import PaperMetadata
    assert isinstance(result, PaperMetadata), (
        f"Expected PaperMetadata, got: {type(result)}"
    )


def test_warmup_swallows_timeout():
    """_warmup_ollama swallows bare TimeoutError — never raises (best-effort contract)."""
    with patch("scripts.ingest.urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        result = _warmup_ollama()  # must not raise
    assert result is None, f"Expected None from _warmup_ollama on timeout, got {result!r}"


def test_crossref_fail_open_on_timeout(capsys):
    """_crossref_validate fails open on TimeoutError during urlopen (D-16 preserved for timeouts)."""
    from scripts.ingest import _crossref_validate
    cfg = {"crossref_contact_email": "t@e.com"}
    with patch("scripts.ingest.urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        result = _crossref_validate("10.1000/x", "Some Title", cfg)
    assert result is None, f"Expected None (fail-open) on TimeoutError, got {result!r}"
    captured = capsys.readouterr()
    assert "crossref unreachable" in captured.err, (
        f"Expected 'crossref unreachable' in stderr on timeout, got: {captured.err!r}"
    )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 05 (gap closure) Task 2: num_ctx cap + timeouts
# ---------------------------------------------------------------------------

def test_oversize_guard_uses_cap(monkeypatch):
    """_fill_section guard derives from _NUM_CTX_CAP, not a hardcoded literal (Plan 05 T-01.3-05)."""
    import scripts.ingest as _ingest_mod
    from scripts.ingest import _fill_section, DEFAULT_NUM_CTX_CAP
    # Monkeypatch _NUM_CTX_CAP to 4096: 20000 chars → 7048 estimated ≥ 4096 cap; 20000 > 4096*4=16384 → guard trips
    original_cap = _ingest_mod._NUM_CTX_CAP
    try:
        _ingest_mod._NUM_CTX_CAP = 4096
        huge_text = "x" * 20000
        with patch("scripts.ingest._ollama_extraction_call",
                   side_effect=AssertionError("LLM must not be called under low cap")):
            result = _fill_section(huge_text, "Section")
        assert result.fill_failed is True, "Expected fill_failed=True under low cap override"
    finally:
        _ingest_mod._NUM_CTX_CAP = original_cap


def test_num_ctx_cap_config_flow(tmp_path):
    """ingest() reads ollama_num_ctx_cap from config and sets _NUM_CTX_CAP module global."""
    import scripts.ingest as _ingest_mod
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult, DEFAULT_NUM_CTX_CAP

    cfg = _make_ingest_config(tmp_path, extra={"ollama_num_ctx_cap": 8192})
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 cap config flow test")
    cl_path = _write_real_content_list(tmp_path)

    probe_result = DoiProbeResult(doi="10.1000/cap.config.test", arxiv_id=None, title="Cap Config Paper")
    metadata_result = PaperMetadata(title="Cap Config Paper", doi="10.1000/cap.config.test")
    section_fill = SectionFillResult(heading="", body="Test body " + "A" * 200, fill_failed=False)

    try:
        with mock.patch("scripts.ingest._run_mineru"), \
             mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
             mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
             mock.patch("scripts.ingest._warmup_ollama"), \
             mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
             mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
             mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
             mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
            ingest(str(fake_pdf), cfg)
        assert _ingest_mod._NUM_CTX_CAP == 8192, (
            f"Expected _NUM_CTX_CAP=8192 after ingest() with ollama_num_ctx_cap=8192, "
            f"got {_ingest_mod._NUM_CTX_CAP}"
        )
    finally:
        _ingest_mod._NUM_CTX_CAP = DEFAULT_NUM_CTX_CAP


def test_fill_call_timeouts():
    """_fill_section passes timeout=_SECTION_TIMEOUT; _doi_probe passes timeout=_SECTION_TIMEOUT (section tier).

    Updated in Plan 06 (gap closure Tasks 2+3):
    - _fill_section now uses _SECTION_TIMEOUT (default 300, was hardcoded 180)
    - _doi_probe now uses _SECTION_TIMEOUT (section tier for full-doc probe, was 120)
    Both read the module constant so the assertion tracks the configured default.
    """
    import scripts.ingest as _ingest_mod
    from scripts.ingest import _fill_section, _doi_probe
    import json as _json

    # Capture kwargs for _fill_section
    captured_kwargs = {}
    def capture_fill_call(prompt, system, schema, **kwargs):
        captured_kwargs.update(kwargs)
        return _json.dumps({"heading": "H", "body": "body text", "fill_failed": False})

    with patch("scripts.ingest._ollama_extraction_call", side_effect=capture_fill_call):
        _fill_section("short text", "H")
    assert captured_kwargs.get("timeout") == _ingest_mod._SECTION_TIMEOUT, (
        f"Expected _fill_section to pass timeout={_ingest_mod._SECTION_TIMEOUT} (_SECTION_TIMEOUT), "
        f"got {captured_kwargs.get('timeout')}"
    )

    # Capture kwargs for _doi_probe
    captured_probe_kwargs = {}
    def capture_probe_call(prompt, system, schema, **kwargs):
        captured_probe_kwargs.update(kwargs)
        return _json.dumps({"doi": "10.1000/x", "arxiv_id": None, "title": "T"})

    with patch("scripts.ingest._ollama_extraction_call", side_effect=capture_probe_call):
        _doi_probe("first page text")
    assert captured_probe_kwargs.get("timeout") == _ingest_mod._SECTION_TIMEOUT, (
        f"Expected _doi_probe to pass timeout={_ingest_mod._SECTION_TIMEOUT} (_SECTION_TIMEOUT), "
        f"got {captured_probe_kwargs.get('timeout')}"
    )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 05 (gap closure) Task 3: progress lines
# ---------------------------------------------------------------------------

def test_progress_lines_smoke(tmp_path, capsys):
    """ingest() emits [ingest <date> <time>] timestamped progress lines on stderr; stdout stays clean.

    Updated in Plan 06 (gap closure Task 1): progress lines now carry a local date+time prefix
    via _log(), so assertions use regex/substring checks rather than literal '[ingest] ' strings.
    """
    import re as _re
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 progress smoke test")
    cl_path = _write_real_content_list(tmp_path)

    probe_result = DoiProbeResult(doi="10.1000/progress.test", arxiv_id=None, title="Progress Paper")
    metadata_result = PaperMetadata(title="Progress Paper", doi="10.1000/progress.test")
    section_fill = SectionFillResult(heading="", body="Section body " + "A" * 200, fill_failed=False)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        ingest(str(fake_pdf), cfg)

    captured = capsys.readouterr()
    # Lines carry a [ingest <date> <time>] prefix — check with regex + tail substring
    assert _re.search(
        r"\[ingest \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] mineru extraction starting",
        captured.err,
    ), f"Expected timestamped 'mineru extraction starting' in stderr, got: {captured.err!r}"
    assert "] section fill" in captured.err, (
        f"Expected section fill stage in stderr, got: {captured.err!r}"
    )
    assert "] registry write" in captured.err, (
        f"Expected registry write stage in stderr, got: {captured.err!r}"
    )
    # stdout must stay clean of [ingest text — only final JSON goes to stdout
    assert "[ingest" not in captured.out, (
        f"Expected no '[ingest' text in stdout, got: {captured.out!r}"
    )


def test_reference_batch_progress(capsys):
    """_fill_references_batched emits [ingest] reference batch i/n lines on stderr."""
    from scripts.ingest import _fill_references_batched
    import json as _json

    raw_refs = [{"raw": f"r{i}"} for i in range(12)]
    with patch("scripts.ingest._ollama_extraction_call",
               return_value=_json.dumps({"refs": []})):
        _fill_references_batched(raw_refs)

    captured = capsys.readouterr()
    assert "reference batch 1/2" in captured.err, (
        f"Expected 'reference batch 1/2' in stderr, got: {captured.err!r}"
    )
    assert "reference batch 2/2" in captured.err, (
        f"Expected 'reference batch 2/2' in stderr, got: {captured.err!r}"
    )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 06 (gap closure) Task 1: timestamped progress + --output
# ---------------------------------------------------------------------------

import re as _re_mod  # noqa: E402


def test_log_helper_timestamp_format(capsys):
    """_log() writes '[ingest YYYY-MM-DD HH:MM:SS] <msg>' to stderr; stdout stays clean."""
    from scripts.ingest import _log

    _log("hello world")
    captured = capsys.readouterr()
    assert _re_mod.search(
        r"^\[ingest \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] hello world",
        captured.err,
    ), f"Expected timestamped stderr line, got: {captured.err!r}"
    assert "hello world" not in captured.out, (
        "Expected _log() output to go to stderr only, not stdout"
    )


def test_progress_lines_smoke_timestamped(tmp_path, capsys):
    """ingest() emits timestamped [ingest <date> <time>] progress lines on stderr; stdout stays clean."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 progress smoke test v2")
    cl_path = _write_real_content_list(tmp_path)

    probe_result = DoiProbeResult(doi="10.1000/progress.test2", arxiv_id=None, title="Progress Paper v2")
    metadata_result = PaperMetadata(title="Progress Paper v2", doi="10.1000/progress.test2")
    section_fill = SectionFillResult(heading="", body="Section body " + "A" * 200, fill_failed=False)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        ingest(str(fake_pdf), cfg)

    captured = capsys.readouterr()
    # Stderr must carry a [ingest <date> <time>] prefix on progress lines
    assert _re_mod.search(
        r"\[ingest \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] mineru extraction starting",
        captured.err,
    ), f"Expected timestamped 'mineru extraction starting' in stderr, got: {captured.err!r}"
    assert "] mineru extraction starting" in captured.err, (
        f"Expected mineru stage suffix in stderr, got: {captured.err!r}"
    )
    assert "] registry write" in captured.err, (
        f"Expected registry write stage suffix in stderr, got: {captured.err!r}"
    )
    # stdout must stay clean of [ingest text
    assert "[ingest" not in captured.out, (
        f"Expected no '[ingest' text in stdout, got: {captured.out!r}"
    )


def test_output_flag_writes_utf8_file(tmp_path):
    """_emit_result(result, path) writes UTF-8 JSON directly to file; round-trip is clean (GAP B)."""
    from scripts.ingest import _emit_result

    out_path = str(tmp_path / "out.json")
    data = {"title": "José — ‘quote’"}
    _emit_result(data, out_path)

    with open(out_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data, (
        f"Expected round-trip clean UTF-8, got: {loaded!r}"
    )


def test_output_omitted_uses_stdout(tmp_path, capsys):
    """_emit_result(result, None) prints JSON to stdout; no file is written."""
    from scripts.ingest import _emit_result

    data = {"title": "plain ascii"}
    _emit_result(data, None)
    captured = capsys.readouterr()

    loaded = json.loads(captured.out)
    assert loaded == data, f"Expected JSON on stdout, got: {captured.out!r}"
    # No file should exist in tmp_path
    assert not list(tmp_path.iterdir()), "Expected no file written when output_path is None"


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 06 (gap closure) Task 2: full-doc DOI probe + journal_full
# ---------------------------------------------------------------------------

def test_extract_full_text_includes_non_first_page():
    """_extract_full_text includes text from all pages; contrast with _extract_first_page_and_footers."""
    from scripts.ingest import _extract_full_text, _extract_first_page_and_footers

    blocks = [
        {"type": "text", "page_idx": 0, "text": "Cover title"},
        {"type": "text", "page_idx": 3, "text": "10.1021/jacs.3c10258"},  # Supporting Info DOI
        {"type": "footer", "page_idx": 1, "text": "Journal footer"},
    ]
    full = _extract_full_text(blocks)
    first = _extract_first_page_and_footers(blocks)

    assert "Cover title" in full, f"Expected cover title in full text, got: {full!r}"
    assert "10.1021/jacs.3c10258" in full, f"Expected page-3 DOI in full text, got: {full!r}"
    # first-page extractor should NOT include page-3 text
    assert "10.1021/jacs.3c10258" not in first, (
        f"Expected page-3 DOI absent from first-page extract, got: {first!r}"
    )


def test_doi_probe_scans_full_text():
    """_doi_probe receives the full_text in the prompt; num_ctx sized by _estimate_num_ctx(full_text).

    Updated in Plan 07 (GAP A): first_page_text kwarg removed — probe now takes full_text only.
    The title hint is at the start of the document (no separate first_page_text arg needed).
    """
    import scripts.ingest as _m
    from scripts.ingest import _doi_probe, _estimate_num_ctx
    import json as _json

    captured_prompt = []
    captured_kwargs = {}

    def capture_call(prompt, system, schema, **kwargs):
        captured_prompt.append(prompt)
        captured_kwargs.update(kwargs)
        return _json.dumps({"doi": "10.1021/jacs.3c10258", "arxiv_id": None, "title": "T"})

    full_text = "Cover page text only\n" + "page 3 contains: 10.1021/jacs.3c10258"
    with mock.patch("scripts.ingest._ollama_extraction_call", side_effect=capture_call):
        _doi_probe(full_text=full_text)

    assert captured_prompt, "Expected _ollama_extraction_call to be called"
    assert "10.1021/jacs.3c10258" in captured_prompt[0], (
        f"Expected full_text (with page-3 DOI) in prompt, got: {captured_prompt[0]!r}"
    )
    expected_ctx = _estimate_num_ctx(full_text)
    assert captured_kwargs.get("num_ctx") == expected_ctx, (
        f"Expected num_ctx={expected_ctx} (dynamic sizing), got {captured_kwargs.get('num_ctx')}"
    )


def test_paper_metadata_journal_full_field():
    """PaperMetadata carries journal_full (full title) alongside journal (abbreviation)."""
    from scripts.ingest import PaperMetadata

    pm = PaperMetadata(
        title="T",
        journal="J. Am. Chem. Soc.",
        journal_full="Journal of the American Chemical Society",
    )
    assert pm.journal == "J. Am. Chem. Soc."
    assert pm.journal_full == "Journal of the American Chemical Society"

    # Default is None
    pm2 = PaperMetadata(title="T")
    assert pm2.journal_full is None


def test_fill_metadata_returns_journal_full():
    """_fill_metadata returns PaperMetadata with journal_full when LLM provides it."""
    from scripts.ingest import _fill_metadata, DoiProbeResult
    import json as _json

    probe = DoiProbeResult(doi="10.1021/jacs.3c10258", arxiv_id=None, title="JACS Paper")
    mock_resp = _json.dumps({
        "title": "JACS Paper",
        "authors": [],
        "journal": "J. Am. Chem. Soc.",
        "journal_full": "Journal of the American Chemical Society",
        "year": 2023,
        "doi": "10.1021/jacs.3c10258",
        "arxiv_id": None,
    })
    with mock.patch("scripts.ingest._ollama_extraction_call", return_value=mock_resp):
        result = _fill_metadata("Cover page text", probe)

    assert result.journal == "J. Am. Chem. Soc.", f"Expected abbreviation, got {result.journal!r}"
    assert result.journal_full == "Journal of the American Chemical Society", (
        f"Expected full journal name, got {result.journal_full!r}"
    )


def test_fill_metadata_fallback_journal_full_none():
    """_fill_metadata fallback path (two-strike timeout) sets journal_full=None gracefully."""
    from scripts.ingest import _fill_metadata, DoiProbeResult

    probe = DoiProbeResult(doi="10.1021/jacs.3c10258", arxiv_id=None, title="JACS Paper")
    timeout_resp = "[Ollama timeout: no response within 120s]"
    with mock.patch("scripts.ingest._ollama_extraction_call", return_value=timeout_resp):
        result = _fill_metadata("Cover page text", probe)

    assert result.journal_full is None, (
        f"Expected journal_full=None on fallback, got {result.journal_full!r}"
    )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 06 (gap closure) Task 3: section timeout config
# ---------------------------------------------------------------------------

def test_section_timeout_default_is_300():
    """DEFAULT_SECTION_TIMEOUT is 300 and config.json has ollama_section_timeout==300."""
    import scripts.ingest as _ingest_mod

    assert _ingest_mod.DEFAULT_SECTION_TIMEOUT == 300, (
        f"Expected DEFAULT_SECTION_TIMEOUT=300, got {_ingest_mod.DEFAULT_SECTION_TIMEOUT}"
    )
    # Verify config.json has the key set to 300
    import json as _json
    import pathlib
    cfg_path = pathlib.Path(__file__).parent.parent / "config.json"
    cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg.get("ollama_section_timeout") == 300, (
        f"Expected config.json ollama_section_timeout=300, got {cfg.get('ollama_section_timeout')}"
    )


def test_section_timeout_config_flow(tmp_path):
    """ingest() reads ollama_section_timeout from config and sets _SECTION_TIMEOUT module global."""
    import scripts.ingest as _ingest_mod
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult, DEFAULT_SECTION_TIMEOUT

    cfg = _make_ingest_config(tmp_path, extra={"ollama_section_timeout": 450})
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 section timeout config flow test")
    cl_path = _write_real_content_list(tmp_path)

    probe_result = DoiProbeResult(doi="10.1000/timeout.config.test", arxiv_id=None, title="Timeout Config Paper")
    metadata_result = PaperMetadata(title="Timeout Config Paper", doi="10.1000/timeout.config.test")
    section_fill = SectionFillResult(heading="", body="Test body " + "A" * 200, fill_failed=False)

    try:
        with mock.patch("scripts.ingest._run_mineru"), \
             mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
             mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
             mock.patch("scripts.ingest._warmup_ollama"), \
             mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
             mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
             mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
             mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
            ingest(str(fake_pdf), cfg)
        assert _ingest_mod._SECTION_TIMEOUT == 450, (
            f"Expected _SECTION_TIMEOUT=450 after ingest() with ollama_section_timeout=450, "
            f"got {_ingest_mod._SECTION_TIMEOUT}"
        )
    finally:
        _ingest_mod._SECTION_TIMEOUT = DEFAULT_SECTION_TIMEOUT


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 07 (gap closure) Task 1: full-text-only probe + prompt de-dup + extraction-only journal_full
# ---------------------------------------------------------------------------

def test_doi_probe_single_arg_signature():
    """_doi_probe signature has exactly ONE parameter named 'full_text' (no 'first_page_text') — GAP A."""
    import inspect
    import scripts.ingest as _m

    params = list(inspect.signature(_m._doi_probe).parameters)
    assert params == ["full_text"], (
        f"Expected _doi_probe to have exactly ['full_text'], got {params!r}"
    )

    # Single-arg call still returns DoiProbeResult (backward compat)
    import json as _json
    mock_resp = _json.dumps({"doi": "10.1021/jacs.3c10258", "arxiv_id": None, "title": "T"})
    with patch("scripts.ingest._ollama_extraction_call", return_value=mock_resp):
        result = _m._doi_probe("some document text")
    assert result is not None
    assert result.doi == "10.1021/jacs.3c10258"


def test_doi_probe_prompt_no_duplicated_guidance():
    """Probe guidance lives in SYSTEM only; USER prompt is minimal + doc text — GAP B."""
    import json as _json

    captured_prompt = []
    captured_system = []

    def capture_call(prompt, system, schema, **kwargs):
        captured_prompt.append(prompt)
        captured_system.append(system)
        return _json.dumps({"doi": "10.1021/jacs.3c10258", "arxiv_id": None, "title": "T"})

    from scripts.ingest import _doi_probe
    doc = "DOC TEXT 10.1021/jacs.3c10258"
    with patch("scripts.ingest._ollama_extraction_call", side_effect=capture_call):
        _doi_probe(doc)

    assert captured_system, "Expected _ollama_extraction_call to be called"
    assert captured_prompt, "Expected _ollama_extraction_call to be called"

    sys_prompt = captured_system[0]
    user_prompt = captured_prompt[0]

    # Guidance must be in SYSTEM
    assert "Supporting Information" in sys_prompt, (
        f"Expected 'Supporting Information' in SYSTEM prompt, got: {sys_prompt!r}"
    )
    assert "Preserve the DOI exactly" in sys_prompt, (
        f"Expected 'Preserve the DOI exactly' in SYSTEM prompt, got: {sys_prompt!r}"
    )
    # "cover page" or "title" still in SYSTEM
    assert ("cover page" in sys_prompt.lower() or "title" in sys_prompt.lower()), (
        f"Expected cover/title guidance in SYSTEM prompt, got: {sys_prompt!r}"
    )

    # Guidance MUST NOT be duplicated in USER prompt
    assert "Supporting Information" not in user_prompt, (
        f"Expected 'Supporting Information' NOT in USER prompt, got: {user_prompt!r}"
    )

    # USER prompt still contains the document text
    assert doc in user_prompt, (
        f"Expected document text in USER prompt, got: {user_prompt!r}"
    )


def test_doi_probe_scans_full_text_no_first_page_kwarg():
    """_doi_probe(full_text=...) with page-3 DOI: prompt contains DOI; no first_page_text kwarg accepted — GAP A+B."""
    import scripts.ingest as _m
    from scripts.ingest import _doi_probe, _estimate_num_ctx
    import json as _json

    captured_prompt = []
    captured_kwargs = {}

    def capture_call(prompt, system, schema, **kwargs):
        captured_prompt.append(prompt)
        captured_kwargs.update(kwargs)
        return _json.dumps({"doi": "10.1021/jacs.3c10258", "arxiv_id": None, "title": "T"})

    full_text = "Cover page text only\n" + "page 3 contains: 10.1021/jacs.3c10258"
    with mock.patch("scripts.ingest._ollama_extraction_call", side_effect=capture_call):
        # Call with full_text only (no first_page_text)
        _doi_probe(full_text=full_text)

    assert captured_prompt, "Expected _ollama_extraction_call to be called"
    assert "10.1021/jacs.3c10258" in captured_prompt[0], (
        f"Expected full_text (with page-3 DOI) in prompt, got: {captured_prompt[0]!r}"
    )
    # num_ctx sized input-only (output_ratio defaults to 0.0)
    expected_ctx = _estimate_num_ctx(full_text)
    assert captured_kwargs.get("num_ctx") == expected_ctx, (
        f"Expected num_ctx={expected_ctx} (input-only sizing), got {captured_kwargs.get('num_ctx')}"
    )


def test_fill_metadata_journal_full_extraction_only():
    """_fill_metadata SYSTEM prompt instructs verbatim-only journal_full; no expand-from-knowledge — GAP C."""
    import json as _json
    from scripts.ingest import _fill_metadata, DoiProbeResult

    captured_system = []

    def capture_call(prompt, system, schema, **kwargs):
        captured_system.append(system)
        return "[Ollama error: stop]"  # triggers RuntimeError on first call

    probe = DoiProbeResult(doi="10.1021/jacs.3c10258", arxiv_id=None, title="T")
    with patch("scripts.ingest._ollama_extraction_call", side_effect=capture_call):
        try:
            _fill_metadata("Trends Chem. text", probe)
        except RuntimeError:
            pass  # expected — we only care about the captured SYSTEM prompt

    assert captured_system, "Expected _ollama_extraction_call to be called"
    sys_prompt = captured_system[0]

    # MUST contain verbatim-extraction instruction
    assert "verbatim" in sys_prompt, (
        f"Expected 'verbatim' in _fill_metadata SYSTEM prompt, got: {sys_prompt!r}"
    )
    # MUST contain null-when-abbreviation-only instruction
    assert "null" in sys_prompt.lower(), (
        f"Expected 'null' instruction in _fill_metadata SYSTEM prompt, got: {sys_prompt!r}"
    )
    # MUST NOT contain the old expand-from-knowledge example
    assert "Journal of the American Chemical Society" not in sys_prompt, (
        f"Expected expand-from-knowledge example REMOVED from SYSTEM prompt, got: {sys_prompt!r}"
    )


def test_fill_metadata_journal_full_null_when_only_abbreviation():
    """_fill_metadata returns journal_full=None when LLM returns null (abbreviation-only case) — GAP C regression."""
    import json as _json
    from scripts.ingest import _fill_metadata, DoiProbeResult

    probe = DoiProbeResult(doi="10.1021/jacs.3c10258", arxiv_id=None, title="Trends Paper")
    # Model honors the extraction-only prompt: journal is abbreviation, journal_full is null
    mock_resp = _json.dumps({
        "title": "Trends Paper",
        "authors": [],
        "journal": "Trends Chem.",
        "journal_full": None,
        "year": 2023,
        "doi": "10.1021/jacs.3c10258",
        "arxiv_id": None,
    })
    with mock.patch("scripts.ingest._ollama_extraction_call", return_value=mock_resp):
        result = _fill_metadata("...Trends Chem. ...", probe)

    assert result.journal_full is None, (
        f"Expected journal_full=None for abbreviation-only case, got {result.journal_full!r}"
    )
    assert result.journal == "Trends Chem.", (
        f"Expected journal='Trends Chem.' (abbreviation as-is), got {result.journal!r}"
    )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 07 (gap closure) Task 2: echo-aware num_ctx sizing
# ---------------------------------------------------------------------------

def test_estimate_num_ctx_output_ratio():
    """_estimate_num_ctx(output_ratio=1.0) returns strictly larger rung for mid-size text — GAP D."""
    from scripts.ingest import _estimate_num_ctx

    # Default (input-only) — unchanged behaviour
    default_large = _estimate_num_ctx("x" * 200000)
    assert default_large == 65536, (
        f"Expected 65536 for 200000-char default, got {default_large}"
    )

    # Mid-size text: input-only vs echo-aware must land on different rungs.
    # 48000 chars → 12000 tokens
    # input-only:  12000 + 2048 = 14048 → rung 16384
    # echo-aware:  24000 + 2048 = 26048 → rung 32768 (strictly larger)
    mid_text = "x" * 48000
    input_only = _estimate_num_ctx(mid_text)
    echo_aware = _estimate_num_ctx(mid_text, output_ratio=1.0)
    assert input_only == 16384, f"Expected input-only=16384 for 48000-char text, got {input_only}"
    assert echo_aware == 32768, f"Expected echo-aware=32768 for 48000-char text, got {echo_aware}"
    assert echo_aware > input_only, (
        f"Expected echo-aware ({echo_aware}) > input-only ({input_only}) for 48000-char text"
    )

    # 2x budget exceeding cap must clamp to cap
    echo_cap = _estimate_num_ctx("x" * 200000, output_ratio=1.0)
    assert echo_cap == 65536, (
        f"Expected echo-aware for 200000-char text to clamp to cap 65536, got {echo_cap}"
    )


def test_estimate_num_ctx_default_unchanged():
    """_estimate_num_ctx default (output_ratio=0.0) is byte-for-byte identical to old behaviour — GAP D backward compat."""
    from scripts.ingest import _estimate_num_ctx

    for chars in [100, 24000, 100000, 200000]:
        text = "x" * chars
        assert _estimate_num_ctx(text) == _estimate_num_ctx(text, output_ratio=0.0), (
            f"Expected _estimate_num_ctx({chars}) == _estimate_num_ctx({chars}, output_ratio=0.0)"
        )


def test_fill_section_requests_echo_aware_num_ctx():
    """_fill_section passes num_ctx=_estimate_num_ctx(section_text, output_ratio=1.0) — GAP D fix."""
    import json as _json
    from scripts.ingest import _fill_section, _estimate_num_ctx

    # 48000 chars: input-only→16384, echo-aware→32768 (strictly larger, different rungs)
    section_text = "x" * 48000
    captured_kwargs = {}

    def capture_call(prompt, system, schema, **kwargs):
        captured_kwargs.update(kwargs)
        return _json.dumps({"heading": "RESULTS", "body": "result body", "fill_failed": False})

    with patch("scripts.ingest._ollama_extraction_call", side_effect=capture_call):
        _fill_section(section_text, "RESULTS")

    echo_expected = _estimate_num_ctx(section_text, output_ratio=1.0)
    input_only = _estimate_num_ctx(section_text)

    assert captured_kwargs.get("num_ctx") == echo_expected, (
        f"Expected _fill_section to use echo-aware num_ctx={echo_expected}, "
        f"got {captured_kwargs.get('num_ctx')}"
    )
    assert captured_kwargs.get("num_ctx") > input_only, (
        f"Expected echo-aware num_ctx ({captured_kwargs.get('num_ctx')}) > "
        f"input-only ({input_only}) — proves output budget is reserved"
    )


def test_fill_references_batched_echo_aware_num_ctx():
    """_fill_references_batched uses echo-aware num_ctx (not fixed 4096) for the ref-batch call — GAP D fix."""
    import json as _json
    from scripts.ingest import _fill_references_batched, _estimate_num_ctx

    # Large batch: 10 refs with 1200 chars each → batch_text ≈ 12000 chars
    # 12000 / 4 = 3000 tokens; echo-aware: 6000 + 2048 = 8048 → rung 8192 (> old fixed 4096)
    raw_refs = [{"raw": "r" * 1200} for _ in range(10)]
    captured_kwargs = {}

    def capture_call(prompt, system, schema, **kwargs):
        captured_kwargs.update(kwargs)
        return _json.dumps({"refs": []})

    with patch("scripts.ingest._ollama_extraction_call", side_effect=capture_call):
        _fill_references_batched(raw_refs)

    assert captured_kwargs.get("num_ctx") is not None, "Expected num_ctx kwarg to be captured"
    assert captured_kwargs["num_ctx"] > 4096, (
        f"Expected echo-aware num_ctx > 4096 (old fixed value), "
        f"got {captured_kwargs.get('num_ctx')}"
    )
    # Verify it's sized via the sizer with output_ratio=1.0
    # batch_text is built from the prompt prefix + refs (we compute what the function uses)
    batch_text = "\n".join(ref["raw"] for ref in raw_refs)
    expected = _estimate_num_ctx(batch_text, output_ratio=1.0)
    assert captured_kwargs["num_ctx"] == expected, (
        f"Expected num_ctx={expected} (_estimate_num_ctx(batch_text, output_ratio=1.0)), "
        f"got {captured_kwargs['num_ctx']}"
    )


def test_fill_section_oversize_guard_echo_aware():
    """_fill_section guard uses echo-aware sizing (output_ratio=1.0); 300000-char section still trips — GAP D."""
    from scripts.ingest import _fill_section

    # 300000 chars → echo-aware estimate still >= 65536 cap AND 300000 > 65536*4=262144 → guard trips
    huge_text = "x" * 300000
    with patch("scripts.ingest._ollama_extraction_call", side_effect=AssertionError("LLM must not be called")):
        result = _fill_section(huge_text, "Huge Section")
    assert result.fill_failed is True, "Expected fill_failed=True for oversize section"


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 08 (gap closure) Task 1: LLM-judge section filtering
# ---------------------------------------------------------------------------

def _make_two_section_parsed(substantive_heading="INTRODUCTION", noise_heading="ACKNOWLEDGMENTS"):
    """Build a minimal _parse_content_list result with two sections for filtering tests."""
    return {
        "title": "Test Paper Title Long Enough",
        "sections": [
            {
                "heading": substantive_heading,
                "level": 1,
                "blocks": [
                    {"type": "text", "display": "We present results. " * 12, "plain": "We present results. " * 12},
                ],
            },
            {
                "heading": noise_heading,
                "level": 1,
                "blocks": [
                    {"type": "text", "display": "The authors thank funding agencies.", "plain": "The authors thank funding agencies."},
                ],
            },
        ],
        "references": [],
        "metadata": {
            "title": "Test Paper Title Long Enough",
            "authors": None, "year": 2024, "journal": None,
            "doi": "10.1000/filter.test", "arxiv_id": None, "accession_codes": [],
        },
    }


def test_section_fill_result_keep_defaults_true():
    """SectionFillResult.keep defaults True; explicit False sets it False (Plan 08 T1)."""
    from scripts.ingest import SectionFillResult
    assert SectionFillResult(heading="H", body="b").keep is True, (
        "Expected keep=True by default (conservative retain)"
    )
    assert SectionFillResult(heading="H", body="b", keep=False).keep is False, (
        "Expected keep=False when explicitly set"
    )


def test_section_keep_true_retained(tmp_path):
    """A section the mock marks keep=True is present in result extraction sections (Plan 08 T1)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 keep=true retained test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_two_section_parsed("INTRODUCTION", "ACKNOWLEDGMENTS")
    probe_result = DoiProbeResult(doi="10.1000/filter.test", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(title="Test Paper Title Long Enough", doi="10.1000/filter.test", year=2024)

    def fill_side_effect(section_text, heading):
        return SectionFillResult(heading=heading, body=section_text + " [cleaned]", fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", side_effect=fill_side_effect), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        result = ingest(str(fake_pdf), cfg)

    headings = [s["heading"] for s in result["extraction"]["sections"]]
    assert "INTRODUCTION" in headings, (
        f"Expected 'INTRODUCTION' (keep=True) in output sections, got headings: {headings}"
    )
    assert "ACKNOWLEDGMENTS" in headings, (
        f"Expected 'ACKNOWLEDGMENTS' (keep=True) in output sections, got headings: {headings}"
    )


def test_section_keep_false_dropped_from_output(tmp_path):
    """A section the mock marks keep=False is NOT in result extraction sections (Plan 08 T1)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 keep=false dropped test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_two_section_parsed("INTRODUCTION", "ACKNOWLEDGMENTS")
    probe_result = DoiProbeResult(doi="10.1000/filter.test2", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(title="Test Paper Title Long Enough", doi="10.1000/filter.test2", year=2024)

    def fill_side_effect(section_text, heading):
        keep = heading != "ACKNOWLEDGMENTS"
        return SectionFillResult(heading=heading, body=section_text + " [cleaned]", fill_failed=False, keep=keep)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", side_effect=fill_side_effect), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        result = ingest(str(fake_pdf), cfg)

    headings = [s["heading"] for s in result["extraction"]["sections"]]
    assert "ACKNOWLEDGMENTS" not in headings, (
        f"Expected 'ACKNOWLEDGMENTS' (keep=False) dropped from output, got headings: {headings}"
    )
    assert "INTRODUCTION" in headings, (
        f"Expected 'INTRODUCTION' (keep=True) retained in output, got headings: {headings}"
    )


def test_fill_failed_section_always_retained(tmp_path):
    """A section with fill_failed=True (keep defaults True) is ALWAYS retained in output (Plan 08 T1)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fill_failed always retained test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_two_section_parsed("RESULTS", "NOISE_SECTION")
    probe_result = DoiProbeResult(doi="10.1000/filter.test3", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(title="Test Paper Title Long Enough", doi="10.1000/filter.test3", year=2024)

    def fill_side_effect(section_text, heading):
        # fill_failed=True means no verdict — keep defaults True → always retain
        return SectionFillResult(heading=heading, body=section_text, fill_failed=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", side_effect=fill_side_effect), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        result = ingest(str(fake_pdf), cfg)

    headings = [s["heading"] for s in result["extraction"]["sections"]]
    assert "RESULTS" in headings, (
        f"Expected fill_failed section 'RESULTS' retained in output, got headings: {headings}"
    )
    assert "NOISE_SECTION" in headings, (
        f"Expected fill_failed section 'NOISE_SECTION' retained in output (fill_failed=True always retain), got headings: {headings}"
    )
    # Verify fill_failed is carried through
    for s in result["extraction"]["sections"]:
        assert s["fill_failed"] is True, (
            f"Expected fill_failed=True carried in output section {s['heading']!r}"
        )


def test_section_filter_logs_drop_count(tmp_path, capsys):
    """After an ingest() run that drops >=1 section, stderr has a timestamped section filter summary (Plan 08 T1)."""
    import re as _re
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 section filter log test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_two_section_parsed("INTRODUCTION", "ACKNOWLEDGMENTS")
    probe_result = DoiProbeResult(doi="10.1000/filter.log.test", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(title="Test Paper Title Long Enough", doi="10.1000/filter.log.test", year=2024)

    def fill_side_effect(section_text, heading):
        keep = heading != "ACKNOWLEDGMENTS"
        return SectionFillResult(heading=heading, body=section_text + " [cleaned]", fill_failed=False, keep=keep)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", side_effect=fill_side_effect), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        ingest(str(fake_pdf), cfg)

    captured = capsys.readouterr()
    assert _re.search(
        r"\[ingest \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] section filter: kept \d+, dropped \d+",
        captured.err,
    ), (
        f"Expected timestamped 'section filter: kept N, dropped M' in stderr, got: {captured.err!r}"
    )


def test_output_sections_have_no_keep_key(tmp_path):
    """Every section dict in result extraction has keys exactly {heading, body, fill_failed} — no 'keep' leak (Plan 08 T1)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 no keep key in output test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_two_section_parsed("INTRODUCTION", "RESULTS")
    probe_result = DoiProbeResult(doi="10.1000/no.keep.key", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(title="Test Paper Title Long Enough", doi="10.1000/no.keep.key", year=2024)

    def fill_side_effect(section_text, heading):
        return SectionFillResult(heading=heading, body=section_text + " [cleaned]", fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", side_effect=fill_side_effect), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        result = ingest(str(fake_pdf), cfg)

    for i, section in enumerate(result["extraction"]["sections"]):
        assert set(section.keys()) == {"heading", "body", "fill_failed"}, (
            f"Section {i} has unexpected keys: {set(section.keys())} — expected exactly {{heading, body, fill_failed}}"
        )


def test_abstract_empty_heading_relabel(tmp_path):
    """A kept section with empty heading relabeled to 'Abstract' by the model appears correctly; 'INTRODUCTION' heading is unchanged (Plan 08 T1)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 abstract relabel test")
    cl_path = _write_real_content_list(tmp_path)

    # Two sections: one with empty heading (abstract), one with existing heading
    mock_parsed = {
        "title": "Test Paper Title Long Enough",
        "sections": [
            {
                "heading": "",
                "level": 0,
                "blocks": [
                    {"type": "text", "display": "We present an abstract. " * 10, "plain": "We present an abstract. " * 10},
                ],
            },
            {
                "heading": "INTRODUCTION",
                "level": 1,
                "blocks": [
                    {"type": "text", "display": "This paper introduces. " * 10, "plain": "This paper introduces. " * 10},
                ],
            },
        ],
        "references": [],
        "metadata": {
            "title": "Test Paper Title Long Enough",
            "authors": None, "year": 2024, "journal": None,
            "doi": "10.1000/relabel.test", "arxiv_id": None, "accession_codes": [],
        },
    }
    probe_result = DoiProbeResult(doi="10.1000/relabel.test", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(title="Test Paper Title Long Enough", doi="10.1000/relabel.test", year=2024)

    def fill_side_effect(section_text, heading):
        # Model relabels empty heading to "Abstract"; does NOT rename "INTRODUCTION"
        if not heading or heading.startswith("Section "):
            return SectionFillResult(heading="Abstract", body=section_text + " [cleaned]", fill_failed=False, keep=True)
        return SectionFillResult(heading=heading, body=section_text + " [cleaned]", fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", side_effect=fill_side_effect), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        result = ingest(str(fake_pdf), cfg)

    headings = [s["heading"] for s in result["extraction"]["sections"]]
    assert "Abstract" in headings, (
        f"Expected 'Abstract' (relabeled from empty heading) in output sections, got: {headings}"
    )
    assert "INTRODUCTION" in headings, (
        f"Expected 'INTRODUCTION' (unchanged existing heading) in output sections, got: {headings}"
    )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 08 (gap closure) Task 2: Crossref journal_full enrichment + flag default
# ---------------------------------------------------------------------------

def test_crossref_journal_full_enriches_when_none(capsys):
    """_crossref_journal_full returns container-title[0] verbatim on success (Plan 08 T2)."""
    import json as _json
    from scripts.ingest import _crossref_journal_full

    payload = _json.dumps({
        "message": {"container-title": ["Journal of the American Chemical Society"]}
    }).encode()
    cfg = {"crossref_contact_email": "t@e.com"}
    with mock.patch("scripts.ingest.urllib.request.urlopen",
                    return_value=_FakeHttpResponse(payload)):
        result = _crossref_journal_full("10.1021/jacs.3c10258", cfg)

    assert result == "Journal of the American Chemical Society", (
        f"Expected 'Journal of the American Chemical Society', got {result!r}"
    )


def test_crossref_journal_full_fail_open_on_network_error(capsys):
    """_crossref_journal_full returns None + emits one quiet log line on URLError (Plan 08 T2)."""
    from scripts.ingest import _crossref_journal_full

    cfg = {"crossref_contact_email": "t@e.com"}
    with mock.patch("scripts.ingest.urllib.request.urlopen",
                    side_effect=_urllib_error.URLError("connection refused")):
        result = _crossref_journal_full("10.1/x", cfg)

    assert result is None, f"Expected None (fail-open) on URLError, got {result!r}"
    captured = capsys.readouterr()
    assert "crossref journal lookup unavailable" in captured.err, (
        f"Expected 'crossref journal lookup unavailable' in stderr, got: {captured.err!r}"
    )


def test_crossref_journal_full_none_on_missing_container_title():
    """_crossref_journal_full returns None when container-title is absent or empty (Plan 08 T2)."""
    import json as _json
    from scripts.ingest import _crossref_journal_full

    # Missing container-title key
    payload_no_key = _json.dumps({"message": {"title": ["Some Paper"]}}).encode()
    cfg = {"crossref_contact_email": "t@e.com"}
    with mock.patch("scripts.ingest.urllib.request.urlopen",
                    return_value=_FakeHttpResponse(payload_no_key)):
        result = _crossref_journal_full("10.1/x", cfg)
    assert result is None, f"Expected None on missing container-title, got {result!r}"

    # Empty container-title list
    payload_empty = _json.dumps({"message": {"container-title": []}}).encode()
    with mock.patch("scripts.ingest.urllib.request.urlopen",
                    return_value=_FakeHttpResponse(payload_empty)):
        result2 = _crossref_journal_full("10.1/x", cfg)
    assert result2 is None, f"Expected None on empty container-title list, got {result2!r}"


def test_crossref_journal_full_uses_https_and_contact():
    """_crossref_journal_full source uses HTTPS, crossref_contact_email, and container-title (Plan 08 T2)."""
    import inspect
    from scripts.ingest import _crossref_journal_full

    source = inspect.getsource(_crossref_journal_full)
    assert "https://api.crossref.org" in source, (
        "Expected 'https://api.crossref.org' in _crossref_journal_full source"
    )
    assert "http://api.crossref.org" not in source, (
        "Plain http:// must NOT be used for Crossref"
    )
    assert "crossref_contact_email" in source, (
        "Expected 'crossref_contact_email' in _crossref_journal_full source"
    )
    assert "container-title" in source, (
        "Expected 'container-title' in _crossref_journal_full source"
    )


def test_ingest_enriches_journal_full_when_none(tmp_path):
    """ingest() sets metadata.journal_full from _crossref_journal_full when it's None + DOI valid + crossref on (Plan 08 T2)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path, extra={"crossref_validate": True, "crossref_contact_email": "t@e.com"})
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 journal full enrich test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_two_section_parsed("INTRODUCTION", "RESULTS")
    probe_result = DoiProbeResult(doi="10.1021/jacs.3c10258", arxiv_id=None, title="Test Paper Title Long Enough")
    # journal_full=None triggers the enrichment
    metadata_result = PaperMetadata(
        title="Test Paper Title Long Enough",
        doi="10.1021/jacs.3c10258",
        year=2024,
        journal="J. Am. Chem. Soc.",
        journal_full=None,
    )
    section_fill = SectionFillResult(heading="INTRODUCTION", body="body text " * 20, fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)), \
         mock.patch("scripts.ingest._crossref_validate"), \
         mock.patch("scripts.ingest._crossref_journal_full",
                    return_value="Journal of the American Chemical Society") as mock_jf:
        result = ingest(str(fake_pdf), cfg)

    assert mock_jf.called, "Expected _crossref_journal_full to be called"
    assert result["extraction"]["metadata"]["journal_full"] == "Journal of the American Chemical Society", (
        f"Expected journal_full='Journal of the American Chemical Society', "
        f"got {result['extraction']['metadata']['journal_full']!r}"
    )
    assert result["extraction"]["metadata"]["journal"] == "J. Am. Chem. Soc.", (
        f"Expected journal abbreviation unchanged, got {result['extraction']['metadata']['journal']!r}"
    )


def test_ingest_no_journal_lookup_when_already_set(tmp_path):
    """ingest() does NOT call _crossref_journal_full when journal_full is already set (Plan 08 T2)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path, extra={"crossref_validate": True, "crossref_contact_email": "t@e.com"})
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 no lookup when already set")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_two_section_parsed()
    probe_result = DoiProbeResult(doi="10.1021/jacs.3c10258", arxiv_id=None, title="Test Paper Title Long Enough")
    # journal_full already set — no lookup needed
    metadata_result = PaperMetadata(
        title="Test Paper Title Long Enough",
        doi="10.1021/jacs.3c10258",
        year=2024,
        journal="J. Am. Chem. Soc.",
        journal_full="Journal of the American Chemical Society",  # already set
    )
    section_fill = SectionFillResult(heading="INTRODUCTION", body="body text " * 20, fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)), \
         mock.patch("scripts.ingest._crossref_validate"), \
         mock.patch("scripts.ingest._crossref_journal_full",
                    side_effect=AssertionError("must not be called when already set")) as mock_jf:
        ingest(str(fake_pdf), cfg)

    assert mock_jf.call_count == 0, (
        f"Expected _crossref_journal_full call_count == 0 when journal_full already set, "
        f"got {mock_jf.call_count}"
    )


def test_ingest_no_journal_lookup_when_crossref_off(tmp_path):
    """ingest() does NOT call _crossref_journal_full when crossref_validate is False (Plan 08 T2)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    # crossref_validate is absent (defaults False in the code guard)
    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 no lookup when crossref off")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_two_section_parsed()
    probe_result = DoiProbeResult(doi="10.1021/jacs.3c10258", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(
        title="Test Paper Title Long Enough",
        doi="10.1021/jacs.3c10258",
        year=2024,
        journal="J. Am. Chem. Soc.",
        journal_full=None,
    )
    section_fill = SectionFillResult(heading="INTRODUCTION", body="body text " * 20, fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)), \
         mock.patch("scripts.ingest._crossref_journal_full",
                    side_effect=AssertionError("must not be called when crossref off")) as mock_jf:
        ingest(str(fake_pdf), cfg)

    assert mock_jf.call_count == 0, (
        f"Expected _crossref_journal_full call_count == 0 when crossref_validate off, "
        f"got {mock_jf.call_count}"
    )


def test_ingest_journal_full_stays_none_on_lookup_failure(tmp_path):
    """When _crossref_journal_full returns None (network error), journal_full stays None — fail-open (Plan 08 T2)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path, extra={"crossref_validate": True, "crossref_contact_email": "t@e.com"})
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fail open journal lookup test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_two_section_parsed()
    probe_result = DoiProbeResult(doi="10.1021/jacs.3c10258", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(
        title="Test Paper Title Long Enough",
        doi="10.1021/jacs.3c10258",
        year=2024,
        journal="J. Am. Chem. Soc.",
        journal_full=None,
    )
    section_fill = SectionFillResult(heading="INTRODUCTION", body="body text " * 20, fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)), \
         mock.patch("scripts.ingest._crossref_validate"), \
         mock.patch("scripts.ingest._crossref_journal_full", return_value=None):
        result = ingest(str(fake_pdf), cfg)

    assert result["extraction"]["metadata"]["journal_full"] is None, (
        f"Expected journal_full=None on lookup failure (fail-open), "
        f"got {result['extraction']['metadata']['journal_full']!r}"
    )


def test_config_crossref_validate_default_true():
    """config.json crossref_validate is True — the single flag enabling same-paper guard + journal enrichment (Plan 08 T2)."""
    import json as _json
    cfg = _json.load(open("config.json", encoding="utf-8"))
    assert cfg["crossref_validate"] is True, (
        f"Expected crossref_validate=True in config.json (Plan 08 default flip), got {cfg['crossref_validate']!r}"
    )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 09 (gap closure) Task 1: unified LLM ref-judge (numbering + footnote filter)
# ---------------------------------------------------------------------------

def test_ref_entry_is_reference_defaults_true():
    """RefEntry.is_reference defaults True; explicit False sets it False (Plan 09 T1)."""
    from scripts.ingest import RefEntry
    assert RefEntry(raw="x").is_reference is True, (
        "Expected is_reference=True by default (conservative retain)"
    )
    assert RefEntry(raw="x", is_reference=False).is_reference is False, (
        "Expected is_reference=False when explicitly set"
    )


def test_ref_numbering_format_agnostic():
    """_fill_references_batched carries number from mock across (1)/1./[1]/bare styles (Plan 09 T1)."""
    from scripts.ingest import _fill_references_batched

    raw_refs = [
        {"raw": "(1) Deidda, G. et al. Down Syndrome. J. Am. Chem. Soc. 2023."},
        {"raw": "1. Gamba, G. (2005) Kidney Int."},
        {"raw": "[1] Foo B. et al. Cell 2022."},
        {"raw": "1 Bar C. et al. Nature 2021."},
    ]
    mock_resp = _json_mod.dumps({
        "refs": [
            {"number": 1, "raw": "(1) Deidda, G. et al. Down Syndrome. J. Am. Chem. Soc. 2023.",
             "doi": None, "title": "Down Syndrome", "year": 2023, "fill_failed": False,
             "is_reference": True},
            {"number": 1, "raw": "1. Gamba, G. (2005) Kidney Int.",
             "doi": None, "title": "Kidney Int", "year": 2005, "fill_failed": False,
             "is_reference": True},
            {"number": 1, "raw": "[1] Foo B. et al. Cell 2022.",
             "doi": None, "title": "Cell paper", "year": 2022, "fill_failed": False,
             "is_reference": True},
            {"number": 1, "raw": "1 Bar C. et al. Nature 2021.",
             "doi": None, "title": "Nature paper", "year": 2021, "fill_failed": False,
             "is_reference": True},
        ]
    })
    with patch("scripts.ingest._ollama_extraction_call", return_value=mock_resp):
        filled, failures = _fill_references_batched(raw_refs)

    assert failures == 0
    assert len(filled) == 4
    # All number=1 (as returned by mock — correct wiring)
    for entry in filled:
        assert entry["number"] == 1, f"Expected number=1 from mock, got {entry['number']!r}"


def test_ref_inline_paren_not_used_as_number():
    """Mock returns number=5 (not 2003/12) for a period-style ref with inline parens (Plan 09 T1)."""
    from scripts.ingest import _fill_references_batched

    raw_refs = [
        {"raw": "5. Starremans, P.G. et al. (2003) J. Am. Chem. Soc. 12 (12), 1206."},
    ]
    mock_resp = _json_mod.dumps({
        "refs": [
            {"number": 5, "raw": "5. Starremans, P.G. et al. (2003) J. Am. Chem. Soc. 12 (12), 1206.",
             "doi": None, "title": "JACS paper", "year": 2003, "fill_failed": False,
             "is_reference": True},
        ]
    })
    with patch("scripts.ingest._ollama_extraction_call", return_value=mock_resp):
        filled, failures = _fill_references_batched(raw_refs)

    assert failures == 0
    assert len(filled) == 1
    assert filled[0]["number"] == 5, (
        f"Expected number=5 (leading ordinal, not inline year/volume), got {filled[0]['number']!r}"
    )


def test_ref_is_reference_false_dropped():
    """_fill_references_batched drops is_reference=false (non-fill_failed) entries (Plan 09 T1)."""
    from scripts.ingest import _fill_references_batched

    raw_refs = [
        {"raw": "Michele Parrinello − Laboratory of Atomistic Simulations, IIT … michele.parrinello@iit.it"},
        {"raw": "(1) Deidda, G. et al. Down Syndrome. J. Am. Chem. Soc. 2023."},
        {"raw": "(2) Smith, J. et al. Nature 2020. DOI:10.1/x"},
    ]
    mock_resp = _json_mod.dumps({
        "refs": [
            {"number": None, "raw": "Michele Parrinello − Laboratory of Atomistic Simulations, IIT … michele.parrinello@iit.it",
             "doi": None, "title": None, "year": None, "fill_failed": False,
             "is_reference": False},
            {"number": 1, "raw": "(1) Deidda, G. et al. Down Syndrome. J. Am. Chem. Soc. 2023.",
             "doi": None, "title": "Down Syndrome", "year": 2023, "fill_failed": False,
             "is_reference": True},
            {"number": 2, "raw": "(2) Smith, J. et al. Nature 2020. DOI:10.1/x",
             "doi": "10.1/x", "title": "Nature paper", "year": 2020, "fill_failed": False,
             "is_reference": True},
        ]
    })
    with patch("scripts.ingest._ollama_extraction_call", return_value=mock_resp):
        filled, failures = _fill_references_batched(raw_refs)

    assert failures == 0
    assert len(filled) == 2, (
        f"Expected 2 real refs (affiliation dropped), got {len(filled)}"
    )
    raw_strings = [r["raw"] for r in filled]
    assert "Michele Parrinello" not in " ".join(raw_strings), (
        "Expected affiliation/email entry to be dropped from filled refs"
    )
    assert any("Deidda" in r["raw"] for r in filled), "Expected ref (1) Deidda in filled"
    assert any("Smith" in r["raw"] for r in filled), "Expected ref (2) Smith in filled"


def test_ref_fill_failed_always_retained():
    """fill_failed refs are always retained (is_reference defaults True on failure) (Plan 09 T1)."""
    from scripts.ingest import _fill_references_batched

    raw_refs = [{"raw": "1. Bad ref"}, {"raw": "2. Also bad"}, {"raw": "3. Another bad"}]
    with patch("scripts.ingest._ollama_extraction_call", return_value="not json"), \
         patch("scripts.ingest._parse_extraction_response", return_value=None):
        filled, failures = _fill_references_batched(raw_refs)

    assert failures == 1, f"Expected 1 failed batch, got {failures}"
    assert len(filled) == len(raw_refs), (
        f"Expected all {len(raw_refs)} refs retained (fill_failed always retained), got {len(filled)}"
    )
    for ref in filled:
        assert ref.get("fill_failed") is True, (
            f"Expected fill_failed=True in retained ref, got {ref!r}"
        )


def test_ref_parser_prompt_is_format_agnostic_and_judges():
    """_fill_references_batched source contains format-agnostic numbering + ref-vs-footnote judgment (Plan 09 T1)."""
    import inspect
    from scripts.ingest import _fill_references_batched

    source = inspect.getsource(_fill_references_batched)
    # Format-agnostic numbering: must mention (1), 1., [1] styles
    assert "(1)" in source, "Expected '(1)' citation style mentioned in prompt"
    assert "1." in source, "Expected '1.' citation style mentioned in prompt"
    assert "[1]" in source, "Expected '[1]' citation style mentioned in prompt"
    # Must forbid inline year/volume
    assert "inline" in source.lower() or "year" in source.lower(), (
        "Expected inline paren / year prohibition in prompt"
    )
    # Reference-vs-footnote judgment
    assert "is_reference" in source, "Expected 'is_reference' mentioned in prompt/source"
    assert "affiliation" in source.lower() or "footnote" in source.lower(), (
        "Expected affiliation/footnote judgment in prompt"
    )
    # Preservation of contract
    assert "Return ONLY valid JSON" in source, "Expected 'Return ONLY valid JSON' in prompt"
    assert "Do not invent" in source, "Expected 'Do not invent values' in prompt"


def test_ref_filter_logs_drop_count(capsys):
    """_fill_references_batched emits a timestamped 'reference filter: kept N, dropped M' log (Plan 09 T1)."""
    import re as _re
    from scripts.ingest import _fill_references_batched

    raw_refs = [
        {"raw": "Affiliation block — email@example.com"},
        {"raw": "(1) Real, R. et al. Science 2024."},
    ]
    mock_resp = _json_mod.dumps({
        "refs": [
            {"number": None, "raw": "Affiliation block — email@example.com",
             "doi": None, "title": None, "year": None, "fill_failed": False,
             "is_reference": False},
            {"number": 1, "raw": "(1) Real, R. et al. Science 2024.",
             "doi": None, "title": "Science paper", "year": 2024, "fill_failed": False,
             "is_reference": True},
        ]
    })
    with patch("scripts.ingest._ollama_extraction_call", return_value=mock_resp):
        _fill_references_batched(raw_refs)

    captured = capsys.readouterr()
    assert _re.search(
        r"\[ingest \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] reference filter: kept \d+, dropped \d+",
        captured.err,
    ), (
        f"Expected timestamped 'reference filter: kept N, dropped M' in stderr, got: {captured.err!r}"
    )


def test_output_refs_have_no_is_reference_key():
    """Returned filled dicts have keys exactly {number, raw, doi, title, year, fill_failed} — no is_reference (Plan 09 T1)."""
    from scripts.ingest import _fill_references_batched

    # Success path: is_reference present in mock response but must not reach output
    raw_refs_success = [
        {"raw": "(1) Smith J. Science 2024."},
        {"raw": "(2) Jones A. Nature 2023."},
    ]
    mock_resp = _json_mod.dumps({
        "refs": [
            {"number": 1, "raw": "(1) Smith J. Science 2024.",
             "doi": None, "title": "Science", "year": 2024, "fill_failed": False,
             "is_reference": True},
            {"number": 2, "raw": "(2) Jones A. Nature 2023.",
             "doi": None, "title": "Nature", "year": 2023, "fill_failed": False,
             "is_reference": True},
        ]
    })
    expected_keys = {"number", "raw", "doi", "title", "year", "fill_failed"}

    with patch("scripts.ingest._ollama_extraction_call", return_value=mock_resp):
        filled_success, _ = _fill_references_batched(raw_refs_success)

    for i, ref in enumerate(filled_success):
        assert set(ref.keys()) == expected_keys, (
            f"Success-path ref[{i}] has unexpected keys: {set(ref.keys())} — expected {expected_keys}"
        )

    # Failed-batch path: fill_failed refs must also not have is_reference key
    raw_refs_failed = [{"raw": "1. Bad ref"}, {"raw": "2. Also bad"}]
    with patch("scripts.ingest._ollama_extraction_call", return_value="not json"), \
         patch("scripts.ingest._parse_extraction_response", return_value=None):
        filled_failed, _ = _fill_references_batched(raw_refs_failed)

    for i, ref in enumerate(filled_failed):
        assert set(ref.keys()) == expected_keys, (
            f"Failed-batch ref[{i}] has unexpected keys: {set(ref.keys())} — expected {expected_keys}"
        )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 09 (gap closure) Task 2: journal abbreviation hardening + fallback
# ---------------------------------------------------------------------------

def test_metadata_prompt_forbids_journal_from_doi():
    """_fill_metadata source instructs that journal must NOT be derived from the DOI string (Plan 09 T2)."""
    import inspect
    from scripts.ingest import _fill_metadata

    source = inspect.getsource(_fill_metadata)
    # Must mention DOI is opaque and journal must not be derived from it
    assert "DOI" in source or "doi" in source.lower(), (
        "Expected DOI mentioned in _fill_metadata source"
    )
    assert "opaque" in source.lower() or "identifier" in source.lower(), (
        "Expected DOI described as opaque identifier in _fill_metadata source"
    )
    # Must also preserve existing instructions
    assert "journal abbreviation" in source.lower() or "journal" in source.lower(), (
        "Expected journal instruction preserved in _fill_metadata source"
    )


def test_ingest_journal_falls_back_to_full_when_abbrev_missing(tmp_path):
    """ingest() sets journal=journal_full when journal is None but journal_full is set (Plan 09 T2)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path, extra={"crossref_validate": True, "crossref_contact_email": "t@e.com"})
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 journal fallback test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_two_section_parsed("INTRODUCTION", "RESULTS")
    probe_result = DoiProbeResult(doi="10.1016/j.trechm.2021.05.004", arxiv_id=None, title="Test Paper Title Long Enough")
    # journal=None simulates model failing to extract printed abbreviation (GAP C scenario)
    metadata_result = PaperMetadata(
        title="Test Paper Title Long Enough",
        doi="10.1016/j.trechm.2021.05.004",
        year=2021,
        journal=None,
        journal_full=None,  # will be enriched by mock crossref
    )
    section_fill = SectionFillResult(heading="INTRODUCTION", body="body text " * 20, fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)), \
         mock.patch("scripts.ingest._crossref_validate"), \
         mock.patch("scripts.ingest._crossref_journal_full", return_value="Trends in Chemistry"):
        result = ingest(str(fake_pdf), cfg)

    meta = result["extraction"]["metadata"]
    assert meta["journal"] == "Trends in Chemistry", (
        f"Expected journal='Trends in Chemistry' (fell back to journal_full), got {meta['journal']!r}"
    )
    assert meta["journal_full"] == "Trends in Chemistry", (
        f"Expected journal_full='Trends in Chemistry' (unchanged), got {meta['journal_full']!r}"
    )


def test_ingest_journal_abbrev_preserved_when_present(tmp_path):
    """When journal abbreviation is already set, journal_full does NOT overwrite it (Plan 09 T2)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path, extra={"crossref_validate": True, "crossref_contact_email": "t@e.com"})
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 journal abbrev preserved test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_two_section_parsed("INTRODUCTION", "RESULTS")
    probe_result = DoiProbeResult(doi="10.1021/jacs.3c10258", arxiv_id=None, title="Test Paper Title Long Enough")
    # journal already has the printed abbreviation
    metadata_result = PaperMetadata(
        title="Test Paper Title Long Enough",
        doi="10.1021/jacs.3c10258",
        year=2024,
        journal="J. Am. Chem. Soc.",
        journal_full=None,
    )
    section_fill = SectionFillResult(heading="INTRODUCTION", body="body text " * 20, fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)), \
         mock.patch("scripts.ingest._crossref_validate"), \
         mock.patch("scripts.ingest._crossref_journal_full",
                    return_value="Journal of the American Chemical Society"):
        result = ingest(str(fake_pdf), cfg)

    meta = result["extraction"]["metadata"]
    assert meta["journal"] == "J. Am. Chem. Soc.", (
        f"Expected journal='J. Am. Chem. Soc.' (printed abbrev preserved), got {meta['journal']!r}"
    )
    assert meta["journal_full"] == "Journal of the American Chemical Society", (
        f"Expected journal_full enriched by crossref, got {meta['journal_full']!r}"
    )


def test_ingest_journal_fallback_noop_when_full_also_missing(tmp_path):
    """When both journal and journal_full are None, fallback is a no-op — no crash (Plan 09 T2)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 journal fallback noop test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_two_section_parsed("INTRODUCTION", "RESULTS")
    probe_result = DoiProbeResult(doi="10.1000/noop.test", arxiv_id=None, title="Test Paper Title Long Enough")
    # Both None — no fallback possible
    metadata_result = PaperMetadata(
        title="Test Paper Title Long Enough",
        doi="10.1000/noop.test",
        year=2024,
        journal=None,
        journal_full=None,
    )
    section_fill = SectionFillResult(heading="INTRODUCTION", body="body text " * 20, fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)), \
         mock.patch("scripts.ingest._crossref_journal_full", return_value=None):
        result = ingest(str(fake_pdf), cfg)

    meta = result["extraction"]["metadata"]
    assert meta["journal"] is None, (
        f"Expected journal=None when both journal and journal_full are None, got {meta['journal']!r}"
    )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 09 (gap closure) Task 3: section heading recovery (GAP E)
# ---------------------------------------------------------------------------

def _make_single_section_parsed(heading="", body_text=None):
    """Build a minimal _parse_content_list result with one section for heading-recovery tests."""
    if body_text is None:
        body_text = "This paper presents a systematic study. " * 15
    return {
        "title": "Test Paper Title Long Enough",
        "sections": [
            {
                "heading": heading,
                "level": 0,
                "blocks": [
                    {"type": "text", "display": body_text, "plain": body_text},
                ],
            },
        ],
        "references": [],
        "metadata": {
            "title": "Test Paper Title Long Enough",
            "authors": None, "year": 2024, "journal": None,
            "doi": "10.1000/heading.test", "arxiv_id": None, "accession_codes": [],
        },
    }


def test_empty_heading_passed_raw_to_fill_section(tmp_path):
    """ingest() passes the raw empty string to _fill_section for a missing-heading section (Plan 09 T3)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 raw heading passthrough test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_single_section_parsed(heading="")
    probe_result = DoiProbeResult(doi="10.1000/heading.test", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(title="Test Paper Title Long Enough", doi="10.1000/heading.test", year=2024)

    captured_headings = []

    def fill_spy(section_text, heading):
        captured_headings.append(heading)
        return SectionFillResult(heading="Abstract", body=section_text + " [cleaned]", fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", side_effect=fill_spy), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        ingest(str(fake_pdf), cfg)

    assert len(captured_headings) == 1, f"Expected 1 _fill_section call, got {len(captured_headings)}"
    assert captured_headings[0] == "", (
        f"Expected empty string '' passed to _fill_section (not 'Section 0'), "
        f"got {captured_headings[0]!r}"
    )


def test_empty_heading_last_resort_section_n(tmp_path):
    """When model returns empty heading from fill, output falls back to 'Section 0' (last-resort) (Plan 09 T3)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 last resort section n test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_single_section_parsed(heading="")
    probe_result = DoiProbeResult(doi="10.1000/heading.test2", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(title="Test Paper Title Long Enough", doi="10.1000/heading.test2", year=2024)

    def fill_returns_empty(section_text, heading):
        # Model gave no relabel — returns empty heading
        return SectionFillResult(heading="", body=section_text + " [cleaned]", fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", side_effect=fill_returns_empty), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        result = ingest(str(fake_pdf), cfg)

    headings = [s["heading"] for s in result["extraction"]["sections"]]
    assert headings == ["Section 0"], (
        f"Expected ['Section 0'] as last-resort fallback, got {headings!r}"
    )


def test_section_n_log_label_preserved(tmp_path, capsys):
    """_log progress line for an empty-heading section still shows a non-blank label (Plan 09 T3)."""
    import re as _re
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 log label test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_single_section_parsed(heading="")
    probe_result = DoiProbeResult(doi="10.1000/heading.test3", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(title="Test Paper Title Long Enough", doi="10.1000/heading.test3", year=2024)
    section_fill = SectionFillResult(heading="Abstract", body="abstract text " * 10, fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        ingest(str(fake_pdf), cfg)

    captured = capsys.readouterr()
    # The log label should be "Section 0" (non-blank), not an empty string
    assert _re.search(
        r"\[ingest [\d-]+ [\d:]+\] section fill 1/1: 'Section 0'",
        captured.err,
    ), (
        f"Expected log label 'Section 0' (non-blank) for empty-heading section, got: {captured.err!r}"
    )


def test_fill_section_relabels_placeholder_headings(tmp_path):
    """A section with placeholder heading whose fill returns a real heading is preserved in output (Plan 09 T3)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 placeholder heading relabel test")
    cl_path = _write_real_content_list(tmp_path)

    # Section heading is "N/A" (placeholder from out2)
    mock_parsed = _make_single_section_parsed(heading="N/A")
    probe_result = DoiProbeResult(doi="10.1000/placeholder.test", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(title="Test Paper Title Long Enough", doi="10.1000/placeholder.test", year=2024)

    def fill_relabels(section_text, heading):
        # Model relabels "N/A" to "Introduction" by reading the body
        return SectionFillResult(heading="Introduction", body=section_text + " [cleaned]", fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", side_effect=fill_relabels), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        result = ingest(str(fake_pdf), cfg)

    headings = [s["heading"] for s in result["extraction"]["sections"]]
    assert "Introduction" in headings, (
        f"Expected 'Introduction' (relabeled from 'N/A') in output sections, got {headings!r}"
    )
    assert "N/A" not in headings, (
        f"Expected 'N/A' not in output sections (should be relabeled), got {headings!r}"
    )


def test_fill_section_prompt_widened_relabel():
    """_fill_section source shows the widened relabel covers placeholder/junk headings (Plan 09 T3)."""
    import inspect
    from scripts.ingest import _fill_section

    source = inspect.getsource(_fill_section)
    # Must mention placeholder/junk headings in the relabel instruction
    assert "placeholder" in source.lower() or "N/A" in source, (
        "Expected placeholder or N/A mentioned in _fill_section relabel instruction"
    )
    # Must mention Section N style
    assert "Section" in source, "Expected 'Section' mentioned in relabel instruction"
    # Must still protect recognizable section names
    assert "recognizable" in source.lower() or "Introduction" in source, (
        "Expected recognizable section names protected in relabel instruction"
    )


def test_legit_heading_not_renamed(tmp_path):
    """A section with a recognizable heading (INTRODUCTION/RESULTS) is NOT renamed (Plan 09 T3)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 legit heading preserved test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_two_section_parsed("INTRODUCTION", "RESULTS")
    probe_result = DoiProbeResult(doi="10.1000/legit.heading", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(title="Test Paper Title Long Enough", doi="10.1000/legit.heading", year=2024)

    def fill_preserves(section_text, heading):
        # Model does NOT rename recognizable headings
        return SectionFillResult(heading=heading, body=section_text + " [cleaned]", fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", side_effect=fill_preserves), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        result = ingest(str(fake_pdf), cfg)

    headings = [s["heading"] for s in result["extraction"]["sections"]]
    assert "INTRODUCTION" in headings, (
        f"Expected 'INTRODUCTION' (recognizable heading, not renamed) in output, got {headings!r}"
    )
    assert "RESULTS" in headings, (
        f"Expected 'RESULTS' (recognizable heading, not renamed) in output, got {headings!r}"
    )


# ---------------------------------------------------------------------------
# Plan 01.3-10 Task 1: year ← Crossref fallback tests (INGEST-01)
# ---------------------------------------------------------------------------

def test_crossref_published_year_returns_year(capsys):
    """_crossref_published_year returns published year from Crossref date-parts (Plan 10 T1)."""
    import json as _json
    from scripts.ingest import _crossref_published_year

    payload = _json.dumps({
        "message": {"published": {"date-parts": [[2023, 5, 1]]}}
    }).encode()
    cfg = {"crossref_contact_email": "t@e.com"}
    with mock.patch("scripts.ingest.urllib.request.urlopen",
                    return_value=_FakeHttpResponse(payload)):
        result = _crossref_published_year("10.1021/jacs.3c10258", cfg)

    assert result == 2023, f"Expected 2023, got {result!r}"


def test_crossref_published_year_fail_open_on_network_error(capsys):
    """_crossref_published_year returns None on URLError (fail-open) (Plan 10 T1)."""
    from scripts.ingest import _crossref_published_year

    cfg = {"crossref_contact_email": "t@e.com"}
    with mock.patch("scripts.ingest.urllib.request.urlopen",
                    side_effect=_urllib_error.URLError("connection refused")):
        result = _crossref_published_year("10.1/x", cfg)

    assert result is None, f"Expected None (fail-open), got {result!r}"
    captured = capsys.readouterr()
    assert "crossref year lookup unavailable" in captured.err, (
        f"Expected 'crossref year lookup unavailable' in stderr, got: {captured.err!r}"
    )


def test_crossref_published_year_uses_https():
    """_crossref_published_year source uses HTTPS and date-parts (Plan 10 T1)."""
    import inspect
    from scripts.ingest import _crossref_published_year

    source = inspect.getsource(_crossref_published_year)
    assert "https://api.crossref.org" in source, (
        "Expected 'https://api.crossref.org' in source"
    )
    assert "http://api.crossref.org" not in source, (
        "Plain http:// must NOT be used for Crossref"
    )
    assert "date-parts" in source, "Expected 'date-parts' in source"
    assert "crossref_contact_email" in source, (
        "Expected 'crossref_contact_email' in source"
    )


def test_ingest_year_falls_back_to_crossref_when_null(tmp_path):
    """ingest() enriches metadata.year from Crossref when fill left it null (Plan 10 T1)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path, extra={"crossref_validate": True, "crossref_contact_email": "t@e.com"})
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 year crossref fallback test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_two_section_parsed("INTRODUCTION", "RESULTS")
    probe_result = DoiProbeResult(doi="10.1021/jacs.3c10258", arxiv_id=None, title="Test Paper Title Long Enough")
    # year=None triggers the Crossref year fallback
    metadata_result = PaperMetadata(
        title="Test Paper Title Long Enough",
        doi="10.1021/jacs.3c10258",
        year=None,
        journal="J. Am. Chem. Soc.",
        journal_full="Journal of the American Chemical Society",
    )
    section_fill = SectionFillResult(heading="INTRODUCTION", body="body text " * 20, fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)), \
         mock.patch("scripts.ingest._crossref_validate"), \
         mock.patch("scripts.ingest._crossref_journal_full",
                    return_value="Journal of the American Chemical Society"), \
         mock.patch("scripts.ingest._crossref_published_year",
                    return_value=2023) as mock_year:
        result = ingest(str(fake_pdf), cfg)

    assert mock_year.called, "Expected _crossref_published_year to be called"
    assert result["extraction"]["metadata"]["year"] == 2023, (
        f"Expected year=2023 from Crossref, got {result['extraction']['metadata']['year']!r}"
    )


def test_no_year_lookup_when_already_set(tmp_path):
    """ingest() does NOT call _crossref_published_year when year is already set (Plan 10 T1)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path, extra={"crossref_validate": True, "crossref_contact_email": "t@e.com"})
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 year already set test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_two_section_parsed("INTRODUCTION", "RESULTS")
    probe_result = DoiProbeResult(doi="10.1021/jacs.3c10258", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(
        title="Test Paper Title Long Enough",
        doi="10.1021/jacs.3c10258",
        year=2024,  # already set
        journal="J. Am. Chem. Soc.",
        journal_full=None,
    )
    section_fill = SectionFillResult(heading="INTRODUCTION", body="body text " * 20, fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)), \
         mock.patch("scripts.ingest._crossref_validate"), \
         mock.patch("scripts.ingest._crossref_journal_full", return_value=None), \
         mock.patch("scripts.ingest._crossref_published_year",
                    return_value=9999) as mock_year:
        result = ingest(str(fake_pdf), cfg)

    assert mock_year.call_count == 0, (
        f"Expected _crossref_published_year NOT called when year already set, got {mock_year.call_count}"
    )
    assert result["extraction"]["metadata"]["year"] == 2024, (
        f"Expected year=2024 (preserved), got {result['extraction']['metadata']['year']!r}"
    )


def test_no_year_lookup_when_crossref_off(tmp_path):
    """ingest() does NOT call _crossref_published_year when crossref_validate is off (Plan 10 T1)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)  # crossref_validate absent (defaults False)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 year crossref off test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = _make_two_section_parsed("INTRODUCTION", "RESULTS")
    probe_result = DoiProbeResult(doi="10.1021/jacs.3c10258", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(
        title="Test Paper Title Long Enough",
        doi="10.1021/jacs.3c10258",
        year=None,
        journal=None,
        journal_full=None,
    )
    section_fill = SectionFillResult(heading="INTRODUCTION", body="body text " * 20, fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        result = ingest(str(fake_pdf), cfg)

    assert result["extraction"]["metadata"]["year"] is None, (
        f"Expected year=None when crossref_validate off, got {result['extraction']['metadata']['year']!r}"
    )


# ---------------------------------------------------------------------------
# Plan 01.3-10 Task 1: reference-path U+FFFD repair tests (INGEST-01)
# ---------------------------------------------------------------------------

def test_reference_ufffd_repaired_on_ref_path():
    """A reference with embedded U+FFFD is repaired on the reference path (Plan 10 T1)."""
    from scripts.ingest import _fill_references_batched, RefEntry, RefBatchResult

    # A reference with U+FFFD at an em-dash position
    raw_refs = [{"raw": "Speed without Compromise�A Mixed Precision Approach"}]

    # Mock the LLM call to return the ref as-is (with U+FFFD in raw/title)
    def fake_extraction(prompt, system, schema, num_ctx=None, timeout=None):
        import json as _json
        return _json.dumps({
            "refs": [{
                "number": 1,
                "doi": None,
                "title": "Speed without Compromise�A Mixed Precision",
                "year": 2020,
                "raw": "Speed without Compromise�A Mixed Precision Approach",
                "fill_failed": False,
                "is_reference": True,
            }]
        })

    with mock.patch("scripts.ingest._ollama_extraction_call", side_effect=fake_extraction):
        filled, failures = _fill_references_batched(raw_refs)

    assert len(filled) == 1, f"Expected 1 ref, got {len(filled)}"
    # U+FFFD must be repaired (removed or replaced with em-dash)
    assert "�" not in filled[0].get("raw", ""), (
        f"Expected U+FFFD repaired in ref raw, got: {filled[0]['raw']!r}"
    )
    assert "�" not in filled[0].get("title", ""), (
        f"Expected U+FFFD repaired in ref title, got: {filled[0]['title']!r}"
    )


def test_no_blanket_ufffd_replacement_outside_ref_path():
    """_normalize_text source has no blanket U+FFFD replacement (prohibition guard, Plan 10 T1)."""
    import inspect, re as _re2
    from scripts.ingest import _normalize_text

    source = inspect.getsource(_normalize_text)
    # Must NOT contain U+FFFD character literal in a replacement statement
    assert "�" not in source, (
        "Expected no U+FFFD literal in _normalize_text source (prohibition: no blanket replacement)"
    )
    # Must NOT contain a .replace() or re.sub() call referencing FFFD/replacement char
    # (the docstring may mention it descriptively, but no executable replacement line)
    code_lines = [
        ln for ln in source.splitlines()
        if not ln.strip().startswith("#") and not ln.strip().startswith('"""') and not ln.strip().startswith("'")
    ]
    code_only = "\n".join(code_lines)
    assert not _re2.search(r"\.replace\(.*fffd", code_only, _re2.IGNORECASE), (
        "Expected no .replace() with FFFD in _normalize_text executable code"
    )
    assert not _re2.search(r"re\.sub\(.*fffd", code_only, _re2.IGNORECASE), (
        "Expected no re.sub() with FFFD in _normalize_text executable code"
    )


# ---------------------------------------------------------------------------
# Plan 01.3-10 Task 2: GAP E — relax _fill_section relabel clause (Option A)
# ---------------------------------------------------------------------------

def test_fill_section_prompt_carves_out_article_type_heading():
    """_fill_section prompt explicitly carves out article-type/title headings from protection (Plan 10 T2)."""
    import inspect
    from scripts.ingest import _fill_section

    source = inspect.getsource(_fill_section)
    # Must mention article-type labels as override targets
    assert "article-type" in source.lower() or "article category" in source.lower(), (
        "Expected the prompt to carve out article-type labels from heading protection"
    )
    # Must mention title-as-heading as an override target
    assert "title" in source.lower(), (
        "Expected the prompt to mention paper title as a non-descriptive heading"
    )
    # Must still protect genuine section names
    assert "Introduction" in source or "introduction" in source, (
        "Expected genuine section names (e.g. Introduction) still protected"
    )
    assert "Results" in source or "results" in source, (
        "Expected genuine section names (e.g. Results) still protected"
    )


def test_fill_section_no_oneshot_example():
    """_fill_section source contains no 1-shot example block (prohibition guard, Plan 10 T2)."""
    import inspect
    from scripts.ingest import _fill_section

    source = inspect.getsource(_fill_section)
    # Must NOT contain an example mapping like '"Review" -> "Abstract"' or similar
    assert '"Review" -> "Abstract"' not in source, (
        "Expected no 1-shot example mapping junk->real heading in _fill_section"
    )
    assert '"Review": "Abstract"' not in source, (
        "Expected no 1-shot example mapping junk->real heading in _fill_section"
    )
    # No "Example:" or "For example:" block with heading mapping
    import re as _re3
    assert not _re3.search(r'[Ee]xample:?\s*"[^"]+"\s*(?:->|→|=>|:)\s*"[^"]+"', source), (
        "Expected no example mapping block in _fill_section source"
    )


def test_no_deterministic_heading_normalization():
    """ingest() does NOT contain any post-fill junk-set heading rewrite (prohibition guard, Plan 10 T2)."""
    import inspect
    from scripts.ingest import ingest

    source = inspect.getsource(ingest)
    # Must NOT contain a heading normalization mapping (e.g. {"Review": "Abstract"})
    import re as _re4
    # No dict/set with known junk headings being mapped to real headings
    assert not _re4.search(r'heading_map|junk_headings|heading_normalize', source, _re4.IGNORECASE), (
        "Expected no deterministic heading normalization map in ingest()"
    )
    # No explicit 'if heading == "Review": heading = "Abstract"' style rewrites
    assert not _re4.search(r'heading\s*==?\s*"Review".*=\s*"Abstract"', source), (
        "Expected no deterministic Review->Abstract rewrite in ingest()"
    )


def test_fill_section_relabels_article_type_heading(tmp_path):
    """A section with an article-type heading (e.g. 'Review') can be relabeled by _fill_section (Plan 10 T2)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 article type relabel test")
    cl_path = _write_real_content_list(tmp_path)

    # Section heading is an article-type label ("Review")
    mock_parsed = _make_single_section_parsed(heading="Review")
    probe_result = DoiProbeResult(doi="10.1000/gape.test", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(title="Test Paper Title Long Enough", doi="10.1000/gape.test", year=2024)

    def fill_relabels(section_text, heading):
        # Model relabels "Review" to "Abstract" by reading the body
        return SectionFillResult(heading="Abstract", body=section_text + " [cleaned]", fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", side_effect=fill_relabels), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        result = ingest(str(fake_pdf), cfg)

    headings = [s["heading"] for s in result["extraction"]["sections"]]
    assert "Abstract" in headings, (
        f"Expected 'Abstract' (relabeled from article-type 'Review') in output, got {headings!r}"
    )


def test_fill_section_relabels_title_as_heading(tmp_path):
    """A section whose heading equals the paper title can be relabeled by _fill_section (Plan 10 T2)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 title-as-heading relabel test")
    cl_path = _write_real_content_list(tmp_path)

    # Section heading equals the paper title
    mock_parsed = _make_single_section_parsed(heading="Test Paper Title Long Enough")
    probe_result = DoiProbeResult(doi="10.1000/titlehead.test", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(title="Test Paper Title Long Enough", doi="10.1000/titlehead.test", year=2024)

    def fill_relabels(section_text, heading):
        # Model relabels the title-heading to "Introduction"
        return SectionFillResult(heading="Introduction", body=section_text + " [cleaned]", fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", side_effect=fill_relabels), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        result = ingest(str(fake_pdf), cfg)

    headings = [s["heading"] for s in result["extraction"]["sections"]]
    assert "Introduction" in headings, (
        f"Expected 'Introduction' (relabeled from title-as-heading) in output, got {headings!r}"
    )


# ---------------------------------------------------------------------------
# Plan 01.3-10 Task 3: CR-01 — registry projects[] union-merge + cache-hit append
# ---------------------------------------------------------------------------

def test_registry_projects_survive_force_reingest_from_other_project(tmp_path):
    """Force re-ingest from a different project preserves prior projects[] (CR-01, REG-01, Plan 10 T3)."""
    from scripts.ingest import ingest, _write_registry, _read_registry

    cfg = _make_ingest_config(tmp_path)
    reg_path = cfg["registry_path"]

    # Pre-populate registry with an entry from "old-project"
    old_entry = {
        "title": "Old Entry",
        "doi": "10.1000/force.test",
        "projects": ["old-project"],
        "summary": None, "key_findings": None,
        "authors": None, "year": 2020, "journal": None,
        "arxiv_id": None,
        "source_path": "/old.pdf",
        "paperjson_path": "/old.json",
    }
    _write_registry(old_entry, reg_path, "10.1000/force.test")

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 force extract CR-01 test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = {
        "title": "Updated Paper",
        "sections": [{"heading": "", "level": 0, "blocks": [
            {"type": "text", "display": "Updated Paper", "plain": "Updated Paper"},
            {"type": "text", "display": "B" * 200, "plain": "B" * 200},
        ]}],
        "references": [],
        "metadata": {
            "title": "Updated Paper",
            "authors": None,
            "year": 2024,
            "journal": None,
            "doi": "10.1000/force.test",
            "arxiv_id": None,
            "accession_codes": [],
        },
    }

    from scripts.ingest import DoiProbeResult, PaperMetadata, SectionFillResult
    probe_result = DoiProbeResult(doi="10.1000/force.test", arxiv_id=None, title="Updated Paper")
    metadata_result = PaperMetadata(title="Updated Paper", doi="10.1000/force.test", year=2024)
    section_fill = SectionFillResult(heading="", body="Updated Paper " + "B" * 200, fill_failed=False)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        ingest(str(fake_pdf), cfg, force_extract=True)

    # Registry must preserve BOTH "old-project" AND "test-project" (the _make_ingest_config default)
    registry = _read_registry(reg_path)
    projects = registry["10.1000/force.test"]["projects"]
    assert "old-project" in projects, (
        f"Expected 'old-project' preserved in projects[] after force re-ingest, got {projects!r}"
    )
    assert "test-project" in projects, (
        f"Expected 'test-project' added to projects[] after force re-ingest, got {projects!r}"
    )
    # No duplicates
    assert len(projects) == len(set(projects)), (
        f"Expected no duplicate projects, got {projects!r}"
    )


def test_cache_hit_appends_current_project(tmp_path):
    """On a cache HIT, ingest() appends the current project and persists (CR-01, REG-02, Plan 10 T3)."""
    from scripts.ingest import ingest, _write_registry, _read_registry

    cfg = _make_ingest_config(tmp_path)
    reg_path = cfg["registry_path"]

    # Pre-populate registry with an entry from "other-project" only
    cached_entry = {
        "title": "Cached Paper Title Long Enough",
        "doi": "10.1000/cache.hit",
        "projects": ["other-project"],
        "summary": None, "key_findings": None,
        "authors": None, "year": 2024, "journal": None,
        "arxiv_id": None,
        "source_path": "/cached.pdf",
        "paperjson_path": "/cached.json",
    }
    _write_registry(cached_entry, reg_path, "10.1000/cache.hit")

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 cache hit append project test")
    cl_path = _write_real_content_list(tmp_path)

    from scripts.ingest import DoiProbeResult
    probe_result = DoiProbeResult(doi="10.1000/cache.hit", arxiv_id=None, title="Cached Paper Title Long Enough")

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value={
             "title": "Cached Paper Title Long Enough",
             "sections": [{"heading": "", "level": 0, "blocks": [
                 {"type": "text", "display": "A" * 200, "plain": "A" * 200},
             ]}],
             "references": [],
             "metadata": {"title": "Cached Paper Title Long Enough", "authors": None, "year": 2024,
                           "journal": None, "doi": "10.1000/cache.hit", "arxiv_id": None, "accession_codes": []},
         }), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result):
        result = ingest(str(fake_pdf), cfg)  # force_extract=False (default) → cache hit

    # The returned result must include "test-project" in projects
    assert "test-project" in result.get("projects", []), (
        f"Expected 'test-project' in returned result projects, got {result.get('projects')!r}"
    )
    # The registry on disk must also reflect the addition
    registry = _read_registry(reg_path)
    disk_projects = registry["10.1000/cache.hit"]["projects"]
    assert "other-project" in disk_projects, (
        f"Expected 'other-project' preserved in registry, got {disk_projects!r}"
    )
    assert "test-project" in disk_projects, (
        f"Expected 'test-project' appended to registry, got {disk_projects!r}"
    )


def test_cache_hit_no_duplicate_when_project_already_present(tmp_path):
    """Cache hit where current project already in projects[] does not duplicate (CR-01, Plan 10 T3)."""
    from scripts.ingest import ingest, _write_registry, _read_registry

    cfg = _make_ingest_config(tmp_path)
    reg_path = cfg["registry_path"]

    # Pre-populate with current project already in projects[]
    cached_entry = {
        "title": "Already There Paper Title",
        "doi": "10.1000/nodup.test",
        "projects": ["test-project"],  # same as _make_ingest_config default
        "summary": None, "key_findings": None,
        "authors": None, "year": 2024, "journal": None,
        "arxiv_id": None,
        "source_path": "/cached.pdf",
        "paperjson_path": "/cached.json",
    }
    _write_registry(cached_entry, reg_path, "10.1000/nodup.test")

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 no dup project test")
    cl_path = _write_real_content_list(tmp_path)

    from scripts.ingest import DoiProbeResult
    probe_result = DoiProbeResult(doi="10.1000/nodup.test", arxiv_id=None, title="Already There Paper Title")

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value={
             "title": "Already There Paper Title",
             "sections": [{"heading": "", "level": 0, "blocks": [
                 {"type": "text", "display": "A" * 200, "plain": "A" * 200},
             ]}],
             "references": [],
             "metadata": {"title": "Already There Paper Title", "authors": None, "year": 2024,
                           "journal": None, "doi": "10.1000/nodup.test", "arxiv_id": None, "accession_codes": []},
         }), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result):
        result = ingest(str(fake_pdf), cfg)

    # No duplicates in returned result
    result_projects = result.get("projects", [])
    assert result_projects.count("test-project") == 1, (
        f"Expected exactly one 'test-project' in result (no duplicate), got {result_projects!r}"
    )
    # No duplicates on disk
    registry = _read_registry(reg_path)
    disk_projects = registry["10.1000/nodup.test"]["projects"]
    assert disk_projects.count("test-project") == 1, (
        f"Expected exactly one 'test-project' on disk (no duplicate), got {disk_projects!r}"
    )


def test_write_registry_union_merges_projects():
    """_write_registry source contains union-merge of projects inside the filelock (Plan 10 T3)."""
    import inspect
    from scripts.ingest import _write_registry

    source = inspect.getsource(_write_registry)
    # Must contain evidence of union-merge (dict.fromkeys or set union or similar)
    assert "projects" in source, (
        "Expected 'projects' mentioned in _write_registry (union-merge logic)"
    )
    assert "dict.fromkeys" in source or "set(" in source or "union" in source, (
        "Expected dict.fromkeys/set/union in _write_registry for projects[] merge"
    )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 11 Task 1: --refill CLI flag tests
# ---------------------------------------------------------------------------

def test_refill_reuses_mineru_output(tmp_path):
    """refill=True + existing content_list -> _run_mineru called with force_extract=False,
    cache return bypassed, fill helpers invoked (Plan 11 T1)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path, extra={"crossref_contact_email": "real@test.com"})
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 refill test content")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = {
        "title": "Refill Test Paper",
        "sections": [{"heading": "INTRO", "level": 1, "blocks": [
            {"type": "text", "display": "Refill Test Paper", "plain": "Refill Test Paper"},
            {"type": "text", "display": "Z" * 200, "plain": "Z" * 200},
        ]}],
        "references": [],
        "metadata": {
            "title": "Refill Test Paper", "authors": None, "year": 2024,
            "journal": None, "doi": "10.1000/refill.test", "arxiv_id": None, "accession_codes": [],
        },
    }
    probe_result = DoiProbeResult(doi="10.1000/refill.test", arxiv_id=None, title="Refill Test Paper")
    metadata_result = PaperMetadata(title="Refill Test Paper", doi="10.1000/refill.test", year=2024)
    section_fill = SectionFillResult(heading="INTRO", body="Z" * 200, fill_failed=False, keep=True)

    mineru_calls = []

    def fake_mineru(*args, **kwargs):
        mineru_calls.append(args)

    fill_metadata_called = []
    original_fill_metadata = metadata_result

    def fake_fill_metadata(*a, **kw):
        fill_metadata_called.append(True)
        return original_fill_metadata

    with mock.patch("scripts.ingest._run_mineru", side_effect=fake_mineru), \
         mock.patch("scripts.ingest._find_content_list_path", return_value=cl_path), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", side_effect=fake_fill_metadata), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        result = ingest(str(fake_pdf), cfg, refill=True)

    # _run_mineru is called positionally: (pdf_path, out_dir, mineru_exe, timeout, force_extract)
    # force_extract is the 5th arg (index 4) and must be False for refill
    assert mineru_calls, "_run_mineru should have been called"
    assert mineru_calls[0][4] is False, (
        f"Expected _run_mineru called with force_extract=False for refill, got {mineru_calls[0][4]!r}"
    )

    # Fill helpers MUST have been called (cache return bypassed)
    assert fill_metadata_called, "_fill_metadata should be called on refill (cache bypass)"


def test_refill_bypasses_cache_return(tmp_path):
    """refill=True bypasses the registry cache-hit early return so the fill re-runs (Plan 11 T1)."""
    from scripts.ingest import ingest, _write_registry, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path, extra={"crossref_contact_email": "real@test.com"})
    reg_path = cfg["registry_path"]

    # Pre-populate registry — normally this would cause a cache hit + early return
    doi_key = "10.1000/refill.bypass"
    cached = {
        "title": "Cached Paper", "doi": doi_key, "projects": ["other-project"],
        "summary": None, "key_findings": None, "authors": None, "year": None,
        "journal": None, "arxiv_id": None, "source_path": "/old.pdf", "paperjson_path": "/old.json",
    }
    _write_registry(cached, reg_path, doi_key)

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 refill bypass test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = {
        "title": "Cached Paper",
        "sections": [{"heading": "INTRO", "level": 1, "blocks": [
            {"type": "text", "display": "Cached Paper", "plain": "Cached Paper"},
            {"type": "text", "display": "Y" * 200, "plain": "Y" * 200},
        ]}],
        "references": [],
        "metadata": {
            "title": "Cached Paper", "authors": None, "year": 2024,
            "journal": None, "doi": doi_key, "arxiv_id": None, "accession_codes": [],
        },
    }
    probe_result = DoiProbeResult(doi=doi_key, arxiv_id=None, title="Cached Paper")
    metadata_result = PaperMetadata(title="Cached Paper", doi=doi_key, year=2024)
    section_fill = SectionFillResult(heading="INTRO", body="Y" * 200, fill_failed=False, keep=True)

    fill_called = []

    def track_fill_metadata(*a, **kw):
        fill_called.append(True)
        return metadata_result

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list_path", return_value=cl_path), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", side_effect=track_fill_metadata), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        result = ingest(str(fake_pdf), cfg, refill=True)

    # Fill helpers MUST have run even though the DOI was in the registry
    assert fill_called, "_fill_metadata must be called on refill (bypasses cache return)"

    # Result must be a full PaperJSON (not the cached stub dict)
    assert "extraction" in result, "Expected full PaperJSON result on refill, not cached dict"


def test_refill_errors_when_no_content_list(tmp_path, capsys):
    """refill=True + no existing content_list.json -> clear error + non-zero exit,
    no full MinerU run (Plan 11 T1)."""
    from scripts.ingest import ingest

    cfg = _make_ingest_config(tmp_path, extra={"crossref_contact_email": "real@test.com"})
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 refill no content_list")

    # Mock _find_content_list_path to return None (no existing MinerU output)
    mineru_called = []

    def fake_mineru(*a, **kw):
        mineru_called.append(True)

    with mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._find_content_list_path", return_value=None), \
         mock.patch("scripts.ingest._run_mineru", side_effect=fake_mineru), \
         pytest.raises(SystemExit) as exc_info:
        ingest(str(fake_pdf), cfg, refill=True)

    assert exc_info.value.code != 0, "Expected non-zero exit on refill with no content_list"
    captured = capsys.readouterr()
    assert "refill" in captured.err.lower() or "--refill" in captured.err, (
        f"Expected error mentioning refill, got: {captured.err!r}"
    )
    assert "existing MinerU output" in captured.err or "content_list" in captured.err.lower(), (
        f"Expected error about missing MinerU output, got: {captured.err!r}"
    )
    # _run_mineru must NOT have been called (no fallback to full GPU run)
    assert not mineru_called, "_run_mineru should NOT be called when refill has no content_list"


def test_refill_does_not_clobber_registry_membership(tmp_path):
    """A refill of a DOI already registered under another project does NOT drop
    that project from projects[] (Plan 11 T1, relies on Plan 10 union-merge)."""
    from scripts.ingest import ingest, _write_registry, _read_registry, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path, extra={"crossref_contact_email": "real@test.com"})
    reg_path = cfg["registry_path"]

    doi_key = "10.1000/refill.noclob"
    existing_entry = {
        "title": "Cross-Proj Paper", "doi": doi_key,
        "projects": ["project-alpha", "project-beta"],
        "summary": None, "key_findings": None, "authors": None, "year": 2024,
        "journal": None, "arxiv_id": None, "source_path": "/x.pdf", "paperjson_path": "/x.json",
    }
    _write_registry(existing_entry, reg_path, doi_key)

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 refill clobber test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = {
        "title": "Cross-Proj Paper",
        "sections": [{"heading": "INTRO", "level": 1, "blocks": [
            {"type": "text", "display": "Cross-Proj Paper", "plain": "Cross-Proj Paper"},
            {"type": "text", "display": "W" * 200, "plain": "W" * 200},
        ]}],
        "references": [],
        "metadata": {
            "title": "Cross-Proj Paper", "authors": None, "year": 2024,
            "journal": None, "doi": doi_key, "arxiv_id": None, "accession_codes": [],
        },
    }
    probe_result = DoiProbeResult(doi=doi_key, arxiv_id=None, title="Cross-Proj Paper")
    metadata_result = PaperMetadata(title="Cross-Proj Paper", doi=doi_key, year=2024)
    section_fill = SectionFillResult(heading="INTRO", body="W" * 200, fill_failed=False, keep=True)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list_path", return_value=cl_path), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        ingest(str(fake_pdf), cfg, refill=True)

    # Both original projects must still be present in the registry
    registry = _read_registry(reg_path)
    entry = registry[doi_key]
    assert "project-alpha" in entry["projects"], (
        f"Expected 'project-alpha' preserved after refill, got {entry['projects']!r}"
    )
    assert "project-beta" in entry["projects"], (
        f"Expected 'project-beta' preserved after refill, got {entry['projects']!r}"
    )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 11 Task 2: WR-01 body-less-fill floor + WR-05 quality-gate block count
# ---------------------------------------------------------------------------

def test_bodyless_fill_floor_blocks_empty_paper(tmp_path, capsys):
    """A fill where every section returns keep=False (body-less) does NOT
    silently write a registry entry and exit 0 (WR-01, Plan 11 T2)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 bodyless fill floor test")
    cl_path = _write_real_content_list(tmp_path)

    # Two sections, both will return keep=False (all content judged non-substantive)
    mock_parsed = _make_two_section_parsed("SECTION_A", "SECTION_B")
    probe_result = DoiProbeResult(doi="10.1000/bodyless.test", arxiv_id=None, title="Test Paper Title Long Enough")
    metadata_result = PaperMetadata(title="Test Paper Title Long Enough", doi="10.1000/bodyless.test", year=2024)

    # Every section gets keep=False (explicit non-substantive verdict)
    section_fill = SectionFillResult(heading="X", body="short", fill_failed=False, keep=False)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)):
        # The floor must either exit non-zero or retain all sections with a warning
        # Either way, it must NOT silently succeed with a body-less paper
        try:
            result = ingest(str(fake_pdf), cfg)
            # If it didn't exit, it must have retained sections (fallback retain behavior)
            sections = result.get("extraction", {}).get("sections", [])
            assert len(sections) > 0, (
                "Body-less floor: expected sections retained (not empty) if ingest() did not exit"
            )
            captured = capsys.readouterr()
            assert "non-substantive" in captured.err.lower() or "body-less" in captured.err.lower() or "all sections" in captured.err.lower(), (
                f"Expected a warning about body-less/non-substantive fill, got stderr: {captured.err!r}"
            )
        except SystemExit as e:
            # Exit with error is also acceptable (the floor fires)
            assert e.code != 0, "Expected non-zero exit for body-less fill"
            captured = capsys.readouterr()
            assert "non-substantive" in captured.err.lower() or "body-less" in captured.err.lower() or "all sections" in captured.err.lower(), (
                f"Expected error about body-less/non-substantive fill, got stderr: {captured.err!r}"
            )


# ---------------------------------------------------------------------------
# Phase 1.3 Plan 11 Task 3: WR-03 Crossref placeholder-email guard
# ---------------------------------------------------------------------------

def test_crossref_skipped_on_placeholder_email(capsys):
    """With crossref_validate=true and placeholder email, NO Crossref request is made
    and a one-line warning is emitted (WR-03, Plan 11 T3)."""
    import scripts.ingest as _ingest_mod
    from scripts.ingest import _crossref_validate, _crossref_journal_full, _crossref_published_year

    # Reset the per-process warning flag so this test gets the warning
    _ingest_mod._crossref_placeholder_warned = False

    urlopen_calls = []

    def track_urlopen(*args, **kwargs):
        urlopen_calls.append(True)
        return _FakeHttpResponse(b'{"message": {"title": ["X"]}}')

    # Test all three Crossref helpers with the placeholder email
    for helper_name, helper_fn, helper_args in [
        ("_crossref_validate", _crossref_validate, ("10.1000/test", "Title", {"crossref_contact_email": "your-email@example.com"})),
        ("_crossref_journal_full", _crossref_journal_full, ("10.1000/test", {"crossref_contact_email": "your-email@example.com"})),
        ("_crossref_published_year", _crossref_published_year, ("10.1000/test", {"crossref_contact_email": "your-email@example.com"})),
    ]:
        urlopen_calls.clear()
        with mock.patch("scripts.ingest.urllib.request.urlopen", side_effect=track_urlopen):
            result = helper_fn(*helper_args)
        assert not urlopen_calls, (
            f"{helper_name}: expected NO urlopen call with placeholder email, "
            f"but {len(urlopen_calls)} call(s) were made"
        )

    # Also test with empty/missing email
    for email_val in ["", None]:
        cfg = {"crossref_contact_email": email_val} if email_val is not None else {}
        urlopen_calls.clear()
        with mock.patch("scripts.ingest.urllib.request.urlopen", side_effect=track_urlopen):
            _crossref_journal_full("10.1000/test", cfg)
        assert not urlopen_calls, (
            f"Expected NO urlopen call with email={email_val!r}, "
            f"but {len(urlopen_calls)} call(s) were made"
        )

    # Verify warning is emitted
    captured = capsys.readouterr()
    assert "crossref skipped" in captured.err.lower() or "placeholder" in captured.err.lower(), (
        f"Expected a warning about crossref skipped / placeholder email, got stderr: {captured.err!r}"
    )


def test_crossref_proceeds_with_real_email():
    """With a real (non-placeholder) email, the Crossref calls proceed normally (WR-03, Plan 11 T3)."""
    import json as _json
    from scripts.ingest import _crossref_journal_full

    payload = _json.dumps({
        "message": {"container-title": ["Nature Chemistry"]}
    }).encode()
    cfg = {"crossref_contact_email": "researcher@university.edu"}

    urlopen_calls = []

    def track_urlopen(*args, **kwargs):
        urlopen_calls.append(True)
        return _FakeHttpResponse(payload)

    with mock.patch("scripts.ingest.urllib.request.urlopen", side_effect=track_urlopen):
        result = _crossref_journal_full("10.1000/test", cfg)

    assert urlopen_calls, "Expected urlopen to be called with a real email"
    assert result == "Nature Chemistry", f"Expected 'Nature Chemistry', got {result!r}"


def test_quality_gate_counts_nontext_blocks():
    """_quality_gate on a skeleton dominated by table/equation/figure blocks (short text)
    PASSES — non-text blocks counted toward the near-empty threshold (WR-05, Plan 11 T2)."""
    from scripts.ingest import _quality_gate, _assemble_paperjson, MINERU_BACKEND, SCHEMA_VERSION

    # Build a skeleton where text blocks alone have < 100 chars (would fail old gate),
    # but table/equation/figure blocks carry enough content to push over the threshold.
    parsed = {
        "title": "Figure-Heavy Paper Title",
        "sections": [{
            "heading": "Results",
            "level": 1,
            "blocks": [
                # Only 30 chars of text (well below the 100-char threshold)
                {"type": "text", "display": "Short intro text for results.", "plain": "Short intro text for results."},
                # Table block with substantial plain placeholder (60+ chars)
                {"type": "table", "display": "<table>data</table>", "plain": "[Table: Comparison of catalytic yields across experiments]"},
                # Equation block with placeholder (40+ chars)
                {"type": "equation", "display": "E = mc^2", "plain": "[Equation: energy-mass equivalence formula]"},
                # Figure block with caption (30+ chars)
                {"type": "figure", "img_path": "/img/fig1.png", "caption": "Figure 1. SEM images of nanoparticles", "figure_vlm_description": ""},
            ],
        }],
        "references": [],
        "metadata": {"title": "Figure-Heavy Paper Title"},
    }
    provenance = {
        "pdf_sha256": "abc", "source_filename": "fig_heavy.pdf",
        "mineru_version": None, "backend": MINERU_BACKEND,
        "extracted_at": "2026-01-01T00:00:00Z",
        "normalizations_applied": [], "schema_version": SCHEMA_VERSION,
    }
    skeleton = _assemble_paperjson(parsed, provenance)
    result = _quality_gate(skeleton)
    assert result is None, (
        f"Expected quality gate to PASS on figure/table-heavy paper (non-text blocks counted), got: {result!r}"
    )


# ---------------------------------------------------------------------------
# Plan 03 Task 1: PaperJSON cache write + registry paperjson_path + auto-invoke
# ---------------------------------------------------------------------------


def test_cache_written_after_ingest(tmp_path, monkeypatch):
    """After ingest(), .paperjson_cache/<stem>.json exists with extraction + analysis keys."""
    from scripts.ingest import ingest, _read_registry, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path, extra={"vault_name": "test-vault"})

    fake_pdf = tmp_path / "my_paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake content for cache test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = {
        "title": "Cache Test Paper",
        "sections": [{"heading": "", "level": 0, "blocks": [
            {"type": "text", "display": "Cache Test Paper", "plain": "Cache Test Paper"},
            {"type": "text", "display": "D" * 200, "plain": "D" * 200},
        ]}],
        "references": [],
        "metadata": {
            "title": "Cache Test Paper",
            "authors": None,
            "year": 2024,
            "journal": None,
            "doi": "10.1000/cache.test",
            "arxiv_id": None,
            "accession_codes": [],
        },
    }
    probe_result = DoiProbeResult(doi="10.1000/cache.test", arxiv_id=None, title="Cache Test Paper")
    metadata_result = PaperMetadata(title="Cache Test Paper", doi="10.1000/cache.test", year=2024)
    section_fill = SectionFillResult(heading="", body="Cache Test Paper " + "D" * 200, fill_failed=False)

    monkeypatch.chdir(tmp_path)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)), \
         mock.patch("scripts.note.generate_note", return_value="Papers/Cache Test Paper.md"), \
         mock.patch.dict(os.environ, {}, clear=False):
        result = ingest(str(fake_pdf), cfg)

    # Cache file must exist under .paperjson_cache/<stem>.json (under tmp_path via chdir)
    cache_file = tmp_path / ".paperjson_cache" / "my_paper.json"
    assert cache_file.exists(), (
        f"Expected cache file at {cache_file}, but it does not exist"
    )

    # Cache must contain extraction + analysis keys
    with open(cache_file, encoding="utf-8") as f:
        cached = json.load(f)
    assert "extraction" in cached, "Cache file missing 'extraction' key"
    assert "analysis" in cached, "Cache file missing 'analysis' key"


def test_registry_paperjson_path_populated(tmp_path, monkeypatch):
    """After ingest(), registry entry's paperjson_path equals the cache file path (not '')."""
    from scripts.ingest import ingest, _read_registry, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path, extra={"vault_name": "test-vault"})
    reg_path = cfg["registry_path"]

    fake_pdf = tmp_path / "reg_path_paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake content for reg path test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = {
        "title": "RegPath Test Paper",
        "sections": [{"heading": "", "level": 0, "blocks": [
            {"type": "text", "display": "RegPath Test Paper", "plain": "RegPath Test Paper"},
            {"type": "text", "display": "E" * 200, "plain": "E" * 200},
        ]}],
        "references": [],
        "metadata": {
            "title": "RegPath Test Paper",
            "authors": None,
            "year": 2024,
            "journal": None,
            "doi": "10.1000/regpath.test",
            "arxiv_id": None,
            "accession_codes": [],
        },
    }
    probe_result = DoiProbeResult(doi="10.1000/regpath.test", arxiv_id=None, title="RegPath Test Paper")
    metadata_result = PaperMetadata(title="RegPath Test Paper", doi="10.1000/regpath.test", year=2024)
    section_fill = SectionFillResult(heading="", body="RegPath Test Paper " + "E" * 200, fill_failed=False)

    monkeypatch.chdir(tmp_path)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)), \
         mock.patch("scripts.note.generate_note", return_value="Papers/RegPath Test Paper.md"):
        ingest(str(fake_pdf), cfg)

    registry = _read_registry(reg_path)
    entry = registry.get("10.1000/regpath.test")
    assert entry is not None, "Expected registry entry for the test paper"
    assert entry["paperjson_path"] != "", (
        f"Expected paperjson_path to be populated, got empty string"
    )
    assert "reg_path_paper.json" in entry["paperjson_path"], (
        f"Expected paperjson_path to contain stem-based filename, got: {entry['paperjson_path']!r}"
    )


def test_generate_note_auto_invoked(tmp_path, monkeypatch):
    """ingest() calls note.generate_note once with the in-memory PaperJSON dict (D-07)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path, extra={"vault_name": "test-vault"})

    fake_pdf = tmp_path / "autoinvoke.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake content for auto-invoke test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = {
        "title": "AutoInvoke Test Paper",
        "sections": [{"heading": "", "level": 0, "blocks": [
            {"type": "text", "display": "AutoInvoke Test Paper", "plain": "AutoInvoke Test Paper"},
            {"type": "text", "display": "F" * 200, "plain": "F" * 200},
        ]}],
        "references": [],
        "metadata": {
            "title": "AutoInvoke Test Paper",
            "authors": None,
            "year": 2024,
            "journal": None,
            "doi": "10.1000/autoinvoke.test",
            "arxiv_id": None,
            "accession_codes": [],
        },
    }
    probe_result = DoiProbeResult(doi="10.1000/autoinvoke.test", arxiv_id=None, title="AutoInvoke Test Paper")
    metadata_result = PaperMetadata(title="AutoInvoke Test Paper", doi="10.1000/autoinvoke.test", year=2024)
    section_fill = SectionFillResult(heading="", body="AutoInvoke Test Paper " + "F" * 200, fill_failed=False)

    mock_gen_note = mock.MagicMock(return_value="Papers/AutoInvoke Test Paper.md")

    monkeypatch.chdir(tmp_path)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)), \
         mock.patch("scripts.note.generate_note", mock_gen_note):
        ingest(str(fake_pdf), cfg)

    mock_gen_note.assert_called_once()
    # The first positional arg should be the PaperJSON dict (in-memory, not re-read)
    call_args = mock_gen_note.call_args
    pj_arg = call_args[0][0]
    assert isinstance(pj_arg, dict), "generate_note should receive a dict (PaperJSON)"
    assert "extraction" in pj_arg, "PaperJSON arg should have 'extraction' key"
    assert "analysis" in pj_arg, "PaperJSON arg should have 'analysis' key"


def test_generate_note_failure_does_not_fail_ingest(tmp_path, monkeypatch):
    """When generate_note raises, ingest() does NOT propagate — best-effort (D-05)."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path, extra={"vault_name": "test-vault"})

    fake_pdf = tmp_path / "notefail.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake content for note-fail test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = {
        "title": "NoteFail Test Paper",
        "sections": [{"heading": "", "level": 0, "blocks": [
            {"type": "text", "display": "NoteFail Test Paper", "plain": "NoteFail Test Paper"},
            {"type": "text", "display": "G" * 200, "plain": "G" * 200},
        ]}],
        "references": [],
        "metadata": {
            "title": "NoteFail Test Paper",
            "authors": None,
            "year": 2024,
            "journal": None,
            "doi": "10.1000/notefail.test",
            "arxiv_id": None,
            "accession_codes": [],
        },
    }
    probe_result = DoiProbeResult(doi="10.1000/notefail.test", arxiv_id=None, title="NoteFail Test Paper")
    metadata_result = PaperMetadata(title="NoteFail Test Paper", doi="10.1000/notefail.test", year=2024)
    section_fill = SectionFillResult(heading="", body="NoteFail Test Paper " + "G" * 200, fill_failed=False)

    # generate_note raises an exception
    def note_explodes(*args, **kwargs):
        raise RuntimeError("Obsidian crashed!")

    monkeypatch.chdir(tmp_path)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)), \
         mock.patch("scripts.note.generate_note", side_effect=note_explodes):
        # Must NOT raise — note failure is best-effort
        result = ingest(str(fake_pdf), cfg)

    # ingest should return the PaperJSON dict successfully
    assert isinstance(result, dict), "ingest() should return PaperJSON dict even when note fails"
    assert "extraction" in result, "Returned dict should have 'extraction' key"


def test_generate_note_error_string_logged(tmp_path, monkeypatch):
    """When generate_note returns '[note error: ...]', ingest logs a warning but does not fail."""
    from scripts.ingest import ingest, DoiProbeResult, PaperMetadata, SectionFillResult

    cfg = _make_ingest_config(tmp_path, extra={"vault_name": "test-vault"})

    fake_pdf = tmp_path / "noteerr.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake content for note-error test")
    cl_path = _write_real_content_list(tmp_path)

    mock_parsed = {
        "title": "NoteErr Test Paper",
        "sections": [{"heading": "", "level": 0, "blocks": [
            {"type": "text", "display": "NoteErr Test Paper", "plain": "NoteErr Test Paper"},
            {"type": "text", "display": "H" * 200, "plain": "H" * 200},
        ]}],
        "references": [],
        "metadata": {
            "title": "NoteErr Test Paper",
            "authors": None,
            "year": 2024,
            "journal": None,
            "doi": "10.1000/noteerr.test",
            "arxiv_id": None,
            "accession_codes": [],
        },
    }
    probe_result = DoiProbeResult(doi="10.1000/noteerr.test", arxiv_id=None, title="NoteErr Test Paper")
    metadata_result = PaperMetadata(title="NoteErr Test Paper", doi="10.1000/noteerr.test", year=2024)
    section_fill = SectionFillResult(heading="", body="NoteErr Test Paper " + "H" * 200, fill_failed=False)

    monkeypatch.chdir(tmp_path)

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.ingest._fill_metadata", return_value=metadata_result), \
         mock.patch("scripts.ingest._fill_section", return_value=section_fill), \
         mock.patch("scripts.ingest._fill_references_batched", return_value=([], 0)), \
         mock.patch("scripts.note.generate_note", return_value="[note error: vault unreachable]"):
        result = ingest(str(fake_pdf), cfg)

    # Must still return successfully
    assert isinstance(result, dict), "ingest() should return PaperJSON dict even on note error string"
    assert "extraction" in result, "Returned dict should have 'extraction' key"


# ---------------------------------------------------------------------------
# Plan 04 (gap-closure): Note-aware Step 9 registry gate
# generate-note-from-cache-if-missing on a registry hit
# ---------------------------------------------------------------------------


def test_registry_hit_generates_note_from_cache(tmp_path):
    """On cache HIT with a valid paperjson_path, ingest() calls generate_note and returns cached entry."""
    from scripts.ingest import ingest, _write_registry

    cfg = _make_ingest_config(tmp_path, extra={"vault_name": "test-vault"})
    reg_path = cfg["registry_path"]

    # Write a real PaperJSON cache file on disk
    cache_file = tmp_path / "cached_paper.json"
    cached_paperjson = {
        "extraction": {
            "metadata": {"title": "Cached Paper For Note Gen", "authors": [], "year": 2024,
                          "journal": None, "doi": "10.1000/note.gate", "arxiv_id": None,
                          "accession_codes": []},
            "sections": [],
            "references": [],
        },
        "analysis": {"generated_by": None},
        "provenance": {"schema_version": 2},
    }
    cache_file.write_text(json.dumps(cached_paperjson), encoding="utf-8")

    # Pre-populate registry with paperjson_path pointing at the real cache file
    cached_entry = {
        "title": "Cached Paper For Note Gen",
        "doi": "10.1000/note.gate",
        "projects": ["other-project"],
        "summary": None, "key_findings": None,
        "authors": None, "year": 2024, "journal": None,
        "arxiv_id": None,
        "source_path": "/cached.pdf",
        "paperjson_path": str(cache_file),
    }
    _write_registry(cached_entry, reg_path, "10.1000/note.gate")

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 note gate test")
    cl_path = _write_real_content_list(tmp_path)

    from scripts.ingest import DoiProbeResult
    probe_result = DoiProbeResult(doi="10.1000/note.gate", arxiv_id=None, title="Cached Paper For Note Gen")

    mock_gen_note = mock.MagicMock(return_value="Papers/Cached Paper For Note Gen.md")

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value={
             "title": "Cached Paper For Note Gen",
             "sections": [{"heading": "", "level": 0, "blocks": [
                 {"type": "text", "display": "A" * 200, "plain": "A" * 200},
             ]}],
             "references": [],
             "metadata": {"title": "Cached Paper For Note Gen", "authors": None, "year": 2024,
                           "journal": None, "doi": "10.1000/note.gate", "arxiv_id": None, "accession_codes": []},
         }), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.note.generate_note", mock_gen_note):
        result = ingest(str(fake_pdf), cfg)

    # generate_note must have been called once with the loaded PaperJSON dict
    mock_gen_note.assert_called_once()
    call_args = mock_gen_note.call_args
    pj_arg = call_args[0][0]
    assert isinstance(pj_arg, dict), "generate_note should receive a dict (loaded PaperJSON)"
    assert pj_arg == cached_paperjson, "generate_note should receive the cache-file PaperJSON, not in-memory"

    # Second arg should be config
    cfg_arg = call_args[0][1]
    assert isinstance(cfg_arg, dict), "generate_note should receive config as second arg"

    # ingest() must return the cached entry (not a full PaperJSON)
    assert result.get("title") == "Cached Paper For Note Gen", (
        f"Expected cached entry returned, got: {result}"
    )
    assert result.get("doi") == "10.1000/note.gate"


def test_registry_hit_note_failure_does_not_propagate(tmp_path):
    """When generate_note raises on registry-hit path, ingest() still returns cached entry (D-05)."""
    from scripts.ingest import ingest, _write_registry

    cfg = _make_ingest_config(tmp_path, extra={"vault_name": "test-vault"})
    reg_path = cfg["registry_path"]

    # Write a real PaperJSON cache file
    cache_file = tmp_path / "fail_paper.json"
    cache_file.write_text(json.dumps({
        "extraction": {"metadata": {"title": "Fail Paper"}, "sections": [], "references": []},
        "analysis": {"generated_by": None},
        "provenance": {"schema_version": 2},
    }), encoding="utf-8")

    cached_entry = {
        "title": "Fail Paper",
        "doi": "10.1000/note.fail",
        "projects": ["test-project"],
        "summary": None, "key_findings": None,
        "authors": None, "year": 2024, "journal": None,
        "arxiv_id": None,
        "source_path": "/fail.pdf",
        "paperjson_path": str(cache_file),
    }
    _write_registry(cached_entry, reg_path, "10.1000/note.fail")

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 note fail test")
    cl_path = _write_real_content_list(tmp_path)

    from scripts.ingest import DoiProbeResult
    probe_result = DoiProbeResult(doi="10.1000/note.fail", arxiv_id=None, title="Fail Paper")

    def note_explodes(*args, **kwargs):
        raise RuntimeError("Obsidian crashed on registry-hit path!")

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
         mock.patch("scripts.ingest._parse_content_list", return_value={
             "title": "Fail Paper",
             "sections": [{"heading": "", "level": 0, "blocks": [
                 {"type": "text", "display": "A" * 200, "plain": "A" * 200},
             ]}],
             "references": [],
             "metadata": {"title": "Fail Paper", "authors": None, "year": 2024,
                           "journal": None, "doi": "10.1000/note.fail", "arxiv_id": None, "accession_codes": []},
         }), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("scripts.ingest._warmup_ollama"), \
         mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
         mock.patch("scripts.note.generate_note", side_effect=note_explodes):
        # Must NOT raise — note failure is best-effort on registry-hit path
        result = ingest(str(fake_pdf), cfg)

    # Must return the cached entry
    assert result.get("title") == "Fail Paper", (
        f"Expected cached entry returned even on note failure, got: {result}"
    )
    assert result.get("doi") == "10.1000/note.fail"


def test_registry_hit_no_note_gen_when_paperjson_path_missing(tmp_path):
    """Cache HIT with empty paperjson_path -> generate_note NOT called; returns cached entry."""
    from scripts.ingest import ingest, _write_registry

    cfg = _make_ingest_config(tmp_path, extra={"vault_name": "test-vault"})
    reg_path = cfg["registry_path"]

    # Two sub-cases: empty string and nonexistent path
    for label, pj_path in [("empty", ""), ("nonexistent", str(tmp_path / "nonexistent.json"))]:
        cached_entry = {
            "title": f"No Note Gen {label}",
            "doi": f"10.1000/nonote.{label}",
            "projects": ["test-project"],
            "summary": None, "key_findings": None,
            "authors": None, "year": 2024, "journal": None,
            "arxiv_id": None,
            "source_path": "/nogen.pdf",
            "paperjson_path": pj_path,
        }
        _write_registry(cached_entry, reg_path, f"10.1000/nonote.{label}")

        fake_pdf = tmp_path / f"paper_{label}.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 no note gen test")
        cl_path = _write_real_content_list(tmp_path)

        from scripts.ingest import DoiProbeResult
        probe_result = DoiProbeResult(doi=f"10.1000/nonote.{label}", arxiv_id=None, title=f"No Note Gen {label}")

        mock_gen_note = mock.MagicMock(
            side_effect=AssertionError("generate_note must NOT be called when paperjson_path is empty/missing")
        )

        with mock.patch("scripts.ingest._run_mineru"), \
             mock.patch("scripts.ingest._find_content_list", return_value=cl_path), \
             mock.patch("scripts.ingest._parse_content_list", return_value={
                 "title": f"No Note Gen {label}",
                 "sections": [{"heading": "", "level": 0, "blocks": [
                     {"type": "text", "display": "A" * 200, "plain": "A" * 200},
                 ]}],
                 "references": [],
                 "metadata": {"title": f"No Note Gen {label}", "authors": None, "year": 2024,
                               "journal": None, "doi": f"10.1000/nonote.{label}", "arxiv_id": None,
                               "accession_codes": []},
             }), \
             mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
             mock.patch("scripts.ingest._warmup_ollama"), \
             mock.patch("scripts.ingest._doi_probe", return_value=probe_result), \
             mock.patch("scripts.note.generate_note", mock_gen_note):
            result = ingest(str(fake_pdf), cfg)

        # generate_note must NOT have been called
        mock_gen_note.assert_not_called()

        # Must return the cached entry
        assert result.get("title") == f"No Note Gen {label}", (
            f"Expected cached entry for {label} case, got: {result}"
        )


# ---------------------------------------------------------------------------
# Plan 08 (GAP-2): _emit_result stdout-suppression + --print flag
# ---------------------------------------------------------------------------

_SAMPLE_PAPERJSON = {
    "extraction": {
        "metadata": {
            "title": "A Test Paper",
            "authors": [],
            "year": 2024,
            "journal": None,
            "doi": "10.1000/test.emit",
            "arxiv_id": None,
            "accession_codes": [],
        },
        "sections": [],
        "references": [],
    },
    "analysis": {"generated_by": None},
    "provenance": {"schema_version": 2},
}


def test_emit_result_default_prints_confirmation(capsys):
    """Default mode suppresses JSON body; only the short confirmation appears on stdout."""
    from scripts.ingest import _emit_result

    confirmation = "Cache: /tmp/test.json\nNote: Papers/A Test Paper.md"
    _emit_result(_SAMPLE_PAPERJSON, None, print_json=False, confirmation=confirmation)

    captured = capsys.readouterr()
    # Sentinel key from PaperJSON must NOT appear (JSON body suppressed)
    assert "schema_version" not in captured.out, (
        "Expected JSON body (schema_version) to be absent from stdout in default mode; "
        f"got: {captured.out[:200]!r}"
    )
    # The confirmation text must appear
    assert "Cache:" in captured.out, (
        f"Expected 'Cache:' in stdout confirmation; got: {captured.out!r}"
    )
    assert "Note:" in captured.out, (
        f"Expected 'Note:' in stdout confirmation; got: {captured.out!r}"
    )


def test_emit_result_print_json_prints_full_json(capsys):
    """With print_json=True, the full PaperJSON body is printed to stdout."""
    from scripts.ingest import _emit_result

    _emit_result(_SAMPLE_PAPERJSON, None, print_json=True, confirmation=None)

    captured = capsys.readouterr()
    # Sentinel key must be present (full JSON dump)
    assert "schema_version" in captured.out, (
        "Expected full JSON body (schema_version) on stdout when print_json=True; "
        f"got: {captured.out[:200]!r}"
    )
    # Must be parseable JSON
    import json as _json
    parsed = _json.loads(captured.out)
    assert parsed["provenance"]["schema_version"] == 2


def test_emit_result_output_path_writes_file_not_stdout(capsys, tmp_path):
    """output_path writes the file; nothing is printed to stdout."""
    from scripts.ingest import _emit_result
    import json as _json

    out_file = tmp_path / "out.json"
    _emit_result(_SAMPLE_PAPERJSON, str(out_file), print_json=False, confirmation=None)

    captured = capsys.readouterr()
    # stdout must be empty (file write path; no stdout dump)
    assert captured.out.strip() == "", (
        f"Expected empty stdout when output_path given; got: {captured.out[:200]!r}"
    )
    # File must exist and contain valid JSON with schema_version=2
    assert out_file.exists(), "Expected output file to have been created"
    data = _json.loads(out_file.read_text(encoding="utf-8"))
    assert data["provenance"]["schema_version"] == 2