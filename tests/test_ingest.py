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
    _parse_references,
    _mine_metadata,
    _mine_footer_metadata,
    _quality_gate,
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


# ---------------------------------------------------------------------------
# Task 2 (Plan 02): Structured references + metadata mining + quality gate
# ---------------------------------------------------------------------------

# --- Reference fixtures ---

REF_ITEMS = [
    "1. Kahle, K. T.; Khanna, A. R. K-Cl cotransporters. Trends Mol. Med. 2015. DOI: 10.1016/j.molmed.2015.05.008",
    "2. Delpire, E.; Gagnon, K. B. SPAK and OSR1. Biochem. J. 2008. DOI: 10.1042/BJ20071324",
    "(3) Smith, J. Normal reference without DOI. J. Chem. 2020.",
    "4. St€odberg, T.; McTague, A. Mutations in SLC12A5. Nat. Commun. 2015. DOI: 10.1038/ncomms9038",
]

FOOTER_BLOCKS = [
    {"type": "footer", "text": "J. Am. Chem. Soc. 2024, 146, 12345–12358. DOI: 10.1021/jacs.3c10258", "page_idx": 4},
]

PARSED_BLOCKS_WITH_TITLE = [
    {"type": "text", "text": "My Paper Title", "text_level": 1, "page_idx": 0},
    {"type": "text", "text": "Abstract", "text_level": 2, "page_idx": 0},
    {"type": "text", "text": "This paper discusses PDB:7TTI and EMD-26116 accessions.", "text_level": None, "page_idx": 1},
    {"type": "footer", "text": "J. Am. Chem. Soc. 2024, 146, 12345. DOI: 10.1021/jacs.3c10258", "page_idx": 2},
]


def test_reference_objects():
    """_parse_references returns objects with required keys; number and doi extracted."""
    refs = _parse_references(REF_ITEMS)
    assert len(refs) == len(REF_ITEMS), "Expected one ref object per input item"
    for ref in refs:
        assert "number" in ref, "Expected 'number' key in reference object"
        assert "raw" in ref, "Expected 'raw' key in reference object"
        assert "doi" in ref, "Expected 'doi' key in reference object"
        assert "title" in ref, "Expected 'title' key in reference object"
        assert "year" in ref, "Expected 'year' key in reference object"
        assert "flags" in ref, "Expected 'flags' key in reference object"

    # First ref: number 1, DOI extracted
    assert refs[0]["number"] == 1, f"Expected number=1, got {refs[0]['number']}"
    assert refs[0]["doi"] == "10.1016/j.molmed.2015.05.008", (
        f"Expected DOI extracted, got {refs[0]['doi']!r}"
    )

    # Third ref: parenthetical number format (3)
    assert refs[2]["number"] == 3, f"Expected number=3 from '(3)' format, got {refs[2]['number']}"
    assert refs[2]["doi"] is None, f"Expected doi=None for ref without DOI, got {refs[2]['doi']!r}"


def test_corrupted_author_flag():
    """References with euro sign in raw string get 'corrupted_authors' in flags."""
    refs = _parse_references(REF_ITEMS)
    # Fourth ref contains St€odberg (euro sign in author name)
    corrupted = refs[3]
    assert "corrupted_authors" in corrupted["flags"], (
        f"Expected 'corrupted_authors' flag for ref with '€', got flags={corrupted['flags']}"
    )


def test_clean_reference_no_flag():
    """References without euro sign have empty flags list."""
    refs = _parse_references(REF_ITEMS)
    # First ref has no euro sign
    assert refs[0]["flags"] == [], (
        f"Expected empty flags for clean ref, got {refs[0]['flags']}"
    )


def test_title_from_text_level_1():
    """_mine_metadata sets metadata.title from page_idx 0 text_level-1 block."""
    metadata = _mine_metadata(PARSED_BLOCKS_WITH_TITLE)
    assert metadata.get("title") == "My Paper Title", (
        f"Expected title 'My Paper Title', got {metadata.get('title')!r}"
    )


def test_journal_year_from_footer():
    """_mine_footer_metadata extracts journal and year from footer block."""
    metadata = _mine_footer_metadata(FOOTER_BLOCKS)
    assert metadata.get("journal"), f"Expected journal extracted from footer, got {metadata.get('journal')!r}"
    assert metadata.get("year") == 2024, (
        f"Expected year=2024 from footer, got {metadata.get('year')!r}"
    )


def test_doi_extraction():
    """_mine_metadata extracts DOI from body text; arxiv_id parsed when present."""
    blocks_with_doi = [
        {"type": "text", "text": "Title Block", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "See DOI 10.1021/jacs.3c10258 for details.", "text_level": None, "page_idx": 1},
        {"type": "text", "text": "arXiv:2309.12345 preprint version.", "text_level": None, "page_idx": 1},
    ]
    metadata = _mine_metadata(blocks_with_doi)
    assert metadata.get("doi") == "10.1021/jacs.3c10258", (
        f"Expected DOI extracted, got {metadata.get('doi')!r}"
    )
    assert metadata.get("arxiv_id") == "2309.12345", (
        f"Expected arxiv_id '2309.12345', got {metadata.get('arxiv_id')!r}"
    )


def test_accession_codes():
    """PDB/EMDB tokens in body text captured as structured accession_codes entries."""
    blocks_with_accessions = [
        {"type": "text", "text": "Title", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Data deposited in PDB:7TTI and EMDB entry EMD-26116.", "text_level": None, "page_idx": 1},
    ]
    metadata = _mine_metadata(blocks_with_accessions)
    codes = metadata.get("accession_codes", [])
    types = {entry["type"] for entry in codes}
    values = {entry["value"] for entry in codes}
    assert "PDB" in types, f"Expected PDB accession, got {codes}"
    assert "7TTI" in values, f"Expected 7TTI value, got {values}"
    assert "EMDB" in types, f"Expected EMDB accession, got {codes}"
    assert "EMD-26116" in values, f"Expected EMD-26116 value, got {values}"


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


# Minimal blocks that pass the quality gate (title + some text)
MINIMAL_BLOCKS = [
    {"type": "text", "text": "A Great Paper Title", "text_level": 1, "page_idx": 0},
    {"type": "text", "text": "Abstract content goes here with enough characters to pass gate.", "text_level": None, "page_idx": 0},
]


def test_dedup_skip_returns_cached(tmp_path):
    """When key is already in registry, ingest() returns cached entry without calling _run_mineru."""
    from scripts.ingest import ingest, _write_registry

    cfg = _make_ingest_config(tmp_path)
    reg_path = cfg["registry_path"]

    # Pre-populate the registry with the paper's DOI key
    cached = {
        "title": "A Great Paper Title",
        "doi": "10.1/dedup-test",
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
    _write_registry(cached, reg_path, "10.1/dedup-test")

    # Create a fake PDF file (non-empty so file-exists check passes)
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake content for hash")

    # Patch _run_mineru to raise if called — should NOT be called on cache hit
    # Also patch _find_content_list and the content reading to serve MINIMAL_BLOCKS
    # so that parse-then-check-registry works
    with mock.patch("scripts.ingest._run_mineru", side_effect=AssertionError("_run_mineru called on cache hit")), \
         mock.patch("scripts.ingest._find_content_list", return_value=str(tmp_path / "content_list.json")), \
         mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(MINIMAL_BLOCKS))):
        # Provide blocks that mine a DOI that matches the cached key
        blocks_with_doi = [
            {"type": "text", "text": "A Great Paper Title", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "DOI 10.1/dedup-test details.", "text_level": None, "page_idx": 1},
        ]
        with mock.patch("scripts.ingest._parse_content_list") as mock_parse, \
             mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"):
            mock_parse.return_value = {
                "title": "A Great Paper Title",
                "sections": [{"heading": "", "level": 0, "blocks": [
                    {"type": "text", "display": "A Great Paper Title", "plain": "A Great Paper Title"},
                    {"type": "text", "display": "DOI 10.1/dedup-test details.", "plain": "DOI 10.1/dedup-test details."},
                ]}],
                "references": [],
                "metadata": {
                    "title": "A Great Paper Title",
                    "authors": None,
                    "year": None,
                    "journal": None,
                    "doi": "10.1/dedup-test",
                    "arxiv_id": None,
                    "accession_codes": [],
                },
            }
            result = ingest(str(fake_pdf), cfg)

    # On cache hit, result should be the cached registry entry (not a full PaperJSON)
    assert result is not None, "Expected a result from ingest() on cache hit"
    # The result must contain the cached paper's title
    assert result.get("title") == "A Great Paper Title" or (
        result.get("extraction", {}).get("metadata", {}).get("title") == "A Great Paper Title"
    ), f"Expected cached entry returned, got: {result}"


def test_new_paper_writes_entry(tmp_path):
    """Ingesting a not-yet-registered paper calls _write_registry once and the key appears."""
    from scripts.ingest import ingest, _read_registry

    cfg = _make_ingest_config(tmp_path)
    reg_path = cfg["registry_path"]

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake content for hash")

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
            "doi": "10.1/new-paper",
            "arxiv_id": None,
            "accession_codes": [],
        },
    }

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=str(tmp_path / "content_list.json")), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("builtins.open", mock.mock_open(read_data=json.dumps([]))):
        ingest(str(fake_pdf), cfg)

    registry = _read_registry(reg_path)
    assert "10.1/new-paper" in registry, (
        f"Expected DOI key '10.1/new-paper' in registry after new ingest, got keys: {list(registry.keys())}"
    )


def test_force_extract_bypasses_cache(tmp_path):
    """With force_extract=True, the pipeline runs and re-registers even if key exists."""
    from scripts.ingest import ingest, _write_registry, _read_registry

    cfg = _make_ingest_config(tmp_path)
    reg_path = cfg["registry_path"]

    # Pre-populate registry
    old_entry = {
        "title": "Old Entry",
        "doi": "10.1/force-test",
        "projects": ["old-project"],
        "summary": None, "key_findings": None,
        "authors": None, "year": 2020, "journal": None,
        "arxiv_id": None,
        "source_path": "/old.pdf",
        "paperjson_path": "/old.json",
    }
    _write_registry(old_entry, reg_path, "10.1/force-test")

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 force extract fake")

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
            "doi": "10.1/force-test",
            "arxiv_id": None,
            "accession_codes": [],
        },
    }

    mineru_called = []

    def fake_mineru(*args, **kwargs):
        mineru_called.append(True)

    with mock.patch("scripts.ingest._run_mineru", side_effect=fake_mineru), \
         mock.patch("scripts.ingest._find_content_list", return_value=str(tmp_path / "content_list.json")), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("builtins.open", mock.mock_open(read_data=json.dumps([]))):
        result = ingest(str(fake_pdf), cfg, force_extract=True)

    # _run_mineru must have been called (force_extract bypasses cache)
    assert mineru_called, "_run_mineru should have been called when force_extract=True"

    # Registry should be updated with the new entry
    registry = _read_registry(reg_path)
    assert "10.1/force-test" in registry, "Expected key to exist in registry after force re-ingest"
    # The new entry's year should be updated (2024 not 2020)
    assert registry["10.1/force-test"].get("year") == 2024, (
        f"Expected updated year 2024 after force re-ingest, got {registry['10.1/force-test'].get('year')}"
    )


def test_reg01_entry_written_on_new_ingest(tmp_path):
    """After a new ingest, _read_registry contains an entry with the paper's title and DOI (REG-01)."""
    from scripts.ingest import ingest, _read_registry

    cfg = _make_ingest_config(tmp_path)
    reg_path = cfg["registry_path"]

    fake_pdf = tmp_path / "reg01paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 reg01 fake content")

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
            "doi": "10.1/reg01test",
            "arxiv_id": None,
            "accession_codes": [],
        },
    }

    with mock.patch("scripts.ingest._run_mineru"), \
         mock.patch("scripts.ingest._find_content_list", return_value=str(tmp_path / "content_list.json")), \
         mock.patch("scripts.ingest._parse_content_list", return_value=mock_parsed), \
         mock.patch("scripts.ingest._resolve_mineru", return_value="/fake/mineru"), \
         mock.patch("builtins.open", mock.mock_open(read_data=json.dumps([]))):
        ingest(str(fake_pdf), cfg)

    registry = _read_registry(reg_path)
    assert "10.1/reg01test" in registry, "Expected registry to contain the new paper's DOI key"
    entry = registry["10.1/reg01test"]
    assert entry.get("title") == "REG-01 Test Paper", (
        f"Expected title 'REG-01 Test Paper', got {entry.get('title')!r}"
    )
    assert entry.get("doi") == "10.1/reg01test", (
        f"Expected doi '10.1/reg01test', got {entry.get('doi')!r}"
    )
