"""
Wave-0 FAILING test contract for Phase 5 semantic search (EMBED-01/02/03).

scripts/embed.py does not exist yet — these tests pin the exact behavior
Plans 02-03 must implement (see 05-01-PLAN.md, 05-02-PLAN.md, 05-03-PLAN.md):
  - test_embed_sections (EMBED-01): sections embedded w/ deterministic IDs + metadata
  - test_embed_no_explicit_abstract (EMBED-01 / Pitfall #2): no crash w/o "Abstract" heading
  - test_embed_skip_if_present (D-13): second run_embed makes zero /api/embed calls
  - test_search_similar_paper_grouping (EMBED-02 / D-06): hits grouped by registry_key
  - test_search_similar_scoring (D-09): score == round(1 - distance, 4), no relevance cutoff
  - test_embed_stub_sweep (EMBED-03 / D-14/D-15): stub embedded at title level
  - test_stub_upgrade_deletes_entry (D-16): stub entry deleted on matching paper upgrade

`scripts.embed` is imported at FUNCTION level (never module top) so pytest
collection succeeds cleanly before the module exists — mirrors the Phase 4
04-01 attribute-access / function-level-import precedent (tests/test_biblio.py
`from scripts import biblio  # noqa: PLC0415`).

Every test injects config["chroma_db_path"] = tmp_path / "chroma_db" so no
test ever writes to the real repo-root chroma_db/ — guarded at session scope
by tests/conftest.py::_guard_real_chroma_db.

A real chromadb.PersistentClient is used (NOT mocked) so actual Chroma cosine
distance/HNSW semantics are exercised; only the Ollama HTTP layer
(`scripts.embed._ollama_embed_call`) is mocked.

Run with: python -m pytest tests/test_embed.py -x
"""

import json
import math
import pathlib
import subprocess
import sys
from unittest import mock

_REPO_ROOT_FOR_SUBPROCESS = pathlib.Path(__file__).resolve().parent.parent
_EMBED_PY = _REPO_ROOT_FOR_SUBPROCESS / "scripts" / "embed.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embed_config(tmp_path, extra=None):
    """Build a minimal config dict pointing chroma_db_path/vault_path/registry_path
    at tmp_path so no test pollutes the real dev-machine stores (Pitfall #8)."""
    (tmp_path / "Papers").mkdir(exist_ok=True)
    (tmp_path / "Stubs").mkdir(exist_ok=True)
    cfg = {
        "chroma_db_path": str(tmp_path / "chroma_db"),
        "embed_model": "nomic-embed-text",
        "embed_timeout": 60,
        "embed_batch_size": 16,
        "embed_section_max_tokens": 2000,
        "vault_path": str(tmp_path),
        "registry_path": str(tmp_path / "registry.json"),
        "paperjson_cache_dir": str(tmp_path / ".paperjson_cache"),
    }
    if extra:
        cfg.update(extra)
    return cfg


def _make_paperjson(sections=None, title="Attention Is All You Need", doi="10.1234/attn", year=2017):
    """Build a minimal PaperJSON v2 dict with the REAL extraction.sections[] shape
    (heading/body only — NEVER blocks[], per Common Pitfall #1)."""
    if sections is None:
        sections = [
            {"heading": "Abstract", "body": "We propose the Transformer.", "fill_failed": False},
            {"heading": "Methods", "body": "Multi-head self-attention layers.", "fill_failed": False},
        ]
    return {
        "extraction": {
            "metadata": {"title": title, "doi": doi, "year": year, "authors": []},
            "sections": sections,
            "references": [],
        },
        "analysis": {},
        "provenance": {"schema_version": 2},
    }


def _write_stub(tmp_path, title, stub_key, raw, filename):
    """Write a stub .md matching scripts/biblio.py's _render_stub() output shape."""
    doi_field = stub_key if stub_key.startswith("10.") else ""
    content = (
        "---\n"
        f'title: "{title}"\n'
        "authors: []\n"
        "year: 2015\n"
        f'doi: "{doi_field}"\n'
        "status: stub\n"
        f'stub_key: "{stub_key}"\n'
        "date_created: 2026-07-06\n"
        "cited_by:\n"
        '  - "Papers/Citing Paper.md"\n'
        "---\n\n"
        "*Stub: this paper has been cited but not yet fully ingested.*\n\n"
        "**Raw citation:**\n"
        f"{raw}\n"
    )
    stubs_dir = tmp_path / "Stubs"
    stubs_dir.mkdir(exist_ok=True)
    (stubs_dir / filename).write_text(content, encoding="utf-8")
    return stubs_dir / filename


def _fixed_vector(seed, dim=4):
    """Deterministic small vector for a given int seed (no real Ollama call)."""
    return [math.sin(seed + i) for i in range(dim)]


def _auto_embed(texts, model="nomic-embed-text", timeout=60):
    """Deterministic fallback embedder: one small vector per input text, length-matched
    (mirrors real _ollama_embed_call's 1-vector-per-input contract) for tests that don't
    need to control exact cosine-distance outcomes."""
    return [_fixed_vector(i) for i in range(len(texts))]


def _get_collection(chroma_db_path):
    import chromadb  # noqa: PLC0415
    client = chromadb.PersistentClient(path=chroma_db_path)
    return client.get_or_create_collection(
        name="papers", configuration={"hnsw": {"space": "cosine"}}
    )


# ---------------------------------------------------------------------------
# EMBED-01: sections embedded with deterministic IDs + metadata
# ---------------------------------------------------------------------------

def test_embed_sections(tmp_path):
    from scripts import embed  # noqa: PLC0415
    from scripts.ingest import _registry_key  # noqa: PLC0415

    config = _make_embed_config(tmp_path)
    paperjson = _make_paperjson()
    registry_key = _registry_key(paperjson["extraction"]["metadata"])

    with mock.patch("scripts.embed._ollama_embed_call", side_effect=_auto_embed) as mock_embed, \
         mock.patch("scripts.embed._unload_model"):
        result = embed.run_embed(paperjson, config)

    assert isinstance(result, str), "run_embed must return a string (non-fatal tail-stage contract)"
    assert mock_embed.called, "expected /api/embed to be called for a never-before-embedded paper"

    collection = _get_collection(config["chroma_db_path"])
    got = collection.get(where={"registry_key": registry_key}, include=["metadatas"])

    assert set(got["ids"]) == {
        f"{registry_key}::0-abstract::0",
        f"{registry_key}::1-methods::0",
    }, f"expected deterministic {{registry_key}}::{{idx}}-{{slug}}::{{part}} ids, got {got['ids']}"

    for meta in got["metadatas"]:
        assert meta["registry_key"] == registry_key
        assert meta["title"] == "Attention Is All You Need"
        assert meta["heading"] in ("Abstract", "Methods")
        assert meta["part"] == 0
        assert meta["doi"] == "10.1234/attn"
        assert meta["year"] == 2017
        assert meta["status"] == "paper"
        assert meta.get("vault_note"), "vault_note metadata must be present and non-empty"


def test_embed_no_explicit_abstract(tmp_path):
    """Pitfall #2: sections[] with no 'Abstract' heading embeds uniformly, no crash/warning."""
    from scripts import embed  # noqa: PLC0415
    from scripts.ingest import _registry_key  # noqa: PLC0415

    config = _make_embed_config(tmp_path)
    sections = [
        {"heading": "Structure of Multi-Head Attention", "body": "Full paper body text here.", "fill_failed": False},
        {"heading": "Experiments", "body": "We benchmark against RNN baselines.", "fill_failed": False},
    ]
    paperjson = _make_paperjson(sections=sections, title="No Abstract Paper", doi="10.5555/noabs", year=2019)
    registry_key = _registry_key(paperjson["extraction"]["metadata"])

    with mock.patch("scripts.embed._ollama_embed_call", side_effect=_auto_embed), \
         mock.patch("scripts.embed._unload_model"):
        result = embed.run_embed(paperjson, config)

    assert isinstance(result, str)
    assert not result.startswith("[embed warning:"), f"unexpected failure: {result}"

    collection = _get_collection(config["chroma_db_path"])
    got = collection.get(where={"registry_key": registry_key})
    assert len(got["ids"]) == 2, "both non-abstract sections should embed uniformly with no special-case crash"


def test_embed_duplicate_heading_sections(tmp_path):
    """CR-03 regression: two sections that slugify identically (e.g. two 'Results'
    headings) must both embed as distinct Chroma entries -- no DuplicateIDError
    swallowed into a silent [embed warning: ...] that would exclude the whole
    paper from search."""
    from scripts import embed  # noqa: PLC0415
    from scripts.ingest import _registry_key  # noqa: PLC0415

    config = _make_embed_config(tmp_path)
    sections = [
        {"heading": "Results", "body": "First results block.", "fill_failed": False},
        {"heading": "Results", "body": "Second results block, distinct content.", "fill_failed": False},
    ]
    paperjson = _make_paperjson(sections=sections, title="Duplicate Heading Paper", doi="10.1234/dup", year=2023)
    registry_key = _registry_key(paperjson["extraction"]["metadata"])

    with mock.patch("scripts.embed._ollama_embed_call", side_effect=_auto_embed), \
         mock.patch("scripts.embed._unload_model"):
        result = embed.run_embed(paperjson, config)

    assert not result.startswith("[embed warning:"), (
        f"a DuplicateIDError must never be swallowed into a silent warning: {result}"
    )

    collection = _get_collection(config["chroma_db_path"])
    got = collection.get(where={"registry_key": registry_key}, include=["metadatas"])
    assert set(got["ids"]) == {
        f"{registry_key}::0-results::0",
        f"{registry_key}::1-results::0",
    }, f"both same-heading sections must embed as distinct idx-prefixed entries, got {got['ids']}"


def test_embed_skip_if_present(tmp_path):
    """D-13: a second run_embed for the same registry_key makes zero /api/embed calls."""
    from scripts import embed  # noqa: PLC0415

    config = _make_embed_config(tmp_path)
    paperjson = _make_paperjson()

    with mock.patch("scripts.embed._ollama_embed_call", side_effect=_auto_embed) as mock_embed, \
         mock.patch("scripts.embed._unload_model"):
        embed.run_embed(paperjson, config)
        assert mock_embed.called, "first run must call /api/embed"
        mock_embed.reset_mock()

        embed.run_embed(paperjson, config)
        mock_embed.assert_not_called()  # zero Ollama calls on the already-embedded second run


# ---------------------------------------------------------------------------
# EMBED-02: search_similar paper-grouping + scoring (D-06, D-09)
# ---------------------------------------------------------------------------

def test_search_similar_paper_grouping(tmp_path):
    """EMBED-02/D-06: hits grouped by registry_key; paper ranked by its BEST section score."""
    from scripts import embed  # noqa: PLC0415
    from scripts.ingest import _registry_key  # noqa: PLC0415

    config = _make_embed_config(tmp_path)

    query_vec = [1.0, 0.0]
    identical_vec = [1.0, 0.0]   # cosine distance 0 -> score 1.0
    orthogonal_vec = [0.0, 1.0]  # cosine distance 1 -> score 0.0

    paper1 = _make_paperjson(
        sections=[
            {"heading": "Abstract", "body": "Highly relevant to the query.", "fill_failed": False},
            {"heading": "Methods", "body": "Unrelated methods text.", "fill_failed": False},
        ],
        title="Paper One", doi="10.1111/paper1", year=2020,
    )
    paper2 = _make_paperjson(
        sections=[{"heading": "Abstract", "body": "Also unrelated.", "fill_failed": False}],
        title="Paper Two", doi="10.2222/paper2", year=2021,
    )
    key1 = _registry_key(paper1["extraction"]["metadata"])
    key2 = _registry_key(paper2["extraction"]["metadata"])

    with mock.patch(
        "scripts.embed._ollama_embed_call",
        side_effect=[[identical_vec, orthogonal_vec], [orthogonal_vec], [query_vec]],
    ), mock.patch("scripts.embed._unload_model"):
        embed.run_embed(paper1, config)
        embed.run_embed(paper2, config)
        result = embed._search("relevant query", 5, config)

    assert isinstance(result, dict) and "papers" in result
    registry_keys = [p["registry_key"] for p in result["papers"]]
    assert len(registry_keys) == len(set(registry_keys)), (
        "each paper must appear once (grouped by registry_key), not once per section hit"
    )
    assert key1 in registry_keys and key2 in registry_keys

    by_key = {p["registry_key"]: p for p in result["papers"]}
    assert by_key[key1]["score"] == 1.0, "paper1's best section (identical vector) should score 1.0"
    assert by_key[key2]["score"] == 0.0, "paper2's only section (orthogonal) should score 0.0"
    assert result["papers"][0]["registry_key"] == key1, "paper ranked by BEST section score (D-06)"


def test_search_similar_scoring(tmp_path):
    """D-09: score == round(1 - distance, 4); no relevance cutoff -- all requested papers returned."""
    from scripts import embed  # noqa: PLC0415

    config = _make_embed_config(tmp_path)

    query_vec = [1.0, 0.0]
    identical_vec = [1.0, 0.0]   # cosine distance 0 -> score 1.0
    orthogonal_vec = [0.0, 1.0]  # cosine distance 1 -> score 0.0
    opposite_vec = [-1.0, 0.0]   # cosine distance 2 -> score -1.0

    paper_a = _make_paperjson(
        sections=[{"heading": "Abstract", "body": "Perfectly relevant.", "fill_failed": False}],
        title="Paper A", doi="10.3333/a", year=2020,
    )
    paper_b = _make_paperjson(
        sections=[{"heading": "Abstract", "body": "Unrelated content.", "fill_failed": False}],
        title="Paper B", doi="10.4444/b", year=2021,
    )
    paper_c = _make_paperjson(
        sections=[{"heading": "Abstract", "body": "Opposite of relevant.", "fill_failed": False}],
        title="Paper C", doi="10.5555/c", year=2022,
    )

    with mock.patch(
        "scripts.embed._ollama_embed_call",
        side_effect=[[identical_vec], [orthogonal_vec], [opposite_vec], [query_vec]],
    ), mock.patch("scripts.embed._unload_model"):
        embed.run_embed(paper_a, config)
        embed.run_embed(paper_b, config)
        embed.run_embed(paper_c, config)
        result = embed._search("relevant query", 3, config)

    scores = {p["title"]: p["score"] for p in result["papers"]}
    assert scores["Paper A"] == 1.0
    assert scores["Paper B"] == 0.0
    assert scores["Paper C"] == -1.0, "no relevance cutoff -- even a negative-score paper must be returned"
    assert len(result["papers"]) == 3, "all 3 requested papers returned regardless of score (D-09 no cutoff)"


# ---------------------------------------------------------------------------
# EMBED-03: stub sweep + stub-upgrade delete (D-14, D-15, D-16)
# ---------------------------------------------------------------------------

def test_embed_stub_sweep(tmp_path):
    """EMBED-03/D-14/D-15: a stub in vault/Stubs/ embeds with status=='stub' from title+raw citation."""
    from scripts import embed  # noqa: PLC0415

    config = _make_embed_config(tmp_path)
    _write_stub(
        tmp_path, title="Deep Learning", stub_key="10.9999/dl",
        raw="LeCun et al., Nature 2015", filename="Deep Learning.md",
    )

    # An unrelated paper's run_embed call triggers the stub sweep as a side effect.
    paperjson = _make_paperjson(title="Unrelated Paper", doi="10.8888/unrelated", year=2018)

    with mock.patch("scripts.embed._ollama_embed_call", side_effect=_auto_embed), \
         mock.patch("scripts.embed._unload_model"):
        embed.run_embed(paperjson, config)

    collection = _get_collection(config["chroma_db_path"])
    got = collection.get(where={"registry_key": "10.9999/dl"}, include=["metadatas", "documents"])

    assert got["ids"], "stub should have been swept and embedded"
    assert got["metadatas"][0]["status"] == "stub"
    doc = got["documents"][0]
    assert "Deep Learning" in doc, "stub document must contain the stub title (D-15)"
    assert "LeCun et al." in doc, "stub document must contain the raw citation text (D-15)"


# ---------------------------------------------------------------------------
# EMBED-01: oversize-section split into labeled parts (D-02, Plan 05 Task 1)
# ---------------------------------------------------------------------------

def test_estimate_tokens_and_split_whole_section_unchanged():
    """A body at or below max_tokens returns unchanged: [(heading, 0, body)]."""
    from scripts import embed  # noqa: PLC0415

    body = "A short section body."
    assert embed._estimate_tokens(body) == max(1, len(body) // 4)

    parts = embed._split_section("Methods", body, max_tokens=2000)
    assert parts == [("Methods", 0, body)]


def test_split_section_oversize_produces_labeled_parts():
    """A body over max_tokens splits on paragraph boundaries into labeled (i/n) parts."""
    from scripts import embed  # noqa: PLC0415

    # Each paragraph ~40 chars (~10 tokens); max_tokens=15 forces a split after
    # every paragraph, giving 5 parts from 5 paragraphs.
    paragraphs = [f"Paragraph number {i} of the body text." for i in range(5)]
    body = "\n".join(paragraphs)
    assert embed._estimate_tokens(body) > 15

    parts = embed._split_section("Methods", body, max_tokens=15)
    assert len(parts) >= 2, "an oversize body must split into 2+ parts"

    n = len(parts)
    for i, (labeled_heading, part_index, text) in enumerate(parts, start=1):
        assert labeled_heading == f"Methods ({i}/{n})"
        assert part_index == i
        assert text.strip(), "every emitted part must be non-empty"
        assert embed._estimate_tokens(text) <= 15, "no part may exceed max_tokens"

    # No paragraph text lost, no mid-sentence cut: every paragraph appears
    # intact in exactly one part.
    rejoined = "\n".join(text for _, _, text in parts)
    for para in paragraphs:
        assert para in rejoined


def test_embed_sections_split_oversize_section_into_labeled_parts(tmp_path):
    """Integration: run_embed splits an oversized section into Chroma entries with
    ids ending ::1..::n and metadata headings matching the (i/n) label; a
    <=threshold section still yields a single ::0 unlabeled-heading entry."""
    from scripts import embed  # noqa: PLC0415
    from scripts.ingest import _registry_key  # noqa: PLC0415

    config = _make_embed_config(tmp_path, extra={"embed_section_max_tokens": 15})
    paragraphs = [f"Paragraph number {i} of the body text." for i in range(5)]
    oversize_body = "\n".join(paragraphs)
    sections = [
        {"heading": "Abstract", "body": "Short abstract text.", "fill_failed": False},
        {"heading": "Methods", "body": oversize_body, "fill_failed": False},
    ]
    paperjson = _make_paperjson(sections=sections, title="Split Paper", doi="10.1234/split", year=2022)
    registry_key = _registry_key(paperjson["extraction"]["metadata"])

    with mock.patch("scripts.embed._ollama_embed_call", side_effect=_auto_embed), \
         mock.patch("scripts.embed._unload_model"):
        embed.run_embed(paperjson, config)

    collection = _get_collection(config["chroma_db_path"])
    got = collection.get(where={"registry_key": registry_key}, include=["metadatas"])
    ids_by_meta = dict(zip(got["ids"], got["metadatas"]))

    assert f"{registry_key}::0-abstract::0" in ids_by_meta, "<=threshold section stays a single ::0 entry"
    abstract_meta = ids_by_meta[f"{registry_key}::0-abstract::0"]
    assert abstract_meta["heading"] == "Abstract", "whole-section heading stays unlabeled"
    assert abstract_meta["part"] == 0

    methods_ids = sorted(
        [i for i in got["ids"] if i.startswith(f"{registry_key}::1-methods::")],
        key=lambda s: int(s.rsplit("::", 1)[1]),
    )
    assert len(methods_ids) >= 2, "oversized Methods section must split into 2+ parts"
    assert methods_ids[0] == f"{registry_key}::1-methods::1"
    n = len(methods_ids)
    for i, entry_id in enumerate(methods_ids, start=1):
        meta = ids_by_meta[entry_id]
        assert meta["heading"] == f"Methods ({i}/{n})"
        assert meta["part"] == i


def test_stub_upgrade_deletes_entry(tmp_path):
    """D-16: after an upgrade embed pass, collection.get() for the OLD stub key + status='stub' is empty."""
    from scripts import embed  # noqa: PLC0415

    config = _make_embed_config(tmp_path)
    stub_key = "10.6666/attn"
    _write_stub(
        tmp_path, title="Attention Is All You Need", stub_key=stub_key,
        raw="Vaswani et al., NeurIPS 2017", filename="Attention Is All You Need.md",
    )

    # Embed the stub first (sweep picks it up on an unrelated paper's run_embed).
    unrelated = _make_paperjson(title="Unrelated Paper", doi="10.7777/unrelated", year=2018)
    with mock.patch("scripts.embed._ollama_embed_call", side_effect=_auto_embed), \
         mock.patch("scripts.embed._unload_model"):
        embed.run_embed(unrelated, config)

    collection = _get_collection(config["chroma_db_path"])
    before = collection.get(where={"$and": [{"registry_key": stub_key}, {"status": "stub"}]})
    assert before["ids"], "precondition: stub must be embedded before the upgrade pass"

    # The full paper matching that stub's DOI is embedded next -- must upgrade-delete the stub.
    # (registry_key of the full paper == stub_key here on purpose: verifies the delete is
    # status-scoped and never clobbers the freshly-written paper sections, per 05-03 key_links.)
    full_paper = _make_paperjson(title="Attention Is All You Need", doi=stub_key, year=2017)
    with mock.patch("scripts.embed._ollama_embed_call", side_effect=_auto_embed), \
         mock.patch("scripts.embed._unload_model"):
        embed.run_embed(full_paper, config)

    after_stub = collection.get(where={"$and": [{"registry_key": stub_key}, {"status": "stub"}]})
    assert not after_stub["ids"], "stub entry must be deleted after the matching paper is embedded (D-16)"

    after_paper = collection.get(where={"$and": [{"registry_key": stub_key}, {"status": "paper"}]})
    assert after_paper["ids"], "the paper's own sections must remain after the upgrade-delete"


def test_upgrade_delete_titlehash_stub_after_unlink(tmp_path):
    """CR-02 pipeline-ordering regression: biblio.run_biblio's real Step 12c/9b
    unlinks the stub .md file BEFORE embed's tail hook (Step 12d/9c) runs, so the
    stub-title-index lookup a single-key resolution relies on MISSES. The stub's
    Chroma entry is keyed by a sha256 title-hash (the key a doiless citer would
    have assigned it) -- the old single-key _upgrade_delete missed this stub
    forever. The multi-candidate delete must still find and remove it."""
    from scripts import embed  # noqa: PLC0415
    from scripts import biblio  # noqa: PLC0415

    config = _make_embed_config(tmp_path)
    title = "Attention Is All You Need"
    # Derive the exact title-hash stub_key a doiless citer would have assigned
    # (a title-only dict has no "doi" field to consult, so _match_key falls
    # straight to the title-hash chain).
    titlehash_key = biblio._match_key({"title": title})
    assert titlehash_key.startswith("sha256:"), "precondition: a genuine title-hash key, not a DOI"

    stub_path = _write_stub(
        tmp_path, title=title, stub_key=titlehash_key,
        raw="Vaswani et al., NeurIPS 2017", filename="Attention Is All You Need.md",
    )

    # Embed the stub first (sweep picks it up on an unrelated paper's run_embed).
    unrelated = _make_paperjson(title="Unrelated Paper", doi="10.7777/unrelated", year=2018)
    with mock.patch("scripts.embed._ollama_embed_call", side_effect=_auto_embed), \
         mock.patch("scripts.embed._unload_model"):
        embed.run_embed(unrelated, config)

    collection = _get_collection(config["chroma_db_path"])
    before = collection.get(where={"$and": [{"registry_key": titlehash_key}, {"status": "stub"}]})
    assert before["ids"], "precondition: title-hash stub must be embedded before the unlink+upgrade pass"

    # Simulate biblio's Step 12c/9b unlink -- happens BEFORE embed's tail hook
    # in the real pipeline -- so the stub-title-index will now MISS.
    stub_path.unlink()

    # The full paper (matching title, now WITH a DOI) is embedded next.
    doi = "10.6666/attn"
    full_paper = _make_paperjson(title=title, doi=doi, year=2017)
    with mock.patch("scripts.embed._ollama_embed_call", side_effect=_auto_embed), \
         mock.patch("scripts.embed._unload_model"):
        embed.run_embed(full_paper, config)

    after_stub = collection.get(where={"$and": [{"registry_key": titlehash_key}, {"status": "stub"}]})
    assert not after_stub["ids"], (
        "the stale title-hash stub entry must be deleted even though its file was "
        "unlinked before the upgrade pass ran (CR-02)"
    )

    after_paper = collection.get(where={"$and": [{"registry_key": doi}, {"status": "paper"}]})
    assert after_paper["ids"], "the paper's own sections must be present after the upgrade"


def test_self_stub_not_reembedded_with_paper(tmp_path):
    """WR-02: a self-stub still on disk (never pre-embedded, e.g. CLI embed,
    --all backfill, or a failed biblio unlink) must be filtered out of the
    pending-stub sweep before upsert, so it's never re-embedded alongside the
    paper's own sections in the same run_embed pass."""
    from scripts import embed  # noqa: PLC0415
    from scripts import biblio  # noqa: PLC0415

    config = _make_embed_config(tmp_path)
    title = "Attention Is All You Need"
    doi = "10.6666/attn"

    paper = _make_paperjson(title=title, doi=doi, year=2017)
    registry_key = biblio._match_key(paper["extraction"]["metadata"])

    # Self-stub still on disk with stub_key == the paper's own DOI-derived
    # registry_key -- written but NEVER pre-embedded (unlike the CR-02 test
    # above, which embeds the stub first). This is the WR-02 scenario: a
    # pending self-stub swept in the SAME run_embed pass as its own paper.
    _write_stub(
        tmp_path, title=title, stub_key=registry_key,
        raw="Vaswani et al., NeurIPS 2017", filename="Attention Is All You Need.md",
    )

    with mock.patch("scripts.embed._ollama_embed_call", side_effect=_auto_embed), \
         mock.patch("scripts.embed._unload_model"):
        embed.run_embed(paper, config)

    collection = _get_collection(config["chroma_db_path"])

    after_paper = collection.get(where={"$and": [{"registry_key": registry_key}, {"status": "paper"}]})
    assert after_paper["ids"], "the paper's own sections must be embedded"

    after_stub = collection.get(where={"$and": [{"registry_key": registry_key}, {"status": "stub"}]})
    assert not after_stub["ids"], (
        "the pending self-stub must be filtered out of the sweep, not upserted "
        "alongside the paper's own sections (WR-02)"
    )


# ---------------------------------------------------------------------------
# EMBED-01/EMBED-03: embed.py --all backfill (D-12, Plan 05 Task 2)
# ---------------------------------------------------------------------------

def test_backfill_all_filters_registry_embeds_cache_and_stubs_idempotently(tmp_path):
    """D-12: _backfill_all filters registry entries by projects[] membership, embeds
    each entry's paperjson_path cache, embeds extra cache-dir files not in the
    registry, embeds pending stubs, skips a desynced missing-cache-file entry, and
    is idempotent (a second call reports zero newly embedded, zero /api/embed calls)."""
    from scripts import embed  # noqa: PLC0415
    from scripts.ingest import _registry_key  # noqa: PLC0415

    config = _make_embed_config(tmp_path, extra={"project_name": "proj-a"})
    cache_dir = pathlib.Path(config["paperjson_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    paper_a = _make_paperjson(title="Paper A", doi="10.1111/paperA", year=2020)
    paper_a_path = cache_dir / "paperA.json"
    paper_a_path.write_text(json.dumps(paper_a), encoding="utf-8")

    extra_paper = _make_paperjson(title="Extra Paper", doi="10.9999/extra", year=2021)
    extra_path = cache_dir / "extra.json"
    extra_path.write_text(json.dumps(extra_paper), encoding="utf-8")

    registry = {
        "10.1111/paperA": {"projects": ["proj-a"], "paperjson_path": str(paper_a_path)},
        "10.2222/paperB": {
            "projects": ["other-proj"],
            "paperjson_path": str(cache_dir / "paperB_never_created.json"),
        },
        "10.3333/paperC": {"projects": ["proj-a"], "paperjson_path": str(cache_dir / "missing.json")},
    }
    pathlib.Path(config["registry_path"]).write_text(json.dumps(registry), encoding="utf-8")

    _write_stub(
        tmp_path, title="Deep Learning", stub_key="10.9999/dl",
        raw="LeCun et al., Nature 2015", filename="Deep Learning.md",
    )

    key_a = _registry_key(paper_a["extraction"]["metadata"])
    key_extra = _registry_key(extra_paper["extraction"]["metadata"])

    with mock.patch("scripts.embed._ollama_embed_call", side_effect=_auto_embed) as mock_embed, \
         mock.patch("scripts.embed._unload_model"):
        counts = embed._backfill_all(config)

    assert mock_embed.called, "first backfill must call /api/embed"
    assert counts["papers"] == 2, "paperA (registry) + extra.json (cache-dir scan) both newly embedded"
    assert counts["stubs"] == 1, "the one stub in vault/Stubs/ is newly embedded"
    assert counts["skipped"] == 1, "paperC's missing cache file is a registry/cache desync skip"

    collection = _get_collection(config["chroma_db_path"])
    assert collection.get(where={"registry_key": key_a})["ids"], "paperA sections embedded"
    assert collection.get(where={"registry_key": key_extra})["ids"], "extra.json sections embedded"
    assert collection.get(where={"registry_key": "10.2222/paperB"})["ids"] == [], (
        "paperB excluded -- not in this project"
    )
    assert collection.get(
        where={"$and": [{"registry_key": "10.9999/dl"}, {"status": "stub"}]}
    )["ids"], "stub embedded"

    with mock.patch("scripts.embed._ollama_embed_call", side_effect=_auto_embed) as mock_embed2, \
         mock.patch("scripts.embed._unload_model"):
        counts2 = embed._backfill_all(config)

    mock_embed2.assert_not_called()  # zero Ollama calls on a fully-embedded second pass
    assert counts2 == {"papers": 0, "stubs": 0, "skipped": 1}, (
        "second run reports zero newly embedded; paperC's desync skip persists"
    )


def test_backfill_preserves_orphan_entries(tmp_path):
    """IN-04 (replaces the brittle test_backfill_all_no_orphan_delete_calls source
    grep): behaviorally proves backfill performs no orphan-reconciliation delete --
    a Chroma entry whose registry_key is absent from the registry, cache dir, and
    Stubs/ survives a backfill pass untouched."""
    from scripts import embed  # noqa: PLC0415

    config = _make_embed_config(tmp_path, extra={"project_name": "proj-a"})
    pathlib.Path(config["registry_path"]).write_text(json.dumps({}), encoding="utf-8")

    collection = _get_collection(config["chroma_db_path"])
    orphan_key = "10.0000/orphan-not-in-registry-cache-or-stubs"
    collection.upsert(
        ids=[f"{orphan_key}::title::0"],
        embeddings=[_fixed_vector(0)],
        documents=["Orphan Paper"],
        metadatas=[{
            "registry_key": orphan_key,
            "title": "Orphan Paper",
            "heading": "(stub)",
            "part": 0,
            "doi": orphan_key,
            "year": 2000,
            "status": "paper",
            "vault_note": "Papers/Orphan Paper.md",
        }],
    )

    with mock.patch("scripts.embed._ollama_embed_call", side_effect=_auto_embed), \
         mock.patch("scripts.embed._unload_model"):
        embed._backfill_all(config)

    after = collection.get(where={"registry_key": orphan_key})
    assert after["ids"], "an orphaned Chroma entry must survive a backfill pass untouched"


def test_backfill_all_filters_by_project_name_and_projects_field():
    """Grep confirmation: _backfill_all's source references both projects[] and
    config's project_name to build the registry membership filter (D-12)."""
    src = (pathlib.Path(__file__).resolve().parent.parent / "scripts" / "embed.py").read_text(encoding="utf-8")
    backfill_src = src.split("def _backfill_all(")[1]
    assert "project_name" in backfill_src
    assert "projects" in backfill_src


def test_cli_all_flag_no_crash_on_empty_registry(tmp_path, monkeypatch, capsys):
    """--all runs without crashing on an empty/nonexistent registry and prints a
    parseable JSON summary to stdout (--all and --query are separate CLI modes)."""
    from scripts import embed  # noqa: PLC0415

    config = _make_embed_config(tmp_path, extra={
        "project_name": "proj-a",
        "registry_path": str(tmp_path / "does_not_exist_registry.json"),
    })
    monkeypatch.setattr(embed, "_load_config", lambda: config)
    monkeypatch.setattr(sys, "argv", ["embed.py", "--all"])

    with mock.patch("scripts.embed._ollama_embed_call", side_effect=_auto_embed), \
         mock.patch("scripts.embed._unload_model"):
        embed.main()

    out = capsys.readouterr().out
    parsed = json.loads(out.strip().splitlines()[-1])
    assert set(parsed) == {"papers", "stubs", "skipped"}
    assert parsed == {"papers": 0, "stubs": 0, "skipped": 0}


# ---------------------------------------------------------------------------
# Live-subprocess regression tests (05-06 gap closure): embed.py invoked as a
# real top-level process via subprocess.run([sys.executable, ...]) -- NOT an
# in-process embed.main() call -- so pytest's own sys.path augmentation can
# never mask a broken standalone-CLI sys.path bootstrap again. Every config
# written here points chroma_db_path/vault_path/registry_path/
# paperjson_cache_dir under tmp_path (conftest._guard_real_chroma_db /
# _guard_real_paperjson_cache safety nets). Neither test requires a live
# Ollama -- the sys.path failure this regression-guards manifests before any
# /api/embed call is ever made.
# ---------------------------------------------------------------------------

def _write_subprocess_config(tmp_path):
    """Write a tmp config.json for a live embed.py subprocess run, with every
    path anchored under tmp_path so the real repo-root stores are never touched."""
    (tmp_path / "Papers").mkdir(exist_ok=True)
    (tmp_path / "Stubs").mkdir(exist_ok=True)
    cfg = {
        "chroma_db_path": str(tmp_path / "chroma_db"),
        "vault_path": str(tmp_path),
        "registry_path": str(tmp_path / "does_not_exist_registry.json"),
        "embed_model": "nomic-embed-text",
        "embed_timeout": 5,
        "embed_batch_size": 16,
        "embed_section_max_tokens": 2000,
        "paperjson_cache_dir": str(tmp_path / "does_not_exist_paperjson_cache"),
    }
    cfg_path = tmp_path / "subprocess_config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg_path


def test_cli_embed_file_subprocess_resolves_imports(tmp_path):
    """The FAILED 05-VERIFICATION must-have, restated for live-subprocess
    semantics: `python scripts/embed.py <cache_file>` run as a real subprocess
    resolves its scripts.ingest / scripts.note / scripts.biblio imports and
    reaches the embed path -- combined stdout+stderr must never contain the
    phrase 'No module named'. (With Ollama unreachable the run legitimately
    ends in an "[embed warning: ...]" Ollama-error/timeout line -- that is
    expected and NOT what this test guards; only module-resolution failure
    is asserted against.)"""
    cfg_path = _write_subprocess_config(tmp_path)
    paperjson = _make_paperjson()
    cache_file = tmp_path / "paper.json"
    cache_file.write_text(json.dumps(paperjson), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_EMBED_PY), "--config", str(cfg_path), str(cache_file)],
        cwd=str(_REPO_ROOT_FOR_SUBPROCESS),
        capture_output=True,
        text=True,
        timeout=60,
    )

    combined = result.stdout + result.stderr
    assert "No module named" not in combined, (
        f"embed.py subprocess failed to resolve scripts.* imports:\n{combined}"
    )


def test_cli_all_subprocess_no_crash_on_empty_registry(tmp_path):
    """The second FAILED 05-VERIFICATION must-have, restated for
    live-subprocess semantics: `python scripts/embed.py --all` run as a real
    subprocess against an empty/nonexistent registry exits 0 and prints a
    parseable JSON summary whose keys are exactly papers/stubs/skipped -- no
    ModuleNotFoundError traceback, no exit code 1."""
    cfg_path = _write_subprocess_config(tmp_path)

    result = subprocess.run(
        [sys.executable, str(_EMBED_PY), "--config", str(cfg_path), "--all"],
        cwd=str(_REPO_ROOT_FOR_SUBPROCESS),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"embed.py --all subprocess exited nonzero:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    assert lines, f"expected at least one non-empty stdout line, got: {result.stdout!r}"
    parsed = json.loads(lines[-1])
    assert set(parsed) == {"papers", "stubs", "skipped"}
    assert parsed == {"papers": 0, "stubs": 0, "skipped": 0}
