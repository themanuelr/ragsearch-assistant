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

import math
from unittest import mock


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
        f"{registry_key}::abstract::0",
        f"{registry_key}::methods::0",
    }, f"expected deterministic {{registry_key}}::{{slug}}::{{part}} ids, got {got['ids']}"

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
