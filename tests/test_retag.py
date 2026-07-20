"""
Unit tests for the non-destructive additive retag (D-06..D-09, Phase 08.1 Plan 02).

This is the RED contract for 08.1-02-PLAN.md -- every test that touches
``scripts.retag`` or ``note._patch_frontmatter_fields`` imports those symbols
LAZILY inside the test body (mirrors ``tests/test_link.py``'s own precedent
for a not-yet-existing module) so collection succeeds before ``scripts/retag.py``
exists. ``scripts.note`` and ``scripts.link`` already exist and are imported
at module level.

Tests cover:
  - note._patch_frontmatter_fields: scalar insert at canonical position, body
    byte-exact; tags: block full replacement; date_ingested/status preserved
    byte-identical (Pitfall 4); missing-fence ValueError.
  - retag.run_retag: additive-only merge (D-06), cache/note tag agreement
    (D-08), the frontmatter write is the sole vault write via
    obsidian_cli.create_note(..., overwrite=True), link.run_link is chained
    after the write and its failure is non-fatal (D-07).
  - retag.main(): --stem/--context CLI entrypoint prints the note path (D-09).

Run with:  python -m pytest tests/test_retag.py -x
"""

import json
import pathlib
import sys

import pytest

from scripts import link
from scripts import note
from scripts.note import _sanitize_filename


# ---------------------------------------------------------------------------
# _patch_frontmatter_fields fixtures
# ---------------------------------------------------------------------------

_BODY_SUFFIX = (
    "---\n"
    "\n"
    "# Sample Paper\n"
    "\n"
    "## Summary\n"
    "\n"
    "Body text here.\n"
)

_SAMPLE_NOTE = (
    '---\n'
    'title: "Sample Paper"\n'
    'tags:\n'
    '  - "old-tag"\n'
    'status: ingested\n'
    'date_ingested: 2026-07-01\n'
) + _BODY_SUFFIX


# ---------------------------------------------------------------------------
# note._patch_frontmatter_fields (Task 2 -- RED until implemented)
# ---------------------------------------------------------------------------

def test_patch_frontmatter_fields_inserts_scalar_updates_at_canonical_position():
    """scalar_updates={"year":2021,"journal":"Trends in Chemistry"} on a note
    missing those inserts them at the canonical position (after title, before
    tags) and leaves the body byte-exact."""
    patched = note._patch_frontmatter_fields(
        _SAMPLE_NOTE,
        scalar_updates={"year": 2021, "journal": "Trends in Chemistry"},
    )

    lines = patched.split("\n")
    assert "year: 2021" in lines
    assert 'journal: "Trends in Chemistry"' in lines

    title_idx = lines.index('title: "Sample Paper"')
    year_idx = lines.index("year: 2021")
    journal_idx = lines.index('journal: "Trends in Chemistry"')
    tags_idx = lines.index("tags:")
    assert title_idx < year_idx < journal_idx < tags_idx, (
        f"Expected canonical order title < year < journal < tags, got:\n{patched}"
    )
    assert patched.endswith(_BODY_SUFFIX), "Expected the body to be byte-exact"


def test_patch_frontmatter_fields_replaces_tags_block():
    """tag_replacement=["a","b","c"] replaces the entire tags block with
    '  - "slug"' lines and leaves other frontmatter lines untouched."""
    patched = note._patch_frontmatter_fields(
        _SAMPLE_NOTE, tag_replacement=["a", "b", "c"]
    )

    assert '  - "old-tag"' not in patched
    for slug in ("a", "b", "c"):
        assert f'  - "{slug}"' in patched
    assert 'title: "Sample Paper"' in patched
    assert "status: ingested" in patched
    assert "date_ingested: 2026-07-01" in patched
    assert patched.endswith(_BODY_SUFFIX), "Expected the body to be byte-exact"


def test_patch_frontmatter_fields_preserves_date_ingested_and_status():
    """date_ingested and status lines are byte-identical before/after any
    patch (Pitfall 4 -- _render_frontmatter is NOT reused)."""
    for patched in (
        note._patch_frontmatter_fields(
            _SAMPLE_NOTE,
            scalar_updates={"year": 2021, "journal": "Trends in Chemistry"},
        ),
        note._patch_frontmatter_fields(_SAMPLE_NOTE, tag_replacement=["x"]),
    ):
        patched_lines = patched.split("\n")
        assert "status: ingested" in patched_lines
        assert "date_ingested: 2026-07-01" in patched_lines


def test_patch_frontmatter_fields_no_fence_raises_value_error():
    """Content with no '---' fence raises ValueError."""
    with pytest.raises(ValueError):
        note._patch_frontmatter_fields("# No frontmatter here\n\nJust body text.\n")


# ---------------------------------------------------------------------------
# retag.run_retag / retag.main fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_vault(tmp_path):
    (tmp_path / "Papers").mkdir()
    (tmp_path / "Topics").mkdir()
    return tmp_path


@pytest.fixture
def config(tmp_vault):
    return {
        "vault_path": str(tmp_vault),
        "registry_path": str(tmp_vault / "registry.json"),
        "paperjson_cache_dir": str(tmp_vault / ".paperjson_cache"),
    }


def _write_paper_note(vault_path, title, tags=None):
    """Seed a Papers/ note with a frontmatter tags: block, mirroring
    test_link.py's own _write_paper_note idiom."""
    filename = _sanitize_filename(title)
    lines = ["---", f'title: "{title}"']
    if tags:
        lines.append("tags:")
        for tag in tags:
            lines.append(f'  - "{tag}"')
    lines.append("status: ingested")
    lines.append("date_ingested: 2026-07-11")
    lines.append("---")
    lines.append("")
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## My Notes")
    lines.append("")
    note_path = pathlib.Path(vault_path) / "Papers" / f"{filename}.md"
    note_path.write_text("\n".join(lines), encoding="utf-8")
    return filename


def _make_retag_paperjson(title):
    """Minimal PaperJSON v2 shape carrying only what run_retag needs: a title
    to locate the live note and sections for _assemble_paper_text."""
    return {
        "extraction": {
            "metadata": {
                "title": title, "doi": None, "authors": None,
                "year": None, "journal": None, "arxiv_id": None,
            },
            "sections": [
                {"heading": "Introduction", "body": "This paper studies methods."},
            ],
        },
        "analysis": {"topics": []},
        "provenance": {},
    }


def _mock_analysis_call(topics):
    """Return a fake note._analysis_call replacement yielding an AnalysisTopics
    with the given topics -- matches note._analysis_call's real signature
    (prompt, system, model_cls, paper_text)."""
    def _fake(prompt, system, model_cls, paper_text):
        return note.AnalysisTopics(topics=list(topics))
    return _fake


# ---------------------------------------------------------------------------
# retag.run_retag -- additive merge (D-06), cache/note agreement (D-08),
# chokepoint write, link chaining (D-07)
# ---------------------------------------------------------------------------

def test_run_retag_additive_merge_never_drops_existing_tag(tmp_vault, config, monkeypatch):
    from scripts import retag  # noqa: PLC0415

    _write_paper_note(tmp_vault, "Retag Paper One", tags=["existing-tag"])
    pj = _make_retag_paperjson("Retag Paper One")

    monkeypatch.setattr(note, "_analysis_call", _mock_analysis_call(["existing-tag", "new-tag"]))
    monkeypatch.setattr(note, "_canonicalize_tags", lambda topics, cfg: list(topics))
    monkeypatch.setattr(link, "run_link", lambda paperjson, cfg: "ok")

    result = retag.run_retag(pj, config)

    assert result == "Papers/Retag Paper One.md"
    note_content = (tmp_vault / "Papers" / "Retag Paper One.md").read_text(encoding="utf-8")
    written_tags = link._parse_tags(note_content)
    assert "existing-tag" in written_tags, "additive-only: existing tag must survive retag (D-06)"
    assert "new-tag" in written_tags


def test_run_retag_cache_and_note_tags_agree(tmp_vault, config, monkeypatch):
    from scripts import retag  # noqa: PLC0415

    _write_paper_note(tmp_vault, "Retag Paper Two", tags=["existing-tag"])
    pj = _make_retag_paperjson("Retag Paper Two")

    monkeypatch.setattr(note, "_analysis_call", _mock_analysis_call(["new-tag"]))
    monkeypatch.setattr(note, "_canonicalize_tags", lambda topics, cfg: list(topics))
    monkeypatch.setattr(link, "run_link", lambda paperjson, cfg: "ok")

    retag.run_retag(pj, config)

    note_content = (tmp_vault / "Papers" / "Retag Paper Two.md").read_text(encoding="utf-8")
    written_tags = link._parse_tags(note_content)
    assert sorted(pj["analysis"]["topics"]) == sorted(written_tags), (
        "cache analysis.topics and note frontmatter tags must agree (D-08)"
    )


def test_run_retag_write_goes_through_create_note_overwrite_true(tmp_vault, config, monkeypatch):
    from scripts import retag  # noqa: PLC0415

    _write_paper_note(tmp_vault, "Retag Paper Three", tags=["existing-tag"])
    pj = _make_retag_paperjson("Retag Paper Three")

    monkeypatch.setattr(note, "_analysis_call", _mock_analysis_call(["new-tag"]))
    monkeypatch.setattr(note, "_canonicalize_tags", lambda topics, cfg: list(topics))
    monkeypatch.setattr(link, "run_link", lambda paperjson, cfg: "ok")

    calls = []

    def _fake_create_note(path, content, cfg, overwrite=False):
        calls.append({"path": path, "content": content, "overwrite": overwrite})
        return str(tmp_vault / path)

    monkeypatch.setattr(retag, "create_note", _fake_create_note)

    retag.run_retag(pj, config)

    assert len(calls) == 1, "expected the frontmatter patch to be the ONLY vault write"
    assert calls[0]["path"] == "Papers/Retag Paper Three.md"
    assert calls[0]["overwrite"] is True
    assert "new-tag" in calls[0]["content"]


def test_run_retag_calls_link_run_link_after_write(tmp_vault, config, monkeypatch):
    from scripts import retag  # noqa: PLC0415

    _write_paper_note(tmp_vault, "Retag Paper Four", tags=["existing-tag"])
    pj = _make_retag_paperjson("Retag Paper Four")

    monkeypatch.setattr(note, "_analysis_call", _mock_analysis_call(["new-tag"]))
    monkeypatch.setattr(note, "_canonicalize_tags", lambda topics, cfg: list(topics))

    events = []
    real_create_note = retag.create_note

    def _tracking_create_note(path, content, cfg, overwrite=False):
        events.append("write")
        return real_create_note(path, content, cfg, overwrite=overwrite)

    def _tracking_run_link(paperjson, cfg):
        events.append("link")
        return "ok"

    monkeypatch.setattr(retag, "create_note", _tracking_create_note)
    monkeypatch.setattr(link, "run_link", _tracking_run_link)

    retag.run_retag(pj, config)

    assert events == ["write", "link"], f"expected the write before the link chain, got {events}"


def test_run_retag_link_exception_is_swallowed_non_fatal(tmp_vault, config, monkeypatch):
    from scripts import retag  # noqa: PLC0415

    _write_paper_note(tmp_vault, "Retag Paper Five", tags=["existing-tag"])
    pj = _make_retag_paperjson("Retag Paper Five")

    monkeypatch.setattr(note, "_analysis_call", _mock_analysis_call(["new-tag"]))
    monkeypatch.setattr(note, "_canonicalize_tags", lambda topics, cfg: list(topics))

    def _raising_run_link(paperjson, cfg):
        raise RuntimeError("boom")

    monkeypatch.setattr(link, "run_link", _raising_run_link)

    result = retag.run_retag(pj, config)

    assert result == "Papers/Retag Paper Five.md", (
        "retag must still return success when link.run_link raises (D-07 non-fatal)"
    )
    note_content = (tmp_vault / "Papers" / "Retag Paper Five.md").read_text(encoding="utf-8")
    assert "new-tag" in note_content, "the frontmatter write must have already happened before the link failure"


# ---------------------------------------------------------------------------
# retag.main() CLI (D-09)
# ---------------------------------------------------------------------------

def test_cli_stem_and_context_prints_note_path(tmp_vault, config, monkeypatch, capsys):
    from scripts import retag  # noqa: PLC0415

    _write_paper_note(tmp_vault, "Retag CLI Paper", tags=["existing-tag"])
    pj = _make_retag_paperjson("Retag CLI Paper")

    cache_dir = tmp_vault / ".paperjson_cache"
    cache_dir.mkdir()
    (cache_dir / "retag-cli-paper.json").write_text(json.dumps(pj), encoding="utf-8")

    cli_config = dict(config)
    cli_config["paperjson_cache_dir"] = str(cache_dir)

    monkeypatch.setattr(retag, "_load_config", lambda: cli_config)
    monkeypatch.setattr(note, "_analysis_call", _mock_analysis_call(["methods-tag"]))
    monkeypatch.setattr(note, "_canonicalize_tags", lambda topics, cfg: list(topics))
    monkeypatch.setattr(link, "run_link", lambda paperjson, cfg: "ok")
    monkeypatch.setattr(
        sys, "argv",
        ["retag.py", "--stem", "retag-cli-paper", "--context", "methods-focused tagging"],
    )

    retag.main()

    out = capsys.readouterr().out
    assert "Papers/Retag CLI Paper.md" in out
