"""
Unit tests for the topic-graph tail stage (GRAPH-01 through GRAPH-03).

This is the Wave-0 RED contract for Phase 6 (06-01-PLAN.md) -- every test below
names symbols in ``scripts.link`` that do not exist yet (that module is authored
in Plan 02/03). Every test imports ``scripts.link`` LAZILY inside the test body
(mirrors ``tests/test_biblio.py`` line 93's ``from scripts import biblio``
precedent) so collection succeeds before ``scripts/link.py`` exists. Do NOT
``pytest.importorskip("scripts.link")`` -- these tests must be RED, not skipped.

Tests cover:
  - test_topic_note_created_with_member_link (GRAPH-01 / D-01): tagged paper -> Topics/<slug>.md
  - test_managed_block_idempotent (GRAPH-01 / D-04): re-run does not duplicate the member link
  - test_nested_tag_slug_topic_note (GRAPH-01 / Pitfall 2, D-03): "/" flattens to "-" in the filename
  - test_related_topics_injection (GRAPH-02 / D-08/D-09): paper note gains a piped Related Topics link
  - test_no_tags_omits_section (GRAPH-02 / D-10): zero tags -> no Related Topics header at all
  - test_related_topics_idempotent (GRAPH-02 / D-10): re-run yields exactly one Related Topics section
  - test_papers_base_write_if_absent (GRAPH-03 / D-13): _Papers.base is written once, never clobbered
  - test_papers_base_yaml_shape (GRAPH-03 / D-12/D-14): valid YAML, correct columns, no negated filters

Run with:  python -m pytest tests/test_link.py -x
"""

import pathlib

import pytest

from scripts.note import _sanitize_filename


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_vault(tmp_path):
    """Create a minimal vault structure under tmp_path.

    Extends test_biblio.py's tmp_vault fixture (lines 74-79) with a Topics/
    directory, per 06-VALIDATION.md's Wave 0 Requirements.
    """
    (tmp_path / "Papers").mkdir()
    (tmp_path / "Stubs").mkdir()
    (tmp_path / "Topics").mkdir()
    return tmp_path


@pytest.fixture
def config(tmp_vault):
    return {"vault_path": str(tmp_vault), "registry_path": str(tmp_vault / "registry.json")}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_paper_note(vault_path, title, tags=None):
    """Seed a Papers/ note with a frontmatter ``tags:`` block and a trailing
    ``## My Notes`` anchor, mirroring note.py's ``_render_frontmatter()`` /
    ``_render_note()`` shape (double-quoted lower-kebab tag slugs, a
    ``## Limitations`` section immediately before ``## My Notes``) so link.py's
    Related Topics injection has the same real anchor a generated note provides.

    Args:
        vault_path: The vault root (tmp_vault).
        title:      Paper title (also the frontmatter title and filename source).
        tags:       Optional list of already-slugified tag strings (lower-kebab,
                    may contain "/" for nested tags per note.py's _slugify_tag).

    Returns:
        The sanitized filename (no ``.md`` suffix) -- the wikilink target
        identity link.py's managed member-list block is expected to use.
    """
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
    lines.append("## Limitations")
    lines.append("")
    lines.append("> [!warning] Limitation")
    lines.append("> None identified.")
    lines.append("")
    lines.append("## My Notes")
    lines.append("")
    note_path = pathlib.Path(vault_path) / "Papers" / f"{filename}.md"
    note_path.write_text("\n".join(lines), encoding="utf-8")
    return filename


def _make_link_paperjson(title, doi=None):
    """Minimal PaperJSON v2 shape carrying only what run_link needs to locate the
    paper's live note (extraction.metadata.title).

    D-02 makes the LIVE NOTE frontmatter the source of truth for tags, not
    analysis.topics -- so this fixture intentionally leaves analysis.topics
    empty; link.py must read tags from the note file written by
    _write_paper_note, not from this dict.
    """
    return {
        "extraction": {
            "metadata": {
                "title": title,
                "doi": doi,
                "authors": None,
                "year": None,
                "journal": None,
                "arxiv_id": None,
            },
        },
        "analysis": {"topics": []},
        "provenance": {},
    }


# ---------------------------------------------------------------------------
# GRAPH-01: Topic index note creation + member-list idempotency
# ---------------------------------------------------------------------------

def test_topic_note_created_with_member_link(tmp_vault, config):
    """GRAPH-01 / D-01: a paper tagged "attention" creates Topics/attention.md
    containing a wikilink to the paper."""
    from scripts import link  # noqa: PLC0415

    filename = _write_paper_note(tmp_vault, "Attention Is All You Need", tags=["attention"])
    pj = _make_link_paperjson("Attention Is All You Need")

    link.run_link(pj, config)

    topic_path = tmp_vault / "Topics" / "attention.md"
    assert topic_path.exists(), "Expected Topics/attention.md to be created"
    content = topic_path.read_text(encoding="utf-8")
    assert f"[[{filename}]]" in content, (
        f"Expected member wikilink [[{filename}]] in Topics/attention.md, got:\n{content}"
    )


def test_managed_block_idempotent(tmp_vault, config):
    """GRAPH-01 / D-04: calling run_link twice does not duplicate the member
    wikilink in the topic note (managed-block strip-and-reinject)."""
    from scripts import link  # noqa: PLC0415

    filename = _write_paper_note(tmp_vault, "Attention Is All You Need", tags=["attention"])
    pj = _make_link_paperjson("Attention Is All You Need")

    link.run_link(pj, config)
    link.run_link(pj, config)

    content = (tmp_vault / "Topics" / "attention.md").read_text(encoding="utf-8")
    assert content.count(f"[[{filename}]]") == 1, (
        f"Expected exactly one member wikilink after two runs, got {content.count(f'[[{filename}]]')}"
    )


def test_nested_tag_slug_topic_note(tmp_vault, config):
    """GRAPH-01 / Pitfall 2, D-03: a nested-tag slug ("input/output-gating")
    still resolves to a discoverable FLAT topic note -- the filename flattens
    "/" to "-" so a non-recursive Topics/*.md glob finds it."""
    from scripts import link  # noqa: PLC0415

    _write_paper_note(tmp_vault, "IO Gating Paper", tags=["input/output-gating"])
    pj = _make_link_paperjson("IO Gating Paper")

    link.run_link(pj, config)

    # Non-recursive glob -- must NOT require Topics/**/*.md (Pitfall 2 guard).
    flat_names = {p.name for p in (tmp_vault / "Topics").glob("*.md")}
    assert "input-output-gating.md" in flat_names, (
        f"Expected flattened Topics/input-output-gating.md discoverable via a "
        f"non-recursive Topics/*.md glob, got {flat_names}"
    )


# ---------------------------------------------------------------------------
# GRAPH-02: Related Topics injection on the paper note itself
# ---------------------------------------------------------------------------

def test_related_topics_injection(tmp_vault, config):
    """GRAPH-02 / D-08/D-09: after run_link, the paper note contains a
    "## Related Topics" section with a piped wikilink (alias-piped, target =
    the flat topic-note basename)."""
    from scripts import link  # noqa: PLC0415

    _write_paper_note(tmp_vault, "IO Gating Paper 2", tags=["input/output-gating"])
    pj = _make_link_paperjson("IO Gating Paper 2")

    link.run_link(pj, config)

    note_content = (tmp_vault / "Papers" / "IO Gating Paper 2.md").read_text(encoding="utf-8")
    assert "## Related Topics" in note_content, "Expected a Related Topics section to be injected"
    assert "[[input-output-gating|" in note_content, (
        f"Expected a piped wikilink target [[input-output-gating|...]], got:\n{note_content}"
    )


def test_no_tags_omits_section(tmp_vault, config):
    """GRAPH-02 / D-10: a paper note with NO tags: key gets no Related Topics
    header at all (never an empty section)."""
    from scripts import link  # noqa: PLC0415

    _write_paper_note(tmp_vault, "No Tags Paper", tags=None)
    pj = _make_link_paperjson("No Tags Paper")

    link.run_link(pj, config)

    note_content = (tmp_vault / "Papers" / "No Tags Paper.md").read_text(encoding="utf-8")
    assert "## Related Topics" not in note_content, (
        "Expected NO Related Topics header for a zero-tag paper"
    )


def test_related_topics_idempotent(tmp_vault, config):
    """GRAPH-02 / D-10: running link.py twice yields exactly one Related Topics
    section (strip-and-reinject, never a duplicate)."""
    from scripts import link  # noqa: PLC0415

    _write_paper_note(tmp_vault, "Attention Idempotent Paper", tags=["attention"])
    pj = _make_link_paperjson("Attention Idempotent Paper")

    link.run_link(pj, config)
    link.run_link(pj, config)

    note_content = (tmp_vault / "Papers" / "Attention Idempotent Paper.md").read_text(encoding="utf-8")
    assert note_content.count("## Related Topics") == 1, (
        f"Expected exactly one Related Topics section, got {note_content.count('## Related Topics')}"
    )


# ---------------------------------------------------------------------------
# GRAPH-03: _Papers.base -- write-once, correct YAML shape, stub-safe
# ---------------------------------------------------------------------------

def test_papers_base_write_if_absent(tmp_vault, config):
    """GRAPH-03 / D-13: _Papers.base is written once at vault root if absent;
    a second run does not overwrite an existing file (write-once, never
    clobbers user edits)."""
    from scripts import link  # noqa: PLC0415

    link.ensure_papers_base(config)
    base_path = tmp_vault / "_Papers.base"
    assert base_path.exists(), "Expected _Papers.base to be created at vault root"

    sentinel = "\n# SENTINEL-DO-NOT-CLOBBER\n"
    with open(base_path, "a", encoding="utf-8") as f:
        f.write(sentinel)

    link.ensure_papers_base(config)

    content = base_path.read_text(encoding="utf-8")
    assert "SENTINEL-DO-NOT-CLOBBER" in content, (
        "ensure_papers_base must not clobber an existing _Papers.base file"
    )


def test_papers_base_yaml_shape(tmp_vault, config):
    """GRAPH-03 / D-12/D-14: _Papers.base parses as YAML; its scope filter
    references both Papers and Stubs folders; the table view's order: list
    contains the required columns; no negated operator appears anywhere
    (Pitfall 3 stub-safety guard)."""
    from scripts import link  # noqa: PLC0415
    import yaml  # noqa: PLC0415

    link.ensure_papers_base(config)
    base_path = tmp_vault / "_Papers.base"
    raw = base_path.read_text(encoding="utf-8")

    parsed = yaml.safe_load(raw)
    assert isinstance(parsed, dict), "Expected _Papers.base to parse as a YAML mapping"

    filters_str = str(parsed.get("filters", {}))
    assert "Papers" in filters_str, f"Expected the filters scope to reference Papers, got {filters_str!r}"
    assert "Stubs" in filters_str, f"Expected the filters scope to reference Stubs, got {filters_str!r}"

    order = None
    for view in parsed.get("views", []):
        if view.get("type") == "table":
            order = view.get("order")
            break
    assert order is not None, "Expected a table view with an order: column list"
    for col in ("title", "authors", "year", "journal", "status", "tags"):
        assert col in order, f"Expected column {col!r} in the table view's order: list, got {order}"

    # Pitfall 3: a negated filter against an optional property silently excludes
    # stub rows whose property is null/missing -- the static file must never
    # author one.
    for forbidden in ("!=", "doesn't contain", "does not contain"):
        assert forbidden not in raw, (
            f"Found forbidden negated operator {forbidden!r} in _Papers.base (Pitfall 3 stub-safety guard)"
        )
