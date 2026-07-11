"""
scripts/link.py — Topic-graph linking for ingested papers (Phase 6, GRAPH-01/02).

Stateless nuclear-task script: reads the LIVE Papers/ note's ``tags:`` frontmatter
(D-02 — the note on disk, never the registry snapshot) and maintains
``Topics/<slug>.md`` index notes (GRAPH-01), then injects a ``## Related Topics``
footer section with alias-piped wikilinks on the paper note itself (GRAPH-02).

Zero LLM calls. Runnable standalone or auto-invoked at ingest Step 12e (miss
path) / Step 9d (cache-hit path).

Public API:
  run_link(paperjson, config) -> str
      Returns the paper's vault-relative note path on success, or
      "[link warning: ...]" on any unhandled failure — never raises.
"""

import datetime
import os
import pathlib
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.note import _sanitize_filename, _slugify_tag
from scripts.obsidian_cli import create_note, _vault_root

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Managed-block sentinels wrapping a topic note's auto-maintained member list
# (HTML comments, hidden in Obsidian's reading view). Modeled on
# scripts/biblio.py's _stub_anchor precedent — a topic note has no fixed
# multi-section skeleton the way a paper note does, so a comment-anchor pair
# is used instead of a heading-based regex.
_MANAGED_BLOCK_RE = re.compile(r"<!--link:members:start-->[\s\S]*?<!--link:members:end-->")

# Regex for stripping an existing ## Related Topics section on re-link
# (idempotency). Cross-reference: scripts/biblio.py::_REFS_SECTION_RE /
# _inject_references_section — this is the same strip-and-reinject shape,
# reused verbatim with the heading literal swapped (Pitfall 4: note.py's
# _render_note always places ## My Notes last, so both modules anchor on the
# identical "\n## My Notes" marker string; a future note.py section-order
# change must be caught by both).
_RELATED_TOPICS_RE = re.compile(r"\n## Related Topics\n[\s\S]*?(?=\n## |\Z)")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Emit a timestamped progress line to stderr."""
    print(
        f"[link {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Frontmatter tag parsing (no yaml import — mirrors biblio._parse_cited_by)
# ---------------------------------------------------------------------------

def _parse_tags(content: str) -> list[str]:
    """Hand-parse the YAML frontmatter ``tags:`` list without a ``yaml`` import.

    Mirrors scripts/biblio.py::_parse_cited_by's no-yaml-import list-parsing
    precedent (06-PATTERNS.md Pattern 3). Treats "tags: key absent" and
    "tags: present but empty" identically as "no tags" — both simply return
    ``[]`` since no ``- "..."`` lines are collected.
    """
    tags: list[str] = []
    in_tags = in_frontmatter = False
    fm_count = 0
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped == "---":
            fm_count += 1
            if fm_count >= 2:
                break
            in_frontmatter = True
            continue
        if not in_frontmatter:
            continue
        if stripped == "tags:":
            in_tags = True
            continue
        if in_tags:
            if stripped.startswith("- "):
                tags.append(stripped[2:].strip().strip('"\''))
            elif stripped and not stripped.startswith("-"):
                break
    return tags


# ---------------------------------------------------------------------------
# Topic identity helpers (D-01/D-03, Pitfall 2)
# ---------------------------------------------------------------------------

def _topic_flat_basename(slug: str) -> str:
    """Flatten a (possibly nested) tag slug into a topic-note FILENAME basename.

    Pitfall 2: ``_slugify_tag``'s charset (``[\\w/-]+``) preserves ``/`` as an
    Obsidian nested-tag separator, so a naive ``Topics/<slug>.md`` path could
    resolve to a nested directory. Option b (06-RESEARCH.md recommendation,
    pinned by 06-01's RED tests): flatten "/" -> "-" in the FILENAME only. The
    frontmatter tag string itself is never rewritten by this phase.
    """
    return slug.replace("/", "-")


def _topic_note_path(slug: str) -> str:
    """Vault-relative path to a topic's index note (flat filename, Pitfall 2)."""
    return f"Topics/{_topic_flat_basename(slug)}.md"


def _topic_alias(slug: str) -> str:
    """Readable de-slug for a topic's D-03 ``aliases:`` entry.

    Deterministic, not semantically authoritative — replaces "/" and "-"
    with spaces and title-cases the result (executor discretion on exact
    casing per 06-02-PLAN.md).
    """
    return slug.replace("/", " ").replace("-", " ").title()


# ---------------------------------------------------------------------------
# Topic note rendering (managed member-list block, D-03/D-04)
# ---------------------------------------------------------------------------

def _render_managed_block(paper_names: list[str]) -> str:
    """Render the sentinel-wrapped member-list block for a topic note.

    One ``- [[name]]`` per SORTED paper basename (deterministic across runs,
    GRAPH-01 / D-04 idempotency).
    """
    lines = [f"- [[{name}]]" for name in sorted(paper_names)]
    return "<!--link:members:start-->\n" + "\n".join(lines) + "\n<!--link:members:end-->"


def _render_topic_note(slug: str, member_names: list[str], existing: str | None) -> str:
    """Render (or idempotently update) a topic note's full content.

    When ``existing`` is None, builds the full scaffold: double-quoted
    frontmatter (title + D-03 ``aliases: ["<Readable Name>"]``, reusing
    note.py's ``"`` -> ``\\"`` escape convention, T-06-03), an H1, the managed
    member-list block, and a preserved user-notes region. When ``existing`` is
    provided, strips ONLY its managed block via ``_MANAGED_BLOCK_RE`` and
    substitutes the freshly-rendered block, leaving frontmatter and any
    user-added content byte-untouched (strip-and-reinject idempotency).
    """
    managed_block = _render_managed_block(member_names)

    if existing is not None and _MANAGED_BLOCK_RE.search(existing):
        return _MANAGED_BLOCK_RE.sub(managed_block, existing)

    if existing is not None:
        # Existing file present but somehow missing the managed-block sentinels
        # (e.g. hand-authored before link.py ran) — append rather than clobber.
        return existing.rstrip("\n") + "\n\n" + managed_block + "\n"

    alias = _topic_alias(slug)
    escaped_alias = alias.replace('"', '\\"')
    lines = [
        "---",
        f'title: "{escaped_alias}"',
        f'aliases: ["{escaped_alias}"]',
        "---",
        "",
        f"# {alias}",
        "",
        managed_block,
        "",
        "<!-- Add your own notes about this topic below. -->",
        "",
    ]
    return "\n".join(lines)


def _build_topic_membership(
    vault_root: pathlib.Path, tags_filter: set[str] | None = None
) -> dict[str, list[str]]:
    """Scan Papers/*.md (NEVER Stubs/*.md, D-05 — stubs have no topics) and
    build ``{slug: [paper_basename, ...]}`` from each note's LIVE ``tags:``
    frontmatter.

    When ``tags_filter`` (a set of slugs) is provided, only those slugs are
    returned (incremental per-ingest case, D-06). When None, every tag found
    across the vault is returned (backfill case, used by Plan 03/04's
    ``--all``/``--refresh``).
    """
    membership: dict[str, list[str]] = {}
    papers_dir = pathlib.Path(vault_root) / "Papers"
    if not papers_dir.is_dir():
        return membership
    for note_path in papers_dir.glob("*.md"):
        content = note_path.read_text(encoding="utf-8")
        paper_name = note_path.stem
        for raw_tag in _parse_tags(content):
            slug = _slugify_tag(raw_tag)
            if slug is None:
                continue
            if tags_filter is not None and slug not in tags_filter:
                continue
            membership.setdefault(slug, []).append(paper_name)
    return membership


# ---------------------------------------------------------------------------
# Related Topics injection on the paper note itself (GRAPH-02, D-07/D-08/D-09/D-10)
# ---------------------------------------------------------------------------

def _inject_related_topics(note_path: str, topics_markdown: str, config: dict) -> None:
    """Inject, replace, or strip the ``## Related Topics`` section on a note.

    Modeled verbatim on scripts/biblio.py::_inject_references_section (see the
    module-level cross-reference comment on ``_RELATED_TOPICS_RE`` above).
    Idempotency: any existing ``## Related Topics`` section is stripped before
    re-insertion, so a second ``run_link`` call produces exactly one section.
    Per D-10, when ``topics_markdown`` is empty the stripped content is
    written back as-is — NO empty header is ever (re-)added. Otherwise the
    section is inserted immediately before the literal ``"\\n## My Notes"``
    marker (copied, not retyped, from biblio.py — Pitfall 4), or appended at
    end-of-file when that marker is absent.
    """
    vault_root = _vault_root(config)
    abs_path = vault_root / note_path
    if not abs_path.exists():
        raise FileNotFoundError(f"note not found: {note_path}")
    content = abs_path.read_text(encoding="utf-8")

    if "\n## Related Topics" in content:
        content = _RELATED_TOPICS_RE.sub("", content)

    if not topics_markdown:
        # D-10: zero-tag paper gets no header at all — write the stripped
        # content back (removes any stale section) and stop.
        create_note(note_path, content, config, overwrite=True)
        return

    marker = "\n## My Notes"
    if marker in content:
        idx = content.index(marker)
        updated = content[:idx] + "\n\n## Related Topics\n\n" + topics_markdown + content[idx:]
    else:
        updated = content.rstrip("\n") + "\n\n## Related Topics\n\n" + topics_markdown + "\n"

    create_note(note_path, updated, config, overwrite=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_link(paperjson: dict, config: dict) -> str:
    """
    Non-fatal tail stage: maintain Topics/<slug>.md index notes for this
    paper's live tags (GRAPH-01) and inject/strip the paper note's
    ## Related Topics section (GRAPH-02).

    Returns the paper's vault-relative note path on success, or
    "[link warning: ...]" on any unhandled failure (verbatim contract shape
    from scripts/embed.py::run_embed) — never raises.

    Args:
        paperjson: PaperJSON v2 dict; only extraction.metadata.title is used
                   to locate the paper's live note (D-02 — tags are read from
                   the note on disk, never from analysis.topics/the registry).
        config:    Config dict with at least "vault_path".
    """
    try:
        metadata = paperjson.get("extraction", {}).get("metadata", {}) or {}
        title = metadata.get("title") or ""
        paper_note_path = f"Papers/{_sanitize_filename(title)}.md"

        vault_root = _vault_root(config)
        abs_note_path = vault_root / paper_note_path
        if not abs_note_path.exists():
            return f"[link warning: paper note not found: {paper_note_path}]"

        content = abs_note_path.read_text(encoding="utf-8")
        raw_tags = _parse_tags(content)
        this_papers_slugs = sorted(
            {slug for slug in (_slugify_tag(t) for t in raw_tags) if slug is not None}
        )
        paper_name = abs_note_path.stem

        if this_papers_slugs:
            membership = _build_topic_membership(vault_root, tags_filter=set(this_papers_slugs))
            for slug in this_papers_slugs:
                member_names = membership.get(slug) or [paper_name]
                topic_path = _topic_note_path(slug)
                abs_topic_path = vault_root / topic_path
                existing = (
                    abs_topic_path.read_text(encoding="utf-8") if abs_topic_path.exists() else None
                )
                rendered = _render_topic_note(slug, member_names, existing)
                create_note(topic_path, rendered, config, overwrite=True)

            # D-08: strictly THIS paper's own topics, no co-topic paper links.
            # D-09: alias-piped [[flat-basename|Readable Name]], sorted for
            # determinism.
            topics_markdown = "\n".join(
                f"- [[{_topic_flat_basename(slug)}|{_topic_alias(slug)}]]"
                for slug in this_papers_slugs
            )
        else:
            # D-10: no tags -> _inject_related_topics strips any stale section
            # and adds no header.
            topics_markdown = ""

        _inject_related_topics(paper_note_path, topics_markdown, config)

        return paper_note_path
    except Exception as e:
        return f"[link warning: {e}]"
