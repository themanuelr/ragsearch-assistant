"""
scripts/biblio.py — Bibliography linking for ingested papers (Phase 4, D-02).

Stateless nuclear-task script: turns extraction.references (RefEntry list from
PaperJSON v2 cache) into a living citation graph by injecting a ## References
section with [[wikilinks]] for vault matches and creating stub notes for misses.

Zero LLM calls. Runnable standalone or auto-invoked at ingest step 12c.

Public API:
  run_biblio(paperjson, config) -> str
      Returns vault-relative citing note path on success or "[biblio warning: ...]" on failure.
  upgrade_stub(stub_key, full_title, vault_path) -> None
      Upgrade a stub to a full note when the stubbed paper is later ingested.
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.ingest import (
    _registry_key,
    _normalize_title,
    _check_registry,
    _read_registry,
    _load_config,
)
from scripts.note import _sanitize_filename, _repair_math_escapes
from scripts.obsidian_cli import (
    create_note,
    note_exists,
    _vault_root,
    _validate_vault_path,
    preflight,
)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Regex for extracting stub_key from stub frontmatter (dedup scan — Pitfall 3)
_FRONTMATTER_KEY_RE = re.compile(r"^stub_key:\s*['\"]?(.+?)['\"]?\s*$", re.MULTILINE)

# Regex for stripping existing ## References section on re-link (idempotency)
_REFS_SECTION_RE = re.compile(r"\n## References\n[\s\S]*?(?=\n## |\Z)")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Emit a timestamped progress line to stderr."""
    print(
        f"[biblio {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _ref_key(ref: dict) -> str:
    """Derive the registry/stub key for a single RefEntry dict (D-05).

    Chain: DOI → title-hash only (RefEntry has no arxiv_id field — Pitfall 2).
    """
    return _registry_key({"doi": ref.get("doi"), "title": ref.get("title")})


def _find_stub_by_key(key: str, config: dict) -> str | None:
    """Return vault-relative stub path for key, or None if not found.

    Dedup is anchored on the stub_key frontmatter field, NOT the filename (Pitfall 3).
    """
    vault_root = _vault_root(config)
    stubs_dir = vault_root / "Stubs"
    if not stubs_dir.is_dir():
        return None
    for stub_file in stubs_dir.glob("*.md"):
        try:
            content = stub_file.read_text(encoding="utf-8")
            m = _FRONTMATTER_KEY_RE.search(content)
            if m and m.group(1).strip() == key:
                return f"Stubs/{stub_file.name}"
        except OSError:
            continue
    return None


def _render_stub(ref: dict, stub_key: str, cited_by_path: str) -> str:
    """Render minimal stub markdown for a cited-but-not-yet-ingested paper.

    YAML injection guard: all YAML string values are double-quoted and
    escaped. Raw citation text is placed in the body only (Pitfall 3 / T-04-03).
    RefEntry has no authors field so stub always carries authors: [] (Pitfall 2).
    """
    title = ref.get("title") or "Untitled"
    doi = ref.get("doi") or ""
    year = ref.get("year") or ""
    raw = ref.get("raw") or ""
    today = datetime.date.today().isoformat()

    title_esc = title.replace('"', '\\"')
    key_esc = stub_key.replace('"', '\\"')
    doi_line = f'doi: "{doi}"' if doi else ""
    year_line = f"year: {year}" if year else ""

    frontmatter_lines = [
        "---",
        f'title: "{title_esc}"',
        "authors: []",
    ]
    if year_line:
        frontmatter_lines.append(year_line)
    if doi_line:
        frontmatter_lines.append(doi_line)
    frontmatter_lines += [
        "status: stub",
        f'stub_key: "{key_esc}"',
        f"date_created: {today}",
        "cited_by:",
        f'  - "{cited_by_path}"',
        "---",
        "",
    ]
    frontmatter = "\n".join(frontmatter_lines) + "\n"

    body_lines = [
        "*Stub: this paper has been cited but not yet fully ingested.*",
        "",
        "> [!info] Upgrade",
        "> Run the ingest pipeline on this paper to upgrade this stub to a full note.",
        "",
        "**Raw citation:**",
        raw,
        "",
    ]
    return frontmatter + "\n".join(body_lines)


def _append_cited_by(stub_path: str, citing_path: str, config: dict) -> None:
    """Append citing_path to the stub's cited_by list if not already present.

    Accumulation for BIBLIO-03b: a second citing paper adds to the list rather
    than overwriting.
    """
    vault_root = _vault_root(config)
    abs_path = vault_root / stub_path
    content = abs_path.read_text(encoding="utf-8")

    # Fast check: already listed?
    if f'"{citing_path}"' in content or f"'{citing_path}'" in content:
        return

    # Find insertion point: after the last "  - " entry in the cited_by block
    lines = content.split("\n")
    cited_by_idx = -1
    last_entry_idx = -1

    for i, line in enumerate(lines):
        if line.strip() == "cited_by:":
            cited_by_idx = i
        elif cited_by_idx >= 0 and (
            line.startswith("  - ") or line.startswith("  -\t")
        ):
            last_entry_idx = i
        elif (
            cited_by_idx >= 0
            and last_entry_idx >= 0
            and line.strip()
            and not line.startswith(" ")
        ):
            break  # end of cited_by block

    insert_at = last_entry_idx if last_entry_idx >= 0 else cited_by_idx
    if insert_at >= 0:
        lines.insert(insert_at + 1, f'  - "{citing_path}"')
    else:
        # cited_by block not found — append (shouldn't happen for well-formed stubs)
        lines.append(f'  - "{citing_path}"')

    updated = "\n".join(lines)
    create_note(stub_path, updated, config, overwrite=True)


def _render_references_markdown(refs: list, config: dict, citing_path: str) -> str:
    """Render the ## References section markdown, creating/updating stubs as side effects.

    Per-ref errors are caught, logged, and emitted as raw text — the loop continues
    (D-09 inner containment / SC4 / BIBLIO-04d).
    """
    registry_path = config.get("registry_path", "")
    lines = []

    for i, ref in enumerate(refs):
        try:
            number = ref.get("number") if ref.get("number") is not None else (i + 1)
            title = ref.get("title") or ""
            raw = ref.get("raw") or ""
            fill_failed = ref.get("fill_failed", False)

            # Malformed/empty refs: emit raw text only, no stub (SC4/BIBLIO-04d)
            if fill_failed or (not title and not raw):
                display = _repair_math_escapes(raw) if raw else "(reference unavailable)"
                lines.append(f"{number}. {display}")
                continue

            key = _ref_key(ref)
            entry = _check_registry(key, registry_path) if registry_path else None

            if entry:
                # Registry hit: derive wikilink from entry["title"] (no vault_note field — Pitfall 1)
                link_title = _sanitize_filename(entry["title"])
                display_title = _repair_math_escapes(link_title)
                lines.append(f"{number}. [[{display_title}]]")
            else:
                # Registry miss: dedup-or-create stub (D-06 / BIBLIO-03)
                stub_title = title or raw
                stub_filename = _sanitize_filename(stub_title)
                stub_path = f"Stubs/{stub_filename}.md"
                existing_stub = _find_stub_by_key(key, config)
                if existing_stub:
                    # Existing stub for this key — accumulate cited_by (BIBLIO-03b)
                    _append_cited_by(existing_stub, citing_path, config)
                else:
                    # New stub — safe default won't clobber an existing stub (overwrite=False)
                    stub_content = _render_stub(ref, key, citing_path)
                    create_note(stub_path, stub_content, config, overwrite=False)

                # Markdown-injection guard: emit as numbered list item, never as heading (T-04-04)
                display_raw = _repair_math_escapes(raw or stub_title)
                lines.append(f"{number}. {display_raw} *(not yet in vault)*")

        except Exception as e:
            _log(f"ref {i} failed, emitting raw: {e}")
            # Per-ref fallback: emit raw string and continue (D-09 inner / SC4)
            raw_fb = ref.get("raw") or ""
            number_fb = ref.get("number") if ref.get("number") is not None else (i + 1)
            lines.append(f"{number_fb}. {raw_fb}")

    return "\n".join(lines)


def _inject_references_section(note_path: str, refs_markdown: str, config: dict) -> None:
    """Inject or replace the ## References section in an existing note.

    Idempotency: strips any existing ## References section before re-inserting,
    so a second run_biblio call produces exactly one ## References section (BIBLIO-02d).
    Insertion order: immediately before ## My Notes when present; append otherwise.
    """
    vault_root = _vault_root(config)
    abs_path = vault_root / note_path
    if not abs_path.exists():
        raise FileNotFoundError(f"note not found: {note_path}")
    content = abs_path.read_text(encoding="utf-8")

    # Idempotency: strip existing ## References section if present
    if "\n## References" in content:
        content = _REFS_SECTION_RE.sub("", content)

    # Insert before ## My Notes (preferred placement per note.py _render_note);
    # fallback: append to end
    marker = "\n## My Notes"
    if marker in content:
        idx = content.index(marker)
        updated = content[:idx] + "\n\n## References\n\n" + refs_markdown + content[idx:]
    else:
        updated = content.rstrip("\n") + "\n\n## References\n\n" + refs_markdown + "\n"

    create_note(note_path, updated, config, overwrite=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _process_refs(refs: list, citing_path: str, config: dict) -> str:
    """Process all refs and return rendered ## References markdown.

    Extracted from run_biblio so tests can patch it to simulate wholesale failure
    (test_biblio_failure_does_not_abort_ingest / BIBLIO-04e / SC4).
    """
    return _render_references_markdown(refs, config, citing_path)


def run_biblio(paperjson: dict, config: dict) -> str:
    """
    Non-fatal tail stage: resolve references and inject ## References section.

    Returns vault-relative citing note path on success, or "[biblio warning: ...]"
    on any unhandled failure (D-09 outer containment / SC4 / BIBLIO-04e).

    Args:
        paperjson: PaperJSON v2 dict with extraction.references list.
        config:    Config dict with vault_path and registry_path.
    """
    try:
        refs = paperjson.get("extraction", {}).get("references", [])
        title = (
            paperjson.get("extraction", {}).get("metadata", {}).get("title", "") or ""
        )
        citing_path = f"Papers/{_sanitize_filename(title)}.md"

        refs_markdown = _process_refs(refs, citing_path, config)
        _inject_references_section(citing_path, refs_markdown, config)
        return citing_path
    except Exception as e:
        return f"[biblio warning: {e}]"


# ---------------------------------------------------------------------------
# Stub upgrade helpers (Plan 03 implements fully; signatures pinned by RED tests 04a/04b)
# ---------------------------------------------------------------------------

def _parse_cited_by(content: str) -> list:
    """Extract cited_by path list from stub YAML frontmatter.

    Uses a line-by-line scan rather than a YAML library to avoid new dependencies
    (A4 assumption). The frontmatter is always written by _render_stub/_append_cited_by
    so its structure is deterministic.
    """
    cited_by: list[str] = []
    in_cited_by = False
    in_frontmatter = False
    frontmatter_count = 0

    for line in content.split("\n"):
        stripped = line.strip()
        if stripped == "---":
            frontmatter_count += 1
            if frontmatter_count >= 2:
                break  # end of frontmatter
            in_frontmatter = True
            continue
        if not in_frontmatter:
            continue
        if stripped == "cited_by:":
            in_cited_by = True
            continue
        if in_cited_by:
            if stripped.startswith("- "):
                val = stripped[2:].strip().strip('"\'')
                cited_by.append(val)
            elif stripped and not stripped.startswith("-"):
                break  # end of cited_by block
    return cited_by


def _parse_frontmatter_field(content: str, field: str) -> str:
    """Extract a scalar field value from YAML frontmatter (regex-based, single-value only)."""
    pattern = re.compile(
        rf'^{re.escape(field)}:\s*["\']?(.+?)["\']?\s*$', re.MULTILINE
    )
    m = pattern.search(content)
    if m:
        return m.group(1).strip().strip('"\'')
    return ""


def _resolve_citing_path(
    cited_path: str, vault_root: pathlib.Path
) -> pathlib.Path | None:
    """Try multiple path forms to resolve a cited_by value to an existing file.

    cited_by values may be stored as vault-relative paths ("Papers/X.md") or
    as bare titles ("X"). Try the most specific form first.
    """
    candidates = [
        vault_root / cited_path,
        vault_root / (cited_path + ".md"),
        vault_root / "Papers" / cited_path,
        vault_root / "Papers" / (cited_path + ".md"),
    ]
    for p in candidates:
        try:
            if p.exists():
                return p
        except OSError:
            continue
    return None


def upgrade_stub(stub_key: str, full_title: str, vault_path: str) -> None:
    """
    Upgrade a stub to a full note: rewrite cited_by backlinks and delete stub.

    Called when a previously-stubbed paper is later fully ingested (BIBLIO-04 / D-07).
    Backlink replacement is scoped to the ## References section of each citing note
    to avoid accidental replacement in other sections (Pitfall 7).

    Args:
        stub_key:   The registry key stored in the stub's stub_key frontmatter field.
        full_title: The full title of the newly ingested paper.
        vault_path: Absolute path to the vault root.
    """
    config = {"vault_path": vault_path}
    vault_root = _vault_root(config)

    # Find stub by scanning stub_key frontmatter
    stub_path = _find_stub_by_key(stub_key, config)
    if not stub_path:
        return  # No stub for this key — nothing to upgrade

    stub_abs = vault_root / stub_path
    stub_content = stub_abs.read_text(encoding="utf-8")

    # Parse stub title (for old wikilink form) and cited_by list
    stub_title_raw = _parse_frontmatter_field(stub_content, "title")
    stub_title = (
        _sanitize_filename(stub_title_raw) if stub_title_raw else stub_abs.stem
    )
    full_link_title = _sanitize_filename(full_title)
    cited_by = _parse_cited_by(stub_content)

    stub_link = f"[[{stub_title}]]"
    full_link = f"[[{full_link_title}]]"

    # Rewrite [[stub_title]] → [[full_title]] in each citing note (bounded to cited_by)
    for citing_path in cited_by:
        abs_citing = _resolve_citing_path(citing_path, vault_root)
        if abs_citing is None:
            continue
        try:
            citing_content = abs_citing.read_text(encoding="utf-8")
        except OSError:
            continue
        if stub_link not in citing_content:
            continue
        # Scope replacement to ## References section (Pitfall 7)
        refs_match = re.search(
            r"\n## References\n([\s\S]*?)(?=\n## |\Z)", citing_content
        )
        if refs_match:
            refs_section = refs_match.group(0).replace(stub_link, full_link)
            updated = (
                citing_content[: refs_match.start()]
                + refs_section
                + citing_content[refs_match.end() :]
            )
        else:
            # No \n## References marker (e.g. section at top of file) — global replace
            updated = citing_content.replace(stub_link, full_link)
        vault_relative = abs_citing.relative_to(vault_root).as_posix()
        create_note(vault_relative, updated, config, overwrite=True)

    # Create placeholder Papers/ note (ingest will overwrite with full content)
    full_note_path = f"Papers/{full_link_title}.md"
    if not note_exists(full_note_path, config):
        create_note(
            full_note_path,
            f"# {full_title}\n\n## My Notes\n\n",
            config,
            overwrite=False,
        )

    # Delete stub (not via create_note — actual filesystem deletion)
    try:
        stub_abs.unlink()
    except OSError as e:
        print(
            f"[biblio warning: could not delete stub {stub_path}: {e}]",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Windows cp1252 guard (Pitfall 5) — mirrors note.py line 629 and ingest.py line 2516
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Run bibliography linking for a PaperJSON v2 cache file."
    )
    parser.add_argument(
        "--paperjson",
        default=None,
        help="Path to the PaperJSON v2 cache file (JSON).",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="PDF/URL stem — resolves .paperjson_cache/<stem>.json when --paperjson omitted.",
    )
    args = parser.parse_args()

    try:
        config = _load_config()
        if args.paperjson:
            pj_path = args.paperjson
        elif args.stem:
            pj_path = str(
                (pathlib.Path(".paperjson_cache") / f"{args.stem}.json").resolve()
            )
        else:
            parser.error("either --paperjson or --stem is required")
        with open(pj_path, encoding="utf-8") as f:
            paperjson = json.load(f)
        result = run_biblio(paperjson, config)
        print(result)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[biblio error: {e}]", file=sys.stderr)
        sys.exit(1)
