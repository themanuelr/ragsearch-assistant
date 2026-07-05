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
import hashlib
import json
import os
import pathlib
import re
import sys
import unicodedata

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.ingest import (
    _normalize_title,
    _check_registry,
    _read_registry,
    _load_config,
    _TITLE_HASH_PREFIX_LEN,
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

# Miss-branch marker appended by _render_references_markdown for a stub-matched
# reference (see the exact f-string there). Single source of truth so the
# upgrade-path backlink rewrite (04-09/Gap F bug 2) targets the ACTUAL rendered
# text instead of an idealized wikilink the miss branch never produces.
_STUB_MISS_MARKER = " *(not yet in vault)*"


def _stub_anchor(stub_key: str) -> str:
    """Invisible HTML-comment anchor tying a rendered miss-branch line to its stub.

    Appended after _STUB_MISS_MARKER at render time and matched EXACTLY at
    upgrade time, so backlink rewriting never has to infer identity from prose
    (CR-02: substring matching on normalized titles false-positives across
    unrelated references, e.g. "Learning" vs "... Deep Learning ...").
    HTML comments are hidden in Obsidian's reading view.
    """
    return f"<!--stub:{stub_key}-->"


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

def _yaml_escape(value: str) -> str:
    """Escape a string for safe interpolation into a double-quoted YAML scalar.

    Backslashes MUST be escaped before quotes (order matters): a trailing
    backslash would otherwise escape the closing quote and swallow subsequent
    frontmatter keys into the string value (CR-01 — plausible for LaTeX-bearing
    scientific titles, cf. _repair_math_escapes).
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


# NOTE (WR-01): the original `_ref_key` helper (DOI → title-hash-only chain via
# _registry_key) was deleted — it re-encoded the exact Layer-2 matching bug
# closed in 04-04 and had no remaining callers. Resolution goes through
# _resolve_ref_in_registry + _match_key instead.


# ---------------------------------------------------------------------------
# Layer-2 offline matching: normalized-title index + accent-folded dedup key
# ---------------------------------------------------------------------------

def _normalize_title_for_match(title: str) -> str:
    """Accent-fold and normalize a title for cross-ref matching and dedup keying.

    Steps:
      1. NFKD-decompose via unicodedata.normalize so accented chars split into
         base letter + combining mark (e.g. "ü" → "u" + combining umlaut).
      2. Drop all combining characters so accented chars collapse to their ASCII
         base (e.g. "Müller" → "Muller").
      3. Delegate to the existing _normalize_title from scripts.ingest which
         lowercases, strips, and collapses whitespace/punctuation runs to a
         single space — giving a strict superset of that normalization.

    This is the SINGLE normalization used for both the title index and _match_key.
    """
    decomposed = unicodedata.normalize("NFKD", title)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _normalize_title(ascii_only)


def _build_title_index(registry_path: str) -> dict:
    """Build a read-only normalized-title → registry_key index from the registry.

    Returns a dict mapping _normalize_title_for_match(entry["title"]) to the
    registry key for every entry that carries a non-empty title field.
    Never writes to the registry (D-08). Returns {} if registry is absent or empty.
    """
    registry = _read_registry(registry_path) if registry_path else {}
    index: dict[str, str] = {}
    for reg_key, entry in registry.items():
        title = entry.get("title") or ""
        if title:
            norm = _normalize_title_for_match(title)
            if norm:
                index[norm] = reg_key
    return index


def _resolve_ref_in_registry(
    ref: dict, registry_path: str, title_index: dict
) -> dict | None:
    """Layered registry resolution for a single reference (Layer 1 + Layer 2).

    Resolution chain (offline, no network):
      1. Exact DOI via _check_registry when the ref carries a doi.
      2. Normalized-title lookup: _normalize_title_for_match(ref title) → title_index
         → matched registry key → _check_registry(matched_key, registry_path).

    Returns the registry entry dict on any hit, or None on miss.
    """
    # Layer 1: exact DOI
    doi = ref.get("doi")
    if doi and registry_path:
        entry = _check_registry(doi, registry_path)
        if entry is not None:
            return entry

    # Layer 2: normalized-title index
    title = ref.get("title") or ""
    if title and title_index and registry_path:
        norm = _normalize_title_for_match(title)
        matched_key = title_index.get(norm)
        if matched_key:
            entry = _check_registry(matched_key, registry_path)
            if entry is not None:
                return entry

    return None


def _match_key(ref_or_meta: dict) -> str:
    """Derive the identity key for stub dedup and upgrade detection.

    DOI if present, else "sha256:<16-hex-chars>" derived from the
    accent-folded normalized title (via _normalize_title_for_match).
    Uses _TITLE_HASH_PREFIX_LEN so the format is byte-identical to
    _registry_key for ASCII-only titles.
    """
    doi = ref_or_meta.get("doi")
    if doi:
        return doi
    title = ref_or_meta.get("title") or ""
    norm = _normalize_title_for_match(title)
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    return "sha256:" + digest[:_TITLE_HASH_PREFIX_LEN]


def _normalize_title_for_dedup(title: str) -> str:
    """Stricter dedup normalizer for the stub-title-index (04-07 / Gap C mode 2).

    Delegates to _normalize_title_for_match (NFKD accent-fold + lowercase +
    punctuation-to-space collapse) then removes all remaining space characters,
    so hyphen/space/joined title variants (e.g. 'cation-chloride', 'cation
    chloride', 'cationchloride') all converge to the same string.

    Used ONLY for stub-dedup convergence via the stub-title-index — _match_key
    and _normalize_title_for_match are left byte-stable so the stub-upgrade
    self_key and Layer-2 wikilink matching are unchanged.
    """
    normalized = _normalize_title_for_match(title)
    return normalized.replace(" ", "")


def _build_stub_title_index(config: dict) -> dict:
    """Build a read-only dedup-normalized-title -> stub_key index from Stubs/.

    Scans vault/Stubs/*.md, parsing each stub's `title` and `stub_key`
    frontmatter fields. Maps _normalize_title_for_dedup(title) -> stub_key for
    every stub with a non-empty title and stub_key. Read-only over the local
    Stubs/ directory only — no registry access, no network (D-08). Returns {}
    when Stubs/ is absent; tolerates per-file OSError by skipping (mirrors
    _find_stub_by_key).
    """
    vault_root = _vault_root(config)
    stubs_dir = vault_root / "Stubs"
    index: dict[str, str] = {}
    if not stubs_dir.is_dir():
        return index
    for stub_file in stubs_dir.glob("*.md"):
        try:
            content = stub_file.read_text(encoding="utf-8")
        except OSError:
            continue
        key_match = _FRONTMATTER_KEY_RE.search(content)
        stub_key = key_match.group(1).strip() if key_match else ""
        title = _parse_frontmatter_field(content, "title")
        if title and stub_key:
            norm = _normalize_title_for_dedup(title)
            if norm:
                index[norm] = stub_key
    return index


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

    title_esc = _yaml_escape(title)
    key_esc = _yaml_escape(stub_key)
    doi_line = f'doi: "{_yaml_escape(doi)}"' if doi else ""
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
        f'  - "{_yaml_escape(cited_by_path)}"',
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
        lines.insert(insert_at + 1, f'  - "{_yaml_escape(citing_path)}"')
    else:
        # cited_by block not found — append (shouldn't happen for well-formed stubs)
        lines.append(f'  - "{_yaml_escape(citing_path)}"')

    updated = "\n".join(lines)
    create_note(stub_path, updated, config, overwrite=True)


def _render_references_markdown(refs: list, config: dict, citing_path: str) -> str:
    """Render the ## References section markdown, creating/updating stubs as side effects.

    Per-ref errors are caught, logged, and emitted as raw text — the loop continues
    (D-09 inner containment / SC4 / BIBLIO-04d).

    Resolution chain (Layer 1 + Layer 2, fully offline):
      - Layer 1: exact DOI registry hit
      - Layer 2: normalized-title index hit (accent-folded, no network)
      - Miss: dedup via stub-title-index (04-07), else create stub keyed by _match_key
    """
    registry_path = config.get("registry_path", "")

    # Build the title index ONCE before the per-ref loop (Layer 2 prerequisite)
    title_index = _build_title_index(registry_path) if registry_path else {}

    # Build the stub-title-index ONCE before the per-ref loop (04-07 dedup prerequisite).
    # Read-only over vault/Stubs/ — updated in-memory below so intra-run duplicates converge.
    stub_title_index = _build_stub_title_index(config)

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

            # Layered registry resolution (Layer 1: exact DOI, Layer 2: normalized title)
            entry = _resolve_ref_in_registry(ref, registry_path, title_index)

            if entry:
                # Registry hit: derive wikilink from entry["title"] (no vault_note field — Pitfall 1)
                link_title = _sanitize_filename(entry["title"])
                display_title = _repair_math_escapes(link_title)
                lines.append(f"{number}. [[{display_title}]]")
            else:
                # Registry miss: dedup-or-create stub (D-06 / BIBLIO-03)
                stub_title = title or raw

                # 04-07: consult the stub-title-index BEFORE computing _match_key so a
                # doiless ref converges onto an existing DOI-keyed stub, and hyphen/
                # space/joined title variants converge onto each other (Gap C).
                dedup_norm = _normalize_title_for_dedup(stub_title)
                indexed_key = stub_title_index.get(dedup_norm) if dedup_norm else None
                stub_key = indexed_key if indexed_key else _match_key(ref)

                stub_filename = _sanitize_filename(stub_title)
                stub_path = f"Stubs/{stub_filename}.md"
                existing_stub = _find_stub_by_key(stub_key, config)
                if existing_stub:
                    # Existing stub for this key — accumulate cited_by (BIBLIO-03b)
                    _append_cited_by(existing_stub, citing_path, config)
                else:
                    # New stub — safe default won't clobber an existing stub (overwrite=False)
                    stub_content = _render_stub(ref, stub_key, citing_path)
                    create_note(stub_path, stub_content, config, overwrite=False)
                    # Register in-memory so a later ref in the SAME run converges (04-07)
                    if dedup_norm:
                        stub_title_index[dedup_norm] = stub_key

                # Markdown-injection guard: emit as numbered list item, never as heading (T-04-04)
                # The trailing _stub_anchor carries the EXACT stub_key so the
                # upgrade-path rewrite can identify this line without fuzzy
                # title matching (CR-02).
                display_raw = _repair_math_escapes(raw or stub_title)
                lines.append(
                    f"{number}. {display_raw}{_STUB_MISS_MARKER}{_stub_anchor(stub_key)}"
                )

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

        # Stub upgrade detection (D-07/SC3): if the paper being ingested matches
        # an existing stub, promote it — merge cited_by into full note frontmatter,
        # rewrite backlinks, and delete the stub. Inner try/except so any upgrade
        # failure is swallowed and citing_path is still returned (D-09).
        try:
            metadata = paperjson.get("extraction", {}).get("metadata", {})
            # 04-09/Gap F bug 1: consult the same stub-title-index the 04-07 miss
            # branch uses BEFORE falling back to _match_key, so a DOI-bearing new
            # ingest still resolves a pre-existing title-hash-keyed stub (the stub's
            # original citers never captured a DOI for it). Falls back to today's
            # _match_key(metadata) byte-for-byte when no title-index hit exists.
            self_title = metadata.get("title") or ""
            stub_title_index = _build_stub_title_index(config)
            dedup_norm = _normalize_title_for_dedup(self_title) if self_title else ""
            indexed_self_key = stub_title_index.get(dedup_norm) if dedup_norm else None
            self_key = indexed_self_key if indexed_self_key else _match_key(metadata)
            stub_path = _find_stub_by_key(self_key, config)
            if stub_path:
                vault_root = _vault_root(config)
                full_note_abs = vault_root / citing_path
                if full_note_abs.exists():
                    stub_abs = vault_root / stub_path
                    stub_content = stub_abs.read_text(encoding="utf-8")
                    cited_by = _parse_cited_by(stub_content)
                    _merge_cited_by_into_note(citing_path, cited_by, config)
                    _upgrade_stub(stub_path, citing_path, config)
        except Exception as _upgrade_err:
            _log(f"stub upgrade failed (non-fatal): {_upgrade_err}")

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


def _merge_cited_by_into_note(full_note_path: str, cited_by: list, config: dict) -> None:
    """Merge cited_by paths into the full note's YAML frontmatter (SC3).

    If the frontmatter has no cited_by block, inserts one immediately before
    the closing --- of the frontmatter. If a block exists, union-merges (skips
    paths already present). Writes atomically via create_note(overwrite=True).

    Args:
        full_note_path: vault-relative path to the full note (Papers/<file>.md).
        cited_by:       List of vault-relative citing-note paths from the stub.
        config:         Config dict with vault_path.
    """
    if not cited_by:
        return

    vault_root = _vault_root(config)
    abs_path = vault_root / full_note_path
    content = abs_path.read_text(encoding="utf-8")

    # Determine which paths are already listed in the note's cited_by block
    existing = set(_parse_cited_by(content))
    new_paths = [p for p in cited_by if p not in existing]
    if not new_paths:
        return

    new_entry_lines = [f'  - "{_yaml_escape(p)}"' for p in new_paths]

    lines = content.split("\n")
    fm_count = 0
    cited_by_line = -1    # index of "cited_by:" in frontmatter
    last_entry_line = -1  # index of last "  - ..." under cited_by
    closing_fm_line = -1  # index of closing "---"
    in_cited_by = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "---":
            fm_count += 1
            if fm_count == 2:
                closing_fm_line = i
                break
            continue
        if fm_count == 0:
            continue
        if stripped == "cited_by:":
            cited_by_line = i
            in_cited_by = True
            continue
        if in_cited_by:
            if line.startswith("  - "):
                last_entry_line = i
            elif stripped and not line.startswith(" "):
                in_cited_by = False

    if cited_by_line >= 0:
        # Union-merge: insert new entries after the last existing entry
        insert_at = last_entry_line if last_entry_line >= 0 else cited_by_line
        for entry in reversed(new_entry_lines):
            lines.insert(insert_at + 1, entry)
    elif closing_fm_line >= 0:
        # No cited_by block yet — insert block before the closing ---
        insert_block = ["cited_by:"] + new_entry_lines
        for entry in reversed(insert_block):
            lines.insert(closing_fm_line, entry)
    else:
        # No recognisable frontmatter — append to end of file
        lines += ["cited_by:"] + new_entry_lines

    create_note(full_note_path, "\n".join(lines), config, overwrite=True)


def _upgrade_stub(stub_path: str, full_note_path: str, config: dict) -> None:
    """Rewrite cited_by backlinks and delete the stub file (D-07/SC3).

    Bounded to the stub's cited_by list — never an O(vault) scan (D-04/D-07).
    Replacement is scoped to the ## References section of each citing note only
    so occurrences outside ## References are untouched (Pitfall 7).
    Stub deletion is wrapped so failure logs a warning and never re-raises (D-09/SC4).

    Called from run_biblio after _merge_cited_by_into_note. The full note must
    already exist; its frontmatter title drives the [[full_title]] wikilink.

    Args:
        stub_path:      Vault-relative path to the stub (Stubs/<file>.md).
        full_note_path: Vault-relative path to the full note (Papers/<file>.md).
        config:         Config dict with vault_path.
    """
    vault_root = _vault_root(config)
    stub_abs = vault_root / stub_path
    stub_content = stub_abs.read_text(encoding="utf-8")

    cited_by = _parse_cited_by(stub_content)
    stub_title_raw = _parse_frontmatter_field(stub_content, "title")

    # CR-02: the rendered miss-branch line carries an exact <!--stub:{key}-->
    # anchor (see _render_references_markdown). Match on that anchor — never on
    # fuzzy prose — so upgrading one stub can never relink an unrelated
    # reference whose text happens to contain this stub's title as a substring.
    key_match = _FRONTMATTER_KEY_RE.search(stub_content)
    stub_key = key_match.group(1).strip() if key_match else ""
    anchor = _stub_anchor(stub_key) if stub_key else ""

    full_note_abs = vault_root / full_note_path
    full_note_content = full_note_abs.read_text(encoding="utf-8")
    full_title_raw = _parse_frontmatter_field(full_note_content, "title")
    full_link_title = (
        _sanitize_filename(full_title_raw) if full_title_raw else full_note_abs.stem
    )

    # Legacy fallback (lines rendered before the anchor scheme): a stub-matched
    # reference is PLAIN TEXT ("{number}. {display} *(not yet in vault)*"), never
    # a "[[stub_title]]" wikilink (04-09/Gap F bug 2). For those lines require
    # EXACT normalized-title equality — substring containment is forbidden (CR-02).
    stub_norm = _normalize_title_for_dedup(stub_title_raw) if stub_title_raw else ""

    # Rewrite the real miss-branch line -> [[full_title]] in each citing note (bounded
    # to cited_by). Scoped to the ## References section only (Pitfall 7).
    for citing_path in cited_by:
        abs_citing = _resolve_citing_path(citing_path, vault_root)
        if abs_citing is None:
            continue
        try:
            citing_content = abs_citing.read_text(encoding="utf-8")
        except OSError:
            continue

        refs_match = re.search(
            r"\n## References\n([\s\S]*?)(?=\n## |\Z)", citing_content
        )
        if not refs_match:
            # No \n## References marker — skip rather than corrupt (Pitfall 7 intent)
            continue

        refs_section = refs_match.group(0)
        section_lines = refs_section.split("\n")
        changed = False
        for i, line in enumerate(section_lines):
            rstripped = line.rstrip()
            if anchor and rstripped.endswith(anchor):
                # Exact identity via the invisible anchor (CR-02) — strip it,
                # then strip the miss marker it was appended after.
                candidate = rstripped[: -len(anchor)]
                anchor_matched = True
            elif rstripped.endswith(_STUB_MISS_MARKER):
                # Legacy line without an anchor — candidate only; identity is
                # confirmed below by EXACT normalized-title equality.
                candidate = rstripped
                anchor_matched = False
            else:
                continue
            if not candidate.endswith(_STUB_MISS_MARKER):
                continue
            display_and_prefix = candidate[: -len(_STUB_MISS_MARKER)]
            sep_idx = display_and_prefix.find(". ")
            if sep_idx == -1:
                continue
            prefix = display_and_prefix[: sep_idx + 2]
            display = display_and_prefix[sep_idx + 2 :]
            if not anchor_matched:
                # CR-02: exact equality only — NEVER substring containment,
                # which false-positives across unrelated references.
                display_norm = _normalize_title_for_dedup(display)
                if not stub_norm or display_norm != stub_norm:
                    continue
            section_lines[i] = f"{prefix}[[{full_link_title}]]"
            changed = True

        if not changed:
            continue

        updated_refs_section = "\n".join(section_lines)
        updated = (
            citing_content[: refs_match.start()]
            + updated_refs_section
            + citing_content[refs_match.end() :]
        )
        vault_relative = abs_citing.relative_to(vault_root).as_posix()
        create_note(vault_relative, updated, config, overwrite=True)

    # Delete stub file (real filesystem unlink — never via create_note)
    try:
        stub_abs.unlink()
    except OSError as e:
        _log(f"[biblio warning: could not delete stub {stub_path}: {e}]")


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
