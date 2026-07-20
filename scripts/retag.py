"""
scripts/retag.py — Non-destructive additive retag (D-06..D-09, Phase 08.1).

New leaf module: nothing imports it, so it may import from BOTH scripts.note
and scripts.link with zero circular-import risk (link.py already imports
FROM note.py, so putting this logic in either of those two modules would
either create a cycle or conflate two very different concerns).

Re-tags an already-ingested paper along a new axis (e.g. "methods-focused
tagging") by suggesting fresh topic slugs via the same Ollama analysis call
note.py's own topics field uses, then MERGING them additively into the
paper's existing tags -- an existing tag is never removed, only new
distinct slugs are appended (D-06).

Keeps the PaperJSON cache (``analysis.topics``) and the LIVE note
frontmatter's ``tags:`` block in agreement (D-08) by writing both in the
same run. Chains ``scripts.link.run_link`` afterward, best-effort/non-fatal,
so the new tags surface in Topics/ without a manual second step (D-07).

Public API:
  run_retag(paperjson, config, context=None, paperjson_path=None) -> str
      Returns the paper's vault-relative note path on success, or a
      "[retag error: ...]" string when the paper's note cannot be located.
      Never raises past this boundary except a genuinely unreachable Ollama
      instance, matching note._analysis_call's own fail-fast contract for
      that one case.
  main() -- CLI entry point (--stem/--paperjson/--context).

Run standalone:
  python scripts/retag.py --stem <stem> [--context "methods-focused tagging"]
"""

import argparse
import datetime
import json
import os
import pathlib
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.ingest import _load_config
from scripts.obsidian_cli import create_note, note_exists, _vault_root
from scripts import link
from scripts import note


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Emit a timestamped progress line to stderr."""
    print(
        f"[retag {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_retag(
    paperjson: dict,
    config: dict,
    context: str | None = None,
    paperjson_path: str | None = None,
) -> str:
    """Additively re-tag a paper along a new axis.

    Reads existing tags from the LIVE note (D-02 precedent — the note on
    disk, never the registry/cache, is the source of truth), suggests new
    topic slugs via the same generic AnalysisTopics call note.py's own
    topics field uses (optionally folding a free-text ``context`` into the
    prompt so the model tags along a requested axis), canonicalizes them
    against the live vault vocabulary, then merges additively — an existing
    tag is never dropped, only new distinct slugs are appended (D-06).

    Writes the merged tag list to BOTH ``paperjson["analysis"]["topics"]``
    (persisted to the PaperJSON cache when ``paperjson_path`` is given) and
    the live note's frontmatter (D-08), keeping the two in agreement. The
    frontmatter write is the ONLY vault write and routes through
    ``obsidian_cli.create_note(..., overwrite=True)`` — never a raw
    ``open()``/``Path.write_text()``. Finally best-effort chains
    ``link.run_link`` so the new tags surface in Topics/ (D-07); a failure
    there is logged and swallowed, never aborting the retag.

    Args:
        paperjson: PaperJSON v2 dict; ``extraction.metadata.title`` locates
            the live note, ``extraction.sections`` feed the tag-suggestion
            prompt via ``note._assemble_paper_text``.
        config: Config dict with at least ``"vault_path"``.
        context: Optional free text folded into the tag-suggestion prompt so
            the model can tag along a requested axis (e.g. "methods-focused
            tagging"). Blank/None runs the same generic topics prompt
            note.py's own analysis pass uses.
        paperjson_path: Optional path to the PaperJSON cache file this dict
            was loaded from. When given, the merged ``analysis.topics`` is
            persisted back to that cache file (D-08 cache half) — mirrors
            ``note.generate_note``'s own ``paperjson_path`` contract. When
            None (e.g. an in-process caller with no cache file on disk), no
            cache write happens; the caller owns persisting the mutated
            ``paperjson`` dict itself.

    Returns:
        The vault-relative note path on success, or a
        "[retag error: ...]" string when the paper's note cannot be located.
    """
    metadata = paperjson.get("extraction", {}).get("metadata", {}) or {}
    title = metadata.get("title") or ""
    note_path = f"Papers/{note._sanitize_filename(title)}.md"

    if not note_exists(note_path, config):
        return f"[retag error: paper note not found: {note_path}]"

    vault_root = _vault_root(config)
    note_content = (vault_root / note_path).read_text(encoding="utf-8")

    # D-02 precedent: the LIVE note's tags are the source of truth, never the
    # registry/cache/analysis.topics as it stood before this run.
    existing_tags = link._parse_tags(note_content)

    paper_text = note._assemble_paper_text(paperjson)

    system = (
        "You are a scientific paper analyst re-tagging an already-ingested "
        "paper along a NEW axis, in addition to its existing tags. "
        "Extract research topics/tags relevant to this paper, focused on "
        "the requested angle when one is given. Use concise topic names "
        "suitable for Obsidian tags. Return ONLY valid JSON matching the "
        "schema."
    )
    if context:
        prompt = (
            f"Re-tag the following research paper with a focus on: "
            f"{context}\n\n{paper_text}"
        )
    else:
        prompt = (
            f"List the research topics covered in the following paper:\n\n"
            f"{paper_text}"
        )

    suggested = note._analysis_call(prompt, system, note.AnalysisTopics, paper_text)
    proposed = suggested.topics if suggested is not None else []

    # Fail-open canonicalizer (quick 260714-dpl) — merges true synonyms into
    # the existing live-vault canonical slug (D-08 canonicalization half).
    canonicalized = note._canonicalize_tags(proposed, config)

    # Additive-only merge (D-06): never drop an existing tag, only append
    # distinct new slugs.
    merged = existing_tags + [t for t in canonicalized if t not in existing_tags]

    paperjson.setdefault("analysis", {})
    paperjson["analysis"]["topics"] = merged

    if paperjson_path:
        try:
            cache_file = pathlib.Path(paperjson_path)
            note._write_paperjson_cache(
                paperjson, cache_file.stem, cache_dir=str(cache_file.parent)
            )
        except Exception as e:
            _log(
                f"[retag warning: failed to persist analysis to cache "
                f"{paperjson_path}: {e}]"
            )

    # D-08 note half: patch ONLY the tags: block, then write through the
    # sole vault-I/O chokepoint.
    patched = note._patch_frontmatter_fields(note_content, tag_replacement=merged)
    create_note(note_path, patched, config, overwrite=True)
    _log(f"wrote retagged note to {note_path}")

    # D-07: best-effort, non-fatal link chain. run_link re-reads the live
    # note's tags itself, so calling it AFTER the write above is sufficient.
    try:
        link.run_link(paperjson, config)
    except Exception as e:
        _log(f"[retag warning: link chaining failed: {e}]")

    return note_path


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Standalone CLI entry point (factored out of
    ``if __name__ == "__main__"``, mirroring ``scripts/link.py``'s
    ``main()`` shape) so it is callable in-process from tests without a real
    subprocess touching the repo-root ``config.json``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Additively re-tag an already-ingested paper along a new axis "
            "(D-06..D-09). Keeps the PaperJSON cache and live note "
            "frontmatter tags in agreement, then best-effort chains link.py."
        )
    )
    parser.add_argument(
        "--paperjson", default=None,
        help="Path to the PaperJSON v2 cache file (JSON).",
    )
    parser.add_argument(
        "--stem", default=None,
        help=(
            "PDF/URL filename stem -- resolves "
            "<paperjson_cache_dir>/<stem>.json when --paperjson is omitted."
        ),
    )
    parser.add_argument(
        "--context", default=None,
        help='Optional free text focusing the retag axis, e.g. "methods-focused tagging".',
    )
    args = parser.parse_args()

    if not (args.paperjson or args.stem):
        parser.error("either --paperjson or --stem is required")

    try:
        config = _load_config()

        if args.paperjson:
            pj_path = args.paperjson
        else:
            pj_path = note._resolve_paperjson_path(
                args.stem, config.get("paperjson_cache_dir", ".paperjson_cache")
            )

        with open(pj_path, encoding="utf-8") as f:
            paperjson = json.load(f)

        result = run_retag(paperjson, config, context=args.context, paperjson_path=pj_path)
        if isinstance(result, str) and result.startswith("[retag error:"):
            print(result, file=sys.stderr)
            sys.exit(1)
        print(result)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[retag error: {e}]", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Windows cp1252 guard — mirrors note.py/link.py/biblio.py/embed.py.
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
