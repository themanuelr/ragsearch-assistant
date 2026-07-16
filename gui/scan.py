"""gui/scan.py — Live pipeline-state scan (D-09), Phase 8 Plan 02.

Strictly read-only. Never writes a GUI-owned state file (D-09 — the
"ghost-state bug class" Phases 4/6 already fought) and never re-derives a
write-side detection contract: imports ``scripts.biblio._REFS_SECTION_RE``,
``scripts.link._RELATED_TOPICS_RE``, ``scripts.embed._get_collection``,
``scripts.ingest._read_registry``, and ``scripts.note._sanitize_filename``
directly (Don't Hand-Roll, 08-RESEARCH.md Pattern 4).

Public API:
  scan_project_papers(config) -> list[dict]
      One dict per registry paper whose ``projects[]`` contains
      ``config["project_name"]``.
  scan_paper(config, registry_key) -> dict | None
      Full stage/metadata/cache-path dict for one registry key, or ``None``
      if the key is not present in the registry.
  scan_uningested(config) -> list[dict]
      ``*.pdf`` files directly inside ``config["uningested_dir"]``
      (non-recursive), sorted by name. Missing directory returns ``[]``
      without creating it.
  scan_universal(config) -> list[dict]
      One dict per registry entry, regardless of ``projects[]`` membership
      (Phase 8 Plan 06, D-11) -- title/year/journal/projects,
      ``in_this_project``, cache-presence booleans (mineru/paperjson), and
      the vault-note relative path (or ``None`` if no note exists yet).
"""

import json
import pathlib

from scripts.biblio import _REFS_SECTION_RE
from scripts.embed import _get_collection
from scripts.ingest import _read_registry
from scripts.link import _RELATED_TOPICS_RE, _parse_tags
from scripts.note import _sanitize_filename

# Analysis-namespace fields that indicate the PaperJSON's analysis has been
# filled by the LLM cascade (as opposed to the empty skeleton written at
# ingest time — see scripts/ingest.py::_assemble_paperjson's `analysis` dict).
_ANALYSIS_POPULATED_KEYS = (
    "generated_by",
    "summary",
    "claims",
    "methods_overview",
    "results",
    "limitations",
    "open_questions",
    "entities",
    "topics",
)


def _stage_mineru(config: dict, stem: str) -> bool:
    """True if any content_list.json exists under mineru_output_dir/<stem>/.

    Recursive glob mirrors scripts/ingest.py::_find_content_list, which looks
    for a MinerU-produced content_list.json anywhere under the per-paper
    output directory (canonical path nests it under a backend-name
    subdirectory, e.g. hybrid_auto/ or auto/).
    """
    mineru_output_dir = config.get("mineru_output_dir") or ".mineru_output"
    stem_dir = pathlib.Path(mineru_output_dir) / stem
    return next(stem_dir.glob("**/*content_list.json"), None) is not None


def _stage_paperjson_fill(paperjson_path: pathlib.Path) -> bool | None:
    """True if paperjson_path exists and its analysis namespace is populated.

    False when the file is absent or the analysis namespace is still the
    empty skeleton. None (unknown) when the file exists but cannot be parsed.
    """
    if not paperjson_path.exists():
        return False
    try:
        with open(paperjson_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    analysis = data.get("analysis") or {}
    return any(analysis.get(key) for key in _ANALYSIS_POPULATED_KEYS)


def _stage_embed(config: dict, registry_key: str) -> bool | None:
    """True/False if the papers collection has (or lacks) an entry for this
    key, None (unknown) if Chroma is unreachable/unavailable.

    Guarded so a missing chroma_db_path never creates one (read-only, D-09)
    and never 500s the page — chromadb.PersistentClient's get_or_create
    would otherwise materialize the directory on first touch.
    """
    chroma_db_path = config.get("chroma_db_path")
    if not chroma_db_path or not pathlib.Path(chroma_db_path).exists():
        return None
    try:
        collection = _get_collection(config)
        result = collection.get(
            where={"$and": [{"registry_key": registry_key}, {"status": "paper"}]}
        )
        return bool(result.get("ids"))
    except Exception:  # noqa: BLE001 - fail-open: unknown, never a 500 (D-09)
        return None


def _note_paths(config: dict, title: str) -> tuple[pathlib.Path, str]:
    """Return (absolute note path, vault-relative note path) for a title."""
    vault_path = config.get("vault_path") or ""
    filename = _sanitize_filename(title or "")
    rel_path = f"Papers/{filename}.md"
    return pathlib.Path(vault_path) / "Papers" / f"{filename}.md", rel_path


def _derive_stem(entry: dict, title: str) -> str:
    """Derive the cache stem from a registry entry's recorded
    ``paperjson_path`` (the exact field name written by
    scripts/ingest.py::_registry_entry) so cache lookups agree with what
    ingest.py actually wrote; falls back to the sanitized title only for
    entries with no recorded paperjson_path. Shared by ``_build_paper_dict``
    and ``scan_universal`` -- one stem-derivation rule for both pages
    (Don't Hand-Roll)."""
    paperjson_path_str = entry.get("paperjson_path")
    return (
        pathlib.Path(paperjson_path_str).stem
        if paperjson_path_str
        else _sanitize_filename(title)
    )


def _build_paper_dict(config: dict, registry_key: str, entry: dict) -> dict:
    """Build the full per-paper dict (metadata + cache paths + seven stages)."""
    title = entry.get("title") or ""
    stem = _derive_stem(entry, title)

    mineru_output_dir = config.get("mineru_output_dir") or ".mineru_output"
    mineru_dir = str(pathlib.Path(mineru_output_dir) / stem)

    paperjson_cache_dir = config.get("paperjson_cache_dir") or ".paperjson_cache"
    paperjson_path = pathlib.Path(paperjson_cache_dir) / f"{stem}.json"

    note_abs_path, note_rel_path = _note_paths(config, title)
    note_present = note_abs_path.is_file()

    note_content = ""
    if note_present:
        try:
            note_content = note_abs_path.read_text(encoding="utf-8")
        except OSError:
            note_content = ""

    tags = _parse_tags(note_content) if note_content else []

    stages = {
        "mineru": _stage_mineru(config, stem),
        "registry": True,
        "paperjson_fill": _stage_paperjson_fill(paperjson_path),
        "note": note_present,
        "biblio": bool(note_content) and _REFS_SECTION_RE.search(note_content) is not None,
        "embed": _stage_embed(config, registry_key),
        "topics": bool(note_content) and _RELATED_TOPICS_RE.search(note_content) is not None,
    }

    return {
        "registry_key": registry_key,
        "title": title,
        "stem": stem,
        "metadata": {
            "title": title,
            "authors": entry.get("authors"),
            "year": entry.get("year"),
            "journal": entry.get("journal"),
            "doi": entry.get("doi"),
            "tags": tags,
        },
        "cache_paths": {
            "mineru_dir": mineru_dir,
            "paperjson_path": str(paperjson_path),
            "vault_note": note_rel_path,
        },
        "stages": stages,
    }


def scan_project_papers(config: dict) -> list:
    """Return one dict per registry paper belonging to this project.

    A paper belongs to this project when ``config["project_name"]`` is a
    member of its registry entry's ``projects[]`` list.
    """
    registry = _read_registry(config.get("registry_path") or "")
    project_name = config.get("project_name")
    papers = []
    for key, entry in registry.items():
        projects = entry.get("projects") or []
        if project_name and project_name in projects:
            papers.append(_build_paper_dict(config, key, entry))
    return papers


def scan_paper(config: dict, registry_key: str):
    """Return the full stage/metadata/cache-path dict for one registry key.

    Returns None if the key is not present in the registry (caller renders
    404).
    """
    registry = _read_registry(config.get("registry_path") or "")
    entry = registry.get(registry_key)
    if entry is None:
        return None
    return _build_paper_dict(config, registry_key, entry)


def scan_universal(config: dict) -> list:
    """Return one dict per registry entry, regardless of ``projects[]``
    membership (Phase 8 Plan 06, D-11).

    Strictly read-only: reuses ``_stage_mineru`` (recursive glob, never
    creates a path) and a plain ``pathlib.Path.exists()`` check for the
    PaperJSON cache -- no write, no ChromaDB touch, no vault write.
    """
    registry = _read_registry(config.get("registry_path") or "")
    project_name = config.get("project_name")
    paperjson_cache_dir = config.get("paperjson_cache_dir") or ".paperjson_cache"

    entries = []
    for key, entry in registry.items():
        title = entry.get("title") or ""
        stem = _derive_stem(entry, title)
        paperjson_path = pathlib.Path(paperjson_cache_dir) / f"{stem}.json"
        note_abs_path, note_rel_path = _note_paths(config, title)
        projects = entry.get("projects") or []

        entries.append(
            {
                "registry_key": key,
                "title": title,
                "year": entry.get("year"),
                "journal": entry.get("journal"),
                "projects": projects,
                "in_this_project": bool(project_name and project_name in projects),
                "mineru_present": _stage_mineru(config, stem),
                "paperjson_present": paperjson_path.exists(),
                "vault_note": note_rel_path if note_abs_path.is_file() else None,
            }
        )
    return entries


def scan_uningested(config: dict) -> list:
    """List *.pdf files directly inside config['uningested_dir'].

    Non-recursive, sorted by name. A missing directory returns [] without
    creating it (read-only, D-09/D-10).
    """
    uningested_dir = config.get("uningested_dir")
    if not uningested_dir:
        return []
    d = pathlib.Path(uningested_dir)
    if not d.is_dir():
        return []
    entries = []
    for p in sorted(d.glob("*.pdf"), key=lambda x: x.name):
        try:
            stat = p.stat()
        except OSError:
            continue
        entries.append({"name": p.name, "size": stat.st_size, "mtime": stat.st_mtime})
    return entries
