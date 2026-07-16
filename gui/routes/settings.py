"""Settings page -- typed config form, atomic save, D-20 data-path confirm
flow (Phase 8 Plan 04, D-19/D-20).

Module-level ``CONFIG_PATH`` (repo-root ``config.json``) is deliberately a
plain module attribute -- tests monkeypatch it to a ``tmp_path`` file so the
suite never risks a real write at repo root (``tests/conftest.py``'s
``_guard_real_config_json`` session guard is the safety net).

``FIELDS`` is the single source of truth for the typed form: all 24 keys from
``config.example.json`` (the 22 documented keys plus the two Phase 8 adds,
``gui_port``/``uningested_dir``), each carrying its Python type, whether it is
one of the six D-20 data-path keys, and any special label
(``restart_required`` for ``gui_port``, ``legacy`` for ``obsidian_exe`` --
confirmed dead config via ``grep -rn "obsidian_exe" scripts/`` at plan time).

Save flow (POST /settings):
  1. Coerce each submitted value against its FIELDS type. A coercion failure
     (e.g. letters in a number field) re-renders the form with a field error
     and writes nothing.
  2. For each changed D-20 data-path key, check whether the OLD resolved
     location exists and is non-empty. If so, return the D-20 confirm
     partial (no write yet) -- the GUI never migrates data, it only
     repoints the config value (D-20 hard constraint).
  3. Once every pending confirmation is satisfied (``confirmed=true``
     resubmission) -- or there was nothing to confirm -- write the full
     config dict via the same atomic temp-file + ``os.replace`` shape as
     ``scripts/obsidian_cli.py::create_note``.
"""

import json
import os
import pathlib
from typing import Optional

from fastapi import APIRouter, Form, Request
from scripts.ingest import _read_registry

router = APIRouter()

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# Monkeypatchable by tests (tests/conftest.py's _guard_real_config_json is
# the session-level safety net if a test forgets to redirect this).
CONFIG_PATH = _REPO_ROOT / "config.json"
_EXAMPLE_CONFIG_PATH = _REPO_ROOT / "config.example.json"

_DATA_PATH_KEYS = (
    "registry_path",
    "vault_path",
    "chroma_db_path",
    "mineru_output_dir",
    "paperjson_cache_dir",
    "uningested_dir",
)

# All 24 keys config.json accepts (22 documented in config.example.json plus
# the two Phase 8 adds, gui_port/uningested_dir), each with its render type,
# D-20 flag, description, and any special label.
FIELDS = [
    {
        "key": "registry_path", "type": "str", "is_data_path": True,
        "label": "Registry Path", "restart_required": False, "legacy": False,
        "description": "Shared papers registry JSON -- every clone points at the same file so an already-ingested paper is never re-processed.",
    },
    {
        "key": "vault_path", "type": "str", "is_data_path": True,
        "label": "Vault Path", "restart_required": False, "legacy": False,
        "description": "This project's Obsidian vault directory.",
    },
    {
        "key": "project_name", "type": "str", "is_data_path": False,
        "label": "Project Name", "restart_required": False, "legacy": False,
        "description": "Name for this project clone, recorded on registry entries' projects[] field.",
    },
    {
        "key": "mineru_path", "type": "str", "is_data_path": False,
        "label": "MinerU Path", "restart_required": False, "legacy": False,
        "description": "Path to the MinerU executable (its own separate Python environment).",
    },
    {
        "key": "mineru_timeout", "type": "int", "is_data_path": False,
        "label": "MinerU Timeout (seconds)", "restart_required": False, "legacy": False,
        "description": "Max seconds to wait for MinerU PDF structure extraction.",
    },
    {
        "key": "obsidian_exe", "type": "str", "is_data_path": False,
        "label": "Obsidian Executable", "restart_required": False, "legacy": True,
        "description": "legacy — unused by any script (vault writes are direct atomic file I/O, not a subprocess to Obsidian).",
    },
    {
        "key": "defuddle_path", "type": "str", "is_data_path": False,
        "label": "Defuddle Path", "restart_required": False, "legacy": False,
        "description": "Path to the defuddle Node executable (web-page ingestion).",
    },
    {
        "key": "web_min_body_chars", "type": "int", "is_data_path": False,
        "label": "Web Min Body Chars", "restart_required": False, "legacy": False,
        "description": "Minimum extracted body length for a web-ingested page to be considered valid.",
    },
    {
        "key": "defuddle_timeout", "type": "int", "is_data_path": False,
        "label": "Defuddle Timeout (seconds)", "restart_required": False, "legacy": False,
        "description": "Max seconds to wait for defuddle web-page extraction.",
    },
    {
        "key": "chroma_db_path", "type": "str", "is_data_path": True,
        "label": "ChromaDB Path", "restart_required": False, "legacy": False,
        "description": "This project's vector store directory.",
    },
    {
        "key": "mineru_output_dir", "type": "str", "is_data_path": True,
        "label": "MinerU Output Dir", "restart_required": False, "legacy": False,
        "description": "Shared MinerU extraction output cache (sibling across clones).",
    },
    {
        "key": "paperjson_cache_dir", "type": "str", "is_data_path": True,
        "label": "PaperJSON Cache Dir", "restart_required": False, "legacy": False,
        "description": "Shared PaperJSON cache (sibling across clones).",
    },
    {
        "key": "embed_model", "type": "str", "is_data_path": False,
        "label": "Embed Model", "restart_required": False, "legacy": False,
        "description": "Ollama embedding model name (nomic-embed-text).",
    },
    {
        "key": "embed_timeout", "type": "int", "is_data_path": False,
        "label": "Embed Timeout (seconds)", "restart_required": False, "legacy": False,
        "description": "Max seconds to wait per embedding call.",
    },
    {
        "key": "embed_batch_size", "type": "int", "is_data_path": False,
        "label": "Embed Batch Size", "restart_required": False, "legacy": False,
        "description": "Number of section chunks embedded per Ollama call.",
    },
    {
        "key": "embed_section_max_tokens", "type": "int", "is_data_path": False,
        "label": "Embed Section Max Tokens", "restart_required": False, "legacy": False,
        "description": "Max tokens per section before it is split for embedding.",
    },
    {
        "key": "crossref_validate", "type": "bool", "is_data_path": False,
        "label": "Crossref Validate", "restart_required": False, "legacy": False,
        "description": "Enable Crossref network lookups for reference enrichment (off keeps the pipeline fully offline, D-13).",
    },
    {
        "key": "crossref_contact_email", "type": "str", "is_data_path": False,
        "label": "Crossref Contact Email", "restart_required": False, "legacy": False,
        "description": "Contact email sent with Crossref API requests (polite-pool eligibility).",
    },
    {
        "key": "crossref_timeout", "type": "int", "is_data_path": False,
        "label": "Crossref Timeout (seconds)", "restart_required": False, "legacy": False,
        "description": "Max seconds to wait per Crossref API call.",
    },
    {
        "key": "crossref_retries", "type": "int", "is_data_path": False,
        "label": "Crossref Retries", "restart_required": False, "legacy": False,
        "description": "Retry attempts for transient Crossref network errors.",
    },
    {
        "key": "ollama_num_ctx_cap", "type": "int", "is_data_path": False,
        "label": "Ollama Context Cap", "restart_required": False, "legacy": False,
        "description": "Max context window tokens (VRAM budget ladder -- RTX 4060 8GB default 65536).",
    },
    {
        "key": "ollama_section_timeout", "type": "int", "is_data_path": False,
        "label": "Ollama Section Timeout (seconds)", "restart_required": False, "legacy": False,
        "description": "Max seconds to wait per LLM section-fill call.",
    },
    {
        "key": "gui_port", "type": "int", "is_data_path": False,
        "label": "GUI Port", "restart_required": True, "legacy": False,
        "description": "Port the local web GUI binds to (127.0.0.1 only) -- requires GUI restart to take effect.",
    },
    {
        "key": "uningested_dir", "type": "str", "is_data_path": True,
        "label": "Uningested Dir", "restart_required": False, "legacy": False,
        "description": "Drop-folder directory scanned for PDFs waiting to be ingested.",
    },
]

_FIELDS_BY_KEY = {f["key"]: f for f in FIELDS}


# ---------------------------------------------------------------------------
# Config load (fallback to config.example.json defaults for absent keys)
# ---------------------------------------------------------------------------

def _load_current_values() -> dict:
    defaults = json.loads(_EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))
    current = {}
    if CONFIG_PATH.exists():
        try:
            current = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = {}
    values = dict(defaults)
    values.update(current)
    return values


# ---------------------------------------------------------------------------
# D-20: does the OLD resolved location for a data-path key hold anything?
# Read-only -- never touches the location itself (the GUI only repoints the
# config value, it never migrates data).
# ---------------------------------------------------------------------------

def _old_location_info(key: str, old_value: str) -> dict:
    if not old_value:
        return {"exists_nonempty": False, "count": 0}

    resolved = pathlib.Path(old_value).expanduser()

    if key == "registry_path":
        if not resolved.exists():
            return {"exists_nonempty": False, "count": 0}
        try:
            registry = _read_registry(str(resolved))
        except ValueError:
            # Corrupt-but-present file: something is there, exact count unknown.
            return {"exists_nonempty": True, "count": 0}
        return {"exists_nonempty": len(registry) > 0, "count": len(registry)}

    if resolved.is_file():
        size = resolved.stat().st_size
        return {"exists_nonempty": size > 0, "count": 1 if size > 0 else 0}

    if resolved.is_dir():
        items = list(resolved.iterdir())
        return {"exists_nonempty": len(items) > 0, "count": len(items)}

    return {"exists_nonempty": False, "count": 0}


def _coerce(raw: dict) -> tuple[dict, dict]:
    """Coerce raw (all-string, plus None for unchecked checkboxes) form values
    against FIELDS types. Returns (coerced_values, errors)."""
    coerced = {}
    errors = {}
    for field in FIELDS:
        key = field["key"]
        submitted = raw.get(key)
        if field["type"] == "bool":
            coerced[key] = bool(submitted)
            continue
        if field["type"] == "int":
            try:
                coerced[key] = int(str(submitted).strip())
            except (TypeError, ValueError):
                errors[key] = f"{field['label']} must be a whole number."
            continue
        coerced[key] = submitted or ""
    return coerced, errors


def _settings_response(request: Request, values: dict, errors: dict, saved: bool = False):
    from gui.app import templates

    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "active_page": "settings",
            "page_title": "Settings",
            "fields": FIELDS,
            "values": values,
            "errors": errors,
            "saved": saved,
        },
    )


@router.get("/settings")
def settings_page(request: Request):
    return _settings_response(request, _load_current_values(), {})


@router.post("/settings")
def settings_save(
    request: Request,
    registry_path: str = Form(""),
    vault_path: str = Form(""),
    project_name: str = Form(""),
    mineru_path: str = Form(""),
    mineru_timeout: str = Form(""),
    obsidian_exe: str = Form(""),
    defuddle_path: str = Form(""),
    web_min_body_chars: str = Form(""),
    defuddle_timeout: str = Form(""),
    chroma_db_path: str = Form(""),
    mineru_output_dir: str = Form(""),
    paperjson_cache_dir: str = Form(""),
    embed_model: str = Form(""),
    embed_timeout: str = Form(""),
    embed_batch_size: str = Form(""),
    embed_section_max_tokens: str = Form(""),
    crossref_validate: Optional[str] = Form(None),
    crossref_contact_email: str = Form(""),
    crossref_timeout: str = Form(""),
    crossref_retries: str = Form(""),
    ollama_num_ctx_cap: str = Form(""),
    ollama_section_timeout: str = Form(""),
    gui_port: str = Form(""),
    uningested_dir: str = Form(""),
    confirmed: Optional[str] = Form(None),
):
    raw = {
        "registry_path": registry_path,
        "vault_path": vault_path,
        "project_name": project_name,
        "mineru_path": mineru_path,
        "mineru_timeout": mineru_timeout,
        "obsidian_exe": obsidian_exe,
        "defuddle_path": defuddle_path,
        "web_min_body_chars": web_min_body_chars,
        "defuddle_timeout": defuddle_timeout,
        "chroma_db_path": chroma_db_path,
        "mineru_output_dir": mineru_output_dir,
        "paperjson_cache_dir": paperjson_cache_dir,
        "embed_model": embed_model,
        "embed_timeout": embed_timeout,
        "embed_batch_size": embed_batch_size,
        "embed_section_max_tokens": embed_section_max_tokens,
        "crossref_validate": crossref_validate,
        "crossref_contact_email": crossref_contact_email,
        "crossref_timeout": crossref_timeout,
        "crossref_retries": crossref_retries,
        "ollama_num_ctx_cap": ollama_num_ctx_cap,
        "ollama_section_timeout": ollama_section_timeout,
        "gui_port": gui_port,
        "uningested_dir": uningested_dir,
    }

    current_values = _load_current_values()
    new_values, errors = _coerce(raw)

    if errors:
        # Re-render with the raw submitted text so the user sees exactly what
        # they typed next to the error -- write nothing.
        display_values = dict(current_values)
        display_values.update(raw)
        return _settings_response(request, display_values, errors)

    pending = []
    for key in _DATA_PATH_KEYS:
        old_val = current_values.get(key, "")
        new_val = new_values.get(key, "")
        if old_val == new_val:
            continue
        info = _old_location_info(key, old_val)
        if info["exists_nonempty"]:
            pending.append({
                "key": key,
                "label": _FIELDS_BY_KEY[key]["label"],
                "old_path": old_val,
                "count": info["count"],
            })

    if pending and confirmed != "true":
        from gui.app import templates

        return templates.TemplateResponse(
            request,
            "partials/settings_confirm.html",
            {"pending": pending, "form_values": raw},
        )

    # Write: temp file + os.replace (same atomic shape as
    # scripts/obsidian_cli.py::create_note). The GUI never touches the data
    # at either the old or new data-path location -- it only repoints the
    # config value (D-20 hard constraint).
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(CONFIG_PATH) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(new_values, f, indent=2)
    os.replace(tmp_path, str(CONFIG_PATH))

    return _settings_response(request, new_values, {}, saved=True)
