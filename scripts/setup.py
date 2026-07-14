"""
scripts/setup.py — Clone-and-go bootstrap for the Research Paper Intelligence
System (SETUP-01/02, Phase 7).

Importable, pure core: every step function below takes explicit params, does
NO print/input, and returns a structured `{"step": ..., "status": ...,
"detail": ...}` result. `run_setup` orchestrates all steps fail-open (D-14) —
a failure in one step (e.g. Ollama unreachable) never blocks the others.

The thin argparse CLI/UX wrapper lands in Plan 04; this module is built
MCP/UI-ready but exposes no MCP tool yet (D-02).

Reuses already-verified in-repo subsystems rather than re-deriving them:
  - scripts.embed._get_collection      (D-11 — no second PersistentClient)
  - scripts.ingest._resolve_mineru / _resolve_defuddle (D-10 — warn, never install)

Public API (grows across this phase's plans):
  bootstrap_vault(config) -> dict
  check_external_tools(config) -> dict
  pull_models(config, models=None) -> dict
  init_chroma(config) -> dict
  write_config(values, config_path) -> dict
  ensure_empty_registry(registry_path) -> dict
  run_setup(values, config, config_path=None) -> dict
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

OLLAMA_BASE = "http://localhost:11434"
REQUIRED_MODELS = ["nomic-embed-text", "gemma4:e4b"]


# ---------------------------------------------------------------------------
# Path safety (T-07-01, V5)
# ---------------------------------------------------------------------------

def _resolve_safe_target(path_str: str) -> dict:
    """Resolve path_str and judge whether it is safe to use as a bootstrap
    target. Pure — no write, no prompt.

    Refuses the repo root, the user's home directory, and any filesystem
    root/anchor (defense in depth; the interactive CLI in Plan 04 is the
    primary gate)."""
    resolved = pathlib.Path(path_str).expanduser().resolve()

    sensitive = {_REPO_ROOT, pathlib.Path.home().resolve()}
    # Filesystem roots/anchors (e.g. "/" on POSIX, "C:\\" on Windows).
    if resolved.anchor:
        sensitive.add(pathlib.Path(resolved.anchor).resolve())

    if resolved in sensitive:
        return {
            "resolved": resolved,
            "safe": False,
            "reason": f"{resolved} is a sensitive location (repo root, home, or filesystem root) — refusing to use it as a bootstrap target",
        }
    return {"resolved": resolved, "safe": True, "reason": ""}


# ---------------------------------------------------------------------------
# bootstrap_vault (SETUP-01/04)
# ---------------------------------------------------------------------------

def bootstrap_vault(config: dict) -> dict:
    """Copy obsidian-vault-template/ into config["vault_path"], renaming
    dot_obsidian/ -> .obsidian/ on copy (Pitfall 6), skipping any destination
    file that already exists (D-13, Pitfall 2 idempotency). Never overwrites."""
    target = _resolve_safe_target(config.get("vault_path", ""))
    if not target["safe"]:
        return {"step": "bootstrap_vault", "status": "error", "detail": target["reason"]}

    vault_root = target["resolved"]
    template_root = _REPO_ROOT / "obsidian-vault-template"

    created = []
    skipped = []

    for src in template_root.rglob("*"):
        rel_parts = list(src.relative_to(template_root).parts)
        # Rename-on-copy: dot_obsidian/... -> .obsidian/... (Pitfall 6).
        if rel_parts and rel_parts[0] == "dot_obsidian":
            rel_parts[0] = ".obsidian"
        dest = vault_root.joinpath(*rel_parts)

        if src.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            skipped.append(str(dest))
            continue
        shutil.copy2(src, dest)
        created.append(str(dest))

    return {
        "step": "bootstrap_vault",
        "status": "ok",
        "detail": {"created": created, "skipped": skipped},
    }


# ---------------------------------------------------------------------------
# check_external_tools (D-10 — warn-only, never install)
# ---------------------------------------------------------------------------

def check_external_tools(config: dict) -> dict:
    """Warn (never fail, never install) when MinerU or defuddle can't be
    resolved via config or PATH (D-10 — MinerU lives in a separate env)."""
    from scripts.ingest import _resolve_defuddle, _resolve_mineru  # noqa: PLC0415

    warnings = []
    if not _resolve_mineru(config):
        warnings.append(
            "MinerU not found (config.mineru_path unset and not on PATH) — "
            "see README Setup step 4 (separate Python 3.10-3.13 env required)"
        )
    if not _resolve_defuddle(config):
        warnings.append(
            "defuddle not found (config.defuddle_path unset and not on PATH) — "
            "run: npm install -g defuddle"
        )

    return {
        "step": "check_external_tools",
        "status": "ok" if not warnings else "warning",
        "detail": warnings,
    }


# ---------------------------------------------------------------------------
# Ollama model presence / pull (D-09/D-13/D-14, Pitfall 1)
# ---------------------------------------------------------------------------

def _ollama_model_names() -> list | None:
    """Returns installed model names (with :tag suffix), or None if Ollama
    is unreachable."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except urllib.error.URLError:
        return None


def _model_present(model: str, installed: list) -> bool:
    """Compare base repo names only, normalizing the ':tag' suffix mismatch
    (Pitfall 1 — /api/tags echoes 'nomic-embed-text:latest' for a bare pull)."""
    base = model.split(":")[0]
    return any(name.split(":")[0] == base for name in installed)


def pull_models(config: dict, models: list | None = None) -> dict:
    """Pull each of `models` (default REQUIRED_MODELS) via list-form argv
    subprocess (T-07-02, shell=False). Fail-open: returns a structured error
    result (never raises) when Ollama is unreachable (D-14)."""
    if models is None:
        models = REQUIRED_MODELS

    installed = _ollama_model_names()
    if installed is None:
        return {
            "step": "pull_models",
            "status": "error",
            "detail": f"Ollama unreachable at {OLLAMA_BASE}",
        }

    results = []
    for model in models:
        if _model_present(model, installed):
            results.append({"model": model, "status": "already_present"})
            continue
        try:
            proc = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True,
                timeout=1800,
                shell=False,
            )
            results.append(
                {
                    "model": model,
                    "status": "pulled" if proc.returncode == 0 else "error",
                    "detail": proc.stderr[-500:] if proc.returncode else None,
                }
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            results.append({"model": model, "status": "error", "detail": str(e)})

    return {"step": "pull_models", "status": "ok", "detail": results}


# ---------------------------------------------------------------------------
# init_chroma (D-11 — delegates to embed._get_collection, no second client)
# ---------------------------------------------------------------------------

def init_chroma(config: dict) -> dict:
    """Delegate ChromaDB bootstrap to scripts.embed._get_collection so both
    modules share the same PersistentClient path anchoring and cosine
    'papers' collection (D-11)."""
    from scripts.embed import _get_collection  # noqa: PLC0415

    try:
        _get_collection(config)
        return {
            "step": "init_chroma",
            "status": "ok",
            "detail": f"collection ready at {config.get('chroma_db_path')}",
        }
    except Exception as e:
        return {"step": "init_chroma", "status": "error", "detail": str(e)}


# ---------------------------------------------------------------------------
# write_config / ensure_empty_registry (D-12/D-13, T-07-03 — never overwrite)
# ---------------------------------------------------------------------------

def write_config(values: dict, config_path: pathlib.Path) -> dict:
    """Write values as config.json, never overwriting an existing file
    (D-13 security invariant)."""
    config_path = pathlib.Path(config_path)
    if config_path.exists():
        return {
            "step": "write_config",
            "status": "skipped",
            "detail": f"{config_path} already exists — not overwritten (D-13)",
        }
    config_path.write_text(json.dumps(values, indent=2), encoding="utf-8")
    return {"step": "write_config", "status": "ok", "detail": f"wrote {config_path}"}


def ensure_empty_registry(registry_path: pathlib.Path) -> dict:
    """Create an empty papers_registry.json ("{}") if absent; never
    overwrite an existing registry (D-12/D-13)."""
    registry_path = pathlib.Path(registry_path)
    if registry_path.exists():
        return {
            "step": "ensure_registry",
            "status": "skipped",
            "detail": f"{registry_path} already exists",
        }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("{}", encoding="utf-8")
    return {
        "step": "ensure_registry",
        "status": "ok",
        "detail": f"created empty registry at {registry_path}",
    }


# ---------------------------------------------------------------------------
# run_setup orchestrator (D-14 — fail-open aggregation)
# ---------------------------------------------------------------------------

def run_setup(values: dict, config: dict, config_path: pathlib.Path | None = None) -> dict:
    """Run every bootstrap step fail-open: a raised exception in any step is
    caught and converted to a {step, status: "error", detail} entry rather
    than aborting the run, so e.g. an unreachable Ollama never blocks vault/
    config/registry provisioning (D-14).

    `values` supplies the user-chosen vault_path/registry_path/project_name
    (and any other config.example.json keys) written into config.json;
    `config` is the effective config dict (usually == values) used to drive
    bootstrap_vault/check_external_tools/pull_models/init_chroma.
    """
    if config_path is None:
        config_path = _REPO_ROOT / "config.json"
    config_path = pathlib.Path(config_path)

    registry_path = config.get("registry_path") or values.get("registry_path")

    steps = []

    def _run_step(name, fn, *args):
        try:
            result = fn(*args)
        except Exception as e:
            result = {"step": name, "status": "error", "detail": str(e)}
        steps.append(result)

    _run_step("bootstrap_vault", bootstrap_vault, config)
    _run_step("check_external_tools", check_external_tools, config)
    _run_step("pull_models", pull_models, config)
    _run_step("init_chroma", init_chroma, config)
    _run_step("write_config", write_config, values, config_path)
    _run_step("ensure_registry", ensure_empty_registry, pathlib.Path(registry_path).expanduser())

    summary = {"ok": 0, "skipped": 0, "warning": 0, "error": 0}
    for step in steps:
        status = step.get("status", "error")
        summary[status] = summary.get(status, 0) + 1

    return {"steps": steps, "summary": summary}


# ---------------------------------------------------------------------------
# Thin argparse CLI front-end (D-01/D-03) -- print/input live ONLY here and
# in the two helpers below. All orchestration stays in run_setup; this
# section only prompts, validates paths, calls the core, and renders.
# ---------------------------------------------------------------------------

def _prompt(label: str, default: str) -> str:
    """Prompt for a value with an Enter-through default (D-03)."""
    raw = input(f"{label} [{default}]: ").strip()
    return raw or default


def _print_summary(result: dict) -> None:
    """Pretty-print the aggregated run_setup per-step summary."""
    print("\nSetup summary:")
    for step in result["steps"]:
        status = step.get("status", "error")
        detail = step.get("detail")
        print(f"  [{status.upper():>7}] {step.get('step')}: {detail}")
    counts = result["summary"]
    print(
        f"\n{counts.get('ok', 0)} ok, {counts.get('skipped', 0)} skipped, "
        f"{counts.get('warning', 0)} warning, {counts.get('error', 0)} error"
    )


def _refuse_sensitive(label: str, path_str: str, force: bool) -> pathlib.Path:
    """Run _resolve_safe_target on a user-supplied path (V5/T-07-01). Aborts
    the process unless --force is given; returns the resolved path either
    way so callers can chain further checks."""
    target = _resolve_safe_target(path_str)
    if not target["safe"]:
        if not force:
            print(f"ERROR: {label} refused -- {target['reason']}", file=sys.stderr)
            print("Pass --force to override this safety check.", file=sys.stderr)
            sys.exit(1)
        print(f"WARNING: {label} -- {target['reason']} (proceeding due to --force)")
    return target["resolved"]


def _validate_vault_target(vault_path_str: str, force: bool) -> pathlib.Path:
    """Refuse a sensitive vault_path (via _refuse_sensitive) AND refuse a
    resolved directory that is already non-empty and does not look like an
    existing vault (no .obsidian/ and no Papers/) -- unless --force is given
    (V5/T-07-01)."""
    resolved = _refuse_sensitive("--vault-path", vault_path_str, force)

    if resolved.exists() and resolved.is_dir() and any(resolved.iterdir()):
        is_vault = (resolved / ".obsidian").exists() or (resolved / "Papers").exists()
        if not is_vault:
            if not force:
                print(
                    f"ERROR: --vault-path '{resolved}' is a non-empty directory that does "
                    "not look like an existing vault (no .obsidian/ or Papers/ found) -- "
                    "refusing to use it as a bootstrap target.",
                    file=sys.stderr,
                )
                print("Pass --force to override this safety check.", file=sys.stderr)
                sys.exit(1)
            print(
                f"WARNING: --vault-path '{resolved}' is non-empty and not a recognized "
                "vault (proceeding due to --force)"
            )
    return resolved


if __name__ == "__main__":
    # Windows cp1252 guard: wrap stdout in UTF-8 before any print (D-PATTERNS,
    # mirrors ingest.py/embed.py/link.py).
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=(
            "One-command clone-and-go bootstrap for a fresh Research Paper "
            "Intelligence System clone: creates the vault folder structure, "
            "pulls the required Ollama models, initializes ChromaDB, and "
            "writes config.json plus an empty papers_registry.json (SETUP-01). "
            "Run with no flags for interactive prompts (Enter-through defaults), "
            "or pass flags directly; add --non-interactive to never prompt."
        )
    )
    parser.add_argument("--vault-path", default=None, help="Target Obsidian vault directory.")
    parser.add_argument(
        "--registry-path", default=None, help="Path to the shared papers_registry.json."
    )
    parser.add_argument("--project-name", default=None, help="Name for this project clone.")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt -- use flags (falling back to config.example.json defaults) for any value not given.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed past a path-safety refusal (sensitive location, or a non-empty directory that isn't already a vault).",
    )
    args = parser.parse_args()

    defaults = json.loads((_REPO_ROOT / "config.example.json").read_text(encoding="utf-8"))

    if args.non_interactive:
        vault_path = args.vault_path or defaults["vault_path"]
        registry_path = args.registry_path or defaults["registry_path"]
        project_name = args.project_name or defaults["project_name"]
    else:
        vault_path = args.vault_path or _prompt("Vault path", defaults["vault_path"])
        registry_path = args.registry_path or _prompt("Registry path", defaults["registry_path"])
        project_name = args.project_name or _prompt("Project name", defaults["project_name"])

    _validate_vault_target(vault_path, args.force)
    _refuse_sensitive("--registry-path", registry_path, args.force)

    values = dict(defaults)
    values["vault_path"] = vault_path
    values["registry_path"] = registry_path
    values["project_name"] = project_name

    result = run_setup(values, values, config_path=_REPO_ROOT / "config.json")

    _print_summary(result)

    # Non-step, one-time manual instruction (Open Question 2) -- never
    # automated; documents the one GUI action Obsidian itself requires.
    print(
        "\nOne-time manual step: in Obsidian, choose 'Open folder as vault' and select:\n"
        f"  {pathlib.Path(vault_path).expanduser()}"
    )
