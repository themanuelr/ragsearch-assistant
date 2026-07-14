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
"""

import pathlib
import shutil
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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
