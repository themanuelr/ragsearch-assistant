"""GUI-specific config wrapper (Phase 8, D-04/D-10).

Never re-implements config loading. Wraps ``scripts.ingest._load_config`` (the
canonical loader — Don't Hand-Roll per 08-RESEARCH.md) and only adds the two
GUI-only defaults: ``gui_port`` and ``uningested_dir``.
"""

import pathlib

from scripts.ingest import _load_config as _core_load_config


def load_gui_config() -> dict:
    """Load config.json via the canonical loader, applying GUI-only defaults.

    ``gui_port`` defaults to 8765 when absent (D-04). ``uningested_dir``
    defaults to ``./uningestedPDFs`` when absent (D-10); expanded here only
    (the shared ``_load_config`` path-expansion tuple in scripts/ingest.py is
    left untouched since this key is GUI-only).
    """
    cfg = _core_load_config()
    cfg.setdefault("gui_port", 8765)
    cfg.setdefault("uningested_dir", "./uningestedPDFs")
    if cfg.get("uningested_dir"):
        cfg["uningested_dir"] = str(pathlib.Path(cfg["uningested_dir"]).expanduser())
    return cfg
