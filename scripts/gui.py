"""Thin launcher for the Phase 8 local web GUI (D-03).

Same "core + thin CLI front-end" shape as every other script/*.py: all the
real logic lives in the ``gui/`` package (FastAPI app, routes, templates);
this file only parses ``--port``/``--no-browser``, resolves the listen port,
optionally opens the browser, and starts uvicorn.

Always binds host="127.0.0.1" -- never "0.0.0.0" (D-04, non-negotiable
privacy constraint; T-08-06).
"""

import argparse
import os
import sys
import webbrowser

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import uvicorn  # noqa: E402

from gui.app import app  # noqa: E402
from gui.config import load_gui_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the local web GUI.")
    parser.add_argument("--port", type=int, default=None, help="Override gui_port from config.json.")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open the browser tab.")
    args = parser.parse_args()

    port = args.port or load_gui_config()["gui_port"]

    if not args.no_browser:
        webbrowser.open(f"http://127.0.0.1:{port}")

    uvicorn.run(app, host="127.0.0.1", port=port)  # NEVER 0.0.0.0 (D-04)


if __name__ == "__main__":
    # Windows cp1252 stdout guard (project-wide convention: ingest.py,
    # biblio.py, link.py, note.py, setup.py all do this in their
    # if-__main__ block before any print()).
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    main()
