# Research Paper Intelligence System

A local-first, AI-powered research management system that ingests scientific papers (PDF or web
URL), transforms them into structured Obsidian notes, connects them through shared topics and
citations, and enables semantic retrieval — all running on a local LLM with **zero per-query API
cost**, driven from the command line or a local web GUI. Drop a PDF or arXiv URL and get a fully
linked Obsidian note — summary, findings, topic connections, and bibliography — without touching
the cloud.

This repo is a **GitHub template**: clone it once per research project, configure the vault path,
and it's ready. Each project is fully isolated (its own vault, its own ChromaDB), but every clone
shares one global papers registry, so a paper already ingested in another project is never
re-processed from scratch.

## Prerequisites

| Requirement | Version tested | Notes |
|---|---|---|
| Python | 3.10+ (3.14.4 tested) | Runs the main pipeline (`scripts/`) |
| Ollama | running at `http://localhost:11434` | Local LLM server — analysis and embeddings |
| [MinerU](https://github.com/opendatalab/MinerU) | 3.2.3 | PDF structure extraction — needs its **own** Python env, see below |
| Node.js + defuddle | Node v24 tested, defuddle 0.19.1 | Web-page → markdown extraction for URL ingestion |
| Obsidian desktop | any recent version | Optional — for viewing the vault only; notes are written directly as `.md` files |

**Hardware baseline:** ~8GB VRAM (RTX 4060 class). The Ollama context window is capped at 65536
tokens by default (`ollama_num_ctx_cap`) to stay within that budget.

## Setup

1. **Clone the template.** Use "Use this template" on GitHub, or `git clone` this repo.

2. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   This installs the runtime dependencies (`mcp`, `ddgs`, `filelock`, `chromadb`, `pydantic`), the
   local web GUI's stack (`fastapi`, `uvicorn`, `jinja2`, `markdown`, `httpx`), and `pytest`, which
   the verification step below needs.

3. **Install Ollama and pull the two required models:**

   ```bash
   ollama pull gemma4:e4b        # analysis LLM — fills PaperJSON + generates note content
   ollama pull nomic-embed-text  # embedding model — overridable via config embed_model
   ```

4. **Install MinerU — in its own Python environment.**

   > **Warning:** MinerU does **not** support Python 3.14. If your main environment is on 3.14
   > (as tested here), you must install MinerU in a **separate** Python 3.10–3.13 environment.
   > Create that environment, `pip install mineru` inside it, then point `config.json`'s
   > `mineru_path` at the resulting executable.
   >
   > Windows example: `C:\Users\you\envs\mineru312\Scripts\mineru.exe`
   >
   > The first MinerU run downloads its models — expect the first ingest to be slow.

5. **Install defuddle** (used for web URL ingestion):

   ```bash
   npm install -g defuddle
   ```

   `defuddle` is located automatically via `PATH`, or you can set `defuddle_path` explicitly in
   `config.json`.

6. **Configure the project.** The repo ships an interactive bootstrap that does this step for you:

   ```bash
   python scripts/setup.py
   ```

   Run with no flags for interactive prompts (Enter-through defaults), or pass `--vault-path`,
   `--registry-path`, and `--project-name` directly (`--non-interactive` to never prompt,
   `--force` to proceed past a path-safety refusal). It writes `config.json`, scaffolds a working
   Obsidian vault at `vault_path` from `obsidian-vault-template/` — the `Papers/`, `Stubs/`, and
   `Topics/` folders, the `_Papers.base` table view, and a preconfigured Obsidian theme and plugin
   setup, never overwriting a file that already exists there — pulls the two required Ollama
   models, initializes ChromaDB, and creates an empty `papers_registry.json` if one isn't already
   present. Finish by opening the vault folder in Obsidian ("Open folder as vault").

   Alternatively, configure by hand:

   ```bash
   cp config.example.json config.json
   ```

   Then fill in `registry_path`, `vault_path`, `mineru_path`, and any other paths for your
   machine.

## Configuration

`config.json` is the **only** file that differs between clones of this template — every script
reads its settings from it. Copy `config.example.json` to `config.json` and adjust as needed. All
keys, grouped by area:

**Core paths**
- `registry_path` — path to the shared global papers registry JSON (shared across all your project clones)
- `vault_path` — path to this project's Obsidian vault
- `project_name` — slug identifying this project in the global registry's `projects` list
- `paperjson_cache_dir` — shared PaperJSON cache directory, a sibling of the registry reused across all your project clones (default: `~/research/paperjson_cache`)

**PDF extraction**
- `mineru_path` — path to the MinerU executable (in its own Python 3.10–3.13 env — see Setup step 4)
- `mineru_timeout` — MinerU subprocess timeout in seconds (default: `1800`)
- `mineru_output_dir` — shared MinerU extraction output directory, a sibling of the registry reused across all your project clones (default: `~/research/mineru_output`)

**Web ingestion**
- `defuddle_path` — path to the defuddle executable (empty = resolved via `PATH`)
- `defuddle_timeout` — defuddle subprocess timeout in seconds (default: `60`)
- `web_min_body_chars` — minimum extracted body length to accept a web page as valid (default: `2000`)

**Embeddings / search**
- `chroma_db_path` — path to this project's local ChromaDB store (default: `./chroma_db`)
- `embed_model` — Ollama embedding model (default: `nomic-embed-text`)
- `embed_timeout` — embedding API call timeout in seconds (default: `60`)
- `embed_batch_size` — number of sections embedded per Ollama call (default: `16`)
- `embed_section_max_tokens` — max tokens per section chunk before splitting (default: `2000`)

**Ollama tuning**
- `ollama_num_ctx_cap` — max context window passed to Ollama, capped for 8GB VRAM (default: `65536`)
- `ollama_section_timeout` — per-section LLM call timeout in seconds (default: `300`)

**Crossref enrichment** (optional, off by default — keeps the pipeline fully offline)
- `crossref_validate` — enable Crossref DOI/metadata lookups (default: `false`)
- `crossref_contact_email` — a real contact email, required by Crossref's API etiquette if you enable `crossref_validate`; a placeholder is treated as "skip Crossref" (default: `""`)
- `crossref_timeout` — Crossref request timeout in seconds (default: `30`)
- `crossref_retries` — number of retry attempts on transient Crossref errors (default: `1`)

**Web GUI**
- `gui_port` — port the local web GUI binds to at `127.0.0.1` (default: `8765`)
- `uningested_dir` — drop-folder directory the GUI's Processing page scans for PDFs waiting to be ingested (default: `./uningestedPDFs`)

## Verify your setup

Run the test suite — it needs no external programs (Ollama, MinerU, defuddle) to pass:

```bash
python -m pytest tests/ -q
```

Confirm the ingest CLI is wired up correctly:

```bash
python scripts/ingest.py --help
```

```
usage: ingest.py [-h] (--pdf PDF | --url URL | --doi DOI) [--force-extract]
                 [--timeout TIMEOUT] [--output OUTPUT] [--refill] [--print]

Ingest a research paper PDF or URL via MinerU/defuddle and emit a short
confirmation (cache path + note path) to stdout. Use --print/--stdout to emit
the full PaperJSON v2 to stdout instead, or use --output/-o to write it to a
file.

options:
  -h, --help           show this help message and exit
  --pdf PDF            Path to the input PDF file.
  --url URL            URL of the paper to ingest (arXiv, PubMed, or journal).
  --doi DOI            Materialize a note for a paper already in
                       papers_registry.json (by DOI, arXiv id, or title)
                       without re-processing the PDF.
  --force-extract      Re-run MinerU even if output already exists (D-15).
  --timeout TIMEOUT    MinerU subprocess timeout in seconds (default: 1800).
  --output, -o OUTPUT  Write the final PaperJSON to this path as UTF-8.
                       Preferred over '>' on Windows — PowerShell '>' re-
                       decodes correct UTF-8 through the OEM console code page
                       and rewrites as UTF-16LE with mojibake (José → 'Jos├⌐',
                       U+2019 → 'ΓÇÖ'). Use -o to bypass redirection entirely.
  --refill             Re-run the LLM fill reusing existing MinerU output (no
                       GPU extraction step). Errors if no prior
                       content_list.json exists for this PDF — run without
                       --refill first. Bypasses the registry cache-hit gate so
                       the fill cascade re-runs. Registry write uses merge
                       (Plan 10 union-merge) — does not clobber other
                       projects.
  --print, --stdout    Emit the full PaperJSON v2 to stdout as UTF-8 JSON. By
                       default only a short confirmation (cache path + note
                       path) is printed; use this flag to restore the full
                       JSON dump.
```

**First ingest:** once `config.json` is filled in and MinerU/Ollama/defuddle are reachable, try:

```bash
python scripts/ingest.py --pdf path/to/paper.pdf
# or
python scripts/ingest.py --url https://arxiv.org/abs/xxxx.xxxxx
```

**Other pipeline commands.** `ingest.py` drives the full pipeline end to end, but each stage is
also its own re-runnable script — useful for regenerating one stage without repeating the others.
Pass `--help` to any of them for the full flag list.

| Command | Purpose |
|---|---|
| `python scripts/note.py --stem <stem> [--force] [--force-analysis]` | Regenerate a note from its cached PaperJSON, optionally overwriting an existing note or re-running the analysis calls. |
| `python scripts/biblio.py --stem <stem>` | Re-run bibliography linking for a cached PaperJSON. |
| `python scripts/embed.py <paperjson-path>` | Embed one PaperJSON's sections into the ChromaDB index. |
| `python scripts/embed.py --all` | Backfill: idempotently embed every not-yet-embedded registry paper, cache file, and stub. |
| `python scripts/embed.py --query "..." [--n-results N]` | Run a semantic search query against the local ChromaDB index and print ranked results (default 5). |
| `python scripts/link.py --stem <stem>` | Rebuild the topic graph for one paper; `--all`/`--refresh` rebuilds it for the whole vault. |
| `python scripts/retag.py --stem <stem> [--context "..."]` | Additively re-tag an already-ingested paper along a new axis, then re-link. |

`scripts/obsidian_cli.py` is the internal vault-write chokepoint every script above writes
through — it has no CLI of its own.

## How it works

- **PDF extraction:** MinerU parses the PDF's structure (title, sections, references, layout) via a subprocess call.
- **LLM fill:** a local LLM (`gemma4:e4b` via Ollama) fills the structured PaperJSON — extraction fields plus generated analysis (summary, key findings, limitations).
- **Note write:** the finished note is written atomically as a `.md` file directly into the Obsidian vault (Obsidian's file watcher indexes it automatically — no CLI subprocess involved).
- **Bibliography:** cited references are linked to existing vault notes, imported from the global registry, or created as stub notes in `Stubs/`, then upgraded automatically when eventually ingested.
- **Semantic search:** each note's sections are embedded (`nomic-embed-text` via Ollama) and stored in a local ChromaDB collection for retrieval by meaning, not just keyword.
- **Topic graph:** shared topics across papers are linked via Obsidian wikilinks, with per-topic index notes kept in sync.

`mcp-ollama/` is the MCP server component exposing these capabilities to Claude Code, as five
tools: `ask_local_model` (send a one-off prompt to the local LLM), `research` (autonomous
multi-search web research synthesized by the local LLM), `check_ollama_status` (verify Ollama is
reachable and list available models), `process_pdf` (run the ingest pipeline on a PDF and return
its PaperJSON), and `search_similar` (run a semantic search query against the vault's embedded
papers). Its own dependencies (`mcp`, `ddgs`) are already covered by the root `requirements.txt`.

## Web GUI

Everything above is also reachable through a local web interface — a front-end over the same
scripts documented above, not a parallel implementation.

```bash
python scripts/gui.py
# or: python scripts/gui.py --port 9000 --no-browser
```

It binds to `127.0.0.1` only, never `0.0.0.0` — reachable from this machine alone, never from the
network. The listen port defaults to `gui_port` in `config.json` (`8765`) and can be overridden
with `--port`; `--no-browser` skips auto-opening a browser tab.

| Page | What it's for |
|---|---|
| Overview (Project) | Live per-paper pipeline-stage visibility for this project's vault, plus the drop-folder's pending PDFs. |
| Overview (Universal) | Read-only view of the shared global papers registry across every clone, with an "Import into This Project" action. |
| Processing | Ingest a PDF or URL, bulk-ingest a drop folder, re-run any subset of note/biblio/embed/link for a single paper in a chosen order, and retag papers along a new axis. |
| Chat | Streaming chat against a local Ollama model, with retrieval scoped to the ChromaDB semantic index, a set of vault notes, or no retrieval at all. |
| Logs & Activity | The live job queue and per-job log streaming for every pipeline action run from the GUI. |
| Docs & Help | This README and the other project docs, rendered in-app. |
| Settings | An in-browser editor for `config.json`, with confirmation before repointing any data path that already holds data. |

## Privacy

All paper content stays local. Nothing is sent to external APIs. The only optional network call
beyond Ollama (localhost) is Crossref metadata enrichment, which is off by default
(`crossref_validate: false`) and, when enabled, sends only bibliographic metadata — never paper
content.

## License

This project is released under the MIT License, copyright (c) 2026 Manuel Munevar — see the
`LICENSE` file at the repository root for the full text. As a clone-per-project template, you
are free to use, modify, and redistribute it under those terms.
