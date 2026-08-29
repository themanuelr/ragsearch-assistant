# Research Paper Intelligence System

A local-first, AI-powered research management system that ingests scientific papers (PDF or web
URL), transforms them into structured Obsidian notes, connects them through shared topics and
citations, and enables semantic retrieval — all running on a local LLM with **zero per-query API
cost**. Drop a PDF or arXiv URL and get a fully linked Obsidian note — summary, findings, topic
connections, and bibliography — without touching the cloud.

This repo is a **GitHub template**: clone it once per research project, configure the vault path,
and it's ready. Each project is fully isolated (its own vault, its own ChromaDB), but every clone
shares one global papers registry, so a paper already ingested in another project is never
re-processed from scratch.

## Prerequisites

| Requirement | Version tested | Notes |
|---|---|---|
| Python | 3.10+ (3.14.4 tested) | Runs the main pipeline (`scripts/`) |
| Ollama | running at `http://localhost:11434` | Local LLM server — analysis and embeddings |
| MinerU | 3.2.3 | PDF structure extraction — needs its **own** Python env, see below |
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

   This installs the runtime dependencies (`mcp`, `ddgs`, `filelock`, `chromadb`, `pydantic`) and
   `pytest`, which the verification step below needs.

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

6. **Configure the project:**

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

**PDF extraction**
- `mineru_path` — path to the MinerU executable (in its own Python 3.10–3.13 env — see Setup step 4)
- `mineru_timeout` — MinerU subprocess timeout in seconds (default: `1800`)

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

## How it works

- **PDF extraction:** MinerU parses the PDF's structure (title, sections, references, layout) via a subprocess call.
- **LLM fill:** a local LLM (`gemma4:e4b` via Ollama) fills the structured PaperJSON — extraction fields plus generated analysis (summary, key findings, limitations).
- **Note write:** the finished note is written atomically as a `.md` file directly into the Obsidian vault (Obsidian's file watcher indexes it automatically — no CLI subprocess involved).
- **Bibliography:** cited references are linked to existing vault notes, imported from the global registry, or created as stub notes in `Stubs/`, then upgraded automatically when eventually ingested.
- **Semantic search:** each note's sections are embedded (`nomic-embed-text` via Ollama) and stored in a local ChromaDB collection for retrieval by meaning, not just keyword.
- **Topic graph:** shared topics across papers are linked via Obsidian wikilinks, with per-topic index notes kept in sync.

`mcp-ollama/` is the MCP server component exposing these capabilities to Claude Code; its own
dependencies (`mcp`, `ddgs`) are already covered by the root `requirements.txt`.

## Privacy

All paper content stays local. Nothing is sent to external APIs. The only optional network call
beyond Ollama (localhost) is Crossref metadata enrichment, which is off by default
(`crossref_validate: false`) and, when enabled, sends only bibliographic metadata — never paper
content.

## License

This project is released under the MIT License, copyright (c) 2026 Manuel Munevar — see the
`LICENSE` file at the repository root for the full text. As a clone-per-project template, you
are free to use, modify, and redistribute it under those terms.
