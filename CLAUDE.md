<!-- GSD:project-start source:PROJECT.md -->

## Project

**Research Paper Intelligence System**

A local-first, AI-powered research management system that ingests scientific papers (PDF or web URL), transforms them into structured Obsidian notes, connects them through shared topics and citations, and enables semantic retrieval — all running on a local LLM with zero per-query API cost. Designed as a GitHub template repository: clone once per research project, configure the vault path, and it's ready. Each project is fully isolated, sharing only a global papers registry so already-ingested papers are never re-processed.

**Core Value:** Drop a PDF or arXiv URL and get a fully linked Obsidian note — summary, findings, topic connections, and bibliography — without touching the cloud.

### Constraints

- **Hardware**: RTX 4060, 8GB VRAM — no models larger than gemma4:e4b
- **Cost**: Zero per-query API cost for document processing — all LLM calls go to local Ollama
- **Privacy**: All paper content stays local — nothing sent to external APIs
- **Vault I/O**: All vault writes go through `scripts/obsidian_cli.py` — the single sanctioned I/O chokepoint; it writes `.md` files atomically under `vault_path` (Obsidian's file watcher indexes them automatically)
- **Portability**: config.json is the only per-clone file that differs — all scripts read from it

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Python 3.x - MCP server implementation and core pipeline scripts (planned)
- Markdown - Obsidian vault note format and project documentation

## Runtime

- Python 3.6+ (via pip)
- Ollama - Local LLM server running at `localhost:11434`
- pip - Python package management
- Lockfile: Not detected (no `poetry.lock`, `Pipfile.lock`, or `requirements.lock`)

## Frameworks

- FastMCP (via `mcp>=1.0.0`) - Model Context Protocol server framework for exposing LLM capabilities as tools
- Ollama SDK - HTTP client for local Gemma 4 e4b model
- ddgs (via `ddgs>=0.1.0`) - DuckDuckGo Search client for web research without API keys
- obsidian-cli - CLI tool for vault I/O (installed as skill, invoked via subprocess)
- defuddle - Web paper extraction tool; the npm package `defuddle` (install globally: `npm install -g defuddle`), invoked via subprocess as `defuddle parse <url> --md`
- ChromaDB - Local embedded vector database for semantic search
- pdfplumber - PDF text extraction and metadata parsing
- pypdf - PDF manipulation and analysis

## Key Dependencies

- `mcp>=1.0.0` - MCP server framework that exposes tools to Claude Code
- `ddgs>=0.1.0` - DuckDuckGo web search integration
- Ollama (external service) - Hosts local LLM model at `localhost:11434`

## Configuration

- Ollama runs at `http://localhost:11434` (hardcoded in `mcp-ollama/server.py` line 11)
- MCP server runs via stdio transport (FastMCP default)
- No `.env` file detected (configuration expected via config.json per STARTING_POINT.md)
- No build tool detected (Python scripts run directly)
- No Docker configuration (pure Python + local Ollama service)

## Platform Requirements

- Python 3.6 or higher
- Ollama service running on localhost:11434
- RTX 4060 or similar GPU with 8GB VRAM (as per STARTING_POINT.md constraints)
- Local disk space for vector DB and vault files
- Same as development — designed for single-machine local operation
- No cloud hosting required (offline-capable after initial model pulls)
- Obsidian desktop application for vault editing

## Notable Constraints

- **Hardware:** RTX 4060 with 8GB VRAM limits to `gemma4:e4b` or smaller models
- **Cost:** Zero per-query API cost for document processing — all LLM calls use local Ollama
- **Privacy:** All data stays local — no paper content sent to external APIs except defuddle for web extraction
- **Template design:** Single `config.json` file differs between project clones; all scripts read from it

## Planned Dependencies (From Phase Roadmap)

- Phase 1: `pdfplumber`, `pypdf` for PDF extraction
- Phase 5: ChromaDB for vector embeddings
- Phase 3: defuddle (npm package, installed globally via `npm install -g defuddle`; invoked via subprocess)

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming Patterns

- Lowercase with underscores: `server.py`, `requirements.txt`
- Purpose-driven names: `server.py` (MCP server implementation), `requirements.txt` (Python dependencies)
- Private/internal modules prefix underscore: Functions like `_ollama_chat()`, `_web_search()` signal internal-only use
- Lowercase with underscores (snake_case): `_ollama_chat()`, `_ollama_chat_with_tools()`, `_web_search()`
- Public tool functions (MCP exports) use lowercase: `ask_local_model()`, `research()`, `check_ollama_status()`
- Private helper functions prefix with underscore: `_ollama_chat()`, `_web_search()`
- Descriptive action verbs: `check_ollama_status()` (not `get_status()`), `_web_search()` (not `search()`)
- Lowercase with underscores: `max_results`, `search_log`, `assistant_msg`, `tool_calls`
- Constants in UPPERCASE: `OLLAMA_BASE`, `DEFAULT_MODEL`
- Loop counters: `iteration`, `i`, `r` (iterator-style short names acceptable in comprehensions)
- Dictionary keys: Lowercase with underscores matching API fields: `"role"`, `"content"`, `"model"`, `"messages"`
- Union syntax using pipe operator: `str | None` (Python 3.10+, see line 21)
- Type hints on function signatures: `def _ollama_chat(prompt: str, model: str, system: str | None = None) -> str:`
- Complex types documented in docstrings when needed

## Code Style

- **Indentation:** 4 spaces (standard Python)
- **Line length:** Lines appear to stay under 100 characters (no formatter config enforced)
- **String literals:** Double quotes preferred (see `"Content-Type": "application/json"`)
- **Dictionary style:** No trailing commas in multiline dictionaries; compact style for short dicts
- No linter configuration detected (`.eslintrc*`, `ruff.toml`, `pyproject.toml` absent)
- Type hints used consistently across function signatures
- No type checking tool configured (mypy, pyright not detected)

## Import Organization

- No path aliases detected (no `@` paths or `sys.path` manipulation)
- Imports are absolute and straightforward: `from mcp.server.fastmcp import FastMCP`

## Error Handling

- **Try-except blocks** catch specific exceptions: `urllib.error.URLError`, `ImportError`, `json.JSONDecodeError`
- **Broad exception catch** only in appropriate contexts: `except Exception as e:` in `_web_search()` (line 81) as a safety net after specific catches
- **Return error strings** instead of raising exceptions: `return f"[Ollama error: {e}]"` (lines 44, 65, 82)
- **Error messages prefixed with brackets** for clarity in logs: `"[Ollama error: ...]"`, `"[ddgs not installed — ...]"`, `"[Search error: ...]"`
- **Timeouts set on network calls:** `timeout=120` for chat requests (line 40), `timeout=180` for tool-calling (line 62), `timeout=5` for status checks (line 228)

## Logging

- **Return-based error reporting:** Functions return human-readable error strings instead of logging to a file
- **Search log tracking:** `search_log` list (line 174) accumulates queries for audit trail: `search_log.append(q)`
- **Search summary prepended:** Final answer includes count and queries: `f"[Searched {len(search_log)} time(s):\n{log_str}]"` (lines 195–196)
- **Tool call logging:** No explicit logging; flow control happens via message history in the research loop

## Comments

- **Section headers** using dashes: `# ---------------------------------------------------------------------------` and descriptive text (lines 17–19, 104–106)
- **Inline explanations** for non-obvious logic: `# Append assistant turn to history` (line 185), `# Final answer — prepend search log summary if searches were made` (line 192)
- **Version notes** for compatibility: `# arguments may arrive as a JSON string in some Ollama versions` (line 205)
- **Action items** not typical; STARTING_POINT.md documents future work, not inline TODOs
- **Module docstring** at top: `"""..."""` (lines 1–3) describes purpose and scope
- **Function docstrings:** All public MCP tools (`ask_local_model`, `research`, `check_ollama_status`) have detailed docstrings
- **Docstring format:** Google-style with Sections like "Args:" and description of use cases
- **Internal helper docstrings:** Not documented (private functions marked with `_` prefix)

## Function Design

- **Small, focused functions:** Helpers like `_ollama_chat()` (9 lines), `_web_search()` (12 lines) do one thing
- **Moderate tool functions:** `ask_local_model()` (1 line), `research()` (70 lines) with clear responsibility
- **Loop structure in `research()`:** Max iterations limit (line 173) prevents runaway loops
- **Minimal parameters:** Most functions take 2–3 params (prompt, model, system)
- **Optional parameters with defaults:** `system: str | None = None`, `max_results: int = 6`
- **Timeout parameters:** Hardcoded for now, could be parameterized in future
- **Consistent return types:** All network/chat functions return strings (error messages or results)
- **Dict returns for complex responses:** `_ollama_chat_with_tools()` returns the full response dict (line 63)
- **Error strings as fallback:** Instead of raising, functions return error messages: `"[Ollama error: {e}]"`

## Module Design

- **Decorator-based exposure:** `@mcp.tool()` decorator marks functions as MCP tools (lines 108, 135, 224)
- **Entry point:** `if __name__ == "__main__": mcp.run(transport="stdio")` (lines 236–237)
- **Private by default:** Functions prefixed with `_` are internal; public functions are those decorated with `@mcp.tool()`
- Constants at top: `OLLAMA_BASE`, `DEFAULT_MODEL` (lines 11–12)
- Helper functions grouped: Internal helpers section (lines 17–102)
- Public tools section: MCP tools with `@mcp.tool()` (lines 104+)
- Entry point at bottom

## Project-Wide Conventions (From STARTING_POINT.md)

- **One paper, one task** — future `ingest.py` scripts should follow this
- **One section, one prompt** — when summarizing, send sections separately to local LLM
- **JSON as handoff format** — future scripts pass structured JSON between stages
- **Stateless tools** — each MCP tool reads what it needs, does one thing, writes output, exits
- **All vault I/O through `scripts/obsidian_cli.py`** — the single sanctioned chokepoint; it writes `.md` files directly and atomically under `vault_path` (Obsidian's file watcher indexes them automatically; no running-instance dependency for the write itself)
- Scripts in `scripts/` directory: `setup.py`, `ingest.py`, `embed.py`, `link.py`, `biblio.py` (from STARTING_POINT.md)
- All use lowercase snake_case filenames
- All accept command-line arguments for configuration

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```text

```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| MCP Server | Expose local Ollama models as Claude tools; handle web search autonomously; provide lightweight LLM inference | `mcp-ollama/server.py` |
| ask_local_model | Send single prompts to gemma4:e4b for text summarization, extraction, reformatting | `mcp-ollama/server.py` (lines 108-132) |
| research | Autonomous web research via DuckDuckGo; Gemma decides searches, synthesizes results | `mcp-ollama/server.py` (lines 135-221) |
| check_ollama_status | Health check and model discovery for Ollama server | `mcp-ollama/server.py` (lines 224-233) |
| Obsidian Skills | Vault I/O, markdown formatting, database views, web paper extraction | Installed in `~/.claude/skills/` |
| PDF Extraction Pipeline | Extract text, metadata, sections from PDFs → structured JSON | [TO BUILD: `scripts/ingest.py`] |
| Embeddings Pipeline | Text → vector embeddings via Ollama → ChromaDB storage | [TO BUILD: `scripts/embed.py`] |
| Bibliography Linker | Parse references → resolve against vault/registry → create stubs | [TO BUILD: `scripts/biblio.py`] |
| Topic Graph Builder | Extract topics from papers → update topic index notes | [TO BUILD: `scripts/link.py`] |
| Setup Automation | Create vault structure, pull models, initialize ChromaDB | [TO BUILD: `scripts/setup.py`] |

## Pattern Overview

- **Stateless tools:** Each component reads input, does one thing, writes output, exits
- **Claude orchestrates, local LLM executes:** Claude decides next steps; Gemma handles document work
- **JSON handoff format:** Every tool passes structured JSON between stages
- **Nuclear tasks:** One paper per invocation; one section per LLM call
- **Local-first:** All processing on local hardware; zero per-query API cost after setup

## Layers

- Purpose: Expose Ollama models and web search as Claude tools
- Location: `mcp-ollama/server.py`
- Contains: Tool definitions, Ollama HTTP clients, DuckDuckGo search integration
- Depends on: Local Ollama server on port 11434
- Used by: Claude Code
- Purpose: Run gemma4:e4b (8B, vision + tools) and nomic-embed-text locally
- Location: localhost:11434
- Contains: Model weights, VRAM management, chat/embedding endpoints
- Depends on: Hardware (RTX 4060, 8GB VRAM minimum)
- Used by: MCP server, embedding pipeline
- Purpose: Store structured paper notes, topics, reviews, stubs
- Location: Per-project vault (configured in `config.json`)
- Contains: `Papers/` (full ingested papers), `Stubs/` (auto-created citation stubs), `Topics/` (topic index), `Reviews/` (literature review drafts), `_Index.md`, `_Papers.base`
- Depends on: Plain markdown format
- Used by: User, topic linker, bibliography linker
- Purpose: Store embeddings for semantic search
- Location: Per-project `chroma_db/` directory
- Contains: Embedded paper sections and abstracts
- Depends on: nomic-embed-text model in Ollama
- Used by: Semantic search queries
- Purpose: Share ingested paper metadata across project clones
- Location: User-configured (e.g., `~/research/papers_registry.json`)
- Contains: Paper metadata (title, authors, year, summary, key findings), DOI, arXiv ID, projects that have ingested it
- Depends on: Single JSON file
- Used by: All project clones for cross-project imports

## Data Flow

### Primary Request Path: Ingest a Paper

### Web Paper Ingestion

### Semantic Search

### Stub Upgrade

- **Vault state:** Plain markdown in Obsidian — changes tracked by Obsidian itself
- **Vector DB state:** ChromaDB persists embeddings per project
- **Registry state:** Single JSON file appended to on each successful ingest
- **No in-process state:** Each tool call is independent

## Key Abstractions

- Purpose: Canonical representation of a fully ingested paper
- Examples: `Papers/Attention Is All You Need.md`, `Papers/BERT Pretraining.md`
- Pattern: Standard YAML frontmatter (title, authors, year, journal, DOI, tags, date_ingested, source_projects) + structured sections (Summary, Key Findings, Methodology, Results, Limitations, Related Topics, Related Papers, References, My Notes, Citation)
- Purpose: Placeholder for papers referenced but not yet ingested
- Examples: `Stubs/Unknown Author - Unknown Title.md`
- Pattern: Minimal frontmatter (title, authors if parseable, year, DOI, tags, status: stub, date_created, cited_by list) + raw citation text + upgrade instruction
- Purpose: Index of all papers under one topic
- Examples: `Topics/Transformers.md`, `Topics/Attention Mechanisms.md`
- Pattern: Wikilinks to all papers tagged with topic + backlinks from papers
- Purpose: Cross-project metadata without re-processing
- Examples: Key=DOI ("10.1145/3290605.3300750"), Value={title, authors, year, journal, summary, key_findings, projects array, vault_note}
- Pattern: Normalized keys (DOI first choice, arXiv second), projects array tracks which clones have ingested it

## Entry Points

- Location: User's Claude Code session
- Triggers: User types a research command (e.g., "ingest this paper", "search for X", "import from registry")
- Responsibilities: Route to correct script/tool, manage multi-step workflows, pass data between pipeline stages
- Location: `mcp-ollama/server.py` (launched via MCP configuration)
- Triggers: Claude Code initialization
- Responsibilities: Expose Ollama tools and research capability to Claude
- `setup.py` [FUTURE]: One-time per project — creates vault folders, pulls models, initializes ChromaDB
- `ingest.py` [FUTURE]: Per-paper — PDF/URL → JSON → registry write
- `embed.py` [FUTURE]: Per-paper — sections → embeddings → ChromaDB
- `biblio.py` [FUTURE]: Per-paper — extract references → link/import/stub
- `link.py` [FUTURE]: Per-paper → update topics

## Architectural Constraints

- **Threading:** Single-threaded event loop in MCP server; Python scripts run sequentially
- **Global state:** Global papers registry (`~/research/papers_registry.json`) is the only shared mutable state across projects; appended to atomically on each ingest
- **Circular imports:** Not applicable — pipeline is DAG (directed acyclic)
- **Hardware:** RTX 4060 (8GB VRAM) dictates model size; gemma4:e4b is at the edge
- **Context windows:** Ollama gemma4:e4b has 131K context; templates designed to send one section at a time to avoid overflow
- **No external APIs:** Zero per-query cost — all LLM inference local; defuddle used only for web paper extraction
- **Vault format:** Plain markdown — supports switching tools later without data loss

## Anti-Patterns

### Monolithic Ingestion

### Tight Coupling to Obsidian API

### Duplicating Registry Entries

### Monolithic Bibliography Parsing

### Storing Secrets in Code

## Error Handling

- **MCP tool failures:** Return `[Tool name error: message]` prefix for visibility in chat (see `server.py` lines 43-44, 64-65, 81-82)
- **Missing dependencies:** Print installation command (see `server.py` line 80: `pip install ddgs`)
- **Ollama unreachable:** Timeout at 120-180s, return error message; Claude can retry or suggest checking server (see `server.py` lines 40-44)
- **Registry lookups:** Return null if not found; trigger full ingest pipeline
- **Vault write failures:** `scripts/obsidian_cli.py` raises on filesystem errors; treat as fatal and verify `vault_path` in `config.json` points to a real Obsidian vault directory

## Cross-Cutting Concerns

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
