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

Full stack reference: .planning/codebase/STACK.md

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

Full conventions reference: .planning/codebase/CONVENTIONS.md
Before writing code, read .planning/codebase/CONVENTIONS.md and match its patterns.

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

Full architecture reference: .planning/codebase/ARCHITECTURE.md

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

| Skill | Description | Path |
| ----- | ----------- | ---- |
| obsidian-note-format | Authoritative reference for how this project writes Obsidian vault notes — frontmatter, callouts, tags, filenames, encoding, and the direct atomic write mechanism. | `.claude/skills/obsidian-note-format/SKILL.md` |
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
