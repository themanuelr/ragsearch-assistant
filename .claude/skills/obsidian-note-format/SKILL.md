---
name: obsidian-note-format
description: Authoritative reference for how this project writes Obsidian vault notes — frontmatter, callouts, tags, filenames, encoding, and the direct atomic write mechanism.
---

# Obsidian Note-Format Conventions

This skill documents the exact formatting rules that `scripts/note.py` (renderer) and
`scripts/obsidian_cli.py` (writer) implement. All new notes written to the vault must
conform to these rules so Obsidian parses and renders them correctly.

---

## Vault Model: Why Direct File Write Works

A vault is a **plain folder of UTF-8 `.md` files**. Obsidian's file watcher
(`chokidar` under the hood) monitors the folder and auto-indexes any new or
changed file within milliseconds. This means:

- A direct atomic write of a `.md` file under `vault_path/` is **functionally
  equivalent** to using `obsidian-cli create` — no running Obsidian instance is
  required for the write itself.
- The write is done through `scripts/obsidian_cli.py`, which remains the **single
  sanctioned vault-I/O chokepoint**. Only the mechanism changes (direct file I/O
  instead of obsidian.COM argv subprocess).

### Why NOT argv/subprocess for the write

Obsidian's built-in CLI (`obsidian.COM --`) serializes all argv tokens into JSON to
forward to the running main process. A ~6.7 KB note body containing ~80 raw newlines
plus LaTeX backslash sequences (`\text{...}`, `\Delta`) corrupts that JSON
(trailing comma, invalid escape → main-process crash → 30 s timeout, note never
written). All tests that mock the subprocess miss this entirely.

**Rule:** Never pass note content through a shell, argv, or any escaping layer.
Write the bytes directly.

---

## YAML Frontmatter

Frontmatter must appear **at the very top** of the file, fenced by `---` on its own
line, with no blank lines before the opening `---`.

```markdown
---
title: "Attention Is All You Need"
authors:
  - "Vaswani, Ashish"
  - "Shazeer, Noam"
year: 2017
journal: "Advances in Neural Information Processing Systems"
doi: "10.48550/arXiv.1706.03762"
tags:
  - transformers
  - attention-mechanisms
  - neural-networks
status: ingested
date_ingested: 2026-06-26
---
```

### Rules

| Rule | Detail |
|------|--------|
| **Double-quote all string values** | Required for colon safety (SC2). `title: "BERT: Pre-training..."` parses; `title: BERT: Pre-training...` does not. |
| **Escape internal `"` as `\"`** | `title: "Foo \"Bar\" Baz"` |
| **Authors as YAML list** | Each author on its own `  - "..."` line, double-quoted. |
| **Tags as lower-kebab slugs** | No spaces, no uppercase. `"Attention Mechanisms"` → `attention-mechanisms`. Rendered as `tags:` block list (not inline). |
| **Nullable fields omitted** | `doi` and `arxiv_id` are omitted entirely when `None`; do not emit `doi: null`. |
| **`status: ingested`** | Fixed literal for fully ingested papers. |
| **`date_ingested`** | ISO 8601 date string (`YYYY-MM-DD`), unquoted — YAML native date. |

**Validation:** `yaml.safe_load()` of the fenced block must succeed without error
(SC2). The renderer uses `_render_frontmatter()` in `scripts/note.py` to enforce this.

---

## Callouts

Obsidian supports GitHub-Flavored-Markdown callout blocks using the
`> [!type] Optional Title` syntax. Reference:
<https://help.obsidian.md/Editing+and+formatting/Callouts>

```markdown
> [!important] Key Contribution
> The transformer architecture eliminates recurrence entirely, enabling full
> parallelisation during training.

> [!warning] Limitation
> Quadratic self-attention complexity makes the model expensive for very long
> sequences (> 1 000 tokens).
```

### Rules

| Rule | Detail |
|------|--------|
| **One blank line before the callout block** | Required for Obsidian to render it as a callout (not a plain blockquote). |
| **One blank line after the callout block** | Required to end the callout. |
| **Every content line starts with `> `** | Including continuation lines. A line without the prefix ends the block. |
| **Title is optional** | `> [!warning]` is valid; the title defaults to the type name. |
| **Types used in this project** | `important` (alias for `tip` — used for key contributions, D-15) and `warning` (used for limitations, D-15). Both are built-in Obsidian types. |

### Required callouts per note (SC3)

Every ingested paper note must contain **at least one `[!important]`** and **at
least one `[!warning]`** callout:

- `[!important] Key Contribution` after the Summary section — first item from
  `analysis.claims`.
- `[!warning] Limitation` at the top of the Limitations section — first item from
  `analysis.limitations`.

The fallback entries `["(generation failed -- see raw extract)"]` ensure these
callouts always render even when LLM generation fails.

---

## Wikilinks and Tags

### Wikilinks

`[[Note Name]]` links to another note in the same vault (Obsidian resolves by
filename, vault-relative). Used in later phases (Phase 3 Topics/Stubs, Phase 4
Bibliography):

```markdown
Related topics: [[Transformers]], [[Attention Mechanisms]]
```

### Tags

Tags appear in **two places**:
1. `tags:` list in YAML frontmatter — the canonical location, lower-kebab.
2. Inline `#tag` — usable in the body (not emitted by the renderer today; reserved
   for user notes in the `## My Notes` section).

---

## Section Order

Every Paper note follows this fixed section order (deterministic renderer in
`_render_note()`):

```
---                    ← YAML frontmatter
---
# <Title>

## Summary

> [!important] Key Contribution
> <claims[0]>

## Key Findings

- <claim 1>
- <claim 2>
...

## Methodology

## Results

## Limitations

> [!warning] Limitation
> <limitations[0]>

- <limitation 1>
...

## My Notes
                       ← user-owned; preserved across --force rewrites
```

---

## Filename Sanitization

Notes live at `Papers/<sanitized_title>.md` under the vault root. Sanitization
rules (implemented in `_sanitize_filename()` in `scripts/note.py`):

1. Strip all Windows-illegal characters: `/ \ : * ? " < > |`
2. Collapse consecutive whitespace to a single space.
3. Strip leading/trailing whitespace.
4. Cap at 200 characters (trim at word boundary if possible).
5. If the result is empty, use `Untitled`.

```python
"Attention: All You Need?"  →  "Attention All You Need"
"BERT / GPT: A Comparison"  →  "BERT  GPT A Comparison"  →  "BERT GPT A Comparison"
```

---

## Encoding and Atomic Write

| Property | Value |
|----------|-------|
| **Character encoding** | UTF-8, no BOM |
| **Line endings** | LF (`\n`) — Obsidian on all platforms handles LF correctly |
| **Write mechanism** | Temp file beside target + `os.replace()` (atomic; no partial note on crash) |
| **Content** | Byte-exact — NO escaping, NO JSON serialization, NO argv token wrapping |

**LaTeX safety:** Characters like `\text{...}`, `\Delta`, `\mathbb{R}` must survive
the write unmodified. Any escaping layer (shell, JSON, argv) corrupts them. The
direct `open(..., encoding="utf-8", newline="\n").write(content)` path preserves
every byte.

### Implementation reference

- **Renderer:** `scripts/note.py` → `_render_frontmatter()`, `_render_note()`
- **Writer:** `scripts/obsidian_cli.py` → `create_note()` (direct write), `preflight()`, `note_exists()`
- **Chokepoint contract:** All vault I/O goes through `scripts/obsidian_cli.py`; no
  other module may write `.md` files to the vault directly.
