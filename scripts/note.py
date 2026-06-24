"""
Note generation pipeline: analysis fill (7 LLM calls) + deterministic render + vault write.

Reads a PaperJSON v2 dict (in-memory or from cache file), runs 7 gemma4:e4b
analysis calls (summary, claims, methods_overview, results, limitations, topics,
open_questions), renders a deterministic Obsidian note with YAML frontmatter and
callouts, and writes it to Papers/<sanitized title>.md via the obsidian_cli
chokepoint.

Run standalone:  python scripts/note.py --paperjson <cache>.json [--force]
"""
