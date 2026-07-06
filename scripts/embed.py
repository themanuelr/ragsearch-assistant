"""
scripts/embed.py — Embed paper sections into a local ChromaDB `papers` collection
via Ollama's `/api/embed` endpoint (EMBED-01), and query them back for semantic
retrieval (EMBED-02).

Nuclear-task, stateless CLI: reads a PaperJSON v2 cache file (or runs a
--query), does one thing, prints a result, exits. Mirrors scripts/biblio.py's
non-fatal tail-stage contract (run_embed never raises — it returns
"[embed warning: ...]" on any unhandled failure) and scripts/ingest.py's
Ollama HTTP client conventions (_ollama_extraction_call: urllib POST, timeout
handling, "[Ollama timeout/error: ...]" string prefixes).

D-05: embedding calls go DIRECTLY to Ollama's /api/embed — never ChromaDB's
bundled OllamaEmbeddingFunction (which targets the deprecated /api/embeddings
endpoint and would silently defeat the offline-first embedding contract).

OLLAMA_BASE is a hardcoded module constant (localhost only, never
config-overridable) — see PROJECT.md constraints and the phase threat model
(T-05-02).
"""

import argparse
import datetime
import json
import pathlib
import re
import sys
import urllib.error
import urllib.request

OLLAMA_BASE = "http://localhost:11434"
COLLECTION_NAME = "papers"
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
DEFAULT_GEMMA_MODEL = "gemma4:e4b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Emit a timestamped progress line to stderr."""
    print(
        f"[embed {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Ollama HTTP client (mirrors scripts/ingest.py's _ollama_extraction_call)
# ---------------------------------------------------------------------------

def _ollama_embed_call(
    texts: list, model: str = DEFAULT_EMBED_MODEL, timeout: int = 60
):
    """
    POST texts to /api/embed and return one vector per input.

    Returns list[list[float]] on success, or an "[Ollama timeout: ...]" /
    "[Ollama error: ...]" string on any failure — never raises.
    """
    payload = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except TimeoutError:
        return f"[Ollama timeout: no response within {timeout}s]"
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            return f"[Ollama timeout: no response within {timeout}s]"
        return f"[Ollama error: {e}]"
    try:
        return data["embeddings"]
    except (KeyError, TypeError):
        return f"[Ollama error: unexpected response envelope: {data}]"


def _unload_model(model: str, timeout: int = 30) -> None:
    """
    Release `model` from VRAM immediately (D-11 — unload gemma before loading
    nomic-embed-text). Non-fatal: any failure is logged as a warning, never raised.
    """
    payload = json.dumps({"model": model, "keep_alive": 0}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except (TimeoutError, urllib.error.URLError) as e:
        _log(f"gemma unload non-fatal warning: {e}")


# ---------------------------------------------------------------------------
# ChromaDB collection bootstrap
# ---------------------------------------------------------------------------

def _get_collection(config: dict):
    """Return the `papers` PersistentClient collection, creating it with cosine
    HNSW space if it doesn't exist yet (Pitfall #4 — un-updatable after creation)."""
    import chromadb  # noqa: PLC0415

    client = chromadb.PersistentClient(path=config.get("chroma_db_path", "./chroma_db"))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )


# ---------------------------------------------------------------------------
# Deterministic ID construction (D-03)
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _section_slug(heading: str) -> str:
    """Lowercase, non-alnum -> '-', collapse repeats, strip leading/trailing '-'."""
    slug = (heading or "").lower()
    slug = _SLUG_RE.sub("-", slug)
    slug = slug.strip("-")
    return slug or "section"


def _paper_entries(paperjson: dict, config: dict):
    """
    Build (ids, docs, metadatas) for every non-empty section of a PaperJSON v2 dict.

    Reads section["heading"]/section["body"] from extraction.sections[] — NEVER
    section["blocks"][].plain (Pitfall #1: that structure is overwritten by
    ingest.py's Step 10b fill cascade before the cache write). Part index is
    fixed at 0 here; oversize-section splitting is Plan 05.
    """
    from scripts.ingest import _registry_key  # noqa: PLC0415
    from scripts.note import _sanitize_filename  # noqa: PLC0415

    metadata = paperjson.get("extraction", {}).get("metadata", {}) or {}
    sections = paperjson.get("extraction", {}).get("sections", []) or []

    registry_key = _registry_key(metadata)
    title = metadata.get("title") or ""
    doi = metadata.get("doi") or ""
    year = metadata.get("year") or 0
    vault_note = f"Papers/{_sanitize_filename(title)}.md"

    ids: list = []
    docs: list = []
    metadatas: list = []
    for section in sections:
        heading = section.get("heading") or ""
        body = section.get("body") or ""
        if not body.strip():
            continue
        part = 0
        entry_id = f"{registry_key}::{_section_slug(heading)}::{part}"
        ids.append(entry_id)
        docs.append(body)
        metadatas.append({
            "registry_key": registry_key,
            "title": title,
            "heading": heading,
            "part": part,
            "doi": doi,
            "year": year,
            "status": "paper",
            "vault_note": vault_note,
        })
    return ids, docs, metadatas


def _delete_paper_entries(collection, registry_key: str) -> None:
    """Delete-then-write per paper (D-17). No prefix-match mode exists on Chroma's
    ids — this uses the registry_key metadata field instead (Pitfall on ID-prefix
    matching)."""
    collection.delete(where={"registry_key": registry_key})


# ---------------------------------------------------------------------------
# Write path (EMBED-01)
# ---------------------------------------------------------------------------

def run_embed(paperjson: dict, config: dict) -> str:
    """
    Non-fatal tail stage: embed a paper's sections into the `papers` collection.

    Returns the paper's vault-relative note path on success (or on skip-if-present
    no-op), or "[embed warning: ...]" on any unhandled failure (mirrors
    biblio.py's run_biblio contract) — never raises.
    """
    try:
        from scripts.ingest import _registry_key  # noqa: PLC0415
        from scripts.note import _sanitize_filename  # noqa: PLC0415

        metadata = paperjson.get("extraction", {}).get("metadata", {}) or {}
        registry_key = _registry_key(metadata)
        title = metadata.get("title") or ""
        vault_note = f"Papers/{_sanitize_filename(title)}.md"

        collection = _get_collection(config)

        # Skip-if-present (D-13): zero Ollama calls, no gemma unload, on an
        # already-embedded registry_key.
        existing = collection.get(where={"registry_key": registry_key}, limit=1)
        if existing["ids"]:
            _log(f"skip-if-present: {registry_key} already embedded")
            return vault_note

        # Unload gemma before loading nomic-embed-text (D-11).
        gemma_model = config.get("model_name", DEFAULT_GEMMA_MODEL)
        _unload_model(gemma_model)

        ids, docs, metadatas = _paper_entries(paperjson, config)
        if not ids:
            _log(f"no non-empty sections to embed for {registry_key}")
            return vault_note

        embed_model = config.get("embed_model", DEFAULT_EMBED_MODEL)
        timeout = config.get("embed_timeout", 60)
        batch_size = config.get("embed_batch_size", 16) or len(docs)

        vectors: list = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            batch_vectors = _ollama_embed_call(
                [f"{DOC_PREFIX}{text}" for text in batch],
                model=embed_model,
                timeout=timeout,
            )
            if isinstance(batch_vectors, str):
                return f"[embed warning: {batch_vectors}]"
            vectors.extend(batch_vectors)

        # Delete-then-write per paper (D-17) — always pass explicit embeddings=
        # so Chroma never falls back to its bundled ONNX model (Pitfall #5).
        _delete_paper_entries(collection, registry_key)
        collection.upsert(ids=ids, embeddings=vectors, documents=docs, metadatas=metadatas)

        _log(f"embedded {len(ids)} section(s) for {registry_key}")
        return vault_note
    except Exception as e:
        return f"[embed warning: {e}]"


# ---------------------------------------------------------------------------
# Search path (EMBED-02)
# ---------------------------------------------------------------------------

def _search(query: str, n_results: int, config: dict) -> dict:
    """
    Embed `query`, query the `papers` collection, and return paper-grouped,
    ranked results.

    D-06: hits are grouped by registry_key; each paper's score is its BEST
    section score. D-08: excerpt = leading ~300 chars of the Chroma-stored
    document. D-09: score = round(1 - cosine distance, 4), no relevance cutoff
    -- up to n_results papers are always returned. The "stubs" block is an
    empty list here (Plan 03 fills it in, D-15).
    """
    embed_model = config.get("embed_model", DEFAULT_EMBED_MODEL)
    timeout = config.get("embed_timeout", 60)

    query_result = _ollama_embed_call(
        [f"{QUERY_PREFIX}{query}"], model=embed_model, timeout=timeout
    )
    if isinstance(query_result, str):
        return {"papers": [], "stubs": [], "error": query_result}
    query_vector = query_result[0]

    collection = _get_collection(config)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results * 5,  # over-fetch sections so enough distinct papers surface (A3)
        where={"status": "paper"},
        include=["documents", "metadatas", "distances"],
    )

    docs = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    papers: dict = {}
    for doc, meta, dist in zip(docs, metadatas, distances):
        score = round(1 - dist, 4)
        key = meta["registry_key"]
        entry = papers.setdefault(key, {
            "title": meta["title"],
            "registry_key": key,
            "status": meta["status"],
            "vault_note": meta["vault_note"],
            "score": score,
            "sections": [],
        })
        entry["sections"].append({
            "heading": meta["heading"],
            "score": score,
            "excerpt": doc[:300],
        })
        entry["score"] = max(entry["score"], score)  # D-06: rank by BEST section score

    ranked = sorted(papers.values(), key=lambda p: p["score"], reverse=True)[:n_results]
    return {"papers": ranked, "stubs": []}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Read config.json from the repo root (parent of scripts/) with UTF-8 encoding."""
    cfg_path = pathlib.Path(__file__).parent.parent / "config.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("registry_path", "vault_path", "chroma_db_path"):
        if key in cfg and cfg[key]:
            cfg[key] = str(pathlib.Path(cfg[key]).expanduser())
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Embed a PaperJSON v2 cache file's sections into the ChromaDB `papers` "
            "collection, or run a semantic search query against it."
        )
    )
    parser.add_argument(
        "paperjson",
        nargs="?",
        default=None,
        help="Path to a PaperJSON v2 cache file to embed.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Run a semantic search query instead of embedding a file.",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of ranked papers to return in --query mode (default: 5).",
    )
    args = parser.parse_args()

    config = _load_config()

    if args.query is not None:
        result = _search(args.query, args.n_results, config)
        print(json.dumps(result))
        return

    if not args.paperjson:
        parser.error("either a PaperJSON cache-file path or --query is required")

    with open(args.paperjson, encoding="utf-8") as f:
        paperjson = json.load(f)
    print(run_embed(paperjson, config))


if __name__ == "__main__":
    # Windows cp1252 guard: wrap stdout in UTF-8 before any print (D-PATTERNS).
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
