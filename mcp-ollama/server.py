"""
MCP server that exposes the local Gemma 4 e4b model via Ollama as a tool for Claude Code.
Includes an autonomous research tool with DuckDuckGo web search via tool-calling.
"""

import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from mcp.server.fastmcp import FastMCP

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:e4b"

mcp = FastMCP("ollama-local")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ollama_chat(prompt: str, model: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data["message"]["content"]
    except urllib.error.URLError as e:
        return f"[Ollama error: {e}]"


def _ollama_chat_with_tools(messages: list, model: str, tools: list) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        return {"error": str(e)}


def _web_search(query: str, max_results: int = 6) -> str:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"{i}. {r['title']}\n   URL: {r['href']}\n   {r['body']}")
        return "\n\n".join(parts)
    except ImportError:
        return "[ddgs not installed — run: pip install ddgs]"
    except Exception as e:
        return f"[Search error: {e}]"


_WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information on a topic. Call this whenever you need up-to-date facts, recent events, or information you are uncertain about.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise search query (as you would type into a search engine)."
                }
            },
            "required": ["query"]
        }
    }
}


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def ask_local_model(prompt: str, system: str = "") -> str:
    """
    Send a prompt to the local Gemma 4 e4b model via Ollama.

    IMPORTANT: Only call this tool when the user explicitly requests it,
    for example by saying "use the local model", "use ask_local_model", or
    "use Gemma". Never call this autonomously or on your own initiative.

    Use this for lightweight, high-frequency tasks where speed and cost matter:
    - Summarizing text or search results
    - Reformatting or cleaning up content
    - Extracting structured data (JSON, lists) from text
    - Simple rewrites, paraphrasing, or translation
    - Generating boilerplate or repetitive code snippets
    - Answering factual questions that don't require deep reasoning

    Do NOT use for: architectural decisions, complex debugging, multi-step
    reasoning, or anything that benefits from Claude's full capabilities.

    Args:
        prompt: The task or question for the local model.
        system: Optional system prompt to guide the model's behaviour.
    """
    return _ollama_chat(prompt, DEFAULT_MODEL, system or None)


@mcp.tool()
def research(query: str) -> str:
    """
    Ask the local Gemma 4 12b model to research a topic autonomously using web search.

    IMPORTANT: Only call this tool when the user explicitly requests it,
    for example by saying "research X", "use the local model to research", or
    "ask Gemma to look up". Never call this autonomously or on your own initiative.

    Gemma autonomously decides what to search, can issue multiple searches,
    and synthesizes a final answer from the results. Uses DuckDuckGo — no API key needed.

    Use this for:
    - Questions that need current or up-to-date information
    - Multi-step research that benefits from several searches
    - Topics where web sources improve answer quality

    Args:
        query: The research question or topic to investigate.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a thorough research assistant with access to web search. "
                "Use the web_search tool to find current, accurate information. "
                "You may search multiple times with different queries to gather enough context. "
                "Once you have sufficient information, synthesize a clear, well-structured answer. "
                "Always cite the sources (URLs) you used."
            )
        },
        {
            "role": "user",
            "content": query
        }
    ]

    tools = [_WEB_SEARCH_TOOL_SCHEMA]
    max_iterations = 8
    search_log = []

    for iteration in range(max_iterations):
        response = _ollama_chat_with_tools(messages, DEFAULT_MODEL, tools)

        if "error" in response:
            return f"[Ollama error: {response['error']}]"

        message = response.get("message", {})
        tool_calls = message.get("tool_calls") or []

        # Append assistant turn to history
        assistant_msg = {"role": "assistant", "content": message.get("content", "")}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            # Final answer — prepend search log summary if searches were made
            final = message.get("content", "[No response]")
            if search_log:
                log_str = "\n".join(f"  • {q}" for q in search_log)
                final = f"[Searched {len(search_log)} time(s):\n{log_str}]\n\n{final}"
            return final

        # Execute tool calls and feed results back
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})

            # arguments may arrive as a JSON string in some Ollama versions
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            if name == "web_search":
                q = args.get("query", "")
                search_log.append(q)
                result = _web_search(q)
            else:
                result = f"[Unknown tool: {name}]"

            messages.append({"role": "tool", "content": result})

    return "[Research stopped: reached maximum search iterations without a final answer]"


@mcp.tool()
def process_pdf(path: str) -> str:
    """
    Extract structured PaperJSON from a PDF file.

    Calls scripts/ingest.py as a subprocess and returns the full PaperJSON string.
    Use this to ingest a research paper PDF before generating an Obsidian note.

    Args:
        path: Absolute path to the PDF file to ingest.
    """
    if not os.path.exists(path):
        return f"[process_pdf error: file not found: {path}]"

    server_dir = os.path.dirname(os.path.abspath(__file__))
    ingest_script = os.path.normpath(
        os.path.join(server_dir, "..", "scripts", "ingest.py")
    )

    try:
        result = subprocess.run(
            [sys.executable, ingest_script, "--pdf", path],
            capture_output=True,
            text=True,
            timeout=360,  # must exceed _extract_with_llm's 300s Ollama timeout
        )
        if result.returncode != 0:
            return f"[process_pdf error: ingest.py failed: {result.stderr.strip()}]"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[process_pdf error: ingest.py timed out after 360s]"
    except Exception as e:
        return f"[process_pdf error: {e}]"


@mcp.tool()
def check_ollama_status() -> str:
    """Check whether the local Ollama server is running and list available models."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            return f"Ollama is running. Models: {', '.join(models)}"
    except urllib.error.URLError as e:
        return f"Ollama is not reachable: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
