"""gui/chat_markdown.py -- the SOLE sanctioned chokepoint for turning
untrusted local-LLM chat output into HTML (Phase 8 Plan 16, GUI-09,
T-08-GAP-31/32).

Chat replies are model output derived from untrusted PDFs/URLs (D-06). This
is a genuinely different trust class from ``gui/routes/docs.py``'s
``markdown.markdown()`` + ``| safe`` idiom, which renders a fixed,
repo-authored whitelist (D-21) -- that idiom must NEVER be copied here.

Pipeline, in this exact order (verified against python-markdown 3.10.2, the
version pinned in requirements.txt):

1. Escape first. The raw text is HTML-escaped (angle brackets + ampersand;
   quote-escaping is skipped -- the output is always emitted as element
   content, never into an attribute) BEFORE markdown ever sees it. This
   alone turns any raw HTML in the model's output (``<script>``, an
   ``onerror``-bearing ``<img>``) into inert visible text.
2. Render a restricted subset. Convert with python-markdown, enabling only
   ``fenced_code`` and ``sane_lists`` -- exactly the formatting subset the
   gap asked for (bold/italic/lists/code). No raw-HTML-friendly extension,
   ``attr_list``, or ``md_in_html`` is ever enabled here.
3. Disable link and image constructs. Escaping alone is NOT sufficient:
   verified empirically during planning, ``[click](javascript:alert(1))``
   still renders as a live ``<a href="javascript:alert(1)">`` anchor under
   default settings -- python-markdown does not sanitize URL schemes. The
   link/image/reference/autolink/automail inline constructs are therefore
   deregistered so no ``href``/``src`` can ever be produced from model text
   at all. Deregistration is defensive (``strict=False``): an unknown
   pattern name in a future python-markdown version is a no-op, never an
   exception -- the OUTCOME (no anchors) is what
   ``tests/test_gui_chat_markdown.py`` pins, not the registry's internals,
   so a version bump that silently reopens this vector via a renamed
   pattern would be caught by the XSS regression tests, not by a crash
   here.
4. Return markup-wrapped output so Jinja emits it unescaped -- the content
   is trusted-by-construction once steps 1-3 have run.

Thread safety: a python-markdown ``Markdown`` instance is stateful (it
needs ``.reset()`` between conversions) and FastAPI runs sync routes in a
threadpool, so two chat renders can genuinely overlap. The converter is
therefore built FRESH on every call -- never shared as a module-level
instance -- chat volume is a handful of messages per page render, so the
construction cost is immaterial next to correctness.

Fail-soft: a conversion failure degrades to escaped plain text and logs a
warning -- a malformed reply must never 500 the chat page.
"""

import datetime
import html
import sys

import markdown
from markupsafe import Markup

# Inline pattern names that build anchors/images, verified against the
# pinned python-markdown 3.10.2 registry (reference/link/image_link/
# image_reference/short_reference/short_image_ref/autolink/automail).
# Deregistered defensively (strict=False) -- see module docstring step 3.
_LINK_IMAGE_CONSTRUCTS = (
    "reference",
    "link",
    "image_link",
    "image_reference",
    "short_reference",
    "short_image_ref",
    "autolink",
    "automail",
)

# Deliberately narrow: only what the gap requires (bold/italic/lists/fenced
# code). Do not widen this list as a "convenience" -- see module docstring.
_EXTENSIONS = ["fenced_code", "sane_lists"]


def _log(msg: str) -> None:
    """Emit a timestamped warning line to stderr (CONVENTIONS.md `_log`
    style: `[chat_markdown YYYY-MM-DD HH:MM:SS] msg`)."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[chat_markdown {ts}] {msg}", file=sys.stderr)


def _build_converter() -> markdown.Markdown:
    """Construct a fresh, restricted python-markdown converter.

    Never share this across calls -- see the module docstring's Thread
    safety note.
    """
    converter = markdown.Markdown(extensions=_EXTENSIONS)
    for name in _LINK_IMAGE_CONSTRUCTS:
        converter.inlinePatterns.deregister(name, strict=False)
    return converter


def render_chat_markdown(text: str | None) -> Markup:
    """Render untrusted chat text into markup safe to emit via `| safe`.

    Escapes ``text`` first, then converts the restricted markdown subset
    (bold/italic/lists/fenced code) with the link/image/reference/autolink
    inline constructs removed, so no scripting-scheme anchor or raw HTML
    element can ever reach the DOM (T-08-GAP-31).

    Args:
        text: Untrusted chat message content. ``None``/empty renders to
            empty output.

    Returns:
        A ``markupsafe.Markup`` instance -- Jinja will not re-escape it.
        Never raises: a rendering failure degrades to escaped plain text
        and logs a warning (T-08-GAP-34).
    """
    if not text:
        return Markup("")
    try:
        escaped = html.escape(text, quote=False)
        converter = _build_converter()
        rendered = converter.convert(escaped)
        return Markup(rendered)
    except Exception as exc:
        _log(f"render_chat_markdown failed, degraded to escaped plain text: {exc}")
        return Markup(html.escape(text, quote=False))
