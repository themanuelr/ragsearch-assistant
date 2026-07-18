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
2. Extract inline math. Find ``$...$``/``$$...$$`` spans in the escaped text
   using the project's one shared math-span definition -- ``_MATH_SPAN``,
   imported from ``scripts.note`` rather than re-derived here (Don't
   Hand-Roll; see ``gui/scan.py``'s module docstring for the established
   precedent of importing a private name from ``scripts.note``). Each span
   is replaced with an opaque placeholder token and its rendered math HTML
   is stashed in a per-call list. The round-trip is required, not
   incidental: a math span's underscores/carets would otherwise be mangled
   by python-markdown's emphasis parsing (a subscript underscore reads as
   ``_emphasis_``). The placeholder's random component is generated fresh
   per call with ``secrets.token_hex`` so a reply containing a literal
   placeholder-looking string cannot forge a token and relocate rendered
   HTML into a later position (T-08-GAP-44).
3. Render a restricted subset. Convert with python-markdown, enabling only
   ``fenced_code`` and ``sane_lists`` -- exactly the formatting subset the
   gap asked for (bold/italic/lists/code). No raw-HTML-friendly extension,
   ``attr_list``, or ``md_in_html`` is ever enabled here. The placeholder
   tokens are plain alphanumeric text with no markdown-significant
   punctuation, so this step never mangles them.
4. Disable link and image constructs. Escaping alone is NOT sufficient:
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
5. Substitute math placeholders. Each placeholder token in the rendered
   HTML is replaced with its stashed rendered math HTML -- built only from
   the already-escaped span text, using a fixed two-element tag vocabulary
   (``sup``/``sub``) with zero attributes ever emitted. No URL scheme and
   no event handler can originate from this pass (T-08-GAP-43).
6. Return markup-wrapped output so Jinja emits it unescaped -- the content
   is trusted-by-construction once steps 1-5 have run.

D-GAP2 ceiling: the math pass (step 2/5) handles the subset the UAT gap
demonstrated -- LaTeX text-mode commands (``\\text{...}``) and simple
caret/underscore super/subscripts -- via a small server-side Python mapper,
not a full math renderer. A full-fidelity renderer (KaTeX/MathJax) would
require bundling offline font/JS assets into ``gui/static/``, a
supply-chain and asset-pipeline change disproportionate to this cosmetic
gap; deliberately left as future work if wanted.

Thread safety: a python-markdown ``Markdown`` instance is stateful (it
needs ``.reset()`` between conversions) and FastAPI runs sync routes in a
threadpool, so two chat renders can genuinely overlap. The converter is
therefore built FRESH on every call -- never shared as a module-level
instance -- chat volume is a handful of messages per page render, so the
construction cost is immaterial next to correctness.

Fail-soft: a conversion failure (including a malformed math span) degrades
to escaped plain text and logs a warning -- a malformed reply must never
500 the chat page.
"""

import datetime
import html
import re
import secrets
import sys

import markdown
from markupsafe import Markup

from scripts.note import _MATH_SPAN

# Inline pattern names that build anchors/images, verified against the
# pinned python-markdown 3.10.2 registry (reference/link/image_link/
# image_reference/short_reference/short_image_ref/autolink/automail).
# Deregistered defensively (strict=False) -- see module docstring step 4.
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

# ---------------------------------------------------------------------------
# Math span mapper (D-GAP2, GUI-09 gap 2) -- handles the subset the UAT gap
# demonstrated: a LaTeX text-mode command wrapping a symbol, plus a
# caret-introduced superscript or underscore-introduced subscript. Anything
# else is left as delimiter-stripped plain text rather than failing.
# ---------------------------------------------------------------------------

# \text{...} -- unwrap the LaTeX text-mode command to its braced argument.
_MATH_TEXT_CMD = re.compile(r'\\text\{([^{}]*)\}')

# Superscript: a braced group (^{...}) or a single character that is not
# whitespace, a brace, a caret, or an underscore.
_MATH_SUP_GROUP = re.compile(r'\^\{([^{}]*)\}')
_MATH_SUP_CHAR = re.compile(r'\^([^\s{}^_])')

# Subscript: same shape, underscore-introduced.
_MATH_SUB_GROUP = re.compile(r'_\{([^{}]*)\}')
_MATH_SUB_CHAR = re.compile(r'_([^\s{}^_])')


def _log(msg: str) -> None:
    """Emit a timestamped warning line to stderr (CONVENTIONS.md `_log`
    style: `[chat_markdown YYYY-MM-DD HH:MM:SS] msg`)."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[chat_markdown {ts}] {msg}", file=sys.stderr)


def _render_math_inner(inner: str) -> str:
    """Map the LaTeX subset the UAT gap demonstrated to HTML.

    Input MUST already be HTML-escaped (it is a substring of the
    already-escaped chat text). Output is assembled only from that escaped
    text plus a fixed two-tag vocabulary (``sup``/``sub``) -- no attribute
    is ever emitted, so no URL scheme and no event handler can originate
    here (T-08-GAP-43). Anything not recognized by the mapper is returned
    unchanged, delimiters already stripped by the caller.
    """
    inner = _MATH_TEXT_CMD.sub(r'\1', inner)
    inner = _MATH_SUP_GROUP.sub(r'<sup>\1</sup>', inner)
    inner = _MATH_SUP_CHAR.sub(r'<sup>\1</sup>', inner)
    inner = _MATH_SUB_GROUP.sub(r'<sub>\1</sub>', inner)
    inner = _MATH_SUB_CHAR.sub(r'<sub>\1</sub>', inner)
    return inner


def _render_math_span(match: re.Match) -> str:
    """Strip the ``$``/``$$`` delimiters off a matched math span and map
    its inner LaTeX subset to HTML (never re-emits raw dollar-delimited
    markup, per the gap's done criteria)."""
    inner = match.group(0).strip("$")
    return _render_math_inner(inner)


def _extract_math_placeholders(escaped: str) -> tuple[str, str, list[str]]:
    """Replace each math span in `escaped` with an opaque placeholder token.

    Returns ``(placeholdered_text, token, rendered_spans)``. ``token`` is a
    per-call ``secrets.token_hex`` component embedded in every placeholder
    so model output cannot forge a placeholder-looking string and relocate
    rendered HTML into a later position (T-08-GAP-44). No-op (empty token,
    empty list) when the text contains no ``$`` at all -- text without a
    dollar sign is therefore byte-identical to the pre-math-pass renderer's
    output.
    """
    if "$" not in escaped:
        return escaped, "", []

    token = secrets.token_hex(16)
    rendered_spans: list[str] = []

    def _stash(match: re.Match) -> str:
        rendered_spans.append(_render_math_span(match))
        return f"MATHSPAN{token}{len(rendered_spans) - 1}ENDSPAN"

    placeholdered = _MATH_SPAN.sub(_stash, escaped)
    return placeholdered, token, rendered_spans


def _substitute_math_placeholders(rendered: str, token: str, math_html: list[str]) -> str:
    """Replace every placeholder token in `rendered` with its stashed math
    HTML. No-op when `token` is empty (the no-math-span fast path)."""
    if not token:
        return rendered

    placeholder_re = re.compile(rf"MATHSPAN{re.escape(token)}(\d+)ENDSPAN")

    def _restore(match: re.Match) -> str:
        return math_html[int(match.group(1))]

    return placeholder_re.sub(_restore, rendered)


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

    Escapes ``text`` first, extracts and renders inline math spans (D-GAP2),
    then converts the restricted markdown subset (bold/italic/lists/fenced
    code) with the link/image/reference/autolink inline constructs removed,
    so no scripting-scheme anchor or raw HTML element can ever reach the DOM
    (T-08-GAP-31) -- including when a hostile construct sits inside a math
    span, since the math pass only ever operates on already-escaped text.

    Args:
        text: Untrusted chat message content. ``None``/empty renders to
            empty output.

    Returns:
        A ``markupsafe.Markup`` instance -- Jinja will not re-escape it.
        Never raises: a rendering failure (including a malformed math span)
        degrades to escaped plain text and logs a warning (T-08-GAP-34).
    """
    if not text:
        return Markup("")
    try:
        escaped = html.escape(text, quote=False)
        placeholdered, token, math_html = _extract_math_placeholders(escaped)
        converter = _build_converter()
        rendered = converter.convert(placeholdered)
        rendered = _substitute_math_placeholders(rendered, token, math_html)
        return Markup(rendered)
    except Exception as exc:
        _log(f"render_chat_markdown failed, degraded to escaped plain text: {exc}")
        return Markup(html.escape(text, quote=False))
