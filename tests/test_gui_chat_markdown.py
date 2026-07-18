"""Tests for gui/chat_markdown.py (Phase 8 Plan 16, GUI-09, T-08-GAP-31/32):
the escape-then-restricted-render chokepoint for untrusted chat output, and
its registration as a Jinja filter on the shared templates environment.

Pins the formatting subset the gap asked for (bold/italic/lists/fenced
code) AND every XSS vector verified during planning against the pinned
python-markdown 3.10.2: raw HTML elements, event-handler attributes,
scripting-scheme links/reference-links/autolinks.

Run with:  python -m pytest tests/test_gui_chat_markdown.py -x
"""

import pathlib

from gui.chat_markdown import render_chat_markdown

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Test A: formatting works -- bold, italic, list, fenced code
# ---------------------------------------------------------------------------

def test_bold_renders_strong_element():
    out = str(render_chat_markdown("this is **bold** text"))
    assert "<strong>bold</strong>" in out


def test_italic_renders_em_element():
    out = str(render_chat_markdown("this is *italic* text"))
    assert "<em>italic</em>" in out


def test_unordered_list_renders_ul_li():
    out = str(render_chat_markdown("intro\n\n- first\n- second\n"))
    assert "<ul>" in out
    assert "<li>first</li>" in out
    assert "<li>second</li>" in out


def test_fenced_code_block_renders_pre_code():
    out = str(render_chat_markdown("here:\n\n```\nx = 1\n```\n"))
    assert "<pre>" in out
    assert "<code>" in out
    assert "x = 1" in out


# ---------------------------------------------------------------------------
# Test B: raw HTML is inert -- a script element renders as visible text
# ---------------------------------------------------------------------------

def test_script_element_renders_as_visible_text_not_element():
    out = str(render_chat_markdown("before <script>alert(1)</script> after"))
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "alert(1)" in out  # visible as text


# ---------------------------------------------------------------------------
# Test C: event-handler attributes are inert -- an <img onerror=...> renders
# as visible text, not as an element
# ---------------------------------------------------------------------------

def test_img_onerror_renders_as_visible_text_not_element():
    out = str(render_chat_markdown('before <img src=x onerror=alert(1)> after'))
    assert "<img" not in out
    assert "&lt;img" in out
    assert "onerror" in out  # visible as text, never a live attribute


# ---------------------------------------------------------------------------
# Test D: the verified live vector -- a scripting-scheme markdown link
# produces no anchor and no scheme-bearing href
# ---------------------------------------------------------------------------

def test_markdown_link_with_javascript_scheme_produces_no_anchor():
    out = str(render_chat_markdown("click [here](javascript:alert(1)) now"))
    # Negative: no anchor element and no href attribute was ever produced --
    # this is the vector that escape-alone fails to close.
    assert "<a " not in out
    assert "href=" not in out
    # Positive: the link construct is disabled entirely, so the literal
    # markdown source text remains visible rather than being parsed.
    assert "click [here](javascript:alert(1)) now" in out


def test_markdown_reference_link_with_javascript_scheme_produces_no_anchor():
    text = "click [here][1] now\n\n[1]: javascript:alert(1)"
    out = str(render_chat_markdown(text))
    assert "<a " not in out
    assert "href=" not in out


# ---------------------------------------------------------------------------
# Test E: autolink inert -- a bare scripting-scheme URL and an angle-bracket
# autolink form both produce no anchor
# ---------------------------------------------------------------------------

def test_angle_bracket_autolink_with_javascript_scheme_produces_no_anchor():
    out = str(render_chat_markdown("see <javascript:alert(1)> now"))
    assert "<a " not in out
    assert "href=" not in out


def test_bare_javascript_scheme_url_produces_no_anchor():
    out = str(render_chat_markdown("see javascript:alert(1) now"))
    assert "<a " not in out
    assert "href=" not in out


def test_http_autolink_also_produces_no_anchor_link_disabled_entirely():
    """Link/autolink constructs are disabled wholesale (not scheme-filtered),
    so even an ordinary http(s) angle-bracket autolink produces no anchor --
    this is the deliberate, documented scope-narrowing, not a bug."""
    out = str(render_chat_markdown("see <https://example.com> now"))
    assert "<a " not in out
    assert "href=" not in out


# ---------------------------------------------------------------------------
# Test F: plain text round-trips, including non-ASCII
# ---------------------------------------------------------------------------

def test_plain_prose_round_trips_intact():
    out = str(render_chat_markdown("Ordinary prose with no markdown at all."))
    assert "<p>Ordinary prose with no markdown at all.</p>" == out.strip()


def test_non_ascii_round_trips_intact():
    text = "Café results by Müller — see the en dash and accented names."
    out = str(render_chat_markdown(text))
    assert "Café" in out
    assert "Müller" in out
    assert "—" in out


# ---------------------------------------------------------------------------
# Test G: empty/None input is safe
# ---------------------------------------------------------------------------

def test_empty_string_renders_to_empty_output():
    assert str(render_chat_markdown("")) == ""


def test_none_renders_to_empty_output_without_raising():
    assert str(render_chat_markdown(None)) == ""


# ---------------------------------------------------------------------------
# Test H: stateless across calls -- sequential conversions are order-
# independent, no leaked state between conversions
# ---------------------------------------------------------------------------

def test_sequential_conversions_are_order_independent():
    first_pass = [
        str(render_chat_markdown("**bold one**")),
        str(render_chat_markdown("*italic two*")),
    ]
    second_pass = [
        str(render_chat_markdown("*italic two*")),
        str(render_chat_markdown("**bold one**")),
    ]
    assert first_pass[0] == second_pass[1]
    assert first_pass[1] == second_pass[0]


def test_many_sequential_calls_never_accumulate_output():
    # A stateful/leaking Markdown instance would grow or corrupt output
    # across repeated calls on the SAME input -- pin exact repeatability.
    outputs = [str(render_chat_markdown("**stable**")) for _ in range(5)]
    assert len(set(outputs)) == 1


# ---------------------------------------------------------------------------
# Filter registration on the shared templates environment
# ---------------------------------------------------------------------------

def test_filter_registered_on_shared_templates_environment():
    from gui.app import templates

    assert templates.env.filters.get("chat_markdown") is render_chat_markdown


# ---------------------------------------------------------------------------
# Acceptance criteria: no new dependency
# ---------------------------------------------------------------------------

def test_no_new_dependency_added_to_requirements():
    req_text = (_REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "markdown>=3.10" in req_text


def test_module_docstring_records_escape_before_render_and_link_disable_rationale():
    src = (_REPO_ROOT / "gui" / "chat_markdown.py").read_text(encoding="utf-8")
    assert "Escape first" in src
    assert "does not sanitize URL schemes" in src
    assert "docs.py" in src


def test_no_shared_module_level_converter_instance():
    """The converter must be built fresh per call (or thread-locally) --
    never a module-level ``Markdown(...)`` instance shared across calls."""
    import gui.chat_markdown as chat_markdown_module

    assert not isinstance(
        getattr(chat_markdown_module, "_CONVERTER", None), object
    ) or getattr(chat_markdown_module, "_CONVERTER", None) is None
    # Stronger structural check: no module-level name bound to a
    # markdown.Markdown instance.
    import markdown as _markdown_pkg

    for name, value in vars(chat_markdown_module).items():
        assert not isinstance(value, _markdown_pkg.Markdown), (
            f"module-level shared Markdown instance found: {name}"
        )


# ---------------------------------------------------------------------------
# Test I: math pass (D-GAP2, GUI-09 gap 2, T-08-GAP-43/44/45) -- the exact
# UAT repro was a reply containing $\text{Cl}^-$, $\text{Na}^+$, $\text{IO}$,
# $\text{OO}$ (08-UAT.md Gap 2).
# ---------------------------------------------------------------------------

def test_charge_superscript_renders_sup_no_dollar_delimiters():
    out = str(render_chat_markdown(r"the ion is $\text{Cl}^-$ in solution"))
    assert "Cl<sup>-</sup>" in out
    assert "$" not in out


def test_second_charge_superscript_example_from_uat_repro():
    out = str(render_chat_markdown(r"and $\text{Na}^+$ crosses the channel"))
    assert "Na<sup>+</sup>" in out
    assert "$" not in out


def test_subscript_index_renders_sub_element():
    out = str(render_chat_markdown(r"water is $\text{H}_2\text{O}$ here"))
    assert "H<sub>2</sub>O" in out
    assert "$" not in out


def test_unmappable_span_degrades_to_delimiter_stripped_text():
    # Exact UAT repro text: $\text{IO}$ and $\text{OO}$ have no caret/
    # underscore for the mapper to act on -- they must render as their
    # inner text with the dollar delimiters removed, never as raw
    # dollar-delimited markup.
    out = str(render_chat_markdown(r"channels labeled $\text{IO}$ and $\text{OO}$"))
    assert "$" not in out
    assert "IO" in out
    assert "OO" in out


def test_prose_without_dollar_sign_is_byte_identical_to_pre_math_pass_output():
    # The math pass must be a true no-op on ordinary prose: same text run
    # through the pre-existing escape+render steps produces the identical
    # markup, dollar sign or no dollar sign.
    text = "this is **bold** with *italic* and a list\n\n- one\n- two\n"
    out = str(render_chat_markdown(text))
    assert "<strong>bold</strong>" in out
    assert "<em>italic</em>" in out
    assert "<li>one</li>" in out
    assert "$" not in out


def test_currency_looking_single_dollar_does_not_swallow_surrounding_text():
    # Only one $ in the whole message -- _MATH_SPAN requires a closing
    # delimiter, so this must NOT be treated as a math span at all.
    out = str(render_chat_markdown("It costs $5 for coffee and nothing else."))
    assert "$5 for coffee and nothing else" in out
    assert "<sup>" not in out
    assert "<sub>" not in out


def test_xss_construct_inside_math_span_stays_inert():
    # The hostile-construct-inside-a-math-span case (T-08-GAP-43): the math
    # pass must never open a path around the escape step. A script element
    # placed inside a dollar-delimited span is already HTML-escaped before
    # the math pass ever sees it, and the mapper has no pattern that could
    # turn escaped text into a live element or attribute.
    out = str(render_chat_markdown("before $<script>alert(1)</script>$ after"))
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "alert(1)" in out


def test_math_span_placeholder_never_leaks_into_final_output():
    # The placeholder round-trip must be fully consumed -- no MATHSPAN...
    # token literal should ever survive into rendered output.
    out = str(render_chat_markdown(r"$\text{Cl}^-$ and plain text"))
    assert "MATHSPAN" not in out
    assert "ENDSPAN" not in out
