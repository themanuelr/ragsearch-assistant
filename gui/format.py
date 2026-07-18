"""gui/format.py -- display-layer-only formatting helpers.

Pure functions that turn raw values into human-readable strings for template
rendering. Nothing here touches the filesystem or any external state -- each
formatter is a total function of its input, safe to register as a Jinja
filter and call from any template with no plumbing (unlike a formatted
sibling key bolted onto one producer's entry dict, which every future
producer of the same raw value would have to re-derive).
"""

_KB = 1024
_MB = 1024 * 1024

_PLACEHOLDER = "--"


def format_file_size(size: object) -> str:
    """Format a byte count as a human-readable string (bytes / KB / MB).

    Chooses the unit by magnitude: under 1 KB shows whole bytes, 1 KB up to
    1 MB shows kilobytes with one decimal place, and 1 MB and above shows
    megabytes with one decimal place. No units beyond MB are supported --
    this formats research-paper PDF sizes, and a gigabyte-scale branch would
    be untestable speculation.

    Never raises: a Jinja filter that raises takes down the whole page
    render, which is a far worse outcome than one unformatted cell. A
    ``None`` or non-numeric input degrades to a short placeholder instead.

    Args:
        size: Byte count. Expected to be an int, but any value a template
            might hand this (``None``, a string, a negative number) is
            tolerated.

    Returns:
        A formatted size string (e.g. ``"512 B"``, ``"1.5 KB"``,
        ``"2.3 MB"``), or a short placeholder (``"--"``) when size is
        missing, negative, or not a number.
    """
    try:
        numeric = float(size)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if numeric < 0:
        return _PLACEHOLDER
    if numeric < _KB:
        return f"{int(numeric)} B"
    if numeric < _MB:
        return f"{numeric / _KB:.1f} KB"
    return f"{numeric / _MB:.1f} MB"
