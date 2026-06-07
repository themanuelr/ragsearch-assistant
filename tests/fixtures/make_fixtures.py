"""
Generate reproducible fixture PDFs for tests/fixtures/.

Usage: python tests/fixtures/make_fixtures.py

Produces:
  sample_onecol.pdf   — single-column academic paper with title, abstract, sections
  sample_twocol.pdf   — two-column paper with arXiv ID in footer (IEEE layout)
  sample_scanned.pdf  — image-only PDF with zero extractable text chars
"""

import io
import os

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

# ---------------------------------------------------------------------------
# directory resolution — always write next to this script
# ---------------------------------------------------------------------------
FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


def make_onecol():
    """Single-column paper: title, author, abstract, introduction, references."""
    path = os.path.join(FIXTURES_DIR, "sample_onecol.pdf")
    c = canvas.Canvas(path, pagesize=LETTER)
    width, height = LETTER

    y = height - 1.0 * inch

    # Title (large font)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(1.0 * inch, y, "Sample Single Column Paper")
    y -= 0.4 * inch

    # Author line
    c.setFont("Helvetica", 12)
    c.drawString(1.0 * inch, y, "Alice Author, Bob Collaborator")
    y -= 0.5 * inch

    # Abstract heading
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.0 * inch, y, "Abstract")
    y -= 0.25 * inch

    c.setFont("Helvetica", 10)
    abstract_text = (
        "This paper presents a comprehensive study of sample PDF generation "
        "for reproducible test fixtures. We demonstrate that programmatically "
        "generated PDFs can reliably stand in for real academic papers in "
        "automated test suites. Results confirm that pdfplumber extracts text "
        "correctly from these fixtures."
    )
    # Wrap text manually
    words = abstract_text.split()
    line = ""
    for word in words:
        test_line = (line + " " + word).strip()
        if c.stringWidth(test_line, "Helvetica", 10) < (width - 2.0 * inch):
            line = test_line
        else:
            c.drawString(1.0 * inch, y, line)
            y -= 0.2 * inch
            line = word
    if line:
        c.drawString(1.0 * inch, y, line)
        y -= 0.4 * inch

    # Introduction heading
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.0 * inch, y, "1. Introduction")
    y -= 0.25 * inch

    c.setFont("Helvetica", 10)
    intro_lines = [
        "Research in natural language processing has advanced rapidly in recent years.",
        "This introduction provides the context and motivation for the present work.",
        "We outline the key contributions and the structure of the remainder of this paper.",
        "Section 2 describes the methodology. Section 3 presents results. Section 4 concludes.",
    ]
    for line_text in intro_lines:
        c.drawString(1.0 * inch, y, line_text)
        y -= 0.2 * inch
    y -= 0.2 * inch

    # Second paragraph in intro
    c.drawString(1.0 * inch, y, "Prior work has established foundational techniques in this domain.")
    y -= 0.2 * inch
    c.drawString(1.0 * inch, y, "Our approach extends these methods with novel contributions.")
    y -= 0.4 * inch

    # References heading
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.0 * inch, y, "References")
    y -= 0.25 * inch

    c.setFont("Helvetica", 10)
    c.drawString(1.0 * inch, y, "[1] Smith, J. and Doe, A. (2020). Foundations of text extraction. Journal A, 1(1), 1-10.")
    y -= 0.2 * inch
    c.drawString(1.0 * inch, y, "[2] Brown, C. et al. (2021). Advances in PDF parsing. Conference B, pp. 50-60.")

    c.save()
    print(f"Created: {path}")


def make_twocol():
    """
    Two-column paper: single-column title block at top, two-column body below.
    arXiv ID in footer for extraction testing.
    """
    path = os.path.join(FIXTURES_DIR, "sample_twocol.pdf")
    c = canvas.Canvas(path, pagesize=LETTER)
    width, height = LETTER

    # ---------------------------------------------------------------------------
    # Single-column title block (top 30%)
    # ---------------------------------------------------------------------------
    y = height - 0.8 * inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(1.0 * inch, y, "Two-Column Layout Sample for IEEE-Style Papers")
    y -= 0.35 * inch

    c.setFont("Helvetica", 11)
    c.drawString(1.0 * inch, y, "Carol Researcher, David Scholar")
    y -= 0.25 * inch

    c.setFont("Helvetica-Bold", 10)
    c.drawString(1.0 * inch, y, "Abstract")
    y -= 0.2 * inch

    c.setFont("Helvetica", 9)
    c.drawString(1.0 * inch, y, "This paper uses a two-column layout to test column detection heuristics.")
    y -= 0.15 * inch
    c.drawString(1.0 * inch, y, "The body is split into left and right columns for reading-order verification.")

    # ---------------------------------------------------------------------------
    # Two-column body (bottom 70%): use textobject frames for two columns
    # ---------------------------------------------------------------------------
    col_top = height * 0.65  # start of two-column section
    col_height = col_top - 0.8 * inch  # leave room for footer
    col_width = (width - 1.5 * inch) / 2  # two equal columns with gutter

    left_x = 0.5 * inch
    right_x = left_x + col_width + 0.5 * inch  # 0.5 inch gutter

    # Left column
    c.setFont("Helvetica-Bold", 10)
    c.drawString(left_x, col_top, "Section A: Left Column")
    c.setFont("Helvetica", 9)
    left_body = [
        "This is the left body text. The left column",
        "contains Section A content that should be",
        "read before Section B in the right column.",
        "Reading order: left-then-right per IEEE style.",
        "More left column content follows here.",
        "Additional left body for length verification.",
        "Final left column line for completeness.",
    ]
    ly = col_top - 0.2 * inch
    for line_text in left_body:
        c.drawString(left_x, ly, line_text)
        ly -= 0.18 * inch

    # Right column
    c.setFont("Helvetica-Bold", 10)
    c.drawString(right_x, col_top, "Section B: Right Column")
    c.setFont("Helvetica", 9)
    right_body = [
        "This is the right body text. The right column",
        "contains Section B content read after Section A.",
        "Two-column detection relies on x-coordinate counts.",
        "Words in the right half should exceed 30 percent.",
        "This confirms the D-06 heuristic works correctly.",
        "Additional right body for word count balance.",
        "Final right column line matches left column length.",
    ]
    ry = col_top - 0.2 * inch
    for line_text in right_body:
        c.drawString(right_x, ry, line_text)
        ry -= 0.18 * inch

    # arXiv footer
    c.setFont("Helvetica", 8)
    c.drawString(left_x, 0.6 * inch, "arXiv:2301.00001v1 [cs.AI] 1 Jan 2023")

    c.save()
    print(f"Created: {path}")


def make_scanned():
    """
    Image-only PDF: one page with a colored rectangle image and ZERO text.
    Total pdfplumber char count will be 0.
    """
    from PIL import Image as PILImage

    path = os.path.join(FIXTURES_DIR, "sample_scanned.pdf")
    c = canvas.Canvas(path, pagesize=LETTER)
    width, height = LETTER

    # Create a small colored rectangle in memory using PIL
    img = PILImage.new("RGB", (400, 300), color=(180, 200, 220))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    # Draw the image onto the PDF page (no text at all)
    from reportlab.lib.utils import ImageReader
    img_reader = ImageReader(img_bytes)
    c.drawImage(img_reader, 1.0 * inch, 3.0 * inch, width=4.0 * inch, height=3.0 * inch)

    c.save()
    print(f"Created: {path}")


if __name__ == "__main__":
    make_onecol()
    make_twocol()
    make_scanned()
    print("All fixtures generated.")
