"""Build small PDFs in memory for tests."""

from __future__ import annotations

import io


def text_pdf(pages: list[str]) -> bytes:
    """A PDF with one text page per string (uses reportlab, a dev dependency)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    for page in pages:
        y = 800
        for line in page.splitlines():
            pdf.drawString(50, y, line)
            y -= 16
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def blank_pdf() -> bytes:
    """A valid PDF whose only page has no text, like a scanned image."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
