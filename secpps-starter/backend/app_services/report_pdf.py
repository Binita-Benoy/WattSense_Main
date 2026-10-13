"""
A tiny helper for the PDF reports (plain reportlab, no extra packages): headings, wrapped
text lines and simple tables, with automatic new pages.
"""
import io

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas


def safe(text) -> str:
    """The built-in PDF fonts only cover Latin-1; anything else would print as a black box."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


class PdfReport:
    MARGIN = 50

    def __init__(self, title: str):
        self.buffer = io.BytesIO()
        self.c = canvas.Canvas(self.buffer, pagesize=A4)
        self.width, self.height = A4
        self.y = self.height - self.MARGIN
        self.c.setFont("Helvetica-Bold", 18)
        self.c.drawString(self.MARGIN, self.y, safe(title))
        self.y -= 30

    def _room(self, needed: float):
        if self.y - needed < self.MARGIN:
            self.c.showPage()
            self.y = self.height - self.MARGIN

    def gap(self, points: float = 8):
        self.y -= points

    def heading(self, text: str):
        self._room(40)
        self.y -= 10
        self.c.setFont("Helvetica-Bold", 13)
        self.c.drawString(self.MARGIN, self.y, safe(text))
        self.y -= 20

    def line(self, text: str, size: float = 10.5, bold: bool = False, indent: float = 0):
        font = "Helvetica-Bold" if bold else "Helvetica"
        max_width = self.width - 2 * self.MARGIN - indent
        for part in simpleSplit(safe(text), font, size, max_width) or [""]:
            self._room(size + 6)
            self.c.setFont(font, size)
            self.c.drawString(self.MARGIN + indent, self.y, part)
            self.y -= size + 4

    def table(self, header, rows, widths):
        def draw(cells, bold):
            self._room(18)
            self.c.setFont("Helvetica-Bold" if bold else "Helvetica", 10)
            x = self.MARGIN
            for cell, w in zip(cells, widths):
                self.c.drawString(x, self.y, safe(cell))
                x += w
            self.y -= 15

        draw(header, True)
        self.c.line(self.MARGIN, self.y + 11, self.MARGIN + sum(widths), self.y + 11)
        for row in rows:
            draw(row, False)

    def finish(self) -> bytes:
        self.c.showPage()
        self.c.save()
        return self.buffer.getvalue()