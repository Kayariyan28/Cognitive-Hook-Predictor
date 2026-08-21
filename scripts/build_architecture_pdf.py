"""Build the SignalFrame architecture and capabilities PDF.

Every number in this document is read from the repository or stated by the
running service. Nothing is estimated.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT = Path("/home/user/Cognitive-Hook-Predictor/docs/SignalFrame-Architecture-and-Capabilities.pdf")

INK = colors.HexColor("#16180F")
MUTED = colors.HexColor("#5C6152")
FAINT = colors.HexColor("#8A8F80")
ACCENT = colors.HexColor("#55670A")
ACCENT_WASH = colors.HexColor("#EDF2CE")
TRIBE = colors.HexColor("#2E5C8A")
TRIBE_WASH = colors.HexColor("#DEE9F3")
RISK = colors.HexColor("#8C3A16")
RISK_WASH = colors.HexColor("#F6E7DF")
RULE = colors.HexColor("#C3C7B8")
PAPER = colors.HexColor("#FFFFFF")

BODY_FONT = "Times-Roman"
BODY_BOLD = "Times-Bold"
BODY_ITALIC = "Times-Italic"
HEAD_FONT = "Helvetica-Bold"
MONO_FONT = "Courier"

styles = getSampleStyleSheet()


def style(name, **kwargs):
    return ParagraphStyle(name, parent=styles["Normal"], **kwargs)


S_TITLE = style("t", fontName=HEAD_FONT, fontSize=27, leading=30, textColor=INK, spaceAfter=6)
S_SUBTITLE = style("st", fontName=BODY_ITALIC, fontSize=12.5, leading=17, textColor=MUTED, spaceAfter=4)
S_KICKER = style("k", fontName="Helvetica", fontSize=7.6, leading=11, textColor=ACCENT, spaceAfter=10)
S_H1 = style("h1", fontName=HEAD_FONT, fontSize=15.5, leading=19, textColor=INK, spaceBefore=16, spaceAfter=7)
S_H2 = style("h2", fontName=HEAD_FONT, fontSize=11, leading=14, textColor=INK, spaceBefore=11, spaceAfter=4)
S_BODY = style("b", fontName=BODY_FONT, fontSize=9.6, leading=13.6, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6)
S_LEAD = style("l", fontName=BODY_FONT, fontSize=10.6, leading=15, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7)
S_SMALL = style("s", fontName=BODY_FONT, fontSize=8.4, leading=11.4, textColor=MUTED, spaceAfter=4)
S_CELL = style("c", fontName=BODY_FONT, fontSize=7.9, leading=10.2, textColor=INK)
S_CELL_MUTED = style("cm", fontName=BODY_FONT, fontSize=7.9, leading=10.2, textColor=MUTED)
S_CELL_HEAD = style("ch", fontName="Helvetica-Bold", fontSize=7.2, leading=9.4, textColor=colors.white)
S_MONO = style("m", fontName=MONO_FONT, fontSize=7.5, leading=10, textColor=INK)
S_BULLET = style("bu", fontName=BODY_FONT, fontSize=9.6, leading=13.4, textColor=INK, leftIndent=11, bulletIndent=2, spaceAfter=3)
S_NOTE = style("n", fontName=BODY_FONT, fontSize=9, leading=12.6, textColor=INK, leftIndent=8, rightIndent=8, spaceBefore=3, spaceAfter=3)


class Rule(Flowable):
    def __init__(self, width, thickness=0.6, color=RULE, space=4):
        super().__init__()
        self.width = width
        self.thickness = thickness
        self.color = color
        self.space = space

    def wrap(self, availWidth, availHeight):
        self.width = availWidth
        return (availWidth, self.thickness + self.space)

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.space, self.width, self.space)


class ArchitectureDiagram(Flowable):
    """The three separated lanes, drawn rather than described."""

    # The drawing runs from height-8 down to height-282, so the declared
    # height must cover that or the next flowable overlaps the diagram.
    def __init__(self, width, height=292):
        super().__init__()
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        self.width = availWidth
        return (availWidth, self.height)

    def _box(self, x, y, w, h, label, sub, fill, stroke, label_color=INK):
        c = self.canv
        c.setFillColor(fill)
        c.setStrokeColor(stroke)
        c.setLineWidth(0.8)
        c.roundRect(x, y, w, h, 3, stroke=1, fill=1)
        c.setFillColor(label_color)
        c.setFont("Helvetica-Bold", 7.4)
        c.drawString(x + 6, y + h - 12, label)
        if sub:
            c.setFillColor(MUTED)
            c.setFont("Times-Roman", 6.6)
            text = c.beginText(x + 6, y + h - 22)
            for line in sub:
                text.textLine(line)
            c.drawText(text)

    def _arrow(self, x1, y1, x2, y2, color=FAINT, dashed=False):
        c = self.canv
        c.setStrokeColor(color)
        c.setLineWidth(0.7)
        if dashed:
            c.setDash(2, 2)
        c.line(x1, y1, x2, y2)
        c.setDash()
        c.setFillColor(color)
        c.circle(x2, y2, 1.4, stroke=0, fill=1)

    def draw(self):
        c = self.canv
        w = self.width
        col = (w - 24) / 3.0

        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 7.4)
        c.drawString(0, self.height - 8, "ONE UPLOADED CLIP  (10-60 s, private, deleted after the job)")

        top = self.height - 26
        self._box(0, top - 30, w, 26, "EVIDENCE JOB  /api/forecast/v1/jobs",
                  ["Authoritative ffprobe probe, bounded workers, atomic publication"], ACCENT_WASH, ACCENT)

        lane_y = top - 132
        lane_h = 92

        self._box(0, lane_y, col, lane_h, "LANE 1  CONTENT SIGNALS",
                  ["Browser-measured indices", "Server PCM / STFT audio", "Media metadata (ffprobe)",
                   "", "Descriptive only.", "Never an audience model."], PAPER, RULE)

        self._box(col + 12, lane_y, col, lane_h, "LANE 2  ENCODER EVIDENCE",
                  ["V-JEPA 2.1 visual windows", "NanoLLaVA keyframes (x2 passes)", "AST AudioSet labels",
                   "Whisper transcript", "Vision / Tesseract on-screen text", "", "behavioralOutcome: false"],
                  PAPER, RULE)

        self._box(2 * (col + 12), lane_y, col, lane_h, "LANE 3  TRIBE v2 CORTICAL",
                  ["Pinned V-JEPA2 -> TRIBE v2", "T x 20,484 fsaverage5 tensor", "Interval / phase / parcel",
                   "descriptors", "", "forecastContribution: false"], TRIBE_WASH, TRIBE, TRIBE)

        for index in range(3):
            self._arrow(col / 2 + index * (col + 12), top - 30, col / 2 + index * (col + 12), lane_y + lane_h)

        bundle_y = lane_y - 44
        self._box(0, bundle_y, w, 34, "EVIDENCE BUNDLE  (8 lanes, every value citable, absent lanes explicit)",
                  ["measured  |  nanollava  |  ast  |  vjepa  |  asr  |  ocr  |  context  |  tribe",
                   "The TRIBE tensor never enters the bundle. Only descriptors do."], ACCENT_WASH, ACCENT)
        for index in range(3):
            self._arrow(col / 2 + index * (col + 12), lane_y, col / 2 + index * (col + 12), bundle_y + 34)

        out_y = bundle_y - 46
        third = (w - 16) / 3.0
        self._box(0, out_y, third, 36, "HOOK READOUT",
                  ["Timeline + checklist.", "Deterministic. No model."], PAPER, ACCENT)
        self._box(third + 8, out_y, third, 36, "INSIGHT / HOOK DOCTOR",
                  ["Cited language from a pinned", "local model. Validator decides."], PAPER, ACCENT)
        self._box(2 * (third + 8), out_y, third, 36, "EXPERIMENTS + VARIANTS",
                  ["Recut, re-measure, compare.", "Signal deltas only."], PAPER, ACCENT)
        for index in range(3):
            self._arrow(third / 2 + index * (third + 8), bundle_y, third / 2 + index * (third + 8), out_y + 36)

        gate_y = out_y - 34
        self._box(0, gate_y, w, 26, "BEHAVIORAL HEADS  -  0 APPROVED, 0 INSTALLED",
                  ["APPROVED_TARGET_CONTRACTS is empty. No probability is available anywhere in the product."],
                  RISK_WASH, RISK, RISK)
        self._arrow(w / 2, out_y, w / 2, gate_y + 26, RISK, dashed=True)


def bullet_list(items, style_=S_BULLET):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{item}", style_) for item in items]


def table(data, widths, head_bg=INK, zebra=True, align_left=True):
    rows = [[Paragraph(cell, S_CELL_HEAD) for cell in data[0]]]
    for row in data[1:]:
        rows.append([cell if isinstance(cell, Paragraph) else Paragraph(cell, S_CELL) for cell in row])
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT" if align_left else "CENTER")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), head_bg),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
    ]
    if zebra:
        for index in range(1, len(rows)):
            if index % 2 == 0:
                commands.append(("BACKGROUND", (0, index), (-1, index), colors.HexColor("#F7F8F3")))
    t.setStyle(TableStyle(commands))
    return t


def callout(title, body, accent=ACCENT, wash=ACCENT_WASH):
    inner = [
        Paragraph(f"<font name='Helvetica-Bold' size='8'>{title}</font>", S_NOTE),
        Paragraph(body, S_NOTE),
    ]
    t = Table([[inner]], colWidths=[None])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), wash),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, accent),
        ("BOX", (0, 0), (-1, -1), 0.4, wash),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
    ]))
    return t


PAGE_W, PAGE_H = A4
MARGIN = 17 * mm
CONTENT_W = PAGE_W - 2 * MARGIN


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
    if doc.page > 1:
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_H - MARGIN + 8, PAGE_W - MARGIN, PAGE_H - MARGIN + 8)
        canvas.setFont("Helvetica", 6.8)
        canvas.setFillColor(FAINT)
        canvas.drawString(MARGIN, PAGE_H - MARGIN + 12, "SIGNALFRAME  /  ARCHITECTURE AND CAPABILITIES")
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 12, "Evidence-first analysis for short-form video")
        canvas.line(MARGIN, MARGIN - 10, PAGE_W - MARGIN, MARGIN - 10)
        canvas.setFont("Helvetica", 6.8)
        canvas.drawString(MARGIN, MARGIN - 19, "No behavioral head is installed. Nothing in this system predicts audience outcomes.")
        canvas.drawRightString(PAGE_W - MARGIN, MARGIN - 19, str(doc.page))
    canvas.restoreState()


def build(story):
    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
        title="SignalFrame: Architecture and Capabilities",
        author="Karan Chandra Dey",
        subject="System architecture, capability inventory, and honest limits",
    )
    frame = Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H - 2 * MARGIN, id="body",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])
    doc.build(story)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from architecture_pdf_content import STORY

    build(STORY)
