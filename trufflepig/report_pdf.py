"""Native, searchable PDF rendering of the same sections used by Markdown."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

from matplotlib import get_data_path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    Image,
    KeepTogether,
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)

from .report_document import find_figure, load_report_document
from .report_language import report_inline_tokens


def report_inline_html(text: str) -> str:
    """Convert supported inline Markdown to PDF text markup, preserving HLA.

    User HTML is escaped. Images remain their alternative text; rendering prose
    never fetches remote resources or interprets a supplied string as code.
    """
    tokens = report_inline_tokens(text)
    rendered = []
    tags = {"strong_open": "<b>", "strong_close": "</b>", "em_open": "<i>", "em_close": "</i>"}
    for token in tokens:
        if token.type in tags:
            rendered.append(tags[token.type])
        elif token.type in {"text", "image", "html_inline"}:
            rendered.append(
                escape(token.content).replace("—", "-").replace("–", "-").replace("‑", "-")
            )
        elif token.type == "code_inline":
            rendered.append('<font name="ReportMono">' + escape(token.content) + "</font>")
        elif token.type == "link_open":
            rendered.append(
                '<link href="'
                + escape(token.attrGet("href") or "", quote=True)
                + '" color="#245b80">'
            )
        elif token.type == "link_close":
            rendered.append("</link>")
        elif token.type in {"softbreak", "hardbreak"}:
            rendered.append("<br/>")
    return "".join(rendered)


def report_pdf_styles() -> dict:
    """Use packaged fonts so Unicode identifiers render consistently across hosts."""
    fonts = Path(get_data_path()) / "fonts" / "ttf"
    for name, filename in (
        ("Report", "DejaVuSans.ttf"),
        ("ReportBold", "DejaVuSans-Bold.ttf"),
        ("ReportItalic", "DejaVuSans-Oblique.ttf"),
        ("ReportBoldItalic", "DejaVuSans-BoldOblique.ttf"),
        ("ReportMono", "DejaVuSansMono.ttf"),
    ):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily(
        "Report",
        normal="Report",
        bold="ReportBold",
        italic="ReportItalic",
        boldItalic="ReportBoldItalic",
    )
    body = ParagraphStyle(
        "Body",
        fontName="Report",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#243340"),
        spaceAfter=8,
        alignment=TA_LEFT,
        splitLongWords=True,
        allowWidows=False,
        allowOrphans=False,
    )
    return {
        "body": body,
        "title": ParagraphStyle(
            "Title", parent=body, fontName="ReportBold", fontSize=20, leading=25, spaceAfter=12
        ),
        "section": ParagraphStyle(
            "Section",
            parent=body,
            fontName="ReportBold",
            fontSize=14,
            leading=19,
            spaceBefore=14,
            spaceAfter=9,
            keepWithNext=False,
            textColor=colors.HexColor("#245b80"),
        ),
        "heading": ParagraphStyle(
            "Heading",
            parent=body,
            fontName="ReportBold",
            fontSize=11,
            leading=15,
            spaceBefore=9,
            spaceAfter=6,
            keepWithNext=False,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=body, fontSize=9, leading=12, textColor=colors.HexColor("#536471")
        ),
        "cell": ParagraphStyle("Cell", parent=body, fontSize=9, leading=12, spaceAfter=0),
        "bullet": ParagraphStyle(
            "Bullet", parent=body, leftIndent=12, firstLineIndent=0, bulletIndent=0
        ),
    }


def report_pdf_flowables(document: dict, analyze_dir: Path) -> list:
    """Render the authored blocks without truncating rationale or reevaluating evidence."""
    if document.get("schema_version") != 2 or not document.get("sections"):
        raise ValueError("The PDF requires report schema 2 with authored sections; rerun analysis.")
    styles = report_pdf_styles()
    content_width = letter[0] - 88
    title = document.get("sample_id") or document["prefix"]
    story = [Paragraph(report_inline_html(str(title)), styles["title"])]
    for section in document["sections"]:
        story.extend(
            [CondPageBreak(72), Paragraph(report_inline_html(section["title"]), styles["section"])]
        )
        for block in section["blocks"]:
            kind = block["kind"]
            if kind in {"paragraph", "heading", "bullet"}:
                if kind == "heading":
                    story.append(CondPageBreak(60))
                style = styles["body"] if kind == "paragraph" else styles[kind]
                story.append(
                    Paragraph(
                        report_inline_html(block["text"]),
                        style,
                        bulletText="•" if kind == "bullet" else None,
                    )
                )
            elif kind == "table":
                rows = [block["headers"], *block["rows"]]
                cells = [
                    [Paragraph(report_inline_html(str(cell)), styles["cell"]) for cell in row]
                    for row in rows
                ]
                table = LongTable(
                    cells,
                    colWidths=[content_width / len(block["headers"])] * len(block["headers"]),
                    repeatRows=1,
                    hAlign="LEFT",
                    splitByRow=True,
                )
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf3f6")),
                            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#7e9baa")),
                            ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#dbe3e8")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 7),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                            ("TOPPADDING", (0, 0), (-1, -1), 7),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                        ]
                    )
                )
                story.extend([table, Spacer(1, 9)])
            elif kind == "figure":
                path = find_figure(analyze_dir, document["prefix"], block["suffix"])
                if path is None:
                    raise FileNotFoundError(
                        "A figure declared present in the report is missing: " + block["suffix"]
                    )
                figure = Image(str(path))
                scale = min(content_width / figure.imageWidth, 450 / figure.imageHeight)
                figure.drawWidth = figure.imageWidth * scale
                figure.drawHeight = figure.imageHeight * scale
                figure.hAlign = "CENTER"
                story.append(
                    KeepTogether(
                        [
                            Paragraph(report_inline_html(block["title"]), styles["heading"]),
                            Paragraph(report_inline_html(block["caption"]), styles["caption"]),
                            figure,
                            Spacer(1, 10),
                        ]
                    )
                )
            else:
                raise ValueError("Unknown report block kind: " + str(kind))
    return story


def build_interpretive_report_pdf(analyze_dir: Path, output: Path | None = None) -> Path:
    """Write a Letter-size PDF with native text, links and automatic pagination."""
    analyze_dir = Path(analyze_dir).resolve()
    document = load_report_document(analyze_dir)
    output = (
        Path(output)
        if output is not None
        else analyze_dir / f"{document['prefix']}-interpretive-report.pdf"
    )
    story = report_pdf_flowables(document, analyze_dir)
    pdf = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        leftMargin=44,
        rightMargin=44,
        topMargin=42,
        bottomMargin=44,
        title=str(document.get("sample_id") or document["prefix"]),
        author="trufflepig",
        subject="RNA evidence and therapeutic review",
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#dbe3e8"))
        canvas.line(44, 31, letter[0] - 44, 31)
        canvas.setFont("Report", 8)
        canvas.setFillColor(colors.HexColor("#536471"))
        canvas.drawString(44, 20, "trufflepig | RNA evidence and therapeutic review")
        canvas.drawRightString(letter[0] - 44, 20, str(doc.page))
        canvas.restoreState()

    pdf.build(story, onFirstPage=footer, onLaterPages=footer)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analyze_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(build_interpretive_report_pdf(args.analyze_dir, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
