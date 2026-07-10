#!/usr/bin/env python3
"""Build a compact paper-style PDF from one trufflepig analyze directory.

The full ``all-figures.pdf`` is useful for exhaustive review, and
``figure-audit.pdf`` is useful for debugging redundancy. This script builds a
shorter reader-facing artifact: report highlights plus the plots that directly
support the interpreted text. Raw reference-context maps are intentionally not
included by default because they bypass the fused cancer-type evidence graph.

The PDF renders entirely from the structured report document
(``<prefix>-report.json``; see :mod:`trufflepig.report_document`) — the finalized
headline, the at-a-glance records, the therapy and target tables, and a
belief-gated figure manifest — and never re-reads the markdown, so it cannot
disagree with the figures or the reports. When the sidecar is absent (an analyze
dir produced before it existed, or a standalone build) the document is
reconstructed from the reports on the fly.
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from trufflepig.report_document import (
    clean_markdown,
    find_figure,
    load_report_document,
    record_value,
    section_records,
    short_text,
)

# The draw helpers below were written against the underscore-prefixed names these
# parsers used before they moved to the shared module; alias to keep them verbatim.
_clean_markdown = clean_markdown
_short_text = short_text

PAGE_W, PAGE_H = 2550, 3300
MARGIN = 150
TEXT_COLOR = (30, 41, 59)
MUTED = (100, 116, 139)
RULE = (203, 213, 225)
ACCENT = (37, 99, 235)
CARD_BG = (248, 250, 252)
SOFT_BLUE = (239, 246, 255)
SOFT_AMBER = (255, 251, 235)
SOFT_GREEN = (240, 253, 244)
SOFT_RED = (254, 242, 242)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    arial = "/System/Library/Fonts/Supplemental/Arial.ttf"
    arial_bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    dejavu = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    dejavu_bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    candidates = (
        arial_bold if bold else arial,
        dejavu_bold if bold else dejavu,
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _line_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])


def _draw_rich_wrapped(
    draw: ImageDraw.ImageDraw,
    segments: list[tuple[str, ImageFont.ImageFont, tuple[int, int, int]]],
    xy: tuple[int, int],
    *,
    max_width: int,
    line_gap: int = 11,
) -> int:
    """Draw wrapped text where individual segments can use bold/muted fonts."""

    x, y = xy
    line: list[tuple[str, ImageFont.ImageFont, tuple[int, int, int], int]] = []
    line_width = 0
    line_height = 0

    def flush() -> None:
        nonlocal y, line, line_width, line_height
        cx = x
        for text, font, fill, width in line:
            draw.text((cx, y), text, fill=fill, font=font)
            cx += width
        y += max(line_height, 1) + line_gap
        line = []
        line_width = 0
        line_height = 0

    for text, font, fill in segments:
        for token in re.findall(r"\S+\s*", text):
            width = _line_width(draw, token, font)
            height = getattr(font, "size", 30)
            if line and line_width + width > max_width:
                flush()
                token = token.lstrip()
                width = _line_width(draw, token, font)
            line.append((token, font, fill, width))
            line_width += width
            line_height = max(line_height, height)
    if line:
        flush()
    return y


def _therapy_summary(line: str) -> tuple[str, str]:
    text = _clean_markdown(line).lstrip("- ").strip()
    if " - " in text:
        gene, rest = text.split(" - ", 1)
    elif " — " in text:
        gene, rest = text.split(" — ", 1)
    else:
        return "", text
    sentences = re.split(r"(?<=\.)\s+", rest)
    headline = sentences[0] if sentences else rest
    caveats = []
    lowered = rest.lower()
    if "confirm msi" in lowered or "confirm msi-h" in lowered:
        caveats.append("confirm MSI/MMR before immunotherapy use")
    elif "confirm mutation" in lowered or "confirm biomarker" in lowered:
        caveats.append("confirm molecular eligibility")
    if "tumor-supported" in lowered:
        caveats.append("RNA supports tumor-source expression")
    elif "mixed-source" in lowered:
        caveats.append("RNA is mixed tumor/background")
    elif "context only" in lowered:
        caveats.append("target RNA is context only")
    suffix = "; ".join(caveats)
    body = headline.rstrip(".") + (f". {suffix}." if suffix else ".")
    return gene.strip(), _short_text(body, max_chars=210)


def _draw_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    label: str,
    value: str,
    accent: tuple[int, int, int],
    bg: tuple[int, int, int] = CARD_BG,
) -> None:
    x0, y0, x1, y1 = box
    label_font = _font(24, bold=True)
    value_font = _font(27)
    draw.rounded_rectangle(box, radius=18, fill=bg, outline=RULE, width=2)
    draw.rectangle((x0, y0, x0 + 12, y1), fill=accent)
    draw.text((x0 + 34, y0 + 26), label.upper(), fill=accent, font=label_font)
    max_width = x1 - x0 - 62
    words = re.findall(r"\S+\s*", _short_text(value or "Not reported", max_chars=120))
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = current + word
        if current and _line_width(draw, candidate, value_font) > max_width:
            lines.append(current.rstrip())
            current = word.lstrip()
        else:
            current = candidate
    if current:
        lines.append(current.rstrip())
    max_lines = 4
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _short_text(lines[-1], max_chars=max(12, len(lines[-1]) - 3))
    y = y0 + 72
    for line in lines:
        draw.text((x0 + 34, y), line, fill=TEXT_COLOR, font=value_font)
        y += 36


def _section_heading(draw: ImageDraw.ImageDraw, text: str, y: int) -> int:
    section_font = _font(36, bold=True)
    draw.text((MARGIN, y), text, fill=TEXT_COLOR, font=section_font)
    draw.rectangle((MARGIN, y + 48, PAGE_W - MARGIN, y + 51), fill=RULE)
    return y + 80


def _draw_labeled_bullet(
    draw: ImageDraw.ImageDraw,
    *,
    label: str,
    body: str,
    y: int,
    max_width: int,
    fill: tuple[int, int, int] = TEXT_COLOR,
) -> int:
    body_font = _font(28)
    bold_font = _font(28, bold=True)
    draw.ellipse((MARGIN, y + 12, MARGIN + 13, y + 25), fill=ACCENT)
    segments = []
    if label:
        segments.append((f"{label}: ", bold_font, TEXT_COLOR))
    segments.append((_short_text(body, max_chars=360), body_font, fill))
    return (
        _draw_rich_wrapped(
            draw,
            segments,
            (MARGIN + 34, y),
            max_width=max_width,
            line_gap=10,
        )
        + 7
    )


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    *,
    width_chars: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = TEXT_COLOR,
    line_gap: int = 12,
) -> int:
    x, y = xy
    for para in text.splitlines():
        wrapped = textwrap.wrap(para, width=width_chars) or [""]
        for line in wrapped:
            draw.text((x, y), line, fill=fill, font=font)
            y += font.size + line_gap if hasattr(font, "size") else 30
    return y


def _wrap_cell(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_px: int,
    *,
    max_lines: int = 2,
) -> list[str]:
    """Wrap *text* into at most *max_lines* lines fitting *max_px*, ellipsizing the
    last line when it still overflows."""
    words = re.findall(r"\S+\s*", text)
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = current + word
        if current and _line_width(draw, candidate, font) > max_px:
            lines.append(current.rstrip())
            current = word.lstrip()
            if len(lines) >= max_lines:
                break
        else:
            current = candidate
    if current and len(lines) < max_lines:
        lines.append(current.rstrip())
    if not lines:
        return [""]
    # Ellipsize the last line whenever it still overflows — including a single
    # unbreakable token that never got a chance to wrap — so a cell never bleeds
    # into its neighbour column.
    if _line_width(draw, lines[-1], font) > max_px:
        while lines[-1] and _line_width(draw, lines[-1] + "...", font) > max_px:
            lines[-1] = lines[-1].rsplit(" ", 1)[0] if " " in lines[-1] else lines[-1][:-1]
        lines[-1] = lines[-1].rstrip() + "..."
    return lines


def _draw_table(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    columns: list[tuple[str, int]],
    rows: list[list[str]],
    *,
    total_width: int,
    max_cell_lines: int = 2,
) -> int:
    """Render a compact table: a bold accent header row over a rule, then wrapped
    cells with a faint separator between rows. *columns* is ``(title, weight)``;
    weights set relative column widths. Returns the y below the table."""
    header_font = _font(23, bold=True)
    cell_font = _font(23)
    total_weight = sum(weight for _, weight in columns) or 1
    col_x: list[int] = []
    col_w: list[int] = []
    cursor = x
    for _, weight in columns:
        width = int(total_width * weight / total_weight)
        col_x.append(cursor)
        col_w.append(width)
        cursor += width
    for (title, _), cxi in zip(columns, col_x):
        draw.text((cxi, y), title, fill=ACCENT, font=header_font)
    y += 36
    draw.rectangle((x, y, x + total_width, y + 2), fill=RULE)
    y += 14
    for row in rows:
        wrapped = [
            _wrap_cell(draw, str(cell), cell_font, width - 24, max_lines=max_cell_lines)
            for cell, width in zip(row, col_w)
        ]
        row_lines = max((len(lines) for lines in wrapped), default=1)
        for lines, cxi in zip(wrapped, col_x):
            ly = y
            for line in lines:
                draw.text((cxi, ly), line, fill=TEXT_COLOR, font=cell_font)
                ly += 30
        y += row_lines * 30 + 16
        draw.rectangle((x, y - 9, x + total_width, y - 8), fill=RULE)
    return y + 6


def _title_page(document: dict, analyze_dir: Path) -> Image.Image:
    records = document.get("records") or []
    title = document.get("prefix") or ""
    img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(img)
    title_font = _font(62, bold=True)
    subtitle_font = _font(27)
    section_font = _font(34, bold=True)
    small_font = _font(24)

    draw.text((MARGIN, MARGIN), title, fill=TEXT_COLOR, font=title_font)
    draw.text(
        (MARGIN, MARGIN + 82),
        "Interpretive snapshot - call, evidence, and downstream implications",
        fill=MUTED,
        font=subtitle_font,
    )
    draw.rectangle((MARGIN, MARGIN + 130, PAGE_W - MARGIN, MARGIN + 136), fill=ACCENT)

    call = record_value(records, "Cancer call")
    mmr = record_value(records, "Mismatch-repair RNA context")
    purity = record_value(records, "Purity")
    sample = record_value(records, "Sample")
    quant = record_value(records, "RNA quant QC")
    sample_card = sample
    if quant:
        sample_card = f"{sample}; {quant}" if sample else quant

    card_y = MARGIN + 185
    gap = 24
    card_w = (PAGE_W - 2 * MARGIN - 3 * gap) // 4
    card_h = 235
    cards = [
        ("Cancer call", call, ACCENT, SOFT_BLUE),
        ("MMR RNA", mmr, (124, 58, 237), (245, 243, 255)),
        ("Purity", purity, (22, 163, 74), SOFT_GREEN),
        ("Assay / QC", sample_card, (217, 119, 6), SOFT_AMBER),
    ]
    for idx, (label, value, accent, bg) in enumerate(cards):
        x0 = MARGIN + idx * (card_w + gap)
        _draw_card(
            draw,
            (x0, card_y, x0 + card_w, card_y + card_h),
            label=label,
            value=value,
            accent=accent,
            bg=bg,
        )

    y = card_y + card_h + 62
    draw.text((MARGIN, y), "What matters first", fill=TEXT_COLOR, font=section_font)
    draw.rectangle((MARGIN, y + 46, PAGE_W - MARGIN, y + 49), fill=RULE)
    y += 78

    interpretation_items = [
        ("Call basis", record_value(records, "Cancer-type basis")),
        ("MMR RNA", record_value(records, "Mismatch-repair RNA context")),
        ("Competing RNA context", record_value(records, "Retained RNA differential")),
        ("Composition caution", record_value(records, "Tissue composition hint")),
        ("Rare-marker prompt", record_value(records, "Rare-marker prompt")),
        ("Disease state", record_value(records, "Disease state")),
    ]
    for label, body in interpretation_items:
        if not body:
            continue
        y = _draw_labeled_bullet(
            draw,
            label=label,
            body=body,
            y=y,
            max_width=PAGE_W - 2 * MARGIN - 40,
        )
        if y > 1500:
            break

    # Therapy shortlist as a compact table (the structured document carries it as
    # {columns, rows}); fall back to the interpreted therapy bullets when the call
    # has no curated panel.
    therapy = document.get("therapy") or {}
    therapy_rows = therapy.get("rows")
    y = _section_heading(draw, "Candidate therapies", y + 25)
    if therapy_rows:
        y = _draw_table(
            draw,
            MARGIN,
            y,
            therapy["columns"],
            therapy_rows,
            total_width=PAGE_W - 2 * MARGIN,
        )
    else:
        therapy_records = [
            record
            for record in section_records(records, "Top candidate therapies")
            if re.match(r"^[A-Za-z0-9.-]+ - ", record["text"])
        ][:4]
        if therapy_records:
            for record in therapy_records[:4]:
                gene, body = _therapy_summary(record["text"])
                y = _draw_labeled_bullet(
                    draw,
                    label=gene,
                    body=body,
                    y=y,
                    max_width=PAGE_W - 2 * MARGIN - 40,
                )
        else:
            for line in (document.get("highlights") or [])[:3]:
                y = _draw_labeled_bullet(
                    draw,
                    label="Therapy",
                    body=line,
                    y=y,
                    max_width=PAGE_W - 2 * MARGIN - 40,
                )

    # Top priority targets with tumor-source TPM and a normal-tissue safety band.
    targets = document.get("targets") or {}
    target_rows = targets.get("rows")
    if target_rows and y < PAGE_H - 950:
        y = _section_heading(draw, "Priority surface targets", y + 22)
        y = _draw_table(
            draw,
            MARGIN,
            y,
            targets["columns"],
            target_rows,
            total_width=PAGE_W - 2 * MARGIN,
        )

    notable = [
        record["text"]
        for record in records
        if record["section"] in {"Notable biomarker outliers", "Notable CTAs"}
        and record["text"]
        and record["text"][0].isalnum()
    ][:4]
    if notable:
        y = _section_heading(draw, "Notable expression outliers", y + 22)
        for line in notable:
            label, body = ("", line)
            if " - " in line:
                label, body = line.split(" - ", 1)
            y = _draw_labeled_bullet(
                draw,
                label=label,
                body=body,
                y=y,
                max_width=PAGE_W - 2 * MARGIN - 40,
                fill=TEXT_COLOR,
            )
            if y > PAGE_H - 560:
                break

    caveats = [
        record["text"]
        for record in section_records(records, "Caveats")
        if record["text"]
    ][:3]
    if caveats and y < PAGE_H - 680:
        y = _section_heading(draw, "Confirm before acting", y + 18)
        for caveat in caveats:
            y = _draw_labeled_bullet(
                draw,
                label="Caveat",
                body=caveat,
                y=y,
                max_width=PAGE_W - 2 * MARGIN - 40,
                fill=MUTED,
            )
            if y > PAGE_H - 420:
                break

    draw.rectangle((MARGIN, PAGE_H - 300, PAGE_W - MARGIN, PAGE_H - 298), fill=RULE)
    _draw_wrapped(
        draw,
        "This compact PDF is generated from the interpreted report and decision-aligned figures. "
        "Audit-only raw reference maps are intentionally omitted unless added explicitly.",
        (MARGIN, PAGE_H - 260),
        width_chars=110,
        font=small_font,
        fill=MUTED,
    )
    _draw_wrapped(
        draw,
        f"Source analyze directory: {analyze_dir}",
        (MARGIN, PAGE_H - 150),
        width_chars=128,
        font=small_font,
        fill=MUTED,
    )
    return img


def _figure_page(path: Path, title: str, interpretation: str = "") -> Image.Image:
    """One figure per page: a bold title and its *interpretation sentence* — what the
    figure means for the decision — instead of the raw PNG filename."""
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)
    title_font = _font(38, bold=True)
    caption_font = _font(26)
    draw.text((MARGIN, MARGIN), title, fill=TEXT_COLOR, font=title_font)

    # Wrap the interpretation sentence to the page width beneath the title.
    caption_lines = textwrap.wrap(interpretation, width=118) if interpretation else []
    y_caption = MARGIN + 58
    for line in caption_lines:
        draw.text((MARGIN, y_caption), line, fill=MUTED, font=caption_font)
        y_caption += 36
    rule_y = max(MARGIN + 95, y_caption + 10)
    draw.rectangle((MARGIN, rule_y, PAGE_W - MARGIN, rule_y + 3), fill=RULE)

    fig_top = rule_y + 50
    with Image.open(path) as src:
        fig = src.convert("RGB")
    box_w = PAGE_W - 2 * MARGIN
    box_h = PAGE_H - MARGIN - fig_top
    scale = min(box_w / fig.width, box_h / fig.height)
    new_size = (max(1, int(fig.width * scale)), max(1, int(fig.height * scale)))
    fig = fig.resize(new_size, Image.Resampling.LANCZOS)
    x = MARGIN + (box_w - fig.width) // 2
    y = fig_top + (box_h - fig.height) // 2
    page.paste(fig, (x, y))
    return page


def build_interpretive_report_pdf(analyze_dir: Path, output: Path | None = None) -> Path:
    analyze_dir = analyze_dir.resolve()
    document = load_report_document(analyze_dir)
    prefix = document["prefix"]
    output = output or analyze_dir / f"{prefix}-interpretive-report.pdf"

    pages = [_title_page(document, analyze_dir)]
    # Figure manifest is belief-gated at the source (a plot is only emitted when
    # its underlying belief passed threshold), so ``present`` guarantees the PDF
    # never ships a figure the report text denies.
    for figure_spec in document.get("figures") or []:
        if not figure_spec.get("present"):
            continue
        figure = find_figure(analyze_dir, prefix, figure_spec["suffix"])
        if figure is not None:
            pages.append(
                _figure_page(figure, figure_spec["title"], figure_spec.get("caption") or "")
            )
    pages[0].save(output, save_all=True, append_images=pages[1:])
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analyze_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = build_interpretive_report_pdf(args.analyze_dir, args.output)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
