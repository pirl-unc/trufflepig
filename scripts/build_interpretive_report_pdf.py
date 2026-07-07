#!/usr/bin/env python3
"""Build a compact paper-style PDF from one trufflepig analyze directory.

The full ``all-figures.pdf`` is useful for exhaustive review, and
``figure-audit.pdf`` is useful for debugging redundancy. This script builds a
shorter reader-facing artifact: report highlights plus the plots that directly
support the interpreted text. Raw reference-context maps are intentionally not
included by default because they bypass the fused cancer-type evidence graph.
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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

FIGURE_SPECS = [
    ("sample-summary.png", "Integrated sample summary"),
    ("cancer-type-signal-matrix.png", "Decision-aligned cancer-type evidence"),
    ("decomposition-composition.png", "Tumor/background composition"),
    ("decomposition-candidates.png", "Top decomposition hypotheses"),
    ("purity-methods.png", "Purity method agreement"),
    ("priority-targets.png", "Prioritized actionable targets"),
    ("priority-target-context.png", "Target evidence context"),
    ("therapy-pathway-state.png", "Therapy pathway state"),
    ("actionable-targets.png", "Actionable target screen"),
]


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


def _clean_markdown(line: str) -> str:
    line = re.sub(r"`([^`]+)`", r"\1", line)
    line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    line = line.replace("\u2014", "-").replace("\u2013", "-")
    return line.strip()


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


def _split_label_value(line: str) -> tuple[str, str]:
    text = line.lstrip("- ").strip()
    markdown = re.match(r"\*\*([^*:]+):\*\*\s*(.*)", text)
    if markdown:
        return _clean_markdown(markdown.group(1)), _clean_markdown(markdown.group(2))
    bold_prefix = re.match(r"\*\*([^*]+)\*\*\s*(.*)", text)
    if bold_prefix and ":" in bold_prefix.group(1):
        label, value = bold_prefix.group(1).split(":", 1)
        combined = (value.strip() + " " + bold_prefix.group(2).strip()).strip()
        return _clean_markdown(label), _clean_markdown(combined)
    clean = _clean_markdown(text)
    if ":" in clean:
        label, value = clean.split(":", 1)
        return label.strip(), value.strip()
    return "", clean


def _summary_records(summary_path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    section = ""
    subsection = ""
    if not summary_path.exists():
        return records
    for raw in summary_path.read_text(errors="replace").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("### "):
            subsection = _clean_markdown(stripped.lstrip("# "))
            continue
        if stripped.startswith("## "):
            section = _clean_markdown(stripped.lstrip("# "))
            subsection = ""
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("|") or set(_clean_markdown(stripped)) <= {"-", " "}:
            continue
        label, value = _split_label_value(stripped)
        if stripped.startswith("- "):
            clean = _clean_markdown(stripped).lstrip("- ").strip()
            if not label:
                label, value = "", clean
        elif not label:
            value = _clean_markdown(stripped)
        records.append(
            {
                "section": section,
                "subsection": subsection,
                "label": label,
                "value": value,
                "text": _clean_markdown(stripped).lstrip("- ").strip(),
            }
        )
    return records


def _record_value(records: list[dict[str, str]], label: str) -> str:
    for record in records:
        if record["label"].lower() == label.lower():
            return record["value"]
    return ""


def _record_text(records: list[dict[str, str]], label: str) -> str:
    for record in records:
        if record["label"].lower() == label.lower():
            return record["text"]
    return ""


def _section_records(records: list[dict[str, str]], section: str) -> list[dict[str, str]]:
    return [record for record in records if record["section"].lower() == section.lower()]


def _short_text(text: str, *, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rsplit(" ", 1)[0] + "..."


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


def _highlight_lines(
    summary_path: Path,
    analysis_path: Path | None,
    *,
    max_lines: int = 28,
) -> list[str]:
    lines: list[str] = []
    if summary_path.exists():
        for raw in summary_path.read_text(errors="replace").splitlines():
            line = _clean_markdown(raw)
            if not line or line.startswith("|") or set(line) <= {"-", " "}:
                continue
            if line.startswith("#"):
                continue
            lines.append(line.lstrip("- ").strip())
            if len(lines) >= max_lines:
                break
    if analysis_path and analysis_path.exists():
        analysis_lines = analysis_path.read_text(errors="replace").splitlines()
        wanted_prefixes = (
            "- **Mismatch-repair RNA context**:",
            "- **Integrated evidence**:",
            "- **Selected report label**:",
        )
        for raw in analysis_lines:
            if any(raw.startswith(prefix) for prefix in wanted_prefixes):
                line = _clean_markdown(raw).lstrip("- ").strip()
                if line and line not in lines:
                    lines.append(line)
            if len(lines) >= max_lines + 6:
                break
    return lines[: max_lines + 6]


def _find_prefix(analyze_dir: Path) -> str:
    summaries = sorted(
        path
        for path in analyze_dir.glob("*-summary.md")
        if not path.name.endswith("-cancer-type-signal-summary.md")
    )
    if summaries:
        latest = max(summaries, key=lambda path: path.stat().st_mtime)
        return latest.name.removesuffix("-summary.md")
    analyses = sorted(analyze_dir.glob("*-analysis.md"))
    if analyses:
        latest = max(analyses, key=lambda path: path.stat().st_mtime)
        return latest.name.removesuffix("-analysis.md")
    raise FileNotFoundError(f"No *-summary.md or *-analysis.md found in {analyze_dir}")


def _find_figure(analyze_dir: Path, prefix: str, suffix: str) -> Path | None:
    name = f"{prefix}-{suffix}"
    candidates = [
        analyze_dir / name,
        analyze_dir / "figures" / name,
    ]
    candidates.extend(analyze_dir.glob(f"**/{name}"))
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


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


def _title_page(title: str, highlights: list[str], analyze_dir: Path) -> Image.Image:
    records = _summary_records(analyze_dir / f"{title}-summary.md")
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

    call = _record_value(records, "Cancer call")
    mmr = _record_value(records, "Mismatch-repair RNA context")
    purity = _record_value(records, "Purity")
    sample = _record_value(records, "Sample")
    quant = _record_value(records, "RNA quant QC")
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
        ("Call basis", _record_value(records, "Cancer-type basis")),
        ("MMR RNA", _record_value(records, "Mismatch-repair RNA context")),
        ("Competing RNA context", _record_value(records, "Retained RNA differential")),
        ("Composition caution", _record_value(records, "Tissue composition hint")),
        ("Rare-marker prompt", _record_value(records, "Rare-marker prompt")),
        ("Disease state", _record_value(records, "Disease state")),
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

    y = _section_heading(draw, "Therapy implications", y + 25)
    therapy_records = [
        record
        for record in _section_records(records, "Top candidate therapies")
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
        for line in highlights[:3]:
            y = _draw_labeled_bullet(
                draw,
                label="Therapy",
                body=line,
                y=y,
                max_width=PAGE_W - 2 * MARGIN - 40,
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
        for record in _section_records(records, "Caveats")
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


def _figure_page(path: Path, caption: str) -> Image.Image:
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)
    caption_font = _font(38, bold=True)
    small_font = _font(23)
    draw.text((MARGIN, MARGIN), caption, fill=TEXT_COLOR, font=caption_font)
    draw.text((MARGIN, MARGIN + 55), path.name, fill=MUTED, font=small_font)
    draw.rectangle((MARGIN, MARGIN + 95, PAGE_W - MARGIN, MARGIN + 98), fill=RULE)

    with Image.open(path) as src:
        fig = src.convert("RGB")
    box_w = PAGE_W - 2 * MARGIN
    box_h = PAGE_H - MARGIN - (MARGIN + 145)
    scale = min(box_w / fig.width, box_h / fig.height)
    new_size = (max(1, int(fig.width * scale)), max(1, int(fig.height * scale)))
    fig = fig.resize(new_size, Image.Resampling.LANCZOS)
    x = MARGIN + (box_w - fig.width) // 2
    y = MARGIN + 145 + (box_h - fig.height) // 2
    page.paste(fig, (x, y))
    return page


def build_interpretive_report_pdf(analyze_dir: Path, output: Path | None = None) -> Path:
    analyze_dir = analyze_dir.resolve()
    prefix = _find_prefix(analyze_dir)
    summary_path = analyze_dir / f"{prefix}-summary.md"
    analysis_path = analyze_dir / f"{prefix}-analysis.md"
    output = output or analyze_dir / f"{prefix}-interpretive-report.pdf"

    highlights = _highlight_lines(summary_path, analysis_path)
    pages = [_title_page(prefix, highlights, analyze_dir)]
    for suffix, caption in FIGURE_SPECS:
        figure = _find_figure(analyze_dir, prefix, suffix)
        if figure is not None:
            pages.append(_figure_page(figure, caption))
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
