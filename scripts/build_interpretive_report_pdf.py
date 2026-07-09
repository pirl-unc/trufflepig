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

# Reader-facing figure manifest. Each entry is (filename-suffix, title,
# interpretation sentence). The interpretation is what the figure *means* for the
# decision — it replaces the old practice of captioning a figure with its PNG
# filename. Figures are gated on existence at build time (``_find_figure`` returns
# None when the underlying analysis didn't emit the plot), so a belief that never
# fired never ships a figure. Ordering is decision-first: the call and its evidence,
# then composition/purity, then what to do (therapy), then sample context.
#
# Therapy figures are emitted under whichever name the run produced
# (``treatments`` for the ranked-therapy view, ``priority-targets`` for the
# score-decomposition view); both are listed and the missing one is skipped. The
# near-duplicate log-TPM dumbbell plots (``priority-target-context``,
# ``actionable-targets``) are intentionally NOT in the reader PDF — they live in
# the audit PDF (see module docstring); they restate the same ~18 genes the
# therapy figures already cover.
FIGURE_SPECS = [
    (
        "sample-summary.png",
        "Integrated sample summary",
        "One-page synthesis: the cancer-type call, tumor-vs-background composition, "
        "and purity for this sample.",
    ),
    (
        "cancer-type-signal-matrix.png",
        "Cancer-type evidence",
        "Which independent signals (expression signature, cohort centroid, "
        "decomposition, mismatch-repair) supported the reported cancer type, and how "
        "strongly each voted.",
    ),
    (
        "decomposition-composition.png",
        "Tumor / background composition",
        "Estimated fraction of the sample that is tumor versus each normal, immune, "
        "and stromal background component.",
    ),
    (
        "decomposition-candidates.png",
        "Decomposition hypotheses",
        "Competing tumor/background decomposition fits ranked by residual; the "
        "top-ranked hypothesis is the one adopted for purity and attribution.",
    ),
    (
        "purity-methods.png",
        "Purity method agreement",
        "Independent purity estimates (signature, lineage, ESTIMATE, decomposition "
        "residual) and their agreement — the reported purity is their fused consensus, "
        "widened when the methods disagree.",
    ),
    (
        "purity.png",
        "Purity estimate",
        "Reported tumor-purity point estimate with its uncertainty interval.",
    ),
    (
        "treatments.png",
        "Candidate therapies",
        "Therapies ranked for this call, each shown with the eligibility gate (the "
        "assay that actually confirms it) — RNA expression alone is not the criterion.",
    ),
    (
        "priority-targets.png",
        "Prioritized targets",
        "Why each actionable target ranks where it does: the score decomposition "
        "behind the therapy shortlist.",
    ),
    (
        "therapy-pathway-state.png",
        "Therapy pathway state",
        "Expression state of therapy-relevant pathways (antigen presentation, "
        "interferon, and others) that modulate the candidate treatments above.",
    ),
    (
        "sample-context.png",
        "Sample in cohort context",
        "Where this sample sits relative to the reference cohort in expression space.",
    ),
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


def _parse_markdown_tables(md_path: Path) -> list[dict]:
    """Extract every pipe-delimited markdown table from *md_path*, tagged with the
    ``##`` / ``###`` heading it sits under.

    The old scrape (:func:`_summary_records`) dropped every ``|`` row on the floor,
    so the therapy and target tables never reached the reader PDF. This recovers
    them structurally: each returned table is
    ``{"section", "subsection", "headers": [...], "rows": [[cell, ...], ...]}`` with
    the ``|---|`` separator row removed and markdown emphasis stripped from cells.
    """
    tables: list[dict] = []
    if not md_path.exists():
        return tables
    section = ""
    subsection = ""
    pending: list[list[str]] = []

    def _flush() -> None:
        nonlocal pending
        if len(pending) >= 2:
            headers = pending[0]
            body = pending[1:]
            # Drop the |---|---:| alignment separator row when present. Tolerate an
            # empty cell in it (``| --- | | --- |``): every non-empty cell must be
            # dashes/colons, and at least one cell must actually carry a dash.
            if (
                body
                and body[0]
                and all(set(cell) <= set("-: ") for cell in body[0])
                and any("-" in cell for cell in body[0])
            ):
                body = body[1:]
            rows = [row for row in body if any(cell.strip() for cell in row)]
            if headers and rows:
                tables.append(
                    {
                        "section": section,
                        "subsection": subsection,
                        "headers": headers,
                        "rows": rows,
                    }
                )
        pending = []

    for raw in md_path.read_text(errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("|"):
            pending.append([_clean_markdown(cell) for cell in stripped.strip("|").split("|")])
            continue
        _flush()
        if stripped.startswith("### "):
            subsection = _clean_markdown(stripped.lstrip("# "))
        elif stripped.startswith("## "):
            section = _clean_markdown(stripped.lstrip("# "))
            subsection = ""
    _flush()
    return tables


def _find_table(
    tables: list[dict],
    *,
    section_contains: str | None = None,
    header_all: tuple[str, ...] = (),
) -> dict | None:
    """First table whose heading contains *section_contains* and whose header row
    contains a column matching every key in *header_all* (case-insensitive)."""
    for table in tables:
        heading = f"{table['section']} {table['subsection']}".lower()
        if section_contains and section_contains.lower() not in heading:
            continue
        headers_low = [h.lower() for h in table["headers"]]
        if header_all and not all(any(key in h for h in headers_low) for key in header_all):
            continue
        return table
    return None


def _col_index(headers: list[str], *keys: str) -> int | None:
    """Index of the first header column containing any of *keys* (case-insensitive)."""
    low = [h.lower() for h in headers]
    for key in keys:
        for idx, header in enumerate(low):
            if key in header:
                return idx
    return None


def _cell(cells: list[str], idx: int | None) -> str:
    if idx is None or idx >= len(cells):
        return ""
    return cells[idx].strip()


def _lead_clause(text: str, *, max_chars: int = 72) -> str:
    """Leading clause of a verbose interpretation cell — the actionable head
    ("confirm MSI-H / dMMR status", "background-dominant", "tumor-supported") before
    the semicolon-joined tail of maturity/eligibility boilerplate."""
    # _clean_markdown has already folded em/en-dashes to "-", so " - " catches the
    # em-dash-delimited head; ";" catches the semicolon-joined form.
    text = _clean_markdown(text)
    for sep in (" - ", "; "):
        if sep in text:
            text = text.split(sep, 1)[0]
            break
    return _short_text(text, max_chars=max_chars)


def _tumor_from_attribution(text: str) -> str:
    """Tumor-source TPM out of an attribution cell like
    'tumor 847 / T cell 0 - broadly expr.' -> '847'."""
    match = re.search(r"tumor\s+([0-9.]+)", text, re.IGNORECASE)
    return match.group(1) if match else ""


def _safety_band(tme: str, attribution: str) -> str:
    """Normal-tissue safety read for a surface target: the TME flag
    (``tissue-explainable`` / ``background-dominant``) plus the broad-normal-
    expression cue. An empty TME flag means no single healthy tissue explains the
    signal, i.e. it is tumor-enriched."""
    band = tme.strip() or "tumor-enriched"
    lowered = attribution.lower()
    if "broadly expr" in lowered:
        band += " · broad normal expr."
    elif "matched-normal over-pred" in lowered:
        band += " · matched-normal overshoot"
    return _short_text(band, max_chars=58)


def _therapy_table(analyze_dir: Path, prefix: str) -> tuple[list[tuple[str, int]] | None, list[list[str]] | None]:
    """(columns, rows) for the therapy shortlist, or (None, None).

    Reads the ``## Therapy Prioritization`` table from the analysis report, dedups
    repeated targets (one target can list several agents), and keeps the top 3.
    """
    tables = _parse_markdown_tables(analyze_dir / f"{prefix}-analysis.md")
    table = _find_table(
        tables,
        section_contains="therapy prioritization",
        header_all=("target", "agent", "phase"),
    )
    if table is None:
        return None, None
    headers = table["headers"]
    i_target = _col_index(headers, "target", "gene")
    i_agent = _col_index(headers, "agent")
    i_phase = _col_index(headers, "phase")
    i_tumor = _col_index(headers, "tumor-source", "tumor-attributed", "tumor source")
    i_interp = _col_index(headers, "interpretation")
    rows: list[list[str]] = []
    seen: set[str] = set()
    for cells in table["rows"]:
        target = _cell(cells, i_target)
        if not target or target in seen:
            continue
        seen.add(target)
        agent = _cell(cells, i_agent)
        phase = _cell(cells, i_phase)
        agent_phase = agent + (f" · {phase}" if phase else "")
        rows.append(
            [
                target,
                agent_phase or "—",
                _cell(cells, i_tumor) or "—",
                _lead_clause(_cell(cells, i_interp)) or "—",
            ]
        )
        if len(rows) >= 3:
            break
    if not rows:
        return None, None
    columns = [("Target", 15), ("Agent / phase", 33), ("Tumor-src TPM", 20), ("Eligibility / RNA source", 44)]
    return columns, rows


def _priority_target_table(analyze_dir: Path, prefix: str) -> tuple[list[tuple[str, int]] | None, list[list[str]] | None]:
    """(columns, rows) for the top priority targets with tumor-source TPM and a
    normal-tissue safety band, or (None, None).

    Prefers the evidence report's ``### Surface Protein Targets`` table (ranked by
    expression, carries the TME safety flag and the tumor/non-tumor attribution
    split); falls back to the summary's tumor-source attribution table when the
    evidence report is absent.
    """
    ev_tables = _parse_markdown_tables(analyze_dir / f"{prefix}-evidence.md")
    table = _find_table(
        ev_tables,
        section_contains="surface protein targets",
        header_all=("gene", "tme"),
    )
    if table is not None:
        headers = table["headers"]
        i_gene = _col_index(headers, "gene", "target")
        i_bulk = _col_index(headers, "bulk tpm")
        i_attr = _col_index(headers, "attribution")
        i_tme = _col_index(headers, "tme")
        i_pct = _col_index(headers, "ref %ile", "%ile")
        rows: list[list[str]] = []
        for cells in table["rows"]:
            gene = _cell(cells, i_gene)
            if not gene:
                continue
            attribution = _cell(cells, i_attr)
            tumor_tpm = _tumor_from_attribution(attribution) or _cell(cells, i_bulk)
            rows.append(
                [
                    gene,
                    tumor_tpm or "—",
                    _safety_band(_cell(cells, i_tme), attribution),
                    _cell(cells, i_pct) or "—",
                ]
            )
            if len(rows) >= 3:
                break
        if rows:
            columns = [("Target", 20), ("Tumor-src TPM", 21), ("Normal-tissue safety", 43), ("Cohort %ile", 16)]
            return columns, rows

    su_tables = _parse_markdown_tables(analyze_dir / f"{prefix}-summary.md")
    table = _find_table(su_tables, header_all=("gene", "tumor-source"))
    if table is not None:
        headers = table["headers"]
        i_gene = _col_index(headers, "gene")
        i_tumor = _col_index(headers, "tumor-source")
        i_attr = _col_index(headers, "non-tumor attribution")
        i_frac = _col_index(headers, "tumor fraction")
        rows = []
        for cells in table["rows"]:
            gene = _cell(cells, i_gene)
            if not gene:
                continue
            rows.append(
                [
                    gene,
                    _cell(cells, i_tumor) or "—",
                    _cell(cells, i_attr) or "—",
                    _cell(cells, i_frac) or "—",
                ]
            )
            if len(rows) >= 3:
                break
        if rows:
            columns = [("Target", 20), ("Tumor-src TPM", 21), ("Top non-tumor source", 43), ("Tumor frac", 16)]
            return columns, rows
    return None, None


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

    # Therapy shortlist as a compact table (PR-1 §2.6a): the interpreted report
    # emits it as a markdown table the old scrape dropped. Render it directly so the
    # reader sees the agent, phase, and tumor-source TPM per target — not just prose.
    # Fall back to the interpreted therapy bullets when the call has no curated panel.
    therapy_cols, therapy_rows = _therapy_table(analyze_dir, title)
    y = _section_heading(draw, "Candidate therapies", y + 25)
    if therapy_cols and therapy_rows:
        y = _draw_table(
            draw,
            MARGIN,
            y,
            therapy_cols,
            therapy_rows,
            total_width=PAGE_W - 2 * MARGIN,
        )
    else:
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

    # Top priority targets with tumor-source TPM and a normal-tissue safety band.
    target_cols, target_rows = _priority_target_table(analyze_dir, title)
    if target_cols and target_rows and y < PAGE_H - 950:
        y = _section_heading(draw, "Priority surface targets", y + 22)
        y = _draw_table(
            draw,
            MARGIN,
            y,
            target_cols,
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
    prefix = _find_prefix(analyze_dir)
    summary_path = analyze_dir / f"{prefix}-summary.md"
    analysis_path = analyze_dir / f"{prefix}-analysis.md"
    output = output or analyze_dir / f"{prefix}-interpretive-report.pdf"

    highlights = _highlight_lines(summary_path, analysis_path)
    pages = [_title_page(prefix, highlights, analyze_dir)]
    for suffix, title, interpretation in FIGURE_SPECS:
        figure = _find_figure(analyze_dir, prefix, suffix)
        if figure is not None:
            pages.append(_figure_page(figure, title, interpretation))
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
