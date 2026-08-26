# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The structured, serializable report document the interpretive PDF renders from.

`docs/report-belief-consistency-and-friendliness-plan.md` §2.6b: render the
decision, don't scrape the markdown. The analyze pipeline writes
``<prefix>-report.json`` at the finalize barrier (see ``main._analyze_body``); the
interpretive-report PDF (``scripts/build_interpretive_report_pdf.py``) reads ONLY
that document and never re-reads the markdown.

The document carries, per sample:

- ``headline`` — the authoritative finalized call + purity, taken directly from the
  frozen :class:`~trufflepig.report_view.ReportView` (the same object the figures
  and markdown headline read), so the PDF's headline can't drift from them.
- ``records`` / ``highlights`` — the summary's at-a-glance label:value lines and
  bullet highlights.
- ``therapy`` / ``targets`` — the therapy shortlist and priority-target tables as
  ``{"columns": [[name, width], ...], "rows": [[cell, ...], ...]}``.
- ``figures`` — a manifest of every reader figure with its interpretation caption
  and a ``present`` flag. A figure is only emitted when its underlying belief
  passed threshold, so ``present`` is belief-gated: the PDF ships a figure iff the
  decision actually produced it, and can never show a figure the text denies.

The ``records``/``therapy``/``targets`` are parsed once, server-side, from the
pipeline's own just-emitted markdown, so the markdown stays byte-stable and the
document is a faithful projection of it (parity by construction). This module is
deliberately dependency-light (stdlib only) so the PDF script can import it
without pulling in matplotlib/pandas.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .report_view import ReportView

SCHEMA_VERSION = 1

# Reader-facing figure manifest: (filename-suffix, title, interpretation sentence).
# The interpretation is what the figure *means* for the decision — it replaces
# captioning a figure with its PNG filename. Figures are gated on existence at
# emit time (a belief that never fired never wrote its plot), so a manifest entry's
# ``present`` flag is belief-gated. Ordering is decision-first: the call and its
# evidence, then composition/purity, then what to do (therapy), then sample context.
#
# Therapy figures are emitted under whichever name the run produced (``treatments``
# for the ranked-therapy view, ``priority-targets`` for the score-decomposition
# view); both are listed and the missing one is skipped. The near-duplicate log-TPM
# dumbbell plots (``priority-target-context``, ``actionable-targets``) are
# intentionally NOT in the reader manifest — they live in the audit PDF and restate
# the same genes the therapy figures already cover.
FIGURE_REGISTRY = [
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
        "Competing tumor/background decomposition fits; the selected model is "
        "listed first and the remaining rows are comparisons.",
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


# --------------------------------------------------------------------------- #
# Pure markdown parsers (relocated from the PDF script so both the pipeline and
# the PDF share one implementation).
# --------------------------------------------------------------------------- #
def clean_markdown(line: str) -> str:
    line = re.sub(r"`([^`]+)`", r"\1", line)
    line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    line = line.replace("—", "-").replace("–", "-")
    return line.strip()


def short_text(text: str, *, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rsplit(" ", 1)[0] + "..."


def split_label_value(line: str) -> tuple[str, str]:
    text = line.lstrip("- ").strip()
    markdown = re.match(r"\*\*([^*:]+):\*\*\s*(.*)", text)
    if markdown:
        return clean_markdown(markdown.group(1)), clean_markdown(markdown.group(2))
    bold_prefix = re.match(r"\*\*([^*]+)\*\*\s*(.*)", text)
    if bold_prefix and ":" in bold_prefix.group(1):
        label, value = bold_prefix.group(1).split(":", 1)
        combined = (value.strip() + " " + bold_prefix.group(2).strip()).strip()
        return clean_markdown(label), clean_markdown(combined)
    clean = clean_markdown(text)
    if ":" in clean:
        label, value = clean.split(":", 1)
        return label.strip(), value.strip()
    return "", clean


def parse_summary_records(summary_path: Path) -> List[dict]:
    records: List[dict] = []
    section = ""
    subsection = ""
    if not summary_path.exists():
        return records
    for raw in summary_path.read_text(errors="replace").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("### "):
            subsection = clean_markdown(stripped.lstrip("# "))
            continue
        if stripped.startswith("## "):
            section = clean_markdown(stripped.lstrip("# "))
            subsection = ""
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("|") or set(clean_markdown(stripped)) <= {"-", " "}:
            continue
        label, value = split_label_value(stripped)
        if stripped.startswith("- "):
            clean = clean_markdown(stripped).lstrip("- ").strip()
            if not label:
                label, value = "", clean
        elif not label:
            value = clean_markdown(stripped)
        records.append(
            {
                "section": section,
                "subsection": subsection,
                "label": label,
                "value": value,
                "text": clean_markdown(stripped).lstrip("- ").strip(),
            }
        )
    return records


def record_value(records: List[dict], label: str) -> str:
    for record in records:
        if record["label"].lower() == label.lower():
            return record["value"]
    return ""


def section_records(records: List[dict], section: str) -> List[dict]:
    return [record for record in records if record["section"].lower() == section.lower()]


def parse_markdown_tables(md_path: Path) -> List[dict]:
    """Extract every pipe-delimited markdown table from *md_path*, tagged with the
    ``##`` / ``###`` heading it sits under. Each returned table is
    ``{"section", "subsection", "headers": [...], "rows": [[cell, ...], ...]}`` with
    the ``|---|`` separator row removed and markdown emphasis stripped from cells.
    """
    tables: List[dict] = []
    if not md_path.exists():
        return tables
    section = ""
    subsection = ""
    pending: List[List[str]] = []

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
            pending.append([clean_markdown(cell) for cell in stripped.strip("|").split("|")])
            continue
        _flush()
        if stripped.startswith("### "):
            subsection = clean_markdown(stripped.lstrip("# "))
        elif stripped.startswith("## "):
            section = clean_markdown(stripped.lstrip("# "))
            subsection = ""
    _flush()
    return tables


def find_table(
    tables: List[dict],
    *,
    section_contains: Optional[str] = None,
    header_all: tuple = (),
    subsection_excludes: Optional[str] = None,
) -> Optional[dict]:
    """First table whose heading contains *section_contains* and whose header row
    contains a column matching every key in *header_all* (case-insensitive).

    When *subsection_excludes* is set, tables whose ``###`` subsection contains it
    are skipped — used to keep the PDF's therapy shortlist on the active rows and
    off the "not supported by this sample" subsection (issue #105)."""
    for table in tables:
        heading = f"{table['section']} {table['subsection']}".lower()
        if section_contains and section_contains.lower() not in heading:
            continue
        if (
            subsection_excludes
            and subsection_excludes.lower() in table["subsection"].lower()
        ):
            continue
        headers_low = [h.lower() for h in table["headers"]]
        if header_all and not all(any(key in h for h in headers_low) for key in header_all):
            continue
        return table
    return None


def col_index(headers: List[str], *keys: str) -> Optional[int]:
    """Index of the first header column containing any of *keys* (case-insensitive)."""
    low = [h.lower() for h in headers]
    for key in keys:
        for idx, header in enumerate(low):
            if key in header:
                return idx
    return None


def cell(cells: List[str], idx: Optional[int]) -> str:
    if idx is None or idx >= len(cells):
        return ""
    return cells[idx].strip()


def lead_clause(text: str, *, max_chars: int = 72) -> str:
    """Leading clause of a verbose interpretation cell — the actionable head before
    the semicolon-joined tail of maturity/eligibility boilerplate."""
    text = clean_markdown(text)
    for sep in (" - ", "; "):
        if sep in text:
            text = text.split(sep, 1)[0]
            break
    return short_text(text, max_chars=max_chars)


def tumor_from_attribution(text: str) -> str:
    """Tumor-source TPM out of an attribution cell like
    'tumor 847 / T cell 0 - broadly expr.' -> '847'."""
    match = re.search(r"tumor\s+([0-9.]+)", text, re.IGNORECASE)
    return match.group(1) if match else ""


def safety_band(tme: str, attribution: str) -> str:
    """Normal-tissue safety read for a surface target: the TME flag plus the
    broad-normal-expression cue. An empty TME flag means no single healthy tissue
    explains the signal, i.e. it is tumor-enriched."""
    band = tme.strip() or "tumor-enriched"
    lowered = attribution.lower()
    if "broadly expr" in lowered:
        band += " · broad normal expr."
    elif "matched-normal over-pred" in lowered:
        band += " · matched-normal overshoot"
    return short_text(band, max_chars=58)


def parse_therapy_recommendations(summary_path: Path) -> Optional[dict]:
    """Return the exact reader-facing therapy shortlist from a summary.

    The detailed analysis intentionally includes a broader therapy landscape,
    including conditional and comparator rows that did not qualify for the
    summary. The structured report represents the reader decision, so it
    serializes the bullets under ``Top candidate therapies`` instead of making
    a second selection from that broader table.
    """
    summary_path = Path(summary_path)
    if not summary_path.exists():
        return None

    in_therapy_section = False
    rows: List[List[str]] = []
    bullet = re.compile(r"^- \*\*([^*]+)\*\*\s+[—-]\s+(.+)$")
    recommendation = re.compile(
        r"^(.*?)\s+\((Approved|Phase\s+\d(?:/\d)?|"
        r"Off-label\s*/\s*transfer rationale)(?:,\s*[^)]*)?\)\.\s*(.*)$",
        re.IGNORECASE,
    )
    for raw in summary_path.read_text(errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("## "):
            heading = clean_markdown(stripped.lstrip("# ")).lower()
            if in_therapy_section and heading != "top candidate therapies":
                break
            in_therapy_section = heading == "top candidate therapies"
            continue
        if not in_therapy_section:
            continue
        match = bullet.match(stripped)
        if not match:
            continue

        target = clean_markdown(match.group(1))
        body = match.group(2).strip()
        parsed = recommendation.match(body)
        if parsed:
            agent = clean_markdown(parsed.group(1))
            phase = clean_markdown(parsed.group(2))
            interpretation = clean_markdown(parsed.group(3))
        else:
            agent = lead_clause(body, max_chars=80)
            phase = ""
            interpretation = body

        tumor_match = re.search(
            r"([0-9.]+)\s+tumor-source bulk TPM\s+"
            r"\(model interval\s+([^)]+)\)",
            interpretation,
            re.IGNORECASE,
        )
        context_match = re.search(
            r"target RNA is context only\s+\(bulk\s+([0-9.]+)\s+TPM",
            interpretation,
            re.IGNORECASE,
        )
        if tumor_match:
            tumor = f"{tumor_match.group(1)} ({tumor_match.group(2)})"
        elif context_match:
            tumor = f"{context_match.group(1)} bulk; context only"
        else:
            tumor = "—"
        rows.append(
            [
                target,
                " · ".join(part for part in (agent, phase) if part),
                tumor,
                lead_clause(interpretation),
            ]
        )
        if len(rows) >= 3:
            break

    if not rows:
        return None
    return {
        "columns": [
            ["Target", 15],
            ["Recommendation", 38],
            ["Tumor-src TPM", 20],
            ["Eligibility / RNA source", 44],
        ],
        "rows": rows,
    }


def _priority_target_table(analyze_dir: Path, prefix: str) -> Optional[dict]:
    """``{"columns", "rows"}`` for the top priority targets with tumor-source TPM
    and a normal-tissue safety band, or ``None``.

    Prefers the evidence report's ``### Surface Protein Targets`` table; falls back
    to the summary's tumor-source attribution table when it is absent.
    """
    ev_tables = parse_markdown_tables(analyze_dir / f"{prefix}-evidence.md")
    table = find_table(
        ev_tables,
        section_contains="surface protein targets",
        header_all=("gene", "tme"),
    )
    if table is not None:
        headers = table["headers"]
        i_gene = col_index(headers, "gene", "target")
        i_bulk = col_index(headers, "bulk tpm")
        i_attr = col_index(headers, "attribution")
        i_tme = col_index(headers, "tme")
        i_pct = col_index(headers, "ref %ile", "%ile")
        rows: List[List[str]] = []
        for cells in table["rows"]:
            gene = cell(cells, i_gene)
            if not gene:
                continue
            attribution = cell(cells, i_attr)
            tumor_tpm = tumor_from_attribution(attribution) or cell(cells, i_bulk)
            rows.append(
                [
                    gene,
                    tumor_tpm or "—",
                    safety_band(cell(cells, i_tme), attribution),
                    cell(cells, i_pct) or "—",
                ]
            )
            if len(rows) >= 3:
                break
        if rows:
            columns = [
                ["Target", 20],
                ["Tumor-src TPM", 21],
                ["Normal-tissue safety", 43],
                ["Cohort %ile", 16],
            ]
            return {"columns": columns, "rows": rows}

    su_tables = parse_markdown_tables(analyze_dir / f"{prefix}-summary.md")
    table = find_table(su_tables, header_all=("gene", "tumor-source"))
    if table is not None:
        headers = table["headers"]
        i_gene = col_index(headers, "gene")
        i_tumor = col_index(headers, "tumor-source")
        i_attr = col_index(headers, "non-tumor attribution")
        i_frac = col_index(headers, "tumor fraction")
        rows = []
        for cells in table["rows"]:
            gene = cell(cells, i_gene)
            if not gene:
                continue
            rows.append(
                [
                    gene,
                    cell(cells, i_tumor) or "—",
                    cell(cells, i_attr) or "—",
                    cell(cells, i_frac) or "—",
                ]
            )
            if len(rows) >= 3:
                break
        if rows:
            columns = [
                ["Target", 20],
                ["Tumor-src TPM", 21],
                ["Top non-tumor source", 43],
                ["Tumor frac", 16],
            ]
            return {"columns": columns, "rows": rows}
    return None


def highlight_lines(
    summary_path: Path,
    analysis_path: Optional[Path],
    *,
    max_lines: int = 28,
) -> List[str]:
    lines: List[str] = []
    seen: set[str] = set()

    def append_once(line: str) -> None:
        line = line.lstrip("- ").strip()
        if line and line not in seen:
            seen.add(line)
            lines.append(line)

    if summary_path.exists():
        for raw in summary_path.read_text(errors="replace").splitlines():
            line = clean_markdown(raw)
            if not line or line.startswith("|") or set(line) <= {"-", " "}:
                continue
            if line.startswith("#"):
                continue
            append_once(line)
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
                append_once(clean_markdown(raw))
            if len(lines) >= max_lines + 6:
                break
    return lines[: max_lines + 6]


def find_prefix(analyze_dir: Path) -> str:
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


def find_figure(analyze_dir: Path, prefix: str, suffix: str) -> Optional[Path]:
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


# --------------------------------------------------------------------------- #
# Document assembly.
# --------------------------------------------------------------------------- #
def build_figure_manifest(
    analyze_dir: Path,
    prefix: str,
    *,
    purity_status: str = "resolved",
) -> List[dict]:
    """The belief-gated reader-figure manifest: every registry figure, each with a
    ``present`` flag (True iff the pipeline actually emitted the plot — which it
    only does when the underlying belief passed threshold) and its resolved path."""
    manifest: List[dict] = []
    unresolved_captions = {
        "sample-summary.png": (
            "One-page synthesis of the cancer-type call and the incompatible "
            "purity scenarios; no consensus tumor/non-tumor fraction is assigned."
        ),
        "decomposition-composition.png": (
            "Selected operational tumor/background model used for attribution; "
            "not a resolved sample-composition measurement."
        ),
        "purity-methods.png": (
            "Independent purity estimators support incompatible scenarios; the "
            "operational value is not a fused consensus estimate."
        ),
        "purity.png": (
            "Operational purity scenario and its within-estimator interval; "
            "quantitative consensus is unresolved."
        ),
    }
    for suffix, title, caption in FIGURE_REGISTRY:
        if purity_status == "discordant_estimators":
            caption = unresolved_captions.get(suffix, caption)
        figure = find_figure(analyze_dir, prefix, suffix)
        present = figure is not None
        manifest.append(
            {
                "suffix": suffix,
                "title": title,
                "caption": caption,
                "present": present,
                "path": figure.name if present else None,
            }
        )
    return manifest


def build_report_document(
    analyze_dir: Path,
    prefix: Optional[str] = None,
    *,
    report_view: "ReportView",
) -> dict:
    """Assemble the structured report document for one analyze directory.

    ``report_view`` is the authoritative headline. Everything else is parsed once
    from the just-emitted markdown so the document is a faithful, byte-stable
    projection of the reports.
    """
    analyze_dir = Path(analyze_dir)
    if prefix is None:
        prefix = find_prefix(analyze_dir)
    summary_path = analyze_dir / f"{prefix}-summary.md"
    analysis_path = analyze_dir / f"{prefix}-analysis.md"

    records = parse_summary_records(summary_path)
    headline = report_view.public_dict()

    return {
        "schema_version": SCHEMA_VERSION,
        "prefix": prefix,
        "sample_id": report_view.sample_id,
        "headline": headline,
        "records": records,
        "highlights": highlight_lines(summary_path, analysis_path),
        "therapy": parse_therapy_recommendations(summary_path),
        "targets": _priority_target_table(analyze_dir, prefix),
        "figures": build_figure_manifest(
            analyze_dir,
            prefix,
            purity_status=report_view.purity.status,
        ),
    }


def write_report_document(
    analyze_dir: Path,
    prefix: str,
    *,
    report_view: "ReportView",
) -> Path:
    """Build and write ``<prefix>-report.json`` into *analyze_dir*; return its path."""
    analyze_dir = Path(analyze_dir)
    document = build_report_document(analyze_dir, prefix, report_view=report_view)
    path = analyze_dir / f"{prefix}-report.json"
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False))
    return path


def load_report_document(analyze_dir: Path, prefix: Optional[str] = None) -> dict:
    """Load the structured ``<prefix>-report.json`` written by the pipeline."""
    analyze_dir = Path(analyze_dir)
    if prefix is None:
        prefix = find_prefix(analyze_dir)
    path = analyze_dir / f"{prefix}-report.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No structured report document found at {path}; rerun analysis to "
            "create it"
        )
    return json.loads(path.read_text(errors="replace"))
