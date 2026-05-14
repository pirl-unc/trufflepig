"""Markdown comparison helpers for multiple ``analyze`` report packets."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path


@dataclass(frozen=True)
class AnalyzeSummaryRecord:
    """Small, report-level view of one completed ``analyze`` output."""

    sample_id: str
    output_dir: str
    cancer_call: str
    cancer_type_basis: str
    purity: str
    purity_pct: float | None
    stromal_enrichment: str
    stromal_fold: float | None
    immune_enrichment: str
    immune_fold: float | None
    mhc_i_status: str
    hla_mean_tpm: float | None
    b2m_tpm: float | None
    disease_state: str
    active_pathway: str
    sample_context: str
    rna_quant_qc: str
    technical_normalization: str
    top_therapies: tuple[str, ...]
    therapy_genes: tuple[str, ...]
    caveats: tuple[str, ...]


def _first_matching_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern!r} file in {directory}")
    return matches[0]


def _section_lines(lines: list[str], heading: str) -> list[str]:
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        return []
    out: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if line.strip():
            out.append(line)
    return out


def _line_after_prefix(lines: list[str], prefix: str) -> str:
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _strip_markdown_prefix(line: str) -> str:
    text = line.strip()
    if text.startswith("- "):
        text = text[2:].strip()
    return text


def _clean_table_cell(value: str) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ").strip()


def _parse_first_percent(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)%", text or "")
    if not match:
        return None
    return float(match.group(1))


def _parse_first_fold(text: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[x×]", text or "", re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def _format_delta(before: float | None, after: float | None, suffix: str = "") -> str:
    if before is None or after is None:
        return "n/a"
    delta = after - before
    if abs(delta) < 0.05:
        delta = 0.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}{suffix}"


def _first_matching_optional_file(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def _load_optional_lines(directory: Path, pattern: str) -> list[str]:
    path = _first_matching_optional_file(directory, pattern)
    if path is None:
        return []
    return path.read_text(errors="replace").splitlines()


def _parse_mhc_i_status(analysis_lines: list[str], evidence_lines: list[str]) -> tuple[str, float | None, float | None]:
    evidence_status = _line_after_prefix(evidence_lines, "- **MHC-I status**:")
    mhc_rows = _section_lines(analysis_lines, "## MHC Expression")
    values: dict[str, float] = {}
    for line in mhc_rows:
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 2 or parts[0] in {"Gene", "------"}:
            continue
        try:
            values[parts[0]] = float(parts[1].replace(",", ""))
        except ValueError:
            continue
    hla_values = [values[gene] for gene in ("HLA-A", "HLA-B", "HLA-C") if gene in values]
    hla_mean = sum(hla_values) / len(hla_values) if hla_values else None
    b2m = values.get("B2M")
    if evidence_status:
        return evidence_status, hla_mean, b2m
    if hla_mean is None and b2m is None:
        return "", hla_mean, b2m
    parts = []
    if hla_mean is not None:
        parts.append(f"HLA mean {hla_mean:.0f} TPM")
    if b2m is not None:
        parts.append(f"B2M {b2m:.0f} TPM")
    return ", ".join(parts), hla_mean, b2m


def _therapy_genes(therapy_lines: tuple[str, ...]) -> tuple[str, ...]:
    genes: list[str] = []
    for line in therapy_lines:
        match = re.match(r"\*\*([^*]+)\*\*", line)
        if match:
            genes.append(match.group(1).strip())
    return tuple(dict.fromkeys(genes))


def load_analyze_summary_record(output_dir: str | Path) -> AnalyzeSummaryRecord:
    """Load the summary fields needed for cross-sample comparison."""
    directory = Path(output_dir)
    summary_path = _first_matching_file(directory, "*-summary.md")
    lines = summary_path.read_text(errors="replace").splitlines()
    analysis_lines = _load_optional_lines(directory, "*-analysis.md")
    evidence_lines = _load_optional_lines(directory, "*-evidence.md")

    header = lines[0].replace("# Summary", "", 1).strip()
    sample_id = header[1:].strip() if header.startswith(":") else directory.name
    if not sample_id:
        sample_id = directory.name

    therapy_lines = [
        _strip_markdown_prefix(line)
        for line in _section_lines(lines, "## Top candidate therapies")
        if line.startswith("- **") or line.startswith("*No approved")
    ]
    caveat_lines = [
        _strip_markdown_prefix(line)
        for line in _section_lines(lines, "## Caveats")
        if line.startswith("- ")
    ]
    top_therapies = tuple(therapy_lines[:3])
    stromal_enrichment = _line_after_prefix(analysis_lines, "- **Stromal** enrichment:")
    immune_enrichment = _line_after_prefix(analysis_lines, "- **Immune** enrichment:")
    mhc_i_status, hla_mean_tpm, b2m_tpm = _parse_mhc_i_status(
        analysis_lines, evidence_lines
    )

    return AnalyzeSummaryRecord(
        sample_id=sample_id,
        output_dir=str(directory),
        cancer_call=_line_after_prefix(lines, "**Cancer call:**"),
        cancer_type_basis=_line_after_prefix(lines, "**Cancer-type basis:**"),
        purity=(purity := _line_after_prefix(lines, "**Purity:**")),
        purity_pct=_parse_first_percent(purity),
        stromal_enrichment=stromal_enrichment,
        stromal_fold=_parse_first_fold(stromal_enrichment),
        immune_enrichment=immune_enrichment,
        immune_fold=_parse_first_fold(immune_enrichment),
        mhc_i_status=mhc_i_status,
        hla_mean_tpm=hla_mean_tpm,
        b2m_tpm=b2m_tpm,
        disease_state=_line_after_prefix(lines, "**Disease state:**"),
        active_pathway=_line_after_prefix(lines, "**Active pathway:**"),
        sample_context=_line_after_prefix(lines, "**Sample:**"),
        rna_quant_qc=_line_after_prefix(lines, "**RNA quant QC:**"),
        technical_normalization=(
            _line_after_prefix(lines, "**Expression QC rescue:**")
            or _line_after_prefix(lines, "**Technical-RNA normalization:**")
        ),
        top_therapies=top_therapies,
        therapy_genes=_therapy_genes(top_therapies),
        caveats=tuple(caveat_lines),
    )


def _short_basis(text: str) -> str:
    if not text:
        return ""
    return text.split(";")[0].strip()


def _short_qc(rec: AnalyzeSummaryRecord) -> str:
    parts: list[str] = []
    qc = rec.rna_quant_qc or ""
    mapping = re.search(r"Salmon mapping\s+([0-9.]+%)", qc)
    detected = re.search(r"([0-9,]+/[0-9,]+ genes >=1 TPM)", qc)
    if mapping:
        parts.append(f"mapping {mapping.group(1)}")
    if detected:
        parts.append(detected.group(1))
    if "gene TPM sum near 1.0M" in qc:
        parts.append("TPM sum near 1.0M")
    elif "Gene TPM sum is not close" in qc:
        parts.append("TPM-sum warning")
    tech = rec.technical_normalization or ""
    removed = re.search(r"([0-9]+(?:\.[0-9]+)?% removed)", tech)
    rescue = "Expression QC rescue" if "raw TPM was dominated" in tech else "tech RNA normalized"
    if removed:
        parts.append(f"{rescue}: {removed.group(1)}")
    elif tech:
        parts.append(rescue)
    return "; ".join(parts)


def _assay_difference(before: AnalyzeSummaryRecord, after: AnalyzeSummaryRecord) -> str:
    if before.sample_context == after.sample_context:
        return "similar reported sample context"
    return "sample context differs"


def _gene_delta(before: AnalyzeSummaryRecord, after: AnalyzeSummaryRecord) -> str:
    before_genes = set(before.therapy_genes)
    after_genes = set(after.therapy_genes)
    gained = sorted(after_genes - before_genes)
    lost = sorted(before_genes - after_genes)
    clauses = []
    if gained:
        clauses.append("gained " + ", ".join(gained[:4]))
    if lost:
        clauses.append("lost " + ", ".join(lost[:4]))
    if not clauses:
        return "no top-target gene change"
    return "; ".join(clauses)


def _same_cancer_call(before: str, after: str) -> bool:
    return before.split(" ", 1)[0] == after.split(" ", 1)[0]


def build_analyze_comparison_markdown(
    output_dirs: list[str | Path],
    *,
    title: str = "Analyze Sample Comparison",
) -> str:
    """Build a compact Markdown comparison across completed analyze outputs."""
    records = [load_analyze_summary_record(path) for path in output_dirs]
    if len(records) < 2:
        raise ValueError("Need at least two analyze output directories to compare")

    lines = [
        f"# {title}",
        "",
        "This comparison is assembled from completed `pirlygenes analyze` summary files. "
        "It is a navigation aid, not a clinical recommendation.",
        "",
        "## Snapshot",
        "",
        "| Sample | Cancer call | Basis | Purity | Stroma / immune | MHC-I | Sample/QC context |",
        "|---|---|---|---|---|---|---|",
    ]
    for rec in records:
        lines.append(
            "| "
            + " | ".join(
                _clean_table_cell(value)
                for value in (
                    rec.sample_id,
                    rec.cancer_call,
                    _short_basis(rec.cancer_type_basis),
                    rec.purity,
                    "; ".join(
                        part
                        for part in (rec.stromal_enrichment, rec.immune_enrichment)
                        if part
                    ),
                    rec.mhc_i_status,
                    f"{rec.sample_context} {_short_qc(rec)}".strip(),
                )
            )
            + " |"
        )

    lines.extend(["", "## Longitudinal Deltas", ""])
    lines.append(
        "| Transition | Cancer-call change | Purity delta | Stroma delta | Immune delta | MHC-I delta | Top-target change | Assay/QC note |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---|")
    for before, after in zip(records, records[1:]):
        cancer_change = (
            "same broad call"
            if _same_cancer_call(before.cancer_call, after.cancer_call)
            else f"{before.cancer_call} -> {after.cancer_call}"
        )
        lines.append(
            "| "
            + " | ".join(
                _clean_table_cell(value)
                for value in (
                    f"{before.sample_id} -> {after.sample_id}",
                    cancer_change,
                    _format_delta(before.purity_pct, after.purity_pct, "%"),
                    _format_delta(before.stromal_fold, after.stromal_fold, "x"),
                    _format_delta(before.immune_fold, after.immune_fold, "x"),
                    _format_delta(before.hla_mean_tpm, after.hla_mean_tpm, " TPM"),
                    _gene_delta(before, after),
                    _assay_difference(before, after),
                )
            )
            + " |"
        )

    lines.extend(["", "## Biology And Response State", ""])
    lines.append("| Sample | Disease state | Active pathway / signature |")
    lines.append("|---|---|---|")
    for rec in records:
        lines.append(
            "| "
            + " | ".join(
                _clean_table_cell(value)
                for value in (
                    rec.sample_id,
                    rec.disease_state or "No summary-level disease-state call.",
                    rec.active_pathway or "No summary-level active-pathway call.",
                )
            )
            + " |"
        )

    lines.extend(["", "## Therapy Shortlists", ""])
    for rec in records:
        lines.append(f"### {rec.sample_id}")
        if rec.top_therapies:
            for therapy in rec.top_therapies:
                lines.append(f"- {therapy}")
        else:
            lines.append("- No therapy shortlist was available in the summary.")
        lines.append("")

    lines.extend(
        [
            "## Interpretation Notes",
            "",
        ]
    )
    if len({rec.cancer_call.split(' ', 1)[0] for rec in records}) > 1:
        lines.append(
            "- Cancer calls differ across samples. Treat RNA-inferred labels as hypotheses and separate true biological evolution from site, purity, and assay effects."
        )
    else:
        lines.append(
            "- Treat RNA-inferred cancer labels as hypotheses unless pathology or clinical diagnosis confirms them."
        )
    if len({rec.sample_context for rec in records if rec.sample_context}) > 1:
        lines.append(
            "- Assay/sample-context labels differ across samples, so raw TPM deltas should be interpreted through the source-attributed and context TPM fields in each evidence report."
        )
    if any(rec.hla_mean_tpm is not None for rec in records):
        lines.append(
            "- MHC-I deltas are expression-based presentation context only; TCR/TCRm eligibility still needs HLA typing and trial-specific criteria."
        )
    lines.extend(
        [
            "- Patient-facing LLM use requires diagnosis, stage, prior lines, current medications, MSI/MMR/TMB, mutations/fusions/CNVs, relevant imaging such as HER2/PSMA, and trial availability.",
        ]
    )

    unique_caveats = []
    seen = set()
    for rec in records:
        for caveat in rec.caveats:
            key = re.sub(r"\s+", " ", caveat).strip().lower()
            if not key or key in seen:
                continue
            if key.startswith("patient-facing llm"):
                continue
            if key.startswith("technical-rna normalization"):
                continue
            seen.add(key)
            unique_caveats.append(caveat)
    if unique_caveats:
        lines.extend(["", "## Report-Specific Caveats To Carry Forward", ""])
        for caveat in unique_caveats[:6]:
            lines.append(f"- {caveat}")

    return "\n".join(lines) + "\n"
