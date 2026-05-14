"""Per-candidate evidence tables for analysis.md.

The candidate-ranking table in analysis.md shows *which* cohorts the
classifier considered and their composite scores, but not *why* — a
reader (human or LLM) can't follow the chain "this gene + that gene
+ this TPM contrast → BRCA-Basal". This module builds the missing
detail layer.

For each top-K candidate, emit:

  - Signature panel evidence: each panel gene's sample TPM, the
    candidate cohort's median (broad or winning subtype), a competing
    cohort's median for contrast, and a percentile-rank score.
  - Rescue payload (when a heuristic rescue fired): the full
    marker readout from ``support_override`` rendered as a table.
  - Tiebreaker payload (when normal-tissue tiebreaker fired): per-
    candidate primary-tissue match scores so the reader can see why
    one tissue pattern outvoted the other.

The format is deliberately verbose. An LLM consuming analysis.md
should be able to re-derive the cancer-biology rationale from the
tables alone, without re-running the pipeline.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

# Per-cohort cohort-median sources to use for "comparison column" in
# evidence tables. The competing cohort is chosen to make the contrast
# biologically informative — basal-BRCA evidence is most useful when
# shown vs. its squamous nemesis (HNSC); the tiebreaker scenarios
# similarly pick deliberate comparisons.
_COMPETING_COHORT_FOR_TABLE = {
    "BRCA": "HNSC",  # basal-BRCA vs squamous overlap
    "HNSC": "BRCA",
    "LUSC": "BRCA",
    "ESCA": "BRCA",
    "CESC": "BRCA",
    "COAD": "READ",  # sister cohorts in the colorectal family
    "READ": "COAD",
    "LUAD": "LUSC",  # adeno vs squamous lung
    "KIRC": "KIRP",  # kidney sisters
    "PRAD": "BLCA",  # prostate vs bladder near-neighbor
    "BLCA": "PRAD",
}


def _format_tpm(value: float | None) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 0.1:
        return f"{v:.2f}"
    if v > 0:
        return f"{v:.3f}"
    return "0"


def _per_subtype_evidence_table(
    cancer_code: str,
    subtype: str | None,
    sample_tpm_by_symbol: dict[str, float],
    cohort_medians: dict[str, float],
    competitor_medians: dict[str, float] | None,
    signature_genes: Iterable[str],
    *,
    title_hint: str | None = None,
) -> list[str]:
    """Markdown lines for one candidate's signature-evidence table."""
    genes = [str(g) for g in signature_genes if g]
    if not genes:
        return []
    label = subtype or cancer_code
    title = title_hint or f"Signature evidence for **{label}**"
    competitor_code = _COMPETING_COHORT_FOR_TABLE.get(cancer_code, "")
    has_competitor = competitor_medians is not None and competitor_code
    lines: list[str] = []
    lines.append(f"#### {title}\n")
    if has_competitor:
        lines.append(
            f"| Gene | Sample TPM | {label} median | {competitor_code} median | Δlog2 |"
        )
        lines.append("|---|---:|---:|---:|---:|")
    else:
        lines.append(f"| Gene | Sample TPM | {label} median |")
        lines.append("|---|---:|---:|")

    for gene in genes:
        sample_v = float(sample_tpm_by_symbol.get(gene, 0.0) or 0.0)
        ref_v = float(cohort_medians.get(gene, 0.0) or 0.0)
        comp_v = (
            float((competitor_medians or {}).get(gene, 0.0) or 0.0)
            if has_competitor
            else None
        )
        if has_competitor:
            try:
                dlog2 = math.log2((ref_v + 1.0) / (comp_v + 1.0))
                dlog2_str = f"{dlog2:+.2f}"
            except (ValueError, ZeroDivisionError):
                dlog2_str = "—"
            lines.append(
                f"| {gene} | {_format_tpm(sample_v)} | {_format_tpm(ref_v)} | "
                f"{_format_tpm(comp_v)} | {dlog2_str} |"
            )
        else:
            lines.append(
                f"| {gene} | {_format_tpm(sample_v)} | {_format_tpm(ref_v)} |"
            )
    if has_competitor:
        lines.append(
            f"\n*Δlog2 = log2((subtype + 1) / ({competitor_code} median + 1)). "
            "Positive values mean the gene is elevated in the candidate vs. its "
            "natural competitor cohort, so its presence in the sample TPM column "
            "supports the candidate call.*\n"
        )
    else:
        lines.append("")
    return lines


def _rescue_evidence_table(support_override: dict | None) -> list[str]:
    """Render the rescue's marker readout as a structured table."""
    if not support_override:
        return []
    kind = str(support_override.get("kind") or "")
    if kind != "tnbc_basal_brca_misclassification":
        return []
    lines: list[str] = []
    lines.append("#### TNBC / basal-BRCA misclassification rescue — marker evidence\n")
    lines.append(
        "The classifier's broad top picks were the squamous family, but the "
        "sample shows the basal-mammary cytokeratin program with the luminal "
        "and squamous programs both off. This is the classic basal-like BRCA "
        "vs. squamous overlap (Hoadley 2014, Damrauer 2014, Lehmann 2011). "
        "Markers that flipped the call:\n"
    )
    sections: list[tuple[str, dict[str, Any] | None]] = [
        ("Basal cytokeratin program (elevated)", support_override.get("keratin_tpm")),
        ("Luminal program (suppressed)", support_override.get("luminal_marker_tpm")),
        ("Basal-mammary positive markers", support_override.get("basal_positive_tpm")),
        ("Squamous program (absent)", support_override.get("squamous_program_tpm")),
    ]
    foxc1_tpm = support_override.get("foxc1_tpm")
    if foxc1_tpm is not None:
        sections.insert(
            2,
            ("Basal-mammary TF (FOXC1)", {"FOXC1": foxc1_tpm}),
        )
    upk_sum = support_override.get("urothelial_panel_sum_tpm")
    if upk_sum is not None:
        sections.append(("Urothelial program sum (off)", {"UPK total": upk_sum}))
    lines.append("| Section | Gene | TPM |")
    lines.append("|---|---|---:|")
    for label, payload in sections:
        if not payload:
            continue
        for gene, val in payload.items():
            lines.append(f"| {label} | {gene} | {_format_tpm(val)} |")
    lines.append("")
    return lines


def _tiebreaker_evidence_table(rows: list[dict]) -> list[str]:
    """Render normal-tissue tiebreaker payload as a structured table."""
    if not rows:
        return []
    tiebreaker_row = next(
        (
            r
            for r in rows
            if r.get("normal_tissue_tiebreaker", {}).get("applied")
        ),
        None,
    )
    annotated_rows = [r for r in rows if r.get("primary_tissue") is not None]
    if not annotated_rows:
        return []
    lines: list[str] = []
    if tiebreaker_row:
        info = tiebreaker_row["normal_tissue_tiebreaker"]
        lines.append(
            "#### Normal-tissue tiebreaker — parental tissue evidence\n"
        )
        lines.append(
            f"The top two cohorts sat within {(1 - info['close_window']) * 100:.0f}% "
            "of each other on composite support. Each candidate's "
            "parental tissue match against the sample's normal-tissue "
            "profile broke the tie:\n"
        )
    else:
        lines.append("#### Parental tissue scores for close candidates\n")
    lines.append("| Cohort | Primary tissue | Tissue match score | Tiebreaker boost |")
    lines.append("|---|---|---:|---:|")
    for row in annotated_rows:
        code = str(row.get("code") or "")
        tissue = str(row.get("primary_tissue") or "")
        score = float(row.get("primary_tissue_match_score") or 0.0)
        info = row.get("normal_tissue_tiebreaker") or {}
        boost = (
            f"+{(info['boost_factor'] - 1.0) * 100:.0f}%"
            if info.get("applied") and info.get("boost_factor")
            else "—"
        )
        lines.append(
            f"| {code} | {tissue} | {score:.3f} | {boost} |"
        )
    lines.append("")
    return lines


def _subtype_medians_lookup() -> dict[tuple[str, str], dict[str, float]]:
    """``{(cancer_code, subtype) -> {symbol -> tumor_tpm_median}}``.

    Pulled from :func:`trufflepig.reference.subtype_deconvolved_expression`
    on the same TPM-normalized footing used for subtype panel selection.
    Cached implicitly via the reference loader's own cache.
    """
    from .reference import subtype_deconvolved_expression

    sub_df = subtype_deconvolved_expression(
        technical_rna_normalize=True,
        renormalize_to_million=True,
    )
    if sub_df is None or sub_df.empty:
        return {}
    out: dict[tuple[str, str], dict[str, float]] = {}
    for (code, subtype), group in sub_df.groupby(["cancer_code", "subtype"], sort=False):
        out[(str(code), str(subtype))] = dict(
            zip(
                group["symbol"].astype(str),
                group["tumor_tpm_median"].astype(float),
            )
        )
    return out


def build_candidate_evidence_block(
    candidate_trace: list[dict] | None,
    sample_tpm_by_symbol: dict[str, float],
    *,
    top_k: int = 3,
) -> list[str]:
    """Build the per-candidate evidence-table markdown block.

    Inputs:
      - ``candidate_trace`` — ranked candidates as returned by
        ``rank_cancer_type_candidates``. Each row carries
        ``code``, ``signature_score``, ``winning_subtype``, optional
        ``support_override`` (rescue payload), and the per-row
        ``normal_tissue_tiebreaker`` / ``primary_tissue`` fields.
      - ``sample_tpm_by_symbol`` — sample TPM keyed by gene symbol.

    Returns a list of markdown lines to extend ``analysis.md`` with.

    Median lookup rules:
      - When a candidate has a ``winning_subtype`` (e.g. BRCA_Basal),
        the "median" column shows the **subtype-deconvolved** tumor
        median for that subtype, not the broad cohort median. This
        matters because the broad-BRCA median is luminal-dominated;
        labeling its values "BRCA_Basal median" would be misleading.
      - Without a subtype, fall back to the broad cohort median from
        ``pan_cancer_expression``.
      - The competitor column always uses the broad cohort median of
        the competing cohort (no subtype available for most of them).
    """
    if not candidate_trace:
        return []

    # Lazy imports — keep this module light when not consumed.
    from .plot_embedding import _get_cancer_type_signature_panels
    from .reference import pan_cancer_expression
    from .subtype_signature import subtype_signature_panels

    panels = _get_cancer_type_signature_panels(n_signature_genes=12)
    sub_panels = subtype_signature_panels(top_n=12)
    subtype_medians = _subtype_medians_lookup()
    pan = pan_cancer_expression(technical_rna_normalize=True)
    pan = pan.drop_duplicates(subset="Symbol").set_index("Symbol")
    pan_dict_cache: dict[str, dict[str, float]] = {}

    def _broad_cohort_medians(code: str) -> dict[str, float]:
        cached = pan_dict_cache.get(code)
        if cached is not None:
            return cached
        col = f"FPKM_{code}"
        if col not in pan.columns:
            pan_dict_cache[code] = {}
            return {}
        series = pan[col].astype(float).dropna()
        as_dict = series.to_dict()
        pan_dict_cache[code] = as_dict
        return as_dict

    def _candidate_medians(code: str, subtype: str | None) -> dict[str, float]:
        if subtype:
            sub = subtype_medians.get((code, subtype))
            if sub:
                return sub
        return _broad_cohort_medians(code)

    lines: list[str] = [
        "### Cancer Type Inference — Per-Candidate Evidence\n",
        "Each block below shows the signature-panel genes that drove the call "
        "for one of the top candidates, with their sample expression alongside "
        "the cohort's reference median and (when informative) a competing "
        "cohort's median. The intent is that an LLM or human reader can "
        "re-derive the biology from the tables without re-running the pipeline.\n",
    ]

    # Rescue evidence first if it fired — it explains *why* the leading
    # row outranks the broad signature top.
    rescue_row = next(
        (row for row in candidate_trace if row.get("support_override")),
        None,
    )
    if rescue_row:
        lines.extend(_rescue_evidence_table(rescue_row.get("support_override")))

    # Normal-tissue tiebreaker (runs after rescues, so render second).
    lines.extend(_tiebreaker_evidence_table(candidate_trace))

    # Per-candidate signature evidence for top-K.
    for row in candidate_trace[:top_k]:
        code = str(row.get("code") or "")
        subtype = row.get("signature_subtype_promoted") or row.get("winning_subtype")
        if subtype and (code, subtype) in sub_panels:
            panel_genes = sub_panels[(code, subtype)]
            title_hint = (
                f"Signature evidence for **{code}** "
                f"(winning subtype: `{subtype}`)"
            )
        else:
            panel_genes = panels.get(code, ())
            title_hint = f"Signature evidence for **{code}**"
        if not panel_genes:
            continue
        cohort = _candidate_medians(code, subtype)
        competitor_code = _COMPETING_COHORT_FOR_TABLE.get(code)
        # Competitor always uses the broad cohort median — most cohorts
        # don't have subtype data, and the contrast we want is "this
        # candidate vs its natural alternative" at the cohort level.
        competitor = (
            _broad_cohort_medians(competitor_code) if competitor_code else None
        )
        lines.extend(
            _per_subtype_evidence_table(
                code,
                subtype,
                sample_tpm_by_symbol,
                cohort,
                competitor,
                panel_genes,
                title_hint=title_hint,
            )
        )

    return lines


__all__ = ["build_candidate_evidence_block"]
