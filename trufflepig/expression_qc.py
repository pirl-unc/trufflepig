"""Per-sample QC narration on top of the pirlygenes gene-QC classifier.

The QC classifier itself — :class:`GeneQcClass`, :func:`classify_gene_qc`,
:func:`is_rescue_feature`, plus the technical-RNA gene-set constants —
lives in pirlygenes 5.1+ alongside the rest of the curated reference
data. This module holds the *analysis-time narration* layer that
trufflepig adds on top: per-sample raw-input profile, group-share
summaries, dominant-class phrasing, and the markdown blocks the report
renders before and after normalization.

Pipeline order:

    1. :func:`raw_qc_profile` — inspect raw TPM/FPKM before any
       filtering. Records rRNA / mt / top-K concentration so the
       report can describe the input.
    2. :func:`trufflepig.expression_normalize.normalize_expression`
       (re-exported from pirlygenes) — drop technical-RNA features and
       renormalize.
    3. Optional second :func:`raw_qc_profile` on the normalized table
       to verify cleanup.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from pirlygenes.expression.qc import (
    TECHNICAL_RNA_GROUPS,
    GeneQcClass,
    classify_gene_qc,
    is_rescue_feature,
)

# Display ordering for the per-category pre-normalization QC block.
# Removed groups come first (drop-by-default), then retained groups; within
# each block, members render in the listed order so reports stay stable.
_QC_GROUP_DISPLAY_ORDER = (
    # Removed:
    "mt_dna",
    "rrna_like",
    "polyadenylation_bias_lncrna",
    "mt_like_pseudogene",
    # Retained:
    "ribosomal_protein",
    "ribosomal_protein_pseudogene",
    "small_ncrna",
    "histone",
    "hemoglobin",
    "immune_receptor",
    "other",
)

_QC_GROUP_DISPLAY_LABEL = {
    "mt_dna": "mitochondrial transcript",
    "mt_like_pseudogene": "mitochondrial pseudogene / NUMT-like",
    "rrna_like": "rRNA / rRNA-pseudogene",
    "polyadenylation_bias_lncrna": "nuclear-retained ENE-stabilized lncRNA",
    "ribosomal_protein": "ribosomal protein",
    "ribosomal_protein_pseudogene": "ribosomal protein pseudogene",
    "small_ncrna": "small noncoding RNA",
    "histone": "histone transcript",
    "hemoglobin": "hemoglobin transcript",
    "immune_receptor": "immune receptor segment",
    "other": "protein-coding / other",
}


def summarize_qc_class_shares(
    gene_tpm_items: Iterable[tuple[str, float]],
) -> dict[str, object]:
    """Summarize total TPM share by QC class/group.

    Includes a per-gene breakdown for the
    ``polyadenylation_bias_lncrna`` panel (MALAT1, NEAT1) so callers
    can surface their individual TPM shares in QC text/visualizations
    before they are dropped by :func:`normalize_expression`. MALAT1 in
    particular is a well-validated proxy for sample integrity —
    high in degraded bulk samples, low in damaged-nucleus single cells.
    """
    group_tpm: dict[str, float] = {}
    class_tpm: dict[str, float] = {}
    lncrna_qc_artifact_tpm: dict[str, float] = {}
    total = 0.0
    for gene, value in gene_tpm_items:
        try:
            tpm = float(value)
        except (TypeError, ValueError):
            continue
        if tpm <= 0:
            continue
        qc = classify_gene_qc(gene)
        total += tpm
        group_tpm[qc.group] = group_tpm.get(qc.group, 0.0) + tpm
        class_tpm[qc.label] = class_tpm.get(qc.label, 0.0) + tpm
        if qc.group == "polyadenylation_bias_lncrna":
            sym = str(gene or "").strip().upper()
            lncrna_qc_artifact_tpm[sym] = lncrna_qc_artifact_tpm.get(sym, 0.0) + tpm

    def _fraction_map(values: Mapping[str, float]) -> dict[str, float]:
        if total <= 0:
            return {key: 0.0 for key in values}
        return {
            key: round(float(val) / total, 6)
            for key, val in sorted(values.items(), key=lambda item: (-item[1], item[0]))
        }

    group_share = _fraction_map(group_tpm)
    class_share = _fraction_map(class_tpm)
    rrna_pseudogene_fraction = float(
        sum(
            val
            for key, val in class_share.items()
            if "rRNA pseudogene" in str(key)
        )
    )
    rrna_like_fraction = float(group_share.get("rrna_like", 0.0))
    mt_dna_fraction = float(group_share.get("mt_dna", 0.0))
    mt_like_pseudogene_fraction = float(group_share.get("mt_like_pseudogene", 0.0))
    mitochondrial_rrna_fraction = float(class_share.get("mitochondrial rRNA", 0.0))
    nuclear_rrna_like_fraction = max(
        0.0, rrna_like_fraction - rrna_pseudogene_fraction
    )
    lncrna_qc_artifact_fraction = float(
        group_share.get("polyadenylation_bias_lncrna", 0.0)
    )
    lncrna_per_gene_share = {
        sym: (val / total) if total > 0 else 0.0
        for sym, val in sorted(
            lncrna_qc_artifact_tpm.items(), key=lambda item: (-item[1], item[0])
        )
    }
    return {
        "total_tpm": float(total),
        "group_tpm": dict(sorted(group_tpm.items())),
        "class_tpm": dict(sorted(class_tpm.items())),
        "group_share": group_share,
        "class_share": class_share,
        "mt_dna_fraction": mt_dna_fraction,
        "mt_like_pseudogene_fraction": mt_like_pseudogene_fraction,
        "mitochondrial_rrna_fraction": mitochondrial_rrna_fraction,
        "mt_non_rrna_fraction": max(0.0, mt_dna_fraction - mitochondrial_rrna_fraction),
        "rrna_like_fraction": rrna_like_fraction,
        "nuclear_rrna_like_fraction": nuclear_rrna_like_fraction,
        "rrna_pseudogene_fraction": rrna_pseudogene_fraction,
        "lncrna_qc_artifact_fraction": lncrna_qc_artifact_fraction,
        "lncrna_qc_artifact_per_gene_share": lncrna_per_gene_share,
        "rrna_plus_mt_fraction": float(
            mt_dna_fraction + mt_like_pseudogene_fraction + rrna_like_fraction
        ),
    }


def top_gene_concentration(
    gene_tpm_items: Iterable[tuple[str, float]],
    k_values: Sequence[int] = (1, 5, 10, 50, 100),
) -> dict[str, object]:
    """Top-K mass concentration of a raw expression vector.

    Returns the share of total positive TPM held by the top ``k`` genes
    for each k in ``k_values``, plus the top-100 gene/TPM tuples for
    audit. Useful before any filtering — a sample dominated by a small
    set of features is usually a degradation, FFPE, or library-prep
    failure signal.
    """
    rows = []
    total = 0.0
    for gene, value in gene_tpm_items:
        try:
            tpm = float(value)
        except (TypeError, ValueError):
            continue
        if tpm <= 0:
            continue
        rows.append((str(gene), tpm))
        total += tpm
    rows.sort(key=lambda r: (-r[1], r[0]))

    top_shares: dict[str, float] = {}
    cum = 0.0
    last_k = 0
    for rank, (_gene, tpm) in enumerate(rows, start=1):
        cum += tpm
        for k in k_values:
            if rank == k:
                top_shares[f"top_{k}_share"] = (cum / total) if total > 0 else 0.0
        if rank >= max(k_values):
            last_k = rank
            break
    # Fill in any k that exceeds the number of positive features.
    for k in k_values:
        top_shares.setdefault(f"top_{k}_share", (cum / total) if total > 0 else 0.0)

    return {
        "total_tpm": float(total),
        "n_positive_features": len(rows),
        "top_shares": top_shares,
        "top_features": [
            {"gene": gene, "tpm": tpm, "share": (tpm / total) if total > 0 else 0.0}
            for gene, tpm in rows[: max(100, max(k_values, default=0))]
        ],
        "last_ranked_k": last_k,
    }


def raw_qc_profile(
    gene_tpm_items: Iterable[tuple[str, float]],
    *,
    top_k_values: Sequence[int] = (1, 5, 10, 50, 100),
) -> dict[str, object]:
    """One-shot raw-input QC profile — call before any filtering.

    Combines :func:`summarize_qc_class_shares` (mtDNA / rRNA-like /
    ribosomal / histone / immune-receptor / hemoglobin TPM shares) and
    :func:`top_gene_concentration` (top-K mass concentration) into a
    single typed report. The intent is to characterize a sample's raw
    composition before normalization makes downstream decisions.
    """
    items = list(gene_tpm_items)
    class_shares = summarize_qc_class_shares(items)
    top = top_gene_concentration(items, k_values=top_k_values)
    return {
        "class_shares": class_shares,
        "top_concentration": top,
    }


def qc_category_levels_lines(
    summary: Mapping[str, object] | None,
    *,
    min_share: float = 0.0005,
) -> list[str]:
    """Pre-normalization QC: TPM share per gene-category, drop-list first.

    Returns markdown bullet lines for the categories present in
    ``summary["group_share"]``. The four drop-by-default groups
    (``mt_dna``, ``rrna_like``, ``polyadenylation_bias_lncrna``,
    ``mt_like_pseudogene``) render first under a "**zeroed before
    renormalization**" header; retained groups follow under a
    "**kept**" header. Categories below ``min_share`` are omitted to
    keep the block legible.

    The block makes the *order* of the technical-RNA removal step
    visible at a glance: a reader scanning the QC section sees what
    will be dropped and what stays, with their raw TPM shares.
    """
    if not summary:
        return []
    group_share = summary.get("group_share") or {}
    if not isinstance(group_share, Mapping):
        return []

    def _fmt(frac: float) -> str:
        if frac >= 0.01:
            return f"{frac:.0%}"
        if frac >= 0.001:
            return f"{frac:.1%}"
        return "<0.1%"

    removed_rows: list[tuple[str, str, float]] = []
    kept_rows: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for group in _QC_GROUP_DISPLAY_ORDER:
        seen.add(group)
        frac = float(group_share.get(group) or 0.0)
        if frac < min_share:
            continue
        label = _QC_GROUP_DISPLAY_LABEL.get(group, group)
        row = (group, label, frac)
        if group in TECHNICAL_RNA_GROUPS:
            removed_rows.append(row)
        else:
            kept_rows.append(row)
    # Pick up any group we don't know about (defensive — shouldn't happen).
    for group, frac in group_share.items():
        if group in seen:
            continue
        try:
            f = float(frac)
        except (TypeError, ValueError):
            continue
        if f < min_share:
            continue
        kept_rows.append((str(group), str(group), f))

    if not removed_rows and not kept_rows:
        return []

    lines: list[str] = ["- **Raw TPM by gene category**:"]
    if removed_rows:
        lines.append(
            "    - *Zeroed before renormalization (technical RNA, dropped for "
            "reference comparability):*"
        )
        for group, label, frac in removed_rows:
            lines.append(f"        - **{label}** — {_fmt(frac)} *(`{group}`)*")
    if kept_rows:
        lines.append("    - *Kept (renormalized to 1.0M):*")
        for group, label, frac in kept_rows:
            lines.append(f"        - {label} — {_fmt(frac)} *(`{group}`)*")
    return lines


def technical_rna_component_phrase(summary: Mapping[str, object] | None) -> str:
    """Human-readable breakdown of mtDNA/rRNA-like / lncRNA-artifact TPM burden."""
    if not summary:
        return ""
    components = [
        ("rRNA pseudogene", float(summary.get("rrna_pseudogene_fraction") or 0.0)),
        ("nuclear rRNA-like", float(summary.get("nuclear_rrna_like_fraction") or 0.0)),
        ("mitochondrial rRNA", float(summary.get("mitochondrial_rrna_fraction") or 0.0)),
        ("other mtDNA", float(summary.get("mt_non_rrna_fraction") or 0.0)),
        (
            "NUMT/mt-like pseudogene",
            float(summary.get("mt_like_pseudogene_fraction") or 0.0),
        ),
    ]
    lncrna = summary.get("lncrna_qc_artifact_per_gene_share") or {}
    if lncrna:
        for sym, frac in lncrna.items():
            components.append((str(sym), float(frac)))
    else:
        lncrna_frac = float(summary.get("lncrna_qc_artifact_fraction") or 0.0)
        if lncrna_frac > 0:
            components.append(("nuclear-retained lncRNA", lncrna_frac))
    shown = [(label, frac) for label, frac in components if frac >= 0.005]
    if not shown:
        return ""
    return ", ".join(f"{label} {frac:.0%}" for label, frac in shown)


def dominant_class_phrase(dominant: list[dict] | None) -> str:
    """Short phrase for warnings when dominant genes share one QC class."""
    rows = dominant or []
    if not rows:
        return ""
    top = rows[0]
    gene = str(top.get("gene") or "").strip()
    label = str(top.get("qc_class") or "").strip()
    if gene and label and label != "protein-coding/other":
        return f"{gene}; {label}"
    if gene:
        return gene
    return label


def expression_qc_rescue_summary_line(record: dict | None) -> str:
    """One-line report summary for mtDNA/rRNA technical normalization."""
    if not record or not record.get("applied"):
        return ""
    removed = float(record.get("removed_fraction") or 0.0)
    high_burden = bool(record.get("high_burden"))
    component_phrase = technical_rna_component_phrase(
        record.get("qc_class_shares") or {}
    )
    removed_label = "<1%" if 0.0 < removed < 0.005 else f"{removed:.0%}"
    component_clause = (
        f" ({component_phrase}; {removed_label} removed)"
        if component_phrase
        else f" ({removed_label} removed)"
    )
    top_removed = record.get("top_removed_genes") or []
    top_clause = ""
    if top_removed and removed >= 0.005:
        top = top_removed[0]
        gene = str(top.get("gene") or "").strip()
        qc_class = str(top.get("qc_class") or "").strip()
        share = float(top.get("share") or 0.0)
        if gene:
            top_clause = f"; top removed feature {gene}"
            if qc_class:
                top_clause += f" ({qc_class}"
                if share >= 0.005:
                    top_clause += f", {share:.0%} of raw TPM"
                top_clause += ")"
    prefix = (
        "**Expression QC rescue:** raw TPM was dominated by technical RNA features"
        if high_burden
        else "**Technical-RNA normalization:** mtDNA/rRNA-like features were removed for reference comparability"
    )
    return (
        f"{prefix}{component_clause}; downstream cancer, target, and pathway "
        "calculations use TPM after zeroing those features and renormalizing "
        f"the remaining genes{top_clause}."
    )


__all__ = [
    "GeneQcClass",
    "classify_gene_qc",
    "is_rescue_feature",
    "summarize_qc_class_shares",
    "top_gene_concentration",
    "raw_qc_profile",
    "qc_category_levels_lines",
    "technical_rna_component_phrase",
    "dominant_class_phrase",
    "expression_qc_rescue_summary_line",
]
