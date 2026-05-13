"""Symbol-level QC classification + raw-input profile.

This module answers *which gene-symbol-level features are usable*, and
*what the raw input looks like before any filtering*. Rescaling /
filtering / unit conversions live in
:mod:`trufflepig.expression_normalize`.

The classifier deliberately uses symbol-level heuristics instead of a
heavy annotation dependency. The failure mode it catches is usually
obvious at the gene-symbol layer: a handful of mitochondrial or
rRNA/pseudogene-like entries consume a large fraction of TPM and
distort all downstream absolute expression values.

Pipeline order:

    1. :func:`raw_qc_profile` — inspect raw TPM/FPKM before any
       filtering. Records rRNA / mt / top-K concentration so the
       report can describe the input.
    2. :func:`trufflepig.expression_normalize.normalize_expression` —
       drop technical-RNA features and renormalize.
    3. Optional second :func:`raw_qc_profile` on the normalized table
       to verify cleanup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class GeneQcClass:
    label: str
    group: str


_GENE_NA = {"", "NAN", "NONE", "NULL", "-"}

# Nuclear-retained, ENE-stabilized lncRNAs that survive degradation
# disproportionately and creep up as a fraction of bulk TPM. Their
# 3′ ends are processed by RNase P to a triple-helical U-rich motif
# (Element for Nuclear Expression) that confers exonuclease resistance.
# In single-cell data the artifact runs the opposite direction —
# low MALAT1 is the strongest single-gene proxy for damaged-nucleus /
# ambient-droplet cells.
#
# Refs:
#   - Brown JA et al., PNAS 2012  (MALAT1/NEAT1 ENE triple helix)
#   - Wilusz JE et al., Cell 2008  (MALAT1 3' processing → mascRNA)
#   - Hopp AK et al., bioRxiv 2024  (MALAT1 ↔ nuclear-fraction QC)
#   - Montserrat-Ayuso & Esteve-Codina, BMC Genomics 2024  (MALAT1-low
#     cell fractions in HCA / Tabula Sapiens / Tabula Muris)
#
# Held back from this default panel: KCNQ1OT1, XIST, HOTAIR — also
# nuclear-retained, but their biological signal (imprinting, Xi,
# HOX-axis) is strong enough that auto-dropping would obscure real
# biology more than it removes artifact.
_POLYA_BIAS_LNCRNA_SYMBOLS = frozenset({"MALAT1", "NEAT1"})

_TECHNICAL_RNA_GROUPS = frozenset(
    {"mt_dna", "mt_like_pseudogene", "rrna_like", "polyadenylation_bias_lncrna"}
)


# pirlygenes.gene_families.<family-name> → (qc_label, qc_group). The
# family naming is biological ("nuclear_retained_lncrna"); the QC
# grouping is downstream-specific (drop-by-default sets). Keeping
# the mapping in one place lets the family CSVs change in pirlygenes
# without QC semantics drifting silently.
_FAMILY_TO_QC = {
    "mitochondrial": ("mitochondrial transcript", "mt_dna"),
    "numt_pseudogene": ("mitochondrial pseudogene / NUMT-like", "mt_like_pseudogene"),
    "nuclear_retained_lncrna": (
        "nuclear-retained ENE-stabilized lncRNA",
        "polyadenylation_bias_lncrna",
    ),
    "rrna_and_pseudogene": ("rRNA / rRNA-pseudogene", "rrna_like"),
    "ribosomal_protein": ("ribosomal protein", "ribosomal_protein"),
    "ribosomal_protein_pseudogene": (
        "ribosomal protein pseudogene",
        "ribosomal_protein_pseudogene",
    ),
    "small_noncoding_rna": ("small noncoding RNA", "small_ncrna"),
    "histone": ("histone transcript", "histone"),
    "hemoglobin": ("hemoglobin transcript", "hemoglobin"),
    "immune_receptor_segment": ("immune receptor segment", "immune_receptor"),
}


def _family_to_qc_class(family: str) -> GeneQcClass:
    """Map a pirlygenes gene-family name to its QC class."""
    label, group = _FAMILY_TO_QC.get(family, ("protein-coding/other", "other"))
    return GeneQcClass(label, group)


def classify_gene_qc(
    symbol: str | None = None,
    *,
    ensembl_id: str | None = None,
) -> GeneQcClass:
    """Return a coarse QC class for a gene by symbol and/or ENSG ID.

    Lookup order:

    1. If ``ensembl_id`` is given and the ID belongs to a curated
       :mod:`pirlygenes.gene_families` panel, use the QC group mapped
       from that family. ENSG-first lookup is stable across HGNC symbol
       renames and version-suffix drift.
    2. If ``symbol`` matches a family by symbol, use that.
    3. Otherwise fall back to the symbol-level regex below. The regex
       is the source-of-truth for family CSV regeneration.

    QC groups returned:

    - ``mt_dna``: mitochondrial genome transcripts.
    - ``mt_like_pseudogene``: NUMT-like mitochondrial pseudogenes.
    - ``rrna_like``: nuclear rRNA and rRNA-pseudogene annotations.
    - ``ribosomal_protein`` / ``ribosomal_protein_pseudogene``: real
      RP biology / library complexity signals.
    - ``small_ncrna``: snoRNA, snRNA, Y RNA, miRNA, vault RNA, SRP RNA.
    - ``histone``: non-polyadenylated replication-dependent histone mRNAs.
    - ``immune_receptor``: immunoglobulin or TCR segments.
    - ``hemoglobin``: erythroid / blood-contamination signal.
    - ``polyadenylation_bias_lncrna``: nuclear-retained ENE-stabilized
      lncRNAs (MALAT1, NEAT1) that survive degradation disproportionately.
    - ``other``: everything else.
    """
    # ENSG-first / symbol-second lookup against the curated pirlygenes
    # gene-family panels. Import lazily — pirlygenes is a runtime
    # dependency, but this fallback path keeps this module importable
    # in test fixtures that monkey-patch around pirlygenes.
    family = None
    try:
        from pirlygenes.gene_families import (
            gene_family_for_ensembl_id,
            gene_family_for_symbol,
        )
    except Exception:
        gene_family_for_ensembl_id = None  # type: ignore[assignment]
        gene_family_for_symbol = None  # type: ignore[assignment]

    if ensembl_id and gene_family_for_ensembl_id is not None:
        family = gene_family_for_ensembl_id(ensembl_id)

    if family is None and symbol and gene_family_for_symbol is not None:
        family = gene_family_for_symbol(symbol)

    if family is not None:
        return _family_to_qc_class(family)

    raw = str(symbol or "").strip()
    upper = raw.upper()
    if upper in _GENE_NA:
        return GeneQcClass("unlabeled feature", "other")

    if upper in _POLYA_BIAS_LNCRNA_SYMBOLS:
        return GeneQcClass(
            "nuclear-retained ENE-stabilized lncRNA",
            "polyadenylation_bias_lncrna",
        )

    if upper in {"MT-RNR1", "MT-RNR2"}:
        return GeneQcClass("mitochondrial rRNA", "mt_dna")
    if upper.startswith("MT-"):
        return GeneQcClass("mitochondrial transcript", "mt_dna")
    if re.fullmatch(r"MT(RNR[12]|ATP[68]|CO[123]|CYB|ND[1-6]|ND4L)P\d+", upper):
        return GeneQcClass("mitochondrial pseudogene / NUMT-like", "mt_like_pseudogene")

    if re.fullmatch(r"RNA5SP\d+", upper):
        return GeneQcClass("5S rRNA pseudogene", "rrna_like")
    if re.fullmatch(r"RNA5-8SP\d+", upper):
        return GeneQcClass("5.8S rRNA pseudogene", "rrna_like")
    if re.fullmatch(r"RNA(18S|28S|45S|5S)(P\d+|\d+|[_-].*)?", upper):
        label = {
            "RNA18S": "18S rRNA-like",
            "RNA28S": "28S rRNA-like",
            "RNA45S": "45S pre-rRNA-like",
            "RNA5S": "5S rRNA-like",
        }
        prefix = next((p for p in label if upper.startswith(p)), "RNA5S")
        return GeneQcClass(label[prefix], "rrna_like")
    if upper.startswith(("RNR", "MTRNR")):
        return GeneQcClass("rRNA-like", "rrna_like")

    if re.fullmatch(r"RP[SL]\d+[A-Z]?(P\d+|P)$", upper):
        return GeneQcClass("ribosomal protein pseudogene", "ribosomal_protein_pseudogene")
    if re.fullmatch(r"RP[SL]\d+[A-Z]?", upper) or upper.startswith("RPLP"):
        return GeneQcClass("ribosomal protein", "ribosomal_protein")

    if upper.startswith(("SNORD", "SNORA", "RNU", "Y_RNA", "MIR")):
        return GeneQcClass("small noncoding RNA", "small_ncrna")

    if upper.startswith(
        ("H1-", "H2AC", "H2BC", "H3C", "H4C", "HIST1H", "HIST2H", "HIST3H", "HIST4H")
    ):
        return GeneQcClass("histone transcript", "histone")

    if re.fullmatch(r"HB(A\d?|B|D|E\d?|G\d?|M|Q\d?|Z|ZP\d?|BP\d?)", upper):
        return GeneQcClass("hemoglobin transcript", "hemoglobin")

    if re.fullmatch(
        r"(IGH[ADGME]\d*|IG[HKL][CVJ][A-Z0-9-]*|TR[ABDG][CVJ][A-Z0-9-]*)",
        upper,
    ):
        return GeneQcClass("immune receptor segment", "immune_receptor")

    return GeneQcClass("protein-coding/other", "other")


def is_rescue_feature(symbol: str | None) -> bool:
    """True when a feature should be removed by mtDNA/rRNA rescue."""
    return classify_gene_qc(symbol).group in _TECHNICAL_RNA_GROUPS


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
    "technical_rna_component_phrase",
    "dominant_class_phrase",
    "expression_qc_rescue_summary_line",
]
