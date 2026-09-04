# Licensed under the Apache License, Version 2.0

"""Typed, explicitly non-diagnostic biomarker proxies from one bulk RNA sample.

These results may prioritize confirmatory testing and add context to downstream
therapy reasoning.  They must never be promoted to assay-defined eligibility.
Every value is tagged as measured patient bulk RNA, an RNA-model estimate,
or an external reference-panel statistic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Mapping


HER2_PROXY_ID = "her2_erbb2_amplicon_rna"
HER2_PROXY_METHOD_VERSION = "her2_erbb2_amplicon_rna_v1_exploratory"
HER2_CORE_GENE = "ERBB2"
HER2_COMPANION_GENES = ("GRB7", "STARD3", "MIEN1", "PGAP3")
HER2_PANEL_GENES = (HER2_CORE_GENE, *HER2_COMPANION_GENES)


@dataclass(frozen=True)
class RNABiomarkerProxy:
    """One auditable RNA-only proxy with its clinical boundary attached."""

    proxy_id: str
    label: str
    status: str
    clinical_claim: str
    eligibility_established: bool
    input_class: str
    reference_cancer_type: str
    reference_panel_kind: str
    reference_panel_component_codes: tuple[str, ...]
    reference_panel_sources: tuple[str, ...]
    reference_panel_aggregation: str
    method_version: str
    calibration_status: str
    genes_expected: tuple[str, ...]
    genes_measured: tuple[str, ...]
    genes_missing: tuple[str, ...]
    gene_evidence: tuple[dict[str, Any], ...]
    panel_geomean_fold_vs_reference: float | None
    decision_basis: str
    confirmation_priority: str
    required_confirmation: tuple[str, ...]
    downstream_effect: str
    caveats: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _clean_symbol_map(values: Mapping[str, Any] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in (values or {}).items():
        symbol = str(key or "").strip().upper()
        number = _finite_float(value)
        if symbol and number is not None:
            out[symbol] = max(0.0, number)
    return out


def _range_rows_by_symbol(ranges_df) -> dict[str, dict[str, Any]]:
    if ranges_df is None or not hasattr(ranges_df, "columns"):
        return {}
    if "symbol" not in ranges_df.columns:
        return {}
    from .common import ranges_records

    return {
        str(row.get("symbol") or "").strip().upper(): row
        for row in ranges_records(ranges_df)
        if str(row.get("symbol") or "").strip()
    }


def _reference_panel(
    cancer_type: str,
) -> tuple[dict[str, float], tuple[str, ...], tuple[str, ...]]:
    """Return external observed-cohort medians for an entity or member union.

    The bounded cancer-reference accessor resolves aggregate ontology entities
    such as CRC to their loadable member cohorts (COAD and READ).  Keeping this
    path aligned with cancer-type context avoids incorrectly calling a complete
    patient panel "missing" merely because the aggregate has no literal
    ``CRC_TPM`` column in the legacy wide matrix.
    """

    import pandas as pd

    from .reference import cancer_reference_expression

    requested = str(cancer_type or "").strip().upper()
    if not requested:
        return {}, (), ()
    try:
        reference = cancer_reference_expression(
            cancer_types=requested,
            genes=list(HER2_PANEL_GENES),
            normalize="tpm_clean",
            format="long",
            include_provenance=True,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError):
        return {}, (), ()
    required = {"Symbol", "expression"}
    if reference is None or reference.empty or not required.issubset(reference.columns):
        return {}, (), ()
    if "normalization" in reference.columns:
        reference = reference[
            reference["normalization"].astype(str).str.lower().eq("tpm_clean")
        ].copy()
    if reference.empty:
        return {}, (), ()
    reference = reference.assign(
        _symbol=reference["Symbol"].fillna("").astype(str).str.strip().str.upper(),
        _expression=pd.to_numeric(reference["expression"], errors="coerce"),
    )
    reference = reference[
        reference["_symbol"].astype(bool) & reference["_expression"].notna()
    ]
    medians = (
        reference.groupby("_symbol", sort=False)["_expression"]
        .median()
        .clip(lower=0.0)
        .astype(float)
        .to_dict()
    )
    component_codes = tuple(
        sorted(
            {
                str(value).strip().upper()
                for value in reference.get("cancer_code", ())
                if str(value).strip()
            }
        )
    )
    sources = tuple(
        sorted(
            {
                str(value).strip()
                for value in reference.get("source_cohort", ())
                if str(value).strip()
            }
        )
    )
    return medians, component_codes, sources


def _geomean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(sum(math.log(max(value, 1e-9)) for value in values) / len(values))


def score_her2_rna_proxy(
    sample_tpm_by_symbol: Mapping[str, Any] | None,
    reference_cancer_type: str,
    *,
    ranges_df=None,
    purity: Mapping[str, Any] | None = None,
) -> RNABiomarkerProxy:
    """Score an exploratory ERBB2/17q12 RNA pattern from one bulk sample.

    The decision uses measured patient bulk TPM relative to the selected cancer
    reference.  Estimated tumor attribution is a source check, not a replacement
    measurement.  Thresholds are intentionally conservative and explicitly
    marked as heuristic until outcome-linked validation is available.
    """

    sample = _clean_symbol_map(sample_tpm_by_symbol)
    reference, reference_components, reference_sources = _reference_panel(
        reference_cancer_type
    )
    range_rows = _range_rows_by_symbol(ranges_df)
    measured = tuple(gene for gene in HER2_PANEL_GENES if gene in sample)
    missing = tuple(gene for gene in HER2_PANEL_GENES if gene not in sample)
    evidence: list[dict[str, Any]] = []
    folds: dict[str, float] = {}

    for gene in HER2_PANEL_GENES:
        bulk_tpm = sample.get(gene)
        reference_tpm = reference.get(gene)
        fold = None
        if bulk_tpm is not None and reference_tpm is not None:
            fold = (bulk_tpm + 0.5) / (reference_tpm + 0.5)
            folds[gene] = fold
        estimate = range_rows.get(gene) or {}
        evidence.append(
            {
                "symbol": gene,
                "patient_bulk_tpm_measured": (
                    round(bulk_tpm, 3) if bulk_tpm is not None else None
                ),
                "external_cancer_reference_panel_median_tpm": (
                    round(reference_tpm, 3) if reference_tpm is not None else None
                ),
                "fold_vs_external_cancer_reference_panel": (
                    round(fold, 3) if fold is not None else None
                ),
                "patient_tumor_attributed_tpm_rna_model_estimate": (
                    round(value, 3)
                    if (value := _finite_float(estimate.get("attr_tumor_tpm")))
                    is not None
                    else None
                ),
                "patient_tumor_attributed_fraction_rna_model_estimate": (
                    round(value, 3)
                    if (
                        value := _finite_float(
                            estimate.get("attr_tumor_fraction")
                        )
                    )
                    is not None
                    else None
                ),
                "pan_cancer_reference_percentile": (
                    round(value, 3)
                    if (value := _finite_float(estimate.get("tcga_percentile")))
                    is not None
                    else None
                ),
                "rna_model_source_status": str(
                    estimate.get("attr_status")
                    or estimate.get("attr_top_compartment")
                    or ""
                ),
            }
        )

    panel_fold = _geomean([folds[g] for g in HER2_PANEL_GENES if g in folds])
    erbb2_tpm = sample.get(HER2_CORE_GENE)
    erbb2_fold = folds.get(HER2_CORE_GENE)
    companion_high = sum(
        1 for gene in HER2_COMPANION_GENES if folds.get(gene, 0.0) >= 2.0
    )
    enough_panel = HER2_CORE_GENE in measured and len(measured) >= 4
    reference_genes = tuple(gene for gene in HER2_PANEL_GENES if gene in reference)
    enough_reference = (
        HER2_CORE_GENE in reference_genes and len(reference_genes) >= 4
    )
    bulk_pattern = bool(
        enough_panel
        and enough_reference
        and erbb2_tpm is not None
        and erbb2_tpm >= 10.0
        and erbb2_fold is not None
        and erbb2_fold >= 3.0
        and companion_high >= 2
        and panel_fold is not None
        and panel_fold >= 2.0
    )

    erbb2_estimate = range_rows.get(HER2_CORE_GENE) or {}
    # This is the fraction of the *measured ERBB2 RNA* assigned to the
    # estimated tumor component.  It is not the sample's tumor fraction.
    erbb2_tumor_attributed_fraction = _finite_float(
        erbb2_estimate.get("attr_tumor_fraction")
    )
    source_conflict = bool(
        bulk_pattern
        and (
            (
                erbb2_tumor_attributed_fraction is not None
                and erbb2_tumor_attributed_fraction < 0.30
            )
            or bool(erbb2_estimate.get("tme_dominant"))
            or bool(erbb2_estimate.get("matched_normal_over_predicted"))
        )
    )

    if not enough_panel:
        status = "indeterminate"
        basis = (
            "Insufficient gene coverage in the patient's measured bulk RNA: "
            f"measured {len(measured)}/{len(HER2_PANEL_GENES)} panel genes."
        )
        priority = "standard"
    elif not enough_reference:
        status = "indeterminate"
        basis = (
            f"All {len(measured)} panel genes were measured in patient bulk RNA, "
            "but the external observed cancer-cohort reference covered only "
            f"{len(reference_genes)}/{len(HER2_PANEL_GENES)} panel genes for "
            f"{reference_cancer_type or 'the selected cancer type'}."
        )
        priority = "standard"
    elif source_conflict:
        status = "discordant"
        basis = (
            "Measured bulk ERBB2/17q12 RNA is elevated, but the RNA source estimate "
            "does not cleanly attribute ERBB2 to the patient tumor component."
        )
        priority = "high"
    elif bulk_pattern:
        status = "supported"
        source_clause = (
            "; the RNA model assigns "
            f"{erbb2_tumor_attributed_fraction:.0%} of measured ERBB2 RNA "
            "to the estimated tumor component"
            if erbb2_tumor_attributed_fraction is not None
            else "; tumor/background source model unavailable"
        )
        basis = (
            f"Measured patient bulk ERBB2 is {erbb2_fold:.1f}x the external "
            f"selected-cancer reference panel and {companion_high}/4 companion "
            f"genes are at least 2x that reference{source_clause}."
        )
        priority = "high"
    elif (
        erbb2_tpm is not None
        and erbb2_fold is not None
        and panel_fold is not None
        and enough_reference
        and erbb2_tpm < 10.0
        and erbb2_fold < 1.0
        and panel_fold < 1.5
    ):
        status = "not_supported"
        basis = (
            "The measured ERBB2/17q12 RNA panel does not show the conservative "
            "amplicon-like elevation pattern in this sample."
        )
        priority = "routine_if_clinically_indicated"
    else:
        status = "indeterminate"
        basis = (
            "ERBB2/17q12 RNA is measurable but does not meet either conservative "
            "support or non-support criteria."
        )
        priority = "standard"

    caveats = [
        "Exploratory RNA heuristic; not clinically validated as a HER2 assay.",
        "RNA cannot establish HER2 IHC score, HER2-low status, amplification, or treatment eligibility.",
    ]
    purity_value = _finite_float((purity or {}).get("overall_estimate"))
    if purity_value is not None and purity_value < 0.20:
        caveats.append(
            "A low estimated tumor fraction can dilute measured bulk ERBB2 and destabilize source attribution."
        )
    if not range_rows:
        caveats.append(
            "Tumor/background attribution was unavailable; the proxy is based on measured mixed-specimen RNA."
        )

    return RNABiomarkerProxy(
        proxy_id=HER2_PROXY_ID,
        label="HER2 / ERBB2 amplicon-like RNA context",
        status=status,
        clinical_claim="context_only",
        eligibility_established=False,
        input_class="single_bulk_rna_gene_tpm",
        reference_cancer_type=str(reference_cancer_type or "").strip(),
        reference_panel_kind="external_observed_cancer_cohort",
        reference_panel_component_codes=reference_components,
        reference_panel_sources=reference_sources,
        reference_panel_aggregation="median_of_component_cohort_medians",
        method_version=HER2_PROXY_METHOD_VERSION,
        calibration_status="exploratory_heuristic_not_clinically_validated",
        genes_expected=HER2_PANEL_GENES,
        genes_measured=measured,
        genes_missing=missing,
        gene_evidence=tuple(evidence),
        panel_geomean_fold_vs_reference=(
            round(panel_fold, 3) if panel_fold is not None else None
        ),
        decision_basis=basis,
        confirmation_priority=priority,
        required_confirmation=(
            "HER2 IHC with specimen, assay, score, and date",
            "reflex ISH/FISH for an equivocal IHC result or when required by the indication",
            "clinical diagnosis, stage, treatment line, and prior HER2-directed therapy",
        ),
        downstream_effect=(
            "prioritize confirmatory HER2 testing and annotate HER2-directed rows; "
            "do not create or remove eligibility"
        ),
        caveats=tuple(caveats),
    )


def score_rna_biomarker_proxies(
    sample_tpm_by_symbol: Mapping[str, Any] | None,
    reference_cancer_type: str,
    *,
    ranges_df=None,
    purity: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the versioned RNA-only proxy collection for one sample."""

    her2 = score_her2_rna_proxy(
        sample_tpm_by_symbol,
        reference_cancer_type,
        ranges_df=ranges_df,
        purity=purity,
    )
    return {"her2": her2.to_dict()}


_HER2_TEXT = re.compile(r"\b(?:her2|erbb2)\b", re.IGNORECASE)


def target_row_uses_her2(target_row: Any) -> bool:
    """Whether a curated therapy row depends on HER2/ERBB2 context."""

    if not hasattr(target_row, "get"):
        return False
    symbol = str(target_row.get("symbol") or "").strip().upper()
    if symbol in {"ERBB2", "HER2"}:
        return True
    text = " ".join(
        str(target_row.get(key) or "")
        for key in (
            "agent",
            "agent_class",
            "indication",
            "rationale",
            "eligibility_note",
            "eligibility_basis",
        )
    )
    return bool(_HER2_TEXT.search(text))


def her2_proxy_for_analysis(analysis: Any) -> dict[str, Any]:
    if not isinstance(analysis, Mapping):
        return {}
    proxies = analysis.get("rna_biomarker_proxies") or {}
    proxy = proxies.get("her2") if isinstance(proxies, Mapping) else None
    return dict(proxy) if isinstance(proxy, Mapping) else {}


def her2_proxy_summary_line(analysis: Any) -> str:
    """One bounded report sentence that never crosses the assay boundary."""

    proxy = her2_proxy_for_analysis(analysis)
    if not proxy:
        return ""
    status = str(proxy.get("status") or "indeterminate")
    basis = str(proxy.get("decision_basis") or "").strip()
    if status == "supported":
        display_status = "high priority for confirmation"
        interpretation = "prioritizes confirmatory HER2 IHC with reflex ISH/FISH"
    elif status == "discordant":
        display_status = "needs source review"
        interpretation = (
            "has a bulk-versus-source discrepancy; resolve it with HER2 IHC and ISH/FISH"
        )
    elif status == "not_supported":
        display_status = "not prioritized by RNA"
        interpretation = (
            "does not support an amplicon-like RNA pattern, but cannot establish "
            "HER2-negative or exclude HER2-low disease"
        )
    else:
        display_status = "uncertain"
        interpretation = "is indeterminate; use the indication-specific clinical HER2 assay"
    return (
        f"**HER2 RNA proxy ({display_status}):** {basis} The RNA result "
        f"{interpretation}. This can prioritize confirmatory testing, but cannot "
        "by itself select treatment or establish eligibility."
    )


def her2_proxy_therapy_context(target_row: Any, analysis: Any) -> str:
    """Downstream therapy annotation for a HER2-dependent curated row."""

    if not target_row_uses_her2(target_row):
        return ""
    proxy = her2_proxy_for_analysis(analysis)
    if not proxy:
        return ""
    status = str(proxy.get("status") or "indeterminate")
    if status == "supported":
        return (
            "RNA HER2 pattern prioritizes IHC/ISH confirmation; "
            "it does not establish eligibility"
        )
    if status == "discordant":
        return (
            "RNA HER2 pattern needs source review; resolve with IHC/ISH "
            "before treatment selection"
        )
    if status == "not_supported":
        return (
            "RNA HER2 pattern does not show an amplicon-like pattern, but "
            "cannot establish HER2-negative or exclude HER2-low disease"
        )
    return "RNA HER2 pattern is uncertain; confirm with the required clinical assay"
