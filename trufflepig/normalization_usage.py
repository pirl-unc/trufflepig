"""Ledger of expression feature spaces used by production analysis.

The base abundance unit is conformed clean TPM. Housekeeping normalization is
not used by any default decision path. Pre-normalization QC separately records
the median TPM of housekeeping genes as a scale check; that is a QC statistic,
not an HK-normalized feature space. Explicit HK A/B modes may remain callable,
but they are outside the default report path.
"""

from __future__ import annotations

from enum import Enum


class Basis(str, Enum):
    CLEAN_TPM = "clean_tpm"
    LOG1P_CLEAN_TPM = "log1p_clean_tpm"
    WITHIN_SAMPLE_PERCENTILE = "within_sample_percentile"
    COHORT_PERCENTILE = "cohort_percentile"
    SPEARMAN = "spearman"
    HK = "housekeeping"


NORMALIZATION_USAGE: dict[str, dict[str, object]] = {
    "cancer_type_signature": {
        "location": "plot_embedding._compute_cancer_type_signature_stats",
        "current": Basis.WITHIN_SAMPLE_PERCENTILE,
        "rationale": (
            "Panel genes are selected from cross-cohort clean-TPM contrasts, then "
            "scored by their rank within the sample. On 592 representative samples, "
            "within-sample percentile reached 0.302 exact type accuracy versus 0.198 "
            "for isolated HK and 0.304 for the former HK-combined score. It improved "
            "lineage accuracy (0.767 versus 0.764) and severe-dilution accuracy at "
            "70% background (0.257 versus 0.242), so HK was not unequivocally superior."
        ),
    },
    "subtype_signature": {
        "location": "subtype_signature.compute_subtype_signature_stats",
        "current": Basis.WITHIN_SAMPLE_PERCENTILE,
        "rationale": (
            "Uses the same sample score as the broad ranker. Cross-cohort and "
            "within-parent specificity are already enforced when subtype panels are "
            "derived, so a second HK cohort comparison duplicated that role."
        ),
    },
    "lineage_programs": {
        "location": "signal_views / lineage_evidence / lineage_marker_recall",
        "current": (
            Basis.LOG1P_CLEAN_TPM,
            Basis.COHORT_PERCENTILE,
        ),
        "rationale": (
            "Absolute program burden and cross-cancer specificity are the two decision "
            "views. Within-sample rank and cohort z-score remain diagnostics, not extra "
            "votes. On 619 representatives, log1p+cohort percentile produced epithelial/"
            "NE AUC 0.894/0.908 versus HK 0.862/0.825."
        ),
    },
    "curated_lineage_panels": {
        "location": "lineage_panels.score_panel",
        "current": (
            Basis.LOG1P_CLEAN_TPM,
            Basis.COHORT_PERCENTILE,
        ),
        "rationale": (
            "Positive markers integrate log burden and cross-cohort percentile; negative "
            "markers stay on absolute clean TPM. Across 16 panels and 619 representatives, "
            "macro AUC improved from 0.817 (HK) to 0.857 and eligible top-parent accuracy "
            "from 0.340 to 0.488."
        ),
    },
    "purity_and_lineage_fraction": {
        "location": "tumor_purity.estimate_tumor_purity",
        "current": Basis.CLEAN_TPM,
        "rationale": (
            "Purity is an additive mixture problem. Sample, cancer reference, and normal/"
            "TME profiles share a per-million clean abundance scale, so gene-program mass "
            "and per-gene mixture equations are evaluated directly. Synthetic dilution "
            "showed clean-TPM mass was perfectly monotonic and had lower endpoint-calibrated "
            "error than HK across generic, adipose, and smooth-muscle backgrounds."
        ),
    },
    "tumor_expression": {
        "location": "plot_tumor_expr.estimate_tumor_expression",
        "current": Basis.CLEAN_TPM,
        "rationale": (
            "Observed, fitted-background, cohort, and tumor-residual quantities are all "
            "additive TPM components. Direct clean-TPM subtraction avoids converting into "
            "and back out of an arbitrary HK fold scale."
        ),
    },
    "whole_profile_centroids": {
        "location": "cancer_type_centroid.centroid_correlations",
        "current": Basis.SPEARMAN,
        "rationale": (
            "Whole-profile Spearman is already rank based. A z-score ensemble regressed "
            "compartment accuracy and confidence calibration in the representative sweep."
        ),
    },
    "decomposition_nnls": {
        "location": "decomposition.engine._fit_one_hypothesis",
        "current": Basis.CLEAN_TPM,
        "rationale": (
            "Linear clean TPM with component-specificity marker weights preserves RNA-mixture "
            "additivity. It recovered four exact healthy-component mixtures with mean fraction "
            "error 5.9e-11, versus 0.013 for the best HK fit and 0.081 for percentile. Earlier "
            "bulk-cancer dilution scores were invalid as fraction truth because the TCGA cancer "
            "profiles already contain unknown normal tissue and TME. Percentile/log-cohort "
            "background-separated cancer typing remains a sequential, independent result "
            "after NNLS."
        ),
    },
    "input_scale_qc": {
        "location": "load_expression._expression_scale_qc",
        "current": "input_linear_tpm_housekeeping_median_qc",
        "rationale": (
            "The housekeeping median is measured on input-derived linear TPM after gene "
            "aggregation and before clean-TPM conformance. It detects wrong log scale, "
            "collapsed library complexity, or implausibly low absolute RNA scale that later "
            "normalization could otherwise hide. It is QC only and contributes no cancer or "
            "purity evidence."
        ),
    },
}


def usage(consumer: str) -> dict[str, object]:
    """Return the current feature-space record for ``consumer``."""
    return NORMALIZATION_USAGE.get(consumer, {})


__all__ = ["Basis", "NORMALIZATION_USAGE", "usage"]
