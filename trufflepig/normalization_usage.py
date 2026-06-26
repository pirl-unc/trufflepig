"""Normalization-basis ledger — which scoring path uses HK-ratio vs z-score vs percentile.

Three bases are available together (so any consumer can be moved without new plumbing):
  - ``hk``         — signal / housekeeping (median-or-sum). Legacy default; purity-sensitive.
  - ``zscore``     — per-gene z-score across cancer types (``_cached_reference_matrices()["z_matrix"]``
                     for the reference side; ``zscore_normalize`` for samples).
  - ``percentile`` — per-gene percentile rank across cancer types
                     (``_cached_reference_matrices()["percentile_matrix"]``; ``oncoref.percentile_rank``).

Why move off HK-ratio: measured on 266 real samples / 54 types, per-gene **z-score ≫ HK-ratio** for
cancer-type classification (0.84 vs 0.78) and is purity-robust. Target policy:
  - **z-score** by default (the measured win),
  - **percentile** where z-score's normality assumption is weak (heavy-tailed / zero-inflated marker
    panels, small gene sets), and
  - **HK** only where it is genuinely essential (QC sanity; any classification that needs the absolute
    housekeeping-relative magnitude rather than a cross-type contrast).

This ledger is the migration tracker: each consumer records where it reads today (``current``), where
it should land (``target``), and why. Update it as each consumer is migrated + threshold-retuned.
"""
from __future__ import annotations

from enum import Enum


class Basis(str, Enum):
    HK = "hk"
    ZSCORE = "zscore"
    PERCENTILE = "percentile"


# consumer key -> migration record
NORMALIZATION_USAGE: dict[str, dict[str, object]] = {
    "cancer_type_centroids": {
        "location": "tumor_purity._cached_reference_matrices(normalize='housekeeping') + centroid scoring",
        "current": Basis.HK,
        "target": Basis.ZSCORE,
        "rationale": "Measured z-score ≫ HK-ratio for type classification; z_matrix already computed alongside.",
    },
    "estimate_purity_surrogate": {
        "location": "tumor_purity._geneset_hk_ratio (stromal/immune/signature)",
        "current": Basis.HK,
        "target": Basis.ZSCORE,
        "rationale": "ESTIMATE-style stroma/immune contrast; z-score of the gene-set vs cohort is the contrast we want.",
    },
    "lineage_panels": {
        "location": "lineage_panels.py / cancer_type_evidence.py (marker / sample_hk_median vs cohort_hk_ratio)",
        "current": Basis.HK,
        "target": Basis.PERCENTILE,
        "rationale": "Curated marker panels are small + heavy-tailed; percentile rank is robust where z-score normality is weak.",
    },
    "epithelial_ne_detection": {
        "location": "lineage_evidence.epithelial_hk_ratio / lineage_marker_recall NE ratios",
        "current": Basis.HK,
        "target": Basis.PERCENTILE,
        "rationale": "Same as lineage panels — small marker sets, rank-robust gating.",
    },
    "decomposition_mixture": {
        "location": "expression_decomposition / tumor_purity._mixture_cohort_lineage_summary (hk_syms)",
        "current": Basis.HK,
        "target": Basis.ZSCORE,
        "rationale": "Subtype concordance is a cross-type contrast; z-score basis aligns with the centroid path.",
    },
    "qc_housekeeping_median": {
        "location": "load_expression.py (core-HK median sanity warning)",
        "current": Basis.HK,
        "target": Basis.HK,
        "rationale": "ESSENTIAL: an absolute HK-median floor is the point of the QC check; not a cross-type contrast.",
    },
}


def usage(consumer: str) -> dict[str, object]:
    """Migration record for a consumer key, or an empty dict if untracked."""
    return NORMALIZATION_USAGE.get(consumer, {})
