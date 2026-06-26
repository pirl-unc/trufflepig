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
    SPEARMAN = "spearman"  # whole-profile log-TPM rank correlation (the centroid scorer)


# consumer key -> migration record
NORMALIZATION_USAGE: dict[str, dict[str, object]] = {
    "cancer_type_centroids": {
        "location": "cancer_type_centroid.centroid_correlations (compartment_call, leaf restriction, veto)",
        "current": Basis.SPEARMAN,
        "target": Basis.SPEARMAN,  # EVALUATED z-score, REJECTED — keep Spearman
        "rationale": (
            "MIGRATION EVALUATED AND REJECTED (Phase 3). The premise — that this scorer was HK-based "
            "at ~0.78 — was wrong: it is whole-profile log-TPM Spearman, already 0.92 type / 0.94 "
            "compartment on the production 118-medoid reference (beats the HK-0.78 baseline). A z-score "
            "ensemble REGRESSED the clean compartment call on 565 real representative samples "
            "(0.943→0.931) and its confidence margins (which gate leaf restriction: 9 wrong-but-confident "
            "at margin 0.05 vs 0 for Spearman). It also does NOT fix the dilution failure it targeted — a "
            "structured tissue contaminant (70% liver) matches that tissue's cancer type (LIHC) on EVERY "
            "whole-profile basis incl. z-score, so that case needs decomposition, not normalization. "
            "z-score's win is real only for a generic (diffuse) contaminant. Primitive kept as "
            "cancer_type_centroid.zscore_centroid_correlations; harnesses: scripts/centroid_zscore_ab.py, "
            "scripts/centroid_compartment_tune.py."
        ),
    },
    "estimate_purity_surrogate": {
        "location": "tumor_purity._geneset_hk_ratio (stromal/immune/signature)",
        "current": Basis.HK,
        "target": Basis.HK,  # EVALUATED z-score, no demonstrated win — keep HK enrichment
        "rationale": (
            "MIGRATION EVALUATED AND REJECTED (Phase 4a). ESTIMATE estimates purity per-type via "
            "stromal/immune ENRICHMENT (sample ratio ÷ the type's reference ratio), which already "
            "removes the type baseline. On synthetic dilution (scripts/estimate_zscore_ab.py), the "
            "production HK-enrichment is PERFECTLY monotonic in true purity WITHIN a type (mean "
            "per-type |Spearman| = 1.000 for generic AND structured stromal contaminants), while a "
            "z-score contrast is slightly WORSE (0.89-0.98). z-score only wins on a pooled cross-type "
            "metric, which is not how ESTIMATE operates (it is always type-relative). ESTIMATE's real "
            "weakness is structural — unsound for heme/mesenchymal (already gated out), not a "
            "normalization-basis problem z-score would fix."
        ),
    },
    "lineage_panels": {
        "location": "lineage_panels.py / cancer_type_evidence.py (marker / sample_hk_median vs cohort_hk_ratio)",
        "current": Basis.HK,
        "target": Basis.PERCENTILE,  # ALREADY in the 5-view consensus; no replacement win (Phase 4b)
        "rationale": (
            "EVALUATED (Phase 4b, scripts/lineage_view_ab.py). The epithelial/NE lineage gate is ALREADY a "
            "5-view consensus (signal_views: hk + within_pct + log1p + cohort_pct(percentile) + "
            "cohort_z, equal-weighted), so the percentile target is already in place. Per-view AUC for "
            "epithelial/NE detection over 565 reps: the equal-weight CONSENSUS (0.935/0.940) beats every "
            "single view; percentile (cohort_pct 0.915/0.935) is a STRONG view (≈/> HK 0.897/0.852); "
            "z-score (cohort_z 0.713/0.812) is the WORST view (small heavy-tailed panels — the percentile-"
            "over-z instinct confirmed). Dropping HK from the consensus is a wash (+0.003 epi / -0.002 NE). "
            "No replacement win — keep the multi-view consensus. (Residual pure-HK path: lineage_panels."
            "score_panel obligate/high-marker gates; percentile≈HK there too, not worth the churn.)"
        ),
    },
    "epithelial_ne_detection": {
        "location": "lineage_evidence.epithelial_hk_ratio / lineage_marker_recall NE ratios",
        "current": Basis.HK,
        "target": Basis.PERCENTILE,  # see lineage_panels — already multi-view; no win (Phase 4b)
        "rationale": "Same finding as lineage_panels (Phase 4b): already a 5-view consensus incl. percentile; consensus beats any single view; z-score worst. No replacement win.",
    },
    "decomposition_mixture": {
        "location": "tumor_purity._mixture_cohort_lineage_summary subtype PICK (_mixture_subtype_pick_scores)",
        "current": Basis.PERCENTILE,  # ACTIONED (Phase 4c) — the only migration that landed
        "target": Basis.PERCENTILE,
        "rationale": (
            "MIGRATED (Phase 4c) — the ONE consumer where HK was genuinely worst, and the only change this "
            "migration landed. The FAITHFUL production check (scripts/decomposition_production_check.py) "
            "confirmed the HK TME-excess-weighted concordance picks the right SARC subtype only 6/15 "
            "(0.40) — WORSE than the simplified proxy, because the TME-excess subtraction removes the "
            "stromal program that mesenchymal subtypes (LMS/SYN/liposarcoma) actually share, with "
            "SARC_LPS_UNSPEC an attractor. Replaced the subtype RANKING with a percentile-cosine pick over "
            "the union panel (_mixture_subtype_pick_scores): production pick accuracy 6/15 -> 14/15 (0.93). "
            "Targeted/low-blast-radius: only the SURFACED '#171 subtype: X-consistent' label changes — the "
            "reported concordance/support_factor stay HK so purity values are untouched. percentile (not "
            "z-score; see Phase 4b). Regression test: test_issue_171_mixture_cohort.py parametrized over "
            "LMS/SYN/LPS_UNSPEC. Caveat: validated on n=15 (all the data that exists — only 3 SARC subtypes "
            "have panel+profile)."
        ),
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
