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


# --------------------------------------------------------------------------- #
# Expression-UNIT catalog — every basis a sample value is expressed in before it hits a threshold,
# and where each is used. The point: see the whole spread in one place so a migration to a SMALLER,
# simpler set of units can be focused. (The NORMALIZATION_USAGE map above tracks the per-consumer
# migration verdicts; this catalog is the broader unit-by-unit inventory.) Base scale for everything
# is clean-TPM (clean_tpm.normalize_to_reference_space — every sample is conformed once).
# --------------------------------------------------------------------------- #
class Unit(str, Enum):
    CLEAN_TPM = "clean_tpm"                       # absolute value on the conform scale
    HK_RATIO = "hk_ratio"                         # value / housekeeping-median (depth-invariant)
    COHORT_PERCENTILE = "cohort_percentile"       # rank of sample vs reference cohorts, [0,1]
    COHORT_ZSCORE = "cohort_zscore"               # (value - cohort mean)/sd, per gene
    WITHIN_SAMPLE_PERCENTILE = "within_sample_pct"  # rank of gene WITHIN the sample, [0,1]
    LOG1P_TPM = "log1p_tpm"                       # log1p(clean-TPM)
    SPEARMAN = "spearman"                         # whole-profile rank correlation, [-1,1]
    LOG2_FOLD = "log2_fold"                       # log2(sample / reference)
    MAD_LOG2_RATIO = "mad_log2_ratio"             # dispersion of per-arm log2 ratios (aneuploidy)


# unit -> {what it means, where it's used (consumer @ rough file:line), migration note}
EXPRESSION_UNITS: dict[Unit, dict[str, object]] = {
    Unit.CLEAN_TPM: {
        "what": "absolute sample value on the 16/9/75 conform scale (~1e6 budget)",
        "used_by": [
            "cancer_type_evidence per-gene anchors: RET>=50/CALCR>=5/CEACAM5>=5, IGF2>=100/DLK1>=5/SALL4>=5, "
            "NUTM1>=10, MYB>=20/75, SMARCB1<=5/40, contrast hi/lo 5/2/1, _LOCAL_REFERENCE_MIN_TPM=5 (:54-196)",
            "tumor_purity TNBC/squamous gates _TNBC_*_TPM (~:3079); _LINEAGE_COMPETITOR_FLOOR=5 (:596)",
            "lineage_panels _HIGH_MARKER_MIN_TPM=5 + low-marker caps 1-50",
            "tumor_evidence CA9>=50, glycolysis baseline 50; healthy_vs_tumor CTA/oncofetal>=3, CTA-sum>=30",
            "load_expression core-HK-median QC <5; subtype_signature min_subtype_tpm=5",
        ],
        "migration": (
            "SPLIT by intent. KEEP absolute where the number encodes genuine high/low expression "
            "(RET/IGF2/CA9/SMARCB1, the TNBC luminal-off gates) — percentile would LOSE that (a top-pctl "
            "gene can be ~0 TPM). For the specificity-PROXY gates (contrast hi/lo, _HIGH_MARKER_MIN_TPM) "
            "→ HK_RATIO for depth-invariance, or COHORT_PERCENTILE. This is the ~40-anchor fragility set."
        ),
    },
    Unit.HK_RATIO: {
        "what": "geneset (or gene) / housekeeping-median — cancels per-sample depth, basis-locked",
        "used_by": [
            "ESTIMATE surrogate (tumor_purity _geneset_hk_ratio); lineage_panels 0.5x cohort-HK gate",
            "lineage_marker_recall NE recall (0.30 / 2.0 HK-ratio); family panels; tumor-specific gene selection",
            "signature detection floor _SIGNATURE_DETECTION_FLOOR_HK=0.1; signal_views 'hk' view",
        ],
        "migration": "Validated already-good for ESTIMATE (perfect per-type monotonicity). Keep; depth-invariant.",
    },
    Unit.COHORT_PERCENTILE: {
        "what": "midrank of the sample value among the reference cohorts (cross-type specificity), [0,1]",
        "used_by": [
            "signature_score (_compute_cancer_type_signature_stats) — the PRIMARY cancer-type signal; "
            "MIGRATED off HK to the COMBINED cohort_pct x within_sample_pct filter (HK-migration #7, "
            "medoid AUTO lineage 95->99/118). The cohort-pct leg keeps specificity + score spread.",
            "mixture subtype pick (_mixture_subtype_pick_scores); signal_views 'cohort_pct'",
        ],
        "migration": (
            "Strong for specificity, but PURITY-SENSITIVE alone (collapses under dilution). Best paired "
            "with within_sample_pct (the combined filter that landed #7). The preferred specificity unit."
        ),
    },
    Unit.COHORT_ZSCORE: {
        "what": "(value - cohort mean)/sd per gene; purity-robust to a GENERIC contaminant only",
        "used_by": [
            "tumor-specific gene SELECTION (zscore_min=1.0/0.25/1.5); signal_views 'cohort_z' vote",
            "z_matrix in _cached_reference_matrices; cancer_type_centroid.zscore_centroid_correlations (built, UNUSED)",
            "optional expression_classifier co-signal (learned, opt-in)",
        ],
        "migration": "WORST basis for small marker panels (Phase 4b). Keep to gene selection / the learned co-signal.",
    },
    Unit.WITHIN_SAMPLE_PERCENTILE: {
        "what": "rank of a gene WITHIN the one sample, [0,1]; reference-free, dominance not specificity",
        "used_by": [
            "expression_decomposition proliferation (>=60) + restricted-marker burden (>95); signal_views 'within_pct'",
        ],
        "migration": "The reference-free DEFAULT for new/ambiguous thresholds (clean-competitive, 0.861). TME-contaminated; don't use for specificity.",
    },
    Unit.LOG1P_TPM: {
        "what": "log1p(clean-TPM) — compresses dynamic range; the substrate for Spearman + log views",
        "used_by": ["cancer_type_centroid Spearman input; signal_views 'log1p'; proliferation log2 panels"],
        "migration": "Transform, not a threshold scale. Leave.",
    },
    Unit.SPEARMAN: {
        "what": "whole-profile rank correlation of sample vs a reference profile, [-1,1]",
        "used_by": [
            "centroid_correlations / compartment_call (margins _COMPARTMENT_CONFIDENT_MARGIN=0.025)",
            "cancer_type_evidence coarse-reference gate (_COARSE_REFERENCE_MIN_RHO=0.75); healthy_vs_tumor HPA match",
        ],
        "migration": "Validated best for the compartment call (z-score regressed it, Phase 3). Keep. Margins are reference-calibrated.",
    },
    Unit.LOG2_FOLD: {
        "what": "log2(sample / reference) — fold over pan-cancer or within-cohort",
        "used_by": ["cancer_type_evidence local-reference (MIN_LOG2_VS_PAN=1.0); subtype_signature panel build"],
        "migration": "Fold-over-reference; semantically close to COHORT_PERCENTILE for specificity. Candidate to unify.",
    },
    Unit.MAD_LOG2_RATIO: {
        "what": "median-abs-deviation of per-chromosome-arm log2 ratios — genomic instability, not expression level",
        "used_by": ["aneuploidy_axis; expression_decomposition (_ANEUPLOIDY_STRONG=0.20); purity_calibration"],
        "migration": "Distinct quantity (instability), not a per-gene expression unit. Leave separate.",
    },
}


def unit_catalog(unit: "Unit | None" = None):
    """The whole EXPRESSION_UNITS catalog, or one unit's record."""
    return EXPRESSION_UNITS if unit is None else EXPRESSION_UNITS.get(unit, {})
