# Licensed under the Apache License, Version 2.0

"""Tumor purity estimation from bulk RNA-seq expression data.

Uses within-sample gene-set enrichment ratios to estimate tumor content.
The key insight: the ratio (gene_set_TPM_sum / housekeeping_TPM_sum) is
scale-independent because both numerator and denominator come from the
same sample.

Comparing this ratio to the TCGA cohort reference gives a purity estimate:

    purity ≈ (tumor_signal / HK)_sample / (tumor_signal / HK)_reference

Multiple gene sets are scored independently:
- Cancer-type signature genes (auto-detected or specified)
- ESTIMATE stromal genes (Yoshihara et al. 2013, Nat Commun)
- ESTIMATE immune genes (Yoshihara et al. 2013, Nat Commun)

Higher stromal/immune scores → lower tumor purity.
"""

from collections import Counter

import numpy as np
import pandas as pd

from pirlygenes.gene_sets_cancer import (

    cancer_family_panels, cancer_family_panels_df, cancer_type_subtypes_of, housekeeping_gene_ids, is_mixture_cohort, resolve_cancer_type,

)

from trufflepig.reference import (

    pan_cancer_expression, subtype_deconvolved_expression,

)
from .common import _build_sample_tpm_by_symbol as _common_build_sample_tpm
from .common import build_sample_tpm_by_gene_id as _build_sample_tpm_by_gene_id
from .format import render_fold
from .reference import estimate_signatures


# -------------------- cancer type → normal tissue mapping --------------------

CANCER_TO_TISSUE = {
    "ACC": "adrenal_gland",
    "BLCA": "urinary_bladder",
    "BRCA": "breast",
    "CESC": "cervix",
    "CHOL": "gallbladder",
    "COAD": "colon",
    "DLBC": "lymph_node",
    "ESCA": "esophagus",
    "GBM": "cerebral_cortex",
    "HNSC": "tongue",
    "KICH": "kidney",
    "KIRC": "kidney",
    "KIRP": "kidney",
    "LAML": "bone_marrow",
    "LGG": "cerebral_cortex",
    "LIHC": "liver",
    "LUAD": "lung",
    "LUSC": "lung",
    "MESO": "lung",
    "OV": "ovary",
    "PAAD": "pancreas",
    "PCPG": "adrenal_gland",
    "PRAD": "prostate",
    "READ": "rectum",
    "SARC": "smooth_muscle",
    "SKCM": "skin",
    "STAD": "stomach",
    "TGCT": "testis",
    "THCA": "thyroid_gland",
    "THYM": "thymus",
    "UCEC": "endometrium",
    "UCS": "endometrium",
    "UVM": "retina",
}

# Median TCGA tumor purity by cancer type (from Aran et al. 2015, Nat Commun;
# consensus purity estimates across ABSOLUTE, ESTIMATE, LUMP, IHC).
# Used to calibrate: TCGA reference ≈ this purity, not 100%.
TCGA_MEDIAN_PURITY = {
    "ACC": 0.79,
    "BLCA": 0.59,
    "BRCA": 0.73,
    "CESC": 0.49,
    "CHOL": 0.68,
    "COAD": 0.59,
    "DLBC": 0.94,
    "ESCA": 0.50,
    "GBM": 0.83,
    "HNSC": 0.60,
    "KICH": 0.84,
    "KIRC": 0.72,
    "KIRP": 0.78,
    "LAML": 0.95,
    "LGG": 0.87,
    "LIHC": 0.73,
    "LUAD": 0.56,
    "LUSC": 0.67,
    "MESO": 0.55,
    "OV": 0.72,
    "PAAD": 0.42,
    "PCPG": 0.69,
    "PRAD": 0.69,
    "READ": 0.60,
    "SARC": 0.66,
    "SKCM": 0.65,
    "STAD": 0.40,
    "TGCT": 0.75,
    "THCA": 0.72,
    "THYM": 0.78,
    "UCEC": 0.71,
    "UCS": 0.65,
    "UVM": 0.85,
}

_HOST_SITE_BACKGROUND_TISSUES = {
    "bone_marrow",
    "lymph_node",
    "spleen",
    "thymus",
    "tonsil",
    "appendix",
    "smooth_muscle",
    "skeletal_muscle",
    "heart_muscle",
    "adipose_tissue",
}

# Broad-family signature panels loaded from data/cancer-family-panels.csv.
# The CSV is the source of truth; this dict is a view built once on import.
#
# Note: trufflepig/family_extensions.py defines extension panels for
# cohorts the CSV does not yet cover (BRCA, BLCA, LUAD, OV, UCEC, etc.)
# but those are NOT wired in here — wiring them flat into this dict
# regresses family_specificity 17× across all rows (each new panel
# competes in the (top - second) / top formula). The correct path is
# a tiered scoring redesign (see docs/CANCER_CALL_DECISION_FLOW.md
# Stages 2-4 + pirlygenes #266). Until then, the extension data
# stays as design documentation, not active scoring input.
_CANCER_FAMILY_PANELS = cancer_family_panels()


def _build_family_panels_by_id():
    """``{family: [versionless Ensembl ID, ...]}`` from the curated panel table.

    The family panels are scored by **Ensembl gene ID**, not HGNC symbol, so the
    scoring is immune to symbol/alias drift between the panel's curation
    vocabulary (Ensembl 112) and trufflepig's reference symbols. Matching by
    symbol made a drifted alias (e.g. TDGF1/CRIPTO) silently resolve to
    ``.get(symbol, 0.0)`` — reading a real, expressed marker as "not expressed"
    and quietly weakening the panel. The panel CSV already carries clean
    unversioned ENSGs; this consumes them.
    """
    from .plot_data_helpers import _strip_ensembl_version

    out = {}
    try:
        df = cancer_family_panels_df()
        for family, grp in df.groupby("Family", sort=False):
            out[str(family)] = [
                _strip_ensembl_version(str(g))
                for g in grp["Ensembl_Gene_ID"].dropna()
                if str(g).strip() and str(g).strip().lower() != "nan"
            ]
    except Exception:
        # Fall back to an empty map; the scorer degrades to all-zero scores
        # rather than crashing if the panel table is unavailable.
        out = {}
    return out


_CANCER_FAMILY_PANELS_BY_ID = _build_family_panels_by_id()

_CANCER_FAMILY_BY_CODE = {
    "PRAD": "PROSTATE",
    "COAD": "CRC",
    "READ": "CRC",
    "STAD": "GASTRIC",
    "ESCA": "ESCA_SQ",
    "HNSC": "SQUAMOUS",
    "LUSC": "SQUAMOUS",
    "CESC": "SQUAMOUS",
    "SARC": "MESENCHYMAL",
    "UCS": "MESENCHYMAL",
    "KIRC": "RENAL",
    "KIRP": "RENAL",
    "KICH": "RENAL",
    "GBM": "GLIAL",
    "LGG": "GLIAL",
    "SKCM": "MELANOCYTIC",
    "UVM": "MELANOCYTIC",
}

_CANCER_FAMILY_CODE_COUNTS = Counter(_CANCER_FAMILY_BY_CODE.values())
_CANCER_FAMILY_GROUP = {
    "ESCA_SQ": "SQUAMOUS",
}
_CANCER_FAMILY_GROUP_CODE_COUNTS = Counter(
    _CANCER_FAMILY_GROUP.get(family, family)
    for family in _CANCER_FAMILY_BY_CODE.values()
)
_CANCER_FAMILY_DISPLAY = {
    "CRC": "CRC",
    "ESCA_SQ": "esophageal squamous",
    "GASTRIC": "gastric",
    "GLIAL": "glial",
    "MELANOCYTIC": "melanocytic",
    "MESENCHYMAL": "mesenchymal / sarcoma-like",
    "PROSTATE": "prostate",
    "RENAL": "renal",
    "SQUAMOUS": "squamous",
}

TUMOR_PURITY_PARAMETERS = {
    "lineage": {
        "missing_support_factor": 0.35,
        "detection_fraction_threshold": 0.05,
    },
    "tumor_specific_markers": {
        # Difference in HK-normalized expression a candidate must clear over
        # the matched-normal background to be eligible at all. Keeps the
        # filter meaningful for low-abundance but legitimately tumor-biased
        # genes.
        "delta_min": 0.02,
        # Strict tier: maximum share of the cancer signal that can come from
        # matched-normal tissue / broad host background, and minimum
        # cancer-vs-background fold change. A lower bound on z-score and
        # absolute HK-normalized expression in the cancer type itself.
        "normal_fraction_max": 0.5,
        "tme_fraction_max": 0.5,
        "cancer_expression_min": 0.5,
        "zscore_min": 1.0,
        "specificity_min": 1.5,
        # Loose tier: if the strict tier leaves a cancer type under-covered,
        # relax the absolute-expression and z-score floors. Still enforces
        # `delta_min` and some minimal specificity.
        "fallback_expression_min": 0.1,
        "fallback_zscore_min": 0.25,
        # Tuning knobs on the loose tier. Kept explicit (rather than
        # hard-coded in the tier construction) so the behavior is visible
        # without reading the filter code.
        "fallback_normal_fraction_max": 0.8,
        "fallback_tme_fraction_max": 0.8,
        "fallback_specificity_min": 1.1,
        # Gene-family exclusions, expressed as regex patterns matched with
        # `re.fullmatch` against gene symbols. The defaults drop:
        #   - rearranged-receptor V/D/J/C segments (IGHV3-33, TRGV9, ...)
        #     which are infiltrate-driven and sequence-unstable
        #   - HLA class II (HLA-D*) which are APC / infiltrate markers
        #   - RPL*/RPS* ribosomal protein genes
        #   - MT-* mitochondrial-encoded transcripts (unstable, degradation
        #     artifact territory)
        # HLA class I (HLA-A/B/C/E/F/G) and non-receptor IG/TR genes
        # (IGHMBP2, TRAF*, TRADD, TRAP1, TRAK1, ...) are deliberately not
        # excluded.
        "excluded_gene_regexes": [
            r"(IGH|IGK|IGL|TRA|TRB|TRG|TRD)[VDJC]\d.*",
            r"HLA-D[A-Z]+\d*",
            r"(RPL|RPS)\d.*",
            r"MT-.*",
        ],
        # Cancer types where the excluded families above ARE the legitimate
        # tumor signal. Covers lymphoid (DLBC → B-cell receptor), myeloid
        # APC-like (LAML — monocytic subtypes retain high HLA-DR / MHC-II;
        # HLA-DR is a standard AML immunophenotyping marker and only APL/M3
        # is reliably HLA-DR-negative), and thymic epithelial (THYM, which
        # also expresses class II). The rearranged-receptor and HLA-D
        # regexes above are bypassed for these codes so the panel captures
        # legitimate lineage markers instead of dropping them.
        "immune_origin_cancer_types": ["DLBC", "LAML", "THYM"],
    },
    "host_background": {
        "expression_min": 0.05,
        "zscore_min": 1.5,
        "specificity_min": 2.0,
        "top_genes": 20,
    },
    "purity_combination": {
        "signature_only_estimate_floor": 0.05,
        "tumor_anchor_weight": 0.7,
        "estimate_weight": 0.3,
        "signature_conflict_ratio": 0.75,
        "signature_stability_min": 0.45,
        "signature_weight_floor": 0.35,
    },
    "family_scoring": {
        "presence_scale": 0.15,
        "within_family_base": 0.35,
        "within_family_gain": 0.65,
        "non_family_penalty": 0.85,
        "min_factor": 0.05,
        "support_fraction_of_top_floor": 0.05,
        "signature_stability_floor": 0.2,
        "family_display_fraction": 0.4,
        "candidate_panel_min_score": 0.05,
        "candidate_panel_top_n": 2,
        "non_penalizing_families": ["MESENCHYMAL"],
        "soft_family_penalty_gain": 0.75,
        "orphan_context_min_signature": 0.55,
        "orphan_context_min_raw_ratio": 0.85,
        "orphan_context_dominant_raw_ratio": 1.30,
    },
}

_SIGNATURE_PANEL_CACHE = {}

# The derived reference matrices (Symbol-indexed frame, cohort TPM expression
# matrix, z-score matrix across cancer types) are identical for every
# cancer code given the same `normalize` parameter. Build them once per
# distinct normalization, not once per (code, normalize) pair — the 33×
# redundant recompute across cancer types was the single biggest source of
# wall-clock in panel scans.
_REFERENCE_MATRIX_CACHE = {}


def _cached_reference_matrices(normalize="housekeeping"):
    """Return cached derived frames for pan-cancer expression.

    Returns a dict with:
        ref_by_sym    — pan_cancer_expression() deduped on Symbol and
                        indexed by Symbol
        cohort_cols     — list of *_TPM column names
        expr_matrix   — ref_by_sym[cohort_cols] as float
        gene_mean     — per-gene mean across cancer types
        gene_std      — per-gene std (zero replaced with NaN for safe div)
        z_matrix      — full z-score matrix (columns: cohort_cols)

    Keyed on `normalize` alone because the per-cancer panel builder only
    needs one view of the reference data. If callers ever need raw
    (non-normalized) values they can pass `normalize=None`.
    """
    cached = _REFERENCE_MATRIX_CACHE.get(normalize)
    if cached is not None:
        return cached

    ref = pan_cancer_expression(normalize=normalize, technical_rna_normalize=True)
    ref_by_sym = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    cohort_cols = [c for c in ref.columns if c.endswith("_TPM")]
    expr_matrix = ref_by_sym[cohort_cols].astype(float)
    gene_mean = expr_matrix.mean(axis=1)
    gene_std = expr_matrix.std(axis=1).replace(0, np.nan)
    z_matrix = expr_matrix.sub(gene_mean, axis=0).div(gene_std, axis=0).fillna(0)

    entry = {
        "ref_by_sym": ref_by_sym,
        "cohort_cols": cohort_cols,
        "expr_matrix": expr_matrix,
        "gene_mean": gene_mean,
        "gene_std": gene_std,
        "z_matrix": z_matrix,
    }
    _REFERENCE_MATRIX_CACHE[normalize] = entry
    return entry


def _params_fingerprint(subkeys):
    """Stable, hashable snapshot of the selected TUMOR_PURITY_PARAMETERS keys.

    Used as part of a cache key so mutating parameters between calls
    invalidates previously cached panels instead of silently serving stale
    results (which would surprise anyone tuning parameters in a REPL or test).
    """
    snap = []
    for key in subkeys:
        value = TUMOR_PURITY_PARAMETERS.get(key, {})
        # Cheap canonicalization — sort top-level keys and coerce nested
        # lists to tuples so the result is hashable.
        flat = tuple(
            (k, tuple(v) if isinstance(v, list) else v)
            for k, v in sorted(value.items())
        )
        snap.append((key, flat))
    return tuple(snap)


def _compile_excluded_gene_matcher():
    """Return a predicate `symbol -> bool` from the configured regex list.

    Factored out so the compiled regex is cached per distinct pattern list
    and re-used across many gene lookups inside a single panel build.
    """
    import re

    params = TUMOR_PURITY_PARAMETERS["tumor_specific_markers"]
    patterns = tuple(params.get("excluded_gene_regexes", ()) or ())
    if not patterns:
        return lambda symbol: False
    joined = "|".join(f"(?:{p})" for p in patterns)
    regex = re.compile(f"^(?:{joined})$")

    def is_excluded(symbol):
        if not symbol:
            return True
        return bool(regex.fullmatch(str(symbol)))

    return is_excluded


def _is_excluded_signature_gene(symbol):
    """Compatibility wrapper — compiles the matcher per call.

    Kept so external callers (tests, plot.py fallback) can ask without
    needing to know about the compiled cache. For tight loops prefer
    `_compile_excluded_gene_matcher()` and call its returned predicate.
    """
    return _compile_excluded_gene_matcher()(symbol)


def get_tumor_purity_parameters():
    """Return the current tumor-purity and family-scoring free parameters."""
    return TUMOR_PURITY_PARAMETERS


_CANCER_NORMAL_TISSUES = {
    "COAD": ["colon", "rectum", "appendix", "small_intestine", "duodenum"],
    "READ": ["rectum", "colon", "appendix", "small_intestine", "duodenum"],
    "STAD": ["stomach", "duodenum", "esophagus", "gallbladder"],
    "ESCA": ["esophagus", "stomach"],
}


# -------------------- helpers --------------------


def _build_sample_tpm_by_symbol(df_gene_expr):
    """Return {symbol: max_TPM} from already-clean sample expression.

    Canonical implementation lives in ``common.build_sample_tpm_by_symbol``;
    this is a thin delegate kept for backward compatibility.
    """
    return _common_build_sample_tpm(df_gene_expr)


def _geneset_hk_ratio(genes, hk_symbols, expr_by_symbol):
    """Sum of gene set expression / sum of housekeeping expression.

    This ratio is scale-independent (cancels out the per-sample
    scaling factor since both come from the same sample/column).
    """
    gs_sum = sum(expr_by_symbol.get(g, 0) for g in genes)
    hk_sum = sum(expr_by_symbol.get(g, 0) for g in hk_symbols)
    if hk_sum <= 0:
        return 0.0
    return gs_sum / hk_sum


def _sample_hk_median(sample_tpm):
    """Return the sample housekeeping median on clean TPM scale."""
    ref = pan_cancer_expression(technical_rna_normalize=True)
    id_to_sym = dict(zip(ref["Ensembl_Gene_ID"], ref["Symbol"]))
    hk_syms = [id_to_sym[gid] for gid in housekeeping_gene_ids() if gid in id_to_sym]
    sample_hk_vals = [sample_tpm[g] for g in hk_syms if sample_tpm.get(g, 0) > 0]
    return float(np.median(sample_hk_vals)) if sample_hk_vals else 0.0


# Lineage genes per cancer type — genes retained in metastases and specific
# enough to calibrate purity. Loaded from data/lineage-genes.csv; only
# genes with low TME background and high expression in the origin tissue
# should be listed there. Keep the name `LINEAGE_GENES` for backward
# compatibility with external importers.
# Canonicalized via the shared accessor so the purity estimator and the
# tumor-type ontology speak one symbol vocabulary (alias-drift immune: the DLBC
# panel's ``CD20`` resolves to the reference's ``MS4A1``, the rituximab target,
# instead of silently missing the reference-vocabulary sample).
from .common import lineage_genes_by_cancer_type_canonical as _lineage_genes_canonical

LINEAGE_GENES = _lineage_genes_canonical()


_PURITY_PANEL_CODES_CACHE: list = []


def lineage_purity_panel_codes() -> frozenset:
    """Codes whose *own* lineage panel anchors their purity estimate.

    The purity estimator only uses a code's own lineage panel when the
    code does not fall back to a broad parent cohort (see
    :func:`_broad_purity_fallback_code`). Fusion-defined sarcoma subtypes
    (``SARC_ASPS``, ``SARC_SFT``, ``SARC_CCS``, ...) fall back to ``SARC``
    for purity, so their lineage entry is never consulted as a purity
    panel — it exists only for *subtype identification* via the
    ontology/recall path. pirlygenes' "subtype-marker" batch lists a
    single pathognomonic fusion/IHC gene for these (TFE3 for ASPS, STAT6
    for SFT, ...), which is correct as a diagnostic marker but is not a
    purity panel.

    The purity-panel size floors (#170 ≥5, #162/#167 ≥2) therefore apply
    only to the codes returned here — every code that actually drives its
    own purity estimate. (Invariant, asserted in the tests: no such code
    has fewer than 5 curated genes.)
    """
    if _PURITY_PANEL_CODES_CACHE:
        return _PURITY_PANEL_CODES_CACHE[0]
    result = frozenset(
        code
        for code in LINEAGE_GENES
        if not (_broad_purity_fallback_code(code) or "").strip()
        or _broad_purity_fallback_code(code) == code
    )
    _PURITY_PANEL_CODES_CACHE.append(result)
    return result


# #162: cross-cohort lineage-panel specificity. When a lineage panel
# gene is expressed at comparable levels in multiple TCGA cohorts
# (e.g. MUC5AC in both STAD and PAAD, KRT19 across all GI epithelia),
# running cohort X's panel on a sample from cohort Y inflates Y's
# lineage purity for X — on the cohort-median battery STAD's lineage
# computed to 0.913 on a PAAD median, which combined with the GASTRIC
# family factor flipped the classifier to STAD. Filter each cohort's
# panel to the genes whose *home* expression dominates the max non-
# home cohort expression by the specificity threshold.
_LINEAGE_SPECIFICITY_MIN = 0.5  # home / (home + max_other) ≥ 0.5
# The filter only fires when the competing cohort's expression is above
# this absolute floor — below it, the competitor isn't really "expressing"
# the gene at levels that would create crosstalk. Pinned to 5 TPM after
# the sarcoma→THYM regression (#167): the original 1 TPM threshold was
# dropping rare subtype markers like MYOD1 (TCGA_SARC median ≈ 0,
# TCGA_UCS median 1.5) even though neither cohort materially expresses
# the gene at median.
_LINEAGE_COMPETITOR_FLOOR = 5.0
_LINEAGE_MIN_GENES_AFTER_FILTER = 2  # don't prune a panel below this
_LINEAGE_SPECIFIC_CACHE: dict = {}


def _cancer_specific_lineage_genes(cancer_code: str) -> list:
    """Return the subset of the cancer type's lineage panel that is
    specific to its home cohort.

    A gene is considered specific when its TPM in ``cancer_code``'s
    cohort exceeds every other cohort's TPM such that
    ``home / (home + max_other) ≥ _LINEAGE_SPECIFICITY_MIN``. Genes
    with near-zero home expression (below
    ``_LINEAGE_SPECIFICITY_IGNORE_BELOW``) are kept regardless — those
    are ambient-level genes not driving cross-cohort crosstalk.

    The filter guarantees at least ``_LINEAGE_MIN_GENES_AFTER_FILTER``
    genes remain; when specificity-filtering would drop too many, the
    original panel is returned (the cohort's panel is too promiscuous
    to filter safely and the lineage estimator's own ``tme_ratio`` check
    remains the safety net).
    """
    if cancer_code in _LINEAGE_SPECIFIC_CACHE:
        return _LINEAGE_SPECIFIC_CACHE[cancer_code]
    panel = LINEAGE_GENES.get(cancer_code, [])
    if not panel:
        _LINEAGE_SPECIFIC_CACHE[cancer_code] = []
        return []
    ref = pan_cancer_expression(technical_rna_normalize=True).drop_duplicates(
        subset="Symbol"
    )
    ref = ref.set_index("Symbol")
    home_col = f"{cancer_code}_TPM"
    other_cols = [c for c in ref.columns if c.endswith("_TPM") and c != home_col]
    if home_col not in ref.columns or not other_cols:
        _LINEAGE_SPECIFIC_CACHE[cancer_code] = list(panel)
        return list(panel)
    # Compute per-gene specificity score; keep the ones above the
    # threshold. For genes below the threshold, retain the top-N most-
    # specific so a heavily-shared panel (STAD's 3-of-4 GI-epithelium
    # markers) still yields enough signal for the purity estimator.
    scored = []
    specific = []
    for gene in panel:
        if gene not in ref.index:
            continue
        home_val = float(ref.loc[gene, home_col] or 0.0)
        other_vals = ref.loc[gene, other_cols].astype(float)
        max_other = float(other_vals.max())
        # Competitor below the floor → no crosstalk risk; keep the
        # gene. This preserves rare / subtype-specific markers whose
        # cohort median is low in every cohort (MYOD1 / MYOG in SARC
        # are the canonical case — only rhabdomyosarcoma subtypes
        # express them, so the pan-cohort median is near zero).
        if max_other < _LINEAGE_COMPETITOR_FLOOR:
            specific.append(gene)
            scored.append((gene, 1.0))
            continue
        denom = home_val + max_other
        if denom <= 0:
            scored.append((gene, 0.0))
            continue
        specificity = home_val / denom
        scored.append((gene, specificity))
        if specificity >= _LINEAGE_SPECIFICITY_MIN:
            specific.append(gene)
    if len(specific) < _LINEAGE_MIN_GENES_AFTER_FILTER and scored:
        # Not enough specific genes to anchor a purity estimate. Fill
        # in by keeping the top-N most-specific genes from the full
        # panel, even if they fall below the threshold — safer than
        # reverting to the full panel (which defeats the filter).
        scored_sorted = sorted(scored, key=lambda pair: pair[1], reverse=True)
        fill = [g for g, _ in scored_sorted if g not in specific]
        for g in fill:
            if len(specific) >= _LINEAGE_MIN_GENES_AFTER_FILTER:
                break
            specific.append(g)
    _LINEAGE_SPECIFIC_CACHE[cancer_code] = specific
    return specific


def _lineage_purity_estimates(
    cancer_code, sample_tpm, ref_by_sym, hk_syms, tcga_purity
):
    """Estimate purity from cancer-type lineage genes using HK-normalized ratios.

    For each lineage gene, computes:
        sample_ratio = gene_sample / HK_sample
        ref_ratio    = gene_TCGA  / HK_TCGA
        tme_ratio    = median(gene_tissue / HK_tissue) across TME tissues
        true_tumor_ratio = (ref_ratio - (1-tcga_purity) * tme_ratio) / tcga_purity
        purity = (sample_ratio - tme_ratio) / (true_tumor_ratio - tme_ratio)

    Returns list of dicts with per-gene purity estimates.
    """
    # #162: filter the panel to genes that are specific to this cohort
    # relative to other TCGA cohorts — avoids the STAD/PAAD-style
    # crosstalk where shared GI-epithelium markers inflate the wrong
    # cancer type's lineage purity.
    genes = _cancer_specific_lineage_genes(cancer_code)
    if not genes:
        return [], []

    ref = pan_cancer_expression(technical_rna_normalize=True)
    ref_dedup = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    ntpm_cols = [c for c in ref.columns if c.endswith("_nTPM")]
    _repro = {"testis", "epididymis", "seminal_vesicle", "placenta", "ovary"}
    ntpm_nonrepro = [c for c in ntpm_cols if c.removesuffix("_nTPM") not in _repro]

    # Curated TME tissues (immune organs + stromal/connective)
    _tme = {
        "bone_marrow",
        "lymph_node",
        "spleen",
        "thymus",
        "tonsil",
        "appendix",
        "smooth_muscle",
        "skeletal_muscle",
        "heart_muscle",
        "adipose_tissue",
    }
    tme_cols = [c for c in ntpm_nonrepro if c.removesuffix("_nTPM") in _tme]

    # HK symbols in reference
    hk_in_ref = [s for s in hk_syms if s in ref_dedup.index]

    # HK medians per column
    cancer_col = f"{cancer_code}_TPM"
    if cancer_col not in ref_dedup.columns:
        return [], []
    ref_hk_cancer = ref_dedup.loc[hk_in_ref, cancer_col].astype(float).median()
    if ref_hk_cancer <= 0:
        return [], []

    tme_hk_medians = {}
    for col in tme_cols:
        tme_hk_medians[col] = ref_dedup.loc[hk_in_ref, col].astype(float).median()

    # Sample HK (median of expressed HK genes)
    sample_hk_vals = [sample_tpm[g] for g in hk_syms if sample_tpm.get(g, 0) > 0]
    sample_hk_med = float(np.median(sample_hk_vals)) if sample_hk_vals else 0.0
    if sample_hk_med <= 0:
        return [], []

    results = []
    # Parallel list of lineage genes that were genuinely present in the
    # sample but dropped by the estimator (TME signal exceeds tumor
    # signal in the reference, so the gene can't anchor a purity
    # estimate). Callers can surface these as "uninformative" rather
    # than "not detected" — the distinction matters for sarcoma-style
    # cases where ACTA2 is at ~190 TPM but gets filtered because its
    # TME-bleed-through in SARC exceeds its tumor contribution.
    skipped_detected = []
    for gene in genes:
        if gene not in ref_dedup.index:
            continue
        s_tpm = sample_tpm.get(gene, 0)
        if s_tpm <= 0:
            continue

        sample_ratio = s_tpm / sample_hk_med
        ref_ratio = float(ref_dedup.loc[gene, cancer_col]) / ref_hk_cancer

        # TME ratio: median across TME tissues (HK-normalized)
        tme_ratios = []
        for col in tme_cols:
            hk_m = tme_hk_medians[col]
            if hk_m > 0:
                tme_ratios.append(float(ref_dedup.loc[gene, col]) / hk_m)
        tme_ratio = float(np.median(tme_ratios)) if tme_ratios else 0.0

        # Deconvolve TCGA to get true tumor ratio
        true_tumor_ratio = (ref_ratio - (1 - tcga_purity) * tme_ratio) / tcga_purity

        if true_tumor_ratio <= tme_ratio:
            skipped_detected.append(
                {
                    "gene": gene,
                    "sample_tpm": s_tpm,
                    "reason": "tme_dominated",
                    "tme_ratio": float(tme_ratio),
                    "tumor_ratio": float(true_tumor_ratio),
                }
            )
            continue

        purity = (sample_ratio - tme_ratio) / (true_tumor_ratio - tme_ratio)
        purity = float(np.clip(purity, 0, 1))

        results.append(
            {
                "gene": gene,
                "sample_tpm": s_tpm,
                "sample_ratio": float(sample_ratio),
                "ref_ratio": float(ref_ratio),
                "tme_ratio": float(tme_ratio),
                "tumor_ratio": float(true_tumor_ratio),
                "purity": purity,
            }
        )

    return results, skipped_detected


def _summarize_lineage_support(lineage_per_gene):
    """Summarize whether the observed lineage pattern matches the candidate tumor.

    A single shared marker can produce a misleadingly high lineage purity. We
    therefore score the *pattern* of lineage genes, not just their median
    purity, using a weighted cosine similarity between the observed lineage
    excess and the candidate's expected tumor lineage profile.
    """
    if not lineage_per_gene:
        return {
            "concordance": None,
            "detection_fraction": 0.0,
            "support_factor": TUMOR_PURITY_PARAMETERS["lineage"][
                "missing_support_factor"
            ],
        }

    sample_excess = np.array(
        [max(0.0, row["sample_ratio"] - row["tme_ratio"]) for row in lineage_per_gene],
        dtype=float,
    )
    tumor_excess = np.array(
        [max(0.0, row["tumor_ratio"] - row["tme_ratio"]) for row in lineage_per_gene],
        dtype=float,
    )
    weights = np.sqrt(np.maximum(tumor_excess, 1e-6))

    sample_weighted = sample_excess * weights
    tumor_weighted = tumor_excess * weights
    denom = float(np.linalg.norm(sample_weighted) * np.linalg.norm(tumor_weighted))
    if denom > 0:
        concordance = float(
            np.clip(sample_weighted.dot(tumor_weighted) / denom, 0.0, 1.0)
        )
    else:
        concordance = 0.0

    detected = sample_excess >= (
        TUMOR_PURITY_PARAMETERS["lineage"]["detection_fraction_threshold"]
        * np.maximum(tumor_excess, 1e-6)
    )
    detection_fraction = float(np.mean(detected))

    # Pattern match matters more than raw detection count. A candidate with a
    # few expressed genes but the wrong overall shape should be penalized hard.
    support_factor = float(np.sqrt(concordance) * (0.5 + 0.5 * detection_fraction))

    return {
        "concordance": concordance,
        "detection_fraction": detection_fraction,
        "support_factor": support_factor,
    }


def _subtype_tumor_tpm_lookup(subtype_code):
    """Return dict of {symbol: tumor_tpm_median} for a subtype.

    Source: ``subtype-deconvolved-expression.csv.gz`` (tumor-only TPM
    per (cancer_code, subtype)). Used by mixture-cohort lineage
    estimation (#171) — the parent-cohort median (``SARC_TPM``) is a
    diluted mixture; the subtype median is a clean per-subtype tumor
    profile.
    """
    sub_df = subtype_deconvolved_expression(technical_rna_normalize=True)
    if sub_df is None:
        return {}
    matched = sub_df[sub_df["cancer_code"] == subtype_code]
    if matched.empty:
        return {}
    return dict(zip(matched["symbol"], matched["tumor_tpm_median"].astype(float)))


def _subtype_lineage_purity_estimates(
    subtype_code,
    panel,
    sample_tpm,
    hk_syms,
    ref_by_sym=None,
):
    """Per-gene purity estimates for a mixture-cohort subtype (#171).

    Like :func:`_lineage_purity_estimates` but uses the subtype's
    tumor-only median TPM as the reference (no TCGA purity deconv
    needed — the source is already tumor-pure). This lets rare markers
    like MYOD1 (near-zero at TCGA-SARC median, but ~65 TPM at RMS_ERMS
    tumor median) actually anchor a purity estimate.

    Returns ``(results, skipped_detected)`` matching the parent helper
    contract so :func:`_summarize_lineage_support` consumes either.
    """
    subtype_tpm = _subtype_tumor_tpm_lookup(subtype_code)
    if not subtype_tpm:
        return [], []

    ref = pan_cancer_expression(technical_rna_normalize=True)
    ref_dedup = ref.drop_duplicates(subset="Symbol").set_index("Symbol")

    # Same TME tissues the parent helper uses — keeps the TME-ratio
    # computation consistent across the mixture / non-mixture paths.
    _tme = {
        "bone_marrow",
        "lymph_node",
        "spleen",
        "thymus",
        "tonsil",
        "appendix",
        "smooth_muscle",
        "skeletal_muscle",
        "heart_muscle",
        "adipose_tissue",
    }
    ntpm_cols = [c for c in ref.columns if c.endswith("_nTPM")]
    tme_cols = [c for c in ntpm_cols if c.removesuffix("_nTPM") in _tme]
    hk_in_ref = [s for s in hk_syms if s in ref_dedup.index]

    tme_hk_medians = {
        col: ref_dedup.loc[hk_in_ref, col].astype(float).median() for col in tme_cols
    }

    sample_hk_vals = [sample_tpm[g] for g in hk_syms if sample_tpm.get(g, 0) > 0]
    sample_hk_med = float(np.median(sample_hk_vals)) if sample_hk_vals else 0.0
    if sample_hk_med <= 0:
        return [], []

    # The subtype tumor-TPM table has no HK row per se; use the HPA
    # median of HK genes as the normalizer that converts subtype tumor
    # TPM into an HK-relative "tumor ratio". This follows the same
    # normalization spirit as the parent-cohort helper but without a
    # cohort-specific TPM column.
    ref_hk_median_cols = [
        c
        for c in ntpm_cols
        if c.removesuffix("_nTPM") not in _tme
        and c.removesuffix("_nTPM")
        not in {"testis", "epididymis", "seminal_vesicle", "placenta", "ovary"}
    ]
    if not ref_hk_median_cols:
        return [], []
    ref_hk_normalizer = float(
        ref_dedup.loc[hk_in_ref, ref_hk_median_cols].astype(float).median().median()
    )
    if ref_hk_normalizer <= 0:
        return [], []

    results = []
    skipped_detected = []
    for gene in panel:
        s_tpm = sample_tpm.get(gene, 0)
        if s_tpm <= 0:
            continue
        tumor_tpm = subtype_tpm.get(gene)
        if tumor_tpm is None or tumor_tpm <= 0:
            continue
        if gene not in ref_dedup.index:
            continue

        sample_ratio = s_tpm / sample_hk_med
        tumor_ratio = float(tumor_tpm) / ref_hk_normalizer

        tme_ratios = []
        for col in tme_cols:
            hk_m = tme_hk_medians[col]
            if hk_m > 0:
                tme_ratios.append(float(ref_dedup.loc[gene, col]) / hk_m)
        tme_ratio = float(np.median(tme_ratios)) if tme_ratios else 0.0

        if tumor_ratio <= tme_ratio:
            skipped_detected.append(
                {
                    "gene": gene,
                    "sample_tpm": s_tpm,
                    "reason": "tme_dominated",
                    "tme_ratio": float(tme_ratio),
                    "tumor_ratio": float(tumor_ratio),
                }
            )
            continue

        purity = (sample_ratio - tme_ratio) / (tumor_ratio - tme_ratio)
        purity = float(np.clip(purity, 0, 1))
        results.append(
            {
                "gene": gene,
                "sample_tpm": s_tpm,
                "sample_ratio": float(sample_ratio),
                "ref_ratio": float(tumor_ratio),  # subtype is already tumor-pure
                "tme_ratio": float(tme_ratio),
                "tumor_ratio": float(tumor_ratio),
                "purity": purity,
            }
        )

    return results, skipped_detected


def _mixture_cohort_lineage_summary(parent_code, sample_tpm, hk_syms):
    """Evaluate lineage per subtype for a mixture parent; return max (#171).

    For each subtype of ``parent_code`` that has both (a) a curated
    lineage panel in ``lineage-genes.csv`` and (b) tumor-only
    expression medians in ``subtype-deconvolved-expression``, compute
    the lineage support. Pick the subtype with the best concordance
    (ties broken by detection_fraction).

    Returns ``None`` when no subtype qualifies — callers should fall
    back to the parent-level lineage computation.
    """
    subtypes = cancer_type_subtypes_of(parent_code)
    if not subtypes:
        return None

    best = None
    per_subtype = []
    for subtype_code in subtypes:
        panel = LINEAGE_GENES.get(subtype_code, [])
        if not panel:
            continue
        per_gene, skipped = _subtype_lineage_purity_estimates(
            subtype_code,
            panel,
            sample_tpm,
            hk_syms,
        )
        if not per_gene:
            per_subtype.append(
                {
                    "code": subtype_code,
                    "concordance": None,
                    "detection_fraction": 0.0,
                    "lineage_per_gene": [],
                    "skipped": skipped,
                }
            )
            continue
        support = _summarize_lineage_support(per_gene)
        record = {
            "code": subtype_code,
            "concordance": support["concordance"],
            "detection_fraction": support["detection_fraction"],
            "support_factor": support["support_factor"],
            "lineage_per_gene": per_gene,
            "skipped": skipped,
        }
        per_subtype.append(record)
        # Rank: (concordance, detection_fraction, panel-size tiebreak).
        concordance = record["concordance"] or 0.0
        detection = record["detection_fraction"] or 0.0
        score = (concordance, detection, len(per_gene))
        if best is None or score > best["_score"]:
            best = {**record, "_score": score}

    if best is None:
        return None
    best.pop("_score", None)
    best["per_subtype"] = per_subtype
    return best


def _select_tumor_specific_genes(cancer_code, n=30):
    """Select genes highly expressed in cancer but NOT in matched normal tissue.

    Returns list of gene symbols sorted by tumor specificity.
    """
    return _select_tumor_specific_genes_for_panel(
        cancer_code,
        n=n,
        exclude_lineage=True,
    )


_REPRODUCTIVE_TISSUES_FOR_BROAD_BACKGROUND = (
    # Reproductive tissues are excluded from the "broad normal" background
    # because they express a lot of otherwise cancer-specific markers (CT
    # antigens, lineage programs) that we do want to use as tumor signals.
    "testis",
    "epididymis",
    "seminal_vesicle",
    "placenta",
    "ovary",
)


def _build_signature_panel_tiers():
    """Return the ordered list of selectivity tiers for panel building.

    Strict tier first, then a loose fallback tier, then an open tier that
    still keeps `delta_min` and a minimal specificity floor. Each tier is
    described by the maximum tolerated share from normal / TME backgrounds
    and minimum cancer-side expression and specificity.

    Kept as data rather than nested constants so the shape is visible next
    to the filter code that consumes it.
    """
    params = TUMOR_PURITY_PARAMETERS["tumor_specific_markers"]
    return [
        {
            "name": "strict",
            "expr_min": params["cancer_expression_min"],
            "zscore_min": params["zscore_min"],
            "normal_frac_max": params["normal_fraction_max"],
            "tme_frac_max": params["tme_fraction_max"],
            "specificity_min": params["specificity_min"],
        },
        {
            "name": "loose",
            "expr_min": max(
                params["fallback_expression_min"],
                params["cancer_expression_min"] * 0.5,
            ),
            "zscore_min": max(0.0, params["fallback_zscore_min"]),
            "normal_frac_max": params["fallback_normal_fraction_max"],
            "tme_frac_max": params["fallback_tme_fraction_max"],
            "specificity_min": params["fallback_specificity_min"],
        },
        {
            "name": "open",
            "expr_min": params["fallback_expression_min"],
            "zscore_min": 0.0,
            "normal_frac_max": 1.0,
            "tme_frac_max": 1.0,
            "specificity_min": 1.0,
        },
    ]


def _panel_reference_frames(cancer_code, ref_by_sym=None):
    """Assemble per-gene expression vectors used by the panel filter.

    Returns a dict with the cancer-side HK-normalized expression, the
    maximum across matched-normal and broad-normal backgrounds (union), the
    max across TME tissues, the combined background, and a z-score over
    cancer types. Returns None if the cancer code has no TPM column.

    `ref_by_sym` is accepted for backward compatibility but ignored — the
    cached reference matrices (`_cached_reference_matrices`) are used so
    the expensive z-score matrix is computed once per normalization, not
    once per cancer code.
    """
    del ref_by_sym  # unused; kept in the signature for call-site compatibility

    cached = _cached_reference_matrices(normalize="housekeeping")
    ref_by_sym = cached["ref_by_sym"]
    expr_matrix = cached["expr_matrix"]
    z_matrix = cached["z_matrix"]

    cancer_col = f"{cancer_code}_TPM"
    if cancer_col not in expr_matrix.columns:
        return None

    ntpm_cols = [c for c in ref_by_sym.columns if c.endswith("_nTPM")]
    ntpm_nonrepro = [
        c
        for c in ntpm_cols
        if c.removesuffix("_nTPM") not in _REPRODUCTIVE_TISSUES_FOR_BROAD_BACKGROUND
    ]

    matched_tissues = list(_CANCER_NORMAL_TISSUES.get(cancer_code, []))
    tissue = CANCER_TO_TISSUE.get(cancer_code)
    if tissue and tissue not in matched_tissues:
        matched_tissues.append(tissue)
    normal_cols = [
        f"{t}_nTPM"
        for t in sorted(set(matched_tissues))
        if f"{t}_nTPM" in ref_by_sym.columns
    ]
    tme_cols = [
        f"{t}_nTPM"
        for t in sorted(_HOST_SITE_BACKGROUND_TISSUES)
        if f"{t}_nTPM" in ref_by_sym.columns
    ]

    z_scores = z_matrix[cancer_col]
    cancer_hk = expr_matrix[cancer_col]

    def _row_max(cols):
        if not cols:
            return pd.Series(0.0, index=ref_by_sym.index, dtype=float)
        return ref_by_sym[cols].astype(float).max(axis=1)

    matched_normal_hk = _row_max(normal_cols)
    broad_normal_hk = _row_max(ntpm_nonrepro)
    normal_hk = pd.concat(
        [matched_normal_hk.rename("matched"), broad_normal_hk.rename("broad")],
        axis=1,
    ).max(axis=1)
    tme_hk = _row_max(tme_cols)
    background_hk = pd.concat(
        [normal_hk.rename("normal"), tme_hk.rename("tme")], axis=1
    ).max(axis=1)

    return {
        "ref_by_sym": ref_by_sym,
        "z_scores": z_scores,
        "cancer_hk": cancer_hk,
        "normal_hk": normal_hk,
        "tme_hk": tme_hk,
        "background_hk": background_hk,
    }


def _select_tumor_specific_genes_for_panel(cancer_code, n=30, exclude_lineage=True):
    """Select robust cancer-signature genes for purity and subtype panels.

    The panel builder is deliberately conservative:
    - drop rearranged-receptor V/D/J/C segments and MHC class II by default
      (configurable, bypassed for hematopoietic / immune-origin cancers
      where those genes are legitimate lineage markers)
    - require meaningful HK-normalized expression in the target cancer type
    - score genes by cancer-type specificity AND visibility above the union
      of matched-normal, broad-normal (minus reproductive tissues), and TME
    - relax thresholds across three tiers only if the strict tier leaves a
      type under-covered

    The cache key includes a fingerprint of TUMOR_PURITY_PARAMETERS so
    tuning parameters in-process invalidates stale panels.
    """
    params_fp = _params_fingerprint(["tumor_specific_markers"])
    cache_key = (cancer_code, int(n), bool(exclude_lineage), params_fp)
    cached = _SIGNATURE_PANEL_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    frames = _panel_reference_frames(cancer_code)
    if frames is None:
        _SIGNATURE_PANEL_CACHE[cache_key] = tuple()
        return []
    ref_by_sym = frames["ref_by_sym"]
    z_scores = frames["z_scores"]
    cancer_hk = frames["cancer_hk"]
    normal_hk = frames["normal_hk"]
    tme_hk = frames["tme_hk"]
    background_hk = frames["background_hk"]

    params = TUMOR_PURITY_PARAMETERS["tumor_specific_markers"]

    # Rank candidates by a compound specificity score: positive z-score,
    # absolute expression, and cancer-vs-background fold.
    score = (
        z_scores.clip(lower=0.0)
        * np.log2(cancer_hk + 1.0)
        * np.log2((cancer_hk + 0.01) / (background_hk + 0.01) + 1.0)
    )
    normal_frac = normal_hk / (cancer_hk + 0.001)
    tme_frac = tme_hk / (cancer_hk + 0.001)
    specificity = (cancer_hk + 0.001) / (background_hk + 0.001)

    # Build the exclusion mask. Immune-origin cancer types (DLBC, LAML, THYM)
    # bypass the family exclusion entirely — for those, rearranged-receptor
    # loci and HLA-D genes ARE the lineage signal we want to capture, not
    # infiltrate contamination to filter out.
    immune_origin = set(params.get("immune_origin_cancer_types", []) or [])
    if cancer_code in immune_origin:
        excluded = pd.Series(False, index=ref_by_sym.index)
    else:
        is_excluded = _compile_excluded_gene_matcher()
        excluded = ref_by_sym.index.to_series().map(is_excluded).astype(bool)

    if exclude_lineage:
        lineage_genes = set(LINEAGE_GENES.get(cancer_code, []))
        if lineage_genes:
            excluded = excluded | ref_by_sym.index.to_series().isin(lineage_genes)

    markers = []
    seen = set()
    for tier in _build_signature_panel_tiers():
        keep = (
            (cancer_hk > tier["expr_min"])
            & (z_scores > tier["zscore_min"])
            & ((cancer_hk - normal_hk) > params["delta_min"])
            & (normal_frac <= tier["normal_frac_max"])
            & (tme_frac <= tier["tme_frac_max"])
            & (specificity >= tier["specificity_min"])
            & ~excluded
        )
        candidates = score[keep].sort_values(ascending=False)
        for gene in candidates.index:
            if gene in seen:
                continue
            seen.add(gene)
            markers.append(gene)
            if len(markers) >= n:
                _SIGNATURE_PANEL_CACHE[cache_key] = tuple(markers[:n])
                return markers[:n]

    # Last-resort fallback: still apply the family exclusion so the panel
    # never quietly regresses to including MT-* / rearranged receptors just
    # because the tiered filter ran short.
    fallback = score[
        (cancer_hk > params["fallback_expression_min"]) & ~excluded
    ].sort_values(ascending=False)
    for gene in fallback.index:
        if gene in seen:
            continue
        seen.add(gene)
        markers.append(gene)
        if len(markers) >= n:
            break

    _SIGNATURE_PANEL_CACHE[cache_key] = tuple(markers[:n])
    return markers


def _summarize_gene_level_purity(per_gene_purities, strategy="winsorized_median"):
    """Summarize per-gene purity estimates robustly.

    Signature genes can contain amplified or noisy outliers, while lineage
    genes can contain low outliers from de-differentiation. The summary should
    reflect the stable center of the distribution rather than a few extremes.
    """
    vals = np.array(
        sorted(float(p) for p in per_gene_purities if p is not None and p > 0),
        dtype=float,
    )
    if len(vals) == 0:
        return None, None, None, None

    lower = float(np.percentile(vals, 25))
    upper = float(np.percentile(vals, 75))
    if strategy == "upper_half":
        core = vals[len(vals) // 2 :] if len(vals) >= 3 else vals
    elif strategy == "winsorized_median" and len(vals) >= 4:
        core = np.clip(vals, lower, upper)
    else:
        core = vals

    overall = float(np.median(core))
    stability = float(np.clip((lower + 0.02) / (upper + 0.02), 0.0, 1.0))
    return overall, lower, upper, stability


def _combine_purity_estimates(
    sig_purity,
    sig_lower,
    sig_upper,
    estimate_purity,
    lineage_purity,
    lineage_lower,
    lineage_upper,
    sig_stability=None,
):
    """Combine purity signals while keeping ESTIMATE as context, not destiny.

    The ESTIMATE-derived purity is useful as an infiltration warning, but in
    highly inflamed metastases it often collapses to ~0 and should not erase a
    coherent tumor/lineage signal. When lineage support exists, combine it with
    the tumor-specific signature directly; otherwise fall back to the available
    evidence.
    """
    has_sig = sig_purity is not None
    has_lineage = lineage_purity is not None
    signature_params = TUMOR_PURITY_PARAMETERS["purity_combination"]
    deprioritize_signature = _signature_conflicts_with_lineage(
        sig_purity=sig_purity,
        lineage_purity=lineage_purity,
        sig_stability=sig_stability,
    )

    if has_sig and has_lineage:
        if deprioritize_signature:
            tumor_anchor = float(lineage_purity)
        else:
            # Weight the signature channel by its own stability. Use
            # `is not None` rather than truthiness so a stability of exactly
            # 0.0 is not promoted back to full weight — the earlier
            # conflict check already gates truly-degenerate cases, but this
            # branch still has to handle the "low but present stability"
            # case coherently.
            raw_sig_weight = float(sig_stability) if sig_stability is not None else 1.0
            sig_weight = max(raw_sig_weight, signature_params["signature_weight_floor"])
            lineage_weight = 1.0
            tumor_anchor = float(
                np.exp(
                    (
                        sig_weight * np.log(max(sig_purity, 1e-6))
                        + lineage_weight * np.log(max(lineage_purity, 1e-6))
                    )
                    / (sig_weight + lineage_weight)
                )
            )
    elif has_lineage:
        tumor_anchor = float(lineage_purity)
    elif has_sig:
        tumor_anchor = float(sig_purity)
    else:
        tumor_anchor = None

    if tumor_anchor is not None and estimate_purity is not None:
        # Tumor-positive and infiltration-negative signals should both matter,
        # but the tumor-specific anchor gets slightly more weight because
        # ESTIMATE can undercall purity in inflamed metastases.
        if has_sig and has_lineage and estimate_purity <= 0:
            overall = float(tumor_anchor)
        elif has_sig and not has_lineage:
            estimate_floor = signature_params["signature_only_estimate_floor"]
            overall = float(
                np.sqrt(max(tumor_anchor, 0.0) * max(estimate_purity, estimate_floor))
            )
        else:
            estimate_floor = signature_params["signature_only_estimate_floor"]
            overall = float(
                (max(tumor_anchor, 0.0) ** signature_params["tumor_anchor_weight"])
                * (
                    max(estimate_purity, estimate_floor)
                    ** signature_params["estimate_weight"]
                )
            )
    elif tumor_anchor is not None:
        overall = float(tumor_anchor)
    elif estimate_purity is not None:
        overall = float(estimate_purity)
    else:
        return None, None, None

    lower_candidates = [overall]
    upper_candidates = [overall]

    for value in (lineage_lower,):
        if value is not None:
            lower_candidates.append(float(value))
    if not deprioritize_signature:
        for value in (sig_lower,):
            if value is not None:
                lower_candidates.append(float(value))
    for value in (lineage_upper,):
        if value is not None:
            upper_candidates.append(float(value))
    if not deprioritize_signature:
        for value in (sig_upper,):
            if value is not None:
                upper_candidates.append(float(value))

    if estimate_purity is not None and (
        estimate_purity > 0 or (has_sig and not has_lineage)
    ):
        lower_candidates.append(float(estimate_purity))

    overall_lower = float(np.clip(min(lower_candidates), 0.0, 1.0))
    overall_upper = float(np.clip(max(upper_candidates), 0.0, 1.0))
    overall = float(np.clip(overall, overall_lower, overall_upper))

    # #101: lineage / signature upper quartiles are themselves depressed in
    # low-purity samples, so `max(upper_candidates)` collapses onto the point
    # estimate and the reported CI becomes one-sided low (e.g. 7–16 around 16).
    # Mirror the lower half onto the upper half so the upper bound is at least
    # as wide as the lower, reflecting the observed spread rather than the
    # floor of the sample itself. The lower bound is left alone.
    lower_span = overall - overall_lower
    upper_span = overall_upper - overall
    if upper_span < lower_span:
        overall_upper = float(np.clip(overall + lower_span, 0.0, 1.0))
    return overall, overall_lower, overall_upper


def _signature_conflicts_with_lineage(sig_purity, lineage_purity, sig_stability):
    """Return True when a weak signature should not drag down a coherent lineage call."""
    if sig_purity is None or lineage_purity is None:
        return False
    params = TUMOR_PURITY_PARAMETERS["purity_combination"]
    stability = float(sig_stability if sig_stability is not None else 1.0)
    return (
        float(sig_purity) < float(lineage_purity) * params["signature_conflict_ratio"]
        and stability < params["signature_stability_min"]
    )


def _registry_parent_code(code):
    try:
        from trufflepig.analyze.cancer_type_context import registry_parent_code

        return registry_parent_code(code)
    except Exception:
        return ""


_BROAD_PURITY_REFERENCE_FALLBACKS = {
    # TARGET osteosarcoma has tumor-expression priors, but no direct purity
    # marker panel yet. The CLI already treats OS as a refined report label
    # over the broad sarcoma reference when purity/decomposition need a broad
    # cohort; keep the direct Python API aligned with that behavior.
    "SARC_OS": "SARC",
}
_BULK_PAN_CANCER_PURITY_REFERENCE_SOURCES = frozenset(
    {"pan_cancer", "parent_pan_cancer"}
)


def _broad_purity_fallback_code(code):
    parent = _registry_parent_code(code)
    if parent:
        return parent
    return _BROAD_PURITY_REFERENCE_FALLBACKS.get(str(code or "").strip().upper(), "")


def _has_direct_purity_markers(cancer_code):
    """Whether ``cancer_code`` has marker panels usable for direct purity."""
    return bool(
        LINEAGE_GENES.get(cancer_code)
        or _select_tumor_specific_genes(cancer_code, n=1)
    )


def _use_estimate_component(reference_expression_source, stromal_genes) -> bool:
    """Whether ESTIMATE is compatible with the active expression reference."""
    return bool(stromal_genes) and (
        reference_expression_source in _BULK_PAN_CANCER_PURITY_REFERENCE_SOURCES
    )


def _resolve_purity_reference(cancer_code, ref_by_sym):
    """Return the expression reference used for purity estimation.

    Local deconvolved references can provide tumor-only expression for rare or
    refined labels, but they are not sufficient by themselves: the purity model
    also needs tumor-specific or lineage markers for that same label. When a
    refined label has expression but no purity markers yet, fall back to its
    registry parent cohort and keep the requested label in the returned result.
    """
    cancer_col = f"{cancer_code}_TPM"
    if cancer_col in ref_by_sym.columns:
        return {
            "reference_cancer_code": cancer_code,
            "reference_expression_source": "pan_cancer",
            "reference_purity": TCGA_MEDIAN_PURITY.get(cancer_code, 0.7),
            "ref_expr": ref_by_sym[cancer_col].to_dict(),
        }

    local_ref_expr = _subtype_tumor_tpm_lookup(cancer_code)
    if local_ref_expr and _has_direct_purity_markers(cancer_code):
        return {
            "reference_cancer_code": cancer_code,
            "reference_expression_source": "subtype_deconvolved",
            "reference_purity": 1.0,
            "ref_expr": local_ref_expr,
        }

    parent_code = _broad_purity_fallback_code(cancer_code)
    parent_col = f"{parent_code}_TPM" if parent_code else ""
    if parent_col and parent_col in ref_by_sym.columns:
        return {
            "reference_cancer_code": parent_code,
            "reference_expression_source": "parent_pan_cancer",
            "reference_purity": TCGA_MEDIAN_PURITY.get(parent_code, 0.7),
            "ref_expr": ref_by_sym[parent_col].to_dict(),
        }

    if local_ref_expr:
        raise ValueError(
            f"Cancer type {cancer_code!r} has a local deconvolved expression "
            "reference but no direct purity marker panel or broad purity "
            "fallback. Pass a broad reference cancer type for purity."
        )

    raise ValueError(
        f"Cancer type {cancer_code!r} has no pan-cancer TPM column "
        "and no local deconvolved expression reference"
    )


# -------------------- main estimation --------------------


def estimate_tumor_purity(df_gene_expr, cancer_type=None):
    """Estimate tumor purity from expression data.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Expression data with gene ID, gene name, and TPM columns.
    cancer_type : str or None
        TCGA cancer type code or alias. If None, auto-detected.

    Returns
    -------
    dict
        cancer_type : str — TCGA code
        overall_estimate : float — purity estimate (0–1)
        overall_lower : float — lower bound
        overall_upper : float — upper bound
        components : dict — per-component details
    """
    # Auto-detect cancer type
    if cancer_type is None:
        from .plot_embedding import _compute_cancer_type_signature_stats

        stats = _compute_cancer_type_signature_stats(df_gene_expr)
        cancer_code = stats[0]["code"]
        cancer_score = stats[0]["score"]
    else:
        cancer_code = resolve_cancer_type(cancer_type)
        cancer_score = None

    sample_tpm = _build_sample_tpm_by_symbol(df_gene_expr)

    # Reference expression by symbol (normalized cohort TPM)
    ref = pan_cancer_expression(technical_rna_normalize=True)
    ref_by_sym = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    reference_context = _resolve_purity_reference(cancer_code, ref_by_sym)
    reference_cancer_code = reference_context["reference_cancer_code"]
    ref_expr = reference_context["ref_expr"]
    reference_expression_source = reference_context["reference_expression_source"]
    reference_purity = reference_context["reference_purity"]

    # HK gene symbols
    hk_ids = housekeeping_gene_ids()
    id_to_sym = dict(zip(ref["Ensembl_Gene_ID"], ref["Symbol"]))
    hk_syms = [id_to_sym[gid] for gid in hk_ids if gid in id_to_sym]

    # TCGA reference purity (the TCGA cohort is NOT 100% pure)
    tcga_purity = reference_purity

    # ---- Component 1: Cancer-type signature genes ----
    sig_genes = _select_tumor_specific_genes(reference_cancer_code, n=30)
    sig_sample_ratio = _geneset_hk_ratio(sig_genes, hk_syms, sample_tpm)
    sig_ref_ratio = _geneset_hk_ratio(sig_genes, hk_syms, ref_expr)

    # Per-gene estimates for bounds
    per_gene = []
    for gene in sig_genes:
        s = sample_tpm.get(gene, 0)
        s_hk = _geneset_hk_ratio([gene], hk_syms, sample_tpm)
        r_hk = _geneset_hk_ratio([gene], hk_syms, ref_expr)
        if r_hk > 0.001:
            raw_p = s_hk / r_hk
            # Calibrate: TCGA reference ≈ tcga_purity, not 100%
            calibrated_p = raw_p * tcga_purity
            per_gene.append(
                {
                    "gene": gene,
                    "sample_tpm": s,
                    "purity_raw": float(raw_p),
                    "purity": float(np.clip(calibrated_p, 0, 1)),
                }
            )

    if sig_ref_ratio > 0:
        sig_purity_raw = sig_sample_ratio / sig_ref_ratio
        sig_purity = float(np.clip(sig_purity_raw * tcga_purity, 0, 1))
    else:
        sig_purity = None

    per_gene_purities = [g["purity"] for g in per_gene]
    sig_purity_robust, sig_lower, sig_upper, sig_stability = (
        _summarize_gene_level_purity(
            per_gene_purities,
            strategy="winsorized_median",
        )
    )
    if sig_purity_robust is not None:
        sig_purity = sig_purity_robust

    # ---- Component 2: ESTIMATE stromal genes ----
    try:
        est_df = estimate_signatures()
        stromal_genes = est_df[est_df["Category"] == "Stromal"]["Symbol"].tolist()
        immune_genes = est_df[est_df["Category"] == "Immune"]["Symbol"].tolist()
    except Exception:
        stromal_genes = []
        immune_genes = []

    stromal_sample = _geneset_hk_ratio(stromal_genes, hk_syms, sample_tpm)
    stromal_ref = _geneset_hk_ratio(stromal_genes, hk_syms, ref_expr)
    immune_sample = _geneset_hk_ratio(immune_genes, hk_syms, sample_tpm)
    immune_ref = _geneset_hk_ratio(immune_genes, hk_syms, ref_expr)

    # Stromal/immune enrichment relative to TCGA reference.
    # In TCGA, (1 - tcga_purity) is already stroma+immune.
    # If sample has more stromal signal → lower purity.
    stromal_enrichment = stromal_sample / stromal_ref if stromal_ref > 0 else 1.0
    immune_enrichment = immune_sample / immune_ref if immune_ref > 0 else 1.0

    # Convert stromal/immune enrichment into a tumor-vs-background odds model
    # rather than a linear fraction model, which otherwise collapses to 0 on
    # inflamed samples.
    tcga_nontumor_odds = (1 - tcga_purity) / max(tcga_purity, 1e-6)
    stromal_purity = 1.0 / (1.0 + tcga_nontumor_odds * max(stromal_enrichment, 0.0))
    immune_purity = 1.0 / (1.0 + tcga_nontumor_odds * max(immune_enrichment, 0.0))
    estimate_purity = float(np.clip(np.sqrt(stromal_purity * immune_purity), 0.0, 1.0))

    # ---- Component 3: Lineage gene refinement ----
    if reference_expression_source == "subtype_deconvolved":
        lineage_per_gene, lineage_skipped_detected = _subtype_lineage_purity_estimates(
            reference_cancer_code,
            LINEAGE_GENES.get(reference_cancer_code, []),
            sample_tpm,
            hk_syms,
            ref_by_sym=ref_by_sym,
        )
    else:
        lineage_per_gene, lineage_skipped_detected = _lineage_purity_estimates(
            reference_cancer_code,
            sample_tpm,
            ref_by_sym,
            hk_syms,
            tcga_purity,
        )
    # #171: mixture-cohort modeling. When the parent cohort is a union
    # of lineage-distinct subtypes (SARC = LMS ∪ liposarcoma ∪ synovial
    # ∪ ...), the TCGA parent median drowns subtype-specific markers.
    # Evaluate each subtype's panel against the subtype's tumor-only
    # expression and, if any subtype carries more anchoring genes or
    # clearly better concordance than the diluted parent panel, use its
    # result and tag the winning subtype.
    #
    # Tiebreak rule: a subtype wins when it either (a) has strictly
    # more anchoring genes (parent panels on mixture cohorts often
    # collapse to 1–2 genes after the specificity+TME filters, which
    # inflates concordance artificially) or (b) matches the parent
    # gene-count and scores better concordance.
    winning_subtype = None
    mixture_subtype_details = None
    if is_mixture_cohort(reference_cancer_code):
        mixture = _mixture_cohort_lineage_summary(
            reference_cancer_code, sample_tpm, hk_syms
        )
        if mixture is not None and mixture.get("lineage_per_gene"):
            parent_support = _summarize_lineage_support(lineage_per_gene)
            parent_conc = parent_support.get("concordance") or 0.0
            parent_n = len(lineage_per_gene)
            subtype_conc = mixture.get("concordance") or 0.0
            subtype_n = len(mixture["lineage_per_gene"])
            subtype_wins = subtype_n > parent_n or (
                subtype_n == parent_n and subtype_conc >= parent_conc
            )
            if subtype_wins:
                lineage_per_gene = mixture["lineage_per_gene"]
                lineage_skipped_detected = mixture.get("skipped") or []
                winning_subtype = mixture["code"]
                mixture_subtype_details = mixture.get("per_subtype")
    lineage_purities = sorted(g["purity"] for g in lineage_per_gene if g["purity"] > 0)
    if len(lineage_purities) >= 3:
        # Use upper-half median: genes giving LOW estimates likely
        # de-differentiated (lost expression) rather than indicating
        # low purity.  Genes giving HIGH estimates are reliable —
        # their signal can't be explained by gene loss.
        mid = len(lineage_purities) // 2
        upper_half = lineage_purities[mid:]
        lineage_purity, lineage_lower, lineage_upper, lineage_stability = (
            _summarize_gene_level_purity(
                upper_half,
                strategy="upper_half",
            )
        )
    else:
        lineage_purity = lineage_lower = lineage_upper = lineage_stability = None
    lineage_support = _summarize_lineage_support(lineage_per_gene)

    # ---- Combine estimates ----
    overall, overall_lower, overall_upper = _combine_purity_estimates(
        sig_purity=sig_purity,
        sig_lower=sig_lower,
        sig_upper=sig_upper,
        estimate_purity=estimate_purity
        if _use_estimate_component(reference_expression_source, stromal_genes)
        else None,
        lineage_purity=lineage_purity,
        lineage_lower=lineage_lower,
        lineage_upper=lineage_upper,
        sig_stability=sig_stability,
    )
    signature_deprioritized = _signature_conflicts_with_lineage(
        sig_purity=sig_purity,
        lineage_purity=lineage_purity,
        sig_stability=sig_stability,
    )
    if signature_deprioritized:
        integration_source = "lineage"
    elif sig_purity is not None and lineage_purity is not None:
        integration_source = "signature+lineage"
    elif lineage_purity is not None:
        integration_source = "lineage"
    elif sig_purity is not None:
        integration_source = "signature"
    elif estimate_purity is not None:
        integration_source = "estimate"
    else:
        integration_source = None

    return {
        "cancer_type": cancer_code,
        "reference_cancer_type": reference_cancer_code,
        "cancer_type_score": cancer_score,
        "tissue": CANCER_TO_TISSUE.get(reference_cancer_code),
        "tcga_median_purity": tcga_purity,
        "reference_expression_source": reference_expression_source,
        "overall_estimate": overall,
        "overall_lower": overall_lower,
        "overall_upper": overall_upper,
        "components": {
            "signature": {
                "genes": sig_genes,
                "purity": sig_purity,
                "lower": sig_lower,
                "upper": sig_upper,
                "stability": sig_stability,
                "per_gene": per_gene,
            },
            "lineage": {
                "genes": [g["gene"] for g in lineage_per_gene],
                "purity": lineage_purity,
                "lower": lineage_lower,
                "upper": lineage_upper,
                "stability": lineage_stability,
                "concordance": lineage_support["concordance"],
                "detection_fraction": lineage_support["detection_fraction"],
                "support_factor": lineage_support["support_factor"],
                "per_gene": lineage_per_gene,
                # Genes that WERE detected in the sample but the
                # estimator couldn't use for purity (TME dominates).
                # Consumers can render these as "uninformative" rather
                # than "not detected" — a calibration signal, not an
                # absence signal.
                "skipped_detected": lineage_skipped_detected,
                # #171: when the parent is a mixture cohort (e.g. SARC
                # = LMS ∪ liposarcoma ∪ synovial ∪ …) and one of the
                # subtypes outscored the parent panel, the winning
                # subtype is surfaced here so the classifier / brief
                # can render "Cancer call: SARC (subtype: X-consistent)".
                "winning_subtype": winning_subtype,
                "mixture_subtype_details": mixture_subtype_details,
            },
            "stromal": {
                "enrichment": stromal_enrichment,
                "n_genes": len(stromal_genes),
            },
            "immune": {
                "enrichment": immune_enrichment,
                "n_genes": len(immune_genes),
            },
            "estimate_purity": estimate_purity,
            "integration": {
                "source": integration_source,
                "signature_deprioritized": signature_deprioritized,
            },
        },
    }


# -------------------- plotting --------------------


def plot_tumor_purity(
    df_gene_expr,
    cancer_type=None,
    sample_mode="auto",
    save_to_filename=None,
    save_dpi=300,
    figsize=(14, 8),
    purity_result=None,
):
    """Plot tumor purity estimation with all components.

    Shows:
    - Left panel: per-gene purity estimates from cancer-type signature
    - Right panel: component summary (signature, stromal, immune, combined)

    Pass ``purity_result`` (e.g. the harmonized ``analysis["purity"]``
    from :func:`analyze_sample` / CLI ``analyze``) to render from a
    precomputed estimate instead of recomputing (issue #86). Without
    this the plot could silently snap back to the signature-based
    estimate even when the rest of the report had adopted a lineage-
    panel purity — producing a user-visible inconsistency between the
    ``*-purity.png`` figure and the markdown reports.
    """
    import matplotlib.pyplot as plt

    if purity_result is None:
        result = estimate_tumor_purity(df_gene_expr, cancer_type=cancer_type)
    else:
        result = purity_result
    cancer_code = result["cancer_type"]
    comp = result["components"]
    if sample_mode == "auto":
        try:
            from .decomposition import infer_sample_mode

            sample_mode = infer_sample_mode(
                cancer_types=[cancer_code], sample_mode="auto"
            )
        except Exception:
            sample_mode = "solid"

    if sample_mode == "heme":
        metric_label = "Fraction estimate"
        component_title = "Fraction / context components"
        summary_title = "Malignant-lineage fraction estimate"
        signature_label = "Malignant signature"
        overall_label = "Overall fraction proxy"
        left_title = (
            f"{cancer_code} lineage-signature fraction estimates\n"
            f"(gene TPM / HK TPM vs TCGA reference, calibrated for "
            f"TCGA median purity {result['tcga_median_purity']:.0%})"
        )
    elif sample_mode == "pure":
        metric_label = "Consistency estimate"
        component_title = "Consistency / context components"
        summary_title = "Population consistency estimate"
        signature_label = "Population signature"
        overall_label = "Overall consistency"
        left_title = (
            f"{cancer_code} lineage-profile consistency estimates\n"
            f"(gene TPM / HK TPM vs TCGA reference, not interpreted as bulk admixture)"
        )
    else:
        metric_label = "Purity estimate"
        component_title = "Purity components"
        summary_title = "Tumor purity estimate"
        signature_label = "Tumor signature"
        overall_label = "Overall estimate"
        left_title = (
            f"{cancer_code} signature gene purity estimates\n"
            f"(gene TPM / HK TPM vs TCGA reference, calibrated for "
            f"TCGA median purity {result['tcga_median_purity']:.0%})"
        )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1]}
    )

    # ---- Left: per-gene purity estimates ----
    per_gene = comp["signature"]["per_gene"]
    if per_gene:
        per_gene_sorted = sorted(per_gene, key=lambda g: -g["purity"])
        genes = [g["gene"] for g in per_gene_sorted]
        purities = [g["purity"] * 100 for g in per_gene_sorted]
        y = np.arange(len(genes))

        colors = [plt.cm.RdYlGn(p / 100) for p in purities]
        ax1.barh(y, purities, color=colors, edgecolor="none", height=0.7)

        ax1.set_yticks(y)
        ax1.set_yticklabels(genes, fontsize=8)
        ax1.set_xlabel(f"{metric_label} (%)", fontsize=10)
        ax1.set_title(left_title, fontsize=10)
        ax1.set_xlim(0, 100)
        ax1.invert_yaxis()
        ax1.axvline(
            x=comp["signature"]["purity"] * 100 if comp["signature"]["purity"] else 0,
            color="black",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label=f"Aggregate: {comp['signature']['purity']:.0%}"
            if comp["signature"]["purity"]
            else "",
        )
        ax1.legend(loc="lower right", fontsize=9)
    else:
        ax1.text(
            0.5,
            0.5,
            "No tumor-specific signature genes found",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )

    # ---- Right: component summary ----
    components = []

    if comp["signature"]["purity"] is not None:
        components.append(
            (
                f"{signature_label}\n({len(comp['signature']['genes'])} genes)",
                comp["signature"]["purity"] * 100,
                comp["signature"]["lower"] * 100
                if comp["signature"]["lower"] is not None
                else None,
                comp["signature"]["upper"] * 100
                if comp["signature"]["upper"] is not None
                else None,
                "#2166ac",
            )
        )

    components.append(
        (
            f"ESTIMATE stromal\n({comp['stromal']['n_genes']} genes)",
            None,  # not a direct purity
            None,
            None,
            "#d6604d",
        )
    )
    components.append(
        (
            f"ESTIMATE immune\n({comp['immune']['n_genes']} genes)",
            None,
            None,
            None,
            "#4393c3",
        )
    )

    if comp.get("estimate_purity") is not None:
        components.append(
            (
                "ESTIMATE combined\n(1 − infiltration)",
                comp["estimate_purity"] * 100,
                None,
                None,
                "#762a83",
            )
        )

    if result["overall_estimate"] is not None:
        components.append(
            (
                overall_label,
                result["overall_estimate"] * 100,
                result["overall_lower"] * 100,
                result["overall_upper"] * 100,
                "#1a1a1a",
            )
        )

    y_positions = []
    y_labels = []
    y_pos = 0
    for name, purity, lower, upper, color in components:
        y_positions.append(y_pos)
        y_labels.append(name)

        if purity is not None:
            ax2.barh(
                y_pos, purity, color=color, edgecolor="none", height=0.6, alpha=0.8
            )
            ax2.text(
                purity + 1,
                y_pos,
                f"{purity:.0f}%",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

            if lower is not None and upper is not None:
                ax2.plot(
                    [lower, upper],
                    [y_pos, y_pos],
                    color=color,
                    linewidth=3,
                    alpha=0.4,
                    solid_capstyle="round",
                )
        else:
            # Show enrichment for stromal/immune
            if "stromal" in name.lower():
                enr = comp["stromal"]["enrichment"]
                label = f"{enr:.1f}× vs TCGA"
                bar_val = min(enr / 5 * 100, 100)  # scale to 0-100
            elif "immune" in name.lower():
                enr = comp["immune"]["enrichment"]
                label = f"{enr:.1f}× vs TCGA"
                bar_val = min(enr / 5 * 100, 100)
            else:
                continue
            ax2.barh(
                y_pos, bar_val, color=color, edgecolor="none", height=0.6, alpha=0.4
            )
            ax2.text(bar_val + 1, y_pos, label, va="center", fontsize=9)

        y_pos += 1

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(y_labels, fontsize=9)
    ax2.set_xlim(0, 110)
    ax2.set_xlabel(f"{metric_label.split()[0]} (%) / enrichment", fontsize=10)
    ax2.set_title(component_title, fontsize=11)
    ax2.invert_yaxis()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        f"{summary_title}: {result['overall_estimate']:.0%} "
        f"[{result['overall_lower']:.0%}–{result['overall_upper']:.0%}]"
        if result["overall_estimate"] is not None
        else f"{summary_title}: N/A",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, result


def plot_purity_method_comparison(
    purity_result,
    save_to_filename=None,
    save_dpi=300,
    figsize=(10, 5.5),
    decomposition_result=None,
    title=None,
):
    """Compare every purity estimation method on one axis (#124).

    The existing :func:`plot_tumor_purity` mixes purity percentages and
    enrichment values on the same component panel, which makes the
    methods hard to compare at a glance. This dedicated figure renders
    every method on a single ``purity %`` axis with CIs as horizontal
    bars where they exist, plus reference lines for the TCGA cohort
    median and the adopted overall estimate.

    Parameters
    ----------
    purity_result : dict
        Return value of :func:`estimate_tumor_purity` or the promoted
        ``analysis["purity"]``.
    decomposition_result : DecompositionResult, optional
        If provided, the decomposition's fitted tumor fraction is
        added as its own row — useful when the decomposition adopted
        its lineage-panel call back into ``analysis["purity"]`` and
        the reader wants to see that signal plotted alongside the
        signature / lineage / ESTIMATE estimates.

    Returns the matplotlib figure; saves to ``save_to_filename`` when
    provided.
    """
    import matplotlib.pyplot as plt

    comp = purity_result.get("components", {}) or {}
    cancer_code = purity_result.get("cancer_type") or ""
    tcga_median = purity_result.get("tcga_median_purity")
    integration_source = (comp.get("integration") or {}).get("source") or ""
    signature_deprioritized = (comp.get("integration") or {}).get(
        "signature_deprioritized", False
    )

    # Re-derive stromal_purity / immune_purity from their enrichment
    # values using the same odds-model conversion estimate_tumor_purity
    # uses internally. Otherwise these two methods land on a different
    # axis than the others.
    stromal_purity = None
    immune_purity = None
    if tcga_median and tcga_median > 0:
        odds_nontumor = (1.0 - tcga_median) / max(tcga_median, 1e-6)
        stromal_enr = (comp.get("stromal") or {}).get("enrichment")
        immune_enr = (comp.get("immune") or {}).get("enrichment")
        if stromal_enr is not None:
            stromal_purity = 1.0 / (1.0 + odds_nontumor * max(float(stromal_enr), 0.0))
        if immune_enr is not None:
            immune_purity = 1.0 / (1.0 + odds_nontumor * max(float(immune_enr), 0.0))

    sig_comp = comp.get("signature") or {}
    lin_comp = comp.get("lineage") or {}

    # Row schema: (label, point, lower, upper, family, n_genes, note)
    rows = []

    # Row ordering (#159 polish): group direct purity estimates first
    # (signature / lineage / decomposition — these are tumor-fraction
    # calls in their own right), then enrichment-derived (ESTIMATE
    # stromal / immune / combined — these are odds-model conversions of
    # TME-enrichment scores, not independent tumor-fraction measurements),
    # then the adopted overall at the bottom so the reader's eye lands
    # on the final call last.
    if sig_comp.get("purity") is not None:
        n = len(sig_comp.get("genes") or [])
        rows.append(
            (
                "Tumor-specific signature",
                float(sig_comp["purity"]),
                _safe_float(sig_comp.get("lower")),
                _safe_float(sig_comp.get("upper")),
                "signature",
                n,
                " (deprioritized)" if signature_deprioritized else "",
            )
        )

    if lin_comp.get("purity") is not None:
        n = len(lin_comp.get("genes") or [])
        rows.append(
            (
                "Lineage panel",
                float(lin_comp["purity"]),
                _safe_float(lin_comp.get("lower")),
                _safe_float(lin_comp.get("upper")),
                "lineage",
                n,
                "",
            )
        )

    if decomposition_result is not None:
        decomp_purity = getattr(decomposition_result, "purity", None)
        if decomp_purity is not None:
            rows.append(
                (
                    "Decomposition (NNLS)",
                    float(decomp_purity),
                    None,
                    None,
                    "decomposition",
                    0,
                    f" [{getattr(decomposition_result, 'cancer_type', '')} / "
                    f"{getattr(decomposition_result, 'template', '')}]",
                )
            )

    # Sentinel row used only as a visual separator between the direct-
    # estimate block above and the enrichment-derived block below.
    n_direct = len(rows)

    if stromal_purity is not None:
        n = (comp.get("stromal") or {}).get("n_genes", 0)
        rows.append(
            (
                "ESTIMATE stromal (derived)",
                stromal_purity,
                None,
                None,
                "estimate",
                n,
                "",
            )
        )

    if immune_purity is not None:
        n = (comp.get("immune") or {}).get("n_genes", 0)
        rows.append(
            (
                "ESTIMATE immune (derived)",
                immune_purity,
                None,
                None,
                "estimate",
                n,
                "",
            )
        )

    if comp.get("estimate_purity") is not None:
        rows.append(
            (
                "ESTIMATE combined (derived)",
                float(comp["estimate_purity"]),
                None,
                None,
                "estimate",
                0,
                "",
            )
        )

    n_before_adopted = len(rows)

    overall = purity_result.get("overall_estimate")
    overall_lower = purity_result.get("overall_lower")
    overall_upper = purity_result.get("overall_upper")
    if overall is not None:
        rows.append(
            (
                "Adopted overall",
                float(overall),
                _safe_float(overall_lower),
                _safe_float(overall_upper),
                "adopted",
                0,
                "",
            )
        )

    # Family → color
    family_color = {
        "signature": "#2166ac",
        "lineage": "#2ca25f",
        "estimate": "#d6604d",
        "decomposition": "#762a83",
        "adopted": "#1a1a1a",
    }

    fig, ax = plt.subplots(figsize=figsize)

    # TCGA cohort median as reference line, labelled on the right edge.
    if tcga_median is not None:
        ax.axvline(
            float(tcga_median) * 100,
            color="#888888",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label=f"TCGA {cancer_code} median ({float(tcga_median):.0%})",
        )

    # Adopted overall marker as a thin vertical line so every row's
    # alignment vs the final call is immediately visible.
    if overall is not None:
        ax.axvline(
            float(overall) * 100,
            color="#1a1a1a",
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
        )

    # Horizontal separator between the direct-estimate block and the
    # enrichment-derived (ESTIMATE) block, and between the
    # enrichment-derived block and the adopted call. Makes the visual
    # hierarchy (direct / derived / final) obvious at a glance.
    for sep_y in (n_direct - 0.5, n_before_adopted - 0.5):
        if 0 < sep_y < len(rows) - 0.5:
            ax.axhline(sep_y, color="#cccccc", linewidth=0.8, alpha=0.7)

    y_positions = list(range(len(rows)))
    y_labels = []
    for y, (label, point, lower, upper, family, n_genes, note) in zip(
        y_positions, rows
    ):
        color = family_color.get(family, "#555555")
        point_pct = point * 100
        # Error bar (if CI available)
        if lower is not None and upper is not None:
            ax.plot(
                [lower * 100, upper * 100],
                [y, y],
                color=color,
                linewidth=6,
                alpha=0.25,
                solid_capstyle="round",
            )
        # Point marker
        ax.plot(
            [point_pct],
            [y],
            marker="o",
            markersize=12,
            color=color,
            markeredgecolor="white",
            markeredgewidth=1.4,
            linestyle="",
        )
        # Value text
        ax.text(
            point_pct + 1.5,
            y,
            f"{point_pct:.0f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )
        # Row label
        gene_note = f" ({n_genes} genes)" if n_genes else ""
        y_labels.append(f"{label}{gene_note}{note}")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(-2, 105)
    ax.set_xlabel("Purity estimate (%)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title is None:
        if (
            overall is not None
            and overall_lower is not None
            and overall_upper is not None
        ):
            title = (
                f"Purity estimation methods — {cancer_code} · "
                f"adopted {overall * 100:.0f}% "
                f"(range {overall_lower * 100:.0f}–{overall_upper * 100:.0f}%)"
            )
        else:
            title = f"Purity estimation methods — {cancer_code}"
        if integration_source:
            title += f" · integration: {integration_source}"
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig


def _safe_float(x):
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


# -------------------- tissue scoring --------------------


def _score_normal_tissues(sample_tpm_by_symbol, top_n=10):
    """Score each HPA normal tissue by signature gene expression in sample.

    For each tissue, selects genes most specifically expressed in that tissue
    (by z-score across 50 tissues) and computes the sample's mean midrank
    percentile for those genes.

    Returns sorted list of (tissue, score, n_genes).
    """
    ref = pan_cancer_expression(technical_rna_normalize=True)
    ref_by_sym = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    ntpm_cols = [c for c in ref.columns if c.endswith("_nTPM")]

    expr = ref_by_sym[ntpm_cols].astype(float)
    gene_mean = expr.mean(axis=1)
    gene_std = expr.std(axis=1).replace(0, np.nan)
    z_matrix = expr.sub(gene_mean, axis=0).div(gene_std, axis=0).fillna(0)

    results = []
    for col in ntpm_cols:
        tissue = col.removesuffix("_nTPM")
        z_col = z_matrix[col]
        expr_col = expr[col]
        sig_genes = list(z_col[expr_col > 0.5].nlargest(20).index)
        if len(sig_genes) < 5:
            continue

        pcts = []
        for gene in sig_genes:
            s_val = sample_tpm_by_symbol.get(gene, 0)
            if gene in expr.index:
                ref_vals = expr.loc[gene].values
                n = len(ref_vals)
                below = np.sum(ref_vals < s_val)
                equal = np.sum(np.isclose(ref_vals, s_val, atol=0.01))
                pcts.append((below + 0.5 * equal) / n)
        if pcts:
            results.append((tissue, float(np.mean(pcts)), len(pcts)))

    results.sort(key=lambda x: -x[1])
    return results[:top_n]


def _score_host_tissue_details(
    sample_tpm_by_symbol, tissues=None, top_n=None, driver_n=5
):
    """Score host tissues and retain the top sample-matching driver genes.

    This is stricter than `_score_normal_tissues()`: genes must be specific for
    the candidate host tissue relative to other tissues and relative to generic
    immune/stromal backgrounds. This prevents lymph node from winning simply
    because a sample is immune-rich.
    """
    ref = pan_cancer_expression(technical_rna_normalize=True)
    ref_by_sym = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    ntpm_cols = [c for c in ref.columns if c.endswith("_nTPM")]
    expr = ref_by_sym[ntpm_cols].astype(float)
    hk_median = _sample_hk_median(sample_tpm_by_symbol)
    if hk_median <= 0:
        hk_median = 1.0

    id_to_sym = dict(zip(ref["Ensembl_Gene_ID"], ref["Symbol"]))
    hk_gene_symbols = [
        id_to_sym[gid]
        for gid in housekeeping_gene_ids()
        if gid in id_to_sym and id_to_sym[gid] in ref_by_sym.index
    ]
    if hk_gene_symbols:
        ref_hk_medians = expr.loc[hk_gene_symbols].median(axis=0).replace(0, np.nan)
    else:
        ref_hk_medians = pd.Series(1.0, index=expr.columns, dtype=float)
    expr_hk = expr.div(ref_hk_medians, axis=1).fillna(0.0)

    gene_mean = expr_hk.mean(axis=1)
    gene_std = expr_hk.std(axis=1).replace(0, np.nan)
    z_matrix = expr_hk.sub(gene_mean, axis=0).div(gene_std, axis=0).fillna(0)

    results = []
    for col in ntpm_cols:
        tissue = col.removesuffix("_nTPM")
        if tissues is not None and tissue not in tissues:
            continue

        background_cols = [
            other
            for other in ntpm_cols
            if other != col
            and other.removesuffix("_nTPM") in _HOST_SITE_BACKGROUND_TISSUES
        ]
        if background_cols:
            background_max = expr_hk[background_cols].max(axis=1)
        else:
            background_max = pd.Series(0.0, index=expr.index)

        tissue_expr = expr_hk[col]
        z_col = z_matrix[col]
        specificity = (tissue_expr + 1e-6) / (background_max + 1e-6)
        score = z_col * np.log2(specificity + 1.0)
        keep = (
            (tissue_expr > TUMOR_PURITY_PARAMETERS["host_background"]["expression_min"])
            & (z_col > TUMOR_PURITY_PARAMETERS["host_background"]["zscore_min"])
            & (
                specificity
                > TUMOR_PURITY_PARAMETERS["host_background"]["specificity_min"]
            )
        )
        sig_genes = list(
            score[keep]
            .sort_values(ascending=False)
            .head(TUMOR_PURITY_PARAMETERS["host_background"]["top_genes"])
            .index
        )
        if len(sig_genes) < 5:
            continue

        pcts = []
        gene_details = []
        for gene in sig_genes:
            s_val = sample_tpm_by_symbol.get(gene, 0.0) / hk_median
            ref_vals = expr_hk.loc[gene].values
            n = len(ref_vals)
            below = np.sum(ref_vals < s_val)
            equal = np.sum(np.isclose(ref_vals, s_val, atol=1e-6))
            pct = (below + 0.5 * equal) / n
            pcts.append(pct)
            gene_details.append(
                {
                    "gene": gene,
                    "percentile": float(pct),
                    "sample_tpm": float(sample_tpm_by_symbol.get(gene, 0.0) or 0.0),
                }
            )
        if pcts:
            gene_details.sort(
                key=lambda row: (-row["percentile"], -row["sample_tpm"], row["gene"])
            )
            results.append(
                {
                    "tissue": tissue,
                    "score": float(np.mean(pcts)),
                    "n_genes": len(pcts),
                    "drivers": gene_details[:driver_n],
                }
            )

    results.sort(key=lambda row: -row["score"])
    if top_n is None:
        return results
    return results[:top_n]


def _score_host_tissues(sample_tpm_by_symbol, tissues=None, top_n=None):
    """Backward-compatible tuple view over `_score_host_tissue_details()`."""
    details = _score_host_tissue_details(
        sample_tpm_by_symbol,
        tissues=tissues,
        top_n=top_n,
    )
    return [(row["tissue"], row["score"], row["n_genes"]) for row in details]


def _sample_hk_median_by_id(sample_tpm_by_id):
    """Housekeeping median on the clean-TPM scale, keyed by Ensembl gene ID.

    Like :func:`_sample_hk_median` but matches the housekeeping panel by ID, so
    it shares the symbol-drift immunity of the ID-keyed family scoring (and is
    simpler — ``housekeeping_gene_ids()`` are already Ensembl IDs).
    """
    from .plot_data_helpers import _strip_ensembl_version

    hk_ids = [_strip_ensembl_version(str(gid)) for gid in housekeeping_gene_ids()]
    vals = [sample_tpm_by_id[g] for g in hk_ids if sample_tpm_by_id.get(g, 0) > 0]
    return float(np.median(vals)) if vals else 0.0


def _score_cancer_family_panels(sample_tpm_by_id):
    """Score broad cancer families before attempting fine subtype ranking.

    Markers are matched by **Ensembl gene ID** (the panels carry curated
    unversioned ENSGs) rather than HGNC symbol, so a marker can't silently drop
    to zero because of symbol/alias drift. ``sample_tpm_by_id`` is the
    versionless-ENSG-keyed sample (:func:`build_sample_tpm_by_gene_id`).
    """
    from .common import assert_tpm_keyed_by_gene_id

    # Loud-fail the ENSG-vs-symbol crossing here rather than silently scoring
    # every family as zero (the #65 regression class).
    assert_tpm_keyed_by_gene_id(sample_tpm_by_id, context="cancer-family-panel sample")
    hk_median = _sample_hk_median_by_id(sample_tpm_by_id)
    if hk_median <= 0:
        return {family: 0.0 for family in _CANCER_FAMILY_PANELS_BY_ID}

    scores = {}
    for family, ids in _CANCER_FAMILY_PANELS_BY_ID.items():
        values = [sample_tpm_by_id.get(gid, 0.0) / hk_median for gid in ids]
        if not values:
            scores[family] = 0.0
            continue
        values = sorted(values)
        upper_half = values[len(values) // 2 :] if len(values) >= 3 else values
        scores[family] = float(np.median(upper_half)) if upper_half else 0.0
    return scores


def _get_mhc_expression(sample_tpm_by_symbol):
    """Get MHC class I and II expression levels."""
    mhc1_genes = ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2"]
    mhc2_genes = ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1"]

    mhc1 = {g: sample_tpm_by_symbol.get(g, 0) for g in mhc1_genes}
    mhc2 = {g: sample_tpm_by_symbol.get(g, 0) for g in mhc2_genes}
    return mhc1, mhc2


def _resolve_family_group(family_label):
    """Collapse a specific family label to its top-level group, if any.

    Some families are expressed at two granularities (e.g. `ESCA_SQ` is a
    specific subtype of the broader `SQUAMOUS` group). When comparing or
    grouping across candidates we want to treat both as the same family.
    Families without a group mapping return their own label.
    """
    if family_label is None:
        return None
    return _CANCER_FAMILY_GROUP.get(family_label, family_label)


def _promote_same_family_alternatives(rows):
    """Surface same-family alternatives immediately after the top candidate.

    Purpose: when the top-ranked candidate belongs to a family with multiple
    TCGA members (e.g. squamous → HNSC / LUSC / CESC / ESCA, or renal →
    KIRC / KIRP / KICH), a user reading the ranked list benefits from
    seeing the sibling subtypes adjacently, even if some out-of-family
    candidate technically scored higher on the next line. This is a *display*
    rerank layered on top of the support-score sort — the top pick is
    preserved and intra-group order still reflects the underlying ranking.

    No-op when the top family has only one code in the cancer set, or when
    no same-family alternatives remain after the top pick.
    """
    if not rows:
        return rows

    best_family_group = _resolve_family_group(rows[0].get("family_label"))
    if not best_family_group:
        return rows
    if _CANCER_FAMILY_GROUP_CODE_COUNTS.get(best_family_group, 0) <= 1:
        return rows

    same_family = []
    other_rows = []
    for row in rows[1:]:
        if _resolve_family_group(row.get("family_label")) == best_family_group:
            same_family.append(row)
        else:
            other_rows.append(row)
    if not same_family:
        return rows

    return [rows[0]] + same_family + other_rows


_PRAD_CONTEXT_MARKERS = (
    "KLK2",
    "KLK3",
    "ACP3",
    "ACPP",
    "KLK4",
    "FOXA1",
    "NKX3-1",
    "HOXB13",
    "MSMB",
)


# --- TNBC / basal-BRCA misclassification rescue --------------------------
#
# HCC1395-like TNBC samples express the basal cytokeratin program
# (KRT5/KRT6A/KRT6B/KRT14) at very high TPM and lack luminal/ER markers.
# Their broad RNA signature overlaps strongly with HNSC/LUSC/ESCA — the
# squamous family — because the TCGA-BRCA reference cohort is luminal-
# dominated. Without explicit `--cancer-type BRCA` the broad classifier
# misclassifies them as ESCA/LUSC/HNSC and silently disables the
# BRCA-specific downstream biology (ER-axis suppression check, etc.).
#
# This rescue is intentionally conservative — it fires only when:
#   (1) the top candidate sits in a squamous family,
#   (2) at least two basal mammary cytokeratins exceed 100 TPM,
#   (3) ESR1 and PGR are deeply suppressed (luminal program off),
#   (4) at least one basal-mammary TF or marker is present
#       (FOXC1 ≥ 5 TPM or EGFR ≥ 20 TPM).
#
# KRT14 in particular is a strong basal-mammary marker (TCGA HNSC/LUSC
# don't typically express it). Combined with absent ESR1 / PGR it
# discriminates from broad-squamous samples.
_BASAL_MAMMARY_KERATINS = ("KRT5", "KRT6A", "KRT6B", "KRT14")
_LUMINAL_MAMMARY_MARKERS = ("ESR1", "PGR", "FOXA1")
# Mammary-positive markers — distinctly elevated in BRCA cohort medians
# (basal-enriched) vs broad squamous and bladder cohort medians:
#   gene    BRCA  BLCA  ESCA  HNSC  LUSC  CESC
#   MIA     4.2   0.4    -    0.7   0.3   0.6
#   GABRP   4.4   0.0    -    1.0   0.8   0.4
# These rule IN basal-mammary identity rather than just ruling OUT
# squamous, and they're stable across the cell-line / FFPE / fresh-tissue
# preservation modes we see in production (HCC1395 hits MIA=181 GABRP=7).
_BASAL_MAMMARY_POSITIVE = ("MIA", "GABRP")
_SQUAMOUS_PROGRAM_MARKERS = ("TP63", "SOX2")
# Urothelial differentiation panel — high in basal/luminal MIBC (BLCA),
# near zero in BRCA and broad squamous medians. Defense-in-depth gate
# against a basal-MIBC sample that somehow lands on a squamous top code.
_UROTHELIAL_MARKERS = ("UPK1A", "UPK1B", "UPK2", "UPK3A", "UPK3B")
_SQUAMOUS_TOP_CODES = {"ESCA", "LUSC", "HNSC", "CESC"}


def _detect_tnbc_basal_brca_pattern(rows, sample_tpm_by_symbol):
    """Detect basal-mammary samples misclassified into the squamous family.

    Returns a hint dict when the pattern fires, else None. Caller is
    expected to inject / promote BRCA in the candidate list and surface
    the hint in reporting.

    Background:
    Basal-like BRCA (PAM50 Basal; ~10–15% of BRCA, dominant in TNBC)
    shares its transcriptional program with squamous-keratin lineages
    (HNSC, LUSC, ESCA-SCC, CESC) and with basal-subtype bladder cancer.
    The shared axis is the keratin-5/6/14 / EGFR / p63-low pattern.
    See:
      - Hoadley et al. 2014 Cell — basal BRCA clusters with HNSC/LUSC in
        the pan-cancer 12-type analysis.
      - Lehmann et al. 2011 JCI — TNBC has BL1/BL2/M/MSL molecular
        subtypes; the BL1/BL2 ones look squamous-like.
      - Park et al. 2011 Histopathology — basal-like phenotype is a
        cross-tissue axis (breast, salivary, bladder, prostate, ovary).
      - Damrauer et al. 2014 PNAS / Choi et al. 2014 Cancer Cell — BLCA
        basal subtype directly mirrors BRCA basal.

    Discriminator stack (all gates must pass — order chosen so cheap
    rejections run first):

      1. Top picked code is in {ESCA, LUSC, HNSC, CESC}. If the
         classifier already lands on BRCA, BLCA, SARC, or anything else,
         the rescue stays quiet — promoting BRCA over a non-squamous
         call would mask other diagnoses.
      2. At least 2 of {KRT5, KRT6A, KRT6B, KRT14} ≥ 100 TPM. This is
         the basal cytokeratin program. TCGA squamous medians also fire
         here — the remaining gates discriminate among them.
      3. Luminal program is OFF: ESR1 < 5 AND PGR < 1. Luminal BRCA
         samples never reach this point because their luminal markers
         are up.
      4. FOXC1 ≥ 10 TPM. This is the load-bearing positive marker —
         FOXC1 is the canonical basal-mammary TF (Ray et al. 2010
         Cancer Res); TCGA-HNSC/LUSC/CESC cohort medians sit at 0–3
         TPM, basal-BRCA at 20–30. Without this gate the rescue false-
         fires on every squamous cohort median.
      5. At least one of {MIA ≥ 2, GABRP ≥ 2}. Mammary-positive
         confirmation — both genes are ~5 TPM in BRCA cohort medians and
         < 1 in HNSC/LUSC/CESC/BLCA. Rules basal-mammary identity IN
         rather than only ruling squamous OUT, which makes the detector
         robust against a hypothetical FOXC1-high squamous variant we
         haven't seen but can't rule out from cohort medians alone.
      6. Squamous program not fully active — NOT (TP63 ≥ 30 AND
         SOX2 ≥ 5). True squamous cohorts carry both up (TP63 ≥ 50,
         SOX2 ≥ 10 at cohort median); basal-BRCA has at most one mildly
         elevated. Two-of-two squamous-TF gate is more specific than
         either alone.
      7. Urothelial program absent — sum of {UPK1A/1B/2/3A/3B} < 10 TPM.
         Defense-in-depth against basal-MIBC; TCGA-BLCA carries the UPK
         panel at 28–92 TPM each, BRCA at < 1.

    Known limitations / follow-up work:
      - This heuristic is calibrated against cohort medians, not the
        full spectrum of within-cohort variation. A rare HNSC sample
        with FOXC1 + MIA elevated would false-positive (not seen in
        TCGA, but possible). The more principled fix is to score the
        sample directly against the curated BRCA_Basal subtype reference
        (already present in subtype-deconvolved-expression.csv.gz) and
        use that as BRCA's candidate signature when basal markers fire.
        That's a deeper change in rank_cancer_type_candidates and is
        tracked as a follow-up.
      - HER2-enriched BRCA (ERBB2-amplified, ESR1-low) does NOT trigger
        this rescue because gates 2 (basal keratin) and 4 (FOXC1) fail.
        HER2-BRCA already classifies as BRCA in the broad classifier.
      - Salivary basal-squamous and adenoid cystic carcinoma are
        classified by TCGA as HNSC; their basal-keratin programs could
        in principle reach gate 2 but they lack FOXC1 / MIA / GABRP.
    """
    if not rows:
        return None
    top_code = str(rows[0].get("code") or "")
    if top_code not in _SQUAMOUS_TOP_CODES:
        return None

    keratin_tpm = {
        sym: float(sample_tpm_by_symbol.get(sym, 0.0) or 0.0)
        for sym in _BASAL_MAMMARY_KERATINS
    }
    high_keratins = [sym for sym, tpm in keratin_tpm.items() if tpm >= 100.0]
    if len(high_keratins) < 2:
        return None

    luminal_tpm = {
        sym: float(sample_tpm_by_symbol.get(sym, 0.0) or 0.0)
        for sym in _LUMINAL_MAMMARY_MARKERS
    }
    if luminal_tpm.get("ESR1", 0.0) >= 5.0:
        return None
    if luminal_tpm.get("PGR", 0.0) >= 1.0:
        return None

    foxc1 = float(sample_tpm_by_symbol.get("FOXC1", 0.0) or 0.0)
    if foxc1 < 10.0:
        return None

    positive_tpm = {
        sym: float(sample_tpm_by_symbol.get(sym, 0.0) or 0.0)
        for sym in _BASAL_MAMMARY_POSITIVE
    }
    if positive_tpm.get("MIA", 0.0) < 2.0 and positive_tpm.get("GABRP", 0.0) < 2.0:
        return None

    squamous_tpm = {
        sym: float(sample_tpm_by_symbol.get(sym, 0.0) or 0.0)
        for sym in _SQUAMOUS_PROGRAM_MARKERS
    }
    if squamous_tpm.get("TP63", 0.0) >= 30.0 and squamous_tpm.get("SOX2", 0.0) >= 5.0:
        return None

    urothelial_sum = sum(
        float(sample_tpm_by_symbol.get(sym, 0.0) or 0.0)
        for sym in _UROTHELIAL_MARKERS
    )
    if urothelial_sum >= 10.0:
        return None

    return {
        "kind": "tnbc_basal_brca_misclassification",
        "recommended_code": "BRCA",
        "recommended_subtype": "BRCA_Basal",
        "competing_top_code": top_code,
        "high_basal_keratins": high_keratins,
        "keratin_tpm": {sym: round(val, 2) for sym, val in keratin_tpm.items()},
        "luminal_marker_tpm": {sym: round(val, 2) for sym, val in luminal_tpm.items()},
        "foxc1_tpm": round(foxc1, 2),
        "basal_positive_tpm": {sym: round(val, 2) for sym, val in positive_tpm.items()},
        "squamous_program_tpm": {sym: round(val, 2) for sym, val in squamous_tpm.items()},
        "urothelial_panel_sum_tpm": round(urothelial_sum, 2),
        "message": (
            f"Basal mammary cytokeratin program is dominant "
            f"({', '.join(f'{k}={keratin_tpm[k]:.0f}' for k in high_keratins)} TPM); "
            f"FOXC1 {foxc1:.0f} TPM, mammary-positive markers up "
            f"(MIA={positive_tpm.get('MIA', 0):.1f}, "
            f"GABRP={positive_tpm.get('GABRP', 0):.1f}); luminal program off "
            f"(ESR1={luminal_tpm.get('ESR1', 0):.2f}, "
            f"PGR={luminal_tpm.get('PGR', 0):.2f}), squamous program absent "
            f"(TP63={squamous_tpm.get('TP63', 0):.1f}, "
            f"SOX2={squamous_tpm.get('SOX2', 0):.1f}), urothelial panel off "
            f"(UPK sum={urothelial_sum:.1f}); the broad "
            f"{top_code} call is the standard misclassification of basal-like "
            "BRCA against a luminal-dominated TCGA-BRCA reference. Treat as "
            "TNBC / basal-BRCA."
        ),
    }


def _apply_tnbc_basal_brca_rescue(rows, sample_tpm_by_symbol):
    """Promote BRCA over the squamous family on basal-mammary samples.

    HCC1395-style TNBC samples: see :func:`_detect_tnbc_basal_brca_pattern`.
    Mirrors the PRAD-stromal rescue pattern — when the pattern fires AND
    BRCA is already a scored candidate, promote it above the current top
    and tag its subtype as ``BRCA_Basal`` so downstream reporting knows
    why. We do NOT synthesize a BRCA row from nothing — if BRCA isn't in
    candidates the broad signature score is too far below the squamous
    top to be honest about; instead surface the hint via ``support_override``
    on the existing top row so reporting can still warn the reader.
    """

    if not rows:
        return rows
    pattern = _detect_tnbc_basal_brca_pattern(rows, sample_tpm_by_symbol)
    if not pattern:
        return rows

    max_support = max((float(row.get("support_score") or 0.0) for row in rows), default=0.0)
    if max_support <= 0:
        return rows

    for idx, row in enumerate(rows, start=1):
        row.setdefault("pre_rescue_rank", idx)
        row.setdefault("pre_rescue_support_score", row.get("support_score"))
        row.setdefault("pre_rescue_support_geomean", row.get("support_geomean"))

    brca = next((r for r in rows if str(r.get("code") or "") == "BRCA"), None)
    if brca is None:
        # BRCA wasn't in the scored candidate pool — surface the hint on
        # the current top row so the report can still warn, but don't
        # change the call. Reporting code reads `support_override` from
        # whichever row is at rank 1.
        rows[0]["support_override"] = {**pattern, "promoted": False}
        return rows

    brca["support_override"] = {**pattern, "promoted": True}
    brca["winning_subtype"] = "BRCA_Basal"
    brca["support_score"] = max(float(brca.get("support_score") or 0.0), max_support * 1.05)
    brca["support_geomean"] = (
        float(brca["support_score"] ** 0.2) if brca["support_score"] > 0 else 0.0
    )

    rows.sort(
        key=lambda row: (
            -row["support_score"],
            -row["signature_score"],
            row["code"],
        )
    )
    return rows


def _apply_normal_tissue_tiebreaker(
    rows,
    sample_tpm_by_symbol,
    *,
    close_window: float = 0.93,
    boost_factor: float = 1.02,
):
    """Break close cohort-call ties with a normal-tissue parity check.

    User-suggested in pirl-unc/trufflepig#28: *"if a few cancer types are
    close and one of them has a parental normal tissue that's present in
    the sample, isn't that enough to figure out what's going on?"*

    Mechanism: when two or more candidates sit within ``close_window`` of
    the top support_score, look up each candidate's primary tissue via
    :data:`CANCER_TO_TISSUE`, and score the sample's match against that
    tissue's HPA nTPM column (already done by
    :func:`_score_host_tissue_details`). The candidate whose primary
    tissue scores best — and beats the current top candidate's tissue
    score — gets a small ``boost_factor`` (default 2%) applied to its
    ``support_score``, which is enough to flip rankings within the
    close window but won't override a clear winner.

    Caveats:
      - Cell lines lose normal-tissue differentiation, so their tissue
        scores tend to be uniformly weak; tiebreaker becomes near-neutral
        — that's fine, the subtype-aware signature path is already
        handling them.
      - Candidates sharing a primary tissue (LUAD + LUSC → lung) can't
        be discriminated by this signal; no boost is applied.
      - Tiebreaker metadata is attached to every close candidate as
        ``primary_tissue`` / ``primary_tissue_match_score`` for report
        transparency, regardless of whether a boost fired.
    """
    if not rows or len(rows) < 2:
        return rows
    top_support = float(rows[0].get("support_score") or 0.0)
    if top_support <= 0:
        return rows
    threshold = top_support * float(close_window)
    close_idx = [
        i
        for i, row in enumerate(rows)
        if float(row.get("support_score") or 0.0) >= threshold
    ]
    if len(close_idx) < 2:
        return rows

    tissue_details = _score_host_tissue_details(sample_tpm_by_symbol, top_n=25)
    tissue_to_score = {
        str(t.get("tissue") or ""): float(t.get("score") or 0.0)
        for t in tissue_details or []
    }

    annotated: list[tuple[int, dict, str, float]] = []
    for i in close_idx:
        row = rows[i]
        code = str(row.get("code") or "")
        tissue = CANCER_TO_TISSUE.get(code)
        if not tissue:
            continue
        tissue_score = tissue_to_score.get(tissue, 0.0)
        row["primary_tissue"] = tissue
        row["primary_tissue_match_score"] = tissue_score
        annotated.append((i, row, tissue, tissue_score))

    if len(annotated) < 2:
        return rows

    best_i, best_row, best_tissue, best_score = max(annotated, key=lambda t: t[3])
    if best_score <= 0:
        return rows
    if best_i == 0:
        # Top candidate already has the best tissue match — nothing to do.
        rows[0]["normal_tissue_tiebreaker"] = {
            "applied": False,
            "reason": "top already best on primary-tissue match",
            "tissue": best_tissue,
            "tissue_score": best_score,
        }
        return rows
    top_tissue = rows[0].get("primary_tissue")
    top_tissue_score = float(rows[0].get("primary_tissue_match_score") or 0.0)
    if best_tissue == top_tissue:
        # Candidates share their primary tissue (e.g. LUAD + LUSC = lung) —
        # nothing to discriminate.
        return rows
    if best_score <= top_tissue_score:
        return rows

    boosted = max(float(best_row.get("support_score") or 0.0), top_support * boost_factor)
    best_row["pre_tiebreaker_support_score"] = best_row.get("support_score")
    best_row["pre_tiebreaker_support_geomean"] = best_row.get("support_geomean")
    best_row["support_score"] = boosted
    best_row["support_geomean"] = (
        float(boosted ** 0.2) if boosted > 0 else 0.0
    )
    best_row["normal_tissue_tiebreaker"] = {
        "applied": True,
        "tissue": best_tissue,
        "tissue_score": best_score,
        "competing_top_code": str(rows[0].get("code") or ""),
        "competing_top_tissue": top_tissue,
        "competing_top_tissue_score": top_tissue_score,
        "boost_factor": float(boost_factor),
        "close_window": float(close_window),
        "all_close_tissues": {
            str(r.get("code") or ""): {
                "tissue": t,
                "tissue_score": s,
            }
            for _, r, t, s in annotated
        },
    }

    rows.sort(
        key=lambda row: (
            -row["support_score"],
            -row["signature_score"],
            row["code"],
        )
    )
    return rows


def _detect_low_purity_prad_stromal_pitfall(
    rows,
    sample_tpm_by_symbol,
    *,
    host_tissue_details=None,
):
    """Detect prostate-context samples where stroma can masquerade as SARC.

    This is intentionally narrow: it only fires when PRAD has the strongest raw
    cancer signature and explicit prostate tissue/marker evidence, while the
    broad SARC call is being driven by a mesenchymal or smooth-muscle lineage
    panel and PRAD lineage detection is attenuated.
    """

    by_code = {str(row.get("code") or ""): row for row in rows}
    sarc = by_code.get("SARC")
    prad = by_code.get("PRAD")
    if not sarc or not prad:
        return None

    top_code = str(rows[0].get("code") or "") if rows else ""
    if top_code != "SARC":
        return None

    prad_sig = float(prad.get("signature_score") or 0.0)
    sarc_sig = float(sarc.get("signature_score") or 0.0)
    if prad_sig < 0.70 or prad_sig < sarc_sig * 1.10:
        return None

    prad_lineage = float(prad.get("lineage_purity") or 0.0)
    prad_detect = float(prad.get("lineage_detection_fraction") or 0.0)
    sarc_detect = float(sarc.get("lineage_detection_fraction") or 0.0)
    if not (prad_lineage <= 0.20 or prad_detect <= 0.35):
        return None
    if sarc_detect < 0.75:
        return None

    if host_tissue_details is None:
        host_tissue_details = _score_host_tissue_details(
            sample_tpm_by_symbol,
            top_n=5,
        )
    prostate_row = next(
        (
            row
            for row in host_tissue_details or []
            if str(row.get("tissue") or "") == "prostate"
        ),
        None,
    )
    prostate_score = float((prostate_row or {}).get("score") or 0.0)
    if prostate_score < 0.85:
        return None

    marker_values = {
        marker: float(sample_tpm_by_symbol.get(marker, 0.0) or 0.0)
        for marker in _PRAD_CONTEXT_MARKERS
    }
    marker_count = sum(1 for value in marker_values.values() if value >= 2.0)
    canonical_count = sum(
        1
        for marker in ("KLK2", "KLK3", "ACP3", "ACPP", "KLK4", "FOXA1")
        if marker_values.get(marker, 0.0) >= 2.0
    )
    if marker_count < 4 or canonical_count < 4:
        return None

    winning_subtype = str(sarc.get("winning_subtype") or "")
    subtype_label = {
        "SARC_LMS": "leiomyosarcoma-like",
        "SARC_SYN": "synovial-sarcoma-like",
        "SARC_UPS": "undifferentiated-pleomorphic-sarcoma-like",
    }.get(winning_subtype, winning_subtype.replace("_", " ").strip())
    subtype_clause = (
        f"; SARC subtype signal was {subtype_label}"
        if subtype_label
        else ""
    )
    return {
        "kind": "low_purity_prad_stromal_context",
        "recommended_code": "PRAD",
        "competing_code": "SARC",
        "message": (
            "Prostate tissue/context is strong and PRAD has the strongest raw "
            "signature, but the epithelial PRAD lineage panel is attenuated while "
            "a mesenchymal/smooth-muscle SARC panel is dominant"
            f"{subtype_clause}."
        ),
        "prostate_background_score": prostate_score,
        "prostate_marker_tpm": {
            marker: round(value, 3)
            for marker, value in marker_values.items()
            if value >= 2.0
        },
        "prad_signature_score": prad_sig,
        "sarc_signature_score": sarc_sig,
        "prad_lineage_purity": prad_lineage,
        "prad_lineage_detection_fraction": prad_detect,
        "sarc_lineage_detection_fraction": sarc_detect,
        "interpretation": (
            "Treat the SARC/smooth-muscle signal as a stromal-admixture pitfall "
            "unless pathology independently supports sarcoma. Confirm tumor "
            "cellularity, preservation/RIN, therapy state, and prostate lineage "
            "markers before using expression-derived therapy context."
        ),
    }


def _apply_prad_stromal_rescue(rows, sample_tpm_by_symbol):
    """Promote PRAD over SARC in the narrow stromal-prostate pitfall."""

    if not rows:
        return rows
    host_tissue_details = _score_host_tissue_details(sample_tpm_by_symbol, top_n=5)
    pitfall = _detect_low_purity_prad_stromal_pitfall(
        rows,
        sample_tpm_by_symbol,
        host_tissue_details=host_tissue_details,
    )
    if not pitfall:
        return rows

    max_support = max((float(row.get("support_score") or 0.0) for row in rows), default=0.0)
    if max_support <= 0:
        return rows

    for idx, row in enumerate(rows, start=1):
        row.setdefault("pre_rescue_rank", idx)
        row.setdefault("pre_rescue_support_score", row.get("support_score"))
        row.setdefault("pre_rescue_support_geomean", row.get("support_geomean"))
    for row in rows:
        if row.get("code") != "PRAD":
            continue
        row["support_override"] = pitfall
        row["support_score"] = max(float(row.get("support_score") or 0.0), max_support * 1.05)
        row["support_geomean"] = (
            float(row["support_score"] ** 0.2) if row["support_score"] > 0 else 0.0
        )
        break

    rows.sort(
        key=lambda row: (
            -row["support_score"],
            -row["signature_score"],
            row["code"],
        )
    )
    return rows


def _candidate_raw_support(row, family_params):
    """Candidate evidence before family/orphan weighting."""

    return (
        float(row.get("signature_score") or 0.0)
        * max(
            float(row.get("purity_estimate") or 0.0),
            family_params["support_fraction_of_top_floor"],
        )
        * float(row.get("lineage_support_factor") or 1.0)
    )


def _recompute_candidate_support(row, family_params, family_factor=None):
    factor = float(row.get("family_factor") if family_factor is None else family_factor)
    support_factors = (
        float(row.get("signature_score") or 0.0),
        max(
            float(row.get("purity_estimate") or 0.0),
            family_params["support_fraction_of_top_floor"],
        ),
        float(row.get("lineage_support_factor") or 1.0),
        max(
            float(row.get("signature_stability") or 0.0),
            family_params["signature_stability_floor"],
        ),
        max(factor, family_params["min_factor"]),
    )
    row["family_factor"] = max(factor, family_params["min_factor"])
    row["support_score"] = float(np.prod(support_factors))
    row["support_geomean"] = (
        float(row["support_score"] ** (1.0 / len(support_factors)))
        if row["support_score"] > 0
        else 0.0
    )


def _apply_coarse_tcga_orphan_rescue(rows, family_params, tissue_signal=None):
    """Let strong tissue-composition context suspend an orphan cancer-type penalty.

    BLCA, CHOL, LIHC, PAAD, etc. do not belong to the broad family panels.
    They can therefore be penalized below a family-coded competitor even when
    the direct cancer evidence and the coarse cancer-reference/normal-tissue read agree.
    This rescue is intentionally restricted to unconstrained auto-detection:
    it only considers the top cancer-reference cohort, requires that cohort to be
    an orphan candidate, and requires either matching normal-tissue context or
    clear raw-signal dominance.
    """

    if not rows or tissue_signal is None:
        return rows

    cancer_hint = str(getattr(tissue_signal, "cancer_hint", "") or "")
    if cancer_hint == "healthy-dominant":
        return rows

    top_tcga = list(getattr(tissue_signal, "top_tcga_cohorts", None) or [])
    if not top_tcga:
        return rows

    coarse_code = str(top_tcga[0][0] or "").removesuffix("_TPM")
    if not coarse_code:
        return rows

    by_code = {str(row.get("code") or ""): row for row in rows}
    coarse_row = by_code.get(coarse_code)
    if coarse_row is None:
        return rows
    if coarse_row.get("family_label") is not None:
        return rows
    if float(coarse_row.get("family_factor") or 0.0) >= 1.0:
        return rows

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -float(row.get("support_score") or 0.0),
            -float(row.get("signature_score") or 0.0),
            str(row.get("code") or ""),
        ),
    )
    current_top = sorted_rows[0]
    if current_top.get("code") == coarse_code:
        return rows
    if current_top.get("family_label") is None:
        return rows

    signature = float(coarse_row.get("signature_score") or 0.0)
    if signature < family_params["orphan_context_min_signature"]:
        return rows

    coarse_raw = _candidate_raw_support(coarse_row, family_params)
    top_raw = _candidate_raw_support(current_top, family_params)
    if top_raw <= 0:
        return rows
    if coarse_raw < family_params["orphan_context_min_raw_ratio"] * top_raw:
        return rows

    top_normals = list(getattr(tissue_signal, "top_normal_tissues", None) or [])
    expected_tissue = CANCER_TO_TISSUE.get(coarse_code)
    observed_tissue = (
        str(top_normals[0][0] or "").removesuffix("_nTPM") if top_normals else ""
    )
    tissue_matches = bool(expected_tissue and observed_tissue == expected_tissue)
    raw_dominates = coarse_raw >= (
        family_params["orphan_context_dominant_raw_ratio"] * top_raw
    )
    if not (tissue_matches or raw_dominates):
        return rows

    for idx, row in enumerate(sorted_rows, start=1):
        row.setdefault("pre_rescue_rank", idx)
        row.setdefault("pre_rescue_support_score", row.get("support_score"))
        row.setdefault("pre_rescue_support_geomean", row.get("support_geomean"))
    context_basis = "normal_tissue_match" if tissue_matches else "raw_signal_dominance"
    if tissue_matches:
        rescue_message = (
            f"Tissue composition screen and expected normal-tissue context support {coarse_code}; "
            "suspending the orphan family penalty for the auto-detected call."
        )
    else:
        rescue_message = (
            f"Tissue composition screen and direct cancer evidence support {coarse_code}; "
            "suspending the orphan family penalty for the auto-detected call."
        )

    coarse_row["support_override"] = {
        "kind": "coarse_tcga_orphan_context",
        "recommended_code": coarse_code,
        "competing_code": str(current_top.get("code") or ""),
        "top_tcga_rho": float(top_tcga[0][1] or 0.0),
        "top_normal_tissue": observed_tissue,
        "expected_tissue": expected_tissue,
        "context_basis": context_basis,
        "raw_signal_ratio": float(coarse_raw / top_raw),
        "message": rescue_message,
    }
    _recompute_candidate_support(coarse_row, family_params, family_factor=1.0)
    rows.sort(
        key=lambda row: (
            -row["support_score"],
            -row["signature_score"],
            row["code"],
        )
    )
    return rows


def rank_cancer_type_candidates(
    df_gene_expr,
    candidate_codes=None,
    top_k=5,
    tissue_signal=None,
    use_subtype_signatures=True,
):
    """Rank cancer-type hypotheses by signature evidence and purity plausibility.

    Pure signature similarity tends to overcall stromal or immune-rich cancer
    types when the sample has heavy admixture. We rank candidates by:

    - cancer-type signature similarity
    - a purity anchor that combines tumor-specific and lineage evidence
    - lineage pattern concordance when lineage genes are available

    This keeps "one of these two is plausible" ambiguity visible while
    downweighting types whose purity model does not fit the sample.
    """
    from .plot_embedding import _compute_cancer_type_signature_stats
    from .subtype_signature import compute_subtype_signature_stats

    unconstrained = candidate_codes is None
    if unconstrained:
        resolved_candidate_codes = None
    else:
        resolved_candidate_codes = [
            resolve_cancer_type(code) for code in candidate_codes
        ]
    stats = _compute_cancer_type_signature_stats(
        df_gene_expr,
        candidate_codes=resolved_candidate_codes,
    )
    signature_score_map = {row["code"]: float(row["score"]) for row in stats}

    # Subtype-aware signature scoring: for cohorts with subtype data
    # (BRCA Basal/LumA/B/Her2/Normal; HNSC HPV+/-; LUAD by driver), score
    # against each subtype's data-derived signature panel. The candidate
    # loop below uses the best subtype score when it exceeds the broad
    # cohort score, and tags the candidate with the winning_subtype.
    # This is the principled answer to the basal-BRCA / squamous overlap
    # problem (see :mod:`trufflepig.subtype_signature`).
    subtype_stats = {}
    if use_subtype_signatures:
        if unconstrained:
            subtype_stats = compute_subtype_signature_stats(df_gene_expr)
        else:
            subtype_parent_codes = {"BRCA", "HNSC", "LUAD"}
            if any(
                code in subtype_parent_codes for code in resolved_candidate_codes
            ):
                subtype_stats = compute_subtype_signature_stats(df_gene_expr)
    sample_tpm = _build_sample_tpm_by_symbol(df_gene_expr)
    # Family panels are scored by Ensembl ID (alias-drift immune); build the
    # versionless-ENSG-keyed sample for that lookup.
    sample_tpm_by_id = _build_sample_tpm_by_gene_id(df_gene_expr)
    family_scores = _score_cancer_family_panels(sample_tpm_by_id)
    family_params = TUMOR_PURITY_PARAMETERS["family_scoring"]
    soft_families = set(family_params.get("non_penalizing_families", []))
    hard_family_scores = {
        family: score
        for family, score in family_scores.items()
        if family not in soft_families
    }
    max_family_score = max(hard_family_scores.values(), default=0.0)
    sorted_family_scores = sorted(hard_family_scores.values(), reverse=True)
    top_family_score = sorted_family_scores[0] if sorted_family_scores else 0.0
    second_family_score = (
        sorted_family_scores[1] if len(sorted_family_scores) > 1 else 0.0
    )
    family_presence = float(
        np.clip(top_family_score / family_params["presence_scale"], 0.0, 1.0)
    )
    family_specificity = 0.0
    if top_family_score > 0:
        family_specificity = float(
            np.clip(
                (top_family_score - second_family_score) / top_family_score, 0.0, 1.0
            )
        )
    ranked_families = sorted(
        family_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    hard_ranked_families = sorted(
        hard_family_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    if unconstrained:
        candidate_codes = [row["code"] for row in stats[:8]]
        for family, score in ranked_families[: family_params["candidate_panel_top_n"]]:
            if score < family_params["candidate_panel_min_score"]:
                continue
            if (
                family in soft_families
                and top_family_score >= family_params["presence_scale"]
            ):
                continue
            family_codes = [
                code
                for code, family_label in _CANCER_FAMILY_BY_CODE.items()
                if family_label == family
            ]
            candidate_codes.extend(family_codes)
        # Always include cohorts with subtype data, even if their broad
        # signature doesn't crack the top-8. The whole point of the
        # subtype-aware scoring path is that a basal-BRCA sample fails
        # the broad-BRCA signature (TCGA-BRCA is luminal-dominated) but
        # succeeds against BRCA_Basal — so BRCA must reach the per-
        # candidate loop where the subtype score is consulted. Same
        # logic for HNSC (HPV+/-) and LUAD (driver mutations).
        # Restrict to codes that are valid TCGA pan-cancer cohorts —
        # BEATAML / TARGET_NBL appear in subtype data but aren't TCGA
        # cohorts and don't have *_TPM columns to score against.
        valid_tcga_codes = {row["code"] for row in stats}
        subtype_aware_codes = sorted(
            code for code in subtype_stats.keys() if code in valid_tcga_codes
        ) if subtype_stats else []
        for code in subtype_aware_codes:
            if code not in candidate_codes:
                candidate_codes.append(code)
        # When the early signature top is squamous and the sample shows
        # the basal-mammary cytokeratin program, force BRCA into the
        # scored pool so the TNBC rescue further down has something to
        # promote. (Kept as defense-in-depth even with subtype scoring —
        # if subtype data is ever missing, the heuristic still anchors.)
        if candidate_codes and candidate_codes[0] in _SQUAMOUS_TOP_CODES:
            preview_top = [{"code": c} for c in candidate_codes[:1]]
            if _detect_tnbc_basal_brca_pattern(preview_top, sample_tpm):
                if "BRCA" not in candidate_codes:
                    candidate_codes.append("BRCA")
    else:
        candidate_codes = resolved_candidate_codes

    seen = set()
    ordered_codes = []
    for code in candidate_codes:
        if code not in seen:
            seen.add(code)
            ordered_codes.append(code)

    rows = []
    top_family_label = hard_ranked_families[0][0] if hard_ranked_families else None
    non_penalizing_families = soft_families
    for code in ordered_codes:
        purity_result = estimate_tumor_purity(df_gene_expr, cancer_type=code)
        purity_estimate = float(purity_result["overall_estimate"] or 0.0)
        broad_signature_score = float(signature_score_map.get(code, 0.0))
        signature_score = broad_signature_score
        signature_subtype_promoted = None
        # If this cohort has subtype-specific signatures (BRCA Basal/LumA/
        # LumB/Her2/Normal; HNSC HPV+/-; LUAD by driver) and the best
        # subtype score beats the broad cohort score, promote the subtype
        # score. This fixes the basal-BRCA-vs-squamous misclassification
        # in a data-driven way — see :mod:`trufflepig.subtype_signature`.
        subtype_candidates = subtype_stats.get(code) if subtype_stats else None
        if subtype_candidates:
            best_subtype_row = subtype_candidates[0]
            best_subtype_score = float(best_subtype_row.get("score") or 0.0)
            if best_subtype_score > signature_score:
                signature_score = best_subtype_score
                signature_subtype_promoted = str(best_subtype_row.get("subtype") or "")
        signature_stability = float(
            purity_result.get("components", {}).get("signature", {}).get("stability")
            or 1.0
        )
        lineage = purity_result.get("components", {}).get("lineage", {})
        lineage_support_factor = float(lineage.get("support_factor") or 1.0)
        lineage_concordance = lineage.get("concordance")
        lineage_detection_fraction = lineage.get("detection_fraction")
        winning_subtype = lineage.get("winning_subtype")
        # If the subtype signature promoted a specific subtype, surface
        # it as the candidate's winning subtype (lineage-side mixture
        # cohort detection is the legacy path and still wins when set).
        if winning_subtype is None and signature_subtype_promoted:
            winning_subtype = signature_subtype_promoted
        family_label = _CANCER_FAMILY_BY_CODE.get(code)
        if family_label is not None:
            if family_label in non_penalizing_families:
                family_factor = float(
                    np.clip(
                        1.0
                        - family_params["soft_family_penalty_gain"] * family_presence,
                        family_params["min_factor"],
                        1.0,
                    )
                )
            else:
                family_relative = (
                    float(family_scores.get(family_label, 0.0) / max_family_score)
                    if max_family_score > 0
                    else 0.0
                )
                family_factor = float(
                    np.clip(
                        family_params["within_family_base"]
                        + family_params["within_family_gain"]
                        * family_presence
                        * family_relative,
                        family_params["min_factor"],
                        1.0,
                    )
                )
        elif family_label is None and family_presence > 0:
            if top_family_label in non_penalizing_families:
                family_factor = 1.0
            else:
                family_factor = float(
                    np.clip(
                        1.0 - family_params["non_family_penalty"] * family_presence,
                        family_params["min_factor"],
                        1.0,
                    )
                )
        else:
            family_factor = 1.0
        support_factors = (
            signature_score,
            max(purity_estimate, family_params["support_fraction_of_top_floor"]),
            lineage_support_factor,
            max(signature_stability, family_params["signature_stability_floor"]),
            max(family_factor, family_params["min_factor"]),
        )
        support_score = float(np.prod(support_factors))
        # Geomean across the 5 factors — same ordering as support_score but
        # stays bounded on [0, 1] instead of collapsing toward zero.
        support_geomean = (
            float(support_score ** (1.0 / len(support_factors)))
            if support_score > 0
            else 0.0
        )
        rows.append(
            {
                "code": code,
                "signature_score": signature_score,
                "broad_signature_score": broad_signature_score,
                "signature_subtype_promoted": signature_subtype_promoted,
                "signature_stability": signature_stability,
                "purity_estimate": purity_estimate,
                "lineage_purity": lineage.get("purity"),
                "lineage_concordance": lineage_concordance,
                "lineage_detection_fraction": lineage_detection_fraction,
                "lineage_support_factor": lineage_support_factor,
                "winning_subtype": winning_subtype,
                "family_label": family_label,
                "family_score": family_scores.get(family_label)
                if family_label is not None
                else None,
                "family_presence": family_presence,
                "family_specificity": family_specificity,
                "family_factor": family_factor,
                "support_score": support_score,
                "support_geomean": support_geomean,
                "purity_result": purity_result,
            }
        )

    # #160: orphan-dominance override. Cancer types that aren't grouped
    # into a family (BLCA, PAAD, MESO, ACC, CHOL, LIHC, ...) were getting
    # penalized by ``non_family_penalty`` even when every direct signal
    # (signature × purity × lineage support) strongly favored them over
    # a family-matched competitor. On the TCGA-median battery this
    # miscalled BLCA → ESCA (orphan BLCA at signature 0.94 losing to
    # ESCA_SQ-family ESCA at 0.66). The override fires only when the
    # orphan's evidence is unambiguously tumor-like — three gates must
    # all pass:
    #
    #   1. ``signature_score ≥ 0.80`` — signature is itself strong,
    #      not a byproduct of TME bleed-through
    #   2. ``purity_estimate ≥ 0.40`` — sample looks like a real
    #      tumor, not a dilute admixture
    #   3. raw-signal dominance (sig × purity × lineage_support) ≥ 1.5×
    #      the top family-matched competitor's — enough to justify
    #      ignoring the family signal
    #
    # On a COAD/lymph-node 30/70 mix DLBC's raw signal can dominate
    # from the TME alone (sig 0.61, pur 0.33). The signature+purity
    # gates reject that case. BLCA on its own median (sig 0.94,
    # pur 0.59) passes all three and correctly promotes.
    # Dominance-ratio threshold was 1.5 at introduction (#160) but the
    # #170 lineage-panel expansion boosted the family-winner's
    # lineage_support_factor on average (LUSC on a BLCA sample moved
    # from ~0.8 to 0.95 because DSG3/TRIM29 anchor the squamous panel),
    # narrowing the margin to ~1.4×. The signature + purity gates
    # remain the primary guard against false positives (DLBC on a
    # COAD/lymph mix has sig 0.61 < 0.80, rejected there). Lowering
    # the ratio to 1.3 preserves the BLCA / PAAD wins without reopening
    # the COAD/lymph regression.
    _ORPHAN_DOMINANCE_RATIO = 1.3
    _ORPHAN_DOMINANCE_MIN_SIGNATURE = 0.80
    _ORPHAN_DOMINANCE_MIN_PURITY = 0.40
    family_matched_rows = [r for r in rows if r["family_label"] is not None]
    if family_matched_rows:

        best_family_raw = max(
            _candidate_raw_support(r, family_params) for r in family_matched_rows
        )
        if best_family_raw > 0:
            for r in rows:
                if r["family_label"] is not None or r["family_factor"] >= 1.0:
                    continue
                if r["signature_score"] < _ORPHAN_DOMINANCE_MIN_SIGNATURE:
                    continue
                if r["purity_estimate"] < _ORPHAN_DOMINANCE_MIN_PURITY:
                    continue
                if (
                    _candidate_raw_support(r, family_params)
                    < _ORPHAN_DOMINANCE_RATIO * best_family_raw
                ):
                    continue
                # Orphan dominates — recompute support without the
                # family-penalty handicap.
                _recompute_candidate_support(r, family_params, family_factor=1.0)

    rows.sort(
        key=lambda row: (
            -row["support_score"],
            -row["signature_score"],
            row["code"],
        )
    )
    if unconstrained:
        if tissue_signal is None:
            needs_coarse_context = (
                rows
                and rows[0].get("family_label") is not None
                and any(
                    row.get("family_label") is None
                    and float(row.get("family_factor") or 0.0) < 1.0
                    for row in rows[1:]
                )
            )
            if needs_coarse_context:
                try:
                    from .healthy_vs_tumor import assess_tissue_composition

                    tissue_signal = assess_tissue_composition(df_gene_expr)
                except Exception:  # noqa: BLE001
                    tissue_signal = None
        rows = _apply_coarse_tcga_orphan_rescue(
            rows,
            family_params,
            tissue_signal=tissue_signal,
        )
        rows = _apply_prad_stromal_rescue(rows, sample_tpm)
        rows = _apply_tnbc_basal_brca_rescue(rows, sample_tpm)
        rows = _apply_normal_tissue_tiebreaker(rows, sample_tpm)
    rows = _promote_same_family_alternatives(rows)

    # ``support_fraction_of_top`` = ``support_score`` / max(support_score over
    # all candidates). The top candidate always has 1.0; the runner-up's
    # value reads as "fraction of the leader's RNA support". Not a
    # probability, not a sum-to-one share — purely relative-to-top scaling
    # so downstream consumers can compare candidates without re-fetching
    # the absolute geomean scale.
    max_support = max((row["support_score"] for row in rows), default=0.0)
    for row in rows:
        row["support_fraction_of_top"] = (
            float(row["support_score"] / max_support) if max_support > 0 else 0.0
        )

    return rows[:top_k]


def _summarize_candidate_family(candidate_trace):
    """Summarize family-level ambiguity from ranked cancer candidates."""
    if not candidate_trace:
        return {
            "label": None,
            "codes": [],
            "display": None,
            "subtype_clause": None,
        }

    best = candidate_trace[0]
    family = best.get("family_label")
    if family is None:
        return {
            "label": None,
            "codes": [best["code"]],
            "display": None,
            "subtype_clause": None,
        }
    family_group = _CANCER_FAMILY_GROUP.get(family, family)

    family_rows = [
        row
        for row in candidate_trace
        if _CANCER_FAMILY_GROUP.get(row.get("family_label"), row.get("family_label"))
        == family_group
        and row["support_score"]
        >= best["support_score"]
        * TUMOR_PURITY_PARAMETERS["family_scoring"]["family_display_fraction"]
    ]
    family_codes = [row["code"] for row in family_rows]
    if _CANCER_FAMILY_GROUP_CODE_COUNTS.get(family_group, 0) < 2:
        return {
            "label": family_group,
            "codes": family_codes or [best["code"]],
            "display": None,
            "subtype_clause": None,
        }
    display_name = _CANCER_FAMILY_DISPLAY.get(family_group, family_group)
    display = f"{display_name} family"
    subtype_clause = None
    if len(family_codes) >= 2:
        display = f"{display_name} family ({' > '.join(family_codes[:3])})"
        subtype_clause = f"{family_codes[0]} > {family_codes[1]}"
    elif family_codes:
        subtype_clause = family_codes[0]

    return {
        "label": family_group,
        "codes": family_codes,
        "display": display,
        "subtype_clause": subtype_clause,
    }


def _summarize_fit_quality(candidate_trace, signature_stats):
    """Describe whether TCGA references provide a focused subtype fit."""
    if not candidate_trace:
        return {
            "label": "unknown",
            "signature_gap": None,
            "support_ratio": None,
            "message": "No cancer candidates were available.",
        }

    best = candidate_trace[0]
    second = candidate_trace[1] if len(candidate_trace) > 1 else None
    support_ratio = None
    if second is not None and second["support_score"] > 0:
        support_ratio = float(best["support_score"] / second["support_score"])

    top_signature = float(signature_stats[0]["score"]) if signature_stats else 0.0
    reference_idx = min(4, len(signature_stats) - 1) if signature_stats else 0
    reference_signature = (
        float(signature_stats[reference_idx]["score"]) if signature_stats else 0.0
    )
    signature_gap = float(top_signature - reference_signature)

    if signature_gap < 0.05:
        return {
            "label": "weak",
            "signature_gap": signature_gap,
            "support_ratio": support_ratio,
            "message": (
                "Subtype fit is weak: the sample sits in a flat TCGA signature landscape, "
                "so broad family interpretation is more trustworthy than the exact top label."
            ),
        }
    if support_ratio is not None and support_ratio < 1.35:
        return {
            "label": "ambiguous",
            "signature_gap": signature_gap,
            "support_ratio": support_ratio,
            "message": "Top subtype candidates remain close; treat the leading label as provisional.",
        }
    return {
        "label": "focused",
        "signature_gap": signature_gap,
        "support_ratio": support_ratio,
        "message": "The leading TCGA reference is materially separated from alternatives.",
    }


# -------------------- comprehensive summary --------------------


def analyze_sample(df_gene_expr, cancer_type=None, tissue_signal=None):
    """Comprehensive sample composition analysis.

    Returns a dict with all analysis results: cancer type, purity,
    background signatures, MHC status, and narrative interpretation.
    """
    from .plot import (
        _compute_cancer_type_signature_stats,
        resolve_cancer_type,
        CANCER_TYPE_NAMES,
    )

    sample_tpm = _build_sample_tpm_by_symbol(df_gene_expr)

    # 1. Cancer type
    stats = _compute_cancer_type_signature_stats(df_gene_expr)
    default_candidates = [row["code"] for row in stats[:8]]
    if cancer_type:
        cancer_code = resolve_cancer_type(cancer_type)
        candidate_trace = rank_cancer_type_candidates(
            df_gene_expr,
            candidate_codes=[cancer_code] + default_candidates,
            top_k=8,
            tissue_signal=tissue_signal,
        )
    else:
        candidate_trace = rank_cancer_type_candidates(
            df_gene_expr,
            candidate_codes=None,
            top_k=8,
            tissue_signal=tissue_signal,
        )
        cancer_code = (
            candidate_trace[0]["code"] if candidate_trace else stats[0]["code"]
        )
    cancer_name = CANCER_TYPE_NAMES.get(cancer_code, cancer_code)
    candidate_lookup = {row["code"]: row for row in candidate_trace}
    selected_candidate = candidate_lookup.get(cancer_code)
    cancer_score = selected_candidate["support_score"] if selected_candidate else None
    cancer_call_rescue = (
        selected_candidate.get("support_override") if selected_candidate else None
    )
    family_summary = _summarize_candidate_family(candidate_trace)
    fit_quality = _summarize_fit_quality(candidate_trace, stats)

    # 2. Purity
    if selected_candidate is not None:
        purity = selected_candidate["purity_result"]
    else:
        purity = estimate_tumor_purity(df_gene_expr, cancer_type=cancer_code)

    # 3. Residual background signatures
    tissue_score_details = _score_host_tissue_details(sample_tpm, top_n=10)
    tissue_scores = [
        (row["tissue"], row["score"], row["n_genes"]) for row in tissue_score_details
    ]

    # 4. MHC expression
    mhc1, mhc2 = _get_mhc_expression(sample_tpm)

    # 5. Top cancer type matches. ``top_cancers`` is report-facing and
    # normalized to the best candidate so the bar plots read as relative
    # support rather than tiny raw support products.
    top_cancers = [
        (row["code"], row.get("support_fraction_of_top", 0.0)) for row in candidate_trace[:5]
    ]
    top_cancers_raw_support = [
        (row["code"], row["support_score"]) for row in candidate_trace[:5]
    ]
    top_cancer_geomean = [
        (row["code"], row.get("support_geomean", 0.0)) for row in candidate_trace[:5]
    ]
    signature_top_cancers = [(s["code"], s["score"]) for s in stats[:5]]

    return {
        "cancer_type": cancer_code,
        "cancer_name": cancer_name,
        "cancer_score": cancer_score,
        "top_cancers": top_cancers,
        "top_cancers_raw_support": top_cancers_raw_support,
        "top_cancer_geomean": top_cancer_geomean,
        "signature_top_cancers": signature_top_cancers,
        "candidate_trace": candidate_trace,
        "cancer_call_rescue": cancer_call_rescue,
        "family_summary": family_summary,
        "fit_quality": fit_quality,
        "purity": purity,
        "tissue_scores": tissue_scores,
        "tissue_score_details": tissue_score_details,
        "mhc1": mhc1,
        "mhc2": mhc2,
    }


def plot_sample_summary(
    df_gene_expr,
    cancer_type=None,
    sample_mode="auto",
    save_to_filename=None,
    save_dpi=300,
    analysis=None,
):
    """Comprehensive sample composition plot.

    Four-panel figure:
    - Top-left: cancer type identification (bar chart)
    - Top-right: tumor purity and microenvironment composition
    - Bottom-left: residual background signatures (where is the non-tumor signal from?)
    - Bottom-right: MHC class I and II expression

    Pass ``analysis`` to skip the internal :func:`analyze_sample` call
    when the caller has already computed one (issue #84). The CLI's
    ``analyze()`` does this so the plotter renders from the same
    enriched/harmonized ``analysis`` dict the rest of the report
    uses — previously this rerendered from a fresh ``analyze_sample``
    which both cost a second full analysis pass and could silently
    drift from the report.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if analysis is None:
        analysis = analyze_sample(df_gene_expr, cancer_type=cancer_type)
    purity = analysis["purity"]
    cancer_code = analysis["cancer_type"]
    cancer_name = analysis["cancer_name"]
    if sample_mode == "auto":
        try:
            from .decomposition import infer_sample_mode

            sample_mode = infer_sample_mode(
                candidate_rows=analysis.get("candidate_trace"),
                cancer_types=[cancer_code],
                sample_mode="auto",
            )
        except Exception:
            sample_mode = "solid"

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ---- Panel 1: Cancer type identification ----
    ax1 = fig.add_subplot(gs[0, 0])
    top_cancers = analysis["top_cancers"]
    codes = [c for c, s in top_cancers]
    scores = [s for c, s in top_cancers]
    colors = ["#2166ac" if c == cancer_code else "#92c5de" for c in codes]
    y = np.arange(len(codes))
    ax1.barh(y, scores, color=colors, edgecolor="none", height=0.6)
    for i, (code, score) in enumerate(top_cancers):
        from .plot import CANCER_TYPE_NAMES as CTN

        label = f"{code} ({CTN.get(code, '')})"
        score_label = f"{score:.2f}"
        ax1.text(score + 0.01, i, score_label, va="center", fontsize=9)
        ax1.text(-0.01, i, label, va="center", ha="right", fontsize=9)
    ax1.set_yticks([])
    ax1.set_xlim(0, 1.1)
    ax1.set_xlabel("Normalized support (top = 1.0)", fontsize=10)
    fit_quality = analysis.get("fit_quality", {})
    fit_label = fit_quality.get("label")
    title = "Cancer type hypotheses"
    if fit_label in {"weak", "ambiguous"}:
        title += f" ({fit_label} fit)"
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.invert_yaxis()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    label_options = [top_cancers[0][0]] if top_cancers else []
    if len(top_cancers) >= 2 and fit_label in {"weak", "ambiguous"}:
        label_options.append(top_cancers[1][0])
    if fit_label or label_options:
        lines = []
        if fit_label:
            lines.append(f"Fit: {fit_label}")
        if len(label_options) == 2:
            lines.append(f"Possible labels: {label_options[0]} or {label_options[1]}")
        elif len(label_options) == 1:
            lines.append(f"Lead label: {label_options[0]}")
        fusion_note = _fusion_plot_note(analysis)
        if fusion_note:
            lines.append(fusion_note)
        ax1.text(
            0.02,
            0.02,
            "\n".join(lines),
            transform=ax1.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
        )

    # ---- Panel 2: Purity and microenvironment ----
    ax2 = fig.add_subplot(gs[0, 1])
    overall = purity["overall_estimate"]
    stromal_enr = purity["components"]["stromal"]["enrichment"]
    immune_enr = purity["components"]["immune"]["enrichment"]

    # Stacked composition bar
    tumor_frac = overall if overall else 0
    stromal_frac = min(
        stromal_enr / (stromal_enr + immune_enr + 0.001), 1 - tumor_frac
    ) * (1 - tumor_frac)
    immune_frac = 1 - tumor_frac - stromal_frac

    if sample_mode == "heme":
        main_label = "Malignant-like"
        stromal_label = "Stromal context"
        immune_label = "Immune context"
        comp_title = "Heme Composition Context"
        comp_xlabel = "Estimated fraction / context (%)"
        detail_prefix = "Malignant-lineage fraction proxy"
    elif sample_mode == "pure":
        main_label = "Dominant population"
        stromal_label = "Residual stromal"
        immune_label = "Residual immune"
        comp_title = "Population Coherence"
        comp_xlabel = "Estimated population / context (%)"
        detail_prefix = "Population consistency"
    else:
        main_label = "Tumor"
        stromal_label = "Stromal"
        immune_label = "Immune"
        comp_title = "Sample Composition"
        comp_xlabel = "Estimated composition (%)"
        detail_prefix = "Tumor purity"

    ax2.barh(
        0,
        tumor_frac * 100,
        color="#2166ac",
        height=0.5,
        label=f"{main_label} ({tumor_frac:.0%})",
    )
    ax2.barh(
        0,
        stromal_frac * 100,
        left=tumor_frac * 100,
        color="#d6604d",
        height=0.5,
        label=f"{stromal_label} ({stromal_frac:.0%})",
    )
    ax2.barh(
        0,
        immune_frac * 100,
        left=(tumor_frac + stromal_frac) * 100,
        color="#4393c3",
        height=0.5,
        label=f"{immune_label} ({immune_frac:.0%})",
    )

    ax2.set_xlim(0, 100)
    ax2.set_yticks([])
    ax2.set_xlabel(comp_xlabel, fontsize=10)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Add text annotations below
    lo = purity["overall_lower"]
    hi = purity["overall_upper"]
    details = [
        f"{detail_prefix}: {overall:.0%}"
        + (f" [{lo:.0%}–{hi:.0%}]" if lo is not None else "")
    ]
    if sample_mode == "solid":
        details.extend(
            [
                f"Stromal enrichment: {render_fold(stromal_enr)} vs TCGA {cancer_code}",
                f"Immune enrichment: {render_fold(immune_enr)} vs TCGA {cancer_code}",
                f"TCGA {cancer_code} median purity: {purity['tcga_median_purity']:.0%}",
            ]
        )
    elif sample_mode == "heme":
        details.extend(
            [
                f"Stromal context: {render_fold(stromal_enr)} vs TCGA {cancer_code}",
                f"Immune context: {render_fold(immune_enr)} vs TCGA {cancer_code}",
                "Interpretation: lineage/background context, not a strict tumor-vs-immune split",
            ]
        )
    else:
        details.extend(
            [
                f"Residual stromal context: {render_fold(stromal_enr)} vs TCGA {cancer_code}",
                f"Residual immune context: {render_fold(immune_enr)} vs TCGA {cancer_code}",
                "Interpretation: consistency vs likely tissue-of-origin profile, not bulk admixture",
            ]
        )
    for i, txt in enumerate(details):
        ax2.text(
            0,
            -0.6 - i * 0.5,
            txt,
            transform=ax2.transData,
            fontsize=9,
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
            if i == 0
            else None,
        )

    ax2.set_ylim(-3.5, 0.8)
    ax2.set_title(comp_title, fontsize=12, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # ---- Panel 3: Background tissue signatures ----
    ax3 = fig.add_subplot(gs[1, 0])
    tissue_scores = analysis["tissue_scores"]
    if tissue_scores:
        tissues = [t.replace("_", " ").title() for t, s, n in tissue_scores]
        t_scores = [s for t, s, n in tissue_scores]
        matched = CANCER_TO_TISSUE.get(cancer_code, "").replace("_", " ").title()
        t_colors = []
        for t, s, n in tissue_scores:
            tname = t.replace("_", " ").title()
            if tname == matched:
                t_colors.append("#2166ac")  # tumor origin tissue
            elif s > 0.7:
                t_colors.append("#b2182b")  # strong non-tumor signal
            else:
                t_colors.append("#92c5de")  # background
        y = np.arange(len(tissues))
        ax3.barh(y, t_scores, color=t_colors, edgecolor="none", height=0.6)
        ax3.set_yticks(y)
        ax3.set_yticklabels(tissues, fontsize=9)
        for i, (t, s, n) in enumerate(tissue_scores):
            ax3.text(s + 0.01, i, f"{s:.3f}", va="center", fontsize=8)
        ax3.set_xlim(0, 1.1)
        ax3.set_xlabel("Background signature score", fontsize=10)
        ax3.invert_yaxis()
        # Legend
        from matplotlib.patches import Patch

        ax3.legend(
            handles=[
                Patch(color="#2166ac", label=f"Expected origin family ({matched})"),
                Patch(color="#b2182b", label="Strong residual background"),
                Patch(color="#92c5de", label="Background"),
            ],
            loc="lower right",
            fontsize=7,
            framealpha=0.9,
        )
    if sample_mode == "heme":
        bg_title = (
            "Lineage / Background Context\n(residual hematopoietic and tissue programs)"
        )
    elif sample_mode == "pure":
        bg_title = "Residual Background Check\n(contamination / off-target context)"
    else:
        bg_title = "Background Tissue Signatures\n(residual non-tumor context)"
    ax3.set_title(bg_title, fontsize=12, fontweight="bold")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # ---- Panel 4: MHC expression ----
    ax4 = fig.add_subplot(gs[1, 1])
    mhc1 = analysis["mhc1"]
    mhc2 = analysis["mhc2"]

    all_genes = list(mhc1.keys()) + list(mhc2.keys())
    all_tpms = [mhc1.get(g, 0) for g in mhc1] + [mhc2.get(g, 0) for g in mhc2]
    y = np.arange(len(all_genes))

    # Color by class
    n1 = len(mhc1)
    colors_mhc = ["#2166ac"] * n1 + ["#b2182b"] * len(mhc2)
    ax4.barh(y, all_tpms, color=colors_mhc, edgecolor="none", height=0.6, alpha=0.8)

    ax4.set_yticks(y)
    ax4.set_yticklabels(all_genes, fontsize=9)
    for i, tpm in enumerate(all_tpms):
        if tpm > 0:
            ax4.text(
                tpm + max(all_tpms) * 0.02, i, f"{tpm:.0f}", va="center", fontsize=8
            )

    # Divider between class I and II
    ax4.axhline(y=n1 - 0.5, color="#cccccc", linewidth=0.8, linestyle="--")
    ax4.text(
        max(all_tpms) * 0.95,
        n1 / 2 - 0.5,
        "Class I",
        ha="right",
        va="center",
        fontsize=9,
        color="#2166ac",
        fontweight="bold",
    )
    ax4.text(
        max(all_tpms) * 0.95,
        n1 + len(mhc2) / 2 - 0.5,
        "Class II",
        ha="right",
        va="center",
        fontsize=9,
        color="#b2182b",
        fontweight="bold",
    )

    ax4.set_xlabel("TPM", fontsize=10)
    ax4.invert_yaxis()
    ax4.set_title("MHC antigen presentation", fontsize=12, fontweight="bold")
    ax4.text(
        0.02,
        0.02,
        _hla_plot_note(analysis),
        transform=ax4.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88),
    )
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # ---- Main title ----
    if sample_mode == "heme":
        mode_title = "hematologic / lymphoid bulk"
    elif sample_mode == "pure":
        mode_title = "pure population / cell culture"
    else:
        mode_title = "solid tumor bulk"
    fig.suptitle(
        f"Sample composition analysis — {cancer_name} ({cancer_code}) [{mode_title}]",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, analysis


def _fusion_plot_note(analysis) -> str:
    inference = analysis.get("fusion_report_scope_inference") or {}
    if inference:
        fusion = inference.get("fusion") or {}
        pair = str(fusion.get("pair") or inference.get("expected_pair") or "").strip()
        label = str(inference.get("label") or inference.get("cancer_type") or "").strip()
        expected = str(inference.get("expected_pair") or "").strip()
        parts = [f"Fusion-supported: {label or 'rare cancer'}"]
        if pair:
            parts.append(pair)
        if expected and expected != pair:
            parts.append(f"rule {expected}")
        return "; ".join(parts)

    findings = analysis.get("fusion_findings") or []
    if findings:
        top = findings[0]
        fusion = top.get("fusion") or {}
        pair = str(fusion.get("pair") or top.get("expected_pair") or "").strip()
        label = str(top.get("label") or "curated fusion finding").strip()
        return f"Fusion finding: {pair} ({label}; no report-scope promotion)"

    rare = analysis.get("rare_report_scope_inference") or {}
    if rare and not analysis.get("fusion_inputs_supplied"):
        surrogate = str(rare.get("surrogate") or "RNA marker").strip()
        return f"RNA rare-cancer hint from {surrogate}; ask for fusion/IHC/FISH"
    return ""


def _hla_plot_note(analysis) -> str:
    constraints = analysis.get("analysis_constraints") or {}
    hla_types = constraints.get("hla_types") or []
    if hla_types:
        return "Supplied HLA type(s): " + ", ".join(str(item) for item in hla_types)
    return "HLA genotype not supplied; MHC RNA is not allele typing"


def plot_cancer_type_hypotheses(analysis, save_to_filename=None, save_dpi=300):
    """Standalone: cancer-type hypothesis ranking and support-factor audit."""
    import matplotlib.pyplot as plt
    from .plot import CANCER_TYPE_NAMES as CTN

    cancer_code = analysis["cancer_type"]
    top_cancers = analysis["top_cancers"]
    fit_quality = analysis.get("fit_quality", {})
    candidate_by_code = {
        row["code"]: row
        for row in analysis.get("candidate_trace", [])
        if isinstance(row, dict) and row.get("code")
    }

    n_rows = max(1, len(top_cancers))
    fig, (ax, ax_factors) = plt.subplots(
        1,
        2,
        figsize=(13.5, max(3.8, 0.46 * n_rows + 1.3)),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    codes = [c for c, s in top_cancers]
    scores = [s for c, s in top_cancers]
    top_code = codes[0] if codes else cancer_code
    colors = []
    for c in codes:
        if c == cancer_code:
            colors.append("#2166ac")
        elif c == top_code:
            colors.append("#f4a340")
        else:
            colors.append("#92c5de")
    y = np.arange(len(codes))
    ax.barh(y, scores, color=colors, edgecolor="none", height=0.6)
    for i, (code, score) in enumerate(top_cancers):
        label = f"{code} ({CTN.get(code, '')})"
        score_label = f"{score:.2f}"
        ax.text(score + 0.01, i, score_label, va="center", fontsize=9)
        ax.text(-0.01, i, label, va="center", ha="right", fontsize=9)
        if code == cancer_code and code != top_code:
            ax.text(
                min(score + 0.22, 1.05),
                i,
                "report call",
                va="center",
                fontsize=8,
                color="#2166ac",
                fontweight="bold",
            )
    ax.set_yticks([])
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Normalized support (top = 1.0)")
    fit_label = fit_quality.get("label")
    title = "Cancer type hypotheses"
    if top_code != cancer_code:
        title += f" — report call {cancer_code}; top RNA-support {top_code}"
    if fit_label in {"weak", "ambiguous"}:
        title += f" ({fit_label} fit)"
    ax.set_title(title, fontweight="bold")

    factor_specs = [
        ("Signature", "signature_score"),
        ("Purity", "purity_estimate"),
        ("Lineage", "lineage_detection_fraction"),
        ("Family", "family_factor"),
        ("Overall", "support_fraction_of_top"),
    ]
    factor_values = []
    for code in codes:
        row = candidate_by_code.get(code, {})
        values = []
        for _label, key in factor_specs:
            value = row.get(key)
            if value is None and key == "support_fraction_of_top":
                value = dict(top_cancers).get(code, 0.0)
            try:
                value = float(value)
            except Exception:
                value = 0.0
            values.append(max(0.0, min(1.0, value)))
        factor_values.append(values)
    factor_array = (
        np.asarray(factor_values, dtype=float)
        if factor_values
        else np.zeros((0, len(factor_specs)))
    )
    if len(codes):
        im = ax_factors.imshow(
            factor_array,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="YlGnBu",
        )
        ax_factors.set_xticks(np.arange(len(factor_specs)))
        ax_factors.set_xticklabels([label for label, _key in factor_specs], fontsize=8)
        ax_factors.set_yticks(np.arange(len(codes)))
        ax_factors.set_yticklabels(codes, fontsize=8.5)
        for i, code in enumerate(codes):
            for j, value in enumerate(factor_array[i]):
                ax_factors.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if value > 0.58 else "#333333",
                )
            if code == cancer_code:
                ax_factors.scatter(
                    -0.62,
                    i,
                    marker=">",
                    s=60,
                    color="#2166ac",
                    clip_on=False,
                    zorder=5,
                )
        ax_factors.set_title("Support factors", fontweight="bold")
        ax_factors.tick_params(axis="both", length=0)
        for spine in ax_factors.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, ax=ax_factors, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
    else:
        ax_factors.axis("off")

    fusion_note = _fusion_plot_note(analysis)
    if fusion_note:
        ax.text(
            0.02,
            0.02,
            fusion_note,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88),
        )
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.text(
        0.5,
        0.965,
        "The report call can be externally supplied or QC-rescued; support factors show why a competing RNA context ranked highly.",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.93])
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    return fig


def plot_background_tissues(analysis, save_to_filename=None, save_dpi=300):
    """Standalone: background tissue signature bar chart."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cancer_code = analysis["cancer_type"]
    tissue_scores = analysis["tissue_scores"]
    if not tissue_scores:
        return None

    matched = CANCER_TO_TISSUE.get(cancer_code, "").replace("_", " ").title()
    tissues = [t.replace("_", " ").title() for t, s, n in tissue_scores]
    t_scores = [s for t, s, n in tissue_scores]
    t_colors = []
    for t, s, n in tissue_scores:
        tname = t.replace("_", " ").title()
        if tname == matched:
            t_colors.append("#2166ac")
        elif s > 0.7:
            t_colors.append("#b2182b")
        else:
            t_colors.append("#92c5de")

    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(tissues))))
    y = np.arange(len(tissues))
    ax.barh(y, t_scores, color=t_colors, edgecolor="none", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(tissues, fontsize=9)
    for i, (t, s, n) in enumerate(tissue_scores):
        ax.text(s + 0.01, i, f"{s:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Background signature score")
    ax.invert_yaxis()
    ax.legend(
        handles=[
            Patch(color="#2166ac", label=f"Expected origin ({matched})"),
            Patch(color="#b2182b", label="Strong residual"),
            Patch(color="#92c5de", label="Background"),
        ],
        loc="lower right",
        fontsize=8,
    )
    ax.set_title("Background Tissue Signatures", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    return fig


def plot_mhc_expression(analysis, save_to_filename=None, save_dpi=300):
    """Standalone: MHC class I and II expression bar chart."""
    import matplotlib.pyplot as plt

    mhc1 = analysis["mhc1"]
    mhc2 = analysis["mhc2"]
    all_genes = list(mhc1.keys()) + list(mhc2.keys())
    all_tpms = [mhc1.get(g, 0) for g in mhc1] + [mhc2.get(g, 0) for g in mhc2]
    n1 = len(mhc1)
    colors_mhc = ["#2166ac"] * n1 + ["#b2182b"] * len(mhc2)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(all_genes))))
    y = np.arange(len(all_genes))
    ax.barh(y, all_tpms, color=colors_mhc, edgecolor="none", height=0.6, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(all_genes, fontsize=9)
    for i, tpm in enumerate(all_tpms):
        if tpm > 0:
            ax.text(
                tpm + max(all_tpms) * 0.02, i, f"{tpm:.0f}", va="center", fontsize=8
            )
    ax.axhline(y=n1 - 0.5, color="#cccccc", linewidth=0.8, linestyle="--")
    ax.set_xlabel("TPM")
    ax.invert_yaxis()
    ax.set_title("MHC Antigen Presentation", fontweight="bold")
    ax.text(
        0.02,
        0.02,
        _hla_plot_note(analysis),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    return fig
