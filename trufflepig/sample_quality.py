# Licensed under the Apache License, Version 2.0

"""Sample quality assessment — degradation, FFPE artifacts, and cell culture detection.

Flags quality issues from a TPM expression matrix without needing raw reads
or BAM files.  Three independent detectors:

1. **RNA degradation / FFPE**: mitochondrial-encoded transcripts are short
   and abundant, so their fraction rises in degraded samples.  Ribosomal
   protein genes (also short) shift up relative to other genes.  Together,
   elevated MT + RP fractions indicate degradation / FFPE.

2. **Cell culture stress**: heat-shock, glycolysis, proliferation, and ER
   stress genes are consistently upregulated in culture-adapted cells
   (Yu et al., Nature Communications 2019).

3. **TME absence**: near-zero immune + stromal signal distinguishes cell
   lines (or immune-desert tumors) from typical tissue biopsies.
"""

from __future__ import annotations

import math

import numpy as np


# ── Gene lists ───────────────────────────────────────────────────────────
#
# Gene panels live in `data/*.csv` and are loaded via `gene_sets_cancer`.
# Module-level aliases are kept so existing imports keep working; the CSV
# is the source of truth.

from pirlygenes.gene_sets_cancer import (

    culture_stress_gene_names, degradation_gene_pairs, mitochondrial_gene_names, tme_marker_gene_names,

)
from .sample_context import library_prep_display_label


# Mitochondrial-encoded gene symbols — short transcripts that survive
# degradation, so their fraction rises in FFPE / degraded RNA.
_MT_GENES = sorted(mitochondrial_gene_names())

# Culture-stress genes — upregulated in cell lines vs primary tissue
# (Yu et al., Nat Commun 2019). Split across HSP / glycolysis /
# proliferation / ER_stress / oxidative_stress / glutamine categories in
# the CSV.
_CULTURE_STRESS_UP = sorted(culture_stress_gene_names())

# TME markers — absent in cell lines, present in tissue biopsies.
# Grouped by Cell_Type (T_cell / B_cell / myeloid / fibroblast /
# endothelial) in the CSV.
_TME_MARKERS = sorted(tme_marker_gene_names())

# Ribosomal protein gene prefixes — fraction rises in degraded samples.
# Kept as a literal prefix tuple because it's used for `startswith` checks
# on all gene symbols rather than an enumerated panel.
_RIBOSOMAL_PROTEIN_PREFIXES = ("RPL", "RPS")

# Matched-expression gene pairs for the transcript-length degradation
# index. See `data/degradation-gene-pairs.csv` for the full schema — each
# row is (short_symbol, short_id, long_symbol, long_id, expected_ratio).
# The short genes are in the bottom-10% of coding-exon length (<1.2 kb),
# long genes in the top-10% (>6.9 kb); both ubiquitously expressed
# (>10 nTPM median across 83 reference columns); expected ratios have
# CV < 0.35 across fresh/frozen tissues. Generated from pyensembl
# release 110 (GRCh38).
_DEGRADATION_GENE_PAIRS = degradation_gene_pairs()


# ── Thresholds ───────────────────────────────────────────────────────────
# Calibrated against TCGA fresh/frozen cohort medians (33 types).

_MT_EXPECTED_MISSING_PREPS = frozenset({"poly_a", "exome_capture"})


QUALITY_THRESHOLDS = {
    # Tissue-matched degradation: flag when sample's MT or RP fraction
    # exceeds the matched tissue baseline by this multiplicative factor.
    # The baseline comes from nTPM normal-tissue profiles, so the threshold
    # is relative, not absolute — heart muscle at 69% MT won't trigger,
    # but heart muscle at 140% MT (2× baseline) will.
    "degradation_fold_moderate": 2.0,
    "degradation_fold_severe": 3.0,
    # Fallback absolute thresholds when no tissue match is available.
    "mt_fraction_fallback_elevated": 0.50,
    "rp_fraction_fallback_elevated": 0.30,
    # Culture stress: z-score of culture-UP genes above expressed-gene
    # median.  TCGA tissues are typically 0.5–2.0; cell lines push 3+.
    "culture_stress_elevated": 2.5,
    "culture_stress_high": 3.5,
    # TME absence: mean TPM of TME markers below this is suspicious.
    "tme_absent_threshold": 5.0,
}


def _compute_tissue_baselines():
    """Precompute MT/RP fraction baselines for each of the 50 normal tissues.

    Returns dict of {tissue_name: {"mt_fraction": float, "rp_fraction": float}}.
    """
    from trufflepig.reference import pan_cancer_expression
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    symbols = ref["Symbol"].tolist()
    mt_set = set(_MT_GENES)
    ntpm_cols = [c for c in ref.columns if c.startswith("nTPM_")]

    baselines = {}
    for col in ntpm_cols:
        tissue = col.replace("nTPM_", "")
        vals = ref[col].astype(float).values
        total = sum(
            v
            for v, s in zip(vals, symbols)
            if not (np.isnan(v) if isinstance(v, float) else False) and v > 0
        )
        if total <= 0:
            continue
        mt_sum = sum(
            v
            for v, s in zip(vals, symbols)
            if s in mt_set
            and not (np.isnan(v) if isinstance(v, float) else False)
            and v > 0
        )
        rp_sum = sum(
            v
            for v, s in zip(vals, symbols)
            if any(s.startswith(p) for p in _RIBOSOMAL_PROTEIN_PREFIXES)
            and not (np.isnan(v) if isinstance(v, float) else False)
            and v > 0
        )
        mt_frac = mt_sum / total
        non_mt = total - mt_sum
        rp_frac = rp_sum / non_mt if non_mt > 0 else 0.0
        baselines[tissue] = {
            "mt_fraction": round(mt_frac, 4),
            "rp_fraction": round(rp_frac, 4),
        }
    return baselines


_TISSUE_BASELINES_CACHE = None


def _get_tissue_baselines():
    """Return cached tissue baselines (computed once)."""
    global _TISSUE_BASELINES_CACHE
    if _TISSUE_BASELINES_CACHE is None:
        _TISSUE_BASELINES_CACHE = _compute_tissue_baselines()
    return _TISSUE_BASELINES_CACHE


def _gene_tpm_lookup(sample_tpm_by_symbol, genes):
    """Return (found_values, n_found) for a gene list, skipping NaN."""
    vals = []
    for gene in genes:
        tpm = sample_tpm_by_symbol.get(gene)
        if tpm is not None and not math.isnan(tpm) and tpm > 0:
            vals.append(float(tpm))
    return vals, len(vals)


def assess_sample_quality(df_gene_expr, tissue_scores=None, library_prep=None):
    """Assess sample quality from a TPM expression matrix.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Expression data with gene identifiers and TPM values.
    tissue_scores : list of (tissue, score, n_genes), optional
        Background tissue scores from analyze_sample().  Used to select
        the tissue-matched baseline for MT/RP comparison.  If None,
        falls back to absolute thresholds.
    library_prep : str, optional
        Library prep inferred by :class:`SampleContext` (e.g.
        ``"exome_capture"``, ``"poly_a"``, ``"ribo_depleted"``,
        ``"total_rna"``).  When set to a prep that legitimately strips
        mitochondrial transcripts (``exome_capture`` / ``poly_a``) and the
        transcript-length pair index is available, the MT-fraction
        warning is suppressed and the degradation level is **not**
        overwritten with ``"unknown"`` (#77) — preservation has already
        been called from the length-pair signal.

    Returns
    -------
    dict with keys:
        degradation : dict
            mt_fraction, rp_fraction, matched_tissue, mt_fold, rp_fold,
            level, message
        culture : dict
            stress_score, tme_mean_tpm, tme_absent, level, message,
            top_stress_genes (list of (gene, tpm))
        flags : list of str
            Human-readable quality flag strings for console output.
        has_issues : bool
            True if any quality concern was detected.
    """
    from .common import build_sample_tpm_by_symbol

    sample_tpm = build_sample_tpm_by_symbol(df_gene_expr)

    # Filter NaN values from the TPM dict (some reference genes have NaN
    # in TCGA cohorts where they were not measured, e.g. BCR/TCR loci).
    clean_tpm = {g: v for g, v in sample_tpm.items() if not math.isnan(v)}
    total_tpm = sum(clean_tpm.values())

    if total_tpm <= 0:
        return {
            "degradation": {"level": "unknown", "message": "No expression detected"},
            "culture": {"level": "unknown", "message": "No expression detected"},
            "flags": ["No expression detected — cannot assess quality"],
            "has_issues": True,
        }

    thresholds = QUALITY_THRESHOLDS

    # ── 1. RNA degradation / FFPE ────────────────────────────────────────

    mt_vals, n_mt = _gene_tpm_lookup(clean_tpm, _MT_GENES)
    mt_total = sum(mt_vals)
    mt_fraction = mt_total / total_tpm

    rp_total = sum(
        tpm
        for gene, tpm in clean_tpm.items()
        if any(gene.startswith(pfx) for pfx in _RIBOSOMAL_PROTEIN_PREFIXES)
    )
    # RP fraction relative to non-MT expression — prevents MT inflation
    # from diluting the RP signal (in degraded samples both rise).
    non_mt_total = total_tpm - mt_total
    rp_fraction = rp_total / non_mt_total if non_mt_total > 0 else 0.0

    # Transcript-length degradation index from matched gene pairs.
    # Each pair has a known expected ratio (long/short) in fresh tissue.
    # In FFPE, the observed ratio drops because long transcripts degrade
    # faster.  We compute observed/expected for each pair and take the
    # median — values near 1.0 are normal, <<1.0 indicates degradation.
    pair_ratios = []
    n_long = 0
    for short_gene, long_gene, expected in _DEGRADATION_GENE_PAIRS:
        s_tpm = clean_tpm.get(short_gene)
        l_tpm = clean_tpm.get(long_gene)
        if s_tpm and s_tpm > 1 and l_tpm is not None:
            observed = l_tpm / s_tpm
            pair_ratios.append(observed / expected if expected > 0 else 1.0)
            n_long += 1
    long_short_ratio = float(np.median(pair_ratios)) if pair_ratios else None

    # Tissue-matched degradation scoring.  Compare MT/RP to the expected
    # baseline for the matched normal tissue.  This avoids false positives
    # on mitochondria-rich tissues (heart=69% MT is normal biology).
    baselines = _get_tissue_baselines()
    matched_tissue = None
    baseline_mt = None
    baseline_rp = None
    if tissue_scores:
        preferred_tissues = set()
        try:
            from .tumor_purity import _HOST_SITE_BACKGROUND_TISSUES

            preferred_tissues = set(_HOST_SITE_BACKGROUND_TISSUES)
        except Exception:
            preferred_tissues = set()

        ordered_tissues = list(tissue_scores)
        if preferred_tissues:
            ordered_tissues = [
                row for row in tissue_scores if row[0] in preferred_tissues
            ] + [row for row in tissue_scores if row[0] not in preferred_tissues]

        for tissue, _score, _n in ordered_tissues:
            if tissue in baselines:
                matched_tissue = tissue
                baseline_mt = baselines[tissue]["mt_fraction"]
                baseline_rp = baselines[tissue]["rp_fraction"]
                break

    # Combine three signals: transcript-length pairs (primary), tissue-
    # matched MT fold (secondary), and tissue-matched RP fold (tertiary).
    # The pair ratio is the most specific FFPE indicator because it's
    # tissue-independent; MT/RP folds add confirmation.

    if baseline_mt is not None and baseline_mt > 0.01:
        mt_fold = mt_fraction / baseline_mt
        rp_fold = rp_fraction / baseline_rp if baseline_rp > 0.01 else 1.0
    else:
        mt_fold = None
        rp_fold = None

    fold_severe = thresholds["degradation_fold_severe"]
    fold_moderate = thresholds["degradation_fold_moderate"]

    # Transcript-length pair ratio is the primary signal.
    # Values near 1.0 are normal; <0.5 is moderate, <0.3 severe.
    # In TCGA fresh/frozen + normal tissue, the index ranges from 0.37 to
    # 1.40 (p5=0.43).  Values below 0.30 are well outside the normal range.
    pair_severe = long_short_ratio is not None and long_short_ratio < 0.20
    pair_moderate = long_short_ratio is not None and long_short_ratio < 0.30
    # MT/RP tissue-matched folds as confirmation
    mt_rp_moderate = (
        mt_fold is not None
        and rp_fold is not None
        and mt_fold > fold_moderate
        and rp_fold > fold_moderate
    )
    mt_rp_severe = (
        mt_fold is not None
        and rp_fold is not None
        and mt_fold > fold_severe
        and rp_fold > fold_severe
    )

    ratio_str = f"{long_short_ratio:.2f}" if long_short_ratio is not None else "n/a"
    tissue_str = f" vs {matched_tissue}" if matched_tissue else ""

    if pair_severe or (pair_moderate and mt_rp_moderate) or mt_rp_severe:
        deg_level = "severe"
        deg_msg = (
            f"Severe RNA degradation (length-pair index={ratio_str}"
            + (f", MT {mt_fold:.1f}×, RP {rp_fold:.1f}×{tissue_str}" if mt_fold else "")
            + "). Consistent with FFPE or heavily degraded RNA."
        )
    elif pair_moderate or mt_rp_moderate:
        deg_level = "moderate"
        deg_msg = (
            f"Moderate RNA degradation (length-pair index={ratio_str}"
            + (f", MT {mt_fold:.1f}×, RP {rp_fold:.1f}×{tissue_str}" if mt_fold else "")
            + "). May indicate FFPE, delayed processing, or partial degradation."
        )
    else:
        deg_level = "normal"
        if matched_tissue:
            deg_msg = (
                f"Within normal range{tissue_str} (length-pair index={ratio_str}"
                + (f", MT {mt_fold:.1f}×, RP {rp_fold:.1f}×" if mt_fold else "")
                + ")."
            )
        else:
            deg_msg = (
                f"No degradation signal (length-pair index={ratio_str}, "
                f"MT={mt_fraction:.1%}, RP={rp_fraction:.1%})."
            )

    degradation = {
        "mt_fraction": round(mt_fraction, 4),
        "rp_fraction": round(rp_fraction, 4),
        "long_short_ratio": round(long_short_ratio, 3)
        if long_short_ratio is not None
        else None,
        "n_mt_found": n_mt,
        "n_long_found": n_long,
        "matched_tissue": matched_tissue,
        "baseline_mt": baseline_mt,
        "baseline_rp": baseline_rp,
        "mt_fold": round(mt_fold, 2) if mt_fold is not None else None,
        "rp_fold": round(rp_fold, 2) if rp_fold is not None else None,
        "level": deg_level,
        "message": deg_msg,
    }

    # ── 2. Cell culture / stress signature ───────────────────────────────

    stress_tpms = []
    top_stress = []
    for gene in _CULTURE_STRESS_UP:
        tpm = clean_tpm.get(gene)
        if tpm is not None and tpm > 0:
            stress_tpms.append(float(tpm))
            top_stress.append((gene, float(tpm)))

    # Compute stress score as mean log2(TPM+1) of culture genes, compared
    # to the sample's overall median log2(TPM+1) for expressed genes.
    if stress_tpms:
        expressed_vals = np.array([v for v in clean_tpm.values() if v > 1.0])
        if len(expressed_vals) > 10:
            sample_median_log = float(np.median(np.log2(expressed_vals + 1)))
            sample_std_log = float(np.std(np.log2(expressed_vals + 1)))
        else:
            sample_median_log = 1.0
            sample_std_log = 1.0
        stress_mean_log = float(np.mean(np.log2(np.array(stress_tpms) + 1)))
        stress_score = (stress_mean_log - sample_median_log) / max(sample_std_log, 0.1)
    else:
        stress_score = 0.0

    top_stress.sort(key=lambda x: -x[1])
    top_stress = top_stress[:5]

    # TME signal: mean TPM of TME markers
    tme_vals, n_tme = _gene_tpm_lookup(clean_tpm, _TME_MARKERS)
    tme_mean = float(np.mean(tme_vals)) if tme_vals else 0.0

    tme_absent = tme_mean < thresholds["tme_absent_threshold"]
    stress_elevated = stress_score > thresholds["culture_stress_elevated"]
    stress_high = stress_score > thresholds["culture_stress_high"]

    if tme_absent and stress_high:
        culture_level = "likely_cell_line"
        culture_msg = (
            f"Expression pattern is consistent with a cell line: "
            f"TME markers absent (mean={tme_mean:.1f} TPM), "
            f"culture-stress signature elevated (z={stress_score:.1f}). "
            f"Top stress genes: {', '.join(f'{g}={t:.0f}' for g, t in top_stress[:3])}."
        )
    elif tme_absent and stress_elevated:
        culture_level = "possible_cell_line"
        culture_msg = (
            f"Possible cell line or sorted population: "
            f"TME markers absent (mean={tme_mean:.1f} TPM), "
            f"moderately elevated culture stress (z={stress_score:.1f})."
        )
    elif tme_absent:
        culture_level = "tme_absent"
        culture_msg = (
            f"TME markers absent (mean={tme_mean:.1f} TPM) but culture stress "
            f"is not elevated (z={stress_score:.1f}). Could be immune-desert "
            "tumor, sorted population, or high-purity sample."
        )
    elif stress_elevated:
        culture_level = "stress_only"
        culture_msg = (
            f"Culture-stress genes are elevated (z={stress_score:.1f}) but "
            f"TME markers are present (mean={tme_mean:.1f} TPM). "
            "Likely reflects proliferative/stressed tumor biology, not culture."
        )
    else:
        culture_level = "normal"
        culture_msg = (
            f"No cell culture signal (stress z={stress_score:.1f}, "
            f"TME mean={tme_mean:.1f} TPM)."
        )

    culture = {
        "stress_score": round(stress_score, 2),
        "tme_mean_tpm": round(tme_mean, 1),
        "tme_absent": tme_absent,
        "top_stress_genes": top_stress,
        "n_tme_found": n_tme,
        "level": culture_level,
        "message": culture_msg,
    }

    # ── Build flags ──────────────────────────────────────────────────────

    flags = []
    has_issues = False

    # Suspicious MT fraction: 0% MT is biologically implausible (real bulk
    # samples range 5–70% depending on tissue).  0% usually means the MT
    # genes were filtered upstream (some salmon/kallisto pipelines drop
    # MT-* contigs) or use a different symbol convention.  Flag this so
    # the user knows the degradation signal is not reliable.
    mt_near_zero = n_mt == 0 or mt_fraction < 0.005
    prep_explains_mt = library_prep in _MT_EXPECTED_MISSING_PREPS
    have_pair_signal = long_short_ratio is not None
    if mt_near_zero and prep_explains_mt and have_pair_signal:
        # #77: RNA capture / poly-A legitimately depress MT-rRNA or MT
        # fraction. The length-pair index still gives a valid preservation
        # call, so don't clobber deg_level to "unknown"; emit an
        # informational flag instead of a false alarm.
        prep_label = library_prep_display_label(library_prep)
        flags.append(
            f"MT fraction {mt_fraction:.1%} — consistent with {prep_label} "
            "library prep; degradation assessed from length-pair index"
        )
    elif mt_near_zero:
        has_issues = True
        # n_mt==0 and n_mt>0-but-low-fraction are biologically distinct
        # states. "Filtered or renamed" only applies when the MT symbols
        # are genuinely absent from the quant table — if 13/15 MT genes
        # are in the input at low TPM, they weren't filtered; the sample
        # just has anomalously low MT share (aggressive MT depletion, a
        # capture panel that under-samples MT contigs, or a genuine
        # low-MT tissue).
        if n_mt == 0:
            flags.append(
                f"Suspicious MT fraction: {mt_fraction:.1%} "
                f"(n_mt_found=0/{len(_MT_GENES)}) — mitochondrial genes "
                "missing from the quant table (filtered or renamed "
                "upstream); degradation signal is unreliable"
            )
        else:
            flags.append(
                f"Low MT fraction: {mt_fraction:.1%} "
                f"(n_mt_found={n_mt}/{len(_MT_GENES)}, genes present but "
                "contribute minimal TPM) — MT-based degradation signal "
                "has reduced discriminative power"
            )
        degradation["level"] = "unknown"
        degradation["message"] = (
            "MT fraction too low for reliable degradation assessment "
            f"(n_mt_found={n_mt}/{len(_MT_GENES)}, fraction={mt_fraction:.1%})"
        )

    if deg_level in ("severe", "moderate"):
        has_issues = True
        if matched_tissue and mt_fold is not None:
            flags.append(
                f"RNA degradation: {deg_level} vs {matched_tissue} baseline "
                f"(MT {mt_fold:.1f}×, RP {rp_fold:.1f}×)"
            )
        else:
            flags.append(
                f"RNA degradation: {deg_level} "
                f"(MT={mt_fraction:.1%}, RP={rp_fraction:.1%})"
            )
    elif deg_level == "mild":
        if matched_tissue and mt_fold is not None:
            flags.append(
                f"RNA quality: mildly elevated vs {matched_tissue} "
                f"(MT {mt_fold:.1f}×, RP {rp_fold:.1f}×)"
            )
        else:
            flags.append(f"RNA quality: mildly elevated (MT={mt_fraction:.1%})")

    if culture_level in ("likely_cell_line", "possible_cell_line"):
        has_issues = True
        flags.append(
            f"Cell culture: {culture_level.replace('_', ' ')} "
            f"(stress z={stress_score:.1f}, TME={tme_mean:.1f} TPM)"
        )
    elif culture_level == "tme_absent":
        flags.append(
            f"TME absent: immune/stromal signal near zero (mean={tme_mean:.1f} TPM)"
        )

    if not flags:
        flags.append("No quality concerns detected")

    return {
        "degradation": degradation,
        "culture": culture,
        "flags": flags,
        "has_issues": has_issues,
    }
