"""Subtype-aware cancer-type signature scoring.

The broad-cohort signature in :mod:`trufflepig.plot_embedding`
(``_compute_cancer_type_signature_stats``) scores a sample against
TCGA cohort medians. That misses subtype-driven samples: basal-like
BRCA looks nothing like the luminal-dominated broad-BRCA median, and
HPV+ HNSC looks nothing like HPV-negative HNSC. The broad classifier
correctly recognizes "TNBC ≠ luminal BRCA" and routes the sample to
ESCA / LUSC / HNSC, which is the literature-documented failure mode
(see Hoadley 2014 Cell on basal BRCA clustering with squamous).

This module fixes the underlying issue rather than papering it over
with marker heuristics: each subtype in
``subtype-deconvolved-expression.csv.gz`` (BRCA Basal/LumA/LumB/Her2/
Normal; HNSC HPV+/-; LUAD by driver mutation; BEATAML by ELN risk)
gets its own signature panel derived from its tumor-only median, and
:func:`rank_cancer_type_candidates` consults the best subtype score
when ranking each cohort.

Why this is the principled fix:
- Subtype panels are *data-driven* (the top-N most-distinctive genes
  per subtype vs. the cross-cohort median), not curated marker lists
  that drift over time.
- The same scoring math (HK-normalized percentile rank in the cross-
  cohort distribution) is used for both broad and subtype panels, so
  the scores are directly comparable.
- It composes: adding a new subtype to the registry / data file
  automatically gives that subtype a signature panel and a place in
  the ranking — no engine change needed.

References:
- Hoadley et al. 2014 Cell - pan-cancer 12-type analysis
- Lehmann et al. 2011 JCI - TNBC molecular subtypes (BL1, BL2, M, MSL)
- TCGA Network 2012 Nature - BRCA PAM50 subtypes
- TCGA Network 2015 Cell  - HNSC HPV+ / HPV- divergence
"""

from __future__ import annotations

import numpy as np

# Subtype panels are deterministic given the reference data + params,
# so cache by parameter tuple to avoid repeated dataframe pivots.
_SUBTYPE_PANEL_CACHE: dict[tuple, dict[tuple[str, str], tuple[str, ...]]] = {}


def subtype_signature_panels(
    *,
    top_n: int = 20,
    min_subtype_tpm: float = 5.0,
    min_log2_within_cohort: float = 1.0,
    min_log2_vs_pan: float = 0.5,
) -> dict[tuple[str, str], tuple[str, ...]]:
    """Build per-subtype signature panels from subtype-deconvolved medians.

    Returns ``dict[(cancer_code, subtype) -> tuple[gene_symbol]]``.

    Selection rule per (cancer_code, subtype):
      1. Drop globally-excluded genes (HK / ribosomal / mt / Ig / rRNA /
         pseudogenes / etc.) using the same regex panel that the broad
         signature builder uses — otherwise universal pseudogenes (EEF1G,
         IGLC1, TOMM6, ...) drown out real markers.
      2. Gene's ``tumor_tpm_median`` in this subtype ≥ ``min_subtype_tpm``.
         Near-zero genes can't anchor a percentile score.
      3. **Within-cohort discriminator**:
         ``log2((subtype_TPM + 1) / (max_other_subtype_TPM + 1)) ≥ min_log2_within_cohort``
         — gene must be elevated in *this* subtype vs every other
         subtype of the same cancer. Without this, a gene that's
         globally high in BRCA (any subtype) would get into every BRCA
         subtype's panel and the scores would converge.
      4. **Cross-cohort discriminator**:
         ``log2((subtype_TPM + 1) / (pan_cancer_mean + 1)) ≥ min_log2_vs_pan``
         — gene also needs to be elevated vs the broad cross-cohort
         baseline so the scoring math (cross-cohort percentile rank) has
         room to discriminate.
      5. Rank by within-cohort log2, then cross-cohort log2; take top-N.

    Result is cached on the parameter tuple.
    """
    cache_key = (top_n, min_subtype_tpm, min_log2_within_cohort, min_log2_vs_pan)
    cached = _SUBTYPE_PANEL_CACHE.get(cache_key)
    if cached is not None:
        return {k: tuple(v) for k, v in cached.items()}

    from .reference import pan_cancer_expression, subtype_deconvolved_expression
    from .tumor_purity import _compile_excluded_gene_matcher

    # Subtype panel selection compares subtype TPM medians across
    # cohorts via log2 ratios. The trufflepig reference surface already
    # renormalizes subtype and pan-cancer cohorts to a common per-million
    # footing before this comparison.
    sub_df = subtype_deconvolved_expression(
        technical_rna_normalize=True,
        renormalize_to_million=True,
    )
    if sub_df is None or sub_df.empty:
        return {}

    pan = pan_cancer_expression(
        technical_rna_normalize=True,
        renormalize_to_million=True,
    )
    pan = pan.drop_duplicates(subset="Symbol").set_index("Symbol")
    cohort_cols = [c for c in pan.columns if c.endswith("_TPM")]
    if not cohort_cols:
        return {}
    cohort_means = pan[cohort_cols].astype(float).mean(axis=1)

    is_excluded = _compile_excluded_gene_matcher()

    panels: dict[tuple[str, str], tuple[str, ...]] = {}
    # Pivot subtype medians so we can index gene-by-subtype within each cohort.
    sub_pivot = (
        sub_df.pivot_table(
            index="symbol",
            columns=["cancer_code", "subtype"],
            values="tumor_tpm_median",
            aggfunc="first",
        )
    )
    for code in sub_pivot.columns.get_level_values(0).unique():
        cohort_block = sub_pivot[code]
        if cohort_block.shape[1] < 1:
            continue
        for subtype in cohort_block.columns:
            this_col = cohort_block[subtype].astype(float)
            other_cols = cohort_block.drop(columns=[subtype]) if cohort_block.shape[1] > 1 else None
            if other_cols is not None and not other_cols.empty:
                # max across all other subtypes of this cohort
                other_max = other_cols.astype(float).max(axis=1).fillna(0.0)
            else:
                other_max = None

            ranked: list[tuple[float, float, str]] = []
            for sym, val in this_col.items():
                sym_str = str(sym)
                if not sym_str or is_excluded(sym_str):
                    continue
                tpm = float(val) if val == val else 0.0  # NaN guard
                if tpm < min_subtype_tpm:
                    continue
                within = (
                    float(
                        np.log2((tpm + 1.0) / (float(other_max.get(sym, 0.0)) + 1.0))
                    )
                    if other_max is not None
                    else float(np.log2(tpm + 1.0))
                )
                if within < min_log2_within_cohort:
                    continue
                pan_base = float(cohort_means.get(sym, 0.1))
                vs_pan = float(np.log2((tpm + 1.0) / (pan_base + 1.0)))
                if vs_pan < min_log2_vs_pan:
                    continue
                ranked.append((within, vs_pan, sym_str))
            if not ranked:
                continue
            ranked.sort(key=lambda t: (-t[0], -t[1], t[2]))
            picked = [sym for _, _, sym in ranked[:top_n]]
            panels[(str(code), str(subtype))] = tuple(picked)

    _SUBTYPE_PANEL_CACHE[cache_key] = {k: tuple(v) for k, v in panels.items()}
    return panels


def compute_subtype_signature_stats(
    df_gene_expr,
    *,
    cancer_code: str | None = None,
    top_n: int = 20,
) -> dict[str, list[dict]]:
    """Score sample against each per-subtype signature panel.

    Returns ``dict[cancer_code -> list[dict]]`` where each inner dict is::

        {"subtype": ..., "score": ..., "n_genes": ..., "gene_details": [...]}

    sorted by score descending. Same scoring math as
    :func:`trufflepig.plot_embedding._compute_cancer_type_signature_stats`:
    HK-normalized percentile rank of each panel gene's expression against
    the cross-cohort distribution, averaged across genes.

    Filter to a single ``cancer_code`` if the caller knows which cohort
    to evaluate (e.g., per-candidate scoring inside the candidate loop).
    """
    from .plot_embedding import _sample_expression_by_symbol
    from .tumor_purity import _cached_reference_matrices

    panels = subtype_signature_panels(top_n=top_n)
    if not panels:
        return {}

    sample_raw_by_symbol, sample_hk_by_symbol = _sample_expression_by_symbol(
        df_gene_expr
    )
    ref_matrices = _cached_reference_matrices(normalize="housekeeping")
    ref_by_sym = ref_matrices["ref_by_sym"]
    expr_matrix = ref_matrices["expr_matrix"]

    results: dict[str, list[dict]] = {}
    for (code, subtype), panel in panels.items():
        if cancer_code is not None and code != cancer_code:
            continue
        percentiles: list[float] = []
        details: list[dict] = []
        for gene in panel:
            if gene not in ref_by_sym.index:
                continue
            sample_hk_val = float(sample_hk_by_symbol.get(gene, 0.0) or 0.0)
            sample_raw = float(sample_raw_by_symbol.get(gene, 0.0) or 0.0)
            ref_vals = expr_matrix.loc[gene].values
            n = len(ref_vals)
            if n == 0:
                continue
            below = int(np.sum(ref_vals < sample_hk_val))
            equal = int(np.sum(np.isclose(ref_vals, sample_hk_val, atol=1e-6)))
            percentile = float((below + 0.5 * equal) / n)
            percentiles.append(percentile)
            details.append(
                {
                    "gene": gene,
                    "sample_raw": sample_raw,
                    "sample_hk": sample_hk_val,
                    "percentile": percentile,
                }
            )
        if not percentiles:
            continue
        score = float(np.mean(percentiles))
        results.setdefault(code, []).append(
            {
                "subtype": subtype,
                "score": score,
                "n_genes": len(percentiles),
                "gene_details": details,
            }
        )

    for code in results:
        results[code].sort(key=lambda row: (-row["score"], row["subtype"]))
    return results


__all__ = [
    "subtype_signature_panels",
    "compute_subtype_signature_stats",
]
