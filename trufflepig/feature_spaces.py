"""Reify a sample's per-gene expression in EVERY feature space as one DataFrame.

Motivation: trufflepig's scoring scatters per-gene thresholds through control flow
(``if sample_hk < 0.1``, ``if cohort_pct > 0.9``, ``if within_pct >= ...``). Each such
test reads the same gene through a different *normalization lens*. This module collects
all the lenses into a single tidy table — one row per gene, one column per feature space,
plus boolean columns for the documented thresholds — so the thresholding becomes a
**data-flow** artifact (filter / compare on a frame) instead of buried control flow, and
so the feature spaces can be compared side-by-side for any gene or signature.

Feature spaces (all on the conformed clean-TPM sample; see CLAUDE.md + normalization_usage):
  clean_tpm                 absolute clean-TPM (depth already normalized; the raw protein proxy)
  log1p_clean_tpm           log1p of the above (variance-stabilized)
  hk_ratio                  clean_tpm / housekeeping median (legacy basis; mostly redundant — see
                            project_normalization_zscore_migration: HK worse/redundant in isolation)
  within_sample_pct         rank of the gene among ALL the sample's genes (purity-robust; survives
                            proportional dilution). Reference-FREE.
  cohort_pct_cancers        midrank of the sample value among the 33 cancer-type cohort MEDIANS
                            (the cross-cancer specificity the signature scorer uses). 33 points.
  cohort_pct_with_normals   same, but vs 33 cancer + 50 HPA normal-tissue medians (83 points) —
                            "is this gene high vs cancers AND healthy tissue". NOT used by the
                            production scorer yet; exposed here so the gap is visible/measurable.
  cohort_z_cancers          z-score of log1p(sample) vs the cancer cohort log-median distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Documented thresholds, by feature space → (column, op, value, meaning). Surfacing these as DATA
# (rather than inline constants) is the point: each becomes a boolean column in the reified frame.
# Keep in sync with the constants in plot_embedding / tumor_purity (cross-referenced in comments).
THRESHOLDS = [
    ("hk_ratio", "ge", 0.1, "detection floor (_SIGNATURE_DETECTION_FLOOR_HK): below = treated undetected"),
    ("within_sample_pct", "ge", 0.90, "within-sample top decile (dominantly expressed in this sample)"),
    ("cohort_pct_cancers", "ge", 0.90, "cross-cancer specific (top decile vs the 33 cohort medians)"),
    ("cohort_pct_with_normals", "ge", 0.90, "specific vs cancers AND normals (top decile of 83 medians)"),
    ("cohort_z_cancers", "ge", 2.0, "≥2σ above the cross-cancer mean (strong cohort outlier)"),
]


def _midrank(value: float, ref_vals: np.ndarray) -> float:
    """Midrank percentile of ``value`` among ``ref_vals`` (ties get half-credit), in [0, 1]."""
    if ref_vals is None or len(ref_vals) == 0:
        return float("nan")
    below = np.sum(ref_vals < value)
    equal = np.sum(np.isclose(ref_vals, value, atol=1e-6))
    return float((below + 0.5 * equal) / len(ref_vals))


def sample_feature_matrix(
    df_gene_expr,
    genes: list[str] | None = None,
    *,
    annotate_signatures: bool = True,
    add_threshold_flags: bool = True,
) -> pd.DataFrame:
    """Return a gene × feature-space DataFrame for one sample.

    Parameters
    ----------
    df_gene_expr : sample expression frame (ensembl id / symbol / TPM); must be clean-TPM.
    genes : restrict to these symbols (default: the union of all cancer-type signature panels —
        i.e. exactly the genes the signature scorer thresholds).
    annotate_signatures : add a ``signature_codes`` column (which cancer-type panels list the gene).
    add_threshold_flags : add one boolean column per entry in :data:`THRESHOLDS`.
    """
    from .plot_tumor_expr import _sample_expression_by_symbol
    from .plot_embedding import _get_cancer_type_signature_panels
    from .tumor_purity import _cached_reference_matrices

    sample_raw, sample_hk = _sample_expression_by_symbol(df_gene_expr)

    raw_ref = _cached_reference_matrices(normalize=None)["expr_matrix"]  # 33 cancer medians, clean-TPM
    ref_by_sym = _cached_reference_matrices(normalize="housekeeping")["ref_by_sym"]
    normal_cols = [c for c in ref_by_sym.columns if c.endswith("_nTPM")]  # 50 HPA tissues

    panels = _get_cancer_type_signature_panels()
    if genes is None:
        genes = sorted({g for gs in panels.values() for g in gs})
    gene_to_codes: dict[str, list[str]] = {}
    if annotate_signatures:
        for code, gs in panels.items():
            for g in gs:
                gene_to_codes.setdefault(g, []).append(code)

    # within-sample percentile over the WHOLE sample (reference-free), computed once
    within = pd.Series(sample_raw, dtype=float).rank(pct=True, method="average")

    rows = []
    for g in genes:
        raw = float(sample_raw.get(g, 0.0))
        cancer_vals = raw_ref.loc[g].to_numpy(float) if g in raw_ref.index else np.array([])
        if g in ref_by_sym.index and normal_cols:
            normal_vals = ref_by_sym.loc[g, normal_cols].to_numpy(float)
            all_vals = np.concatenate([cancer_vals, normal_vals]) if cancer_vals.size else normal_vals
        else:
            all_vals = cancer_vals
        if cancer_vals.size:
            logc = np.log1p(cancer_vals)
            sd = float(logc.std())
            z = float((np.log1p(raw) - logc.mean()) / sd) if sd > 0 else 0.0
        else:
            z = float("nan")
        rows.append(
            {
                "gene": g,
                "clean_tpm": raw,
                "log1p_clean_tpm": float(np.log1p(raw)),
                "hk_ratio": float(sample_hk.get(g, 0.0)),
                "within_sample_pct": float(within.get(g, np.nan)),
                "cohort_pct_cancers": _midrank(raw, cancer_vals),
                "cohort_pct_with_normals": _midrank(raw, all_vals),
                "cohort_z_cancers": z,
                "signature_codes": ",".join(gene_to_codes.get(g, [])) if annotate_signatures else "",
            }
        )
    df = pd.DataFrame(rows).set_index("gene")

    if add_threshold_flags:
        ops = {"ge": np.greater_equal, "gt": np.greater, "le": np.less_equal, "lt": np.less}
        for col, op, val, _meaning in THRESHOLDS:
            if col in df.columns:
                df[f"pass[{col}>={val}]" if op == "ge" else f"pass[{col} {op} {val}]"] = ops[op](
                    df[col].to_numpy(float), val
                )
    return df


__all__ = ["sample_feature_matrix", "THRESHOLDS", "_midrank"]
