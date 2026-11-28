#!/usr/bin/env python
"""Gather genes + gene-sets, every normalization form, AND every decomposition choice into ONE tidy
DataFrame — the unified feature space that does not exist in production today.

RESEARCH / EVAL ONLY — nothing here is imported by the pipeline. It answers "could we represent a
sample as a single unified feature frame?" concretely, so strategies/feature-sets can be compared on a
common substrate.

The frame is LONG/tidy so heterogeneous features coexist:
    kind          gene | geneset
    feature       gene symbol | signature code
    normalization clean_tpm | log1p | hk_ratio | within_pct | cohort_pct_cancers |
                  cohort_pct_with_normals | cohort_z | residual
    decomp_mode   none | solid | mesenchymal | heme | embryonal   (the "choice of decomposition")
    value         float

Axes:
  - GENE x {7 normalizations} x {decomp_mode=none}          (from feature_spaces.sample_feature_matrix)
  - GENE x {normalization=residual} x {4 decomposition modes}   (per-mode NNLS tumor residual)
  - GENESET x {signature score in each normalization} x {none}  (per-type signature scores)
  - GENESET x {signature score on each mode's residual}          (lineage_fit-style readout)

Run:  python scripts/unified_feature_frame.py [CANCER_TYPE]   (default COAD)
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from pirlygenes.expression.accessors import representative_cohort_samples
from trufflepig.clean_tpm import normalize_to_reference_space
from trufflepig.feature_spaces import sample_feature_matrix
from trufflepig.plot_tumor_expr import _sample_expression_by_symbol
from trufflepig.plot_embedding import (
    _compute_cancer_type_signature_stats,
    _full_cohort_signature_panels,
)
from trufflepig.expression_decomposition import _refs, _as_series, decompose_mode, signature_score

MODES = ["solid", "mesenchymal", "heme", "embryonal"]


def unified_feature_frame(sample_df, *, top_genes=None):
    """Return the tidy unified feature frame for one (clean-TPM) sample frame."""
    sample_raw, _sample_hk = _sample_expression_by_symbol(sample_df)
    rows = []

    # 1) GENE x normalizations  (decomp_mode = none)
    genes = top_genes if top_genes is not None else None
    gm = sample_feature_matrix(sample_df, genes=genes, add_threshold_flags=False)
    norm_cols = ["clean_tpm", "log1p_clean_tpm", "hk_ratio", "within_sample_pct",
                 "cohort_pct_cancers", "cohort_pct_with_normals", "cohort_z_cancers"]
    long = gm[norm_cols].reset_index().melt(id_vars="gene", var_name="normalization", value_name="value")
    long["kind"] = "gene"
    long["feature"] = long["gene"]
    long["decomp_mode"] = "none"
    rows.append(long[["kind", "feature", "normalization", "decomp_mode", "value"]])

    # 2) GENE x per-mode decomposition residual  (normalization = residual)
    templates, signatures = _refs()
    sample = _as_series(dict(sample_raw))
    panel_genes = set(gm.index)
    residual_by_mode = {}
    for m in MODES:
        try:
            _metrics, residual, _info = decompose_mode(m, sample, templates, signatures,
                                                       "percentile", None, None)
        except Exception:
            continue
        residual_by_mode[m] = residual
        sub = residual[residual.index.isin(panel_genes)] if panel_genes else residual
        rows.append(pd.DataFrame({"kind": "gene", "feature": sub.index, "normalization": "residual",
                                  "decomp_mode": m, "value": sub.to_numpy(float)}))

    # 3) GENESET (signature) score per cancer type, in each scoring normalization (decomp_mode = none)
    #    Production signature score (combined cohort_pct x within), then bare cohort_pct + within for contrast.
    sig = sorted(_compute_cancer_type_signature_stats(sample_df), key=lambda r: -r["score"])
    for r in sig:
        rows.append(pd.DataFrame([{"kind": "geneset", "feature": r["code"],
                                   "normalization": "signature_combined", "decomp_mode": "none",
                                   "value": float(r["score"])}]))

    # 4) GENESET score ON each mode's residual (the tumor-lineage readout per decomposition choice)
    panels = _full_cohort_signature_panels()
    for m, residual in residual_by_mode.items():
        for code, panel in list(panels.items()):
            if not panel:
                continue
            sc = signature_score(residual, frozenset(panel), "percentile")
            if sc is not None:
                rows.append(pd.DataFrame([{"kind": "geneset", "feature": code,
                                           "normalization": "signature_on_residual",
                                           "decomp_mode": m, "value": float(sc)}]))

    return pd.concat(rows, ignore_index=True)


def main():
    code = sys.argv[1] if len(sys.argv) > 1 else "COAD"
    raw = representative_cohort_samples(code).drop_duplicates("Ensembl_Gene_ID")
    col = [c for c in raw.columns if c not in ("Ensembl_Gene_ID", "Symbol")][0]
    samp = pd.DataFrame({"ensembl_gene_id": raw["Ensembl_Gene_ID"], "gene_symbol": raw["Symbol"],
                         "TPM": raw[col].astype(float)})
    try:
        samp = normalize_to_reference_space(samp, value_cols=["TPM"])
    except Exception:
        pass
    frame = unified_feature_frame(samp)

    print(f"\n=== unified feature frame for {code} ({col}) ===")
    print(f"shape: {frame.shape}  ({frame['kind'].eq('gene').sum()} gene rows, "
          f"{frame['kind'].eq('geneset').sum()} geneset rows)")
    print("\naxes present:")
    print("  normalizations:", sorted(frame['normalization'].unique()))
    print("  decomp_modes  :", sorted(frame['decomp_mode'].unique()))
    # demonstrate: one marker gene across every (normalization, decomp_mode) cell
    g = "CEACAM5"
    sub = frame[(frame.kind == "gene") & (frame.feature == g)]
    if not sub.empty:
        print(f"\n{g} across the unified space (pivot normalization x decomp_mode):")
        print(sub.pivot_table(index="normalization", columns="decomp_mode", values="value",
                              aggfunc="first").round(3).to_string())
    # demonstrate: top signature_on_residual per mode (which type each decomposition residual looks like)
    sr = frame[frame.normalization == "signature_on_residual"]
    if not sr.empty:
        print("\ntop type by signature_on_residual, per decomposition mode:")
        for m in MODES:
            mm = sr[sr.decomp_mode == m]
            if not mm.empty:
                best = mm.loc[mm.value.idxmax()]
                print(f"  {m:12s} -> {best.feature:14s} ({best.value:.3f})")
    out = f"/tmp/unified_feature_frame_{code}.csv"
    frame.to_csv(out, index=False)
    print(f"\nsaved tidy frame -> {out}")


if __name__ == "__main__":
    main()
