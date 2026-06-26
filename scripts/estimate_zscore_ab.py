#!/usr/bin/env python3
"""A/B: does a z-score stromal/immune contrast track tumor purity better than the current
HK-ratio enrichment (the ESTIMATE surrogate, tumor_purity._geneset_hk_ratio)?

The ESTIMATE surrogate estimates purity from how enriched the stromal/immune gene sets are
vs a reference. We synthesise a purity gradient — a tumor cohort profile diluted with a TME
background (a normal stromal/immune-rich tissue) at known fraction f, so true tumor fraction
≈ (1-f) — and ask which CONTRAST basis is more faithfully monotonic in true purity:

  HK-ratio :  sum(geneset) / sum(housekeeping)         (current)
  z-score  : (geneset_mean - cohort_mean) / cohort_sd   (per-gene z, averaged over the set)

Metric = |Spearman(contrast, true_purity)| pooled over types and dilution levels (higher is a
more faithful purity signal). We report both a GENERIC contaminant (pan-cancer mean) and a
STRUCTURED one (a real TME tissue), since z-score's robustness is contaminant-shape-dependent.

Run:  python3 scripts/estimate_zscore_ab.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from trufflepig.reference import pan_cancer_expression, estimate_signatures
from trufflepig.tumor_purity import _geneset_hk_ratio
from pirlygenes.gene_sets_cancer import housekeeping_gene_ids


TYPES = ["COAD", "BRCA", "LUAD", "PRAD", "LIHC", "STAD", "KIRC", "PAAD", "HNSC", "BLCA"]
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
TME_TISSUES = ["adipose_tissue", "smooth_muscle"]  # structured stromal contaminants


def main():
    ref = pan_cancer_expression(technical_rna_normalize=True).drop_duplicates("Symbol").set_index("Symbol")
    id_to_sym = dict(zip(
        pan_cancer_expression(technical_rna_normalize=True)["Ensembl_Gene_ID"],
        pan_cancer_expression(technical_rna_normalize=True)["Symbol"]))
    hk_syms = [id_to_sym[g] for g in housekeeping_gene_ids() if g in id_to_sym]

    est = estimate_signatures()
    stromal = est[est["Category"] == "Stromal"]["Symbol"].tolist()
    immune = est[est["Category"] == "Immune"]["Symbol"].tolist()
    tme_genes = [g for g in stromal + immune if g in ref.index]

    tcols = [f"{t}_TPM" for t in TYPES if f"{t}_TPM" in ref.columns]
    # cohort distribution of the stromal/immune signal (for the z-score basis)
    all_tcols = [c for c in ref.columns if c.endswith("_TPM")]

    def geneset_mean(genes, col_vals):
        v = [col_vals.get(g, 0.0) for g in genes if g in ref.index]
        return float(np.mean(v)) if v else 0.0

    # per-gene cohort mean/sd over log1p reference (z-score reference for the TME genes)
    logref = np.log1p(ref[all_tcols].astype(float))
    mu = logref.mean(axis=1)
    sd = logref.std(axis=1).replace(0, np.nan)

    def zscore_contrast(col_vals):
        zs = []
        for g in tme_genes:
            if g in mu.index and np.isfinite(sd.get(g, np.nan)):
                zs.append((np.log1p(col_vals.get(g, 0.0)) - mu[g]) / sd[g])
        return float(np.mean(zs)) if zs else 0.0

    def hk_contrast(col_vals):
        # stromal+immune enrichment over HK (higher => more TME => lower purity)
        return _geneset_hk_ratio(tme_genes, hk_syms, col_vals)

    def sp(a, b):
        ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
        return float(np.corrcoef(ra, rb)[0, 1])

    def run(contaminant_name, bg_vals):
        # pooled across types, using HK ENRICHMENT (the production method: ratio ÷ the type's own
        # f=0 reference) so type baseline is removed for BOTH methods — the fair comparison.
        rows = []   # (true_purity, hk_enrichment, z_contrast)
        per_type_hk, per_type_z = [], []
        for t in tcols:
            tumor = ref[t].astype(float)
            hk0 = None
            tp_t, hk_t, z_t = [], [], []
            for f in FRACS:
                mix = (tumor * (1 - f) + pd.Series(bg_vals).reindex(tumor.index).fillna(0.0) * f)
                mix = mix / mix.sum() * 1e6
                cv = mix.to_dict()
                raw_hk = hk_contrast(cv)
                if hk0 is None:
                    hk0 = raw_hk or 1.0
                enr = raw_hk / hk0 if hk0 else 0.0
                rows.append((1 - f, enr, zscore_contrast(cv)))
                tp_t.append(1 - f); hk_t.append(enr); z_t.append(zscore_contrast(cv))
            per_type_hk.append(abs(sp(hk_t, tp_t)))
            per_type_z.append(abs(sp(z_t, tp_t)))
        arr = np.array(rows)
        tp, hk, z = arr[:, 0], arr[:, 1], arr[:, 2]
        print(f"  [{contaminant_name}]")
        print(f"      pooled |Spearman(contrast, purity)|:   HK-enrichment {abs(sp(hk, tp)):.3f}   "
              f"z-score {abs(sp(z, tp)):.3f}   (n={len(rows)})")
        print(f"      mean per-type |Spearman|:              HK-enrichment {np.mean(per_type_hk):.3f}   "
              f"z-score {np.mean(per_type_z):.3f}")

    print(f"types={len(tcols)} fracs={FRACS} tme_genes={len(tme_genes)}", file=sys.stderr)
    print("\n=== ESTIMATE contrast vs true purity (higher |corr| = better purity signal) ===")
    # generic contaminant = pan-cancer mean
    run("generic (pan-cancer mean)", ref[all_tcols].astype(float).mean(axis=1).to_dict())
    for tis in TME_TISSUES:
        col = f"{tis}_nTPM"
        if col in ref.columns:
            run(f"structured ({tis})", ref[col].astype(float).to_dict())


if __name__ == "__main__":
    main()
