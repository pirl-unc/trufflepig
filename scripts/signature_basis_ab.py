#!/usr/bin/env python3
"""HK-migration #7: the PRIMARY cancer-type signal (signature_score, plot_embedding.
_compute_cancer_type_signature_stats) scores each type's curated panel by the cross-cohort
PERCENTILE of the sample's HK-normalized value. A/B the sample basis that feeds that percentile,
on representative-sample type + lineage accuracy:

  cohort_pct(HK)      — current: percentile of sample/HK-median among cohort HK values
  cohort_pct(cleanTPM)— drop HK: percentile of the raw clean-TPM value among cohort clean-TPM
  within_sample_pct   — rank of the gene WITHIN the sample (reference-free)
  combined            — cohort_pct(HK) * within_sample_pct  (two weak filters, per the user's idea)
  combined_raw        — cohort_pct(cleanTPM) * within_sample_pct

Score per type = mean over its panel; argmax type. Reports EXACT-type and LINEAGE accuracy (lineage
is where cohort-percentile is expected to shine for coarse calls).

Run:  python3 scripts/signature_basis_ab.py
"""
import sys, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from pirlygenes.expression.accessors import available_representative_cohorts, representative_cohort_samples
from pirlygenes.gene_sets_cancer import cancer_lineage_group, housekeeping_gene_ids
from oncoref.normalization import clean_tpm
from trufflepig.reference import pan_cancer_expression
from trufflepig.plot_embedding import _get_cancer_type_signature_panels
from trufflepig.expression_decomposition import _group_to_mode


def _primary(code):
    return str(code).split("_")[0]


def main():
    panels = _get_cancer_type_signature_panels(n_signature_genes=20)
    types = sorted(panels)                                  # the 33 panel types
    pan = pan_cancer_expression(technical_rna_normalize=True).drop_duplicates("Symbol").set_index("Symbol")
    tcols = [c for c in pan.columns if c.endswith("_TPM")]
    raw_ref = pan[tcols].apply(pd.to_numeric, errors="coerce").fillna(0.0)   # gene x cohort, clean-TPM
    id2sym = dict(zip(pan_cancer_expression(technical_rna_normalize=True)["Ensembl_Gene_ID"],
                      pan_cancer_expression(technical_rna_normalize=True)["Symbol"]))
    hk_syms = [id2sym[g] for g in housekeeping_gene_ids() if g in id2sym]
    # reference HK-median per cohort → HK-normalized reference
    hk_in_ref = [s for s in hk_syms if s in raw_ref.index]
    cohort_hk = raw_ref.loc[hk_in_ref].median(axis=0).replace(0, np.nan)
    hk_ref = raw_ref.div(cohort_hk, axis=1)

    # rep samples (primary types present in the panels)
    samples = []
    for c in sorted(available_representative_cohorts()):
        if _primary(c) not in types:
            continue
        d = representative_cohort_samples(c).drop_duplicates("Ensembl_Gene_ID")
        cols = [x for x in d.columns if x not in ("Ensembl_Gene_ID", "Symbol")]
        gt = pd.DataFrame({"Ensembl_Gene_ID": d["Ensembl_Gene_ID"].values, "Symbol": d["Symbol"].values})
        cl = clean_tpm(d.set_index("Ensembl_Gene_ID")[cols].astype(float), gene_table=gt.set_index(d.index))
        cl.index = d["Symbol"].values
        cl = cl[~cl.index.duplicated(keep="first")]
        for rep in cl.columns:
            samples.append((_primary(c), cl[rep].clip(lower=0)))

    def pct_among(ref_row_vals, x):
        v = ref_row_vals[np.isfinite(ref_row_vals)]
        return float((v < x).mean()) if v.size else 0.5

    def score_all(sample, basis):
        s = sample
        hk_med = float(np.median([s[g] for g in hk_syms if g in s.index and s[g] > 0]) or 1.0)
        wp = s.rank(pct=True)                                  # within-sample percentile
        out = {}
        for t in types:
            ps = []
            for g in panels[t]:
                if g not in raw_ref.index or g not in s.index:
                    continue
                sv = float(s[g])
                if basis == "cohort_hk":
                    ps.append(pct_among(hk_ref.loc[g].to_numpy(), sv / hk_med))
                elif basis == "cohort_raw":
                    ps.append(pct_among(raw_ref.loc[g].to_numpy(), sv))
                elif basis == "within_pct":
                    ps.append(float(wp.get(g, 0.5)))
                elif basis == "combined":
                    ps.append(pct_among(hk_ref.loc[g].to_numpy(), sv / hk_med) * float(wp.get(g, 0.5)))
                elif basis == "combined_raw":
                    ps.append(pct_among(raw_ref.loc[g].to_numpy(), sv) * float(wp.get(g, 0.5)))
            out[t] = float(np.mean(ps)) if ps else 0.0
        return max(out, key=out.get)

    bases = ["cohort_hk", "cohort_raw", "within_pct", "combined", "combined_raw"]
    print(f"{len(samples)} samples / {len(types)} panel types", file=sys.stderr)
    # pan-cancer-mean background for the dilution stress
    bg = raw_ref.mean(axis=1)

    def acc(b, dilute_p=0.0):
        ex = lin = 0
        for truth, s in samples:
            if dilute_p > 0:
                mix = s * (1 - dilute_p) + bg.reindex(s.index).fillna(0.0) * dilute_p
                s_use = (mix / mix.sum() * 1e6) if mix.sum() > 0 else s
            else:
                s_use = s
            pred = score_all(s_use, b)
            ex += (pred == truth)
            lin += (_group_to_mode(cancer_lineage_group(pred) or "") ==
                    _group_to_mode(cancer_lineage_group(truth) or ""))
        return ex / len(samples), lin / len(samples)

    print(f"\n=== signature_score sample-basis A/B (n={len(samples)}) — exact-type acc ===")
    print(f"{'basis':14s} | {'clean':>6s} | {'p=0.3':>6s} | {'p=0.5':>6s} | {'p=0.7':>6s} | {'lineage(clean)':>14s}")
    for b in bases:
        ex0, lin0 = acc(b, 0.0)
        ex3, _ = acc(b, 0.3)
        ex5, _ = acc(b, 0.5)
        ex7, _ = acc(b, 0.7)
        print(f"{b:14s} | {ex0:>6.3f} | {ex3:>6.3f} | {ex5:>6.3f} | {ex7:>6.3f} | {lin0:>14.3f}")


if __name__ == "__main__":
    main()
