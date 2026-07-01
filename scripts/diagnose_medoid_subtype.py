#!/usr/bin/env python
"""Per-medoid top call + lineage. For each medoid type prints: truth_lineage, top_call, call_lineage,
ok, and the top-3 (code, support_score) — a quick read on what the ranker calls each cohort's medoid.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from pirlygenes.expression.accessors import available_representative_cohorts, representative_cohort_samples
from pirlygenes.gene_sets_cancer import cancer_lineage_group
from oncoref.normalization import clean_tpm
from trufflepig.tumor_purity import rank_cancer_type_candidates
from trufflepig.expression_decomposition import _group_to_mode


def medoid_df(t):
    d = representative_cohort_samples(t).drop_duplicates("Ensembl_Gene_ID")
    cols = [c for c in d.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    gt = pd.DataFrame({"Ensembl_Gene_ID": d["Ensembl_Gene_ID"].values, "Symbol": d["Symbol"].values})
    cl = clean_tpm(d.set_index("Ensembl_Gene_ID")[cols].astype(float), gene_table=gt.set_index(d.index))
    return pd.DataFrame({"ensembl_gene_id": d["Ensembl_Gene_ID"].values,
                         "gene_symbol": d["Symbol"].values, "TPM": cl.mean(axis=1).values})


def lin(code):
    g = cancer_lineage_group(code) if code else None
    return _group_to_mode(g) if g else None


def main():
    for t in sorted(available_representative_cohorts()):
        try:
            r = rank_cancer_type_candidates(medoid_df(t), top_k=3)
            top = r[0]["code"] if r else None
            top3 = "; ".join(f"{x['code']}={x['support_score']:.3f}" for x in r[:3])
        except Exception as e:  # noqa: BLE001
            top, top3 = "ERR:" + str(e)[:40], ""
        tl, cl = lin(t), lin(top)
        ok = tl is not None and tl == cl
        print(f"{t}\t{tl}\t{top}\t{cl}\t{'OK' if ok else 'WRONG'}\t{top3}")


if __name__ == "__main__":
    main()
