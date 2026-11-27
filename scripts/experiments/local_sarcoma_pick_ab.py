#!/usr/bin/env python3
"""Real-sample check of the 4c percentile subtype pick: on the local clinical SARCOMA samples,
what does the HK concordance ranking pick vs the percentile-cosine pick?

The subtype-pick ACCURACY table (HK 0.40 vs percentile 0.93) was measured on idealized
representative-cohort medoids. The local sarcomas are real clinical RNA-seq (alvin = sarcoma,
pfo004 = osteosarcoma). Their exact subtype isn't one of the 3 paneled pick candidates
(LMS/SYN/LPS_UNSPEC), so this is a stability/sanity check, not an accuracy score: does the
basis change the pick on real samples, and is the percentile pick sensible?

Run:  python3 scripts/local_sarcoma_pick_ab.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

import trufflepig.tumor_purity as tp
from trufflepig.tumor_purity import _mixture_cohort_lineage_summary, _build_sample_tpm_by_symbol
from trufflepig.load_expression import load_expression_data
from pirlygenes.gene_sets_cancer import housekeeping_gene_ids
from trufflepig.reference import pan_cancer_expression

D = "/Users/iskander/data"
SARCOMAS = [
    ("alvin-sarcoma", f"{D}/alvin/RNA/2025-10-31_salmon/quant.gene_tpm.csv"),
    ("pfo004-osteosarc", f"{D}/pathfinder/pfo004/analysis/gene-expression.csv"),
    ("pfo004-osteo-salmon", f"{D}/pathfinder/pfo004/analysis/transcripts_quant/quant.gene_tpm.csv"),
]


def _load(path):
    for agg in (False, True):
        try:
            return load_expression_data(path, aggregate_gene_expression=agg,
                                        save_aggregated_gene_expression=False, verbose=False, progress=False)
        except Exception:  # noqa: BLE001
            continue
    return None


def main():
    pan = pan_cancer_expression(technical_rna_normalize=True)
    id2sym = dict(zip(pan["Ensembl_Gene_ID"], pan["Symbol"]))
    hk_syms = [id2sym[g] for g in housekeeping_gene_ids() if g in id2sym]

    print(f"\n{'sample':22s} {'percentile pick (live)':28s} {'HK-ranking pick':22s}")
    print("-" * 74)
    for name, path in SARCOMAS:
        df = _load(path)
        if df is None:
            print(f"{name:22s} <could not load>")
            continue
        sym = _build_sample_tpm_by_symbol(df)

        live = _mixture_cohort_lineage_summary("SARC", sym, hk_syms)  # percentile pick (current code)
        live_pick = (live or {}).get("code")
        live_scores = {r["code"]: round(r.get("pick_score") or -1, 3)
                       for r in (live or {}).get("per_subtype", []) if r.get("pick_score") is not None}

        # force the HK ranking by disabling the percentile pick
        orig = tp._mixture_subtype_pick_scores
        tp._mixture_subtype_pick_scores = lambda *a, **k: {}
        try:
            hk = _mixture_cohort_lineage_summary("SARC", sym, hk_syms)
        finally:
            tp._mixture_subtype_pick_scores = orig
        hk_pick = (hk or {}).get("code")
        hk_conc = {r["code"]: round(r.get("concordance") or -1, 3)
                   for r in (hk or {}).get("per_subtype", []) if r.get("concordance") is not None}

        flag = "" if live_pick == hk_pick else "   <-- basis changes pick"
        print(f"{name:22s} {str(live_pick):28s} {str(hk_pick):22s}{flag}")
        print(f"  percentile pick_scores: {live_scores}")
        print(f"  HK concordances:        {hk_conc}")


if __name__ == "__main__":
    main()
