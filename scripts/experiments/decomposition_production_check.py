#!/usr/bin/env python3
"""Faithful 4c check: does the PRODUCTION mixture concordance (_mixture_cohort_lineage_summary —
TME-excess-weighted cosine, not the simplified plain cosine of decomposition_concordance_ab.py)
actually mispick SARC subtypes on HK? Answers "is there a real production gap to close?"

For each SARC subtype that is itself a representative cohort (LMS / SYN / LPS_UNSPEC), run the
production mixture summary on each of its representative samples and check whether the picked
subtype == the true subtype.

Run:  python3 scripts/decomposition_production_check.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from pirlygenes.expression.accessors import representative_cohort_samples
from oncoref.normalization import clean_tpm
from pirlygenes.gene_sets_cancer import housekeeping_gene_ids
from trufflepig.reference import pan_cancer_expression
from trufflepig.tumor_purity import (
    _mixture_cohort_lineage_summary, _lineage_genes_map, _subtype_tumor_tpm_lookup,
)
from trufflepig.cancer_ontology import cancer_type_subtypes_of

PARENT = "SARC"


def _sym_clean(code):
    d = representative_cohort_samples(code).drop_duplicates("Ensembl_Gene_ID")
    cols = [x for x in d.columns if x not in ("Ensembl_Gene_ID", "Symbol")]
    gt = pd.DataFrame({"Ensembl_Gene_ID": d["Ensembl_Gene_ID"].values, "Symbol": d["Symbol"].values})
    clean = clean_tpm(d.set_index("Ensembl_Gene_ID")[cols].astype(float), gene_table=gt.set_index(d.index))
    clean.index = d["Symbol"].values
    return clean[~clean.index.duplicated(keep="first")]


def main():
    lg = _lineage_genes_map()
    subs = [s for s in (cancer_type_subtypes_of(PARENT) or [])
            if lg.get(s) and _subtype_tumor_tpm_lookup(s)]
    pc = pan_cancer_expression(technical_rna_normalize=True)
    id2sym = dict(zip(pc["Ensembl_Gene_ID"], pc["Symbol"]))
    hk_syms = [id2sym[g] for g in housekeeping_gene_ids() if g in id2sym]

    print(f"candidate subtypes: {subs}", file=sys.stderr)
    correct = n = 0
    confusion = {}
    for true_s in subs:
        clean = _sym_clean(true_s)
        for rep in clean.columns:
            sym = clean[rep].clip(lower=0).to_dict()
            summary = _mixture_cohort_lineage_summary(PARENT, sym, hk_syms)
            picked = (summary or {}).get("code")
            n += 1
            correct += (picked == true_s)
            confusion[(true_s, picked)] = confusion.get((true_s, picked), 0) + 1
    print(f"\n=== PRODUCTION mixture-summary subtype-pick accuracy (TME-excess-weighted, HK basis) ===")
    print(f"  {correct}/{n} = {correct/n:.3f}   (n={n} SARC samples, {len(subs)} subtypes)")
    print("  confusion (true -> picked: count):")
    for (t, p), c in sorted(confusion.items()):
        flag = "" if t == p else "  <-- MISPICK"
        print(f"    {t:18s} -> {str(p):18s}: {c}{flag}")


if __name__ == "__main__":
    main()
