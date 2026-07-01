#!/usr/bin/env python3
"""A/B for the decomposition-mixture target: does a z-score (or percentile) marker-pattern
concordance pick the correct mixture subtype better than the current HK-ratio concordance?

The mixture summary (tumor_purity._mixture_cohort_lineage_summary, #171) picks the subtype whose
curated lineage panel best PATTERN-matches the sample (a weighted cosine of HK-ratio excess vs the
subtype's tumor-only profile). The ledger flags this HK-ratio basis for a z-score move. We test the
essence — subtype-pick accuracy — directly: for the SARC subtypes that have BOTH a curated panel and
a tumor-only profile (LMS / SYN / LPS_UNSPEC, the real #171 scope), classify each subtype's
representative samples by cosine concordance to each candidate subtype, under four bases.

Run:  python3 scripts/decomposition_concordance_ab.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from pirlygenes.expression.accessors import representative_cohort_samples
from oncoref.normalization import clean_tpm
from trufflepig.tumor_purity import _lineage_genes_map, _subtype_tumor_tpm_lookup
from trufflepig.signal_views import _cohort_reference
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
    print(f"candidate subtypes: {subs}", file=sys.stderr)
    panel = sorted({g for s in subs for g in lg[s]})            # union panel
    ref = _cohort_reference()                                   # per-gene cohort distribution
    tumor = {s: _subtype_tumor_tpm_lookup(s) for s in subs}     # subtype tumor-only profiles

    def transform(vec_by_sym, basis, hk_med=None):
        """{gene: scalar} over the union panel under a basis."""
        out = {}
        for g in panel:
            v = float(vec_by_sym.get(g, 0.0) or 0.0)
            if basis == "hk":
                out[g] = v / hk_med if hk_med else 0.0
            elif basis == "log1p":
                out[g] = np.log1p(v)
            elif basis in ("percentile", "percentile_sq", "percentile_p15", "percentile_cube"):
                d = ref.get(g)
                p = float((d < v).mean()) if d is not None and len(d) else 0.5
                out[g] = {"percentile": p, "percentile_sq": p * p,
                          "percentile_p15": p ** 1.5, "percentile_cube": p ** 3}[basis]
            elif basis == "zscore":
                d = ref.get(g)
                if d is None or len(d) == 0 or d.std() == 0:
                    out[g] = 0.0
                else:
                    out[g] = float((np.log1p(v) - np.log1p(d).mean()) / np.log1p(d).std())
        return np.array([out[g] for g in panel], float)

    def cosine(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

    # precompute subtype reference vectors per basis
    bases = ["hk", "log1p", "percentile", "zscore",
             "percentile_p15", "percentile_sq", "percentile_cube"]
    ref_vecs = {b: {s: transform(tumor[s], b, hk_med=1.0) for s in subs} for b in bases}

    hk_syms = None
    from pirlygenes.gene_sets_cancer import housekeeping_gene_ids
    from trufflepig.reference import pan_cancer_expression
    pc = pan_cancer_expression(technical_rna_normalize=True)
    id2sym = dict(zip(pc["Ensembl_Gene_ID"], pc["Symbol"]))
    hk_syms = [id2sym[g] for g in housekeeping_gene_ids() if g in id2sym]

    correct = {b: 0 for b in bases}
    n = 0
    for true_s in subs:
        clean = _sym_clean(true_s)
        for rep in clean.columns:
            sym = clean[rep].clip(lower=0).to_dict()
            hkvals = [sym[g] for g in hk_syms if sym.get(g, 0) > 0]
            hk_med = float(np.median(hkvals)) if hkvals else 1.0
            n += 1
            for b in bases:
                svec = transform(sym, b, hk_med=hk_med)
                picked = max(subs, key=lambda s: cosine(svec, ref_vecs[b][s]))
                correct[b] += (picked == true_s)
    print(f"\n=== SARC subtype-pick accuracy via marker concordance (n={n} samples, {len(subs)} subtypes, "
          f"panel={len(panel)} genes) ===")
    for b in bases:
        print(f"  {b:11s} {correct[b]}/{n} = {correct[b]/n:.3f}")


if __name__ == "__main__":
    main()
