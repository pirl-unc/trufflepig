#!/usr/bin/env python3
"""Real-sample check of the 4c subtype pick across ALL local clinical samples, three bases.

For every local sample (eval_nohint_validation.REPORTS), force the SARC mixture subtype pick to
use HK / percentile / z-score and report what each picks. The subtype-pick ACCURACY table
(HK 0.40 vs percentile 0.93) was on idealized cohort medoids; this is a real-sample stability /
sanity check. Only the sarcomas (alvin, pfo004) are genuine SARC subtypes — for non-sarcoma
samples the SARC pick is off-label and just probes basis stability.

Run:  python3 scripts/local_pick_ab.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np

import trufflepig.tumor_purity as tp
from trufflepig.tumor_purity import (
    _mixture_cohort_lineage_summary, _build_sample_tpm_by_symbol,
    _lineage_genes_map, _subtype_tumor_tpm_lookup,
)
from trufflepig.load_expression import load_expression_data
from pirlygenes.gene_sets_cancer import housekeeping_gene_ids
from trufflepig.reference import pan_cancer_expression
from trufflepig.signal_views import _cohort_reference
from eval_nohint_validation import REPORTS


def _load(path):
    for agg in (False, True):
        try:
            return load_expression_data(path, aggregate_gene_expression=agg,
                                        save_aggregated_gene_expression=False, verbose=False, progress=False)
        except Exception:  # noqa: BLE001
            continue
    return None


def _make_pick_fn(basis):
    """Return a _mixture_subtype_pick_scores-shaped fn computing a union-panel cosine under `basis`."""
    ref = _cohort_reference()
    lg = _lineage_genes_map()

    def fn(paneled_subtypes, sample_tpm):
        profiles = {s: _subtype_tumor_tpm_lookup(s) for s in paneled_subtypes}
        profiles = {s: p for s, p in profiles.items() if p}
        if len(profiles) < 2 or not ref:
            return {}
        panel = sorted({g for s in profiles for g in lg.get(s, [])})

        def vec(vbs):
            out = []
            for g in panel:
                d = ref.get(g)
                v = float(vbs.get(g, 0.0) or 0.0)
                if basis == "percentile":
                    out.append(float((d < v).mean()) if d is not None and len(d) else 0.5)
                elif basis == "zscore":
                    if d is None or len(d) == 0 or np.log1p(d).std() == 0:
                        out.append(0.0)
                    else:
                        out.append(float((np.log1p(v) - np.log1p(d).mean()) / np.log1p(d).std()))
            return np.asarray(out, float)

        s = vec(sample_tpm); sn = float(np.linalg.norm(s))
        if sn <= 0:
            return {}
        scores = {}
        for sub, p in profiles.items():
            t = vec(p); tn = float(np.linalg.norm(t))
            scores[sub] = float(s @ t / (sn * tn)) if tn > 0 else 0.0
        return scores
    return fn


def _pick(sym, hk_syms, basis):
    orig = tp._mixture_subtype_pick_scores
    tp._mixture_subtype_pick_scores = (lambda *a, **k: {}) if basis == "hk" else _make_pick_fn(basis)
    try:
        out = _mixture_cohort_lineage_summary("SARC", sym, hk_syms)
    finally:
        tp._mixture_subtype_pick_scores = orig
    return (out or {}).get("code")


def main():
    pan = pan_cancer_expression(technical_rna_normalize=True)
    id2sym = dict(zip(pan["Ensembl_Gene_ID"], pan["Symbol"]))
    hk_syms = [id2sym[g] for g in housekeeping_gene_ids() if g in id2sym]

    print(f"\n{'sample':22s} {'truth':12s} {'HK':18s} {'percentile':18s} {'zscore':18s}")
    print("-" * 92)
    for name, path, truth in REPORTS:
        df = _load(path)
        if df is None:
            print(f"{name:22s} {truth:12s} <could not load>")
            continue
        sym = _build_sample_tpm_by_symbol(df)
        picks = {b: _pick(sym, hk_syms, b) for b in ("hk", "percentile", "zscore")}
        agree = "" if len(set(picks.values())) == 1 else "   <-- bases differ"
        print(f"{name:22s} {truth:12s} {str(picks['hk']):18s} {str(picks['percentile']):18s} "
              f"{str(picks['zscore']):18s}{agree}")


if __name__ == "__main__":
    main()
