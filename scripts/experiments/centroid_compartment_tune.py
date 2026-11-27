#!/usr/bin/env python3
"""Tune _COMPARTMENT_CONFIDENT_MARGIN for the z-score centroid basis.

compartment_call accepts a precomputed correlation Series (``_corr``), so we can score the
z-score basis WITHOUT wiring it into centroid_correlations yet. For every representative
cohort sample we record the compartment call + its margin under (a) the current Spearman
basis and (b) the z-score basis, then sweep the confidence margin to find the threshold
that maximises confident-coverage while keeping every confident call correct.

Run:  python3 scripts/centroid_compartment_tune.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from pirlygenes.expression.accessors import (
    available_representative_cohorts, representative_cohort_samples,
)
from pirlygenes.gene_sets_cancer import cancer_lineage_group
from oncoref.normalization import clean_tpm
from trufflepig.cancer_type_centroid import (
    centroid_correlations, zscore_centroid_correlations, compartment_call,
)
from trufflepig.expression_decomposition import _group_to_mode


def _samples():
    out = []
    for c in sorted(available_representative_cohorts()):
        grp = cancer_lineage_group(c)
        if not grp:
            continue
        d = representative_cohort_samples(c).drop_duplicates("Ensembl_Gene_ID")
        cols = [x for x in d.columns if x not in ("Ensembl_Gene_ID", "Symbol")]
        gt = pd.DataFrame({"Ensembl_Gene_ID": d["Ensembl_Gene_ID"].values, "Symbol": d["Symbol"].values})
        clean = clean_tpm(d.set_index("Ensembl_Gene_ID")[cols].astype(float), gene_table=gt.set_index(d.index))
        clean.index = d["Symbol"].values
        clean = clean[~clean.index.duplicated(keep="first")]
        for rep in clean.columns:
            sym = clean[rep].clip(lower=0).to_dict()
            out.append((c, grp, sym))
    return out


def evaluate(scorer, samples):
    """[(truth_group, pred_group, margin, correct)] for a scorer fn."""
    rows = []
    for code, grp, sym in samples:
        corr = scorer(sym)
        call = compartment_call(sym, _corr=corr)
        pred = call["compartment"]
        rows.append((grp, pred, float(call["margin"]), pred == grp))
    return rows


def sweep(rows, name):
    margins = sorted({r[2] for r in rows})
    print(f"\n=== {name}: overall argmax-compartment accuracy "
          f"{sum(r[3] for r in rows)}/{len(rows)} = {np.mean([r[3] for r in rows]):.3f} ===")
    print(f"{'thresh':>7s} | {'confident n':>11s} | {'conf correct':>12s} | {'conf acc':>8s} | {'deferred n':>10s}")
    for t in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]:
        conf = [r for r in rows if r[2] >= t]
        deferred = len(rows) - len(conf)
        acc = (sum(r[3] for r in conf) / len(conf)) if conf else float("nan")
        print(f"{t:>7.2f} | {len(conf):>11d} | {sum(r[3] for r in conf):>12d} | {acc:>8.3f} | {deferred:>10d}")
    # the wrong confident calls at a couple thresholds
    for t in (0.025, 0.05):
        bad = [(r[0], r[1], round(r[2], 3)) for r in rows if r[2] >= t and not r[3]]
        print(f"  wrong-but-confident at thresh {t}: {len(bad)} -> {bad[:8]}")


def main():
    samples = _samples()
    print(f"{len(samples)} samples, {len({s[0] for s in samples})} cohorts", file=sys.stderr)
    sweep(evaluate(centroid_correlations, samples), "Spearman-log (production)")
    sweep(evaluate(zscore_centroid_correlations, samples), "z-score (rejected for compartment)")


if __name__ == "__main__":
    main()
