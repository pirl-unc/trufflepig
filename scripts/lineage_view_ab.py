#!/usr/bin/env python3
"""A/B for the lineage-panel migration target: does the percentile (cohort_pct) or z-score
(cohort_z) view discriminate lineage better than the HK view (the consumer the ledger flags)?

The epithelial / NE lineage gate is already a 5-view consensus (signal_views.signal_report:
hk, within_pct, log1p, cohort_pct, cohort_z — equal-weighted presence votes). The ledger's
"lineage_panels → percentile" target is therefore largely already in place; the open question
is whether HK should be DROPPED / down-weighted in favour of percentile/z. We answer it by
measuring each view's standalone discrimination (AUC of its presence-vote) for EPITHELIAL vs
non-epithelial across the representative cohort samples, and compare to the full consensus.

Run:  python3 scripts/lineage_view_ab.py
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
from trufflepig.lineage_evidence import EPITHELIAL_MARKERS
from trufflepig.lineage_marker_recall import NE_PROGRAM
from trufflepig.signal_views import signal_report, VIEW_NAMES
from trufflepig.tumor_purity import _sample_hk_median


def _auc(scores, labels):
    """AUC of scores vs binary labels (rank-based; ties handled)."""
    s = pd.Series(scores); y = np.asarray(labels, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = s.rank().to_numpy()
    return (r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _samples():
    out = []  # (lineage_group, {symbol: clean tpm})
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
            out.append((grp, clean[rep].clip(lower=0).to_dict()))
    return out


def evaluate(signal_name, genes, positive_group, samples):
    evaluated_views = ("hk", *VIEW_NAMES)
    per_view = {v: [] for v in evaluated_views}
    consensus = []
    labels = []
    for grp, sym in samples:
        hkmed = _sample_hk_median(sym)
        sig = signal_report(signal_name, genes, sym)
        hk_value = (
            float(np.median([float(sym.get(g, 0.0)) / hkmed for g in genes]))
            if hkmed > 0
            else 0.0
        )
        per_view["hk"].append(float(np.clip((hk_value - 0.10) / 1.40, 0.0, 1.0)))
        for v in VIEW_NAMES:
            per_view[v].append(sig.presence.get(v, np.nan))
        consensus.append(sig.confidence)
        labels.append(grp == positive_group)
    n_pos = sum(labels)
    print(f"\n=== {signal_name} presence: {positive_group} (n_pos={n_pos}) vs rest (n={len(labels)}) — AUC per view ===")
    for v in evaluated_views:
        sc = [x for x in per_view[v]]
        print(f"  {v:11s} AUC {_auc(sc, labels):.3f}")
    print(f"  {'PRODUCTION':11s} AUC {_auc(consensus, labels):.3f}   (log1p + cohort_pct)")
    # leave-one-view-out: does dropping HK help the consensus?
    arr = {v: np.array(per_view[v], float) for v in evaluated_views}
    for drop in evaluated_views:
        keep = [v for v in evaluated_views if v != drop]
        m = np.nanmean(np.vstack([arr[v] for v in keep]), axis=0)
        print(f"  consensus w/o {drop:5s} AUC {_auc(m, labels):.3f}  (views: {keep})")
    for keep in (
        ("within_pct", "log1p", "cohort_pct"),
        ("log1p", "cohort_pct"),
        ("within_pct", "log1p", "cohort_pct", "cohort_z"),
    ):
        stack = np.vstack([arr[v] for v in keep])
        m = np.nanmean(stack, axis=0)
        label = "+".join(keep)
        print(f"  {label:36s} AUC {_auc(m, labels):.3f}")
        if len(keep) == 3:
            median_vote = np.nanmedian(stack, axis=0)
            geomean_vote = np.prod(np.clip(stack, 0.0, 1.0), axis=0) ** (1 / 3)
            print(f"  {'median(' + label + ')':36s} AUC {_auc(median_vote, labels):.3f}")
            print(f"  {'geomean(' + label + ')':36s} AUC {_auc(geomean_vote, labels):.3f}")
    # the combined filter that landed #7: cohort_pct x within_pct
    combined = np.nan_to_num(arr["cohort_pct"]) * np.nan_to_num(arr["within_pct"])
    print(f"  {'COMBINED':11s} AUC {_auc(combined, labels):.3f}   (cohort_pct x within_pct — the #7 filter)")


def main():
    print("Per-view lineage discrimination (higher AUC = better lineage signal)", file=sys.stderr)
    samples = _samples()
    evaluate("epithelial", EPITHELIAL_MARKERS, "Epithelial", samples)
    evaluate("neuroendocrine", NE_PROGRAM, "Neuroendocrine", samples)


if __name__ == "__main__":
    main()
