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


def evaluate(signal_name, genes, positive_group):
    samples = _samples()
    per_view = {v: [] for v in VIEW_NAMES}
    consensus = []
    labels = []
    for grp, sym in samples:
        hkmed = _sample_hk_median(sym)
        sig = signal_report(signal_name, genes, sym, sample_hk_median=hkmed)
        for v in VIEW_NAMES:
            per_view[v].append(sig.presence.get(v, np.nan))
        consensus.append(sig.confidence)
        labels.append(grp == positive_group)
    n_pos = sum(labels)
    print(f"\n=== {signal_name} presence: {positive_group} (n_pos={n_pos}) vs rest (n={len(labels)}) — AUC per view ===")
    for v in VIEW_NAMES:
        sc = [x for x in per_view[v]]
        print(f"  {v:11s} AUC {_auc(sc, labels):.3f}")
    print(f"  {'CONSENSUS':11s} AUC {_auc(consensus, labels):.3f}   (equal-weight 5-view confidence)")
    # leave-one-view-out: does dropping HK help the consensus?
    arr = {v: np.array(per_view[v], float) for v in VIEW_NAMES}
    for drop in ("hk", "none"):
        keep = [v for v in VIEW_NAMES if v != drop]
        m = np.nanmean(np.vstack([arr[v] for v in keep]), axis=0)
        print(f"  consensus w/o {drop:5s} AUC {_auc(m, labels):.3f}  (views: {keep})")
    # the combined filter that landed #7: cohort_pct x within_pct
    combined = np.nan_to_num(arr["cohort_pct"]) * np.nan_to_num(arr["within_pct"])
    print(f"  {'COMBINED':11s} AUC {_auc(combined, labels):.3f}   (cohort_pct x within_pct — the #7 filter)")


def main():
    print("Per-view lineage discrimination (higher AUC = better lineage signal)", file=sys.stderr)
    evaluate("epithelial", EPITHELIAL_MARKERS, "Epithelial")
    evaluate("neuroendocrine", NE_PROGRAM, "Neuroendocrine")


if __name__ == "__main__":
    main()
