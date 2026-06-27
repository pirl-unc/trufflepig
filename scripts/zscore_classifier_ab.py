#!/usr/bin/env python3
"""Ask 3: is the per-gene z-score a good FEATURE SPACE for a simple learned classifier?

Builds features for the 266 representative-cohort samples (54 primary types) and compares, under
leakage-free 5-fold stratified CV (the scaler/centroid is fit on the TRAIN fold only):
  - LogisticRegression on raw log1p(clean-TPM)
  - LogisticRegression on per-gene Z-SCORE space (StandardScaler -> LR) — the question
  - Nearest-centroid (correlation) on z-score space — the current research-harness method (~0.838)

"What we z-score against": per-gene mean/std of the reference. In the deployable pipeline that's a
FIXED reference (cohort medoids); here CV fits mean/std on the train fold so the number is honest.

Run:  python3 scripts/zscore_classifier_ab.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import pirlygenes as pg
from oncoref.normalization import clean_tpm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestCentroid


def assemble():
    coh = sorted(c for c in pg.available_representative_cohorts() if "_" not in c)
    frames, labels, base = [], [], None
    for c in coh:
        d = pg.representative_cohort_samples(c).drop_duplicates("Ensembl_Gene_ID").set_index("Ensembl_Gene_ID")
        sc = [x for x in d.columns if x != "Symbol"]
        if base is None:
            base = d[["Symbol"]].copy()
        frames.append(d[sc]); labels += [c] * len(sc)
    expr = pd.concat(frames, axis=1).dropna(how="any")
    gt = pd.DataFrame({"Ensembl_Gene_ID": expr.index, "Symbol": base["Symbol"].reindex(expr.index).values})
    clean = clean_tpm(expr.astype(float), gene_table=gt)
    L = np.log1p(clean.clip(lower=0))
    X = np.nan_to_num(L.T.values, nan=0.0, posinf=0.0, neginf=0.0)   # samples × genes
    y = np.array(labels)
    # top high-variance genes (the discriminative subset)
    var = np.nan_to_num(L.var(axis=1).values)
    top = np.argsort(var)[::-1][:2000]
    return X[:, top], y


def main():
    X, y = assemble()
    print(f"{X.shape[0]} samples / {len(set(y))} classes / {X.shape[1]} genes  "
          f"chance={1/len(set(y)):.3f}", file=sys.stderr)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    def score(model):
        accs = []
        for tr, te in cv.split(X, y):
            model.fit(X[tr], y[tr])
            accs.append((model.predict(X[te]) == y[te]).mean())
        return np.mean(accs), np.std(accs)

    configs = {
        "LR raw-log1p":        make_pipeline(LogisticRegression(max_iter=2000, C=1.0)),
        "LR z-score (C=1)":    make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "LR z-score (C=0.1)":  make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)),
        "NearestCentroid z":   make_pipeline(StandardScaler(), NearestCentroid(metric="euclidean")),
    }
    print(f"\n=== 5-fold stratified CV accuracy (266 samples, 54 types, chance {1/len(set(y)):.3f}) ===")
    print(f"{'method':22s} | {'acc':>6s} | {'±sd':>5s}")
    print("-" * 40)
    for name, model in configs.items():
        m, s = score(model)
        print(f"{name:22s} | {m:6.3f} | {s:5.3f}")

    # ---- purity-dilution stress: train on CLEAN, test on samples diluted with a generic
    # (pan-cancer-mean) background at fraction p. The decisive question for real (impure) samples:
    # does a learned LR keep its clean-data win, or collapse like raw clean-TPM while only the
    # z-score basis holds? (mirrors /tmp/hk_harness2.py's stress test, but for a learned model)
    bg = X.mean(axis=0)
    print(f"\n=== purity stress: train on clean TRAIN fold, test on HELD-OUT diluted samples (5-fold) ===")
    ps = (0.0, 0.3, 0.5, 0.7)
    print(f"{'method':22s} | " + " | ".join(f"p={p:.1f}" for p in ps))
    for name, model in configs.items():
        rows = {p: [] for p in ps}
        for tr, te in cv.split(X, y):
            model.fit(X[tr], y[tr])
            for p in ps:
                Xd = X[te] * (1 - p) + bg * p          # dilute only the held-out test samples
                rows[p].append((model.predict(Xd) == y[te]).mean())
        print(f"{name:22s} | " + " | ".join(f"{np.mean(rows[p]):5.3f}" for p in ps))


if __name__ == "__main__":
    main()
