#!/usr/bin/env python3
"""A/B: production whole-profile centroid scorer (Spearman on log1p clean-TPM) vs a
per-gene z-score nearest-centroid, on the SAME production 118-medoid reference
(`cancer_type_centroid._bulk_centroids`).

Mirrors the standalone research harness (/tmp/hk_harness2.py: z-score 0.838 vs HK 0.782,
purity-robust) but inside trufflepig's reference + symbol space, so the win can be wired
into `centroid_correlations` with confidence. Reports clean accuracy AND purity-diluted
accuracy (each test sample mixed with a pan-cancer-mean background at fraction p).

Run:  python3 scripts/centroid_zscore_ab.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from pirlygenes.expression.accessors import (
    available_representative_cohorts,
    representative_cohort_samples,
)
from oncoref.normalization import clean_tpm
from trufflepig.cancer_type_centroid import _bulk_centroids, _rankdata


def _primary(code):
    """Map a cohort code to its primary type (subtype -> parent): BRCA_Basal -> BRCA."""
    return str(code).split("_")[0]


def _load_samples(primary_only=True):
    """{cohort: log1p-clean-TPM DataFrame [Symbol x rep]} for every representative cohort."""
    out = {}
    for c in sorted(available_representative_cohorts()):
        if primary_only and "_" in c:
            continue
        d = representative_cohort_samples(c).drop_duplicates("Ensembl_Gene_ID")
        cols = [x for x in d.columns if x not in ("Ensembl_Gene_ID", "Symbol")]
        gt = pd.DataFrame({"Ensembl_Gene_ID": d["Ensembl_Gene_ID"].values, "Symbol": d["Symbol"].values})
        clean = clean_tpm(d.set_index("Ensembl_Gene_ID")[cols].astype(float),
                          gene_table=gt.set_index(d.index))
        clean.index = d["Symbol"].values
        clean = clean[~clean.index.duplicated(keep="first")]
        out[c] = clean
    return out


def main():
    bulk, informative = _bulk_centroids()          # log1p reference [Symbol x COHORT]
    ref_codes = list(bulk.columns)
    # z-reference: per-gene standardize across cohorts
    mu = bulk.mean(axis=1)
    sd = bulk.std(axis=1).replace(0, np.nan)
    zref = bulk.sub(mu, axis=0).div(sd, axis=0)
    # high-variance discriminative gene subset (the z-score recipe), among informative genes
    var = bulk.loc[informative].var(axis=1).sort_values(ascending=False)
    HIVAR = list(var.index[:2000])
    HIVAR5K = list(var.index[:5000])
    ALLINF = list(informative)

    samples = _load_samples(primary_only=True)
    # pan-cancer mean background (TPM space, then log) for dilution stress
    bg_log = bulk.mean(axis=1)

    # Build per-sample test vectors (log1p clean-TPM, Symbol-keyed) with truth labels
    tests = []   # (truth_primary, pd.Series log1p over Symbol)
    for c, clean in samples.items():
        for rep in clean.columns:
            tests.append((_primary(c), np.log1p(clean[rep].clip(lower=0))))

    def spearman_argmax(slog, codes):
        shared = slog.index.intersection(informative)
        if len(shared) < 200:
            return None
        sr = _rankdata(slog.loc[shared].to_numpy())
        R = bulk.loc[shared]
        best, bestc = -2, None
        for code in codes:
            rr = _rankdata(R[code].to_numpy())
            v = float(np.corrcoef(sr, rr)[0, 1])
            if v > best:
                best, bestc = v, code
        return bestc

    def zscore_scores(slog, codes, genes):
        shared = [g for g in genes if g in slog.index]
        if len(shared) < 100:
            return None
        zs = (slog.reindex(shared) - mu.reindex(shared)) / sd.reindex(shared)
        zs = zs.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        Z = zref.reindex(shared)
        a = zs - zs.mean()
        na = np.linalg.norm(a) or 1.0
        out = {}
        for code in codes:
            b = Z[code].fillna(0.0).to_numpy()
            b = b - b.mean()
            out[code] = float(a @ b / (na * (np.linalg.norm(b) or 1.0)))
        return out

    def spearman_scores(slog, codes):
        shared = slog.index.intersection(informative)
        if len(shared) < 200:
            return None
        sr = _rankdata(slog.loc[shared].to_numpy())
        R = bulk.loc[shared]
        return {code: float(np.corrcoef(sr, _rankdata(R[code].to_numpy()))[0, 1]) for code in codes}

    def zscore_argmax(slog, codes, genes):
        sc = zscore_scores(slog, codes, genes)
        return max(sc, key=sc.get) if sc else None

    METHODS = ["spearman", "z2k", "z5k", "zall", "ens2k", "ensall"]

    def predict(slog):
        sp = spearman_scores(slog, ref_codes)
        z2 = zscore_scores(slog, ref_codes, HIVAR)
        z5 = zscore_scores(slog, ref_codes, HIVAR5K)
        za = zscore_scores(slog, ref_codes, ALLINF)
        if sp is None or z2 is None or za is None:
            return None
        def amax(d): return max(d, key=d.get)
        # ensemble: mean of the two correlation scores per code
        ens2 = {c: 0.5 * sp[c] + 0.5 * z2[c] for c in ref_codes}
        ensa = {c: 0.5 * sp[c] + 0.5 * za[c] for c in ref_codes}
        return {"spearman": amax(sp), "z2k": amax(z2), "z5k": amax(z5),
                "zall": amax(za), "ens2k": amax(ens2), "ensall": amax(ensa)}

    def accuracy(dilute_p=0.0):
        ok = {m: 0 for m in METHODS}; n = 0
        for truth, slog in tests:
            if dilute_p > 0:
                tpm = np.expm1(slog)
                bg = np.expm1(bg_log.reindex(slog.index).fillna(0.0))
                slog_use = np.log1p((tpm * (1 - dilute_p) + bg * dilute_p).clip(lower=0))
            else:
                slog_use = slog
            pred = predict(slog_use)
            if pred is None:
                continue
            n += 1
            for m in METHODS:
                ok[m] += (_primary(pred[m]) == truth)
        return {m: ok[m] / n for m in METHODS}, n

    print(f"tests: {len(tests)} samples, {len(set(t for t,_ in tests))} primary types, "
          f"{len(ref_codes)} reference cohorts", file=sys.stderr)
    print(f"\n{'p':>4s} | " + " | ".join(f"{m:>7s}" for m in METHODS) + " | n")
    print("-" * 72)
    for p in [0.0, 0.3, 0.5, 0.7]:
        acc, n = accuracy(p)
        print(f"{p:>4.1f} | " + " | ".join(f"{acc[m]:>7.3f}" for m in METHODS) + f" | {n}")


if __name__ == "__main__":
    main()
