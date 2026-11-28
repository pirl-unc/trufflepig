#!/usr/bin/env python
"""Compare classification STRATEGIES x NORMALIZATIONS x FEATURE SETS on hard + typical cases.

A RESEARCH HARNESS — nothing here is wired into production. It exists to answer "which
representation actually classifies the hard cases best?" without baking any choice in.

For each representative-cohort sample we predict the cancer type with several strategies and report
exact / family / lineage accuracy split by a TYPICAL subset (clear, well-separated types) and a HARD
subset (sibling pairs, rare/pediatric/heme types, basal-drift, biphasic — the documented failure
modes). Candidates = the full ~120 cohort set (realistic open-set), so a strategy is penalised for
being confused by the *other* 100 types, not just the subset.

Strategies:
  PANEL-BASED (score = mean over the type's signature panel of a per-gene normalized value; argmax):
    hk_ratio          sample / housekeeping-median (legacy)
    cohort_pct_33     midrank of sample_hk among the 33 TCGA cohort medians (HK)
    cohort_pct_170    midrank of sample_hk among the 170-cohort reference (118 cancer + 50 normal)
    within_pct        within-sample percentile (reference-free, purity-robust)
    zscore_33         (log1p(sample) - cohort_log_mean) / cohort_log_std, 33 cohorts
    combined_170      cohort_pct_170 * within_pct  (the production signature filter)
  WHOLE-PROFILE (all shared genes; argmax over cohort centroids):
    spearman_centroid Spearman(sample_log, cohort_log_profile)  (rank-based, scale-invariant)
    cosine_centroid   cosine(sample_log, cohort_log_profile)

Run:  python scripts/classification_strategy_eval.py
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from pirlygenes.expression.accessors import (
    available_representative_cohorts,
    representative_cohort_samples,
)
from pirlygenes.gene_sets_cancer import cancer_lineage_group
from trufflepig.clean_tpm import normalize_to_reference_space
from trufflepig.plot_tumor_expr import _sample_expression_by_symbol
from trufflepig.plot_embedding import _full_cohort_hk_reference, _full_cohort_signature_panels
from trufflepig.reference import cancer_reference_expression
from trufflepig.tumor_purity import _cached_reference_matrices

TYPICAL = ["COAD", "SKCM", "LIHC", "KIRC", "THCA", "PRAD", "LUAD", "GBM", "OV", "LAML"]
HARD = ["READ", "ESCA", "BRCA_Basal", "ATRT", "SCLC", "BL", "NBL", "MBL", "SARC_SYN", "MESO",
        "DLBC", "HNSC_HPVpos"]


def _base(code):
    """Strip a molecular-subtype suffix to the parent cohort (BRCA_Basal -> BRCA)."""
    return code.split("_")[0] if "_" in code else code


def _lin(code):
    try:
        return cancer_lineage_group(code)
    except Exception:
        return None


def _match(pred, truth):
    if pred == truth:
        return "exact"
    if _base(pred) == _base(truth):
        return "family"
    lp, lt = _lin(pred), _lin(truth)
    if lp and lt and lp == lt:
        return "lineage"
    return "miss"


def midrank(value, ref_vals):
    if ref_vals is None or len(ref_vals) == 0:
        return 0.5
    below = np.sum(ref_vals < value)
    equal = np.sum(np.isclose(ref_vals, value, atol=1e-6))
    return float((below + 0.5 * equal) / len(ref_vals))


def main():
    reps = set(available_representative_cohorts())
    typical = [t for t in TYPICAL if t in reps]
    hard = [t for t in HARD if t in reps]
    print(f"typical={len(typical)} hard={len(hard)}  (missing from reps: "
          f"{sorted((set(TYPICAL) | set(HARD)) - reps)})", file=sys.stderr)

    # --- references -------------------------------------------------------------------------------
    full_hk = _full_cohort_hk_reference()                       # 170 cohorts, HK, Symbol-indexed
    m33 = _cached_reference_matrices(normalize="housekeeping")
    hk33 = m33["expr_matrix"]                                   # 33 cohorts, HK, Symbol-indexed
    panels = _full_cohort_signature_panels()                   # panels for ~120 cohorts
    candidates = [c for c in panels if panels[c]]               # open-set candidate codes

    cref = cancer_reference_expression(format="wide").drop_duplicates("Symbol").set_index("Symbol")
    prof_cols = [c for c in cref.columns if str(c).endswith("_TPM_clean")]
    cent = np.log1p(cref[prof_cols].astype(float).fillna(0.0))  # cohort log-profiles (NaN=undetected=0)
    cent_codes = [c.replace("_TPM_clean", "") for c in prof_cols]
    # per-gene cohort log mean/std (33) for the z-score strategy
    log33 = np.log1p(hk33.astype(float))
    z_mean, z_std = log33.mean(axis=1), log33.std(axis=1).replace(0, np.nan)

    STRATS = ["hk_ratio", "cohort_pct_33", "cohort_pct_170", "within_pct", "zscore_33",
              "combined_170", "spearman_centroid", "cosine_centroid"]

    def predict(sample_raw, sample_hk, within, strat):
        if strat in ("spearman_centroid", "cosine_centroid"):
            s = pd.Series(sample_raw, dtype=float)
            s = np.log1p(s[s.index.isin(cent.index)])
            common = s.index.intersection(cent.index)
            if len(common) < 50:
                return None
            sv = s.loc[common].to_numpy()
            C = cent.loc[common, :].to_numpy()  # genes x cohorts
            if strat == "spearman_centroid":
                sr = pd.Series(sv).rank().to_numpy()
                Cr = pd.DataFrame(C).rank(axis=0).to_numpy()
                scores = np.corrcoef(sr, Cr.T)[0, 1:]
            else:
                svn = sv - sv.mean()
                Cn = C - C.mean(axis=0)
                num = svn @ Cn
                den = (np.linalg.norm(svn) * np.linalg.norm(Cn, axis=0)) + 1e-9
                scores = num / den
            if scores is None or not np.any(np.isfinite(scores)):
                return None
            return cent_codes[int(np.nanargmax(scores))]
        # panel-based
        best, best_code = -1.0, None
        for c in candidates:
            vals = []
            for g in panels[c]:
                if strat == "hk_ratio":
                    vals.append(float(sample_hk.get(g, 0.0)))
                elif strat == "within_pct":
                    vals.append(float(within.get(g, 0.5)))
                elif strat == "cohort_pct_33":
                    rv = hk33.loc[g].to_numpy() if g in hk33.index else None
                    vals.append(midrank(float(sample_hk.get(g, 0.0)), rv))
                elif strat in ("cohort_pct_170", "combined_170"):
                    rv = full_hk.loc[g].to_numpy(float) if g in full_hk.index else None
                    rv = rv[~np.isnan(rv)] if rv is not None else None
                    cp = midrank(float(sample_hk.get(g, 0.0)), rv)
                    vals.append(cp * float(within.get(g, 0.5)) if strat == "combined_170" else cp)
                elif strat == "zscore_33":
                    # z stats are from log(HK-normalized cohort), so the sample must be HK too
                    if g in z_mean.index and np.isfinite(z_std.get(g, np.nan)):
                        vals.append((np.log1p(float(sample_hk.get(g, 0.0))) - z_mean[g]) / z_std[g])
            if vals:
                sc = float(np.mean(vals))
                if sc > best:
                    best, best_code = sc, c
        return best_code

    # --- evaluate ---------------------------------------------------------------------------------
    results = {s: {"typical": [], "hard": []} for s in STRATS}
    for subset_name, types in (("typical", typical), ("hard", hard)):
        for t in types:
            raw = representative_cohort_samples(t).drop_duplicates("Ensembl_Gene_ID")
            scols = [c for c in raw.columns if c not in ("Ensembl_Gene_ID", "Symbol")][:4]
            for col in scols:
                samp = pd.DataFrame({"ensembl_gene_id": raw["Ensembl_Gene_ID"],
                                     "gene_symbol": raw["Symbol"], "TPM": raw[col].astype(float)})
                try:
                    conf = normalize_to_reference_space(samp, value_cols=["TPM"])
                    sample_raw, sample_hk = _sample_expression_by_symbol(conf)
                except Exception:
                    sample_raw, sample_hk = _sample_expression_by_symbol(samp)
                within = pd.Series(sample_raw, dtype=float).rank(pct=True, method="average").to_dict()
                for s in STRATS:
                    pred = predict(sample_raw, sample_hk, within, s)
                    if pred is not None:
                        results[s][subset_name].append(_match(pred, t))

    # --- report -----------------------------------------------------------------------------------
    def acc(matches, levels):
        if not matches:
            return float("nan")
        return sum(m in levels for m in matches) / len(matches)

    print("\n=== classification strategy eval (open-set, ~120 candidates) ===")
    print(f"{'strategy':18} | {'TYP exact':>9} {'TYP fam':>8} {'TYP lin':>8} | "
          f"{'HARD exact':>10} {'HARD fam':>9} {'HARD lin':>9}")
    print("-" * 92)
    for s in STRATS:
        ty, ha = results[s]["typical"], results[s]["hard"]
        print(f"{s:18} | {acc(ty,{'exact'}):>9.2f} {acc(ty,{'exact','family'}):>8.2f} "
              f"{acc(ty,{'exact','family','lineage'}):>8.2f} | "
              f"{acc(ha,{'exact'}):>10.2f} {acc(ha,{'exact','family'}):>9.2f} "
              f"{acc(ha,{'exact','family','lineage'}):>9.2f}   (n_typ={len(ty)} n_hard={len(ha)})")


if __name__ == "__main__":
    main()
