"""Optional LEARNED cancer-type co-signal — a multinomial logistic regression over the per-gene
z-score feature space, trained on the representative-cohort samples.

This is a deliberately OPTIONAL second opinion. It is **not** imported or called by the default
analysis pipeline — the hierarchical compartment→leaf system (curated panels, vetoes, family logic,
interpretability, 145+ code coverage incl. rare/zero-sample types, structured-contamination handling)
remains the primary path. This module exists so a caller can opt in to a cheap, purity-robust learned
vote and fuse it as one more signal.

Measured (scripts/zscore_classifier_ab.py, 266 samples / 54 primary types, leakage-free 5-fold CV):
  - LR (z-score) 0.865 clean, > nearest-centroid 0.835;
  - perfectly purity-robust to a GENERIC (pan-cancer-mean) contaminant — 0.865 flat to 30% purity,
    where nearest-centroid collapses to ~0.16.
Honest caveats it does NOT fix: (1) STRUCTURED contamination (liver→LIHC, immune-rich→heme) still
misleads any linear model; (2) rare types with ≤5 (or zero) training samples can't get a reliable
boundary — class_weight='balanced' helps but does not invent data; (3) it is not interpretable. Use
it as a co-signal, never as the sole call.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

# Informative-gene subset size (top cross-sample variance) — matches the validation harness.
_N_GENES = 2000
_MIN_SHARED_GENES = 200


@lru_cache(maxsize=1)
def _trained_model():
    """``(pipeline, gene_symbols, classes)`` — lazily trained once, cached.

    Features are log1p(clean-TPM) over a fixed high-variance SYMBOL index; the pipeline z-scores
    (StandardScaler) then fits a balanced multinomial logistic regression. Returns ``None`` if the
    reference samples or sklearn are unavailable (callers then skip the co-signal)."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from oncoref.normalization import clean_tpm
        from pirlygenes.expression.accessors import (
            available_representative_cohorts,
            representative_cohort_samples,
        )
    except Exception:  # noqa: BLE001 — optional dependency / environment
        return None

    try:
        frames, labels, base = [], [], None
        for code in sorted(available_representative_cohorts()):
            d = (
                representative_cohort_samples(code)
                .drop_duplicates("Ensembl_Gene_ID")
                .set_index("Ensembl_Gene_ID")
            )
            cols = [c for c in d.columns if c != "Symbol"]
            if base is None:
                base = d[["Symbol"]].copy()
            frames.append(d[cols])
            labels += [code] * len(cols)
        expr = pd.concat(frames, axis=1).dropna(how="any")
        gt = pd.DataFrame(
            {"Ensembl_Gene_ID": expr.index, "Symbol": base["Symbol"].reindex(expr.index).values}
        )
        clean = clean_tpm(expr.astype(float), gene_table=gt)
        clean.index = base["Symbol"].reindex(expr.index).values
        clean = clean[~clean.index.duplicated(keep="first")]
        logc = np.log1p(clean.clip(lower=0))
        var = logc.var(axis=1)
        genes = list(var.sort_values(ascending=False).index[:_N_GENES])
        X = np.nan_to_num(logc.loc[genes].T.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.asarray(labels)
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        )
        pipe.fit(X, y)
        return pipe, genes, list(pipe.classes_)
    except Exception:  # noqa: BLE001 — never let an optional co-signal crash a caller
        return None


def classify_expression(sample_tpm_by_symbol, top_k=5):
    """Optional learned co-signal: ``[(code, probability), …]`` top-``top_k``, sorted descending.

    ``sample_tpm_by_symbol`` is clean-TPM keyed by gene symbol (e.g. ``_build_sample_tpm_by_symbol``).
    Returns ``[]`` when the model can't be trained or the sample shares too few genes with the
    feature index. This is a SECOND OPINION — fuse it, don't trust it alone (see module docstring).
    """
    if not sample_tpm_by_symbol:
        return []
    if len(sample_tpm_by_symbol) < _MIN_SHARED_GENES:
        return []
    trained = _trained_model()
    if trained is None:
        return []
    pipe, genes, classes = trained
    present = sum(1 for g in genes if g in sample_tpm_by_symbol)
    if present < _MIN_SHARED_GENES:
        return []
    vec = np.log1p(
        np.asarray([max(0.0, float(sample_tpm_by_symbol.get(g, 0.0) or 0.0)) for g in genes])
    )
    try:
        proba = pipe.predict_proba(vec.reshape(1, -1))[0]
    except Exception:  # noqa: BLE001
        return []
    order = np.argsort(proba)[::-1][:top_k]
    return [(str(classes[i]), float(proba[i])) for i in order]
