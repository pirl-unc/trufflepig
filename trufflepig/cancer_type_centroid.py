"""Data-derived cancer-type matching: whole-profile centroid correlation + a
range-restriction plausibility check.

This is the data-derived primitive of the cancer-type re-architecture (#83). The
curated marker panels saturate for *promiscuous* lineages — the (removed)
MESENCHYMAL panel and the SQUAMOUS/HNSC panel fire on the stroma/squamous-
contamination present in essentially every solid tumor — so a stroma-heavy or
low-purity tumor mis-calls to SARC or HNSC. Matching the *whole expression
profile* against the reference cohort centroids is immune to that: it can't be
fooled by a handful of non-specific markers.

Two complementary signals, neither of which uses the marker panels:

- :func:`centroid_correlations` — Spearman correlation of the sample's full profile
  against each bulk TCGA cohort centroid (median per gene). Robust coarse
  tissue-of-origin: on the local blind truth set it puts both sarcomas, prostate
  and bladder at #1 with no marker-panel saturation. (Whole-transcriptome
  correlation, NOT a discriminative-gene subset — the subset sharpened some calls
  but broke others, e.g. it dropped a real sarcoma to rank 5.)

- :func:`range_plausibility` — fraction of a candidate type's most-specific tumor
  markers whose sample value falls inside that type's plausible [p1, p99] range
  (estimated log-normally from the deconvolved tumor-only q1/q3). A *sanity veto*:
  it rejects e.g. HNSC on a sarcoma (squamous markers in-range ~17% of the time on
  non-squamous samples vs ~90-95% for the true type). Only trustworthy at adequate
  purity — a true type's markers also dilute out of range at low purity — so
  callers must gate it (see :func:`reconcile`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Whole-transcriptome correlation is the robust coarse signal; we still drop the
# bottom of the distribution (genes near zero everywhere carry no information and
# only add rank-tie noise) by requiring a cohort-reference value above this floor
# in at least one cohort.
_MIN_INFORMATIVE_LOG_TPM = np.log1p(1.0)

# Range-restriction plausibility (see module docstring + reconcile()).
_PLAUSIBILITY_N_MARKERS = 40
_PLAUSIBILITY_MIN_MARKERS = 8
_P99_Z = 2.3263478740408408  # norm.ppf(0.99)

_bulk_cache: dict = {}
_dec_cache: dict = {}
_deconv_centroid_cache: dict = {}


# --------------------------------------------------------------------------- #
# Reference centroids (bulk) and range references (deconvolved tumor-only).
# --------------------------------------------------------------------------- #
def _bulk_centroids():
    """``(log1p bulk DataFrame [Symbol x CODE], informative-gene index)``, cached.

    The bulk pan-cancer reference is per-cohort median TPM (includes stroma/TME,
    which is exactly why correlating against it is robust for stroma-heavy tumors:
    the reference carries the same compartments the sample does).
    """
    if "ref" in _bulk_cache:
        return _bulk_cache["ref"]
    from .reference import pan_cancer_expression

    pan = (
        pan_cancer_expression(technical_rna_normalize=True)
        .drop_duplicates("Symbol")
        .set_index("Symbol")
    )
    cols = [c for c in pan.columns if c.endswith("_TPM")]
    bulk = np.log1p(pan[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0))
    bulk.columns = [c[:-4] for c in cols]
    informative = bulk.index[(bulk.max(axis=1) >= _MIN_INFORMATIVE_LOG_TPM)]
    out = (bulk, informative)
    _bulk_cache["ref"] = out
    return out


def _deconvolved_range():
    """``{med, q1, q3, markers}`` from the deconvolved tumor-only reference, cached.

    ``markers[code]`` = that cohort's most cohort-specific genes (high tumor-only
    median vs the other cohorts), used for the plausibility check. ``None`` if the
    deconvolved reference is unavailable (plausibility then abstains everywhere).
    """
    if "ref" in _dec_cache:
        return _dec_cache["ref"]
    try:
        from .reference import tcga_deconvolved_expression

        dec = tcga_deconvolved_expression(technical_rna_normalize=True)
        med = dec.pivot_table(
            index="symbol", columns="cancer_code", values="tumor_tpm_median",
            aggfunc="median",
        )
        q1 = dec.pivot_table(
            index="symbol", columns="cancer_code", values="tumor_tpm_q1",
            aggfunc="median",
        )
        q3 = dec.pivot_table(
            index="symbol", columns="cancer_code", values="tumor_tpm_q3",
            aggfunc="median",
        )
        logmed = np.log1p(med)
        markers = {}
        for code in med.columns:
            others = logmed.drop(columns=[code]).median(axis=1)
            spec = (logmed[code] - others).dropna().sort_values(ascending=False)
            markers[str(code)] = list(spec.head(_PLAUSIBILITY_N_MARKERS).index)
        ref = {"med": med, "q1": q1, "q3": q3, "markers": markers}
    except Exception:  # noqa: BLE001
        ref = None
    _dec_cache["ref"] = ref
    return ref


# --------------------------------------------------------------------------- #
# Whole-profile centroid correlation.
# --------------------------------------------------------------------------- #
def _rankdata(arr):
    """Average-rank transform (Spearman = Pearson on ranks). numpy/pandas only."""
    return pd.Series(arr).rank(method="average").to_numpy()


# --------------------------------------------------------------------------- #
# Tumor-only (deconvolved) centroids — the #83 increment-4 uniform space.
# --------------------------------------------------------------------------- #
def _deconvolved_centroids():
    """``(log1p tumor-only reference [symbol x code], code list)`` across EVERY
    deconvolved cohort, cached.

    Combines the TCGA (``tcga_deconvolved``) and subtype/observed-bulk
    (``subtype_deconvolved``) tumor-only medians — ~63 codes including the subtypes
    and rare/non-TCGA types (NUTM, RB, ATRT, the SARC_* subtypes, …) that the 33
    bulk centroids can't reach. This is the reference side of increment 4: a sample
    whose tumor compartment has been recovered (``attr_tumor_tpm``) is matched here,
    in a uniform tumor-only space — broadening coverage AND removing the stroma
    confound (both sides are tumor-only, so the bulk-vs-deconvolved mismatch is gone).
    """
    if "ref" in _deconv_centroid_cache:
        return _deconv_centroid_cache["ref"]
    frames = []
    try:
        from .reference import (
            tcga_deconvolved_expression,
            subtype_deconvolved_expression,
        )

        for fn in (tcga_deconvolved_expression, subtype_deconvolved_expression):
            d = fn(technical_rna_normalize=True)
            frames.append(
                d.pivot_table(
                    index="symbol", columns="cancer_code",
                    values="tumor_tpm_median", aggfunc="median",
                )
            )
    except Exception:  # noqa: BLE001
        frames = []
    if not frames:
        out = (pd.DataFrame(), [])
        _deconv_centroid_cache["ref"] = out
        return out
    combined = frames[0]
    for f in frames[1:]:
        extra = [c for c in f.columns if c not in combined.columns]
        combined = combined.join(f[extra], how="outer")
    out = (np.log1p(combined), list(combined.columns))
    _deconv_centroid_cache["ref"] = out
    return out


def tumor_only_correlations(tumor_sample_by_symbol, restrict_to=None):
    """Spearman of a TUMOR-ONLY sample profile (the deconvolved tumor compartment,
    e.g. ``attr_tumor_tpm``) against the tumor-only reference centroids for every
    deconvolved cohort (~63 codes incl. subtypes / NUTM).

    The uniform-space match of #83 increment 4: because both the sample and the
    references are tumor-only, the stroma confound that mis-calls bulk samples to
    SARC/HNSC is gone, and coverage extends to the subtypes/rare types absent from
    the bulk centroids. Returns ``{code: rho}`` sorted descending.
    """
    ref, codes = _deconvolved_centroids()
    if ref.empty or not tumor_sample_by_symbol:
        return pd.Series(dtype=float)
    sample = pd.Series(tumor_sample_by_symbol, dtype=float)
    sample = np.log1p(sample[sample.index.notna()].clip(lower=0))
    shared = sample.index.intersection(ref.index)
    if len(shared) < 200:
        return pd.Series(dtype=float)
    sample_ranks = _rankdata(sample.loc[shared].to_numpy())
    R = ref.loc[shared]
    scored = codes if restrict_to is None else [c for c in codes if c in set(restrict_to)]
    out = {}
    for code in scored:
        v = R[code].to_numpy()
        ok = ~np.isnan(v)
        if ok.sum() < 100:
            continue
        out[code] = float(np.corrcoef(sample_ranks[ok], _rankdata(v[ok]))[0, 1])
    return pd.Series(out).sort_values(ascending=False)


def centroid_correlations(sample_tpm_by_symbol, restrict_to=None):
    """Spearman correlation of the sample's whole profile against each cohort centroid.

    Parameters
    ----------
    sample_tpm_by_symbol : dict[str, float]
        Sample expression keyed by gene symbol (e.g. ``_build_sample_tpm_by_symbol``).
    restrict_to : iterable[str] | None
        If given, only score these cohort codes.

    Returns
    -------
    pandas.Series  {cohort_code: spearman_rho}, sorted descending (may be empty).
    """
    bulk, informative = _bulk_centroids()
    if not sample_tpm_by_symbol:
        return pd.Series(dtype=float)
    sample = pd.Series(sample_tpm_by_symbol, dtype=float)
    sample = np.log1p(sample[sample.index.notna()].clip(lower=0))
    shared = sample.index.intersection(informative)
    if len(shared) < 200:
        return pd.Series(dtype=float)
    sample_ranks = _rankdata(sample.loc[shared].to_numpy())
    ref = bulk.loc[shared]
    codes = list(ref.columns) if restrict_to is None else [
        c for c in ref.columns if c in set(restrict_to)
    ]
    out = {}
    for code in codes:
        ref_ranks = _rankdata(ref[code].to_numpy())
        out[code] = float(np.corrcoef(sample_ranks, ref_ranks)[0, 1])
    return pd.Series(out).sort_values(ascending=False)


def coarse_lineage_scores(sample_tpm_by_symbol):
    """Aggregate centroid correlations into coarse histogenesis groups.

    Uses pirlygenes' ``cancer_lineage_group`` (Sarcoma / Epithelial / Neuroendocrine
    / CNS / Melanoma / Heme / Germ cell / Embryonal). Returns ``{group: best_rho}``
    sorted descending — the data-derived coarse-lineage prior.
    """
    corr = centroid_correlations(sample_tpm_by_symbol)
    if corr.empty:
        return pd.Series(dtype=float)
    try:
        from pirlygenes.gene_sets_cancer import cancer_lineage_group
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float)
    groups: dict[str, float] = {}
    for code, rho in corr.items():
        grp = cancer_lineage_group(code)
        if not grp:
            continue
        if grp not in groups or rho > groups[grp]:
            groups[grp] = float(rho)
    return pd.Series(groups).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Range-restriction plausibility.
# --------------------------------------------------------------------------- #
def range_plausibility(code, sample_tpm_by_symbol):
    """Fraction of ``code``'s most-specific markers whose sample value is within the
    cohort's estimated [p1, p99] range (log-normal from deconvolved q1/q3).

    1.0 = fully plausible. Returns 1.0 (abstain) for codes with no deconvolved
    reference or too few assessable markers — it only ever *rejects*.
    """
    ref = _deconvolved_range()
    if not ref or code not in ref["markers"]:
        return 1.0
    med, q1, q3 = ref["med"], ref["q1"], ref["q3"]
    in_range = 0
    assessed = 0
    for gene in ref["markers"][code]:
        sv = sample_tpm_by_symbol.get(gene)
        if sv is None:
            continue
        try:
            a = float(q1.at[gene, code])
            b = float(q3.at[gene, code])
            m = float(med.at[gene, code])
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(m) and b > a):
            continue
        sigma = (np.log1p(b) - np.log1p(a)) / 1.349  # IQR -> log-normal sigma
        lo = float(np.expm1(np.log1p(m) - _P99_Z * sigma))
        hi = float(np.expm1(np.log1p(m) + _P99_Z * sigma))
        assessed += 1
        if lo <= float(sv) <= hi:
            in_range += 1
    if assessed < _PLAUSIBILITY_MIN_MARKERS:
        return 1.0
    return in_range / assessed
