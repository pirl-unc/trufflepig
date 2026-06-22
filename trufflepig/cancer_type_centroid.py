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
  against the FULL subtype-aware cohort reference (~118 representative-cohort medoids
  incl. BRCA_Basal / the SARC_* subtypes / rare types — NOT the 33 subtype-averaged
  TCGA bulk centroids, which mis-place subtype-shifted tumors like basal breast).
  Robust coarse tissue-of-origin with no marker-panel saturation. (Whole-transcriptome
  correlation, NOT a discriminative-gene subset — the subset sharpened some calls but
  broke others, e.g. it dropped a real sarcoma to rank 5.)

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
    """``(log1p reference DataFrame [Symbol x COHORT], informative-gene index)``, cached.

    The reference is the FULL set of representative cohort medoids from pirlygenes
    (``available_representative_cohorts`` -> ``cohort_expression_views().clean_tpm``) —
    ~118 cohorts INCLUDING subtypes (BRCA_Basal / BRCA_LumA / …, the SARC_* subtypes,
    and the rare / non-TCGA types). This is deliberately NOT the 33 TCGA bulk
    centroids: a bulk pan-cancer centroid is subtype-averaged (the BRCA column is
    luminal-dominated), so a basal/TNBC or otherwise subtype-shifted tumor
    mis-correlates — the canonical failure was hcc1395 (basal/EMT breast) drifting to
    SKCM because it matched neither the luminal BRCA bulk centroid nor real melanoma.
    A subtype-resolved reference (BRCA_Basal present) fixes that.

    The bare broad ``SARC`` pseudo-cohort is dropped (sarcoma is a grouping, resolved
    by its subtypes — never used as a single centroid). Falls back to the 33-cohort
    pan-cancer bulk only if the expanded reference is unavailable.
    """
    if "ref" in _bulk_cache:
        return _bulk_cache["ref"]
    bulk = None
    try:
        from pirlygenes.expression import (
            available_representative_cohorts,
            cohort_expression_views,
        )

        reps = [c for c in available_representative_cohorts() if str(c) != "SARC"]
        view = cohort_expression_views(reps).clean_tpm
        if "Symbol" in getattr(view, "columns", []):
            view = view.set_index("Symbol")
        view = view.drop(
            columns=[c for c in ("Ensembl_Gene_ID", "Description") if c in view.columns],
            errors="ignore",
        )
        view = view[~view.index.duplicated(keep="first")].apply(
            pd.to_numeric, errors="coerce"
        )
        if view.shape[1] >= 40:
            bulk = np.log1p(view.clip(lower=0.0).fillna(0.0))
    except Exception:  # noqa: BLE001
        bulk = None
    if bulk is None:  # fallback: 33 TCGA bulk centroids
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
    """Average-rank transform with pandas-compatible tie/NaN behavior."""
    values = np.asarray(arr, dtype=float)
    ranks = np.full(values.shape, np.nan, dtype=float)
    ok = ~np.isnan(values)
    valid = values[ok]
    if valid.size == 0:
        return ranks

    order = np.argsort(valid, kind="mergesort")
    sorted_values = valid[order]
    group_start = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    starts = np.flatnonzero(group_start)
    counts = np.diff(np.r_[starts, valid.size])
    # Ranks are 1-based. For a tie group with zero-based start ``s`` and
    # length ``n``, the average rank is mean(s + 1, ..., s + n).
    group_ranks = starts + (counts + 1) / 2.0
    sorted_ranks = np.repeat(group_ranks, counts)
    valid_ranks = np.empty(valid.size, dtype=float)
    valid_ranks[order] = sorted_ranks
    ranks[ok] = valid_ranks
    return ranks


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
    / CNS / Melanoma / Heme / Germ cell / Embryonal). Returns ``{group: top-K-mean
    rho}`` sorted descending — the data-derived coarse-lineage prior.
    """
    corr = centroid_correlations(sample_tpm_by_symbol)
    return _coarse_from_correlations(corr)


# Each compartment is scored by the MEAN of its top-K best-correlating cohorts, not
# its single best. With the expanded subtype-aware reference a few rare cohorts are
# "promiscuous" (a noisy medoid like SARC_PEC free-rides on the housekeeping bulk of
# a whole-transcriptome Spearman and correlates ~0.8 with almost anything). Taking
# the single best cohort lets one such cohort define a whole compartment; requiring
# the top-K to agree means a prostate tumor can't summon 3 sarcoma cohorts — only the
# one promiscuous one — so the spurious compartment win collapses.
_COMPARTMENT_TOPK = 3


def _coarse_from_correlations(corr):
    """``{group: top-K-mean rho}`` from an already-computed ``centroid_correlations``.

    Split out so the ranker (which already holds the correlation Series) can derive
    the compartment without a second centroid pass. Robust aggregation: each group's
    score is the mean of its top-``_COMPARTMENT_TOPK`` cohort correlations (NaNs
    dropped), not its single best — see ``_COMPARTMENT_TOPK``.
    """
    if corr is None or len(corr) == 0:
        return pd.Series(dtype=float)
    try:
        from pirlygenes.gene_sets_cancer import cancer_lineage_group
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float)
    by_group: dict[str, list[float]] = {}
    for code, rho in corr.items():
        grp = cancer_lineage_group(code)
        rho = float(rho)
        if not grp or rho != rho:  # skip ungrouped / NaN
            continue
        by_group.setdefault(grp, []).append(rho)
    groups: dict[str, float] = {}
    for grp, rhos in by_group.items():
        rhos.sort(reverse=True)
        groups[grp] = float(np.mean(rhos[: min(_COMPARTMENT_TOPK, len(rhos))]))
    return pd.Series(groups).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Stage 1 of the hierarchical call: the histogenesis compartment (LOCKED).
# --------------------------------------------------------------------------- #
# When the compartment wins by a clear top-K-mean rho margin we trust it to *restrict*
# stage-2 leaves to that compartment; below the margin we abstain (no restriction) and
# defer to the marker ranker rather than risk excluding the true leaf. 0.025 in
# Spearman-rho: on the local set the confident-correct compartments sit at margin
# 0.04-0.08 while the residual near-tie mistakes (a basal/EMT or pure-cell-line profile
# grazing an adjacent compartment) sit at 0.009-0.019, so this cleanly separates "act"
# from "defer". Raised from 0.02 after the subtype-aware-reference redesign.
_COMPARTMENT_CONFIDENT_MARGIN = 0.025


def compartment_call(sample_tpm_by_symbol, _corr=None):
    """Stage 1 of the hierarchical cancer-type call: the histogenesis compartment.

    Whole-profile correlation against the FULL subtype-aware cohort reference
    (see :func:`_bulk_centroids`), aggregated by pirlygenes ``cancer_lineage_group``
    (Epithelial / Sarcoma / Hematolymphoid / Melanoma / Neuroendocrine / CNS /
    Germ cell / Embryonal) via a robust top-K-mean per group. On the local blind
    truth set it gets the compartment right on 15/18, and — the property that makes
    it safe to *act* on — every CONFIDENT call is correct, while the residual
    near-ties (basal/EMT or pure-cell-line profiles grazing an adjacent compartment)
    fall below the margin and defer to the marker ranker. The marker-saturation-
    immune coarse call the leaf screen cannot make reliably. **Sarcoma is a broad
    grouping, never a leaf** (the SARC-is-broad rule); stage 2 narrows *within* the
    pinned compartment.

    Parameters
    ----------
    sample_tpm_by_symbol : dict[str, float]
        Sample expression keyed by gene symbol.
    _corr : pandas.Series | None
        Pre-computed :func:`centroid_correlations` result, to avoid a second
        centroid pass when the caller already has it. Internal optimization.

    Returns
    -------
    dict with keys:
      ``compartment``  top lineage group, or ``None`` if no correlation was possible
      ``score``        best rho within that compartment
      ``runner_up``    second-best compartment (or ``None``)
      ``margin``       ``score`` minus runner-up score (``0.0`` if only one group)
      ``confident``    ``margin >= _COMPARTMENT_CONFIDENT_MARGIN`` — gate for whether
                       callers should restrict stage-2 leaves to this compartment
      ``scores``       full ``{group: rho}`` Series, sorted descending
    """
    corr = centroid_correlations(sample_tpm_by_symbol) if _corr is None else _corr
    groups = _coarse_from_correlations(corr)
    if groups.empty:
        return {
            "compartment": None, "score": float("nan"), "runner_up": None,
            "margin": 0.0, "confident": False, "scores": groups,
        }
    top = str(groups.index[0])
    score = float(groups.iloc[0])
    runner_up = str(groups.index[1]) if len(groups) > 1 else None
    margin = score - float(groups.iloc[1]) if len(groups) > 1 else 0.0
    return {
        "compartment": top, "score": score, "runner_up": runner_up,
        "margin": float(margin),
        "confident": float(margin) >= _COMPARTMENT_CONFIDENT_MARGIN,
        "scores": groups,
    }


def in_compartment(code, compartment):
    """Does leaf ``code`` belong to histogenesis ``compartment``?

    Pure membership test via ``cancer_lineage_group`` — the basis for stage-1 leaf
    restriction. Sarcoma is broad: every ``SARC``/``SARC_*`` subtype maps to the
    ``Sarcoma`` group, so all of them are in-compartment when the call is Sarcoma,
    and none of them is ever promoted to a single sarcoma leaf by this test.
    Codes with no known lineage group are treated as in-compartment (never
    excluded on missing metadata — fail-open).
    """
    if not compartment:
        return True
    try:
        from pirlygenes.gene_sets_cancer import cancer_lineage_group
    except Exception:  # noqa: BLE001
        return True
    grp = cancer_lineage_group(str(code))
    if not grp:
        return True
    return grp == compartment


def restrict_rows_to_compartment(rows, compartment, confident):
    """Stage-1 leaf restriction for the ranker's candidate rows.

    Annotates every row with ``compartment_in_set`` and, when the compartment call
    is ``confident`` and the candidate set straddles the boundary (has both in- and
    out-of-compartment leaves), applies a **stable** sort that floats in-compartment
    leaves above out-of-compartment ones. Stable -> within each tier the incoming
    order (the marker-panel support ranking) is preserved untouched, so the only
    reorder is across the compartment boundary — exactly the saturation mis-calls.

    Never restricts to an empty set: if every candidate is out-of-compartment (e.g. a
    caller-constrained set), nothing is reordered (and the disagreement stays visible
    as ``compartment_in_set=False`` on every row). Mutates the row dicts in place and
    may reorder ``rows``.

    Returns ``(rows, restricted: bool)``.
    """
    for r in rows:
        r["compartment_in_set"] = bool(
            compartment is None or in_compartment(r["code"], compartment)
        )
    restricted = bool(
        compartment is not None
        and confident
        and any(not r["compartment_in_set"] for r in rows)
        and any(r["compartment_in_set"] for r in rows)
    )
    if restricted:
        rows.sort(key=lambda r: 0 if r["compartment_in_set"] else 1)
    return rows, restricted


# --------------------------------------------------------------------------- #
# Hallmark-gene veto: a candidate whose DEFINING markers are absent is a horrible
# fit no matter how the whole-profile correlation lands. The canonical case is a
# Melanoma/SKCM candidate on a sample with MLANA/PMEL/TYR/SOX10 all ~0 — the call
# is then a reference artifact, not melanocytic identity. This is the same idea as
# range_plausibility but (a) works off the FULL subtype-aware reference (so it can
# judge rare cohorts like SARC_PEC that have no deconvolved reference) and (b) keys
# on each type's highly-expressed type-SPECIFIC genes.
# --------------------------------------------------------------------------- #
_HALLMARK_N = 20
_HALLMARK_MIN_REF_LOGTPM = np.log1p(20.0)  # the type must actually express it highly
_HALLMARK_SPEC_MIN = float(np.log(4.0))    # >= ~4x the cross-cohort median (log space)
_HALLMARK_PRESENT_FRACTION = 0.10          # sample "expresses" it if >= 10% of ref level
# Below this fraction of hallmark genes present, the candidate is a horrible fit. Kept
# low (a true type can dilute at low purity / dropout) so the veto only fires on a
# near-total absence of the defining program, never on a merely weak one.
_HALLMARK_VETO_FIT = 0.15
_hallmark_cache: dict = {}


def _bulk_median():
    """Cross-cohort per-gene median of the reference (cached) — the background the
    hallmark specificity is measured against."""
    if "median" not in _bulk_cache:
        bulk, _ = _bulk_centroids()
        _bulk_cache["median"] = bulk.median(axis=1)
    return _bulk_cache["median"]


def _hallmark_genes(code):
    """``[(gene, reference_tpm), …]`` — ``code``'s highly-expressed, type-SPECIFIC
    markers (cached). Genes where the cohort reference is both HIGH (really expressed)
    and well above the cross-cohort median; their absence in a sample is the
    horrible-fit signal."""
    if code in _hallmark_cache:
        return _hallmark_cache[code]
    bulk, _ = _bulk_centroids()
    genes: list = []
    if code in bulk.columns:
        col = bulk[code]
        spec = col - _bulk_median()
        mask = (col >= _HALLMARK_MIN_REF_LOGTPM) & (spec >= _HALLMARK_SPEC_MIN)
        ranked = spec[mask].sort_values(ascending=False).head(_HALLMARK_N)
        genes = [(str(g), float(np.expm1(col[g]))) for g in ranked.index]
    _hallmark_cache[code] = genes
    return genes


def hallmark_fit(code, sample_tpm_by_symbol):
    """Fraction of ``code``'s hallmark markers the sample expresses (>= 10% of the
    cohort's reference level).

    1.0 = the defining program is present; ~0.0 = horrible fit (the type's defining
    genes are absent — the candidate is a reference/correlation artifact). Returns
    1.0 (abstain) when too few hallmark genes are available to judge — it only ever
    *flags* a clear absence, never invents support.
    """
    genes = _hallmark_genes(code)
    if len(genes) < 5:
        return 1.0
    hits = 0
    for gene, ref_tpm in genes:
        sv = float(sample_tpm_by_symbol.get(gene, 0.0) or 0.0)
        if sv >= max(1.0, _HALLMARK_PRESENT_FRACTION * ref_tpm):
            hits += 1
    return hits / len(genes)


def hallmark_veto(code, sample_tpm_by_symbol):
    """``True`` if ``code`` is a horrible hallmark fit (defining markers near-absent)
    and should be dropped from the candidate set. Never fires when there aren't enough
    hallmark genes to judge."""
    genes = _hallmark_genes(code)
    if len(genes) < 5:
        return False
    return hallmark_fit(code, sample_tpm_by_symbol) < _HALLMARK_VETO_FIT


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
