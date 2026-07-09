# Licensed under the Apache License, Version 2.0

"""Statistically-principled fusion of independent tumor-purity estimates.

trufflepig estimates purity several near-independent ways — a cancer-type expression **signature**,
a **lineage** marker panel, an ESTIMATE-style stroma/immune surrogate, and an NNLS **decomposition**
(itself a spread of template hypotheses). The historical combiner (`tumor_purity._combine_purity_
estimates`) blends these with hand-tuned anchor/deprioritise rules, and the fragility gate then
widens or rejects after the fact. Both are ad-hoc.

This module offers the principled alternative: treat each method's purity as a **measurement with an
uncertainty** and combine them with a **random-effects meta-analysis** (DerSimonian–Laird). The
combined interval then *inflates automatically* when the methods disagree, so "encode the
uncertainty when the signals conflict" is a property of the estimator rather than a threshold:

  * methods agree              → between-method variance τ² ≈ 0 → tight interval, high agreement;
  * methods disagree           → τ² large               → wide interval (the old "widen", derived);
  * one method is an outlier   → down-weighted by its own variance + the shared τ².

The fusion is done in **logit space** so the pooled point and interval are guaranteed to lie in
(0, 1); intervals are converted to a logit standard deviation via the normal approximation and
transformed back with the logistic function.

This is a pure, dependency-light function (``math`` only). It does not change any pipeline behavior
on import; a caller opts in by fusing its estimate set through :func:`integrate_purity_estimates`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

# 97.5th percentile of the standard normal — the half-width multiplier for a 95% interval.
_Z95 = 1.959963984540054
# Clamp probabilities away from {0, 1} before the logit so the transform stays finite.
_EPS = 1e-4
# Fallback logit-space SD for an estimate that supplies a point but no interval, scaled by how
# reliable the method is (weight ∈ (0, 1]); a lower weight ⇒ larger SD ⇒ less pull on the pool.
# Fallback logit-space SD for a method that reports a point but no interval (ESTIMATE, the
# decomposition residual fraction). These are directionally-informative-but-uncalibrated, so the band
# should be wide enough to stay honest yet tight enough to be actionable: 0.7 puts a single such
# method's 95% band at roughly ±25 pp around a mid-range point (e.g. 0.55 → ~[0.28, 0.79]). Genuine
# BETWEEN-method disagreement still widens the pool automatically on top of this via τ².
_DEFAULT_LOGIT_SD = 0.7


def _logit(p: float) -> float:
    p = min(1.0 - _EPS, max(_EPS, float(p)))
    return math.log(p / (1.0 - p))


def _expit(x: float) -> float:
    # Numerically stable logistic.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class PurityMeasurement:
    """One method's purity estimate as a measurement in logit space.

    ``y`` is the logit of the point estimate and ``var`` its variance in logit space. Build these
    with :func:`measurement_from_interval` rather than by hand so the interval→SD conversion and the
    no-interval fallback stay in one place.
    """

    name: str
    y: float
    var: float
    point: float


def measurement_from_interval(
    name: str,
    point: Optional[float],
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    *,
    weight: float = 1.0,
) -> Optional[PurityMeasurement]:
    """Convert a method's (point, 95%-interval) purity to a logit-space measurement.

    The interval half-width sets the logit SD via the normal approximation
    (``sd = (logit(upper) − logit(lower)) / (2·z₉₅)``); when no usable interval is supplied the SD
    falls back to ``_DEFAULT_LOGIT_SD``. In BOTH cases ``weight`` (a reliability multiplier on
    precision) inflates the SD by ``1/sqrt(weight)`` — so a less-reliable method both pulls the pool
    less AND contributes less heterogeneity. This matters for a confidently-wrong measurement: a
    broken signature with a tight native interval far from the other methods would otherwise force the
    random-effects τ² up and balloon the pooled interval; discounting it (small ``weight``) widens its
    variance so it stops dominating. Returns ``None`` for a missing/degenerate point so callers filter.
    """
    if point is None:
        return None
    try:
        p = float(point)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p) or p <= 0.0 or p >= 1.0:
        # A saturated 0/1 point carries no usable information for a logit fusion; drop it (the
        # fragility guard elsewhere already treats a saturated signature as untrustworthy).
        return None
    y = _logit(p)

    sd: Optional[float] = None
    if lower is not None and upper is not None:
        try:
            lo, hi = float(lower), float(upper)
        except (TypeError, ValueError):
            lo = hi = None  # type: ignore[assignment]
        if lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            sd = (_logit(hi) - _logit(lo)) / (2.0 * _Z95)
    if sd is None or sd <= 0.0:
        sd = _DEFAULT_LOGIT_SD
    w = weight if (isinstance(weight, (int, float)) and weight > 0) else 1.0
    sd = sd / math.sqrt(w)  # reliability inflates variance whether the SD came from an interval or not
    return PurityMeasurement(name=name, y=y, var=sd * sd, point=p)


def integrate_purity_estimates(
    measurements: Sequence[PurityMeasurement],
    *,
    ci: float = 0.95,
) -> Optional[dict]:
    """Random-effects (DerSimonian–Laird) fusion of purity measurements in logit space.

    Returns a dict with the pooled ``overall_estimate`` / ``overall_lower`` / ``overall_upper`` in
    probability space, the between-method heterogeneity (``tau2`` in logit space and ``i2`` ∈ [0, 1]
    — the fraction of dispersion beyond sampling noise, a natural method-agreement/confidence
    signal), and the per-method random-effects ``weights``. ``None`` if no usable measurement.

    A single measurement passes through (point + its own interval). With ≥2, the pooled interval
    widens with disagreement: DerSimonian–Laird estimates τ² from the weighted dispersion Q, adds it
    to every within-method variance, and re-pools — so conflicting methods yield a wide, honest
    interval automatically.
    """
    ms = [m for m in measurements if m is not None]
    if not ms:
        return None

    z = _Z95 if abs(ci - 0.95) < 1e-9 else _z_for_ci(ci)

    if len(ms) == 1:
        m = ms[0]
        se = math.sqrt(m.var)
        return {
            "overall_estimate": round(_expit(m.y), 4),
            "overall_lower": round(_expit(m.y - z * se), 4),
            "overall_upper": round(_expit(m.y + z * se), 4),
            "tau2": 0.0,
            "i2": 0.0,
            "n_methods": 1,
            "weights": {m.name: 1.0},
            "heterogeneity_q": 0.0,
        }

    # Fixed-effect weights and weighted mean.
    w = [1.0 / m.var for m in ms]
    sw = sum(w)
    ybar = sum(wi * m.y for wi, m in zip(w, ms)) / sw

    # Cochran's Q and DerSimonian–Laird τ² (both in logit space).
    q = sum(wi * (m.y - ybar) ** 2 for wi, m in zip(w, ms))
    df = len(ms) - 1
    c = sw - sum(wi * wi for wi in w) / sw
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0

    # Random-effects re-pool: inflate every within-method variance by the shared τ².
    wr = [1.0 / (m.var + tau2) for m in ms]
    swr = sum(wr)
    yhat = sum(wi * m.y for wi, m in zip(wr, ms)) / swr
    se = math.sqrt(1.0 / swr)

    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0

    return {
        "overall_estimate": round(_expit(yhat), 4),
        "overall_lower": round(_expit(yhat - z * se), 4),
        "overall_upper": round(_expit(yhat + z * se), 4),
        "tau2": round(tau2, 6),
        "i2": round(i2, 4),
        "n_methods": len(ms),
        "weights": {m.name: round(wi / swr, 4) for wi, m in zip(wr, ms)},
        "heterogeneity_q": round(q, 4),
    }


def _z_for_ci(ci: float) -> float:
    """Inverse standard-normal CDF at ``(1 + ci) / 2`` (Acklam's rational approximation).

    Kept local so the module needs no scipy; accurate to ~1e-9 over the usable range.
    """
    p = (1.0 + float(ci)) / 2.0
    p = min(1.0 - 1e-12, max(1e-12, p))
    # Acklam's algorithm.
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


# Reliability multipliers on each method's PRECISION when fusing the INTERVAL (not the point). These
# only widen/narrow the fused interval; they never override the point estimate, so they are low-risk
# and not over-fit to any one calibration sample. Grounded in the HCC1395×HPA mixture benchmark + the
# architecture: the decomposition physically models tumor-vs-background; ESTIMATE is monotone but
# biased-high (corroborator, not driver); lineage is a tissue-identity floor, not a purity term.
_METHOD_RELIABILITY = {
    "decomposition": 1.0,
    "signature": 1.0,
    "estimate": 0.5,
    "lineage": 0.3,
}
# A purity read at/above this with no corroborating reliable signal is treated as ESTIMATE-driven
# saturation (the benchmark: ESTIMATE never drops below ~0.66 and climbs to ~0.97).
_SATURATION_FLOOR = 0.85
# A purity read pinned this close to the 1.0 ceiling carries NO information — a tissue biopsy is
# ~never 100% tumor, and a value at the ceiling is the estimator maxing out, not measuring. Such a
# read is ALWAYS replaced by the physical fused consensus regardless of corroboration (a lineage-
# identity + ESTIMATE co-saturation to 100%, as in NUT carcinoma, is the canonical case — the
# decomposition residual fraction stays ~0.55 and exposes it).
_CEILING_PINNED = 0.98
# A method corroborates a high purity only if it independently reads at least this high.
_CORROBORATION_FLOOR = 0.6
# Lineage at/above this is saturated (identity signal maxed out) — not usable as a purity point.
_LINEAGE_SATURATED = 0.95


def best_purity_estimate(
    purity: dict,
    *,
    decomposition: Optional[dict] = None,
) -> Optional[dict]:
    """Produce the best-available purity: keep the empirically-best point, add an honest interval.

    Evidence (HCC1395×HPA in-silico mixtures): no single method is trustworthy across the purity
    range — the signature can be flat/broken on atypical inputs, ESTIMATE is monotone but biased high
    and saturating, lineage is a tissue-identity signal — while the existing combiner's *point* is
    the most accurate estimator available (mixture MAE ≈ 0.10). So this does two evidence-backed
    things and nothing speculative:

    1. **Honest interval + agreement.** Fuse the method estimates with the random-effects model and
       report its interval (which widens automatically when methods disagree) unioned with the
       incoming interval, plus ``method_agreement`` = 1 − I² (1.0 = methods concur).
    2. **Anti-saturation guard.** A read pinned at the 1.0 ceiling is never a real measurement (a
       biopsy is ~never 100% tumor) — it is ALWAYS replaced by the physical fused consensus (chiefly
       the decomposition residual fraction). Below the ceiling but still high (≥ the saturation
       floor), the point is replaced only when NO reliable method (signature / non-saturated lineage
       / non-fragile decomposition) independently corroborates it — the ESTIMATE-driven-saturation
       case the benchmark exposes. Both paths keep the wide, disagreement-aware interval. This is
       what pulls a rare type like NUT carcinoma — whose lineage-identity genes and ESTIMATE both
       saturate to 100% while the residual fraction says ~55% — off a spurious 100%.

    Returns a dict of ``overall_estimate`` / ``overall_lower`` / ``overall_upper`` / ``i2`` /
    ``method_agreement`` / ``n_methods`` / ``point_source`` (``"combiner"`` or ``"desaturated_fusion"``),
    or ``None`` if no method is usable (caller keeps its existing purity).
    """
    comp = (purity or {}).get("components") or {}
    sig = comp.get("signature") or {}
    lin = comp.get("lineage") or {}
    lin_p = lin.get("purity")
    lineage_usable = isinstance(lin_p, (int, float)) and float(lin_p) < _LINEAGE_SATURATED

    # The signature's own stability IS its reliability: a broken/unstable signature (e.g. on a cell
    # line or a rare type whose panel fits poorly) reads a confidently-wrong value that would otherwise
    # force the pooled interval wide open. Discount it by its stability (floored so it never zeroes,
    # neutral 1.0 when stability is absent) so a shaky signature stops dominating the fusion.
    sig_stability = sig.get("stability")
    sig_weight = _METHOD_RELIABILITY["signature"]
    if isinstance(sig_stability, (int, float)):
        sig_weight *= max(float(sig_stability), 0.15)
    ms = [
        measurement_from_interval("signature", sig.get("purity"), sig.get("lower"), sig.get("upper"),
                                  weight=sig_weight),
        measurement_from_interval("estimate", comp.get("estimate_purity"),
                                  weight=_METHOD_RELIABILITY["estimate"]),
    ]
    if lineage_usable:
        ms.append(measurement_from_interval("lineage", lin_p, lin.get("lower"), lin.get("upper"),
                                            weight=_METHOD_RELIABILITY["lineage"]))
    decomp_point = None
    if decomposition:
        decomp_point = decomposition.get("overall_estimate")
        fragile = bool(decomposition.get("fragile"))
        ms.append(measurement_from_interval(
            "decomposition", decomp_point,
            decomposition.get("hypothesis_purity_lo"), decomposition.get("hypothesis_purity_hi"),
            weight=_METHOD_RELIABILITY["decomposition"] * (0.35 if fragile else 1.0),
        ))
    ms = [m for m in ms if m is not None]
    fused = integrate_purity_estimates(ms)
    if fused is None:
        return None

    incoming = purity.get("overall_estimate")

    # Does any RELIABLE method independently corroborate a high purity?
    sig_p = sig.get("purity")
    corroborated_high = any([
        isinstance(sig_p, (int, float)) and float(sig_p) >= _CORROBORATION_FLOOR,
        lineage_usable and float(lin_p) >= _CORROBORATION_FLOOR,
        isinstance(decomp_point, (int, float)) and float(decomp_point) >= _CORROBORATION_FLOOR
        and decomposition and not decomposition.get("fragile"),
    ])
    point_source = "combiner"
    point = float(incoming) if isinstance(incoming, (int, float)) else fused["overall_estimate"]
    if isinstance(incoming, (int, float)):
        inc = float(incoming)
        # A ceiling-pinned read (≈1.0) is never a real measurement → always take the physical fused
        # consensus. Below the ceiling but still high, desaturate only when no reliable method
        # corroborates the high read.
        if inc >= _CEILING_PINNED or (inc >= _SATURATION_FLOOR and not corroborated_high):
            point = fused["overall_estimate"]  # desaturate: fused point blends in the physical signals
            point_source = "desaturated_fusion"

    # Interval: the random-effects fused interval — the honest uncertainty of the fusion itself. It
    # is NOT unioned with the legacy pre-fusion interval, so a poorly-formed legacy bound (e.g. a 0%
    # lower carried by the old combiner) cannot poison it; the fused interval already incorporates
    # every method's own interval and their between-method disagreement (τ²). Only widened to contain
    # the reported point.
    fl, fh = fused["overall_lower"], fused["overall_upper"]
    fl = min(fl, point)
    fh = max(fh, point)
    return {
        "overall_estimate": round(point, 4),
        "overall_lower": round(max(0.0, fl), 4),
        "overall_upper": round(min(1.0, fh), 4),
        "i2": fused["i2"],
        "method_agreement": round(1.0 - fused["i2"], 4),
        "n_methods": fused["n_methods"],
        "point_source": point_source,
        "method_weights": fused["weights"],
    }


def integrate_from_components(purity: dict, *, decomposition: Optional[dict] = None) -> Optional[dict]:
    """Convenience wrapper: pull the standard method estimates off a purity dict and fuse them.

    Reads ``components.signature`` (point + lower/upper), ``components.lineage`` (point + lower/upper),
    ``components.estimate_purity`` (point only), and — when ``decomposition`` is supplied — its point
    plus its template-hypothesis range (``hypothesis_purity_lo``/``hi`` from
    ``analyze.flow.decomposition_purity_stability``) as the decomposition measurement's interval.
    """
    comp = (purity or {}).get("components") or {}
    sig = comp.get("signature") or {}
    lin = comp.get("lineage") or {}
    # Weight each method by the SAME `_METHOD_RELIABILITY` map the live `best_purity_estimate`
    # uses, so the two entry points can never fuse the same components with different weights
    # (ESTIMATE down-weighted as a corroborator, lineage as a tissue-identity floor). This
    # helper omits only best_purity_estimate's point-selection extras (saturation guard,
    # signature-stability discount) — the interval fusion itself is identical.
    ms = [
        measurement_from_interval(
            "signature", sig.get("purity"), sig.get("lower"), sig.get("upper"),
            weight=_METHOD_RELIABILITY["signature"],
        ),
        measurement_from_interval(
            "lineage", lin.get("purity"), lin.get("lower"), lin.get("upper"),
            weight=_METHOD_RELIABILITY["lineage"],
        ),
        measurement_from_interval(
            "estimate", comp.get("estimate_purity"),
            weight=_METHOD_RELIABILITY["estimate"],
        ),
    ]
    if decomposition:
        ms.append(
            measurement_from_interval(
                "decomposition",
                decomposition.get("overall_estimate"),
                decomposition.get("hypothesis_purity_lo"),
                decomposition.get("hypothesis_purity_hi"),
                weight=_METHOD_RELIABILITY["decomposition"],
            )
        )
    return integrate_purity_estimates([m for m in ms if m is not None])
