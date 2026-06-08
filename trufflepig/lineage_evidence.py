"""Tumour-intrinsic lineage evidence: exclusion gates over broad lineage.

The cross-cohort signature screen is fooled by admixture: a carcinoma with heavy
stroma scores a mesenchymal cohort (SARC) high, because the *stromal* mesenchymal
program matches the (also-stromal) SARC reference. We tried to fix this with
deconvolution residuals and it didn't work — the decomposition has no generic
stromal compartment, so the stromal signal survives into the tumour residual
(see docs/cancer-type-residual-matching-findings.md).

What does work is the validated lineage biology: a carcinoma robustly expresses
the **epithelial differentiation program** (EPCAM / cytokeratins / CDH1) in its
*tumour cells*; a true sarcoma does not. So epithelial-program presence is a clean
exclusion of the mesenchymal (and hematolymphoid) branches — the empirically
strong anti-correlations from the lineage sweep (epithelial ⟂ mesenchymal ≈ -0.36,
epithelial ⟂ heme). These markers are tumour-intrinsic and HK-ratio normalised, so
the gate is scale-invariant and not contaminated by the stromal program it's
meant to discount.

Measured separation on local clinical samples: epithelial HK-ratio runs 0.22–5.1×
in carcinomas vs 0.01–0.07× in real sarcomas — a clean ordering. The demotion is
**confidence-proportional**: the stronger the epithelial program, the harder the
non-epithelial branches are demoted (a strongly-keratin+ bladder cancer crushes
SARC; a borderline-epithelial case only nudges it).

This is purely *additive* over the signature screen — it only down-weights
non-epithelial candidates when epithelial differentiation is present; it never
invents or up-ranks a candidate. Real sarcomas (epithelial-absent) are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Tumour-intrinsic epithelial differentiation program.
EPITHELIAL_MARKERS = ("EPCAM", "KRT8", "KRT18", "KRT19", "CDH1")

# Broad lineages a robust epithelial program excludes (carcinoma is not these).
EPITHELIAL_EXCLUDES = ("mesenchymal", "hematolymphoid")

# The gate fires on the epithelial signal's multi-view CONFIDENCE
# (:mod:`trufflepig.signal_views`), which integrates five normalizations — chiefly
# log1p(clean TPM), the tightest within-class separator (3.5× tighter than the old
# single HK-ratio gate, sep 1.46 vs 0.45). On the rep + local sweep, carcinoma
# confidence ≈ 0.62 (median) vs sarcoma ≈ 0.07; 0.45 sits in the gap (the only
# overlap is epithelioid sarcomas, which are genuinely keratin+ — no expression-
# only fix). Demotion is proportional to confidence.
DEFAULT_EPITHELIAL_CONFIDENCE = 0.45
_DEMOTE_SLOPE = 0.9
_DEMOTE_FLOOR = 0.35


@dataclass
class LineageEvidence:
    """Per-broad-lineage multiplicative factors plus a readable note."""

    factors: dict[str, float]
    epithelial_hk_ratio: float
    notes: list[str] = field(default_factory=list)
    signal: object | None = None  # the SignalViews fingerprint (5 normalizations)


def _median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def epithelial_hk_ratio(sample_tpm_by_symbol, sample_hk_median) -> float:
    """Median HK-ratio of the epithelial program (scale-invariant presence)."""
    if not sample_hk_median or sample_hk_median <= 0:
        return 0.0
    return _median(
        [sample_tpm_by_symbol.get(g, 0.0) / sample_hk_median for g in EPITHELIAL_MARKERS]
    )


def _fingerprint(signal) -> str:
    """One-line 5-view fingerprint for the trace."""
    v = signal.views
    return (
        f"epithelial views — HK {v['hk']:.2f}× · within-sample pct {v['within_pct']:.2f} · "
        f"log1p(TPM) {v['log1p']:.2f} · cohort-pct {v['cohort_pct']:.2f} · "
        f"cohort-z {v['cohort_z']:+.2f} → {signal.call} (confidence {signal.confidence:.2f}, "
        f"concordance {signal.concordance:.2f})"
    )


def lineage_exclusion_evidence(
    sample_tpm_by_symbol,
    sample_hk_median,
    *,
    confidence_threshold: float = DEFAULT_EPITHELIAL_CONFIDENCE,
    cohort_reference=None,
) -> LineageEvidence:
    """Compute broad-lineage demotion factors from the epithelial signal's
    multi-view confidence.

    Returns all-1.0 factors (a no-op) when the epithelial program isn't
    confidently present, so real sarcomas / lymphomas pass through untouched.
    """
    from .signal_views import signal_report

    sig = signal_report(
        "epithelial", EPITHELIAL_MARKERS, sample_tpm_by_symbol,
        sample_hk_median=sample_hk_median, cohort_reference=cohort_reference,
    )
    factors: dict[str, float] = {}
    notes: list[str] = []
    if sig.confidence >= confidence_threshold:
        demote = max(_DEMOTE_FLOOR, 1.0 - _DEMOTE_SLOPE * sig.confidence)
        for broad in EPITHELIAL_EXCLUDES:
            factors[broad] = demote
        notes.append(
            f"Epithelial differentiation present — carcinoma; "
            f"{', '.join(EPITHELIAL_EXCLUDES)} candidates demoted ×{demote:.2f} "
            f"(a true sarcoma/lymphoma would lack this tumour-intrinsic program). "
            f"[{_fingerprint(sig)}]"
        )
        for flag in sig.flags:
            notes.append(f"epithelial signal: {flag}")
    return LineageEvidence(
        factors=factors,
        epithelial_hk_ratio=float(sig.views.get("hk", 0.0)),
        notes=notes,
        signal=sig,
    )
