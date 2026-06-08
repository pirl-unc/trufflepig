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

# Fire above this epithelial HK-ratio. 0.15 sits cleanly between the highest real
# sarcoma (~0.07×HK) and the lowest carcinoma (~0.22×HK) observed locally.
DEFAULT_EPITHELIAL_THRESHOLD = 0.15
# Demotion curve: factor = 1 - SLOPE * min(epi, CAP) / CAP, floored at FLOOR.
_DEMOTE_SLOPE = 0.6
_DEMOTE_CAP = 3.0
_DEMOTE_FLOOR = 0.35


@dataclass
class LineageEvidence:
    """Per-broad-lineage multiplicative factors plus a readable note."""

    factors: dict[str, float]
    epithelial_hk_ratio: float
    notes: list[str] = field(default_factory=list)


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


def lineage_exclusion_evidence(
    sample_tpm_by_symbol,
    sample_hk_median,
    *,
    threshold: float = DEFAULT_EPITHELIAL_THRESHOLD,
) -> LineageEvidence:
    """Compute broad-lineage demotion factors from tumour-intrinsic markers.

    Returns all-1.0 factors (a no-op) when no epithelial program is present, so
    real sarcomas / lymphomas pass through untouched.
    """
    epi = epithelial_hk_ratio(sample_tpm_by_symbol, sample_hk_median)
    factors: dict[str, float] = {}
    notes: list[str] = []
    if epi >= threshold:
        demote = max(
            _DEMOTE_FLOOR, 1.0 - _DEMOTE_SLOPE * min(epi, _DEMOTE_CAP) / _DEMOTE_CAP
        )
        for broad in EPITHELIAL_EXCLUDES:
            factors[broad] = demote
        notes.append(
            f"Epithelial differentiation present (EPCAM/keratins {epi:.2f}×HK) — "
            f"carcinoma; {', '.join(EPITHELIAL_EXCLUDES)} candidates demoted ×{demote:.2f} "
            f"(a true sarcoma/lymphoma would lack this tumour-intrinsic program)."
        )
    return LineageEvidence(factors=factors, epithelial_hk_ratio=float(epi), notes=notes)
