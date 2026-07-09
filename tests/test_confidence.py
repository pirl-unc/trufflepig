"""Tests for the confidence-tier module (#109)."""

from trufflepig.confidence import (
    ConfidenceTier,
    concise_confidence_reasons,
    compute_purity_confidence,
    compute_target_confidence,
)


def _purity(point, lo, hi):
    return {
        "overall_estimate": point,
        "overall_lower": lo,
        "overall_upper": hi,
    }


def test_tight_ci_high_tier():
    tier = compute_purity_confidence(_purity(0.64, 0.58, 0.70))
    assert tier.tier == "high"
    assert tier.badge == ""


def test_moderate_ci_span_moderate_tier():
    tier = compute_purity_confidence(_purity(0.50, 0.35, 0.65))
    assert tier.tier == "moderate"
    assert "span" in tier.inline_note


def test_wide_ci_low_tier():
    tier = compute_purity_confidence(_purity(0.64, 0.19, 1.00))
    assert tier.tier == "low"
    assert "wide purity CI" in tier.inline_note
    assert tier.badge == "low"


def test_concise_call_confidence_reasons_keep_summary_skimmable():
    tier = ConfidenceTier(
        tier="moderate",
        reasons=[
            "fit quality is ambiguous — preserve alternate cancer hypotheses (long detail)",
            "top candidate SARC beats runner-up ESCA by only 4% on geomean (0.498 vs 0.477) — call is ambiguous",
            "Tissue composition screen favored BLCA but the classifier picked SARC",
        ],
    )
    assert (
        concise_confidence_reasons(tier)
        == "ambiguous fit; near tie with ESCA; tissue composition favors BLCA"
    )


def test_low_purity_regime_bumps_tier():
    # Tight CI but point estimate is in the low-purity regime — the
    # tier should step down because dividing by a small number
    # amplifies any non-tumor residual.
    tier = compute_purity_confidence(_purity(0.10, 0.08, 0.14))
    assert tier.tier in {"moderate", "low"}
    assert any("low-purity regime" in r for r in tier.reasons)


def test_severe_degradation_forces_low():
    tier = compute_purity_confidence(
        _purity(0.60, 0.55, 0.68),
        degradation_severity="severe",
    )
    assert tier.tier == "low"
    assert any("severe RNA degradation" in r for r in tier.reasons)


def test_moderate_degradation_bumps_high_to_moderate():
    tier = compute_purity_confidence(
        _purity(0.60, 0.55, 0.68),
        degradation_severity="moderate",
    )
    assert tier.tier == "moderate"


def test_target_confidence_inherits_purity_tier():
    purity_tier = ConfidenceTier(tier="moderate", reasons=["wide purity CI"])
    target = {"observed_tpm": 150, "tme_dominant": False, "tme_explainable": False}
    tier = compute_target_confidence(target, purity_tier)
    assert tier.tier == "moderate"
    assert "wide purity CI" in tier.reasons


def test_target_confidence_tme_dominant_forces_low():
    purity_tier = ConfidenceTier(tier="high", reasons=[])
    target = {
        "observed_tpm": 1850,
        "tme_dominant": True,
        "attribution": {"T_cell": 1400, "myeloid": 300},
        "attr_tumor_fraction": 0.15,
    }
    tier = compute_target_confidence(target, purity_tier)
    assert tier.tier == "low"
    assert any("TME-dominant" in r for r in tier.reasons)
    assert any("T cell" in r for r in tier.reasons)


def test_target_confidence_partial_tumor_fraction_moderate():
    purity_tier = ConfidenceTier(tier="high", reasons=[])
    target = {
        "observed_tpm": 200,
        "tme_dominant": False,
        "tme_explainable": False,
        "attr_tumor_fraction": 0.40,
    }
    tier = compute_target_confidence(target, purity_tier)
    assert tier.tier == "moderate"
    assert any("40% of signal" in r for r in tier.reasons)


def test_sample_purity_is_low_keys_off_low_purity_regime_reason():
    from trufflepig.confidence import ConfidenceTier, sample_purity_is_low

    low = ConfidenceTier(tier="low", reasons=["low-purity regime (28%)"])
    assert sample_purity_is_low(low) is True
    # Moderate tier without the low-purity-regime reason -> not flagged.
    wide = ConfidenceTier(tier="moderate", reasons=["moderate purity CI span (21 pp)"])
    assert sample_purity_is_low(wide) is False
    assert sample_purity_is_low(None) is False
