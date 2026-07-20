"""Regression test for issue #161 — a zero-width purity CI must be
labeled as "degenerate" rather than "high confidence".

Deterministic inputs (TCGA cohort medians, decomposition templates,
synthetic expected-expression probes) have no per-gene variance, so
the purity estimator returns ``lower == upper == estimate``. Calling
that ``high confidence`` misleads the reader — it is not "very
certain", it is "the estimator had no spread to bound it with".
"""


from trufflepig.main import _ci_confidence_tier
from trufflepig.confidence import ConfidenceTier, compute_purity_confidence


# ── _ci_confidence_tier ─────────────────────────────────────────────


def test_zero_width_ci_returns_degenerate_tier():
    assert _ci_confidence_tier(0.79, 0.79) == "degenerate"


def test_tiny_positive_span_still_high_tier():
    # A genuinely-tight CI (≤ 1pp but not exactly zero) is still "high",
    # not "degenerate" — the estimator gave a real bound.
    assert _ci_confidence_tier(0.68, 0.69) == "high"


def test_typical_ci_spans_map_to_expected_tiers():
    # 10pp — high
    assert _ci_confidence_tier(0.65, 0.75) == "high"
    # 20pp — moderate
    assert _ci_confidence_tier(0.55, 0.75) == "moderate"
    # 40pp — low
    assert _ci_confidence_tier(0.30, 0.70) == "low"


def test_invalid_inputs_return_unknown():
    assert _ci_confidence_tier(None, None) == "unknown"
    assert _ci_confidence_tier("not-a-number", 0.5) == "unknown"


# ── compute_purity_confidence ───────────────────────────────────────


def _purity_dict(estimate, lower, upper):
    return {
        "overall_estimate": estimate,
        "overall_lower": lower,
        "overall_upper": upper,
    }


def test_degenerate_purity_returns_degenerate_tier():
    tier = compute_purity_confidence(_purity_dict(0.79, 0.79, 0.79))
    assert tier.tier == "degenerate"
    # Reason string mentions the key phrase "deterministic input"
    assert any("deterministic" in r.lower() for r in tier.reasons)


def test_degenerate_purity_badge_is_dash():
    tier = ConfidenceTier(tier="degenerate", reasons=["test"])
    assert tier.badge == "\u2014"  # em-dash


def test_non_zero_span_keeps_high_tier():
    tier = compute_purity_confidence(_purity_dict(0.69, 0.65, 0.73))
    assert tier.tier == "high"
    # "high" tier with no reasons returns empty render
    assert tier.render() == ""


def test_degenerate_tier_render_formats_reasons():
    tier = ConfidenceTier(
        tier="degenerate",
        reasons=["deterministic input — purity CI not estimated"],
    )
    rendered = tier.render()
    assert "degenerate" in rendered
    assert "deterministic" in rendered
