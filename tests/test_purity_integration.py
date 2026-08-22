"""Tests for the random-effects tumor-purity fusion (trufflepig.purity_integration).

Pins the key statistical properties: methods that agree pool to a tight interval, methods that
disagree pool to a WIDE interval (heterogeneity inflates it automatically, no threshold), a single
method passes through, and the pooled point/interval always lie in (0, 1).
"""
from __future__ import annotations

import pytest

from trufflepig.purity_integration import (
    best_purity_estimate,
    integrate_from_components,
    integrate_purity_estimates,
    measurement_from_interval,
)


def _fuse(triples):
    ms = [measurement_from_interval(*t[:1], *t[1:]) for t in triples]
    return integrate_purity_estimates([m for m in ms if m is not None])


def test_agreeing_methods_pool_to_a_tight_interval():
    out = _fuse([
        ("signature", 0.40, 0.33, 0.47),
        ("lineage", 0.42, 0.35, 0.50),
        ("decomposition", 0.40, 0.30, 0.50),
    ])
    assert out["overall_estimate"] == pytest.approx(0.40, abs=0.03)
    assert out["i2"] < 0.2  # little heterogeneity
    # tight: the pooled interval is no wider than the widest input interval
    assert (out["overall_upper"] - out["overall_lower"]) <= 0.22


def test_disagreeing_methods_inflate_the_interval():
    tight = _fuse([
        ("signature", 0.40, 0.33, 0.47),
        ("lineage", 0.42, 0.35, 0.50),
    ])
    conflict = _fuse([
        ("signature", 0.85, 0.75, 0.92),
        ("lineage", 0.60, 0.45, 0.75),
        ("decomposition", 0.10, 0.03, 0.54),
    ])
    conflict_width = conflict["overall_upper"] - conflict["overall_lower"]
    tight_width = tight["overall_upper"] - tight["overall_lower"]
    assert conflict["i2"] > 0.5  # high heterogeneity
    assert conflict_width > tight_width  # disagreement → wider interval, automatically
    assert conflict_width > 0.4


def test_single_method_passes_through():
    out = _fuse([("signature", 0.43, 0.33, 0.61)])
    assert out["overall_estimate"] == pytest.approx(0.43, abs=0.02)
    assert out["i2"] == 0.0
    assert out["n_methods"] == 1
    assert out["weights"] == {"signature": 1.0}


def test_pooled_estimate_and_interval_are_bounded_in_unit_interval():
    out = _fuse([
        ("signature", 0.97, 0.90, 0.99),
        ("decomposition", 0.02, 0.01, 0.10),
    ])
    for key in ("overall_estimate", "overall_lower", "overall_upper"):
        assert 0.0 < out[key] < 1.0
    assert out["overall_lower"] <= out["overall_estimate"] <= out["overall_upper"]


def test_saturated_and_missing_points_are_dropped():
    # A saturated 1.0 / 0.0 point carries no logit information → dropped; None → dropped.
    assert measurement_from_interval("sig", 1.0) is None
    assert measurement_from_interval("sig", 0.0) is None
    assert measurement_from_interval("sig", None) is None
    # A dict with only a saturated signature + one real method → single-method passthrough.
    out = integrate_purity_estimates([
        measurement_from_interval("signature", 1.0),
        measurement_from_interval("lineage", 0.30, 0.20, 0.40),
    ])
    assert out["n_methods"] == 1


def test_no_measurements_returns_none():
    assert integrate_purity_estimates([]) is None
    assert integrate_from_components({}) is None


def test_integrate_from_components_reads_the_standard_purity_shape():
    purity = {
        "components": {
            "signature": {"purity": 0.55, "lower": 0.48, "upper": 0.62},
            "lineage": {"purity": 0.58, "lower": 0.50, "upper": 0.66},
            "estimate_purity": 0.52,
        }
    }
    decomposition = {
        "overall_estimate": 0.50,
        "hypothesis_purity_lo": 0.40,
        "hypothesis_purity_hi": 0.60,
    }
    out = integrate_from_components(purity, decomposition=decomposition)
    assert out["n_methods"] == 4
    assert 0.45 < out["overall_estimate"] < 0.60
    assert set(out["weights"]) == {"signature", "lineage", "estimate", "decomposition"}


def test_integrate_from_components_applies_method_reliability_weights():
    # Signature and lineage with IDENTICAL point + interval must NOT get equal fused weight:
    # lineage is down-weighted (tissue-identity floor) via the SAME _METHOD_RELIABILITY map the
    # live best_purity_estimate uses, so the two fusion entry points can't diverge on weighting.
    purity = {
        "components": {
            "signature": {"purity": 0.55, "lower": 0.48, "upper": 0.62},
            "lineage": {"purity": 0.55, "lower": 0.48, "upper": 0.62},
        }
    }
    out = integrate_from_components(purity)
    assert out["weights"]["signature"] > out["weights"]["lineage"]


# --- best_purity_estimate: keep the empirically-best point, add an honest interval + saturation guard ---


def test_best_purity_does_not_replace_a_high_but_nonendpoint_reading():
    """A merely high point is not saturation and cannot be replaced by context channels."""
    purity = {
        "overall_estimate": 0.98,
        "overall_lower": 0.90,
        "overall_upper": 1.0,
        "components": {
            "signature": {"purity": 0.26, "lower": 0.12, "upper": 0.45},
            "lineage": {"purity": None},
            "estimate_purity": 0.97,
        },
    }
    out = best_purity_estimate(purity, decomposition=None)
    assert out["point_source"] == "combiner"
    assert out["overall_estimate"] == 0.98
    assert out["overall_lower"] == 0.90
    assert out["overall_upper"] == 1.0


def test_best_purity_preserves_a_corroborated_high_reading():
    """When the signature independently reads high too, the high purity is real — do NOT desaturate."""
    purity = {
        "overall_estimate": 0.90,
        "overall_lower": 0.82,
        "overall_upper": 0.97,
        "components": {
            "signature": {"purity": 0.88, "lower": 0.80, "upper": 0.94},
            "lineage": {"purity": 0.85, "lower": 0.75, "upper": 0.92},
            "estimate_purity": 0.93,
        },
    }
    out = best_purity_estimate(purity, decomposition=None)
    assert out["point_source"] == "combiner"
    assert out["overall_estimate"] >= 0.85


def test_best_purity_decomposition_rescues_a_pure_sample_with_broken_signature():
    """A pure cell line: signature is broken-low and there's no lineage, but the decomposition finds
    ~no background (high purity). Decomposition is the discriminator that corroborates 'pure', so the
    guard must NOT desaturate."""
    purity = {
        "overall_estimate": 0.96,
        "overall_lower": 0.88,
        "overall_upper": 1.0,
        "components": {
            "signature": {"purity": 0.10, "lower": 0.03, "upper": 0.28},
            "lineage": {"purity": None},
            "estimate_purity": 0.97,
        },
    }
    decomp = {
        "overall_estimate": 0.90,
        "hypothesis_purity_lo": 0.80,
        "hypothesis_purity_hi": 0.97,
        "fragile": False,
    }
    out = best_purity_estimate(purity, decomposition=decomp)
    assert out["point_source"] == "combiner"
    assert out["overall_estimate"] >= 0.85


def test_best_purity_preserves_the_point_and_gives_a_tight_interval_when_methods_agree():
    """A middling sample where methods roughly agree: point is preserved and the fused interval is
    tight (high agreement → low heterogeneity → narrow band), reflecting the fusion's own uncertainty
    rather than any legacy pre-fusion interval."""
    purity = {
        "overall_estimate": 0.42,
        "overall_lower": 0.34,
        "overall_upper": 0.52,
        "components": {
            "signature": {"purity": 0.40, "lower": 0.33, "upper": 0.47},
            "lineage": {"purity": 0.45, "lower": 0.36, "upper": 0.54},
            "estimate_purity": 0.44,
        },
    }
    out = best_purity_estimate(purity, decomposition=None)
    assert out["point_source"] == "combiner"
    assert out["overall_estimate"] == pytest.approx(0.42, abs=1e-6)
    assert out["method_agreement"] > 0.8  # methods concur
    assert out["overall_lower"] <= 0.42 <= out["overall_upper"]  # interval contains the point
    assert (out["overall_upper"] - out["overall_lower"]) < 0.35  # agreeing → not blown wide open


def test_best_purity_preserves_primary_interval_when_component_methods_disagree():
    """PFO004-shaped regression: internal context disagreement is audit metadata,
    not license to turn a useful 6–37% primary interval into 2–98%."""
    purity = {
        "overall_estimate": 0.12,
        "overall_lower": 0.06,
        "overall_upper": 0.37,
        "components": {
            "signature": {
                "purity": 0.08,
                "lower": 0.03,
                "upper": 0.18,
                "stability": 0.25,
            },
            "lineage": {"purity": 0.24, "lower": 0.10, "upper": 0.75},
            "estimate_purity": 0.92,
        },
    }
    decomposition = {
        "overall_estimate": 0.12,
        "hypothesis_purity_lo": 0.12,
        "hypothesis_purity_hi": 0.12,
        "fragile": True,
    }

    out = best_purity_estimate(purity, decomposition=decomposition)

    assert out["overall_estimate"] == 0.12
    assert out["overall_lower"] == 0.06
    assert out["overall_upper"] == 0.37
    assert out["interval_source"] == "integrated_estimate"
    assert out["method_agreement"] < 0.5


def test_best_purity_ceiling_pinned_drops_to_the_decomposition_residual():
    """The real NUT-carcinoma pattern (nutm1-0028): signature absent, lineage-identity genes and
    ESTIMATE both saturate to 100%, but the decomposition residual fraction — the physical tumor-vs-
    background reading — says ~55%. A ceiling-pinned 100% is never real, so it must drop to the
    residual-driven consensus even though the fused decomposition point (0.57) clears the 0.6-ish
    corroboration bar's neighborhood."""
    purity = {
        "overall_estimate": 1.0,
        "overall_lower": 0.97,
        "overall_upper": 1.0,
        "components": {
            "signature": {"purity": None},
            "lineage": {"purity": 1.0, "lower": 0.97, "upper": 1.0},  # identity genes → saturated
            "estimate_purity": 1.0,
        },
    }
    decomp = {"overall_estimate": 0.57, "fragile": False}  # residual_fraction
    out = best_purity_estimate(purity, decomposition=decomp)
    assert out["point_source"] == "desaturated_fusion"
    assert out["interval_source"] == "background_residual"
    assert out["overall_estimate"] == pytest.approx(0.57, abs=0.02)  # tracks the physical signal
    assert out["overall_upper"] < 0.98  # no longer pinned at the ceiling
    assert out["ceiling_excluded_methods"] == ["lineage", "estimate"]


def test_best_purity_ceiling_pinned_desaturates_even_when_decomposition_reads_high():
    """A ceiling-pinned read is replaced by the physical consensus regardless of corroboration: even
    if decomposition reads high (0.9), 100% is implausible for a biopsy, so it drops to ~0.9, not
    stays at 1.0."""
    purity = {
        "overall_estimate": 1.0,
        "overall_lower": 1.0,
        "overall_upper": 1.0,
        "components": {
            "signature": {"purity": None},
            "lineage": {"purity": 1.0},
            "estimate_purity": 1.0,
        },
    }
    out = best_purity_estimate(purity, decomposition={"overall_estimate": 0.9, "fragile": False})
    assert out["point_source"] == "desaturated_fusion"
    assert out["overall_estimate"] == pytest.approx(0.9, abs=0.02)  # high, but not a pinned 100%


def test_fragility_flag_does_not_turn_a_single_residual_into_a_vacuous_range():
    """A TME warning remains a caveat; it is not a calibrated variance and
    must not multiply a residual-only interval into the 9–91% failure mode."""
    purity = {
        "overall_estimate": 1.0,
        "overall_lower": 1.0,
        "overall_upper": 1.0,
        "components": {
            "signature": {"purity": None},
            "lineage": {"purity": 1.0},
            "estimate_purity": 1.0,
        },
    }

    out = best_purity_estimate(
        purity,
        decomposition={"overall_estimate": 0.51, "fragile": True},
    )

    assert out["overall_estimate"] == pytest.approx(0.51, abs=0.01)
    assert out["overall_upper"] - out["overall_lower"] < 0.80


def test_best_purity_corroborated_decomposition_below_ceiling_stays():
    """An 80%-purity sample (nutm1-0026): signature broken-low (0.15) but decomposition residual and
    ESTIMATE both independently say ~0.80. Below the ceiling and corroborated → point preserved."""
    purity = {
        "overall_estimate": 0.80,
        "overall_lower": 0.06,
        "overall_upper": 0.80,
        "components": {
            "signature": {"purity": 0.15, "lower": 0.06, "upper": 0.34, "stability": 0.22},
            "lineage": {"purity": None},
            "estimate_purity": 0.80,
        },
    }
    out = best_purity_estimate(purity, decomposition={"overall_estimate": 0.80, "fragile": False})
    assert out["point_source"] == "combiner"
    assert out["overall_estimate"] == pytest.approx(0.80, abs=1e-6)


def test_best_purity_saturated_lineage_is_audit_only_below_endpoint():
    """A saturated lineage is not a purity read, but cannot manufacture a new
    point or interval when the integrated estimate itself is below the endpoint."""
    purity = {
        "overall_estimate": 0.97,
        "overall_lower": 0.90,
        "overall_upper": 1.0,
        "components": {
            "signature": {"purity": 0.30, "lower": 0.18, "upper": 0.46},
            "lineage": {"purity": 0.99},  # saturated → dropped, no corroboration
            "estimate_purity": 0.96,
        },
    }
    out = best_purity_estimate(purity, decomposition=None)
    assert out["point_source"] == "combiner"
    assert out["overall_estimate"] == 0.97
    assert out["overall_lower"] == 0.90
    assert out["overall_upper"] == 1.0
