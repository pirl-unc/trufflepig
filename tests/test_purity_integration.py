"""Tests for the random-effects tumor-purity fusion (trufflepig.purity_integration).

Pins the key statistical properties: methods that agree pool to a tight interval, methods that
disagree pool to a WIDE interval (heterogeneity inflates it automatically, no threshold), a single
method passes through, and the pooled point/interval always lie in (0, 1).
"""
from __future__ import annotations

import pytest

from trufflepig.purity_integration import (
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
