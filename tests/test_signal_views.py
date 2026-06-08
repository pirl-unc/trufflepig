"""Tests for the five-view signal normalization + reasoning.

Synthetic {symbol: TPM} maps with an explicit cohort reference, so they are fast
and don't load reference matrices.
"""

import numpy as np

from trufflepig.signal_views import (
    VIEW_NAMES,
    compute_views,
    reason_over_views,
    signal_report,
)

HK = 100.0
GENES = ["A", "B", "C"]
# cohort reference: each gene's distribution across 10 pretend cohorts (low)
COHORT = {g: np.array([1.0, 2.0, 1.5, 0.5, 3.0, 1.0, 2.0, 0.0, 1.0, 2.0]) for g in GENES}


def _views(tpm, within=None):
    return compute_views(
        "sig", GENES, tpm, sample_hk_median=HK, within_sample_pct=within, cohort_reference=COHORT
    )


def test_all_five_views_are_computed():
    v = _views({"A": 300.0, "B": 300.0, "C": 300.0}, within={"A": 0.99, "B": 0.99, "C": 0.99})
    assert set(v) == set(VIEW_NAMES)
    assert all(np.isfinite(x) for x in v.values())


def test_high_signal_concordant_present():
    # very high across the board vs a low cohort background
    tpm = {"A": 500.0, "B": 500.0, "C": 500.0}
    r = signal_report("sig", GENES, tpm, sample_hk_median=HK,
                      within_sample_pct={g: 0.97 for g in GENES}, cohort_reference=COHORT)
    assert r.call == "present"
    assert r.confidence >= 0.6
    assert r.views["cohort_pct"] == 1.0  # above every reference cohort
    assert r.views["cohort_z"] > 0


def test_absent_signal_concordant_absent():
    tpm = {"A": 0.0, "B": 0.0, "C": 0.0}
    r = signal_report("sig", GENES, tpm, sample_hk_median=HK,
                      within_sample_pct={g: 0.1 for g in GENES}, cohort_reference=COHORT)
    assert r.call == "absent"
    assert r.confidence <= 0.35


def test_admixture_flag_when_above_background_but_not_dominant():
    # high vs cohort background (cohort_z high) but NOT dominant in-sample
    # (within_pct low) -> the low-purity / admixture diagnostic.
    tpm = {"A": 50.0, "B": 50.0, "C": 50.0}  # well above the ~1.5 cohort mean
    r = signal_report("sig", GENES, tpm, sample_hk_median=HK,
                      within_sample_pct={g: 0.40 for g in GENES}, cohort_reference=COHORT)
    assert any("admixture" in f or "low purity" in f for f in r.flags)


def test_concordance_low_when_views_disagree():
    # within-sample dominant but cohort background also high -> mixed signals
    big_ref = {g: np.array([400.0] * 10) for g in GENES}
    r = signal_report("sig", GENES, {"A": 450.0, "B": 450.0, "C": 450.0}, sample_hk_median=HK,
                      within_sample_pct={g: 0.99 for g in GENES}, cohort_reference=big_ref)
    # cohort_pct/z near the middle, within_pct high -> disagreement
    assert r.concordance < 1.0


def test_reason_over_views_handles_missing_views():
    r = reason_over_views("sig", {"hk": 2.0, "within_pct": 0.9, "log1p": 6.0,
                                  "cohort_pct": float("nan"), "cohort_z": float("nan")})
    assert r.call in ("present", "ambiguous", "absent")
    assert 0.0 <= r.confidence <= 1.0
