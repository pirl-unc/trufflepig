"""Tests for the neuroendocrine lineage-marker recall layer.

Uses synthetic clean-TPM maps with an explicit cross-cohort reference, so they
are fast, deterministic, and exercise the production feature space.
"""

import numpy as np

from trufflepig.lineage_marker_recall import (
    CORE_MARKERS,
    GATING_MARKERS,
    NE_NO_REFERENCE,
    neuroendocrine_recall,
    recall_candidates,
)

SCALE = 100.0
COHORT = {
    marker: np.array([0.0, 0.2, 0.5, 1.0, 2.0, 3.0])
    for marker in GATING_MARKERS
}


def _tpm(**ratios):
    return {sym: r * SCALE for sym, r in ratios.items()}


def test_fires_on_well_differentiated_ne():
    # NET/PCPG-like: huge granins.
    p = neuroendocrine_recall(
        _tpm(CHGA=12.0, CHGB=12.0, SYP=1.0, INSM1=0.3),
        cohort_reference=COHORT,
    )
    assert p is not None
    assert p.broad == "neuroendocrine"
    assert set(p.entities) == set(NE_NO_REFERENCE)


def test_fires_on_high_grade_sclc_modest_markers():
    # SCLC-like on the ribosomal-free HK scale: CHGA ~1.5x, ASCL1 ~0.7x — modest
    # for NE but well clear of the 0.3 bar, with a core marker present.
    p = neuroendocrine_recall(
        _tpm(CHGA=1.52, ASCL1=0.68, CHGB=0.36, INSM1=0.37),
        cohort_reference=COHORT,
    )
    assert p is not None


def test_fires_on_single_dominant_core_granin():
    # NET_MIDGUT-like: one very high core granin (CHGB ~5.5x), the rest modest.
    # The 2-marker minimum would miss it; the strong-single-core path catches it.
    p = neuroendocrine_recall(
        _tpm(CHGB=5.5, INSM1=0.0, SYP=0.0, SCG2=0.0),
        cohort_reference=COHORT,
    )
    assert p is not None


def test_single_modest_core_still_needs_two_markers():
    # A core marker just over the bar but below strong_core, alone, must not fire.
    p = neuroendocrine_recall(
        {"CHGB": 3.0, "INSM1": 0.0, "SYP": 0.0, "SCG2": 0.0},
        cohort_reference=COHORT,
    )
    assert p is None


def test_silent_on_non_ne():
    # COAD/BRCA-like: NE program essentially off.
    assert (
        neuroendocrine_recall(
            _tpm(CHGA=0.0, SYP=0.0, SCG2=0.01),
            cohort_reference=COHORT,
        )
        is None
    )


def test_ascl1_alone_does_not_fire_without_core_marker():
    # NUT carcinoma / NE-LUAD: ASCL1 elevated, but no core granin/INSM1.
    # ASCL1 + an incidental SCG2 clear the 2-marker count, so this pins the
    # core-marker obligate (the pfo019 false-positive that motivated it).
    p = neuroendocrine_recall(
        _tpm(ASCL1=0.62, SCG2=0.35, CHGA=0.0, CHGB=0.0, INSM1=0.0),
        cohort_reference=COHORT,
    )
    assert p is None


def test_core_marker_alone_insufficient_needs_two():
    # A single core marker over threshold shouldn't fire on its own.
    p = neuroendocrine_recall(
        {"CHGA": 3.0, "CHGB": 0.0, "INSM1": 0.0, "SYP": 0.0},
        cohort_reference=COHORT,
    )
    assert p is None


def test_empty_profile_is_safe():
    assert neuroendocrine_recall({}, cohort_reference=COHORT) is None


def test_core_markers_are_granins_and_insm1():
    assert set(CORE_MARKERS) == {"CHGA", "CHGB", "INSM1"}


def test_recall_candidates_collects_fired_proposals():
    fired = recall_candidates(
        _tpm(CHGA=12.0, CHGB=12.0, SYP=1.0),
        cohort_reference=COHORT,
    )
    assert len(fired) == 1
    assert fired[0].broad == "neuroendocrine"
    assert recall_candidates(
        _tpm(CHGA=0.0), cohort_reference=COHORT
    ) == []
