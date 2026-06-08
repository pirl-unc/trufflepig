"""Tests for the neuroendocrine lineage-marker recall layer.

Uses synthetic {symbol: TPM} maps with an explicit housekeeping median, so they
are fast and don't load reference matrices. The HK-ratio values mirror the
empirically-measured NE-rep vs non-NE-rep separation (see
docs/cancer-type-residual-matching-findings.md).
"""

import pytest

from trufflepig.cancer_type_ontology import classify_cancer_type_ontology
from trufflepig.lineage_marker_recall import (
    CORE_MARKERS,
    NE_NO_REFERENCE,
    neuroendocrine_recall,
    recall_candidates,
)

HK = 100.0  # housekeeping median; markers given as multiples below


def _tpm(**ratios):
    return {sym: r * HK for sym, r in ratios.items()}


def test_fires_on_well_differentiated_ne():
    # NET/PCPG-like: huge granins.
    p = neuroendocrine_recall(_tpm(CHGA=12.0, CHGB=12.0, SYP=1.0, INSM1=0.3), HK)
    assert p is not None
    assert p.broad == "neuroendocrine"
    assert set(p.entities) == set(NE_NO_REFERENCE)


def test_fires_on_high_grade_sclc_modest_markers():
    # SCLC-like: CHGA ~0.5x, ASCL1 ~0.24x — modest but a core marker present.
    p = neuroendocrine_recall(_tpm(CHGA=0.54, ASCL1=0.24, CHGB=0.13, INSM1=0.13), HK)
    assert p is not None


def test_silent_on_non_ne():
    # COAD/BRCA-like: NE program essentially off.
    assert neuroendocrine_recall(_tpm(CHGA=0.0, SYP=0.0, SCG2=0.01), HK) is None


def test_ascl1_alone_does_not_fire_without_core_marker():
    # NUT carcinoma / NE-LUAD: ASCL1 elevated, but no core granin/INSM1.
    # ASCL1 + an incidental SCG2 would clear the 2-marker count, so this pins the
    # core-marker obligate (the pfo019 false-positive that motivated it).
    p = neuroendocrine_recall(_tpm(ASCL1=0.33, SCG2=0.20, CHGA=0.0, CHGB=0.0, INSM1=0.0), HK)
    assert p is None


def test_core_marker_alone_insufficient_needs_two():
    # A single core marker just over threshold shouldn't fire on its own.
    p = neuroendocrine_recall(_tpm(CHGA=0.2, CHGB=0.0, INSM1=0.0, SYP=0.0), HK)
    assert p is None


def test_zero_hk_median_is_safe():
    assert neuroendocrine_recall(_tpm(CHGA=10.0), 0.0) is None


def test_core_markers_are_granins_and_insm1():
    assert set(CORE_MARKERS) == {"CHGA", "CHGB", "INSM1"}


def test_recall_candidates_collects_fired_proposals():
    fired = recall_candidates(_tpm(CHGA=12.0, CHGB=12.0, SYP=1.0), HK)
    assert len(fired) == 1
    assert fired[0].broad == "neuroendocrine"
    assert recall_candidates(_tpm(CHGA=0.0), HK) == []


def test_recall_injection_routes_to_neuroendocrine_branch():
    # screen scattered an NE sample across LUAD/DLBC; recall rescues it.
    rows = [
        {"code": "LUAD", "signature_score": 0.45, "support_geomean": 0.45},
        {"code": "DLBC", "signature_score": 0.40, "support_geomean": 0.40},
    ]
    props = recall_candidates(_tpm(CHGA=12.0, CHGB=12.0, SYP=1.0), HK)
    r = classify_cancer_type_ontology(ranked_rows=rows, recall_proposals=props)
    assert all(c in NE_NO_REFERENCE for c in r.candidates)
    assert r.recall_notes
    assert r.trace[0].startswith("[recall]")


def test_recall_is_additive_does_not_lower_screen_candidates():
    rows = [{"code": "LUAD", "signature_score": 0.45, "support_geomean": 0.45}]
    r = classify_cancer_type_ontology(ranked_rows=rows, recall_proposals=[], use_recall=True)
    assert r.scores["LUAD"] == 0.45  # untouched when nothing fires
