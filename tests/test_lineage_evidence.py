"""Tests for the tumour-intrinsic epithelial-exclusion lineage gate (multi-view)."""

import numpy as np

from trufflepig.cancer_type_ontology import classify_cancer_type_ontology
from trufflepig.lineage_evidence import (
    DEFAULT_EPITHELIAL_CONFIDENCE,
    EPITHELIAL_EXCLUDES,
    EPITHELIAL_MARKERS,
    lineage_exclusion_evidence,
)

HK = 100.0
# synthetic cross-cohort background for the epithelial markers (low) — keeps the
# tests fast (no reference-matrix load) and deterministic.
COHORT = {g: np.array([1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 3.0, 1.0, 0.0, 2.0]) for g in EPITHELIAL_MARKERS}


def _sample(marker_ratio):
    """A sample where the epithelial markers sit at ``marker_ratio`` x HK among a
    low background — so within-sample percentile is high when markers are high."""
    d = {f"FILL{i}": 1.0 for i in range(30)}
    for g in EPITHELIAL_MARKERS:
        d[g] = marker_ratio * HK
    return d


def _ev(marker_ratio, **kw):
    return lineage_exclusion_evidence(_sample(marker_ratio), HK, cohort_reference=COHORT, **kw)


def test_silent_when_epithelial_absent():
    ev = _ev(0.0)  # real sarcoma: epithelial program off
    assert ev.factors == {}
    assert ev.notes == []
    assert ev.signal.call == "absent"


def test_demotes_non_epithelial_when_epithelial_present():
    ev = _ev(4.0)  # strong epithelial program
    assert set(ev.factors) == set(EPITHELIAL_EXCLUDES)
    assert all(f < 1.0 for f in ev.factors.values())
    assert ev.notes
    assert ev.signal.call == "present"


def test_demotion_is_confidence_proportional():
    weak = _ev(0.8)
    strong = _ev(5.0)
    assert strong.signal.confidence > weak.signal.confidence
    if weak.factors and strong.factors:
        assert strong.factors["mesenchymal"] <= weak.factors["mesenchymal"]


def test_fingerprint_reports_all_five_views():
    ev = _ev(4.0)
    note = ev.notes[0]
    for token in ("HK", "within-sample pct", "log1p(TPM)", "cohort-pct", "cohort-z"):
        assert token in note, note


def test_threshold_is_a_confidence_in_unit_interval():
    assert 0.0 < DEFAULT_EPITHELIAL_CONFIDENCE < 1.0


def test_stromal_sarcoma_confound_resolved_in_walk():
    # screen confused by stroma: READ top, SARC a close second.
    rows = [
        {"code": "READ", "signature_score": 0.67, "support_geomean": 0.67},
        {"code": "SARC", "signature_score": 0.54, "support_geomean": 0.54},
        {"code": "COAD", "signature_score": 0.52, "support_geomean": 0.52},
    ]
    ev = _ev(4.0)
    r = classify_cancer_type_ontology(ranked_rows=rows, lineage_evidence=ev, use_recall=False)
    assert "SARC" not in r.candidates  # mesenchymal demoted out
    assert r.candidates[0] == "READ"
    assert any(t.startswith("[exclude]") for t in r.trace)


def test_real_sarcoma_untouched_in_walk():
    rows = [
        {"code": "SARC", "signature_score": 0.55, "support_geomean": 0.55},
        {"code": "BRCA", "signature_score": 0.40, "support_geomean": 0.40},
    ]
    ev = _ev(0.0)  # epithelial absent
    r = classify_cancer_type_ontology(ranked_rows=rows, lineage_evidence=ev, use_recall=False)
    assert r.candidates == ["SARC"]


def test_exclusion_disabled_is_noop():
    rows = [
        {"code": "READ", "signature_score": 0.55, "support_geomean": 0.55},
        {"code": "SARC", "signature_score": 0.54, "support_geomean": 0.54},
    ]
    ev = _ev(4.0)
    r = classify_cancer_type_ontology(
        ranked_rows=rows, lineage_evidence=ev, use_lineage_exclusion=False, use_recall=False
    )
    assert "SARC" in r.candidates
