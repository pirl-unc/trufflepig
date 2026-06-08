"""Tests for the tumour-intrinsic epithelial-exclusion lineage gate."""

from trufflepig.cancer_type_ontology import classify_cancer_type_ontology
from trufflepig.lineage_evidence import (
    DEFAULT_EPITHELIAL_THRESHOLD,
    EPITHELIAL_EXCLUDES,
    epithelial_hk_ratio,
    lineage_exclusion_evidence,
)

HK = 100.0


def _tpm(**ratios):
    return {sym: r * HK for sym, r in ratios.items()}


def test_silent_when_epithelial_absent():
    # real sarcoma: epithelial program off.
    ev = lineage_exclusion_evidence(_tpm(EPCAM=0.0, KRT8=0.0, KRT18=0.05), HK)
    assert ev.factors == {}
    assert ev.notes == []


def test_demotes_non_epithelial_when_epithelial_present():
    ev = lineage_exclusion_evidence(_tpm(EPCAM=3.0, KRT8=3.0, KRT18=3.0, KRT19=3.0, CDH1=3.0), HK)
    assert set(ev.factors) == set(EPITHELIAL_EXCLUDES)
    assert all(f < 1.0 for f in ev.factors.values())
    assert ev.notes


def test_demotion_is_confidence_proportional():
    weak = lineage_exclusion_evidence(_tpm(EPCAM=0.3, KRT8=0.3, KRT18=0.3, KRT19=0.3, CDH1=0.3), HK)
    strong = lineage_exclusion_evidence(_tpm(EPCAM=5.0, KRT8=5.0, KRT18=5.0, KRT19=5.0, CDH1=5.0), HK)
    assert strong.factors["mesenchymal"] < weak.factors["mesenchymal"]


def test_zero_hk_is_safe():
    assert epithelial_hk_ratio(_tpm(EPCAM=10.0), 0.0) == 0.0


def test_threshold_between_sarcoma_and_carcinoma():
    # observed on the ribosomal-free HK scale: sarcoma <=0.11x, carcinoma >=0.43x.
    assert 0.11 < DEFAULT_EPITHELIAL_THRESHOLD < 0.43


def test_stromal_sarcoma_confound_resolved_in_walk():
    # screen confused by stroma: READ top, SARC a close second.
    rows = [
        {"code": "READ", "signature_score": 0.67, "support_geomean": 0.67},
        {"code": "SARC", "signature_score": 0.54, "support_geomean": 0.54},
        {"code": "COAD", "signature_score": 0.52, "support_geomean": 0.52},
    ]
    ev = lineage_exclusion_evidence(_tpm(EPCAM=3.0, KRT8=3.0, KRT18=3.0, KRT19=3.0, CDH1=3.0), HK)
    r = classify_cancer_type_ontology(ranked_rows=rows, lineage_evidence=ev, use_recall=False)
    assert "SARC" not in r.candidates  # mesenchymal demoted out
    assert r.candidates[0] == "READ"
    assert any(t.startswith("[exclude]") for t in r.trace)


def test_real_sarcoma_untouched_in_walk():
    # epithelial absent -> no exclusion -> SARC survives.
    rows = [
        {"code": "SARC", "signature_score": 0.55, "support_geomean": 0.55},
        {"code": "BRCA", "signature_score": 0.40, "support_geomean": 0.40},
    ]
    ev = lineage_exclusion_evidence(_tpm(EPCAM=0.05, KRT8=0.0), HK)
    r = classify_cancer_type_ontology(ranked_rows=rows, lineage_evidence=ev, use_recall=False)
    assert r.candidates == ["SARC"]


def test_exclusion_disabled_is_noop():
    rows = [
        {"code": "READ", "signature_score": 0.55, "support_geomean": 0.55},
        {"code": "SARC", "signature_score": 0.54, "support_geomean": 0.54},
    ]
    ev = lineage_exclusion_evidence(_tpm(EPCAM=3.0, KRT8=3.0, KRT18=3.0), HK)
    r = classify_cancer_type_ontology(
        ranked_rows=rows, lineage_evidence=ev, use_lineage_exclusion=False, use_recall=False
    )
    # without the gate, READ/SARC stay close -> broad tie surfaces both
    assert "SARC" in r.candidates
