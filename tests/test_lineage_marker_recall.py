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
    _ribo_free_hk_symbols,
    marker_hk_median,
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
    # SCLC-like on the ribosomal-free HK scale: CHGA ~1.5x, ASCL1 ~0.7x — modest
    # for NE but well clear of the 0.3 bar, with a core marker present.
    p = neuroendocrine_recall(_tpm(CHGA=1.52, ASCL1=0.68, CHGB=0.36, INSM1=0.37), HK)
    assert p is not None


def test_fires_on_single_dominant_core_granin():
    # NET_MIDGUT-like: one very high core granin (CHGB ~5.5x), the rest modest.
    # The 2-marker minimum would miss it; the strong-single-core path catches it.
    p = neuroendocrine_recall(_tpm(CHGB=5.5, INSM1=0.12, SYP=0.12, SCG2=0.28), HK)
    assert p is not None


def test_single_modest_core_still_needs_two_markers():
    # A core marker just over the bar but below strong_core, alone, must not fire.
    p = neuroendocrine_recall(_tpm(CHGB=0.5, INSM1=0.0, SYP=0.0, SCG2=0.0), HK)
    assert p is None


def test_silent_on_non_ne():
    # COAD/BRCA-like: NE program essentially off.
    assert neuroendocrine_recall(_tpm(CHGA=0.0, SYP=0.0, SCG2=0.01), HK) is None


def test_ascl1_alone_does_not_fire_without_core_marker():
    # NUT carcinoma / NE-LUAD: ASCL1 elevated, but no core granin/INSM1.
    # ASCL1 + an incidental SCG2 clear the 2-marker count, so this pins the
    # core-marker obligate (the pfo019 false-positive that motivated it).
    p = neuroendocrine_recall(_tpm(ASCL1=0.62, SCG2=0.35, CHGA=0.0, CHGB=0.0, INSM1=0.0), HK)
    assert p is None


def test_core_marker_alone_insufficient_needs_two():
    # A single core marker over threshold shouldn't fire on its own.
    p = neuroendocrine_recall(_tpm(CHGA=0.4, CHGB=0.0, INSM1=0.0, SYP=0.0), HK)
    assert p is None


def test_zero_hk_median_is_safe():
    assert neuroendocrine_recall(_tpm(CHGA=10.0), 0.0) is None


def test_core_markers_are_granins_and_insm1():
    assert set(CORE_MARKERS) == {"CHGA", "CHGB", "INSM1"}


def test_marker_hk_set_excludes_ribosomal_protein_genes():
    # The gate denominator must exclude RPL*/RPS* — those are in v4 clean-TPM's
    # pinned ribosomal compartment, which would couple the HK median to the
    # normalization (~2.4x inflation on v4 vs ~1.4x on legacy).
    syms = _ribo_free_hk_symbols()
    assert syms, "ribosomal-free HK set is empty"
    assert not any(s.startswith(("RPL", "RPS")) for s in syms)
    # the median ignores ribosomal genes even when they're huge
    tpm = {s: 1.0 for s in syms}
    tpm["RPLP0"] = 99999.0
    assert marker_hk_median(tpm) == 1.0


def test_recall_candidates_collects_fired_proposals():
    fired = recall_candidates(_tpm(CHGA=12.0, CHGB=12.0, SYP=1.0), HK)
    assert len(fired) == 1
    assert fired[0].broad == "neuroendocrine"
    assert recall_candidates(_tpm(CHGA=0.0), HK) == []


def test_recall_injection_routes_to_neuroendocrine_branch():
    # screen scattered a no-reference NE sample low across LUAD/DLBC; recall
    # rescues it (injected at the 0.5 floor, clearly above the scattered signal).
    rows = [
        {"code": "LUAD", "signature_score": 0.40, "support_geomean": 0.40},
        {"code": "DLBC", "signature_score": 0.35, "support_geomean": 0.35},
    ]
    props = recall_candidates(_tpm(CHGA=12.0, CHGB=12.0, SYP=1.0), HK)
    r = classify_cancer_type_ontology(ranked_rows=rows, recall_proposals=props)
    assert all(c in NE_NO_REFERENCE for c in r.candidates)
    assert r.recall_notes
    assert r.trace[0].startswith("[recall]")


def test_recall_only_ties_a_confident_signature():
    # SCLC-like overlap: a strong lung signature (LUAD 0.6) competes with the NE
    # program. Recall is capped at the signature, so the broad walk surfaces BOTH
    # lineages rather than letting markers override the call.
    rows = [
        {"code": "LUAD", "signature_score": 0.60, "support_geomean": 0.60},
        {"code": "DLBC", "signature_score": 0.55, "support_geomean": 0.55},
    ]
    props = recall_candidates(_tpm(CHGA=12.0, CHGB=12.0, SYP=1.0), HK)
    r = classify_cancer_type_ontology(ranked_rows=rows, recall_proposals=props)
    cands = set(r.candidates)
    assert "LUAD" in cands and cands & set(NE_NO_REFERENCE)  # both surfaced
    # NE injected at most to the signature ceiling, never above it
    assert max(r.scores[c] for c in NE_NO_REFERENCE) <= 0.60 + 1e-9


def test_recall_is_additive_does_not_lower_screen_candidates():
    rows = [{"code": "LUAD", "signature_score": 0.45, "support_geomean": 0.45}]
    r = classify_cancer_type_ontology(ranked_rows=rows, recall_proposals=[], use_recall=True)
    assert r.scores["LUAD"] == 0.45  # untouched when nothing fires
