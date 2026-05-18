"""Regression tests for issue #169 — cancer-type call confidence.

The classifier's top candidate is its best guess, but that guess can
be fragile when orthogonal signals disagree. ``compute_call_confidence``
downgrades the tier to ``low`` / ``moderate`` and surfaces reader-
facing reasons when:

1. Top candidate's lineage concordance < 0.2 (near zero)
2. Geomean gap to runner-up < 1.1× (tied call)
3. Step-0 top-ρ TCGA cohort disagrees with the classifier's pick

Canonical failure the tier catches: a real sarcoma validation
sample at 4.35.0 → THYM with concordance 0.000 and a 0.002 geomean
margin over SARC.
"""

from trufflepig.confidence import compute_call_confidence


def _candidate(code, **overrides):
    base = {
        "code": code,
        "signature_score": 0.6,
        "support_geomean": 0.5,
        "lineage_concordance": 0.8,
        "family_label": None,
        "purity_estimate": 0.5,
    }
    base.update(overrides)
    return base


def test_clean_call_returns_high_tier():
    """A top candidate with strong concordance and a big lead returns
    ``high`` confidence with no reasons."""
    analysis = {
        "candidate_trace": [
            _candidate("PRAD", support_geomean=0.70, lineage_concordance=0.95),
            _candidate("BRCA", support_geomean=0.30, lineage_concordance=0.20),
        ],
    }
    tier = compute_call_confidence(analysis)
    assert tier.tier == "high"
    assert tier.reasons == []


def test_zero_concordance_top_call_is_low_confidence():
    """The SARC → THYM regression: top candidate's lineage genes
    aren't expressed in the sample (concordance 0.0). That's a
    contested call, not a clean win."""
    analysis = {
        "candidate_trace": [
            _candidate("THYM", support_geomean=0.43, lineage_concordance=0.0),
            _candidate("SARC", support_geomean=0.42, lineage_concordance=1.0),
        ],
    }
    tier = compute_call_confidence(analysis)
    assert tier.tier == "low"
    # Reason should mention the concordance value
    assert any("concordance" in r.lower() for r in tier.reasons)
    assert any("0.00" in r for r in tier.reasons)


def test_tied_geomean_downgrades_to_moderate():
    """Geomean gap of < 10% is a tied call — downgrade to moderate."""
    analysis = {
        "candidate_trace": [
            _candidate("A", support_geomean=0.50, lineage_concordance=0.8),
            _candidate("B", support_geomean=0.48, lineage_concordance=0.8),
        ],
    }
    tier = compute_call_confidence(analysis)
    assert tier.tier == "moderate"
    assert any("ambiguous" in r.lower() for r in tier.reasons)


def test_step0_mismatch_downgrades_to_moderate():
    """Step-0 correlation favors a cohort the classifier didn't pick
    — surface the mismatch."""

    class _HVT:
        top_tcga_cohorts = [("SARC_TPM", 0.77)]

    analysis = {
        "candidate_trace": [
            _candidate("THYM", support_geomean=0.43, lineage_concordance=0.8),
            _candidate("BRCA", support_geomean=0.30, lineage_concordance=0.6),
        ],
        "healthy_vs_tumor": _HVT(),
    }
    tier = compute_call_confidence(analysis)
    assert tier.tier == "moderate"
    assert any("Step-0" in r and "SARC" in r for r in tier.reasons)


def test_multiple_contradictions_all_surface_at_low_tier():
    """Zero concordance + Step-0 mismatch + tied geomean → tier stays
    ``low`` and all three reasons appear."""

    class _HVT:
        top_tcga_cohorts = [("SARC_TPM", 0.77)]

    analysis = {
        "candidate_trace": [
            _candidate("THYM", support_geomean=0.431, lineage_concordance=0.0),
            _candidate("SARC", support_geomean=0.429, lineage_concordance=1.0),
        ],
        "healthy_vs_tumor": _HVT(),
    }
    tier = compute_call_confidence(analysis)
    assert tier.tier == "low"
    assert len(tier.reasons) == 3


def test_missing_candidate_trace_returns_unknown():
    tier = compute_call_confidence({})
    assert tier.tier == "unknown"
    tier = compute_call_confidence({"candidate_trace": []})
    assert tier.tier == "unknown"


def test_step0_match_does_not_downgrade():
    """Step-0 agreeing with the classifier keeps the tier high."""

    class _HVT:
        top_tcga_cohorts = [("PRAD_TPM", 0.82)]

    analysis = {
        "candidate_trace": [
            _candidate("PRAD", support_geomean=0.70, lineage_concordance=0.95),
            _candidate("BRCA", support_geomean=0.30, lineage_concordance=0.20),
        ],
        "healthy_vs_tumor": _HVT(),
    }
    tier = compute_call_confidence(analysis)
    assert tier.tier == "high"


def test_weak_fit_quality_caps_call_confidence_low():
    analysis = {
        "fit_quality": {
            "label": "weak",
            "message": "Top support remains close to background.",
        },
        "candidate_trace": [
            _candidate("READ", support_geomean=0.70, lineage_concordance=0.95),
            _candidate("COAD", support_geomean=0.30, lineage_concordance=0.80),
        ],
    }
    tier = compute_call_confidence(analysis)
    assert tier.tier == "low"
    assert any("fit quality is weak" in r for r in tier.reasons)


def test_raw_signature_tension_downgrades_clean_support_call():
    analysis = {
        "candidate_trace": [
            _candidate("READ", signature_score=0.42, support_geomean=0.70),
            _candidate("SARC", signature_score=0.65, support_geomean=0.30),
        ],
    }
    tier = compute_call_confidence(analysis)
    assert tier.tier == "moderate"
    assert any("raw signature score favored SARC" in r for r in tier.reasons)
