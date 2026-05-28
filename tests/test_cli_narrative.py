# Licensed under the Apache License, Version 2.0

"""Unit tests for CLI narrative and report helper functions.

Covers: compose_disease_state_narrative (BRCA ER/HER2, EMT, IFN),
annotate_surface_targets_with_cross_signals, _summarize_sample_call,
_candidate_label_options.
"""

from types import SimpleNamespace
import pytest

from trufflepig.main import (
    compose_disease_state_narrative,
    annotate_surface_targets_with_cross_signals,
    _candidate_label_options,
    _summarize_sample_call,
    _CORE_ISG_SURFACE,
)


def _mock_therapy_scores(**axis_states):
    """Build a therapy_scores dict with named axes set to given states."""
    scores = {}
    for cls, state in axis_states.items():
        scores[cls] = SimpleNamespace(state=state, message=f"{cls} is {state}")
    return scores


def _base_analysis(**overrides):
    """Minimal analysis dict."""
    a = {
        "cancer_type": "PRAD",
        "cancer_name": "Prostate Adenocarcinoma",
        "therapy_response_scores": {},
        "purity": {
            "overall_estimate": 0.6,
            "components": {"lineage": {"per_gene": []}},
        },
        "candidate_trace": [{"code": "PRAD", "support_fraction_of_top": 1.0}],
        "fit_quality": {},
    }
    a.update(overrides)
    return a


# ── compose_disease_state_narrative: BRCA patterns ──────────────────────


def test_brca_er_down_pattern():
    analysis = _base_analysis(
        cancer_type="BRCA",
        therapy_response_scores=_mock_therapy_scores(ER_signaling="down"),
        purity={
            "overall_estimate": 0.6,
            "components": {
                "lineage": {
                    "per_gene": [
                        {"gene": "ESR1", "purity": 0.01},
                    ]
                }
            },
        },
    )
    narrative = compose_disease_state_narrative(analysis)
    assert "ER" in narrative
    assert "suppressed" in narrative.lower() or "endocrine" in narrative.lower()


def test_brca_her2_up_pattern():
    analysis = _base_analysis(
        cancer_type="BRCA",
        therapy_response_scores=_mock_therapy_scores(HER2_signaling="up"),
    )
    narrative = compose_disease_state_narrative(analysis)
    assert "HER2" in narrative


def test_brca_combined_er_down_her2_up():
    analysis = _base_analysis(
        cancer_type="BRCA",
        therapy_response_scores=_mock_therapy_scores(
            ER_signaling="down",
            HER2_signaling="up",
        ),
        purity={
            "overall_estimate": 0.6,
            "components": {
                "lineage": {
                    "per_gene": [
                        {"gene": "ESR1", "purity": 0.01},
                    ]
                }
            },
        },
    )
    narrative = compose_disease_state_narrative(analysis)
    assert "ER" in narrative
    assert "HER2" in narrative


# ── compose_disease_state_narrative: EMT / hypoxia ──────────────────────


def test_emt_plus_hypoxia_aggressive_pattern():
    analysis = _base_analysis(
        therapy_response_scores=_mock_therapy_scores(EMT="up", hypoxia="up"),
    )
    narrative = compose_disease_state_narrative(analysis)
    assert "EMT" in narrative
    assert "hypoxia" in narrative.lower()
    assert "aggressive" in narrative.lower()


def test_emt_only_mesenchymal_switch():
    analysis = _base_analysis(
        therapy_response_scores=_mock_therapy_scores(EMT="up"),
    )
    narrative = compose_disease_state_narrative(analysis)
    assert "EMT" in narrative
    assert "mesenchymal" in narrative.lower()


def test_hypoxia_only_no_mention():
    """Hypoxia alone without EMT doesn't trigger the combined pattern."""
    analysis = _base_analysis(
        therapy_response_scores=_mock_therapy_scores(hypoxia="up"),
    )
    narrative = compose_disease_state_narrative(analysis)
    assert "aggressive" not in narrative.lower()


# ── compose_disease_state_narrative: IFN ─────────────────────────────────


def test_ifn_up_mentions_inflation():
    analysis = _base_analysis(
        therapy_response_scores=_mock_therapy_scores(IFN_response="up"),
    )
    narrative = compose_disease_state_narrative(analysis)
    assert "IFN" in narrative
    assert "inflation" in narrative.lower() or "IFN-driven" in narrative


def test_ifn_down_no_mention():
    analysis = _base_analysis(
        therapy_response_scores=_mock_therapy_scores(IFN_response="down"),
    )
    narrative = compose_disease_state_narrative(analysis)
    assert "inflation" not in narrative.lower()


# ── annotate_surface_targets_with_cross_signals ──────────────────────────


def test_annotate_ifn_active_tags_core_isg():
    scores = {"IFN_response": SimpleNamespace(state="up")}
    result = annotate_surface_targets_with_cross_signals(None, scores)
    assert "HLA-A" in result
    assert result["HLA-A"] == "IFN-driven"
    assert "B2M" in result


def test_annotate_ifn_inactive_returns_empty():
    scores = {"IFN_response": SimpleNamespace(state="down")}
    result = annotate_surface_targets_with_cross_signals(None, scores)
    assert result == {}


def test_annotate_ifn_missing_returns_empty():
    result = annotate_surface_targets_with_cross_signals(None, {})
    assert result == {}


# ── _candidate_label_options ─────────────────────────────────────────────


def test_candidate_label_single_strong():
    analysis = {
        "candidate_trace": [{"code": "PRAD"}, {"code": "BRCA"}],
        "fit_quality": {"label": "strong"},
    }
    assert _candidate_label_options(analysis) == ["PRAD"]


def test_candidate_label_weak_shows_two():
    analysis = {
        "candidate_trace": [{"code": "COAD"}, {"code": "READ"}],
        "fit_quality": {"label": "weak"},
    }
    assert _candidate_label_options(analysis) == ["COAD", "READ"]


def test_candidate_label_ambiguous_shows_two():
    analysis = {
        "candidate_trace": [{"code": "LUAD"}, {"code": "LUSC"}],
        "fit_quality": {"label": "ambiguous"},
    }
    assert _candidate_label_options(analysis) == ["LUAD", "LUSC"]


def test_candidate_label_respects_forced_cancer_type():
    analysis = {
        "analysis_constraints": {"cancer_type": "COAD"},
        "candidate_trace": [{"code": "SARC"}, {"code": "COAD"}],
        "fit_quality": {"label": "ambiguous"},
    }
    assert _candidate_label_options(analysis) == ["COAD"]


def test_candidate_label_empty_trace():
    analysis = {"candidate_trace": [], "fit_quality": {}}
    assert _candidate_label_options(analysis) == []


# ── _summarize_sample_call ───────────────────────────────────────────────


def _mock_decomp_result(**kwargs):
    defaults = {
        "cancer_type": "COAD",
        "template": "solid_primary",
        "score": 0.9,
        "purity": 0.6,
        "warnings": [],
        "template_site_factor": 0.9,
        "template_tissue_score": 0.8,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_summarize_call_primary_template():
    analysis = _base_analysis(
        cancer_type="COAD",
        fit_quality={"label": "strong"},
        candidate_trace=[{"code": "COAD", "support_fraction_of_top": 1.0}],
    )
    best = _mock_decomp_result(template="solid_primary")
    result = _summarize_sample_call(analysis, [best], sample_mode="solid")
    assert result["reported_context"] == "primary"
    assert result["site_indeterminate"] is False


def test_summarize_call_met_template_with_good_site():
    analysis = _base_analysis(
        cancer_type="COAD",
        fit_quality={"label": "strong"},
    )
    best = _mock_decomp_result(
        template="met_liver",
        template_site_factor=0.9,
        template_tissue_score=0.8,
        warnings=[],
    )
    result = _summarize_sample_call(analysis, [best], sample_mode="solid")
    assert result["reported_context"] == "met"
    assert result["reported_site"] == "liver-associated host context"


def test_summarize_call_weak_fit_met_is_indeterminate():
    analysis = _base_analysis(
        cancer_type="COAD",
        fit_quality={"label": "weak"},
    )
    best = _mock_decomp_result(template="met_liver")
    result = _summarize_sample_call(analysis, [best], sample_mode="solid")
    assert result["site_indeterminate"] is True
    assert result["site_note"] is not None


def test_summarize_call_low_site_factor_is_indeterminate():
    analysis = _base_analysis(
        cancer_type="COAD",
        fit_quality={"label": "strong"},
    )
    best = _mock_decomp_result(
        template="met_liver",
        template_site_factor=0.5,  # below 0.75
        template_tissue_score=0.3,  # below 0.4
        warnings=[],
    )
    result = _summarize_sample_call(analysis, [best], sample_mode="solid")
    assert result["site_indeterminate"] is True
