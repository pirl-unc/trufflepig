"""Tests for the validated report-finalization boundary."""

from __future__ import annotations

import dataclasses

import pytest

from trufflepig.confidence import (
    compute_call_confidence,
    purity_confidence_for_analysis,
)
from trufflepig.report_view import (
    Purity,
    build_report_view,
)


def _read_analysis():
    """A synthetic analysis dict shaped like the finalized READ/caris sample."""
    return {
        "cancer_type": "READ",
        "cancer_name": "Rectum Adenocarcinoma",
        "sample_mode": "solid",
        "top_cancers": [("READ", 1.0), ("COAD", 0.77), ("LUSC", 0.74)],
        "candidate_trace": [
            {"code": "READ", "support_fraction_of_top": 1.0, "signature_score": 0.78,
             "lineage_concordance": 0.998},
            {"code": "COAD", "support_fraction_of_top": 0.77, "signature_score": 0.80},
        ],
        "purity": {
            "overall_estimate": 0.10,
            "overall_lower": 0.06,
            "overall_upper": 0.16,
            "purity_source": "decomposition",
            "components": {"integration": {"source": "estimate+decomposition"}},
        },
        "sample_context": None,
    }


def test_build_report_view_extracts_finalized_purity():
    view = build_report_view(_read_analysis(), sample_id="S1")
    assert view.purity.estimate == pytest.approx(0.10)
    assert view.purity.lower == pytest.approx(0.06)
    assert view.purity.upper == pytest.approx(0.16)
    # purity_source wins over the integration source.
    assert view.purity.method == "decomposition"
    assert view.sample_id == "S1"


def test_purity_method_falls_back_to_integration_source():
    analysis = _read_analysis()
    del analysis["purity"]["purity_source"]
    view = build_report_view(analysis)
    assert view.purity.method == "estimate+decomposition"


def test_report_view_freezes_discordant_purity_status_and_scenarios():
    analysis = _read_analysis()
    analysis["purity"].update(
        {
            "quantitative_status": "discordant_estimators",
            "estimator_scenarios": [
                {
                    "source": "lineage_panel",
                    "estimate": 0.10,
                    "lower": 0.06,
                    "upper": 0.16,
                },
                {
                    "source": "signature",
                    "estimate": 0.43,
                    "lower": 0.32,
                    "upper": 0.55,
                },
            ],
        }
    )

    view = build_report_view(analysis)
    analysis["purity"]["quantitative_status"] = "resolved"
    analysis["purity"]["estimator_scenarios"] = []

    assert view.purity.status == "discordant_estimators"
    assert view.purity.scenarios[1] == ("signature", 0.43, 0.32, 0.55)
    assert view.purity == Purity(
        estimate=0.10,
        lower=0.06,
        upper=0.16,
        method="decomposition",
        confidence=view.purity.confidence,
        status="discordant_estimators",
        scenarios=view.purity.scenarios,
    )


def test_alternatives_are_ranker_candidates_minus_the_headline():
    view = build_report_view(_read_analysis())
    assert view.cancer_type == "READ"
    assert view.cancer_type_name == "Rectum Adenocarcinoma"
    # headline READ excluded; runner-ups retained with normalized support.
    assert view.cancer_type_alternatives == (("COAD", 0.77), ("LUSC", 0.74))


def test_alternatives_keep_the_true_top_candidate_when_scope_overrode_the_ranker():
    """Regression for the review finding: never positionally strip
    top_cancers[0]. When the report scope makes the headline differ from the
    ranker winner, the winner is a genuine alternative and must be kept."""
    analysis = _read_analysis()
    analysis["cancer_type"] = "READ"
    # ranker winner is COAD, headline is READ (report-scope override).
    analysis["top_cancers"] = [("COAD", 1.0), ("READ", 0.9), ("LUSC", 0.7)]
    view = build_report_view(analysis)
    codes = [c for c, _ in view.cancer_type_alternatives]
    assert "COAD" in codes  # the true strongest competitor is retained
    assert "READ" not in codes  # the headline is excluded


def test_alternatives_exclude_the_reference_cohort():
    analysis = _read_analysis()
    analysis["cancer_type"] = "READ_MSI"
    analysis["reference_cancer_type"] = "READ"
    analysis["top_cancers"] = [("READ", 1.0), ("COAD", 0.8)]
    view = build_report_view(analysis)
    codes = [c for c, _ in view.cancer_type_alternatives]
    assert codes == ["COAD"]  # both READ_MSI and its READ reference excluded


def test_confidence_tiers_do_not_drift_from_live_computation():
    """The view must reuse the exact confidence computations the markdown uses,
    so a report can never show one tier in a figure and another in text."""
    analysis = _read_analysis()
    view = build_report_view(analysis)
    assert view.call_confidence == compute_call_confidence(analysis)
    assert view.purity.confidence == purity_confidence_for_analysis(analysis)


def test_report_view_is_frozen():
    view = build_report_view(_read_analysis())
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.purity.estimate = 0.99  # type: ignore[misc]


@pytest.mark.parametrize("missing", ("cancer_type", "sample_mode", "purity"))
def test_build_report_view_requires_finalized_analysis_fields(missing):
    analysis = _read_analysis()
    del analysis[missing]

    with pytest.raises(ValueError, match=missing):
        build_report_view(analysis)


@pytest.mark.parametrize("sample_mode", (None, "", "auto", 7, []))
def test_build_report_view_rejects_nonfinal_sample_modes(sample_mode):
    analysis = _read_analysis()
    analysis["sample_mode"] = sample_mode

    with pytest.raises(ValueError, match="sample_mode"):
        build_report_view(analysis)


def test_report_finalization_rejects_malformed_alternatives():
    analysis = _read_analysis()
    analysis["top_cancers"] = [
        ("READ", 1.0),
        ("COAD", None),
        "garbage",
        ("LUSC", 0.7),
    ]

    with pytest.raises(TypeError, match="candidate support"):
        build_report_view(analysis)
