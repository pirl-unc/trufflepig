"""Subtype-aware code matching for the calibration script.

A model answer of "BRCA_LumB" against an expected label of "BRCA"
should count as correct (the call is MORE specific than the
expected parent cohort, not wrong). Symmetrically, "BRCA" against
expected "BRCA_LumB" is correct as a parent-class agreement.
"LUSC" against expected "BRCA" remains wrong (different cancer).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

# Load the script as a module (it isn't an installed package).
_spec = importlib.util.spec_from_file_location(
    "calibrate_decomposition",
    Path(__file__).resolve().parents[1] / "scripts" / "calibrate_decomposition.py",
)
calib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calib)


def test_exact_match_is_exact_kind():
    matched, kind = calib._codes_match("BRCA", "BRCA")
    assert matched is True
    assert kind == "exact"


def test_subtype_of_expected_counts_as_correct():
    """``BRCA_LumB`` vs expected ``BRCA`` → MORE specific, correct."""
    matched, kind = calib._codes_match("BRCA_LumB", "BRCA")
    assert matched is True
    assert kind == "subtype_of_expected"


def test_parent_of_expected_counts_as_correct():
    """``BRCA`` vs expected ``BRCA_LumB`` → less specific, correct family."""
    matched, kind = calib._codes_match("BRCA", "BRCA_LumB")
    assert matched is True
    assert kind == "parent_of_expected"


def test_unrelated_cohort_is_wrong():
    matched, kind = calib._codes_match("LUSC", "BRCA")
    assert matched is False
    assert kind == "none"


def test_sibling_subtype_is_wrong():
    """``BRCA_LumA`` vs expected ``BRCA_LumB`` → same parent but
    not the same subtype; counted as wrong because the model picked
    the wrong subtype of the right parent."""
    matched, kind = calib._codes_match("BRCA_LumA", "BRCA_LumB")
    assert matched is False, (
        "Sibling subtypes are different cancer types; classifier should "
        "not get credit for the wrong subtype just because the parent matches."
    )


def test_empty_inputs_are_wrong():
    assert calib._codes_match("", "BRCA") == (False, "none")
    assert calib._codes_match("BRCA", "") == (False, "none")


def test_top3_kind_picks_best_match_class():
    """If exact AND subtype both appear in top-3, ``kind`` should
    report the strongest agreement (exact > subtype > parent)."""
    matched, kind = calib._any_in_kin(["LUSC", "BRCA_LumB", "BRCA"], "BRCA")
    assert matched is True
    assert kind == "exact"


def test_top3_subtype_only_when_no_exact():
    matched, kind = calib._any_in_kin(["LUSC", "BRCA_LumB", "ESCA"], "BRCA")
    assert matched is True
    assert kind == "subtype_of_expected"


def test_top3_parent_only():
    matched, kind = calib._any_in_kin(["LUSC", "BRCA", "ESCA"], "BRCA_LumB")
    assert matched is True
    assert kind == "parent_of_expected"


def test_top3_no_kin():
    matched, kind = calib._any_in_kin(["LUSC", "ESCA", "HNSC"], "BRCA")
    assert matched is False
    assert kind == "none"


def test_local_expectation_uses_canonical_registry_aliases():
    """Legacy OS truth labels are equivalent to the canonical SARC_OS code."""

    assert calib._accept_expected("SARC_OS", "OS") is True


def test_cancer_type_decision_calibration_matches_production(
    monkeypatch,
):
    """Calibration must not count a residual call production would withhold."""

    import trufflepig.cancer_type_evidence as cancer_type_evidence
    import trufflepig.decomposition as decomposition
    import trufflepig.healthy_vs_tumor as healthy_vs_tumor
    import trufflepig.main as main
    import trufflepig.rare_inference as rare_inference
    import trufflepig.tumor_purity as tumor_purity

    trace = [
        {
            "code": code,
            "support_fraction_of_top": 1.0 - rank * 0.05,
        }
        for rank, code in enumerate(
            ["DLBC", "BRCA", "PRAD", "ACC", "SARC"]
        )
    ]
    monkeypatch.setattr(
        healthy_vs_tumor,
        "assess_healthy_vs_tumor",
        lambda _frame: {},
    )
    monkeypatch.setattr(
        tumor_purity,
        "analyze_sample",
        lambda _frame, tissue_signal=None: {"candidate_trace": trace},
    )
    monkeypatch.setattr(
        rare_inference,
        "infer_rare_cancer_marker_hypotheses_from_rna",
        lambda _frame, _analysis: [],
    )

    observed = {}

    def fake_decompose(_frame, **kwargs):
        observed["top_k"] = kwargs["top_k"]
        observed["candidate_codes"] = list(kwargs["cancer_types"])
        return list(range(40))

    def fake_residual(results, **_kwargs):
        observed["realizations_seen"] = len(results)
        return decomposition.CancerTypeDecision.from_dict(
            {
                "status": "resolved",
                "supported_code": "DLBC",
                "panel_code": "DLBC",
                "current_code": "SARC",
            }
        )

    monkeypatch.setattr(decomposition, "decompose_sample", fake_decompose)
    monkeypatch.setattr(main, "decompose_sample", fake_decompose)
    monkeypatch.setattr(
        decomposition,
        "decompose_identity_backgrounds",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        main,
        "decompose_identity_backgrounds",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        decomposition,
        "decide_cancer_type_from_decomposition",
        fake_residual,
    )

    decision_calls = []

    def fake_select(_frame, _analysis, **kwargs):
        if kwargs.get("cancer_type_decision"):
            decision_calls.append(kwargs["cancer_type_decision"])
        return {
            "selected": {
                "cancer_type": "SARC",
                "selected_by": "fused_evidence",
            }
        }

    monkeypatch.setattr(
        cancer_type_evidence,
        "select_report_scope_from_evidence",
        fake_select,
    )

    _trace, summary = calib._classify_one(
        pd.DataFrame(),
        include_cancer_type_decision=True,
    )

    assert observed["top_k"] is None
    assert observed["candidate_codes"] == [
        "SARC",
        "DLBC",
        "BRCA",
        "PRAD",
        "ACC",
    ]
    assert observed["realizations_seen"] == 40
    assert decision_calls[0].selection_allowed is False
    assert decision_calls[0].sample_mode == "solid"
    assert decision_calls[0].supported_code_mode == "heme"
    assert summary["consolidated_cancer_type"] == "SARC"


def test_cancer_type_decision_calibration_rolls_back_unconfirmed_scope(
    monkeypatch,
):
    """Calibration counts only cancer-type decisions that survive the production refit."""

    import trufflepig.cancer_type_evidence as cancer_type_evidence
    import trufflepig.decomposition as decomposition
    import trufflepig.healthy_vs_tumor as healthy_vs_tumor
    import trufflepig.main as main
    import trufflepig.rare_inference as rare_inference
    import trufflepig.tumor_purity as tumor_purity

    trace = [
        {"code": "CHOL", "support_fraction_of_top": 1.0},
        {"code": "BLCA", "support_fraction_of_top": 0.95},
    ]
    monkeypatch.setattr(
        healthy_vs_tumor,
        "assess_healthy_vs_tumor",
        lambda _frame: {},
    )
    monkeypatch.setattr(
        tumor_purity,
        "analyze_sample",
        lambda _frame, tissue_signal=None: {"candidate_trace": trace},
    )
    monkeypatch.setattr(
        rare_inference,
        "infer_rare_cancer_marker_hypotheses_from_rna",
        lambda _frame, _analysis: [],
    )
    monkeypatch.setattr(
        decomposition,
        "infer_sample_mode",
        lambda **_kwargs: "solid",
    )

    selection_calls = 0

    def fake_select(_frame, _analysis, **kwargs):
        nonlocal selection_calls
        selection_calls += 1
        if kwargs.get("cancer_type_decision"):
            return {
                "selected": {
                    "cancer_type": "BLCA",
                    "selected_by": "entity_consensus",
                }
            }
        return {
            "selected": {
                "cancer_type": "CHOL",
                "selected_by": "fused_evidence",
            }
        }

    monkeypatch.setattr(
        cancer_type_evidence,
        "select_report_scope_from_evidence",
        fake_select,
    )

    fit_scopes = []

    def fake_fit(_frame, _analysis, **kwargs):
        report_code = kwargs["report_code"]
        fit_scopes.append(report_code)
        return [report_code], [report_code, "BLCA"], {}

    monkeypatch.setattr(main, "_fit_report_scope_decompositions", fake_fit)

    identity_calls = 0

    def fake_identity(_results, **_kwargs):
        nonlocal identity_calls
        identity_calls += 1
        if identity_calls == 1:
            return decomposition.CancerTypeDecision.from_dict(
                {
                    "status": "resolved",
                    "supported_code": "BLCA",
                    "panel_code": "BLCA",
                    "current_code": "CHOL",
                }
            )
        return decomposition.CancerTypeDecision(
            status="ambiguous",
            selection_allowed=False,
        )

    monkeypatch.setattr(
        decomposition,
        "decide_cancer_type_from_decomposition",
        fake_identity,
    )
    def fake_propagate(analysis, _frame, *, report_scope_cancer_type, **_kwargs):
        analysis["report_scope_cancer_type"] = report_scope_cancer_type
        analysis["cancer_type"] = report_scope_cancer_type
        analysis["reference_cancer_type"] = report_scope_cancer_type
        analysis["expression_reference_cancer_type"] = report_scope_cancer_type

    monkeypatch.setattr(main, "_propagate_report_scope_selection", fake_propagate)

    class FakeContext:
        def __init__(self, analysis):
            self.analysis = analysis

        def code_for(self, _role):
            return str(
                self.analysis.get("report_scope_cancer_type")
                or self.analysis.get("cancer_type")
                or ""
            )

    monkeypatch.setattr(
        main,
        "_synchronize_cancer_type_context",
        lambda analysis, **_kwargs: FakeContext(analysis),
    )

    _trace, summary = calib._classify_one(
        pd.DataFrame(),
        include_cancer_type_decision=True,
    )

    assert selection_calls == 2
    assert fit_scopes == ["CHOL", "BLCA", "CHOL"]
    assert summary["pre_decision_cancer_type"] == "CHOL"
    assert summary["consolidated_cancer_type"] == "CHOL"
    assert summary["cancer_type_decision_refit_performed"] is True
    assert summary["cancer_type_decision_refit_accepted"] is False
    assert summary["cancer_type_decision_status"] == "ambiguous"


def test_non_decision_calibration_keeps_production_composition_evidence(
    monkeypatch,
):
    """Disabling cancer-type decision analysis must not change upstream cancer evidence."""
    import trufflepig.cancer_type_evidence as cancer_type_evidence
    import trufflepig.healthy_vs_tumor as healthy_vs_tumor
    import trufflepig.rare_inference as rare_inference
    import trufflepig.tumor_purity as tumor_purity

    tissue_signal = object()
    observed = {}
    trace = [
        {"code": "BLCA", "support_fraction_of_top": 1.0},
        {"code": "KIRP", "support_fraction_of_top": 0.8},
    ]
    monkeypatch.setattr(
        healthy_vs_tumor,
        "assess_healthy_vs_tumor",
        lambda _frame: tissue_signal,
    )

    def fake_analyze(_frame, *, tissue_signal=None):
        observed["analyze_tissue_signal"] = tissue_signal
        return {"candidate_trace": trace}

    monkeypatch.setattr(tumor_purity, "analyze_sample", fake_analyze)
    monkeypatch.setattr(
        rare_inference,
        "infer_rare_cancer_marker_hypotheses_from_rna",
        lambda _frame, _analysis: [],
    )

    def fake_select(_frame, analysis, **_kwargs):
        observed["selected_analysis"] = analysis
        return {
            "selected": {
                "cancer_type": "BLCA",
                "selected_by": "coarse_composition_reference",
            }
        }

    monkeypatch.setattr(
        cancer_type_evidence,
        "select_report_scope_from_evidence",
        fake_select,
    )

    returned_trace, summary = calib._classify_one(
        pd.DataFrame(),
        include_cancer_type_decision=False,
    )

    assert returned_trace == trace
    assert observed["analyze_tissue_signal"] is tissue_signal
    assert observed["selected_analysis"]["healthy_vs_tumor"] is tissue_signal
    assert summary["consolidated_cancer_type"] == "BLCA"
