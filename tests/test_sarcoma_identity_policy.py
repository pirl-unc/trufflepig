"""Muscle/stromal RNA must never supply a sarcoma diagnosis or therapy scope."""

import copy

import pandas as pd
import pytest

from trufflepig.cancer_type_policy import (
    SARCOMA_IDENTITY_REASON,
    finalize_sarcoma_identity,
    mark_unresolved_identity_purity,
)


@pytest.mark.parametrize("code", ["SARC", "SARC_LMS", "SARC_RMS_ERMS", "SARC_PEC", "SARC_OS"])
def test_sarcoma_identity_is_a_persistent_blocker_across_rna_selectors(code):
    from trufflepig.cancer_type_evidence import CancerTypeEvidence, _persistent_report_label_blockers

    row = CancerTypeEvidence(cancer_type=code, broad_rna_support=1, fine_reference_support=1,
                             learned_expression_support=1, rna_marker_support=1)
    for selector in ("fine_reference", "local_expression_reference", "lineage_panel",
                     "rare_marker", "fused_evidence", "entity_evidence_consensus"):
        row.consider_for_report_label(selected_by=selector, can_select=True,
                                      blocking_reasons=(), priority=(5, 1.0))
        assert SARCOMA_IDENTITY_REASON in _persistent_report_label_blockers(row)
    row.direct_fusion_support = 1
    assert not _persistent_report_label_blockers(row)


@pytest.mark.parametrize("code,markers", [
    ("SARC_LMS", {"ACTA2": 5000, "TAGLN": 5000, "MYH11": 5000, "DES": 5000}),
    ("SARC_RMS_ERMS", {"MYOD1": 200, "MYOG": 200, "MYF5": 200, "DES": 5000}),
    ("SARC", {"COL1A1": 9000, "COL1A2": 9000, "DCN": 9000, "VIM": 9000}),
])
def test_muscle_and_stromal_programs_cannot_select_sarcoma(code, markers):
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence
    from trufflepig.common import build_sample_tpm_by_symbol, panel_symbols_to_gene_ids

    gene_ids = panel_symbols_to_gene_ids(markers)
    expression = pd.DataFrame({"canonical_gene_name": list(markers), "TPM": list(markers.values()),
                               "ensembl_gene_id": [gene_ids[gene] for gene in markers]})
    assert build_sample_tpm_by_symbol(expression) == markers
    analysis = {"cancer_type": "SARC", "candidate_trace": [
        {"code": "SARC", "support_fraction_of_top": 1, "winning_subtype": code},
    ]}
    result = select_report_scope_from_evidence(expression, analysis)
    assert result["selected"] is None
    assert result["staged_evidence_graph"]["selected"] is None
    sarcoma = [row for row in result["evidence"] if row["cancer_type"].startswith("SARC")]
    assert sarcoma
    assert all(not row["can_select_report_label"] for row in sarcoma)
    assert all(SARCOMA_IDENTITY_REASON in row["blocking_reasons"] for row in sarcoma)


def unresolved_analysis():
    return {"cancer_type": "SARC", "cancer_name": "Sarcoma", "cancer_type_source": "auto-detected",
            "sample_mode": "mesenchymal", "top_cancers": [("SARC_LMS", 1.0)],
            "purity": {"overall_estimate": .8, "overall_lower": .7, "overall_upper": .9},
            "variant_records": [{"gene": "EGFR", "variant": "kinase domain duplication"}]}


def test_ranker_fallback_abstains_and_cannot_supply_a_therapy_scope():
    from trufflepig.brief import _cancer_type_basis_line
    from trufflepig.report_view import build_report_view
    from trufflepig.reporting import cancer_therapy_panel_for_analysis
    from trufflepig.therapy_eligibility import evaluate_therapy_eligibility

    analysis = unresolved_analysis()
    with pytest.raises(ValueError, match="RNA-only sarcoma"):
        build_report_view(analysis)
    assert finalize_sarcoma_identity(analysis)
    mark_unresolved_identity_purity(analysis)
    view = build_report_view(analysis)
    assert view.cancer_type == "UNRESOLVED"
    assert view.call_confidence.tier == "unknown"
    assert not view.cancer_type_alternatives
    assert view.purity.unresolved_reason == "cancer_type_unresolved"
    assert analysis["reference_cancer_type"] == "SARC"
    assert SARCOMA_IDENTITY_REASON in _cancer_type_basis_line(analysis, view.cancer_type)
    assert cancer_therapy_panel_for_analysis("SARC", analysis)[2].empty
    eligibility = evaluate_therapy_eligibility(
        {"symbol": "VEGFA", "agent": "pazopanib", "indication": "soft tissue sarcoma"}, analysis,
    )
    assert not eligibility.permits_review
    assert any(r.kind == "scope" and r.status == "missing" for r in eligibility.requirements)
    assert not finalize_sarcoma_identity(analysis)  # idempotent


@pytest.mark.parametrize("extra", [
    {"cancer_type_source": "user-specified", "analysis_constraints": {"cancer_type": "SARC"}},
    {"cancer_type_evidence": {"selected": {"cancer_type": "SARC", "selected_by": "direct_fusion",
      "can_select_report_label": True, "metrics": {"direct_fusion_support": 1}}}},
])
def test_supplied_diagnosis_or_defining_molecular_identity_is_preserved(extra):
    analysis = {**unresolved_analysis(), **extra}
    before = copy.deepcopy(analysis)
    assert not finalize_sarcoma_identity(analysis)
    assert analysis == before


def test_residual_fit_and_aneuploidy_cannot_establish_sarcoma():
    analysis = unresolved_analysis()
    analysis.update(cancer_type_decision={"status": "supported", "supported_code": "SARC",
                                         "background_separation_confirmed": True},
                    residual_aneuploidy_amplitude=.9, proliferation_score=1)
    assert finalize_sarcoma_identity(analysis)


def test_residual_decision_cannot_propose_sarcoma_even_after_subtraction():
    from trufflepig.decomposition.cancer_type_decision import CancerTypeDecision

    decision = CancerTypeDecision(status="resolved", current_code="PRAD", supported_code="SARC_LMS",
                                  background_separation_confirmed=True)
    assert not decision.selection_allowed
    assert not decision.proposed_code
    assert not decision.supports("SARC_LMS")
    assert decision.to_dict()["block_reason"] == SARCOMA_IDENTITY_REASON


def test_supplied_sarcoma_parent_does_not_license_a_muscle_derived_subtype():
    from trufflepig.reporting import candidate_winning_subtype_for_analysis

    analysis = {**unresolved_analysis(), "cancer_type_source": "user-specified",
                "analysis_constraints": {"cancer_type": "SARC"},
                "candidate_trace": [{"code": "SARC", "winning_subtype": "SARC_LMS"}]}
    assert candidate_winning_subtype_for_analysis(analysis) is None
    assert not finalize_sarcoma_identity(analysis)


def test_public_sample_analysis_cannot_return_the_sarcoma_ranker_fallback(monkeypatch):
    from trufflepig import plot, tumor_purity

    monkeypatch.setattr(plot, "_compute_cancer_type_signature_stats", lambda _df: [{"code": "SARC"}])
    monkeypatch.setattr(tumor_purity, "_build_sample_tpm_by_symbol", lambda _df: {})
    monkeypatch.setattr(tumor_purity, "rank_cancer_type_candidates", lambda *_a, **_kw: [
        {"code": "SARC", "support_score": 1, "support_fraction_of_top": 1, "purity_result": {}},
    ])
    monkeypatch.setattr(tumor_purity, "finalize_winner_purity", lambda *_a, **_kw: {})
    monkeypatch.setattr(tumor_purity, "_summarize_candidate_family", lambda _rows: {})
    monkeypatch.setattr(tumor_purity, "_summarize_fit_quality", lambda *_a: {})
    monkeypatch.setattr(tumor_purity, "_score_host_tissue_details", lambda *_a, **_kw: [])
    monkeypatch.setattr(tumor_purity, "_get_mhc_expression", lambda _sample: ({}, {}))
    result = tumor_purity.analyze_sample(pd.DataFrame())
    assert result["cancer_type"] == "UNRESOLVED"
    assert result["reference_cancer_type"] == "SARC"
    assert result["purity"]["quantitative_unresolved_reason"] == "cancer_type_unresolved"


def test_unresolved_report_cannot_reintroduce_sarcoma_via_markers_or_narrative():
    from trufflepig.main import _integrated_evidence_bullets, _rare_marker_hypotheses_markdown, _tumor_type_sanity_markdown
    from trufflepig.report_content import build_report_content, render_report_summary
    from trufflepig.report_view import build_report_view

    analysis = unresolved_analysis()
    analysis.update(
        rare_marker_hypotheses=[{"cancer_type": "SARC_DFSP", "surrogate": "COL1A1", "surrogate_tpm": 900}],
        tumor_type_sanity={"code": "SARC", "name": "Sarcoma"},
        candidate_trace=[{"code": "SARC", "support_fraction_of_top": 1},
                         {"code": "SARC_LMS", "support_fraction_of_top": .2}],
    )
    finalize_sarcoma_identity(analysis)
    mark_unresolved_identity_purity(analysis)
    view = build_report_view(analysis)
    content = build_report_content(analysis, pd.DataFrame(), "UNRESOLVED", "", report_view=view)
    summary = render_report_summary(content)
    assert "Disease-specific therapies are withheld" in summary
    assert "not yet in the curated" not in summary
    assert "Rare-marker prompt" not in summary
    assert "SARC_DFSP" not in summary
    assert "do not establish tumor-cell origin" in summary
    assert not _rare_marker_hypotheses_markdown(analysis)
    assert "Report label name" not in _tumor_type_sanity_markdown(analysis)
    bullets = "\n".join(_integrated_evidence_bullets(analysis))
    assert "Cancer identity**: unresolved" in bullets
    assert "integrated evidence selected UNRESOLVED" not in bullets
    assert "ahead of" not in bullets
