from __future__ import annotations

import json

import pandas as pd
import pytest

from trufflepig.brief import recommend_therapies
from trufflepig.reporting import cancer_therapy_panel_for_analysis
from trufflepig.plot_target_deep_dive import _priority_target_rows
from trufflepig.treatment_history import (
    TreatmentRecord,
    add_population_therapy_evidence,
    parse_treatment_history,
    treatment_history_blocks_row,
    treatment_history_context,
    treatment_history_marks_current,
    treatment_history_supplement_rows,
)
from trufflepig.therapeutic_agents import agent_identity


@pytest.fixture
def synovial_therapy_case(clinical_hla_context):
    analysis = {
        "cancer_type": "SARC_SYN",
        "clinical_context": clinical_hla_context(["A*02:01"]),
    }
    _, _, panel = cancer_therapy_panel_for_analysis("SARC_SYN", analysis)
    expression = {
        **_fap_expression(), "symbol": "MAGEA4", "observed_tpm": 100.0,
        "attr_tumor_tpm": 90.0, "attr_tumor_tpm_low": 80.0,
        "attr_tumor_tpm_high": 100.0, "attr_tumor_fraction": 0.9,
        "attr_tumor_fraction_low": 0.8, "attr_tumor_fraction_high": 1.0,
        "attr_support_fraction": 1.0, "attr_top_compartment": "tumor",
        "tme_dominant": False, "tme_explainable": False,
    }
    ranges = pd.DataFrame([expression])
    selected = recommend_therapies(panel, ranges, analysis=analysis)
    assert selected[0].therapy["agent"] == "afami-cel (Tecelra)"
    assert selected[0].expression["symbol"] == "MAGEA4"
    return analysis, panel, ranges


@pytest.mark.parametrize("status", ["contraindicated", "progression", "current"])
@pytest.mark.parametrize("name", ["afami-cel", "Tecelra", "afamitresgene autoleucel"])
def test_real_synovial_panel_applies_named_history(synovial_therapy_case, status, name):
    analysis, original_panel, ranges = synovial_therapy_case
    analysis = {**analysis, "treatment_history": [{"therapy": name, "status": status}]}
    _, _, panel = cancer_therapy_panel_for_analysis("SARC_SYN", analysis)
    selected = recommend_therapies(panel, ranges, analysis=analysis)
    assert all(r.therapy["agent"] != "afami-cel (Tecelra)" for r in selected)
    assert len(panel) == len(original_panel)
    afami = panel.loc[panel["agent"].eq("afami-cel (Tecelra)")].iloc[0]
    assert treatment_history_blocks_row(afami, analysis) == (status != "current")
    assert treatment_history_marks_current(afami, analysis) == (status == "current")
    assert treatment_history_context(afami, analysis)
    assert treatment_history_supplement_rows(
        analysis, cancer_code="SARC_SYN", existing_rows=panel,
    ) == []


@pytest.mark.parametrize("status", ["contraindicated", "progression", "current"])
@pytest.mark.parametrize("target", ["MAGEA4", "MAGE-A4", "mage-a4"])
def test_real_synovial_panel_applies_class_history(synovial_therapy_case, status, target):
    analysis, original_panel, ranges = synovial_therapy_case
    analysis = {**analysis, "treatment_history": [
        {"target": target, "modality": "TCR_T", "status": status},
    ]}
    _, _, panel = cancer_therapy_panel_for_analysis("SARC_SYN", analysis)
    selected = recommend_therapies(panel, ranges, analysis=analysis)
    assert all(r.therapy["agent"] != "afami-cel (Tecelra)" for r in selected)
    assert len(panel) == len(original_panel)
    afami = panel.loc[panel["agent"].eq("afami-cel (Tecelra)")].iloc[0]
    assert treatment_history_blocks_row(afami, analysis) == (status != "current")
    assert treatment_history_marks_current(afami, analysis) == (status == "current")
    assert treatment_history_context(afami, analysis)
    assert treatment_history_supplement_rows(
        analysis, cancer_code="SARC_SYN", existing_rows=panel,
    ) == []


def test_named_and_class_history_retain_their_distinct_scope():
    named = {"treatment_history": [{
        "therapy": "afami-cel (Tecelra)", "target": "mage-a4",
        "modality": "TCR_T", "status": "contraindicated",
    }]}
    class_wide = {"treatment_history": [{
        "target": "mage-a4", "modality": "TCR_T", "status": "contraindicated",
    }]}
    same = {"agent": "afamitresgene autoleucel", "symbol": "MAGEA4", "agent_class": "TCR_T"}
    other_agent = {**same, "agent": "ADP-A2M4CD8"}
    other_modality = {**other_agent, "agent_class": "ADC"}
    other_antigen = {**other_agent, "symbol": "MAGEA1"}
    assert treatment_history_blocks_row(same, named)
    assert not treatment_history_blocks_row(other_agent, named)
    assert treatment_history_blocks_row(other_agent, class_wide)
    assert not treatment_history_blocks_row(other_modality, class_wide)
    assert not treatment_history_blocks_row(other_antigen, class_wide)
    assert treatment_history_blocks_row(
        {**same, "symbol": pd.NA, "target_gene": "MAGE-A4"}, class_wide,
    )
    assert TreatmentRecord.from_mapping(class_wide["treatment_history"][0]).target == "MAGEA4"


@pytest.mark.parametrize("record", [
    {"therapy": "afami-cel (Tecelra)", "status": "contraindicated"},
    {"target": "MAGE-A4", "modality": "TCR-T", "status": "contraindicated"},
])
def test_history_supplements_resolve_the_same_agent_and_target_metadata(record):
    analysis = {"treatment_history": [record]}
    rows = treatment_history_supplement_rows(analysis, cancer_code="SARC_SYN")
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "MAGEA4" and row["phase"] == "approved"
    assert agent_identity(row["agent"]) == agent_identity("afami-cel")
    assert treatment_history_blocks_row(row, analysis)


def test_recommendation_api_limit_and_input_contract(synovial_therapy_case):
    analysis, panel, ranges = synovial_therapy_case
    before_panel, before_ranges = panel.copy(deep=True), ranges.copy(deep=True)
    recommendations = recommend_therapies(panel, ranges, limit=1, analysis=analysis)
    assert len(recommendations) == 1
    assert recommendations[0].therapy["agent"] == "afami-cel (Tecelra)"
    assert recommendations[0].expression["observed_tpm"] == 100.0
    assert recommend_therapies(panel, ranges, limit=0, analysis=analysis) == []
    with pytest.raises(ValueError, match="nonnegative"):
        recommend_therapies(panel, ranges, limit=-1, analysis=analysis)
    pd.testing.assert_frame_equal(panel, before_panel)
    pd.testing.assert_frame_equal(ranges, before_ranges)


def test_recommendation_api_accepts_explicit_resolved_subtype(synovial_therapy_case):
    analysis, panel, ranges = synovial_therapy_case
    parent_context = {**analysis, "cancer_type": "SARC"}
    unresolved = recommend_therapies(panel, ranges, analysis=parent_context)
    resolved = recommend_therapies(
        panel, ranges, analysis=parent_context, panel_subtype="synovial_sarcoma",
    )
    assert all(r.therapy["agent"] != "afami-cel (Tecelra)" for r in unresolved)
    assert resolved[0].therapy["agent"] == "afami-cel (Tecelra)"
    assert parent_context == {**analysis, "cancer_type": "SARC"}


def test_contraindication_overrides_prior_benefit_under_another_alias(synovial_therapy_case):
    analysis, panel, ranges = synovial_therapy_case
    history = {**analysis, "treatment_history": [
        {"therapy": "afami-cel", "status": "major_benefit"},
        {"therapy": "Tecelra", "status": "contraindicated"},
    ]}
    selected = recommend_therapies(panel, ranges, analysis=history)
    assert all(r.therapy["agent"] != "afami-cel (Tecelra)" for r in selected)


def test_population_evidence_uses_the_same_registered_agent_identity():
    panel = pd.DataFrame([{"agent": name} for name in (
        "afami-cel (Tecelra)", "afamitresgene autoleucel", "Tecelra",
        "ADP-A2M4CD8", "afami-cel (different formulation)",
    )])
    enriched = add_population_therapy_evidence(
        panel, cancer_code="SARC", subtype="synovial_sarcoma",
    )
    assert enriched["benefit_tier"].tolist() == ["high_response"] * 3 + ["", ""]
    assert enriched.iloc[:3]["benefit_endpoint"].nunique() == 1
    assert enriched.iloc[0]["therapy_evidence_url"]


def _fap_history(status="major_benefit"):
    return {
        "treatment_history": [
            {
                "therapy": "FAP-targeted radioligand therapy",
                "target": "FAP",
                "modality": "RLT",
                "status": status,
                "note": "Very effective",
                "source": "clinical history",
            }
        ]
    }


def _fap_target():
    return {
        "cancer_code": "SARC_OS",
        "symbol": "FAP",
        "agent": "177Lu-FAP-2286",
        "agent_class": "RLT",
        "modality": "RLT",
        "phase": "phase_2",
        "indication": "FAP-directed investigational therapy",
        "treatment_path_tier": "trial_follow_up",
    }


def _fap_expression():
    return {
        "symbol": "FAP",
        "observed_tpm": 35.6,
        "attr_tumor_tpm": 4.3,
        "attr_tumor_tpm_low": 0.0,
        "attr_tumor_tpm_high": 8.0,
        "attr_tumor_fraction": 0.12,
        "attr_tumor_fraction_low": 0.0,
        "attr_tumor_fraction_high": 0.22,
        "attr_support_fraction": 0.1,
        "attr_top_compartment": "fibroblast",
        "attr_top_compartment_tpm": 31.3,
        "tme_dominant": True,
        "tme_explainable": True,
        "matched_normal_over_predicted": False,
    }


def test_parse_treatment_history_normalizes_plain_language_json(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            {
                "treatments": [
                    {
                        "therapy": "FAP-targeted radioligand therapy",
                        "target": "fap",
                        "modality": "radioligand therapy",
                        "status": "very effective",
                        "note": "Observed clinical response",
                    }
                ]
            }
        )
    )

    records = parse_treatment_history(path)

    assert len(records) == 1
    assert records[0].target == "FAP"
    assert records[0].modality == "RLT"
    assert records[0].status == "major_benefit"
    assert records[0].source_path == str(path)


def test_prior_benefit_keeps_background_attributed_fap_rlt_in_shortlist():
    target = _fap_target()
    expression = _fap_expression()

    without_history = recommend_therapies(
        pd.DataFrame([target]),
        pd.DataFrame([expression]),
        analysis={},
    )
    with_history = recommend_therapies(
        pd.DataFrame([target]),
        pd.DataFrame([expression]),
        analysis=_fap_history(),
    )

    assert without_history == []
    assert len(with_history) == 1
    assert with_history[0][0]["symbol"] == "FAP"
    bullet = therapy_review_text(
        with_history[0][0],
        with_history[0][1],
        analysis=_fap_history(),
    )
    assert "major prior benefit" in bullet
    assert "outranks the RNA source estimate" in bullet
    assert "current suitability" in bullet


def test_patient_history_supplements_therapy_missing_from_disease_panel():
    base = pd.DataFrame(
        [
            {
                "cancer_code": "SARC_OS",
                "symbol": "ERBB2",
                "agent": "trastuzumab deruxtecan",
                "agent_class": "ADC",
                "phase": "phase_2",
                "indication": "HER2-expressing osteosarcoma",
            }
        ]
    )

    _, _, panel = cancer_therapy_panel_for_analysis(
        "SARC_OS",
        {"cancer_type": "SARC_OS", **_fap_history()},
        therapy_targets_loader=lambda _code, subtype=None: base.copy(),
    )

    fap = panel.loc[panel["symbol"].astype(str).eq("FAP")]
    assert len(fap) == 1
    assert fap.iloc[0]["agent"] == "FAP-targeted radioligand therapy"
    assert fap.iloc[0]["phase"] == "patient_history"
    assert fap.iloc[0]["indication"] == "patient supplied treatment history"


def test_negative_outcome_blocks_only_the_named_agent():
    analysis = {
        "treatment_history": [
            {
                "therapy": "doxorubicin",
                "status": "progression",
                "note": "Progressed on treatment",
            }
        ]
    }
    doxorubicin = {"agent": "doxorubicin", "symbol": "", "agent_class": "small_molecule"}
    pazopanib = {"agent": "pazopanib", "symbol": "", "agent_class": "small_molecule"}

    assert treatment_history_blocks_row(doxorubicin, analysis) is True
    assert treatment_history_blocks_row(pazopanib, analysis) is False
    assert "do not prioritize" in treatment_history_context(doxorubicin, analysis).casefold()

    class_specific = {
        "treatment_history": [
            {
                "therapy": "177Lu-FAP-2286",
                "target": "FAP",
                "modality": "RLT",
                "status": "progression",
            }
        ]
    }
    another_fap_rlt = {
        "agent": "225Ac-FAPI-46",
        "symbol": "FAP",
        "agent_class": "RLT",
    }
    assert treatment_history_blocks_row(another_fap_rlt, class_specific) is False
    assert treatment_history_context(another_fap_rlt, class_specific) == ""


def test_negative_history_is_not_hidden_by_a_response_record():
    analysis = {
        "treatment_history": [
            {"therapy": "doxorubicin", "status": "benefit"},
            {"therapy": "doxorubicin", "status": "progression"},
        ]
    }
    row = {"agent": "doxorubicin", "symbol": "", "agent_class": "small_molecule"}

    assert treatment_history_blocks_row(row, analysis) is True
    assert "prior progression" in treatment_history_context(row, analysis)


def test_current_treatment_is_context_not_a_new_candidate():
    analysis = {
        "treatment_history": [
            {"therapy": "177Lu-FAP-2286", "status": "current"}
        ]
    }
    target = _fap_target()
    expression = _fap_expression()
    expression["attr_tumor_tpm"] = 25.0
    expression["attr_tumor_fraction"] = 0.70

    assert recommend_therapies(
        pd.DataFrame([target]),
        pd.DataFrame([expression]),
        analysis=analysis,
    ) == []
    _, figure_rows = _priority_target_rows(
        pd.DataFrame([expression]),
        "SARC_OS",
        target_panel=pd.DataFrame([target]),
        target_symbols=["FAP"],
        analysis=analysis,
    )
    assert figure_rows == []


def test_priority_figure_uses_the_same_patient_evidence_precedence():
    _, rows = _priority_target_rows(
        pd.DataFrame([_fap_expression()]),
        "SARC_OS",
        target_panel=pd.DataFrame([_fap_target()]),
        target_symbols=["FAP"],
        analysis=_fap_history(),
    )

    assert len(rows) == 1
    assert rows[0]["status_key"] == "patient_treatment_evidence"
    assert "patient treatment benefit supplied" in rows[0]["gate_label"]


def test_population_outcomes_join_by_exact_agent_and_disease():
    panel = pd.DataFrame(
        [
            {
                "cancer_code": "PRAD",
                "symbol": "FOLH1",
                "agent": "177Lu-PSMA-617",
                "agent_class": "radioligand",
                "phase": "approved",
            },
            {
                "cancer_code": "PRAD",
                "symbol": "FOLH1",
                "agent": "225Ac-PSMA-617",
                "agent_class": "radioligand",
                "phase": "phase_2",
            },
        ]
    )

    enriched = add_population_therapy_evidence(panel, cancer_code="PRAD")

    matched = enriched.loc[enriched["agent"].eq("177Lu-PSMA-617")].iloc[0]
    other = enriched.loc[enriched["agent"].eq("225Ac-PSMA-617")].iloc[0]
    assert matched["benefit_tier"] == "major_survival"
    assert matched["toxicity_tier"] == "moderate"
    assert "VISION" in matched["benefit_endpoint"]
    assert other["benefit_tier"] == ""


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA, ""])
def test_missing_modality_uses_agent_class_for_history_exclusion(missing):
    row = {**_fap_target(), "modality": missing}
    analysis = {"treatment_history": [
        {"target": "FAP", "modality": "RLT", "status": "progression"},
        {"therapy": "doxorubicin", "status": "progression"},
    ]}
    _, _, panel = cancer_therapy_panel_for_analysis(
        "SARC_OS", analysis,
        therapy_targets_loader=lambda _code, subtype=None: pd.DataFrame([row]),
    )
    fap = panel.loc[panel["agent"].eq(row["agent"])].iloc[0]
    assert treatment_history_blocks_row(fap, analysis)
    top = recommend_therapies(panel, pd.DataFrame([_fap_expression()]), analysis=analysis)
    assert all(target["agent"] != fap["agent"] for target, _ in top)


@pytest.mark.parametrize("status", ["progression", "contraindicated", "current"])
@pytest.mark.parametrize("name", ["FAP-2286", "177Lu-FAP-2286"])
def test_registered_alias_history_applies_only_to_same_agent(status, name):
    analysis = {"treatment_history": [
        {"therapy": name, "target": "FAP", "modality": "RLT", "status": status}
    ]}
    same = _fap_target()
    other = {**same, "agent": "225Ac-FAPI-46"}
    assert treatment_history_blocks_row(same, analysis) == (status != "current")
    assert treatment_history_marks_current(same, analysis) == (status == "current")
    assert not treatment_history_blocks_row(other, analysis)
    assert not treatment_history_marks_current(other, analysis)
    assert treatment_history_context(other, analysis) == ""
    assert treatment_history_supplement_rows(
        analysis, cancer_code="SARC_OS", existing_rows=[same]
    ) == []
    supplements = treatment_history_supplement_rows(
        analysis, cancer_code="SARC_OS", existing_rows=[other]
    )
    assert len(supplements) == 1
    assert supplements[0]["agent"] == name
    assert status.replace("contraindicated", "contraindication") in treatment_history_context(
        supplements[0], analysis
    )


@pytest.mark.parametrize("subtype,has_evidence", [("", False), ("leiomyosarcoma", False), ("gist", True)])
def test_population_subtype_evidence_requires_matching_context(subtype, has_evidence):
    panel = pd.DataFrame([{"agent": "imatinib", "symbol": "KIT"}])
    enriched = add_population_therapy_evidence(panel, cancer_code="SARC", subtype=subtype)
    assert (enriched.iloc[0].get("benefit_tier", "") == "major_survival") == has_evidence


def test_direct_gist_code_retains_population_evidence():
    code, subtype, panel = cancer_therapy_panel_for_analysis(
        "SARC_GIST", {"cancer_type": "SARC_GIST"}
    )
    assert (code, subtype) == ("SARC", "gist")
    imatinib = panel.loc[panel["agent"].eq("imatinib")].iloc[0]
    assert imatinib["benefit_tier"] == "major_survival"
    assert imatinib["toxicity_tier"]


def history_report_content(analysis, ranges):
    from trufflepig.report_content import build_report_content
    from trufflepig.report_view import build_report_view
    analysis = {"sample_mode": "solid", "purity": {}, **analysis}
    view = build_report_view(analysis)
    return build_report_content(analysis, ranges, analysis["cancer_type"], "", report_view=view)


def test_prior_treatment_shortlist_round_trips_rationale(tmp_path):
    analysis = {"cancer_type": "SARC", **_fap_history()}
    content = history_report_content(analysis, pd.DataFrame([_fap_expression()]))
    recommendation = next(row for row in content.therapy["rows"] if "Prior treatment" in row[1])
    assert "FAP" in recommendation[1]
    assert "major prior benefit" in recommendation[3]
    assert content.treatment_history


def test_osteosarcoma_paths_carry_disease_matched_evidence_without_rna_selection(tmp_path):
    analysis = {"cancer_type": "SARC_OS"}
    _, _, panel = cancer_therapy_panel_for_analysis("SARC_OS", analysis)
    top = recommend_therapies(panel, pd.DataFrame(), analysis=analysis)
    assert {row["agent"] for row, _ in top} == {"regorafenib", "cabozantinib"}
    doc = history_report_content(analysis, pd.DataFrame()).therapy
    assert len(doc["sources"]) == 2
    rego = next(row for row in doc["rows"] if row[1].startswith("regorafenib"))
    for expected in ("3.6 vs 1.7", "recurrent, progressive", "64%", "not established"):
        assert expected in rego[3]


def test_bladder_pembrolizumab_uses_treatment_setting_not_pd_l1_assay():
    from trufflepig.reporting import indication_biomarker

    _, _, panel = cancer_therapy_panel_for_analysis("BLCA", {"cancer_type": "BLCA"})
    row = panel.loc[panel["agent"].eq("pembrolizumab")].iloc[0]
    assert indication_biomarker(row) == "histology_only"
    assert "platinum" in row["eligibility_note"]
    assert "prior checkpoint therapy" in row["eligibility_note"]


def test_bladder_avelumab_requires_post_platinum_nonprogression_for_maintenance():
    _, _, panel = cancer_therapy_panel_for_analysis("BLCA", {"cancer_type": "BLCA"})
    row = panel.loc[panel["agent"].eq("avelumab")].iloc[0]
    assert "maintenance pathway" in row["eligibility_note"]
    assert "without progression after first-line platinum" in row["eligibility_note"]
    assert "prior checkpoint exposure" in row["eligibility_note"]


def test_agent_only_salvage_rationale_survives_both_detailed_tables():
    from trufflepig.brief import build_actionable
    from trufflepig.main import _build_target_report
    from trufflepig.report_view import build_report_view

    analysis = {
        "cancer_type": "SARC_OS", "cancer_name": "Osteosarcoma",
        "sample_mode": "solid",
        "purity": {"overall_estimate": 0.5, "overall_lower": 0.3, "overall_upper": 0.7},
    }
    ranges = pd.DataFrame({
        "symbol": pd.Series(dtype=str), "category": pd.Series(dtype=str),
        "is_cta": pd.Series(dtype=bool), "is_surface": pd.Series(dtype=bool),
        "median_est": pd.Series(dtype=float), "observed_tpm": pd.Series(dtype=float),
    })
    reports = [
        build_actionable(analysis, ranges, cancer_code="SARC_OS", disease_state="",
                         report_view=build_report_view(analysis)),
        _build_target_report(ranges, analysis, "SARC_OS", analysis["purity"]),
    ]
    for report in reports:
        rego = next(line for line in report.splitlines() if "| regorafenib |" in line)
        assert "recurrent, progressive" in rego
        assert "SARC024" in rego and "3.6 vs 1.7" in rego
        assert "64%" in rego
        assert "agent-only / no direct gene target" not in rego


def therapy_review_text(target, expression, target_panel=None, **context):
    from trufflepig.report_content import assess_therapy
    from trufflepig.brief import _expression_independent_evidence_gap
    assessment = assess_therapy(target, expression, target_panel=target_panel, **context)
    return " ".join([assessment['agent'], assessment['phase'], assessment['indication'],
                     *assessment['rationale'], assessment['maturity'],
                     _expression_independent_evidence_gap(target, context.get('analysis'))])
