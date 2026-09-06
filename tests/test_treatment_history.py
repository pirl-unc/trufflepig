from __future__ import annotations

import json

import pandas as pd

from trufflepig.brief import _format_therapy_bullet, _top_therapies
from trufflepig.reporting import cancer_therapy_panel_for_analysis
from trufflepig.plot_target_deep_dive import _priority_target_rows
from trufflepig.treatment_history import (
    add_population_therapy_evidence,
    parse_treatment_history,
    treatment_history_blocks_row,
    treatment_history_context,
)


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

    without_history = _top_therapies(
        pd.DataFrame([target]),
        pd.DataFrame([expression]),
        analysis={},
    )
    with_history = _top_therapies(
        pd.DataFrame([target]),
        pd.DataFrame([expression]),
        analysis=_fap_history(),
    )

    assert without_history == []
    assert len(with_history) == 1
    assert with_history[0][0]["symbol"] == "FAP"
    bullet = _format_therapy_bullet(
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
    assert "do not prioritize" in treatment_history_context(doxorubicin, analysis)

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

    assert _top_therapies(
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
