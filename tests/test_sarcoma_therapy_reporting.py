"""Clinician-facing tests for exact spindle-pattern sarcoma therapy panels."""

import pandas as pd

from trufflepig.alterations import parse_alteration_inputs
from trufflepig.brief import build_summary
from trufflepig.reporting import cancer_therapy_panel_for_analysis


def _analysis(code, *alterations):
    records = [
        parse_alteration_inputs(text)[0].public_dict() for text in alterations
    ]
    return {
        "cancer_type": code,
        "cancer_name": {
            "SARC_IMT": "Inflammatory myofibroblastic tumor",
            "SARC_DFSP": "Dermatofibrosarcoma protuberans",
            "SARC_PEC": "PEComa",
        }.get(code, code),
        "purity": {},
        "therapy_response_scores": {},
        "alteration_inputs_supplied": bool(records),
        "alteration_records": records,
    }


def _ranges(**values):
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "observed_tpm": tpm,
                "attr_tumor_tpm": tpm * 0.9,
                "attr_tumor_fraction": 0.9,
                "attr_top_compartment": "",
                "attr_top_compartment_tpm": 0.0,
                "tme_dominant": False,
                "tme_explainable": False,
            }
            for symbol, tpm in values.items()
        ]
    )


def test_imt_panel_is_exact_and_crizotinib_requires_supplied_alk_event():
    analysis = _analysis("SARC_IMT", "ALK fusion")
    panel_code, panel_subtype, panel = cancer_therapy_panel_for_analysis(
        "SARC_IMT", analysis
    )
    report = build_summary(
        analysis,
        _ranges(ALK=120.0),
        cancer_code="SARC_IMT",
        disease_state="",
    )

    assert panel_code == "SARC_IMT"
    assert panel_subtype is None
    assert set(panel["subtype"]) == {"exact_sarcoma_molecular"}
    assert "crizotinib" in set(panel["agent"])
    assert "- **ALK** — crizotinib" in report
    assert "verified activating ALK fusion/rearrangement" in report
    assert "imatinib-resistant GIST" not in report

    expression_only = build_summary(
        _analysis("SARC_IMT"),
        _ranges(ALK=120.0),
        cancer_code="SARC_IMT",
        disease_state="",
    )
    assert "- **ALK** — crizotinib" not in expression_only

    for incompatible_event in ("ALK loss", "ALK amplification"):
        incompatible = build_summary(
            _analysis("SARC_IMT", incompatible_event),
            _ranges(ALK=120.0),
            cancer_code="SARC_IMT",
            disease_state="",
        )
        assert "- **ALK** — crizotinib" not in incompatible


def test_exact_report_scope_wins_over_broad_reference_argument():
    analysis = _analysis("SARC_IMT", "ALK fusion")
    analysis.update(
        {
            "report_scope_cancer_type": "SARC_IMT",
            "reference_cancer_type": "SARC",
            "expression_reference_cancer_type": "SARC_IMT",
        }
    )

    panel_code, panel_subtype, panel = cancer_therapy_panel_for_analysis(
        "SARC", analysis
    )

    assert panel_code == "SARC_IMT"
    assert panel_subtype is None
    assert set(panel["subtype"]) == {"exact_sarcoma_molecular"}
    assert "crizotinib" in set(panel["agent"])
    assert "imatinib-resistant GIST" not in set(panel["indication"])


def test_dfsp_fusion_matches_pdgfb_and_surfaces_only_imatinib():
    analysis = _analysis("SARC_DFSP", "COL1A1-PDGFB fusion")
    _, _, panel = cancer_therapy_panel_for_analysis("SARC_DFSP", analysis)
    report = build_summary(
        analysis,
        _ranges(PDGFB=75.0),
        cancer_code="SARC_DFSP",
        disease_state="",
    )

    assert list(panel["agent"]) == ["imatinib"]
    assert "- **PDGFB** — imatinib" in report
    assert "COL1A1-PDGFB" in report
    assert "target RNA is context only" in report


def test_pecoma_uses_histology_gated_nab_sirolimus_without_expression_gate():
    analysis = _analysis("SARC_PEC")
    _, _, panel = cancer_therapy_panel_for_analysis("SARC_PEC", analysis)
    report = build_summary(
        analysis,
        _ranges(TSC1=0.0, TSC2=0.0, MTOR=0.0),
        cancer_code="SARC_PEC",
        disease_state="",
    )

    assert list(panel["agent"]) == ["nab-sirolimus (Fyarro)"]
    assert "nab-sirolimus (Fyarro)" in report
    assert "pathologically confirmed" in report
    assert "TSC1/TSC2" in report


def test_unmapped_sarcoma_child_does_not_inherit_sibling_subtype_drugs():
    panel_code, panel_subtype, panel = cancer_therapy_panel_for_analysis(
        "SARC_SFT", _analysis("SARC_SFT")
    )

    assert panel_code == "SARC_SFT"
    assert panel_subtype is None
    assert panel.empty
