"""Clinician-facing tests for exact spindle-pattern sarcoma therapy panels."""

import json

import pandas as pd

from trufflepig.alterations import parse_alteration_inputs
from trufflepig.brief import build_summary
from trufflepig.fusions import FusionRecord
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
    assert set(panel["cancer_code"]) == {"SARC_IMT"}
    assert panel["subtype"].isna().all()
    assert "crizotinib" in set(panel["agent"])
    assert panel["agent"].eq("crizotinib").sum() == 1
    assert "tumor_agnostic_alteration" in set(panel["eligibility_basis"])
    assert "- **ALK** — crizotinib" in report
    assert "validated ALK IHC or a molecular method such as FISH" in report
    assert "imatinib-resistant GIST" not in report

    expression_only = build_summary(
        _analysis("SARC_IMT"),
        _ranges(ALK=120.0),
        cancer_code="SARC_IMT",
        disease_state="",
    )
    assert "- **ALK** — crizotinib" not in expression_only
    assert "- **NTRK" not in expression_only

    for incompatible_event in (
        "ALK loss",
        "ALK amplification",
        "ALK fusion failed",
        "ALK rearrangement inconclusive",
    ):
        incompatible = build_summary(
            _analysis("SARC_IMT", incompatible_event),
            _ranges(ALK=120.0),
            cancer_code="SARC_IMT",
            disease_state="",
        )
        assert "- **ALK** — crizotinib" not in incompatible


def test_dedicated_alk_fusion_input_enables_exact_imt_therapy():
    analysis = _analysis("SARC_IMT")
    analysis["fusion_records"] = [
        FusionRecord(
            gene_a="TPM3",
            gene_b="ALK",
            source_path="arriba.tsv",
            confidence="high",
        ).public_dict()
    ]

    report = build_summary(
        analysis,
        _ranges(ALK=120.0),
        cancer_code="SARC_IMT",
        disease_state="",
    )

    assert "- **ALK** — crizotinib" in report


def test_imt_singular_and_plural_rearrangement_wording_enable_crizotinib():
    for alteration in ("ALK rearranged", "ALK rearrangements", "ALK translocations"):
        analysis = _analysis("SARC_IMT", alteration)

        assert analysis["alteration_records"][0]["alteration_type"] == "fusion"
        report = build_summary(
            analysis,
            _ranges(ALK=120.0),
            cancer_code="SARC_IMT",
            disease_state="",
        )

        assert "- **ALK** — crizotinib" in report


def test_structured_negative_alk_result_never_enables_crizotinib(tmp_path):
    cases = (
        ("csv", ",", "Result", "Negative (below detection limit)"),
        ("tsv", "\t", "Status", "Not detected - insufficient support"),
        ("csv", ",", "Result", "Fusion Not Detected"),
        ("tsv", "\t", "Status", "Result: Negative"),
    )
    for suffix, separator, result_column, result_value in cases:
        path = tmp_path / f"alk_result.{suffix}"
        pd.DataFrame(
            [
                {
                    "Gene": "ALK",
                    "Alteration": "rearrangement",
                    result_column: result_value,
                }
            ]
        ).to_csv(path, sep=separator, index=False)
        record = parse_alteration_inputs(str(path))[0].public_dict()
        analysis = _analysis("SARC_IMT")
        analysis.update(
            alteration_inputs_supplied=True,
            alteration_records=[record],
        )

        report = build_summary(
            analysis,
            _ranges(ALK=120.0),
            cancer_code="SARC_IMT",
            disease_state="",
        )

        assert record["result_status"] == result_value
        assert "- **ALK** — crizotinib" not in report


def test_machine_readable_negative_alk_results_never_enable_crizotinib(tmp_path):
    non_positive_results = (
        False,
        0,
        "No",
        "NEG",
        "Inconclusive",
        "FAIL",
        "Failed",
        "Failure",
        "QNS",
        "Pending",
        "Canceled",
        "Cancelled",
        "Not assessed",
        "No call",
        "VUS",
        "Benign",
        "Likely benign",
    )
    for index, result_value in enumerate(non_positive_results):
        path = tmp_path / f"alk_machine_result_{index}.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "Gene": "ALK",
                        "Alteration": "rearrangement",
                        "Result": result_value,
                    }
                ]
            )
        )
        record = parse_alteration_inputs(str(path))[0].public_dict()
        analysis = _analysis("SARC_IMT")
        analysis.update(
            alteration_inputs_supplied=True,
            alteration_records=[record],
        )

        report = build_summary(
            analysis,
            _ranges(ALK=120.0),
            cancer_code="SARC_IMT",
            disease_state="",
        )

        assert record["result_status"] == str(result_value)
        assert "- **ALK** — crizotinib" not in report


def test_structured_pass_alk_results_enable_crizotinib(tmp_path):
    for index, result_value in enumerate(("PASS", "PASSED")):
        path = tmp_path / f"alk_pass_result_{index}.csv"
        pd.DataFrame(
            [
                {
                    "Gene": "ALK",
                    "Alteration": "rearrangement",
                    "Status": result_value,
                }
            ]
        ).to_csv(path, index=False)
        record = parse_alteration_inputs(str(path))[0].public_dict()
        analysis = _analysis("SARC_IMT")
        analysis.update(
            alteration_inputs_supplied=True,
            alteration_records=[record],
        )

        report = build_summary(
            analysis,
            _ranges(ALK=120.0),
            cancer_code="SARC_IMT",
            disease_state="",
        )

        assert record["result_status"] == result_value
        assert "- **ALK** — crizotinib" in report


def test_classification_status_does_not_suppress_verified_alk_event(tmp_path):
    for index, status in enumerate(("Pathogenic", "Somatic")):
        path = tmp_path / f"alk_classification_status_{index}.csv"
        pd.DataFrame(
            [
                {
                    "Gene": "ALK",
                    "Alteration": "rearrangement",
                    "Status": status,
                }
            ]
        ).to_csv(path, index=False)
        record = parse_alteration_inputs(str(path))[0].public_dict()
        analysis = _analysis("SARC_IMT")
        analysis.update(
            alteration_inputs_supplied=True,
            alteration_records=[record],
        )

        report = build_summary(
            analysis,
            _ranges(ALK=120.0),
            cancer_code="SARC_IMT",
            disease_state="",
        )

        assert record["result_status"] == status
        assert "- **ALK** — crizotinib" in report


def test_structured_filter_status_controls_alk_therapy_evidence(tmp_path):
    for index, (filter_status, expected) in enumerate(
        (("PASS", True), (".", True), ("FAIL", False), ("LowQual", False))
    ):
        path = tmp_path / f"alk_filter_status_{index}.csv"
        pd.DataFrame(
            [
                {
                    "Gene": "ALK",
                    "Alteration": "rearrangement",
                    "Result": "Positive",
                    "FILTER": filter_status,
                }
            ]
        ).to_csv(path, index=False)
        record = parse_alteration_inputs(str(path))[0].public_dict()
        analysis = _analysis("SARC_IMT")
        analysis.update(
            alteration_inputs_supplied=True,
            alteration_records=[record],
        )

        report = build_summary(
            analysis,
            _ranges(ALK=120.0),
            cancer_code="SARC_IMT",
            disease_state="",
        )

        assert record["filter_status"] == filter_status
        assert ("- **ALK** — crizotinib" in report) is expected


def test_generic_filter_confidence_does_not_suppress_alk_therapy_evidence(tmp_path):
    for index, filter_value in enumerate(("HighConfidence", "Tier 1")):
        path = tmp_path / f"alk_confidence_filter_{index}.csv"
        pd.DataFrame(
            [
                {
                    "Gene": "ALK",
                    "Alteration": "rearrangement",
                    "Result": "Positive",
                    "Filter": filter_value,
                }
            ]
        ).to_csv(path, index=False)
        record = parse_alteration_inputs(str(path))[0].public_dict()
        analysis = _analysis("SARC_IMT")
        analysis.update(
            alteration_inputs_supplied=True,
            alteration_records=[record],
        )

        report = build_summary(
            analysis,
            _ranges(ALK=120.0),
            cancer_code="SARC_IMT",
            disease_state="",
        )

        assert record["filter_semantics"] == "generic"
        assert "- **ALK** — crizotinib" in report


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
    assert set(panel["cancer_code"]) == {"SARC_IMT"}
    assert panel["subtype"].isna().all()
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

    histology_only = build_summary(
        _analysis("SARC_DFSP"),
        _ranges(PDGFB=75.0),
        cancer_code="SARC_DFSP",
        disease_state="",
    )
    assert "- **PDGFB** — imatinib" in histology_only


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


def test_broad_sarcoma_expression_subtype_does_not_unlock_pecoma_indication():
    analysis = _analysis("SARC")
    analysis["candidate_trace"] = [
        {"code": "SARC", "winning_subtype": "SARC_PEC"},
    ]

    panel_code, panel_subtype, panel = cancer_therapy_panel_for_analysis(
        "SARC",
        analysis,
    )
    report = build_summary(
        analysis,
        _ranges(TSC1=0.0, TSC2=0.0, MTOR=0.0),
        cancer_code="SARC",
        disease_state="",
    )

    assert panel_code == "SARC"
    assert panel_subtype is None
    assert panel.empty
    assert "nab-sirolimus" not in report
    assert "pecoma-specific therapy evidence" not in report.lower()

    molecular_analysis = _analysis("SARC", "EGFR kinase domain duplication / KDD")
    molecular_analysis["candidate_trace"] = analysis["candidate_trace"]
    _, _, molecular_panel = cancer_therapy_panel_for_analysis(
        "SARC",
        molecular_analysis,
    )
    assert set(molecular_panel["symbol"]) == {"EGFR"}
    assert "nab-sirolimus (Fyarro)" not in set(molecular_panel["agent"])


def test_unmapped_sarcoma_child_does_not_inherit_sibling_subtype_drugs():
    panel_code, panel_subtype, panel = cancer_therapy_panel_for_analysis(
        "SARC_SFT", _analysis("SARC_SFT")
    )

    assert panel_code == "SARC_SFT"
    assert panel_subtype is None
    assert panel.empty


def test_exact_mpnst_uses_its_curated_parent_subtype_panel():
    panel_code, panel_subtype, panel = cancer_therapy_panel_for_analysis(
        "SARC_MPNST",
        _analysis("SARC_MPNST"),
    )

    assert panel_code == "SARC"
    assert panel_subtype == "MPNST"
    assert len(panel) == 5
    assert set(panel["subtype"]) == {"MPNST"}
    assert set(panel["agent"]) == {
        "BET inhibitors (trials)",
        "SHP2 inhibitors (trials)",
        "selumetinib",
        "tazemetostat",
        "trametinib",
    }
