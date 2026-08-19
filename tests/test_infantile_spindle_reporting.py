"""Clinician-facing tests for IFS/CMN/spindle molecular interpretation."""

import pandas as pd

from trufflepig.alterations import alteration_record_genes, parse_alteration_inputs
from trufflepig.brief import build_summary
import trufflepig.infantile_spindle as spindle
from trufflepig.reporting import (
    cancer_therapy_panel_for_analysis,
    supplied_alteration_supports_target_row,
)


def _record(text):
    return parse_alteration_inputs(text)[0].public_dict()


def _analysis(code, *alterations):
    records = [_record(text) for text in alterations]
    return {
        "cancer_type": code,
        "cancer_name": {
            "SARC": "Sarcoma",
            "SARC_IFS": "Infantile fibrosarcoma",
            "CMN": "Congenital mesoblastic nephroma",
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


def test_fusion_partner_is_available_to_therapy_matching():
    record = _record("ETV6-NTRK3 fusion")

    assert alteration_record_genes(record) == ("ETV6", "NTRK3")
    assert supplied_alteration_supports_target_row(
        {
            "symbol": "NTRK3",
            "indication": "NTRK gene fusion-positive solid tumor",
        },
        {"alteration_records": [record]},
    )


def test_fusion_commentary_does_not_create_a_negated_partner():
    record = _record("ALK fusion, NTRK3 not detected")

    assert alteration_record_genes(record) == ("ALK",)
    report = build_summary(
        _analysis("SARC", "ALK fusion, NTRK3 not detected"),
        _ranges(ALK=80.0, NTRK3=60.0),
        cancer_code="SARC",
        disease_state="",
    )
    assert "confirmed supplied NTRK fusion" not in report
    assert "- **NTRK3** — larotrectinib" not in report


def test_explicitly_negative_fusion_calls_are_not_confirmed():
    for alteration in (
        "ETV6-NTRK3 fusion not detected",
        "NTRK3 fusion: negative",
        "NTRK3 rearrangement - not detected",
    ):
        record = _record(alteration)

        assert alteration_record_genes(record) == ()
        report = build_summary(
            _analysis("SARC_IFS", alteration),
            _ranges(NTRK3=60.0),
            cancer_code="SARC_IFS",
            disease_state="",
        )
        assert "confirmed supplied NTRK fusion" not in report
        assert "- **NTRK3** — larotrectinib" not in report


def test_singular_and_plural_rearrangement_wording_enable_ntrk_therapy():
    for alteration in (
        "ETV6-NTRK3 rearrangement",
        "ETV6-NTRK3 fusions",
        "ETV6-NTRK3 rearrangements",
    ):
        analysis = _analysis("SARC_IFS", alteration)

        assert analysis["alteration_records"][0]["alteration_type"] == "fusion"
        report = build_summary(
            analysis,
            _ranges(NTRK3=20.0),
            cancer_code="SARC_IFS",
            disease_state="",
        )

        assert "confirmed supplied NTRK fusion involving NTRK3" in report
        assert "- **NTRK3** — larotrectinib" in report


def test_ifs_summary_prioritizes_larotrectinib_for_confirmed_ntrk_fusion():
    analysis = _analysis("SARC_IFS", "ETV6-NTRK3 fusion")

    report = build_summary(
        analysis,
        _ranges(NTRK3=20.0),
        cancer_code="SARC_IFS",
        disease_state="",
    )

    assert "IFS is a heterogeneous infantile MAPK-rearranged spindle-cell tumor" in report
    assert "confirmed supplied NTRK fusion involving NTRK3" in report
    assert "- **NTRK3** — larotrectinib" in report
    assert "94% objective response" in report
    assert "target RNA is context only" in report


def test_ntrk_expression_alone_does_not_create_a_drug_shortlist():
    analysis = _analysis("SARC_IFS")

    report = build_summary(
        analysis,
        _ranges(NTRK1=80.0, NTRK2=30.0, NTRK3=120.0),
        cancer_code="SARC_IFS",
        disease_state="",
    )

    assert "larotrectinib" in report  # conditional treatment-relevance guidance
    assert "- **NTRK" not in report
    assert "structural-variant testing that covers NTRK1/2/3 fusions" in report


def test_braf_internal_deletion_prompts_workup_without_inventing_a_drug():
    analysis = _analysis("SARC_IFS", "BRAF internal deletion")

    report = build_summary(
        analysis,
        _ranges(BRAF=80.0),
        cancer_code="SARC_IFS",
        disease_state="",
    )

    assert "BRAF rearrangement/internal deletion" in report
    assert "- **BRAF**" not in report
    assert "dabrafenib" not in report.lower()
    assert "trametinib" not in report.lower()


def test_cmn_egfr_kdd_surfaces_case_level_egfr_tki_review_with_site_caveat():
    analysis = _analysis("CMN", "EGFR kinase domain duplication (KDD)")

    report = build_summary(
        analysis,
        _ranges(EGFR=583.0),
        cancer_code="CMN",
        disease_state="",
    )

    assert "CMN is a kidney-site-qualified pathologic diagnosis" in report
    assert "neither event alone establishes the diagnosis" in report
    assert (
        "- **EGFR** — EGFR TKI review (afatinib; osimertinib secondary)" in report
    )
    assert "case-level evidence only" in report
    assert "target RNA is context only" in report


def test_egfr_expression_alone_does_not_create_an_egfr_tki_shortlist():
    analysis = _analysis("CMN")

    report = build_summary(
        analysis,
        _ranges(EGFR=583.0),
        cancer_code="CMN",
        disease_state="",
    )

    assert "EGFR KDD/ITD" in report
    assert "- **EGFR**" not in report


def test_exact_spindle_entities_have_complete_conditional_target_panels():
    for code in ("SARC_IFS", "CMN", "SARC_NTRK_SPINDLE"):
        panel_code, panel_subtype, panel = cancer_therapy_panel_for_analysis(
            code,
            _analysis(code),
        )
        assert panel_code == code
        assert panel_subtype is None
        assert {"NTRK1", "NTRK2", "NTRK3"}.issubset(set(panel["symbol"]))
        assert {"larotrectinib", "entrectinib", "repotrectinib"}.issubset(
            set(panel["agent"])
        )
        assert set(panel["subtype"]) == {"infantile_spindle_molecular"}
        assert panel["requires_supplied_alteration"].fillna(False).all()


def test_broad_sarcoma_egfr_kdd_is_context_not_a_cmn_relabel():
    analysis = _analysis("SARC", "EGFR KDD")

    report = build_summary(
        analysis,
        _ranges(EGFR=300.0),
        cancer_code="SARC",
        disease_state="",
    )

    assert "raises an infantile spindle/CMN differential" in report
    assert "do not establish IFS or CMN" in report
    assert "- **EGFR**" in report


def test_full_report_driver_spectrum_preserves_entity_and_frequency(monkeypatch):
    spectra = {
        "SARC_IFS": (
            {
                "driver_event": "ETV6-NTRK3",
                "frequency": "18/27 (67%)",
                "relationship": "common_observed",
                "evidence_source": "PMID:29915264 Supplementary Data 1",
            },
        ),
        "CMN": (
            {
                "driver_event": "EGFR kinase-domain ITD",
                "frequency": "43/80 (54%)",
                "relationship": "common_observed",
                "evidence_source": "PMID:29915264 Supplementary Data 1",
            },
        ),
    }
    monkeypatch.setattr(
        spindle,
        "_upstream_driver_spectrum",
        lambda code: spectra.get(code, ()),
    )

    exact = spindle.infantile_spindle_driver_spectrum_markdown(
        "CMN", _analysis("CMN")
    )
    differential = spindle.infantile_spindle_driver_spectrum_markdown(
        "SARC", _analysis("SARC", "EGFR KDD")
    )

    assert "| CMN | EGFR kinase-domain ITD | 43/80 (54%) |" in exact
    assert "SARC_IFS" not in exact
    assert "| SARC_IFS | ETV6-NTRK3 | 18/27 (67%) |" in differential
    assert "| CMN | EGFR kinase-domain ITD | 43/80 (54%) |" in differential
    assert "not proof of diagnosis in this sample" in differential
