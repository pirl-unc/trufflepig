"""Public alteration-ingestion contracts shared by molecular report layers."""

import pandas as pd
import pytest

from trufflepig.alterations import (
    alteration_record_genes,
    alteration_record_passes_assay_filters,
    molecular_evidence_for_gene,
    molecular_evidence_records,
    parse_alteration_inputs,
)
from trufflepig.fusions import FusionRecord, parse_fusion_file
from trufflepig.reporting import supplied_alteration_supports_target_row


def _record(value):
    return parse_alteration_inputs(value)[0].public_dict()


@pytest.mark.parametrize(
    ("description", "expected_genes"),
    (
        ("ETV6-NTRK3 fusion breakpoint unknown", ("ETV6", "NTRK3")),
        ("NTRK3 fusion partner unknown", ("NTRK3",)),
    ),
)
def test_unknown_event_details_do_not_negate_a_positive_call(
    description,
    expected_genes,
):
    record = _record(description)

    assert alteration_record_passes_assay_filters(record)
    assert alteration_record_genes(record) == expected_genes


@pytest.mark.parametrize(
    "description",
    (
        "NTRK3 fusion status unknown",
        "NTRK3 fusion: unknown",
        "ETV6-NTRK3 rearrangement call is unknown",
    ),
)
def test_explicit_unknown_event_outcomes_are_not_positive_calls(description):
    assert alteration_record_genes(_record(description)) == ()


@pytest.mark.parametrize("result", ("Unknown", "Unknown (assay unresolved)"))
def test_structured_unknown_results_fail_closed(tmp_path, result):
    path = tmp_path / "alteration.csv"
    pd.DataFrame(
        [{"Gene": "NTRK3", "Alteration": "fusion", "Result": result}]
    ).to_csv(path, index=False)

    record = _record(str(path))

    assert not alteration_record_passes_assay_filters(record)
    assert alteration_record_genes(record) == ()


@pytest.mark.parametrize("filter_value", ("HighConfidence", "Tier 1", "Tier 10"))
def test_generic_filter_metadata_does_not_reject_a_positive_call(
    tmp_path,
    filter_value,
):
    path = tmp_path / "alteration.csv"
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

    record = _record(str(path))

    assert record["filter_semantics"] == "generic"
    assert alteration_record_passes_assay_filters(record)
    assert alteration_record_genes(record) == ("ALK",)


@pytest.mark.parametrize("filter_value", ("FAIL", "LowQual", "LowConfidence"))
def test_explicit_generic_filter_failures_are_rejected(tmp_path, filter_value):
    path = tmp_path / "alteration.csv"
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

    record = _record(str(path))

    assert record["filter_semantics"] == "generic"
    assert not alteration_record_passes_assay_filters(record)
    assert alteration_record_genes(record) == ()


def test_vcf_filter_column_uses_strict_filter_semantics(tmp_path):
    path = tmp_path / "alteration.csv"
    pd.DataFrame(
        [
            {
                "Gene": "ALK",
                "Alteration": "rearrangement",
                "Result": "Positive",
                "FILTER": "strand_bias",
            }
        ]
    ).to_csv(path, index=False)

    record = _record(str(path))

    assert record["filter_semantics"] == "vcf"
    assert not alteration_record_passes_assay_filters(record)
    assert alteration_record_genes(record) == ()


def test_dedicated_fusion_records_share_the_molecular_evidence_contract():
    fusion = FusionRecord(
        gene_a="KIF5B",
        gene_b="RET",
        source_path="caller.tsv",
        confidence="high",
    ).public_dict()

    records = molecular_evidence_records({"fusion_records": [fusion]})

    assert len(records) == 1
    assert records[0]["alteration_type"] == "fusion"
    assert records[0]["evidence_source_types"] == ["fusion"]
    assert alteration_record_genes(records[0]) == ("KIF5B", "RET")
    assert supplied_alteration_supports_target_row(
        {
            "symbol": "RET",
            "indication": "RET fusion-positive solid tumor",
            "eligibility_basis": "tumor_agnostic_alteration",
        },
        {"fusion_records": [fusion]},
    )


def test_same_fusion_from_two_interfaces_is_one_molecular_event():
    alteration = _record("ETV6-NTRK3 fusion")
    fusion = FusionRecord(gene_a="ETV6", gene_b="NTRK3").public_dict()

    records = molecular_evidence_records(
        {
            "alteration_records": [alteration],
            "fusion_records": [fusion],
        }
    )

    assert len(records) == 1
    assert records[0]["evidence_source_types"] == ["alteration", "fusion"]
    assert alteration_record_genes(records[0]) == ("ETV6", "NTRK3")


def test_conflicting_cross_interface_fusion_calls_do_not_enable_therapy():
    negative = _record("ETV6-NTRK3 fusion not detected")
    positive = FusionRecord(gene_a="ETV6", gene_b="NTRK3").public_dict()
    analysis = {
        "alteration_records": [negative],
        "fusion_records": [positive],
    }

    records = molecular_evidence_records(analysis)

    assert len(records) == 1
    assert records[0]["evidence_conflict"] is True
    assert records[0]["evidence_source_types"] == ["alteration", "fusion"]
    assert molecular_evidence_for_gene(analysis, "NTRK3") == []
    assert not supplied_alteration_supports_target_row(
        {
            "symbol": "NTRK3",
            "indication": "NTRK gene fusion-positive solid tumor",
            "eligibility_basis": "tumor_agnostic_alteration",
        },
        analysis,
    )


@pytest.mark.parametrize("confidence", ("LowQual", "LowConfidence"))
def test_low_confidence_dedicated_fusion_does_not_enable_therapy(confidence):
    analysis = {
        "fusion_records": [
            FusionRecord(
                gene_a="TPM3",
                gene_b="ALK",
                confidence=confidence,
            ).public_dict()
        ]
    }

    assert molecular_evidence_for_gene(analysis, "ALK") == []


@pytest.mark.parametrize("status", ("Not Reportable", "Non-reportable"))
def test_explicitly_nonreportable_fusions_do_not_enable_therapy(status):
    fusion = FusionRecord(
        gene_a="TPM3",
        gene_b="ALK",
        reportable=status,
    ).public_dict()
    analysis = {"fusion_records": [fusion]}

    assert molecular_evidence_for_gene(analysis, "ALK") == []
    assert not supplied_alteration_supports_target_row(
        {
            "symbol": "ALK",
            "eligibility_basis": "alk_positive",
            "indication": "ALK-positive inflammatory myofibroblastic tumor",
        },
        analysis,
    )


def test_negative_same_gene_event_does_not_erase_distinct_positive_event():
    analysis = {
        "alteration_records": [
            _record("EGFR amplification"),
            _record("EGFR KDD not detected"),
        ]
    }

    assert supplied_alteration_supports_target_row(
        {
            "symbol": "EGFR",
            "indication": "EGFR amplification",
            "eligibility_basis": "tumor_agnostic_alteration",
        },
        analysis,
    )
    assert not supplied_alteration_supports_target_row(
        {
            "symbol": "EGFR",
            "indication": "EGFR kinase domain duplication",
            "eligibility_basis": "tumor_agnostic_alteration",
        },
        analysis,
    )


def test_conflicting_same_class_calls_fail_closed():
    analysis = {
        "alteration_records": [
            _record("EGFR amplification"),
            _record("EGFR amplification not detected"),
        ]
    }

    assert molecular_evidence_for_gene(analysis, "EGFR") == []


def test_conflicting_same_variant_calls_fail_closed():
    analysis = {
        "alteration_records": [
            _record("BRAF V600E"),
            _record("BRAF V600E not detected"),
        ]
    }

    assert molecular_evidence_for_gene(analysis, "BRAF") == []


def test_hyphenated_gene_symbol_is_not_split_without_fusion_context(tmp_path):
    path = tmp_path / "readthrough_amplification.csv"
    pd.DataFrame(
        [
            {
                "Gene": "NME1-NME2",
                "Alteration": "amplification",
                "Result": "Positive",
            }
        ]
    ).to_csv(path, index=False)

    record = parse_alteration_inputs(str(path))[0].public_dict()

    assert record["gene"] == "NME1-NME2"
    assert record["alteration_type"] == "amplification"
    assert alteration_record_genes(record) == ("NME1-NME2",)


def test_structured_nonreportable_fusion_fails_closed(tmp_path):
    path = tmp_path / "fusions.csv"
    pd.DataFrame(
        [{"gene5": "TPM3", "gene3": "ALK", "reportable": False}]
    ).to_csv(path, index=False)
    record = parse_fusion_file(path)[0]

    assert record.reportable == "False"
    assert molecular_evidence_for_gene(
        {"fusion_records": [record.public_dict()]},
        "ALK",
    ) == []
