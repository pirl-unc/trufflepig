"""Public alteration-ingestion contracts shared by molecular report layers."""

import pandas as pd
import pytest

from trufflepig.alterations import (
    alteration_record_genes,
    alteration_record_passes_assay_filters,
    parse_alteration_inputs,
)


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


@pytest.mark.parametrize("filter_value", ("HighConfidence", "Tier 1"))
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


@pytest.mark.parametrize("filter_value", ("FAIL", "LowQual"))
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
