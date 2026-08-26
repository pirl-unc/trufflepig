"""Public variant-ingestion contracts shared by report layers."""

import json

import pandas as pd
import pytest

from trufflepig.variants import (
    VariantCoordinate,
    VariantRecord,
    classify_variant_type,
    normalize_genome_build,
    normalize_variant_record,
    parse_variant_file,
    parse_variant_inputs,
    validate_variant_genome_builds,
    variant_record_genes,
    variant_record_passes_assay_filters,
    variant_evidence_for_gene,
    variant_evidence_records,
)
from trufflepig.fusions import FusionRecord, parse_fusion_file
from trufflepig.reporting import supplied_variant_supports_target_row


def _record(value):
    return parse_variant_inputs(value)[0].public_dict()


@pytest.mark.parametrize(
    ("value", "expected"),
    (("GRCh37", "GRCh37"), ("hg19", "GRCh37"), ("hg38", "GRCh38")),
)
def test_genome_build_aliases_have_one_public_representation(value, expected):
    assert normalize_genome_build(value) == expected


def test_unknown_genome_build_is_rejected():
    with pytest.raises(ValueError, match="Unsupported genome build"):
        normalize_genome_build("T2T-CHM13")


def test_symbolic_variant_is_explicitly_assembly_neutral():
    record = parse_variant_inputs("EGFR KDD", genome_build="hg38")[0]

    assert record.representation == "symbolic"
    assert record.genome_build == ""
    assert record.coordinates == ()
    assert record.source_format == "inline"
    assert validate_variant_genome_builds([record]) == {
        "genome_build": "",
        "expected_genome_build": "",
        "coordinate_records": 0,
        "assembly_neutral_records": 1,
        "coordinate_records_without_build": 0,
    }

    programmatic = VariantRecord(gene="EGFR", genome_build="GRCh38")
    assert programmatic.genome_build == ""


def test_coordinate_variant_has_typed_public_provenance():
    record = VariantRecord(
        gene="BRAF",
        genes=("BRAF",),
        variant="V600E",
        variant_type="mutation",
        source_format="csv",
        genome_build="hg38",
        ensembl_release=114,
        coordinates=(
            VariantCoordinate(contig="chr7", start=140753336, ref="a", alt="t"),
        ),
    )

    assert record.representation == "coordinate"
    assert record.genes == ("BRAF",)
    assert record.genome_build == "GRCh38"
    assert record.public_dict()["coordinates"] == [
        {
            "contig": "chr7",
            "start": 140753336,
            "end": 140753336,
            "ref": "A",
            "alt": "T",
            "role": "",
        }
    ]
    assert normalize_variant_record(record) == record.public_dict()


@pytest.mark.parametrize(
    "coordinate",
    (
        {"contig": "7", "start": 10.5},
        {"contig": "7", "start": 10, "end": 20.9},
    ),
)
def test_fractional_coordinate_endpoints_are_rejected(coordinate):
    with pytest.raises(ValueError, match="must be a positive integer"):
        VariantCoordinate(**coordinate)


def test_coordinate_variant_requires_a_build():
    record = VariantRecord(
        gene="BRAF",
        coordinates=(VariantCoordinate("7", 140753336),),
    )

    with pytest.raises(ValueError, match="require an explicit genome build"):
        validate_variant_genome_builds([record])


def test_mixed_coordinate_builds_are_rejected():
    records = [
        VariantRecord(
            gene="BRAF",
            genome_build=build,
            coordinates=(VariantCoordinate("7", 1),),
        )
        for build in ("GRCh37", "GRCh38")
    ]

    with pytest.raises(ValueError, match="contradictory genome builds"):
        validate_variant_genome_builds(records)


def test_expected_genome_build_must_match_coordinates():
    record = VariantRecord(
        gene="BRAF",
        genome_build="GRCh38",
        coordinates=(VariantCoordinate("7", 1),),
    )

    with pytest.raises(ValueError, match="does not match expected build"):
        validate_variant_genome_builds([record], expected_build="GRCh37")


def test_generic_coordinate_table_preserves_source_and_build(tmp_path):
    path = tmp_path / "variants.tsv"
    pd.DataFrame(
        [
            {
                "Gene": "BRAF",
                "Variant": "V600E",
                "Chromosome": "7",
                "Position": 140753336,
                "Ref": "A",
                "Alt": "T",
                "NCBI_Build": "GRCh38",
                "Ensembl_Release": 114,
                "Caller_Version": "example 1.0",
            }
        ]
    ).to_csv(path, sep="\t", index=False)

    record = parse_variant_file(path)[0]

    assert record.source_format == "tsv"
    assert record.caller_version == "example 1.0"
    assert record.genome_build == "GRCh38"
    assert record.ensembl_release == 114
    assert record.representation == "coordinate"
    assert record.coordinates[0].start == 140753336


def test_requested_build_fills_missing_coordinate_table_build(tmp_path):
    path = tmp_path / "variants.csv"
    pd.DataFrame(
        [{"Gene": "BRAF", "Variant": "V600E", "Chr": "7", "Pos": 140753336}]
    ).to_csv(path, index=False)

    record = parse_variant_file(path, genome_build="hg19")[0]

    assert record.genome_build == "GRCh37"
    assert record.source_format == "csv"


def test_requested_build_cannot_override_table_build(tmp_path):
    path = tmp_path / "variants.csv"
    pd.DataFrame(
        [
            {
                "Gene": "BRAF",
                "Variant": "V600E",
                "Chr": "7",
                "Pos": 140753336,
                "Assembly": "GRCh38",
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="conflicts with the requested build"):
        parse_variant_file(path, genome_build="GRCh37")


def test_structured_fusion_retains_both_participating_genes(tmp_path):
    path = tmp_path / "variants.csv"
    pd.DataFrame(
        [{"Gene": "ETV6::NTRK3", "Variant": "fusion", "Result": "Positive"}]
    ).to_csv(path, index=False)

    record = parse_variant_file(path)[0]

    assert record.representation == "symbolic"
    assert record.genes == ("ETV6", "NTRK3")


def test_empty_structured_variant_file_fails_closed(tmp_path):
    path = tmp_path / "variants.csv"
    pd.DataFrame([{"Comment": "no variant records"}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="No recognizable variant records"):
        parse_variant_file(path)


def test_typed_variant_json_round_trips_without_losing_provenance(tmp_path):
    original = VariantRecord(
        gene="BRAF",
        genes=("BRAF",),
        variant="V600E",
        variant_type="mutation",
        source_path="caller-output.csv",
        row_index=7,
        source_format="csv",
        caller_version="caller 2.1",
        genome_build="GRCh38",
        ensembl_release=114,
        coordinates=(
            VariantCoordinate("7", 140753336, ref="A", alt="T"),
        ),
    )
    path = tmp_path / "variant.json"
    path.write_text(json.dumps(original.public_dict()))

    parsed = parse_variant_file(path)

    assert len(parsed) == 1
    assert parsed[0].public_dict() == original.public_dict()


def test_typed_variant_json_rejects_fractional_coordinates(tmp_path):
    path = tmp_path / "variant.json"
    path.write_text(
        json.dumps(
            {
                "gene": "BRAF",
                "variant": "V600E",
                "variant_type": "mutation",
                "representation": "coordinate",
                "genome_build": "GRCh38",
                "coordinates": [{"contig": "7", "start": 140753336.5}],
            }
        )
    )

    with pytest.raises(ValueError, match="must be a positive integer"):
        parse_variant_file(path)


@pytest.mark.parametrize("suffix", ("vcf", "vcf.gz", "maf", "maf.gz"))
def test_standard_variant_files_fail_closed_until_their_adapter_exists(
    tmp_path,
    suffix,
):
    path = tmp_path / f"variants.{suffix}"
    path.write_text("Gene\tVariant\nBRAF\tV600E\n")

    with pytest.raises(ValueError, match="source-specific adapter"):
        parse_variant_file(path)


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

    assert variant_record_passes_assay_filters(record)
    assert variant_record_genes(record) == expected_genes


@pytest.mark.parametrize(
    "description",
    (
        "NTRK3 fusion status unknown",
        "NTRK3 fusion: unknown",
        "ETV6-NTRK3 rearrangement call is unknown",
    ),
)
def test_explicit_unknown_event_outcomes_are_not_positive_calls(description):
    assert variant_record_genes(_record(description)) == ()


@pytest.mark.parametrize("result", ("Unknown", "Unknown (assay unresolved)"))
def test_structured_unknown_results_fail_closed(tmp_path, result):
    path = tmp_path / "alteration.csv"
    pd.DataFrame(
        [{"Gene": "NTRK3", "Alteration": "fusion", "Result": result}]
    ).to_csv(path, index=False)

    record = _record(str(path))

    assert not variant_record_passes_assay_filters(record)
    assert variant_record_genes(record) == ()


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
    assert record["variant"] == "rearrangement"
    assert variant_record_passes_assay_filters(record)
    assert variant_record_genes(record) == ("ALK",)


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
    assert not variant_record_passes_assay_filters(record)
    assert variant_record_genes(record) == ()


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
    assert not variant_record_passes_assay_filters(record)
    assert variant_record_genes(record) == ()


def test_dedicated_fusion_records_share_the_variant_evidence_contract():
    fusion = FusionRecord(
        gene_a="KIF5B",
        gene_b="RET",
        source_path="caller.tsv",
        confidence="high",
    ).public_dict()

    records = variant_evidence_records({"fusion_records": [fusion]})

    assert len(records) == 1
    assert records[0]["variant_type"] == "fusion"
    assert records[0]["genes"] == ["KIF5B", "RET"]
    assert records[0]["source_format"] == "fusion"
    assert records[0]["evidence_source_types"] == ["fusion"]
    assert variant_record_genes(records[0]) == ("KIF5B", "RET")
    assert supplied_variant_supports_target_row(
        {
            "symbol": "RET",
            "indication": "RET fusion-positive solid tumor",
            "eligibility_basis": "tumor_agnostic_alteration",
        },
        {"fusion_records": [fusion]},
    )


def test_distinct_coordinate_variants_are_not_merged_by_gene_label():
    records = [
        VariantRecord(
            gene="BRAF",
            variant="mutation",
            variant_type="mutation",
            genome_build="GRCh38",
            coordinates=(VariantCoordinate("7", position, ref="A", alt="T"),),
        ).public_dict()
        for position in (140753336, 140753337)
    ]

    normalized = variant_evidence_records({"variant_records": records})

    assert len(normalized) == 2


def test_conflicting_coordinate_calls_fail_closed_for_therapy():
    coordinate = VariantCoordinate("7", 140753336, ref="A", alt="T")
    analysis = {
        "variant_records": [
            VariantRecord(
                gene="BRAF",
                variant=variant,
                variant_type="mutation",
                genome_build="GRCh38",
                coordinates=(coordinate,),
            ).public_dict()
            for variant in ("BRAF V600E", "BRAF V600E not detected")
        ]
    }

    records = variant_evidence_records(analysis)

    assert len(records) == 1
    assert records[0]["evidence_conflict"] is True
    assert variant_evidence_for_gene(analysis, "BRAF") == []
    assert not supplied_variant_supports_target_row(
        {
            "symbol": "BRAF",
            "indication": "BRAF V600E solid tumor",
            "eligibility_basis": "tumor_agnostic_alteration",
        },
        analysis,
    )


def test_coordinate_reconciliation_canonicalizes_primary_contig_aliases():
    analysis = {
        "variant_records": [
            VariantRecord(
                gene="BRAF",
                variant=variant,
                variant_type="mutation",
                genome_build="GRCh38",
                coordinates=(
                    VariantCoordinate(contig, 140753336, ref="A", alt="T"),
                ),
            ).public_dict()
            for contig, variant in (
                ("7", "BRAF V600E"),
                ("chr7", "BRAF V600E not detected"),
            )
        ]
    }

    records = variant_evidence_records(analysis)

    assert len(records) == 1
    assert records[0]["evidence_conflict"] is True
    assert variant_evidence_for_gene(analysis, "BRAF") == []


def test_structured_fusion_participants_drive_target_matching_without_prose_pair():
    record = VariantRecord(
        gene="ETV6",
        genes=("ETV6", "NTRK3"),
        variant="verified rearrangement",
        variant_type="fusion",
    ).public_dict()
    analysis = {"variant_records": [record]}

    assert variant_record_genes(record) == ("ETV6", "NTRK3")
    matching = variant_evidence_for_gene(analysis, "NTRK3")
    assert len(matching) == 1
    assert matching[0]["genes"] == ["ETV6", "NTRK3"]
    assert supplied_variant_supports_target_row(
        {
            "symbol": "NTRK3",
            "indication": "NTRK gene fusion-positive solid tumor",
            "eligibility_basis": "tumor_agnostic_alteration",
        },
        analysis,
    )


def test_same_fusion_from_two_interfaces_is_one_variant():
    alteration = _record("ETV6-NTRK3 fusion")
    fusion = FusionRecord(gene_a="ETV6", gene_b="NTRK3").public_dict()

    records = variant_evidence_records(
        {
            "variant_records": [alteration],
            "fusion_records": [fusion],
        }
    )

    assert len(records) == 1
    assert records[0]["evidence_source_types"] == ["variant", "fusion"]
    assert variant_record_genes(records[0]) == ("ETV6", "NTRK3")


def test_conflicting_cross_interface_fusion_calls_do_not_enable_therapy():
    negative = _record("ETV6-NTRK3 fusion not detected")
    positive = FusionRecord(gene_a="ETV6", gene_b="NTRK3").public_dict()
    analysis = {
        "variant_records": [negative],
        "fusion_records": [positive],
    }

    records = variant_evidence_records(analysis)

    assert len(records) == 1
    assert records[0]["evidence_conflict"] is True
    assert records[0]["evidence_source_types"] == ["variant", "fusion"]
    assert variant_evidence_for_gene(analysis, "NTRK3") == []
    assert not supplied_variant_supports_target_row(
        {
            "symbol": "NTRK3",
            "indication": "NTRK gene fusion-positive solid tumor",
            "eligibility_basis": "tumor_agnostic_alteration",
        },
        analysis,
    )


def test_coordinate_fusion_reconciles_with_dedicated_fusion_by_pair():
    negative = VariantRecord(
        gene="ETV6",
        genes=("ETV6", "NTRK3"),
        variant="ETV6-NTRK3 fusion not detected",
        variant_type="fusion",
        result_status="not detected",
        genome_build="GRCh38",
        coordinates=(
            VariantCoordinate("chr12", 12000000, role="5p_breakpoint"),
            VariantCoordinate("15", 88000000, role="3p_breakpoint"),
        ),
    ).public_dict()
    positive = FusionRecord(gene_a="ETV6", gene_b="NTRK3").public_dict()
    analysis = {
        "variant_records": [negative],
        "fusion_records": [positive],
    }

    records = variant_evidence_records(analysis)

    assert len(records) == 1
    assert records[0]["evidence_conflict"] is True
    assert records[0]["evidence_source_types"] == ["variant", "fusion"]
    assert variant_evidence_for_gene(analysis, "NTRK3") == []
    assert not supplied_variant_supports_target_row(
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

    assert variant_evidence_for_gene(analysis, "ALK") == []


@pytest.mark.parametrize("status", ("Not Reportable", "Non-reportable"))
def test_explicitly_nonreportable_fusions_do_not_enable_therapy(status):
    fusion = FusionRecord(
        gene_a="TPM3",
        gene_b="ALK",
        reportable=status,
    ).public_dict()
    analysis = {"fusion_records": [fusion]}

    assert variant_evidence_for_gene(analysis, "ALK") == []
    assert not supplied_variant_supports_target_row(
        {
            "symbol": "ALK",
            "eligibility_basis": "alk_positive",
            "indication": "ALK-positive inflammatory myofibroblastic tumor",
        },
        analysis,
    )


def test_negative_same_gene_event_does_not_erase_distinct_positive_event():
    analysis = {
        "variant_records": [
            _record("EGFR amplification"),
            _record("EGFR KDD not detected"),
        ]
    }

    assert supplied_variant_supports_target_row(
        {
            "symbol": "EGFR",
            "indication": "EGFR amplification",
            "eligibility_basis": "tumor_agnostic_alteration",
        },
        analysis,
    )
    assert not supplied_variant_supports_target_row(
        {
            "symbol": "EGFR",
            "indication": "EGFR kinase domain duplication",
            "eligibility_basis": "tumor_agnostic_alteration",
        },
        analysis,
    )


def test_conflicting_same_class_calls_fail_closed():
    analysis = {
        "variant_records": [
            _record("EGFR amplification"),
            _record("EGFR amplification not detected"),
        ]
    }

    assert variant_evidence_for_gene(analysis, "EGFR") == []


def test_conflicting_same_variant_calls_fail_closed():
    analysis = {
        "variant_records": [
            _record("BRAF V600E"),
            _record("BRAF V600E not detected"),
        ]
    }

    assert variant_evidence_for_gene(analysis, "BRAF") == []


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

    record = parse_variant_inputs(str(path))[0].public_dict()

    assert record["gene"] == "NME1-NME2"
    assert record["variant_type"] == "amplification"
    assert variant_record_genes(record) == ("NME1-NME2",)


def test_structured_nonreportable_fusion_fails_closed(tmp_path):
    path = tmp_path / "fusions.csv"
    pd.DataFrame(
        [{"gene5": "TPM3", "gene3": "ALK", "reportable": False}]
    ).to_csv(path, index=False)
    record = parse_fusion_file(path)[0]

    assert record.reportable == "False"
    assert variant_evidence_for_gene(
        {"fusion_records": [record.public_dict()]},
        "ALK",
    ) == []


def test_sample_level_msi_state_is_not_classified_as_one_variant():
    assert classify_variant_type("MSI-H / dMMR") == "unknown"
    with pytest.raises(ValueError, match="sample-level genomic state"):
        parse_variant_inputs("MSI-H / dMMR")


def test_sample_level_msi_table_row_is_not_classified_as_one_variant(tmp_path):
    path = tmp_path / "states.tsv"
    pd.DataFrame([{"Gene": "MSI", "Variant": "MSI-H"}]).to_csv(
        path,
        sep="\t",
        index=False,
    )

    with pytest.raises(ValueError, match="sample-level genomic state"):
        parse_variant_inputs(path)


def test_new_empty_variant_stream_does_not_fall_back_to_stale_legacy_records():
    analysis = {
        "variant_records": [],
        "alteration_records": [
            {
                "gene": "ALK",
                "alteration": "ALK fusion",
                "alteration_type": "fusion",
            }
        ],
    }

    assert variant_evidence_for_gene(analysis, "ALK") == []


def test_legacy_alteration_parser_is_a_narrow_variant_alias():
    from trufflepig.alterations import parse_alteration_inputs

    record = parse_alteration_inputs("EGFR KDD")[0]

    assert record.variant == "EGFR KDD"
    assert record.variant_type == "kdd"
    assert record.alteration == record.variant
    assert record.alteration_type == record.variant_type
