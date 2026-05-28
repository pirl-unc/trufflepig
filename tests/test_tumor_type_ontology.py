from pirlygenes.gene_sets_cancer import cancer_type_registry

from trufflepig.tumor_type_ontology import (
    tumor_type_ontology,
    tumor_type_ontology_entry,
    tumor_type_sanity_check,
)


def test_every_registry_code_has_ontology_marker_expectations():
    registry_codes = set(cancer_type_registry()["code"].dropna().astype(str))
    ontology = tumor_type_ontology()

    assert set(ontology) == registry_codes
    missing_high = [
        code for code, entry in ontology.items() if not entry.expected_high_genes
    ]
    missing_low = [
        code for code, entry in ontology.items() if not entry.expected_low_genes
    ]

    assert not missing_high
    assert not missing_low


def test_ontology_records_exact_expression_reference_and_markers_for_os():
    entry = tumor_type_ontology_entry("OS")

    assert entry is not None
    assert entry.family == "pediatric-bone"
    assert entry.expression_reference_code == "OS"
    assert entry.expression_reference_direct
    assert {"RUNX2", "COL1A1", "ALPL"} <= set(entry.expected_high_genes)
    assert {"EPCAM", "PTPRC"} <= set(entry.expected_low_genes)


def test_ontology_records_fallback_reference_and_literature_markers_for_adcc():
    entry = tumor_type_ontology_entry("ADCC")

    assert entry is not None
    assert entry.family == "salivary"
    assert entry.expression_reference_code == "HNSC"
    assert not entry.expression_reference_direct
    assert "salivary family fallback" in entry.expression_reference_reason
    assert {"MYB", "MYBL1", "NFIB"} <= set(entry.expected_high_genes)


def test_ontology_has_subtype_contrast_for_brca_her2():
    entry = tumor_type_ontology_entry("BRCA_HER2")

    assert entry is not None
    assert entry.parent_code == "BRCA"
    assert entry.expression_reference_code == "BRCA_HER2"
    assert {"ERBB2", "GRB7"} <= set(entry.expected_high_genes)
    assert {"ESR1", "PGR"} <= set(entry.expected_low_genes)


def test_tumor_type_sanity_check_reports_observed_high_and_low_markers():
    sanity = tumor_type_sanity_check(
        "BRCA_HER2",
        {
            "ERBB2": 95.0,
            "GRB7": 42.0,
            "GATA3": 8.0,
            "ESR1": 0.1,
            "PGR": 0.0,
            "PTPRC": 18.0,
        },
    )

    assert sanity["status"] == "consistent"
    assert sanity["expected_high_detected"][0]["gene"] == "ERBB2"
    assert {row["gene"] for row in sanity["expected_low_present"]} == {"PTPRC"}
    assert "2 TPM" in sanity["summary"]
