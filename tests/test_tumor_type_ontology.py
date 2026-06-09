from pirlygenes.gene_sets_cancer import cancer_type_registry

from trufflepig.tumor_type_ontology import (
    tumor_type_ontology,
    tumor_type_ontology_entry,
    tumor_type_sanity_check,
)


def test_every_registry_code_has_ontology_marker_expectations():
    registry_codes = set(cancer_type_registry()["code"].dropna().astype(str))
    ontology = tumor_type_ontology()

    # Coverage contract: every upstream registry code must have ontology marker
    # expectations curated here. Report *which* codes are uncovered so a new
    # pirlygenes code fails actionably ("here's the gap to fill") instead of as
    # an opaque set-inequality.
    uncovered = sorted(registry_codes - set(ontology))
    assert not uncovered, f"registry codes lacking ontology entries: {uncovered}"
    # And no stale ontology entries for codes the registry dropped or renamed.
    stale = sorted(set(ontology) - registry_codes)
    assert not stale, f"ontology entries for codes absent from the registry: {stale}"
    missing_high = [
        code for code, entry in ontology.items() if not entry.expected_high_genes
    ]
    missing_low = [
        code for code, entry in ontology.items() if not entry.expected_low_genes
    ]

    assert not missing_high
    assert not missing_low


def test_ontology_records_exact_expression_reference_and_markers_for_os():
    entry = tumor_type_ontology_entry("SARC_OS")

    assert entry is not None
    # OS moved from `pediatric-bone` to the lineage-only `sarcoma` family (5.12).
    assert entry.family == "sarcoma"
    assert entry.expression_reference_code == "SARC_OS"
    assert entry.expression_reference_direct
    assert {"RUNX2", "COL1A1", "ALPL"} <= set(entry.expected_high_genes)
    assert {"EPCAM", "PTPRC"} <= set(entry.expected_low_genes)


def test_ontology_records_direct_reference_and_literature_markers_for_adcc():
    # ADCC gained its own salivary-gland-carcinoma cohort in pirlygenes >=5.11,
    # so it now resolves to a direct reference rather than the HNSC fallback.
    entry = tumor_type_ontology_entry("ADCC")

    assert entry is not None
    assert entry.family == "salivary"
    assert entry.expression_reference_code == "ADCC"
    assert entry.expression_reference_direct
    assert entry.expression_reference_reason == ""
    assert {"MYB", "MYBL1", "NFIB"} <= set(entry.expected_high_genes)


def test_ontology_records_salivary_family_fallback_for_acinic():
    # ACINIC has no direct cohort of its own and still documents the
    # salivary-family fallback to HNSC.
    entry = tumor_type_ontology_entry("ACINIC")

    assert entry is not None
    assert entry.family == "salivary"
    assert entry.expression_reference_code == "HNSC"
    assert not entry.expression_reference_direct
    assert "salivary family fallback" in entry.expression_reference_reason


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
