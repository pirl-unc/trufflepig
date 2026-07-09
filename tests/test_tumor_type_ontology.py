from trufflepig.cancer_ontology import cancer_type_registry

from trufflepig.tumor_type_ontology import (
    tumor_type_ontology,
    tumor_type_ontology_entry,
    tumor_type_sanity_check,
)


def test_lineage_panels_lose_no_genes_to_alias_drift():
    """Every canonical lineage-panel symbol must resolve in the reference.

    Curated panels speak HGNC symbols; the symbol-keyed sample/reference speak
    the reference's symbol vocabulary. Canonicalizing through Ensembl ID
    (common.lineage_genes_by_cancer_type_canonical) must leave zero panel symbols
    that the reference can't see — otherwise a marker (e.g. CD20/MS4A1) silently
    scores zero. Guards against future alias drift re-introducing lost genes.
    """
    from trufflepig.common import lineage_genes_by_cancer_type_canonical
    from trufflepig.reference import pan_cancer_expression

    ref_symbols = set(pan_cancer_expression()["Symbol"].astype(str))
    panels = lineage_genes_by_cancer_type_canonical()
    lost = {
        code: [g for g in genes if g not in ref_symbols]
        for code, genes in panels.items()
    }
    lost = {code: miss for code, miss in lost.items() if miss}
    assert not lost, f"lineage panel symbols absent from the reference: {lost}"
    # CD20 specifically must have been canonicalized to MS4A1 (rituximab target).
    assert "MS4A1" in panels.get("DLBC", [])
    assert "CD20" not in panels.get("DLBC", [])


def test_biomarker_symbols_canonicalized_to_reference_vocabulary():
    """cancer-key-genes biomarker aliases must be recovered into the reference
    vocabulary for the marker-expectation channel (it has no Ensembl column, so
    this routes through the alias resolver). CD20 -> MS4A1 is the canary; if
    pyensembl is unavailable the resolver degrades to a no-op, so only assert the
    recovery when the raw alias was present and resolution is possible.
    """
    from trufflepig.tumor_type_ontology import _key_biomarker_genes
    from trufflepig.common import canonicalize_symbols_to_reference

    # B_ALL curates CD20 as a biomarker; it must surface as MS4A1, not CD20.
    genes = _key_biomarker_genes("B_ALL")
    if "MS4A1" in canonicalize_symbols_to_reference(["CD20"]):
        assert "MS4A1" in genes
        assert "CD20" not in genes


def test_every_registry_code_has_ontology_marker_expectations():
    reg = cancer_type_registry()
    registry_codes = set(reg["code"].dropna().astype(str))
    # Abstract grouping nodes (a code that is the parent_code of other codes, e.g.
    # CRC -> COAD/READ) aren't classifiable cohorts and have no own marker
    # expectations — the classifier scores the child cohorts. Exempt them from the
    # *coverage* requirement (but they remain valid registry codes, so an ontology
    # entry for one is NOT stale — BRCA/COAD/... are parents that do have entries).
    parent_nodes = set(reg["parent_code"].dropna().astype(str)) - {"", "nan"}
    # Aggregate / scope nodes with no classifiable cohort of their own are the
    # same case even when *childless* — exempt them from the own-marker
    # requirement too (their members carry the markers). oncoref #326 marks these
    # with ontology_level "grouping" (SARC/CRC/NET) and "evidence_scope" (the
    # literature-pooling buckets NET_NONPANCREATIC, NEN_EXTRAPULMONARY_HG). Gate on
    # that semantic level, not a hand-maintained code list.
    grouping_nodes = (
        set(reg.loc[
            reg["ontology_level"].astype(str).isin(("grouping", "evidence_scope")),
            "code",
        ].astype(str))
        if "ontology_level" in reg.columns
        else set()
    )
    exempt = parent_nodes | grouping_nodes
    ontology = tumor_type_ontology()

    # Coverage contract: every upstream registry code must have ontology marker
    # expectations curated here. Report *which* codes are uncovered so a new
    # pirlygenes code fails actionably ("here's the gap to fill") instead of as
    # an opaque set-inequality.
    uncovered = sorted((registry_codes - exempt) - set(ontology))
    assert not uncovered, f"registry codes lacking ontology entries: {uncovered}"
    # And no stale ontology entries for codes the registry dropped or renamed.
    stale = sorted(set(ontology) - registry_codes)
    assert not stale, f"ontology entries for codes absent from the registry: {stale}"
    # Abstract parent / grouping nodes get a placeholder ontology entry but no own
    # high/low markers — exempt them here too (their child cohorts carry markers).
    missing_high = [
        code
        for code, entry in ontology.items()
        if code not in exempt and not entry.expected_high_genes
    ]
    missing_low = [
        code
        for code, entry in ontology.items()
        if code not in exempt and not entry.expected_low_genes
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
