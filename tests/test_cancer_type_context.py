import pytest
from pirlygenes.gene_sets_cancer import cancer_type_registry

import trufflepig.analyze.cancer_type_context as context_module
from trufflepig import reference as reference_module
from trufflepig.analyze import (
    ExpressionReferenceRecord,
    cancer_type_context_from_analysis,
    effective_expression_reference,
    expression_reference_options,
)


def test_context_keeps_prad_as_single_active_label_when_no_refined_label_exists():
    context = cancer_type_context_from_analysis(
        {
            "cancer_type": "PRAD",
            "reference_cancer_type": "PRAD",
            "cancer_type_source": "auto-detected",
        }
    )

    assert context.code_for("report") == "PRAD"
    assert context.code_for("reference") == "PRAD"
    assert context.code_for("expression") == "PRAD"
    assert context.relationship == "same"
    assert not context.uses_distinct_reference


def test_context_names_direct_reference_when_same_code_is_not_pan_cancer():
    context = cancer_type_context_from_analysis(
        {
            "cancer_type": "WILMS",
            "reference_cancer_type": "WILMS",
            "cancer_type_source": "user-specified",
        }
    )

    lines = "\n".join(context.markdown_lines())

    assert context.best_expression_source_kind == "deconvolved_tumor_reference"
    assert "direct expression reference is available" in lines
    assert "Broad reference context" not in lines


def test_context_exposes_refined_child_and_parent_reference():
    context = cancer_type_context_from_analysis(
        {
            "cancer_type": "SARC_SYN",
            "reference_cancer_type": "SARC",
            "report_scope_cancer_type": "SARC_SYN",
            "report_scope_parent_cancer_type": "SARC",
            "analysis_constraints": {"cancer_type": "SARC_SYN"},
            "cancer_type_source": "user-specified",
        }
    )

    assert context.code_for("therapy") == "SARC_SYN"
    assert context.code_for("cohort") == "SARC"
    assert context.code_for("parent") == "SARC"
    assert context.relationship == "fine_child_of_reference"
    assert context.uses_distinct_reference


def test_context_excludes_sibling_reference_from_active_report_roles():
    from trufflepig.main import _cancer_type_context_line

    context = cancer_type_context_from_analysis(
        {
            "cancer_type": "BRCA_Basal",
            "report_scope_cancer_type": "BRCA_Basal",
            "reference_cancer_type": "BRCA_HER2",
            "expression_reference_cancer_type": "BRCA_HER2",
        }
    )

    assert context.code_for("report") == "BRCA_Basal"
    assert context.code_for("reference") == "BRCA"
    assert context.code_for("expression") == "BRCA_Basal"
    assert context.reference_relationship == "ancestor"
    assert context.expression_relationship == "same"
    assert context.excluded_sibling_codes == ("BRCA_HER2",)
    exported = context.to_dict()
    assert exported["report_role"] == "diagnosis"
    assert exported["reference_role"] == "ancestor_analysis_context"
    assert exported["expression_role"] == "same_expression_reference"

    lines = "\n".join(context.markdown_lines())
    assert "BRCA_Basal (Basal-like)" in lines
    assert "BRCA (Breast Invasive Carcinoma)" in lines
    assert "does not replace the report diagnosis" in lines
    assert "BRCA_HER2 (HER2-enriched)" in lines
    assert "is not used as a report, reference, or expression context" in lines
    report_line = _cancer_type_context_line(context)
    assert "diagnosis/report node is BRCA_Basal (Basal-like)" in report_line
    assert "BRCA (Breast Invasive Carcinoma) is the ancestor analysis context only" in (
        report_line
    )
    assert "competing sibling BRCA_HER2 (HER2-enriched) is explicitly excluded" in (
        report_line
    )


def test_direct_broad_report_reference_replaces_independent_fallback():
    from trufflepig.main import _cancer_type_context_line

    context = cancer_type_context_from_analysis(
        {
            "cancer_type": "CRC",
            "report_scope_cancer_type": "CRC",
            "reference_cancer_type": "SARC_PLEOLPS",
            "expression_reference_cancer_type": "CRC",
        }
    )

    report_line = _cancer_type_context_line(context)

    assert context.code_for("reference") == "CRC"
    assert context.reference_relationship == "same"
    assert context.requested_reference_code == "SARC_PLEOLPS"
    assert "SARC_PLEOLPS" not in report_line


def test_synchronized_active_context_does_not_resurrect_requested_sibling():
    context = cancer_type_context_from_analysis(
        {
            "cancer_type": "BRCA_Basal",
            "report_scope_cancer_type": "BRCA_Basal",
            "reference_cancer_type": "BRCA",
            "expression_reference_cancer_type": "BRCA_Basal",
            "requested_reference_cancer_type": "BRCA_HER2",
            "requested_expression_reference_cancer_type": "BRCA_HER2",
        }
    )

    assert context.code_for("reference") == "BRCA"
    assert context.code_for("expression") == "BRCA_Basal"
    assert context.requested_reference_code == "BRCA_HER2"
    assert context.requested_expression_code == "BRCA_HER2"


def test_context_labels_descendant_expression_as_reference_only(monkeypatch):
    basal_record = ExpressionReferenceRecord(
        requested_code="BRCA_Basal",
        reference_code="BRCA_Basal",
        source_kind="deconvolved_tumor_reference",
        source="TCGA_BRCA_PAM50",
    )
    her2_record = ExpressionReferenceRecord(
        requested_code="BRCA_HER2",
        reference_code="BRCA_HER2",
        source_kind="deconvolved_tumor_reference",
        source="TCGA_BRCA_PAM50",
    )
    monkeypatch.setattr(
        context_module,
        "_direct_expression_reference_records",
        lambda: {
            "BRCA_Basal": (basal_record,),
            "BRCA_HER2": (her2_record,),
        },
    )

    context = cancer_type_context_from_analysis(
        {
            "cancer_type": "BRCA",
            "reference_cancer_type": "BRCA_HER2",
        }
    )

    assert context.code_for("report") == "BRCA"
    assert context.code_for("reference") == "BRCA"
    assert context.code_for("expression") == "BRCA_HER2"
    assert context.reference_relationship == "same"
    assert context.expression_relationship == "descendant"
    lines = "\n".join(context.markdown_lines())
    assert "Best expression reference (descendant only)" in lines
    assert "does not refine the report diagnosis BRCA" in lines


def test_context_defaults_refined_label_to_registry_parent_reference():
    context = cancer_type_context_from_analysis({"cancer_type": "SARC_SYN"})

    assert context.code_for("report") == "SARC_SYN"
    assert context.code_for("reference") == "SARC"
    assert context.code_for("parent") == "SARC"
    assert context.relationship == "fine_child_of_reference"
    assert context.uses_distinct_reference


def test_context_prefers_fine_expression_reference_when_available():
    context = cancer_type_context_from_analysis(
        {
            "cancer_type": "SARC_OS",
            "reference_cancer_type": "SARC",
            "report_scope_cancer_type": "SARC_OS",
            "analysis_constraints": {"cancer_type": "SARC_OS"},
            "cancer_type_source": "user-specified",
        }
    )

    assert context.code_for("report") == "SARC_OS"
    assert context.code_for("reference") == "SARC"
    assert context.report_has_expression_ref
    assert context.code_for("expression") == "SARC_OS"
    assert context.best_expression_source == "TREEHOUSE_POLYA_25_01"
    assert context.best_expression_gene_key == "symbol_only"
    assert context.best_expression_direct
    exported = context.to_dict()
    assert exported["expression_code"] == "SARC_OS"
    assert exported["fallback_expression_code"] == "SARC"


def test_context_separates_nutm_report_label_from_fallback_reference():
    context = cancer_type_context_from_analysis(
        {
            "cancer_type": "NUTM",
            "reference_cancer_type": "LUSC",
            "report_scope_cancer_type": "NUTM",
            "cancer_type_source": "user-specified",
        }
    )

    lines = "\n".join(context.markdown_lines())

    assert context.code_for("report") == "NUTM"
    assert context.code_for("reference") == "LUSC"
    assert context.code_for("expression") == "NUTM"
    assert context.best_expression_direct
    assert "Fallback expression/reference context" in lines
    assert "Best expression reference" in lines
    assert "expression data are available for the report label" in lines


def test_context_uses_documented_fallback_for_noncohort_scope():
    # NET_NONPANCREATIC is an evidence/source scope, not an expression cohort.
    # It therefore resolves through its typed member-union parent rather than
    # borrowing one site-specific neuroendocrine cohort.
    context = cancer_type_context_from_analysis(
        {
            "cancer_type": "NET_NONPANCREATIC",
            "report_scope_cancer_type": "NET_NONPANCREATIC",
        }
    )

    assert context.code_for("report") == "NET_NONPANCREATIC"
    assert not context.report_has_expression_ref
    assert context.code_for("expression") == "NET"
    assert context.best_expression_source_kind == "computed_member_union_reference"
    assert context.best_expression_fallback_reason == "registry parent"
    assert not context.best_expression_direct


def test_grouping_uses_its_typed_member_union_reference():
    # Complete member unions are direct expression references; they no longer
    # borrow one child cohort as a temporary stand-in.  Whether a grouping is
    # itself classifiable is a separate registry policy.
    for code in ("CRC", "NET", "NSCLC", "SGC"):
        record = effective_expression_reference(code)
        assert record is not None
        assert record.reference_code == code
        assert record.source_kind == "computed_member_union_reference"
        assert record.fallback_reason == ""
        assert record.direct


def test_unavailable_grouping_does_not_claim_a_direct_reference():
    # BTC remains an evidence/source scope until both CHOL and GBC are backed.
    # Retain the safe in-branch CHOL fallback rather than advertising a
    # CHOL-only aggregate as a pan-biliary direct reference.
    record = effective_expression_reference("BTC")
    assert record is not None
    assert record.reference_code == "CHOL"
    assert record.fallback_reason == "member cohort"
    assert not record.direct


def test_context_markdown_reports_expression_fallback_without_parent_context():
    context = cancer_type_context_from_analysis(
        {"cancer_type": "NET_NONPANCREATIC"}
    )

    lines = "\n".join(context.markdown_lines())

    assert "Best expression reference" in lines
    assert "registry parent" in lines


def test_expression_reference_options_canonicalize_source_codes():
    records = expression_reference_options("BRCA_HER2", include_fallback=False)

    assert any(
        record.reference_code == "BRCA_HER2"
        and record.source_code == "BRCA_Her2"
        and record.source == "TCGA_BRCA_PAM50"
        and record.source_kind == "deconvolved_tumor_reference"
        for record in records
    )


@pytest.mark.parametrize(
    (
        "code",
        "reference_code",
        "source_kind",
        "source",
        "gene_key",
        "direct",
        "fallback_reason",
    ),
    [
        (
            "PRAD",
            "PRAD",
            "deconvolved_tumor_reference",
            "TCGA",
            "ensembl_symbol",
            True,
            "",
        ),
        (
            "SARC_OS",
            "SARC_OS",
            "deconvolved_tumor_reference",
            "TREEHOUSE_POLYA_25_01",
            "symbol_only",
            True,
            "",
        ),
        (
            "SARC_CHOR",
            "SARC_CHOR",
            "deconvolved_tumor_reference",
            "TREEHOUSE_RIBOD_25_01",
            "symbol_only",
            True,
            "",
        ),
        (
            "NUTM",
            "NUTM",
            "deconvolved_tumor_reference",
            "TREEHOUSE_POLYA_25_01",
            "symbol_only",
            True,
            "",
        ),
        (
            "MM",
            "MM",
            "observed_bulk_reference",
            "MMRF_COMMPASS",
            "ensembl_symbol",
            True,
            "",
        ),
        (
            "CLL",
            "CLL",
            "observed_bulk_reference",
            "CLLMAP_2022",
            "ensembl_symbol",
            True,
            "",
        ),
        (
            "PCN",
            "MM",
            "observed_bulk_reference",
            "MMRF_COMMPASS",
            "ensembl_symbol",
            False,
            "registry parent",
        ),
        # Types that gained their own cohort in pirlygenes >=5.11 and now
        # resolve directly instead of falling back to a parent/family.
        (
            "NBL",
            "NBL",
            "observed_bulk_reference",
            "TARGET_NBL_2018",
            "ensembl_symbol",
            True,
            "",
        ),
        (
            "SARC_ASPS",
            "SARC_ASPS",
            "observed_bulk_reference",
            "TREEHOUSE_POLYA_25_01",
            "ensembl_symbol",
            True,
            "",
        ),
        (
            "ADCC",
            "ADCC",
            "observed_bulk_reference",
            "GSE294016_BARTL_2025_SGC",
            "ensembl_symbol",
            True,
            "",
        ),
        (
            "NET_LUNG",
            "NET_LUNG",
            "observed_bulk_reference",
            "DRMETRICS_ALCALA_2019_LNEN",
            "ensembl_symbol",
            True,
            "",
        ),
        # Types that still fall back — keep each documented fallback branch
        # under test now that ADCC/MTC/NET_LUNG resolve directly. (The
        # deconvolved-tumor-reference fallback PATH stays covered by SARC_GCTB
        # below; the old neuroendocrine-fallback example was retired because the
        # rebuild gave every NE code its own cohort — see NEC_MERKEL.)
        (
            # Evidence/source scopes are not expression cohorts. The typed
            # member-union parent supplies a deterministic broad reference.
            "NET_NONPANCREATIC",
            "NET",
            "computed_member_union_reference",
            "DRMETRICS_ALCALA_2019_LNEN + GSE118014_ALVAREZ_2018 + "
            "GSE98894_ALVAREZ_2018_NET",
            "ensembl_symbol",
            False,
            "registry parent",
        ),
        (
            # NEC_MERKEL gained its own Merkel-cell cohort in the reference rebuild,
            # so it now self-references instead of falling back to SCLC.
            "NEC_MERKEL",
            "NEC_MERKEL",
            "observed_bulk_reference",
            "GSE235092_MERKEL_2024",
            "ensembl_symbol",
            True,
            "",
        ),
        (
            "SARC_GCTB",
            "SARC",
            "deconvolved_tumor_reference",
            "TCGA",
            "ensembl_symbol",
            False,
            "registry parent",
        ),
    ],
)
def test_effective_expression_reference_examples_cover_reference_paths(
    code,
    reference_code,
    source_kind,
    source,
    gene_key,
    direct,
    fallback_reason,
):
    record = effective_expression_reference(code)

    assert record is not None
    assert record.requested_code == code
    assert record.reference_code == reference_code
    assert record.source_kind == source_kind
    assert record.source == source
    assert record.normalization == "clean_tpm"
    assert record.gene_key == gene_key
    assert record.direct is direct
    assert record.fallback_reason == fallback_reason


def test_mbl_subgroup_uses_its_artifact_when_present_and_parent_otherwise():
    """Optional exact artifacts take precedence without breaking older bundles."""
    manifest = reference_module.cancer_reference_manifest()
    exact_available = manifest["cancer_code"].eq("MBL_G3").any()

    record = effective_expression_reference("MBL_G3")

    assert record is not None
    assert record.requested_code == "MBL_G3"
    assert record.reference_code == ("MBL_G3" if exact_available else "MBL")
    assert record.direct is bool(exact_available)
    assert record.fallback_reason == ("" if exact_available else "registry parent")


def test_reference_discovery_keeps_other_sources_when_pan_reference_fails(monkeypatch):
    def raise_pan_error(*args, **kwargs):
        raise RuntimeError("pan reference unavailable")

    context_module._direct_expression_reference_records.cache_clear()
    monkeypatch.setattr(reference_module, "pan_cancer_expression", raise_pan_error)
    try:
        records = {
            code: context_module.effective_expression_reference(code)
            for code in (
                "PRAD",
                "SARC_OS",
                "SARC_CHOR",
                "MM",
                "PCN",
                "NBL",
                "ADCC",
                "MTC",
                "NET_LUNG",
            )
        }
        her2_records = context_module.expression_reference_options(
            "BRCA_HER2", include_fallback=False
        )
        stk11_records = context_module.expression_reference_options(
            "LUAD_STK11", include_fallback=False
        )
    finally:
        context_module._direct_expression_reference_records.cache_clear()

    assert records["PRAD"] is not None
    assert records["PRAD"].reference_code == "PRAD"
    assert records["SARC_OS"] is not None
    assert records["SARC_OS"].source == "TREEHOUSE_POLYA_25_01"
    assert records["SARC_CHOR"] is not None
    assert records["SARC_CHOR"].source == "TREEHOUSE_RIBOD_25_01"
    assert records["MM"] is not None
    assert records["MM"].source_kind == "observed_bulk_reference"
    assert records["PCN"] is not None
    assert records["PCN"].reference_code == "MM"
    # These four gained their own (non-pan) cohorts in pirlygenes >=5.11, so
    # they resolve directly even when the pan-cancer reference is unavailable.
    assert records["NBL"] is not None
    assert records["NBL"].reference_code == "NBL"
    assert records["ADCC"] is not None
    assert records["ADCC"].reference_code == "ADCC"
    assert records["MTC"] is not None
    assert records["MTC"].reference_code == "MTC"
    assert records["NET_LUNG"] is not None
    assert records["NET_LUNG"].reference_code == "NET_LUNG"
    assert any(record.source_code == "BRCA_Her2" for record in her2_records)
    assert any(record.source_code == "LUAD_KRAS_STK11" for record in stk11_records)


def test_reference_discovery_does_not_materialize_full_observed_bulk(monkeypatch):
    """Reference availability comes from the lightweight provenance artifact.

    Calling cancer_reference_expression without a code filter materializes the
    full object-heavy table (~7.4 GB in pirlygenes 5.23), which is catastrophic
    when xdist repeats it per worker.
    """

    def reject_full_frame(*args, **kwargs):
        raise AssertionError("full observed-bulk expression frame was loaded")

    context_module._direct_expression_reference_records.cache_clear()
    monkeypatch.setattr(
        reference_module,
        "cancer_reference_expression",
        reject_full_frame,
    )
    try:
        record = context_module.effective_expression_reference("NEC_MERKEL")
    finally:
        context_module._direct_expression_reference_records.cache_clear()

    assert record is not None
    assert record.reference_code == "NEC_MERKEL"
    assert record.source_kind == "observed_bulk_reference"
    assert record.source == "GSE235092_MERKEL_2024"


def test_effective_references_never_borrow_a_sibling_cohort():
    missing = [
        code
        for code in cancer_type_registry()["code"].dropna().astype(str)
        if effective_expression_reference(code) is None
    ]

    assert not missing
    for code in cancer_type_registry()["code"].dropna().astype(str):
        record = effective_expression_reference(code)
        if record is None:
            continue
        assert (
            context_module.cancer_type_tree_relationship(
                code,
                record.reference_code,
            )
            != "sibling"
        ), (code, record.reference_code)

    thymic_carcinoma = effective_expression_reference("THYMCA")
    assert thymic_carcinoma is not None
    assert thymic_carcinoma.reference_code == "THYM_EPITHELIAL"
    assert thymic_carcinoma.fallback_reason == "registry parent"


def test_expression_reference_contract_is_clean_tpm_with_known_gene_keys():
    records = []
    for code in cancer_type_registry()["code"].dropna().astype(str):
        records.extend(expression_reference_options(code, include_fallback=True))

    assert records
    assert {record.normalization for record in records} == {"clean_tpm"}
    assert {record.gene_key for record in records} <= {"ensembl_symbol", "symbol_only"}
    assert all(record.reference_code for record in records)
    assert all(record.source for record in records)
