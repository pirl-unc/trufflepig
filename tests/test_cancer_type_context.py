from pirlygenes.gene_sets_cancer import cancer_type_registry

import pytest

import trufflepig.analyze.cancer_type_context as context_module
from trufflepig import reference as reference_module
from trufflepig.analyze import (
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


def test_context_uses_documented_fallback_when_fine_expression_is_missing():
    # ACINIC (acinic cell carcinoma) has no direct cohort of its own. Since
    # oncoref 1.8.95 reparented it under SGC, it now resolves via a two-hop
    # path: registry parent (SGC) -> salivary family fallback (HNSC). SGC itself
    # has no direct cohort, so the salivary-family fallback still lands on HNSC.
    # (ADCC gained its own cohort in pirlygenes >=5.11 and now resolves directly.)
    context = cancer_type_context_from_analysis(
        {
            "cancer_type": "ACINIC",
            "reference_cancer_type": "HNSC",
            "report_scope_cancer_type": "ACINIC",
        }
    )

    assert context.code_for("report") == "ACINIC"
    assert not context.report_has_expression_ref
    assert context.code_for("expression") == "HNSC"
    assert context.best_expression_source_kind == "deconvolved_tumor_reference"
    assert (
        context.best_expression_fallback_reason
        == "registry parent; salivary family fallback"
    )
    assert not context.best_expression_direct


def test_context_markdown_reports_expression_fallback_without_parent_context():
    context = cancer_type_context_from_analysis({"cancer_type": "ACINIC"})

    lines = "\n".join(context.markdown_lines())

    assert "Best expression reference" in lines
    assert "HNSC" in lines
    assert "salivary family fallback" in lines


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
            "MBL_G3",
            "MBL_G3",
            "observed_bulk_reference",
            "TREEHOUSE_POLYA_25_01_MBL_SUBGROUP_MARKERS",
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
            "MTC",
            "MTC",
            "observed_bulk_reference",
            "GSE32662_PRINGLE_2012_MTC",
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
        # deconvolved-tumor-reference fallback PATH stays covered by ACINIC
        # below + SARC_GCTB; the old neuroendocrine-fallback example was retired
        # because the rebuild gave every NE code its own cohort — see NEC_MERKEL.)
        (
            # Two-hop fallback since oncoref 1.8.95 reparented ACINIC under SGC:
            # registry parent (SGC, no direct cohort) -> salivary family (HNSC).
            "ACINIC",
            "HNSC",
            "deconvolved_tumor_reference",
            "TCGA",
            "ensembl_symbol",
            False,
            "registry parent; salivary family fallback",
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


def test_every_registry_code_has_an_effective_expression_reference():
    missing = [
        code
        for code in cancer_type_registry()["code"].dropna().astype(str)
        if effective_expression_reference(code) is None
    ]

    assert not missing


def test_expression_reference_contract_is_clean_tpm_with_known_gene_keys():
    records = []
    for code in cancer_type_registry()["code"].dropna().astype(str):
        records.extend(expression_reference_options(code, include_fallback=True))

    assert records
    assert {record.normalization for record in records} == {"clean_tpm"}
    assert {record.gene_key for record in records} <= {"ensembl_symbol", "symbol_only"}
    assert all(record.reference_code for record in records)
    assert all(record.source for record in records)
