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
            "cancer_type": "OS",
            "reference_cancer_type": "SARC",
            "report_scope_cancer_type": "OS",
            "analysis_constraints": {"cancer_type": "OS"},
            "cancer_type_source": "user-specified",
        }
    )

    assert context.code_for("report") == "OS"
    assert context.code_for("reference") == "SARC"
    assert context.report_has_expression_ref
    assert context.code_for("expression") == "OS"
    assert context.best_expression_source == "TREEHOUSE_POLYA_25_01"
    assert context.best_expression_gene_key == "symbol_only"
    assert context.best_expression_direct


def test_context_uses_documented_fallback_when_fine_expression_is_missing():
    context = cancer_type_context_from_analysis(
        {
            "cancer_type": "ADCC",
            "reference_cancer_type": "HNSC",
            "report_scope_cancer_type": "ADCC",
        }
    )

    assert context.code_for("report") == "ADCC"
    assert not context.report_has_expression_ref
    assert context.code_for("expression") == "HNSC"
    assert context.best_expression_source_kind == "deconvolved_tumor_reference"
    assert context.best_expression_fallback_reason == "salivary family fallback"
    assert not context.best_expression_direct


def test_context_markdown_reports_expression_fallback_without_parent_context():
    context = cancer_type_context_from_analysis({"cancer_type": "ADCC"})

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
            "OS",
            "OS",
            "deconvolved_tumor_reference",
            "TREEHOUSE_POLYA_25_01",
            "symbol_only",
            True,
            "",
        ),
        (
            "CHOR",
            "CHOR",
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
        (
            "NBL",
            "NBL_MYCN_nonamp",
            "deconvolved_tumor_reference",
            "TARGET_NBL_2018",
            "symbol_only",
            False,
            "curated code fallback",
        ),
        (
            "MBL_G3",
            "MBL",
            "deconvolved_tumor_reference",
            "TREEHOUSE_POLYA_25_01",
            "symbol_only",
            False,
            "registry parent",
        ),
        (
            "SARC_ASPS",
            "SARC",
            "deconvolved_tumor_reference",
            "TCGA",
            "ensembl_symbol",
            False,
            "registry parent",
        ),
        (
            "ADCC",
            "HNSC",
            "deconvolved_tumor_reference",
            "TCGA",
            "ensembl_symbol",
            False,
            "salivary family fallback",
        ),
        (
            "MTC",
            "THCA",
            "deconvolved_tumor_reference",
            "TCGA",
            "ensembl_symbol",
            False,
            "endocrine family fallback",
        ),
        (
            "LUNG_NET_LC",
            "SCLC",
            "deconvolved_tumor_reference",
            "SCLC_UCOLOGNE_2015",
            "symbol_only",
            False,
            "net family fallback",
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
                "OS",
                "CHOR",
                "MM",
                "PCN",
                "NBL",
                "ADCC",
                "MTC",
                "LUNG_NET_LC",
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
    assert records["OS"] is not None
    assert records["OS"].source == "TREEHOUSE_POLYA_25_01"
    assert records["CHOR"] is not None
    assert records["CHOR"].source == "TREEHOUSE_RIBOD_25_01"
    assert records["MM"] is not None
    assert records["MM"].source_kind == "observed_bulk_reference"
    assert records["PCN"] is not None
    assert records["PCN"].reference_code == "MM"
    assert records["NBL"] is not None
    assert records["NBL"].reference_code == "NBL_MYCN_nonamp"
    assert records["ADCC"] is not None
    assert records["ADCC"].reference_code == "HNSC"
    assert records["MTC"] is not None
    assert records["MTC"].reference_code == "THCA"
    assert records["LUNG_NET_LC"] is not None
    assert records["LUNG_NET_LC"].reference_code == "SCLC"
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
