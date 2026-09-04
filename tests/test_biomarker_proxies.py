import pandas as pd

from trufflepig.biomarker_proxies import (
    HER2_PANEL_GENES,
    her2_proxy_summary_line,
    her2_proxy_therapy_context,
    score_her2_rna_proxy,
    score_rna_biomarker_proxies,
)


def _reference(monkeypatch, value=1.0):
    monkeypatch.setattr(
        "trufflepig.biomarker_proxies._reference_panel",
        lambda cancer_type: (
            {gene: value for gene in HER2_PANEL_GENES},
            (cancer_type,),
            ("mock_external_cohort",),
        ),
    )


def _high_sample():
    return {
        "ERBB2": 30.0,
        "GRB7": 12.0,
        "STARD3": 9.0,
        "MIEN1": 8.0,
        "PGAP3": 7.0,
    }


def test_her2_proxy_supported_is_context_only(monkeypatch):
    _reference(monkeypatch)
    ranges = pd.DataFrame(
        [
            {
                "symbol": "ERBB2",
                "attr_tumor_tpm": 24.0,
                "attr_tumor_fraction": 0.80,
                "tcga_percentile": 0.95,
            }
        ]
    )

    result = score_her2_rna_proxy(
        _high_sample(),
        "BRCA",
        ranges_df=ranges,
        purity={"overall_estimate": 0.55},
    )

    assert result.status == "supported"
    assert result.clinical_claim == "context_only"
    assert result.eligibility_established is False
    assert result.confirmation_priority == "high"
    erbb2 = next(row for row in result.gene_evidence if row["symbol"] == "ERBB2")
    assert erbb2["patient_bulk_tpm_measured"] == 30.0
    assert erbb2["patient_tumor_attributed_tpm_rna_model_estimate"] == 24.0
    assert erbb2["external_cancer_reference_panel_median_tpm"] == 1.0
    assert result.reference_panel_kind == "external_observed_cancer_cohort"
    assert result.reference_panel_component_codes == ("BRCA",)


def test_her2_proxy_marks_bulk_source_disagreement(monkeypatch):
    _reference(monkeypatch)
    ranges = pd.DataFrame(
        [
            {
                "symbol": "ERBB2",
                "attr_tumor_tpm": 2.0,
                "attr_tumor_fraction": 0.08,
                "tme_dominant": True,
            }
        ]
    )

    result = score_her2_rna_proxy(_high_sample(), "BRCA", ranges_df=ranges)

    assert result.status == "discordant"
    assert "RNA source estimate" in result.decision_basis
    assert result.eligibility_established is False


def test_her2_proxy_carries_low_estimated_tumor_fraction_caveat(monkeypatch):
    _reference(monkeypatch)

    result = score_her2_rna_proxy(
        _high_sample(),
        "BRCA",
        purity={"overall_estimate": 0.12},
    )

    assert any("low estimated tumor fraction" in caveat.lower() for caveat in result.caveats)


def test_her2_proxy_non_support_does_not_claim_her2_negative(monkeypatch):
    _reference(monkeypatch, value=5.0)
    sample = {gene: 0.5 for gene in HER2_PANEL_GENES}

    result = score_her2_rna_proxy(sample, "BRCA")
    analysis = {"rna_biomarker_proxies": {"her2": result.to_dict()}}
    summary = her2_proxy_summary_line(analysis)

    assert result.status == "not_supported"
    assert "cannot establish HER2-negative" in summary
    assert "cannot by itself select treatment" in summary
    assert "establish eligibility" in summary


def test_her2_proxy_calls_gene_source_fraction_plainly(monkeypatch):
    _reference(monkeypatch)
    ranges = pd.DataFrame(
        [{"symbol": "ERBB2", "attr_tumor_tpm": 24.0, "attr_tumor_fraction": 0.80}]
    )

    result = score_her2_rna_proxy(_high_sample(), "BRCA", ranges_df=ranges)

    assert "assigns 80% of measured ERBB2 RNA" in result.decision_basis
    assert "estimated tumor fraction 80%" not in result.decision_basis


def test_her2_proxy_is_indeterminate_when_panel_is_missing(monkeypatch):
    _reference(monkeypatch)

    result = score_her2_rna_proxy({"ERBB2": 100.0}, "BRCA")

    assert result.status == "indeterminate"
    assert result.genes_measured == ("ERBB2",)
    assert len(result.genes_missing) == 4


def test_her2_proxy_separates_patient_coverage_from_reference_availability(
    monkeypatch,
):
    monkeypatch.setattr(
        "trufflepig.biomarker_proxies._reference_panel",
        lambda _cancer_type: ({}, (), ()),
    )

    result = score_her2_rna_proxy(_high_sample(), "CRC")

    assert result.status == "indeterminate"
    assert "All 5 panel genes were measured in patient bulk RNA" in result.decision_basis
    assert "external observed cancer-cohort reference" in result.decision_basis


def test_her2_proxy_requires_enough_reference_genes(monkeypatch):
    monkeypatch.setattr(
        "trufflepig.biomarker_proxies._reference_panel",
        lambda _code: (
            {"ERBB2": 3.0, "GRB7": 2.0, "STARD3": 2.0},
            ("BRCA",),
            ("mock_external_cohort",),
        ),
    )

    result = score_her2_rna_proxy(
        {gene: 100.0 for gene in HER2_PANEL_GENES},
        "BRCA",
    )

    assert result.status == "indeterminate"
    assert "covered only 3/5 panel genes" in result.decision_basis


def test_aggregate_crc_reference_uses_external_member_union(monkeypatch):
    import trufflepig.reference

    frame = pd.DataFrame(
        [
            {
                "Symbol": gene,
                "expression": value,
                "normalization": "TPM_clean",
                "cancer_code": code,
                "source_cohort": "external_tcga_reference",
            }
            for code, value in (("COAD", 1.0), ("READ", 3.0))
            for gene in HER2_PANEL_GENES
        ]
    )
    monkeypatch.setattr(
        trufflepig.reference,
        "cancer_reference_expression",
        lambda **_kwargs: frame,
    )

    result = score_her2_rna_proxy(_high_sample(), "CRC")

    assert result.reference_panel_component_codes == ("COAD", "READ")
    assert result.reference_panel_sources == ("external_tcga_reference",)
    erbb2 = next(row for row in result.gene_evidence if row["symbol"] == "ERBB2")
    assert erbb2["external_cancer_reference_panel_median_tpm"] == 2.0


def test_proxy_collection_and_therapy_context_feed_downstream(monkeypatch):
    _reference(monkeypatch)
    proxies = score_rna_biomarker_proxies(_high_sample(), "BRCA")
    analysis = {"rna_biomarker_proxies": proxies}
    her2_row = {
        "symbol": "ERBB2",
        "agent": "trastuzumab deruxtecan",
        "indication": "HER2-positive solid tumor",
    }
    unrelated_row = {"symbol": "EGFR", "agent": "cetuximab"}

    assert proxies["her2"]["status"] == "supported"
    context = her2_proxy_therapy_context(her2_row, analysis)
    assert "IHC/ISH confirmation" in context
    assert "does not establish eligibility" in context
    assert her2_proxy_therapy_context(unrelated_row, analysis) == ""
