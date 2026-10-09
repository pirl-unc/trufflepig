"""Reader conclusions must retain evidence boundaries and usable figure links."""

import pandas as pd
import pytest
from PIL import Image
from pypdf import PdfReader

from trufflepig import brief
from trufflepig.report_content import (
    ReportContent, build_report_content, conditional_therapy_groups, render_report_summary,
)
from trufflepig.report_language import report_literal
from trufflepig.report_document import write_report_document, load_report_document
from trufflepig.report_pdf import build_interpretive_report_pdf
from trufflepig.report_takeaways import therapeutic_lead_basis
from trufflepig.report_view import build_report_view
from types import SimpleNamespace


def test_rna_msi_lead_has_drug_and_citation_without_shortlisting(monkeypatch):
    monkeypatch.setattr(brief, "mismatch_repair_rna_state", lambda _: "MSI-like")
    analysis = {"cancer_type": "COAD", "sample_mode": "solid", "purity": {}}
    view = build_report_view(analysis)
    content = build_report_content(analysis, pd.DataFrame(), "COAD", "", report_view=view)
    assessment = next(a for a in content.therapy_assessments if a["agent"] == "pembrolizumab")
    assert not assessment["selected"]
    summary = render_report_summary(content)
    assert "MSI-like RNA prioritizes testing; clinical MSI-H/dMMR is unconfirmed" in summary
    assert "pembrolizumab" in summary and "https://pubmed.ncbi.nlm.nih.gov/33264544/" in summary
    assert "Vulnerabilities worth investigating" in summary
    conditional = conditional_therapy_groups(content.therapy_assessments, analysis,
                                             already_discussed={assessment["id"]})
    assert all(a["agent"] != "pembrolizumab" for group in conditional for a in group["assessments"])


def test_missing_clinical_results_keep_colorectal_options_visible_without_rna(monkeypatch):
    monkeypatch.setattr(brief, "mismatch_repair_rna_state", lambda _: "Discordant")
    analysis = {"cancer_type": "COAD", "sample_mode": "solid", "purity": {}}
    content = build_report_content(analysis, pd.DataFrame(), "COAD", "",
                                   report_view=build_report_view(analysis))
    groups = conditional_therapy_groups(content.therapy_assessments, analysis)
    assert len(groups) == 5  # Shared KRAS and EGFR criteria each form one group.
    assert sum(len(group["assessments"]) for group in groups) == 7
    assert not any(a["selected"] or a["eligibility"]["permits_review"] for a in content.therapy_assessments)
    therapy = next(s for s in content.sections if s["id"] == "therapies")
    text = " ".join(b.get("text", "") for b in therapy["blocks"])
    assert "Approved options pending clinical confirmation" in text
    assert "Vulnerabilities worth investigating" not in text
    assert "No therapy" not in text
    assert "not a complete treatment plan" in text
    for term in ("BRAF V600E", "KRAS G12C", "HER2-positive", "RAS wild-type", "MSI-H/dMMR"):
        assert term in text
    for assessment in content.therapy_assessments:
        assert report_literal(assessment["agent"]) in text
        assert assessment["source_url"] in text
    assert "210496s021lbl.pdf" in text  # Current label accompanies the earlier trial citation.
    assert text.count("reconcile prior checkpoint therapy and immune toxicity") == 1


@pytest.mark.parametrize("result", ["MSS", "pending", "conflicting"])
def test_conditional_options_do_not_override_clinical_mmr_results(result):
    from trufflepig.clinical_context import ClinicalAssay, ClinicalContext, ClinicalSource

    def assay(value):
        return ClinicalAssay(kind="msi", result=value, method="PCR", specimen_id="sample",
                             scope="current", validity="validated", reportability="reportable",
                             source=ClinicalSource(title="Synthetic clinical report"))

    assays = (assay("MSI-H"), assay("MSS")) if result == "conflicting" else (assay(result),)
    analysis = {"cancer_type": "COAD", "sample_mode": "solid", "purity": {},
                "clinical_context": ClinicalContext(specimen_id="sample", assays=assays).public_dict()}
    content = build_report_content(analysis, pd.DataFrame(), "COAD", "",
                                   report_view=build_report_view(analysis))
    groups = conditional_therapy_groups(content.therapy_assessments, analysis)
    assert all(a["agent"] != "pembrolizumab" for group in groups for a in group["assessments"])
    assert not next(a for a in content.therapy_assessments if a["agent"] == "pembrolizumab")["selected"]


def test_conditional_options_preserve_scope_conflicts_and_hla_criteria():
    from trufflepig.report_content import assess_therapy

    row = dict(symbol="MAGEA4", agent="afamitresgene autoleucel", phase="approved",
               indication="synovial sarcoma", subtype="synovial_sarcoma",
               eligibility_note="requires the indicated antigen assay", hla_restriction="A*02:01")
    analysis = {"cancer_type": "SARC"}
    assessment = assess_therapy(row, analysis=analysis)
    assert conditional_therapy_groups([assessment], analysis) == []
    analysis["cancer_type"] = "SARC_SYN"
    assessment = assess_therapy(row, analysis=analysis)
    groups = conditional_therapy_groups([assessment], analysis)
    assert "high-resolution HLA" in " ".join(groups[0]["criteria"])
    assert "A*02:01" in " ".join(groups[0]["criteria"])
    analysis["analysis_constraints"] = {"hla_types": ["A*01:01", "A*03:01"]}
    assert conditional_therapy_groups([assess_therapy(row, analysis=analysis)], analysis) == []
    assert conditional_therapy_groups([assessment], {"cancer_type_abstention": {"reason": "muscle"}}) == []

    row = dict(symbol="KRAS", agent="sotorasib", cancer_code="COAD", phase="approved",
               indication="KRAS G12C colorectal cancer", requires_verified_alteration=True)
    analysis = {"cancer_type": "COAD", "variant_inputs_supplied": True, "variant_records": [
        {"gene": "KRAS", "variant": "p.G12D", "variant_type": "mutation", "status": "detected"},
    ]}
    assert conditional_therapy_groups([assess_therapy(row, analysis=analysis)], analysis) == []


def test_abundant_gene_rna_cannot_create_mutation_or_background_leads():
    assessment = {
        "eligibility": {"requirements": [{"kind": "mutation", "status": "missing"}]},
        "rna_source": {"tier": "tumor_supported"},
        "observation": {"state": "measured"},
        "tumor_band": "1000", "target": "KRAS", "selected": False,
    }
    for kind in ("mutation", "tmb_high", "histology_only"):
        assessment["eligibility"]["requirements"] = [{"kind": kind, "status": "missing"}]
        assert therapeutic_lead_basis(assessment, {}) == ""
    assessment["eligibility"]["requirements"] = []
    assessment["rna_source"]["tier"] = "background_dominant"
    assert therapeutic_lead_basis(assessment, {}) == ""
    assessment["rna_source"]["tier"] = "tumor_supported"
    assert "estimated tumor component" in therapeutic_lead_basis(assessment, {})
    assert therapeutic_lead_basis(assessment, {"cancer_type_abstention": {"reason": "muscle"}}) == ""
    assessment["eligibility"]["requirements"] = [{"kind": "wildtype", "status": "blocked"}]
    assert therapeutic_lead_basis(assessment, {}) == ""


def test_tissue_ambiguity_and_scale_conversion_remain_visible():
    analysis = {
        "cancer_type": "COAD", "sample_mode": "solid", "purity": {},
        "healthy_vs_tumor": SimpleNamespace(cancer_hint="healthy-dominant"),
        "expression_scale_qc": {"converted_from": "log2_tpm_plus_one"},
    }
    content = build_report_content(analysis, pd.DataFrame(), "COAD", "", report_view=build_report_view(analysis))
    summary = render_report_summary(content)
    assert "healthy-dominant" in summary
    assert "converted to linear TPM before interpretation" in summary


def test_assay_lead_requires_a_matching_rna_proxy_not_just_drug_target_abundance():
    assessment = {
        "eligibility": {"requirements": [{"kind": "clinical_target_assay", "status": "missing"}]},
        "rna_source": {"tier": "tumor_supported"},
        "observation": {"state": "measured"},
        "tumor_band": "150", "target": "CDK4", "selected": False,
    }
    assert therapeutic_lead_basis(assessment, {}) == ""
    supported_her2 = {"rna_biomarker_proxies": {"her2": {"status": "supported"}}}
    assert therapeutic_lead_basis(assessment, supported_her2) == ""
    assessment["target"] = "ERBB2"
    assert therapeutic_lead_basis(assessment, {}) == ""
    assert "estimated tumor component" in therapeutic_lead_basis(assessment, supported_her2)
    assessment["eligibility"]["direct_evidence_supported"] = True
    assert "Supplied clinical evidence" in therapeutic_lead_basis(assessment, {})


def test_opening_preserves_supplied_identity_and_qualifies_bulk_emt():
    analysis = {
        "cancer_type": "BRCA", "sample_mode": "solid", "purity": {},
        "analysis_constraints": {"cancer_type": "BRCA"},
        "candidate_trace": [
            {"code": "SARC", "support_fraction_of_top": 1.0},
            {"code": "BRCA", "support_fraction_of_top": 0.8},
        ],
        "fit_quality": {"label": "ambiguous"},
        "therapy_response_scores": {"EMT": SimpleNamespace(state="up")},
    }
    content = build_report_content(analysis, pd.DataFrame(), "BRCA", "EMT program is active.",
                                   report_view=build_report_view(analysis))
    opening = " ".join(b.get("text", "") for b in content.sections[0]["blocks"])
    assert "supplied diagnosis" in opening
    assert "SARC" not in opening
    assert "does not establish a tumor-cell transition" in opening
    assert "RNA classifier check" in render_report_summary(content)


def test_numbered_references_match_actual_figures_in_markdown_and_pdf(tmp_path):
    figures = tmp_path / "figures"
    figures.mkdir()
    for suffix in ("priority-target-context.png", "purity-methods.png"):
        Image.new("RGB", (320, 180), "white").save(figures / f"S-{suffix}")
    content = ReportContent("S", [
        {"id": "conclusion", "title": "Takeaways", "blocks": [{
            "kind": "bullet", "text": "Target source is conditional.",
            "figure_suffixes": ["priority-target-context.png", "sample-context.png", "purity-methods.png"],
        }]},
        {"id": "evidence", "title": "Evidence", "blocks": []},
    ], [], [], [])
    view = build_report_view({"cancer_type": "COAD", "sample_mode": "solid", "purity": {}})
    write_report_document(tmp_path, "S", content=content, report_view=view)
    document = load_report_document(tmp_path, "S")
    text = document["sections"][0]["blocks"][0]["text"]
    assert "[Figure 1](#figure-1), [Figure 2](#figure-2)" in text
    summary = (tmp_path / "S-summary.md").read_text()
    assert '<a id="figure-1"></a>' in summary
    assert "figures/S-priority-target-context.png" in summary
    reader = PdfReader(build_interpretive_report_pdf(tmp_path))
    destinations = [a.get_object().get("/Dest") for p in reader.pages for a in p.get("/Annots", [])]
    assert len([d for d in destinations if d]) == 2
    assert "Figure 2. Purity method agreement" in " ".join(p.extract_text() for p in reader.pages)
    # Rebuild with no plots: no fabricated references, stale numbers, or mutation
    # of the source blocks used for another renderer.
    for path in figures.glob("*.png"):
        path.unlink()
    write_report_document(tmp_path, "S", content=content, report_view=view)
    assert "Figure" not in (tmp_path / "S-summary.md").read_text()
    assert content.sections[0]["blocks"][0]["text"] == "Target source is conditional."
