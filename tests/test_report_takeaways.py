"""Reader conclusions must retain evidence boundaries and usable figure links."""

import pandas as pd
from PIL import Image
from pypdf import PdfReader

from trufflepig import brief
from trufflepig.report_content import ReportContent, build_report_content, render_report_summary
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


def test_abundant_gene_rna_cannot_create_mutation_or_background_leads():
    assessment = {
        "eligibility": {"requirements": [{"kind": "mutation", "status": "missing"}]},
        "rna_source": {"tier": "tumor_supported"},
        "observation": {"state": "measured"},
        "tumor_band": "1000", "target": "KRAS", "selected": False,
    }
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
