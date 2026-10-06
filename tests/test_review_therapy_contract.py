"""Regression cases found while reviewing the shared therapeutic report contract."""

import pandas as pd
import pytest

from trufflepig.brief import recommend_therapies
from trufflepig.report_content import (
    assess_therapy, build_report_content, render_report_summary, therapy_evidence_sources,
)
from trufflepig.report_language import report_literal, report_plain_text, render_report_template
from trufflepig.report_view import build_report_view
from trufflepig.reporting import cancer_therapy_panel_for_analysis, supplied_variant_supports_target_row
from trufflepig.therapy_eligibility import collect_evidence_requests, evaluate_therapy_eligibility
from trufflepig.variants import parse_variant_file


def panel_row(code, agent, analysis=None):
    _, subtype, panel = cancer_therapy_panel_for_analysis(code, analysis or {"cancer_type": code})
    return subtype, next(row for row in panel.to_dict("records") if agent in str(row["agent"]))


def test_parsed_protein_substitutions_satisfy_their_exact_therapy_gate(tmp_path):
    path = tmp_path / "variants.csv"
    path.write_text("gene,variant\nKRAS,p.G12C\nBRAF,V600E\n")
    analysis = {
        "cancer_type": "COAD",
        "variant_records": [record.public_dict() for record in parse_variant_file(path)],
    }
    for agent in ("sotorasib", "encorafenib"):
        subtype, row = panel_row("COAD", agent, analysis)
        assert evaluate_therapy_eligibility(row, analysis, panel_subtype=subtype).permits_review


@pytest.mark.parametrize("variant,variant_type,expected", [
    ("MET L1195V", "mutation", False),
    ("MET amplification", "amplification", False),
    ("MET exon 13 skipping", "mutation", False),
    ("MET exon 14 deletion", "loss", False),
    ("MET exon 14 skipping", "mutation", True),
])
def test_named_exon_event_controls_eligibility_and_selection(variant, variant_type, expected):
    subtype, row = panel_row("LUAD", "capmatinib")
    analysis = {"cancer_type": "LUAD", "variant_records": [
        {"gene": "MET", "variant": variant, "variant_type": variant_type},
    ]}
    assert bool(supplied_variant_supports_target_row(row, analysis)) is expected
    assert evaluate_therapy_eligibility(row, analysis, panel_subtype=subtype).permits_review is expected
    assert bool(recommend_therapies(pd.DataFrame([row]), pd.DataFrame(), analysis=analysis)) is expected


def test_fusion_supports_a_broad_altered_indication():
    analysis = {"cancer_type": "BLCA", "fusion_records": [
        {"gene_a": "FGFR3", "gene_b": "TACC3", "name": "FGFR3--TACC3"},
    ]}
    _, row = panel_row("BLCA", "erdafitinib", analysis)
    assert supplied_variant_supports_target_row(row, analysis)


def test_mutation_specific_indication_requires_evidence_even_without_upstream_gate():
    analysis = {"cancer_type": "OV"}
    subtype, row = panel_row("OV", "olaparib", analysis)
    assert not evaluate_therapy_eligibility(row, analysis, panel_subtype=subtype).permits_review
    assert not recommend_therapies(pd.DataFrame([row]), pd.DataFrame(), analysis=analysis)


@pytest.mark.parametrize("code,agent,target,required_context", [
    ("COAD", "cetuximab", "EGFR", "RAS"),
    ("BRCA", "sacituzumab govitecan", "TACSTD2", "ER/PR/HER2"),
])
def test_assay_request_does_not_substitute_the_drug_target_for_its_biomarker(
    code, agent, target, required_context,
):
    analysis = {"cancer_type": code}
    _, subtype, panel = cancer_therapy_panel_for_analysis(code, analysis)
    row = next(row for row in panel.to_dict("records") if row["agent"] == agent)
    requirements = evaluate_therapy_eligibility(row, analysis, panel_subtype=subtype).requirements
    assay = next(r for r in requirements if r.kind in {"wildtype", "clinical_target_assay"})
    assert target not in assay.description
    assert required_context in assay.question


def test_unestablished_subtype_remains_an_information_request():
    row = {"symbol": "KIT", "agent": "imatinib", "cancer_code": "SARC",
           "subtype": "SARC_GIST", "indication": "gastrointestinal stromal tumor", "phase": "approved"}
    analysis = {"cancer_type": "SARC"}
    decision = evaluate_therapy_eligibility(row, analysis)
    assert not decision.permits_review
    assert not decision.has_known_blocker
    assert any(request["kind"] == "scope" for request in collect_evidence_requests([
        assess_therapy(row, analysis=analysis),
    ]))


def test_target_mutation_does_not_establish_a_histologic_subtype():
    row = {"symbol": "PDGFRA", "agent": "avapritinib", "cancer_code": "SARC",
           "subtype": "gist", "indication": "PDGFRA D842V GIST", "phase": "approved"}
    analysis = {"cancer_type": "SARC", "variant_records": [
        {"gene": "PDGFRA", "variant": "D842V", "variant_type": "mutation"},
    ]}
    decision = evaluate_therapy_eligibility(row, analysis)
    assert decision.supplied_variant_supported
    assert not decision.permits_review
    assert any(r.kind == "scope" and r.status == "missing" for r in decision.requirements)


@pytest.mark.parametrize("text", ["FoundationOne&reg;", "A&amp;B", "HLA-A*02:01", "a | b [c] *d*"])
def test_literal_provenance_survives_markdown_entities(text):
    assert report_plain_text(report_literal(text)) == text


def test_table_blocks_are_contiguous_markdown():
    summary = render_report_template("report", sample_id="S", sections=[{
        "title": "T", "blocks": [{"kind": "table", "headers": ["A", "B"], "rows": [["1", "2"], ["3", "4"]]}],
    }])
    lines = summary.splitlines()
    header = lines.index("| A | B |")
    assert lines[header + 1:header + 4] == ["| --- | --- |", "| 1 | 2 |", "| 3 | 4 |"]


def test_curated_sources_keep_each_citation_and_unknown_provenance():
    row = {"source": "PMID:26286086; PMID:30255937; NCT05372640; source & note"}
    sources = therapy_evidence_sources(row)
    assert [s["url"] for s in sources] == [
        "https://pubmed.ncbi.nlm.nih.gov/26286086/",
        "https://pubmed.ncbi.nlm.nih.gov/30255937/",
        "https://clinicaltrials.gov/study/NCT05372640", "",
    ]
    assessment = assess_therapy(row)
    assert assessment["sources"] == sources
    assert assessment["source_url"] == ""
    assert "source & note" in assessment["source"]


def test_updated_trial_source_takes_precedence_over_older_phase_citation():
    _, row = panel_row("PRAD", "ifinatamab deruxtecan")
    assert therapy_evidence_sources(row) == [{
        "label": "IDeate-Prostate01 (NCT06925737)",
        "url": "https://clinicaltrials.gov/study/NCT06925737",
    }]


def test_therapy_curation_error_cannot_restore_unfiltered_rows(monkeypatch):
    from trufflepig import reporting

    def invalid_correction(row):
        raise ValueError("invalid curation")

    monkeypatch.setattr(reporting, "_current_therapy_row_overrides", invalid_correction)
    rows = pd.DataFrame([{"cancer_code": "COAD", "symbol": "KRAS", "agent": "sotorasib"}])
    with pytest.raises(ValueError, match="invalid curation"):
        reporting.filter_current_therapy_targets(rows)


@pytest.mark.parametrize("code,extra", [
    ("SARC_OS", {}),
    ("NUTM", {}),
    ("SARC", {"variant_records": [
        {"gene": "EGFR", "variant": "EGFR kinase domain duplication", "variant_type": "mutation"},
    ]}),
])
def test_selected_therapy_table_and_sources_are_in_the_authored_report(tmp_path, code, extra):
    from pypdf import PdfReader
    from trufflepig.report_document import write_report_document
    from trufflepig.report_pdf import build_interpretive_report_pdf

    analysis = {"cancer_type": code, "sample_mode": "solid", "purity": {}, **extra}
    view = build_report_view(analysis, sample_id="cited-therapies")
    content = build_report_content(analysis, pd.DataFrame(), code, "", report_view=view)
    selected = [a for a in content.therapy_assessments if a["selected"]]
    assert selected
    summary = render_report_summary(content)
    assert "| Target | Recommendation |" in summary
    for assessment in selected:
        assert assessment["sources"]
        for source in assessment["sources"]:
            assert source["url"] in summary
    write_report_document(tmp_path, "cited-therapies", report_view=view, content=content)
    (tmp_path / "cited-therapies-summary.md").write_text(summary)
    pdf = PdfReader(build_interpretive_report_pdf(tmp_path))
    text = " ".join(page.extract_text() for page in pdf.pages)
    assert "Evidence source" in text
    assert all(a["agent"] in text for a in selected)
    assert all(source["label"] in text for a in selected for source in a["sources"])
