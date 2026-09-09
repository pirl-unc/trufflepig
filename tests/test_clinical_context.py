"""Clinical assay results, specimen limitations and the shared report path."""

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from trufflepig.clinical_context import (
    ClinicalAssay,
    ClinicalContext,
    ClinicalSource,
    evaluate_msi_mmr,
    load_clinical_context,
)


def assay(result="MSI-H", **changes):
    values = dict(
        kind="mmr" if result in {"dMMR", "pMMR"} else "msi",
        result=result,
        method="IHC" if result in {"dMMR", "pMMR"} else "PCR",
        specimen_id="specimen-A",
        scope="current",
        validity="validated",
        reportability="reportable",
        reported_at="2026-08-01",
        source=ClinicalSource(title="Synthetic clinical pathology report"),
    )
    return ClinicalAssay(**{**values, **changes})


def context(*assays):
    return ClinicalContext(specimen_id="specimen-A", assays=assays)


@pytest.mark.parametrize(
    "result, expected",
    [
        ("MSI-H", "positive"),
        ("dMMR", "positive"),
        ("MSI-L", "negative"),
        ("MSS", "negative"),
        ("pMMR", "negative"),
        ("indeterminate", "unresolved"),
        ("pending", "unresolved"),
        ("not_tested", "unresolved"),
        ("unknown", "unresolved"),
    ],
)
def test_clinical_assay_results_are_not_presence_checks(result, expected):
    assert evaluate_msi_mmr(context(assay(result))).status == expected


@pytest.mark.parametrize(
    "changes, limitation",
    [
        ({"scope": "historical"}, "historical"),
        ({"specimen_id": "specimen-B"}, "different specimen"),
        ({"specimen_id": ""}, "identity is missing"),
        ({"validity": "failed"}, "validity is failed"),
        ({"validity": "unverified"}, "unverified"),
        ({"reportability": "unreportable"}, "unreportable"),
        ({"method": "RNA_expression"}, "clinical assay method"),
        ({"method": "RNA_reads"}, "clinical assay method"),
        ({"source": ClinicalSource()}, "source is missing"),
        ({"source": ClinicalSource(title="Extraction", review_status="proposed")}, "proposed"),
    ],
)
def test_unusable_positive_assays_retain_their_limits(changes, limitation):
    decision = evaluate_msi_mmr(context(assay(**changes)))
    assert decision.status == "unresolved"
    assert limitation in " ".join(decision.assays[0]["limitations"])


def test_context_without_explicit_specimen_binding_cannot_open_a_gate():
    assert evaluate_msi_mmr(ClinicalContext(assays=(assay(),))).status == "unresolved"


@pytest.mark.parametrize("negative", ["MSS", "pMMR"])
def test_conflicting_current_sources_are_preserved(negative):
    ctx = context(assay(), assay(negative))
    decision = evaluate_msi_mmr(ctx)
    assert decision.status == "conflicting"
    assert len(decision.assays) == 2
    assert load_clinical_context(ctx.public_dict()) == ctx


def test_historical_or_failed_negatives_do_not_override_current_usable_positive():
    for changes in ({"scope": "historical"}, {"validity": "failed"}, {"specimen_id": "other"}):
        decision = evaluate_msi_mmr(context(assay(), assay("MSS", **changes)))
        assert decision.satisfied
        assert decision.assays[1]["limitations"]


def test_mmr_protein_results_are_retained_without_inventing_an_overall_result():
    proteins = {"MLH1": "lost", "PMS2": "lost", "MSH2": "retained", "MSH6": "retained"}
    uncalled = assay("unknown", kind="mmr", method="IHC", protein_results=proteins)
    assert evaluate_msi_mmr(context(uncalled)).status == "unresolved"
    contradictory = replace(uncalled, result="pMMR")
    assert evaluate_msi_mmr(context(contradictory)).status == "conflicting"
    assert evaluate_msi_mmr(context(assay(), replace(contradictory, validity="failed"))).satisfied
    ctx = context(assay("dMMR", protein_results=proteins))
    assert evaluate_msi_mmr(ctx).satisfied
    assert load_clinical_context(ctx.public_dict()).assays[0].protein_results == proteins


@pytest.mark.parametrize(
    "payload",
    [
        [],
        None,
        "text",
        {"schema_version": 2},
        {"schema_version": True},
        {"assays": {}},
        {"typo": "MSI-H"},
        {"assays": [{"kind": "MSI"}]},
        {"assays": [{"kind": "msi", "result": "possibly MSI-H"}]},
        {"assays": [{"kind": "msi", "reported_at": "2026-99-01"}]},
        {"assays": [{"kind": "msi", "source": {"titel": "typo"}}]},
        {"assays": [{"kind": "msi", "protein_results": {"MLH1": "lost"}}]},
    ],
)
def test_invalid_json_contract_fails_loudly(tmp_path, payload):
    path = tmp_path / "context.json"
    path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, TypeError)):
        load_clinical_context(path)


def test_duplicate_ids_and_impossible_dates_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        context(assay(), assay())
    with pytest.raises(ValueError, match="precedes"):
        assay(collected_at="2026-09-01", reported_at="2026-08-01")


def test_config_captures_context_before_source_file_changes(tmp_path):
    from trufflepig.analyze.models import AnalyzeConfig

    path = tmp_path / "context.json"
    original = context(assay())
    path.write_text(json.dumps(original.public_dict()))
    config = AnalyzeConfig(input_path="sample.tsv", clinical_context=path)
    path.write_text(json.dumps(context(assay("MSS")).public_dict()))
    assert config.public_dict()["clinical_context"] == original.public_dict()


def colorectal_analysis(ctx=None):
    return {
        "cancer_type": "COAD",
        "cancer_type_source": "user-specified",
        "analysis_constraints": {"cancer_type": "COAD"},
        "sample_mode": "solid",
        "purity": {},
        "clinical_context": (ctx or context()).public_dict(),
    }


def colorectal_pembrolizumab(analysis):
    from trufflepig.reporting import cancer_therapy_panel_for_analysis

    _, subtype, panel = cancer_therapy_panel_for_analysis("COAD", analysis)
    row = next(r for r in panel.to_dict("records") if r["agent"] == "pembrolizumab")
    return row, subtype


@pytest.mark.parametrize(
    "result, expected",
    [
        ("MSI-H", True),
        ("dMMR", True),
        ("MSS", False),
        ("pMMR", False),
        ("pending", False),
        ("unknown", False),
        ("indeterminate", False),
    ],
)
def test_real_colorectal_panel_uses_clinical_assays_without_fake_variants(result, expected):
    from trufflepig.brief import recommend_therapies
    from trufflepig.report_content import assess_therapy
    from trufflepig.therapy_eligibility import evaluate_therapy_eligibility

    analysis = colorectal_analysis(context(assay(result)))
    row, subtype = colorectal_pembrolizumab(analysis)
    decision = evaluate_therapy_eligibility(row, analysis, panel_subtype=subtype)
    assert decision.direct_evidence_supported is expected
    assert not decision.supplied_variant_supported
    assert decision.permits_review is expected
    assert (
        bool(recommend_therapies(pd.DataFrame([row]), pd.DataFrame(), analysis=analysis))
        is expected
    )
    assessment = assess_therapy(row, analysis=analysis, panel_subtype=subtype)
    assert assessment["eligibility"]["permits_review"] is expected
    assert result in " ".join(assessment["rationale"])


def test_positive_assay_does_not_clear_a_supplied_contraindication():
    from trufflepig.therapy_eligibility import evaluate_therapy_eligibility

    analysis = colorectal_analysis(context(assay()))
    analysis["treatment_history"] = [{"therapy": "Keytruda", "status": "contraindicated"}]
    row, subtype = colorectal_pembrolizumab(analysis)
    decision = evaluate_therapy_eligibility(row, analysis, panel_subtype=subtype)
    assert decision.direct_evidence_supported
    assert not decision.permits_review


def test_fake_gene_variant_and_untyped_status_cannot_supply_msi():
    from trufflepig.therapy_eligibility import evaluate_therapy_eligibility

    analysis = colorectal_analysis()
    analysis["analysis_constraints"]["msi_status"] = "MSI-H"
    analysis["variant_records"] = [
        {"gene": "PDCD1", "variant": "MSI-H", "variant_type": "mutation"}
    ]
    row, subtype = colorectal_pembrolizumab(analysis)
    assert not evaluate_therapy_eligibility(row, analysis, panel_subtype=subtype).permits_review


def test_negative_and_conflicting_assays_block_even_with_prior_benefit():
    from trufflepig.therapy_eligibility import evaluate_therapy_eligibility

    for ctx in (context(assay("MSS")), context(assay(), assay("pMMR"))):
        analysis = colorectal_analysis(ctx)
        analysis["treatment_history"] = [{"therapy": "pembrolizumab", "status": "major_benefit"}]
        row, subtype = colorectal_pembrolizumab(analysis)
        decision = evaluate_therapy_eligibility(row, analysis, panel_subtype=subtype)
        assert decision.history_supported and not decision.permits_review


def test_negative_assay_does_not_suppress_unrelated_diagnosis_request():
    from trufflepig.report_content import build_report_content
    from trufflepig.report_view import build_report_view

    analysis = colorectal_analysis(context(assay("MSS")))
    analysis["cancer_type_source"] = "inferred"
    analysis["analysis_constraints"] = {}
    content = build_report_content(
        analysis, pd.DataFrame(), "COAD", "", report_view=build_report_view(analysis)
    )
    assert any(r["key"] == "diagnosis" for r in content.evidence_requests)
    assert not any(r["key"] == "msi_high" for r in content.evidence_requests)


@pytest.mark.parametrize(
    "result, phrase",
    [
        ("MSI-H", "satisfies the biomarker requirement"),
        ("MSS", "does not satisfy an MSI-H/dMMR indication"),
    ],
)
def test_both_detailed_tables_preserve_the_assay_decision(result, phrase):
    from trufflepig.brief import build_actionable
    from trufflepig.main import _build_target_report
    from trufflepig.report_view import build_report_view

    analysis = colorectal_analysis(context(assay(result)))
    analysis["purity"] = {"overall_estimate": 0.5, "overall_lower": 0.3, "overall_upper": 0.7}
    ranges = pd.DataFrame(
        {
            "symbol": pd.Series(dtype=str),
            "category": pd.Series(dtype=str),
            "is_cta": pd.Series(dtype=bool),
            "is_surface": pd.Series(dtype=bool),
            "median_est": pd.Series(dtype=float),
            "observed_tpm": pd.Series(dtype=float),
        }
    )
    for text in (
        build_actionable(analysis, ranges, "COAD", "", report_view=build_report_view(analysis)),
        _build_target_report(ranges, analysis, "COAD", analysis["purity"]),
    ):
        row = next(line for line in text.splitlines() if "| pembrolizumab |" in line)
        assert phrase in row
        assert "eligibility not supplied" not in row


@pytest.mark.parametrize("result", ["MSI-H", "MSS", "pending"])
def test_answering_request_updates_shared_markdown_json_pdf_and_manifest(tmp_path, result):
    from pypdf import PdfReader
    from trufflepig.analyze.models import AnalyzeConfig, AnalyzeRun, AnalyzePaths, InputResolution
    from trufflepig.analyze.flow import write_analysis_output_records
    from trufflepig.report_content import build_report_content, render_report_summary
    from trufflepig.report_view import build_report_view

    analysis = colorectal_analysis()
    view = build_report_view(analysis, sample_id="synthetic-clinical-assay")
    initial = build_report_content(analysis, pd.DataFrame(), "COAD", "", report_view=view)
    request = next(r for r in initial.evidence_requests if r["kind"] == "msi_high")
    assert "pembrolizumab" in request["affects"]
    ctx = context(assay(result))
    analysis["clinical_context"] = ctx.public_dict()
    content = build_report_content(analysis, pd.DataFrame(), "COAD", "", report_view=view)
    followup = [r for r in content.evidence_requests if r["kind"] == "msi_high"]
    assert bool(followup) is (result == "pending")
    if followup:
        assert followup[0]["id"] == request["id"]
        assert followup[0]["evidence"][0]["assays"][0]["result"] == result
    candidate = next(a for a in content.therapy_assessments if a["agent"] == "pembrolizumab")
    assert candidate["selected"] is (result == "MSI-H")
    assert candidate["selection"]["status"] == {
        "MSI-H": "reviewable", "MSS": "clinical_blocker", "pending": "eligibility_pending",
    }[result]
    if candidate["selected"]:
        assert any(r["kind"] == "clinical_setting" for r in content.evidence_requests)
    prefix = "synthetic-clinical-assay"
    summary = render_report_summary(content)
    if candidate["selected"]:
        information = summary.split("## Information needed", 1)[1].split("## ", 1)[0]
        assert "unresectable or metastatic colorectal cancer" in information
        assert "requires validated MSI-H/dMMR" not in information
    elif followup:
        assert "MSI/MMR" in summary and "Msi high" not in summary
    (tmp_path / f"{prefix}-summary.md").write_text(summary)
    run = AnalyzeRun(
        AnalyzeConfig(input_path="synthetic.tsv", clinical_context=ctx),
        InputResolution("synthetic.tsv", None, False, "gene"),
        AnalyzePaths(tmp_path, prefix, prefix),
    )
    write_analysis_output_records(run, view, content=content)
    doc = json.loads((tmp_path / f"{prefix}-report.json").read_text())
    manifest = json.loads((tmp_path / f"{prefix}-manifest.json").read_text())
    assert doc["clinical_context"] == manifest["config"]["clinical_context"] == ctx.public_dict()
    pdf = PdfReader(tmp_path / f"{prefix}-interpretive-report.pdf")
    pdf_text = " ".join(" ".join(p.extract_text() for p in pdf.pages).split())
    for phrase in [result, "Synthetic clinical pathology report", "specimen-A", "2026-08-01"]:
        assert phrase in summary and phrase in pdf_text
    if result == "MSS":
        assert "does not satisfy an MSI-H/dMMR indication" in summary
        assert "1 is blocked by known clinical or disease-scope exclusions" in summary
    if result != "MSI-H":
        assert "No therapy was shortlisted" in summary and "No therapy was shortlisted" in pdf_text
        assert "missing usable target RNA" not in summary


def test_rna_triage_changes_request_priority_without_opening_gate(monkeypatch):
    import trufflepig.brief as brief
    from trufflepig.report_content import build_report_content, render_report_summary
    from trufflepig.report_view import build_report_view

    channel = {"details": {"mismatch_repair": {"msi_probability": 0.9}}}
    monkeypatch.setattr(brief, "mismatch_repair_summary_context", lambda analysis: channel)
    analysis = colorectal_analysis()
    view = build_report_view(analysis, sample_id="synthetic-rna-proxy")
    content = build_report_content(analysis, pd.DataFrame(), "COAD", "", report_view=view)
    assert content.evidence_requests[0]["key"] == "msi_high"
    assert content.evidence_requests[0]["priority"] == "high"
    assert not any(
        a["selected"] for a in content.therapy_assessments if a["agent"] == "pembrolizumab"
    )
    analysis["clinical_context"] = context(assay("MSS")).public_dict()
    content = build_report_content(analysis, pd.DataFrame(), "COAD", "", report_view=view)
    assert not any(r["key"] == "msi_high" for r in content.evidence_requests)
    assert "Clinical/RNA discordance" in render_report_summary(content)
    assert not any(
        a["selected"] for a in content.therapy_assessments if a["agent"] == "pembrolizumab"
    )


def test_cli_context_round_trip(tmp_path, monkeypatch):
    from trufflepig import cli, main
    from trufflepig.analyze.models import AnalyzeConfig

    ctx = context(assay("dMMR"))
    path = tmp_path / "clinical.json"
    path.write_text(json.dumps(ctx.public_dict()))
    captured = {}

    def capture(**kwargs):
        captured.update(AnalyzeConfig(**kwargs).public_dict())

    monkeypatch.setattr(main, "analyze", capture)
    assert (
        cli.main(
            [
                "run",
                "--sample",
                str(tmp_path / "sample.tsv"),
                "--workspace",
                str(tmp_path / "out"),
                "--clinical-context",
                str(path),
            ]
        )
        == 0
    )
    assert captured["clinical_context"] == ctx.public_dict()


def test_web_upload_normalizes_the_same_context(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from trufflepig.web import WebSettings, create_app

    captured = {}

    def capture(cmd, log_path, status_path):
        captured["cmd"] = cmd

    monkeypatch.setattr("trufflepig.web.runs._spawn", capture)
    settings = WebSettings(runs_root=tmp_path / "runs", uploads_root=tmp_path / "uploads")
    client = TestClient(create_app(settings))
    ctx = context(assay())
    response = client.post(
        "/api/run",
        files={
            "sample": ("sample.tsv", b"gene\tTPM\n", "text/plain"),
            "clinical_context": ("context.json", json.dumps(ctx.public_dict()), "application/json"),
        },
    )
    assert response.status_code == 200, response.text
    cmd = captured["cmd"]
    path = Path(cmd[cmd.index("--clinical-context") + 1])
    assert load_clinical_context(path) == ctx
    assert "Clinical assay results" in client.get("/").text
    # A JSON scalar is data, not permission to load a server-side file.
    response = client.post(
        "/api/run",
        files={
            "sample": ("sample.tsv", b"gene\tTPM\n", "text/plain"),
            "clinical_context": ("context.json", json.dumps(str(path)), "application/json"),
        },
    )
    assert response.status_code == 400


def test_clinical_provenance_renders_literally(tmp_path):
    from pypdf import PdfReader
    from trufflepig.report_language import render_report_paragraph, report_plain_text
    from trufflepig.report_content import build_report_content, render_report_summary
    from trufflepig.report_document import write_report_document
    from trufflepig.report_pdf import build_interpretive_report_pdf
    from trufflepig.report_view import build_report_view

    source = ClinicalSource(
        title="Lab | amended report", excerpt="Literal A*02:01 / A*24:02; {{source}}"
    )
    ctx = context(assay(source=source, specimen_id="specimen|A"))
    decision = evaluate_msi_mmr(ctx)
    text = render_report_paragraph("clinical_assay_decision", decision=decision.public_dict())
    assert r"specimen\|A" in text
    assert "specimen|A" in report_plain_text(text)
    analysis = colorectal_analysis(ctx)
    view = build_report_view(analysis, sample_id="literal-clinical-source")
    content = build_report_content(analysis, pd.DataFrame(), "COAD", "", report_view=view)
    (tmp_path / "literal-clinical-source-summary.md").write_text(render_report_summary(content))
    write_report_document(tmp_path, "literal-clinical-source", report_view=view, content=content)
    pdf = PdfReader(build_interpretive_report_pdf(tmp_path))
    extracted = " ".join(" ".join(p.extract_text() for p in pdf.pages).split())
    assert source.title in extracted and source.excerpt in extracted
