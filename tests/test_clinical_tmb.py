"""Absolute clinical TMB and independent MSI/MMR decisions in the shared report."""

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from trufflepig.clinical_context import (
    ClinicalAssay, ClinicalContext, ClinicalMeasurement, ClinicalSource,
    TmbCriterion, clinical_assay_records, evaluate_msi_mmr, evaluate_tmb,
    load_clinical_context,
)
from trufflepig.therapy_eligibility import (
    collect_evidence_requests, evaluate_therapy_eligibility,
    tmb_criterion_for_therapy, tmb_requirement,
)


def tmb_assay(value=18, **changes):
    return ClinicalAssay(**{
        "kind": "tmb", "result": "measured", "method": "NGS",
        "measurement": ClinicalMeasurement(value, "mut/Mb"),
        "test_id": "FDA:P170019", "specimen_type": "tissue",
        "specimen_id": "specimen-A", "scope": "current",
        "validity": "validated", "reportability": "reportable",
        "reported_at": "2026-08-01",
        "source": ClinicalSource(title="Synthetic molecular pathology report"),
        **changes,
    })


def tmb_context(*assays):
    return ClinicalContext(specimen_id="specimen-A", assays=assays)


def tmb_row(**changes):
    return {
        "agent": "pembrolizumab", "agent_class": "antibody", "phase": "approved",
        "symbol": "PDCD1", "cancer_code": "COAD", "indication_biomarker": "tmb_high",
        "indication": "TMB-H solid tumors", "requires_verified_alteration": True,
        "clinical_setting_note": "Confirm unresectable or metastatic disease, progression after prior treatment, and no satisfactory alternatives.",
        **changes,
    }


def tmb_analysis(ctx):
    return {
        "cancer_type": "COAD", "cancer_type_source": "user-specified",
        "analysis_constraints": {"cancer_type": "COAD"},
        "sample_mode": "solid", "purity": {}, "clinical_context": ctx.public_dict(),
    }


@pytest.mark.parametrize("value,status", [(0, "negative"), (9.99, "negative"), (10, "positive"), (18, "positive")])
def test_absolute_tmb_uses_the_criterion_without_rounding_across_its_threshold(value, status):
    ctx = tmb_context(tmb_assay(value))
    decision = evaluate_tmb(ctx, criterion=tmb_criterion_for_therapy(tmb_row()))
    assert decision.status == status
    assert decision.assays[0]["measurement"]["value"] == value
    assert load_clinical_context(ctx.public_dict()) == ctx
    assert evaluate_msi_mmr(ctx).status == "missing"


@pytest.mark.parametrize("changes,limit", [
    ({"measurement": None}, "measurement is missing"),
    ({"measurement": ClinicalMeasurement(None, "mut/Mb")}, "measurement is missing"),
    ({"measurement": ClinicalMeasurement(99, "percentile")}, "mutations per megabase"),
    ({"measurement": ClinicalMeasurement(18, "")}, "mutations per megabase"),
    ({"result": "unknown"}, "result is unknown"),
    ({"result": "pending"}, "result is pending"),
    ({"result": "not_tested"}, "result is not_tested"),
    ({"result": "indeterminate"}, "result is indeterminate"),
    ({"scope": "historical"}, "historical"),
    ({"specimen_id": "specimen-B"}, "different specimen"),
    ({"test_id": ""}, "test identity"),
    ({"test_id": "FDA:P190032"}, "test identity"),
    ({"specimen_type": "plasma"}, "tissue specimen material"),
    ({"validity": "failed"}, "validity is failed"),
    ({"reportability": "unreportable"}, "unreportable"),
    ({"method": "RNA_expression"}, "clinical assay method"),
    ({"source": ClinicalSource()}, "source is missing"),
    ({"source": ClinicalSource(title="Proposed extraction", review_status="proposed")}, "proposed"),
])
def test_unsuitable_tmb_assertions_remain_visible_and_unresolved(changes, limit):
    ctx = tmb_context(tmb_assay(**changes))
    decision = evaluate_tmb(ctx, criterion=tmb_criterion_for_therapy(tmb_row()))
    assert decision.status == "unresolved"
    assert limit in " ".join(decision.assays[0]["limitations"])
    assert not evaluate_therapy_eligibility(tmb_row(), tmb_analysis(ctx)).permits_review
    assert load_clinical_context(ctx.public_dict()) == ctx


@pytest.mark.parametrize("value", [True, False, -1, float("inf"), float("nan"), "18", {}, []])
def test_invalid_numeric_measurements_are_rejected(value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        ClinicalMeasurement(value, "mut/Mb")


def test_tmb_units_are_normalized_without_converting_percentiles_or_counts():
    assert ClinicalMeasurement(0, "mutations/megabase") == ClinicalMeasurement(0, "mut/Mb")
    assert ClinicalMeasurement(99, "percentile").unit == "percentile"
    assert ClinicalMeasurement(99, "mutation count").unit == "mutation count"
    with pytest.raises(ValueError, match="currently belongs to a TMB"):
        ClinicalAssay(kind="msi", measurement=ClinicalMeasurement(18, "mut/Mb"))
    with pytest.raises(ValueError, match="Invalid TMB result"):
        tmb_assay(result="high")


def test_threshold_test_and_material_are_scoped_to_the_therapy_criterion():
    row = tmb_row()
    criterion = tmb_criterion_for_therapy(row)
    assert isinstance(criterion, TmbCriterion)
    stricter = replace(criterion, minimum=20)
    ctx = tmb_context(tmb_assay(18))
    assert evaluate_tmb(ctx, criterion=criterion).satisfied
    assert not evaluate_tmb(ctx, criterion=stricter).satisfied
    assert evaluate_therapy_eligibility({**row, "tmb_criterion": stricter.public_dict()}, tmb_analysis(ctx)).has_known_blocker
    assert tmb_criterion_for_therapy(tmb_row(agent="Keytruda")) == criterion
    assert tmb_criterion_for_therapy(tmb_row(agent="nivolumab")) is None
    assert tmb_criterion_for_therapy(tmb_row(agent="pembrolizumab + chemotherapy")) is None
    assert tmb_criterion_for_therapy(tmb_row(indication_biomarker="msi_high")) is None
    assert evaluate_tmb(ctx, criterion=None).status == "unresolved"
    for changes in ({"minimum": 0}, {"unit": "percentile"}, {"source": ""}, {"accepted_test_ids": ()}, {"specimen_type": ""}):
        with pytest.raises(ValueError):
            replace(criterion, **changes)


def test_conflicting_and_historical_tmb_results_keep_their_scope():
    criterion = tmb_criterion_for_therapy(tmb_row())
    positive, negative = tmb_assay(18), tmb_assay(3)
    ctx = tmb_context(positive, negative)
    assert evaluate_tmb(ctx, criterion=criterion).status == "conflicting"
    assert tmb_requirement(tmb_row(), tmb_analysis(ctx)).priority == "high"
    for changes in ({"scope": "historical"}, {"validity": "failed"}, {"specimen_id": "specimen-B"}):
        decision = evaluate_tmb(tmb_context(positive, replace(negative, **changes)), criterion=criterion)
        assert decision.satisfied
        assert len(decision.assays) == 2 and decision.assays[1]["limitations"]


def test_tmb_msi_and_prior_contraindication_are_independent():
    from trufflepig.reporting import direct_eligibility_evidence_supported

    msi = ClinicalAssay(kind="msi", result="MSS", method="PCR", specimen_id="specimen-A",
                        scope="current", validity="validated", reportability="reportable",
                        source=ClinicalSource(title="Synthetic MSI report"))
    ctx = tmb_context(tmb_assay(), msi)
    analysis = tmb_analysis(ctx)
    assert len(clinical_assay_records(ctx)) == 2
    assert evaluate_msi_mmr(ctx).status == "negative"
    assert len(evaluate_msi_mmr(ctx).assays) == 1
    assert evaluate_therapy_eligibility(tmb_row(), analysis).permits_review
    assert not evaluate_therapy_eligibility(tmb_row(indication_biomarker="msi_high"), analysis).permits_review
    assert not direct_eligibility_evidence_supported(analysis, "tmb_high")
    assert direct_eligibility_evidence_supported(analysis, "tmb_high", target_row=tmb_row())
    analysis["treatment_history"] = [{"therapy": "Keytruda", "status": "contraindicated"}]
    decision = evaluate_therapy_eligibility(tmb_row(), analysis)
    assert decision.direct_evidence_supported and decision.has_known_blocker
    assert not decision.permits_review


def test_msi_assay_wire_shape_and_id_remain_unchanged():
    assay = ClinicalAssay(kind="msi", result="MSI-H", method="PCR", specimen_id="specimen-A",
                          scope="current", validity="validated", reportability="reportable",
                          reported_at="2026-08-01", source=ClinicalSource(title="Synthetic clinical pathology report"))
    assert assay.id == "msi-1e1ac1a8c3b0b133"
    assert set(assay.public_dict()) == {"kind", "result", "method", "specimen_id", "scope", "collected_at",
                                        "reported_at", "validity", "reportability", "source", "protein_results", "id"}


def test_rna_untyped_tmb_and_pole_variants_cannot_supply_absolute_tmb():
    analysis = tmb_analysis(tmb_context())
    analysis.update(tmb=99, tmb_high=True, msi_status="MSI-H",
                    variant_records=[{"gene": "POLE", "variant": "p.P286R", "status": "positive"}])
    analysis["analysis_constraints"].update(tmb=99, tmb_units="mut/Mb")
    requirement = tmb_requirement(tmb_row(), analysis)
    assert requirement.status == "missing"
    assert not evaluate_therapy_eligibility(tmb_row(), analysis).permits_review


def test_one_tmb_request_preserves_different_treatment_thresholds():
    first = tmb_row()
    second = tmb_row(agent="Synthetic protocol", tmb_criterion=replace(tmb_criterion_for_therapy(first), minimum=20))
    requests = collect_evidence_requests([
        {"agent": row["agent"], "eligibility": evaluate_therapy_eligibility(row, tmb_analysis(tmb_context())).public_dict()}
        for row in (first, second)
    ])
    assert len(requests) == 1
    assert requests[0]["affects"] == ["pembrolizumab", "Synthetic protocol"]
    assert {e["criterion"]["minimum"] for e in requests[0]["evidence"]} == {10, 20}
    assert len(requests[0]["details"]) == 2


def test_cli_and_web_accept_the_same_tmb_context(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from trufflepig import cli, main
    from trufflepig.analyze.models import AnalyzeConfig
    from trufflepig.web import WebSettings, create_app

    ctx = tmb_context(tmb_assay(0))
    path = tmp_path / "clinical.json"
    path.write_text(json.dumps(ctx.public_dict()))
    captured = {}

    def capture_analysis(**kwargs):
        captured.update(AnalyzeConfig(**kwargs).public_dict())

    monkeypatch.setattr(main, "analyze", capture_analysis)
    assert cli.main(["run", "--sample", "synthetic.tsv", "--workspace", str(tmp_path / "out"),
                     "--clinical-context", str(path)]) == 0
    assert captured["clinical_context"] == ctx.public_dict()

    def capture_run(cmd, log_path, status_path):
        captured["cmd"] = cmd

    monkeypatch.setattr("trufflepig.web.runs._spawn", capture_run)
    client = TestClient(create_app(WebSettings(runs_root=tmp_path / "runs", uploads_root=tmp_path / "uploads")))
    response = client.post("/api/run", files={
        "sample": ("sample.tsv", b"gene\tTPM\n", "text/plain"),
        "clinical_context": ("context.json", json.dumps(ctx.public_dict()), "application/json"),
    })
    assert response.status_code == 200, response.text
    cmd = captured["cmd"]
    assert load_clinical_context(Path(cmd[cmd.index("--clinical-context") + 1])) == ctx


@pytest.mark.parametrize("value,result,expected", [(18, "measured", "satisfied"), (0, "measured", "blocked"), (None, "pending", "unresolved")])
def test_answering_tmb_request_updates_main_report_records_and_pdf(tmp_path, monkeypatch, value, result, expected):
    from pypdf import PdfReader
    import trufflepig.brief as brief
    from trufflepig.analyze.flow import write_analysis_output_records
    from trufflepig.analyze.models import AnalyzeConfig, AnalyzePaths, AnalyzeRun, InputResolution
    from trufflepig.report_content import build_report_content, render_report_summary
    from trufflepig.report_view import build_report_view

    row = tmb_row()
    monkeypatch.setattr(brief, "_curated_target_panel_for_sample", lambda *args, **kwargs: ("COAD", None, pd.DataFrame([row])))
    analysis = tmb_analysis(tmb_context())
    view = build_report_view(analysis, sample_id="synthetic-tmb")
    initial = build_report_content(analysis, pd.DataFrame(), "COAD", "", report_view=view)
    request = next(r for r in initial.evidence_requests if r["kind"] == "tmb_high")
    assert request["affects"] == ["pembrolizumab"]
    path = tmp_path / "clinical-context.json"
    ctx = tmb_context(tmb_assay(value, result=result))
    path.write_text(json.dumps(ctx.public_dict()))
    config = AnalyzeConfig(input_path="synthetic.tsv", clinical_context=path)
    analysis["clinical_context"] = config.clinical_context.public_dict()
    content = build_report_content(analysis, pd.DataFrame(), "COAD", "", report_view=view)
    assessment = content.therapy_assessments[0]
    requirement = next(r for r in assessment["eligibility"]["requirements"] if r["kind"] == "tmb_high")
    assert requirement["status"] == expected
    assert assessment["selected"] is (expected == "satisfied")
    followup = [r for r in content.evidence_requests if r["kind"] == "tmb_high"]
    assert bool(followup) is (expected == "unresolved")
    if followup:
        assert followup[0]["id"] == request["id"]
    assert not any(r["kind"] == "msi_high" for r in content.evidence_requests)
    summary = render_report_summary(content)
    assert "Clinical MSI/MMR evidence" not in summary
    if expected == "satisfied":
        assert "progression after prior treatment" in summary
    prefix = "synthetic-tmb"
    (tmp_path / f"{prefix}-summary.md").write_text(summary)
    run = AnalyzeRun(config, InputResolution("synthetic.tsv", None, False, "gene"), AnalyzePaths(tmp_path, prefix, prefix))
    write_analysis_output_records(run, view, content=content)
    document = json.loads((tmp_path / f"{prefix}-report.json").read_text())
    manifest = json.loads((tmp_path / f"{prefix}-manifest.json").read_text())
    assert document["clinical_context"] == manifest["config"]["clinical_context"] == ctx.public_dict()
    pdf = PdfReader(tmp_path / f"{prefix}-interpretive-report.pdf")
    text = " ".join(" ".join(p.extract_text() for p in pdf.pages).split())
    for phrase in ("TMB", "specimen-A", "FDA:P170019", "tissue", "Synthetic molecular pathology report", "2026-08-01"):
        assert phrase in summary and phrase in text
    if value is not None:
        assert f"{value} mut/Mb" in summary and f"{value} mut/Mb" in text
