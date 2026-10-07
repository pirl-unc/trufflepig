"""The structured report document (§2.6b): the interpretive PDF renders from this
serialized decision, not by scraping the markdown, so figure/table/headline
content can't disagree with the reports. These tests pin the belief-gated figure
manifest, the ReportView-authoritative headline, cross-artifact parity, and the
structured sidecar contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trufflepig import report_document as rd
from trufflepig.report_content import ReportContent
from trufflepig.report_language import report_plain_text
from trufflepig.analyze import (
    AnalyzeConfig,
    AnalyzePaths,
    AnalyzeRun,
    InputResolution,
    write_analysis_output_records,
)
from trufflepig.report_view import build_report_view


@pytest.mark.parametrize(
    "text, expected",
    [
        ("HLA A*02:01 / A*24:02", "HLA A*02:01 / A*24:02"),
        ("**HLA A*02:01 / A*24:02**", "HLA A*02:01 / A*24:02"),
        ("*HLA A*02:01 / A*24:02*", "HLA A*02:01 / A*24:02"),
        ("`HLA A*02:01 / A*24:02`", "HLA A*02:01 / A*24:02"),
        ("**A*02:01P**; A*02:05P", "A*02:01P; A*02:05P"),
        ("[A*02:01:01:01N](https://example.org)", "A*02:01:01:01N"),
        (r"\*literal\* and A\*02:01", "*literal* and A*02:01"),
        ("**Important:** A*02:01 is *required*.", "Important: A*02:01 is required."),
    ],
)
def test_markdown_preserves_hla_literals_and_genuine_formatting(text, expected):
    assert report_plain_text(text) == expected


_PREFIX = "sampleX"


def _content():
    return ReportContent(
        _PREFIX,
        [
            {
                "id": "conclusion",
                "title": "Conclusion and supporting evidence",
                "blocks": [
                    {
                        "kind": "paragraph",
                        "text": "**Cancer call:** PRAD. Low confidence; competing evidence remains.",
                    },
                    {"kind": "paragraph", "text": "**Supplied HLA:** A*02:01 / A*24:02"},
                ],
            },
            {
                "id": "therapies",
                "title": "Therapy rationale and blockers",
                "blocks": [
                    {
                        "kind": "paragraph",
                        "text": "Prior treatment benefit supports review; current eligibility remains unresolved.",
                    },
                ],
            },
            {"id": "information", "title": "Information needed", "blocks": []},
            {"id": "evidence", "title": "Detailed evidence and figures", "blocks": []},
        ],
        [{"agent": "FAP radioligand", "selected": True, "rationale": ["major prior benefit"]}],
        [{"key": "hla_typing", "affects": ["afami-cel"]}],
        None,
        [],
    )


def _write_reports(tmp_path, *, emit_figures=("sample-context", "purity-methods")):
    # Stale Markdown must have no influence on the authoritative document.
    (tmp_path / f"{_PREFIX}-summary.md").write_text("# Stale call and recommendations")
    for suffix in emit_figures:
        (tmp_path / f"{_PREFIX}-{suffix}.png").write_bytes(b"\x89PNG")
    return tmp_path


def _report_view():
    return build_report_view(
        {
            "cancer_type": "PRAD",
            "cancer_name": "Prostate adenocarcinoma",
            "purity": {"overall_estimate": 0.10, "overall_lower": 0.06, "overall_upper": 0.16},
            "top_cancers": [("PRAD", 1.0), ("COAD", 0.4)],
            "sample_mode": "solid",
        },
        sample_id=_PREFIX,
    )


def test_document_preserves_authored_content_without_reading_markdown(tmp_path):
    _write_reports(tmp_path)
    content = _content()
    doc = rd.build_report_document(tmp_path, _PREFIX, report_view=_report_view(), content=content)
    assert doc["schema_version"] == 2
    assert doc["headline"]["cancer_type"] == "PRAD"
    assert doc["headline"]["purity"] == 0.10
    assert doc["therapy_assessments"] == content.therapy_assessments
    assert doc["evidence_requests"] == content.evidence_requests
    assert "A*02:01 / A*24:02" in doc["sections"][0]["blocks"][1]["text"]
    assert [s["id"] for s in doc["sections"]] == [
        "conclusion",
        "therapies",
        "information",
        "evidence",
    ]
    assert content.sections[-1]["blocks"] == []  # serialization does not mutate content
    assert "Stale" not in json.dumps(doc)


def test_required_content_cannot_fall_back_to_markdown(tmp_path):
    _write_reports(tmp_path)
    with pytest.raises(TypeError, match="content"):
        rd.build_report_document(tmp_path, _PREFIX, report_view=_report_view())


def test_write_load_preserves_long_rationale_and_supplied_history(tmp_path):
    content = _content()
    long = "Evidence and limitations. " * 150 + "RATIONALE-END"
    content.sections[1]["blocks"][0]["text"] = long
    content.treatment_history = [
        {"therapy": "FAP-2286", "status": "major_benefit", "source": "oncology assessment"}
    ]
    path = rd.write_report_document(tmp_path, _PREFIX, report_view=_report_view(), content=content)
    doc = rd.load_report_document(tmp_path, _PREFIX)
    assert doc == json.loads(path.read_text())
    assert doc["sections"][1]["blocks"][0]["text"] == long
    assert doc["treatment_history"] == content.treatment_history


def test_load_requires_structured_document(tmp_path):
    _write_reports(tmp_path)
    with pytest.raises(FileNotFoundError, match="rerun analysis"):
        rd.load_report_document(tmp_path)


def test_figure_manifest_is_belief_gated(tmp_path):
    # Only two figures emitted; the rest of the registry must be present=False.
    _write_reports(tmp_path, emit_figures=("sample-context", "purity-methods"))
    doc = rd.build_report_document(
        tmp_path, _PREFIX, report_view=_report_view(), content=_content()
    )
    figures = {f["suffix"]: f for f in doc["figures"]}

    # Every registry entry appears in the manifest.
    assert set(figures) == {suffix for suffix, _, _ in rd.FIGURE_REGISTRY}
    # The two emitted plots are present with a resolved path + a caption.
    assert figures["sample-context.png"]["present"] is True
    assert figures["sample-context.png"]["path"] == f"{_PREFIX}-sample-context.png"
    assert figures["sample-context.png"]["caption"]
    # A plot the run never emitted (belief never fired) is gated out with no path.
    assert figures["therapy-pathway-state.png"]["present"] is False
    assert figures["therapy-pathway-state.png"]["path"] is None


def test_document_preserves_unresolved_purity_and_caveats_figure_captions(tmp_path):
    _write_reports(tmp_path)
    view = build_report_view(
        {
            "cancer_type": "READ",
            "cancer_name": "Rectum Adenocarcinoma",
            "purity": {
                "overall_estimate": 0.05,
                "overall_lower": 0.01,
                "overall_upper": 0.12,
                "quantitative_status": "discordant_estimators",
                "estimator_scenarios": [
                    {
                        "source": "lineage_panel",
                        "estimate": 0.05,
                        "lower": 0.01,
                        "upper": 0.12,
                    },
                    {
                        "source": "signature",
                        "estimate": 0.43,
                        "lower": 0.32,
                        "upper": 0.55,
                    },
                ],
            },
            "top_cancers": [("READ", 1.0)],
            "sample_mode": "solid",
        }
    )

    doc = rd.build_report_document(tmp_path, _PREFIX, report_view=view, content=_content())
    figures = {row["suffix"]: row for row in doc["figures"]}

    assert doc["headline"]["purity_status"] == "discordant_estimators"
    assert doc["headline"]["purity_scenarios"][1] == (
        "signature",
        0.43,
        0.32,
        0.55,
    )
    assert (
        "not a resolved sample-composition measurement"
        in figures["decomposition-composition.png"]["caption"]
    )
    assert "not a fused consensus estimate" in figures["purity-methods.png"]["caption"]


def test_output_finalization_writes_structured_records_without_figures(tmp_path):
    _write_reports(tmp_path, emit_figures=())
    history = [
        {
            "therapy": "FAP-targeted radioligand therapy",
            "target": "FAP",
            "modality": "RLT",
            "status": "major_benefit",
            "note": "Very effective",
            "source": "clinical history",
        }
    ]
    run = AnalyzeRun(
        config=AnalyzeConfig(input_path="sample.tsv", output_dir=str(tmp_path)),
        inputs=InputResolution(
            gene_input="sample.tsv",
            transcript_input=None,
            aggregate_gene_expression=False,
            input_level="gene",
        ),
        paths=AnalyzePaths(
            out_dir=tmp_path,
            prefix_base=_PREFIX,
            sample_display_id=_PREFIX,
        ),
    )
    run.note_step("input", outputs={"treatment_history": history})

    content = _content()
    content.treatment_history = history
    outputs = write_analysis_output_records(run, _report_view(), content=content)

    report_path = Path(outputs["report_document"])
    manifest_path = Path(outputs["manifest"])
    assert report_path.name == f"{_PREFIX}-report.json"
    assert report_path.exists()
    assert Path(outputs["report_pdf"]).is_file()
    assert manifest_path.name == f"{_PREFIX}-manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["steps"]["output"]["outputs"]["report_document"] == str(report_path)
    assert json.loads(report_path.read_text())["treatment_history"] == history
    assert manifest["steps"]["output"]["outputs"]["report_pdf"] == outputs["report_pdf"]
    assert not any(artifact["kind"] == "figure" for artifact in manifest["artifacts"])
