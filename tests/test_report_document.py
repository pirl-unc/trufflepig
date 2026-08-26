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
from trufflepig.analyze import (
    AnalyzeConfig,
    AnalyzePaths,
    AnalyzeRun,
    InputResolution,
    write_analysis_output_records,
)
from trufflepig.report_view import build_report_view

_SUMMARY = """# Summary

**Cancer call:** PRAD (Prostate adenocarcinoma).
**Purity:** 10% (model interval 6%–16%, moderate confidence).
**Sample:** exome capture; preservation inferred as FFPE.
**RNA quant QC:** salmon; 21k genes.
**Cancer-type basis:** RNA-inferred PRAD context.
**Mismatch-repair RNA context:** MMR ensemble favors proficient.
**Disease state:** castrate-resistant pattern.

## Top candidate therapies

### Approved / disease-matched

- **FOLH1** — lutetium-177 PSMA (Approved, mCRPC). tumor-supported; 128 tumor-source bulk TPM (model interval 100-150); guideline-standard approved pathway.
- **AR** — enzalutamide (Approved, mCRPC). mixed-source; 48 tumor-source bulk TPM (model interval 40-50); guideline-standard approved pathway.

## Notable biomarker outliers

- FOLH1 - amplified, top 2%.

## Caveats

- Confirm MSI-H / dMMR status before immunotherapy.
"""

_ANALYSIS = """# Analysis

## Therapy Prioritization

| Target | Agent | Class | Phase | Indication | Bulk TPM (measured) | Tumor-source bulk TPM (model) | Context TPM (model) | Interpretation |
|--------|-------|-------|-------|------------|----------|-------------------------------|---------------------|----------------|
| **FOLH1** | lu-psma | radioligand | approved | mCRPC | 142.0 | tumor 128 | 12 | tumor-supported; approved standard |
| AR | enzalutamide | ARSI | approved | mCRPC | 50.0 | tumor 48 | 2 | tumor-supported |
"""

_EVIDENCE = """# Evidence

### Surface Protein Targets

| Gene | Value | Model interval | Bulk TPM | vs ref | Ref %ile | TME | Attribution | Therapies |
|------|-------|----------------|----------|--------|----------|-----|-------------|-----------|
| FOLH1 | 128 | 100-150 | 142 | +3 | 97 | tumor-enriched | tumor 128 / endothelial 12 - broadly expr. | lu-psma |
"""

_PREFIX = "sampleX"


def _write_reports(tmp_path: Path, *, emit_figures=("sample-summary", "purity-methods")) -> Path:
    (tmp_path / f"{_PREFIX}-summary.md").write_text(_SUMMARY)
    (tmp_path / f"{_PREFIX}-analysis.md").write_text(_ANALYSIS)
    (tmp_path / f"{_PREFIX}-evidence.md").write_text(_EVIDENCE)
    for suffix in emit_figures:
        # A one-byte stand-in is enough: the manifest gates on file presence, not content.
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


def test_document_carries_headline_records_tables_and_figures(tmp_path):
    _write_reports(tmp_path)
    doc = rd.build_report_document(tmp_path, _PREFIX, report_view=_report_view())

    assert doc["schema_version"] == rd.SCHEMA_VERSION
    assert doc["prefix"] == _PREFIX
    # Headline is taken from the frozen ReportView, not scraped.
    assert doc["headline"]["cancer_type"] == "PRAD"
    assert doc["headline"]["purity"] == 0.10
    assert doc["headline"]["purity_lo"] == 0.06 and doc["headline"]["purity_hi"] == 0.16
    # Tables are structured {columns, rows}.
    assert [c[0] for c in doc["therapy"]["columns"]][0] == "Target"
    assert len(doc["therapy"]["rows"]) == 2
    assert doc["targets"]["rows"] and doc["targets"]["rows"][0][0] == "FOLH1"
    # At-a-glance records are present.
    assert rd.record_value(doc["records"], "Cancer call").startswith("PRAD")


def test_document_deduplicates_repeated_summary_highlights(tmp_path):
    _write_reports(tmp_path)
    repeated = (
        "Technical-RNA normalization: mtDNA/rRNA-like features were removed "
        "for reference comparability."
    )
    summary = tmp_path / f"{_PREFIX}-summary.md"
    summary.write_text(summary.read_text() + f"\n- {repeated}\n- {repeated}\n")

    doc = rd.build_report_document(tmp_path, _PREFIX, report_view=_report_view())

    assert doc["highlights"].count(repeated) == 1


def test_figure_manifest_is_belief_gated(tmp_path):
    # Only two figures emitted; the rest of the registry must be present=False.
    _write_reports(tmp_path, emit_figures=("sample-summary", "purity-methods"))
    doc = rd.build_report_document(tmp_path, _PREFIX, report_view=_report_view())
    figures = {f["suffix"]: f for f in doc["figures"]}

    # Every registry entry appears in the manifest.
    assert set(figures) == {suffix for suffix, _, _ in rd.FIGURE_REGISTRY}
    # The two emitted plots are present with a resolved path + a caption.
    assert figures["sample-summary.png"]["present"] is True
    assert figures["sample-summary.png"]["path"] == f"{_PREFIX}-sample-summary.png"
    assert figures["sample-summary.png"]["caption"]
    # A plot the run never emitted (belief never fired) is gated out with no path.
    assert figures["cancer-type-signal-matrix.png"]["present"] is False
    assert figures["cancer-type-signal-matrix.png"]["path"] is None


def test_headline_purity_agrees_with_parsed_purity_record(tmp_path):
    # Cross-artifact parity: the structural ReportView headline the PDF cards can
    # read must not disagree with the "Purity" line the markdown rendered.
    _write_reports(tmp_path)
    doc = rd.build_report_document(tmp_path, _PREFIX, report_view=_report_view())
    purity_record = rd.record_value(doc["records"], "Purity")
    assert f"{round(doc['headline']['purity'] * 100)}%" in purity_record  # "10%"


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

    doc = rd.build_report_document(tmp_path, _PREFIX, report_view=view)
    figures = {row["suffix"]: row for row in doc["figures"]}

    assert doc["headline"]["purity_status"] == "discordant_estimators"
    assert doc["headline"]["purity_scenarios"][1] == (
        "signature",
        0.43,
        0.32,
        0.55,
    )
    assert "no consensus tumor/non-tumor fraction" in figures[
        "sample-summary.png"
    ]["caption"]
    assert "not a fused consensus estimate" in figures["purity-methods.png"][
        "caption"
    ]


def test_write_and_load_roundtrip(tmp_path):
    _write_reports(tmp_path)
    path = rd.write_report_document(tmp_path, _PREFIX, report_view=_report_view())
    assert path.name == f"{_PREFIX}-report.json"
    on_disk = json.loads(path.read_text())
    # load_report_document reads the sidecar verbatim when present.
    assert rd.load_report_document(tmp_path, _PREFIX) == on_disk
    assert on_disk["headline"]["purity"] == 0.10


def test_output_finalization_writes_structured_records_without_figures(tmp_path):
    _write_reports(tmp_path, emit_figures=())
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

    outputs = write_analysis_output_records(run, _report_view())

    report_path = Path(outputs["report_document"])
    manifest_path = Path(outputs["manifest"])
    assert report_path.name == f"{_PREFIX}-report.json"
    assert report_path.exists()
    assert manifest_path.name == f"{_PREFIX}-manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["steps"]["output"]["outputs"]["report_document"] == str(
        report_path
    )
    assert not any(
        artifact["kind"] == "figure" for artifact in manifest["artifacts"]
    )


def test_load_requires_structured_sidecar(tmp_path):
    _write_reports(tmp_path)
    assert not (tmp_path / f"{_PREFIX}-report.json").exists()
    with pytest.raises(FileNotFoundError, match="rerun analysis"):
        rd.load_report_document(tmp_path)


def test_empty_dir_therapy_and_targets_are_none(tmp_path):
    # No therapy/target tables in the reports -> null table sections, not a crash.
    (tmp_path / f"{_PREFIX}-summary.md").write_text("# Summary\n\n**Cancer call:** UNKNOWN.\n")
    (tmp_path / f"{_PREFIX}-analysis.md").write_text("# Analysis\n")
    doc = rd.build_report_document(tmp_path, _PREFIX, report_view=_report_view())
    assert doc["therapy"] is None
    assert doc["targets"] is None
    assert all(f["present"] is False for f in doc["figures"])


# The detailed analysis contains a broader therapy landscape than the reader
# summary. The structured document must reproduce the summary decision exactly
# rather than independently selecting from the broader table.
_ANALYSIS_AUDIT_SPLIT = """# Analysis

## Therapy Prioritization

### Sample-supported / clinically reviewable rows

| Target | Agent | Class | Phase | Indication | Bulk TPM (measured) | Tumor-source bulk TPM (model) | Context TPM (model) | Interpretation |
|--------|-------|-------|-------|------------|----------|-------------------------------|---------------------|----------------|
| **FOLH1** | lu-psma | radioligand | approved | mCRPC | 142.0 | tumor 128 | 12 | tumor-supported; approved standard |

### Other curated rows — not supported by this sample

These rows remain visible as disease-curation provenance or negative evidence.

| Target | Agent | Class | Phase | Indication | Bulk TPM (measured) | Tumor-source bulk TPM (model) | Context TPM (model) | Interpretation |
|--------|-------|-------|-------|------------|----------|-------------------------------|---------------------|----------------|
| **FGFR3** | erdafitinib | FGFR-inhibitor | approved | urothelial | 10.6 | tumor 0 (0-0) | 1 | not sample-supported; negative/background evidence; background-dominant |
"""


def test_therapy_recommendations_follow_summary_not_broader_analysis(tmp_path):
    (tmp_path / f"{_PREFIX}-summary.md").write_text(_SUMMARY)
    (tmp_path / f"{_PREFIX}-analysis.md").write_text(_ANALYSIS_AUDIT_SPLIT)
    (tmp_path / f"{_PREFIX}-evidence.md").write_text(_EVIDENCE)

    table = rd.parse_therapy_recommendations(
        tmp_path / f"{_PREFIX}-summary.md"
    )
    assert table is not None
    targets = {row[0] for row in table["rows"]}
    assert targets == {"FOLH1", "AR"}
    assert "FGFR3" not in targets
    assert "erdafitinib" not in {cell for row in table["rows"] for cell in row}
    assert table["rows"][0][1] == "lutetium-177 PSMA · Approved"
    assert table["rows"][0][2] == "128 (100-150)"


_ANALYSIS_ALL_AUDIT = """# Analysis

## Therapy Prioritization

### Sample-supported / clinically reviewable rows

*No curated therapy row had tumor-supported or clinically reviewable RNA evidence in this sample.*

### Other curated rows — not supported by this sample

| Target | Agent | Class | Phase | Indication | Bulk TPM (measured) | Tumor-source bulk TPM (model) | Context TPM (model) | Interpretation |
|--------|-------|-------|-------|------------|----------|-------------------------------|---------------------|----------------|
| **FGFR3** | erdafitinib | FGFR-inhibitor | approved | urothelial | 10.6 | tumor 0 (0-0) | 1 | not sample-supported; negative/background evidence; background-dominant |
"""


def test_therapy_recommendations_are_none_when_summary_shortlist_is_empty(tmp_path):
    summary = """# Summary

## Top candidate therapies

*Therapy shortlist is empty: no curated row qualified.*
"""
    (tmp_path / f"{_PREFIX}-summary.md").write_text(summary)
    (tmp_path / f"{_PREFIX}-analysis.md").write_text(_ANALYSIS_ALL_AUDIT)
    assert rd.parse_therapy_recommendations(
        tmp_path / f"{_PREFIX}-summary.md"
    ) is None
