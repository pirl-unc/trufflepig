"""The structured report document (§2.6b): the interpretive PDF renders from this
serialized decision, not by scraping the markdown, so figure/table/headline
content can't disagree with the reports. These tests pin the belief-gated figure
manifest, the ReportView-authoritative headline, cross-artifact parity, and the
load-vs-build-on-the-fly fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

from trufflepig import report_document as rd
from trufflepig.report_view import build_report_view

_SUMMARY = """# Summary

**Cancer call:** PRAD (Prostate adenocarcinoma).
**Purity:** 10% (model interval 6%–16%, moderate confidence).
**Sample:** exome capture; preservation inferred as FFPE.
**RNA quant QC:** salmon; 21k genes.
**Cancer-type basis:** RNA-inferred PRAD context.
**Mismatch-repair RNA context:** MMR ensemble favors proficient.
**Disease state:** castrate-resistant pattern.

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


def test_headline_falls_back_to_records_without_report_view(tmp_path):
    # Standalone build (no ReportView): the headline is derived from the parsed
    # summary records so the document is still self-describing.
    _write_reports(tmp_path)
    doc = rd.build_report_document(tmp_path, _PREFIX, report_view=None)
    assert doc["headline"]["cancer_type_name"].startswith("PRAD")
    assert "10%" in doc["headline"]["purity_display"]


def test_write_and_load_roundtrip(tmp_path):
    _write_reports(tmp_path)
    path = rd.write_report_document(tmp_path, _PREFIX, report_view=_report_view())
    assert path.name == f"{_PREFIX}-report.json"
    on_disk = json.loads(path.read_text())
    # load_report_document reads the sidecar verbatim when present.
    assert rd.load_report_document(tmp_path, _PREFIX) == on_disk
    assert on_disk["headline"]["purity"] == 0.10


def test_load_builds_on_the_fly_when_sidecar_absent(tmp_path):
    # Backward compatibility: an analyze dir produced before the sidecar existed
    # (reports present, no <prefix>-report.json) still yields a usable document.
    _write_reports(tmp_path)
    assert not (tmp_path / f"{_PREFIX}-report.json").exists()
    doc = rd.load_report_document(tmp_path)  # prefix auto-discovered
    assert doc["prefix"] == _PREFIX
    assert doc["therapy"]["rows"]
    # Without a written sidecar there is no ReportView, so the headline is the
    # records-derived fallback shape.
    assert "purity_display" in doc["headline"]


def test_empty_dir_therapy_and_targets_are_none(tmp_path):
    # No therapy/target tables in the reports -> null table sections, not a crash.
    (tmp_path / f"{_PREFIX}-summary.md").write_text("# Summary\n\n**Cancer call:** UNKNOWN.\n")
    (tmp_path / f"{_PREFIX}-analysis.md").write_text("# Analysis\n")
    doc = rd.build_report_document(tmp_path, _PREFIX, report_view=None)
    assert doc["therapy"] is None
    assert doc["targets"] is None
    assert all(f["present"] is False for f in doc["figures"])
