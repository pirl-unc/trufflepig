"""Guards for the reader-facing interpretive-report PDF and its shared document.

The reader PDF captions each figure with an *interpretation sentence* (what the
figure means for the decision), not its PNG filename, and its figure manifest must
stay in sync with the curated patient packet. Preliminary labels, alternative
decomposition fits, and raw target surveys remain audit-only. The parsing/manifest
logic now lives in
:mod:`trufflepig.report_document` (shared with the pipeline so the PDF renders from
one serialized decision); these are cheap structural checks, and the rendering
itself is validated by eye against a real report.
"""

import importlib.util
import inspect
from pathlib import Path

import pytest

from trufflepig import report_document as rd
from trufflepig.report_view import build_report_view


def _load_pdf_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_interpretive_report_pdf.py"
    spec = importlib.util.spec_from_file_location("build_interpretive_report_pdf", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_figure_registry_entries_are_suffix_title_interpretation_triples():
    for entry in rd.FIGURE_REGISTRY:
        assert len(entry) == 3, f"expected (suffix, title, interpretation): {entry!r}"
        suffix, title, interpretation = entry
        assert suffix.endswith(".png")
        assert title and not title.endswith(".png")  # a title, never a filename
        assert interpretation and interpretation[0].isupper() and interpretation.rstrip().endswith(".")


def test_reader_manifest_keeps_final_analyses_and_excludes_preliminary_views():
    suffixes = {suffix for suffix, _, _ in rd.FIGURE_REGISTRY}
    assert {
        "sample-context.png",
        "decomposition-composition.png",
        "decomposition-components.png",
        "purity-methods.png",
    }.issubset(suffixes)
    assert {
        "sample-summary.png",
        "cancer-hypotheses.png",
        "cancer-type-signal-matrix.png",
        "decomposition-candidates.png",
        "background-tissues.png",
        "mhc-expression.png",
        "treatments.png",
        "purity-ctas.png",
        "purity-surface.png",
        "priority-targets.png",
        "priority-target-context.png",
        "actionable-targets.png",
    }.isdisjoint(suffixes)
    assert rd.is_patient_figure("sample-decomposition-composition.png")
    assert not rd.is_patient_figure("sample-decomposition-candidates.png")
    assert not rd.is_patient_figure("sample-cancer-type-signal-matrix.png")
    assert not rd.is_patient_figure("sample-prad-genes.png")


def test_figure_page_captions_with_interpretation_not_filename():
    b = _load_pdf_builder()
    params = list(inspect.signature(b._figure_page).parameters)
    assert params[:3] == ["path", "title", "interpretation"]


# --- §2.6b: the document parsers recover the therapy + priority-target tables ----
# The old scrape dropped every markdown table row on the floor. These pin that the
# shared document builder recovers the therapy shortlist and the top surface targets
# (with estimated patient attribution and healthy-reference context) straight from the
# report markdown, as structured {columns, rows} tables.

_ANALYSIS_MD = """\
# Analysis

## Therapy Prioritization

Some preamble sentence.

| Target | Agent | Class | Phase | Indication | Patient bulk TPM (measured) | Patient tumor-attributed TPM (estimated by RNA model) | Patient context TPM (RNA model estimate) | Interpretation |
|--------|-------|-------|-------|------------|----------|-------------------------------|---------------------|----------------|
| **CD274** | pembrolizumab | antibody | Approved | MSI-H / dMMR mCRC | 2.5 | 0 (0-0) | 0 | target expression is not the eligibility criterion — confirm MSI-H / dMMR status; approved antibody |
| **CEACAM5** | labetuzumab govitecan | ADC | Phase 2 | CEACAM5+ mCRC | 990.8 | 847 (353-895) | 991 | tumor-supported; vital-tissue concern; mid-clinical ADC |
| **EGFR** | cetuximab | antibody | Approved | RAS-WT mCRC | 11.5 | 0 (0-3) | 4 | background-dominant; approved antibody (2 agents) |
| **EGFR** | panitumumab | antibody | Approved | RAS-WT mCRC | 11.5 | 0 (0-3) | 4 | background-dominant; approved antibody (2 agents) |

*Footer note.*
"""

_EVIDENCE_MD = """\
# Evidence

### Surface Protein Targets (ADC / CAR-T / Bispecific)

Surface proteins with high context expression and source-attribution caveats.

| Gene | Patient context TPM (RNA model estimate) | Model interval | Patient bulk TPM (measured) | vs selected cancer reference | Pan-cancer reference %ile | Estimated background | Estimated attribution | Therapies |
|------|-----------|----------------|---------------------|---------|-----------|-----|-------------|-----------|
| **CEACAM5** | 991 | 991–991 | 991 | 0.10× | 94% | tissue-explainable | patient tumor-attributed 847 / T cell 0 (estimated by RNA model) | ADC, TCR-T |
| EPCAM | 825 | 825–825 | 825 | 0.12× | 88% | tissue-explainable | patient tumor-attributed 595 / endothelial 3 (estimated by RNA model) · broadly expr. |  |
| SLC2A1 | 784 | 654–1956 | 516 | 1.35× | 100% |  | patient tumor-attributed 493 / endothelial 9 (estimated by RNA model) · broadly expr. |  |
| BSG | 525 | 525–525 | 525 | 0.18× | 9% | background-dominant | patient tumor-attributed 204 / endothelial 38 (estimated by RNA model) |  |
"""

_SUMMARY_WITH_ATTR_MD = """\
# Sample coad_medoid

- **Cancer call:** COAD (Colon Adenocarcinoma)

## Top candidate therapies

**Where target RNA signal appears to come from**

| Gene | Patient bulk TPM (measured) | Patient tumor-attributed TPM (estimated by RNA model) | Estimated tumor fraction | Top estimated non-tumor contribution | Estimated component TPM | Main reason |
|---|---:|---:|---:|---|---:|---|
| CD274 | 2.49 | 0 | 0% | myeloid | 0.51 | RNA is context only |
| ERBB2 | 54.2 | 13.3 | 25% | endothelial | 1.56 | interval includes material tumor signal |
| BRAF | 9.2 | 1.33 | 14% | myeloid | 1.87 | RNA is context only |
"""

_SUMMARY_WITH_THERAPIES_MD = """\
# Sample coad_medoid

## Top candidate therapies

### Approved pathway / eligibility pending

- **CD274** — pembrolizumab (Approved, MSI-H / dMMR mCRC). target expression is not the eligibility criterion — confirm MSI-H / dMMR status; target RNA is context only (patient bulk 2.5 TPM, measured; it does not establish eligibility).
- **CEACAM5** — labetuzumab govitecan (Phase 2, CEACAM5+ mCRC). tumor-supported; 847 patient tumor-attributed TPM (estimated by RNA model; model interval 353-895); mid-clinical ADC.
- **EGFR** — cetuximab (Approved, RAS-WT mCRC). background-dominant; 0 patient tumor-attributed TPM (estimated by RNA model; model interval 0-3); approved antibody.
- **BRAF** — encorafenib (Approved, BRAF V600E mCRC). target expression is not the eligibility criterion; target RNA is context only (patient bulk 9.2 TPM, measured; it does not establish eligibility).
"""


def test_parse_markdown_tables_tags_sections_and_strips_separator(tmp_path):
    md = tmp_path / "x-analysis.md"
    md.write_text(_ANALYSIS_MD)
    tables = rd.parse_markdown_tables(md)
    assert len(tables) == 1
    table = tables[0]
    assert table["section"] == "Therapy Prioritization"
    assert table["headers"][0] == "Target"
    # the |---| alignment row is dropped; only real data rows survive
    assert len(table["rows"]) == 4
    assert table["rows"][0][0] == "CD274"  # markdown bold stripped


def test_therapy_recommendations_follow_summary_and_limit_to_three(tmp_path):
    summary = tmp_path / "s-summary.md"
    summary.write_text(_SUMMARY_WITH_THERAPIES_MD)
    table = rd.parse_therapy_recommendations(summary)
    cols, rows = table["columns"], table["rows"]
    assert [c[0] for c in cols] == [
        "Target",
        "Recommendation",
        "Estimated tumor TPM (RNA model)",
        "Eligibility / RNA provenance",
    ]
    assert [r[0] for r in rows] == ["CD274", "CEACAM5", "EGFR"]
    assert rows[0][1] == "pembrolizumab · Approved"
    assert rows[0][2] == "2.5 bulk; context only"
    # leading actionable clause only, not the whole eligibility tail
    assert rows[0][3].startswith("target expression is not the eligibility criterion")
    assert rows[1][3] == "tumor-supported"


def test_priority_target_table_reads_tumor_source_and_safety_band(tmp_path):
    (tmp_path / "s-evidence.md").write_text(_EVIDENCE_MD)
    table = rd._priority_target_table(tmp_path, "s")
    cols, rows = table["columns"], table["rows"]
    assert [c[0] for c in cols] == [
        "Target",
        "Estimated tumor TPM (RNA model)",
        "Healthy-tissue context (reference panel)",
        "Pan-cancer ref %ile",
    ]
    assert [r[0] for r in rows] == ["CEACAM5", "EPCAM", "SLC2A1"]  # top 3 by rank
    assert rows[0][1] == "847"  # tumor-source TPM parsed out of the attribution cell
    assert rows[0][2] == "tissue-explainable"  # TME safety band
    assert "broad normal expr." in rows[1][2]  # broadly-expressed cue appended
    assert rows[2][2].startswith(
        "not explained by one healthy-tissue reference"
    )
    assert rows[0][3] == "94%"


def test_priority_target_table_falls_back_to_summary_attribution(tmp_path):
    # No evidence.md: the summary's tumor-source attribution table is the fallback.
    (tmp_path / "s-summary.md").write_text(_SUMMARY_WITH_ATTR_MD)
    table = rd._priority_target_table(tmp_path, "s")
    cols, rows = table["columns"], table["rows"]
    assert [c[0] for c in cols][:2] == [
        "Target",
        "Estimated tumor TPM (RNA model)",
    ]
    assert [r[0] for r in rows] == ["CD274", "ERBB2", "BRAF"]
    assert rows[1][1] == "13.3"  # tumor-source bulk TPM column


def test_headline_cards_prefer_reportview_headline_over_records():
    b = _load_pdf_builder()
    # A ReportView-shape headline is present: the call + purity cards come from it,
    # not from the parsed markdown records — so the PDF headline reads the decision.
    doc = {
        "headline": {
            "cancer_type": "PRAD",
            "cancer_type_name": "Prostate adenocarcinoma",
            "purity": 0.10,
            "purity_lo": 0.06,
            "purity_hi": 0.16,
            "purity_confidence": "moderate",
        },
        "records": [
            {"section": "", "subsection": "", "label": "Cancer call", "value": "STALE", "text": ""},
            {"section": "", "subsection": "", "label": "Purity", "value": "99% stale", "text": ""},
        ],
    }
    call, purity = b._headline_cards(doc)
    assert call == "PRAD (Prostate adenocarcinoma)"
    assert purity == "10% (model interval 6%-16%, moderate confidence)"
    assert "STALE" not in call and "99%" not in purity


def test_headline_cards_require_the_structured_headline():
    b = _load_pdf_builder()
    doc = {
        "headline": {"cancer_type_name": "PRAD", "purity_display": "10% ..."},
        "records": [
            {"section": "", "subsection": "", "label": "Cancer call",
             "value": "PRAD (Prostate adenocarcinoma)", "text": ""},
            {"section": "", "subsection": "", "label": "Purity",
             "value": "10% (model interval 6%-16%, moderate confidence)", "text": ""},
        ],
    }
    with pytest.raises(ValueError, match="cancer_type"):
        b._headline_cards(doc)


def test_missing_tables_degrade_to_none_and_pdf_still_builds(tmp_path):
    b = _load_pdf_builder()
    # A report with no therapy/target tables (the local-sweep safety case): the
    # extractors return None and the PDF still renders from the document.
    summary = tmp_path / "s-summary.md"
    summary.write_text("# Sample s\n\n- **Cancer call:** SARC\n")
    assert rd.parse_therapy_recommendations(summary) is None
    assert rd._priority_target_table(tmp_path, "s") is None
    view = build_report_view(
        {
            "cancer_type": "SARC",
            "cancer_name": "Sarcoma",
            "sample_mode": "mesenchymal",
            "top_cancers": [("SARC", 1.0)],
            "purity": {
                "overall_estimate": 0.5,
                "overall_lower": 0.3,
                "overall_upper": 0.7,
            },
        },
        sample_id="s",
    )
    rd.write_report_document(tmp_path, "s", report_view=view)
    out = b.build_interpretive_report_pdf(tmp_path)
    assert out.exists() and out.stat().st_size > 0
