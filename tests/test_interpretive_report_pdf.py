"""Guards for the reader-facing interpretive-report PDF and its shared document.

The reader PDF captions each figure with an *interpretation sentence* (what the
figure means for the decision), not its PNG filename, and its figure manifest must
stay in sync with the figures the pipeline actually emits — a stale manifest
silently omitted the therapy figure (``treatments.png``) and shipped names that no
longer exist. The parsing/manifest logic now lives in
:mod:`trufflepig.report_document` (shared with the pipeline so the PDF renders from
one serialized decision); these are cheap structural checks, and the rendering
itself is validated by eye against a real report.
"""

import importlib.util
import inspect
from pathlib import Path

from trufflepig import report_document as rd


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


def test_reader_manifest_includes_therapy_and_excludes_near_duplicates():
    suffixes = {suffix for suffix, _, _ in rd.FIGURE_REGISTRY}
    # The ranked-therapy figure must be present (was silently dropped by the old
    # manifest, which looked only for priority-targets/actionable-targets).
    assert "treatments.png" in suffixes
    # The near-duplicate log-TPM dumbbell plots belong to the audit PDF only.
    assert "priority-target-context.png" not in suffixes
    assert "actionable-targets.png" not in suffixes


def test_figure_page_captions_with_interpretation_not_filename():
    b = _load_pdf_builder()
    params = list(inspect.signature(b._figure_page).parameters)
    assert params[:3] == ["path", "title", "interpretation"]


# --- §2.6b: the document parsers recover the therapy + priority-target tables ----
# The old scrape dropped every markdown table row on the floor. These pin that the
# shared document builder recovers the therapy shortlist and the top surface targets
# (with their tumor-source TPM and normal-tissue safety band) straight from the
# report markdown, as structured {columns, rows} tables.

_ANALYSIS_MD = """\
# Analysis

## Therapy Prioritization

Some preamble sentence.

| Target | Agent | Class | Phase | Indication | Bulk TPM (measured) | Tumor-source bulk TPM (model) | Context TPM (model) | Interpretation |
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

| Gene | Context TPM (model) | Model interval | Bulk TPM (measured) | vs ref | Ref %ile | TME | Attribution | Therapies |
|------|-----------|----------------|---------------------|---------|-----------|-----|-------------|-----------|
| **CEACAM5** | 991 | 991–991 | 991 | 0.10× | 94% | tissue-explainable | tumor 847 / T cell 0 | ADC, TCR-T |
| EPCAM | 825 | 825–825 | 825 | 0.12× | 88% | tissue-explainable | tumor 595 / endothelial 3 · broadly expr. |  |
| SLC2A1 | 784 | 654–1956 | 516 | 1.35× | 100% |  | tumor 493 / endothelial 9 · broadly expr. |  |
| BSG | 525 | 525–525 | 525 | 0.18× | 9% | background-dominant | tumor 204 / endothelial 38 |  |
"""

_SUMMARY_WITH_ATTR_MD = """\
# Sample coad_medoid

- **Cancer call:** COAD (Colon Adenocarcinoma)

## Top candidate therapies

**Where target RNA signal appears to come from**

| Gene | Bulk TPM | Tumor-source bulk TPM | Tumor fraction | Top non-tumor attribution | Component TPM | Main reason |
|---|---:|---:|---:|---|---:|---|
| CD274 | 2.49 | 0 | 0% | myeloid | 0.51 | RNA is context only |
| ERBB2 | 54.2 | 13.3 | 25% | endothelial | 1.56 | interval includes material tumor signal |
| BRAF | 9.2 | 1.33 | 14% | myeloid | 1.87 | RNA is context only |
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


def test_therapy_table_dedups_targets_and_limits_to_three(tmp_path):
    (tmp_path / "s-analysis.md").write_text(_ANALYSIS_MD)
    table = rd._therapy_table(tmp_path, "s")
    cols, rows = table["columns"], table["rows"]
    assert [c[0] for c in cols] == ["Target", "Agent / phase", "Tumor-src TPM", "Eligibility / RNA source"]
    # EGFR appears twice in the source but is deduped; capped at 3 rows.
    assert [r[0] for r in rows] == ["CD274", "CEACAM5", "EGFR"]
    assert rows[0][1] == "pembrolizumab · Approved"
    assert rows[0][2] == "0 (0-0)"  # tumor-source TPM column, not bulk
    # leading actionable clause only, not the whole eligibility tail
    assert rows[0][3] == "target expression is not the eligibility criterion"
    assert rows[1][3] == "tumor-supported"


def test_priority_target_table_reads_tumor_source_and_safety_band(tmp_path):
    (tmp_path / "s-evidence.md").write_text(_EVIDENCE_MD)
    table = rd._priority_target_table(tmp_path, "s")
    cols, rows = table["columns"], table["rows"]
    assert [c[0] for c in cols] == ["Target", "Tumor-src TPM", "Normal-tissue safety", "Cohort %ile"]
    assert [r[0] for r in rows] == ["CEACAM5", "EPCAM", "SLC2A1"]  # top 3 by rank
    assert rows[0][1] == "847"  # tumor-source TPM parsed out of the attribution cell
    assert rows[0][2] == "tissue-explainable"  # TME safety band
    assert "broad normal expr." in rows[1][2]  # broadly-expressed cue appended
    assert rows[2][2].startswith("tumor-enriched")  # empty TME → tumor-enriched
    assert rows[0][3] == "94%"


def test_priority_target_table_falls_back_to_summary_attribution(tmp_path):
    # No evidence.md: the summary's tumor-source attribution table is the fallback.
    (tmp_path / "s-summary.md").write_text(_SUMMARY_WITH_ATTR_MD)
    table = rd._priority_target_table(tmp_path, "s")
    cols, rows = table["columns"], table["rows"]
    assert [c[0] for c in cols][:2] == ["Target", "Tumor-src TPM"]
    assert [r[0] for r in rows] == ["CD274", "ERBB2", "BRAF"]
    assert rows[1][1] == "13.3"  # tumor-source bulk TPM column


def test_missing_tables_degrade_to_none_and_pdf_still_builds(tmp_path):
    b = _load_pdf_builder()
    # A report with no therapy/target tables (the local-sweep safety case): the
    # extractors return None and the PDF still renders from the document.
    (tmp_path / "s-summary.md").write_text("# Sample s\n\n- **Cancer call:** SARC\n")
    assert rd._therapy_table(tmp_path, "s") is None
    assert rd._priority_target_table(tmp_path, "s") is None
    out = b.build_interpretive_report_pdf(tmp_path)
    assert out.exists() and out.stat().st_size > 0
