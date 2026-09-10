"""Inspect actual native PDFs, including long rationale, HLA and source links."""

import json

import pytest
from pypdf import PdfReader
from PIL import Image

from trufflepig import report_document as rd
from trufflepig.report_pdf import build_interpretive_report_pdf, report_inline_html


def test_figure_registry_entries_render_complete_interpretations(tmp_path):
    from trufflepig.report_view import build_report_view

    view = build_report_view({"cancer_type": "SARC", "sample_mode": "solid", "purity": {}})
    for entry in rd.FIGURE_REGISTRY:
        assert len(entry) == 3, f"expected (suffix, title, interpretation): {entry!r}"
        suffix, title, interpretation = entry
        assert suffix.endswith(".png")
        assert title and not title.endswith(".png")  # a title, never a filename
        if interpretation is None:
            assert suffix == "purity-methods.png"
    for figure in rd.build_figure_manifest(tmp_path, "synthetic", purity=view.purity):
        interpretation = figure["caption"]
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


def document():
    return {
        "schema_version": 2,
        "prefix": "synthetic",
        "sample_id": "Synthetic evidence review",
        "sections": [
            {
                "id": "conclusion",
                "title": "Conclusion and supporting evidence",
                "blocks": [
                    {
                        "kind": "paragraph",
                        "text": "**Cancer call:** PRAD; low confidence, competing evidence remains.",
                    },
                    {
                        "kind": "paragraph",
                        "text": "Tumor fraction is unresolved; the operating model is not a measurement.",
                    },
                    {
                        "kind": "paragraph",
                        "text": "**HLA A*02:01 / A*24:02**; excluded group A*02:05P.",
                    },
                ],
            },
            {
                "id": "therapies",
                "title": "Therapy rationale and blockers",
                "blocks": [
                    {"kind": "heading", "text": "1. Historical therapy · Prior treatment"},
                    {
                        "kind": "paragraph",
                        "text": "Prior benefit supports clinical review. " * 150 + "RATIONALE-END",
                    },
                    {
                        "kind": "paragraph",
                        "text": "[Primary evidence](https://example.org/evidence)",
                    },
                ],
            },
            {
                "id": "information",
                "title": "Information needed",
                "blocks": [
                    {"kind": "bullet", "text": "Reconcile organ function and treatment history."},
                ],
            },
            {"id": "evidence", "title": "Detailed evidence and figures", "blocks": []},
        ],
    }


def write_document(tmp_path, doc):
    (tmp_path / "synthetic-summary.md").write_text("# STALE CLINICAL CLAIM")
    (tmp_path / "synthetic-report.json").write_text(json.dumps(doc))


def test_native_pdf_preserves_complete_rationale_hla_confidence_and_links(tmp_path):
    doc = document()
    write_document(tmp_path, doc)
    pdf = PdfReader(build_interpretive_report_pdf(tmp_path))
    text = "\n".join(page.extract_text() for page in pdf.pages)
    assert len(pdf.pages) >= 2
    for expected in (
        "RATIONALE-END",
        "A*02:01 / A*24:02",
        "A*02:05P",
        "Prior treatment",
        "low confidence",
        "fraction is unresolved",
    ):
        assert expected in text
    assert "STALE" not in text
    assert [text.index(s["title"]) for s in doc["sections"]] == sorted(
        text.index(s["title"]) for s in doc["sections"]
    )
    assert all(list(page.mediabox) == [0, 0, 612, 792] for page in pdf.pages)
    links = [
        annotation.get_object().get("/A", {}).get("/URI")
        for page in pdf.pages
        for annotation in page.get("/Annots", [])
    ]
    assert "https://example.org/evidence" in links


def test_figure_caption_and_image_are_rendered_with_authored_evidence(tmp_path):
    doc = document()
    doc["sections"][1]["blocks"] = []
    doc["sections"][-1]["blocks"] = [
        {
            "kind": "figure",
            "suffix": "purity-methods.png",
            "title": "Purity evidence",
            "caption": "Estimator disagreement limits interpretation.",
        }
    ]
    Image.new("RGB", (800, 400), "#bcd5e6").save(tmp_path / "synthetic-purity-methods.png")
    write_document(tmp_path, doc)
    pdf = PdfReader(build_interpretive_report_pdf(tmp_path))
    text = "\n".join(page.extract_text() for page in pdf.pages)
    assert "Estimator disagreement limits interpretation." in text
    assert "purity-methods.png" not in text
    assert sum(len(page.images) for page in pdf.pages) == 1


@pytest.mark.parametrize("filler_lines", [32, 35, 38, 41])
@pytest.mark.parametrize("long_body", [False, True])
def test_section_and_therapy_headings_stay_with_their_first_content(tmp_path, filler_lines, long_body):
    doc = document()
    doc["sections"] = [
        {"id": "conclusion", "title": "Conclusion", "blocks": [
            {"kind": "paragraph", "text": "\n".join(f"Evidence line {i}." for i in range(filler_lines))},
        ]},
        {"id": "therapies", "title": "Therapy rationale", "blocks": [
            {"kind": "heading", "text": "A conditional candidate"},
            {"kind": "paragraph", "text": "RATIONALE-START. " + "Supporting evidence. " * (300 if long_body else 20)},
        ]},
        {"id": "information", "title": "Information needed", "blocks": [
            {"kind": "paragraph", "text": "REQUEST-START. " + "Reconcile the supplied assay with the clinical report. " * 8},
        ]},
    ]
    write_document(tmp_path, doc)
    pdf = PdfReader(build_interpretive_report_pdf(tmp_path))
    pages = [page.extract_text() for page in pdf.pages]
    for heading, start in [
        ("Therapy rationale", "RATIONALE-START"),
        ("A conditional candidate", "RATIONALE-START"),
        ("Information needed", "REQUEST-START"),
    ]:
        page = next(text for text in pages if heading in text)
        assert start in page, f"Orphaned heading: {heading}"


def test_declared_figure_cannot_disappear_silently(tmp_path):
    doc = document()
    doc["sections"][-1]["blocks"] = [
        {
            "kind": "figure",
            "suffix": "missing.png",
            "title": "Evidence",
            "caption": "Required figure.",
        }
    ]
    write_document(tmp_path, doc)
    with pytest.raises(FileNotFoundError, match="declared present"):
        build_interpretive_report_pdf(tmp_path)


def test_legacy_document_requires_regeneration(tmp_path):
    write_document(tmp_path, {"schema_version": 1, "prefix": "synthetic"})
    with pytest.raises(ValueError, match="rerun analysis"):
        build_interpretive_report_pdf(tmp_path)


def test_html_is_text_and_hla_asterisks_are_not_emphasis():
    text = report_inline_html("<script>untrusted</script> **A*02:01 / A*24:02**")
    assert "&lt;script&gt;" in text
    assert "<b>A*02:01 / A*24:02</b>" in text
