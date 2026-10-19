"""Report statements must match the method evidence shown in the figure."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest

from trufflepig.report_document import build_figure_manifest
from trufflepig.report_language import render_report_paragraph
from trufflepig.report_view import build_report_view, purity_method_estimates
from trufflepig.tumor_purity import plot_purity_method_comparison


def analysis(purity):
    return {"cancer_type": "SARC", "sample_mode": "solid", "purity": purity}


def test_method_collection_preserves_the_plotted_calibration_and_sources():
    result = {
        "reference_expression_source": "pan_cancer", "tcga_median_purity": 0.5,
        "components": {
            "signature": {"purity": 0.3, "lower": 0.2, "upper": 0.4, "genes": ["A", "B"]},
            "lineage": {"purity": 0.32, "lower": 0.21, "upper": 0.43, "genes": ["C"]},
            "integration": {"signature_deprioritized": True},
            "decomposition": {"residual_fraction": 0.34, "mode": "mesenchymal"},
            "stromal": {"enrichment": 2.0, "n_genes": 141},
            "immune": {"enrichment": 0.0, "n_genes": 140},
            "estimate_purity": 0.35,
        },
    }
    original = deepcopy(result)
    rows = purity_method_estimates(result, SimpleNamespace(purity=0.8, warnings=[]))
    assert [r.estimate for r in rows] == pytest.approx([0.3, 0.32, 0.34, 1 / 3, 1.0, 0.35])
    assert [r.family for r in rows] == ["signature", "lineage", "decomposition", "estimate", "estimate", "estimate"]
    assert (rows[0].lower, rows[0].upper, rows[0].genes, rows[0].note) == (0.2, 0.4, 2, " (deprioritized)")
    assert rows[2].note == " [mesenchymal]" and rows[3].genes == 141
    assert result == original


@pytest.mark.parametrize("source, count", [
    ("pan_cancer", 2), ("parent_pan_cancer", 2), ("member_union_pan_cancer", 2),
    ("subtype_deconvolved", 0), ("", 0),
])
def test_tumor_only_reference_does_not_supply_quantitative_lineage_or_estimate_rows(source, count):
    rows = purity_method_estimates({
        "reference_expression_source": source, "tcga_median_purity": 0.5,
        "components": {"lineage": {"purity": 0.4}, "stromal": {"enrichment": 2.0}},
    })
    assert len(rows) == count


@pytest.mark.parametrize("value", [None, False, True, float("nan"), float("inf"), -0.1, 1.1, "unknown"])
def test_invalid_component_values_cannot_invent_an_estimator(value):
    assert purity_method_estimates({
        "reference_expression_source": "pan_cancer",
        "components": {"signature": {"purity": value}, "lineage": {"purity": value},
                       "decomposition": {"residual_fraction": value}, "estimate_purity": value},
    }) == ()


def test_zero_method_estimate_is_retained():
    method, = purity_method_estimates({"components": {"signature": {"purity": 0.0}}})
    assert method.estimate == 0.0


def test_unresolved_decomposition_warning_does_not_invent_a_method():
    result = SimpleNamespace(purity=1.0, warnings=["No non-tumor components resolved"])
    assert purity_method_estimates({}, result) == ()


def test_frozen_methods_drive_figure_and_caption_after_mutation(tmp_path):
    result = {"overall_estimate": 0.35, "overall_lower": 0.27, "overall_upper": 0.44,
              "purity_source": "lineage", "components": {"signature": {"purity": 0.09},
              "decomposition": {"residual_fraction": 0.97}}}
    view = build_report_view(analysis(result))
    result["components"]["signature"]["purity"] = 0.78
    result["components"]["decomposition"]["residual_fraction"] = 0.81
    result["purity_source"] = "stale source"
    with pytest.raises(FrozenInstanceError):
        view.purity.methods[0].estimate = 0.7
    fig = plot_purity_method_comparison(result, report_view=view)
    try:
        assert [t.get_text() for t in fig.axes[0].texts] == ["9%", "97%", "35%"]
        assert "basis: lineage" in fig.axes[0].get_title(loc="left")
    finally:
        plt.close(fig)
    caption = next(f["caption"] for f in build_figure_manifest(tmp_path, "synthetic", purity=view.purity)
                   if f["suffix"] == "purity-methods.png")
    assert "2 displayed" in caption and "9%–97%" in caption
    assert "does not encompass the full spread" in caption
    assert "not independent measurements" in caption
    assert view.public_dict()["purity_methods"][0]["estimate"] == 0.09
    summary = render_report_paragraph("purity_summary", purity=view.purity)
    assert "27%–44%" in summary and "9%–97%" in summary


def test_adopted_value_is_not_counted_as_another_method():
    result = {"overall_estimate": 0.79, "overall_lower": 0.49, "overall_upper": 0.94,
              "components": {"decomposition": {"residual_fraction": 0.79}}}
    view = build_report_view(analysis(result))
    assert len(view.purity.methods) == 1
    caption = render_report_paragraph("purity_methods", purity=view.purity)
    assert "One RNA-derived method estimate" in caption
    assert "agreement between methods cannot be assessed" in caption
    assert "not an additional method" in caption
    assert "Only one method estimate" in render_report_paragraph("purity_summary", purity=view.purity)


def test_absent_method_evidence_does_not_manufacture_agreement():
    view = build_report_view(analysis({"overall_estimate": 0.5}))
    caption = render_report_paragraph("purity_methods", purity=view.purity)
    assert "No method-level estimates are available" in caption
    assert not view.purity.methods


def test_missing_purity_is_consistent_in_prose_and_json():
    view = build_report_view(analysis({}))
    assert view.public_dict()["purity"] is None
    assert view.public_dict()["purity_confidence"] == "unknown"
    summary = render_report_paragraph("purity_summary", purity=view.purity, show_confidence_reasons=True)
    assert "unavailable; no quantitative estimate" in summary
    assert "deterministic" not in summary and "no estimated spread" not in summary


def test_point_without_bounds_reports_the_missing_interval():
    view = build_report_view(analysis({"overall_estimate": 0.4}))
    assert "model interval unavailable; unknown confidence" in render_report_paragraph("purity_summary", purity=view.purity)
    assert "without a complete uncertainty interval" in render_report_paragraph("purity_attribution", purity=view.purity)


@pytest.mark.parametrize("value", [None, False, float("nan"), "unknown"])
def test_standalone_figure_omits_unavailable_adopted_and_reference_values(value):
    result = {"overall_estimate": value, "overall_lower": value, "overall_upper": value,
              "tcga_median_purity": value, "reference_expression_source": "pan_cancer",
              "components": {"signature": {"purity": 0.3}}}
    fig = plot_purity_method_comparison(result)
    try:
        assert [t.get_text() for t in fig.axes[0].texts] == ["30%"]
        assert not fig.axes[0].get_legend_handles_labels()[0]
    finally:
        plt.close(fig)


def test_unresolved_attribution_guidance_takes_precedence_over_low_purity():
    view = build_report_view(analysis({
        "overall_estimate": 0.11, "overall_lower": 0.06, "overall_upper": 0.34,
        "quantitative_status": "discordant_estimators",
        "quantitative_unresolved_reason": "same_lineage_not_identifiable",
    }))
    assert view.purity.is_low and view.purity.is_unresolved
    guidance = render_report_paragraph("purity_attribution", purity=view.purity)
    assert "separation is unresolved" in guidance
    assert "conditional on the selected operating model" in guidance
    assert "Prefer" not in guidance


def test_ceiling_is_explicit_without_inventing_its_cause():
    view = build_report_view(analysis({"overall_estimate": 1.0, "overall_lower": 0.83, "overall_upper": 1.0}))
    for template in ("purity_summary", "purity_methods"):
        text = render_report_paragraph(template, purity=view.purity)
        assert "model's upper boundary" in text and "100% tumor cellularity" in text
        assert "because" not in text
