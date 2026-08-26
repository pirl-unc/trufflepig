"""Regression: the sample-summary figure must render the FINALIZED purity.

The 78%-vs-10% headline bug: ``plot_sample_summary`` (the first, patient-facing
figure) reads ``analysis["purity"]`` to draw its composition/purity panel. That
value is not final until the decomposition-purity adoption + lineage-panel
override + interval-cap block inside ``_analyze_body``. When the render was
emitted *before* that block it captured the pre-decomposition *candidate* purity
(78%), while every other figure and all markdown showed the adopted purity (10%).

This is an interim source-order contract, in the style of
``test_battery_audit_fixes.test_decomp_purity_adoption_guard_matches_docstring``:
the pipeline is one large function today, so the ordering is only expressible as
"the render call follows the finalization call". When the finalize/ReportView
barrier lands (renderers take a frozen view instead of the live ``analysis``
dict, so a mid-pipeline read is structurally impossible), replace this with the
cross-artifact behavioral invariant: figure purity == purity-methods purity ==
summary.md purity.
"""

from __future__ import annotations

import inspect

import trufflepig.main as main_mod
from trufflepig.report_view import build_report_view


def _analyze_body_source() -> str:
    return inspect.getsource(main_mod._analyze_body)


def test_sample_summary_renders_after_purity_finalization():
    src = _analyze_body_source()

    # Purity is finalized by two calls, in this order, before any purity-dependent
    # figure renders:
    #   1. _reconcile_purity_after_decomposition — decomposition-purity adoption +
    #      lineage-panel override + the decomposition interval cap.
    #   2. _finalize_fused_purity — random-effects fusion + anti-saturation guard.
    reconcile_pos = src.index("_reconcile_purity_after_decomposition(")
    finalize_pos = src.index("_finalize_fused_purity(")

    # The purity-dependent overview figure.
    render_pos = src.index("plot_sample_summary(")

    assert render_pos > reconcile_pos, (
        "plot_sample_summary is rendered BEFORE decomposition-purity reconciliation; "
        "its composition/purity panel will show the pre-decomposition candidate "
        "purity and contradict purity-methods and the markdown (the 78%-vs-10% "
        "headline bug)."
    )
    assert render_pos > finalize_pos, (
        "plot_sample_summary is rendered before the fused/anti-saturation "
        "finalization; the rendered purity can differ from purity-methods/markdown."
    )


def test_purity_finalization_markers_are_singular():
    """Guard the test above: if any of the markers it keys on stops being a
    unique call site (e.g. a refactor duplicates it), ``str.index`` would match
    the wrong occurrence and the ordering check would be meaningless. Fail here
    instead, loudly, so the contract test is re-derived rather than silently
    passing.
    """
    src = _analyze_body_source()
    for marker in (
        "_reconcile_purity_after_decomposition(",
        "_finalize_fused_purity(",
        "plot_sample_summary(",
    ):
        assert src.count(marker) == 1, (
            f"expected exactly one {marker!r} call site in _analyze_body; "
            f"found {src.count(marker)} — re-derive the render-ordering contract."
        )


def test_report_view_freezes_the_entire_purity_result():
    analysis = {
        "purity": {
            "overall_estimate": 0.10,
            "overall_lower": 0.06,
            "overall_upper": 0.16,
            "quantitative_status": "discordant_estimators",
            "estimator_scenarios": [
                {
                    "source": "lineage_panel",
                    "estimate": 0.10,
                    "lower": 0.06,
                    "upper": 0.16,
                }
            ],
        },
        "cancer_type": "READ",
        "cancer_name": "Rectum Adenocarcinoma",
        "top_cancers": [("READ", 1.0)],
        "sample_mode": "solid",
    }
    view = build_report_view(analysis)

    analysis["purity"].update(
        {
            "overall_estimate": 0.78,
            "overall_lower": 0.70,
            "overall_upper": 0.85,
            "quantitative_status": "resolved",
            "estimator_scenarios": [],
        }
    )

    assert view.purity.estimate == 0.10
    assert view.purity.lower == 0.06
    assert view.purity.upper == 0.16
    assert view.purity.status == "discordant_estimators"
    assert view.purity.scenarios == (
        ("lineage_panel", 0.10, 0.06, 0.16),
    )
