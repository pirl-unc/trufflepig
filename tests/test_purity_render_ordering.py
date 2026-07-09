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
from trufflepig.tumor_purity import _finalized_purity_headline


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


# --- Behavioral invariant (the durable replacement for the source-order contract) ---
#
# The sample-summary figure now reads its headline purity from _finalized_purity_headline, which
# prefers the FROZEN ReportView snapshot over the live analysis["purity"] dict. That makes the
# 78%-vs-10% divergence structurally impossible: even if the live dict still holds a stale
# pre-decomposition candidate purity at render time, the figure draws the finalized value the
# snapshot captured.


def test_headline_purity_prefers_frozen_snapshot_over_stale_live_dict():
    # Live dict still carries a STALE pre-decomposition candidate purity (78%)...
    analysis = {
        "purity": {
            "overall_estimate": 0.78,
            "overall_lower": 0.70,
            "overall_upper": 0.85,
        },
        "cancer_type": "READ",
        "cancer_name": "Rectum Adenocarcinoma",
        "top_cancers": [("READ", 1.0)],
        "sample_mode": "solid",
    }
    # ...but the frozen snapshot was captured AFTER finalization (10%).
    analysis["report_view"] = build_report_view(
        {
            **analysis,
            "purity": {
                "overall_estimate": 0.10,
                "overall_lower": 0.06,
                "overall_upper": 0.16,
                "purity_source": "decomposition",
            },
        }
    )
    # The figure draws the FINALIZED purity, not the 78% candidate.
    assert _finalized_purity_headline(analysis) == (0.10, 0.06, 0.16)


def test_headline_purity_falls_back_to_live_dict_without_snapshot():
    # A standalone plot_sample_summary call may never build a ReportView → live dict is authoritative.
    analysis = {
        "purity": {
            "overall_estimate": 0.42,
            "overall_lower": 0.30,
            "overall_upper": 0.55,
        }
    }
    assert _finalized_purity_headline(analysis) == (0.42, 0.30, 0.55)


def test_headline_purity_falls_back_field_by_field_when_snapshot_field_is_none():
    # A snapshot that carries no CI (purity_lo/hi None) must not blank out a live-dict CI.
    analysis = {
        "purity": {
            "overall_estimate": 0.30,
            "overall_lower": 0.20,
            "overall_upper": 0.40,
        }
    }
    analysis["report_view"] = build_report_view({"purity": {"overall_estimate": 0.30}})
    overall, lo, hi = _finalized_purity_headline(analysis)
    assert overall == 0.30
    assert lo == 0.20 and hi == 0.40  # live-dict CI preserved
