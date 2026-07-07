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


def _analyze_body_source() -> str:
    return inspect.getsource(main_mod._analyze_body)


def test_sample_summary_renders_after_purity_finalization():
    src = _analyze_body_source()

    # The last two writes that can change analysis["purity"]:
    #   1. decomposition-purity adoption (+ lineage-panel override that follows it)
    #   2. the decomposition interval cap (tightens overall_upper)
    adopt_pos = src.index("should_adopt_decomposition_purity(")
    cap_pos = src.index("_constrain_purity_interval_with_decomposition(")

    # The purity-dependent overview figure.
    render_pos = src.index("plot_sample_summary(")

    assert render_pos > adopt_pos, (
        "plot_sample_summary is rendered BEFORE decomposition-purity adoption; "
        "its composition/purity panel will show the pre-decomposition candidate "
        "purity and contradict purity-methods and the markdown (the 78%-vs-10% "
        "headline bug)."
    )
    assert render_pos > cap_pos, (
        "plot_sample_summary is rendered before the decomposition interval cap; "
        "the rendered purity interval can differ from purity-methods/markdown."
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
        "should_adopt_decomposition_purity(",
        "_constrain_purity_interval_with_decomposition(",
        "plot_sample_summary(",
    ):
        assert src.count(marker) == 1, (
            f"expected exactly one {marker!r} call site in _analyze_body; "
            f"found {src.count(marker)} — re-derive the render-ordering contract."
        )
