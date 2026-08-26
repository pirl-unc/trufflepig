"""A flat, frozen snapshot of the finalized per-sample conclusions.

This is the single surface that report renderers (figures, markdown, PDF) should
read, so they cannot disagree with each other or with the analysis. See
``docs/report-belief-consistency-and-friendliness-plan.md`` (Tier 1). The
headline failure it exists to prevent: ``plot_sample_summary`` reading a
pre-decomposition *candidate* purity (78%) while every other artifact shows the
adopted purity (10%).

Phase 1 (instrumentation): the view is BUILT once, after every belief-producing
step has run (cancer-type call, decomposition, purity finalization), and attached
as ``analysis["report_view"]``. Renderers are migrated onto it incrementally in
later phases; until then it is emitted and validated but not yet consumed, so
building it is a pure, behavior-neutral addition (the call site wraps it in
try/except to keep that guarantee even if extraction hits a bad shape).

Design note: deliberately flat and boring — plain fields, uncertainty carried as
``_lo``/``_hi`` suffixes and ``_alternatives`` tuples — NOT a hierarchy of
Estimate/Call/Belief objects. The value is the finalize-then-freeze barrier and
the single read surface, not a type zoo.

Deliberately omitted in Phase 1: ``composition``. The decomposition's tumor
fraction and the adopted ``purity`` come from two sources that can diverge under
a lineage-panel purity override, so surfacing both here would re-embed a
tumor-vs-purity contradiction. Composition is added in Phase 3, once the tumor
share is pinned to the adopted purity at the source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .confidence import compute_call_confidence, purity_confidence_for_analysis


@dataclass(frozen=True)
class ReportView:
    """Immutable snapshot of the conclusions a report should render.

    Frozen on purpose: a "new" conclusion is a new object, so the in-place
    mutation that let a stale purity reach one figure cannot recur once
    renderers read from here.
    """

    # --- the call ---
    cancer_type: str
    cancer_type_name: str
    # NOTE: this is the ranker-contest confidence (compute_call_confidence over
    # candidate_trace), the SAME tier brief.build_summary pairs with the report
    # label. When the report scope overrides the ranker winner, the label and
    # this tier come from different contests — a pre-existing semantic question
    # that is resolved uniformly across markdown + view in a later phase, not
    # papered over here (diverging would create markdown-vs-view drift).
    cancer_type_confidence: str  # low | moderate | high | unknown | ...
    # Runner-up hypotheses with their normalized (top = 1.0) support. Excludes
    # the report label AND its reference cohort, so the true strongest competitor
    # is retained even when the report scope overrode the ranker winner.
    cancer_type_alternatives: Tuple[Tuple[str, float], ...]

    # --- purity: a number that carries its own range + how it was chosen ---
    purity: Optional[float]
    purity_lo: Optional[float]
    purity_hi: Optional[float]
    purity_method: Optional[str]  # integration source, e.g. "estimate+decomposition"
    purity_confidence: str
    purity_status: str  # resolved | discordant_estimators
    purity_scenarios: Tuple[
        Tuple[str, Optional[float], Optional[float], Optional[float]], ...
    ]

    sample_mode: str
    sample_id: Optional[str] = None


def finalized_purity_headline(analysis):
    """Return the ``(overall, lower, upper)`` purity a report should DISPLAY.

    Prefer the frozen ``ReportView`` snapshot (captured at purity finalization) over the live,
    still-mutable ``analysis["purity"]`` dict, so a headline artifact — the sample-summary figure,
    the summary markdown — can never show a stale pre-decomposition *candidate* purity even if it is
    built before finalization has updated the live dict (the 78%-vs-10% belief-consistency bug).
    Falls back to the live dict field-by-field when the snapshot is absent (e.g. a standalone
    ``plot_sample_summary`` call that never built one) or carries no value for a field. This is the
    single read surface every purity headline should route through, so figure and text cannot
    diverge by construction.
    """
    purity = (analysis or {}).get("purity") or {}
    view = (analysis or {}).get("report_view")

    def _pick(view_attr, live_key):
        val = getattr(view, view_attr, None) if view is not None else None
        return val if val is not None else purity.get(live_key)

    return (
        _pick("purity", "overall_estimate"),
        _pick("purity_lo", "overall_lower"),
        _pick("purity_hi", "overall_upper"),
    )


def finalized_purity_context(analysis):
    """Return frozen ``(status, scenarios)`` alongside the purity headline.

    A scenario is ``(source, estimate, lower, upper)``. Keeping this on the
    same immutable surface as :func:`finalized_purity_headline` prevents a
    figure from presenting an operational model as consensus after text
    reports have already declared estimator disagreement.
    """
    purity = (analysis or {}).get("purity") or {}
    view = (analysis or {}).get("report_view")
    status = getattr(view, "purity_status", None) if view is not None else None
    scenarios = (
        getattr(view, "purity_scenarios", None) if view is not None else None
    )
    return (
        str(status or purity.get("quantitative_status") or "resolved"),
        scenarios if scenarios is not None else _purity_scenarios(purity),
    )


def _purity_method(purity: dict) -> Optional[str]:
    source = purity.get("purity_source")
    if source:
        return str(source)
    integration = (purity.get("components") or {}).get("integration") or {}
    src = integration.get("source")
    return str(src) if src else None


def _purity_scenarios(purity: dict):
    scenarios = []
    for row in purity.get("estimator_scenarios") or ():
        if not hasattr(row, "get"):
            continue

        def _number(key):
            value = row.get(key)
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        scenarios.append(
            (
                str(row.get("source") or "unspecified"),
                _number("estimate"),
                _number("lower"),
                _number("upper"),
            )
        )
    return tuple(scenarios)


def _alternatives(analysis) -> Tuple[Tuple[str, float], ...]:
    """Runner-up cancer-type hypotheses with normalized support (top = 1.0).

    Alternatives are every ranked candidate EXCEPT the report label and its
    reference cohort — never a positional strip of ``top_cancers[0]``, which
    would drop the genuine strongest competitor whenever the report scope
    overrode the ranker winner (e.g. a fine-subtype resolution setting the
    headline to READ while the ranker's top candidate is COAD).
    """
    top = analysis.get("top_cancers") or []
    excluded = {
        str(analysis.get("cancer_type") or ""),
        str(analysis.get("reference_cancer_type") or ""),
    }
    excluded.discard("")
    alts = []
    for entry in top:
        try:
            code, frac = entry
            code = str(code)
            if code in excluded:
                continue
            frac = float(frac)
        except (TypeError, ValueError):
            continue
        alts.append((code, frac))
    return tuple(alts)


def build_report_view(analysis, sample_id: Optional[str] = None) -> ReportView:
    """Extract the finalized per-sample conclusions into a frozen snapshot.

    Call this ONLY after purity finalization and decomposition adoption — the
    fields are read as-is, so a mid-pipeline call would capture pre-finalization
    values (the exact failure mode this barrier exists to prevent).
    """
    purity = analysis.get("purity") or {}

    def _f(key):
        val = purity.get(key)
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    return ReportView(
        cancer_type=str(analysis.get("cancer_type") or ""),
        cancer_type_name=str(
            analysis.get("cancer_name") or analysis.get("cancer_type") or ""
        ),
        cancer_type_confidence=str(
            getattr(compute_call_confidence(analysis), "tier", "unknown")
        ),
        cancer_type_alternatives=_alternatives(analysis),
        purity=_f("overall_estimate"),
        purity_lo=_f("overall_lower"),
        purity_hi=_f("overall_upper"),
        purity_method=_purity_method(purity),
        purity_confidence=str(
            getattr(purity_confidence_for_analysis(analysis), "tier", "unknown")
        ),
        purity_status=str(purity.get("quantitative_status") or "resolved"),
        purity_scenarios=_purity_scenarios(purity),
        sample_mode=str(analysis.get("sample_mode") or ""),
        sample_id=sample_id,
    )
