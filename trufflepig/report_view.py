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

    sample_mode: str
    sample_id: Optional[str] = None


def _purity_method(purity: dict) -> Optional[str]:
    source = purity.get("purity_source")
    if source:
        return str(source)
    integration = (purity.get("components") or {}).get("integration") or {}
    src = integration.get("source")
    return str(src) if src else None


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
        sample_mode=str(analysis.get("sample_mode") or ""),
        sample_id=sample_id,
    )
