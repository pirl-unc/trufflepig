# Licensed under the Apache License, Version 2.0
"""Step-by-step reasoning framework.

Each pipeline step emits a typed dataclass. An ``AnalysisState``
composes all the step outputs so far, and reasoning rules are
standalone typed functions that take the state (plus the fields
they specifically need) and return a ``RuleOutcome`` or ``None``.

Design intent:

- Rules are testable in isolation — no global state, no hidden
  dependencies on upstream analysis dicts.
- Rules are ordered as a list (data, not code) so ordering changes
  are configurational, not edits-to-decision-logic.
- Steps accumulate: Step-1 analysis reads Step-0 signals + fresh
  Step-1 fields; Step-2 reads Step-0 + Step-1 + Step-2.
- Every step has the same ``cancer_hint / reasoning_trace`` contract
  so downstream code that only needs the hint doesn't care which
  step produced the call.

For now this module implements the Step-0 rule set; Step-1 through
Step-6 are planned follow-ups. The framework is ready to extend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .healthy_vs_tumor import TissueCompositionSignal


# Thresholds — duplicated from healthy_vs_tumor so the rules can be
# inspected / tested without importing the legacy constants. Keep in
# sync when the original module's thresholds change.
_PROLIFERATION_HIGH_LOG2 = 4.5
_PROLIFERATION_QUIET_LOG2 = 3.5
_HPA_MARGIN_STRONG = 0.05
_HPA_MARGIN_WEAK = 0.02
_CTA_STRONG_COUNT = 4
_CTA_STRONG_SUM_TPM = 30.0
_ONCOFETAL_STRONG_COUNT = 2
_TUMOR_UP_PANEL_STRONG_COUNT = 2


@dataclass(frozen=True)
class DerivedFlags:
    """Precomputed boolean derivations used by Step-0 reasoning rules.

    Populated by :func:`compute_derived_flags` once from the
    :class:`TissueCompositionSignal` fields so rules don't re-derive
    them. Typed, immutable, and part of the Step-0 contract (unlike
    the earlier hack of attaching private attributes to the signal).
    """

    # Structural-ambiguity: the regime where bulk-RNA correlation
    # genuinely cannot distinguish tumor from tissue of origin.
    lymphoid_ambiguity: bool = False
    mesenchymal_ambiguity: bool = False
    # Strong-evidence booleans for each tumor-specific channel —
    # pre-computed from counts/sums against the strong thresholds.
    cta_strong: bool = False
    oncofetal_strong: bool = False
    type_specific_strong: bool = False
    # Soft-evidence booleans — used to demote healthy-dominant calls.
    cta_soft: bool = False
    oncofetal_soft: bool = False
    type_specific_soft: bool = False
    # Cached correlation margin (top HPA ρ − top TCGA ρ).
    correlation_margin: float = 0.0

    @property
    def any_tumor_marker_strong(self) -> bool:
        """Any CTA / oncofetal / type-specific category crossed its strong threshold."""
        return self.cta_strong or self.oncofetal_strong or self.type_specific_strong

    @property
    def any_tumor_marker_soft(self) -> bool:
        """Any tumor-specific category has at least soft evidence."""
        return self.cta_soft or self.oncofetal_soft or self.type_specific_soft

    @property
    def in_ambiguous_regime(self) -> bool:
        return self.lymphoid_ambiguity or self.mesenchymal_ambiguity


@dataclass
class RuleOutcome:
    """Standard outcome of a reasoning rule.

    ``hint`` is the cancer-call it asserts. ``rule_name`` is a short
    human-readable identifier. ``structural`` marks structural-
    ambiguity outcomes (lymphoid / mesenchymal) that must not be
    suppressed by downstream tumor-evidence checks. ``rationale``
    lists the evidence that fired the rule.
    """

    hint: str
    rule_name: str
    structural: bool = False
    rationale: str = ""


# Type alias: rules are pure functions of (signal, flags) returning
# a RuleOutcome (fires) or None (defers to next rule).
Step0Rule = Callable[["TissueCompositionSignal", DerivedFlags], Optional[RuleOutcome]]


def rule(name: str, *, structural: bool = False):
    """Decorator that stamps a descriptive ``rule_name`` on the function.

    The rule runner reads the stamped name for the trace; rule bodies
    don't need to restate the name in their RuleOutcome.
    """

    def decorator(fn):
        fn.rule_name = name
        fn.structural = structural
        return fn

    return decorator


def compute_derived_flags(signal) -> DerivedFlags:
    """Pre-compute the Step-0 derived booleans from raw signal fields.

    Single source of truth for the evidence-channel thresholds — rules
    read from the returned flags rather than re-deriving. Structural-
    ambiguity (lymphoid / mesenchymal) booleans are not derivable from
    the signal alone; callers that know the tissue-of-origin classes
    should ``dataclasses.replace`` the returned flags with the correct
    ambiguity values.
    """
    margin = 0.0
    if signal.top_normal_tissues and signal.top_tcga_cohorts:
        margin = signal.top_normal_tissues[0][1] - signal.top_tcga_cohorts[0][1]
    return DerivedFlags(
        lymphoid_ambiguity=False,
        mesenchymal_ambiguity=False,
        cta_strong=(
            signal.cta_count_above_1_tpm >= _CTA_STRONG_COUNT
            or signal.cta_panel_sum_tpm >= _CTA_STRONG_SUM_TPM
        ),
        oncofetal_strong=(
            signal.oncofetal_count_above_threshold >= _ONCOFETAL_STRONG_COUNT
        ),
        type_specific_strong=(
            len(signal.type_specific_hits) >= _TUMOR_UP_PANEL_STRONG_COUNT
        ),
        cta_soft=(signal.cta_count_above_1_tpm >= 2),
        oncofetal_soft=(signal.oncofetal_count_above_threshold >= 1),
        type_specific_soft=(len(signal.type_specific_hits) >= 1),
        correlation_margin=margin,
    )


# ---- Step-0 rules ----
#
# Each rule is a pure function of (signal, flags) → RuleOutcome | None.
# The @rule decorator stamps a descriptive name + the structural flag
# on the function so the rule runner can build the trace without rules
# restating their own name. Rule bodies read precomputed booleans from
# ``flags`` instead of re-deriving thresholds inline.


@rule("tumor-marker-overrides-ambiguity")
def tumor_marker_overrides_ambiguity(s, f) -> Optional[RuleOutcome]:
    """A strong tumor-specific marker — CTA, oncofetal, or type-specific
    — overrides the lymphoid/mesenchymal-ambiguity flag.

    Canonical case: a real SARC sample with ~58 CTA hits at high TPM
    is definitively tumor despite the mesenchymal correlation regime.
    """
    if not (f.in_ambiguous_regime and f.any_tumor_marker_strong):
        return None
    reasons = _tumor_marker_reasons(s, f)
    return RuleOutcome(
        hint="tumor-consistent",
        rule_name=tumor_marker_overrides_ambiguity.rule_name,
        rationale=",".join(reasons),
    )


@rule("lymphoid-tissue-tumor-indistinguishable", structural=True)
def lymphoid_tissue_ambiguity(s, f) -> Optional[RuleOutcome]:
    """Normal lymphoid tissue and lymphoid malignancy are indistinguishable
    by bulk-RNA correlation — the TCGA DLBC reference is itself >90%
    lymphoid. Flag the ambiguity; downstream analysis proceeds under
    the tumor-sample prior."""
    if not f.lymphoid_ambiguity:
        return None
    return RuleOutcome(
        hint="possibly-tumor",
        rule_name=lymphoid_tissue_ambiguity.rule_name,
        structural=True,
        rationale=_top_pair_rationale(s),
    )


@rule("mesenchymal-tissue-tumor-indistinguishable", structural=True)
def mesenchymal_tissue_ambiguity(s, f) -> Optional[RuleOutcome]:
    """Well-differentiated SARC shares a mesenchymal expression program
    with smooth muscle / adipose / skeletal muscle / endometrial
    myometrium; correlation cannot cleanly distinguish."""
    if not f.mesenchymal_ambiguity:
        return None
    return RuleOutcome(
        hint="possibly-tumor",
        rule_name=mesenchymal_tissue_ambiguity.rule_name,
        structural=True,
        rationale=_top_pair_rationale(s),
    )


@rule("aggregate-tumor-evidence")
def aggregate_tumor_evidence(s, f) -> Optional[RuleOutcome]:
    """Non-ambiguous tissue: aggregated evidence across all six
    categories (CTA, oncofetal, type-specific, proliferation, hypoxia,
    glycolysis) sums to ≥ 1.0, OR any single tumor-marker category is
    strong on its own → tumor-consistent. Catches the low-purity case
    where multiple soft categories co-occur (e.g. a ~16%-purity PRAD
    sample with soft CTA + soft oncofetal + soft glycolysis)."""
    agg = s.evidence.aggregate_score
    if not (agg >= 1.0 or f.any_tumor_marker_strong):
        return None
    reasons = []
    if agg >= 1.0:
        reasons.append(f"aggregate={agg:.2f}≥1.0")
    reasons.extend(_tumor_marker_reasons(s, f))
    return RuleOutcome(
        hint="tumor-consistent",
        rule_name=aggregate_tumor_evidence.rule_name,
        rationale=",".join(reasons),
    )


@rule("high-proliferation-panel")
def high_proliferation_panel(s, f) -> Optional[RuleOutcome]:
    """Coordinated mitotic program at log2-TPM geomean ≥ 4.5 →
    tumor-consistent. Single channel is enough here because the panel
    guards against physiological proliferation (germinal-center
    spleen fires the lymphoid-ambiguity override instead)."""
    if s.evidence.prolif_log2 < _PROLIFERATION_HIGH_LOG2:
        return None
    return RuleOutcome(
        hint="tumor-consistent",
        rule_name=high_proliferation_panel.rule_name,
        rationale=f"panel_log2={s.evidence.prolif_log2:.1f}",
    )


@rule("confident-healthy-tissue")
def confident_healthy_tissue(s, f) -> Optional[RuleOutcome]:
    """Quiet proliferation + strong healthy-correlation margin + no
    soft tumor evidence → healthy-dominant."""
    if not (
        s.evidence.prolif_log2 < _PROLIFERATION_QUIET_LOG2
        and f.correlation_margin >= _HPA_MARGIN_STRONG
    ):
        return None
    if f.any_tumor_marker_soft:
        return None
    return RuleOutcome(
        hint="healthy-dominant",
        rule_name=confident_healthy_tissue.rule_name,
        rationale=f"margin={f.correlation_margin:+.2f}",
    )


@rule("healthy-tissue-with-soft-tumor-signal")
def healthy_with_soft_tumor_signal(s, f) -> Optional[RuleOutcome]:
    """Healthy-correlation preconditions of :func:`confident_healthy_tissue`
    met BUT a soft tumor-marker signal (CTA ≥2 hits, or any oncofetal /
    type-specific hit) fires — demote to possibly-tumor rather than
    call healthy."""
    if not (
        s.evidence.prolif_log2 < _PROLIFERATION_QUIET_LOG2
        and f.correlation_margin >= _HPA_MARGIN_STRONG
    ):
        return None
    if not f.any_tumor_marker_soft:
        return None
    demotes = []
    if f.cta_soft:
        demotes.append(f"CTA_soft(n={s.cta_count_above_1_tpm})")
    if f.oncofetal_soft:
        demotes.append(f"oncofetal_soft(n={s.oncofetal_count_above_threshold})")
    if f.type_specific_soft:
        demotes.append("type_specific_soft")
    return RuleOutcome(
        hint="possibly-tumor",
        rule_name=healthy_with_soft_tumor_signal.rule_name,
        rationale=",".join(demotes),
    )


@rule("weak-healthy-lean")
def weak_healthy_lean(s, f) -> Optional[RuleOutcome]:
    """Weak (0.02 ≤ margin < 0.05) healthy correlation without strong
    proliferation → possibly-tumor caveat."""
    if f.correlation_margin >= _HPA_MARGIN_WEAK:
        return RuleOutcome(
            hint="possibly-tumor",
            rule_name=weak_healthy_lean.rule_name,
            rationale=f"margin={f.correlation_margin:+.2f}",
        )
    return None


@rule("tcga-dominant-correlation")
def tcga_dominant_correlation(s, f) -> Optional[RuleOutcome]:
    """Default: neither strong proliferation nor healthy margin fires —
    correlation favours the TCGA reference → tumor-consistent."""
    return RuleOutcome(
        hint="tumor-consistent",
        rule_name=tcga_dominant_correlation.rule_name,
    )


# ---- Helpers ----


def _top_pair_rationale(s) -> str:
    h = s.top_normal_tissues[0][0] if s.top_normal_tissues else "-"
    t = s.top_tcga_cohorts[0][0] if s.top_tcga_cohorts else "-"
    return f"top_HPA={h} vs top_TCGA={t}"


def _tumor_marker_reasons(s, f: DerivedFlags) -> list[str]:
    """Per-category rationale strings for whichever tumor markers
    (CTA / oncofetal / type-specific) are strong on this sample."""
    out = []
    if f.cta_strong:
        out.append(f"CTA_strong(n={s.cta_count_above_1_tpm})")
    if f.oncofetal_strong:
        out.append(f"oncofetal_strong(n={s.oncofetal_count_above_threshold})")
    if f.type_specific_strong:
        out.append(f"type_specific_strong(n={len(s.type_specific_hits)})")
    return out


# Ordered rule list. Rules are applied in order; the first to fire
# (return a non-None ``RuleOutcome``) wins and downstream rules are
# skipped. Re-orderable without editing rule bodies.
STEP0_RULES: list[Step0Rule] = [
    tumor_marker_overrides_ambiguity,
    lymphoid_tissue_ambiguity,
    mesenchymal_tissue_ambiguity,
    aggregate_tumor_evidence,
    high_proliferation_panel,
    confident_healthy_tissue,
    healthy_with_soft_tumor_signal,
    weak_healthy_lean,
    tcga_dominant_correlation,
]


def run_step0_rules(
    signal,
    flags: DerivedFlags | None = None,
    rules: list[Step0Rule] | None = None,
) -> tuple[RuleOutcome, list[str]]:
    """Apply rules in order. Return (first-fire outcome, trace).

    ``flags`` holds the pre-computed Step-0 derivations; when omitted
    it is computed from ``signal``. ``rules`` defaults to
    :data:`STEP0_RULES` but can be overridden for unit tests that
    want to reorder or subset the rule list.
    """
    if flags is None:
        flags = compute_derived_flags(signal)
    if rules is None:
        rules = STEP0_RULES
    for fn in rules:
        outcome = fn(signal, flags)
        if outcome is not None:
            return outcome, [outcome.rule_name]
    # Should be unreachable — tcga_dominant_correlation is an
    # unconditional default.
    return (
        RuleOutcome(hint="tumor-consistent", rule_name="default"),
        ["default"],
    )


# ---- Analysis state (composable; grows step by step) ----


@dataclass
class AnalysisState:
    """Accumulating typed container for step outputs.

    Rules and downstream analyses read whichever step fields they
    need. ``step0`` is the Step-0 tissue-composition signal;
    ``step1``..``step6`` will be populated as the pipeline runs
    (Step-1 cancer candidates, Step-2 purity, Step-3 decomposition,
    etc.). A step that hasn't run yet is ``None`` — rules that
    depend on it must check before reading.

    For now only step0 is typed through; the other slots are kept
    as the raw ``analysis`` dict pirlygenes already threads through
    the pipeline. Future PRs can replace each slot with a typed
    dataclass as the corresponding step is refactored.
    """

    step0: object | None = None  # TissueCompositionSignal
    # Future slots — left as Any so pre-refactor pipeline still works:
    step1_candidates: list | None = None
    step2_purity: dict | None = None
    step3_decomp: object | None = None
    step4_therapy_axes: dict | None = None
    step5_expression_ranges: object | None = None
    reasoning_trace: list[str] = field(default_factory=list)
