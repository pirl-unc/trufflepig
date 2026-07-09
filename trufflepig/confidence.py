# Licensed under the Apache License, Version 2.0

"""Confidence tiers for pirlygenes analyses (#109).

A single source-of-truth for "how much should the reader trust this
number" that every renderer consumes. The goal is structural: no
quantity shows up in any report identically regardless of how shaky
its underlying evidence is.

The tier system has three levels:

- ``"high"``: no caveats. CI tight, decomposition stable, no artefact
  flags. Renderer shows the number bare.
- ``"moderate"``: one signal is weaker than typical but the number is
  still usable. Renderer shows the number with a short inline note.
- ``"low"``: at least one core input is unreliable. Renderer shows the
  number with a prominent low-confidence tag and, for summary-level
  lists, excludes the row or requires the inline caveat.

Two computed tiers:

- ``compute_purity_confidence``: sample-level — purity CI width, point
  estimate, degradation severity, prep-specific caveats.
- ``compute_target_confidence``: per-target — rolls in the purity
  tier, per-gene attribution (#108), TME flags, purity amplification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List


@dataclass(frozen=True)
class ConfidenceTier:
    tier: str  # "high" | "moderate" | "low" | "degenerate" | "unknown"
    reasons: List[str] = field(default_factory=list)

    @property
    def badge(self) -> str:
        return {
            "low": "low",
            "moderate": "moderate",
            "high": "",
            "degenerate": "—",
            "unknown": "?",
        }.get(self.tier, "")

    @property
    def inline_note(self) -> str:
        if not self.reasons:
            return ""
        return "; ".join(self.reasons)

    def render(self) -> str:
        if self.tier == "high":
            return ""
        note = self.inline_note
        prefix = f"**{self.tier} confidence**"
        return f"{prefix} ({note})" if note else prefix


def sample_purity_is_low(purity_tier) -> bool:
    """True when the sample is in a low-purity regime that distorts tumor-source TPM.

    Keyed off the pipeline's explicit ``"low-purity regime"`` purity-confidence reason —
    the same trigger as the summary caveat that steers readers to tumor-attributed values
    — so the inline tumor-source caveat fires exactly when that regime is flagged.
    """
    if purity_tier is None:
        return False
    reasons = getattr(purity_tier, "reasons", None) or []
    return any("low-purity regime" in str(reason) for reason in reasons)


def concise_confidence_reasons(tier: ConfidenceTier, max_reasons: int = 3) -> str:
    """Return reader-facing shorthand for verbose confidence reasons.

    The full reason strings remain in ``analysis.md`` / evidence tables.
    The one-page summary needs the same information in a compact form so
    the cancer-call line stays skimmable across cancer types.
    """
    notes: list[str] = []
    for reason in tier.reasons or []:
        text = str(reason)
        low = text.lower()
        note = ""
        if "fit quality is weak" in low:
            note = "weak fit"
        elif "fit quality is ambiguous" in low:
            note = "ambiguous fit"
        elif "lineage-pattern concordance" in low:
            note = "lineage mismatch"
        elif "beats runner-up" in low:
            match = re.search(r"runner-up ([A-Za-z0-9_]+)", text)
            note = (
                f"near tie with {match.group(1)}"
                if match
                else "near tie with runner-up"
            )
        elif "raw signature score favored" in low:
            match = re.search(r"raw signature score favored ([A-Za-z0-9_]+)", text)
            note = (
                f"raw signature favors {match.group(1)}"
                if match
                else "raw signature conflict"
            )
        elif (
            "tissue composition screen favored" in low
            or "step-0 correlation favored" in low
        ):
            match = re.search(
                r"(?:Tissue composition screen|Step-0 correlation) favored "
                r"([A-Za-z0-9_]+)",
                text,
            )
            note = (
                f"tissue composition favors {match.group(1)}"
                if match
                else "tissue composition conflict"
            )
        else:
            note = text.split(" — ", 1)[0].strip()
        if note and note not in notes:
            notes.append(note)
        if len(notes) >= max_reasons:
            break
    return "; ".join(notes)


def compute_purity_confidence(
    purity,
    sample_context=None,
    degradation_severity: str = "none",
) -> ConfidenceTier:
    """Tier the overall purity estimate.

    ``purity`` is the dict produced by ``estimate_tumor_purity`` /
    ``analyze_sample`` (keys ``overall_estimate``, ``overall_lower``,
    ``overall_upper``).
    """
    reasons: List[str] = []
    try:
        overall = float(purity.get("overall_estimate") or 0.0)
        lower = float(purity.get("overall_lower") or 0.0)
        upper = float(purity.get("overall_upper") or 0.0)
    except (TypeError, AttributeError, ValueError):
        return ConfidenceTier(tier="unknown", reasons=["no purity estimate available"])

    tier = "high"
    span = upper - lower
    if span <= 1e-9:
        # Zero-width CI means the estimator saw no per-gene variation
        # (synthetic / cohort-median / deterministic input). Surfacing
        # that as "high confidence" is misleading — the estimator
        # couldn't produce uncertainty, not because the answer is
        # certain but because the input has no spread to bound it.
        # #161: tier this as ``degenerate`` so renderers show a
        # specific "deterministic input — CI not estimated" message.
        return ConfidenceTier(
            tier="degenerate",
            reasons=[
                "deterministic input (no per-gene variation) — purity CI not estimated"
            ],
        )
    if span >= 0.35:
        tier = "low"
        reasons.append(f"wide purity CI ({lower:.0%}–{upper:.0%})")
    elif span >= 0.15:
        tier = "moderate"
        reasons.append(f"moderate purity CI span ({span * 100:.0f} pp)")

    if overall < 0.15:
        # Low-purity regime: even a tight CI is hard to act on because
        # any non-tumor signal dominates after dividing by purity.
        if tier != "low":
            tier = "moderate" if tier == "high" else tier
        reasons.append(f"low-purity regime ({overall:.0%})")
        if overall < 0.08:
            tier = "low"

    # Anti-saturation guard provenance: if the honest-fusion pass replaced a near-100% single-method
    # reading (typically ESTIMATE, which the mixture benchmark shows saturates high) with the method
    # consensus because nothing corroborated it, say so and cap the tier — a desaturated estimate is
    # never "high confidence". See main.py best_purity_estimate wiring.
    best_integration = None
    try:
        best_integration = purity.get("best_integration")
    except AttributeError:
        best_integration = None
    if isinstance(best_integration, dict) and best_integration.get("point_source") == "desaturated_fusion":
        if tier == "high":
            tier = "moderate"
        reasons.append("high single-method purity reading uncorroborated — using method consensus")

    sev = (degradation_severity or "none").lower()
    if sev in ("moderate", "severe"):
        # Severe RNA degradation biases long-transcript quantification;
        # target TPMs downstream are less trustworthy.
        if sev == "severe":
            tier = "low"
        elif tier == "high":
            tier = "moderate"
        reasons.append(f"{sev} RNA degradation")

    if sample_context is not None:
        flags = getattr(sample_context, "flags", None) or []
        for flag in flags:
            # targeted-panel inputs are handled elsewhere but worth
            # flagging at the confidence layer so the tier stays honest.
            if "targeted_panel" in str(flag).lower():
                tier = "low"
                reasons.append("likely targeted-panel input")
                break

    return ConfidenceTier(tier=tier, reasons=reasons)


def purity_confidence_for_analysis(analysis) -> ConfidenceTier:
    """Purity confidence tier from a full analysis dict.

    Single source for the ``compute_purity_confidence`` call: derives the
    degradation severity from ``analysis['sample_context']`` and applies it.
    Both the markdown (``_generate_text_reports``) and the ReportView snapshot
    call this so a report can never show one purity tier in a figure and a
    different one in text.
    """
    purity = analysis.get("purity") or {}
    deg = getattr(
        analysis.get("sample_context"), "degradation_severity", "none"
    )
    return compute_purity_confidence(
        purity,
        sample_context=analysis.get("sample_context"),
        degradation_severity=deg,
    )


def compute_call_confidence(analysis) -> ConfidenceTier:
    """Tier the cancer-type call itself based on orthogonal-signal contradictions.

    The classifier's top candidate is its best guess, but that guess
    can be fragile when orthogonal signals disagree. pirlygenes
    computes several such signals that never feed back into the
    confidence tier:

    - ``lineage_concordance`` — how well the sample's lineage-gene
      pattern matches the candidate's expected pattern. Near-zero
      concordance means the classifier picked a candidate whose
      lineage genes aren't expressed in the sample.
    - Tissue-composition screen top cancer-reference match vs the classifier's
      pick. When the screen ranks cohort A first by correlation but the
      classifier picks cohort B, that's a mismatch worth surfacing.
    - Geomean gap to the runner-up. Top geomean 0.431 vs second 0.429
      is a tied call, not a clean win.

    This tier is independent of ``compute_purity_confidence`` — a
    clean purity CI does not rescue a contested cancer-type call, and
    a wide purity CI does not condemn an uncontested call.

    Returns a ``ConfidenceTier`` with tier ∈ {high, moderate, low}
    and reader-facing reason strings. ``low`` is the new tier for
    "contested call" — reserved for cases where the call would be
    materially different if any of the orthogonal signals were the
    tiebreaker.

    Issue #169 motivated this: a real sarcoma validation sample was
    classified as THYM with concordance=0.000 at 4.35.0, and the
    report emitted a clean "Cancer call: THYM" with no caveat.
    """
    reasons: List[str] = []
    tier = "high"
    call_rescue = analysis.get("cancer_call_rescue") or {}
    if call_rescue:
        tier = "moderate"
        message = str(call_rescue.get("message") or "").strip()
        if message:
            reasons.append(message)
        else:
            reasons.append("call depends on a documented classifier pitfall rule")

    candidate_trace = analysis.get("candidate_trace") or []
    if not candidate_trace:
        return ConfidenceTier(tier="unknown", reasons=["no candidate trace"])

    top = candidate_trace[0]
    top_code = top.get("code")

    fit_quality = analysis.get("fit_quality") or {}
    fit_label = str(fit_quality.get("label") or "").strip().lower()
    fit_message = str(fit_quality.get("message") or "").strip()
    if fit_label == "weak":
        tier = "low"
        reason = "fit quality is weak — exact cancer label is provisional"
        if fit_message:
            reason += f" ({fit_message})"
        reasons.append(reason)
    elif fit_label == "ambiguous":
        if tier == "high":
            tier = "moderate"
        reason = "fit quality is ambiguous — preserve alternate cancer hypotheses"
        if fit_message:
            reason += f" ({fit_message})"
        reasons.append(reason)

    # 1. Lineage concordance near zero — classifier picked a candidate
    # whose lineage genes aren't expressed.
    concordance = top.get("lineage_concordance")
    if concordance is not None:
        try:
            concordance = float(concordance)
        except (TypeError, ValueError):
            concordance = None
    if concordance is not None and concordance < 0.2:
        tier = "low"
        reasons.append(
            f"lineage-pattern concordance is {concordance:.2f} "
            f"(near zero) — sample does not match the expected "
            f"{top_code} lineage-gene pattern"
        )

    def _confidence_support(row):
        for key in ("support_fraction_of_top", "support_rank_score", "support_geomean"):
            try:
                value = float(row.get(key) or 0.0)
            except (TypeError, ValueError, AttributeError):
                value = 0.0
            if value > 0:
                label = "post-gate support" if key != "support_geomean" else "geomean"
                return value, label
        return 0.0, "support"

    # 2. Post-gate support gap to the strongest competitors. Within 15% of the
    # runner-up is a tied 2-way call (downgrade by one tier). Within 15% of the
    # 3rd-place candidate is a tied 3-way call (downgrade all the way to low /
    # provisional). Confidence must not assume display order is support order:
    # candidate traces can intentionally keep same-family alternatives adjacent,
    # while ``support_fraction_of_top`` / ``support_rank_score`` carry the
    # support metric that should drive confidence.
    if len(candidate_trace) >= 2:
        top_support, support_label = _confidence_support(top)
        competitors = sorted(
            candidate_trace[1:],
            key=lambda row: _confidence_support(row)[0],
            reverse=True,
        )
        second = competitors[0]
        second_support, _ = _confidence_support(second)
        third = competitors[1] if len(competitors) >= 2 else None
        third_support = _confidence_support(third)[0] if third else 0.0

        def _rel_gap(candidate_support: float) -> float:
            return 1.0 - candidate_support / top_support if top_support > 0 else 1.0

        gap_to_second = _rel_gap(second_support)
        gap_to_third = _rel_gap(third_support) if third else 1.0
        TIED_GAP_THRESHOLD = 0.15

        if second_support > 0 and gap_to_second <= TIED_GAP_THRESHOLD:
            second_code = second.get("code")
            if tier == "high":
                tier = "moderate"
            reasons.append(
                f"top candidate {top_code} beats runner-up {second_code} "
                f"by only {gap_to_second * 100:.0f}% on {support_label} "
                f"({top_support:.3f} vs {second_support:.3f}) — call is ambiguous"
            )
            try:
                second_concordance = float(second.get("lineage_concordance"))
            except (TypeError, ValueError):
                second_concordance = None
            if (
                concordance is not None
                and second_concordance is not None
                and second_concordance >= concordance + 0.25
            ):
                tier = "low"
                reasons.append(
                    f"runner-up {second_code} has much stronger lineage-pattern "
                    f"concordance than {top_code} ({second_concordance:.2f} vs "
                    f"{concordance:.2f}) in a near-tied call — treat the top label "
                    f"as provisional"
                )
            # 3-way tie: third-place is also within threshold. The top-1
            # call is provisional regardless of which one of the three
            # the classifier picked.
            if third and third_support > 0 and gap_to_third <= TIED_GAP_THRESHOLD:
                third_code = third.get("code")
                tier = "low"
                reasons.append(
                    f"3-way tied call: {top_code} / {second.get('code')} / "
                    f"{third_code} all within {TIED_GAP_THRESHOLD * 100:.0f}% on "
                    f"{support_label} ({top_support:.3f} / {second_support:.3f} / {third_support:.3f}) — "
                    f"treat the top label as provisional"
                )

    # 2b. Raw signature tension: the final support score can promote a
    # candidate via purity/family factors even when another cancer type has
    # the stronger raw expression signature. That is legitimate, but the
    # reader should not see a cleaner confidence banner than the evidence.
    try:
        top_sig = float(top.get("signature_score") or 0.0)
        best_sig = max(
            candidate_trace,
            key=lambda row: float(row.get("signature_score") or 0.0),
        )
        best_sig_score = float(best_sig.get("signature_score") or 0.0)
    except (TypeError, ValueError):
        top_sig = 0.0
        best_sig = top
        best_sig_score = 0.0
    best_sig_code = best_sig.get("code")
    if (
        best_sig_code
        and top_code
        and best_sig_code != top_code
        and top_sig > 0
        and best_sig_score > top_sig * 1.05
    ):
        if tier == "high":
            tier = "moderate"
        reasons.append(
            f"raw signature score favored {best_sig_code} over {top_code} "
            f"({best_sig_score:.3f} vs {top_sig:.3f})"
        )

    # 3. Tissue-composition top cancer-reference match disagrees with the classifier's
    # pick. Sample-level correlation is the coarsest signal and can
    # be more reliable than the classifier's geomean when the panel
    # evidence is weak.
    hvt = analysis.get("healthy_vs_tumor")
    step0_top_code = None
    if hvt is not None:
        tcga = getattr(hvt, "top_tcga_cohorts", None) or []
        if tcga:
            name, _rho = tcga[0]
            step0_top_code = (
                name.removesuffix("_TPM") if isinstance(name, str) else None
            )
    if step0_top_code and top_code and step0_top_code != top_code:
        if tier == "high":
            tier = "moderate"
        reasons.append(
            f"Tissue composition screen favored {step0_top_code} but the "
            f"classifier picked {top_code}"
        )

    seen = set()
    deduped: List[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)

    return ConfidenceTier(tier=tier, reasons=deduped)


def compute_target_confidence(
    row,
    purity_tier: ConfidenceTier,
    sample_context=None,
) -> ConfidenceTier:
    """Tier a single therapy-target row.

    ``row`` is a dict- or Series-like with at least ``observed_tpm``,
    optionally ``tme_dominant``, ``tme_explainable``,
    ``attr_tumor_fraction``, and ``attribution`` (the new #108 columns).
    """
    reasons: List[str] = []
    tier = (
        purity_tier.tier
        if purity_tier.tier in {"low", "moderate", "high"}
        else "moderate"
    )
    if purity_tier.tier in {"low", "moderate"} and purity_tier.reasons:
        # Fold in the sample-level reasons so target rows carry them
        # when they're the dominant limit on confidence.
        reasons.extend(purity_tier.reasons)

    def _get(key, default=None):
        try:
            value = row.get(key) if hasattr(row, "get") else row[key]
        except (KeyError, AttributeError):
            return default
        return value if value is not None else default

    if _get("tme_dominant"):
        tier = "low"
        attribution = _get("attribution")
        if isinstance(attribution, dict) and attribution:
            top = max(attribution, key=lambda k: attribution[k])
            reasons.append(
                f"TME-dominant (tumor < 30% of observed; "
                f"top non-tumor compartment: {top.replace('_', ' ')})"
            )
        else:
            reasons.append("TME-dominant (≥70% of signal is non-tumor)")
    elif _get("tme_explainable"):
        if tier == "high":
            tier = "moderate"
        reasons.append("could be explained by a single healthy tissue's expression")

    # #131: matched-normal over-prediction. When the fitted
    # matched-normal (or any) compartment predicts more of the gene
    # than the sample actually contains, the attribution math's
    # tumor residual is zero by construction — but that's a ceiling
    # effect of the model, not evidence the tumor isn't expressing
    # the gene. Fire a moderate downgrade with a reader-facing reason
    # so clinically curated targets (KLK3 / TACSTD2 / FOLH1 on CRPC
    # samples) aren't silently dismissed.
    if _get("matched_normal_over_predicted"):
        if tier == "high":
            tier = "moderate"
        reasons.append(
            "matched-normal reference over-predicts this gene — "
            "tumor attribution hit the zero floor; the raw observed "
            "TPM is the better read than the attributed fraction"
        )

    # #128: broadly-expressed flag. A gene expressed across many
    # non-reproductive HPA tissues cannot be claimed as tumor-cell-
    # specific even when the residual attribution puts most of the
    # observed TPM into the tumor compartment, because the sample's
    # healthy cells alone carry a baseline broader than any single
    # compartment in the fitted decomposition.
    #
    # Amplification overrides: HER2 in HER2+ BRCA, MDM2 in WD/DD-LPS,
    # GPC3 in HCC — broadly expressed per HPA but observed well above
    # peak-healthy in the sample, which IS a tumor-specificity signal
    # regardless of breadth. The ``broadly_expressed`` column in
    # ranges_df is already gated on ``not amplified_over_healthy`` so
    # amplification-driven targets don't even trigger this branch.
    n_tissues = _get("n_healthy_tissues_expressed")
    amp_fold = _get("amplification_fold")
    if _get("broadly_expressed"):
        tier = "low"
        amp_note = ""
        if isinstance(amp_fold, (int, float)) and float(amp_fold) < 5.0:
            amp_note = f", peak-healthy fold only {float(amp_fold):.1f}×"
        if isinstance(n_tissues, (int, float)):
            reasons.append(
                f"broadly expressed across {int(n_tissues)} healthy "
                f"tissues{amp_note} — not tumor-cell-specific"
            )
        else:
            reasons.append(
                "broadly expressed across many healthy tissues — "
                "not tumor-cell-specific"
            )

    attr_fraction = _get("attr_tumor_fraction")
    if isinstance(attr_fraction, (int, float)) and 0.3 <= float(attr_fraction) < 0.5:
        if tier == "high":
            tier = "moderate"
        reasons.append(f"only {float(attr_fraction):.0%} of signal attributed to tumor")

    # Deduplicate reasons while preserving order.
    seen = set()
    deduped: List[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            deduped.append(r)

    return ConfidenceTier(tier=tier, reasons=deduped)
