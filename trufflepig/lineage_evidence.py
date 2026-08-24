"""Tumour-intrinsic lineage evidence: exclusion gates over broad lineage.

The cross-cohort signature screen is fooled by admixture: a carcinoma with heavy
stroma scores a mesenchymal cohort (SARC) high, because the *stromal* mesenchymal
program matches the (also-stromal) SARC reference. We tried to fix this with
deconvolution residuals and it didn't work — the decomposition has no generic
stromal compartment, so the stromal signal survives into the tumour residual
(see docs/cancer-type-residual-matching-findings.md).

What does work is the validated lineage biology: a carcinoma robustly expresses
the **epithelial differentiation program** (EPCAM / cytokeratins / CDH1) in its
*tumour cells*; a true sarcoma does not. So epithelial-program presence is a clean
exclusion of the mesenchymal (and hematolymphoid) branches — the empirically
strong anti-correlations from the lineage sweep (epithelial ⟂ mesenchymal ≈ -0.36,
epithelial ⟂ heme). These markers are tumour-intrinsic; their absolute clean-TPM
burden and cross-cancer specificity are not contaminated by the stromal program
the gate is meant to discount.

Measured separation on representative and local samples is clean under those
two non-HK views. The demotion is
**confidence-proportional**: the stronger the epithelial program, the harder the
non-epithelial branches are demoted (a strongly-keratin+ bladder cancer crushes
SARC; a borderline-epithelial case only nudges it).

This is purely *additive* over the signature screen — it only down-weights
non-epithelial candidates when epithelial differentiation is present; it never
invents or up-ranks a candidate. Real sarcomas (epithelial-absent) are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Tumour-intrinsic epithelial differentiation program.
EPITHELIAL_MARKERS = ("EPCAM", "KRT8", "KRT18", "KRT19", "CDH1")

# Broad lineages a robust epithelial program excludes (carcinoma is not these).
EPITHELIAL_EXCLUDES = ("mesenchymal", "hematolymphoid")

# Every broad-lineage node (see ``cancer_type_ontology.broad_lineage``). Used by
# *exclusive* programs below.
ALL_BROAD_LINEAGES = (
    "epithelial",
    "mesenchymal",
    "hematolymphoid",
    "neural",
    "melanocytic",
    "germ",
    "embryonal",
    "neuroendocrine",
)

# Lineage-*specific* programs: their markers are lineage-defining (≈0 in any other
# lineage), so a confident call means the tumour **is** that lineage — and every
# *other* broad lineage is excluded, not a hand-picked subset. This is the
# generalizable rule (cf. the epithelial gate, which is deliberately NOT here:
# keratins are shared/co-expressed, so epithelial only excludes its validated
# mesenchymal/heme subset). Each entry: (markers, asserted broad lineage,
# confidence threshold).
#
# - neuroendocrine: CHGA/CHGB/SYP/INSM1 — ~0 in carcinomas/gliomas/sarcomas, high
#   in SCLC/NET/NEC/PCPG. NE tumours are keratin+ so epithelial-*absence* can't
#   separate them; the specific program does, demoting epithelial AND neural AND
#   the rest so the NE candidate (own lineage untouched) wins. (#71)
# - melanocytic: MLANA/PMEL/TYR/DCT — lineage-defining, same exclusive logic.
SPECIFIC_LINEAGE_PROGRAMS = (
    ("neuroendocrine", ("CHGA", "CHGB", "SYP", "INSM1"), "neuroendocrine", 0.45),
    ("melanocytic", ("MLANA", "PMEL", "TYR", "DCT"), "melanocytic", 0.45),
)

# The gate fires on the epithelial signal's non-HK program confidence
# (:mod:`trufflepig.signal_views`), integrating absolute log1p(clean TPM) burden
# and specificity against cross-cancer cohorts. On the rep + local sweep, carcinoma
# confidence ≈ 0.62 (median) vs sarcoma ≈ 0.07; 0.45 sits in the gap (the only
# overlap is epithelioid sarcomas, which are genuinely keratin+ — no expression-
# only fix). Demotion is proportional to confidence.
DEFAULT_EPITHELIAL_CONFIDENCE = 0.45
_DEMOTE_SLOPE = 0.9
_DEMOTE_FLOOR = 0.35

# Epithelial-*absent* arm (#69): the inverse of the present gate. A confident
# absence of the tumour-intrinsic epithelial program means the tumour is NOT a
# carcinoma, so the epithelial branch is demoted — fixing the sarcoma->carcinoma
# cluster where a stroma-matched carcinoma out-competes a true sarcoma. The two
# thresholds leave a deliberate dead band (0.15..0.45): carcinomas sit well above
# (median ~0.62), real sarcomas well below (~0.07), so only
# a *confident* absence fires and a borderline low-purity carcinoma is untouched.
_EPITHELIAL_ABSENT_CONFIDENCE = 0.15

# Specific-program demotion is *stronger* than the epithelial gate's: epithelial
# markers (keratins) are shared/co-expressed so its demotion is deliberately
# gentle, but a lineage-*defining* program (NE / melanocytic — markers ≈0 outside
# that lineage) is near-diagnostic, so a confident call decisively demotes the
# other lineages rather than just nudging them. Same proportional-to-confidence
# shape, steeper slope + lower floor.
_SPECIFIC_DEMOTE_SLOPE = 1.3
_SPECIFIC_DEMOTE_FLOOR = 0.15


@dataclass
class LineageEvidence:
    """Per-broad-lineage multiplicative factors plus a readable note."""

    factors: dict[str, float]
    epithelial_signal: float
    notes: list[str] = field(default_factory=list)
    signal: object | None = None  # the SignalViews fingerprint


def _fingerprint(signal) -> str:
    """One-line non-HK fingerprint for the trace."""
    v = signal.views
    return (
        f"epithelial views — log1p(clean TPM) {v['log1p']:.2f} · "
        f"cohort-pct {v['cohort_pct']:.2f} · within-sample pct {v['within_pct']:.2f} · "
        f"cohort-z {v['cohort_z']:+.2f} → {signal.call} (confidence {signal.confidence:.2f}, "
        f"concordance {signal.concordance:.2f})"
    )


def lineage_exclusion_evidence(
    sample_tpm_by_symbol,
    *,
    confidence_threshold: float = DEFAULT_EPITHELIAL_CONFIDENCE,
    cohort_reference=None,
) -> LineageEvidence:
    """Compute broad-lineage demotion factors from the epithelial signal's
    multi-view confidence.

    Returns all-1.0 factors (a no-op) when the epithelial program isn't
    confidently present, so real sarcomas / lymphomas pass through untouched.
    """
    from .signal_views import signal_report

    sig = signal_report(
        "epithelial", EPITHELIAL_MARKERS, sample_tpm_by_symbol,
        cohort_reference=cohort_reference,
    )
    factors: dict[str, float] = {}
    notes: list[str] = []
    if sig.confidence >= confidence_threshold:
        demote = max(_DEMOTE_FLOOR, 1.0 - _DEMOTE_SLOPE * sig.confidence)
        for broad in EPITHELIAL_EXCLUDES:
            factors[broad] = demote
        notes.append(
            f"Epithelial differentiation present — carcinoma; "
            f"{', '.join(EPITHELIAL_EXCLUDES)} candidates demoted ×{demote:.2f} "
            f"(a true sarcoma/lymphoma would lack this tumour-intrinsic program). "
            f"[{_fingerprint(sig)}]"
        )
        for flag in sig.flags:
            notes.append(f"epithelial signal: {flag}")
    elif sig.confidence <= _EPITHELIAL_ABSENT_CONFIDENCE:
        # Epithelial program confidently ABSENT -> not a carcinoma. Demote the
        # epithelial branch so a true sarcoma / lymphoma isn't out-competed by a
        # stroma-matched carcinoma (#69). Confidence-proportional via the *absence*
        # (1 - confidence); a deeper absence demotes harder, floored at _DEMOTE_FLOOR.
        absence = 1.0 - float(sig.confidence)
        demote = max(_DEMOTE_FLOOR, 1.0 - _DEMOTE_SLOPE * absence)
        factors["epithelial"] = min(factors.get("epithelial", 1.0), demote)
        notes.append(
            f"Epithelial differentiation absent — not a carcinoma; epithelial "
            f"candidates demoted ×{demote:.2f} (a stroma-rich carcinoma would still "
            f"express the tumour-intrinsic epithelial program). [{_fingerprint(sig)}]"
        )

    # Specific-lineage arms (#71/#75): the lineage-defining programs are COMPETING
    # hypotheses about the tumour's lineage. Survey them all, then demote each
    # broad lineage by the *evidence margin* against it — how far the best-supported
    # specific lineage outscores that lineage's own program. This integrates every
    # program's confidence rather than discarding any (NOT winner-take-all):
    #   * one program firing  -> the others have margin == its confidence, so they
    #     are demoted exactly as a single decisive call (unchanged behaviour);
    #   * two co-firing       -> the stronger lineage has margin 0 (intact) and the
    #     weaker is demoted only by their *gap*, so a genuinely biphasic / ambiguous
    #     tumour surfaces as provisional — never the degenerate "every lineage
    #     demoted equally" that independent per-program demotion produced.
    # (Fires even when epithelial is co-present — an NE carcinoma is keratin+ —
    # which epithelial-absence cannot catch.)
    prog_sigs = []
    support: dict[str, float] = {}  # asserted lineage -> best above-threshold confidence
    for prog_name, markers, asserted_lineage, threshold in SPECIFIC_LINEAGE_PROGRAMS:
        prog_sig = signal_report(
            prog_name, markers, sample_tpm_by_symbol,
            cohort_reference=cohort_reference,
        )
        prog_sigs.append((prog_name, asserted_lineage, prog_sig))
        if prog_sig.confidence >= threshold:
            support[asserted_lineage] = max(
                support.get(asserted_lineage, 0.0), float(prog_sig.confidence)
            )
    if support:
        top_lineage = max(support, key=lambda lineage: support[lineage])
        top_conf = support[top_lineage]
        for broad in ALL_BROAD_LINEAGES:
            margin = top_conf - support.get(broad, 0.0)
            if margin > 0.0:
                factors[broad] = min(
                    factors.get(broad, 1.0),
                    max(_SPECIFIC_DEMOTE_FLOOR, 1.0 - _SPECIFIC_DEMOTE_SLOPE * margin),
                )
        ranked = ", ".join(
            f"{lineage}={conf:.2f}"
            for lineage, conf in sorted(support.items(), key=lambda kv: -kv[1])
        )
        notes.append(
            f"Lineage-specific program(s) present [{ranked}] -> tumour lineage = "
            f"{top_lineage}; competing lineages demoted by evidence margin (a "
            f"co-firing weaker program is demoted only by its gap, leaving a "
            f"biphasic tumour provisional rather than forcing a call)."
        )
        for prog_name, asserted_lineage, prog_sig in prog_sigs:
            if asserted_lineage in support:
                for flag in prog_sig.flags:
                    notes.append(f"{prog_name} signal: {flag}")

    return LineageEvidence(
        factors=factors,
        epithelial_signal=float(sig.confidence),
        notes=notes,
        signal=sig,
    )
