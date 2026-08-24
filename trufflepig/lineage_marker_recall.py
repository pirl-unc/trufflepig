"""Lineage-marker recall: propose no-reference entities the cross-cohort screen
structurally cannot surface.

The signature screen (:func:`tumor_purity.rank_cancer_type_candidates`) can only
propose cancer types that have a TCGA reference cohort. Whole families have **no**
reference — most importantly neuroendocrine: of 14 NE registry codes, only PCPG
is a TCGA cohort, so SCLC / NETs / Merkel / neuroblastoma are scattered across
LUAD/DLBC and never reach the classifier.

This module adds a high-recall layer for exactly those entities, using **tumour-
intrinsic** lineage markers. Unlike the TME-contaminated heme/epithelial panels
(PTPRC lights up on any immune-infiltrated tumour; keratins clear an epithelial
floor in sarcomas), the neuroendocrine program (CHGA / SYP / INSM1 / CHGB / ASCL1)
is **absent from stroma and immune background** — empirically it separates NE from
non-NE by 100–1000× (see docs/cancer-type-residual-matching-findings.md). We
therefore require both absolute clean-TPM burden and specificity against the
cross-cancer cohort distribution. It proposes; the ontology and clinical context
resolve.

This layer is purely *additive*: it never removes or reranks the screen's
candidates, so the 11/11 production behaviour on referenced types is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Tumour-intrinsic pan-neuroendocrine program. Any-of presence; CHGA/CHGB/INSM1/
# ASCL1 are the strongest (and cleanest of stromal/immune background).
NE_PROGRAM = ("CHGA", "CHGB", "INSM1", "ASCL1", "SYP", "NCAM1", "ENO2", "SCG2")

# Only the NE-specific markers gate the call. ENO2 (neuron-specific enolase) and
# NCAM1 (CD56) are broadly expressed and would leak; they're reported as
# supporting context but never gate.
GATING_MARKERS = ("CHGA", "CHGB", "INSM1", "ASCL1", "SYP", "SCG2")

# Core NE-defining markers — the granins and the pan-NE TF INSM1. At least one
# must be present (obligate-style). ASCL1/SYP/SCG2 are shared by neural, squamous
# (NUT carcinoma) and NE-differentiated LUAD, so they can co-elevate without true
# NE differentiation; requiring a core granin/INSM1 stops those false positives
# (e.g. pfo019 NUT carcinoma: ASCL1 0.33 but CHGA/CHGB/INSM1 all 0 -> silent).
CORE_MARKERS = ("CHGA", "CHGB", "INSM1")

# Markers that strengthen a specific NE subtype call (used only as hints — the
# entity still has to clear the NE program presence test first).
_SUBTYPE_MARKERS = {
    "NBL": ("PHOX2B", "MYCN", "LIN28B", "DBH"),        # embryonal neural-crest
    "NEC_MERKEL": ("KRT20",),                          # Merkel-cell carcinoma
    "SCLC": ("NEUROD1", "POU2F3", "YAP1"),             # small-cell molecular subtypes
    "NET_PANCREAS": ("SSTR2", "PCSK1", "PAX6"),        # well-diff pancreatic NET
    "PCPG": ("PNMT", "TH"),                            # adrenal chromaffin
}

# NE entities the cross-cohort screen cannot propose (no TCGA cohort). PCPG is
# listed for completeness/sub-typing but the screen already covers it.
NE_NO_REFERENCE = ("SCLC", "NET_PANCREAS", "NET_LUNG", "NET_MIDGUT", "NEC_MERKEL", "NBL")

@dataclass
class RecallProposal:
    """A lineage-marker candidate the screen could not surface on its own."""

    broad: str
    entities: list[str]
    program_score: float                       # integrated non-HK support
    present_markers: list[tuple[str, float]]   # (symbol, support), strongest first
    subtype_hint: str | None
    rationale: str
    extras: dict = field(default_factory=dict)


def _marker_supports(sample_tpm_by_symbol, symbols, cohort_reference):
    """Return single-marker program confidence in the shared non-HK space."""
    from .signal_views import signal_report

    return {
        symbol: signal_report(
            symbol,
            (symbol,),
            sample_tpm_by_symbol,
            cohort_reference=cohort_reference,
        ).confidence
        for symbol in symbols
    }


def neuroendocrine_recall(
    sample_tpm_by_symbol,
    *,
    cohort_reference=None,
) -> RecallProposal | None:
    """Propose neuroendocrine entities when the tumour-intrinsic NE program is on.

    Returns ``None`` when the NE program is absent (the common case — this must
    stay silent on the ~145-1 non-NE codes and all non-NE clinical samples).
    """
    from .signal_views import PRESENT_CONFIDENCE, signal_report

    program = signal_report(
        "neuroendocrine",
        GATING_MARKERS,
        sample_tpm_by_symbol,
        cohort_reference=cohort_reference,
    )
    supports = _marker_supports(
        sample_tpm_by_symbol, GATING_MARKERS, cohort_reference
    )
    # Only the NE-specific markers gate; ENO2/NCAM1 remain contextual.
    present = sorted(
        (
            (symbol, support)
            for symbol, support in supports.items()
            if support >= PRESENT_CONFIDENCE
        ),
        key=lambda kv: -kv[1],
    )
    core_present = any(symbol in CORE_MARKERS for symbol, _ in present)
    # Obligate: at least one core granin / INSM1 must be independently present.
    # Without this,
    # ASCL1 (shared by neural / NUT-carcinoma / NE-LUAD) plus an incidental
    # SYP/SCG2 would mis-fire on non-NE tumours.
    if not core_present:
        return None
    # The obligate core marker has already cleared both the absolute-burden and
    # cross-cancer-specificity views. Additional present markers establish a
    # broader program; a lone core marker preserves secretory NETs dominated by
    # one granin.

    # subtype hint: which NE entity's secondary markers are co-elevated
    subtype_hint = None
    best_hint = 0.0
    subtype_supports = {}
    for entity, syms in _SUBTYPE_MARKERS.items():
        marker_support = _marker_supports(
            sample_tpm_by_symbol, syms, cohort_reference
        )
        subtype_supports[entity] = marker_support
        top = max(marker_support.values(), default=0.0)
        if top > best_hint:
            best_hint, subtype_hint = top, entity
    if best_hint < PRESENT_CONFIDENCE:
        subtype_hint = None

    present_str = ", ".join(f"{s} support {r:.2f}" for s, r in present[:4])
    rationale = (
        f"Neuroendocrine RNA program present ({present_str}); these markers are "
        f"tumour-intrinsic (near-absent from stroma/immune background), so this is "
        f"a lineage call the cross-cohort screen cannot make without an NE "
        f"reference cohort. Proposed for confirmation"
        + (f"; secondary markers favour {subtype_hint}." if subtype_hint else ".")
    )
    return RecallProposal(
        broad="neuroendocrine",
        entities=list(NE_NO_REFERENCE),
        program_score=float(max(program.confidence, *(r for _, r in present))),
        present_markers=[(s, float(r)) for s, r in present],
        subtype_hint=subtype_hint,
        rationale=rationale,
        extras={
            "feature_space": "log1p_clean_tpm+cohort_percentile",
            "program_confidence": float(program.confidence),
            "subtype_marker_support": subtype_supports,
        },
    )


# Registry of recall scorers. Kept as a list so future tumour-intrinsic,
# no-reference lineages (e.g. a dedicated germ-cell or hepatoblast recall) slot
# in without touching callers.
RECALL_SCORERS = (neuroendocrine_recall,)


def recall_candidates(sample_tpm_by_symbol, **kw) -> list[RecallProposal]:
    """Run every recall scorer; return the proposals that fired (often empty)."""
    out = []
    for scorer in RECALL_SCORERS:
        proposal = scorer(sample_tpm_by_symbol, **kw)
        if proposal is not None:
            out.append(proposal)
    return out
