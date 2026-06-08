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
non-NE by 100–1000× (see docs/cancer-type-residual-matching-findings.md). So a
simple **absolute HK-ratio presence test** (marker TPM relative to the sample's own
housekeeping median — scale-invariant, needs no cohort reference) is both safe and
sufficient. It proposes; the ontology and clinical context resolve.

This layer is purely *additive*: it never removes or reranks the screen's
candidates, so the 11/11 production behaviour on referenced types is untouched.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np


@functools.lru_cache(maxsize=1)
def _ribo_free_hk_symbols() -> tuple[str, ...]:
    """Housekeeping symbols EXCLUDING ribosomal-protein genes (RPL*/RPS*).

    The default housekeeping set has 7 ribosomal-protein genes among 30. Those
    genes are part of pirlygenes' v4 clean-TPM *pinned* ribosomal/technical
    compartment (fixed 25% of the budget), so a HK median that includes them is
    coupled to the clean-TPM normalization: it inflates ~2.4× on v4-clean input
    vs ~1.4× on legacy-clean input. Dropping them makes the marker-gate
    denominator invariant to the clean-TPM compartment handling, so the
    HK-ratio thresholds transfer across normalizations.
    """
    from .reference import pan_cancer_expression
    from .tumor_purity import housekeeping_gene_ids

    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    id_to_sym = dict(zip(ref["Ensembl_Gene_ID"], ref["Symbol"]))
    syms = (str(id_to_sym.get(i, i)) for i in housekeeping_gene_ids())
    return tuple(s for s in syms if not s.startswith(("RPL", "RPS")))


def marker_hk_median(sample_tpm_by_symbol) -> float:
    """Median expression of the ribosomal-free housekeeping set (gate denominator).

    This is the canonical denominator for every tumour-intrinsic marker gate
    (NE recall, epithelial exclusion) — chosen to be invariant to the v4
    two-compartment clean-TPM handling (see :func:`_ribo_free_hk_symbols`).
    """
    vals = [float(sample_tpm_by_symbol.get(s, 0.0)) for s in _ribo_free_hk_symbols()]
    return float(np.median(vals)) if vals else 0.0

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

# Calibrated against the RIBOSOMAL-FREE HK median (:func:`marker_hk_median`).
# NE core markers (CHGA/CHGB/INSM1): SCLC 1.52×, NET 37×, PCPG 606× — vs every
# non-NE rep and local at <=0.14× (LUAD's lone ASCL1 has no core marker). A 0.3
# bar on the specific GATING_MARKERS with a 2-marker minimum + a core-marker
# obligate sits in a 10× gap, firing on all three NE reps and staying silent on
# every non-NE rep and all locals.
DEFAULT_HK_RATIO_THRESHOLD = 0.30
DEFAULT_MIN_PRESENT = 2
# A single core granin/INSM1 this high is unambiguously neuroendocrine on its own
# (well-differentiated NETs can present one dominant granin — e.g. NET_MIDGUT's
# CHGB at 5.5×HK with the rest modest). No non-NE cohort comes near a core marker
# at this level (they sit at ~0), so this relaxes the 2-marker minimum without
# costing specificity.
DEFAULT_STRONG_CORE = 2.0


@dataclass
class RecallProposal:
    """A lineage-marker candidate the screen could not surface on its own."""

    broad: str
    entities: list[str]
    program_score: float                       # max HK-ratio across the program
    present_markers: list[tuple[str, float]]   # (symbol, HK-ratio), strongest first
    subtype_hint: str | None
    rationale: str
    extras: dict = field(default_factory=dict)


def _hk_ratios(sample_tpm_by_symbol, hk_median, symbols):
    if not hk_median or hk_median <= 0:
        return {}
    return {
        s: float(sample_tpm_by_symbol.get(s, 0.0)) / hk_median for s in symbols
    }


def neuroendocrine_recall(
    sample_tpm_by_symbol,
    sample_hk_median,
    *,
    threshold: float = DEFAULT_HK_RATIO_THRESHOLD,
    min_present: int = DEFAULT_MIN_PRESENT,
    strong_core: float = DEFAULT_STRONG_CORE,
) -> RecallProposal | None:
    """Propose neuroendocrine entities when the tumour-intrinsic NE program is on.

    Returns ``None`` when the NE program is absent (the common case — this must
    stay silent on the ~145-1 non-NE codes and all non-NE clinical samples).
    """
    ratios = _hk_ratios(sample_tpm_by_symbol, sample_hk_median, NE_PROGRAM)
    if not ratios:
        return None
    # Only the NE-specific markers gate; the broad ones (ENO2/NCAM1) report only.
    present = sorted(
        ((s, r) for s, r in ratios.items() if s in GATING_MARKERS and r >= threshold),
        key=lambda kv: -kv[1],
    )
    program_score = max((ratios[s] for s in GATING_MARKERS if s in ratios), default=0.0)
    core_max = max((ratios.get(s, 0.0) for s in CORE_MARKERS), default=0.0)
    # Fire on either (a) >=min_present gating markers with a core marker present,
    # or (b) one dominant core granin/INSM1 (>= strong_core) on its own — a
    # well-differentiated NET can present a single very high granin.
    if core_max >= strong_core:
        pass  # dominant single core marker is sufficient
    elif len(present) < min_present:
        return None
    # Obligate: at least one core granin / INSM1 must be present. Without this,
    # ASCL1 (shared by neural / NUT-carcinoma / NE-LUAD) plus an incidental
    # SYP/SCG2 would mis-fire on non-NE tumours.
    core_present = core_max >= threshold
    if not core_present:
        return None

    # subtype hint: which NE entity's secondary markers are co-elevated
    subtype_hint = None
    best_hint = 0.0
    sub_ratios = {}
    for entity, syms in _SUBTYPE_MARKERS.items():
        r = _hk_ratios(sample_tpm_by_symbol, sample_hk_median, syms)
        sub_ratios[entity] = r
        top = max(r.values(), default=0.0)
        if top >= threshold and top > best_hint:
            best_hint, subtype_hint = top, entity

    present_str = ", ".join(f"{s} {r:.1f}×HK" for s, r in present[:4])
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
        program_score=float(program_score),
        present_markers=[(s, float(r)) for s, r in present],
        subtype_hint=subtype_hint,
        rationale=rationale,
        extras={"subtype_marker_ratios": sub_ratios},
    )


# Registry of recall scorers. Kept as a list so future tumour-intrinsic,
# no-reference lineages (e.g. a dedicated germ-cell or hepatoblast recall) slot
# in without touching callers.
RECALL_SCORERS = (neuroendocrine_recall,)


def recall_candidates(sample_tpm_by_symbol, sample_hk_median, **kw) -> list[RecallProposal]:
    """Run every recall scorer; return the proposals that fired (often empty)."""
    out = []
    for scorer in RECALL_SCORERS:
        proposal = scorer(sample_tpm_by_symbol, sample_hk_median, **kw)
        if proposal is not None:
            out.append(proposal)
    return out
