"""Cancer-type evidence selection.

Each row is one cancer-type hypothesis with the same feature columns:
pan-cancer signature-ranker support, related expression-context support,
RNA marker support, exact-reference support, learned full-profile support,
lineage-panel support, and direct-fusion support. Evidence from different
sources accumulates on the same row instead of becoming separate control-flow
branches.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger(__name__)

# Source-class rank used as the tie-breaker when two evidence kinds compute
# identical (class_rank, strength). Higher = preferred. The numeric values are
# arbitrary; only their relative order matters. Keep this list in sync with
# every selected_by string that calls consider_for_report_label.
_SELECTED_BY_TIEBREAK_RANK: dict[str, int] = {
    "direct_fusion": 6,
    "fused_evidence": 5,
    "entity_evidence_consensus": 5,
    "fine_reference": 5,
    "local_expression_reference": 4,
    "lineage_panel": 4,
    "pan_cancer_signature_subtype": 4,
    "learned_expression_classifier": 4,
    "contrast_discriminator": 3,
    "tumor_label_refinement": 3,
    "coarse_composition_reference": 2,
    "rare_marker": 2,
    "pan_cancer_signature_ranker": 1,
    "primary_expression_match": 1,
}

# Definitive molecular evidence: a detected fusion is diagnostic and outranks whole-profile
# expression (which can be contaminated / low-purity), so the centroid corroboration in
# ``_pick_selected`` never overrides it.
_DEFINITIVE_SELECTORS: frozenset[str] = frozenset({"direct_fusion"})

# Whole-profile centroid corroboration margin (Spearman rho). At a CONFIDENT compartment call the
# centroid re-ranks the already-selectable hypotheses, preferring one it out-correlates the authority
# winner by at least this much — enough to flip a marker tie (NUT carcinoma over the salivary ADCC
# look-alike; osteosarcoma over the MDM2-amplicon liposarcoma) while leaving near-ties to the markers.
_CENTROID_CORROBORATION_MARGIN = 0.015
_CENTROID_SAME_LINEAGE_MARKER_MARGIN = 0.035

# Thresholds for the lineage_panel selector — gates when a
# trufflepig.lineage_panels panel result is strong enough to
# propose a cancer type. See issue #42 + lineage_panels.py for
# the design rationale. Conservative defaults so this only fires
# on clean within-family discrimination cases.
_LINEAGE_PANEL_MIN_SCORE = 0.60
_LINEAGE_PANEL_MIN_MARGIN_OVER_SECOND = 0.20
_LINEAGE_PANEL_OUT_OF_BEAM_MIN_SCORE = 0.85
_LINEAGE_PANEL_OUT_OF_BEAM_MIN_MARGIN = 0.25

# How many evaluated panels to record in ``lineage_panel_all`` for
# downstream rendering / debugging. Mirrors the broad-top-5 cap
# used by other selectors so reports stay scannable.
_LINEAGE_PANEL_ALL_LIMIT = 5

_SQUAMOUS_CONTEXT_CODES = frozenset({"HNSC", "LUSC", "ESCA", "CESC", "THYM"})
_SALIVARY_CODES = frozenset({"ADCC", "ACINIC"})
_RHABDOID_CODES = frozenset({"ATRT", "RT"})
_RHABDOID_SMARCB1_MAX_TPM_BY_CODE = {
    "RT": 5.0,
    "ATRT": 40.0,
}
_OS_OSTEOGENIC_MARKERS = (
    "RUNX2",
    "SATB2",
    "IBSP",
    "BGLAP",
    "ALPL",
    "SPP1",
    "SP7",
    "DLX5",
    "DLX6",
    "DMP1",
    "MEPE",
)
_OS_MATRIX_MARKERS = (
    "COL1A1",
    "COL1A2",
    "LUM",
    "FN1",
    "BGN",
    "COL6A1",
    "PCOLCE",
    "MXRA8",
    "COL5A2",
    "MMP14",
)
_OS_AMPLICON_MARKERS = ("MDM2", "CDK4", "FRS2")
_BACKGROUND_LIKE_FAMILIES = frozenset({"MESENCHYMAL"})
_CRC_REGISTRY_ROOT = "CRC"
_CRC_FAMILY_COMPETITOR_MIN_SUPPORT = 0.80
_CRC_FAMILY_COMPETITOR_MIN_FAMILY_SUPPORT = 0.60
_CRC_FAMILY_COMPETITOR_MAX_RANK = 6
_PRIMARY_CONTEXT_DOMINANCE_MIN_SUPPORT = 0.85
_PRIMARY_CONTEXT_DOMINANCE_MIN_RATIO = 0.95
_TUMOR_LABEL_MIN_SUPPORT = 0.70
_TUMOR_LABEL_MIN_SIGNATURE_RATIO = 0.85
_TUMOR_LABEL_MIN_FAMILY_SUPPORT = 0.65
_LEARNED_EXPRESSION_MIN_PROBABILITY = 0.55
_LEARNED_EXPRESSION_STRONG_PROBABILITY = 0.85
_LEARNED_EXPRESSION_CONTEXT_FREE_STRONG_PROBABILITY = 0.97
_LEARNED_EXPRESSION_MIN_MARGIN = 0.10
_LEARNED_EXPRESSION_CONTEXT_FREE_MIN_MARGIN = 0.50
_LEARNED_EXPRESSION_MIN_CONTEXT_SUPPORT = 0.30
_LEARNED_EXPRESSION_MIN_HIERARCHICAL_CONTEXT = 0.70
# Final hierarchy adjudication is deliberately stricter than ordinary learned
# evidence admission.  On the deterministic 3-train/2-test representative
# split, hierarchy-entity calls at p>=0.97 and margin>=0.50 were 50/50
# entity-compatible.  Family and compartment agreement make the release gate
# narrower still.
_LEARNED_HIERARCHY_ENTITY_ADJUDICATION_MIN_SUPPORT = 0.97
_LEARNED_HIERARCHY_ENTITY_ADJUDICATION_MIN_MARGIN = 0.50
_LEARNED_HIERARCHY_ENTITY_ADJUDICATION_MIN_FAMILY_SUPPORT = 0.90
_LEARNED_HIERARCHY_ENTITY_ADJUDICATION_MIN_COMPARTMENT_SUPPORT = 0.90
# Cross-lineage arbitration has two independently defensible paths.  The
# strong-entity path was 79/79 lineage-compatible on the leakage-free holdout
# at p>=0.85/margin>=0.50.  The lower entity path requires near-unanimous
# family+compartment votes plus a selection-grade curated marker program; it
# recovers heterogeneous sibling-distributed cases without memorizing weak
# all-QC-fail/outlier representatives.
_LEARNED_HIERARCHY_LINEAGE_STRONG_ENTITY_SUPPORT = 0.85
_LEARNED_HIERARCHY_LINEAGE_STRONG_ENTITY_MARGIN = 0.50
_LEARNED_HIERARCHY_LINEAGE_CORROBORATED_ENTITY_SUPPORT = 0.50
_LEARNED_HIERARCHY_LINEAGE_CORROBORATED_FAMILY_SUPPORT = 0.95
_LEARNED_HIERARCHY_LINEAGE_CORROBORATED_COMPARTMENT_SUPPORT = 0.95
# Entity refinement below the calibrated single-model path is deliberately a
# consensus rule, not another probability threshold.  A learned entity vote
# must be joined by two independent evidence groups and win a majority of the
# available groups.  This structural requirement applies uniformly to every
# cancer type and keeps one unusually confident model from manufacturing a
# leaf call on its own.
_ENTITY_CONSENSUS_MIN_SUPPORTING_AXES = 3
_ENTITY_CONSENSUS_MIN_NONLEARNED_AXES = 2
_FUSED_EVIDENCE_CONTEXT_FREE_LEARNED_PROBABILITY = 0.97
_FUSED_EVIDENCE_CONTEXT_FREE_CENTROID_SUPPORT = 0.35
_FUSED_EVIDENCE_MIN_SCORE = 0.25
_FUSED_EVIDENCE_PAN_SIGNATURE_WEIGHT = 0.30
_FUSED_EVIDENCE_CENTROID_WEIGHT = 0.40
_FUSED_EVIDENCE_LEARNED_WEIGHT = 1.20
_FUSED_EVIDENCE_STRONG_LEARNED_ENTITY_PROBABILITY = 0.93
_FUSED_EVIDENCE_STRONG_LEARNED_ENTITY_MARGIN = 0.50
_FUSED_EVIDENCE_STRONG_LEARNED_ENTITY_SUPPORT = 0.90
_FUSED_EVIDENCE_STRONG_LEARNED_LINEAGE_SUPPORT = 0.80
_FUSED_EVIDENCE_EXACT_REFERENCE_WEIGHT = 1.25
_FUSED_EVIDENCE_LINEAGE_PANEL_WEIGHT = 1.10
_FUSED_EVIDENCE_EXACT_STAGE_BONUS = 0.45
_FUSED_EVIDENCE_LEARNED_WEAK_REFERENCE_MAX_FRACTION = 0.45
_FUSED_EVIDENCE_LEARNED_WEAK_REFERENCE_MAX_BURDEN = 0.20
_FUSED_EVIDENCE_MARKER_COHERENT_REFERENCE_MIN_FRACTION = 0.75
_FUSED_EVIDENCE_MARKER_COHERENT_REFERENCE_MIN_BURDEN = 0.50
_FUSED_EVIDENCE_CENTROID_ANCHORED_REFERENCE_MIN_CENTROID = 0.95
_FUSED_EVIDENCE_CENTROID_ANCHORED_REFERENCE_MIN_SUPPORT = 0.30
_FUSED_EVIDENCE_CENTROID_ANCHORED_REFERENCE_MIN_FRACTION = 0.35
_FUSED_EVIDENCE_CENTROID_ANCHORED_REFERENCE_MIN_BURDEN = 0.10
_FUSED_EVIDENCE_LEARNED_COMPARTMENT_CONTEXT_MIN_LEARNED = 0.95
_FUSED_EVIDENCE_LEARNED_COMPARTMENT_CONTEXT_MIN_CENTROID = 0.95
_FUSED_EVIDENCE_LEARNED_COMPARTMENT_CONTEXT_MIN_SIGNATURE = 0.75
_FUSED_EVIDENCE_LEARNED_FAMILY_CONTEXT_MIN_SUPPORT = 0.25
_FUSED_EVIDENCE_LEARNED_FAMILY_CONTEXT_MIN_MARGIN = 0.10
_FUSED_EVIDENCE_LEARNED_FAMILY_CONTEXT_MIN_SIGNATURE = 0.85
_FUSED_EVIDENCE_LEARNED_FAMILY_CONTEXT_MAX_RANK = 3
# The learned-family-anchored context COMPONENT is base + support·fam + signature·broad.
# Named here so the admission/channel gates key off the component's guaranteed minimum (when the
# feature flag fires: fam≥MIN_SUPPORT, broad≥MIN_SIGNATURE) instead of a rounder bar the component
# can never reach — otherwise a legitimately family-anchored candidate lands in a dead zone where
# the flag is set yet it is blocked as "lacks a non-ranker admission path".
_FUSED_EVIDENCE_LEARNED_FAMILY_ANCHOR_BASE = 0.45
_FUSED_EVIDENCE_LEARNED_FAMILY_ANCHOR_SUPPORT_WEIGHT = 0.30
_FUSED_EVIDENCE_LEARNED_FAMILY_ANCHOR_SIGNATURE_WEIGHT = 0.25
_FUSED_EVIDENCE_LEARNED_FAMILY_ANCHOR_MIN_COMPONENT = round(
    _FUSED_EVIDENCE_LEARNED_FAMILY_ANCHOR_BASE
    + _FUSED_EVIDENCE_LEARNED_FAMILY_ANCHOR_SUPPORT_WEIGHT
    * _FUSED_EVIDENCE_LEARNED_FAMILY_CONTEXT_MIN_SUPPORT
    + _FUSED_EVIDENCE_LEARNED_FAMILY_ANCHOR_SIGNATURE_WEIGHT
    * _FUSED_EVIDENCE_LEARNED_FAMILY_CONTEXT_MIN_SIGNATURE,
    4,
)
_FUSED_EVIDENCE_LEARNED_FAMILY_CONFLICT_MIN_TOP_SUPPORT = (
    _FUSED_EVIDENCE_LEARNED_FAMILY_CONTEXT_MIN_SUPPORT
)
_FUSED_EVIDENCE_LEARNED_FAMILY_CONFLICT_MAX_CANDIDATE_SUPPORT = 0.25
_REPORT_LABEL_BLOCKING_ORTHOGONAL_AXES = frozenset(
    {
        "amplification_status",
        "copy_number_p53",
        "driver_mutation",
        "hematologic_risk_status",
        "mismatch_repair",
        "molecular_status",
        "polymerase_epsilon",
        "viral_status",
    }
)
_PAN_SIGNATURE_MARKER_PROGRAM_MIN_SUPPORT = 0.45
_LOCAL_REFERENCE_TOP_MARKERS = 24
_LOCAL_REFERENCE_MIN_TPM = 5.0
_LOCAL_REFERENCE_MIN_LOG2_VS_PAN = 1.0
_LOCAL_REFERENCE_MIN_CONTEXT_SUPPORT = 0.45
_LOCAL_REFERENCE_MIN_MARKER_FRACTION = 0.45
_LOCAL_REFERENCE_MIN_MARKER_FRACTION_BY_CODE = {
    "MBL": 0.85,
}
_LOCAL_REFERENCE_MIN_BURDEN_RATIO = 0.10
# Anchor used to turn the burden ratio into a [0,1] score in the
# local-reference support computation. ``burden_support = min(burden_ratio
# / _ANCHOR, 1.0)`` saturates at this burden ratio — chosen empirically
# such that a panel hitting ~35% of its expected reference burden scores
# 1.0 on the burden term, leaving the remaining headroom for marker
# fraction and context support. Independent of the *blocker* threshold
# above (``_MIN_BURDEN_RATIO = 0.10``), which only gates whether the
# panel fires at all.
_LOCAL_REFERENCE_BURDEN_SCORE_ANCHOR = 0.35
_LOCAL_REFERENCE_MIN_SUPPORT = 0.65
_LOCAL_REFERENCE_NEAR_TOP_CONTEXT_MIN_SUPPORT = 0.90
_LOCAL_REFERENCE_NEAR_TOP_MAX_TOP_RATIO = 1.10
_LOCAL_REFERENCE_DECONVOLVED_PRIORITY_BONUS = 0.06
_LOCAL_REFERENCE_CONTEXT_FOCUS_PRIORITY_BONUS = 0.08
_LOCAL_REFERENCE_FAMILY_SPECIFICITY_PRIORITY_BONUS = 0.03
_LOCAL_REFERENCE_STATUS_PARENT_PRIORITY_PENALTY = 0.05
_LOCAL_REFERENCE_HIGH_CONFIDENCE_SUPPORT = 0.90
_LOCAL_REFERENCE_HIGH_CONFIDENCE_CONTEXT_SUPPORT = 0.85
_LOCAL_REFERENCE_HIGH_CONFIDENCE_MARKER_FRACTION = 0.75
_LOCAL_REFERENCE_HIGH_CONFIDENCE_BURDEN_RATIO = 0.50
_LOCAL_REFERENCE_SIGNATURE_ANCHOR_MIN_SUPPORT = 0.90
_LOCAL_REFERENCE_SIGNATURE_ANCHOR_MIN_MARKER_FRACTION = 0.80
_LOCAL_REFERENCE_SIGNATURE_ANCHOR_MIN_BURDEN_RATIO = 0.50
_LOCAL_REFERENCE_SIGNATURE_ANCHOR_MIN_SPECIFIC_MARKERS = 2
_LOCAL_REFERENCE_SIGNATURE_ANCHOR_MIN_SPECIFIC_FRACTION = 0.30
_LOCAL_REFERENCE_SKIP_FAMILIES = frozenset({"rare"})
# Cross-lineage local-reference promotions need a clean ontology marker program.
# ``mixed`` means expected-low/conflicting markers are also present; that can be
# useful annotation, but it is not enough to relabel a sample across lineages.
_LOCAL_REFERENCE_CROSS_LINEAGE_MARKER_STATUSES = frozenset({"consistent"})
_LOCAL_REFERENCE_SIGNATURE_ANCHOR_GENERIC_PRIMARY_TISSUES = frozenset(
    {"", "soft_tissue", "connective_tissue", "smooth_muscle"}
)
_PERMISSIVE_BACKGROUND_EXPECTED_LOW_GENES = frozenset(
    {
        "PTPRC",
        "MS4A1",
        "CD3D",
        "CD3E",
        "CD79A",
        "CD79B",
    }
)
_UNEXPECTED_LOW_GENE_LINEAGE: dict[str, str] = {
    "EPCAM": "epithelial",
    "KRT5": "epithelial",
    "KRT8": "epithelial",
    "KRT14": "epithelial",
    "KRT18": "epithelial",
    "KRT19": "epithelial",
    "ACTA2": "mesenchymal",
    "DES": "mesenchymal",
    "MYOD1": "mesenchymal",
    "MYOG": "mesenchymal",
}
# Only a defining fusion constrains whether the entity itself may be selected.
# ``subtype`` means that a molecular subset exists inside the diagnosis (for
# example TFE3-rearranged PEComa); it does not make that fusion a prerequisite
# for the broader entity.
_FUSION_DRIVEN_REFERENCE_STATES = frozenset({"defining"})
_LOCAL_REFERENCE_SIGNATURE_ANCHOR_BLOCKED_LINEAGES_BY_PRIMARY_TISSUE = {
    "": frozenset({"epithelial"}),
    "soft_tissue": frozenset({"epithelial"}),
    "connective_tissue": frozenset({"epithelial"}),
    "smooth_muscle": frozenset({"epithelial"}),
    # SOX10/S100B/peripheral-nerve programs overlap CNS glial and melanocytic
    # tumors. Without explicit sarcoma/SARC context they are useful annotation,
    # not enough to relabel a coherent GBM/LGG/SKCM/UVM first-pass call.
    "nerve_sheath": frozenset({"epithelial", "melanocytic", "neural"}),
    # Retinal differentiation can overlap neuroendocrine programs; require
    # CNS/embryonal-compatible context before it can displace SCLC/PCPG-like
    # broad RNA evidence.
    "retina": frozenset({"neuroendocrine"}),
}
_LOCAL_REFERENCE_GENERIC_ANCHOR_MARKERS = frozenset(
    {
        "ACTA2",
        "CD34",
        "CDK4",
        "COL1A1",
        "COL1A2",
        "EPCAM",
        "FRS2",
        "HMGA2",
        "KRT5",
        "KRT8",
        "KRT14",
        "KRT18",
        "MDM2",
        "PTPRC",
        "SOX2",
        "TP63",
        "TSPAN31",
        "VIM",
        "YEATS4",
    }
)
_COARSE_REFERENCE_MIN_RHO = 0.75
_COARSE_REFERENCE_MIN_MARGIN = 0.03
_COARSE_REFERENCE_MIN_TYPE_SPECIFIC_HITS = 2
_COARSE_REFERENCE_TISSUE_TIE_WINDOW = 0.02
_COARSE_REFERENCE_MIN_TISSUE_TIE_SCORE = 0.75
_COARSE_REFERENCE_TISSUE_TIE_MIN_DELTA = 0.03
_COARSE_REFERENCE_MIN_PRIMARY_TISSUE_SUPPORT = 0.75
_CONTRAST_DISCRIMINATOR_MIN_CONTEXT_SUPPORT = 0.60
_CONTRAST_DISCRIMINATOR_MIN_SCORE = 0.65
_CONTRAST_DISCRIMINATOR_MIN_MARGIN = 0.25
_CONTRAST_DISCRIMINATOR_MIN_PRIMARY_HITS = 1
_CONTRAST_DISCRIMINATOR_MIN_TOTAL_HITS = 2
_CONTRAST_DISCRIMINATOR_STRONG_SCORE = 0.80
_CONTRAST_DISCRIMINATOR_STRONG_MARGIN = 0.35
_CONTRAST_DISCRIMINATOR_STRONG_TOTAL_HITS = 3
_CONTRAST_DISCRIMINATOR_PRIMARY_HIGH_TPM = 5.0
_CONTRAST_DISCRIMINATOR_SUPPORTING_HIGH_TPM = 2.0
_CONTRAST_DISCRIMINATOR_LOW_TPM = 1.0
_ENTITY_CONSENSUS_MARKER_AXIS = "curated_marker_program"
_ENTITY_CONSENSUS_REFERENCE_AXIS = "exact_expression_reference"
_ENTITY_CONSENSUS_COMPOSITION_AXIS = "composition_reference"
_ENTITY_CONSENSUS_RESIDUAL_AXIS = "decomposition_residual_identity"
_ADCC_PROMOTING_MIN_MYB_TPM = 20.0
_ADCC_STRONG_MYB_AXIS_TPM = 75.0
_ADCC_LOW_MYB_BASAL_BREAST_MIN_SCORE = 0.75
_ADCC_LUMINAL_BREAST_MIN_SCORE = 0.60
_ADCC_LUMINAL_BREAST_MIN_POSITIVE_HITS = 4
_ADCC_LOW_MYB_BASAL_BREAST_MIN_BRCA_CONTEXT = 0.75
_MTC_NEURAL_CREST_CONTEXT_CODES = frozenset({"PCPG"})
_MTC_NEURAL_CREST_COMPETITOR_CODES = ("NBL_MYCNnonamp", "NBL_MYCNamp", "PCPG")
_MTC_SPECIFIC_ANCHOR_MIN_TPM = {
    "CEACAM5": 5.0,
    "CALCR": 5.0,
    "RET": 50.0,
}
_HEPB_ADULT_LIVER_CONTEXT_CODES = frozenset({"LIHC"})
_HEPB_FETAL_ANCHOR_MIN_TPM = {
    "DLK1": 5.0,
    "SALL4": 5.0,
    "IGF2": 100.0,
}
_FUSION_DEFINED_NATIVE_TISSUE_MIN_SCORE = 0.75
_FUSION_DEFINED_NATIVE_TISSUE_TIE_WINDOW = 0.02
_FUSION_DEFINED_STRONG_RNA_SURROGATES = {
    "ADCC": (("MYB", _ADCC_STRONG_MYB_AXIS_TPM), ("MYBL1", _ADCC_STRONG_MYB_AXIS_TPM)),
    "NUTM": (("NUTM1", 10.0),),
}
# Keyed on pirlygenes' lineage-only family ontology (5.12+). Retired
# ``pediatric-*`` / ``net`` keys are kept as harmless back-compat aliases (older
# pirlygenes); ``test_taxonomy_robustness`` guarantees every *live* family is
# covered so ongoing pirlygenes curation can't silently leave one unmapped.
_LOCAL_REFERENCE_CONTEXT_CODES_BY_FAMILY = {
    "sarcoma": ("SARC",),
    "pediatric-bone": ("SARC",),
    "pediatric-soft": ("SARC",),
    # No TCGA salivary cohort exists. Use the broad epithelial cohorts that
    # salivary/lacrimal/breast ADCC medoids actually project onto, then require
    # the ADCC marker/fusion axis before a fusion-defined salivary code can
    # become the report label.
    "salivary": ("HNSC", "LUAD", "BRCA"),
    "heme-bcell": ("DLBC", "LAML", "THYM"),
    "heme-plasma": ("DLBC", "LAML", "THYM"),
    "heme-tcell": ("DLBC", "LAML", "THYM"),
    "heme-myeloid": ("LAML", "DLBC", "THYM"),
    "cns": ("GBM", "LGG"),
    "pediatric-cns": ("GBM", "LGG"),
    "embryonal": ("GBM",),
    "pediatric-liver": ("LIHC", "CHOL"),
    "neuroendocrine": ("PCPG", "LUAD", "LUSC", "PAAD"),
    "net": ("PCPG", "LUAD", "LUSC", "PAAD"),
    "pediatric-net": ("PCPG", "LUAD", "LUSC"),
}


@dataclass
class CancerTypeEvidence:
    """Accumulated evidence for one cancer type."""

    cancer_type: str
    expression_reference_cancer_type: str = ""
    reference_cancer_type: str = ""
    broad_rna_support: float = 0.0
    broad_rna_rank: int | None = None
    related_context_code: str = ""
    related_context_support: float = 0.0
    related_context_is_top: bool = False
    rna_marker_support: float = 0.0
    fine_reference_support: float = 0.0
    direct_fusion_support: float = 0.0
    family_marker_support: float = 0.0
    background_label_support: float = 0.0
    coarse_composition_support: float = 0.0
    contrast_discriminator_support: float = 0.0
    learned_expression_support: float = 0.0
    report_label_candidate: bool = False
    can_select_report_label: bool = False
    blocking_reasons: tuple[str, ...] = ()
    selected_by: str = ""
    label_status: str = "not_considered"
    label_basis: str = ""
    evidence_sources: tuple[str, ...] = ()
    basis: str = ""
    confirmatory_tests: str = ""
    caveat: str = ""
    source: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    adjudication_support: dict[str, dict[str, float]] = field(
        default_factory=dict,
        repr=False,
    )
    adjudication_exclusions: dict[str, dict[str, tuple[str, ...]]] = field(
        default_factory=dict,
        repr=False,
    )
    selection_offers: dict[str, tuple[int, float, int]] = field(
        default_factory=dict,
        repr=False,
    )
    selection_priority: tuple[int, float, int] = field(
        default=(0, 0.0, 0), repr=False
    )

    def add_source(self, source: str) -> None:
        if source and source not in self.evidence_sources:
            self.evidence_sources = self.evidence_sources + (source,)

    def admit_adjudication_support(
        self,
        axis: str,
        support: float,
        *,
        selector: str,
        blocking_reasons: list[str] | tuple[str, ...] = (),
    ) -> None:
        """Record selector evidence only when that selector admitted it.

        Raw strengths remain on their ordinary fields for reporting and
        diagnostics.  This separate positive ledger prevents a later
        adjudicator from turning a blocked score back into a vote.
        """

        selector = _clean(selector) or "unspecified"
        reasons = tuple(dict.fromkeys(reason for reason in blocking_reasons if reason))
        if reasons:
            exclusions = self.adjudication_exclusions.setdefault(axis, {})
            existing = exclusions.get(selector, ())
            exclusions[selector] = tuple(
                dict.fromkeys((*existing, *reasons))
            )
            return
        supports = self.adjudication_support.setdefault(axis, {})
        supports[selector] = max(
            float(supports.get(selector, 0.0)),
            float(support),
        )

    def exclude_adjudication_support(
        self,
        axis: str,
        *,
        selector: str,
        blocking_reasons: list[str] | tuple[str, ...],
    ) -> None:
        """Invalidate an axis when a cross-panel conflict is found later."""

        selector = _clean(selector) or "unspecified"
        supports = self.adjudication_support.get(axis, {})
        supports.pop(selector, None)
        if not supports:
            self.adjudication_support.pop(axis, None)
        reasons = tuple(dict.fromkeys(reason for reason in blocking_reasons if reason))
        exclusions = self.adjudication_exclusions.setdefault(axis, {})
        existing = exclusions.get(selector, ())
        exclusions[selector] = tuple(
            dict.fromkeys((*existing, *reasons))
        )

    def adjudication_axis_support(self, axis: str) -> float:
        """Strongest independently admitted selector on one evidence axis."""

        return max(self.adjudication_support.get(axis, {}).values(), default=0.0)

    def withdraw_report_label_selector(
        self,
        selector: str,
        *,
        blocking_reasons: list[str] | tuple[str, ...],
    ) -> None:
        """Withdraw one selector without discarding independent selectors."""

        self.selection_offers.pop(selector, None)
        if self.selection_offers:
            selected_by, priority = max(
                self.selection_offers.items(),
                key=lambda item: item[1],
            )
            self.can_select_report_label = True
            self.blocking_reasons = ()
            self.selected_by = selected_by
            self.label_status = "selected"
            self.label_basis = selected_by
            self.selection_priority = priority
            return
        self.can_select_report_label = False
        self.label_status = "blocked"
        self.label_basis = selector
        self.blocking_reasons = tuple(
            dict.fromkeys((*self.blocking_reasons, *blocking_reasons))
        )
        self.selection_priority = (0, 0.0, 0)

    def consider_for_report_label(
        self,
        *,
        selected_by: str,
        can_select: bool,
        blocking_reasons: list[str] | tuple[str, ...],
        priority: tuple[int, float],
    ) -> None:
        self.report_label_candidate = True
        # Append a stable tie-break rank so two evidence kinds that compute
        # identical (class_rank, strength) resolve in a documented order
        # instead of falling back to dict-insertion order.
        tiebreak = _SELECTED_BY_TIEBREAK_RANK.get(selected_by, 0)
        full_priority = (priority[0], priority[1], tiebreak)
        if can_select:
            if (
                self.can_select_report_label
                and self.selected_by
                and self.selected_by not in self.selection_offers
            ):
                self.selection_offers[self.selected_by] = self.selection_priority
            self.selection_offers[selected_by] = max(
                self.selection_offers.get(selected_by, (0, 0.0, 0)),
                full_priority,
            )
            selected_by, full_priority = max(
                self.selection_offers.items(),
                key=lambda item: item[1],
            )
            self.can_select_report_label = True
            self.blocking_reasons = ()
            self.selected_by = selected_by
            self.label_status = "selected"
            self.label_basis = selected_by
            self.selection_priority = full_priority
        elif not self.can_select_report_label and blocking_reasons:
            self.selected_by = self.selected_by or selected_by
            self.label_status = "blocked"
            self.label_basis = selected_by
            self.blocking_reasons = tuple(blocking_reasons)

    def public_dict(self) -> dict[str, Any]:
        expression_reference = (
            self.expression_reference_cancer_type
            or self.reference_cancer_type
            or self.cancer_type
        )
        coarse_context = self.reference_cancer_type or expression_reference
        metrics = {
            "pan_cancer_signature_support": round(float(self.broad_rna_support), 4),
            "broad_rna_support": round(float(self.broad_rna_support), 4),
            "related_context_support": round(float(self.related_context_support), 4),
            "related_context_is_top": bool(self.related_context_is_top),
            "rna_marker_support": round(float(self.rna_marker_support), 4),
            "fine_reference_support": round(float(self.fine_reference_support), 4),
            "direct_fusion_support": round(float(self.direct_fusion_support), 4),
            "family_marker_support": round(float(self.family_marker_support), 4),
            "background_label_support": round(float(self.background_label_support), 4),
            "coarse_composition_support": round(
                float(self.coarse_composition_support),
                4,
            ),
            "contrast_discriminator_support": round(
                float(self.contrast_discriminator_support),
                4,
            ),
            "learned_expression_support": round(
                float(self.learned_expression_support),
                4,
            ),
        }
        row = {
            "cancer_type": self.cancer_type,
            "inferred_cancer_type": self.cancer_type,
            "expression_reference_cancer_type": expression_reference,
            # Back-compat alias for existing report/therapy code.
            "reference_cancer_type": coarse_context,
            "coarse_expression_context_cancer_type": coarse_context,
            "evidence_sources": list(self.evidence_sources),
            "selected_by": self.selected_by,
            "selection_method": _selection_method_label(self.selected_by),
            "label_decision": {
                "status": self.label_status,
                "basis": _selection_method_label(self.selected_by or self.label_basis),
                "reasons": list(self.blocking_reasons),
            },
            "decision_features": _hypothesis_decision_features(self),
            "report_label_candidate": bool(self.report_label_candidate),
            "can_select_report_label": bool(self.can_select_report_label),
            "blocking_reasons": list(self.blocking_reasons),
            "adjudication_admissible_support": {
                axis: round(float(self.adjudication_axis_support(axis)), 4)
                for axis in sorted(self.adjudication_support)
            },
            "adjudication_admissible_support_by_selector": {
                axis: {
                    selector: round(float(support), 4)
                    for selector, support in sorted(supports.items())
                }
                for axis, supports in sorted(self.adjudication_support.items())
            },
            "adjudication_exclusions": {
                axis: {
                    selector: list(reasons)
                    for selector, reasons in sorted(exclusions.items())
                }
                for axis, exclusions in sorted(self.adjudication_exclusions.items())
            },
            "metrics": metrics,
            "lineage_path": _lineage_path_for_code(self.cancer_type),
            "orthogonal_axes": _orthogonal_axes_for_hypothesis(self),
            "evidence_channels": _hypothesis_evidence_channels(self),
            "basis": self.basis,
            "confirmatory_tests": self.confirmatory_tests,
            "caveat": self.caveat,
            "source": self.source,
        }
        if self.broad_rna_rank is not None:
            row["broad_rna_rank"] = self.broad_rna_rank
            row["pan_cancer_signature_rank"] = self.broad_rna_rank
        if self.related_context_code:
            row["related_context_code"] = self.related_context_code
        row.update(self.details)
        return row


@dataclass(frozen=True)
class FineReferenceSpec:
    cancer_type: str
    reference_cancer_type: str
    reference_code: str
    marker_groups: Mapping[str, tuple[str, ...]]
    selection_terms: tuple[tuple[str, float], ...]
    minimum_metrics: Mapping[str, float]
    min_fine_reference_support: float
    basis: str


@dataclass(frozen=True)
class RareRnaPolicy:
    """Per-cancer-type policy for the rare-RNA-marker selector.

    The ``effective_context`` quantity consumed by the marker/context
    score is ``context_support`` unless ``top_is_context`` and
    ``top_context_promotes_to_full`` are both True, in which case it
    saturates to 1.0. Saturation makes "the broad classifier's top
    pick is one of my compatible contexts" enough on its own — this
    is the load-bearing semantic for NUTM (squamous-top fully
    supports the NUT-carcinoma promotion) and for salivary entities
    (their compatible contexts are HNSC, which the broad classifier
    is expected to pick on these samples). It is also the reason the
    *default* policy can promote on top-context alone: marker support
    contributes ``marker_weight * marker_support`` and the saturated
    context contributes ``context_weight``, summing past
    ``min_marker_context_support = 0.75`` when both are strong.
    Flip ``top_context_promotes_to_full=False`` to require real
    context support (i.e. the broad classifier must have spread
    support across the cohort, not just put it at rank 1).
    """

    marker_weight: float = 0.40
    context_weight: float = 0.45
    top_context_weight: float = 0.0
    min_marker_context_support: float = 0.75
    min_related_context_support: float = 0.60
    require_top_context: bool = False
    required_top_context_codes: frozenset[str] = frozenset()
    required_top_lineages: frozenset[str] = frozenset()
    top_context_promotes_to_full: bool = True

    def effective_context_support(
        self,
        *,
        top_is_context: bool,
        related_context_support: float,
    ) -> float:
        """Return the context-support value consumed by the marker score."""
        if top_is_context and self.top_context_promotes_to_full:
            return max(related_context_support, 1.0)
        return related_context_support

    def context_passes(
        self,
        *,
        top_code: str,
        top_is_context: bool,
        related_context_support: float,
    ) -> bool:
        if self.required_top_context_codes:
            return top_code in self.required_top_context_codes
        if self.require_top_context:
            return top_is_context
        return top_is_context or related_context_support >= self.min_related_context_support

    def top_lineage_passes(self, *, top_code: str) -> bool:
        if not self.required_top_lineages:
            return True
        return _broad_lineage_for_code(top_code) in self.required_top_lineages


_FINE_REFERENCE_SPECS = (
    FineReferenceSpec(
        cancer_type="SARC_OS",
        reference_cancer_type="SARC",
        reference_code="SARC_OS",
        marker_groups={
            "osteogenic": _OS_OSTEOGENIC_MARKERS,
            "matrix": _OS_MATRIX_MARKERS,
            "amplicon": _OS_AMPLICON_MARKERS,
            "all_markers": tuple(
                dict.fromkeys(
                    _OS_OSTEOGENIC_MARKERS
                    + _OS_MATRIX_MARKERS
                    + _OS_AMPLICON_MARKERS
                )
            ),
        },
        selection_terms=(
            ("related_context_support", 0.25),
            ("all_marker_fraction", 0.25),
            ("osteogenic_burden_ratio", 0.30),
            ("osteogenic_marker_fraction", 0.15),
            ("amplicon_marker_fraction", 0.05),
        ),
        minimum_metrics={
            "related_context_support": 0.50,
            "osteogenic_burden_ratio": 0.75,
            "osteogenic_marker_fraction": 0.45,
        },
        min_fine_reference_support=0.70,
        basis=(
            "pan-cancer signature-ranker context is sarcoma-like and TARGET "
            "osteosarcoma reference markers show an osteogenic tumor program"
        ),
    ),
)
_SPECIAL_FINE_REFERENCE_CODES = frozenset(
    spec.cancer_type for spec in _FINE_REFERENCE_SPECS
)
_RARE_RNA_POLICIES = {
    "NUTM": RareRnaPolicy(
        marker_weight=0.55,
        context_weight=0.40,
        min_related_context_support=0.70,
        required_top_lineages=frozenset({"epithelial"}),
    ),
    **{
        code: RareRnaPolicy(
            marker_weight=0.45,
            context_weight=0.45,
            top_context_weight=0.10,
            min_marker_context_support=0.80,
            min_related_context_support=0.80,
            required_top_lineages=frozenset({"epithelial"}),
            top_context_promotes_to_full=False,
        )
        for code in _SALIVARY_CODES
    },
}
_DEFAULT_RARE_RNA_POLICY = RareRnaPolicy()


@dataclass(frozen=True)
class CancerTypeEvidenceChannel:
    """One interpretable signal feeding the staged cancer-type decision."""

    channel: str
    stage: str
    role: str
    code: str = ""
    context_code: str = ""
    support: float = 0.0
    status: str = "informative"
    selects_report_label: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "channel": self.channel,
            "stage": self.stage,
            "role": self.role,
            "status": self.status,
            "selects_report_label": bool(self.selects_report_label),
        }
        if self.code:
            row["code"] = self.code
        if self.context_code:
            row["context_code"] = self.context_code
        if self.support > 0:
            row["support"] = round(float(self.support), 4)
        if self.details:
            row["details"] = dict(self.details)
        return row


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if result != result:
        return float(default)
    return result


def _learned_hierarchy_support(details: Mapping[str, Any]) -> float:
    """Return the strongest candidate-wide learned hierarchy support.

    Older learned-expression rows store the flat classifier's hierarchy context
    as ``learned_expression_hierarchical_context_support``. The candidate-wide
    hierarchy pass stores the same decision signal under the shorter
    ``learned_expression_hierarchy_support`` key. Fused arbitration should see
    either source without caring which stage attached it.
    """

    return max(
        _safe_float(details.get("learned_expression_hierarchical_context_support")),
        _safe_float(details.get("learned_expression_hierarchy_support")),
    )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _safe_bool(value: object, default: bool = False) -> bool:
    text = str(value if value is not None else "").strip().lower()
    if not text or text == "nan":
        return bool(default)
    return text in {"1", "true", "yes", "y"}


def _broad_lineage_for_code(code: str) -> str:
    code = _clean(code)
    if not code:
        return ""
    try:
        from .cancer_type_ontology import broad_lineage
    except ImportError:
        return ""
    try:
        return _clean(broad_lineage(code))
    except Exception:  # noqa: BLE001
        return ""


def _split_semicolon(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _selection_method_label(selected_by: str) -> str:
    # One distinct display label per selector — debugging from the JSON
    # output should not require cross-referencing ``evidence_sources`` to
    # tell which of ``local_expression_reference`` vs ``fine_reference``
    # actually fired (both previously rendered as ``exact_expression_reference``).
    return {
        "primary_expression_match": "primary_expression_match",
        "pan_cancer_signature_ranker": "pan_cancer_signature_ranker",
        "fused_evidence": "fused_evidence",
        "tumor_label_refinement": "tumor_label_refinement",
        "coarse_composition_reference": "coarse_composition_reference",
        "local_expression_reference": "local_expression_reference",
        "fine_reference": "fine_reference",
        "broad_rna_subtype": "pan_cancer_signature_subtype",
        "pan_cancer_signature_subtype": "pan_cancer_signature_subtype",
        "learned_expression_classifier": "learned_expression_classifier",
        "entity_evidence_consensus": "entity_evidence_consensus",
        "contrast_discriminator": "contrast_discriminator",
        "rare_marker": "rna_marker_with_expression_context",
        "direct_fusion": "direct_fusion",
    }.get(selected_by, selected_by)


def _registry_family_for_code(code: str) -> str:
    row = _registry_by_code().get(_clean(code), {})
    family = _clean(row.get("family")).lower()
    if family:
        return family
    return _broad_lineage_for_code(code).lower()


def _decision_stage_for_selector(selected_by: str) -> str:
    # Selector stages are part of the staged_evidence_graph output contract.
    # Add new selectors here when they first call consider_for_report_label.
    if selected_by in {
        "direct_fusion",
        "fused_evidence",
        "fine_reference",
        "local_expression_reference",
        "lineage_panel",
        "broad_rna_subtype",
        "pan_cancer_signature_subtype",
        "learned_expression_classifier",
        "entity_evidence_consensus",
        "rare_marker",
        "contrast_discriminator",
    }:
        return "exact_subtype"
    if selected_by in {
        "coarse_composition_reference",
        "pan_cancer_signature_ranker",
        "primary_expression_match",
        "tumor_label_refinement",
    }:
        return "coarse_type"
    return "unselected"


def _decision_stage_for_hypothesis(
    hypothesis: CancerTypeEvidence | None,
) -> str:
    if hypothesis is None:
        return "unselected"
    if hypothesis.details.get("local_reference_status_child_code"):
        return "coarse_type"
    if (
        hypothesis.details.get("entity_consensus_adjudication_mode")
        == "common_ancestor_abstention"
    ):
        return "coarse_type"
    return _decision_stage_for_selector(hypothesis.selected_by)


def _channel_status(
    hypothesis: CancerTypeEvidence,
    *,
    selector: str = "",
    default: str = "informative",
) -> str:
    if (
        selector
        and hypothesis.can_select_report_label
        and hypothesis.selected_by == selector
    ):
        return "selected_report_label"
    if hypothesis.label_status == "blocked":
        return "blocked"
    return default


def _channel_selects(
    hypothesis: CancerTypeEvidence,
    selector: str,
) -> bool:
    return bool(
        selector
        and hypothesis.can_select_report_label
        and hypothesis.selected_by == selector
    )


def _hypothesis_evidence_channels(
    hypothesis: CancerTypeEvidence,
) -> list[dict[str, Any]]:
    channels: list[CancerTypeEvidenceChannel] = []

    def add(
        *,
        channel: str,
        stage: str,
        role: str,
        support: float,
        selector: str = "",
        code: str = "",
        context_code: str = "",
        status: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        support_value = _safe_float(support)
        if support_value <= 0:
            return
        channels.append(
            CancerTypeEvidenceChannel(
                channel=channel,
                stage=stage,
                role=role,
                code=code or hypothesis.cancer_type,
                context_code=context_code,
                support=support_value,
                status=status
                or _channel_status(hypothesis, selector=selector),
                selects_report_label=_channel_selects(hypothesis, selector),
                details=details or {},
            )
        )

    add(
        channel="pan_cancer_signature_ranker",
        stage="coarse_type",
        role="top_ranked_candidate"
        if hypothesis.broad_rna_rank == 1
        else "ranked_candidate",
        support=hypothesis.broad_rna_support,
        status="candidate_generation" if hypothesis.broad_rna_rank == 1 else "",
        details={
            "rank": hypothesis.broad_rna_rank,
            "legacy_metric_alias": "broad_rna_support",
        }
        if hypothesis.broad_rna_rank is not None
        else None,
    )
    add(
        channel="pan_cancer_signature_ranker",
        stage="coarse_type",
        role="related_signature_context",
        support=hypothesis.related_context_support,
        code=hypothesis.cancer_type,
        context_code=hypothesis.related_context_code,
        status="context_only",
    )
    add(
        channel="fused_evidence",
        stage=_decision_stage_for_selector("fused_evidence"),
        role="integrated_evidence_selection",
        support=hypothesis.details.get("fused_evidence_score"),
        selector="fused_evidence",
        status="selected_report_label"
        if _channel_selects(hypothesis, "fused_evidence")
        else "informative",
        details={
            **(hypothesis.details.get("fused_evidence_components") or {}),
            **(
                {"blockers": hypothesis.details.get("fused_evidence_blockers")}
                if hypothesis.details.get("fused_evidence_blockers")
                else {}
            ),
            **(
                {
                    "primary_context_conflict": hypothesis.details.get(
                        "fused_evidence_primary_context_conflict"
                    )
                }
                if hypothesis.details.get("fused_evidence_primary_context_conflict")
                else {}
            ),
        },
    )
    structured_abstention = (
        hypothesis.details.get("fallback_context_adjudication") or {}
    )
    if (
        isinstance(structured_abstention, Mapping)
        and structured_abstention.get("mode") == "structured_parent_abstention"
    ):
        add(
            channel="entity_evidence_consensus",
            stage=_decision_stage_for_hypothesis(hypothesis),
            role="structured_parent_abstention",
            support=hypothesis.related_context_support,
            selector="entity_evidence_consensus",
            details=structured_abstention,
        )
    entity_consensus = hypothesis.details.get("entity_evidence_consensus") or {}
    if (
        not structured_abstention
        and isinstance(entity_consensus, Mapping)
        and entity_consensus
    ):
        available_axes = sum(
            bool(axis.get("available"))
            for axis in entity_consensus.get("axes") or []
            if isinstance(axis, Mapping)
        )
        candidate_votes = _safe_int(entity_consensus.get("candidate_votes"))
        selected_votes = _safe_int(entity_consensus.get("selected_votes"))
        if hypothesis.cancer_type == entity_consensus.get("candidate_code"):
            supporting_axes = candidate_votes
        elif hypothesis.cancer_type == entity_consensus.get("selected_code"):
            supporting_axes = selected_votes
        elif (
            hypothesis.details.get("entity_consensus_adjudication_mode")
            == "common_ancestor_abstention"
        ):
            # The parent is a neutral abstention rather than either leaf. Its
            # support reflects the smaller opposing bloc that made the leaf
            # conflict material, not the winning leaf's vote count.
            supporting_axes = min(candidate_votes, selected_votes)
        else:
            supporting_axes = 0
        add(
            channel="entity_evidence_consensus",
            stage=_decision_stage_for_hypothesis(hypothesis),
            role=(
                "common_ancestor_abstention"
                if hypothesis.details.get("entity_consensus_adjudication_mode")
                == "common_ancestor_abstention"
                else "independent_axis_entity_adjudication"
            ),
            support=(
                supporting_axes / available_axes if available_axes else 0.0
            ),
            selector="entity_evidence_consensus",
            details=entity_consensus,
        )
    add(
        channel="lineage_panel",
        stage="exact_subtype",
        role="positive_negative_marker_panel",
        support=hypothesis.details.get("lineage_panel_score"),
        selector="lineage_panel",
        details={
            "panel": hypothesis.details.get("lineage_panel_top"),
            "margin": hypothesis.details.get("lineage_panel_margin_over_second"),
            "entity_margin": hypothesis.details.get(
                "lineage_panel_margin_over_competing_entity"
            ),
            "decision_basis": hypothesis.details.get(
                "lineage_panel_decision_basis"
            ),
            "rationale": hypothesis.details.get("lineage_panel_rationale"),
            "out_of_beam_rescue": hypothesis.details.get(
                "lineage_panel_out_of_beam_rescue"
            ),
            "parent_marker_coherence": hypothesis.details.get(
                "lineage_panel_parent_marker_coherence"
            ),
        },
    )
    add(
        channel="learned_expression_classifier",
        stage="exact_subtype",
        role="full_profile_discriminative_vote",
        support=hypothesis.learned_expression_support,
        selector="learned_expression_classifier",
        context_code=hypothesis.details.get("learned_expression_context_code") or "",
        details={
            "probability": hypothesis.details.get("learned_expression_probability"),
            "margin": hypothesis.details.get("learned_expression_margin"),
            "context_support": hypothesis.details.get(
                "learned_expression_context_support"
            ),
            "top_predictions": hypothesis.details.get(
                "learned_expression_top_predictions"
            ),
        },
    )
    stage_map = {
        "compartment": "family",
        "family": "family",
        "entity": "coarse_type",
        "subtype_axis": "exact_subtype",
        "mismatch_repair": "orthogonal_state",
    }
    hierarchy_votes = (
        hypothesis.details.get("learned_expression_hierarchy_votes")
        or hypothesis.details.get("learned_expression_hierarchical_votes")
        or []
    )
    for vote in hierarchy_votes:
        learned_stage = _clean(vote.get("stage"))
        label = _clean(vote.get("label"))
        if not learned_stage or not label:
            continue
        add(
            channel="learned_expression_classifier",
            stage=stage_map.get(learned_stage, learned_stage),
            role=f"hierarchical_{learned_stage}_vote",
            support=_safe_float(vote.get("probability")),
            code=label,
            context_code=hypothesis.cancer_type,
            status="admission_context",
            details={
                "learned_stage": learned_stage,
                "label_space": vote.get("label_space"),
                "margin": vote.get("margin"),
                "top_predictions": vote.get("top_predictions"),
                "training_split_policy": vote.get("training_split_policy"),
                "holdout_top1_accuracy": vote.get("holdout_top1_accuracy"),
                "holdout_medoid_top1_accuracy": vote.get(
                    "holdout_medoid_top1_accuracy"
                ),
                "oof_precision_at_threshold": vote.get(
                    "oof_precision_at_threshold"
                ),
                "oof_top3_recovery": vote.get("oof_top3_recovery"),
                "training_sample_count": vote.get("training_sample_count"),
                "training_cohorts": vote.get("training_cohorts") or [],
                "mismatch_repair": vote.get("details") or {},
            },
        )
    add(
        channel="composition_reference",
        stage="coarse_type",
        role="independent_tissue_composition",
        support=hypothesis.adjudication_axis_support(
            _ENTITY_CONSENSUS_COMPOSITION_AXIS
        ),
        selector="coarse_composition_reference",
        details={
            "rho": (
                hypothesis.details.get("coarse_reference_rho")
                or hypothesis.details.get("coarse_reference_context_rho")
            ),
            "margin": hypothesis.details.get("coarse_reference_margin"),
            "type_specific_hits": hypothesis.details.get(
                "coarse_reference_type_specific_hit_count"
            ),
        },
    )
    local_kind = str(hypothesis.details.get("local_reference_kind") or "")
    exact_channel = (
        "deconvolved_tumor_reference"
        if local_kind == "deconvolved_tumor_reference"
        else "exact_expression_reference"
    )
    orthogonal_axes = _orthogonal_axes_for_hypothesis(hypothesis)
    exact_stage = "orthogonal_state" if orthogonal_axes else "exact_subtype"
    exact_role = (
        "orthogonal_status_reference"
        if orthogonal_axes
        else (
            "tumor_program_reference"
            if exact_channel == "deconvolved_tumor_reference"
            else "exact_reference_match"
        )
    )
    add(
        channel=exact_channel,
        stage=exact_stage,
        role=exact_role,
        support=hypothesis.fine_reference_support,
        selector=hypothesis.selected_by
        if hypothesis.selected_by in {"fine_reference", "local_expression_reference"}
        else "",
        context_code=hypothesis.reference_cancer_type,
        details={
            "reference_kind": local_kind,
            "expression_source": hypothesis.details.get(
                "local_reference_expression_source"
            ),
            "source_cohort": hypothesis.details.get("local_reference_source_cohort"),
            "orthogonal_axes": orthogonal_axes,
        },
    )
    add(
        channel="marker_program",
        stage="family",
        role="family_marker_coherence",
        support=hypothesis.family_marker_support,
        selector="tumor_label_refinement",
        details={
            "family": hypothesis.details.get("tumor_label_family")
            or _registry_family_for_code(hypothesis.cancer_type),
        },
    )
    add(
        channel="purity_attribution",
        stage="coarse_type",
        role="background_aware_tumor_label",
        support=hypothesis.background_label_support,
        selector="tumor_label_refinement",
        context_code=hypothesis.details.get("competing_background_code") or "",
        details={
            "background_family": hypothesis.details.get("competing_background_family"),
        },
    )
    add(
        channel="rare_fusion_anchor",
        stage="exact_subtype",
        role="rna_marker_anchor",
        support=hypothesis.rna_marker_support,
        selector="rare_marker",
        context_code=hypothesis.related_context_code,
    )
    add(
        channel="rare_fusion_anchor",
        stage="exact_subtype",
        role="direct_fusion_anchor",
        support=hypothesis.direct_fusion_support,
        selector="direct_fusion",
    )
    add(
        channel="contrast_discriminator",
        stage="exact_subtype",
        role="local_marker_discriminator",
        support=hypothesis.contrast_discriminator_support,
        selector="contrast_discriminator",
        context_code=hypothesis.details.get("contrast_discriminator_context_code") or "",
        details=hypothesis.details.get("contrast_discriminator_active_ambiguity")
        or {},
    )
    return [channel.public_dict() for channel in channels]


def _candidate_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in (analysis.get("candidate_trace") or [])]


def _candidate_support_by_code(analysis: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in _candidate_rows(analysis):
        code = _clean(row.get("code"))
        if code:
            out[code] = max(out.get(code, 0.0), _safe_float(row.get("support_fraction_of_top")))
    return out


def _coarse_reference_pairs(analysis: Mapping[str, Any]) -> list[tuple[str, float]]:
    signal = analysis.get("healthy_vs_tumor")
    cohorts = list(getattr(signal, "top_tcga_cohorts", None) or [])
    if not cohorts and isinstance(signal, Mapping):
        cohorts = list(signal.get("top_tcga_cohorts") or [])
    pairs: list[tuple[str, float]] = []
    for item in cohorts:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        code = _clean(str(item[0]).removesuffix("_TPM"))
        rho = _safe_float(item[1])
        if code and rho > 0:
            pairs.append((code, rho))
    return pairs


def _coarse_reference_support_by_code(analysis: Mapping[str, Any]) -> dict[str, float]:
    """Return observed correlations without manufacturing relative confidence.

    A weak 0.010 versus 0.009 match must remain weak rather than becoming
    1.0 versus 0.9 after normalization. The selecting composition path below
    owns the absolute-quality and specificity checks.
    """

    out: dict[str, float] = {}
    for code, rho in _coarse_reference_pairs(analysis):
        out[code] = max(out.get(code, 0.0), float(np.clip(rho, 0.0, 1.0)))
    return out


def _add_quality_gated_composition_context_features(
    hypotheses: dict[str, CancerTypeEvidence],
    analysis: Mapping[str, Any],
) -> None:
    """Admit credible raw cohort fits without normalizing weak noise.

    The resolved top cohort is owned by the selecting composition helper: if
    that helper blocks a cross-code promotion, this context pass must not add
    it back. Secondary cohorts are still useful independent evidence when
    their *own* absolute correlation is informative. A top cohort that agrees
    with the broad winner is also safe context because it cannot relabel the
    sample.
    """

    signal = analysis.get("healthy_vs_tumor")
    cancer_hint = _clean(
        _tissue_composition_signal_value(signal, "cancer_hint", "")
    ).lower()
    if cancer_hint not in {"tumor-consistent", "possibly-tumor"}:
        return
    resolved_top = _clean(_resolved_coarse_reference(analysis).get("code"))
    broad_top = _top_code(analysis)
    for code, rho in _coarse_reference_support_by_code(analysis).items():
        if rho < _COARSE_REFERENCE_MIN_RHO or code not in _registry_by_code():
            continue
        hypothesis = _hypothesis(hypotheses, code)
        if hypothesis.details.get(
            "coarse_reference_structural_tissue_only_ambiguity"
        ):
            # The selecting composition pass already established that this is
            # host/structural context rather than tumor identity. Do not
            # re-admit the same correlation through a generic context channel.
            continue
        if (
            code == resolved_top
            and code != broad_top
            and hypothesis.adjudication_axis_support(
                _ENTITY_CONSENSUS_COMPOSITION_AXIS
            )
            <= 0
        ):
            continue
        hypothesis.add_source("coarse_composition_reference")
        hypothesis.details["coarse_reference_context_rho"] = round(rho, 4)
        hypothesis.admit_adjudication_support(
            _ENTITY_CONSENSUS_COMPOSITION_AXIS,
            rho,
            selector="quality_gated_composition_context",
        )


def _tissue_composition_signal_value(
    signal: object,
    attr: str,
    default: Any = None,
) -> Any:
    if isinstance(signal, Mapping):
        return signal.get(attr, default)
    return getattr(signal, attr, default)


def _normal_tissue_score_by_name(analysis: Mapping[str, Any]) -> dict[str, float]:
    signal = analysis.get("healthy_vs_tumor")
    tissues = list(_tissue_composition_signal_value(signal, "top_normal_tissues", []) or [])
    out: dict[str, float] = {}
    for item in tissues:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        tissue = str(item[0] or "").strip().removesuffix("_nTPM")
        score = _safe_float(item[1])
        if tissue and score > 0:
            out[tissue] = max(out.get(tissue, 0.0), score)
    return out


def _primary_tissue_for_code(code: str) -> str:
    try:
        from .tumor_purity import CANCER_TO_TISSUE
    except ImportError:
        return ""
    return str(CANCER_TO_TISSUE.get(_clean(code)) or "").strip()


def _primary_tissue_key_for_code(code: str) -> str:
    tissue = _primary_tissue_for_code(code)
    return _clean(tissue).lower().replace(" ", "_")


_STRUCTURAL_MESENCHYMAL_PRIMARY_TISSUES = frozenset(
    {
        "adipose",
        "adipose_tissue",
        "muscle",
        "nerve",
        "nerve_sheath",
        "peripheral_nerve",
        "skeletal_muscle",
        "smooth_muscle",
        "soft_tissue",
    }
)


def _structural_mesenchymal_tissue_only_ambiguity(
    *,
    top_code: str,
    resolved: Mapping[str, Any],
    margin: float,
    type_specific_count: int,
    primary_tissue_score: float,
) -> bool:
    return bool(
        _code_lineage_token(top_code) == "mesenchymal"
        and _clean(resolved.get("primary_tissue")).lower().replace(" ", "_")
        in _STRUCTURAL_MESENCHYMAL_PRIMARY_TISSUES
        and margin < _COARSE_REFERENCE_MIN_MARGIN
        and type_specific_count < _COARSE_REFERENCE_MIN_TYPE_SPECIFIC_HITS
        and primary_tissue_score >= _COARSE_REFERENCE_MIN_PRIMARY_TISSUE_SUPPORT
        and not bool(resolved.get("tissue_tiebreak_applied"))
    )


def _resolved_coarse_reference(analysis: Mapping[str, Any]) -> dict[str, Any]:
    pairs = _coarse_reference_pairs(analysis)
    if not pairs:
        return {}
    top_code, top_rho = pairs[0]
    normal_scores = _normal_tissue_score_by_name(analysis)
    close = [
        (code, rho)
        for code, rho in pairs[:3]
        if float(top_rho - rho) <= _COARSE_REFERENCE_TISSUE_TIE_WINDOW
    ]
    selected_code = top_code
    selected_rho = top_rho
    tissue_tiebreak_applied = False
    selected_tissue = _primary_tissue_for_code(top_code)
    selected_tissue_score = _normal_tissue_lookup_score(normal_scores, selected_tissue)
    if len(close) > 1 and normal_scores:
        best = max(
            close,
            key=lambda item: (
                _normal_tissue_lookup_score(
                    normal_scores,
                    _primary_tissue_for_code(item[0]),
                ),
                item[1],
            ),
        )
        best_tissue = _primary_tissue_for_code(best[0])
        best_tissue_score = _normal_tissue_lookup_score(normal_scores, best_tissue)
        if (
            best[0] != top_code
            and best_tissue_score >= _COARSE_REFERENCE_MIN_TISSUE_TIE_SCORE
            and best_tissue_score
            >= selected_tissue_score + _COARSE_REFERENCE_TISSUE_TIE_MIN_DELTA
        ):
            selected_code, selected_rho = best
            selected_tissue = best_tissue
            selected_tissue_score = best_tissue_score
            tissue_tiebreak_applied = True
    second_rho = max((rho for code, rho in pairs if code != selected_code), default=0.0)
    # When the tissue tie-break deliberately picks a close non-top cohort, this
    # margin can be negative. Consumers must read it with tissue_tiebreak_applied.
    return {
        "code": selected_code,
        "rho": float(selected_rho),
        "top_code": top_code,
        "top_rho": float(top_rho),
        "second_rho": float(second_rho),
        "margin": float(selected_rho - second_rho),
        "primary_tissue": selected_tissue,
        "primary_tissue_score": float(selected_tissue_score),
        "tissue_tiebreak_applied": bool(tissue_tiebreak_applied),
        "close_codes": [code for code, _rho in close],
    }


def _same_tissue_close_coarse_codes(resolved: Mapping[str, Any]) -> list[str]:
    selected_code = _clean(resolved.get("code"))
    selected_tissue = _primary_tissue_key_for_code(selected_code)
    if not selected_code or not selected_tissue:
        return []
    out: list[str] = []
    for code in resolved.get("close_codes") or []:
        code = _clean(code)
        if (
            code
            and code != selected_code
            and _primary_tissue_key_for_code(code) == selected_tissue
        ):
            out.append(code)
    return out


def _marker_coherence(
    code: str,
    sample_tpm_by_symbol: Mapping[str, float],
) -> dict[str, Any]:
    code = _clean(code)
    if not code or not sample_tpm_by_symbol:
        return {}
    try:
        from .tumor_type_ontology import tumor_type_sanity_check
    except ImportError:
        return {}
    try:
        sanity = tumor_type_sanity_check(code, sample_tpm_by_symbol)
    except Exception:  # noqa: BLE001
        _LOGGER.warning("tumor-type marker coherence failed for %s", code, exc_info=True)
        return {}
    if not sanity:
        return {}
    return {
        "code": _clean(sanity.get("code")),
        "status": _clean(sanity.get("status")),
        "detected": len(sanity.get("expected_high_detected") or []),
        "total": len(sanity.get("expected_high") or []),
        "unexpected_low_detected": len(sanity.get("expected_low_present") or []),
        "unexpected_low_genes": [
            _clean(row.get("gene") or row.get("symbol"))
            for row in (sanity.get("expected_low_present") or [])
            if _clean(row.get("gene") or row.get("symbol"))
        ],
        "required_for_consistent": _safe_int(
            sanity.get("required_high_for_consistent"),
            0,
        ),
        "detected_fraction": round(
            _safe_float(sanity.get("expected_high_detected_fraction")),
            4,
        ),
        "detected_genes": [
            _clean(row.get("gene") or row.get("symbol"))
            for row in (sanity.get("expected_high_detected") or [])
            if _clean(row.get("gene") or row.get("symbol"))
        ],
        "summary": sanity.get("summary") or "",
    }


def _marker_coherence_unexpected_low_count(coherence: Mapping[str, Any]) -> int:
    return _safe_int(coherence.get("unexpected_low_detected"), 0)


def _marker_coherence_unexpected_low_lineages(
    coherence: Mapping[str, Any],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for gene in coherence.get("unexpected_low_genes") or []:
        gene = _clean(gene).upper()
        lineage = _UNEXPECTED_LOW_GENE_LINEAGE.get(gene)
        if lineage:
            out.setdefault(lineage, []).append(gene)
    return out


def _marker_coherence_selection_grade(coherence: Mapping[str, Any]) -> bool:
    status = _clean(coherence.get("status"))
    # Mixed means the positive lineage program is present but expected-low
    # genes are also seen, usually from background/TME; it should add caution,
    # not erase a coherent tumor-lineage call.
    return status in {"consistent", "mixed", "not_evaluable", ""}


def _marker_coherence_strong(coherence: Mapping[str, Any]) -> bool:
    status = _clean(coherence.get("status"))
    detected = _safe_int(coherence.get("detected"), 0)
    total = _safe_int(coherence.get("total"), 0)
    required = _safe_int(coherence.get("required_for_consistent"), 0)
    fraction = _safe_float(coherence.get("detected_fraction"))
    return bool(
        status in {"consistent", "mixed"}
        and total >= 4
        and detected >= max(required, 3)
        and fraction >= 0.75
        and _marker_coherence_unexpected_low_count(coherence) == 0
    )


def _marker_coherence_positive_complete(coherence: Mapping[str, Any]) -> bool:
    status = _clean(coherence.get("status"))
    detected = _safe_int(coherence.get("detected"), 0)
    total = _safe_int(coherence.get("total"), 0)
    required = _safe_int(coherence.get("required_for_consistent"), 0)
    fraction = _safe_float(coherence.get("detected_fraction"))
    return bool(
        status in {"consistent", "mixed"}
        and total >= 4
        and detected >= max(required, 3)
        and fraction >= 0.75
    )


def _permissive_background_expected_low_only(coherence: Mapping[str, Any]) -> bool:
    genes = [
        _clean(gene).upper()
        for gene in (coherence.get("unexpected_low_genes") or [])
        if _clean(gene)
    ]
    if not genes:
        return True
    return bool(
        len(genes) == 1
        and genes[0] in _PERMISSIVE_BACKGROUND_EXPECTED_LOW_GENES
    )


def _pan_signature_marker_program_selectable(coherence: Mapping[str, Any]) -> bool:
    """Return true when ranker marker evidence can admit a candidate.

    Positive marker completeness is stronger than isolated immune background.
    A single permissive expected-low immune marker (for example PTPRC) is kept
    as a caveat but does not erase a complete positive tumor program. Broader
    expected-low mixtures still stay contextual only.
    """

    return bool(
        _marker_coherence_positive_complete(coherence)
        and _permissive_background_expected_low_only(coherence)
    )


def _row_bool(row: Mapping[str, Any], key: str, default: bool = False) -> bool:
    if key not in row:
        return bool(default)
    value = row.get(key)
    if isinstance(value, bool):
        return value
    return _safe_bool(value, default=default)


def _hypothesis_decision_features(
    hypothesis: CancerTypeEvidence,
    *,
    centroid_support: float | None = None,
) -> dict[str, Any]:
    """Return the flat feature view consumed by fused selection and reports."""

    details = hypothesis.details
    marker_coherence = details.get("pan_cancer_signature_marker_coherence") or {}
    local_marker_coherence = details.get("local_reference_marker_coherence") or {}
    if not isinstance(marker_coherence, Mapping):
        marker_coherence = {}
    if not isinstance(local_marker_coherence, Mapping):
        local_marker_coherence = {}
    selected_by = hypothesis.selected_by if hypothesis.can_select_report_label else ""
    centroid = (
        _safe_float(centroid_support)
        if centroid_support is not None
        else _safe_float(details.get("fused_evidence_centroid_support"))
    )
    pan_marker_selectable = bool(
        details.get("pan_cancer_signature_marker_selectable")
        if "pan_cancer_signature_marker_selectable" in details
        else _pan_signature_marker_program_selectable(marker_coherence)
    )
    learned_compartment_support = _safe_float(
        details.get("learned_expression_compartment_support")
    )
    learned_family_support = _safe_float(
        details.get("learned_expression_family_support")
    )
    learned_entity_support = _safe_float(
        details.get("learned_expression_entity_support")
    )
    learned_margin = _safe_float(details.get("learned_expression_margin"))
    learned_hierarchical_context = _learned_hierarchy_support(details)
    learned_flat_lineage_support = _safe_float(
        details.get("learned_expression_flat_lineage_support")
    )
    learned_entity_label = _clean(details.get("learned_expression_entity_label"))
    learned_family_label = _clean(details.get("learned_expression_family_label"))
    learned_top_family_label = _clean(
        details.get("learned_expression_top_family_label")
    )
    learned_top_family_support = _safe_float(
        details.get("learned_expression_top_family_support")
    )
    learned_top_family_margin = _safe_float(
        details.get("learned_expression_top_family_margin")
    )
    learned_strong_entity_call = bool(
        hypothesis.learned_expression_support
        >= _FUSED_EVIDENCE_STRONG_LEARNED_ENTITY_PROBABILITY
        and learned_margin >= _FUSED_EVIDENCE_STRONG_LEARNED_ENTITY_MARGIN
        and learned_entity_support >= _FUSED_EVIDENCE_STRONG_LEARNED_ENTITY_SUPPORT
        and learned_hierarchical_context
        >= _FUSED_EVIDENCE_STRONG_LEARNED_ENTITY_SUPPORT
        and learned_flat_lineage_support
        >= _FUSED_EVIDENCE_STRONG_LEARNED_LINEAGE_SUPPORT
        and learned_entity_label == hypothesis.cancer_type
    )
    learned_family_contradicted = bool(
        learned_top_family_label
        and learned_family_label
        and learned_top_family_label != learned_family_label
        and learned_top_family_support
        >= _FUSED_EVIDENCE_LEARNED_FAMILY_CONFLICT_MIN_TOP_SUPPORT
        and learned_top_family_margin
        >= _FUSED_EVIDENCE_LEARNED_FAMILY_CONTEXT_MIN_MARGIN
        and learned_family_support
        <= _FUSED_EVIDENCE_LEARNED_FAMILY_CONFLICT_MAX_CANDIDATE_SUPPORT
    )
    learned_compartment_anchored_context = bool(
        not learned_family_contradicted
        and learned_compartment_support
        >= _FUSED_EVIDENCE_LEARNED_COMPARTMENT_CONTEXT_MIN_LEARNED
        and centroid >= _FUSED_EVIDENCE_LEARNED_COMPARTMENT_CONTEXT_MIN_CENTROID
        and hypothesis.broad_rna_support
        >= _FUSED_EVIDENCE_LEARNED_COMPARTMENT_CONTEXT_MIN_SIGNATURE
    )
    learned_family_anchored_context = bool(
        learned_family_support >= _FUSED_EVIDENCE_LEARNED_FAMILY_CONTEXT_MIN_SUPPORT
        and learned_top_family_label == learned_family_label
        and learned_top_family_margin
        >= _FUSED_EVIDENCE_LEARNED_FAMILY_CONTEXT_MIN_MARGIN
        and hypothesis.broad_rna_support
        >= _FUSED_EVIDENCE_LEARNED_FAMILY_CONTEXT_MIN_SIGNATURE
        and hypothesis.broad_rna_rank is not None
        and hypothesis.broad_rna_rank
        <= _FUSED_EVIDENCE_LEARNED_FAMILY_CONTEXT_MAX_RANK
        and not _orthogonal_axes_that_block_report_label(hypothesis.cancer_type)
    )
    return {
        "pan_cancer_signature_support": round(
            float(hypothesis.broad_rna_support),
            4,
        ),
        "pan_cancer_signature_rank": hypothesis.broad_rna_rank,
        "pan_cancer_signature_can_select_alone": False,
        "pan_cancer_signature_marker_support": round(
            _safe_float(details.get("pan_cancer_signature_marker_support")),
            4,
        ),
        "pan_cancer_signature_marker_status": _clean(
            details.get("pan_cancer_signature_marker_status")
        ),
        "pan_cancer_signature_marker_selectable": pan_marker_selectable,
        "pan_cancer_signature_marker_unexpected_low_count": (
            _marker_coherence_unexpected_low_count(marker_coherence)
        ),
        "pan_cancer_signature_marker_unexpected_low_genes": list(
            marker_coherence.get("unexpected_low_genes") or []
        ),
        "centroid_support": round(float(centroid), 4),
        "learned_expression_compartment_support": round(
            float(learned_compartment_support),
            4,
        ),
        "learned_expression_compartment_label": _clean(
            details.get("learned_expression_compartment_label")
        ),
        "learned_expression_family_support": round(
            float(learned_family_support),
            4,
        ),
        "learned_expression_family_label": learned_family_label,
        "learned_expression_top_family_label": learned_top_family_label,
        "learned_expression_top_family_support": round(
            learned_top_family_support,
            4,
        ),
        "learned_expression_top_family_margin": round(
            learned_top_family_margin,
            4,
        ),
        "learned_expression_entity_support": round(
            float(learned_entity_support),
            4,
        ),
        "learned_expression_entity_label": learned_entity_label,
        "learned_expression_hierarchy_support": round(
            max(
                learned_hierarchical_context,
                learned_compartment_support,
                learned_family_support,
                learned_entity_support,
            ),
            4,
        ),
        "learned_expression_flat_lineage_support": round(
            learned_flat_lineage_support,
            4,
        ),
        "learned_compartment_anchored_pan_cancer_context": (
            learned_compartment_anchored_context
        ),
        "learned_family_anchored_pan_cancer_context": (
            learned_family_anchored_context
        ),
        "learned_compartment_family_contradicted": learned_family_contradicted,
        "learned_strong_entity_call": learned_strong_entity_call,
        "learned_expression_support": round(
            float(hypothesis.learned_expression_support),
            4,
        ),
        "learned_expression_probability": round(
            _safe_float(details.get("learned_expression_probability")),
            4,
        ),
        "learned_expression_margin": round(
            learned_margin,
            4,
        ),
        "learned_expression_hierarchical_context_support": round(
            learned_hierarchical_context,
            4,
        ),
        "lineage_panel_score": round(
            _safe_float(details.get("lineage_panel_score")),
            4,
        ),
        "lineage_panel_margin_over_second": round(
            _safe_float(details.get("lineage_panel_margin_over_second")),
            4,
        ),
        "fine_reference_support": round(float(hypothesis.fine_reference_support), 4),
        "local_reference_marker_fraction": round(
            _safe_float(details.get("local_reference_marker_fraction")),
            4,
        ),
        "local_reference_marker_burden_ratio": round(
            _safe_float(details.get("local_reference_marker_burden_ratio")),
            4,
        ),
        "local_reference_marker_status": _clean(
            local_marker_coherence.get("status")
        ),
        "local_reference_unexpected_low_lineage_conflict": bool(
            details.get("local_reference_unexpected_low_lineage_conflict")
        ),
        "local_reference_explicit_negative_fusion": bool(
            details.get("local_reference_explicit_negative_fusion")
        ),
        "contrast_discriminator_support": round(
            float(hypothesis.contrast_discriminator_support),
            4,
        ),
        "rare_marker_support": round(float(hypothesis.rna_marker_support), 4),
        "family_marker_support": round(float(hypothesis.family_marker_support), 4),
        "direct_fusion_support": round(float(hypothesis.direct_fusion_support), 4),
        "pre_fused_selector": selected_by,
    }


def _background_top_compartment_conflict(top: Mapping[str, Any]) -> str:
    """Return the centroid compartment when a background-like broad top conflicts.

    Background-like broad labels such as SARC can win by stromal/smooth-muscle
    expression. If the same candidate row carries a whole-profile compartment
    outside that lineage, downstream exact references should treat the
    background label as context rather than a report-scope anchor.
    """
    if not _is_background_like_candidate(top):
        return ""
    compartment_lineage = _lineage_token(top.get("centroid_coarse_lineage"))
    if not compartment_lineage:
        return ""
    top_lineage = _code_lineage_token(_clean(top.get("code")))
    if not top_lineage or top_lineage == compartment_lineage:
        return ""
    if "compartment_in_set" in top and _row_bool(top, "compartment_in_set", True):
        return ""
    return compartment_lineage


def _close_trace_candidate_for_lineage(
    rows: list[dict[str, Any]],
    lineage: str,
    *,
    min_support: float = 0.80,
) -> dict[str, Any]:
    lineage_token = _lineage_token(lineage)
    best: dict[str, Any] = {}
    best_support = 0.0
    for rank, row in enumerate(rows, start=1):
        code = _clean(row.get("code"))
        support = _safe_float(row.get("support_fraction_of_top"))
        if support <= 0 and rank == 1:
            support = 1.0
        if (
            code
            and support >= min_support
            and _code_lineage_token(code) == lineage_token
            and support > best_support
        ):
            best = dict(row)
            best_support = support
    return best


def _local_reference_unexpected_low_lineage_conflict(
    code: str,
    coherence: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    min_support: float = 0.60,
) -> dict[str, Any]:
    code_lineage = _broad_lineage_for_code(code)
    low_lineages = _marker_coherence_unexpected_low_lineages(coherence)
    if not code_lineage or not low_lineages:
        return {}
    rows = _candidate_rows(analysis)
    for lineage, genes in sorted(low_lineages.items()):
        if lineage == code_lineage:
            continue
        candidate = _close_trace_candidate_for_lineage(
            rows,
            lineage,
            min_support=min_support,
        )
        if candidate:
            return {
                "code": _clean(code),
                "code_lineage": code_lineage,
                "conflicting_lineage": lineage,
                "unexpected_low_genes": list(genes),
                "close_candidate": _clean(candidate.get("code")),
                "close_candidate_support": round(
                    _safe_float(candidate.get("support_fraction_of_top")),
                    4,
                ),
                "min_support": float(min_support),
            }
    return {}


def _crc_family_locked_against_non_crc_composition(
    *,
    broad_top_code: str,
    top_code: str,
    type_specific_count: int,
    analysis: Mapping[str, Any],
    sample_tpm_by_symbol: Mapping[str, float],
) -> dict[str, Any]:
    """Return context when a close CRC-family RNA call should not yield to site.

    COAD/READ/CRC subtype labels are intentionally allowed to mix up with one
    another, but a generic GI composition/reference signal should not move a
    tight CRC-family RNA call to another GI site unless that site contributes
    coherent marker evidence. This keeps host/normal tissue context from
    becoming a second cancer-type classifier.
    """
    if _code_has_registry_ancestor(top_code, _CRC_REGISTRY_ROOT):
        return {}
    broad_top_is_crc = _code_has_registry_ancestor(broad_top_code, _CRC_REGISTRY_ROOT)
    crc_candidate = _best_crc_family_trace_candidate(
        analysis,
        min_support=0.75 if broad_top_is_crc else _CRC_FAMILY_COMPETITOR_MIN_SUPPORT,
        min_family_support=0.0 if broad_top_is_crc else _CRC_FAMILY_COMPETITOR_MIN_FAMILY_SUPPORT,
    )
    if not crc_candidate:
        return {}
    support_by_code = _candidate_support_by_code(analysis)
    top_support = _safe_float(support_by_code.get(top_code))
    crc_support = _safe_float(crc_candidate.get("support_fraction_of_top"))
    if not broad_top_is_crc and top_support > 0 and crc_support < 0.95 * top_support:
        return {}
    marker_coherence = _marker_coherence(top_code, sample_tpm_by_symbol)
    if not broad_top_is_crc:
        if not marker_coherence:
            return {}
        if (
            _marker_coherence_strong(marker_coherence)
            and _marker_coherence_unexpected_low_count(marker_coherence) == 0
        ):
            return {}
    return {
        "broad_top_code": broad_top_code,
        "blocked_code": top_code,
        "blocked_code_support_fraction_of_top": round(float(top_support), 4),
        "blocked_code_marker_coherence": marker_coherence,
        "crc_candidates": [
            {
                "code": _clean(crc_candidate.get("code")),
                "support_fraction_of_top": round(float(crc_support), 4),
                "rank": _safe_int(crc_candidate.get("rank"), 0),
                "family_support": round(
                    _safe_float(crc_candidate.get("family_support")),
                    4,
                ),
            }
        ],
        "required_type_specific_hits": _COARSE_REFERENCE_MIN_TYPE_SPECIFIC_HITS,
        "observed_type_specific_hits": int(type_specific_count),
    }


def _best_crc_family_trace_candidate(
    analysis: Mapping[str, Any],
    *,
    min_support: float = _CRC_FAMILY_COMPETITOR_MIN_SUPPORT,
    min_family_support: float = _CRC_FAMILY_COMPETITOR_MIN_FAMILY_SUPPORT,
    max_rank: int = _CRC_FAMILY_COMPETITOR_MAX_RANK,
) -> dict[str, Any]:
    """Return the strongest close COAD/READ/CRC trace row with marker support."""

    best: dict[str, Any] = {}
    best_priority = 0.0
    for rank, row in enumerate(_candidate_rows(analysis), start=1):
        if rank > max_rank:
            break
        code = _clean(row.get("code"))
        if not code or not _code_has_registry_ancestor(code, _CRC_REGISTRY_ROOT):
            continue
        support = _safe_float(row.get("support_fraction_of_top"))
        family_support = _family_marker_support(row)
        if support < min_support or family_support < min_family_support:
            continue
        priority = support + 0.20 * family_support - 0.01 * rank
        if not best or priority > best_priority:
            best = {
                "code": code,
                "rank": rank,
                "support_fraction_of_top": round(float(support), 4),
                "signature_score": round(_safe_float(row.get("signature_score")), 4),
                "family_label": _clean(row.get("family_label")),
                "family_score": round(_safe_float(row.get("family_score")), 4),
                "family_support": round(float(family_support), 4),
                "min_support": float(min_support),
                "min_family_support": float(min_family_support),
            }
            best_priority = float(priority)
    return best


def _top_label_can_yield_to_crc_family(
    top: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Context for a likely immune/background top label yielding to CRC evidence."""

    top_code = _clean(top.get("code"))
    if not top_code or _code_has_registry_ancestor(top_code, _CRC_REGISTRY_ROOT):
        return {}
    top_lineage = _code_lineage_token(top_code)
    if top_lineage != "hematolymphoid":
        return {}
    crc_candidate = _best_crc_family_trace_candidate(analysis)
    if not crc_candidate:
        return {}
    return {
        "competing_top_code": top_code,
        "competing_top_lineage": top_lineage,
        "competing_top_support_fraction_of_top": round(
            _safe_float(top.get("support_fraction_of_top")),
            4,
        ),
        "crc_candidate": crc_candidate,
        "reason": (
            "hematolymphoid top-ranked RNA context has a close CRC-family "
            "candidate with explicit family-marker support"
        ),
    }


def _mtc_neural_crest_conflict(
    *,
    top_code: str,
    support_by_code: Mapping[str, float],
    sample_tpm_by_symbol: Mapping[str, float],
) -> dict[str, Any]:
    """Return why an MTC RNA prompt is not specific enough in neural-crest context."""
    pcpg_support = max(
        float(top_code in _MTC_NEURAL_CREST_CONTEXT_CODES),
        max(
            (
                _safe_float(support_by_code.get(context_code))
                for context_code in _MTC_NEURAL_CREST_CONTEXT_CODES
            ),
            default=0.0,
        ),
    )
    if pcpg_support < 0.90:
        return {}
    anchors = {
        symbol: _safe_float(sample_tpm_by_symbol.get(symbol))
        for symbol, threshold in _MTC_SPECIFIC_ANCHOR_MIN_TPM.items()
        if _safe_float(sample_tpm_by_symbol.get(symbol)) >= threshold
    }
    if anchors:
        return {}
    competitors: list[dict[str, Any]] = []
    for competitor_code in _MTC_NEURAL_CREST_COMPETITOR_CODES:
        coherence = _marker_coherence(competitor_code, sample_tpm_by_symbol)
        if _marker_coherence_strong(coherence):
            competitors.append(
                {
                    "code": competitor_code,
                    "status": coherence.get("status"),
                    "detected": coherence.get("detected"),
                    "total": coherence.get("total"),
                    "detected_genes": coherence.get("detected_genes") or [],
                }
            )
    if not competitors:
        return {}
    return {
        "context_code": "PCPG",
        "context_support": round(float(pcpg_support), 4),
        "specific_anchor_thresholds": dict(_MTC_SPECIFIC_ANCHOR_MIN_TPM),
        "competing_marker_programs": competitors,
    }


def _hepb_adult_liver_context_conflict(
    *,
    top_code: str,
    context_codes: tuple[str, ...],
    support_by_code: Mapping[str, float],
    sample_tpm_by_symbol: Mapping[str, float],
) -> dict[str, Any]:
    """Return details when an HEPB reference lacks fetal-liver anchors.

    Adult LIHC and HEPB share liver/oncofetal markers such as AFP, GPC3,
    EPCAM, KRT8 and KRT18. When the broad RNA context is already coherent
    LIHC, the pediatric label needs at least one fetal-liver anchor before it
    replaces the adult liver call.
    """
    active_contexts = {
        code
        for code in context_codes
        if code in _HEPB_ADULT_LIVER_CONTEXT_CODES
        and _safe_float(support_by_code.get(code)) > 0
    }
    if top_code in _HEPB_ADULT_LIVER_CONTEXT_CODES:
        active_contexts.add(top_code)
    if not active_contexts:
        return {}
    context_support = max(
        (_safe_float(support_by_code.get(code)) for code in active_contexts),
        default=0.0,
    )
    if context_support < 0.75:
        return {}
    observed = {
        symbol: _safe_float(sample_tpm_by_symbol.get(symbol))
        for symbol in _HEPB_FETAL_ANCHOR_MIN_TPM
    }
    passing = {
        symbol: tpm
        for symbol, tpm in observed.items()
        if tpm >= _HEPB_FETAL_ANCHOR_MIN_TPM[symbol]
    }
    if passing:
        return {}
    return {
        "context_codes": sorted(active_contexts),
        "context_support": round(float(context_support), 4),
        "observed_anchor_tpm": {
            symbol: round(float(tpm), 3) for symbol, tpm in observed.items()
        },
        "anchor_thresholds": dict(_HEPB_FETAL_ANCHOR_MIN_TPM),
    }


def _local_reference_cross_lineage_conflict(
    code: str,
    context_codes: tuple[str, ...],
    coherence: Mapping[str, Any],
) -> dict[str, Any]:
    registry = _registry_by_code()
    code_family = _clean(registry.get(_clean(code), {}).get("family")).lower()
    context_families = {
        _clean(registry.get(_clean(context_code), {}).get("family")).lower()
        for context_code in context_codes
    }
    if code_family.startswith("cns-") and any(
        family.startswith("cns-") for family in context_families
    ):
        return {}
    code_lineage = _broad_lineage_for_code(code)
    context_lineages = {
        _broad_lineage_for_code(context_code)
        for context_code in context_codes
        if _broad_lineage_for_code(context_code)
    }
    if not code_lineage or not context_lineages or code_lineage in context_lineages:
        return {}
    total = _safe_int(coherence.get("total"), 0)
    if total <= 0:
        return {}
    status = _clean(coherence.get("status"))
    unexpected_low_count = _marker_coherence_unexpected_low_count(coherence)
    if (
        status in _LOCAL_REFERENCE_CROSS_LINEAGE_MARKER_STATUSES
        and unexpected_low_count <= 0
    ):
        return {}
    detected = _safe_int(coherence.get("detected"), 0)
    required = _safe_int(coherence.get("required_for_consistent"), 0)
    return {
        "code": _clean(code),
        "code_lineage": code_lineage,
        "context_codes": list(context_codes),
        "context_lineages": sorted(context_lineages),
        "marker_status": status or "not_evaluable",
        "detected": detected,
        "total": total,
        "required_for_consistent": required,
        "unexpected_low_detected": unexpected_low_count,
        "unexpected_low_genes": list(coherence.get("unexpected_low_genes") or []),
    }


def _local_reference_background_compartment_conflict(
    code: str,
    panel: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Block mesenchymal exact references against an epithelial compartment.

    Mesenchymal programs can be real biology in an epithelial specimen, but
    they are often matched-normal/stromal signal. If the first-pass broad top
    is background-like, the whole-profile compartment points elsewhere, and a
    close candidate exists in that compartment, the local exact reference is
    annotation unless another selector supplies a more specific tumor anchor.
    """
    if _code_lineage_token(code) != "mesenchymal":
        return {}
    primary_tissue = _clean(panel.get("primary_tissue")).lower()
    rows = _candidate_rows(analysis)
    if not rows:
        return {}
    top = rows[0]
    compartment_lineage = _background_top_compartment_conflict(top)
    if not compartment_lineage:
        return {}
    close_candidate = _close_trace_candidate_for_lineage(
        rows[1:],
        compartment_lineage,
        min_support=0.80,
    )
    if not close_candidate:
        return {}
    return {
        "code": _clean(code),
        "local_reference_primary_tissue": primary_tissue,
        "background_top_code": _clean(top.get("code")),
        "background_top_family": _clean(top.get("family_label")),
        "centroid_compartment_lineage": compartment_lineage,
        "close_compartment_candidate": _clean(close_candidate.get("code")),
        "close_compartment_candidate_support": round(
            _safe_float(close_candidate.get("support_fraction_of_top")),
            4,
        ),
    }


def _strong_conflicting_coarse_reference(
    analysis: Mapping[str, Any],
    code: str,
) -> dict[str, Any]:
    resolved = _resolved_coarse_reference(analysis)
    if not resolved:
        return {}
    top_code = _clean(resolved.get("code"))
    top_rho = _safe_float(resolved.get("rho"))
    code = _clean(code)
    if not top_code or top_code == code or top_rho < _COARSE_REFERENCE_MIN_RHO:
        return {}
    signal = analysis.get("healthy_vs_tumor")
    cancer_hint = _clean(
        _tissue_composition_signal_value(signal, "cancer_hint", "")
    ).lower()
    if cancer_hint not in {"tumor-consistent", "possibly-tumor"}:
        return {}
    fit_quality = analysis.get("fit_quality") or {}
    fit_label = ""
    if isinstance(fit_quality, Mapping):
        fit_label = _clean(fit_quality.get("label")).lower()
    if fit_label not in {"weak", "ambiguous"}:
        return {}
    second_rho = _safe_float(resolved.get("second_rho"))
    margin = _safe_float(resolved.get("margin"))
    type_specific_cohort = _clean(
        str(
            _tissue_composition_signal_value(signal, "type_specific_cohort", "")
            or ""
        ).removesuffix("_TPM")
    )
    type_specific_hits = list(
        _tissue_composition_signal_value(signal, "type_specific_hits", []) or []
    )
    type_specific_count = (
        len(type_specific_hits) if type_specific_cohort == top_code else 0
    )
    primary_tissue_score = _safe_float(resolved.get("primary_tissue_score"))
    same_tissue_close_codes = _same_tissue_close_coarse_codes(resolved)
    tissue_only_same_tissue_ambiguity = bool(
        margin < _COARSE_REFERENCE_MIN_MARGIN
        and type_specific_count < _COARSE_REFERENCE_MIN_TYPE_SPECIFIC_HITS
        and primary_tissue_score >= _COARSE_REFERENCE_MIN_PRIMARY_TISSUE_SUPPORT
        and same_tissue_close_codes
    )
    structural_tissue_only_ambiguity = (
        _structural_mesenchymal_tissue_only_ambiguity(
            top_code=top_code,
            resolved=resolved,
            margin=margin,
            type_specific_count=type_specific_count,
            primary_tissue_score=primary_tissue_score,
        )
    )
    if (
        tissue_only_same_tissue_ambiguity
        and _primary_tissue_key_for_code(code)
        == _primary_tissue_key_for_code(top_code)
    ):
        return {}
    if structural_tissue_only_ambiguity:
        return {}
    if (
        margin < _COARSE_REFERENCE_MIN_MARGIN
        and type_specific_count < _COARSE_REFERENCE_MIN_TYPE_SPECIFIC_HITS
        and primary_tissue_score < _COARSE_REFERENCE_MIN_PRIMARY_TISSUE_SUPPORT
        and not bool(resolved.get("tissue_tiebreak_applied"))
    ):
        return {}
    return {
        "code": top_code,
        "rho": round(float(top_rho), 4),
        "second_rho": round(float(second_rho), 4),
        "margin": round(float(margin), 4),
        "type_specific_hit_count": type_specific_count,
        "fit_label": fit_label,
        "primary_tissue": resolved.get("primary_tissue") or "",
        "primary_tissue_score": round(primary_tissue_score, 4),
        "same_tissue_close_codes": list(same_tissue_close_codes),
        "tissue_tiebreak_applied": bool(resolved.get("tissue_tiebreak_applied")),
    }


def _top_coarse_reference_code(analysis: Mapping[str, Any]) -> str:
    pairs = _coarse_reference_pairs(analysis)
    return pairs[0][0] if pairs else ""


def _context_support_by_code(analysis: Mapping[str, Any]) -> dict[str, float]:
    out = _candidate_support_by_code(analysis)
    for code, support in _coarse_reference_support_by_code(analysis).items():
        out[code] = max(out.get(code, 0.0), support)
    return out


@lru_cache(maxsize=1)
def _contrast_discriminator_rows() -> tuple[dict[str, Any], ...]:
    """Curated two-way expression discriminators shipped by pirlygenes."""
    try:
        from pirlygenes import cancer_type_discriminators_df
    except ImportError:
        return ()
    try:
        # Use the public accessor rather than loading the CSV directly. Since
        # pirlygenes 5.23.45 this validates and exposes the evidence-role and
        # report-promotion safety policy carried by every row.
        df = cancer_type_discriminators_df()
    except (FileNotFoundError, KeyError, ValueError):
        _LOGGER.warning(
            "pirlygenes cancer-type discriminator table unavailable",
            exc_info=True,
        )
        return ()
    if df is None or getattr(df, "empty", True):
        return ()
    rows: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        contrast = _clean(row.get("contrast"))
        type_a = _clean(row.get("type_a")).upper()
        type_b = _clean(row.get("type_b")).upper()
        favors = _clean(row.get("favors")).upper()
        symbol = _clean(row.get("Symbol") or row.get("symbol"))
        if not contrast or not type_a or not type_b or favors not in {type_a, type_b}:
            continue
        if not symbol:
            continue
        rows.append(
            {
                "contrast": contrast,
                "type_a": type_a,
                "type_b": type_b,
                "favors": favors,
                "symbol": symbol,
                "direction": _clean(row.get("direction")).lower() or "high",
                "tier": _clean(row.get("tier")).lower() or "supporting",
                "separability": _clean(row.get("separability")).lower(),
                "source": _clean(row.get("source")),
                "support_type": _clean(row.get("support_type")),
                "source_anchor": _clean(row.get("source_anchor")),
                "evidence_role": _clean(row.get("evidence_role")),
                "promote_report_scope": _safe_bool(
                    row.get("promote_report_scope"),
                    default=False,
                ),
                "validation_scope": _clean(row.get("validation_scope")),
                "conflict_policy": _clean(row.get("conflict_policy")),
            }
        )
    return tuple(rows)


def _contrast_marker_weight(row: Mapping[str, Any]) -> float:
    return 2.0 if _clean(row.get("tier")).lower() == "primary" else 1.0


def _contrast_marker_passes(row: Mapping[str, Any], observed_tpm: float) -> bool:
    direction = _clean(row.get("direction")).lower()
    tier = _clean(row.get("tier")).lower()
    if direction == "low":
        return observed_tpm <= _CONTRAST_DISCRIMINATOR_LOW_TPM
    threshold = (
        _CONTRAST_DISCRIMINATOR_PRIMARY_HIGH_TPM
        if tier == "primary"
        else _CONTRAST_DISCRIMINATOR_SUPPORTING_HIGH_TPM
    )
    return observed_tpm >= threshold


def _contrast_side_signal(
    rows: list[Mapping[str, Any]],
    sample_tpm_by_symbol: Mapping[str, float],
) -> dict[str, Any]:
    total_weight = 0.0
    hit_weight = 0.0
    primary_hits = 0
    total_hits = 0
    marker_details: list[dict[str, Any]] = []
    high_burden = 0.0
    sources: set[str] = set()
    support_types: set[str] = set()
    separability: set[str] = set()
    for row in rows:
        symbol = _clean(row.get("symbol"))
        if not symbol:
            continue
        observed = _safe_float(sample_tpm_by_symbol.get(symbol))
        weight = _contrast_marker_weight(row)
        passes = _contrast_marker_passes(row, observed)
        total_weight += weight
        if passes:
            hit_weight += weight
            total_hits += 1
            if _clean(row.get("tier")).lower() == "primary":
                primary_hits += 1
        if _clean(row.get("direction")).lower() != "low":
            high_burden += observed
        source = _clean(row.get("source"))
        support_type = _clean(row.get("support_type"))
        sep = _clean(row.get("separability")).lower()
        if source:
            sources.add(source)
        if support_type:
            support_types.add(support_type)
        if sep:
            separability.add(sep)
        marker_details.append(
            {
                "gene": symbol,
                "tpm": round(float(observed), 3),
                "direction": _clean(row.get("direction")).lower() or "high",
                "tier": _clean(row.get("tier")).lower() or "supporting",
                "passes": bool(passes),
            }
        )
    marker_details.sort(
        key=lambda item: (
            not bool(item.get("passes")),
            0 if item.get("tier") == "primary" else 1,
            -_safe_float(item.get("tpm")),
            str(item.get("gene") or ""),
        )
    )
    return {
        "score": float(hit_weight / total_weight) if total_weight > 0 else 0.0,
        "primary_hits": primary_hits,
        "total_hits": total_hits,
        "total_markers": len(marker_details),
        "high_burden_tpm": round(float(high_burden), 3),
        "markers": marker_details,
        "sources": sorted(sources),
        "support_types": sorted(support_types),
        "separability": sorted(separability),
    }


def _contrast_minimums_for_signal(signal: Mapping[str, Any]) -> tuple[float, float, int]:
    separability = set(signal.get("separability") or [])
    min_score = _CONTRAST_DISCRIMINATOR_MIN_SCORE
    min_margin = _CONTRAST_DISCRIMINATOR_MIN_MARGIN
    min_total_hits = _CONTRAST_DISCRIMINATOR_MIN_TOTAL_HITS
    if "poor" in separability:
        min_score += 0.10
        min_margin += 0.10
        min_total_hits += 1
    return min_score, min_margin, min_total_hits


def _contrast_signal_is_strong(signal: Mapping[str, Any], margin: float) -> bool:
    return bool(
        _safe_float(signal.get("score")) >= _CONTRAST_DISCRIMINATOR_STRONG_SCORE
        and margin >= _CONTRAST_DISCRIMINATOR_STRONG_MARGIN
        and _safe_int(signal.get("primary_hits"), 0)
        >= _CONTRAST_DISCRIMINATOR_MIN_PRIMARY_HITS
        and _safe_int(signal.get("total_hits"), 0)
        >= _CONTRAST_DISCRIMINATOR_STRONG_TOTAL_HITS
    )


def _contrast_participant_for_code(code: str, type_a: str, type_b: str) -> str:
    """Return the contrast side represented by an exact or descendant code."""
    code = _clean(code)
    participants = (type_a, type_b)
    if code in participants:
        return code
    return next(
        (
            ancestor
            for ancestor in _registry_parent_chain(code)
            if ancestor in participants
        ),
        "",
    )


def _contrast_participant_support(
    participant: str,
    support_by_code: Mapping[str, float],
) -> tuple[float, str]:
    """Return the strongest context support represented by a contrast side."""
    matches = [
        (float(_safe_float(support)), code)
        for code, support in support_by_code.items()
        if _safe_float(support) > 0
        and _code_has_registry_ancestor(code, participant)
    ]
    if not matches:
        return 0.0, ""
    support, matched_code = max(
        matches,
        key=lambda item: (item[0], item[1] == participant, item[1]),
    )
    return support, matched_code


def _contrast_context_code(
    type_a: str,
    type_b: str,
    support_by_code: Mapping[str, float],
    top_code: str,
) -> tuple[str, float, str, str]:
    top_participant = _contrast_participant_for_code(top_code, type_a, type_b)
    candidates: list[tuple[str, float, str]] = []
    for participant in (type_a, type_b):
        support, matched_code = _contrast_participant_support(
            participant,
            support_by_code,
        )
        if participant == top_participant:
            support = max(1.0, support)
            matched_code = top_code
        candidates.append((participant, support, matched_code))
    participant, support, matched_code = max(
        candidates,
        key=lambda item: (item[1], item[0] == top_participant, item[0]),
    )
    if support <= 0:
        return "", 0.0, "", top_participant
    return participant, float(support), matched_code, top_participant


def _contrast_consensus_context(
    analysis: Mapping[str, Any],
    type_a: str,
    type_b: str,
) -> str:
    """Return shared broad/coarse support at the contrast-participant level."""
    broad_participant = _contrast_participant_for_code(
        _top_code(analysis),
        type_a,
        type_b,
    )
    coarse_participant = _contrast_participant_for_code(
        _top_coarse_reference_code(analysis),
        type_a,
        type_b,
    )
    if broad_participant and broad_participant == coarse_participant:
        return broad_participant
    return ""


def _contrast_active_ambiguity(
    *,
    contrast: str,
    type_a: str,
    type_b: str,
    winner_code: str,
    context_code: str,
    context_match_code: str,
    top_code: str,
    top_participant: str,
    context_is_primary: bool,
    broad_uncertain: bool,
    context_marker_incoherent: bool,
    strong_signal: bool,
) -> dict[str, Any]:
    """Describe whether a contrast is local enough to select a label."""
    top_participates = bool(top_participant)
    context_is_top = context_code == top_participant
    same_top = winner_code == top_participant
    same_context = winner_code == context_code
    return {
        "contrast": contrast,
        "participants": [type_a, type_b],
        "winner": winner_code,
        "top_code": top_code,
        "top_participant": top_participant,
        "context_code": context_code,
        "context_match_code": context_match_code,
        "top_participates": bool(top_participates),
        "context_is_top": bool(context_is_top),
        "context_is_primary": bool(context_is_primary),
        "same_top": bool(same_top),
        "same_context": bool(same_context),
        "broad_uncertain": bool(broad_uncertain),
        "context_marker_incoherent": bool(context_marker_incoherent),
        "strong_signal": bool(strong_signal),
        "active_for_report_label": bool(
            top_participates
            and context_is_top
            and not same_top
            and (
                same_context
                or broad_uncertain
                or context_marker_incoherent
            )
        ),
    }


def _add_contrast_discriminator_features(
    hypotheses: dict[str, CancerTypeEvidence],
    sample_tpm_by_symbol: Mapping[str, float],
    analysis: Mapping[str, Any],
) -> None:
    """Apply pirlygenes two-way discriminators as interpretable contrast evidence."""
    if not sample_tpm_by_symbol:
        return
    rows = _contrast_discriminator_rows()
    if not rows:
        return
    registry = _registry_by_code()
    support_by_code = _context_support_by_code(analysis)
    top_code = _top_code(analysis)
    primary_contexts = _primary_context_codes(analysis)
    fit_quality = analysis.get("fit_quality") or {}
    fit_label = (
        _clean(fit_quality.get("label")).lower()
        if isinstance(fit_quality, Mapping)
        else ""
    )
    broad_uncertain = fit_label in {"weak", "ambiguous"}

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_clean(row.get("contrast")), []).append(row)
    activated_nominations: list[tuple[str, str, CancerTypeEvidence]] = []

    for contrast, contrast_rows in grouped.items():
        first = contrast_rows[0]
        type_a = _clean(first.get("type_a")).upper()
        type_b = _clean(first.get("type_b")).upper()
        if type_a not in registry or type_b not in registry:
            continue
        consensus_context = _contrast_consensus_context(
            analysis,
            type_a,
            type_b,
        )
        (
            context_code,
            context_support,
            context_match_code,
            top_participant,
        ) = _contrast_context_code(
            type_a,
            type_b,
            support_by_code,
            top_code,
        )
        if (
            not context_code
            or context_support < _CONTRAST_DISCRIMINATOR_MIN_CONTEXT_SUPPORT
        ):
            continue

        by_side = {
            code: [row for row in contrast_rows if _clean(row.get("favors")).upper() == code]
            for code in (type_a, type_b)
        }
        signal_by_side = {
            code: _contrast_side_signal(side_rows, sample_tpm_by_symbol)
            for code, side_rows in by_side.items()
        }
        winner_code = max(
            (type_a, type_b),
            key=lambda code: (
                _safe_float(signal_by_side[code].get("score")),
                _safe_int(signal_by_side[code].get("primary_hits"), 0),
                _safe_float(signal_by_side[code].get("high_burden_tpm")),
                -1 if code == context_code else 0,
                code,
            ),
        )
        loser_code = type_b if winner_code == type_a else type_a
        winner_signal = signal_by_side[winner_code]
        loser_signal = signal_by_side[loser_code]
        winner_score = _safe_float(winner_signal.get("score"))
        loser_score = _safe_float(loser_signal.get("score"))
        margin = winner_score - loser_score
        min_score, min_margin, min_total_hits = _contrast_minimums_for_signal(
            winner_signal
        )
        if winner_score < min_score or margin < min_margin:
            continue
        if (
            _safe_int(winner_signal.get("primary_hits"), 0)
            < _CONTRAST_DISCRIMINATOR_MIN_PRIMARY_HITS
            or _safe_int(winner_signal.get("total_hits"), 0) < min_total_hits
        ):
            continue
        policy = {
            "evidence_roles": sorted(
                {
                    _clean(row.get("evidence_role"))
                    for row in contrast_rows
                    if _clean(row.get("evidence_role"))
                }
            ),
            "promote_report_scope": bool(
                contrast_rows
                and all(
                    _safe_bool(
                        row.get("promote_report_scope"),
                        default=False,
                    )
                    for row in contrast_rows
                )
            ),
            "validation_scopes": sorted(
                {
                    _clean(row.get("validation_scope"))
                    for row in contrast_rows
                    if _clean(row.get("validation_scope"))
                }
            ),
            "conflict_policies": sorted(
                {
                    _clean(row.get("conflict_policy"))
                    for row in contrast_rows
                    if _clean(row.get("conflict_policy"))
                }
            ),
        }

        same_context = winner_code == context_code
        same_top = winner_code == top_participant
        top_participates = bool(top_participant)
        context_is_top = context_code == top_participant
        context_is_primary = any(
            _contrast_participant_for_code(code, type_a, type_b) == context_code
            for code in primary_contexts
        )
        strong_signal = _contrast_signal_is_strong(winner_signal, margin)
        marker_coherence = _marker_coherence(winner_code, sample_tpm_by_symbol)
        # A parent contrast can be activated by a more specific registry child
        # (for example BRCA_Basal participating in BRCA_vs_SARC_EPITH). Judge
        # whether the active diagnosis is coherent on that matched child, not
        # on the broader participant's potentially different marker program.
        context_marker_coherence_code = context_match_code or context_code
        context_marker_coherence = _marker_coherence(
            context_marker_coherence_code,
            sample_tpm_by_symbol,
        )
        context_marker_incoherent = bool(
            context_marker_coherence
            and not _marker_coherence_selection_grade(context_marker_coherence)
        )
        active_ambiguity = _contrast_active_ambiguity(
            contrast=contrast,
            type_a=type_a,
            type_b=type_b,
            winner_code=winner_code,
            context_code=context_code,
            context_match_code=context_match_code,
            top_code=top_code,
            top_participant=top_participant,
            context_is_primary=context_is_primary,
            broad_uncertain=broad_uncertain,
            context_marker_incoherent=context_marker_incoherent,
            strong_signal=strong_signal,
        )
        blockers: list[str] = []
        # If the contrast agrees with the current top broad RNA label, it is
        # explanatory support only. Let the existing primary-expression row
        # carry that label; otherwise a same-top contrast can accidentally
        # outrank exact rare-reference evidence without changing the code.
        can_select = False if same_top else same_context
        if can_select and (not top_participates or not context_is_top):
            blockers.append(
                "contrast marker evidence is outside the active top-code "
                "ambiguity; coarse-only or secondary-context contrasts are "
                "recorded but do not set the report label"
            )
            can_select = False
        if not can_select and not same_top:
            can_select = bool(
                top_participates
                and context_is_top
                and (broad_uncertain or context_marker_incoherent)
            )
            if not top_participates or not context_is_top:
                blockers.append(
                    f"contrast {contrast} does not resolve the active top RNA "
                    f"context {top_code}; cross-code contrast promotion "
                    "requires the top RNA call to be one side of the contrast"
                )
            if not can_select:
                blockers.append(
                    f"contrast favors {winner_code}, but first-pass context "
                    f"{context_code} is not uncertain or marker-incoherent"
                )
        if (
            can_select
            and not same_context
            and consensus_context
            and consensus_context == context_code
            and not strong_signal
        ):
            blockers.append(
                "pan-cancer signature-ranker context and coarse reference matching both support "
                f"{context_code}; contrast marker evidence is recorded but does "
                "not override that consensus"
            )
            can_select = False
        if (
            can_select
            and marker_coherence
            and not _marker_coherence_selection_grade(marker_coherence)
        ):
            blockers.append(
                f"{winner_code} marker program is {marker_coherence.get('status')} "
                f"({marker_coherence.get('detected')}/"
                f"{marker_coherence.get('total')} expected high markers; "
                f"{marker_coherence.get('required_for_consistent')} required)"
            )
            can_select = False
        if not policy["promote_report_scope"]:
            blockers.append(
                "pirlygenes validates this discriminator as pairwise "
                "hypothesis evidence only; it cannot promote report scope or "
                "supply an entity-consensus vote"
            )
            can_select = False

        support = float(
            np.clip(
                0.55 * winner_score
                + 0.25 * max(margin, 0.0)
                + 0.20 * min(context_support, 1.0),
                0.0,
                1.0,
            )
        )
        # A parent-level contrast agreeing with an active child is explanatory
        # evidence for that child, not a competing report-label hypothesis. If
        # it were placed on a new parent row, later hierarchy/centroid fusion
        # could broaden the already-more-specific child call to its parent.
        evidence_code = (
            top_code
            if same_top and _code_has_registry_ancestor(top_code, winner_code)
            else winner_code
        )
        hypothesis = _hypothesis(hypotheses, evidence_code)
        hypothesis.add_source("contrast_discriminator")
        hypothesis.expression_reference_cancer_type = (
            hypothesis.expression_reference_cancer_type or winner_code
        )
        hypothesis.reference_cancer_type = (
            hypothesis.reference_cancer_type or context_code or winner_code
        )
        hypothesis.related_context_code = hypothesis.related_context_code or context_code
        hypothesis.related_context_support = max(
            hypothesis.related_context_support,
            context_support,
        )
        hypothesis.contrast_discriminator_support = max(
            hypothesis.contrast_discriminator_support,
            support,
        )
        hypothesis.basis = hypothesis.basis or (
            f"{contrast} contrast marker program favors {winner_code} over "
            f"{loser_code} in a {context_code} expression context"
        )
        hypothesis.details.update(
            {
                "contrast_discriminator": contrast,
                "contrast_discriminator_context_code": context_code,
                "contrast_discriminator_context_match_code": context_match_code,
                "contrast_discriminator_context_marker_coherence_code": (
                    context_marker_coherence_code
                ),
                "contrast_discriminator_top_participant": top_participant,
                "contrast_discriminator_context_support": round(
                    float(context_support),
                    4,
                ),
                "contrast_discriminator_context_is_primary": bool(context_is_primary),
                "contrast_discriminator_winner": winner_code,
                "contrast_discriminator_loser": loser_code,
                "contrast_discriminator_winner_score": round(float(winner_score), 4),
                "contrast_discriminator_loser_score": round(float(loser_score), 4),
                "contrast_discriminator_margin": round(float(margin), 4),
                "contrast_discriminator_strong_signal": bool(strong_signal),
                "contrast_discriminator_broad_fit_label": fit_label,
                "contrast_discriminator_consensus_context": consensus_context,
                "contrast_discriminator_active_ambiguity": active_ambiguity,
                "contrast_discriminator_sources": winner_signal.get("sources") or [],
                "contrast_discriminator_support_types": (
                    winner_signal.get("support_types") or []
                ),
                "contrast_discriminator_winner_markers": (
                    winner_signal.get("markers") or []
                )[:8],
                "contrast_discriminator_loser_markers": (
                    loser_signal.get("markers") or []
                )[:8],
                "contrast_discriminator_upstream_policy": policy,
            }
        )
        if marker_coherence:
            hypothesis.details["contrast_discriminator_marker_coherence"] = (
                marker_coherence
            )
        if context_marker_coherence:
            hypothesis.details["contrast_discriminator_context_marker_coherence"] = (
                context_marker_coherence
            )
        hypothesis.admit_adjudication_support(
            _ENTITY_CONSENSUS_MARKER_AXIS,
            support,
            selector="contrast_discriminator",
            blocking_reasons=blockers,
        )
        hypothesis.consider_for_report_label(
            selected_by="contrast_discriminator",
            can_select=can_select,
            blocking_reasons=blockers,
            priority=(1, support),
        )
        activated_nominations.append((contrast, winner_code, hypothesis))

    if activated_nominations:
        try:
            from pirlygenes import cancer_type_discriminator_consensus

            upstream_consensus = cancer_type_discriminator_consensus(
                [
                    (contrast, winner_code)
                    for contrast, winner_code, _ in activated_nominations
                ]
            )
        except (ImportError, KeyError, TypeError, ValueError):
            _LOGGER.warning(
                "pirlygenes contrast consensus unavailable or rejected the "
                "activated nominations; keeping them hypothesis-only",
                exc_info=True,
            )
            upstream_consensus = {
                "status": "unavailable",
                "hypotheses": sorted(
                    {winner for _, winner, _ in activated_nominations}
                ),
                "promote_report_scope": False,
                "report_code": None,
            }
        if not _safe_bool(
            upstream_consensus.get("promote_report_scope"),
            default=False,
        ):
            reason = (
                "pirlygenes discriminator consensus is "
                f"{_clean(upstream_consensus.get('status')) or 'hypothesis-only'}; "
                "pairwise nominations remain contextual evidence and cannot "
                "set the report label"
            )
            for _, _, hypothesis in activated_nominations:
                hypothesis.details["contrast_discriminator_upstream_consensus"] = (
                    upstream_consensus
                )
                hypothesis.exclude_adjudication_support(
                    _ENTITY_CONSENSUS_MARKER_AXIS,
                    selector="contrast_discriminator",
                    blocking_reasons=(reason,),
                )
                hypothesis.withdraw_report_label_selector(
                    "contrast_discriminator",
                    blocking_reasons=(reason,),
                )
        else:
            report_code = _clean(upstream_consensus.get("report_code"))
            for _, winner_code, hypothesis in activated_nominations:
                hypothesis.details["contrast_discriminator_upstream_consensus"] = (
                    upstream_consensus
                )
                if not report_code or winner_code != report_code:
                    reason = (
                        "pirlygenes discriminator consensus did not select "
                        f"{winner_code} as its report-scope entity"
                    )
                    hypothesis.exclude_adjudication_support(
                        _ENTITY_CONSENSUS_MARKER_AXIS,
                        selector="contrast_discriminator",
                        blocking_reasons=(reason,),
                    )
                    hypothesis.withdraw_report_label_selector(
                        "contrast_discriminator",
                        blocking_reasons=(reason,),
                    )


def _top_code(analysis: Mapping[str, Any]) -> str:
    rows = _candidate_rows(analysis)
    if rows:
        return _clean(rows[0].get("code"))
    return _clean(analysis.get("cancer_type"))


def _primary_context_codes(analysis: Mapping[str, Any]) -> tuple[str, ...]:
    codes: list[str] = []
    for code in (_top_code(analysis),):
        if code and code not in codes:
            codes.append(code)
    top_coarse = _top_coarse_reference_code(analysis)
    if top_coarse and top_coarse not in codes:
        codes.append(top_coarse)
    return tuple(codes)


def _matched_local_reference_context(
    context_codes: tuple[str, ...],
    primary_contexts: tuple[str, ...],
    support_by_code: Mapping[str, float],
    *,
    context_is_top: bool,
) -> str:
    if context_is_top:
        primary_match = next(
            (context for context in primary_contexts if context in context_codes),
            "",
        )
        if primary_match:
            return primary_match
    return max(
        context_codes,
        key=lambda context_code: _safe_float(support_by_code.get(context_code)),
    )


def _broad_coarse_consensus_context(analysis: Mapping[str, Any]) -> str:
    top = _top_code(analysis)
    top_coarse = _top_coarse_reference_code(analysis)
    if top and top == top_coarse:
        return top
    return ""


def _adcc_breast_program_conflict(
    sample_tpm_by_gene_id: Mapping[str, float],
    analysis: Mapping[str, Any],
    *,
    myb_tpm: float,
) -> dict[str, Any]:
    """Return mammary-program conflict evidence for no-fusion ADCC RNA calls.

    This is intentionally a combined-program veto, not a single-marker
    exception: ADCC can still be RNA-provisional without a fusion when the
    sample matches the exact ADCC cohort cleanly, but exact-reference support
    should not override a coherent conventional mammary program in
    BRCA-compatible expression context without direct fusion support.
    """
    if not sample_tpm_by_gene_id:
        return {}
    brca_context_support = _safe_float(_context_support_by_code(analysis).get("BRCA"))
    if brca_context_support < _ADCC_LOW_MYB_BASAL_BREAST_MIN_BRCA_CONTEXT:
        return {}
    try:
        from .lineage_panels import LINEAGE_PANELS, score_panel
    except ImportError:
        return {}
    panels = {
        p.name: p
        for p in LINEAGE_PANELS
        if p.name in {"BRCA_BASAL", "BRCA_LUMINAL"}
    }
    if not panels:
        return {}
    try:
        evidence_by_panel = {
            name: score_panel(panel, sample_tpm_by_gene_id)
            for name, panel in panels.items()
        }
    except Exception:  # noqa: BLE001
        _LOGGER.warning(
            "BRCA conflict check failed; continuing without ADCC veto",
            exc_info=True,
        )
        return {}

    evidence = evidence_by_panel.get("BRCA_BASAL")
    if (
        myb_tpm < _ADCC_PROMOTING_MIN_MYB_TPM
        and evidence is not None
        and _safe_float(evidence.score) >= _ADCC_LOW_MYB_BASAL_BREAST_MIN_SCORE
    ):
        return {
            "program": evidence.panel_name,
            "score": round(_safe_float(evidence.score), 4),
            "brca_context_support": round(float(brca_context_support), 4),
            "rationale": evidence.rationale,
            "reason": "low_myb_basal_mammary",
        }

    evidence = evidence_by_panel.get("BRCA_LUMINAL")
    if evidence is None:
        return {}
    score = _safe_float(evidence.score)
    positive_hits = len(getattr(evidence, "high_hits", ()) or ())
    if (
        score < _ADCC_LUMINAL_BREAST_MIN_SCORE
        or positive_hits < _ADCC_LUMINAL_BREAST_MIN_POSITIVE_HITS
    ):
        return {}
    return {
        "program": evidence.panel_name,
        "score": round(score, 4),
        "positive_hits": positive_hits,
        "brca_context_support": round(float(brca_context_support), 4),
        "rationale": evidence.rationale,
        "reason": "luminal_mammary_program",
    }


def _normal_tissue_lookup_score(
    scores_by_name: Mapping[str, float],
    tissue: str,
) -> float:
    tissue = _clean(tissue).lower()
    if not tissue:
        return 0.0
    normalized_scores = {
        _clean(name).lower(): _safe_float(score)
        for name, score in scores_by_name.items()
    }
    variants = {
        tissue,
        tissue.replace("_", " "),
        tissue.replace(" ", "_"),
    }
    return max(
        (normalized_scores.get(variant, 0.0) for variant in variants),
        default=0.0,
    )


def _strong_fusion_defined_rna_surrogate(
    code: str,
    sample_tpm_by_symbol: Mapping[str, float],
) -> dict[str, Any]:
    markers = _FUSION_DEFINED_STRONG_RNA_SURROGATES.get(_clean(code), ())
    if not markers:
        return {}
    best_gene = ""
    best_tpm = 0.0
    best_threshold = 0.0
    for gene, threshold in markers:
        tpm = _safe_float(sample_tpm_by_symbol.get(gene))
        if tpm > best_tpm:
            best_gene = gene
            best_tpm = tpm
            best_threshold = float(threshold)
    return {
        "strong": bool(best_gene and best_tpm >= best_threshold),
        "gene": best_gene,
        "tpm": round(float(best_tpm), 4),
        "threshold_tpm": round(float(best_threshold), 4),
        "markers": [
            {"gene": gene, "threshold_tpm": threshold}
            for gene, threshold in markers
        ],
    }


_FUSION_DRIVER_GENE_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,10}\b")


def _fusion_driver_gene_sets(driver: str) -> tuple[frozenset[str], ...]:
    """Return alternative expected gene sets for a registry driver string."""
    alternatives: list[frozenset[str]] = []
    for part in re.split(r"[;,]", _clean(driver).upper()):
        genes = frozenset(
            gene
            for gene in _FUSION_DRIVER_GENE_RE.findall(part)
            if gene not in {"FUSION", "REARRANGEMENT", "TRANSLOCATION"}
        )
        if genes:
            alternatives.append(genes)
    return tuple(alternatives)


def _fusion_record_genes(record: Any) -> frozenset[str]:
    values: list[Any] = []
    if isinstance(record, Mapping):
        values.extend(
            record.get(key)
            for key in ("gene_a", "gene_b", "pair", "pair_key", "raw_name")
        )
    else:
        values.extend(
            getattr(record, key, "")
            for key in ("gene_a", "gene_b", "pair_key", "raw_name")
        )
        pair = getattr(record, "pair", "")
        if pair:
            values.append(pair)
    genes: set[str] = set()
    for value in values:
        for gene in _FUSION_DRIVER_GENE_RE.findall(str(value or "").upper()):
            if gene not in {"FUSION", "GENE"}:
                genes.add(gene)
    return frozenset(genes)


def _fusion_input_missing_expected_driver(
    *,
    driver: str,
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a blocker when supplied fusion evidence excludes a driver.

    RNA-only expression references for fusion-driven entities are allowed when
    no fusion data were supplied. Once a fusion file is supplied, absence of
    the expected driver genes is treated as negative evidence for that
    fusion-driven refinement unless a direct fusion rule already selected it.
    """
    if not analysis.get("fusion_inputs_supplied"):
        return {}
    expected_sets = _fusion_driver_gene_sets(driver)
    if not expected_sets:
        return {}
    records = tuple(analysis.get("fusion_records") or ())
    observed_sets = tuple(_fusion_record_genes(record) for record in records)
    for expected in expected_sets:
        if any(expected.issubset(observed) for observed in observed_sets):
            return {}
    return {
        "expected_driver": _clean(driver),
        "expected_gene_sets": [sorted(genes) for genes in expected_sets],
        "observed_gene_sets": [sorted(genes) for genes in observed_sets],
        "fusion_record_count": len(records),
    }


def _fusion_input_missing_expected_driver_reason(
    conflict: Mapping[str, Any],
) -> str:
    driver = _clean(conflict.get("expected_driver")) or "the expected driver"
    record_count = int(_safe_float(conflict.get("fusion_record_count")))
    return (
        "fusion-driven expression reference is not allowed to select because "
        f"fusion input was supplied ({record_count} parsed call"
        f"{'' if record_count == 1 else 's'}) but did not contain {driver}"
    )


def _unconfirmed_fusion_defined_context_conflict(
    *,
    code: str,
    context_codes: tuple[str, ...],
    primary_tissue: str,
    analysis: Mapping[str, Any],
    sample_tpm_by_symbol: Mapping[str, float],
) -> dict[str, Any]:
    """Return native-context conflict for fusion-defined RNA refinements.

    A fusion-defined fine label may refine a broad fallback cohort without
    direct fusion evidence when the RNA surrogate is very specific, or when
    the sample itself has expression support for the fine label's primary
    tissue. It should not override a broad/coarse native-context consensus on
    ordinary lineage/background expression alone.
    """
    code = _clean(code)
    if code not in _FUSION_DEFINED_STRONG_RNA_SURROGATES:
        return {}
    consensus_context = _broad_coarse_consensus_context(analysis)
    if (
        not consensus_context
        or consensus_context == code
        or consensus_context not in context_codes
    ):
        return {}
    surrogate = _strong_fusion_defined_rna_surrogate(code, sample_tpm_by_symbol)
    if surrogate.get("strong"):
        return {}

    normal_scores = _normal_tissue_score_by_name(analysis)
    native_tissue = _primary_tissue_for_code(consensus_context)
    native_score = _normal_tissue_lookup_score(normal_scores, native_tissue)
    fine_score = _normal_tissue_lookup_score(normal_scores, primary_tissue)
    fine_primary_tissue_supported = bool(
        fine_score >= _FUSION_DEFINED_NATIVE_TISSUE_MIN_SCORE
        and fine_score + _FUSION_DEFINED_NATIVE_TISSUE_TIE_WINDOW >= native_score
    )
    if fine_primary_tissue_supported:
        return {}
    return {
        "code": code,
        "consensus_context": consensus_context,
        "native_tissue": native_tissue,
        "native_tissue_score": round(float(native_score), 4),
        "fine_primary_tissue": primary_tissue,
        "fine_primary_tissue_score": round(float(fine_score), 4),
        "fine_primary_tissue_supported": bool(fine_primary_tissue_supported),
        "surrogate": surrogate,
    }


def _fusion_defined_context_conflict_reason(
    conflict: Mapping[str, Any],
) -> str:
    code = _clean(conflict.get("code")) or "fusion-defined label"
    context = _clean(conflict.get("consensus_context")) or "native context"
    surrogate = conflict.get("surrogate") or {}
    gene = _clean(surrogate.get("gene")) or "driver-axis"
    tpm = _safe_float(surrogate.get("tpm"))
    threshold = _safe_float(surrogate.get("threshold_tpm"))
    return (
        f"unconfirmed fusion-defined {code} RNA refinement conflicts with a "
        f"native {context} broad/coarse expression consensus; direct fusion "
        f"evidence or strong {gene} RNA surrogate is required "
        f"({gene} {tpm:.1f} TPM < {threshold:.1f} TPM)"
    )


def _near_top_context_is_supported(
    context_codes: tuple[str, ...],
    support_by_code: Mapping[str, float],
    analysis: Mapping[str, Any],
    *,
    local_support: float,
    allow_consensus_override: bool = False,
) -> bool:
    if local_support < 0.90:
        return False
    consensus_context = _broad_coarse_consensus_context(analysis)
    if (
        consensus_context
        and consensus_context not in context_codes
    ):
        if not allow_consensus_override:
            return False
        consensus_lineage = _broad_lineage_for_code(consensus_context)
        if not consensus_lineage:
            return False
        if not any(
            _broad_lineage_for_code(context_code) == consensus_lineage
            for context_code in context_codes
        ):
            return False
    top = _top_code(analysis)
    top_support = _safe_float(support_by_code.get(top))
    if not top or top_support <= 0:
        return False
    context_support = max(
        (_safe_float(support_by_code.get(context_code)) for context_code in context_codes),
        default=0.0,
    )
    if context_support < _LOCAL_REFERENCE_NEAR_TOP_CONTEXT_MIN_SUPPORT:
        return False
    return (top_support / max(context_support, 1e-9)) <= _LOCAL_REFERENCE_NEAR_TOP_MAX_TOP_RATIO


def _is_high_confidence_local_reference(
    *,
    context_support: float,
    marker_fraction: float,
    burden_ratio: float,
    local_support: float,
) -> bool:
    return (
        local_support >= _LOCAL_REFERENCE_HIGH_CONFIDENCE_SUPPORT
        and context_support >= _LOCAL_REFERENCE_HIGH_CONFIDENCE_CONTEXT_SUPPORT
        and marker_fraction >= _LOCAL_REFERENCE_HIGH_CONFIDENCE_MARKER_FRACTION
        and burden_ratio >= _LOCAL_REFERENCE_HIGH_CONFIDENCE_BURDEN_RATIO
    )


def _context_expected_high_genes(context_codes: tuple[str, ...]) -> frozenset[str]:
    if not context_codes:
        return frozenset()
    try:
        from .tumor_type_ontology import tumor_type_ontology_entry
    except ImportError:
        return frozenset()
    genes: set[str] = set()
    for context_code in context_codes:
        try:
            entry = tumor_type_ontology_entry(context_code)
        except Exception:  # noqa: BLE001
            entry = None
        if entry is None:
            continue
        genes.update(getattr(entry, "expected_high_genes", ()) or ())
    return frozenset(_clean(gene) for gene in genes if _clean(gene))


def _local_reference_anchor_specificity(
    marker_coherence: Mapping[str, Any],
    context_codes: tuple[str, ...],
) -> dict[str, Any]:
    detected = [
        _clean(gene)
        for gene in (marker_coherence.get("detected_genes") or [])
        if _clean(gene)
    ]
    if not detected:
        return {
            "detected": 0,
            "specific": 0,
            "specific_fraction": 0.0,
            "specific_genes": [],
        }
    context_genes = _context_expected_high_genes(context_codes)
    generic = _LOCAL_REFERENCE_GENERIC_ANCHOR_MARKERS
    specific_genes = [
        gene
        for gene in detected
        if gene not in context_genes and gene not in generic
    ]
    specific_fraction = float(len(specific_genes) / len(detected))
    return {
        "detected": len(detected),
        "specific": len(specific_genes),
        "specific_fraction": round(specific_fraction, 4),
        "specific_genes": specific_genes,
    }


def _local_reference_signature_anchor_support(
    *,
    marker_fraction: float,
    burden_ratio: float,
    marker_coherence: Mapping[str, Any],
    context_codes: tuple[str, ...],
) -> float:
    """Context-independent exact-reference support from the candidate program.

    Broad TCGA context is allowed to be wrong when a rare/local reference has a
    clean, high-burden marker program. This score is deliberately based on the
    exact panel plus ontology marker coherence, not on the broad cohort rank, so
    it can rescue cases like organ-background-dominated GIST liver metastasis
    without letting weak one-off marker hits select a report label.
    """
    if not _marker_coherence_positive_complete(marker_coherence):
        return 0.0
    if marker_fraction < _LOCAL_REFERENCE_SIGNATURE_ANCHOR_MIN_MARKER_FRACTION:
        return 0.0
    if burden_ratio < _LOCAL_REFERENCE_SIGNATURE_ANCHOR_MIN_BURDEN_RATIO:
        return 0.0
    specificity = _local_reference_anchor_specificity(
        marker_coherence,
        context_codes,
    )
    if (
        _safe_int(specificity.get("specific"), 0)
        < _LOCAL_REFERENCE_SIGNATURE_ANCHOR_MIN_SPECIFIC_MARKERS
    ):
        return 0.0
    if (
        _safe_float(specificity.get("specific_fraction"))
        < _LOCAL_REFERENCE_SIGNATURE_ANCHOR_MIN_SPECIFIC_FRACTION
    ):
        return 0.0
    burden_support = min(burden_ratio / _LOCAL_REFERENCE_BURDEN_SCORE_ANCHOR, 1.0)
    coherence_fraction = _safe_float(marker_coherence.get("detected_fraction"))
    support = (
        0.45 * marker_fraction
        + 0.35 * burden_support
        + 0.20 * coherence_fraction
    )
    return float(support if support >= _LOCAL_REFERENCE_SIGNATURE_ANCHOR_MIN_SUPPORT else 0.0)


def _local_reference_signature_anchor_context_blocked(
    primary_tissue: str,
    context_codes: tuple[str, ...],
) -> bool:
    blocked_lineages = (
        _LOCAL_REFERENCE_SIGNATURE_ANCHOR_BLOCKED_LINEAGES_BY_PRIMARY_TISSUE.get(
            _clean(primary_tissue).lower(),
            frozenset(),
        )
    )
    if not blocked_lineages:
        return False
    context_lineages = {
        _broad_lineage_for_code(context_code)
        for context_code in context_codes
        if _broad_lineage_for_code(context_code)
    }
    return bool(context_lineages & blocked_lineages)


def _local_reference_first_pass_context_blocker(
    primary_tissue: str,
    primary_contexts: tuple[str, ...],
    context_codes: tuple[str, ...],
) -> str:
    first_context = _clean(primary_contexts[0]) if primary_contexts else ""
    if not first_context or first_context in context_codes:
        return ""
    blocked_lineages = (
        _LOCAL_REFERENCE_SIGNATURE_ANCHOR_BLOCKED_LINEAGES_BY_PRIMARY_TISSUE.get(
            _clean(primary_tissue).lower(),
            frozenset(),
        )
    )
    first_lineage = _broad_lineage_for_code(first_context)
    if not first_lineage or first_lineage not in blocked_lineages:
        return ""
    primary = _clean(primary_tissue).lower() or "generic"
    return (
        f"{primary} exact-reference refinement cannot override first-pass "
        f"{first_context} {first_lineage} context without direct compatible "
        "first-pass expression support"
    )


def _local_reference_context_focus(
    context_codes: tuple[str, ...],
    matched_context: str,
) -> float:
    if not context_codes or not matched_context:
        return 0.0
    try:
        from .cancer_type_ontology import broad_lineage
    except ImportError:
        return 0.0
    try:
        matched_lineage = _clean(broad_lineage(matched_context))
    except Exception:  # noqa: BLE001
        matched_lineage = ""
    if not matched_lineage:
        return 0.0
    same_lineage = 0
    considered = 0
    for context in context_codes:
        try:
            lineage = _clean(broad_lineage(context))
        except Exception:  # noqa: BLE001
            continue
        if not lineage:
            continue
        considered += 1
        if lineage == matched_lineage:
            same_lineage += 1
    if not considered:
        return 0.0
    return float(same_lineage / considered)


def _local_reference_family_specificity_bonus(family: str) -> float:
    family = _clean(family).lower()
    if "-" not in family:
        return 0.0
    return _LOCAL_REFERENCE_FAMILY_SPECIFICITY_PRIORITY_BONUS


def _is_molecular_status_expression_source(value: object) -> bool:
    cleaned = _clean(value)
    registry_row = _registry_by_code().get(cleaned, {})
    if (
        _clean(registry_row.get("ontology_kind")).lower()
        == "molecular_status_subtype"
    ):
        return True
    text = cleaned.lower()
    if not text:
        return False
    normalized = text.replace(";", "/").replace(",", "/").replace("_", "/")
    tokens = {token.strip() for token in normalized.split("/") if token.strip()}
    status_terms = {
        "mut",
        "mutation",
        "hpv",
        "pam50",
        "msi",
        "cyto",
        "cytogenetic",
        "cytogenetics",
        "apl",
    }
    return bool((tokens & status_terms) or any(token.startswith("eln") for token in tokens))


def _registry_row_for_code(code: str) -> dict[str, Any]:
    return dict(_registry_by_code().get(_clean(code), {}))


def _registry_parent_chain(code: str) -> tuple[str, ...]:
    registry = _registry_by_code()
    current = _clean(code)
    seen: set[str] = set()
    out: list[str] = []
    while current and current not in seen:
        seen.add(current)
        parent = _clean(registry.get(current, {}).get("parent_code"))
        if not parent:
            break
        out.append(parent)
        current = parent
    return tuple(out)


def _code_has_registry_ancestor(code: str, ancestor: str) -> bool:
    code_text = _clean(code)
    ancestor_text = _clean(ancestor)
    return bool(
        code_text
        and ancestor_text
        and (
            code_text == ancestor_text
            or ancestor_text in _registry_parent_chain(code_text)
        )
    )


def _lineage_token(value: Any) -> str:
    text = _clean(value).lower().replace("-", "_").replace(" ", "_")
    return {
        "epithelial": "epithelial",
        "carcinoma": "epithelial",
        "sarcoma": "mesenchymal",
        "mesenchymal": "mesenchymal",
        "hematolymphoid": "hematolymphoid",
        "heme": "hematolymphoid",
        "melanoma": "melanocytic",
        "melanocytic": "melanocytic",
        "neuroendocrine": "neuroendocrine",
        "neural": "neural",
        "cns": "neural",
        "embryonal": "embryonal",
        "germ": "germ",
        "germ_cell": "germ",
    }.get(text, text)


def _code_lineage_token(code: str) -> str:
    return _lineage_token(_broad_lineage_for_code(code))


def _code_suffix_for_parent(code: str, parent_code: str = "") -> str:
    code_text = _clean(code)
    parent_text = _clean(parent_code)
    if parent_text and code_text.startswith(parent_text + "_"):
        return code_text[len(parent_text) + 1 :]
    if "_" in code_text:
        return code_text.rsplit("_", 1)[-1]
    return ""


def _humanize_status_token(token: str) -> str:
    token = _clean(token)
    if not token:
        return ""
    acronyms = {
        "MSI": "MSI-H / dMMR",
        "MSS": "MSS / MMR-proficient",
        "POLE": "POLE-ultramutated",
        "CNH": "copy-number-high / p53-abnormal",
        "CNL": "copy-number-low / NSMP",
        "HPVpos": "HPV-positive",
        "HPVneg": "HPV-negative",
        "HER2": "HER2-enriched",
        "LumA": "luminal A",
        "LumB": "luminal B",
        "MYCNamp": "MYCN-amplified",
        "MYCNnonamp": "MYCN-non-amplified",
        "ELNfav": "ELN favorable",
        "ELNint": "ELN intermediate",
        "ELNadv": "ELN adverse",
    }
    if token in acronyms:
        return acronyms[token]
    return token.replace("_", " ").replace("-", " ").strip()


def _orthogonal_axes_for_code(
    code: str,
    *,
    support: float = 0.0,
    status: str = "informative",
    evidence_source: str = "registry",
    selects_report_label: bool = False,
) -> list[dict[str, Any]]:
    """Return molecular/status axes encoded by a registry child code.

    Registry parent/ancestor links describe the diagnosis lineage. Some children
    instead encode an orthogonal state (MSI/MSS, POLE, HPV, PAM50, mutation,
    fusion, cytogenetic/risk group). Those states should annotate the lineage
    call rather than force every cancer to model its own bespoke subtype tree.
    """
    code_text = _clean(code)
    row = _registry_row_for_code(code_text)
    if not code_text or not row:
        return []

    parent_code = _clean(row.get("parent_code"))
    suffix = _code_suffix_for_parent(code_text, parent_code)
    suffix_upper = suffix.upper()
    expression_source = _clean(row.get("expression_source"))
    source_cohort = _clean(row.get("source_cohort"))
    ontology_kind = _clean(row.get("ontology_kind")).lower()
    name = _clean(row.get("name"))
    notes = _clean(row.get("notes"))
    viral_agent = _clean(row.get("viral_agent"))
    viral_etiology = _clean(row.get("viral_etiology")).lower()
    fusion_driver = _clean(row.get("fusion_driver"))
    text = " ".join(
        part.lower()
        for part in (
            code_text,
            suffix,
            expression_source,
            source_cohort,
            name,
            notes,
            viral_agent,
            fusion_driver,
        )
        if part
    )

    axes: list[dict[str, Any]] = []

    def add_axis(
        axis: str,
        state: str,
        *,
        system: str = "",
        state_code: str = "",
        confidence: str = "registry",
    ) -> None:
        state_text = _clean(state) or _humanize_status_token(suffix)
        if not axis or not state_text:
            return
        key = (axis, state_text)
        if any((row["axis"], row["state"]) == key for row in axes):
            return
        axis_row: dict[str, Any] = {
            "axis": axis,
            "state": state_text,
            "code": code_text,
            "base_code": parent_code or code_text,
            "parent_code": parent_code,
            "ancestors": list(_registry_parent_chain(code_text)),
            "evidence_source": evidence_source,
            "status": status,
            "selects_report_label": bool(selects_report_label),
            "confidence": confidence,
        }
        if state_code:
            axis_row["state_code"] = state_code
        if system:
            axis_row["system"] = system
        if support > 0:
            axis_row["support"] = round(float(support), 4)
        axes.append(axis_row)

    mmr_state_by_suffix = {
        "MSI": "MSI-H / dMMR",
        "MSIH": "MSI-H / dMMR",
        "DMMR": "MSI-H / dMMR",
        "MMRD": "MSI-H / dMMR",
        "MSS": "MSS / MMR-proficient",
        "PMMR": "MSS / MMR-proficient",
        "MMRP": "MSS / MMR-proficient",
    }
    if suffix_upper in mmr_state_by_suffix:
        state = mmr_state_by_suffix[suffix_upper]
        add_axis("mismatch_repair", state, system="MSI/MMR", state_code=suffix_upper)
    if suffix_upper == "POLE":
        add_axis("polymerase_epsilon", "POLE-ultramutated", system="TCGA molecular class")
    if suffix_upper in {"CNH", "CNL"}:
        add_axis(
            "copy_number_p53",
            _humanize_status_token(suffix) or "copy-number state",
            system="TCGA molecular class",
            state_code=suffix_upper,
        )
    if suffix_upper.startswith("HPV"):
        state = (
            "HPV-negative"
            if suffix_upper.endswith("NEG") or "hpv-negative" in text
            else "HPV-positive"
        )
        add_axis("viral_status", state, system="viral etiology", state_code=suffix)
    elif viral_agent.upper() == "HPV" and viral_etiology == "defining":
        add_axis("viral_status", "HPV-positive", system="viral etiology")
    elif viral_agent and (
        viral_etiology == "defining" or parent_code and viral_etiology == "subtype"
    ):
        add_axis("viral_status", viral_agent, system="viral etiology")
    if (
        "pam50" in expression_source.lower()
        or (
            code_text.startswith("BRCA_")
            and suffix in {"Basal", "HER2", "LumA", "LumB", "Normal"}
        )
    ):
        add_axis(
            "expression_subtype",
            _humanize_status_token(suffix),
            system="PAM50",
            state_code=suffix,
        )
    if "/mut" in expression_source.lower() or "mutation" in expression_source.lower():
        add_axis(
            "driver_mutation",
            _humanize_status_token(suffix),
            system=expression_source or "mutation-defined cohort",
            state_code=suffix,
        )
    if suffix_upper.startswith("ELN") or suffix_upper == "APL":
        add_axis(
            "hematologic_risk_status",
            _humanize_status_token(suffix),
            system="hematologic molecular/risk group",
            state_code=suffix,
        )
    if "MYCN" in suffix_upper:
        add_axis(
            "amplification_status",
            _humanize_status_token(suffix),
            system="copy-number/amplification",
            state_code=suffix,
        )
    if suffix_upper in {"ASCL1", "NEUROD1", "POU2F3", "YAP1"}:
        add_axis(
            "transcriptional_state",
            suffix,
            system="transcription-factor dominance",
            state_code=suffix,
        )
    if (
        not axes
        and parent_code
        and (
            ontology_kind == "molecular_status_subtype"
            or _is_molecular_status_expression_source(expression_source)
            or _is_molecular_status_expression_source(source_cohort)
            or _is_molecular_status_expression_source(code_text)
        )
    ):
        add_axis(
            "molecular_status",
            _humanize_status_token(suffix),
            system=expression_source or source_cohort or "registry status child",
            state_code=suffix,
            confidence="inferred_from_registry_source",
        )

    return axes


def _lineage_path_for_code(code: str) -> list[dict[str, Any]]:
    code_text = _clean(code)
    row = _registry_row_for_code(code_text)
    if not code_text or not row:
        return []
    axes = _orthogonal_axes_for_code(code_text)
    base_code = (
        _clean(row.get("parent_code"))
        if axes and _clean(row.get("parent_code"))
        else code_text
    )
    base_row = _registry_row_for_code(base_code) or row
    ancestors = list(_registry_parent_chain(base_code))
    path: list[dict[str, Any]] = []
    family = _clean(base_row.get("family") or row.get("family"))
    if family:
        path.append({"stage": "family", "family": family})
    for idx, ancestor in enumerate(reversed(ancestors)):
        path.append(
            {
                "stage": "coarse_type" if idx == 0 else "intermediate_type",
                "code": ancestor,
            }
        )
    if base_code:
        stage = "exact_subtype" if ancestors else "coarse_type"
        if not path or path[-1].get("code") != base_code:
            path.append({"stage": stage, "code": base_code})
    return path


def _orthogonal_axes_for_hypothesis(
    hypothesis: CancerTypeEvidence | None,
) -> list[dict[str, Any]]:
    if hypothesis is None:
        return []
    axes = _orthogonal_axes_for_code(
        hypothesis.cancer_type,
        support=max(
            hypothesis.fine_reference_support,
            hypothesis.rna_marker_support,
            hypothesis.direct_fusion_support,
        ),
        status=(
            "selected_report_label"
            if hypothesis.can_select_report_label
            else hypothesis.label_status or "informative"
        ),
        evidence_source=hypothesis.selected_by or "registry",
        selects_report_label=hypothesis.can_select_report_label,
    )
    observed_fusion = _clean(hypothesis.details.get("fusion"))
    if hypothesis.direct_fusion_support > 0 and observed_fusion:
        axes.append(
            {
                "axis": "fusion_driver",
                "state": observed_fusion,
                "code": hypothesis.cancer_type,
                "base_code": hypothesis.cancer_type,
                "parent_code": "",
                "ancestors": list(_registry_parent_chain(hypothesis.cancer_type)),
                "evidence_source": "direct_fusion",
                "status": "selected_report_label"
                if hypothesis.can_select_report_label
                else "informative",
                "selects_report_label": bool(hypothesis.can_select_report_label),
                "confidence": "observed_fusion",
                "support": round(float(hypothesis.direct_fusion_support), 4),
                "system": "fusion",
            }
        )
    child_code = _clean(hypothesis.details.get("local_reference_status_child_code"))
    if child_code and child_code != hypothesis.cancer_type:
        axes.extend(
            _orthogonal_axes_for_code(
                child_code,
                support=hypothesis.fine_reference_support,
                status="supports_parent_label",
                evidence_source="local_expression_reference",
                selects_report_label=False,
            )
        )
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for axis in axes:
        key = (
            str(axis.get("axis") or ""),
            str(axis.get("state") or ""),
            str(axis.get("code") or ""),
        )
        unique[key] = axis
    return list(unique.values())


def _hypothesis(
    hypotheses: dict[str, CancerTypeEvidence],
    code: str,
) -> CancerTypeEvidence:
    code = _clean(code)
    if code not in hypotheses:
        hypotheses[code] = CancerTypeEvidence(cancer_type=code)
    return hypotheses[code]


@lru_cache(maxsize=1)
def _rare_rules_by_id() -> dict[str, dict[str, Any]]:
    try:
        from .rare_inference import rare_cancer_rna_surrogate_rules_df
    except ImportError:
        _LOGGER.warning(
            "rare_cancer_rna_surrogate_rules_df is unavailable; rare-marker "
            "evidence will be empty",
            exc_info=True,
        )
        return {}
    try:
        df = rare_cancer_rna_surrogate_rules_df().fillna("")
    except (FileNotFoundError, KeyError, ValueError):
        _LOGGER.warning(
            "rare_cancer_rna_surrogate_rules_df failed to load; rare-marker "
            "evidence will be empty",
            exc_info=True,
        )
        return {}
    return {
        _clean(row.get("rule_id")): dict(row)
        for _, row in df.iterrows()
        if _clean(row.get("rule_id"))
    }


@lru_cache(maxsize=1)
def _registry_by_code() -> dict[str, dict[str, Any]]:
    try:
        from trufflepig.cancer_ontology import cancer_type_registry
    except ImportError:
        _LOGGER.warning(
            "pirlygenes.gene_sets_cancer.cancer_type_registry is unavailable; "
            "registry lookups will be empty",
            exc_info=True,
        )
        return {}
    try:
        df = cancer_type_registry().fillna("")
    except (FileNotFoundError, KeyError, ValueError):
        _LOGGER.warning(
            "cancer_type_registry failed to load; registry lookups will be empty",
            exc_info=True,
        )
        return {}
    return {
        _clean(row.get("code")): dict(row)
        for _, row in df.iterrows()
        if _clean(row.get("code"))
    }


def _local_reference_context_codes(
    code: str,
    registry_row: Mapping[str, Any],
) -> tuple[str, ...]:
    parent = _clean(registry_row.get("parent_code"))
    if parent:
        return (parent,)
    family = _clean(registry_row.get("family")).lower()
    primary_tissue = _clean(registry_row.get("primary_tissue")).lower()
    if family == "embryonal" and "liver" in primary_tissue:
        return ("LIHC",)
    contexts = _LOCAL_REFERENCE_CONTEXT_CODES_BY_FAMILY.get(family)
    if contexts:
        return tuple(contexts)
    if "-" in family:
        root_family = family.split("-", 1)[0]
        return tuple(_LOCAL_REFERENCE_CONTEXT_CODES_BY_FAMILY.get(root_family, ()))
    return ()


@lru_cache(maxsize=32)
def _local_expression_reference_panels(
    compatible_contexts: tuple[str, ...] = (),
) -> dict[str, dict[str, Any]]:
    """Build marker panels for exact non-TCGA expression references.

    Observed pirlygenes cancer references and trufflepig deconvolved
    references both contribute here. Deconvolved references are preferred
    when both exist for a code because they represent tumor-cell expression;
    observed references fill in codes that do not have a deconvolved artifact.
    The broad RNA classifier still provides the coarse context.

    Tests that monkeypatch ``subtype_deconvolved_expression`` or
    ``cancer_reference_expression`` should call
    ``_local_expression_reference_panels.cache_clear()`` first (and again
    after the test) — the cache key only includes ``compatible_contexts``,
    so a patched underlying frame is not seen otherwise.
    """
    try:
        from pirlygenes.gene_sets_cancer import is_extended_housekeeping_symbol

        from .reference import (
            cancer_reference_expression,
            pan_cancer_expression,
            subtype_deconvolved_expression,
        )
        from .tumor_purity import _compile_excluded_gene_matcher
    except ImportError:
        _LOGGER.warning(
            "trufflepig.reference / tumor_purity imports failed; local "
            "expression reference panels will be empty",
            exc_info=True,
        )
        return {}

    try:
        pan = (
            pan_cancer_expression(technical_rna_normalize=True)
            .drop_duplicates(subset="Symbol")
            .set_index("Symbol")
        )
    except (FileNotFoundError, KeyError, ValueError):
        _LOGGER.warning(
            "pan_cancer_expression failed to load; local expression "
            "reference panels will be empty",
            exc_info=True,
        )
        return {}

    cohort_cols = [col for col in pan.columns if str(col).endswith("_TPM")]
    if not cohort_cols:
        return {}
    pan_median = pan[cohort_cols].astype(float).median(axis=1)
    pan_symbols = set(pan.index.astype(str))
    is_excluded = _compile_excluded_gene_matcher()
    registry = _registry_by_code()
    compatible_context_set = {
        _clean(context) for context in compatible_contexts if _clean(context)
    }
    eligible_codes: set[str] = set()
    for code, row in registry.items():
        family = _clean(row.get("family")).lower()
        if family in _LOCAL_REFERENCE_SKIP_FAMILIES:
            continue
        context_codes = _local_reference_context_codes(code, row)
        if not context_codes:
            continue
        if compatible_context_set and not (
            set(context_codes) & compatible_context_set
        ):
            continue
        eligible_codes.add(code)
    if compatible_context_set and not eligible_codes:
        return {}

    panels: dict[str, dict[str, Any]] = {}

    def add_panel(
        *,
        code_value: Any,
        ref: Any,
        value_col: str,
        symbol_col: str,
        reference_kind: str,
        source_label: str,
        priority: int,
    ) -> None:
        code = _clean(code_value)
        if not code:
            return
        registry_row = registry.get(code, {})
        family = _clean(registry_row.get("family")).lower()
        if family in _LOCAL_REFERENCE_SKIP_FAMILIES:
            return
        context_codes = _local_reference_context_codes(code, registry_row)
        if not context_codes:
            return
        context_cols = [
            f"{context_code}_TPM"
            for context_code in context_codes
            if f"{context_code}_TPM" in pan.columns
        ]
        if context_cols:
            context_base = pan[context_cols].astype(float).median(axis=1)
            base_by_symbol = pd.concat(
                [pan_median.rename("pan"), context_base.rename("context")],
                axis=1,
            ).max(axis=1)
        else:
            base_by_symbol = pan_median
        ref = (
            ref.groupby(symbol_col, as_index=False)[value_col]
            .median()
            .reset_index(drop=True)
        )
        # Vectorized over the grouped reference. This replaced a per-row
        # ``iterrows`` that built ~37k pandas Series PER PANEL (×100+ panels =
        # millions of Series, each triggering ``__finalize__`` attrs-deepcopy) —
        # historically the single largest analyze-time cost. Same filters, same
        # ordering, same outputs; ``is_excluded`` (a compiled matcher) now runs
        # only on the surviving symbols rather than every row.
        syms_arr = ref[symbol_col].map(_clean).to_numpy()
        vals = pd.to_numeric(ref[value_col], errors="coerce").fillna(0.0).to_numpy()
        keep = (syms_arr != "") & (vals >= _LOCAL_REFERENCE_MIN_TPM)
        keep &= np.fromiter(
            (symbol in pan_symbols for symbol in syms_arr),
            dtype=bool,
            count=len(syms_arr),
        )
        if keep.any():
            kept_idx = np.where(keep)[0]
            excluded = np.fromiter(
                (
                    is_excluded(syms_arr[i])
                    or is_extended_housekeeping_symbol(syms_arr[i], scope="markers")
                    for i in kept_idx
                ),
                dtype=bool,
                count=len(kept_idx),
            )
            keep[kept_idx[excluded]] = False
        if not keep.any():
            return
        k_syms = syms_arr[keep]
        k_vals = vals[keep]
        base = np.fromiter(
            (_safe_float(base_by_symbol.get(s), default=0.0) for s in k_syms),
            dtype=float,
            count=len(k_syms),
        )
        log2 = np.log2((k_vals + 1.0) / (base + 1.0))
        passes = log2 >= _LOCAL_REFERENCE_MIN_LOG2_VS_PAN
        if not passes.any():
            return
        f_syms = k_syms[passes]
        f_vals = k_vals[passes]
        f_log2 = log2[passes]
        ranked = sorted(
            (
                (float(lg), float(v), -len(s), s)
                for lg, v, s in zip(f_log2, f_vals, f_syms)
            ),
            key=lambda item: (-item[0], -item[1], item[3]),
        )
        markers = tuple(
            symbol
            for _log2, _value, _length, symbol in ranked[
                :_LOCAL_REFERENCE_TOP_MARKERS
            ]
        )
        marker_set = set(markers)
        ref_by_symbol: dict[str, float] = {
            s: float(v)
            for s, v in zip(f_syms, f_vals)
            if s in marker_set
        }
        old = panels.get(code)
        if old is not None and int(old.get("_priority", 0)) > priority:
            return
        panels[code] = {
            "markers": markers,
            "ref_medians": ref_by_symbol,
            "context_codes": context_codes,
            "parent_code": _clean(registry_row.get("parent_code")),
            "family": _clean(registry_row.get("family")),
            "primary_tissue": _clean(registry_row.get("primary_tissue")),
            "expression_source": _clean(registry_row.get("expression_source")),
            "fusion_driven": _clean(registry_row.get("fusion_driven")),
            "fusion_driver": _clean(registry_row.get("fusion_driver")),
            "registry_notes": _clean(registry_row.get("notes")),
            "source_cohort": source_label,
            "reference_kind": reference_kind,
            "_priority": priority,
        }

    try:
        observed = cancer_reference_expression(
            cancer_types=tuple(sorted(eligible_codes)) if eligible_codes else None,
            normalize="tpm_clean",
            format="long",
            include_provenance=True,
        )
    except (FileNotFoundError, KeyError, ValueError):
        _LOGGER.warning(
            "cancer_reference_expression failed to load; observed-cohort "
            "panels will be skipped",
            exc_info=True,
        )
        observed = None
    if observed is not None and not observed.empty:
        if "normalization" in observed.columns:
            observed = observed[
                observed["normalization"].astype(str).str.lower().eq("tpm_clean")
            ].copy()
        for code_value, group in observed.groupby("cancer_code"):
            cohorts: list[str] = []
            if "source_cohort" in group.columns:
                cohorts = [
                    _clean(value)
                    for value in group["source_cohort"].dropna().unique()
                    if _clean(value)
                ]
            add_panel(
                code_value=code_value,
                ref=group,
                value_col="expression",
                symbol_col="Symbol",
                reference_kind="observed_bulk_reference",
                source_label=", ".join(sorted(cohorts)) or "pirlygenes",
                priority=1,
            )

    try:
        sub = subtype_deconvolved_expression(technical_rna_normalize=True)
    except (FileNotFoundError, KeyError, ValueError):
        _LOGGER.warning(
            "subtype_deconvolved_expression failed to load; deconvolved "
            "panels will be skipped",
            exc_info=True,
        )
        sub = None
    if sub is not None and not sub.empty:
        if eligible_codes and "cancer_code" in sub.columns:
            sub = sub[
                sub["cancer_code"].astype(str).str.upper().isin(eligible_codes)
            ].copy()
        if "subtype" in sub.columns:
            subtype_values = (
                sub["subtype"].fillna("").astype(str).map(_clean)
            )
            sub = sub[subtype_values == ""].copy()
        # When the dataframe lacks a ``subtype`` column at all, treat
        # every row as having empty subtype (the historical fallback
        # tried ``sub[scalar_bool]`` which silently mis-indexed).
        for code_value, group in sub.groupby("cancer_code"):
            cohorts = []
            if "source_cohort" in group.columns:
                cohorts = [
                    _clean(value)
                    for value in group["source_cohort"].dropna().unique()
                    if _clean(value)
                ]
            add_panel(
                code_value=code_value,
                ref=group,
                value_col="tumor_tpm_median",
                symbol_col="symbol",
                reference_kind="deconvolved_tumor_reference",
                source_label=", ".join(sorted(cohorts)) or "trufflepig",
                priority=2,
            )

    for panel in panels.values():
        panel.pop("_priority", None)
    return panels


@lru_cache(maxsize=None)
def _reference_medians(reference_code: str) -> dict[str, float]:
    try:
        from .reference import subtype_deconvolved_expression
    except ImportError:
        _LOGGER.warning(
            "trufflepig.reference is unavailable; %s reference medians "
            "will be empty",
            reference_code,
            exc_info=True,
        )
        return {}
    try:
        df = subtype_deconvolved_expression()
    except (FileNotFoundError, KeyError, ValueError):
        _LOGGER.warning(
            "subtype_deconvolved_expression failed to load; %s reference "
            "medians will be empty",
            reference_code,
            exc_info=True,
        )
        return {}
    if df is None or df.empty:
        return {}
    if "cancer_code" in df.columns:
        df = df[df["cancer_code"].astype(str).str.upper() == reference_code.upper()]
    if df.empty:
        return {}
    if df["symbol"].duplicated().any():
        df = (
            df.groupby("symbol", as_index=False)["tumor_tpm_median"]
            .median()
            .reset_index(drop=True)
        )
    return {
        str(row["symbol"]): _safe_float(row.get("tumor_tpm_median"))
        for _, row in df.iterrows()
        if _clean(row.get("symbol"))
    }


@lru_cache(maxsize=None)
def _cohort_bulk_gene_median(reference_code: str, gene: str) -> float | None:
    """Cohort-median BULK clean-TPM for one gene, or ``None`` if unavailable.

    Bulk (not deconvolved) on purpose: the denominator must match the *bulk* sample
    MLH1 the classifier surfaces, and the MSI-vs-MSS calibration (~0.2-0.3x silenced
    vs ~1x retained) was measured on these same bulk cohort medians.
    """
    code = _clean(reference_code)
    if not code or not gene:
        return None
    try:
        from .reference import cancer_reference_expression
    except ImportError:
        return None
    try:
        df = cancer_reference_expression(
            cancer_types=(code,),
            genes=(gene,),
            normalize="tpm_clean",
            format="long",
            include_provenance=False,
        )
    except (FileNotFoundError, KeyError, ValueError):
        return None
    if df is None or df.empty or "expression" not in df.columns:
        return None
    if "normalization" in df.columns:
        df = df[df["normalization"].astype(str).str.lower().eq("tpm_clean")]
    if "Symbol" in df.columns:
        df = df[df["Symbol"].astype(str) == gene]
    values = [
        float(v)
        for v in df["expression"]
        if isinstance(v, (int, float)) and float(v) == float(v)
    ]
    return float(values[0]) if values else None


def _enrich_mmr_vote_mlh1_cohort_context(
    vote_dict: Mapping[str, Any],
    reference_code: str,
) -> dict[str, Any]:
    """Add MLH1's cohort-relative level to an MMR vote's details.

    The classifier surfaces only the sample's raw (bulk) MLH1 clean-TPM; "retained vs
    promoter-silenced" is a cohort-relative call, so we divide by the cohort-typical
    (bulk median) MLH1 here — where the reference cohort is in scope, and in the same
    bulk space as the numerator. Silenced MLH1 collapses to a small fraction of the
    cohort median (measured ~0.2-0.3x), so the ratio cleanly separates retained (~1x)
    from silenced. Absent reference → unchanged (the report's tension clause does not
    fire).
    """
    vote_dict = dict(vote_dict)
    # The release vote's details are flat here (``mlh1_expression`` is a direct key);
    # they are re-nested under ``mismatch_repair`` when the evidence channel is built.
    details = vote_dict.get("details")
    if not isinstance(details, Mapping):
        return vote_dict
    mlh1 = details.get("mlh1_expression")
    tpm = mlh1.get("tpm") if isinstance(mlh1, Mapping) else None
    if not isinstance(tpm, (int, float)):
        return vote_dict
    median = _cohort_bulk_gene_median(reference_code, "MLH1") if reference_code else None
    if not isinstance(median, (int, float)) or median <= 0:
        return vote_dict
    mlh1 = dict(mlh1)
    mlh1["cohort_median_tpm"] = round(float(median), 3)
    mlh1["cohort_ratio"] = round(float(tpm) / float(median), 4)
    details = dict(details)
    details["mlh1_expression"] = mlh1
    vote_dict["details"] = details
    return vote_dict


def _marker_fraction_against_reference(
    sample_tpm_by_symbol: Mapping[str, float],
    genes: tuple[str, ...],
    ref_medians: Mapping[str, float],
) -> tuple[float, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    hits = 0
    considered = 0
    for gene in genes:
        ref = _safe_float(ref_medians.get(gene))
        if ref <= 0:
            continue
        considered += 1
        observed = _safe_float(sample_tpm_by_symbol.get(gene))
        ratio = observed / ref
        hit = observed >= ref
        hits += int(hit)
        details.append(
            {
                "gene": gene,
                "observed_tpm": round(observed, 3),
                "reference_median_tpm": round(ref, 3),
                "ratio_to_reference": round(ratio, 3),
                "meets_reference_median": bool(hit),
            }
        )
    return (float(hits / considered) if considered else 0.0), details


def _burden_ratio(
    sample_tpm_by_symbol: Mapping[str, float],
    genes: tuple[str, ...],
    ref_medians: Mapping[str, float],
) -> float:
    observed = sum(_safe_float(sample_tpm_by_symbol.get(gene)) for gene in genes)
    reference = sum(_safe_float(ref_medians.get(gene)) for gene in genes)
    if reference <= 0:
        return 0.0
    return float(observed / reference)


def _local_reference_marker_support(
    sample_tpm_by_symbol: Mapping[str, float],
    markers: tuple[str, ...],
    ref_medians: Mapping[str, float],
) -> tuple[float, float, list[dict[str, Any]]]:
    hits = 0
    considered = 0
    observed_sum = 0.0
    reference_sum = 0.0
    details: list[dict[str, Any]] = []
    for gene in markers:
        reference = _safe_float(ref_medians.get(gene))
        if reference <= 0:
            continue
        observed = _safe_float(sample_tpm_by_symbol.get(gene))
        threshold = max(2.0, 0.10 * reference)
        hit = observed >= threshold
        hits += int(hit)
        considered += 1
        observed_sum += observed
        reference_sum += reference
        details.append(
            {
                "gene": gene,
                "observed_tpm": round(observed, 3),
                "reference_median_tpm": round(reference, 3),
                "threshold_tpm": round(threshold, 3),
                "meets_threshold": bool(hit),
            }
        )
    marker_fraction = float(hits / considered) if considered else 0.0
    burden_ratio = float(observed_sum / reference_sum) if reference_sum > 0 else 0.0
    details.sort(
        key=lambda row: (
            not bool(row.get("meets_threshold")),
            -_safe_float(row.get("observed_tpm")),
            str(row.get("gene") or ""),
        )
    )
    return marker_fraction, burden_ratio, details


def _threshold_blocking_reasons(
    metrics: Mapping[str, float],
    minimums: Mapping[str, float],
    labels: Mapping[str, str],
) -> list[str]:
    reasons: list[str] = []
    for metric, minimum in minimums.items():
        observed = _safe_float(metrics.get(metric))
        if observed < minimum:
            label = labels.get(metric, metric.replace("_", " "))
            reasons.append(f"{label} {observed:.2f} is below {minimum:.2f}")
    return reasons


def add_pan_cancer_signature_ranker_evidence(
    hypotheses: dict[str, CancerTypeEvidence],
    analysis: Mapping[str, Any],
) -> None:
    candidate_rows = _candidate_rows(analysis)
    for rank, row in enumerate(candidate_rows, start=1):
        code = _clean(row.get("code"))
        if not code:
            continue
        support = _safe_float(row.get("support_fraction_of_top"))
        if support <= 0 and rank == 1:
            support = 1.0
        hypothesis = _hypothesis(hypotheses, code)
        hypothesis.add_source("pan_cancer_signature_ranker")
        hypothesis.expression_reference_cancer_type = code
        hypothesis.reference_cancer_type = code
        hypothesis.broad_rna_support = max(hypothesis.broad_rna_support, support)
        hypothesis.broad_rna_rank = rank
        hypothesis.details.update(dict(row))
        hypothesis.details.pop("code", None)
        hypothesis.details["support_fraction_of_top"] = round(support, 4)
        hypothesis.details["pan_cancer_signature_support"] = round(support, 4)
        hypothesis.details["pan_cancer_signature_rank"] = rank
        if not _code_has_registry_ancestor(code, _CRC_REGISTRY_ROOT):
            crc_competitor = _best_crc_family_trace_candidate(analysis)
            if crc_competitor:
                hypothesis.details["pan_cancer_signature_crc_family_competitor"] = (
                    crc_competitor
                )
        if rank == 1:
            hypothesis.basis = hypothesis.basis or (
                "top pan-cancer signature-ranker match"
            )
            hypothesis.consider_for_report_label(
                selected_by="pan_cancer_signature_ranker",
                can_select=False,
                blocking_reasons=(
                    "pan-cancer signature-ranker evidence is candidate/context "
                    "only; fused evidence must supply an independent admission "
                    "path before it can select the report label",
                ),
                priority=(0, support),
            )
        winning_subtype = _clean(row.get("winning_subtype"))
        if winning_subtype and winning_subtype != code:
            subtype_hypothesis = _hypothesis(hypotheses, winning_subtype)
            subtype_hypothesis.add_source("pan_cancer_signature_subtype")
            subtype_hypothesis.expression_reference_cancer_type = winning_subtype
            subtype_hypothesis.reference_cancer_type = code
            subtype_hypothesis.related_context_code = code
            subtype_hypothesis.related_context_support = max(
                subtype_hypothesis.related_context_support,
                support,
            )
            subtype_hypothesis.broad_rna_support = max(
                subtype_hypothesis.broad_rna_support,
                support,
            )
            subtype_hypothesis.broad_rna_rank = rank
            subtype_hypothesis.basis = subtype_hypothesis.basis or (
                f"{code} pan-cancer signature-ranker candidate resolves to {winning_subtype}"
            )
            subtype_hypothesis.details.update(
                {
                    "parent_broad_rna_code": code,
                    "parent_broad_rna_rank": rank,
                    "parent_pan_cancer_signature_code": code,
                    "parent_pan_cancer_signature_rank": rank,
                    "parent_support_fraction_of_top": round(float(support), 4),
                    "winning_subtype_from_candidate_trace": winning_subtype,
                }
            )
            if rank == 1:
                blockers: list[str] = []
                background_compartment = _background_top_compartment_conflict(row)
                close_candidate = (
                    _close_trace_candidate_for_lineage(
                        candidate_rows[rank:],
                        background_compartment,
                        min_support=0.80,
                    )
                    if background_compartment
                    else {}
                )
                if close_candidate:
                    subtype_hypothesis.details[
                        "broad_subtype_background_compartment_conflict"
                    ] = {
                        "parent_code": code,
                        "winning_subtype": winning_subtype,
                        "centroid_compartment_lineage": background_compartment,
                        "close_compartment_candidate": _clean(
                            close_candidate.get("code")
                        ),
                        "close_compartment_candidate_support": round(
                            _safe_float(
                                close_candidate.get("support_fraction_of_top")
                            ),
                            4,
                        ),
                    }
                    blockers.append(
                        f"{code} winning subtype {winning_subtype} is treated "
                        "as background/context because the signature parent is "
                        "background-like while the whole-profile compartment "
                        f"and a close broad candidate support {background_compartment} "
                        f"({subtype_hypothesis.details['broad_subtype_background_compartment_conflict']['close_compartment_candidate']})"
                    )
                if blockers:
                    subtype_hypothesis.report_label_candidate = True
                    subtype_hypothesis.label_status = "blocked"
                    subtype_hypothesis.label_basis = "pan_cancer_signature_subtype"
                    subtype_hypothesis.blocking_reasons = tuple(blockers)


def _add_pan_cancer_signature_marker_features(
    hypotheses: dict[str, CancerTypeEvidence],
    sample_tpm_by_symbol: Mapping[str, float],
) -> None:
    if not sample_tpm_by_symbol:
        return
    for hypothesis in hypotheses.values():
        if (
            "pan_cancer_signature_ranker" not in hypothesis.evidence_sources
            or hypothesis.broad_rna_support <= 0
        ):
            continue
        coherence = _marker_coherence(hypothesis.cancer_type, sample_tpm_by_symbol)
        if not coherence:
            continue
        hypothesis.details["pan_cancer_signature_marker_coherence"] = coherence
        if _marker_coherence_positive_complete(coherence):
            fraction = _safe_float(coherence.get("detected_fraction"))
            support = float(
                np.clip(0.5 * hypothesis.broad_rna_support + 0.5 * fraction, 0.0, 1.0)
            )
            selectable = _pan_signature_marker_program_selectable(coherence)
            if selectable:
                hypothesis.family_marker_support = max(
                    hypothesis.family_marker_support,
                    support,
                )
            hypothesis.details["pan_cancer_signature_marker_support"] = round(
                support,
                4,
            )
            hypothesis.details["pan_cancer_signature_marker_selectable"] = bool(
                selectable
            )
            hypothesis.details["pan_cancer_signature_marker_status"] = (
                "positive_complete_mixed"
                if _marker_coherence_unexpected_low_count(coherence)
                else "positive_complete"
            )


def _context_support_for_code_or_parent(
    code: str,
    support_by_code: Mapping[str, float],
) -> tuple[float, str]:
    code = _clean(code)
    for candidate in (code, *_registry_parent_chain(code)):
        support = _safe_float(support_by_code.get(candidate))
        if support > 0:
            return support, candidate
    return 0.0, ""


def _learned_hierarchical_votes(
    sample_tpm_by_symbol: Mapping[str, float],
) -> list[dict[str, Any]]:
    if not sample_tpm_by_symbol:
        return []
    try:
        from .expression_classifier import classify_expression_hierarchy
    except ImportError:
        return []
    return [
        vote.public_dict()
        for vote in classify_expression_hierarchy(sample_tpm_by_symbol)
    ]


def _learned_vote_supports_by_stage(
    votes: list[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for vote in votes:
        stage = _clean(vote.get("stage"))
        if not stage:
            continue
        supports = out.setdefault(stage, {})
        for item in vote.get("top_predictions") or []:
            label = _clean(item.get("label") or item.get("code"))
            if label:
                supports[label] = max(supports.get(label, 0.0), _safe_float(item.get("probability")))
    return out


def _learned_entity_context_support(
    code: str,
    supports_by_stage: Mapping[str, Mapping[str, float]],
) -> tuple[float, str]:
    return _context_support_for_code_or_parent(
        code,
        supports_by_stage.get("entity", {}),
    )


def _learned_compartment_context(
    code: str,
    supports_by_stage: Mapping[str, Mapping[str, float]],
) -> tuple[float, str]:
    learned_lineage = _code_lineage_token(code)
    compartment_supports = supports_by_stage.get("compartment", {})
    best_label = ""
    best_support = 0.0
    for label, support in compartment_supports.items():
        if _lineage_token(label) == learned_lineage and support > best_support:
            best_label = label
            best_support = _safe_float(support)
    return best_support, best_label


def _learned_family_context_support(
    code: str,
    supports_by_stage: Mapping[str, Mapping[str, float]],
) -> tuple[float, str]:
    try:
        from .expression_classifier import _learned_family_for_code
    except ImportError:
        return 0.0, ""
    label = _clean(_learned_family_for_code(code))
    if not label:
        return 0.0, ""
    return _safe_float(supports_by_stage.get("family", {}).get(label)), label


def _top_learned_stage_support(
    supports_by_stage: Mapping[str, Mapping[str, float]],
    stage: str,
) -> tuple[str, float]:
    supports = supports_by_stage.get(stage, {})
    best_label = ""
    best_support = 0.0
    for label, support in supports.items():
        value = _safe_float(support)
        if value > best_support:
            best_label = _clean(label)
            best_support = value
    return best_label, best_support


def _top_learned_stage_margin(
    votes: list[Mapping[str, Any]],
    stage: str,
) -> float:
    for vote in votes:
        if _clean(vote.get("stage")) == stage:
            return _safe_float(vote.get("margin"))
    return 0.0


_GLOBAL_LEARNED_AUDIT_FIELDS = (
    "learned_expression_flat_top_predictions",
    "learned_expression_flat_entity_supports",
    "learned_expression_flat_feature_space",
    "learned_expression_top_entity_label",
    "learned_expression_top_entity_support",
    "learned_expression_top_entity_margin",
    "learned_expression_top_family_label",
    "learned_expression_top_family_support",
    "learned_expression_top_family_margin",
    "learned_expression_top_compartment_label",
    "learned_expression_top_compartment_support",
)


def _public_learned_vote_dict(vote: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": _clean(vote.get("stage")),
        "label_space": _clean(vote.get("label_space")),
        "label": _clean(vote.get("label")),
        "probability": round(_safe_float(vote.get("probability")), 4),
        "margin": round(_safe_float(vote.get("margin")), 4),
        "top_predictions": vote.get("top_predictions") or [],
        "training_split_policy": vote.get("training_split_policy"),
        "holdout_top1_accuracy": vote.get("holdout_top1_accuracy"),
        "holdout_medoid_top1_accuracy": vote.get(
            "holdout_medoid_top1_accuracy"
        ),
        "oof_precision_at_threshold": vote.get("oof_precision_at_threshold"),
        "oof_top3_recovery": vote.get("oof_top3_recovery"),
        "training_sample_count": vote.get("training_sample_count"),
        "training_cohorts": vote.get("training_cohorts") or [],
        "details": vote.get("details") or {},
    }


def _candidate_mismatch_repair_votes(
    sample_tpm_by_symbol: Mapping[str, float],
    code: str,
    flat_predictions: list[tuple[str, float]],
) -> list[dict[str, Any]]:
    """Build MMR votes in the destination cancer entity's context."""

    try:
        from .expression_classifier import (
            classify_mismatch_repair_expression,
            mismatch_repair_context_group,
            mismatch_repair_sibling_vote_from_predictions,
        )
    except ImportError:
        return []
    if not mismatch_repair_context_group(code):
        return []

    votes: list[dict[str, Any]] = []
    release_vote = classify_mismatch_repair_expression(
        sample_tpm_by_symbol,
        code,
    )
    if release_vote is not None:
        votes.append(
            _enrich_mmr_vote_mlh1_cohort_context(
                _public_learned_vote_dict(release_vote.public_dict()),
                code,
            )
        )
    if flat_predictions:
        sibling_vote = mismatch_repair_sibling_vote_from_predictions(
            tuple(flat_predictions),
            code,
        )
        if sibling_vote is not None:
            votes.append(
                _public_learned_vote_dict(sibling_vote.public_dict())
            )
    return votes


def _merge_global_learned_audit_details(
    candidate: CancerTypeEvidence,
    hierarchy_details: Mapping[str, Any],
) -> None:
    """Merge sample-global hierarchy stages without transplanting MMR state."""

    source_votes = (
        hierarchy_details.get("learned_expression_hierarchy_votes")
        or hierarchy_details.get("learned_expression_hierarchical_votes")
        or []
    )
    candidate_votes = (
        candidate.details.get("learned_expression_hierarchy_votes") or []
    )
    global_votes = [
        dict(vote)
        for vote in source_votes
        if isinstance(vote, Mapping)
        and _clean(vote.get("stage")) != "mismatch_repair"
    ]
    candidate_mmr_votes = [
        dict(vote)
        for vote in candidate_votes
        if isinstance(vote, Mapping)
        and _clean(vote.get("stage")) == "mismatch_repair"
    ]
    for key in _GLOBAL_LEARNED_AUDIT_FIELDS:
        if key in hierarchy_details:
            candidate.details[key] = hierarchy_details[key]
    if global_votes or candidate_mmr_votes:
        candidate.details["learned_expression_hierarchy_votes"] = [
            *global_votes,
            *candidate_mmr_votes,
        ]


def _refresh_candidate_mismatch_repair_votes(
    candidate: CancerTypeEvidence,
    sample_tpm_by_symbol: Mapping[str, float] | None,
) -> None:
    """Recompute final-row MMR votes after an entity-changing adjudication."""

    if not sample_tpm_by_symbol:
        return
    try:
        from .expression_classifier import (
            classify_expression,
            mismatch_repair_context_group,
        )
    except ImportError:
        return
    candidate_votes = (
        candidate.details.get("learned_expression_hierarchy_votes") or []
    )
    global_votes = [
        dict(vote)
        for vote in candidate_votes
        if isinstance(vote, Mapping)
        and _clean(vote.get("stage")) != "mismatch_repair"
    ]
    if not mismatch_repair_context_group(candidate.cancer_type):
        candidate.details["learned_expression_hierarchy_votes"] = global_votes
        return
    flat_predictions = classify_expression(sample_tpm_by_symbol, top_k=1000)
    candidate.details["learned_expression_hierarchy_votes"] = [
        *global_votes,
        *_candidate_mismatch_repair_votes(
            sample_tpm_by_symbol,
            candidate.cancer_type,
            flat_predictions,
        ),
    ]


def _add_learned_hierarchy_candidate_features(
    hypotheses: dict[str, CancerTypeEvidence],
    sample_tpm_by_symbol: Mapping[str, float],
) -> None:
    """Attach learned compartment/family/entity votes to every candidate row."""

    if not hypotheses or not sample_tpm_by_symbol:
        return
    hierarchical_votes = _learned_hierarchical_votes(sample_tpm_by_symbol)
    if not hierarchical_votes:
        return
    supports_by_stage = _learned_vote_supports_by_stage(hierarchical_votes)
    top_compartment_label, top_compartment_support = _top_learned_stage_support(
        supports_by_stage,
        "compartment",
    )
    top_family_label, top_family_support = _top_learned_stage_support(
        supports_by_stage,
        "family",
    )
    top_family_margin = _top_learned_stage_margin(hierarchical_votes, "family")
    top_entity_label, top_entity_support = _top_learned_stage_support(
        supports_by_stage,
        "entity",
    )
    top_entity_margin = _top_learned_stage_margin(hierarchical_votes, "entity")
    flat_predictions: list[tuple[str, float]] = []
    try:
        from .expression_classifier import classify_expression
    except ImportError:
        flat_predictions = []
    else:
        flat_predictions = classify_expression(sample_tpm_by_symbol, top_k=1000)
    flat_top_predictions = [
        {"code": _clean(code), "probability": round(_safe_float(prob), 4)}
        for code, prob in flat_predictions[:10]
    ]
    flat_entity_supports = _aggregate_learned_predictions_by_entity(
        [
            {"code": _clean(code), "probability": _safe_float(prob)}
            for code, prob in flat_predictions
        ]
    )
    public_votes = [
        _public_learned_vote_dict(vote)
        for vote in hierarchical_votes
        if _clean(vote.get("stage")) != "mismatch_repair"
    ]

    for hypothesis in hypotheses.values():
        code = hypothesis.cancer_type
        entity_support, entity_label = _learned_entity_context_support(
            code,
            supports_by_stage,
        )
        compartment_support, compartment_label = _learned_compartment_context(
            code,
            supports_by_stage,
        )
        family_support, family_label = _learned_family_context_support(
            code,
            supports_by_stage,
        )
        learned_lineage = _code_lineage_token(code)
        flat_neighborhood_support = sum(
            _safe_float(prob)
            for pred_code, prob in flat_predictions
            if _code_lineage_token(_clean(pred_code)) == learned_lineage
        )
        hierarchy_support = max(entity_support, family_support, compartment_support)
        if hierarchy_support <= 0:
            continue
        hypothesis_public_votes = [
            *public_votes,
            *_candidate_mismatch_repair_votes(
                sample_tpm_by_symbol,
                code,
                flat_predictions,
            ),
        ]
        hypothesis.details.update(
            {
                "learned_expression_hierarchy_votes": hypothesis_public_votes,
                "learned_expression_flat_top_predictions": flat_top_predictions,
                # Decision support is aggregated from the complete classifier
                # vector before the top-ten, rounded audit view is produced.
                # Several subtype/status leaves can represent one report entity;
                # dropping tail leaves can otherwise change the entity leader.
                "learned_expression_flat_entity_supports": dict(
                    sorted(
                        flat_entity_supports.items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ),
                "learned_expression_flat_feature_space": (
                    "within_sample_percentile"
                ),
                "learned_expression_flat_lineage_support": round(
                    float(flat_neighborhood_support),
                    4,
                ),
                "learned_expression_entity_support": round(
                    float(entity_support),
                    4,
                ),
                "learned_expression_entity_label": entity_label,
                "learned_expression_top_entity_label": top_entity_label,
                "learned_expression_top_entity_support": round(
                    float(top_entity_support),
                    4,
                ),
                "learned_expression_top_entity_margin": round(
                    float(top_entity_margin),
                    4,
                ),
                "learned_expression_family_support": round(
                    float(family_support),
                    4,
                ),
                "learned_expression_family_label": family_label,
                "learned_expression_top_family_label": top_family_label,
                "learned_expression_top_family_support": round(
                    float(top_family_support),
                    4,
                ),
                "learned_expression_top_family_margin": round(
                    float(top_family_margin),
                    4,
                ),
                "learned_expression_compartment_support": round(
                    float(compartment_support),
                    4,
                ),
                "learned_expression_compartment_label": compartment_label,
                "learned_expression_top_compartment_label": top_compartment_label,
                "learned_expression_top_compartment_support": round(
                    float(top_compartment_support),
                    4,
                ),
                "learned_expression_hierarchy_support": round(
                    float(hierarchy_support),
                    4,
                ),
            }
        )


@lru_cache(maxsize=1)
def _learned_hierarchy_entity_contexts() -> frozenset[tuple[str, str, str]]:
    """Return valid ``(entity, family, compartment)`` paths from the registry.

    Hierarchy stages are trained separately, so their top labels must be
    checked for semantic coherence before they can arbitrate a final call.  A
    broad entity such as ``SARC`` can legitimately pair with a child family
    such as ``SARC_OTHER``; deriving paths from every registry leaf handles
    that without diagnosis-specific exceptions.
    """

    try:
        from .expression_classifier import (
            _learned_compartment_for_code,
            _learned_entity_for_code,
            _learned_family_for_code,
        )
    except ImportError:
        return frozenset()
    contexts: set[tuple[str, str, str]] = set()
    for code in _registry_by_code():
        entity = _clean(_learned_entity_for_code(code))
        family = _clean(_learned_family_for_code(code))
        compartment = _clean(_learned_compartment_for_code(code))
        if entity:
            contexts.add((entity, family, compartment))
    return frozenset(contexts)


def _learned_hierarchy_path_consistent(
    entity_code: str,
    *,
    family_label: str = "",
    compartment_label: str = "",
) -> bool:
    entity = _clean(entity_code)
    family = _clean(family_label)
    compartment = _clean(compartment_label)
    if not entity:
        return False
    return any(
        path_entity == entity
        and (not family or path_family == family)
        and (not compartment or path_compartment == compartment)
        for path_entity, path_family, path_compartment
        in _learned_hierarchy_entity_contexts()
    )


def _learned_hierarchy_details(
    hypotheses: Mapping[str, CancerTypeEvidence],
    selected: CancerTypeEvidence | None,
) -> Mapping[str, Any]:
    rows = ([selected] if selected is not None else []) + list(hypotheses.values())
    for row in rows:
        if row is None:
            continue
        if _clean(row.details.get("learned_expression_top_entity_label")):
            return row.details
    return {}


def _learned_entity_support_for_code(
    details: Mapping[str, Any],
    code: str,
) -> float:
    """Return the strongest learned full-profile support assigned to ``code``.

    The calibrated hierarchy and quantifier-robust flat classifier are two
    transformations of the same expression profile, so their maximum support
    is one learned evidence axis rather than two independent votes.
    """

    votes = (
        details.get("learned_expression_hierarchy_votes")
        or details.get("learned_expression_hierarchical_votes")
        or []
    )
    supports: dict[str, float] = {}
    for vote in votes:
        if not isinstance(vote, Mapping) or _clean(vote.get("stage")) != "entity":
            continue
        for label, value in _aggregate_learned_predictions_by_entity(
            vote.get("top_predictions") or []
        ).items():
            # Multiple hierarchy votes are related views of the same profile.
            # Sum mutually exclusive leaves within a view, then retain the
            # strongest view instead of counting related views twice.
            supports[label] = max(supports.get(label, 0.0), value)
    if not supports:
        top_label = _learned_prediction_entity_code(
            details.get("learned_expression_top_entity_label")
        )
        if top_label:
            supports[top_label] = _safe_float(
                details.get("learned_expression_top_entity_support")
            )
    flat_supports = _flat_learned_entity_supports(details)
    for label, value in flat_supports.items():
        supports[label] = max(supports.get(label, 0.0), value)

    support, _label = _context_support_for_code_or_parent(code, supports)
    descendant_support = max(
        (
            _safe_float(value)
            for label, value in supports.items()
            if _code_has_registry_ancestor(label, code)
        ),
        default=0.0,
    )
    return float(max(support, descendant_support))


def _aggregate_learned_predictions_by_entity(
    predictions: Any,
) -> dict[str, float]:
    """Sum one softmax view after rolling its leaves to report entities.

    Predictions within a classifier view are mutually exclusive outcomes. If
    several leaves represent the same report entity (for example ``COAD_MSI``
    and ``COAD_MSS``), their probability mass belongs to that entity before a
    leader is selected. Related hierarchy and flat views remain separate and
    are combined by maximum elsewhere.
    """

    supports: dict[str, float] = {}
    for item in predictions or []:
        if not isinstance(item, Mapping):
            continue
        raw_code = _clean(item.get("code") or item.get("label"))
        entity_code = _learned_prediction_entity_code(raw_code)
        if not entity_code:
            continue
        supports[entity_code] = supports.get(entity_code, 0.0) + _safe_float(
            item.get("probability")
        )
    return supports


def _flat_learned_entity_supports(
    details: Mapping[str, Any],
) -> dict[str, float]:
    """Collapse flat leaf predictions to entity support without double counting."""

    full_supports = details.get("learned_expression_flat_entity_supports")
    if isinstance(full_supports, Mapping):
        supports = {
            _clean(code): _safe_float(value)
            for code, value in full_supports.items()
            if _clean(code) and _safe_float(value) > 0
        }
        if supports:
            return supports
    return _aggregate_learned_predictions_by_entity(
        details.get("learned_expression_flat_top_predictions") or []
    )


def _learned_entity_prediction_codes(
    details: Mapping[str, Any],
) -> list[tuple[str, float]]:
    """Return the entity leader from each related learned view.

    The hierarchy and flat classifiers are alternative transformations of the
    same expression profile, not an open-ended candidate generator.  A tail
    prediction is useful audit context, but treating every nonzero tail as an
    affirmative learned vote lets unrelated reference axes promote an entity
    that neither learned view actually selected.  The adjudication beam is
    therefore the union of the hierarchy leader and the flat-view leader.
    """

    votes = (
        details.get("learned_expression_hierarchy_votes")
        or details.get("learned_expression_hierarchical_votes")
        or []
    )
    order: list[str] = []
    support_by_code: dict[str, float] = {}

    def add_leader(code: str, support: float) -> None:
        code = _clean(code)
        if not code:
            return
        if code not in support_by_code:
            order.append(code)
        support_by_code[code] = max(
            support_by_code.get(code, 0.0),
            _safe_float(support),
        )

    for vote in votes:
        if not isinstance(vote, Mapping) or _clean(vote.get("stage")) != "entity":
            continue
        predictions = [
            item
            for item in vote.get("top_predictions") or []
            if isinstance(item, Mapping)
            and _clean(item.get("label") or item.get("code"))
        ]
        entity_supports = _aggregate_learned_predictions_by_entity(predictions)
        if entity_supports:
            entity_code, entity_support = max(
                entity_supports.items(),
                key=lambda item: item[1],
            )
            add_leader(entity_code, entity_support)
        break
    if not order:
        add_leader(
            _learned_prediction_entity_code(
                details.get("learned_expression_top_entity_label")
            ),
            details.get("learned_expression_top_entity_support"),
        )

    flat_supports = _flat_learned_entity_supports(details)
    if flat_supports:
        flat_code, flat_support = max(
            flat_supports.items(),
            key=lambda item: item[1],
        )
        add_leader(flat_code, flat_support)
    return [(code, support_by_code[code]) for code in order]


def _learned_prediction_entity_code(code: str) -> str:
    """Normalize a learned leaf to the entity adjudicated by consensus.

    Molecular/status and expression-subtype predictions are evidence for their
    registry parent at the entity stage. The original leaf remains in the audit
    fields and can be applied later on its own subtype axis.
    """

    entity_code = _clean(code)
    axes = _orthogonal_axes_for_code(entity_code)
    if axes:
        entity_code = _clean(axes[0].get("base_code")) or entity_code
    while entity_code and _orthogonal_axes_that_block_report_label(entity_code):
        parents = _registry_parent_chain(entity_code)
        entity_code = parents[0] if parents else ""
    return entity_code


def _entity_consensus_candidate_beam(
    hypotheses: dict[str, CancerTypeEvidence],
    selected: CancerTypeEvidence,
    hierarchy_details: Mapping[str, Any],
    *,
    sample_tpm_by_symbol: Mapping[str, float] | None = None,
    cen=None,
    centroid_confident: bool = False,
    residual_identity_evidence: Mapping[str, Any] | None = None,
) -> CancerTypeEvidence | None:
    """Adjudicate each learned-view or invariant-residual leader.

    The hierarchy and quantifier-robust flat views may have different leaders.
    An invariant post-background identity may add one parent-level entity to
    that small union, but is still only one consensus axis. Lower-ranked
    learned tails remain audit-only. Every candidate retains the same
    available-axis majority and hard-blocker requirements.
    """

    predictions = _learned_entity_prediction_codes(hierarchy_details)
    residual_code = _qualified_residual_identity_candidate(
        residual_identity_evidence
    )
    residual_prediction_appended = bool(
        residual_code
        and not any(
            _learned_prediction_entity_code(code) == residual_code
            for code, _support in predictions
        )
    )
    if residual_prediction_appended:
        predictions.append(
            (
                residual_code,
                _learned_entity_support_for_code(
                    hierarchy_details,
                    residual_code,
                ),
            )
        )
    if not predictions:
        return None
    hierarchy_votes = (
        hierarchy_details.get("learned_expression_hierarchy_votes")
        or hierarchy_details.get("learned_expression_hierarchical_votes")
        or []
    )
    supports_by_stage = _learned_vote_supports_by_stage(hierarchy_votes)
    selected_compartment_support, _ = _learned_compartment_context(
        selected.cancer_type,
        supports_by_stage,
    )
    selected_lineage = _code_lineage_token(selected.cancer_type)
    decisive: list[
        tuple[tuple[float, float, float, float], CancerTypeEvidence, dict[str, Any]]
    ] = []
    evaluated_entities: set[str] = set()
    for prediction_rank, (raw_code, predicted_support) in enumerate(
        predictions,
        start=1,
    ):
        entity_code = _learned_prediction_entity_code(raw_code)
        if (
            not entity_code
            or entity_code == selected.cancer_type
            or entity_code not in _registry_by_code()
        ):
            continue
        if entity_code in evaluated_entities:
            continue
        evaluated_entities.add(entity_code)

        candidate = _hypothesis(hypotheses, entity_code)
        candidate_lineage = _code_lineage_token(entity_code)
        # A parent-level residual result may corroborate a learned child as one
        # branch-level axis, but it did not originate that child hypothesis.
        # Only the exact parent row appended above receives the residual-origin
        # audit label and the corresponding cross-lineage admission path.
        residual_origin = bool(
            residual_prediction_appended
            and raw_code == residual_code
            and entity_code == residual_code
        )
        if (
            not residual_origin
            and _fused_parent_abstention_blocks_entity_consensus(candidate)
        ):
            candidate.details["entity_consensus_preserved_fused_abstention"] = (
                dict(
                    candidate.details.get(
                        "fused_evidence_structured_parent_abstention"
                    )
                    or {}
                )
            )
            continue
        if (
            not residual_origin
            and candidate_lineage
            and selected_lineage
            and candidate_lineage != selected_lineage
        ):
            candidate_compartment_support, _ = _learned_compartment_context(
                entity_code,
                supports_by_stage,
            )
            if candidate_compartment_support <= selected_compartment_support:
                candidate_composition = candidate.adjudication_axis_support(
                    _ENTITY_CONSENSUS_COMPOSITION_AXIS
                )
                selected_composition = selected.adjudication_axis_support(
                    _ENTITY_CONSENSUS_COMPOSITION_AXIS
                )
                if candidate_composition <= selected_composition:
                    continue

        candidate.add_source(
            "decomposition_residual_identity"
            if residual_origin
            else "learned_expression_classifier"
        )
        entity_support = max(
            predicted_support,
            _learned_entity_support_for_code(hierarchy_details, entity_code),
        )
        candidate.learned_expression_support = max(
            candidate.learned_expression_support,
            entity_support,
        )
        candidate.expression_reference_cancer_type = (
            candidate.expression_reference_cancer_type or entity_code
        )
        _merge_global_learned_audit_details(
            candidate,
            hierarchy_details,
        )
        if sample_tpm_by_symbol:
            candidate.details["learned_hierarchy_entity_marker_coherence"] = dict(
                _marker_coherence(entity_code, sample_tpm_by_symbol)
            )
        _attach_entity_descendant_reference_support(hypotheses, candidate)
        _attach_entity_descendant_reference_support(hypotheses, selected)
        _attach_entity_descendant_ranker_support(hypotheses, candidate)
        _attach_entity_descendant_ranker_support(hypotheses, selected)
        consensus = _entity_evidence_consensus(
            candidate,
            selected,
            hierarchy_details,
            sample_tpm_by_symbol=sample_tpm_by_symbol,
            cen=cen,
            centroid_confident=centroid_confident,
            residual_identity_evidence=residual_identity_evidence,
        )
        consensus["learned_entity_prediction_rank"] = prediction_rank
        consensus["learned_entity_prediction_support"] = round(
            float(entity_support),
            4,
        )
        consensus["learned_entity_prediction_raw_code"] = raw_code
        consensus["learned_entity_prediction_entity_code"] = entity_code
        consensus["entity_prediction_origin"] = (
            "invariant_residual_identity"
            if residual_origin
            else "learned_expression_view"
        )
        credible_learned_candidate = bool(
            entity_support >= _LEARNED_EXPRESSION_MIN_PROBABILITY
        )
        origin_features = _entity_consensus_origin_features(
            candidate,
            selected,
            consensus,
            credible_learned_candidate=credible_learned_candidate,
            residual_origin=residual_origin,
        )
        consensus.update(origin_features)
        candidate_origin_credible = origin_features[
            "candidate_origin_credible"
        ]
        hard_blockers = _persistent_report_label_blockers(candidate)
        if hard_blockers:
            consensus["evidence_decisive_candidate"] = bool(
                consensus.get("decisive_candidate")
            )
            consensus["decisive_candidate"] = False
            consensus["selection_blocked"] = True
            consensus["candidate_hard_blockers"] = list(hard_blockers)
        if (
            not consensus.get("decisive_candidate")
            or not candidate_origin_credible
        ):
            continue
        nonlearned_support = sum(
            _safe_float(axis.get("candidate_support"))
            for axis in consensus.get("axes") or []
            if axis.get("axis") != "learned_full_profile"
            and axis.get("preference") == "candidate"
        )
        score = (
            float(
                _safe_int(consensus.get("candidate_votes"))
                - _safe_int(consensus.get("selected_votes"))
            ),
            float(nonlearned_support),
            _safe_float(consensus.get("candidate_advantage")),
            float(entity_support),
        )
        decisive.append((score, candidate, consensus))

    if not decisive:
        return None
    _score, candidate, consensus = max(decisive, key=lambda item: item[0])
    _refresh_candidate_mismatch_repair_votes(
        candidate,
        sample_tpm_by_symbol,
    )
    selected.details["entity_evidence_consensus"] = dict(consensus)
    candidate.details.update(
        {
            "entity_evidence_consensus": dict(consensus),
            "entity_consensus_adjudicated": True,
            "entity_consensus_adjudication_mode": "candidate_beam_consensus",
            "entity_consensus_previous_code": selected.cancer_type,
        }
    )
    candidate.basis = (
        f"the entity consensus beam and a majority of independent evidence "
        f"groups support {candidate.cancer_type} over {selected.cancer_type}"
    )
    candidate.consider_for_report_label(
        selected_by="entity_evidence_consensus",
        can_select=True,
        blocking_reasons=(),
        priority=(
            4,
            1.0 + abs(_safe_float(consensus.get("candidate_advantage"))),
        ),
    )
    if candidate.can_select_report_label:
        _withdraw_fused_selection_superseded_by_entity_adjudication(
            selected,
            candidate.cancer_type,
        )
    return candidate if candidate.can_select_report_label else None


def _attach_entity_descendant_reference_support(
    hypotheses: Mapping[str, CancerTypeEvidence],
    entity: CancerTypeEvidence,
) -> None:
    """Roll valid child expression-reference support up to its entity.

    Expression-subtype rows cannot select a diagnosis by themselves, but a
    BRCA_Basal-like reference is still direct evidence for the BRCA entity.
    Preserve the child identity for audit.  The selector-owned admissibility
    ledger is authoritative: any reference rejected for context, marker
    burden, molecular, lineage, or other reasons remains visible but cannot
    become an entity-consensus vote.
    """

    best_support = 0.0
    best_code = ""
    for child in hypotheses.values():
        child_support = _safe_float(
            child.adjudication_axis_support(
                _ENTITY_CONSENSUS_REFERENCE_AXIS
            )
        )
        if (
            child is entity
            or child_support <= 0
            or not _code_has_registry_ancestor(
                child.cancer_type,
                entity.cancer_type,
            )
        ):
            continue
        if child_support > best_support:
            best_support = child_support
            best_code = child.cancer_type
    if best_support <= 0:
        return
    entity.details["entity_descendant_exact_reference_support"] = round(
        best_support,
        4,
    )
    entity.details["entity_descendant_exact_reference_code"] = best_code


def _attach_entity_descendant_ranker_support(
    hypotheses: Mapping[str, CancerTypeEvidence],
    entity: CancerTypeEvidence,
) -> None:
    """Roll observed child ranker evidence up to an aggregate report entity.

    Grouping nodes such as CRC have loadable COAD/READ children rather than a
    separate first-pass row.  Entity adjudication must compare the parent's
    best observed child with the selected leaf, not treat the parent as having
    zero signature evidence merely because its references live below it.
    """

    descendants = [
        child
        for child in hypotheses.values()
        if child is not entity
        and "pan_cancer_signature_ranker" in child.evidence_sources
        and _code_has_registry_ancestor(
            child.cancer_type,
            entity.cancer_type,
        )
    ]
    if not descendants:
        return
    best = max(
        descendants,
        key=lambda row: (
            _raw_signature_support(row),
            row.broad_rna_support,
            -row.broad_rna_rank,
        ),
    )
    best_signature = _raw_signature_support(best)
    entity.broad_rna_support = max(
        entity.broad_rna_support,
        max(row.broad_rna_support for row in descendants),
    )
    positive_ranks = [row.broad_rna_rank for row in descendants if row.broad_rna_rank]
    if positive_ranks:
        rank_candidates = list(positive_ranks)
        if entity.broad_rna_rank:
            rank_candidates.append(entity.broad_rna_rank)
        entity.broad_rna_rank = min(rank_candidates)
    if "signature_score" not in entity.details and best_signature > 0:
        entity.details["signature_score"] = best_signature
    entity.details["entity_descendant_pan_cancer_signature_code"] = (
        best.cancer_type
    )
    entity.details["entity_descendant_pan_cancer_signature_score"] = round(
        best_signature,
        4,
    )


def _entity_marker_program_support(
    hypothesis: CancerTypeEvidence,
    sample_tpm_by_symbol: Mapping[str, float] | None,
) -> tuple[float, Mapping[str, Any]]:
    """Collapse curated positive/negative marker evidence into one axis.

    Contrast, rare-marker, lineage-panel, and raw ontology-coherence signals
    are all curated marker programs and therefore are one evidence *group*,
    not four independent votes. ``family_marker_support`` and
    ``pan_cancer_signature_marker_support`` are intentionally excluded here:
    the signature ranker can contribute to both, so counting either beside the
    separate pan-cancer-signature axis would duplicate the same RNA evidence.
    When no selector attached an independent score, the ontology sanity panel
    contributes its detected fraction, discounted by explicit expected-low
    violations.
    """

    details = hypothesis.details
    coherence = (
        details.get("learned_hierarchy_entity_marker_coherence")
        or details.get("entity_evidence_marker_coherence")
        or details.get("learned_expression_marker_coherence")
        or details.get("pan_cancer_signature_marker_coherence")
        or details.get("local_reference_marker_coherence")
        or {}
    )
    if not isinstance(coherence, Mapping):
        coherence = {}
    if not coherence and sample_tpm_by_symbol:
        coherence = _marker_coherence(
            hypothesis.cancer_type,
            sample_tpm_by_symbol,
        )
    total = _safe_int(coherence.get("total"), 0)
    unexpected = _marker_coherence_unexpected_low_count(coherence)
    coherence_support = 0.0
    if total > 0 and _marker_coherence_positive_complete(coherence):
        coherence_support = _safe_float(coherence.get("detected_fraction")) * (
            1.0 - min(1.0, unexpected / total)
        )
    support = max(
        coherence_support,
        _safe_float(
            hypothesis.adjudication_axis_support(
                _ENTITY_CONSENSUS_MARKER_AXIS
            )
        ),
    )
    return float(np.clip(support, 0.0, 1.0)), coherence


def _same_registry_branch(left: str, right: str) -> bool:
    """Whether two codes have an ancestor/descendant relationship."""

    left_code = _clean(left)
    right_code = _clean(right)
    return bool(
        left_code
        and right_code
        and (
            left_code == right_code
            or _code_has_registry_ancestor(left_code, right_code)
            or _code_has_registry_ancestor(right_code, left_code)
        )
    )


def _residual_identity_support(
    residual_identity_evidence: Mapping[str, Any] | None,
    cancer_code: str,
) -> float:
    """Return one structural residual-identity vote for a compatible branch.

    Residual identity is one consensus vote, never a standalone selector. A
    cross-branch result must either have a matching lineage-panel program or a
    fully audited ontology result invariant across every background model.
    """

    if not isinstance(residual_identity_evidence, Mapping):
        return 0.0
    if residual_identity_evidence.get("adjudication_eligible") is False:
        return 0.0
    if _clean(residual_identity_evidence.get("status")) not in {
        "candidate",
        "corroborated",
    }:
        return 0.0
    residual_code = _clean(residual_identity_evidence.get("candidate_code"))
    current_code = _clean(residual_identity_evidence.get("current_code"))
    if not residual_code:
        return 0.0
    if current_code and not _same_registry_branch(residual_code, current_code):
        panel_code = _clean(
            residual_identity_evidence.get("panel_candidate_code")
        )
        ontology_code = _clean(
            residual_identity_evidence.get("ontology_candidate_code")
        )
        if not (
            panel_code
            and _same_registry_branch(residual_code, panel_code)
        ) and not (
            ontology_code
            and _same_registry_branch(residual_code, ontology_code)
            and _residual_identity_is_structurally_invariant(
                residual_identity_evidence
            )
        ):
            return 0.0
    return 1.0 if _same_registry_branch(residual_code, cancer_code) else 0.0


def _residual_identity_is_structurally_invariant(
    residual_identity_evidence: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(residual_identity_evidence, Mapping):
        return False
    residual_code = _clean(residual_identity_evidence.get("candidate_code"))
    models = residual_identity_evidence.get("background_models") or ()
    if not residual_code or not models:
        return False
    return all(
        isinstance(model, Mapping)
        and _same_registry_branch(
            _clean(model.get("candidate_code")),
            residual_code,
        )
        and _safe_int(model.get("realizations"), 0) > 0
        for model in models
    )


def _residual_identity_is_source_resolved(
    residual_identity_evidence: Mapping[str, Any] | None,
    cancer_code: str,
) -> bool:
    """Whether decomposition resolved a complete tumor program from host RNA."""

    if not isinstance(residual_identity_evidence, Mapping):
        return False
    residual_code = _clean(residual_identity_evidence.get("candidate_code"))
    return bool(
        residual_identity_evidence.get("source_resolved_identity")
        and residual_identity_evidence.get("adjudication_eligible") is not False
        and _same_registry_branch(residual_code, cancer_code)
        and _residual_identity_is_structurally_invariant(
            residual_identity_evidence
        )
    )


def _source_resolved_identity_corroborators(
    candidate: CancerTypeEvidence,
    selected: CancerTypeEvidence,
    hierarchy_details: Mapping[str, Any],
    axes: Iterable[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Independent bulk views that agree with a source-resolved residual.

    The residual object already integrates the curated positive/negative
    program with candidate-independent background subtraction, so the bulk
    marker axis is intentionally not counted again.  A separate whole-profile,
    reference, composition, or learned-family result must still agree before a
    cross-entity label can change.
    """

    corroborators = [
        _clean(axis.get("axis"))
        for axis in axes
        if _clean(axis.get("preference")) == "candidate"
        and _clean(axis.get("axis"))
        not in {
            _ENTITY_CONSENSUS_MARKER_AXIS,
            _ENTITY_CONSENSUS_RESIDUAL_AXIS,
        }
    ]
    top_family = _clean(
        hierarchy_details.get("learned_expression_top_family_label")
    )
    if top_family:
        try:
            from .expression_classifier import _learned_family_for_code
        except ImportError:
            candidate_family = selected_family = ""
        else:
            candidate_family = _clean(
                _learned_family_for_code(candidate.cancer_type)
            )
            selected_family = _clean(
                _learned_family_for_code(selected.cancer_type)
            )
        top_compartment = _clean(
            hierarchy_details.get("learned_expression_top_compartment_label")
        )
        candidate_path_consistent = _learned_hierarchy_path_consistent(
            candidate.cancer_type,
            family_label=top_family,
            compartment_label=top_compartment,
        )
        if (
            top_family == candidate_family
            and top_family != selected_family
            and candidate_path_consistent
        ):
            corroborators.append("learned_family_leader")
    return tuple(dict.fromkeys(corroborators))


def _qualified_residual_identity_candidate(
    residual_identity_evidence: Mapping[str, Any] | None,
) -> str:
    """Return a consensus-eligible residual entity, never a direct call."""

    if not isinstance(residual_identity_evidence, Mapping):
        return ""
    if residual_identity_evidence.get("adjudication_eligible") is False:
        return ""
    if _clean(residual_identity_evidence.get("status")) != "candidate":
        return ""
    residual_code = _clean(residual_identity_evidence.get("candidate_code"))
    panel_code = _clean(residual_identity_evidence.get("panel_candidate_code"))
    ontology_code = _clean(
        residual_identity_evidence.get("ontology_candidate_code")
    )
    panel_supported = bool(
        panel_code and _same_registry_branch(residual_code, panel_code)
    )
    ontology_supported = bool(
        ontology_code
        and _same_registry_branch(residual_code, ontology_code)
        and _residual_identity_is_structurally_invariant(
            residual_identity_evidence
        )
    )
    return residual_code if residual_code and (panel_supported or ontology_supported) else ""


def _raw_signature_support(hypothesis: CancerTypeEvidence) -> float:
    """Signature-only support, falling back for legacy/test callers."""

    if "signature_score" in hypothesis.details:
        return _safe_float(hypothesis.details.get("signature_score"))
    return _safe_float(hypothesis.broad_rna_support)


def _fused_parent_abstention_blocks_entity_consensus(
    hypothesis: CancerTypeEvidence,
) -> bool:
    """Keep a structured family abstention authoritative downstream.

    A close, better-supported child of a structured family is an explicit
    instruction not to promote the competing leaf. The later learned/entity
    consensus may audit that leaf, but must not recreate it from raw signature
    and centroid axes after fused evidence rejected it.
    """

    return bool(
        hypothesis.details.get("fused_evidence_can_select") is False
        and hypothesis.details.get(
            "fused_evidence_structured_parent_abstention"
        )
    )


def _entity_evidence_consensus(
    candidate: CancerTypeEvidence,
    selected: CancerTypeEvidence,
    hierarchy_details: Mapping[str, Any],
    *,
    sample_tpm_by_symbol: Mapping[str, float] | None = None,
    cen=None,
    centroid_confident: bool = False,
    residual_identity_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare two entity hypotheses across independent evidence groups.

    Each group gets one vote, regardless of how many correlated subfeatures it
    contains.  The decision is intentionally relative: it asks which entity is
    better supported by the same axis in this sample.  No cancer-code-specific
    constants or sample exceptions are involved.
    """

    centroid_supports = _centroid_supports_for_hypotheses(
        {
            candidate.cancer_type: candidate,
            selected.cancer_type: selected,
        },
        cen if centroid_confident else None,
    )
    candidate_marker, candidate_coherence = _entity_marker_program_support(
        candidate,
        sample_tpm_by_symbol,
    )
    selected_marker, selected_coherence = _entity_marker_program_support(
        selected,
        sample_tpm_by_symbol,
    )
    if candidate_coherence:
        candidate.details["entity_evidence_marker_coherence"] = dict(
            candidate_coherence
        )
    if selected_coherence:
        selected.details["entity_evidence_marker_coherence"] = dict(
            selected_coherence
        )

    candidate_residual_support = _residual_identity_support(
        residual_identity_evidence,
        candidate.cancer_type,
    )
    selected_residual_support = _residual_identity_support(
        residual_identity_evidence,
        selected.cancer_type,
    )
    # The residual identity programs reuse the curated high/low marker
    # vocabulary. When that axis is available, omit the bulk marker axis so the
    # same program is not counted twice.
    if candidate_residual_support > 0 or selected_residual_support > 0:
        candidate_marker = 0.0
        selected_marker = 0.0

    raw_axes = {
        "learned_full_profile": (
            _learned_entity_support_for_code(
                hierarchy_details,
                candidate.cancer_type,
            ),
            _learned_entity_support_for_code(
                hierarchy_details,
                selected.cancer_type,
            ),
        ),
        "pan_cancer_signature": (
            _raw_signature_support(candidate),
            _raw_signature_support(selected),
        ),
        "whole_profile_centroid": (
            _safe_float(centroid_supports.get(candidate.cancer_type)),
            _safe_float(centroid_supports.get(selected.cancer_type)),
        ),
        "curated_marker_program": (candidate_marker, selected_marker),
        _ENTITY_CONSENSUS_REFERENCE_AXIS: (
            max(
                _safe_float(
                    candidate.adjudication_axis_support(
                        _ENTITY_CONSENSUS_REFERENCE_AXIS
                    )
                ),
                _safe_float(
                    candidate.details.get(
                        "entity_descendant_exact_reference_support"
                    )
                ),
            ),
            max(
                _safe_float(
                    selected.adjudication_axis_support(
                        _ENTITY_CONSENSUS_REFERENCE_AXIS
                    )
                ),
                _safe_float(
                    selected.details.get(
                        "entity_descendant_exact_reference_support"
                    )
                ),
            ),
        ),
        _ENTITY_CONSENSUS_COMPOSITION_AXIS: (
            candidate.adjudication_axis_support(
                _ENTITY_CONSENSUS_COMPOSITION_AXIS
            ),
            selected.adjudication_axis_support(
                _ENTITY_CONSENSUS_COMPOSITION_AXIS
            ),
        ),
        _ENTITY_CONSENSUS_RESIDUAL_AXIS: (
            candidate_residual_support,
            selected_residual_support,
        ),
    }
    axes: list[dict[str, Any]] = []
    candidate_votes = 0
    selected_votes = 0
    candidate_nonlearned_votes = 0
    candidate_advantage = 0.0
    for axis, (candidate_value, selected_value) in raw_axes.items():
        candidate_support = float(np.clip(_safe_float(candidate_value), 0.0, 1.0))
        selected_support = float(np.clip(_safe_float(selected_value), 0.0, 1.0))
        available = bool(candidate_support > 0 or selected_support > 0)
        preference = "abstain"
        relative_advantage = 0.0
        if available and not np.isclose(
            candidate_support,
            selected_support,
            rtol=1e-6,
            atol=1e-6,
        ):
            scale = max(candidate_support, selected_support)
            relative_advantage = (
                (candidate_support - selected_support) / scale if scale > 0 else 0.0
            )
            if relative_advantage > 0:
                preference = "candidate"
                candidate_votes += 1
                if axis != "learned_full_profile":
                    candidate_nonlearned_votes += 1
            else:
                preference = "selected"
                selected_votes += 1
            candidate_advantage += float(relative_advantage)
        axes.append(
            {
                "axis": axis,
                "candidate_support": round(candidate_support, 4),
                "selected_support": round(selected_support, 4),
                "relative_candidate_advantage": round(
                    float(relative_advantage),
                    4,
                ),
                "preference": preference,
                "available": available,
            }
        )

    candidate_has_learned_vote = any(
        axis["axis"] == "learned_full_profile"
        and axis["preference"] == "candidate"
        for axis in axes
    )
    candidate_has_residual_vote = any(
        axis["axis"] == _ENTITY_CONSENSUS_RESIDUAL_AXIS
        and axis["preference"] == "candidate"
        for axis in axes
    )
    available_axis_count = sum(axis["available"] for axis in axes)
    candidate_has_originating_axis = (
        candidate_has_learned_vote or candidate_has_residual_vote
    )
    required_nonlearned_votes = (
        _ENTITY_CONSENSUS_MIN_NONLEARNED_AXES
        if candidate_has_learned_vote
        else _ENTITY_CONSENSUS_MIN_SUPPORTING_AXES
    )
    majority_decisive_candidate = bool(
        candidate_has_originating_axis
        and candidate_votes >= _ENTITY_CONSENSUS_MIN_SUPPORTING_AXES
        and candidate_nonlearned_votes >= required_nonlearned_votes
        and candidate_votes * 2 > available_axis_count
        and candidate_advantage > 0
    )
    source_resolved_identity = _residual_identity_is_source_resolved(
        residual_identity_evidence,
        candidate.cancer_type,
    )
    source_resolved_corroborators = (
        _source_resolved_identity_corroborators(
            candidate,
            selected,
            hierarchy_details,
            axes,
        )
        if source_resolved_identity and candidate_has_residual_vote
        else ()
    )
    # This is not a plurality exception.  A source-resolved residual is a
    # compound identity selector: every positive/negative tumor program agreed
    # across a candidate-independent background beam, and a separate bulk view
    # must corroborate it.  It is therefore adjudicated alongside, rather than
    # pretending to be several independent votes inside, the majority rule.
    source_resolved_decisive_candidate = bool(
        source_resolved_identity
        and candidate_has_residual_vote
        and source_resolved_corroborators
    )
    decisive_candidate = bool(
        majority_decisive_candidate or source_resolved_decisive_candidate
    )
    return {
        "schema_version": 1,
        "candidate_code": candidate.cancer_type,
        "selected_code": selected.cancer_type,
        "axes": axes,
        "candidate_votes": candidate_votes,
        "candidate_nonlearned_votes": candidate_nonlearned_votes,
        "selected_votes": selected_votes,
        "available_axis_count": available_axis_count,
        "candidate_advantage": round(float(candidate_advantage), 4),
        "candidate_has_learned_vote": candidate_has_learned_vote,
        "candidate_has_residual_vote": candidate_has_residual_vote,
        "majority_decisive_candidate": majority_decisive_candidate,
        "source_resolved_identity": source_resolved_identity,
        "source_resolved_identity_decisive": (
            source_resolved_decisive_candidate
        ),
        "source_resolved_identity_corroborators": list(
            source_resolved_corroborators
        ),
        "decisive_candidate": decisive_candidate,
        "conflicted": bool(candidate_votes > 0 and selected_votes > 0),
        "decision_rule": (
            "a learned-view or invariant-residual entity plus independent "
            "evidence groups must win the available-axis majority; a complete "
            "source-resolved residual may instead select only with separate "
            "bulk corroboration"
        ),
    }


def _entity_consensus_has_candidate_identity_vote(
    consensus: Mapping[str, Any],
) -> bool:
    """Return whether a tumor-identity axis supports the candidate."""

    identity_axes = {
        _ENTITY_CONSENSUS_MARKER_AXIS,
        _ENTITY_CONSENSUS_REFERENCE_AXIS,
        _ENTITY_CONSENSUS_RESIDUAL_AXIS,
    }
    return any(
        _clean(axis.get("axis")) in identity_axes
        and _clean(axis.get("preference")) == "candidate"
        for axis in consensus.get("axes") or ()
        if isinstance(axis, Mapping)
    )


def _entity_consensus_origin_features(
    candidate: CancerTypeEvidence,
    selected: CancerTypeEvidence,
    consensus: Mapping[str, Any],
    *,
    credible_learned_candidate: bool,
    residual_origin: bool = False,
) -> dict[str, bool]:
    """Describe whether the candidate has a valid entity origin.

    Family-level context can establish a different lineage, but it cannot
    distinguish siblings inside one lineage. Same-lineage refinement therefore
    needs the broad top, a credible learned entity, or an identity-specific
    marker/reference/residual vote.
    """

    candidate_lineage = _code_lineage_token(candidate.cancer_type)
    selected_lineage = _code_lineage_token(selected.cancer_type)
    same_lineage_as_selected = bool(
        candidate_lineage
        and selected_lineage
        and candidate_lineage == selected_lineage
    )
    candidate_has_identity_vote = _entity_consensus_has_candidate_identity_vote(
        consensus
    )
    candidate_has_family_origin = _entity_consensus_has_family_anchored_origin(
        candidate
    )
    candidate_has_residual_vote = bool(
        residual_origin or consensus.get("candidate_has_residual_vote")
    )
    candidate_is_broad_top = candidate.broad_rna_rank == 1
    candidate_origin_credible = bool(
        credible_learned_candidate
        or candidate_has_residual_vote
        or candidate_has_identity_vote
        or candidate_is_broad_top
        or (
            candidate_has_family_origin
            and not same_lineage_as_selected
        )
    )
    return {
        "credible_learned_candidate": bool(credible_learned_candidate),
        "candidate_has_identity_vote": candidate_has_identity_vote,
        "candidate_has_family_anchored_origin": candidate_has_family_origin,
        "candidate_has_residual_origin": candidate_has_residual_vote,
        "candidate_is_broad_top": candidate_is_broad_top,
        "same_lineage_as_selected": same_lineage_as_selected,
        "candidate_origin_credible": candidate_origin_credible,
    }


def _entity_consensus_has_family_anchored_origin(
    candidate: CancerTypeEvidence,
) -> bool:
    """Whether learned family and broad RNA independently establish origin.

    The fused component is present only when the hierarchy's leading family
    agrees with this candidate's family and the candidate is already a close
    broad-ranker context. Reusing that admitted component avoids treating a
    negligible learned entity tail as credible while still allowing
    entity-level centroid/reference votes to resolve within the established
    family.
    """

    components = candidate.details.get("fused_evidence_components") or {}
    return bool(
        _safe_float(
            components.get(
                "learned_family_anchored_pan_cancer_context"
            )
        )
        > 0
    )


def _withdraw_fused_selection_superseded_by_entity_adjudication(
    selected: CancerTypeEvidence,
    replacement_code: str,
) -> None:
    """Retain the old fused row as audit evidence, not a second selector."""

    if selected.selected_by != "fused_evidence":
        return
    reason = (
        "entity adjudication superseded fused selection with "
        f"{replacement_code}"
    )
    selected.withdraw_report_label_selector(
        "fused_evidence",
        blocking_reasons=(reason,),
    )
    selected.details["entity_consensus_superseded_fused_selection"] = (
        replacement_code
    )


def _persistent_report_label_blockers(
    hypothesis: CancerTypeEvidence,
) -> tuple[str, ...]:
    """Return vetoes that expression-only adjudication must never clear.

    Ordinary admission blockers can be overcome by genuinely independent RNA
    evidence.  Explicit molecular negatives and registry-declared orthogonal
    molecular/status states cannot: they require matching direct evidence, not
    a stronger expression consensus.  Keep this distinction structural so a
    later selector cannot accidentally erase a safety decision by calling
    ``consider_for_report_label(can_select=True)``.
    """

    if hypothesis.direct_fusion_support > 0:
        return ()
    details = hypothesis.details
    registry_row = _registry_row_for_code(hypothesis.cancer_type)
    declared_blockers = details.get("hard_report_label_blockers") or ()
    if isinstance(declared_blockers, str):
        declared_blockers = (declared_blockers,)
    blockers = [
        _clean(reason)
        for reason in declared_blockers
        if _clean(reason)
    ]
    if registry_row and not _safe_bool(
        registry_row.get("is_classification_target"),
        default=True,
    ):
        blockers.append(
            f"{hypothesis.cancer_type} is registry-declared as a hypothesis/"
            "context code rather than a classification target; direct "
            "molecular evidence is required before it can become the report "
            "label"
        )
    if details.get("local_reference_explicit_negative_fusion"):
        conflict = details.get("local_reference_explicit_negative_fusion_details") or {}
        blockers.append(
            _fusion_input_missing_expected_driver_reason(conflict)
            if isinstance(conflict, Mapping)
            else "fusion-driven expression reference conflicts with supplied fusion evidence"
        )
    orthogonal_axes = _orthogonal_axes_that_block_report_label(
        hypothesis.cancer_type
    )
    if orthogonal_axes and not _rare_marker_channel_admitted(hypothesis):
        axes = ", ".join(
            sorted({_clean(axis.get("axis")) for axis in orthogonal_axes})
        )
        blockers.append(
            f"{hypothesis.cancer_type} encodes orthogonal molecular/status "
            f"state ({axes}); direct molecular evidence is required before "
            "that state can become the report label"
        )
    return tuple(dict.fromkeys(blockers))


def _lowest_common_registry_ancestor(left: str, right: str) -> str:
    """Deepest shared registry parent, excluding the two input leaves."""

    left_code = _clean(left)
    right_code = _clean(right)
    if not left_code or not right_code or left_code == right_code:
        return ""
    right_path = {right_code, *_registry_parent_chain(right_code)}
    for code in (left_code, *_registry_parent_chain(left_code)):
        if code in right_path and code not in {left_code, right_code}:
            row = _registry_row_for_code(code)
            if _safe_bool(row.get("is_classification_target"), default=True):
                return code
    return ""


def _adjudicate_selection_with_learned_hierarchy(
    hypotheses: dict[str, CancerTypeEvidence],
    selected: CancerTypeEvidence | None,
    *,
    sample_tpm_by_symbol: Mapping[str, float] | None = None,
    cen=None,
    centroid_confident: bool = False,
    residual_identity_evidence: Mapping[str, Any] | None = None,
) -> CancerTypeEvidence | None:
    """Apply calibrated hierarchy arbitration after ordinary fused selection.

    Three narrow paths are allowed:

    * a high-precision entity refinement with strong, mutually coherent
      entity/family/compartment votes; and
    * a multi-axis entity refinement when the learned vote and at least two
      independent evidence groups converge; and
    * a lineage-only safety path when the selected and learned entities occupy
      different lineages but the entity/family stages form a valid hierarchy.

    When same-lineage evidence is genuinely split between two descendant
    branches, the selector may abstain to their deepest shared registry parent.
    That preserves the defensible entity scope without fabricating a leaf.

    Definitive molecular selectors and molecular/status annotation codes are
    never overridden by this expression-only adjudicator.
    """

    if (
        selected is None
        or selected.selected_by in _DEFINITIVE_SELECTORS
        or (
            selected.selected_by == "lineage_panel"
            and selected.details.get(
                "lineage_panel_identity_program_decisive"
            )
        )
    ):
        return selected
    details = _learned_hierarchy_details(hypotheses, selected)

    def consensus_beam() -> CancerTypeEvidence | None:
        return _entity_consensus_candidate_beam(
            hypotheses,
            selected,
            details,
            sample_tpm_by_symbol=sample_tpm_by_symbol,
            cen=cen,
            centroid_confident=centroid_confident,
            residual_identity_evidence=residual_identity_evidence,
        )

    learned_entity_code = _clean(
        details.get("learned_expression_top_entity_label")
    )
    if not learned_entity_code or learned_entity_code not in _registry_by_code():
        beam_candidate = consensus_beam()
        if beam_candidate is not None:
            return beam_candidate
        return selected
    entity_code = learned_entity_code
    while entity_code and _orthogonal_axes_that_block_report_label(entity_code):
        parents = _registry_parent_chain(entity_code)
        entity_code = parents[0] if parents else ""
    if not entity_code:
        beam_candidate = consensus_beam()
        return beam_candidate if beam_candidate is not None else selected
    if entity_code == selected.cancer_type:
        beam_candidate = consensus_beam()
        return beam_candidate if beam_candidate is not None else selected

    entity_support = _safe_float(
        details.get("learned_expression_top_entity_support")
    )
    entity_margin = _safe_float(
        details.get("learned_expression_top_entity_margin")
    )
    family_label = _clean(details.get("learned_expression_top_family_label"))
    family_support = _safe_float(
        details.get("learned_expression_top_family_support")
    )
    compartment_label = _clean(
        details.get("learned_expression_top_compartment_label")
    )
    compartment_support = _safe_float(
        details.get("learned_expression_top_compartment_support")
    )
    family_consistent = _learned_hierarchy_path_consistent(
        entity_code,
        family_label=family_label,
    )
    compartment_consistent = _learned_hierarchy_path_consistent(
        entity_code,
        compartment_label=compartment_label,
    )

    try:
        from .cancer_ontology import cancer_codes_entity_compatible
    except ImportError:
        entity_incompatible = entity_code != selected.cancer_type
    else:
        entity_incompatible = not cancer_codes_entity_compatible(
            entity_code,
            selected.cancer_type,
        )

    strong_entity_refinement = bool(
        entity_incompatible
        and family_consistent
        and compartment_consistent
        and entity_support
        >= _LEARNED_HIERARCHY_ENTITY_ADJUDICATION_MIN_SUPPORT
        and entity_margin >= _LEARNED_HIERARCHY_ENTITY_ADJUDICATION_MIN_MARGIN
        and family_support
        >= _LEARNED_HIERARCHY_ENTITY_ADJUDICATION_MIN_FAMILY_SUPPORT
        and compartment_support
        >= _LEARNED_HIERARCHY_ENTITY_ADJUDICATION_MIN_COMPARTMENT_SUPPORT
    )
    selected_lineage = _code_lineage_token(selected.cancer_type)
    entity_lineage = _code_lineage_token(entity_code)
    lineage_disagreement = bool(
        selected_lineage
        and entity_lineage
        and selected_lineage != entity_lineage
    )
    candidate = _hypothesis(hypotheses, entity_code)
    if _fused_parent_abstention_blocks_entity_consensus(candidate):
        candidate.details["entity_consensus_preserved_fused_abstention"] = dict(
            candidate.details.get(
                "fused_evidence_structured_parent_abstention"
            )
            or {}
        )
        return selected
    candidate.add_source("learned_expression_classifier")
    candidate.learned_expression_support = max(
        candidate.learned_expression_support,
        entity_support,
    )
    candidate.expression_reference_cancer_type = (
        candidate.expression_reference_cancer_type or entity_code
    )
    # Preserve sample-global hierarchy stages without transplanting
    # destination-specific context or MMR votes from the previously selected
    # row. Marker coherence is likewise recomputed for this entity below.
    _merge_global_learned_audit_details(
        candidate,
        details,
    )
    candidate_marker_coherence: Mapping[str, Any] = {}
    if sample_tpm_by_symbol:
        candidate_marker_coherence = _marker_coherence(
            entity_code,
            sample_tpm_by_symbol,
        )
        candidate.details["learned_hierarchy_entity_marker_coherence"] = dict(
            candidate_marker_coherence
        )
    consensus = _entity_evidence_consensus(
        candidate,
        selected,
        details,
        sample_tpm_by_symbol=sample_tpm_by_symbol,
        cen=cen,
        centroid_confident=centroid_confident,
        residual_identity_evidence=residual_identity_evidence,
    )
    credible_learned_candidate = bool(
        entity_support >= _LEARNED_EXPRESSION_MIN_PROBABILITY
        and entity_margin >= _LEARNED_EXPRESSION_MIN_MARGIN
    )
    origin_features = _entity_consensus_origin_features(
        candidate,
        selected,
        consensus,
        credible_learned_candidate=credible_learned_candidate,
    )
    consensus.update(origin_features)
    candidate_origin_credible = origin_features["candidate_origin_credible"]
    hard_blockers = _persistent_report_label_blockers(candidate)
    if hard_blockers:
        consensus["evidence_decisive_candidate"] = bool(
            consensus.get("decisive_candidate")
        )
        consensus["decisive_candidate"] = False
        consensus["selection_blocked"] = True
        consensus["candidate_hard_blockers"] = list(hard_blockers)
        candidate.details["entity_consensus_hard_blockers"] = list(
            hard_blockers
        )
    selected.details["entity_evidence_consensus"] = dict(consensus)
    candidate.details["entity_evidence_consensus"] = dict(consensus)
    if hard_blockers:
        return selected
    selected_consensus_majority = bool(
        _safe_int(consensus.get("selected_votes")) * 2
        > _safe_int(consensus.get("available_axis_count"))
        and _safe_float(consensus.get("candidate_advantage")) < 0
    )
    if strong_entity_refinement and selected_consensus_majority:
        strong_entity_refinement = False
        candidate.details[
            "learned_hierarchy_withheld_by_independent_majority"
        ] = dict(consensus)
    multi_axis_entity_refinement = bool(
        entity_incompatible
        and family_consistent
        and compartment_consistent
        and candidate_origin_credible
        and consensus.get("decisive_candidate")
        and (
            not lineage_disagreement
            or any(
                axis.get("axis") == _ENTITY_CONSENSUS_RESIDUAL_AXIS
                and axis.get("preference") == "candidate"
                for axis in consensus.get("axes") or ()
            )
        )
    )

    marker_coherence = (
        candidate_marker_coherence
        or candidate.details.get("learned_hierarchy_entity_marker_coherence")
        or candidate.details.get("entity_evidence_marker_coherence")
        or candidate.details.get("learned_expression_marker_coherence")
        or {}
    )
    marker_corroborated = bool(
        isinstance(marker_coherence, Mapping)
        and _marker_coherence_selection_grade(marker_coherence)
    )
    strong_lineage_entity = bool(
        entity_support
        >= _LEARNED_HIERARCHY_LINEAGE_STRONG_ENTITY_SUPPORT
        and entity_margin
        >= _LEARNED_HIERARCHY_LINEAGE_STRONG_ENTITY_MARGIN
    )
    corroborated_lineage_entity = bool(
        compartment_consistent
        and marker_corroborated
        and entity_support
        >= _LEARNED_HIERARCHY_LINEAGE_CORROBORATED_ENTITY_SUPPORT
        and family_support
        >= _LEARNED_HIERARCHY_LINEAGE_CORROBORATED_FAMILY_SUPPORT
        and compartment_support
        >= _LEARNED_HIERARCHY_LINEAGE_CORROBORATED_COMPARTMENT_SUPPORT
    )
    lineage_safety = bool(
        lineage_disagreement
        and not selected_consensus_majority
        and family_consistent
        and (strong_lineage_entity or corroborated_lineage_entity)
    )
    common_ancestor = ""
    if (
        entity_incompatible
        and not lineage_disagreement
        and family_consistent
        and compartment_consistent
        and consensus.get("conflicted")
        and candidate_origin_credible
        and not strong_entity_refinement
        and not multi_axis_entity_refinement
    ):
        common_ancestor = _lowest_common_registry_ancestor(
            entity_code,
            selected.cancer_type,
        )
    if common_ancestor:
        parent = _hypothesis(hypotheses, common_ancestor)
        parent.add_source("entity_evidence_consensus")
        parent.expression_reference_cancer_type = common_ancestor
        parent.reference_cancer_type = common_ancestor
        _merge_global_learned_audit_details(parent, details)
        _refresh_candidate_mismatch_repair_votes(
            parent,
            sample_tpm_by_symbol,
        )
        parent.details.update(
            {
                "entity_evidence_consensus": dict(consensus),
                "entity_consensus_adjudicated": True,
                "entity_consensus_adjudication_mode": (
                    "common_ancestor_abstention"
                ),
                "entity_consensus_previous_code": selected.cancer_type,
                "entity_consensus_learned_code": entity_code,
                "entity_consensus_learned_raw_code": learned_entity_code,
                "entity_consensus_common_ancestor": common_ancestor,
            }
        )
        parent.basis = (
            f"independent RNA evidence disagreed between {selected.cancer_type} "
            f"and {entity_code}; report scope abstained to their shared "
            f"registry parent {common_ancestor}"
        )
        parent.consider_for_report_label(
            selected_by="entity_evidence_consensus",
            can_select=True,
            blocking_reasons=(),
            priority=(4, 1.0 + abs(_safe_float(consensus.get("candidate_advantage")))),
        )
        if parent.can_select_report_label:
            _withdraw_fused_selection_superseded_by_entity_adjudication(
                selected,
                parent.cancer_type,
            )
        return parent if parent.can_select_report_label else selected

    if (
        not strong_entity_refinement
        and not multi_axis_entity_refinement
        and not lineage_safety
    ):
        beam_candidate = _entity_consensus_candidate_beam(
            hypotheses,
            selected,
            details,
            sample_tpm_by_symbol=sample_tpm_by_symbol,
            cen=cen,
            centroid_confident=centroid_confident,
            residual_identity_evidence=residual_identity_evidence,
        )
        if beam_candidate is not None:
            return beam_candidate
        return selected

    mode = (
        "high_precision_entity_refinement"
        if strong_entity_refinement
        else (
            "multi_axis_entity_refinement"
            if multi_axis_entity_refinement
            else "cross_lineage_safety"
        )
    )
    candidate.details.update(
        {
            "learned_hierarchy_adjudicated": True,
            "learned_hierarchy_adjudication_mode": mode,
            "learned_hierarchy_previous_code": selected.cancer_type,
            "learned_hierarchy_previous_selector": selected.selected_by,
        }
    )
    candidate.basis = (
        f"learned expression hierarchy adjudicated {selected.cancer_type} to "
        f"{entity_code} ({mode.replace('_', ' ')})"
    )
    candidate.consider_for_report_label(
        selected_by=(
            "entity_evidence_consensus"
            if multi_axis_entity_refinement
            else "learned_expression_classifier"
        ),
        can_select=True,
        blocking_reasons=(),
        priority=(4, 1.0 + entity_support + 0.25 * family_support),
    )
    if candidate.can_select_report_label:
        _withdraw_fused_selection_superseded_by_entity_adjudication(
            selected,
            candidate.cancer_type,
        )
        _refresh_candidate_mismatch_repair_votes(
            candidate,
            sample_tpm_by_symbol,
        )
        return candidate
    return selected


def _add_learned_expression_classifier_features(
    hypotheses: dict[str, CancerTypeEvidence],
    sample_tpm_by_symbol: Mapping[str, float],
    analysis: Mapping[str, Any],
) -> None:
    """Add the optional discriminative full-profile classifier as a gated co-signal.

    The learned classifier is strong on reference-like whole-transcriptome profiles, but it is
    deliberately not an oracle: it only selects a report label when its probability/margin are
    adequate and the staged ranker, marker ontology, and compartment evidence do not contradict it.
    """
    if not sample_tpm_by_symbol:
        return
    try:
        from .expression_classifier import classify_expression
    except ImportError:
        return
    predictions = classify_expression(sample_tpm_by_symbol, top_k=5)
    if not predictions:
        return
    hierarchical_votes = _learned_hierarchical_votes(sample_tpm_by_symbol)
    hierarchical_supports = _learned_vote_supports_by_stage(hierarchical_votes)

    code = _clean(predictions[0][0])
    probability = _safe_float(predictions[0][1])
    second_probability = _safe_float(predictions[1][1]) if len(predictions) > 1 else 0.0
    margin = probability - second_probability
    if not code or code not in _registry_by_code():
        return

    support_by_code = _context_support_by_code(analysis)
    context_support, context_code = _context_support_for_code_or_parent(
        code,
        support_by_code,
    )
    learned_entity_support, learned_entity_context = _learned_entity_context_support(
        code,
        hierarchical_supports,
    )
    learned_compartment_support, learned_compartment_label = _learned_compartment_context(
        code,
        hierarchical_supports,
    )
    candidate_rows = _candidate_rows(analysis)
    top_row = candidate_rows[0] if candidate_rows else {}
    top_code = _clean(top_row.get("code")) or _clean(analysis.get("cancer_type"))
    learned_lineage = _code_lineage_token(code)
    top_lineage = _code_lineage_token(top_code)
    marker_coherence = _marker_coherence(code, sample_tpm_by_symbol)

    blockers: list[str] = []
    if probability < _LEARNED_EXPRESSION_MIN_PROBABILITY:
        blockers.append(
            f"learned classifier probability {probability:.2f} is below "
            f"{_LEARNED_EXPRESSION_MIN_PROBABILITY:.2f}"
        )
    if margin < _LEARNED_EXPRESSION_MIN_MARGIN:
        blockers.append(
            f"learned classifier margin {margin:.2f} is below "
            f"{_LEARNED_EXPRESSION_MIN_MARGIN:.2f}"
        )
    marker_blocks_selection = bool(
        marker_coherence
        and not _marker_coherence_selection_grade(marker_coherence)
    )
    if marker_blocks_selection:
        blockers.append(
            f"{code} marker program is {marker_coherence.get('status')} "
            f"({marker_coherence.get('detected')}/"
            f"{marker_coherence.get('total')} expected high markers; "
            f"{marker_coherence.get('required_for_consistent')} required)"
        )

    orthogonal_blocking_axes = _orthogonal_axes_that_block_report_label(code)
    if orthogonal_blocking_axes:
        axes = ", ".join(
            sorted({_clean(axis.get("axis")) for axis in orthogonal_blocking_axes})
        )
        blockers.append(
            f"{code} encodes orthogonal molecular/status state ({axes}); "
            "the learned classifier may annotate this state but cannot make it "
            "the report-scope cancer label without definitive molecular evidence"
        )

    background_compartment = _background_top_compartment_conflict(top_row)
    close_candidate: dict[str, Any] = {}
    if background_compartment and learned_lineage != _lineage_token(background_compartment):
        close_candidate = _close_trace_candidate_for_lineage(
            candidate_rows[1:],
            background_compartment,
            min_support=0.80,
        )
        if close_candidate:
            blockers.append(
                f"learned {code} call follows a background-like {top_code} "
                "context while the whole-profile compartment and a close broad "
                f"candidate support {background_compartment} "
                f"({_clean(close_candidate.get('code'))})"
            )

    compartment_lineage = _lineage_token(top_row.get("centroid_coarse_lineage"))
    compartment_confident = _safe_bool(
        top_row.get("centroid_lineage_confident"),
        default=False,
    )
    compartment_in_set = _row_bool(top_row, "compartment_in_set", True)
    if (
        compartment_confident
        and compartment_lineage
        and learned_lineage
        and learned_lineage != compartment_lineage
        and not compartment_in_set
    ):
        blockers.append(
            f"learned {code} lineage ({learned_lineage}) conflicts with "
            f"confident whole-profile compartment {compartment_lineage}"
        )

    strong_probability = probability >= _LEARNED_EXPRESSION_STRONG_PROBABILITY
    context_corrob = context_support >= _LEARNED_EXPRESSION_MIN_CONTEXT_SUPPORT
    same_context = bool(
        code == top_code
        or _code_has_registry_ancestor(code, top_code)
        or _code_has_registry_ancestor(top_code, code)
    )
    learned_hierarchical_context = max(
        learned_entity_support,
        learned_compartment_support,
    )
    learned_hierarchical_rescue = bool(
        probability >= _LEARNED_EXPRESSION_CONTEXT_FREE_STRONG_PROBABILITY
        and margin >= _LEARNED_EXPRESSION_CONTEXT_FREE_MIN_MARGIN
        and learned_hierarchical_context >= _LEARNED_EXPRESSION_MIN_HIERARCHICAL_CONTEXT
        and not marker_blocks_selection
    )
    if not context_corrob and not (strong_probability and same_context) and not learned_hierarchical_rescue:
        blockers.append(
            f"learned {code} call lacks broad-ranker context support "
            f"({context_support:.2f}; need "
            f"{_LEARNED_EXPRESSION_MIN_CONTEXT_SUPPORT:.2f} or same-context "
            f"probability >= {_LEARNED_EXPRESSION_STRONG_PROBABILITY:.2f}; "
            "context-free selection requires strong hierarchical learned "
            f"support >= {_LEARNED_EXPRESSION_MIN_HIERARCHICAL_CONTEXT:.2f})"
        )

    cross_lineage = bool(top_lineage and learned_lineage and learned_lineage != top_lineage)
    if cross_lineage and not (
        strong_probability
        and (context_corrob or learned_hierarchical_rescue)
    ):
        blockers.append(
            f"cross-lineage learned call {code} needs probability >= "
            f"{_LEARNED_EXPRESSION_STRONG_PROBABILITY:.2f} and either "
            f"broad-ranker context support >= {_LEARNED_EXPRESSION_MIN_CONTEXT_SUPPORT:.2f} "
            "or calibrated hierarchical learned context support"
        )

    hypothesis = _hypothesis(hypotheses, code)
    hypothesis.add_source("learned_expression_classifier")
    hypothesis.expression_reference_cancer_type = (
        hypothesis.expression_reference_cancer_type or code
    )
    hypothesis.reference_cancer_type = hypothesis.reference_cancer_type or (
        context_code or code
    )
    hypothesis.related_context_code = (
        hypothesis.related_context_code or context_code or top_code
    )
    hypothesis.related_context_support = max(
        hypothesis.related_context_support,
        context_support,
    )
    hypothesis.learned_expression_support = max(
        hypothesis.learned_expression_support,
        probability,
    )
    top_predictions = [
        {"code": _clean(pred_code), "probability": round(_safe_float(prob), 4)}
        for pred_code, prob in predictions
    ]
    hypothesis.details.update(
        {
            "learned_expression_probability": round(float(probability), 4),
            "learned_expression_second_probability": round(
                float(second_probability),
                4,
            ),
            "learned_expression_margin": round(float(margin), 4),
            "learned_expression_context_support": round(float(context_support), 4),
            "learned_expression_context_code": context_code,
            "learned_expression_hierarchical_context_support": round(
                float(learned_hierarchical_context),
                4,
            ),
            "learned_expression_entity_context_support": round(
                float(learned_entity_support),
                4,
            ),
            "learned_expression_entity_context": learned_entity_context,
            "learned_expression_compartment_support": round(
                float(learned_compartment_support),
                4,
            ),
            "learned_expression_compartment_label": learned_compartment_label,
            "learned_expression_hierarchical_rescue": bool(learned_hierarchical_rescue),
            "learned_expression_hierarchical_votes": hierarchical_votes,
            "learned_expression_top_predictions": top_predictions,
            "learned_expression_broad_top_code": top_code,
            "learned_expression_broad_top_lineage": top_lineage,
            "learned_expression_lineage": learned_lineage,
            "learned_expression_marker_coherence": marker_coherence,
        }
    )
    if close_candidate:
        hypothesis.details["learned_expression_background_conflict"] = {
            "centroid_compartment_lineage": background_compartment,
            "close_compartment_candidate": _clean(close_candidate.get("code")),
            "close_compartment_candidate_support": round(
                _safe_float(close_candidate.get("support_fraction_of_top")),
                4,
            ),
        }
    hypothesis.basis = hypothesis.basis or (
        f"full-profile learned expression classifier supports {code} "
        f"(p={probability:.2f}, margin={margin:.2f})"
    )
    priority_strength = (
        1.0
        + 0.5 * probability
        + 0.20 * min(max(context_support, learned_entity_support), 1.0)
        + (0.20 if learned_hierarchical_rescue else 0.0)
    )
    hypothesis.consider_for_report_label(
        selected_by="learned_expression_classifier",
        can_select=not blockers,
        blocking_reasons=blockers,
        priority=(2, priority_strength),
    )


def _add_coarse_composition_reference_features(
    hypotheses: dict[str, CancerTypeEvidence],
    sample_tpm_by_symbol: Mapping[str, float],
    analysis: Mapping[str, Any],
) -> None:
    """Promote a strong independent cancer-reference composition match.

    This selector is intentionally data-derived: it uses the pan-reference
    tissue-composition screen plus tumor-up-vs-matched-normal hits for that
    top cohort. It is only allowed to change the report label when the
    first-pass RNA candidate is ambiguous or marker-incoherent, so a clean
    broad classifier call is not displaced by generic tissue background.
    """
    signal = analysis.get("healthy_vs_tumor")
    if signal is None:
        return
    resolved = _resolved_coarse_reference(analysis)
    if not resolved:
        return
    top_code = _clean(resolved.get("code"))
    top_rho = _safe_float(resolved.get("rho"))
    if top_rho < _COARSE_REFERENCE_MIN_RHO:
        return
    if top_code not in _registry_by_code():
        return

    cancer_hint = _clean(
        _tissue_composition_signal_value(signal, "cancer_hint", "")
    ).lower()
    if cancer_hint not in {"tumor-consistent", "possibly-tumor"}:
        return

    second_rho = _safe_float(resolved.get("second_rho"))
    margin = _safe_float(resolved.get("margin"))
    type_specific_cohort = _clean(
        str(
            _tissue_composition_signal_value(signal, "type_specific_cohort", "")
            or ""
        ).removesuffix("_TPM")
    )
    type_specific_hits = list(
        _tissue_composition_signal_value(signal, "type_specific_hits", []) or []
    )
    type_specific_count = (
        len(type_specific_hits) if type_specific_cohort == top_code else 0
    )
    primary_tissue_score = _safe_float(resolved.get("primary_tissue_score"))
    same_tissue_close_codes = _same_tissue_close_coarse_codes(resolved)
    tissue_only_same_tissue_ambiguity = bool(
        margin < _COARSE_REFERENCE_MIN_MARGIN
        and type_specific_count < _COARSE_REFERENCE_MIN_TYPE_SPECIFIC_HITS
        and primary_tissue_score >= _COARSE_REFERENCE_MIN_PRIMARY_TISSUE_SUPPORT
        and same_tissue_close_codes
    )
    structural_tissue_only_ambiguity = (
        _structural_mesenchymal_tissue_only_ambiguity(
            top_code=top_code,
            resolved=resolved,
            margin=margin,
            type_specific_count=type_specific_count,
            primary_tissue_score=primary_tissue_score,
        )
    )

    fit_quality = analysis.get("fit_quality") or {}
    fit_label = ""
    if isinstance(fit_quality, Mapping):
        fit_label = _clean(fit_quality.get("label")).lower()
    broad_uncertain = fit_label in {"weak", "ambiguous"}
    broad_top_code = _top_code(analysis)
    broad_top_marker_coherence = _marker_coherence(
        broad_top_code,
        sample_tpm_by_symbol,
    )
    broad_marker_incoherent = bool(
        broad_top_marker_coherence
        and not _marker_coherence_selection_grade(broad_top_marker_coherence)
    )
    same_as_broad_top = bool(broad_top_code and broad_top_code == top_code)
    close_codes = list(resolved.get("close_codes") or [])
    broad_top_is_close_coarse_match = bool(
        broad_top_code
        and broad_top_code != top_code
        and broad_top_code in close_codes
    )
    broad_and_composition_share_entity_context = bool(
        broad_top_code
        and broad_top_code != top_code
        and _registry_family_for_code(broad_top_code)
        and _primary_tissue_key_for_code(broad_top_code)
        and _registry_family_for_code(broad_top_code)
        == _registry_family_for_code(top_code)
        and _primary_tissue_key_for_code(broad_top_code)
        == _primary_tissue_key_for_code(top_code)
    )
    has_specificity = (
        margin >= _COARSE_REFERENCE_MIN_MARGIN
        or type_specific_count >= _COARSE_REFERENCE_MIN_TYPE_SPECIFIC_HITS
        or primary_tissue_score >= _COARSE_REFERENCE_MIN_PRIMARY_TISSUE_SUPPORT
        or bool(resolved.get("tissue_tiebreak_applied"))
    )

    blockers: list[str] = []
    if not same_as_broad_top and not (broad_uncertain or broad_marker_incoherent):
        blockers.append(
            f"first-pass top-1 ({broad_top_code or 'unknown'}) is not ambiguous "
            "or marker-incoherent, so composition remains contextual"
        )
    if not has_specificity:
        blockers.append(
            "top cancer-reference composition lacks separation or type-specific "
            "tumor-up evidence"
        )
    if tissue_only_same_tissue_ambiguity:
        blockers.append(
            "primary-tissue composition supports "
            f"{resolved.get('primary_tissue') or 'the selected tissue'}, but "
            f"does not distinguish {top_code} from same-tissue close cohort(s) "
            f"{', '.join(same_tissue_close_codes)}; exact report-label "
            "selection requires cancer-reference separation, type-specific "
            "tumor-up evidence, or an exact-reference selector"
        )
    if broad_and_composition_share_entity_context:
        blockers.append(
            "composition supports the same tissue and cancer family as the "
            f"resolved first-pass entity {broad_top_code}, so it cannot by "
            f"itself distinguish sibling entity {top_code}; preserve the "
            "resolved entity unless an entity-specific marker, exact "
            "reference, or learned consensus supports the change"
        )
    if structural_tissue_only_ambiguity:
        blockers.append(
            "primary normal-tissue composition supports structural "
            f"{resolved.get('primary_tissue') or 'mesenchymal'} context, but "
            f"does not distinguish {top_code} tumor from host/background "
            "mesenchymal signal; sarcoma selection requires cancer-reference "
            "separation, type-specific tumor-up hits, an exact reference, "
            "marker panel, fusion, or learned family/entity support"
        )
    if (
        broad_top_is_close_coarse_match
        and margin < _COARSE_REFERENCE_MIN_MARGIN
        and primary_tissue_score < _COARSE_REFERENCE_MIN_PRIMARY_TISSUE_SUPPORT
        and not bool(resolved.get("tissue_tiebreak_applied"))
    ):
        blockers.append(
            "top cancer-reference composition is tied with the first-pass RNA "
            f"winner ({broad_top_code}); type-specific tumor-up evidence alone "
            "does not override the broad RNA call"
        )
    crc_family_lock = _crc_family_locked_against_non_crc_composition(
        broad_top_code=broad_top_code,
        top_code=top_code,
        type_specific_count=type_specific_count,
        analysis=analysis,
        sample_tpm_by_symbol=sample_tpm_by_symbol,
    )
    if crc_family_lock:
        blockers.append(
            "first-pass RNA support is concentrated in the CRC family "
            f"({', '.join(row['code'] for row in crc_family_lock['crc_candidates'])}); "
            f"{top_code} composition remains contextual unless an exact-reference "
            "selector or a coherent marker program corroborates that non-CRC "
            "report label"
        )

    support = float(
        np.clip(
            0.70 * top_rho
            + 0.20 * min(type_specific_count / 4.0, 1.0)
            + 0.10 * _safe_float(resolved.get("primary_tissue_score")),
            0.0,
            1.0,
        )
    )
    hypothesis = _hypothesis(hypotheses, top_code)
    hypothesis.add_source("coarse_composition_reference")
    hypothesis.expression_reference_cancer_type = (
        hypothesis.expression_reference_cancer_type or top_code
    )
    hypothesis.reference_cancer_type = hypothesis.reference_cancer_type or top_code
    hypothesis.related_context_code = hypothesis.related_context_code or top_code
    hypothesis.related_context_support = max(
        hypothesis.related_context_support,
        _coarse_reference_support_by_code(analysis).get(top_code, 0.0),
    )
    hypothesis.coarse_composition_support = max(
        hypothesis.coarse_composition_support,
        support,
    )
    hypothesis.basis = hypothesis.basis or (
        f"independent tissue-composition screen favored {top_code} and "
        "matched its tumor-up expression evidence"
    )
    top_hit_symbols: list[dict[str, Any]] = []
    for item in type_specific_hits[:8]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            top_hit_symbols.append(
                {"gene": str(item[0]), "tpm": round(_safe_float(item[1]), 3)}
            )
    hypothesis.details.update(
        {
            "coarse_reference_rho": round(float(top_rho), 4),
            "coarse_reference_second_rho": round(float(second_rho), 4),
            "coarse_reference_margin": round(float(margin), 4),
            "coarse_reference_raw_top_code": resolved.get("top_code") or top_code,
            "coarse_reference_raw_top_rho": round(
                _safe_float(resolved.get("top_rho")),
                4,
            ),
            "coarse_reference_close_codes": close_codes,
            "coarse_reference_primary_tissue": resolved.get("primary_tissue") or "",
            "coarse_reference_primary_tissue_score": round(
                primary_tissue_score,
                4,
            ),
            "coarse_reference_same_tissue_close_codes": same_tissue_close_codes,
            "coarse_reference_structural_tissue_only_ambiguity": (
                structural_tissue_only_ambiguity
            ),
            "coarse_reference_tissue_tiebreak_applied": bool(
                resolved.get("tissue_tiebreak_applied")
            ),
            "coarse_reference_cancer_hint": cancer_hint,
            "coarse_reference_type_specific_cohort": type_specific_cohort,
            "coarse_reference_type_specific_hit_count": type_specific_count,
            "coarse_reference_crc_family_lock": crc_family_lock,
            "coarse_reference_top_type_specific_hits": top_hit_symbols,
            "coarse_reference_broad_top_code": broad_top_code,
            "coarse_reference_same_tissue_family_broad_top": (
                broad_and_composition_share_entity_context
            ),
            "coarse_reference_broad_fit_label": fit_label,
            "coarse_reference_broad_top_marker_coherence": broad_top_marker_coherence,
        }
    )
    hypothesis.admit_adjudication_support(
        _ENTITY_CONSENSUS_COMPOSITION_AXIS,
        support,
        selector="coarse_composition_reference",
        blocking_reasons=blockers,
    )
    if broad_and_composition_share_entity_context:
        # Same-tissue composition cannot select between sibling entities, but
        # it remains one legitimate corroborating axis for a later learned or
        # marker-led entity consensus.
        hypothesis.admit_adjudication_support(
            _ENTITY_CONSENSUS_COMPOSITION_AXIS,
            support,
            selector="same_tissue_family_composition_context",
        )
    hypothesis.consider_for_report_label(
        selected_by="coarse_composition_reference",
        can_select=not blockers,
        blocking_reasons=blockers,
        priority=(1, support),
    )


def _family_marker_support(row: Mapping[str, Any]) -> float:
    score = _safe_float(row.get("family_score"))
    if score <= 0:
        return 0.0
    return float(min(score / 0.5, 1.0))


def _is_background_like_candidate(row: Mapping[str, Any]) -> bool:
    # ``family_label`` describes the expression program that drove this row,
    # not the registry lineage of its code.  A mesenchymal program attached to
    # an epithelial code (for example UCS) is precisely the situation where
    # the broad label may be tracking background rather than tumor identity.
    return _clean(row.get("family_label")) in _BACKGROUND_LIKE_FAMILIES


def _add_tumor_label_refinement_features(
    hypotheses: dict[str, CancerTypeEvidence],
    analysis: Mapping[str, Any],
) -> None:
    """Select a tumor label when the top broad label is likely background."""
    rows = _candidate_rows(analysis)
    if len(rows) < 2:
        return
    top = rows[0]
    top_is_background_like = _is_background_like_candidate(top)
    crc_family_context: dict[str, Any] = {}
    if not top_is_background_like:
        crc_family_context = _top_label_can_yield_to_crc_family(top, analysis)
    if not top_is_background_like and not crc_family_context:
        return

    top_code = _clean(top.get("code"))
    # Default 0.0 for missing ``support_fraction_of_top`` — the original default of
    # 1.0 silently treated a missing-field background candidate as fully
    # supported, which would have permitted refinement promotion in
    # degenerate traces where the top row was missing its support entry.
    top_support = _safe_float(top.get("support_fraction_of_top"), default=0.0)
    top_signature = _safe_float(top.get("signature_score"))
    if not top_code or top_support <= 0 or top_signature <= 0:
        return
    top_compartment_conflict = _background_top_compartment_conflict(top)

    best: dict[str, Any] | None = None
    best_priority = 0.0
    basal_brca_best: dict[str, Any] | None = None
    for row in rows[1:]:
        code = _clean(row.get("code"))
        if not code or _is_background_like_candidate(row):
            continue
        if crc_family_context and not _code_has_registry_ancestor(
            code,
            _CRC_REGISTRY_ROOT,
        ):
            continue
        support = _safe_float(row.get("support_fraction_of_top"))
        signature = _safe_float(row.get("signature_score"))
        # ``top_signature > 0`` is guaranteed by the outer guard above —
        # the conditional is kept defensive but is always-true here.
        signature_ratio = signature / top_signature
        family_support = _family_marker_support(row)
        basal_brca_override = (
            code == "BRCA"
            and _clean(row.get("winning_subtype")) == "BRCA_Basal"
        )
        row_in_conflict_compartment = bool(
            top_compartment_conflict
            and _code_lineage_token(code) == top_compartment_conflict
        )
        min_support = (
            0.60
            if basal_brca_override
            else (0.90 if row_in_conflict_compartment else _TUMOR_LABEL_MIN_SUPPORT)
        )
        if support < min_support:
            continue
        min_signature_ratio = (
            0.60
            if basal_brca_override
            else (
                0.70
                if row_in_conflict_compartment
                else _TUMOR_LABEL_MIN_SIGNATURE_RATIO
            )
        )
        if signature_ratio < min_signature_ratio:
            continue
        effective_family_support = max(
            family_support,
            1.0 if basal_brca_override else 0.0,
        )
        if effective_family_support < _TUMOR_LABEL_MIN_FAMILY_SUPPORT:
            continue
        priority = (
            0.55 * support
            + 0.25 * min(signature_ratio, 1.0)
            + 0.20 * effective_family_support
            + (0.30 if basal_brca_override else 0.0)
            + (0.10 if row_in_conflict_compartment else 0.0)
        )
        if best is None or priority > best_priority:
            best = dict(row)
            best["tumor_label_basal_brca_override"] = basal_brca_override
            best["tumor_label_effective_family_support"] = effective_family_support
            best["tumor_label_compartment_conflict_override"] = (
                row_in_conflict_compartment
            )
            best["tumor_label_compartment_conflict_lineage"] = (
                top_compartment_conflict
            )
            best_priority = float(priority)
        if basal_brca_override:
            basal_brca_best = dict(row)
            basal_brca_best["tumor_label_basal_brca_override"] = True
            basal_brca_best["tumor_label_effective_family_support"] = (
                effective_family_support
            )
            basal_brca_best["tumor_label_priority"] = float(priority)

    if basal_brca_best is not None:
        best = basal_brca_best
    if best is None:
        return

    code = _clean(best.get("code"))
    support = _safe_float(best.get("support_fraction_of_top"))
    signature = _safe_float(best.get("signature_score"))
    # ``top_signature > 0`` is guaranteed by the outer guard at line 854.
    signature_ratio = signature / top_signature
    family_support = _safe_float(
        best.get("tumor_label_effective_family_support"),
        default=_family_marker_support(best),
    )
    hypothesis = _hypothesis(hypotheses, code)
    hypothesis.add_source("tumor_label_refinement")
    hypothesis.expression_reference_cancer_type = code
    hypothesis.reference_cancer_type = code
    hypothesis.related_context_code = top_code
    hypothesis.related_context_support = max(
        hypothesis.related_context_support,
        top_support,
    )
    hypothesis.family_marker_support = max(
        hypothesis.family_marker_support,
        family_support,
    )
    hypothesis.background_label_support = max(
        hypothesis.background_label_support,
        top_support,
    )
    hypothesis.basis = hypothesis.basis or (
        f"{code} is the strongest non-background tumor label while the top "
        f"{top_code} signal is background-like"
    )
    hypothesis.details.update(
        {
            "competing_background_code": top_code,
            "competing_background_family": _clean(top.get("family_label")),
            "competing_background_support_fraction_of_top": round(float(top_support), 4),
            "tumor_label_support_fraction_of_top": round(float(support), 4),
            "tumor_label_signature_ratio": round(float(signature_ratio), 4),
            "tumor_label_family": _clean(best.get("family_label")),
            "tumor_label_family_support": round(float(family_support), 4),
            "tumor_label_basal_brca_override": bool(
                best.get("tumor_label_basal_brca_override")
            ),
            "tumor_label_compartment_conflict_override": bool(
                best.get("tumor_label_compartment_conflict_override")
            ),
            "tumor_label_compartment_conflict_lineage": _clean(
                best.get("tumor_label_compartment_conflict_lineage")
            ),
            "tumor_label_crc_family_context": crc_family_context,
        }
    )
    hypothesis.consider_for_report_label(
        selected_by="tumor_label_refinement",
        can_select=True,
        blocking_reasons=(),
        priority=(1, best_priority),
    )


def _add_fine_reference_features(
    hypotheses: dict[str, CancerTypeEvidence],
    spec: FineReferenceSpec,
    sample_tpm_by_symbol: Mapping[str, float],
    analysis: Mapping[str, Any],
) -> None:
    support_by_code = _context_support_by_code(analysis)
    context_support = support_by_code.get(spec.reference_cancer_type, 0.0)
    if context_support <= 0:
        return

    ref = _reference_medians(spec.reference_code)
    if not ref:
        return

    metrics: dict[str, float] = {"related_context_support": context_support}
    marker_details_by_group: dict[str, list[dict[str, Any]]] = {}
    for group_name, genes in spec.marker_groups.items():
        fraction, marker_details = _marker_fraction_against_reference(
            sample_tpm_by_symbol,
            genes,
            ref,
        )
        burden = _burden_ratio(sample_tpm_by_symbol, genes, ref)
        metrics[f"{group_name}_marker_fraction"] = fraction
        metrics[f"{group_name}_burden_ratio"] = burden
        marker_details_by_group[group_name] = marker_details

    metrics["all_marker_fraction"] = metrics.get("all_markers_marker_fraction", 0.0)
    fine_reference_support = sum(
        weight * min(metrics.get(metric, 0.0), 1.0)
        for metric, weight in spec.selection_terms
    )

    blockers = _threshold_blocking_reasons(
        metrics,
        spec.minimum_metrics,
        {
            "related_context_support": f"broad {spec.reference_cancer_type} support",
            "osteogenic_burden_ratio": "osteogenic marker burden",
            "osteogenic_marker_fraction": "osteogenic marker fraction",
        },
    )
    if fine_reference_support < spec.min_fine_reference_support:
        blockers.append(
            "fine-reference support "
            f"{fine_reference_support:.2f} is below {spec.min_fine_reference_support:.2f}"
        )

    all_markers = marker_details_by_group.get("all_markers", [])
    top_markers = sorted(
        all_markers,
        key=lambda row: (
            -_safe_float(row.get("ratio_to_reference")),
            row.get("gene", ""),
        ),
    )[:8]

    hypothesis = _hypothesis(hypotheses, spec.cancer_type)
    hypothesis.add_source("fine_reference")
    hypothesis.expression_reference_cancer_type = spec.reference_code
    hypothesis.reference_cancer_type = spec.reference_cancer_type
    hypothesis.related_context_code = spec.reference_cancer_type
    hypothesis.related_context_support = max(
        hypothesis.related_context_support,
        context_support,
    )
    hypothesis.related_context_is_top = spec.reference_cancer_type in _primary_context_codes(analysis)
    hypothesis.fine_reference_support = max(
        hypothesis.fine_reference_support,
        fine_reference_support,
    )
    hypothesis.basis = hypothesis.basis or spec.basis
    hypothesis.details.update(
        {
            "parent_support_fraction_of_top": round(float(context_support), 4),
            f"{spec.reference_cancer_type.lower()}_support_fraction_of_top": round(
                float(context_support),
                4,
            ),
            "fine_reference_strength": round(float(fine_reference_support), 4),
            "marker_fraction": round(metrics.get("all_marker_fraction", 0.0), 4),
            "osteogenic_fraction": round(
                metrics.get("osteogenic_marker_fraction", 0.0),
                4,
            ),
            "osteogenic_burden_ratio": round(
                metrics.get("osteogenic_burden_ratio", 0.0),
                4,
            ),
            "matrix_fraction": round(metrics.get("matrix_marker_fraction", 0.0), 4),
            "amplicon_fraction": round(metrics.get("amplicon_marker_fraction", 0.0), 4),
            "top_markers": top_markers,
            "osteogenic_markers": marker_details_by_group.get("osteogenic", []),
            "amplicon_markers": marker_details_by_group.get("amplicon", []),
        }
    )
    hypothesis.admit_adjudication_support(
        _ENTITY_CONSENSUS_REFERENCE_AXIS,
        fine_reference_support,
        selector="fine_reference",
        blocking_reasons=blockers,
    )
    hypothesis.consider_for_report_label(
        selected_by="fine_reference",
        can_select=not blockers,
        blocking_reasons=blockers,
        priority=(2, fine_reference_support),
    )


def _add_local_expression_reference_features(
    hypotheses: dict[str, CancerTypeEvidence],
    sample_tpm_by_symbol: Mapping[str, float],
    sample_tpm_by_gene_id: Mapping[str, float],
    analysis: Mapping[str, Any],
) -> None:
    # No exact-expression marker program can be evaluated without measured
    # expression. Return before materializing the shared reference panels;
    # empty-input adjudication still retains ranker/refinement metadata from
    # ``analysis`` but no longer pays a large setup cost for an axis that must
    # abstain.
    if not sample_tpm_by_symbol and not sample_tpm_by_gene_id:
        return
    support_by_code = _context_support_by_code(analysis)
    primary_contexts = _primary_context_codes(analysis)
    top = primary_contexts[0] if primary_contexts else _top_code(analysis)
    if not support_by_code or not top:
        return

    # Always request the unfiltered panel set so the lru_cache hits a
    # single ``()`` key. The earlier code passed ``tuple(sorted(support_by_code))``
    # as an early-skip optimization, but ``support_by_code`` varies per
    # sample, which thrashed the cache and rebuilt the panels for every
    # analyzed sample. The consumer below filters by each panel's own
    # ``context_codes`` vs the live ``support_by_code``, so semantically
    # we lose nothing by building all panels once.
    panels = _local_expression_reference_panels()
    if not panels:
        return

    for code, panel in panels.items():
        if code in support_by_code:
            continue
        if code in _SPECIAL_FINE_REFERENCE_CODES:
            continue
        context_codes = tuple(_clean(c) for c in panel.get("context_codes") or ())
        if not context_codes:
            continue
        context_support = max(
            (support_by_code.get(context_code, 0.0) for context_code in context_codes),
            default=0.0,
        )
        context_is_top = any(context in context_codes for context in primary_contexts)
        matched_context = _matched_local_reference_context(
            context_codes,
            primary_contexts,
            support_by_code,
            context_is_top=context_is_top,
        )
        consensus_context = _broad_coarse_consensus_context(analysis)
        context_conflicts_with_consensus = bool(
            consensus_context and consensus_context not in context_codes
        )
        parent_code = _clean(panel.get("parent_code"))
        expression_source = _clean(panel.get("expression_source"))
        molecular_status_source = _is_molecular_status_expression_source(
            expression_source
        ) or _is_molecular_status_expression_source(
            code
        ) or _is_molecular_status_expression_source(
            panel.get("source_cohort")
        )
        fusion_driven = _clean(panel.get("fusion_driven")).lower()
        existing_hypothesis = hypotheses.get(code)
        has_direct_fusion = bool(
            existing_hypothesis and existing_hypothesis.direct_fusion_support > 0
        )
        markers = tuple(panel.get("markers") or ())
        ref_medians = dict(panel.get("ref_medians") or {})
        marker_fraction, burden_ratio, marker_details = _local_reference_marker_support(
            sample_tpm_by_symbol,
            markers,
            ref_medians,
        )
        if marker_fraction <= 0 and burden_ratio <= 0:
            continue
        burden_support = min(burden_ratio / _LOCAL_REFERENCE_BURDEN_SCORE_ANCHOR, 1.0)
        local_support = (
            0.40 * context_support
            + 0.35 * marker_fraction
            + 0.25 * burden_support
        )
        reference_high_confidence = _is_high_confidence_local_reference(
            context_support=context_support,
            marker_fraction=marker_fraction,
            burden_ratio=burden_ratio,
            local_support=local_support,
        )
        code_marker_coherence: dict[str, Any] = {}
        strong_code_marker_coherence = False
        marker_coherence_supported_reference = False
        if (
            not reference_high_confidence
            and local_support >= 0.80
            and context_support >= 0.75
            and marker_fraction >= 0.60
            and burden_ratio >= 0.30
        ):
            code_marker_coherence = _marker_coherence(code, sample_tpm_by_symbol)
            strong_code_marker_coherence = _marker_coherence_strong(
                code_marker_coherence
            )
            marker_coherence_supported_reference = bool(
                strong_code_marker_coherence
                and context_support >= 0.80
                and marker_fraction >= 0.65
                and burden_ratio >= 0.35
            )
        high_confidence_local_reference = bool(
            reference_high_confidence or marker_coherence_supported_reference
        )
        cross_lineage_marker_conflict: dict[str, Any] = {}
        if not code_marker_coherence:
            code_marker_coherence = _marker_coherence(code, sample_tpm_by_symbol)
        active_context_codes = (
            (matched_context or top,) if (matched_context or top) else context_codes
        )
        if code_marker_coherence:
            cross_lineage_marker_conflict = (
                _local_reference_cross_lineage_conflict(
                    code,
                    tuple(c for c in active_context_codes if c),
                    code_marker_coherence,
                )
            )
            if not cross_lineage_marker_conflict and primary_contexts:
                cross_lineage_marker_conflict = (
                    _local_reference_cross_lineage_conflict(
                        code,
                        tuple(c for c in primary_contexts if c),
                        code_marker_coherence,
                    )
                )
                if cross_lineage_marker_conflict:
                    cross_lineage_marker_conflict["basis"] = "first_pass_context"
        unexpected_low_lineage_conflict: dict[str, Any] = {}
        if code_marker_coherence:
            unexpected_low_lineage_conflict = (
                _local_reference_unexpected_low_lineage_conflict(
                    code,
                    code_marker_coherence,
                    analysis,
                )
            )
        primary_tissue = _clean(panel.get("primary_tissue")).lower()
        coherent_code_marker_program = bool(
            code_marker_coherence
            and _marker_coherence_selection_grade(code_marker_coherence)
            and _marker_coherence_unexpected_low_count(code_marker_coherence) == 0
        )
        marker_program_allows_signature_anchor = bool(
            not code_marker_coherence or coherent_code_marker_program
        )
        signature_anchor_support = (
            0.0
            if (
                molecular_status_source
                or not marker_program_allows_signature_anchor
                or context_support >= _LOCAL_REFERENCE_NEAR_TOP_CONTEXT_MIN_SUPPORT
                or primary_tissue
                in _LOCAL_REFERENCE_SIGNATURE_ANCHOR_GENERIC_PRIMARY_TISSUES
                or _local_reference_signature_anchor_context_blocked(
                    primary_tissue,
                    tuple(c for c in primary_contexts if c),
                )
            )
            else _local_reference_signature_anchor_support(
                marker_fraction=marker_fraction,
                burden_ratio=burden_ratio,
                marker_coherence=code_marker_coherence,
                context_codes=tuple(c for c in active_context_codes if c),
            )
        )
        signature_anchored = signature_anchor_support > 0
        signature_anchor_specificity = (
            _local_reference_anchor_specificity(
                code_marker_coherence,
                tuple(c for c in active_context_codes if c),
            )
            if code_marker_coherence
            else {}
        )
        if (
            not context_is_top
            and marker_coherence_supported_reference
            and context_support >= 0.80
        ):
            consensus_lineage = _broad_lineage_for_code(consensus_context)
            context_lineages = {
                _broad_lineage_for_code(context_code)
                for context_code in context_codes
            }
            if not consensus_context or (
                consensus_lineage and consensus_lineage in context_lineages
            ):
                context_is_top = True
        if not context_is_top and _near_top_context_is_supported(
            context_codes,
            support_by_code,
            analysis,
            local_support=local_support,
            allow_consensus_override=(
                high_confidence_local_reference and not parent_code
            ),
        ):
            context_is_top = True
        matched_context = _matched_local_reference_context(
            context_codes,
            primary_contexts,
            support_by_code,
            context_is_top=context_is_top,
        )
        blockers: list[str] = []
        competing_composition_code = next(
            (
                candidate_code
                for candidate_code, candidate in sorted(hypotheses.items())
                if candidate_code != code
                and candidate.can_select_report_label
                and candidate.selected_by == "coarse_composition_reference"
                and _broad_lineage_for_code(candidate_code)
                != _broad_lineage_for_code(code)
            ),
            "",
        )
        if competing_composition_code and not coherent_code_marker_program:
            blockers.append(
                "cross-lineage exact-reference refinement conflicts with an "
                "independently selectable composition identity "
                f"({competing_composition_code}) and lacks a coherent "
                "code-specific expected-high/expected-low marker program"
            )
        if len(markers) < 2 and parent_code:
            blockers.append(
                "a single-gene child expression reference is contextual "
                "evidence, not a coherent program for refining its parent "
                f"{parent_code}"
            )
        first_pass_context_blocker = _local_reference_first_pass_context_blocker(
            primary_tissue,
            primary_contexts,
            context_codes,
        )
        if first_pass_context_blocker:
            blockers.append(first_pass_context_blocker)
        if (
            context_conflicts_with_consensus
            and not context_is_top
            and not signature_anchored
        ):
            blockers.append(
                "pan-cancer signature-ranker context and coarse reference matching both support "
                f"{consensus_context}; exact-reference context "
                f"{', '.join(context_codes)} is only a secondary near-match"
            )
        if cross_lineage_marker_conflict and not signature_anchored:
            unexpected_low = _safe_int(
                cross_lineage_marker_conflict.get("unexpected_low_detected"),
                0,
            )
            unexpected_low_clause = ""
            if unexpected_low:
                genes = ", ".join(
                    cross_lineage_marker_conflict.get("unexpected_low_genes") or []
                )
                unexpected_low_clause = (
                    f"; {unexpected_low} expected-low marker"
                    f"{'' if unexpected_low == 1 else 's'}"
                    f"{f' ({genes})' if genes else ''} contradict the label"
                )
            blockers.append(
                "cross-lineage exact-reference refinement requires a coherent "
                f"{code} marker program; observed "
                f"{cross_lineage_marker_conflict['detected']}/"
                f"{cross_lineage_marker_conflict['total']} expected high "
                f"markers ({cross_lineage_marker_conflict['marker_status']}) "
                f"{unexpected_low_clause} "
                "while the compatible RNA context is "
                f"{', '.join(cross_lineage_marker_conflict['context_codes'])}"
            )
        if unexpected_low_lineage_conflict and not signature_anchored:
            genes = ", ".join(
                unexpected_low_lineage_conflict.get("unexpected_low_genes") or []
            )
            blockers.append(
                "expected-low marker conflict: "
                f"{genes or 'unexpected-low markers'} support a close "
                f"{unexpected_low_lineage_conflict['conflicting_lineage']} "
                "first-pass candidate "
                f"{unexpected_low_lineage_conflict['close_candidate']} "
                f"({unexpected_low_lineage_conflict['close_candidate_support']:.2f}x "
                "top support), so the local expression reference is recorded "
                "as context rather than selecting the report label"
            )
        # A child absent from the first-pass beam is being introduced solely by
        # its local reference here. Expected-low genes that resolve to another
        # lineage make that leaf internally contradictory even when no close
        # sibling happened to enter the truncated ranker beam. Preserve the
        # established parent rather than manufacturing unsupported precision.
        divergent_expected_low = _marker_coherence_unexpected_low_lineages(
            code_marker_coherence
        )
        if (
            parent_code
            and parent_code != code
            and not has_direct_fusion
            and divergent_expected_low
            and not signature_anchored
        ):
            blockers.append(
                "child exact-reference refinement has a contradictory "
                "expected-low lineage program ("
                + "; ".join(
                    f"{lineage}: {', '.join(genes)}"
                    for lineage, genes in sorted(divergent_expected_low.items())
                )
                + "); retain the established parent until independent identity "
                "evidence resolves the leaf"
            )
        background_compartment_conflict = (
            _local_reference_background_compartment_conflict(
                code,
                panel,
                analysis,
            )
        )
        if background_compartment_conflict:
            blockers.append(
                "mesenchymal exact-reference support is treated as "
                "background/context because the broad top is background-like "
                f"({background_compartment_conflict['background_top_code']}) "
                "while the whole-profile compartment and a close broad "
                "candidate support "
                f"{background_compartment_conflict['centroid_compartment_lineage']} "
                f"({background_compartment_conflict['close_compartment_candidate']})"
            )
        molecular_status_blocker = (
            "molecular-status expression reference requires direct molecular "
            "evidence before it can select the report label"
        )
        if molecular_status_source:
            blockers.append(
                molecular_status_blocker
            )
        parent_context_required = bool(
            parent_code
            and parent_code != code
            and parent_code in context_codes
            and parent_code not in support_by_code
            and not _primary_tissue_for_code(parent_code)
        )
        if parent_context_required:
            parent_hypothesis = hypotheses.get(parent_code)
            parent_context_established = bool(
                parent_hypothesis
                and parent_hypothesis.can_select_report_label
            )
            if not parent_context_established:
                blockers.append(
                    f"{code} subgroup expression reference requires established "
                    f"{parent_code} parent-context support before it can select "
                    "the report label"
                )
        smarcb1_tpm = _safe_float(sample_tpm_by_symbol.get("SMARCB1"))
        rhabdoid_smarcb1_max = _RHABDOID_SMARCB1_MAX_TPM_BY_CODE.get(code)
        if (
            code in _RHABDOID_CODES
            and rhabdoid_smarcb1_max is not None
            and smarcb1_tpm > rhabdoid_smarcb1_max
        ):
            blockers.append(
                "rhabdoid exact-reference promotion requires SMARCB1-loss-compatible "
                f"RNA; SMARCB1 is {smarcb1_tpm:.1f} TPM"
            )
        hepb_adult_liver_context_conflict: dict[str, Any] = {}
        if code == "HEPB":
            hepb_adult_liver_context_conflict = _hepb_adult_liver_context_conflict(
                top_code=top,
                context_codes=context_codes,
                support_by_code=support_by_code,
                sample_tpm_by_symbol=sample_tpm_by_symbol,
            )
            if hepb_adult_liver_context_conflict:
                blockers.append(
                    "HEPB exact-reference promotion over a strong LIHC context "
                    "requires fetal-liver anchor support (DLK1, SALL4, or IGF2)"
                )
        adcc_breast_program_conflict: dict[str, Any] = {}
        if code == "ADCC" and fusion_driven == "defining" and not has_direct_fusion:
            myb_tpm = _safe_float(sample_tpm_by_symbol.get("MYB"))
            adcc_breast_program_conflict = _adcc_breast_program_conflict(
                sample_tpm_by_gene_id,
                analysis,
                myb_tpm=myb_tpm,
            )
            if (
                adcc_breast_program_conflict
                or (
                    myb_tpm < _ADCC_PROMOTING_MIN_MYB_TPM
                    and not high_confidence_local_reference
                )
            ):
                if adcc_breast_program_conflict:
                    program = _clean(
                        adcc_breast_program_conflict.get("program")
                    ) or "BRCA"
                    blockers.append(
                        "no-fusion ADCC RNA surrogate conflicts with a strong "
                        f"{program} lineage program in BRCA-compatible "
                        "expression context"
                    )
                else:
                    blockers.append(
                        "ADCC MYB promoter signal "
                        f"{myb_tpm:.1f} TPM is below "
                        f"{_ADCC_PROMOTING_MIN_MYB_TPM:.1f} TPM"
                    )
        fusion_input_driver_conflict: dict[str, Any] = {}
        if fusion_driven in _FUSION_DRIVEN_REFERENCE_STATES and not has_direct_fusion:
            fusion_input_driver_conflict = _fusion_input_missing_expected_driver(
                driver=_clean(panel.get("fusion_driver")),
                analysis=analysis,
            )
            if fusion_input_driver_conflict:
                blockers.append(
                    _fusion_input_missing_expected_driver_reason(
                        fusion_input_driver_conflict
                    )
                )
        fusion_defined_context_conflict: dict[str, Any] = {}
        if (
            fusion_driven == "defining"
            and not has_direct_fusion
            and not fusion_input_driver_conflict
        ):
            fusion_defined_context_conflict = (
                _unconfirmed_fusion_defined_context_conflict(
                    code=code,
                    context_codes=context_codes,
                    primary_tissue=_clean(panel.get("primary_tissue")),
                    analysis=analysis,
                    sample_tpm_by_symbol=sample_tpm_by_symbol,
                )
            )
            if fusion_defined_context_conflict:
                blockers.append(
                    _fusion_defined_context_conflict_reason(
                        fusion_defined_context_conflict
                    )
                )
        fusion_confirmation_caveat = ""
        if (
            fusion_driven in _FUSION_DRIVEN_REFERENCE_STATES
            and not has_direct_fusion
            and not fusion_input_driver_conflict
        ):
            if high_confidence_local_reference:
                driver = _clean(panel.get("fusion_driver")) or "the defining fusion"
                fusion_confirmation_caveat = (
                    f"{code} has high-confidence exact expression-reference support, "
                    f"but this entity is fusion-defined ({driver}); confirm the "
                    "fusion before treating the RNA label as definitive."
                )
            elif fusion_driven == "defining":
                blockers.append(
                    "fusion-defined expression reference requires a matching fusion "
                    "or high-confidence exact expression-reference support before it "
                    "can select the report label"
                )
        if not context_is_top and not signature_anchored:
            blockers.append(
                "expression-reference context is not the top compatible context "
                f"({', '.join(context_codes)})"
            )
        if (
            context_support < _LOCAL_REFERENCE_MIN_CONTEXT_SUPPORT
            and not signature_anchored
        ):
            blockers.append(
                "compatible first-pass RNA support "
                f"{context_support:.2f} is below "
                f"{_LOCAL_REFERENCE_MIN_CONTEXT_SUPPORT:.2f}"
            )
        min_marker_fraction = _LOCAL_REFERENCE_MIN_MARKER_FRACTION_BY_CODE.get(
            code,
            _LOCAL_REFERENCE_MIN_MARKER_FRACTION,
        )
        if marker_fraction < min_marker_fraction:
            blockers.append(
                "local-reference marker fraction "
                f"{marker_fraction:.2f} is below "
                f"{min_marker_fraction:.2f}"
            )
        if burden_ratio < _LOCAL_REFERENCE_MIN_BURDEN_RATIO:
            blockers.append(
                "local-reference marker burden "
                f"{burden_ratio:.2f} is below "
                f"{_LOCAL_REFERENCE_MIN_BURDEN_RATIO:.2f}"
            )
        if local_support < _LOCAL_REFERENCE_MIN_SUPPORT and not signature_anchored:
            blockers.append(
                "local-reference support "
                f"{local_support:.2f} is below "
                f"{_LOCAL_REFERENCE_MIN_SUPPORT:.2f}"
            )
        reference_kind = str(panel.get("reference_kind") or "")
        context_focus = _local_reference_context_focus(context_codes, matched_context)
        specificity_bonus = _local_reference_family_specificity_bonus(
            str(panel.get("family") or "")
        )
        priority_support = local_support + (
            _LOCAL_REFERENCE_DECONVOLVED_PRIORITY_BONUS
            if reference_kind == "deconvolved_tumor_reference"
            else 0.0
        ) + (
            _LOCAL_REFERENCE_CONTEXT_FOCUS_PRIORITY_BONUS * context_focus
        ) + specificity_bonus

        hypothesis = _hypothesis(hypotheses, code)
        hypothesis.add_source("local_expression_reference")
        hypothesis.expression_reference_cancer_type = (
            hypothesis.expression_reference_cancer_type or code
        )
        hypothesis.reference_cancer_type = hypothesis.reference_cancer_type or (
            matched_context or top
        )
        hypothesis.related_context_code = ",".join(context_codes)
        hypothesis.related_context_support = max(
            hypothesis.related_context_support,
            context_support,
        )
        # ``related_context_is_top`` reflects the *best* panel — the one
        # whose ``local_support`` becomes the hypothesis-level strength.
        # OR-accumulating would let a passing-but-weaker panel's True
        # leak forward to a stronger panel whose context isn't top
        # (silently inflating the reader's confidence about which
        # context backed the call).
        if local_support > hypothesis.fine_reference_support:
            hypothesis.related_context_is_top = context_is_top
        hypothesis.fine_reference_support = max(
            hypothesis.fine_reference_support,
            local_support,
        )
        hypothesis.basis = hypothesis.basis or (
            f"{code} has an exact local expression reference inside the "
            f"{hypothesis.reference_cancer_type} RNA context"
        )
        if fusion_confirmation_caveat and not hypothesis.caveat:
            hypothesis.caveat = fusion_confirmation_caveat
        if fusion_confirmation_caveat and not hypothesis.confirmatory_tests:
            driver = _clean(panel.get("fusion_driver")) or "defining fusion"
            hypothesis.confirmatory_tests = f"{driver} fusion testing"
        hypothesis.details.update(
            {
                "local_reference_context_codes": list(context_codes),
                "local_reference_parent_code": parent_code,
                "local_reference_primary_context_codes": list(primary_contexts),
                "local_reference_matched_context": matched_context,
                "local_reference_context_is_top": context_is_top,
                "local_reference_strength": round(float(local_support), 4),
                "local_reference_marker_fraction": round(
                    float(marker_fraction),
                    4,
                ),
                "local_reference_marker_count": len(markers),
                "local_reference_marker_burden_ratio": round(
                    float(burden_ratio),
                    4,
                ),
                "local_reference_context_focus": round(float(context_focus), 4),
                "local_reference_high_confidence": bool(
                    high_confidence_local_reference
                ),
                "local_reference_reference_high_confidence": bool(
                    reference_high_confidence
                ),
                "local_reference_marker_coherence_supported": bool(
                    marker_coherence_supported_reference
                ),
                "local_reference_signature_anchored": bool(signature_anchored),
                "local_reference_competing_composition_code": (
                    competing_composition_code
                ),
                "local_reference_coherent_code_marker_program": bool(
                    coherent_code_marker_program
                ),
                "local_reference_signature_anchor_support": round(
                    float(signature_anchor_support),
                    4,
                ),
                "local_reference_signature_anchor_specificity": (
                    signature_anchor_specificity
                ),
                "local_reference_specificity_bonus": round(
                    float(specificity_bonus),
                    4,
                ),
                "local_reference_priority_support": round(float(priority_support), 4),
                "local_reference_family": panel.get("family") or "",
                "local_reference_primary_tissue": panel.get("primary_tissue") or "",
                "local_reference_source_cohort": panel.get("source_cohort") or "",
                "local_reference_kind": reference_kind,
                "local_reference_expression_source": expression_source,
                "local_reference_fusion_driven": panel.get("fusion_driven") or "",
                "local_reference_fusion_driver": panel.get("fusion_driver") or "",
                "local_reference_requires_fusion_confirmation": bool(
                    fusion_confirmation_caveat
                ),
                "local_reference_explicit_negative_fusion": bool(
                    fusion_input_driver_conflict
                ),
                "local_reference_cross_lineage_marker_conflict": bool(
                    cross_lineage_marker_conflict
                ),
                "local_reference_unexpected_low_lineage_conflict": bool(
                    unexpected_low_lineage_conflict
                ),
                "local_reference_background_compartment_conflict": bool(
                    background_compartment_conflict
                ),
                "fusion_defined_context_conflict": bool(
                    fusion_defined_context_conflict
                ),
                "hepb_adult_liver_context_conflict": bool(
                    hepb_adult_liver_context_conflict
                ),
                "adcc_low_myb_basal_breast_conflict": bool(
                    adcc_breast_program_conflict
                    and adcc_breast_program_conflict.get("reason")
                    == "low_myb_basal_mammary"
                ),
                "adcc_breast_program_conflict": bool(
                    adcc_breast_program_conflict
                ),
                "local_reference_registry_notes": panel.get("registry_notes") or "",
                "top_markers": marker_details[:8],
            }
        )
        if adcc_breast_program_conflict:
            hypothesis.details["adcc_breast_program_conflict_details"] = (
                adcc_breast_program_conflict
            )
        if fusion_defined_context_conflict:
            hypothesis.details["fusion_defined_context_conflict_details"] = (
                fusion_defined_context_conflict
            )
        if fusion_input_driver_conflict:
            hypothesis.details["local_reference_explicit_negative_fusion_details"] = (
                fusion_input_driver_conflict
            )
        if hepb_adult_liver_context_conflict:
            hypothesis.details["hepb_adult_liver_context_conflict_details"] = (
                hepb_adult_liver_context_conflict
            )
        if code_marker_coherence:
            hypothesis.details["local_reference_marker_coherence"] = (
                code_marker_coherence
            )
        if cross_lineage_marker_conflict:
            hypothesis.details["local_reference_cross_lineage_marker_conflict_details"] = (
                cross_lineage_marker_conflict
            )
        if unexpected_low_lineage_conflict:
            hypothesis.details[
                "local_reference_unexpected_low_lineage_conflict_details"
            ] = unexpected_low_lineage_conflict
        if background_compartment_conflict:
            hypothesis.details["local_reference_background_compartment_conflict_details"] = (
                background_compartment_conflict
            )
        hypothesis.admit_adjudication_support(
            _ENTITY_CONSENSUS_REFERENCE_AXIS,
            local_support,
            selector="local_expression_reference",
            blocking_reasons=blockers,
        )
        hypothesis.consider_for_report_label(
            selected_by="local_expression_reference",
            can_select=not blockers,
            blocking_reasons=blockers,
            priority=(
                2 if (high_confidence_local_reference or signature_anchored) else 1,
                max(priority_support, signature_anchor_support),
            ),
        )
        if (
            parent_code
            and parent_code != code
            and molecular_status_source
            and high_confidence_local_reference
        ):
            parent_blockers = [
                reason for reason in blockers if reason != molecular_status_blocker
            ]
            if not (
                _code_has_registry_ancestor(top, parent_code)
                or _code_has_registry_ancestor(parent_code, top)
            ):
                parent_blockers.append(
                    f"{code} molecular/status expression can annotate or refine "
                    f"only an established {parent_code} parent diagnosis; the "
                    f"primary RNA context is {top}"
                )
            parent_marker_coherence = _marker_coherence(
                parent_code,
                sample_tpm_by_symbol,
            )
            if (
                parent_marker_coherence
                and not _marker_coherence_selection_grade(parent_marker_coherence)
                and parent_code != top
            ):
                parent_blockers.append(
                    f"{parent_code} marker program is "
                    f"{parent_marker_coherence.get('status')} "
                    f"({parent_marker_coherence.get('detected')}/"
                    f"{parent_marker_coherence.get('total')} expected high markers; "
                    f"{parent_marker_coherence.get('required_for_consistent')} "
                    "required), so a molecular/status expression subtype cannot "
                    "promote the parent label by itself"
                )
            conflicting_coarse = _strong_conflicting_coarse_reference(
                analysis,
                parent_code,
            )
            if conflicting_coarse:
                parent_blockers.append(
                    "molecular/status expression subtype conflicts with a strong "
                    f"independent composition reference ({conflicting_coarse['code']} "
                    f"rho {conflicting_coarse['rho']:.2f}; "
                    f"{conflicting_coarse['type_specific_hit_count']} "
                    "type-specific tumor-up hits)"
                )
            parent = _hypothesis(hypotheses, parent_code)
            parent.add_source("local_expression_reference")
            parent.expression_reference_cancer_type = (
                parent.expression_reference_cancer_type or parent_code
            )
            parent.reference_cancer_type = parent.reference_cancer_type or (
                matched_context or parent_code
            )
            parent.related_context_code = ",".join(context_codes)
            parent.related_context_support = max(
                parent.related_context_support,
                context_support,
            )
            parent_priority = (
                1,
                max(
                    0.0,
                    priority_support
                    - _LOCAL_REFERENCE_STATUS_PARENT_PRIORITY_PENALTY,
                ),
            )
            parent_full_priority = (
                parent_priority[0],
                parent_priority[1],
                _SELECTED_BY_TIEBREAK_RANK.get("local_expression_reference", 0),
            )
            parent_details_match_selection = bool(
                not parent_blockers
                and parent_full_priority > parent.selection_priority
            )
            parent_has_selected_local_reference = bool(
                parent.can_select_report_label
                and parent.selected_by == "local_expression_reference"
            )
            parent_details_should_update = bool(
                parent_details_match_selection
                or (
                    not parent_has_selected_local_reference
                    and local_support > parent.fine_reference_support
                )
            )
            if parent_details_should_update:
                parent.related_context_is_top = context_is_top
                parent.details.update(
                    {
                        "local_reference_status_child_code": code,
                        "local_reference_context_codes": list(context_codes),
                        "local_reference_primary_context_codes": list(primary_contexts),
                        "local_reference_matched_context": matched_context,
                        "local_reference_context_is_top": context_is_top,
                        "local_reference_strength": round(float(local_support), 4),
                        "local_reference_marker_fraction": round(
                            float(marker_fraction),
                            4,
                        ),
                        "local_reference_marker_burden_ratio": round(
                            float(burden_ratio),
                            4,
                        ),
                        "local_reference_context_focus": round(
                            float(context_focus),
                            4,
                        ),
                        "local_reference_high_confidence": bool(
                            high_confidence_local_reference
                        ),
                        "local_reference_priority_support": round(
                            float(priority_support),
                            4,
                        ),
                        "local_reference_child_expression_source": expression_source,
                        "local_reference_source_cohort": panel.get("source_cohort") or "",
                        "local_reference_kind": reference_kind,
                        "local_reference_parent_marker_coherence": parent_marker_coherence,
                        "local_reference_conflicting_coarse_reference": conflicting_coarse,
                        "top_markers": marker_details[:8],
                    }
                )
            parent.fine_reference_support = max(
                parent.fine_reference_support,
                local_support,
            )
            parent.basis = parent.basis or (
                f"{parent_code} is supported by a high-confidence {code} "
                "expression reference; the molecular/status subtype itself "
                "requires orthogonal evidence"
            )
            parent.admit_adjudication_support(
                _ENTITY_CONSENSUS_REFERENCE_AXIS,
                local_support,
                selector="local_expression_reference",
                blocking_reasons=parent_blockers,
            )
            parent.consider_for_report_label(
                selected_by="local_expression_reference",
                can_select=not parent_blockers,
                blocking_reasons=parent_blockers,
                # A molecular/status child cohort can corroborate its parent
                # cancer type, but it is not the same evidentiary object as a
                # direct exact cancer-type reference. Keep it below
                # high-confidence direct references so mutation- or
                # expression-status cohorts do not overrule a coherent
                # cancer-type cohort match.
                priority=parent_priority,
            )


def _rare_rna_policy(code: str) -> RareRnaPolicy:
    return _RARE_RNA_POLICIES.get(code, _DEFAULT_RARE_RNA_POLICY)


def _rare_marker_support_status(
    finding: Mapping[str, Any],
    rule: Mapping[str, Any],
) -> tuple[bool, int, int, int]:
    support_genes = list(finding.get("support_genes") or [])
    missing_support_genes = list(finding.get("missing_support_genes") or [])
    min_support = _safe_int(
        finding.get("min_support_genes"),
        _safe_int(rule.get("min_support_genes"), 0),
    )
    required_count = _safe_int(
        finding.get("required_support_gene_count"),
        len(support_genes) + len(missing_support_genes),
    )
    support_count = _safe_int(finding.get("support_gene_count"), len(support_genes))
    if "support_pass" in finding:
        support_pass = _safe_bool(finding.get("support_pass"), default=False)
    else:
        support_pass = support_count >= min_support
    return support_pass, support_count, min_support, required_count


def _rare_marker_expression_support(
    finding: Mapping[str, Any],
    *,
    support_count: int,
    min_support: int,
    required_count: int,
) -> float:
    primary_tpm = _safe_float(finding.get("surrogate_tpm"))
    threshold = max(_safe_float(finding.get("threshold_tpm")), 1e-9)
    primary_support = min(primary_tpm / threshold, 1.0)
    if required_count <= 0:
        return primary_support
    threshold_support = min(support_count / max(min_support, 1), 1.0)
    breadth_support = min(support_count / max(required_count, 1), 1.0)
    co_marker_support = 0.70 * threshold_support + 0.30 * breadth_support
    return float(0.75 * primary_support + 0.25 * co_marker_support)


def _analysis_has_expression_concentration_warning(analysis: Mapping[str, Any]) -> bool:
    """Whether broad whole-profile calls should be treated as technically fragile."""

    def _level_is_concentrated(value: Any) -> bool:
        return _clean(value).lower() in {"high", "extreme"}

    def _flags_have_concentration(flags: Any) -> bool:
        return any(
            "concentration" in str(flag).lower()
            or "dominated by a tiny transcript set" in str(flag).lower()
            for flag in (flags or [])
        )

    for key in ("expression_scale_qc", "raw_expression_scale_qc"):
        scale_qc = analysis.get(key)
        if not isinstance(scale_qc, Mapping):
            continue
        if _level_is_concentrated(
            scale_qc.get("expression_concentration_level")
            or scale_qc.get("concentration_level")
        ):
            return True
        if _flags_have_concentration(scale_qc.get("warnings")):
            return True
        if _safe_float(scale_qc.get("top_gene_share_of_total_tpm")) >= 0.20:
            return True

    sample_context = analysis.get("sample_context")
    if isinstance(sample_context, Mapping):
        signals = sample_context.get("signals") or {}
        flags = sample_context.get("flags") or []
    else:
        signals = getattr(sample_context, "signals", {}) if sample_context is not None else {}
        flags = getattr(sample_context, "flags", []) if sample_context is not None else []
    if isinstance(signals, Mapping):
        if _level_is_concentrated(signals.get("expression_concentration_level")):
            return True
        if _safe_float(signals.get("top_gene_share_of_total_tpm")) >= 0.20:
            return True
    if _flags_have_concentration(flags):
        return True

    quality = analysis.get("quality")
    if isinstance(quality, Mapping) and _flags_have_concentration(
        quality.get("filtered_flags") or quality.get("flags")
    ):
        return True
    return False


def _analysis_has_expression_lineage_conflict(analysis: Mapping[str, Any]) -> bool:
    """Whether decomposition says the selected code and expression lineage diverge."""

    def _mapping_has_conflict(payload: Any) -> bool:
        if not isinstance(payload, Mapping):
            return False
        if _safe_bool(payload.get("lineage_conflict"), default=False):
            return True
        expression_lineage = _clean(
            payload.get("expression_lineage_compartment")
            or payload.get("expression_lineage")
            or payload.get("compartment")
        ).lower()
        code_lineage = _clean(
            payload.get("lineage_compartment")
            or payload.get("code_lineage")
            or payload.get("expected_lineage_compartment")
            or payload.get("expected_lineage")
        ).lower()
        return bool(
            expression_lineage
            and code_lineage
            and expression_lineage != code_lineage
        )

    if _mapping_has_conflict(analysis):
        return True
    purity = analysis.get("purity")
    if _mapping_has_conflict(purity):
        return True
    if isinstance(purity, Mapping):
        components = purity.get("components")
        if _mapping_has_conflict(components):
            return True
        if isinstance(components, Mapping):
            for key in ("decomposition", "expression_decomposition"):
                if _mapping_has_conflict(components.get(key)):
                    return True
        for key in ("decomposition", "expression_decomposition"):
            if _mapping_has_conflict(purity.get(key)):
                return True
    for key in ("decomposition", "expression_decomposition"):
        if _mapping_has_conflict(analysis.get(key)):
            return True
    return False


def _add_lineage_panel_features(
    hypotheses: dict[str, CancerTypeEvidence],
    sample_tpm_by_symbol: Mapping[str, float],
    analysis: Mapping[str, Any],
    *,
    sample_tpm_by_gene_id: Mapping[str, float] | None = None,
) -> dict[str, Any] | None:
    """Evaluate ``trufflepig.lineage_panels`` against the sample and
    register hypotheses for any panel that clearly wins.

    A "clear entity win" requires:
      - top panel score >= ``_LINEAGE_PANEL_MIN_SCORE`` (positive
        markers cohort-comparable, low markers compliant, obligates
        passed),
      - either score separation from the best panel belonging to a
        different cancer entity, or a complete positive/negative-marker
        program with no complete competing entity program,
      - panel's parent_cohort is a valid registry code.

    Returns the ``summarize_evidence`` dict (with an extra
    ``promotion`` block describing whether the panel actually
    promoted a report-label hypothesis) whenever panels were
    successfully evaluated, even if no panel cleared the promotion
    gates. The caller can stash this on the
    cancer_type_evidence return so downstream consumers (reports,
    analysis-parameters.json) can render the panel verdict. Returns
    ``None`` when evaluation was skipped (empty sample, missing
    module, evaluator exception).

    Graceful degradation: when ``pirlygenes`` or the pan-cancer
    expression frame is unavailable, the cross-code path falls back to
    requiring ``fit_quality.label in {"weak", "ambiguous"}``
    (the same-code shortcut still works since it only compares
    string codes, not registry families). This degrades panel
    sensitivity but keeps the rest of the consolidator running —
    preferred over crashing.

    See issue #42 and ``docs/CANCER_CALL_DECISION_FLOW.md`` for the
    design context. Failures are non-fatal — selector logs and
    returns; the broader cancer-type-evidence pipeline still runs.
    """
    if not sample_tpm_by_symbol and not sample_tpm_by_gene_id:
        return None
    try:
        from .lineage_panels import (
            LINEAGE_PANELS,
            complete_program_entity_decision,
            evaluate_panels,
            summarize_evidence,
        )
    except ImportError:
        _LOGGER.warning(
            "lineage_panels unavailable; skipping selector", exc_info=True
        )
        return None

    # Internal lookups go through gene_id. If the caller didn't pre-
    # compute the ID-keyed view (the existing main.py path passes
    # only sample_tpm_by_symbol), build it on the fly via the shared
    # symbol→ID resolver in common.py. Both views are equivalent for
    # our purposes; the ID-keyed view is what ``evaluate_panels``
    # actually consumes.
    if sample_tpm_by_gene_id is None or not sample_tpm_by_gene_id:
        try:
            from .common import panel_symbols_to_gene_ids
        except ImportError:
            return None
        sym_to_id = panel_symbols_to_gene_ids(list(sample_tpm_by_symbol.keys()))
        built: dict[str, float] = {}
        for sym, tpm in sample_tpm_by_symbol.items():
            gid = sym_to_id.get(sym)
            if not gid:
                continue
            existing = built.get(gid)
            f_tpm = float(tpm)
            if existing is None or f_tpm > existing:
                built[gid] = f_tpm
        sample_tpm_by_gene_id = built

    try:
        evidence = evaluate_panels(LINEAGE_PANELS, sample_tpm_by_gene_id)
    except Exception:  # noqa: BLE001
        _LOGGER.warning("evaluate_panels failed; skipping selector", exc_info=True)
        return None

    summary = dict(summarize_evidence(evidence))
    program_decision = complete_program_entity_decision(evidence)
    summary["entity_program_decision"] = program_decision
    # Initialize the promotion block once — every return path below
    # rewrites it (either with the appropriate blocker string for an
    # early-return reason, or with the final can_promote verdict at
    # the bottom). No setdefault dance needed.
    summary["promotion"] = {"promoted": False, "code": None, "blockers": []}
    top_score = float(summary.get("top_score") or 0.0)
    margin = float(summary.get("margin_over_second") or 0.0)
    top_panel = summary.get("top_panel")
    if not top_panel or top_score < _LINEAGE_PANEL_MIN_SCORE:
        summary["promotion"]["blockers"].append(
            f"top score {top_score:.2f} below threshold "
            f"{_LINEAGE_PANEL_MIN_SCORE:.2f}"
        )
        return summary
    entity_margin = _safe_float(
        program_decision.get("margin_over_competing_entity"),
        margin,
    )
    separated_by_score = entity_margin >= _LINEAGE_PANEL_MIN_MARGIN_OVER_SECOND
    separated_by_complete_program = bool(program_decision.get("decisive"))
    if not separated_by_score and not separated_by_complete_program:
        summary["promotion"]["blockers"].append(
            f"margin over competing cancer entity {entity_margin:.2f} below "
            f"threshold {_LINEAGE_PANEL_MIN_MARGIN_OVER_SECOND:.2f}, and "
            f"{program_decision.get('reason') or 'the top program is not decisive'}"
        )
        return summary

    # Map panel back to its parent_cohort (the cancer code we'd
    # propose). Find the LineagePanel by name.
    panel = next((p for p in LINEAGE_PANELS if p.name == top_panel), None)
    if panel is None or not panel.parent_cohort:
        summary["promotion"]["blockers"].append(
            "winning panel has no parent_cohort mapping"
        )
        return summary
    code = _clean(panel.parent_cohort)
    if not code:
        summary["promotion"]["blockers"].append(
            "winning panel parent_cohort did not clean to a code"
        )
        return summary

    # A complete phenotype plus explicit multi-marker tissue-identity groups
    # is qualitatively different from a generic morphology panel. It can
    # resolve an out-of-beam site call because it establishes origin with
    # lineage-specific markers rather than re-reading the same whole-profile
    # similarity that produced the broad ranker, centroid, and composition
    # calls. ``complete_program_entity_decision`` also requires that no other
    # cancer entity has a complete competing program.
    identity_program_decisive = bool(
        separated_by_complete_program
        and getattr(panel, "identity_marker_groups", ())
        and program_decision.get("top_identity_specific")
    )

    # Stamp the winning panel's biological program note onto the
    # summary so brief.py can render the subtype line without
    # importing LINEAGE_PANELS. Single source of truth: the note
    # lives ONLY on LineagePanel.program_note.
    summary["top_panel_program_note"] = str(getattr(panel, "program_note", "") or "")

    top_rationale = summary.get("top_rationale") or f"{top_panel} matched"
    hypothesis = _hypothesis(hypotheses, code)
    hypothesis.add_source("lineage_panel")
    # ``basis`` is set ONLY when no prior selector has stamped one. A
    # higher-priority selector (fine_reference, local_expression_reference)
    # that fired earlier already owns the rationale string used in the
    # report; lineage_panel piggybacks its own narrative via the
    # ``lineage_panel_*`` keys in ``details`` (rendered separately by
    # brief.py). This avoids the panel narrative clobbering a more
    # specific selector's basis line.
    hypothesis.basis = hypothesis.basis or (
        f"lineage panel reading: {top_rationale}"
    )
    hypothesis.details.update(
        {
            "lineage_panel_top": top_panel,
            "lineage_panel_score": round(top_score, 4),
            "lineage_panel_margin_over_second": round(margin, 4),
            "lineage_panel_margin_over_competing_entity": round(entity_margin, 4),
            "lineage_panel_decision_basis": (
                "score_separation"
                if separated_by_score
                else "complete_program_dominance"
            ),
            "lineage_panel_entity_program_decision": program_decision,
            "lineage_panel_identity_program_decisive": (
                identity_program_decisive
            ),
            "lineage_panel_rationale": top_rationale,
            "lineage_panel_all": [
                {"name": e.panel_name, "score": round(e.score, 4)}
                for e in evidence[:_LINEAGE_PANEL_ALL_LIMIT]
            ],
        }
    )
    # Promotion gate: lineage_panel is a tie-breaker, not an override.
    # The whole point is to augment uncertainty — when the broad classifier
    # is confident, defer to it. Two gates must pass to promote:
    #
    #   (a) ``code`` must be in the broad top-5 candidates. Outside that
    #       set the panel can't promote (BRCA_LUMINAL firing on a UCEC
    #       sample because FOXA1+GATA3 overlap is the classic failure;
    #       it cost ~12% consolidated top-1 on TCGA-160 in early testing).
    #   (b) One of:
    #       * SAME-CODE REINFORCEMENT — ``code`` equals the broad top-1.
    #         The cancer call doesn't change; the panel's contribution
    #         is the subtype/program detail recorded in ``details``
    #         (rendered via the brief.py subtype-signal line).
    #       * BROAD EXPLICITLY UNCERTAIN — ``fit_quality.label`` is "weak"
    #         or "ambiguous". The only condition under which a panel
    #         may CHANGE the cancer code: broad has told us it doesn't
    #         trust its own call.
    #
    #   Two earlier iterations were tried and rejected:
    #     - Top-1/top-2 support ratio < 1.30. Fired on sibling-subtype
    #       draws (SKCM vs UVM, both melanocytic) where the family IS
    #       settled — it just enabled cross-family overrides (SKCM →
    #       BRCA) the user explicitly didn't want.
    #     - Same-family (registry ``family`` column matches). Within a
    #       family, panels can still be too similar — STAD and PAAD are
    #       both ``carcinoma-gi``, and the PAAD panel scored high on a
    #       broad-confident STAD sample (TCGA-D7-6524-01), flipping the
    #       call to PAAD. Family-only check is not specific enough.
    #
    # If neither gate passes, the selector still records the panel
    # evidence in details so reports / downstream confidence can surface
    # the rationale — just no report-label promotion.
    trace = analysis.get("candidate_trace") or []
    broad_top_codes = [str(r.get("code") or "") for r in trace[:5]]
    in_broad_top = code in broad_top_codes

    fit_label = ""
    fit_quality = analysis.get("fit_quality") or {}
    if isinstance(fit_quality, Mapping):
        fit_label = str(fit_quality.get("label") or "").strip().lower()
    broad_uncertain = fit_label in {"weak", "ambiguous"}

    broad_top_code = _clean(trace[0].get("code")) if trace else ""
    same_code = bool(broad_top_code and broad_top_code == code)

    marker_coherence = _marker_coherence(code, sample_tpm_by_symbol)
    broad_top_marker_coherence = (
        _marker_coherence(broad_top_code, sample_tpm_by_symbol)
        if broad_top_code
        else {}
    )
    broad_top_unexpected_low_conflict = (
        _local_reference_unexpected_low_lineage_conflict(
            broad_top_code,
            broad_top_marker_coherence,
            analysis,
            min_support=0.60,
        )
        if broad_top_marker_coherence
        else {}
    )
    broad_top_marker_contradicted = bool(
        broad_top_marker_coherence
        and (
            not _marker_coherence_selection_grade(broad_top_marker_coherence)
            or broad_top_unexpected_low_conflict
        )
    )
    concentration_warning = _analysis_has_expression_concentration_warning(analysis)
    lineage_conflict_warning = _analysis_has_expression_lineage_conflict(analysis)
    out_of_beam_rescue = bool(
        not in_broad_top
        and (concentration_warning or lineage_conflict_warning)
        and top_score >= _LINEAGE_PANEL_OUT_OF_BEAM_MIN_SCORE
        and margin >= _LINEAGE_PANEL_OUT_OF_BEAM_MIN_MARGIN
    )
    identity_program_rescue = bool(
        not in_broad_top and identity_program_decisive
    )

    can_promote = bool(
        (in_broad_top and (same_code or broad_uncertain))
        or out_of_beam_rescue
        or identity_program_rescue
    )
    blockers: list[str] = []
    if not in_broad_top and not (out_of_beam_rescue or identity_program_rescue):
        blockers.append(
            f"{code} is not among the top-5 first-pass RNA candidates; "
            "lineage panels only refine candidates the first-pass "
            "classifier already considered unless a very strong panel is "
            "paired with a technical expression-concentration warning or "
            "an expression/code lineage conflict"
        )
    elif not (out_of_beam_rescue or identity_program_rescue) and not (
        same_code or broad_uncertain
    ):
        blockers.append(
            f"first-pass top-1 ({broad_top_code or 'unknown'}) differs "
            f"from panel parent_cohort ({code}) and the first-pass "
            "classifier is confident — the lineage panel reading is "
            "noted but the first-pass call is preserved"
        )
    consensus_context = _broad_coarse_consensus_context(analysis)
    if (
        can_promote
        and not same_code
        and not identity_program_decisive
        and consensus_context
        and consensus_context != code
    ):
        blockers.append(
            "pan-cancer signature-ranker context and coarse reference matching both support "
            f"{consensus_context}; lineage panel {top_panel} is noted as a "
            "program signal, not a cross-site report-label override"
        )
        can_promote = False
    conflicting_coarse = (
        _strong_conflicting_coarse_reference(analysis, code)
        if can_promote and not same_code and not identity_program_decisive
        else {}
    )
    if conflicting_coarse:
        blockers.append(
            "lineage panel program conflicts with independent composition "
            f"reference {conflicting_coarse['code']} (rho "
            f"{conflicting_coarse['rho']:.2f}; primary tissue "
            f"{conflicting_coarse.get('primary_tissue') or 'unknown'} "
            f"rho {conflicting_coarse.get('primary_tissue_score', 0.0):.2f}); "
            "recorded as a program signal rather than a cross-site report-label "
            "override"
        )
        can_promote = False
    # The lineage panel is already a complete positive/negative program at
    # the resolution it names. Do not veto a coherent descendant program with
    # the generic parent marker set: basal BRCA, for example, is expected to
    # lack luminal markers that make up much of the broad BRCA sanity panel.
    # Keep broad parent coherence as audit context, not a second, correlated
    # gate over the more specific program.
    if marker_coherence:
        hypothesis.details["lineage_panel_parent_marker_coherence"] = (
            marker_coherence
        )
    if out_of_beam_rescue:
        hypothesis.details["lineage_panel_out_of_beam_rescue"] = {
            "score": round(top_score, 4),
            "margin_over_second": round(margin, 4),
            "expression_concentration_warning": bool(concentration_warning),
            "expression_lineage_conflict": bool(lineage_conflict_warning),
            "broad_top_marker_contradicted": bool(broad_top_marker_contradicted),
            "broad_top_marker_coherence": broad_top_marker_coherence,
            "broad_top_unexpected_low_conflict": broad_top_unexpected_low_conflict,
            "panel_marker_coherence": top_rationale,
            "generic_marker_coherence": marker_coherence,
        }
    if identity_program_rescue:
        hypothesis.details["lineage_panel_identity_program_rescue"] = {
            "program_decision": dict(program_decision),
            "identity_marker_groups": [
                list(group) for group in panel.identity_marker_groups
            ],
            "identity_marker_hits": list(
                (summary.get("panels") or [{}])[0].get(
                    "identity_marker_hits",
                    (),
                )
            ),
            "broad_top_code": broad_top_code,
            "role": "specific_tumor_identity_over_shared_morphology",
        }
    # Class-rank policy:
    #   - SAME-CODE REINFORCEMENT (panel agrees with broad top-1) →
    #     class 2. This lets the panel DEFEND a correct broad call
    #     against an aggressive local_expression_reference promotion
    #     to a different code (the LIHC→HEPB / PCPG→NBL pattern
    #     observed on TCGA-160). Same-code reinforcement is safe to
    #     elevate because it cannot change the cancer code — it only
    #     blocks other selectors from changing it.
    #   - CROSS-CODE PROMOTION (panel proposes a different code via
    #     fit_quality=weak/ambiguous) → class 1. The panel still
    #     competes with rare_marker and primary_expression_match, but
    #     fine_reference and local_expression_reference (which use
    #     class 2) still win when they have stronger support.
    class_rank = 4 if identity_program_decisive else (2 if same_code else 1)
    hypothesis.admit_adjudication_support(
        _ENTITY_CONSENSUS_MARKER_AXIS,
        top_score,
        selector="lineage_panel",
        blocking_reasons=blockers,
    )
    hypothesis.consider_for_report_label(
        selected_by="lineage_panel",
        can_select=can_promote,
        blocking_reasons=blockers,
        priority=(class_rank, top_score),
    )
    summary["promotion"] = {
        "promoted": bool(can_promote),
        "code": code,
        "blockers": list(blockers),
    }
    return summary


def _add_rare_marker_features(
    hypotheses: dict[str, CancerTypeEvidence],
    finding: Mapping[str, Any],
    analysis: Mapping[str, Any],
    sample_tpm_by_symbol: Mapping[str, float],
) -> None:
    code = _clean(finding.get("cancer_type"))
    rule_id = _clean(finding.get("rule_id"))
    if not code or not rule_id:
        return

    rule = _rare_rules_by_id().get(rule_id, {})
    support_pass, support_count, min_support, required_count = (
        _rare_marker_support_status(finding, rule)
    )
    if finding.get("exclusion_genes_observed") or not support_pass:
        return
    rule_promotes = _safe_bool(rule.get("promote_report_scope"), default=True)
    context_codes = set(_split_semicolon(rule.get("context_codes")))
    support_by_code = _context_support_by_code(analysis)
    top = _top_code(analysis)
    hypothesis = _hypothesis(hypotheses, code)
    has_local_exact_support = (
        "local_expression_reference" in hypothesis.evidence_sources
        and hypothesis.fine_reference_support >= _LOCAL_REFERENCE_MIN_SUPPORT
    )
    has_high_confidence_local_exact_support = bool(
        has_local_exact_support
        and hypothesis.details.get("local_reference_high_confidence")
    )
    adcc_basal_breast_conflict = bool(
        hypothesis.details.get("adcc_low_myb_basal_breast_conflict")
    )
    adcc_breast_program_conflict = bool(
        hypothesis.details.get("adcc_breast_program_conflict")
    )
    fusion_defined_context_conflict = (
        hypothesis.details.get("fusion_defined_context_conflict_details")
        if hypothesis.details.get("fusion_defined_context_conflict")
        else {}
    )
    cross_lineage_marker_conflict = (
        hypothesis.details.get("local_reference_cross_lineage_marker_conflict_details")
        if hypothesis.details.get("local_reference_cross_lineage_marker_conflict")
        else {}
    )
    has_direct_fusion = hypothesis.direct_fusion_support > 0
    registry_row = _registry_by_code().get(code, {})
    classification_target = _safe_bool(
        registry_row.get("is_classification_target"),
        default=True,
    )
    fusion_defined_code = _clean(registry_row.get("fusion_driven")).lower() == "defining"
    context_support = max(
        (support_by_code.get(context_code, 0.0) for context_code in context_codes),
        default=0.0,
    )
    top_is_context = bool(top and top in context_codes)
    marker_support = _rare_marker_expression_support(
        finding,
        support_count=support_count,
        min_support=min_support,
        required_count=required_count,
    )

    policy = _rare_rna_policy(code)
    effective_context = policy.effective_context_support(
        top_is_context=top_is_context,
        related_context_support=context_support,
    )
    marker_context_support = (
        policy.marker_weight * marker_support
        + policy.context_weight * effective_context
        + policy.top_context_weight * float(top_is_context)
    )
    top_lineage_passes = policy.top_lineage_passes(top_code=top)
    context_passes = top_lineage_passes and policy.context_passes(
        top_code=top,
        top_is_context=top_is_context,
        related_context_support=context_support,
    )
    blockers: list[str] = []
    if not rule_promotes:
        blockers.append("rule is a diagnostic prompt only")
    if rule_promotes and not classification_target and not has_direct_fusion:
        blockers.append(
            f"{code} is not a registry classification target; RNA-surrogate "
            "evidence remains a diagnostic prompt unless direct molecular "
            "evidence establishes the entity"
        )
    if (
        code == "ADCC"
        and rule_promotes
        and not has_direct_fusion
        and adcc_breast_program_conflict
    ):
        details = hypothesis.details.get("adcc_breast_program_conflict_details") or {}
        program = _clean(details.get("program")) or "BRCA"
        blockers.append(
            "no-fusion ADCC RNA surrogate conflicts with a strong "
            f"{program} lineage program in BRCA-compatible expression context"
        )
    if (
        fusion_defined_code
        and rule_promotes
        and not has_direct_fusion
        and not fusion_defined_context_conflict
    ):
        primary_tissue = _clean(registry_row.get("primary_tissue"))
        fusion_defined_context_conflict = (
            _unconfirmed_fusion_defined_context_conflict(
                code=code,
                context_codes=tuple(sorted(context_codes)),
                primary_tissue=primary_tissue,
                analysis=analysis,
                sample_tpm_by_symbol=sample_tpm_by_symbol,
            )
        )
    if (
        fusion_defined_code
        and rule_promotes
        and not has_direct_fusion
        and fusion_defined_context_conflict
    ):
        blockers.append(
            _fusion_defined_context_conflict_reason(
                fusion_defined_context_conflict
            )
        )
    if rule_promotes and not has_direct_fusion and not cross_lineage_marker_conflict:
        marker_coherence = _marker_coherence(code, sample_tpm_by_symbol)
        if marker_coherence:
            if top_is_context and top:
                active_context_codes = (top,)
            elif context_codes:
                best_context = max(
                    context_codes,
                    key=lambda context_code: support_by_code.get(context_code, 0.0),
                )
                active_context_codes = (best_context,)
            else:
                active_context_codes = ()
            cross_lineage_marker_conflict = _local_reference_cross_lineage_conflict(
                code,
                tuple(active_context_codes),
                marker_coherence,
            )
    strong_rna_surrogate = _strong_fusion_defined_rna_surrogate(
        code,
        sample_tpm_by_symbol,
    )
    if (
        rule_promotes
        and not has_direct_fusion
        and cross_lineage_marker_conflict
        and not bool(strong_rna_surrogate.get("strong"))
    ):
        blockers.append(
            "cross-lineage RNA-marker promotion requires a coherent "
            f"{code} marker program; observed "
            f"{cross_lineage_marker_conflict['detected']}/"
            f"{cross_lineage_marker_conflict['total']} expected high "
            f"markers ({cross_lineage_marker_conflict['marker_status']}) "
            "while the compatible RNA context is "
            f"{', '.join(cross_lineage_marker_conflict['context_codes'])}"
        )
    adcc_low_myb_marker = (
        code == "ADCC"
        and rule_promotes
        and _clean(finding.get("surrogate")).upper() == "MYB"
        and _safe_float(finding.get("surrogate_tpm")) < _ADCC_PROMOTING_MIN_MYB_TPM
    )
    if adcc_low_myb_marker and not adcc_breast_program_conflict:
        if adcc_basal_breast_conflict:
            blockers.append(
                "low-MYB ADCC RNA surrogate conflicts with a strong "
                "BRCA_BASAL lineage program in BRCA-compatible expression context"
            )
        elif not has_high_confidence_local_exact_support:
            blockers.append(
                "ADCC MYB promoter signal "
                f"{_safe_float(finding.get('surrogate_tpm')):.1f} TPM is below "
                f"{_ADCC_PROMOTING_MIN_MYB_TPM:.1f} TPM"
            )
    if (
        code == "ADCC"
        and rule_promotes
        and not has_direct_fusion
        and support_count < required_count
        and (
            not has_high_confidence_local_exact_support
            or adcc_basal_breast_conflict
            or adcc_breast_program_conflict
        )
    ):
        blockers.append(
            "ADCC RNA promotion requires a complete MYB-axis co-marker set "
            "or high-confidence ADCC expression-reference support"
        )
    if (
        code == "ADCC"
        and rule_promotes
        and not has_direct_fusion
        and not has_high_confidence_local_exact_support
    ):
        adcc_marker_coherence = hypothesis.details.get("local_reference_marker_coherence")
        if not isinstance(adcc_marker_coherence, Mapping):
            adcc_marker_coherence = _marker_coherence(code, sample_tpm_by_symbol)
        unexpected_low_genes = list(
            (adcc_marker_coherence or {}).get("unexpected_low_genes") or []
        )
        if unexpected_low_genes:
            blockers.append(
                "ADCC RNA promotion without fusion or high-confidence exact "
                "expression-reference support requires a clean marker program; "
                "expected-low markers are present "
                f"({', '.join(_clean(gene) for gene in unexpected_low_genes[:6])})"
            )
    if not top_lineage_passes:
        allowed = ", ".join(sorted(policy.required_top_lineages))
        observed = _broad_lineage_for_code(top) or "unknown"
        blockers.append(
            f"top expression-reference lineage {observed} is not one of {allowed}"
        )
    if top_lineage_passes and not context_passes:
        if context_codes:
            allowed = ", ".join(sorted(context_codes))
            blockers.append(f"expression-reference context is not one of {allowed}")
        else:
            blockers.append("expression-reference context does not support this marker")
    if marker_context_support < policy.min_marker_context_support:
        blockers.append(
            "marker/context support "
            f"{marker_context_support:.2f} is below "
            f"{policy.min_marker_context_support:.2f}"
        )
    mtc_neural_crest_conflict: dict[str, Any] = {}
    if code == "MTC" and rule_promotes and not has_direct_fusion:
        mtc_neural_crest_conflict = _mtc_neural_crest_conflict(
            top_code=top,
            support_by_code=support_by_code,
            sample_tpm_by_symbol=sample_tpm_by_symbol,
        )
        if mtc_neural_crest_conflict:
            blockers.append(
                "MTC CALCA-axis RNA prompt is not specific enough to override "
                "a PCPG/neural-crest expression context without CEACAM5/CALCR "
                "or high RET anchor support"
            )

    hypothesis.add_source("rare_marker")
    hypothesis.expression_reference_cancer_type = (
        hypothesis.expression_reference_cancer_type or top
    )
    hypothesis.reference_cancer_type = hypothesis.reference_cancer_type or top
    # Use comma delimiter to match the local-reference path at line 1138 —
    # the field is a multi-context list and downstream consumers should not
    # need to handle two different separators based on which selector fired.
    hypothesis.related_context_code = (
        ",".join(sorted(context_codes)) if context_codes else hypothesis.related_context_code
    )
    hypothesis.related_context_support = max(
        hypothesis.related_context_support,
        context_support,
    )
    hypothesis.related_context_is_top = hypothesis.related_context_is_top or top_is_context
    hypothesis.rna_marker_support = max(hypothesis.rna_marker_support, marker_support)
    hypothesis.basis = hypothesis.basis or finding.get("basis") or ""
    hypothesis.confirmatory_tests = (
        hypothesis.confirmatory_tests or finding.get("confirmatory_tests") or ""
    )
    hypothesis.caveat = hypothesis.caveat or finding.get("caveat") or ""
    hypothesis.source = hypothesis.source or finding.get("source") or ""
    hypothesis.details.update(
        {
            "rule_id": rule_id,
            "rule_promotes_report_scope": rule_promotes,
            "registry_classification_target": classification_target,
            "surrogate": finding.get("surrogate"),
            "surrogate_tpm": finding.get("surrogate_tpm"),
            "threshold_tpm": finding.get("threshold_tpm"),
            "support_genes": list(finding.get("support_genes") or []),
            "missing_support_genes": list(finding.get("missing_support_genes") or []),
            "support_gene_count": support_count,
            "min_support_genes": min_support,
            "required_support_gene_count": required_count,
            "support_pass": support_pass,
            "context_codes": sorted(context_codes),
            "context_support_fraction_of_top": round(float(context_support), 4),
            "top_reference_cancer_type": top,
            "top_is_context": top_is_context,
            "marker_context_support": round(float(marker_context_support), 4),
            "rare_marker_can_select": not blockers,
            "rare_marker_blocking_reasons": list(blockers),
            "rare_marker_complete_context_gated_axis": False,
            "strong_fusion_defined_rna_surrogate": (
                dict(strong_rna_surrogate) if strong_rna_surrogate else {}
            ),
            "fusion_defined_context_conflict": bool(
                fusion_defined_context_conflict
            ),
            "rare_marker_cross_lineage_marker_conflict": bool(
                cross_lineage_marker_conflict
            ),
            "mtc_neural_crest_context_conflict": bool(
                mtc_neural_crest_conflict
            ),
        }
    )
    if fusion_defined_context_conflict:
        hypothesis.details["fusion_defined_context_conflict_details"] = (
            dict(fusion_defined_context_conflict)
        )
    if cross_lineage_marker_conflict:
        hypothesis.details["rare_marker_cross_lineage_marker_conflict_details"] = (
            dict(cross_lineage_marker_conflict)
        )
    if mtc_neural_crest_conflict:
        hypothesis.details["mtc_neural_crest_context_conflict_details"] = (
            dict(mtc_neural_crest_conflict)
        )
    priority_class = 1
    priority_strength = marker_context_support
    complete_context_gated_marker_axis = bool(
        rule_promotes
        and not has_local_exact_support
        and required_count > 0
        and support_count >= required_count
        and marker_support >= 0.95
        and (top_is_context or context_support >= 0.90)
    )
    hypothesis.details["rare_marker_complete_context_gated_axis"] = (
        complete_context_gated_marker_axis
    )
    if (
        complete_context_gated_marker_axis
        and fusion_defined_code
        and bool(strong_rna_surrogate.get("strong"))
    ):
        priority_class = 3
        priority_strength = max(marker_context_support, marker_support)
    if fusion_defined_code and has_local_exact_support:
        priority_class = max(priority_class, 2)
        priority_strength = max(
            marker_context_support,
            0.5 * marker_context_support + 0.5 * hypothesis.fine_reference_support,
            priority_strength,
        )

    hypothesis.admit_adjudication_support(
        _ENTITY_CONSENSUS_MARKER_AXIS,
        marker_support,
        selector="rare_marker",
        blocking_reasons=blockers,
    )
    hypothesis.consider_for_report_label(
        selected_by="rare_marker",
        can_select=not blockers,
        blocking_reasons=blockers,
        priority=(priority_class, priority_strength),
    )


def _add_direct_fusion_features(
    hypotheses: dict[str, CancerTypeEvidence],
    fusion_scope_inference: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> None:
    code = _clean((fusion_scope_inference or {}).get("cancer_type"))
    if not code:
        return
    hypothesis = _hypothesis(hypotheses, code)
    hypothesis.add_source("direct_fusion")
    hypothesis.expression_reference_cancer_type = (
        hypothesis.expression_reference_cancer_type or _top_code(analysis)
    )
    hypothesis.reference_cancer_type = hypothesis.reference_cancer_type or _top_code(analysis)
    hypothesis.direct_fusion_support = 1.0
    hypothesis.basis = hypothesis.basis or fusion_scope_inference.get("basis") or ""
    hypothesis.confirmatory_tests = (
        hypothesis.confirmatory_tests
        or fusion_scope_inference.get("confirmatory_tests")
        or ""
    )
    hypothesis.caveat = hypothesis.caveat or fusion_scope_inference.get("caveat") or ""
    hypothesis.source = hypothesis.source or fusion_scope_inference.get("source") or ""
    hypothesis.details.update(
        {
            "fusion": fusion_scope_inference.get("fusion"),
            "expected_pair": fusion_scope_inference.get("expected_pair"),
        }
    )
    hypothesis.consider_for_report_label(
        selected_by="direct_fusion",
        can_select=True,
        blocking_reasons=(),
        priority=(3, 1.0),
    )


def _public_evidence_sort_key(evidence: CancerTypeEvidence):
    """Sort key for the public evidence list — highest-priority first,
    *ascending* cancer_type for deterministic tie-break.

    The negation pattern lets the caller use plain ``sorted(...)``
    (ascending) so the alphabetical-tie field stays in natural
    ascending order. Mirrors ``_pick_selected`` in
    ``select_report_scope_from_evidence`` — both must agree so the
    "selected" hypothesis matches the top of the evidence list on
    ties.
    """
    class_rank, strength, tiebreak = evidence.selection_priority
    return (
        -(1 if evidence.can_select_report_label else 0),
        -(1 if evidence.report_label_candidate else 0),
        -class_rank,
        -strength,
        -tiebreak,
        evidence.cancer_type,
    )


def _post_label_context_channels(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Channels that are important context but must not select the diagnosis here."""
    channels: list[CancerTypeEvidenceChannel] = []
    decomp = analysis.get("decomposition_results") or []
    if decomp:
        best = decomp[0]
        channels.append(
            CancerTypeEvidenceChannel(
                channel="deconvolution",
                stage="coarse_type",
                role="post_label_mixture_context",
                code=str(getattr(best, "cancer_type", "") or ""),
                context_code=str(getattr(best, "template", "") or ""),
                support=_safe_float(getattr(best, "score", 0.0)),
                status="not_used_for_report_label",
                details={
                    "purity": round(_safe_float(getattr(best, "purity", 0.0)), 4),
                },
            )
        )
    else:
        channels.append(
            CancerTypeEvidenceChannel(
                channel="deconvolution",
                stage="coarse_type",
                role="post_label_mixture_context",
                status="not_available_pre_label_selection",
            )
        )
    channels.append(
        CancerTypeEvidenceChannel(
            channel="therapy_context",
            stage="exact_subtype",
            role="downstream_compatibility_check",
            status="downstream_consumer_not_selector",
        )
    )
    # Lineage-routed decomposition + purity signals, surfaced alongside the cancer-type evidence so all
    # axes can be reasoned about together (mode / lineage_fit residual / aneuploidy / ESTIMATE gating /
    # purity reconciliation). Context only — never selects the report label.
    purity = analysis.get("purity") or {}
    comps = purity.get("components") or {}
    dc = comps.get("decomposition") if isinstance(comps.get("decomposition"), Mapping) else {}
    if dc and dc.get("mode"):
        channels.append(
            CancerTypeEvidenceChannel(
                channel="lineage_decomposition",
                stage="family",
                role="post_label_lineage_purity_context",
                code=str(dc.get("mode") or ""),
                context_code=str(comps.get("lineage_compartment") or ""),
                support=_safe_float(dc.get("residual_fraction") or 0.0),
                status="not_used_for_report_label",
                details={
                    "residual_fraction": dc.get("residual_fraction"),
                    "aneuploidy_purity": dc.get("aneuploidy_purity"),
                    "expression_lineage": dc.get("expression_lineage"),
                    "lineage_conflict": dc.get("lineage_conflict"),
                    "estimate_gated_for_lineage": comps.get("estimate_gated_for_lineage"),
                    "purity_consistency": purity.get("purity_consistency") or [],
                },
            )
        )
    return [channel.public_dict() for channel in channels]


def _build_staged_evidence_graph(
    rows: list[CancerTypeEvidence],
    selected: CancerTypeEvidence | None,
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the serializable lineage graph plus orthogonal subtype axes."""
    selected_code = selected.cancer_type if selected is not None else _top_code(analysis)
    selected_by = selected.selected_by if selected is not None else ""
    selected_stage = _decision_stage_for_hypothesis(selected)
    selected_family = _registry_family_for_code(selected_code)
    reference_code = (
        selected.reference_cancer_type
        if selected is not None and selected.reference_cancer_type
        else selected_code
    )
    lineage_path = _lineage_path_for_code(selected_code)
    selected_axes = _orthogonal_axes_for_hypothesis(selected)
    expression_reference = (
        selected.expression_reference_cancer_type
        if selected is not None and selected.expression_reference_cancer_type
        else reference_code
    )
    exact_selected = bool(selected is not None and selected_stage == "exact_subtype")

    channels: list[dict[str, Any]] = []
    global_learned_roles = {
        "hierarchical_compartment_vote",
        "hierarchical_family_vote",
        "hierarchical_entity_vote",
        "hierarchical_subtype_axis_vote",
    }
    seen_global_learned: set[tuple[str, str]] = set()
    seen_selected_mmr: set[tuple[str, str]] = set()
    for row in sorted(rows, key=_public_evidence_sort_key):
        for channel in _hypothesis_evidence_channels(row):
            channel = dict(channel)
            source = _clean(channel.get("channel"))
            role = _clean(channel.get("role"))
            if source == "learned_expression_classifier" and role in global_learned_roles:
                # These are one sample-level hierarchy vote, copied onto every
                # hypothesis only so candidate scoring can consume it. Emit
                # each vote once in the public graph and attribute it to the
                # label the model actually predicted.
                key = (role, _clean(channel.get("code")))
                if key in seen_global_learned:
                    continue
                seen_global_learned.add(key)
                channel["context_code"] = ""
                channel["candidate_code"] = _clean(channel.get("code"))
            elif (
                source == "learned_expression_classifier"
                and role == "hierarchical_mismatch_repair_vote"
            ):
                # MMR is meaningful only in the selected CRC/STAD/UCEC
                # context. Candidate rows elsewhere are counterfactual model
                # probes and belong in neither the report nor signal matrix.
                if row.cancer_type != selected_code:
                    continue
                details = channel.get("details") or {}
                label_space = (
                    _clean(details.get("label_space"))
                    if isinstance(details, Mapping)
                    else ""
                )
                key = (label_space, _clean(channel.get("code")))
                if key in seen_selected_mmr:
                    continue
                seen_selected_mmr.add(key)
                channel["candidate_code"] = selected_code
                channel["context_code"] = selected_code
            else:
                channel["candidate_code"] = row.cancer_type
            channels.append(channel)
    channels.extend(_post_label_context_channels(analysis))

    return {
        "schema_version": 1,
        "selection_order": ["family", "coarse_type", "exact_subtype"],
        "orthogonal_axis_order": [
            "mismatch_repair",
            "polymerase_epsilon",
            "copy_number_p53",
            "viral_status",
            "expression_subtype",
            "driver_mutation",
            "fusion_driver",
        ],
        "selected": {
            "code": selected_code,
            "family": selected_family,
            "coarse_type": reference_code,
            "expression_reference_cancer_type": expression_reference,
            "selected_by": selected_by,
            "selects_report_label": bool(
                selected is not None and selected.can_select_report_label
            ),
            "status": selected.label_status if selected is not None else "",
            "stage": selected_stage,
            "lineage_path": lineage_path,
            "orthogonal_axes": selected_axes,
        }
        if selected_code
        else None,
        "stages": [
            {
                "stage": "family",
                "status": "inferred" if selected_family else "not_resolved",
                "family": selected_family,
                "code": selected_code,
                "basis": "registry_family_or_broad_lineage",
            },
            {
                "stage": "coarse_type",
                "status": "selected" if reference_code else "not_resolved",
                "code": reference_code,
                "basis": _selection_method_label(selected_by)
                if selected_by
                else "primary_expression_context",
            },
            {
                "stage": "exact_subtype",
                "status": "selected" if exact_selected else "not_resolved",
                "code": selected_code if exact_selected else "",
                "basis": _selection_method_label(selected_by)
                if exact_selected
                else "",
            },
        ],
        "lineage_path": lineage_path,
        "orthogonal_axes": selected_axes,
        "channels": channels,
    }


def _same_lineage_subtype_pair(left: str, right: str) -> bool:
    """Return true for sibling/same-lineage labels that need marker support to swap."""

    left = str(left or "").strip()
    right = str(right or "").strip()
    if not left or not right or left == right:
        return False
    try:
        from .tumor_type_ontology import is_sarcoma_code, tumor_type_ontology_entry

        if is_sarcoma_code(left) and is_sarcoma_code(right):
            return True
        left_entry = tumor_type_ontology_entry(left)
        right_entry = tumor_type_ontology_entry(right)
    except Exception:  # noqa: BLE001
        return False
    if left_entry is None or right_entry is None:
        return False
    if left_entry.parent_code and left_entry.parent_code == right_entry.parent_code:
        return True
    if left in right_entry.ancestors or right in left_entry.ancestors:
        return True
    return False


def _marker_status_rank(status: str) -> int:
    return {"consistent": 3, "partial": 2, "mixed": 1, "weak": 0}.get(
        str(status or "").strip(),
        0,
    )


def _centroid_subtype_override_allowed(
    *,
    winner,
    best,
    win_rho: float,
    best_rho: float,
    sample_tpm_by_symbol: Mapping[str, float] | None,
) -> bool:
    """Gate same-lineage centroid subtype swaps with marker expectations.

    Whole-profile centroids are excellent broad-lineage corroborators, but
    within one lineage they can prefer a sibling cohort because of shared stroma,
    differentiation state, or batch effects. For those swaps, require the target
    subtype's positive/negative marker sanity check to corroborate the centroid,
    or demand a wider centroid margin.
    """

    if not _same_lineage_subtype_pair(winner.cancer_type, best.cancer_type):
        return True
    if best_rho - win_rho >= _CENTROID_SAME_LINEAGE_MARKER_MARGIN:
        return True
    if not sample_tpm_by_symbol:
        return False
    try:
        from .tumor_type_ontology import tumor_type_sanity_check

        best_sanity = tumor_type_sanity_check(best.cancer_type, sample_tpm_by_symbol)
        win_sanity = tumor_type_sanity_check(winner.cancer_type, sample_tpm_by_symbol)
    except Exception:  # noqa: BLE001
        return False

    best_rank = _marker_status_rank(str(best_sanity.get("status") or ""))
    win_rank = _marker_status_rank(str(win_sanity.get("status") or ""))
    best_fraction = float(best_sanity.get("expected_high_detected_fraction") or 0.0)
    win_fraction = float(win_sanity.get("expected_high_detected_fraction") or 0.0)
    if best_rank < _marker_status_rank("consistent"):
        return False
    if win_rank >= _marker_status_rank("partial") and best_fraction <= win_fraction + 0.10:
        return False
    return True


def _pick_selected(
    hypotheses,
    cen=None,
    compartment_confident=False,
    sample_tpm_by_symbol: Mapping[str, float] | None = None,
):
    """Select the report label from the selectable hypotheses (replaces the centroid veto, #98/#99).

    Authority first: among the selectable hypotheses the highest ``selection_priority`` (class_rank,
    then strength, then the source tiebreak; ascending ``cancer_type`` for determinism) wins, and a
    definitive molecular call (a detected fusion) outranks everything else.

    Then the whole-profile centroid CORROBORATES: when the compartment call is confident it re-ranks
    ONLY the already-selectable hypotheses, deferring to the one it best matches if that
    out-correlates the authority winner by ``_CENTROID_CORROBORATION_MARGIN``. This tips a marker tie
    toward the cohort the whole transcriptome actually looks like (NUT carcinoma over the salivary
    ADCC look-alike; osteosarcoma over the MDM2-amplicon liposarcoma) WITHOUT proposing a type no
    selector supported — so it can never promote a tissue contaminant the markers never backed. It
    replaces the centroid veto + its snapshot / conditional-undo + cross-lineage-flip guard.
    """
    sel = [row for row in hypotheses.values() if row.can_select_report_label]
    if not sel:
        return None

    def _authority_key(row):
        return (-row.selection_priority[0], -row.selection_priority[1],
                -row.selection_priority[2], row.cancer_type)

    # Definitive molecular evidence (a detected fusion) is never overridden by expression.
    definitive = [row for row in sel if row.selected_by in _DEFINITIVE_SELECTORS]
    if definitive:
        return min(definitive, key=_authority_key)

    # A complete, unopposed phenotype whose explicit tissue-identity groups
    # passed is more specific than whole-profile morphology. Preserve it over
    # fused/centroid similarity so a basal urothelial tumor is not relabeled as
    # another squamous cancer merely because shared keratins dominate the
    # library. Direct molecular evidence still takes precedence above.
    identity_programs = [
        row
        for row in sel
        if row.selected_by == "lineage_panel"
        and row.details.get("lineage_panel_identity_program_decisive")
    ]
    if identity_programs:
        return min(identity_programs, key=_authority_key)

    winner = min(sel, key=_authority_key)
    if winner.selected_by == "fused_evidence" or winner.details.get("fused_evidence_selected"):
        return winner
    if cen is None or not compartment_confident:
        return winner

    def _rho(code):
        try:
            return float(cen.get(code, float("nan")))
        except (TypeError, ValueError):
            return float("nan")

    win_rho = _rho(winner.cancer_type)
    best, best_rho = winner, win_rho
    for row in sel:
        rho = _rho(row.cancer_type)
        if rho == rho and (best_rho != best_rho or rho > best_rho):  # rho == rho skips NaN
            best, best_rho = row, rho
    # Defer to the best-correlated hypothesis only when BOTH it and the authority winner are
    # centroid-visible and the margin is clear — otherwise keep the authority winner.
    if (
        best is not winner
        and win_rho == win_rho
        and best_rho == best_rho
        and best_rho - win_rho >= _CENTROID_CORROBORATION_MARGIN
        and _centroid_subtype_override_allowed(
            winner=winner,
            best=best,
            win_rho=win_rho,
            best_rho=best_rho,
            sample_tpm_by_symbol=sample_tpm_by_symbol,
        )
    ):
        return best
    return winner


def _fallback_context_selected(
    hypotheses: dict[str, CancerTypeEvidence],
    analysis: Mapping[str, Any],
) -> CancerTypeEvidence | None:
    """Return a structured abstention or the best contextual RNA hypothesis.

    A fused-evidence blocker can carry a structured alternative.  Honor that
    adjudication instead of immediately reintroducing the blocked top ranker
    row as the fallback.  When the blocker names an ontology parent for
    abstention, that parent can select while the supporting child remains the
    expression-reference context.  Otherwise the returned row remains
    context-only.
    """

    top_code = _top_code(analysis)
    if not top_code:
        return None
    hypothesis = hypotheses.get(top_code)
    if hypothesis is None:
        return None
    if "pan_cancer_signature_ranker" not in hypothesis.evidence_sources:
        return None
    structured_abstention = hypothesis.details.get(
        "fused_evidence_structured_parent_abstention"
    )
    if isinstance(structured_abstention, Mapping):
        supporting_candidate = structured_abstention.get("supporting_candidate")
        supporting_code = (
            _clean(supporting_candidate.get("code"))
            if isinstance(supporting_candidate, Mapping)
            else ""
        )
        alternative = hypotheses.get(supporting_code)
        if (
            alternative is not None
            and "pan_cancer_signature_ranker" in alternative.evidence_sources
        ):
            abstention_code = _clean(structured_abstention.get("abstention_code"))
            abstention_is_parent = bool(
                abstention_code
                and abstention_code in _registry_by_code()
                and (
                    supporting_code == abstention_code
                    or _code_has_registry_ancestor(
                        supporting_code,
                        abstention_code,
                    )
                )
            )
            if abstention_is_parent:
                abstention = _hypothesis(hypotheses, abstention_code)
                abstention.add_source("entity_evidence_consensus")
                abstention.expression_reference_cancer_type = supporting_code
                abstention.reference_cancer_type = supporting_code
                abstention.related_context_code = supporting_code
                abstention.related_context_support = (
                    alternative.broad_rna_support
                )
                abstention.broad_rna_support = alternative.broad_rna_support
                abstention.broad_rna_rank = alternative.broad_rna_rank
                abstention.details["fallback_context_adjudication"] = {
                    "mode": "structured_parent_abstention",
                    "blocked_top_code": top_code,
                    "supporting_child_code": supporting_code,
                    "abstention_code": abstention_code,
                    "conflict": dict(structured_abstention),
                }
                abstention.details["entity_consensus_adjudication_mode"] = (
                    "structured_parent_abstention"
                )
                abstention.basis = (
                    f"{top_code} was vetoed by convergent {abstention_code} "
                    f"family evidence; report scope abstained to "
                    f"{abstention_code} while retaining {supporting_code} as the "
                    "expression-reference context"
                )
                abstention.consider_for_report_label(
                    selected_by="entity_evidence_consensus",
                    can_select=True,
                    blocking_reasons=(),
                    priority=(4, 1.0 + alternative.broad_rna_support),
                )
                return abstention
    # The ranker already exposes calibrated ordinal tiers.  If several sibling
    # leaves occupy the same leading tier and no selector could admit one of
    # them, the evidence supports their parent, not the alphabetically or
    # numerically first leaf.  This is an ontology abstention, not a new score
    # threshold, and prevents false leaf precision such as a three-way tied
    # liposarcoma result becoming DDLPS by fallback alone.
    top_tier = hypothesis.details.get("support_rank_tier")
    parent_chain = _registry_parent_chain(top_code)
    parent_code = parent_chain[0] if parent_chain else ""
    if top_tier is not None and parent_code:
        tied_siblings = [
            row
            for row in hypotheses.values()
            if row.cancer_type != top_code
            and "pan_cancer_signature_ranker" in row.evidence_sources
            and (_registry_parent_chain(row.cancer_type) or ("",))[0]
            == parent_code
            and row.details.get("support_rank_tier") == top_tier
        ]
        parent_row = _registry_row_for_code(parent_code) or {}
        if tied_siblings and _safe_bool(
            parent_row.get("is_classification_target"),
            default=True,
        ):
            abstention = _hypothesis(hypotheses, parent_code)
            abstention.add_source("entity_evidence_consensus")
            abstention.expression_reference_cancer_type = parent_code
            abstention.reference_cancer_type = parent_code
            abstention.broad_rna_support = max(
                [hypothesis.broad_rna_support]
                + [row.broad_rna_support for row in tied_siblings]
            )
            abstention.broad_rna_rank = hypothesis.broad_rna_rank
            tied_codes = [top_code, *(row.cancer_type for row in tied_siblings)]
            abstention.details["fallback_context_adjudication"] = {
                "mode": "tied_sibling_parent_abstention",
                "abstention_code": parent_code,
                "support_rank_tier": top_tier,
                "tied_sibling_codes": tied_codes,
            }
            abstention.details["entity_consensus_adjudication_mode"] = (
                "tied_sibling_parent_abstention"
            )
            abstention.basis = (
                f"no tumor-identity selector resolved the leading tied siblings "
                f"{', '.join(tied_codes)}; report scope abstained to their "
                f"shared parent {parent_code}"
            )
            abstention.consider_for_report_label(
                selected_by="entity_evidence_consensus",
                can_select=True,
                blocking_reasons=(),
                priority=(4, 1.0 + abstention.broad_rna_support),
            )
            if abstention.can_select_report_label:
                return abstention
    hypothesis.report_label_candidate = True
    # This is a BLOCKED fallback context row (no label was selectable) admitted via the
    # pan-cancer signature ranker. Force the selector to the ranker rather than
    # ``or``-preserving whatever is there: a hypothesis that was initially selectable and
    # later demoted by ``_add_fused_evidence_features`` can still carry a stale selectable
    # ``selected_by`` (e.g. ``local_expression_reference``) while ``can_select_report_label``
    # is False. ``_apply_cancer_type_evidence`` treats any non-ranker ``selected_by`` as a
    # report-scope selection, so preserving the stale value would let this blocked hypothesis
    # drive the final report label.
    hypothesis.selected_by = "pan_cancer_signature_ranker"
    hypothesis.label_basis = "pan_cancer_signature_ranker"
    hypothesis.label_status = hypothesis.label_status or "blocked"
    if not hypothesis.blocking_reasons:
        hypothesis.blocking_reasons = (
            "pan-cancer signature-ranker evidence is candidate/context only; "
            "fused evidence must supply an independent admission path before "
            "it can select the report label",
        )
    return hypothesis


def _centroid_and_confidence(sample_tpm_by_symbol: Mapping[str, float]):
    """``(centroid_correlations Series, compartment-confident bool)`` for selection, or ``(None, False)``.

    One whole-profile centroid pass, used by ``_pick_selected`` to corroborate the selectable
    hypotheses. Fail-open (``(None, False)``) so selection simply falls back to authority priority
    on any error.
    """
    if not sample_tpm_by_symbol:
        return None, False
    try:
        from .cancer_type_centroid import centroid_correlations, compartment_call

        cen = centroid_correlations(sample_tpm_by_symbol)
        comp = compartment_call(sample_tpm_by_symbol, _corr=cen)
        return cen, bool(comp.get("confident"))
    except Exception:  # noqa: BLE001
        return None, False


def _centroid_supports_for_hypotheses(
    hypotheses: Mapping[str, CancerTypeEvidence],
    cen,
) -> dict[str, float]:
    if cen is None or not len(cen):
        return {}
    raw: dict[str, float] = {}
    for code, hypothesis in hypotheses.items():
        centroid_codes = [
            code,
            hypothesis.reference_cancer_type,
            hypothesis.expression_reference_cancer_type,
            hypothesis.related_context_code,
            hypothesis.details.get("winning_subtype"),
            hypothesis.details.get("winning_subtype_from_candidate_trace"),
        ]
        # Aggregate registry entities (for example CRC) may not have their own
        # centroid because their observed references live at child cohorts
        # (COAD and READ). Roll those loadable descendants up for the entity
        # comparison while keeping the final label at the parent.
        centroid_codes.extend(
            centroid_code
            for centroid_code in getattr(cen, "index", ())
            if _code_has_registry_ancestor(_clean(centroid_code), code)
        )
        best_rho = float("nan")
        for centroid_code in centroid_codes:
            cleaned = _clean(centroid_code)
            if not cleaned or "," in cleaned:
                continue
            try:
                rho = float(cen.get(cleaned, float("nan")))
            except (TypeError, ValueError):
                rho = float("nan")
            if rho == rho and (best_rho != best_rho or rho > best_rho):
                best_rho = rho
        if best_rho == best_rho:
            raw[code] = best_rho
    if not raw:
        return {}
    top = max(raw.values())
    return {
        code: float(np.clip(1.0 - max(0.0, top - rho) / 0.08, 0.0, 1.0))
        for code, rho in raw.items()
    }


def _centroid_anchored_expression_reference_supported(
    hypothesis: CancerTypeEvidence,
    *,
    centroid_support: float,
) -> bool:
    """Return true when centroid + local markers can rescue a blocked reference.

    This is the fused-evidence replacement for the old "first-pass context must
    already be right" escape hatch. It is intentionally narrower than ordinary
    local-reference selection: the whole-profile centroid must make the
    expression reference a top match, and the local marker program must be
    present without relying on the pan-cancer signature-ranker context.
    """

    if "local_expression_reference" not in hypothesis.evidence_sources:
        return False
    if hypothesis.can_select_report_label and hypothesis.selected_by in {
        "fine_reference",
        "local_expression_reference",
    }:
        return False
    details = hypothesis.details
    if details.get("local_reference_explicit_negative_fusion"):
        return False
    expression_source = _clean(details.get("local_reference_expression_source"))
    child_expression_source = _clean(
        details.get("local_reference_child_expression_source")
    )
    if _is_molecular_status_expression_source(
        expression_source
    ) or _is_molecular_status_expression_source(child_expression_source):
        return False
    if (
        centroid_support
        < _FUSED_EVIDENCE_CENTROID_ANCHORED_REFERENCE_MIN_CENTROID
    ):
        return False
    if (
        hypothesis.fine_reference_support
        < _FUSED_EVIDENCE_CENTROID_ANCHORED_REFERENCE_MIN_SUPPORT
    ):
        return False
    marker_fraction = _safe_float(details.get("local_reference_marker_fraction"))
    burden_ratio = _safe_float(details.get("local_reference_marker_burden_ratio"))
    if marker_fraction < _FUSED_EVIDENCE_CENTROID_ANCHORED_REFERENCE_MIN_FRACTION:
        return False
    if burden_ratio < _FUSED_EVIDENCE_CENTROID_ANCHORED_REFERENCE_MIN_BURDEN:
        return False
    marker_coherence = details.get("local_reference_marker_coherence") or {}
    if not isinstance(marker_coherence, Mapping):
        return False
    if _marker_coherence_unexpected_low_count(marker_coherence) > 0:
        return False
    status = _clean(marker_coherence.get("status")).lower()
    detected = _safe_int(marker_coherence.get("detected"), 0)
    return bool(status in {"consistent", "partial"} and detected >= 3)


def _weak_non_crc_fused_call_conflicts_with_crc_family(
    hypothesis: CancerTypeEvidence,
    features: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the close CRC competitor context for a weak non-CRC epithelial call."""

    if _code_has_registry_ancestor(hypothesis.cancer_type, _CRC_REGISTRY_ROOT):
        return {}
    if _code_lineage_token(hypothesis.cancer_type) != "epithelial":
        return {}
    competitor = hypothesis.details.get("pan_cancer_signature_crc_family_competitor")
    if not isinstance(competitor, Mapping):
        return {}
    competitor_support = _safe_float(competitor.get("support_fraction_of_top"))
    if competitor_support <= 0:
        return {}
    if hypothesis.broad_rna_support > 0 and competitor_support < (
        0.90 * hypothesis.broad_rna_support
    ):
        return {}
    learned = _safe_float(features.get("learned_expression_probability"))
    learned_entity = _safe_float(features.get("learned_expression_entity_support"))
    learned_margin = _safe_float(features.get("learned_expression_margin"))
    learned_is_weak = bool(
        learned < _LEARNED_EXPRESSION_MIN_PROBABILITY
        or learned_entity < _LEARNED_EXPRESSION_MIN_PROBABILITY
        or learned_margin < _FUSED_EVIDENCE_STRONG_LEARNED_ENTITY_MARGIN
    )
    if not learned_is_weak:
        return {}
    marker_status = _clean(
        features.get("pan_cancer_signature_marker_status")
    ).lower()
    marker_unexpected_low = _safe_int(
        features.get("pan_cancer_signature_marker_unexpected_low_count"),
        0,
    )
    marker_mixed = bool(marker_unexpected_low > 0 or "mixed" in marker_status)
    if not marker_mixed:
        return {}
    return {
        "candidate_code": hypothesis.cancer_type,
        "candidate_support_fraction_of_top": round(
            float(hypothesis.broad_rna_support),
            4,
        ),
        "candidate_learned_probability": round(float(learned), 4),
        "candidate_learned_entity_support": round(float(learned_entity), 4),
        "candidate_learned_margin": round(float(learned_margin), 4),
        "candidate_marker_status": marker_status,
        "candidate_unexpected_low_marker_count": int(marker_unexpected_low),
        "supporting_candidate": dict(competitor),
        "abstention_code": _CRC_REGISTRY_ROOT,
    }


def _primary_context_root_for_code(code: str) -> str:
    """Return the diagnosis root used to compare primary-context hypotheses."""

    code_text = _clean(code)
    if not code_text:
        return ""
    chain = _registry_parent_chain(code_text)
    return chain[-1] if chain else code_text


def _ranker_context_row_is_primary_like(
    row: Mapping[str, Any],
    *,
    candidate_code: str,
) -> bool:
    code = _clean(row.get("code"))
    if not code:
        return False
    if _is_background_like_candidate(row):
        return False
    row_lineage = _code_lineage_token(code)
    candidate_lineage = _code_lineage_token(candidate_code)
    # Heme/immune top rows often reflect sample composition. Do not let them
    # block solid-tumor candidates unless the candidate is also heme-lineage.
    if (
        row_lineage == "hematolymphoid"
        and candidate_lineage
        and candidate_lineage != row_lineage
    ):
        return False
    return True


def _dominant_primary_context_competitor(
    hypothesis: CancerTypeEvidence,
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a stronger ranker primary context that conflicts with a fused call."""

    candidate_code = hypothesis.cancer_type
    candidate_context = _primary_context_root_for_code(candidate_code)
    if not candidate_context:
        return {}
    candidate_support = _safe_float(hypothesis.broad_rna_support)
    context_support = candidate_support
    for rank, row in enumerate(_candidate_rows(analysis), start=1):
        code = _clean(row.get("code"))
        if not code:
            continue
        if _primary_context_root_for_code(code) != candidate_context:
            continue
        support = _safe_float(row.get("support_fraction_of_top"))
        if support <= 0 and rank == 1:
            support = 1.0
        context_support = max(context_support, support)
    best: dict[str, Any] = {}
    best_priority = 0.0
    for rank, row in enumerate(_candidate_rows(analysis), start=1):
        code = _clean(row.get("code"))
        if not code or not _ranker_context_row_is_primary_like(
            row,
            candidate_code=candidate_code,
        ):
            continue
        support = _safe_float(row.get("support_fraction_of_top"))
        if support <= 0 and rank == 1:
            support = 1.0
        if support < _PRIMARY_CONTEXT_DOMINANCE_MIN_SUPPORT:
            continue
        context = _primary_context_root_for_code(code)
        if not context or context == candidate_context:
            continue
        if (
            context_support > 0
            and context_support
            >= _PRIMARY_CONTEXT_DOMINANCE_MIN_RATIO * support
        ):
            continue
        priority = support - 0.01 * rank
        if not best or priority > best_priority:
            best = {
                "code": code,
                "context": context,
                "rank": rank,
                "support_fraction_of_top": round(float(support), 4),
                "candidate_code": candidate_code,
                "candidate_context": candidate_context,
                "candidate_support_fraction_of_top": round(
                    float(candidate_support),
                    4,
                ),
                "candidate_context_support_fraction_of_top": round(
                    float(context_support),
                    4,
                ),
                "min_support": _PRIMARY_CONTEXT_DOMINANCE_MIN_SUPPORT,
                "min_ratio": _PRIMARY_CONTEXT_DOMINANCE_MIN_RATIO,
            }
            best_priority = float(priority)
    return best


def _orthogonal_axes_that_block_report_label(code: str) -> list[dict[str, Any]]:
    axes = _orthogonal_axes_for_code(code)
    return [
        axis
        for axis in axes
        if _clean(axis.get("axis")) in _REPORT_LABEL_BLOCKING_ORTHOGONAL_AXES
        # Registry metadata can describe an etiologic property of a base
        # diagnosis itself (for example, HPV is defining for CESC). That is
        # report context, not a child status label to strip away. Orthogonal
        # axes block selection only when the registry supplies a distinct
        # diagnosis node that can carry the state.
        and bool(_clean(axis.get("parent_code")))
        and _clean(axis.get("base_code")) != _clean(axis.get("code"))
    ]


def _rare_marker_channel_admitted(hypothesis: CancerTypeEvidence) -> bool:
    """Whether rare-marker evidence can contribute as an admitted evidence channel."""

    return bool(
        hypothesis.can_select_report_label
        and hypothesis.rna_marker_support > 0
        and "rare_marker" in hypothesis.evidence_sources
        and hypothesis.details.get("rare_marker_can_select")
    )


def _fused_component_scores(
    hypothesis: CancerTypeEvidence,
    *,
    centroid_support: float,
) -> dict[str, float]:
    details = hypothesis.details
    features = _hypothesis_decision_features(
        hypothesis,
        centroid_support=centroid_support,
    )
    selected_by = hypothesis.selected_by if hypothesis.can_select_report_label else ""
    admissible_exact_reference = _safe_float(
        hypothesis.adjudication_axis_support(_ENTITY_CONSENSUS_REFERENCE_AXIS)
    )
    admissible_contrast = _safe_float(
        hypothesis.adjudication_axis_support(_ENTITY_CONSENSUS_MARKER_AXIS)
    )
    admissible_composition = _safe_float(
        hypothesis.adjudication_axis_support(_ENTITY_CONSENSUS_COMPOSITION_AXIS)
    )
    exact_reference_admitted = bool(
        admissible_exact_reference > 0
        and selected_by in {"fine_reference", "local_expression_reference"}
    )
    lineage_panel_admitted = selected_by == "lineage_panel"
    rare_marker_admitted = _rare_marker_channel_admitted(hypothesis)
    strong_rare_rna_surrogate = bool(
        (details.get("strong_fusion_defined_rna_surrogate") or {}).get("strong")
    )
    rare_marker_component_admitted = bool(
        rare_marker_admitted
        and (not exact_reference_admitted or strong_rare_rna_surrogate)
    )
    contrast_admitted = bool(
        admissible_contrast > 0
        and selected_by == "contrast_discriminator"
    )
    refinement_admitted = selected_by == "tumor_label_refinement"
    composition_admitted = selected_by == "coarse_composition_reference"
    lineage_panel = _safe_float(details.get("lineage_panel_score"))
    pan_signature_marker_support = _safe_float(
        features.get("pan_cancer_signature_marker_support")
    )
    pan_signature_marker_selectable = bool(
        features.get("pan_cancer_signature_marker_selectable")
    )
    signature_anchor_specificity = (
        details.get("local_reference_signature_anchor_specificity") or {}
    )
    signature_anchor_specific_fraction = _safe_float(
        signature_anchor_specificity.get("specific_fraction")
        if isinstance(signature_anchor_specificity, Mapping)
        else 0.0
    )
    signature_anchored_exact_reference = bool(
        exact_reference_admitted
        and details.get("local_reference_signature_anchored")
    )
    status_child_parent_reference = bool(
        details.get("local_reference_status_child_code")
    )
    local_marker_coherence = details.get("local_reference_marker_coherence") or {}
    marker_coherent_reference = bool(
        admissible_exact_reference > 0
        and
        "local_expression_reference" in hypothesis.evidence_sources
        and not exact_reference_admitted
        and isinstance(local_marker_coherence, Mapping)
        and _marker_coherence_unexpected_low_count(local_marker_coherence) == 0
        and str(local_marker_coherence.get("status") or "").strip().lower()
        == "consistent"
        and _safe_float(details.get("local_reference_marker_fraction"))
        >= _FUSED_EVIDENCE_MARKER_COHERENT_REFERENCE_MIN_FRACTION
        and _safe_float(details.get("local_reference_marker_burden_ratio"))
        >= _FUSED_EVIDENCE_MARKER_COHERENT_REFERENCE_MIN_BURDEN
    )
    centroid_anchored_reference = bool(
        admissible_exact_reference > 0
        and _centroid_anchored_expression_reference_supported(
            hypothesis,
            centroid_support=centroid_support,
        )
    )
    components = {
        "pan_cancer_signature_ranker": (
            _FUSED_EVIDENCE_PAN_SIGNATURE_WEIGHT * hypothesis.broad_rna_support
        ),
        "pan_cancer_signature_marker_program": (
            0.75 * pan_signature_marker_support
            if hypothesis.broad_rna_support >= _PAN_SIGNATURE_MARKER_PROGRAM_MIN_SUPPORT
            and pan_signature_marker_selectable
            else 0.0
        ),
        "centroid_spearman": _FUSED_EVIDENCE_CENTROID_WEIGHT * centroid_support,
        "learned_compartment_anchored_pan_cancer_context": (
            0.50
            + 0.30
            * min(
                1.0,
                _safe_float(features.get("learned_expression_compartment_support")),
            )
            + 0.20 * min(1.0, hypothesis.broad_rna_support)
            if features.get("learned_compartment_anchored_pan_cancer_context")
            else 0.0
        ),
        "learned_family_anchored_pan_cancer_context": (
            _FUSED_EVIDENCE_LEARNED_FAMILY_ANCHOR_BASE
            + _FUSED_EVIDENCE_LEARNED_FAMILY_ANCHOR_SUPPORT_WEIGHT
            * min(
                1.0,
                _safe_float(features.get("learned_expression_family_support")),
            )
            + _FUSED_EVIDENCE_LEARNED_FAMILY_ANCHOR_SIGNATURE_WEIGHT
            * min(1.0, hypothesis.broad_rna_support)
            if features.get("learned_family_anchored_pan_cancer_context")
            else 0.0
        ),
        "learned_expression_classifier": (
            _FUSED_EVIDENCE_LEARNED_WEIGHT * hypothesis.learned_expression_support
        ),
        "exact_expression_reference": (
            (
                0.35
                if status_child_parent_reference
                else _FUSED_EVIDENCE_EXACT_REFERENCE_WEIGHT
            )
            * admissible_exact_reference
            if exact_reference_admitted
            else 0.15 * admissible_exact_reference
        ),
        "signature_anchored_exact_reference": (
            _FUSED_EVIDENCE_EXACT_STAGE_BONUS
            + 0.35 * signature_anchor_specific_fraction
            + 0.25
            * _safe_float(details.get("local_reference_signature_anchor_support"))
            if signature_anchored_exact_reference
            else 0.0
        ),
        "marker_coherent_expression_reference": (
            0.30
            + 0.20 * min(1.0, _safe_float(details.get("local_reference_marker_fraction")))
            if marker_coherent_reference
            else 0.0
        ),
        "centroid_anchored_expression_reference": (
            0.55
            + 0.25 * min(1.0, _safe_float(details.get("local_reference_marker_fraction")))
            + 0.20 * min(1.0, _safe_float(details.get("local_reference_marker_burden_ratio")))
            if centroid_anchored_reference
            else 0.0
        ),
        "lineage_panel": (
            _FUSED_EVIDENCE_LINEAGE_PANEL_WEIGHT * lineage_panel
            if lineage_panel_admitted
            else 0.0
        ),
        "contrast_discriminator": (
            1.10 * admissible_contrast
            if contrast_admitted
            else 0.0
        ),
        "direct_fusion": 2.0 * hypothesis.direct_fusion_support,
        "rare_marker": (
            2.00 * hypothesis.rna_marker_support
            if rare_marker_component_admitted
            else 0.0
        ),
        "marker_program": (
            0.60 * max(hypothesis.family_marker_support, hypothesis.rna_marker_support)
            if refinement_admitted or rare_marker_component_admitted
            else 0.0
        ),
        "background_label": (
            1.00 * hypothesis.background_label_support if refinement_admitted else 0.0
        ),
        "coarse_composition_reference": (
            1.00 * admissible_composition
            if composition_admitted
            else 0.0
        ),
    }
    if (
        "pan_cancer_signature_subtype" in hypothesis.evidence_sources
        and hypothesis.label_status != "blocked"
    ):
        components["pan_cancer_signature_subtype"] = (
            _FUSED_EVIDENCE_EXACT_STAGE_BONUS
            + 0.25 * hypothesis.broad_rna_support
        )
    return {
        key: round(float(value), 4)
        for key, value in components.items()
        if _safe_float(value) > 0
    }


def _group_fused_evidence_components(
    components: Mapping[str, float],
) -> dict[str, float]:
    """Collapse correlated feature views into independent evidence groups.

    Flat, hierarchical, family-anchored, and compartment-anchored learned
    scores are views of one trained expression model, not four votes.
    Likewise, subtype/signature, reference, curated-marker, and composition
    variants each describe one biological evidence source. Keeping their raw
    components in the audit is useful, but summing them lets one source
    overwhelm several genuinely independent lines of evidence.
    """

    groups = {
        "learned_expression_model": (
            "learned_expression_classifier",
            "learned_compartment_anchored_pan_cancer_context",
            "learned_family_anchored_pan_cancer_context",
        ),
        "pan_cancer_signature": (
            "pan_cancer_signature_ranker",
            "pan_cancer_signature_subtype",
            "pan_cancer_signature_marker_program",
        ),
        "expression_reference": (
            "exact_expression_reference",
            "signature_anchored_exact_reference",
            "marker_coherent_expression_reference",
            "centroid_anchored_expression_reference",
        ),
        "curated_marker_program": (
            "lineage_panel",
            "contrast_discriminator",
            "rare_marker",
            "marker_program",
        ),
        "composition_context": (
            "background_label",
            "coarse_composition_reference",
        ),
    }
    grouped: dict[str, float] = {}
    grouped_keys = {key for keys in groups.values() for key in keys}
    for group, keys in groups.items():
        support = max(
            (_safe_float(components.get(key)) for key in keys),
            default=0.0,
        )
        if support > 0:
            grouped[group] = round(float(support), 4)
    for key, value in components.items():
        support = _safe_float(value)
        if key not in grouped_keys and support > 0:
            grouped[key] = round(float(support), 4)
    return grouped


def _fused_evidence_eligible(
    hypothesis: CancerTypeEvidence,
    analysis: Mapping[str, Any],
    *,
    score: float,
    centroid_support: float,
    components: Mapping[str, float],
) -> tuple[bool, list[str]]:
    if hypothesis.direct_fusion_support > 0:
        return True, []
    blockers: list[str] = []
    features = _hypothesis_decision_features(
        hypothesis,
        centroid_support=centroid_support,
    )
    learned = hypothesis.learned_expression_support
    learned_hierarchy = _learned_hierarchy_support(hypothesis.details)
    lineage_panel = _safe_float(hypothesis.details.get("lineage_panel_score"))
    admissible_exact_reference = _safe_float(
        hypothesis.adjudication_axis_support(_ENTITY_CONSENSUS_REFERENCE_AXIS)
    )
    admissible_contrast = _safe_float(
        hypothesis.adjudication_axis_support(_ENTITY_CONSENSUS_MARKER_AXIS)
    )
    raw_selected_by = _clean(hypothesis.selected_by)
    selected_by = raw_selected_by if hypothesis.can_select_report_label else ""
    learned_admitted = selected_by == "learned_expression_classifier"
    exact_reference_admitted = bool(
        admissible_exact_reference > 0
        and selected_by
        in {
            "fine_reference",
            "local_expression_reference",
        }
    )
    signature_anchor_specificity = (
        hypothesis.details.get("local_reference_signature_anchor_specificity") or {}
    )
    signature_anchor_specific_fraction = _safe_float(
        signature_anchor_specificity.get("specific_fraction")
        if isinstance(signature_anchor_specificity, Mapping)
        else 0.0
    )
    signature_anchored_exact_reference = bool(
        exact_reference_admitted
        and hypothesis.details.get("local_reference_signature_anchored")
        and signature_anchor_specific_fraction >= 0.75
    )
    learned_hierarchy_available = bool(
        hypothesis.details.get("learned_expression_hierarchy_votes")
    )
    learned_neighborhood_support_for_candidate = max(
        _safe_float(features.get("learned_expression_compartment_support")),
        _safe_float(features.get("learned_expression_family_support")),
        _safe_float(features.get("learned_expression_entity_support")),
    )
    signature_anchor_learned_contraindicated = bool(
        signature_anchored_exact_reference
        and learned_hierarchy_available
        and _safe_float(features.get("learned_expression_flat_lineage_support")) < 0.30
        and learned_neighborhood_support_for_candidate < 0.20
    )
    refinement_learned_contraindicated = bool(
        raw_selected_by == "tumor_label_refinement"
        and learned_hierarchy_available
        and _safe_float(features.get("learned_expression_flat_lineage_support")) < 0.30
        and learned_neighborhood_support_for_candidate < 0.20
        and _safe_int(
            features.get("pan_cancer_signature_marker_unexpected_low_count"),
            0,
        )
        > 0
    )
    signature_anchor_can_escape_expected_low = bool(
        signature_anchored_exact_reference
        and not signature_anchor_learned_contraindicated
    )
    strong_learned_entity_call = bool(features.get("learned_strong_entity_call"))
    strong_independent_learned_call = bool(
        strong_learned_entity_call
        or (
            learned_admitted
            and learned >= _LEARNED_EXPRESSION_STRONG_PROBABILITY
            and (
                learned_hierarchy >= _LEARNED_EXPRESSION_MIN_HIERARCHICAL_CONTEXT
                or centroid_support >= 0.85
            )
        )
    )
    rare_marker_admitted = _rare_marker_channel_admitted(hypothesis)
    crc_family_conflict = _weak_non_crc_fused_call_conflicts_with_crc_family(
        hypothesis,
        features,
    )
    if (
        crc_family_conflict
        and not strong_independent_learned_call
        and not exact_reference_admitted
        and selected_by
        not in {
            "lineage_panel",
            "rare_marker",
            "contrast_discriminator",
            "tumor_label_refinement",
            "coarse_composition_reference",
        }
    ):
        supporting_candidate = (
            crc_family_conflict.get("supporting_candidate") or {}
        )
        blockers.append(
            "weak non-CRC epithelial fused evidence is competing with close "
            f"CRC-family RNA support ({supporting_candidate.get('code', 'CRC')} "
            f"rank {supporting_candidate.get('rank', '?')}, "
            f"{_safe_float(supporting_candidate.get('support_fraction_of_top')):.2f}x top; "
            f"family support {_safe_float(supporting_candidate.get('family_support')):.2f}); "
            "require stronger learned, exact-reference, marker, lineage-panel, "
            "fusion, or contrast support before selecting the non-CRC label"
        )
        hypothesis.details["fused_evidence_structured_parent_abstention"] = (
            crc_family_conflict
        )
    orthogonal_blocking_axes = _orthogonal_axes_that_block_report_label(
        hypothesis.cancer_type
    )
    if orthogonal_blocking_axes and not (
        hypothesis.direct_fusion_support > 0 or rare_marker_admitted
    ):
        axes = ", ".join(
            sorted({_clean(axis.get("axis")) for axis in orthogonal_blocking_axes})
        )
        blockers.append(
            f"{hypothesis.cancer_type} encodes orthogonal molecular/status "
            f"state ({axes}); keep it as an annotation on the parent diagnosis "
            "unless definitive molecular evidence makes the state itself the label"
        )
        hypothesis.details["fused_evidence_orthogonal_status_axes"] = (
            orthogonal_blocking_axes
        )
    if hypothesis.details.get("local_reference_explicit_negative_fusion"):
        details = (
            hypothesis.details.get("local_reference_explicit_negative_fusion_details")
            or {}
        )
        blockers.append(
            _fusion_input_missing_expected_driver_reason(details)
            if isinstance(details, Mapping)
            else "fusion-driven expression reference conflicts with supplied fusion evidence"
        )
    if (
        hypothesis.details.get("local_reference_unexpected_low_lineage_conflict")
        and not strong_independent_learned_call
        and not signature_anchor_can_escape_expected_low
    ):
        conflict = (
            hypothesis.details.get(
                "local_reference_unexpected_low_lineage_conflict_details"
            )
            or {}
        )
        if isinstance(conflict, Mapping):
            genes = ", ".join(conflict.get("unexpected_low_genes") or [])
            blockers.append(
                "local expression reference has expected-low marker conflict"
                f"{f' ({genes})' if genes else ''} supporting close "
                f"{conflict.get('conflicting_lineage', 'alternate-lineage')} "
                f"candidate {conflict.get('close_candidate', '')}"
            )
        else:
            blockers.append(
                "local expression reference has expected-low marker conflict"
            )
        if signature_anchor_learned_contraindicated:
            blockers.append(
                "signature-anchored expression reference is contradicted by the "
                "learned expression neighborhood for this lineage"
            )
    if refinement_learned_contraindicated:
        blockers.append(
            "tumor-label refinement is contradicted by the learned expression "
            "neighborhood and expected-low marker conflicts for this lineage"
        )
    if score < _FUSED_EVIDENCE_MIN_SCORE:
        blockers.append(
            f"integrated evidence score {score:.2f} is below "
            f"{_FUSED_EVIDENCE_MIN_SCORE:.2f}"
        )
    refinement_admitted = selected_by == "tumor_label_refinement"
    lineage_panel_admitted = selected_by == "lineage_panel"
    contrast_admitted = selected_by == "contrast_discriminator"
    composition_admitted = selected_by == "coarse_composition_reference"
    centroid_anchored_expression_reference = bool(
        _safe_float(components.get("centroid_anchored_expression_reference")) > 0
    )
    learned_compartment_anchored_context = bool(
        _safe_float(
            components.get("learned_compartment_anchored_pan_cancer_context")
        )
        >= 0.75
    )
    learned_family_anchored_context = bool(
        _safe_float(
            components.get("learned_family_anchored_pan_cancer_context")
        )
        >= _FUSED_EVIDENCE_LEARNED_FAMILY_ANCHOR_MIN_COMPONENT
    )
    pan_signature_marker_program = bool(
        features.get("pan_cancer_signature_marker_selectable")
        and _safe_float(components.get("pan_cancer_signature_marker_program"))
        >= _PAN_SIGNATURE_MARKER_PROGRAM_MIN_SUPPORT
    )
    pan_marker_coherence = (
        hypothesis.details.get("pan_cancer_signature_marker_coherence") or {}
    )
    pan_signature_identity_specific = bool(
        pan_signature_marker_program
        and isinstance(pan_marker_coherence, Mapping)
        and _clean(pan_marker_coherence.get("code"))
        == hypothesis.cancer_type
        and _pan_signature_marker_program_selectable(pan_marker_coherence)
    )
    primary_context_conflict = _dominant_primary_context_competitor(
        hypothesis,
        analysis,
    )
    learned_component = _safe_float(components.get("learned_expression_classifier"))
    context_free_learned_triplet = bool(
        learned >= _FUSED_EVIDENCE_CONTEXT_FREE_LEARNED_PROBABILITY
        and learned_hierarchy >= _LEARNED_EXPRESSION_MIN_HIERARCHICAL_CONTEXT
        and centroid_support >= _FUSED_EVIDENCE_CONTEXT_FREE_CENTROID_SUPPORT
    )
    weak_reference_for_learned = bool(
        learned_component > 0
        and "local_expression_reference" in hypothesis.evidence_sources
        and not exact_reference_admitted
        and not context_free_learned_triplet
        and not strong_learned_entity_call
        and (
            _safe_float(hypothesis.details.get("local_reference_marker_fraction"))
            < _FUSED_EVIDENCE_LEARNED_WEAK_REFERENCE_MAX_FRACTION
            or _safe_float(hypothesis.details.get("local_reference_marker_burden_ratio"))
            < _FUSED_EVIDENCE_LEARNED_WEAK_REFERENCE_MAX_BURDEN
        )
    )
    if weak_reference_for_learned:
        blockers.append(
            "learned fused support is paired only with a weak or blocked "
            "local expression reference; require coherent marker fraction and "
            "burden before overriding the pan-cancer signature context"
        )
    identity_specific_override = bool(
        exact_reference_admitted
        or lineage_panel_admitted
        or rare_marker_admitted
        or contrast_admitted
        or refinement_admitted
        or signature_anchor_can_escape_expected_low
        or centroid_anchored_expression_reference
        or pan_signature_identity_specific
    )
    broad_top_code = _top_code(analysis)
    unsupported_cross_context_fusion = bool(
        broad_top_code
        and broad_top_code != hypothesis.cancer_type
        and _primary_context_root_for_code(broad_top_code)
        != _primary_context_root_for_code(hypothesis.cancer_type)
        and not strong_independent_learned_call
        and not identity_specific_override
        and (
            not hypothesis.can_select_report_label
            or selected_by == "learned_expression_classifier"
        )
    )
    if unsupported_cross_context_fusion:
        blockers.append(
            "fused evidence cannot create a cross-context entity from broad "
            "signature, centroid, family, or compartment similarity alone; "
            "require an admitted entity selector, a strong learned entity, "
            "an entity-specific marker program, or an expression reference"
        )
        hypothesis.details[
            "fused_evidence_unsupported_cross_context_fusion"
        ] = dict(
            primary_context_conflict
            or {
                "code": broad_top_code,
                "context": _primary_context_root_for_code(broad_top_code),
                "candidate_code": hypothesis.cancer_type,
                "candidate_context": _primary_context_root_for_code(
                    hypothesis.cancer_type
                ),
            }
        )
    primary_context_override = bool(
        strong_independent_learned_call
        or exact_reference_admitted
        or lineage_panel_admitted
        or rare_marker_admitted
        or contrast_admitted
        or refinement_admitted
        or composition_admitted
        or signature_anchor_can_escape_expected_low
        or centroid_anchored_expression_reference
        or learned_family_anchored_context
        or pan_signature_identity_specific
        or hypothesis.direct_fusion_support > 0
    )
    if primary_context_conflict and not primary_context_override:
        blockers.append(
            "integrated evidence conflicts with a stronger pan-cancer "
            "primary-context ranker hypothesis "
            f"({primary_context_conflict['code']} -> "
            f"{primary_context_conflict['context']}, rank "
            f"{primary_context_conflict['rank']}, "
            f"{_safe_float(primary_context_conflict['support_fraction_of_top']):.2f}x top); "
            "require admitted exact/reference, marker, lineage-panel, fusion, "
            "contrast, composition/refinement, or strong learned evidence before "
            f"selecting {hypothesis.cancer_type}"
        )
        hypothesis.details["fused_evidence_primary_context_conflict"] = (
            primary_context_conflict
        )
    has_admission_path = bool(
        (
            learned >= _LEARNED_EXPRESSION_STRONG_PROBABILITY
            and (
                learned_admitted
                or centroid_support >= 0.45
                or signature_anchored_exact_reference
                or exact_reference_admitted
                or context_free_learned_triplet
                or strong_learned_entity_call
            )
            and (
                learned_hierarchy >= _LEARNED_EXPRESSION_MIN_HIERARCHICAL_CONTEXT
                or centroid_support >= 0.45
                or hypothesis.broad_rna_support >= 0.45
                or admissible_exact_reference >= 0.45
            )
        )
        or (
            centroid_support >= 0.95
            and (
                learned >= 0.20
                or exact_reference_admitted
                or lineage_panel_admitted
                or rare_marker_admitted
                or contrast_admitted
                or refinement_admitted
                or composition_admitted
                or signature_anchored_exact_reference
                or hypothesis.direct_fusion_support > 0
            )
            and (
                learned >= 0.20
                or admissible_exact_reference >= 0.15
                or lineage_panel >= _LINEAGE_PANEL_MIN_SCORE
                or hypothesis.rna_marker_support >= 0.20
                or admissible_contrast >= 0.20
                or hypothesis.coarse_composition_support >= 0.20
            )
        )
        or (
            admissible_exact_reference >= _LOCAL_REFERENCE_MIN_SUPPORT
            and hypothesis.can_select_report_label
            and hypothesis.selected_by in {"fine_reference", "local_expression_reference"}
            and (
                hypothesis.related_context_support >= _LOCAL_REFERENCE_MIN_CONTEXT_SUPPORT
                or centroid_support >= 0.45
                or learned >= _LEARNED_EXPRESSION_STRONG_PROBABILITY
                or signature_anchored_exact_reference
            )
        )
        or centroid_anchored_expression_reference
        or learned_compartment_anchored_context
        or learned_family_anchored_context
        or pan_signature_marker_program
        or composition_admitted
        or contrast_admitted
        or (
            lineage_panel >= _LINEAGE_PANEL_MIN_SCORE
            and hypothesis.can_select_report_label
            and hypothesis.selected_by == "lineage_panel"
        )
        or refinement_admitted
        or rare_marker_admitted
    )
    if not has_admission_path:
        blockers.append(
            "integrated evidence lacks a non-ranker admission path: no strong "
            "learned/hierarchical context, centroid-backed exact reference, "
            "centroid-anchored expression reference, learned-compartment "
            "or learned-family anchored pan-cancer context, ranker marker program, "
            "lineage panel, rare marker, contrast discriminator, or "
            "another candidate-specific tumor-identity selector"
        )
    return not blockers, blockers


def _add_fused_evidence_features(
    hypotheses: dict[str, CancerTypeEvidence],
    analysis: Mapping[str, Any],
    *,
    sample_tpm_by_symbol: Mapping[str, float] | None = None,
    cen=None,
    centroid_confident: bool = False,
) -> None:
    if not hypotheses:
        return
    centroid_supports = _centroid_supports_for_hypotheses(
        hypotheses,
        cen if centroid_confident else None,
    )
    scored: list[tuple[float, str, CancerTypeEvidence, list[str]]] = []
    for code, hypothesis in hypotheses.items():
        hypothesis.details["fused_evidence_preselector"] = (
            hypothesis.selected_by if hypothesis.can_select_report_label else ""
        )
        centroid_support = _safe_float(centroid_supports.get(code))
        components = _fused_component_scores(
            hypothesis,
            centroid_support=centroid_support,
        )
        grouped_components = _group_fused_evidence_components(components)
        score = round(float(sum(grouped_components.values())), 4)
        can_select, blockers = _fused_evidence_eligible(
            hypothesis,
            analysis,
            score=score,
            centroid_support=centroid_support,
            components=components,
        )
        hypothesis.details["fused_evidence_score"] = score
        hypothesis.details["fused_evidence_components"] = dict(components)
        hypothesis.details["fused_evidence_grouped_components"] = dict(
            grouped_components
        )
        hypothesis.details["fused_evidence_centroid_support"] = round(
            float(centroid_support),
            4,
        )
        hypothesis.details["fused_evidence_can_select"] = bool(can_select)
        if blockers:
            hypothesis.details["fused_evidence_blockers"] = list(blockers)
        hard_blockers = _persistent_report_label_blockers(hypothesis)
        if hard_blockers:
            blockers = list(
                dict.fromkeys((*blockers, *hard_blockers))
            )
            can_select = False
            hypothesis.details["fused_evidence_can_select"] = False
            hypothesis.details["fused_evidence_blockers"] = list(blockers)
        if hypothesis.can_select_report_label and hard_blockers:
            hypothesis.can_select_report_label = False
            hypothesis.label_status = "blocked"
            hypothesis.label_basis = hypothesis.selected_by
            hypothesis.blocking_reasons = tuple(
                dict.fromkeys((*hypothesis.blocking_reasons, *hard_blockers))
            )
            hypothesis.selection_priority = (0, 0.0, 0)
        if (
            hypothesis.can_select_report_label
            and hypothesis.selected_by
            in {"fine_reference", "local_expression_reference"}
            and any(
                "signature-anchored expression reference is contradicted"
                in reason
                for reason in blockers
            )
        ):
            hypothesis.can_select_report_label = False
            hypothesis.label_status = "blocked"
            hypothesis.label_basis = hypothesis.selected_by
            hypothesis.blocking_reasons = tuple(blockers)
            hypothesis.selection_priority = (0, 0.0, 0)
        if (
            hypothesis.can_select_report_label
            and hypothesis.selected_by == "tumor_label_refinement"
            and any(
                "tumor-label refinement is contradicted"
                in reason
                for reason in blockers
            )
        ):
            hypothesis.can_select_report_label = False
            hypothesis.label_status = "blocked"
            hypothesis.label_basis = hypothesis.selected_by
            hypothesis.blocking_reasons = tuple(blockers)
            hypothesis.selection_priority = (0, 0.0, 0)
        if (
            hypothesis.can_select_report_label
            and hypothesis.selected_by == "learned_expression_classifier"
            and blockers
        ):
            # A learned-classifier self-admission is a weak, subordinate channel, and every fused
            # blocker that can survive to here for it is a genuine withhold-directive: the
            # strong-learned escape hatches (strong_independent_learned_call → primary_context_override
            # and has_admission_path) suppress the primary-context / CRC-family / admission-path
            # blockers BEFORE they are appended, so a surviving blocker means the learned call is weak
            # AND conflicts with a dominant pan-cancer context. Enforce it — otherwise _pick_selected
            # would still choose the label the fused layer just said "must not be selected" (e.g. a
            # weak STAD learned call self-selecting over a dominant READ/CRC primary context).
            hypothesis.can_select_report_label = False
            hypothesis.label_status = "blocked"
            hypothesis.label_basis = hypothesis.selected_by
            hypothesis.blocking_reasons = tuple(blockers)
            hypothesis.selection_priority = (0, 0.0, 0)
        scored.append((score, code, hypothesis, blockers))
    if not scored:
        return

    scored.sort(key=lambda item: (-item[0], item[1]))
    for rank, (_score, _code, hypothesis, _blockers) in enumerate(scored, start=1):
        hypothesis.details["fused_evidence_rank"] = rank
    eligible = [
        (score, code, hypothesis, blockers)
        for score, code, hypothesis, blockers in scored
        if not blockers
    ]
    if not eligible:
        return
    best_score, _best_code, best, best_blockers = eligible[0]
    runner_score = eligible[1][0] if len(eligible) > 1 else 0.0
    best.details["fused_evidence_margin"] = round(float(best_score - runner_score), 4)
    best.details["fused_evidence_selected"] = True
    best.basis = best.basis or "integrated RNA evidence consensus"
    if best.can_select_report_label and best.selected_by:
        best.selection_priority = max(
            best.selection_priority,
            (
                3,
                best_score,
                _SELECTED_BY_TIEBREAK_RANK.get(best.selected_by, 0),
            ),
        )
        best.details["fused_evidence_preserved_selector"] = best.selected_by
    else:
        best.consider_for_report_label(
            selected_by="fused_evidence",
            can_select=True,
            blocking_reasons=(),
            priority=(3, best_score),
        )


def select_report_scope_from_evidence(
    df_expr,
    analysis: Mapping[str, Any],
    *,
    rare_marker_hypotheses: list[Mapping[str, Any]] | None = None,
    fusion_scope_inference: Mapping[str, Any] | None = None,
    residual_identity_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build cancer-type hypotheses and return the selected report label."""
    try:
        from .common import build_sample_tpm_by_gene_id, build_sample_tpm_by_symbol
    except ImportError:
        _LOGGER.warning(
            "trufflepig.common is unavailable; sample TPM lookup will be empty",
            exc_info=True,
        )
        sample_tpm_by_symbol: Mapping[str, float] = {}
        sample_tpm_by_gene_id: Mapping[str, float] = {}
    else:
        try:
            sample_tpm_by_symbol = build_sample_tpm_by_symbol(df_expr)
        except (KeyError, ValueError, TypeError):
            _LOGGER.warning(
                "build_sample_tpm_by_symbol failed; sample TPM lookup will be empty",
                exc_info=True,
            )
            sample_tpm_by_symbol = {}
        try:
            sample_tpm_by_gene_id = build_sample_tpm_by_gene_id(df_expr)
        except (KeyError, ValueError, TypeError):
            _LOGGER.warning(
                "build_sample_tpm_by_gene_id failed; ID lookup will be empty",
                exc_info=True,
            )
            sample_tpm_by_gene_id = {}

    hypotheses: dict[str, CancerTypeEvidence] = {}
    add_pan_cancer_signature_ranker_evidence(hypotheses, analysis)
    _add_pan_cancer_signature_marker_features(hypotheses, sample_tpm_by_symbol)
    _add_learned_expression_classifier_features(
        hypotheses,
        sample_tpm_by_symbol,
        analysis,
    )
    _add_coarse_composition_reference_features(
        hypotheses,
        sample_tpm_by_symbol,
        analysis,
    )
    _add_quality_gated_composition_context_features(hypotheses, analysis)
    _add_tumor_label_refinement_features(hypotheses, analysis)
    _add_direct_fusion_features(hypotheses, fusion_scope_inference or {}, analysis)
    for spec in _FINE_REFERENCE_SPECS:
        _add_fine_reference_features(
            hypotheses,
            spec,
            sample_tpm_by_symbol,
            analysis,
        )
    _add_local_expression_reference_features(
        hypotheses,
        sample_tpm_by_symbol,
        sample_tpm_by_gene_id,
        analysis,
    )
    lineage_panel_evidence = _add_lineage_panel_features(
        hypotheses,
        sample_tpm_by_symbol,
        analysis,
        sample_tpm_by_gene_id=sample_tpm_by_gene_id,
    )
    _add_contrast_discriminator_features(
        hypotheses,
        sample_tpm_by_symbol,
        analysis,
    )
    for finding in rare_marker_hypotheses or []:
        _add_rare_marker_features(
            hypotheses,
            finding,
            analysis,
            sample_tpm_by_symbol,
        )

    _add_learned_hierarchy_candidate_features(
        hypotheses,
        sample_tpm_by_symbol,
    )

    # One whole-profile centroid pass.  Correlations become selection evidence
    # only when the associated compartment call is confident; otherwise they
    # remain characterization data and cannot be normalized into a vote.
    _cen, _cen_confident = _centroid_and_confidence(sample_tpm_by_symbol)
    _add_fused_evidence_features(
        hypotheses,
        analysis,
        sample_tpm_by_symbol=sample_tpm_by_symbol,
        cen=_cen,
        centroid_confident=_cen_confident,
    )

    selected = _pick_selected(
        hypotheses,
        cen=_cen,
        compartment_confident=_cen_confident,
        sample_tpm_by_symbol=sample_tpm_by_symbol,
    )
    if selected is None:
        selected = _fallback_context_selected(hypotheses, analysis)
    selected = _adjudicate_selection_with_learned_hierarchy(
        hypotheses,
        selected,
        sample_tpm_by_symbol=sample_tpm_by_symbol,
        cen=_cen,
        centroid_confident=_cen_confident,
        residual_identity_evidence=residual_identity_evidence,
    )
    rows = list(hypotheses.values())

    primary_context = _top_code(analysis)
    primary_context_support = _candidate_support_by_code(analysis).get(
        primary_context,
        0.0,
    )

    return {
        "selected": selected.public_dict() if selected is not None else None,
        "evidence": [
            row.public_dict()
            for row in sorted(rows, key=_public_evidence_sort_key)
        ],
        "staged_evidence_graph": _build_staged_evidence_graph(
            rows,
            selected,
            analysis,
        ),
        "primary_expression_context": {
            "cancer_type": primary_context,
            "support": round(float(primary_context_support), 4),
        }
        if primary_context
        else None,
        # Back-compat alias for older report code/tests.
        "top_reference_cancer_type": primary_context,
        # Wiring point #2 (lineage_panels.py contract): expose the
        # panel verdict so analysis-parameters.json and report
        # rendering can surface it without re-running the evaluator.
        "lineage_panel_evidence": lineage_panel_evidence,
        "residual_identity_evidence": (
            dict(residual_identity_evidence)
            if isinstance(residual_identity_evidence, Mapping)
            else None
        ),
    }


__all__ = [
    "CancerTypeEvidence",
    "add_pan_cancer_signature_ranker_evidence",
    "select_report_scope_from_evidence",
]
