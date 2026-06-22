"""Cancer-type evidence selection.

Each row is one cancer-type hypothesis with the same feature columns:
primary expression-reference support, related expression-context support,
RNA marker support, exact-reference support, and direct-fusion support.
Evidence from different sources accumulates on the same row instead of
becoming separate control-flow branches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger(__name__)

# Source-class rank used as the tie-breaker when two evidence kinds compute
# identical (class_rank, strength). Higher = preferred. The numeric values are
# arbitrary; only their relative order matters. Keep this list in sync with
# every selected_by string that calls consider_for_report_label.
_SELECTED_BY_TIEBREAK_RANK: dict[str, int] = {
    "direct_fusion": 6,
    "fine_reference": 5,
    "local_expression_reference": 4,
    "lineage_panel": 4,
    "contrast_discriminator": 3,
    "tumor_label_refinement": 3,
    "coarse_composition_reference": 2,
    "rare_marker": 2,
    "primary_expression_match": 1,
}

# Thresholds for the lineage_panel selector — gates when a
# trufflepig.lineage_panels panel result is strong enough to
# propose a cancer type. See issue #42 + lineage_panels.py for
# the design rationale. Conservative defaults so this only fires
# on clean within-family discrimination cases.
_LINEAGE_PANEL_MIN_SCORE = 0.60
_LINEAGE_PANEL_MIN_MARGIN_OVER_SECOND = 0.20

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
_TUMOR_LABEL_MIN_SUPPORT = 0.70
_TUMOR_LABEL_MIN_SIGNATURE_RATIO = 0.85
_TUMOR_LABEL_MIN_FAMILY_SUPPORT = 0.65
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
_LOCAL_REFERENCE_CROSS_LINEAGE_MARKER_STATUSES = frozenset({"consistent", "mixed"})
_LOCAL_REFERENCE_SIGNATURE_ANCHOR_GENERIC_PRIMARY_TISSUES = frozenset(
    {"", "soft_tissue", "connective_tissue", "smooth_muscle"}
)
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
    selection_priority: tuple[int, float, int] = field(
        default=(0, 0.0, 0), repr=False
    )

    def add_source(self, source: str) -> None:
        if source and source not in self.evidence_sources:
            self.evidence_sources = self.evidence_sources + (source,)

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
        if can_select and full_priority > self.selection_priority:
            self.can_select_report_label = True
            self.blocking_reasons = ()
            self.selected_by = selected_by
            self.label_status = "selected"
            self.label_basis = selected_by
            self.selection_priority = full_priority
        elif not self.can_select_report_label and blocking_reasons:
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
            "report_label_candidate": bool(self.report_label_candidate),
            "can_select_report_label": bool(self.can_select_report_label),
            "blocking_reasons": list(self.blocking_reasons),
            "metrics": metrics,
            "basis": self.basis,
            "confirmatory_tests": self.confirmatory_tests,
            "caveat": self.caveat,
            "source": self.source,
        }
        if self.broad_rna_rank is not None:
            row["broad_rna_rank"] = self.broad_rna_rank
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
            "primary expression-reference match is sarcoma-like and TARGET "
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
        "tumor_label_refinement": "tumor_label_refinement",
        "coarse_composition_reference": "coarse_composition_reference",
        "local_expression_reference": "local_expression_reference",
        "fine_reference": "fine_reference",
        "contrast_discriminator": "contrast_discriminator",
        "rare_marker": "rna_marker_with_expression_context",
        "direct_fusion": "direct_fusion",
    }.get(selected_by, selected_by)


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
    pairs = _coarse_reference_pairs(analysis)
    top_rho = max((rho for _, rho in pairs), default=0.0)
    if top_rho <= 0:
        return {}
    out: dict[str, float] = {}
    for code, rho in pairs:
        out[code] = max(out.get(code, 0.0), float(np.clip(rho / top_rho, 0.0, 1.0)))
    return out


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
            and best_tissue_score > selected_tissue_score
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
    )


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
    if status in _LOCAL_REFERENCE_CROSS_LINEAGE_MARKER_STATUSES:
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
    if (
        margin < _COARSE_REFERENCE_MIN_MARGIN
        and type_specific_count < _COARSE_REFERENCE_MIN_TYPE_SPECIFIC_HITS
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
        "primary_tissue_score": round(_safe_float(resolved.get("primary_tissue_score")), 4),
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
        from pirlygenes import get_data
    except ImportError:
        return ()
    try:
        df = get_data("cancer-type-discriminators")
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


def _contrast_context_code(
    type_a: str,
    type_b: str,
    support_by_code: Mapping[str, float],
    top_code: str,
) -> tuple[str, float]:
    candidates = [(type_a, _safe_float(support_by_code.get(type_a)))]
    candidates.append((type_b, _safe_float(support_by_code.get(type_b))))
    if top_code in {type_a, type_b}:
        candidates.append((top_code, max(1.0, _safe_float(support_by_code.get(top_code)))))
    code, support = max(candidates, key=lambda item: (item[1], item[0]))
    return (code, float(support)) if support > 0 else ("", 0.0)


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
    consensus_context = _broad_coarse_consensus_context(analysis)

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_clean(row.get("contrast")), []).append(row)

    for contrast, contrast_rows in grouped.items():
        first = contrast_rows[0]
        type_a = _clean(first.get("type_a")).upper()
        type_b = _clean(first.get("type_b")).upper()
        if type_a not in registry or type_b not in registry:
            continue
        context_code, context_support = _contrast_context_code(
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

        same_context = winner_code == context_code
        same_top = winner_code == top_code
        top_participates = top_code in {type_a, type_b}
        context_is_top = context_code == top_code
        context_is_primary = context_code in primary_contexts
        strong_signal = _contrast_signal_is_strong(winner_signal, margin)
        marker_coherence = _marker_coherence(winner_code, sample_tpm_by_symbol)
        context_marker_coherence = _marker_coherence(context_code, sample_tpm_by_symbol)
        context_marker_incoherent = bool(
            context_marker_coherence
            and not _marker_coherence_selection_grade(context_marker_coherence)
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
                "broad RNA ranking and coarse reference matching both support "
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

        support = float(
            np.clip(
                0.55 * winner_score
                + 0.25 * max(margin, 0.0)
                + 0.20 * min(context_support, 1.0),
                0.0,
                1.0,
            )
        )
        hypothesis = _hypothesis(hypotheses, winner_code)
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
        hypothesis.consider_for_report_label(
            selected_by="contrast_discriminator",
            can_select=can_select,
            blocking_reasons=blockers,
            priority=(1, support),
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
        sample_hk_median = _sample_hk_median(sample_tpm_by_gene_id)
        evidence_by_panel = {
            name: score_panel(panel, sample_tpm_by_gene_id, sample_hk_median)
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
    if not _marker_coherence_strong(marker_coherence):
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
    text = _clean(value).lower()
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
        from pirlygenes.gene_sets_cancer import cancer_type_registry
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


@lru_cache(maxsize=1)
def _hk_normalization_gene_ids() -> frozenset[str]:
    """Versionless Ensembl IDs of pirlygenes housekeeping genes.
    Computed once per process — every sample shares the same HK gene
    set so there is no point in re-deriving it inside
    ``_add_lineage_panel_features``.

    Returns ``frozenset()`` when pirlygenes is unavailable; callers
    should fall back to a naive normalization (e.g.
    ``sample_hk_median=1.0``).
    """
    try:
        from pirlygenes.gene_sets_cancer import housekeeping_gene_ids
    except ImportError:
        return frozenset()
    try:
        ids = set(housekeeping_gene_ids())
    except Exception:  # noqa: BLE001
        return frozenset()
    out: set[str] = set()
    for gid in ids:
        s = str(gid or "").strip()
        if not s:
            continue
        out.add(s.split(".", 1)[0])
    return frozenset(out)


def _sample_hk_median(sample_tpm_by_gene_id: Mapping[str, float]) -> float:
    hk_ids = _hk_normalization_gene_ids()
    if not hk_ids:
        return 1.0
    hk_vals = [
        sample_tpm_by_gene_id.get(g, 0.0) for g in hk_ids
        if sample_tpm_by_gene_id.get(g, 0.0) > 0
    ]
    return float(np.median(hk_vals)) if hk_vals else 1.0


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
        from .reference import (
            cancer_reference_expression,
            pan_cancer_expression,
            subtype_deconvolved_expression,
        )
        from .tumor_purity import _compile_excluded_gene_matcher
        from pirlygenes.gene_sets_cancer import is_extended_housekeeping_symbol
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


def _add_broad_rna_features(
    hypotheses: dict[str, CancerTypeEvidence],
    analysis: Mapping[str, Any],
) -> None:
    for rank, row in enumerate(_candidate_rows(analysis), start=1):
        code = _clean(row.get("code"))
        if not code:
            continue
        support = _safe_float(row.get("support_fraction_of_top"))
        hypothesis = _hypothesis(hypotheses, code)
        hypothesis.add_source("broad_rna")
        hypothesis.expression_reference_cancer_type = code
        hypothesis.reference_cancer_type = code
        hypothesis.broad_rna_support = max(hypothesis.broad_rna_support, support)
        hypothesis.broad_rna_rank = rank
        hypothesis.details.update(dict(row))
        hypothesis.details.pop("code", None)
        hypothesis.details["support_fraction_of_top"] = round(support, 4)
        if rank == 1:
            hypothesis.basis = hypothesis.basis or "top RNA expression-reference match"
            hypothesis.consider_for_report_label(
                selected_by="primary_expression_match",
                can_select=True,
                blocking_reasons=(),
                priority=(0, support),
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
    has_specificity = (
        margin >= _COARSE_REFERENCE_MIN_MARGIN
        or type_specific_count >= _COARSE_REFERENCE_MIN_TYPE_SPECIFIC_HITS
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
    if (
        broad_top_is_close_coarse_match
        and margin < _COARSE_REFERENCE_MIN_MARGIN
        and not bool(resolved.get("tissue_tiebreak_applied"))
    ):
        blockers.append(
            "top cancer-reference composition is tied with the first-pass RNA "
            f"winner ({broad_top_code}); type-specific tumor-up evidence alone "
            "does not override the broad RNA call"
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
                _safe_float(resolved.get("primary_tissue_score")),
                4,
            ),
            "coarse_reference_tissue_tiebreak_applied": bool(
                resolved.get("tissue_tiebreak_applied")
            ),
            "coarse_reference_cancer_hint": cancer_hint,
            "coarse_reference_type_specific_cohort": type_specific_cohort,
            "coarse_reference_type_specific_hit_count": type_specific_count,
            "coarse_reference_top_type_specific_hits": top_hit_symbols,
            "coarse_reference_broad_top_code": broad_top_code,
            "coarse_reference_broad_fit_label": fit_label,
            "coarse_reference_broad_top_marker_coherence": broad_top_marker_coherence,
        }
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
    if not _is_background_like_candidate(top):
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

    best: dict[str, Any] | None = None
    best_priority = 0.0
    for row in rows[1:]:
        code = _clean(row.get("code"))
        if not code or _is_background_like_candidate(row):
            continue
        support = _safe_float(row.get("support_fraction_of_top"))
        signature = _safe_float(row.get("signature_score"))
        # ``top_signature > 0`` is guaranteed by the outer guard above —
        # the conditional is kept defensive but is always-true here.
        signature_ratio = signature / top_signature
        family_support = _family_marker_support(row)
        if support < _TUMOR_LABEL_MIN_SUPPORT:
            continue
        if signature_ratio < _TUMOR_LABEL_MIN_SIGNATURE_RATIO:
            continue
        if family_support < _TUMOR_LABEL_MIN_FAMILY_SUPPORT:
            continue
        priority = (
            0.55 * support
            + 0.25 * min(signature_ratio, 1.0)
            + 0.20 * family_support
        )
        if best is None or priority > best_priority:
            best = dict(row)
            best_priority = float(priority)

    if best is None:
        return

    code = _clean(best.get("code"))
    support = _safe_float(best.get("support_fraction_of_top"))
    signature = _safe_float(best.get("signature_score"))
    # ``top_signature > 0`` is guaranteed by the outer guard at line 854.
    signature_ratio = signature / top_signature
    family_support = _family_marker_support(best)
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
        primary_tissue = _clean(panel.get("primary_tissue")).lower()
        signature_anchor_support = (
            0.0
            if (
                molecular_status_source
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
                "broad RNA ranking and coarse reference matching both support "
                f"{consensus_context}; exact-reference context "
                f"{', '.join(context_codes)} is only a secondary near-match"
            )
        if cross_lineage_marker_conflict:
            blockers.append(
                "cross-lineage exact-reference refinement requires a coherent "
                f"{code} marker program; observed "
                f"{cross_lineage_marker_conflict['detected']}/"
                f"{cross_lineage_marker_conflict['total']} expected high "
                f"markers ({cross_lineage_marker_conflict['marker_status']}) "
                "while the compatible RNA context is "
                f"{', '.join(cross_lineage_marker_conflict['context_codes'])}"
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
        fusion_defined_context_conflict: dict[str, Any] = {}
        if fusion_driven == "defining" and not has_direct_fusion:
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
        if fusion_driven == "defining" and not has_direct_fusion:
            if high_confidence_local_reference:
                driver = _clean(panel.get("fusion_driver")) or "the defining fusion"
                fusion_confirmation_caveat = (
                    f"{code} has high-confidence exact expression-reference support, "
                    f"but this entity is fusion-defined ({driver}); confirm the "
                    "fusion before treating the RNA label as definitive."
                )
            else:
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
                "local_reference_cross_lineage_marker_conflict": bool(
                    cross_lineage_marker_conflict
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
            parent_marker_coherence = _marker_coherence(
                parent_code,
                sample_tpm_by_symbol,
            )
            if (
                parent_marker_coherence
                and not _marker_coherence_selection_grade(parent_marker_coherence)
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


def _add_lineage_panel_features(
    hypotheses: dict[str, CancerTypeEvidence],
    sample_tpm_by_symbol: Mapping[str, float],
    analysis: Mapping[str, Any],
    *,
    sample_tpm_by_gene_id: Mapping[str, float] | None = None,
) -> dict[str, Any] | None:
    """Evaluate ``trufflepig.lineage_panels`` against the sample and
    register hypotheses for any panel that clearly wins.

    A "clear win" requires:
      - top panel score >= ``_LINEAGE_PANEL_MIN_SCORE`` (positive
        markers cohort-comparable, low markers compliant, obligates
        passed),
      - margin over second-best panel score >=
        ``_LINEAGE_PANEL_MIN_MARGIN_OVER_SECOND`` (no ambiguity
        between siblings),
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
    expression frame is unavailable, HK normalization falls back to
    ``sample_hk_median=1.0`` and the cross-code path falls back to
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

    # Sample-side HK median for normalization. HK genes are pulled
    # directly by ID from the cached set so the symbol → ID step is
    # skipped entirely for the normalization path.
    sample_hk_median = _sample_hk_median(sample_tpm_by_gene_id)

    try:
        evidence = evaluate_panels(LINEAGE_PANELS, sample_tpm_by_gene_id, sample_hk_median)
    except Exception:  # noqa: BLE001
        _LOGGER.warning("evaluate_panels failed; skipping selector", exc_info=True)
        return None

    summary = dict(summarize_evidence(evidence))
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
    if margin < _LINEAGE_PANEL_MIN_MARGIN_OVER_SECOND:
        summary["promotion"]["blockers"].append(
            f"margin over second {margin:.2f} below threshold "
            f"{_LINEAGE_PANEL_MIN_MARGIN_OVER_SECOND:.2f}"
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

    can_promote = bool(in_broad_top and (same_code or broad_uncertain))
    blockers: list[str] = []
    if not in_broad_top:
        blockers.append(
            f"{code} is not among the top-5 first-pass RNA candidates; "
            "lineage panels only refine candidates the first-pass "
            "classifier already considered"
        )
    elif not (same_code or broad_uncertain):
        blockers.append(
            f"first-pass top-1 ({broad_top_code or 'unknown'}) differs "
            f"from panel parent_cohort ({code}) and the first-pass "
            "classifier is confident — the lineage panel reading is "
            "noted but the first-pass call is preserved"
        )
    consensus_context = _broad_coarse_consensus_context(analysis)
    if can_promote and not same_code and consensus_context and consensus_context != code:
        blockers.append(
            "broad RNA ranking and coarse reference matching both support "
            f"{consensus_context}; lineage panel {top_panel} is noted as a "
            "program signal, not a cross-site report-label override"
        )
        can_promote = False
    marker_coherence = _marker_coherence(code, sample_tpm_by_symbol)
    if (
        can_promote
        and not same_code
        and marker_coherence
        and not _marker_coherence_selection_grade(marker_coherence)
    ):
        blockers.append(
            f"{code} marker program is {marker_coherence.get('status')} "
            f"({marker_coherence.get('detected')}/"
            f"{marker_coherence.get('total')} expected high markers; "
            f"{marker_coherence.get('required_for_consistent')} required)"
        )
        can_promote = False
    if marker_coherence:
        hypothesis.details["lineage_panel_marker_coherence"] = marker_coherence
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
    class_rank = 2 if same_code else 1
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
    if complete_context_gated_marker_axis:
        priority_class = 3
        priority_strength = max(marker_context_support, marker_support)
    if fusion_defined_code and has_local_exact_support:
        priority_class = max(priority_class, 2)
        priority_strength = max(
            marker_context_support,
            0.5 * marker_context_support + 0.5 * hypothesis.fine_reference_support,
            priority_strength,
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
    ascending order. Mirrors ``_selectable_sort_key`` in
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


def select_report_scope_from_evidence(
    df_expr,
    analysis: Mapping[str, Any],
    *,
    rare_marker_hypotheses: list[Mapping[str, Any]] | None = None,
    fusion_scope_inference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build cancer-type hypotheses and return the selected report label."""
    try:
        from .common import build_sample_tpm_by_symbol, build_sample_tpm_by_gene_id
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
    _add_broad_rna_features(hypotheses, analysis)
    _add_coarse_composition_reference_features(
        hypotheses,
        sample_tpm_by_symbol,
        analysis,
    )
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

    rows = list(hypotheses.values())
    selectable = [row for row in rows if row.can_select_report_label]
    selected = None
    if selectable:
        # Tie-break: highest selection_priority first (class_rank, then
        # strength, then tiebreak slot), then *ascending* cancer_type
        # alphabetical for determinism. The previous form
        # ``sort((priority, cancer_type), reverse=True)`` flipped both
        # fields, producing Z-before-A on ties (e.g. UCS over THCA at
        # equal priority) — surprising and undocumented.
        def _selectable_sort_key(row):
            cls, strength, tb = row.selection_priority
            return (-cls, -strength, -tb, row.cancer_type)

        selectable.sort(key=_selectable_sort_key)
        selected = selectable[0]

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
    }


__all__ = ["CancerTypeEvidence", "select_report_scope_from_evidence"]
