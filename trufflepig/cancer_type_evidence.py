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

_LOGGER = logging.getLogger(__name__)

# Source-class rank used as the tie-breaker when two evidence kinds compute
# identical (class_rank, strength). Higher = preferred. The numeric values are
# arbitrary; only their relative order matters. Keep this list in sync with
# every selected_by string that calls consider_for_report_label.
_SELECTED_BY_TIEBREAK_RANK: dict[str, int] = {
    "direct_fusion": 6,
    "fine_reference": 5,
    "local_expression_reference": 4,
    "tumor_label_refinement": 3,
    "rare_marker": 2,
    "primary_expression_match": 1,
}

_SQUAMOUS_CONTEXT_CODES = frozenset({"HNSC", "LUSC", "ESCA", "CESC", "THYM"})
_SALIVARY_CODES = frozenset({"ADCC", "ACINIC"})
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
_LOCAL_REFERENCE_SKIP_FAMILIES = frozenset({"rare", "salivary"})
_LOCAL_REFERENCE_CONTEXT_CODES_BY_FAMILY = {
    "pediatric-bone": ("SARC",),
    "pediatric-soft": ("SARC",),
    "sarcoma": ("SARC",),
    "heme-bcell": ("DLBC", "LAML", "THYM"),
    "heme-plasma": ("DLBC", "LAML", "THYM"),
    "heme-tcell": ("DLBC", "LAML", "THYM"),
    "heme-myeloid": ("LAML", "DLBC", "THYM"),
    "cns": ("GBM", "LGG"),
    "pediatric-cns": ("GBM", "LGG"),
    "pediatric-liver": ("LIHC", "CHOL"),
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


_FINE_REFERENCE_SPECS = (
    FineReferenceSpec(
        cancer_type="OS",
        reference_cancer_type="SARC",
        reference_code="OS",
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
        required_top_context_codes=_SQUAMOUS_CONTEXT_CODES,
    ),
    **{
        code: RareRnaPolicy(
            marker_weight=0.25,
            context_weight=0.60,
            top_context_weight=0.15,
            require_top_context=True,
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
        "local_expression_reference": "local_expression_reference",
        "fine_reference": "fine_reference",
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


def _top_code(analysis: Mapping[str, Any]) -> str:
    rows = _candidate_rows(analysis)
    if rows:
        return _clean(rows[0].get("code"))
    return _clean(analysis.get("cancer_type"))


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


def _local_reference_context_codes(
    code: str,
    registry_row: Mapping[str, Any],
) -> tuple[str, ...]:
    parent = _clean(registry_row.get("parent_code"))
    if parent:
        return (parent,)
    family = _clean(registry_row.get("family")).lower()
    return tuple(_LOCAL_REFERENCE_CONTEXT_CODES_BY_FAMILY.get(family, ()))


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
        ref = (
            ref.groupby(symbol_col, as_index=False)[value_col]
            .median()
            .reset_index(drop=True)
        )
        ranked: list[tuple[float, float, float, str]] = []
        ref_by_symbol: dict[str, float] = {}
        for _, row in ref.iterrows():
            symbol = _clean(row.get(symbol_col))
            if not symbol or is_excluded(symbol):
                continue
            value = _safe_float(row.get(value_col))
            if value < _LOCAL_REFERENCE_MIN_TPM:
                continue
            base = _safe_float(pan_median.get(symbol), default=0.1)
            log2_vs_pan = float(np.log2((value + 1.0) / (base + 1.0)))
            if log2_vs_pan < _LOCAL_REFERENCE_MIN_LOG2_VS_PAN:
                continue
            ref_by_symbol[symbol] = value
            ranked.append((log2_vs_pan, value, -len(symbol), symbol))
        if not ranked:
            return
        ranked.sort(key=lambda item: (-item[0], -item[1], item[3]))
        markers = tuple(
            symbol
            for _log2, _value, _length, symbol in ranked[
                :_LOCAL_REFERENCE_TOP_MARKERS
            ]
        )
        old = panels.get(code)
        if old is not None and int(old.get("_priority", 0)) > priority:
            return
        panels[code] = {
            "markers": markers,
            "ref_medians": ref_by_symbol,
            "context_codes": context_codes,
            "family": _clean(registry_row.get("family")),
            "primary_tissue": _clean(registry_row.get("primary_tissue")),
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
    support_by_code = _candidate_support_by_code(analysis)
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
    hypothesis.related_context_is_top = _top_code(analysis) == spec.reference_cancer_type
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
    analysis: Mapping[str, Any],
) -> None:
    support_by_code = _candidate_support_by_code(analysis)
    top = _top_code(analysis)
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
        context_is_top = top in context_codes
        matched_context = (
            top
            if context_is_top
            else max(
                context_codes,
                key=lambda context_code: support_by_code.get(context_code, 0.0),
            )
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
        blockers: list[str] = []
        if not context_is_top:
            blockers.append(
                "expression-reference context is not the top compatible context "
                f"({', '.join(context_codes)})"
            )
        if context_support < _LOCAL_REFERENCE_MIN_CONTEXT_SUPPORT:
            blockers.append(
                "compatible broad RNA support "
                f"{context_support:.2f} is below "
                f"{_LOCAL_REFERENCE_MIN_CONTEXT_SUPPORT:.2f}"
            )
        if marker_fraction < _LOCAL_REFERENCE_MIN_MARKER_FRACTION:
            blockers.append(
                "local-reference marker fraction "
                f"{marker_fraction:.2f} is below "
                f"{_LOCAL_REFERENCE_MIN_MARKER_FRACTION:.2f}"
            )
        if burden_ratio < _LOCAL_REFERENCE_MIN_BURDEN_RATIO:
            blockers.append(
                "local-reference marker burden "
                f"{burden_ratio:.2f} is below "
                f"{_LOCAL_REFERENCE_MIN_BURDEN_RATIO:.2f}"
            )
        if local_support < _LOCAL_REFERENCE_MIN_SUPPORT:
            blockers.append(
                "local-reference support "
                f"{local_support:.2f} is below "
                f"{_LOCAL_REFERENCE_MIN_SUPPORT:.2f}"
            )

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
        hypothesis.details.update(
            {
                "local_reference_context_codes": list(context_codes),
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
                "local_reference_family": panel.get("family") or "",
                "local_reference_primary_tissue": panel.get("primary_tissue") or "",
                "local_reference_source_cohort": panel.get("source_cohort") or "",
                "local_reference_kind": panel.get("reference_kind") or "",
                "top_markers": marker_details[:8],
            }
        )
        hypothesis.consider_for_report_label(
            selected_by="local_expression_reference",
            can_select=not blockers,
            blocking_reasons=blockers,
            priority=(1, local_support),
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


def _add_rare_marker_features(
    hypotheses: dict[str, CancerTypeEvidence],
    finding: Mapping[str, Any],
    analysis: Mapping[str, Any],
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
    support_by_code = _candidate_support_by_code(analysis)
    top = _top_code(analysis)
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
    context_passes = policy.context_passes(
        top_code=top,
        top_is_context=top_is_context,
        related_context_support=context_support,
    )
    blockers: list[str] = []
    if not rule_promotes:
        blockers.append("rule is a diagnostic prompt only")
    if not context_passes:
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

    hypothesis = _hypothesis(hypotheses, code)
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
        }
    )
    hypothesis.consider_for_report_label(
        selected_by="rare_marker",
        can_select=not blockers,
        blocking_reasons=blockers,
        priority=(1, marker_context_support),
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
        from .common import build_sample_tpm_by_symbol
    except ImportError:
        _LOGGER.warning(
            "trufflepig.common is unavailable; sample TPM lookup will be empty",
            exc_info=True,
        )
        sample_tpm_by_symbol: Mapping[str, float] = {}
    else:
        try:
            sample_tpm_by_symbol = build_sample_tpm_by_symbol(df_expr)
        except (KeyError, ValueError, TypeError):
            _LOGGER.warning(
                "build_sample_tpm_by_symbol failed; sample TPM lookup will be empty",
                exc_info=True,
            )
            sample_tpm_by_symbol = {}

    hypotheses: dict[str, CancerTypeEvidence] = {}
    _add_broad_rna_features(hypotheses, analysis)
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
        analysis,
    )
    for finding in rare_marker_hypotheses or []:
        _add_rare_marker_features(hypotheses, finding, analysis)

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
    }


__all__ = ["CancerTypeEvidence", "select_report_scope_from_evidence"]
