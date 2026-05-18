"""Cancer-type evidence selection.

Each row is one cancer-type hypothesis with the same feature columns:
broad RNA support, related broad-context support, RNA marker support,
fine-reference support, and direct-fusion support.  Evidence from different
sources accumulates on the same row instead of becoming separate control-flow
branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping

import numpy as np

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
    _selection_priority: tuple[int, float] = field(default=(0, 0.0), repr=False)

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
        self.label_status = "blocked"
        self.label_basis = selected_by
        if can_select and priority > self._selection_priority:
            self.can_select_report_label = True
            self.blocking_reasons = ()
            self.selected_by = selected_by
            self.label_status = "selected"
            self._selection_priority = priority
        elif not self.can_select_report_label and blocking_reasons:
            self.blocking_reasons = tuple(blocking_reasons)

    def public_dict(self) -> dict[str, Any]:
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
            "reference_cancer_type": self.reference_cancer_type,
            "evidence_sources": list(self.evidence_sources),
            "selected_by": self.selected_by,
            "label_decision": {
                "status": self.label_status,
                "basis": self.selected_by or self.label_basis,
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
    marker_weight: float = 0.40
    context_weight: float = 0.45
    top_context_weight: float = 0.0
    min_marker_context_support: float = 0.75
    min_related_context_support: float = 0.60
    require_top_context: bool = False
    required_top_context_codes: frozenset[str] = frozenset()

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
            "broad RNA call is sarcoma-like and TARGET osteosarcoma "
            "reference markers show an osteogenic tumor program"
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
    except Exception:
        return float(default)
    if result != result:
        return float(default)
    return result


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


def _candidate_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in (analysis.get("candidate_trace") or [])]


def _candidate_support_by_code(analysis: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in _candidate_rows(analysis):
        code = _clean(row.get("code"))
        if code:
            out[code] = max(out.get(code, 0.0), _safe_float(row.get("support_norm")))
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

        df = rare_cancer_rna_surrogate_rules_df().fillna("")
        return {
            _clean(row.get("rule_id")): dict(row)
            for _, row in df.iterrows()
            if _clean(row.get("rule_id"))
        }
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _registry_by_code() -> dict[str, dict[str, Any]]:
    try:
        from pirlygenes.gene_sets_cancer import cancer_type_registry

        df = cancer_type_registry().fillna("")
        return {
            _clean(row.get("code")): dict(row)
            for _, row in df.iterrows()
            if _clean(row.get("code"))
        }
    except Exception:
        return {}


def _local_reference_context_codes(
    code: str,
    registry_row: Mapping[str, Any],
) -> tuple[str, ...]:
    parent = _clean(registry_row.get("parent_code"))
    if parent:
        return (parent,)
    family = _clean(registry_row.get("family")).lower()
    return tuple(_LOCAL_REFERENCE_CONTEXT_CODES_BY_FAMILY.get(family, ()))


@lru_cache(maxsize=1)
def _local_expression_reference_panels() -> dict[str, dict[str, Any]]:
    """Build marker panels for exact non-TCGA expression references.

    The selector uses these panels for future exact references such as MM,
    CLL, ALL, or pediatric/fine sarcoma cohorts. The broad RNA classifier
    still provides the coarse context; this table only decides whether an
    exact local cohort is a better report label within that context.
    """
    try:
        from .reference import pan_cancer_expression, subtype_deconvolved_expression
        from .tumor_purity import _compile_excluded_gene_matcher

        sub = subtype_deconvolved_expression(technical_rna_normalize=True)
        if sub is None or sub.empty:
            return {}
        subtype_values = (
            sub.get("subtype").fillna("").astype(str).map(_clean)
            if "subtype" in sub.columns
            else ""
        )
        sub = sub[subtype_values == ""].copy()
        if sub.empty:
            return {}

        pan = (
            pan_cancer_expression(technical_rna_normalize=True)
            .drop_duplicates(subset="Symbol")
            .set_index("Symbol")
        )
        cohort_cols = [col for col in pan.columns if str(col).endswith("_TPM")]
        if not cohort_cols:
            return {}
        pan_median = pan[cohort_cols].astype(float).median(axis=1)
        is_excluded = _compile_excluded_gene_matcher()
        registry = _registry_by_code()
        panels: dict[str, dict[str, Any]] = {}
        for code_value, group in sub.groupby("cancer_code"):
            code = _clean(code_value)
            if not code:
                continue
            registry_row = registry.get(code, {})
            family = _clean(registry_row.get("family")).lower()
            if family in _LOCAL_REFERENCE_SKIP_FAMILIES:
                continue
            context_codes = _local_reference_context_codes(code, registry_row)
            if not context_codes:
                continue
            ref = (
                group.groupby("symbol", as_index=False)["tumor_tpm_median"]
                .median()
                .reset_index(drop=True)
            )
            ranked: list[tuple[float, float, float, str]] = []
            ref_by_symbol: dict[str, float] = {}
            for _, row in ref.iterrows():
                symbol = _clean(row.get("symbol"))
                if not symbol or is_excluded(symbol):
                    continue
                value = _safe_float(row.get("tumor_tpm_median"))
                if value < _LOCAL_REFERENCE_MIN_TPM:
                    continue
                base = _safe_float(pan_median.get(symbol), default=0.1)
                log2_vs_pan = float(np.log2((value + 1.0) / (base + 1.0)))
                if log2_vs_pan < _LOCAL_REFERENCE_MIN_LOG2_VS_PAN:
                    continue
                ref_by_symbol[symbol] = value
                ranked.append((log2_vs_pan, value, -len(symbol), symbol))
            if not ranked:
                continue
            ranked.sort(key=lambda item: (-item[0], -item[1], item[3]))
            markers = tuple(
                symbol
                for _log2, _value, _length, symbol in ranked[
                    :_LOCAL_REFERENCE_TOP_MARKERS
                ]
            )
            panels[code] = {
                "markers": markers,
                "ref_medians": ref_by_symbol,
                "context_codes": context_codes,
                "family": _clean(registry_row.get("family")),
                "primary_tissue": _clean(registry_row.get("primary_tissue")),
                "source_cohort": _clean(registry_row.get("source_cohort")),
            }
        return panels
    except Exception:
        return {}


@lru_cache(maxsize=None)
def _reference_medians(reference_code: str) -> dict[str, float]:
    try:
        from .reference import subtype_deconvolved_expression

        df = subtype_deconvolved_expression()
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
    except Exception:
        return {}


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
        support = _safe_float(row.get("support_norm"))
        hypothesis = _hypothesis(hypotheses, code)
        hypothesis.add_source("broad_rna")
        hypothesis.reference_cancer_type = code
        hypothesis.broad_rna_support = max(hypothesis.broad_rna_support, support)
        hypothesis.broad_rna_rank = rank
        hypothesis.details.update(dict(row))
        hypothesis.details.pop("code", None)
        hypothesis.details["support_norm"] = round(support, 4)


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
    top_support = _safe_float(top.get("support_norm"), default=1.0)
    top_signature = _safe_float(top.get("signature_score"))
    if not top_code or top_support <= 0 or top_signature <= 0:
        return

    best: dict[str, Any] | None = None
    best_priority = 0.0
    for row in rows[1:]:
        code = _clean(row.get("code"))
        if not code or _is_background_like_candidate(row):
            continue
        support = _safe_float(row.get("support_norm"))
        signature = _safe_float(row.get("signature_score"))
        signature_ratio = signature / top_signature if top_signature > 0 else 0.0
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
    support = _safe_float(best.get("support_norm"))
    signature = _safe_float(best.get("signature_score"))
    signature_ratio = signature / top_signature if top_signature > 0 else 0.0
    family_support = _family_marker_support(best)
    hypothesis = _hypothesis(hypotheses, code)
    hypothesis.add_source("tumor_label_refinement")
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
            "competing_background_support_norm": round(float(top_support), 4),
            "tumor_label_support_norm": round(float(support), 4),
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
            "parent_support_norm": round(float(context_support), 4),
            f"{spec.reference_cancer_type.lower()}_support_norm": round(
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
        markers = tuple(panel.get("markers") or ())
        ref_medians = dict(panel.get("ref_medians") or {})
        marker_fraction, burden_ratio, marker_details = _local_reference_marker_support(
            sample_tpm_by_symbol,
            markers,
            ref_medians,
        )
        if marker_fraction <= 0 and burden_ratio <= 0:
            continue
        burden_support = min(burden_ratio / 0.35, 1.0)
        local_support = (
            0.40 * context_support
            + 0.35 * marker_fraction
            + 0.25 * burden_support
        )
        blockers: list[str] = []
        if not context_is_top:
            blockers.append(
                "broad RNA context is not the top compatible context "
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
        hypothesis.reference_cancer_type = hypothesis.reference_cancer_type or (
            context_codes[0] if context_codes else top
        )
        hypothesis.related_context_code = ",".join(context_codes)
        hypothesis.related_context_support = max(
            hypothesis.related_context_support,
            context_support,
        )
        hypothesis.related_context_is_top = (
            hypothesis.related_context_is_top or context_is_top
        )
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


def _add_rare_marker_features(
    hypotheses: dict[str, CancerTypeEvidence],
    finding: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> None:
    code = _clean(finding.get("cancer_type"))
    rule_id = _clean(finding.get("rule_id"))
    if not code or not rule_id:
        return
    if finding.get("exclusion_genes_observed") or finding.get("missing_support_genes"):
        return

    rule = _rare_rules_by_id().get(rule_id, {})
    rule_promotes = _safe_bool(rule.get("promote_report_scope"), default=True)
    context_codes = set(_split_semicolon(rule.get("context_codes")))
    support_by_code = _candidate_support_by_code(analysis)
    top = _top_code(analysis)
    context_support = max(
        (support_by_code.get(context_code, 0.0) for context_code in context_codes),
        default=0.0,
    )
    top_is_context = bool(top and top in context_codes)
    primary_tpm = _safe_float(finding.get("surrogate_tpm"))
    threshold = max(_safe_float(finding.get("threshold_tpm")), 1e-9)
    marker_support = min(primary_tpm / threshold, 1.0)

    policy = _rare_rna_policy(code)
    effective_context = max(context_support, float(top_is_context))
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
            blockers.append(f"broad RNA context is not one of {allowed}")
        else:
            blockers.append("broad RNA context does not support this marker")
    if marker_context_support < policy.min_marker_context_support:
        blockers.append(
            "marker/context support "
            f"{marker_context_support:.2f} is below "
            f"{policy.min_marker_context_support:.2f}"
        )

    hypothesis = _hypothesis(hypotheses, code)
    hypothesis.add_source("rare_marker")
    hypothesis.reference_cancer_type = hypothesis.reference_cancer_type or top
    hypothesis.related_context_code = (
        ";".join(sorted(context_codes)) if context_codes else hypothesis.related_context_code
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
            "context_codes": sorted(context_codes),
            "context_support_norm": round(float(context_support), 4),
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


def _public_evidence_sort_key(evidence: CancerTypeEvidence) -> tuple[int, int, int, float, str]:
    class_rank, strength = evidence._selection_priority
    return (
        1 if evidence.can_select_report_label else 0,
        1 if evidence.report_label_candidate else 0,
        class_rank,
        strength,
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

        sample_tpm_by_symbol = build_sample_tpm_by_symbol(df_expr)
    except Exception:
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
        selectable.sort(key=lambda row: row._selection_priority, reverse=True)
        selected = selectable[0]

    return {
        "selected": selected.public_dict() if selected is not None else None,
        "evidence": [
            row.public_dict()
            for row in sorted(rows, key=_public_evidence_sort_key, reverse=True)
        ],
        "top_reference_cancer_type": _top_code(analysis),
    }


__all__ = ["CancerTypeEvidence", "select_report_scope_from_evidence"]
