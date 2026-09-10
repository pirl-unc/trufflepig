"""The typed finalization boundary shared by every report renderer.

The analysis pipeline remains mutable while evidence is integrated.  Once it is
finished, :func:`build_report_view` validates the required fields and freezes the
report conclusions.  Renderers receive that object explicitly; they never merge
it with later values from the analysis dictionary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any, Optional, Tuple

from .confidence import (
    BULK_PURITY_REFERENCE_SOURCES,
    ConfidenceTier,
    compute_call_confidence,
    purity_fraction,
    purity_confidence_for_analysis,
    sample_purity_is_low,
)


PurityScenario = Tuple[str, Optional[float], Optional[float], Optional[float]]


@dataclass(frozen=True)
class PurityMethodEstimate:
    """One displayed method estimate, separate from the adopted report value.

    Shared inputs and enrichment-derived estimates are not independent assays.
    ``family`` retains that distinction in the figure and structured report.
    """

    label: str
    estimate: float
    lower: Optional[float] = None
    upper: Optional[float] = None
    family: str = ""
    genes: int = 0
    note: str = ""


def purity_method_estimates(purity: Mapping[str, Any], decomposition=None) -> Tuple[PurityMethodEstimate, ...]:
    """Collect the existing quantitative figure rows without refitting a model.

    Tumor-only lineage references cannot measure admixture. ESTIMATE components
    retain the existing bulk-reference odds conversion and are labeled derived.
    Missing/invalid values cannot supply a method or establish agreement.
    """
    components = purity.get("components") or {}
    bulk_reference = purity.get("reference_expression_source") in BULK_PURITY_REFERENCE_SOURCES
    methods = []
    for family, label in (("signature", "Tumor-specific signature"), ("lineage", "Lineage panel")):
        component = components.get(family) or {}
        point = purity_fraction(component.get("purity"))
        if point is None or (family == "lineage" and not bulk_reference):
            continue
        note = " (deprioritized)" if family == "signature" and (
            components.get("integration") or {}).get("signature_deprioritized") else ""
        methods.append(PurityMethodEstimate(
            label, point, purity_fraction(component.get("lower")), purity_fraction(component.get("upper")),
            family, len(component.get("genes") or []), note,
        ))
    residual = components.get("decomposition") or {}
    residual_fraction = purity_fraction(residual.get("residual_fraction"))
    if residual_fraction is not None:
        methods.append(PurityMethodEstimate(
            "Background-residual fraction", residual_fraction, family="decomposition",
            note=f" [{residual.get('mode', 'lineage-routed')}]",
        ))
    elif decomposition is not None:
        point = purity_fraction(getattr(decomposition, "purity", None))
        warnings = getattr(decomposition, "warnings", None) or []
        if point is not None and not any("No non-tumor" in str(warning) for warning in warnings):
            methods.append(PurityMethodEstimate(
                "Decomposition (NNLS)", point, family="decomposition",
                note=f" [{getattr(decomposition, 'cancer_type', '')} / {getattr(decomposition, 'template', '')}]",
            ))
    median = purity_fraction(purity.get("tcga_median_purity"))
    if bulk_reference and median is not None and median > 0:
        odds_nontumor = (1 - median) / max(median, 1e-6)
        for family in ("stromal", "immune"):
            component = components.get(family) or {}
            enrichment = component.get("enrichment")
            if enrichment is None or isinstance(enrichment, bool):
                continue
            try:
                enrichment = float(enrichment)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(enrichment):
                continue
            point = 1 / (1 + odds_nontumor * max(enrichment, 0))
            methods.append(PurityMethodEstimate(
                f"ESTIMATE {family} (derived)", point, family="estimate",
                genes=component.get("n_genes", 0),
            ))
    combined = purity_fraction(components.get("estimate_purity"))
    if combined is not None:
        methods.append(PurityMethodEstimate("ESTIMATE combined (derived)", combined, family="estimate"))
    return tuple(methods)


@dataclass(frozen=True)
class Purity:
    """One immutable purity result for every report surface."""

    estimate: Optional[float]
    lower: Optional[float]
    upper: Optional[float]
    method: Optional[str]
    confidence: ConfidenceTier
    status: str
    scenarios: Tuple[PurityScenario, ...]
    unresolved_reason: Optional[str] = None
    methods: Tuple[PurityMethodEstimate, ...] = ()

    @property
    def is_unresolved(self) -> bool:
        return self.status == "discordant_estimators"

    @property
    def at_ceiling(self) -> bool:
        return self.estimate is not None and self.estimate >= 0.995

    @property
    def is_low(self) -> bool:
        return sample_purity_is_low(self.confidence)

    @property
    def has_wide_interval(self) -> bool:
        return self.lower is not None and self.upper is not None and self.upper - self.lower >= 0.35

    @property
    def method_range(self) -> Optional[Tuple[float, float]]:
        points = [method.estimate for method in self.methods]
        return (min(points), max(points)) if points else None

    @property
    def methods_exceed_interval(self) -> bool:
        bounds = self.method_range
        return bool(bounds and len(self.methods) > 1 and self.lower is not None and self.upper is not None
                    and (bounds[0] < self.lower or bounds[1] > self.upper))

    @classmethod
    def from_analysis(cls, analysis: Mapping[str, Any]) -> "Purity":
        """Validate and freeze purity from one analysis mapping."""
        purity = analysis["purity"]
        if not isinstance(purity, Mapping):
            raise TypeError("Finalized analysis purity must be a mapping")

        def number(field: str, value):
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Finalized purity {field} must be numeric")
            result = float(value)
            if not 0.0 <= result <= 1.0:
                raise ValueError(
                    f"Finalized purity {field} must be between 0 and 1"
                )
            return result

        method = purity.get("purity_source")
        if not method:
            components = purity.get("components") or {}
            if not isinstance(components, Mapping):
                raise TypeError("Finalized purity components must be a mapping")
            integration = components.get("integration") or {}
            if not isinstance(integration, Mapping):
                raise TypeError(
                    "Finalized purity integration component must be a mapping"
                )
            method = integration.get("source")
        if method is not None and not isinstance(method, str):
            raise TypeError("Finalized purity method must be a string or None")
        scenarios = []
        for row in purity.get("estimator_scenarios") or ():
            if not isinstance(row, Mapping):
                raise TypeError("Finalized purity scenarios must be mappings")
            source = row.get("source") or "unspecified"
            if not isinstance(source, str):
                raise TypeError("Finalized purity scenario source must be a string")
            scenarios.append(
                (
                    source,
                    number("scenario estimate", row.get("estimate")),
                    number("scenario lower", row.get("lower")),
                    number("scenario upper", row.get("upper")),
                )
            )
        status = purity.get("quantitative_status") or "resolved"
        if status not in {"resolved", "discordant_estimators"}:
            raise ValueError(f"Unsupported finalized purity status {status!r}")
        return cls(
            estimate=number("estimate", purity.get("overall_estimate")),
            lower=number("lower bound", purity.get("overall_lower")),
            upper=number("upper bound", purity.get("overall_upper")),
            method=method or None,
            confidence=purity_confidence_for_analysis(analysis),
            status=status,
            scenarios=tuple(scenarios),
            unresolved_reason=(
                str(purity.get("quantitative_unresolved_reason"))
                if purity.get("quantitative_unresolved_reason")
                else None
            ),
            methods=purity_method_estimates(
                purity, next(iter(analysis.get("decomposition_results") or ()), None),
            ),
        )

    def public_dict(self) -> dict[str, Any]:
        """Return the stable, flat headline schema used by report JSON."""
        return {
            "purity": self.estimate,
            "purity_lo": self.lower,
            "purity_hi": self.upper,
            "purity_method": self.method,
            "purity_confidence": self.confidence.tier,
            "purity_confidence_reasons": list(self.confidence.reasons),
            "purity_status": self.status,
            "purity_scenarios": self.scenarios,
            "purity_unresolved_reason": self.unresolved_reason,
            "purity_methods": [asdict(method) for method in self.methods],
        }


@dataclass(frozen=True)
class ReportView:
    """Immutable snapshot of the conclusions a report should render.

    Frozen on purpose: a "new" conclusion is a new object, so the in-place
    mutation that let a stale purity reach one figure cannot recur once
    renderers read from here.
    """

    cancer_type: str
    cancer_type_name: str
    call_confidence: ConfidenceTier
    cancer_type_alternatives: Tuple[Tuple[str, float], ...]

    purity: Purity

    sample_mode: str
    sample_id: Optional[str] = None

    def public_dict(self) -> dict[str, Any]:
        """Return the stable flat headline schema consumed outside Python."""
        return {
            "cancer_type": self.cancer_type,
            "cancer_type_name": self.cancer_type_name,
            "cancer_type_confidence": self.call_confidence.tier,
            "cancer_type_alternatives": self.cancer_type_alternatives,
            **self.purity.public_dict(),
            "sample_mode": self.sample_mode,
            "sample_id": self.sample_id,
        }


def build_report_view(
    analysis: Mapping[str, Any],
    sample_id: Optional[str] = None,
) -> ReportView:
    """Freeze finalized cancer-call and purity conclusions for rendering.

    Call this only after purity finalization and decomposition adoption.  Every
    renderer subsequently receives this exact object.
    """
    if not isinstance(analysis, Mapping):
        raise TypeError("Finalized analysis must be a mapping")
    missing = [
        key
        for key in ("cancer_type", "sample_mode", "purity")
        if key not in analysis
    ]
    if missing:
        raise ValueError(
            "Cannot finalize report without " + ", ".join(sorted(missing))
        )
    cancer_type = analysis["cancer_type"]
    sample_mode = analysis["sample_mode"]
    if not isinstance(cancer_type, str) or not cancer_type.strip():
        raise ValueError("Finalized analysis cancer_type must be a non-empty string")
    if not isinstance(sample_mode, str) or sample_mode not in {
        "solid",
        "mesenchymal",
        "heme",
        "embryonal",
        "pure",
    }:
        raise ValueError(
            "Finalized analysis sample_mode must be solid, mesenchymal, heme, "
            "embryonal, or pure"
        )
    cancer_name = analysis.get("cancer_name") or cancer_type
    if not isinstance(cancer_name, str):
        raise TypeError("Finalized analysis cancer_name must be a string")
    if sample_id is not None and not isinstance(sample_id, str):
        raise TypeError("Report sample_id must be a string or None")
    reference_cancer_type = analysis.get("reference_cancer_type")
    if reference_cancer_type is not None and not isinstance(
        reference_cancer_type, str
    ):
        raise TypeError("Finalized analysis reference_cancer_type must be a string")
    excluded = {cancer_type}
    if reference_cancer_type:
        excluded.add(reference_cancer_type)
    alternatives = []
    for entry in analysis.get("top_cancers") or ():
        try:
            code, frac = entry
        except (TypeError, ValueError):
            raise TypeError(
                "Finalized analysis top_cancers entries must be (code, support) pairs"
            ) from None
        if not isinstance(code, str):
            raise TypeError("Finalized analysis candidate codes must be strings")
        if isinstance(frac, bool) or not isinstance(frac, Real):
            raise TypeError("Finalized analysis candidate support must be numeric")
        if code in excluded:
            continue
        alternatives.append((code, float(frac)))

    return ReportView(
        cancer_type=cancer_type,
        cancer_type_name=cancer_name,
        call_confidence=compute_call_confidence(analysis),
        cancer_type_alternatives=tuple(alternatives),
        purity=Purity.from_analysis(analysis),
        sample_mode=sample_mode,
        sample_id=sample_id,
    )
