"""Cancer-type label context shared across analysis, reports, and plots.

The analyze pipeline often carries more than one cancer label:

- a report label, used for the clinical/curation surface;
- a broad reference label, used by coarse classifier and cohort math;
- sometimes a finer expression reference, when an exact subtype cohort exists.

This module keeps that relationship explicit so downstream code can choose the
right level without re-deriving parent/subtype rules locally.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Mapping


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


@lru_cache(maxsize=1)
def _registry_records() -> dict[str, dict[str, Any]]:
    try:
        from pirlygenes.gene_sets_cancer import cancer_type_registry

        df = cancer_type_registry()
        return {
            _clean(row.get("code")): dict(row)
            for _, row in df.iterrows()
            if _clean(row.get("code"))
        }
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _exact_expression_sources() -> dict[str, frozenset[str]]:
    sources_by_code: dict[str, set[str]] = {}
    try:
        from trufflepig.reference import (
            subtype_deconvolved_expression, tcga_deconvolved_expression,
        )

        tcga = tcga_deconvolved_expression()
        if tcga is not None and "cancer_code" in tcga.columns:
            for code, group in tcga.groupby("cancer_code"):
                code_text = _clean(code)
                if not code_text:
                    continue
                values = {
                    _clean(v)
                    for v in group.get("source_cohort", []).dropna().unique()
                    if _clean(v)
                }
                sources_by_code.setdefault(code_text, set()).update(values or {"TCGA"})

        sub = subtype_deconvolved_expression()
        if sub is not None and "cancer_code" in sub.columns:
            for code, group in sub.groupby("cancer_code"):
                code_text = _clean(code)
                if not code_text:
                    continue
                values = {
                    _clean(v)
                    for v in group.get("source_cohort", []).dropna().unique()
                    if _clean(v)
                }
                subtype_values = (
                    group.get("subtype")
                    .fillna("")
                    .astype(str)
                    .map(_clean)
                    if "subtype" in group.columns
                    else []
                )
                if all(not _clean(v) for v in subtype_values):
                    sources_by_code.setdefault(code_text, set()).update(
                        values or {"subtype_reference"}
                    )
            if "subtype" in sub.columns:
                for subtype, group in sub.dropna(subset=["subtype"]).groupby(
                    "subtype"
                ):
                    subtype_text = _clean(subtype)
                    if not subtype_text:
                        continue
                    values = {
                        _clean(v)
                        for v in group.get("source_cohort", []).dropna().unique()
                        if _clean(v)
                    }
                    sources_by_code.setdefault(subtype_text, set()).update(
                        values or {"subtype_reference"}
                    )
    except Exception:
        pass
    return {code: frozenset(sources) for code, sources in sources_by_code.items()}


def registry_parent_code(code: str | None) -> str:
    row = _registry_records().get(_clean(code), {})
    return _clean(row.get("parent_code"))


def registry_display_name(code: str | None) -> str:
    code_text = _clean(code)
    if not code_text:
        return ""
    row = _registry_records().get(code_text, {})
    return _clean(row.get("name")) or code_text


def cancer_type_context_label(code: str | None) -> str:
    code_text = _clean(code)
    if not code_text:
        return ""
    name = registry_display_name(code_text)
    if name and name.lower() != code_text.lower():
        return f"{code_text} ({name})"
    return code_text


def expression_reference_sources(code: str | None) -> tuple[str, ...]:
    return tuple(sorted(_exact_expression_sources().get(_clean(code), ())))


def has_expression_reference(code: str | None) -> bool:
    return bool(expression_reference_sources(code))


@dataclass(frozen=True)
class CancerTypeContext:
    report_code: str
    reference_code: str
    fine_code: str
    coarse_code: str
    best_expression_code: str
    parent_code: str = ""
    supplied_code: str = ""
    source: str = ""
    relationship: str = "same"
    report_has_expression_ref: bool = False
    reference_has_expression_ref: bool = False
    fine_expression_sources: tuple[str, ...] = ()
    reference_expression_sources: tuple[str, ...] = ()

    def code_for(self, role: str) -> str:
        key = _clean(role).lower().replace("-", "_")
        if key in {"report", "report_scope", "fine", "curation", "therapy"}:
            return self.fine_code or self.report_code or self.coarse_code
        if key in {
            "reference",
            "coarse",
            "cohort",
            "classifier",
            "purity",
            "decomposition",
            "tcga",
            "broad",
        }:
            return self.coarse_code or self.reference_code or self.report_code
        if key in {"expression", "best_expression", "plot_expression"}:
            return (
                self.best_expression_code
                or self.fine_code
                or self.coarse_code
                or self.report_code
            )
        if key == "parent":
            return self.parent_code
        return self.report_code or self.reference_code

    def label_for(self, role: str) -> str:
        return cancer_type_context_label(self.code_for(role))

    @property
    def uses_distinct_reference(self) -> bool:
        return bool(self.report_code and self.reference_code) and (
            self.report_code != self.reference_code
        )

    @property
    def fine_expression_available(self) -> bool:
        return bool(self.fine_code and self.report_has_expression_ref)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["report_label"] = cancer_type_context_label(self.report_code)
        data["reference_label"] = cancer_type_context_label(self.reference_code)
        data["fine_label"] = cancer_type_context_label(self.fine_code)
        data["coarse_label"] = cancer_type_context_label(self.coarse_code)
        data["best_expression_label"] = cancer_type_context_label(self.best_expression_code)
        return data

    def markdown_lines(self) -> list[str]:
        if not self.report_code:
            return []
        lines = [f"- **Report label**: {cancer_type_context_label(self.report_code)}."]
        if self.uses_distinct_reference:
            lines.append(
                f"- **Broad reference context**: {cancer_type_context_label(self.reference_code)} "
                "is used when a step needs a coarse cohort reference."
            )
        else:
            lines.append(
                "- **Broad reference context**: same as report label; no finer "
                "registry label is active."
            )
        if self.parent_code:
            lines.append(
                f"- **Hierarchy**: {cancer_type_context_label(self.report_code)} is modeled "
                f"as a refined label under {cancer_type_context_label(self.parent_code)}."
            )
        if self.best_expression_code:
            if self.best_expression_code == self.report_code:
                lines.append(
                    "- **Best expression reference**: fine-grained expression data "
                    "are available for the report label."
                )
            elif self.uses_distinct_reference:
                lines.append(
                    f"- **Best expression reference**: falls back to "
                    f"{cancer_type_context_label(self.best_expression_code)} because an "
                    "exact fine-grained expression cohort is not available to this "
                    "analysis stage."
                )
        if self.fine_expression_available and self.uses_distinct_reference:
            lines.append(
                "- **Context caveat**: subtype-aware modules may use the fine-grained "
                "reference; coarse pan-reference modules explicitly use the broad "
                "reference."
            )
        return lines


def cancer_type_context_from_analysis(
    analysis: Mapping[str, Any], supplied_cancer_type: str | None = None
) -> CancerTypeContext:
    report_code = _clean(
        analysis.get("report_scope_cancer_type") or analysis.get("cancer_type")
    )
    parent_code = _clean(analysis.get("report_scope_parent_cancer_type"))
    registry_parent = registry_parent_code(report_code)
    if not parent_code and registry_parent:
        parent_code = registry_parent
    reference_code = _clean(
        analysis.get("reference_cancer_type")
        or analysis.get("report_scope_parent_cancer_type")
    )
    if not reference_code:
        reference_code = parent_code or report_code

    report_sources = expression_reference_sources(report_code)
    reference_sources = expression_reference_sources(reference_code)
    if report_sources:
        best_expression_code = report_code
    elif reference_sources:
        best_expression_code = reference_code
    else:
        best_expression_code = reference_code or report_code

    if report_code and reference_code and report_code == reference_code:
        relationship = "same"
    elif parent_code and reference_code and parent_code == reference_code:
        relationship = "fine_child_of_reference"
    elif parent_code:
        relationship = "fine_child_with_independent_reference"
    elif report_code and reference_code:
        relationship = "report_scope_with_independent_reference"
    else:
        relationship = "unresolved"

    return CancerTypeContext(
        report_code=report_code,
        reference_code=reference_code,
        fine_code=report_code,
        coarse_code=reference_code or parent_code or report_code,
        best_expression_code=best_expression_code,
        parent_code=parent_code,
        supplied_code=_clean(
            supplied_cancer_type
            or (analysis.get("analysis_constraints") or {}).get("cancer_type")
        ),
        source=_clean(analysis.get("cancer_type_source")),
        relationship=relationship,
        report_has_expression_ref=bool(report_sources),
        reference_has_expression_ref=bool(reference_sources),
        fine_expression_sources=report_sources,
        reference_expression_sources=reference_sources,
    )


def cancer_type_context_code(analysis: Mapping[str, Any], role: str) -> str:
    return cancer_type_context_from_analysis(analysis).code_for(role)
