"""Tumor-identity evidence from decomposition residuals.

The decomposition engine fits background components and leaves tumor as the
non-negative residual.  Its aggregate ``score`` deliberately includes the
upstream cancer rank, so that score is useful for choosing a subtraction model
but is not independent cancer-identity evidence.

This module keeps those roles separate.  It evaluates curated positive and
negative tumor programs on every usable residual, groups candidate-specific
fits that represent the same structural background model, and only nominates
an identity when the conclusion is invariant across every realization.  No
decomposition score, ranker support, or candidate purity is used in the
identity decision.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from ..cancer_ontology import cancer_codes_entity_compatible, registry_parent_code
from ..tumor_type_ontology import tumor_type_ontology, tumor_type_sanity_check


_ATTRIBUTION_METADATA_COLUMNS = frozenset(
    {
        "gene_id",
        "symbol",
        "observed_tpm",
        "overexplained_tpm",
        "tumor_fraction_of_total",
    }
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def scope_residual_identity_to_decomposition_mode(
    evidence: Mapping[str, Any] | None,
    *,
    sample_mode: str,
) -> dict[str, Any]:
    """Make residual evidence selectable only in the regime that produced it.

    Heme and solid decompositions subtract different biological populations,
    so their residuals are not interchangeable. Pure-population mode describes
    sample preparation rather than lineage and therefore remains unrestricted.
    """

    scoped = dict(evidence or {})
    candidate_code = _clean(scoped.get("candidate_code"))
    mode = _clean(sample_mode)
    if not candidate_code or mode not in {"solid", "heme"}:
        scoped["adjudication_eligible"] = True
        scoped.pop("adjudication_blocker", None)
        return scoped

    try:
        from ..expression_decomposition import resolve_mode

        candidate_mode, _routing, _type_code = resolve_mode(candidate_code)
    except (ImportError, KeyError, TypeError, ValueError):
        candidate_mode = None

    candidate_regime = (
        "heme"
        if candidate_mode == "heme"
        else "solid"
        if candidate_mode
        else "unknown"
    )
    compatible = candidate_regime == mode
    scoped["adjudication_eligible"] = compatible
    scoped["decomposition_sample_mode"] = mode
    scoped["candidate_sample_mode"] = candidate_regime
    if compatible:
        scoped.pop("adjudication_blocker", None)
    else:
        scoped["adjudication_blocker"] = (
            f"{candidate_code} requires {candidate_regime} decomposition, "
            f"but the residual was generated in {mode} mode"
        )
    return scoped


def _residual_views(gene_attribution) -> tuple[dict[str, float], dict[str, float]]:
    """Return tumor-primary residual views by gene ID and symbol.

    A large absolute residual can still be background leakage when the fitted
    background explains more of that gene than the tumor does. Identity
    programs therefore see a gene only when tumor is its primary attribution.
    This is an exact attribution comparison, not another expression cutoff.
    """

    if gene_attribution is None or getattr(gene_attribution, "empty", True):
        return {}, {}
    by_id: dict[str, float] = {}
    by_symbol: dict[str, float] = {}
    component_columns = [
        column
        for column in getattr(gene_attribution, "columns", ())
        if _clean(column)
        and _clean(column) not in _ATTRIBUTION_METADATA_COLUMNS
        and _clean(column) != "tumor"
    ]
    for row in gene_attribution.itertuples(index=False):
        tumor_value = float(getattr(row, "tumor", 0.0) or 0.0)
        background_value = sum(
            float(getattr(row, column, 0.0) or 0.0)
            for column in component_columns
        )
        value = tumor_value if tumor_value > background_value else 0.0
        gene_id = _clean(getattr(row, "gene_id", "")).split(".", 1)[0]
        symbol = _clean(getattr(row, "symbol", "")).upper()
        if gene_id:
            by_id[gene_id] = max(value, by_id.get(gene_id, 0.0))
        if symbol:
            by_symbol[symbol] = max(value, by_symbol.get(symbol, 0.0))
    return by_id, by_symbol


def _background_model_key(result) -> tuple[str, tuple[str, ...]]:
    """Structural model identity, excluding the candidate cancer label."""

    attribution = getattr(result, "gene_attribution", None)
    columns = getattr(attribution, "columns", ())
    components = tuple(
        sorted(
            _clean(column)
            for column in columns
            if _clean(column)
            and _clean(column) not in _ATTRIBUTION_METADATA_COLUMNS
            and _clean(column) != "tumor"
        )
    )
    return _clean(getattr(result, "template", "")), components


def _usable_result(result) -> bool:
    attribution = getattr(result, "gene_attribution", None)
    if attribution is None or getattr(attribution, "empty", True):
        return False
    if "tumor" not in set(getattr(attribution, "columns", ())):
        return False
    warnings = {_clean(row) for row in (getattr(result, "warnings", None) or ())}
    if "No non-tumor components in template" in warnings:
        return False
    site = getattr(result, "site_evidence", None) or {}
    template = _clean(getattr(result, "template", ""))
    return not (
        template.startswith("met_")
        and site
        and not bool(site.get("site_supported", False))
    )


def _complete_ontology_program(sanity: Mapping[str, Any]) -> bool:
    """Whether every existing high/low expectation is satisfied."""

    expected_high = list(sanity.get("expected_high") or ())
    detected_high = list(sanity.get("expected_high_detected") or ())
    return bool(
        expected_high
        and len(detected_high) == len(expected_high)
        and not list(sanity.get("expected_low_present") or ())
    )


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    current = _clean(descendant)
    seen: set[str] = set()
    while current and current not in seen:
        if current == _clean(ancestor):
            return True
        seen.add(current)
        current = registry_parent_code(current)
    return False


def _most_specific_codes(codes: Iterable[str]) -> tuple[str, ...]:
    """Drop a complete parent when a complete descendant is also present."""

    unique = tuple(dict.fromkeys(_clean(code) for code in codes if _clean(code)))
    return tuple(
        code
        for code in unique
        if not any(
            other != code and _is_ancestor(code, other)
            for other in unique
        )
    )


def _ancestor_path(code: str) -> tuple[str, ...]:
    current = _clean(code)
    seen: set[str] = set()
    path: list[str] = []
    while current and current not in seen:
        seen.add(current)
        path.append(current)
        current = registry_parent_code(current)
    return tuple(path)


def _common_registry_ancestor(codes: Iterable[str]) -> str:
    """Deepest common registry node, used as an explicit abstention label."""

    unique = tuple(dict.fromkeys(_clean(code) for code in codes if _clean(code)))
    if not unique:
        return ""
    paths = [_ancestor_path(code) for code in unique]
    shared = set(paths[0]).intersection(*(set(path) for path in paths[1:]))
    return next((code for code in paths[0] if code in shared), "")


def _ontology_program_resolution(code: str) -> str:
    """Roll an ontology result up when sibling leaves share one program.

    Expression cannot justify a leaf distinction when the ontology assigns
    exactly the same expected-high and expected-low genes to multiple siblings.
    Preserve the deepest common registry parent instead of letting candidate
    enumeration arbitrarily choose one sibling.
    """

    candidate = _clean(code)
    entry = tumor_type_ontology().get(candidate)
    if entry is None or not entry.parent_code:
        return candidate
    signature = (
        frozenset(entry.expected_high_genes),
        frozenset(entry.expected_low_genes),
    )
    indistinguishable = [
        sibling.code
        for sibling in tumor_type_ontology().values()
        if sibling.parent_code == entry.parent_code
        and (
            frozenset(sibling.expected_high_genes),
            frozenset(sibling.expected_low_genes),
        )
        == signature
    ]
    return (
        _common_registry_ancestor(indistinguishable)
        if len(indistinguishable) > 1
        else candidate
    )


def _active_panel_evidence(
    evidence: Iterable[Any],
    candidate_codes: tuple[str, ...],
) -> tuple[Any, ...]:
    return tuple(
        row
        for row in evidence
        if any(
            cancer_codes_entity_compatible(
                _clean(getattr(row, "parent_cohort", "")),
                code,
            )
            for code in candidate_codes
        )
    )


def _fraction_at_least(
    left_numerator: int,
    left_denominator: int,
    right_numerator: int,
    right_denominator: int,
) -> bool:
    """Compare fractions exactly, without another floating-point cutoff."""

    left_denominator = max(1, int(left_denominator))
    right_denominator = max(1, int(right_denominator))
    return (
        int(left_numerator) * right_denominator
        >= int(right_numerator) * left_denominator
    )


def _panel_program_vector(row: Any) -> tuple[bool, int, int, int, int]:
    high_hits = len(getattr(row, "high_hits", ()) or ())
    high_total = high_hits + len(getattr(row, "high_misses", ()) or ())
    low_passes = len(getattr(row, "low_passes", ()) or ())
    low_total = low_passes + len(getattr(row, "low_violations", ()) or ())
    if low_total == 0:
        low_passes, low_total = 1, 1
    return (
        bool(getattr(row, "obligate_passed", False)),
        high_hits,
        high_total,
        low_passes,
        low_total,
    )


def _program_vector_dominates(
    left: tuple[bool, int, int, int, int],
    right: tuple[bool, int, int, int, int],
) -> bool:
    left_obligate, left_high, left_high_total, left_low, left_low_total = left
    right_obligate, right_high, right_high_total, right_low, right_low_total = right
    obligate_at_least = left_obligate or not right_obligate
    high_at_least = _fraction_at_least(
        left_high, left_high_total, right_high, right_high_total
    )
    low_at_least = _fraction_at_least(
        left_low, left_low_total, right_low, right_low_total
    )
    strictly_better = bool(
        (left_obligate and not right_obligate)
        or not _fraction_at_least(
            right_high, right_high_total, left_high, left_high_total
        )
        or not _fraction_at_least(
            right_low, right_low_total, left_low, left_low_total
        )
    )
    return obligate_at_least and high_at_least and low_at_least and strictly_better


def _pareto_panel_candidate(evidence: tuple[Any, ...]) -> str:
    """Unique entity whose panel dominates every competing entity panel."""

    if not evidence:
        return ""
    by_parent: dict[str, list[Any]] = defaultdict(list)
    for row in evidence:
        parent = _clean(getattr(row, "parent_cohort", ""))
        if parent:
            by_parent[parent].append(row)
    winners: list[str] = []
    for parent, rows in by_parent.items():
        competitors = [
            row
            for other_parent, other_rows in by_parent.items()
            if other_parent != parent
            for row in other_rows
        ]
        if competitors and any(
            all(
                _program_vector_dominates(
                    _panel_program_vector(row),
                    _panel_program_vector(other),
                )
                for other in competitors
            )
            for row in rows
        ):
            winners.append(parent)
    return winners[0] if len(winners) == 1 else ""


def _ontology_program_vector(
    sanity: Mapping[str, Any],
) -> tuple[bool, int, int, int, int]:
    high_hits = len(sanity.get("expected_high_detected") or ())
    high_total = len(sanity.get("expected_high") or ())
    low_violations = len(sanity.get("expected_low_present") or ())
    low_total = len(sanity.get("expected_low") or ())
    low_passes = max(0, low_total - low_violations)
    if low_total == 0:
        low_passes, low_total = 1, 1
    return high_hits > 0, high_hits, high_total, low_passes, low_total


def _pareto_ontology_candidate(
    ontology: Mapping[str, Mapping[str, Any]],
) -> str:
    """Unique nondominated identity, or a shared-parent abstention."""

    vectors = {
        code: _ontology_program_vector(sanity)
        for code, sanity in ontology.items()
    }
    nondominated = [
        code
        for code, vector in vectors.items()
        if vector[0]
        and not any(
            other != code
            and _program_vector_dominates(other_vector, vector)
            for other, other_vector in vectors.items()
        )
    ]
    if len(nondominated) == 1:
        return nondominated[0]
    return _common_registry_ancestor(nondominated)


def _unanimous(values: Iterable[str | None]) -> str:
    cleaned = tuple(_clean(value) for value in values)
    if not cleaned or any(not value for value in cleaned):
        return ""
    return cleaned[0] if len(set(cleaned)) == 1 else ""


def evaluate_residual_identity(
    decomposition_results: Iterable[Any],
    *,
    candidate_codes: Iterable[str],
    current_code: str | None = None,
) -> dict[str, Any]:
    """Evaluate tumor identity independently across decomposition residuals.

    A candidate is nominated only when every candidate-specific realization of
    each structural background model agrees, and every usable background model
    agrees with the others.  Ambiguity is preserved rather than converted into
    a score or a plurality vote.
    """

    candidates = tuple(
        dict.fromkeys(_clean(code) for code in candidate_codes if _clean(code))
    )
    empty = {
        "status": "not_evaluable",
        "role": "independent_residual_identity",
        "candidate_code": None,
        "current_code": _clean(current_code) or None,
        "models_evaluated": 0,
        "realizations_evaluated": 0,
        "background_models": [],
        "reason": "no usable decomposition residuals or candidate programs",
    }
    if not candidates:
        return empty

    try:
        from ..lineage_panels import (
            LINEAGE_PANELS,
            complete_program_entity_decision,
            evaluate_panels,
        )
    except ImportError:
        LINEAGE_PANELS = ()
        complete_program_entity_decision = None
        evaluate_panels = None

    grouped: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(
        list
    )
    for result in decomposition_results or ():
        if not _usable_result(result):
            continue
        by_id, by_symbol = _residual_views(getattr(result, "gene_attribution", None))
        if not by_id and not by_symbol:
            continue

        panel_decision: dict[str, Any] = {}
        if evaluate_panels is not None and complete_program_entity_decision is not None:
            panel_rows = _active_panel_evidence(
                evaluate_panels(
                    LINEAGE_PANELS,
                    by_id,
                ),
                candidates,
            )
            if panel_rows:
                panel_decision = dict(
                    complete_program_entity_decision(panel_rows)
                )
                if not panel_decision.get("decisive"):
                    pareto_panel_candidate = _pareto_panel_candidate(panel_rows)
                    if pareto_panel_candidate:
                        panel_decision = {
                            **panel_decision,
                            "decisive": True,
                            "top_parent_cohort": pareto_panel_candidate,
                            "reason": (
                                "one residual marker program Pareto-dominates "
                                "every active competing entity program"
                            ),
                            "decision_basis": "pareto_dominance",
                        }

        ontology: dict[str, dict[str, Any]] = {
            code: tumor_type_sanity_check(code, by_symbol) for code in candidates
        }
        complete_codes = _most_specific_codes(
            code
            for code, sanity in ontology.items()
            if _complete_ontology_program(sanity)
        )
        if complete_codes:
            ontology_candidate = (
                complete_codes[0]
                if len(complete_codes) == 1
                else _common_registry_ancestor(complete_codes)
            )
        else:
            ontology_candidate = _pareto_ontology_candidate(ontology)
        raw_ontology_candidate = ontology_candidate
        if ontology_candidate:
            ontology_candidate = _ontology_program_resolution(
                ontology_candidate
            )
        grouped[_background_model_key(result)].append(
            {
                "decomposition_cancer_type": _clean(
                    getattr(result, "cancer_type", "")
                ),
                "template": _clean(getattr(result, "template", "")),
                "purity": float(getattr(result, "purity", 0.0) or 0.0),
                "reconstruction_error": float(
                    getattr(result, "reconstruction_error", 0.0) or 0.0
                ),
                "panel_candidate": (
                    _clean(panel_decision.get("top_parent_cohort"))
                    if panel_decision.get("decisive")
                    else ""
                ),
                "panel_decisive": bool(panel_decision.get("decisive")),
                "panel_reason": _clean(panel_decision.get("reason")),
                "ontology_candidate": ontology_candidate,
                "raw_ontology_candidate": raw_ontology_candidate,
                "ontology_complete_codes": list(complete_codes),
                "ontology_programs": {
                    code: {
                        "status": sanity.get("status"),
                        "complete": _complete_ontology_program(sanity),
                        "expected_high_detected": len(
                            sanity.get("expected_high_detected") or ()
                        ),
                        "expected_high_total": len(
                            sanity.get("expected_high") or ()
                        ),
                        "expected_low_violations": len(
                            sanity.get("expected_low_present") or ()
                        ),
                        "expected_low_total": len(
                            sanity.get("expected_low") or ()
                        ),
                    }
                    for code, sanity in ontology.items()
                },
            }
        )

    if not grouped:
        return empty

    background_models: list[dict[str, Any]] = []
    for (template, components), rows in sorted(grouped.items()):
        panel_candidate = _unanimous(
            row.get("panel_candidate") for row in rows
        )
        ontology_candidate = _unanimous(
            row.get("ontology_candidate") for row in rows
        )
        if panel_candidate and ontology_candidate:
            if cancer_codes_entity_compatible(panel_candidate, ontology_candidate):
                model_candidate = (
                    ontology_candidate
                    if _is_ancestor(panel_candidate, ontology_candidate)
                    else panel_candidate
                )
            else:
                model_candidate = ""
        else:
            model_candidate = panel_candidate or ontology_candidate
        background_models.append(
            {
                "template": template,
                "components": list(components),
                "realizations": len(rows),
                "candidate_code": model_candidate or None,
                "panel_candidate": panel_candidate or None,
                "ontology_candidate": ontology_candidate or None,
                "rows": rows,
            }
        )

    invariant_panel_candidate = _unanimous(
        row.get("panel_candidate") for row in background_models
    )
    invariant_ontology_candidate = _unanimous(
        row.get("ontology_candidate") for row in background_models
    )
    candidate = _unanimous(
        row.get("candidate_code") for row in background_models
    )
    realizations = sum(int(row["realizations"]) for row in background_models)
    if not candidate:
        return {
            **empty,
            "status": "ambiguous",
            "models_evaluated": len(background_models),
            "realizations_evaluated": realizations,
            "background_models": background_models,
            "panel_candidate_code": invariant_panel_candidate or None,
            "ontology_candidate_code": invariant_ontology_candidate or None,
            "reason": (
                "residual identity was not invariant across every usable "
                "background model and candidate-specific realization"
            ),
        }

    current = _clean(current_code)
    return {
        "status": (
            "corroborated"
            if current and cancer_codes_entity_compatible(candidate, current)
            else "candidate"
        ),
        "role": "independent_residual_identity",
        "candidate_code": candidate,
        "panel_candidate_code": invariant_panel_candidate or None,
        "ontology_candidate_code": invariant_ontology_candidate or None,
        "decision_basis": (
            "panel_and_ontology"
            if invariant_panel_candidate and invariant_ontology_candidate
            else "panel_program"
            if invariant_panel_candidate
            else "ontology_program"
        ),
        "current_code": current or None,
        "models_evaluated": len(background_models),
        "realizations_evaluated": realizations,
        "background_models": background_models,
        "reason": (
            f"{candidate} remained the unique residual identity "
            "across every usable background model and realization"
        ),
    }
