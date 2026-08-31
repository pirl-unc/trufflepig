"""Cancer-type decisions from background-separated tumor expression.

The decomposition engine fits background components and leaves tumor as the
non-negative residual.  Its aggregate ``score`` deliberately includes the
upstream cancer rank, so that score is useful for choosing a subtraction model
but is not independent cancer-identity evidence.

This module keeps those roles separate. It evaluates curated positive and
negative tumor programs after background subtraction, groups fits that
represent the same background model, and returns one :class:`CancerTypeDecision`.
No decomposition score, ranker support, or candidate purity is reused in that
decision.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from ..analyze.cancer_type_context import cancer_type_tree_relationship
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


@dataclass(frozen=True)
class CancerTypeDecision:
    """What the tumor expression supports after modeled background is removed.

    This is the public contract between decomposition, cancer-type selection,
    and reporting. Callers do not interpret evaluator status strings or inspect
    background-model rows to decide whether a label may change.
    """

    status: str
    supported_code: str = ""
    current_code: str = ""
    panel_code: str = ""
    ontology_code: str = ""
    decision_basis: str = ""
    background_separation_confirmed: bool = False
    selection_allowed: bool = True
    block_reason: str = ""
    models_evaluated: int = 0
    realizations_evaluated: int = 0
    background_models: tuple[Mapping[str, Any], ...] = ()
    background_attributed_genes: tuple[str, ...] = ()
    separated_background_models: int = 0
    sample_mode: str = ""
    supported_code_mode: str = ""
    reason: str = ""
    refit_accepted: bool = False
    selected_by_consensus: bool = False
    consensus_required_background_separation: bool = False

    @classmethod
    def from_dict(
        cls,
        evidence: Mapping[str, Any] | None,
        *,
        current_code: str | None = None,
    ) -> "CancerTypeDecision":
        data = dict(evidence or {})
        status = _clean(data.get("status")) or "not_evaluable"
        return cls(
            status=status,
            supported_code=_clean(data.get("supported_code")),
            current_code=_clean(current_code or data.get("current_code")),
            panel_code=_clean(data.get("panel_code")),
            ontology_code=_clean(data.get("ontology_code")),
            decision_basis=_clean(data.get("decision_basis")),
            background_separation_confirmed=bool(
                data.get("background_separation_confirmed")
            ),
            selection_allowed=bool(data.get("selection_allowed", True)),
            block_reason=_clean(data.get("block_reason")),
            models_evaluated=int(data.get("models_evaluated") or 0),
            realizations_evaluated=int(
                data.get("realizations_evaluated") or 0
            ),
            background_models=tuple(
                dict(row)
                for row in (data.get("background_models") or ())
                if isinstance(row, Mapping)
            ),
            background_attributed_genes=tuple(
                str(gene)
                for gene in (
                    data.get("background_attributed_genes") or ()
                )
                if str(gene).strip()
            ),
            separated_background_models=int(
                data.get("separated_background_models") or 0
            ),
            sample_mode=_clean(data.get("sample_mode")),
            supported_code_mode=_clean(data.get("supported_code_mode")),
            reason=_clean(data.get("reason")),
        )

    @classmethod
    def from_analysis(
        cls,
        analysis: Mapping[str, Any],
        report_code: str | None = None,
    ) -> "CancerTypeDecision":
        report = _clean(
            report_code
            or analysis.get("report_scope_cancer_type")
            or analysis.get("cancer_type")
        )
        decision = cls.from_dict(
            analysis.get("cancer_type_decision"),
            current_code=report,
        )
        selected = (
            (analysis.get("cancer_type_evidence") or {}).get("selected")
            or {}
        )
        consensus = selected.get("entity_evidence_consensus") or {}
        return replace(
            decision,
            refit_accepted=bool(
                (analysis.get("cancer_type_decision_refit") or {}).get(
                    "accepted"
                )
                is True
            ),
            selected_by_consensus=(
                selected.get("selected_by") == "entity_evidence_consensus"
            ),
            consensus_required_background_separation=bool(
                consensus.get("decomposition_decision_was_decisive")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used in analysis artifacts."""

        return {
            "status": self.status,
            "supported_code": self.supported_code or None,
            "current_code": self.current_code or None,
            "relationship": self.relationship,
            "panel_code": self.panel_code or None,
            "ontology_code": self.ontology_code or None,
            "decision_basis": self.decision_basis or None,
            "background_separation_confirmed": (
                self.background_separation_confirmed
            ),
            "selection_allowed": self.selection_allowed,
            "block_reason": self.block_reason or None,
            "models_evaluated": self.models_evaluated,
            "realizations_evaluated": self.realizations_evaluated,
            "background_models": [dict(row) for row in self.background_models],
            "background_attributed_genes": list(
                self.background_attributed_genes
            ),
            "separated_background_models": self.separated_background_models,
            "sample_mode": self.sample_mode or None,
            "supported_code_mode": self.supported_code_mode or None,
            "reason": self.reason or None,
        }

    @property
    def is_resolved(self) -> bool:
        return bool(self.status == "resolved" and self.supported_code)

    @property
    def relationship(self) -> str:
        """How the supported code relates to the current report code."""

        return cancer_type_tree_relationship(
            self.current_code,
            self.supported_code,
        )

    @property
    def structurally_consistent(self) -> bool:
        """Whether every evaluated background model supports one branch."""

        return bool(
            self.supported_code
            and self.background_models
            and all(
                int(model.get("realizations") or 0) > 0
                and cancer_type_tree_relationship(
                    self.supported_code,
                    model.get("candidate_code"),
                )
                in {"same", "ancestor", "descendant"}
                for model in self.background_models
            )
        )

    @property
    def has_cancer_type_support(self) -> bool:
        """Whether a marker or ontology program supports the resolved code."""

        # The evaluator sets this only after complete marker and ontology
        # programs agree across candidate-independent background models.
        if self.background_separation_confirmed:
            return True
        panel_supports = bool(
            self.panel_code
            and cancer_type_tree_relationship(
                self.supported_code,
                self.panel_code,
            )
            in {"same", "ancestor", "descendant"}
        )
        ontology_supports = bool(
            self.ontology_code
            and self.structurally_consistent
            and cancer_type_tree_relationship(
                self.supported_code,
                self.ontology_code,
            )
            in {"same", "ancestor", "descendant"}
        )
        return panel_supports or ontology_supports

    @property
    def proposed_code(self) -> str:
        """Cancer code this decision is allowed to propose, if any.

        A broader parent may replace an unjustified child only when explicit
        background separation established that parent. A narrower child is not
        inferred from parent-compatible evidence.
        """

        if (
            not self.is_resolved
            or not self.selection_allowed
            or not self.has_cancer_type_support
        ):
            return ""
        if self.relationship in {"independent", "sibling", "unknown"}:
            return self.supported_code
        if (
            self.relationship == "ancestor"
            and self.background_separation_confirmed
        ):
            return self.supported_code
        return ""

    def supports(self, cancer_code: str) -> bool:
        """Whether this decision directly supports ``cancer_code``."""

        return bool(
            self.is_resolved
            and self.selection_allowed
            and self.has_cancer_type_support
            and self.supported_code == _clean(cancer_code)
        )

    def background_separation_supports(self, cancer_code: str) -> bool:
        """Whether independent background models support ``cancer_code``."""

        return bool(
            self.background_separation_confirmed
            and self.is_resolved
            and self.selection_allowed
            and self.supported_code == _clean(cancer_code)
        )

    def confirms(
        self,
        report_code: str,
        *,
        require_background_separation: bool = False,
    ) -> bool:
        """Whether a final refit establishes at least ``report_code``."""

        relationship = cancer_type_tree_relationship(
            report_code,
            self.supported_code,
        )
        return bool(
            self.is_resolved
            and self.selection_allowed
            and self.has_cancer_type_support
            and relationship in {"same", "descendant"}
            and (
                not require_background_separation
                or self.background_separation_confirmed
            )
        )

    def block_selection(self, reason: str) -> "CancerTypeDecision":
        """Return the same audit result with report-label selection disabled."""

        return replace(
            self,
            selection_allowed=False,
            block_reason=_clean(reason),
        )

    @property
    def is_selection_basis(self) -> bool:
        """Whether this decision established the final report scope."""

        return bool(
            self.relationship in {"same", "descendant"}
            and self.background_separation_supports(self.supported_code)
            and self.refit_accepted
            and self.selected_by_consensus
            and self.consensus_required_background_separation
        )

    @property
    def refit_confirmed(self) -> bool:
        """Whether the final decomposition retained this decision."""

        return bool(
            self.supported_code
            and self.background_separation_supports(self.supported_code)
            and self.refit_accepted
        )

    def for_sample_mode(self, sample_mode: str) -> "CancerTypeDecision":
        """Allow selection only in the decomposition regime that produced it."""

        mode = _clean(sample_mode)
        if not self.supported_code:
            return replace(
                self,
                sample_mode=mode,
            )
        if mode not in {"solid", "heme"}:
            return replace(
                self,
                selection_allowed=False,
                block_reason=f"unsupported decomposition mode: {mode or 'missing'}",
                sample_mode=mode,
            )

        try:
            from ..expression_decomposition import resolve_mode

            candidate_mode, _routing, _type_code = resolve_mode(
                self.supported_code
            )
        except (ImportError, KeyError, TypeError, ValueError):
            candidate_mode = None

        supported_mode = (
            "heme"
            if candidate_mode == "heme"
            else "solid"
            if candidate_mode
            else "unknown"
        )
        compatible = supported_mode == mode
        return replace(
            self,
            selection_allowed=compatible,
            block_reason=(
                ""
                if compatible
                else f"{self.supported_code} requires {supported_mode} "
                f"decomposition, but this decision used {mode} mode"
            ),
            sample_mode=mode,
            supported_code_mode=supported_mode,
        )


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


def _semantic_background_component(component: str) -> str:
    """Collapse identity-only reference names to their biological component."""

    value = _clean(component)
    return value.removesuffix("_identity")


def _dominant_identity_backgrounds(result) -> tuple[str, ...]:
    """Return the largest fitted normal component in an identity-only model."""

    if _clean(getattr(result, "model_role", "")) != "identity_background":
        return ()
    fractions = getattr(result, "fractions", None) or {}
    backgrounds = {
        _semantic_background_component(component): float(value or 0.0)
        for component, value in fractions.items()
        if _clean(component) != "tumor" and float(value or 0.0) > 0.0
    }
    if not backgrounds:
        return ()
    largest = max(backgrounds.values())
    return tuple(
        sorted(component for component, value in backgrounds.items() if value == largest)
    )


def _background_attributed_panel_symbols(
    panels: Iterable[Any],
    components: Iterable[str],
) -> tuple[str, ...]:
    component_set = {
        _semantic_background_component(component) for component in components
    }
    return tuple(
        sorted(
            {
                _clean(symbol).upper()
                for panel in panels
                for symbol, component in getattr(
                    panel,
                    "background_attribution_markers",
                    (),
                )
                if _semantic_background_component(component) in component_set
                and _clean(symbol)
            }
        )
    )


def _background_only_panel_candidate(
    evidence: Iterable[Any],
    panels_by_name: Mapping[str, Any],
) -> str:
    """Return a complete panel blocked only by attributable background genes."""

    candidates: list[str] = []
    for row in evidence:
        panel = panels_by_name.get(_clean(getattr(row, "panel_name", "")))
        attributable = {
            _clean(symbol).upper()
            for symbol, _component in getattr(
                panel,
                "background_attribution_markers",
                (),
            )
        } if panel is not None else set()
        violations = {
            _clean(item[0]).upper()
            for item in (getattr(row, "low_violations", ()) or ())
            if item
        }
        if (
            getattr(row, "obligate_passed", False)
            and getattr(row, "identity_marker_groups_passed", True)
            and getattr(row, "high_hits", ())
            and not getattr(row, "high_misses", ())
            and violations
            and violations.issubset(attributable)
        ):
            parent = _clean(getattr(row, "parent_cohort", ""))
            if parent:
                candidates.append(parent)
    return candidates[0] if len(set(candidates)) == 1 else ""


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


def _panel_eligible_for_pareto(row: Any) -> bool:
    """Whether a panel has affirmative identity evidence to compare.

    A failed tissue-identity gate returns an intentionally empty score row.
    Treating that row as a valid 0/0 high-marker program (and a vacuous 1/1
    expected-low program) can manufacture a Pareto winner from an all-zero
    residual. Pareto ordering refines incomplete *positive* programs; it must
    never turn a gate failure or complete lack of expected-high evidence into
    an identity vote.
    """

    return bool(
        getattr(row, "obligate_passed", False)
        and getattr(row, "identity_marker_groups_passed", True)
        and (getattr(row, "high_hits", ()) or ())
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
        if not _panel_eligible_for_pareto(row):
            continue
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


def _conflicting_candidates(values: Iterable[str | None]) -> tuple[str, ...]:
    """Return distinct non-empty identities when an evidence axis disagrees."""

    candidates = tuple(sorted({_clean(value) for value in values if _clean(value)}))
    return candidates if len(candidates) > 1 else ()


def decide_cancer_type_from_decomposition(
    decomposition_results: Iterable[Any],
    *,
    candidate_codes: Iterable[str],
    current_code: str | None = None,
) -> CancerTypeDecision:
    """Decide which cancer type the background-separated tumor supports.

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
        "supported_code": None,
        "current_code": _clean(current_code) or None,
        "models_evaluated": 0,
        "realizations_evaluated": 0,
        "background_models": [],
        "reason": "no usable decomposition residuals or candidate programs",
    }
    if not candidates:
        return CancerTypeDecision.from_dict(empty)

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

    panels_by_name = {
        _clean(getattr(panel, "name", "")): panel
        for panel in LINEAGE_PANELS
    }

    grouped: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(
        list
    )
    for result in decomposition_results or ():
        if not _usable_result(result):
            continue
        by_id, by_symbol = _residual_views(getattr(result, "gene_attribution", None))
        if not by_id and not by_symbol:
            continue

        dominant_backgrounds = _dominant_identity_backgrounds(result)
        attributed_symbols = _background_attributed_panel_symbols(
            LINEAGE_PANELS,
            dominant_backgrounds,
        )
        if attributed_symbols:
            from ..common import panel_symbols_to_gene_ids

            symbol_to_gene = panel_symbols_to_gene_ids(attributed_symbols)
            for symbol in attributed_symbols:
                by_symbol[symbol] = 0.0
                gene_id = _clean(symbol_to_gene.get(symbol)).split(".", 1)[0]
                if gene_id:
                    by_id[gene_id] = 0.0

        panel_decision: dict[str, Any] = {}
        background_only_panel_candidate = ""
        if evaluate_panels is not None and complete_program_entity_decision is not None:
            panel_rows = _active_panel_evidence(
                evaluate_panels(
                    LINEAGE_PANELS,
                    by_id,
                ),
                candidates,
            )
            if panel_rows:
                background_only_panel_candidate = _background_only_panel_candidate(
                    panel_rows,
                    panels_by_name,
                )
                panel_decision = dict(
                    complete_program_entity_decision(panel_rows)
                )
                if panel_decision.get("decisive"):
                    panel_decision["decision_basis"] = "complete_program"
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
                "panel_decision_basis": _clean(
                    panel_decision.get("decision_basis")
                ),
                "panel_reason": _clean(panel_decision.get("reason")),
                "background_only_panel_candidate": background_only_panel_candidate,
                "background_attributed_expected_low_genes": list(
                    attributed_symbols
                ),
                "dominant_identity_backgrounds": list(dominant_backgrounds),
                "decomposition_model_role": _clean(
                    getattr(result, "model_role", "report_decomposition")
                ),
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
        return CancerTypeDecision.from_dict(empty)

    background_models: list[dict[str, Any]] = []
    for (template, components), rows in sorted(grouped.items()):
        panel_votes = tuple(row.get("panel_candidate") for row in rows)
        panel_or_background_votes = tuple(
            row.get("panel_candidate")
            or row.get("background_only_panel_candidate")
            for row in rows
        )
        complete_panel_votes = tuple(
            row.get("panel_candidate")
            if row.get("panel_decision_basis") == "complete_program"
            else ""
            for row in rows
        )
        complete_panel_or_background_votes = tuple(
            (
                row.get("panel_candidate")
                if row.get("panel_decision_basis") == "complete_program"
                else row.get("background_only_panel_candidate")
            )
            for row in rows
        )
        ontology_votes = tuple(row.get("ontology_candidate") for row in rows)
        panel_candidate = _unanimous(panel_votes)
        panel_or_background_candidate = _unanimous(panel_or_background_votes)
        complete_panel_candidate = _unanimous(complete_panel_votes)
        complete_panel_or_background_candidate = _unanimous(
            complete_panel_or_background_votes
        )
        ontology_candidate = _unanimous(ontology_votes)
        panel_conflicts = _conflicting_candidates(panel_votes)
        ontology_conflicts = _conflicting_candidates(ontology_votes)
        identity_only_model = bool(rows) and all(
            row.get("decomposition_model_role") == "identity_background"
            for row in rows
        )
        complete_identity_panel_candidate = (
            complete_panel_candidate or complete_panel_or_background_candidate
        )
        if panel_conflicts or ontology_conflicts:
            # A disagreeing axis is contradictory evidence, not a missing
            # axis. It must remain an abstention even when the other evidence
            # view is unanimous across the same candidate-specific residuals.
            model_candidate = ""
        elif identity_only_model:
            # Candidate-independent subtraction is intentionally held to the
            # strongest standard. A normal reference can leave tiny residuals
            # that Pareto-rank one ontology program by accident; require a
            # complete curated panel as well. A bulk panel blocked solely by a
            # named expected-low background gene may stand in only when the
            # structural beam independently models that source.
            if (
                complete_identity_panel_candidate
                and ontology_candidate
                and cancer_codes_entity_compatible(
                    complete_identity_panel_candidate,
                    ontology_candidate,
                )
            ):
                model_candidate = (
                    ontology_candidate
                    if _is_ancestor(
                        complete_identity_panel_candidate,
                        ontology_candidate,
                    )
                    else complete_identity_panel_candidate
                )
            else:
                model_candidate = ""
        elif panel_candidate and ontology_candidate:
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
                "model_role": (
                    "identity_background"
                    if identity_only_model
                    else "report_decomposition"
                ),
                "realizations": len(rows),
                "candidate_code": model_candidate or None,
                "panel_candidate": panel_candidate or None,
                "panel_or_background_candidate": (
                    panel_or_background_candidate or None
                ),
                "complete_panel_candidate": complete_panel_candidate or None,
                "complete_panel_or_background_candidate": (
                    complete_panel_or_background_candidate or None
                ),
                "background_attributed_expected_low_genes": sorted(
                    {
                        gene
                        for row in rows
                        for gene in row.get(
                            "background_attributed_expected_low_genes",
                            (),
                        )
                    }
                ),
                "ontology_candidate": ontology_candidate or None,
                "panel_conflicting_candidates": list(panel_conflicts),
                "ontology_conflicting_candidates": list(ontology_conflicts),
                "rows": rows,
            }
        )

    invariant_panel_candidate = _unanimous(
        row.get("panel_candidate") for row in background_models
    )
    invariant_panel_or_background_candidate = _unanimous(
        row.get("panel_or_background_candidate")
        for row in background_models
    )
    background_attributed_genes = sorted(
        {
            gene
            for row in background_models
            for gene in row.get(
                "background_attributed_expected_low_genes",
                (),
            )
        }
    )
    if (
        not invariant_panel_candidate
        and invariant_panel_or_background_candidate
        and background_attributed_genes
    ):
        invariant_panel_candidate = invariant_panel_or_background_candidate
    invariant_ontology_candidate = _unanimous(
        row.get("ontology_candidate") for row in background_models
    )
    identity_background_models = [
        row
        for row in background_models
        if row.get("model_role") == "identity_background"
    ]
    separated_panel_code = _unanimous(
        row.get("complete_panel_or_background_candidate")
        for row in identity_background_models
    )
    separated_ontology_code = _unanimous(
        row.get("ontology_candidate") for row in identity_background_models
    )
    separated_code = _unanimous(
        row.get("candidate_code") for row in identity_background_models
    )
    background_separation_confirmed = bool(
        separated_code
        and separated_panel_code
        and separated_ontology_code
        and cancer_codes_entity_compatible(
            separated_code,
            separated_panel_code,
        )
        and cancer_codes_entity_compatible(
            separated_code,
            separated_ontology_code,
        )
        and background_attributed_genes
    )
    if not invariant_panel_candidate and background_separation_confirmed:
        invariant_panel_candidate = separated_panel_code
    candidate = _unanimous(
        row.get("candidate_code") for row in background_models
    )
    realizations = sum(int(row["realizations"]) for row in background_models)
    if not candidate:
        return CancerTypeDecision.from_dict({
            **empty,
            "status": "ambiguous",
            "models_evaluated": len(background_models),
            "realizations_evaluated": realizations,
            "background_models": background_models,
            "panel_code": invariant_panel_candidate or None,
            "ontology_code": invariant_ontology_candidate or None,
            "reason": (
                "the supported cancer type differed across usable "
                "background model and candidate-specific realization"
            ),
        })

    current = _clean(current_code)
    return CancerTypeDecision.from_dict({
        "status": "resolved",
        "supported_code": candidate,
        "panel_code": invariant_panel_candidate or None,
        "ontology_code": invariant_ontology_candidate or None,
        "decision_basis": (
            "panel_and_ontology"
            if invariant_panel_candidate and invariant_ontology_candidate
            else "panel_program"
            if invariant_panel_candidate
            else "ontology_program"
        ),
        "background_separation_confirmed": background_separation_confirmed,
        "separated_background_models": len(identity_background_models),
        "background_attributed_genes": background_attributed_genes,
        "current_code": current or None,
        "models_evaluated": len(background_models),
        "realizations_evaluated": realizations,
        "background_models": background_models,
        "reason": (
            f"{candidate} remained the only supported cancer type across "
            "every usable background model and realization"
        ),
    })
