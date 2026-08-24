"""Pure analyze orchestration helpers.

This module is where information-flow rules live when they are not tied to
plotting or report rendering. The intent is to make business decisions
testable without executing the full ``analyze`` command.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .models import AnalyzeArtifact, AnalyzeConfig, AnalyzePaths, InputResolution


def resolve_analyze_inputs(
    config: AnalyzeConfig,
    *,
    sniff_input_level: Callable[[str], str],
) -> InputResolution:
    """Resolve gene/transcript inputs exactly once.

    The CLI accepts either a positional path, explicit ``--genes`` /
    ``--transcripts`` paths, or a forced transcript aggregation flag.
    Downstream steps should consume this object instead of re-reading
    option combinations.
    """
    genes = config.genes
    transcripts = config.transcripts
    aggregate = bool(config.aggregate_gene_expression)
    notes: list[str] = []
    input_level = "explicit"

    if not genes and not transcripts:
        if aggregate:
            transcripts = config.input_path
            input_level = "transcript"
        else:
            input_level = sniff_input_level(config.input_path)
            if input_level == "transcript":
                transcripts = config.input_path
                aggregate = True
                notes.append(
                    "[input] Auto-detected transcript-level input, will aggregate to gene level"
                )
            else:
                genes = config.input_path

    if transcripts and not genes:
        aggregate = True

    gene_input = genes or transcripts
    if not gene_input:
        raise ValueError("No expression input was resolved from analyze arguments")

    return InputResolution(
        gene_input=gene_input,
        transcript_input=transcripts if genes else None,
        aggregate_gene_expression=aggregate,
        input_level=input_level,
        notes=tuple(notes),
    )


def build_analyze_paths(
    config: AnalyzeConfig,
    resolution: InputResolution,
    *,
    default_output_dir: Callable[[], str],
    derive_sample_display_id: Callable[[str, str | None], str],
    sanitize_output_basename: Callable[[str | None], str],
) -> AnalyzePaths:
    output_dir = config.output_dir
    # Accept the legacy "pirlygenes-output" sentinel for one release so
    # in-flight callers don't break; the canonical sentinel is now
    # "trufflepig-output".
    if not output_dir or output_dir in {"pirlygenes-output", "trufflepig-output"}:
        output_dir = default_output_dir()
    out_dir = Path(output_dir)

    sample_display_id = derive_sample_display_id(
        resolution.gene_input,
        config.sample_id_value,
    )
    prefix_base = (
        sanitize_output_basename(config.output_image_prefix)
        if config.output_image_prefix
        else sample_display_id
    ) or "sample"

    return AnalyzePaths(
        out_dir=out_dir,
        prefix_base=prefix_base,
        sample_display_id=sample_display_id,
    )


def apply_sample_context_to_purity(analysis: dict[str, Any], sample_context) -> bool:
    """Widen purity CI according to the upstream sample context.

    Returns ``True`` when the analysis dict was modified. This makes the
    sample-context -> purity information flow explicit and unit-testable.
    """
    if sample_context is None or "purity" not in analysis:
        return False
    ci_factor = sample_context.purity_ci_widening_factor()
    if ci_factor <= 1.0:
        return False

    purity_block = analysis["purity"]
    est = purity_block.get("overall_estimate")
    lo = purity_block.get("overall_lower")
    hi = purity_block.get("overall_upper")
    if est is None or lo is None or hi is None:
        return False
    existing_caveat = purity_block.get("degradation_caveat") or {}
    rounded_lo = round(float(lo), 4)
    rounded_hi = round(float(hi), 4)
    prior_widened_lo = existing_caveat.get("widened_lower")
    prior_widened_hi = existing_caveat.get("widened_upper")
    same_factor = purity_block.get("ci_widening_factor") == round(ci_factor, 3)
    # A physical constraint may tighten one side after degradation widening.
    # An unchanged opposite endpoint proves this is the same widened interval,
    # not a replacement interval that still needs the context adjustment.
    constrained_after_widening = bool(
        same_factor
        and (
            (
                rounded_lo == prior_widened_lo
                and prior_widened_hi is not None
                and rounded_hi <= prior_widened_hi
            )
            or (
                rounded_hi == prior_widened_hi
                and prior_widened_lo is not None
                and rounded_lo >= prior_widened_lo
            )
        )
    )
    already_widened = bool(
        same_factor
        and prior_widened_lo == rounded_lo
        and prior_widened_hi == rounded_hi
    )
    if already_widened or constrained_after_widening:
        if constrained_after_widening:
            existing_caveat["widened_lower"] = rounded_lo
            existing_caveat["widened_upper"] = rounded_hi
        return False

    half_lo = max(0.0, est - lo) * ci_factor
    half_hi = max(0.0, hi - est) * ci_factor
    widened_lower = round(max(0.0, est - half_lo), 4)
    widened_upper = round(min(1.0, est + half_hi), 4)
    purity_block["overall_lower"] = widened_lower
    purity_block["overall_upper"] = widened_upper
    purity_block["ci_widening_factor"] = round(ci_factor, 3)
    purity_block["degradation_caveat"] = {
        "severity": sample_context.degradation_severity,
        "index": sample_context.degradation_index,
        "base_lower": round(float(lo), 4),
        "base_upper": round(float(hi), 4),
        "widened_lower": widened_lower,
        "widened_upper": widened_upper,
        "message": (
            f"Purity confidence interval widened x{ci_factor:.2f} "
            f"to reflect {sample_context.degradation_severity} RNA degradation; "
            "tumor-specific genes with long transcripts are under-represented, "
            "biasing the point estimate low and the precision high."
        ),
    }
    return True


def should_adopt_decomposition_purity(classifier_code: str, decomp_result) -> bool:
    """Return whether a decomposition purity can replace classifier purity.

    The decomposition fit supplies a tumor/TME subtraction template. It is
    only a purity estimate when the template is the report entity or one of
    its descendants, has non-tumor components, and exposes a populated purity
    result. A descendant fit can quantify a deliberately broad report parent
    (for example READ under CRC) without promoting that child to the headline.
    """
    if decomp_result is None:
        return False
    decomposition_code = str(getattr(decomp_result, "cancer_type", None) or "")
    classifier_code = str(classifier_code or "")
    if decomposition_code != classifier_code and not _is_descendant_code(
        decomposition_code, classifier_code
    ):
        return False
    fractions = getattr(decomp_result, "fractions", None)
    if isinstance(fractions, dict) and fractions and not any(
        str(component) != "tumor" for component in fractions
    ):
        return False
    n_measured = getattr(decomp_result, "n_measured_in_fit", None)
    if n_measured is not None and int(n_measured or 0) <= 0:
        return False
    warnings = getattr(decomp_result, "warnings", None) or []
    if any("No non-tumor components in template" in warning for warning in warnings):
        return False
    return bool(getattr(decomp_result, "purity_result", None))


def _is_descendant_code(code: str, parent_code: str) -> bool:
    """Whether ``code`` is below ``parent_code`` in the registry hierarchy."""
    if not code or not parent_code or code == parent_code:
        return False
    try:
        from trufflepig.cancer_ontology import registry_parent_code

        seen = set()
        current = code
        while current and current not in seen:
            seen.add(current)
            current = str(registry_parent_code(current) or "")
            if current == parent_code:
                return True
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return False


# Fragility threshold for repeated estimates made by the same estimator under
# different decomposition templates. Values from different estimators are
# alternative biological scenarios, not replicates, and are never pooled into
# this spread.
_DECOMP_PURITY_FRAGILE_SPREAD = 0.35
_TME_OVEREXPLAINED_MARKER = "overexplained by the tme background"


def decomposition_purity_stability(decomp_results, adopted=None) -> dict:
    """Describe template stability without mixing incompatible purity estimators.

    A decomposition tumor fraction is only trustworthy when it is stable across the plausible
    template hypotheses AND the background did not over-absorb tumor signal. Across the local cohort,
    most samples fail one of those: the top hypotheses disagree on tumor fraction by tens of points,
    and the low-purity calls carry a "many genes overexplained by the TME background" warning
    (over-subtraction → deflated purity — the READ 10%-vs-78% failure mode). This records the signals
    so reconciliation can distinguish template sensitivity from estimator disagreement.

    ``DecompositionResult.purity`` is not always the same measurement: a
    matched-normal primary fit may use a lineage-panel tumor fraction while a
    metastatic fit retains the upstream signature estimate. Those values are
    alternative estimator scenarios, not confidence limits for one estimator.

    Returns, for the adopted hypothesis:
      - ``hypothesis_purity_spread``: max−min tumor fraction among hypotheses
        with the same cancer type *and purity source* (template sensitivity).
      - ``estimator_scenarios``: source-preserving estimates for audit/reporting.
      - ``estimator_disagreement``: credible source intervals are disjoint.
      - ``quantitatively_resolved``: false only when estimator disagreement is
        paired with a fragile/over-subtracted adopted fit.
      - ``tme_overexplained``: the adopted fit flagged TME over-subtraction.
      - ``top_hypotheses``: (template, cancer_type, purity, reconstruction_error) for the top few.
      - ``fragile``: wide spread OR over-subtraction — read the point purity as a range, not a number.
    """
    results = list(decomp_results or [])
    if not results:
        return {}
    if adopted is None:
        adopted = results[0]
    adopted_type = getattr(adopted, "cancer_type", None)
    adopted_source = str(
        getattr(adopted, "purity_source", None)
        or (getattr(adopted, "purity_result", None) or {}).get("purity_source")
        or "unspecified"
    )
    # Template sensitivity is defined only within one cancer type and one
    # estimator. A different source is retained below as a named scenario.
    same_type = [r for r in results if getattr(r, "cancer_type", None) == adopted_type]
    same_estimator = [
        r
        for r in same_type
        if str(
            getattr(r, "purity_source", None)
            or (getattr(r, "purity_result", None) or {}).get("purity_source")
            or "unspecified"
        )
        == adopted_source
    ]
    purities = [
        float(getattr(r, "purity", 0.0))
        for r in (same_estimator or [adopted])
        if getattr(r, "purity", None) is not None
    ]
    spread = round(max(purities) - min(purities), 4) if len(purities) >= 2 else 0.0
    hyp_lo = round(min(purities), 4) if purities else None
    hyp_hi = round(max(purities), 4) if purities else None
    warns = " ".join(getattr(adopted, "warnings", None) or []).lower()
    tme_overexplained = _TME_OVEREXPLAINED_MARKER in warns

    # Preserve every estimator as its own scenario. Bounds come from the
    # estimator's own result where available; they are never expanded using a
    # different source. Neutral decomposition priors are useful operationally
    # but are not evidence and therefore cannot create estimator disagreement.
    scenarios_by_source = {}
    for row in same_type:
        result = getattr(row, "purity_result", None) or {}
        source = str(
            getattr(row, "purity_source", None)
            or result.get("purity_source")
            or "unspecified"
        )
        try:
            point = float(getattr(row, "purity"))
        except (TypeError, ValueError):
            continue
        lower = result.get("overall_lower")
        upper = result.get("overall_upper")
        try:
            lower = float(lower) if lower is not None else point
        except (TypeError, ValueError):
            lower = point
        try:
            upper = float(upper) if upper is not None else point
        except (TypeError, ValueError):
            upper = point
        lower, upper = min(lower, point), max(upper, point)
        scenario = scenarios_by_source.setdefault(
            source,
            {
                "source": source,
                "estimate": point,
                "lower": lower,
                "upper": upper,
                "templates": [],
            },
        )
        # Keep the adopted/highest-ranked point as the representative estimate,
        # while spanning repeated measurements made by this same estimator.
        scenario["lower"] = min(float(scenario["lower"]), lower)
        scenario["upper"] = max(float(scenario["upper"]), upper)
        template = str(getattr(row, "template", None) or "")
        if template and template not in scenario["templates"]:
            scenario["templates"].append(template)

    scenarios = sorted(
        scenarios_by_source.values(),
        key=lambda item: (item["source"] != adopted_source, item["source"]),
    )
    try:
        adopted_point = float(getattr(adopted, "purity"))
    except (TypeError, ValueError):
        adopted_point = None
    for scenario in scenarios:
        if scenario["source"] == adopted_source and adopted_point is not None:
            scenario["estimate"] = adopted_point
        scenario["estimate"] = round(float(scenario["estimate"]), 4)
        scenario["lower"] = round(float(scenario["lower"]), 4)
        scenario["upper"] = round(float(scenario["upper"]), 4)
    credible_scenarios = [
        scenario
        for scenario in scenarios
        if scenario["source"] not in {"neutral_decomposition_prior", "override"}
    ]
    estimator_disagreement = any(
        left["upper"] < right["lower"] or right["upper"] < left["lower"]
        for i, left in enumerate(credible_scenarios)
        for right in credible_scenarios[i + 1 :]
    )
    fragile = bool(spread >= _DECOMP_PURITY_FRAGILE_SPREAD or tme_overexplained)
    top = [
        {
            "template": getattr(r, "template", None),
            "cancer_type": getattr(r, "cancer_type", None),
            "purity": round(float(getattr(r, "purity", 0.0) or 0.0), 4),
            "purity_source": str(
                getattr(r, "purity_source", None)
                or (getattr(r, "purity_result", None) or {}).get("purity_source")
                or "unspecified"
            ),
            "reconstruction_error": round(float(getattr(r, "reconstruction_error", 0.0) or 0.0), 4),
        }
        for r in results[:4]
    ]
    return {
        "hypothesis_purity_spread": spread,
        "hypothesis_purity_lo": hyp_lo,
        "hypothesis_purity_hi": hyp_hi,
        "tme_overexplained": tme_overexplained,
        "fragile": fragile,
        "adopted_purity_source": adopted_source,
        "estimator_scenarios": scenarios,
        "estimator_disagreement": estimator_disagreement,
        "quantitatively_resolved": not (fragile and estimator_disagreement),
        "top_hypotheses": top,
    }


# Margin by which a decomposition point purity must differ from an independent signal to count as
# disagreeing with it. Falling back requires disagreeing with EVERY present independent signal.
_DECOMP_INDEPENDENT_DISAGREE_MARGIN = 0.15


def _independent_purity_signals(classifier_purity) -> list[float]:
    """The independent (non-decomposition) purity estimates from the classifier purity dict:
    signature, lineage, and ESTIMATE. Only present, finite, positive values are returned."""
    comp = (classifier_purity or {}).get("components") or {}
    raw = [
        (comp.get("signature") or {}).get("purity"),
        (comp.get("lineage") or {}).get("purity"),
        comp.get("estimate_purity"),
    ]
    out = []
    for v in raw:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f and f > 0.0:  # finite and non-degenerate
            out.append(f)
    return out


def _classifier_purity_trustworthy(classifier_purity) -> bool:
    """Whether the classifier purity is a safe fall-back target for a rejected decomposition purity.

    The TME-overexplained low-purity samples (the dominant fragile case) have a signature purity that
    is SATURATED high — pinned near 1.0 because the tumor-signature genes fire even at low purity —
    which is exactly what the decomposition exists to correct. Falling back to a saturated (or
    degenerate zero-width interval) signature would revert an honest low estimate to a wrong ~100%.
    Only treat a non-saturated purity that carries a real interval as a trustworthy fall-back.
    """
    point = classifier_purity.get("overall_estimate")
    lo = classifier_purity.get("overall_lower")
    hi = classifier_purity.get("overall_upper")
    if not isinstance(point, (int, float)) or not (0.05 < float(point) < 0.95):
        return False
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        return False
    return float(hi) - float(lo) > 1e-6


def reconcile_decomposition_purity(classifier_purity, decomp_purity, stability):
    """Reconcile a fragile decomposition purity against the independent classifier signals.

    Policy (chosen 2026-07-07): when the adopted decomposition purity is *fragile* (its plausible
    template hypotheses disagree widely and/or the TME background over-subtracted):

      * when the disagreement is between different estimator sources, retain
        each as a named scenario and mark purity quantitatively unresolved;
        never merge those scenarios into one interval;
      * otherwise WIDEN the reported interval to span same-estimator template
        sensitivity and mark the purity as template-sensitive; and
      * FALL BACK to the classifier's independent purity when the decomposition point disagrees with
        EVERY independent signal (signature, lineage, ESTIMATE) by more than the margin — a fragile
        estimate that no other method corroborates is not trustworthy as the headline.

    A non-fragile decomposition purity is returned unchanged. Returns ``(action, purity_dict)`` where
    ``action`` is ``"adopt"`` (unchanged), ``"discordant"`` (operational point plus separate
    scenarios), ``"widen"`` (adopt point, widened interval + caveat), or ``"reject"`` (do not adopt;
    use ``classifier_purity``). The returned dict is always safe to adopt.
    """
    stability = stability or {}
    if not isinstance(decomp_purity, dict) or not stability.get("fragile"):
        return "adopt", decomp_purity

    if not stability.get("quantitatively_resolved", True):
        unresolved = dict(decomp_purity)
        unresolved["quantitative_status"] = "discordant_estimators"
        unresolved["estimator_scenarios"] = list(
            stability.get("estimator_scenarios") or []
        )
        unresolved["operational_estimate_only"] = True
        classifier_purity = classifier_purity or {}
        unresolved["pre_decomposition_estimate"] = classifier_purity.get(
            "overall_estimate"
        )
        unresolved["pre_decomposition_lower"] = classifier_purity.get(
            "overall_lower"
        )
        unresolved["pre_decomposition_upper"] = classifier_purity.get(
            "overall_upper"
        )
        unresolved["pre_decomposition_source"] = classifier_purity.get(
            "purity_source"
        )
        unresolved["quantitative_caveat"] = (
            "Purity estimators support incompatible scenarios; the selected "
            "estimate is retained for downstream modeling but is not a consensus "
            "purity measurement."
        )
        return "discordant", unresolved

    point = decomp_purity.get("overall_estimate")
    independents = _independent_purity_signals(classifier_purity)
    try:
        point_f = float(point)
    except (TypeError, ValueError):
        point_f = None

    # Fall back when the fragile decomposition point disagrees with EVERY independent signal.
    fully_inconsistent = bool(
        point_f is not None
        and independents
        and all(abs(point_f - s) > _DECOMP_INDEPENDENT_DISAGREE_MARGIN for s in independents)
    )
    if fully_inconsistent and isinstance(classifier_purity, dict) and _classifier_purity_trustworthy(
        classifier_purity
    ):
        reverted = dict(classifier_purity)
        reverted["decomposition_purity_rejected"] = {
            "decomposition_estimate": round(point_f, 4),
            "independent_estimates": [round(s, 4) for s in independents],
            "reason": "fragile decomposition purity disagreed with every independent signal",
        }
        return "reject", reverted

    # Otherwise widen the interval to span the plausible hypothesis range (∪ the existing interval).
    widened = dict(decomp_purity)
    lo_candidates = [decomp_purity.get("overall_lower"), stability.get("hypothesis_purity_lo")]
    hi_candidates = [decomp_purity.get("overall_upper"), stability.get("hypothesis_purity_hi")]
    lo_vals = [float(x) for x in lo_candidates if isinstance(x, (int, float))]
    hi_vals = [float(x) for x in hi_candidates if isinstance(x, (int, float))]
    if point_f is not None:
        lo_vals.append(point_f)
        hi_vals.append(point_f)
    if lo_vals:
        widened["overall_lower"] = round(max(0.0, min(lo_vals)), 4)
    if hi_vals:
        widened["overall_upper"] = round(min(1.0, max(hi_vals)), 4)
    widened["purity_interval_widened_for_fragility"] = {
        "hypothesis_spread": stability.get("hypothesis_purity_spread"),
        "tme_overexplained": stability.get("tme_overexplained"),
        "reason": "decomposition purity is template-sensitive; interval spans the plausible fits",
    }
    return "widen", widened


def build_analysis_parameters(
    *,
    config: AnalyzeConfig,
    resolution: InputResolution,
    template_overrides: list[str],
    selected_sample_mode: str,
    quality: dict[str, Any],
    tumor_purity_parameters: dict[str, Any],
    decomposition_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Stable JSON payload for ``*-analysis-parameters.json``."""
    return {
        "input": {
            "path": resolution.gene_input,
            "transcript_path": resolution.transcript_input,
            "input_level": resolution.input_level,
            "aggregate_gene_expression": resolution.aggregate_gene_expression,
            "gene_name_col": config.gene_name_col,
            "gene_id_col": config.gene_id_col,
            "sample_id_col": config.sample_id_col,
            "sample_id_value": config.sample_id_value,
            "cancer_type": config.cancer_type,
            "cancer_type_source": "user-specified"
            if config.cancer_type
            else "auto-detected",
            "sample_mode": config.sample_mode,
            "tumor_context": config.tumor_context,
            "site_hint": config.site_hint,
            "met_site": config.met_site,
            "decomposition_templates": template_overrides,
            "hla_types": config.hla_type_list(),
            "fusions": config.fusion_path_list(),
            "variants": config.variant_input_list(),
            "expression_qc_rescue": config.expression_qc_rescue,
        },
        "tumor_purity": tumor_purity_parameters,
        "decomposition": decomposition_parameters,
        "selected_sample_mode": selected_sample_mode,
        "embedding_methods": ["pan_reference_mds", "pan_reference_nearest_references"],
        "sample_quality": {
            "degradation_level": quality["degradation"]["level"],
            "degradation_pair_index": quality["degradation"]["long_short_ratio"],
            "culture_level": quality["culture"]["level"],
            "culture_stress_score": quality["culture"]["stress_score"],
            "has_issues": quality["has_issues"],
        },
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> str:
    target = Path(path)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(target)


def discover_output_artifacts(
    out_dir: str | Path, prefix_base: str
) -> list[AnalyzeArtifact]:
    """Create a best-effort manifest from files emitted by a run."""
    root = Path(out_dir)
    candidates: list[Path] = []
    if root.is_dir():
        candidates.extend(
            path
            for path in root.iterdir()
            if path.is_file() and path.name.startswith(f"{prefix_base}-")
        )
        figures_dir = root / "figures"
        if figures_dir.is_dir():
            candidates.extend(
                path
                for path in figures_dir.iterdir()
                if path.is_file() and path.name.startswith(f"{prefix_base}-")
            )

    artifacts: list[AnalyzeArtifact] = []
    for path in sorted(candidates):
        suffix = path.suffix.lower()
        if suffix == ".png":
            kind = "figure"
        elif suffix == ".pdf":
            kind = "figure-packet"
        elif suffix in {".tsv", ".csv"}:
            kind = "table"
        elif suffix == ".md":
            kind = "report"
        elif suffix == ".json":
            kind = "metadata"
        else:
            kind = "artifact"
        artifacts.append(
            AnalyzeArtifact(
                path=str(path),
                kind=kind,
                step="discover",
                role=path.stem.removeprefix(f"{prefix_base}-"),
            )
        )
    return artifacts
