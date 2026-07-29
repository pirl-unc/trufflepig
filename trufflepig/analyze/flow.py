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
    already_widened = bool(
        purity_block.get("ci_widening_factor") == round(ci_factor, 3)
        and existing_caveat.get("widened_lower") == round(float(lo), 4)
        and existing_caveat.get("widened_upper") == round(float(hi), 4)
    )
    if already_widened:
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
    only a purity estimate when the template agrees with the classifier,
    has non-tumor components, and exposes a populated purity result.
    """
    if decomp_result is None:
        return False
    if getattr(decomp_result, "cancer_type", None) != classifier_code:
        return False
    warnings = getattr(decomp_result, "warnings", None) or []
    if any("No non-tumor components in template" in warning for warning in warnings):
        return False
    return bool(getattr(decomp_result, "purity_result", None))


# Fragility thresholds for the observed decomposition purity (instrumentation, not yet a gate).
_DECOMP_PURITY_FRAGILE_SPREAD = 0.35
_TME_OVEREXPLAINED_MARKER = "overexplained by the tme background"


def decomposition_purity_stability(decomp_results, adopted=None) -> dict:
    """Observe how fragile the adopted decomposition purity is — INSTRUMENTATION, no behavior change.

    A decomposition tumor fraction is only trustworthy when it is stable across the plausible
    template hypotheses AND the background did not over-absorb tumor signal. Across the local cohort,
    most samples fail one of those: the top hypotheses disagree on tumor fraction by tens of points,
    and the low-purity calls carry a "many genes overexplained by the TME background" warning
    (over-subtraction → deflated purity — the READ 10%-vs-78% failure mode). This records the signals
    so reports can encode that uncertainty and a later gate can consume them; it changes nothing.

    Returns, for the adopted hypothesis:
      - ``hypothesis_purity_spread``: max−min tumor fraction among the same-cancer-type hypotheses
        (template-choice sensitivity of the purity).
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
    # Template-choice sensitivity: compare purities only across the SAME-cancer-type hypotheses —
    # a different-cancer template's tumor fraction is not a comparable measurement of this call.
    same_type = [r for r in results if getattr(r, "cancer_type", None) == adopted_type]
    purities = [
        float(getattr(r, "purity", 0.0))
        for r in (same_type or results)[:6]
        if getattr(r, "purity", None) is not None
    ]
    spread = round(max(purities) - min(purities), 4) if len(purities) >= 2 else 0.0
    hyp_lo = round(min(purities), 4) if purities else None
    hyp_hi = round(max(purities), 4) if purities else None
    warns = " ".join(getattr(adopted, "warnings", None) or []).lower()
    tme_overexplained = _TME_OVEREXPLAINED_MARKER in warns
    top = [
        {
            "template": getattr(r, "template", None),
            "cancer_type": getattr(r, "cancer_type", None),
            "purity": round(float(getattr(r, "purity", 0.0) or 0.0), 4),
            "reconstruction_error": round(float(getattr(r, "reconstruction_error", 0.0) or 0.0), 4),
        }
        for r in results[:4]
    ]
    return {
        "hypothesis_purity_spread": spread,
        "hypothesis_purity_lo": hyp_lo,
        "hypothesis_purity_hi": hyp_hi,
        "tme_overexplained": tme_overexplained,
        "fragile": bool(spread >= _DECOMP_PURITY_FRAGILE_SPREAD or tme_overexplained),
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

      * always WIDEN the reported interval to span the plausible hypothesis range and mark the
        purity as template-sensitive (the point estimate is kept); and
      * FALL BACK to the classifier's independent purity when the decomposition point disagrees with
        EVERY independent signal (signature, lineage, ESTIMATE) by more than the margin — a fragile
        estimate that no other method corroborates is not trustworthy as the headline.

    A non-fragile decomposition purity is returned unchanged. Returns ``(action, purity_dict)`` where
    ``action`` is ``"adopt"`` (unchanged), ``"widen"`` (adopt point, widened interval + caveat), or
    ``"reject"`` (do not adopt; use ``classifier_purity``). The returned dict is always safe to adopt.
    """
    stability = stability or {}
    if not isinstance(decomp_purity, dict) or not stability.get("fragile"):
        return "adopt", decomp_purity

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
            "alterations": config.alteration_input_list(),
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
