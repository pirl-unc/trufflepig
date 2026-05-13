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
    if not output_dir or output_dir == "pirlygenes-output":
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
    Step-0 -> purity information flow explicit and unit-testable.
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

    half_lo = max(0.0, est - lo) * ci_factor
    half_hi = max(0.0, hi - est) * ci_factor
    purity_block["overall_lower"] = round(max(0.0, est - half_lo), 4)
    purity_block["overall_upper"] = round(min(1.0, est + half_hi), 4)
    purity_block["ci_widening_factor"] = round(ci_factor, 3)
    purity_block["degradation_caveat"] = {
        "severity": sample_context.degradation_severity,
        "index": sample_context.degradation_index,
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
