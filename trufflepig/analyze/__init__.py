"""Structured API boundary for the ``trufflepig`` analyze pipeline.

The data contracts in this package are intentionally free of plotting
and heavy reference imports — they define the handoff surface between
the user-facing markdown reports and downstream tooling (longitudinal
deltas, web UI, LLM consumers).
"""

from .flow import (
    apply_sample_context_to_purity,
    build_analysis_parameters,
    build_analyze_paths,
    discover_output_artifacts,
    resolve_analyze_inputs,
    should_adopt_decomposition_purity,
    write_json,
)
from .comparison import (
    AnalyzeSummaryRecord,
    build_analyze_comparison_markdown,
    compute_longitudinal_delta_sets,
    load_analyze_summary_record,
)
from .deltas import (
    LongitudinalDelta,
    LongitudinalDeltaSet,
    ResponseAxisState,
    TargetShortlistEntry,
    compute_pairwise_deltas,
    parse_response_axes,
    parse_target_shortlist,
    write_deltas_json,
)
from .models import (
    AnalyzeArtifact,
    AnalyzeConfig,
    AnalyzePaths,
    AnalyzeRun,
    InputResolution,
    StepRecord,
)
from .cancer_type_context import (
    CancerTypeContext,
    cancer_type_context_code,
    cancer_type_context_from_analysis,
    cancer_type_context_label,
    expression_reference_sources,
    has_expression_reference,
    registry_parent_code,
)

__all__ = [
    "AnalyzeArtifact",
    "AnalyzeConfig",
    "AnalyzePaths",
    "AnalyzeRun",
    "AnalyzeSummaryRecord",
    "CancerTypeContext",
    "InputResolution",
    "LongitudinalDelta",
    "LongitudinalDeltaSet",
    "ResponseAxisState",
    "StepRecord",
    "TargetShortlistEntry",
    "apply_sample_context_to_purity",
    "build_analysis_parameters",
    "build_analyze_comparison_markdown",
    "build_analyze_paths",
    "cancer_type_context_code",
    "cancer_type_context_from_analysis",
    "cancer_type_context_label",
    "compute_longitudinal_delta_sets",
    "compute_pairwise_deltas",
    "discover_output_artifacts",
    "expression_reference_sources",
    "has_expression_reference",
    "load_analyze_summary_record",
    "parse_response_axes",
    "parse_target_shortlist",
    "registry_parent_code",
    "resolve_analyze_inputs",
    "should_adopt_decomposition_purity",
    "write_deltas_json",
    "write_json",
]
