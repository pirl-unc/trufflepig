"""Structured API boundary for the ``pirlygenes analyze`` pipeline.

The command-line implementation still lives in :mod:`pirlygenes.cli`, but
the data contracts in this package are intentionally free of plotting and
heavy reference imports. They define the handoff surface that can move to
the future ``trufflepig`` repository without dragging the whole gene-set
package with it.
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
    load_analyze_summary_record,
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
    "StepRecord",
    "apply_sample_context_to_purity",
    "build_analysis_parameters",
    "build_analyze_comparison_markdown",
    "build_analyze_paths",
    "cancer_type_context_code",
    "cancer_type_context_from_analysis",
    "cancer_type_context_label",
    "discover_output_artifacts",
    "expression_reference_sources",
    "has_expression_reference",
    "load_analyze_summary_record",
    "registry_parent_code",
    "resolve_analyze_inputs",
    "should_adopt_decomposition_purity",
    "write_json",
]
