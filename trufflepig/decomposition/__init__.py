# Licensed under the Apache License, Version 2.0

"""Broad-compartment sample decomposition with external purity anchoring.

Decomposes bulk RNA-seq into tumor + TME components (immune, stromal,
site-specific host cells) using weighted NNLS on component-enriched
marker genes.  Tumor purity is estimated separately, then the non-tumor
fraction is distributed across reference-supported compartments.
"""

from .engine import (
    decompose_sample,
    DecompositionResult,
    get_decomposition_parameters,
    infer_sample_mode,
)
from .panels import (
    build_matched_normal_biased_panel,
    build_shared_lineage_panel,
    build_tumor_biased_panel,
    estimate_lineage_tumor_fraction,
    summarize_panels,
)
from .plot import (
    plot_decomposition_candidates,
    plot_decomposition_component_breakdown,
    plot_decomposition_composition,
    plot_decomposition_summary,
)
from .templates import (
    EPITHELIAL_MATCHED_NORMAL_TISSUE,
    epithelial_matched_normal_component,
)

__all__ = [
    "decompose_sample",
    "DecompositionResult",
    "EPITHELIAL_MATCHED_NORMAL_TISSUE",
    "epithelial_matched_normal_component",
    "build_tumor_biased_panel",
    "build_matched_normal_biased_panel",
    "build_shared_lineage_panel",
    "estimate_lineage_tumor_fraction",
    "summarize_panels",
    "get_decomposition_parameters",
    "infer_sample_mode",
    "plot_decomposition_summary",
    "plot_decomposition_composition",
    "plot_decomposition_component_breakdown",
    "plot_decomposition_candidates",
]
