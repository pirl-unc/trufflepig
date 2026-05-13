# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Re-export facade for the plot submodules.

The implementation lives in five focused modules:

- ``plot_strip``        — gene-expression strip plots
- ``plot_scatter``      — sample-vs-cancer scatter plots, cancer-type aliases
- ``plot_therapy``      — therapy-target / geneset / FN1-EDB plots
- ``plot_tumor_expr``   — tumor-expression estimation and range plots
- ``plot_embedding``    — cancer-type embedding, cohort heatmaps, MDS/PCA/UMAP

All public names are re-exported here so ``from .plot import X`` continues to
work unchanged throughout the codebase.
"""

from .common import _guess_gene_cols as _guess_gene_cols  # noqa: F401

# ── plot_strip ───────────────────────────────────────────────────────────
from .plot_strip import (  # noqa: F401
    _load_gene_sets,
    pick_genes_to_annotate,
    _normalize_label_token,
    resolve_always_label_gene_ids,
    _build_default_gene_sets,
    default_gene_sets,
    plot_gene_expression,
)

# ── plot_scatter ─────────────────────────────────────────────────────────
from .plot_scatter import (  # noqa: F401
    CANCER_TYPE_NAMES,
    CANCER_TYPE_ALIASES,
    resolve_cancer_type,
    _prepare_sample_vs_cancer_data,
    _draw_scatter_panel,
    plot_sample_vs_cancer,
)

# ── plot_therapy ─────────────────────────────────────────────────────────
from .plot_therapy import (  # noqa: F401
    _FN1_GENE_ID,
    _FN1_EDB_MIN_TPM,
    _FN1_EDB_MIN_FRACTION,
    _FN1_EDB_TRANSCRIPT_IDS,
    _FN1_EDB_TRANSCRIPT_NAMES,
    _FN1_EDB_EXON_IDS,
    _FN1_EDB_EXON_INTERVALS,
    _fn1_edb_transcript_ids,
    _summarize_fn1_edb_transcript_support,
    _apply_therapy_support_gate,
    _ordered_therapy_tuple,
    _therapy_combo_label,
    _therapy_combo_sort_key,
    _therapy_base_colors,
    _therapy_combo_colors,
    _draw_therapy_marker,
    _approved_radioligand_gene_ids,
    _collect_ranked_therapy_targets,
    plot_therapy_target_tissues,
    plot_therapy_target_safety,
    _resolve_gene_set_symbols,
    plot_geneset_vs_vital_tissues,
    plot_ctas_vs_cancer_type_detail,
)

# ── plot_tumor_expr ──────────────────────────────────────────────────────
from .plot_tumor_expr import (  # noqa: F401
    _REPRODUCTIVE_TISSUES,
    _STROMAL_TISSUES,
    _IMMUNE_TISSUES,
    _TME_TISSUES,
    MET_SITE_TISSUE_AUGMENTATION,
    MET_SITES,
    _sample_expression_by_symbol,
    estimate_tumor_expression,
    estimate_tumor_expression_ranges,
    plot_matched_normal_attribution,
    plot_target_attribution,
    plot_tumor_expression_ranges,
    plot_purity_adjusted_targets,
)

# ── plot_embedding ───────────────────────────────────────────────────────
from .plot_embedding import (  # noqa: F401
    _signature_panel_cache,
    _get_cancer_type_signature_panels,
    _compute_cancer_type_signature_stats,
    _select_embedding_genes_bottleneck,
    _select_tme_low_genes,
    _select_embedding_genes,
    _cancer_type_score_matrix,
    _reference_cancer_expression_df,
    _hierarchy_feature_labels,
    _hierarchy_feature_vector,
    _reference_family_feature_matrix,
    _reference_site_feature_matrix,
    _hierarchy_embedding_metadata,
    _cancer_type_hierarchy_matrix,
    _sanitize_embedding_matrix,
    _subtype_expression_values_for_ref,
    _cancer_type_feature_matrix,
    get_embedding_feature_metadata,
    _plot_embedding_with_labels,
    plot_cancer_type_genes,
    plot_cancer_type_disjoint_genes,
    plot_cohort_heatmap,
    plot_cohort_disjoint_counts,
    plot_cohort_pca,
    plot_cohort_therapy_targets,
    _plot_geneset_by_cancer_heatmap,
    plot_cohort_surface_proteins,
    plot_cohort_ctas,
    plot_cancer_type_pca,
    plot_cancer_type_mds,
    plot_cancer_type_neighborhood,
    plot_cancer_type_umap,
)

# ── plot_target_deep_dive ────────────────────────────────────────────────
from .plot_target_deep_dive import (  # noqa: F401
    actionable_surface_targets,
    plot_actionable_targets,
    plot_curated_target_evidence,
    plot_priority_target_context,
    plot_priority_targets,
    plot_tumor_attribution,
    plot_cta_deep_dive,
)

# ── plot_subtype_signature ───────────────────────────────────────────────
from .plot_subtype_signature import (  # noqa: F401
    SUBTYPE_CONTRASTS,
    plot_subtype_signature,
)
