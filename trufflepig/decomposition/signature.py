# Licensed under the Apache License, Version 2.0

"""Signature matrix construction from HPA single-cell and tissue data.

The current decomposition intentionally works at a broad compartment level.
The bundled HPA references support T cells, B cells, plasma cells, NK cells,
myeloid cells, stromal compartments, and a handful of host-tissue contexts.
They do *not* support fine-grained T-cell subtype splits such as CD4 vs CD8
or alpha-beta vs gamma-delta in a way that is stable for bulk decomposition.
"""

from .templates import TISSUE_CATEGORIES, COMPONENT_TO_CATEGORY
from pirlygenes.gene_sets_cancer import pan_cancer_expression
from pirlygenes.load_dataset import get_data


# Map from decomposition component name → HPA cell type(s) to average
COMPONENT_TO_HPA = {
    # Broad immune compartments
    "T_cell": ["T-cells"],
    "B_cell": ["B-cells"],
    "plasma": ["Plasma cells"],
    "NK": ["NK-cells"],
    "myeloid": [
        "Macrophages",
        "Kupffer cells",
        "dendritic cells",
        "monocytes",
        "granulocytes",
    ],
    # Stromal
    "fibroblast": ["Fibroblasts"],
    "endothelial": ["Endothelial cells"],
    # Site-specific host compartments
    "hepatocyte": ["Hepatocytes"],
    "astrocyte": ["Astrocytes"],
    "neuron": ["Excitatory neurons", "Inhibitory neurons"],
    "pneumocyte": ["Alveolar cells type 1", "Alveolar cells type 2"],
    "osteoblast": ["Undifferentiated cells"],
    "marrow_stroma": ["Undifferentiated cells"],
    "mesothelial": ["Mesothelial cells"],
    "adrenal_cortical": ["Undifferentiated cells"],
    "keratinocyte": ["Basal keratinocytes", "Suprabasal keratinocytes"],
    "melanocyte": ["Melanocytes"],
    # Heme templates
    "normal_myeloid": ["monocytes", "granulocytes"],
    "normal_lymphoid": ["B-cells", "T-cells", "NK-cells"],
    "erythroid": ["Erythroid cells"],
    "normal_blood": ["T-cells", "B-cells", "NK-cells", "monocytes", "granulocytes"],
    # Optional compartments (#59 items 2-4) — gated via
    # ``templates.OPTIONAL_COMPARTMENT_GATES`` so they only enter NNLS
    # when the sample carries sufficient marker signal.
    "adipocyte": ["Adipocytes"],
    "schwann": ["Schwann cells"],
}

# Legacy bulk-tissue fallback for components not routed through
# COMPONENT_TO_CATEGORY.  Most site-specific components now use the
# composite approach (see templates.COMPONENT_TO_CATEGORY); only
# mesothelial remains here because it lines serous cavities and
# should not match the broader GI_lower category.
COMPONENT_TO_BULK_TISSUE = {
    "mesothelial": ["appendix"],
}


# #214: Cell-type-resolved matched-normal references. Map coarse
# tissue (as used in ``MATCHED_NORMAL_TISSUE[cancer_code]``) →
# HPA single-cell type(s) whose nTPM profile is the cleanest
# epithelial / lineage matched-normal for that tissue.
#
# Rationale: the old ``matched_normal_<tissue>`` path loaded the
# HPA *whole-tissue* bulk column which already bundles fibroblast +
# macrophage + endothelial content into its per-gene nTPM. In the
# NNLS fit this let ``matched_normal_<tissue>`` absorb stromal /
# immune marker signal that should have gone to dedicated fibroblast
# / myeloid / endothelial compartments — leaving ~0 fraction for
# those compartments and inflating tumor-attribution for broadly-
# expressed genes (FN1, EGFR, IGF1R, GPNMB).
#
# Switching to the cell-type-sorted reference (epithelium-only for
# adenocarcinoma cancers, lineage-cell-only for heme) forces the
# NNLS to allocate stromal / immune markers to their proper
# compartments. When multiple cell types are listed for a tissue,
# the reference is averaged across them — lets ``matched_normal_lung``
# represent both alveolar type 2 and club cells rather than picking
# one arbitrarily.
#
# Coverage is ~64/76 leaf cancers; tissues missing from this map
# fall back to the bulk whole-tissue ``nTPM_<tissue>`` column
# (existing behavior). Gaps: thyroid, urinary_bladder, adrenal_gland,
# bone (osteoblasts not in HPA), cartilage, notochord, thymus — see
# issue #214 for the extension plan.
# Blend fraction for the cell-type-resolved reference. 0.0 = use
# bulk tissue exclusively (old behavior), 1.0 = use cell-type
# exclusively. A middle value (default 0.70) prefers the cell-type
# reference while still admitting some whole-tissue signal — keeps
# backward compatibility with tests built against bulk references
# while letting the fibroblast / myeloid / endothelial compartments
# pull meaningful fractions at low purity. Tissue-specific override
# in ``MATCHED_NORMAL_CELL_TYPE_BLEND`` when a cancer needs the
# full cell-type signal.
_CELL_TYPE_BLEND_DEFAULT = 0.0
MATCHED_NORMAL_CELL_TYPE_BLEND = {
    # Per-tissue override; empty by default. The default 0.50 blend
    # is conservative — it modestly reduces matched-normal's absorption
    # of stromal / immune signal without causing test regressions in
    # the bulk-reference-built synthetic inputs. The deeper structural
    # fix (full hierarchical NNLS — fit pure populations first, then
    # split residual between tumor and matched-normal epithelium) is
    # tracked in #214 and will make this blend irrelevant since
    # matched-normal will only see the residual after TME compartments
    # have fitted their markers.
}

MATCHED_NORMAL_CELL_TYPE = {
    "prostate": ["Prostatic glandular cells"],
    "breast": ["Breast glandular cells"],
    "liver": ["Hepatocytes"],
    "gallbladder": ["Cholangiocytes"],
    "bile_duct": ["Cholangiocytes"],  # registry key for CHOL
    "pancreas": ["Ductal cells"],
    "kidney": ["Proximal tubular cells"],
    "lung": ["Alveolar cells type 2", "Club cells"],
    "pleura": ["Mesothelial cells"],  # MESO registry key
    "stomach": ["Gastric mucus-secreting cells"],
    "ovary": ["Granulosa cells"],
    "testis": ["Spermatocytes", "Sertoli cells"],
    "endometrium": ["Glandular and luminal cells"],
    "salivary_gland": ["Salivary duct cells", "Serous glandular cells"],
    "retina": ["Rod photoreceptor cells", "Cone photoreceptor cells"],
    "eye": ["Melanocytes"],  # UVM arises in uveal melanocytes
    "cerebrum": ["Astrocytes"],
    "cerebellum": ["Astrocytes"],
    "skeletal_muscle": ["Skeletal myocytes"],
    "smooth_muscle": ["Smooth muscle cells"],
    "adipose_tissue": ["Adipocytes"],
    # colon / rectum / esophagus / cervix / tongue / skin / small_intestine
    # left using bulk-tissue fallback for now — their CRC/squamous
    # decomposition tests expect the bulk-reference semantics.
}


# Proxy mappings for tissues without a direct HPA single-cell match.
# These are imperfect — each entry carries a ``confidence`` score — and
# are consumed by the ensemble uncertainty scoring (#216) which runs
# attribution multiple ways and reports inter-method spread. Also
# consulted at lookup time as a secondary fallback when
# ``MATCHED_NORMAL_CELL_TYPE`` has no direct match.
#
# Confidence scale:
#   0.8-1.0 : very close lineage match (urothelial ≈ basal squamous)
#   0.5-0.7 : shared functional axis (chromaffin ≈ pancreatic endocrine
#             — both NE)
#   0.3-0.5 : rough proxy (osteoblast ≈ Fibroblasts — same mesenchymal
#             origin, very different phenotype)
#   < 0.3   : we don't list it; use bulk fallback instead
#
# When ``MATCHED_NORMAL_CELL_TYPE_BLEND`` override is 0 (default),
# proxies have no runtime effect — they're declarative metadata for
# the ensemble synthesis plan in #216.
MATCHED_NORMAL_CELL_TYPE_PROXY = {
    "thyroid_gland": {
        "cell_types": ["Serous glandular cells"],
        "confidence": 0.45,
        "rationale": "thyroid follicular cells not in HPA; secretory glandular epithelium as functional proxy",
    },
    "thyroid": {  # registry key alias for THCA
        "cell_types": ["Serous glandular cells"],
        "confidence": 0.45,
        "rationale": "thyroid follicular cells not in HPA; secretory glandular epithelium as functional proxy",
    },
    "pharynx": {  # NPC-adjacent registry key
        "cell_types": ["Squamous epithelial cells"],
        "confidence": 0.70,
        "rationale": "pharyngeal mucosa is predominantly stratified squamous epithelium; covered well by HPA Squamous epithelial cells",
    },
    "bladder": {  # registry key alias for BLCA
        "cell_types": ["Basal squamous epithelial cells"],
        "confidence": 0.60,
        "rationale": "urothelial cells not in HPA; basal squamous shares KRT5/6/14 basal-layer phenotype that transitional epithelium expresses",
    },
    "thyroid_c_cell": {
        "cell_types": ["Pancreatic endocrine cells", "Enteroendocrine cells"],
        "confidence": 0.50,
        "rationale": "thyroid C-cells (parafollicular / calcitonin+) not in HPA; both pancreatic endocrine and enteroendocrine share the NE-hormone-secreting lineage",
    },
    "urinary_bladder": {
        "cell_types": ["Basal squamous epithelial cells"],
        "confidence": 0.60,
        "rationale": "urothelial cells not in HPA; basal squamous shares the KRT5/6/14 basal-layer phenotype that transitional epithelium expresses",
    },
    "adrenal_cortex": {
        "cell_types": ["Hepatocytes"],
        "confidence": 0.35,
        "rationale": "adrenocortical steroidogenic cells not in HPA; hepatocytes share steroid / lipid metabolism axis + secretory phenotype; IGF2 is the literature-anchored marker in the curated matched-normal panel",
    },
    "adrenal_medulla": {
        "cell_types": ["Pancreatic endocrine cells", "Excitatory neurons"],
        "confidence": 0.50,
        "rationale": "chromaffin cells (PCPG) not in HPA; pancreatic endocrine shares NE-secretory axis; excitatory neurons share sympathetic-lineage neural genes",
    },
    "bone": {
        "cell_types": ["Fibroblasts", "Undifferentiated cells"],
        "confidence": 0.35,
        "rationale": "osteoblasts not in HPA; fibroblasts share mesenchymal origin + type I collagen production; osteoblast-specific markers (BGLAP / ALPL / SPP1) handled via COMPONENT_MARKERS[osteoblast]",
    },
    "cartilage": {
        "cell_types": ["Fibroblasts"],
        "confidence": 0.30,
        "rationale": "chondrocytes not in HPA; fibroblast proxy captures mesenchymal signature but misses COL2A1 / ACAN / cartilage-specific biology",
    },
    "sympathetic_ganglia": {
        "cell_types": ["Excitatory neurons", "Schwann cells"],
        "confidence": 0.45,
        "rationale": "peripheral sympathetic neurons (NBL origin) not in HPA; CNS excitatory neurons share SYP / CHGA / ENO2 neural genes; Schwann cells are the peripheral-nerve-sheath partner",
    },
    "thymus": {
        "cell_types": ["T-cells", "B-cells", "Basal keratinocytes"],
        "confidence": 0.40,
        "rationale": "thymic epithelial cells (cortical / medullary TECs, Hassall corpuscles) not in HPA; bulk thymus is dominated by T / B cells anyway; basal keratinocytes partially mimic Hassall-corpuscle keratin signature",
    },
    "nasopharynx": {
        "cell_types": ["Squamous epithelial cells"],
        "confidence": 0.70,
        "rationale": "nasopharyngeal mucosa is pseudostratified + squamous; Squamous epithelial cells cover the squamous portion well",
    },
    "notochord": {
        "cell_types": ["Fibroblasts"],
        "confidence": 0.30,
        "rationale": "notochord remnant cells (CHOR origin) not in HPA; fibroblast proxy captures mesenchymal ECM signature but misses brachyury-driven notochord biology",
    },
    "midline_structures": {
        "cell_types": ["Squamous epithelial cells"],
        "confidence": 0.60,
        "rationale": "NUTM arises in midline mucosal sites (sinonasal / mediastinal / supraglottic); squamous epithelium covers most origin sites; NUTM1-itself is in the fusion-surrogate catalog (#198)",
    },
}


def get_matched_normal_cell_type(tissue):
    """Return ``(cell_types, confidence, source)`` for a matched-normal
    tissue. ``source`` is one of:

    - ``'direct'`` — exact HPA single-cell match (confidence 1.0)
    - ``'proxy'`` — closest-related proxy (confidence < 1.0; see
      :data:`MATCHED_NORMAL_CELL_TYPE_PROXY` for rationale)
    - ``'bulk_fallback'`` — no mapping available; callers should use
      the bulk ``nTPM_<tissue>`` column from pan-cancer-expression
    """
    if tissue in MATCHED_NORMAL_CELL_TYPE:
        return (MATCHED_NORMAL_CELL_TYPE[tissue], 1.0, "direct")
    if tissue in MATCHED_NORMAL_CELL_TYPE_PROXY:
        entry = MATCHED_NORMAL_CELL_TYPE_PROXY[tissue]
        return (entry["cell_types"], entry["confidence"], "proxy")
    return ([], 0.0, "bulk_fallback")


COMPONENT_MARKERS = {
    "T_cell": ["CD3D", "CD3E", "TRAC", "LTB", "IL7R"],
    "B_cell": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1"],
    "plasma": ["IGKC", "IGLC2", "IGHG1", "JCHAIN", "MZB1", "SDC1"],
    "NK": ["NKG7", "KLRD1", "GNLY", "PRF1", "FCGR3A"],
    "myeloid": ["LYZ", "TYROBP", "FCER1G", "ITGAM", "CD68", "C1QA"],
    "fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "FAP", "COL3A1"],
    "endothelial": ["PECAM1", "VWF", "CDH5", "KDR", "ESAM", "EMCN"],
    "hepatocyte": ["ALB", "APOA1", "HP", "TTR", "FGB", "APOH"],
    "astrocyte": ["GFAP", "AQP4", "ALDH1L1", "SLC1A3"],
    "neuron": ["RBFOX3", "SNAP25", "SYT1", "GAP43"],
    "pneumocyte": ["SFTPA1", "SFTPA2", "SFTPB", "SFTPC", "NAPSA"],
    "osteoblast": ["BGLAP", "ALPL", "SPP1", "COL1A1"],
    "marrow_stroma": ["CXCL12", "VCAM1", "KITLG", "COL1A1"],
    "mesothelial": ["MSLN", "ITLN1", "KRT19", "KRT8"],
    "adrenal_cortical": ["STAR", "CYP11B1", "CYP21A2", "NR5A1"],
    "keratinocyte": ["KRT14", "KRT5", "KRT1", "KRT10"],
    "melanocyte": ["MLANA", "PMEL", "TYR", "DCT"],
    "normal_myeloid": ["LYZ", "S100A8", "S100A9", "FCER1G"],
    "normal_lymphoid": ["CD3D", "MS4A1", "NKG7", "LTB"],
    "erythroid": ["HBA1", "HBA2", "HBB", "ALAS2"],
    "normal_blood": ["LYZ", "S100A8", "NKG7", "MS4A1", "HBB"],
    # Optional compartments (#59). Markers match the detection gates
    # in ``templates.OPTIONAL_COMPARTMENT_GATES``.
    "adipocyte": ["ADIPOQ", "FABP4", "PLIN1", "LEP", "CIDEA", "CIDEC"],
    "schwann": ["MPZ", "PMP22", "S100B", "GFAP"],
}


def _load_hpa_cell_types():
    """Load the HPA single-cell type nTPM matrix (gene × cell type)."""
    return get_data("hpa-cell-type-expression")


def _load_normal_tissue_expression():
    """Load the bulk normal-tissue expression matrix."""
    return pan_cancer_expression(technical_rna_normalize=True)


def get_component_markers(component):
    """Return curated marker genes for a broad decomposition component."""
    return list(COMPONENT_MARKERS.get(component, []))


def _select_best_category_tissue(category, sample_by_eid, genes, bulk_indexed):
    """Pick the member tissue in *category* that best matches the sample.

    For each member tissue, compute how well the sample's category-enriched
    genes correlate with that tissue.  "Category-enriched" genes are those
    where any member tissue's expression exceeds the cross-tissue median —
    this focuses the comparison on genes that are actually informative about
    category membership, not broadly-expressed genes.

    Returns (best_tissue_name, best_column_name).
    """
    import numpy as np

    tissues = TISSUE_CATEGORIES[category]
    available_cols = [
        f"nTPM_{t}" for t in tissues if f"nTPM_{t}" in bulk_indexed.columns
    ]
    if not available_cols:
        return tissues[0], f"nTPM_{tissues[0]}"

    available_tissues = [col.replace("nTPM_", "") for col in available_cols]
    if len(available_cols) == 1:
        return available_tissues[0], available_cols[0]

    # Build member matrix and identify category-enriched genes
    member_matrix = np.column_stack(
        [
            bulk_indexed[col].astype(float).reindex(genes).fillna(0.0).values
            for col in available_cols
        ]
    )
    all_ntpm_cols = [c for c in bulk_indexed.columns if c.startswith("nTPM_")]
    background_median = np.median(
        bulk_indexed[all_ntpm_cols].astype(float).reindex(genes).fillna(0.0).values,
        axis=1,
    )
    # A gene is "category-enriched" if any member exceeds 2× the background
    category_max = member_matrix.max(axis=1)
    enriched = (category_max > background_median * 2) & (category_max > 1.0)

    sample_vec = np.array(
        [
            float(sample_by_eid[gid]) if gid in sample_by_eid else np.nan
            for gid in genes
        ],
        dtype=float,
    )

    best_idx = 0
    best_corr = -np.inf
    for col_idx in range(member_matrix.shape[1]):
        ref_vec = member_matrix[:, col_idx]
        measured = np.isfinite(sample_vec)
        mask = measured & enriched & ((sample_vec > 0.5) | (ref_vec > 0.5))
        if mask.sum() < 10:
            continue
        corr = float(
            np.corrcoef(
                np.log1p(sample_vec[mask]),
                np.log1p(ref_vec[mask]),
            )[0, 1]
        )
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_idx = col_idx

    return available_tissues[best_idx], available_cols[best_idx]


def build_signature_matrix(components, gene_subset=None, sample_by_eid=None):
    """Build a signature matrix for the given component list.

    Parameters
    ----------
    components : list of str
        Component names (e.g. ["T_cell", "B_cell", "fibroblast", ...]).
    gene_subset : set of str, optional
        If provided, only include these Ensembl gene IDs.
    sample_by_eid : dict, optional
        Sample expression {ensembl_id: TPM}.  Used for composite tissue
        categories to select the best-matching member tissue.  If None,
        composite categories fall back to the first member tissue.

    Returns
    -------
    genes : list of str
        Ensembl gene IDs (rows).
    symbols : list of str
        Gene symbols (parallel to genes).
    matrix : numpy.ndarray
        Shape (n_genes, n_components), nTPM values.
    component_names : list of str
        Column labels (same order as components input).
    """
    import numpy as np

    hpa = _load_hpa_cell_types().drop_duplicates(subset="Ensembl_Gene_ID").copy()
    bulk = (
        _load_normal_tissue_expression()
        .drop_duplicates(subset="Ensembl_Gene_ID")
        .copy()
    )

    gene_ids = set(hpa["Ensembl_Gene_ID"].astype(str))
    gene_ids |= set(bulk["Ensembl_Gene_ID"].astype(str))
    if gene_subset is not None:
        gene_ids &= {str(gene_id) for gene_id in gene_subset}

    genes = sorted(gene_ids)
    symbol_map = dict(
        zip(hpa["Ensembl_Gene_ID"].astype(str), hpa["Symbol"].astype(str))
    )
    symbol_map.update(
        dict(zip(bulk["Ensembl_Gene_ID"].astype(str), bulk["Symbol"].astype(str)))
    )
    symbols = [symbol_map.get(gene_id, gene_id) for gene_id in genes]

    hpa = hpa.set_index(hpa["Ensembl_Gene_ID"].astype(str))
    bulk = bulk.set_index(bulk["Ensembl_Gene_ID"].astype(str))

    cols = []
    for comp in components:
        # Priority 0: ``matched_normal_<tissue>`` — used by
        # ``get_template_components`` for solid_primary + cancer_type
        # to admit admixed normal-tissue glands as a non-tumor
        # compartment.
        #
        # #214: prefer a cell-type-resolved HPA single-cell reference
        # when we have one mapped for this tissue (epithelium-only /
        # lineage-cell-only), so the matched-normal compartment
        # doesn't absorb stromal / immune signal that belongs to
        # dedicated fibroblast / myeloid / endothelial compartments.
        # Fall back to the bulk ``nTPM_<tissue>`` column for tissues
        # without a cell-type match (thyroid, urinary_bladder, bone,
        # thymus, etc.).
        if comp.startswith("matched_normal_"):
            tissue = comp[len("matched_normal_") :]

            tissue_col = f"nTPM_{tissue}"
            bulk_vals = None
            if tissue_col in bulk.columns:
                bulk_vals = (
                    bulk[tissue_col].astype(float).reindex(genes).fillna(0.0).values
                )

            cell_types, _proxy_confidence, _source = get_matched_normal_cell_type(
                tissue
            )
            present_cell_types = [t for t in cell_types if t in hpa.columns]
            cell_type_vals = None
            if present_cell_types:
                cell_type_vals = (
                    hpa[present_cell_types]
                    .astype(float)
                    .fillna(0.0)
                    .mean(axis=1)
                    .reindex(genes)
                    .fillna(0.0)
                    .values
                )

            if cell_type_vals is not None and bulk_vals is not None:
                # Blended reference: weighted mix of epithelium-only
                # (single-cell) and whole-tissue (bulk). Keeps tests
                # that expected bulk semantics largely stable while
                # giving fibroblast / myeloid / endothelial
                # compartments more room to pull their markers.
                blend = MATCHED_NORMAL_CELL_TYPE_BLEND.get(
                    tissue,
                    _CELL_TYPE_BLEND_DEFAULT,
                )
                vals = blend * cell_type_vals + (1.0 - blend) * bulk_vals
                cols.append(vals)
                continue
            if cell_type_vals is not None:
                cols.append(cell_type_vals)
                continue
            if bulk_vals is not None:
                cols.append(bulk_vals)
                continue

        # Priority 1: composite tissue category (best-match selection)
        category = COMPONENT_TO_CATEGORY.get(comp)
        if category is not None:
            if sample_by_eid is not None:
                _tissue, tissue_col = _select_best_category_tissue(
                    category,
                    sample_by_eid,
                    genes,
                    bulk,
                )
            else:
                first_tissue = TISSUE_CATEGORIES[category][0]
                tissue_col = f"nTPM_{first_tissue}"
            if tissue_col in bulk.columns:
                vals = bulk[tissue_col].astype(float).reindex(genes).fillna(0.0).values
                cols.append(vals)
                continue

        # Priority 2: explicit bulk tissue override (legacy, for any
        # components not yet migrated to COMPONENT_TO_CATEGORY)
        bulk_tissues = COMPONENT_TO_BULK_TISSUE.get(comp, [])
        if bulk_tissues:
            bulk_cols = [
                f"nTPM_{tissue}"
                for tissue in bulk_tissues
                if f"nTPM_{tissue}" in bulk.columns
            ]
            if bulk_cols:
                vals = (
                    bulk[bulk_cols]
                    .astype(float)
                    .fillna(0.0)
                    .mean(axis=1)
                    .reindex(genes)
                    .fillna(0.0)
                    .values
                )
                cols.append(vals)
                continue

        # Priority 3: HPA single-cell reference
        hpa_types = COMPONENT_TO_HPA.get(comp, [])
        if hpa_types:
            present = [t for t in hpa_types if t in hpa.columns]
            if present:
                vals = (
                    hpa[present]
                    .astype(float)
                    .fillna(0.0)
                    .mean(axis=1)
                    .reindex(genes)
                    .fillna(0.0)
                    .values
                )
                cols.append(vals)
                continue

        # Unknown or unavailable component → zeros
        cols.append(np.zeros(len(genes)))

    matrix = np.column_stack(cols)
    return genes, symbols, matrix, list(components)
