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

# The migrated CLI body retains argh-style @named(...) decorators on
# `analyze`, `compare_analyze`, `data`, `cancers`, `plot-expression`,
# `plot-cancer-cohorts` for documentation. They were the argh CLI's
# command-name overrides; nothing dispatches off them now (the
# trufflepig CLI in :mod:`trufflepig.cli` calls these functions
# directly by Python name). Provide a no-op so argh isn't a runtime
# dependency.
def named(_name):  # noqa: D401 - argh-style decorator shim
    """No-op decorator preserved from the pre-migration argh CLI."""

    def _wrap(fn):
        return fn

    return _wrap
import logging
from functools import lru_cache
from pathlib import Path
import re
from typing import Optional, Set

_LOGGER = logging.getLogger(__name__)

from .analyze import (
    AnalyzeConfig,
    AnalyzeRun,
    apply_sample_context_to_purity,
    build_analysis_parameters,
    build_analyze_paths,
    cancer_type_context_from_analysis,
    decomposition_purity_stability,
    discover_output_artifacts,
    reconcile_decomposition_purity,
    resolve_analyze_inputs,
    should_adopt_decomposition_purity,
    write_json,
)
from .tumor_purity import (
    analyze_sample,
    get_tumor_purity_parameters,
    plot_sample_summary,
    plot_tumor_purity,
    plot_purity_method_comparison,
    plot_cancer_type_hypotheses,
    plot_background_tissues,
    plot_mhc_expression,
)
from pirlygenes.gene_sets_cancer import (
    cancer_biomarker_genes, cancer_therapy_targets, therapy_target_gene_id_to_name, pMHC_TCE_target_gene_id_to_name, surface_TCE_target_gene_id_to_name, cancer_type_gene_sets,
)
from trufflepig.reference import (
    cancer_types,
)
from PIL import Image
from .load_expression import load_expression_data
from .load_expression import apply_expression_qc_rescue
from .clean_tpm import (
    assert_clean_tpm,
    normalize_to_reference_space,
    resolve_gene_columns,
)
from trufflepig.expression_qc import (
    expression_qc_rescue_summary_line,
    qc_category_levels_lines,
    technical_rna_component_phrase,
)
from .plot import (
    plot_gene_expression,
    plot_sample_vs_cancer,
    plot_cancer_type_mds,
    plot_cancer_type_neighborhood,
    plot_therapy_target_tissues,
    plot_cohort_heatmap,
    plot_cohort_disjoint_counts,
    plot_cohort_pca,
    plot_cohort_therapy_targets,
    plot_cohort_surface_proteins,
    plot_cohort_ctas,
    plot_priority_target_context,
    plot_priority_targets,
    get_embedding_feature_metadata,
    estimate_tumor_expression_ranges,
    plot_matched_normal_attribution,
    plot_target_attribution,
    plot_tumor_expression_ranges,
    CANCER_TYPE_ALIASES,
    CANCER_TYPE_NAMES,
)
from .decomposition import (
    decompose_sample,
    epithelial_matched_normal_component,
    get_decomposition_parameters,
    infer_sample_mode,
    plot_decomposition_candidates,
    plot_decomposition_component_breakdown,
    plot_decomposition_composition,
)
from .sample_context import (
    heuristic_support_label,
    infer_sample_context,
    length_pair_display_label,
    library_prep_clause,
    library_prep_display_label,
    plot_degradation_index,
    plot_expression_concentration_curve_qc,
    plot_expression_concentration_top_features_qc,
    plot_reference_mtdna_fraction_qc,
    plot_reference_technical_rna_burden_qc,
    plot_sample_context,
)
from .sample_quality import assess_sample_quality
from .rna_qc import collect_rna_quant_qc, rna_quant_qc_markdown
from .therapy_response import infer_mapk_activity_sources, score_therapy_signatures
from .cancer_type_signal_matrix import write_cancer_type_signal_artifacts
from .format import (
    render_fold,
    render_fraction_no_decimal,
    render_tpm,
)
from .reporting import (
    cancer_code_display_name,
    canonical_target_symbol,
    cancer_key_genes_lookup_for_analysis,
    context_expression_band_cell,
    expression_independent_indication,
    expression_independent_interpretation,
    expression_independent_rna_context,
    filter_current_therapy_targets,
    format_missing_observation_cell,
    format_missing_observation_interp,
    normal_expression_context,
    report_disease_state_text,
    resolved_subtype_code_for_analysis,
    select_mismatch_repair_channel_for_report,
    subtype_curation_scope_note,
    target_observation_state,
    therapy_path_context,
    therapy_path_rank,
    therapy_rna_context_conflict,
    therapy_row_rna_context_inactive,
    therapy_state_caution,
    tumor_band_cell,
    target_interpretation_summary,
    target_reliability_reasons,
    target_reliability_status,
    tpm_semantics_note,
    tumor_attribution_band_text,
    tumor_attribution_context,
)

_DATASET_SOURCES = {
    "ADC-approved": "Wiley, doi:10.1002/cac2.12517",
    "ADC-trials": "EJC, doi:10.1016/j.ejca.2023.113342",
    "ADC-withdrawn": "FDA records",
    "bispecific-antibodies-approved": "FDA/EMA approvals",
    "cancer-driver-genes": "Bailey et al. 2018, Cell",
    "cancer-driver-variants": "Bailey et al. 2018, Cell",
    "cancer-surfaceome": "Hu et al. 2021, Nature Cancer (TCSA L3)",
    "cancer-testis-antigens": "CTpedia, CTexploreR, daSilva2017 + HPA v23",
    "CAR-T-approved": "FDA approvals",
    # class1-mhc-presentation-pathway and interferon-response merged into gene-sets.csv
    "housekeeping-genes": "Eisenberg & Levanon 2013",
    "gene-sets": "Pre-resolved gene sets (immune, oncogenic, DNA repair, MHC1 presentation, IFN response)",
    "multispecific-tcell-engager-trials": "Literature curation 2024",
    "pan-cancer-expression": "HPA v23 (nTPM) + GDC/STAR (median TPM), 33 TCGA types",
    "radioligand-targets": "Literature curation 2026",
    "surface-proteins": "Bausch-Fluck et al. 2018, PNAS (SURFY/CSPA)",
    "TCR-T-approved": "FDA approvals",
    "TCR-T-trials": "Literature curation 2024",
    "disease-state-rules": "Per-cancer narrative rule catalog (#202)",
    "narrative-gene-sets": "Named gene sets for #202 disease-state rules",
    "degenerate-subtype-pairs": "Subtype disambiguation catalog (#198)",
    "fusion-surrogate-expression": "Fusion-surrogate gene catalog (#198)",
    "rare-cancer-rna-surrogates": "Rare-cancer RNA-surrogate hypothesis rules",
    "rare-cancer-fusion-rules": "Rare-cancer direct-fusion hypothesis rules",
    "fusion-expression-effects": "Fusion downstream-expression consistency rules",
    "mutation-expression-effects": "Mutation/CNV downstream-expression hypothesis rules",
}

_GENERIC_OUTPUT_NAME_PARTS = {
    "",
    "sample",
    "input",
    "expression",
    "gene-expression",
    "gene_expression",
    "gene-expression.csv",
    "gene_expression_salmon",
    "transcript_expression_salmon",
    "rna_stringtie_gene_expression",
    "gene_abundance",
    "abundance",
    "quant",
    "quant.gene_tpm",
    "pipeline_results",
    "mcdb-workflow_results",
    "fda-submission",
    "analysis",
    "results",
    "processed",
    "salmon_rich_quant",
}

_PREFERRED_SAMPLE_ID_PATTERNS = [
    re.compile(r"(?i)\b(pfo\d{3,})\b"),
    re.compile(r"(?i)\b(rs)\b"),
    re.compile(r"\b(TL-\d{2}-[A-Z0-9]+)\b"),
    re.compile(r"\b(BG\d{5,})\b"),
    re.compile(r"\b(PSNLDx\d+)\b", re.IGNORECASE),
]


@named("data")
def print_dataset_info():
    """List all bundled datasets with row counts and sources."""
    import pandas as pd

    from pirlygenes.load_dataset import get_all_csv_paths

    total_size = 0
    print("\nBundled datasets (shipped with pip install, no downloads needed):\n")
    print(f"  {'Dataset':<40s} {'Rows':>6s}  {'Size':>8s}  Source")
    print(f"  {'─' * 40}  {'─' * 6}  {'─' * 8}  {'─' * 40}")
    for csv_path in get_all_csv_paths():
        df = pd.read_csv(str(csv_path))
        name = csv_path.name.removesuffix(".gz").removesuffix(".csv")
        size = csv_path.stat().st_size
        total_size += size
        if size >= 1024 * 1024:
            size_str = "%.1f MB" % (size / 1024 / 1024)
        else:
            size_str = "%.0f KB" % (size / 1024)
        source = _DATASET_SOURCES.get(name, "")
        print(f"  {name:<40s} {len(df):>6d}  {size_str:>8s}  {source}")
    print(f"\n  Total: {total_size / 1024 / 1024:.1f} MB\n")

    # Cancer types
    types = cancer_types()
    print(f"  Cancer types ({len(types)}):\n")
    print(f"  {'Code':<6s}  {'Full name':<45s}  Aliases")
    print(f"  {'─' * 6}  {'─' * 45}  {'─' * 30}")
    # Build reverse alias map: code -> [aliases]
    code_to_aliases = {}
    for alias, code in CANCER_TYPE_ALIASES.items():
        code_to_aliases.setdefault(code, []).append(alias)
    for code in types:
        full_name = CANCER_TYPE_NAMES.get(code, "")
        aliases = ", ".join(sorted(code_to_aliases.get(code, [])))
        print(f"  {code:<6s}  {full_name:<45s}  {aliases}")
    print()


_CLINICAL_GROUP_ORDER = {
    "Carcinomas and epithelial tumors": 0,
    "Sarcoma, bone, and soft-tissue tumors": 1,
    "Hematologic malignancies": 2,
    "Neuroendocrine and neural-crest tumors": 3,
    "CNS tumors": 4,
    "Embryonal and blastemal tumors": 5,
    "Other pediatric solid tumors": 6,
    "Endocrine tumors": 7,
    "Melanoma": 8,
    "Germ-cell tumors": 9,
    "Salivary tumors": 10,
}


def clinical_group_for_row(row) -> str:
    """Map a registry row to its reader-facing clinical group.

    Sarcoma membership routes through the canonical
    ``tumor_type_ontology.is_sarcoma_code`` (i.e. pirlygenes'
    ``sarcoma_lineage_codes()``) rather than a hardcoded SARC_* allowlist,
    so new sarcoma subtypes are grouped correctly without a trufflepig edit.
    """
    from .tumor_type_ontology import is_sarcoma_code

    getter = row.get if hasattr(row, "get") else (lambda k, d=None: row[k])
    family_value = str(getter("family", "") or "").strip()
    code = str(getter("code", "") or "").strip()
    if family_value.startswith("carcinoma-") or code in {"NUTM", "THYM"}:
        return "Carcinomas and epithelial tumors"
    if is_sarcoma_code(code) or family_value in {
        "sarcoma",
        "pediatric-bone",
        "pediatric-soft",
    }:
        return "Sarcoma, bone, and soft-tissue tumors"
    if family_value.startswith("heme-"):
        return "Hematologic malignancies"
    if family_value in {"neuroendocrine", "net", "pediatric-net"}:
        return "Neuroendocrine and neural-crest tumors"
    if family_value in {"cns", "pediatric-cns"}:
        return "CNS tumors"
    if family_value in {"embryonal", "pediatric-embryonal"}:
        return "Embryonal and blastemal tumors"
    if family_value.startswith("pediatric-"):
        return "Other pediatric solid tumors"
    if family_value == "endocrine":
        return "Endocrine tumors"
    if family_value == "melanoma":
        return "Melanoma"
    if family_value == "germ-cell":
        return "Germ-cell tumors"
    if family_value in {"salivary"}:
        return "Salivary tumors"
    return family_value or "Other"


@named("cancers")
def print_cancer_registry(
    family: str = None,
    tissue: str = None,
    show_all: bool = True,
    details: bool = False,
):
    """List cancer types in the registry with data-availability columns.

    :param family: Restrict to one family (e.g. "sarcoma", "heme-myeloid", "net").
    :param tissue: Restrict to one primary tissue (e.g. "bone", "lymph_node").
    :param show_all: Show full registry including rows without expression data.
    :param details: Print per-row expression and curation provenance.
    """
    from collections import defaultdict

    from trufflepig.cancer_ontology import cancer_type_registry
    from pirlygenes.gene_sets_cancer import (

        cancer_biomarker_genes, cancer_therapy_targets,

    )

    from trufflepig.reference import (

        tcga_deconvolved_expression, subtype_deconvolved_expression,

    )
    from pirlygenes.load_dataset import get_data

    registry_df = cancer_type_registry()
    df = registry_df
    if family:
        df = df[df["family"] == family]
    if tissue:
        df = df[df["primary_tissue"] == tissue]
    if df.empty:
        print("No registry entries match the given filters.")
        return

    tcga_deconv = tcga_deconvolved_expression()
    tcga_codes = (
        set(tcga_deconv["cancer_code"].unique()) if tcga_deconv is not None else set()
    )
    sub_deconv = subtype_deconvolved_expression()
    expression_sources_by_code = defaultdict(set)
    exact_expression_sources_by_code = defaultdict(set)
    child_subtype_refs_by_parent = defaultdict(set)
    canonical_code_by_upper = {
        str(row["code"]).strip().upper(): str(row["code"]).strip()
        for _, row in registry_df.iterrows()
        if str(row["code"]).strip()
    }

    def _canonical_code(value):
        code = str(value).strip()
        return canonical_code_by_upper.get(code.upper(), code)

    def _source_cohort_values(frame):
        # Deconvolved-expression frames may omit ``source_cohort`` (the TCGA
        # deconvolved reference carries no per-cohort provenance), so guard
        # the column rather than relying on a DataFrame ``.get`` default.
        if "source_cohort" not in frame.columns:
            return set()
        return {
            str(v)
            for v in frame["source_cohort"].dropna().unique()
            if str(v).strip()
        }

    parent_by_code = {
        str(row["code"]).strip(): _parent
        for _, row in registry_df.iterrows()
        for _parent in [str(row.get("parent_code") or "").strip()]
        if str(row["code"]).strip()
    }
    if tcga_deconv is not None:
        for code, group_df in tcga_deconv.groupby("cancer_code"):
            code_s = _canonical_code(code)
            sources = _source_cohort_values(group_df)
            expression_sources_by_code[code_s].update(sources)
            exact_expression_sources_by_code[code_s].update(sources)
    if sub_deconv is not None and "subtype" in sub_deconv.columns:
        for code, group_df in sub_deconv.groupby("cancer_code"):
            code_s = _canonical_code(code)
            sources = _source_cohort_values(group_df)
            nonblank_subtype = group_df["subtype"].fillna("").astype(str).str.strip()
            if nonblank_subtype.eq("").all():
                expression_sources_by_code[code_s].update(sources)
                exact_expression_sources_by_code[code_s].update(sources)
                parent = parent_by_code.get(code_s, "")
                if parent:
                    child_subtype_refs_by_parent[parent].add(code_s)
        for subtype, group_df in sub_deconv.dropna(subset=["subtype"]).groupby(
            "subtype"
        ):
            subtype_s = _canonical_code(subtype)
            if not subtype_s or subtype_s.lower() == "nan":
                continue
            sources = _source_cohort_values(group_df)
            expression_sources_by_code[subtype_s].update(sources)
            exact_expression_sources_by_code[subtype_s].update(sources)
            parent = parent_by_code.get(subtype_s, "")
            if not parent and "_" in subtype_s:
                parent = subtype_s.split("_", 1)[0]
            if parent:
                child_subtype_refs_by_parent[parent].add(subtype_s)

    def _count_by_code(dataset_name, column):
        try:
            data = get_data(dataset_name)
        except Exception:
            return {}
        if column not in data.columns:
            return {}
        return data[column].dropna().astype(str).value_counts().to_dict()

    lineage_counts = _count_by_code("lineage-genes", "Cancer_Type")
    matched_normal_counts = defaultdict(int)
    for dataset_name in (
        "tumor-up-vs-matched-normal",
        "heme-tumor-up-vs-matched-normal",
    ):
        for code, count in _count_by_code(dataset_name, "cancer_code").items():
            matched_normal_counts[code] += int(count)
    therapy_axis_counts = defaultdict(int)
    try:
        therapy_axes = get_data("therapy-response-signatures")
        for value in therapy_axes.get("cancer_context", []).dropna():
            for part in str(value).split(";"):
                code = part.strip()
                if code and code != "pan_cancer":
                    therapy_axis_counts[code] += 1
    except Exception:
        pass

    def _safe_count(fn, *args, **kwargs):
        try:
            return len(fn(*args, **kwargs) or [])
        except Exception:
            return 0

    def _clean(value):
        """Normalise pandas NaN / blank / 'nan' to the empty string."""
        if value is None:
            return ""
        s = str(value).strip()
        if s.lower() == "nan":
            return ""
        return s

    def _resolve_lookup(row):
        """Hierarchical key-genes lookup.

        Prefers the explicit ``subtype_key`` column when non-empty
        (e.g. SARC_LMS -> SARC + 'leiomyosarcoma'). Falls back to the
        code-suffix heuristic only when the code abbreviation already
        matches the key-genes subtype literally (SARC_GIST -> 'gist');
        otherwise returns the code itself with no subtype so the
        lookup falls back to the parent's full-panel union."""
        code = _clean(row["code"])
        parent_code = _clean(row.get("parent_code"))
        subtype_key = _clean(row.get("subtype_key"))
        if subtype_key and parent_code:
            return parent_code, subtype_key
        if parent_code and code.startswith(parent_code + "_"):
            suffix = code[len(parent_code) + 1 :].lower()
            if suffix:
                return parent_code, suffix
        return code, None

    def _source_label(value):
        source = _clean(value)
        labels = {
            "BEATAML_OHSU_2022": "BeatAML OHSU 2022",
            "GSE118014_ALVAREZ_2018": "GEO GSE118014",
            "GSE299759_MEIJER_2026": "GEO GSE299759",
            "GSE75885_DELESPAUL_2017": "GEO GSE75885",
            "ICGC": "ICGC",
            "LITERATURE_CURATED": "curated literature",
            "MMRF_COMMPASS": "MMRF CoMMpass",
            "TARGET_ALL_2018": "TARGET ALL 2018",
            "TARGET_NBL_2018": "TARGET NBL 2018",
            "TARGET_OS_2020": "TARGET OS 2020",
            "TARGET_RMS_2014": "TARGET RMS 2014",
            "TARGET_RT_2017": "TARGET rhabdoid 2017",
            "TARGET_AML_2018": "TARGET AML 2018",
            "TARGET_UNSPECIFIED": "TARGET",
            "TARGET_WT_2015": "TARGET Wilms 2015",
            "SCLC_UCOLOGNE_2015": "SCLC UCologne 2015",
            "TCGA_BRCA_PAM50": "TCGA BRCA PAM50",
            "TCGA_HNSC": "TCGA HNSC",
            "TCGA_HNSC_HPV": "TCGA HNSC HPV strata",
            "TCGA_LUAD": "TCGA LUAD",
            "TCGA_LUAD_MUT": "TCGA LUAD mutation strata",
            "TCGA_XENA_TOIL": "TCGA/Xena/TOIL",
            "TREEHOUSE_POLYA_25_01": "Treehouse v25.01 PolyA",
            "TREEHOUSE_RIBOD_25_01": "Treehouse v25.01 RiboD",
            "TREEHOUSE_v25.01": "Treehouse v25.01",
        }
        return labels.get(source, source or "unknown")

    def _source_prefix(value):
        source = _clean(value)
        prefixes = {
            "BEATAML_OHSU_2022": "BeatAML",
            "GSE118014_ALVAREZ_2018": "GEO",
            "GSE299759_MEIJER_2026": "GEO",
            "GSE75885_DELESPAUL_2017": "GEO",
            "ICGC": "ICGC",
            "MMRF_COMMPASS": "CoMMpass",
            "SCLC_UCOLOGNE_2015": "UCologne",
            "TARGET_ALL_2018": "TARGET",
            "TARGET_AML_2018": "TARGET",
            "TARGET_NBL_2018": "TARGET",
            "TARGET_OS_2020": "TARGET",
            "TARGET_RMS_2014": "TARGET",
            "TARGET_RT_2017": "TARGET",
            "TARGET_UNSPECIFIED": "TARGET",
            "TARGET_WT_2015": "TARGET",
            "TCGA_BRCA_PAM50": "TCGA/PAM50",
            "TCGA_HNSC": "TCGA",
            "TCGA_HNSC_HPV": "TCGA/HPV",
            "TCGA_LUAD": "TCGA",
            "TCGA_LUAD_MUT": "TCGA/mut",
            "TCGA_XENA_TOIL": "TCGA",
            "TREEHOUSE_POLYA_25_01": "Treehouse",
            "TREEHOUSE_RIBOD_25_01": "Treehouse/RiboD",
            "TREEHOUSE_v25.01": "Treehouse",
        }
        return prefixes.get(source, source or "unknown")

    def _family_label(value):
        family_value = _clean(value)
        labels = {
            "carcinoma-breast": "breast carcinomas",
            "carcinoma-gi": "gastrointestinal carcinomas",
            "carcinoma-gu": "genitourinary / gynecologic carcinomas",
            "carcinoma-head-neck": "head and neck carcinomas",
            "carcinoma-lung": "lung carcinomas",
            "carcinoma-mesothelial": "mesothelial tumors",
            "cns": "adult CNS tumors",
            "endocrine": "endocrine tumors",
            "germ-cell": "germ-cell tumors",
            "heme-bcell": "B-cell malignancies",
            "heme-myeloid": "myeloid malignancies",
            "heme-plasma": "plasma-cell malignancies",
            "heme-tcell": "T-cell malignancies",
            "melanoma": "melanoma",
            "neuroendocrine": "neuroendocrine tumors",
            "embryonal": "embryonal / blastemal tumors",
            "net": "neuroendocrine tumors",
            "pediatric-bone": "pediatric bone tumors",
            "pediatric-cns": "pediatric CNS tumors",
            "pediatric-embryonal": "pediatric embryonal tumors",
            "pediatric-eye": "pediatric eye tumors",
            "pediatric-liver": "pediatric liver tumors",
            "pediatric-net": "pediatric neural-crest tumors",
            "pediatric-soft": "pediatric soft-tissue tumors",
            "rare": "rare epithelial / notochord tumors",
            "salivary": "salivary gland tumors",
            "sarcoma": "adult / refined sarcoma labels",
            "thymic": "thymic tumors",
        }
        return labels.get(family_value, family_value.replace("-", " ") or "other")

    def _expression_label(code):
        labels = []
        if code in tcga_codes:
            labels.append(f"TCGA:{code}")
        for source in sorted(exact_expression_sources_by_code.get(code, set())):
            prefix = _source_prefix(source)
            label = f"{prefix}:{code}"
            if label not in labels:
                labels.append(label)
        if code in child_subtype_refs_by_parent:
            labels.append(f"{len(child_subtype_refs_by_parent[code])} child refs")
        return "; ".join(labels) if labels else "none"

    def _exact_expression_source_label(code):
        sources = sorted(exact_expression_sources_by_code.get(code, set()))
        return "; ".join(_source_label(source) for source in sources) or "-"

    def _expression_source_label(code):
        sources = set(expression_sources_by_code.get(code, set()))
        for child_code in child_subtype_refs_by_parent.get(code, set()):
            sources.update(expression_sources_by_code.get(child_code, set()))
        sources = sorted(sources)
        return "; ".join(_source_label(source) for source in sources) or "-"

    def _curation_source_label(row):
        source = _source_label(row.get("source_cohort"))
        pmid = _clean(row.get("source_pmid"))
        if pmid:
            return f"{source}; {pmid}"
        return source

    _clinical_group = clinical_group_for_row
    group_order = _CLINICAL_GROUP_ORDER

    def _group_sort_key(group):
        return group_order.get(str(group), 99)

    def _clip(value, width):
        text = "" if value is None else str(value).rstrip()
        if not text.strip():
            text = "-"
        if len(text) <= width:
            return text
        if width <= 1:
            return text[:width]
        return text[: width - 1] + "."

    def _row(cells, widths, aligns=None):
        aligns = aligns or ["<"] * len(cells)
        parts = []
        for value, width, align in zip(cells, widths, aligns):
            clipped = _clip(value, width)
            parts.append(f"{clipped:{align}{width}}")
        return "  ".join(parts).rstrip()

    def _rule(widths):
        return "  ".join("-" * width for width in widths)

    records = []
    for _, row in df.iterrows():
        code = _clean(row["code"])
        lookup_code, subtype = _resolve_lookup(row)
        if subtype is not None:
            bm_count = _safe_count(cancer_biomarker_genes, lookup_code, subtype=subtype)
            tg_df = cancer_therapy_targets(lookup_code, subtype=subtype)
        else:
            bm_count = _safe_count(cancer_biomarker_genes, lookup_code)
            tg_df = cancer_therapy_targets(lookup_code) if code else None
        tg_df = filter_current_therapy_targets(tg_df)
        tg_count = 0 if tg_df is None else len(tg_df)
        expression = _expression_label(code)
        parent = _clean(row.get("parent_code"))
        lineage_count = int(lineage_counts.get(code, 0))
        matched_normal_count = int(matched_normal_counts.get(code, 0))
        therapy_axis_count = int(therapy_axis_counts.get(code, 0))

        if not show_all and expression == "none" and not bm_count and not tg_count:
            continue

        records.append(
            {
                "group": _clinical_group(row),
                "family": _clean(row.get("family")),
                "is_child": bool(parent),
                "code": code,
                "parent": parent or "-",
                "name": _clean(row.get("name")),
                "tissue": _clean(row.get("primary_tissue")),
                "template": _clean(row.get("primary_template")),
                "expression": expression,
                "expression_source": _expression_source_label(code),
                "exact_expression_source": _exact_expression_source_label(code),
                "biomarkers": bm_count,
                "targets": tg_count,
                "lineage": lineage_count,
                "matched_normal": matched_normal_count,
                "therapy_axes": therapy_axis_count,
                "curation_source": _curation_source_label(row),
            }
        )

    print(f"\nCancer-type registry — {len(records)} entries\n")
    print("Legend:")
    print("  Expr ref: source-qualified numeric expression context, e.g. TCGA:SARC or Treehouse:SARC_SYN.")
    print("            Cancer codes stay unprefixed; source prefixes only describe the expression reference.")
    print("  Normal means matched-normal marker rows; Response means therapy-response axis rows.")
    print("  Parent rows are coarse/default reference scopes; child rows are refined labels.")
    print("  Curation source is separate from expression data; pass --details to print row-level provenance.\n")

    coverage_fields = [
        ("Expression ref", lambda r: r["expression"] != "none"),
        ("Biomarkers", lambda r: r["biomarkers"] > 0),
        ("Targets", lambda r: r["targets"] > 0),
        ("Lineage", lambda r: r["lineage"] > 0),
        ("Matched normal", lambda r: r["matched_normal"] > 0),
        ("Response axes", lambda r: r["therapy_axes"] > 0),
    ]
    print("Coverage audit:")
    audit_widths = [24, 10, 10]
    print(_row(["Layer", "With data", "Missing"], audit_widths, ["<", ">", ">"]))
    print(_rule(audit_widths))
    for label, predicate in coverage_fields:
        have = sum(1 for record in records if predicate(record))
        print(_row([label, have, len(records) - have], audit_widths, ["<", ">", ">"]))

    expression_source_examples = defaultdict(list)
    for record in records:
        for source in record["exact_expression_source"].split("; "):
            if source and source != "-":
                expression_source_examples[source].append(record["code"])
    if expression_source_examples:
        print("\nExpression sources represented by exact row refs:")
        for source, codes in sorted(expression_source_examples.items()):
            examples = ", ".join(codes[:8])
            suffix = "" if len(codes) <= 8 else f", ... (+{len(codes) - 8})"
            print(f"- {source}: {len(codes)} rows ({examples}{suffix})")

    gap_rows = []
    for group_name in sorted({r["group"] for r in records}, key=_group_sort_key):
        group_records = [r for r in records if r["group"] == group_name]
        missing_expression = [r["code"] for r in group_records if r["expression"] == "none"]
        missing_targets = [r["code"] for r in group_records if r["targets"] == 0]
        missing_biomarkers = [r["code"] for r in group_records if r["biomarkers"] == 0]
        if missing_expression or missing_targets or missing_biomarkers:
            gap_rows.append(
                [
                    group_name,
                    len(missing_expression),
                    ", ".join(missing_expression[:10])
                    + (f", ... (+{len(missing_expression) - 10})" if len(missing_expression) > 10 else ""),
                    len(missing_biomarkers),
                    len(missing_targets),
                ]
            )
    if gap_rows:
        print("\nMissing-data focus:")
        for row in gap_rows:
            group_name, no_expr, gap_examples, no_biomarkers, no_targets = row
            print(
                f"- {group_name}: no expr {no_expr}, no biomarkers "
                f"{no_biomarkers}, no targets {no_targets}; expr gaps: "
                f"{gap_examples or '-'}"
            )

    records = sorted(
        records,
        key=lambda r: (
            _group_sort_key(r["group"]),
            r["family"],
            r["is_child"],
            r["code"],
        ),
    )
    current_group = None
    row_widths = [18, 8, 27, 10, 7, 7, 6, 8, 16, 36]
    row_aligns = ["<", "<", "<", ">", ">", ">", ">", ">", "<", "<"]
    current_family = None
    show_family_sections = False
    for record in records:
        if record["group"] != current_group:
            current_group = record["group"]
            group_records = [r for r in records if r["group"] == current_group]
            family_values = sorted({r["family"] for r in group_records})
            families = ", ".join(f"{f} ({_family_label(f)})" for f in family_values)
            show_family_sections = len(family_values) > 1
            current_family = None
            print(f"\nClinical group: {current_group} ({len(group_records)} entries)")
            print(f"Families: {families}")
            parent_rows = [
                r
                for r in group_records
                if any(child["parent"] == r["code"] for child in group_records)
            ]
            if parent_rows:
                print("Parent scopes:")
                for parent_row in parent_rows:
                    child_codes = [
                        child["code"]
                        for child in group_records
                        if child["parent"] == parent_row["code"]
                    ]
                    examples = ", ".join(child_codes[:8])
                    suffix = (
                        ""
                        if len(child_codes) <= 8
                        else f", ... (+{len(child_codes) - 8})"
                    )
                    print(
                        f"  {parent_row['code']}: fallback parent for "
                        f"{len(child_codes)} refined labels ({examples}{suffix})"
                    )
            print(
                _row(
                    [
                        "Code",
                        "Parent",
                        "Expr ref",
                        "Biomarkers",
                        "Targets",
                        "Lineage",
                        "Normal",
                        "Response",
                        "Tissue",
                        "Name",
                    ],
                    row_widths,
                    row_aligns,
                )
            )
            print(_rule(row_widths))

        if show_family_sections and record["family"] != current_family:
            current_family = record["family"]
            print(f"[{current_family}] {_family_label(current_family)}")

        code_display = (
            f"  {record['code']}" if record["parent"] != "-" else record["code"]
        )
        print(
            _row(
                [
                    code_display,
                    record["parent"],
                    record["expression"],
                    record["biomarkers"],
                    record["targets"],
                    record["lineage"],
                    record["matched_normal"],
                    record["therapy_axes"],
                    record["tissue"],
                    record["name"],
                ],
                row_widths,
                row_aligns,
            )
        )
        if details:
            print(
                "  "
                + _clip(record["code"], 18).ljust(18)
                + "  expression data="
                + _clip(record["expression_source"], 52)
                + "; exact="
                + _clip(record["exact_expression_source"], 36)
                + "; curation="
                + _clip(record["curation_source"], 48)
            )
    print()


def _parse_always_label_genes(always_label_genes: Optional[str]) -> Set[str]:
    if always_label_genes is None:
        return set()
    return {token.strip() for token in always_label_genes.split(",") if token.strip()}


_MT_EXPECTED_MISSING_PREPS = frozenset({"poly_a", "exome_capture"})

# The AR-transactivation output panel used by the CRPC-pattern
# narrative rule moved to ``narrative-gene-sets.csv`` (#202). The
# disease-state rule engine looks it up by name (``AR_targets``) so
# the synthesis layer stays data-driven.

# Genes annotated in the therapy_response AR_signaling *up* panel but
# also core ISGs — used to tag per-gene surface-target fold-changes as
# "IFN-driven" when the IFN_response axis is active.
_CORE_ISG_SURFACE = frozenset(
    {
        "HLA-A",
        "HLA-B",
        "HLA-C",
        "HLA-F",
        "HLA-E",
        "HLA-DPA1",
        "HLA-DPB1",
        "HLA-DQA1",
        "HLA-DQB1",
        "HLA-DRA",
        "HLA-DRB1",
        "B2M",
        "TAP1",
        "TAP2",
        "STAT1",
        "IRF1",
        "ISG15",
        "IFIT1",
        "IFIT3",
        "MX1",
        "OAS1",
        "OAS2",
        "CXCL9",
        "CXCL10",
    }
)


def _metabolic_axes_rows(evidence) -> list[tuple[str, str, str]]:
    """Return (axis, signal, therapy) rows for active metabolic programs (#158).

    The tissue-composition screen already computes proliferation / hypoxia / glycolysis
    channel scores on :class:`TumorEvidenceScore`. This helper converts
    those scores into therapy-report rows so readers see actionable
    metabolic axes (CA9-directed ADC, MCT inhibitors, CDK4/6) without
    having to cross-reference the tissue-composition narrative against the target
    landscape. Rows emit only when the corresponding channel is
    meaningfully elevated so the section is silent on non-metabolic
    tumors.
    """
    if evidence is None:
        return []
    rows: list[tuple[str, str, str]] = []

    hypoxia = float(getattr(evidence, "hypoxia", 0.0) or 0.0)
    ca9_tpm = float(getattr(evidence, "ca9_tpm", 0.0) or 0.0)
    if hypoxia >= 0.5 or ca9_tpm >= 20.0:
        rows.append(
            (
                "Hypoxia / CA9",
                f"CA9 observed {ca9_tpm:.0f} TPM (hypoxia score {hypoxia:.2f})",
                "Acetazolamide; CA9-directed ADCs (DS-6157a / trials). Consider "
                "HIF-2α inhibitors (belzutifan) in ccRCC context.",
            )
        )

    glycolysis = float(getattr(evidence, "glycolysis", 0.0) or 0.0)
    glyc_fold = float(getattr(evidence, "glycolysis_geomean_fold", 0.0) or 0.0)
    if glycolysis >= 0.5:
        rows.append(
            (
                "Glycolysis / MCT",
                f"Panel geomean {glyc_fold:.1f}× over median (score {glycolysis:.2f})",
                "MCT1/4 inhibitors (AZD3965 trials); LDHA / HK2 inhibitors "
                "(preclinical). Metformin where comorbidity supports it.",
            )
        )

    prolif = float(getattr(evidence, "proliferation", 0.0) or 0.0)
    prolif_log2 = float(getattr(evidence, "prolif_log2", 0.0) or 0.0)
    if prolif >= 0.6:
        rows.append(
            (
                "Proliferation / cell-cycle",
                f"Panel log2-TPM {prolif_log2:.2f} (score {prolif:.2f})",
                "CDK4/6 inhibitors (palbociclib / ribociclib) where cancer-type "
                "context supports; WEE1 inhibitors (adavosertib trials).",
            )
        )

    return rows


def compose_disease_state_narrative(analysis) -> str:
    """Synthesize the disease-state narrative line (#78).

    As of #202 the per-cancer logic lives in ``disease-state-rules.csv``
    and ``narrative-gene-sets.csv``. This function only prepares the
    three inputs the rule engine consumes (axis states + lineage-
    retained / lineage-collapsed gene sets) and dispatches.

    Adding a new cancer-specific narrative is a CSV edit — no Python
    change. This keeps the analyze pipeline uniform across cancer
    types; differences live in data, not in ``if/elif cancer_code``
    branches.
    """
    from .disease_state_rules import synthesize_disease_state

    cancer_code = analysis.get("cancer_type")
    therapy_scores = analysis.get("therapy_response_scores") or {}
    purity = analysis.get("purity") or {}
    components = purity.get("components") or {}
    lineage = components.get("lineage") or {}
    per_gene = lineage.get("per_gene") or []

    retained: set[str] = set()
    collapsed: set[str] = set()
    for g in per_gene:
        sym = g.get("gene")
        est = g.get("purity")
        if not sym or est is None:
            continue
        if est >= 0.30:
            retained.add(sym)
        elif est < 0.05:
            collapsed.add(sym)

    axis_states: dict[str, Optional[str]] = {}
    for axis_name, score in therapy_scores.items():
        axis_states[axis_name] = getattr(score, "state", None)

    return synthesize_disease_state(
        cancer_code=cancer_code,
        axis_states=axis_states,
        retained=retained,
        collapsed=collapsed,
    )


def annotate_surface_targets_with_cross_signals(ranges_df, therapy_scores):
    """Return a map ``{symbol: note}`` annotating surface / intracellular
    targets whose elevation is likely *driven by* an active therapy-
    response axis rather than tumor-cell specificity (issue #78).

    Currently covers IFN: when ``IFN_response`` axis is active and the
    gene is a core ISG, tag with "IFN-driven". The renderer attaches
    these notes inline in the target tables so a high fold-change
    isn't read as pure tumor-cell selectivity.
    """
    ifn = therapy_scores.get("IFN_response")
    if ifn is None or getattr(ifn, "state", None) != "up":
        return {}
    return {sym: "IFN-driven" for sym in _CORE_ISG_SURFACE}


def _tumor_tpm_by_symbol_from_ranges(ranges_df) -> dict[str, float]:
    if ranges_df is None or "symbol" not in ranges_df.columns:
        return {}
    value_col = "attr_tumor_tpm" if "attr_tumor_tpm" in ranges_df.columns else None
    if value_col is None and "median_est" in ranges_df.columns:
        value_col = "median_est"
    if value_col is None:
        return {}
    import pandas as pd

    mapping: dict[str, float] = {}
    from .common import ranges_records
    for row in ranges_records(ranges_df):
        symbol = str(row.get("symbol") or "").strip()
        if not symbol or symbol.lower() == "nan":
            continue
        value = pd.to_numeric(row.get(value_col), errors="coerce")
        if pd.isna(value):
            continue
        mapping[symbol] = float(value)
    return mapping


def _store_alteration_effect_reasoning(
    analysis,
    *,
    fusion_records=None,
    alteration_records=None,
    sample_tpm_by_symbol=None,
    ranges_df=None,
    cancer_code=None,
) -> dict:
    """Attach uncertainty-aware downstream alteration-effect evidence."""
    tumor_tpm_by_symbol = _tumor_tpm_by_symbol_from_ranges(ranges_df)
    if tumor_tpm_by_symbol:
        analysis["tumor_tpm_by_symbol"] = tumor_tpm_by_symbol
    try:
        from .fusion_effects import (
            infer_fusion_expression_hypotheses,
            match_fusion_expression_effects,
        )

        fusion_effects = match_fusion_expression_effects(
            fusion_records or [],
            sample_tpm_by_symbol,
            tumor_tpm_by_symbol=tumor_tpm_by_symbol,
        )
        fusion_hypotheses = (
            []
            if fusion_records
            else infer_fusion_expression_hypotheses(
                sample_tpm_by_symbol,
                tumor_tpm_by_symbol=tumor_tpm_by_symbol,
                cancer_code=cancer_code or analysis.get("cancer_type"),
                candidate_trace=analysis.get("candidate_trace") or [],
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[alteration-effects] fusion-effect evaluation skipped: {exc}")
        fusion_effects = []
        fusion_hypotheses = []

    try:
        from .alteration_effects import infer_mutation_expression_hypotheses

        mutation_hypotheses = infer_mutation_expression_hypotheses(
            sample_tpm_by_symbol,
            tumor_tpm_by_symbol=tumor_tpm_by_symbol,
            cancer_code=cancer_code or analysis.get("cancer_type"),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[alteration-effects] mutation-effect evaluation skipped: {exc}")
        mutation_hypotheses = []

    analysis["fusion_expression_effects"] = fusion_effects
    analysis["fusion_expression_hypotheses"] = fusion_hypotheses
    analysis["mutation_expression_hypotheses"] = mutation_hypotheses
    pathway_inferences = infer_mapk_activity_sources(analysis, ranges_df=ranges_df)
    analysis["pathway_activity_inferences"] = pathway_inferences
    if alteration_records is not None:
        analysis["alteration_records"] = [
            record.public_dict() if hasattr(record, "public_dict") else dict(record)
            for record in alteration_records
        ]
    analysis["alteration_effect_summary"] = {
        "fusion_effects": len(fusion_effects),
        "fusion_expression_hypotheses": len(fusion_hypotheses),
        "mutation_expression_hypotheses": len(mutation_hypotheses),
        "pathway_activity_inferences": len(pathway_inferences),
        "supplied_alterations": len(alteration_records or []),
        "expression_source": "tumor_inferred" if tumor_tpm_by_symbol else "bulk",
    }
    return analysis["alteration_effect_summary"]


def _select_actionable_plot_genes(
    ranges_df, cancer_code, target_panel=None, max_genes=18
):
    """Return a compact gene list for the default actionable-target plot.

    The simple figure should not drift away from the curated report
    panel, but it also should not hide strong sample-specific surface
    targets that sit outside that panel (for example FAP in a bone-rich
    sarcoma sample). Use the curated panel when available, then append
    strong therapy-linked sample hits ranked by tumor-source support.
    """
    try:
        from .plot_target_deep_dive import actionable_surface_targets
    except Exception:
        actionable_surface_targets = None

    curated = []
    if (
        target_panel is not None
        and len(target_panel)
        and "symbol" in target_panel.columns
    ):
        curated = [
            str(sym).strip()
            for sym in target_panel["symbol"].dropna().astype(str)
            if str(sym).strip() and str(sym).strip().lower() != "nan"
        ]
    elif actionable_surface_targets is not None:
        try:
            curated = actionable_surface_targets(cancer_code)
        except Exception:
            curated = []

    if ranges_df is None or len(ranges_df) == 0 or "symbol" not in ranges_df.columns:
        return list(dict.fromkeys(curated))[:max_genes]

    reliability_rank = {"supported": 0, "provisional": 1, "unsupported": 2}
    extras = []
    therapy_supported_set: set[str] = set()
    from .common import ranges_records
    for row in ranges_records(ranges_df):
        sym = str(row.get("symbol") or "").strip()
        if not sym or sym.lower() == "nan":
            continue
        try:
            observed = float(row.get("observed_tpm") or 0.0)
        except Exception:
            observed = 0.0
        if observed < 1.0:
            continue
        therapies = str(row.get("therapies") or "").strip()
        therapy_list = [
            piece.strip() for piece in therapies.split(",") if piece.strip()
        ]
        therapy_supported = row.get("therapy_supported") is True
        category = str(row.get("category") or "").strip()
        is_surface = bool(row.get("is_surface"))
        if not therapy_list and not therapy_supported and category != "therapy_target":
            continue
        if not is_surface and category != "therapy_target":
            continue
        reliability = target_reliability_status(row)
        extras.append(
            (
                reliability_rank.get(reliability, 9),
                0 if therapy_supported or therapy_list else 1,
                0 if is_surface else 1,
                -len(therapy_list),
                -float(row.get("attr_tumor_tpm") or 0.0),
                -observed,
                sym,
            )
        )
        if therapy_supported:
            therapy_supported_set.add(sym)

    ranked_extras = [sym for *_unused, sym in sorted(extras)]
    # Sample-specific therapy_supported hits are the strongest signal we
    # have for a target, so they ride ahead of the curated cohort panel —
    # otherwise a 5-slot figure full of cohort defaults starves the one
    # genuinely actionable sample finding (e.g. FAP in a bone-rich
    # sarcoma).  Other sample hits stay after the curated panel.
    priority_extras = [sym for sym in ranked_extras if sym in therapy_supported_set]
    other_extras = [sym for sym in ranked_extras if sym not in therapy_supported_set]
    genes: list[str] = []
    for sym in priority_extras + list(curated) + other_extras:
        if sym not in genes:
            genes.append(sym)
    return genes[:max_genes]


def _filter_quality_flags_against_context(flags, sample_context):
    """Drop / rewrite quality flags that the step-1 SampleContext
    already explains (issue #77).

    Specifically: the "Suspicious MT fraction" warning fires whenever
    MT detection is near zero, but that's *expected* under the poly-A
    and exome-capture library preps we already inferred. In those
    cases the warning is noise — replace it with an informational
    line so readers see the reason rather than a false alarm.
    """
    if sample_context is None:
        return list(flags)
    prep = getattr(sample_context, "library_prep", None)
    out = []
    for flag in flags:
        if "Suspicious MT fraction" in flag and prep in _MT_EXPECTED_MISSING_PREPS:
            prep_label = library_prep_display_label(prep)
            out.append(
                f"MT fraction near zero — consistent with {prep_label} "
                "library prep; degradation signal from MT fold is not "
                "assessable (this is informational, not a warning)"
            )
        else:
            out.append(flag)
    return out


def _technical_qc_quality_flags(
    *,
    expression_qc_rescue: Optional[dict],
    raw_expression_scale_qc: Optional[dict],
    rna_quant_qc: Optional[dict],
) -> list[str]:
    """Promote expression-scale/read-level QC into the shared quality state."""

    out: list[str] = []
    rescue = expression_qc_rescue or {}
    raw_scale = raw_expression_scale_qc or {}
    if rescue.get("high_burden"):
        removed = float(rescue.get("removed_fraction") or 0.0)
        technical_phrase = technical_rna_component_phrase(
            rescue.get("qc_class_shares") or raw_scale
        )
        top_removed = rescue.get("top_removed_genes") or []
        top_clause = ""
        if top_removed:
            top = top_removed[0]
            gene = str(top.get("gene") or "").strip()
            qc_class = str(top.get("qc_class") or "").strip()
            share = float(top.get("share") or 0.0)
            if gene:
                top_clause = f"; top removed feature {gene}"
                if qc_class:
                    top_clause += f" ({qc_class}"
                    if share:
                        top_clause += f", {share:.0%} of raw TPM"
                    top_clause += ")"
        component_clause = (
            f" ({technical_phrase}; {removed:.0%} removed)"
            if technical_phrase
            else f" ({removed:.0%} removed)"
        )
        out.append(
            "Expression concentration QC: raw TPM dominated by technical RNA "
            f"features{component_clause}{top_clause}"
        )
    elif raw_scale.get("warnings"):
        out.append(f"Expression concentration QC: {raw_scale['warnings'][0]}")

    for warning in (rna_quant_qc or {}).get("warnings") or []:
        out.append(f"RNA quantification QC: {warning}")
    return out


def _render_vs_tcga_cell(row):
    """Render the "vs TCGA" column for a target-table row.

    The naive ``render_fold(row["pct_cancer_median"])`` produces ``∞×``
    whenever the TCGA cohort's tumor-component deconvolves to ~0 for
    this gene — which happens in two biologically distinct cases:

    - **not_in_cohort**: the raw cohort median is itself ~0 (the gene
      genuinely isn't expressed at cohort median — the CTA-in-solid-
      cohort case). Here ``∞×`` is not wrong but is degenerate for a
      clinician; instead show the raw cohort TPM so the reader sees
      "gene is absent from the reference cohort".
    - **tme_explained**: the raw cohort was non-trivial but the TME
      deconvolution zeroed the tumor component. The ratio isn't
      meaningful because the deconvolution subtracted everything;
      label as "TME-only" with the raw cohort TPM for context.

    Keeps the column narrow; preserves information rather than capping.
    """
    import math as _math
    import pandas as _pd

    from .common import VS_TCGA_REF_FLOOR_TPM

    state = row.get("tcga_ref_state")
    vs_tcga = row.get("pct_cancer_median")
    cohort_tpm = row.get("tcga_cohort_median_tpm")

    if state == "finite" and _pd.notna(vs_tcga):
        # #85.3: a "finite" cohort median that is still below the detection
        # floor makes the fold a near-zero-denominator artifact (e.g. NTRK1
        # "137×" off a ~0.16 TPM cohort median). Show the reference TPM
        # instead — the same honest rendering as ``not_in_cohort`` below — so a
        # noise-amplified ratio isn't surfaced as over-expression. The numeric
        # ``pct_cancer_median`` is untouched; only this display cell changes.
        if cohort_tpm is not None and cohort_tpm < VS_TCGA_REF_FLOOR_TPM:
            return f"ref {cohort_tpm:.2f} TPM" if cohort_tpm > 0 else "ref 0"
        return render_fold(vs_tcga)
    if state == "not_in_cohort":
        if cohort_tpm is not None and cohort_tpm > 0:
            return f"ref {cohort_tpm:.2f} TPM"
        return "ref 0"
    if state == "tme_explained":
        if cohort_tpm is not None:
            return f"TME-only ({cohort_tpm:.1f} TPM)"
        return "TME-only"
    if state == "both_absent":
        return "—"
    # Back-compat: when tcga_ref_state is missing (older ranges_df),
    # fall back to the raw fold. Inf still lands here and renders ∞×;
    # new runs always set the state.
    if vs_tcga is None or (isinstance(vs_tcga, float) and _math.isinf(vs_tcga)):
        if cohort_tpm is not None and cohort_tpm > 0:
            return f"ref {cohort_tpm:.2f} TPM"
        return "ref 0"
    return render_fold(vs_tcga) if _pd.notna(vs_tcga) else "—"


def _format_attribution_cell(row):
    """Compact per-target Attribution cell for targets.md (#108, #128).

    Renders "tumor T / comp C" when the decomposition produced a per-
    compartment attribution for this gene; returns "—" when attribution
    isn't available.

    For genes flagged ``broadly_expressed`` (#128), appends a brief
    "broadly expr." tag so the reader can't mistake a high
    tumor-attribution number for a specificity claim.
    """
    try:
        observed = float(row.get("observed_tpm") or 0.0)
    except (TypeError, ValueError):
        observed = 0.0
    if observed <= 0:
        return "—"
    attribution = row.get("attribution")
    try:
        attr_tumor = float(row.get("attr_tumor_tpm") or 0.0)
        top_comp = row.get("attr_top_compartment") or ""
        top_tpm = float(row.get("attr_top_compartment_tpm") or 0.0)
    except (TypeError, ValueError):
        return "—"
    broadly = bool(row.get("broadly_expressed"))
    amplified = bool(row.get("amplified_over_healthy"))
    over_predicted = bool(row.get("matched_normal_over_predicted"))
    sm_leakage = bool(row.get("smooth_muscle_stromal_leakage"))
    try:
        amp_fold = float(row.get("amplification_fold") or 0.0)
    except (TypeError, ValueError):
        amp_fold = 0.0

    def _tag():
        # Over-prediction is the strongest caveat — when matched-normal
        # alone predicts more of the gene than we observed, the
        # attribution math can't say anything reliable about the tumor
        # contribution (#131). Surface this first so a reader looking
        # at e.g. KLK3 / TACSTD2 on a CRPC sample knows the "tumor 0"
        # number isn't independent evidence that tumor cells aren't
        # expressing the gene — it means the reference over-predicted.
        if over_predicted:
            return "matched-normal over-predicted"
        # #59 item 1: smooth-muscle stromal leakage. Fibromuscular-
        # stroma density varies sample-to-sample; matched-normal
        # references carry the cohort-average and under-subtract
        # SM-lineage genes when the biopsy is SM-rich. The tumor
        # story for TAGLN / ACTA2 / MYH11 / CNN1 should be read with
        # that caveat.
        if sm_leakage:
            return "likely smooth-muscle stromal leakage"
        # Amplification is a positive specificity signal and takes
        # precedence over the broadly-expressed caution (#128). A gene
        # that's broadly expressed at baseline but observed >= 5× over
        # the peak healthy tissue is telling an amplification /
        # overexpression story — HER2 in HER2+ BRCA, MDM2 in WD/DD-LPS.
        if amplified and amp_fold >= 1.5:
            return f"amplified {amp_fold:.1f}\u00d7"
        if broadly:
            return "broadly expr."
        return ""

    tag = _tag()
    if not attribution or not isinstance(attribution, dict):
        if tag:
            return tag
        return "—"
    if not top_comp:
        base = f"tumor {attr_tumor:.0f}"
    else:
        comp_label = top_comp.replace("_", " ")
        base = f"tumor {attr_tumor:.0f} / {comp_label} {top_tpm:.0f}"
    if tag:
        base += f" · {tag}"
    return base


def _ci_confidence_tier(overall_lower, overall_upper):
    """Map a (lower, upper) span to a confidence tier (issue #79).

    Reader-facing tags on the purity estimate so a 19–100% CI is
    visibly different from a 58–70% CI in the report.

    A *zero-width* CI (``lower == upper == estimate``) is degenerate —
    the estimator saw no per-gene variation, which happens on synthetic
    / deterministic inputs (TCGA cohort medians, decomposition
    templates, expected-expression probes). Surfacing that as "high
    confidence" is misleading — the estimator couldn't produce
    uncertainty, not because the answer is certain but because the
    input has no spread to bound it with (#161).
    """
    try:
        span = float(overall_upper) - float(overall_lower)
    except (TypeError, ValueError):
        return "unknown"
    if span <= 1e-9:
        return "degenerate"
    if span < 0.15:
        return "high"
    if span < 0.35:
        return "moderate"
    return "low"


def _default_output_dir() -> str:
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"trufflepig-{ts}"


# ── Output-directory lock (#82) ─────────────────────────────────────────
#
# Two concurrent ``analyze`` invocations pointed at the same
# ``--output-dir`` would happily overwrite each other's artifacts
# because ``_clean_prefix_outputs`` wipes stale files at start and both
# runs race to write to the same filenames. The result is a silently
# inconsistent report (figures from one run, markdowns from the other).
#
# Cheap fix: write an advisory ``.trufflepig.lock`` with the runner's
# pid + start time. A second invocation into the same directory while
# the lock-owner is still alive bails out with a clear error directing
# the user to ``--force`` or a different ``--output-dir``.
#
# Not a true concurrency guarantee — a race still exists between the
# stale-lock unlink and the new write — but the common accidental
# double-launch case is caught and diagnosed instead of silently
# corrupting the output.
_LOCKFILE_NAME = ".trufflepig.lock"


def _pid_is_alive(pid: int) -> bool:
    """Cheap liveness check.

    ``os.kill(pid, 0)`` raises :class:`ProcessLookupError` when the
    process does not exist and :class:`PermissionError` when it
    exists but the caller can't signal it (which still counts as
    alive for our purposes — we don't want a lock held by another
    user to look stale just because we can't ping it).
    """
    if pid <= 0:
        return False
    import os as _os

    try:
        _os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _acquire_output_dir_lock(out_dir, force: bool = False):
    """Claim the output dir for this process. Returns the lockfile Path
    so the caller can unlink on exit.

    Raises :class:`RuntimeError` with a clear message when another
    trufflepig process is already writing there and ``force`` is
    False. Stale locks (pid no longer alive) are reclaimed with a
    console note.
    """
    import json
    import os as _os
    from datetime import datetime

    lock_path = out_dir / _LOCKFILE_NAME
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text() or "{}")
            holder_pid = int(payload.get("pid") or 0)
            holder_started = str(payload.get("started_at") or "unknown time")
        except (ValueError, OSError):
            holder_pid = 0
            holder_started = "unknown time"
        if holder_pid and _pid_is_alive(holder_pid):
            if not force:
                raise RuntimeError(
                    f"Another trufflepig analyze process (pid={holder_pid}, "
                    f"started {holder_started}) is writing to {out_dir}. "
                    "Use a different --output-dir, wait for it to finish, "
                    "or pass --force to override (only do this if you're "
                    "sure the lock is stale)."
                )
            print(
                f"[output] --force: ignoring live lock held by pid={holder_pid} "
                f"(started {holder_started})"
            )
        else:
            print(
                f"[output] Cleaning stale lockfile (pid={holder_pid} "
                f"not running, started {holder_started})"
            )

    lock_path.write_text(
        json.dumps(
            {
                "pid": _os.getpid(),
                "started_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )
    return lock_path


def _clean_prefix_outputs(out_dir: Path, prefix_path: str) -> int:
    """Delete stale files from a prior run sharing the same output prefix.

    Removes files in ``out_dir`` (and the ``figures/`` subdir) whose name
    starts with ``{prefix_basename}-``, so a partial rerun cannot leave
    misleading artifacts from an earlier successful run. Only touches
    files matching our prefix — anything else the user placed in the
    directory is preserved (issue #30).

    Also removes ``{prefix_basename}-vs-cancer`` scatter subdirectories.
    Returns the number of files deleted.
    """
    base = Path(prefix_path).name
    if not base:
        return 0
    stale_prefix = f"{base}-"
    removed = 0
    for directory in (out_dir, out_dir / "figures", out_dir / "figures" / "deprecated"):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.name.startswith(stale_prefix):
                path.unlink()
                removed += 1
            elif path.is_dir() and path.name.startswith(stale_prefix):
                for child in path.iterdir():
                    if child.is_file():
                        child.unlink()
                        removed += 1
                try:
                    path.rmdir()
                except OSError:
                    pass
    return removed


def _sanitize_output_basename(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if "/" in text or "\\" in text:
        text = Path(text).name
    stem = Path(text).stem.strip()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("._-")
    return slug


def _derive_sample_display_id(
    input_path: str, sample_id_value: Optional[str] = None
) -> str:
    explicit = _sanitize_output_basename(sample_id_value)
    if explicit:
        return explicit

    path = Path(str(input_path))
    parts = [path.stem] + [parent.name for parent in path.parents if parent.name]
    for pattern in _PREFERRED_SAMPLE_ID_PATTERNS:
        for part in parts:
            match = pattern.search(part)
            if match:
                derived = _sanitize_output_basename(match.group(1))
                if derived:
                    return derived

    for part in parts:
        derived = _sanitize_output_basename(part)
        if derived and derived.lower() not in _GENERIC_OUTPUT_NAME_PARTS:
            return derived
    return "sample"


_TX_HEADER_TOKENS = frozenset(
    {
        "name",
        "target_id",
        "transcript_id",
        "transcript",
        "transcriptid",
        "targetid",
        "effectivelength",
        "numreads",
    }
)
_GENE_HEADER_TOKENS = frozenset(
    {
        "gene",
        "gene symbol",
        "gene_symbol",
        "genesymbol",
        "geneid",
        "gene_id",
        "ensembl_gene_id",
    }
)


def _sniff_input_level(path: str) -> str:
    """Read the header of a tabular file and guess whether it contains
    transcript-level or gene-level quantification.

    Returns ``"transcript"`` or ``"gene"``.
    """
    import csv

    with open(path) as f:
        reader = csv.reader(
            f, delimiter="\t" if path.endswith((".sf", ".tsv")) else ","
        )
        try:
            header = next(reader)
        except StopIteration:
            return "gene"
    lower = {h.strip().lower() for h in header}
    if lower & _TX_HEADER_TOKENS:
        return "transcript"
    return "gene"


# When ``--no-figures`` is set, ``_analyze_body`` rebinds every plot_*
# attribute it can reach to a no-op for the duration of the call.
# Each rebinding is appended to this queue so ``analyze()``'s
# ``finally`` clause can restore the originals before returning —
# important because a second invocation in the same process (tests,
# batch runners, service mode) would otherwise inherit the no-op
# state and silently skip figure generation.
_NOOP_PLOT_RESTORE_QUEUE: list[tuple[object, str, object]] = []


def _restore_noop_plot_patches() -> None:
    """Reverse every entry in ``_NOOP_PLOT_RESTORE_QUEUE`` and clear it.

    Best-effort: a failed restore on one entry must not block the
    others (e.g. a module that was unloaded between patch and
    restore). Idempotent: clearing the queue means a second call
    returns immediately.
    """
    while _NOOP_PLOT_RESTORE_QUEUE:
        target, attr, original = _NOOP_PLOT_RESTORE_QUEUE.pop()
        try:
            if isinstance(target, dict):
                target[attr] = original
            else:
                setattr(target, attr, original)
        except (AttributeError, TypeError):
            pass


@named("analyze")
def analyze(
    input_path: str,
    output_dir: str = "trufflepig-output",
    output_image_prefix: Optional[str] = None,
    aggregate_gene_expression: bool = False,
    genes: Optional[str] = None,
    transcripts: Optional[str] = None,
    label_genes: Optional[str] = None,
    gene_name_col: Optional[str] = None,
    gene_id_col: Optional[str] = None,
    sample_id_col: Optional[str] = None,
    sample_id_value: Optional[str] = None,
    output_dpi: int = 300,
    plot_height: float = 14.0,
    plot_aspect: float = 1.4,
    cancer_type: Optional[str] = None,
    sample_mode: str = "auto",
    tumor_context: str = "auto",
    site_hint: Optional[str] = None,
    met_site: Optional[str] = None,
    decomposition_templates: Optional[str] = None,
    hla_types: Optional[str] = None,
    fusions: Optional[str] = None,
    alterations: Optional[str] = None,
    alignment_qc: Optional[str] = None,
    expression_qc_rescue: str = "auto",
    expression_qc_remove_noncoding: bool = False,
    therapy_target_top_k: int = 10,
    therapy_target_tpm_threshold: float = 30.0,
    deprecated_figures: bool = False,
    no_figures: bool = False,
    force: bool = False,
):
    """Analyze gene expression from a quantification file.

    The positional argument auto-detects whether the input is
    transcript-level (salmon quant.sf, kallisto abundance.tsv) or
    gene-level. Transcript-level inputs are aggregated to gene-level
    automatically. Use --genes / --transcripts for explicit control
    or to supply both simultaneously::

        trufflepig run --sample quant.sf --workspace out                  # auto-detect: transcript → aggregate
        trufflepig run --sample gene_tpm.csv --workspace out               # auto-detect: gene-level
        trufflepig run --genes g.csv --transcripts quant.sf --workspace out  # explicit both
    """
    config = AnalyzeConfig(
        input_path=input_path,
        output_dir=output_dir,
        output_image_prefix=output_image_prefix,
        aggregate_gene_expression=aggregate_gene_expression,
        genes=genes,
        transcripts=transcripts,
        label_genes=label_genes,
        gene_name_col=gene_name_col,
        gene_id_col=gene_id_col,
        sample_id_col=sample_id_col,
        sample_id_value=sample_id_value,
        output_dpi=output_dpi,
        plot_height=plot_height,
        plot_aspect=plot_aspect,
        cancer_type=cancer_type,
        sample_mode=sample_mode,
        tumor_context=tumor_context,
        site_hint=site_hint,
        met_site=met_site,
        decomposition_templates=decomposition_templates,
        hla_types=hla_types,
        fusions=fusions,
        alterations=alterations,
        alignment_qc=alignment_qc,
        expression_qc_rescue=expression_qc_rescue,
        expression_qc_remove_noncoding=expression_qc_remove_noncoding,
        therapy_target_top_k=therapy_target_top_k,
        therapy_target_tpm_threshold=therapy_target_tpm_threshold,
        deprecated_figures=deprecated_figures,
        no_figures=no_figures,
        force=force,
    )

    # Validate met_site before any I/O so bad values fail fast.
    if config.met_site is not None:
        from .plot import MET_SITE_TISSUE_AUGMENTATION as _MET_SITE_MAP

        if config.met_site not in _MET_SITE_MAP:
            raise ValueError(
                f"--met-site must be one of {sorted(_MET_SITE_MAP.keys())}, got {config.met_site!r}"
            )

    resolution = resolve_analyze_inputs(config, sniff_input_level=_sniff_input_level)
    for note in resolution.notes:
        print(note)

    paths = build_analyze_paths(
        config,
        resolution,
        default_output_dir=_default_output_dir,
        derive_sample_display_id=_derive_sample_display_id,
        sanitize_output_basename=_sanitize_output_basename,
    )
    out_dir = paths.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] Writing to {out_dir}/")

    # #82: advisory lock so a second concurrent analyze into the same
    # output dir fails fast instead of silently corrupting artifacts.
    lock_path = _acquire_output_dir_lock(out_dir, force=config.force)
    try:
        _analyze_body(AnalyzeRun(config=config, inputs=resolution, paths=paths))
    finally:
        # Restore any plot_* functions rebound to no-ops by
        # ``--no-figures``; harmless when nothing was patched.
        _restore_noop_plot_patches()
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


@named("compare-analyze")
def compare_analyze(
    output_dirs: str,
    output_path: str = "analyze-comparison.md",
    title: str = "Analyze Sample Comparison",
):
    """Build a comparison Markdown report from analyze output directories.

    ``output_dirs`` is a comma- or newline-separated list of directories
    emitted by ``pirlygenes analyze``.
    """
    from .analyze import build_analyze_comparison_markdown

    dirs = [
        part.strip() for part in re.split(r"[,\n]+", str(output_dirs)) if part.strip()
    ]
    if len(dirs) < 2:
        raise ValueError(
            "Provide at least two analyze output directories, separated by commas"
        )
    markdown = build_analyze_comparison_markdown(dirs, title=title)
    Path(output_path).write_text(markdown)
    print(f"[report] Wrote {output_path}")


def _analyze_body(run: AnalyzeRun):
    config = run.config
    resolution = run.inputs
    paths = run.paths

    input_path = resolution.gene_input
    out_dir = paths.out_dir
    output_image_prefix = config.output_image_prefix
    aggregate_gene_expression = resolution.aggregate_gene_expression
    label_genes = config.label_genes
    gene_name_col = config.gene_name_col
    gene_id_col = config.gene_id_col
    sample_id_col = config.sample_id_col
    sample_id_value = config.sample_id_value
    output_dpi = config.output_dpi
    plot_height = config.plot_height
    plot_aspect = config.plot_aspect
    # Single bundled context for plotting. During the per-module migration
    # (trufflepig figure-context series), each plot_* function will accept
    # `ctx` and early-out when ctx.enabled is False; for now the existing
    # positional kwargs stay live and ctx is constructed only so callers
    # adopting it can begin to use it.
    from .plot_context import PlottingContext as _PlottingContext

    plot_ctx = _PlottingContext(
        enabled=not bool(config.no_figures),
        output_dir=out_dir,
        prefix=output_image_prefix,
        dpi=output_dpi,
        height=plot_height,
        aspect=plot_aspect,
        deprecated_figures=bool(config.deprecated_figures),
    )
    # ``--no-figures`` propagation: call sites now check
    # ``plot_ctx.enabled`` before doing plot work, but keep this as a
    # safety net for already-imported plotting functions while the
    # figure-context migration continues. The originals are appended to
    # ``_NOOP_PLOT_RESTORE_QUEUE`` so ``analyze()``'s try/finally
    # restores them on exit — a second invocation in the same process
    # (tests, batch runners, the upcoming service mode) does not
    # inherit the no-op state.
    if not plot_ctx.enabled:
        import sys as _sys
        def _noop_plot(*_a, **_k):
            return None
        # Patch main.py's own module globals — this catches the
        # ``from .X import plot_Y`` bindings at the top of this file
        # (these are the actual names that the call sites resolve to,
        # not the source modules).
        _main_globals = globals()
        for _key in list(_main_globals.keys()):
            if _key.startswith("plot_") and callable(_main_globals[_key]):
                _NOOP_PLOT_RESTORE_QUEUE.append((_main_globals, _key, _main_globals[_key]))
                _main_globals[_key] = _noop_plot
        # Also patch source modules so any function-level
        # ``from .X import plot_Y`` later in _analyze_body also gets
        # the no-op binding.
        for _mod_name, _mod in list(_sys.modules.items()):
            if _mod is None or not _mod_name.startswith("trufflepig."):
                continue
            if not any(_mod_name.startswith(p) for p in (
                "trufflepig.plot",
                "trufflepig.tumor_purity",
                "trufflepig.provenance",
            )):
                continue
            for _attr in dir(_mod):
                if not _attr.startswith("plot_"):
                    continue
                try:
                    _orig = getattr(_mod, _attr)
                    if callable(_orig):
                        _NOOP_PLOT_RESTORE_QUEUE.append((_mod, _attr, _orig))
                        setattr(_mod, _attr, _noop_plot)
                except (AttributeError, TypeError):
                    pass
    cancer_type = config.cancer_type
    sample_mode = config.sample_mode
    tumor_context = config.tumor_context
    site_hint = config.site_hint
    met_site = config.met_site
    transcript_path = resolution.transcript_input
    fusion_paths = config.fusion_path_list()
    alteration_inputs = config.alteration_input_list()
    therapy_target_top_k = config.therapy_target_top_k
    therapy_target_tpm_threshold = config.therapy_target_tpm_threshold
    deprecated_figures = bool(config.deprecated_figures)
    sample_display_id = paths.sample_display_id
    prefix = paths.prefix
    analysis_cancer_type, report_scope_cancer_type = _analysis_input_cancer_type(
        cancer_type
    )

    stale_removed = _clean_prefix_outputs(out_dir, prefix)
    legacy_default_prefix = str(out_dir / "sample")
    if not output_image_prefix and prefix != legacy_default_prefix:
        stale_removed += _clean_prefix_outputs(out_dir, legacy_default_prefix)
    if stale_removed:
        print(f"[output] Removed {stale_removed} stale files from prior run")

    df_expr = load_expression_data(
        input_path,
        aggregate_gene_expression=aggregate_gene_expression,
        gene_name_col=gene_name_col,
        gene_id_col=gene_id_col,
        sample_id_col=sample_id_col,
        sample_id_value=sample_id_value,
        transcript_path=transcript_path,
    )
    df_expr_raw = df_expr
    fusion_records = []
    if fusion_paths:
        from .fusions import parse_fusion_files

        fusion_records = parse_fusion_files(fusion_paths)
        print(
            f"[fusion] Parsed {len(fusion_records)} fusion calls from "
            f"{len(fusion_paths)} file(s)"
        )
    alteration_records = []
    if alteration_inputs:
        from .alterations import parse_alteration_inputs

        alteration_records = parse_alteration_inputs(alteration_inputs)
        print(
            f"[alteration] Parsed {len(alteration_records)} alteration calls from "
            f"{len(alteration_inputs)} input(s)"
        )
    run.note_step(
        "input",
        outputs={
            "rows": int(len(df_expr)),
            "columns": [str(c) for c in df_expr.columns],
            "fusion_files": len(fusion_paths),
            "fusion_records": len(fusion_records),
            "alteration_inputs": len(alteration_inputs),
            "alteration_records": len(alteration_records),
        },
    )
    forced_labels = _parse_always_label_genes(label_genes)
    template_overrides = config.template_overrides()
    raw_expression_scale_qc = dict(df_expr_raw.attrs.get("expression_scale_qc") or {})
    raw_technical_phrase = technical_rna_component_phrase(raw_expression_scale_qc)
    if raw_expression_scale_qc and raw_technical_phrase:
        print(f"[load] Technical RNA burden: {raw_technical_phrase}")
    for warning in raw_expression_scale_qc.get("warnings") or []:
        print(f"[load] Expression scale QC: {warning}")
    rna_quant_qc = collect_rna_quant_qc(
        input_path,
        gene_df=df_expr_raw,
        transcript_path=transcript_path,
        alignment_qc_path=config.alignment_qc,
    )
    if rna_quant_qc.get("available"):
        if rna_quant_qc.get("summary"):
            print(f"[rna-qc] {rna_quant_qc['summary']}")
        for warning in rna_quant_qc.get("warnings") or []:
            print(f"[rna-qc WARNING] {warning}")
        run.note_step(
            "rna_quant_qc",
            status="warning" if rna_quant_qc.get("warnings") else "completed",
            outputs={
                "source": rna_quant_qc.get("source"),
                "summary": rna_quant_qc.get("summary"),
                "warnings": len(rna_quant_qc.get("warnings") or []),
            },
            warnings=list(rna_quant_qc.get("warnings") or []),
        )

    df_expr, expression_qc_rescue = apply_expression_qc_rescue(
        df_expr_raw,
        mode=config.expression_qc_rescue,
        remove_noncoding=config.expression_qc_remove_noncoding,
    )
    # Conform the sample to the cohort reference's clean-TPM space (#74): the
    # reference ships strict-technical-RNA zeroed, but a clean_tpm_v4 input keeps
    # that compartment at a fixed fraction, diluting biological genes ~15-25%
    # relative to the reference. Deterministically zero+renormalize so sample and
    # reference share one space regardless of the clean state the input arrived in.
    _sample_label_col, _sample_id_col = resolve_gene_columns(df_expr)
    df_expr = normalize_to_reference_space(
        df_expr,
        value_cols=["TPM"],
        label_col=_sample_label_col,
        id_col=_sample_id_col,
    )
    assert_clean_tpm(
        df_expr,
        value_cols=["TPM"],
        label_col=_sample_label_col,
        id_col=_sample_id_col,
        context="analysis sample expression",
    )
    expression_scale_qc = dict(df_expr.attrs.get("expression_scale_qc") or {})
    if expression_qc_rescue.get("applied"):
        removed = float(expression_qc_rescue.get("removed_fraction") or 0.0)
        high_burden = bool(expression_qc_rescue.get("high_burden"))
        top_removed = expression_qc_rescue.get("top_removed_genes") or []
        top_clause = ""
        if top_removed:
            top = top_removed[0]
            top_clause = (
                f"; top removed feature {top.get('gene')} "
                f"({top.get('qc_class')}, {float(top.get('share') or 0.0):.0%})"
            )
        print(
            "[load] Expression technical-RNA normalization: zeroed mtDNA/rRNA/"
            f"pseudogene-like features covering {removed:.0%} of raw TPM and "
            "renormalized downstream TPM"
            f"{top_clause}"
        )
        run.note_step(
            "expression_qc_rescue",
            status="warning" if high_burden else "completed",
            outputs={
                "mode": expression_qc_rescue.get("mode"),
                "removed_fraction": expression_qc_rescue.get("removed_fraction"),
                "removed_gene_count": expression_qc_rescue.get("removed_gene_count"),
                "high_burden": high_burden,
                "renormalization_factor": expression_qc_rescue.get(
                    "renormalization_factor"
                ),
            },
            warnings=[
                "Downstream cancer/target/pathway calculations used mtDNA/rRNA-rescued TPM"
            ]
            if high_burden
            else [],
        )
    elif expression_qc_rescue.get("reason"):
        run.note_step(
            "expression_qc_rescue",
            status="completed",
            outputs={
                "mode": expression_qc_rescue.get("mode"),
                "applied": False,
                "reason": expression_qc_rescue.get("reason"),
                "removed_fraction": expression_qc_rescue.get("removed_fraction"),
            },
        )

    # Sample context: infer library prep and preservation before
    # cancer-type inference. Downstream steps (purity CIs, decomposition,
    # tumor-value adjustment, reporting) read from it as the base layer
    # of expression expectations.
    print("[context] Inferring sample context (library prep + preservation)...")
    sample_context = infer_sample_context(df_expr_raw)
    print(f"[context] {sample_context.summary_line()}")
    for flag in sample_context.flags:
        print(f"[context] {flag}")
    run.note_step(
        "sample_context",
        outputs={
            "library_prep": sample_context.library_prep,
            "preservation": sample_context.preservation,
            "degradation_severity": sample_context.degradation_severity,
            "degradation_index": sample_context.degradation_index,
        },
        warnings=list(sample_context.flags),
    )

    # #149: tissue-composition screen. Races the sample against
    # normal-tissue and cancer-reference columns in
    # pan_cancer_expression() and checks the public mitotic
    # proliferation panel from proliferation_panel_gene_names().
    # Fixes the GTEx-style
    # mis-classification where a healthy sample gets force-called
    # into a cancer reference. Informational only — doesn't override
    # cancer-type inference; surfaces as a banner in brief/actionable
    # when the call is "healthy" or "ambiguous".
    from .healthy_vs_tumor import assess_healthy_vs_tumor

    try:
        healthy_vs_tumor = assess_healthy_vs_tumor(df_expr)
        print(f"[tissue-composition] {healthy_vs_tumor.verdict}")
    except Exception as exc:  # noqa: BLE001
        print(f"[tissue-composition] screen failed: {exc}")
        healthy_vs_tumor = None

    context_png = "%s-sample-context.png" % prefix if prefix else "sample-context.png"
    if plot_ctx.enabled:
        try:
            plot_sample_context(
                sample_context, save_to_filename=context_png, save_dpi=output_dpi
            )
            print(f"[plot] Saved sample-context diagnostic to {context_png}")
        except Exception as exc:  # noqa: BLE001 — plotting must not break analyze
            print(f"[plot] sample-context plot failed: {exc}")

    concentration_top_png = (
        "%s-expression-top-features-qc.png" % prefix
        if prefix
        else "expression-top-features-qc.png"
    )
    concentration_curve_png = (
        "%s-expression-concentration-curve-qc.png" % prefix
        if prefix
        else "expression-concentration-curve-qc.png"
    )
    if plot_ctx.enabled:
        try:
            out = plot_expression_concentration_top_features_qc(
                df_expr_raw,
                save_to_filename=concentration_top_png,
                save_dpi=output_dpi,
            )
            if out:
                print(
                    f"[plot] Saved expression top-feature QC to {concentration_top_png}"
                )
            out = plot_expression_concentration_curve_qc(
                df_expr_raw,
                save_to_filename=concentration_curve_png,
                save_dpi=output_dpi,
            )
            if out:
                print(
                    "[plot] Saved expression concentration curve QC to "
                    f"{concentration_curve_png}"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[plot] expression concentration QC plot failed: {exc}")

    reference_mtdna_qc_png = (
        "%s-qc-reference-mtdna.png" % prefix
        if prefix
        else "qc-reference-mtdna.png"
    )
    if plot_ctx.enabled:
        try:
            out = plot_reference_mtdna_fraction_qc(
                df_expr_raw,
                save_to_filename=reference_mtdna_qc_png,
                save_dpi=output_dpi,
            )
            if out:
                print(f"[plot] Saved mtDNA reference QC to {reference_mtdna_qc_png}")
        except Exception as exc:  # noqa: BLE001
            print(f"[plot] mtDNA reference QC plot failed: {exc}")

    burden_qc_png = (
        "%s-qc-reference-technical-rna-burden.png" % prefix
        if prefix
        else "qc-reference-technical-rna-burden.png"
    )
    if plot_ctx.enabled:
        try:
            out = plot_reference_technical_rna_burden_qc(
                df_expr_raw,
                save_to_filename=burden_qc_png,
                save_dpi=output_dpi,
            )
            if out:
                print(
                    "[plot] Saved combined technical-RNA burden QC to "
                    f"{burden_qc_png}"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[plot] technical-RNA burden QC plot failed: {exc}")

    # #27: gene-pair degradation index scatter. Emitted whenever any
    # degradation signal is available (including the "none" call, so
    # users can visually confirm a non-degraded sample lies on the
    # diagonal).
    degradation_png = (
        "%s-degradation-index.png" % prefix if prefix else "degradation-index.png"
    )
    if plot_ctx.enabled:
        try:
            out = plot_degradation_index(
                df_expr_raw,
                sample_context,
                save_to_filename=degradation_png,
                save_dpi=output_dpi,
            )
            if out:
                print(f"[plot] Saved degradation-index scatter to {degradation_png}")
        except Exception as exc:  # noqa: BLE001
            print(f"[plot] degradation-index plot failed: {exc}")

    # Strip plot: therapy modalities. The per-category strip plots
    # (Immune_checkpoints / Oncogenes / CTAs / ...) are emitted
    # elsewhere; the aggregate immune / tumor / antigens overview
    # panels were retired as redundant in v4.46.0.
    # Therapy modalities
    therapy_sets = {
        "TCR-T": therapy_target_gene_id_to_name("TCR-T"),
        "CAR-T": therapy_target_gene_id_to_name("CAR-T"),
        "bispecifics": therapy_target_gene_id_to_name("bispecific-antibodies"),
        "pMHC-TCEs": pMHC_TCE_target_gene_id_to_name(),
        "surface-TCEs": surface_TCE_target_gene_id_to_name(),
        "ADCs": therapy_target_gene_id_to_name("ADC"),
        "Radio": therapy_target_gene_id_to_name("radioligand"),
    }

    # The ``immune`` / ``tumor`` / ``antigens`` overview strip plots
    # duplicate the 10 curated category strip plots (Immune_checkpoints,
    # Oncogenes, Tumor_suppressors, CTAs, Cancer_surfaceome, …) that
    # this CLI also emits. Per the figure audit (docs/figure-audit.md),
    # the overview set is the redundant one — retired in 4.46.0. The
    # ``treatments`` plot stays because it's organized by therapy
    # modality (ADC / TCR-T / CAR-T / bispecific / …), not by gene-set
    # category, so it's not covered by the per-category plots.
    strip_plots = [
        ("treatments", therapy_sets),
    ]
    if plot_ctx.enabled:
        for i, (name, gene_sets) in enumerate(strip_plots):
            output_image = "%s-%s.png" % (prefix, name) if prefix else "%s.png" % name
            print(f"[plot] Generating {name} strip plot...")
            plot_gene_expression(
                df_expr,
                gene_sets=gene_sets,
                save_to_filename=output_image,
                save_dpi=output_dpi,
                plot_height=plot_height,
                plot_aspect=plot_aspect,
                always_label_genes=forced_labels,
                verbose=(i == 0),  # only log remaps on first call
                source_file=input_path,
            )

    import matplotlib.pyplot as _plt

    _plt.close("all")

    # Sample composition analysis
    print("[analysis] Running sample composition analysis...")
    analysis = analyze_sample(
        df_expr,
        cancer_type=analysis_cancer_type,
        tissue_signal=healthy_vs_tumor,
    )
    rna_inferred_cancer_type = analysis.get("cancer_type")
    rna_inferred_cancer_name = analysis.get("cancer_name")
    analysis["inferred_cancer_type"] = rna_inferred_cancer_type
    analysis["expression_reference_cancer_type"] = rna_inferred_cancer_type
    analysis["primary_expression_context"] = {
        "cancer_type": rna_inferred_cancer_type,
        "support": (
            analysis.get("candidate_trace", [{}])[0].get("support_fraction_of_top")
            if analysis.get("candidate_trace")
            else None
        ),
    }
    analysis["healthy_vs_tumor"] = healthy_vs_tumor
    fusion_findings = []
    fusion_scope_inference = None
    analysis["fusion_inputs_supplied"] = bool(fusion_paths)
    analysis["fusion_input_paths"] = list(fusion_paths)
    analysis["alteration_inputs_supplied"] = bool(alteration_inputs)
    analysis["alteration_inputs"] = list(alteration_inputs)
    analysis["expression_scale_qc"] = expression_scale_qc
    analysis["raw_expression_scale_qc"] = raw_expression_scale_qc
    analysis["expression_qc_rescue"] = expression_qc_rescue
    analysis["rna_quant_qc"] = rna_quant_qc
    analysis["fusion_records"] = [
        record.public_dict() if hasattr(record, "public_dict") else dict(record)
        for record in fusion_records
    ]
    analysis["alteration_records"] = [
        record.public_dict() if hasattr(record, "public_dict") else dict(record)
        for record in alteration_records
    ]
    if fusion_records:
        from .rare_inference import (
            infer_rare_cancer_report_scope_from_fusions,
            match_rare_cancer_fusion_rules,
        )

        fusion_findings = match_rare_cancer_fusion_rules(fusion_records)
        fusion_scope_inference = infer_rare_cancer_report_scope_from_fusions(
            fusion_records,
            analysis,
        )
        analysis["fusion_findings"] = fusion_findings
        if fusion_scope_inference:
            analysis["fusion_report_scope_inference"] = fusion_scope_inference
        run.note_step(
            "fusion_evidence",
            outputs={
                "records": len(fusion_records),
                "rule_matches": len(fusion_findings),
                "report_scope": (fusion_scope_inference or {}).get("cancer_type"),
            },
        )
    else:
        analysis["fusion_findings"] = []
    rare_scope_inference = None
    fine_scope_inference = None
    try:
        from .rare_inference import infer_rare_cancer_marker_hypotheses_from_rna
    except ImportError:
        _LOGGER.warning(
            "infer_rare_cancer_marker_hypotheses_from_rna unavailable; "
            "rare-marker hypotheses will be empty",
            exc_info=True,
        )
        analysis["rare_marker_hypotheses"] = []
    else:
        try:
            analysis["rare_marker_hypotheses"] = (
                infer_rare_cancer_marker_hypotheses_from_rna(df_expr, analysis)
            )
        except (KeyError, ValueError, TypeError):
            _LOGGER.warning(
                "infer_rare_cancer_marker_hypotheses_from_rna failed; "
                "rare-marker hypotheses will be empty",
                exc_info=True,
            )
            analysis["rare_marker_hypotheses"] = []
    cancer_type_evidence = None
    selected_scope = None
    if not cancer_type and not report_scope_cancer_type:
        (
            cancer_type_evidence,
            selected_scope,
            report_scope_cancer_type,
            rare_scope_inference,
            fine_scope_inference,
        ) = _apply_cancer_type_evidence(
            analysis,
            df_expr,
            rna_inferred_cancer_type=rna_inferred_cancer_type,
            fusion_scope_inference=fusion_scope_inference,
            report_scope_cancer_type=report_scope_cancer_type,
            rare_scope_inference=rare_scope_inference,
            fine_scope_inference=fine_scope_inference,
        )
    # Veto a deconvolved local-reference flip that crosses lineage against an agreeing bulk +
    # compartment call. No-op for fusion, rare-marker, and fine-reference rescues.
    report_scope_cancer_type = _veto_local_reference_lineage_flip(
        analysis, df_expr, report_scope_cancer_type, rna_inferred_cancer_type, selected_scope)
    if not report_scope_cancer_type:
        selected_scope = None  # veto cleared the evidence selection → fall back to the bulk classifier call
    if report_scope_cancer_type:
        selected_reference_cancer_type = (
            (selected_scope or {}).get("reference_cancer_type")
        ) or rna_inferred_cancer_type
        analysis["reference_cancer_type"] = selected_reference_cancer_type
        analysis["expression_reference_cancer_type"] = (
            (selected_scope or {}).get("expression_reference_cancer_type")
            or selected_reference_cancer_type
        )
        analysis["inferred_cancer_type"] = report_scope_cancer_type
        analysis["reference_cancer_name"] = cancer_code_display_name(
            selected_reference_cancer_type,
            selected_reference_cancer_type or rna_inferred_cancer_name,
        )
        analysis["report_scope_cancer_type"] = report_scope_cancer_type
        # Make the reported PURITY consistent with the final evidence-refined call: analyze_sample
        # bound purity (+ its one decomposition) to the pre-evidence ranking winner, so an ATRT
        # scored as LGG had a solid-mode decomposition; once evidence calls ATRT (embryonal) the
        # reported purity lineage should match. No-op when the winner already equals the final call.
        _finalize_purity_for_final_call(analysis, df_expr, report_scope_cancer_type)
        if analysis_cancer_type:
            analysis["report_scope_parent_cancer_type"] = analysis_cancer_type
        elif fine_scope_inference and fine_scope_inference.get("reference_cancer_type"):
            analysis["report_scope_parent_cancer_type"] = fine_scope_inference[
                "reference_cancer_type"
            ]
        if fusion_scope_inference:
            analysis["fusion_report_scope_inference"] = fusion_scope_inference
        if rare_scope_inference:
            analysis["rare_report_scope_inference"] = rare_scope_inference
        if fine_scope_inference:
            analysis["fine_report_scope_inference"] = fine_scope_inference
        analysis["cancer_type"] = report_scope_cancer_type
        analysis["cancer_name"] = cancer_code_display_name(
            report_scope_cancer_type, report_scope_cancer_type
        )
    analysis["cancer_type_source"] = (
        "user-specified" if cancer_type else "auto-detected"
    )
    analysis["sample_context"] = sample_context

    # Sample-context propagation: widen purity confidence intervals under
    # detected degradation (#26). A noisier sample has a noisier purity
    # estimate; we don't re-estimate, just scale the reported band and
    # attach a ``degradation_caveat`` so downstream consumers (reports,
    # downstream analyses) can cite the reason for the wider band
    # without having to re-derive it from the raw sample_context.
    apply_sample_context_to_purity(analysis, sample_context)
    analysis["sample_mode"] = infer_sample_mode(
        candidate_rows=analysis.get("candidate_trace"),
        cancer_types=[rna_inferred_cancer_type]
        if rna_inferred_cancer_type
        else ([analysis_cancer_type] if analysis_cancer_type else None),
        sample_mode=sample_mode,
    )
    analysis["analysis_constraints"] = _analysis_constraints(
        cancer_type=cancer_type,
        sample_mode=sample_mode,
        tumor_context=tumor_context,
        site_hint=site_hint,
        decomposition_templates=template_overrides,
        met_site=met_site,
        hla_types=config.hla_types,
        alterations=config.alterations,
    )
    cancer_type_context = cancer_type_context_from_analysis(
        analysis,
        supplied_cancer_type=cancer_type,
    )
    _apply_cancer_type_context_roles(analysis, cancer_type_context)
    cancer_type_context = cancer_type_context_from_analysis(
        analysis,
        supplied_cancer_type=cancer_type,
    )
    analysis["cancer_type_context"] = cancer_type_context.to_dict()
    cancer_code = cancer_type_context.code_for("report")
    reference_cancer_code = cancer_type_context.code_for("cohort")
    inferred_site_context = _infer_likely_met_site_context(
        analysis,
        explicit_site_hint=site_hint,
        explicit_met_site=met_site,
        tumor_context=tumor_context,
        decomposition_templates=template_overrides,
    )
    if inferred_site_context:
        analysis["inferred_site_context"] = inferred_site_context
    purity = analysis["purity"]
    fit_quality = analysis.get("fit_quality", {})
    candidate_trace_for_print = analysis.get("candidate_trace", [])
    top_row = candidate_trace_for_print[0] if candidate_trace_for_print else None
    if top_row:
        parts = [
            f"signature={top_row.get('signature_score', 0.0):.2f}",
            f"geomean={top_row.get('support_geomean', 0.0):.2f}",
            f"normalized={top_row.get('support_fraction_of_top', 0.0):.2f}",
        ]
        if len(candidate_trace_for_print) > 1:
            runner = candidate_trace_for_print[1]
            parts.append(
                f"(runner-up {runner['code']} {runner.get('support_fraction_of_top', 0.0):.2f})"
            )
        top_code_for_print = str(top_row.get("code") or "").strip()
        supplied_label_diverges = bool(cancer_type) and (
            top_code_for_print != str(cancer_code or "").strip()
        )
        if report_scope_cancer_type or supplied_label_diverges:
            print(
                f"[analysis] Cancer label context: {analysis['cancer_name']} ({cancer_code}); "
                f"RNA top candidate: {top_code_for_print or rna_inferred_cancer_type}, "
                + ", ".join(parts)
            )
        else:
            print(
                f"[analysis] Cancer type: {analysis['cancer_name']} ({cancer_code}), "
                + ", ".join(parts)
            )
    else:
        print(f"[analysis] Cancer type: {analysis['cancer_name']} ({cancer_code})")
    if fit_quality.get("label"):
        print(
            f"[analysis] Fit quality: {fit_quality['label']} — {fit_quality.get('message', '')}"
        )
    print(f"[analysis] Sample mode: {_sample_mode_display(analysis['sample_mode'])}")
    if analysis["analysis_constraints"]:
        print(f"[analysis] Constraints: {analysis['analysis_constraints']}")
    purity_text = _format_purity_interval(
        purity.get("overall_estimate"),
        purity.get("overall_lower"),
        purity.get("overall_upper"),
    )
    print(
        f"[analysis] {_purity_metric_label(analysis['sample_mode']).capitalize()}: "
        f"{purity_text}"
    )
    print(
        f"[analysis] Stromal enrichment: {render_fold(purity['components']['stromal']['enrichment'])} vs TCGA"
    )
    print(
        f"[analysis] Immune enrichment: {render_fold(purity['components']['immune']['enrichment'])} vs TCGA"
    )
    top_tissues = analysis["tissue_scores"][:3]
    tissue_str = ", ".join(f"{t} ({s:.2f})" for t, s, _ in top_tissues)
    print(f"[analysis] Top background signatures: {tissue_str}")
    if inferred_site_context:
        print(f"[analysis] Inferred site context: {inferred_site_context['message']}")
    mhc1 = analysis["mhc1"]
    print(
        f"[analysis] MHC-I: HLA-A={mhc1.get('HLA-A', 0):.0f}, "
        f"HLA-B={mhc1.get('HLA-B', 0):.0f}, "
        f"HLA-C={mhc1.get('HLA-C', 0):.0f}, "
        f"B2M={mhc1.get('B2M', 0):.0f} TPM"
    )
    run.note_step(
        "cancer_call",
        outputs={
            "cancer_type": cancer_code,
            "reference_cancer_type": reference_cancer_code,
            "cancer_type_context": analysis["cancer_type_context"],
            "cancer_type_source": analysis["cancer_type_source"],
            "cancer_call_rescue": (analysis.get("cancer_call_rescue") or {}).get(
                "kind"
            ),
            "sample_mode": analysis["sample_mode"],
            "purity": {
                "overall_estimate": purity.get("overall_estimate"),
                "overall_lower": purity.get("overall_lower"),
                "overall_upper": purity.get("overall_upper"),
            },
            "inferred_site_context": inferred_site_context,
        },
    )

    # Sample quality assessment — run after analysis so tissue_scores
    # are available for tissue-matched degradation baselines.
    # #77: pass the step-1 library_prep so the assessor skips the
    # "Suspicious MT fraction" override (and doesn't clobber the
    # length-pair-derived degradation level) when MT being near-zero is
    # explained by the inferred prep.
    quality = assess_sample_quality(
        df_expr_raw,
        tissue_scores=analysis.get("tissue_scores"),
        library_prep=getattr(sample_context, "library_prep", None)
        if sample_context is not None
        else None,
    )
    analysis["quality"] = quality
    # #77: filter quality flags against the step-1 SampleContext — the
    # "Suspicious MT fraction" warning is a false alarm when the
    # library prep we already inferred (RNA capture / poly-A) legitimately
    # depresses MT/rRNA signal. Same filtered list is used by the markdown reports,
    # so the three documents agree on the same set of concerns.
    filtered_flags = _filter_quality_flags_against_context(
        quality["flags"], sample_context
    )
    technical_quality_flags = _technical_qc_quality_flags(
        expression_qc_rescue=expression_qc_rescue,
        raw_expression_scale_qc=raw_expression_scale_qc,
        rna_quant_qc=rna_quant_qc,
    )
    if technical_quality_flags and filtered_flags == ["No quality concerns detected"]:
        filtered_flags = []
    filtered_flags.extend(technical_quality_flags)
    quality["filtered_flags"] = filtered_flags
    if technical_quality_flags:
        quality["has_issues"] = True
    for flag in filtered_flags:
        qtag = "[quality]" if not quality["has_issues"] else "[quality WARNING]"
        print(f"{qtag} {flag}")
    run.note_step(
        "sample_quality",
        status="warning" if quality["has_issues"] else "completed",
        outputs={
            "degradation_level": quality["degradation"]["level"],
            "culture_level": quality["culture"]["level"],
        },
        warnings=list(filtered_flags),
    )

    # Therapy-response signatures (#57) — score each applicable axis
    # (AR / ER / HER2 / MAPK-EGFR / NE / EMT / hypoxia / IFN) so the
    # report can explain *why* individual genes are high or low
    # (e.g. KLK3 ↓ + FOLH1 ↑ → AR-suppressed, consistent with ADT).
    sample_tpm_by_symbol = None
    try:
        from .common import build_sample_tpm_by_symbol

        sample_tpm_by_symbol = build_sample_tpm_by_symbol(df_expr)
        therapy_scores = score_therapy_signatures(
            sample_tpm_by_symbol, reference_cancer_code
        )
    except (KeyError, ValueError, TypeError) as exc:
        print(f"[therapy-response] scoring skipped: {exc}")
        therapy_scores = {}
    analysis["tumor_type_sanity"] = {}
    if sample_tpm_by_symbol is not None:
        try:
            from .tumor_type_ontology import tumor_type_sanity_check

            analysis["tumor_type_sanity"] = tumor_type_sanity_check(
                cancer_code,
                sample_tpm_by_symbol,
            )
        except Exception as exc:
            print(f"[cancer-type] marker sanity check skipped: {exc}")
    analysis["therapy_response_scores"] = therapy_scores
    # #149: make the tissue-composition call available to
    # brief / actionable / summary renderers downstream.
    analysis["healthy_vs_tumor"] = healthy_vs_tumor
    for cls, score in therapy_scores.items():
        if score.state in ("up", "down"):
            tag = "[therapy-state]"
            print(f"{tag} {cls}: {score.state} — {score.message}")

    # Individual summary panels (replaces the crowded 4-panel composite, #97).
    # Figure-path names are defined here (the PDF manifest below references
    # them), but the RENDER is deferred until after purity is finalized —
    # see the render block further down, just before the purity figures.
    # Rendering the sample-summary panel here would capture the
    # pre-decomposition *candidate* purity and contradict every other figure
    # and the markdown (the 78%-vs-10% headline-purity bug): purity is not
    # adopted from decomposition / lineage-panel / interval-cap until the
    # `if decomp_results:` block below.
    summary_png = "%s-sample-summary.png" % prefix if prefix else "sample-summary.png"
    # Keep the composite for backward compatibility but also emit standalone PNGs
    hypotheses_png = "%s-cancer-hypotheses.png" % prefix
    tissues_png = "%s-background-tissues.png" % prefix
    mhc_png = "%s-mhc-expression.png" % prefix

    # #136: one-figure therapy-pathway state readout. Renders the
    # disease-state narrative visually — dumbbells for each AR / NE /
    # EMT / hypoxia / IFN axis, caption restating the synthesis
    # sentence. Emitted only when at least one axis has a measurable
    # up/down geomean; skipped for empty therapy-scores dicts.
    pathway_state_png = (
        "%s-therapy-pathway-state.png" % prefix
        if prefix
        else "therapy-pathway-state.png"
    )
    if plot_ctx.enabled:
        try:
            from .plot_therapy import plot_therapy_pathway_state

            fig_ps = plot_therapy_pathway_state(
                therapy_response_scores=therapy_scores,
                cancer_code=reference_cancer_code,
                disease_state_caption=compose_disease_state_narrative(analysis),
                save_to_filename=pathway_state_png,
                save_dpi=output_dpi,
            )
            if fig_ps is None:
                pathway_state_png = None
        except Exception as _tps_err:
            print(f"[warn] Therapy-pathway state plot failed: {_tps_err}")
            pathway_state_png = None
    else:
        pathway_state_png = None

    print("[analysis] Running broad-compartment decomposition...")
    decomp_png = None
    composition_png = None
    components_png = None
    candidates_png = None
    candidate_codes = []
    candidate_trace_codes = [
        str(row.get("code") or "").strip()
        for row in analysis.get("candidate_trace", [])
        if row.get("code")
    ]
    expression_reference_code = str(
        analysis.get("expression_reference_cancer_type") or ""
    ).strip()
    scoped_decomposition_codes = [reference_cancer_code]
    for code in (expression_reference_code, cancer_code):
        code = str(code or "").strip()
        if not code or code in scoped_decomposition_codes:
            continue
        if _is_defining_fusion_label(code):
            continue
        if _has_operational_analysis_reference(code):
            scoped_decomposition_codes.append(code)
    for code in (*scoped_decomposition_codes, *candidate_trace_codes[:4]):
        code = str(code or "").strip()
        if code and code not in candidate_codes:
            candidate_codes.append(code)
    decomposition_site_hint = site_hint or (
        inferred_site_context.get("site") if inferred_site_context else None
    )
    decomposition_tumor_context = (
        "met"
        if inferred_site_context and tumor_context == "auto" and not template_overrides
        else tumor_context
    )
    candidate_tsv = (
        "%s-cancer-candidates.tsv" % prefix if prefix else "cancer-candidates.tsv"
    )
    import pandas as pd

    pd.DataFrame(
        [
            {
                "rank": idx + 1,
                "cancer_type": row["code"],
            "signature_score": row["signature_score"],
            "purity_estimate": row["purity_estimate"],
            "lineage_purity": row.get("lineage_purity"),
            "lineage_concordance": row.get("lineage_concordance"),
            "lineage_detection_fraction": row.get("lineage_detection_fraction"),
                "lineage_support_factor": row.get("lineage_support_factor"),
                "family_label": row.get("family_label"),
                "family_score": row.get("family_score"),
                "family_factor": row.get("family_factor"),
                "signature_stability": row.get("signature_stability"),
                "support_score": row["support_score"],
                "support_geomean": row.get("support_geomean"),
                "support_raw_fraction_of_max": row.get("support_raw_fraction_of_max"),
                "support_rank_tier": row.get("support_rank_tier"),
                "support_rank_score": row.get("support_rank_score"),
                "support_fraction_of_top": row["support_fraction_of_top"],
                "centroid_correlation": row.get("centroid_correlation"),
                "range_plausibility": row.get("range_plausibility"),
                "compartment_in_set": row.get("compartment_in_set"),
                "centroid_coarse_lineage": row.get("centroid_coarse_lineage"),
                "centroid_compartment_restricted": row.get(
                    "centroid_compartment_restricted"
                ),
            }
            for idx, row in enumerate(analysis.get("candidate_trace", []))
        ]
    ).to_csv(candidate_tsv, sep="\t", index=False)

    params_json = (
        "%s-analysis-parameters.json" % prefix if prefix else "analysis-parameters.json"
    )
    write_json(
        params_json,
        build_analysis_parameters(
            config=config,
            resolution=resolution,
            template_overrides=template_overrides,
            selected_sample_mode=analysis["sample_mode"],
            quality=quality,
            tumor_purity_parameters=get_tumor_purity_parameters(),
            decomposition_parameters=get_decomposition_parameters(),
        ),
    )

    decomp_results = decompose_sample(
        df_expr,
        cancer_types=candidate_codes or [cancer_code],
        top_k=24,
        sample_mode=analysis["sample_mode"],
        tumor_context=decomposition_tumor_context,
        site_hint=decomposition_site_hint,
        templates=template_overrides or None,
        sample_context=sample_context,
        # #85: hand off the already-ranked candidate rows so
        # decompose_sample reuses analyze_sample's work instead of
        # re-ranking (which internally re-runs estimate_tumor_purity
        # per candidate).
        candidate_rows=analysis.get("candidate_trace"),
    )
    decomp_results = _prioritize_report_compatible_decomposition(
        decomp_results,
        reference_code=reference_cancer_code,
        report_code=cancer_code,
        enabled=True,
        analysis=analysis,
    )
    call_summary = _summarize_sample_call(
        analysis,
        decomp_results,
        sample_mode=analysis["sample_mode"],
    )
    if report_scope_cancer_type and report_scope_cancer_type != reference_cancer_code:
        call_summary = dict(call_summary)
        call_summary["reference_label_options"] = call_summary.get(
            "label_options", []
        )
        call_summary["label_options"] = [report_scope_cancer_type]
        call_summary["label_display"] = _cancer_label(report_scope_cancer_type)
    analysis["call_summary"] = call_summary
    effective_cancer_type = reference_cancer_code
    report_cancer_type = cancer_code
    effective_purity = purity
    if decomp_results:
        analysis["decomposition_results"] = decomp_results
        best_decomp = decomp_results[0]
        run.note_step(
            "decomposition",
            outputs={
                "n_hypotheses": len(decomp_results),
                "best_cancer_type": best_decomp.cancer_type,
                "best_template": best_decomp.template,
                "best_score": best_decomp.score,
            },
            warnings=list(best_decomp.warnings or []),
        )
        # #198: surface the decomposition top template into the analysis
        # dict so the degenerate-subtype resolver (invoked from brief.py)
        # can consult it as a tiebreaker context. ``best_template`` is
        # the top-ranked template name (e.g. ``met_bone``); full ranked
        # list available for future tiebreakers that need the runner-up.
        analysis["decomposition"] = {
            "best_template": best_decomp.template,
            "best_cancer_type": best_decomp.cancer_type,
            "supported_met_site": _supported_decomposition_met_site(analysis),
            "hypotheses": [
                {"template": d.template, "cancer_type": d.cancer_type, "score": d.score}
                for d in decomp_results[:5]
            ],
            # Instrumentation only (nothing consumes this yet): how fragile the adopted
            # decomposition purity is across the plausible template hypotheses. See
            # analyze.flow.decomposition_purity_stability.
            "purity_stability": decomposition_purity_stability(decomp_results, best_decomp),
        }
        # The classifier's top call (``cancer_code``) is the authoritative
        # cancer-type identification across every downstream report; the
        # decomposer only fits *subtraction templates* to separate tumor
        # from TME. When the best-fit template belongs to a different
        # cancer type than the classifier's call that's a decomposition-
        # fit observation, not a re-classification — do NOT overwrite
        # effective_cancer_type here (prior behavior leaked the template
        # label into brief/actionable/targets and recommended BRCA agents
        # on a COAD sample).
        #
        # The decomp's purity is only safe to adopt when its cancer_type
        # matches the classifier AND its template has non-tumor
        # compartments. A template without TME compartments trivially
        # returns fraction=1.0 (everything maps to tumor by construction),
        # which is not a purity measurement — the canonical failure
        # case is a CRC sample whose classifier said 36% (COAD) but
        # whose best-fit decomposition template was BRCA / solid_primary
        # with "No non-tumor components in template"; fraction=100%
        # propagated as the headline purity.
        # Post-decomposition purity reconciliation (adopt decomposition purity ->
        # lineage-panel override -> interval cap), run once for the winner. Mutates
        # analysis["purity"] in place; refresh the local `purity`, which later
        # rendering still reads. See _reconcile_purity_after_decomposition.
        effective_purity = _reconcile_purity_after_decomposition(
            analysis,
            best_decomp,
            reference_cancer_code=reference_cancer_code,
            effective_purity=effective_purity,
            run=run,
        )
        purity = analysis["purity"]
        if call_summary.get("site_indeterminate"):
            print(
                f"[analysis] Retained label differential: {call_summary['label_display']}; "
                f"site/template indeterminate"
            )
        else:
            print(
                f"[analysis] Best decomposition: {best_decomp.cancer_type} / {best_decomp.template}, "
                f"{_decomposition_fraction_label(analysis['sample_mode'])}={best_decomp.purity:.0%}, "
                f"score={best_decomp.score:.3f}"
            )

        # Quality-informed caveats on decomposition
        if quality["degradation"]["level"] in ("moderate", "severe"):
            print(
                "[quality WARNING] RNA degradation detected — decomposition component "
                "fractions and purity estimate may be unreliable. Long-transcript "
                "genes (>7 kb) are systematically underrepresented."
            )
            best_decomp.warnings.append(
                f"RNA degradation detected (pair index="
                f"{quality['degradation']['long_short_ratio']})"
            )
        if quality["culture"]["level"] in ("likely_cell_line", "possible_cell_line"):
            print(
                "[quality WARNING] Sample appears to be a cell line — "
                "decomposition TME components are not meaningful."
            )

        # Prefer standalone decomposition plots over the crowded legacy
        # 4-panel composite so each figure reads cleanly on its own.
        # Keep ``plot_decomposition_summary`` available in the API, but
        # do not emit it by default from ``analyze``.
        # Standalone presentation-ready plots for the best hypothesis
        label_options = call_summary.get("label_options") or []
        if label_options:
            decomp_title_prefix = _cancer_label(label_options[0])
        else:
            decomp_title_prefix = call_summary.get("label_display") or _cancer_label(
                best_decomp.cancer_type
            )
        if call_summary.get("site_indeterminate"):
            decomp_context_suffix = "host context indeterminate"
        elif call_summary.get("reported_site"):
            decomp_context_suffix = call_summary["reported_site"]
        else:
            decomp_context_suffix = _template_site_display(
                best_decomp.template,
                analysis=analysis,
                cancer_code=analysis.get("cancer_type"),
            )
        composition_title = f"Sample composition — {decomp_title_prefix}"
        component_title = f"TME cell-type breakdown — {decomp_title_prefix}"
        if decomp_context_suffix:
            composition_title += f" ({decomp_context_suffix})"
            component_title += f" ({decomp_context_suffix})"
        composition_png = (
            "%s-decomposition-composition.png" % prefix
            if prefix
            else "decomposition-composition.png"
        )
        if plot_ctx.enabled:
            plot_decomposition_composition(
                best_decomp,
                save_to_filename=composition_png,
                save_dpi=output_dpi,
                title=composition_title,
            )
        components_png = (
            "%s-decomposition-components.png" % prefix
            if prefix
            else "decomposition-components.png"
        )
        if plot_ctx.enabled:
            plot_decomposition_component_breakdown(
                best_decomp,
                save_to_filename=components_png,
                save_dpi=output_dpi,
                title=component_title,
            )
        # Sample decomposition candidate bars — one row per (cancer × template)
        # candidate, showing the 3-segment composition (tumor / template-
        # specific / shared immune+stroma). Replaces the "extra=..." text
        # annotation on the composite summary with a structural picture.
        candidates_png = (
            "%s-decomposition-candidates.png" % prefix
            if prefix
            else "decomposition-candidates.png"
        )
        if plot_ctx.enabled:
            plot_decomposition_candidates(
                decomp_results,
                save_to_filename=candidates_png,
                save_dpi=output_dpi,
                labels=[
                    _hypothesis_display_label(
                        row,
                        primary_code=analysis.get("cancer_type"),
                        analysis=analysis,
                    )
                    for row in decomp_results[:6]
                ],
            )

        hypotheses_tsv = (
            "%s-decomposition-hypotheses.tsv" % prefix
            if prefix
            else "decomposition-hypotheses.tsv"
        )
        pd.DataFrame(
            [
                {
                    "rank": idx + 1,
                    "cancer_type": row.cancer_type,
                    "template": row.template,
                    "score": row.score,
                    "purity": row.purity,
                    "reconstruction_error": row.reconstruction_error,
                    "cancer_signature_score": row.cancer_signature_score,
                    "cancer_purity_score": row.cancer_purity_score,
                    "cancer_support_score": row.cancer_support_score,
                    "template_tissue_score": row.template_tissue_score,
                    "template_origin_tissue_score": row.template_origin_tissue_score,
                    "template_site_factor": row.template_site_factor,
                    "template_extra_fraction": row.template_extra_fraction,
                    "n_measured_in_fit": row.n_measured_in_fit,
                    "warnings": "; ".join(row.warnings),
                }
                for idx, row in enumerate(decomp_results)
            ]
        ).to_csv(hypotheses_tsv, sep="\t", index=False)

        if not best_decomp.component_trace.empty:
            best_decomp.component_trace.to_csv(
                "%s-decomposition-components.tsv" % prefix
                if prefix
                else "decomposition-components.tsv",
                sep="\t",
                index=False,
            )
        if not best_decomp.marker_trace.empty:
            best_decomp.marker_trace.to_csv(
                "%s-decomposition-markers.tsv" % prefix
                if prefix
                else "decomposition-markers.tsv",
                sep="\t",
                index=False,
            )
        if not best_decomp.gene_attribution.empty:
            best_decomp.gene_attribution.to_csv(
                "%s-decomposition-gene-attribution.tsv" % prefix
                if prefix
                else "decomposition-gene-attribution.tsv",
                sep="\t",
                index=False,
            )

    signal_png = (
        "%s-cancer-type-signal-matrix.png" % prefix
        if prefix
        else "cancer-type-signal-matrix.png"
    )
    try:
        signal_tsv, signal_md, signal_matrix = write_cancer_type_signal_artifacts(
            prefix,
            analysis,
            sample_id=sample_display_id or None,
            decomp_results=decomp_results,
        )
        analysis["cancer_type_signal_matrix"] = {
            "path": signal_tsv,
            "summary_path": signal_md,
            "rows": int(len(signal_matrix)),
        }
        run.note_step(
            "cancer_type_signal_matrix",
            outputs={
                "path": signal_tsv,
                "summary_path": signal_md,
                "rows": int(len(signal_matrix)),
            },
        )
        if plot_ctx.enabled:
            from .cancer_type_signal_matrix import plot_cancer_type_signal_matrix

            plot_cancer_type_signal_matrix(
                signal_matrix,
                signal_png,
                dpi=output_dpi,
            )
    except Exception as exc:  # noqa: BLE001 - a diagnostic artifact must never abort a finished run
        # This is a warn-and-continue diagnostic sidecar: the whole analysis is
        # already done. A narrower catch (OSError/ValueError/TypeError) let a
        # KeyError/AttributeError from a shape drift abort the entire run just to
        # skip an optional artifact, so catch broadly and keep the report.
        print(f"[warn] Cancer-type signal matrix failed: {exc}")

    # Final purity pass: honest random-effects interval + anti-saturation guard, run
    # BEFORE the ReportView freeze so every headline read (figure, brief, markdown) sees
    # the same finalized number by construction. See ``_finalize_fused_purity``.
    _finalize_fused_purity(analysis)

    # Build the frozen ReportView snapshot now that purity is finalized
    # (decomposition adoption + lineage-panel override + interval cap above) and
    # the decomposition results are attached. Phase 1 instrumentation: renderers
    # are migrated onto this single read surface incrementally, so building it
    # here is behavior-neutral. See
    # docs/report-belief-consistency-and-friendliness-plan.md (Tier 1).
    from .report_view import build_report_view

    try:
        analysis["report_view"] = build_report_view(
            analysis, sample_id=sample_display_id or None
        )
    except Exception:  # noqa: BLE001 - instrumentation must never abort a run
        _LOGGER.warning(
            "build_report_view failed; report_view unavailable", exc_info=True
        )
        analysis["report_view"] = None

    # #97 sample-summary panels, rendered HERE — after purity finalization
    # (decomposition adoption + lineage-panel override + interval cap above) and
    # after the `if decomp_results:` block, so the composition/purity panel reads
    # the SAME adopted purity as purity-methods and the markdown. This ordering is
    # the fix for the 78%-vs-10% headline-purity contradiction; do not move the
    # render back above the purity-finalization block.
    if plot_ctx.enabled:
        plot_sample_summary(
            df_expr,
            cancer_type=reference_cancer_code,
            sample_mode=analysis["sample_mode"],
            save_to_filename=summary_png,
            save_dpi=output_dpi,
            analysis=analysis,
        )
        plot_cancer_type_hypotheses(
            analysis, save_to_filename=hypotheses_png, save_dpi=output_dpi
        )
        plot_background_tissues(
            analysis, save_to_filename=tissues_png, save_dpi=output_dpi
        )
        plot_mhc_expression(analysis, save_to_filename=mhc_png, save_dpi=output_dpi)

    purity_png = "%s-purity.png" % prefix if prefix else "purity.png"
    if plot_ctx.enabled:
        print("[plot] Generating tumor purity detail plot...")
        plot_tumor_purity(
            df_expr,
            cancer_type=effective_cancer_type,
            sample_mode=analysis["sample_mode"],
            save_to_filename=purity_png,
            save_dpi=output_dpi,
            # #86: render the harmonized purity (may be lineage-panel-
            # adopted) so the figure agrees with the rest of the report
            # instead of silently recomputing a signature-only estimate.
            purity_result=analysis["purity"],
        )
        _plt.close("all")

    # #124: side-by-side comparison of every purity estimation method
    # on a single purity axis. The existing plot_tumor_purity panel
    # mixes enrichment scales with purity % on the same row, which
    # hides how much the methods agree / disagree. This dedicated
    # figure renders signature / lineage / ESTIMATE stromal / ESTIMATE
    # immune / ESTIMATE combined / decomposition / adopted overall on
    # one purity axis with CI bars, plus TCGA cohort median reference.
    methods_png = "%s-purity-methods.png" % prefix if prefix else "purity-methods.png"
    best_for_methods = decomp_results[0] if decomp_results else None
    if plot_ctx.enabled:
        print("[plot] Generating purity-method comparison plot...")
        plot_purity_method_comparison(
            analysis["purity"],
            save_to_filename=methods_png,
            save_dpi=output_dpi,
            decomposition_result=best_for_methods,
            report_view=analysis.get("report_view"),
        )
        _plt.close("all")

    # Scatter plots: sample vs pan-cancer reference
    scatter_pdf = "%s-vs-cancer.pdf" % prefix if prefix else "vs-cancer.pdf"
    if plot_ctx.enabled:
        print("[plot] Generating sample vs cancer scatter plots...")
        plot_sample_vs_cancer(
            df_expr,
            # #83: use the resolved (possibly auto-detected and decomp-
            # promoted) cancer type so the scatter agrees with the
            # summary/purity outputs. Previously fell back to the raw CLI
            # argument, which for the default auto-detect path meant the
            # pan-cancer mean instead of the inferred tumor type.
            cancer_type=effective_cancer_type,
            save_to_filename=scatter_pdf,
            save_dpi=output_dpi,
            always_label_genes=forced_labels,
        )
        _plt.close("all")

    # Therapy target tissue expression / safety
    tissue_pdf = "%s-target-tissues.pdf" % prefix if prefix else "target-tissues.pdf"
    curated_target_symbols = set(forced_labels or [])
    if plot_ctx.enabled:
        print("[plot] Generating therapy target tissue expression...")
        try:
            panel_code_for_tissues, panel_subtype_for_tissues = (
                cancer_key_genes_lookup_for_analysis(effective_cancer_type, analysis)
            )
            target_panel_for_tissues = (
                cancer_therapy_targets(
                    panel_code_for_tissues, subtype=panel_subtype_for_tissues
                )
                if panel_subtype_for_tissues
                else cancer_therapy_targets(panel_code_for_tissues)
            )
            target_panel_for_tissues = filter_current_therapy_targets(
                target_panel_for_tissues
            )
            if target_panel_for_tissues is not None and not target_panel_for_tissues.empty:
                curated_target_symbols.update(
                    str(sym).strip()
                    for sym in target_panel_for_tissues.get("symbol", [])
                    if str(sym).strip()
                )
            curated_target_symbols.update(
                cancer_biomarker_genes(
                    panel_code_for_tissues, subtype=panel_subtype_for_tissues
                )
                if panel_subtype_for_tissues
                else cancer_biomarker_genes(panel_code_for_tissues)
            )
        except Exception:
            pass
        plot_therapy_target_tissues(
            df_expr,
            top_k=therapy_target_top_k,
            tpm_threshold=therapy_target_tpm_threshold,
            extra_symbols=curated_target_symbols,
            save_to_filename=tissue_pdf,
            save_dpi=output_dpi,
        )
    deprecated_figures_dir = Path(prefix).parent / "figures" / "deprecated"
    if plot_ctx.enabled and deprecated_figures:
        deprecated_figures_dir.mkdir(parents=True, exist_ok=True)
        deprecated_tissue_png = (
            deprecated_figures_dir / f"{Path(prefix).name}-target-tissues.png"
        )
        print(
            "[plot] Generating deprecated target-tissue PNG fan-out "
            f"under {deprecated_figures_dir}/"
        )
        plot_therapy_target_tissues(
            df_expr,
            top_k=therapy_target_top_k,
            tpm_threshold=therapy_target_tpm_threshold,
            extra_symbols=curated_target_symbols,
            save_to_filename=str(deprecated_tissue_png),
            save_dpi=output_dpi,
        )
        try:
            from .plot_therapy import plot_therapy_target_safety

            deprecated_safety_png = (
                deprecated_figures_dir / f"{Path(prefix).name}-target-safety.png"
            )
            plot_therapy_target_safety(
                df_expr,
                top_k=therapy_target_top_k,
                tpm_threshold=therapy_target_tpm_threshold,
                extra_symbols=curated_target_symbols,
                save_to_filename=str(deprecated_safety_png),
                save_dpi=output_dpi,
            )
        except Exception as exc:
            print(f"[plot] deprecated target-safety failed: {exc}")
    if plot_ctx.enabled:
        _plt.close("all")

    # Cancer-type signature gene grids (``plot_cancer_type_genes`` +
    # ``plot_cancer_type_disjoint_genes``) were removed from the default
    # plot set in 4.40.1 — they duplicated the candidate-ranking table
    # in analysis.md without adding interpretive value. The functions
    # remain in ``pirlygenes.plot_embedding`` for Python-API consumers.

    # Sample-among-reference context: emit a global normal-inclusive MDS plus
    # a ranked nearest-reference distance plot. These are audit/context views:
    # they intentionally bypass the fused evidence graph, so they are kept out
    # of the main figure packet and grouped in figure-audit instead.
    audit_only_pngs = []
    mds_png = "%s-reference-mds.png" % prefix
    neighborhood_png = "%s-reference-neighborhood.png" % prefix
    embedding_pngs = [mds_png, neighborhood_png]
    if plot_ctx.enabled:
        print("[plot] Generating pan-reference MDS embedding...")
        plot_cancer_type_mds(
            df_expr,
            method="panref",
            include_normals=True,
            include_subtypes=True,
            label_nearest_cancers=5,
            label_nearest_normals=5,
            label_all=False,
            save_to_filename=mds_png,
            save_dpi=output_dpi,
        )
        print("[plot] Generating nearest-reference distance ranking...")
        plot_cancer_type_neighborhood(
            df_expr,
            method="panref",
            include_normals=True,
            include_subtypes=True,
            label_nearest_cancers=5,
            label_nearest_normals=5,
            label_all=False,
            focus_nearest_cancers=25,
            focus_nearest_normals=10,
            save_to_filename=neighborhood_png,
            save_dpi=output_dpi,
        )
        _plt.close("all")

    # Canonical target screen plus CTA/subtype drill-down plots.
    p_est = purity.get("overall_estimate")
    targets_deep_png = None
    cta_deep_png = None
    subtype_png = None
    if plot_ctx.enabled:
        from .plot_target_deep_dive import (
            plot_actionable_targets,
            plot_cta_deep_dive,
        )
        from .plot_subtype_signature import plot_subtype_signature

        try:
            targets_deep_png = "%s-actionable-targets.png" % prefix
            fig = plot_actionable_targets(
                df_expr,
                cancer_type=effective_cancer_type,
                purity_estimate=p_est,
                save_to_filename=targets_deep_png,
                save_dpi=output_dpi,
            )
            if fig is not None:
                print(f"[plot] Saved actionable targets to {targets_deep_png}")
            else:
                targets_deep_png = None
            _plt.close("all")
        except Exception as exc:
            print(f"[plot] actionable targets failed: {exc}")
            targets_deep_png = None

        try:
            cta_deep_png = "%s-cta-deep-dive.png" % prefix
            fig = plot_cta_deep_dive(
                df_expr,
                cancer_type=effective_cancer_type,
                purity_estimate=p_est,
                save_to_filename=cta_deep_png,
                save_dpi=output_dpi,
            )
            if fig is not None:
                print(f"[plot] Saved CTA deep dive to {cta_deep_png}")
            else:
                cta_deep_png = None
            _plt.close("all")
        except Exception as exc:
            print(f"[plot] CTA deep dive failed: {exc}")
            cta_deep_png = None

    # The pre-#108 ``plot_tumor_attribution`` (reference-based 2-color
    # tumor-vs-TME view) was removed from the default plot set in
    # v4.46.0 — ``plot_target_attribution`` (per-compartment stacked
    # bars keyed on the #108 attribution dict) is strictly more
    # informative and already emits per-category ``target-attribution-
    # *.png``. The older function remains in
    # ``pirlygenes.plot_target_deep_dive`` for Python-API consumers.
    attrib_targets_png = None
    attrib_cta_png = None

    if plot_ctx.enabled:
        try:
            subtype_png = "%s-subtype-signature.png" % prefix
            fig = plot_subtype_signature(
                df_expr,
                cancer_type=effective_cancer_type,
                save_to_filename=subtype_png,
                save_dpi=output_dpi,
            )
            if fig is None:
                subtype_png = None
            else:
                print(f"[plot] Saved subtype signature to {subtype_png}")
            _plt.close("all")
        except Exception as exc:
            print(f"[plot] subtype signature failed: {exc}")
            subtype_png = None

    _plt.close("all")

    # Generate text reports
    print("[report] Generating text reports...")
    _embedding_meta = get_embedding_feature_metadata(method="panref")
    _generate_text_reports(
        analysis,
        _embedding_meta,
        prefix,
        decomp_results=decomp_results,
        input_path=input_path,
        df_expr=df_expr,
    )

    # Cancer-type-specific gene set plot (only when --cancer-type specified
    # and backed by the pan-cancer expression reference).
    ct_png = None
    if plot_ctx.enabled and analysis_cancer_type:
        from .plot import resolve_cancer_type

        code = resolve_cancer_type(analysis_cancer_type)
        ct_gene_sets = cancer_type_gene_sets(analysis_cancer_type)
        if ct_gene_sets:
            ct_png = (
                "%s-%s-genes.png" % (prefix, code.lower())
                if prefix
                else "%s-genes.png" % code.lower()
            )
            plot_gene_expression(
                df_expr,
                gene_sets=ct_gene_sets,
                save_to_filename=ct_png,
                save_dpi=output_dpi,
                plot_height=plot_height,
                plot_aspect=plot_aspect,
                always_label_genes=forced_labels,
                source_file=input_path,
            )

    # Purity-adjusted tumor expression analysis (9-point ranges, one PNG per category)
    print("[analysis] Generating tumor expression range analysis...")
    purity_dict = effective_purity
    adj_pngs = []
    ranges_df = None
    expression_reference_cancer_code = (
        cancer_type_context.code_for("expression") or effective_cancer_type
    )
    _range_plot_categories = [
        ("CTA", "ctas"),
        ("surface", "surface"),
    ]
    _attribution_categories = [
        ("therapy_target", "targets"),
        ("CTA", "ctas"),
        ("surface", "surface"),
    ]
    try:
        ranges_df = estimate_tumor_expression_ranges(
            df_expr,
            cancer_type=effective_cancer_type,
            purity_result=purity_dict,
            decomposition_results=decomp_results,
            met_site=_effective_met_site_for_background(analysis),
            expression_reference_type=expression_reference_cancer_code,
        )
        ranges_tsv = (
            "%s-tumor-expression-ranges.tsv" % prefix
            if prefix
            else "tumor-expression-ranges.tsv"
        )
        ranges_df.to_csv(ranges_tsv, sep="\t", index=False)
        run.note_step(
            "tumor_expression_ranges",
            outputs={
                "rows": int(len(ranges_df)),
                "path": ranges_tsv,
            },
        )
        alteration_effect_summary = _store_alteration_effect_reasoning(
            analysis,
            fusion_records=fusion_records,
            alteration_records=alteration_records,
            sample_tpm_by_symbol=sample_tpm_by_symbol,
            ranges_df=ranges_df,
            cancer_code=report_cancer_type,
        )
        run.note_step(
            "alteration_effects",
            outputs=alteration_effect_summary,
        )
        if plot_ctx.enabled:
            for cat_key, cat_slug in _range_plot_categories:
                cat_png = (
                    "%s-purity-%s.png" % (prefix, cat_slug)
                    if prefix
                    else "purity-%s.png" % cat_slug
                )
                plot_tumor_expression_ranges(
                    ranges_df,
                    purity_result=purity_dict,
                    cancer_type=effective_cancer_type,
                    top_n=15,
                    categories=[cat_key],
                    save_to_filename=cat_png,
                    save_dpi=output_dpi,
                )
                adj_pngs.append(cat_png)
                _plt.close("all")

        # Per-target compositional attribution (#108). One PNG per
        # category showing the per-gene tumor + compartment
        # breakdown. Emitted only when decomposition produced an
        # attribution; the function returns None otherwise and no file
        # is written, which the CLI respects by not appending to
        # adj_pngs.
        if (
            plot_ctx.enabled
            and "attribution" in ranges_df.columns
            and ranges_df["attribution"]
            .apply(lambda v: isinstance(v, dict) and len(v) > 0)
            .any()
        ):
            for cat_key, cat_slug in _attribution_categories:
                attr_png = (
                    "%s-target-attribution-%s.png" % (prefix, cat_slug)
                    if prefix
                    else "target-attribution-%s.png" % cat_slug
                )
                fig = plot_target_attribution(
                    ranges_df,
                    cancer_type=effective_cancer_type,
                    category=cat_key,
                    top_n=15,
                    save_to_filename=attr_png,
                    save_dpi=output_dpi,
                    sample_tpm_by_symbol=sample_tpm_by_symbol,
                )
                if fig is not None:
                    audit_only_pngs.append(attr_png)
                _plt.close("all")

        # Per-gene subtype-reference correction audit (#56 / #58).
        # This is mainly provenance/debug material, so keep it out of
        # the main all-figures packet and reserve it for figure-audit.
        if (
            plot_ctx.enabled
            and "subtype_refined" in ranges_df.columns
            and ranges_df["subtype_refined"].astype(bool).any()
        ):
            from .plot_tumor_expr import plot_subtype_attribution

            for cat_key, cat_slug in _attribution_categories:
                sub_png = (
                    "%s-subtype-attribution-%s.png" % (prefix, cat_slug)
                    if prefix
                    else "subtype-attribution-%s.png" % cat_slug
                )
                fig = plot_subtype_attribution(
                    ranges_df,
                    category=cat_key,
                    top_n=15,
                    save_to_filename=sub_png,
                    save_dpi=output_dpi,
                )
                if fig is not None:
                    audit_only_pngs.append(sub_png)
                _plt.close("all")

        # Per-gene matched-normal attribution (issue #55). One PNG per
        # category, only emitted when matched-normal subtraction was
        # active. Separate plots rather than a composite figure per the
        # project's crowding preference.
        if (
            plot_ctx.enabled
            and "matched_normal_tpm" in ranges_df.columns
            and (ranges_df["matched_normal_tpm"].astype(float) > 0).any()
        ):
            for cat_key, cat_slug in _attribution_categories:
                mn_png = (
                    "%s-matched-normal-%s.png" % (prefix, cat_slug)
                    if prefix
                    else "matched-normal-%s.png" % cat_slug
                )
                fig = plot_matched_normal_attribution(
                    ranges_df,
                    cancer_type=effective_cancer_type,
                    category=cat_key,
                    top_n=15,
                    save_to_filename=mn_png,
                    save_dpi=output_dpi,
                    sample_tpm_by_symbol=sample_tpm_by_symbol,
                )
                if fig is not None:
                    audit_only_pngs.append(mn_png)
                _plt.close("all")

        target_report_md = _build_target_report(
            ranges_df,
            analysis,
            cancer_type=report_cancer_type,
            purity_result=purity_dict,
            decomp_results=decomp_results,
        )

        priority_targets_png = None
        priority_target_context_png = None
        try:
            from pirlygenes.gene_sets_cancer import cancer_key_genes_cancer_types

            disease_state_for_priority = compose_disease_state_narrative(analysis)
            panel_code, panel_subtype = cancer_key_genes_lookup_for_analysis(
                report_cancer_type,
                analysis,
                ranges_df=ranges_df,
            )
            target_panel = None
            actionable_genes = None
            if panel_code in cancer_key_genes_cancer_types():
                target_panel = (
                    cancer_therapy_targets(panel_code, subtype=panel_subtype)
                    if panel_subtype
                    else cancer_therapy_targets(panel_code)
                )
                target_panel = filter_current_therapy_targets(target_panel)
                actionable_genes = _select_actionable_plot_genes(
                    ranges_df,
                    panel_code,
                    target_panel=target_panel,
                )
                if plot_ctx.enabled and targets_deep_png:
                    try:
                        fig = plot_actionable_targets(
                            df_expr,
                            cancer_type=effective_cancer_type,
                            purity_estimate=p_est,
                            custom_genes=actionable_genes,
                            ranges_df=ranges_df,
                            save_to_filename=targets_deep_png,
                            save_dpi=output_dpi,
                            title=f"Actionable expression screen — {panel_code}",
                        )
                        if fig is not None:
                            print(
                                f"[plot] Refreshed actionable targets to {targets_deep_png}"
                            )
                        _plt.close("all")
                    except Exception as action_plot_err:
                        print(
                            f"[plot] actionable targets refresh failed: {action_plot_err}"
                        )
            if plot_ctx.enabled:
                priority_targets_png = (
                    "%s-priority-targets.png" % prefix if prefix else "priority-targets.png"
                )
                fig = plot_priority_targets(
                    ranges_df,
                    cancer_type=panel_code
                    if target_panel is not None
                    else effective_cancer_type,
                    target_panel=(
                        target_panel.reset_index(drop=True)
                        if target_panel is not None
                        else None
                    ),
                    df_gene_expr=df_expr,
                    target_symbols=actionable_genes,
                    top_n=len(actionable_genes) if actionable_genes else 12,
                    analysis=analysis,
                    disease_state=disease_state_for_priority,
                    save_to_filename=priority_targets_png,
                    save_dpi=output_dpi,
                )
                if fig is not None:
                    adj_pngs.append(priority_targets_png)
                    print(f"[plot] Saved priority targets to {priority_targets_png}")
                else:
                    priority_targets_png = None
                _plt.close("all")

                priority_target_context_png = (
                    "%s-priority-target-context.png" % prefix
                    if prefix
                    else "priority-target-context.png"
                )
                fig = plot_priority_target_context(
                    ranges_df,
                    cancer_type=panel_code
                    if target_panel is not None
                    else effective_cancer_type,
                    target_panel=(
                        target_panel.reset_index(drop=True)
                        if target_panel is not None
                        else None
                    ),
                    df_gene_expr=df_expr,
                    target_symbols=actionable_genes,
                    top_n=len(actionable_genes) if actionable_genes else 12,
                    analysis=analysis,
                    disease_state=disease_state_for_priority,
                    save_to_filename=priority_target_context_png,
                    save_dpi=output_dpi,
                )
                if fig is not None:
                    # priority-target-context.png is audit-only (§2.5): its tumor-
                    # source/safety-band cue is now folded into priority-targets.png
                    # (the single reader target figure). Route to audit, not adj_pngs
                    # (which flows to the reader packet).
                    audit_only_pngs.append(priority_target_context_png)
                    print(
                        f"[plot] Saved priority target context to {priority_target_context_png}"
                    )
                else:
                    priority_target_context_png = None
                _plt.close("all")

            if plot_ctx.enabled and deprecated_figures:
                deprecated_figures_dir.mkdir(parents=True, exist_ok=True)
                deprecated_purity_targets_png = (
                    deprecated_figures_dir / f"{Path(prefix).name}-purity-targets.png"
                )
                plot_tumor_expression_ranges(
                    ranges_df,
                    purity_result=purity_dict,
                    cancer_type=effective_cancer_type,
                    top_n=15,
                    categories=["therapy_target"],
                    save_to_filename=str(deprecated_purity_targets_png),
                    save_dpi=output_dpi,
                )
                _plt.close("all")
                if target_panel is not None:
                    try:
                        from .plot_target_deep_dive import plot_curated_target_evidence

                        deprecated_curated_png = (
                            deprecated_figures_dir
                            / f"{Path(prefix).name}-curated-target-evidence.png"
                        )
                        fig = plot_curated_target_evidence(
                            ranges_df,
                            target_panel.reset_index(drop=True),
                            cancer_type=panel_code,
                            df_gene_expr=df_expr,
                            save_to_filename=str(deprecated_curated_png),
                            save_dpi=output_dpi,
                        )
                        if fig is not None:
                            print(
                                "[plot] Saved deprecated curated target evidence "
                                f"to {deprecated_curated_png}"
                            )
                        _plt.close("all")
                    except Exception as exc:
                        print(f"[plot] deprecated curated target evidence failed: {exc}")

        except Exception as curated_err:
            print(f"[plot] priority target plots failed: {curated_err}")
            priority_targets_png = None
            priority_target_context_png = None

        # The public markdown surface is intentionally compact:
        # summary.md for the distilled read, analysis.md for the full
        # interpreted report, and evidence.md for the stepwise/rawer
        # support tables.
        try:
            from .brief import build_summary

            disease_state_for_summary = compose_disease_state_narrative(analysis)
            sample_id = sample_display_id or None
            _generate_text_reports(
                analysis,
                _embedding_meta,
                prefix,
                decomp_results=decomp_results,
                input_path=input_path,
                ranges_df=ranges_df,
                sample_id=sample_id,
                df_expr=df_expr,
            )
            summary_md = build_summary(
                analysis,
                ranges_df,
                cancer_code=report_cancer_type,
                disease_state=disease_state_for_summary,
                sample_id=sample_id,
            )
            evidence_md = _build_evidence_report(
                analysis,
                ranges_df,
                decomp_results,
                cancer_code=report_cancer_type,
                sample_id=sample_id,
                target_report_md=target_report_md,
            )
            summary_path = "%s-summary.md" % prefix if prefix else "summary.md"
            evidence_path = "%s-evidence.md" % prefix if prefix else "evidence.md"
            with open(summary_path, "w") as f:
                f.write(summary_md)
            with open(evidence_path, "w") as f:
                f.write(evidence_md)
            print(f"[report] Saved {summary_path}")
            print(f"[report] Saved {evidence_path}")
            if plot_ctx.enabled:
                from .provenance import plot_provenance_funnel

                prov_png = "%s-provenance.png" % prefix if prefix else "provenance.png"
                fig_out = plot_provenance_funnel(
                    analysis,
                    ranges_df,
                    decomp_results,
                    save_to_filename=prov_png,
                    save_dpi=output_dpi,
                )
                if fig_out:
                    print(f"[plot] Saved {prov_png}")
        except Exception as brief_err:
            print(f"[warn] Summary / evidence rendering failed: {brief_err}")
    except Exception as e:
        print(f"[warn] Purity-adjusted analysis failed: {e}")
        import traceback

        traceback.print_exc()

    if not plot_ctx.enabled:
        print("[output] --no-figures: skipped figure PDF collection")
        return

    # Collect all figures into one PDF (native resolution)
    from PIL import Image, ImageDraw, ImageFont

    all_pdf = "%s-all-figures.pdf" % prefix if prefix else "all-figures.pdf"
    print("[output] Collecting figures into PDF...")
    # Report-flow order (user direction 2026-04-14): QC first
    # (context_png + degradation_png), then the headline cancer call
    # (summary_png), then deeper detail (decomposition / purity /
    # strip plots / embeddings). Plots missing from this list didn't
    # make it into all-figures.pdf and got left out of the moved-to-
    # figures/ step.
    png_files = [
        context_png,
        concentration_top_png,
        concentration_curve_png,
        reference_mtdna_qc_png,
        burden_qc_png,
        degradation_png,
        # sample-summary.png (the legacy 4-panel composite) is audit-only (§2.5):
        # its four panels are already emitted as standalone reader figures below
        # (hypotheses_png, purity_png, tissues_png, mhc_png), so in the reader packet
        # the composite only duplicates them. Routed to audit_only_pngs (below).
        decomp_png,
        # Standalone decomposition PNGs — composition / component breakdown
        # / candidate bars. Historically missed the move-to-figures/ step
        # because they weren't listed here.
        composition_png,
        components_png,
        candidates_png,
        purity_png,
        # Standalone analysis panels (cancer hypotheses, background
        # tissues, MHC) — new outputs added while splitting the legacy
        # 4-panel composite. Without being listed here they never made
        # it into the PDF or the figures/ move step.
        hypotheses_png,
        signal_png,
        tissues_png,
        mhc_png,
        # Purity-method comparison (#124) and sample-provenance (#106)
        # — added later and initially missed the figures/ move list.
        methods_png,
        # Provenance PNG may not exist when decomposition / range
        # computation was skipped; resolve by path rather than a
        # possibly-undefined variable.
        "%s-provenance.png" % prefix if prefix else "provenance.png",
        # #136: therapy-pathway-state dumbbell figure.
        pathway_state_png,
        "%s-treatments.png" % prefix if prefix else "treatments.png",
    ]
    audit_only_pngs.extend(embedding_pngs)
    # The 4-panel composite (see note in png_files) ships in the audit packet only;
    # the reader packet keeps the four standalone panels it duplicates.
    audit_only_pngs.append(summary_png)
    if ct_png:
        png_files.append(ct_png)
    # actionable-targets.png is audit-only (§2.5): it is a near-duplicate target
    # dumbbell of priority-targets.png (the single reader target figure, which now
    # carries the tumor-source/safety cue), so route it to audit while the reader
    # keeps one target figure.
    if targets_deep_png and Path(targets_deep_png).exists():
        audit_only_pngs.append(targets_deep_png)
    # Deep-dive plots
    for _ddp in [
        cta_deep_png,
        attrib_targets_png,
        attrib_cta_png,
        subtype_png,
    ]:
        if _ddp and Path(_ddp).exists():
            png_files.append(_ddp)

    # Per-category curated-panel scatter PNGs (DNA_repair, Oncogenes, CTAs, …) are
    # audit-only (§2.5): ~10 near-overlapping panel scatters bury the ~decision
    # figures in the reader packet (all-figures.pdf). Route them to audit_only_pngs
    # so they still move into figures/ and appear in figure-audit.pdf, but no longer
    # pad the reader packet.
    scatter_dir = Path(scatter_pdf).parent / Path(scatter_pdf).stem
    if scatter_dir.is_dir():
        audit_only_pngs.extend(sorted(str(p) for p in scatter_dir.glob("*.png")))
    # Purity-adjusted plots go last (different RNA measure)
    for adj_p in adj_pngs:
        if Path(adj_p).exists():
            png_files.append(adj_p)

    def _pdf_font(size: int, *, bold: bool = False):
        """Return a scalable PDF text font; fall back gracefully."""
        candidates = (
            (
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
                if bold
                else "/System/Library/Fonts/Supplemental/Arial.ttf"
            ),
            (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                if bold
                else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ),
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
            "Arial Bold.ttf" if bold else "Arial.ttf",
        )
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size=size)
            except Exception:
                continue
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    def _with_filename_caption(img, filename):
        """Add filename strips around the figure.

        The top-left label is intentionally visible in the figure audit
        PDF, where readers are deciding which source PNGs are good or
        bad. The bottom-right caption is retained for the all-figures
        packet. Both sit outside the plot area so labels never obscure
        the figure.
        """
        caption_font = _pdf_font(26)
        header_font = _pdf_font(32, bold=True)
        header_h = 58
        caption_h = 42
        new_w, new_h = img.width, img.height + header_h + caption_h
        canvas = Image.new("RGB", (new_w, new_h), color="white")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, new_w, header_h), fill="#f3f4f6")
        text = filename
        draw.text((18, 12), text, fill="#333333", font=header_font)
        canvas.paste(img, (0, header_h))
        # Bottom-right, light gray.
        try:
            bbox = draw.textbbox((0, 0), text, font=caption_font)
            tw = bbox[2] - bbox[0]
        except AttributeError:  # pragma: no cover — very old PIL
            tw = len(text) * 14
        draw.text(
            (max(12, new_w - tw - 18), header_h + img.height + 6),
            text,
            fill="#888888",
            font=caption_font,
        )
        return canvas

    images = []
    if plot_ctx.enabled:
        for png_path in png_files:
            if not png_path:
                continue
            p = Path(png_path)
            if p.exists():
                img = Image.open(p).convert("RGB")
                images.append(_with_filename_caption(img, p.name))

    if images:
        images[0].save(
            all_pdf, save_all=True, append_images=images[1:], resolution=output_dpi
        )
        print(f"Saved {all_pdf} ({len(images)} pages)")
    else:
        print("No images to collect into PDF")

    # Move PNGs and per-figure PDFs into figures/ subdir,
    # keeping all-figures.pdf and markdown reports in place.
    fig_out_dir = Path(prefix).parent
    figures_dir = fig_out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    moved = 0
    move_png_files = png_files + [p for p in audit_only_pngs if p]

    for png_path in move_png_files:
        if not png_path:
            continue
        p = Path(png_path)
        if p.exists() and p.suffix == ".png":
            p.rename(figures_dir / p.name)
            moved += 1
    # Move scatter dir contents and per-plot PDFs
    if scatter_dir.is_dir():
        for p in scatter_dir.glob("*.png"):
            p.rename(figures_dir / p.name)
            moved += 1
        # Remove empty scatter dir
        try:
            scatter_dir.rmdir()
        except OSError:
            pass
    for extra in [scatter_pdf, tissue_pdf]:
        p = Path(extra) if isinstance(extra, str) else extra
        if p.exists():
            p.rename(figures_dir / p.name)
            moved += 1
    if moved:
        print(f"[output] Moved {moved} figures to {figures_dir}/")

    figure_audit_pdf = "%s-figure-audit.pdf" % prefix if prefix else "figure-audit.pdf"

    def _make_audit_text_page(title, lines):
        from textwrap import wrap

        width, height = 1800, 2400
        img = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(img)
        title_font = _pdf_font(62, bold=True)
        body_font = _pdf_font(36)
        y = 80
        draw.text((90, y), title, fill="black", font=title_font)
        y += 105
        for line in lines:
            wrapped = wrap(str(line), width=68) or [""]
            for piece in wrapped:
                draw.text((90, y), piece, fill="#333333", font=body_font)
                y += 48
            y += 22
        return img

    def _existing_figure_paths(*suffixes):
        out = []
        suffixes = tuple(suffixes)
        for path in sorted(figures_dir.iterdir()):
            if path.is_file() and any(path.name.endswith(suffix) for suffix in suffixes):
                out.append(path)
        return out

    def _artifact_page(path):
        path = Path(path)
        if path.suffix.lower() == ".png":
            img = Image.open(path).convert("RGB")
            return _with_filename_caption(img, path.name)
        return _make_audit_text_page(
            f"Figure Artifact: {path.name}",
            [
                "This figure is emitted as a PDF artifact rather than a single PNG page.",
                f"Path: {path}",
                "If a sibling PNG page series exists, those pages appear elsewhere in this audit packet.",
            ],
        )

    audit_sections = [
        (
            "Figure Themes",
            [
                {
                    "title": "QC / Provenance",
                    "note": "Technical context for whether the expression matrix and sample handling look usable.",
                    "files": _existing_figure_paths(
                        "sample-context.png",
                        "expression-top-features-qc.png",
                        "expression-concentration-curve-qc.png",
                        "qc-reference-mtdna.png",
                        "degradation-index.png",
                        "sample-summary.png",
                        "provenance.png",
                    ),
                },
                {
                    "title": "Cancer-Type / Reference Context",
                    "note": "Decision-aligned cancer-call plots first; raw reference-space maps are context-only audit views.",
                    "files": _existing_figure_paths(
                        "cancer-hypotheses.png",
                        "cancer-type-signal-matrix.png",
                        "reference-mds.png",
                        "reference-neighborhood.png",
                        "background-tissues.png",
                        "subtype-signature.png",
                        "vs-cancer.pdf",
                    ),
                },
                {
                    "title": "Purity / Decomposition",
                    "note": "Tumor fraction, non-tumor components, and how purity estimates affect expression interpretation.",
                    "files": _existing_figure_paths(
                        "decomposition.png",
                        "composition.png",
                        "component-breakdown.png",
                        "candidate-comparison.png",
                        "purity.png",
                        "purity-methods.png",
                    ),
                },
                {
                    "title": "Active Pathways / Signatures",
                    "note": "Pathway and state plots, including the MAPK/ERK activity score and other response/signature axes.",
                    "files": _existing_figure_paths(
                        "therapy-pathway-state.png",
                        "subtype-signature.png",
                        "MHC.png",
                        "TLR.png",
                        "DNA_repair.png",
                    ),
                },
                {
                    "title": "Targets / Actionability",
                    "note": "Canonical target figures: broad actionable screen, ranked priority shortlist, and the matching evidence context.",
                    "files": _existing_figure_paths(
                        "treatments.png",
                        "actionable-targets.png",
                        "priority-targets.png",
                        "priority-target-context.png",
                        "target-tissues.pdf",
                    ),
                },
            ],
        ),
        (
            "Potentially Redundant Groups",
            [
                {
                    "title": "Sample Snapshot Cluster",
                    "note": "These all answer variants of 'what kind of sample is this and is it technically trustworthy?'",
                    "files": _existing_figure_paths(
                        "sample-summary.png",
                        "sample-context.png",
                        "degradation-index.png",
                        "background-tissues.png",
                    ),
                },
                {
                    "title": "Cancer-Call Cluster",
                    "note": "The signal matrix and hypotheses plot are decision-aligned. MDS/neighborhood plots are raw reference context and should not be read as selectors.",
                    "files": _existing_figure_paths(
                        "cancer-hypotheses.png",
                        "cancer-type-signal-matrix.png",
                        "reference-mds.png",
                        "reference-neighborhood.png",
                        "subtype-signature.png",
                        "vs-cancer.pdf",
                    ),
                },
                {
                    "title": "Purity + Expression Range Cluster",
                    "note": "These explain purity and then apply that estimate to non-target expression categories; targets use the canonical actionable-targets figure.",
                    "files": _existing_figure_paths(
                        "purity.png",
                        "purity-methods.png",
                        "purity-ctas.png",
                        "purity-surface.png",
                    ),
                },
                {
                    "title": "Attribution / Matched-Normal Cluster",
                    "note": "Provenance/audit views for why tumor-source calls differ from observed TPM; useful for debugging, not primary decision figures.",
                    "files": _existing_figure_paths(
                        "target-attribution-targets.png",
                        "target-attribution-surface.png",
                        "matched-normal-targets.png",
                        "matched-normal-surface.png",
                        "subtype-attribution-targets.png",
                        "subtype-attribution-surface.png",
                        "priority-targets.png",
                        "priority-target-context.png",
                    ),
                },
            ],
        ),
        (
            "Low-Value Figures",
            [
                {
                    "title": "Low-Value in the Default Packet",
                    "note": "These add the least unique decision support relative to the rest of the packet, or are mainly provenance/debug/context views.",
                    "files": _existing_figure_paths(
                        "reference-mds.png",
                        "reference-neighborhood.png",
                        "background-tissues.png",
                        "subtype-attribution-targets.png",
                        "subtype-attribution-ctas.png",
                        "subtype-attribution-surface.png",
                        "TLR.png",
                        "DNA_repair.png",
                    ),
                },
            ],
        ),
        (
            "Unique Figures I Like",
            [
                {
                    "title": "Distinctive / Keep",
                    "note": "These each answer a question that the rest of the packet does not cover cleanly.",
                    "files": _existing_figure_paths(
                        "sample-summary.png",
                        "cancer-hypotheses.png",
                        "cancer-type-signal-matrix.png",
                        "purity-methods.png",
                        "actionable-targets.png",
                        "priority-targets.png",
                        "priority-target-context.png",
                        "provenance.png",
                        "therapy-pathway-state.png",
                    ),
                },
            ],
        ),
    ]
    audit_seen = set()

    audit_images = (
        [
            _make_audit_text_page(
                "Figure Audit",
                [
                    "This PDF groups emitted figures by report theme, decision support, and audit/context role.",
                    "It also groups the same artifacts by report theme so pathway/state plots are easier to find.",
                    "PNG pages are reproduced directly after each group cover page; PDF-only figures are listed on the cover page but not rasterized here.",
                    f"Source directory: {figures_dir}",
                ],
            )
        ]
        if plot_ctx.enabled
        else []
    )
    for section_title, groups in audit_sections if plot_ctx.enabled else []:
        audit_images.append(
            _make_audit_text_page(
                section_title,
                [
                    "The following pages are grouped by how they function in the report packet.",
                ],
            )
        )
        for group in groups:
            files = group["files"]
            file_labels = (
                ", ".join(path.name for path in files)
                if files
                else "No matching figures emitted for this run."
            )
            audit_images.append(
                _make_audit_text_page(
                    group["title"],
                    [
                        group["note"],
                        f"Included: {file_labels}",
                    ],
                )
            )
            for path in files:
                audit_seen.add(path.name)
                audit_images.append(_artifact_page(path))

    remaining_files = (
        [
            path
            for path in sorted(figures_dir.iterdir())
            if path.is_file() and path.name not in audit_seen
        ]
        if plot_ctx.enabled
        else []
    )
    if remaining_files:
        audit_images.append(
            _make_audit_text_page(
                "Coverage Appendix",
                [
                    "Every emitted figure is included at least once in this packet.",
                    "The following pages cover artifacts that did not fit one of the opinionated groups above.",
                ],
            )
        )
        audit_images.append(
            _make_audit_text_page(
                "Other Emitted Figures",
                [
                    "Included: " + ", ".join(path.name for path in remaining_files),
                ],
            )
        )
        for path in remaining_files:
            audit_images.append(_artifact_page(path))

    if audit_images:
        audit_images[0].save(
            figure_audit_pdf,
            save_all=True,
            append_images=audit_images[1:],
            resolution=output_dpi,
        )
        print(f"Saved {figure_audit_pdf} ({len(audit_images)} pages)")

    # Write README explaining output files
    readme_path = Path(prefix).parent / "README.md"
    cancer_code = analysis["cancer_type"]
    cancer_name = analysis["cancer_name"]
    readme = f"""# PIRLy Genes Analysis Output

Sample analyzed as **{cancer_code}** ({cancer_name}).

Raw QC figures use the original expression table. Downstream biology uses
technical-RNA-normalized TPM by default: mitochondrial transcripts, NUMT-like
mitochondrial pseudogenes, rRNA-like features, and rRNA-pseudogene rows are
zeroed and the remaining TPM is renormalized. Downstream reference comparisons
use the same normalized analysis view; raw sample/reference values remain
available for QC and provenance.

## Reports

| File | Description |
|------|-------------|
| `*-summary.md` | One-page distilled read (≤ 40 lines) — cancer call, purity, top therapies, caveats |
| `*-analysis.md` | Main interpreted report — disease-state, tissue-composition evidence, candidate trace, purity components, decomposition, and therapy landscape |
| `*-evidence.md` | Stepwise/raw appendix — attribution chain plus full biomarker/target evidence tables |
| `*-analysis-parameters.json` | Free model parameters plus selected sample mode and embedding methods |
| `*-all-figures.pdf` | Main report figures combined into a single PDF; context-only plots are kept in the audit packet |
| `*-figure-audit.pdf` | Figure packet grouped into decision-support, context-only, redundant, and distinctive sections |
| `*-cancer-candidates.tsv` | Candidate cancer-type support trace |
| `*-decomposition-hypotheses.tsv` | Ranked decomposition hypotheses |
| `*-decomposition-components.tsv` | Component-level fit for best decomposition |
| `*-decomposition-markers.tsv` | Marker-gene evidence for best decomposition |
| `*-decomposition-gene-attribution.tsv` | Per-gene TME/tumor attribution for best decomposition |
| `*-tumor-expression-ranges.tsv` | Purity-adjusted tumor-expression ranges with broad-reference context |

## Figures (in `figures/`)

Prefer the standalone decomposition figures for review and sharing. They replace the crowded legacy composite by splitting composition, component breakdown, and candidate comparison into separate PNGs.

| Figure | Description |
|--------|-------------|
| `*-sample-summary.png` | Quick overview: cancer type, purity, background signatures |
| `*-decomposition-composition.png` | Standalone composition bar (tumor + TME) for the best hypothesis |
| `*-decomposition-components.png` | Standalone TME cell-type breakdown for the best hypothesis |
| `*-decomposition-candidates.png` | Standalone per-candidate composition bars (tumor / template-specific / shared host) across top decomposition candidates |
| `*-cancer-type-signal-matrix.png` | Decision-aligned cancer-type evidence trace; shows the final call, learned votes, MMR context where relevant, ranker/reference/decomposition signals, blockers, and confidence |
| `*-purity.png` | Tumor purity estimation detail |
| `*-treatments.png` | Therapy target expression by modality |
| `*-actionable-targets.png` | Canonical actionable-target screen: observed expression, tumor-source estimate, normal-tissue context, and readiness caveats |
| `*-priority-targets.png` | Actionable target priority ranking, split by approval/readiness tier |
| `*-priority-target-context.png` | Matching actionable-target evidence page: tumor-expression range plus tumor-source and healthy-tissue context |
| `*-target-tissues.pdf` | Detailed per-gene tissue-expression appendix for reviewed therapy targets |
| `*-purity-ctas.png` | Tumor-expression ranges for CTAs |
| `*-purity-surface.png` | Tumor-expression ranges for surface proteins |
| `*-reference-mds.png` | Audit-only raw reference map: sample among TCGA cancer medians, subtype references, and normal tissues; does not represent fused evidence selection |
| `*-reference-neighborhood.png` | Audit-only nearest cancer/subtype/normal reference distance ranking; preserves input feature distance but does not represent fused evidence selection |

Optional deprecated comparison figures are only emitted with
`--deprecated-figures` and are written under `figures/deprecated/`. They are
kept out of the main figure packet because the canonical target figures above
carry the integrated target, disease-context, tumor-source, eligibility, and
uncertainty story.
When curated agent-level `benefit_tier` / `toxicity_tier` fields are present,
priority ranking can use them; otherwise survival benefit and toxicity are not
inferred from expression alone.
"""
    readme_path.write_text(readme)
    print(f"[output] Wrote {readme_path}")

    # Structured report document (§2.6b): one serialized artifact the interpretive
    # PDF renders from, so the PDF can never disagree with the figures/markdown. It
    # carries the frozen ReportView headline + the just-emitted reports (parsed once,
    # server-side, so the markdown stays byte-stable) + a belief-gated figure
    # manifest. Written before manifest discovery so it is itself catalogued.
    # Best-effort: a serialization failure must never abort the analysis (the PDF
    # falls back to building the document from the markdown directly).
    try:
        from .report_document import write_report_document

        report_document_path = write_report_document(
            paths.out_dir,
            paths.prefix_base,
            report_view=analysis.get("report_view"),
        )
        print(f"[output] Wrote {report_document_path}")
    except (OSError, ValueError, TypeError):
        # Tolerate only the realistic failure surface of "parse the reports, build a
        # dict, json.dump it, write the file": filesystem errors and value/type
        # errors from (de)serialization. Anything else here — a KeyError,
        # AttributeError — is a bug in the document builder, not a runtime data
        # condition, so let it surface loudly instead of silently shipping a run
        # whose interpretive PDF has to fall back to scraping the markdown.
        _LOGGER.warning(
            "write_report_document failed; interpretive PDF will fall back to markdown",
            exc_info=True,
        )

    manifest_path = "%s-manifest.json" % prefix if prefix else "manifest.json"
    run.artifacts = discover_output_artifacts(paths.out_dir, paths.prefix_base)
    run.add_artifact(
        manifest_path,
        kind="metadata",
        step="output",
        role="run_manifest",
        description="Machine-readable list of emitted reports, figures, tables, and metadata.",
    )
    run.note_step(
        "output",
        outputs={
            "manifest": manifest_path,
            "n_artifacts": len(run.artifacts),
        },
    )
    write_json(manifest_path, run.public_manifest())
    print(f"[output] Wrote {manifest_path}")


def _sample_mode_display(sample_mode):
    labels = {
        "solid": "solid tumor bulk",
        "heme": "hematologic / lymphoid bulk",
        "pure": "pure population / cell culture",
        "auto": "auto",
    }
    return labels.get(sample_mode, sample_mode)


def _purity_metric_label(sample_mode):
    if sample_mode == "heme":
        return "malignant-lineage fraction proxy"
    if sample_mode == "pure":
        return "population purity consistency"
    return "tumor purity"


def _format_percent(value) -> str:
    try:
        if value is None:
            return ""
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return ""


def _format_purity_interval(estimate, lower, upper) -> str:
    estimate_text = _format_percent(estimate)
    if not estimate_text:
        return "not estimated"
    lower_text = _format_percent(lower)
    upper_text = _format_percent(upper)
    if lower_text and upper_text:
        return f"{estimate_text} [{lower_text}-{upper_text}]"
    return estimate_text


def _purity_ci_phrase(purity):
    """Render the purity estimate and heuristic range with an explicit low-confidence
    tag when the interval is so wide it provides almost no constraint
    (issue #79). A 19%-100% range should NOT look the same to a reader as
    a 58%-70% range.
    """
    est = purity["overall_estimate"]
    lo = purity["overall_lower"]
    hi = purity["overall_upper"]
    if est is None or lo is None or hi is None:
        return "**not estimated**"
    tier = _ci_confidence_tier(lo, hi)
    core = f"**{est:.0%}** (model interval {lo:.0%}–{hi:.0%})"
    if tier == "degenerate":
        core += (
            " — **degenerate range**: input had no per-gene variation so "
            "uncertainty could not be estimated from the data (typical of synthetic "
            "/ cohort-median inputs)"
        )
    elif tier == "low":
        core += (
            " — **low confidence**: the range spans "
            f"{(hi - lo):.0%}, so per-gene tumor-expression estimates "
            "derived from this purity carry wide error bars"
        )
    elif tier == "moderate":
        core += " (moderate-width range)"
    return core


def _background_section_config(sample_mode):
    if sample_mode == "pure":
        return (
            "Residual Background Signatures",
            "In pure-population mode these scores are mostly a contamination check. "
            "Small residual matches do not imply a real anatomical site.\n",
        )
    if sample_mode == "heme":
        return (
            "Lineage / Background Context",
            "These scores summarize hematopoietic or tissue-related programs that remain in the sample. "
            "For heme specimens they should be read as lineage/background context rather than metastatic site calls.\n",
        )
    return (
        "Residual Tissue-like Programs",
        "These scores are normalized similarity to reference nTPM profiles. "
        "They summarize residual tissue-like programs left in the sample after normalization, "
        "which can reflect host background, ectopic lineage programs, developmental/neuronal programs, "
        "or CTA-like/testis-like signal. They are not literal composition percentages. "
        "Strong off-primary organ signal is surfaced as inferred host-site context in the decomposition section. "
        "Not every anatomic site exists as its own reference row here "
        "(for example, the panel has bone marrow but not a standalone bone/osteoblast tissue row). "
        "The `N genes` column is the size of the retained tissue-specific panel, and "
        "`Top matching genes` shows the strongest sample genes from that panel.\n",
    )


def _decomposition_section_title(sample_mode):
    if sample_mode == "pure":
        return "Pure-Population Fit"
    if sample_mode == "heme":
        return "Heme Context Decomposition"
    return "Broad Decomposition"


def _decomposition_fraction_label(sample_mode):
    if sample_mode == "pure":
        return "population purity"
    if sample_mode == "heme":
        return "malignant-lineage fraction"
    return "tumor fraction"


def _target_report_mode_intro(sample_mode, cancer_code, p_lo, p_mid, p_hi):
    if p_lo is None or p_mid is None or p_hi is None:
        if sample_mode == "pure":
            return (
                "Population-expression range using purity-like consistency "
                "**not estimated**. In pure mode these values are close to direct "
                "observed cellular expression rather than a bulk tumor-vs-background "
                f"deconvolution against {cancer_code} TCGA.\n"
            )
        if sample_mode == "heme":
            return (
                "Malignant-lineage expression range using fraction proxy "
                "**not estimated**. In heme mode these values reflect "
                "malignant-lineage-enriched expression under hematopoietic "
                "background assumptions rather than a strict epithelial "
                "tumor-vs-immune subtraction.\n"
            )
        return (
            "Purity-adjusted tumor expression range using purity "
            "**not estimated** (low / estimate / high).\n"
        )
    if sample_mode == "pure":
        return (
            f"Population-expression range using purity-like consistency **{p_lo:.0%} / {p_mid:.0%} / {p_hi:.0%}** "
            f"(low / estimate / high). In pure mode these values are close to direct observed cellular expression rather "
            f"than a bulk tumor-vs-background deconvolution against {cancer_code} TCGA.\n"
        )
    if sample_mode == "heme":
        return (
            f"Malignant-lineage expression range using fraction proxy **{p_lo:.0%} / {p_mid:.0%} / {p_hi:.0%}** "
            f"(low / estimate / high). In heme mode these values reflect malignant-lineage-enriched expression under "
            f"hematopoietic background assumptions rather than a strict epithelial tumor-vs-immune subtraction.\n"
        )
    return (
        f"Purity-adjusted tumor expression range using purity **{p_lo:.0%} / {p_mid:.0%} / {p_hi:.0%}** "
        f"(low / estimate / high).\n"
    )


def _target_value_label(sample_mode):
    if sample_mode == "pure":
        return "Context TPM (model)"
    if sample_mode == "heme":
        return "Context TPM (model)"
    return "Context TPM (model)"


def _mhc1_status_text(mhc1):
    mhc1 = mhc1 or {}
    b2m = float(mhc1.get("B2M", 0) or 0)
    hla_mean = sum(float(mhc1.get(g, 0) or 0) for g in ("HLA-A", "HLA-B", "HLA-C")) / 3
    if hla_mean > 50 and b2m > 100:
        return (
            "adequate",
            f"adequate (HLA mean={hla_mean:.0f}, B2M={b2m:.0f} TPM) — "
            "intracellular targets are plausibly presentable.",
        )
    if hla_mean > 10:
        return (
            "reduced",
            f"reduced (HLA mean={hla_mean:.0f}, B2M={b2m:.0f} TPM) — "
            "intracellular targeting may have limited efficacy.",
        )
    return (
        "low/absent",
        f"low/absent (HLA mean={hla_mean:.0f}, B2M={b2m:.0f} TPM) — "
        "surface-directed strategies are safer than TCR-style approaches.",
    )


def _lineage_caveat_text(sample_mode, cancer_code):
    if sample_mode == "pure":
        return (
            "these are lineage-identity genes. In pure-population mode they help "
            "confirm that the population still resembles the expected lineage, but "
            "they do not prove every expressing cell is malignant."
        )
    if sample_mode == "heme":
        return (
            "these are lineage-identity genes. They help separate the malignant "
            "program from unrelated hematopoietic background, but reactive cells of "
            "the same lineage can express them too, so they do not by themselves "
            "distinguish malignant from benign same-lineage cells."
        )
    if epithelial_matched_normal_component(cancer_code) is not None:
        return (
            "these are lineage-identity genes. They help confirm tissue-of-origin "
            "against unrelated TME background, but epithelial primaries can share "
            "them with admixed benign parent tissue. They do NOT by themselves "
            "distinguish tumor cells from benign cells of the same lineage."
        )
    return (
        "these are lineage-identity genes. They help confirm tissue-of-origin "
        "against unrelated host background, but benign cells of the same lineage "
        "can also express them. They do NOT by themselves distinguish malignant "
        "from benign cells when both share the same program."
    )


def _matched_normal_split_summary(ranges_df):
    if "matched_normal_tissue" not in ranges_df.columns:
        return None
    import pandas as pd

    mn_values = ranges_df["matched_normal_tissue"].dropna().astype(str)
    mn_nonempty = [v for v in mn_values.unique() if v]
    if not mn_nonempty:
        return None
    mn_tissue = mn_nonempty[0]
    mn_frac_series = ranges_df.get(
        "matched_normal_fraction", pd.Series(dtype=float)
    ).dropna()
    mn_frac = float(mn_frac_series.iloc[0]) if len(mn_frac_series) else 0.0
    if mn_frac < 0.01:
        return None
    tissue_label = mn_tissue.replace("_", " ")
    return (
        f"estimated benign parent-tissue admixture from a {tissue_label}-like matched-normal reference at **{mn_frac:.0%}** of the sample. "
        "Per-gene estimates subtract both stromal/immune TME and benign parent-tissue "
        "signal before dividing by purity."
    )


def _selected_report_scope_label(analysis):
    code = str(analysis.get("report_scope_cancer_type") or "").strip()
    if code:
        return code
    evidence = analysis.get("cancer_type_evidence") or {}
    selected = evidence.get("selected") if isinstance(evidence, dict) else {}
    if not isinstance(selected, dict):
        return ""
    code = str(selected.get("cancer_type") or "").strip()
    selected_by = str(selected.get("selected_by") or "").strip()
    if code and selected_by not in {
        "primary_expression_match",
        "pan_cancer_signature_ranker",
    }:
        return code
    return ""


def _selected_report_scope_basis_label(analysis):
    evidence = analysis.get("cancer_type_evidence") or {}
    selected = evidence.get("selected") if isinstance(evidence, dict) else {}
    selected_by = ""
    if isinstance(selected, dict):
        selected_by = str(selected.get("selected_by") or "").strip()
    if not selected_by and analysis.get("fusion_report_scope_inference"):
        selected_by = "direct_fusion"
    if not selected_by and analysis.get("rare_report_scope_inference"):
        selected_by = "rare_marker"
    if not selected_by and analysis.get("cancer_type_source") == "user-specified":
        return "the supplied cancer label"
    return {
        "direct_fusion": "direct fusion evidence",
        "rare_marker": "rare RNA-marker and expression-context evidence",
        "local_expression_reference": "exact local expression-reference evidence",
        "fine_reference": "fine-grained expression-reference evidence",
        "learned_expression_classifier": "learned full-profile expression classifier evidence",
        "broad_rna_subtype": "pan-cancer signature-ranker subtype context",
        "pan_cancer_signature_subtype": "pan-cancer signature-ranker subtype context",
        "lineage_panel": "lineage-panel evidence",
        "tumor_label_refinement": "tumor-label refinement evidence",
        "coarse_composition_reference": "coarse composition-reference evidence",
        "pan_cancer_signature_ranker": "pan-cancer signature-ranker context",
        "primary_expression_match": "legacy leading RNA expression-reference context",
    }.get(selected_by, "integrated classifier evidence")


def _selected_report_scope_caveat(analysis):
    evidence = analysis.get("cancer_type_evidence") or {}
    selected = evidence.get("selected") if isinstance(evidence, dict) else {}
    if not isinstance(selected, dict):
        return ""
    return str(selected.get("caveat") or "").strip()


def _candidate_label_options(analysis):
    candidate_trace = analysis.get("candidate_trace", [])
    fit_quality = analysis.get("fit_quality", {})
    constraints = analysis.get("analysis_constraints") or {}
    constrained_code = str(constraints.get("cancer_type") or "").strip()
    if constrained_code:
        return [constrained_code]
    if not candidate_trace:
        return []
    labels = [candidate_trace[0]["code"]]
    if len(candidate_trace) >= 2 and fit_quality.get("label") in {"weak", "ambiguous"}:
        labels.append(candidate_trace[1]["code"])
    selected_report_scope = _selected_report_scope_label(analysis)
    if selected_report_scope:
        if selected_report_scope not in labels:
            return [selected_report_scope]
        if labels and labels[0] != selected_report_scope:
            reordered = [selected_report_scope]
            reordered.extend(label for label in labels if label != selected_report_scope)
            return reordered[:2]
    return labels[:2]


_TEMPLATE_PRIMARY_COMPATIBILITY = {
    "met_adrenal": ("adrenal",),
    "met_bone": ("bone",),
    "met_brain": ("brain", "cerebr", "cns"),
    "met_liver": ("liver",),
    "met_lung": ("lung",),
    "met_lymph_node": ("lymph_node", "lymph", "tonsil", "thymus", "spleen"),
    "met_peritoneal": ("peritone", "mesothelial", "appendix"),
    "met_skin": ("skin",),
    "met_soft_tissue": (
        "soft_tissue",
        "smooth_muscle",
        "adipose",
        "skeletal_muscle",
        "vascular",
    ),
}


_MET_SITE_TISSUE_TO_HINT = {
    "adrenal_gland": "adrenal",
    "bone_marrow": "bone",
    "cerebellum": "brain",
    "cerebral_cortex": "brain",
    "hippocampal_formation": "brain",
    "amygdala": "brain",
    "basal_ganglia": "brain",
    "hypothalamus": "brain",
    "midbrain": "brain",
    "choroid_plexus": "brain",
    "spinal_cord": "brain",
    "liver": "liver",
    "lung": "lung",
    "lymph_node": "lymph_node",
    "skin": "skin",
}

_MET_TEMPLATE_TO_SITE_HINT = {
    "met_adrenal": "adrenal",
    "met_bone": "bone",
    "met_brain": "brain",
    "met_liver": "liver",
    "met_lung": "lung",
    "met_lymph_node": "lymph_node",
    "met_peritoneal": "peritoneal",
    "met_skin": "skin",
    "met_soft_tissue": "soft_tissue",
}

_WEAK_MET_SITE_WARNING = "Metastatic-site evidence below template-specific threshold"


def _canonical_met_site_hint(value):
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return None
    if text in set(_MET_TEMPLATE_TO_SITE_HINT.values()):
        return text
    if text in {"cns", "central_nervous_system", "central_nervous"}:
        return "brain"
    return _MET_SITE_TISSUE_TO_HINT.get(text)


def _primary_site_hints_for_analysis(analysis=None, cancer_code=None):
    hints = {
        hint
        for hint in (
            _canonical_met_site_hint(tissue)
            for tissue in _primary_tissues_for_analysis(
                analysis=analysis,
                cancer_code=cancer_code,
            )
        )
        if hint
    }
    try:
        desc, _ = _analysis_primary_descriptor(
            analysis=analysis,
            cancer_code=cancer_code,
        )
    except Exception:
        desc = ""
    desc = str(desc or "").lower().replace("-", "_").replace(" ", "_")
    for template, site_hint in _MET_TEMPLATE_TO_SITE_HINT.items():
        keywords = _TEMPLATE_PRIMARY_COMPATIBILITY.get(template, ())
        if any(str(keyword).lower() in desc for keyword in keywords):
            hints.add(site_hint)
    return hints


def _primary_tissues_for_analysis(analysis=None, cancer_code=None):
    lookup_code = None
    if analysis is not None:
        try:
            lookup_code = resolved_subtype_code_for_analysis(analysis)
        except Exception:
            lookup_code = None
    lookup_code = lookup_code or str(cancer_code or "").strip() or None
    if not lookup_code:
        return []
    try:
        from .decomposition.templates import get_template_host_tissues

        tissues = get_template_host_tissues("solid_primary", cancer_type=lookup_code)
        if tissues:
            return tissues
    except Exception:
        pass
    try:
        from trufflepig.cancer_ontology import cancer_type_registry

        reg = cancer_type_registry()
        match = reg[reg["code"].astype(str) == str(lookup_code)]
        if not match.empty:
            tissue = str(match.iloc[0].get("primary_tissue") or "").strip()
            if tissue and tissue.lower() != "nan":
                return [tissue]
    except Exception:
        pass
    return []


def _infer_likely_met_site_context(
    analysis,
    *,
    explicit_site_hint=None,
    explicit_met_site=None,
    tumor_context="auto",
    decomposition_templates=None,
):
    """Infer likely metastatic host site from strong off-primary tissue signal.

    This is intentionally not a user constraint. It is a report/decomposition
    prior for cases like BLCA with a high liver residual program: strong host
    tissue evidence should make the liver-met model compete even if the user did
    not pass ``--site-hint liver``.
    """

    if not analysis or analysis.get("sample_mode") not in {None, "auto", "solid"}:
        return None
    if explicit_site_hint or explicit_met_site or decomposition_templates:
        return None
    if tumor_context == "primary":
        return None

    tissue_scores = list(analysis.get("tissue_scores") or [])
    if not tissue_scores:
        return None
    top_tissue, top_score, top_n = tissue_scores[0]
    top_tissue = str(top_tissue or "").strip()
    top_score = float(top_score or 0.0)
    site_hint = _MET_SITE_TISSUE_TO_HINT.get(top_tissue)
    if not site_hint or top_score < 0.90:
        return None

    cancer_code = str(analysis.get("cancer_type") or "").strip()
    primary_tissues = _primary_tissues_for_analysis(
        analysis=analysis,
        cancer_code=cancer_code,
    )
    if top_tissue in set(primary_tissues):
        return None
    if site_hint in _primary_site_hints_for_analysis(
        analysis=analysis,
        cancer_code=cancer_code,
    ):
        return None

    score_by_tissue = {
        str(tissue): float(score or 0.0) for tissue, score, _ in tissue_scores
    }
    primary_scores = [
        (tissue, score_by_tissue.get(tissue, 0.0)) for tissue in primary_tissues
    ]
    primary_tissue, primary_score = (
        max(primary_scores, key=lambda item: item[1])
        if primary_scores
        else ("", 0.0)
    )
    if top_score - primary_score < 0.10:
        return None

    return {
        "site": site_hint,
        "tissue": top_tissue,
        "score": top_score,
        "n_genes": int(top_n or 0),
        "primary_tissue": primary_tissue,
        "primary_tissue_score": float(primary_score),
        "basis": "strong_off_primary_host_tissue",
        "message": (
            f"Strong {top_tissue.replace('_', ' ')} host/background program "
            f"({top_score:.2f}) exceeds the expected primary-tissue program "
            f"{primary_tissue.replace('_', ' ') if primary_tissue else 'unknown'} "
            f"({primary_score:.2f}); treating {site_hint.replace('_', ' ')} as a likely "
            "metastatic host site for decomposition and target-background modeling."
        ),
    }


def _site_evidence_supports_met_site(site_evidence):
    if not isinstance(site_evidence, dict):
        return None
    if "site_supported" in site_evidence:
        return bool(site_evidence.get("site_supported"))
    return None


def _decomposition_met_site_is_supported(decomp_result, *, analysis=None):
    template = str(getattr(decomp_result, "template", "") or "")
    if not template.startswith("met_"):
        return False
    site_evidence = getattr(decomp_result, "site_evidence", None)
    site_supported = _site_evidence_supports_met_site(site_evidence)
    warnings = getattr(decomp_result, "warnings", None) or []
    if any("No non-tumor components in template" in str(warning) for warning in warnings):
        return False
    explicit_site_context = (
        isinstance(site_evidence, dict) and site_evidence.get("basis") == "site_hint"
    )
    if site_supported is not None:
        if not site_supported or explicit_site_context:
            return site_supported
        overexplained = any(
            "overexplained by the TME background" in str(warning)
            for warning in warnings
        )
        if overexplained:
            return False
        return True
    if any(_WEAK_MET_SITE_WARNING in str(warning) for warning in warnings):
        return False
    if (
        "Primary tissue support exceeds metastatic-site support" in warnings
        or (getattr(decomp_result, "template_site_factor", 0.0) or 0.0) < 0.75
        or (getattr(decomp_result, "template_tissue_score", 0.0) or 0.0) < 0.4
    ):
        return False
    return True


def _decomposition_met_site_can_propagate(decomp_result, *, analysis=None):
    """Whether a met template should act as a literal metastatic site.

    A primary-compatible template can be well-supported evidence for the
    background/site *fit* (for example GBM with a brain template) without being
    evidence of metastasis. Keep that propagation decision separate from the
    evidence-support decision used by report summaries.
    """
    if not _decomposition_met_site_is_supported(decomp_result, analysis=analysis):
        return False
    template = str(getattr(decomp_result, "template", "") or "")
    if analysis is not None and _template_primary_compatible(
        template,
        analysis=analysis,
        cancer_code=analysis.get("cancer_type"),
    ):
        return False
    return True


def _supported_decomposition_met_site(analysis):
    if not isinstance(analysis, dict):
        return None
    decomp_results = analysis.get("decomposition_results") or []
    if decomp_results:
        best = decomp_results[0]
        if _decomposition_met_site_can_propagate(best, analysis=analysis):
            return _MET_TEMPLATE_TO_SITE_HINT.get(str(getattr(best, "template", "")))
    decomposition = analysis.get("decomposition") or {}
    supported = decomposition.get("supported_met_site")
    return supported or None


def _effective_met_site_for_background(analysis):
    constraints = analysis.get("analysis_constraints") or {}
    return (
        constraints.get("met_site")
        or ((analysis.get("inferred_site_context") or {}).get("site"))
        or _supported_decomposition_met_site(analysis)
    )


def _decomposition_matches_report_reference(decomp_result, *, reference_code, report_code):
    """Whether a decomposition hypothesis is compatible with the report label."""
    code = str(getattr(decomp_result, "cancer_type", "") or "").strip()
    reference_code = str(reference_code or "").strip()
    report_code = str(report_code or "").strip()
    if not code:
        return False
    if code in {reference_code, report_code}:
        return True
    try:
        from .analyze.cancer_type_context import registry_parent_code

        return bool(
            (report_code and registry_parent_code(report_code) == code)
            or (code and registry_parent_code(code) in {reference_code, report_code})
        )
    except Exception:
        return False


def _decomposition_is_acceptable_background_model(
    decomp_result,
    *,
    analysis=None,
    report_code=None,
):
    template = str(getattr(decomp_result, "template", "") or "")
    if not template.startswith("met_"):
        return True
    if _template_primary_compatible(
        template,
        analysis=analysis,
        cancer_code=report_code or (analysis or {}).get("cancer_type"),
    ):
        return True
    return _decomposition_met_site_is_supported(decomp_result, analysis=analysis)


def _prioritize_report_compatible_decomposition(
    decomp_results,
    *,
    reference_code,
    report_code,
    enabled,
    analysis=None,
):
    """Put the report-compatible decomposition first for target attribution.

    The decomposer scores fit hypotheses, not diagnoses. For a known supplied
    or refined report label, a discordant template can remain useful as an
    uncertainty signal, but it should not silently become the background model
    for target subtraction when a reference-compatible fit exists.
    """
    if not decomp_results:
        return decomp_results
    reference_code = str(reference_code or "").strip()
    report_code = str(report_code or "").strip()
    if not enabled and not (reference_code or report_code):
        return decomp_results
    first = decomp_results[0]
    if _decomposition_matches_report_reference(
        first, reference_code=reference_code, report_code=report_code
    ) and _decomposition_is_acceptable_background_model(
        first, analysis=analysis, report_code=report_code
    ):
        return decomp_results
    compatible = [
        row
        for row in decomp_results
        if _decomposition_matches_report_reference(
            row, reference_code=reference_code, report_code=report_code
        )
        and _decomposition_is_acceptable_background_model(
            row, analysis=analysis, report_code=report_code
        )
    ]
    if not compatible:
        return decomp_results
    selected = max(compatible, key=lambda row: float(getattr(row, "score", 0.0) or 0.0))
    selected_code = str(getattr(selected, "cancer_type", "") or "").strip()
    if selected_code == reference_code and selected_code != report_code:
        warning = (
            "Selected fallback-reference decomposition for target/background "
            f"attribution; raw best fit was {first.cancer_type}/{first.template}"
        )
    else:
        warning = (
            "Selected report-compatible decomposition for target/background "
            f"attribution; raw best fit was {first.cancer_type}/{first.template}"
        )
    warnings = getattr(selected, "warnings", None)
    if isinstance(warnings, list) and warning not in warnings:
        warnings.append(warning)
    return [selected] + [row for row in decomp_results if row is not selected]


def _reconcile_purity_after_decomposition(
    analysis, best_decomp, *, reference_cancer_code, effective_purity, run
):
    """Post-decomposition purity reconciliation, run once for the winning decomposition
    BEFORE ``_finalize_fused_purity``: adopt a valid decomposition purity, promote a
    lineage-panel override, then cap the interval by the modeled non-tumor mass.

    The decomposition's purity is only safe to adopt when its cancer_type matches the
    classifier AND its template has non-tumor compartments
    (``should_adopt_decomposition_purity``): a template with no TME compartments trivially
    returns fraction=1.0 (everything maps to tumor by construction), which is not a purity
    measurement — the canonical failure case is a CRC sample the classifier scored at 36%
    (COAD) whose best-fit template was BRCA/solid_primary with "No non-tumor components",
    propagating fraction=100% as the headline purity.

    Mutates ``analysis["purity"]`` (and the winner's candidate_trace row, via
    ``_set_analysis_purity``) in place, and returns the possibly-updated
    ``effective_purity`` the caller threads onward.
    """
    if should_adopt_decomposition_purity(reference_cancer_code, best_decomp):
        decomp_purity = best_decomp.purity_result
        if isinstance(decomp_purity, dict):
            # Reconcile a FRAGILE decomposition purity against the independent classifier
            # signals before adoption (policy chosen 2026-07-07): a fragile fit that disagrees
            # with every independent signal is rejected (keep the classifier purity); otherwise a
            # fragile fit is adopted with its interval widened to span the plausible template
            # range. A non-fragile fit is adopted unchanged (prior behavior). On "reject",
            # effective_purity stays the classifier purity (set by the caller), so the
            # lineage-panel override below never promotes the rejected fit.
            classifier_purity = analysis.get("purity")
            stability = (analysis.get("decomposition") or {}).get("purity_stability")
            action, reconciled = reconcile_decomposition_purity(
                classifier_purity, decomp_purity, stability
            )
            if isinstance(analysis.get("decomposition"), dict):
                analysis["decomposition"]["purity_reconciliation"] = action
            if action != "reject":
                # Single source of truth: analysis['purity'] and the winner's candidate_trace row
                # become the same object, so the in-place lineage-panel override below stays in sync.
                effective_purity = reconciled
                _set_analysis_purity(analysis, effective_purity)

    # Propagate a lineage-panel purity override back into ``analysis["purity"]`` so every
    # downstream report is consistent (bug 2026-04-14: user saw 23% in the headline and 64%
    # in the decomposition-hypotheses table). When the decomposition's best hypothesis used a
    # non-signature purity source we promote it, preserve the original estimate as
    # ``signature_based_estimate``, and reset the CI widening and downstream pct_cancer_median
    # math to use the new anchor.
    purity_source_best = (
        effective_purity.get("purity_source")
        if isinstance(effective_purity, dict)
        else None
    )
    if (
        purity_source_best in ("lineage_panel",)
        and isinstance(effective_purity, dict)
        and "overall_estimate" in effective_purity
    ):
        orig_purity = dict(analysis["purity"])
        analysis["purity"]["signature_based_estimate"] = orig_purity.get(
            "overall_estimate"
        )
        analysis["purity"]["signature_based_lower"] = orig_purity.get("overall_lower")
        analysis["purity"]["signature_based_upper"] = orig_purity.get("overall_upper")
        analysis["purity"]["overall_estimate"] = effective_purity["overall_estimate"]
        analysis["purity"]["overall_lower"] = effective_purity.get(
            "overall_lower", effective_purity["overall_estimate"]
        )
        analysis["purity"]["overall_upper"] = effective_purity.get(
            "overall_upper", effective_purity["overall_estimate"]
        )
        analysis["purity"]["purity_source"] = purity_source_best
        analysis["purity"]["lineage_tumor_fraction"] = effective_purity.get(
            "lineage_tumor_fraction"
        )
        print(
            f"[analysis] Adopted lineage-panel purity "
            f"{analysis['purity']['overall_estimate']:.0%} "
            f"(signature-based estimate was "
            f"{orig_purity.get('overall_estimate', 0):.0%})"
        )

    purity = analysis["purity"]
    if _constrain_purity_interval_with_decomposition(purity, best_decomp):
        interval_cap = purity.get("components", {}).get("decomposition_interval_cap", {})
        effective_purity = purity
        print(
            "[analysis] Purity upper bound constrained by decomposition: "
            f"{interval_cap.get('original_upper', 0):.0%} -> "
            f"{interval_cap.get('constrained_upper', 0):.0%}"
        )
        run.note_step(
            "purity_interval_refinement",
            outputs={
                "overall_estimate": purity.get("overall_estimate"),
                "overall_lower": purity.get("overall_lower"),
                "overall_upper": purity.get("overall_upper"),
                "decomposition_interval_cap": interval_cap,
            },
        )
    return effective_purity


def _finalize_fused_purity(analysis):
    """Final purity pass: honest random-effects interval + anti-saturation guard.

    After every adoption/override/cap above has settled ``analysis["purity"]``, fuse
    the per-method estimates (signature / lineage / ESTIMATE / decomposition) with the
    random-effects model so the reported interval widens automatically when the methods
    disagree, and guard against a saturated near-100% read that no reliable signal
    corroborates — the failure mode that made a rare tissue type (e.g. NUT carcinoma)
    report a spurious 100% purity when its lineage-identity genes and the ESTIMATE
    surrogate co-saturated to 1.0 while the physical decomposition residual said ~55%.
    The empirically-best combiner point is preserved except on that guarded case.
    Grounded in the HCC1395×HPA in-silico mixture benchmark; see
    ``trufflepig.purity_integration.best_purity_estimate``. Callers run this BEFORE the
    ReportView freeze so every headline read (figure, brief, markdown) sees the same
    finalized number by construction. Mutates ``analysis["purity"]`` in place; any
    failure is swallowed (keep the pre-fusion purity) so it never aborts a finished run.
    """
    try:
        from .purity_integration import best_purity_estimate

        final_purity = analysis.get("purity")
        if not isinstance(final_purity, dict):
            return
        stability = (analysis.get("decomposition") or {}).get("purity_stability")
        # Decomposition's contribution to the fusion is its RESIDUAL FRACTION — the mass
        # fraction of the tumor-specific residual after background subtraction (CLAUDE.md:
        # "the PRIMARY purity signal"). This is the independent PHYSICAL tumor-vs-background
        # reading, and it is what discriminates a genuinely-pure sample from a lineage-identity
        # artifact: for a rare tissue type (NUT carcinoma) whose lineage-identity genes and
        # ESTIMATE both saturate to 100%, the residual fraction stays ~0.55, exposing the
        # saturation. (The purity_stability hypothesis range mirrors the already-adopted overall
        # estimate, so it would echo the saturated 100% and defeat the guard — do NOT use it as
        # the decomposition point.)
        decomp_comp = (final_purity.get("components") or {}).get("decomposition") or {}
        resid = decomp_comp.get("residual_fraction")
        decomp_arg = None
        if isinstance(resid, (int, float)):
            decomp_arg = {
                "overall_estimate": float(resid),
                "fragile": bool((stability or {}).get("fragile")),
            }
        best = best_purity_estimate(final_purity, decomposition=decomp_arg)
        if best is None:
            return
        final_purity["pre_fusion_estimate"] = final_purity.get("overall_estimate")
        # Preserve the physical decomposition cap. When the adopted decomposition models
        # non-tumor mass, _constrain_purity_interval_with_decomposition may have already
        # capped overall_upper below 1.0 (purity cannot exceed 1 − modeled non-tumor mass).
        # The random-effects fusion is unaware of that structural bound, so on method
        # disagreement its pooled upper can widen back above the cap (e.g. 0.84 → 0.97),
        # re-reporting an impossible near-100% interval. Clamp the fused interval to the cap.
        decomp_cap = (
            (final_purity.get("components") or {}).get("decomposition_interval_cap") or {}
        ).get("constrained_upper")
        est, lo, up = _clamp_interval_to_cap(
            best["overall_estimate"],
            best["overall_lower"],
            best["overall_upper"],
            decomp_cap,
        )
        final_purity["overall_estimate"] = est
        final_purity["overall_lower"] = lo
        final_purity["overall_upper"] = up
        final_purity["best_integration"] = {
            "i2": best["i2"],
            "method_agreement": best["method_agreement"],
            "n_methods": best["n_methods"],
            "point_source": best["point_source"],
        }
    except Exception:  # noqa: BLE001 - purity honesty pass must never abort a finished run
        _LOGGER.warning(
            "best_purity_estimate failed; keeping pre-fusion purity", exc_info=True
        )


def _clamp_interval_to_cap(estimate, lower, upper, cap):
    """Clamp a (point, lower, upper) purity interval so ``upper`` never exceeds ``cap``.

    ``cap`` is the physical decomposition ceiling written by
    ``_constrain_purity_interval_with_decomposition`` (purity cannot exceed
    ``1 − modeled non-tumor mass``). The random-effects fusion is unaware of it,
    so its pooled upper can widen back above the cap on method disagreement.
    A no-op when ``cap`` is absent/non-numeric. The point stays the best estimate
    but is pulled down to the ceiling if it (or the lower bound) somehow exceeds it,
    keeping ``lower ≤ estimate ≤ upper`` coherent.
    """
    estimate = float(estimate)
    lower = float(lower)
    upper = float(upper)
    if isinstance(cap, (int, float)):
        upper = min(upper, float(cap))
    if estimate > upper:
        estimate = upper
    if lower > upper:
        lower = upper
    return estimate, lower, upper


def _constrain_purity_interval_with_decomposition(purity, decomp_result):
    """Cap impossible 100% purity endpoints when TME is explicitly modeled."""

    if not isinstance(purity, dict) or decomp_result is None:
        return False
    warnings = getattr(decomp_result, "warnings", None) or []
    if any("No non-tumor components in template" in warning for warning in warnings):
        return False
    try:
        estimate = float(purity.get("overall_estimate"))
        old_upper = float(purity.get("overall_upper"))
    except (TypeError, ValueError):
        return False
    if old_upper <= estimate:
        return False

    fractions = getattr(decomp_result, "fractions", None) or {}
    non_tumor_fraction = 0.0
    for component, value in fractions.items():
        if str(component) == "tumor":
            continue
        try:
            non_tumor_fraction += max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    if non_tumor_fraction < 0.05:
        return False

    # Conservative: require only half of the point-estimated non-tumor mass to
    # remain possible, rather than treating decomposition as an exact bound.
    min_non_tumor_floor = 0.5 * non_tumor_fraction
    capped_upper = max(estimate, min(old_upper, 1.0 - min_non_tumor_floor))
    if capped_upper >= old_upper - 1e-9:
        return False

    purity["overall_upper"] = float(capped_upper)
    components = purity.setdefault("components", {})
    if isinstance(components, dict):
        components["decomposition_interval_cap"] = {
            "original_upper": old_upper,
            "constrained_upper": float(capped_upper),
            "modeled_non_tumor_fraction": float(non_tumor_fraction),
            "minimum_non_tumor_floor": float(min_non_tumor_floor),
            "basis": "best_fit_decomposition_non_tumor_components",
        }
    return True


def _analysis_primary_descriptor(analysis=None, cancer_code=None):
    lookup_code = None
    if analysis is not None:
        try:
            lookup_code = resolved_subtype_code_for_analysis(analysis)
        except Exception:
            lookup_code = None
    lookup_code = lookup_code or str(cancer_code or "").strip() or None
    if not lookup_code:
        return "", None
    try:
        from trufflepig.cancer_ontology import cancer_type_registry

        reg = cancer_type_registry()
        match = reg[reg["code"] == lookup_code]
        if not match.empty:
            row = match.iloc[0]
            bits = [
                str(row.get("primary_tissue") or "").strip(),
                str(row.get("primary_template") or "").strip(),
            ]
            desc = " ".join(bit for bit in bits if bit and bit.lower() != "nan").lower()
            return desc, lookup_code
    except Exception:
        pass
    return str(lookup_code).lower(), lookup_code


def _template_primary_compatible(template_name, *, analysis=None, cancer_code=None):
    template_text = str(template_name or "").strip()
    if not template_text.startswith("met_"):
        return False
    site_hint = _MET_TEMPLATE_TO_SITE_HINT.get(template_text)
    if site_hint and site_hint in _primary_site_hints_for_analysis(
        analysis=analysis,
        cancer_code=cancer_code,
    ):
        return True
    desc, _ = _analysis_primary_descriptor(analysis=analysis, cancer_code=cancer_code)
    keywords = _TEMPLATE_PRIMARY_COMPATIBILITY.get(template_text, ())
    return any(keyword in desc for keyword in keywords)


def _template_site_display(template_name, *, analysis=None, cancer_code=None):
    mapping = {
        "met_adrenal": "adrenal",
        "met_bone": "bone",
        "met_brain": "brain",
        "met_liver": "liver",
        "met_lung": "lung",
        "met_lymph_node": "lymph node",
        "met_peritoneal": "peritoneal",
        "met_skin": "skin",
        "met_soft_tissue": "soft-tissue",
        "solid_primary": "primary site",
        "pure_population": "pure population",
        "heme_blood": "blood",
        "heme_marrow": "marrow",
        "heme_nodal": "lymphoid / nodal",
    }
    template_text = str(template_name or "").strip()
    base = mapping.get(template_text, template_text.replace("_", " "))
    if template_text == "solid_primary":
        return "primary site"
    if template_text.startswith("met_"):
        if _template_primary_compatible(
            template_text, analysis=analysis, cancer_code=cancer_code
        ):
            return f"primary-compatible {base} context"
        return f"{base}-associated host context"
    return base


def _hypothesis_display_label(result, *, primary_code=None, analysis=None):
    cancer_code = str(getattr(result, "cancer_type", "") or "").strip()
    template = str(getattr(result, "template", "") or "").strip()
    if template.startswith("met_") and not _decomposition_met_site_is_supported(
        result, analysis=analysis
    ):
        cancer_label = _cancer_label(cancer_code)
        if primary_code and cancer_code == primary_code:
            return f"{cancer_label} host-background fit pattern (site indeterminate)"
        return f"{cancer_label}-like host-background fit pattern (site indeterminate)"
    return _hypothesis_label(
        f"{cancer_code} / {template}",
        primary_code=primary_code,
        analysis=analysis,
    )


def _report_scope_cancer_type(cancer_type):
    """Resolve non-TCGA registry labels that should remain visible in reports."""
    text = str(cancer_type or "").strip()
    if not text:
        return None
    norm = text.lower().replace("-", " ").replace("_", " ").strip()
    aliases = {
        "nut": "NUTM",
        "nut carcinoma": "NUTM",
        "nut midline carcinoma": "NUTM",
        "nutm": "NUTM",
        "nutm1": "NUTM",
        "nutm1 rearranged carcinoma": "NUTM",
    }
    if norm in aliases:
        return aliases[norm]
    code = text.upper()
    try:
        from trufflepig.cancer_ontology import cancer_type_registry

        reg = cancer_type_registry()
        codes_by_upper = {
            str(reg_code).upper(): str(reg_code)
            for reg_code in reg["code"].dropna().astype(str)
        }
        if code in codes_by_upper:
            return codes_by_upper[code]
        match = reg[reg["name"].astype(str).str.lower() == norm]
        if not match.empty:
            return str(match.iloc[0]["code"])
    except Exception:
        return None
    return None


def _set_analysis_purity(analysis, purity, code=None):
    """Single source of truth for the winner's purity.

    Writes ``analysis['purity']`` and the matching ``candidate_trace`` row's ``purity_result`` to
    the SAME object in one step. The winner's purity lives in both places — ``analysis['purity']``
    is what reports read, and ``decompose_sample`` reuses the candidate row — so every write must
    update both or a later consumer reads a stale duplicate (the reroute-desync bug class). Routing
    all purity writes through here makes that impossible by construction. ``code`` selects the row
    (defaults to the current call); pass it explicitly when writing before ``cancer_type`` is set.
    """
    analysis["purity"] = purity
    target = str(code or analysis.get("cancer_type") or "")
    for row in analysis.get("candidate_trace") or []:
        if str(row.get("code")) == target:
            row["purity_result"] = purity
            break


def _finalize_purity_for_final_call(analysis, df_expr, final_code):
    """Recompute the WHOLE purity for the FINAL evidence-refined cancer type when it differs
    from the code the purity was originally computed for.

    ``analyze_sample`` binds purity (and its one decomposition, via ``finalize_winner_purity``)
    to the pre-evidence RANKING WINNER, which runs before the evidence selector. When the
    selector then refines the call to a DIFFERENT code, everything code/lineage-dependent —
    ``cancer_type``, ``lineage_compartment``, ``estimate_gated_for_lineage``, the gated
    ``overall_estimate``, integration source, the signature/reference purity, AND the
    decomposition — still describes the pre-evidence code (an ATRT scored as LGG got a
    solid-mode decomposition; a Sarcoma/Heme rescue still used ESTIMATE). This recomputes the
    full purity for ``final_code`` (explicit → trusted, no expression override) so the reported
    purity is consistent with the final call. No-op when they already agree; fully fallback-safe.

    This does NOT feed the cancer-TYPE call (``final_call`` is chosen upstream by the ranker +
    selector); it only makes the reported PURITY consistent with that call. Eliminating the
    recompute entirely — computing the decomposition once, directly for the final code — is
    redesign-doc Phase 4 Stage 4 (fold report-scope selection into the ranking so the winner
    already equals the final code), which changes what the selector reads and so is gated by the
    565-sample confusion eval; this function is the intermediate that keeps purity correct until
    then. Renamed from ``_reroute_decomposition_to_call`` (which misleadingly named a decomposition
    re-route for a full purity recompute).
    """
    try:
        purity = analysis.get("purity") or {}
        dc = (purity.get("components") or {}).get("decomposition") or {}
        if not final_code or not isinstance(dc, dict) or not dc.get("mode"):
            return
        # Recompute whenever the evidence-refined call differs from the code the purity was
        # actually computed for (``purity['cancer_type']``). Comparing the decomposition
        # ``dc['mode']`` instead is unsafe: the decomposition component may have ALREADY
        # lineage-overridden its mode to the refined lineage, so equal modes can still hide a
        # stale cancer_type / lineage_compartment / ESTIMATE-gating / signature+reference purity
        # (the report would say SARC while every other purity field came from the pre-evidence code).
        if str(final_code) == str(purity.get("cancer_type") or ""):
            return  # purity already computed for the final code → consistent
        from .tumor_purity import estimate_tumor_purity

        # full recompute for the refined call → consistent gating / overall / compartment / decomposition
        # (the refined call can be a reference-free rare type; estimate_tumor_purity degrades gracefully)
        rerouted = estimate_tumor_purity(df_expr, cancer_type=final_code)
        # Single source of truth: write analysis['purity'] AND the refined call's candidate_trace row
        # together. ``decompose_sample`` is later handed ``candidate_rows`` and reuses that row's
        # ``purity_result``; left stale it would overwrite the rerouted purity and the report would
        # lose the refined lineage/gating. (cancer_type isn't set to final_code yet here → pass it.)
        _set_analysis_purity(analysis, rerouted, code=final_code)
    except Exception:  # noqa: BLE001 — reporting refinement; never break the analysis
        _LOGGER.warning("purity re-route to evidence call failed", exc_info=True)


def _veto_local_reference_lineage_flip(
    analysis, df_expr, report_scope_cancer_type, rna_inferred_cancer_type, selected_scope
):
    """Veto a LOCAL-EXPRESSION-REFERENCE flip that crosses lineage against an agreeing
    bulk classifier + compartment_call.

    The deconvolved local-reference channel mis-rescues epithelial samples to sarcoma (the
    epithelial→sarcoma attractor was the original failure mode, but the same contract applies to
    most cross-lineage local-reference flips). Targeted, low-regression: only the
    ``local_expression_reference`` channel is vetoed (a fusion / marker / literature rescue is
    trusted), and only when the refined lineage disagrees with BOTH the bulk classifier's call AND
    the compartment_call top — two independent signals agreeing on the lineage outweigh a single
    deconvolved-reference flip.

    Carve-out: a flip INTO the **embryonal** lineage is exempt. Embryonal tumors
    (hepatoblastoma, neuroblastoma) systematically transcribe their solid tissue-of-origin program,
    so the bulk classifier AND compartment_call *both* (wrongly) agree on "solid" while the
    deconvolved local reference — which carries an actual embryonal reference profile — is the one
    signal that is right. Vetoing that flip would revert a correct rescue (HEPB→LIHC, NBL→PCPG) to a
    worse call, so embryonal targets are never vetoed.

    Returns the (possibly reverted) ``report_scope_cancer_type`` and records the veto on
    ``analysis``.
    """
    if not report_scope_cancer_type or report_scope_cancer_type == rna_inferred_cancer_type:
        return report_scope_cancer_type
    if (selected_scope or {}).get("selected_by") != "local_expression_reference":
        return report_scope_cancer_type
    try:
        from .cancer_type_centroid import compartment_call
        from .expression_decomposition import _group_to_mode
        from trufflepig.cancer_ontology import cancer_lineage_group
        from .tumor_purity import _build_sample_tpm_by_symbol

        # Only treat the compartment as a corroborating signal when it ACTUALLY returned a
        # confident call: an abstention (compartment=None) or a near-tie (confident=False) is not
        # an independent signal, and mapping its empty/grazing label through _group_to_mode would
        # silently default to "solid" — making the veto fire as if two signals agreed when one was
        # simply absent. Leave comp=None in those cases so the `comp and …` guard below short-circuits.
        comp_call = compartment_call(_build_sample_tpm_by_symbol(df_expr))
        comp = (
            _group_to_mode(comp_call["compartment"])
            if comp_call.get("compartment") and comp_call.get("confident")
            else None
        )
        refined = _group_to_mode(cancer_lineage_group(report_scope_cancer_type) or "")
        pre = (_group_to_mode(cancer_lineage_group(rna_inferred_cancer_type) or "")
               if rna_inferred_cancer_type else None)
        # refined != "embryonal": a flip into the embryonal lineage is a legitimate rescue, not the
        # spurious sarcoma attractor this veto targets (see the carve-out in the docstring). Both the
        # bulk classifier and compartment_call systematically mis-read embryonal tumors as their
        # solid tissue-of-origin, so their "agreement" here is not two independent signals.
        if (
            comp
            and refined
            and pre
            and refined != comp
            and pre == comp
            and refined != "embryonal"
        ):
            veto_reason = (
                f"local_expression_reference cross-lineage flip ({pre} -> {refined}) "
                f"conflicts with the bulk classifier ({pre}) and compartment_call ({comp})"
            )
            analysis["cancer_type_evidence_vetoed"] = {
                "vetoed_call": report_scope_cancer_type,
                "kept_call": rna_inferred_cancer_type,
                "reason": veto_reason,
            }
            # _apply_cancer_type_evidence already wrote the vetoed selection into the analysis; revert
            # those analysis-level keys too, or downstream label helpers re-surface the vetoed SARC_* call.
            analysis["inferred_cancer_type"] = rna_inferred_cancer_type
            analysis["expression_reference_cancer_type"] = rna_inferred_cancer_type
            analysis.pop("primary_expression_context", None)
            cte = analysis.get("cancer_type_evidence")
            if isinstance(cte, dict):
                cte["selected"] = None
                graph = cte.get("staged_evidence_graph")
                if isinstance(graph, dict):
                    graph_selected = graph.get("selected")
                    if graph_selected:
                        graph["vetoed_selected"] = dict(graph_selected)
                    graph["selected"] = None
                    for channel in graph.get("channels") or []:
                        if not isinstance(channel, dict):
                            continue
                        if not channel.get("selects_report_label"):
                            continue
                        candidate = (
                            channel.get("candidate_code") or channel.get("code") or ""
                        )
                        if str(candidate) != str(report_scope_cancer_type):
                            continue
                        channel["selects_report_label"] = False
                        channel["status"] = "vetoed"
                        details = channel.get("details")
                        if not isinstance(details, dict):
                            details = {}
                            channel["details"] = details
                        details["veto_reason"] = veto_reason
            return None  # revert to the pre-evidence (bulk) call
    except Exception:  # noqa: BLE001 — refinement guard; never break the analysis
        _LOGGER.warning("local-reference lineage veto failed", exc_info=True)
    return report_scope_cancer_type


def _apply_cancer_type_evidence(
    analysis,
    df_expr,
    *,
    rna_inferred_cancer_type,
    fusion_scope_inference,
    report_scope_cancer_type,
    rare_scope_inference,
    fine_scope_inference,
):
    """Run the unified cancer-type evidence selector and propagate its output.

    Returns ``(cancer_type_evidence, selected_scope, report_scope_cancer_type,
    rare_scope_inference, fine_scope_inference)``. Mutates ``analysis`` in
    place with ``cancer_type_evidence``, ``inferred_cancer_type``,
    ``expression_reference_cancer_type``, and ``primary_expression_context``
    keys when a selection is made.

    Lifted out of ``_analyze_body`` so the wiring contract has a small,
    testable surface — see ``tests/test_cancer_type_evidence.py``.
    """
    try:
        from .cancer_type_evidence import select_report_scope_from_evidence
    except ImportError:
        _LOGGER.warning(
            "select_report_scope_from_evidence is unavailable; cancer-type "
            "evidence will not be computed",
            exc_info=True,
        )
        return (
            None,
            None,
            report_scope_cancer_type,
            rare_scope_inference,
            fine_scope_inference,
        )

    try:
        cancer_type_evidence = select_report_scope_from_evidence(
            df_expr,
            analysis,
            rare_marker_hypotheses=analysis.get("rare_marker_hypotheses"),
            fusion_scope_inference=fusion_scope_inference,
        )
    except (KeyError, ValueError, TypeError):
        _LOGGER.warning(
            "select_report_scope_from_evidence failed; cancer-type "
            "evidence will not be computed",
            exc_info=True,
        )
        return (
            None,
            None,
            report_scope_cancer_type,
            rare_scope_inference,
            fine_scope_inference,
        )

    analysis["cancer_type_evidence"] = cancer_type_evidence
    # Wiring point #2: expose the lineage-panel verdict as a
    # top-level analysis key so analysis-parameters.json carries
    # it for debugging and downstream consumers that don't dig
    # into cancer_type_evidence themselves.
    panel_evidence = (cancer_type_evidence or {}).get("lineage_panel_evidence")
    if panel_evidence is not None:
        analysis["lineage_panel_evidence"] = panel_evidence
    selected_scope = (cancer_type_evidence or {}).get("selected")
    if selected_scope and selected_scope.get("cancer_type"):
        analysis["inferred_cancer_type"] = selected_scope["cancer_type"]
        analysis["expression_reference_cancer_type"] = (
            selected_scope.get("expression_reference_cancer_type")
            or rna_inferred_cancer_type
        )
        analysis["primary_expression_context"] = (
            (cancer_type_evidence or {}).get("primary_expression_context")
        )
        selected_by = str(selected_scope.get("selected_by") or "")
        if selected_by not in {
            "primary_expression_match",
            "pan_cancer_signature_ranker",
        }:
            report_scope_cancer_type = selected_scope["cancer_type"]
            # Route by the winning selector, not by evidence-source
            # set-membership: multiple selectors can fire on the same
            # hypothesis (e.g. fine_reference + rare_marker both
            # contribute weight), and the prior set-membership check
            # would have routed a fine_reference-selected hypothesis
            # into ``rare_scope_inference`` purely because rare_marker
            # also appeared in the source list.
            if selected_by == "rare_marker":
                rare_scope_inference = selected_scope
            elif selected_by in ("fine_reference", "local_expression_reference"):
                fine_scope_inference = selected_scope
    return (
        cancer_type_evidence,
        selected_scope,
        report_scope_cancer_type,
        rare_scope_inference,
        fine_scope_inference,
    )


def _apply_cancer_type_context_roles(analysis, cancer_type_context):
    """Propagate resolved cancer-type roles to top-level analysis fields.

    The report label, broad fallback reference, and best expression reference
    are related but not interchangeable. Keeping them synchronized here avoids
    downstream modules re-deriving subtly different answers.
    """
    report_code = cancer_type_context.code_for("report")
    reference_code = (
        cancer_type_context.code_for("reference")
        or cancer_type_context.code_for("cohort")
        or report_code
    )
    expression_code = (
        cancer_type_context.code_for("expression")
        or reference_code
        or report_code
    )
    if reference_code:
        analysis["reference_cancer_type"] = reference_code
        analysis["fallback_expression_reference_cancer_type"] = reference_code
    if expression_code:
        analysis["expression_reference_cancer_type"] = expression_code
    if expression_code == report_code and expression_code == reference_code:
        role = "same_reference"
    elif expression_code == report_code and cancer_type_context.best_expression_direct:
        role = "report_label_exact"
    elif expression_code == reference_code:
        role = "fallback_reference"
    else:
        role = "related_exact_reference"
    analysis["expression_reference_role"] = role
    analysis["expression_reference_is_direct"] = bool(
        cancer_type_context.best_expression_direct
    )


@lru_cache(maxsize=256)
def _has_operational_analysis_reference(code):
    """Whether ``code`` can anchor purity/decomposition analysis directly.

    A TCGA-style pan-cancer column is sufficient, but not required: some
    pirlygenes cohorts (SCLC, NBL, MBL, RT, ...) have observed/deconvolved
    expression references plus marker support even though they are absent from
    the broad TCGA matrix. Those should still be usable as the analysis anchor
    for a refined child label.
    """
    if not code:
        return False
    if _has_pan_cancer_expression_cohort(code):
        return True
    try:
        from .analyze.cancer_type_context import expression_reference_options
        from .tumor_purity import _has_direct_purity_markers

        direct_refs = expression_reference_options(code, include_fallback=False)
        has_deconvolved_ref = any(
            record.source_kind == "deconvolved_tumor_reference"
            for record in direct_refs
        )
        return has_deconvolved_ref and bool(_has_direct_purity_markers(code))
    except Exception:
        return False


def _is_defining_fusion_label(code):
    """True for rare labels whose diagnosis should stay report-scoped.

    These may have exact expression references, but a broad lineage fallback can
    still be useful for cohort-normalized plots/pathway context. NUT carcinoma
    is the canonical case: the report label is NUTM, while squamous fallback
    context remains helpful unless the fusion/diagnostic layer asks otherwise.
    """
    if not code:
        return False
    try:
        from trufflepig.cancer_ontology import cancer_type_registry

        reg = cancer_type_registry()
        match = reg[reg["code"].astype(str) == str(code)]
        if match.empty:
            return False
        return str(match.iloc[0].get("fusion_driven") or "").strip().lower() == (
            "defining"
        )
    except Exception:
        return False


def _registry_parent_analysis_scope(report_scope):
    """Return an operational parent scope for a registry child label."""
    if not report_scope:
        return None
    try:
        from trufflepig.cancer_ontology import cancer_type_registry
        from .plot import resolve_cancer_type

        reg = cancer_type_registry()
        match = reg[reg["code"].astype(str) == str(report_scope)]
        if match.empty:
            return None
        parent = str(match.iloc[0].get("parent_code") or "").strip()
        if not parent:
            return None
        parent_code = resolve_cancer_type(parent)
        if not _has_operational_analysis_reference(parent_code):
            return None
        return parent_code
    except Exception:
        return None


@lru_cache(maxsize=1)
def _pan_cancer_expression_cohort_codes():
    """Codes with actual broad TPM columns in the analysis reference matrix."""
    try:
        from .reference import pan_cancer_expression

        df = pan_cancer_expression(technical_rna_normalize=True)
        return frozenset(
            str(col).removesuffix("_TPM")
            for col in df.columns
            if str(col).endswith("_TPM")
        )
    except Exception:
        return None


def _has_pan_cancer_expression_cohort(code):
    """True when ``code`` has a broad pan-cancer TPM column."""
    if not code:
        return False
    codes = _pan_cancer_expression_cohort_codes()
    if codes is None:
        # Preserve older control flow if the reference loader is unavailable;
        # the downstream lookup will surface the original failure.
        return True
    return str(code).strip().upper() in codes


def _analysis_input_cancer_type(cancer_type):
    """Return (composition_scope, report_scope) for an analyze label.

    Three kinds of labels are distinguished:

    1. Plain pan-cancer expression-cohort code (TCGA-backed cohort, e.g.
       COAD, SARC) — constrains analysis directly.
       Returns ``(code, None)``.
    2. Registry child of an expression cohort (e.g. SARC_SYN → SARC) —
       analysis runs against the parent cohort, the child stays as the
       report label.  Returns ``(parent_code, child_code)``.
    3. Registry-only or local-reference-only labels without a pan-cancer
       cohort column (e.g. NUTM, OS) — stay report-only while the RNA
       classifier runs unconstrained for cross-checking.
       Returns ``(None, code)``.
    """
    if not cancer_type:
        return None, None
    from .plot import resolve_cancer_type

    try:
        resolved = resolve_cancer_type(cancer_type)
    except ValueError:
        report_scope = _report_scope_cancer_type(cancer_type)
        if report_scope:
            return _registry_parent_analysis_scope(report_scope), report_scope
        raise

    # A typed own-cohort or member-union reference is operational in its own right. Check it before
    # following the registry parent; otherwise a concrete cohort such as COAD is incorrectly
    # promoted to its now-operational grouping parent CRC.
    if _has_pan_cancer_expression_cohort(resolved):
        return resolved, None

    parent = _registry_parent_analysis_scope(resolved)
    if parent:
        return parent, resolved
    if not _has_pan_cancer_expression_cohort(resolved):
        report_scope = _report_scope_cancer_type(resolved)
        if report_scope:
            parent = _registry_parent_analysis_scope(report_scope)
            if parent:
                return parent, report_scope
            if (
                _has_operational_analysis_reference(resolved)
                and not _is_defining_fusion_label(resolved)
            ):
                return resolved, None
            return None, report_scope
    if _is_registry_only_label(resolved):
        return None, resolved
    return resolved, None


def _is_registry_only_label(code):
    """True when the code has a registry row but no expression cohort.

    Used to distinguish curated-only labels (NUT carcinoma, ...) from
    full expression-cohort codes (COAD, SARC, OS, ...). Registry-only
    labels carry ``source_cohort = LITERATURE_CURATED`` (or empty).
    """
    if not code:
        return False
    try:
        from trufflepig.cancer_ontology import cancer_type_registry

        reg = cancer_type_registry()
        match = reg[reg["code"].astype(str) == str(code)]
        if match.empty:
            return False
        source = str(match.iloc[0].get("source_cohort") or "").strip().upper()
        return source in {"LITERATURE_CURATED", ""}
    except Exception:
        return False


def _infer_registry_report_scope_from_rna(df_expr, analysis):
    """Infer rare non-TCGA report scopes from strong RNA surrogates.

    This is intentionally hypothesis-level. It lets expression-only runs
    surface NUT carcinoma when NUTM1 is ectopically expressed, but the
    report still keeps the TCGA-backed RNA classifier as a cross-check.
    """
    from .rare_inference import infer_rare_cancer_report_scope_from_rna

    return infer_rare_cancer_report_scope_from_rna(df_expr, analysis)


def _analysis_constraints(
    cancer_type=None,
    sample_mode="auto",
    tumor_context="auto",
    site_hint=None,
    decomposition_templates=None,
    met_site=None,
    hla_types=None,
    alterations=None,
):
    constraints = {}
    if cancer_type:
        constraints["cancer_type"] = cancer_type
    if sample_mode and sample_mode != "auto":
        constraints["sample_mode"] = sample_mode
    if tumor_context and tumor_context != "auto":
        constraints["tumor_context"] = tumor_context
    if site_hint:
        constraints["site_hint"] = site_hint
    if decomposition_templates:
        constraints["decomposition_templates"] = list(decomposition_templates)
    if met_site:
        constraints["met_site"] = met_site
    if hla_types:
        from .hla import parse_hla_types

        parsed_hla = parse_hla_types(hla_types)
        if parsed_hla:
            constraints["hla_types"] = parsed_hla
    if alterations:
        from .alterations import split_alteration_inputs

        parsed_alterations = split_alteration_inputs(alterations)
        if parsed_alterations:
            constraints["alterations"] = parsed_alterations
    return constraints


def _summarize_sample_call(analysis, decomp_results, sample_mode):
    fit_quality = analysis.get("fit_quality", {})
    label_options = _candidate_label_options(analysis)
    best = decomp_results[0] if decomp_results else None
    hypothesis_options = []
    if decomp_results:
        hypothesis_options.append(decomp_results[0])
        if len(decomp_results) >= 2:
            second = decomp_results[1]
            score_ratio = (
                float(decomp_results[0].score / second.score)
                if second.score not in (0, None)
                else None
            )
            if fit_quality.get("label") in {"weak", "ambiguous"} or (
                score_ratio is not None and score_ratio < 1.2
            ):
                hypothesis_options.append(second)

    site_indeterminate = False
    context_indeterminate = False
    reported_context = None
    reported_site = None
    site_note = None
    site_primary_compatible = False

    if best is not None:
        if sample_mode == "pure":
            reported_context = "pure"
        elif sample_mode == "heme":
            reported_context = "heme"
        elif best.template == "solid_primary":
            reported_context = "primary"
        elif best.template.startswith("met_"):
            reported_context = "met"
        site_primary_compatible = _template_primary_compatible(
            best.template,
            analysis=analysis,
            cancer_code=analysis.get("cancer_type"),
        )

        if fit_quality.get("label") == "weak" and best.template.startswith("met_"):
            site_indeterminate = True
            context_indeterminate = True
            site_note = "Weak subtype fit prevents a reliable site-context call."
        elif best.template.startswith("met_"):
            if not _decomposition_met_site_is_supported(best, analysis=analysis):
                site_indeterminate = True
                context_indeterminate = True
                site_evidence = getattr(best, "site_evidence", None)
                if (
                    isinstance(site_evidence, dict)
                    and site_evidence.get("status") == "fit_only"
                ):
                    site_note = (
                        "Template fit is useful for decomposition, but metastatic-site "
                        "evidence is not strong enough to call a specific host site."
                    )
                else:
                    site_note = "Background evidence is not strong enough to support a specific background/site model."

        if not site_indeterminate:
            reported_site = _template_site_display(
                best.template,
                analysis=analysis,
                cancer_code=analysis.get("cancer_type"),
            )
            if site_primary_compatible and best.template.startswith("met_"):
                reported_context = "primary"
                site_note = (
                    "This background/site match is compatible with the cancer's native primary tissue, "
                    "so it is not treated as evidence of metastasis."
                )

    label_display = (
        " or ".join(label_options) if label_options else analysis.get("cancer_type")
    )
    hypothesis_display = [
        _hypothesis_display_label(
            row,
            primary_code=analysis.get("cancer_type"),
            analysis=analysis,
        )
        for row in hypothesis_options[:2]
    ]
    return {
        "label_options": label_options,
        "label_display": label_display,
        "reported_context": None if context_indeterminate else reported_context,
        "reported_site": reported_site,
        "site_indeterminate": site_indeterminate,
        "site_note": site_note,
        "site_primary_compatible": site_primary_compatible,
        "hypothesis_display": hypothesis_display,
    }


def _next_best_support_gap(candidate_trace):
    """Return (next_best_code, support_ratio) — ratio of top support_fraction_of_top
    over the second candidate's support_fraction_of_top, or None when unavailable.
    """
    if not candidate_trace or len(candidate_trace) < 2:
        return None, None
    top = candidate_trace[0]
    runner = candidate_trace[1]
    top_n = float(top.get("support_fraction_of_top", 0.0) or 0.0)
    runner_n = float(runner.get("support_fraction_of_top", 0.0) or 0.0)
    if runner_n <= 0:
        return runner.get("code"), None
    return runner.get("code"), top_n / runner_n


def _cancer_label(code, *, include_code=True):
    code_text = str(code or "").strip()
    if not code_text:
        return "this cancer type"
    display_name = cancer_code_display_name(
        code_text,
        fallback=CANCER_TYPE_NAMES.get(code_text, code_text),
    )
    if not display_name or display_name.lower() == code_text.lower():
        return code_text
    if include_code:
        return f"{code_text} ({display_name})"
    return display_name


def _cancer_type_context_line(cancer_type_context):
    report = cancer_type_context.label_for("report")
    reference = cancer_type_context.label_for("reference")
    expression = cancer_type_context.label_for("expression")
    if not report:
        return ""
    source_kind = str(
        getattr(cancer_type_context, "best_expression_source_kind", "") or ""
    )
    if source_kind == "deconvolved_tumor_reference":
        same_reference_label = "direct deconvolved tumor expression reference"
    elif source_kind == "observed_bulk_reference":
        same_reference_label = "direct observed expression reference"
    else:
        same_reference_label = "broad expression reference"
    if not cancer_type_context.uses_distinct_reference:
        return (
            f"- **Cancer label context**: {report} is used as both the report "
            f"label and the {same_reference_label}; no finer active subtype "
            "is being carried."
        )
    line = (
        f"- **Cancer label context**: fine/report label is {report}; "
        f"fallback expression reference is {reference}"
    )
    if expression and expression != reference:
        line += (
            f"; exact expression reference is {expression} and is preferred by "
            "modules that support fine-grained cohorts"
        )
    elif expression:
        line += f"; expression-context stages use {expression}"
    line += "."
    return line


def _candidate_trace_rank_and_row(candidate_trace, code):
    target = str(code or "").strip()
    if not target:
        return None, None
    for rank, row in enumerate(candidate_trace or [], start=1):
        if str(row.get("code") or "").strip() == target:
            return rank, row
    return None, None


def _report_label_code(analysis, call_summary=None):
    call_summary = call_summary or {}
    labels = call_summary.get("label_options") or []
    if labels:
        return str(labels[0] or "").strip()
    selected = (analysis.get("cancer_type_evidence") or {}).get("selected") or {}
    if isinstance(selected, dict):
        code = str(selected.get("cancer_type") or selected.get("code") or "").strip()
        if code:
            return code
    scope = _selected_report_scope_label(analysis)
    if scope:
        return scope
    return str(analysis.get("cancer_type") or "").strip()


def _score_text(value, *, digits=3):
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "—"


_CHANNEL_DISPLAY_LABELS = {
    "pan_cancer_signature_ranker": "Pan-cancer signature ranker",
    "fused_evidence": "Fused evidence model",
    "lineage_panel": "Lineage marker panel",
    "learned_expression_classifier": "Learned expression classifier",
    "composition_reference": "Coarse composition reference",
    "exact_expression_reference": "Exact expression reference",
    "deconvolved_tumor_reference": "Deconvolved tumor reference",
    "marker_program": "Marker-program coherence",
    "purity_attribution": "Purity/background attribution",
    "rare_fusion_anchor": "Rare marker/fusion anchor",
    "contrast_discriminator": "Local contrast discriminator",
}


def _channel_display_label(channel):
    return _CHANNEL_DISPLAY_LABELS.get(
        str(channel or "").strip(),
        str(channel or "evidence").replace("_", " ").title(),
    )


def _channel_role_label(role):
    return str(role or "").replace("_", " ")


def _cancer_type_vote_summary_markdown(analysis, call_summary=None, *, max_rows=8):
    evidence = analysis.get("cancer_type_evidence") or {}
    selected = evidence.get("selected") if isinstance(evidence, dict) else {}
    if not isinstance(selected, dict):
        selected = {}
    selected_code = _report_label_code(analysis, call_summary)
    if not selected_code:
        return ""

    selected_by = str(selected.get("selected_by") or "").strip()
    basis_label = _selected_report_scope_basis_label(analysis)
    lines = ["### Cancer-Type Evidence Votes\n"]
    if selected_by:
        lines.append(
            f"- **Selection driver**: {basis_label} (`{selected_by}`). This is "
            "provenance for the strongest selecting path, not the complete evidence model."
        )
    else:
        lines.append(
            "- **Selection driver**: no single selecting channel was recorded; "
            "the table below summarizes available support for the active report label."
        )

    candidate_trace = analysis.get("candidate_trace") or []
    rank, trace_row = _candidate_trace_rank_and_row(candidate_trace, selected_code)
    if trace_row:
        rank_clause = f"rank {rank}" if rank else "rank unavailable"
        lines.append(
            f"- **Pan-cancer ranker position for report label**: {rank_clause}; "
            f"normalized support {_score_text(trace_row.get('support_fraction_of_top'))}, "
            f"signature {_score_text(trace_row.get('signature_score'))}, "
            f"lineage concordance {_score_text(trace_row.get('lineage_concordance'))}."
        )

    rows_by_key = {}
    for row in selected.get("evidence_channels") or []:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "").strip()
        channel = str(row.get("channel") or "").strip()
        if not channel:
            continue
        channel_code = str(row.get("code") or row.get("candidate_code") or "").strip()
        if role.startswith("hierarchical_") and channel_code != selected_code:
            continue
        try:
            support_value = float(row.get("support"))
        except (TypeError, ValueError):
            continue
        if support_value <= 0:
            continue
        key = (channel, role)
        previous = rows_by_key.get(key)
        if previous is None or support_value > float(previous.get("support") or 0.0):
            rows_by_key[key] = row

    rows = sorted(
        rows_by_key.values(),
        key=lambda row: (
            not bool(row.get("selects_report_label")),
            -float(row.get("support") or 0.0),
            str(row.get("channel") or ""),
            str(row.get("role") or ""),
        ),
    )
    if rows:
        from .reporting import md_table_cell

        lines.append("")
        lines.append("| Signal | Role for active label | Status | Score | Details |")
        lines.append("|---|---|---|---:|---|")
        for row in rows[:max_rows]:
            status = str(row.get("status") or "").replace("_", " ")
            if row.get("selects_report_label"):
                status = "selected report-label vote"
            # Escape the same way the decision-trace table does: a free-text
            # channel rationale can carry `|`/newlines that would break the row.
            details = md_table_cell(_decision_channel_detail_summary(row.get("details") or {}))
            lines.append(
                f"| {_channel_display_label(row.get('channel'))} | "
                f"{_channel_role_label(row.get('role'))} | "
                f"{status or 'supporting/context'} | "
                f"{_score_text(row.get('support'))} | {details or '—'} |"
            )
    lines.append("")
    lines.append(
        "Retained alternatives are isolated in the differential section below; "
        "subsequent interpretation uses the active report label unless explicitly noted."
    )
    return "\n".join(lines)


def _format_candidate_trace_alternatives(candidate_trace, selected_code, *, limit=3):
    selected_code = str(selected_code or "").strip()
    if not candidate_trace:
        return ""
    ranked = sorted(
        candidate_trace,
        key=lambda row: (
            float(row.get("support_fraction_of_top") or 0.0),
            float(row.get("support_geomean") or 0.0),
        ),
        reverse=True,
    )
    top_support = float(ranked[0].get("support_fraction_of_top") or 0.0)
    chunks = []
    for rank, row in enumerate(ranked, start=1):
        code = str(row.get("code") or "").strip()
        if not code or code == selected_code:
            continue
        norm = row.get("support_fraction_of_top")
        if isinstance(norm, (int, float)) and float(norm) < 0.75 and len(chunks) >= 1:
            continue
        ratio_clause = ""
        if isinstance(norm, (int, float)) and top_support > 0:
            ratio_clause = f", {float(norm) / top_support:.2f}x top support"
        signature = row.get("signature_score")
        signature_clause = (
            f", signature {float(signature):.2f}"
            if isinstance(signature, (int, float))
            else ""
        )
        chunks.append(
            f"{_cancer_label(code)} (rank {rank}{ratio_clause}{signature_clause})"
        )
        if len(chunks) >= limit:
            break
    return "; ".join(chunks)


def _format_decomposition_alternatives(decomp_results, selected_code, *, limit=3):
    if not decomp_results:
        return ""
    selected_code = str(selected_code or "").strip()
    best = decomp_results[0]
    chunks = []
    for row in decomp_results[1 : limit + 1]:
        same_hypothesis = (
            str(getattr(row, "cancer_type", "") or "").strip()
            == str(getattr(best, "cancer_type", "") or "").strip()
            and str(getattr(row, "template", "") or "").strip()
            == str(getattr(best, "template", "") or "").strip()
        )
        if same_hypothesis:
            continue
        label = _hypothesis_display_label(
            row,
            primary_code=selected_code,
            analysis={"cancer_type": selected_code},
        )
        score = getattr(row, "score", None)
        score_clause = (
            f", fit score {float(score):.3f}" if isinstance(score, (int, float)) else ""
        )
        chunks.append(
            f"{_hypothesis_label(label, primary_code=selected_code)}{score_clause}"
        )
        if len(chunks) >= limit:
            break
    return "; ".join(chunks)


def _retained_cancer_type_differential_markdown(
    analysis,
    call_summary=None,
    decomp_results=None,
):
    call_summary = call_summary or {}
    decomp_results = decomp_results or []
    selected_code = _report_label_code(analysis, call_summary)
    if not selected_code:
        return ""

    label_options = [
        str(label or "").strip()
        for label in (call_summary.get("label_options") or [])
        if str(label or "").strip()
    ]
    retained_labels = [label for label in label_options[1:] if label != selected_code]
    candidate_trace = analysis.get("candidate_trace") or []
    candidate_alts = _format_candidate_trace_alternatives(
        candidate_trace,
        selected_code,
    )
    decomp_alts = _format_decomposition_alternatives(
        decomp_results,
        selected_code,
    )
    rare_marker_hypotheses = [
        finding
        for finding in (analysis.get("rare_marker_hypotheses") or [])
        if str(finding.get("cancer_type") or "").strip() != selected_code
    ]
    if not (retained_labels or candidate_alts or decomp_alts or rare_marker_hypotheses):
        return ""

    lines = ["### Cancer-Type Differential / Retained Alternatives\n"]
    lines.append(
        f"The active report label is **{_cancer_label(selected_code)}**. "
        "The entries below are retained for audit and differential diagnosis; "
        "they do not control downstream expression references, decomposition, "
        "biomarker panels, or therapy curation unless a later section explicitly "
        "promotes them."
    )
    if retained_labels:
        lines.append(
            "- **Retained report-label differential**: "
            + "; ".join(_cancer_label(label) for label in retained_labels)
            + "."
        )
    if candidate_alts:
        rank, selected_row = _candidate_trace_rank_and_row(
            candidate_trace,
            selected_code,
        )
        selected_clause = ""
        if selected_row and rank:
            selected_clause = (
                f" Active label rank {rank}, normalized support "
                f"{_score_text(selected_row.get('support_fraction_of_top'))}."
            )
        lines.append(
            f"- **Pan-cancer ranker differential**: {candidate_alts}."
            + selected_clause
        )
    if decomp_alts:
        best_label = (
            _hypothesis_display_label(
                decomp_results[0],
                primary_code=selected_code,
                analysis=analysis,
            )
            if decomp_results
            else _cancer_label(selected_code)
        )
        lines.append(
            "- **Decomposition differential**: selected report-compatible fit is "
            f"{_hypothesis_label(best_label, primary_code=selected_code, analysis=analysis)}; "
            f"retained raw/nearby fits include {decomp_alts}."
        )
    if rare_marker_hypotheses:
        chunks = []
        for finding in rare_marker_hypotheses[:3]:
            label = _cancer_label(str(finding.get("cancer_type") or "rare cancer"))
            surrogate = str(finding.get("surrogate") or "marker").strip()
            tpm = finding.get("surrogate_tpm")
            tpm_clause = f" {tpm:g} TPM" if isinstance(tpm, (int, float)) else ""
            chunks.append(f"{label} ({surrogate}{tpm_clause})")
        lines.append(
            "- **Rare-marker differential prompts**: "
            + "; ".join(chunks)
            + "."
        )
    return "\n".join(lines)


def _tumor_type_sanity_markdown(analysis, *, max_rows: int = 6) -> str:
    sanity = analysis.get("tumor_type_sanity") or {}
    if not sanity:
        return ""
    code = str(sanity.get("code") or "").strip()
    name = str(sanity.get("name") or "").strip()
    family = str(sanity.get("family") or "").strip()
    parent = str(sanity.get("parent_code") or "").strip()
    expression_ref = sanity.get("expression_reference") or {}
    high_rows = sanity.get("expected_high") or []
    high_detected = sanity.get("expected_high_detected") or []
    low_rows = sanity.get("expected_low") or []
    low_present = sanity.get("expected_low_present") or []
    high_floor = float(sanity.get("high_tpm_floor") or 0.0)
    low_ceiling = float(sanity.get("low_tpm_ceiling") or 0.0)

    status_label = {
        "consistent": "consistent",
        "partial": "partial",
        "weak": "weak",
        "mixed": "mixed lineage/background signal",
        "not_evaluable": "not evaluable",
    }.get(str(sanity.get("status") or ""), str(sanity.get("status") or ""))

    label = _cancer_label(code) if code else "selected cancer type"
    placement_bits = [label]
    if parent:
        placement_bits.append(f"parent {parent}")
    if family:
        placement_bits.append(f"family {family}")

    lines = ["### Tumor-Type Marker Sanity Check\n"]
    lines.append(f"- **Ontology placement**: {'; '.join(placement_bits)}.")
    ref_code = str(expression_ref.get("code") or "").strip()
    ref_source = str(expression_ref.get("source") or "").strip()
    ref_kind = str(expression_ref.get("kind") or "").strip().replace("_", " ")
    ref_direct = bool(expression_ref.get("direct"))
    ref_reason = str(expression_ref.get("reason") or "").strip()
    if ref_code:
        reason = f"; fallback reason: {ref_reason}" if ref_reason else ""
        direct_word = "direct" if ref_direct else "compatible fallback"
        source_clause = f" from {ref_source}" if ref_source else ""
        lines.append(
            f"- **Expression reference**: {ref_code}{source_clause} "
            f"({direct_word}; {ref_kind or 'expression reference'}{reason})."
        )
    if name and code and name.lower() != code.lower():
        lines.append(f"- **Report label name**: {name}.")
    lines.append(
        f"- **Marker summary**: {len(high_detected)}/{len(high_rows)} expected high "
        f"markers are at least {high_floor:g} TPM; {len(low_present)}/{len(low_rows)} "
        f"expected low markers are above {low_ceiling:g} TPM. Status: {status_label}."
    )
    selected_scope = (analysis.get("cancer_type_evidence") or {}).get("selected") or {}
    selected_by = str(selected_scope.get("selected_by") or "").strip()
    if (
        selected_by == "coarse_composition_reference"
        and str(sanity.get("status") or "") in {"weak", "partial"}
    ):
        rho = selected_scope.get("coarse_reference_rho")
        hits = selected_scope.get("coarse_reference_type_specific_hit_count")
        evidence_bits = []
        if isinstance(rho, (int, float)):
            evidence_bits.append(f"top cancer-reference rho {float(rho):.2f}")
        if isinstance(hits, int) and hits > 0:
            evidence_bits.append(f"{hits} type-specific tumor-up hits")
        if selected_scope.get("coarse_reference_tissue_tiebreak_applied"):
            tissue = str(selected_scope.get("coarse_reference_primary_tissue") or "").strip()
            tissue_score = selected_scope.get("coarse_reference_primary_tissue_score")
            if tissue and isinstance(tissue_score, (int, float)):
                tissue_label = tissue.replace("_", " ")
                evidence_bits.append(
                    f"{tissue_label} normal-tissue tie-break rho {float(tissue_score):.2f}"
                )
        evidence_clause = "; ".join(evidence_bits)
        if evidence_clause:
            evidence_clause = f" ({evidence_clause})"
        lines.append(
            "- **Interpretation**: this curated marker panel is a sanity check, "
            "not the selecting evidence. The report label came from independent "
            f"composition/reference evidence{evidence_clause}, so low conventional "
            "lineage markers should be read as subtype/differentiation caveat."
        )

    if high_rows:
        lines.append("")
        lines.append("| Expected high marker | TPM | Source |")
        lines.append("|---|---:|---|")
        for row in high_rows[:max_rows]:
            gene = str(row.get("gene") or "").strip()
            source = str(row.get("source") or "").strip()
            inherited = str(row.get("inherited_from") or "").strip()
            if inherited:
                source = f"{source}; inherited from {inherited}"
            lines.append(
                f"| {gene} | {render_tpm(row.get('tpm'))} | {source or 'marker panel'} |"
            )
    if low_present:
        lines.append("")
        lines.append(
            "Expected-low genes are not treated as diagnostic by themselves; high values "
            "can reflect immune, stromal, or mixed-lineage background that should be checked."
        )
        lines.append("")
        lines.append("| Expected low marker above threshold | TPM | Source |")
        lines.append("|---|---:|---|")
        for row in low_present[:max_rows]:
            gene = str(row.get("gene") or "").strip()
            source = str(row.get("source") or "").strip()
            lines.append(
                f"| {gene} | {render_tpm(row.get('tpm'))} | {source or 'contrast panel'} |"
            )
    return "\n".join(lines)


def _hypothesis_label(text, *, primary_code=None, analysis=None):
    label = str(text or "").strip()
    if not label:
        return ""
    if " pattern" in label or "-like " in label:
        return label
    if " / " not in label:
        return _cancer_label(label)
    cancer_code, template = [part.strip() for part in label.split(" / ", 1)]
    cancer_label = _cancer_label(cancer_code)
    template_label = _template_site_display(
        template,
        analysis=analysis,
        cancer_code=primary_code or cancer_code,
    )
    if template_label == "primary-site-compatible context":
        if primary_code and cancer_code == primary_code:
            return f"{cancer_label} primary-site pattern"
        return f"{cancer_label}-like primary-site pattern"
    if template_label.endswith("context"):
        if primary_code and cancer_code == primary_code:
            return f"{cancer_label} {template_label}"
        return f"{cancer_label}-like {template_label}"
    if primary_code and cancer_code == primary_code:
        return f"{cancer_label} {template_label} pattern"
    return f"{cancer_label}-like {template_label} pattern"


def _strip_terminal_punctuation(text):
    if text is None:
        return ""
    return str(text).strip().rstrip(".;:! ")


def _report_expression_source_label(value):
    source = str(value or "expression").strip().lower().replace("-", "_")
    labels = {
        "tumor_inferred": "tumor-source/context",
        "tumor_source": "tumor-source",
        "tumor_attributed": "tumor-source",
        "bulk": "bulk TPM",
        "mixed": "mixed source",
        "unavailable": "unavailable",
        "expression": "expression",
    }
    return labels.get(source, source.replace("_", " "))


def _format_tissue_driver_genes(driver_rows, *, top_n=5):
    drivers = list(driver_rows or [])[:top_n]
    if not drivers:
        return "—"
    parts = []
    for row in drivers:
        gene = str(row.get("gene") or "").strip()
        if not gene:
            continue
        try:
            tpm = float(row.get("sample_tpm") or 0.0)
        except Exception:
            tpm = 0.0
        tpm_text = f"{tpm:.0f}" if tpm >= 10 else f"{tpm:.1f}"
        parts.append(f"{gene} ({tpm_text} TPM)")
    return ", ".join(parts) if parts else "—"


def _target_reliability_series(df, *, category=None):
    import pandas as pd

    if df is None:
        return pd.Series(dtype=object)
    if len(df) == 0:
        return pd.Series(dtype=object, index=df.index)
    from .common import ranges_records

    return pd.Series(
        [target_reliability_status(row, category=category) for row in ranges_records(df)],
        index=df.index,
    )


def _composition_highlights(decomp_result, *, top_n=3):
    if decomp_result is None:
        return []
    fractions = dict(getattr(decomp_result, "fractions", {}) or {})
    non_tumor = sorted(
        (
            (comp, float(frac))
            for comp, frac in fractions.items()
            if comp != "tumor" and float(frac) >= 0.03
        ),
        key=lambda kv: -kv[1],
    )
    highlights = []
    for comp, frac in non_tumor[:top_n]:
        label = (
            str(comp).replace("matched_normal_", "matched-normal ").replace("_", " ")
        )
        highlights.append(f"{label} {frac:.0%}")
    return highlights


def _integrated_evidence_bullets(analysis, decomp_results=None):
    candidate_trace = analysis.get("candidate_trace", [])
    fit_quality = analysis.get("fit_quality", {})
    cancer_code = analysis.get("cancer_type")
    cancer_type_context = cancer_type_context_from_analysis(analysis)
    reference_cancer_code = cancer_type_context.code_for("reference") or cancer_code
    purity = analysis.get("purity") or {}
    sample_context = analysis.get("sample_context")
    hvt = analysis.get("healthy_vs_tumor")
    call_summary = analysis.get("call_summary") or _summarize_sample_call(
        analysis,
        decomp_results or [],
        sample_mode=analysis.get("sample_mode", "auto"),
    )
    best_decomp = decomp_results[0] if decomp_results else None
    bullets = []
    inferred_site = analysis.get("inferred_site_context") or {}
    if inferred_site:
        tissue = str(inferred_site.get("tissue") or inferred_site.get("site") or "")
        primary = str(inferred_site.get("primary_tissue") or "")
        primary_clause = (
            f", above the expected primary-tissue program {primary.replace('_', ' ')} "
            f"({float(inferred_site.get('primary_tissue_score') or 0.0):.2f})"
            if primary
            else ""
        )
        bullets.append(
            "- **Inferred host site**: "
            f"{tissue.replace('_', ' ')} background is strong "
            f"({float(inferred_site.get('score') or 0.0):.2f}){primary_clause}; "
            f"treating {str(inferred_site.get('site') or '').replace('_', ' ')} as a "
            "likely metastatic host context for decomposition and target-background modeling."
        )
    call_rescue = analysis.get("cancer_call_rescue") or {}
    if call_rescue:
        rescue_kind = str(call_rescue.get("kind") or "")
        label = (
            "QC/call pitfall"
            if rescue_kind == "low_purity_prad_stromal_context"
            else "Cancer-call rescue"
        )
        marker_clause = ""
        if rescue_kind == "low_purity_prad_stromal_context":
            markers = call_rescue.get("prostate_marker_tpm") or {}
            if markers:
                top_markers = sorted(
                    markers.items(),
                    key=lambda item: (-float(item[1]), item[0]),
                )[:5]
                marker_clause = "; prostate markers " + ", ".join(
                    f"{gene} {value:g} TPM" for gene, value in top_markers
                )
        bullets.append(
            f"- **{label}**: "
            + str(call_rescue.get("message") or "").rstrip(".")
            + marker_clause
            + ". "
            + str(call_rescue.get("interpretation") or "")
        )

    mmr_channels = [
        channel
        for channel in (
            ((analysis.get("cancer_type_evidence") or {}).get("staged_evidence_graph") or {}).get(
                "channels"
            )
            or []
        )
        if isinstance(channel, dict)
        and channel.get("role") == "hierarchical_mismatch_repair_vote"
    ]
    selected_mmr = select_mismatch_repair_channel_for_report(
        mmr_channels,
        cancer_code,
    )
    if selected_mmr:
        details = selected_mmr.get("details") or {}
        mmr = details.get("mismatch_repair") or {}
        if isinstance(mmr, dict) and mmr:
            p_msi = mmr.get("msi_probability")
            top_state = selected_mmr.get("code") or ""
            context = mmr.get("context_group") or ""
            if isinstance(p_msi, (int, float)):
                bullets.append(
                    "- **Mismatch-repair RNA context**: "
                    f"{context + ' ' if context else ''}MMR ensemble favors "
                    f"{top_state or 'an MMR state'} with MSI-like probability "
                    f"{p_msi:.2f}. This is expression context only; confirm "
                    "MSI/MMR status with MSI-PCR, MMR IHC, or validated "
                    "clinical sequencing before using it for immunotherapy "
                    "eligibility."
                )

    if candidate_trace:
        best = candidate_trace[0]
        best_code = str(best.get("code") or "").strip()
        runner = candidate_trace[1] if len(candidate_trace) > 1 else None
        distinct_reference_used = cancer_type_context.uses_distinct_reference
        supplied_discordant = (
            analysis.get("cancer_type_source") == "user-specified"
            and best_code != str(cancer_code).strip()
        )
        evidence_selected_discordant = (
            analysis.get("cancer_type_source") == "auto-detected"
            and best_code != str(cancer_code).strip()
            and _selected_report_scope_label(analysis) == str(cancer_code or "").strip()
        )
        if evidence_selected_discordant:
            sentence = (
                "- **RNA classifier line**: the pan-cancer signature ranker "
                "does not independently set the report label; integrated evidence selected "
                f"{_cancer_label(cancer_code)} as the active report label"
            )
        elif distinct_reference_used and best_code == str(reference_cancer_code).strip():
            sentence = (
                f"- **RNA reference line**: {_cancer_label(best['code'])} is the "
                "active fallback expression/reference context for cohort-normalized "
                f"downstream analyses; {_cancer_label(cancer_code)} remains the report label"
            )
        elif distinct_reference_used:
            sentence = (
                f"- **RNA classifier line**: {_cancer_label(best['code'])} is the "
                "leading pan-cancer signature-ranker candidate, but "
                f"{_cancer_label(reference_cancer_code)} is the active fallback "
                "expression/reference context; "
                f"{_cancer_label(cancer_code)} remains the report label"
            )
        elif supplied_discordant:
            sentence = (
                f"- **RNA classifier line**: {_cancer_label(best['code'])} is the "
                "leading pan-cancer signature-ranker candidate, but the supplied "
                f"{_cancer_label(cancer_code)} label remains the report label"
            )
        else:
            if best.get("support_override"):
                sentence = (
                    f"- **Classifier line**: {_cancer_label(best['code'])} is the "
                    "leading label after pitfall-aware rescue"
                )
            else:
                sentence = (
                    f"- **Classifier line**: {_cancer_label(best['code'])} is the leading label"
                )
        if supplied_discordant and runner is not None:
            runner_norm = float(runner.get("support_fraction_of_top", 0.0) or 0.0)
            if str(runner.get("code") or "").strip() == str(cancer_code).strip():
                if runner_norm > 0:
                    sentence += (
                        f"; supplied label is RNA rank 2 "
                        f"({_cancer_label(best['code'])}:{_cancer_label(cancer_code)} "
                        f"support ratio {1.0 / runner_norm:.1f}x)"
                    )
                else:
                    sentence += "; supplied label is the next RNA candidate"
            elif runner_norm > 0:
                sentence += (
                    f", with {_cancer_label(runner['code'])} next by "
                    f"{1.0 / runner_norm:.1f}x normalized support"
                )
            else:
                sentence += f", with {_cancer_label(runner['code'])} the next candidate"
        elif runner is not None:
            runner_norm = float(runner.get("support_fraction_of_top", 0.0) or 0.0)
            runner_code = str(runner.get("code") or "").strip()
            if (
                evidence_selected_discordant
                and runner_code == str(cancer_code or "").strip()
            ):
                if runner_norm > 0:
                    sentence += (
                        f"; active report label is RNA rank 2 "
                        f"({runner_norm:.2f}x of top ranker support)"
                    )
                else:
                    sentence += "; selected report label is the next RNA candidate"
            elif runner_norm > 0:
                sentence += (
                    f", ahead of {_cancer_label(runner['code'])} "
                    f"by {1.0 / runner_norm:.1f}x on normalized support"
                )
            else:
                sentence += f", with {_cancer_label(runner['code'])} the next candidate"
        if best.get("signature_score") is not None:
            sentence += f"; signature {best['signature_score']:.2f}"
        if best.get("lineage_concordance") is not None:
            sentence += f", lineage concordance {best['lineage_concordance']:.2f}"
        sentence += "."
        bullets.append(sentence)

    if hvt is not None and getattr(hvt, "top_tcga_cohorts", None):
        coarse = str(hvt.top_tcga_cohorts[0][0]).removesuffix("_TPM")
        coarse_label = _cancer_label(coarse)
        distinct_reference_used = cancer_type_context.uses_distinct_reference
        if distinct_reference_used and coarse == reference_cancer_code:
            basis_label = _selected_report_scope_basis_label(analysis)
            bullets.append(
                f"- **Coarse prior**: tissue composition screen is `{hvt.cancer_hint}` and its top "
                f"cancer-reference match, {coarse_label}, agrees with the broad "
                f"reference context; {_cancer_label(cancer_code)} remains the "
                f"report label from {basis_label}."
            )
        elif coarse == cancer_code:
            bullets.append(
                f"- **Coarse prior**: tissue composition screen is `{hvt.cancer_hint}` and its top cancer-reference match, "
                f"{coarse_label}, agrees with the working call."
            )
        elif coarse in call_summary.get("label_options", []):
            bullets.append(
                f"- **Coarse prior**: tissue composition screen is `{hvt.cancer_hint}` and keeps {coarse_label} alive "
                "as one of the same competing labels carried downstream."
            )
        else:
            rare_inference = analysis.get("rare_report_scope_inference") or {}
            fusion_inference = analysis.get("fusion_report_scope_inference") or {}
            if rare_inference:
                bullets.append(
                    f"- **Coarse prior**: tissue composition screen is `{hvt.cancer_hint}`, and its top cancer-reference match "
                    f"is {coarse_label}; the {_cancer_label(cancer_code)} label is an RNA-surrogate "
                    "rare-cancer hypothesis, so expression references are context rather than diagnosis."
                )
            elif fusion_inference:
                bullets.append(
                    f"- **Coarse prior**: tissue composition screen is `{hvt.cancer_hint}`, and its top cancer-reference match "
                    f"is {coarse_label}; the {_cancer_label(cancer_code)} label is fusion-supported, "
                    "so expression references are context rather than diagnosis."
                )
            elif distinct_reference_used:
                bullets.append(
                    f"- **Coarse prior**: tissue composition screen is `{hvt.cancer_hint}`, and its top cancer-reference match "
                    f"is {coarse_label}; this is broad expression context only. "
                    f"Keep {_cancer_label(cancer_code)} as the report label unless "
                    "orthogonal clinical/pathology evidence changes it."
                )
            else:
                bullets.append(
                    f"- **Coarse prior**: tissue composition screen is `{hvt.cancer_hint}`, but its top cancer-reference match "
                    f"is {coarse_label}; the final {_cancer_label(cancer_code)} call depends on the "
                    "later classifier/decomposition evidence rather than the coarse screen alone."
                )

    if best_decomp is not None:
        comp_bits = _composition_highlights(best_decomp)
        sentence = (
            f"- **Decomposition line**: best fit is "
            f"{_hypothesis_display_label(best_decomp, primary_code=cancer_code, analysis=analysis)}"
        )
        if best_decomp.cancer_type == cancer_code:
            sentence += ", consistent with the active report label"
        elif (
            cancer_type_context.uses_distinct_reference
            and best_decomp.cancer_type == reference_cancer_code
        ):
            sentence += (
                ", consistent with the broad reference context rather than a "
                "separate refined subtype call"
            )
        else:
            sentence += (
                f", diverging from the report-label {_cancer_label(cancer_code)} call"
            )
        if call_summary.get("site_primary_compatible"):
            sentence += "; this host context is compatible with the tumor's native primary tissue and is not specific for metastasis"
        if comp_bits:
            sentence += "; dominant non-tumor components are " + ", ".join(comp_bits)
        if getattr(best_decomp, "warnings", None):
            warning = next(
                (
                    str(item)
                    for item in best_decomp.warnings
                    if "raw best fit" not in str(item).lower()
                ),
                "",
            )
            if warning:
                sentence += f"; key warning: {_strip_terminal_punctuation(warning)}"
            elif any(
                "raw best fit" in str(item).lower()
                for item in best_decomp.warnings
            ):
                sentence += (
                    "; retained raw decomposition alternatives are summarized "
                    "in the cancer-type differential"
                )
        sentence += "."
        bullets.append(sentence)

    active_biology = []
    for finding in (analysis.get("pathway_activity_inferences") or [])[:3]:
        label = str(finding.get("label") or finding.get("axis") or "Pathway").strip()
        fold = finding.get("up_geomean_fold")
        score_name = str(finding.get("score_name") or "RNA score").strip()
        support = ", ".join(finding.get("support_genes") or [])
        source_labels = [
            str(source.get("label") or "").strip()
            for source in (finding.get("candidate_sources") or [])[:3]
            if str(source.get("label") or "").strip()
        ]
        unresolved = ", ".join(finding.get("unresolved_sources") or [])
        parts = []
        if isinstance(fold, (int, float)):
            parts.append(f"{score_name} {fold:.2f}x context")
        if support:
            parts.append(f"support genes {support}")
        if source_labels:
            parts.append("candidate sources " + "; ".join(source_labels))
        elif unresolved:
            parts.append("possible sources to test: " + unresolved)
        active_biology.append(f"{label}: active ({'; '.join(parts)})")
    for finding in (analysis.get("fusion_expression_effects") or [])[:3]:
        status = str(finding.get("status") or "")
        if status not in {"active", "partial", "not_evident"}:
            continue
        genes = ", ".join(finding.get("observed_genes") or [])
        if not genes:
            genes = "expected downstream markers not clearly active"
        source = _report_expression_source_label(finding.get("expression_source"))
        active_biology.append(
            f"{finding.get('label')}: {status} downstream program ({genes}; {source})"
        )
    for finding in (analysis.get("fusion_expression_hypotheses") or [])[:2]:
        genes = ", ".join(finding.get("observed_genes") or [])
        source = _report_expression_source_label(finding.get("expression_source"))
        active_biology.append(
            f"{finding.get('label')}: RNA-only fusion-effect testing prompt "
            f"({genes}; {source})"
        )
    for finding in (analysis.get("mutation_expression_hypotheses") or [])[:3]:
        high = ", ".join(finding.get("observed_up_genes") or [])
        low = ", ".join(finding.get("observed_low_genes") or [])
        support = "; ".join(
            part
            for part in [
                f"high {high}" if high else "",
                f"low {low}" if low else "",
            ]
            if part
        )
        source = _report_expression_source_label(finding.get("expression_source"))
        active_biology.append(
            f"{finding.get('label')}: compatible with {finding.get('alteration')} "
            f"({support}; {source})"
        )
    if active_biology:
        bullets.append(
            "- **Active biology / alteration-effect checks**: "
            + "; ".join(dict.fromkeys(active_biology))
            + ". These widen the hypothesis set and should be confirmed with orthogonal assays."
        )

    uncertainty = []
    fit_label = fit_quality.get("label")
    fit_message = _strip_terminal_punctuation(fit_quality.get("message"))
    if fit_label in {"weak", "ambiguous"} and fit_message:
        uncertainty.append(fit_message)
    try:
        overall = float(purity.get("overall_estimate"))
        lower = float(purity.get("overall_lower"))
        upper = float(purity.get("overall_upper"))
        if overall < 0.20:
            uncertainty.append("low purity amplifies residual host/background signal")
        if upper - lower >= 0.25:
            uncertainty.append(f"purity remains broad at {lower:.0%}-{upper:.0%}")
    except (TypeError, ValueError):
        pass
    integration = purity.get("components", {}).get("integration", {})
    if integration.get("signature_deprioritized"):
        uncertainty.append(
            "tumor-specific signature evidence was downweighted relative to lineage/background evidence"
        )
    if sample_context is not None and getattr(sample_context, "is_degraded", False):
        uncertainty.append(
            "RNA degradation lowers confidence for long-transcript negatives"
        )
    if call_summary.get("site_indeterminate"):
        note = _strip_terminal_punctuation(
            call_summary.get("site_note")
            or "site/template assignment remains indeterminate"
        )
        if note:
            uncertainty.append(note)
    if (
        best_decomp is not None
        and getattr(best_decomp, "matched_normal_fraction", 0.0) >= 0.15
    ):
        uncertainty.append(
            f"benign parent-tissue admixture is large ({best_decomp.matched_normal_fraction:.0%})"
        )
    if uncertainty:
        bullets.append(
            "- **Main uncertainties**: " + "; ".join(dict.fromkeys(uncertainty)) + "."
        )

    return bullets


def _strip_markdown_wrapper(md: str) -> str:
    """Drop top-level title/comment wrappers and trailing cross-links."""
    lines = md.splitlines()
    while lines and (
        not lines[0].strip() or lines[0].startswith("# ") or lines[0].startswith("<!--")
    ):
        lines.pop(0)
    while lines and (
        not lines[-1].strip()
        or lines[-1].startswith("*See also:")
        or lines[-1].startswith("*Full detail:")
    ):
        lines.pop()
    return "\n".join(lines).strip()


def _demote_markdown_headings(md: str, levels: int = 1) -> str:
    if levels <= 0:
        return md
    prefix = "#" * levels
    out = []
    for line in md.splitlines():
        if line.startswith("#"):
            out.append(prefix + line)
        else:
            out.append(line)
    return "\n".join(out)


def _extract_markdown_section(md: str, heading: str) -> str:
    """Return one H2 section from ``md`` including its heading."""
    lines = md.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == heading:
            start = i
            break
    if start is None:
        return ""
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break
    return "\n".join(lines[start:end]).strip()


def _fusion_evidence_markdown(analysis, *, heading: str = "## Fusion evidence") -> str:
    records = analysis.get("fusion_records") or []
    findings = analysis.get("fusion_findings") or []
    rare_inference = analysis.get("rare_report_scope_inference") or {}
    fusion_effects = analysis.get("fusion_expression_effects") or []
    fusion_hypotheses = analysis.get("fusion_expression_hypotheses") or []
    fusion_inputs_supplied = bool(analysis.get("fusion_inputs_supplied"))
    if (
        not records
        and not findings
        and not rare_inference
        and not fusion_effects
        and not fusion_hypotheses
        and not fusion_inputs_supplied
    ):
        return ""

    lines = [heading, ""]
    if findings:
        lines.append(
            "Fusion calls were matched against the curated direct-fusion rule table. "
            "Rule columns are oriented as gene_a = expected 5-prime partner and "
            "gene_b = expected 3-prime partner; matching tolerates reversed or "
            "unoriented caller output when the rule allows it.\n"
        )
        lines.append(
            "| Reported fusion | Expected 5-prime/3-prime rule | Finding | Report scope? | Confirmatory step | Caveat |"
        )
        lines.append("|---|---|---|---|---|---|")
        for finding in findings[:10]:
            fusion = finding.get("fusion") or {}
            reported = str(fusion.get("pair") or "").strip() or "—"
            expected = str(finding.get("expected_pair") or "").strip() or "—"
            label = str(finding.get("label") or finding.get("cancer_type") or "—")
            scope = (
                str(finding.get("cancer_type") or "yes")
                if finding.get("promote_report_scope")
                else "no"
            )
            confirm = str(finding.get("confirmatory_tests") or "—")
            caveat = str(finding.get("caveat") or finding.get("orientation_note") or "—")
            lines.append(
                f"| {reported} | {expected} | {label} | {scope} | {confirm} | {caveat} |"
            )
        lines.append("")
    elif records:
        lines.append(
            f"{len(records)} fusion call(s) were parsed, but none matched a curated "
            "rare-cancer report-scope rule.\n"
        )
    elif fusion_inputs_supplied:
        paths = analysis.get("fusion_input_paths") or []
        count_text = f"{len(paths)} file(s)" if paths else "file(s)"
        lines.append(
            f"Fusion input {count_text} were supplied, but no usable fusion calls "
            "were parsed. Treat direct-fusion evidence as unavailable until the "
            "file format/content is reviewed.\n"
        )
        if rare_inference:
            surrogate = str(rare_inference.get("surrogate") or "RNA marker").strip()
            confirm = str(
                rare_inference.get("confirmatory_tests") or "fusion testing"
            ).strip()
            lines.append(
                f"Because {surrogate} RNA supports a rare-cancer hypothesis, "
                f"confirm with {confirm} or a usable fusion/cytogenetic callset.\n"
            )
    elif rare_inference:
        surrogate = str(rare_inference.get("surrogate") or "RNA marker").strip()
        confirm = str(
            rare_inference.get("confirmatory_tests") or "fusion testing"
        ).strip()
        lines.append(
            f"No fusion input file was supplied. Because {surrogate} RNA supports a "
            f"rare-cancer hypothesis, ask whether {confirm} data are available.\n"
        )
    if fusion_effects:
        lines.append("### Fusion downstream-expression check\n")
        lines.append(
            "| Fusion | Expected program | Status | Expression source | Supporting genes | Caveat |"
        )
        lines.append("|---|---|---|---|---|---|")
        for finding in fusion_effects[:8]:
            fusion = finding.get("fusion") or {}
            genes = ", ".join(finding.get("observed_genes") or []) or "—"
            lines.append(
                f"| {fusion.get('pair') or '—'} | {finding.get('label') or '—'} | "
                f"{finding.get('status') or '—'} | {_report_expression_source_label(finding.get('expression_source'))} | "
                f"{genes} | {finding.get('caveat') or '—'} |"
            )
        lines.append("")
    if fusion_hypotheses:
        lines.append("### RNA-only fusion-effect hypotheses\n")
        lines.append(
            "These are testing prompts from downstream expression, not fusion calls.\n"
        )
        lines.append(
            "| Candidate program | Expected fusion class | Expression source | Supporting genes | Caveat |"
        )
        lines.append("|---|---|---|---|---|")
        for finding in fusion_hypotheses[:8]:
            genes = ", ".join(finding.get("observed_genes") or []) or "—"
            lines.append(
                f"| {finding.get('label') or '—'} | {finding.get('expected_pair') or '—'} | "
                f"{_report_expression_source_label(finding.get('expression_source'))} | {genes} | {finding.get('caveat') or '—'} |"
            )
        lines.append("")
    return "\n".join(lines)


def _rare_marker_hypotheses_markdown(
    analysis,
    *,
    heading: str = "## Rare-marker hypotheses",
) -> str:
    current_code = str(analysis.get("cancer_type") or "").strip()
    hypotheses = [
        finding
        for finding in (analysis.get("rare_marker_hypotheses") or [])
        if str(finding.get("cancer_type") or "").strip() != current_code
    ]
    if not hypotheses:
        return ""
    lines = [heading, ""]
    lines.append(
        "These are RNA-marker testing prompts. A primary marker can add a rare "
        "cancer to the hypothesis set, but support co-markers, expected absences, "
        "expression-reference context, and pathology/fusion/IHC evidence determine whether it "
        "should constrain the report scope.\n"
    )
    lines.append(
        "| Candidate | Primary marker | Supporting co-markers | Missing/low expected co-markers | Expected absent genes seen absent | Confirmatory step | Caveat |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for finding in hypotheses[:10]:
        marker = str(finding.get("surrogate") or "marker")
        tpm = finding.get("surrogate_tpm")
        marker_cell = (
            f"{marker} {tpm:g} TPM" if isinstance(tpm, (int, float)) else marker
        )
        lines.append(
            f"| {finding.get('cancer_type') or 'rare cancer'} | {marker_cell} | "
            f"{', '.join(finding.get('support_genes') or []) or '—'} | "
            f"{', '.join(finding.get('missing_support_genes') or []) or '—'} | "
            f"{', '.join(finding.get('absent_genes_confirmed') or []) or '—'} | "
            f"{finding.get('confirmatory_tests') or 'orthogonal testing'} | "
            f"{finding.get('caveat') or 'Hypothesis only'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _alteration_effect_markdown(
    analysis,
    *,
    heading: str = "## Alteration-expression hypotheses",
) -> str:
    hypotheses = analysis.get("mutation_expression_hypotheses") or []
    records = analysis.get("alteration_records") or []
    if not hypotheses and not records and not analysis.get("alteration_inputs_supplied"):
        return ""
    lines = [heading, ""]
    if records:
        lines.append(
            "Supplied alteration calls are carried forward as orthogonal driver or "
            "eligibility evidence. They are not inferred from RNA and should still "
            "be verified against the source molecular report.\n"
        )
        lines.append("| Gene | Supplied alteration | Type | Source |")
        lines.append("|---|---|---|---|")
        for record in records[:12]:
            if not hasattr(record, "get"):
                continue
            source = str(record.get("source_path") or "inline/user input")
            lines.append(
                f"| {record.get('gene') or '—'} | {record.get('alteration') or '—'} | "
                f"{record.get('alteration_type') or '—'} | {source} |"
            )
        if len(records) > 12:
            lines.append(f"| ... | {len(records) - 12} additional supplied calls | ... | ... |")
        lines.append("")
    elif analysis.get("alteration_inputs_supplied"):
        lines.append(
            "Alteration input was supplied, but no usable alteration calls were parsed. "
            "Review the file format/content before using therapies that require "
            "alteration evidence.\n"
        )
    if not hypotheses:
        return "\n".join(lines)
    lines.append(
        "Curated mutation/CNV expression-effect rules are interpreted as uncertain "
        "biology hypotheses. They use tumor-source attribution and context TPM "
        "when available, and bulk TPM otherwise; they should prompt confirmatory "
        "DNA/RNA testing rather than replace it.\n"
    )
    lines.append(
        "| Compatible biology | Alteration class to test | Expression source | Supporting high/low genes | Suggested assay | Caveat |"
    )
    lines.append("|---|---|---|---|---|---|")
    for finding in hypotheses[:10]:
        high = ", ".join(finding.get("observed_up_genes") or [])
        low = ", ".join(finding.get("observed_low_genes") or [])
        support = "; ".join(part for part in [f"high {high}" if high else "", f"low {low}" if low else ""] if part) or "—"
        lines.append(
            f"| {finding.get('label') or '—'} | {finding.get('alteration') or '—'} | "
            f"{_report_expression_source_label(finding.get('expression_source'))} | {support} | "
            f"{finding.get('suggested_assay') or '—'} | {finding.get('caveat') or '—'} |"
        )
    return "\n".join(lines)


def _decision_channel_detail_summary(details):
    if not isinstance(details, dict) or not details:
        return ""
    parts = []
    mmr = details.get("mismatch_repair") or {}
    if isinstance(mmr, dict) and mmr:
        p_msi = mmr.get("msi_probability")
        if isinstance(p_msi, (int, float)):
            parts.append(f"MSI RNA probability={p_msi:.3f}")
        context = mmr.get("context_group")
        if context:
            parts.append(f"context {context}")
        members = []
        for row in mmr.get("member_probabilities") or []:
            if not isinstance(row, dict):
                continue
            probability = row.get("msi_probability")
            member = row.get("member")
            if member and isinstance(probability, (int, float)):
                members.append(f"{member} {probability:.2f}")
        if members:
            parts.append("members " + ", ".join(members[:3]))
        validation = mmr.get("validation") or {}
        if isinstance(validation, dict):
            lto = validation.get("leave_tissue_out_accuracy")
            if isinstance(lto, (int, float)):
                parts.append(f"LTO accuracy={lto:.3f}")
    if details.get("learned_stage"):
        parts.append(f"learned stage {details.get('learned_stage')}")
    if details.get("label_space"):
        parts.append(f"space {details.get('label_space')}")
    if details.get("training_split_policy"):
        parts.append(f"training {details.get('training_split_policy')}")
    if details.get("panel"):
        parts.append(f"panel {details.get('panel')}")
    if details.get("out_of_beam_rescue"):
        parts.append("out-of-beam rescue")
    for key, label in (
        ("probability", "p"),
        ("margin", "margin"),
        ("context_support", "context"),
        ("holdout_top1_accuracy", "held-out top1"),
        ("holdout_medoid_top1_accuracy", "medoid top1"),
        ("oof_top3_recovery", "top3 recovery"),
        ("rho", "rho"),
    ):
        value = details.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{label}={value:.3f}")
    top = details.get("top_predictions") or []
    if top:
        formatted = []
        for row in top[:3]:
            if not isinstance(row, dict):
                continue
            code = row.get("code") or row.get("label")
            probability = row.get("probability")
            if code and isinstance(probability, (int, float)):
                formatted.append(f"{code} {probability:.2f}")
            elif code:
                formatted.append(str(code))
        if formatted:
            parts.append("top " + ", ".join(formatted))
    rationale = details.get("rationale")
    if rationale:
        parts.append(str(rationale)[:140])
    return "; ".join(parts)


def _cancer_type_decision_trace_markdown(analysis):
    evidence = analysis.get("cancer_type_evidence") or {}
    graph = evidence.get("staged_evidence_graph") or {}
    channels = graph.get("channels") or []
    if not channels:
        return ""
    selected = evidence.get("selected") or {}
    graph_selected = graph.get("selected") or {}
    veto = analysis.get("cancer_type_evidence_vetoed") or {}
    lines = ["## Cancer-type decision evidence trace\n"]
    if veto:
        kept = veto.get("kept_call") or analysis.get("inferred_cancer_type") or ""
        vetoed = veto.get("vetoed_call") or (
            graph.get("vetoed_selected") or {}
        ).get("code")
        lines.append(
            f"Evidence selection vetoed: **{_cancer_label(vetoed)}** was not "
            f"used as the report label; report scope reverted to "
            f"**{_cancer_label(kept)}**. Rows below preserve the selected, "
            "vetoed, blocked, and context-only signals used by the cancer-type "
            "decision process.\n"
        )
    elif selected:
        selected_code = selected.get("cancer_type") or selected.get("code") or ""
        selected_by = selected.get("selected_by") or "primary_expression_context"
        if not selected.get("can_select_report_label", True):
            lines.append(
                f"Fallback RNA context: **{_cancer_label(selected_code)}** via "
                f"`{selected_by}`. This row did not independently select the "
                "report label; rows below include selectable, blocked, and "
                "context-only signals used by the cancer-type decision process.\n"
            )
        else:
            lines.append(
                f"Selected report label: **{_cancer_label(selected_code)}**. "
                f"Selection driver: `{selected_by}`. Rows below include selected, blocked, and "
                "context-only signals used by the cancer-type decision process.\n"
            )
    elif graph_selected:
        selected = graph_selected
        selected_code = selected.get("code") or ""
        selected_by = selected.get("selected_by") or "primary_expression_context"
        if not selected.get("selects_report_label"):
            lines.append(
                f"Fallback RNA context: **{_cancer_label(selected_code)}** via "
                f"`{selected_by}`. This row did not independently select the "
                "report label; rows below include selectable, blocked, and "
                "context-only signals used by the cancer-type decision process.\n"
            )
        else:
            lines.append(
                f"Selected report label: **{_cancer_label(selected_code)}**. "
                f"Selection driver: `{selected_by}`. Rows below include selected, blocked, and "
                "context-only signals used by the cancer-type decision process.\n"
            )
    lines.append("| Stage | Candidate | Channel | Role | Status | Support | Details |")
    lines.append("|---|---|---|---|---|---:|---|")
    ordered = sorted(
        channels,
        key=lambda row: (
            str(row.get("stage") or ""),
            str(row.get("candidate_code") or row.get("code") or ""),
            str(row.get("channel") or ""),
            str(row.get("role") or ""),
        ),
    )
    # A channel detail carries free-text rationale that can contain a literal
    # `|` (splits the row into a stray column) or a newline (breaks the table).
    # Escape it via the shared public reporting helper before interpolation.
    from .reporting import md_table_cell

    for row in ordered[:80]:
        support = row.get("support")
        support_text = (
            f"{float(support):.3f}" if isinstance(support, (int, float)) else ""
        )
        candidate = row.get("candidate_code") or row.get("code") or ""
        detail = md_table_cell(_decision_channel_detail_summary(row.get("details") or {}))
        lines.append(
            f"| {row.get('stage') or ''} | {_cancer_label(candidate)} | "
            f"`{row.get('channel') or ''}` | {row.get('role') or ''} | "
            f"{row.get('status') or ''} | {support_text} | {detail or '—'} |"
        )
    if len(ordered) > 80:
        lines.append(
            f"\nAdditional low-priority/context rows omitted from this table: {len(ordered) - 80}."
        )
    return "\n".join(lines)


def _build_evidence_report(
    analysis,
    ranges_df,
    decomp_results,
    cancer_code,
    sample_id,
    target_report_md,
):
    from .provenance import build_provenance_md

    header_id = f": {sample_id}" if sample_id else ""
    provenance_md = build_provenance_md(
        analysis,
        ranges_df,
        decomp_results,
        cancer_code=cancer_code,
        sample_id=sample_id,
    )
    provenance_body = _demote_markdown_headings(
        _strip_markdown_wrapper(provenance_md),
        levels=1,
    )
    target_body = _demote_markdown_headings(
        _strip_markdown_wrapper(target_report_md),
        levels=1,
    )

    lines = [f"# Evidence{header_id}\n"]
    lines.append(
        "This appendix keeps the stepwise and table-heavy support behind the "
        "distilled reports. Use it to audit how the call was assembled, inspect "
        "uncertainty, or review the full target evidence tables."
    )
    lines.append("")
    lines.append("## Stepwise attribution chain\n")
    lines.append(provenance_body)
    lines.append("")
    rna_qc_body = rna_quant_qc_markdown(
        analysis.get("rna_quant_qc"),
        heading="## RNA quantification QC",
    )
    if rna_qc_body:
        lines.append(rna_qc_body)
        lines.append("")
    rescue_line = expression_qc_rescue_summary_line(
        analysis.get("expression_qc_rescue")
    )
    if rescue_line:
        lines.append("## Expression-table QC rescue\n")
        lines.append(rescue_line)
        top_removed = (
            analysis.get("expression_qc_rescue", {}).get("top_removed_genes") or []
        )
        if top_removed:
            lines.append("")
            lines.append("| Feature | Raw TPM | Raw share | QC class |")
            lines.append("|---|---:|---:|---|")
            for row in top_removed[:10]:
                # A few-thousand-TPM technical feature can be ~0.1-0.4% of raw
                # mass, which "{:.0%}" rounds to a misleading "0%" next to a
                # large Raw TPM. Render "<1%" for a nonzero sub-0.5% share,
                # mirroring the QC-summary idiom (#85 minor).
                share_val = float(row.get("share") or 0.0)
                share_label = "<1%" if 0.0 < share_val < 0.005 else f"{share_val:.0%}"
                lines.append(
                    f"| {row.get('gene') or '—'} | {float(row.get('tpm') or 0.0):.1f} | "
                    f"{share_label} | {row.get('qc_class') or '—'} |"
                )
        lines.append("")
    fusion_body = _fusion_evidence_markdown(analysis)
    if fusion_body:
        lines.append(fusion_body)
        lines.append("")
    rare_marker_body = _rare_marker_hypotheses_markdown(analysis)
    if rare_marker_body:
        lines.append(rare_marker_body)
        lines.append("")
    alteration_body = _alteration_effect_markdown(analysis)
    if alteration_body:
        lines.append(alteration_body)
        lines.append("")
    decision_trace_body = _cancer_type_decision_trace_markdown(analysis)
    if decision_trace_body:
        lines.append(decision_trace_body)
        lines.append("")
    lines.append("## Full target evidence\n")
    lines.append(target_body)
    return "\n".join(lines)


# A lineage gene whose per-gene purity ratio (sample ÷ TCGA reference) is a low
# outlier but which is still expressed at a real TPM is a reference-scaling
# artifact — its TCGA reference is very high — NOT evidence the tumor lost the
# gene (#85.2). Narrating CDK4 at 101 TPM as "de-differentiated" contradicts the
# therapy tables that show it highly expressed. Only a gene that is BOTH a
# low-ratio outlier AND expressed below this floor reads as possible loss.
_LINEAGE_EXPRESSED_TPM_FLOOR = 10.0


def _classify_lineage_calibration_genes(sorted_genes, median_p):
    """Split lineage-calibration genes for the purity-per-gene narration.

    Returns ``(retained, expressed_below_scale, possibly_lost)``:

    - ``retained`` — per-gene purity within half the median: a reliable anchor.
    - Genes below that half-median threshold are excluded from the estimate as
      outliers; the reason matters for how they read:
      - ``expressed_below_scale`` — still expressed (sample TPM ≥
        ``_LINEAGE_EXPRESSED_TPM_FLOOR``): the low ratio is reference scaling,
        not loss.
      - ``possibly_lost`` — low ratio AND low expression: a genuine low/absent
        signal that can honestly read as de-differentiation.
    """
    if median_p is None or median_p <= 0:
        return list(sorted_genes), [], []
    retained, expressed_below_scale, possibly_lost = [], [], []
    for g in sorted_genes:
        if g["purity"] >= median_p * 0.5:
            retained.append(g)
        elif (g.get("sample_tpm") or 0.0) >= _LINEAGE_EXPRESSED_TPM_FLOOR:
            expressed_below_scale.append(g)
        else:
            possibly_lost.append(g)
    return retained, expressed_below_scale, possibly_lost


def _generate_text_reports(
    analysis,
    embedding_meta,
    prefix,
    decomp_results=None,
    input_path=None,
    ranges_df=None,
    sample_id=None,
    df_expr=None,
):
    """Write the detailed ``*-analysis.md`` report."""
    from .report_view import finalized_purity_headline

    cancer_code = analysis["cancer_type"]
    purity = analysis["purity"]
    # Pin the adopted-purity headline to the single frozen ReportView surface so
    # analysis.md can never show a stale pre-finalization purity — figure ==
    # summary == analysis.md by construction. Byte-stable today (rendered after
    # finalization); the guard survives a future reorder that isn't.
    _fp_overall, _fp_lower, _fp_upper = finalized_purity_headline(analysis)
    if _fp_overall is not None:
        purity = {
            **purity,
            "overall_estimate": _fp_overall,
            "overall_lower": _fp_lower,
            "overall_upper": _fp_upper,
        }
    mhc1 = analysis["mhc1"]
    top_tissues = analysis["tissue_scores"][:5]
    tissue_score_details = {
        row.get("tissue"): row
        for row in (analysis.get("tissue_score_details") or [])
        if row.get("tissue")
    }
    candidate_trace = analysis.get("candidate_trace", [])
    family_summary = analysis.get("family_summary", {})
    fit_quality = analysis.get("fit_quality", {})
    sample_mode = analysis.get("sample_mode", "auto")
    constraints = analysis.get("analysis_constraints", {})
    cancer_type_context = cancer_type_context_from_analysis(analysis)
    reference_cancer_code = cancer_type_context.code_for("reference") or cancer_code
    reference_cancer_name = cancer_code_display_name(
        reference_cancer_code,
        CANCER_TYPE_NAMES.get(reference_cancer_code, reference_cancer_code),
    )
    call_summary = analysis.get("call_summary") or _summarize_sample_call(
        analysis,
        decomp_results or [],
        sample_mode=sample_mode,
    )
    best_decomp = decomp_results[0] if decomp_results else None
    family_display = family_summary.get("display")
    subtype_clause = family_summary.get("subtype_clause")

    sample_context = analysis.get("sample_context")

    # Disease-state synthesis (#78) — still used by analysis.md.
    disease_state_paragraph = compose_disease_state_narrative(analysis)
    disease_state_display = report_disease_state_text(
        disease_state_paragraph,
        analysis=analysis,
    )

    # The old free-form ``*-summary.md`` paragraph (disease-state +
    # QC + headline + composition + therapy-response state) was
    # retired in 4.41.0 — every block already lived in analysis.md.
    # The name ``summary.md`` now carries what used to be
    # ``brief.md`` (the 1-page clinician file).

    # --- Detailed report ---
    lines = ["# Detailed Sample Analysis\n"]
    if input_path:
        # Input path at the top so the file is self-identifying even
        # without sample_context downstream. Propagated from
        # ``analyze()`` along with the rest of the analysis dict.
        lines.append(f"*Input*: `{input_path}`\n")

    lines.append("## Sample framing\n")
    if sample_context is not None:
        prep_clause = library_prep_clause(sample_context.library_prep)
        pres_label = sample_context.preservation.replace("_", " ")
        if pres_label == "fresh frozen":
            pres_label = "fresh/frozen-like"
        lines.append(
            f"- **Input type**: {prep_clause}; "
            f"preservation inferred as {pres_label} from RNA QC."
        )
    scale_qc = analysis.get("expression_scale_qc") or {}
    if scale_qc.get("converted_from") == "log2_tpm_plus_one":
        post_sum = scale_qc.get("post_conversion_sum_tpm") or scale_qc.get("sum_tpm")
        sum_clause = f"; post-conversion sum {post_sum/1_000_000:.2f}M" if post_sum else ""
        lines.append(
            "- **Expression scale QC**: input resembled log2(TPM+1); converted "
            f"to linear TPM before interpretation{sum_clause}."
        )
    elif scale_qc.get("warnings"):
        lines.append(f"- **Expression scale QC**: {scale_qc['warnings'][0]}.")
    rna_qc = analysis.get("rna_quant_qc") or {}
    if rna_qc.get("summary"):
        warning_clause = ""
        if rna_qc.get("warnings"):
            warning_clause = f"; caution: {str(rna_qc['warnings'][0]).rstrip('.')}"
        lines.append(f"- **RNA quantification QC**: {rna_qc['summary']}{warning_clause}.")
    rescue_line = expression_qc_rescue_summary_line(
        analysis.get("expression_qc_rescue")
    )
    if rescue_line:
        lines.append("- " + rescue_line)
    if call_summary.get("label_options"):
        active_label = call_summary["label_options"][0]
        if len(call_summary["label_options"]) == 1:
            lines.append(f"- **Working cancer call**: {_cancer_label(active_label)}.")
        else:
            lines.append(
                f"- **Working cancer call**: {_cancer_label(active_label)} "
                "(provisional; retained alternatives are summarized under "
                "Cancer-Type Differential)."
            )
    context_line = _cancer_type_context_line(cancer_type_context)
    if context_line:
        lines.append(context_line)
    if reference_cancer_code != cancer_code:
        if cancer_type_context.fine_expression_available:
            lines.append(
                f"- **Fallback expression reference**: exact {_cancer_label(cancer_code)} "
                "expression is preferred where supported; broad cohort-normalized "
                f"steps fall back to {reference_cancer_code} ({reference_cancer_name}) "
                "for purity/range context, pathway fold-changes, and reference-space plots."
            )
        else:
            lines.append(
                f"- **Fallback expression reference**: {reference_cancer_code} "
                f"({reference_cancer_name}) is used for cohort-normalized expression, "
                "purity/range context, pathway fold-changes, and reference-space plots because "
                "an exact expression reference is not available; "
                f"{_cancer_label(cancer_code)} remains the report label."
            )
    lines.append(
        f"- **{_purity_metric_label(sample_mode).title()}**: {_purity_ci_phrase(purity)}."
    )
    if call_summary.get("site_indeterminate"):
        lines.append("- **Background/site context**: indeterminate.")
    elif call_summary.get("reported_site"):
        lines.append(f"- **Background/site context**: {call_summary['reported_site']}.")
    if call_summary.get("site_note"):
        lines.append(
            f"- **Note**: {_strip_terminal_punctuation(call_summary['site_note'])}."
        )
    lines.append("")

    if disease_state_display:
        disease_state_markdown = str(disease_state_display).replace(
            " **Active IFN response**",
            "\n\n**Active IFN response**",
        )
        lines.append("## Disease and pathway state\n")
        lines.append(disease_state_markdown)
        lines.append("")

    integrated_bullets = _integrated_evidence_bullets(
        analysis,
        decomp_results or [],
    )
    if integrated_bullets:
        lines.append("## Integrated evidence synthesis\n")
        lines.extend(integrated_bullets)
        lines.append("")

    fusion_body = _fusion_evidence_markdown(analysis)
    if fusion_body:
        lines.append(fusion_body)
        lines.append("")
    rare_marker_body = _rare_marker_hypotheses_markdown(analysis)
    if rare_marker_body:
        lines.append(rare_marker_body)
        lines.append("")
    alteration_body = _alteration_effect_markdown(
        analysis,
        heading="## Mutation/CNV expression-effect hypotheses",
    )
    if alteration_body:
        lines.append(alteration_body)
        lines.append("")

    # Tissue-composition screen (#149) — same signal that drives the
    # brief banner and the summary top-line, surfaced here with the
    # full top-3 tissues / top-3 cohorts so a reader doing a deep
    # analysis.md read sees the tissue-composition prior inline.
    hvt = analysis.get("healthy_vs_tumor")
    if hvt is not None and hvt.top_normal_tissues:
        from pirlygenes.gene_sets_cancer import proliferation_panel_gene_names

        _prolif_panel_size = len(proliferation_panel_gene_names())
        lines.append("## Tissue Composition Screen\n")
        lines.append(f"- **Cancer hint**: {hvt.cancer_hint}")
        lines.append(
            f"- **Proliferation panel**: "
            f"{hvt.proliferation_log2_mean:.2f} log2-TPM mean across "
            f"{hvt.proliferation_genes_observed}/{_prolif_panel_size} genes observed"
        )
        hpa_line = ", ".join(
            f"{t.removesuffix('_nTPM').replace('_', ' ')} (ρ={rho:.2f})"
            for t, rho in hvt.top_normal_tissues[:3]
        )
        lines.append(f"- **Top normal-tissue matches**: {hpa_line}")
        tcga_line = ", ".join(
            f"{t.removesuffix('_TPM')} (ρ={rho:.2f})"
            for t, rho in hvt.top_tcga_cohorts[:3]
        )
        lines.append(f"- **Top cancer-reference matches**: {tcga_line}")
        # Surface the reasoning trace (which rule fired + rationale)
        # so a reader can audit how the tissue-composition screen arrived at the hint.
        # Full tumor-evidence breakdown (per-channel scores + aggregate)
        # so the hint isn't a black box.
        lines.append(f"- **Evidence**: {hvt.evidence.synthesis()}")
        lines.append("")

    # Input characterization (#68) — surfaced before quality/decomp so
    # readers see whether the file we analysed was transcript-level vs
    # gene-level, whole-transcriptome vs panel, poly-A vs total RNA,
    # before they scrutinise downstream numbers.
    if sample_context is not None:
        ctx_signals = sample_context.signals or {}
        lines.append("## Input characterization\n")
        if input_path:
            lines.append(f"- **Source file**: `{input_path}`")
        lines.append(
            f"- **Library prep**: {library_prep_display_label(sample_context.library_prep)} "
            f"({heuristic_support_label(sample_context.library_prep_confidence)})"
        )
        lines.append(
            f"- **Preservation**: {sample_context.preservation.replace('_', ' ')}"
            + f"; length-pair {length_pair_display_label(sample_context)}"
        )
        raw_scale_qc = analysis.get("raw_expression_scale_qc") or {}
        scale_qc = analysis.get("expression_scale_qc") or {}
        if raw_scale_qc:
            lines.append(
                f"- **Expression scale QC (raw)**: sum {raw_scale_qc.get('sum_tpm', 0.0):.0f}; "
                f"max {raw_scale_qc.get('max_tpm', 0.0):.1f}; "
                f"housekeeping median {raw_scale_qc.get('housekeeping_median_tpm', 0.0):.1f}."
            )
            technical_phrase = technical_rna_component_phrase(raw_scale_qc)
            if technical_phrase:
                lines.append(
                    f"- **Technical RNA burden (raw)**: {technical_phrase}."
                )
                if sample_context.degradation_index is not None:
                    lines.append(
                        "- **QC-axis interpretation**: the degradation index is a "
                        "long/short transcript-pair check. It does not clear "
                        "rRNA-pseudogene or rDNA/repeat-mapping denominator artifacts; "
                        "interpret both QC axes together."
                    )
            lines.extend(qc_category_levels_lines(raw_scale_qc))
        if scale_qc and analysis.get("expression_qc_rescue", {}).get("applied"):
            lines.append(
                f"- **Expression scale QC (analysis TPM)**: sum {scale_qc.get('sum_tpm', 0.0):.0f}; "
                f"max {scale_qc.get('max_tpm', 0.0):.1f}; "
                "mtDNA/rRNA/pseudogene-like features set to zero before renormalization."
            )
        rescue_line = expression_qc_rescue_summary_line(
            analysis.get("expression_qc_rescue")
        )
        if rescue_line:
            lines.append("- " + rescue_line)
            lines.append(
                "- **QC figures**: see `*-expression-top-features-qc.png`, "
                "`*-expression-concentration-curve-qc.png`, "
                "`*-qc-reference-mtdna.png`, and "
                "`*-qc-reference-technical-rna-burden.png`."
            )
        rna_qc_body = rna_quant_qc_markdown(
            analysis.get("rna_quant_qc"),
            heading="### Run-level quantifier QC",
        )
        if rna_qc_body:
            lines.append("")
            lines.append(rna_qc_body)
        n_det = ctx_signals.get("genes_detected_above_1_tpm")
        if n_det is not None:
            # Labeled by frame so it is not read as contradicting the
            # "Gene detection" count in the run-level quantifier QC table:
            # that count is the raw quantifier frame at >=1 TPM, this one is
            # the symbol-collapsed (transcript/proteoform-folded) frame at
            # >1 TPM, so the two legitimately differ (#85 minor).
            lines.append(
                f"- **Detection breadth** (gene-symbol frame): {n_det} genes with TPM > 1 "
                f"({ctx_signals.get('genes_detected_above_10_tpm', 0)} with TPM > 10, "
                f"{ctx_signals.get('genes_detected_above_0p5_tpm', 0)} with TPM > 0.5); "
                "counted on symbol-collapsed TPM, so it can differ from the run-level "
                "quantifier detection count."
            )
        top50 = ctx_signals.get("top_50_share_of_total_tpm")
        top2000 = ctx_signals.get("top_2000_share_of_total_tpm")
        if top50 is not None:
            concentration = (
                f"- **Concentration**: top 50 genes carry {top50:.0%} of total TPM"
            )
            if top2000 is not None:
                concentration += f"; top 2000 carry {top2000:.0%}"
            lines.append(concentration)
        concentration_level = str(
            ctx_signals.get("expression_concentration_level") or ""
        ).strip()
        if concentration_level in {"high", "extreme"}:
            dominant = ctx_signals.get("dominant_expression_genes") or []
            top_bits = []
            for row in dominant[:5]:
                gene = str(row.get("gene") or "").strip()
                share = row.get("share")
                qc_class = str(row.get("qc_class") or "").strip()
                if gene and isinstance(share, (int, float)):
                    class_suffix = (
                        f" ({qc_class})"
                        if qc_class and qc_class != "protein-coding/other"
                        else ""
                    )
                    top_bits.append(f"{gene} {share:.0%}{class_suffix}")
            suffix = f"; dominant genes: {', '.join(top_bits)}" if top_bits else ""
            lines.append(
                f"- **Expression concentration QC**: {concentration_level}ly concentrated "
                "TPM distribution; this can reflect technical RNA, "
                "ribosomal-pseudogene/small-RNA carryover, blood or immune-clone "
                f"dominance, low library complexity, or assay/input issues{suffix}."
            )
        if ctx_signals.get("likely_targeted_panel"):
            lines.append(
                "- **Likely targeted panel** (few detected genes or >90% TPM "
                "concentrated in top 2000 genes) — downstream scores assume "
                "whole-transcriptome input; interpret carefully."
            )
        log2_med = ctx_signals.get("log2_tpm_median")
        if log2_med is not None:
            lines.append(
                f"- **Expression range**: log2(TPM+1) median={log2_med:.2f}, "
                f"IQR={ctx_signals.get('log2_tpm_iqr', 0):.2f}, "
                f"p95={ctx_signals.get('log2_tpm_p95', 0):.2f}"
            )
        if sample_context.missing_mt:
            lines.append(
                "- **Mitochondrial genes missing** from quant table — "
                "degradation signal from MT fraction is unreliable."
            )
        lines.append("")

    # Sample quality — driven by the step-1 SampleContext so the
    # three report sections (summary / analysis / targets) agree
    # (#77). The raw sample_quality output is still used for its
    # tissue-matched baselines, but the top-level "is this sample
    # degraded?" answer comes from sample_context.preservation.
    quality = analysis.get("quality")
    if quality:
        lines.append("## Sample Quality\n")
        deg = quality["degradation"]
        cul = quality["culture"]

        if sample_context is not None:
            # Preservation call is the sample_context call; the
            # sample_quality raw-signal read is shown as supporting
            # detail under it, not as a competing verdict.
            preservation_label = sample_context.preservation.replace("_", " ")
            prep = getattr(sample_context, "library_prep", "unknown")
            prep_label = prep.replace("_", " ")
            lines.append(
                f"**Preservation**: {preservation_label} "
                f"(library prep inferred as *{prep_label}*, "
                f"{heuristic_support_label(sample_context.library_prep_confidence)})"
            )
            if (
                sample_context.degradation_severity
                and sample_context.degradation_severity != "none"
            ):
                lines.append(
                    f"- Severity: {sample_context.degradation_severity}"
                    + (
                        f" (length-pair index {sample_context.degradation_index:.2f})"
                        if sample_context.degradation_index is not None
                        else ""
                    )
                )
        else:
            lines.append(f"**RNA degradation**: {deg['level']}")
        lines.append(f"- Mitochondrial fraction: {deg['mt_fraction']:.1%}")
        lines.append(f"- Ribosomal protein fraction: {deg['rp_fraction']:.1%}")
        if deg.get("long_short_ratio") is not None:
            lines.append(
                f"- Long/short transcript index: {deg['long_short_ratio']:.2f}"
            )
        # #77: only echo the raw sample_quality "MT filtered" message
        # when the inferred library prep doesn't already explain it.
        prep = getattr(sample_context, "library_prep", None) if sample_context else None
        mt_expected_missing = prep in _MT_EXPECTED_MISSING_PREPS
        suppress_mt_rp_baseline = (
            mt_expected_missing and deg.get("mt_fraction", 1.0) < 0.005
        )
        if deg.get("matched_tissue") and not suppress_mt_rp_baseline:
            lines.append(
                f"- Normal-tissue QC baseline: {deg['matched_tissue']} "
                f"(expected MT={deg['baseline_mt']:.1%}, RP={deg['baseline_rp']:.1%})"
            )
            if deg.get("mt_fold") is not None:
                lines.append(
                    f"- Relative to that QC baseline: MT {deg['mt_fold']:.1f}×, "
                    f"RP {deg['rp_fold']:.1f}×"
                )
        elif deg.get("matched_tissue") and suppress_mt_rp_baseline:
            lines.append(
                "- MT/RP baseline comparison is not emphasized here because this "
                "library prep can under-represent mitochondrial reads; preservation "
                "is judged mainly from the long/short transcript index."
            )
        if deg["level"] not in ("normal", "unknown"):
            lines.append(f"- *{deg['message']}*")
        elif deg["level"] == "unknown" and not mt_expected_missing:
            lines.append(f"- *{deg['message']}*")
        lines.append("")

        # Cell-culture / stress section is only meaningful when the
        # sample may plausibly be a cell line. Skip for solid-tumor
        # biopsies (#77) — the HSP90AA1 / GLS elevated pattern is
        # almost always hypoxia / tumor metabolism, already covered
        # by the therapy-response hypoxia axis.
        skip_culture = sample_mode == "solid"
        if not skip_culture:
            lines.append(
                f"**Cell culture / cell line**: {cul['level'].replace('_', ' ')}"
            )
            lines.append(f"- Culture-stress z-score: {cul['stress_score']:.1f}")
            lines.append(
                f"- TME marker mean: {cul['tme_mean_tpm']:.1f} TPM "
                f"({'absent' if cul['tme_absent'] else 'present'})"
            )
            if cul["top_stress_genes"]:
                top_genes_str = ", ".join(
                    f"{g}={t:.0f}" for g, t in cul["top_stress_genes"][:5]
                )
                lines.append(f"- Top stress genes: {top_genes_str}")
            if cul["level"] != "normal":
                lines.append(f"- *{cul['message']}*")
            lines.append("")

    # Cancer type identification
    lines.append("## Cancer Type Identification\n")
    lines.append(f"- **Sample mode**: {_sample_mode_display(sample_mode)}")
    # #33 source attribution — show whether the call was auto-detected
    # or user-specified. Used to live only in the retired summary.md.
    source_label = {
        "auto-detected": "auto-detected",
        "user-specified": "user-specified",
    }.get(analysis.get("cancer_type_source"))
    if source_label:
        lines.append(f"- **Call source**: {source_label}")
    if fit_quality.get("label"):
        lines.append(f"- **Fit quality**: {fit_quality['label']}")
    if fit_quality.get("message"):
        lines.append(f"- **Fit note**: {fit_quality['message']}")
    if call_summary.get("label_options"):
        active_label = call_summary["label_options"][0]
        lines.append(f"- **Report label**: {_cancer_label(active_label)}")
        scope_caveat = _selected_report_scope_caveat(analysis)
        if scope_caveat:
            lines.append(f"- **Report-label caveat**: {scope_caveat}")
        if len(call_summary["label_options"]) >= 2:
            lines.append(
                "- **Retained alternatives**: see Cancer-Type Differential below; "
                "these are not active downstream report labels."
            )
    if family_display:
        lines.append(f"- **Broad family context**: {family_display}")
        if subtype_clause:
            family_codes = [str(code) for code in family_summary.get("codes") or []]
            if len(family_codes) >= 2:
                lines.append(
                    f"- **Broad reference candidates within family**: {subtype_clause}"
                )
            elif reference_cancer_code != cancer_code and family_codes:
                lines.append(
                    f"- **Reference cohort within family**: {_cancer_label(family_codes[0])} "
                    "is used for broad expression context; it is not a separate "
                    "refined report label."
                )
            elif family_codes:
                lines.append(
                    f"- **Top reference candidate within family**: {_cancer_label(family_codes[0])}"
                )
    if constraints:
        if constraints.get("cancer_type"):
            lines.append(
                f"- **Externally supplied report label**: {constraints['cancer_type']}"
            )
        if constraints.get("sample_mode"):
            lines.append(
                f"- **Requested sample mode**: {_sample_mode_display(constraints['sample_mode'])}"
            )
        if constraints.get("tumor_context"):
            lines.append(
                f"- **Requested tumor context**: {constraints['tumor_context']}"
            )
        if constraints.get("site_hint"):
            lines.append(f"- **Requested site hint**: {constraints['site_hint']}")
        if constraints.get("decomposition_templates"):
            lines.append(
                "- **Requested decomposition templates**: "
                + ", ".join(constraints["decomposition_templates"])
            )
    inferred_site = analysis.get("inferred_site_context") or {}
    if inferred_site:
        lines.append(
            "- **Inferred host site**: "
            f"{str(inferred_site.get('site') or '').replace('_', ' ')} "
            f"(background tissue {str(inferred_site.get('tissue') or '').replace('_', ' ')}, "
            f"score {float(inferred_site.get('score') or 0.0):.2f}); inferred from "
            "off-primary expression background, not supplied as a user constraint."
        )
    # Display candidates in the order of the stated ranking metric (Normalized =
    # support_fraction_of_top, top = 1.0). The stored trace order can lag this,
    # which previously rendered a lower-scoring row above a higher-scoring one.
    ranked_candidates = (
        sorted(
            candidate_trace,
            key=lambda r: (
                float(r.get("support_fraction_of_top") or 0.0),
                float(r.get("support_geomean") or 0.0),
            ),
            reverse=True,
        )
        if candidate_trace
        else []
    )
    if not candidate_trace:
        top_cancers = analysis.get(
            "top_cancers", [(cancer_code, analysis["cancer_score"])]
        )
        for code, score in top_cancers[:5]:
            lines.append(f"- **{_cancer_label(code)}**: {score:.3f}")
    tumor_type_sanity = _tumor_type_sanity_markdown(analysis)
    if tumor_type_sanity:
        lines.append("")
        lines.append(tumor_type_sanity)
    lines.append("")

    if candidate_trace:
        vote_summary = _cancer_type_vote_summary_markdown(
            analysis,
            call_summary=call_summary,
        )
        if vote_summary:
            lines.append("")
            lines.append(vote_summary)
        differential = _retained_cancer_type_differential_markdown(
            analysis,
            call_summary=call_summary,
            decomp_results=decomp_results or [],
        )
        if differential:
            lines.append("")
            lines.append(differential)
        lines.append("")
        lines.append("### Cancer Type Inference — Candidate Ranking\n")
        lines.append(
            "Each row is a top-level cancer-code hypothesis considered by the classifier. "
            "Most of these labels are broad reference cohorts, but later steps can also "
            "use finer subtype and decomposition references from non-TCGA sources. Three scores summarize the "
            "match; the top row is the pan-cancer signature-ranker leader, which may differ from the "
            "active report label when stronger orthogonal evidence selects another row.\n\n"
            "- **Signature** (0–1): raw match quality between the sample's expression and "
            "this candidate's broad reference signature, computed from z-scored expression "
            "of cancer-type-enriched genes. Interpretable on its own — higher means the "
            "candidate pattern is strongly present. Does not account for purity or lineage "
            "concordance.\n"
            "- **Geomean** (0–1): the geometric mean of the five factors that feed the "
            "ranking (signature × purity × lineage support × signature stability × family "
            "factor). Stays bounded on [0, 1] so it's comparable across samples, unlike "
            "the raw product which collapses toward zero.\n"
            "- **Normalized** (0–1, top = 1.0): each candidate's composite support score "
            "divided by the top candidate's. Use this to judge separation — if the runner-up "
            "is ≪ 1.0, the top call is well-isolated; values near 1.0 mean the call is "
            "ambiguous between rows.\n\n"
            "Supporting columns: **Purity** is the overall tumor-purity estimate under "
            "this hypothesis; **Lineage** is a purity estimate derived only from the "
            "curated lineage genes for that candidate cancer type; **Concordance** is "
            "how well those lineage genes match the expected pattern for that candidate.\n"
        )
        lines.append(
            "| Cancer | Family | Signature | Geomean | Normalized | Purity | Lineage | Concordance |"
        )
        lines.append(
            "|--------|--------|-----------|---------|------------|--------|---------|-------------|"
        )
        for row in ranked_candidates[:8]:
            lineage = row.get("lineage_purity")
            concordance = row.get("lineage_concordance")
            lines.append(
                f"| {_cancer_label(row['code'])} | {row.get('family_label') or '—'} | "
                f"{row.get('signature_score', 0.0):.3f} | "
                f"{row.get('support_geomean', 0.0):.3f} | "
                f"{row.get('support_fraction_of_top', 0.0):.3f} | "
                f"{row.get('purity_estimate', 0.0):.3f} | "
                f"{'%.3f' % lineage if lineage is not None else '—'} | "
                f"{'%.3f' % concordance if concordance is not None else '—'} |"
            )
        rescue_row = next(
            (row for row in candidate_trace[:8] if row.get("support_override")),
            None,
        )
        if rescue_row:
            support_override = rescue_row.get("support_override") or {}
            rescue_kind = str(support_override.get("kind") or "")
            lines.append("")
            if rescue_kind == "low_purity_prad_stromal_context":
                lines.append(
                    "*Pitfall-aware ranking note*: the PRAD row was promoted above the "
                    "stromal/SARC row because prostate tissue markers and the raw PRAD "
                    "signature were strong while the epithelial PRAD lineage program was "
                    "attenuated. The SARC row remains visible as a near-tied stromal/"
                    "smooth-muscle alternative, not as a resolved diagnosis."
                )
            elif rescue_kind == "coarse_tcga_orphan_context":
                recommended = str(
                    support_override.get("recommended_code")
                    or rescue_row.get("code")
                    or ""
                )
                competitor = str(support_override.get("competing_code") or "")
                competitor_clause = (
                    f" over {_cancer_label(competitor)}" if competitor else ""
                )
                lines.append(
                    "*Context-aware ranking note*: "
                    f"{_cancer_label(recommended)} was promoted{competitor_clause} "
                    "because tissue-composition context and direct cancer evidence supported "
                    "the orphan cancer type more strongly than the broad-family penalty "
                    "allowed."
                )
            else:
                lines.append(
                    "*Context-aware ranking note*: the leading row was promoted by a "
                    "documented cancer-call rescue rule; treat the alternatives as "
                    "visible unresolved hypotheses until pathology/clinical context "
                    "confirms the label."
                )
        lines.append("")

        # Per-candidate evidence tables (pirl-unc/trufflepig#29) — let an
        # LLM consumer re-derive the cancer-biology rationale from
        # tabulated TPMs without re-running the pipeline.
        try:
            from .evidence_tables import build_candidate_evidence_block

            sample_tpm_by_symbol = {}
            if df_expr is not None and "TPM" in df_expr.columns:
                sym_col = (
                    "gene_symbol"
                    if "gene_symbol" in df_expr.columns
                    else (
                        "canonical_gene_name"
                        if "canonical_gene_name" in df_expr.columns
                        else None
                    )
                )
                if sym_col is not None:
                    grouped = df_expr.groupby(sym_col)["TPM"].max()
                    sample_tpm_by_symbol = {
                        str(k): float(v) for k, v in grouped.items()
                    }
            evidence_lines = build_candidate_evidence_block(
                candidate_trace,
                sample_tpm_by_symbol,
                top_k=3,
            )
            if evidence_lines:
                lines.extend(evidence_lines)
        except Exception as exc:  # noqa: BLE001 — never let evidence tables break the report
            lines.append(
                f"*Per-candidate evidence tables unavailable: {type(exc).__name__}*"
            )

    # Embedding features. The per-cancer-type / per-normal-tissue reference gene
    # lists are STATIC reference-panel documentation (identical for every sample),
    # so they are collected into an appendix flushed at the end of the report rather
    # than interrupting ~100 lines of sample-specific decision content mid-document.
    embedding_gene_appendix: list[str] = []
    lines.append("## Embedding Features\n")
    lines.append(f"- **Method**: {embedding_meta.get('method', 'unknown')}")
    feature_kind = embedding_meta.get("feature_kind")
    if feature_kind == "hierarchical_scores":
        lines.append(
            f"- **Feature space**: {embedding_meta['n_features']} features built from "
            "family-aware cancer support scores, site/background context, and a purity anchor"
        )
        if embedding_meta.get("families"):
            lines.append(
                f"- **Families represented**: {', '.join(embedding_meta['families'])}"
            )
        if embedding_meta.get("sites"):
            lines.append(
                f"- **Site/background axes**: {', '.join(embedding_meta['sites'][:8])}"
                + (" ..." if len(embedding_meta["sites"]) > 8 else "")
            )
        lines.append(f"- **Cancer types represented**: {embedding_meta['n_types']}/33")
        lines.append("")
        lines.append(
            "The hierarchy embedding uses the same broad-family gating and subtype "
            "support logic as the main classifier, while adding host/background "
            "context axes. The 2D embedding therefore reflects the same evidence "
            "hierarchy shown in the candidate trace rather than a flat gene-only space."
        )
    elif feature_kind == "pan_reference_genes":
        lines.append(f"- **Total genes**: {embedding_meta['n_genes']}")
        lines.append(f"- **Cancer types represented**: {embedding_meta['n_types']}/33")
        lines.append(
            f"- **Normal tissues represented**: {embedding_meta.get('n_normals', 0)}"
        )
        lines.append(
            f"- **Selection density**: {embedding_meta.get('n_genes_per_type', 'NA')} "
            "genes per TCGA cancer type; "
            f"{embedding_meta.get('n_genes_per_normal', 'NA')} per normal tissue"
        )
        if embedding_meta.get("anchor_added"):
            lines.append(
                f"- **Curated tissue anchors added**: "
                f"{', '.join(embedding_meta['anchor_added'])}"
            )
        lines.append("")
        lines.append(
            "The pan-reference embedding is a shared reference-map gene set: it "
            "selects discriminating genes for TCGA cancer medians and HPA normal "
            "tissues, then scales cancers, subtype references, normals, and the "
            "sample in one robust log-expression space. It is intended for visual "
            "orientation, not as the cancer-call classifier."
        )
        lines.append("")
        lines.append(
            "Per-cancer-type and per-normal-tissue reference gene lists are tabulated "
            "in the **Reference-panel gene sets** appendix at the end of this report."
        )
        embedding_gene_appendix.append("### Genes per cancer type\n")
        embedding_gene_appendix.append("| Cancer | Genes |")
        embedding_gene_appendix.append("|--------|-------|")
        for ct in sorted(embedding_meta["per_type"]):
            genes = embedding_meta["per_type"][ct]
            if genes:
                embedding_gene_appendix.append(f"| {ct} | {', '.join(genes)} |")
        embedding_gene_appendix.append("")
        embedding_gene_appendix.append("### Genes per normal tissue\n")
        embedding_gene_appendix.append("| Tissue | Genes |")
        embedding_gene_appendix.append("|--------|-------|")
        for tissue in sorted(embedding_meta.get("per_normal", {})):
            genes = embedding_meta["per_normal"][tissue]
            if genes:
                embedding_gene_appendix.append(f"| {tissue} | {', '.join(genes)} |")
    else:
        lines.append(f"- **Total genes**: {embedding_meta['n_genes']}")
        lines.append(f"- **Cancer types represented**: {embedding_meta['n_types']}/33")
        if embedding_meta.get("fallback_types"):
            lines.append(
                f"- **Fallback types** (z-score only, no S/N filter): "
                f"{', '.join(embedding_meta['fallback_types'])}"
            )
        if embedding_meta.get("cta_added"):
            lines.append(
                f"- **Curated CTAs added**: {', '.join(embedding_meta['cta_added'])}"
            )
        lines.append("")
        lines.append(
            "Per-cancer-type reference gene lists are tabulated in the "
            "**Reference-panel gene sets** appendix at the end of this report."
        )
        embedding_gene_appendix.append("### Genes per cancer type\n")
        embedding_gene_appendix.append("| Cancer | Genes |")
        embedding_gene_appendix.append("|--------|-------|")
        for ct in sorted(embedding_meta["per_type"]):
            genes = embedding_meta["per_type"][ct]
            if genes:
                embedding_gene_appendix.append(f"| {ct} | {', '.join(genes)} |")
    lines.append("")

    # Purity / composition
    lines.append(f"## {_purity_metric_label(sample_mode).title()}\n")
    # #109: compute the sample-level confidence tier once and surface it
    # inline so readers see a 19–100% CI render visibly different from a
    # 58–70% CI (#79). The tier consumes purity CI width, point estimate,
    # degradation severity, and sample-context flags.
    from .confidence import purity_confidence_for_analysis

    # Single source shared with the ReportView snapshot (report_view.py) so a
    # report can never show one purity tier in a figure and a different one in
    # text.
    purity_tier = purity_confidence_for_analysis(analysis)
    analysis["purity_confidence"] = purity_tier
    # Sample-level low-purity flag rides along on ranges_df so every tumor-source TPM
    # cell (via reporting.tumor_attribution_context) carries the caveat inline, not just
    # the summary caveats block. ranges_df is not reassigned after it is built, so the
    # column persists to every downstream renderer.
    if ranges_df is not None and len(ranges_df) > 0:
        from .confidence import sample_purity_is_low

        ranges_df["sample_low_purity"] = bool(sample_purity_is_low(purity_tier))
    tier_note = str(getattr(purity_tier, "inline_note", "") or "")
    tier_note = tier_note.replace("purity CI", "purity range")
    tier_note = tier_note.replace("CI", "range")
    if purity_tier.tier == "degenerate":
        tier_suffix = f" — **degenerate range**: {tier_note}"
    elif purity_tier.tier in {"low", "moderate"} and purity_tier.reasons:
        tier_suffix = f" — **{purity_tier.tier} confidence** ({tier_note})"
    else:
        tier_suffix = ""
    lines.append(
        "- **Overall estimate**: "
        f"{_format_purity_interval(purity.get('overall_estimate'), purity.get('overall_lower'), purity.get('overall_upper'))}"
        f"{tier_suffix}"
    )
    components = purity.get("components", {})
    for comp_name in ("stromal", "immune"):
        comp = components.get(comp_name, {})
        if isinstance(comp, dict):
            enrichment = comp.get("enrichment", 0)
            lines.append(
                f"- **{comp_name.title()}** enrichment: {render_fold(enrichment)} "
                f"vs {reference_cancer_code} broad reference"
            )
    integration = components.get("integration", {})
    if integration.get("signature_deprioritized"):
        lines.append(
            "- **Integration note**: the tumor-specific signature panel was weaker and less stable "
            "than the lineage panel, so it was downweighted rather than used as a hard lower bound."
        )
    interval_cap = components.get("decomposition_interval_cap", {})
    if isinstance(interval_cap, dict) and interval_cap:
        lines.append(
            "- **Decomposition interval cap**: upper purity bound constrained from "
            f"{interval_cap.get('original_upper', 0):.0%} to "
            f"{interval_cap.get('constrained_upper', 0):.0%} because the best-fit "
            "decomposition assigns "
            f"{interval_cap.get('modeled_non_tumor_fraction', 0):.0%} of the sample "
            "to non-tumor components."
        )
    if sample_mode == "pure":
        lines.append(
            "- **Interpretation**: in pure-population mode this estimate is a consistency check "
            "against the matched lineage profile, not a bulk admixture fraction."
        )
    elif sample_mode == "heme":
        lines.append(
            "- **Interpretation**: in heme mode this estimate is a malignant-lineage proxy, "
            "not a strict tumor-vs-immune split."
        )

    # Lineage gene narrative
    lineage = components.get("lineage", {})
    lineage_genes = lineage.get("per_gene", [])
    if lineage_genes:
        lines.append("")
        lines.append("### Lineage Gene Calibration\n")
        lines.append(
            "Purity was refined using cancer-type lineage genes — genes with "
            "known high expression in this tumor type and low TME background. "
            "Each gene independently estimates purity by comparing the sample's "
            "HK-normalized expression to the TCGA reference (adjusted for "
            "TCGA cohort purity).\n"
        )

        # Sort genes into clusters
        sorted_genes = sorted(lineage_genes, key=lambda g: g["purity"], reverse=True)
        median_p = lineage.get("purity")

        # Identify retained vs excluded, and split the excluded cluster by
        # whether the gene is actually expressed — a high-TPM low-ratio gene is a
        # reference-scaling outlier, not de-differentiation (#85.2).
        retained, expressed_below_scale, possibly_lost = (
            _classify_lineage_calibration_genes(sorted_genes, median_p)
        )

        # Not found in sample — but distinguish genuinely absent from
        # detected-but-uninformative. The estimator skips lineage genes
        # whose TME bleed-through in the reference exceeds their tumor
        # contribution; those genes are still present in the sample at
        # real TPM, they just can't anchor a lineage purity. The
        # canonical case is SARC-ACTA2: ACTA2 at ~190 TPM in the
        # sample but TME-dominated at the TCGA SARC reference.
        from .tumor_purity import LINEAGE_GENES

        cancer_code_local = purity.get("cancer_type", cancer_code)
        all_lineage = LINEAGE_GENES.get(cancer_code_local, [])
        found_names = {g["gene"] for g in lineage_genes}
        skipped_detected = {
            entry["gene"]: entry for entry in lineage.get("skipped_detected", [])
        }
        not_found = [
            g for g in all_lineage if g not in found_names and g not in skipped_detected
        ]

        lines.append(
            "**Purity est.** per row = an independent purity estimate from that "
            "gene alone: sample expression ÷ the TCGA reference, after TCGA's own "
            "cohort impurity has been deconvolved from the reference. A value of "
            "20% means the sample expresses this gene at 20% of pure-tumor levels.\n"
        )
        lines.append(
            f"**Lineage caveat**: {_lineage_caveat_text(sample_mode, cancer_code_local)}\n"
        )
        lines.append("| Gene | Purity est. | Interpretation |")
        lines.append("|------|------------|----------------|")
        _interp_by_gene = {}
        for g in retained:
            _interp_by_gene[g["gene"]] = "retained — reliable"
        for g in expressed_below_scale:
            _interp_by_gene[g["gene"]] = "expressed — below reference scale"
        for g in possibly_lost:
            _interp_by_gene[g["gene"]] = "likely de-differentiated / low"
        for g in sorted_genes:
            interp = _interp_by_gene.get(g["gene"], "retained — reliable")
            lines.append(f"| {g['gene']} | {g['purity']:.1%} | {interp} |")
        for gene_name, entry in skipped_detected.items():
            s_tpm = entry["sample_tpm"]
            # Keep resolution proportional to magnitude so a 0.3 TPM
            # gene doesn't render as "0 TPM".
            tpm_str = f"{s_tpm:.1f}" if s_tpm < 10 else f"{s_tpm:.0f}"
            lines.append(
                f"| {gene_name} | — | detected {tpm_str} TPM — "
                "uninformative (TME background exceeds tumor contribution "
                "for this cancer type) |"
            )
        for g in not_found:
            lines.append(f"| {g} | — | not detected |")

        lines.append("")

        if retained:
            retained_names = ", ".join(g["gene"] for g in retained)
            if all(
                lineage.get(key) is not None for key in ("purity", "lower", "upper")
            ):
                lines.append(
                    f"**Reliable cluster** ({lineage['purity']:.0%}, "
                    f"IQR {lineage['lower']:.0%}\u2013{lineage['upper']:.0%}): "
                    f"{retained_names}. "
                    "These genes are expressed at levels consistent with their "
                    "TCGA reference, indicating retained lineage identity (does not distinguish tumor cells from normal cells of the same lineage)."
                )
            else:
                lines.append(
                    f"**Reliable cluster**: {retained_names}. "
                    "These genes are expressed at levels consistent with their "
                    "TCGA reference, indicating retained lineage identity (does not distinguish tumor cells from normal cells of the same lineage)."
                )
        if expressed_below_scale:
            exp_names = ", ".join(g["gene"] for g in expressed_below_scale)
            lines.append(
                f"\n**Expressed, below reference scale**: {exp_names}. "
                "These genes are expressed in this sample (see the bulk TPM in the "
                "target tables) but give low per-gene purity ratios against a high "
                "TCGA reference — a reference-scaling effect, not loss of expression. "
                "They are excluded from the purity estimate as scaling outliers, not "
                "read as de-differentiation."
            )
        if possibly_lost:
            lost_names = ", ".join(g["gene"] for g in possibly_lost)
            lines.append(
                f"\n**Possible low / de-differentiated**: {lost_names}. "
                "These genes give much lower purity estimates and are expressed at "
                "low TPM, so the tumor may have reduced or lost expression of these "
                "markers — common in metastatic or treatment-resistant disease. "
                "These are excluded from the purity estimate."
            )
        if not_found:
            lines.append(f"\n**Not detected**: {', '.join(not_found)}.")

    lines.append("")

    # MHC expression
    lines.append("## MHC Expression\n")
    lines.append("| Gene | TPM |")
    lines.append("|------|-----|")
    for gene in ["HLA-A", "HLA-B", "HLA-C", "B2M"]:
        lines.append(f"| {gene} | {mhc1.get(gene, 0):.0f} |")
    mhc2 = analysis.get("mhc2", {})
    if mhc2:
        for gene, val in sorted(mhc2.items()):
            lines.append(f"| {gene} | {val:.0f} |")
    lines.append("")

    # Background / context signatures
    bg_title, bg_intro = _background_section_config(sample_mode)
    lines.append(f"## {bg_title}\n")
    lines.append(bg_intro)
    lines.append("| Reference tissue/program | Score | N genes | Top matching genes |")
    lines.append("|--------------------------|-------|---------|--------------------|")
    for tissue, score, n in top_tissues:
        detail = tissue_score_details.get(tissue, {})
        drivers = _format_tissue_driver_genes(detail.get("drivers"))
        lines.append(f"| {tissue} | {score:.3f} | {n} | {drivers} |")
    lines.append("")

    if (
        decomp_results
        or call_summary.get("site_indeterminate")
        or call_summary.get("hypothesis_display")
    ):
        lines.append(f"## {_decomposition_section_title(sample_mode)}\n")
        if call_summary.get("site_indeterminate"):
            lines.append("Reported site/template call: **indeterminate**.\n")
        elif call_summary.get("reported_site") is not None:
            lines.append(
                f"Reported template/site call: **{call_summary['reported_site']}**"
                + (
                    f" ({call_summary['reported_context']})\n"
                    if call_summary.get("reported_context")
                    else "\n"
                )
            )
        if call_summary.get("site_note"):
            lines.append(call_summary["site_note"] + "\n")
        if len(call_summary.get("hypothesis_display", [])) == 2:
            lines.append(
                "Raw decomposition audit: active report-compatible fit is **"
                + _hypothesis_label(
                    call_summary["hypothesis_display"][0],
                    primary_code=cancer_code,
                    analysis=analysis,
                )
                + "**; retained raw/nearby fit is **"
                + _hypothesis_label(
                    call_summary["hypothesis_display"][1],
                    primary_code=cancer_code,
                    analysis=analysis,
                )
                + "**. This is decomposition context, not a second report label.\n"
            )
        if decomp_results:
            lines.append("| Hypothesis | Use | Score | Tumor fraction | Tissue score | Warnings |")
            lines.append("|------------|-----|-------|----------------|--------------|----------|")
            for row in decomp_results[:6]:
                warnings = "; ".join(row.warnings) if row.warnings else ""
                use = (
                    "adopted for target/background attribution"
                    if row is best_decomp
                    else "audit only"
                )
                lines.append(
                    f"| {_hypothesis_display_label(row, primary_code=cancer_code, analysis=analysis)} | "
                    f"{use} | {row.score:.3f} | {row.purity:.3f} | "
                    f"{row.template_tissue_score:.3f} | {warnings} |"
                )
            lines.append("")

        if best_decomp is not None and not best_decomp.component_trace.empty:
            lines.append("### Best-Fit Components\n")
            lines.append("| Component | Fraction | Marker score | Top markers |")
            lines.append("|-----------|----------|--------------|-------------|")
            # #79: collapse components with < 0.5% fraction into a
            # single summary row so the useful entries are legible.
            trace_df = best_decomp.component_trace
            shown = trace_df[trace_df["fraction"] >= 0.005]
            hidden = trace_df[trace_df["fraction"] < 0.005]
            for _, row in shown.iterrows():
                comp = row["component"]
                marker_score = row["marker_score"]
                score_cell = (
                    f"{marker_score:.3f}"
                    if isinstance(marker_score, (int, float))
                    and marker_score is not None
                    else (marker_score if marker_score else "—")
                )
                top_markers_cell = row["top_markers"]
                # Matched-normal compartments have no decomposition
                # markers by design (see panels.py); annotate rather
                # than leave the empty cell unexplained (#79).
                if str(comp).startswith("matched_normal_") and (
                    not top_markers_cell or str(top_markers_cell).strip() == ""
                ):
                    top_markers_cell = "*matched-normal compartment — fraction derived from lineage-panel estimate (#52), not discriminative markers*"
                    if score_cell in ("—", "", "0.000", "0.0"):
                        score_cell = "n/a"
                lines.append(
                    f"| {comp} | {row['fraction']:.3f} | {score_cell} | {top_markers_cell} |"
                )
            if len(hidden):
                lines.append(
                    f"| *other components* | *{hidden['fraction'].sum():.3f} total across "
                    f"{len(hidden)} components < 0.5%* | — | — |"
                )
            lines.append("")

    if ranges_df is not None:
        try:
            from .brief import build_actionable

            therapy_md = build_actionable(
                analysis,
                ranges_df,
                cancer_code=cancer_code,
                disease_state=disease_state_display,
                sample_id=sample_id,
            )
            therapy_section = _extract_markdown_section(
                therapy_md,
                "## Therapy Prioritization",
            )
            if therapy_section:
                lines.append(therapy_section)
                lines.append("")
        except Exception as exc:
            print(f"[warn] Could not fold therapy landscape into analysis.md: {exc}")

        lines.append(
            "*Stepwise deductions and the full target tables live in `*-evidence.md`.*"
        )

    # Flush the static reference-panel gene tables as an appendix at the very end,
    # after all sample-specific decision content (see the Embedding Features note).
    if embedding_gene_appendix:
        lines.append("")
        lines.append("## Appendix: reference-panel gene sets\n")
        lines.append(
            "*Static embedding reference-map gene selection — the discriminating "
            "genes chosen for TCGA cancer medians and HPA normal tissues. Identical "
            "for every sample; included for provenance, not sample-specific "
            "interpretation.*\n"
        )
        lines.extend(embedding_gene_appendix)

    analysis_path = "%s-analysis.md" % prefix if prefix else "analysis.md"
    with open(analysis_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[report] Saved {analysis_path}")


def _build_target_report(
    ranges_df,
    analysis,
    cancer_type,
    purity_result,
    decomp_results=None,
):
    """Return tumor-expression range report using purity/decomposition bounds."""
    import pandas as pd

    cancer_type_context = cancer_type_context_from_analysis(analysis)
    cancer_code = cancer_type
    cancer_name = cancer_code_display_name(
        cancer_code,
        CANCER_TYPE_NAMES.get(cancer_code, cancer_code),
    )
    reference_cancer_code = cancer_type_context.code_for("reference") or cancer_code
    reference_cancer_name = cancer_code_display_name(
        reference_cancer_code,
        CANCER_TYPE_NAMES.get(reference_cancer_code, reference_cancer_code),
    )
    sample_mode = analysis.get("sample_mode", "solid")
    value_label = _target_value_label(sample_mode)
    p_lo = purity_result["overall_lower"]
    p_mid = purity_result["overall_estimate"]
    p_hi = purity_result["overall_upper"]

    lines = [f"# Therapeutic Target Analysis — {cancer_code} ({cancer_name})\n"]
    lines.append(_target_report_mode_intro(sample_mode, cancer_code, p_lo, p_mid, p_hi))
    if reference_cancer_code != cancer_code:
        expr_ref_code = cancer_type_context.code_for("expression")
        expr_ref_source = ""
        expr_ref_kind = ""
        if (
            expr_ref_code
            and ranges_df is not None
            and len(ranges_df) > 0
            and "expression_reference_code" in ranges_df.columns
        ):
            expr_rows = ranges_df[
                ranges_df["expression_reference_code"].astype(str).eq(expr_ref_code)
            ]
            if not expr_rows.empty and "expression_reference_source" in expr_rows.columns:
                sources = [
                    str(value).strip()
                    for value in expr_rows["expression_reference_source"].dropna().unique()
                    if str(value).strip()
                ]
                expr_ref_source = sources[0] if sources else ""
            if not expr_rows.empty and "expression_reference_kind" in expr_rows.columns:
                kinds = [
                    str(value).strip()
                    for value in expr_rows["expression_reference_kind"].dropna().unique()
                    if str(value).strip()
                ]
                expr_ref_kind = kinds[0] if kinds else ""
        if (
            expr_ref_code
            and expr_ref_code != reference_cancer_code
            and (expr_ref_source or cancer_type_context.fine_expression_available)
        ):
            source_label = (
                expr_ref_source
                or ", ".join(cancer_type_context.fine_expression_sources)
                or "exact expression"
            )
            kind_text = (
                expr_ref_kind
                or cancer_type_context.best_expression_source_kind
                or "expression_reference"
            )
            reference_kind = {
                "deconvolved_tumor_reference": "deconvolved tumor-cell",
                "observed_bulk_reference": "observed clean-TPM",
                "observed_pan_cancer_reference": "observed clean-TPM",
            }.get(kind_text, "clean-TPM")
            expression_clause = (
                f" Tumor-expression priors use the exact {expr_ref_code} "
                f"{reference_kind} reference ({source_label}) where genes are present; "
                f"sample tumor-specific expression is still estimated from this "
                f"sample's clean TPM, purity, and non-tumor attribution. Genes absent "
                f"from that exact reference and broad-cohort "
                f"percentiles use fallback {reference_cancer_code}."
            )
        else:
            expression_clause = (
                f" Tumor-expression priors and broad-cohort percentiles use fallback "
                f"{reference_cancer_code}."
            )
        lines.append(
            f"> **RNA reference context**: exact {cancer_code} expression is preferred "
            "where supported; decomposition, purity, and broad-cohort context use "
            f"fallback {reference_cancer_code} ({reference_cancer_name}) because this "
            f"stage currently needs a broad pan-reference cohort while {cancer_code} is "
            "the refined/registry report label. Curated therapy rows below remain "
            f"keyed to {cancer_code}.{expression_clause}\n"
        )

    # Low-purity TME-inflation caveat (#35). Below 20% purity, every
    # residual TPM is amplified ≥5× by the tumor-value division.
    # Combined with incomplete TME subtraction, this can rank classic
    # stromal / ECM genes (FN1, COL1A1/2, DCN) as high-expressing tumor
    # markers. Users must read the caveat before interpreting the CAR-T
    # / ADC / radioligand target tables.
    if p_mid is not None and p_mid < 0.20:
        lines.append(
            f"> **Low-purity caveat**: estimated purity is "
            f"**{p_mid:.0%}**, so residual TPM is divided by a small "
            "number and amplified ≥5×. Genes heavily expressed in "
            "fibroblast / endothelial / immune compartments (FN1, "
            "COL1A1/2, DCN, etc.) can appear high in context-expression "
            "even when most of the signal is stromal. Treat the "
            "`tme_explainable=true` column in the TSV as the primary "
            "filter for therapy-target safety; cross-check any "
            "`median_est` > 30 TPM against `tme_fold_med` before acting.\n"
        )

    # Sample-context caveat (step 1 propagation): when the sample was
    # flagged as degraded / FFPE, every marker-selection step down the
    # pipeline is noisier and the target medians are correspondingly
    # less reliable.
    sample_context = analysis.get("sample_context")
    if sample_context is not None and sample_context.is_degraded:
        index_str = (
            f" (length-pair index {sample_context.degradation_index:.2f})"
            if sample_context.degradation_index is not None
            else ""
        )
        lines.append(
            f"> **Degradation caveat**: sample flagged as "
            f"`{sample_context.degradation_severity}` degradation"
            f"{index_str}. Long-transcript TPMs are under-represented; "
            "tumor-expression estimates for long-gene targets carry "
            "higher uncertainty than the reported ranges suggest.\n"
        )
    if sample_mode == "pure":
        lines.append(
            "Each gene is reported as a bounded expression estimate around the observed sample value, "
            f"then contextualized against fallback {reference_cancer_code} broad-reference context where an exact reference is not used.\n"
        )
    elif sample_mode == "heme":
        lines.append(
            "Each gene is reported as a bounded malignant-lineage-enriched expression estimate across "
            f"hematopoietic background assumptions, then contextualized against fallback {reference_cancer_code} "
            "broad-reference context where an exact reference is not used.\n"
        )
    else:
        lines.append(
            "Each gene is reported as a bounded deconvolution across purity and "
            "TME-background assumptions, then contextualized against fallback "
            f"{reference_cancer_code} broad-reference context where an exact reference is not used.\n"
        )
    lines.append(tpm_semantics_note() + "\n")

    def _flag_series(df, column):
        if column in df.columns:
            return df[column].fillna(False).astype(bool)
        return pd.Series(False, index=df.index)

    def _format_target_stub(row, *, include_tcga=False):
        source = tumor_attribution_context(row)
        normal = normal_expression_context(row)
        parts = [tumor_attribution_band_text(row), normal["label"]]
        therapies = str(row.get("therapies") or "").strip()
        if therapies:
            parts.append(therapies)
        if include_tcga and pd.notna(row.get("tcga_percentile")):
            parts.append(f"TCGA {row['tcga_percentile']:.0%}")
        if source["tier"] != "tumor_supported":
            parts.append(source["label"])
        if source.get("notes"):
            parts.append(source["notes"][0])
        elif normal.get("details"):
            parts.append(normal["details"][0])
        return f"{row['symbol']} ({'; '.join(parts)})"

    def _target_interpretation_cell(
        target_row, expr_row, *, include_maturity=False, target_panel=None
    ):
        if expr_row is None:
            if target_row is not None and expression_independent_indication(target_row):
                parts = [
                    expression_independent_interpretation(target_row),
                    expression_independent_rna_context(None),
                ]
            else:
                sym_for_state = (
                    str(target_row.get("symbol") or "") if target_row is not None else ""
                )
                state = target_observation_state(sym_for_state, ranges_df)
                parts = [format_missing_observation_interp(state)]
            if target_row is not None:
                path_context = therapy_path_context(
                    target_row,
                    analysis=analysis,
                    disease_state=disease_state,
                )
                if path_context:
                    parts.append(path_context)
                conflict = therapy_rna_context_conflict(
                    target_row,
                    analysis=analysis,
                    disease_state=disease_state,
                )
                if conflict:
                    parts.append(conflict)
                state_caution = therapy_state_caution(
                    target_row,
                    analysis=analysis,
                    disease_state=disease_state,
                )
                if state_caution:
                    parts.append(f"current-therapy check: {state_caution}")
            return "; ".join(part for part in parts if part)
        if include_maturity:
            return target_interpretation_summary(
                target_row,
                expr_row,
                target_panel=target_panel,
                analysis=analysis,
                disease_state=disease_state,
            )
        source = tumor_attribution_context(expr_row)
        normal = normal_expression_context(expr_row)
        expr_independent = target_row is not None and expression_independent_indication(
            target_row
        )
        if expr_independent:
            parts = [
                expression_independent_interpretation(target_row),
                expression_independent_rna_context(expr_row),
            ]
        else:
            parts = [source["label"], normal["label"]]
        if expr_independent:
            pass
        elif source.get("notes"):
            parts.append(source["notes"][0])
        elif normal.get("details"):
            parts.append(normal["details"][0])
        if target_row is not None:
            path_context = therapy_path_context(
                target_row,
                analysis=analysis,
                disease_state=disease_state,
            )
            if path_context:
                parts.append(path_context)
            conflict = therapy_rna_context_conflict(
                target_row,
                analysis=analysis,
                disease_state=disease_state,
            )
            if conflict:
                parts.append(conflict)
            state_caution = therapy_state_caution(
                target_row,
                analysis=analysis,
                disease_state=disease_state,
            )
            if state_caution:
                parts.append(f"current-therapy check: {state_caution}")
        return "; ".join(part for part in parts if part)

    def _low_purity_cap_audit_md(target_symbols):
        if not target_symbols or ranges_df is None or len(ranges_df) == 0:
            return []
        if "symbol" not in ranges_df.columns:
            return []
        if (
            "low_purity_cap_applied" not in ranges_df.columns
            and "tumor_attributed_bulk_tpm_pre_low_purity_cap" not in ranges_df.columns
        ):
            return []

        def _num(value, default=0.0):
            try:
                if value is None or pd.isna(value):
                    return default
                return float(value)
            except Exception:
                return default

        target_set = {str(sym).strip() for sym in target_symbols if str(sym).strip()}
        sub = ranges_df[ranges_df["symbol"].astype(str).isin(target_set)].copy()
        if len(sub) == 0:
            return []
        sub["_cap_applied"] = _flag_series(sub, "low_purity_cap_applied").values
        sub["_observed"] = sub["observed_tpm"].map(_num)
        sub["_post_cap"] = sub.get(
            "tumor_attributed_bulk_tpm", sub.get("attr_tumor_tpm", 0.0)
        ).map(_num)
        if "tumor_attributed_bulk_tpm_pre_low_purity_cap" in sub.columns:
            sub["_pre_cap"] = sub[
                "tumor_attributed_bulk_tpm_pre_low_purity_cap"
            ].map(_num)
        else:
            sub["_pre_cap"] = sub["_post_cap"]
        if "low_purity_cap_tpm" in sub.columns:
            sub["_cap_tpm"] = sub["low_purity_cap_tpm"].map(lambda v: _num(v, None))
        else:
            sub["_cap_tpm"] = None

        sub = sub.sort_values(
            ["_cap_applied", "_observed", "symbol"],
            ascending=[False, False, True],
        ).head(12)
        if len(sub) == 0:
            return []

        capped_n = int(sub["_cap_applied"].sum())
        lines_out = [
            "### Low-purity cap audit\n",
            "When purity is low, the attribution model caps tumor-attributed "
            "bulk TPM at `observed bulk TPM × purity × headroom`. Cap status is "
            "tracked across the purity interval: a row can have cap activity in "
            "a low-purity scenario even when the median post-cap value is unchanged. "
            "The full audit is in `*-tumor-expression-ranges.tsv`.\n",
            f"*Shown for therapy-panel genes; {capped_n} of {len(sub)} shown rows have cap activity in at least one purity scenario.*\n",
            "| Gene | Bulk TPM | Pre-cap tumor-source bulk TPM | Post-cap tumor-source bulk TPM | Cap ceiling range | Status |",
            "|------|---------:|-------------------------------:|--------------------------------:|------------------:|--------|",
        ]
        for _, row in sub.iterrows():
            cap = row.get("_cap_tpm")
            cap_lo = _num(row.get("low_purity_cap_tpm_low"), None)
            cap_hi = _num(row.get("low_purity_cap_tpm_high"), None)

            def _has_number(value):
                return value is not None and not pd.isna(value)

            if (
                _has_number(cap_lo)
                and _has_number(cap_hi)
                and abs(float(cap_hi) - float(cap_lo)) > 0.05
            ):
                cap_cell = f"{float(cap_lo):.1f}-{float(cap_hi):.1f}"
            elif _has_number(cap):
                cap_cell = f"{float(cap):.1f}"
            else:
                cap_cell = "—"
            delta = max(0.0, float(row["_pre_cap"]) - float(row["_post_cap"]))
            if bool(row.get("_cap_applied")) and delta > 0.05:
                status = "median capped upper bound"
            elif bool(row.get("_cap_applied")):
                status = "cap active in interval"
            else:
                status = "fit below cap / not capped"
            lines_out.append(
                f"| {row['symbol']} | {float(row['_observed']):.1f} | "
                f"{float(row['_pre_cap']):.1f} | {float(row['_post_cap']):.1f} | "
                f"{cap_cell} | {status} |"
            )
        lines_out.append("")
        return lines_out

    def _prad_steap_contrast_md(sym_to_row, targets_df=None):
        if cancer_code != "PRAD":
            return []
        steap_rows = {
            sym: sym_to_row.get(sym)
            for sym in ("STEAP1", "STEAP2")
            if sym_to_row.get(sym) is not None
        }
        if len(steap_rows) < 2:
            return []

        def _agents(sym):
            if targets_df is None or len(targets_df) == 0:
                return "—"
            if "symbol" not in targets_df.columns:
                return "—"
            rows = targets_df[targets_df["symbol"].astype(str) == sym]
            labels = []
            for _, target_row in rows.iterrows():
                agent = str(target_row.get("agent") or "").strip()
                phase = str(target_row.get("phase") or "").replace("_", " ").strip()
                if agent and agent.lower() != "nan":
                    labels.append(f"{agent} ({phase})" if phase else agent)
            return "; ".join(labels) if labels else "—"

        lines_out = [
            "### STEAP1 / STEAP2 contrast\n",
            "These prostate-lineage targets are interpreted separately because "
            "they can diverge under treatment pressure and low purity. A lower "
            "STEAP1 tumor-source estimate should not be collapsed into the "
            "STEAP2 result, and a capped STEAP2 estimate should be read as an "
            "upper-bound expression anchor rather than a precise fitted value.\n",
            "| Target | Bulk TPM | Tumor-source bulk TPM | Context TPM | Tumor source | Cap status | Curated option |",
            "|--------|---------:|----------------------:|------------:|--------------|------------|----------------|",
        ]
        for sym in ("STEAP1", "STEAP2"):
            row = steap_rows[sym]
            source = tumor_attribution_context(row)
            obs = float(row.get("observed_tpm") or 0.0)
            cap_status = (
                "capped upper bound"
                if bool(row.get("low_purity_cap_applied"))
                else "not capped"
            )
            lines_out.append(
                f"| {sym} | {render_tpm(obs)} | {tumor_band_cell(row)} | "
                f"{context_expression_band_cell(row)} | "
                f"{source['label']} ({source['attr_tumor_fraction']:.0%} tumor bulk fraction) | "
                f"{cap_status} | {_agents(sym)} |"
            )
        lines_out.append(
            "\nPractical reading: prioritize the therapy row by both clinical "
            "maturity and tumor-source support. A more mature STEAP1 therapy "
            "can still be less sample-supported than STEAP2 if STEAP1 is "
            "background-dominant or de-differentiated in this specimen.\n"
        )
        return lines_out

    therapy_scores = analysis.get("therapy_response_scores") or {}
    ts_to_show = [
        (cls, s)
        for cls, s in therapy_scores.items()
        if s.state in ("up", "down") and s.per_gene
    ]
    call_summary = analysis.get("call_summary") or _summarize_sample_call(
        analysis,
        decomp_results or [],
        sample_mode=sample_mode,
    )
    fit_quality = analysis.get("fit_quality", {})
    family_display = (analysis.get("family_summary") or {}).get("display")
    disease_state = report_disease_state_text(
        compose_disease_state_narrative(analysis),
        analysis=analysis,
    )
    matched_normal_summary = _matched_normal_split_summary(ranges_df)
    mhc_status_label, mhc_status_text = _mhc1_status_text(analysis.get("mhc1"))
    panel_target_symbols = set()

    # #110: cancer-type-scoped biomarker panel + therapy landscape.
    # Emitted before the general surface/intracellular tables so a
    # clinician reading top-down sees the curated, decision-relevant
    # rows first. Silently omitted for cancer types that aren't yet
    # curated in ``data/cancer-key-genes.csv`` — that's explicit, not
    # a fallback to the general tables.
    try:
        from pirlygenes.gene_sets_cancer import (
            cancer_biomarker_genes, cancer_therapy_targets, cancer_key_genes_cancer_types,
        )

        panel_code, panel_subtype = cancer_key_genes_lookup_for_analysis(
            cancer_code,
            analysis,
            ranges_df=ranges_df,
        )
        panel_display = (
            f"{panel_code} ({str(panel_subtype).replace('_', ' ')})"
            if panel_subtype
            else panel_code
        )
        if panel_code in cancer_key_genes_cancer_types():
            # ID-based lookup with symbol fallback: biomarker panel
            # comes in as symbols (curated by cancer_biomarker_genes()),
            # we resolve to Ensembl IDs once and query the ID-keyed
            # ranges view. Falls back to symbol view for legacy
            # ranges_df frames without gene_id (tests, etc.).
            from .common import (
                ranges_by_gene_id,
                ranges_by_symbol,
                panel_symbols_to_gene_ids,
            )
            biomarker_syms_for_lookup = cancer_biomarker_genes(
                panel_code, subtype=panel_subtype
            ) if panel_subtype else cancer_biomarker_genes(panel_code)
            _panel_sym_to_id = panel_symbols_to_gene_ids(biomarker_syms_for_lookup)
            _id_to_row = ranges_by_gene_id(ranges_df)
            _sym_to_row_fallback = ranges_by_symbol(ranges_df)

            class _IdFirstLookup:
                """Symbol-keyed adapter that internally resolves via gene_id."""
                def get(self, sym, default=None):
                    gid = _panel_sym_to_id.get(str(sym).strip())
                    row = _id_to_row.get(gid) if gid else None
                    if row is None:
                        row = _sym_to_row_fallback.get(sym, default)
                    return row if row is not None else default

            sym_to_row = _IdFirstLookup()

            lines.append(f"## Biomarker Panel — {panel_display}\n")
            lines.append(
                "Clinician-relevant biomarkers for this cancer type: "
                "lineage confirmation, disease-state indicators, and "
                "diagnostic markers. **Not therapy targets** — see the "
                "Therapy Landscape below for druggable genes.\n"
            )
            if panel_code != cancer_code or panel_subtype:
                lines.append(
                    "*Curation scope:* "
                    + subtype_curation_scope_note(
                        panel_code,
                        panel_subtype=panel_subtype,
                        base_code=cancer_code,
                        base_name=analysis.get("cancer_name") or cancer_code,
                        noun="biomarker and therapy curation",
                    )
                    + "\n"
                )
            biomarker_syms = (
                cancer_biomarker_genes(panel_code, subtype=panel_subtype)
                if panel_subtype
                else cancer_biomarker_genes(panel_code)
            )
            if biomarker_syms:
                lines.append(
                    "| Gene | Bulk TPM (measured) | Tumor-source bulk TPM (model) | Context TPM (model) | Attribution |"
                )
                lines.append(
                    "|------|---------------------|-------------------------------|---------------------|-------------|"
                )
                for raw_sym in biomarker_syms:
                    sym = canonical_target_symbol(raw_sym)
                    row = sym_to_row.get(sym)
                    if row is None:
                        missing_cell = format_missing_observation_cell(
                            target_observation_state(sym, ranges_df)
                        )
                        lines.append(f"| {sym} | {missing_cell} | — | — | — |")
                        continue
                    obs = float(row.get("observed_tpm") or 0.0)
                    attribution_cell = _format_attribution_cell(row)
                    tumor_source = tumor_band_cell(row)
                    context_cell = context_expression_band_cell(row)
                    lines.append(
                        f"| {sym} | {render_tpm(obs)} | {tumor_source} | {context_cell} | "
                        f"{attribution_cell} |"
                    )
                lines.append("")
            else:
                lines.append("*No biomarker genes curated for this cancer type.*\n")

            lines.append(f"## Therapy Target Landscape — {panel_display}\n")
            lines.append(
                "Approved and trialed agents with an indication for this "
                f"{cancer_code_display_name(panel_code, panel_display)}, "
                "cross-referenced against sample expression. Rows where the "
                "target is absent from the sample are still shown to make "
                "that explicit.\n"
            )
            lines.append(tpm_semantics_note() + "\n")
            lines.append(
                "Interpretation separates **tumor-source support** "
                "(`tumor-supported`, `mixed-source`, `background-dominant`) "
                "from **normal-expression context** "
                "(`same-lineage expected`, `broad healthy expression`, "
                "`vital-tissue concern`, etc.). Phase / class still carry "
                "the clinical-development tier. Treatment-path context flags "
                "standard options, later-line requirements, trial follow-ups, "
                "and possible current/prior therapy exposure.\n"
            )
            lines.append(
                "The broader **actionable expression** figures are discovery "
                "screens: they include surface/ADC/bispecific/radioligand/TCR "
                "targets with expression signal even when the agent is generic, "
                "approved only in another indication, or not disease-matched. "
                "The **priority** list is intentionally narrower: it ranks the "
                "curated cancer-specific therapy landscape by indication fit, "
                "required alteration or HLA evidence, clinical maturity, and "
                "tumor-source attribution. A target such as HER3/ERBB3 or "
                "ADAM9 can therefore appear in the expression screen without "
                "being a priority recommendation for this sample.\n"
            )
            targets_df = (
                cancer_therapy_targets(panel_code, subtype=panel_subtype)
                if panel_subtype
                else cancer_therapy_targets(panel_code)
            )
            targets_df = filter_current_therapy_targets(targets_df)
            panel_target_symbols = {
                canon
                for canon in (
                    canonical_target_symbol(sym)
                    for sym in targets_df.get("symbol", pd.Series(dtype=object)).dropna()
                )
                if canon and canon.lower() != "nan"
            }
            if len(targets_df):
                # Approved first, then phase_3, phase_2, phase_1,
                # preclinical. Within phase, agent name for stability.
                phase_order = {
                    "approved": 0,
                    "phase_3": 1,
                    "phase_2": 2,
                    "phase_1": 3,
                    "preclinical": 4,
                }
                targets_sorted = targets_df.assign(
                    _inactive_key=[
                        1 if therapy_row_rna_context_inactive(
                            trow,
                            analysis=analysis,
                            disease_state=disease_state,
                        ) else 0
                        for _, trow in targets_df.iterrows()
                    ],
                    _path_key=[
                        therapy_path_rank(
                            trow,
                            analysis=analysis,
                            disease_state=disease_state,
                        )
                        for _, trow in targets_df.iterrows()
                    ],
                    _phase_key=targets_df["phase"].map(
                        lambda p: phase_order.get(str(p), 99)
                    ),
                ).sort_values(
                    ["_inactive_key", "_path_key", "_phase_key", "symbol", "agent"]
                )

                def _cell(value):
                    if value is None:
                        return "—"
                    s = str(value).strip()
                    if s == "" or s.lower() == "nan":
                        return "—"
                    return s

                def _target_display_record(trow):
                    sym = canonical_target_symbol(_cell(trow.get("symbol")))
                    agent = _cell(trow.get("agent"))
                    agent_class = _cell(trow.get("agent_class"))
                    phase = _cell(trow.get("phase")).replace("_", " ")
                    indication = _cell(trow.get("indication"))
                    expr = None if sym == "—" else sym_to_row.get(sym)
                    reliability = "provisional"
                    if sym == "—":
                        obs_cell = "*not measured*"
                        tumor_source_cell = "—"
                        context_cell = "—"
                        attr_cell = "—"
                        interpretation_cell = "agent-only / no direct gene target"
                    elif expr is None:
                        obs_state = target_observation_state(sym, ranges_df)
                        obs_cell = format_missing_observation_cell(obs_state)
                        tumor_source_cell = "—"
                        context_cell = "—"
                        attr_cell = "—"
                        interpretation_cell = format_missing_observation_interp(
                            obs_state
                        )
                        reliability = (
                            "provisional"
                            if expression_independent_indication(trow)
                            else "unsupported"
                        )
                    else:
                        # Unify bulk-TPM formatting with the Surface Protein /
                        # target-landscape tables (render_tpm drops the decimal
                        # at >=100 TPM) so a gene like EGFR/IGF1R reads the same
                        # value in every section (#85 minor).
                        obs_cell = render_tpm(expr.get("observed_tpm") or 0.0)
                        tumor_source_cell = tumor_band_cell(expr)
                        context_cell = context_expression_band_cell(expr)
                        attr_cell = _format_attribution_cell(expr)
                        interpretation_cell = _target_interpretation_cell(
                            trow,
                            expr,
                            target_panel=targets_df,
                        )
                        reliability = target_reliability_status(
                            expr,
                            target_row=trow,
                        )
                    audit_only = (
                        reliability == "unsupported"
                        and not expression_independent_indication(trow)
                    )
                    if audit_only:
                        interpretation_cell = (
                            "audit-only negative/background evidence; "
                            + interpretation_cell
                        )
                    return {
                        "sym": sym,
                        "agent": agent,
                        "agent_class": agent_class,
                        "phase": phase,
                        "indication": indication,
                        "obs_cell": obs_cell,
                        "tumor_source_cell": tumor_source_cell,
                        "context_cell": context_cell,
                        "attr_cell": attr_cell,
                        "interpretation_cell": interpretation_cell,
                        "audit_only": audit_only,
                    }

                target_records = [
                    _target_display_record(trow)
                    for _, trow in targets_sorted.iterrows()
                ]
                active_records = [row for row in target_records if not row["audit_only"]]
                audit_records = [row for row in target_records if row["audit_only"]]

                def _render_target_records(records):
                    lines.append(
                        "| Target | Agent | Class | Phase | Indication | "
                        "Bulk TPM (measured) | Tumor-source bulk TPM (model) | Context TPM (model) | Attribution | Interpretation |"
                    )
                    lines.append(
                        "|--------|-------|-------|-------|------------|"
                        "----------|-------------------------------|---------------------|-------------|----------------|"
                    )
                    for rec in records:
                        bold = (
                            "**"
                            if rec["phase"] == "approved" and rec["sym"] != "—"
                            else ""
                        )
                        lines.append(
                            f"| {bold}{rec['sym']}{bold} | {rec['agent']} | {rec['agent_class']} | "
                            f"{rec['phase']} | {rec['indication']} | {rec['obs_cell']} | "
                            f"{rec['tumor_source_cell']} | {rec['context_cell']} | {rec['attr_cell']} | "
                            f"{rec['interpretation_cell']} |"
                        )
                    lines.append("")

                if audit_records:
                    lines.append("### Sample-supported / clinically reviewable rows\n")
                    if active_records:
                        _render_target_records(active_records)
                    else:
                        lines.append(
                            "*No curated therapy row had tumor-supported or clinically reviewable RNA evidence in this sample.*\n"
                        )
                    lines.append("### Audit-only rows: not tumor-supported in this sample\n")
                    lines.append(
                        "These rows remain visible as disease-curation provenance or negative evidence. "
                        "They should not be read as expression-supported therapeutic opportunities unless "
                        "orthogonal molecular evidence supplies eligibility.\n"
                    )
                    _render_target_records(audit_records)
                else:
                    _render_target_records(active_records)
                lines.extend(_prad_steap_contrast_md(sym_to_row, targets_df))
                lines.extend(_low_purity_cap_audit_md(panel_target_symbols))
            else:
                lines.append(
                    "*No clinician-validated therapy targets curated for "
                    "this cancer type yet. The general Surface / "
                    "Intracellular tables below are ranked by raw "
                    "expression, not by approved indication.*\n"
                )
    except Exception as _lm_err:
        # Never fail the whole report on curation / loader issues.
        print(f"[warn] Could not render cancer-key-genes panel: {_lm_err}")

    # Pre-sort the key target categories so the summary and the full
    # tables agree on what "top" means.
    ctas = (
        ranges_df[ranges_df["is_cta"] & (ranges_df["median_est"] > 0.5)]
        .sort_values("median_est", ascending=False)
        .copy()
    )
    _excluded = ranges_df.get(
        "excluded_from_ranking", pd.Series(False, index=ranges_df.index)
    )
    surface_targets = (
        ranges_df[
            ranges_df["is_surface"]
            & (ranges_df["median_est"] > 1)
            & ~ranges_df["is_cta"]
            & ~_excluded
        ]
        .sort_values("median_est", ascending=False)
        .copy()
    )
    intracellular = (
        ranges_df[
            ~ranges_df["is_surface"]
            & (ranges_df["median_est"] > 5)
            & (ranges_df["category"].isin(["therapy_target", "CTA"]))
            & ~_excluded
        ]
        .sort_values("median_est", ascending=False)
        .copy()
    )

    surface_status = _target_reliability_series(surface_targets)
    intracellular_status = _target_reliability_series(intracellular)

    def _headline_surface_row(row):
        normal = normal_expression_context(row)
        detail_text = " ".join(normal.get("details") or [])
        therapies = str(row.get("therapies") or "").strip()
        if therapies:
            return True
        if normal["tier"] in {"broad_healthy_expression", "vital_tissue_concern"}:
            return False
        if "broader healthy-tissue signal" in detail_text:
            return False
        return True

    headline_surface = (
        surface_targets[surface_targets.apply(_headline_surface_row, axis=1)].copy()
        if len(surface_targets)
        else surface_targets.head(0).copy()
    )
    if len(headline_surface):
        headline_surface["_status"] = surface_status.loc[headline_surface.index].values
        headline_surface["_status_rank"] = (
            headline_surface["_status"]
            .map({"supported": 0, "provisional": 1, "unsupported": 2})
            .fillna(9)
        )
        headline_surface["_therapy_rank"] = (
            headline_surface["therapies"]
            .fillna("")
            .astype(str)
            .map(lambda text: 0 if text.strip() else 1)
        )
        headline_surface["_therapy_count"] = (
            headline_surface["therapies"]
            .fillna("")
            .astype(str)
            .map(
                lambda text: len([piece for piece in text.split(",") if piece.strip()])
            )
        )
        headline_surface["_category_rank"] = (
            headline_surface["category"]
            .fillna("")
            .astype(str)
            .map(lambda text: 0 if text == "therapy_target" else 1)
        )
        headline_surface = headline_surface.sort_values(
            [
                "_status_rank",
                "_therapy_rank",
                "_category_rank",
                "_therapy_count",
                "median_est",
            ],
            ascending=[True, True, True, False, False],
        )
    else:
        headline_surface = headline_surface.copy()
        headline_surface["_status"] = pd.Series(dtype=object)
        headline_surface["_status_rank"] = pd.Series(dtype=float)
    safe_surface = headline_surface[headline_surface["_status"] == "supported"].head(3)
    mixed_surface = headline_surface[headline_surface["_status"] == "provisional"].head(
        3
    )
    extra_surface = headline_surface[
        (headline_surface["_status"] == "supported")
        & ~headline_surface["symbol"].astype(str).isin(panel_target_symbols)
        & ~headline_surface["symbol"]
        .astype(str)
        .isin(safe_surface["symbol"].astype(str))
    ].head(3)
    best_cta = ctas.head(3)
    clean_intracellular = intracellular[intracellular_status == "supported"].head(3)

    lines.append("## Tumor context for interpretation\n")
    if call_summary.get("label_options"):
        active_label = call_summary["label_options"][0]
        lines.append(f"- **Working label**: **{_cancer_label(active_label)}**.")
        if len(call_summary["label_options"]) >= 2:
            lines.append(
                "- **Retained alternatives**: documented in Cancer-Type Differential; "
                "downstream target and biomarker interpretation below uses the working label."
            )
    else:
        lines.append(f"- **Working label**: **{_cancer_label(cancer_code)}**.")
    supplied_discordant = False
    evidence_selected_discordant = False
    candidate_trace = analysis.get("candidate_trace") or []
    if candidate_trace:
        top_code = str(candidate_trace[0].get("code") or "").strip()
        supplied_discordant = (
            analysis.get("cancer_type_source") == "user-specified"
            and top_code
            and top_code != str(cancer_code).strip()
        )
        evidence_selected_discordant = (
            analysis.get("cancer_type_source") == "auto-detected"
            and top_code
            and top_code != str(cancer_code).strip()
            and _selected_report_scope_label(analysis) == str(cancer_code or "").strip()
        )
    if family_display and (supplied_discordant or evidence_selected_discordant):
        source_label = "supplied" if supplied_discordant else "selected"
        lines.append(
            f"- **RNA family signal**: {family_display}; interpret this as "
            "classifier-conflict/background context, not as a replacement for "
            f"the {source_label} {_cancer_label(cancer_code)} label."
        )
    elif family_display:
        lines.append(f"- **Family-level framing**: {family_display}.")
    if fit_quality.get("label"):
        fit_label = "RNA-reference fit quality" if supplied_discordant else "Fit quality"
        fit_line = f"- **{fit_label}**: {fit_quality['label']}"
        if fit_quality.get("message"):
            fit_line += f" — {_strip_terminal_punctuation(fit_quality['message'])}"
        lines.append(fit_line + ".")
    if call_summary.get("site_indeterminate"):
        lines.append(
            "- **Background/site model**: indeterminate; treat this part of the decomposition as provisional."
        )
    elif call_summary.get("reported_site"):
        lines.append(f"- **Background/site model**: {call_summary['reported_site']}.")
    inferred_site = analysis.get("inferred_site_context") or {}
    if inferred_site:
        lines.append(
            "- **Inferred host site**: "
            f"{str(inferred_site.get('site') or '').replace('_', ' ')} "
            f"(background tissue {str(inferred_site.get('tissue') or '').replace('_', ' ')}, "
            f"score {float(inferred_site.get('score') or 0.0):.2f}); this was inferred "
            "from expression, not supplied as a user constraint."
        )
    lines.append(f"- **Analysis mode**: {_sample_mode_display(sample_mode)}.")
    lines.append(
        f"- **{_purity_metric_label(sample_mode).title()}**: "
        f"{_purity_ci_phrase(purity_result)}."
    )
    if disease_state:
        lines.append(f"- **Disease-state synthesis**: {disease_state.rstrip('.')}.")
    lines.append(f"- **MHC-I status**: {mhc_status_text}")
    if matched_normal_summary:
        lines.append(f"- **Matched-normal split**: {matched_normal_summary}")
        lines.append(
            "- **Per-gene provenance**: see TSV `estimation_path` "
            "(`tme_only`, `matched_normal_split`, `clamped`, `tme_fold_fallback`)."
        )

    context_cautions = []
    integration = purity_result.get("components", {}).get("integration", {})
    if integration.get("signature_deprioritized"):
        context_cautions.append(
            "tumor-specific signature evidence was weaker than lineage/background evidence, so purity leaned more on the latter"
        )
    if p_mid is not None and p_mid < 0.20:
        context_cautions.append("low purity amplifies residual host/background signal")
    if sample_context is not None and sample_context.is_degraded:
        context_cautions.append(
            "RNA degradation widens uncertainty for long transcripts"
        )
    if (
        therapy_scores.get("IFN_response") is not None
        and therapy_scores["IFN_response"].state == "up"
    ):
        context_cautions.append(
            "active IFN response can inflate HLA/MHC-class and other interferon-stimulated targets"
        )
    if context_cautions:
        lines.append(f"- **Interpretation caveats**: {'; '.join(context_cautions)}.")
    integrated_bullets = _integrated_evidence_bullets(analysis, decomp_results or [])
    if integrated_bullets:
        lines.append("")
        lines.append("### Integrated evidence synthesis\n")
        lines.extend(integrated_bullets)
    lines.append("")

    lines.append("## Therapy Prioritization at a Glance\n")
    if len(safe_surface):
        lines.append(
            "- **Surface-directed modalities**: "
            + ", ".join(_format_target_stub(row) for _, row in safe_surface.iterrows())
            + "."
        )
    elif len(mixed_surface):
        lines.append(
            "- **Surface-directed modalities**: no surface target stayed tumor-supported across the current range; "
            "best mixed-source options are "
            + ", ".join(_format_target_stub(row) for _, row in mixed_surface.iterrows())
            + "."
        )
    elif len(surface_targets):
        lines.append(
            "- **Surface-directed modalities**: no surface target stayed tumor-supported across the current range; "
            "the leading rows remain mixed-source or background-dominant."
        )
    else:
        lines.append(
            "- **Surface-directed modalities**: no surface target rose above the reporting threshold."
        )
    if len(extra_surface):
        lines.append(
            "- **Additional non-panel surface leads**: "
            + ", ".join(_format_target_stub(row) for _, row in extra_surface.iterrows())
            + "."
        )
    if len(best_cta):
        lines.append(
            "- **CTA / vaccine ideas**: "
            + ", ".join(
                _format_target_stub(row, include_tcga=True)
                for _, row in best_cta.iterrows()
            )
            + "."
        )
    else:
        lines.append(
            "- **CTA / vaccine ideas**: no CTA rose above the reporting threshold."
        )
    if mhc_status_label == "adequate" and len(clean_intracellular):
        lines.append(
            "- **Intracellular / TCR-style ideas**: "
            + ", ".join(
                _format_target_stub(row) for _, row in clean_intracellular.iterrows()
            )
            + "."
        )
    elif mhc_status_label == "adequate" and len(intracellular):
        lines.append(
            "- **Intracellular / TCR-style ideas**: MHC-I is adequate, but the current intracellular rows remain "
            "mixed-source or background-dominant under the current tumor-source range."
        )
    elif len(intracellular):
        lines.append(
            "- **Intracellular / TCR-style ideas**: detectable candidates exist, but reduced antigen presentation makes "
            "surface-directed strategies safer to prioritize first."
        )
    else:
        lines.append(
            "- **Intracellular / TCR-style ideas**: no intracellular target rose above the reporting threshold."
        )

    landscape_cautions = []
    if (
        len(surface_targets)
        and _flag_series(surface_targets.head(10), "tme_dominant").any()
    ):
        landscape_cautions.append(
            "some of the numerically highest surface rows are TME-dominant and should not be treated as tumor-cell targets"
        )
    if len(surface_targets) and (surface_status.head(10) == "provisional").any():
        landscape_cautions.append(
            "several high-signal surface rows remain mixed-source because their tumor contribution is not cleanly separated from benign/background signal"
        )
    if matched_normal_summary:
        landscape_cautions.append("benign parent-tissue admixture is active")
    if (
        "therapy_supported" in ranges_df.columns
        and "therapy_support_note" in ranges_df.columns
    ):
        fn1_blocked = ranges_df[
            ranges_df["symbol"].eq("FN1")
            # Treat missing therapy_supported as True (the common case for
            # rows without any therapy flag set); avoid .fillna on the
            # object-dtype series because pandas 2.x emits a FutureWarning
            # about silent downcasting. Mask + default keeps the intent
            # explicit.
            & ~ranges_df["therapy_supported"].map(
                lambda v: True
                if v is None or (isinstance(v, float) and pd.isna(v))
                else bool(v)
            )
            & ranges_df["therapy_support_note"].fillna("").astype(str).str.len().gt(0)
        ]
        if len(fn1_blocked):
            landscape_cautions.append(
                fn1_blocked.sort_values("observed_tpm", ascending=False).iloc[0][
                    "therapy_support_note"
                ]
            )
    if landscape_cautions:
        cleaned_cautions = [
            _strip_terminal_punctuation(str(item))
            for item in landscape_cautions
            if str(item).strip()
        ]
        lines.append(f"- **Landscape cautions**: {'; '.join(cleaned_cautions)}.")
    lines.append("")

    # Metabolic axes (#158): the tissue-composition screen already computes proliferation /
    # hypoxia / glycolysis channel scores on TumorEvidenceScore but only
    # uses them for the tissue-composition narrative. Surface any meaningfully-
    # elevated metabolic programs here so the therapy report names the
    # axes a reader could act on (CA9-directed ADCs, MCT inhibitors,
    # CDK4/6). Emits nothing when no channel clears the threshold.
    metabolic_rows = _metabolic_axes_rows(
        getattr(analysis.get("healthy_vs_tumor"), "evidence", None)
    )
    if metabolic_rows:
        lines.append("## Metabolic axes\n")
        lines.append(
            "Metabolic programs active in tissue-composition scoring (proliferation / "
            "hypoxia / glycolysis). Rows emit only when the corresponding "
            "channel is meaningfully elevated.\n"
        )
        lines.append("| Axis | Signal | Therapy considerations |")
        lines.append("|------|--------|------------------------|")
        for axis, signal, therapy in metabolic_rows:
            lines.append(f"| {axis} | {signal} | {therapy} |")
        lines.append("")

    # Pathway and treatment-state signals (#57): enumerate the chain of evidence
    # behind any active / suppressed signaling axis so the reader can
    # see *why* specific genes are off-baseline rather than having to
    # reconstruct the pattern from per-gene tables.
    if ts_to_show:
        lines.append("## Pathway and Treatment-State Signals\n")
        lines.append(
            "Cohort-referenced fold changes across curated signaling-axis "
            "panels (#57). Interpretation: a ≥2× up-panel elevation "
            "signals active signaling; a ≤0.5× up-panel drop with elevated "
            "down-panel genes signals therapy exposure (e.g. ADT in PRAD).\n"
        )
        # Render one axis: bold header + the top-8 most-divergent per-gene rows
        # (concrete evidence chain without overflowing the report).
        def _emit_axis(s, label):
            verb = "active" if s.state == "up" else "suppressed"
            lines.append(f"**{label}** — {verb}. {s.message}\n")
            entries = sorted(
                s.per_gene,
                key=lambda e: abs((e["fold_vs_cohort"] or 1.0) - 1.0),
                reverse=True,
            )[:8]
            lines.append(
                "| Gene | Direction | Sample TPM | Cohort median | Fold | Mechanism |"
            )
            lines.append(
                "|------|-----------|------------|---------------|------|-----------|"
            )
            for e in entries:
                lines.append(
                    f"| {e['symbol']} | {e['direction']} | "
                    f"{e['sample_tpm']:.1f} | {e['cohort_median']:.1f} | "
                    f"{e['fold_vs_cohort']:.2f}× | {e['mechanism']} |"
                )
            lines.append("")

        # The aPD1_* axes are not separate agents — they are distinct response /
        # resistance sub-programs under ONE checkpoint marker. Group them so the
        # "aPD1" prefix isn't repeated on every line, and state the regimen
        # implication once (monotherapy vs combination) rather than per axis.
        _CHECKPOINT_PREFIX = "aPD1_"
        checkpoint_axes = [
            (cls, s) for cls, s in ts_to_show if cls.startswith(_CHECKPOINT_PREFIX)
        ]
        for cls, s in ts_to_show:
            if cls.startswith(_CHECKPOINT_PREFIX):
                continue
            _emit_axis(s, cls.replace("_", " "))
        if checkpoint_axes:
            lines.append(
                "### Checkpoint-immunotherapy response axis (anti-PD-1 / anti-PD-L1)\n"
            )
            lines.append(
                "One checkpoint marker, several response/resistance sub-programs below "
                "— not separate agents. Active T-cell-exclusion programs (Wnt, "
                "angiogenesis, adenosine, TGF-β) argue for *combination* "
                "(anti-PD-1 + anti-CTLA-4, +anti-VEGF, or adenosine blockade) over "
                "anti-PD-(L)1 monotherapy; antigen-presentation / B2M loss signals "
                "intrinsic checkpoint resistance.\n"
            )
            for cls, s in checkpoint_axes:
                _emit_axis(s, cls[len(_CHECKPOINT_PREFIX):].replace("_", " "))

    # --- CTAs: vaccination targets ---
    lines.append("## Cancer-Testis Antigens (Vaccination Targets)\n")
    lines.append(
        "CTAs are expressed in tumor but not normal adult tissue (except testis/placenta). "
        "Any expressed CTA is a potential vaccination target regardless of trial status.\n"
    )
    if len(ctas):
        lines.append(
            f"| Gene | {value_label} | Model interval | Bulk TPM (measured) | vs ref | Ref %ile | TME | Surface | Therapies |"
        )
        lines.append(
            "|------|-----------|----------------|---------------------|---------|-----------|-----|---------|-----------|"
        )
        for _, row in ctas.head(20).iterrows():
            surf = "yes" if row["is_surface"] else ""
            tme_warn = "tissue-explainable" if row.get("tme_explainable") else ""
            lines.append(
                f"| **{row['symbol']}** | {render_tpm(row['median_est'])} | "
                f"{render_tpm(row['est_1'])}\u2013{render_tpm(row['est_9'])} | "
                f"{render_tpm(row['observed_tpm'])} | "
                f"{_render_vs_tcga_cell(row)} | "
                f"{render_fraction_no_decimal(row['tcga_percentile'])} | "
                f"{tme_warn} | {surf} | {row['therapies']} |"
            )
        high_ctas = ctas[ctas["tcga_percentile"] > 0.7]
        if len(high_ctas):
            names = ", ".join(high_ctas["symbol"].head(5))
            lines.append(
                f"\n**Above broad-reference median for {reference_cancer_code}**: {names}"
            )
        # Propagate healthy-tissue-explainability warnings across all target
        # tables, not just surface (issue #50 point 5). For a CTA this is
        # usually a false alarm (CTAs activated in a subset of tumors),
        # but the flag still conveys real ambiguity — e.g. testis-
        # retained genes in TGCT samples — so surface it consistently.
        if "tme_explainable" in ctas.columns and ctas.head(20)["tme_explainable"].any():
            lines.append(
                "\n`tissue-explainable` = sample signal could be entirely explained by a single healthy "
                "tissue's expression. For CTAs this is usually benign (cohort-median "
                "≈ 0 is normal for CTAs), but verify the flagged gene is not a germline "
                "/ germ-cell lineage marker in this patient's context."
            )
    else:
        lines.append("No CTAs detected above threshold.\n")
    lines.append("")

    # --- Surface therapy targets ---
    # #60: drop extended-housekeeping symbols (excluded_from_ranking)
    # from the ranked output so they can't appear as spurious high
    # tumor-source-supported targets. They remain in the TSV with the flag.
    lines.append("## Surface Protein Targets (ADC / CAR-T / Bispecific)\n")
    lines.append(
        "Surface proteins with high context expression and source-attribution caveats. "
        "These can be targeted by antibody-drug conjugates, CAR-T, "
        "or bispecific T-cell engagers.\n"
    )
    if len(surface_targets):
        lines.append(
            f"| Gene | {value_label} | Model interval | Bulk TPM (measured) | vs ref | Ref %ile | TME | Attribution | Therapies |"
        )
        lines.append(
            "|------|-----------|----------------|---------------------|---------|-----------|-----|-------------|-----------|"
        )
        # #78: cross-signal annotations — flag IFN-driven surface /
        # MHC targets when the IFN_response axis is active, so the
        # reader knows a 287× HLA-F fold change isn't tumor-specific.
        cross_notes = annotate_surface_targets_with_cross_signals(
            ranges_df, analysis.get("therapy_response_scores") or {}
        )
        for _, row in surface_targets.head(20).iterrows():
            bold = "**" if row["therapies"] else ""
            if row.get("tme_dominant"):
                tme_warn = "background-dominant"
            elif row.get("tme_explainable"):
                tme_warn = "tissue-explainable"
            else:
                tme_warn = ""
            # Append IFN-driven cross-signal note to the therapies
            # column for core ISGs when IFN is active.
            therapies_cell = row["therapies"]
            cross = cross_notes.get(row["symbol"])
            if cross:
                therapies_cell = (
                    f"{therapies_cell} ({cross})" if therapies_cell else f"*{cross}*"
                )
            attribution_cell = _format_attribution_cell(row)
            lines.append(
                f"| {bold}{row['symbol']}{bold} | {render_tpm(row['median_est'])} | "
                f"{render_tpm(row['est_1'])}\u2013{render_tpm(row['est_9'])} | "
                f"{render_tpm(row['observed_tpm'])} | "
                f"{_render_vs_tcga_cell(row)} | "
                f"{render_fraction_no_decimal(row['tcga_percentile'])} | "
                f"{tme_warn} | {attribution_cell} | {therapies_cell} |"
            )
        head20 = surface_targets.head(20)
        any_dominant = (
            "tme_dominant" in surface_targets.columns and head20["tme_dominant"].any()
        )
        any_explainable = (
            "tme_explainable" in surface_targets.columns
            and head20["tme_explainable"].any()
        )
        if any_dominant:
            lines.append(
                "\n`background-dominant` = the decomposition attribution "
                "assigns less than 30% of the observed TPM to the tumor "
                "compartment (#108). These targets are very likely "
                "stromal / immune rather than tumor-cell expressed — "
                "excluding them from therapy-target consideration is the "
                "safest default (#35)."
            )
            lines.append(
                "\nAttribution column shows `tumor {tpm} / {dominant "
                "non-tumor compartment} {tpm}` from the decomposition "
                "fit; `—` means no decomposition attribution was "
                "available for this gene. A trailing `· broadly expr.` "
                "flag (#128) means the gene is detected above 5 nTPM "
                "in many non-reproductive HPA tissues and has no "
                "strongly enriched tissue — tumor-cell specificity is "
                "not supported regardless of the numeric tumor-attributed "
                "TPM."
            )
        if any_explainable:
            lines.append(
                "\n`tissue-explainable` = sample signal could be entirely explained by a single healthy "
                "tissue's expression (max across non-reproductive tissues ≥ 50% of "
                "observed TPM). The tumor-cell attribution for these genes is "
                "unreliable — consider stromal / immune origin."
            )
    else:
        lines.append("No surface targets above threshold.\n")
    lines.append("")

    # --- Cytosolic / intracellular targets (TCR-T, pMHC) ---
    lines.append("## Intracellular Targets (TCR-T / pMHC Vaccination)\n")
    lines.append(
        "Intracellular proteins presented via MHC-I. Targetable by "
        "TCR-T cell therapy or peptide vaccination.\n"
    )
    if len(intracellular):
        lines.append(
            f"| Gene | {value_label} | Model interval | vs ref | Ref %ile | TME | Attribution | CTA | Therapies |"
        )
        lines.append(
            "|------|-----------|----------------|---------|-----------|-----|-------------|-----|-----------|"
        )
        for _, row in intracellular.head(15).iterrows():
            cta_flag = "yes" if row["is_cta"] else ""
            tme_warn = "tissue-explainable" if row.get("tme_explainable") else ""
            attribution_cell = _format_attribution_cell(row)
            lines.append(
                f"| {row['symbol']} | {render_tpm(row['median_est'])} | "
                f"{render_tpm(row['est_1'])}\u2013{render_tpm(row['est_9'])} | "
                f"{_render_vs_tcga_cell(row)} | "
                f"{render_fraction_no_decimal(row['tcga_percentile'])} | "
                f"{tme_warn} | {attribution_cell} | {cta_flag} | {row['therapies']} |"
            )
        if (
            "tme_explainable" in intracellular.columns
            and intracellular.head(15)["tme_explainable"].any()
        ):
            lines.append(
                "\n`tissue-explainable` = sample signal could be entirely explained by a single healthy "
                "tissue's expression (max across non-reproductive tissues ≥ 50% of "
                "observed TPM). The tumor-cell attribution for these genes is "
                "unreliable — consider lineage-retained / stromal origin."
            )
    else:
        lines.append("No intracellular targets above threshold.\n")
    lines.append("")

    # --- Top recommendation summary ---
    # #79: Recommendations must respect the reliability flags from the
    # per-category tables. Previously the top-3 surface targets were
    # chosen by TPM rank alone, which promoted TME-dominant genes
    # dominant / very likely stromal) as "best surface targets" — the
    # opposite of the safe default. Now: skip unsupported rows entirely
    # from the summary; provisional retained rows carry an inline caveat.
    lines.append("## Recommended Targets Summary\n")

    def _reliability_badge(row):
        status = target_reliability_status(row)
        if status == "unsupported":
            return "background-dominant"
        if status == "provisional":
            return "provisional"
        return ""

    # Best surface — promote only rows that remain supported after the
    # attribution reliability pass.
    safe_surface = headline_surface[headline_surface["_status"] == "supported"].head(3)
    mixed_surface = headline_surface[headline_surface["_status"] == "provisional"].head(
        3
    )
    dropped_dominant = int((surface_status.head(10) == "unsupported").sum())
    if len(safe_surface):
        lines.append("**Best surface targets** (ADC/CAR-T/bispecific):")
        for _, row in safe_surface.iterrows():
            badge = _reliability_badge(row)
            reasons = target_reliability_reasons(row)
            caveat = (
                f", {badge}: {reasons[0]}" if badge == "provisional" and reasons else ""
            )
            therapy_note = (
                f" — active in {row['therapies']}" if row["therapies"] else ""
            )
            lines.append(
                f"- **{row['symbol']}** ({row['median_est']:.0f} TPM, "
                f"model interval {row['est_1']:.0f}\u2013{row['est_9']:.0f}"
                f"{caveat}){therapy_note}"
            )
        if dropped_dominant:
            lines.append(
                f"- *{dropped_dominant} top-ranked row"
                + ("s" if dropped_dominant != 1 else "")
                + " excluded from this summary because they remained background-dominant"
                " (see Surface Protein Targets table).*"
            )
        lines.append("")
    elif len(mixed_surface):
        lines.append("**Best surface targets** (ADC/CAR-T/bispecific):")
        for _, row in mixed_surface.iterrows():
            reasons = target_reliability_reasons(row)
            caveat = f", provisional: {reasons[0]}" if reasons else ""
            therapy_note = (
                f" — active in {row['therapies']}" if row["therapies"] else ""
            )
            lines.append(
                f"- **{row['symbol']}** ({row['median_est']:.0f} TPM, "
                f"{tumor_attribution_band_text(row)}{caveat}){therapy_note}"
            )
        lines.append(
            "- *These remain mixed-source rather than tumor-supported; treat them as lineage-guided options that need the full attribution and safety context.*"
        )
        lines.append("")
    elif dropped_dominant:
        lines.append(
            "**Best surface targets** (ADC/CAR-T/bispecific): "
            "all top-ranked rows remained background-dominant "
            "after the current uncertainty pass. See Surface Protein Targets "
            "table for full context."
        )
        lines.append("")

    # Best CTAs — same flag respect (though CTA flags usually benign,
    # cohort-median ≈ 0 is normal for CTAs).
    best_cta = ctas.head(3)
    if len(best_cta):
        lines.append("**Best CTA targets** (vaccination even without active trials):")
        for _, row in best_cta.iterrows():
            badge = _reliability_badge(row)
            reasons = target_reliability_reasons(row, category="CTA")
            # Don't print the badge twice when the first reason IS the badge
            # (e.g. badge "background-dominant" + reason "background-dominant"
            # rendered "— background-dominant background-dominant").
            caveat = ""
            if badge and reasons:
                caveat = (
                    f" — {badge}"
                    if reasons[0] == badge
                    else f" — {badge} {reasons[0]}"
                )
            lines.append(
                f"- **{row['symbol']}** ({row['median_est']:.0f} TPM, "
                f"TCGA {row['tcga_percentile']:.0%}){caveat}"
            )
        lines.append("")

    # MHC context for intracellular targeting
    lines.append(f"**MHC-I status**: {mhc_status_text}")
    lines.append("")

    return "\n".join(lines)


@named("plot-expression")
def plot_expression(
    input_path: str,
    output_image_prefix: Optional[str] = None,
    aggregate_gene_expression: bool = False,
    label_genes: Optional[str] = None,
    gene_name_col: Optional[str] = None,
    gene_id_col: Optional[str] = None,
    sample_id_col: Optional[str] = None,
    sample_id_value: Optional[str] = None,
    output_dpi: int = 300,
    plot_height: float = 14.0,
    plot_aspect: float = 1.4,
    cancer_type: Optional[str] = None,
    sample_mode: str = "auto",
    tumor_context: str = "auto",
    site_hint: Optional[str] = None,
    decomposition_templates: Optional[str] = None,
    hla_types: Optional[str] = None,
    alterations: Optional[str] = None,
    therapy_target_top_k: int = 10,
    therapy_target_tpm_threshold: float = 30.0,
    deprecated_figures: bool = False,
):
    """Deprecated: use 'analyze' instead."""
    import warnings

    warnings.warn(
        "plot-expression is deprecated, use 'analyze' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return analyze(
        input_path,
        output_image_prefix=output_image_prefix,
        aggregate_gene_expression=aggregate_gene_expression,
        label_genes=label_genes,
        gene_name_col=gene_name_col,
        gene_id_col=gene_id_col,
        sample_id_col=sample_id_col,
        sample_id_value=sample_id_value,
        output_dpi=output_dpi,
        plot_height=plot_height,
        plot_aspect=plot_aspect,
        cancer_type=cancer_type,
        sample_mode=sample_mode,
        tumor_context=tumor_context,
        site_hint=site_hint,
        decomposition_templates=decomposition_templates,
        hla_types=hla_types,
        alterations=alterations,
        therapy_target_top_k=therapy_target_top_k,
        therapy_target_tpm_threshold=therapy_target_tpm_threshold,
        deprecated_figures=deprecated_figures,
    )


@named("plot-cancer-cohorts")
def plot_cancer_cohorts(
    output_prefix: Optional[str] = None,
    output_dpi: int = 300,
):
    """Visualize curated gene sets across all 33 TCGA cancer types (no sample needed)."""
    from pathlib import Path

    prefix = output_prefix or "cohort"
    png_files = []

    # Plots without zscore parameter
    simple_plots = [
        ("disjoint-counts", plot_cohort_disjoint_counts),
        ("pca", plot_cohort_pca),
    ]
    for name, fn in simple_plots:
        out = f"{prefix}-{name}.png"
        fn(save_to_filename=out, save_dpi=output_dpi)
        png_files.append(out)

    # Heatmaps: emit both z-score and HK-normalized versions
    heatmap_plots = [
        ("heatmap", plot_cohort_heatmap),
        ("therapy-targets", plot_cohort_therapy_targets),
        ("surface-proteins", plot_cohort_surface_proteins),
        ("ctas", plot_cohort_ctas),
    ]
    for name, fn in heatmap_plots:
        for suffix, zs in [("zscore", True), ("hk", False)]:
            out = f"{prefix}-{name}-{suffix}.png"
            fn(save_to_filename=out, save_dpi=output_dpi, zscore=zs)
            png_files.append(out)

    # Collect into PDF (native resolution)
    pdf_path = f"{prefix}-all.pdf"
    images = []
    for png_path in png_files:
        if Path(png_path).exists():
            images.append(Image.open(png_path).convert("RGB"))
    if images:
        images[0].save(
            pdf_path, save_all=True, append_images=images[1:], resolution=output_dpi
        )
        print(f"Saved {pdf_path} ({len(images)} pages)")


def main():  # pragma: no cover - intentional redirect to the canonical CLI
    """Redirect to :mod:`trufflepig.cli`.

    This module is the migrated body of the old pirlygenes CLI; it
    still defines :func:`analyze`, :func:`compare_analyze`, and the
    per-stage helpers as a Python API. The user-facing entry point is
    :mod:`trufflepig.cli`. Without this redirect, running
    ``python -m trufflepig.main`` would silently bypass the workspace
    metadata, ``analyze/`` output layout, and lock filename that the
    rest of the tool expects.
    """
    import sys

    sys.stderr.write(
        "trufflepig.main is the migrated analysis Python API; the CLI "
        "lives in trufflepig.cli. Use `trufflepig run ...` or "
        "`python -m trufflepig.cli run ...` instead.\n"
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
