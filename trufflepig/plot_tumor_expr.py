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

import math
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt

from .common import _guess_gene_cols, ensembl_id_to_symbol_map
from .plot_data_helpers import _strip_ensembl_version
from pirlygenes.gene_sets_cancer import (
    housekeeping_gene_ids, is_extended_housekeeping_symbol, CTA_gene_id_to_name, therapy_target_gene_id_to_name, cancer_surfaceome_gene_id_to_name,
)
from trufflepig.reference import (
    cancer_reference_expression,
    pan_cancer_expression,
    subtype_deconvolved_expression,
    tcga_deconvolved_expression,
)
from .plot_scatter import resolve_cancer_type
from .plot_therapy import (
    _summarize_fn1_edb_transcript_support,
    _apply_therapy_support_gate,
)
from .reporting import tumor_attribution_context

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPRODUCTIVE_TISSUES = {"testis", "epididymis", "seminal_vesicle", "placenta", "ovary"}

_STROMAL_TISSUES = {
    "smooth_muscle",
    "skeletal_muscle",
    "heart_muscle",
    "adipose_tissue",
}

# Immune/lymphoid tissues that represent TME infiltrate.  Curated to
# exclude epithelial organs that merely contain resident immune cells
# (which would inflate the TME background estimate).
_IMMUNE_TISSUES = {
    "bone_marrow",
    "lymph_node",
    "spleen",
    "thymus",
    "tonsil",
    "appendix",
}

_TME_TISSUES = _STROMAL_TISSUES | _IMMUNE_TISSUES

# Met-site tissue augmentation (#13). When a biopsy was taken from a
# specific metastatic site, its host-tissue contribution should be
# represented in the TME background so tumor-expression estimates are
# not inflated by unsubtracted host-tissue signal. Each entry lists
# tissues to **add** to the TME reference; passing an empty set (for
# `primary`) leaves the default set untouched.
MET_SITE_TISSUE_AUGMENTATION = {
    "primary": set(),
    "lymph_node": {"lymph_node", "spleen", "thymus", "tonsil"},
    "liver": {"liver"},
    "brain": {"cerebral_cortex", "cerebellum"},
    "lung": {"lung"},
    "bone": {"bone_marrow"},
}

MET_SITES = tuple(MET_SITE_TISSUE_AUGMENTATION.keys())


@lru_cache(maxsize=128)
def _deconvolved_tumor_tpm_reference(code: str) -> tuple[dict[str, float], str]:
    """Return exact tumor-only TPM medians for a report/expression code.

    The pan-cancer matrix is ID-keyed and broad-cohort oriented. Trufflepig's
    regenerated deconvolved references are the right source for tumor-cell
    expression priors, especially for local/rare labels such as OS or NUTM
    that intentionally do not have a pan-cancer ``<CODE>_TPM`` column.
    """
    code_text = str(code or "").strip().upper()
    if not code_text:
        return {}, ""

    def lookup_codes() -> tuple[str, ...]:
        codes = [code_text]
        try:
            from trufflepig.analyze import expression_reference_options

            for record in expression_reference_options(
                code_text,
                include_fallback=False,
            ):
                if record.source_kind != "deconvolved_tumor_reference":
                    continue
                for value in (record.source_code, record.reference_code):
                    candidate = str(value or "").strip().upper()
                    if candidate:
                        codes.append(candidate)
        except Exception:
            pass
        return tuple(dict.fromkeys(codes))

    candidate_codes = set(lookup_codes())

    def source_label(frame, default: str) -> str:
        if "source_cohort" not in frame.columns:
            return default
        cohorts = sorted(
            {
                str(value).strip()
                for value in frame["source_cohort"].dropna().unique()
                if str(value).strip()
            }
        )
        if not cohorts:
            return default
        return default + ":" + ",".join(cohorts[:3])

    try:
        tcga = tcga_deconvolved_expression()
        if "cancer_code" in tcga.columns:
            sub = tcga[
                tcga["cancer_code"].astype(str).str.upper().isin(candidate_codes)
            ].copy()
            if not sub.empty:
                values = (
                    sub.groupby("symbol", dropna=False)["tumor_tpm_median"]
                    .median()
                    .astype(float)
                    .to_dict()
                )
                return (
                    {str(k): float(v) for k, v in values.items()},
                    source_label(sub, "tcga_deconvolved"),
                )
    except Exception:
        pass

    try:
        subtype = subtype_deconvolved_expression()
        if "cancer_code" in subtype.columns:
            sub = subtype[
                subtype["cancer_code"].astype(str).str.upper().isin(candidate_codes)
            ].copy()
        else:
            sub = subtype.iloc[0:0].copy()
        if sub.empty and "subtype" in subtype.columns:
            sub = subtype[
                subtype["subtype"]
                .fillna("")
                .astype(str)
                .str.upper()
                .isin(candidate_codes)
            ].copy()
        if not sub.empty:
            values = (
                sub.groupby("symbol", dropna=False)["tumor_tpm_median"]
                .median()
                    .astype(float)
                    .to_dict()
            )
            return (
                {str(k): float(v) for k, v in values.items()},
                source_label(sub, "subtype_deconvolved"),
            )
    except Exception:
        pass

    return {}, ""


@lru_cache(maxsize=128)
def _observed_bulk_tpm_reference(code: str) -> tuple[dict[str, float], str]:
    """Return observed clean-TPM cohort medians for an exact cancer code."""
    code_text = str(code or "").strip().upper()
    if not code_text:
        return {}, ""
    try:
        df = cancer_reference_expression(
            cancer_types=code_text,
            normalize="tpm_clean",
            format="long",
            include_provenance=True,
        )
    except Exception:
        return {}, ""
    if df is None or df.empty:
        return {}, ""
    if "normalization" in df.columns:
        df = df[df["normalization"].astype(str).str.lower().eq("tpm_clean")].copy()
    if df.empty or "Symbol" not in df.columns or "expression" not in df.columns:
        return {}, ""
    values = (
        df.groupby("Symbol", dropna=False)["expression"]
        .median()
        .astype(float)
        .to_dict()
    )
    source = "observed_bulk_reference"
    if "source_cohort" in df.columns:
        cohorts = sorted(
            {
                str(value).strip()
                for value in df["source_cohort"].dropna().unique()
                if str(value).strip()
            }
        )
        if cohorts:
            source = "observed_bulk_reference:" + ",".join(cohorts[:3])
    return {str(k): float(v) for k, v in values.items()}, source


@lru_cache(maxsize=128)
def _exact_expression_tpm_reference(code: str) -> tuple[dict[str, float], str, str]:
    """Return the best exact expression reference for a report code.

    Deconvolved tumor-cell references are preferred. Observed clean-TPM
    pirlygenes references are used when no deconvolved artifact exists.
    """
    values, source = _deconvolved_tumor_tpm_reference(code)
    if values:
        return values, source, "deconvolved_tumor_reference"
    values, source = _observed_bulk_tpm_reference(code)
    if values:
        return values, source, "observed_bulk_reference"
    return {}, "", ""

# #128: tissue-breadth thresholds for the "broadly-expressed" flag and
# the breadth-floor baseline used by the robust attribution algorithm.
#
# Sources & assumptions — kept explicit so downstream callers (or
# cancer-type-specific tuning) can override them via the tunable
# parameters below rather than digging into call sites.
#
# HK_TISSUE_NTPM_THRESHOLD (5.0 nTPM):
#   HPA's convention for "detectable expression" in their tissue
#   browser. A gene at <5 nTPM in a given tissue is effectively not
#   expressed there. Used to count how many tissues a gene reaches
#   above detection.
#
# BROAD_TISSUE_COUNT (15 of ~50 non-reproductive HPA tissues):
#   Threshold for "broadly expressed" is a **necessary but not
#   sufficient** condition — it's combined with the enrichment gate
#   below so a gene highly enriched in one specific tissue (e.g.
#   KLK3 in prostate, or a CTA in testis-plus-low-background) cannot
#   trip the flag just because low-level detection crosses many
#   tissues. 15/50 is ~30% — genes above that with **no** strong
#   tissue preference are genuinely broad.
#
# BROADLY_ENRICHED_MAX_RATIO (3.0×):
#   `max_healthy_tpm / mean_top_healthy_tpm` — a gene's peak tissue
#   must be less than this multiple of the mean of its top-N
#   tissues to qualify as broadly expressed. Tissue-restricted
#   genes (prostate-only KLK3, brain-only NEUROD1) have very high
#   enrichment ratios because max is far above mean. This keeps the
#   flag tissue-type agnostic — it depends on the gene's own
#   expression breadth, not on which tissues happen to be reference
#   tissues in HPA.
#
# BREADTH_BASELINE_TOP_N (10 of ~50):
#   Used for the attribution breadth floor. Mean of the 10 highest
#   healthy tissues gives a "what healthy cells would look like if
#   not drawn from any single tissue" baseline. Smaller N makes the
#   baseline more conservative (harder to trip); larger N pulls
#   toward the whole-tissue median. 10 is a reasonable middle.
#
# All three are module-level so a consumer tuning for a specific
# cancer type can adjust without forking — a blood-cancer pipeline
# might legitimately lower BROAD_TISSUE_COUNT, for example, since
# lymphoid-lineage genes appear across only a few tissues even at
# high levels.
HK_TISSUE_NTPM_THRESHOLD = 5.0
BROAD_TISSUE_COUNT = 15
BROADLY_ENRICHED_MAX_RATIO = 3.0
BREADTH_BASELINE_TOP_N = 10

# AMPLIFICATION_MIN_FOLD (5.0):
#   When the sample's observed TPM exceeds ``max_healthy_tpm`` by this
#   multiple, treat the observation as an amplification / over-
#   expression signal even if the gene is otherwise broadly expressed
#   at baseline. HER2 at 1000 TPM in a HER2+ breast cancer sample
#   against max-healthy ~100 is the canonical case: broadly
#   expressed per HPA, but the tumor-cell story is real and clinically
#   actionable. A 5× fold-over-peak is conservative; routine
#   tissue-to-tissue variability alone can account for 2-3×.
#
#   When ``amplified`` fires AND ``broadly_expressed`` is also True,
#   the reliability tier does NOT downgrade and the Attribution cell
#   renders "amplified N×" rather than "broadly expr." — the reader
#   sees the amplification story, not a spurious caution.
AMPLIFICATION_MIN_FOLD = 5.0

# Smooth-muscle lineage markers (#59 item 1). Canonical vascular +
# visceral SM identity genes — not rhabdomyosarcoma / cardiac markers
# (DES is omitted here because it also fires in skeletal / cardiac
# muscle and on RMS samples). When these land in the tumor-attributed
# column at materially high TPM, the reader should treat the
# tumor-cell story with caution: the matched-normal reference carries
# average fibromuscular-stroma density for the parent tissue, and a
# biopsy with above-average SM content leaks SM signal into tumor.
_SMOOTH_MUSCLE_LINEAGE_MARKERS = frozenset(
    {
        "TAGLN",  # Transgelin / SM22α
        "ACTA2",  # α-smooth muscle actin
        "MYH11",  # Smooth muscle myosin heavy chain 11
        "CNN1",  # Calponin 1
        "MYL9",  # Myosin light chain 9
        "CALD1",  # Caldesmon
        "SMTN",  # Smoothelin
        "MYLK",  # Myosin light-chain kinase
        "TPM2",  # Tropomyosin 2 (SM-enriched isoform)
    }
)
# Firing thresholds. Kept conservative — this is an annotation, not a
# refitting override, so over-annotating is cheap but under-annotating
# misses the case.
_SM_LEAKAGE_MIN_OBSERVED_TPM = 50.0
_SM_LEAKAGE_MIN_TUMOR_FRACTION = 0.30

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def _sample_expression_by_symbol(df_gene_expr):
    import pandas as pd
    from trufflepig.clean_tpm import assert_clean_tpm

    gene_id_col, gene_name_col = _guess_gene_cols(df_gene_expr)
    df = df_gene_expr.copy()
    df[gene_id_col] = df[gene_id_col].astype(str).map(_strip_ensembl_version)

    tpm_col = (
        "TPM"
        if "TPM" in df.columns
        else next((c for c in df.columns if c.lower() == "tpm"), None)
    )
    if tpm_col is None:
        raise KeyError(f"No TPM column found. Columns: {list(df.columns)}")
    assert_clean_tpm(
        df,
        value_cols=[tpm_col],
        label_col=gene_name_col,
        id_col=gene_id_col,
        context="analysis sample expression",
    )

    clean_tpm_values = df[tpm_col].astype(float)
    hk_mask = df[gene_id_col].isin(housekeeping_gene_ids())
    hk_median = df.loc[hk_mask, tpm_col].astype(float).median()
    if not (hk_median > 0):  # catches NaN and <= 0
        hk_median = 1.0
    hk_values = clean_tpm_values / hk_median

    # Resolve symbols from Ensembl IDs via pan-cancer reference.
    id_to_symbol = ensembl_id_to_symbol_map()
    if "canonical_gene_name" in df.columns:
        fallback = df["canonical_gene_name"].fillna("").astype(str)
    else:
        fallback = df[gene_name_col].fillna("").astype(str)
    symbols = df[gene_id_col].map(id_to_symbol).fillna(fallback)

    expr_df = pd.DataFrame(
        {
            "gene_id": df[gene_id_col],
            "Symbol": symbols,
            # Historical key: this is clean TPM before HK normalization, not
            # pre-QC raw TPM.
            "sample_raw": clean_tpm_values,
            "sample_hk": hk_values,
        }
    )
    expr_df = expr_df[expr_df["Symbol"].astype(str).str.strip().ne("")]
    # Aggregate by Ensembl ID (unique), then map to symbol.
    # Sum across rows with same ID (alt-haplotype reads are split by aligner).
    grouped = expr_df.groupby("gene_id", as_index=False, sort=False).agg(
        {"Symbol": "first", "sample_raw": "sum", "sample_hk": "sum"}
    )
    return (
        dict(zip(grouped["Symbol"], grouped["sample_raw"])),
        dict(zip(grouped["Symbol"], grouped["sample_hk"])),
    )


def estimate_tumor_expression(
    df_gene_expr,
    cancer_type,
    purity,
):
    """Estimate true tumor cell expression by deconvolving TME contribution.

    For each gene: ``tumor_expr = (observed - (1-purity) * tme_ref) / purity``

    Genes are categorized into:
    - **CTA**: cancer-testis antigens (vaccination targets)
    - **therapy_target**: genes with active therapy trials
    - **surface**: known surface proteins (ADC/CAR-T/bispecific targets)
    - **other**: remaining genes with meaningful tumor signal

    Returns a DataFrame with columns: gene_id, symbol, category,
    observed_tpm, tme_expected, tumor_adjusted, tcga_median,
    tcga_percentile, is_surface, therapies.
    """
    import pandas as pd
    from pirlygenes.gene_sets_cancer import (
        surface_protein_gene_ids,
    )

    cancer_code = resolve_cancer_type(cancer_type)

    # Sample expression
    sample_raw, _ = _sample_expression_by_symbol(df_gene_expr)

    # Reference data
    ref = pan_cancer_expression(technical_rna_normalize=True)
    ref_dedup = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    cohort_cols = [c for c in ref.columns if c.endswith("_TPM")]
    ntpm_cols = [c for c in ref.columns if c.endswith("_nTPM")]
    ntpm_nonrepro = [
        c for c in ntpm_cols if c.removesuffix("_nTPM") not in _REPRODUCTIVE_TISSUES
    ]

    # TME tissues
    ptprc_row = ref_dedup.loc["PTPRC"] if "PTPRC" in ref_dedup.index else None
    if ptprc_row is not None:
        ptprc_vals = ptprc_row[ntpm_nonrepro].astype(float)
        immune_cols = [c for c in ntpm_nonrepro if ptprc_vals[c] > ptprc_vals.median()]
    else:
        immune_cols = []
    stromal_cols = [
        c for c in ntpm_nonrepro if c.removesuffix("_nTPM") in _STROMAL_TISSUES
    ]
    tme_cols = list(set(immune_cols + stromal_cols))

    # Cancer type origin tissue: map cancer type to closest normal tissue
    cancer_col = f"{cancer_code}_TPM"
    tcga_expr = (
        ref_dedup[cancer_col].astype(float) if cancer_col in ref_dedup.columns else None
    )

    # Build gene lookup sets
    cta_map = CTA_gene_id_to_name()  # {ensembl_id: name}
    cta_symbols = set(cta_map.values())

    # Therapy targets across all therapy types
    _all_therapy_keys = [
        "ADC",
        "ADC-approved",
        "CAR-T",
        "CAR-T-approved",
        "TCR-T",
        "TCR-T-approved",
        "bispecific-antibodies",
        "bispecific-antibodies-approved",
        "radioligand",
    ]
    gene_therapies = {}  # symbol -> set of base therapy types
    for tt in _all_therapy_keys:
        try:
            tmap = therapy_target_gene_id_to_name(tt)
            base = tt.replace("-approved", "").replace("-trials", "")
            for gid, gname in tmap.items():
                gene_therapies.setdefault(gname, set()).add(base)
        except Exception:
            pass
    fn1_support = _summarize_fn1_edb_transcript_support(df_gene_expr)

    # Surface proteins
    try:
        surf_ids = surface_protein_gene_ids()
        cancer_surf = cancer_surfaceome_gene_id_to_name()
        ref_flat = ref.drop_duplicates(subset="Ensembl_Gene_ID")
        eid_to_sym = dict(zip(ref_flat["Ensembl_Gene_ID"], ref_flat["Symbol"]))
        surf_symbols = {eid_to_sym.get(eid, "") for eid in surf_ids}
        surf_symbols |= set(cancer_surf.values())
        surf_symbols.discard("")
    except Exception:
        surf_symbols = set()

    # TME reference: mean across TME tissues for each gene
    if tme_cols:
        tme_mean = ref_dedup[tme_cols].astype(float).mean(axis=1)
    else:
        tme_mean = pd.Series(0, index=ref_dedup.index)

    # TCGA distribution for percentile calculation
    cancer_expr_all = ref_dedup[cohort_cols].astype(float)

    # Build result rows — only process genes that the sample expresses
    # or that are in a known target category
    interesting_symbols = set(cta_symbols) | set(gene_therapies.keys())
    interesting_symbols |= {s for s, v in sample_raw.items() if v > 0.1}

    rows = []
    purity_clamp = max(purity, 0.01)  # avoid division by zero

    for symbol in interesting_symbols:
        if symbol not in ref_dedup.index:
            continue
        observed = sample_raw.get(symbol, 0.0)
        tme_ref = float(tme_mean.get(symbol, 0))
        tcga_med = float(tcga_expr[symbol]) if tcga_expr is not None else 0.0

        # Purity adjustment
        tumor_adj = max(0, (observed - (1 - purity_clamp) * tme_ref) / purity_clamp)

        # TCGA percentile
        ref_vals = cancer_expr_all.loc[symbol].values
        n = len(ref_vals)
        below = np.sum(ref_vals < tumor_adj)
        equal = np.sum(np.isclose(ref_vals, tumor_adj, atol=0.01))
        pctile = float((below + 0.5 * equal) / n)

        # Categorize
        is_cta = symbol in cta_symbols
        is_surface = symbol in surf_symbols
        (
            therapies,
            therapy_supported,
            therapy_support_note,
            therapy_support_tpm,
            therapy_support_fraction,
            therapy_supporting_transcripts,
        ) = _apply_therapy_support_gate(
            symbol,
            gene_therapies.get(symbol, set()),
            fn1_support,
        )
        is_therapy = bool(therapies)

        # Filter: only include genes with meaningful tumor signal
        # or that are in a known category
        if tumor_adj < 0.5 and not is_cta and not is_therapy:
            continue

        if is_cta:
            category = "CTA"
        elif is_therapy:
            category = "therapy_target"
        elif is_surface and tumor_adj > 1:
            category = "surface"
        else:
            category = "other"

        eid = (
            ref_dedup.loc[symbol, "Ensembl_Gene_ID"]
            if "Ensembl_Gene_ID" in ref_dedup.columns
            else ""
        )

        rows.append(
            {
                "gene_id": eid,
                "symbol": symbol,
                "category": category,
                "observed_tpm": round(observed, 2),
                "tme_expected": round(tme_ref, 2),
                "tumor_adjusted": round(tumor_adj, 2),
                "tcga_median": round(tcga_med, 2),
                "tcga_percentile": round(pctile, 3),
                "is_surface": is_surface,
                "is_cta": is_cta,
                "therapy_supported": therapy_supported,
                "therapy_support_note": therapy_support_note,
                "therapy_support_tpm": round(therapy_support_tpm, 2)
                if therapy_support_tpm is not None
                else None,
                "therapy_support_fraction": round(therapy_support_fraction, 3)
                if therapy_support_fraction is not None
                else None,
                "therapy_supporting_transcripts": therapy_supporting_transcripts,
                "therapies": ", ".join(sorted(therapies)) if therapies else "",
            }
        )

    result = pd.DataFrame(rows)
    result = result.sort_values("tumor_adjusted", ascending=False).reset_index(
        drop=True
    )
    return result


def estimate_tumor_expression_ranges(
    df_gene_expr,
    cancer_type,
    purity_result,
    decomposition_results=None,
    met_site=None,
    expression_reference_type=None,
):
    """Estimate tumor-specific expression with uncertainty bounds.

    For each gene, computes a 3x3 grid of estimates by crossing
    (low, med, high) TME background with (low, med, high) purity:

        tumor_expr = max(0, (observed - (1-purity) * tme_bg) / purity)

    TME bounds come from either:

    - the 25th / 50th / 75th percentile across TME reference tissues, or
    - the 25th / 50th / 75th percentile across candidate decomposition
      hypotheses when ``decomposition_results`` is provided.

    Purity bounds come from the ``overall_lower`` / ``overall_estimate`` /
    ``overall_upper`` fields of ``estimate_tumor_purity()``.

    Parameters
    ----------
    df_gene_expr : DataFrame
        Sample gene expression with TPM column.
    cancer_type : str
        TCGA cancer type code or name.
    purity_result : dict
        Return value of ``estimate_tumor_purity()``.
    decomposition_results : list, optional
        Candidate ``DecompositionResult`` objects from
        ``pirlygenes.decomposition.decompose_sample()``.
    expression_reference_type : str, optional
        Fine-grained report/expression code to use for tumor-expression priors
        when trufflepig has a regenerated deconvolved reference. The broad
        ``cancer_type`` still controls decomposition and TME context.

    Returns
    -------
    DataFrame with columns: symbol, category, observed_tpm,
        tme_lo, tme_med, tme_hi,
        est_1 ... est_9 (the 3x3 grid, ascending order),
        median_est, therapies, is_surface, is_cta.
    """
    import pandas as pd
    from pirlygenes.gene_sets_cancer import (
        surface_protein_gene_ids,
    )

    from pirlygenes.gene_sets_cancer import housekeeping_gene_ids
    from .decomposition.templates import EPITHELIAL_MATCHED_NORMAL_TISSUE
    from .tumor_purity import TCGA_MEDIAN_PURITY

    cancer_code = resolve_cancer_type(cancer_type)
    expression_reference_code = resolve_cancer_type(
        expression_reference_type or cancer_type
    )
    exact_ref_tpm, exact_ref_source, exact_ref_kind = _exact_expression_tpm_reference(
        expression_reference_code
    )
    epithelial_context = cancer_code in EPITHELIAL_MATCHED_NORMAL_TISSUE

    # --- Sample expression (clean TPM and HK-normalized) ---
    sample_raw, sample_hk = _sample_expression_by_symbol(df_gene_expr)

    # Sample HK median (for converting back from fold-HK to TPM)
    hk_ids = housekeeping_gene_ids()
    ref_full = pan_cancer_expression(technical_rna_normalize=True)
    ref_flat = ref_full.drop_duplicates(subset="Ensembl_Gene_ID")
    id_to_sym = dict(zip(ref_flat["Ensembl_Gene_ID"], ref_flat["Symbol"]))
    hk_syms = {id_to_sym[gid] for gid in hk_ids if gid in id_to_sym}
    sample_hk_vals = [sample_raw[s] for s in hk_syms if sample_raw.get(s, 0) > 0]
    sample_hk_median = float(np.median(sample_hk_vals)) if sample_hk_vals else 1.0

    # --- Reference data ---
    ref_dedup = ref_full.drop_duplicates(subset="Symbol").set_index("Symbol")
    ntpm_cols = [c for c in ref_full.columns if c.endswith("_nTPM")]
    cohort_cols = [c for c in ref_full.columns if c.endswith("_TPM")]
    ntpm_nonrepro = [
        c for c in ntpm_cols if c.removesuffix("_nTPM") not in _REPRODUCTIVE_TISSUES
    ]

    # TME tissues (curated immune + stromal). Met-site aware: when the
    # caller supplies ``met_site``, union in the host tissues for that
    # biopsy site so the TME background reflects, e.g., lymph-node
    # infiltrate for a nodal met (#13). Uniform median falls back to
    # the default curated set.
    effective_tme_tissues = set(_TME_TISSUES)
    if met_site:
        effective_tme_tissues |= MET_SITE_TISSUE_AUGMENTATION.get(met_site, set())
    tme_cols = [
        c for c in ntpm_nonrepro if c.removesuffix("_nTPM") in effective_tme_tissues
    ]

    # --- HK-normalize reference columns ---
    # Each column (nTPM tissue or TCGA cohort TPM) gets its own HK median.
    hk_in_ref = sorted(hk_syms & set(ref_dedup.index))
    ref_hk_medians = {}
    for col in tme_cols + cohort_cols:
        ref_hk_medians[col] = ref_dedup.loc[hk_in_ref, col].astype(float).median()

    # TME reference in HK-fold space: per gene, per tissue or decomposition.
    # When decomposition hypotheses are available, use their inferred
    # non-tumor background profile instead of a generic TME tissue panel.
    decomp_backgrounds = []
    if decomposition_results:
        for result in decomposition_results:
            bg = getattr(result, "tme_background_hk", None)
            if bg:
                decomp_backgrounds.append(bg)

    # Per-gene max across healthy tissues (precomputed vector; used to
    # flag rows where sample signal could be entirely TME-explained).
    # Vectorized once here to avoid a per-gene `.loc[].max()` inside the
    # main loop — that was a measured ~30x slowdown on full panels.
    #
    # #128: also precompute **tissue breadth** (how many non-reproductive
    # HPA tissues express the gene above the detection threshold) and
    # the **mean of the top-N healthy tissues**. Broadly-expressed genes
    # can't be attributed to one compartment cleanly; the per-gene
    # attribution was silently inflating tumor-inferred residuals for
    # universally-expressed housekeeping-like and surface genes. These
    # two metrics drive a breadth floor on non-tumor attribution and a
    # new ``broadly_expressed`` reliability flag.
    if ntpm_nonrepro:
        _all_healthy = ref_dedup[ntpm_nonrepro].astype(float)
        _max_healthy = _all_healthy.max(axis=1)
        max_healthy_tpm_by_symbol = _max_healthy.to_dict()
        # Count healthy tissues with nTPM >= HK_TISSUE_NTPM_THRESHOLD.
        # Threshold matches the detection bar elsewhere in pirlygenes:
        # 5 nTPM is "detectable expression" per HPA conventions.
        _n_tissues_expressed = (_all_healthy >= HK_TISSUE_NTPM_THRESHOLD).sum(axis=1)
        n_healthy_tissues_by_symbol = _n_tissues_expressed.to_dict()
        # Mean of the top-N healthy tissues per gene — used as a "breadth
        # baseline" for the non-tumor floor below. Genes expressed in
        # only one tissue have a small mean (dominated by zeros); genes
        # expressed ubiquitously have a large mean.
        _mean_top_healthy = _all_healthy.apply(
            lambda row: float(row.nlargest(BREADTH_BASELINE_TOP_N).mean())
            if row.notna().any()
            else 0.0,
            axis=1,
        )
        mean_top_healthy_tpm_by_symbol = _mean_top_healthy.to_dict()
    else:
        max_healthy_tpm_by_symbol = {}
        n_healthy_tissues_by_symbol = {}
        mean_top_healthy_tpm_by_symbol = {}

    # --- Full-coverage TPM-space TME background (fixes issue #45) --------
    #
    # The decomposition's `tme_background_hk` dict only covers genes that
    # were in the decomposition's signature panel. For target-list genes
    # like FN1 / COL1A1 / IGKC that AREN'T signature genes but ARE clearly
    # stromal/immune expressed, `tme_background_hk.get(sym, 0.0)` returns 0
    # -> no TME subtraction -> `tumor_tpm ~ sample_tpm / purity` which
    # inflates the "Tumor TPM" reported in the target table.
    #
    # Fix: when we have a decomposition result with a `fractions` dict,
    # build a full-gene signature matrix using the same cell-type /
    # bulk-tissue references the decomposition engine uses, and compute:
    #
    #   tme_bg_tpm[g] = sum_{c != tumor} fractions[c] * ref_tpm[g, c]
    #
    # This gives a per-gene TPM-scale TME background for every gene in
    # the reference, not just signature genes. The formula in the loop
    # prefers this TPM-space path when available and falls back to the
    # HK-fold path when it isn't.
    #
    # Matched-normal split (issue #50): when the decomposition included a
    # `matched_normal_<tissue>` component (epithelial primaries with
    # `use_matched_normal=True`), its contribution is tracked separately
    # as `matched_normal_tpm_by_symbol`. This lets the target report
    # distinguish "subtracted stromal/immune background" from "subtracted
    # benign parent-tissue contribution" per gene. The formula sums
    # them -- total non-tumor background is still `tme_only + matched_normal`.
    tme_bg_tpm_by_symbol = None
    tme_only_tpm_by_symbol = None
    matched_normal_tpm_by_symbol = None
    matched_normal_component_name = None
    matched_normal_tissue = None
    matched_normal_fraction_global = 0.0
    per_compartment_tpm_by_symbol = None  # #108: per-gene per-compartment TPM
    top_fractions = {}
    if decomposition_results:
        top_result = decomposition_results[0]
        top_fractions = getattr(top_result, "fractions", None) or {}
        non_tumor_components = [
            c for c, f in top_fractions.items() if c != "tumor" and f > 0
        ]
        matched_normal_component_name = getattr(
            top_result,
            "matched_normal_tissue",
            None,
        )
        if matched_normal_component_name is not None:
            matched_normal_tissue = matched_normal_component_name
            matched_normal_component_name = f"matched_normal_{matched_normal_tissue}"
            matched_normal_fraction_global = float(
                getattr(top_result, "matched_normal_fraction", 0.0) or 0.0
            )
        if non_tumor_components:
            from .decomposition.signature import build_signature_matrix

            try:
                _genes, sym_list, matrix, _cols = build_signature_matrix(
                    non_tumor_components,
                    gene_subset=None,
                    sample_by_eid=None,
                )
                non_tumor_fracs = np.array(
                    [float(top_fractions[c]) for c in non_tumor_components],
                    dtype=float,
                )
                # Per-gene expected non-tumor contribution to sample TPM:
                # sum_c fractions[c] * ref_tpm[g, c]. `fractions[c]` is
                # already scaled by (1 - tumor_fraction), so summing
                # directly gives absolute non-tumor TPM -- no extra
                # (1 - p) multiplier needed at the formula site.
                tme_tpm_vec = matrix @ non_tumor_fracs
                tme_bg_tpm_by_symbol = {
                    str(sym): float(val) for sym, val in zip(sym_list, tme_tpm_vec)
                }
                # #108: keep the per-compartment breakdown so target
                # rendering can show attribution columns instead of only
                # a collapsed TME total. Each entry maps a gene symbol to
                # a {compartment: attributed_tpm} dict (non-zero only).
                per_comp_mat = matrix * non_tumor_fracs[np.newaxis, :]
                per_compartment_tpm_by_symbol = {}
                for i, sym in enumerate(sym_list):
                    breakdown = {}
                    for j, comp in enumerate(non_tumor_components):
                        val = float(per_comp_mat[i, j])
                        if val >= 0.01:
                            breakdown[comp] = round(val, 2)
                    if breakdown:
                        per_compartment_tpm_by_symbol[str(sym)] = breakdown
                # Split out the matched-normal contribution so the report
                # can distinguish TME-only subtraction from parent-tissue
                # subtraction (issue #50). The matched-normal column is
                # present only when the decomposition was run with
                # `use_matched_normal=True` on an epithelial primary.
                if matched_normal_component_name in non_tumor_components:
                    mn_idx = non_tumor_components.index(matched_normal_component_name)
                    mn_frac = float(non_tumor_fracs[mn_idx])
                    mn_vec = matrix[:, mn_idx] * mn_frac
                    matched_normal_tpm_by_symbol = {
                        str(sym): float(val) for sym, val in zip(sym_list, mn_vec)
                    }
                    tme_only_vec = tme_tpm_vec - mn_vec
                    tme_only_tpm_by_symbol = {
                        str(sym): float(max(0.0, val))
                        for sym, val in zip(sym_list, tme_only_vec)
                    }
                else:
                    tme_only_tpm_by_symbol = dict(tme_bg_tpm_by_symbol)
            except Exception:
                tme_bg_tpm_by_symbol = None
                tme_only_tpm_by_symbol = None
                matched_normal_tpm_by_symbol = None

    # #56: post-hoc CAF / TAM reference swap. The NNLS anchors
    # ``fibroblast`` to HPA generic fibroblast and ``myeloid`` to HPA
    # generic macrophage; both under-represent the tumor-activated
    # states (CAF ≠ primary fibroblast; TAM ≠ generic macrophage).
    # Swap the per-gene reference for canonical marker genes (FAP /
    # POSTN / CD163 / MRC1 / ...) before computing tumor_tpm so
    # unambiguously-stromal / myeloid genes don't leak into the tumor
    # residual. NNLS is not re-run — strictly per-gene reference swap.
    subtype_refinement_provenance: dict = {}
    tme_bg_tpm_before_refinement = None
    if tme_bg_tpm_by_symbol is not None:
        from .decomposition.subtype_refs import refine_tme_per_gene

        tme_bg_tpm_before_refinement = dict(tme_bg_tpm_by_symbol)
        refined_tme, subtype_refinement_provenance = refine_tme_per_gene(
            tme_bg_tpm_by_symbol=tme_bg_tpm_by_symbol,
            per_compartment_tpm_by_symbol=per_compartment_tpm_by_symbol,
            sample_tpm_by_symbol=sample_raw,
        )
        tme_bg_tpm_by_symbol = refined_tme
        # Mirror the refinement into the TME-only view so matched-normal
        # stays unchanged but the activated-state correction propagates.
        # The per-compartment attribution map is also updated because
        # target evidence uses it to compute tumor-source bulk TPM.
        if subtype_refinement_provenance:
            for gene, prov in subtype_refinement_provenance.items():
                delta = float(prov["after"]) - float(prov["before"])
                if tme_only_tpm_by_symbol is not None:
                    tme_only_tpm_by_symbol[gene] = float(
                        tme_only_tpm_by_symbol.get(gene, 0.0) + delta
                    )
                comp = str(prov.get("compartment") or "")
                comp_after = prov.get("compartment_after")
                if (
                    comp
                    and comp_after is not None
                    and per_compartment_tpm_by_symbol is not None
                ):
                    per_comp = dict(per_compartment_tpm_by_symbol.get(gene, {}) or {})
                    per_comp[comp] = round(float(comp_after), 2)
                    per_compartment_tpm_by_symbol[gene] = per_comp

    # --- Purity-adjusted TCGA (HK-normalized, then deconvolved) ---
    # For each TCGA cohort TPM column, compute:
    #   tcga_hk = cohort TPM / cohort TPM HK median
    #   tme_hk  = median TME tissue fold (same as sample TME reference)
    #   tcga_tumor_hk = (tcga_hk - (1-tcga_purity) * tme_hk) / tcga_purity
    # We'll compute this per-gene in the loop.

    # --- Purity bounds ---
    p_lo = max(purity_result.get("overall_lower") or 0.01, 0.01)
    p_med = max(purity_result.get("overall_estimate") or 0.05, 0.01)
    p_hi = max(purity_result.get("overall_upper") or p_med, 0.01)
    p_lo, p_med, p_hi = sorted([p_lo, p_med, p_hi])

    LOW_PURITY_THRESHOLD = 0.25
    LOW_PURITY_HEADROOM = 3.0

    non_tumor_source_by_symbol = {}
    if epithelial_context:
        from .decomposition.subtype_refs import (
            CAF_MARKER_FOLDS,
            EXHAUSTED_T_MARKER_FOLDS,
            MDSC_MARKER_FOLDS,
            TAM_MARKER_FOLDS,
            TI_PLASMA_MARKER_FOLDS,
            TLS_B_MARKER_FOLDS,
            TUMOR_ENDOTHELIUM_MARKER_FOLDS,
        )

        for marker in CAF_MARKER_FOLDS:
            non_tumor_source_by_symbol[marker] = "stromal/CAF marker"
        for marker in TUMOR_ENDOTHELIUM_MARKER_FOLDS:
            non_tumor_source_by_symbol[marker] = "endothelial marker"
        for marker in TAM_MARKER_FOLDS:
            non_tumor_source_by_symbol[marker] = "myeloid/TAM marker"
        for marker in MDSC_MARKER_FOLDS:
            non_tumor_source_by_symbol[marker] = "myeloid/MDSC marker"
        for marker in EXHAUSTED_T_MARKER_FOLDS:
            non_tumor_source_by_symbol[marker] = "T-cell marker"
        for marker in TLS_B_MARKER_FOLDS:
            non_tumor_source_by_symbol[marker] = "B-cell/TLS marker"
        for marker in TI_PLASMA_MARKER_FOLDS:
            non_tumor_source_by_symbol[marker] = "plasma-cell marker"
        for marker in (
            "ACTA2",
            "TAGLN",
            "MYH11",
            "CNN1",
            "DES",
            "TPM2",
            "MYL9",
            "COL6A1",
            "COL6A2",
            "COL6A3",
        ):
            non_tumor_source_by_symbol[marker] = "smooth-muscle/stromal marker"
        for marker in (
            "PTPRC",
            "CD3D",
            "CD3E",
            "CD3G",
            "CD4",
            "CD8A",
            "CD8B",
            "MS4A1",
            "CD79A",
            "CD79B",
            "IGHG1",
            "IGKC",
            "HLA-A",
            "HLA-B",
            "HLA-C",
            "HLA-DRA",
            "HLA-DRB1",
            "B2M",
        ):
            non_tumor_source_by_symbol[marker] = "immune/MHC marker"

    def _scale_non_tumor_tpm(base_tpm, purity_used):
        denom = max(1.0 - float(p_med), 1e-3)
        numer = max(0.0, 1.0 - float(purity_used))
        return float(base_tpm) * (numer / denom)

    def _tumor_fraction_for_purity(purity_used):
        tumor_fraction_fit = float(top_fractions.get("tumor", 0.0) or 0.0)
        if tumor_fraction_fit <= 0 or p_med <= 0:
            return max(0.0, min(1.0, float(purity_used)))
        return max(
            0.0,
            min(1.0, tumor_fraction_fit * float(purity_used) / float(p_med)),
        )

    # --- Gene category lookups ---
    cta_symbols = set(CTA_gene_id_to_name().values())

    _all_therapy_keys = [
        "ADC",
        "ADC-approved",
        "CAR-T",
        "CAR-T-approved",
        "TCR-T",
        "TCR-T-approved",
        "bispecific-antibodies",
        "bispecific-antibodies-approved",
        "radioligand",
    ]
    gene_therapies = {}
    for tt in _all_therapy_keys:
        try:
            tmap = therapy_target_gene_id_to_name(tt)
            base = tt.replace("-approved", "").replace("-trials", "")
            for gid, gname in tmap.items():
                gene_therapies.setdefault(gname, set()).add(base)
        except Exception:
            pass
    fn1_support = _summarize_fn1_edb_transcript_support(df_gene_expr)

    try:
        surf_ids = surface_protein_gene_ids()
        cancer_surf = cancer_surfaceome_gene_id_to_name()
        eid_to_sym = dict(zip(ref_flat["Ensembl_Gene_ID"], ref_flat["Symbol"]))
        surf_symbols = {eid_to_sym.get(eid, "") for eid in surf_ids}
        surf_symbols |= set(cancer_surf.values())
        surf_symbols.discard("")
    except Exception:
        surf_symbols = set()

    # --- Compute 9-point estimates for every expressed gene ---
    cancer_expr_all = ref_dedup[cohort_cols].astype(float)
    rows = []
    for symbol in sample_raw:
        if symbol not in ref_dedup.index:
            continue
        observed = sample_raw[symbol]
        if observed < 0.01:
            continue

        # HK-normalize sample
        sample_fold = observed / sample_hk_median

        # TME background in HK-fold space.
        if decomp_backgrounds:
            tme_folds = [float(bg.get(symbol, 0.0)) for bg in decomp_backgrounds]
        else:
            tme_folds = []
            for col in tme_cols:
                hk_m = ref_hk_medians.get(col, 0)
                if hk_m > 0:
                    tme_folds.append(float(ref_dedup.loc[symbol, col]) / hk_m)
        if not tme_folds:
            tme_folds = [0.0]
        tme_fold_lo = float(np.percentile(tme_folds, 25))
        tme_fold_med = float(np.median(tme_folds))
        tme_fold_hi = float(np.percentile(tme_folds, 75))

        # TME-explainability flag: the max TPM this gene reaches in ANY
        # single healthy reference tissue. Used below to decide how
        # aggressively to clamp toward the cohort prior.
        max_healthy_tpm = float(max_healthy_tpm_by_symbol.get(symbol, 0.0))
        tme_explainable = max_healthy_tpm >= observed * 0.5

        # Cohort prior for tumor expression in this cancer type. Computed
        # by deconvolving the TCGA cancer-cohort median against the cohort's
        # assumed median purity, then rescaling to the sample's TPM scale.
        # This prior implicitly includes the "reactive stroma" signal
        # typical for this cancer type (TCGA medians are bulk tumor
        # samples, so tissue-specific cancer-associated fibroblast and
        # infiltrate contributions are baked in). Used for empirical-
        # Bayes shrinkage of the sample-based estimate at low purity.
        cancer_col = f"{cancer_code}_TPM"
        cohort_prior_tpm = 0.0
        tcga_tumor_fold = 0.0
        expression_reference_source = "pan_cancer_deconvolved"
        expression_reference_kind = "deconvolved_tumor_reference"
        expression_reference_used = cancer_code
        # Raw cohort median TPM in HK-normalized space (before TME
        # deconvolution). Kept separate so a downstream renderer can
        # distinguish "cohort genuinely doesn't express this gene" from
        # "cohort median was TME-only and got subtracted to zero" —
        # both currently collapse to ∞× fold, which is degenerate for
        # a clinician-facing table.
        tcga_fold_raw = 0.0
        tcga_cohort_tpm_raw = None  # None = gene not in ref (not_measurable)
        if symbol in exact_ref_tpm:
            expression_reference_source = exact_ref_source
            expression_reference_kind = exact_ref_kind
            expression_reference_used = expression_reference_code
            cohort_prior_tpm = max(0.0, float(exact_ref_tpm.get(symbol, 0.0)))
            tcga_tumor_fold = cohort_prior_tpm / sample_hk_median
            tcga_fold_raw = tcga_tumor_fold
            tcga_cohort_tpm_raw = cohort_prior_tpm
        elif (
            cancer_col in ref_dedup.columns
            and cancer_col in ref_hk_medians
            and ref_hk_medians[cancer_col] > 0
        ):
            cancer_hk_m = ref_hk_medians[cancer_col]
            tcga_fold = float(ref_dedup.loc[symbol, cancer_col]) / cancer_hk_m
            tcga_fold_raw = tcga_fold
            tcga_cohort_tpm_raw = float(ref_dedup.loc[symbol, cancer_col])
            tcga_p = TCGA_MEDIAN_PURITY.get(cancer_code, 0.7)
            tcga_tumor_fold = max(
                0.0, (tcga_fold - (1 - tcga_p) * tme_fold_med) / tcga_p
            )
            cohort_prior_tpm = tcga_tumor_fold * sample_hk_median

        # Empirical-Bayes shrinkage weight. At high purity the sample-
        # based estimate is reliable (w_sample -> 1). At low purity the
        # 1/p division in the deconvolution inflates noise; shrinkage
        # pulls estimates back toward the cohort prior (w_sample -> 0
        # as purity -> 0). `k_shrinkage` is the purity at which weights
        # are 50/50 -- tuned so that CTAs and real tumor markers at
        # moderate purity (~0.4) are mostly sample-driven, while
        # low-purity stromal-like genes get anchored to cohort.
        #
        # `k_shrinkage` doubles for tme_explainable genes (to ~2x
        # stronger shrinkage), reflecting the extra uncertainty about
        # whether the signal is genuinely tumor-cell-derived.
        k_shrinkage = 0.20
        if tme_explainable:
            k_shrinkage = 0.40

        # Shrinkage floor: when the cohort prior is near-zero (e.g.
        # CTAs -- activated in a minority of samples, so median = 0),
        # the prior mean is uninformative. Empirical-Bayes shrinkage
        # toward 0 would wrongly pull real sample-specific signal
        # down. Skip shrinkage in that regime and trust the sample.
        skip_shrinkage = cohort_prior_tpm < 1.0

        # 9-point estimate grid. TPM-space path (preferred) uses the
        # decomposition's per-gene expected TME contribution directly.
        # HK-fold path is the fallback when no decomposition is
        # available or the gene is absent from the reference. Every
        # grid point is shrunk toward the cohort prior and clamped at
        # `observed_tpm` when tme_explainable=True (tumor cells can't
        # contribute more than the observed signal if a single healthy
        # tissue could explain it alone).
        def _apply_priors(raw_tumor_tpm, purity_used):
            if skip_shrinkage:
                shrunk = raw_tumor_tpm
            else:
                w_sample = float(purity_used) / (float(purity_used) + k_shrinkage)
                shrunk = w_sample * raw_tumor_tpm + (1.0 - w_sample) * cohort_prior_tpm
            if tme_explainable:
                shrunk = min(shrunk, observed)
            return max(0.0, shrunk)

        estimates = []
        if tme_bg_tpm_by_symbol is not None and symbol in tme_bg_tpm_by_symbol:
            bg_tpm_mid = tme_bg_tpm_by_symbol[symbol]
            # Uncertainty on the TME estimate itself: +/-50%. Reference
            # cell-type profiles may under- or over-estimate the actual
            # infiltrate composition in the specific sample.
            for bg_tpm in [bg_tpm_mid * 0.5, bg_tpm_mid, bg_tpm_mid * 1.5]:
                for p in [p_lo, p_med, p_hi]:
                    raw = max(0.0, (observed - bg_tpm)) / p
                    estimates.append(_apply_priors(raw, p))
        else:
            for bg in [tme_fold_lo, tme_fold_med, tme_fold_hi]:
                for p in [p_lo, p_med, p_hi]:
                    tumor_fold = max(0.0, (sample_fold - (1 - p) * bg) / p)
                    raw = tumor_fold * sample_hk_median
                    estimates.append(_apply_priors(raw, p))
        estimates.sort()
        median_est = float(np.median(estimates))
        tumor_cell_tpm_low = float(estimates[0]) if estimates else median_est
        tumor_cell_tpm_high = float(estimates[-1]) if estimates else median_est

        ref_vals = cancer_expr_all.loc[symbol].values
        n = len(ref_vals)
        below = np.sum(ref_vals < median_est)
        equal = np.sum(np.isclose(ref_vals, median_est, atol=0.01))
        tcga_percentile = float((below + 0.5 * equal) / n)

        # Ratio vs purity-adjusted TCGA median for matched cancer type.
        # Uses the already-computed cohort tumor fold above.
        #
        # Three outcomes, each consumed by a distinct plot label:
        #   - finite positive -> fold-change vs TCGA ("0.3x", "1.5x", ...)
        #   - inf             -> sample expresses the gene but the TCGA
        #                       cohort tumor-component is essentially zero.
        #                       Rendered as a red "absent in TCGA" alert --
        #                       flags atypical expression for this cancer.
        #   - None            -> both the sample tumor-component and TCGA
        #                       tumor-component are essentially zero.
        #                       Rendered as a gray "0 in TCGA" -- nothing to
        #                       compare.
        #
        # `tcga_tumor_fold` clips to exactly 0.0 whenever the TCGA cohort
        # median is explainable by TME alone; previously the <= 0 branch
        # collapsed to None unconditionally, so a CTA with <cancer>_TPM
        # median ~ 0 and strong sample expression rendered as a quiet gray
        # label instead of the intended red "absent in TCGA" alert. The
        # sample-side check below restores the intended semantics.
        our_tumor_fold = median_est / sample_hk_median
        # Three diagnostic states for the cohort-reference comparison:
        #   "finite"        — cohort has detectable tumor component,
        #                     fold = sample / cohort
        #   "not_in_cohort" — raw cohort median ≈ 0 (gene genuinely not
        #                     expressed at cohort median; this is the
        #                     CTA-in-solid-cohort case)
        #   "tme_explained" — raw cohort was non-trivial but TME
        #                     subtraction zeroed the tumor component
        #   "both_absent"   — neither sample nor cohort expresses
        # The tumor-expression table can use the state to render a more
        # informative label than ∞×.
        if tcga_tumor_fold > 0.001:
            vs_tcga = float(our_tumor_fold / tcga_tumor_fold)
            tcga_ref_state = "finite"
        elif our_tumor_fold > 0.001:
            vs_tcga = float("inf")
            if tcga_fold_raw <= 0.001:
                tcga_ref_state = "not_in_cohort"
            else:
                tcga_ref_state = "tme_explained"
        else:
            vs_tcga = None
            tcga_ref_state = "both_absent"

        # Categorize
        is_cta = symbol in cta_symbols
        is_surface = symbol in surf_symbols
        (
            therapies,
            therapy_supported,
            therapy_support_note,
            therapy_support_tpm,
            therapy_support_fraction,
            therapy_supporting_transcripts,
        ) = _apply_therapy_support_gate(
            symbol,
            gene_therapies.get(symbol, set()),
            fn1_support,
        )

        if is_cta:
            category = "CTA"
        elif therapies:
            category = "therapy_target"
        elif is_surface:
            category = "surface"
        else:
            category = "other"

        eid = (
            ref_dedup.loc[symbol, "Ensembl_Gene_ID"]
            if "Ensembl_Gene_ID" in ref_dedup.columns
            else ""
        )

        # Per-gene matched-normal split reporting (issue #50). Zero when
        # no matched-normal component is active or the gene isn't in the
        # signature matrix. `estimation_path` records which branch
        # produced the estimate, so the target report can annotate each
        # gene with its provenance.
        mn_tpm = 0.0
        if matched_normal_tpm_by_symbol is not None:
            mn_tpm = float(matched_normal_tpm_by_symbol.get(symbol, 0.0))
        if tme_only_tpm_by_symbol is not None:
            tme_only_tpm = float(tme_only_tpm_by_symbol.get(symbol, 0.0))
        elif tme_bg_tpm_by_symbol is not None:
            tme_only_tpm = float(tme_bg_tpm_by_symbol.get(symbol, 0.0))
        else:
            tme_only_tpm = 0.0

        if tme_explainable:
            estimation_path = "clamped"
        elif mn_tpm > 0.0:
            estimation_path = "matched_normal_split"
        elif tme_bg_tpm_by_symbol is not None and symbol in tme_bg_tpm_by_symbol:
            estimation_path = "tme_only"
        else:
            estimation_path = "tme_fold_fallback"

        # #60: mark extended-housekeeping symbols so downstream
        # target tables can filter them out without silently dropping
        # rows from the TSV (power users still see them with the flag).
        excluded_from_ranking = bool(
            is_extended_housekeeping_symbol(symbol, scope="ranking")
        )

        # #108: per-compartment attribution. When decomposition ran,
        # apportion the observed TPM across TME compartments using the
        # per-compartment reference × fitted fractions, then derive a
        # tumor residual.
        #
        # #128: robust attribution. Add a **breadth floor** on non-
        # tumor attribution. The observation motivating it: on a 28%-
        # purity PRAD sample the decomposition's matched_normal_prostate
        # (~26%) + T_cell (3%) + endothelial (3%) compartments couldn't
        # absorb the signal of broadly-expressed genes like CRIM1,
        # HLA-F, IL6ST, TBCE, NPM1 — so the residual defaulted into
        # the tumor-inferred compartment and every one of these came out as 95–99%
        # tumor-attributed, which isn't defensible for genes expressed
        # in 15+ HPA tissues. The floor says: **for genes broadly
        # expressed, non-tumor cells in the sample carry a baseline
        # equal to non_tumor_frac × mean-of-top-N healthy tissues**.
        # If that exceeds the compartment-fit estimate, it's the
        # better prior for what healthy cells contribute.
        #
        # This does *not* clobber prostate-restricted PRAD targets:
        # KLK3 / NKX3-1 / HOXB13 have a small mean-of-top-N because
        # only prostate expresses them at meaningful levels, so the
        # breadth floor is tiny and the matched_normal_prostate
        # compartment fit dominates (as intended).
        attribution = {}
        if per_compartment_tpm_by_symbol is not None:
            raw = per_compartment_tpm_by_symbol.get(symbol)
            if raw:
                attribution = dict(raw)
        attr_tme_total_raw = sum(attribution.values())

        # #131 / #134 Option A: when the per-gene compartment fit
        # over-predicts (sum of reference × fitted_fraction exceeds
        # observed), we can't use the reference-weighted attribution
        # because it's internally inconsistent — either the fitted
        # matched-normal fraction is too high, or the sample's
        # matched-normal cells are expressing this gene below the
        # HPA reference (CRPC / AR-suppression is the canonical case
        # for KLK3 / KLK2 / TACSTD2 / FOLH1 on the rs sample).
        #
        # Fall back to a **purity-weighted split**: each compartment
        # receives `observed × fitted_fraction` (agnostic to the
        # per-gene HPA reference), and tumor receives
        # `observed × tumor_fraction`. Per-gene attribution is no
        # longer "reference × fit" but "sample-level-fraction ×
        # observed" — honest about the fact that the per-gene signal
        # alone can't distinguish tumor from matched-normal when the
        # reference over-predicts. The
        # ``matched_normal_over_predicted`` flag marks these rows so
        # the confidence tier and Attribution cell both surface the
        # caveat.
        #
        # This recovers clinically useful nonzero tumor attribution for
        # exactly the PRAD targets that matter (KLK3 82 TPM observed
        # → tumor ~23 TPM on a 28% purity sample, vs the old
        # proportional-rescale path that reported tumor = 0).
        matched_normal_over_predicted = (
            observed > 0 and attribution and attr_tme_total_raw > observed
        )
        if matched_normal_over_predicted and top_fractions:
            for comp in list(attribution.keys()):
                frac = float(top_fractions.get(comp, 0.0) or 0.0)
                attribution[comp] = round(observed * frac, 2)
        elif matched_normal_over_predicted:
            # No fitted fractions available — fall back to proportional
            # rescale so compartment display sums to observed.
            scale = observed / attr_tme_total_raw
            attribution = {
                comp: round(val * scale, 2) for comp, val in attribution.items()
            }
        attr_tme_total = sum(attribution.values())

        # Breadth metrics (precomputed once above).
        n_healthy_tissues_expressed = int(n_healthy_tissues_by_symbol.get(symbol, 0))
        mean_top_healthy_tpm = float(mean_top_healthy_tpm_by_symbol.get(symbol, 0.0))
        # Two-part gate for "broadly expressed":
        # 1. Detected above HK_TISSUE_NTPM_THRESHOLD in at least
        #    BROAD_TISSUE_COUNT non-reproductive HPA tissues; AND
        # 2. peak-to-mean-of-top-N ratio below BROADLY_ENRICHED_MAX_RATIO
        #    so a gene strongly enriched in one tissue (KLK3 in
        #    prostate, NEUROD1 in brain) cannot trip the flag purely
        #    because low-level detection crossed the count threshold.
        tissue_enrichment_ratio = (
            max_healthy_tpm / mean_top_healthy_tpm
            if mean_top_healthy_tpm > 0
            else float("inf")
        )
        broadly_at_baseline = (
            n_healthy_tissues_expressed >= BROAD_TISSUE_COUNT
            and tissue_enrichment_ratio < BROADLY_ENRICHED_MAX_RATIO
        )

        # Amplification gate. If observed is well above the peak
        # healthy tissue, the sample-level story is amplification / over-
        # expression — which overrides the breadth caveat for
        # reader-facing reliability. The raw attribution is unchanged
        # either way; only the warning / downgrade logic treats the
        # two differently.
        amplification_fold = (
            observed / max(max_healthy_tpm, 0.5) if observed > 0 else 0.0
        )
        amplified_over_healthy = amplification_fold >= AMPLIFICATION_MIN_FOLD

        # The reader-facing "broadly expressed" flag ONLY fires for
        # broadly-expressed-at-baseline genes that are NOT also
        # showing amplification. HER2-amplified BRCA, MDM2-amplified
        # LPS, GPC3-overexpressed HCC — all broadly expressed per HPA
        # but clinically actionable because of the amplification.
        # Those keep a clean rendering with an "amplified Nx" tag
        # instead of a suppression caveat.
        broadly_expressed = broadly_at_baseline and not amplified_over_healthy

        # Sample-level non-tumor fraction used for the breadth floor.
        # When purity isn't known yet (shouldn't happen in practice),
        # use a conservative 0.5.
        non_tumor_frac = max(0.0, min(1.0, 1.0 - p_med))
        source_marker_compartment = non_tumor_source_by_symbol.get(symbol, "")
        source_marker_non_tumor_prior = bool(
            source_marker_compartment and not amplified_over_healthy
        )
        breadth_floor = non_tumor_frac * mean_top_healthy_tpm
        if source_marker_non_tumor_prior:
            # In epithelial cancers, canonical CAF/immune/smooth-muscle
            # markers are stronger priors for admixed non-tumor biology
            # than for tumor-cell expression. Without a fitted
            # decomposition compartment, at least the non-tumor cellular
            # fraction should be allowed to explain these genes before the
            # residual is called tumor-source.
            breadth_floor = max(breadth_floor, non_tumor_frac * observed)

        def _attribution_candidate(purity_used):
            scaled_attr = {
                comp: _scale_non_tumor_tpm(val, purity_used)
                for comp, val in attribution.items()
            }
            attr_total_candidate = sum(scaled_attr.values())
            breadth_floor_candidate = (
                max(0.0, min(1.0, 1.0 - float(purity_used))) * mean_top_healthy_tpm
            )
            if source_marker_non_tumor_prior:
                breadth_floor_candidate = max(
                    breadth_floor_candidate,
                    max(0.0, min(1.0, 1.0 - float(purity_used))) * observed,
                )
            over_predicted_candidate = observed > 0 and attr_total_candidate > observed
            if over_predicted_candidate:
                tumor_fraction_fit = _tumor_fraction_for_purity(purity_used)
                tumor_candidate = max(
                    0.0,
                    min(
                        observed * tumor_fraction_fit,
                        observed - breadth_floor_candidate,
                    ),
                )
            else:
                tumor_candidate = max(
                    0.0,
                    observed - max(attr_total_candidate, breadth_floor_candidate),
                )

            capped = False
            tumor_candidate_pre_cap = float(tumor_candidate)
            purity_cap = None
            if (
                purity_used is not None
                and float(purity_used) < LOW_PURITY_THRESHOLD
                and not over_predicted_candidate
                and observed > 0
            ):
                purity_cap = observed * float(purity_used) * LOW_PURITY_HEADROOM
                if tumor_candidate > purity_cap:
                    tumor_candidate = purity_cap
                    capped = True

            tumor_fraction_candidate = (
                float(tumor_candidate / observed) if observed > 0 else 0.0
            )
            return (
                tumor_candidate,
                tumor_fraction_candidate,
                over_predicted_candidate,
                capped,
                tumor_candidate_pre_cap,
                purity_cap,
            )

        # Effective non-tumor attribution = max of
        # (per-compartment fit, breadth baseline). For gene-sparse
        # cases where the compartment fit is tiny but healthy cells
        # alone carry a meaningful baseline, breadth floor wins.
        #
        # #134 Option A: when matched-normal over-predicts, tumor is
        # computed directly from the purity split rather than the
        # residual — `observed × tumor_fraction`. The breadth floor
        # is still applied as a sanity check so broadly-expressed
        # genes don't claim unreasonably high tumor attribution even
        # under the purity-weighted fallback. Non-over-predicted
        # rows keep the residual semantics unchanged.
        if matched_normal_over_predicted and top_fractions:
            tumor_fraction_fit = float(top_fractions.get("tumor", 0.0) or 0.0)
            if tumor_fraction_fit <= 0:
                tumor_fraction_fit = float(p_med or 0.0)
            purity_inferred_tumor = observed * tumor_fraction_fit
            attr_tumor_tpm = max(
                0.0, min(purity_inferred_tumor, observed - breadth_floor)
            )
        else:
            effective_non_tumor = max(attr_tme_total, breadth_floor)
            attr_tumor_tpm = max(0.0, observed - effective_non_tumor)

        # #204: low-purity attribution cap. At purity < 25% the fitted
        # per-compartment TPMs systematically under-represent stromal
        # / TME contribution for broadly-expressed genes — the
        # reference tables (HPA per-cell-type nTPM) under-predict what
        # fibroblast / endothelial / macrophage compartments actually
        # contribute to bulk tissue, so the residual defaults to the
        # tumor compartment. Anchor attr_tumor_tpm to
        # ``purity * observed`` with a 3× headroom factor so genuine
        # amplification still registers (tumor-intrinsic can exceed
        # bulk-observed), but pure-stromal inflation is damped.
        #
        # Example (rs PRAD, 16% pure): FN1 obs=801, attr_tumor before
        # cap = 368 (46%); cap = 0.16 * 801 * 3 = 384 → unchanged.
        # IGF1R obs=279, attr_tumor before cap = 267 (96%); cap =
        # 0.16 * 279 * 3 = 134 → dropped to 134 (the tumor share this
        # compartment pattern + purity can plausibly support).
        low_purity_cap_applied = False
        if (
            p_med is not None
            and p_med < LOW_PURITY_THRESHOLD
            and not matched_normal_over_predicted
            and observed > 0
        ):
            purity_cap = observed * float(p_med) * LOW_PURITY_HEADROOM
            if attr_tumor_tpm > purity_cap:
                attr_tumor_tpm = purity_cap
                low_purity_cap_applied = True

        attr_estimates = []
        attr_fraction_estimates = []
        attr_over_predicted_flags = []
        attr_capped_flags = []
        attr_pre_cap_estimates = []
        attr_cap_estimates = []
        for purity_used in [p_lo, p_med, p_hi]:
            (
                tumor_candidate,
                fraction_candidate,
                over_flag,
                capped_flag,
                pre_cap_candidate,
                cap_candidate,
            ) = _attribution_candidate(purity_used)
            attr_estimates.append(float(tumor_candidate))
            attr_fraction_estimates.append(float(fraction_candidate))
            attr_over_predicted_flags.append(bool(over_flag))
            attr_capped_flags.append(bool(capped_flag))
            attr_pre_cap_estimates.append(float(pre_cap_candidate))
            if cap_candidate is not None:
                attr_cap_estimates.append(float(cap_candidate))
        attr_tumor_tpm = float(np.median(attr_estimates))
        attr_tumor_tpm_pre_cap = float(np.median(attr_pre_cap_estimates))
        attr_tumor_fraction = (
            float(np.median(attr_fraction_estimates))
            if attr_fraction_estimates
            else 0.0
        )
        attr_tumor_tpm_low = (
            float(min(attr_estimates)) if attr_estimates else attr_tumor_tpm
        )
        attr_tumor_tpm_high = (
            float(max(attr_estimates)) if attr_estimates else attr_tumor_tpm
        )
        attr_tumor_tpm_pre_cap_low = (
            float(min(attr_pre_cap_estimates))
            if attr_pre_cap_estimates
            else attr_tumor_tpm_pre_cap
        )
        attr_tumor_tpm_pre_cap_high = (
            float(max(attr_pre_cap_estimates))
            if attr_pre_cap_estimates
            else attr_tumor_tpm_pre_cap
        )
        low_purity_cap_tpm = (
            float(np.median(attr_cap_estimates)) if attr_cap_estimates else None
        )
        low_purity_cap_tpm_low = (
            float(min(attr_cap_estimates)) if attr_cap_estimates else None
        )
        low_purity_cap_tpm_high = (
            float(max(attr_cap_estimates)) if attr_cap_estimates else None
        )
        low_purity_cap_delta_tpm = max(
            0.0, float(attr_tumor_tpm_pre_cap - attr_tumor_tpm)
        )
        attr_tumor_fraction_low = (
            float(min(attr_fraction_estimates))
            if attr_fraction_estimates
            else attr_tumor_fraction
        )
        attr_tumor_fraction_high = (
            float(max(attr_fraction_estimates))
            if attr_fraction_estimates
            else attr_tumor_fraction
        )
        attr_support_fraction = (
            float(
                np.mean(
                    [
                        1.0 if (tpm >= 1.0 and frac >= 0.30) else 0.0
                        for tpm, frac in zip(attr_estimates, attr_fraction_estimates)
                    ]
                )
            )
            if attr_estimates
            else 0.0
        )
        low_purity_cap_applied = bool(low_purity_cap_applied or any(attr_capped_flags))
        matched_normal_over_predicted = bool(
            matched_normal_over_predicted or any(attr_over_predicted_flags)
        )

        if attribution:
            attr_top_comp, attr_top_tpm = max(attribution.items(), key=lambda kv: kv[1])
        else:
            attr_top_comp, attr_top_tpm = "", 0.0

        # #35: a gene whose observed TPM is mostly explained by non-
        # tumor compartments is a low-confidence tumor-expression claim,
        # especially risky at low purity where residual TPM is divided
        # by a small number and amplified. When decomposition attribution
        # is available, fire the flag from `attr_tumor_fraction < 0.3`
        # so it's grounded in the fitted compartments instead of a
        # generic TME-fold. Fall back to the old formula when no
        # decomposition ran.
        if attribution or breadth_floor > 0:
            tme_dominant = observed > 0 and attr_tumor_fraction < 0.30
        else:
            tme_dominant = observed > 0 and round(
                tme_fold_med, 4
            ) * sample_hk_median >= round(0.7 * observed, 4)
        low_confidence_tumor = tme_dominant or broadly_expressed
        if source_marker_non_tumor_prior:
            low_confidence_tumor = True

        rows.append(
            {
                "gene_id": eid,
                "symbol": symbol,
                "category": category,
                "observed_tpm": round(observed, 2),
                "tme_fold_lo": round(tme_fold_lo, 4),
                "tme_fold_med": round(tme_fold_med, 4),
                "tme_fold_hi": round(tme_fold_hi, 4),
                "max_healthy_tpm": round(max_healthy_tpm, 2),
                "tme_explainable": bool(tme_explainable),
                "tme_dominant": tme_dominant,
                "low_confidence_tumor": low_confidence_tumor,
                "source_marker_non_tumor_prior": source_marker_non_tumor_prior,
                "source_marker_compartment": source_marker_compartment,
                "cohort_prior_tpm": round(cohort_prior_tpm, 2),
                "expression_reference_code": expression_reference_used,
                "expression_reference_source": expression_reference_source,
                "expression_reference_kind": expression_reference_kind,
                "expression_reference_is_tumor_cell_estimate": (
                    expression_reference_kind == "deconvolved_tumor_reference"
                ),
                "tme_only_tpm": round(tme_only_tpm, 2),
                "matched_normal_tpm": round(mn_tpm, 2),
                "matched_normal_tissue": matched_normal_tissue or "",
                "matched_normal_fraction": round(matched_normal_fraction_global, 4),
                "estimation_path": estimation_path,
                # #108: per-compartment attribution. `attribution` is a dict
                # of {compartment: attributed_tpm}; `attr_tumor_tpm` is the
                # residual after subtracting those compartments; the top-
                # compartment shortcut keeps the common case cheap for
                # markdown rendering.
                "attribution": attribution,
                "tumor_attributed_bulk_tpm": round(attr_tumor_tpm, 2),
                "tumor_attributed_bulk_tpm_low": round(attr_tumor_tpm_low, 2),
                "tumor_attributed_bulk_tpm_high": round(attr_tumor_tpm_high, 2),
                "tumor_attributed_bulk_tpm_pre_low_purity_cap": round(
                    attr_tumor_tpm_pre_cap, 2
                ),
                "tumor_attributed_bulk_tpm_pre_low_purity_cap_low": round(
                    attr_tumor_tpm_pre_cap_low, 2
                ),
                "tumor_attributed_bulk_tpm_pre_low_purity_cap_high": round(
                    attr_tumor_tpm_pre_cap_high, 2
                ),
                "attr_tumor_tpm": round(attr_tumor_tpm, 2),
                "attr_tumor_fraction": round(attr_tumor_fraction, 4),
                "attr_tumor_tpm_low": round(attr_tumor_tpm_low, 2),
                "attr_tumor_tpm_high": round(attr_tumor_tpm_high, 2),
                "attr_tumor_fraction_low": round(attr_tumor_fraction_low, 4),
                "attr_tumor_fraction_high": round(attr_tumor_fraction_high, 4),
                "attr_support_fraction": round(attr_support_fraction, 4),
                "attr_top_compartment": attr_top_comp,
                "attr_top_compartment_tpm": round(float(attr_top_tpm), 2),
                # #204: True when the low-purity cap damped attr_tumor_tpm
                # away from the raw residual; downstream renderers can
                # tag these rows as "low-purity-capped" so clinicians know
                # the tumor share is bounded by purity × headroom, not
                # fitted directly.
                "low_purity_cap_applied": bool(low_purity_cap_applied),
                "low_purity_cap_tpm": (
                    round(low_purity_cap_tpm, 2)
                    if low_purity_cap_tpm is not None
                    else None
                ),
                "low_purity_cap_tpm_low": (
                    round(low_purity_cap_tpm_low, 2)
                    if low_purity_cap_tpm_low is not None
                    else None
                ),
                "low_purity_cap_tpm_high": (
                    round(low_purity_cap_tpm_high, 2)
                    if low_purity_cap_tpm_high is not None
                    else None
                ),
                "low_purity_cap_delta_tpm": round(low_purity_cap_delta_tpm, 2),
                # #128: breadth metrics used by the robust attribution.
                # `n_healthy_tissues_expressed` counts non-reproductive HPA
                # tissues with nTPM >= HK_TISSUE_NTPM_THRESHOLD;
                # `mean_top_healthy_tpm` is the mean of the top-N healthy
                # tissues. `broadly_expressed` is the reader-facing flag.
                # `breadth_floor_tpm` records the baseline that was applied
                # (useful for debugging / understanding why a specific
                # gene's attr_tumor was dampened).
                "n_healthy_tissues_expressed": n_healthy_tissues_expressed,
                "mean_top_healthy_tpm": round(mean_top_healthy_tpm, 2),
                "tissue_enrichment_ratio": (
                    round(tissue_enrichment_ratio, 2)
                    if tissue_enrichment_ratio != float("inf")
                    else None
                ),
                "broadly_at_baseline": broadly_at_baseline,
                "broadly_expressed": broadly_expressed,
                "amplification_fold": round(amplification_fold, 2),
                "amplified_over_healthy": amplified_over_healthy,
                "breadth_floor_tpm": round(breadth_floor, 2),
                # #131: flag when the per-compartment fit over-predicted
                # the gene's TPM. Attribution values above have been
                # proportionally rescaled to sum to observed; raw sum is
                # kept for audit.
                "matched_normal_over_predicted": matched_normal_over_predicted,
                "attribution_raw_sum_tpm": round(attr_tme_total_raw, 2),
                # #59 item 1: smooth-muscle stromal leakage. Matched-normal
                # references carry average fibromuscular-stroma density for
                # the parent tissue; a biopsy with above-average SM content
                # leaks SM-lineage signal into the tumor column. When a
                # canonical SM marker lands with a substantial tumor share
                # at a non-trivial TPM, annotate rather than silently treat
                # as tumor-cell expressed. NOT a refitting override — this
                # is strictly a reader-facing caveat that the tumor-cell
                # story should be read with skepticism.
                "smooth_muscle_stromal_leakage": bool(
                    symbol in _SMOOTH_MUSCLE_LINEAGE_MARKERS
                    and observed >= _SM_LEAKAGE_MIN_OBSERVED_TPM
                    and attr_tumor_fraction >= _SM_LEAKAGE_MIN_TUMOR_FRACTION
                ),
                # #56: post-hoc CAF / TAM reference swap provenance.
                # ``subtype_refined`` is True when the per-gene TME
                # contribution got reference-swapped; the *_before column
                # carries the TPM that would have been attributed to tumor
                # under the generic-reference path for comparison.
                "subtype_refined": bool(
                    subtype_refinement_provenance
                    and symbol in subtype_refinement_provenance
                ),
                "subtype_refinement_label": (
                    subtype_refinement_provenance.get(symbol, {}).get("subtype", "")
                    if subtype_refinement_provenance
                    else ""
                ),
                "tme_tpm_before_subtype_refinement": (
                    round(float(tme_bg_tpm_before_refinement.get(symbol, 0.0)), 2)
                    if tme_bg_tpm_before_refinement is not None
                    else None
                ),
                **{f"est_{i + 1}": round(estimates[i], 2) for i in range(9)},
                "median_est": round(median_est, 2),
                "tumor_cell_tpm": round(median_est, 2),
                "tumor_cell_tpm_low": round(tumor_cell_tpm_low, 2),
                "tumor_cell_tpm_high": round(tumor_cell_tpm_high, 2),
                "pct_cancer_median": round(vs_tcga, 2)
                if vs_tcga is not None and not math.isinf(vs_tcga)
                else vs_tcga,
                "tcga_ref_state": tcga_ref_state,
                "reference_ref_state": tcga_ref_state,
                "tcga_cohort_median_tpm": (
                    round(tcga_cohort_tpm_raw, 3)
                    if tcga_cohort_tpm_raw is not None
                    else None
                ),
                "reference_cohort_median_tpm": (
                    round(tcga_cohort_tpm_raw, 3)
                    if tcga_cohort_tpm_raw is not None
                    else None
                ),
                "tcga_percentile": round(tcga_percentile, 3),
                "is_surface": is_surface,
                "is_cta": is_cta,
                "excluded_from_ranking": excluded_from_ranking,
                "therapy_supported": therapy_supported,
                "therapy_support_note": therapy_support_note,
                "therapy_support_tpm": round(therapy_support_tpm, 2)
                if therapy_support_tpm is not None
                else None,
                "therapy_support_fraction": round(therapy_support_fraction, 3)
                if therapy_support_fraction is not None
                else None,
                "therapy_supporting_transcripts": therapy_supporting_transcripts,
                "therapies": ", ".join(sorted(therapies)) if therapies else "",
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("median_est", ascending=False).reset_index(
            drop=True
        )
    # Stash the set of symbols actually present in the input file (regardless
    # of expression level) so downstream renderers can distinguish
    # "below detection (measured as ~0)" from "not in input file" — the two
    # have very different clinical meaning (the former is a real negative for
    # the target; the latter is a coverage gap that needs investigation).
    from .common import NoDeepcopyFrozenSet

    result.attrs["sample_input_symbols"] = NoDeepcopyFrozenSet(sample_raw)
    return result


def plot_matched_normal_attribution(
    df_ranges,
    cancer_type,
    category,
    top_n=15,
    save_to_filename=None,
    save_dpi=300,
    figsize=None,
    sample_tpm_by_symbol=None,
):
    """Stacked horizontal-bar plot of per-gene tumor / matched-normal / TME
    attribution for a single target category (issue #55).

    For each of the top ``top_n`` genes in ``category`` (ranked by
    ``median_est``), draw a horizontal bar broken into:

    - tumor-attributed bulk TPM from the decomposition residual
    - matched-normal / benign parent-tissue contribution
    - other background compartments from the decomposition attribution

    Only useful when ``df_ranges`` carries non-zero ``matched_normal_tpm``
    for at least one gene in the category (i.e. the decomposition ran
    with ``use_matched_normal=True`` for an epithelial primary). Returns
    ``None`` otherwise -- the CLI uses that to skip emitting an empty
    figure.

    Emitted as a standalone PNG (one per category) rather than as a
    panel in a composite figure, following the project's plot-crowding
    preference.
    """

    cancer_code = resolve_cancer_type(cancer_type)

    sub = df_ranges[df_ranges["category"] == category].head(top_n).copy()
    if sub.empty:
        return None
    if "matched_normal_tpm" not in sub.columns:
        return None
    if (sub["matched_normal_tpm"].astype(float) <= 0).all():
        return None

    sub = sub.sort_values("median_est", ascending=True).reset_index(drop=True)
    n = len(sub)
    if figsize is None:
        figsize = (10, max(3.0, 0.4 * n + 1.5))

    def _safe_float_value(value, default=0.0):
        try:
            result = float(value)
        except Exception:
            return float(default)
        if not np.isfinite(result):
            return float(default)
        return result

    def _attribution_dict(value):
        return value if isinstance(value, dict) else {}

    def _split_display_components(row):
        observed_value = _safe_float_value(row.get("observed_tpm"), 0.0)
        matched_value = _safe_float_value(row.get("matched_normal_tpm"), 0.0)
        other_value = _safe_float_value(row.get("tme_only_tpm"), 0.0)
        tumor_value = max(0.0, observed_value - matched_value - other_value)

        attr = _attribution_dict(row.get("attribution"))
        if "attr_tumor_tpm" in row:
            tumor_value = _safe_float_value(row.get("attr_tumor_tpm"), tumor_value)
        if attr:
            matched_from_attr = sum(
                _safe_float_value(value)
                for name, value in attr.items()
                if str(name).startswith("matched_normal")
            )
            other_from_attr = sum(
                _safe_float_value(value)
                for name, value in attr.items()
                if str(name) != "tumor" and not str(name).startswith("matched_normal")
            )
            matched_value = max(matched_value, matched_from_attr)
            other_value = max(other_value, other_from_attr)

        tumor_value = max(0.0, min(tumor_value, observed_value))
        matched_value = max(0.0, matched_value)
        other_value = max(0.0, other_value)
        # If robust attribution says tumor is small but the explicit
        # background compartments are sparse/thresholded, do not put the
        # unexplained remainder back into tumor. Keep the displayed stack
        # faithful to the tumor-attribution residual.
        other_value = max(other_value, observed_value - tumor_value - matched_value)

        total = tumor_value + matched_value + other_value
        if observed_value > 0 and total > observed_value:
            scale = observed_value / total
            tumor_value *= scale
            matched_value *= scale
            other_value *= scale
        return tumor_value, matched_value, other_value

    split_components = [_split_display_components(row) for _, row in sub.iterrows()]
    tumor_attr = np.array([parts[0] for parts in split_components], dtype=float)
    mn = np.array([parts[1] for parts in split_components], dtype=float)
    tme = np.array([parts[2] for parts in split_components], dtype=float)

    y = np.arange(n)
    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(y, tumor_attr, color="#e74c3c", label="tumor cells")
    ax.barh(y, mn, left=tumor_attr, color="#3498db", label="matched-normal tissue")
    ax.barh(
        y,
        tme,
        left=tumor_attr + mn,
        color="#95a5a6",
        label="other TME (stromal/immune)",
    )

    labels = []
    for _, row in sub.iterrows():
        sym = str(row["symbol"])
        if row.get("therapies"):
            sym = f"{sym}  [{row['therapies']}]"
        flags = []
        if row.get("tme_explainable"):
            flags.append("tissue-explainable")
        if row.get("matched_normal_over_predicted"):
            flags.append("MN>obs")
        elif str(row.get("estimation_path", "")) == "clamped":
            # Older ranges_df rows (before #131) won't have the
            # matched_normal_over_predicted column — fall back to
            # the estimation_path signal but with a human-readable
            # marker instead of the raw "clamp" jargon.
            flags.append("MN>obs")
        if flags:
            sym = f"{sym}  {' '.join(flags)}"
        labels.append(sym)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)

    # Cohort-prior marker: small tick on each bar so reviewers can see
    # where the TCGA cohort would have placed tumor-cell expression.
    if "cohort_prior_tpm" in sub.columns:
        priors = sub["cohort_prior_tpm"].astype(float).values
        for i, prior in enumerate(priors):
            if prior > 0:
                ax.plot(
                    [prior],
                    [i],
                    marker="|",
                    color="black",
                    markersize=14,
                    markeredgewidth=2,
                )

    ax.set_xlabel(
        "Bulk TPM attribution (stacked: tumor + matched-normal + other background)",
        fontsize=10,
    )
    ax.set_xscale("symlog", linthresh=1.0)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # Sample-wide 90th-percentile reference line — faint dashed anchor
    # so readers see where each gene sits relative to the rest of the
    # transcriptome.
    if sample_tpm_by_symbol is not None:
        from .plot_reference_lines import add_p90_reference_line

        add_p90_reference_line(
            ax,
            sample_tpm_by_symbol,
            orientation="vertical",
        )

    mn_tissue = ""
    if "matched_normal_tissue" in sub.columns:
        nonempty = sub["matched_normal_tissue"].astype(str).replace("nan", "")
        nonempty = [v for v in nonempty.unique() if v]
        if nonempty:
            mn_tissue = nonempty[0]
    title = f"Matched-normal attribution \u2014 {cancer_code} {category}"
    if mn_tissue:
        title += f"\n(benign {mn_tissue} and other decomposition background; black tick = TCGA cohort prior)"
    ax.set_title(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig


def plot_target_attribution(
    df_ranges,
    cancer_type,
    category,
    top_n=15,
    sample_tpm_by_symbol=None,
    save_to_filename=None,
    save_dpi=300,
    figsize=None,
):
    """Per-target compositional attribution stacked bars (#108).

    For each of the selected ``top_n`` targets in ``category``, draw a
    horizontal bar broken into the tumor
    plus each non-tumor compartment that contributes at least 1% of the
    observed TPM. Compartments are read from the ``attribution`` column
    of ``df_ranges`` (a dict of {compartment: TPM}); tumor TPM is
    taken from ``attr_tumor_tpm``.

    Emitted as a standalone PNG (one per category, per project
    plot-crowding preference). Returns ``None`` and does not write a
    file when no row in the category has an attribution breakdown (e.g.
    decomposition didn't run, or none of the top targets overlapped the
    reference matrix).
    """
    cancer_code = resolve_cancer_type(cancer_type)

    def _therapy_list(row):
        raw = str(row.get("therapies") or "").strip()
        if not raw:
            return []
        return [piece.strip() for piece in raw.split(",") if piece.strip()]

    def _select_rows(frame):
        sub = frame[frame["category"] == category].copy()
        if sub.empty:
            return sub
        tier_order = {
            "mixed_source": 0,
            "background_dominant": 1,
            "tumor_supported": 2,
        }
        source_tiers = []
        sort_keys = []
        for _, row in sub.iterrows():
            source = tumor_attribution_context(row)
            therapies = _therapy_list(row)
            therapy_supported = row.get("therapy_supported") is True or bool(therapies)
            source_tiers.append(source["tier"])
            sort_keys.append(
                (
                    0 if therapy_supported else 1,
                    tier_order.get(source["tier"], 9),
                    -len(therapies),
                    -float(row.get("observed_tpm") or 0.0),
                    -float(row.get("attr_tumor_tpm") or 0.0),
                    str(row.get("symbol") or ""),
                )
            )
        sub["_source_tier"] = source_tiers
        sub["_sort_key"] = sort_keys
        ranked = sub.sort_values("_sort_key").copy()
        selected = []
        base_quota = 2 if top_n >= 6 else 1
        for tier in ("mixed_source", "background_dominant", "tumor_supported"):
            tier_rows = ranked[ranked["_source_tier"] == tier]
            selected.extend(list(tier_rows.head(base_quota).index))
        for idx in ranked.index:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= top_n:
                break
        chosen = ranked.loc[selected[:top_n]].copy()
        return chosen

    sub = _select_rows(df_ranges)
    if sub.empty or "attribution" not in sub.columns:
        return None

    def _has_breakdown(v):
        return isinstance(v, dict) and len(v) > 0

    if not sub["attribution"].apply(_has_breakdown).any():
        return None

    sub = sub.sort_values("median_est", ascending=True).reset_index(drop=True)
    n = len(sub)
    if figsize is None:
        figsize = (11, max(3.0, 0.4 * n + 1.5))

    # Collect all compartments that appear across the shown targets, in
    # descending order of aggregate contribution so the legend ranks the
    # most-impactful compartments first.
    totals = {}
    for attr in sub["attribution"]:
        if not isinstance(attr, dict):
            continue
        for comp, tpm in attr.items():
            totals[comp] = totals.get(comp, 0.0) + float(tpm)
    compartments = sorted(totals, key=lambda c: -totals[c])
    palette = plt.cm.tab20.colors
    comp_colors = {c: palette[i % len(palette)] for i, c in enumerate(compartments)}

    y = np.arange(n)
    fig, ax = plt.subplots(figsize=figsize)

    tumor_attr = sub["attr_tumor_tpm"].astype(float).values
    ax.barh(y, tumor_attr, color="#e74c3c", label="tumor")
    left = tumor_attr.copy()
    for comp in compartments:
        vals = np.array(
            [
                float(attr.get(comp, 0.0)) if isinstance(attr, dict) else 0.0
                for attr in sub["attribution"]
            ]
        )
        if not np.any(vals > 0):
            continue
        ax.barh(
            y, vals, left=left, color=comp_colors[comp], label=comp.replace("_", " ")
        )
        left = left + vals

    labels = []
    for _, row in sub.iterrows():
        sym = str(row["symbol"])
        if row.get("therapies"):
            sym = f"{sym}  [{row['therapies']}]"
        if row.get("tme_dominant"):
            sym = f"{sym}  tumor-low"
        elif row.get("tme_explainable"):
            sym = f"{sym}  tissue-explainable"
        labels.append(sym)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel(
        "TPM (stacked: tumor + background compartments = observed)",
        fontsize=10,
    )
    ax.set_xscale("symlog", linthresh=1.0)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, ncol=1)
    cat_label = {
        "therapy_target": "Therapy-linked targets",
        "CTA": "CTAs",
        "surface": "Surface proteins",
    }.get(category, category.replace("_", " "))
    ax.set_title(
        f"Target source breakdown — {cat_label} — {cancer_code}\n"
        "Selected to show clinically relevant genes plus mixed/background cases",
        fontsize=10,
        fontweight="bold",
    )
    # Sample-wide 90th-percentile reference line (faint dashed).
    if sample_tpm_by_symbol is not None:
        from .plot_reference_lines import add_p90_reference_line

        add_p90_reference_line(
            ax,
            sample_tpm_by_symbol,
            orientation="vertical",
        )

    plt.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig


def plot_subtype_attribution(
    df_ranges,
    category,
    top_n=15,
    save_to_filename=None,
    save_dpi=300,
    figsize=None,
):
    """Per-gene before/after deltas from subtype refinement (#56 / #58).

    For each of the top-N genes in ``category`` that were refined by
    ``refine_tme_per_gene``, draw paired horizontal bars:

    - BEFORE: tumor residual + aggregate TME under the generic-reference
      path (pre-#56 behavior).
    - AFTER: tumor residual + aggregate TME under the tumor-activated
      reference swap.

    Each gene's pair is labelled with the winning subtype reference
    (CAF / TAM / exhausted_T / ...). This is an audit/provenance view:
    it shows which genes changed materially after swapping from the
    generic background reference to a subtype-specific one.

    Returns ``None`` when no row in the category has subtype-refinement
    provenance (e.g. decomposition didn't run, or no marker gene
    crossed the fold threshold on this category).
    """
    if "subtype_refined" not in df_ranges.columns:
        return None
    sub = df_ranges[
        (df_ranges["category"] == category)
        & (df_ranges["subtype_refined"].astype(bool))
    ].copy()
    if sub.empty:
        return None

    # Rank by the size of the per-gene correction so the most-impactful
    # refinements appear at the top of the chart.
    sub["_delta"] = (
        sub["tme_tpm_before_subtype_refinement"].astype(float)
        - (sub["observed_tpm"].astype(float) - sub["attr_tumor_tpm"].astype(float))
    ).abs()
    sub = sub.sort_values("_delta", ascending=False).head(top_n)
    sub = sub.sort_values("_delta", ascending=True).reset_index(drop=True)

    n = len(sub)
    if n == 0:
        return None
    if figsize is None:
        figsize = (11, max(3.0, 0.55 * n + 1.5))

    # Two rows per gene: "before" above, "after" below.
    fig, ax = plt.subplots(figsize=figsize)

    tumor_color = "#e74c3c"
    tme_color_before = "#95a5a6"  # grey — generic reference
    tme_color_after = "#f39c12"  # orange — tumor-activated reference

    row_positions = []
    labels = []
    for i, (_, row) in enumerate(sub.iterrows()):
        observed = float(row.get("observed_tpm") or 0.0)
        tme_before = float(row.get("tme_tpm_before_subtype_refinement") or 0.0)
        tumor_before = max(0.0, observed - tme_before)
        tumor_after = float(row.get("attr_tumor_tpm") or 0.0)
        tme_after = max(0.0, observed - tumor_after)

        y_before = 2 * i
        y_after = 2 * i + 0.75
        ax.barh(y_before, tumor_before, color=tumor_color)
        ax.barh(y_before, tme_before, left=tumor_before, color=tme_color_before)
        ax.barh(y_after, tumor_after, color=tumor_color)
        ax.barh(y_after, tme_after, left=tumor_after, color=tme_color_after)

        sym = str(row["symbol"])
        label_subtype = row.get("subtype_refinement_label", "") or "refined"
        row_positions.extend([y_before, y_after])
        labels.append(f"{sym} · before")
        labels.append(f"{sym} · after ({label_subtype})")

    ax.set_yticks(row_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(
        "TPM (stacked: tumor + background = observed)",
        fontsize=10,
    )
    ax.set_xscale("symlog", linthresh=1.0)
    ax.grid(axis="x", alpha=0.2)

    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(color=tumor_color, label="tumor"),
        mpatches.Patch(color=tme_color_before, label="background — generic reference"),
        mpatches.Patch(color=tme_color_after, label="background — refined reference"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    cat_label = {
        "therapy_target": "Therapy-linked targets",
        "CTA": "CTAs",
        "surface": "Surface proteins",
    }.get(category, category.replace("_", " "))
    ax.set_title(
        f"Subtype-reference correction audit — {cat_label}\n"
        "Audit-only view: shows genes whose background estimate changed "
        "after subtype-specific refinement.",
        fontsize=9,
        fontweight="bold",
        loc="left",
    )

    plt.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig


def plot_tumor_expression_ranges(
    df_ranges,
    purity_result,
    cancer_type,
    top_n=15,
    categories=None,
    save_to_filename=None,
    save_dpi=300,
    figsize=None,
):
    """Strip plot of 9-point tumor expression estimates per gene.

    Parameters
    ----------
    df_ranges : DataFrame
        Output of ``estimate_tumor_expression_ranges()``.
    purity_result : dict
        Output of ``estimate_tumor_purity()``.
    cancer_type : str
        Cancer type code for title.
    top_n : int
        Max genes per category panel.
    categories : list of str, optional
        Which categories to plot. Default: CTA, therapy_target, surface.
    save_to_filename : str, optional
        Path to save the figure.
    """

    if categories is None:
        categories = ["therapy_target", "CTA", "surface"]

    cancer_code = resolve_cancer_type(cancer_type)
    p_lo = max(purity_result.get("overall_lower") or 0.01, 0.01)
    p_med = max(purity_result.get("overall_estimate") or 0.05, 0.01)
    p_hi = max(purity_result.get("overall_upper") or p_med, 0.01)

    cat_titles = {
        "CTA": "Cancer-Testis Antigens",
        "therapy_target": "Therapeutic Targets",
        "surface": "Surface Proteins",
        "other": "Other Tumor Genes",
    }
    cat_colors = {
        "CTA": "#e74c3c",
        "therapy_target": "#3498db",
        "surface": "#2ecc71",
        "other": "#95a5a6",
    }
    # Count genes per panel to size adaptively
    panel_counts = []
    for cat in categories:
        n = min(top_n, len(df_ranges[df_ranges["category"] == cat]))
        panel_counts.append(max(n, 1))
    total_genes = sum(panel_counts)

    n_panels = len(categories)
    if figsize is None:
        # ~0.4 inches per gene row, minimum 2 inches per panel
        panel_heights = [max(2.0, 0.4 * n) for n in panel_counts]
        figsize = (14, sum(panel_heights) + 1.5)

    fig, axes = plt.subplots(
        n_panels,
        2,
        figsize=figsize,
        squeeze=False,
        gridspec_kw={
            "width_ratios": [3, 1],
            "height_ratios": panel_counts,
        },
    )
    est_cols = [f"est_{i + 1}" for i in range(9)]

    # Adaptive font/marker sizes: larger when fewer genes
    base_font = min(12, max(8, int(200 / max(total_genes, 1))))
    marker_s = min(80, max(30, int(600 / max(total_genes, 1))))
    diamond_s = min(120, max(50, int(900 / max(total_genes, 1))))

    for ax_idx, cat in enumerate(categories):
        ax_strip = axes[ax_idx, 0]
        ax_pct = axes[ax_idx, 1]
        sub = df_ranges[df_ranges["category"] == cat].head(top_n).copy()
        sub = sub.sort_values("median_est", ascending=True).reset_index(drop=True)

        if sub.empty:
            ax_strip.set_title(cat_titles.get(cat, cat))
            ax_strip.text(
                0.5,
                0.5,
                "No genes",
                ha="center",
                va="center",
                transform=ax_strip.transAxes,
                fontsize=base_font,
                color="gray",
            )
            ax_pct.set_visible(False)
            continue

        color = cat_colors.get(cat, "#95a5a6")
        y_positions = np.arange(len(sub))

        # --- Left panel: 9-point strip plot ---
        for i, (_, row) in enumerate(sub.iterrows()):
            vals = [row[c] for c in est_cols]
            vals_plot = [max(v, 0.01) for v in vals]
            median_v = max(row["median_est"], 0.01)

            ax_strip.scatter(
                vals_plot, [i] * 9, color=color, alpha=0.4, s=marker_s, zorder=3
            )
            ax_strip.scatter(
                [median_v],
                [i],
                color=color,
                marker="D",
                s=diamond_s,
                edgecolors="black",
                linewidths=0.7,
                zorder=5,
            )
            ax_strip.plot(
                [min(vals_plot), max(vals_plot)],
                [i, i],
                color=color,
                alpha=0.3,
                linewidth=2.5,
                zorder=2,
            )

        labels = []
        for _, row in sub.iterrows():
            label = row["symbol"]
            if row["therapies"]:
                label += f"  [{row['therapies']}]"
            labels.append(label)

        ax_strip.set_yticks(y_positions)
        ax_strip.set_yticklabels(labels, fontsize=base_font)
        ax_strip.set_xscale("log")
        ax_strip.set_xlabel(
            "Tumor-cell-equivalent TPM (non-tumor subtracted, divided by purity)",
            fontsize=base_font,
        )
        ax_strip.set_title(
            cat_titles.get(cat, cat),
            fontsize=base_font + 2,
            fontweight="bold",
            color=color,
        )
        ax_strip.set_ylim(-0.5, len(sub) - 0.5)
        ax_strip.grid(axis="x", alpha=0.2)

        # --- Right panel: % of cancer type median (log scale) ---
        for i, (_, row) in enumerate(sub.iterrows()):
            pct = row.get("pct_cancer_median")
            if pct is None or (isinstance(pct, float) and np.isnan(pct)):
                # Gene not in TCGA reference for this cancer type
                ax_pct.text(
                    0.5,
                    i,
                    "0 in reference",
                    fontsize=base_font - 2,
                    color="gray",
                    ha="center",
                    va="center",
                    transform=ax_pct.get_yaxis_transform(),
                )
                continue
            if isinstance(pct, float) and np.isinf(pct):
                # Sample expresses the gene but the TCGA cohort reference is
                # zero -- so this gene is absent from TCGA {cancer_code} but
                # present in THIS sample. Draw a solid dark-red band across
                # the row with white text inside so the reader can't mistake
                # it for a tissue-decomposition claim or a general property
                # of the cancer type.
                ax_pct.axhspan(
                    i - 0.35,
                    i + 0.35,
                    color="#6b0000",
                    alpha=1.0,
                    zorder=3,
                    linewidth=0,
                )
                ax_pct.text(
                    0.5,
                    i,
                    f"absent in {cancer_code} reference",
                    fontsize=base_font - 2,
                    color="white",
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=4,
                    transform=ax_pct.get_yaxis_transform(),
                )
                continue
            bar_color = color if pct >= 0.5 else "#d4a017"
            ax_pct.barh(i, max(pct, 0.001), color=bar_color, alpha=0.7, height=0.6)
            lbl = f"{pct:.1f}\u00d7" if pct < 10 else f"{pct:.0f}\u00d7"
            ax_pct.text(
                max(pct, 0.001) * 1.2,
                i,
                lbl,
                fontsize=base_font - 1,
                va="center",
                color="black",
            )

        ax_pct.set_xscale("log")
        ax_pct.axvline(1.0, color="black", linestyle="--", alpha=0.4, linewidth=1)
        ax_pct.set_yticks([])
        ax_pct.set_xlabel(f"vs {cancer_code} reference median", fontsize=base_font)
        ax_pct.set_title(f"vs {cancer_code} reference", fontsize=base_font, color="gray")
        ax_pct.set_ylim(-0.5, len(sub) - 0.5)
        ax_pct.grid(axis="x", alpha=0.15)

    # Suptitle with purity info and caveat
    fig.suptitle(
        f"Tumor-cell-equivalent expression \u2014 {cancer_code}\n"
        f"Purity: {p_lo:.0%} / {p_med:.0%} / {p_hi:.0%} (low / est / high)\n"
        "Context-adjusted per-cell range; source-attributed bulk TPM is in tables.",
        fontsize=10,
        fontweight="bold",
        y=1.04,
    )
    plt.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")

    return fig


def plot_purity_adjusted_targets(
    df_gene_expr,
    cancer_type,
    purity,
    save_to_filename=None,
    save_dpi=300,
    figsize=(14, 10),
    top_n=40,
):
    """Plot modelled context expression for key gene categories.

    Shows observed vs context expression for CTAs, therapy targets,
    and surface proteins, with TCGA percentile context.
    """
    import pandas as pd

    adj = estimate_tumor_expression(df_gene_expr, cancer_type, purity)
    cancer_code = resolve_cancer_type(cancer_type)

    # Select top genes per category
    categories = ["CTA", "therapy_target", "surface"]
    selected = []
    for cat in categories:
        sub = adj[adj["category"] == cat].head(top_n // len(categories))
        selected.append(sub)
    selected = pd.concat(selected, ignore_index=True) if selected else adj.head(0)
    # Add high-expression "other" if space remains
    remaining = top_n - len(selected)
    if remaining > 0:
        other = adj[(adj["category"] == "other") & (adj["tumor_adjusted"] > 10)]
        selected = pd.concat([selected, other.head(remaining)], ignore_index=True)

    if selected.empty:
        return None

    selected = selected.sort_values(
        ["category", "tumor_adjusted"], ascending=[True, False]
    ).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1]}
    )

    # Left: horizontal bar chart of context expression
    y = np.arange(len(selected))
    cat_colors = {
        "CTA": "#e74c3c",
        "therapy_target": "#3498db",
        "surface": "#2ecc71",
        "other": "#95a5a6",
    }
    colors = [cat_colors.get(c, "#95a5a6") for c in selected["category"]]

    ax1.barh(y, selected["tumor_adjusted"], color=colors, alpha=0.8, height=0.7)
    # Overlay observed as dots
    ax1.scatter(
        selected["observed_tpm"], y, color="black", s=20, zorder=5, label="observed TPM"
    )
    ax1.set_yticks(y)
    labels = []
    for _, row in selected.iterrows():
        suffix = ""
        if row["is_surface"]:
            suffix += " [S]"
        if row["therapies"]:
            suffix += f" ({row['therapies']})"
        labels.append(f"{row['symbol']}{suffix}")
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Expression (TPM)")
    ax1.set_xscale("symlog", linthresh=1)
    ax1.set_title(
        f"Purity-adjusted tumor expression \u2014 {cancer_code} (purity={purity:.0%})"
    )
    ax1.invert_yaxis()
    ax1.legend(fontsize=8, loc="lower right")
    try:
        from .common import build_sample_tpm_by_symbol
        from .plot_reference_lines import add_p90_reference_line

        add_p90_reference_line(
            ax1,
            build_sample_tpm_by_symbol(df_gene_expr),
            orientation="vertical",
        )
    except Exception:
        pass

    # Right: TCGA percentile heatmap
    pctiles = selected["tcga_percentile"].values
    ax2.barh(y, pctiles, color=colors, alpha=0.8, height=0.7)
    ax2.set_xlim(0, 1)
    ax2.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set_yticks([])
    ax2.set_xlabel("TCGA percentile")
    ax2.set_title("vs TCGA cancer types")
    ax2.invert_yaxis()

    # Category legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#e74c3c", label="CTA (vaccination target)"),
        Patch(facecolor="#3498db", label="Therapy target (in trials)"),
        Patch(facecolor="#2ecc71", label="Surface protein"),
        Patch(facecolor="#95a5a6", label="Other tumor gene"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")

    return fig
