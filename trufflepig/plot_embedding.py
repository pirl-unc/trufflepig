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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from adjustText import adjust_text

from .common import _guess_gene_cols
from .plot_data_helpers import _strip_ensembl_version
from pirlygenes.gene_sets_cancer import (
    housekeeping_gene_ids, therapy_target_gene_id_to_name,
)
from trufflepig.reference import (
    pan_cancer_expression,
)
from pirlygenes.load_dataset import get_data
from .plot_scatter import CANCER_TYPE_NAMES
from .plot_tumor_expr import (
    _sample_expression_by_symbol,
    _REPRODUCTIVE_TISSUES,
    _STROMAL_TISSUES,
)


_IG_TR_PREFIXES = ("IGH", "IGK", "IGL", "TRA", "TRB", "TRG", "TRD")
_CURATED_CTAS = [
    "PRAME",  # UCS/SKCM/UCEC — high-expression melanoma/uterine marker
    "MAGEA3",  # SKCM/LUSC — melanoma and squamous cancers
    "CTCFL",  # UCS/OV — gynecologic cancers (BORIS, CTCF paralog)
    "SMC1B",  # CESC/LAML — meiotic cohesin, cervical cancer marker
    "LIN28B",  # TGCT/UCS — stem cell / embryonal marker
    "SSX1",  # THCA/UVM/SKCM — thyroid + melanomas
    "C1orf94",  # LGG/GBM — brain tumor CTA marker
    "SYCP3",  # LAML/TGCT — synaptonemal complex, meiosis marker
    "FATE1",  # ACC — adrenocortical CTA marker
]

# Clinically important genes with low TME background.  The data-driven
# algorithm may miss these because their best z-score peaks in a sibling
# type or they sit just below per-type selection cutoffs.
_CURATED_TME_BOOST = [
    # Cancer-testis antigens
    "PRAME",  # S/N_tme≈179  UCS/UCEC/SKCM — immunotherapy target
    "MAGEA3",  # S/N_tme≈27   SKCM — vaccine target
    "LIN28B",  # S/N_tme≈19   TGCT — embryonal/stem marker
    "SYCP3",  # S/N_tme≈12   LAML — meiosis marker
    # Glioma / neuroendocrine (brain-restricted, TME-silent)
    "DLL3",  # S/N_tme≈94   LGG/GBM — Rova-T & BiTE target
    "PTPRZ1",  # S/N_tme≈29   LGG/GBM — glioma phosphatase
    # Melanocyte lineage (melanocyte-restricted)
    "MLANA",  # S/N_tme≈1472 UVM/SKCM — MART-1, TIL/TCR target
    "TYR",  # S/N_tme≈726  UVM/SKCM — tyrosinase
    # Therapy targets with low TME expression
    "MSLN",  # S/N_tme≈10   MESO/OV — mesothelin, CAR-T target
    "CDKN2A",  # S/N_tme≈13   UCS/OV — p16, broad tumor marker
    "COL11A1",  # S/N_tme≈11   MESO/BRCA/PAAD — desmoplastic, ADC target
    # Lineage transcription factors
    "FOXA1",  # S/N_tme≈5    PRAD/BRCA — luminal breast/prostate
    "ASCL2",  # S/N_tme≈10   COAD/READ — intestinal stem cell TF
    "DLX5",  # S/N_tme≈9    UCEC/UCS — homeobox, endometrial
    "SOX2",  # S/N_tme≈22   LUSC/LGG — squamous & neural stem cell TF
    "TH",  # S/N_tme≈792  PCPG — tyrosine hydroxylase, catecholamine
    # Therapy targets & lineage markers for additional types
    "FLT3",  # S/N_tme≈47   LAML — gilteritinib/midostaurin target
    "LIN28A",  # S/N_tme≈319  TGCT — embryonal pluripotency marker
    "NANOG",  # S/N_tme≈111  TGCT — pluripotency TF, germ cell tumors
    "PMEL",  # S/N_tme≈104  UVM/SKCM — gp100, melanoma vaccine target
    "OLIG2",  # S/N_tme≈10   LGG/GBM — oligodendrocyte lineage TF
    "NKX3-1",  # S/N_tme≈6    PRAD — prostate lineage TF, diagnostic
    "STEAP2",  # S/N_tme≈4    PRAD — prostate surface antigen
    "MITF",  # S/N_tme≈4    UVM/SKCM — master melanocyte TF
    "FOXN1",  # S/N_tme≈8    THYM — thymic epithelial TF
]

# Lineage markers for cancer types whose defining genes are also expressed
# in normal tissue (high TME background).  NOT TME-low — these won't help
# at very low purity, but they prevent these types from collapsing into a
# featureless cluster in embedding space.
_CURATED_LINEAGE_BOOST = [
    "UPK2",  # BLCA — uroplakin, 120x vs other cancers
    "APOA2",  # LIHC — liver secretory protein, 70000x vs others
    "SFTPB",  # LUAD — surfactant protein B, 12000x vs others
    "NAPSA",  # LUAD — napsin A, lung adeno IHC marker
    "KRT6A",  # ESCA — squamous keratin, 214x vs others
    "PGC",  # STAD — pepsinogen C, best available (3x vs others)
]

_CURATED_PAN_REFERENCE_ANCHORS = [
    # Immune / stromal / vascular tissue axes
    "PTPRC",
    "CD3D",
    "MS4A1",
    "MZB1",
    "COL1A1",
    "DCN",
    "PECAM1",
    "VWF",
    "ACTA2",
    "MYH11",
    "ADIPOQ",
    # Epithelial and organ-lineage axes
    "EPCAM",
    "KRT8",
    "KRT18",
    "KRT5",
    "KRT14",
    "CDX2",
    "MUC2",
    "ALB",
    "APOA1",
    "SFTPB",
    "NAPSA",
    "KLK3",
    "NKX3-1",
    "UPK2",
    "PAX8",
    "TG",
    "GATA3",
    # Neural / endocrine / melanocytic / mesenchymal axes
    "GFAP",
    "OLIG2",
    "SOX10",
    "MLANA",
    "PMEL",
    "TH",
    "CHGA",
    "INS",
    "TBXT",
    "COL2A1",
    "ACAN",
]

_signature_panel_cache = {}
_embedding_gene_cache = {}
_tme_gene_cache = {}
_bottleneck_gene_cache = {}
_pan_reference_gene_cache = {}
_hierarchy_feature_cache = {}
_hierarchy_site_cache = {}
_hierarchy_family_cache = {}
_tcga_parent_code_cache = None


def _tcga_parent_codes():
    global _tcga_parent_code_cache
    if _tcga_parent_code_cache is None:
        ref = pan_cancer_expression(technical_rna_normalize=True)
        _tcga_parent_code_cache = {
            c.removesuffix("_TPM") for c in ref.columns if c.endswith("_TPM")
        }
    return _tcga_parent_code_cache


def _get_cancer_type_signature_panels(n_signature_genes=20):
    """Return robust per-cancer signature panels used for ranking and plotting.

    Reuses `_select_tumor_specific_genes_for_panel` from `tumor_purity` so
    the plotting / ranking paths see the same tier-filtered, family-excluded
    panels as the purity estimator. When the strict tiers leave a cancer
    type short of `n_signature_genes`, a z-score fallback is appended — but
    that fallback still respects the configured family-exclusion regex and
    the `immune_origin_cancer_types` bypass, so the two paths never silently
    diverge on borderline cases.

    Cache is keyed on a fingerprint of TUMOR_PURITY_PARAMETERS so tuning
    parameters in-process invalidates stale panels.
    """
    from .tumor_purity import (
        TUMOR_PURITY_PARAMETERS,
        _cached_reference_matrices,
        _compile_excluded_gene_matcher,
        _params_fingerprint,
        _select_tumor_specific_genes_for_panel,
    )

    params_fp = _params_fingerprint(["tumor_specific_markers"])
    cache_key = (int(n_signature_genes), params_fp)
    cached = _signature_panel_cache.get(cache_key)
    if cached is not None:
        return {code: list(genes) for code, genes in cached.items()}

    ref_matrices = _cached_reference_matrices(normalize="housekeeping")
    cohort_cols = ref_matrices["cohort_cols"]
    expr_matrix = ref_matrices["expr_matrix"]
    z_matrix = ref_matrices["z_matrix"]

    is_excluded_default = _compile_excluded_gene_matcher()
    immune_origin = set(
        TUMOR_PURITY_PARAMETERS["tumor_specific_markers"].get(
            "immune_origin_cancer_types", []
        )
        or []
    )

    panels = {}
    for col in cohort_cols:
        code = col.removesuffix("_TPM")
        genes = _select_tumor_specific_genes_for_panel(
            code,
            n=n_signature_genes,
            exclude_lineage=False,
        )
        if len(genes) >= n_signature_genes:
            panels[code] = genes[:n_signature_genes]
            continue

        # Fallback top-up: raw top-N-by-z-score, but still respecting the
        # configured family exclusion so the plotting panel can't silently
        # pick up MT-* / rearranged-receptor genes that the purity path
        # just filtered out. Immune-origin codes skip the filter for the
        # same reason the purity path does.
        is_excluded = (
            (lambda _sym: False) if code in immune_origin else is_excluded_default
        )
        z_col = z_matrix[col]
        expr_col = expr_matrix[col]
        fallback_candidates = z_col[expr_col > 0.01].nlargest(n_signature_genes * 3)
        for gene in fallback_candidates.index:
            if gene in genes:
                continue
            if is_excluded(gene):
                continue
            genes.append(gene)
            if len(genes) >= n_signature_genes:
                break
        panels[code] = genes[:n_signature_genes]

    _signature_panel_cache[cache_key] = {
        code: tuple(genes) for code, genes in panels.items()
    }
    return panels


# Detection floor (housekeeping-relative) below which a signature gene is treated
# as not expressed and so cannot contribute positive cross-cancer-percentile
# evidence — see the "zero-floor percentile inflation" fix below. Anchored at the
# RNA-seq quantifier-noise boundary: ~0.1× the housekeeping median is ≈1–2 TPM for
# a typical HK level — below robust detection. The pathological cases sit at
# ~0.01–0.05 HK (clear noise); genuine markers at ≥1 HK. Set at the noise/detection
# boundary, not tuned to any sample (it must stay low enough that a genuinely-but-
# weakly-expressed marker of the true type is NOT clamped).
_SIGNATURE_DETECTION_FLOOR_HK = 0.1

# Weight of the within-sample-percentile leg in the combined signature filter. cohort-pct (HK) is
# DOMINANT; within-pct enters as a [1-w, 1] factor. 0.0 = pure cohort-pct (pre-#7); 1.0 = full product.
# Set to 1.0 (full product): the ALL-SAMPLES 565 eval (the authority — medoids mislead) showed w=0.5
# is strictly WORSE than w=1.0 on every metric incl. exact (242 vs 246 exact, 409 vs 414 entity, 523 vs
# 526 lineage). The single-medoid "COAD>READ recovered exact" smoke was an artifact; on 565 samples the
# full product wins. (Cohort leg stays HK: switching to raw clean-TPM regressed -4 — HK bridges the
# different clean-TPM scales of sample vs reference; it is load-bearing here, not redundant.)
_WITHIN_PCT_WEIGHT = 1.0


def _compute_cancer_type_signature_stats(
    df_gene_expr,
    n_signature_genes=20,
    min_fold=2.0,
    candidate_codes=None,
):
    """Score each cancer type by how well the sample matches its signature genes.

    Uses z-score–based gene selection (most specifically expressed genes per
    cancer type) and midrank percentile scoring — the sample's expression of
    each signature gene is ranked against the cross-cancer distribution.
    This is robust to per-sample total-expression differences.
    """
    import numpy as np

    from .tumor_purity import _cached_reference_matrices

    sample_raw_by_symbol, sample_hk_by_symbol = _sample_expression_by_symbol(
        df_gene_expr
    )
    # Per-gene score basis: the COMBINED filter cross-cohort-percentile x within-sample-percentile
    # (two weak filters multiplied). The cohort-percentile leg supplies SPECIFICITY + the score spread
    # the ranker's margins need; the within-sample-percentile leg supplies purity-ROBUSTNESS (a
    # within-sample rank of curated markers survives proportional dilution, whereas cross-cohort
    # percentile is purity-SENSITIVE — dilution shifts where the sample falls in the cohort
    # distribution). A/B'd on 392 rep samples + dilution (scripts/signature_basis_ab.py) and validated
    # end-to-end: the medoid AUTO cancer-type lineage rises 95->99/118 vs the HK-percentile basis
    # (within-sample-pct ALONE regressed -3 because its compressed scores collapse the ranker margins;
    # the product keeps the spread). Both legs in [0,1] -> product in [0,1].
    ref_matrices = _cached_reference_matrices(normalize="housekeeping")
    ref_by_sym = ref_matrices["ref_by_sym"]
    expr_matrix = ref_matrices["expr_matrix"]
    sig = _get_cancer_type_signature_panels(n_signature_genes=n_signature_genes)

    # Within-sample percentile (average-rank midrank, pct) of the sample's raw clean-TPM, once.
    import pandas as _pd

    _within_pct = (
        _pd.Series(sample_raw_by_symbol, dtype=float).rank(pct=True, method="average").to_dict()
        if sample_raw_by_symbol
        else {}
    )

    stats = []
    if candidate_codes is None:
        codes_to_score = sorted(sig.keys())
    else:
        requested = {str(code) for code in candidate_codes}
        codes_to_score = sorted(code for code in sig.keys() if code in requested)

    for code in codes_to_score:
        genes = sig[code]
        cohort_col = f"{code}_TPM"
        gene_details = []
        percentiles = []
        for gene in genes:
            sample_raw = float(sample_raw_by_symbol.get(gene, 0.0))
            sample_hk = float(sample_hk_by_symbol.get(gene, 0.0))
            cohort_hk = 0.0
            cohort_pct = 0.5
            if gene in ref_by_sym.index and cohort_col in ref_by_sym.columns:
                cohort_hk = float(ref_by_sym.loc[gene, cohort_col])
                ref_vals = expr_matrix.loc[gene].values  # midrank cross-cohort percentile (HK)
                n = len(ref_vals)
                below = np.sum(ref_vals < sample_hk)
                equal = np.sum(np.isclose(ref_vals, sample_hk, atol=1e-6))
                cohort_pct = float((below + 0.5 * equal) / n)
                if sample_hk < _SIGNATURE_DETECTION_FLOOR_HK:  # zero-floor inflation guard
                    cohort_pct = min(cohort_pct, 0.5)
            # COMBINED filter (two weak filters): cross-cohort percentile (specificity + the
            # discrimination SPREAD the ranker margins need) modulated by within-sample percentile
            # (dominance, purity-robust — a within-sample rank survives proportional dilution).
            # cohort-pct DOMINANT; within-pct is a [1-w, 1] factor (w=_WITHIN_PCT_WEIGHT) so it never
            # zeros the cohort discrimination and the score stays near the cohort-pct scale the
            # thresholds expect. A/B'd in scripts/signature_basis_ab.py; w=0.5 validated (medoid AUTO
            # lineage 95→99, exact recovered vs the full product). Both legs [0,1] → result [0,1].
            within_pct = float(_within_pct.get(gene, 0.5))
            percentile = cohort_pct * ((1.0 - _WITHIN_PCT_WEIGHT) + _WITHIN_PCT_WEIGHT * within_pct)
            percentiles.append(percentile)
            log_diff = abs(np.log2(sample_hk + 1) - np.log2(cohort_hk + 1))
            gene_details.append(
                {
                    "gene": gene,
                    "sample_raw": sample_raw,
                    "sample_hk": sample_hk,
                    "cohort_hk": cohort_hk,
                    "log_diff": log_diff,
                    "percentile": percentile,
                }
            )

        if gene_details:
            score = float(np.mean(percentiles))
            mean_sample_raw = float(np.mean([g["sample_raw"] for g in gene_details]))
        else:
            score = 0.0
            mean_sample_raw = 0.0

        stats.append(
            {
                "code": code,
                "genes": genes,
                "n_genes": len(genes),
                "score": score,
                "mean_sample_raw": mean_sample_raw,
                "gene_details": gene_details,
            }
        )

    stats.sort(key=lambda row: (-row["score"], row["code"]))
    for rank, row in enumerate(stats, start=1):
        row["rank"] = rank
    return stats


# ---------------------------------------------------------------------------
# Unified embedding gene selection
# ---------------------------------------------------------------------------


def _select_embedding_genes_bottleneck(n_genes_per_type=5):
    """Select genes for cancer-type embedding using bottleneck scoring.

    For each gene × cancer type, two z-scores are computed:

    * **z_tme** — how far the gene's cancer-type expression is above
      the distribution of TME (immune + stromal) tissue expression.
      High z_tme ⇒ gene visible above microenvironment background.

    * **z_other** — how far the gene's cancer-type expression is above
      the distribution of *all* cancer types.
      High z_other ⇒ gene is specific to this cancer type.

    The combined score is ``min(z_tme, z_other)`` — the bottleneck.
    A gene ranks high only if it scores well on *both* axes.  This
    naturally balances purity robustness (TME silence) against
    cancer-type discrimination without any hard threshold on either.

    Evaluation on 160 individual TCGA samples diluted 1:1 with GTEx
    immune expression (simulating 5% tumor purity) showed this method
    achieves 56% top-1 / 76% top-5 nearest-neighbor accuracy with only
    158 genes — the best purity-robust performance tested.  See
    ``eval/`` for the full comparison across 21 gene sets, 9
    normalizations, and 5 purity levels.

    Parameters
    ----------
    n_genes_per_type : int
        Number of top-scoring genes to select per cancer type (default 5).

    Returns
    -------
    ref_filtered : DataFrame
        Subset of pan-cancer reference for the selected genes.
    metadata : dict
        Per-type gene lists, total gene count, etc.
    """
    import numpy as np

    cache_key = n_genes_per_type
    if cache_key in _bottleneck_gene_cache:
        return _bottleneck_gene_cache[cache_key]

    ref = pan_cancer_expression(technical_rna_normalize=True)
    cohort_cols = [c for c in ref.columns if c.endswith("_TPM")]
    ntpm_cols = [c for c in ref.columns if c.endswith("_nTPM")]
    ntpm_nonrepro = [
        c for c in ntpm_cols if c.removesuffix("_nTPM") not in _REPRODUCTIVE_TISSUES
    ]

    ref_dedup = ref.drop_duplicates(subset="Symbol")
    cancer_expr = ref_dedup[cohort_cols].astype(float)

    # Identify TME tissues (immune via PTPRC, plus stromal)
    ptprc_row = ref_dedup[ref_dedup["Symbol"] == "PTPRC"]
    if len(ptprc_row):
        ptprc_vals = ptprc_row[ntpm_nonrepro].astype(float).iloc[0]
        immune_cols = [c for c in ntpm_nonrepro if ptprc_vals[c] > ptprc_vals.median()]
    else:
        immune_cols = []
    stromal_cols = [
        c for c in ntpm_nonrepro if c.removesuffix("_nTPM") in _STROMAL_TISSUES
    ]
    tme_cols = list(set(immune_cols + stromal_cols))
    tme_expr = ref_dedup[tme_cols].astype(float)

    # IG/TR exclusion
    is_rearranged = ref_dedup["Symbol"].apply(
        lambda s: any(s.startswith(p) for p in _IG_TR_PREFIXES)
    )

    # Log-transform
    log_cancer = np.log2(cancer_expr + 1)
    log_tme = np.log2(tme_expr + 1)

    # z_tme: z-score of cancer expr against TME tissue distribution
    tme_mean = log_tme.mean(axis=1)
    tme_std = log_tme.std(axis=1).replace(0, 0.1)
    z_tme = log_cancer.sub(tme_mean.values, axis=0).div(tme_std.values, axis=0)

    # z_other: z-score of cancer expr against all cancer types
    cancer_mean = log_cancer.mean(axis=1)
    cancer_std = log_cancer.std(axis=1).replace(0, 0.1)
    z_other = log_cancer.sub(cancer_mean.values, axis=0).div(cancer_std.values, axis=0)

    # Bottleneck score: min of the two positive z-scores
    z_tme_pos = z_tme.clip(lower=0)
    z_other_pos = z_other.clip(lower=0)
    import pandas as pd

    bottleneck = pd.DataFrame(
        np.minimum(z_tme_pos.values, z_other_pos.values),
        index=cancer_expr.index,
        columns=cancer_expr.columns,
    )

    # Select top genes per type
    selected_idx = []
    per_type = {}
    for col in cohort_cols:
        code = col.removesuffix("_TPM")
        mask = (cancer_expr[col] > 0.5) & (~is_rearranged.values)
        valid = bottleneck[col][mask]
        top = valid.nlargest(n_genes_per_type)
        syms = list(ref_dedup.loc[top.index, "Symbol"].values)
        per_type[code] = syms
        selected_idx.extend(top.index)

    selected_idx = list(dict.fromkeys(selected_idx))
    ref_filtered = ref_dedup.loc[selected_idx]

    metadata = {
        "per_type": per_type,
        "n_genes": len(selected_idx),
        "n_types": len([t for t, g in per_type.items() if g]),
        "method": "bottleneck",
        "tme_tissues": sorted(c.removesuffix("_nTPM") for c in tme_cols),
    }

    result = (ref_filtered, metadata)
    _bottleneck_gene_cache[cache_key] = result
    return result


def _select_pan_reference_genes(n_genes_per_type=6, n_genes_per_normal=None):
    """Select genes for a shared cancer/subtype/normal reference embedding.

    The bottleneck and TME-low selectors are optimized for tumor calling,
    especially at low purity. A normal-inclusive dimensionality-reduction
    plot needs a different feature space: every TCGA median and every HPA
    normal tissue should have enough markers to become a meaningful point.

    This selector therefore takes top genes for each cancer centroid and
    each normal-tissue centroid using the same specificity score, then adds
    a compact set of curated tissue/lineage anchors. It is meant as a
    general reference-map gene set rather than a classifier.
    """
    import numpy as np
    n_genes_per_type = int(n_genes_per_type)
    n_genes_per_normal = (
        max(3, n_genes_per_type // 2)
        if n_genes_per_normal is None
        else int(n_genes_per_normal)
    )
    cache_key = (n_genes_per_type, n_genes_per_normal)
    if cache_key in _pan_reference_gene_cache:
        return _pan_reference_gene_cache[cache_key]

    # Disk cache: the entire computation depends only on the pirlygenes
    # cohort matrix (fixed per package version), so the per-subprocess
    # ~10–20 min cost — iterating 70 cohort cols + 50 normal-tissue cols
    # and computing median/quantile/MAD over a 20k-gene × 119-column
    # DataFrame for each — can be amortized across runs by pickling.
    # First subprocess pays the cost, every subsequent subprocess loads
    # from disk in <1s. Cache key includes the pirlygenes version so
    # an upgrade invalidates correctly.
    import os as _os
    import pickle as _pickle
    from pathlib import Path as _Path
    try:
        import pirlygenes as _pirlygenes
        _pirlygenes_version = str(_pirlygenes.__version__)
    except Exception:  # noqa: BLE001
        _pirlygenes_version = "unknown"
    try:
        from .version import __version__ as _trufflepig_version
    except Exception:  # noqa: BLE001
        _trufflepig_version = "unknown"
    # Cache key includes BOTH versions so any change in either side's
    # logic invalidates correctly. Without trufflepig's version, a
    # change to ``_select_pan_reference_genes`` (e.g. modified scoring
    # weights) without a pirlygenes bump would silently serve stale
    # data from the cache.
    _cache_root = _Path(
        _os.environ.get("XDG_CACHE_HOME") or str(_Path.home() / ".cache")
    ) / "trufflepig"
    _disk_cache_path = (
        _cache_root
        / (
            f"pan_reference_genes-"
            f"pirlygenes-{_pirlygenes_version}-"
            f"trufflepig-{_trufflepig_version}-"
            f"{n_genes_per_type}-{n_genes_per_normal}.pkl"
        )
    )
    if _disk_cache_path.exists():
        try:
            with _disk_cache_path.open("rb") as _f:
                result = _pickle.load(_f)
            _pan_reference_gene_cache[cache_key] = result
            return result
        except Exception:  # noqa: BLE001 — fall through to recompute on any cache error
            pass

    ref = pan_cancer_expression(technical_rna_normalize=True)
    cohort_cols = [c for c in ref.columns if c.endswith("_TPM")]
    ntpm_cols = [c for c in ref.columns if c.endswith("_nTPM")]
    ref_dedup = ref.drop_duplicates(subset="Symbol").copy()
    reference_cols = cohort_cols + ntpm_cols
    expr = ref_dedup[reference_cols].astype(float)
    log_expr = np.log2(expr + 1.0)

    is_rearranged = ref_dedup["Symbol"].apply(
        lambda s: any(str(s).startswith(p) for p in _IG_TR_PREFIXES)
    )
    base_mask = (~is_rearranged.values) & (expr.max(axis=1) > 0.2)

    selected_idx = []
    per_type = {}
    per_normal = {}

    def _top_for_column(column, n):
        target = log_expr[column]
        others = log_expr.drop(columns=[column])
        other_median = others.median(axis=1)
        other_q3 = others.quantile(0.75, axis=1)
        other_mad = (
            others.sub(other_median, axis=0)
            .abs()
            .median(axis=1)
            .replace(0, np.nan)
        )
        scale = (1.4826 * other_mad).fillna(0.15).clip(lower=0.15)
        z_specific = (target - other_median) / scale
        delta = target - other_q3
        score = (z_specific.clip(lower=0.0) + delta.clip(lower=0.0)) * target
        mask = base_mask & (target > np.log2(1.2)) & (delta > 0.0)
        top = score[mask].nlargest(n)
        return list(top.index)

    for col in cohort_cols:
        code = col.removesuffix("_TPM")
        top_idx = _top_for_column(col, n_genes_per_type)
        per_type[code] = list(ref_dedup.loc[top_idx, "Symbol"].astype(str))
        selected_idx.extend(top_idx)

    for col in ntpm_cols:
        tissue = col.removesuffix("_nTPM")
        top_idx = _top_for_column(col, n_genes_per_normal)
        per_normal[tissue] = list(ref_dedup.loc[top_idx, "Symbol"].astype(str))
        selected_idx.extend(top_idx)

    selected_syms = set(
        ref_dedup.loc[list(dict.fromkeys(selected_idx)), "Symbol"].astype(str)
        if selected_idx
        else []
    )
    anchor_added = []
    for sym in _CURATED_PAN_REFERENCE_ANCHORS:
        if sym in selected_syms:
            continue
        hits = ref_dedup[ref_dedup["Symbol"].astype(str) == sym]
        if hits.empty:
            continue
        idx = hits.index[0]
        selected_idx.append(idx)
        selected_syms.add(sym)
        anchor_added.append(sym)

    selected_idx = list(dict.fromkeys(selected_idx))
    ref_filtered = ref_dedup.loc[selected_idx]

    metadata = {
        "method": "panref",
        "feature_kind": "pan_reference_genes",
        "per_type": per_type,
        "per_normal": per_normal,
        "anchor_added": anchor_added,
        "n_genes": len(selected_idx),
        "n_types": len([t for t, genes in per_type.items() if genes]),
        "n_normals": len([t for t, genes in per_normal.items() if genes]),
        "n_genes_per_type": n_genes_per_type,
        "n_genes_per_normal": n_genes_per_normal,
    }

    result = (ref_filtered, metadata)
    _pan_reference_gene_cache[cache_key] = result
    # Persist to disk for subsequent subprocesses. Best-effort: write to a
    # temp file and rename atomically so a crash mid-write doesn't leave a
    # corrupt cache. Failures here are non-fatal — the next subprocess
    # just recomputes.
    try:
        _cache_root.mkdir(parents=True, exist_ok=True)
        _tmp = _disk_cache_path.with_suffix(_disk_cache_path.suffix + ".tmp")
        with _tmp.open("wb") as _f:
            _pickle.dump(result, _f, protocol=_pickle.HIGHEST_PROTOCOL)
        _tmp.replace(_disk_cache_path)
    except Exception:  # noqa: BLE001
        pass
    return result


def pan_reference_embedding_genes(n_genes_per_type=6, n_genes_per_normal=None):
    """Return the reusable pan-reference embedding gene set.

    The returned dataframe contains one row per selected gene and includes
    provenance columns naming the TCGA cancer medians, HPA normal tissues,
    or curated anchor list that selected the gene.
    """
    ref_filtered, metadata = _select_pan_reference_genes(
        n_genes_per_type=n_genes_per_type,
        n_genes_per_normal=n_genes_per_normal,
    )
    cancer_refs_by_gene = {}
    normal_refs_by_gene = {}
    for code, genes in metadata["per_type"].items():
        for gene in genes:
            cancer_refs_by_gene.setdefault(gene, []).append(code)
    for tissue, genes in metadata["per_normal"].items():
        for gene in genes:
            normal_refs_by_gene.setdefault(gene, []).append(tissue)

    columns = [c for c in ["Ensembl_Gene_ID", "Symbol"] if c in ref_filtered.columns]
    result = ref_filtered.loc[:, columns].copy()
    result["selected_for_cancer_refs"] = result["Symbol"].map(
        lambda gene: ";".join(cancer_refs_by_gene.get(gene, []))
    )
    result["selected_for_normal_refs"] = result["Symbol"].map(
        lambda gene: ";".join(normal_refs_by_gene.get(gene, []))
    )
    result["curated_anchor"] = result["Symbol"].isin(metadata["anchor_added"])
    return result.reset_index(drop=True)


def _select_tme_low_genes(n_genes_per_type=3, sn_tme_threshold=10):
    """Select genes with low tumor microenvironment (TME) background.

    These genes are silent in immune and stromal cells, so their signal
    is detectable even at very low tumor purity (down to ~5%).

    Parameters
    ----------
    sn_tme_threshold : float
        Minimum ratio of cancer expression to TME tissue expression.
        Default 10 means genes are visible at ~10% purity.
    """
    import numpy as np

    cache_key = (n_genes_per_type, sn_tme_threshold)
    if cache_key in _tme_gene_cache:
        return _tme_gene_cache[cache_key]

    ref = pan_cancer_expression(technical_rna_normalize=True)
    cohort_cols = [c for c in ref.columns if c.endswith("_TPM")]
    ntpm_cols = [c for c in ref.columns if c.endswith("_nTPM")]
    ntpm_nonrepro = [
        c for c in ntpm_cols if c.removesuffix("_nTPM") not in _REPRODUCTIVE_TISSUES
    ]

    ref_dedup = ref.drop_duplicates(subset="Symbol")
    cancer_expr = ref_dedup[cohort_cols].astype(float)
    normal_expr = ref_dedup[ntpm_nonrepro].astype(float)

    # Immune tissues (PTPRC-defined) + stromal tissues = TME background
    ptprc_row = ref_dedup[ref_dedup["Symbol"] == "PTPRC"]
    if len(ptprc_row):
        ptprc_vals = ptprc_row[ntpm_nonrepro].astype(float).iloc[0]
        immune_cols = [c for c in ntpm_nonrepro if ptprc_vals[c] > ptprc_vals.median()]
    else:
        immune_cols = []
    stromal_cols = [
        c for c in ntpm_nonrepro if c.removesuffix("_nTPM") in _STROMAL_TISSUES
    ]
    tme_cols = list(set(immune_cols + stromal_cols))
    tme_max = normal_expr[tme_cols].max(axis=1) if tme_cols else normal_expr.max(axis=1)

    sn_tme = cancer_expr.max(axis=1) / (tme_max + 0.01)

    # IG/TR exclusion
    is_rearranged = ref_dedup["Symbol"].apply(
        lambda s: any(s.startswith(p) for p in _IG_TR_PREFIXES)
    )

    # Z-scores for cancer-type specificity
    g_mean = cancer_expr.mean(axis=1)
    g_std_raw = cancer_expr.std(axis=1).replace(0, np.nan)
    z_mat = cancer_expr.sub(g_mean, axis=0).div(g_std_raw, axis=0).fillna(0)
    best_z = z_mat.max(axis=1)

    base_mask = (~is_rearranged.values) & (cancer_expr.max(axis=1) > 1) & (best_z > 1)

    # Tiered selection: strict S/N first, then relax for underrepresented types
    selected_idx = []
    per_type = {}
    covered = set()

    for tier_thresh in [sn_tme_threshold, 3, 1.5]:
        tier_mask = base_mask & (sn_tme > tier_thresh)
        tier_best_cancer = z_mat[tier_mask].idxmax(axis=1)
        for code_col in cohort_cols:
            code = code_col.removesuffix("_TPM")
            if code in covered:
                continue
            genes = tier_best_cancer[tier_best_cancer == code_col].index
            top = best_z.loc[genes].nlargest(n_genes_per_type).index
            if len(top):
                syms = list(ref_dedup.loc[top, "Symbol"].values)
                per_type[code] = syms
                selected_idx.extend(top)
                covered.add(code)

    # Final fallback: composite score for any still-missing types
    fallback_mask = base_mask & (sn_tme > 0.5)
    if fallback_mask.any():
        fallback_score = (
            best_z[fallback_mask]
            * np.log2(cancer_expr.max(axis=1)[fallback_mask] + 1)
            * np.minimum(sn_tme[fallback_mask], 3)
            / 3
        )
        fallback_best = z_mat[fallback_mask].idxmax(axis=1)
        for code_col in cohort_cols:
            code = code_col.removesuffix("_TPM")
            if code in covered:
                continue
            genes = fallback_best[fallback_best == code_col].index
            top = fallback_score.loc[genes].nlargest(n_genes_per_type).index
            syms = list(ref_dedup.loc[top, "Symbol"].values) if len(top) else []
            per_type[code] = syms
            selected_idx.extend(top)
            covered.add(code)

    # Fill any remaining types with empty
    for code_col in cohort_cols:
        code = code_col.removesuffix("_TPM")
        if code not in per_type:
            per_type[code] = []

    selected_idx = list(dict.fromkeys(selected_idx))

    # --- Curated boost: clinically important TME-low markers ---
    selected_syms = set(ref_dedup.loc[selected_idx, "Symbol"].values)
    boost_added = []
    for sym in _CURATED_TME_BOOST:
        if sym in selected_syms:
            continue
        hits = ref_dedup[ref_dedup["Symbol"] == sym]
        if hits.empty:
            continue
        idx = hits.index[0]
        gene_sn = sn_tme.loc[idx]
        gene_expr = cancer_expr.loc[idx].max()
        if gene_sn > 3 and gene_expr > 1:
            selected_idx.append(idx)
            selected_syms.add(sym)
            boost_added.append(sym)

    # --- Lineage boost: high-discrimination markers for types without TME-low genes ---
    lineage_added = []
    for sym in _CURATED_LINEAGE_BOOST:
        if sym in selected_syms:
            continue
        hits = ref_dedup[ref_dedup["Symbol"] == sym]
        if hits.empty:
            continue
        idx = hits.index[0]
        gene_expr = cancer_expr.loc[idx].max()
        if gene_expr > 5:
            selected_idx.append(idx)
            selected_syms.add(sym)
            lineage_added.append(sym)

    selected_idx = list(dict.fromkeys(selected_idx))
    ref_filtered = ref_dedup.loc[selected_idx]

    metadata = {
        "per_type": per_type,
        "boost_added": boost_added,
        "lineage_added": lineage_added,
        "n_genes": len(selected_idx),
        "n_types": len([t for t, g in per_type.items() if g]),
        "sn_tme_threshold": sn_tme_threshold,
        "tme_tissues": sorted(c.removesuffix("_nTPM") for c in tme_cols),
    }

    result = (ref_filtered, metadata)
    _tme_gene_cache[cache_key] = result
    return result


def _select_embedding_genes(n_genes_per_type=3):
    """Select a unified gene set for cancer-type embeddings.

    Applies biologically-informed filters to select genes that discriminate
    cancer types without being confounded by immune infiltrate or normal
    tissue contamination.

    Returns
    -------
    ref_filtered : DataFrame
        Subset of pan-cancer reference data for the selected genes.
    metadata : dict
        Per-cancer-type gene lists, excluded genes, CTA additions, etc.
    """
    import numpy as np

    cache_key = n_genes_per_type
    if cache_key in _embedding_gene_cache:
        return _embedding_gene_cache[cache_key]

    ref = pan_cancer_expression(technical_rna_normalize=True)
    cohort_cols = [c for c in ref.columns if c.endswith("_TPM")]
    ntpm_cols = [c for c in ref.columns if c.endswith("_nTPM")]

    # Exclude reproductive tissues from the normal-tissue denominator
    # so that cancer-testis antigens can pass the S/N filter.
    ntpm_nonrepro = [
        c for c in ntpm_cols if c.removesuffix("_nTPM") not in _REPRODUCTIVE_TISSUES
    ]

    ref_dedup = ref.drop_duplicates(subset="Symbol")
    cancer_expr = ref_dedup[cohort_cols].astype(float)
    normal_expr = ref_dedup[ntpm_nonrepro].astype(float)
    normal_max = normal_expr.max(axis=1)

    # --- Data-driven immune tissue identification via PTPRC (CD45) ---
    ptprc_row = ref_dedup[ref_dedup["Symbol"] == "PTPRC"]
    if len(ptprc_row):
        ptprc_vals = ptprc_row[ntpm_nonrepro].astype(float).iloc[0]
        immune_cols = [c for c in ntpm_nonrepro if ptprc_vals[c] > ptprc_vals.median()]
    else:
        immune_cols = []
    immune_sum = normal_expr[immune_cols].sum(axis=1) if immune_cols else 0
    total_sum = normal_expr.sum(axis=1)
    immune_frac = np.where(total_sum > 0.01, immune_sum / total_sum, 0.0)

    # --- Exclusion masks ---
    is_immune = (immune_frac > 0.5) & (total_sum > 10)
    is_rearranged = ref_dedup["Symbol"].apply(
        lambda s: any(s.startswith(p) for p in _IG_TR_PREFIXES)
    )
    excluded = is_immune | is_rearranged.values

    # --- Z-scores and S/N ---
    g_mean = cancer_expr.mean(axis=1)
    g_std_raw = cancer_expr.std(axis=1).replace(0, np.nan)
    z_mat = cancer_expr.sub(g_mean, axis=0).div(g_std_raw, axis=0).fillna(0)
    best_z = z_mat.max(axis=1)
    sn = cancer_expr.max(axis=1) / (normal_max + 0.01)

    # --- Primary selection: S/N pathway ---
    primary_mask = (
        (best_z > 1)
        & (sn > 3)
        & (cancer_expr.max(axis=1) > 0.1)
        & (normal_max >= 0.5)
        & ~excluded
    )
    primary_best = z_mat[primary_mask].idxmax(axis=1)

    selected_idx = []
    per_type = {}
    for code_col in cohort_cols:
        code = code_col.removesuffix("_TPM")
        code_genes = primary_best[primary_best == code_col].index
        top = best_z.loc[code_genes].nlargest(n_genes_per_type).index
        syms = list(ref_dedup.loc[top, "Symbol"].values)
        per_type[code] = syms
        selected_idx.extend(top)

    # --- Fallback for underrepresented cancer types ---
    # Use a composite score: z-score × log2(expr+1) × min(S/N, 3)/3
    # This favors genes that are type-specific (z), reasonably expressed
    # (log-expr), and have at least partial cancer vs. normal enrichment (S/N).
    fallback_types = []
    fallback_mask = (best_z > 1) & (cancer_expr.max(axis=1) > 0.1) & ~excluded
    for code_col in cohort_cols:
        code = code_col.removesuffix("_TPM")
        if len(per_type.get(code, [])) >= 2:
            continue
        fallback_types.append(code)
        avail = fallback_mask & ~ref_dedup.index.isin(selected_idx)
        z_col = z_mat.loc[avail, code_col]
        expr_col = cancer_expr.loc[avail, code_col]
        sn_col = sn.loc[avail]
        composite = z_col * np.log2(expr_col + 1) * np.clip(sn_col, 0, 3) / 3
        top = composite.nlargest(n_genes_per_type).index
        syms = list(ref_dedup.loc[top, "Symbol"].values)
        per_type[code] = per_type.get(code, []) + syms
        selected_idx.extend(top)

    # --- CTA boost ---
    selected_syms = set(
        ref_dedup.loc[list(dict.fromkeys(selected_idx)), "Symbol"].values
    )
    cta_added = []
    for cta in _CURATED_CTAS:
        if cta in selected_syms:
            continue
        cta_rows = ref_dedup[ref_dedup["Symbol"] == cta]
        if len(cta_rows) == 0:
            continue
        cta_row = cta_rows.iloc[0]
        cta_expr = cancer_expr.loc[cta_row.name]
        if cta_expr.max() < 1.0:
            continue  # median < 1 TPM in best type
        selected_idx.append(cta_row.name)
        cta_added.append(cta)

    selected_idx = list(dict.fromkeys(selected_idx))
    ref_filtered = ref_dedup.loc[selected_idx]

    metadata = {
        "per_type": per_type,
        "fallback_types": fallback_types,
        "cta_added": cta_added,
        "n_genes": len(selected_idx),
        "n_types": len([t for t, g in per_type.items() if g]),
    }

    result = (ref_filtered, metadata)
    _embedding_gene_cache[cache_key] = result
    return result


def _cancer_type_score_matrix(df_gene_expr, n_signature_genes=20):
    """Build feature matrix using cancer-type signature scores.

    Each cancer type (and the sample) is represented as a vector of scores:
    "how well does this expression profile match each cancer type's signature?"

    For reference cancer types, the score is the midrank percentile of that
    type's median expression among all cancer types, for the target type's
    signature genes.  For the sample, same scoring via
    ``_compute_cancer_type_signature_stats``.

    Returns (matrix, labels) where matrix is (34, 33) — 33 cancer types + sample.
    """
    import numpy as np

    from .tumor_purity import _cached_reference_matrices

    ref_matrices = _cached_reference_matrices(normalize="housekeeping")
    cohort_cols = ref_matrices["cohort_cols"]
    labels = [c.removesuffix("_TPM") for c in cohort_cols]

    expr_matrix = ref_matrices["expr_matrix"]
    sig = _get_cancer_type_signature_panels(n_signature_genes=n_signature_genes)

    # Score each reference cancer type against all signatures
    ref_scores = np.zeros((len(labels), len(labels)))
    for j, target_code in enumerate(labels):
        genes = sig[target_code]
        for i, source_code in enumerate(labels):
            source_col = f"{source_code}_TPM"
            pcts = []
            for gene in genes:
                if gene not in expr_matrix.index:
                    continue
                val = float(expr_matrix.loc[gene, source_col])
                ref_vals = expr_matrix.loc[gene].values
                n = len(ref_vals)
                below = np.sum(ref_vals < val)
                equal = np.sum(np.isclose(ref_vals, val, atol=1e-6))
                pcts.append((below + 0.5 * equal) / n)
            ref_scores[i, j] = float(np.mean(pcts)) if pcts else 0.5

    # Score the sample
    sample_stats = _compute_cancer_type_signature_stats(
        df_gene_expr,
        n_signature_genes=n_signature_genes,
    )
    sample_scores = np.zeros(len(labels))
    for stat in sample_stats:
        j = labels.index(stat["code"])
        sample_scores[j] = stat["score"]

    matrix = np.vstack([ref_scores, sample_scores[None, :]])
    labels.append("SAMPLE")
    return matrix, labels


def _reference_cancer_expression_df(cancer_code):
    """Return a TCGA-median expression frame for one cancer type."""
    import pandas as pd

    ref = pan_cancer_expression(technical_rna_normalize=True).drop_duplicates(
        subset="Ensembl_Gene_ID"
    )
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{cancer_code}_TPM"].astype(float),
        }
    )


def _hierarchy_feature_labels():
    from .tumor_purity import _CANCER_FAMILY_PANELS, CANCER_TO_TISSUE

    ref = pan_cancer_expression(technical_rna_normalize=True)
    codes = [c.removesuffix("_TPM") for c in ref.columns if c.endswith("_TPM")]
    families = list(_CANCER_FAMILY_PANELS)
    site_labels = sorted(
        set(CANCER_TO_TISSUE.values())
        | {
            "appendix",
            "bone_marrow",
            "lymph_node",
            "smooth_muscle",
            "spleen",
            "adipose_tissue",
        }
    )
    feature_labels = (
        [f"support::{code}" for code in codes]
        + [f"family::{family}" for family in families]
        + [f"site::{site}" for site in site_labels]
        + ["purity::best_estimate"]
    )
    return codes, families, site_labels, feature_labels


def _hierarchy_feature_vector(
    df_gene_expr, candidate_codes, family_labels, site_labels
):
    """Build one hierarchy-aware embedding vector for a sample/profile."""
    from .tumor_purity import (
        rank_cancer_type_candidates,
        _score_cancer_family_panels,
        _score_host_tissues,
    )

    candidate_trace = rank_cancer_type_candidates(
        df_gene_expr,
        candidate_codes=candidate_codes,
        top_k=len(candidate_codes),
    )
    trace_by_code = {row["code"]: row for row in candidate_trace}
    sample_raw_by_symbol, _ = _sample_expression_by_symbol(df_gene_expr)
    # Family panels are scored by Ensembl ID (alias-drift immune); pass the
    # ID-keyed clean-TPM sample, not the symbol-keyed one.
    from .common import build_sample_tpm_by_gene_id

    sample_raw_by_id = build_sample_tpm_by_gene_id(df_gene_expr)
    family_scores = _score_cancer_family_panels(sample_raw_by_id)
    max_family_score = max(family_scores.values(), default=0.0)
    if max_family_score > 0:
        family_features = [
            float(family_scores.get(family, 0.0) / max_family_score)
            for family in family_labels
        ]
    else:
        family_features = [0.0 for _ in family_labels]

    site_scores = {
        tissue: score
        for tissue, score, _ in _score_host_tissues(
            sample_raw_by_symbol,
            tissues=site_labels,
            top_n=None,
        )
    }
    max_site_score = max(site_scores.values(), default=0.0)
    if max_site_score > 0:
        site_features = [
            float(site_scores.get(site, 0.0) / max_site_score) for site in site_labels
        ]
    else:
        site_features = [0.0 for _ in site_labels]

    support_features = [
        float(trace_by_code.get(code, {}).get("support_fraction_of_top", 0.0))
        for code in candidate_codes
    ]
    best_purity = (
        float(candidate_trace[0]["purity_estimate"]) if candidate_trace else 0.0
    )
    return np.array(
        support_features + family_features + site_features + [best_purity], dtype=float
    )


def _reference_family_feature_matrix(candidate_codes, family_labels):
    """Build normalized family-panel features for TCGA reference centroids.

    Scores each cohort with the SAME ENSG-keyed scorer the sample side uses
    (:func:`_score_cancer_family_panels`), so the ``family::*`` block is computed
    one single way in both halves of the embedding's feature space. Previously
    the reference half re-implemented the upper-half-median scoring *by HGNC
    symbol* against the reference's ``Symbol`` column, while the sample half
    scored *by Ensembl ID* — a latent vocabulary split that would silently
    diverge the moment the panel's curation symbols drifted from the reference
    symbols. Building an ENSG-keyed cohort TPM dict and reusing the one scorer
    removes the split by construction (the #65 class, structurally).
    """
    import pandas as pd
    from .tumor_purity import _score_cancer_family_panels
    from .plot_data_helpers import _strip_ensembl_version

    cache_key = (tuple(candidate_codes), tuple(family_labels))
    cached = _hierarchy_family_cache.get(cache_key)
    if cached is not None:
        return cached

    # Raw clean per-cohort TPM (not housekeeping-normalized): the shared scorer
    # does its own housekeeping division internally, exactly as on the sample.
    ref = pan_cancer_expression(technical_rna_normalize=True)
    gene_ids = ref["Ensembl_Gene_ID"].astype(str).map(_strip_ensembl_version)
    id_valid = gene_ids.notna() & gene_ids.str.strip().str.len().gt(0)

    # One max-per-versionless-gene-id collapse across every cohort column at
    # once (mirrors build_sample_tpm_by_gene_id's max-per-id), instead of a
    # groupby per candidate code. Each column then feeds the shared scorer as a
    # plain {ENSG: TPM} dict.
    cohort_cols = [f"{code}_TPM" for code in candidate_codes]
    numeric = ref.loc[id_valid, cohort_cols].apply(pd.to_numeric, errors="coerce")
    by_id = numeric.groupby(gene_ids[id_valid]).max()

    rows = []
    for code in candidate_codes:
        cohort_tpm_by_id = by_id[f"{code}_TPM"].dropna().to_dict()
        family_scores = _score_cancer_family_panels(cohort_tpm_by_id)
        family_values = [float(family_scores.get(family, 0.0)) for family in family_labels]
        max_family = max(family_values, default=0.0)
        if max_family > 0:
            family_values = [float(value / max_family) for value in family_values]
        rows.append(family_values)
    result = np.asarray(rows, dtype=float)
    _hierarchy_family_cache[cache_key] = result
    return result


def _reference_site_feature_matrix(candidate_codes, site_labels):
    """Build normalized host/background context features for TCGA centroids."""
    from .tumor_purity import _score_host_tissues

    cache_key = (tuple(candidate_codes), tuple(site_labels))
    cached = _hierarchy_site_cache.get(cache_key)
    if cached is not None:
        return cached

    ref = (
        pan_cancer_expression(technical_rna_normalize=True)
        .drop_duplicates(subset="Symbol")
        .set_index("Symbol")
    )
    rows = []
    for code in candidate_codes:
        sample_raw_by_symbol = ref[f"{code}_TPM"].astype(float).to_dict()
        site_scores = {
            tissue: score
            for tissue, score, _ in _score_host_tissues(
                sample_raw_by_symbol,
                tissues=site_labels,
                top_n=None,
            )
        }
        max_site_score = max(site_scores.values(), default=0.0)
        if max_site_score > 0:
            site_values = [
                float(site_scores.get(site, 0.0) / max_site_score)
                for site in site_labels
            ]
        else:
            site_values = [0.0 for _ in site_labels]
        rows.append(site_values)
    cached = np.asarray(rows, dtype=float)
    _hierarchy_site_cache[cache_key] = cached
    return cached


def _hierarchy_embedding_metadata():
    """Describe the hierarchy-aware embedding feature space."""
    candidate_codes, family_labels, site_labels, feature_labels = (
        _hierarchy_feature_labels()
    )
    return {
        "method": "hierarchy",
        "feature_kind": "hierarchical_scores",
        "n_features": len(feature_labels),
        "n_types": len(candidate_codes),
        "n_genes": 0,
        "per_type": {},
        "families": family_labels,
        "sites": site_labels,
        "codes": candidate_codes,
        "feature_labels": feature_labels,
    }


def _cancer_type_hierarchy_matrix(df_gene_expr):
    """Build a family-aware support-space matrix for TCGA centroids + sample."""
    import numpy as np
    from .tumor_purity import TCGA_MEDIAN_PURITY

    candidate_codes, family_labels, site_labels, _feature_labels = (
        _hierarchy_feature_labels()
    )
    cache_key = tuple(candidate_codes)
    cached = _hierarchy_feature_cache.get(cache_key)
    if cached is None:
        ref_scores, labels = _cancer_type_score_matrix(
            _reference_cancer_expression_df(candidate_codes[0])
        )
        ref_labels = labels[:-1]
        ref_matrix = ref_scores[:-1]
        ref_families = _reference_family_feature_matrix(ref_labels, family_labels)
        ref_sites = _reference_site_feature_matrix(ref_labels, site_labels)
        ref_purity = np.array(
            [[float(TCGA_MEDIAN_PURITY.get(code, 0.5))] for code in ref_labels],
            dtype=float,
        )
        cached = (
            np.hstack([ref_matrix, ref_families, ref_sites, ref_purity]),
            ref_labels,
        )
        _hierarchy_feature_cache[cache_key] = cached

    ref_matrix, labels = cached
    sample_vector = _hierarchy_feature_vector(
        df_gene_expr, candidate_codes, family_labels, site_labels
    )
    matrix = np.vstack([ref_matrix, sample_vector[None, :]])
    out_labels = list(labels) + ["SAMPLE"]
    return matrix, out_labels


def _subtype_expression_values_for_ref(
    ref_norm,
    *,
    normalize_housekeeping=False,
    exclude_codes=None,
):
    """Return subtype-reference expression columns aligned to ``ref_norm``.

    The main pan-cancer table has one ``<TCGA>_TPM`` column per parent
    cohort. Additional curated cancer types live in the registry and, for
    some subtypes, in ``subtype-deconvolved-expression`` as long-form
    tumor-TPM medians. This helper widens only those subtype references
    that have measured/deconvolved expression rather than fabricating
    centroids from marker panels.
    """
    import pandas as pd

    if ref_norm is None or len(ref_norm) == 0 or "Symbol" not in ref_norm.columns:
        return [], np.zeros((len(ref_norm), 0), dtype=float)

    try:
        from pirlygenes.gene_sets_cancer import (
            housekeeping_gene_names,
        )
        from trufflepig.reference import (
            subtype_deconvolved_expression,
        )

        subtype_df = subtype_deconvolved_expression(technical_rna_normalize=True)
    except Exception:
        return [], np.zeros((len(ref_norm), 0), dtype=float)

    if subtype_df is None or subtype_df.empty:
        return [], np.zeros((len(ref_norm), 0), dtype=float)

    symbols = [str(symbol) for symbol in ref_norm["Symbol"].astype(str)]
    symbol_set = set(symbols)
    sub = subtype_df[subtype_df["symbol"].astype(str).isin(symbol_set)].copy()
    if sub.empty:
        return [], np.zeros((len(ref_norm), 0), dtype=float)

    subtype_code = (
        sub.get("subtype", pd.Series("", index=sub.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    cancer_code = sub["cancer_code"].fillna("").astype(str).str.strip()
    sub["_embed_code"] = subtype_code.where(
        subtype_code.ne("") & subtype_code.ne("nan"), cancer_code
    )
    sub = sub[sub["_embed_code"].ne("") & sub["_embed_code"].ne("nan")]
    if sub.empty:
        return [], np.zeros((len(ref_norm), 0), dtype=float)

    pivot = sub.pivot_table(
        index="symbol",
        columns="_embed_code",
        values="tumor_tpm_median",
        aggfunc="median",
    )
    exclude_codes = set(exclude_codes or [])
    labels = sorted(
        str(label) for label in pivot.columns if str(label) not in exclude_codes
    )
    if not labels:
        return [], np.zeros((len(ref_norm), 0), dtype=float)

    if normalize_housekeeping:
        hk_symbols = set(housekeeping_gene_names())
        hk = subtype_df[subtype_df["symbol"].astype(str).isin(hk_symbols)].copy()
        if not hk.empty:
            hk_subtype = (
                hk.get("subtype", pd.Series("", index=hk.index))
                .fillna("")
                .astype(str)
                .str.strip()
            )
            hk_cancer = hk["cancer_code"].fillna("").astype(str).str.strip()
            hk["_embed_code"] = hk_subtype.where(
                hk_subtype.ne("") & hk_subtype.ne("nan"), hk_cancer
            )
            hk = hk[hk["_embed_code"].isin(labels)]
            hk_medians = hk.groupby("_embed_code")["tumor_tpm_median"].median()
            for label in labels:
                denom = float(hk_medians.get(label, 1.0) or 1.0)
                if denom > 0:
                    pivot[label] = pivot[label].astype(float) / denom

    values = (
        pivot.reindex(index=symbols, columns=labels)
        .fillna(0.0)
        .astype(float)
        .to_numpy()
    )
    return labels, values


def _sanitize_embedding_matrix(matrix):
    """Remove all-NaN/non-finite feature columns before distance geometry."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 1), dtype=float)
    finite_cols = np.isfinite(matrix).all(axis=0)
    if finite_cols.any():
        matrix = matrix[:, finite_cols]
    elif matrix.shape[1] > 0:
        matrix = np.zeros((matrix.shape[0], 1), dtype=float)
    if matrix.shape[1] == 0:
        matrix = np.zeros((matrix.shape[0], 1), dtype=float)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def _cancer_type_feature_matrix(
    df_gene_expr,
    n_genes=10,
    method="zscore",
    include_normals=False,
    include_subtypes=False,
):
    """Build feature matrix for PCA/MDS of cancer types + sample.

    Gene selection is unified: a single biologically-informed gene set is
    used regardless of normalization method.  The *method* parameter only
    controls how expression values are transformed before embedding.

    Parameters
    ----------
    method : str
        ``"zscore"`` — z-score of log2(1+raw) across cancer types (default).
        ``"hk"`` — log2(HK-normalized + 1).
        ``"hk_zscore"`` — z-score of log2(HK-normalized + 1).
        ``"rank"`` — percentile rank within each gene across cancer types.
        ``"score"`` — cancer-type signature scores (33-d vector).
        ``"hierarchy"`` — family-aware support-score space with purity anchor.
        ``"panref"`` — shared cancer/subtype/normal reference genes for
        normal-inclusive maps.
    include_normals : bool
        When true, append normal-tissue nTPM centroids to the embedded
        reference rows. Supported for expression-space methods; ignored for
        score/hierarchy spaces where the normal vectors are not defined.
    include_subtypes : bool
        When true, append available subtype-deconvolved tumor-TPM references
        (for example SARC_LMS, OS, EWS) in addition to TCGA parent cohorts.
        Registry rows without expression medians are still not embedded.
    """
    import warnings

    import numpy as np
    from scipy.stats import rankdata

    if method == "robust":
        warnings.warn(
            "method='robust' is deprecated; gene selection is now unified. "
            "Using 'zscore'.",
            DeprecationWarning,
            stacklevel=2,
        )
        method = "zscore"

    if method == "score":
        return _cancer_type_score_matrix(df_gene_expr)
    if method == "hierarchy":
        return _cancer_type_hierarchy_matrix(df_gene_expr)

    gene_id_col, _ = _guess_gene_cols(df_gene_expr)
    df = df_gene_expr.copy()
    df[gene_id_col] = df[gene_id_col].astype(str).map(_strip_ensembl_version)

    tpm_col = (
        "TPM"
        if "TPM" in df.columns
        else next((c for c in df.columns if c.lower() == "tpm"), None)
    )

    if method in ("hk", "hk_zscore"):
        hk_mask = df[gene_id_col].isin(housekeeping_gene_ids())
        hk_median = df.loc[hk_mask, tpm_col].astype(float).median()
        if not (hk_median > 0):  # catches NaN and <= 0
            hk_median = 1.0
        sample_by_id = dict(
            zip(
                df[gene_id_col].astype(str),
                df[tpm_col].astype(float) / hk_median,
            )
        )
        ref_full = pan_cancer_expression(
            normalize="housekeeping",
            technical_rna_normalize=True,
        )
    else:
        sample_by_id = dict(
            zip(
                df[gene_id_col].astype(str),
                df[tpm_col].astype(float),
            )
        )
        ref_full = pan_cancer_expression(technical_rna_normalize=True)

    cohort_cols = [c for c in ref_full.columns if c.endswith("_TPM")]
    labels = [c.removesuffix("_TPM") for c in cohort_cols]
    normal_cols = []
    normal_labels = []
    if include_normals:
        normal_cols = [c for c in ref_full.columns if c.endswith("_nTPM")]
        normal_labels = [
            "normal:" + c.removesuffix("_nTPM").replace("_", " ") for c in normal_cols
        ]

    # Gene selection
    if method == "tme":
        ref_filtered, _meta = _select_tme_low_genes(n_genes_per_type=n_genes)
    elif method == "bottleneck":
        ref_filtered, _meta = _select_embedding_genes_bottleneck(
            n_genes_per_type=n_genes
        )
    elif method == "panref":
        ref_filtered, _meta = _select_pan_reference_genes(n_genes_per_type=n_genes)
    else:
        ref_filtered, _meta = _select_embedding_genes(n_genes_per_type=n_genes)

    # Map gene set to potentially HK-normalized reference
    gene_ids = list(ref_filtered["Ensembl_Gene_ID"])
    ref_norm = ref_full[ref_full["Ensembl_Gene_ID"].isin(gene_ids)].drop_duplicates(
        subset="Ensembl_Gene_ID"
    )
    # Preserve the gene order from _select_embedding_genes
    ref_norm = (
        ref_norm.set_index("Ensembl_Gene_ID")
        .loc[[gid for gid in gene_ids if gid in ref_norm["Ensembl_Gene_ID"].values]]
        .reset_index()
    )

    sample_vals = np.array(
        [
            sample_by_id.get(row["Ensembl_Gene_ID"], 0.0)
            for _, row in ref_norm.iterrows()
        ]
    )
    ref_vals = ref_norm[cohort_cols].astype(float).values  # (genes, cancers)
    subtype_labels, subtype_vals = (
        _subtype_expression_values_for_ref(
            ref_norm,
            normalize_housekeeping=method in ("hk", "hk_zscore"),
            exclude_codes=set(labels),
        )
        if include_subtypes
        else ([], np.zeros((len(ref_norm), 0), dtype=float))
    )
    normal_vals = (
        ref_norm[normal_cols].astype(float).values
        if normal_cols
        else np.zeros((len(ref_norm), 0), dtype=float)
    )

    if method in ("zscore", "hk_zscore", "tme", "bottleneck", "panref"):
        log_ref = np.log2(ref_vals + 1)
        log_sample = np.log2(sample_vals + 1)
        log_subtypes = np.log2(subtype_vals + 1) if subtype_labels else None
        log_normals = np.log2(normal_vals + 1) if normal_cols else None
        if method == "panref":
            scale_parts = [log_ref]
            if log_subtypes is not None and log_subtypes.shape[1]:
                scale_parts.append(log_subtypes)
            if log_normals is not None and log_normals.shape[1]:
                scale_parts.append(log_normals)
            log_scale = np.hstack(scale_parts)
            g_center = np.nanmedian(log_scale, axis=1)
            q1 = np.nanpercentile(log_scale, 25, axis=1)
            q3 = np.nanpercentile(log_scale, 75, axis=1)
            g_std = (q3 - q1) / 1.349
            fallback_std = np.nanstd(log_scale, axis=1)
            g_std = np.where(g_std >= 0.08, g_std, fallback_std)
            g_std = np.where(g_std >= 0.08, g_std, 0.08)
            var_mask = (np.nanmax(log_scale, axis=1) - np.nanmin(log_scale, axis=1)) >= 0.2
        else:
            g_std = log_ref.std(axis=1)
            g_center = log_ref.mean(axis=1)
            var_mask = g_std >= 0.1
        log_ref = log_ref[var_mask]
        log_sample = log_sample[var_mask]
        if log_subtypes is not None:
            log_subtypes = log_subtypes[var_mask]
        if log_normals is not None:
            log_normals = log_normals[var_mask]
        g_std = g_std[var_mask]
        g_center = g_center[var_mask]
        clip_limit = 4 if method == "panref" else 3
        z_ref = np.clip(
            (log_ref - g_center[:, None]) / g_std[:, None],
            -clip_limit,
            clip_limit,
        )
        z_sample = np.clip(
            (log_sample - g_center) / g_std,
            -clip_limit,
            clip_limit,
        )
        parts = [z_ref.T]
        if log_subtypes is not None and log_subtypes.shape[1]:
            z_subtypes = np.clip(
                (log_subtypes - g_center[:, None]) / g_std[:, None],
                -clip_limit,
                clip_limit,
            )
            parts.append(z_subtypes.T)
        if log_normals is not None and log_normals.shape[1]:
            z_normals = np.clip(
                (log_normals - g_center[:, None]) / g_std[:, None],
                -clip_limit,
                clip_limit,
            )
            parts.append(z_normals.T)
        parts.append(z_sample[None, :])
        matrix = np.vstack(parts)
    elif method == "hk":
        parts = [ref_vals.T]
        if subtype_labels:
            parts.append(subtype_vals.T)
        if normal_cols:
            parts.append(normal_vals.T)
        parts.append(sample_vals[None, :])
        combined = np.vstack(parts)
        matrix = np.log2(combined + 1)
    elif method == "rank":
        parts = [ref_vals.T]
        if subtype_labels:
            parts.append(subtype_vals.T)
        if normal_cols:
            parts.append(normal_vals.T)
        parts.append(sample_vals[None, :])
        combined = np.vstack(parts)
        ranked = np.apply_along_axis(
            lambda col: rankdata(col, method="average") / len(col),
            axis=0,
            arr=combined,
        )
        matrix = ranked
    else:
        raise ValueError(f"Unknown method: {method}")

    labels.extend(subtype_labels)
    labels.extend(normal_labels)
    labels.append("SAMPLE")
    return _sanitize_embedding_matrix(matrix), labels


def get_embedding_feature_metadata(method="hierarchy", n_genes=10):
    """Return metadata describing the active embedding feature space."""
    if method == "hierarchy":
        return _hierarchy_embedding_metadata()
    if method == "tme":
        return _select_tme_low_genes(n_genes_per_type=n_genes)[1]
    if method == "bottleneck":
        return _select_embedding_genes_bottleneck(n_genes_per_type=n_genes)[1]
    if method == "panref":
        return _select_pan_reference_genes(n_genes_per_type=n_genes)[1]
    return _select_embedding_genes(n_genes_per_type=n_genes)[1]


def _plot_embedding_with_labels(
    coords,
    labels,
    *,
    title,
    xlabel,
    ylabel,
    method=None,
    label_nearest_cancers=None,
    label_nearest_normals=None,
    label_all=True,
    nearest_neighbors=None,
    nearest_basis="plotted 2D distance",
    sample_centered_distance=False,
    footnote=None,
    save_to_filename=None,
    save_dpi=300,
    figsize=(12, 10),
):
    from matplotlib.lines import Line2D

    def _label_kind(label):
        return _embedding_label_kind(label)

    def _display_label(label):
        return str(label).replace("normal:", "")

    fig, ax = plt.subplots(figsize=figsize)
    texts = []
    family_palette = {}
    label_to_family = {}
    if method == "hierarchy":
        from .tumor_purity import _CANCER_FAMILY_BY_CODE

        family_order = []
        for label in labels:
            if label == "SAMPLE" or _label_kind(label) == "normal":
                continue
            family = _CANCER_FAMILY_BY_CODE.get(label, label)
            label_to_family[label] = family
            if family not in family_order:
                family_order.append(family)
        palette = sns.color_palette("tab20", max(len(family_order), 1))
        family_palette = {
            family: palette[idx] for idx, family in enumerate(family_order)
        }

    if nearest_neighbors is None:
        nearest_neighbors = []
    else:
        nearest_neighbors = list(nearest_neighbors)
    if not nearest_neighbors and "SAMPLE" in labels:
        sample_idx = labels.index("SAMPLE")
        sample_coords = coords[sample_idx]
        for i, label in enumerate(labels):
            if label == "SAMPLE":
                continue
            dist = float(np.linalg.norm(coords[i] - sample_coords))
            nearest_neighbors.append((dist, label, _label_kind(label)))
        nearest_neighbors.sort(key=lambda item: (item[0], item[1]))

    nearest_cancer_ordered = [
        label for _, label, kind in nearest_neighbors if kind == "cancer"
    ]
    nearest_subtype_ordered = [
        label for _, label, kind in nearest_neighbors if kind == "subtype"
    ]
    nearest_normal_ordered = [
        label for _, label, kind in nearest_neighbors if kind == "normal"
    ]
    nearest_cancer_labels = set(nearest_cancer_ordered)
    nearest_subtype_labels = set(nearest_subtype_ordered)
    nearest_normal_labels = set(nearest_normal_ordered)
    if label_nearest_cancers is not None:
        n_nearest_cancers = max(0, int(label_nearest_cancers))
        nearest_cancer_labels = set(
            nearest_cancer_ordered[:n_nearest_cancers]
        )
        n_nearest_subtypes = n_nearest_cancers
        nearest_subtype_labels = set(nearest_subtype_ordered[:n_nearest_subtypes])
    if label_nearest_normals is not None:
        nearest_normal_labels = set(
            nearest_normal_ordered[: int(label_nearest_normals)]
        )

    def _should_label(label):
        if label == "SAMPLE":
            return True
        if label_all:
            return True
        kind = _label_kind(label)
        if kind == "cancer":
            return label in nearest_cancer_labels
        if kind == "subtype":
            return label in nearest_subtype_labels
        if kind == "normal":
            return label in nearest_normal_labels
        return False

    for i, label in enumerate(labels):
        if label == "SAMPLE":
            ax.scatter(
                coords[i, 0],
                coords[i, 1],
                s=220,
                color="red",
                edgecolors="black",
                linewidths=1.5,
                zorder=5,
                marker="*",
            )
            texts.append(
                ax.text(
                    coords[i, 0],
                    coords[i, 1],
                    label,
                    fontsize=10,
                    fontweight="bold",
                    color="red",
                    va="center",
                )
            )
        else:
            kind = _label_kind(label)
            if kind == "normal":
                point_color = "#8da08d"
                edge_color = "#f8f8f8"
                marker = "s"
                size = 38
                alpha = 0.52
            elif kind == "subtype":
                point_color = "#6da3cf"
                edge_color = "white"
                marker = "^"
                size = 54
                alpha = 0.62
            else:
                point_color = family_palette.get(
                    label_to_family.get(label), "steelblue"
                )
                edge_color = "white"
                marker = "o"
                size = 60
                alpha = 0.7
            ax.scatter(
                coords[i, 0],
                coords[i, 1],
                s=size,
                alpha=alpha,
                color=point_color,
                edgecolors=edge_color,
                linewidths=0.5,
                zorder=2,
                marker=marker,
            )
            if _should_label(label):
                texts.append(
                    ax.text(
                        coords[i, 0],
                        coords[i, 1],
                        _display_label(label),
                        fontsize=7,
                        alpha=0.85,
                        va="center",
                    )
                )

    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="#999999", alpha=0.35),
        expand=(1.05, 1.2),
    )
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.2)
    if sample_centered_distance:
        ax.set_aspect("equal", adjustable="datalim")
        if "SAMPLE" in labels:
            sx, sy = coords[labels.index("SAMPLE")]
            for radius in (0.9, 1.0, 1.1):
                ax.add_patch(
                    plt.Circle(
                        (sx, sy),
                        radius,
                        fill=False,
                        linestyle="--",
                        linewidth=0.7,
                        color="#d9d9d9",
                        zorder=1,
                    )
                )
    if footnote:
        ax.text(
            0.01,
            0.01,
            footnote,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            color="#555555",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.78,
                edgecolor="#dddddd",
            ),
        )

    if nearest_neighbors:
        if any(kind == "normal" for _, _, kind in nearest_neighbors):
            n_cancers = (
                5 if label_nearest_cancers is None else int(label_nearest_cancers)
            )
            n_normals = (
                5 if label_nearest_normals is None else int(label_nearest_normals)
            )
            cancer_text = "\n".join(
                f"{_display_label(label)} ({dist:.2f})"
                for dist, label, kind in nearest_neighbors
                if kind == "cancer"
            )
            subtype_text = "\n".join(
                f"{_display_label(label)} ({dist:.2f})"
                for dist, label, kind in nearest_neighbors
                if kind == "subtype"
            )
            normal_text = "\n".join(
                f"{_display_label(label)} ({dist:.2f})"
                for dist, label, kind in nearest_neighbors
                if kind == "normal"
            )
            nearest_text = (
                f"Nearest by {nearest_basis}\n\n"
                "TCGA parent cancers\n"
                + "\n".join(cancer_text.splitlines()[:n_cancers])
            )
            if subtype_text:
                nearest_text += "\n\nSubtype references\n" + "\n".join(
                    subtype_text.splitlines()[:n_cancers]
                )
            nearest_text += "\n\nNearest normal tissues\n" + "\n".join(
                normal_text.splitlines()[:n_normals]
            )
            nearest_title = ""
        else:
            n_cancers = (
                5 if label_nearest_cancers is None else int(label_nearest_cancers)
            )
            nearest_text = "\n".join(
                f"{label} ({dist:.2f})"
                for dist, label, _kind in nearest_neighbors[:n_cancers]
            )
            nearest_title = f"Nearest by {nearest_basis}\n"
        ax.text(
            0.98,
            0.02,
            nearest_title + nearest_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.9,
                edgecolor="#cccccc",
            ),
        )

    if family_palette:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=7,
                label=family,
            )
            for family, color in list(family_palette.items())[:10]
        ]
        ax.legend(
            handles=handles,
            title="Family",
            loc="upper left",
            fontsize=8,
            title_fontsize=9,
            framealpha=0.9,
        )
    elif any(_label_kind(label) == "normal" for label in labels):
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="steelblue",
                markeredgecolor="white",
                markersize=7,
                label="TCGA cancer",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor="#6da3cf",
                markeredgecolor="white",
                markersize=7,
                label="subtype reference",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="#8da08d",
                markeredgecolor="white",
                markersize=6,
                label="normal tissue",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="red",
                markeredgecolor="black",
                markersize=10,
                linestyle="none",
                label="sample",
            ),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


def _embedding_label_kind(label):
    if label == "SAMPLE":
        return "sample"
    if str(label).startswith("normal:"):
        return "normal"
    if str(label) in _tcga_parent_codes():
        return "cancer"
    return "subtype"


def _nearest_neighbors_from_distances(labels, distances, sample_idx):
    neighbors = []
    for i, label in enumerate(labels):
        if i == sample_idx or label == "SAMPLE":
            continue
        neighbors.append(
            (float(distances[sample_idx, i]), label, _embedding_label_kind(label))
        )
    neighbors.sort(key=lambda item: (item[0], item[1]))
    return neighbors


def _sample_neighborhood_indices(
    labels,
    distances,
    sample_idx,
    *,
    nearest_cancers=None,
    nearest_normals=None,
):
    """Return row indices for a sample-centered embedding neighborhood."""
    if nearest_cancers is None and nearest_normals is None:
        return list(range(len(labels)))
    neighbors = _nearest_neighbors_from_distances(labels, distances, sample_idx)
    keep = {sample_idx}
    n_cancers = 0
    n_normals = 0
    if nearest_cancers is not None:
        for _dist, label, kind in neighbors:
            if kind not in {"cancer", "subtype"}:
                continue
            keep.add(labels.index(label))
            n_cancers += 1
            if n_cancers >= int(nearest_cancers):
                break
    if nearest_normals is not None:
        for _dist, label, kind in neighbors:
            if kind != "normal":
                continue
            keep.add(labels.index(label))
            n_normals += 1
            if n_normals >= int(nearest_normals):
                break
    return [i for i in range(len(labels)) if i in keep]


def _euclidean_distances(X):
    """Pairwise Euclidean distances without depending on scikit-learn."""
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))


def _pca_coordinates(X, n_components=2):
    """Return PCA coordinates and explained-variance ratios using NumPy SVD."""
    X = np.asarray(X, dtype=float)
    n_rows = X.shape[0] if X.ndim else 0
    if X.ndim != 2 or n_rows == 0:
        return np.zeros((n_rows, n_components), dtype=float), np.zeros(n_components)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = X - X.mean(axis=0, keepdims=True)
    max_components = min(n_components, X.shape[0], X.shape[1])
    if max_components < 1 or np.allclose(X, 0.0):
        return np.zeros((X.shape[0], n_components), dtype=float), np.zeros(n_components)
    try:
        u, s, _vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((X.shape[0], n_components), dtype=float), np.zeros(n_components)
    coords = u[:, :max_components] * s[:max_components]
    if max_components < n_components:
        coords = np.column_stack(
            [coords, np.zeros((X.shape[0], n_components - max_components))]
        )
    variance = s * s
    total = float(np.sum(variance))
    ratios = np.zeros(n_components, dtype=float)
    if total > 0:
        ratios[: min(n_components, len(variance))] = variance[:n_components] / total
    return coords[:, :n_components], ratios


def _classical_mds_coordinates(distances, n_components=2):
    """Classical metric MDS coordinates from a precomputed distance matrix."""
    distances = np.asarray(distances, dtype=float)
    n = distances.shape[0] if distances.ndim == 2 else 0
    if n == 0:
        return np.zeros((0, n_components), dtype=float)
    distances = np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)
    squared = distances * distances
    center = np.eye(n) - np.ones((n, n), dtype=float) / n
    gram = -0.5 * center @ squared @ center
    try:
        values, vectors = np.linalg.eigh(gram)
    except np.linalg.LinAlgError:
        return np.zeros((n, n_components), dtype=float)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    positive = np.maximum(values[:n_components], 0.0)
    coords = vectors[:, : len(positive)] * np.sqrt(positive)
    if coords.shape[1] < n_components:
        coords = np.column_stack(
            [coords, np.zeros((n, n_components - coords.shape[1]))]
        )
    return coords[:, :n_components]


def plot_cancer_type_genes(
    df_gene_expr,
    n_per_tail=5,
    save_to_filename=None,
    save_dpi=300,
    figsize=(14, 22),
):
    """Show all signature genes for the closest and most distant cancer types.

    Cancer types are ranked by signature similarity to the sample. The plot
    then shows the top ``n_per_tail`` closest and bottom ``n_per_tail`` most
    distant cancer types, while still plotting all signature genes for each
    selected row. The gray bar marks the mean sample TPM for that row.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Patient expression data.
    n_per_tail : int
        Number of cancer types to show in each tail (closest / most distant).
    save_to_filename : str or None
        Output path.
    """
    import numpy as np

    stats = _compute_cancer_type_signature_stats(df_gene_expr, n_signature_genes=20)
    if not stats:
        return None, None

    top_stats = stats[:n_per_tail]
    bottom_stats = sorted(
        stats[-n_per_tail:], key=lambda row: (row["score"], row["code"])
    )
    selected_stats = top_stats + bottom_stats

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(42)

    y_pos = 0
    y_ticks = []
    y_labels = []
    texts = []

    for idx, row in enumerate(selected_stats):
        code = row["code"]
        gene_details = sorted(
            row["gene_details"],
            key=lambda item: (-item["sample_raw"], item["gene"]),
        )
        label = f"{row['rank']:>2}. {code} — score={row['score']:.2f}"
        y_ticks.append(y_pos)
        y_labels.append(label)

        x_values = [detail["sample_raw"] + 0.01 for detail in gene_details]
        ax.scatter(
            x_values,
            y_pos + rng.uniform(-0.14, 0.14, len(x_values)),
            s=18,
            alpha=0.45,
            color="#2166ac",
            edgecolors="none",
            zorder=2,
        )

        mean_x = row["mean_sample_raw"] + 0.01
        ax.plot(
            [mean_x, mean_x],
            [y_pos - 0.28, y_pos + 0.28],
            color="#999999",
            linewidth=1.6,
            alpha=0.7,
            zorder=1,
        )

        for detail in gene_details[:5]:
            x = detail["sample_raw"] + 0.01
            jitter = rng.uniform(-0.12, 0.12)
            ax.scatter(
                x,
                y_pos + jitter,
                s=30,
                alpha=0.85,
                color="#2166ac",
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
            texts.append(
                ax.text(
                    x,
                    y_pos + jitter,
                    detail["gene"],
                    fontsize=8,
                    va="center",
                    ha="left",
                    alpha=0.85,
                    color="#2166ac",
                )
            )

        if idx == len(top_stats) - 1 and bottom_stats:
            ax.axhline(
                y=y_pos + 0.5, color="#cccccc", linewidth=0.8, alpha=0.8, zorder=0
            )

        y_pos += 1

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Sample TPM", fontsize=11)
    ax.set_ylabel("")
    ax.set_title(
        "Cancer-type signature genes: sample expression\n"
        "(top 5 closest and bottom 5 most distant cancer types; gray bar = mean sample TPM)",
        fontsize=11,
    )
    ax.invert_yaxis()

    # Reference lines
    for tpm_thresh in (10, 100):
        ax.axvline(
            x=tpm_thresh,
            color="#cccccc",
            linestyle="--",
            linewidth=0.7,
            alpha=0.5,
            zorder=1,
        )

    # Fix x-axis limits before adjustText to prevent blowout
    ax.set_xlim(left=0.005)
    ax.autoscale_view(scalex=True, scaley=False)
    try:
        from .common import build_sample_tpm_by_symbol
        from .plot_reference_lines import add_p90_reference_line

        add_p90_reference_line(
            ax,
            build_sample_tpm_by_symbol(df_gene_expr),
            orientation="vertical",
        )
    except Exception:
        pass

    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="#999999", alpha=0.25),
        expand=(1.03, 1.2),
        expand_axes=False,
        ensure_inside_axes=True,
    )

    fig.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


def plot_cancer_type_disjoint_genes(
    df_gene_expr,
    n_genes=20,
    save_to_filename=None,
    save_dpi=300,
    figsize=(14, 12),
):
    """Bar chart of cancer-type signature similarity scores.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Patient expression data.
    n_genes : int
        Max disjoint genes per cancer type to consider.
    save_to_filename : str or None
        Output path.
    """
    stats = _compute_cancer_type_signature_stats(
        df_gene_expr,
        n_signature_genes=n_genes,
        min_fold=2.0,
    )

    fig, ax = plt.subplots(figsize=figsize)

    scores = [row["score"] for row in stats]
    labels = [
        f"{row['code']} ({CANCER_TYPE_NAMES.get(row['code'], row['code'])})"
        for row in stats
    ]
    y = np.arange(len(stats))

    # Color bars by score intensity
    colors = [plt.cm.Blues(0.25 + 0.65 * s) for s in scores]
    ax.barh(y, scores, color=colors, edgecolor="none", height=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Signature similarity score", fontsize=10)
    # Definition belongs on the axis label, not in a subtitle.
    ax.set_title("Cancer type similarity score", fontsize=11)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 1)

    # Annotate bars with top contributing genes
    for i, row in enumerate(stats):
        top3 = sorted(row["gene_details"], key=lambda detail: -detail["percentile"])[:3]
        top3_str = ", ".join(detail["gene"] for detail in top3)
        if top3_str:
            ax.text(
                min(row["score"] + 0.01, 0.99),
                i,
                top3_str,
                fontsize=5.5,
                va="center",
                alpha=0.7,
            )

    fig.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


# -------------------- cohort-only plots (no sample needed) --------------------


def plot_cohort_heatmap(
    save_to_filename=None,
    save_dpi=300,
    figsize=(18, 14),
    zscore=True,
):
    """Heatmap of curated cancer-type genes × cancer types."""
    import numpy as np
    from trufflepig.reference import pan_cancer_expression, cancer_types
    ct_df = get_data("cancer-type-genes")
    gene_symbols = sorted(ct_df["Symbol"].unique())

    # Get expression
    ref = pan_cancer_expression(
        genes=gene_symbols,
        normalize="housekeeping",
        technical_rna_normalize=True,
    )
    codes = cancer_types()
    cohort_cols = [f"{c}_TPM" for c in codes if f"{c}_TPM" in ref.columns]
    codes = [c.removesuffix("_TPM") for c in cohort_cols]

    ref_dedup = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    present = [s for s in gene_symbols if s in ref_dedup.index]
    matrix = ref_dedup.loc[present, cohort_cols].astype(float)
    matrix = np.log2(matrix + 1)

    if zscore:
        row_mean = matrix.mean(axis=1)
        row_std = matrix.std(axis=1).clip(lower=0.1)
        matrix = matrix.sub(row_mean, axis=0).div(row_std, axis=0)
        cmap, vmin, vmax = "RdBu_r", -3, 3
        subtitle = "z-score of log2 expression across cancers"
        cbar_label = "z-score"
    else:
        cmap, vmin, vmax = "RdBu_r", -5, 5
        subtitle = "log2 housekeeping-normalized"
        cbar_label = "log2(HK-normalized)"
    matrix.columns = codes

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        matrix.values,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(codes, fontsize=7, rotation=90)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=4)
    ax.set_title(
        f"Curated cancer-type genes × TCGA cancer types — {subtitle}", fontsize=11
    )
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.6)
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


def plot_cohort_disjoint_counts(
    n_genes=30,
    save_to_filename=None,
    save_dpi=300,
    figsize=(12, 10),
):
    """Bar chart of disjoint signature gene counts per cancer type (no sample)."""
    import numpy as np
    from trufflepig.reference import top_enriched_per_cancer_type
    sig = top_enriched_per_cancer_type(n=n_genes, disjoint=True, min_fold=2.0)
    stats = [(code, len(genes)) for code, genes in sig.items()]
    stats.sort(key=lambda x: -x[1])

    codes = [s[0] for s in stats]
    counts = [s[1] for s in stats]
    labels = [f"{c} ({CANCER_TYPE_NAMES.get(c, c)})" for c in codes]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(codes))
    ax.barh(y, counts, color="#2166ac", edgecolor="none", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"Disjoint signature genes (of {n_genes} max)", fontsize=10)
    ax.set_title("Cancer-type-specific disjoint gene counts", fontsize=11)
    ax.invert_yaxis()

    for i, (code, count) in enumerate(stats):
        top3 = sig[code][:3]
        ax.text(count + 0.3, i, ", ".join(top3), fontsize=5.5, va="center", alpha=0.7)

    fig.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


def plot_cohort_pca(
    n_genes=20,
    save_to_filename=None,
    save_dpi=300,
    figsize=(12, 10),
):
    """PCA of 33 TCGA cancer type centroids (no sample)."""
    import numpy as np
    from trufflepig.reference import top_enriched_per_cancer_type, pan_cancer_expression
    sig = top_enriched_per_cancer_type(n=n_genes, disjoint=True)
    all_symbols = set()
    for genes in sig.values():
        all_symbols.update(genes)

    ref = pan_cancer_expression(
        normalize="housekeeping",
        technical_rna_normalize=True,
    )
    cohort_cols = [c for c in ref.columns if c.endswith("_TPM")]
    codes = [c.removesuffix("_TPM") for c in cohort_cols]

    ref_filtered = ref[ref["Symbol"].isin(all_symbols)].copy()
    gene_order = sorted(ref_filtered["Symbol"].unique())

    feature_matrix = []
    for col in cohort_cols:
        vals = []
        for sym in gene_order:
            row_mask = ref_filtered["Symbol"] == sym
            v = ref_filtered.loc[row_mask, col].astype(float).values
            vals.append(v[0] if len(v) > 0 else 0)
        feature_matrix.append(vals)

    X = np.array(feature_matrix)
    X = np.log2(X + 1)

    coords, variance = _pca_coordinates(X, n_components=2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=80,
        alpha=0.7,
        color="steelblue",
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )
    for i, code in enumerate(codes):
        ax.text(
            coords[i, 0], coords[i, 1], f" {code}", fontsize=8, alpha=0.8, va="center"
        )

    ax.set_xlabel(f"PC1 ({variance[0]:.0%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({variance[1]:.0%} variance)", fontsize=11)
    ax.set_title("TCGA cancer type centroids in gene-signature PCA space", fontsize=12)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


def plot_cohort_therapy_targets(
    save_to_filename=None,
    save_dpi=300,
    figsize=(16, 10),
    zscore=True,
):
    """Heatmap of therapy targets × cancer types showing expression.

    Rows are therapy target genes (from ADC, CAR-T, bispecific, radioligand,
    TCR-T registries), columns are cancer types.
    """
    import numpy as np
    from trufflepig.reference import (
        pan_cancer_expression, cancer_types,
    )

    # Collect all therapy targets
    all_targets = {}
    for therapy in ["ADC", "CAR-T", "TCR-T", "bispecific-antibodies", "radioligand"]:
        d = therapy_target_gene_id_to_name(therapy)
        all_targets.update(d)
    target_symbols = sorted(set(all_targets.values()))

    ref = pan_cancer_expression(
        genes=target_symbols,
        normalize="housekeeping",
        technical_rna_normalize=True,
    )
    codes = cancer_types()
    cohort_cols = [f"{c}_TPM" for c in codes if f"{c}_TPM" in ref.columns]
    codes = [c.removesuffix("_TPM") for c in cohort_cols]

    ref_dedup = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    present = [s for s in target_symbols if s in ref_dedup.index]
    matrix = ref_dedup.loc[present, cohort_cols].astype(float)
    matrix = np.log2(matrix + 1)

    if zscore:
        row_mean = matrix.mean(axis=1)
        row_std = matrix.std(axis=1).clip(lower=0.1)
        matrix = matrix.sub(row_mean, axis=0).div(row_std, axis=0)
        cmap, vmin, vmax = "RdBu_r", -3, 3
        subtitle = "z-score of log2 expression across cancers"
        cbar_label = "z-score"
    else:
        cmap, vmin, vmax = "YlOrRd", -5, 3
        subtitle = "log2 housekeeping-normalized"
        cbar_label = "log2(HK-normalized)"
    matrix.columns = codes

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        matrix.values,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(codes, fontsize=7, rotation=90)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=6)
    ax.set_title(f"Therapy targets × TCGA cancer types — {subtitle}", fontsize=11)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.6)
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


def _plot_geneset_by_cancer_heatmap(
    gene_symbols,
    title,
    save_to_filename=None,
    save_dpi=300,
    figsize=(16, 12),
    cmap="YlOrRd",
    top_n_per_cancer=None,
    zscore=True,
):
    """Shared helper: heatmap of gene set × cancer types."""
    import numpy as np
    from trufflepig.reference import pan_cancer_expression, cancer_types
    ref = pan_cancer_expression(
        genes=gene_symbols,
        normalize="housekeeping",
        technical_rna_normalize=True,
    )
    codes = cancer_types()
    cohort_cols = [f"{c}_TPM" for c in codes if f"{c}_TPM" in ref.columns]
    codes = [c.removesuffix("_TPM") for c in cohort_cols]

    ref_dedup = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    present = [s for s in gene_symbols if s in ref_dedup.index]

    if not present:
        return None, None

    matrix = ref_dedup.loc[present, cohort_cols].astype(float)

    # Filter to top N genes by max expression across any cancer type
    if top_n_per_cancer and len(present) > top_n_per_cancer:
        max_expr = matrix.max(axis=1)
        top_idx = max_expr.nlargest(top_n_per_cancer).index
        matrix = matrix.loc[top_idx]
        present = list(top_idx)

    # Sort rows by mean expression (highest at top)
    row_means = matrix.mean(axis=1)
    sort_order = row_means.sort_values(ascending=False).index
    matrix = matrix.loc[sort_order]
    present = list(sort_order)

    matrix = np.log2(matrix + 1)

    if zscore:
        row_mean = matrix.mean(axis=1)
        row_std = matrix.std(axis=1).clip(lower=0.1)
        matrix = matrix.sub(row_mean, axis=0).div(row_std, axis=0)
        use_cmap, vmin, vmax = "RdBu_r", -3, 3
        subtitle = "z-score of log2 expression across cancers"
        cbar_label = "z-score"
    else:
        use_cmap, vmin, vmax = cmap, -5, 3
        subtitle = "log2 housekeeping-normalized"
        cbar_label = "log2(HK-normalized)"
    matrix.columns = codes

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        matrix.values,
        aspect="auto",
        cmap=use_cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(codes, fontsize=7, rotation=90)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=5 if len(present) > 50 else 6)
    ax.set_title(f"{title} — {subtitle}", fontsize=11)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.6)
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


def plot_cohort_surface_proteins(
    save_to_filename=None,
    save_dpi=300,
    figsize=(16, 14),
    zscore=True,
):
    """Heatmap of cancer surfaceome targets × cancer types."""
    from pirlygenes.gene_sets_cancer import cancer_surfaceome_gene_names

    genes = sorted(cancer_surfaceome_gene_names())
    return _plot_geneset_by_cancer_heatmap(
        genes,
        "Tumor-specific surface proteins (TCSA L3) × cancer types",
        save_to_filename=save_to_filename,
        save_dpi=save_dpi,
        figsize=figsize,
        cmap="YlOrRd",
        zscore=zscore,
    )


def plot_cohort_ctas(
    save_to_filename=None,
    save_dpi=300,
    figsize=(16, 14),
    zscore=True,
):
    """Heatmap of CTA genes × cancer types."""
    import numpy as np
    from pirlygenes.gene_sets_cancer import CTA_gene_names

    from trufflepig.reference import pan_cancer_expression, cancer_types
    genes = sorted(CTA_gene_names())
    ref = pan_cancer_expression(
        genes=genes,
        technical_rna_normalize=True,
    )  # TPM values after technical-RNA normalization
    codes = cancer_types()
    cohort_cols = [f"{c}_TPM" for c in codes if f"{c}_TPM" in ref.columns]
    codes_clean = [c.removesuffix("_TPM") for c in cohort_cols]

    ref_dedup = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    present = [s for s in genes if s in ref_dedup.index]
    matrix = ref_dedup.loc[present, cohort_cols].astype(float)

    # Filter to top 50 by max expression, sort by mean descending
    max_expr = matrix.max(axis=1)
    top50 = max_expr.nlargest(50).index
    matrix = matrix.loc[top50]
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]
    present = list(matrix.index)

    matrix = np.log2(matrix + 1)

    if zscore:
        row_mean = matrix.mean(axis=1)
        row_std = matrix.std(axis=1).clip(lower=0.1)
        matrix = matrix.sub(row_mean, axis=0).div(row_std, axis=0)
        cmap, vmin, vmax = "RdBu_r", -3, 3
        subtitle = "z-score of log2 cohort TPM across cancers"
        cbar_label = "z-score"
    else:
        cmap, vmin, vmax = "magma_r", -3, 8
        subtitle = "log2 cohort TPM"
        cbar_label = "log2(TPM + 1)"
    matrix.columns = codes_clean

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        matrix.values,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(codes_clean)))
    ax.set_xticklabels(codes_clean, fontsize=7, rotation=90)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=6)
    ax.set_title(
        f"Cancer-testis antigens × cancer types (top 50) — {subtitle}", fontsize=11
    )
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.6)
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


_METHOD_LABELS = {
    "zscore": "z-score",
    "hk": "HK-normalized",
    "hk_zscore": "HK z-score",
    "rank": "percentile rank",
    "hierarchy": "hierarchical support space",
    "tme": "TME-low genes",
    "score": "signature scores",
    "bottleneck": "bottleneck genes",
    "panref": "pan-reference genes",
}


def plot_cancer_type_pca(
    df_gene_expr,
    n_genes=10,
    method="zscore",
    include_normals=False,
    include_subtypes=False,
    save_to_filename=None,
    save_dpi=300,
    figsize=(12, 10),
):
    """PCA scatter showing where the sample falls among cancer-type centroids."""
    X, labels = _cancer_type_feature_matrix(
        df_gene_expr,
        n_genes=n_genes,
        method=method,
        include_normals=include_normals,
        include_subtypes=include_subtypes,
    )
    coords, variance = _pca_coordinates(X, n_components=2)
    mlabel = _METHOD_LABELS.get(method)
    title = "Sample among TCGA parent cancer cohorts — PCA"
    if include_subtypes and include_normals:
        title = "Sample among TCGA parent cohorts, subtype references, and normal tissues — PCA"
    elif include_subtypes:
        title = "Sample among TCGA parent cohorts and subtype references — PCA"
    elif include_normals:
        title = "Sample among TCGA parent cancer cohorts and normal tissues — PCA"
    if mlabel:
        title += f" ({mlabel})"
    return _plot_embedding_with_labels(
        coords,
        labels,
        title=title,
        xlabel=f"PC1 ({variance[0]:.0%} variance)",
        ylabel=f"PC2 ({variance[1]:.0%} variance)",
        method=method,
        save_to_filename=save_to_filename,
        save_dpi=save_dpi,
        figsize=figsize,
    )


def plot_cancer_type_mds(
    df_gene_expr,
    n_genes=10,
    method="zscore",
    include_normals=False,
    include_subtypes=False,
    label_nearest_cancers=None,
    label_nearest_normals=None,
    label_all=True,
    focus_nearest_cancers=None,
    focus_nearest_normals=None,
    save_to_filename=None,
    save_dpi=300,
    figsize=(12, 10),
):
    """MDS embedding of the sample with TCGA cancer type centroids."""
    X, labels = _cancer_type_feature_matrix(
        df_gene_expr,
        n_genes=n_genes,
        method=method,
        include_normals=include_normals,
        include_subtypes=include_subtypes,
    )
    full_distances = _euclidean_distances(X)
    focused = False
    if "SAMPLE" in labels:
        sample_idx = labels.index("SAMPLE")
        keep_idx = _sample_neighborhood_indices(
            labels,
            full_distances,
            sample_idx,
            nearest_cancers=focus_nearest_cancers,
            nearest_normals=focus_nearest_normals,
        )
        if len(keep_idx) < len(labels):
            X = X[keep_idx]
            labels = [labels[i] for i in keep_idx]
            full_distances = full_distances[np.ix_(keep_idx, keep_idx)]
            focused = True
    distances = full_distances
    nearest_neighbors = None
    if "SAMPLE" in labels:
        sample_idx = labels.index("SAMPLE")
        nearest_neighbors = _nearest_neighbors_from_distances(
            labels, distances, sample_idx
        )
    coords = _classical_mds_coordinates(distances, n_components=2)
    mlabel = _METHOD_LABELS.get(method)
    title = "Sample among TCGA parent cancer cohorts — MDS"
    if include_subtypes and include_normals:
        title = "Sample among TCGA parent cohorts, subtype references, and normal tissues — MDS"
    elif include_subtypes:
        title = "Sample among TCGA parent cohorts and subtype references — MDS"
    elif include_normals:
        title = "Sample among TCGA parent cancer cohorts and normal tissues — MDS"
    if focused:
        title = title.replace("Sample among", "Sample neighborhood among")
    if mlabel:
        title += f" ({mlabel})"
    xlabel = "MDS1"
    ylabel = "MDS2"
    if method in {"zscore", "hk_zscore", "tme", "bottleneck", "panref", "hk"}:
        xlabel = "MDS1 (log-expression distance)"
        ylabel = "MDS2 (log-expression distance)"
    return _plot_embedding_with_labels(
        coords,
        labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        method=method,
        label_nearest_cancers=label_nearest_cancers,
        label_nearest_normals=label_nearest_normals,
        label_all=label_all,
        nearest_neighbors=nearest_neighbors,
        nearest_basis="input feature distance",
        save_to_filename=save_to_filename,
        save_dpi=save_dpi,
        figsize=figsize,
    )


def plot_cancer_type_neighborhood(
    df_gene_expr,
    n_genes=10,
    method="panref",
    include_normals=True,
    include_subtypes=True,
    label_nearest_cancers=5,
    label_nearest_normals=5,
    label_all=False,
    focus_nearest_cancers=25,
    focus_nearest_normals=10,
    save_to_filename=None,
    save_dpi=300,
    figsize=(12, 10),
):
    """Rank nearest reference cohorts by original sample-to-reference distance.

    The global MDS plot is useful for spatial context but necessarily distorts
    some local relationships. This plot is deliberately not an embedding: it is
    a ranked distance chart, so the nearest cancer, subtype, and normal-tissue
    references are readable and all labels are explicit.
    """
    X, labels = _cancer_type_feature_matrix(
        df_gene_expr,
        n_genes=n_genes,
        method=method,
        include_normals=include_normals,
        include_subtypes=include_subtypes,
    )
    distances = _euclidean_distances(X)
    if "SAMPLE" not in labels:
        return None
    sample_idx = labels.index("SAMPLE")
    nearest_neighbors = _nearest_neighbors_from_distances(labels, distances, sample_idx)

    cancer_like_target = (
        focus_nearest_cancers
        if focus_nearest_cancers is not None
        else label_nearest_cancers
    )
    normal_target = (
        focus_nearest_normals
        if focus_nearest_normals is not None
        else label_nearest_normals
    )
    cancer_like_target = 10 if cancer_like_target is None else int(cancer_like_target)
    normal_target = 5 if normal_target is None else int(normal_target)

    grouped = {"cancer": [], "subtype": [], "normal": []}
    for dist, label, kind in nearest_neighbors:
        if kind in grouped:
            grouped[kind].append((dist, label, kind))

    cancer_like = (grouped["cancer"] + grouped["subtype"])
    cancer_like.sort(key=lambda item: (item[0], item[1]))
    selected = cancer_like[: max(0, cancer_like_target)]
    selected += grouped["normal"][: max(0, normal_target)]
    if not selected:
        return None

    kind_order = {"cancer": 0, "subtype": 1, "normal": 2}
    selected.sort(key=lambda item: (kind_order.get(item[2], 9), item[0], item[1]))

    y_positions = []
    y_labels = []
    colors = []
    values = []
    group_headers = []
    y = 0.0
    last_kind = None
    kind_labels = {
        "cancer": "TCGA parent cancers",
        "subtype": "Subtype references",
        "normal": "Normal tissues",
    }
    kind_colors = {
        "cancer": "#4a90d9",
        "subtype": "#6da3cf",
        "normal": "#8da08d",
    }
    for dist, label, kind in selected:
        if kind != last_kind:
            if y_positions:
                y += 0.65
            group_headers.append((y - 0.62, kind_labels.get(kind, kind)))
            last_kind = kind
        y_positions.append(y)
        y_labels.append(str(label).replace("normal:", ""))
        values.append(float(dist))
        colors.append(kind_colors.get(kind, "#777777"))
        y += 1.0

    fig, ax = plt.subplots(figsize=(11, max(4.8, 0.36 * max(y_positions) + 2.2)))
    ax.barh(y_positions, values, color=colors, height=0.72, alpha=0.88)
    for y_pos, value in zip(y_positions, values):
        ax.text(value, y_pos, f"  {value:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.invert_yaxis()
    xmin, xmax = ax.get_xlim()
    for header_y, label in group_headers:
        ax.axhline(header_y + 0.18, color="#e4e4e4", linewidth=0.8, zorder=0)
        ax.text(
            xmin,
            header_y,
            label,
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#555555",
            clip_on=False,
        )
    ax.set_xlim(xmin, max(xmax, max(values) * 1.10))
    ax.set_ylim(max(y_positions) + 0.8, -1.05)

    mlabel = _METHOD_LABELS.get(method, method)
    title = "Nearest reference distances"
    if include_subtypes and include_normals:
        title += " — TCGA parents, subtype refs, and normal tissues"
    elif include_subtypes:
        title += " — TCGA parents and subtype refs"
    elif include_normals:
        title += " — TCGA parents and normal tissues"
    if mlabel:
        title += f" ({mlabel})"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Input feature-space distance from sample (lower = closer)")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.5,
        0.965,
        "This ranking preserves the original sample-to-reference distances; the MDS plot is only a 2-D projection.",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.93])
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


def plot_cancer_type_umap(
    df_gene_expr,
    n_genes=10,
    method="zscore",
    save_to_filename=None,
    save_dpi=300,
    figsize=(12, 10),
):
    """UMAP embedding of the sample with TCGA cancer type centroids."""
    from umap import UMAP

    X, labels = _cancer_type_feature_matrix(
        df_gene_expr, n_genes=n_genes, method=method
    )
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value")
        coords = UMAP(
            n_components=2,
            n_neighbors=min(15, len(labels) - 1),
            random_state=42,
        ).fit_transform(X)
    mlabel = _METHOD_LABELS.get(method)
    title = "Sample among TCGA cancer types — UMAP"
    if mlabel:
        title += f" ({mlabel})"
    return _plot_embedding_with_labels(
        coords,
        labels,
        title=title,
        xlabel="UMAP1",
        ylabel="UMAP2",
        method=method,
        save_to_filename=save_to_filename,
        save_dpi=save_dpi,
        figsize=figsize,
    )
