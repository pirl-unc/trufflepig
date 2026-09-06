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

"""Deep-dive therapy-target and CTA plots.

Three families of visualisation:

1. **Actionable surface targets** — observed TPM + tumor-cell-equivalent TPM
   for a curated panel vs the matched TCGA cancer type, healthy vital
   tissues, and pan-cancer TCGA.
2. **CTA deep dive** — same treatment for cancer-testis antigens.
3. **Tumor vs TME attribution** — per-gene bar showing how much of
   the observed signal comes from tumor vs microenvironment.

All three are generalised: a cancer-type-aware panel function selects
the right genes for any TCGA code.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pirlygenes.gene_sets_cancer import (

    CTA_gene_id_to_name,

)

from trufflepig.reference import (

    pan_cancer_expression,

)
from .plot_scatter import resolve_cancer_type
from .reporting import (
    clinical_maturity_info,
    expression_independent_indication,
    indication_biomarker_label,
    normal_expression_context,
    supplied_variant_supports_target_row,
    target_hla_eligibility,
    therapy_state_caution,
    tumor_attribution_context,
)
from .treatment_history import (
    treatment_history_blocks_row,
    treatment_history_context,
    treatment_history_marks_current,
    treatment_history_rank,
    treatment_history_supports_review,
)

# ── Essential tissues for safety context ─────────────────────────────────

VITAL_TISSUES = [
    "heart_muscle",
    "liver",
    "kidney",
    "lung",
    "cerebral_cortex",
    "colon",
    "bone_marrow",
    "pancreas",
    "stomach",
]

_PRIORITY_PHASE_PRIORITY = {
    "approved": 0,
    "phase_3": 1,
    "phase_2": 2,
    "phase_1": 3,
    "preclinical": 4,
}
_PRIORITY_PHASE_POINTS = {
    "approved": 3.0,
    "phase_3": 2.6,
    "phase_2": 2.2,
    "phase_1": 1.7,
    "preclinical": 1.0,
}
_PRIORITY_STATUS_ORDER = {
    "patient_treatment_evidence": 0,
    "approved_disease_matched": 1,
    "clinical_disease_matched": 2,
    "approved_other_context": 3,
    "exploratory_or_expression_linked": 4,
}
_PRIORITY_STATUS_LABELS = {
    "patient_treatment_evidence": "Patient treatment evidence",
    "approved_disease_matched": "Approved pathway / eligibility pending",
    "clinical_disease_matched": "Clinical trial / eligibility pending",
    "approved_other_context": "Approved elsewhere / generic target",
    "exploratory_or_expression_linked": "Exploratory / expression-linked",
}
_PRIORITY_SOURCE_RANK = {
    "tumor_supported": 0,
    "mixed_source": 1,
    "background_dominant": 2,
}
_PRIORITY_SOURCE_MARKERS = {
    "tumor_supported": "o",
    "mixed_source": "D",
    "background_dominant": "X",
}
_PRIORITY_NORMAL_COLORS = {
    "cta_restricted": "#2ca02c",
    "restricted_outside_lineage": "#2ca02c",
    "same_lineage_expected": "#1f77b4",
    "broad_healthy_expression": "#ff7f0e",
    "vital_tissue_concern": "#d62728",
}
_PRIORITY_SOURCE_COLORS = {
    "tumor_supported": "#2e8b57",
    "mixed_source": "#c28f2c",
    "background_dominant": "#c44e52",
}

# ── Per-cancer-type actionable surface target panels ─────────────────────
#
# Curated: genes that are current or emerging therapy targets for each
# cancer type.  Includes approved targets, late-phase trial targets, and
# diagnostic biomarkers that are actionable for patient selection.

_CANCER_SURFACE_TARGETS = {
    "PRAD": [
        "FOLH1",  # PSMA — 177Lu-PSMA-617 (Pluvicto), approved
        "KLK3",  # PSA — diagnostic, AR target readout
        "KLK2",  # hK2 — diagnostic, AR target
        "STEAP1",  # STEAP1 — AMG 509 (xaluritamig), phase III
        "STEAP2",  # STEAP2 — emerging ADC/bispecific target
        "PSCA",  # PSCA — CAR-T trials
        "TACSTD2",  # TROP2 — sacituzumab govitecan (Trodelvy)
        "CD276",  # B7-H3 — enoblituzumab, CAR-T
        "ERBB2",  # HER2 — T-DXd (Enhertu), expanding indications
        "CD46",  # CD46 — FOR46 ADC, PRAD trials
        "DLL3",  # DLL3 — tarlatamab (Imdelltra), NEPC
        "GRPR",  # GRP receptor — 177Lu-RM2 PSMA alternative
        "AR",  # Androgen receptor — enzalutamide/abiraterone readout
        "NKX3-1",  # Prostate lineage TF — diagnostic
        "HOXB13",  # Prostate lineage — diagnostic, germline marker
        "TMPRSS2",  # TMPRSS2 — fusion partner, AR target readout
    ],
    "BRCA": [
        "ERBB2",  # HER2 — trastuzumab, T-DXd
        "ESR1",  # ER — endocrine therapy readout
        "PGR",  # PR — endocrine therapy readout
        "TACSTD2",  # TROP2 — sacituzumab govitecan
        "NECTIN4",  # Nectin-4 — enfortumab vedotin (expanding)
        "CD276",  # B7-H3
        "FOLR1",  # FRα — mirvetuximab soravtansine
        "MUC16",  # CA-125 — diagnostic, ADC target
        "MUC1",  # MUC1 — CAR-T trials
        "CEACAM5",  # CEA — tusamitamab ravtansine
    ],
    "LUAD": [
        "EGFR",  # EGFR — osimertinib, amivantamab
        "ERBB2",  # HER2 — T-DXd
        "MET",  # MET — capmatinib, tepotinib
        "TACSTD2",  # TROP2 — datopotamab deruxtecan
        "CEACAM5",  # CEA — tusamitamab ravtansine
        "CD276",  # B7-H3
        "MSLN",  # Mesothelin — CAR-T
        "NECTIN4",  # Nectin-4
        "DLL3",  # DLL3 — SCLC/NEPC, expanding
        "FOLR1",  # FRα
    ],
    "LUSC": [
        "EGFR",
        "NECTIN4",
        "TACSTD2",
        "CD276",
        "FGFR1",  # FGFR — erdafitinib
        "DLL3",
    ],
    "COAD": [
        "CEACAM5",  # CEA
        "ERBB2",  # HER2
        "EGFR",  # EGFR — cetuximab, panitumumab
        "TACSTD2",
        "CD276",
        "GPC3",  # Glypican-3 — trials
        "MET",
        "LGR5",  # Stem cell marker
        "GUCY2C",  # Guanylyl cyclase C — CAR-T trials
    ],
    "SKCM": [
        "CD274",  # PD-L1
        "CTLA4",  # CTLA-4 — ipilimumab
        "PDCD1",  # PD-1
        "MLANA",  # MART-1 — TCR-T / TIL therapy
        "PMEL",  # gp100 — tebentafusp
        "TYR",  # Tyrosinase — TCR-T
        "TACSTD2",
        "CD276",
    ],
    "OV": [
        "FOLR1",  # FRα — mirvetuximab soravtansine (approved)
        "MUC16",  # CA-125 — diagnostic, ADC
        "MSLN",  # Mesothelin
        "TACSTD2",  # TROP2
        "ERBB2",
        "NECTIN4",
        "NaPi2b",  # SLC34A2 — lifastuzumab vedotin
        "CD276",
    ],
    "LIHC": [
        "GPC3",  # Glypican-3 — CAR-T, bispecific trials
        "CD274",  # PD-L1
        "AFP",  # AFP — diagnostic
        "EPCAM",  # EpCAM — catumaxomab
        "TACSTD2",
        "MET",
        "FGFR4",  # FGFR4 — fisogatinib
    ],
    "GBM": [
        "EGFR",  # EGFRvIII — rindopepimut, CAR-T
        "IL13RA2",  # IL-13Rα2 — CAR-T
        "DLL3",  # DLL3 — NEPC/neuroendocrine
        "CD276",  # B7-H3
        "GD2",  # GD2 — dinutuximab (repurposing)
        "PDGFRA",
    ],
    "BLCA": [
        "NECTIN4",  # Nectin-4 — enfortumab vedotin (approved)
        "TACSTD2",  # TROP2 — sacituzumab govitecan
        "ERBB2",  # HER2
        "FGFR3",  # FGFR3 — erdafitinib (approved)
        "CD274",  # PD-L1
        "CD276",
    ],
}

# Default fallback for cancer types without a curated panel
_DEFAULT_SURFACE_TARGETS = [
    "TACSTD2",
    "CD276",
    "ERBB2",
    "NECTIN4",
    "CEACAM5",
    "MSLN",
    "FOLR1",
    "EPCAM",
    "MUC1",
    "CD274",
]


def actionable_surface_targets(cancer_type):
    """Return the curated actionable surface target panel for a cancer type."""
    code = resolve_cancer_type(cancer_type)
    return list(_CANCER_SURFACE_TARGETS.get(code, _DEFAULT_SURFACE_TARGETS))


# ── Reference data helpers ───────────────────────────────────────────────


def _build_reference_context(symbols, cancer_code):
    """Build a reference context dict for a list of gene symbols.

    Returns {symbol: {cancer_tpm, origin_tissue_ntpm,
    vital_tissues: {tissue: ntpm}, all_cancer_median, all_cancer_max}}.
    """
    ref = (
        pan_cancer_expression(technical_rna_normalize=True)
        .drop_duplicates(subset="Symbol")
        .set_index("Symbol")
    )
    cohort_cols = [c for c in ref.columns if c.endswith("_TPM")]
    from .tumor_purity import CANCER_TO_TISSUE

    origin_tissue = CANCER_TO_TISSUE.get(cancer_code)

    result = {}
    for sym in symbols:
        if sym not in ref.index:
            continue
        row = ref.loc[sym]
        cancer_col = f"{cancer_code}_TPM"
        cancer_tpm = float(row[cancer_col]) if cancer_col in ref.columns else 0.0

        origin_ntpm = 0.0
        if origin_tissue:
            origin_col = f"{origin_tissue}_nTPM"
            if origin_col in ref.columns:
                origin_ntpm = float(row[origin_col])

        vital = {}
        for tissue in VITAL_TISSUES:
            col = f"{tissue}_nTPM"
            if col in ref.columns:
                vital[tissue] = float(row[col])

        all_cancer_vals = [float(row[c]) for c in cohort_cols if c in ref.columns]
        result[sym] = {
            "cancer_tpm": cancer_tpm,
            "origin_tissue_ntpm": origin_ntpm,
            "vital_tissues": vital,
            "all_cancer_median": float(np.median(all_cancer_vals))
            if all_cancer_vals
            else 0.0,
            "all_cancer_max": float(np.max(all_cancer_vals))
            if all_cancer_vals
            else 0.0,
        }
    return result


def _get_sample_tpm_by_symbol(df_gene_expr):
    """Return {symbol: tpm} for the sample, preferring direct symbol column."""
    from .sample_context import _build_tpm_by_symbol

    return _build_tpm_by_symbol(df_gene_expr)


def _estimate_tumor_tpm(observed, tme_ref, purity):
    """Single-point tumor-adjusted TPM."""
    p = max(purity, 0.01)
    return max(0.0, (observed - (1.0 - p) * tme_ref) / p)


def _finite_float(value, default=0.0):
    try:
        result = float(value)
    except Exception:
        return default
    if not np.isfinite(result):
        return default
    return result


def _get_tme_reference(symbols, cancer_code):
    """Return {symbol: tme_ntpm} — mean expression across TME tissues."""
    ref = (
        pan_cancer_expression(technical_rna_normalize=True)
        .drop_duplicates(subset="Symbol")
        .set_index("Symbol")
    )
    from .plot_tumor_expr import _TME_TISSUES

    tme_cols = [f"{t}_nTPM" for t in _TME_TISSUES if f"{t}_nTPM" in ref.columns]
    result = {}
    for sym in symbols:
        if sym in ref.index and tme_cols:
            result[sym] = float(ref.loc[sym, tme_cols].astype(float).mean())
        else:
            result[sym] = 0.0
    return result


# ── Plot 1: Actionable targets deep dive ────────────────────────────────


def plot_actionable_targets(
    df_gene_expr,
    cancer_type,
    purity_estimate=None,
    custom_genes=None,
    ranges_df=None,
    save_to_filename=None,
    save_dpi=300,
    title=None,
):
    """Dot plot: observed TPM + tumor-cell-equivalent TPM for actionable targets
    vs TCGA cancer-type median, healthy tissues, and pan-cancer context.

    Each gene gets one row. Columns show:
    - Sample observed TPM (black dot)
    - Tumor-cell-equivalent TPM (red dot, if purity/range model available)
    - TCGA cancer-type median (blue bar)
    - Max vital-tissue expression (gray bar — safety signal)

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Expression data with gene ID / symbol + TPM.
    cancer_type : str
        TCGA code or alias.
    purity_estimate : float or None
        Overall purity estimate (0–1). If None, tumor-adjusted not shown.
    custom_genes : list[str] or None
        Override the default curated panel.
    ranges_df : DataFrame or None
        Optional output of ``estimate_tumor_expression_ranges()``. When present,
        tumor-cell TPM comes from the central attribution/deconvolution model
        instead of the older generic-TME fallback.
    """
    try:
        cancer_code = resolve_cancer_type(cancer_type)
    except Exception:
        cancer_code = str(cancer_type)
    genes = custom_genes or actionable_surface_targets(cancer_code)

    sample_tpm = _get_sample_tpm_by_symbol(df_gene_expr)
    ref_ctx = _build_reference_context(genes, cancer_code)
    tme_ref = _get_tme_reference(genes, cancer_code)
    ranges_by_symbol = {}
    if ranges_df is not None and "symbol" in getattr(ranges_df, "columns", []):
        ranges_by_symbol = {
            str(row.get("symbol") or ""): row
            for _, row in ranges_df.iterrows()
            if str(row.get("symbol") or "")
        }

    # Filter to genes present in reference
    genes = [g for g in genes if g in ref_ctx]
    if not genes:
        return None

    # Build data
    rows = []
    for sym in genes:
        ctx = ref_ctx[sym]
        range_row = ranges_by_symbol.get(sym)
        obs = (
            float(range_row.get("observed_tpm") or 0.0)
            if range_row is not None
            else sample_tpm.get(sym, 0.0)
        )
        tme = tme_ref.get(sym, 0.0)
        tumor_adj = None
        tumor_low = None
        tumor_high = None
        central_model = False
        source_label = ""
        if range_row is not None:
            attr_mid = _finite_float(range_row.get("attr_tumor_tpm"), None)
            if purity_estimate and attr_mid is not None:
                purity = max(float(purity_estimate), 0.01)
                tumor_adj = attr_mid / purity
                tumor_low = (
                    _finite_float(range_row.get("attr_tumor_tpm_low"), tumor_adj * purity)
                    / purity
                )
                tumor_high = (
                    _finite_float(
                        range_row.get("attr_tumor_tpm_high"), tumor_adj * purity
                    )
                    / purity
                )
            else:
                tumor_adj = _finite_float(
                    range_row.get("tumor_cell_tpm"),
                    _finite_float(range_row.get("median_est"), 0.0),
                )
                tumor_low = _finite_float(
                    range_row.get("tumor_cell_tpm_low"),
                    _finite_float(range_row.get("est_1"), tumor_adj),
                )
                tumor_high = _finite_float(
                    range_row.get("tumor_cell_tpm_high"),
                    _finite_float(range_row.get("est_9"), tumor_adj),
                )
            central_model = True
            try:
                source_label = tumor_attribution_context(range_row)["label"]
            except Exception:
                source_label = ""
        elif purity_estimate:
            tumor_adj = _estimate_tumor_tpm(obs, tme, purity_estimate)
            tumor_low = tumor_adj
            tumor_high = tumor_adj
        max_vital = max(ctx["vital_tissues"].values()) if ctx["vital_tissues"] else 0.0
        rows.append(
            {
                "symbol": sym,
                "observed": obs,
                "tumor_adjusted": tumor_adj,
                "tumor_low": tumor_low,
                "tumor_high": tumor_high,
                "central_model": central_model,
                "source_label": source_label,
                "cancer_median": ctx["cancer_tpm"],
                "max_vital_tissue": max_vital,
                "origin_tissue": ctx["origin_tissue_ntpm"],
                "tme_background": tme,
            }
        )

    # Sort by observed TPM descending
    rows.sort(key=lambda r: r["observed"], reverse=True)
    n = len(rows)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * n)))
    y_pos = np.arange(n)
    symbols = [r["symbol"] for r in rows]

    # Background bars
    ax.barh(
        y_pos,
        [r["cancer_median"] for r in rows],
        height=0.35,
        color="#4A90D9",
        alpha=0.3,
        label=f"TCGA {cancer_code} median",
    )
    ax.barh(
        y_pos + 0.35,
        [r["max_vital_tissue"] for r in rows],
        height=0.25,
        color="#999999",
        alpha=0.3,
        label="Max vital tissue (safety)",
    )

    # Sample dots
    observed_y = y_pos - 0.11
    adjusted_y = y_pos + 0.11

    ax.scatter(
        [r["observed"] for r in rows],
        observed_y,
        color="black",
        s=60,
        zorder=5,
        label="Sample observed TPM",
    )

    if any(r["tumor_adjusted"] is not None for r in rows):
        for i, row in enumerate(rows):
            if row["tumor_adjusted"] is None:
                continue
            if row["tumor_low"] is not None and row["tumor_high"] is not None:
                ax.hlines(
                    adjusted_y[i],
                    max(float(row["tumor_low"]), 0.01),
                    max(float(row["tumor_high"]), 0.01),
                    color="#E74C3C",
                    linewidth=2.0,
                    alpha=0.35,
                    zorder=4,
                )
        tumor_vals = [
            r["tumor_adjusted"] if r["tumor_adjusted"] is not None else np.nan
            for r in rows
        ]
        ax.scatter(
            tumor_vals,
            adjusted_y,
            color="#E74C3C",
            s=60,
            marker="D",
            zorder=5,
            label=(
                "Estimated tumor TPM (RNA model)"
                if any(r["central_model"] for r in rows)
                else f"Fallback tumor-cell estimate (purity={purity_estimate:.0%})"
            ),
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(symbols, fontsize=9)
    ax.set_xlabel("TPM (log scale)")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title or f"Actionable target expression screen — {cancer_code}")

    # Sample-wide 90th-percentile anchor (faint dashed).
    try:
        from .plot_reference_lines import add_p90_reference_line
        from .common import build_sample_tpm_by_symbol

        add_p90_reference_line(
            ax,
            build_sample_tpm_by_symbol(df_gene_expr),
            orientation="vertical",
        )
    except Exception:
        pass

    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    return fig


# ── Plot 2: Tumor vs TME attribution ────────────────────────────────────


def plot_tumor_attribution(
    df_gene_expr,
    cancer_type,
    purity_estimate,
    custom_genes=None,
    category="surface",
    top_n=20,
    save_to_filename=None,
    save_dpi=300,
):
    """Stacked horizontal bar: for each gene, show how much of the
    observed TPM is attributable to tumor vs TME background.

    tumor_component = purity × tumor_adjusted
    tme_component = (1 - purity) × tme_reference
    """
    try:
        cancer_code = resolve_cancer_type(cancer_type)
    except Exception:
        cancer_code = str(cancer_type)

    if category == "CTA":
        cta_map = CTA_gene_id_to_name()
        genes = custom_genes or sorted(cta_map.values())
    else:
        genes = custom_genes or actionable_surface_targets(cancer_code)

    sample_tpm = _get_sample_tpm_by_symbol(df_gene_expr)
    tme_ref = _get_tme_reference(genes, cancer_code)
    ref_ctx = _build_reference_context(genes, cancer_code)

    genes = [g for g in genes if g in ref_ctx]
    if not genes:
        return None

    purity = max(purity_estimate, 0.01)
    rows = []
    for sym in genes:
        obs = sample_tpm.get(sym, 0.0)
        tme = tme_ref.get(sym, 0.0)
        tumor_adj = _estimate_tumor_tpm(obs, tme, purity)
        tumor_component = purity * tumor_adj
        tme_component = (1.0 - purity) * tme
        rows.append(
            {
                "symbol": sym,
                "observed": obs,
                "tumor_component": tumor_component,
                "tme_component": tme_component,
                "tumor_adjusted": tumor_adj,
                "pct_tumor": tumor_component / max(obs, 0.01),
            }
        )

    # Sort by observed, take top N
    rows.sort(key=lambda r: r["observed"], reverse=True)
    rows = [r for r in rows if r["observed"] > 0.1][:top_n]
    if not rows:
        return None

    n = len(rows)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * n)))
    y_pos = np.arange(n)
    symbols = [r["symbol"] for r in rows]

    tumor_vals = [r["tumor_component"] for r in rows]
    tme_vals = [r["tme_component"] for r in rows]

    ax.barh(
        y_pos,
        tumor_vals,
        color="#E74C3C",
        alpha=0.8,
        label="Estimated tumor component (RNA model)",
    )
    ax.barh(
        y_pos,
        tme_vals,
        left=tumor_vals,
        color="#4A90D9",
        alpha=0.5,
        label="Estimated stromal/immune reference contribution",
    )

    # Annotate with percentage
    for i, r in enumerate(rows):
        pct = r["pct_tumor"]
        if r["observed"] > 0.5:
            ax.text(
                r["observed"] + 0.5,
                i,
                f"{pct:.0%} estimated tumor",
                va="center",
                fontsize=7,
                color="#555555",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(symbols, fontsize=9)
    ax.set_xlabel("TPM (estimated attribution of measured patient bulk RNA)")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    cat_label = "CTAs" if category == "CTA" else "Actionable Targets"
    ax.set_title(
        f"Estimated Patient RNA Attribution — {cat_label} — "
        f"{cancer_code} (purity={purity:.0%})"
    )

    try:
        from .plot_reference_lines import add_p90_reference_line

        add_p90_reference_line(ax, sample_tpm, orientation="vertical")
    except Exception:
        pass
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    return fig


def plot_curated_target_evidence(
    ranges_df,
    target_panel,
    cancer_type,
    top_n=12,
    df_gene_expr=None,
    save_to_filename=None,
    save_dpi=300,
):
    """Integrated view for curated targets: expression evidence + context."""
    from matplotlib.lines import Line2D

    if (
        ranges_df is None
        or target_panel is None
        or len(ranges_df) == 0
        or len(target_panel) == 0
        or "symbol" not in ranges_df.columns
        or "symbol" not in target_panel.columns
    ):
        return None

    try:
        cancer_code = resolve_cancer_type(cancer_type)
    except Exception:
        cancer_code = str(cancer_type)
    sym_to_expr = {
        str(row["symbol"]): row
        for _, row in ranges_df.iterrows()
        if str(row.get("symbol") or "")
    }
    phase_priority = {
        "approved": 0,
        "phase_3": 1,
        "phase_2": 2,
        "phase_1": 3,
        "preclinical": 4,
    }
    best_by_symbol = {}
    for _, trow in target_panel.iterrows():
        sym = str(trow.get("symbol") or "").strip()
        if not sym or sym.lower() == "nan" or sym not in sym_to_expr:
            continue
        expr = sym_to_expr[sym]
        try:
            observed = float(expr.get("observed_tpm") or 0.0)
        except Exception:
            observed = 0.0
        if observed < 1.0:
            continue
        sort_key = (
            phase_priority.get(str(trow.get("phase") or ""), 99),
            -float(expr.get("attr_tumor_tpm") or 0.0),
            sym,
        )
        if sym not in best_by_symbol or sort_key < best_by_symbol[sym]["sort_key"]:
            best_by_symbol[sym] = {"target": trow, "expr": expr, "sort_key": sort_key}

    if not best_by_symbol:
        return None

    normal_colors = {
        "cta_restricted": "#2ca02c",
        "restricted_outside_lineage": "#2ca02c",
        "same_lineage_expected": "#1f77b4",
        "broad_healthy_expression": "#ff7f0e",
        "vital_tissue_concern": "#d62728",
    }
    source_markers = {
        "tumor_supported": "o",
        "mixed_source": "D",
        "background_dominant": "X",
    }

    def _short_source_label(source):
        return {
            "tumor_supported": "mostly tumor (RNA model)",
            "mixed_source": "mixed tumor/background (RNA model)",
            "background_dominant": "mostly background (RNA model)",
        }.get(source.get("tier"), str(source.get("label") or "unknown"))

    def _short_normal_label(normal):
        return {
            "cta_restricted": "CTA",
            "restricted_outside_lineage": "restricted",
            "same_lineage_expected": "lineage",
            "broad_healthy_expression": "broad",
            "vital_tissue_concern": "vital",
        }.get(normal.get("tier"), str(normal.get("label") or "unknown"))

    def _short_stage_label(target_row):
        phase = str(target_row.get("phase") or "").strip()
        if phase == "approved":
            return "approved"
        if phase.startswith("phase_"):
            return phase.replace("phase_", "phase ")
        if phase == "preclinical":
            return "preclin"
        return phase.replace("_", " ") or "curated"

    rows = []
    for sym, payload in best_by_symbol.items():
        trow = payload["target"]
        expr = payload["expr"]
        source = tumor_attribution_context(expr)
        normal = normal_expression_context(expr)
        rows.append(
            {
                "symbol": sym,
                "phase_key": phase_priority.get(str(trow.get("phase") or ""), 99),
                "mid": float(source["attr_tumor_tpm"]),
                "low": float(source["attr_tumor_tpm_low"]),
                "high": float(source["attr_tumor_tpm_high"]),
                "observed": float(expr.get("observed_tpm") or 0.0),
                "source": source,
                "normal": normal,
                "source_short": _short_source_label(source),
                "normal_short": _short_normal_label(normal),
                "maturity": _short_stage_label(trow),
                "color": normal_colors.get(normal["tier"], "#444444"),
                "marker": source_markers.get(source["tier"], "o"),
            }
        )

    rows.sort(
        key=lambda row: (
            row["phase_key"],
            {"tumor_supported": 0, "mixed_source": 1, "background_dominant": 2}.get(
                row["source"]["tier"], 9
            ),
            -row["mid"],
            row["symbol"],
        )
    )
    rows = rows[:top_n]
    if not rows:
        return None

    fig, (ax_expr, ax_ctx) = plt.subplots(
        1,
        2,
        figsize=(12.5, max(4.8, 0.52 * len(rows) + 1.8)),
        gridspec_kw={"width_ratios": [1.1, 0.9]},
    )
    y_pos = np.arange(len(rows))
    for i, row in enumerate(rows):
        ax_expr.hlines(
            i,
            row["low"],
            row["high"],
            color=row["color"],
            lw=5,
            alpha=0.78,
            zorder=2,
        )
        ax_expr.scatter(
            row["mid"],
            i,
            s=90,
            marker=row["marker"],
            color=row["color"],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax_expr.scatter(
            row["observed"],
            i,
            s=45,
            marker="|",
            color="black",
            linewidth=1.2,
            zorder=4,
        )

        ax_ctx.text(
            0.02,
            i,
            row["source_short"],
            va="center",
            fontsize=8.5,
            color="#222222",
        )
        ax_ctx.text(
            0.82,
            i,
            row["normal_short"],
            va="center",
            fontsize=8.5,
            color="#222222",
        )
        ax_ctx.text(
            1.62,
            i,
            row["maturity"],
            va="center",
            fontsize=8.5,
            color="#222222",
        )

    ax_expr.set_yticks(y_pos)
    ax_expr.set_yticklabels([row["symbol"] for row in rows], fontsize=9.5)
    ax_expr.invert_yaxis()
    ax_expr.set_xscale("symlog", linthresh=1.0)
    ax_expr.set_xlabel(
        "Estimated tumor TPM range (RNA model); black tick = patient bulk TPM (measured)"
    )
    ax_expr.set_title("Expression Evidence", fontsize=12, fontweight="bold")
    ax_expr.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.7)
    ax_expr.spines["top"].set_visible(False)
    ax_expr.spines["right"].set_visible(False)

    ax_ctx.set_xlim(0.0, 2.45)
    ax_ctx.set_ylim(-0.5, len(rows) - 0.5)
    ax_ctx.invert_yaxis()
    ax_ctx.axis("off")
    for xpos, label in [
        (0.02, "Estimated source"),
        (0.82, "Healthy ref."),
        (1.62, "Stage"),
    ]:
        ax_ctx.text(
            xpos,
            -0.85,
            label,
            fontsize=9.5,
            fontweight="bold",
            color="#333333",
            clip_on=False,
        )
    ax_ctx.set_title("Context", fontsize=12, fontweight="bold")

    range_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            lw=5,
            label="estimated tumor range (RNA model)",
        ),
        Line2D(
            [0],
            [0],
            marker="|",
            color="black",
            linestyle="none",
            markersize=12,
            markeredgewidth=1.5,
            label="patient bulk TPM (measured)",
        ),
    ]
    fig.legend(
        handles=range_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.015),
        fontsize=8,
        frameon=False,
        ncol=2,
    )
    if df_gene_expr is not None:
        try:
            from .common import build_sample_tpm_by_symbol
            from .plot_reference_lines import add_p90_reference_line

            add_p90_reference_line(
                ax_expr,
                build_sample_tpm_by_symbol(df_gene_expr),
                orientation="vertical",
            )
        except Exception:
            pass
    fig.suptitle(f"Curated Targets — {cancer_code}", fontsize=13, y=0.985)
    fig.text(
        0.5,
        0.955,
        "Compact labels: source=tumor/mixed/background; normal=CTA/restricted/lineage/broad/vital.",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.91])

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    return fig


def _priority_target_rows(
    ranges_df,
    cancer_type,
    target_panel=None,
    df_gene_expr=None,
    top_n=12,
    target_symbols=None,
    split_by_status=True,
    analysis=None,
    disease_state=None,
):
    """Build the shared ranked target rows used by the priority plots."""
    if ranges_df is None or len(ranges_df) == 0 or "symbol" not in ranges_df.columns:
        return None, []

    from .plot_therapy import _collect_ranked_therapy_targets

    try:
        cancer_code = resolve_cancer_type(cancer_type)
    except Exception:
        cancer_code = str(cancer_type)

    curated_by_symbol = {}
    history_excluded_symbols = set()
    if (
        target_panel is not None
        and len(target_panel)
        and "symbol" in target_panel.columns
    ):
        for _, trow in target_panel.iterrows():
            sym = str(trow.get("symbol") or "").strip()
            if not sym or sym.lower() == "nan":
                continue
            if treatment_history_blocks_row(trow, analysis) or treatment_history_marks_current(
                trow, analysis
            ):
                history_excluded_symbols.add(sym)
                continue
            sort_key = (
                treatment_history_rank(trow, analysis),
                _PRIORITY_PHASE_PRIORITY.get(str(trow.get("phase") or ""), 99),
                str(trow.get("agent") or ""),
            )
            if (
                sym not in curated_by_symbol
                or sort_key < curated_by_symbol[sym]["sort_key"]
            ):
                curated_by_symbol[sym] = {"target": trow, "sort_key": sort_key}
    history_excluded_symbols.difference_update(curated_by_symbol)

    generic_by_symbol = {}
    if df_gene_expr is not None:
        try:
            generic_records = _collect_ranked_therapy_targets(
                df_gene_expr,
                top_k=max(int(top_n) * 4, 30),
                tpm_threshold=1.0,
            )
        except Exception:
            generic_records = []
        for record in generic_records:
            sym = str(record.get("symbol") or "").strip()
            if sym:
                generic_by_symbol[sym] = record
    target_symbol_set = None
    if target_symbols is not None:
        target_symbol_set = {
            str(sym).strip()
            for sym in target_symbols
            if str(sym).strip() and str(sym).strip().lower() != "nan"
        }

    def _therapy_list(row):
        raw = str(row.get("therapies") or "").strip()
        if not raw:
            return []
        return [piece.strip() for piece in raw.split(",") if piece.strip()]

    def _normal_component(normal):
        base = {
            "cta_restricted": 2.2,
            "restricted_outside_lineage": 2.0,
            "same_lineage_expected": 1.7,
            "broad_healthy_expression": 0.9,
            "vital_tissue_concern": 0.6,
        }.get(normal["tier"], 1.0)
        for detail in normal.get("details") or []:
            text = str(detail)
            if "broader healthy-tissue signal" in text:
                base -= 0.45
            elif "expression is appreciable" in text:
                base -= 0.20
        return max(0.25, base)

    def _source_component(source, row):
        score = {
            "tumor_supported": 4.0,
            "mixed_source": 2.6,
            "background_dominant": 0.5,
        }.get(source["tier"], 1.0)
        score += 1.4 * float(source.get("attr_support_fraction") or 0.0)
        score += 1.1 * min(
            1.0, float(source.get("attr_tumor_fraction_low") or 0.0) / 0.5
        )
        if bool(row.get("matched_normal_over_predicted")):
            score -= 0.8
        if bool(row.get("tme_dominant")):
            score -= 1.0
        return max(0.1, score)

    def _strength_component(source, row):
        tcga_percentile = 0.0
        try:
            tcga_percentile = float(row.get("tcga_percentile") or 0.0)
        except Exception:
            tcga_percentile = 0.0
        tcga_percentile = min(1.0, max(0.0, tcga_percentile))
        return min(
            3.0,
            0.9 * np.log10(float(source["attr_tumor_tpm"]) + 1.0)
            + 0.8 * tcga_percentile,
        )

    def _status_for_target(curated, generic):
        if curated is not None and treatment_history_supports_review(curated, analysis):
            key = "patient_treatment_evidence"
        elif curated is not None:
            phase = str(curated.get("phase") or "").strip()
            if phase == "approved":
                key = "approved_disease_matched"
            else:
                key = "clinical_disease_matched"
        elif generic is not None and generic.get("has_approved"):
            key = "approved_other_context"
        else:
            key = "exploratory_or_expression_linked"
        return key, _PRIORITY_STATUS_ORDER[key], _PRIORITY_STATUS_LABELS[key]

    def _actionability_components(sym, row):
        curated = curated_by_symbol.get(sym, {}).get("target")
        generic = generic_by_symbol.get(sym)
        therapies = _therapy_list(row)
        status_key, status_rank, status_label = _status_for_target(curated, generic)
        if curated is not None:
            maturity = clinical_maturity_info(curated, target_panel=target_panel)
            phase = str(curated.get("phase") or "")
            points = _PRIORITY_PHASE_POINTS.get(phase, 0.8)
            points += min(0.4, 0.15 * max(0, maturity.get("n_modalities", 0) - 1))
            points += min(0.3, 0.05 * max(0, maturity.get("n_agents", 0) - 1))
            label = maturity["summary"]
            matched_panel = True
        elif generic is not None and generic.get("has_approved"):
            approved_therapies = generic.get("approved_therapies") or ()
            points = 1.9 + min(0.4, 0.15 * max(0, len(approved_therapies) - 1))
            label = (
                f"generic approved {str(generic.get('approved_label') or '').lower()}"
            )
            matched_panel = False
        else:
            points = 1.1 + min(0.4, 0.15 * max(0, len(therapies) - 1))
            label = (
                f"generic {' + '.join(therapies).lower()}"
                if therapies
                else "generic therapy-linked"
            )
            matched_panel = False
        return points, label.strip(), matched_panel, status_key, status_rank, status_label

    def _eligibility_component(curated):
        """Score orthogonal eligibility and therapy-state evidence.

        Expression is only one piece of the actionability chain. HLA-restricted,
        variant-required, or currently-modulated therapy rows should carry
        that uncertainty into the rank instead of being promoted by TPM alone.
        """
        if curated is None:
            return 0.0, "generic expression-linked"

        score = 0.35
        notes = []

        if treatment_history_supports_review(curated, analysis):
            score += 1.35
            notes.append("patient treatment benefit supplied")

        hla = target_hla_eligibility(curated, analysis=analysis)
        hla_status = hla.get("status")
        if hla_status == "matched":
            score += 0.85
            notes.append("HLA matched")
        elif hla_status == "insufficient_resolution":
            score -= 0.20
            notes.append("HLA low-resolution")
        elif hla_status == "unknown":
            score -= 0.35
            notes.append("HLA needed")
        elif hla_status == "mismatched":
            score -= 3.0
            notes.append("HLA mismatch")

        if expression_independent_indication(curated):
            supported = supplied_variant_supports_target_row(curated, analysis)
            if supported:
                score += 1.0
                notes.append("required variant supplied")
            else:
                label = indication_biomarker_label(curated)
                score -= 1.0
                notes.append(f"confirm {label}")
        else:
            score += 0.25

        caution = therapy_state_caution(
            curated, analysis=analysis, disease_state=disease_state
        )
        if caution:
            score -= 0.55
            notes.append("current-therapy state caveat")

        history_context = treatment_history_context(curated, analysis)
        if history_context and not treatment_history_supports_review(curated, analysis):
            notes.append("treatment history requires review")

        return max(0.0, min(1.7, score)), "; ".join(notes)

    def _tradeoff_component(curated):
        """Optional survival-benefit/toxicity curation hook.

        Most current rows do not yet carry structured outcome fields. When a
        curated table does provide them, fold them into the rank explicitly
        rather than deriving clinical benefit from expression, phase, or target
        class alone.
        """
        if curated is None or not hasattr(curated, "get"):
            return 0.0, ""
        benefit = str(curated.get("benefit_tier") or "").strip().lower()
        toxicity = str(curated.get("toxicity_tier") or "").strip().lower()
        benefit_points = {
            "curative": 2.0,
            "durable_rfs": 1.8,
            "durable_remission": 1.8,
            "major_survival": 1.5,
            "high_response": 1.3,
            "meaningful_pfs": 1.0,
            "incremental": 0.55,
            "modest": 0.45,
        }.get(benefit, 0.0)
        toxicity_penalty = {
            "minimal": 0.25,
            "low": 0.15,
            "moderate": 0.0,
            "high": -0.35,
            "very_high": -0.70,
        }.get(toxicity, 0.0)
        score = max(0.0, benefit_points + toxicity_penalty)
        labels = []
        if benefit:
            labels.append(f"benefit={benefit.replace('_', ' ')}")
        if toxicity:
            labels.append(f"toxicity={toxicity.replace('_', ' ')}")
        return min(2.0, score), "; ".join(labels)

    rows = []
    for _, row in ranges_df.iterrows():
        sym = str(row.get("symbol") or "").strip()
        if not sym or sym.lower() == "nan":
            continue
        if target_symbol_set is not None and sym not in target_symbol_set:
            continue
        if sym in history_excluded_symbols:
            continue
        therapies = _therapy_list(row)
        curated = curated_by_symbol.get(sym, {}).get("target")
        generic = generic_by_symbol.get(sym)
        if curated is None and not therapies and generic is None:
            continue
        try:
            observed = float(row.get("observed_tpm") or 0.0)
        except Exception:
            observed = 0.0
        if observed < 1.0:
            continue
        if row.get("therapy_supported") is False and curated is None:
            continue
        if (
            curated is not None
            and target_hla_eligibility(curated, analysis=analysis).get("status")
            == "mismatched"
        ):
            continue

        source = tumor_attribution_context(row)
        normal = normal_expression_context(row)
        (
            actionability_points,
            actionability_label,
            matched_panel,
            status_key,
            status_rank,
            status_label,
        ) = _actionability_components(sym, row)
        curated = curated_by_symbol.get(sym, {}).get("target")
        gate_points, gate_label = _eligibility_component(curated)
        tradeoff_points, tradeoff_label = _tradeoff_component(curated)
        source_points = _source_component(source, row)
        normal_points = _normal_component(normal)
        strength_points = _strength_component(source, row)
        curation_points = 0.8 if matched_panel else 0.0
        rows.append(
            {
                "symbol": sym,
                "observed": observed,
                "low": float(source["attr_tumor_tpm_low"]),
                "mid": float(source["attr_tumor_tpm"]),
                "high": float(source["attr_tumor_tpm_high"]),
                "source": source,
                "normal": normal,
                "matched_panel": matched_panel,
                "status_key": status_key,
                "status_rank": status_rank,
                "status_label": status_label,
                "clinical_label": actionability_label,
                "source_points": source_points,
                "actionability_points": actionability_points,
                "gate_points": gate_points,
                "gate_label": gate_label,
                "tradeoff_points": tradeoff_points,
                "tradeoff_label": tradeoff_label,
                "normal_points": normal_points,
                "strength_points": strength_points,
                "curation_points": curation_points,
                "total_score": (
                    source_points
                    + actionability_points
                    + gate_points
                    + tradeoff_points
                    + normal_points
                    + strength_points
                    + curation_points
                ),
                "phase_key": _PRIORITY_PHASE_PRIORITY.get(
                    str(curated.get("phase") or "") if curated is not None else "",
                    99 if curated is None else 9,
                ),
                "source_key": _PRIORITY_SOURCE_RANK.get(source["tier"], 9),
                "color": _PRIORITY_NORMAL_COLORS.get(normal["tier"], "#444444"),
                "marker": _PRIORITY_SOURCE_MARKERS.get(source["tier"], "o"),
            }
        )

    rows.sort(
        key=lambda row: (
            row["status_rank"] if split_by_status else 0,
            -row["total_score"],
            row["source_key"],
            row["phase_key"],
            -row["mid"],
            row["symbol"],
        )
    )
    return cancer_code, rows[:top_n]


def _priority_group_positions(rows):
    positions = []
    group_headers = []
    y = 0.0
    last_status = None
    for row in rows:
        status = row.get("status_label") or "Targets"
        if status != last_status:
            if positions:
                y += 0.95
            group_headers.append((y - 0.70, status))
            last_status = status
        positions.append(y)
        y += 1.0
    return np.asarray(positions, dtype=float), group_headers


def _draw_priority_group_headers(ax, group_headers):
    xmin, xmax = ax.get_xlim()
    for y, label in group_headers:
        ax.axhline(y + 0.18, color="#e4e4e4", linewidth=0.8, zorder=0)
        ax.text(
            -0.02,
            y,
            label,
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#555555",
            clip_on=False,
            transform=ax.get_yaxis_transform(),
        )
    ax.set_xlim(xmin, xmax)


def plot_priority_targets(
    ranges_df,
    cancer_type,
    target_panel=None,
    df_gene_expr=None,
    top_n=12,
    target_symbols=None,
    analysis=None,
    disease_state=None,
    save_to_filename=None,
    save_dpi=300,
):
    """Compact ranking plot for the integrated priority score."""
    cancer_code, rows = _priority_target_rows(
        ranges_df,
        cancer_type,
        target_panel=target_panel,
        df_gene_expr=df_gene_expr,
        top_n=top_n,
        target_symbols=target_symbols,
        analysis=analysis,
        disease_state=disease_state,
    )
    if not rows:
        return None

    y_pos, group_headers = _priority_group_positions(rows)
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.55 * max(y_pos) + 2.1)))
    score_parts = [
        ("Tumor support", "source_points", "#2e8b57"),
        ("Clinical readiness", "actionability_points", "#4a90d9"),
        ("Eligibility / state fit", "gate_points", "#d35400"),
        ("Disease-matched curation", "curation_points", "#f39c12"),
        ("Healthy-tissue specificity", "normal_points", "#9b59b6"),
        ("Tumor level", "strength_points", "#8c8c8c"),
    ]
    if any(float(row.get("tradeoff_points") or 0.0) > 0 for row in rows):
        score_parts.insert(
            4,
            ("Benefit/toxicity trade-off", "tradeoff_points", "#16a085"),
        )
    left = np.zeros(len(rows))
    for label, key, color in score_parts:
        values = [row[key] for row in rows]
        ax.barh(
            y_pos,
            values,
            left=left,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            height=0.72,
            label=label,
        )
        left += np.array(values)

    # §2.5: fold the tumor-source-vs-safety-band cue into this (now single) reader
    # target figure so dropping the near-duplicate context/actionable dumbbells does
    # not lose the source/safety signal. One marker per target, left of its bar:
    # shape = tumor-source tier (o supported / D mixed / X background), fill =
    # healthy-tissue safety tier — reusing the same row["marker"]/row["color"] the
    # audit-only context plot renders.
    cue_x = -0.62
    for i, row in enumerate(rows):
        ax.text(
            row["total_score"] + 0.14,
            y_pos[i],
            f"{row['total_score']:.1f}",
            va="center",
            fontsize=9,
            color="#444444",
        )
        ax.scatter(
            cue_x,
            y_pos[i],
            marker=row["marker"],
            s=90,
            facecolor=row["color"],
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
            clip_on=False,
        )

    labels = [row["symbol"] for row in rows]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    _draw_priority_group_headers(ax, group_headers)
    ax.set_ylim(max(y_pos) + 0.7, -1.15)
    ax.set_xlim(
        cue_x - 0.45,
        max(row["total_score"] for row in rows) + 0.95,
    )
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Integrated priority score")
    ax.set_title(
        f"Target Priority Ranking — {cancer_code}",
        fontsize=13,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Keep the score legend outside the data rectangle. A lower-right in-axes
    # legend overlaps the final group whenever a report has only a few
    # exploratory targets, obscuring exactly the rows it is meant to explain.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        frameon=False,
    )
    fig.text(
        0.5,
        0.008,
        "Marker left of each target — shape = tumor source (o supported / D mixed / "
        "X background); fill = healthy-tissue safety "
        "(blue same-lineage · green restricted/CTA · orange broad · red vital-tissue).",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.97])

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    return fig


def plot_priority_target_context(
    ranges_df,
    cancer_type,
    target_panel=None,
    df_gene_expr=None,
    top_n=12,
    target_symbols=None,
    analysis=None,
    disease_state=None,
    save_to_filename=None,
    save_dpi=300,
):
    """Separate evidence plot for tumor range plus source / safety / maturity."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cancer_code, rows = _priority_target_rows(
        ranges_df,
        cancer_type,
        target_panel=target_panel,
        df_gene_expr=df_gene_expr,
        top_n=top_n,
        target_symbols=target_symbols,
        analysis=analysis,
        disease_state=disease_state,
    )
    if not rows:
        return None

    y_pos, group_headers = _priority_group_positions(rows)
    fig, ax_range = plt.subplots(
        figsize=(11.5, max(5.0, 0.52 * max(y_pos) + 2.6)),
    )
    labels = [row["symbol"] for row in rows]
    source_labels = {row["source"]["label"] for row in rows}
    normal_labels = {row["normal"]["label"] for row in rows}

    def _log_tpm(value):
        return np.log10(max(0.0, float(value)) + 1.0)

    max_raw = max(max(row["high"], row["observed"], row["mid"], 1.0) for row in rows)

    for i, row in enumerate(rows):
        y = y_pos[i]
        low = _log_tpm(row["low"])
        mid = _log_tpm(row["mid"])
        high = _log_tpm(row["high"])
        observed = _log_tpm(row["observed"])
        ax_range.hlines(y, low, high, color=row["color"], lw=5, alpha=0.8)
        ax_range.scatter(
            mid,
            y,
            s=105,
            marker=row["marker"],
            color=row["color"],
            edgecolor="black",
            linewidth=0.9,
            zorder=3,
        )
        ax_range.scatter(
            observed,
            y,
            s=70,
            marker="|",
            color="black",
            linewidth=1.4,
            zorder=4,
        )
        ax_range.text(
            _log_tpm(max_raw) + 0.05,
            y,
            f"{row['total_score']:.1f}",
            va="center",
            fontsize=8.5,
            color="#555555",
        )

    ax_range.set_yticks(y_pos)
    ax_range.set_yticklabels(labels, fontsize=10)
    _draw_priority_group_headers(ax_range, group_headers)
    raw_ticks = [0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
    raw_ticks = [tick for tick in raw_ticks if tick <= max_raw * 1.2]
    if raw_ticks[-1] < max_raw and max_raw > raw_ticks[-1] * 1.25:
        raw_ticks.append(float(np.ceil(max_raw)))
    ax_range.set_xticks([_log_tpm(tick) for tick in raw_ticks])
    ax_range.set_xticklabels([f"{tick:g}" for tick in raw_ticks])
    ax_range.set_xlim(left=0.0, right=_log_tpm(max_raw) + 0.35)
    ax_range.set_ylim(max(y_pos) + 0.7, -1.15)
    ax_range.set_xlabel(
        "Expression, log10(TPM+1): range/diamond = estimated tumor TPM "
        "(RNA model estimate); black tick = patient bulk TPM (measured); right number = priority score"
    )
    ax_range.set_title(
        "Estimated patient tumor attribution vs measured bulk expression",
        fontsize=12,
        fontweight="bold",
    )
    ax_range.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.7)
    ax_range.set_axisbelow(True)
    ax_range.spines["top"].set_visible(False)
    ax_range.spines["right"].set_visible(False)
    if df_gene_expr is not None:
        try:
            from .common import build_sample_tpm_by_symbol
            from .plot_reference_lines import add_p90_reference_line

            add_p90_reference_line(
                ax_range,
                build_sample_tpm_by_symbol(df_gene_expr),
                orientation="vertical",
                value_transform=_log_tpm,
            )
        except Exception:
            pass

    constant_notes = []
    if len(source_labels) == 1:
        constant_notes.append(
            f"estimated patient tumor source: {next(iter(source_labels))}"
        )
    if len(normal_labels) == 1:
        constant_notes.append(
            f"healthy-tissue reference context: {next(iter(normal_labels))}"
        )
    if constant_notes:
        fig.text(
            0.50,
            0.915,
            "All rows share " + "; ".join(constant_notes) + ".",
            ha="center",
            va="top",
            fontsize=8.5,
            color="#555555",
        )

    normal_handles = [
        Patch(facecolor=color, edgecolor="black", label=label)
        for label, color in [
            ("also expected in healthy tissue", _PRIORITY_NORMAL_COLORS["same_lineage_expected"]),
            (
                "restricted / CTA-like",
                _PRIORITY_NORMAL_COLORS["restricted_outside_lineage"],
            ),
            (
                "broad healthy expression",
                _PRIORITY_NORMAL_COLORS["broad_healthy_expression"],
            ),
            ("important healthy tissue expression", _PRIORITY_NORMAL_COLORS["vital_tissue_concern"]),
        ]
    ]
    source_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor=_PRIORITY_SOURCE_COLORS.get(label, "#dddddd"),
            markeredgecolor="black",
            linestyle="none",
            markersize=8,
            label=label.replace("_", "-"),
        )
        for label, marker in _PRIORITY_SOURCE_MARKERS.items()
    ]
    source_handles.append(
        Line2D(
            [0],
            [0],
            marker="|",
            color="black",
            markeredgewidth=1.8,
            linestyle="none",
            markersize=12,
            label="bulk sample TPM",
        )
    )
    normal_legend = fig.legend(
        handles=normal_handles,
        loc="lower center",
        bbox_to_anchor=(0.33, 0.015),
        fontsize=8,
        title="Healthy-tissue context",
        frameon=False,
        ncol=2,
    )
    fig.add_artist(normal_legend)
    fig.legend(
        handles=source_handles,
        loc="lower center",
        bbox_to_anchor=(0.76, 0.015),
        fontsize=8,
        title="Marker / tick meaning",
        frameon=False,
        ncol=2,
    )

    fig.suptitle(
        f"Target Expression and Priority Score — {cancer_code}",
        fontsize=13,
        y=0.985,
    )
    fig.text(
        0.5,
        0.955,
        "Rows are split by approval/readiness tier; colors show external healthy-tissue reference context, marker shapes show estimated patient tumor support, and scores include HLA/variant/current-therapy fit plus curated benefit/toxicity when available.",
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=[0.06, 0.13, 1.0, 0.89])

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    return fig


# ── Plot 3: CTA deep dive ───────────────────────────────────────────────


def plot_cta_deep_dive(
    df_gene_expr,
    cancer_type,
    purity_estimate=None,
    top_n=20,
    save_to_filename=None,
    save_dpi=300,
):
    """Same layout as plot_actionable_targets but for CTAs.

    CTAs are cancer-testis antigens — normally restricted to testis/placenta,
    so any expression in somatic tissue is tumor-specific. The safety context
    (vital tissues) is especially important here because true CTAs should
    have near-zero expression in all vital tissues.
    """
    cancer_code = resolve_cancer_type(cancer_type)
    cta_map = CTA_gene_id_to_name()
    cta_symbols = sorted(cta_map.values())

    sample_tpm = _get_sample_tpm_by_symbol(df_gene_expr)
    ref_ctx = _build_reference_context(cta_symbols, cancer_code)
    tme_ref = _get_tme_reference(cta_symbols, cancer_code)

    cta_symbols = [g for g in cta_symbols if g in ref_ctx]

    rows = []
    for sym in cta_symbols:
        ctx = ref_ctx[sym]
        obs = sample_tpm.get(sym, 0.0)
        tme = tme_ref.get(sym, 0.0)
        tumor_adj = (
            _estimate_tumor_tpm(obs, tme, purity_estimate) if purity_estimate else None
        )
        max_vital = max(ctx["vital_tissues"].values()) if ctx["vital_tissues"] else 0.0
        rows.append(
            {
                "symbol": sym,
                "observed": obs,
                "tumor_adjusted": tumor_adj,
                "cancer_median": ctx["cancer_tpm"],
                "max_vital_tissue": max_vital,
                "tme_background": tme,
            }
        )

    rows.sort(key=lambda r: r["observed"], reverse=True)
    rows = rows[:top_n]
    if not rows:
        return None

    n = len(rows)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * n)))
    y_pos = np.arange(n)
    symbols = [r["symbol"] for r in rows]

    ax.barh(
        y_pos,
        [r["cancer_median"] for r in rows],
        height=0.35,
        color="#27AE60",
        alpha=0.3,
        label=f"TCGA {cancer_code} median",
    )
    ax.barh(
        y_pos + 0.35,
        [r["max_vital_tissue"] for r in rows],
        height=0.25,
        color="#999999",
        alpha=0.3,
        label="Max vital tissue (safety)",
    )

    ax.scatter(
        [r["observed"] for r in rows],
        y_pos,
        color="black",
        s=60,
        zorder=5,
        label="Sample observed TPM",
    )
    if purity_estimate:
        tumor_vals = [r["tumor_adjusted"] for r in rows]
        ax.scatter(
            tumor_vals,
            y_pos,
            color="#E74C3C",
            s=60,
            marker="D",
            zorder=5,
            label=f"Tumor-adjusted (purity={purity_estimate:.0%})",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(symbols, fontsize=9)
    ax.set_xlabel("TPM (log scale)")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"Cancer-Testis Antigens — {cancer_code}")

    try:
        from .plot_reference_lines import add_p90_reference_line
        from .common import build_sample_tpm_by_symbol

        add_p90_reference_line(
            ax,
            build_sample_tpm_by_symbol(df_gene_expr),
            orientation="vertical",
        )
    except Exception:
        pass

    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    return fig
