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

"""Cancer subtype / lineage-plasticity signature plots.

Visualises therapy-response axis signatures (AR_signaling, NE_differentiation,
ER_signaling, HER2_signaling, EMT, etc.) as heatmap-style or dot-strip plots
that show per-gene fold-change vs cohort median, making subtype transitions
(e.g. PRAD adenocarcinoma → NEPC) immediately visible.

Generalised: any pair of therapy-response axes can be contrasted for any
cancer type where the signatures are applicable.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trufflepig.reference import pan_cancer_expression
from .therapy_response import load_therapy_signatures, _sigs_for_cancer
from .plot_scatter import resolve_cancer_type


# ── Subtype signature contrasts ──────────────────────────────────────────
#
# Curated pairs of therapy-response axes whose relative state distinguishes
# clinically important subtypes.

SUBTYPE_CONTRASTS = {
    "PRAD": [
        {
            "name": "Adenocarcinoma vs NEPC",
            "axis_a": "AR_signaling",
            "axis_b": "NE_differentiation",
            "interpretation": {
                "a_up_b_down": "Classic adenocarcinoma — AR program active, no NE differentiation",
                "a_down_b_up": "Neuroendocrine prostate cancer (NEPC) — AR collapsed, NE markers elevated",
                "a_down_b_down": "AR-suppressed (post-ADT) without NE emergence — CRPC, adenocarcinoma lineage retained",
                "a_up_b_up": "Mixed phenotype — AR active with emerging NE features (treatment-emergent t-NEPC)",
            },
        },
    ],
    "BRCA": [
        {
            "name": "ER/PR status",
            "axis_a": "ER_signaling",
            "axis_b": "HER2_signaling",
            "interpretation": {
                "a_up_b_down": "ER+/HER2- (luminal A/B) — endocrine therapy candidate",
                "a_down_b_up": "ER-/HER2+ — trastuzumab / T-DXd candidate",
                "a_down_b_down": "Triple-negative pattern — checkpoint / TROP2 ADC candidates",
                "a_up_b_up": "ER+/HER2+ — dual-targeted therapy",
            },
        },
    ],
    "LUAD": [
        {
            "name": "EGFR/EMT",
            "axis_a": "MAPK_EGFR_signaling",
            "axis_b": "EMT",
            "interpretation": {
                "a_up_b_down": "EGFR-driven, epithelial — TKI candidate",
                "a_down_b_up": "Mesenchymal phenotype — TKI-resistant, checkpoint candidate",
                "a_up_b_up": "EGFR-active with mesenchymal features — emerging resistance",
                "a_down_b_down": "Neither EGFR-driven nor mesenchymal — other driver likely",
            },
        },
    ],
}


def _score_axis_genes(sample_tpm, ref_by_sym, cancer_code, axis_genes):
    """Return list of {symbol, sample_tpm, cohort_median, fold_change, direction}."""
    cancer_col = f"{cancer_code}_TPM"
    rows = []
    for rec in axis_genes:
        sym = rec["symbol"]
        obs = sample_tpm.get(sym, 0.0)
        cohort_med = (
            float(ref_by_sym.loc[sym, cancer_col])
            if sym in ref_by_sym.index and cancer_col in ref_by_sym.columns
            else 0.0
        )
        if cohort_med > 0.1:
            fold = obs / cohort_med
        elif obs > 0.1:
            fold = 10.0  # expressed in sample but not in cohort
        else:
            fold = 1.0
        rows.append(
            {
                "symbol": sym,
                "sample_tpm": obs,
                "cohort_median": cohort_med,
                "fold_change": fold,
                "log2_fold": float(np.log2(max(fold, 0.001))),
            }
        )
    return rows


def plot_subtype_signature(
    df_gene_expr,
    cancer_type,
    contrast_index=0,
    save_to_filename=None,
    save_dpi=300,
):
    """Two-panel dot plot showing a subtype contrast.

    Left panel: axis_a genes (fold vs cohort). Right panel: axis_b genes.
    Color encodes direction: blue = below cohort, red = above cohort.
    Interpretation text at the bottom.

    Returns None if the cancer type has no curated contrasts.
    """
    cancer_code = resolve_cancer_type(cancer_type)
    contrasts = SUBTYPE_CONTRASTS.get(cancer_code)
    if not contrasts or contrast_index >= len(contrasts):
        return None
    contrast = contrasts[contrast_index]

    all_sigs = load_therapy_signatures()
    cancer_sigs = _sigs_for_cancer(all_sigs, cancer_code)
    axis_a_name = contrast["axis_a"]
    axis_b_name = contrast["axis_b"]

    if axis_a_name not in cancer_sigs or axis_b_name not in cancer_sigs:
        return None

    # Get sample TPM
    from .sample_context import _build_tpm_by_symbol

    sample_tpm = _build_tpm_by_symbol(df_gene_expr)

    ref = (
        pan_cancer_expression(technical_rna_normalize=True)
        .drop_duplicates(subset="Symbol")
        .set_index("Symbol")
    )

    axis_a_up = _score_axis_genes(
        sample_tpm, ref, cancer_code, cancer_sigs[axis_a_name].get("up", [])
    )
    axis_a_down = _score_axis_genes(
        sample_tpm, ref, cancer_code, cancer_sigs[axis_a_name].get("down", [])
    )
    axis_b_up = _score_axis_genes(
        sample_tpm, ref, cancer_code, cancer_sigs[axis_b_name].get("up", [])
    )
    axis_b_down = _score_axis_genes(
        sample_tpm, ref, cancer_code, cancer_sigs[axis_b_name].get("down", [])
    )

    # Combine up + down for each axis, tag direction
    def _tagged(rows, tag):
        for r in rows:
            r["panel_role"] = tag
        return rows

    axis_a_all = _tagged(axis_a_up, "up") + _tagged(axis_a_down, "down")
    axis_b_all = _tagged(axis_b_up, "up") + _tagged(axis_b_down, "down")

    if not axis_a_all and not axis_b_all:
        return None

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, max(4, 0.4 * max(len(axis_a_all), len(axis_b_all))))
    )

    def _draw_axis_panel(ax, genes, axis_name):
        if not genes:
            ax.set_visible(False)
            return
        genes.sort(key=lambda g: g["log2_fold"])
        n = len(genes)
        y_pos = np.arange(n)
        symbols = [g["symbol"] for g in genes]
        log2_folds = [g["log2_fold"] for g in genes]
        colors = ["#E74C3C" if lf > 0 else "#3498DB" for lf in log2_folds]

        ax.barh(y_pos, log2_folds, color=colors, alpha=0.7, height=0.6)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(symbols, fontsize=9)
        ax.set_xlabel("log₂(sample / TCGA cohort median)")
        ax.set_title(axis_name.replace("_", " "), fontsize=11, fontweight="bold")

        # Annotate with TPM
        for i, g in enumerate(genes):
            label = (
                f"{g['sample_tpm']:.0f}"
                if g["sample_tpm"] >= 1
                else f"{g['sample_tpm']:.1f}"
            )
            side = "left" if g["log2_fold"] > 0 else "right"
            offset = 0.1 if g["log2_fold"] > 0 else -0.1
            ax.text(
                g["log2_fold"] + offset,
                i,
                f"{label} TPM",
                va="center",
                ha=side,
                fontsize=7,
                color="#555555",
            )

    _draw_axis_panel(ax1, axis_a_all, axis_a_name)
    _draw_axis_panel(ax2, axis_b_all, axis_b_name)

    # Determine which quadrant we're in
    a_median_fold = np.median([g["log2_fold"] for g in axis_a_all]) if axis_a_all else 0
    b_median_fold = np.median([g["log2_fold"] for g in axis_b_all]) if axis_b_all else 0
    interp = contrast["interpretation"]
    if a_median_fold > 0 and b_median_fold <= 0:
        call = interp.get("a_up_b_down", "")
    elif a_median_fold <= 0 and b_median_fold > 0:
        call = interp.get("a_down_b_up", "")
    elif a_median_fold <= 0 and b_median_fold <= 0:
        call = interp.get("a_down_b_down", "")
    else:
        call = interp.get("a_up_b_up", "")

    fig.suptitle(
        f"{contrast['name']} — {cancer_code}", fontsize=13, fontweight="bold", y=1.02
    )
    if call:
        fig.text(
            0.5,
            -0.02,
            call,
            ha="center",
            fontsize=10,
            style="italic",
            wrap=True,
            color="#333333",
        )

    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    return fig
