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
that show per-gene fold-change vs cohort median. These bulk RNA contrasts
provide subtype context; pathology and clinical assays establish clinical states.

Generalised: any pair of therapy-response axes can be contrasted for any
cancer type where the signatures are applicable.
"""

from textwrap import fill

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
            "name": "AR and neuroendocrine RNA programs",
            "axis_a": "AR_signaling",
            "axis_b": "NE_differentiation",
            "interpretation": {
                "a_up_b_down": "AR-associated RNA is higher; neuroendocrine RNA is lower than the cohort reference.",
                "a_down_b_up": "AR-associated RNA is lower; neuroendocrine RNA is higher. Correlate with pathology to assess lineage.",
                "a_down_b_down": "Both RNA programs are lower. This does not establish prior androgen deprivation, resistance or retained lineage.",
                "a_up_b_up": "Both RNA programs are higher. A mixed RNA pattern does not establish treatment-emergent NEPC.",
            },
        },
    ],
    "BRCA": [
        {
            "name": "ER and HER2 RNA programs",
            "axis_a": "ER_signaling",
            "axis_b": "HER2_signaling",
            "interpretation": {
                "a_up_b_down": "ER-associated RNA is higher; HER2-associated RNA is lower. Confirm clinical ER/PR and HER2 assays.",
                "a_down_b_up": "HER2-associated RNA is higher; ER-associated RNA is lower. Confirm clinical ER/PR and HER2 assays.",
                "a_down_b_down": "Both RNA programs are lower. Triple-negative status requires clinical ER, PR and HER2 testing.",
                "a_up_b_up": "Both RNA programs are higher. Protein and amplification assays determine receptor eligibility.",
            },
        },
    ],
    "LUAD": [
        {
            "name": "EGFR/EMT",
            "axis_a": "MAPK_EGFR_signaling",
            "axis_b": "EMT",
            "interpretation": {
                "a_up_b_down": "EGFR-pathway RNA is higher; EMT RNA is lower. This does not establish an actionable EGFR alteration.",
                "a_down_b_up": "EMT RNA is higher; EGFR-pathway RNA is lower. This does not establish resistance or checkpoint eligibility.",
                "a_up_b_up": "Both RNA programs are higher. Confirm molecular findings and treatment course before inferring resistance.",
                "a_down_b_down": "Both RNA programs are lower. No alternative driver is established by this contrast.",
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

    # Display raw expression of every panel gene; the caption describes RNA,
    # not signed pathway activity or a clinical diagnosis.
    axis_a_all = axis_a_up + axis_a_down
    axis_b_all = axis_b_up + axis_b_down

    if not axis_a_all and not axis_b_all:
        return None

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, max(4.8, 0.4 * max(len(axis_a_all), len(axis_b_all))))
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
        ax.set_yticklabels(symbols, fontsize=12)
        ax.set_xlabel("log₂(sample / TCGA cohort median)")
        ax.set_title(axis_name.replace("_", " "), fontsize=13, fontweight="bold")

        # A separate in-panel column keeps TPM labels away from gene names
        # and negative bars, including values near the plotting boundary.
        low, high = min(log2_folds + [0]), max(log2_folds + [0])
        span = max(high - low, 1.0)
        ax.set_xlim(low - 0.04 * span, high + 0.35 * span)
        for i, gene in enumerate(genes):
            ax.text(0.98, i, f"{gene['sample_tpm']:.1f} TPM",
                    transform=ax.get_yaxis_transform(), va="center", ha="right",
                    fontsize=10, color="#555555")

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
            fill(call, width=110),
            ha="center",
            fontsize=11,
            style="italic",
            wrap=True,
            color="#333333",
        )

    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    return fig
