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

import seaborn as sns
import matplotlib.pyplot as plt

from adjustText import adjust_text

from .clean_tpm import assert_clean_tpm
from .common import _guess_gene_cols
from .plot_data_helpers import prepare_gene_expr_df
from pirlygenes.gene_ids import find_canonical_gene_ids_and_names
from pirlygenes.gene_sets_cancer import (
    housekeeping_gene_ids, CTA_gene_id_to_name, cancer_surfaceome_gene_id_to_name,
)
from pirlygenes.load_dataset import get_data


def _load_gene_sets():
    """Load immune gene sets as {category: {ensembl_id: symbol}} from CSV."""
    df = get_data("gene-sets")
    result = {}
    for cat, group in df.groupby("Category"):
        result[cat] = dict(zip(group["Ensembl_Gene_ID"], group["Symbol"]))
    return result


# ------------------------ helpers ------------------------


def pick_genes_to_annotate(
    df,
    num_per_category=10,
    verbose=False,
    category_col="category",
    tpm_col="TPM",
    gene_id_col="gene_id",
    gene_name_col="gene_display_name",
):
    """
    Returns a set of GENE IDs to annotate (top N by TPM per category).
    Note: we store IDs for stable matching; labels come from df['gene_display_name'] later.
    """
    genes_to_annotate = set()

    for c in [gene_id_col, gene_name_col, tpm_col, category_col]:
        assert c in df.columns, "%s not in %s" % (c, df.columns)

    for cat, df_cat in df.groupby(category_col, observed=True):
        df_cat_sorted = df_cat[df_cat[tpm_col] > 0.1].sort_values(
            tpm_col, ascending=False
        )
        top = df_cat_sorted.head(num_per_category)
        if verbose:
            cols = [gene_id_col, gene_name_col, tpm_col, category_col]
            print(cat)
            print(top[cols])
        genes_to_annotate.update(top[gene_id_col].tolist())
    return genes_to_annotate


def _normalize_label_token(token):
    return str(token).strip().upper()


def resolve_always_label_gene_ids(
    df,
    always_label_genes,
    gene_id_col="gene_id",
    gene_name_col="gene_display_name",
):
    if not always_label_genes:
        return set()
    wanted = {_normalize_label_token(tok) for tok in always_label_genes}
    forced = set()
    sub = df[[gene_id_col, gene_name_col]].drop_duplicates()
    gids = sub[gene_id_col].astype(str)
    names = sub[gene_name_col].astype(str)
    norm_gids = gids.map(_normalize_label_token)
    norm_names = names.map(_normalize_label_token)
    forced.update(gids[norm_gids.isin(wanted)])
    forced.update(gids[norm_names.isin(wanted) & ~norm_gids.isin(wanted)])

    # Also resolve user-provided gene names/aliases to canonical Ensembl IDs.
    resolved_ids, _ = find_canonical_gene_ids_and_names(list(always_label_genes))
    expr_gene_ids = set(df[gene_id_col].astype(str))
    for gid in resolved_ids:
        if gid:
            gid_str = str(gid).split(".", 1)[0]
            if gid_str in expr_gene_ids:
                forced.add(gid_str)
    return forced


# ------------------------ defaults ------------------------


def _build_default_gene_sets():
    """Build default gene sets using pre-resolved Ensembl IDs (no pyensembl lookup)."""
    sets = (
        _load_gene_sets()
    )  # APM, MHC1, TLR, Growth_receptors, Oncogenes, Immune_checkpoints
    sets["CTAs"] = CTA_gene_id_to_name()
    sets["Cancer_surfaceome"] = cancer_surfaceome_gene_id_to_name()
    return sets


default_gene_sets = _build_default_gene_sets()

# ------------------------ main plot ------------------------


def plot_gene_expression(
    df_gene_expr,
    gene_sets=default_gene_sets,
    save_to_filename=None,
    save_dpi=300,
    plot_height=14.0,
    plot_aspect=1.6,
    num_labels_per_category=5,
    always_label_genes=None,
    adjust_args=dict(
        expand=(1.4, 2.0),
        arrowprops=dict(arrowstyle="->", color="red", alpha=0.3),
        min_arrow_len=7,
        expand_axes=True,
        ensure_inside_axes=False,
    ),
    verbose=True,
    source_file=None,
):
    # Pick the correct ID/name columns from the incoming DF
    gene_id_col, gene_name_col = _guess_gene_cols(df_gene_expr)
    assert_clean_tpm(
        df_gene_expr,
        value_cols=["TPM"],
        label_col=gene_name_col,
        id_col=gene_id_col,
        context="strip plot sample expression",
    )

    # - join by IDs, label with names, include 'other' as a category and place it first.
    df_gene_expr_annot = prepare_gene_expr_df(
        df_gene_expr,
        gene_sets=gene_sets,
        gene_id_col=gene_id_col,
        gene_name_col=gene_name_col,  # ok if None; helper will resolve names
        other_category_name="other",  # <- ensure lowercase to match your filter
        place_other_first=True,  # <- 'other' left-most
        verbose=verbose,
        source_file=source_file,
    )

    # Just in case the "prepared" DF changed any column names, get them again
    gene_id_col, gene_name_col = _guess_gene_cols(df_gene_expr_annot)

    # Choose which IDs to annotate (top N per category)
    genes_to_annotate = pick_genes_to_annotate(
        df_gene_expr_annot,
        num_per_category=num_labels_per_category,
        gene_id_col=gene_id_col,
        gene_name_col=gene_name_col,
    )
    forced_gene_ids = resolve_always_label_gene_ids(
        df_gene_expr_annot,
        always_label_genes=always_label_genes,
        gene_id_col=gene_id_col,
        gene_name_col=gene_name_col,
    )
    genes_to_annotate.update(forced_gene_ids)

    # Sanity checks
    for col in ("category", "TPM", gene_id_col, gene_name_col):
        assert col in df_gene_expr_annot.columns, df_gene_expr_annot.columns

    # Plot — build palette with gray for "other", tab10 for named categories
    other_name = "other"
    cat_col = df_gene_expr_annot["category"]
    cat_order = (
        list(cat_col.cat.categories)
        if hasattr(cat_col, "cat") and hasattr(cat_col.cat, "categories")
        else list(dict.fromkeys(cat_col))
    )
    named_cats_strip = [c for c in cat_order if c != other_name]
    named_palette = sns.color_palette("tab10", len(named_cats_strip))
    palette_dict = {c: color for c, color in zip(named_cats_strip, named_palette)}
    palette_dict[other_name] = (0.8, 0.8, 0.8)  # gray for "other"

    if verbose:
        print(f"[plot] rendering strip plot ({len(df_gene_expr_annot)} points)...")
    cat = sns.catplot(
        data=df_gene_expr_annot,
        x="category",
        y="TPM",
        jitter=0.01,
        height=plot_height,
        aspect=plot_aspect,
        alpha=0.5,
        hue="category",
        palette=palette_dict,
    )
    plt.yscale("log")

    # Fade "other" category points so named categories stand out
    ax = cat.ax
    for coll in ax.collections:
        offsets = coll.get_offsets()
        if len(offsets) == 0:
            continue
        # Check if this collection corresponds to "other" by x-position
        cats = list(df_gene_expr_annot["category"].cat.categories)
        if other_name in cats:
            other_idx = cats.index(other_name)
            # PathCollection x-coords are jittered around integer category index
            x_coords = offsets[:, 0]
            is_other = (x_coords > other_idx - 0.5) & (x_coords < other_idx + 0.5)
            if is_other.any() and is_other.all():
                coll.set_alpha(0.12)

    # Highlight housekeeping genes in "other" with a distinct color and labels
    _hk_ids = housekeeping_gene_ids()
    hk_in_other = df_gene_expr_annot[
        (df_gene_expr_annot["category"] == other_name)
        & (df_gene_expr_annot[gene_id_col].isin(_hk_ids))
    ]
    if len(hk_in_other) > 0:
        cats = list(df_gene_expr_annot["category"].cat.categories)
        other_idx = cats.index(other_name) if other_name in cats else 0
        import numpy as _np

        hk_x = other_idx + _np.random.default_rng(42).uniform(
            -0.01, 0.01, len(hk_in_other)
        )
        ax.scatter(
            hk_x,
            hk_in_other["TPM"].values,
            color="#e74c3c",
            alpha=0.7,
            s=18,
            zorder=5,
            label="Housekeeping",
            edgecolors="none",
        )
        # Label top 5, bottom 5, and always ACTB — fed through adjust_text
        top5 = hk_in_other.nlargest(5, "TPM")
        bot5 = hk_in_other.nsmallest(5, "TPM")
        label_ids = set(top5[gene_id_col]) | set(bot5[gene_id_col])
        # Always include ACTB
        actb_rows = hk_in_other[hk_in_other[gene_name_col].str.upper() == "ACTB"]
        label_ids.update(actb_rows[gene_id_col])

        hk_texts = []
        for (_, row), xi in zip(hk_in_other.iterrows(), hk_x):
            if row[gene_id_col] in label_ids:
                hk_texts.append(
                    ax.text(
                        xi,
                        row["TPM"],
                        row[gene_name_col],
                        fontsize=7,
                        color="#c0392b",
                        alpha=0.85,
                        ha="right",
                        va="top",
                    )
                )
        adjust_text(
            hk_texts,
            force_text=(0.5, 1.0),
            force_points=(0.5, 1.0),
            **{**adjust_args, "expand": (1.4, 2.0)},
        )

    # Reference lines at key TPM thresholds
    for tpm_thresh in (100, 1000):
        ax.axhline(
            y=tpm_thresh,
            color="#cccccc",
            linestyle="--",
            linewidth=0.7,
            alpha=0.5,
            zorder=1,
        )

    # Annotate with display names, never raw ENSG; skip the 'other' column
    # Map categories to integer x-positions for adjust_text compatibility
    cat_col = df_gene_expr_annot["category"]
    if hasattr(cat_col, "cat"):
        cat_order = list(cat_col.cat.categories)
    else:
        cat_order = list(dict.fromkeys(cat_col))
    cat_to_x = {c: i for i, c in enumerate(cat_order)}

    texts = []
    point_x = []
    point_y = []
    forced_mask = df_gene_expr_annot[gene_id_col].isin(forced_gene_ids)
    mask = (
        ((df_gene_expr_annot.TPM > 0.1) | forced_mask)
        & (df_gene_expr_annot.category != "other")
        & (df_gene_expr_annot[gene_id_col].isin(genes_to_annotate))
    )
    for _, row in df_gene_expr_annot[mask].iterrows():
        # Use the friendly display name
        label = row[gene_name_col]
        texts.append(
            cat.ax.text(
                cat_to_x[row.category],
                row.TPM,
                label,
                fontsize=8,
                color="black",
                alpha=0.8,
                ha="right",
                va="top",
            )
        )
        point_x.append(cat_to_x[row.category])
        point_y.append(row.TPM)

    adjust_text(
        texts,
        x=point_x,
        y=point_y,
        ax=cat.ax,
        force_text=(0.5, 1.0),
        force_points=(0.5, 1.0),
        **adjust_args,
    )

    if save_to_filename:
        cat.figure.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")

    return cat
