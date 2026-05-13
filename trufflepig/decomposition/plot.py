# Licensed under the Apache License, Version 2.0

"""Plot helpers for broad-compartment decomposition results."""

import numpy as np
import matplotlib.pyplot as plt


_COMPOSITION_PALETTE = [
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#d62728",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#7f7f7f",
]


def _render_composition_bar(ax, best, title="Sample composition (tumor + TME)"):
    """Horizontal stacked bar — tumor + TME components as fractions of sample."""
    raw_items = sorted(best.fractions.items(), key=lambda item: item[1], reverse=True)
    frac_items = [(name, value) for name, value in raw_items if value >= 0.005]
    minor_total = sum(value for _, value in raw_items if value < 0.005)
    if minor_total > 0:
        frac_items.append(("other_minor", minor_total))
    left = 0.0
    for idx, (name, value) in enumerate(frac_items):
        label_name = str(name).replace("_", " ")
        ax.barh(
            [0],
            [value * 100],
            left=left * 100,
            color=_COMPOSITION_PALETTE[idx % len(_COMPOSITION_PALETTE)],
            edgecolor="none",
            height=0.55,
            label=f"{label_name} ({value:.0%})",
        )
        left += value
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Estimated composition (%)")
    ax.set_title(title, fontweight="bold")
    ax.legend(
        bbox_to_anchor=(0.0, -0.25),
        loc="upper left",
        fontsize=8,
        ncol=3,
        framealpha=0.9,
        borderaxespad=0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def _render_component_breakdown(ax, best, title="TME cell-type breakdown"):
    """Horizontal bar per TME component — fraction (%) + marker-support annotation.

    Marker-support number (median observed/expected ratio across each
    component's marker genes) is shown unlabelled next to each bar;
    readers get a sense of how well each component's signal matches its
    reference without the chart being cluttered with 'marker=' prefixes.
    """
    comp_df = best.component_trace.copy()
    if comp_df.empty:
        ax.text(
            0.5,
            0.5,
            "No component trace",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return
    comp_df = comp_df.sort_values("fraction", ascending=True).reset_index(drop=True)
    y = np.arange(len(comp_df))
    ax.barh(y, comp_df["fraction"] * 100, color="#4c78a8", alpha=0.85, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(comp_df["component"], fontsize=9)
    ax.set_xlabel("Fraction of sample (%)")
    ax.set_title(title, fontweight="bold")
    for idx, row in comp_df.iterrows():
        if row["fraction"] < 0.005:
            continue  # skip labels for sub-0.5% components (#96)
        # A marker_score of 0.00 with n_markers=0 means the compartment
        # had no markers to evaluate (typical for matched-normal
        # compartments in mixture-cohort templates). Showing "0.00"
        # reads as a bad fit; "n/a" is more honest.
        n_markers = int(row.get("n_markers") or 0)
        if n_markers == 0 or row["marker_score"] is None:
            txt = "n/a"
        else:
            txt = f"{row['marker_score']:.2f}"
        ax.text(row["fraction"] * 100 + 0.8, idx, txt, va="center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_decomposition_composition(
    best,
    save_to_filename=None,
    save_dpi=300,
    title=None,
):
    """Standalone stacked-bar of tumor + TME composition for the best hypothesis.

    Same content as panel 2 of plot_decomposition_summary, rendered
    larger as its own figure for inclusion in slide decks or focused
    reports.
    """
    fig, ax = plt.subplots(figsize=(12, 3.5))
    _render_composition_bar(
        ax,
        best,
        title=title or f"Sample composition — {best.cancer_type} / {best.template}",
    )
    fig.subplots_adjust(bottom=0.35)
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig


def plot_decomposition_component_breakdown(
    best,
    save_to_filename=None,
    save_dpi=300,
    title=None,
):
    """Standalone horizontal-bar plot of per-component TME fractions.

    Same content as panel 3 of plot_decomposition_summary, rendered
    larger with a cleaner title and numeric marker-support annotations
    (no 'marker=' prefix).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    _render_component_breakdown(
        ax,
        best,
        title=title
        or f"TME cell-type breakdown — {best.cancer_type} / {best.template}",
    )
    fig.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig


# Palette for the 3-segment candidate composition bar. Chosen so the
# tumor / template-specific / shared-host triplet is intuitive:
# - blue for tumor signal (the thing the sample is)
# - green for the template's tissue-specific compartment (matched-normal
#   epithelium for a primary, target-tissue signature for a met) — the
#   compartment whose "fit" the ranking cares about
# - warm gray for the shared immune + stroma basis that every template
#   shares, so it visually recedes
_CANDIDATE_SEGMENT_COLORS = {
    "tumor": "#1f77b4",
    "template_specific": "#2ca02c",
    "shared_host": "#b3a18a",
}


def _candidate_composition_segments(row):
    """Split a decomposition candidate into (tumor, template_specific, shared_host) fractions of the sample.

    Each segment is a fraction in [0, 1]; the three sum to 1. Computed
    directly from the row fields:
        tumor            = purity
        template_specific = (1 - purity) * template_extra_fraction
        shared_host      = (1 - purity) * (1 - template_extra_fraction)

    Where `template_extra_fraction` is already the share of the non-tumor
    portion decomposed into the template's tissue-specific extra components
    (see decomposition/engine.py — `extra_fraction × (1 − tumor_fraction)`
    is `extra_sample_fraction`, but the stored field on the result is
    `template_extra_fraction` which is the pre-scaled ratio).
    """
    purity = float(row.purity or 0.0)
    extra_ratio = float(row.template_extra_fraction or 0.0)
    non_tumor = max(0.0, 1.0 - purity)
    template_specific = non_tumor * extra_ratio
    shared_host = max(0.0, non_tumor - template_specific)
    return purity, template_specific, shared_host


def plot_decomposition_candidates(
    results,
    top_candidates=6,
    save_to_filename=None,
    save_dpi=300,
    figsize=None,
    labels=None,
    title="Sample decomposition candidates",
):
    """Render one row per candidate as a 3-segment composition bar.

    Each bar occupies the full x-axis (0–100% of the sample) and is split
    into tumor / template-specific / shared-host segments, making the
    *structural* difference between candidates readable at a glance:

    - a `solid_primary` candidate's non-tumor fraction is dominated by
      matched-normal epithelium (large green segment)
    - a `met_liver` candidate in the same sample shifts the non-tumor
      fraction into the template's hepatocyte compartment if the signal
      is there, or collapses to shared immune/stroma if it isn't (tiny
      green, large gray)
    - an immune-only template (e.g. `met_lymph`) has no template-specific
      compartment at all — the row is just tumor + shared (no green)

    The combined decomposition `score` is shown as a text annotation per
    row rather than encoded in bar length, so composition differences
    remain directly comparable across candidates.

    Parameters
    ----------
    results : list of DecompositionResult
        Typically `engine.decompose_sample(...)`. Candidates are plotted
        in score-descending order; top entry sits at the top of the axis.
    top_candidates : int
        Maximum number of candidate rows to render.
    save_to_filename : str or None
        If given, save the figure to this path as a standalone PNG.
    save_dpi : int
        DPI used when saving.
    figsize : tuple or None
        Explicit figure size; defaults to (12, 0.55 * n_candidates + 1.5).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not results:
        return None

    rows = results[:top_candidates]
    n = len(rows)
    if figsize is None:
        figsize = (12, 0.55 * n + 1.8)

    fig, ax = plt.subplots(figsize=figsize)

    labels = (
        list(labels)
        if labels is not None
        else [f"{row.cancer_type} / {row.template}" for row in rows]
    )
    y = np.arange(n)

    tumor_vals = np.zeros(n)
    tmpl_vals = np.zeros(n)
    shared_vals = np.zeros(n)
    for i, row in enumerate(rows):
        tumor, tmpl, shared = _candidate_composition_segments(row)
        tumor_vals[i] = tumor
        tmpl_vals[i] = tmpl
        shared_vals[i] = shared

    # Stacked horizontal bars — widths in percent of sample for readability.
    ax.barh(
        y,
        tumor_vals * 100,
        color=_CANDIDATE_SEGMENT_COLORS["tumor"],
        edgecolor="white",
        linewidth=0.5,
        height=0.6,
        label="Tumor",
    )
    ax.barh(
        y,
        tmpl_vals * 100,
        left=tumor_vals * 100,
        color=_CANDIDATE_SEGMENT_COLORS["template_specific"],
        edgecolor="white",
        linewidth=0.5,
        height=0.6,
        label="Template-specific compartment",
    )
    ax.barh(
        y,
        shared_vals * 100,
        left=(tumor_vals + tmpl_vals) * 100,
        color=_CANDIDATE_SEGMENT_COLORS["shared_host"],
        edgecolor="white",
        linewidth=0.5,
        height=0.6,
        label="Shared immune / stroma",
    )

    # Inline percent labels inside each segment, only if the segment is
    # wide enough to hold a readable label.
    def _annotate(vals, offsets, fmt):
        for i, (v, off) in enumerate(zip(vals, offsets)):
            if v * 100 >= 7:
                ax.text(
                    off * 100 + v * 100 / 2,
                    i,
                    fmt(v),
                    va="center",
                    ha="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )

    _annotate(tumor_vals, np.zeros(n), lambda v: f"tumor {v:.0%}")
    _annotate(tmpl_vals, tumor_vals, lambda v: f"site {v:.0%}")
    _annotate(shared_vals, tumor_vals + tmpl_vals, lambda v: f"immune/stroma {v:.0%}")

    # Right-side per-row score text. Kept textual (not a second bar) so
    # it's not confused with a width-encoded quantity.
    #
    # #123: also surface **fit quality** (raw reconstruction_error —
    # smaller is better) and **median marker support** (how well-
    # supported each fitted compartment is by its own markers) so the
    # reader can tell whether a candidate is high-scoring because it
    # actually explains the sample or because the composite score
    # happens to land high. Candidates flagged with warnings by the
    # decomposition engine (purity floor, low marker support, template
    # incomplete, etc.) get a hatched pattern overlaid on their bars —
    # a visible signal that the row is gated, not just aggregated.
    import matplotlib.patches as mpatches

    for i, row in enumerate(rows):
        score = float(row.score or 0.0)
        cancer = float(row.cancer_support_score or 0.0)
        site = float(row.template_tissue_score or 0.0)
        err = float(getattr(row, "reconstruction_error", 0.0) or 0.0)

        # Median marker support across fitted compartments. Typical range
        # 0.0 (no marker evidence) to ~1.0 (each compartment solidly
        # supported by its markers). Falls back to 0.0 for rows without
        # a component_trace (e.g. the template-incomplete early return).
        try:
            trace = getattr(row, "component_trace", None)
            if (
                trace is not None
                and not trace.empty
                and "marker_score" in trace.columns
            ):
                ms = trace["marker_score"].astype(float)
                median_marker = float(np.nanmedian(ms)) if len(ms) else 0.0
            else:
                median_marker = 0.0
        except Exception:
            median_marker = 0.0

        ax.text(
            101,
            i - 0.18,
            f"score {score:.2f}  (cancer {cancer:.2f} · site {site:.2f})",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#333333",
        )
        ax.text(
            101,
            i + 0.20,
            f"fit err {err:.2f} · markers {median_marker:.2f}",
            va="center",
            ha="left",
            fontsize=8,
            color="#555555",
            style="italic",
        )

        # Hatched overlay when the engine flagged this row (purity-floor
        # penalty, low marker support, etc.).
        warnings_list = list(getattr(row, "warnings", None) or [])
        if warnings_list:
            ax.barh(
                [i],
                [100],
                left=[0],
                height=0.6,
                facecolor="none",
                edgecolor="#555555",
                hatch="///",
                linewidth=0.0,
                alpha=0.35,
                zorder=5,
            )

    # Legend artist for the hatched "gated" pattern — the stacked bars
    # already carry the three segment labels.
    any_warnings = any((getattr(r, "warnings", None) or []) for r in rows)
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    if any_warnings:
        gated_patch = mpatches.Patch(
            facecolor="none",
            edgecolor="#555555",
            hatch="///",
            label="gated (see row warnings)",
        )
        ax.legend(
            existing_handles + [gated_patch],
            existing_labels + ["gated (see row warnings)"],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            frameon=False,
            fontsize=9,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percent of sample")
    # Legend above the plot area so it never overlaps the last row's
    # segment labels — the title still sits above the legend. Legend
    # entries include the hatched "gated" pattern only when at least
    # one row has an engine-issued warning (so unaffected plots stay
    # uncluttered).
    if not any_warnings:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
            fontsize=9,
        )
    # Title kept single-line — the per-bar segments + legend already
    # explain that each row is one candidate's composition.
    ax.set_title(title, fontweight="bold", pad=28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig


def plot_decomposition_summary(
    results,
    call_summary=None,
    save_to_filename=None,
    save_dpi=300,
    top_hypotheses=6,
    top_markers_per_component=4,
):
    """Render a human-interpretable summary of decomposition logic."""
    if not results:
        return None

    best = results[0]
    hypotheses = results[:top_hypotheses]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 12),
        gridspec_kw={"width_ratios": [1.3, 1.0], "height_ratios": [1.0, 1.0]},
    )
    ax_hyp, ax_frac = axes[0]
    ax_comp, ax_mark = axes[1]

    # --- Panel 1: ranked hypotheses ---
    labels = [f"{row.cancer_type} / {row.template}" for row in hypotheses]
    scores = [row.score for row in hypotheses]
    colors = ["#1f77b4" if i == 0 else "#9ecae1" for i in range(len(hypotheses))]
    y = np.arange(len(hypotheses))
    ax_hyp.barh(y, scores, color=colors, height=0.65)
    ax_hyp.set_yticks(y)
    ax_hyp.set_yticklabels(labels, fontsize=9)
    ax_hyp.invert_yaxis()
    ax_hyp.set_xlabel("Combined decomposition score")
    ax_hyp.set_title("Hypotheses", fontweight="bold")
    for idx, row in enumerate(hypotheses):
        ax_hyp.text(
            scores[idx] + 0.005,
            idx,
            (
                f"purity={row.purity:.0%}, cancer={row.cancer_support_score:.2f}, "
                f"site={row.template_tissue_score:.2f}, factor={row.template_site_factor:.2f}, "
                f"extra={row.template_extra_fraction:.0%}"
            ),
            va="center",
            fontsize=8,
        )
    ax_hyp.spines["top"].set_visible(False)
    ax_hyp.spines["right"].set_visible(False)
    if call_summary:
        lines = []
        if call_summary.get("label_options"):
            if len(call_summary["label_options"]) == 2:
                lines.append(
                    f"Possible labels: {call_summary['label_options'][0]} or {call_summary['label_options'][1]}"
                )
            else:
                lines.append(f"Resolved label: {call_summary['label_options'][0]}")
        if call_summary.get("site_indeterminate"):
            lines.append("Site/template: indeterminate")
        elif call_summary.get("reported_site"):
            lines.append(f"Site/template: {call_summary['reported_site']}")
        if lines:
            ax_hyp.text(
                0.02,
                0.02,
                "\n".join(lines),
                transform=ax_hyp.transAxes,
                fontsize=8.5,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
            )

    # --- Panel 2: final composition ---
    _render_composition_bar(ax_frac, best, title="Sample composition")

    # --- Panel 3: TME component breakdown ---
    _render_component_breakdown(ax_comp, best, title="TME cell-type breakdown")

    # --- Panel 4: marker trace text ---
    ax_mark.set_title("Marker logic", fontweight="bold")
    ax_mark.axis("off")
    if best.marker_trace.empty:
        ax_mark.text(0.02, 0.98, "No marker trace", ha="left", va="top", fontsize=10)
    else:
        lines = []
        for component in best.component_trace["component"].tolist()[
            : min(6, len(best.component_trace))
        ]:
            sub = best.marker_trace[best.marker_trace["component"] == component].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(
                ["observed_tpm", "specificity"], ascending=[False, False]
            ).head(top_markers_per_component)
            entries = [
                f"{row.symbol}={row.observed_tpm:.1f} TPM" for row in sub.itertuples()
            ]
            lines.append(f"{component}: " + ", ".join(entries))

        if best.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend([f"- {warning}" for warning in best.warnings[:4]])

        ax_mark.text(
            0.02,
            0.98,
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
        )

    title = f"Broad-compartment decomposition — {best.cancer_type} / {best.template}"
    if call_summary and call_summary.get("site_indeterminate"):
        title = f"Broad-compartment decomposition — {call_summary.get('label_display', best.cancer_type)} (site indeterminate)"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig
