# Licensed under the Apache License, Version 2.0

"""Shared reference-line helpers for sample-TPM plots.

Every plot that renders per-gene sample TPM values benefits from a
consistent visual anchor for "where does this gene's expression sit
relative to the rest of the sample?". A single faint dashed line at
the sample's 90th-percentile TPM gives that anchor cheaply — readers
can immediately see which genes are in the top decile of the whole
transcriptome.

The percentile is computed from the expressed portion of the sample
(TPM > 0) rather than the full gene universe, so near-zero gene
noise doesn't pull the reference down toward zero.
"""

from __future__ import annotations

import numpy as np


_P90_LINE_COLOR = "#d81b60"
_P90_LINE_DASHES = (4, 2)
_P90_LINE_WIDTH = 1.2
_P90_LINE_ALPHA = 0.72
_P90_LABEL_FONTSIZE = 7
_P90_LABEL_COLOR = "#a31348"


def compute_sample_p90_tpm(sample_tpm_by_symbol, *, min_tpm=0.0, percentile=90.0):
    """Return the nth-percentile TPM over expressed genes (TPM > ``min_tpm``).

    Parameters
    ----------
    sample_tpm_by_symbol : dict, pandas Series, or iterable of floats
    min_tpm : float
        Excludes genes at or below this TPM (default 0.0 = keep only
        positively-expressed genes).
    percentile : float
        Percentile to compute (default 90).

    Returns
    -------
    float or None
        The percentile value, or ``None`` when the sample has too few
        expressed genes (< 50) to compute a meaningful percentile.
    """
    try:
        import pandas as pd

        if isinstance(sample_tpm_by_symbol, dict):
            values = np.array(
                list(sample_tpm_by_symbol.values()),
                dtype=float,
            )
        elif isinstance(sample_tpm_by_symbol, pd.Series):
            values = sample_tpm_by_symbol.astype(float).to_numpy()
        else:
            values = np.asarray(list(sample_tpm_by_symbol), dtype=float)
    except Exception:
        return None

    expressed = values[(values > min_tpm) & np.isfinite(values)]
    if expressed.size < 50:
        return None
    return float(np.percentile(expressed, percentile))


def add_p90_reference_line(
    ax,
    sample_tpm_by_symbol,
    *,
    orientation="vertical",
    label_fmt=None,
    min_tpm=0.0,
    percentile=90.0,
    value_transform=None,
):
    """Overlay a faint dashed line at the sample's 90th-percentile TPM.

    ``orientation="vertical"`` draws a vertical line at x = p90 — use
    for horizontal-bar plots where TPM is the x-axis.
    ``orientation="horizontal"`` draws a horizontal line at y = p90 —
    use for scatter / vertical-bar plots where TPM is the y-axis.

    The line is cheap visual chrome: no-op when the percentile can't
    be computed (too few expressed genes). A tiny label is placed
    near the line so the reader knows what the threshold represents.

    ``value_transform`` is for plots whose axis is already in a derived
    TPM coordinate, e.g. ``log10(TPM + 1)``. The label still reports the
    raw TPM percentile.

    Returns the p90 TPM (float) or ``None`` when skipped.
    """
    p90 = compute_sample_p90_tpm(
        sample_tpm_by_symbol,
        min_tpm=min_tpm,
        percentile=percentile,
    )
    if p90 is None or p90 <= 0:
        return None

    if label_fmt is None:
        label_fmt = f"bulk sample p{int(percentile)} ({{:.0f}} TPM)"

    line_value = p90
    if value_transform is not None:
        try:
            line_value = float(value_transform(p90))
        except Exception:
            return None
    if not np.isfinite(line_value):
        return None

    def _expand_axis_for_value(axis_orientation, value):
        if axis_orientation == "vertical":
            low, high = ax.get_xlim()
            scale = ax.get_xscale()
            set_lim = ax.set_xlim
        else:
            low, high = ax.get_ylim()
            scale = ax.get_yscale()
            set_lim = ax.set_ylim
        lo, hi = sorted((float(low), float(high)))
        if lo <= value <= hi:
            return
        if scale == "log" and value > 0:
            pad_low = value / 1.25
            pad_high = value * 1.25
        else:
            pad = max(abs(value) * 0.08, 0.1)
            pad_low = value - pad
            pad_high = value + pad
        new_lo = min(lo, pad_low)
        new_hi = max(hi, pad_high)
        if low <= high:
            set_lim(new_lo, new_hi)
        else:
            set_lim(new_hi, new_lo)

    _expand_axis_for_value(orientation, line_value)

    if orientation == "vertical":
        ax.axvline(
            line_value,
            color=_P90_LINE_COLOR,
            linestyle=(0, _P90_LINE_DASHES),
            linewidth=_P90_LINE_WIDTH,
            alpha=_P90_LINE_ALPHA,
            zorder=0.5,
        )
        ax.text(
            line_value,
            ax.get_ylim()[1],
            " " + label_fmt.format(p90),
            ha="left",
            va="top",
            fontsize=_P90_LABEL_FONTSIZE,
            color=_P90_LABEL_COLOR,
            alpha=_P90_LINE_ALPHA + 0.2,
        )
    elif orientation == "horizontal":
        ax.axhline(
            line_value,
            color=_P90_LINE_COLOR,
            linestyle=(0, _P90_LINE_DASHES),
            linewidth=_P90_LINE_WIDTH,
            alpha=_P90_LINE_ALPHA,
            zorder=0.5,
        )
        ax.text(
            ax.get_xlim()[1],
            line_value,
            label_fmt.format(p90) + " ",
            ha="right",
            va="bottom",
            fontsize=_P90_LABEL_FONTSIZE,
            color=_P90_LABEL_COLOR,
            alpha=_P90_LINE_ALPHA + 0.2,
        )
    else:
        raise ValueError(
            f"orientation must be 'vertical' or 'horizontal'; got {orientation!r}"
        )
    return p90
