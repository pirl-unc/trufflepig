# Licensed under the Apache License, Version 2.0

"""Single source of truth for how numbers are rendered in markdown
reports (#120, follow-up from #79 Problem 4).

Before this module existed, each renderer had its own ad-hoc format
string — the candidate table used ``{:.3f}`` while the summary used
``{:.1f}×`` and the surface targets table used ``{:.1f}x`` with
``infx`` for infinity. The reader saw the same quantity rendered
three different ways across a single run.

Every markdown-facing number should go through one of the four helpers
below so the formatting contract is defined in one place:

- :func:`render_fold` — fold-change vs reference, always with ``×``.
- :func:`render_fraction` — percentages.
- :func:`render_tpm` — TPM values, tighter precision for large values.
- :func:`render_score` — unit-less scores and support values.

TSV outputs stay machine-parseable with full precision; these helpers
are strictly for human-facing markdown.
"""

from __future__ import annotations

import math
from typing import Union

Number = Union[int, float, None]


def _none_or_nan(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return False


def render_fold(x: Number, *, placeholder: str = "—") -> str:
    """Fold change vs reference.

    Two decimals, U+00D7 multiplication sign, ``∞×`` for infinity,
    ``placeholder`` for None / NaN. Negative values are allowed and
    render with a leading sign.

    >>> render_fold(3.542)
    '3.54×'
    >>> render_fold(2.5)
    '2.50×'
    >>> render_fold(float("inf"))
    '∞×'
    >>> render_fold(None)
    '—'
    """
    if _none_or_nan(x):
        return placeholder
    x = float(x)
    if math.isinf(x):
        return "-∞×" if x < 0 else "∞×"
    return f"{x:.2f}\u00d7"


def render_fraction(x: Number, *, placeholder: str = "—") -> str:
    """Fraction rendered as a one-decimal percentage.

    Accepts either a [0, 1] fraction or a 0-to-100 percentage — the
    helper auto-detects on magnitude. This is deliberate: a lot of
    existing call-sites carry fractions (``0.64``) and some carry
    already-computed percentages (``64.3``); both should render as
    ``64.0%`` / ``64.3%`` without requiring callers to pre-convert.

    For values that exceed 1 but are plausibly fractions (a CI
    upper bound can hit exactly 1.0), the helper treats the whole
    [0, 1.001] range as a fraction.

    >>> render_fraction(0.643)
    '64.3%'
    >>> render_fraction(1.0)
    '100.0%'
    >>> render_fraction(None)
    '—'
    """
    if _none_or_nan(x):
        return placeholder
    x = float(x)
    if -0.001 <= x <= 1.001:
        x = x * 100.0
    return f"{x:.1f}%"


def render_fraction_no_decimal(x: Number, *, placeholder: str = "—") -> str:
    """Fraction rendered as a zero-decimal percentage.

    Same auto-detect semantics as :func:`render_fraction` but emits
    ``64%`` instead of ``64.3%`` — used where the narrative is the
    focus and precision noise hurts readability (e.g. the Brief).
    """
    if _none_or_nan(x):
        return placeholder
    x = float(x)
    if -0.001 <= x <= 1.001:
        x = x * 100.0
    return f"{x:.0f}%"


def render_tpm(x: Number, *, placeholder: str = "—") -> str:
    """TPM value with magnitude-aware precision.

    - Above 100: no decimals. Large numbers don't benefit from a
      trailing ``.0`` — ``1852.0`` reads as fake precision, ``1852``
      is what the reader wants.
    - Between 10 and 100: one decimal.
    - Between 1 and 10: one decimal.
    - Below 1: two decimals so dim signals aren't collapsed to ``0.0``.
    - Negative values render the same way (shouldn't occur; safety net).
    - None / NaN: ``placeholder``.

    >>> render_tpm(1852.4)
    '1852'
    >>> render_tpm(142.0)
    '142'
    >>> render_tpm(27.8)
    '27.8'
    >>> render_tpm(4.7)
    '4.7'
    >>> render_tpm(0.12)
    '0.12'
    >>> render_tpm(None)
    '—'
    """
    if _none_or_nan(x):
        return placeholder
    x = float(x)
    ax = abs(x)
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 1:
        return f"{x:.1f}"
    return f"{x:.2f}"


def render_score(x: Number, *, placeholder: str = "—") -> str:
    """Unit-less score with three decimals.

    Used for the candidate-ranking columns (signature / geomean /
    normalized) and for support scores.

    >>> render_score(0.670)
    '0.670'
    >>> render_score(1.0)
    '1.000'
    >>> render_score(None)
    '—'
    """
    if _none_or_nan(x):
        return placeholder
    return f"{float(x):.3f}"


__all__ = [
    "render_fold",
    "render_fraction",
    "render_fraction_no_decimal",
    "render_tpm",
    "render_score",
]
