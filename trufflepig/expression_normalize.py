"""Expression matrix transforms — distinct from QC classification.

The mechanical transforms are **canonical in oncoref** (``oncoref.normalization``); pirlygenes'
``pirlygenes.expression.normalize`` already delegates to it. This module re-exports oncoref's
transforms so existing trufflepig import paths (``from trufflepig.expression_normalize import …``)
keep working while sourcing the canonical implementation directly (no pirlygenes hop).

These re-exports carry oncoref's MODERN signature (e.g. ``normalize_expression`` takes
``remove_groups=``, not the older pirlygenes ``remove_noncoding=`` / ``biotype_col=`` / ``protect=``;
``tpm_to_housekeeping_normalized`` takes ``panel_ids=``, not ``panel=``). No trufflepig caller uses
the legacy kwargs through THIS path — the one that did (``load_expression``) uses ``remove_groups``,
and code that genuinely needs the wider pirlygenes wrapper (``reference.py``) imports
``pirlygenes.expression.normalize`` directly. So write new code against the oncoref signature; if you
need the legacy-kwarg wrapper, import from ``pirlygenes.expression.normalize``.

Boundary:

    oncoref    ─ canonical rescaling primitives + technical-RNA gene-set definitions + QC classifier
    pirlygenes ─ gene↔biology panels (delegates normalization to oncoref)
    trufflepig ─ per-sample QC narration, decomposition, sample-level judgments (analysis layer)
"""

from __future__ import annotations

from oncoref.normalization import (
    fpkm_to_tpm,
    normalize_expression,
    normalize_technical_rna_columns,
    normalize_technical_rna_long_table,
    renormalize_to_million,
    tpm_to_housekeeping_normalized,
)

__all__ = [
    "fpkm_to_tpm",
    "normalize_expression",
    "normalize_technical_rna_columns",
    "normalize_technical_rna_long_table",
    "renormalize_to_million",
    "tpm_to_housekeeping_normalized",
]
