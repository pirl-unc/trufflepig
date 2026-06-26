"""Expression matrix transforms — distinct from QC classification.

The mechanical transforms are **canonical in oncoref** (``oncoref.normalization``); pirlygenes'
``pirlygenes.expression.normalize`` already delegates to it. This module re-exports oncoref's
transforms so existing trufflepig import paths (``from trufflepig.expression_normalize import …``)
keep working while sourcing the canonical implementation directly (no pirlygenes hop).

Boundary:

    oncoref    ─ canonical rescaling primitives + technical-RNA gene-set definitions + QC classifier
    pirlygenes ─ gene↔biology panels (delegates normalization to oncoref)
    trufflepig ─ per-sample QC narration, decomposition, sample-level judgments (analysis layer)

If you are writing new code, prefer importing directly from ``oncoref.normalization``.
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
