"""Expression matrix transforms — distinct from QC classification.

The mechanical transforms live in pirlygenes 5.1+ (see
``pirlygenes.expression.normalize``). This module is a thin
re-export so existing trufflepig import paths
(``from trufflepig.expression_normalize import …``) continue to work.

Boundary:

    pirlygenes ─ rescaling primitives + technical-RNA gene-set
                 definitions + QC classifier (curated reference data
                 + mechanical operations on it)
    trufflepig ─ per-sample QC narration, decomposition, sample-level
                 judgments (analysis layer)

If you are writing new code, prefer importing directly from
``pirlygenes``: the re-exports here exist only for back-compat.
"""

from __future__ import annotations

from pirlygenes.expression.normalize import (
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
