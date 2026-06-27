"""Expression matrix transforms — distinct from QC classification.

The mechanical transforms are **canonical in oncoref** (``oncoref.normalization``). This module is
a back-compat shim so existing ``from trufflepig.expression_normalize import …`` paths keep working.

Two of them — ``normalize_expression`` and ``tpm_to_housekeeping_normalized`` — are re-exported from
**pirlygenes** rather than oncoref, deliberately: pirlygenes' wrappers expose a WIDER, legacy
signature (``remove_noncoding``, ``biotype_col``, ``protect``, ``technical_fraction`` / ``panel``,
``min_hk_positive_*``) that callers of this back-compat path still pass, and they delegate to oncoref
internally so the numerics stay canonical. Importing them straight from oncoref would raise TypeError
on those legacy kwargs (oncoref exposes a narrower signature). The purely-mechanical transforms
(``fpkm_to_tpm`` etc.) have stable signatures and come straight from oncoref.

Boundary:

    oncoref    ─ canonical rescaling primitives + technical-RNA gene-set definitions + QC classifier
    pirlygenes ─ gene↔biology panels + the legacy-signature normalization wrappers (delegate to oncoref)
    trufflepig ─ per-sample QC narration, decomposition, sample-level judgments (analysis layer)

If you are writing NEW code with the narrower modern signature, prefer importing directly from
``oncoref.normalization``; this module exists for the existing wider-signature call sites.
"""

from __future__ import annotations

from oncoref.normalization import (
    fpkm_to_tpm,
    normalize_technical_rna_columns,
    normalize_technical_rna_long_table,
    renormalize_to_million,
)

# Legacy-signature wrappers (delegate to oncoref internally) — keep the wider kwargs that
# back-compat callers of this import path still use (e.g. remove_noncoding=, biotype_col=, panel=).
from pirlygenes.expression.normalize import (
    normalize_expression,
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
