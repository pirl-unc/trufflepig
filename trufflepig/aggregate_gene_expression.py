"""Transcript-level → gene-level TPM rollup.

Thin re-export of :func:`pirlygenes.aggregate_gene_expression`. The
implementation moved to pirlygenes 5.1+ alongside the other expression
primitives; this module is kept so existing imports
(``from trufflepig.aggregate_gene_expression import aggregate_gene_expression``)
keep working.

The trufflepig wrapper preserves the historical loud-by-default
behavior (``verbose=True, progress=True``) — pirlygenes flipped both
to False to be a quieter library, but the analysis CLI here surfaces
the chatter as part of its normal run output.
"""

from __future__ import annotations

import pandas as pd

from pirlygenes.expression.aggregate import (
    _expanded_tx_map,
    aggregate_gene_expression as _pirlygenes_aggregate_gene_expression,
)


def aggregate_gene_expression(
    df: pd.DataFrame,
    tx_to_gene_name=None,
    transcript_id_column_candidates=None,
    tpm_column_candidates=("tpm",),
    verbose: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Aggregate transcript-level TPM values to gene-level TPM values.

    See :func:`pirlygenes.aggregate_gene_expression` for the underlying
    implementation. This wrapper restores the trufflepig CLI defaults
    (``verbose=True, progress=True``) so existing call-sites keep their
    output.
    """
    kwargs = {
        "tpm_column_candidates": tpm_column_candidates,
        "verbose": verbose,
        "progress": progress,
    }
    if tx_to_gene_name is not None:
        kwargs["tx_to_gene_name"] = tx_to_gene_name
    if transcript_id_column_candidates is not None:
        kwargs["transcript_id_column_candidates"] = transcript_id_column_candidates
    return _pirlygenes_aggregate_gene_expression(df, **kwargs)


__all__ = ["aggregate_gene_expression", "_expanded_tx_map"]
