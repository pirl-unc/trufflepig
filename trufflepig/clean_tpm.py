"""Runtime guards for clean TPM expression matrices."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import pandas as pd
from pirlygenes.expression.qc import classify_gene_qc


TECHNICAL_RNA_QC_GROUPS = frozenset(
    {
        "mt_dna",
        "mt_like_pseudogene",
        "rrna_like",
        "polyadenylation_bias_lncrna",
    }
)


@lru_cache(maxsize=200_000)
def _cached_qc_group(label: str, ensembl_id: str | None = None) -> str:
    return classify_gene_qc(label, ensembl_id=ensembl_id).group


def technical_rna_mask(
    df: pd.DataFrame,
    *,
    label_col: str | None = "Symbol",
    id_col: str | None = "Ensembl_Gene_ID",
) -> pd.Series:
    """Return rows classified as technical-RNA artifacts."""
    if label_col and label_col in df.columns:
        labels = df[label_col].fillna("").astype(str)
    else:
        labels = pd.Series([""] * len(df), index=df.index)
    if id_col and id_col in df.columns:
        ids = df[id_col].fillna("").astype(str)
        qc_classes = [
            _cached_qc_group(str(label), str(ensg))
            for label, ensg in zip(labels, ids)
        ]
    else:
        qc_classes = [_cached_qc_group(str(label), None) for label in labels]
    return pd.Series(
        [group in TECHNICAL_RNA_QC_GROUPS for group in qc_classes],
        index=df.index,
        dtype=bool,
    )


def assert_clean_tpm(
    df: pd.DataFrame,
    *,
    value_cols: Iterable[str],
    label_col: str | None = "Symbol",
    id_col: str | None = "Ensembl_Gene_ID",
    context: str = "clean TPM",
    tolerance: float = 1e-9,
    technical_fraction: float | None = None,
    fraction_slack: float = 0.15,
) -> None:
    """Raise when a TPM matrix has not been cleaned of technical-RNA inflation.

    Two contracts, selected by ``technical_fraction``:

    - **legacy "zeroed"** (default, ``technical_fraction=None``): the historical
      clean-TPM transform zeros technical-RNA features, so their summed TPM must
      be ~0 (within ``tolerance``).
    - **v4 "fixed-fraction"** (``technical_fraction`` set, e.g. ``0.25``):
      pirlygenes clean_tpm_v4 deliberately *pins* the technical-RNA + ribosomal
      compartment to a fixed fraction of the 1e6 budget rather than zeroing it
      (avoids inflating the biological genes). Technical RNA is therefore
      expected to be non-zero; assert only that the strict technical-RNA rows
      (a subset of that compartment) don't *exceed* ``technical_fraction +
      fraction_slack`` of the column total — which still catches egregiously
      un-normalized (technical-dominant) input without rejecting v4 data.
    """
    cols = [col for col in value_cols if col in df.columns]
    if not cols or df.empty:
        return
    mask = technical_rna_mask(df, label_col=label_col, id_col=id_col)
    if not bool(mask.any()):
        return
    values = df.loc[mask, cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    per_col = values.sum(axis=0)

    labels = (
        df.loc[mask, label_col].fillna("").astype(str)
        if label_col and label_col in df.columns
        else pd.Series([""] * int(mask.sum()), index=df.index[mask])
    )
    row_totals = values.abs().sum(axis=1).sort_values(ascending=False)

    if technical_fraction is not None:
        # v4 two-compartment: bound the technical fraction rather than zero it.
        col_totals = (
            df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=0)
        )
        limit = float(technical_fraction) + float(fraction_slack)
        frac = pd.Series(
            {
                col: (float(per_col[col]) / float(col_totals[col]))
                if float(col_totals[col]) > 0
                else 0.0
                for col in cols
            }
        )
        bad = frac[frac > limit]
        if bad.empty:
            return
        top = []
        for idx, total in row_totals.head(5).items():
            label = labels.loc[idx] if idx in labels.index else str(idx)
            top.append(f"{label or idx}={float(total):.3g}")
        bad_cols = ", ".join(f"{col}={val:.1%}" for col, val in bad.head(8).items())
        raise ValueError(
            f"{context} technical-RNA fraction exceeds the v4 bound "
            f"({limit:.0%}): {bad_cols}. Top technical rows: {', '.join(top)}"
        )

    bad = per_col[per_col.abs() > tolerance]
    if bad.empty:
        return
    top = []
    for idx, total in row_totals.head(5).items():
        label = labels.loc[idx] if idx in labels.index else str(idx)
        top.append(f"{label or idx}={float(total):.3g}")
    bad_cols = ", ".join(f"{col}={float(val):.3g}" for col, val in bad.head(8).items())
    raise ValueError(
        f"{context} contains nonzero technical-RNA TPM after cleaning: "
        f"{bad_cols}. Top technical rows: {', '.join(top)}"
    )
