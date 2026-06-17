"""Runtime guards for clean TPM expression matrices."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import pandas as pd
from pirlygenes.expression.qc import (
    TECHNICAL_FRACTION,
    TECHNICAL_RNA_GROUPS,
    classify_gene_qc,
)

# pirlygenes owns the gene-QC taxonomy AND the definition of which groups make up
# the zero-and-renormalize technical-RNA compartment. Consume its PUBLIC set
# rather than keeping a local copy (which would silently drift when the upstream
# technical-RNA set changes). Re-exported under the trufflepig name so existing
# `from trufflepig.clean_tpm import TECHNICAL_RNA_QC_GROUPS` imports still resolve.
TECHNICAL_RNA_QC_GROUPS = frozenset(TECHNICAL_RNA_GROUPS)


@lru_cache(maxsize=200_000)
def _cached_qc_group(label: str, ensembl_id: str | None = None) -> str:
    return classify_gene_qc(label, ensembl_id=ensembl_id).group


# Gene label / id column names seen across the loaders + reference frames, in
# resolution priority. Centralized so callers don't re-spell the candidate list.
_LABEL_COL_CANDIDATES = (
    "gene_display_name",
    "canonical_gene_name",
    "gene",
    "gene_symbol",
    "symbol",
    "Symbol",
)
_ID_COL_CANDIDATES = (
    "canonical_gene_id",
    "ensembl_gene_id",
    "gene_id",
    "Ensembl_Gene_ID",
)


def resolve_gene_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return the (label_col, id_col) present in ``df`` for clean-TPM helpers.

    The loaders and reference frames spell these columns several ways
    (``canonical_gene_name`` vs ``symbol`` vs ``Symbol``, etc.); resolve them
    once here so callers pass the frame, not a re-spelled candidate list.
    """
    label = next((c for c in _LABEL_COL_CANDIDATES if c in df.columns), None)
    id_col = next((c for c in _ID_COL_CANDIDATES if c in df.columns), None)
    return label, id_col


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


def normalize_to_reference_space(
    df: pd.DataFrame,
    *,
    value_cols: Iterable[str],
    label_col: str | None = "Symbol",
    id_col: str | None = "Ensembl_Gene_ID",
) -> pd.DataFrame:
    """Conform an expression sample to the cohort reference's clean-TPM space by
    **deferring the transform to pirlygenes** — the normalization owner.

    The clean-TPM definition (16% ribosomal-protein, 9% other-technical, 75%
    biological — three separately-pinned compartments of the 1e6 budget) lives in
    :func:`pirlygenes.expression.normalize.clean_tpm_matrix`. The reference matrices
    are built with it, so the sample is conformed by the *same* code and lands in
    the identical space — and trufflepig encodes no normalization definition of its
    own, so it tracks any upstream change automatically.

    A frame with no technical-RNA compartment (a partial frame / synthetic fixture,
    or one lacking an id column) is left as-is rather than forced through the
    transform.
    """
    cols = [col for col in value_cols if col in df.columns]
    if not cols:
        return df
    if not id_col or id_col not in df.columns:
        return df
    # Partial frame / synthetic fixture (no technical compartment) — leave as-is
    # rather than force a rescale. (The only trufflepig-side check; the transform
    # itself is entirely pirlygenes'.)
    if not bool(technical_rna_mask(df, label_col=label_col, id_col=id_col).any()):
        return df
    # Defer the ENTIRE clean-TPM conform to pirlygenes' df-level entry point
    # (16/9/75 fixed_fraction): df in, cleaned df out, no masks/tables/fractions
    # built here. Same transform that produced the reference matrices.
    from pirlygenes.expression.normalize import normalize_expression

    out, _ = normalize_expression(
        df,
        label_col=label_col,
        id_col=id_col,
        value_cols=cols,
        censored_fill="fixed_fraction",
    )
    return out


def assert_clean_tpm(
    df: pd.DataFrame,
    *,
    value_cols: Iterable[str],
    label_col: str | None = "Symbol",
    id_col: str | None = "Ensembl_Gene_ID",
    context: str = "clean TPM",
    tolerance: float = 1e-9,
    technical_fraction: float | None = TECHNICAL_FRACTION,
    fraction_slack: float = 0.15,
) -> None:
    """Raise when a TPM matrix has not been cleaned of technical-RNA inflation.

    Two contracts, selected by ``technical_fraction``:

    - **fixed-fraction** (DEFAULT — ``technical_fraction`` defaults to pirlygenes'
      consumed ``TECHNICAL_FRACTION``): clean TPM deliberately *pins* the
      technical-RNA + ribosomal compartment to a fixed fraction of the 1e6 budget
      rather than zeroing it (avoids inflating biological genes). Technical RNA is
      therefore expected to be non-zero; assert only that the strict technical-RNA
      rows (a subset of that compartment) don't *exceed* ``technical_fraction +
      fraction_slack`` of the column total — which still catches egregiously
      un-normalized (technical-dominant) input without rejecting clean data.
    - **legacy "zeroed"** (OPT-IN — pass ``technical_fraction=None``): the
      historical transform zeroed technical-RNA features, so their summed TPM must
      be ~0 (within ``tolerance``). A caller that genuinely requires strict-zeroed
      data must pass ``technical_fraction=None`` explicitly — it is no longer the
      default now that all references ship in the fixed-fraction form.
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
