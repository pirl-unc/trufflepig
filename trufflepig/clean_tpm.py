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
    """Conform an expression matrix to the cohort reference's clean-TPM space.

    The cohort reference (``pan_cancer_expression`` / ``tcga_deconvolved_expression``)
    ships with the strict technical-RNA compartment (mt-DNA / rRNA-like / mt-like
    pseudogenes / polyA-bias lncRNA) zeroed. A ``clean_tpm_v4`` input sample keeps
    that compartment at a fixed fraction (~15% of the 1e6 budget), so the sample's
    biological genes are systematically diluted ~15-25% relative to the reference —
    a space mismatch in every sample↔cohort comparison (#74).

    This deterministically zeroes that same compartment and renormalizes each
    column back to 1e6, so **any** input (v4 fixed-fraction, legacy-zeroed, or raw)
    lands in the one reference space. It's a uniform rescale of the surviving
    (biological + ribosomal-protein) genes, so rank/correlation/HK-ratio scoring is
    unchanged; only the absolute budget shared with the reference is corrected.
    Idempotent: an already-zeroed input is unaffected. QC (mt-fraction, RNA-quant)
    runs upstream on the raw frame, so this never hides degradation signal.
    """
    cols = [col for col in value_cols if col in df.columns]
    if not cols:
        return df
    mask = technical_rna_mask(df, label_col=label_col, id_col=id_col)
    if not bool(mask.any()):
        # No technical-RNA compartment present (already conformed, or a partial
        # frame / synthetic fixture) — nothing to zero, so leave the budget as-is
        # rather than force a 1e6 rescale that would alter the input.
        return df
    out = df.copy()
    out.loc[mask, cols] = 0.0
    for col in cols:
        total = float(pd.to_numeric(out[col], errors="coerce").fillna(0.0).sum())
        if total > 0:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0) * 1e6 / total
    return out


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
