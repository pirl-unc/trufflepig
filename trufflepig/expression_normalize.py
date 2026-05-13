"""Expression matrix transforms — distinct from QC classification.

QC (``trufflepig.expression_qc``) answers *which features are usable*.
This module answers *how to rescale what survives* and how to convert
between common quantification scales:

- :func:`normalize_expression` — zero technical-RNA features (mtDNA,
  rRNA-like, mt-pseudogenes) and renormalize the remaining mass back to
  the original per-column total so a comparison is not driven by
  technical-RNA denominator drift.
- :func:`fpkm_to_tpm` — rescale each expression column so it sums to
  10⁶, the standard TPM convention. Combine with
  :func:`normalize_expression` to get the analysis-view matrix.
- :func:`tpm_to_housekeeping_normalized` — divide each column by the
  geometric mean of a curated housekeeping panel, producing a unit-free
  ratio scale that survives library-prep totals.
- :func:`renormalize_to_million` — bare utility: rescale columns to
  sum to 10⁶ without dropping anything.

The technical-RNA group definition lives in :mod:`expression_qc` and is
imported from there; this module never reclassifies genes.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from .expression_qc import _TECHNICAL_RNA_GROUPS, classify_gene_qc


# Default removal panel. Mirrors :data:`expression_qc._TECHNICAL_RNA_GROUPS`
# — mtDNA, NUMT-like pseudogenes, nuclear rRNA-like, and the
# polyadenylation-bias lncRNA panel (MALAT1, NEAT1). See the
# expression_qc module for the per-group citations.
_DEFAULT_NORMALIZE_REMOVE_GROUPS = _TECHNICAL_RNA_GROUPS
_KEEP_NONCODING_NORMALIZATION_BIOTYPES = frozenset(
    {
        "protein_coding",
        "protein_coding_cds_not_defined",
        "ig_c_gene",
        "ig_d_gene",
        "ig_j_gene",
        "ig_v_gene",
        "tr_c_gene",
        "tr_d_gene",
        "tr_j_gene",
        "tr_v_gene",
    }
)
_BIOTYPE_COLUMN_CANDIDATES = (
    "biotype",
    "gene_biotype",
    "gene_type",
    "gene_type_name",
    "feature_biotype",
    "feature_type",
)


def _coerce_bool_mask(values):
    import pandas as pd

    return pd.Series(values).fillna(False).astype(bool)


def _infer_biotype_col(df, explicit: str | None = None) -> str | None:
    if explicit and explicit in df.columns:
        return explicit
    for col in _BIOTYPE_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _is_kept_biotype(value: object) -> bool:
    import pandas as pd

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    token = str(value).strip().lower()
    if not token:
        return True
    if token.startswith("protein_coding"):
        return True
    return token in _KEEP_NONCODING_NORMALIZATION_BIOTYPES


def normalize_expression(
    df,
    *,
    label_col: str = "Symbol",
    id_col: str | None = "Ensembl_Gene_ID",
    value_cols: Iterable[str] | None = None,
    group_cols: Iterable[str] | None = None,
    biotype_col: str | None = None,
    remove_noncoding: bool = False,
    remove_groups: Iterable[str] = _DEFAULT_NORMALIZE_REMOVE_GROUPS,
):
    """Zero technical-RNA features and rescale each column's remaining mass.

    The default transform zeroes mitochondrial transcripts, NUMT-like
    mitochondrial pseudogenes, rRNA/rRNA-pseudogene rows, and the
    polyadenylation-bias lncRNA panel (MALAT1, NEAT1), then rescales
    every expression vector so the remaining non-missing TPM mass
    stays on the original per-column total. Raw expression should be
    kept for QC and provenance; this helper defines the comparable
    biology view used after QC.

    Classification is done via :func:`classify_gene_qc`, which prefers
    ENSG-id lookup against the curated pirlygenes QC tables and falls
    back to symbol-level regex. When ``id_col`` is present in the
    frame, ENSG IDs are passed through to the classifier; otherwise
    only ``label_col`` is consulted. Versioned IDs
    (``ENSG00000251562.5``) are stripped to unversioned form before
    lookup.

    ``remove_noncoding=True`` additionally zeroes rows with noncoding
    biotypes when a biotype column is present, while keeping
    protein-coding, immunoglobulin, and TCR biotypes. Off by default
    because lncRNAs and small RNAs can be real biology in some assays.
    """
    import pandas as pd

    if df is None:
        return None, {"applied": False, "reason": "no table", "columns": {}, "groups": {}}
    if label_col not in df.columns:
        return df.copy(), {
            "applied": False,
            "reason": f"label column {label_col!r} not present",
            "columns": {},
            "groups": {},
        }

    out = df.copy()
    if value_cols is None:
        value_cols = [
            c
            for c in out.columns
            if str(c).startswith(("TPM", "nTPM_", "FPKM_", "tcga_"))
        ]
    value_cols = [str(c) for c in value_cols if str(c) in out.columns]
    if not value_cols:
        return out, {
            "applied": False,
            "reason": "no expression value columns",
            "columns": {},
            "groups": {},
        }

    labels = out[label_col].fillna("").astype(str).str.strip()
    remove_group_set = {str(group) for group in remove_groups}
    if id_col and id_col in out.columns:
        ids = out[id_col].fillna("").astype(str).str.split(".").str[0].str.strip()
        qc_classes = [
            classify_gene_qc(sym, ensembl_id=ensg)
            for sym, ensg in zip(labels, ids)
        ]
        qc_classes = pd.Series(qc_classes, index=out.index)
    else:
        qc_classes = labels.map(classify_gene_qc)
    technical_mask = qc_classes.map(lambda qc: qc.group in remove_group_set).astype(bool)
    biotype_col = _infer_biotype_col(out, biotype_col)
    noncoding_mask = _coerce_bool_mask([False] * len(out))
    noncoding_mask.index = out.index
    if remove_noncoding and biotype_col is not None:
        noncoding_mask = ~out[biotype_col].map(_is_kept_biotype).astype(bool)
    removable = (technical_mask | noncoding_mask).astype(bool)
    technical_count = int(technical_mask.sum())
    noncoding_count = int(noncoding_mask.sum()) if remove_noncoding else 0

    def _normalize_indices(indices, record_target):
        nonlocal out
        any_applied_inner = False
        idx = list(indices)
        idx_set = set(idx)
        group_removable = removable.loc[idx]
        group_technical = technical_mask.loc[idx]
        group_noncoding = noncoding_mask.loc[idx]
        for col in value_cols:
            vals = pd.to_numeric(out.loc[idx, col], errors="coerce")
            valid = vals.notna()
            removable_valid = group_removable & valid
            keep_valid = (~group_removable) & valid
            raw_sum = float(vals.sum())
            removed = float(vals[removable_valid].sum())
            remaining = raw_sum - removed
            removed_fraction = removed / raw_sum if raw_sum > 0 else 0.0
            record_target[col] = {
                "input_sum": raw_sum,
                "removed_tpm": removed,
                "removed_fraction": removed_fraction,
                "removed_gene_count": int(group_removable.sum()),
                "removed_technical_gene_count": int(group_technical.sum()),
                "removed_noncoding_gene_count": int(group_noncoding.sum()),
                "renormalization_factor": (
                    float(raw_sum / remaining) if raw_sum > 0 and remaining > 0 else 1.0
                ),
            }
            if raw_sum <= 0 or removed <= 0:
                continue
            remove_idx = [
                i for i in removable_valid[removable_valid].index if i in idx_set
            ]
            if remaining <= 0:
                out.loc[remove_idx, col] = 0.0
                any_applied_inner = True
                continue
            scale = raw_sum / remaining
            keep_idx = [i for i in keep_valid[keep_valid].index if i in idx_set]
            out.loc[remove_idx, col] = 0.0
            out.loc[keep_idx, col] = vals.loc[keep_idx] * scale
            any_applied_inner = True
        return any_applied_inner

    column_records = {}
    group_records = {}
    any_applied = False
    if group_cols is None:
        any_applied = _normalize_indices(out.index, column_records)
    else:
        group_cols = [str(c) for c in group_cols if str(c) in out.columns]
        if not group_cols:
            return out, {
                "applied": False,
                "reason": "missing grouping columns",
                "columns": {},
                "groups": {},
            }
        grouped = out.groupby(group_cols, dropna=False).groups
        for key, idx in grouped.items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            key_label = "|".join(str(part) for part in key_tuple)
            group_records[key_label] = {}
            any_applied = _normalize_indices(idx, group_records[key_label]) or any_applied

    reason = (
        "technical/noncoding expression rows zeroed and remaining expression renormalized"
        if any_applied and remove_noncoding
        else (
            "technical RNA features zeroed and remaining expression renormalized"
            if any_applied
            else "no removable technical/noncoding burden"
        )
    )
    return out, {
        "applied": any_applied,
        "reason": reason,
        "columns": column_records,
        "groups": group_records,
        "label_col": label_col,
        "value_cols": value_cols,
        "group_cols": list(group_cols or []),
        "biotype_col": biotype_col,
        "remove_groups": sorted(remove_group_set),
        "remove_noncoding": bool(remove_noncoding),
        "removed_technical_gene_count": technical_count,
        "removed_noncoding_gene_count": noncoding_count,
        "removed_feature_mode": "zeroed_then_renormalized",
    }


def normalize_technical_rna_columns(
    df,
    *,
    label_col: str = "Symbol",
    value_cols: Iterable[str] | None = None,
):
    """Zero mtDNA/rRNA-like features and renormalize every expression column.

    Shared comparability transform for reference matrices. Preserves
    each column's total expression mass after removing technical-RNA
    features. Raw-expression QC should be computed before this
    transform (see :mod:`trufflepig.expression_qc`).
    """
    return normalize_expression(
        df,
        label_col=label_col,
        value_cols=value_cols,
        remove_noncoding=False,
    )


def normalize_technical_rna_long_table(
    df,
    *,
    label_col: str = "symbol",
    group_cols: Iterable[str] = ("cancer_code", "subtype"),
    value_cols: Iterable[str] = ("tumor_tpm_median", "tumor_tpm_q1", "tumor_tpm_q3"),
):
    """Apply technical-RNA normalization within each long-table cohort group."""
    return normalize_expression(
        df,
        label_col=label_col,
        group_cols=group_cols,
        value_cols=value_cols,
        remove_noncoding=False,
    )


# ---------- conversions ----------


def renormalize_to_million(
    df,
    *,
    value_cols: Iterable[str] | None = None,
):
    """Rescale each expression column so its non-NaN sum equals 10⁶.

    Bare utility — drops nothing. Use this after
    :func:`normalize_expression` if you also want the post-filter total
    pinned at exactly 10⁶ (the standard TPM convention). The default
    technical-RNA normalization preserves the *input* total, which may
    or may not already be 10⁶ depending on how the upstream quantifier
    handled denominator features.
    """
    import pandas as pd

    if df is None:
        return None, {"applied": False, "reason": "no table", "columns": {}}
    out = df.copy()
    if value_cols is None:
        value_cols = [
            c
            for c in out.columns
            if str(c).startswith(("TPM", "nTPM_", "FPKM_", "tcga_"))
        ]
    value_cols = [str(c) for c in value_cols if str(c) in out.columns]
    if not value_cols:
        return out, {
            "applied": False,
            "reason": "no expression value columns",
            "columns": {},
        }

    columns = {}
    any_applied = False
    for col in value_cols:
        vals = pd.to_numeric(out[col], errors="coerce")
        col_sum = float(vals.sum())
        columns[col] = {"input_sum": col_sum}
        if col_sum <= 0:
            columns[col]["scale"] = 1.0
            continue
        scale = 1e6 / col_sum
        out[col] = vals * scale
        columns[col]["scale"] = scale
        columns[col]["output_sum"] = 1e6
        any_applied = True
    return out, {
        "applied": any_applied,
        "reason": "rescaled to TPM convention (sum = 1e6)" if any_applied else "no positive column sums",
        "columns": columns,
        "value_cols": value_cols,
    }


def fpkm_to_tpm(
    df,
    *,
    value_cols: Iterable[str] | None = None,
):
    """Convert FPKM-scale expression columns to TPM by per-column rescaling.

    For each column, ``TPM_i = FPKM_i / sum(FPKM_j) * 1e6`` over rows
    with a finite value. Pair this with :func:`normalize_expression` to
    drop technical-RNA features before renormalizing — the standard
    flow is:

        1. :func:`fpkm_to_tpm` — convert quantifier output to TPM scale.
        2. :func:`trufflepig.expression_qc.raw_qc_profile` — record raw
           rRNA / mt / top-k composition for the report before any
           filtering.
        3. :func:`normalize_expression` — drop technical RNA and
           renormalize the remaining mass.
        4. :func:`renormalize_to_million` (optional) — pin the
           post-filter total at exactly 10⁶.

    This is mathematically identical to :func:`renormalize_to_million`
    but takes an FPKM-named argument and exists as a self-documenting
    entry point in the pipeline.
    """
    return renormalize_to_million(df, value_cols=value_cols)


def _housekeeping_panel_symbols(panel: Iterable[str] | None = None) -> set[str]:
    if panel is not None:
        return {str(s).strip().upper() for s in panel if str(s).strip()}
    from pirlygenes import housekeeping_gene_names

    return {str(s).strip().upper() for s in housekeeping_gene_names() if str(s).strip()}


def _housekeeping_panel_ensembl_ids(panel: Iterable[str] | None = None) -> set[str]:
    """Unversioned ENSG IDs for the housekeeping panel."""
    if panel is not None:
        return {str(s).split(".", 1)[0].strip() for s in panel if str(s).strip()}
    from pirlygenes import housekeeping_gene_ids

    return {str(s).split(".", 1)[0].strip() for s in housekeeping_gene_ids() if str(s).strip()}


def tpm_to_housekeeping_normalized(
    df,
    *,
    label_col: str = "Symbol",
    id_col: str | None = "Ensembl_Gene_ID",
    value_cols: Iterable[str] | None = None,
    panel: Iterable[str] | None = None,
    panel_ids: Iterable[str] | None = None,
    pseudocount: float = 0.1,
):
    """Divide each expression column by the geometric mean of a HK panel.

    Uses :func:`pirlygenes.housekeeping_gene_names` by default (~22
    classic housekeeping genes, e.g. ACTB, GAPDH, B2M, HPRT1, PGK1).
    Pass ``panel`` to override.

    A small ``pseudocount`` is added before taking the log to make the
    geometric mean robust to zeros in the panel. Returns the
    HK-normalized frame plus a record dict with the panel-symbol
    coverage and per-column denominator value, so callers can audit
    whether the panel was actually present.

    The output is on a unit-free ratio scale (gene expression relative
    to the housekeeping baseline). It survives library-prep total
    drift in a way that TPM does not — useful when comparing samples
    across very different sequencing depth or capture chemistry.
    """
    import numpy as np
    import pandas as pd

    if df is None:
        return None, {"applied": False, "reason": "no table", "columns": {}}
    if label_col not in df.columns:
        return df.copy(), {
            "applied": False,
            "reason": f"label column {label_col!r} not present",
            "columns": {},
        }

    out = df.copy()
    if value_cols is None:
        value_cols = [
            c
            for c in out.columns
            if str(c).startswith(("TPM", "nTPM_", "FPKM_", "tcga_"))
        ]
    value_cols = [str(c) for c in value_cols if str(c) in out.columns]
    if not value_cols:
        return out, {
            "applied": False,
            "reason": "no expression value columns",
            "columns": {},
        }

    # ENSG-first matching when both an id_col is present and we have
    # ENSG IDs from pirlygenes; falls back to symbol matching when
    # either side is missing.
    if id_col and id_col in out.columns:
        panel_ensgs = _housekeeping_panel_ensembl_ids(panel_ids)
        ids = out[id_col].fillna("").astype(str).str.split(".").str[0].str.strip()
        hk_mask = ids.isin(panel_ensgs)
        match_mode = "ensembl_id"
        panel_size = len(panel_ensgs)
    else:
        panel_syms = _housekeeping_panel_symbols(panel)
        labels = out[label_col].fillna("").astype(str).str.strip().str.upper()
        hk_mask = labels.isin(panel_syms)
        match_mode = "symbol"
        panel_size = len(panel_syms)
    n_hk_present = int(hk_mask.sum())
    if n_hk_present == 0:
        return out, {
            "applied": False,
            "reason": f"no housekeeping panel genes found via {match_mode}",
            "columns": {},
            "panel_size": panel_size,
            "match_mode": match_mode,
        }

    columns = {}
    any_applied = False
    for col in value_cols:
        vals = pd.to_numeric(out[col], errors="coerce")
        hk_vals = vals[hk_mask].dropna()
        if hk_vals.empty:
            columns[col] = {"hk_geomean": None, "scale": 1.0, "n_hk_used": 0}
            continue
        log_vals = np.log(hk_vals.astype(float) + pseudocount)
        hk_geomean = float(np.exp(log_vals.mean()) - pseudocount)
        if hk_geomean <= 0:
            columns[col] = {"hk_geomean": hk_geomean, "scale": 1.0, "n_hk_used": int(len(hk_vals))}
            continue
        out[col] = vals / hk_geomean
        columns[col] = {
            "hk_geomean": hk_geomean,
            "scale": 1.0 / hk_geomean,
            "n_hk_used": int(len(hk_vals)),
        }
        any_applied = True
    return out, {
        "applied": any_applied,
        "reason": "divided by housekeeping geomean per column" if any_applied else "no positive HK geomeans",
        "columns": columns,
        "value_cols": value_cols,
        "panel_size": panel_size,
        "panel_present_in_table": n_hk_present,
        "match_mode": match_mode,
        "pseudocount": pseudocount,
    }


__all__ = [
    "normalize_expression",
    "normalize_technical_rna_columns",
    "normalize_technical_rna_long_table",
    "renormalize_to_million",
    "fpkm_to_tpm",
    "tpm_to_housekeeping_normalized",
]
