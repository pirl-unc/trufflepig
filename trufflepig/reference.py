"""Expression-backed reference data for the analysis engine.

Thin wrapper around the canonical pirlygenes 5.1 accessors. The CSVs
and the rescaling primitives now live in pirlygenes — trufflepig
preserves the legacy wide kwarg surface (``technical_rna_normalize``,
``remove_noncoding``, ``renormalize_to_million``) and the
``_PAN_CANCER_CACHE`` / ``_tcga_deconv_wide`` monkey-patch points so
existing analysis callers and tests continue to bind here.

Boundary (post-pirlygenes#246/#247/#248):

    pirlygenes ─ CSVs + accessors + rescaling primitives
                 (``pan_cancer_expression``, ``normalize_expression``,
                  ``renormalize_to_million``, …)
    trufflepig ─ analysis composition: this module orchestrates the
                 wide-form merge, the technical-RNA scrub, and the
                 renormalize/housekeeping passes on top of pirlygenes
                 data, and serves as the in-process cache for them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pirlygenes.load_dataset import get_data as _pirlygenes_get_data


# ---------- generic loader ----------


def get_reference_data(name: str) -> pd.DataFrame:
    """Load a bundled expression CSV from pirlygenes by short name.

    Thin pass-through to :func:`pirlygenes.load_dataset.get_data` so
    tests can still monkey-patch this attribute. Raises ``ValueError``
    when the dataset isn't bundled.
    """
    return _pirlygenes_get_data(name)


# ---------- pan-cancer / TCGA-deconvolved expression ----------


_PAN_CANCER_CACHE: dict = {}


def _pan_cancer_cache_key(
    genes,
    normalize,
    log_transform,
    technical_rna_normalize,
    remove_noncoding,
    renormalize_to_million=True,
):
    genes_key = None if genes is None else frozenset(str(g).upper() for g in genes)
    return (
        genes_key,
        normalize,
        bool(log_transform),
        bool(technical_rna_normalize),
        bool(remove_noncoding),
        bool(renormalize_to_million),
    )


def tcga_deconvolved_expression():
    """Per-(symbol, TCGA code) tumor-only TPM from the offline deconvolution.

    Long-form frame with columns ``symbol``, ``cancer_code``,
    ``tumor_tpm_median``, ``tumor_tpm_q1``, ``tumor_tpm_q3``, ``n_samples``.
    Returns ``None`` if the CSV is not bundled (the bundled-with-pirlygenes
    case never fires this branch but it preserves the legacy contract
    for tests that monkey-patch ``get_reference_data`` to raise).
    """
    try:
        return get_reference_data("tcga-deconvolved-expression")
    except ValueError:
        return None


def _tcga_deconv_wide(cache={}):
    """Wide-form (Symbol-indexed) view of the tumor-only TPM median.

    One column per TCGA code (``tcga_PRAD``, ``tcga_BRCA`` …). Cached
    in-process; returns ``None`` when the deconv CSV is absent.
    """
    if "value" in cache:
        return cache["value"]
    long = tcga_deconvolved_expression()
    if long is None or long.empty:
        cache["value"] = None
        return None
    wide = long.pivot_table(
        index="symbol",
        columns="cancer_code",
        values="tumor_tpm_median",
        aggfunc="median",
    )
    wide.columns = [f"tcga_{c}" for c in wide.columns]
    wide = wide.reset_index().rename(columns={"symbol": "Symbol"})
    cache["value"] = wide
    return wide


def subtype_deconvolved_expression(
    technical_rna_normalize=False,
    remove_noncoding=False,
    renormalize_to_million=True,
):
    """Per-(cancer_code, subtype, symbol) tumor-only TPM from multi-cohort deconv.

    Long-form frame; ``None`` when the CSV isn't bundled.
    ``technical_rna_normalize`` zeros mtDNA / rRNA-like rows per subtype
    group; ``remove_noncoding`` adds a biotype gate when a biotype
    column is present. ``renormalize_to_million`` rescales each
    (cancer_code, subtype) group's value columns to sum=1e6.
    """
    from .expression_normalize import (
        normalize_expression,
        normalize_technical_rna_long_table,
    )

    try:
        df = get_reference_data("subtype-deconvolved-expression")
    except ValueError:
        return None

    if renormalize_to_million:
        df = df.copy()
        for value_col in ("tumor_tpm_median", "tumor_tpm_q1", "tumor_tpm_q3"):
            if value_col not in df.columns:
                continue
            vals = pd.to_numeric(df[value_col], errors="coerce")
            group_sums = vals.groupby(
                [df["cancer_code"], df["subtype"]]
            ).transform("sum")
            scale = pd.Series(
                np.where(group_sums > 0, 1e6 / group_sums, 1.0),
                index=df.index,
            )
            df[value_col] = vals * scale

    if not technical_rna_normalize and not remove_noncoding:
        return df
    if remove_noncoding:
        df_norm, record = normalize_expression(
            df,
            label_col="symbol",
            group_cols=("cancer_code", "subtype"),
            value_cols=("tumor_tpm_median", "tumor_tpm_q1", "tumor_tpm_q3"),
            remove_noncoding=True,
        )
    else:
        df_norm, record = normalize_technical_rna_long_table(df)
    df_norm.attrs["technical_rna_normalization"] = record
    return df_norm


def pan_cancer_expression(
    genes=None,
    normalize=None,
    log_transform=False,
    technical_rna_normalize=False,
    remove_noncoding=False,
    renormalize_to_million=True,
):
    """Expression across 50 normal tissues (nTPM) and 33 TCGA cancer types.

    Wraps :func:`pirlygenes.pan_cancer_expression`'s narrower API with
    the legacy trufflepig kwarg surface — ``technical_rna_normalize``,
    ``remove_noncoding``, and ``renormalize_to_million`` compose
    pirlygenes' :func:`normalize_expression` and
    :func:`renormalize_to_million` primitives on top of the underlying
    pan-cancer + tcga-deconvolved-merged matrix.

    Result is cached in :data:`_PAN_CANCER_CACHE` keyed on every
    combination of arguments. Tests that need a clean slate clear the
    dict directly.
    """
    from pirlygenes.gene_sets_cancer import housekeeping_gene_ids

    from .expression_normalize import normalize_expression
    from .expression_normalize import (
        renormalize_to_million as _renormalize_to_million,
    )

    cache_key = _pan_cancer_cache_key(
        genes,
        normalize,
        log_transform,
        technical_rna_normalize,
        remove_noncoding,
        renormalize_to_million,
    )
    cached = _PAN_CANCER_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    df = get_reference_data("pan-cancer-expression")
    deconv_wide = _tcga_deconv_wide()
    if deconv_wide is not None:
        df = df.merge(deconv_wide, on="Symbol", how="left")
    value_cols = [
        c
        for c in df.columns
        if c.startswith("nTPM_") or c.startswith("FPKM_") or c.startswith("tcga_")
    ]
    if renormalize_to_million:
        df, _renorm_record = _renormalize_to_million(df, value_cols=value_cols)
        df.attrs["renormalize_to_million"] = _renorm_record
    if technical_rna_normalize or remove_noncoding:
        df, normalization_record = normalize_expression(
            df,
            label_col="Symbol",
            value_cols=value_cols,
            remove_noncoding=remove_noncoding,
        )
        df.attrs["technical_rna_normalization"] = normalization_record
    if genes is not None:
        genes_upper = {str(g).upper() for g in genes}
        mask = df["Ensembl_Gene_ID"].str.upper().isin(genes_upper) | df[
            "Symbol"
        ].str.upper().isin(genes_upper)
        df = df[mask]

    value_cols = [
        c
        for c in df.columns
        if c.startswith("nTPM_") or c.startswith("FPKM_") or c.startswith("tcga_")
    ]

    if normalize is not None:
        df = df.copy()
        if normalize == "percentile":
            for col in value_cols:
                vals = df[col].astype(float)
                df[col] = vals.rank(pct=True) * 100
        elif normalize == "housekeeping":
            hk_ids = housekeeping_gene_ids()
            hk_mask = df["Ensembl_Gene_ID"].isin(hk_ids)
            for col in value_cols:
                vals = df[col].astype(float)
                hk_median = vals[hk_mask].median()
                if not (np.isnan(hk_median) or hk_median <= 0):
                    df[col] = vals / hk_median
                else:
                    df[col] = np.nan
        else:
            raise ValueError(
                f"normalize must be 'percentile', 'housekeeping', or None, got {normalize!r}"
            )

    if log_transform:
        df = df.copy() if normalize is None else df
        for col in value_cols:
            df[col] = np.log2(df[col].astype(float) + 1)

    _PAN_CANCER_CACHE[cache_key] = df
    return df.copy()


def cancer_types():
    """Return the list of available TCGA cancer type codes (FPKM_ columns)."""
    df = get_reference_data("pan-cancer-expression")
    return sorted(c.replace("FPKM_", "") for c in df.columns if c.startswith("FPKM_"))


def cancer_expression(cancer_type, genes=None):
    """Expression for a single cancer type as a simple gene-level DataFrame.

    Returns columns ``Ensembl_Gene_ID``, ``Symbol``, and ``expression``
    (housekeeping-normalized, technical-RNA-normalized). Raises
    ``ValueError`` if ``cancer_type`` does not resolve to a TCGA code.
    """
    from pirlygenes.gene_sets_cancer import resolve_cancer_type

    code = resolve_cancer_type(cancer_type)
    df = pan_cancer_expression(
        genes=genes,
        normalize="housekeeping",
        technical_rna_normalize=True,
    )
    col = f"FPKM_{code}"
    return df[["Ensembl_Gene_ID", "Symbol", col]].rename(columns={col: "expression"})


def cancer_enriched_genes(cancer_type, min_fold=3.0, min_expression=0.01):
    """Genes enriched in one cancer type vs the median of all others."""
    from pirlygenes.gene_sets_cancer import resolve_cancer_type

    code = resolve_cancer_type(cancer_type)
    df = pan_cancer_expression(normalize="housekeeping", technical_rna_normalize=True)
    target_col = f"FPKM_{code}"
    if target_col not in df.columns:
        raise ValueError(f"Cancer type {cancer_type!r} → {code!r} not in pan-cancer reference")
    other_cols = [c for c in df.columns if c.startswith("FPKM_") and c != target_col]
    target = df[target_col].astype(float)
    other_median = df[other_cols].astype(float).median(axis=1)
    fold = target / other_median.replace(0, float("nan"))
    keep = (fold >= min_fold) & (target >= min_expression)
    out = df.loc[keep, ["Ensembl_Gene_ID", "Symbol", target_col]].copy()
    out = out.rename(columns={target_col: "expression"})
    out["fold_over_others"] = fold[keep].values
    return out.sort_values("fold_over_others", ascending=False)


def top_enriched_per_cancer_type(top_n=20, min_fold=3.0, min_expression=0.01):
    """Top-N enriched genes per TCGA code (dict keyed by code)."""
    out: dict[str, pd.DataFrame] = {}
    for code in cancer_types():
        enriched = cancer_enriched_genes(
            code,
            min_fold=min_fold,
            min_expression=min_expression,
        )
        out[code] = enriched.head(top_n)
    return out


# ---------- matched-normal expression panels ----------


def tumor_up_vs_matched_normal(cancer_code: str | None = None):
    """Solid-tumor genes dramatically up vs matched normal tissue.

    ``None`` when the CSV is not bundled (the bundled-with-pirlygenes
    case never fires this branch but it preserves the legacy contract).
    """
    try:
        df = get_reference_data("tumor-up-vs-matched-normal")
    except ValueError:
        return None
    if cancer_code:
        df = df[df["cancer_code"] == cancer_code]
    return df.copy()


def heme_tumor_up_vs_matched_normal(cancer_code: str | None = None):
    """Heme analogue of :func:`tumor_up_vs_matched_normal`."""
    try:
        df = get_reference_data("heme-tumor-up-vs-matched-normal")
    except ValueError:
        return None
    if cancer_code:
        df = df[df["cancer_code"] == cancer_code]
    return df.copy()


# ---------- supporting reference panels ----------


def hpa_cell_type_expression():
    """Per-(gene, cell-type) nTPM from the HPA single-cell consensus."""
    return get_reference_data("hpa-cell-type-expression")


def estimate_signatures():
    """ESTIMATE stromal / immune signature panels (Yoshihara 2013).

    Returns a frame with columns ``Symbol``, ``Ensembl_Gene_ID``,
    ``Category`` (``Stromal`` or ``Immune``).
    """
    return get_reference_data("estimate-signatures")
