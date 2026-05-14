"""Expression-backed reference data for the analysis engine.

This module owns the expression matrices (pan-cancer + 50 normal tissues,
TCGA deconvolved tumor-only TPM, subtype-deconvolved cohorts, HPA cell-type
expression, matched-normal tumor-up panels, stromal/immune signature panels)
and the typed Python accessors that the rest of the trufflepig pipeline
calls into.

Boundary (trufflepig#23):

    pirlygenes ─ curated gene knowledge: gene lists, panels, registry,
                 rule tables, gene-id/name helpers
    trufflepig ─ analysis engine + expression-backed reference models;
                 this module is the canonical entry point for the latter

External consumers should import from here, not from
``trufflepig/data/`` directly — the in-process cache, the deconv-wide
join into pan-cancer-expression, and the technical-RNA normalization
all live in the typed accessors.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Bundled reference data lives next to this module.
DATA_DIR: Path = Path(__file__).resolve().parent / "data"


# ---------- generic loader ----------


def get_reference_data(name: str) -> pd.DataFrame:
    """Load ``trufflepig/data/<name>.csv`` (or ``.csv.gz``).

    Mirrors :func:`pirlygenes.load_dataset.get_data` but scoped to the
    expression references that trufflepig now owns. Raises ``ValueError``
    if no matching file is bundled, so callers can route around
    optional cohort references that haven't been generated yet.
    """
    plain = DATA_DIR / f"{name}.csv"
    gz = DATA_DIR / f"{name}.csv.gz"
    # ``low_memory=False`` is required for ``subtype-deconvolved-
    # expression.csv.gz`` — pandas otherwise reads the file in chunks
    # and emits a ``DtypeWarning`` on the ``subtype`` column (mixed
    # types across releases). Loading the whole frame at once yields a
    # consistent dtype and is fine for these reference matrices.
    if plain.is_file():
        return pd.read_csv(plain, low_memory=False)
    if gz.is_file():
        return pd.read_csv(gz, low_memory=False)
    raise ValueError(f"No reference dataset named {name!r} in {DATA_DIR}")


# ---------- pan-cancer / TCGA-deconvolved expression ----------


_PAN_CANCER_CACHE: dict = {}


def _pan_cancer_cache_key(
    genes,
    normalize,
    log_transform,
    technical_rna_normalize,
    remove_noncoding,
):
    genes_key = None if genes is None else frozenset(str(g).upper() for g in genes)
    return (
        genes_key,
        normalize,
        bool(log_transform),
        bool(technical_rna_normalize),
        bool(remove_noncoding),
    )


def tcga_deconvolved_expression():
    """Per-(symbol, TCGA code) tumor-only TPM from the offline deconvolution.

    Long-form frame with columns ``symbol``, ``cancer_code``,
    ``tumor_tpm_median``, ``tumor_tpm_q1``, ``tumor_tpm_q3``, ``n_samples``.
    Returns ``None`` if the CSV is not bundled (fresh checkouts where
    the offline batch hasn't been run yet). Callers must handle ``None``.
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
):
    """Per-(cancer_code, subtype, symbol) tumor-only TPM from multi-cohort deconv.

    See :func:`pirlygenes.gene_sets_cancer.subtype_deconvolved_expression`
    (now removed) for the full cohort list. Long-form frame; ``None``
    when the CSV isn't bundled. ``technical_rna_normalize`` zeros
    mtDNA / rRNA-like rows per subtype group; ``remove_noncoding`` adds
    a biotype gate when a biotype column is present.
    """
    from .expression_normalize import normalize_expression, normalize_technical_rna_long_table

    try:
        df = get_reference_data("subtype-deconvolved-expression")
    except ValueError:
        return None
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
):
    """Expression across 50 normal tissues (nTPM) and 33 TCGA cancer types.

    Normal tissues from HPA v23 consensus nTPM. Cancer types from HPA
    (21 types, median FPKM) and GDC/STAR reprocessing (12 additional types,
    median TPM). Column names prefixed ``nTPM_`` or ``FPKM_``.

    When ``trufflepig/data/tcga-deconvolved-expression.csv.gz`` is
    present, tumor-only columns prefixed ``tcga_`` are merged in
    alongside the FPKM_ columns.

    Parameters mirror the pre-#23 pirlygenes signature so the rewrite
    is a pure import path change for analysis callers.
    """
    import numpy as np

    from pirlygenes.gene_sets_cancer import housekeeping_gene_ids
    from .expression_normalize import normalize_expression

    cache_key = _pan_cancer_cache_key(
        genes,
        normalize,
        log_transform,
        technical_rna_normalize,
        remove_noncoding,
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
    """Genes enriched in one cancer type vs the median of all others.

    Wraps :func:`pan_cancer_expression(normalize="housekeeping")` and
    returns the symbols above ``min_fold`` × the cross-cancer median
    with the cancer-type expression itself above ``min_expression``.
    """
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
    """Top-N enriched genes per TCGA code (dict keyed by code).

    Convenience wrapper: runs :func:`cancer_enriched_genes` for every
    TCGA code in :func:`cancer_types()` and trims to ``top_n`` rows each.
    """
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

    Per-cancer rows with ``fold_change_vs_matched_normal``, ``tumor_tpm``,
    ``matched_normal_ntpm``, and cross-tissue max columns. Built offline
    by racing :func:`tcga_deconvolved_expression` tumor-only medians
    against ``nTPM_<tissue>`` columns in :func:`pan_cancer_expression`.
    ``None`` when the CSV is not bundled.
    """
    try:
        df = get_reference_data("tumor-up-vs-matched-normal")
    except ValueError:
        return None
    if cancer_code:
        df = df[df["cancer_code"] == cancer_code]
    return df.copy()


def heme_tumor_up_vs_matched_normal(cancer_code: str | None = None):
    """Heme analogue of :func:`tumor_up_vs_matched_normal`.

    DLBC vs lymph_node, LAML vs bone_marrow; filter looser than the
    solid panel because heme tumors are themselves immune tissue.
    """
    try:
        df = get_reference_data("heme-tumor-up-vs-matched-normal")
    except ValueError:
        return None
    if cancer_code:
        df = df[df["cancer_code"] == cancer_code]
    return df.copy()


# ---------- supporting reference panels ----------


def hpa_cell_type_expression():
    """Per-(gene, cell-type) nTPM from the HPA single-cell consensus.

    Used by :mod:`trufflepig.decomposition.signature` to score lineage
    purity against curated cell-type reference profiles.
    """
    return get_reference_data("hpa-cell-type-expression")


def estimate_signatures():
    """ESTIMATE stromal / immune signature panels (Yoshihara 2013).

    Returns a frame with columns ``Symbol``, ``Ensembl_Gene_ID``,
    ``Category`` (``Stromal`` or ``Immune``). Used by
    :func:`trufflepig.tumor_purity.estimate_score` for the ESTIMATE
    purity surrogate alongside the lineage-panel estimator.
    """
    return get_reference_data("estimate-signatures")
