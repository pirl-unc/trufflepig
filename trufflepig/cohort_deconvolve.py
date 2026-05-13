# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Non-TCGA cohort deconvolution / summarisation (#23 companion, v4.0 era).

Where :mod:`pirlygenes.tcga_decompose` is hard-wired to the Xena TOIL
TCGA TPM matrix + barcode-to-project pickle, this module handles the
rest of the world — cBioPortal datahub cohorts, GEO series matrices,
ICGC exports — anything that arrives as a (gene_index, sample_column)
expression matrix with some ancillary per-sample subtype file.

Shape:
1. Read the expression matrix (cBioPortal style: Hugo_Symbol + Entrez_Gene_Id
   header, then one column per sample). Normalise to TPM.
2. Optionally attach subtype labels per sample.
3. For each sample, either:
   - run full pirlygenes deconvolution (solid-primary templates) —
     appropriate for solid-tumour cohorts with TME admixture;
   - OR, in ``high_purity_passthrough`` mode, treat observed TPM as
     tumor TPM directly. This is correct for sorted heme malignancies
     (AML peripheral blood is ~90%+ malignant clone by construction)
     and avoids forcing the solid-primary TME templates onto samples
     that don't fit them.
4. Aggregate per (cohort_code, subtype, symbol) to median + IQR + N,
   output in the same schema as ``tcga-deconvolved-expression.csv.gz``.

Not imported at runtime; invoked via ``python -m pirlygenes.cohort_deconvolve``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _normalise_gene_column(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Return a copy with the gene-symbol column named ``symbol``."""
    for candidate in ("symbol", "Gene", "Hugo_Symbol"):
        if candidate in df.columns:
            if candidate == "symbol":
                return df.copy()
            return df.rename(columns={candidate: "symbol"})
    raise ValueError(f"Expected one of symbol/Gene/Hugo_Symbol columns in {path}")


def validate_tpm_sample_sums(
    tpm_frame: pd.DataFrame,
    *,
    expected_total: float = 1_000_000.0,
    tolerance_fraction: float = 0.01,
) -> pd.DataFrame:
    """Validate that sample TPM columns approximately sum to one million.

    This catches the most common scale mistake when importing public
    matrices: accidentally treating log2(TPM+1), CPM, counts, or already
    filtered gene subsets as full TPM. The returned frame is useful for
    logging; a ``ValueError`` is raised when any non-empty sample falls
    outside ``expected_total ± tolerance_fraction``.
    """
    sample_cols = [c for c in tpm_frame.columns if c != "symbol"]
    if not sample_cols:
        return pd.DataFrame(columns=["sample", "total_tpm", "fraction_of_expected"])

    sums = tpm_frame[sample_cols].sum(axis=0).astype(float)
    lower = expected_total * (1.0 - tolerance_fraction)
    upper = expected_total * (1.0 + tolerance_fraction)
    bad = sums[(sums < lower) | (sums > upper)]
    if not bad.empty:
        details = ", ".join(f"{sample}={total:.0f}" for sample, total in bad.head(8).items())
        if len(bad) > 8:
            details += f", ... (+{len(bad) - 8})"
        raise ValueError(
            "TPM sample-sum QC failed: expected each sample to sum to "
            f"{expected_total:.0f}±{tolerance_fraction:.0%}; bad samples: {details}"
        )

    return pd.DataFrame(
        {
            "sample": sums.index.astype(str),
            "total_tpm": sums.values,
            "fraction_of_expected": sums.values / expected_total,
        }
    )


def load_cbioportal_rpkm(path: str | Path) -> pd.DataFrame:
    """Load a cBioPortal-datahub ``data_mrna_seq_rpkm.txt`` file.

    The format is::

        Hugo_Symbol   Entrez_Gene_Id   SAMPLE_1   SAMPLE_2   ...
        TSPAN6        7105             0.098      0.516       ...

    Returns a frame with a ``symbol`` column and one column per sample
    holding RPKM values. Duplicate symbols are collapsed by summing.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if "Hugo_Symbol" not in df.columns:
        raise ValueError(f"Expected Hugo_Symbol column in {path}")
    df = df.rename(columns={"Hugo_Symbol": "symbol"})
    sample_cols = [c for c in df.columns if c not in ("symbol", "Entrez_Gene_Id")]
    df[sample_cols] = df[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df = df[df["symbol"].notna() & (df["symbol"].astype(str).str.len() > 0)]
    df = df.groupby("symbol", as_index=False, sort=False)[sample_cols].sum()
    return df


def load_log2_tpm(
    path: str | Path,
    sample_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load a gene × sample matrix encoded as log2(TPM+1).

    Treehouse public compendium matrices use this representation. We
    invert it back to linear TPM, collapse duplicate gene symbols by
    summing, then run sample-sum QC so sparse/imported references do not
    silently enter the bundled summaries on the wrong scale.
    """
    sample_id_set = set(sample_ids or [])
    if sample_id_set:
        keep = sample_id_set | {"symbol", "Gene", "Hugo_Symbol"}
        df = pd.read_csv(
            path,
            sep="\t",
            usecols=lambda column: column in keep,
            low_memory=False,
        )
    else:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    df = _normalise_gene_column(df, path)
    if sample_id_set:
        missing = sample_id_set - {c for c in df.columns if c != "symbol"}
        if missing:
            preview = ", ".join(sorted(missing)[:8])
            if len(missing) > 8:
                preview += f", ... (+{len(missing) - 8})"
            raise ValueError(f"Missing requested sample columns in {path}: {preview}")
    sample_cols = [c for c in df.columns if c != "symbol"]
    df[sample_cols] = df[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df[sample_cols] = np.power(2.0, df[sample_cols]) - 1.0
    df = df[df["symbol"].notna() & (df["symbol"].astype(str).str.len() > 0)]
    df = df.groupby("symbol", as_index=False, sort=False)[sample_cols].sum()
    validate_tpm_sample_sums(df)
    return df


def rpkm_to_tpm(rpkm_frame: pd.DataFrame) -> pd.DataFrame:
    """Convert per-sample RPKM columns to TPM in-place-style (returns new frame).

    TPM = RPKM / sum(RPKM) * 1e6, per sample. Assumes the first
    column is ``symbol`` (or other non-numeric ID) and every other
    column is a sample's RPKM vector.
    """
    out = rpkm_frame.copy()
    sample_cols = [c for c in out.columns if c != "symbol"]
    totals = out[sample_cols].sum(axis=0)
    totals = totals.replace(0, np.nan)
    out[sample_cols] = out[sample_cols].divide(totals, axis=1) * 1_000_000
    out[sample_cols] = out[sample_cols].fillna(0.0)
    return out


def load_subtype_labels(path: str | Path) -> dict[str, str]:
    """Load a 2-column CSV (no header): ``sample_id,subtype``."""
    df = pd.read_csv(path, header=None, names=["sample", "subtype"])
    return dict(zip(df["sample"].astype(str), df["subtype"].astype(str)))


def summarise_passthrough(
    tpm_frame: pd.DataFrame,
    cohort_code: str,
    subtype_map: dict[str, str] | None = None,
    min_tpm: float = 0.01,
) -> pd.DataFrame:
    """Aggregate the per-sample TPM table without running deconvolution.

    Use this mode when deconvolution doesn't make biological sense —
    e.g. sorted heme malignancies where TME templates would
    mis-attribute malignant clone signal to normal immune components.
    Each sample's observed TPM becomes its tumor TPM; we then take the
    median / IQR across samples within each (cohort_code, subtype).

    Returns long-form frame with columns: symbol, cancer_code,
    [subtype,] tumor_tpm_median, tumor_tpm_q1, tumor_tpm_q3, n_samples.
    """
    sample_cols = [c for c in tpm_frame.columns if c != "symbol"]
    # Drop near-zero-expression rows across every sample to keep the
    # aggregate compact — low-TPM genes are noise anyway.
    row_max = tpm_frame[sample_cols].max(axis=1)
    dense = tpm_frame[row_max >= min_tpm].copy()

    melted = dense.melt(
        id_vars="symbol",
        value_vars=sample_cols,
        var_name="sample",
        value_name="tumor_tpm",
    )
    melted["cancer_code"] = cohort_code

    has_subtype = subtype_map is not None
    if has_subtype:
        melted["subtype"] = melted["sample"].map(subtype_map).fillna("")
        melted = melted[melted["subtype"].astype(str).str.len() > 0]
        group_cols = ["cancer_code", "subtype", "symbol"]
    else:
        group_cols = ["cancer_code", "symbol"]

    if melted.empty:
        cols = ["symbol", "cancer_code"]
        if has_subtype:
            cols.append("subtype")
        cols += ["tumor_tpm_median", "tumor_tpm_q1", "tumor_tpm_q3", "n_samples"]
        return pd.DataFrame(columns=cols)

    grouped = melted.groupby(group_cols)["tumor_tpm"]
    summary = grouped.agg(
        tumor_tpm_median="median",
        tumor_tpm_q1=lambda s: float(np.quantile(s, 0.25)),
        tumor_tpm_q3=lambda s: float(np.quantile(s, 0.75)),
        n_samples="count",
    ).reset_index()

    cols = ["symbol", "cancer_code"]
    if has_subtype:
        cols.append("subtype")
    cols += ["tumor_tpm_median", "tumor_tpm_q1", "tumor_tpm_q3", "n_samples"]
    return summary[cols]


def run(
    expression_path: str,
    cohort_code: str,
    output_csv: str,
    subtype_labels_csv: str | None = None,
    input_format: str = "cbioportal_rpkm",
    passthrough: bool = True,
    min_tpm: float = 0.01,
) -> None:
    t0 = time.time()
    if input_format == "cbioportal_rpkm":
        rpkm = load_cbioportal_rpkm(expression_path)
        tpm = rpkm_to_tpm(rpkm)
    elif input_format == "tpm_tsv":
        tpm = _normalise_gene_column(
            pd.read_csv(expression_path, sep="\t", low_memory=False),
            expression_path,
        )
    elif input_format == "log2_tpm_tsv":
        tpm = load_log2_tpm(expression_path)
    else:
        raise ValueError(f"Unknown input_format={input_format!r}")

    print(
        f"[cohort] {cohort_code}: loaded {tpm.shape[0]} genes x "
        f"{tpm.shape[1] - 1} samples in {time.time() - t0:.1f}s",
        flush=True,
    )

    subtype_map = (
        load_subtype_labels(subtype_labels_csv) if subtype_labels_csv else None
    )
    if subtype_map:
        print(f"[cohort] {cohort_code}: {len(subtype_map)} subtype labels", flush=True)

    if not passthrough:
        raise NotImplementedError(
            "Full pirlygenes deconv on non-TCGA cohorts is not wired yet — "
            "the HGNC→Ensembl mapping + template selection for cohort-specific "
            "sample modes (heme/sorted/FFPE) is a follow-up. Use "
            "passthrough=True for now; documented in the module docstring."
        )

    summary = summarise_passthrough(
        tpm,
        cohort_code,
        subtype_map=subtype_map,
        min_tpm=min_tpm,
    )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    print(
        f"[cohort] {cohort_code}: wrote {len(summary)} summary rows to {output_csv}",
        flush=True,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Non-TCGA cohort deconvolution / summarisation.",
    )
    p.add_argument(
        "--expression-path",
        required=True,
        help="Path to the cohort's expression matrix",
    )
    p.add_argument(
        "--cohort-code",
        required=True,
        help="Short code for this cohort (e.g. BEATAML, TARGET_NBL)",
    )
    p.add_argument("--output-csv", required=True, help="Output CSV path")
    p.add_argument(
        "--subtype-labels",
        default=None,
        help="Optional 2-col CSV (no header): sample_id,subtype",
    )
    p.add_argument(
        "--input-format",
        default="cbioportal_rpkm",
        choices=["cbioportal_rpkm", "tpm_tsv", "log2_tpm_tsv"],
        help="How to parse the expression matrix",
    )
    p.add_argument(
        "--passthrough",
        action="store_true",
        default=True,
        help="Treat observed TPM as tumor TPM (default). "
        "Appropriate for high-purity heme malignancies.",
    )
    p.add_argument(
        "--min-tpm",
        type=float,
        default=0.01,
        help="Drop genes whose max TPM across samples is below this.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    run(
        expression_path=args.expression_path,
        cohort_code=args.cohort_code,
        output_csv=args.output_csv,
        subtype_labels_csv=args.subtype_labels,
        input_format=args.input_format,
        passthrough=args.passthrough,
        min_tpm=args.min_tpm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
