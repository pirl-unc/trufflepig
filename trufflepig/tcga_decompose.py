# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Offline TCGA per-sample deconvolution (#21).

Runs the pirlygenes decomposition engine on every TCGA sample in the
Xena TOIL RSEM TPM matrix, extracts the tumor-only TPM per gene, then
aggregates per TCGA cancer code to median + IQR + N. The resulting
CSV (``data/tcga-deconvolved-expression.csv``) feeds #22 — the
decon-derived tumor-only columns that replace the current HPA/FPKM
references in :func:`pirlygenes.gene_sets_cancer.pan_cancer_expression`.

This module is NOT imported by the package at runtime. It exists as
an offline batch script that the maintainer runs once per TCGA update.
Expect ~1.5 hours for ~10K samples on a modern laptop and ~5 GB RAM
to hold the full TPM matrix in memory.

Usage
-----

.. code-block:: shell

    python -m trufflepig.tcga_decompose \\
        --tpm-gz eval/tcga_RSEM_gene_tpm.gz \\
        --barcode-project-pkl eval/barcode_to_project.pkl \\
        --output-csv tcga-deconvolved-expression.csv

The output CSV is committed back to the `pirlygenes` data package
(``pirlygenes/data/tcga-deconvolved-expression.csv.gz``) — the
recomputation lives in trufflepig but the curated artifact ships
with the data package.

Smoke-test with a handful of samples::

    python -m trufflepig.tcga_decompose ... --max-samples-per-type 5
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def load_barcode_to_project(pkl_path: str | Path) -> dict[str, str]:
    """Return ``{patient_barcode: TCGA_code}`` from the Xena pickle.

    The Xena TCGA pickle is keyed by 12-char patient barcode
    (``TCGA-XX-YYYY``). TPM column headers are full sample barcodes
    (``TCGA-XX-YYYY-01``). We match on the 12-char prefix.
    """
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    if not isinstance(d, dict):
        raise TypeError(f"Expected dict in {pkl_path}, got {type(d)}")
    return {str(k): str(v) for k, v in d.items()}


def sample_barcode_to_project(
    sample_barcode: str,
    patient_to_project: dict[str, str],
) -> str | None:
    """Resolve a TCGA sample barcode to its TCGA project code.

    Returns ``None`` when the sample's patient barcode is not in the
    phenotype table (orphan or recently-added sample).
    """
    # Patient barcode is the first 12 chars: TCGA-XX-YYYY.
    if not sample_barcode.startswith("TCGA-"):
        return None
    patient = sample_barcode[:12]
    return patient_to_project.get(patient)


def is_primary_tumor_sample(sample_barcode: str) -> bool:
    """Return True for primary-tumor sample codes.

    TCGA sample-type codes 01, 03, 09 are solid tumor, primary blood
    cancer, and primary blood-derived cancer respectively. We skip
    normals (10/11/14) and metastases (06/07) so the reference reflects
    primary-tumor expression.
    """
    parts = sample_barcode.split("-")
    if len(parts) < 4:
        return False
    # Vial suffix may be present (e.g. ``01A``); strip anything non-numeric.
    code = parts[3][:2]
    return code in {"01", "03", "09"}


def load_tcga_tpm_matrix(
    tpm_gz_path: str | Path,
    sample_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the Xena TOIL TCGA TPM matrix, un-logged to linear TPM.

    Xena TOIL stores ``log2(TPM + 0.001)``. We reverse the transform
    so decomposition operates on linear TPM like every other caller
    in the package.

    Parameters
    ----------
    tpm_gz_path : path
        Gzip TSV with gene rows (Ensembl IDs with version) and sample
        columns.
    sample_columns : list of str, optional
        Restrict to this subset of sample barcodes. Useful for
        smoke-testing without reading the full 740 MB file.

    Returns
    -------
    pd.DataFrame
        Index = bare Ensembl ID (version stripped). Columns = sample
        barcodes. Values = linear TPM, float32.
    """
    usecols = None
    if sample_columns is not None:
        usecols = ["sample"] + list(sample_columns)
    print(f"[tcga] Reading {tpm_gz_path} ...", flush=True)
    t0 = time.time()
    df = pd.read_csv(
        tpm_gz_path,
        sep="\t",
        index_col=0,
        usecols=usecols,
        dtype={"sample": str},
        compression="gzip",
    )
    print(
        f"[tcga] Loaded {df.shape[0]} genes x {df.shape[1]} samples "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )
    # Un-log: TPM = 2^x - 0.001, clip negatives to zero.
    tpm = np.power(2.0, df.to_numpy(dtype=np.float32)) - np.float32(0.001)
    np.maximum(tpm, 0.0, out=tpm)
    out = pd.DataFrame(tpm, index=df.index, columns=df.columns)
    out.index = out.index.astype(str).str.split(".", n=1).str[0]
    return out


def sample_frame(
    tpm_column: pd.Series,
    versioned_ids: pd.Index,
) -> pd.DataFrame:
    """Build the per-sample DataFrame that ``decompose_sample`` expects.

    ``_guess_gene_cols`` recognises ``gene_id`` (lower-case). We keep
    the versioned ID there because pirlygenes's loader strips the
    version internally.
    """
    # ``_guess_gene_cols`` requires a gene-name column too. Leave it
    # empty — downstream resolution rebuilds the symbol from the pan-
    # cancer reference's ``Ensembl_Gene_ID`` → ``Symbol`` map.
    return pd.DataFrame(
        {
            "gene_id": versioned_ids,
            "gene_name": "",
            "TPM": tpm_column.to_numpy(dtype=np.float32),
        }
    )


def _observed_as_tumor(
    tpm_column: pd.Series,
    versioned_ids: pd.Index,
) -> pd.DataFrame:
    """Build a ``[symbol, tumor_tpm]`` frame using the observed TPM directly.

    Used when the decomposer reports ≥99.9% tumor fraction — in that
    degenerate case it returns an empty ``gene_attribution`` because
    no TME compartment is active, but the conservative interpretation
    for a primary-tumor reference is that the observed TPM *is* the
    tumor TPM.
    """
    from pirlygenes.gene_sets_cancer import pan_cancer_expression

    ref = pan_cancer_expression()[["Ensembl_Gene_ID", "Symbol"]].drop_duplicates(
        subset="Ensembl_Gene_ID"
    )
    eid_to_symbol = dict(zip(ref["Ensembl_Gene_ID"], ref["Symbol"]))
    bare_ids = pd.Index(versioned_ids).astype(str).str.split(".", n=1).str[0]
    symbols = [eid_to_symbol.get(eid, "") for eid in bare_ids]
    out = pd.DataFrame(
        {
            "symbol": symbols,
            "tumor_tpm": tpm_column.to_numpy(dtype=float),
        }
    )
    out = out[out["symbol"].astype(str).str.len() > 0]
    out = out[out["tumor_tpm"] >= 0.01]
    # Collapse duplicate symbols (mapped from multiple ENSGs): sum TPM.
    return out.groupby("symbol", as_index=False, sort=False)["tumor_tpm"].sum()


def decompose_one_sample(
    sample_barcode: str,
    tpm_column: pd.Series,
    cancer_code: str,
    versioned_ids: pd.Index,
    sample_mode: str = "solid",
) -> pd.DataFrame | None:
    """Decompose one TCGA sample and return ``[symbol, tumor_tpm]``.

    Returns ``None`` when the decomposition throws or returns no
    candidate (rare — a nearly-empty expression vector). When the
    best hypothesis collapses to ≥99.9% tumor (and therefore empty
    ``gene_attribution``), we fall back to the observed TPM as the
    tumor TPM — correct for TCGA primary-tumor samples that are
    already tumor-enriched by the consortium's pathology review.
    """
    from trufflepig.decomposition import decompose_sample

    df = sample_frame(tpm_column, versioned_ids)
    try:
        results = decompose_sample(
            df,
            cancer_types=[cancer_code],
            top_k=1,
            sample_mode=sample_mode,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[tcga] {sample_barcode} ({cancer_code}): decompose failed — {exc}",
            flush=True,
        )
        return None
    if not results:
        return None
    best = results[0]
    if best.gene_attribution.empty:
        if best.purity >= 0.999:
            attr = _observed_as_tumor(tpm_column, versioned_ids)
        else:
            return None
    else:
        attr = best.gene_attribution[["symbol", "tumor"]].copy()
        attr = attr.rename(columns={"tumor": "tumor_tpm"})
    attr["sample"] = sample_barcode
    attr["cancer_code"] = cancer_code
    return attr


def aggregate_per_type(
    per_sample_rows: pd.DataFrame,
    group_by_subtype: bool = False,
) -> pd.DataFrame:
    """Reduce per-(sample, symbol) tumor TPM to per-(cancer_code[, subtype], symbol) stats.

    When ``group_by_subtype=True`` the input must carry a ``subtype``
    column and the output grows a ``subtype`` column — samples with a
    blank subtype are dropped so each row represents a coherent
    sub-cohort rather than a mix of annotated + unannotated samples.

    Output columns: ``symbol``, ``cancer_code``, [``subtype``,]
    ``tumor_tpm_median``, ``tumor_tpm_q1``, ``tumor_tpm_q3``, ``n_samples``.
    """
    base_cols = [
        "symbol",
        "cancer_code",
        "tumor_tpm_median",
        "tumor_tpm_q1",
        "tumor_tpm_q3",
        "n_samples",
    ]
    cols = base_cols[:2] + (["subtype"] if group_by_subtype else []) + base_cols[2:]

    if per_sample_rows.empty:
        return pd.DataFrame(columns=cols)

    df = per_sample_rows
    group_cols = ["cancer_code", "symbol"]
    if group_by_subtype:
        if "subtype" not in df.columns:
            raise ValueError(
                "group_by_subtype=True but per_sample_rows lacks a 'subtype' column"
            )
        df = df[df["subtype"].astype(str).str.len() > 0]
        if df.empty:
            return pd.DataFrame(columns=cols)
        group_cols = ["cancer_code", "subtype", "symbol"]

    grouped = df.groupby(group_cols)["tumor_tpm"]
    summary = grouped.agg(
        tumor_tpm_median="median",
        tumor_tpm_q1=lambda s: float(np.quantile(s, 0.25)),
        tumor_tpm_q3=lambda s: float(np.quantile(s, 0.75)),
        n_samples="count",
    ).reset_index()
    return summary[cols]


def select_samples(
    columns: pd.Index,
    patient_to_project: dict[str, str],
    cancer_types: list[str] | None = None,
    max_samples_per_type: int | None = None,
    primary_only: bool = True,
) -> list[tuple[str, str]]:
    """Return ``[(sample_barcode, cancer_code), ...]`` to process.

    Handles the three knobs a caller cares about: primary-tumor filter,
    cancer-type filter, and per-type sample cap.
    """
    by_type: dict[str, list[str]] = defaultdict(list)
    for col in columns:
        if primary_only and not is_primary_tumor_sample(col):
            continue
        code = sample_barcode_to_project(col, patient_to_project)
        if code is None:
            continue
        if cancer_types is not None and code not in cancer_types:
            continue
        by_type[code].append(col)

    selected: list[tuple[str, str]] = []
    for code, samples in by_type.items():
        if max_samples_per_type is not None:
            samples = samples[:max_samples_per_type]
        selected.extend((s, code) for s in samples)
    return selected


def load_subtype_map(path: str | Path) -> dict[str, str]:
    """Return ``{patient_barcode: subtype}`` from a 2-col CSV (no header).

    Keys are 12-char patient barcodes (``TCGA-XX-YYYY``). Values are
    free-form subtype strings (``BRCA_LumA``, ``panNET_G2``, ``apl``…).
    Samples whose patient barcode is not in the map are tagged with an
    empty subtype and later dropped by :func:`aggregate_per_type` under
    ``group_by_subtype=True``.
    """
    df = pd.read_csv(path, header=None, names=["patient", "subtype"])
    return dict(zip(df["patient"].astype(str), df["subtype"].astype(str)))


def run(
    tpm_gz: str,
    barcode_project_pkl: str,
    output_csv: str,
    cancer_types: list[str] | None = None,
    max_samples_per_type: int | None = None,
    checkpoint_every: int = 500,
    subtype_map: dict[str, str] | None = None,
) -> None:
    patient_to_project = load_barcode_to_project(barcode_project_pkl)

    print(
        f"[tcga] {len(patient_to_project)} patients mapped to TCGA projects", flush=True
    )

    if max_samples_per_type is not None or cancer_types is not None:
        # Cheap dry path: peek at header only to subset before full load.
        header = pd.read_csv(tpm_gz, sep="\t", nrows=0, compression="gzip")
        sample_cols = [c for c in header.columns if c != "sample"]
        pairs = select_samples(
            pd.Index(sample_cols),
            patient_to_project,
            cancer_types=cancer_types,
            max_samples_per_type=max_samples_per_type,
        )
        subset = [p[0] for p in pairs]
        tpm = load_tcga_tpm_matrix(tpm_gz, sample_columns=subset)
    else:
        tpm = load_tcga_tpm_matrix(tpm_gz)
        pairs = select_samples(tpm.columns, patient_to_project)

    print(f"[tcga] Decomposing {len(pairs)} samples", flush=True)
    versioned_ids = pd.Index(tpm.index).astype(str)
    group_by_subtype = subtype_map is not None

    checkpoint_path = Path(output_csv).with_suffix(".partial.csv")
    accum: list[pd.DataFrame] = []
    t0 = time.time()
    for i, (sample, code) in enumerate(pairs, start=1):
        col = tpm[sample]
        attr = decompose_one_sample(sample, col, code, versioned_ids)
        if attr is not None:
            if group_by_subtype:
                attr["subtype"] = subtype_map.get(sample[:12], "")
            accum.append(attr)
        if i % checkpoint_every == 0:
            elapsed = time.time() - t0
            per = elapsed / i
            eta = per * (len(pairs) - i)
            print(
                f"[tcga] {i}/{len(pairs)} "
                f"({elapsed:.0f}s elapsed, {per:.2f}s/sample, ETA {eta / 60:.1f} min)",
                flush=True,
            )
            pd.concat(accum, ignore_index=True).to_csv(checkpoint_path, index=False)

    if not accum:
        print(
            "[tcga] No samples decomposed successfully — nothing to write", flush=True
        )
        return

    per_sample = pd.concat(accum, ignore_index=True)
    summary = aggregate_per_type(per_sample, group_by_subtype=group_by_subtype)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    print(
        f"[tcga] Wrote {len(summary)} (cancer_code, symbol) rows to {output_csv}",
        flush=True,
    )
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TCGA offline per-sample tumor deconvolution (#21).",
    )
    p.add_argument("--tpm-gz", required=True, help="Path to tcga_RSEM_gene_tpm.gz")
    p.add_argument(
        "--barcode-project-pkl",
        required=True,
        help="Path to barcode_to_project.pkl",
    )
    p.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV path (e.g. pirlygenes/data/tcga-deconvolved-expression.csv)",
    )
    p.add_argument(
        "--cancer-types",
        default=None,
        help="Comma-separated TCGA codes to restrict to (default: all)",
    )
    p.add_argument(
        "--max-samples-per-type",
        type=int,
        default=None,
        help="Cap per-type sample count (default: all). Useful for smoke tests.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Write a .partial.csv every N processed samples",
    )
    p.add_argument(
        "--subtype-map",
        default=None,
        help=(
            "Optional 2-column CSV (no header): patient_barcode,subtype. "
            "When provided, output is aggregated per (cancer_code, subtype, symbol) "
            "instead of per (cancer_code, symbol). Samples whose patient is not "
            "in the map are dropped."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cancer_types = None
    if args.cancer_types:
        cancer_types = [c.strip() for c in args.cancer_types.split(",") if c.strip()]
    subtype_map = load_subtype_map(args.subtype_map) if args.subtype_map else None
    run(
        tpm_gz=args.tpm_gz,
        barcode_project_pkl=args.barcode_project_pkl,
        output_csv=args.output_csv,
        cancer_types=cancer_types,
        max_samples_per_type=args.max_samples_per_type,
        checkpoint_every=args.checkpoint_every,
        subtype_map=subtype_map,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
