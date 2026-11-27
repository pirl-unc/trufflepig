#!/usr/bin/env python3
"""A/B the pan-cancer signature ranker against learned, centroid, and fused calls.

This is intentionally an evaluation harness, not production logic. It runs the
same candidate ranker with alternate signature bases:

  - hk              cohort percentile in HK-normalized reference space
  - raw             cohort percentile in raw clean-TPM/nTPM reference space
  - within_sample   within-sample marker percentile only

It also reports the discriminative learned classifier top-1, whole-profile
centroid top-1, and the full fused report selector.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import warnings
from typing import Any

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_nohint_validation import full_granularity_call  # noqa: E402
from eval_per_sample_confusion import _clean_cohort, match_level  # noqa: E402
from pirlygenes.expression.accessors import available_representative_cohorts  # noqa: E402
from trufflepig.cancer_type_centroid import centroid_correlations  # noqa: E402
from trufflepig.expression_classifier import classify_expression  # noqa: E402
from trufflepig.plot_embedding import _compute_cancer_type_signature_stats  # noqa: E402
from trufflepig.tumor_purity import (  # noqa: E402
    _build_sample_tpm_by_symbol,
    rank_cancer_type_candidates,
)

BASES = ("hk", "raw", "within_sample")
METHODS = tuple(f"signature_{basis}" for basis in BASES) + (
    "learned_top1",
    "centroid_top1",
    "fused_selector",
)


def _sample_df(ensembl, symbols, values) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ensembl_gene_id": ensembl,
            "gene_symbol": symbols,
            "TPM": values,
        }
    )


def _ranker_call(df: pd.DataFrame, basis: str) -> str:
    stats = _compute_cancer_type_signature_stats(df, cohort_basis=basis)
    trace = rank_cancer_type_candidates(
        df,
        top_k=8,
        precomputed_signature_stats=stats,
    )
    if not trace:
        return ""
    top = trace[0]
    return str(top.get("winning_subtype") or top.get("code") or "")


def _learned_call(sample_tpm_by_symbol: dict[str, float]) -> str:
    predictions = classify_expression(sample_tpm_by_symbol, top_k=1)
    return str(predictions[0][0]) if predictions else ""


def _centroid_call(sample_tpm_by_symbol: dict[str, float]) -> str:
    corr = centroid_correlations(sample_tpm_by_symbol)
    return str(corr.index[0]) if len(corr) else ""


def _score_record(truth: str, sample: str, calls: dict[str, str]) -> dict[str, Any]:
    return {
        "truth": truth,
        "sample": sample,
        "calls": calls,
        "match_levels": {
            method: match_level(call, truth) for method, call in calls.items()
        },
    }


def _summarize(
    records: list[dict[str, Any]],
    methods: tuple[str, ...] = METHODS,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    n = len(records)
    for method in methods:
        counts = collections.Counter(
            row["match_levels"].get(method, "miss") for row in records
        )
        entity = counts["exact"] + counts["subtype"] + counts["sibling"]
        lineage = n - counts["miss"]
        out[method] = {
            "n": n,
            "exact": counts["exact"],
            "subtype": counts["subtype"],
            "sibling": counts["sibling"],
            "lineage": counts["lineage"],
            "miss": counts["miss"],
            "entity_compatible": entity,
            "lineage_compatible": lineage,
            "entity_rate": round(entity / n, 4) if n else 0.0,
            "lineage_rate": round(lineage / n, 4) if n else 0.0,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A/B pan-cancer signature ranker bases and fused evidence."
    )
    parser.add_argument("--types", nargs="*", default=None)
    parser.add_argument("--limit-types", type=int, default=None)
    parser.add_argument("--samples-per-type", type=int, default=None)
    parser.add_argument("--skip-fused", action="store_true")
    parser.add_argument("--jsonl", default="")
    args = parser.parse_args()

    types = sorted(args.types or available_representative_cohorts())
    if args.limit_types is not None:
        types = types[: args.limit_types]

    methods = METHODS if not args.skip_fused else METHODS[:-1]
    records: list[dict[str, Any]] = []
    out_handle = open(args.jsonl, "w") if args.jsonl else None
    try:
        for truth in types:
            ensembl, symbols, cleaned, sample_cols = _clean_cohort(truth)
            if args.samples_per_type is not None:
                sample_cols = sample_cols[: args.samples_per_type]
            for sample in sample_cols:
                df = _sample_df(ensembl, symbols, cleaned[sample].values)
                sample_tpm = _build_sample_tpm_by_symbol(df)
                calls = {
                    f"signature_{basis}": _ranker_call(df, basis)
                    for basis in BASES
                }
                calls["learned_top1"] = _learned_call(sample_tpm)
                calls["centroid_top1"] = _centroid_call(sample_tpm)
                if not args.skip_fused:
                    _bulk, fused = full_granularity_call(df)
                    calls["fused_selector"] = fused
                record = _score_record(truth, sample, calls)
                records.append(record)
                if out_handle is not None:
                    out_handle.write(json.dumps(record, sort_keys=True) + "\n")
                    out_handle.flush()
            print(f"[{len(records):4d}] {truth}", flush=True)
    finally:
        if out_handle is not None:
            out_handle.close()

    summary = _summarize(records, methods=methods)
    print("\nmethod                  exact entity lineage miss")
    print("------------------------------------------------")
    for method in methods:
        row = summary[method]
        print(
            f"{method:22s} {row['exact']:5d} "
            f"{row['entity_compatible']:6d} {row['lineage_compatible']:7d} "
            f"{row['miss']:4d}"
        )
    print(json.dumps({"summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
