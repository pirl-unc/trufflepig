#!/usr/bin/env python3
"""Leakage-check the optional learned expression classifier on representative cohorts.

Default split is deterministic per cohort:

  - train on up to 3 samples per cohort;
  - test on the next up to 2 samples per cohort;
  - for cohorts with fewer than 5 samples, keep at least one training sample.

The scaler, high-variance gene selection, and logistic model are fit on the
training split only. Medoid evaluation uses that same trained model.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import warnings
from dataclasses import dataclass
from typing import Iterable

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from oncoref.normalization import clean_tpm
from pirlygenes.expression.accessors import (
    available_representative_cohorts,
    representative_cohort_samples,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval_per_sample_confusion import match_level


@dataclass(frozen=True)
class SplitData:
    clean_tpm: pd.DataFrame
    labels: dict[str, str]
    per_type: dict[str, list[str]]


def _load_representative_samples(types: Iterable[str]) -> SplitData:
    frames: list[pd.DataFrame] = []
    labels: dict[str, str] = {}
    per_type: dict[str, list[str]] = {}
    base_symbols: pd.Series | None = None
    for code in types:
        raw = (
            representative_cohort_samples(code)
            .drop_duplicates("Ensembl_Gene_ID")
            .set_index("Ensembl_Gene_ID")
        )
        sample_cols = [col for col in raw.columns if col != "Symbol"]
        if base_symbols is None:
            base_symbols = raw["Symbol"].copy()
        renamed = [f"{code}::{i + 1}::{col}" for i, col in enumerate(sample_cols)]
        frame = raw[sample_cols].astype(float).copy()
        frame.columns = renamed
        frames.append(frame)
        per_type[code] = renamed
        labels.update({col: code for col in renamed})

    expr = pd.concat(frames, axis=1).dropna(how="any")
    if base_symbols is None:
        raise RuntimeError("no representative cohorts loaded")
    gene_table = pd.DataFrame(
        {
            "Ensembl_Gene_ID": expr.index,
            "Symbol": base_symbols.reindex(expr.index).values,
        }
    )
    cleaned = clean_tpm(expr.astype(float), gene_table=gene_table)
    cleaned.index = base_symbols.reindex(expr.index).values
    cleaned = cleaned[~cleaned.index.duplicated(keep="first")]
    return SplitData(clean_tpm=cleaned, labels=labels, per_type=per_type)


def _split_columns(
    per_type: dict[str, list[str]],
    *,
    train_per_type: int,
    test_per_type: int,
    seed: int,
    shuffle: bool,
) -> tuple[list[str], list[str], list[str]]:
    rng = np.random.default_rng(seed)
    train: list[str] = []
    test: list[str] = []
    unused: list[str] = []
    for code in sorted(per_type):
        cols = list(per_type[code])
        if shuffle:
            rng.shuffle(cols)
        if len(cols) <= 1:
            train.extend(cols)
            continue
        n_train = min(train_per_type, len(cols) - 1)
        n_test = min(test_per_type, len(cols) - n_train)
        train.extend(cols[:n_train])
        test.extend(cols[n_train : n_train + n_test])
        unused.extend(cols[n_train + n_test :])
    return train, test, unused


def _fit_model(
    clean_tpm_matrix: pd.DataFrame,
    labels: dict[str, str],
    train_cols: list[str],
    *,
    n_genes: int,
):
    log_tpm = np.log1p(clean_tpm_matrix.clip(lower=0))
    variance = log_tpm[train_cols].var(axis=1)
    genes = list(variance.sort_values(ascending=False).index[:n_genes])
    x_train = np.nan_to_num(
        log_tpm.loc[genes, train_cols].T.to_numpy(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    y_train = np.asarray([labels[col] for col in train_cols])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
    )
    model.fit(x_train, y_train)
    return model, genes


def _predict_matrix(model, genes: list[str], clean_tpm_matrix: pd.DataFrame, cols: list[str], top_k: int):
    if not cols:
        return {}
    log_tpm = np.log1p(clean_tpm_matrix.clip(lower=0))
    x = np.nan_to_num(
        log_tpm.loc[genes, cols].T.to_numpy(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    probs = model.predict_proba(x)
    classes = list(model.classes_)
    out = {}
    for col, row in zip(cols, probs, strict=True):
        order = np.argsort(row)[::-1][:top_k]
        out[col] = [(str(classes[i]), float(row[i])) for i in order]
    return out


def _predict_medoids(
    model,
    genes: list[str],
    clean_tpm_matrix: pd.DataFrame,
    per_type: dict[str, list[str]],
    *,
    columns_by_type: dict[str, list[str]] | None,
    top_k: int,
):
    rows: list[np.ndarray] = []
    codes: list[str] = []
    for code in sorted(per_type):
        cols = columns_by_type.get(code, []) if columns_by_type is not None else per_type[code]
        if not cols:
            continue
        medoid_clean = clean_tpm_matrix.loc[genes, cols].mean(axis=1)
        rows.append(np.log1p(medoid_clean.clip(lower=0)).to_numpy())
        codes.append(code)
    if not rows:
        return {}
    probs = model.predict_proba(np.nan_to_num(np.vstack(rows), nan=0.0, posinf=0.0, neginf=0.0))
    classes = list(model.classes_)
    out = {}
    for code, row in zip(codes, probs, strict=True):
        order = np.argsort(row)[::-1][:top_k]
        out[code] = [(str(classes[i]), float(row[i])) for i in order]
    return out


def _records_for_predictions(labels: dict[str, str], predictions: dict[str, list[tuple[str, float]]]):
    records = []
    for sample_id in sorted(predictions):
        truth = labels[sample_id]
        top = predictions[sample_id]
        call = top[0][0] if top else ""
        prob = float(top[0][1]) if top else 0.0
        second = float(top[1][1]) if len(top) > 1 else 0.0
        records.append(
            {
                "sample": sample_id,
                "truth": truth,
                "call": call,
                "probability": prob,
                "margin": prob - second,
                "match_level": match_level(call, truth),
                "top_predictions": [
                    {"code": code, "probability": round(float(p), 6)}
                    for code, p in top
                ],
            }
        )
    return records


def _records_for_medoids(predictions: dict[str, list[tuple[str, float]]], *, split: str):
    records = []
    for truth in sorted(predictions):
        top = predictions[truth]
        call = top[0][0] if top else ""
        prob = float(top[0][1]) if top else 0.0
        second = float(top[1][1]) if len(top) > 1 else 0.0
        records.append(
            {
                "sample": f"{split}:{truth}",
                "truth": truth,
                "call": call,
                "probability": prob,
                "margin": prob - second,
                "match_level": match_level(call, truth),
                "top_predictions": [
                    {"code": code, "probability": round(float(p), 6)}
                    for code, p in top
                ],
            }
        )
    return records


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    levels = collections.Counter(str(row["match_level"]) for row in records)
    n = len(records)
    compatible = levels["exact"] + levels["subtype"] + levels["sibling"]
    top3_exact = 0
    top3_compatible = 0
    for row in records:
        truth = str(row["truth"])
        top_codes = [str(item["code"]) for item in row["top_predictions"][:3]]
        if truth in top_codes:
            top3_exact += 1
        if any(match_level(code, truth) in {"exact", "subtype", "sibling"} for code in top_codes):
            top3_compatible += 1
    return {
        "n": n,
        "levels": {level: levels[level] for level in ("exact", "subtype", "sibling", "lineage", "miss")},
        "entity_compatible": compatible,
        "lineage_compatible": n - levels["miss"],
        "top3_exact": top3_exact,
        "top3_entity_compatible": top3_compatible,
    }


def _print_summary(name: str, records: list[dict[str, object]]) -> None:
    summary = _summary(records)
    n = int(summary["n"])
    print(f"\n==== {name}: {n} samples ====")
    for level, count in summary["levels"].items():
        pct = 100.0 * int(count) / n if n else 0.0
        print(f"  {level:8s}: {count:4d} ({pct:5.1f}%)")
    for key, label in (
        ("entity_compatible", "entity-compatible"),
        ("lineage_compatible", "lineage-compatible"),
        ("top3_exact", "top-3 exact"),
        ("top3_entity_compatible", "top-3 entity-compatible"),
    ):
        count = int(summary[key])
        pct = 100.0 * count / n if n else 0.0
        print(f"  {label:24s}: {count:4d}/{n} ({pct:5.1f}%)")


def _print_misses(name: str, records: list[dict[str, object]], *, limit: int) -> None:
    bad = [row for row in records if row["match_level"] in {"lineage", "miss"}]
    print(f"\n{name} cross-entity calls ({len(bad)}):")
    for row in bad[:limit]:
        top = ", ".join(
            f"{item['code']}={item['probability']:.3f}"
            for item in row["top_predictions"][:3]
        )
        print(
            f"  {row['truth']:18s} -> {row['call']:18s} "
            f"[{row['match_level']}] p={row['probability']:.3f} "
            f"margin={row['margin']:.3f} top={top}"
        )
    if len(bad) > limit:
        print(f"  ... {len(bad) - limit} more")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-type", type=int, default=3)
    parser.add_argument("--test-per-type", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--jsonl", default="")
    parser.add_argument("--miss-limit", type=int, default=80)
    args = parser.parse_args()

    types = sorted(available_representative_cohorts())
    split_data = _load_representative_samples(types)
    train_cols, test_cols, unused_cols = _split_columns(
        split_data.per_type,
        train_per_type=args.train_per_type,
        test_per_type=args.test_per_type,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    model, genes = _fit_model(
        split_data.clean_tpm,
        split_data.labels,
        train_cols,
        n_genes=args.n_genes,
    )
    test_predictions = _predict_matrix(
        model,
        genes,
        split_data.clean_tpm,
        test_cols,
        args.top_k,
    )
    test_records = _records_for_predictions(split_data.labels, test_predictions)

    test_by_type: dict[str, list[str]] = collections.defaultdict(list)
    for col in test_cols:
        test_by_type[split_data.labels[col]].append(col)
    medoid_all_records = _records_for_medoids(
        _predict_medoids(
            model,
            genes,
            split_data.clean_tpm,
            split_data.per_type,
            columns_by_type=None,
            top_k=args.top_k,
        ),
        split="medoid_all",
    )
    medoid_test_records = _records_for_medoids(
        _predict_medoids(
            model,
            genes,
            split_data.clean_tpm,
            split_data.per_type,
            columns_by_type=dict(test_by_type),
            top_k=args.top_k,
        ),
        split="medoid_test",
    )

    print(
        f"# {len(types)} cohorts, {len(split_data.labels)} samples; "
        f"train={len(train_cols)}, test={len(test_cols)}, unused={len(unused_cols)}; "
        f"genes={len(genes)}, seed={args.seed}, shuffle={not args.no_shuffle}"
    )
    _print_summary("held-out samples", test_records)
    _print_summary("all-sample medoids", medoid_all_records)
    _print_summary("held-out medoids", medoid_test_records)
    _print_misses("held-out sample", test_records, limit=args.miss_limit)
    _print_misses("held-out medoid", medoid_test_records, limit=args.miss_limit)

    if args.jsonl:
        with open(args.jsonl, "w") as handle:
            for split, records in (
                ("heldout", test_records),
                ("medoid_all", medoid_all_records),
                ("medoid_test", medoid_test_records),
            ):
                for row in records:
                    out = dict(row)
                    out["split"] = split
                    handle.write(json.dumps(out, sort_keys=True) + "\n")
        print(f"\nwrote {args.jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
