"""Empirical calibration for the TPM-renormalized decomposition pipeline.

PR #41 dropped the ``renormalize_to_million=False`` opt-out that
decomposition used to run against native-scale pirlygenes references. The
templates were recalibrated against the new TPM-1e6 footing
(``met_origin_penalty_*``, ``met_site_dominance_*``, ``fit_score_*``).

This script gives a quantitative regression artifact for that change.
It runs three calibration tracks:

1. **TCGA per-sample classification** — for each per-sample TCGA TPM
   profile in ``pirlygenes/eval/eval_tpm.parquet``, run the broad RNA
   classifier (``rank_cancer_type_candidates``) and check whether the
   top call matches the known TCGA cohort label. Reports per-cohort
   top-1 / top-3 accuracy.

2. **HPA normal-tissue baseline** — for each HPA normal tissue column,
   synthesize a per-tissue expression frame and run
   ``assess_tissue_composition``; verify it routes to the healthy or
   structural-ambiguity bucket rather than a confident cancer call.

3. **Local sample replay (optional)** — if ``--local-data-root`` is
   supplied, walks an existing ``trufflepig-local-reports`` directory
   tree and extracts the per-sample call from each ``analysis.json``;
   compares to an optional ``--expected-labels`` TSV of
   ``sample,expected_code`` rows.

Outputs:

* ``--out-json`` (default ``calibration-report.json``) — full report.
* ``--out-tsv`` (optional) — flat per-sample TSV.

Run with PYTHONPATH set to the local pirlygenes checkout so the script
sees the pirlygenes ``eval/`` directory and the API expected by
trufflepig (>=5.2)::

    PYTHONPATH=$HOME/code/pirlygenes:$PYTHONPATH \\
        python scripts/calibrate_decomposition.py \\
        --eval-tpm $HOME/code/pirlygenes/eval/eval_tpm.parquet \\
        --eval-samples $HOME/code/pirlygenes/eval/eval_samples.pkl

The script is intentionally read-only and emits no plots — the JSON
artifact is the regression guard. Future bisects can re-run this and
diff against a committed baseline.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Defer trufflepig imports until after argparse so --help is fast.


_DEFAULT_EVAL_TPM = Path.home() / "code" / "pirlygenes" / "eval" / "eval_tpm.parquet"
_DEFAULT_EVAL_SAMPLES = (
    Path.home() / "code" / "pirlygenes" / "eval" / "eval_samples.pkl"
)
_DEFAULT_FULL_TCGA_GZ = (
    Path.home() / "code" / "pirlygenes" / "eval" / "tcga_RSEM_gene_tpm.gz"
)
_DEFAULT_BARCODE_TO_PROJECT = (
    Path.home() / "code" / "pirlygenes" / "eval" / "barcode_to_project.pkl"
)
# Cached parquet derived from tcga_RSEM_gene_tpm.gz — the gzip parse
# takes minutes, parquet load takes seconds. Stored under the user's
# cache dir to keep the trufflepig tree clean.
_FULL_TCGA_PARQUET_CACHE = (
    Path.home() / ".cache" / "trufflepig" / "tcga_RSEM_gene_tpm.parquet"
)

# Pirlygenes-style local-reports manifest produced by
# pirlygenes/local_reports/regenerate_from_manifest.py. The most recent
# manifest at PR #41 time is hard-coded as the default so the calibration
# CLI works without the user needing to remember the path; override with
# ``--local-manifest`` if a newer one is available.
_DEFAULT_LOCAL_MANIFEST = (
    Path.home()
    / "code"
    / "pirlygenes"
    / "local_reports"
    / "rnaseq-20260506-132208"
    / "manifest.json"
)

# Expected calls per manifest-run name, from PR #41's "no-override replay"
# claims. Each entry is ``{"expected_cancer_type": str, "expected_selected_by":
# str|None, "notes": str}``. ``expected_selected_by=None`` means we accept
# either a primary_expression_match or a more specific selector. Multiple
# acceptable codes are joined by ``|`` (e.g. ``COAD|READ`` for the
# colon/rectum continuum).
_LOCAL_REPLAY_EXPECTED: dict[str, dict[str, Any]] = {
    "sid-pfo004-osteosarcoma-gene-expression": {
        "expected_cancer_type": "OS",
        "expected_selected_by": "fine_reference",
        "notes": "PFO004 osteosarcoma — fine OS reference inside SARC context",
    },
    "sid-pfo004-osteosarcoma-salmon-gene-tpm": {
        "expected_cancer_type": "OS",
        "expected_selected_by": "fine_reference",
        "notes": "PFO004 osteosarcoma salmon",
    },
    "pfo002-personalis-gene-expression": {
        "expected_cancer_type": "COAD|READ",
        "expected_selected_by": None,
        "notes": "PFO002 CRC (personalis)",
    },
    "pfo002-washu-kallisto-gene-abundance": {
        "expected_cancer_type": "COAD|READ",
        "expected_selected_by": None,
        "notes": "PFO002 CRC (washu kallisto)",
    },
    "pfo002-washu-stringtie-gene-expression": {
        "expected_cancer_type": "COAD|READ",
        "expected_selected_by": None,
        "notes": "PFO002 CRC (washu stringtie)",
    },
    "rs-tempus-salmon-rich-gene-tpm": {
        "expected_cancer_type": "PRAD",
        "expected_selected_by": None,
        "notes": "Kat-DL Tempus PRAD",
    },
    "asy-salmon-gene-tpm": {
        "expected_cancer_type": "COAD|READ",
        "expected_selected_by": None,
        "notes": "ASY CRC context",
    },
    "alvin-salmon-gene-tpm": {
        "expected_cancer_type": "SARC",
        "expected_selected_by": None,
        "notes": "Alvin SARC",
    },
    "hcc1395-kallisto-gene-abundance": {
        "expected_cancer_type": "BRCA",
        "expected_selected_by": None,
        "notes": "HCC1395 BRCA (kallisto)",
    },
    "hcc1395-stringtie-gene-expression": {
        "expected_cancer_type": "BRCA",
        "expected_selected_by": None,
        "notes": "HCC1395 BRCA (stringtie)",
    },
    "pfo019-kallisto-gene-abundance": {
        "expected_cancer_type": "NUTM",
        "expected_selected_by": "rare_marker",
        "notes": "PFO019 NUTM by RNA marker (no fusion data here)",
    },
    "pfo019-stringtie-gene-expression": {
        "expected_cancer_type": "NUTM",
        "expected_selected_by": "rare_marker",
        "notes": "PFO019 NUTM by RNA marker (stringtie)",
    },
    "pfo017-pfo017-bladder-2023": {
        "expected_cancer_type": "BLCA",
        "expected_selected_by": None,
        "notes": "PFO017 bladder 2023",
    },
    "pfo017-pfo017-bladder-2025": {
        "expected_cancer_type": "BLCA",
        "expected_selected_by": None,
        "notes": "PFO017 bladder 2025",
    },
    "pfo017-pfo017-liver-2023": {
        "expected_cancer_type": "BLCA",
        "expected_selected_by": None,
        "notes": "PFO017 liver met (BLCA primary)",
    },
    "tempus-nutm1-unc_0026-tl-21-ks8t3eyh": {
        "expected_cancer_type": "NUTM",
        "expected_selected_by": "rare_marker",
        "notes": "Tempus NUTM1 case 0026",
    },
    "tempus-nutm1-unc_0027-tl-21-tayehe9b": {
        "expected_cancer_type": "NUTM",
        "expected_selected_by": "rare_marker",
        "notes": "Tempus NUTM1 case 0027",
    },
    "tempus-nutm1-unc_0028-tl-20-10a3fc": {
        "expected_cancer_type": "NUTM",
        "expected_selected_by": "rare_marker",
        "notes": "Tempus NUTM1 case 0028",
    },
    "low-purity-prad-cegat-salmon-gene-tpm": {
        "expected_cancer_type": "PRAD",
        "expected_selected_by": None,
        "notes": "Kat-DL CeGaT PRAD (low purity)",
    },
}


def _materialise_full_tcga_parquet(
    gz_path: Path,
    parquet_path: Path,
) -> Path:
    """Convert the gzipped TCGA RSEM matrix to a float32 parquet (cached).

    The gzip is ~740 MB and re-parsing it on every calibration run takes
    several minutes. The parquet version loads in seconds and the file
    is ~1.5 GB. Stored under ``~/.cache/trufflepig/`` so the repo stays
    clean.
    """
    if parquet_path.exists():
        return parquet_path
    print(
        f"[full-tcga] materialising {parquet_path.name} from {gz_path.name} "
        "(one-time, ~5min)",
        file=sys.stderr,
        flush=True,
    )
    t0 = time.time()
    df = pd.read_csv(gz_path, sep="\t", compression="gzip", low_memory=False)
    # The RSEM matrix has 'sample' as the first column with Ensembl gene
    # IDs (possibly with version suffix). All other columns are sample
    # barcodes with log2(TPM+0.001) values — same encoding as eval_tpm.
    if "sample" not in df.columns:
        raise SystemExit(f"{gz_path} missing 'sample' column")
    df = df.rename(columns={"sample": "gene_id"})
    df["gene_id"] = df["gene_id"].astype(str).str.split(".").str[0]
    sample_cols = [c for c in df.columns if c != "gene_id"]
    df[sample_cols] = df[sample_cols].astype("float32")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, compression="zstd")
    print(
        f"[full-tcga] wrote parquet in {time.time()-t0:.0f}s "
        f"({df.shape[0]} genes × {len(sample_cols)} samples)",
        file=sys.stderr,
        flush=True,
    )
    return parquet_path


def _load_full_tcga_tpm(
    gz_path: Path,
    parquet_path: Path,
    barcode_to_project_path: Path,
    primary_only: bool = True,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Return ``(tpm_frame, sample_to_expected_code, sample_ids)``.

    ``tpm_frame`` is gene_id-indexed with float32 sample columns (raw
    log2(TPM+0.001), as in eval_tpm). ``sample_to_expected_code`` is the
    barcode-project lookup for the labelled subset. ``sample_ids`` is
    the ordered list of labelled, classifiable sample barcodes.

    ``primary_only=True`` keeps only TCGA sample-type ``01`` (primary
    tumor); ``False`` includes 03 (primary blood), 06 (metastatic) and
    11 (matched normal — recorded but classified separately).
    """
    parquet = _materialise_full_tcga_parquet(gz_path, parquet_path)
    print(
        f"[full-tcga] loading parquet {parquet.name}",
        file=sys.stderr,
        flush=True,
    )
    t0 = time.time()
    df = pd.read_parquet(parquet)
    print(
        f"[full-tcga] loaded in {time.time()-t0:.1f}s ({df.shape[1]-1} samples)",
        file=sys.stderr,
        flush=True,
    )
    with barcode_to_project_path.open("rb") as fh:
        barcode_to_project = pickle.load(fh)

    sample_to_expected: dict[str, str] = {}
    sample_ids: list[str] = []
    allowed_suffix = {"01"} if primary_only else {"01", "03", "06", "11"}
    for col in df.columns:
        if col == "gene_id":
            continue
        parts = str(col).split("-")
        if len(parts) < 4:
            continue
        suffix = parts[3]
        if suffix not in allowed_suffix:
            continue
        barcode = "-".join(parts[:3])
        project = barcode_to_project.get(barcode)
        if not project:
            continue
        # Sample-type ``11`` is matched normal — only useful if we want
        # to check the classifier doesn't false-call it as cancer. We
        # do not expect tumor cohort labels on normals; tag them
        # explicitly so the caller can split.
        expected_code = project if suffix in {"01", "03", "06"} else f"NORMAL:{project}"
        sample_to_expected[col] = expected_code
        sample_ids.append(col)
    return df, sample_to_expected, sample_ids


# ---------- subtype-aware code matching ----------
#
# The classifier may answer with a SUBTYPE code (e.g. "BRCA_LumB")
# where the expected label is the PARENT cohort ("BRCA"). Treating
# a subtype call as "wrong" against a parent label penalizes
# clinically-MORE-specific answers. Likewise, when an expected
# label is a subtype but the model returns the parent cohort,
# the model picked the correct family — less specific but not
# wrong (the classifier just didn't refine).
#
# These helpers use the pirlygenes ``cancer_type_registry`` to walk
# parent_code chains, so the kinship logic stays in sync with the
# registry rather than baking in pattern-matching on underscores
# (some codes are multi-token like RMS_ARMS, LAML_ELN_Fav).


@lru_cache(maxsize=1)
def _registry_parent_map() -> dict[str, str]:
    """Map ``cancer_code → parent_code`` (empty string when no parent)."""
    try:
        from pirlygenes.gene_sets_cancer import cancer_type_registry
    except ImportError:
        return {}
    df = cancer_type_registry().fillna("")
    return {
        str(row["code"]): str(row.get("parent_code") or "")
        for _, row in df.iterrows()
    }


def _ancestors(code: str) -> list[str]:
    """All ancestor codes up the parent chain (excludes ``code`` itself)."""
    out: list[str] = []
    parent_map = _registry_parent_map()
    current = parent_map.get(code, "") if code else ""
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        out.append(current)
        current = parent_map.get(current, "")
    return out


def _codes_match(observed: str, expected: str) -> tuple[bool, str]:
    """Return ``(matched, kind)`` for a single observed-vs-expected call.

    ``kind`` is one of:
      - ``"exact"`` — strings are identical (current behavior).
      - ``"subtype_of_expected"`` — observed is a strict subtype of
        expected (e.g. observed="BRCA_LumB", expected="BRCA"). MORE
        specific than the label; still correct.
      - ``"parent_of_expected"`` — observed is a strict ancestor of
        expected (e.g. observed="BRCA", expected="BRCA_LumB"). LESS
        specific but in the correct family; counted as correct.
      - ``"none"`` — different cancer types entirely; wrong.
    """
    if not observed or not expected:
        return False, "none"
    if observed == expected:
        return True, "exact"
    # Subtype of expected — observed's ancestors include expected.
    if expected in _ancestors(observed):
        return True, "subtype_of_expected"
    # Parent of expected — observed appears in expected's ancestor chain.
    if observed in _ancestors(expected):
        return True, "parent_of_expected"
    return False, "none"


def _any_in_kin(codes: list[str], expected: str) -> tuple[bool, str]:
    """Top-3 variant: return the best match across a list of codes.

    The kind prefers ``exact`` over ``subtype_of_expected`` over
    ``parent_of_expected`` over ``none`` so the headline ``kind`` for
    a top-3 hit reflects the most specific agreement.
    """
    best_kind = "none"
    rank = {"exact": 3, "subtype_of_expected": 2, "parent_of_expected": 1, "none": 0}
    matched = False
    for c in codes or []:
        m, k = _codes_match(c, expected)
        if m:
            matched = True
            if rank[k] > rank[best_kind]:
                best_kind = k
    return matched, best_kind


def _classify_sample_for_worker(args: tuple[str, str]) -> dict[str, Any]:
    """Worker entry: classify a single TCGA sample by barcode.

    Reads the per-worker globals set by ``_init_worker``: the loaded
    full-TCGA parquet plus the Ensembl→symbol map. Returns the
    per-sample record dict the main loop appends.
    """
    sample_id, expected_code = args
    global _WORKER_TPM, _WORKER_SYMBOL_BY_ID
    df_expr = _sample_to_expression_frame(
        sample_id, _WORKER_TPM[sample_id], _WORKER_SYMBOL_BY_ID
    )
    if df_expr.empty:
        return {
            "sample_id": sample_id,
            "expected_code": expected_code,
            "status": "empty",
        }
    t0 = time.time()
    _trace, summary = _classify_one(df_expr)
    elapsed = time.time() - t0
    broad_code = summary["broad_top_code"] or ""
    broad_top3 = summary.get("broad_top3") or []
    consolidated_code = summary["consolidated_cancer_type"] or broad_code
    selected_by = summary.get("consolidated_selected_by") or ""
    base_expected = (
        expected_code.split(":", 1)[1]
        if str(expected_code).startswith("NORMAL:")
        else expected_code
    )
    broad_top1, broad_top1_kind = _codes_match(broad_code, base_expected)
    broad_top3_hit, broad_top3_kind = _any_in_kin(broad_top3, base_expected)
    consolidated_top1, consolidated_top1_kind = _codes_match(
        consolidated_code, base_expected
    )
    return {
        "sample_id": sample_id,
        "expected_code": expected_code,
        "is_normal": str(expected_code).startswith("NORMAL:"),
        "broad_top_code": broad_code,
        "broad_top_support": round(float(summary["broad_top_support"]), 4),
        "broad_top3": broad_top3,
        "broad_top1_match": bool(broad_top1),
        "broad_top1_match_kind": broad_top1_kind,
        "broad_top3_match": bool(broad_top3_hit),
        "broad_top3_match_kind": broad_top3_kind,
        "consolidated_cancer_type": consolidated_code,
        "consolidated_selected_by": selected_by,
        "consolidated_top1_match": bool(consolidated_top1),
        "consolidated_top1_match_kind": consolidated_top1_kind,
        "top_code": broad_code,
        "top_support": round(float(summary["broad_top_support"]), 4),
        "top3": broad_top3,
        "top1_match": bool(broad_top1),
        "top1_match_kind": broad_top1_kind,
        "top3_match": bool(broad_top3_hit),
        "top3_match_kind": broad_top3_kind,
        "status": "ok",
        "seconds": round(elapsed, 2),
    }


# Per-worker globals — populated by ``_init_worker`` so the heavy
# pandas frames and pirlygenes reference are loaded once per worker
# rather than passed through pickle for every task.
_WORKER_TPM: pd.DataFrame | None = None
_WORKER_SYMBOL_BY_ID: dict[str, str] | None = None


def _init_worker(parquet_path_str: str) -> None:
    """Multiprocessing initializer: load reference + TCGA matrix once."""
    global _WORKER_TPM, _WORKER_SYMBOL_BY_ID
    from trufflepig.common import ensembl_id_to_symbol_map

    df = pd.read_parquet(parquet_path_str)
    # Project to gene_id-indexed TPM frame, back-transforming
    # log2(TPM+0.001) → TPM as in _load_eval_tpm.
    if "gene_id" not in df.columns:
        raise SystemExit("worker parquet missing 'gene_id'")
    sample_cols = [c for c in df.columns if c != "gene_id"]
    tpm = np.power(2.0, df[sample_cols].astype(float)) - 0.001
    tpm = tpm.clip(lower=0.0)
    out = pd.DataFrame(tpm.values, columns=sample_cols, index=df["gene_id"].astype(str))
    out.index.name = "ensembl_gene_id"
    _WORKER_TPM = out
    _WORKER_SYMBOL_BY_ID = {
        k.split(".")[0]: v for k, v in ensembl_id_to_symbol_map().items()
    }


def _tcga_full_per_sample_track(
    *,
    gz_path: Path,
    parquet_cache: Path,
    barcode_to_project_path: Path,
    workers: int,
    max_samples: int,
    cohorts: set[str] | None,
    include_normals: bool,
) -> dict[str, Any]:
    """All-TCGA per-sample classification under the full evidence model."""
    parquet = _materialise_full_tcga_parquet(gz_path, parquet_cache)
    # We don't need the loaded frame in the main process — workers
    # re-load it via _init_worker — but we DO need the sample list and
    # labels.
    _, sample_to_expected, sample_ids = _load_full_tcga_tpm(
        gz_path,
        parquet_cache,
        barcode_to_project_path,
        primary_only=not include_normals,
    )
    if cohorts:
        sample_ids = [
            sid
            for sid in sample_ids
            if sample_to_expected[sid].split(":", 1)[-1] in cohorts
        ]
    if max_samples and max_samples > 0:
        sample_ids = sample_ids[:max_samples]

    print(
        f"[full-tcga] classifying {len(sample_ids)} samples "
        f"with {workers} worker(s)",
        file=sys.stderr,
        flush=True,
    )
    tasks = [(sid, sample_to_expected[sid]) for sid in sample_ids]

    per_sample: list[dict[str, Any]] = []
    if workers <= 1:
        # In-process path — useful for debugging and for environments
        # where multiprocessing.set_start_method has been pinned.
        _init_worker(str(parquet))
        for i, task in enumerate(tasks, 1):
            row = _classify_sample_for_worker(task)
            per_sample.append(row)
            if i % 50 == 0:
                print(
                    f"[full-tcga] {i}/{len(tasks)} processed",
                    file=sys.stderr,
                    flush=True,
                )
    else:
        from multiprocessing import get_context

        ctx = get_context("spawn")
        with ctx.Pool(
            workers,
            initializer=_init_worker,
            initargs=(str(parquet),),
        ) as pool:
            t0 = time.time()
            for i, row in enumerate(
                pool.imap_unordered(_classify_sample_for_worker, tasks, chunksize=4),
                start=1,
            ):
                per_sample.append(row)
                if i % 200 == 0:
                    rate = i / max(time.time() - t0, 1e-9)
                    eta = (len(tasks) - i) / max(rate, 1e-9)
                    print(
                        f"[full-tcga] {i}/{len(tasks)} processed "
                        f"({rate:.1f}/s, ETA {eta/60:.1f}min)",
                        file=sys.stderr,
                        flush=True,
                    )

    return _summarise_full_tcga(per_sample)


def _summarise_full_tcga(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in per_sample if r.get("status") == "ok"]
    cancers = [r for r in ok if not r.get("is_normal")]
    normals = [r for r in ok if r.get("is_normal")]

    def _bucket(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        per_cohort: dict[str, dict[str, int]] = defaultdict(
            lambda: {"n": 0, "broad_top1": 0, "broad_top3": 0, "consolidated_top1": 0}
        )
        for r in rows:
            code = str(r["expected_code"]).split(":", 1)[-1]
            per_cohort[code]["n"] += 1
            per_cohort[code]["broad_top1"] += int(r.get("broad_top1_match"))
            per_cohort[code]["broad_top3"] += int(r.get("broad_top3_match"))
            per_cohort[code]["consolidated_top1"] += int(
                r.get("consolidated_top1_match")
            )
        return dict(per_cohort)

    per_cohort = _bucket(cancers)
    flip_counts = {
        "both_correct": 0,
        "both_wrong": 0,
        "broad_correct_consolidated_wrong": 0,
        "broad_wrong_consolidated_correct": 0,
    }
    per_selector: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "top1": 0}
    )
    for r in cancers:
        b = r.get("broad_top1_match")
        c = r.get("consolidated_top1_match")
        if b and c:
            flip_counts["both_correct"] += 1
        elif not b and not c:
            flip_counts["both_wrong"] += 1
        elif b and not c:
            flip_counts["broad_correct_consolidated_wrong"] += 1
        else:
            flip_counts["broad_wrong_consolidated_correct"] += 1
        selector = r.get("consolidated_selected_by") or "no_selection"
        per_selector[selector]["n"] += 1
        per_selector[selector]["top1"] += int(c)

    total_n = len(cancers)
    total_broad_top1 = sum(c["broad_top1"] for c in per_cohort.values())
    total_broad_top3 = sum(c["broad_top3"] for c in per_cohort.values())
    total_consolidated_top1 = sum(c["consolidated_top1"] for c in per_cohort.values())
    broad_top1_ci = _bootstrap_ci([r["broad_top1_match"] for r in cancers])
    broad_top3_ci = _bootstrap_ci([r["broad_top3_match"] for r in cancers])
    cons_top1_ci = _bootstrap_ci([r["consolidated_top1_match"] for r in cancers])

    return {
        "per_sample": per_sample,
        "per_cohort": {
            code: {
                "n": stats["n"],
                "broad_top1_accuracy": (
                    round(stats["broad_top1"] / stats["n"], 4) if stats["n"] else 0.0
                ),
                "broad_top3_accuracy": (
                    round(stats["broad_top3"] / stats["n"], 4) if stats["n"] else 0.0
                ),
                "consolidated_top1_accuracy": (
                    round(stats["consolidated_top1"] / stats["n"], 4)
                    if stats["n"]
                    else 0.0
                ),
            }
            for code, stats in sorted(per_cohort.items())
        },
        "summary": {
            "n_evaluated": total_n,
            "n_normals_held_out": len(normals),
            "layer_breakdown": {
                "broad_top1_accuracy": (
                    round(total_broad_top1 / total_n, 4) if total_n else 0.0
                ),
                "broad_top1_ci_95": broad_top1_ci,
                "broad_top3_accuracy": (
                    round(total_broad_top3 / total_n, 4) if total_n else 0.0
                ),
                "broad_top3_ci_95": broad_top3_ci,
                "consolidated_top1_accuracy": (
                    round(total_consolidated_top1 / total_n, 4) if total_n else 0.0
                ),
                "consolidated_top1_ci_95": cons_top1_ci,
                "flip_counts": flip_counts,
                "delta_consolidated_vs_broad_top1": (
                    round((total_consolidated_top1 - total_broad_top1) / total_n, 4)
                    if total_n
                    else 0.0
                ),
            },
            "per_selector": {
                name: {
                    "n": stats["n"],
                    "top1_accuracy": (
                        round(stats["top1"] / stats["n"], 4) if stats["n"] else 0.0
                    ),
                }
                for name, stats in sorted(per_selector.items())
            },
        },
    }


def _load_eval_tpm(path: Path) -> pd.DataFrame:
    """Read the TCGA per-sample eval matrix from parquet.

    The eval frame stores ``log2(TPM + 0.001)`` per sample column with
    ``gene_id`` as the row key. This back-transforms to raw TPM and
    returns a frame indexed by Ensembl gene id (versionless).
    """
    df = pd.read_parquet(path)
    if "gene_id" not in df.columns:
        raise SystemExit(f"eval_tpm parquet at {path} is missing 'gene_id'")
    sample_cols = [c for c in df.columns if c != "gene_id"]
    # Back-transform: log2(TPM + 0.001) -> TPM. Floor negative values that
    # the round-trip occasionally produces from float32 storage.
    tpm = np.power(2.0, df[sample_cols].astype(float)) - 0.001
    tpm = tpm.clip(lower=0.0)
    out = pd.DataFrame(tpm.values, columns=sample_cols, index=df["gene_id"].astype(str))
    out.index.name = "ensembl_gene_id"
    return out


def _sample_to_expression_frame(
    sample_id: str,
    tpm_by_gene_id: pd.Series,
    symbol_by_id: dict[str, str],
) -> pd.DataFrame:
    """Build a trufflepig-shaped df_expr for one TCGA sample."""
    from trufflepig.clean_tpm import technical_rna_mask

    rows = []
    for gene_id, tpm in tpm_by_gene_id.items():
        symbol = symbol_by_id.get(gene_id, "")
        if not symbol:
            continue
        rows.append(
            {
                "ensembl_gene_id": gene_id,
                "canonical_gene_name": symbol,
                "TPM": float(tpm),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Drop technical-RNA rows so the clean-TPM gate downstream doesn't
    # reject. The classifier expects clean inputs, so the calibration
    # mirrors what apply_expression_qc_rescue would produce.
    mask = technical_rna_mask(
        frame,
        label_col="canonical_gene_name",
        id_col="ensembl_gene_id",
    )
    frame = frame.loc[~mask].reset_index(drop=True)
    total = float(frame["TPM"].sum())
    if total > 0:
        frame["TPM"] = frame["TPM"] * (1_000_000.0 / total)
    return frame


def _classify_one(df_expr: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return ``(candidate_trace, summary)`` for one sample.

    The summary records BOTH the broad-RNA-only call (step 1:
    ``rank_cancer_type_candidates``) and the cancer-type-evidence-
    consolidated call (step 2: ``select_report_scope_from_evidence``).
    Future regressions can be isolated to whichever step changed: if the
    broad call still matches but the consolidated call doesn't, the
    cancer-type-evidence layer regressed; vice versa for the broad
    classifier.
    """
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence
    from trufflepig.tumor_purity import rank_cancer_type_candidates

    trace = rank_cancer_type_candidates(
        df_expr,
        top_k=5,
        use_subtype_signatures=False,
    )
    if not trace:
        return [], {
            "broad_top_code": None,
            "broad_top_support": 0.0,
            "broad_top3": [],
            "n": 0,
            "consolidated_cancer_type": None,
            "consolidated_selected_by": None,
        }
    top = trace[0]
    broad_top_code = str(top.get("code") or "")
    broad_top3 = [str(row.get("code") or "") for row in trace[:3]]

    # Layer 2: cancer_type_evidence consolidation. Same inputs the live
    # pipeline feeds it from main.py (broad RNA candidates + RNA marker
    # hypotheses). No fusion-scope input — that requires alterations
    # which the TCGA per-sample sweep does not have.
    analysis = {
        "candidate_trace": trace,
        "cancer_type": broad_top_code,
    }
    try:
        from trufflepig.rare_inference import (
            infer_rare_cancer_marker_hypotheses_from_rna,
        )

        rare = infer_rare_cancer_marker_hypotheses_from_rna(df_expr, analysis) or []
    except (ImportError, KeyError, ValueError, TypeError):
        rare = []
    analysis["rare_marker_hypotheses"] = rare
    try:
        evidence = select_report_scope_from_evidence(
            df_expr,
            analysis,
            rare_marker_hypotheses=rare,
            fusion_scope_inference=None,
        )
    except (KeyError, ValueError, TypeError):
        evidence = {}
    selected = (evidence or {}).get("selected") or {}
    consolidated_cancer_type = (
        str(selected.get("cancer_type") or "") or broad_top_code
    )
    return trace, {
        "broad_top_code": broad_top_code,
        "broad_top_support": float(top.get("support_fraction_of_top") or 0.0),
        "broad_top3": broad_top3,
        "n": len(trace),
        "consolidated_cancer_type": consolidated_cancer_type,
        "consolidated_selected_by": str(selected.get("selected_by") or ""),
    }


def _bootstrap_ci(
    bools: list[bool],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float] | None:
    """Return ``(lo, hi)`` confidence interval for a Bernoulli proportion.

    Returns ``None`` for an empty input. Uses a fixed-seed numpy resample
    so the reported interval is reproducible between runs (a real
    regression-bar should not flap from RNG noise alone).
    """
    if not bools:
        return None
    rng = np.random.default_rng(seed)
    vals = np.asarray(bools, dtype=float)
    idx = rng.integers(0, len(vals), size=(n_resamples, len(vals)))
    means = vals[idx].mean(axis=1)
    lo = float(np.quantile(means, (1.0 - confidence) / 2.0))
    hi = float(np.quantile(means, 1.0 - (1.0 - confidence) / 2.0))
    return (round(lo, 4), round(hi, 4))


def _tcga_per_sample_track(
    eval_tpm_path: Path,
    eval_samples_path: Path,
    sample_limit_per_cohort: int | None,
    cohorts: set[str] | None,
) -> dict[str, Any]:
    print(f"[tcga] loading {eval_tpm_path}", file=sys.stderr, flush=True)
    tpm = _load_eval_tpm(eval_tpm_path)
    with eval_samples_path.open("rb") as fh:
        samples_by_cancer = pickle.load(fh)

    from trufflepig.common import ensembl_id_to_symbol_map

    symbol_by_id = ensembl_id_to_symbol_map()
    # Strip Ensembl version suffix from the lookup index just in case.
    symbol_by_id = {k.split(".")[0]: v for k, v in symbol_by_id.items()}

    per_sample: list[dict[str, Any]] = []
    per_cohort: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "broad_top1": 0, "broad_top3": 0, "consolidated_top1": 0}
    )
    # Per-selector breakdown: which evidence path picked the
    # consolidated call (primary_expression_match / rare_marker /
    # fine_reference / etc.). When a regression hits, this tells us
    # whether the broad classifier or a specific selector misfired.
    per_selector_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "top1": 0}
    )
    flip_counts = {
        "broad_correct_consolidated_wrong": 0,
        "broad_wrong_consolidated_correct": 0,
        "both_correct": 0,
        "both_wrong": 0,
    }
    for expected_code in sorted(samples_by_cancer):
        if cohorts and expected_code not in cohorts:
            continue
        sample_ids = list(samples_by_cancer[expected_code])
        if sample_limit_per_cohort is not None:
            sample_ids = sample_ids[:sample_limit_per_cohort]
        for sample_id in sample_ids:
            if sample_id not in tpm.columns:
                print(
                    f"[tcga] {expected_code}/{sample_id}: missing column",
                    file=sys.stderr,
                )
                continue
            t0 = time.time()
            df_expr = _sample_to_expression_frame(
                sample_id, tpm[sample_id], symbol_by_id
            )
            if df_expr.empty:
                continue
            _trace, summary = _classify_one(df_expr)
            elapsed = time.time() - t0
            broad_code = summary["broad_top_code"] or ""
            broad_top3 = summary.get("broad_top3") or []
            consolidated_code = summary["consolidated_cancer_type"] or broad_code
            selected_by = summary.get("consolidated_selected_by") or ""
            broad_top1 = broad_code == expected_code
            broad_top3_hit = expected_code in broad_top3
            consolidated_top1 = consolidated_code == expected_code
            per_cohort[expected_code]["n"] += 1
            per_cohort[expected_code]["broad_top1"] += int(broad_top1)
            per_cohort[expected_code]["broad_top3"] += int(broad_top3_hit)
            per_cohort[expected_code]["consolidated_top1"] += int(consolidated_top1)
            per_selector_counts[selected_by or "no_selection"]["n"] += 1
            per_selector_counts[selected_by or "no_selection"]["top1"] += int(
                consolidated_top1
            )
            if broad_top1 and consolidated_top1:
                flip_counts["both_correct"] += 1
            elif not broad_top1 and not consolidated_top1:
                flip_counts["both_wrong"] += 1
            elif broad_top1 and not consolidated_top1:
                flip_counts["broad_correct_consolidated_wrong"] += 1
            else:
                flip_counts["broad_wrong_consolidated_correct"] += 1
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "expected_code": expected_code,
                    "broad_top_code": broad_code,
                    "broad_top_support": round(float(summary["broad_top_support"]), 4),
                    "broad_top3": broad_top3,
                    "broad_top1_match": bool(broad_top1),
                    "broad_top3_match": bool(broad_top3_hit),
                    "consolidated_cancer_type": consolidated_code,
                    "consolidated_selected_by": selected_by,
                    "consolidated_top1_match": bool(consolidated_top1),
                    # Back-compat aliases for older tooling reading the
                    # report (TSV consumers, baseline JSONs from before
                    # the per-step split).
                    "top_code": broad_code,
                    "top_support": round(float(summary["broad_top_support"]), 4),
                    "top3": broad_top3,
                    "top1_match": bool(broad_top1),
                    "top3_match": bool(broad_top3_hit),
                    "seconds": round(elapsed, 2),
                }
            )
            print(
                f"[tcga] {expected_code}/{sample_id}: "
                f"broad={broad_code} top3={broad_top3} "
                f"consolidated={consolidated_code} ({selected_by or '-'}) "
                f"broad_top1={broad_top1} cons_top1={consolidated_top1} "
                f"({elapsed:.1f}s)",
                file=sys.stderr,
                flush=True,
            )

    total_n = sum(c["n"] for c in per_cohort.values())
    total_broad_top1 = sum(c["broad_top1"] for c in per_cohort.values())
    total_broad_top3 = sum(c["broad_top3"] for c in per_cohort.values())
    total_consolidated_top1 = sum(c["consolidated_top1"] for c in per_cohort.values())
    broad_top1_ci = _bootstrap_ci([row["broad_top1_match"] for row in per_sample])
    broad_top3_ci = _bootstrap_ci([row["broad_top3_match"] for row in per_sample])
    cons_top1_ci = _bootstrap_ci(
        [row["consolidated_top1_match"] for row in per_sample]
    )
    return {
        "per_sample": per_sample,
        "per_cohort": {
            code: {
                "n": stats["n"],
                "broad_top1_accuracy": (
                    round(stats["broad_top1"] / stats["n"], 4) if stats["n"] else 0.0
                ),
                "broad_top3_accuracy": (
                    round(stats["broad_top3"] / stats["n"], 4) if stats["n"] else 0.0
                ),
                "consolidated_top1_accuracy": (
                    round(stats["consolidated_top1"] / stats["n"], 4)
                    if stats["n"]
                    else 0.0
                ),
                # Back-compat aliases.
                "top1_accuracy": (
                    round(stats["broad_top1"] / stats["n"], 4) if stats["n"] else 0.0
                ),
                "top3_accuracy": (
                    round(stats["broad_top3"] / stats["n"], 4) if stats["n"] else 0.0
                ),
            }
            for code, stats in sorted(per_cohort.items())
        },
        "summary": {
            "n_samples": total_n,
            "layer_breakdown": {
                "broad_top1_accuracy": (
                    round(total_broad_top1 / total_n, 4) if total_n else 0.0
                ),
                "broad_top1_ci_95": broad_top1_ci,
                "broad_top3_accuracy": (
                    round(total_broad_top3 / total_n, 4) if total_n else 0.0
                ),
                "broad_top3_ci_95": broad_top3_ci,
                "consolidated_top1_accuracy": (
                    round(total_consolidated_top1 / total_n, 4) if total_n else 0.0
                ),
                "consolidated_top1_ci_95": cons_top1_ci,
                "flip_counts": dict(flip_counts),
                "delta_consolidated_vs_broad_top1": (
                    round((total_consolidated_top1 - total_broad_top1) / total_n, 4)
                    if total_n
                    else 0.0
                ),
            },
            "per_selector": {
                name: {
                    "n": stats["n"],
                    "top1_accuracy": (
                        round(stats["top1"] / stats["n"], 4) if stats["n"] else 0.0
                    ),
                }
                for name, stats in sorted(per_selector_counts.items())
            },
            # Back-compat: older tooling reads these top-level keys.
            "top1_accuracy": (
                round(total_broad_top1 / total_n, 4) if total_n else 0.0
            ),
            "top1_ci_95": broad_top1_ci,
            "top3_accuracy": (
                round(total_broad_top3 / total_n, 4) if total_n else 0.0
            ),
            "top3_ci_95": broad_top3_ci,
        },
    }


def _hpa_normal_track(tissues: list[str] | None) -> dict[str, Any]:
    print("[hpa] loading pan_cancer_expression", file=sys.stderr, flush=True)
    from trufflepig.healthy_vs_tumor import assess_tissue_composition
    from trufflepig.reference import pan_cancer_expression

    pan = (
        pan_cancer_expression(technical_rna_normalize=True)
        .drop_duplicates(subset="Symbol")
        .reset_index(drop=True)
    )
    hpa_cols = [c for c in pan.columns if c.endswith("_nTPM")]
    if tissues:
        wanted = {f"{t}_nTPM" for t in tissues}
        hpa_cols = [c for c in hpa_cols if c in wanted]

    per_tissue: list[dict[str, Any]] = []
    bucket_counts: dict[str, int] = defaultdict(int)
    for col in sorted(hpa_cols):
        tissue = col.removesuffix("_nTPM")
        sample = pan[["Ensembl_Gene_ID", "Symbol", col]].rename(
            columns={
                "Ensembl_Gene_ID": "ensembl_gene_id",
                "Symbol": "canonical_gene_name",
                col: "TPM",
            }
        )
        total = float(sample["TPM"].sum(skipna=True))
        if total <= 0:
            continue
        sample["TPM"] = sample["TPM"].astype(float) * (1_000_000.0 / total)
        signal = assess_tissue_composition(sample)
        hint = str(getattr(signal, "cancer_hint", "") or "")
        bucket = (
            "healthy"
            if "healthy" in hint
            else "ambiguous"
            if "ambig" in hint or hint.endswith("indistinguishable")
            else "tumor-like"
            if hint
            else "no-call"
        )
        bucket_counts[bucket] += 1
        top_normal = (
            f"{signal.top_normal_tissues[0][0]}={signal.top_normal_tissues[0][1]:.3f}"
            if getattr(signal, "top_normal_tissues", None)
            else ""
        )
        top_tcga = (
            f"{signal.top_tcga_cohorts[0][0]}={signal.top_tcga_cohorts[0][1]:.3f}"
            if getattr(signal, "top_tcga_cohorts", None)
            else ""
        )
        per_tissue.append(
            {
                "tissue": tissue,
                "hint": hint,
                "bucket": bucket,
                "top_normal": top_normal,
                "top_tcga": top_tcga,
            }
        )
        print(
            f"[hpa] {tissue}: hint={hint or '-'} bucket={bucket} "
            f"normal={top_normal} tcga={top_tcga}",
            file=sys.stderr,
            flush=True,
        )

    return {
        "per_tissue": per_tissue,
        "summary": {
            "n_tissues": len(per_tissue),
            "by_bucket": dict(sorted(bucket_counts.items())),
        },
    }


def _read_local_input(path: Path) -> pd.DataFrame:
    """Read a local sample input without going through load_expression.

    Bypasses ``trufflepig.load_expression.load_expression_data`` so that
    this calibration script keeps working even when in-progress edits to
    the loader are temporarily broken. Only handles the input shapes
    actually present in the pirlygenes local manifest (csv/tsv/sf with
    gene-level rows).
    """
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".csv"):
        return pd.read_csv(path)
    if suffix.endswith((".tsv", ".sf", ".txt")):
        return pd.read_csv(path, sep="\t")
    raise SystemExit(f"unsupported local input extension: {path}")


def _coerce_local_sample(
    raw: pd.DataFrame,
    sample_id_value: str | None,
) -> pd.DataFrame:
    """Project a local input frame to ``[ensembl_gene_id, canonical_gene_name, TPM]``.

    Recognises the manifest's input shapes:

    * ``gene_id, gene, TPM, ensembl_release`` (salmon quant.gene_tpm.csv,
      pfo004 gene-expression.csv).
    * ``gene_id, gene_name, <sample_col>, ...`` (salmon.merged.gene_tpm.tsv —
      multi-sample, requires ``sample_id_value``).
    * ``gene, gene_name, abundance, counts, length`` (kallisto
      gene_abundance.tsv).
    * ``Gene ID, Gene Name, ..., FPKM, TPM`` (stringtie gene-expression).
    * ``ensembl_gene, gene_raw, ..., gene_tpm_cognizant_corrector``
      (tempus normalized_rna.csv — needs ``sample_id_value`` to select
      a single ``partner_sample_id``).
    * Personalis ``Gene ID``/``Gene Symbol``/``TPM`` long-form report.
    """
    df = raw.copy()
    cols = {c.lower(): c for c in df.columns}

    # Tempus normalized_rna.csv: long-form per (patient, sample, gene).
    if "ensembl_gene" in cols and "partner_sample_id" in cols:
        sample_col = cols["partner_sample_id"]
        if sample_id_value:
            df = df[df[sample_col].astype(str) == sample_id_value]
        else:
            first_sample = sorted(df[sample_col].dropna().unique())[0]
            df = df[df[sample_col].astype(str) == str(first_sample)]
        if "gene_tpm_cognizant_corrector" in cols:
            tpm_col = cols["gene_tpm_cognizant_corrector"]
        elif "gene_raw" in cols:
            tpm_col = cols["gene_raw"]
        else:
            raise SystemExit("normalized_rna.csv missing a TPM column")
        return pd.DataFrame(
            {
                "ensembl_gene_id": df[cols["ensembl_gene"]].astype(str),
                "canonical_gene_name": df.get(
                    cols.get("gene_norm_cognizant_corrector", ""),
                    pd.Series([""] * len(df), index=df.index),
                ).astype(str),
                "TPM": pd.to_numeric(df[tpm_col], errors="coerce").fillna(0.0),
            }
        )

    # Multi-sample salmon.merged.gene_tpm.tsv: pick the requested column.
    sample_specific = None
    if sample_id_value and sample_id_value in df.columns:
        sample_specific = sample_id_value
    if sample_specific is None:
        # Wide stringtie-style frames may have a single TPM column —
        # detect any non-id numeric column.
        numeric_value_candidates = [
            c
            for c in df.columns
            if c.lower() not in {"gene_id", "gene_name", "gene", "ensembl_release", "length",
                                 "counts", "abundance", "fpkm", "tpm", "coverage", "reference",
                                 "strand", "start", "end", "gene id", "gene name", "gene symbol"}
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if numeric_value_candidates and "gene_id" in cols and "gene_name" in cols:
            sample_specific = numeric_value_candidates[0]
    if sample_specific is not None:
        return pd.DataFrame(
            {
                "ensembl_gene_id": df[cols.get("gene_id", "gene_id")].astype(str),
                "canonical_gene_name": df.get(
                    cols.get("gene_name", "gene_name"),
                    pd.Series([""] * len(df), index=df.index),
                ).astype(str),
                "TPM": pd.to_numeric(df[sample_specific], errors="coerce").fillna(0.0),
            }
        )

    # Direct TPM column.
    if "tpm" in cols:
        # Salmon quant.gene_tpm.csv: gene_id, gene, TPM, ensembl_release.
        gene_id_col = (
            cols.get("gene_id")
            or cols.get("ensembl_gene_id")
            or cols.get("gene id")
            or cols.get("ensembl_id")
        )
        gene_name_col = (
            cols.get("gene")
            or cols.get("gene_name")
            or cols.get("gene name")
            or cols.get("gene symbol")
            or cols.get("symbol")
        )
        # Personalis tumor RNA reports use NCBI/Entrez IDs only — recover
        # Ensembl IDs by symbol via the trufflepig pan-cancer reference.
        if gene_id_col is None and gene_name_col is not None:
            from trufflepig.common import ensembl_id_to_symbol_map

            sym_to_id = {
                sym.upper(): ensg
                for ensg, sym in ensembl_id_to_symbol_map().items()
                if sym
            }
            symbols = df[gene_name_col].astype(str).str.upper()
            ensg = symbols.map(sym_to_id).fillna("")
            return pd.DataFrame(
                {
                    "ensembl_gene_id": ensg,
                    "canonical_gene_name": df[gene_name_col].astype(str),
                    "TPM": pd.to_numeric(df[cols["tpm"]], errors="coerce").fillna(0.0),
                }
            )
        if gene_id_col is None or gene_name_col is None:
            raise SystemExit(
                f"could not locate gene-id and gene-name columns in {list(df.columns)}"
            )
        return pd.DataFrame(
            {
                "ensembl_gene_id": df[gene_id_col].astype(str),
                "canonical_gene_name": df[gene_name_col].astype(str),
                "TPM": pd.to_numeric(df[cols["tpm"]], errors="coerce").fillna(0.0),
            }
        )

    # Kallisto gene_abundance.tsv: gene_name, gene, abundance, counts, length.
    if "abundance" in cols and "length" in cols:
        # Kallisto abundance is TPM-like (already library-size-normalised).
        return pd.DataFrame(
            {
                "ensembl_gene_id": df[cols.get("gene", "gene")].astype(str),
                "canonical_gene_name": df[cols["gene_name"]].astype(str),
                "TPM": pd.to_numeric(df[cols["abundance"]], errors="coerce").fillna(0.0),
            }
        )

    # Stringtie: Gene ID, Gene Name, ..., FPKM. Convert FPKM → TPM per sample.
    if "fpkm" in cols and "gene id" in cols:
        fpkm = pd.to_numeric(df[cols["fpkm"]], errors="coerce").fillna(0.0)
        total = float(fpkm.sum())
        tpm = fpkm * (1_000_000.0 / total) if total > 0 else fpkm
        return pd.DataFrame(
            {
                "ensembl_gene_id": df[cols["gene id"]].astype(str),
                "canonical_gene_name": df[cols.get("gene name", "Gene Name")].astype(str),
                "TPM": tpm,
            }
        )

    raise SystemExit(f"could not coerce local input shape: {list(df.columns)}")


def _prepare_local_df_expr(
    path: Path,
    sample_id_value: str | None,
) -> pd.DataFrame:
    from trufflepig.clean_tpm import technical_rna_mask

    raw = _read_local_input(path)
    df = _coerce_local_sample(raw, sample_id_value=sample_id_value)
    df["ensembl_gene_id"] = df["ensembl_gene_id"].str.split(".").str[0]
    df = df[df["ensembl_gene_id"].str.startswith("ENSG")].reset_index(drop=True)
    if df.empty:
        return df
    mask = technical_rna_mask(
        df,
        label_col="canonical_gene_name",
        id_col="ensembl_gene_id",
    )
    df = df.loc[~mask].reset_index(drop=True)
    total = float(df["TPM"].sum())
    if total > 0:
        df["TPM"] = df["TPM"] * (1_000_000.0 / total)
    return df


def _classify_local_with_evidence(
    df_expr: pd.DataFrame,
) -> dict[str, Any]:
    """Run broad RNA + cancer_type_evidence selection on a local sample."""
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence
    from trufflepig.tumor_purity import rank_cancer_type_candidates

    trace = rank_cancer_type_candidates(
        df_expr,
        top_k=5,
        use_subtype_signatures=False,
    )
    analysis = {
        "candidate_trace": trace,
        "cancer_type": trace[0]["code"] if trace else "",
    }
    try:
        from trufflepig.rare_inference import (
            infer_rare_cancer_marker_hypotheses_from_rna,
        )

        rare = infer_rare_cancer_marker_hypotheses_from_rna(df_expr, analysis) or []
    except (ImportError, KeyError, ValueError, TypeError):
        rare = []
    analysis["rare_marker_hypotheses"] = rare
    evidence = select_report_scope_from_evidence(
        df_expr,
        analysis,
        rare_marker_hypotheses=rare,
        fusion_scope_inference=None,
    )
    selected = (evidence or {}).get("selected") or {}
    return {
        "top_code": trace[0]["code"] if trace else "",
        "top3": [row["code"] for row in trace[:3]],
        "selected_cancer_type": selected.get("cancer_type", ""),
        "selected_by": selected.get("selected_by", ""),
        "evidence_sources": selected.get("evidence_sources", []),
    }


def _accept_expected(observed: str, expected: str) -> bool:
    """``expected`` may be ``"COAD|READ"`` to accept either."""
    options = {part.strip().upper() for part in expected.split("|") if part.strip()}
    return observed.upper() in options


def _local_no_override_track(
    manifest_path: Path,
    expected_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    print(f"[local] reading manifest {manifest_path}", file=sys.stderr, flush=True)
    manifest = json.loads(manifest_path.read_text())
    runs = manifest.get("runs") or []

    expected_table = dict(_LOCAL_REPLAY_EXPECTED)
    if expected_overrides:
        expected_table.update(expected_overrides)

    per_sample: list[dict[str, Any]] = []
    for run in runs:
        name = str(run.get("name") or "")
        if name not in expected_table:
            continue
        input_path = Path(str(run.get("input") or ""))
        if not input_path.exists():
            per_sample.append(
                {
                    "name": name,
                    "input_path": str(input_path),
                    "status": "input_missing",
                }
            )
            print(f"[local] {name}: input missing — skipping", file=sys.stderr)
            continue
        sample_id_value = _extract_sample_id_value(run)
        expected = expected_table[name]
        t0 = time.time()
        try:
            df_expr = _prepare_local_df_expr(input_path, sample_id_value)
        except SystemExit as exc:
            per_sample.append(
                {
                    "name": name,
                    "input_path": str(input_path),
                    "status": "load_error",
                    "error": str(exc),
                }
            )
            print(f"[local] {name}: load error: {exc}", file=sys.stderr)
            continue
        if df_expr.empty:
            per_sample.append(
                {
                    "name": name,
                    "input_path": str(input_path),
                    "status": "empty_after_qc",
                }
            )
            print(f"[local] {name}: empty after technical-RNA mask", file=sys.stderr)
            continue
        outcome = _classify_local_with_evidence(df_expr)
        elapsed = time.time() - t0
        expected_code = str(expected.get("expected_cancer_type") or "")
        expected_selected_by = expected.get("expected_selected_by") or ""
        inferred = (
            outcome["selected_cancer_type"]
            or outcome["top_code"]
        )
        code_match = bool(expected_code) and _accept_expected(inferred, expected_code)
        selector_match = (
            not expected_selected_by
            or outcome["selected_by"] == expected_selected_by
        )
        per_sample.append(
            {
                "name": name,
                "input_path": str(input_path),
                "sample_id_value": sample_id_value,
                "expected_cancer_type": expected_code,
                "expected_selected_by": expected_selected_by or None,
                "inferred_cancer_type": inferred,
                "top_code": outcome["top_code"],
                "top3": outcome["top3"],
                "selected_by": outcome["selected_by"],
                "evidence_sources": outcome["evidence_sources"],
                "code_match": code_match,
                "selector_match": selector_match,
                "status": "ok",
                "notes": expected.get("notes") or "",
                "seconds": round(elapsed, 2),
            }
        )
        print(
            f"[local] {name}: inferred={inferred} (top={outcome['top_code']}, "
            f"selected_by={outcome['selected_by'] or '-'}) "
            f"expected={expected_code}/{expected_selected_by or '-'} "
            f"code_match={code_match} selector_match={selector_match} "
            f"({elapsed:.1f}s)",
            file=sys.stderr,
            flush=True,
        )

    ok_rows = [row for row in per_sample if row.get("status") == "ok"]
    n_total = len(ok_rows)
    n_code_match = sum(1 for row in ok_rows if row.get("code_match"))
    n_selector_match = sum(
        1 for row in ok_rows if row.get("code_match") and row.get("selector_match")
    )

    # Layer breakdown: how many would have been right if we'd only looked
    # at the broad RNA classifier (the candidate trace's top code,
    # accepting "|"-multi-options)? This lets us tell whether the
    # cancer_type_evidence layer is doing real work (lifting NUTM /
    # OS calls out of LUSC / SARC) vs noise.
    n_broad_match = 0
    for row in ok_rows:
        top_code = (row.get("top_code") or "").upper()
        expected = row.get("expected_cancer_type") or ""
        if _accept_expected(top_code, expected):
            n_broad_match += 1
    per_selector_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "code_match": 0}
    )
    for row in ok_rows:
        selector = row.get("selected_by") or "no_selection"
        per_selector_counts[selector]["n"] += 1
        per_selector_counts[selector]["code_match"] += int(bool(row.get("code_match")))
    return {
        "manifest_path": str(manifest_path),
        "per_sample": per_sample,
        "summary": {
            "n_evaluated": n_total,
            "n_code_match": n_code_match,
            "n_full_match": n_selector_match,
            "code_match_rate": round(n_code_match / n_total, 4) if n_total else None,
            "full_match_rate": (
                round(n_selector_match / n_total, 4) if n_total else None
            ),
            "layer_breakdown": {
                "broad_only_code_match_rate": (
                    round(n_broad_match / n_total, 4) if n_total else None
                ),
                "consolidated_code_match_rate": (
                    round(n_code_match / n_total, 4) if n_total else None
                ),
                "delta_consolidated_vs_broad": (
                    round((n_code_match - n_broad_match) / n_total, 4)
                    if n_total
                    else None
                ),
                "n_lifted_by_consolidation": max(0, n_code_match - n_broad_match),
            },
            "per_selector": {
                name: {
                    "n": stats["n"],
                    "code_match_rate": (
                        round(stats["code_match"] / stats["n"], 4)
                        if stats["n"]
                        else 0.0
                    ),
                }
                for name, stats in sorted(per_selector_counts.items())
            },
        },
    }


def _extract_sample_id_value(run: dict[str, Any]) -> str | None:
    """Recover ``--sample-id-value`` if the manifest command had one."""
    cmd = run.get("command") or []
    for i, tok in enumerate(cmd):
        if tok == "--sample-id-value" and i + 1 < len(cmd):
            return str(cmd[i + 1])
    return None


def _local_replay_track(
    local_root: Path,
    expected_labels: Path | None,
) -> dict[str, Any]:
    expected: dict[str, str] = {}
    if expected_labels and expected_labels.exists():
        for line in expected_labels.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            expected[parts[0]] = parts[1].strip().upper()

    per_sample: list[dict[str, Any]] = []
    n_match = 0
    n_known = 0
    for analysis_json in local_root.rglob("analysis.json"):
        try:
            analysis = json.loads(analysis_json.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[local] {analysis_json}: read error: {exc}", file=sys.stderr)
            continue
        sample_id = str(
            analysis.get("sample_id")
            or analysis.get("sample")
            or analysis_json.parent.name
        )
        top = str(
            analysis.get("cancer_type")
            or analysis.get("inferred_cancer_type")
            or ""
        ).upper()
        expected_code = expected.get(sample_id, "")
        match = bool(expected_code) and top == expected_code
        if expected_code:
            n_known += 1
            n_match += int(match)
        per_sample.append(
            {
                "path": str(analysis_json),
                "sample_id": sample_id,
                "top_code": top,
                "expected_code": expected_code or None,
                "match": match,
            }
        )

    return {
        "per_sample": per_sample,
        "summary": {
            "n_samples": len(per_sample),
            "n_with_expected": n_known,
            "n_match": n_match,
            "match_rate": round(n_match / n_known, 4) if n_known else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-tpm",
        type=Path,
        default=_DEFAULT_EVAL_TPM,
        help="Per-sample TCGA TPM parquet (log2(TPM+0.001) values).",
    )
    p.add_argument(
        "--eval-samples",
        type=Path,
        default=_DEFAULT_EVAL_SAMPLES,
        help="Pickled {cancer_code: [sample_id,...]} dict.",
    )
    p.add_argument(
        "--samples-per-cohort",
        type=int,
        default=5,
        help="TCGA samples to classify per cohort (0 = all).",
    )
    p.add_argument(
        "--cohorts",
        type=str,
        default="",
        help="Comma-separated TCGA codes to include (default: all).",
    )
    p.add_argument(
        "--skip-tcga",
        action="store_true",
        help="Skip the TCGA per-sample classification track.",
    )
    p.add_argument(
        "--use-full-tcga-matrix",
        action="store_true",
        help=(
            "Replace the eval_tpm-based 5/cohort TCGA track with the full "
            "tcga_RSEM_gene_tpm.gz matrix (~10k samples). Uses "
            "barcode_to_project.pkl for ground-truth labels. Requires "
            "--workers > 1 for tractable wall time."
        ),
    )
    p.add_argument(
        "--full-tcga-gz",
        type=Path,
        default=_DEFAULT_FULL_TCGA_GZ,
        help="Path to tcga_RSEM_gene_tpm.gz when --use-full-tcga-matrix is set.",
    )
    p.add_argument(
        "--full-tcga-parquet-cache",
        type=Path,
        default=_FULL_TCGA_PARQUET_CACHE,
        help=(
            "Where to write/read the cached parquet derived from the gz. "
            f"Defaults to {_FULL_TCGA_PARQUET_CACHE}."
        ),
    )
    p.add_argument(
        "--barcode-to-project",
        type=Path,
        default=_DEFAULT_BARCODE_TO_PROJECT,
        help="Barcode→TCGA project pickle for ground-truth labels.",
    )
    p.add_argument(
        "--include-normals",
        action="store_true",
        help=(
            "When --use-full-tcga-matrix is set, include sample-type 11 "
            "(matched normal) samples — recorded under a NORMAL: prefix."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Multiprocessing workers for --use-full-tcga-matrix (default 1).",
    )
    p.add_argument(
        "--max-tcga-samples",
        type=int,
        default=0,
        help=(
            "Cap on total TCGA samples classified under "
            "--use-full-tcga-matrix (0 = no cap)."
        ),
    )
    p.add_argument(
        "--skip-hpa",
        action="store_true",
        help="Skip the HPA normal-tissue baseline track.",
    )
    p.add_argument(
        "--hpa-tissues",
        type=str,
        default="",
        help="Comma-separated HPA tissues to include (default: all).",
    )
    p.add_argument(
        "--local-manifest",
        type=Path,
        default=None,
        help=(
            "Pirlygenes-style local_reports/<run>/manifest.json to replay "
            "without overrides. Defaults to the bundled "
            f"{_DEFAULT_LOCAL_MANIFEST.name} when --include-local-no-override "
            "is passed."
        ),
    )
    p.add_argument(
        "--include-local-no-override",
        action="store_true",
        help=(
            "Run the local 17-sample no-override replay track using the "
            "bundled or --local-manifest path."
        ),
    )
    p.add_argument(
        "--local-data-root",
        type=Path,
        default=None,
        help=(
            "If given, scan this dir for analysis.json files (legacy local "
            "replay; complements --include-local-no-override)."
        ),
    )
    p.add_argument(
        "--expected-labels",
        type=Path,
        default=None,
        help="Optional TSV of sample_id\\texpected_code for local replay.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("calibration-report.json"),
    )
    p.add_argument(
        "--out-tsv",
        type=Path,
        default=None,
        help="Optional flat per-sample TSV for the TCGA track.",
    )
    p.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help=(
            "Compare against a previous calibration JSON: emits per-cohort "
            "and summary deltas under report['vs_baseline']. Use this to "
            "guard against silent regressions when scoring constants "
            "change."
        ),
    )
    args = p.parse_args(argv)

    report: dict[str, Any] = {
        "trufflepig_version": _trufflepig_version(),
        "pirlygenes_version": _pirlygenes_version(),
        "decomposition_parameters": _decomposition_parameters(),
    }
    if not args.skip_tcga:
        sample_limit = (
            None if args.samples_per_cohort <= 0 else args.samples_per_cohort
        )
        cohorts = (
            {c.strip().upper() for c in args.cohorts.split(",") if c.strip()}
            if args.cohorts
            else None
        )
        if args.use_full_tcga_matrix:
            report["tcga_full_per_sample"] = _tcga_full_per_sample_track(
                gz_path=args.full_tcga_gz,
                parquet_cache=args.full_tcga_parquet_cache,
                barcode_to_project_path=args.barcode_to_project,
                workers=max(1, args.workers),
                max_samples=max(0, args.max_tcga_samples),
                cohorts=cohorts,
                include_normals=args.include_normals,
            )
        else:
            report["tcga_per_sample"] = _tcga_per_sample_track(
                args.eval_tpm,
                args.eval_samples,
                sample_limit_per_cohort=sample_limit,
                cohorts=cohorts,
            )
    if not args.skip_hpa:
        tissues = [t.strip() for t in args.hpa_tissues.split(",") if t.strip()]
        report["hpa_normal_baseline"] = _hpa_normal_track(tissues or None)
    if args.include_local_no_override:
        manifest_path = args.local_manifest or _DEFAULT_LOCAL_MANIFEST
        if not manifest_path.exists():
            raise SystemExit(
                f"--include-local-no-override needs an existing manifest, "
                f"got {manifest_path}"
            )
        report["local_no_override"] = _local_no_override_track(manifest_path)
    if args.local_data_root is not None:
        report["local_replay"] = _local_replay_track(
            args.local_data_root, args.expected_labels
        )

    if args.baseline_json is not None:
        if not args.baseline_json.exists():
            raise SystemExit(f"baseline JSON not found: {args.baseline_json}")
        baseline = json.loads(args.baseline_json.read_text())
        report["vs_baseline"] = _baseline_diff(report, baseline)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[done] wrote {args.out_json}", file=sys.stderr)

    if args.out_tsv is not None and "tcga_per_sample" in report:
        per_sample = report["tcga_per_sample"]["per_sample"]
        if per_sample:
            df = pd.DataFrame(per_sample)
            df.to_csv(args.out_tsv, sep="\t", index=False)
            print(f"[done] wrote {args.out_tsv}", file=sys.stderr)

    for key in ("tcga_per_sample", "tcga_full_per_sample"):
        if key not in report:
            continue
        s = report[key]["summary"]
        layers = s.get("layer_breakdown") or {}
        n = s.get("n_samples") or s.get("n_evaluated")
        print(
            f"[summary] {key}: n={n} "
            f"broad_top1={layers.get('broad_top1_accuracy', s.get('top1_accuracy')):.3f} "
            f"broad_top3={layers.get('broad_top3_accuracy', s.get('top3_accuracy')):.3f} "
            f"consolidated_top1="
            f"{layers.get('consolidated_top1_accuracy', s.get('top1_accuracy')):.3f} "
            f"(Δ={layers.get('delta_consolidated_vs_broad_top1', 0.0):+.3f})"
        )
        if layers.get("flip_counts"):
            print(f"[summary] {key} flips: {layers['flip_counts']}")
        if (s.get("per_selector") or {}):
            print(f"[summary] {key} per-selector: {s['per_selector']}")
    if "hpa_normal_baseline" in report:
        s = report["hpa_normal_baseline"]["summary"]
        print(f"[summary] HPA normal baseline: {s}")
    if "local_no_override" in report:
        s = report["local_no_override"]["summary"]
        layers = s.get("layer_breakdown") or {}
        print(
            f"[summary] Local no-override replay: "
            f"n={s['n_evaluated']} code_match={s['code_match_rate']} "
            f"full_match={s['full_match_rate']} "
            f"broad_only={layers.get('broad_only_code_match_rate')} "
            f"(consolidation lifted {layers.get('n_lifted_by_consolidation')} "
            "samples)"
        )
        if (s.get("per_selector") or {}):
            print(f"[summary] Local per-selector: {s['per_selector']}")
    if "local_replay" in report:
        s = report["local_replay"]["summary"]
        print(f"[summary] Local replay: {s}")
    return 0


def _baseline_diff(report: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Compute per-track deltas between the current report and a baseline."""
    diff: dict[str, Any] = {
        "baseline_trufflepig_version": baseline.get("trufflepig_version"),
        "baseline_pirlygenes_version": baseline.get("pirlygenes_version"),
    }

    cur_tcga = report.get("tcga_per_sample") or {}
    base_tcga = baseline.get("tcga_per_sample") or {}
    if cur_tcga and base_tcga:
        cur_sum = cur_tcga.get("summary") or {}
        base_sum = base_tcga.get("summary") or {}
        diff["tcga_summary_delta"] = {
            "top1_accuracy": _delta(cur_sum.get("top1_accuracy"), base_sum.get("top1_accuracy")),
            "top3_accuracy": _delta(cur_sum.get("top3_accuracy"), base_sum.get("top3_accuracy")),
        }
        per_cohort_delta = {}
        cur_pc = cur_tcga.get("per_cohort") or {}
        base_pc = base_tcga.get("per_cohort") or {}
        for code in sorted(set(cur_pc) | set(base_pc)):
            per_cohort_delta[code] = {
                "top1_accuracy": _delta(
                    (cur_pc.get(code) or {}).get("top1_accuracy"),
                    (base_pc.get(code) or {}).get("top1_accuracy"),
                ),
                "top3_accuracy": _delta(
                    (cur_pc.get(code) or {}).get("top3_accuracy"),
                    (base_pc.get(code) or {}).get("top3_accuracy"),
                ),
            }
        diff["tcga_per_cohort_delta"] = per_cohort_delta
        # Sample-level regression flags: samples that USED to be correct
        # and now are wrong.
        cur_by_id = {
            row.get("sample_id"): row for row in (cur_tcga.get("per_sample") or [])
        }
        regressions: list[dict[str, Any]] = []
        for row in base_tcga.get("per_sample") or []:
            sid = row.get("sample_id")
            if not sid:
                continue
            if not row.get("top1_match"):
                continue
            cur_row = cur_by_id.get(sid)
            if cur_row is None:
                continue
            if not cur_row.get("top1_match"):
                regressions.append(
                    {
                        "sample_id": sid,
                        "expected_code": row.get("expected_code"),
                        "baseline_top": row.get("top_code"),
                        "current_top": cur_row.get("top_code"),
                        "current_top3": cur_row.get("top3"),
                    }
                )
        diff["tcga_sample_regressions"] = regressions

    cur_local = report.get("local_no_override") or {}
    base_local = baseline.get("local_no_override") or {}
    if cur_local and base_local:
        diff["local_summary_delta"] = {
            "code_match_rate": _delta(
                (cur_local.get("summary") or {}).get("code_match_rate"),
                (base_local.get("summary") or {}).get("code_match_rate"),
            ),
            "full_match_rate": _delta(
                (cur_local.get("summary") or {}).get("full_match_rate"),
                (base_local.get("summary") or {}).get("full_match_rate"),
            ),
        }
        cur_local_by_name = {
            row.get("name"): row for row in (cur_local.get("per_sample") or [])
        }
        regressions = []
        for row in base_local.get("per_sample") or []:
            name = row.get("name")
            if not name or not row.get("code_match"):
                continue
            cur_row = cur_local_by_name.get(name)
            if cur_row is None or not cur_row.get("code_match"):
                regressions.append(
                    {
                        "name": name,
                        "baseline_inferred": row.get("inferred_cancer_type"),
                        "current_inferred": (cur_row or {}).get("inferred_cancer_type"),
                        "expected": row.get("expected_cancer_type"),
                    }
                )
        diff["local_sample_regressions"] = regressions
    return diff


def _delta(current: Any, baseline: Any) -> Any:
    if current is None or baseline is None:
        return None
    try:
        return round(float(current) - float(baseline), 4)
    except (TypeError, ValueError):
        return None


def _trufflepig_version() -> str:
    try:
        from trufflepig.version import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - version metadata is best-effort
        return "unknown"


def _pirlygenes_version() -> str:
    try:
        import pirlygenes

        return str(getattr(pirlygenes, "__version__", "unknown"))
    except Exception:  # pragma: no cover
        return "unavailable"


def _decomposition_parameters() -> dict[str, Any]:
    """Snapshot the scoring constants the decomposition engine reads."""
    try:
        from trufflepig.tumor_purity import TUMOR_PURITY_PARAMETERS

        keys = ("scoring", "family_scoring", "purity_anchor")
        return {
            k: TUMOR_PURITY_PARAMETERS[k]
            for k in keys
            if k in TUMOR_PURITY_PARAMETERS
        }
    except Exception:  # pragma: no cover
        return {}


if __name__ == "__main__":
    raise SystemExit(main())
