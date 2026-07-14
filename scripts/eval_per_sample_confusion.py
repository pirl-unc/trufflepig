#!/usr/bin/env python3
"""Per-SAMPLE no-hint cancer-type calls across every representative cohort (~565 individual samples),
and the confusion between the truth type and the FINAL call — i.e. what we ACTUALLY call each sample
after the full pipeline (bulk classifier → evidence selector → lineage veto), reported at the ENTITY
level (not just lineage).

Match levels per sample:
  exact     — final call == truth type
  subtype   — parent↔child (e.g. SCLC vs SCLC_ASCL1, COAD vs CRC)
  sibling   — same immediate registry parent (e.g. COAD vs READ under CRC)
  lineage   — same lineage mode only (solid/mesenchymal/heme/embryonal), different entity
  miss      — different lineage

Run:  python3 scripts/eval_per_sample_confusion.py
"""
import collections
import argparse
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from oncoref.normalization import clean_tpm
from pirlygenes.expression.accessors import (
    available_representative_cohorts,
    representative_cohort_samples,
)
from pirlygenes.gene_sets_cancer import cancer_lineage_group, cancer_type_registry
from trufflepig.cancer_ontology import cancer_codes_entity_compatible
from trufflepig.cancer_type_evidence import _primary_tissue_key_for_code
from trufflepig.expression_decomposition import _group_to_mode
from trufflepig.cancer_type_signal_matrix import (
    build_cancer_type_signal_matrix,
    build_signal_sample_summary,
    build_signal_matrix_summary_markdown,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_nohint_validation import full_granularity_analysis, full_granularity_call  # noqa: E402

_REGISTRY = {r["code"]: r for r in cancer_type_registry().to_dict("records")}
_NAME = {code: str(row.get("name") or code) for code, row in _REGISTRY.items()}
_REG_UPPER = {code.upper(): code for code in _REGISTRY}
_FIXTURE_ADJUDICATIONS = (
    Path(__file__).resolve().parent
    / "data"
    / "representative-fixture-adjudications.csv"
)


def name(code):
    """Full cancer-type name for a code (e.g. ACC → Adrenocortical Carcinoma)."""
    return _NAME.get(_canon(code), code)


def _canon(code):
    """Resolve a call to its registry code: case-insensitive, and for a molecular subtype not in the
    registry (e.g. LUAD_KRAS_STK11 from the subtype-signature layer) fall back to the longest
    registry-code prefix (→ LUAD). Returns the code unchanged if nothing matches."""
    if not code:
        return ""
    upper = str(code).upper()
    if upper in _REG_UPPER:
        return _REG_UPPER[upper]
    parts = upper.split("_")
    for i in range(len(parts) - 1, 0, -1):
        candidate = "_".join(parts[:i])
        if candidate in _REG_UPPER:
            return _REG_UPPER[candidate]
    return code


def _lineage(code):
    group = cancer_lineage_group(code) if code else None
    return _group_to_mode(group) if group else None


def _parent(code):
    value = str(_REGISTRY.get(code, {}).get("parent_code") or "").strip().upper()
    return "" if value in ("", "NAN", "NONE") else value


def _organ(code):
    try:
        return _primary_tissue_key_for_code(code) or ""
    except (KeyError, ValueError):
        return ""


def match_level(call, truth):
    """exact > subtype (parent↔child) > sibling (shared parent, e.g. COAD/READ → CRC) >
    lineage (same compartment: solid/mesenchymal/heme/embryonal, different entity, e.g. SKCM→GBM) >
    organ (SAME primary site but DIFFERENT compartment — a co-located cross-lineage confusion, e.g. a
    uterine sarcoma vs endometrial carcinoma; the 'less bad' cross-compartment mixup) > miss.

    NOT using the registry ``family`` (too broad: 'carcinoma-gi' lumps colon+stomach), so COAD→READ
    scores 'sibling' while COAD→STAD scores 'lineage'. Note: generic SARC_* codes have no organ, so a
    sarcoma↔carcinoma mixup only scores 'organ' when the sarcoma code is site-specific (e.g. UCS)."""
    if not call:
        return "miss"
    cc, ct = _canon(call), _canon(truth)
    if cc.upper() == ct.upper():                              # same registry entity (case-insensitive)
        return "exact" if str(call).upper() == str(truth).upper() else "subtype"
    if _parent(cc) == ct.upper() or _parent(ct) == cc.upper():
        return "subtype"
    pc, pt = _parent(cc), _parent(ct)
    if pc and pc == pt:
        return "sibling"
    # The registry can express entity compatibility through more than one
    # parent hop (for example COAD_MSI -> COAD -> CRC).  The old one-level
    # harness under-counted those calls even though the production ontology
    # explicitly treats them as entity-compatible.
    if cancer_codes_entity_compatible(cc, ct):
        return "subtype"
    if _lineage(cc) and _lineage(cc) == _lineage(ct):
        return "lineage"
    oc, ot = _organ(cc), _organ(ct)
    if oc and oc == ot:
        return "organ"
    return "miss"


def load_fixture_adjudications(path=None):
    """Load independent source/QC adjudications for the representative gate.

    These rows never alter a production call.  They only let the evaluation
    distinguish nominal artifact labels from qualified, source-grouped truth:
    exact duplicate aliases count once, invalid fixtures are excluded, and a
    documented dual-lineage tumor can name both acceptable lineage modes.
    """
    source = Path(path) if path is not None else _FIXTURE_ADJUDICATIONS
    if not source.exists():
        return {}
    frame = pd.read_csv(source, keep_default_na=False)
    required = {
        "representative_id",
        "source_group_id",
        "truth_code",
        "benchmark_eligible",
        "allowed_lineage_modes",
        "adjudication",
        "source_issue",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"fixture adjudication file is missing columns: {sorted(missing)}"
        )
    out = {}
    for row in frame.to_dict("records"):
        representative_id = str(row["representative_id"]).strip()
        if not representative_id or representative_id in out:
            raise ValueError(
                "fixture adjudication representative_id values must be non-empty and unique"
            )
        eligible = str(row["benchmark_eligible"]).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        modes = tuple(
            mode.strip()
            for mode in str(row["allowed_lineage_modes"]).split(";")
            if mode.strip()
        )
        out[representative_id] = {
            **row,
            "benchmark_eligible": eligible,
            "allowed_lineage_modes": modes,
        }
    return out


def qualified_gate_summary(results, adjudications=None):
    """Score qualified representatives once per independent source group.

    Returns raw counts plus the exact excluded/adjudicated rows so a 100% result
    cannot conceal why its denominator differs from the nominal artifact gate.
    Group correctness is conservative: every alias in a source group must pass.
    """
    adjudications = (
        load_fixture_adjudications()
        if adjudications is None
        else adjudications
    )
    groups = collections.defaultdict(list)
    excluded = []
    adjudicated = []
    for truth, sample, call, level in results:
        row = adjudications.get(sample, {})
        if row and str(row.get("truth_code") or "") != str(truth):
            raise ValueError(
                f"fixture adjudication truth mismatch for {sample}: "
                f"{row.get('truth_code')} != {truth}"
            )
        if row and not bool(row.get("benchmark_eligible")):
            excluded.append(
                {
                    "truth": truth,
                    "sample": sample,
                    "call": call,
                    "adjudication": row.get("adjudication"),
                    "source_issue": row.get("source_issue"),
                }
            )
            continue
        allowed_modes = tuple(row.get("allowed_lineage_modes") or ())
        nominal_lineage_correct = level in {
            "exact",
            "subtype",
            "sibling",
            "lineage",
        }
        lineage_correct = bool(
            nominal_lineage_correct
            or (allowed_modes and _lineage(call) in allowed_modes)
        )
        if row:
            adjudicated.append(
                {
                    "truth": truth,
                    "sample": sample,
                    "call": call,
                    "nominal_level": level,
                    "allowed_lineage_modes": list(allowed_modes),
                    "lineage_correct": lineage_correct,
                    "adjudication": row.get("adjudication"),
                    "source_issue": row.get("source_issue"),
                }
            )
        group_id = str(row.get("source_group_id") or sample)
        groups[group_id].append(
            {
                "lineage_correct": lineage_correct,
                "entity_correct": level in {"exact", "subtype", "sibling"},
            }
        )

    lineage_correct = sum(
        all(member["lineage_correct"] for member in members)
        for members in groups.values()
    )
    entity_correct = sum(
        all(member["entity_correct"] for member in members)
        for members in groups.values()
    )
    return {
        "n": len(groups),
        "lineage_correct": lineage_correct,
        "entity_correct": entity_correct,
        "excluded": excluded,
        "adjudicated": adjudicated,
        "aliases_collapsed": sum(len(members) - 1 for members in groups.values()),
    }


# data/format failures we tolerate per sample (anything else propagates so genuine bugs surface)
EXPECTED_SAMPLE_ERRORS = (ValueError, KeyError, FloatingPointError)


def _clean_cohort(type_code):
    """Return (ensembl_ids, symbols, cleaned_dataframe[sample_columns]) for a cohort."""
    raw = representative_cohort_samples(type_code).drop_duplicates("Ensembl_Gene_ID")
    sample_cols = [c for c in raw.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    gene_table = pd.DataFrame(
        {"Ensembl_Gene_ID": raw["Ensembl_Gene_ID"].values, "Symbol": raw["Symbol"].values}
    )
    cleaned = clean_tpm(
        raw.set_index("Ensembl_Gene_ID")[sample_cols].astype(float),
        gene_table=gene_table.set_index(raw.index),
    )
    return raw["Ensembl_Gene_ID"].values, raw["Symbol"].values, cleaned, sample_cols


def call_each_sample(
    type_code,
    *,
    collect_signal_matrix=False,
    collect_signal_summary=False,
):
    """Yield (sample_column, final_call) for every individual sample in a cohort."""
    ensembl, symbols, cleaned, sample_cols = _clean_cohort(type_code)
    for col in sample_cols:
        df = pd.DataFrame(
            {"ensembl_gene_id": ensembl, "gene_symbol": symbols, "TPM": cleaned[col].values}
        )
        try:
            if collect_signal_matrix or collect_signal_summary:
                _bulk, final_call, analysis = full_granularity_analysis(df)
                matrix = build_cancer_type_signal_matrix(
                    analysis,
                    sample_id=f"{type_code}:{col}",
                )
                if collect_signal_summary:
                    summary = build_signal_sample_summary(matrix)
                    summary.insert(1, "truth_type", type_code)
                    summary.insert(2, "truth_sample", col)
                    summary_records = summary.to_dict("records")
                else:
                    summary_records = []
                if collect_signal_matrix:
                    matrix.insert(1, "truth_type", type_code)
                    matrix.insert(2, "truth_sample", col)
                    matrix_records = matrix.to_dict("records")
                else:
                    matrix_records = []
            else:
                _bulk, final_call = full_granularity_call(df)
                matrix_records = []
                summary_records = []
        except EXPECTED_SAMPLE_ERRORS as exc:
            final_call = f"ERROR:{str(exc)[:30]}"
            matrix_records = []
            summary_records = []
        yield col, final_call, matrix_records, summary_records


def _eval_one_cohort(truth):
    """Worker: all (sample, final_call) for one cohort. Top-level + picklable for the pool."""
    try:
        return truth, [
            (sample, call)
            for sample, call, _matrix, _summary in call_each_sample(truth)
        ]
    except Exception as exc:  # noqa: BLE001 — a cohort that fails to load shouldn't kill the run
        return truth, [("<cohort-error>", f"ERROR:{str(exc)[:40]}")]


def _eval_one_cohort_with_signal_summary(truth):
    """Worker: calls plus compact signal summary rows for one cohort."""
    try:
        return truth, list(call_each_sample(truth, collect_signal_summary=True))
    except Exception as exc:  # noqa: BLE001 — a cohort that fails to load shouldn't kill the run
        return truth, [("<cohort-error>", f"ERROR:{str(exc)[:40]}", [], [])]


def _default_workers():
    """min(cpu-2, free_RAM/1.8GB) — each worker re-imports trufflepig + the reference matrices
    (~1.5-1.8 GB RSS), so cap by memory like ``test.sh`` to avoid OOM on a fat machine."""
    cpu = os.cpu_count() or 4
    cap = max(1, cpu - 2)
    try:  # macOS: free + inactive pages from vm_stat
        import re
        import subprocess
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = int(re.search(r"page size of (\d+)", out).group(1))
        free = sum(int(re.search(rf"Pages {k}:\s+(\d+)", out).group(1)) for k in ("free", "inactive"))
        gb = free * page / 1e9
        cap = min(cap, max(1, int(gb // 1.8)))
    except Exception:  # noqa: BLE001 — fall back to the cpu-based cap
        pass
    return cap


def main(workers=None, signal_matrix_out=None, signal_summary_out=None):
    types = sorted(available_representative_cohorts())
    collect_signal_matrix = bool(signal_matrix_out)
    collect_signal_summary = bool(signal_summary_out) or collect_signal_matrix
    if collect_signal_matrix and (workers is None or workers != 1):
        print("# --signal-matrix-out requested; forcing serial run so evidence rows stay local", flush=True)
        workers = 1
    if workers is None:
        workers = _default_workers()
    results = []  # (truth_type, sample, final_call, match_level)
    signal_frames = []
    signal_summary_frames = []
    print(f"# {len(types)} cohorts, {workers} worker(s)", flush=True)
    if workers <= 1:
        stream = (
            (
                t,
                list(
                    call_each_sample(
                        t,
                        collect_signal_matrix=collect_signal_matrix,
                        collect_signal_summary=collect_signal_summary,
                    )
                ),
            )
            for t in types
        )
        pool = None
    else:
        # 'spawn' (macOS default) gives each worker a clean interpreter that re-imports the
        # reference matrices once; 'fork' would copy the parent's half-built caches.
        pool = mp.get_context("spawn").Pool(workers)
        worker = (
            _eval_one_cohort_with_signal_summary
            if collect_signal_summary
            else _eval_one_cohort
        )
        stream = pool.imap_unordered(worker, types)
    done = 0
    for truth, calls in stream:
        done += 1
        for row in calls:
            if len(row) == 4:
                sample, final_call, matrix_records, summary_records = row
            elif len(row) == 3:
                sample, final_call, matrix_records = row
                summary_records = []
            else:
                sample, final_call = row
                matrix_records = []
                summary_records = []
            results.append((truth, sample, final_call, match_level(final_call, truth)))
            if matrix_records:
                signal_frames.append(pd.DataFrame(matrix_records))
            if summary_records:
                signal_summary_frames.append(pd.DataFrame(summary_records))
        print(
            f"  [{done}/{len(types)}] {truth:16s} -> "
            f"{[row[1] for row in calls]}",
            flush=True,
        )
    if pool is not None:
        pool.close()
        pool.join()

    levels = collections.Counter(r[3] for r in results)
    n = len(results)
    compatible = levels["exact"] + levels["subtype"] + levels["sibling"]
    print(f"\n==== {n} samples across {len(types)} types ====")
    for level in ("exact", "subtype", "sibling", "lineage", "organ", "miss"):
        print(f"  {level:8s}: {levels[level]:4d}  ({100*levels[level]/n:.0f}%)")
    print(f"  entity-correct (exact+subtype+sibling): {compatible}/{n} ({100*compatible/n:.0f}%)")
    lineage_compatible = compatible + levels["lineage"]
    print(
        "  lineage-correct (same compartment): "
        f"{lineage_compatible}/{n} ({100*lineage_compatible/n:.0f}%)"
    )

    qualified = qualified_gate_summary(results)
    qn = qualified["n"]
    qentity = qualified["entity_correct"]
    qlineage = qualified["lineage_correct"]
    print("\n==== qualified, source-grouped gate ====")
    print(
        f"  entity-compatible: {qentity}/{qn} "
        f"({100*qentity/qn:.1f}%)"
    )
    print(
        f"  lineage-compatible: {qlineage}/{qn} "
        f"({100*qlineage/qn:.1f}%)"
    )
    print(
        f"  qualification accounting: {len(qualified['excluded'])} excluded, "
        f"{qualified['aliases_collapsed']} duplicate alias collapsed, "
        f"{len(qualified['adjudicated'])} adjudicated rows"
    )
    for row in qualified["excluded"]:
        print(
            f"    excluded {row['sample']}: {row['adjudication']} "
            f"({row['source_issue']})"
        )
    for row in qualified["adjudicated"]:
        print(
            f"    adjudicated {row['sample']}: {row['adjudication']} "
            f"allowed={','.join(row['allowed_lineage_modes'])} "
            f"({row['source_issue']})"
        )

    # confusion: per truth type, the distribution of final calls (only types with any non-exact call)
    by_type = collections.defaultdict(collections.Counter)
    for truth, _s, call, _lvl in results:
        by_type[truth][call] += 1
    print("\n==== per-type calls (types with ANY non-exact call) ====")
    print(f"{'truth':16s} {'n':>2s}  calls (count)")
    for truth in types:
        dist = by_type[truth]
        if list(dist) == [truth]:
            continue  # all exact, skip
        total = sum(dist.values())
        summary = "  ".join(f"{c}×{k}" for c, k in
                            sorted(((v, k) for k, v in dist.items()), reverse=True))
        print(f"{truth:16s} {total:>2d}  {summary}")

    # the mixup pairs (truth -> wrong call at miss/lineage level), most frequent first
    mixups = collections.Counter()
    for truth, _s, call, lvl in results:
        if lvl in ("lineage", "miss") and not str(call).startswith("ERROR"):
            mixups[(truth, call, lvl)] += 1
    print("\n==== cross-entity mixups (lineage-only or miss), most frequent ====")
    for (truth, call, lvl), cnt in mixups.most_common(30):
        print(f"  {cnt}×  {truth:16s} -> {call:16s} [{lvl}]")
    if signal_matrix_out and signal_frames:
        matrix = pd.concat(signal_frames, ignore_index=True)
        matrix.to_csv(signal_matrix_out, sep="\t", index=False)
        summary_path = str(signal_matrix_out).rsplit(".", 1)[0] + ".md"
        with open(summary_path, "w") as fh:
            fh.write(
                build_signal_matrix_summary_markdown(
                    matrix,
                    title="565 Representative-Sample Cancer-Type Signal Matrix Summary",
                )
            )
        print(f"\n[signal-matrix] wrote {signal_matrix_out}")
        print(f"[signal-matrix] wrote {summary_path}")
    if signal_summary_out and signal_summary_frames:
        summary = pd.concat(signal_summary_frames, ignore_index=True)
        summary.to_csv(signal_summary_out, sep="\t", index=False)
        print(f"\n[signal-summary] wrote {signal_summary_out}")
    return results


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="Per-sample cancer-type confusion over all cohorts.")
    _ap.add_argument("--workers", type=int, default=None,
                     help="parallel worker processes (default: memory-aware cap; 1 = serial)")
    _ap.add_argument(
        "--signal-matrix-out",
        default=None,
        help="Optional TSV path for the concatenated cancer-type signal matrix. Forces serial mode.",
    )
    _ap.add_argument(
        "--signal-summary-out",
        default=None,
        help=(
            "Optional compact one-row-per-sample cancer-type signal summary TSV. "
            "Can run in parallel."
        ),
    )
    _args = _ap.parse_args()
    sys.exit(
        0
        if main(
            workers=_args.workers,
            signal_matrix_out=_args.signal_matrix_out,
            signal_summary_out=_args.signal_summary_out,
        )
        is not None
        else 1
    )
