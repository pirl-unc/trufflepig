#!/usr/bin/env python3
"""Diagnose WHY specific per-sample calls missed: re-run each flagged sample with instrumentation and
report the failure stage (bulk classifier vs evidence channel vs veto) + the signals that drove it.

Reads the per-sample output (/tmp/per_sample2.out), finds the samples whose call hit a given
match-level (default 'miss'), and for each dumps: the bulk top-3 candidates, compartment_call, the
evidence channel that won, and the decomposition lineage_fit per mode.

Run:  python3 scripts/diagnose_misses.py [match_level]   # default: miss
"""
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from trufflepig.cancer_type_centroid import compartment_call
from trufflepig.cancer_type_evidence import select_report_scope_from_evidence
from trufflepig.expression_decomposition import decompose_expression
from trufflepig.tumor_purity import _build_sample_tpm_by_symbol, analyze_sample

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_per_sample_confusion import _clean_cohort, match_level, name  # noqa: E402

MODES = ["solid", "mesenchymal", "heme", "embryonal"]
OUTPUT = "/tmp/per_sample2.out"


def _parse_target_samples(level="miss"):
    """Yield (truth_type, sample_index, call) for samples whose call scored the given match level."""
    for line in open(OUTPUT):
        m = re.match(r"\s*\[\d+/\d+\]\s+(\S+)\s+->\s+\[(.*)\]", line)
        if not m:
            continue
        truth = m.group(1)
        calls = [c.strip().strip("'") for c in m.group(2).split(",")]
        for i, call in enumerate(calls):
            if match_level(call, truth) == level:
                yield truth, i, call


def diagnose(df):
    analysis = analyze_sample(df)
    bulk = analysis.get("cancer_type")
    trace = analysis.get("candidate_trace") or []
    top = [(c.get("code"), round(float(c.get("score") or 0), 3)) for c in trace[:4]]
    sym = _build_sample_tpm_by_symbol(df)
    call = compartment_call(sym)
    scope = select_report_scope_from_evidence(df, analysis)
    selected = scope.get("selected") or {}
    decomposition = decompose_expression(sym, cancer=None, run_all=True)
    fits = {m: decomposition["modes"].get(m, {}).get("lineage_fit") for m in MODES}
    return {
        "bulk": bulk,
        "bulk_top": top,
        "compartment": call.get("compartment"),
        "compartment_confident": bool(call.get("confident")),
        "compartment_margin": round(float(call.get("margin") or 0), 3),
        "compartment_runner_up": call.get("runner_up"),
        "evidence_call": selected.get("cancer_type"),
        "evidence_channel": selected.get("selected_by"),
        "decomp_mode": decomposition.get("selected_mode"),
        "lineage_fit": {m: (round(v, 1) if isinstance(v, (int, float)) else v) for m, v in fits.items()},
    }


def main():
    level = sys.argv[1] if len(sys.argv) > 1 else "miss"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    targets = list(_parse_target_samples(level))
    if limit:
        # one sample per distinct truth type first (breadth over depth), then fill to the cap
        seen, ordered = set(), []
        for t in targets:
            if t[0] not in seen:
                seen.add(t[0]); ordered.append(t)
        ordered += [t for t in targets if t not in ordered]
        targets = ordered[:limit]
    print(f"diagnosing {len(targets)} '{level}' samples\n")
    # cache cleaned cohorts so we only clean_tpm each type once
    cohort_cache = {}
    for truth, idx, call in targets:
        if truth not in cohort_cache:
            cohort_cache[truth] = _clean_cohort(truth)
        ensembl, symbols, cleaned, sample_cols = cohort_cache[truth]
        if idx >= len(sample_cols):
            continue
        df = pd.DataFrame({"ensembl_gene_id": ensembl, "gene_symbol": symbols,
                           "TPM": cleaned[sample_cols[idx]].values})
        d = diagnose(df)
        print(f"=== {name(truth)} [{truth}] sample#{idx} -> CALLED {name(call)} [{call}] ===")
        print(f"  bulk classifier: {d['bulk']}   top: {d['bulk_top']}")
        print(f"  compartment_call: {d['compartment']} (confident={d['compartment_confident']}, "
              f"margin={d['compartment_margin']}, runner_up={d['compartment_runner_up']})")
        print(f"  evidence: {d['evidence_call']} via '{d['evidence_channel']}'")
        print(f"  decomposition mode={d['decomp_mode']}  lineage_fit={d['lineage_fit']}")
        print()
    return targets


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
