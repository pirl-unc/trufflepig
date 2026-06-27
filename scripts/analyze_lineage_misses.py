#!/usr/bin/env python3
"""Re-derive the FULL set of lineage misses from a finished eval_per_sample_confusion run, by
parsing its per-type calls section (no pipeline re-run) and mapping each call to its lineage mode.

A "miss" = the call's lineage mode (solid/mesenchymal/heme/embryonal/...) differs from truth's.
Groups misses by (truth_lineage -> call_lineage) so the error MODES are visible.

Run:  python3 scripts/analyze_lineage_misses.py /tmp/eval565_<hash>.out
"""
import sys, re, collections
import warnings
warnings.filterwarnings("ignore")

from pirlygenes.gene_sets_cancer import cancer_lineage_group
from trufflepig.expression_decomposition import _group_to_mode


def lineage(code):
    grp = cancer_lineage_group(code)
    return _group_to_mode(grp) if grp else None


def main(path):
    lines = open(path).read().splitlines()
    # isolate the per-type section
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("==== per-type calls"))
        end = next(i for i, l in enumerate(lines) if l.startswith("==== cross-entity"))
    except StopIteration:
        print("could not find per-type section"); return
    by_transition = collections.Counter()
    miss_examples = collections.defaultdict(list)
    total_miss = 0
    for l in lines[start + 2:end]:
        l = l.rstrip()
        if not l or l.startswith("truth"):
            continue
        m = re.match(r"^(\S+)\s+(\d+)\s+(.*)$", l)
        if not m:
            continue
        truth, _n, calls_str = m.group(1), m.group(2), m.group(3)
        tlin = lineage(truth)
        for tok in re.findall(r"(\d+)×(\S+)", calls_str):
            cnt, call = int(tok[0]), tok[1]
            clin = lineage(call)
            if tlin and clin and tlin != clin:
                total_miss += cnt
                by_transition[(tlin, clin)] += cnt
                miss_examples[(tlin, clin)].append(f"{truth}->{call}×{cnt}")
    print(f"\n=== {total_miss} lineage misses, by (truth_lineage -> call_lineage) ===")
    for (tl, cl), cnt in by_transition.most_common():
        print(f"  {cnt:3d}×  {tl:12s} -> {cl:12s}   {sorted(set(miss_examples[(tl, cl)]))}")
    print("\n=== miss count by TRUTH lineage (which lineages are hardest) ===")
    by_truth = collections.Counter()
    for (tl, _cl), cnt in by_transition.items():
        by_truth[tl] += cnt
    for tl, cnt in by_truth.most_common():
        print(f"  {cnt:3d}×  {tl}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/eval565_8ef9485.out")
