#!/usr/bin/env python3
"""A/B the centroid-authoritative fine-subtype resolver (#98) on the per-sample battery.

Runs the SAME samples through ``full_granularity_call`` under two (or more) arms in ONE
process — no stashing, no rebuild — by monkeypatching ``cancer_type_centroid.resolve_fine_
subtype`` (which the ranker imports per-call):

  * baseline   — resolver replaced by a passthrough (returns the curated/signature label
                 unchanged), reproducing pre-#98 behavior exactly.
  * margin=M   — the real resolver at margin M (default: the shipped constant).

For each arm it reports exact / entity-correct / lineage-correct counts over the battery,
and lists every sample whose final call FLIPPED vs baseline, classed improved / regressed /
neutral (by match level against truth). This is the validate-before-commit harness the issue
mandates: pick the margin that maximizes exact-match with zero regressions.

Run:
    python3 scripts/fine_subtype_ab.py                       # baseline vs shipped margin, all types
    python3 scripts/fine_subtype_ab.py --margins 0.0 0.01 0.015 0.02 0.03
    python3 scripts/fine_subtype_ab.py --types SARC BRCA COAD NET_PANCREAS
    python3 scripts/fine_subtype_ab.py --children-only       # only types that HAVE subtypes (fast)
"""
import argparse
import collections
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import trufflepig.cancer_type_centroid as ctc
from pirlygenes.expression.accessors import available_representative_cohorts
from trufflepig.cancer_ontology import cancer_type_subtypes_of

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_nohint_validation import full_granularity_call  # noqa: E402
from eval_per_sample_confusion import _clean_cohort, match_level  # noqa: E402

_REAL_RESOLVER = ctc.resolve_fine_subtype
_MEDOID_ONLY = False


def _passthrough(broad_code, cen_corr, current_subtype=None, margin=None):
    """Pre-#98 behavior: never override the curated/signature label."""
    return current_subtype


def _at_margin(margin):
    def _resolver(broad_code, cen_corr, current_subtype=None, _m=margin):
        return _REAL_RESOLVER(broad_code, cen_corr, current_subtype=current_subtype, margin=_m)

    return _resolver


def _final_calls_for_arm(samples, resolver):
    """{(_truth, sample): final_call} for every sample, with the resolver patched in."""
    ctc.resolve_fine_subtype = resolver
    try:
        out = {}
        for truth, sample_id, df in samples:
            try:
                _bulk, call = full_granularity_call(df)
            except (ValueError, KeyError, FloatingPointError) as exc:
                call = f"ERROR:{str(exc)[:30]}"
            out[(truth, sample_id)] = call
        return out
    finally:
        ctc.resolve_fine_subtype = _REAL_RESOLVER


def _load_samples(types):
    samples = []
    for truth in types:
        try:
            ensembl, symbols, cleaned, cols = _clean_cohort(truth)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {truth}: {str(e)[:50]}", file=sys.stderr)
            continue
        if _MEDOID_ONLY:  # one representative (cohort-mean) sample per type — ~10x faster
            df = pd.DataFrame({"ensembl_gene_id": ensembl, "gene_symbol": symbols,
                               "TPM": cleaned[cols].mean(axis=1).values})
            samples.append((truth, f"{truth}_medoid", df))
            continue
        for col in cols:
            df = pd.DataFrame(
                {"ensembl_gene_id": ensembl, "gene_symbol": symbols, "TPM": cleaned[col].values}
            )
            samples.append((truth, col, df))
    return samples


def _score(calls):
    levels = collections.Counter(match_level(c, t) for (t, _s), c in calls.items())
    n = sum(levels.values()) or 1
    exact = levels["exact"]
    entity = levels["exact"] + levels["subtype"] + levels["sibling"]
    lineage = n - levels["miss"] - levels["organ"]
    return exact, entity, lineage, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--types", nargs="*", default=None, help="cohort codes (default: all)")
    ap.add_argument("--children-only", action="store_true",
                    help="restrict to broad types that have reference subtypes (where the resolver can act)")
    ap.add_argument("--medoid", action="store_true",
                    help="one cohort-mean medoid per type instead of every sample (~10x faster, full lineage coverage)")
    ap.add_argument("--margins", nargs="*", type=float, default=[None],
                    help="resolver margins to test (default: shipped constant). 'baseline' arm is always included")
    args = ap.parse_args()

    global _MEDOID_ONLY
    _MEDOID_ONLY = bool(args.medoid)

    types = args.types or sorted(available_representative_cohorts())
    if args.children_only:
        ref = set(ctc._bulk_centroids()[0].columns.astype(str))
        broad = {c.split("_")[0] for c in ref}
        keep = set()
        for b in broad:
            kids = [c for c in cancer_type_subtypes_of(b) if c in ref]
            if len(kids) >= 2:
                keep |= {b, *kids}
        types = [t for t in types if t in keep or t.split("_")[0] in keep]

    print(f"loading {len(types)} cohorts ...", flush=True)
    samples = _load_samples(types)
    print(f"{len(samples)} samples\n", flush=True)

    base = _final_calls_for_arm(samples, _passthrough)
    be, bent, blin, n = _score(base)
    print(f"{'arm':>12s}   {'exact':>12s}  {'entity':>12s}  {'lineage':>12s}")
    print(f"{'baseline':>12s}   {be:4d} ({100*be/n:2.0f}%)    {bent:4d} ({100*bent/n:2.0f}%)   {blin:4d} ({100*blin/n:2.0f}%)")

    for m in args.margins:
        arm = _final_calls_for_arm(samples, _at_margin(m) if m is not None else _REAL_RESOLVER)
        e, ent, lin, _ = _score(arm)
        label = "shipped" if m is None else f"margin={m}"
        de, dent, dlin = e - be, ent - bent, lin - blin
        print(f"{label:>12s}   {e:4d} ({de:+d})       {ent:4d} ({dent:+d})      {lin:4d} ({dlin:+d})")

        flips = [(t, s, base[(t, s)], arm[(t, s)]) for (t, s) in base if base[(t, s)] != arm[(t, s)]]
        improved = regressed = neutral = 0
        detail = []
        for t, s, b, a in flips:
            lb, la = match_level(b, t), match_level(a, t)
            order = ["miss", "organ", "lineage", "sibling", "subtype", "exact"]
            verdict = ("improved" if order.index(la) > order.index(lb)
                       else "regressed" if order.index(la) < order.index(lb) else "neutral")
            improved += verdict == "improved"
            regressed += verdict == "regressed"
            neutral += verdict == "neutral"
            detail.append((verdict, t, s, b, a, lb, la))
        print(f"               flips: {len(flips)}  (improved {improved}, regressed {regressed}, neutral {neutral})")
        for verdict in ("regressed", "improved", "neutral"):
            for v, t, s, b, a, lb, la in detail:
                if v == verdict:
                    print(f"                 [{v:9s}] {t:14s} {s[:18]:18s} {b} ({lb}) -> {a} ({la})")
        print()


if __name__ == "__main__":
    main()
