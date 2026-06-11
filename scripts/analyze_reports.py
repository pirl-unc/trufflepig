#!/usr/bin/env python3
"""Score a directory of trufflepig reports against what we know about each sample.

Generic correctness / regression harness. Given a tree of report workspaces and
a notion of the *expected* cancer type for each one, it reports how often the
"Working cancer call" is compatible with the truth, and — if a baseline tree is
supplied — which calls changed and whether each change was an improvement or a
regression.

It is parametric in two things:

1. **Where the reports live** — ``--reports DIR`` (and optional ``--baseline DIR``
   to diff against a previous run). DIR is a tree of ``<sample>/<sample>-analysis.md``
   workspaces, optionally nested one level under a grouping dir
   (``<group>/<sample>/<sample>-analysis.md``).

2. **What we know about each sample** — the ground truth. Two sources, in order
   of precedence:
     * ``--truth FILE`` — JSON or CSV mapping a sample id (or a group code) to
       ``{expected, notes}``. This is the place to record curated knowledge about
       real samples ("alvin -> SARC, EGFR-KDD pediatric spindle cell sarcoma").
     * ``--infer-expected`` — derive the expected code from the grouping dir name
       (the per-code *representative* convention: ``SARC_OS/SARC_OS_rep01/...`` ->
       expected ``SARC_OS``).

"Compatible" is deliberately lenient about the fine/coarse axis: an exact match,
a parent<->subtype prefix relationship (``HNSC`` ~ ``HNSC_HPV``), the same
top-level registry parent (``COAD``/``READ`` -> ``CRC``), a shared base token, or
the same broad lineage all count. The point is to catch lineage-level mistakes
(a sarcoma called as a carcinoma), not to punish COAD-vs-READ ties.

Examples
--------
    # per-code representative sweep, lift vs the previous run
    scripts/analyze_reports.py --reports /tmp/rep_reports_all \
        --baseline /tmp/rep_reports_baseline --infer-expected

    # real local reports, truth curated in a manifest
    scripts/analyze_reports.py --reports ~/trufflepig-local-reports/latest \
        --truth local_truth.json --per-sample out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

_CALL = re.compile(r"\*\*Working cancer call\*\*:\s*([A-Za-z][A-Za-z0-9_]+)")


def _registry_parent_map():
    """code -> parent_code, from pirlygenes (empty dict if unavailable)."""
    try:
        from pirlygenes.gene_sets_cancer import cancer_type_registry

        reg = cancer_type_registry()
        return dict(
            zip(reg["code"].astype(str), reg["parent_code"].fillna("").astype(str))
        )
    except Exception:
        return {}


def _broad_lineage_fn():
    try:
        from trufflepig.cancer_type_ontology import broad_lineage

        return broad_lineage
    except Exception:
        return lambda code: ""


class Compat:
    """Lenient compatibility test between a called code and an expected code."""

    def __init__(self, parent_map, broad_lineage):
        self._parent = parent_map
        self._broad = broad_lineage

    def _root(self, code):
        seen = code
        for _ in range(6):
            parent = self._parent.get(seen, "")
            if not parent or parent == "nan":
                break
            seen = parent
        return seen

    @staticmethod
    def _base(code):
        return code.split("_")[0] if code else ""

    def __call__(self, called, expected):
        if not called or not expected:
            return False
        if called == expected:
            return True
        # parent <-> subtype prefix (HNSC ~ HNSC_HPV, BRCA ~ BRCA_LumA)
        if called.startswith(expected + "_") or expected.startswith(called + "_"):
            return True
        # same top-level registry parent (COAD/READ -> CRC, subtypes -> their parent)
        root_e = self._root(expected)
        if root_e and self._root(called) == root_e:
            return True
        # same coarse base token
        base_e = self._base(expected)
        if base_e and self._base(called) == base_e:
            return True
        # same broad lineage (catches the cross-lineage mistakes we care about)
        bc, be = self._broad(called), self._broad(expected)
        if bc and bc == be and bc not in ("other",):
            return True
        return False


def _call_of(analysis_md: Path):
    try:
        text = analysis_md.read_text(encoding="utf-8")
    except OSError:
        return None
    m = _CALL.search(text)
    return m.group(1) if m else None


def _iter_samples(reports: Path):
    """Yield (group, sample_id, analysis_md_path) for every report under ``reports``.

    Supports both ``reports/<sample>/<sample>-analysis.md`` and the nested
    ``reports/<group>/<sample>/<sample>-analysis.md`` representative layout.
    """
    for md in sorted(reports.glob("**/*-analysis.md")):
        sample_id = md.name[: -len("-analysis.md")]
        # group = the dir just under ``reports`` (or the sample dir itself).
        rel = md.relative_to(reports).parts
        group = rel[0] if len(rel) > 1 else sample_id
        yield group, sample_id, md


def _load_truth(path: Path):
    """Load a sample-id/group -> {expected, notes} mapping from JSON or CSV."""
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        data = raw.get("samples", raw) if isinstance(raw, dict) else {}
        out = {}
        for key, val in data.items():
            if isinstance(val, str):
                out[key] = {"expected": val, "notes": ""}
            else:
                out[key] = {
                    "expected": str(val.get("expected", "")),
                    "notes": str(val.get("notes", "")),
                }
        return out
    out = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = row.get("sample") or row.get("sample_id") or row.get("code") or ""
            if not key:
                continue
            out[key] = {
                "expected": str(row.get("expected") or row.get("expected_code") or ""),
                "notes": str(row.get("notes") or ""),
            }
    return out


def _expected_for(group, sample_id, truth, infer):
    if sample_id in truth:
        return truth[sample_id]["expected"], truth[sample_id]["notes"]
    if group in truth:
        return truth[group]["expected"], truth[group]["notes"]
    if infer:
        return group, ""
    return "", ""


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Score trufflepig reports against known/expected cancer types.",
    )
    ap.add_argument("--reports", required=True, type=Path, help="report tree to score")
    ap.add_argument("--baseline", type=Path, help="prior report tree to diff against")
    ap.add_argument("--truth", type=Path, help="JSON/CSV of sample/group -> expected,notes")
    ap.add_argument(
        "--infer-expected",
        action="store_true",
        help="derive expected code from the grouping dir name (representative convention)",
    )
    ap.add_argument("--per-sample", type=Path, help="write a per-sample CSV here")
    ap.add_argument("--show", type=int, default=25, help="rows to print per section")
    args = ap.parse_args(argv)

    if not args.reports.is_dir():
        ap.error(f"--reports {args.reports} is not a directory")
    if not args.truth and not args.infer_expected:
        ap.error("supply --truth FILE and/or --infer-expected so we know the truth")

    truth = _load_truth(args.truth) if args.truth else {}
    compat = Compat(_registry_parent_map(), _broad_lineage_fn())

    rows = []
    for group, sample_id, md in _iter_samples(args.reports):
        expected, notes = _expected_for(group, sample_id, truth, args.infer_expected)
        new_call = _call_of(md)
        base_call = None
        if args.baseline:
            base_md = args.baseline / md.relative_to(args.reports)
            base_call = _call_of(base_md)
        rows.append(
            {
                "group": group,
                "sample": sample_id,
                "expected": expected,
                "baseline_call": base_call,
                "new_call": new_call,
                "notes": notes,
            }
        )

    scored = [r for r in rows if r["expected"]]
    n = len(scored)
    new_ok = sum(compat(r["new_call"], r["expected"]) for r in scored)
    print(f"reports scored: {n} (of {len(rows)} found; {len(rows) - n} without a known expected)")
    print(f"compatible-call rate (new): {new_ok}/{n} ({100 * new_ok / n:.0f}%)" if n else "no scored reports")

    if args.baseline:
        have_both = [r for r in scored if r["baseline_call"] and r["new_call"]]
        base_ok = sum(compat(r["baseline_call"], r["expected"]) for r in have_both)
        changed = [r for r in have_both if r["baseline_call"] != r["new_call"]]
        improved = [r for r in changed if compat(r["new_call"], r["expected"]) and not compat(r["baseline_call"], r["expected"])]
        regressed = [r for r in changed if compat(r["baseline_call"], r["expected"]) and not compat(r["new_call"], r["expected"])]
        print(f"compatible-call rate (baseline): {base_ok}/{len(have_both)} ({100 * base_ok / len(have_both):.0f}%)" if have_both else "")
        print(f"changed calls baseline->new: {len(changed)}  (improved {len(improved)}, REGRESSED {len(regressed)}, neutral {len(changed) - len(improved) - len(regressed)})")
        for r in regressed[: args.show]:
            print(f"   REGRESS {r['group']}/{r['sample']}: {r['baseline_call']} -> {r['new_call']}  (expected {r['expected']})")
        for r in improved[: args.show]:
            print(f"   improve {r['group']}/{r['sample']}: {r['baseline_call']} -> {r['new_call']}  (expected {r['expected']})")

    bad = [r for r in scored if not compat(r["new_call"], r["expected"])]
    print(f"\nincompatible new calls: {len(bad)}/{n}")
    by_group = Counter(r["group"] for r in bad)
    for group, cnt in by_group.most_common(args.show):
        egs = [r["new_call"] or "?" for r in bad if r["group"] == group][:3]
        print(f"   {group}: {cnt}  (e.g. {', '.join(egs)})")

    if args.per_sample:
        with args.per_sample.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["group", "sample", "expected", "baseline_call", "new_call", "compatible", "notes"],
            )
            w.writeheader()
            for r in rows:
                w.writerow({**r, "compatible": compat(r["new_call"], r["expected"]) if r["expected"] else ""})
        print(f"\nper-sample CSV -> {args.per_sample}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
