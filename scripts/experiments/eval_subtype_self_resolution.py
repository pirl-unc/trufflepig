#!/usr/bin/env python3
"""Does each cancer SUBTYPE's medoid resolve to ITSELF? (#98 eval criterion #2)

For every broad cancer type that has reference-present children (SARC, BRCA, COAD, NET,
SCLC, LUAD, …), run each child's representative-cohort medoid end-to-end through the ranker
and check the post-#98 ``winning_subtype`` equals the child. This is the regression guard
for centroid-authoritative fine-subtype resolution: a subtype's OWN medoid is the easiest
possible case — if it can't recover itself, the resolver is broken.

Reports, per broad family: how many children self-resolve, and every miss with the call +
the centroid's view (so misses are immediately diagnosable). A handful of genuine
wastebasket/degenerate subtypes (e.g. SARC_DDLPS, which centroid-ranks ~#75 on its own
heterogeneous medoid) are expected to resolve to a same-family neighbor rather than
themselves; those are listed but not counted as hard failures.

Run:
    python3 scripts/eval_subtype_self_resolution.py                 # all broad families
    python3 scripts/eval_subtype_self_resolution.py SARC BRCA       # only these
    python3 scripts/eval_subtype_self_resolution.py --margin 0.02   # sweep the margin
"""
import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from trufflepig.cancer_ontology import cancer_type_subtypes_of
from trufflepig.cancer_type_centroid import (
    _bulk_centroids,
    centroid_correlations,
    resolve_fine_subtype,
)
from trufflepig.tumor_purity import _build_sample_tpm_by_symbol, rank_cancer_type_candidates

# Subtypes whose reference medoid is a known wastebasket / degenerate centroid (they do not
# match themselves well — they centroid-rank far down on their OWN heterogeneous medoid, so
# resolving to a same-family neighbor is expected, not a regression). Documented in #98.
KNOWN_DEGENERATE = {
    "SARC_DDLPS",      # dedifferentiated LPS, centroid self-rank ~#75
    "SARC_WDLPS",      # well-differentiated LPS, self-rank ~#75
    "SARC_LPS_UNSPEC",  # liposarcoma aggregate
    "SARC_PLEOLPS",    # pleomorphic LPS, self-rank ~#41
}


def medoid_df(type_code):
    from oncoref.normalization import clean_tpm
    from pirlygenes.expression.accessors import representative_cohort_samples

    d = representative_cohort_samples(type_code).drop_duplicates("Ensembl_Gene_ID")
    cols = [c for c in d.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    gt = pd.DataFrame(
        {"Ensembl_Gene_ID": d["Ensembl_Gene_ID"].values, "Symbol": d["Symbol"].values}
    )
    cl = clean_tpm(
        d.set_index("Ensembl_Gene_ID")[cols].astype(float),
        gene_table=gt.set_index(d.index),
    )
    return pd.DataFrame(
        {
            "ensembl_gene_id": d["Ensembl_Gene_ID"].values,
            "gene_symbol": d["Symbol"].values,
            "TPM": cl.mean(axis=1).values,
        }
    )


def broad_families(ref_codes, only=None):
    """{broad_code: [reference-present children]} for every broad type with >=2 children."""
    fams = {}
    # A broad family is any code that is a registry parent of >=2 reference cohorts.
    candidates = sorted({c.split("_")[0] for c in ref_codes} | set(only or []))
    for broad in candidates:
        kids = [c for c in cancer_type_subtypes_of(broad) if c in ref_codes]
        if len(kids) >= 2:
            fams[broad] = sorted(kids)
    if only:
        fams = {k: v for k, v in fams.items() if k in set(only)}
    return fams


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("families", nargs="*", help="broad codes to test (default: all)")
    ap.add_argument("--margin", type=float, default=None, help="override resolver margin")
    args = ap.parse_args()

    ref_codes = set(_bulk_centroids()[0].columns.astype(str))
    fams = broad_families(ref_codes, only=args.families or None)

    total_hard = total_hit = 0
    all_misses = []
    for broad, kids in fams.items():
        hits = []
        misses = []
        for child in kids:
            try:
                df = medoid_df(child)
            except Exception as e:  # noqa: BLE001
                misses.append((child, f"<load error: {str(e)[:40]}>", None, None))
                continue
            sym = _build_sample_tpm_by_symbol(df)
            cc = centroid_correlations(sym)
            rows = rank_cancer_type_candidates(df, top_k=1)
            code = rows[0].get("code") if rows else None  # the BROAD call
            ws = rows[0].get("winning_subtype") if rows else None
            if args.margin is not None:
                ws = resolve_fine_subtype(code or broad, cc, current_subtype=ws, margin=args.margin)
            cen_rank = (list(cc.index).index(child) + 1) if child in cc.index else None
            if ws == child:
                hits.append(child)
            else:
                misses.append((child, ws, cen_rank, code))
        hard_kids = [k for k in kids if k not in KNOWN_DEGENERATE]
        hard_hit = sum(1 for k in hits if k not in KNOWN_DEGENERATE)
        total_hard += len(hard_kids)
        total_hit += hard_hit
        flag = "" if hard_hit == len(hard_kids) else "  <-- check"
        print(f"{broad:8s} {hard_hit}/{len(hard_kids)} self-resolve{flag}")
        for child, ws, cen_rank, code in misses:
            # A miss is BROAD-level (out of fine-resolution scope) when the broad call
            # itself landed outside this family; FINE-level when the family is right but
            # the subtype label is not. Only fine-level misses are resolver regressions.
            broad_miss = bool(code) and code not in (broad, *kids)
            if child in KNOWN_DEGENERATE:
                tag = "degenerate-ok"
            elif broad_miss:
                tag = "BROAD-miss"
            else:
                tag = "FINE-miss"
            print(
                f"          {tag:14s} {child:18s} -> {ws}   "
                f"(broad call={code}, centroid self-rank #{cen_rank})"
            )
            if child not in KNOWN_DEGENERATE and not broad_miss:
                all_misses.append((broad, child, ws, cen_rank))

    print("\n" + "=" * 70)
    print(f"TOTAL (excluding known-degenerate): {total_hit}/{total_hard} subtypes self-resolve")
    if all_misses:
        print(f"hard misses: {len(all_misses)}")
        for broad, child, ws, cen_rank in all_misses:
            print(f"   {child} -> {ws} (centroid self-rank #{cen_rank})")
    return 0 if not all_misses else 1


if __name__ == "__main__":
    sys.exit(main())
