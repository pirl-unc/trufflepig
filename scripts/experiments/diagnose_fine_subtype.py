#!/usr/bin/env python3
"""Trace how the FINE cancer subtype gets resolved for one sample (#98 diagnostic).

For a cancer-type medoid (``--type SARC_OS``) or a real sample quant file
(``--sample path/to/quant.sf`` / a kallisto/salmon dir / a TSV with Ensembl+TPM),
dump every signal that determines the fine subtype and exactly which one wins:

  - the ranker's broad call + its curated/signature ``winning_subtype``
  - the whole-profile compartment call (group, margin, confident)
  - the centroid top-K cohorts (whole-transcriptome Spearman vs ~116 medoids)
  - the broad call's CHILDREN with their centroid rho (the data-derived subtype signal)
  - what ``resolve_fine_subtype`` anchors to, and the margin that drove it

This is the general tool for debugging centroid-authoritative subtype resolution —
"the centroid says X but the call is Y, why?". Reusable for any broad call, any sample.

Examples
--------
    python3 scripts/diagnose_fine_subtype.py --type SARC_OS
    python3 scripts/diagnose_fine_subtype.py --type SARC_UPS --type SARC_DDLPS
    python3 scripts/diagnose_fine_subtype.py --sample ~/data/pathfinder/pfo004
    python3 scripts/diagnose_fine_subtype.py --all-subtypes-of SARC   # every SARC_* medoid
"""
import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from trufflepig.cancer_type_centroid import (
    _fine_subtype_candidates,
    centroid_correlations,
    compartment_call,
    resolve_fine_subtype,
)
from trufflepig.cancer_ontology import cancer_type_subtypes_of
from trufflepig.tumor_purity import (
    _build_sample_tpm_by_symbol,
    rank_cancer_type_candidates,
)


def medoid_df(type_code):
    """(df_gene_expr) for a cancer type's representative-cohort clean-TPM mean."""
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


def sample_df(path):
    """Load a real sample exactly as the production CLI does, up to ``analyze_sample``.

    Mirrors ``main.py``: ``load_expression_data`` (gene-aggregated) -> the sample-conform
    chokepoint ``normalize_to_reference_space`` (so the sample shares the reference's
    clean-TPM space). This is the same frame ``analyze_sample`` receives at runtime, so the
    centroid/compartment/ranker signals match the report for a real sample."""
    from trufflepig.clean_tpm import normalize_to_reference_space, resolve_gene_columns
    from trufflepig.load_expression import load_expression_data

    df = load_expression_data(path, aggregate_gene_expression=True, verbose=False)
    label_col, id_col = resolve_gene_columns(df)
    return normalize_to_reference_space(
        df, value_cols=["TPM"], label_col=label_col, id_col=id_col
    )


def diagnose(label, df, topk=12, margin=None):
    sym = _build_sample_tpm_by_symbol(df)
    cc = centroid_correlations(sym)
    comp = compartment_call(sym, _corr=cc)
    rows = rank_cancer_type_candidates(df, top_k=4)
    top = rows[0] if rows else {}
    broad = str(top.get("code") or "")
    winning = top.get("winning_subtype")

    print("=" * 92)
    print(f"{label}")
    print("=" * 92)
    print(
        f"  broad call         : {broad}   (support={top.get('support_score'):.3f}"
        if rows
        else "  broad call         : <none>"
    )
    print(
        f"  headline subtype   : {winning}   <- ranker winning_subtype (post-#98 centroid anchor)"
    )
    print(
        f"  compartment        : {comp['compartment']}  margin={comp['margin']:.3f}  "
        f"confident={comp['confident']}"
    )
    print(f"\n  centroid top-{topk} (whole-profile Spearman):")
    for code, rho in list(cc.items())[:topk]:
        star = "  <- broad call" if code == broad else ""
        print(f"      {code:18s} {float(rho):.3f}{star}")

    # The broad call's children with centroid rho — the data-derived subtype signal.
    kids = _fine_subtype_candidates(broad, cc.index)
    if kids:
        scored = sorted(
            ((float(cc[c]), c) for c in kids if c in cc.index and np.isfinite(cc[c])),
            reverse=True,
        )
        print(f"\n  children of {broad} by centroid rho:")
        for i, (rho, c) in enumerate(scored[:topk]):
            lead = ""
            if i == 0 and len(scored) > 1:
                lead = f"   (leads runner-up by {scored[0][0] - scored[1][0]:+.3f})"
            print(f"      {c:18s} {rho:.3f}{lead}")
        kw = {} if margin is None else {"margin": margin}
        resolved = resolve_fine_subtype(broad, cc, current_subtype=winning, **kw)
        print(f"\n  resolve_fine_subtype -> {resolved}")
    else:
        print(f"\n  {broad} has no reference-present children — fine subtype = curated label")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--type", action="append", default=[], help="cancer-type medoid code")
    ap.add_argument("--sample", action="append", default=[], help="sample quant path")
    ap.add_argument(
        "--all-subtypes-of",
        default=None,
        help="run every reference-present subtype medoid of this broad code",
    )
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument(
        "--margin",
        type=float,
        default=None,
        help="override the fine-subtype lead margin (default: module constant)",
    )
    args = ap.parse_args()

    types = list(args.type)
    if args.all_subtypes_of:
        avail = set(centroid_correlations.__globals__["_bulk_centroids"]()[0].columns.astype(str))
        types += sorted(c for c in cancer_type_subtypes_of(args.all_subtypes_of) if c in avail)

    if not types and not args.sample:
        ap.error("supply at least one --type, --sample, or --all-subtypes-of")

    for t in types:
        try:
            diagnose(f"TYPE {t} (representative-cohort medoid)", medoid_df(t),
                     topk=args.topk, margin=args.margin)
        except Exception as e:  # noqa: BLE001
            print(f"!! {t}: {e}", file=sys.stderr)
    for s in args.sample:
        try:
            diagnose(f"SAMPLE {s}", sample_df(s), topk=args.topk, margin=args.margin)
        except Exception as e:  # noqa: BLE001
            print(f"!! {s}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
