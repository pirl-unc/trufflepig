#!/usr/bin/env python3
"""Re-measure the hard cases through the FULL pipeline (analyze_sample + the cancer-type EVIDENCE
selector), not the expression-decomposition axis in isolation.

The medoid eval measured estimate_tumor_purity's lighter auto-detect; the real call goes through
analyze_sample (ranking + cancer_call_rescue) and select_report_scope_from_evidence (literature /
lineage-panel / fusion / cancer-type evidence). This checks whether the hard cases the
decomposition-alone eval missed are resolved by the full evidence pipeline.

Run:  python3 scripts/eval_full_pipeline_hardcases.py
"""
import sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pirlygenes.expression.accessors import representative_cohort_samples
from pirlygenes.gene_sets_cancer import cancer_lineage_group
from oncoref.normalization import clean_tpm
from trufflepig.expression_decomposition import _group_to_mode
from trufflepig.tumor_purity import analyze_sample
from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

# hard cases the decomposition-alone eval missed + controls (truth = lineage mode)
CASES = ["ATRT", "HEPB", "MESO", "DLBC",
         "SARC_DSRCT", "SARC_EPITH", "SARC_GIST", "SARC_MPNST", "SARC_OS", "SARC_SYN",
         "COAD", "BRCA", "LAML", "SKCM"]   # last 4 controls


def _medoid_df(t):
    d = representative_cohort_samples(t).drop_duplicates("Ensembl_Gene_ID")
    cols = [c for c in d.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    gt = pd.DataFrame({"Ensembl_Gene_ID": d["Ensembl_Gene_ID"].values, "Symbol": d["Symbol"].values})
    cl = clean_tpm(d.set_index("Ensembl_Gene_ID")[cols].astype(float), gene_table=gt.set_index(d.index))
    return pd.DataFrame({"ensembl_gene_id": d["Ensembl_Gene_ID"].values,
                         "gene_symbol": d["Symbol"].values, "TPM": cl.mean(axis=1).values})


def _lineage(code):
    grp = cancer_lineage_group(code) if code else None
    return _group_to_mode(grp) if grp else None


def main():
    rows = []
    for t in CASES:
        truth = _lineage(t)
        try:
            df = _medoid_df(t)
            a = analyze_sample(df)
            analyze_call = a.get("cancer_type")
            scope = select_report_scope_from_evidence(df, a)
            sel = scope.get("selected") or {}
            evid_call = sel.get("cancer_type") or scope.get("top_reference_cancer_type") or analyze_call
            rows.append({"type": t, "truth": truth, "analyze_call": analyze_call,
                         "analyze_lineage": _lineage(analyze_call),
                         "evid_call": evid_call, "evid_lineage": _lineage(evid_call),
                         "selected_by": sel.get("selected_by")})
        except Exception as e:  # noqa: BLE001
            rows.append({"type": t, "truth": truth, "error": str(e)[:70]})

    print(f"\n{'type':12s} {'truth':11s} | {'analyze_call':14s} {'a_lin':11s} | {'evidence_call':16s} {'e_lin':11s} {'ok':3s}")
    print("-" * 100)
    ok = 0; n = 0
    for r in rows:
        if "error" in r:
            print(f"{r['type']:12s} {str(r['truth']):11s} ERROR: {r['error']}")
            continue
        n += 1
        correct = r["evid_lineage"] == r["truth"]
        ok += correct
        print(f"{r['type']:12s} {str(r['truth']):11s} | {str(r['analyze_call'])[:14]:14s} {str(r['analyze_lineage']):11s} | "
              f"{str(r['evid_call'])[:16]:16s} {str(r['evid_lineage']):11s} {'Y' if correct else 'n':3s}")
    print("-" * 100)
    hard = [r for r in rows if "error" not in r and r["type"] not in ("COAD", "BRCA", "LAML", "SKCM")]
    print(f"\nFull-pipeline LINEAGE correct (evidence call): {ok}/{n}")
    print(f"  on the HARD cases alone: {sum(r['evid_lineage']==r['truth'] for r in hard)}/{len(hard)}")
    print("  (compare: decomposition-alone eval got these hard cases mostly wrong)")
    return rows


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
