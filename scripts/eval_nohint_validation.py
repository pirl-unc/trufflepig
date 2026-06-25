#!/usr/bin/env python3
"""Validate cancer-type inference WITHOUT hints, on (1) locally generated real reports and (2) medoid
samples — the honest, non-circular measure. Runs the full analyze_sample + cancer-type evidence
selector (no cancer_type passed) and scores the LINEAGE of the final call against curated truth.

Run:  python3 scripts/eval_nohint_validation.py
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
from trufflepig.load_expression import load_expression_data

D = "/Users/iskander/data"
# locally generated reports — curated truth lineage (from provenance)
REPORTS = [
    ("alvin-sarcoma", f"{D}/alvin/RNA/2025-10-31_salmon/quant.gene_tpm.csv", "mesenchymal"),
    ("hcc1395-breast", f"{D}/hcc1395/rnaseq/kallisto_expression/gene_abundance.tsv", "solid"),
    ("pfo002-colon", f"{D}/pathfinder/pfo002/WashU/mcdb032-BG002179-2022-05-colon/mcdb-workflow_results/gene_abundance.tsv", "solid"),
    ("pfo004-osteosarc", f"{D}/pathfinder/pfo004/analysis/gene-expression.csv", "mesenchymal"),
    ("pfo019-sinonasal", f"{D}/pathfinder/pfo019/BostonGene-BG011335-2024-03-20-nasal/Processed/final_results/final_results/rnaseq/kallisto_expression/gene_abundance.tsv", "solid"),
    ("tempus-nutm1", f"{D}/tempus-unc-nutm1/data_backfill/Data/Group_Level_Molecular/normalized_rna.csv", "solid"),
]
# medoid hard cases + controls (truth = lineage)
MEDOIDS = ["SARC_DSRCT", "SARC_GIST", "SARC_OS", "ATRT", "HEPB", "MESO", "DLBC", "NUTM", "SCLC",
           "COAD", "BRCA", "LAML", "SKCM", "LUAD"]


def _load_report(path):
    last = None
    for agg in (False, True):
        try:
            return load_expression_data(path, aggregate_gene_expression=agg,
                                        save_aggregated_gene_expression=False, verbose=False, progress=False)
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


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


def _nohint_call(df):
    a = analyze_sample(df)                                       # no cancer_type → auto
    scope = select_report_scope_from_evidence(df, a)
    sel = scope.get("selected") or {}
    code = sel.get("cancer_type") or scope.get("top_reference_cancer_type") or a.get("cancer_type")
    return a.get("cancer_type"), code


def run(name, df, truth, rows):
    try:
        analyze_call, evid_call = _nohint_call(df)
        rows.append({"name": name, "truth": truth, "analyze": analyze_call,
                     "a_lin": _lineage(analyze_call), "evid": evid_call, "e_lin": _lineage(evid_call)})
    except Exception as e:  # noqa: BLE001
        rows.append({"name": name, "truth": truth, "error": str(e)[:60]})


def main():
    for group, items in (("LOCAL REPORTS", REPORTS), ("MEDOIDS", MEDOIDS)):
        rows = []
        print(f"\n=== {group} (no hint) ===")
        print(f"{'sample':18s} {'truth':11s} | {'analyze':14s} {'a_lin':11s} | {'evidence':14s} {'e_lin':11s} ok")
        for it in items:
            if group == "LOCAL REPORTS":
                name, path, truth = it
                try:
                    run(name, _load_report(path), truth, rows)
                except Exception as e:  # noqa: BLE001
                    rows.append({"name": name, "truth": truth, "error": str(e)[:60]})
            else:
                run(it, _medoid_df(it), _lineage(it), rows)
        ok = n = 0
        for r in rows:
            if "error" in r:
                print(f"{r['name']:18s} {str(r['truth']):11s} ERROR: {r['error']}")
                continue
            n += 1; c = r["e_lin"] == r["truth"]; ok += c
            print(f"{r['name']:18s} {str(r['truth']):11s} | {str(r['analyze'])[:14]:14s} {str(r['a_lin']):11s} | "
                  f"{str(r['evid'])[:14]:14s} {str(r['e_lin']):11s} {'Y' if c else 'n'}")
        print(f"  {group} no-hint LINEAGE correct: {ok}/{n}")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
