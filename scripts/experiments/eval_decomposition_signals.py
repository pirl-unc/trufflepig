#!/usr/bin/env python3
"""Evaluate how to best USE the decomposition + purity signals, on real local samples.

For each curated real sample (truth from provenance), recompute the new signals and compare the
ways we could pick lineage/mode and purity:
  - compartment_call routing (the classifier's mode)        [no-hint]
  - lineage_fit argmax over all 4 modes (goodness-of-fit)   [no-hint]
  - estimate_tumor_purity: overall_estimate, ESTIMATE, residual_fraction, aneuploidy_purity

Run:  python3 scripts/eval_decomposition_signals.py
(uses ~/trufflepig-local-reports inputs via their original paths)
"""
import sys, warnings
warnings.filterwarnings("ignore")

from trufflepig.load_expression import load_expression_data
from trufflepig.tumor_purity import _build_sample_tpm_by_symbol, estimate_tumor_purity
from trufflepig.expression_decomposition import decompose_expression

MODES = ["solid", "mesenchymal", "heme", "embryonal"]

# (name, input_path, truth_lineage_mode, truth_note). Truth from sample provenance.
D = "/Users/iskander/data"
CORPUS = [
    ("alvin-sarcoma",      f"{D}/alvin/RNA/2025-10-31_salmon/quant.gene_tpm.csv",                                   "mesenchymal", "spindle-cell sarcoma (EGFR-KDD)"),
    ("hcc1395-kallisto",   f"{D}/hcc1395/rnaseq/kallisto_expression/gene_abundance.tsv",                             "solid",       "breast ca cell line"),
    ("hcc1395-stringtie",  f"{D}/hcc1395/rnaseq/stringtie_expression/stringtie_gene_expression.tsv",                "solid",       "breast ca cell line"),
    ("pfo002-washu-kal",   f"{D}/pathfinder/pfo002/WashU/mcdb032-BG002179-2022-05-colon/mcdb-workflow_results/gene_abundance.tsv", "solid", "colon"),
    ("pfo004-osteosarc",   f"{D}/pathfinder/pfo004/analysis/gene-expression.csv",                                    "mesenchymal", "osteosarcoma"),
    ("pfo004-osteo-salmon",f"{D}/pathfinder/pfo004/analysis/transcripts_quant/quant.gene_tpm.csv",                  "mesenchymal", "osteosarcoma"),
    ("pfo017-bladder",     f"{D}/pathfinder/pfo017/salmon.merged.gene_tpm.tsv",                                      "solid",       "bladder"),
    ("pfo019-nasal-kal",   f"{D}/pathfinder/pfo019/BostonGene-BG011335-2024-03-20-nasal/Processed/final_results/final_results/rnaseq/kallisto_expression/gene_abundance.tsv", "solid", "sinonasal"),
    ("tempus-nutm1-26",    f"{D}/tempus-unc-nutm1/data_backfill/Data/Group_Level_Molecular/normalized_rna.csv",     "solid",       "NUT carcinoma"),
]


def load(path):
    last = None
    for agg in (False, True):
        try:
            return load_expression_data(path, aggregate_gene_expression=agg,
                                        save_aggregated_gene_expression=False, verbose=False, progress=False)
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


def evaluate():
    rows = []
    for name, path, truth, note in CORPUS:
        try:
            df = load(path)
            sym = _build_sample_tpm_by_symbol(df)
            r = decompose_expression(sym, cancer=None, run_all=True)         # no-hint: signals pick lineage
            fits = {m: r["modes"].get(m, {}).get("lineage_fit") for m in MODES}
            fit_argmax = max(fits, key=lambda m: (fits[m] if fits[m] is not None else -1e9))
            classifier_mode = r["selected_mode"]
            pur = estimate_tumor_purity(df)                                  # all other signals, one call
            comp = pur.get("components", {})
            dc = comp.get("decomposition") or {}
            rows.append({
                "name": name, "truth": truth, "note": note,
                "classifier_mode": classifier_mode, "fit_argmax": fit_argmax,
                "fits": fits,
                "auto_code": pur.get("cancer_type"),
                "overall": pur.get("overall_estimate"),
                "estimate": comp.get("estimate_purity"),
                "estimate_gated": comp.get("estimate_gated_for_lineage"),
                "residual_fraction": dc.get("residual_fraction"),
                "aneuploidy_purity": dc.get("aneuploidy_purity"),
            })
        except Exception as e:  # noqa: BLE001
            rows.append({"name": name, "truth": truth, "error": str(e)[:80]})
    return rows


def main():
    rows = evaluate()
    print(f"\n{'sample':22s} {'truth':12s} {'classifier':11s} {'fit_argmax':11s} {'auto_code':10s} | "
          f"{'overall':>7s} {'ESTIM':>6s} {'gated':>5s} {'resid_f':>7s} {'aneu_p':>6s}")
    print("-" * 120)
    clf_ok = fit_ok = n = 0
    for x in rows:
        if "error" in x:
            print(f"{x['name']:22s} {x['truth']:12s} ERROR: {x['error']}")
            continue
        n += 1
        clf_ok += (x["classifier_mode"] == x["truth"])
        fit_ok += (x["fit_argmax"] == x["truth"])
        def f(v, w=7): return ("%*.2f" % (w, v)) if isinstance(v, (int, float)) else f"{str(v):>{w}s}"
        print(f"{x['name']:22s} {x['truth']:12s} {x['classifier_mode']:11s} {x['fit_argmax']:11s} "
              f"{str(x['auto_code']):10s} | {f(x['overall'])} {f(x['estimate'],6)} {str(x['estimate_gated']):>5s} "
              f"{f(x['residual_fraction'])} {f(x['aneuploidy_purity'],6)}")
    print("-" * 120)
    print(f"lineage correct  — compartment_call: {clf_ok}/{n}   |   lineage_fit argmax: {fit_ok}/{n}")
    print("\nper-sample lineage_fit by mode (solid / mesench / heme / embryonal):")
    for x in rows:
        if "error" not in x:
            fv = "  ".join(f"{m[:5]}={(x['fits'][m] if x['fits'][m] is not None else float('nan')):5.1f}" for m in MODES)
            mark = "ok" if x["fit_argmax"] == x["truth"] else "MISS"
            print(f"  {x['name']:22s} [{x['truth']:11s} {mark:4s}] {fv}")
    return rows


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
