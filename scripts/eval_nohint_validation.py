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
# ALL locally generated real-sample reports — curated truth lineage (from provenance)
REPORTS = [
    ("alvin-sarcoma", f"{D}/alvin/RNA/2025-10-31_salmon/quant.gene_tpm.csv", "mesenchymal"),
    ("hcc1395-kallisto", f"{D}/hcc1395/rnaseq/kallisto_expression/gene_abundance.tsv", "solid"),
    ("hcc1395-stringtie", f"{D}/hcc1395/rnaseq/stringtie_expression/stringtie_gene_expression.tsv", "solid"),
    ("pfo002-colon", f"{D}/pathfinder/pfo002/WashU/mcdb032-BG002179-2022-05-colon/mcdb-workflow_results/gene_abundance.tsv", "solid"),
    ("pfo004-osteosarc", f"{D}/pathfinder/pfo004/analysis/gene-expression.csv", "mesenchymal"),
    ("pfo004-osteo-salmon", f"{D}/pathfinder/pfo004/analysis/transcripts_quant/quant.gene_tpm.csv", "mesenchymal"),
    ("pfo017-bladder", f"{D}/pathfinder/pfo017/salmon.merged.gene_tpm.tsv", "solid"),
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


def classify_without_hint(df):
    """Run the full no-hint cancer-type pipeline → ``(bulk_classifier_call, final_call, purity_result)``.

    Mirrors ``main._analyze_body`` with no cancer-type hint: the bulk classifier (``analyze_sample``)
    → the cancer-type evidence selector → the deconvolved local-reference lineage veto → the purity
    reroute (so the returned purity is consistent with the final call).
    """
    from trufflepig.main import (
        _reroute_decomposition_to_call,
        _veto_local_reference_lineage_flip,
    )
    analysis = analyze_sample(df)                                       # no cancer_type → auto-detect
    scope = select_report_scope_from_evidence(df, analysis)
    selected = scope.get("selected") or {}
    bulk_classifier_call = analysis.get("cancer_type")
    evidence_call = (selected.get("cancer_type") or scope.get("top_reference_cancer_type")
                     or bulk_classifier_call)
    final_call = (_veto_local_reference_lineage_flip(analysis, df, evidence_call,
                                                     bulk_classifier_call, selected)
                  or bulk_classifier_call)
    _reroute_decomposition_to_call(analysis, df, final_call)            # purity consistent with the final call
    return bulk_classifier_call, final_call, (analysis.get("purity") or {})


def result_row(name, truth_lineage, bulk_classifier_call, final_call, purity):
    """Assemble one result row from a classification. Pure dict access — never raises, so it stays
    OUTSIDE the caller's try (only ``classify_without_hint`` is fallible)."""
    components = purity.get("components", {})
    decomposition = components.get("decomposition") or {}
    return {
        "sample": name,
        "truth_lineage": truth_lineage,
        "bulk_classifier_call": bulk_classifier_call,
        "final_call": final_call,
        "final_lineage": _lineage(final_call),
        "overall_purity_estimate": purity.get("overall_estimate"),
        "estimate_method_purity": components.get("estimate_purity"),
        "estimate_gated_for_lineage": components.get("estimate_gated_for_lineage"),
        "decomposition_residual_fraction": decomposition.get("residual_fraction"),
        "aneuploidy_purity": decomposition.get("aneuploidy_purity"),
        "purity_reconciliation_flag": bool(purity.get("purity_consistency")),
    }


# Per-sample failures tolerated while sweeping a heterogeneous corpus (anything else propagates so
# genuine bugs surface): unreadable / odd-format inputs, and types without a reference column.
EXPECTED_SAMPLE_ERRORS = (FileNotFoundError, ValueError, KeyError)

_LEGEND = (
    "Columns — bulk_classifier_call: analyze_sample's ranking winner; final_call: the call after the "
    "evidence selector + lineage veto; lineage_ok: final lineage == truth lineage; overall_purity: "
    "headline purity; estimate_method_purity: ESTIMATE stroma/immune purity (None when gated); "
    "estimate_gated: ESTIMATE disabled for a heme/sarcoma lineage; residual_fraction: decomposition "
    "tumor fraction; aneuploidy_purity: aneuploidy-calibrated purity; reconciliation: the purity "
    "consistency flag fired."
)


def _format_value(value, width):
    return ("%*.2f" % (width, value)) if isinstance(value, (int, float)) else f"{str(value):>{width}s}"


def _classify_corpus_item(group, item):
    """Load + classify one corpus item → a result row (or an error row for an expected failure)."""
    if group == "LOCAL REPORTS":
        name, path, truth_lineage = item
        load = lambda: _load_report(path)
    else:
        name, truth_lineage, load = item, _lineage(item), lambda: _medoid_df(item)
    try:
        result = classify_without_hint(load())                         # the only fallible step
    except EXPECTED_SAMPLE_ERRORS as exc:
        return {"sample": name, "truth_lineage": truth_lineage, "error": str(exc)[:70]}
    return result_row(name, truth_lineage, *result)


def main():
    print(_LEGEND)
    for group, items in (("LOCAL REPORTS", REPORTS), ("MEDOIDS", MEDOIDS)):
        rows = [_classify_corpus_item(group, item) for item in items]
        print(f"\n=== {group} (no hint) ===")
        print(f"{'sample':18s} {'bulk_classifier_call':21s} {'final_call':14s} {'lineage_ok':10s} | "
              f"{'overall_purity':>14s} {'estimate_method_purity':>22s} {'estimate_gated':>14s} "
              f"{'residual_fraction':>17s} {'aneuploidy_purity':>17s} {'reconciliation':>14s}")
        correct = total = 0
        for row in rows:
            if "error" in row:
                print(f"{row['sample']:18s} {str(row['truth_lineage']):21s} ERROR: {row['error']}")
                continue
            total += 1
            lineage_ok = row["final_lineage"] == row["truth_lineage"]
            correct += lineage_ok
            print(f"{row['sample']:18s} {str(row['bulk_classifier_call'])[:21]:21s} "
                  f"{str(row['final_call'])[:14]:14s} {('yes' if lineage_ok else 'NO'):10s} | "
                  f"{_format_value(row['overall_purity_estimate'], 14)} "
                  f"{_format_value(row['estimate_method_purity'], 22)} "
                  f"{str(row['estimate_gated_for_lineage']):>14s} "
                  f"{_format_value(row['decomposition_residual_fraction'], 17)} "
                  f"{_format_value(row['aneuploidy_purity'], 17)} "
                  f"{('fired' if row['purity_reconciliation_flag'] else '-'):>14s}")
        print(f"  {group} no-hint LINEAGE correct: {correct}/{total}")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
