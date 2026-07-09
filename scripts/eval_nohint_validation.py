#!/usr/bin/env python3
"""Validate cancer-type inference WITHOUT hints on local truth samples and medoids.

Runs the full ``analyze_sample`` + cancer-type evidence selector with no
``cancer_type`` hint, including rare-marker RNA hypotheses, then scores both
entity compatibility and lineage compatibility against curated truth.

Run:  python3 scripts/eval_nohint_validation.py
"""
import argparse
import sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pirlygenes.expression.accessors import representative_cohort_samples
from pirlygenes.gene_sets_cancer import cancer_lineage_group
from oncoref.normalization import clean_tpm
from trufflepig.cancer_ontology import registry_parent_code
from trufflepig.expression_decomposition import _group_to_mode
from trufflepig.tumor_purity import analyze_sample
from trufflepig.cancer_type_evidence import select_report_scope_from_evidence
from trufflepig.cancer_type_signal_matrix import (
    build_cancer_type_signal_matrix,
    build_signal_sample_summary,
    build_signal_matrix_summary_markdown,
)
from trufflepig.load_expression import load_expression_data
from trufflepig.rare_inference import infer_rare_cancer_marker_hypotheses_from_rna

D = "/Users/iskander/data"
# Local truth samples. ``expected_codes`` may name an accepted parent, e.g. CRC
# accepts COAD/READ/CRC-compatible calls.
REPORTS = [
    {"name": "alvin-sarcoma", "path": f"{D}/alvin/RNA/2025-10-31_salmon/quant.gene_tpm.csv", "expected_codes": ["SARC"]},
    {"name": "asy-salmon-old", "path": f"{D}/asy/salmon-output-asy/quant.gene_tpm.csv", "gene_id_col": "gene_id", "gene_name_col": "gene", "expected_codes": ["CRC", "COAD", "READ"]},
    {"name": "asy-caris-salmon", "path": f"{D}/asy/caris/salmon-output-caris/quant.gene_tpm.csv", "gene_id_col": "gene_id", "gene_name_col": "gene", "expected_codes": ["CRC", "COAD", "READ"]},
    {"name": "hcc1395-kallisto", "path": f"{D}/hcc1395/rnaseq/kallisto_expression/gene_abundance.tsv", "expected_codes": ["BRCA"]},
    {"name": "hcc1395-stringtie", "path": f"{D}/hcc1395/rnaseq/stringtie_expression/stringtie_gene_expression.tsv", "expected_codes": ["BRCA"]},
    {"name": "pfo002-personalis", "path": f"{D}/pathfinder/pfo002/Personalis-PSNLDx20240402135/Processed/RNA_Pipeline/Expression_Reports/tsv/RNA_PSNLDx20240402135_tumor_rna_gene_expression_report.tsv", "expected_codes": ["CRC", "COAD", "READ"]},
    {"name": "pfo002-kallisto", "path": f"{D}/pathfinder/pfo002/WashU/mcdb032-BG002179-2022-05-colon/mcdb-workflow_results/gene_abundance.tsv", "expected_codes": ["CRC", "COAD", "READ"]},
    {"name": "pfo002-stringtie", "path": f"{D}/pathfinder/pfo002/WashU/mcdb032-BG002179-2022-05-colon/mcdb-workflow_results/rna_stringtie_gene_expression.tsv", "expected_codes": ["CRC", "COAD", "READ"]},
    {"name": "pfo004-osteosarc", "path": f"{D}/pathfinder/pfo004/analysis/gene-expression.csv", "expected_codes": ["SARC_OS", "SARC"]},
    {"name": "pfo004-osteo-salmon", "path": f"{D}/pathfinder/pfo004/analysis/transcripts_quant/quant.gene_tpm.csv", "expected_codes": ["SARC_OS", "SARC"]},
    {"name": "pfo017-bladder-2023", "path": f"{D}/pathfinder/pfo017/salmon.merged.gene_tpm.tsv", "sample_id_value": "PFO017-bladder-2023", "gene_id_col": "gene_id", "gene_name_col": "gene_name", "expected_codes": ["BLCA"]},
    {"name": "pfo017-bladder-2025", "path": f"{D}/pathfinder/pfo017/salmon.merged.gene_tpm.tsv", "sample_id_value": "PFO017-bladder-2025", "gene_id_col": "gene_id", "gene_name_col": "gene_name", "expected_codes": ["BLCA"]},
    {"name": "pfo017-liver-2023", "path": f"{D}/pathfinder/pfo017/salmon.merged.gene_tpm.tsv", "sample_id_value": "PFO017-liver-2023", "gene_id_col": "gene_id", "gene_name_col": "gene_name", "expected_codes": ["BLCA"]},
    {"name": "pfo019-kallisto", "path": f"{D}/pathfinder/pfo019/BostonGene-BG011335-2024-03-20-nasal/Processed/final_results/final_results/rnaseq/kallisto_expression/gene_abundance.tsv", "expected_codes": ["NUTM"]},
    {"name": "pfo019-stringtie", "path": f"{D}/pathfinder/pfo019/BostonGene-BG011335-2024-03-20-nasal/Processed/final_results/final_results/rnaseq/stringtie_expression/stringtie_gene_expression.tsv", "expected_codes": ["NUTM"]},
    {"name": "tempus-nutm1-26", "path": f"{D}/tempus-unc-nutm1/data_backfill/Data/Group_Level_Molecular/normalized_rna.csv", "sample_id_col": "partner_sample_id", "sample_id_value": "TL-21-KS8T3EYH", "gene_id_col": "ensembl_gene", "expected_codes": ["NUTM"]},
]
# medoid hard cases + controls (truth = lineage)
MEDOIDS = ["SARC_DSRCT", "SARC_GIST", "SARC_OS", "ATRT", "HEPB", "MESO", "DLBC", "NUTM", "SCLC",
           "COAD", "BRCA", "LAML", "SKCM", "LUAD"]


def _load_report(spec):
    last = None
    kwargs = {
        "sample_id_col": spec.get("sample_id_col"),
        "sample_id_value": spec.get("sample_id_value"),
        "gene_id_col": spec.get("gene_id_col"),
        "gene_name_col": spec.get("gene_name_col"),
    }
    for agg in (False, True):
        try:
            return load_expression_data(spec["path"], aggregate_gene_expression=agg,
                                        save_aggregated_gene_expression=False, verbose=False, progress=False,
                                        **kwargs)
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


def _ancestor_chain(code):
    code = str(code or "").strip()
    out = []
    while code and code not in out:
        out.append(code)
        code = registry_parent_code(code)
    return out


def _entity_compatible(call, expected_codes):
    call_chain = set(_ancestor_chain(call))
    for expected in expected_codes:
        expected_chain = set(_ancestor_chain(expected))
        if call_chain & expected_chain:
            return True
    return False


def classify_without_hint_with_analysis(df):
    """Run the full no-hint cancer-type pipeline and keep the analysis object.

    Mirrors ``main._analyze_body`` with no cancer-type hint: the bulk classifier (``analyze_sample``)
    → the cancer-type evidence selector → the deconvolved local-reference lineage veto → the purity
    reroute (so the returned purity is consistent with the final call).
    """
    from trufflepig.main import (
        _reroute_decomposition_to_call,
        _veto_local_reference_lineage_flip,
    )
    analysis = analyze_sample(df)                                       # no cancer_type → auto-detect
    rare_marker_hypotheses = infer_rare_cancer_marker_hypotheses_from_rna(df, analysis)
    analysis["rare_marker_hypotheses"] = rare_marker_hypotheses
    scope = select_report_scope_from_evidence(
        df,
        analysis,
        rare_marker_hypotheses=rare_marker_hypotheses,
    )
    analysis["cancer_type_evidence"] = scope
    selected = scope.get("selected") or {}
    bulk_classifier_call = analysis.get("cancer_type")
    evidence_call = (selected.get("cancer_type") or scope.get("top_reference_cancer_type")
                     or bulk_classifier_call)
    final_call = (_veto_local_reference_lineage_flip(analysis, df, evidence_call,
                                                     bulk_classifier_call, selected)
                  or bulk_classifier_call)
    _reroute_decomposition_to_call(analysis, df, final_call)            # purity consistent with the final call
    analysis["cancer_type"] = final_call
    analysis["report_scope_cancer_type"] = final_call
    analysis["reference_cancer_type"] = (
        selected.get("reference_cancer_type") or final_call
    )
    return bulk_classifier_call, final_call, (analysis.get("purity") or {}), analysis


def classify_without_hint(df):
    """Run the full no-hint pipeline → ``(bulk_classifier_call, final_call, purity_result)``."""
    bulk_classifier_call, final_call, purity, _analysis = classify_without_hint_with_analysis(df)
    return bulk_classifier_call, final_call, purity


def full_granularity_analysis(df):
    """Return ``(bulk_classifier_call, finest_final_call, analysis)``."""
    from trufflepig.degenerate_subtype import resolve_degenerate_subtype
    from trufflepig.reporting import candidate_winning_subtype_for_analysis
    from trufflepig.tumor_purity import _build_sample_tpm_by_symbol

    bulk_classifier_call, entity, _purity, analysis = classify_without_hint_with_analysis(df)
    analysis["cancer_type"] = entity
    base = candidate_winning_subtype_for_analysis(analysis) or entity
    resolution = resolve_degenerate_subtype(
        base,
        tumor_tpm_by_symbol=_build_sample_tpm_by_symbol(df),
    )
    final_call = resolution.get("final_subtype") or base
    analysis["cancer_type"] = final_call
    analysis["full_granularity_call"] = final_call
    return bulk_classifier_call, final_call, analysis


def full_granularity_call(df):
    """The FINEST call the system would actually report → ``(bulk_classifier_call, final_call)``.

    Extends ``classify_without_hint`` past the entity level through the two refinement layers the
    real report applies: the molecular subtype (``candidate_winning_subtype_for_analysis`` — e.g.
    BRCA → BRCA_LumA, LUAD → LUAD_KRAS_STK11) and then degenerate-subtype resolution (look-alike
    groups — e.g. ADCC → NUTM). So the call can be coarser than truth (a sibling/parent) or finer
    (a molecular subtype), reflecting exactly what the system commits to.
    """
    bulk_classifier_call, final_call, _analysis = full_granularity_analysis(df)
    return bulk_classifier_call, final_call


def result_row(name, expected_codes, bulk_classifier_call, final_call, purity):
    """Assemble one result row from a classification. Pure dict access — never raises, so it stays
    OUTSIDE the caller's try (only ``classify_without_hint`` is fallible)."""
    truth_lineage = _lineage(expected_codes[0]) if expected_codes else None
    components = purity.get("components", {})
    decomposition = components.get("decomposition") or {}
    return {
        "sample": name,
        "expected_codes": "|".join(expected_codes),
        "entity_ok": _entity_compatible(final_call, expected_codes),
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


def _classify_corpus_item(group, item, signal_frames=None):
    """Load + classify one corpus item → a result row (or an error row for an expected failure)."""
    if group == "LOCAL REPORTS":
        name = item["name"]
        expected_codes = item["expected_codes"]
        load = lambda: _load_report(item)
    else:
        name, expected_codes, load = item, [item], lambda: _medoid_df(item)
    try:
        result = classify_without_hint_with_analysis(load())            # the only fallible step
    except EXPECTED_SAMPLE_ERRORS as exc:
        return {
            "sample": name,
            "expected_codes": "|".join(expected_codes),
            "truth_lineage": _lineage(expected_codes[0]) if expected_codes else None,
            "error": str(exc)[:70],
        }
    bulk_classifier_call, final_call, purity, analysis = result
    if signal_frames is not None:
        matrix = build_cancer_type_signal_matrix(analysis, sample_id=name)
        matrix.insert(1, "validation_group", group)
        matrix.insert(2, "expected_codes", "|".join(expected_codes))
        signal_frames.append(matrix)
    return result_row(name, expected_codes, bulk_classifier_call, final_call, purity)


def main(signal_matrix_out=None):
    print(_LEGEND)
    signal_frames = [] if signal_matrix_out else None
    for group, items in (("LOCAL REPORTS", REPORTS), ("MEDOIDS", MEDOIDS)):
        rows = [_classify_corpus_item(group, item, signal_frames=signal_frames) for item in items]
        print(f"\n=== {group} (no hint) ===")
        print(f"{'sample':22s} {'expected':16s} {'bulk_classifier_call':21s} {'final_call':14s} {'entity_ok':9s} {'lineage_ok':10s} | "
              f"{'overall_purity':>14s} {'estimate_method_purity':>22s} {'estimate_gated':>14s} "
              f"{'residual_fraction':>17s} {'aneuploidy_purity':>17s} {'reconciliation':>14s}")
        entity_correct = lineage_correct = total = 0
        for row in rows:
            if "error" in row:
                print(f"{row['sample']:22s} {str(row['expected_codes'])[:16]:16s} ERROR: {row['error']}")
                continue
            total += 1
            lineage_ok = row["final_lineage"] == row["truth_lineage"]
            entity_ok = bool(row["entity_ok"])
            entity_correct += entity_ok
            lineage_correct += lineage_ok
            print(f"{row['sample']:22s} {str(row['expected_codes'])[:16]:16s} "
                  f"{str(row['bulk_classifier_call'])[:21]:21s} "
                  f"{str(row['final_call'])[:14]:14s} "
                  f"{('yes' if entity_ok else 'NO'):9s} {('yes' if lineage_ok else 'NO'):10s} | "
                  f"{_format_value(row['overall_purity_estimate'], 14)} "
                  f"{_format_value(row['estimate_method_purity'], 22)} "
                  f"{str(row['estimate_gated_for_lineage']):>14s} "
                  f"{_format_value(row['decomposition_residual_fraction'], 17)} "
                  f"{_format_value(row['aneuploidy_purity'], 17)} "
                  f"{('fired' if row['purity_reconciliation_flag'] else '-'):>14s}")
        print(f"  {group} no-hint ENTITY compatible: {entity_correct}/{total}")
        print(f"  {group} no-hint LINEAGE compatible: {lineage_correct}/{total}")
    if signal_matrix_out and signal_frames:
        matrix = pd.concat(signal_frames, ignore_index=True)
        out_path = str(signal_matrix_out)
        matrix.to_csv(out_path, sep="\t", index=False)
        sample_summary_path = out_path.rsplit(".", 1)[0] + "-sample-summary.tsv"
        build_signal_sample_summary(matrix).to_csv(
            sample_summary_path,
            sep="\t",
            index=False,
        )
        summary_path = out_path.rsplit(".", 1)[0] + ".md"
        with open(summary_path, "w") as fh:
            fh.write(
                build_signal_matrix_summary_markdown(
                    matrix,
                    title="No-Hint Validation Cancer-Type Signal Matrix Summary",
                )
            )
        print(f"\n[signal-matrix] wrote {out_path}")
        print(f"[signal-summary] wrote {sample_summary_path}")
        print(f"[signal-matrix] wrote {summary_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signal-matrix-out",
        default=None,
        help="Optional TSV path for a concatenated cancer-type signal matrix.",
    )
    args = parser.parse_args()
    sys.exit(0 if main(signal_matrix_out=args.signal_matrix_out) else 1)
