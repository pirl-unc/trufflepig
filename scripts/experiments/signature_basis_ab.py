#!/usr/bin/env python3
"""Cancer-signature feature-space comparison on representative samples.

The production score is within-sample percentile over a fixed reference gene
universe. This script compares it with historical cohort-relative bases on
representative-sample type and lineage accuracy:

  cohort_pct(HK)      — percentile of sample/HK-median among cohort HK values
  cohort_pct(cleanTPM)— drop HK: percentile of the raw clean-TPM value among cohort clean-TPM
  cohort_pct(log1p)   — percentile of log1p(clean-TPM); mathematically identical to clean-TPM
  within_sample_pct   — rank of the gene within the sample
  combined            — cohort_pct(HK) * within_sample_pct  (two weak filters, per the user's idea)
  combined_raw        — cohort_pct(cleanTPM) * within_sample_pct
  combined_log        — cohort_pct(log1p(cleanTPM)) * within_sample_pct

Score per type = mean over its panel; argmax type. Reports EXACT-type and LINEAGE accuracy (lineage
is where cohort-percentile is expected to shine for coarse calls).

Run:  python3 scripts/signature_basis_ab.py
"""
import argparse
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from pirlygenes.expression.accessors import available_representative_cohorts, representative_cohort_samples
from pirlygenes.gene_sets_cancer import cancer_lineage_group, housekeeping_gene_ids
from oncoref.normalization import clean_tpm
from trufflepig.reference import pan_cancer_expression
from trufflepig.plot_embedding import _get_cancer_type_signature_panels
from trufflepig.expression_decomposition import _group_to_mode


def _primary(code):
    return str(code).split("_")[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    panels = _get_cancer_type_signature_panels(n_signature_genes=20)
    types = sorted(panels)                                  # the 33 panel types
    pan = pan_cancer_expression(technical_rna_normalize=True).drop_duplicates("Symbol").set_index("Symbol")
    tcols = [c for c in pan.columns if c.endswith("_TPM")]
    raw_ref = pan[tcols].apply(pd.to_numeric, errors="coerce").fillna(0.0)   # gene x cohort, clean-TPM
    id2sym = dict(zip(pan_cancer_expression(technical_rna_normalize=True)["Ensembl_Gene_ID"],
                      pan_cancer_expression(technical_rna_normalize=True)["Symbol"]))
    hk_syms = [id2sym[g] for g in housekeeping_gene_ids() if g in id2sym]
    # reference HK-median per cohort → HK-normalized reference
    hk_in_ref = [s for s in hk_syms if s in raw_ref.index]
    cohort_hk = raw_ref.loc[hk_in_ref].median(axis=0).replace(0, np.nan)
    hk_ref = raw_ref.div(cohort_hk, axis=1)

    # rep samples (primary types present in the panels)
    samples = []
    for c in sorted(available_representative_cohorts()):
        if _primary(c) not in types:
            continue
        d = representative_cohort_samples(c).drop_duplicates("Ensembl_Gene_ID")
        cols = [x for x in d.columns if x not in ("Ensembl_Gene_ID", "Symbol")]
        gt = pd.DataFrame({"Ensembl_Gene_ID": d["Ensembl_Gene_ID"].values, "Symbol": d["Symbol"].values})
        cl = clean_tpm(d.set_index("Ensembl_Gene_ID")[cols].astype(float), gene_table=gt.set_index(d.index))
        cl.index = d["Symbol"].values
        cl = cl[~cl.index.duplicated(keep="first")]
        for rep in cl.columns:
            samples.append((_primary(c), cl[rep].clip(lower=0)))

    bases = [
        "cohort_hk",
        "cohort_raw",
        "cohort_log",
        "within_pct",
        "combined",
        "combined_raw",
        "combined_log",
        "beta_25",
        "beta_50",
        "beta_75",
    ]
    panel_genes = sorted(
        {
            gene
            for panel in panels.values()
            for gene in panel
            if gene in raw_ref.index
        }
    )
    panel_gene_index = {gene: idx for idx, gene in enumerate(panel_genes)}
    panel_gene_indices = {
        code: np.asarray(
            [panel_gene_index[gene] for gene in genes if gene in panel_gene_index],
            dtype=int,
        )
        for code, genes in panels.items()
    }
    raw_panel_ref = raw_ref.loc[panel_genes].to_numpy(dtype=float)
    hk_panel_ref = hk_ref.loc[panel_genes].to_numpy(dtype=float)

    def _panel_winner(values):
        scores = {
            code: float(np.mean(values[idx])) if len(idx) else 0.0
            for code, idx in panel_gene_indices.items()
        }
        return max(scores, key=scores.get)

    def score_all_bases(sample):
        s = sample
        positive_hk = [s[g] for g in hk_syms if g in s.index and s[g] > 0]
        hk_med = float(np.median(positive_hk)) if positive_hk else 1.0
        wp = s.rank(pct=True)
        sample_values = s.reindex(panel_genes).fillna(0.0).to_numpy(dtype=float)
        within = wp.reindex(panel_genes).fillna(0.5).to_numpy(dtype=float)
        cp_raw = np.mean(raw_panel_ref < sample_values[:, None], axis=1)
        cp_hk = np.mean(
            hk_panel_ref < (sample_values / max(hk_med, 1e-12))[:, None],
            axis=1,
        )
        feature_values = {
            "cohort_hk": cp_hk,
            "cohort_raw": cp_raw,
            # log1p is strictly monotone, so its percentile ranks are exactly raw.
            "cohort_log": cp_raw,
            "within_pct": within,
            "combined": cp_hk * within,
            "combined_raw": cp_raw * within,
            "combined_log": cp_raw * within,
            "beta_25": cp_hk * (0.75 + 0.25 * within),
            "beta_50": cp_hk * (0.50 + 0.50 * within),
            "beta_75": cp_hk * (0.25 + 0.75 * within),
        }
        return {
            basis: _panel_winner(feature_values[basis])
            for basis in bases
        }

    print(f"{len(samples)} samples / {len(types)} panel types", file=sys.stderr)
    # pan-cancer-mean background for the dilution stress
    bg = raw_ref.mean(axis=1)

    dilutions = (0.0, 0.3, 0.5, 0.7)
    counts = {
        basis: {dilution: [0, 0] for dilution in dilutions}
        for basis in bases
    }
    for truth, sample in samples:
        for dilution in dilutions:
            if dilution > 0:
                mix = (
                    sample * (1 - dilution)
                    + bg.reindex(sample.index).fillna(0.0) * dilution
                )
                sample_use = (mix / mix.sum() * 1e6) if mix.sum() > 0 else sample
            else:
                sample_use = sample
            predictions = score_all_bases(sample_use)
            truth_mode = _group_to_mode(cancer_lineage_group(truth) or "")
            for basis, prediction in predictions.items():
                counts[basis][dilution][0] += int(prediction == truth)
                counts[basis][dilution][1] += int(
                    _group_to_mode(cancer_lineage_group(prediction) or "")
                    == truth_mode
                )

    print(f"\n=== signature_score sample-basis A/B (n={len(samples)}) — exact-type acc ===")
    print(f"{'basis':14s} | {'clean':>6s} | {'p=0.3':>6s} | {'p=0.5':>6s} | {'p=0.7':>6s} | {'lineage(clean)':>14s}")
    for b in bases:
        ex0, lin0 = (value / len(samples) for value in counts[b][0.0])
        ex3 = counts[b][0.3][0] / len(samples)
        ex5 = counts[b][0.5][0] / len(samples)
        ex7 = counts[b][0.7][0] / len(samples)
        print(f"{b:14s} | {ex0:>6.3f} | {ex3:>6.3f} | {ex5:>6.3f} | {ex7:>6.3f} | {lin0:>14.3f}")


if __name__ == "__main__":
    main()
