#!/usr/bin/env python3
"""Compare lineage-panel feature spaces on representative expression samples.

The production panel gate historically compares marker/HK ratios between a
sample and its parent cohort.  This harness keeps the curated marker programs
and negative-marker logic fixed while changing only the positive-marker
representation:

* ``hk``: marker divided by each profile's housekeeping median;
* ``clean_tpm``: linear clean TPM;
* ``log1p``: log1p(clean TPM);
* ``within_pct``: within-profile percentile rank;
* ``cohort_pct``: percentile among cancer reference cohorts.

Scores are evaluated as one-vs-rest AUC for each parent program and as top-parent
accuracy on samples whose truth has a curated panel.  The script is deliberately
an evaluation harness, not production threshold tuning.
"""

from __future__ import annotations

import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from oncoref.normalization import clean_tpm
from pirlygenes.expression.accessors import (
    available_representative_cohorts,
    representative_cohort_samples,
)
from pirlygenes.gene_sets_cancer import housekeeping_gene_ids

from trufflepig.analyze.cancer_type_context import registry_ancestor_codes
from trufflepig.cancer_ontology import cancer_codes_entity_compatible
from trufflepig.lineage_panels import LINEAGE_PANELS
from trufflepig.reference import pan_cancer_expression


FEATURE_SPACES = (
    "hk",
    "clean_tpm",
    "log1p",
    "within_pct",
    "cohort_pct",
    "log+cohort",
    "rank+log+cohort",
)


def _auc(scores, labels):
    values = pd.Series(scores, dtype=float)
    truth = np.asarray(labels, dtype=bool)
    n_pos = int(truth.sum())
    n_neg = int((~truth).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = values.rank(method="average").to_numpy()
    return float(
        (ranks[truth].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def _truth_matches_parent(truth, parent):
    return bool(
        truth == parent
        or parent in registry_ancestor_codes(truth)
        or cancer_codes_entity_compatible(truth, parent)
    )


def _reference_state():
    frame = (
        pan_cancer_expression(technical_rna_normalize=True)
        .drop_duplicates("Symbol")
        .set_index("Symbol")
    )
    cohort_columns = [column for column in frame if column.endswith("_TPM")]
    raw = frame[cohort_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    id_to_symbol = dict(
        zip(
            pan_cancer_expression(technical_rna_normalize=True)["Ensembl_Gene_ID"],
            pan_cancer_expression(technical_rna_normalize=True)["Symbol"],
        )
    )
    hk_symbols = [
        id_to_symbol[gene_id]
        for gene_id in housekeeping_gene_ids()
        if gene_id in id_to_symbol
    ]
    hk_in_reference = [gene for gene in hk_symbols if gene in raw.index]
    hk_medians = raw.loc[hk_in_reference].median(axis=0).replace(0.0, np.nan)
    within = raw.rank(axis=0, pct=True, method="average")
    cohort_percentiles = raw.rank(axis=1, pct=True, method="average")
    return raw, hk_medians, within, cohort_percentiles, hk_symbols


def _sample_profiles():
    for truth in sorted(available_representative_cohorts()):
        frame = representative_cohort_samples(truth).drop_duplicates(
            "Ensembl_Gene_ID"
        )
        sample_columns = [
            column
            for column in frame
            if column not in {"Ensembl_Gene_ID", "Symbol"}
        ]
        gene_table = frame[["Ensembl_Gene_ID", "Symbol"]].copy()
        cleaned = clean_tpm(
            frame.set_index("Ensembl_Gene_ID")[sample_columns].astype(float),
            gene_table=gene_table.set_index(frame.index),
        )
        cleaned.index = frame["Symbol"].astype(str).values
        cleaned = cleaned[~cleaned.index.duplicated(keep="first")]
        for sample_name in cleaned:
            yield truth, str(sample_name), cleaned[sample_name].clip(lower=0.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    raw_ref, ref_hk, ref_within, ref_cohort_pct, hk_symbols = _reference_state()
    parents = sorted({panel.parent_cohort for panel in LINEAGE_PANELS})
    panels_by_parent = defaultdict(list)
    for panel in LINEAGE_PANELS:
        panels_by_parent[panel.parent_cohort].append(panel)

    records = []
    for truth, sample_name, sample in _sample_profiles():
        sample_hk_values = [
            float(sample.get(gene, 0.0))
            for gene in hk_symbols
            if float(sample.get(gene, 0.0)) > 0.0
        ]
        sample_hk = (
            float(np.median(sample_hk_values)) if sample_hk_values else 1.0
        )
        sample_within = sample.rank(method="average", pct=True)
        parent_scores = {
            feature: {parent: 0.0 for parent in parents}
            for feature in FEATURE_SPACES
        }

        for panel in LINEAGE_PANELS:
            reference_columns = [
                f"{code}_TPM"
                for code in (panel.reference_cohorts or (panel.parent_cohort,))
                if f"{code}_TPM" in raw_ref
            ]
            if not reference_columns:
                continue

            identity_ok = all(
                any(float(sample.get(symbol, 0.0)) >= 5.0 for symbol in group)
                for group in panel.identity_marker_groups
            )
            if not identity_ok:
                continue
            low_fraction = (
                float(
                    np.mean(
                        [
                            float(sample.get(symbol, 0.0)) <= float(cap)
                            for symbol, cap in panel.low_markers
                        ]
                    )
                )
                if panel.low_markers
                else 1.0
            )

            supports = {
                feature: []
                for feature in FEATURE_SPACES
                if feature not in {"log+cohort", "rank+log+cohort"}
            }
            for gene in panel.high_markers:
                if gene not in raw_ref.index:
                    continue
                observed = float(sample.get(gene, 0.0))
                reference = float(
                    raw_ref.loc[gene, reference_columns].median()
                )
                reference_hk = float(ref_hk[reference_columns].median())
                reference_within = float(
                    ref_within.loc[gene, reference_columns].median()
                )
                reference_cohort_pct = float(
                    ref_cohort_pct.loc[gene, reference_columns].median()
                )
                supports["hk"].append(
                    np.clip(
                        (observed / max(sample_hk, 1e-12))
                        / max(reference / max(reference_hk, 1e-12), 1e-12),
                        0.0,
                        1.0,
                    )
                )
                supports["clean_tpm"].append(
                    np.clip(observed / max(reference, 1e-12), 0.0, 1.0)
                )
                supports["log1p"].append(
                    np.clip(
                        np.log1p(observed) / max(np.log1p(reference), 1e-12),
                        0.0,
                        1.0,
                    )
                )
                supports["within_pct"].append(
                    np.clip(
                        float(sample_within.get(gene, 0.0))
                        / max(reference_within, 1e-12),
                        0.0,
                        1.0,
                    )
                )
                observed_cohort_pct = float(
                    np.mean(raw_ref.loc[gene].to_numpy(dtype=float) < observed)
                )
                supports["cohort_pct"].append(
                    np.clip(
                        observed_cohort_pct / max(reference_cohort_pct, 1e-12),
                        0.0,
                        1.0,
                    )
                )

            supports["log+cohort"] = [
                float(np.mean(values))
                for values in zip(
                    supports["log1p"],
                    supports["cohort_pct"],
                )
            ]
            supports["rank+log+cohort"] = [
                float(np.mean(values))
                for values in zip(
                    supports["within_pct"],
                    supports["log1p"],
                    supports["cohort_pct"],
                )
            ]

            for feature, values in supports.items():
                high_support = float(np.mean(values)) if values else 0.0
                score = (high_support**0.6) * (low_fraction**0.4)
                parent_scores[feature][panel.parent_cohort] = max(
                    parent_scores[feature][panel.parent_cohort], score
                )

        records.append(
            {
                "truth": truth,
                "sample": sample_name,
                "scores": parent_scores,
            }
        )

    print(f"samples={len(records)} parent programs={len(parents)}")
    print("\nfeature       macro_auc top_parent_accuracy eligible")
    print("------------------------------------------------------")
    for feature in FEATURE_SPACES:
        aucs = []
        for parent in parents:
            labels = [
                _truth_matches_parent(record["truth"], parent)
                for record in records
            ]
            auc = _auc(
                [record["scores"][feature][parent] for record in records],
                labels,
            )
            if np.isfinite(auc):
                aucs.append(auc)
        eligible = [
            record
            for record in records
            if any(
                _truth_matches_parent(record["truth"], parent)
                for parent in parents
            )
        ]
        correct = 0
        for record in eligible:
            prediction = max(
                record["scores"][feature],
                key=record["scores"][feature].get,
            )
            correct += int(_truth_matches_parent(record["truth"], prediction))
        print(
            f"{feature:12s} {np.mean(aucs):9.3f} "
            f"{correct / max(len(eligible), 1):19.3f} {len(eligible):8d}"
        )


if __name__ == "__main__":
    main()
