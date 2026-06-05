"""Regression tests for the PANNET cohort integration (part of #152).

Alvarez 2018 (GSE118014, 33 pancreatic neuroendocrine samples) —
log2(TPM+1) matrix from RSEM/STAR, aggregated to per-gene
(median, Q1, Q3) and appended to ``subtype-deconvolved-expression``
as a new ``cancer_code=PANNET`` block.

The registry entry was already present in v4.38.0 as
``expression_source=curated`` / metadata-only; this test pins the
real-data promotion.
"""

import pandas as pd

from pirlygenes.gene_sets_cancer import (
    cancer_type_registry,
    lineage_genes_by_cancer_type,
)
from trufflepig.reference import subtype_deconvolved_expression
from trufflepig.tumor_purity import LINEAGE_GENES


def test_pannet_registry_points_at_alvarez_cohort():
    reg = cancer_type_registry()
    row = reg[reg["code"] == "PANNET"].iloc[0]
    assert row["expression_source"] == "GEO"
    assert row["source_cohort"] == "GSE118014_ALVAREZ_2018"
    # `net` family renamed to lineage-only `neuroendocrine` (pirlygenes 5.12).
    assert row["family"] == "neuroendocrine"
    assert row["primary_tissue"] == "pancreas"


def test_pannet_expression_data_present_in_subtype_deconvolved():
    sub = subtype_deconvolved_expression()
    pannet = sub[sub["cancer_code"] == "PANNET"]
    # 33 samples → ~21K genes; after Ensembl-ID resolution we keep
    # about 79% (~17K). Tolerate some variance.
    assert len(pannet) > 10_000, (
        f"PANNET row count {len(pannet)} is too low — expected >10K "
        "post-Ensembl-resolution"
    )
    assert pannet["n_samples"].iloc[0] == 33
    assert pannet["source_cohort"].iloc[0] == "GSE118014_ALVAREZ_2018"


def test_pannet_canonical_ne_markers_are_high():
    """Sanity check that the cohort biology is correct — canonical
    pancreatic-NE markers should dominate the expression profile."""
    sub = subtype_deconvolved_expression()
    pannet = sub[sub["cancer_code"] == "PANNET"].set_index("symbol")

    # CHGA (chromogranin A) is the canonical panNET neuroendocrine
    # marker. Expect a very high median (~thousands of TPM).
    assert pannet.loc["CHGA", "tumor_tpm_median"] > 500, (
        f"CHGA median {pannet.loc['CHGA', 'tumor_tpm_median']:.1f} is "
        "unexpectedly low — cohort parse may be off"
    )
    # SYP (synaptophysin), ENO2 (NSE) — neuroendocrine core.
    assert pannet.loc["SYP", "tumor_tpm_median"] > 50
    assert pannet.loc["ENO2", "tumor_tpm_median"] > 50
    # SSTR2 — therapy target (octreotide / 177Lu-DOTATATE). The cohort
    # is panNET-enriched so SSTR2 should be meaningfully present.
    assert pannet.loc["SSTR2", "tumor_tpm_median"] > 5


def test_pannet_lineage_panel_present():
    panels = lineage_genes_by_cancer_type()
    assert "PANNET" in panels, "PANNET lineage panel not registered"
    panel = set(panels["PANNET"])
    # Canonical NE markers must anchor the panel — any future edit
    # that drops these needs to be explicit.
    for gene in ("CHGA", "SYP", "ENO2"):
        assert gene in panel, f"PANNET lineage panel missing {gene}"
    # Therapy target included so downstream reporting surfaces it.
    assert "SSTR2" in panel


def test_pannet_panel_loaded_into_LINEAGE_GENES():
    """The module-level ``LINEAGE_GENES`` dict consumed by the purity
    estimator should carry the PANNET panel."""
    assert "PANNET" in LINEAGE_GENES
    assert "CHGA" in LINEAGE_GENES["PANNET"]
