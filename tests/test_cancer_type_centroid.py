"""Data-derived centroid + range-plausibility (the #83 coarse-lineage anchor).

Self-recovery tests: a cohort's own reference profile must match itself. These use
only the shipped reference matrices (no ~/data dependency), so they're
deterministic and CI-safe.
"""
import numpy as np
import pytest

from trufflepig.cancer_type_centroid import (
    centroid_correlations,
    coarse_lineage_scores,
    range_plausibility,
)
from trufflepig.reference import pan_cancer_expression, tcga_deconvolved_expression


def _bulk_cohort_as_sample(code):
    """Build a {symbol: tpm} sample from a cohort's own bulk centroid column."""
    pan = pan_cancer_expression(technical_rna_normalize=True).drop_duplicates("Symbol")
    col = f"{code}_TPM"
    return dict(zip(pan["Symbol"].astype(str), pan[col].astype(float)))


def _dec_cohort_as_sample(code):
    """Build a {symbol: tpm} sample from a cohort's deconvolved tumor-only median."""
    dec = tcga_deconvolved_expression(technical_rna_normalize=True)
    sub = dec[dec["cancer_code"].astype(str) == code]
    return dict(zip(sub["symbol"].astype(str), sub["tumor_tpm_median"].astype(float)))


@pytest.mark.parametrize("code", ["COAD", "SARC", "PRAD", "BLCA", "BRCA"])
def test_cohort_centroid_matches_itself(code):
    """A cohort's own bulk profile must correlate highest with its own centroid."""
    corr = centroid_correlations(_bulk_cohort_as_sample(code))
    assert not corr.empty
    assert corr.index[0] == code, f"{code}: top centroid was {corr.index[0]}, top3={list(corr.head(3).items())}"
    # self-correlation is (near) perfect
    assert corr.iloc[0] > 0.99


def test_sarc_centroid_coarse_lineage_is_sarcoma():
    """The mesenchymal cohort resolves to the Sarcoma coarse group, not Epithelial —
    this is the data-derived signal that fixes the stroma->SARC/HNSC mis-call."""
    lin = coarse_lineage_scores(_bulk_cohort_as_sample("SARC"))
    assert not lin.empty
    assert lin.index[0] == "Sarcoma"


def test_epithelial_centroid_coarse_lineage_is_epithelial():
    lin = coarse_lineage_scores(_bulk_cohort_as_sample("COAD"))
    assert not lin.empty
    assert lin.index[0] == "Epithelial"


def test_range_plausibility_high_for_own_cohort_low_for_wrong_lineage():
    """A cohort's own tumor-only profile is plausible for itself; a cross-lineage
    type's markers are NOT in range — this is the sanity veto's signal."""
    sample = _dec_cohort_as_sample("SARC")
    assert range_plausibility("SARC", sample) > 0.6
    # HNSC (squamous) markers are implausible on a sarcoma profile
    assert range_plausibility("HNSC", sample) < range_plausibility("SARC", sample)


def test_centroid_correlations_empty_on_empty_input():
    assert centroid_correlations({}).empty
    assert coarse_lineage_scores({}).empty


def test_range_plausibility_abstains_for_unknown_code():
    # a code with no deconvolved reference returns 1.0 (abstain, never invents)
    assert range_plausibility("NOT_A_REAL_CODE", {"TP53": 10.0}) == 1.0
