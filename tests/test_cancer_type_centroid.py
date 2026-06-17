"""Data-derived centroid + range-plausibility (the #83 coarse-lineage anchor).

Self-recovery tests: a cohort's own reference profile must match itself. These use
only the shipped reference matrices (no ~/data dependency), so they're
deterministic and CI-safe.
"""
import numpy as np
import pandas as pd
import pytest

from trufflepig.cancer_type_centroid import (
    centroid_correlations,
    coarse_lineage_scores,
    range_plausibility,
)
from trufflepig.reference import (
    pan_cancer_expression,
    subtype_deconvolved_expression,
    tcga_deconvolved_expression,
)


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


def _deconv_cohort_as_sample(code):
    """Tumor-only {symbol: tpm} from a cohort's deconvolved tumor median."""
    for fn in (tcga_deconvolved_expression, subtype_deconvolved_expression):
        d = fn(technical_rna_normalize=True)
        sub = d[d["cancer_code"].astype(str) == code]
        if len(sub):
            return dict(zip(sub["symbol"].astype(str), sub["tumor_tpm_median"].astype(float)))
    return {}


@pytest.mark.parametrize("code", ["COAD", "SARC", "BRCA", "NUTM"])
def test_tumor_only_cohort_matches_itself(code):
    """A cohort's own deconvolved tumor profile matches its own tumor-only centroid —
    including NUTM, which is invisible to the 33-code bulk centroids."""
    from trufflepig.cancer_type_centroid import tumor_only_correlations

    corr = tumor_only_correlations(_deconv_cohort_as_sample(code))
    assert not corr.empty
    assert corr.index[0] == code, f"{code}: top was {corr.index[0]}"
    assert corr.iloc[0] > 0.99


def test_tumor_only_coverage_exceeds_bulk_and_includes_rare_types():
    """The tumor-only space covers the subtypes / rare types the bulk centroids miss."""
    from trufflepig.cancer_type_centroid import _deconvolved_centroids

    _, codes = _deconvolved_centroids()
    cs = set(codes)
    assert len(codes) > 50  # ~63, far beyond the 33 bulk cohorts
    for code in ("NUTM", "RB"):  # absent from the 33 TCGA bulk columns
        assert code in cs, f"{code} missing from tumor-only coverage"


def test_tumor_only_correlations_empty_on_empty_input():
    from trufflepig.cancer_type_centroid import tumor_only_correlations

    assert tumor_only_correlations({}).empty


def _cohort_sample_df(code):
    """A gene-expression frame built from a cohort's own bulk centroid column."""
    ref = pan_cancer_expression(technical_rna_normalize=True).drop_duplicates(
        subset="Ensembl_Gene_ID"
    )
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{code}_TPM"].astype(float),
        }
    )


def test_ranker_annotates_centroid_crosscheck_and_stays_additive():
    """The centroid cross-check wiring in rank_cancer_type_candidates must annotate
    every candidate (additive — new keys only) and must NOT change the call.

    A cohort's own median classifies as itself (the marker-panel ranking), and on a
    SARC profile the data-derived coarse lineage is Sarcoma and agrees with the
    SARC call.
    """
    from trufflepig.tumor_purity import rank_cancer_type_candidates

    rows = rank_cancer_type_candidates(_cohort_sample_df("SARC"), top_k=5)
    assert rows
    # additive: the cross-check adds these to every candidate
    for r in rows:
        assert "centroid_correlation" in r
        assert "range_plausibility" in r
        assert 0.0 <= float(r["range_plausibility"]) <= 1.0
    top = rows[0]
    # the marker-panel call (SARC on its own median) is unchanged by the cross-check
    assert top["code"] == "SARC"
    # data-derived coarse lineage is computed and agrees with the call
    assert top.get("centroid_coarse_lineage") == "Sarcoma"
    assert top.get("centroid_lineage_agreement") is True
    assert top.get("centroid_top_code")


def test_ranker_crosscheck_flags_cross_lineage_disagreement():
    """When the call's lineage disagrees with the data-derived one, the flag is
    False — the mis-call detector. A SARC-profile sample CONSTRAINED to call COAD
    (epithelial) must flag the disagreement (Sarcoma vs Epithelial)."""
    from trufflepig.tumor_purity import rank_cancer_type_candidates

    rows = rank_cancer_type_candidates(
        _cohort_sample_df("SARC"), candidate_codes=["COAD"], top_k=3
    )
    assert rows and rows[0]["code"] == "COAD"
    # the data still says Sarcoma; the epithelial call disagrees
    assert rows[0].get("centroid_coarse_lineage") == "Sarcoma"
    assert rows[0].get("centroid_lineage_agreement") is False
