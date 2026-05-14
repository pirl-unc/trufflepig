"""Tests for the non-TCGA cohort summarisation helpers.

These exercise the pure transforms — RPKM→TPM, per-cohort aggregation,
subtype-partitioned aggregation — without requiring the cBioPortal
cohort files (too large for CI; produced offline by the maintainer).
"""

import pandas as pd
import pytest

from trufflepig.cohort_deconvolve import (
    load_log2_tpm,
    load_subtype_labels,
    rpkm_to_tpm,
    summarise_passthrough,
    validate_tpm_sample_sums,
)


def test_rpkm_to_tpm_columns_sum_to_one_million():
    rpkm = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "s1": [10.0, 30.0, 40.0, 20.0],
            "s2": [5.0, 15.0, 0.0, 80.0],
        }
    )
    tpm = rpkm_to_tpm(rpkm)
    # Each sample's TPM column must sum to 1e6 (the definition of TPM).
    assert tpm[["s1", "s2"]].sum(axis=0).round(2).tolist() == [1_000_000.0, 1_000_000.0]


def test_rpkm_to_tpm_zero_column_survives_without_crash():
    """A sample with all-zero RPKM must not divide-by-zero; the column
    is left as zero rather than propagating NaNs."""
    rpkm = pd.DataFrame({"symbol": ["A", "B"], "s1": [0.0, 0.0], "s2": [100.0, 100.0]})
    tpm = rpkm_to_tpm(rpkm)
    assert (tpm["s1"] == 0).all()
    assert tpm["s2"].sum().round(2) == 1_000_000.0


def test_load_log2_tpm_inverts_and_qcs_sample_sums(tmp_path):
    matrix = tmp_path / "treehouse.tsv"
    matrix.write_text(
        "Gene\ts1\ts2\n"
        "A\t18.93157145471137\t18.93157145471137\n"
        "B\t18.93157145471137\t18.93157145471137\n"
    )

    tpm = load_log2_tpm(matrix)

    assert tpm[["s1", "s2"]].sum(axis=0).round(2).tolist() == [
        1_000_000.0,
        1_000_000.0,
    ]
    assert tpm.loc[tpm["symbol"] == "A", "s1"].iloc[0] == pytest.approx(500_000.0)


def test_tpm_sample_sum_qc_rejects_wrong_scale():
    tpm = pd.DataFrame({"symbol": ["A", "B"], "s1": [10.0, 20.0]})

    with pytest.raises(ValueError, match="TPM sample-sum QC failed"):
        validate_tpm_sample_sums(tpm)


def test_summarise_passthrough_without_subtype():
    tpm = pd.DataFrame(
        {
            "symbol": ["GAPDH", "KLK3"],
            "s1": [500.0, 0.1],
            "s2": [600.0, 0.2],
            "s3": [700.0, 0.3],
        }
    )
    summary = summarise_passthrough(tpm, cohort_code="TEST_COHORT")
    assert list(summary.columns) == [
        "symbol",
        "cancer_code",
        "tumor_tpm_median",
        "tumor_tpm_q1",
        "tumor_tpm_q3",
        "n_samples",
    ]
    gapdh = summary[summary["symbol"] == "GAPDH"].iloc[0]
    assert gapdh["tumor_tpm_median"] == pytest.approx(600.0)
    assert gapdh["n_samples"] == 3


def test_summarise_passthrough_with_subtype_partition():
    """APL vs non-APL subtype split — the canonical AML curation case
    where MPO is expected to trend differently across groups."""
    tpm = pd.DataFrame(
        {
            "symbol": ["MPO"],
            "s1": [1000.0],  # APL
            "s2": [1200.0],  # APL
            "s3": [300.0],  # non-APL
            "s4": [400.0],  # non-APL
            "s5": [350.0],  # non-APL
        }
    )
    subtype_map = {
        "s1": "APL",
        "s2": "APL",
        "s3": "non_APL",
        "s4": "non_APL",
        "s5": "non_APL",
    }
    summary = summarise_passthrough(tpm, cohort_code="AML", subtype_map=subtype_map)
    assert "subtype" in summary.columns
    apl = summary[summary["subtype"] == "APL"].iloc[0]
    non_apl = summary[summary["subtype"] == "non_APL"].iloc[0]
    assert apl["tumor_tpm_median"] == pytest.approx(1100.0)
    assert non_apl["tumor_tpm_median"] == pytest.approx(350.0)
    # APL should have n_samples=2, non_APL=3.
    assert apl["n_samples"] == 2
    assert non_apl["n_samples"] == 3


def test_summarise_passthrough_drops_unlabeled_samples_when_subtype_map_given():
    """When caller passes a subtype map, samples NOT in the map get
    empty string subtype → dropped. Mixing unlabelled samples into a
    "subtype" group would pollute the reference."""
    tpm = pd.DataFrame(
        {
            "symbol": ["GAPDH"],
            "s1": [500.0],
            "s2": [600.0],
            "s3_unlabelled": [10.0],
        }
    )
    subtype_map = {"s1": "A", "s2": "A"}
    summary = summarise_passthrough(tpm, cohort_code="T", subtype_map=subtype_map)
    # Only s1 + s2 contribute.
    assert len(summary) == 1
    assert summary.iloc[0]["n_samples"] == 2
    # 10.0 must NOT show up in the median if it did, it'd be pulled far down.
    assert summary.iloc[0]["tumor_tpm_median"] == pytest.approx(550.0)


def test_load_subtype_labels_reads_csv(tmp_path):
    csv = tmp_path / "labels.csv"
    csv.write_text("TCGA-XX-YYYY,BRCA_LumA\nTCGA-AA-BBBB,BRCA_Basal\n")
    labels = load_subtype_labels(csv)
    assert labels["TCGA-XX-YYYY"] == "BRCA_LumA"
    assert labels["TCGA-AA-BBBB"] == "BRCA_Basal"


def test_summarise_passthrough_min_tpm_filter():
    """Genes whose max TPM across all samples is below ``min_tpm``
    must be dropped — keeps the shipped reference small by filtering
    noise genes."""
    tpm = pd.DataFrame(
        {
            "symbol": ["HIGH", "LOW"],
            "s1": [100.0, 0.001],
            "s2": [200.0, 0.002],
        }
    )
    summary = summarise_passthrough(tpm, cohort_code="T", min_tpm=0.01)
    symbols = set(summary["symbol"])
    assert "HIGH" in symbols
    assert "LOW" not in symbols


def test_subtype_deconvolved_expression_loads_shipped_data():
    """Smoke test against the shipped subtype-deconvolved CSV. Picks
    a BRCA-subtype → ERBB2 anchor that will fail loudly if the
    PAM50 re-partition broke."""
    from trufflepig.reference import subtype_deconvolved_expression

    d = subtype_deconvolved_expression()
    if d is None:
        pytest.skip("subtype-deconvolved-expression.csv.gz not in this checkout")
    # ERBB2 in BRCA_Her2 must be much higher than any other BRCA subtype —
    # this is the canonical HER2-amplification biology.
    her2_amp = d[
        (d["cancer_code"] == "BRCA")
        & (d["subtype"] == "BRCA_Her2")
        & (d["symbol"] == "ERBB2")
    ]
    basal = d[
        (d["cancer_code"] == "BRCA")
        & (d["subtype"] == "BRCA_Basal")
        & (d["symbol"] == "ERBB2")
    ]
    assert not her2_amp.empty and not basal.empty
    assert (
        her2_amp.iloc[0]["tumor_tpm_median"] > 5 * basal.iloc[0]["tumor_tpm_median"]
    ), (
        "HER2-subtype ERBB2 must be >5x the basal median — check the "
        "PAM50 re-partition didn't shuffle subtypes"
    )
