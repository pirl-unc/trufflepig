"""Tests for the #22 deconvolved-TCGA reference columns.

#22 adds ``tcga_<CODE>`` columns to :func:`pan_cancer_expression`
sourced from the offline-deconvolved CSV shipped in ``data/``. These
tests exercise both the present-CSV and absent-CSV paths without
requiring the real ~2 MB reference CSV to be in the repo.
"""

import pandas as pd
import pytest

import pirlygenes.gene_sets_cancer as gsc


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch):
    """Each test starts with empty pan-cancer / deconv caches."""
    gsc._PAN_CANCER_CACHE.clear()
    # _tcga_deconv_wide closes over a default-arg dict; reset by
    # clearing the ``cache`` kwarg's container.
    gsc._tcga_deconv_wide.__defaults__[0].clear()
    yield
    gsc._PAN_CANCER_CACHE.clear()
    gsc._tcga_deconv_wide.__defaults__[0].clear()


def test_absent_deconv_csv_is_graceful(monkeypatch):
    """Without the deconv CSV, pan_cancer_expression returns the
    usual frame with no tcga_ columns."""
    monkeypatch.setattr(gsc, "tcga_deconvolved_expression", lambda: None)
    df = gsc.pan_cancer_expression()
    assert not any(c.startswith("tcga_") for c in df.columns)
    assert any(c.startswith("FPKM_") for c in df.columns)


def test_present_deconv_csv_adds_tcga_columns(monkeypatch):
    synthetic = pd.DataFrame(
        [
            {
                "symbol": "KLK3",
                "cancer_code": "PRAD",
                "tumor_tpm_median": 11000.0,
                "tumor_tpm_q1": 5500.0,
                "tumor_tpm_q3": 16000.0,
                "n_samples": 30,
            },
            {
                "symbol": "KLK3",
                "cancer_code": "BRCA",
                "tumor_tpm_median": 0.1,
                "tumor_tpm_q1": 0.0,
                "tumor_tpm_q3": 0.5,
                "n_samples": 30,
            },
            {
                "symbol": "ERBB2",
                "cancer_code": "BRCA",
                "tumor_tpm_median": 150.0,
                "tumor_tpm_q1": 80.0,
                "tumor_tpm_q3": 300.0,
                "n_samples": 30,
            },
        ]
    )
    monkeypatch.setattr(gsc, "tcga_deconvolved_expression", lambda: synthetic)
    df = gsc.pan_cancer_expression(genes=["KLK3", "ERBB2"])
    tcga_cols = [c for c in df.columns if c.startswith("tcga_")]
    assert "tcga_PRAD" in tcga_cols
    assert "tcga_BRCA" in tcga_cols

    klk3 = df[df["Symbol"] == "KLK3"].iloc[0]
    assert klk3["tcga_PRAD"] == pytest.approx(11000.0)
    assert klk3["tcga_BRCA"] == pytest.approx(0.1)

    erbb2 = df[df["Symbol"] == "ERBB2"].iloc[0]
    assert erbb2["tcga_BRCA"] == pytest.approx(150.0)
    # ERBB2 missing from PRAD in the synthetic frame → NaN is the
    # correct merge outcome, not 0.
    assert pd.isna(erbb2["tcga_PRAD"])


def test_technical_rna_reference_normalization_is_opt_in_and_preserves_nans(monkeypatch):
    base = pd.DataFrame(
        [
            {
                "Ensembl_Gene_ID": "ENSG_RRNA",
                "Symbol": "RNA5SP389",
                "nTPM_prostate": 10.0,
                "FPKM_PRAD": 10.0,
            },
            {
                "Ensembl_Gene_ID": "ENSG_KLK3",
                "Symbol": "KLK3",
                "nTPM_prostate": 90.0,
                "FPKM_PRAD": 90.0,
            },
            {
                "Ensembl_Gene_ID": "ENSG_ERBB2",
                "Symbol": "ERBB2",
                "nTPM_prostate": 5.0,
                "FPKM_PRAD": 5.0,
            },
        ]
    )
    deconv = pd.DataFrame(
        [
            {
                "symbol": "RNA5SP389",
                "cancer_code": "PRAD",
                "tumor_tpm_median": 10.0,
                "tumor_tpm_q1": 5.0,
                "tumor_tpm_q3": 15.0,
                "n_samples": 3,
            },
            {
                "symbol": "KLK3",
                "cancer_code": "PRAD",
                "tumor_tpm_median": 90.0,
                "tumor_tpm_q1": 45.0,
                "tumor_tpm_q3": 135.0,
                "n_samples": 3,
            },
            {
                "symbol": "ERBB2",
                "cancer_code": "BRCA",
                "tumor_tpm_median": 150.0,
                "tumor_tpm_q1": 80.0,
                "tumor_tpm_q3": 220.0,
                "n_samples": 3,
            },
        ]
    )

    monkeypatch.setattr(gsc, "get_data", lambda name: base.copy())
    monkeypatch.setattr(gsc, "tcga_deconvolved_expression", lambda: deconv)

    raw = gsc.pan_cancer_expression()
    assert raw.loc[raw["Symbol"] == "RNA5SP389", "FPKM_PRAD"].iloc[0] == 10.0
    assert raw.loc[raw["Symbol"] == "RNA5SP389", "tcga_PRAD"].iloc[0] == 10.0

    normalized = gsc.pan_cancer_expression(technical_rna_normalize=True)
    assert normalized.loc[normalized["Symbol"] == "RNA5SP389", "FPKM_PRAD"].iloc[0] == 0.0
    assert normalized.loc[normalized["Symbol"] == "KLK3", "FPKM_PRAD"].iloc[0] == pytest.approx(90.0 * 105.0 / 95.0)
    assert normalized.loc[normalized["Symbol"] == "RNA5SP389", "tcga_PRAD"].iloc[0] == 0.0
    assert normalized.loc[normalized["Symbol"] == "KLK3", "tcga_PRAD"].iloc[0] == pytest.approx(100.0)
    assert pd.isna(normalized.loc[normalized["Symbol"] == "ERBB2", "tcga_PRAD"].iloc[0])


def test_subtype_deconvolved_technical_rna_normalization_is_opt_in(monkeypatch):
    synthetic = pd.DataFrame(
        [
            {
                "symbol": "RNA5SP389",
                "cancer_code": "PRAD_NEPC",
                "subtype": "",
                "tumor_tpm_median": 20.0,
                "tumor_tpm_q1": 10.0,
                "tumor_tpm_q3": 30.0,
                "n_samples": 4,
            },
            {
                "symbol": "KLK3",
                "cancer_code": "PRAD_NEPC",
                "subtype": "",
                "tumor_tpm_median": 80.0,
                "tumor_tpm_q1": 40.0,
                "tumor_tpm_q3": 120.0,
                "n_samples": 4,
            },
        ]
    )

    monkeypatch.setattr(gsc, "get_data", lambda name: synthetic.copy())

    raw = gsc.subtype_deconvolved_expression()
    assert raw.loc[raw["symbol"] == "RNA5SP389", "tumor_tpm_median"].iloc[0] == 20.0
    assert raw.loc[raw["symbol"] == "KLK3", "tumor_tpm_median"].iloc[0] == 80.0

    normalized = gsc.subtype_deconvolved_expression(technical_rna_normalize=True)
    assert (
        normalized.loc[
            normalized["symbol"] == "RNA5SP389",
            "tumor_tpm_median",
        ].iloc[0]
        == 0.0
    )
    assert normalized.loc[
        normalized["symbol"] == "KLK3",
        "tumor_tpm_median",
    ].iloc[0] == pytest.approx(100.0)


def test_tcga_columns_participate_in_percentile_normalize(monkeypatch):
    """Percentile normalize must treat tcga_ columns the same as FPKM_."""
    synthetic = pd.DataFrame(
        [
            {
                "symbol": "KLK3",
                "cancer_code": "PRAD",
                "tumor_tpm_median": 11000.0,
                "tumor_tpm_q1": 0.0,
                "tumor_tpm_q3": 0.0,
                "n_samples": 30,
            },
            {
                "symbol": "GAPDH",
                "cancer_code": "PRAD",
                "tumor_tpm_median": 500.0,
                "tumor_tpm_q1": 0.0,
                "tumor_tpm_q3": 0.0,
                "n_samples": 30,
            },
        ]
    )
    monkeypatch.setattr(gsc, "tcga_deconvolved_expression", lambda: synthetic)
    df = gsc.pan_cancer_expression(normalize="percentile")
    assert "tcga_PRAD" in df.columns
    # Values are percentile ranks in [0, 100].
    s = df["tcga_PRAD"].dropna()
    assert (s.min() >= 0) and (s.max() <= 100)


def test_tcga_deconvolved_expression_returns_none_when_missing(monkeypatch):
    """Raising ValueError from get_data must be swallowed as None."""

    def _missing(_name):
        raise ValueError("Dataset tcga-deconvolved-expression not found")

    monkeypatch.setattr(gsc, "get_data", _missing)
    assert gsc.tcga_deconvolved_expression() is None
