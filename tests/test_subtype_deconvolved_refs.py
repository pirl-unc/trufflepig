"""Subtype-deconvolved expression reference contract — cross-package.

Pins specific cohort row-sets in
``subtype-deconvolved-expression.csv.gz`` so registry codes that
reference a cohort (via ``expression_source`` in
``cancer-type-registry.csv``) actually have the rows. Lived in
``pirlygenes/tests/test_cancer_type_registry.py`` before the #23
expression-data move; relocated here because
:func:`trufflepig.reference.subtype_deconvolved_expression` owns the
matrix now.
"""

from trufflepig.reference import subtype_deconvolved_expression


def test_gse75885_sarcoma_expression_refs_are_bundled():
    d = subtype_deconvolved_expression()
    assert d is not None

    expected = {
        "SARC_DDLPS": 19,
        "SARC_PLEOLPS": 4,
        "SARC_LGFMS": 2,
    }
    for code, n_samples in expected.items():
        rows = d[
            (d["cancer_code"] == code)
            & (d["source_cohort"] == "GSE75885_DELESPAUL_2017")
        ]
        assert not rows.empty
        assert int(rows["n_samples"].max()) == n_samples

    ddlps = d[d["cancer_code"] == "SARC_DDLPS"].set_index("symbol")
    assert ddlps.loc["MDM2", "tumor_tpm_median"] > 1000
    assert ddlps.loc["CDK4", "tumor_tpm_median"] > 500

    lgfms = d[d["cancer_code"] == "SARC_LGFMS"].set_index("symbol")
    assert lgfms.loc["MUC4", "tumor_tpm_median"] > 1000
    assert lgfms.loc["CREB3L2", "tumor_tpm_median"] > 100


def test_gse299759_chondrosarcoma_expression_ref_is_bundled():
    d = subtype_deconvolved_expression()
    assert d is not None

    rows = d[
        (d["cancer_code"] == "SARC_CHON")
        & (d["source_cohort"] == "GSE299759_MEIJER_2026")
    ]
    assert not rows.empty
    assert int(rows["n_samples"].max()) == 54

    chon = rows.set_index("symbol")
    assert chon.loc["COL2A1", "tumor_tpm_median"] > 1000
    assert chon.loc["ACAN", "tumor_tpm_median"] > 1000
    assert chon.loc["SOX9", "tumor_tpm_median"] > 100


def test_treehouse_ribod_sparse_expression_refs_are_bundled():
    d = subtype_deconvolved_expression()
    assert d is not None

    expected = {
        "RB": 15,
        "SARC_CHOR": 3,
    }
    for code, n_samples in expected.items():
        rows = d[
            (d["cancer_code"] == code)
            & (d["source_cohort"] == "TREEHOUSE_RIBOD_25_01")
        ]
        assert not rows.empty
        assert int(rows["n_samples"].max()) == n_samples

    rb = d[d["cancer_code"] == "RB"].set_index("symbol")
    assert rb.loc["CRX", "tumor_tpm_median"] > 25
    assert rb.loc["ARR3", "tumor_tpm_median"] > 5

    chor = d[d["cancer_code"] == "SARC_CHOR"].set_index("symbol")
    assert chor.loc["COL2A1", "tumor_tpm_median"] > 25
    assert chor.loc["ACAN", "tumor_tpm_median"] > 10
