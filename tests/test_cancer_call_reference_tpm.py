"""Direct tests for the reference-TPM readers in ``trufflepig.cancer_call``.

Regression for the review finding: a missing reference cell reads back as NaN, and
``float(nan or 0.0)`` is NaN (``bool(nan) is True``), so a tumor-up gene whose reference cell is
missing would pass the distinctiveness fold-change guards as a fake "distinctive tumor marker" and
let a dominant-normal-contamination candidate escape demotion. The readers must map any non-finite
cell to 0.0 and never return NaN from the max helper.
"""
from __future__ import annotations

import pandas as pd

from trufflepig import cancer_call


def test_reference_gene_tpm_maps_nan_cell_to_zero(monkeypatch):
    frame = pd.DataFrame(
        {"COAD_TPM": [float("nan"), 12.0]},
        index=["GENEA", "GENEB"],
    )
    monkeypatch.setattr(cancer_call, "_pan_reference_by_symbol", lambda: frame)
    cancer_call._reference_gene_tpm.cache_clear()
    try:
        # NaN reference cell → 0.0, NOT NaN (would otherwise read as a distinctive marker).
        assert cancer_call._reference_gene_tpm("COAD", "GENEA") == 0.0
        # A finite cell is returned verbatim.
        assert cancer_call._reference_gene_tpm("COAD", "GENEB") == 12.0
        # An absent symbol/column is 0.0, not a KeyError.
        assert cancer_call._reference_gene_tpm("COAD", "MISSING") == 0.0
        assert cancer_call._reference_gene_tpm("NOPE", "GENEB") == 0.0
    finally:
        cancer_call._reference_gene_tpm.cache_clear()


def test_max_reference_gene_tpm_never_returns_nan(monkeypatch):
    # max([nan, 12.0]) is order-dependent and can be NaN; the helper must ignore the NaN cell.
    frame = pd.DataFrame(
        {"COAD_TPM": [float("nan")], "CRC_TPM": [12.0]},
        index=["GENEA"],
    )
    monkeypatch.setattr(cancer_call, "_pan_reference_by_symbol", lambda: frame)
    monkeypatch.setattr(
        cancer_call, "_candidate_reference_codes", lambda code: ("COAD", "CRC")
    )
    cancer_call._reference_gene_tpm.cache_clear()
    try:
        assert cancer_call._max_reference_gene_tpm("COAD", "GENEA") == 12.0
    finally:
        cancer_call._reference_gene_tpm.cache_clear()
