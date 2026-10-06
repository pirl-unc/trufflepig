"""Training labels and library denominators must survive normalization."""

import numpy as np
import pandas as pd
import pytest

from trufflepig import expression_classifier as classifier


@pytest.mark.parametrize("builder", ["_training_matrices", "_mismatch_repair_training_matrix"])
def test_normalization_preserves_sample_identity_and_complete_denominators(monkeypatch, builder):
    import oncoref.normalization
    import pirlygenes.expression.accessors as accessors

    cohorts = {
        "CRC_MSI": pd.DataFrame({
            "Ensembl_Gene_ID": ["shared", "msi_only"],
            "Symbol": ["SHARED", "MSI_ONLY"],
            "z_msi": [10.0, 90.0],
        }),
        "CRC_MSS": pd.DataFrame({
            "Ensembl_Gene_ID": ["shared", "mss_only"],
            "Symbol": ["SHARED", "MSS_ONLY"],
            "a_mss": [20.0, 180.0],
        }),
    }
    seen = []

    def normalize(frame, *, gene_table):
        seen.append(set(frame.index))
        # A normalizer may sort columns while retaining their public names.
        return frame.div(frame.sum(axis=0), axis=1)[sorted(frame.columns)] * 1_000_000

    monkeypatch.setattr(accessors, "available_representative_cohorts", lambda: tuple(cohorts))
    monkeypatch.setattr(accessors, "representative_cohort_samples", cohorts.__getitem__)
    monkeypatch.setattr(oncoref.normalization, "clean_tpm", normalize)
    build = getattr(classifier, builder)
    build.cache_clear()
    try:
        result = build()
        assert result is not None
        matrix, labels, genes = result[0] if builder == "_training_matrices" else result[:3]
        assert seen == [{"shared", "msi_only", "mss_only"}]
        assert genes == ["SHARED"]
        np.testing.assert_allclose(matrix[:, 0], np.log1p([100_000, 100_000]))
        assert list(labels) == (["CRC_MSS", "CRC_MSI"] if builder == "_training_matrices" else ["MSS", "MSI"])
        if builder == "_mismatch_repair_training_matrix":
            assert result[4] == ("CRC_MSS", "CRC_MSI")
    finally:
        build.cache_clear()
