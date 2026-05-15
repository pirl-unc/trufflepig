"""Trufflepig-surface contracts for the pirlygenes 5.1.1 accessors.

The bulk of the composition logic (deconv merge, technical-RNA scrub,
renormalize-to-1e6, percentile / housekeeping normalization) lives in
pirlygenes 5.1.1 and is covered by pirlygenes' own test suite. The
tests below pin the *trufflepig-specific* contracts:

- ``trufflepig.reference.pan_cancer_expression()`` flips
  ``renormalize_to_million`` to True by default (pirlygenes' own
  default is False, so the user RNA-seq path here lands on a true
  TPM-1e6 footing without each caller having to opt in).
- ``trufflepig.reference.subtype_deconvolved_expression()`` does the
  same flip for the per-(cancer_code, subtype) groups.
- The ``tcga_<CODE>`` columns from the offline deconvolution are
  present in the trufflepig-surface pan-cancer frame (the wide-merge
  is wired up).
"""

import pytest

import trufflepig.reference as gsc


def test_pan_cancer_expression_renormalizes_to_million_by_default():
    """Trufflepig default: every value column should sum to ~1e6 so the
    analysis pipeline can compare user-sample TPM to cohort columns on
    a shared per-million footing."""
    df = gsc.pan_cancer_expression()
    value_cols = [
        c
        for c in df.columns
        if c.startswith(("FPKM_", "nTPM_", "tcga_"))
    ]
    assert value_cols, "expected per-tissue / per-cohort value columns"
    sample_col = next(c for c in value_cols if c.startswith("FPKM_"))
    col_sum = float(df[sample_col].astype(float).sum(skipna=True))
    assert col_sum == pytest.approx(1e6, rel=1e-3), col_sum


def test_pan_cancer_expression_native_scale_when_opted_out():
    """Decomposition opts out (renormalize_to_million=False, #27 pending);
    that path must return the native FPKM / nTPM column sums, which are
    *not* on a per-million footing."""
    df = gsc.pan_cancer_expression(renormalize_to_million=False)
    sample_col = next(c for c in df.columns if c.startswith("FPKM_"))
    col_sum = float(df[sample_col].astype(float).sum(skipna=True))
    # FPKM_* column totals in the bundled HPA reference are well below 1e6
    # (~200K–700K depending on cancer type); guard the broad range so
    # this stays stable across reference-data refreshes.
    assert 1e5 <= col_sum <= 9e5, col_sum


def test_pan_cancer_expression_has_tcga_deconvolved_columns():
    """``tcga_<CODE>`` columns from the offline deconvolution merge into
    the wide pan-cancer frame so callers can pull tumor-only medians
    alongside the FPKM_ cohort columns."""
    df = gsc.pan_cancer_expression(genes=["KLK3"])
    tcga_cols = [c for c in df.columns if c.startswith("tcga_")]
    assert tcga_cols, "expected tcga_<CODE> columns merged onto pan-cancer frame"
    assert "tcga_PRAD" in tcga_cols


def test_pan_cancer_expression_cache_returns_defensive_copies(monkeypatch):
    """The trufflepig shim caches transformed frames but callers still
    receive copies they can mutate safely."""
    import pandas as pd

    gsc._PAN_CANCER_CACHE.clear()
    calls = []

    def fake_pan_cancer_expression(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["ENSG00000142515"],
                "Symbol": ["KLK3"],
                "FPKM_PRAD": [1.0],
            }
        )

    monkeypatch.setattr(
        gsc._pirlygenes,
        "pan_cancer_expression",
        fake_pan_cancer_expression,
    )
    try:
        first = gsc.pan_cancer_expression(genes=["klk3"])
        first.loc[0, "FPKM_PRAD"] = 99.0
        second = gsc.pan_cancer_expression(genes=["KLK3"])

        assert len(calls) == 1
        assert second.loc[0, "FPKM_PRAD"] == 1.0
    finally:
        gsc._PAN_CANCER_CACHE.clear()


def test_subtype_deconvolved_expression_renormalizes_per_group_by_default():
    """Trufflepig default: each (cancer_code, subtype) group's
    tumor_tpm_median should sum to ~1e6 so cross-subtype comparisons are
    on a per-million footing."""
    df = gsc.subtype_deconvolved_expression()
    # Pick a well-populated cohort to avoid sparse-subtype edge cases.
    grp = df.groupby(["cancer_code", "subtype"], dropna=False)["tumor_tpm_median"].sum()
    nonzero = grp[grp > 0]
    assert len(nonzero) > 0
    median_sum = float(nonzero.median())
    assert median_sum == pytest.approx(1e6, rel=1e-3), median_sum


def test_subtype_deconvolved_expression_native_scale_when_opted_out():
    """Opt-out path returns the native bundled scale, which is *not* on a
    per-million footing per group."""
    df = gsc.subtype_deconvolved_expression(renormalize_to_million=False)
    grp = df.groupby(["cancer_code", "subtype"], dropna=False)["tumor_tpm_median"].sum()
    nonzero = grp[grp > 0]
    median_sum = float(nonzero.median())
    # Native per-subtype sums in the bundled subtype-deconv CSV range
    # roughly 1.3e5 – 1.15e6 (BRCA_Basal ~138K, BEATAML ~1.14M); pin a
    # broad guard around the typical band.
    assert median_sum != pytest.approx(1e6, rel=1e-3), median_sum
    assert 1e5 <= median_sum <= 1.2e6, median_sum


def test_cancer_expression_uses_trufflepig_reference_surface(monkeypatch):
    """Legacy helper composes trufflepig's pan-cancer wrapper, not the
    upstream pirlygenes default surface."""
    import pandas as pd

    calls = []

    def fake_pan_cancer_expression(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["ENSG00000142515"],
                "Symbol": ["KLK3"],
                "FPKM_PRAD": [42.0],
            }
        )

    monkeypatch.setattr(gsc, "pan_cancer_expression", fake_pan_cancer_expression)

    out = gsc.cancer_expression("PRAD", genes=["KLK3"])

    assert calls == [
        {
            "genes": ["KLK3"],
            "normalize": "housekeeping",
            "technical_rna_normalize": True,
        }
    ]
    assert list(out.columns) == ["Ensembl_Gene_ID", "Symbol", "expression"]
    assert out["expression"].tolist() == [42.0]


def test_cancer_enriched_genes_preserves_legacy_columns(monkeypatch):
    """The trufflepig surface keeps the historical ``fold_over_others``
    column instead of exposing pirlygenes' newer ``fold_change`` schema."""
    import pandas as pd

    def fake_pan_cancer_expression(**kwargs):
        assert kwargs == {
            "normalize": "housekeeping",
            "technical_rna_normalize": True,
        }
        return pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["ENSG_A", "ENSG_B", "ENSG_C"],
                "Symbol": ["A", "B", "C"],
                "FPKM_PRAD": [10.0, 1.0, 6.0],
                "FPKM_BRCA": [1.0, 10.0, 2.0],
                "FPKM_LUAD": [3.0, 8.0, 2.0],
            }
        )

    monkeypatch.setattr(gsc, "pan_cancer_expression", fake_pan_cancer_expression)

    out = gsc.cancer_enriched_genes("PRAD", min_fold=2.0, min_expression=0.0)

    assert list(out["Symbol"]) == ["A", "C"]
    assert "fold_over_others" in out.columns
    assert "fold_change" not in out.columns
    assert out["fold_over_others"].tolist() == pytest.approx([5.0, 3.0])
