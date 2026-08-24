# Licensed under the Apache License, Version 2.0

"""Unit tests for decomposition engine internals.

Covers: _weighted_constrained_nnls, template scoring, and
_fit_one_hypothesis lineage-panel stability threshold.
"""

import numpy as np
import pandas as pd
import pytest

from trufflepig.decomposition import panels as panel_mod
from trufflepig.decomposition.engine import (
    DECOMPOSITION_PARAMETERS,
    _weighted_constrained_nnls,
    _is_excluded_auto_marker,
)
from trufflepig.decomposition.panels import build_shared_lineage_panel


# ── _weighted_constrained_nnls ───────────────────────────────────────────


def test_prad_shared_lineage_panel_keeps_klk3():
    panel = build_shared_lineage_panel("PRAD")

    assert "KLK3" in set(panel["symbol"])


def test_prad_shared_lineage_panel_curated_klk3_survives_ratio_cliff():
    panel = build_shared_lineage_panel("PRAD", tolerance_log2=0.1)
    klk3 = panel[panel["symbol"] == "KLK3"].iloc[0]

    assert klk3["shared_lineage_basis"] == "curated_lineage"


def test_shared_lineage_curated_fallback_is_cancer_agnostic(monkeypatch):
    fake_panel = pd.DataFrame(
        {
            "symbol": ["CURATED_MARKER", "GENERIC_MARKER"],
            "tumor_tpm": [20.0, 20.0],
            "normal_ntpm": [80.0, 80.0],
            "tcga_bulk_tpm": [20.0, 20.0],
        }
    )

    monkeypatch.setattr(
        panel_mod,
        "_load_cohort_vs_tissue",
        lambda cancer_code, tissue=None: (fake_panel.copy(), "fake_tissue"),
    )
    monkeypatch.setattr(
        panel_mod,
        "lineage_gene_symbols",
        lambda cancer_code: ["CURATED_MARKER"] if cancer_code == "FAKE" else [],
    )

    panel = build_shared_lineage_panel(
        "FAKE",
        tolerance_log2=0.1,
        curated_lineage_tolerance_log2=3.0,
        min_expression=5.0,
    )

    assert panel["symbol"].tolist() == ["CURATED_MARKER"]
    assert panel["shared_lineage_basis"].tolist() == ["curated_lineage"]


def test_nnls_sum_to_one():
    """Solution fractions should sum to approximately 1.0."""
    rng = np.random.default_rng(42)
    A = rng.random((20, 3))
    true_mix = np.array([0.5, 0.3, 0.2])
    b = A @ true_mix + rng.normal(0, 0.01, 20)
    solution, residual = _weighted_constrained_nnls(A, b)
    assert abs(solution.sum() - 1.0) < 0.01, f"Sum = {solution.sum()}"


def test_nnls_recovers_known_mix():
    """Should approximately recover a known mixing vector."""
    rng = np.random.default_rng(0)
    # Create well-separated component signatures
    A = np.zeros((30, 3))
    A[:10, 0] = rng.uniform(5, 10, 10)  # component 0 markers
    A[10:20, 1] = rng.uniform(5, 10, 10)  # component 1 markers
    A[20:30, 2] = rng.uniform(5, 10, 10)  # component 2 markers
    true_mix = np.array([0.6, 0.25, 0.15])
    b = A @ true_mix
    solution, residual = _weighted_constrained_nnls(A, b)
    assert np.allclose(solution, true_mix, atol=0.05), (
        f"Got {solution}, expected {true_mix}"
    )


def test_nnls_fit_is_invariant_to_tpm_scale():
    """The same mixture in different linear units has the same fit quality."""
    A = np.array(
        [
            [8.0, 1.0],
            [1.0, 9.0],
            [6.0, 2.0],
            [2.0, 7.0],
        ]
    )
    b = A @ np.array([0.65, 0.35]) + np.array([0.1, -0.1, 0.2, -0.2])

    solution, residual = _weighted_constrained_nnls(
        A,
        b,
        sum_to_one_weight=0.0,
        l2_penalty=0.0,
    )
    scaled_solution, scaled_residual = _weighted_constrained_nnls(
        A * 10_000.0,
        b * 10_000.0,
        sum_to_one_weight=0.0,
        l2_penalty=0.0,
    )

    assert scaled_solution == pytest.approx(solution)
    assert scaled_residual == pytest.approx(residual)
    assert 0.0 < residual < 1.0


def test_nnls_empty_matrix():
    """Empty matrix should return empty solution with inf residual."""
    A = np.zeros((0, 3))
    b = np.zeros(0)
    solution, residual = _weighted_constrained_nnls(A, b)
    assert len(solution) == 3
    assert np.all(solution == 0)
    assert residual == float("inf")


def test_nnls_zero_columns():
    """Zero-column matrix should return empty solution."""
    A = np.zeros((5, 0))
    b = np.ones(5)
    solution, residual = _weighted_constrained_nnls(A, b)
    assert len(solution) == 0
    assert residual == float("inf")


def test_nnls_single_component():
    """Single component should get fraction = 1.0."""
    A = np.array([[1.0], [2.0], [3.0]])
    b = np.array([1.0, 2.0, 3.0])
    solution, residual = _weighted_constrained_nnls(A, b)
    assert len(solution) == 1
    assert solution[0] == pytest.approx(1.0, abs=0.01)
    assert residual < 0.1


def test_nnls_with_weights():
    """Weights should influence the fit — high-weight rows matter more."""
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    b = np.array([0.8, 0.2, 1.0])
    # Weight the first row heavily
    weights = np.array([10.0, 1.0, 1.0])
    solution, _ = _weighted_constrained_nnls(A, b, weights=weights)
    # Component 0 should dominate
    assert solution[0] > solution[1]


def test_nnls_ridge_reduces_collinear_inflation():
    """Ridge penalty should prevent one component from absorbing everything
    when components are near-collinear."""
    # Two nearly identical components
    A = np.array([[1.0, 1.01], [2.0, 2.02], [3.0, 3.03]])
    b = np.array([1.5, 3.0, 4.5])
    solution_ridge, _ = _weighted_constrained_nnls(A, b, l2_penalty=0.05)
    solution_no_ridge, _ = _weighted_constrained_nnls(A, b, l2_penalty=0.0)
    # With ridge, the solution should be more balanced
    ridge_diff = abs(solution_ridge[0] - solution_ridge[1])
    no_ridge_diff = abs(solution_no_ridge[0] - solution_no_ridge[1])
    # Ridge should produce less extreme imbalance (or at least not worse)
    assert ridge_diff <= no_ridge_diff + 0.1


def test_nnls_nonnegative():
    """All solution components must be >= 0."""
    rng = np.random.default_rng(7)
    A = rng.random((15, 4))
    b = rng.random(15)
    solution, _ = _weighted_constrained_nnls(A, b)
    assert np.all(solution >= 0)


# ── _hk_normalize ────────────────────────────────────────────────────────


# ── _is_excluded_auto_marker ─────────────────────────────────────────────


def test_auto_marker_excludes_mhc_ii():
    assert _is_excluded_auto_marker("HLA-DRA")
    assert _is_excluded_auto_marker("HLA-DRB1")
    assert _is_excluded_auto_marker("CD74")


def test_auto_marker_excludes_ribosomal():
    assert _is_excluded_auto_marker("RPL13")
    assert _is_excluded_auto_marker("RPS27A")


def test_auto_marker_preserves_legitimate_markers():
    assert not _is_excluded_auto_marker("CD8A")
    assert not _is_excluded_auto_marker("FOXP3")
    assert not _is_excluded_auto_marker("EGFR")


def test_auto_marker_handles_non_string():
    assert not _is_excluded_auto_marker(None)
    assert not _is_excluded_auto_marker(42)


# ── Template scoring parameters ──────────────────────────────────────────


def test_template_scoring_parameters_consistency():
    """Primary site base + gain should reach 1.0 when origin_tissue_score=1.0."""
    ts = DECOMPOSITION_PARAMETERS["template_scoring"]
    max_primary = ts["primary_site_base"] + ts["primary_site_gain"]
    assert max_primary == pytest.approx(1.0)


def test_met_site_factor_range():
    """Met site factor should range from base to base+gain."""
    ts = DECOMPOSITION_PARAMETERS["template_scoring"]
    lo = ts["met_site_base"]
    hi = ts["met_site_base"] + ts["met_site_gain"]
    assert lo > 0
    assert hi <= 1.0


def test_lineage_override_upward_delta_is_positive():
    lo = DECOMPOSITION_PARAMETERS["lineage_override"]
    assert lo["max_upward_delta"] > 0
    assert lo["max_upward_ratio"] > 1.0


def test_min_tumor_fraction_parameter_exists():
    ts = DECOMPOSITION_PARAMETERS["template_scoring"]
    assert "min_tumor_fraction" in ts
    assert 0 < ts["min_tumor_fraction"] < 0.1


def test_low_purity_candidate_is_penalised():
    """A decomposition candidate with purity < min_tumor_fraction should
    have a lower score than one at normal purity, all else equal."""
    import pandas as pd
    from trufflepig.reference import pan_cancer_expression
    from trufflepig.decomposition import decompose_sample

    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    # Build a PRAD sample — should score well for PRAD, poorly for unrelated types
    df = pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref["PRAD_TPM"].astype(float),
        }
    )
    results = decompose_sample(
        df,
        cancer_types=["PRAD", "HNSC"],
        templates=["solid_primary"],
        top_k=2,
    )
    assert len(results) == 2
    prad_result = next(r for r in results if r.cancer_type == "PRAD")
    hnsc_result = next(r for r in results if r.cancer_type == "HNSC")
    # If HNSC has very low purity, it should have been penalised
    if hnsc_result.purity < 0.02:
        assert hnsc_result.score < prad_result.score * 0.5
        assert any("floor" in w for w in hnsc_result.warnings)
