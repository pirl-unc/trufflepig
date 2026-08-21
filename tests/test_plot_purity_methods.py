"""Tests for the purity-method comparison plot (#124)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from trufflepig.tumor_purity import plot_purity_method_comparison


def _purity_result(**overrides):
    base = {
        "cancer_type": "PRAD",
        "reference_expression_source": "pan_cancer",
        "tcga_median_purity": 0.69,
        "overall_estimate": 0.28,
        "overall_lower": 0.19,
        "overall_upper": 0.40,
        "components": {
            "signature": {
                "purity": 0.30,
                "lower": 0.18,
                "upper": 0.45,
                "stability": 0.6,
                "genes": ["AR", "KLK3", "KLK2", "STEAP1", "FOLH1"],
                "per_gene": [],
            },
            "lineage": {
                "purity": 0.32,
                "lower": 0.22,
                "upper": 0.48,
                "stability": 0.8,
                "genes": ["AR", "NKX3-1", "HOXB13"],
                "per_gene": [],
                "concordance": 0.7,
                "detection_fraction": 0.85,
                "support_factor": 0.9,
            },
            "stromal": {"enrichment": 1.8, "n_genes": 141},
            "immune": {"enrichment": 2.2, "n_genes": 141},
            "estimate_purity": 0.35,
            "integration": {
                "source": "signature+lineage",
                "signature_deprioritized": False,
            },
        },
    }
    base.update(overrides)
    return base


def test_comparison_plot_renders_all_methods(tmp_path: Path):
    result = _purity_result()
    out = tmp_path / "cmp.png"
    fig = plot_purity_method_comparison(result, save_to_filename=str(out))
    assert fig is not None
    assert out.exists() and out.stat().st_size > 1000

    # Each method's label should appear as a y-tick label.
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert any("Tumor-specific signature" in label for label in labels)
    assert any("Lineage panel" in label for label in labels)
    assert any("ESTIMATE stromal" in label for label in labels)
    assert any("ESTIMATE immune" in label for label in labels)
    assert any("ESTIMATE combined" in label for label in labels)
    assert "Final estimate" in labels


def test_comparison_plot_includes_decomposition_when_provided(tmp_path: Path):
    class _Decomp:
        purity = 0.34
        cancer_type = "PRAD"
        template = "solid_primary"

    out = tmp_path / "cmp_with_decomp.png"
    fig = plot_purity_method_comparison(
        _purity_result(),
        save_to_filename=str(out),
        decomposition_result=_Decomp(),
    )
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert any("Decomposition" in label for label in labels)
    # Template name surfaces in the label so multiple decomp results
    # are distinguishable.
    assert any("solid_primary" in label for label in labels)


def test_comparison_plot_handles_missing_components(tmp_path: Path):
    """Samples with only signature / only lineage / only ESTIMATE must still
    produce a plot with the available methods and not crash on None."""
    result = _purity_result()
    # Drop lineage completely — simulates a cancer type with no lineage panel.
    result["components"]["lineage"] = {
        "purity": None,
        "lower": None,
        "upper": None,
        "genes": [],
        "per_gene": [],
    }
    result["components"]["signature"]["lower"] = None
    result["components"]["signature"]["upper"] = None

    out = tmp_path / "cmp_sparse.png"
    fig = plot_purity_method_comparison(result, save_to_filename=str(out))
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert any("Tumor-specific signature" in label for label in labels)
    # Lineage panel should NOT appear (no purity) — plot stays honest.
    assert not any(label.startswith("Lineage panel") for label in labels)


def test_comparison_plot_highlights_deprioritized_signature(tmp_path: Path):
    result = _purity_result()
    result["components"]["integration"]["signature_deprioritized"] = True
    out = tmp_path / "cmp_depri.png"
    fig = plot_purity_method_comparison(result, save_to_filename=str(out))
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    sig_label = next(lbl for lbl in labels if "Tumor-specific signature" in lbl)
    assert "deprioritized" in sig_label


def test_comparison_plot_without_overall_still_renders(tmp_path: Path):
    """If the pipeline couldn't produce a final overall estimate, the
    plot should show the available methods without a reference line."""
    result = _purity_result()
    result["overall_estimate"] = None
    result["overall_lower"] = None
    result["overall_upper"] = None
    out = tmp_path / "cmp_no_overall.png"
    fig = plot_purity_method_comparison(result, save_to_filename=str(out))
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert "Final estimate" not in labels
