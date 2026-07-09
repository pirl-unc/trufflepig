"""Tests for the therapy-pathway state plot (#136)."""

import matplotlib

matplotlib.use("Agg")

from trufflepig.plot_therapy import plot_therapy_pathway_state
from trufflepig.therapy_response import TherapyAxisScore


def _crpc_scores():
    """Canonical castrate-resistant PRAD pattern."""
    return {
        "AR_signaling": TherapyAxisScore(
            therapy_class="AR_signaling",
            state="down",
            up_geomean_fold=0.33,
            down_geomean_fold=2.54,
            up_genes_measured=10,
            down_genes_measured=4,
        ),
        "NE_differentiation": TherapyAxisScore(
            therapy_class="NE_differentiation",
            state="up",
            up_geomean_fold=2.08,
            down_geomean_fold=None,
            up_genes_measured=6,
        ),
        "EMT": TherapyAxisScore(
            therapy_class="EMT",
            state="up",
            up_geomean_fold=8.95,
            down_geomean_fold=1.12,
            up_genes_measured=5,
            down_genes_measured=3,
        ),
    }


def test_renders_dumbbell_and_caption(tmp_path):
    out = tmp_path / "tps.png"
    fig = plot_therapy_pathway_state(
        therapy_response_scores=_crpc_scores(),
        cancer_code="PRAD",
        disease_state_caption="Castrate-resistant pattern: AR collapsed, NE active.",
        save_to_filename=str(out),
    )
    assert fig is not None
    assert out.exists() and out.stat().st_size > 1000

    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert any("AR signaling" in label for label in labels)
    assert any("NE differentiation" in label for label in labels)
    assert any("EMT" in label for label in labels)
    assert any("expected-up genes" in label for label in labels)
    assert any("expected-down genes" in label for label in labels)
    assert len(labels) == 5


def test_state_tag_matches_direction(tmp_path):
    """AR-down renders 'suppressed'; NE-up renders 'active'."""
    fig = plot_therapy_pathway_state(
        therapy_response_scores=_crpc_scores(),
        cancer_code="PRAD",
        disease_state_caption="",
        save_to_filename=str(tmp_path / "tps2.png"),
    )
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    ar_label = next(lbl for lbl in labels if "AR signaling" in lbl)
    ne_label = next(lbl for lbl in labels if "NE differentiation" in lbl)
    assert "suppressed" in ar_label
    assert "active" in ne_label


def test_empty_scores_returns_none(tmp_path):
    """No measurable axes -> no figure (no empty plot emitted)."""
    out = tmp_path / "tps_empty.png"
    fig = plot_therapy_pathway_state(
        therapy_response_scores={},
        cancer_code="PRAD",
        save_to_filename=str(out),
    )
    assert fig is None
    assert not out.exists()


def test_axes_with_only_up_fold_still_render(tmp_path):
    """Axes without down-panel data (hypoxia, IFN) still draw the
    up-panel point."""
    scores = {
        "hypoxia": TherapyAxisScore(
            therapy_class="hypoxia",
            state="up",
            up_geomean_fold=3.5,
            down_geomean_fold=None,
            up_genes_measured=6,
        ),
    }
    out = tmp_path / "tps_up_only.png"
    fig = plot_therapy_pathway_state(
        therapy_response_scores=scores,
        cancer_code="PRAD",
        save_to_filename=str(out),
    )
    assert fig is not None and out.exists()


def test_mapk_axis_uses_activity_label(tmp_path):
    scores = {
        "MAPK_EGFR_signaling": TherapyAxisScore(
            therapy_class="MAPK_EGFR_signaling",
            state="up",
            up_geomean_fold=4.2,
            down_geomean_fold=None,
            up_genes_measured=10,
        ),
    }
    fig = plot_therapy_pathway_state(
        therapy_response_scores=scores,
        cancer_code="SARC",
        save_to_filename=str(tmp_path / "mapk.png"),
    )
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert any("MAPK / ERK activity" in label for label in labels)


def test_state_ordering_active_suppressed_first(tmp_path):
    """Active / suppressed axes sort before indeterminate — first
    row readers see is the clinically informative one."""
    scores = {
        "near_baseline": TherapyAxisScore(
            therapy_class="near_baseline",
            state="indeterminate",
            up_geomean_fold=1.05,
            down_geomean_fold=0.98,
        ),
        "suppressed_axis": TherapyAxisScore(
            therapy_class="suppressed_axis",
            state="down",
            up_geomean_fold=0.4,
            down_geomean_fold=2.0,
        ),
    }
    fig = plot_therapy_pathway_state(
        therapy_response_scores=scores,
        cancer_code="",
        save_to_filename=str(tmp_path / "tps_order.png"),
    )
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert "suppressed axis" in labels[0]
    assert "near cohort median" in labels[2]


def test_indeterminate_enrichment_is_labeled_mild_not_baseline(tmp_path):
    scores = {
        "angiogenesis": TherapyAxisScore(
            therapy_class="aPD1_exclusion_angiogenesis",
            state="indeterminate",
            up_geomean_fold=1.52,
            up_genes_measured=5,
        ),
    }
    fig = plot_therapy_pathway_state(
        therapy_response_scores=scores,
        cancer_code="READ",
        save_to_filename=str(tmp_path / "tps_mild.png"),
    )
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "mild enrichment" in labels[0]
    assert "near baseline" not in labels[0]
    assert ax.get_title(loc="left") == "Therapy-response pathway RNA — READ"
    assert "selected cancer-cohort median" in ax.get_xlabel()
    legend_labels = [
        text.get_text()
        for legend in [ax.get_legend()]
        if legend is not None
        for text in legend.get_texts()
    ]
    assert "mild enrichment" in legend_labels
