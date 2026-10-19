"""The quality figure must not turn unavailable measurements into reassurance."""
from copy import deepcopy
import math

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pytest

from trufflepig.sample_context import SampleContext, length_pair_display_label, length_pair_index_available, plot_sample_context


@pytest.fixture
def rendered_context(tmp_path, monkeypatch):
    figures = []
    original_close = plt.close

    def capture_and_close(figure=None):
        if hasattr(figure, "axes"):
            figures.append(figure)
        original_close(figure)

    monkeypatch.setattr(plt, "close", capture_and_close)
    monkeypatch.setattr("trufflepig.sample_context._load_artifact_expectations", lambda: {
        (prep, "*", kind): (0.0, 0.01, "synthetic expected range")
        for prep in ("poly_a", "exome_capture")
        for kind in ("histone_cluster", "mitochondrial")
    })

    def render(signals, **kwargs):
        context = SampleContext(signals=signals, **kwargs)
        output = tmp_path / f"qc-{len(figures)}.png"
        plot_sample_context(context, str(output), save_dpi=60)
        assert output.stat().st_size > 1000
        return figures[-1].axes

    return render


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.01, 1.1, False, "0"])
def test_unavailable_diagnostic_fractions_do_not_become_quality_claims(rendered_context, value):
    signals = dict.fromkeys(("histone_fraction", "mt_fraction", "mt_rrna_fraction_of_mt"), value)
    axes = rendered_context(signals, library_prep="poly_a", missing_mt=True)
    assert len(axes) == 1
    text = "\n".join(label.get_text() for label in axes[0].texts)
    assert "Mitochondrial genes missing" in text
    assert "near zero" not in text and "0.000" not in text
    assert not axes[0].patches


def test_absent_measurements_do_not_suggest_capture_preparation(rendered_context):
    axes = rendered_context({}, missing_mt=True)
    text = "\n".join(label.get_text() for label in axes[0].texts)
    assert "Unknown" in text and "Mitochondrial genes missing" in text
    assert "RNA capture library preparation" not in text
    assert "no degradation signal" not in text.casefold()


@pytest.mark.parametrize("concentration", [0.068, 0.25, 0.95])
def test_quality_summary_keeps_concentration_distinct_from_integrity(rendered_context, concentration):
    signals = {"histone_fraction": 0.0, "mt_fraction": 0.001,
               "mt_rrna_fraction_of_mt": 0.0,
               "top_10_share_of_total_tpm": concentration}
    original = deepcopy(signals)
    axes = rendered_context(signals, library_prep="unknown", degradation_index=1.0)
    text = "\n".join(label.get_text() for label in axes[0].texts)
    assert f"Top 10 genes: {concentration:.1%} of RNA" in text
    flattened = " ".join(text.split())
    assert "not a direct measure of RNA integrity" in flattened or "a few genes dominate" in flattened
    assert "near zero" not in text
    assert signals == original
    canvas = FigureCanvasAgg(axes[0].figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    for artist in axes[0].texts:
        bounds = artist.get_window_extent(renderer)
        assert axes[0].figure.bbox.contains(bounds.x0, bounds.y0)
        assert axes[0].figure.bbox.contains(bounds.x1, bounds.y1)


@pytest.mark.parametrize("index", [None, math.nan, math.inf, -1.0, True, "1.0"])
def test_no_evaluable_length_pair_index_is_not_no_degradation(index):
    context = SampleContext(degradation_index=index)
    assert not length_pair_index_available(context)
    assert length_pair_display_label(context) == "unavailable"
    context.degradation_severity = "mild"
    assert length_pair_display_label(context) == "unavailable; orthogonal mild"


def test_an_evaluated_length_pair_index_retains_its_value():
    context = SampleContext(degradation_index=1.0)
    assert length_pair_index_available(context)
    assert length_pair_display_label(context) == "no degradation signal (1.00)"
