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
def test_unavailable_diagnostic_fractions_are_not_numeric_or_in_range(rendered_context, value):
    signals = dict.fromkeys(("histone_fraction", "mt_fraction", "mt_rrna_fraction_of_mt"), value)
    header, chart = rendered_context(signals, library_prep="poly_a", missing_mt=True)
    labels = [text.get_text() for text in chart.texts]
    assert labels.count("unavailable") == 3
    assert not any("0.000" in text or " ok" in text or "near zero" in text for text in labels)
    assert not chart.patches  # Neither observed bars nor expected-range verdicts exist.
    canvas = FigureCanvasAgg(chart.figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    for label in chart.texts:
        if label.get_text() == "unavailable":
            bounds = label.get_window_extent(renderer)
            assert chart.bbox.y0 <= bounds.y0 < bounds.y1 <= chart.bbox.y1
    assert "Length-pair:   unavailable" in [text.get_text() for text in header.texts]


def test_absent_measurements_do_not_suggest_capture_preparation(rendered_context):
    header, chart = rendered_context({}, missing_mt=True)
    text = "\n".join(label.get_text() for axis in (header, chart) for label in axis.texts)
    assert "Unknown library prep" in text
    assert "support 0%" not in text
    assert "capture" not in text.casefold()
    assert "no degradation signal" not in text
    assert text.count("unavailable") == 4


@pytest.mark.parametrize("prep", ["unknown", "poly_a", "exome_capture"])
def test_observed_zero_is_preserved_without_inventing_library_context(rendered_context, prep):
    signals = {"histone_fraction": 0.0, "mt_fraction": 0.001, "mt_rrna_fraction_of_mt": 0.0,
               "top_10_share_of_total_tpm": 0.25}
    original = deepcopy(signals)
    _, chart = rendered_context(signals, library_prep=prep, degradation_index=1.0)
    text = "\n".join(label.get_text() for label in chart.texts)
    assert text.count("0.000") == 2
    assert "0.001" in text and "0.250" in text
    assert "unavailable" not in text
    assert "Observed diagnostic fractions are near zero" in text
    assert ("inferred RNA capture" in text) == (prep == "exome_capture")
    assert signals == original


def test_an_undefined_ratio_is_not_an_observed_zero(rendered_context):
    _, chart = rendered_context({"histone_fraction": 0.005, "mt_fraction": 0.0,
                                  "mt_rrna_fraction_of_mt": None})
    text = "\n".join(label.get_text() for label in chart.texts)
    assert text.count("unavailable") == 1
    assert text.count("0.000") == 1
    assert "0.005" in text
    assert "near zero" not in text


@pytest.mark.parametrize("prep", ["unknown", "exome_capture"])
@pytest.mark.parametrize("concentration", [0.068, 0.25, 0.95])
def test_low_fraction_explanation_does_not_obscure_measurements_or_labels(
    rendered_context, prep, concentration,
):
    _, chart = rendered_context({
        "histone_fraction": 0.005,
        "mt_fraction": 0.002,
        "mt_rrna_fraction_of_mt": 0.0,
        "top_10_share_of_total_tpm": concentration,
    }, library_prep=prep)
    canvas = FigureCanvasAgg(chart.figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    note, = [label for label in chart.texts if label.get_text().startswith("Observed diagnostic")]
    bounds = note.get_bbox_patch().get_window_extent(renderer)
    for artist in [*chart.texts, *chart.patches, *chart.get_xticklabels(), chart.xaxis.label]:
        if artist is not note:
            assert not bounds.overlaps(artist.get_window_extent(renderer)), artist
    assert chart.figure.bbox.contains(bounds.x0, bounds.y0)
    assert chart.figure.bbox.contains(bounds.x1, bounds.y1)


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
