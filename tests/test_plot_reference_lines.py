"""Tests for the sample-wide p90 TPM reference-line helper."""

import pytest

from trufflepig.plot_reference_lines import (
    add_p90_reference_line,
    compute_sample_p90_tpm,
)


def test_compute_p90_from_dict():
    sample = {f"GENE{i}": float(i) for i in range(200)}
    p90 = compute_sample_p90_tpm(sample)
    # Genes 1..199 are expressed (0 filtered out). 90th percentile of
    # 1..199 is 179.2 (approximately).
    assert p90 is not None
    assert 170 < p90 < 190


def test_compute_p90_returns_none_for_small_sample():
    """Fewer than 50 expressed genes → too noisy for a meaningful
    percentile."""
    sample = {f"G{i}": float(i) for i in range(10)}
    assert compute_sample_p90_tpm(sample) is None


def test_compute_p90_ignores_zero_genes():
    """Zero-TPM genes shouldn't drag the percentile down."""
    sample = {f"G{i}": 0.0 for i in range(200)}
    sample.update({f"E{i}": 50.0 + i for i in range(100)})
    p90 = compute_sample_p90_tpm(sample)
    assert p90 is not None
    assert 130 < p90 < 155  # 90th %ile of 50..149


def test_add_p90_vertical():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 10)
    sample = {f"G{i}": float(i) for i in range(200)}
    p90 = add_p90_reference_line(ax, sample, orientation="vertical")
    assert p90 is not None
    # Should have added one vertical line
    vlines = [ln for ln in ax.get_lines() if ln.get_xdata()[0] == ln.get_xdata()[-1]]
    assert len(vlines) >= 1
    plt.close(fig)


def test_add_p90_horizontal():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 200)
    sample = {f"G{i}": float(i) for i in range(200)}
    p90 = add_p90_reference_line(ax, sample, orientation="horizontal")
    assert p90 is not None
    plt.close(fig)


def test_add_p90_noop_on_empty_sample():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    p90 = add_p90_reference_line(ax, {})
    assert p90 is None
    plt.close(fig)


def test_add_p90_raises_on_bad_orientation():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        add_p90_reference_line(
            ax,
            {f"G{i}": float(i) for i in range(100)},
            orientation="diagonal",
        )
    plt.close(fig)


def test_add_p90_accepts_pandas_series():
    import matplotlib.pyplot as plt
    import pandas as pd

    s = pd.Series({f"G{i}": float(i) for i in range(200)})
    fig, ax = plt.subplots()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 10)
    p90 = add_p90_reference_line(ax, s, orientation="vertical")
    assert p90 is not None
    assert isinstance(p90, (int, float))
    plt.close(fig)


def test_add_p90_custom_percentile():
    sample = {f"G{i}": float(i) for i in range(200)}
    p50 = compute_sample_p90_tpm(sample, percentile=50)
    p99 = compute_sample_p90_tpm(sample, percentile=99)
    # p99 > p90 > p50 by definition when using standard dataset
    assert p50 < p99
