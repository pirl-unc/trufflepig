"""Tests for the number-formatting helpers (#120)."""


from trufflepig.format import (
    render_fold,
    render_fraction,
    render_fraction_no_decimal,
    render_score,
    render_tpm,
)


# ── render_fold ─────────────────────────────────────────────────────────


def test_fold_two_decimals_with_multiplication_sign():
    assert render_fold(3.542) == "3.54\u00d7"
    assert render_fold(2.5) == "2.50\u00d7"
    assert render_fold(1.0) == "1.00\u00d7"


def test_fold_handles_infinity():
    assert render_fold(float("inf")) == "\u221e\u00d7"
    assert render_fold(float("-inf")) == "-\u221e\u00d7"


def test_fold_placeholder_for_missing():
    assert render_fold(None) == "\u2014"
    assert render_fold(float("nan")) == "\u2014"


def test_fold_never_renders_lowercase_x():
    # Regression guard: the old renderers sometimes used "x" which
    # looks like a letter next to numbers. Enforce the math sign.
    for value in (0.5, 1.0, 287.3, 1e5):
        rendered = render_fold(value)
        assert "x" not in rendered.lower() or "×" in rendered, rendered
        assert rendered.endswith("\u00d7"), f"missing ×: {rendered!r}"


# ── render_fraction ─────────────────────────────────────────────────────


def test_fraction_autodetects_fraction_vs_percentage():
    # Fraction in [0, 1]
    assert render_fraction(0.643) == "64.3%"
    assert render_fraction(0.5) == "50.0%"
    # Already-computed percentage
    assert render_fraction(64.3) == "64.3%"
    assert render_fraction(5) == "5.0%"


def test_fraction_boundaries():
    assert render_fraction(0.0) == "0.0%"
    assert render_fraction(1.0) == "100.0%"
    # Just over 1.0 is treated as already-a-percentage.
    assert render_fraction(1.001) == "100.1%"


def test_fraction_placeholder_for_missing():
    assert render_fraction(None) == "\u2014"
    assert render_fraction(float("nan")) == "\u2014"


def test_fraction_no_decimal_variant():
    assert render_fraction_no_decimal(0.64) == "64%"
    assert render_fraction_no_decimal(1.0) == "100%"
    assert render_fraction_no_decimal(None) == "\u2014"


# ── render_tpm ──────────────────────────────────────────────────────────


def test_tpm_no_decimals_above_100():
    assert render_tpm(1852.4) == "1852"
    assert render_tpm(142.0) == "142"
    assert render_tpm(100.0) == "100"


def test_tpm_one_decimal_between_1_and_100():
    assert render_tpm(27.8) == "27.8"
    assert render_tpm(4.7) == "4.7"
    assert render_tpm(1.0) == "1.0"


def test_tpm_two_decimals_below_1():
    assert render_tpm(0.12) == "0.12"
    assert render_tpm(0.04) == "0.04"
    assert render_tpm(0.001) == "0.00"


def test_tpm_placeholder_for_missing():
    assert render_tpm(None) == "\u2014"
    assert render_tpm(float("nan")) == "\u2014"


def test_tpm_boundary_behavior():
    # Just below 100 should still be one decimal.
    assert render_tpm(99.9) == "99.9"
    # Just above 100 rounds to integer.
    assert render_tpm(100.4) == "100"


# ── render_score ────────────────────────────────────────────────────────


def test_score_always_three_decimals():
    assert render_score(0.67) == "0.670"
    assert render_score(0.001) == "0.001"
    assert render_score(1.0) == "1.000"
    assert render_score(1) == "1.000"


def test_score_placeholder_for_missing():
    assert render_score(None) == "\u2014"
    assert render_score(float("nan")) == "\u2014"


# ── integration: rendering a realistic row ──────────────────────────────


def test_rendering_a_target_row_is_consistent():
    """A target row touches all four helpers; the rendered cells should
    be consistent with the contract."""
    row = {
        "tpm": 1852.3,
        "fold": 287.34,
        "fraction": 0.83,
        "score": 0.672,
    }
    assert render_tpm(row["tpm"]) == "1852"
    assert render_fold(row["fold"]) == "287.34\u00d7"
    assert render_fraction(row["fraction"]) == "83.0%"
    assert render_score(row["score"]) == "0.672"


def test_none_handling_is_uniform():
    for fn in (
        render_fold,
        render_fraction,
        render_fraction_no_decimal,
        render_tpm,
        render_score,
    ):
        assert fn(None) == "\u2014"
        assert fn(float("nan")) == "\u2014"


def test_custom_placeholder_override():
    assert render_tpm(None, placeholder="n/a") == "n/a"
    assert render_fold(None, placeholder="—") == "—"
