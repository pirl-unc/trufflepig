# Licensed under the Apache License, Version 2.0

"""Unit tests for tumor_purity internal estimation functions.

Covers: _compile_excluded_gene_matcher (immune-origin bypass),
_combine_purity_estimates (all branches), _signature_conflicts_with_lineage,
_summarize_gene_level_purity edge cases, _lineage_purity_estimates basics.
"""

import numpy as np
import pytest

from trufflepig.tumor_purity import (
    TUMOR_PURITY_PARAMETERS,
    _combine_purity_estimates,
    _compile_excluded_gene_matcher,
    _override_collapsed_signature_purity,
    _signature_conflicts_with_lineage,
    _summarize_gene_level_purity,
    _select_tumor_specific_genes_for_panel,
)
from trufflepig.purity_calibration import _reference_code_candidates


# ── _compile_excluded_gene_matcher ───────────────────────────────────────


def test_excluded_matcher_drops_rearranged_receptor_segments():
    is_excluded = _compile_excluded_gene_matcher()
    assert is_excluded("IGHV3-33")
    assert is_excluded("TRGV9")
    assert is_excluded("IGLC2")
    assert is_excluded("TRDJ1")


def test_excluded_matcher_drops_hla_class_ii():
    is_excluded = _compile_excluded_gene_matcher()
    assert is_excluded("HLA-DRA")
    assert is_excluded("HLA-DRB1")
    assert is_excluded("HLA-DQA1")
    assert is_excluded("HLA-DPB1")


def test_excluded_matcher_preserves_hla_class_i():
    is_excluded = _compile_excluded_gene_matcher()
    assert not is_excluded("HLA-A")
    assert not is_excluded("HLA-B")
    assert not is_excluded("HLA-C")
    assert not is_excluded("HLA-E")


def test_excluded_matcher_drops_ribosomal_and_mt():
    is_excluded = _compile_excluded_gene_matcher()
    assert is_excluded("RPL13A")
    assert is_excluded("RPS27")
    assert is_excluded("MT-CO1")
    assert is_excluded("MT-ND4")


def test_excluded_matcher_preserves_non_receptor_ig_tr_genes():
    """IGHMBP2, TRAF3, TRADD, TRAP1, TRAK1 should NOT be excluded."""
    is_excluded = _compile_excluded_gene_matcher()
    assert not is_excluded("IGHMBP2")
    assert not is_excluded("TRAF3")
    assert not is_excluded("TRADD")
    assert not is_excluded("TRAP1")
    assert not is_excluded("TRAK1")


def test_excluded_matcher_handles_empty_and_none():
    is_excluded = _compile_excluded_gene_matcher()
    assert is_excluded("")
    assert is_excluded(None)


def test_dlbc_panel_bypasses_exclusion():
    """DLBC is immune-origin: panel should include IG/HLA-D genes."""
    genes = _select_tumor_specific_genes_for_panel("DLBC", n=30, exclude_lineage=False)
    # At least some IG or HLA-D genes should survive
    has_ig_or_hla_d = any(
        g.startswith("IGH")
        or g.startswith("IGL")
        or g.startswith("IGK")
        or g.startswith("HLA-D")
        for g in genes
    )
    assert has_ig_or_hla_d, (
        f"DLBC panel should include IG/HLA-D genes, got: {genes[:10]}"
    )


def test_laml_panel_bypasses_exclusion():
    """LAML is immune-origin: panel should include IG/HLA-D genes
    that would be excluded for non-immune-origin types."""
    genes = _select_tumor_specific_genes_for_panel("LAML", n=30, exclude_lineage=False)
    # LAML should retain rearranged-receptor genes (IG segments) that
    # the default exclusion regexes would filter out
    has_ig_or_hla_d = any(
        g.startswith("IGH")
        or g.startswith("IGL")
        or g.startswith("IGK")
        or g.startswith("HLA-D")
        for g in genes
    )
    assert has_ig_or_hla_d, (
        f"LAML panel should include IG/HLA-D genes, got: {genes[:10]}"
    )


def test_thym_panel_bypasses_exclusion():
    """THYM is immune-origin: panel should include TR receptor genes
    that would be excluded for non-immune-origin types."""
    genes = _select_tumor_specific_genes_for_panel("THYM", n=30, exclude_lineage=False)
    # THYM should retain rearranged T-cell receptor genes (TRB/TRD/TRG segments)
    has_tr = any(
        g.startswith("TRB")
        or g.startswith("TRD")
        or g.startswith("TRG")
        or g.startswith("TRA")
        and len(g) > 4
        and g[3] in "VDJC"
        for g in genes
    )
    assert has_tr, f"THYM panel should include TR receptor genes, got: {genes[:10]}"


def test_non_immune_origin_panel_excludes_ig_hla_d():
    """A non-immune-origin type like PRAD should exclude IG/HLA-D."""
    genes = _select_tumor_specific_genes_for_panel("PRAD", n=30, exclude_lineage=False)
    bad = [g for g in genes if g.startswith("HLA-D") or g.startswith("IGHV")]
    assert not bad, f"PRAD panel should not include {bad}"


# ── _combine_purity_estimates ────────────────────────────────────────────


def test_combine_purity_both_signals_geometric_anchor():
    """When both signature and lineage exist, tumor anchor is a geometric mean."""
    overall, lo, hi = _combine_purity_estimates(
        sig_purity=0.60,
        sig_lower=0.50,
        sig_upper=0.70,
        estimate_purity=0.55,
        lineage_purity=0.65,
        lineage_lower=0.55,
        lineage_upper=0.75,
        sig_stability=0.8,
    )
    assert overall is not None
    assert 0.0 < lo <= overall <= hi <= 1.0


def test_combine_purity_lineage_only():
    """When signature is None, should use lineage directly."""
    overall, lo, hi = _combine_purity_estimates(
        sig_purity=None,
        sig_lower=None,
        sig_upper=None,
        estimate_purity=0.50,
        lineage_purity=0.70,
        lineage_lower=0.60,
        lineage_upper=0.80,
    )
    assert overall is not None
    # Lineage-only path: combined with ESTIMATE
    assert 0.50 <= overall <= 0.80


def test_combine_purity_signature_only():
    """When lineage is None, should use signature with ESTIMATE floor."""
    overall, lo, hi = _combine_purity_estimates(
        sig_purity=0.40,
        sig_lower=0.30,
        sig_upper=0.50,
        estimate_purity=0.35,
        lineage_purity=None,
        lineage_lower=None,
        lineage_upper=None,
    )
    assert overall is not None
    assert 0.0 < overall <= 1.0


def test_combine_purity_both_none_returns_none():
    """When both signature and lineage are None, returns None."""
    result = _combine_purity_estimates(
        sig_purity=None,
        sig_lower=None,
        sig_upper=None,
        estimate_purity=None,
        lineage_purity=None,
        lineage_lower=None,
        lineage_upper=None,
    )
    assert result == (None, None, None)


def test_combine_purity_estimate_only():
    """When only ESTIMATE is available, returns it directly."""
    overall, lo, hi = _combine_purity_estimates(
        sig_purity=None,
        sig_lower=None,
        sig_upper=None,
        estimate_purity=0.60,
        lineage_purity=None,
        lineage_lower=None,
        lineage_upper=None,
    )
    assert overall == pytest.approx(0.60, abs=0.01)


def test_combine_purity_zero_estimate_with_lineage_is_preserved():
    """When ESTIMATE is 0 but lineage + sig exist, tumor anchor wins."""
    overall, lo, hi = _combine_purity_estimates(
        sig_purity=0.65,
        sig_lower=0.55,
        sig_upper=0.75,
        estimate_purity=0.0,
        lineage_purity=0.60,
        lineage_lower=0.50,
        lineage_upper=0.70,
        sig_stability=0.7,
    )
    # Should not collapse to 0 — the tumor anchor dominates
    assert overall > 0.30


def test_combine_purity_conflict_deprioritizes_signature():
    """When sig is very low but lineage is high, lineage wins."""
    overall, _, _ = _combine_purity_estimates(
        sig_purity=0.10,
        sig_lower=0.05,
        sig_upper=0.15,
        estimate_purity=0.50,
        lineage_purity=0.70,
        lineage_lower=0.60,
        lineage_upper=0.80,
        sig_stability=0.3,  # low stability
    )
    # Should be closer to lineage than signature
    assert overall > 0.40


def test_combine_purity_ci_symmetric_when_upper_candidates_depressed():
    """#101: low-purity samples had `max(upper_candidates) == overall` because
    the upper quartiles were themselves low, producing a one-sided CI like
    7–16% around 16%. The fix mirrors the wider half of the interval so the
    upper bound reflects the observed spread, not the floor of the sample."""
    overall, lo, hi = _combine_purity_estimates(
        sig_purity=0.16,
        sig_lower=0.07,
        sig_upper=0.16,
        estimate_purity=0.10,
        lineage_purity=0.16,
        lineage_lower=0.08,
        lineage_upper=0.16,
        sig_stability=0.5,
    )
    assert overall is not None
    lower_span = overall - lo
    upper_span = hi - overall
    assert upper_span >= lower_span - 1e-9, (
        f"upper span {upper_span:.4f} should not be smaller than lower span "
        f"{lower_span:.4f} (got CI {lo:.3f}–{hi:.3f} around {overall:.3f})"
    )
    # Upper bound must meaningfully exceed the point estimate in this scenario
    assert hi > overall + 0.01


# ── _signature_conflicts_with_lineage ────────────────────────────────────


def test_signature_conflict_returns_false_when_missing():
    assert not _signature_conflicts_with_lineage(None, 0.7, 0.8)
    assert not _signature_conflicts_with_lineage(0.5, None, 0.8)


def test_signature_conflict_detects_low_sig_high_lineage():
    """Low signature + low stability + high lineage → conflict."""
    params = TUMOR_PURITY_PARAMETERS["purity_combination"]
    # sig well below lineage * ratio, stability below floor
    result = _signature_conflicts_with_lineage(
        sig_purity=0.10,
        lineage_purity=0.70,
        sig_stability=0.2,  # below signature_stability_min (0.45)
    )
    assert result is True


def test_signature_conflict_no_conflict_when_stable():
    """High stability → no conflict even if sig < lineage."""
    result = _signature_conflicts_with_lineage(
        sig_purity=0.50,
        lineage_purity=0.70,
        sig_stability=0.9,  # very stable
    )
    assert result is False


# ── _summarize_gene_level_purity edge cases ──────────────────────────────


def test_summarize_empty_returns_all_none():
    assert _summarize_gene_level_purity([]) == (None, None, None, None)


def test_summarize_all_zeros_returns_all_none():
    """Zeros are filtered out (only p > 0 kept)."""
    assert _summarize_gene_level_purity([0.0, 0.0, 0.0]) == (None, None, None, None)


def test_summarize_single_value():
    overall, lo, hi, stability = _summarize_gene_level_purity([0.5])
    assert overall == pytest.approx(0.5)
    assert lo is not None
    assert hi is not None


def test_summarize_two_values():
    overall, lo, hi, stability = _summarize_gene_level_purity([0.3, 0.7])
    assert lo is not None
    assert hi is not None
    assert lo <= overall <= hi


def test_summarize_winsorized_clips_outliers():
    """With >= 4 values, winsorized_median clips to IQR before taking median."""
    purities = [0.01, 0.40, 0.50, 0.55, 0.60, 0.95]
    overall, lo, hi, stability = _summarize_gene_level_purity(
        purities, strategy="winsorized_median"
    )
    # The extreme 0.01 and 0.95 should be clipped
    assert 0.35 < overall < 0.65


def test_summarize_upper_half_ignores_bottom():
    purities = [0.05, 0.10, 0.15, 0.60, 0.65, 0.70]
    overall, _, _, _ = _summarize_gene_level_purity(purities, strategy="upper_half")
    # Upper half: [0.15, 0.60, 0.65, 0.70] or [0.60, 0.65, 0.70]
    assert overall >= 0.50


def test_summarize_stability_increases_with_tighter_distribution():
    tight = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55]
    wide = [0.10, 0.30, 0.50, 0.70, 0.90, 0.95]
    _, _, _, stab_tight = _summarize_gene_level_purity(tight)
    _, _, _, stab_wide = _summarize_gene_level_purity(wide)
    assert stab_tight > stab_wide


def test_override_collapsed_signature_purity_consensus():
    """A wrong-type signature collapse (≈0) is overridden by the ESTIMATE + decomposition-residual
    consensus (the hcc1395 cell-line case: breast line miscalled HNSC → signature silent)."""
    overall, lower, upper, source = _override_collapsed_signature_purity(
        0.075, 0.0, 0.15, "signature",
        sig_purity=0.006, lineage_purity=None, estimate_purity=0.93,
        decomposition={"residual_fraction": 0.77})
    assert source == "estimate+decomposition"
    assert 0.7 < overall < 0.95          # consensus of 0.93 and 0.77, not the 0.075 signature drag


def test_override_collapsed_signature_noop_during_ranking():
    """No decomposition (the candidate-ranking path) → strict no-op, so a low signature stays the
    discriminator between candidate types."""
    result = _override_collapsed_signature_purity(
        0.075, 0.0, 0.15, "signature",
        sig_purity=0.006, lineage_purity=None, estimate_purity=0.93, decomposition=None)
    assert result == (0.075, 0.0, 0.15, "signature")


def test_override_collapsed_signature_noop_when_residual_uncorroborated():
    """Only ONE type-agnostic signal high (ESTIMATE) while the residual is low → no override (the
    low-coverage guard: both ESTIMATE and residual must corroborate)."""
    result = _override_collapsed_signature_purity(
        0.075, 0.0, 0.15, "signature",
        sig_purity=0.006, lineage_purity=None, estimate_purity=0.93,
        decomposition={"residual_fraction": 0.2})
    assert result == (0.075, 0.0, 0.15, "signature")


def test_override_collapsed_signature_noop_with_lineage_anchor():
    """A present lineage anchor means the signature wasn't the sole driver → no override."""
    result = _override_collapsed_signature_purity(
        0.4, 0.2, 0.6, "signature",
        sig_purity=0.006, lineage_purity=0.5, estimate_purity=0.93,
        decomposition={"residual_fraction": 0.8})
    assert result == (0.4, 0.2, 0.6, "signature")


def test_aneuploidy_reference_parent_fallback_candidates():
    """A sarcoma subtype without its own pan-cancer column falls back to the SARC parent."""
    candidates = _reference_code_candidates("SARC_DSRCT")
    assert candidates[0] == "SARC_DSRCT"
    assert "SARC" in candidates
