# Licensed under the Apache License, Version 2.0

"""Edge-case tests for sample_context heuristics.

Covers: histone limbo band, MT-rRNA boundary for exome capture,
expression distribution metrics, preservation refinement.
"""

import pandas as pd

from trufflepig.sample_context import (
    SampleContext,
    infer_sample_context,
    _MT_MRNA_SYMBOLS,
    _MT_RRNA_SYMBOLS,
    _build_tpm_by_symbol,
    _infer_library_prep,
    _summarise_expression_distribution,
)


def _frame_from_pairs(pairs):
    """Build a minimal (gene_symbol, TPM) expression frame."""
    return pd.DataFrame([{"gene_symbol": s, "TPM": float(t)} for s, t in pairs])


def _broad_profile(rows):
    return rows + [(f"PROFILE{i}", 2.0) for i in range(4000)]


# ── Histone limbo band ───────────────────────────────────────────────────


def test_histone_limbo_band_with_absent_mt_is_unresolved():
    """Low histones cannot turn unrepresented MT genes into capture evidence."""
    rows = [("ACTB", 500), ("GAPDH", 400), ("TUBB", 300)]
    # Histones present but in the limbo band (0.1-0.5% of total)
    # total background ≈ 1200; need histone ≈ 1.5–6 TPM
    rows += [("H2BC1", 1.5), ("H3C1", 1.0), ("H4C1", 0.8)]
    # NO MT genes at all
    frame = _frame_from_pairs(rows)
    ctx = infer_sample_context(frame)
    assert ctx.library_prep == "unknown"
    assert ctx.signals["mt_fraction"] is None
    assert ctx.signals["histone_genes_valid"] == 3


def test_histone_below_polya_ceiling_no_mt_is_unresolved():
    """One low histone observation and no MT rows cannot identify capture."""
    rows = [("ACTB", 500), ("GAPDH", 400)]
    # Tiny histone signal
    rows += [("H2BC1", 0.1)]
    # No MT
    frame = _frame_from_pairs(rows)
    ctx = infer_sample_context(frame)
    assert ctx.library_prep == "unknown"


def test_histone_limbo_with_mt_present_not_exome():
    """Histones in limbo band BUT MT present → should NOT be exome_capture."""
    rows = [("ACTB", 500), ("GAPDH", 400), ("TUBB", 300)]
    rows += [("H2BC1", 1.5), ("H3C1", 1.0)]
    # MT present — rules out capture kit
    rows += [(sym, 30) for sym in list(_MT_MRNA_SYMBOLS)[:5]]
    frame = _frame_from_pairs(rows)
    ctx = infer_sample_context(frame)
    assert ctx.library_prep != "exome_capture"


# ── MT-rRNA zero boundary ────────────────────────────────────────────────


def test_mt_rrna_exactly_zero_high_confidence_exome():
    """Measured zeros across the MT panel can support a capture call."""
    rows = [("ACTB", 1000), ("GAPDH", 800)]
    # All MT features were quantified; only one mRNA has tiny leakage.
    rows += [(sym, 0.01 if sym == "MT-CO1" else 0.0)
             for sym in (_MT_MRNA_SYMBOLS | _MT_RRNA_SYMBOLS)]
    rows += [(sym, 0) for sym in ("H2BC1", "H2AC1", "H3C1")]
    frame = _frame_from_pairs(_broad_profile(rows))
    ctx = infer_sample_context(frame)
    assert ctx.library_prep == "exome_capture"
    assert ctx.library_prep_confidence >= 0.8
    assert ctx.signals["mt_genes_present"] == ctx.signals["mt_genes_total"]
    assert ctx.signals["mt_genes_detected"] == 1


def test_low_mt_fraction_with_mt_mrnas_is_not_reported_as_mt_stripped():
    """Low chrM fraction can support RNA capture without inventing an MT-stripped class."""
    rows = [("ACTB", 50000), ("GAPDH", 45000), ("EEF1A1", 40000)]
    rows += [("MT-ND1", 5.0), ("MT-CO1", 4.0), ("MT-ATP6", 3.0)]
    rows += [(sym, 0) for sym in (_MT_MRNA_SYMBOLS | _MT_RRNA_SYMBOLS)
             if sym not in {"MT-ND1", "MT-CO1", "MT-ATP6"}]
    rows += [(sym, 0) for sym in ("H2BC1", "H2AC1", "H3C1")]
    frame = _frame_from_pairs(_broad_profile(rows))
    ctx = infer_sample_context(frame)
    assert ctx.library_prep == "exome_capture"
    assert ctx.signals["mt_genes_detected"] >= 3
    assert not ctx.missing_mt
    assert "stripped" not in ctx.summary_line().lower()
    assert "RNA hybrid-capture" in ctx.summary_line()


def test_mt_rrna_near_zero_lower_confidence():
    """MT fraction < suspicious floor but rRNA NOT exactly zero
    → exome_capture at moderate confidence (0.6)."""
    rows = [("ACTB", 50000), ("GAPDH", 49900)]
    # Tiny MT rRNA — nonzero but fraction < suspicious floor
    rows += [
        (
            _MT_RRNA_SYMBOLS[0]
            if isinstance(_MT_RRNA_SYMBOLS, list)
            else list(_MT_RRNA_SYMBOLS)[0],
            0.01,
        )
    ]
    rows.append(("MT-CO1", 0.01))
    rows += [(sym, 0) for sym in (_MT_MRNA_SYMBOLS | _MT_RRNA_SYMBOLS)
             if sym not in {"MT-RNR1", "MT-CO1"}]
    rows += [(sym, 0) for sym in ("H2BC1", "H2AC1", "H3C1")]
    frame = _frame_from_pairs(_broad_profile(rows))
    tpm = _build_tpm_by_symbol(frame)
    signals = {}
    prep, conf = _infer_library_prep(tpm, signals)
    # With nonzero rRNA, the exact-zero path doesn't fire.
    # This should still yield exome_capture at lower confidence
    # or poly_a depending on histone level.
    assert prep == "exome_capture"
    assert conf == 0.6


def test_mt_feature_coverage_separates_absent_invalid_zero_and_partial():
    base = [("ACTB", 500), ("GAPDH", 400), ("TUBB", 300)]
    cases = (
        ([], 0, 0, None, "rows absent"),
        ([(sym, float("nan")) for sym in _MT_MRNA_SYMBOLS | _MT_RRNA_SYMBOLS],
         15, 0, None, "TPM values are unavailable"),
        ([(sym, 0) for sym in _MT_MRNA_SYMBOLS | _MT_RRNA_SYMBOLS],
         15, 15, 0.0, "none detected above zero"),
        ([("MT-CO1", 0), ("MT-RNR1", 0)],
         2, 2, 0.0, "none detected above zero"),
    )
    for mt_rows, present, valid, fraction, flag in cases:
        for broad in (False, True):
            rows = _broad_profile(base + mt_rows) if broad else base + mt_rows
            ctx = infer_sample_context(_frame_from_pairs(rows))
            assert ctx.signals["mt_genes_present"] == present
            assert ctx.signals["mt_genes_valid"] == valid
            assert ctx.signals["mt_genes_detected"] == 0
            assert ctx.signals["mt_fraction"] == fraction
            assert any(flag in text for text in ctx.flags)
            assert ctx.library_prep == "unknown"


def test_high_histone_mt_stripped_profile_does_not_imply_capture():
    rows = _broad_profile([("ACTB", 500), ("GAPDH", 400)])
    rows += [(sym, 0.01 if sym == "MT-CO1" else 0.0)
             for sym in (_MT_MRNA_SYMBOLS | _MT_RRNA_SYMBOLS)]
    rows += [(sym, 200) for sym in ("H2BC1", "H2AC1", "H3C1")]
    ctx = infer_sample_context(_frame_from_pairs(rows))
    assert ctx.library_prep == "unknown"


def test_fully_measured_but_zero_mt_panel_stays_unresolved():
    rows = _broad_profile([("ACTB", 500), ("GAPDH", 400)])
    rows += [(sym, 0) for sym in (_MT_MRNA_SYMBOLS | _MT_RRNA_SYMBOLS)]
    rows += [(sym, 0) for sym in ("H2BC1", "H2AC1", "H3C1")]
    ctx = infer_sample_context(_frame_from_pairs(rows))
    assert ctx.library_prep == "unknown"
    assert ctx.signals["mt_genes_valid"] == 15
    assert ctx.signals["mt_fraction"] == 0.0
    assert any("none detected above zero" in f for f in ctx.flags)


def test_invalid_mt_values_do_not_count_as_measured_zero():
    rows = _broad_profile([("ACTB", 500), ("GAPDH", 400)])
    rows += [("MT-RNR1", float("nan")), ("MT-RNR2", float("inf")),
             ("MT-CO1", -1.0)]
    ctx = infer_sample_context(_frame_from_pairs(rows))
    assert ctx.library_prep == "unknown"
    assert ctx.signals["mt_genes_present"] == 3
    assert ctx.signals["mt_genes_valid"] == 0
    assert ctx.signals["mt_fraction"] is None


def test_partially_represented_positive_mt_panel_stays_unresolved():
    rows = _broad_profile([("ACTB", 500), ("GAPDH", 400)])
    rows += [("MT-CO1", 0.01), ("MT-CO2", 0.01), ("MT-ND1", 0.01)]
    ctx = infer_sample_context(_frame_from_pairs(rows))
    assert ctx.library_prep == "unknown"
    assert ctx.missing_mt
    assert ctx.signals["mt_genes_present"] == 3
    assert ctx.signals["mt_genes_detected"] == 3
    assert any("Only 3/15 mitochondrial gene TPM values are valid" in f for f in ctx.flags)


# ── Expression distribution metrics ──────────────────────────────────────


def test_expression_distribution_populates_detection_counts():
    tpm = {"GENE_A": 100.0, "GENE_B": 5.0, "GENE_C": 0.3, "GENE_D": 0.0}
    signals = {}
    _summarise_expression_distribution(tpm, signals)
    assert signals["genes_detected_above_1_tpm"] == 2  # A, B
    assert signals["genes_detected_above_0p5_tpm"] == 2
    assert signals["genes_detected_above_10_tpm"] == 1  # A only
    assert signals["n_genes_with_any_expression"] == 3  # A, B, C (not D)


def test_expression_distribution_log2_median():
    tpm = {f"GENE_{i}": float(2**i) for i in range(1, 11)}  # 2, 4, ..., 1024
    signals = {}
    _summarise_expression_distribution(tpm, signals)
    assert "log2_tpm_median" in signals
    # Median of log2(x+1) for expressed genes (>1 TPM)
    assert signals["log2_tpm_median"] > 0


def test_expression_distribution_empty():
    signals = {}
    _summarise_expression_distribution({}, signals)
    assert signals["n_genes_with_any_expression"] == 0
    assert "log2_tpm_median" not in signals


def test_expression_distribution_panel_detection():
    """A narrow panel with few genes and concentrated TPM should flag."""
    # Simulate a 200-gene targeted panel
    tpm = {f"GENE_{i}": 100.0 for i in range(200)}
    signals = {}
    _summarise_expression_distribution(tpm, signals)
    # With only 200 genes above 1 TPM and top-2000 being 100%,
    # likely_targeted_panel should be True
    assert signals["likely_targeted_panel"] is True


def test_expression_distribution_flags_extreme_concentration_without_panel_call():
    """Many detected genes can still have an abnormal dominant-transcript tail."""
    tpm = {f"GENE_{i}": 10.0 for i in range(12_000)}
    tpm["RNA5S_DOMINANT"] = 520_000.0
    signals = {}
    _summarise_expression_distribution(tpm, signals)

    assert signals["genes_detected_above_1_tpm"] > 4000
    assert signals["likely_targeted_panel"] is False
    assert signals["expression_concentration_level"] == "extreme"
    assert signals["dominant_expression_genes"][0]["gene"] == "RNA5S_DOMINANT"


def test_expression_distribution_whole_transcriptome_not_flagged():
    """A normal whole-transcriptome sample should NOT flag as panel."""
    import numpy as np

    rng = np.random.default_rng(42)
    # 15000 expressed genes with a realistic log-normal distribution
    values = rng.lognormal(2.0, 2.0, 15000)
    tpm = {f"GENE_{i}": float(v) for i, v in enumerate(values)}
    signals = {}
    _summarise_expression_distribution(tpm, signals)
    assert signals["likely_targeted_panel"] is False


# ── SampleContext.purity_ci_widening_factor ───────────────────────────────


def test_purity_ci_widening_factor_is_one_for_none():
    ctx = SampleContext(
        library_prep="poly_a",
        library_prep_confidence=0.8,
        preservation="fresh_frozen",
        degradation_severity="none",
        degradation_index=None,
    )
    assert ctx.purity_ci_widening_factor() == 1.0


def test_purity_ci_widening_factor_increases_with_severity():
    mild = SampleContext(
        library_prep="poly_a",
        library_prep_confidence=0.8,
        preservation="ffpe",
        degradation_severity="moderate",
        degradation_index=0.25,
    )
    severe = SampleContext(
        library_prep="poly_a",
        library_prep_confidence=0.8,
        preservation="ffpe",
        degradation_severity="severe",
        degradation_index=0.15,
    )
    assert mild.purity_ci_widening_factor() >= 1.0
    assert severe.purity_ci_widening_factor() > mild.purity_ci_widening_factor()
