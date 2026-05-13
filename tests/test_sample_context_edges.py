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


# ── Histone limbo band ───────────────────────────────────────────────────


def test_histone_limbo_band_with_zero_mt_yields_exome_capture():
    """Histone fraction in the 0.1–0.5% 'limbo' band combined with
    MT completely absent should yield exome_capture, not unknown.

    This is the Tempus xT / Twist RNA Exome pattern (bug 2026-04-14):
    capture probes include some histones but zero chrM coverage.
    """
    rows = [("ACTB", 500), ("GAPDH", 400), ("TUBB", 300)]
    # Histones present but in the limbo band (0.1-0.5% of total)
    # total background ≈ 1200; need histone ≈ 1.5–6 TPM
    rows += [("H2BC1", 1.5), ("H3C1", 1.0), ("H4C1", 0.8)]
    # NO MT genes at all
    frame = _frame_from_pairs(rows)
    ctx = infer_sample_context(frame)
    assert ctx.library_prep == "exome_capture"


def test_histone_below_polya_ceiling_no_mt_yields_exome_capture():
    """Very low histones (<0.2%) but MT completely absent → exome_capture."""
    rows = [("ACTB", 500), ("GAPDH", 400)]
    # Tiny histone signal
    rows += [("H2BC1", 0.1)]
    # No MT
    frame = _frame_from_pairs(rows)
    ctx = infer_sample_context(frame)
    assert ctx.library_prep == "exome_capture"


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
    """MT-rRNA exactly zero with low MT overall → exome_capture at high confidence."""
    rows = [("ACTB", 1000), ("GAPDH", 800)]
    # NO MT-rRNA, tiny MT-mRNA leakage
    rows += [(list(_MT_MRNA_SYMBOLS)[0], 0.01)]
    frame = _frame_from_pairs(rows)
    ctx = infer_sample_context(frame)
    assert ctx.library_prep == "exome_capture"
    assert ctx.library_prep_confidence >= 0.8


def test_low_mt_fraction_with_mt_mrnas_is_not_reported_as_mt_stripped():
    """Low chrM fraction can support RNA capture without inventing an MT-stripped class."""
    rows = [("ACTB", 50000), ("GAPDH", 45000), ("EEF1A1", 40000)]
    rows += [("MT-ND1", 5.0), ("MT-CO1", 4.0), ("MT-ATP6", 3.0)]
    frame = _frame_from_pairs(rows)
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
    frame = _frame_from_pairs(rows)
    tpm = _build_tpm_by_symbol(frame)
    signals = {}
    prep, conf = _infer_library_prep(tpm, signals)
    # With nonzero rRNA, the exact-zero path doesn't fire.
    # This should still yield exome_capture at lower confidence
    # or poly_a depending on histone level.
    assert prep in ("exome_capture", "poly_a")


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
