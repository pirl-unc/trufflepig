import pandas as pd
import pytest

from pirlygenes.gene_sets_cancer import degradation_gene_pairs
from trufflepig.reference import pan_cancer_expression
from trufflepig.sample_context import (
    SampleContext,
    infer_sample_context,
    library_prep_clause,
    _HISTONE_SYMBOL_PREFIXES,
    _MT_MRNA_SYMBOLS,
    _MT_RRNA_SYMBOLS,
    _build_tpm_by_symbol,
)


def _frame_from_pairs(pairs):
    """Build a minimal (gene_symbol, TPM) expression frame from [(sym, tpm)]."""
    return pd.DataFrame([{"gene_symbol": s, "TPM": float(t)} for s, t in pairs])


def _pan_cancer_ntpm_sample(tissue):
    """Synthetic sample equal to a reference normal tissue column."""
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{tissue}_nTPM"].astype(float),
        }
    )


def _fresh_degradation_pair_expectations():
    """Return list of (short, long, expected) pairs whose ``expected`` ratio
    is used as the reference 'fresh' value."""
    return list(degradation_gene_pairs())


# ── Library-prep inference ────────────────────────────────────────────────


def test_library_prep_clause_does_not_duplicate_library_word():
    assert library_prep_clause("poly_a") == "poly-A capture library"
    assert library_prep_clause("unknown") == "unknown library prep"


def test_library_prep_polya_signature_is_detected():
    """Histones absent, MT rRNAs absent, MT mRNAs present → poly-A."""
    # Background expression to make histone + MT fractions small/large
    # relative to total.
    rows = [("ACTB", 500), ("GAPDH", 400), ("TUBB", 300)]
    # MT mRNAs present (typical poly-A capture keeps them)
    rows += [(sym, 50) for sym in _MT_MRNA_SYMBOLS]
    # MT rRNAs absent. Histones absent.
    frame = _frame_from_pairs(rows)

    ctx = infer_sample_context(frame)
    assert ctx.library_prep == "poly_a"
    assert ctx.library_prep_confidence >= 0.5


def test_library_prep_total_rna_signature_is_detected():
    """Histones high, MT rRNAs dominant → total RNA."""
    rows = [("ACTB", 500), ("GAPDH", 400), ("TUBB", 300)]
    # High histone mass — use modern HGNC names so prefix detection
    # catches H2AC* / H2BC* / H3C* / H4C*.
    rows += [
        ("H2BC1", 80),
        ("H2BC3", 70),
        ("H2BC4", 65),
        ("H2AC1", 60),
        ("H3C1", 55),
        ("H4C1", 50),
    ]
    # MT rRNAs dominate MT signal (characteristic of total RNA libraries)
    rows += [(sym, 200) for sym in _MT_RRNA_SYMBOLS]
    rows += [(sym, 20) for sym in list(_MT_MRNA_SYMBOLS)[:5]]
    frame = _frame_from_pairs(rows)

    ctx = infer_sample_context(frame)
    assert ctx.library_prep == "total_rna"
    assert ctx.library_prep_confidence >= 0.7


def test_library_prep_ribo_depleted_signature_is_detected():
    """Histones present, MT rRNAs depleted → ribo-depleted."""
    rows = [("ACTB", 500), ("GAPDH", 400), ("TUBB", 300)]
    rows += [
        ("H2BC1", 80),
        ("H2BC3", 70),
        ("H2AC1", 60),
        ("H3C1", 55),
        ("H4C1", 50),
    ]
    # MT rRNAs absent or very low; MT mRNAs present
    rows += [(sym, 40) for sym in _MT_MRNA_SYMBOLS]
    frame = _frame_from_pairs(rows)

    ctx = infer_sample_context(frame)
    assert ctx.library_prep == "ribo_depleted"


def test_library_prep_exome_capture_signature_is_detected():
    """MT absent AND histones absent → exome capture."""
    rows = [("ACTB", 500), ("GAPDH", 400), ("TUBB", 300), ("TP53", 20), ("BRCA1", 10)]
    # Nothing else — no MT, no histones.
    frame = _frame_from_pairs(rows)
    ctx = infer_sample_context(frame)
    assert ctx.library_prep == "exome_capture"


def test_build_tpm_by_symbol_uses_gene_column_without_fallback(monkeypatch):
    """Aggregated pirlygenes outputs use ``gene`` rather than
    ``gene_symbol``; sample-context inference should treat that as a
    direct symbol column and avoid the slower tumor-purity fallback.
    """
    frame = pd.DataFrame(
        {
            "gene": ["TP53", "TP53", "EGFR"],
            "TPM": [1.0, 3.0, 2.5],
        }
    )

    def _should_not_run(_df):
        raise AssertionError("tumor-purity fallback should not be used")

    monkeypatch.setattr(
        "trufflepig.common.build_sample_tpm_by_symbol",
        _should_not_run,
    )

    out = _build_tpm_by_symbol(frame)
    assert out == {"TP53": 3.0, "EGFR": 2.5}


# ── Preservation + degradation ────────────────────────────────────────────


def test_fresh_sample_not_flagged_as_degraded():
    """A synthetic sample equal to a reference tissue column has
    degradation-pair ratios equal to the expected values, so the
    inference should land on fresh_frozen."""
    df = _pan_cancer_ntpm_sample("liver")
    ctx = infer_sample_context(df)
    assert ctx.preservation == "fresh_frozen"
    assert ctx.degradation_severity == "none"
    assert ctx.degradation_index is not None
    assert 0.7 < ctx.degradation_index < 1.5


def test_ffpe_sample_triggers_ffpe_preservation():
    """Construct a sample where long transcripts are systematically
    depressed relative to their matched short transcripts — the
    canonical FFPE signature.
    """
    import random

    random.seed(0)
    rows = []
    # Background
    rows += [("ACTB", 500), ("GAPDH", 400), ("TUBB", 300)]
    # MT genes present (poly-A-like) for library_prep side signal
    rows += [(sym, 30) for sym in _MT_MRNA_SYMBOLS]
    # For each degradation pair, short gets normal TPM but long gets
    # 10x lower than the fresh expected ratio would predict.
    for short_sym, long_sym, expected in _fresh_degradation_pair_expectations()[:18]:
        short_tpm = 50.0 + random.random() * 20
        long_tpm = short_tpm * float(expected) * 0.10  # 10× depletion
        rows.append((short_sym, short_tpm))
        rows.append((long_sym, long_tpm))
    frame = _frame_from_pairs(rows)

    ctx = infer_sample_context(frame)
    assert ctx.preservation == "ffpe"
    assert ctx.degradation_severity in ("moderate", "severe")
    assert ctx.degradation_index is not None and ctx.degradation_index < 0.3
    assert ctx.long_transcript_weight_factor() < 1.0
    assert ctx.purity_ci_widening_factor() > 1.0


# ── Missing-MT ────────────────────────────────────────────────────────────


def test_missing_mt_flag_set_when_mt_genes_absent():
    """Expression frames without any MT gene should set missing_mt."""
    rows = [("ACTB", 500), ("GAPDH", 400), ("TUBB", 300)]
    frame = _frame_from_pairs(rows)
    ctx = infer_sample_context(frame)
    assert ctx.missing_mt is True
    assert any("Mitochondrial genes missing" in f for f in ctx.flags)


# ── Downstream adjustment factors ─────────────────────────────────────────


def test_long_transcript_weight_factor_monotone_with_severity():
    base = SampleContext(degradation_severity="none")
    mild = SampleContext(degradation_severity="mild")
    moderate = SampleContext(degradation_severity="moderate")
    severe = SampleContext(degradation_severity="severe")
    assert base.long_transcript_weight_factor() == 1.0
    assert (
        severe.long_transcript_weight_factor()
        < moderate.long_transcript_weight_factor()
        < mild.long_transcript_weight_factor()
        <= 1.0
    )


def test_purity_ci_widening_factor_monotone_with_severity():
    base = SampleContext(degradation_severity="none")
    mild = SampleContext(degradation_severity="mild")
    moderate = SampleContext(degradation_severity="moderate")
    severe = SampleContext(degradation_severity="severe")
    assert base.purity_ci_widening_factor() == 1.0
    assert (
        severe.purity_ci_widening_factor()
        > moderate.purity_ci_widening_factor()
        > mild.purity_ci_widening_factor()
        > 1.0
    )


def test_summary_line_has_prep_and_preservation():
    ctx = SampleContext(
        library_prep="poly_a",
        preservation="ffpe",
        degradation_severity="severe",
    )
    line = ctx.summary_line()
    assert "poly-A" in line
    assert "FFPE" in line
    assert "severe" in line


def test_expression_distribution_signals_populated():
    """The full-range expression summary should capture detection
    breadth and dynamic range for every sample."""
    df = _pan_cancer_ntpm_sample("liver")
    ctx = infer_sample_context(df)
    s = ctx.signals
    for key in (
        "n_genes_with_any_expression",
        "genes_detected_above_0p5_tpm",
        "genes_detected_above_1_tpm",
        "genes_detected_above_10_tpm",
        "log2_tpm_median",
        "log2_tpm_iqr",
        "log2_tpm_p95",
        "top_1pct_share_of_total_tpm",
    ):
        assert key in s, f"{key} missing from signals"
    assert s["n_genes_with_any_expression"] > 5000
    assert 0 < s["top_1pct_share_of_total_tpm"] <= 1.0


def test_tempus_like_exome_capture_ffpe_detected_correctly():
    """Bug 2026-04-14: Tempus FFPE exome-capture samples got labeled
    'unknown library prep, fresh_frozen'. Simulate the signal profile
    from that sample (MT-rRNA absent, MT-mRNA present at vanishing
    fractions, histone in the 0.1–0.5% 'limbo' band, and a long-pair
    index >> 1 from capture enrichment) and verify the detector now
    routes to ``exome_capture`` + ``preservation=unknown``.
    """
    # Background that dominates total TPM.
    rows = [("ACTB", 1000), ("GAPDH", 800), ("TUBB", 600), ("EEF1A1", 500)]
    # Fill out the gene universe enough that histones land in the
    # "limbo" band (~0.3% of total). Use 8 bulk genes at ~100 TPM each.
    for i in range(8):
        rows.append((f"FILLER{i}", 200))
    # Very small amounts of a few MT-mRNA genes (probe bleed-through).
    rows += [("MT-CO1", 0.1), ("MT-ND1", 0.1)]
    # Histones in the "limbo" band — ~0.3% of total.
    total_bg = 1000 + 800 + 600 + 500 + 8 * 200 + 0.2
    target_histone = total_bg * 0.003 / (1 - 0.003)  # ≈ 0.3% of final total
    rows += [("H2BC1", target_histone / 6)] * 6
    # Degradation pairs: long transcripts over-represented ~3x.
    from pirlygenes.gene_sets_cancer import degradation_gene_pairs

    for short_sym, long_sym, expected in list(degradation_gene_pairs())[:15]:
        rows.append((short_sym, 20.0))
        rows.append((long_sym, 20.0 * float(expected) * 3.0))

    frame = _frame_from_pairs(rows)
    ctx = infer_sample_context(frame)

    # Library prep: exome_capture high confidence (MT-rRNA absent,
    # MT-total tiny).
    assert ctx.library_prep == "exome_capture", ctx.library_prep
    assert ctx.library_prep_confidence >= 0.85

    # Preservation: must NOT be labeled fresh_frozen on a capture-
    # biased sample — the length-pair index is inflated artefactually.
    assert ctx.preservation != "fresh_frozen"
    assert ctx.degradation_index is not None
    assert ctx.degradation_index > 1.4
    # The flag must explain the capture-bias interpretation.
    assert any("exon-capture" in f or "capture enrichment" in f for f in ctx.flags), (
        ctx.flags
    )


def test_degradation_index_above_upper_bound_yields_unknown_preservation():
    """Regression: the index > 1.4 path must route to
    ``preservation=unknown`` rather than ``fresh_frozen``."""
    rows = [("ACTB", 500), ("GAPDH", 400)]
    # Every pair has long >> short, index ≈ 5× expected.
    from pirlygenes.gene_sets_cancer import degradation_gene_pairs

    for short_sym, long_sym, expected in list(degradation_gene_pairs())[:12]:
        rows.append((short_sym, 10.0))
        rows.append((long_sym, 10.0 * float(expected) * 5.0))
    rows += [(s, 30) for s in _MT_MRNA_SYMBOLS]  # Present — not exome capture
    frame = _frame_from_pairs(rows)

    ctx = infer_sample_context(frame)
    assert ctx.preservation == "unknown"
    assert ctx.degradation_index is not None and ctx.degradation_index > 1.4


def test_compute_isoform_length_bias_returns_index_for_synthetic_ffpe():
    """Within-gene shorter-isoform preference (FFPE pattern) must yield
    an isoform-length-bias index well below 1.0; balanced isoform usage
    must sit near 1.0."""
    import pandas as pd
    from trufflepig.sample_context import compute_isoform_length_bias

    # Two genes, each with 3 isoforms of different lengths.
    rows = []
    for gene_idx in range(60):  # need >= 50 multi-isoform genes for the call
        gid = f"ENSG{gene_idx:011d}"
        # FFPE pattern: short isoform dominates.
        rows += [
            (f"ENST{gene_idx:08d}A.1", 600, 100.0, gid),
            (f"ENST{gene_idx:08d}B.1", 3000, 5.0, gid),
            (f"ENST{gene_idx:08d}C.1", 8000, 1.0, gid),
        ]
    df = pd.DataFrame(
        rows, columns=["transcript_id", "length", "TPM", "ensembl_gene_id"]
    )

    index, n_genes = compute_isoform_length_bias(df)
    assert n_genes == 60
    assert index is not None
    # Heavy short-isoform preference → index well below 1.0
    assert index < 0.5

    # Now balanced usage — index should sit near 1.0.
    balanced_rows = []
    for gene_idx in range(60):
        gid = f"ENSG{gene_idx:011d}"
        balanced_rows += [
            (f"ENST{gene_idx:08d}A.1", 600, 50.0, gid),
            (f"ENST{gene_idx:08d}B.1", 3000, 50.0, gid),
            (f"ENST{gene_idx:08d}C.1", 8000, 50.0, gid),
        ]
    bdf = pd.DataFrame(
        balanced_rows, columns=["transcript_id", "length", "TPM", "ensembl_gene_id"]
    )
    bidx, _ = compute_isoform_length_bias(bdf)
    assert bidx is not None
    assert 0.7 < bidx < 1.3, f"Balanced usage expected ~1.0, got {bidx}"


def test_compute_ffpe_marker_score_returns_value_for_present_panel():
    from trufflepig.sample_context import compute_ffpe_marker_score

    # Synthetic sample where stable refs are high and FFPE-sensitive
    # genes (the v4.4.1 biology-neutral long-scaffold panel per #75)
    # are very low — the canonical FFPE pattern.
    sample = {
        # stable_in_ffpe: ACTB, GAPDH, RPLP0, RPL13, PPIA, HPRT1, TBP
        "ACTB": 1000,
        "GAPDH": 800,
        "RPLP0": 600,
        "RPL13": 500,
        "PPIA": 400,
        "HPRT1": 100,
        "TBP": 50,
        # drops_in_ffpe: AHNAK, DYNC1H1, HUWE1, UBR4, MACF1, PLEC,
        # BIRC6, SON, SPTAN1, TRRAP (biology-neutral giant scaffolds)
        "AHNAK": 0.1,
        "DYNC1H1": 0.1,
        "HUWE1": 0.1,
        "UBR4": 0.1,
        "MACF1": 0.1,
        "PLEC": 0.1,
        "BIRC6": 0.1,
        "SON": 0.1,
        "SPTAN1": 0.1,
        "TRRAP": 0.1,
    }
    score, n = compute_ffpe_marker_score(sample)
    assert score is not None
    assert n >= 10
    # Heavy suppression of the panel relative to refs → score >> 100.
    assert score > 100.0


def test_capture_biased_ffpe_now_detected_via_orthogonal_signals():
    """Synthesises the Tempus FFPE pattern: capture-enriched (length-
    pair index inflated >1.4) AND FFPE marker panel suppressed AND
    isoform usage shifted short. The orthogonal-signal refinement
    must promote preservation from `unknown` to `ffpe`.
    """
    import pandas as pd
    from pirlygenes.gene_sets_cancer import degradation_gene_pairs
    from trufflepig.sample_context import infer_sample_context

    # Background: enough reference genes and stable HK refs.
    rows = [
        ("ACTB", 1000),
        ("GAPDH", 800),
        ("RPLP0", 600),
        ("RPL13", 500),
        ("PPIA", 400),
        ("HPRT1", 100),
        ("TBP", 50),
        ("TUBB", 300),
        ("EEF1A1", 500),
    ]
    # FFPE-sensitive panel (v4.4.1 biology-neutral scaffolds per #75):
    # collapsed to near-zero.
    for sym in (
        "AHNAK",
        "DYNC1H1",
        "HUWE1",
        "UBR4",
        "MACF1",
        "PLEC",
        "BIRC6",
        "SON",
        "SPTAN1",
        "TRRAP",
    ):
        rows.append((sym, 0.05))
    # MT mRNA present (so library_prep doesn't route to exome here —
    # we want preservation refinement to fire on its own).
    for sym in _MT_MRNA_SYMBOLS:
        rows.append((sym, 30.0))
    # Length-pair index: long >> short, ~3× expected (capture-bias regime).
    for short_sym, long_sym, expected in list(degradation_gene_pairs())[:18]:
        rows.append((short_sym, 20.0))
        rows.append((long_sym, 20.0 * float(expected) * 3.0))

    frame = _frame_from_pairs(rows)

    # Synthesize a transcript-level frame attached via ``attrs`` —
    # within-gene short-isoform preference for 60 multi-isoform genes.
    tx_rows = []
    for i in range(60):
        gid = f"ENSG{i:011d}"
        tx_rows += [
            (f"ENST{i:08d}A.1", 600, 120.0, gid),
            (f"ENST{i:08d}B.1", 3000, 5.0, gid),
            (f"ENST{i:08d}C.1", 8000, 0.5, gid),
        ]
    tx_df = pd.DataFrame(
        tx_rows, columns=["transcript_id", "length", "TPM", "ensembl_gene_id"]
    )
    frame.attrs["transcript_expression"] = tx_df

    ctx = infer_sample_context(frame)
    assert ctx.preservation == "ffpe", (ctx.preservation, ctx.flags)
    assert ctx.degradation_severity in ("moderate", "severe")
    # The flag must mention orthogonal signals.
    assert any("orthogonal signals" in f for f in ctx.flags), ctx.flags


def test_plot_sample_context_writes_png(tmp_path):
    from trufflepig.sample_context import plot_sample_context

    df = _pan_cancer_ntpm_sample("liver")
    ctx = infer_sample_context(df)
    out = tmp_path / "sample-context.png"
    plot_sample_context(ctx, save_to_filename=str(out), save_dpi=80)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_expression_concentration_qc_writes_png(tmp_path):
    from trufflepig.sample_context import (
        plot_expression_concentration_curve_qc,
        plot_expression_concentration_top_features_qc,
    )

    df = pd.DataFrame(
        {
            "gene": ["RNA5SP389", "RNA5-8SP6", "KLK3", "ACTB", "RPLP2"],
            "TPM": [520_000.0, 220_000.0, 30_000.0, 160_000.0, 70_000.0],
        }
    )
    out_top = tmp_path / "expression-top-features-qc.png"
    plot_expression_concentration_top_features_qc(
        df, save_to_filename=str(out_top), save_dpi=80
    )
    out_curve = tmp_path / "expression-concentration-curve-qc.png"
    plot_expression_concentration_curve_qc(
        df, save_to_filename=str(out_curve), save_dpi=80
    )
    assert out_top.exists()
    assert out_top.stat().st_size > 1000
    assert out_curve.exists()
    assert out_curve.stat().st_size > 1000


def test_plot_reference_technical_rna_qc_writes_pngs(tmp_path):
    from trufflepig.sample_context import (
        plot_reference_mtdna_fraction_qc,
        plot_reference_technical_rna_burden_qc,
    )

    df = pd.DataFrame(
        {
            "gene": ["RNA5SP389", "MT-ATP8", "KLK3", "ACTB"],
            "TPM": [520_000.0, 30_000.0, 25_000.0, 425_000.0],
        }
    )
    out2 = tmp_path / "reference-technical-rna-burden.png"
    out3 = tmp_path / "reference-mtdna.png"
    plot_reference_technical_rna_burden_qc(
        df, save_to_filename=str(out2), save_dpi=80
    )
    plot_reference_mtdna_fraction_qc(df, save_to_filename=str(out3), save_dpi=80)
    assert out2.exists()
    assert out2.stat().st_size > 1000
    assert out3.exists()
    assert out3.stat().st_size > 1000


def test_reference_technical_rna_qc_uses_raw_reference_companions(monkeypatch):
    from trufflepig import reference as reference_module
    from trufflepig.sample_context import _reference_qc_fraction_rows

    monkeypatch.setattr(
        reference_module,
        "pan_cancer_expression",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "Ensembl_Gene_ID": [
                    "ENSG00000211459",
                    "ENSG00000198888",
                    "ENSG00000111640",
                ],
                "Symbol": ["MT-RNR1", "MT-ND1", "ACTB"],
                "COAD_TPM": [0.0, 0.0, 1_000_000.0],
                "COAD_TPM_raw": [200_000.0, 100_000.0, 700_000.0],
                "colon_nTPM": [0.0, 0.0, 1_000_000.0],
                "colon_nTPM_raw": [50_000.0, 50_000.0, 900_000.0],
            }
        ),
    )

    rows = _reference_qc_fraction_rows()
    by_label = {row["label"]: row for row in rows}

    assert by_label["COAD"]["mt_dna_fraction"] == pytest.approx(0.3)
    assert by_label["colon"]["mt_dna_fraction"] == pytest.approx(0.1)


def test_histone_prefixes_include_both_hgnc_old_and_new():
    # Guard against accidental deletion: we detect histones whether
    # annotations use old HIST1H* names or modern HGNC H2AC*/H3C*/H4C*.
    prefixes = _HISTONE_SYMBOL_PREFIXES
    assert "H2BC" in prefixes
    assert "H3C" in prefixes
    assert "HIST1H" in prefixes
