"""Tests for the v4.2 bundle: #25, #26, #27, #35, #60, #68.

Each test pins a specific contract introduced in the propagation work:

- #60: extended housekeeping symbols never surface as auto-picked
  decomposition markers, and carry ``excluded_from_ranking=True`` in
  the therapy-target output. ``B2M`` stays visible in ranking (MHC-I
  context) but is still skipped from marker selection.
- #25: when ``sample_context.is_degraded`` is True, markers drawn from
  the ``long`` column of ``degradation-gene-pairs.csv`` have their
  NNLS fit weight scaled by ``long_transcript_weight_factor``.
- #26: the purity block gains a ``degradation_caveat`` dict when the
  sample is degraded.
- #27: ``plot_degradation_index`` emits a PNG for any sample with at
  least one pair where the short gene is expressed.
- #35: ranges rows include ``tme_dominant`` / ``low_confidence_tumor``
  columns; TME-dominant genes are flagged in the report output.
- #68: SampleContext signals include top-50 / top-2000 concentration,
  detection breadth, and a targeted-panel heuristic flag.
"""

import pandas as pd
import pytest

from trufflepig.decomposition.engine import (
    _is_excluded_auto_marker,
    _select_marker_rows,
)
from pirlygenes.gene_sets_cancer import (
    degradation_gene_pairs,
    is_extended_housekeeping_symbol,
)
from trufflepig.plot import estimate_tumor_expression_ranges
from trufflepig.sample_context import (
    SampleContext,
    infer_sample_context,
    plot_degradation_index,
)
from trufflepig.reference import pan_cancer_expression


# ── #60: extended housekeeping exclusion ───────────────────────────────


def test_is_extended_housekeeping_covers_all_listed_families():
    # Sample representatives from each regex family in the panel.
    excluded_both = [
        "MT-CO1",  # mitochondrial
        "RPL13A",  # cytosolic ribosomal
        "RPS6",  # cytosolic ribosomal
        "MRPL12",  # mito ribosomal
        "MRPS7",  # mito ribosomal
        "EEF1A1",  # translation factor
        "EIF3A",  # translation factor
        "HNRNPA1",  # hnRNP
        "SRSF1",  # SR splicing protein
        "SNRPB",  # snRNP
        "PSMA1",  # proteasome
        "PSMB5",  # proteasome
        "PPIA",  # cyclophilin
        "TUBA1A",  # tubulin
        "TUBB2A",  # tubulin
        "IGHV1-2",  # rearranged IG
        "TRAV1-1",  # rearranged TR
        "ACTB",  # classic HK
        "GAPDH",  # classic HK
    ]
    for sym in excluded_both:
        assert is_extended_housekeeping_symbol(sym, scope="markers"), (
            f"{sym} missing from markers scope"
        )
        assert is_extended_housekeeping_symbol(sym, scope="ranking"), (
            f"{sym} missing from ranking scope"
        )


def test_b2m_excluded_from_markers_but_kept_in_ranking():
    """B2M is a canonical housekeeping but has clinical MHC-I relevance
    so it must stay visible in the ranked therapy-target output.
    """
    assert is_extended_housekeeping_symbol("B2M", scope="markers")
    assert not is_extended_housekeeping_symbol("B2M", scope="ranking")


def test_real_biology_not_in_extended_housekeeping():
    """Guard against over-filtering — lineage / signalling / MHC-I
    genes must NOT be caught by the regex families."""
    for sym in ["TP53", "HLA-A", "HLA-B", "HLA-C", "EGFR", "MYC", "AR", "KLK3"]:
        assert not is_extended_housekeeping_symbol(sym)


def test_marker_auto_selection_skips_extended_housekeeping(monkeypatch):
    import numpy as np

    # Synthetic matrix where the excluded symbols happen to be high
    # in the B_cell column. Without the #60 exclusion they would be
    # auto-picked by the specificity rule; with it they must not.
    excluded = ["MT-CO1", "RPL13A", "HNRNPA1", "EEF1A1", "PSMA1"]
    clean = ["MS4A1", "CD79A", "BANK1"]
    genes = excluded + clean + [f"OTHER{i}" for i in range(50)]
    symbols = list(genes)

    n = len(genes)
    mat = np.full((n, 3), 0.1)
    for i in range(len(excluded)):
        mat[i, 1] = 200.0
    for i in range(len(excluded), len(excluded) + len(clean)):
        mat[i, 1] = 150.0

    _, _, marker_df = _select_marker_rows(
        genes=genes,
        symbols=symbols,
        signature_tpm=mat,
        comp_names=["T_cell", "B_cell", "myeloid"],
    )
    b_cell_markers = set(marker_df[marker_df["component"] == "B_cell"]["symbol"])
    for sym in excluded:
        assert sym not in b_cell_markers


def test_is_excluded_auto_marker_still_blocks_mhc2_superset():
    """#31 blacklist (CD74 / HLA-DPB1 / …) stays in force on top of
    the #60 extended housekeeping panel."""
    for sym in ("CD74", "HLA-DPB1", "HLA-DQB1"):
        assert _is_excluded_auto_marker(sym)


# ── #25: long-transcript downweight on FFPE ────────────────────────────


def test_long_transcript_markers_downweighted_under_ffpe_context(monkeypatch):
    """Under a degraded SampleContext, markers drawn from the long
    side of degradation-gene-pairs get ``fit_weight`` scaled by the
    long-transcript weight factor."""
    import numpy as np

    # Pick a long-transcript gene from the bundled degradation pairs
    # and force it to look like a clearly-discriminative fibroblast
    # marker. Under no-context baseline it should get full weight;
    # under severe degradation it should be scaled down.
    pairs = list(degradation_gene_pairs())
    _, long_sym, _ = pairs[0]
    genes = [long_sym, "FOO1", "FOO2", "FOO3"]
    symbols = list(genes)
    mat = np.full((len(genes), 2), 0.1)
    mat[0, 1] = 200.0  # long gene = strong fibroblast signal

    _, weights_no_ctx, df_no_ctx = _select_marker_rows(
        genes=genes,
        symbols=symbols,
        signature_tpm=mat,
        comp_names=["T_cell", "fibroblast"],
        sample_context=None,
    )
    _, weights_severe, df_severe = _select_marker_rows(
        genes=genes,
        symbols=symbols,
        signature_tpm=mat,
        comp_names=["T_cell", "fibroblast"],
        sample_context=SampleContext(degradation_severity="severe"),
    )

    long_row_ctx = df_severe[df_severe["symbol"] == long_sym]
    long_row_no = df_no_ctx[df_no_ctx["symbol"] == long_sym]
    assert not long_row_no.empty
    assert not long_row_ctx.empty
    assert float(long_row_ctx["fit_weight"].iloc[0]) < float(
        long_row_no["fit_weight"].iloc[0]
    )


# ── #27: degradation-index plot ────────────────────────────────────────


def _pan_cancer_ntpm_sample(tissue):
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{tissue}_nTPM"].astype(float),
        }
    )


def test_plot_degradation_index_writes_file(tmp_path):
    df = _pan_cancer_ntpm_sample("liver")
    ctx = infer_sample_context(df)
    out = tmp_path / "degradation-index.png"
    result = plot_degradation_index(df, ctx, save_to_filename=str(out), save_dpi=80)
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 1000


def test_plot_degradation_index_returns_none_when_no_pairs():
    df = pd.DataFrame(
        [
            {"gene_symbol": "ACTB", "TPM": 100.0},
        ]
    )
    ctx = SampleContext(degradation_severity="none")
    result = plot_degradation_index(df, ctx, save_to_filename="/tmp/never.png")
    assert result is None


# ── #35 / #60: ranges_df carries TME flags and excluded marker ─────────


def test_estimate_tumor_expression_ranges_adds_tme_flags_and_exclusion_flag():
    df = _pan_cancer_ntpm_sample("liver")
    purity = {
        "overall_estimate": 0.6,
        "overall_lower": 0.4,
        "overall_upper": 0.8,
        "components": {"stromal": {"enrichment": 1.0}, "immune": {"enrichment": 1.0}},
    }
    ranges = estimate_tumor_expression_ranges(df, "COAD", purity)

    # New columns from the v4.2 bundle.
    for col in (
        "tme_explainable",
        "tme_dominant",
        "low_confidence_tumor",
        "excluded_from_ranking",
    ):
        assert col in ranges.columns, f"{col} missing from ranges_df"

    # An extended-housekeeping symbol that happens to be in the
    # output universe (if present) must be marked excluded_from_ranking.
    # Look at ACTB specifically.
    actb = ranges[ranges["symbol"] == "ACTB"]
    if not actb.empty:
        assert bool(actb.iloc[0]["excluded_from_ranking"]) is True, (
            "ACTB must be flagged excluded_from_ranking"
        )
    # And B2M, when present, must NOT be excluded from ranking.
    b2m = ranges[ranges["symbol"] == "B2M"]
    if not b2m.empty:
        assert bool(b2m.iloc[0]["excluded_from_ranking"]) is False, (
            "B2M must stay visible in ranking output"
        )


# ── #68: input characterization signals ────────────────────────────────


def test_sample_context_signals_include_concentration_and_panel_heuristic():
    df = _pan_cancer_ntpm_sample("liver")
    ctx = infer_sample_context(df)
    s = ctx.signals
    for key in (
        "top_50_share_of_total_tpm",
        "top_2000_share_of_total_tpm",
        "likely_targeted_panel",
    ):
        assert key in s, f"{key} missing from signals"
    # Whole-transcriptome reference sample should NOT trigger the
    # targeted-panel heuristic.
    assert s["likely_targeted_panel"] is False


def test_sample_context_flags_likely_panel_for_sparse_input():
    """A frame with only a handful of genes should trip the heuristic."""
    df = pd.DataFrame(
        [
            {"gene_symbol": sym, "TPM": tpm}
            for sym, tpm in [
                ("EGFR", 500.0),
                ("MYC", 400.0),
                ("TP53", 300.0),
                ("AR", 200.0),
                ("KLK3", 100.0),
            ]
        ]
    )
    ctx = infer_sample_context(df)
    assert ctx.signals.get("likely_targeted_panel") is True


# ── #26: degradation_caveat field presence is driven by cli.analyze ────
# Tested via cli integration elsewhere — here we just assert the shape
# of the SampleContext widening factor is plumbed.


def test_purity_ci_widening_factor_numeric_and_greater_than_one_under_ffpe():
    ctx = SampleContext(degradation_severity="severe")
    assert ctx.purity_ci_widening_factor() > 1.0
    # Sanity: clean sample leaves CIs alone.
    clean = SampleContext(degradation_severity="none")
    assert clean.purity_ci_widening_factor() == pytest.approx(1.0)
