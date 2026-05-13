"""Regression tests for #59 item 1 — smooth-muscle stromal leakage annotation.

The matched-normal reference (``nTPM_prostate`` / ``nTPM_colon`` / …)
carries *average* fibromuscular-stroma density for the parent tissue.
A biopsy with above-average smooth-muscle content leaks SM-lineage
signal into the tumor-attributed column. This annotation flags
canonical SM markers (TAGLN, ACTA2, MYH11, CNN1, MYL9, CALD1, SMTN,
MYLK, TPM2) with material tumor attribution so the reader treats the
tumor-cell story with skepticism — rather than silently treating as
tumor-expressed.

NOT a refitting override — strictly a reader-facing caveat.
"""

from trufflepig.main import _format_attribution_cell
from trufflepig.plot_tumor_expr import (
    _SM_LEAKAGE_MIN_OBSERVED_TPM,
    _SM_LEAKAGE_MIN_TUMOR_FRACTION,
    _SMOOTH_MUSCLE_LINEAGE_MARKERS,
)


# ── Panel contents ────────────────────────────────────────────────────


def test_panel_has_canonical_sm_markers():
    for gene in ("TAGLN", "ACTA2", "MYH11", "CNN1", "MYL9", "CALD1"):
        assert gene in _SMOOTH_MUSCLE_LINEAGE_MARKERS


def test_panel_excludes_rhabdomyosarcoma_marker():
    """DES / MYOD1 / MYOG are rhabdomyosarcoma (or cardiac) markers,
    not smooth-muscle-specific. Flagging them as SM leakage would
    mis-annotate RMS / cardiac samples. Pin the carve-out."""
    for gene in ("DES", "MYOD1", "MYOG", "MYF5"):
        assert gene not in _SMOOTH_MUSCLE_LINEAGE_MARKERS


# ── Attribution-cell rendering ───────────────────────────────────────


def _row(symbol, observed, attr_tumor, **kwargs):
    """Build a minimal ranges_df row for rendering."""
    fraction = (attr_tumor / observed) if observed else 0.0
    defaults = {
        "symbol": symbol,
        "observed_tpm": observed,
        "attr_tumor_tpm": attr_tumor,
        "attribution": {"matched_normal_prostate": observed - attr_tumor},
        "attr_top_compartment": "matched_normal_prostate",
        "attr_top_compartment_tpm": observed - attr_tumor,
        "attr_tumor_fraction": fraction,
        "smooth_muscle_stromal_leakage": (
            symbol in _SMOOTH_MUSCLE_LINEAGE_MARKERS
            and observed >= _SM_LEAKAGE_MIN_OBSERVED_TPM
            and fraction >= _SM_LEAKAGE_MIN_TUMOR_FRACTION
        ),
    }
    defaults.update(kwargs)
    return defaults


def test_sm_leakage_tag_fires_on_tagln_with_material_tumor_fraction():
    row = _row("TAGLN", observed=1500.0, attr_tumor=900.0)
    cell = _format_attribution_cell(row)
    assert "smooth-muscle stromal leakage" in cell


def test_sm_leakage_tag_does_not_fire_below_observed_threshold():
    """Observed TPM below the floor — too faint to matter."""
    row = _row("TAGLN", observed=30.0, attr_tumor=20.0)
    cell = _format_attribution_cell(row)
    assert "smooth-muscle stromal leakage" not in cell


def test_sm_leakage_tag_does_not_fire_when_tumor_share_is_small():
    """Gene mostly attributed to matched-normal already — no need to
    flag. The caveat is specifically for cases where tumor share is
    material enough to mislead the reader."""
    row = _row("TAGLN", observed=1500.0, attr_tumor=100.0)  # 6.7% tumor
    cell = _format_attribution_cell(row)
    assert "smooth-muscle stromal leakage" not in cell


def test_sm_leakage_tag_does_not_fire_on_non_sm_gene():
    row = _row("TP53", observed=500.0, attr_tumor=400.0)
    cell = _format_attribution_cell(row)
    assert "smooth-muscle stromal leakage" not in cell


def test_matched_normal_over_predicted_wins_over_sm_leakage():
    """Over-prediction is the strongest caveat — when matched-normal
    alone predicts more than observed, the tumor attribution is at
    the zero floor. The over-predicted tag reads first; SM leakage is
    the weaker / downstream annotation."""
    row = _row(
        "TAGLN",
        observed=1500.0,
        attr_tumor=0.0,  # zero-floor case
        matched_normal_over_predicted=True,
        attr_tumor_fraction=0.0,
        smooth_muscle_stromal_leakage=False,  # 0% fraction → no SM tag
    )
    cell = _format_attribution_cell(row)
    assert "matched-normal over-predicted" in cell


# ── End-to-end: column lands in ranges_df ────────────────────────────


def test_ranges_df_emits_smooth_muscle_stromal_leakage_column(tmp_path):
    """Pin the new column surface area on the real estimator."""
    import pandas as pd
    from trufflepig.reference import pan_cancer_expression
    from trufflepig.plot import estimate_tumor_expression_ranges

    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    df = pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref["nTPM_prostate"].astype(float) + 1.0,
        }
    )
    # Inject high TAGLN so the flag has a chance to fire.
    df.loc[df["gene_symbol"] == "TAGLN", "TPM"] = 1500.0

    purity = {
        "overall_estimate": 0.3,
        "overall_lower": 0.2,
        "overall_upper": 0.4,
        "components": {"stromal": {"enrichment": 1.2}, "immune": {"enrichment": 1.1}},
    }
    out = estimate_tumor_expression_ranges(
        df_gene_expr=df,
        cancer_type="PRAD",
        purity_result=purity,
    )
    assert "smooth_muscle_stromal_leakage" in out.columns
    # Column dtype must be boolean-coercible (the render path calls
    # ``bool(row.get(...))``).
    leakage_vals = out["smooth_muscle_stromal_leakage"]
    assert set(leakage_vals.unique()).issubset({True, False})
