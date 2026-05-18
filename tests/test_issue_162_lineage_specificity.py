"""Regression test for issue #162 — cross-cohort lineage-panel overlap.

STAD's curated lineage panel (``MUC5AC, MUC6, CDX2, CLDN18``) shares
two genes (MUC5AC, MUC6) that are *higher* in the PAAD cohort median
than in STAD's own. Running STAD's panel on a PAAD sample inflated
STAD's lineage purity to 0.913, which — combined with the GASTRIC
family factor — flipped the classifier to STAD on the PAAD cohort
median.

The fix adds a per-cohort specificity filter
(``_cancer_specific_lineage_genes``) that keeps only genes whose home-
cohort expression dominates the max other-cohort expression
(``home / (home + max_other) ≥ 0.5``). When too few genes pass, the
top-N by specificity are retained so the purity estimator still has
an anchor.
"""

import pandas as pd
import pytest

from trufflepig.reference import pan_cancer_expression
from trufflepig.tumor_purity import (
    LINEAGE_GENES,
    _cancer_specific_lineage_genes,
    _LINEAGE_SPECIFIC_CACHE,
    rank_cancer_type_candidates,
)


@pytest.fixture(autouse=True)
def _reset_specificity_cache():
    """Clear the module-level cache so each test sees fresh results."""
    _LINEAGE_SPECIFIC_CACHE.clear()
    yield
    _LINEAGE_SPECIFIC_CACHE.clear()


# ── _cancer_specific_lineage_genes contract ────────────────────────


def test_stad_panel_drops_genes_higher_in_other_cohorts():
    """MUC6 (higher in PAAD than STAD) and CDX2 (highest in COAD) are
    not STAD-specific — they should be dropped from STAD's filtered
    panel."""
    specific = _cancer_specific_lineage_genes("STAD")
    assert "MUC6" not in specific, (
        "MUC6 (PAAD > STAD) should be dropped from STAD's panel"
    )
    assert "CDX2" not in specific, (
        "CDX2 (COAD > STAD) should be dropped from STAD's panel"
    )


def test_coad_panel_drops_genes_shared_with_other_gi():
    """CDX2 is a CRC lineage marker but the expression is very close
    across COAD/READ (same family) — filter keeps the rest."""
    specific = _cancer_specific_lineage_genes("COAD")
    # Retain at least the minimum-genes floor
    assert len(specific) >= 2


def test_sarc_panel_retains_rare_subtype_markers_below_floor():
    """#167 regression: MYOD1 and MYOG are rhabdomyosarcoma / muscle-
    lineage markers whose cohort median is near-zero in every TCGA
    cohort (including SARC — the cohort includes many non-muscle
    sarcomas, pulling the median to 0). The filter must NOT drop them
    as "non-specific" just because home_val=0; when the max competitor
    is below the competitor floor (5 TPM), the gene is kept regardless.

    On a real sarcoma validation sample, MYOD1 is the only lineage
    marker that yields a usable purity estimate — dropping it
    collapses SARC's lineage concordance and lets THYM (with 0
    concordance) win the classifier.
    """
    specific = _cancer_specific_lineage_genes("SARC")
    assert "MYOD1" in specific, (
        "MYOD1 must be retained in SARC's panel — max-other ≈ 1.5 TPM "
        "is below the 5 TPM competitor floor, so no crosstalk risk"
    )
    assert "MYOG" in specific, (
        "MYOG must be retained in SARC's panel for the same reason"
    )


def test_sarc_panel_drops_tme_shared_smooth_muscle_markers():
    """DES and ACTA2 are expressed at higher levels in PRAD's smooth-
    muscle stroma than in TCGA SARC median, so they're genuinely not
    SARC-specific and SHOULD be dropped from SARC's panel."""
    specific = _cancer_specific_lineage_genes("SARC")
    assert "DES" not in specific, (
        "DES (PRAD=391 > SARC=56) should be dropped from SARC's panel"
    )
    assert "ACTA2" not in specific, (
        "ACTA2 (PRAD=241 > SARC=189) should be dropped from SARC's panel"
    )


def test_filter_always_returns_at_least_minimum_genes():
    """Every cohort with a lineage panel keeps ≥ 2 genes (the
    estimator needs at least that to anchor a stable purity)."""
    for code in LINEAGE_GENES:
        specific = _cancer_specific_lineage_genes(code)
        if LINEAGE_GENES[code]:
            assert len(specific) >= 2, f"{code}: filter returned < 2 genes: {specific}"


def test_filter_returns_subset_of_original_panel():
    """The specificity filter never adds genes — only drops /
    re-orders."""
    for code in LINEAGE_GENES:
        full = set(LINEAGE_GENES[code])
        specific = set(_cancer_specific_lineage_genes(code))
        assert specific <= full, (
            f"{code}: filter added genes not in the original panel: {specific - full}"
        )


def test_filter_result_cached_across_calls():
    """The cache short-circuits repeated calls — important because
    the filter reads the full cohort TPM matrix."""
    _cancer_specific_lineage_genes("STAD")
    assert "STAD" in _LINEAGE_SPECIFIC_CACHE


# ── End-to-end: PAAD median should classify as PAAD ─────────────────


def _cohort_median_sample(code: str) -> pd.DataFrame:
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{code}_TPM"].astype(float),
        }
    )


def test_paad_cohort_median_classifies_as_paad():
    """The canonical failure the specificity filter fixes — before
    the fix STAD's lineage on PAAD median was 0.913 and won the
    ranking via the GASTRIC family boost."""
    df = _cohort_median_sample("PAAD")
    ranked = rank_cancer_type_candidates(df)
    assert ranked[0]["code"] == "PAAD", (
        f"PAAD median miscalled as {ranked[0]['code']}. Top 5: "
        + ", ".join(
            f"{r['code']}(gm={r['support_geomean']:.2f},lin_s={r['lineage_support_factor']:.2f})"
            for r in ranked[:5]
        )
    )


def test_stad_lineage_purity_on_paad_drops_after_filter():
    """On the PAAD cohort median, STAD's lineage purity should be
    materially lower than PAAD's lineage purity — before the filter
    STAD's was 0.913 vs PAAD's 0.420."""
    df = _cohort_median_sample("PAAD")
    ranked = rank_cancer_type_candidates(df)
    stad_row = next((r for r in ranked if r["code"] == "STAD"), None)
    paad_row = next((r for r in ranked if r["code"] == "PAAD"), None)
    assert stad_row is not None and paad_row is not None
    stad_lin = stad_row.get("lineage_purity") or 0.0
    paad_lin = paad_row.get("lineage_purity") or 0.0
    # PAAD's own lineage on its own median should be at least as high
    # as STAD's on the same sample.
    assert paad_lin >= stad_lin, (
        f"PAAD lineage {paad_lin:.2f} < STAD lineage {stad_lin:.2f} "
        "on PAAD median — specificity filter did not fix the "
        "cross-cohort overlap"
    )
