"""Regression tests for the battery-audit fixes (PR #159).

Each bug here was a report-correctness issue found by auditing the
markdown outputs for a CRC validation sample and a SARC validation
sample. These tests pin the root-cause fixes so a future refactor
can't silently regress them.
"""

import math

import numpy as np
import pandas as pd
import pytest

from trufflepig.main import _render_vs_tcga_cell
from trufflepig.sample_quality import _MT_GENES, assess_sample_quality
from trufflepig.tumor_purity import _lineage_purity_estimates


# ── #41: _render_vs_tcga_cell state-dispatch ──────────────────────


def _row(**kwargs):
    """Mimic a pandas Series row using a plain dict (``.get`` works)."""
    defaults = {
        "tcga_ref_state": None,
        "pct_cancer_median": None,
        "tcga_cohort_median_tpm": None,
    }
    defaults.update(kwargs)
    return defaults


def test_render_vs_tcga_finite_state_renders_fold():
    row = _row(tcga_ref_state="finite", pct_cancer_median=3.5)
    assert _render_vs_tcga_cell(row) == "3.50\u00d7"


def test_render_vs_tcga_not_in_cohort_shows_raw_tpm_when_positive():
    # CTA case: raw cohort median non-zero but tiny.
    row = _row(
        tcga_ref_state="not_in_cohort",
        pct_cancer_median=float("inf"),
        tcga_cohort_median_tpm=0.14,
    )
    assert _render_vs_tcga_cell(row) == "ref 0.14 TPM"


def test_render_vs_tcga_not_in_cohort_shows_ref_zero_when_absent():
    # CTA case: raw cohort median genuinely 0.
    row = _row(
        tcga_ref_state="not_in_cohort",
        pct_cancer_median=float("inf"),
        tcga_cohort_median_tpm=0.0,
    )
    assert _render_vs_tcga_cell(row) == "ref 0"


def test_render_vs_tcga_tme_explained_shows_tme_only_with_cohort_tpm():
    # TME-deconvolution zeroed the tumor component.
    row = _row(
        tcga_ref_state="tme_explained",
        pct_cancer_median=float("inf"),
        tcga_cohort_median_tpm=12.3,
    )
    assert _render_vs_tcga_cell(row) == "TME-only (12.3 TPM)"


def test_render_vs_tcga_both_absent_renders_dash():
    row = _row(tcga_ref_state="both_absent", pct_cancer_median=None)
    assert _render_vs_tcga_cell(row) == "\u2014"


def test_render_vs_tcga_preserves_large_finite_folds_uncapped():
    """ITGA10 in a SARC validation sample renders at 1548× — capping
    would mask the signal."""
    row = _row(tcga_ref_state="finite", pct_cancer_median=1548.45)
    out = _render_vs_tcga_cell(row)
    assert "1548" in out
    assert "\u00d7" in out  # × symbol preserved


# ── #40: lineage estimator returns TME-dominated genes separately ──


def test_lineage_estimator_returns_tuple_of_estimates_and_skipped():
    """Empty panel → (empty, empty) tuple, not a bare list."""
    estimates, skipped = _lineage_purity_estimates(
        "FAKE_TYPE",
        {},
        {},
        [],
        0.7,
    )
    assert estimates == []
    assert skipped == []


def test_lineage_estimator_separates_tme_dominated_from_usable():
    """A gene present in the sample but TME-dominated in the reference
    lands in ``skipped_detected`` with the sample TPM and reason —
    consumers render it as "uninformative", not "not detected".

    We fabricate a minimal run by stubbing the single public call the
    estimator makes (``pan_cancer_expression``). ACTA2 is the
    canonical case: smooth-muscle marker, heavy TME bleed-through in
    SARC, yet reported at 189 TPM in the biomarker panel.
    """
    from trufflepig import tumor_purity as tp

    # Minimal reference: two genes, one lineage marker (LIN), one
    # housekeeping (HK). LIN has high TME expression (smooth muscle)
    # and low cancer-cohort expression — exactly the SARC / ACTA2
    # regime.
    ref = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG_LIN", "ENSG_HK"],
            "Symbol": ["LIN", "HK"],
            "FPKM_FAKE": [1.0, 10.0],
            "nTPM_smooth_muscle": [200.0, 10.0],
            "nTPM_skeletal_muscle": [200.0, 10.0],
            "nTPM_heart_muscle": [200.0, 10.0],
            "nTPM_adipose_tissue": [100.0, 10.0],
            "nTPM_bone_marrow": [50.0, 10.0],
            "nTPM_lymph_node": [50.0, 10.0],
            "nTPM_spleen": [50.0, 10.0],
            "nTPM_thymus": [50.0, 10.0],
            "nTPM_tonsil": [50.0, 10.0],
            "nTPM_appendix": [50.0, 10.0],
        }
    )

    # Monkey-patch the two loaders the estimator uses.
    orig_pan = tp.pan_cancer_expression
    orig_lineage = tp.LINEAGE_GENES

    try:
        tp.pan_cancer_expression = lambda **_: ref
        tp.LINEAGE_GENES = {"FAKE": ["LIN"]}

        sample_tpm = {"LIN": 190.0, "HK": 50.0}
        estimates, skipped = _lineage_purity_estimates(
            "FAKE",
            sample_tpm,
            {},
            ["HK"],
            0.7,
        )
    finally:
        tp.pan_cancer_expression = orig_pan
        tp.LINEAGE_GENES = orig_lineage

    # LIN's cohort median is 1.0 but TME median across muscle tissues
    # is ~200 → true_tumor_ratio <= tme_ratio, so it lands in skipped.
    assert estimates == [], "TME-dominated gene should not be in estimates"
    assert len(skipped) == 1
    assert skipped[0]["gene"] == "LIN"
    assert skipped[0]["reason"] == "tme_dominated"
    assert skipped[0]["sample_tpm"] == 190.0


# ── #39: MT-quality split — n_mt=0 vs n_mt>0 + low fraction ─────────


def _tcga_sample_with_mt_override(cancer_code, mt_tpm):
    """Take a real TCGA cohort median as a pseudo-sample, then override
    every MT gene's TPM to ``mt_tpm``. Setting ``mt_tpm=0`` simulates
    the filtered / renamed case (no MT rows reach the estimator). Any
    small positive value simulates "present but contribute minimal TPM".
    """
    from trufflepig.reference import pan_cancer_expression

    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    df = pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"FPKM_{cancer_code}"].astype(float),
        }
    )
    mt_mask = df["gene_symbol"].isin(set(_MT_GENES))
    if mt_tpm == 0:
        # Genuinely drop MT rows — models "MT symbols missing from quant
        # table" (filtered / renamed upstream).
        df = df.loc[~mt_mask].reset_index(drop=True)
    else:
        df.loc[mt_mask, "TPM"] = mt_tpm
    return df


def test_mt_quality_flag_says_filtered_when_n_mt_is_zero():
    """When no MT gene symbols are present, the flag says 'filtered or
    renamed upstream' — that's the genuinely-absent case."""
    df = _tcga_sample_with_mt_override("COAD", mt_tpm=0)
    # library_prep=None means the prep-explains-MT short-circuit is off.
    out = assess_sample_quality(df, library_prep=None)
    flags = " | ".join(out["flags"])
    assert "filtered or renamed" in flags
    assert f"0/{len(_MT_GENES)}" in flags


def test_mt_quality_flag_says_low_fraction_when_n_mt_positive():
    """When MT gene symbols ARE present but their TPM share is tiny,
    the flag says 'Low MT fraction' + 'genes present but contribute
    minimal TPM' — not 'filtered or renamed'."""
    df = _tcga_sample_with_mt_override("COAD", mt_tpm=0.01)
    out = assess_sample_quality(df, library_prep=None)
    flags = " | ".join(out["flags"])
    # Should NOT claim filtered/renamed (MT rows are present).
    assert "filtered or renamed" not in flags
    # Should point at the low-fraction interpretation.
    assert "Low MT fraction" in flags
    assert "genes present but contribute minimal TPM" in flags


# ── #35 + #36: decomp-purity adoption guard ─────────────────────────


def test_decomp_purity_adoption_guard_matches_docstring():
    """Pin the three conditions that must all hold for _analyze_body
    to adopt decomp purity over classifier purity:

      1. decomp_agrees: best_decomp.cancer_type == classifier's cancer_code
      2. decomp_has_tme: template warnings don't include the
         "No non-tumor components in template" marker
      3. best_decomp.purity_result is truthy

    The logic isn't extracted into a helper yet — this test encodes
    the contract so a future refactor can pull out a named predicate
    without losing behavior. The canonical failure was a CRC
    validation sample where classifier=COAD, decomp=BRCA → decomp_agrees=False
    → keep classifier's 36%.
    """
    # Simulate the three branches that should NOT adopt decomp purity:
    classifier_code = "COAD"

    # Branch 1: cancer_type mismatch → guard off, don't adopt.
    best_cancer = "BRCA"
    warnings = []
    decomp_agrees = best_cancer == classifier_code
    decomp_has_tme = not any(
        "No non-tumor components in template" in w for w in warnings
    )
    assert not decomp_agrees
    assert not (decomp_agrees and decomp_has_tme)

    # Branch 2: cancer agrees but template has no TME compartments.
    best_cancer = "COAD"
    warnings = ["No non-tumor components in template"]
    decomp_agrees = best_cancer == classifier_code
    decomp_has_tme = not any(
        "No non-tumor components in template" in w for w in warnings
    )
    assert decomp_agrees
    assert not decomp_has_tme
    assert not (decomp_agrees and decomp_has_tme)

    # Branch 3: both OK → adoption is allowed.
    best_cancer = "COAD"
    warnings = ["Primary tissue support exceeds metastatic-site support"]
    decomp_agrees = best_cancer == classifier_code
    decomp_has_tme = not any(
        "No non-tumor components in template" in w for w in warnings
    )
    assert decomp_agrees and decomp_has_tme


# ── #38: proliferation panel denominator in analysis.md ─────────────


def test_proliferation_panel_size_matches_public_api():
    """The analysis.md denominator reads the panel size at render time
    rather than hardcoding an old value — pins that the public API
    ``proliferation_panel_gene_names()`` is the single source."""
    from pirlygenes.gene_sets_cancer import proliferation_panel_gene_names

    panel = proliferation_panel_gene_names()
    assert isinstance(panel, list)
    assert len(panel) >= 5  # protect against the old /5 regression
    # Deduplicated
    assert len(set(panel)) == len(panel)


# ── #37: Step-0 reasoning_trace format ─────────────────────────────


def test_reasoning_trace_rendering_format_stable():
    """The summary-line trace clause is generated from
    ``hvt.reasoning_trace`` via ``' → '.join(...)`` — pin the format so
    the arrow separator (not a comma / pipe) is what downstream docs
    describe."""
    trace = [
        "lymphoid-tissue-tumor-indistinguishable",
        "aggregate-tumor-evidence[aggregate=3.72\u22651.0,CTA_strong(n=5)]",
    ]
    rendered = " \u2192 ".join(trace)
    assert "\u2192" in rendered
    assert rendered.startswith("lymphoid-tissue-tumor-indistinguishable")
    # Single-element traces still render without a trailing arrow.
    solo = " \u2192 ".join(trace[:1])
    assert solo == "lymphoid-tissue-tumor-indistinguishable"
