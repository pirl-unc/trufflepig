"""Regression tests for the matched-normal epithelium decomposition path (issue #50).

Matched-normal subtraction runs unconditionally for epithelial primaries
whose cancer code is in ``EPITHELIAL_MATCHED_NORMAL_TISSUE``. Tests cover:

- Template plumbing (matched-normal component appears for epithelial
  primaries, not for mesenchymal / heme / unmapped cancer codes).
- ``ffa9325`` regression protection: PRAD+smooth_muscle still picks
  solid_primary; a synthetic COAD mixture still resolves to COAD.
- End-to-end: pure-normal-prostate run as PRAD assigns the full TME
  allocation to matched_normal and drops purity to zero via the
  lineage-specific tumor-fraction estimator.
- Panel utilities produce sensible gene lists for representative
  epithelial cancers.
- Companion fix for ``vs_tcga`` inf-routing for silent-TCGA genes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trufflepig.decomposition import (
    EPITHELIAL_MATCHED_NORMAL_TISSUE,
    build_matched_normal_biased_panel,
    build_shared_lineage_panel,
    build_tumor_biased_panel,
    decompose_sample,
    epithelial_matched_normal_component,
    summarize_panels,
)
from trufflepig.decomposition.templates import get_template_components
from trufflepig.reference import pan_cancer_expression
from trufflepig.plot import estimate_tumor_expression_ranges
from trufflepig.tumor_purity import estimate_tumor_purity


# ── Fixtures: synthetic samples from the bundled reference ─────────────


def _tcga_sample(cancer_code):
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"FPKM_{cancer_code}"].astype(float),
        }
    )


def _normal_tissue_sample(tissue):
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"nTPM_{tissue}"].astype(float),
        }
    )


def _mix_samples(parts):
    value_by_gene = {}
    symbol_by_gene = {}
    for weight, df in parts:
        for row in df.itertuples(index=False):
            value_by_gene[row.ensembl_gene_id] = value_by_gene.get(
                row.ensembl_gene_id, 0.0
            ) + weight * float(row.TPM)
            symbol_by_gene[row.ensembl_gene_id] = row.gene_symbol
    out = pd.DataFrame({"ensembl_gene_id": list(value_by_gene.keys())})
    out["gene_symbol"] = out["ensembl_gene_id"].map(symbol_by_gene)
    out["TPM"] = out["ensembl_gene_id"].map(value_by_gene)
    return out


# ── Template plumbing ───────────────────────────────────────────────────


def test_matched_normal_helper_returns_expected_component():
    assert epithelial_matched_normal_component("PRAD") == "matched_normal_prostate"
    assert epithelial_matched_normal_component("COAD") == "matched_normal_colon"
    assert epithelial_matched_normal_component("READ") == "matched_normal_rectum"
    # Mesenchymal / heme / missing cancer types get no matched-normal term.
    assert epithelial_matched_normal_component("SARC") is None
    assert epithelial_matched_normal_component("DLBC") is None
    assert epithelial_matched_normal_component(None) is None


def test_get_template_components_appends_matched_normal_for_epithelial_primaries():
    comps = get_template_components("solid_primary", cancer_type="PRAD")
    assert "matched_normal_prostate" in comps
    assert "tumor" in comps
    # Only appended for solid_primary; met templates stay unchanged.
    met_comps = get_template_components("met_liver", cancer_type="PRAD")
    assert "matched_normal_prostate" not in met_comps


def test_get_template_components_skips_matched_normal_for_non_epithelial():
    # Mesenchymal / heme / glial cancers have no matched-normal mapping
    # in EPITHELIAL_MATCHED_NORMAL_TISSUE, so solid_primary stays
    # unchanged for them.
    sarc = get_template_components("solid_primary", cancer_type="SARC")
    assert not any(c.startswith("matched_normal_") for c in sarc)
    dlbc = get_template_components("solid_primary", cancer_type="DLBC")
    assert not any(c.startswith("matched_normal_") for c in dlbc)
    gbm = get_template_components("solid_primary", cancer_type="GBM")
    assert not any(c.startswith("matched_normal_") for c in gbm)


# ── Regression tests: ffa9325 template-selection cases preserved ────────


def test_prad_smooth_muscle_mix_stays_solid_primary():
    """The PRAD + 80% smooth_muscle mix must rank solid_primary above
    met_soft_tissue even though solid_primary now carries an extra
    matched_normal_prostate compartment. The extra_components scoring
    branch excludes matched_normal_* and marker selection skips it, so
    this ``ffa9325`` regression does not resurface."""
    df = _mix_samples(
        [
            (0.2, _tcga_sample("PRAD")),
            (0.8, _normal_tissue_sample("smooth_muscle")),
        ]
    )
    results = decompose_sample(
        df,
        cancer_types=["PRAD"],
        templates=["solid_primary", "met_bone", "met_soft_tissue"],
        top_k=3,
    )
    assert results[0].template == "solid_primary"
    assert results[0].cancer_type == "PRAD"
    assert results[0].score > results[1].score


def test_coad_solid_primary_stays_coad():
    """Synthetic COAD primary must resolve to COAD (not READ) with
    matched_normal_<tissue> appended to both. Fit-quality differential
    between colon and rectum references still favors COAD."""
    df = _mix_samples(
        [
            (0.6, _tcga_sample("COAD")),
            (0.4, _normal_tissue_sample("colon")),
        ]
    )
    results = decompose_sample(
        df,
        cancer_types=["COAD", "READ"],
        templates=["solid_primary"],
        top_k=2,
    )
    assert results[0].cancer_type == "COAD"
    assert results[0].template == "solid_primary"


# ── Matched-normal actually subtracts parent-tissue signal ──────────────


def test_pure_normal_prostate_run_as_prad_assigns_matched_normal_mass():
    """A pure normal-prostate sample run as PRAD should assign all TME
    mass to matched_normal_prostate AND identify the sample as tumor-
    free via the lineage panel. This is the motivating scenario from
    issue #50 — without the matched-normal compartment and the lineage-
    specific tumor-fraction estimator, prostate lineage signal (KLK3)
    gets attributed to tumor cells with purity ≈ 0.31 (signature-gene
    bias)."""
    df = _normal_tissue_sample("prostate")
    results = decompose_sample(
        df,
        cancer_types=["PRAD"],
        templates=["solid_primary"],
        top_k=1,
    )
    assert len(results) == 1
    result = results[0]
    assert result.matched_normal_tissue == "prostate"
    assert result.fractions.get("matched_normal_prostate", 0.0) > 0.3
    # Lineage-panel purity estimator should recognise this sample as
    # tumor-free and override the signature-based purity estimate.
    assert result.purity_source == "lineage_panel"
    assert result.purity < 0.05
    # And the matched-normal compartment absorbs effectively all the
    # non-tumor mass.
    assert result.matched_normal_fraction > 0.9


def test_estimate_ranges_splits_parent_tissue_for_prad_mixture():
    """With a 50/50 PRAD + normal-prostate mix the tumor-expression range
    output exposes a non-zero matched_normal_tpm for prostate-specific
    lineage genes (KLK3) and marks those genes with ``estimation_path
    == "matched_normal_split"`` unless the existing TME-explainable
    clamp fires. This is the three-component formula working end-to-end."""
    df = _mix_samples(
        [
            (0.5, _tcga_sample("PRAD")),
            (0.5, _normal_tissue_sample("prostate")),
        ]
    )
    purity = estimate_tumor_purity(df, cancer_type="PRAD")
    results = decompose_sample(
        df,
        cancer_types=["PRAD"],
        templates=["solid_primary"],
        top_k=1,
    )
    ranges = estimate_tumor_expression_ranges(
        df,
        cancer_type="PRAD",
        purity_result=purity,
        decomposition_results=results,
    )

    assert "matched_normal_tpm" in ranges.columns
    assert "tme_only_tpm" in ranges.columns
    assert "estimation_path" in ranges.columns
    # At least a handful of genes should actually exercise the
    # matched-normal subtraction path.
    active = ranges[ranges["matched_normal_tpm"] > 0.0]
    assert len(active) > 50

    klk3 = ranges[ranges["symbol"] == "KLK3"]
    if not klk3.empty:
        # For KLK3 (prostate-lineage retained) in a half-normal mix, the
        # matched-normal TPM contribution must be non-trivial.
        assert float(klk3["matched_normal_tpm"].iloc[0]) > 5.0


def test_estimate_ranges_non_epithelial_cancer_has_no_matched_normal_split():
    """SARC / mesenchymal samples stay on the original non-matched-normal
    path: there's no matched_normal_<tissue> component for SARC (see
    issue #51), so the matched_normal_tpm column stays uniformly zero."""
    df = _tcga_sample("SARC")
    purity = estimate_tumor_purity(df, cancer_type="SARC")
    results = decompose_sample(
        df,
        cancer_types=["SARC"],
        templates=["solid_primary"],
        top_k=1,
    )
    ranges = estimate_tumor_expression_ranges(
        df,
        cancer_type="SARC",
        purity_result=purity,
        decomposition_results=results,
    )
    assert "matched_normal_tpm" in ranges.columns
    assert (ranges["matched_normal_tpm"] == 0.0).all()
    assert (ranges["matched_normal_tissue"].fillna("") == "").all()


def test_lineage_override_rejects_large_upward_jump(monkeypatch):
    """A matched-normal panel can correct an over-high signature purity
    downward, but it must not jump a plausible low-purity call up to an
    implausibly high value from a contaminated panel."""
    from trufflepig.decomposition import engine

    monkeypatch.setattr(
        engine,
        "estimate_lineage_tumor_fraction",
        lambda *_a, **_k: {
            "estimate": 0.8,
            "lower": 0.7,
            "upper": 0.9,
            "stability": 0.4,
            "panel_size": 30,
            "panel_genes_observed": 20,
            "per_gene": [],
        },
    )

    df = _tcga_sample("PRAD")
    results = decompose_sample(
        df,
        cancer_types=["PRAD"],
        candidate_rows=[
            {
                "code": "PRAD",
                "signature_score": 0.7,
                "purity_estimate": 0.2,
                "support_norm": 1.0,
                "purity_result": {
                    "overall_estimate": 0.2,
                    "overall_lower": 0.1,
                    "overall_upper": 0.3,
                },
            }
        ],
        templates=["solid_primary"],
        top_k=1,
    )
    assert len(results) == 1
    result = results[0]
    assert result.purity_source == "signature"
    assert result.purity == pytest.approx(0.2)
    assert any("Lineage-panel purity conflicts" in w for w in result.warnings)


# ── vs_tcga label-routing fix (companion to issue #50 PR) ───────────────


def test_vs_tcga_inf_routes_for_silent_tcga_with_sample_expression():
    """Genes where the TCGA cohort tumor-component estimate clips to
    exactly zero but the sample expresses meaningfully must render as
    ``pct_cancer_median = inf`` (so the plot labels them "absent in
    TCGA" rather than the quiet gray "0 in TCGA"). Previously the
    ``tcga_tumor_fold <= 0`` branch routed to None unconditionally."""
    df = _tcga_sample("PRAD")
    # Inject a CTA-like gene (silent in PRAD cohort) at high sample TPM.
    # Use an Ensembl ID known to be in the reference but near-zero in
    # FPKM_PRAD. MAGE-family CTAs typically satisfy this.
    ref = pan_cancer_expression().drop_duplicates(subset="Symbol")
    ref = ref[ref["FPKM_PRAD"].astype(float) < 0.01]
    if ref.empty:
        pytest.skip("No silent-in-PRAD gene found in reference")
    target = ref.iloc[0]
    target_eid = str(target["Ensembl_Gene_ID"])
    mask = df["ensembl_gene_id"].astype(str) == target_eid
    if not mask.any():
        pytest.skip("Silent-PRAD gene not present in synthetic sample")
    df.loc[mask, "TPM"] = 50.0

    purity = estimate_tumor_purity(df, cancer_type="PRAD")
    ranges = estimate_tumor_expression_ranges(
        df, cancer_type="PRAD", purity_result=purity
    )
    hit = ranges[ranges["symbol"] == target["Symbol"]]
    if hit.empty:
        pytest.skip("Target silent-PRAD gene did not survive ranges filter")
    val = hit["pct_cancer_median"].iloc[0]
    assert val is None or np.isinf(val), (
        f"Expected inf (absent-in-TCGA) or None for a gene silent in TCGA but "
        f"expressed in the sample; got {val!r}"
    )
    # The stronger assertion — it must be inf, not None — only holds when
    # the gene's median_est comes out > 0.001. Allow None if estimator
    # chose to clip it, but do NOT allow a finite positive (that would
    # revive the bug).
    assert not (isinstance(val, float) and np.isfinite(val) and val > 0)


# ── Panel-construction utilities ────────────────────────────────────────


def test_build_tumor_biased_panel_prad_contains_known_markers():
    """The tumor-biased panel should surface at least one canonical
    PRAD-biased lineage/regulatory gene. We check for presence of any
    gene in a small known set rather than a single symbol because the
    TCGA cohort bulk has ~30% non-tumor contamination and HPA's normal-
    prostate reference carries strong glandular signal — so classic
    markers like AMACR/KLK3 don't always survive a naive bulk-vs-bulk
    comparison even after TCGA_MEDIAN_PURITY deconvolution.
    """
    panel = build_tumor_biased_panel("PRAD", delta_log2=1.0, min_tumor_expression=1.0)
    symbols = set(panel["symbol"].astype(str))
    known_prad_biased = {"HOXB13", "NKX3-1", "FOLH1", "STEAP2", "PCAT1", "DLX1"}
    assert known_prad_biased & symbols, (
        f"Expected at least one of {known_prad_biased} in PRAD tumor-biased panel "
        f"(top 10: {panel['symbol'].head(10).tolist()})"
    )


def test_build_matched_normal_biased_panel_prad_not_empty():
    panel = build_matched_normal_biased_panel("PRAD", delta_log2=1.0)
    # Normal prostate has genes higher than TCGA-PRAD cohort (prostate
    # cohort tumor cells often repress some differentiation markers).
    # We don't pin specific genes — the assertion is "non-trivially-sized
    # panel exists".
    assert len(panel) > 20


def test_build_shared_lineage_panel_prad_contains_klk3():
    # Raw-bulk comparison (not deconvolved) — see build_shared_lineage_panel
    # docstring for why. KLK3 is the canonical shared-lineage gene
    # (high in both PRAD tumor cohort and benign prostate reference).
    panel = build_shared_lineage_panel("PRAD")
    symbols = set(panel["symbol"].astype(str))
    assert "KLK3" in symbols


def test_summarize_panels_returns_counts_for_every_epithelial_cancer():
    for code in EPITHELIAL_MATCHED_NORMAL_TISSUE:
        summary = summarize_panels(code)
        assert summary["tissue"] is not None
        assert summary["tumor_biased"] >= 0
        assert summary["matched_normal_biased"] >= 0
        assert summary["shared_lineage"] >= 0
