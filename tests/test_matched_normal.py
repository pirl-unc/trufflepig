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
            "TPM": ref[f"{cancer_code}_TPM"].astype(float),
        }
    )


def _normal_tissue_sample(tissue):
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{tissue}_nTPM"].astype(float),
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


def test_coad_solid_primary_stays_colorectal():
    """Synthetic COAD primary must stay in colorectal scope.

    COAD and READ references are deliberately treated as close siblings by the
    report classifier. This decomposition regression guards against unrelated
    tissue/template drift while allowing the expected COAD/READ ambiguity.
    """
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
    assert results[0].cancer_type in {"COAD", "READ"}
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
                "support_fraction_of_top": 1.0,
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
    """Real PRAD-silent CTA expression routes to the absent-cohort state.

    Select a panel of genuine cancer-testis antigens that are zero in the
    shipped PRAD bulk reference, express that panel in the sample, and assert
    against the public range-table state. This is more stable than choosing
    the first zero-valued reference row, whose deconvolved tumor prior may be
    nonzero even when its bulk median is merely near zero.
    """
    df = _tcga_sample("PRAD")
    ref = pan_cancer_expression().drop_duplicates(subset="Symbol")
    symbols = ref["Symbol"].fillna("").astype(str)
    silent_ctas = ref[
        symbols.str.startswith(("MAGE", "GAGE", "PAGE", "CTAG"))
        & ref["PRAD_TPM"].astype(float).le(0.001)
    ]
    assert len(silent_ctas) >= 5
    target_ids = set(silent_ctas["Ensembl_Gene_ID"].astype(str))
    mask = df["ensembl_gene_id"].astype(str).isin(target_ids)
    assert mask.sum() >= 5
    df.loc[mask, "TPM"] = 50.0

    purity = estimate_tumor_purity(df, cancer_type="PRAD")
    ranges = estimate_tumor_expression_ranges(
        df, cancer_type="PRAD", purity_result=purity
    )
    hits = ranges[
        ranges["symbol"].isin(set(silent_ctas["Symbol"].astype(str)))
    ]
    absent = hits[hits["tcga_ref_state"] == "not_in_cohort"]
    assert not absent.empty
    assert absent["pct_cancer_median"].map(np.isinf).all()
    assert absent["median_est"].gt(0.001).all()


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
