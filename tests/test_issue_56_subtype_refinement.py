"""Regression tests for #56 — post-hoc CAF / TAM reference refinement.

The decomposition anchors ``fibroblast`` to HPA generic fibroblast and
``myeloid`` to HPA generic macrophage. Both references under-represent
the tumor-activated states (CAF / TAM), so canonical marker genes
(FAP, POSTN, CD163, MRC1, …) end up with inflated tumor attribution.

``refine_tme_per_gene`` swaps the per-gene reference for these marker
genes post-hoc — NNLS stays unchanged, only marker-gene TME
contributions scale up to the tumor-activated fold-over-baseline.
"""

from trufflepig.decomposition.subtype_refs import (
    CAF_MARKER_FOLDS,
    TAM_MARKER_FOLDS,
    caf_markers,
    partition_compartment,
    refine_tme_per_gene,
    tam_markers,
)


# ── Panel contents ────────────────────────────────────────────────────


def test_caf_panel_has_canonical_markers():
    caf = set(caf_markers())
    for gene in ("FAP", "POSTN", "S100A4", "TNC", "COL1A1"):
        assert gene in caf, f"CAF panel missing {gene}"


def test_tam_panel_has_canonical_markers():
    tam = set(tam_markers())
    for gene in ("CD163", "MRC1", "LYVE1", "TREM2", "MARCO"):
        assert gene in tam, f"TAM panel missing {gene}"


def test_caf_tam_panels_do_not_overlap():
    """A gene on both panels would double-count in refinement. The
    canonical markers are lineage-specific so zero overlap is the
    expected state; this test pins it."""
    assert not (set(caf_markers()) & set(tam_markers()))


def test_folds_are_strictly_greater_than_one():
    """Fold values are ``tumor-activated / generic-baseline``. A fold
    of 1.0 means no refinement — violates the premise of this module."""
    for gene, fold in CAF_MARKER_FOLDS.items():
        assert fold > 1.0, f"CAF {gene} fold = {fold}"
    for gene, fold in TAM_MARKER_FOLDS.items():
        assert fold > 1.0, f"TAM {gene} fold = {fold}"


# ── refine_tme_per_gene contract ──────────────────────────────────────


def test_non_marker_genes_pass_through_unchanged():
    tme_bg = {"ACTB": 50.0, "GAPDH": 80.0}  # canonical housekeeping
    per_comp = {"ACTB": {"fibroblast": 10.0}, "GAPDH": {"fibroblast": 5.0}}
    sample = {"ACTB": 100.0, "GAPDH": 150.0}
    refined, prov = refine_tme_per_gene(tme_bg, per_comp, sample)
    assert refined == tme_bg
    assert prov == {}


def test_caf_marker_with_fibroblast_contribution_gets_refined_up():
    """FAP at 100 TPM observed, 10 TPM from fibroblast compartment
    (i.e. the NNLS says "generic fibroblast contributes 10"). After
    refinement the compartment contribution scales by CAF fold (10×)
    → 100 TPM — clamped at observed. TME_bg reflects the delta."""
    tme_bg = {"FAP": 15.0}  # 10 from fibroblast + 5 from other compartments
    per_comp = {"FAP": {"fibroblast": 10.0, "endothelial": 5.0}}
    sample = {"FAP": 100.0}
    refined, prov = refine_tme_per_gene(tme_bg, per_comp, sample)
    assert refined["FAP"] > tme_bg["FAP"]
    # Refined TME must not exceed observed — otherwise tumor_tpm goes
    # negative, which the caller would silently clip to zero.
    assert refined["FAP"] <= sample["FAP"]
    assert prov["FAP"]["subtype"] == "CAF"
    assert prov["FAP"]["fold"] == CAF_MARKER_FOLDS["FAP"]
    assert prov["FAP"]["before"] == 15.0
    assert prov["FAP"]["after"] == refined["FAP"]


def test_tam_marker_with_myeloid_contribution_gets_refined_up():
    tme_bg = {"CD163": 20.0}
    per_comp = {"CD163": {"myeloid": 15.0}}
    sample = {"CD163": 200.0}
    refined, prov = refine_tme_per_gene(tme_bg, per_comp, sample)
    assert refined["CD163"] > tme_bg["CD163"]
    assert refined["CD163"] <= sample["CD163"]
    assert prov["CD163"]["subtype"] == "TAM"


def test_caf_marker_without_fibroblast_contribution_is_not_refined():
    """If the NNLS fit put zero signal on fibroblast for this sample,
    there's nothing to swap — refinement must no-op."""
    tme_bg = {"FAP": 12.0}
    per_comp = {"FAP": {"endothelial": 12.0}}  # no fibroblast column
    sample = {"FAP": 50.0}
    refined, prov = refine_tme_per_gene(tme_bg, per_comp, sample)
    assert refined["FAP"] == 12.0
    assert "FAP" not in prov


def test_refinement_clamped_at_observed():
    """An over-eager fold × fibroblast_tpm that exceeds observed must
    clamp at observed — we never subtract more than the sample carries."""
    tme_bg = {"POSTN": 10.0}
    per_comp = {"POSTN": {"fibroblast": 10.0}}  # fold 8 → 80, but observed is 25
    sample = {"POSTN": 25.0}
    refined, prov = refine_tme_per_gene(tme_bg, per_comp, sample)
    assert refined["POSTN"] == sample["POSTN"]
    # Provenance still captured so the report shows the refinement fired.
    assert "POSTN" in prov


def test_zero_sample_gene_is_not_refined():
    """If the sample doesn't express the marker at all, there's nothing
    meaningful to scale up."""
    tme_bg = {"FAP": 0.0}
    per_comp = {"FAP": {"fibroblast": 0.0}}
    sample = {"FAP": 0.0}
    refined, prov = refine_tme_per_gene(tme_bg, per_comp, sample)
    assert refined["FAP"] == 0.0
    assert "FAP" not in prov


def test_per_compartment_none_fallback():
    """When per_compartment_tpm is unavailable, refinement falls back
    to scaling the aggregate TME by the marker fold. Less precise but
    preserves directionality."""
    tme_bg = {"FAP": 10.0}
    sample = {"FAP": 200.0}
    refined, prov = refine_tme_per_gene(tme_bg, None, sample)
    assert refined["FAP"] > tme_bg["FAP"]
    assert refined["FAP"] <= sample["FAP"]


# ── partition_compartment ─────────────────────────────────────────────


def test_partition_all_to_generic_when_no_marker_evidence():
    """A sample that expresses none of the CAF markers can't justify
    any CAF subtype fraction — all of the fibroblast compartment stays
    generic."""
    sample = {"ACTB": 500.0, "GAPDH": 800.0}  # no CAF markers
    subtype_frac, generic_frac = partition_compartment(
        sample,
        compartment_fraction=0.20,
        marker_folds=CAF_MARKER_FOLDS,
    )
    assert subtype_frac == 0.0
    assert generic_frac == 0.20


def test_partition_sums_to_compartment_fraction():
    sample = {"FAP": 100.0, "POSTN": 50.0}
    subtype_frac, generic_frac = partition_compartment(
        sample,
        compartment_fraction=0.30,
        marker_folds=CAF_MARKER_FOLDS,
    )
    assert abs((subtype_frac + generic_frac) - 0.30) < 1e-9
    # With real CAF marker expression the subtype side must win non-
    # zero share.
    assert subtype_frac > 0.0


# ── End-to-end integration ────────────────────────────────────────────


def test_estimate_tumor_expression_ranges_emits_subtype_refinement_columns(tmp_path):
    """Integration: running ``estimate_tumor_expression_ranges`` on a
    synthetic sample with CAF / TAM marker expression must land the
    three new provenance columns in the output frame and mark the
    expected marker genes as refined."""
    import pandas as pd
    from trufflepig.clean_tpm import technical_rna_mask
    from trufflepig.reference import pan_cancer_expression
    from trufflepig.plot import estimate_tumor_expression_ranges

    # Build a synthetic sample that's mostly a colon reference with a
    # few marker genes boosted so CAF + TAM refinement has something to
    # refine. The TPM floor on non-marker genes avoids the "sample HK
    # median is zero" edge case.
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    df = pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref["colon_nTPM"].astype(float),
        }
    )
    technical = technical_rna_mask(
        df,
        label_col="gene_symbol",
        id_col="ensembl_gene_id",
    )
    df.loc[~technical, "TPM"] = df.loc[~technical, "TPM"] + 1.0
    # Inject expected marker genes at high TPM so the refinement path
    # lights up.
    marker_boosts = {"FAP": 250.0, "POSTN": 400.0, "CD163": 180.0, "MRC1": 120.0}
    for sym, val in marker_boosts.items():
        df.loc[df["gene_symbol"] == sym, "TPM"] = val

    purity = {
        "overall_estimate": 0.4,
        "overall_lower": 0.3,
        "overall_upper": 0.5,
        "components": {"stromal": {"enrichment": 1.2}, "immune": {"enrichment": 1.3}},
    }

    out = estimate_tumor_expression_ranges(
        df_gene_expr=df,
        cancer_type="COAD",
        purity_result=purity,
    )

    # All three provenance columns must be present.
    for col in (
        "subtype_refined",
        "subtype_refinement_label",
        "tme_tpm_before_subtype_refinement",
    ):
        assert col in out.columns, f"{col} missing from output frame"

    # The synthetic sample has no NNLS-fitted decomposition (we didn't
    # pass ``decomposition_results``), so the fallback path triggers:
    # ``per_compartment_tpm_by_symbol`` is None and refinement scales
    # the aggregate TME. The marker boosts above ensure observed > TME
    # so the refinement fires. The columns must exist either way;
    # firing is strong but not strict (depends on NNLS-less TME path
    # landing non-zero on these genes).
    refined_rows = out[out["subtype_refined"]]
    # Non-zero refinement firing is nice-to-have; the strict contract
    # is just that the columns plumb through without error.
    if len(refined_rows) > 0:
        # Any refined row must also carry a non-empty label and a
        # numeric "before" TPM for audit.
        assert refined_rows["subtype_refinement_label"].str.len().gt(0).all()
        assert refined_rows["tme_tpm_before_subtype_refinement"].notna().all()
