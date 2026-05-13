"""Tests for APL (acute promyelocytic leukaemia) subtype curation
within LAML (#23 partial).

APL is the chemo-free sub-entity of AML — PML-RARA fusion, ATRA +
arsenic trioxide as the standard of care. It needs its own subtype
row set because the treatment landscape diverges sharply from the
rest of AML (FLT3 / IDH / KMT2A / venetoclax blocks)."""

from pirlygenes.gene_sets_cancer import (
    cancer_biomarker_genes,
    cancer_therapy_targets,
    cancer_key_genes_subtypes,
)


def test_laml_has_apl_subtype():
    assert "apl" in cancer_key_genes_subtypes("LAML")


def test_apl_biomarkers_include_pml_rara_fusion_partners():
    bm = cancer_biomarker_genes("LAML", subtype="apl")
    assert "PML" in bm
    assert "RARA" in bm
    # FLT3-ITD co-mutation is the canonical hyperleukocytosis flag
    # within APL — separate from the broader FLT3 biomarker role in
    # non-APL AML.
    assert "FLT3" in bm


def test_apl_targets_include_atra_and_arsenic_trioxide():
    tg = cancer_therapy_targets("LAML", subtype="apl")
    agents = set(tg["agent"].astype(str).str.lower())
    assert "tretinoin" in agents
    assert "arsenic trioxide" in agents
    approved_agents = set(
        tg[tg["phase"] == "approved"]["agent"].astype(str).str.lower()
    )
    assert "tretinoin" in approved_agents
    assert "arsenic trioxide" in approved_agents


def test_apl_subtype_filter_excludes_non_apl_laml_rows():
    """Subtype filter must isolate APL rows — FLT3 / IDH inhibitors
    that belong to the broader AML panel should not leak into the
    APL subtype view."""
    apl_targets = cancer_therapy_targets("LAML", subtype="apl")
    agents = set(apl_targets["agent"].astype(str).str.lower())
    # These are broader-AML agents, not APL-specific.
    assert "midostaurin" not in agents
    assert "gilteritinib" not in agents
    assert "ivosidenib" not in agents
    assert "venetoclax" not in agents


def test_laml_unfiltered_still_includes_both_apl_and_non_apl():
    """Back-compat: unfiltered cancer_therapy_targets('LAML') returns
    the union including APL + the rest of AML."""
    all_laml = cancer_therapy_targets("LAML")
    agents = set(all_laml["agent"].astype(str).str.lower())
    assert "tretinoin" in agents
    assert "midostaurin" in agents  # non-APL
