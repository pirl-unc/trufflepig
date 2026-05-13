"""Tests for SARC subtype-aware cancer-key-genes curation (#126)."""

from pirlygenes.gene_sets_cancer import (
    cancer_biomarker_genes,
    cancer_therapy_targets,
    cancer_key_genes_cancer_types,
    cancer_key_genes_subtypes,
)


def test_sarc_is_in_curated_codes():
    assert "SARC" in cancer_key_genes_cancer_types()


def test_sarc_subtypes_present():
    subtypes = cancer_key_genes_subtypes("SARC")
    expected = {
        "leiomyosarcoma",
        "dedifferentiated_liposarcoma",
        "myxoid_liposarcoma",
        "synovial_sarcoma",
        "dsrct",
        "gist",
        "ewing_sarcoma",
    }
    missing = expected - set(subtypes)
    assert not missing, f"missing subtypes: {missing}"


def test_synovial_sarcoma_has_afami_cel_target():
    """afami-cel is the first FDA-approved engineered TCR-T — the
    canonical synovial-sarcoma targeted therapy. Missing this row
    means the curation is materially incomplete."""
    tg = cancer_therapy_targets("SARC", subtype="synovial_sarcoma")
    agents = set(tg["agent"].astype(str))
    assert any("afami-cel" in a or "Tecelra" in a for a in agents), (
        f"expected afami-cel / Tecelra in synovial_sarcoma targets; got {agents}"
    )
    # Target gene for afami-cel is MAGE-A4
    mage = tg[tg["symbol"] == "MAGE-A4"]
    assert len(mage) >= 1


def test_gist_has_four_approved_kit_inhibitors():
    """GIST has a well-established approved KIT/PDGFRA targeted-therapy
    ladder — imatinib / sunitinib / regorafenib / ripretinib / avapritinib
    at minimum should be present."""
    tg = cancer_therapy_targets("SARC", subtype="gist")
    agents = set(tg[tg["phase"] == "approved"]["agent"].astype(str).str.lower())
    expected = {"imatinib", "sunitinib", "regorafenib", "ripretinib", "avapritinib"}
    missing = expected - agents
    assert not missing, f"missing approved GIST agents: {missing}"


def test_gist_biomarkers_include_kit_and_sdh():
    bm = cancer_biomarker_genes("SARC", subtype="gist")
    # KIT is near-universal; SDH-deficient WT GIST is a distinct
    # subtype that the curation should flag.
    assert "KIT" in bm
    assert "PDGFRA" in bm
    assert any(g in bm for g in ("SDHA", "SDHB"))


def test_dedifferentiated_liposarcoma_has_mdm2_amplification_and_targets():
    """MDM2 amplification is pathognomonic for WDLPS/DDLPS; targeted
    MDM2-p53 antagonists (brigimadlin / milademetan) and CDK4/6
    inhibitors are the active target class."""
    bm = cancer_biomarker_genes("SARC", subtype="dedifferentiated_liposarcoma")
    assert "MDM2" in bm
    assert "CDK4" in bm

    tg = cancer_therapy_targets("SARC", subtype="dedifferentiated_liposarcoma")
    agents = set(tg["agent"].astype(str))
    assert any("brigimadlin" in a or "milademetan" in a for a in agents), (
        f"expected brigimadlin / milademetan in DDLPS targets; got {agents}"
    )


def test_myxoid_liposarcoma_has_fus_ddit3_and_ct_antigen_targets():
    """Myxoid LPS: FUS-DDIT3 is pathognomonic; NY-ESO-1 (CTAG1B) and
    MAGE-A4 are TCR-T targets with active trials."""
    bm = cancer_biomarker_genes("SARC", subtype="myxoid_liposarcoma")
    assert "DDIT3" in bm
    assert "CTAG1B" in bm  # NY-ESO-1
    assert "MAGE-A4" in bm


def test_synovial_has_ss18_fusion_marker():
    bm = cancer_biomarker_genes("SARC", subtype="synovial_sarcoma")
    assert "SS18" in bm
    assert "TLE1" in bm  # diagnostic IHC


def test_dsrct_has_ewsr1_wt1_fusion_markers():
    bm = cancer_biomarker_genes("SARC", subtype="dsrct")
    assert "EWSR1" in bm
    assert "WT1" in bm


def test_ewing_has_fusion_and_cd99():
    bm = cancer_biomarker_genes("SARC", subtype="ewing_sarcoma")
    assert "EWSR1" in bm
    assert "CD99" in bm
    assert "FLI1" in bm


def test_subtype_filter_returns_only_matching_rows():
    """A subtype filter must isolate rows; no cross-contamination
    between leiomyosarcoma and synovial_sarcoma markers."""
    lms = set(cancer_biomarker_genes("SARC", subtype="leiomyosarcoma"))
    syn = set(cancer_biomarker_genes("SARC", subtype="synovial_sarcoma"))
    # LMS markers (DES, CALD1) shouldn't leak into synovial panel.
    assert "CALD1" not in syn
    # Synovial-specific markers (SS18, TLE1) shouldn't leak into LMS.
    assert "SS18" not in lms
    assert "TLE1" not in lms


def test_no_subtype_arg_returns_all_sarc_rows():
    """Back-compat: ``cancer_biomarker_genes("SARC")`` without a
    subtype filter returns the union — useful when the report renders
    before subtype classification."""
    all_sarc = set(cancer_biomarker_genes("SARC"))
    gist_only = set(cancer_biomarker_genes("SARC", subtype="gist"))
    lms_only = set(cancer_biomarker_genes("SARC", subtype="leiomyosarcoma"))
    assert gist_only.issubset(all_sarc)
    assert lms_only.issubset(all_sarc)


def test_subtype_filter_on_non_sarc_returns_existing_rows_unchanged():
    """The subtype kwarg must be opt-in; PRAD / BRCA / etc. don't
    populate the subtype column and should return their full panel
    when subtype is unset."""
    prad = cancer_biomarker_genes("PRAD")
    assert len(prad) >= 10
    # With a subtype that doesn't exist, return empty (correct semantics)
    assert cancer_biomarker_genes("PRAD", subtype="nonexistent") == []
