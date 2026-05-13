"""Tests for Tranche C cancer-key-genes curation (#127).

THCA, LGG, GBM, CHOL, OV, TGCT. Per the curation-bar memory, TGCT
is deliberately biomarker-only — no validated targeted therapy — so
one test specifically guards that decision.
"""

from pirlygenes.gene_sets_cancer import (
    cancer_biomarker_genes,
    cancer_therapy_targets,
    cancer_key_genes_cancer_types,
)


def test_all_tranche_c_codes_curated():
    covered = set(cancer_key_genes_cancer_types())
    for code in ("THCA", "LGG", "GBM", "CHOL", "OV", "TGCT"):
        assert code in covered, f"{code} missing from cancer-key-genes"


def test_thca_has_kinase_driver_biomarkers_and_approved_tkis():
    bm = cancer_biomarker_genes("THCA")
    # BRAF V600E, RET fusion, RAS family — the three major molecular
    # classes in PTC / FTC.
    for g in ("BRAF", "RET", "HRAS", "NRAS"):
        assert g in bm, f"THCA biomarker missing: {g}"

    tg = cancer_therapy_targets("THCA")
    approved = set(tg[tg["phase"] == "approved"]["agent"].astype(str).str.lower())
    # RAI-refractory DTC backbone
    assert "lenvatinib" in approved
    assert "sorafenib" in approved
    # BRAF V600E anaplastic
    assert any("dabrafenib" in a for a in approved)
    # RET-altered
    assert "selpercatinib" in approved or "pralsetinib" in approved


def test_lgg_has_idh_mutation_and_vorasidenib():
    """IDH1/2 is the defining biomarker for IDH-mutant LGG;
    vorasidenib is the 2024 approved targeted therapy."""
    bm = cancer_biomarker_genes("LGG")
    assert "IDH1" in bm
    assert "IDH2" in bm
    assert "ATRX" in bm  # astrocytoma lineage marker

    tg = cancer_therapy_targets("LGG")
    agents = set(tg["agent"].astype(str).str.lower())
    assert "vorasidenib" in agents


def test_gbm_biomarkers_separate_from_lgg():
    """GBM is now IDH-wildtype per 2021 WHO; MGMT + EGFR + TERT
    are the three most-asked-about biomarkers."""
    bm = cancer_biomarker_genes("GBM")
    assert "MGMT" in bm
    assert "EGFR" in bm
    assert "TERT" in bm
    assert "PTEN" in bm

    tg = cancer_therapy_targets("GBM")
    agents = set(tg["agent"].astype(str).str.lower())
    # Temozolomide (MGMT-methylated) and bevacizumab (recurrent) are the
    # two long-approved; EGFR ADCs are trial phase.
    assert "temozolomide" in agents
    assert "bevacizumab" in agents


def test_chol_has_fgfr2_and_idh1_and_approved_targeted_agents():
    """Cholangiocarcinoma has the deepest targeted-therapy portfolio
    among hepatobiliary cancers — FGFR2 fusion inhibitors
    (pemigatinib / infigratinib / futibatinib) and IDH1 inhibitor
    ivosidenib are all approved."""
    bm = cancer_biomarker_genes("CHOL")
    assert "FGFR2" in bm
    assert "IDH1" in bm

    tg = cancer_therapy_targets("CHOL")
    agents = set(tg["agent"].astype(str).str.lower())
    assert "pemigatinib" in agents
    assert "futibatinib" in agents
    assert "ivosidenib" in agents


def test_ov_has_brca_panel_and_parp_inhibitors():
    """HGSOC — BRCA / HRD + PARP inhibitor + mirvetuximab (FRα)
    is the canonical curated panel."""
    bm = cancer_biomarker_genes("OV")
    assert "BRCA1" in bm
    assert "BRCA2" in bm
    assert "FOLR1" in bm  # FRα for mirvetuximab
    assert "TP53" in bm  # near-universal in HGSOC

    tg = cancer_therapy_targets("OV")
    agents = set(tg["agent"].astype(str).str.lower())
    assert "olaparib" in agents
    assert "niraparib" in agents
    # Mirvetuximab soravtansine is listed with commercial name Elahere
    assert any("mirvetuximab" in a for a in agents)


def test_tgct_is_biomarker_only():
    """Per the curation bar: where no clinician-validated targeted
    therapy exists, leave the target panel empty rather than pad
    with speculative rows. TGCT's chemo-dominant landscape (BEP) is
    the canonical case."""
    bm = cancer_biomarker_genes("TGCT")
    assert len(bm) >= 5, f"TGCT biomarker panel too sparse: {bm}"
    assert "KIT" in bm  # seminoma marker
    assert "AFP" in bm  # non-seminoma serum

    tg = cancer_therapy_targets("TGCT")
    assert len(tg) == 0, (
        f"TGCT target panel should be empty (no validated targeted "
        f"therapy); got {list(tg['agent'])}"
    )


def test_fgfr2_vs_fgfr2_fusion_annotation_matches_indications():
    """Sanity-check that CHOL targets for FGFR2 specifically name
    the fusion requirement in the indication string — helps the
    brief render the correct eligibility."""
    tg = cancer_therapy_targets("CHOL")
    fgfr = tg[tg["symbol"] == "FGFR2"]
    for _, row in fgfr.iterrows():
        indication = str(row["indication"]).lower()
        assert "fusion" in indication, (
            f"FGFR2 target {row['agent']} indication missing 'fusion' "
            f"qualifier: {row['indication']}"
        )
