"""Regression tests for #58 — extended TME refinement beyond CAF / TAM.

Same mechanism as #56 (``refine_tme_per_gene``), but adds panels for:

- exhausted / tumor-infiltrating T cells (PDCD1, CTLA4, LAG3, …)
- tumor endothelium / tip cells (DLL4, ESM1, ANGPT2, …)
- myeloid-derived suppressor cells (ARG1, S100A8/9, …)
- TLS-B germinal-center B cells (CXCR5, BCL6, AICDA, …)
- tumor-infiltrating plasma (MZB1, IGHG1, …)

The panels land on the ``T_cell`` / ``endothelial`` / ``myeloid`` /
``B_cell`` / ``plasma`` compartments respectively — all of which are
in the standard ``solid_primary`` template.
"""

from trufflepig.decomposition.subtype_refs import (
    EXHAUSTED_T_MARKER_FOLDS,
    MDSC_MARKER_FOLDS,
    PANELS,
    TI_PLASMA_MARKER_FOLDS,
    TLS_B_MARKER_FOLDS,
    TUMOR_ENDOTHELIUM_MARKER_FOLDS,
    panel_labels,
    panel_markers,
    refine_tme_per_gene,
)


# ── Panel registry ────────────────────────────────────────────────────


def test_panel_registry_covers_all_expected_labels():
    labels = set(panel_labels())
    expected = {
        "CAF",
        "TAM",
        "MDSC",
        "exhausted_T",
        "tumor_endothelium",
        "TLS_B",
        "TI_plasma",
    }
    missing = expected - labels
    assert not missing, f"missing panels: {missing}"


def test_panel_registry_compartments_are_from_solid_primary_template():
    """Every panel must target a compartment the NNLS decomposition
    actually fits. Otherwise refinement never has anything to act on."""
    from trufflepig.decomposition.templates import TEMPLATES

    solid_primary_compartments = set(TEMPLATES["solid_primary"]["components"])
    for label, compartment, _folds in PANELS:
        assert compartment in solid_primary_compartments, (
            f"{label} targets ``{compartment}`` but that compartment "
            f"is not in solid_primary template "
            f"({sorted(solid_primary_compartments)})"
        )


def test_panel_folds_are_strictly_greater_than_one():
    """A fold of 1.0 means no refinement — every entry in every panel
    must have a meaningful elevation over baseline."""
    for label, _comp, folds in PANELS:
        for gene, fold in folds.items():
            assert fold > 1.0, f"{label}/{gene} fold = {fold}"


# ── Exhausted T / checkpoint markers ──────────────────────────────────


def test_exhausted_t_panel_has_canonical_checkpoint_markers():
    panel = set(EXHAUSTED_T_MARKER_FOLDS.keys())
    for gene in ("PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT"):
        assert gene in panel, f"exhausted-T panel missing {gene}"


def test_exhausted_t_refinement_fires_on_T_cell_compartment():
    """A sample with PDCD1 expression + T_cell NNLS contribution must
    get refined; the provenance label is ``exhausted_T``."""
    tme_bg = {"PDCD1": 12.0}
    per_comp = {"PDCD1": {"T_cell": 8.0}}
    sample = {"PDCD1": 90.0}
    refined, prov = refine_tme_per_gene(tme_bg, per_comp, sample)
    assert refined["PDCD1"] > tme_bg["PDCD1"]
    assert prov["PDCD1"]["subtype"] == "exhausted_T"


def test_exhausted_t_refinement_no_op_on_wrong_compartment():
    """Same PDCD1 expression but NNLS put the signal on fibroblast —
    exhausted-T refinement targets T_cell, must no-op."""
    tme_bg = {"PDCD1": 12.0}
    per_comp = {"PDCD1": {"fibroblast": 8.0}}
    sample = {"PDCD1": 90.0}
    refined, prov = refine_tme_per_gene(tme_bg, per_comp, sample)
    assert refined["PDCD1"] == 12.0
    assert "PDCD1" not in prov


# ── Tumor endothelium ──────────────────────────────────────────────────


def test_tumor_endothelium_panel_has_angiogenic_markers():
    panel = set(TUMOR_ENDOTHELIUM_MARKER_FOLDS.keys())
    for gene in ("DLL4", "ESM1", "ANGPT2", "PLVAP"):
        assert gene in panel


def test_tumor_endothelium_refinement_on_endothelial_compartment():
    tme_bg = {"DLL4": 18.0}
    per_comp = {"DLL4": {"endothelial": 12.0}}
    sample = {"DLL4": 140.0}
    refined, prov = refine_tme_per_gene(tme_bg, per_comp, sample)
    assert refined["DLL4"] > tme_bg["DLL4"]
    assert prov["DLL4"]["subtype"] == "tumor_endothelium"


# ── MDSC (shares myeloid compartment with TAM) ─────────────────────────


def test_mdsc_panel_has_canonical_markers():
    panel = set(MDSC_MARKER_FOLDS.keys())
    for gene in ("ARG1", "S100A8", "S100A9", "MPO"):
        assert gene in panel


def test_mdsc_and_tam_do_not_share_markers():
    """A MDSC marker appearing in the TAM panel would compound refinements
    via the first-match-wins rule. Canonical markers don't overlap —
    pin that."""
    mdsc = set(MDSC_MARKER_FOLDS.keys())
    from trufflepig.decomposition.subtype_refs import TAM_MARKER_FOLDS

    tam = set(TAM_MARKER_FOLDS.keys())
    assert not (mdsc & tam), (
        f"MDSC and TAM share markers: {mdsc & tam} — compounded refinement"
    )


def test_mdsc_refinement_on_myeloid_compartment():
    tme_bg = {"ARG1": 10.0}
    per_comp = {"ARG1": {"myeloid": 6.0}}
    sample = {"ARG1": 75.0}
    refined, prov = refine_tme_per_gene(tme_bg, per_comp, sample)
    assert refined["ARG1"] > tme_bg["ARG1"]
    assert prov["ARG1"]["subtype"] == "MDSC"


# ── TLS B / TI plasma ──────────────────────────────────────────────────


def test_tls_b_panel_has_germinal_center_markers():
    panel = set(TLS_B_MARKER_FOLDS.keys())
    for gene in ("CXCR5", "BCL6", "AICDA"):
        assert gene in panel


def test_tls_b_refinement_on_B_cell_compartment():
    tme_bg = {"AICDA": 6.0}
    per_comp = {"AICDA": {"B_cell": 4.0}}
    sample = {"AICDA": 55.0}
    refined, prov = refine_tme_per_gene(tme_bg, per_comp, sample)
    assert refined["AICDA"] > tme_bg["AICDA"]
    assert prov["AICDA"]["subtype"] == "TLS_B"


def test_ti_plasma_panel_has_class_switch_markers():
    panel = set(TI_PLASMA_MARKER_FOLDS.keys())
    for gene in ("MZB1", "IGHG1", "JCHAIN"):
        assert gene in panel


def test_ti_plasma_refinement_on_plasma_compartment():
    tme_bg = {"IGHG1": 25.0}
    per_comp = {"IGHG1": {"plasma": 18.0}}
    sample = {"IGHG1": 180.0}
    refined, prov = refine_tme_per_gene(tme_bg, per_comp, sample)
    assert refined["IGHG1"] > tme_bg["IGHG1"]
    assert prov["IGHG1"]["subtype"] == "TI_plasma"


# ── First-match-wins: a gene on two panels doesn't double-refine ──────


def test_first_match_wins_on_shared_gene(monkeypatch):
    """A gene that ends up on two panels (shouldn't happen with the
    canonical markers, but pin the contract) must refine exactly once,
    labeled by the first panel in PANELS order."""
    from trufflepig.decomposition import subtype_refs as sr

    # Inject a test-only panel that would duplicate PDCD1's CAF-side
    # refinement. PANELS is iterated in order; the existing
    # ``exhausted_T`` panel is the canonical home for PDCD1 so it
    # should win.
    original = list(sr.PANELS)
    try:
        sr.PANELS = original + [
            ("FAKE_LAST", "fibroblast", {"PDCD1": 99.0}),
        ]
        tme_bg = {"PDCD1": 10.0}
        per_comp = {"PDCD1": {"T_cell": 5.0, "fibroblast": 5.0}}
        sample = {"PDCD1": 100.0}
        refined, prov = sr.refine_tme_per_gene(tme_bg, per_comp, sample)
        # First-match-wins: exhausted_T refines PDCD1 before the fake
        # panel can run.
        assert prov["PDCD1"]["subtype"] == "exhausted_T"
    finally:
        sr.PANELS = original


# ── panel_markers / panel_labels public API ───────────────────────────


def test_panel_markers_returns_marker_list_for_known_label():
    t_markers = panel_markers("exhausted_T")
    assert "PDCD1" in t_markers
    assert "CTLA4" in t_markers


def test_panel_markers_returns_empty_for_unknown_label():
    assert panel_markers("NOT_A_REAL_PANEL") == []
