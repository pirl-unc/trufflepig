"""Regression tests for #51 — SARC subtype-aware matched-normal.

The epithelial matched-normal mechanism (#50) gives every epithelial
primary an optional ``matched_normal_<tissue>`` compartment so admixed
benign parent tissue is absorbed as non-tumor signal. SARC parent is
deliberately absent from that map because it's a mixture of lineage-
distinct subtypes with different benign analogs (LMS → smooth muscle;
liposarcoma → adipose; MPNST → Schwann).

With the mixture-cohort classifier (#171) producing ``winning_subtype``
on the SARC parent row, the decomposition engine now routes
matched-normal selection per subtype.

UPS / MFS / synovial / angiosarcoma have no defensible benign
counterpart and stay on the unassigned path.
"""

from trufflepig.decomposition.templates import (
    MATCHED_NORMAL_TISSUE,
    epithelial_matched_normal_component,
    get_template_components,
    matched_normal_component,
)


def test_sarc_parent_without_subtype_has_no_matched_normal():
    """Without ``winning_subtype``, the SARC parent code alone must not
    pull any matched-normal component — the mixture parent has no single
    defensible benign analog."""
    assert matched_normal_component("SARC") is None


def test_sarc_lms_subtype_picks_smooth_muscle():
    assert (
        matched_normal_component(
            "SARC",
            winning_subtype="SARC_LMS",
        )
        == "matched_normal_smooth_muscle"
    )


def test_sarc_liposarcoma_subtypes_pick_adipose():
    for code in ("SARC_DDLPS", "SARC_WDLPS", "SARC_MYXLPS", "SARC_LPS_UNSPEC"):
        assert (
            matched_normal_component(
                "SARC",
                winning_subtype=code,
            )
            == "matched_normal_adipose_tissue"
        ), code


def test_sarc_unassigned_subtypes_have_no_matched_normal():
    """UPS / MFS / synovial / angiosarcoma / MPNST have no defensible
    benign reference in HPA and must stay on the unassigned path."""
    for code in ("SARC_UPS", "SARC_MYXFIB", "SARC_SYN", "SARC_ANGIO", "SARC_MPNST"):
        assert (
            matched_normal_component(
                "SARC",
                winning_subtype=code,
            )
            is None
        ), code


def test_epithelial_primaries_unaffected_by_subtype_override():
    """An epithelial primary keeps its existing matched-normal
    regardless of winning_subtype (which wouldn't be set for
    non-mixture parents anyway, but defend the invariant)."""
    assert matched_normal_component("PRAD") == "matched_normal_prostate"
    assert (
        matched_normal_component(
            "PRAD",
            winning_subtype=None,
        )
        == "matched_normal_prostate"
    )


def test_get_template_components_wires_subtype_through():
    """``get_template_components`` must route the winning_subtype to
    matched_normal lookup so the engine picks the correct compartment
    for NNLS fitting."""
    comps_parent = get_template_components("solid_primary", "SARC")
    assert not any(c.startswith("matched_normal_") for c in comps_parent), (
        f"SARC parent without subtype should not add matched-normal: {comps_parent}"
    )
    comps_lms = get_template_components(
        "solid_primary",
        "SARC",
        winning_subtype="SARC_LMS",
    )
    assert "matched_normal_smooth_muscle" in comps_lms
    comps_lps = get_template_components(
        "solid_primary",
        "SARC",
        winning_subtype="SARC_LPS_UNSPEC",
    )
    assert "matched_normal_adipose_tissue" in comps_lps


def test_deprecated_epithelial_alias_still_works():
    """Outside importers kept ``epithelial_matched_normal_component``
    — the alias must keep returning the parent-code result and must not
    consult winning_subtype (SARC-subtype routing is only via the new
    name)."""
    assert epithelial_matched_normal_component("PRAD") == "matched_normal_prostate"
    assert epithelial_matched_normal_component("SARC") is None
    assert (
        epithelial_matched_normal_component("SARC_LMS")
        == "matched_normal_smooth_muscle"
    )


def test_matched_normal_map_contains_expected_codes():
    """Registry invariant: epithelial primaries + the four defensible
    SARC subtypes must be present."""
    for code in ("BRCA", "PRAD", "COAD", "LUAD"):
        assert code in MATCHED_NORMAL_TISSUE
    for code in (
        "SARC_LMS",
        "SARC_DDLPS",
        "SARC_WDLPS",
        "SARC_MYXLPS",
        "SARC_LPS_UNSPEC",
    ):
        assert code in MATCHED_NORMAL_TISSUE
    # Parent SARC must remain absent so the mixture-cohort path owns it.
    assert "SARC" not in MATCHED_NORMAL_TISSUE
