"""Regression tests for ``trufflepig.cancer_ontology.resolve_cancer_type``.

Both the oncoref resolution and the pirlygenes fallback are validated against the merged
``cancer_type_registry`` (trufflepig's supported code set), so:
  - supported entities/subtypes resolve,
  - pirlygenes-only codes (ASTB) resolve via the fallback,
  - typed member-union aggregates (NET, CRC_MSI) resolve as supported references.
"""
import pytest

from trufflepig.cancer_ontology import cancer_type_registry, resolve_cancer_type


def test_supported_entities_and_subtypes_resolve():
    for code in ("COAD", "BRCA", "SARC_LMS", "BRCA_Basal", "LUAD_EGFR"):
        assert resolve_cancer_type(code) == code


def test_pirlygenes_only_code_resolves_via_fallback():
    """ASTB is pirlygenes-only (oncoref#221) but in the merged registry — must resolve even if
    the installed oncoref can't (it self-deactivates once oncoref>=1.8.17 ships ASTB)."""
    assert "ASTB" in set(cancer_type_registry()["code"].astype(str))
    assert resolve_cancer_type("ASTB") == "ASTB"


@pytest.mark.parametrize("aggregate", ["NET", "CRC_MSI"])
def test_member_union_aggregates_are_supported(aggregate):
    """Typed member-union references are operational even without a single source cohort."""
    registry = cancer_type_registry().set_index("code")
    assert registry.loc[aggregate, "reference_source"] == "member_union"
    assert resolve_cancer_type(aggregate) == aggregate


def test_genuinely_unknown_label_raises():
    with pytest.raises((ValueError, KeyError)):
        resolve_cancer_type("NOT_A_REAL_CANCER_CODE_XYZ")
