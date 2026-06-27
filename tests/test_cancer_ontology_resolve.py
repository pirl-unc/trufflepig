"""Regression tests for ``trufflepig.cancer_ontology.resolve_cancer_type``.

Both the oncoref resolution and the pirlygenes fallback are validated against the merged
``cancer_type_registry`` (trufflepig's supported code set), so:
  - supported entities/subtypes resolve,
  - pirlygenes-only codes (ASTB) resolve via the fallback,
  - oncoref-only aggregates dropped from the merged registry (NET, CRC_MSI) are REJECTED
    rather than returned to fail later in purity/reference lookup.
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
def test_oncoref_only_aggregates_are_rejected(aggregate):
    """Aggregates oncoref carries but trufflepig drops (no markers/reference) must NOT be
    returned — they would only fail later in purity/reference lookup. Reject at resolve time."""
    assert aggregate not in set(cancer_type_registry()["code"].astype(str))
    with pytest.raises((ValueError, KeyError)):
        resolve_cancer_type(aggregate)


def test_genuinely_unknown_label_raises():
    with pytest.raises((ValueError, KeyError)):
        resolve_cancer_type("NOT_A_REAL_CANCER_CODE_XYZ")
