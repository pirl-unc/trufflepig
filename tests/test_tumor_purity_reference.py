import pytest

from trufflepig.reference import pan_cancer_expression
from trufflepig.tumor_purity import (
    _resolve_purity_reference,
    _use_estimate_component,
)


def _reference_by_symbol():
    return (
        pan_cancer_expression(technical_rna_normalize=True)
        .drop_duplicates(subset="Symbol")
        .set_index("Symbol")
    )


def test_markerless_local_purity_reference_falls_back_to_broad_parent():
    ref_by_sym = _reference_by_symbol()

    context = _resolve_purity_reference("SARC_OS", ref_by_sym)

    assert context["reference_cancer_code"] == "SARC"
    assert context["reference_expression_source"] == "parent_pan_cancer"


def test_markerless_local_purity_reference_without_broad_fallback_errors(monkeypatch):
    # A code with a local deconvolved reference but no purity markers AND no broad
    # fallback must error (not silently return a bad reference). pirlygenes now
    # ships markers or a broad fallback for every real code that used to hit this
    # (e.g. NUTM gained a purity panel), so force the scenario: NUTM keeps its
    # local deconvolved ref, but with markers/fallback patched off it must raise.
    import trufflepig.tumor_purity as tp

    monkeypatch.setattr(tp, "_has_direct_purity_markers", lambda code: False)
    monkeypatch.setattr(tp, "_broad_purity_fallback_code", lambda code: "")
    ref_by_sym = _reference_by_symbol()

    with pytest.raises(ValueError, match="no direct purity marker panel"):
        tp._resolve_purity_reference("NUTM", ref_by_sym)


def test_parent_pan_cancer_fallback_keeps_estimate_component():
    stromal_genes = ["COL1A1", "LUM"]

    assert _use_estimate_component("pan_cancer", stromal_genes)
    assert _use_estimate_component("parent_pan_cancer", stromal_genes)
    assert not _use_estimate_component("subtype_deconvolved", stromal_genes)
    assert not _use_estimate_component("observed_bulk_reference", stromal_genes)
    assert not _use_estimate_component("parent_pan_cancer", [])
