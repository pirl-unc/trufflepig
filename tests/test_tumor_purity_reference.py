import pytest

from trufflepig.reference import pan_cancer_expression
from trufflepig.tumor_purity import _resolve_purity_reference


def _reference_by_symbol():
    return (
        pan_cancer_expression(technical_rna_normalize=True)
        .drop_duplicates(subset="Symbol")
        .set_index("Symbol")
    )


def test_markerless_local_purity_reference_falls_back_to_broad_parent():
    ref_by_sym = _reference_by_symbol()

    context = _resolve_purity_reference("OS", ref_by_sym)

    assert context["reference_cancer_code"] == "SARC"
    assert context["reference_expression_source"] == "parent_pan_cancer"


def test_markerless_local_purity_reference_without_broad_fallback_errors():
    ref_by_sym = _reference_by_symbol()

    with pytest.raises(ValueError, match="no direct purity marker panel"):
        _resolve_purity_reference("NUTM", ref_by_sym)
