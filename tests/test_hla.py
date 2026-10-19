"""Clinical HLA requirements preserve nomenclature and exclusion precedence."""

import json

import pandas as pd
import pytest

from trufflepig.hla import (
    evaluate_hla_eligibility,
    extract_hla_types_from_text,
    normalize_hla_type,
    parse_hla_types,
    protein_group_reference,
)
from trufflepig.brief import recommend_therapies
from trufflepig.reporting import (
    cancer_therapy_panel_for_analysis,
    hla_eligibility_context,
    target_hla_eligibility,
)


@pytest.mark.parametrize("raw, expected", [
    ("HLA-A0201", "A*02:01"),
    ("A02:01", "A*02:01"),
    ("A*02:01:01:01", "A*02:01:01:01"),
    ("A*02:01P", "A*02:01P"),
    ("A*02:01:01G", "A*02:01:01G"),
    ("A*02:01:01:02N", "A*02:01:01:02N"),
    ("A*02:01:01:01L", "A*02:01:01:01L"),
])
def test_hla_normalization_retains_resolution_and_annotations(raw, expected):
    assert normalize_hla_type(raw) == expected
    assert parse_hla_types([raw]) == [expected]


@pytest.mark.parametrize("raw", ["not typed", "A*02:01/A*02:05", "DQB1*06:02", "garbage"])
def test_supplied_invalid_or_ambiguous_typing_is_not_silently_discarded(raw):
    with pytest.raises(ValueError, match="Invalid or ambiguous HLA"):
        parse_hla_types(["A*02:01", raw])


def test_prose_restrictions_use_the_same_nomenclature():
    text = "Requires **HLA-A*02:01P** or A*24:02:01, not trial SARC021 or AEWS1221."
    assert extract_hla_types_from_text(text) == ["A*02:01P", "A*24:02:01"]


@pytest.mark.parametrize("supplied, status", [
    (["A*02:01"], "matched"),
    (["A*02:02P"], "matched"),
    (["A*02:03"], "matched"),
    (["A*02:06P"], "matched"),
    (["A*02:01:01:01"], "matched"),
    (["A*02:05"], "excluded"),
    (["A*02:05P"], "excluded"),
    (["A*02:01", "A*02:05"], "excluded"),
    (["A*02:05:01:01", "A*02:06"], "excluded"),
    (["A*02"], "insufficient_resolution"),
    (["A*02:01", "A*02"], "insufficient_resolution"),
    (["A*02:01N"], "mismatched"),
    (["A*02:01N", "A*02:06"], "matched"),
    (["A*02:01G"], "insufficient_resolution"),
    (["A*24:02"], "mismatched"),
    ([], "unknown"),
])
def test_real_afami_panel_enforces_hla_policy(supplied, status, clinical_hla_context, clinical_magea4_assay):
    context = clinical_hla_context(supplied)
    context["assays"].append(clinical_magea4_assay().public_dict())
    analysis = {
        "cancer_type": "SARC_SYN",
        "cancer_type_source": "user-specified",
        "clinical_context": context,
    }
    _, subtype, panel = cancer_therapy_panel_for_analysis("SARC_SYN", analysis)
    ranges = pd.DataFrame([{
        "symbol": "MAGEA4", "observed_tpm": 100.0,
        "attr_tumor_tpm": 90.0, "attr_tumor_tpm_low": 80.0,
        "attr_tumor_tpm_high": 100.0, "attr_tumor_fraction": 0.9,
        "attr_tumor_fraction_low": 0.8, "attr_tumor_fraction_high": 1.0,
        "attr_support_fraction": 1.0, "attr_top_compartment": "tumor",
        "tme_dominant": False, "tme_explainable": False,
    }])
    row = panel.loc[panel.agent.eq("afami-cel (Tecelra)")].iloc[0]
    result = target_hla_eligibility(row, analysis=analysis)
    assert result["status"] == status
    assert result["excluded"] == ["A*02:05P"]
    assert result["nomenclature_version"] == "IPD-IMGT/HLA 3.65.0"
    assert result["source"].startswith("https://www.fda.gov/")
    assert json.loads(json.dumps(result)) == result
    selected = recommend_therapies(panel, ranges, analysis=analysis, panel_subtype=subtype)
    assert any(r.therapy["agent"] == row.agent for r in selected) == (status == "matched")
    if status == "excluded":
        context = hla_eligibility_context(row, analysis=analysis)
        assert "HLA exclusion" in context and "A*02:05P" in context
        assert result["source"] in context


def test_protein_group_uses_published_membership_not_a_prefix():
    groups, _ = protein_group_reference()
    # A*02:09 belongs to the A*02:01P binding-domain protein group.
    assert groups[("A", ("02", "09"), ())] == {"A*02:01P"}
    result = evaluate_hla_eligibility(["A*02:09"], ["A*02:01P"])
    assert result.status == "matched"


def test_annotated_allele_cannot_satisfy_ordinary_surface_hla():
    result = evaluate_hla_eligibility(["A*02:01N"], ["A*02:01"])
    assert result.status == "mismatched"
    assert result.supplied == ("A*02:01N",)


def test_unrestricted_targets_do_not_require_hla():
    assert evaluate_hla_eligibility([], []).status == "not_hla_restricted"
