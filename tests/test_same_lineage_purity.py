"""Osteosarcoma same-lineage source stress tests."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from trufflepig.main import _same_lineage_purity_limit, _same_lineage_target_rank_rows
from trufflepig.plot_tumor_expr import estimate_tumor_expression_ranges
from trufflepig.reporting import target_reliability_status, tumor_attribution_context


def _decomposition(tumor, fibroblast):
    return SimpleNamespace(
        fractions={"tumor": tumor, "fibroblast": fibroblast},
        component_reference_tissues=None,
        matched_normal_tissue=None,
        matched_normal_fraction=0.0,
    )


def test_stable_osteosarcoma_mixture_needs_no_same_lineage_stress_test():
    purity = {
        "reference_cancer_type": "SARC",
        "overall_estimate": 0.72,
        "overall_lower": 0.61,
        "overall_upper": 0.81,
    }
    assert _same_lineage_purity_limit(
        "SARC_OS", [_decomposition(0.72, 0.18)], purity
    ) is None


def test_ambiguous_osteosarcoma_reallocation_widens_target_source_bands(monkeypatch):
    import trufflepig.decomposition.signature as signature
    import trufflepig.plot_tumor_expr as mod

    decomp = _decomposition(0.12, 0.88)
    purity = {
        "reference_cancer_type": "SARC",
        "overall_estimate": 0.12,
        "overall_lower": 0.06,
        "overall_upper": 0.34,
    }
    limit = _same_lineage_purity_limit("SARC_OS", [decomp], purity)
    assert limit["scenario_lower"] == 0.06
    assert limit["scenario_upper"] == 1.0
    assert [round(row["fraction"], 2) for row in limit["scenario_estimates"]] == [
        0.12, 0.56, 1.0,
    ]
    purity["same_lineage_limit"] = limit

    def signature_matrix(components, **_kwargs):
        assert components == ["fibroblast"]
        return [], ["PTK2", "FDPS"], np.array([[30.0], [100.0]]), components

    monkeypatch.setattr(signature, "build_signature_matrix", signature_matrix)
    sample = pd.DataFrame({
        "ensembl_gene_id": ["ENSG00000169398", "ENSG00000160752"],
        "gene_symbol": ["PTK2", "FDPS"],
        "TPM": [500.0, 20.0],
    })
    try:
        ranges = estimate_tumor_expression_ranges(
            sample, "SARC", purity, decomposition_results=[decomp],
            expression_reference_type="SARC_OS",
        ).set_index("symbol")
    finally:
        mod._exact_expression_tpm_reference.cache_clear()
    stable = ranges.loc["PTK2"]
    variable = ranges.loc["FDPS"]

    assert stable["same_lineage_scenario_status"] == "stable_material"
    assert variable["same_lineage_scenario_status"] == "scenario_dependent"
    assert stable["attr_tumor_tpm_high"] > stable["attr_tumor_tpm"]
    assert variable["attr_tumor_tpm_high"] > variable["attr_tumor_tpm"]
    assert variable["same_lineage_scenario_tpm"][0] < variable["same_lineage_scenario_tpm"][-1]
    assert target_reliability_status(variable) == "provisional"
    assert "same-lineage" in tumor_attribution_context(variable)["band"]


def test_target_source_rank_changes_are_explicit_across_allocations():
    rows = {
        "A": {"same_lineage_scenario_tpm": [30, 30, 30],
              "same_lineage_scenario_status": "stable_material"},
        "B": {"same_lineage_scenario_tpm": [5, 20, 40],
              "same_lineage_scenario_status": "scenario_dependent"},
        "C": {"same_lineage_scenario_tpm": [10, 10, 10],
              "same_lineage_scenario_status": "stable_material"},
    }
    ranked = _same_lineage_target_rank_rows(rows, rows)
    assert [(row["symbol"], row["ranks"]) for row in ranked] == [
        ("A", [1, 1, 2]), ("C", [2, 3, 3]), ("B", [3, 2, 1]),
    ]
