"""Regression tests for ``_veto_local_reference_lineage_flip`` (main.py).

The veto reverts a LOCAL-EXPRESSION-REFERENCE (deconvolved) flip into the sarcoma
(mesenchymal) compartment only when TWO independent signals — the bulk classifier's
pre-call AND ``compartment_call`` — agree on the *other* (non-mesenchymal) lineage.

A ``compartment_call`` abstention (``compartment=None``) or a near-tie
(``confident=False``) is NOT an independent signal: it must not be mapped through
``_group_to_mode`` (which defaults an empty label to ``"solid"``) and counted as if
the compartment had actively corroborated the bulk call. These tests pin that.
"""
from __future__ import annotations

import pandas as pd
import pytest

from trufflepig import main as tp_main


@pytest.fixture
def patched_veto(monkeypatch):
    """Patch the veto's cheap collaborators so the test is hermetic (no reference load).

    Scenario: a colon ``READ`` (epithelial → ``solid``) sample whose local-reference
    channel flipped the call to ``SARC_DDLPS`` (mesenchymal). The bulk pre-call agrees
    with epithelial; whether the veto fires hinges entirely on the compartment call.
    """
    # The veto imports its collaborators *inside* the function, so patch them at their
    # source modules (not on ``main``), or the local imports re-bind the real ones.
    import trufflepig.cancer_ontology as tp_onto
    import trufflepig.tumor_purity as tp_purity

    monkeypatch.setattr(tp_purity, "_build_sample_tpm_by_symbol", lambda df: {})
    monkeypatch.setattr(
        tp_onto, "cancer_lineage_group",
        lambda code: {"READ": "Epithelial", "SARC_DDLPS": "Sarcoma"}.get(str(code), ""),
    )

    def _run(compartment_call_result):
        import trufflepig.cancer_type_centroid as ctc

        monkeypatch.setattr(ctc, "compartment_call", lambda *a, **k: compartment_call_result)
        analysis = {"cancer_type_evidence": {"selected": {"code": "SARC_DDLPS"}}}
        return tp_main._veto_local_reference_lineage_flip(
            analysis,
            pd.DataFrame({"gene_id": ["X"], "tpm": [1.0]}),
            report_scope_cancer_type="SARC_DDLPS",
            rna_inferred_cancer_type="READ",
            selected_scope={"selected_by": "local_expression_reference"},
        ), analysis

    return _run


def test_veto_does_not_fire_on_compartment_abstention(patched_veto):
    """compartment_call abstains (compartment=None) → no second signal → keep the flip."""
    result, analysis = patched_veto({"compartment": None, "confident": False})
    assert result == "SARC_DDLPS"  # NOT reverted
    assert "cancer_type_evidence_vetoed" not in analysis


def test_veto_does_not_fire_on_unconfident_compartment(patched_veto):
    """compartment_call returns epithelial but as a near-tie (confident=False) → not an
    independent signal → keep the flip."""
    result, analysis = patched_veto({"compartment": "Epithelial", "confident": False})
    assert result == "SARC_DDLPS"
    assert "cancer_type_evidence_vetoed" not in analysis


def test_veto_fires_on_confident_epithelial_compartment(patched_veto):
    """Positive control: a CONFIDENT epithelial compartment agreeing with the epithelial
    bulk pre-call is the two-independent-signals case the veto exists for → revert."""
    result, analysis = patched_veto({"compartment": "Epithelial", "confident": True})
    assert result is None  # reverted to the bulk call
    assert analysis["cancer_type_evidence_vetoed"]["kept_call"] == "READ"
    assert analysis["inferred_cancer_type"] == "READ"
