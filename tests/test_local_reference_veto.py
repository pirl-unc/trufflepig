"""Regression tests for ``_veto_local_reference_lineage_flip`` (main.py).

The veto reverts a LOCAL-EXPRESSION-REFERENCE (deconvolved) cross-lineage flip
only when TWO independent signals — the bulk classifier's pre-call AND
``compartment_call`` — agree on the original lineage.

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
        analysis = {
            "cancer_type_evidence": {
                "selected": {"code": "SARC_DDLPS"},
                "staged_evidence_graph": {
                    "selected": {
                        "code": "SARC_DDLPS",
                        "selected_by": "local_expression_reference",
                        "selects_report_label": True,
                    },
                    "channels": [
                        {
                            "candidate_code": "SARC_DDLPS",
                            "code": "SARC_DDLPS",
                            "channel": "deconvolved_tumor_reference",
                            "role": "tumor_program_reference",
                            "stage": "exact_subtype",
                            "status": "selected_report_label",
                            "support": 0.91,
                            "selects_report_label": True,
                            "details": {},
                        }
                    ],
                },
            }
        }
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
    graph = analysis["cancer_type_evidence"]["staged_evidence_graph"]
    assert graph["selected"] is None
    assert graph["vetoed_selected"]["code"] == "SARC_DDLPS"
    channel = graph["channels"][0]
    assert channel["selects_report_label"] is False
    assert channel["status"] == "vetoed"
    assert "veto_reason" in channel["details"]


def test_decision_trace_uses_veto_over_stale_graph_selection():
    """Legacy report JSON may still carry a graph-selected vetoed label."""

    body = tp_main._cancer_type_decision_trace_markdown(
        {
            "inferred_cancer_type": "READ",
            "cancer_type_evidence_vetoed": {
                "vetoed_call": "SARC_DDLPS",
                "kept_call": "READ",
                "reason": "local-expression reference conflicted with bulk context",
            },
            "cancer_type_evidence": {
                "selected": None,
                "staged_evidence_graph": {
                    "selected": {
                        "code": "SARC_DDLPS",
                        "selected_by": "local_expression_reference",
                        "selects_report_label": True,
                    },
                    "channels": [
                        {
                            "candidate_code": "SARC_DDLPS",
                            "code": "SARC_DDLPS",
                            "channel": "deconvolved_tumor_reference",
                            "role": "tumor_program_reference",
                            "stage": "exact_subtype",
                            "status": "selected_report_label",
                            "support": 0.91,
                            "selects_report_label": True,
                            "details": {},
                        }
                    ],
                },
            },
        }
    )

    assert "Evidence selection vetoed" in body
    assert "READ" in body
    assert "Selected report label: **SARC_DDLPS" not in body


def test_veto_fires_on_non_sarcoma_cross_lineage_flip(monkeypatch):
    """Generalized guard: the local-reference channel cannot flip a BRCA-like bulk call to heme
    when the confident compartment also supports epithelial/solid lineage."""
    import trufflepig.cancer_ontology as tp_onto
    import trufflepig.cancer_type_centroid as ctc
    import trufflepig.tumor_purity as tp_purity

    monkeypatch.setattr(tp_purity, "_build_sample_tpm_by_symbol", lambda df: {})
    monkeypatch.setattr(
        ctc,
        "compartment_call",
        lambda *a, **k: {"compartment": "Epithelial", "confident": True},
    )
    monkeypatch.setattr(
        tp_onto,
        "cancer_lineage_group",
        lambda code: {"BRCA": "Epithelial", "LAML": "Heme"}.get(str(code), ""),
    )

    analysis = {"cancer_type_evidence": {"selected": {"code": "LAML"}}}
    result = tp_main._veto_local_reference_lineage_flip(
        analysis,
        pd.DataFrame({"gene_id": ["X"], "tpm": [1.0]}),
        report_scope_cancer_type="LAML",
        rna_inferred_cancer_type="BRCA",
        selected_scope={"selected_by": "local_expression_reference"},
    )

    assert result is None
    assert analysis["cancer_type_evidence_vetoed"]["kept_call"] == "BRCA"


def test_veto_preserves_flip_into_embryonal_lineage(monkeypatch):
    """Carve-out: a flip INTO the embryonal lineage is a legitimate rescue, not the spurious
    sarcoma attractor the veto targets. A hepatoblastoma (HEPB) sample is systematically read as
    its solid tissue-of-origin (LIHC) by BOTH the bulk classifier and the confident solid
    compartment, so their agreement is not two independent signals. The local-reference flip to the
    embryonal call must survive — vetoing it back to LIHC would revert a correct rescue."""
    import trufflepig.cancer_ontology as tp_onto
    import trufflepig.cancer_type_centroid as ctc
    import trufflepig.tumor_purity as tp_purity

    monkeypatch.setattr(tp_purity, "_build_sample_tpm_by_symbol", lambda df: {})
    monkeypatch.setattr(
        ctc,
        "compartment_call",
        lambda *a, **k: {"compartment": "Epithelial", "confident": True},  # → solid
    )
    monkeypatch.setattr(
        tp_onto,
        "cancer_lineage_group",
        lambda code: {"LIHC": "Epithelial", "HEPB": "Embryonal"}.get(str(code), ""),
    )

    analysis = {"cancer_type_evidence": {"selected": {"code": "HEPB"}}}
    result = tp_main._veto_local_reference_lineage_flip(
        analysis,
        pd.DataFrame({"gene_id": ["X"], "tpm": [1.0]}),
        report_scope_cancer_type="HEPB",
        rna_inferred_cancer_type="LIHC",
        selected_scope={"selected_by": "local_expression_reference"},
    )

    assert result == "HEPB"  # NOT reverted — the embryonal rescue is preserved
    assert "cancer_type_evidence_vetoed" not in analysis
