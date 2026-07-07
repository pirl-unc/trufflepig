"""Tests for the OPTIONAL learned cancer-type co-signal (trufflepig.expression_classifier).

The model trains lazily on the representative cohorts (~1-2 min on first call, then cached), so this
is a small, high-value suite: it confirms the co-signal recovers confident types, spreads sensibly
within a family, and degrades safely on empty/sparse input — not an exhaustive accuracy sweep (that
lives in scripts/zscore_classifier_ab.py).
"""
import pandas as pd
import pytest

from trufflepig.expression_classifier import (
    _learned_family_for_code,
    classify_expression,
    classify_expression_hierarchy,
)
from trufflepig.reference import pan_cancer_expression
from trufflepig.tumor_purity import _build_sample_tpm_by_symbol


def _bulk_sample(code):
    ref = pan_cancer_expression().drop_duplicates("Ensembl_Gene_ID")
    df = pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{code}_TPM"].astype(float),
        }
    )
    return _build_sample_tpm_by_symbol(df)


def test_empty_input_returns_empty_without_training():
    # Must short-circuit before the (expensive) model training.
    assert classify_expression({}) == []
    assert classify_expression(None) == []


@pytest.mark.parametrize("code", ["SKCM", "LIHC"])
def test_confident_types_recovered_in_top_call(code):
    """SKCM/LIHC have sharp, specific programs — the co-signal should put the true type first."""
    top = classify_expression(_bulk_sample(code), top_k=3)
    assert top, "co-signal returned nothing (sklearn/reference unavailable?)"
    assert top[0][0] == code, f"{code}: top call was {top}"
    assert 0.0 <= top[0][1] <= 1.0


def test_family_is_recovered_for_subtype_dense_type():
    """COAD trains alongside its colorectal siblings/subtypes, so probability spreads within the
    CRC family — the top call should still be a colorectal code, not an unrelated lineage."""
    top = classify_expression(_bulk_sample("COAD"), top_k=3)
    assert top
    assert any(c.startswith(("COAD", "READ", "CRC")) for c, _ in top), top


def test_learned_family_walks_registry_ancestors_for_crc_subtypes():
    assert _learned_family_for_code("READ") == "CRC"
    assert _learned_family_for_code("READ_MSS") == "CRC"
    assert _learned_family_for_code("COAD_MSI") == "CRC"


def test_hierarchical_votes_are_stage_scoped():
    votes = classify_expression_hierarchy(_bulk_sample("SKCM"), top_k=3)
    assert votes
    by_stage = {vote.stage: vote for vote in votes}
    assert {"compartment", "family", "entity"} <= set(by_stage)
    assert by_stage["compartment"].label
    assert by_stage["entity"].label
    assert by_stage["compartment"].public_dict()["label_space"] == "learned_compartment"
