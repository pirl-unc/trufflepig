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
    _mismatch_repair_state_for_code,
    classify_expression,
    classify_expression_hierarchy,
    classify_mismatch_repair_expression,
    classify_mismatch_repair_sibling_expression,
    mismatch_repair_context_group,
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


def _representative_sample(code, column=None):
    from oncoref.normalization import clean_tpm
    from pirlygenes.expression.accessors import representative_cohort_samples

    d = (
        representative_cohort_samples(code)
        .drop_duplicates("Ensembl_Gene_ID")
        .set_index("Ensembl_Gene_ID")
    )
    cols = [c for c in d.columns if c != "Symbol"]
    col = column or cols[0]
    expr = d[[col]].astype(float)
    gt = pd.DataFrame(
        {
            "Ensembl_Gene_ID": expr.index,
            "Symbol": d["Symbol"].reindex(expr.index).values,
        }
    )
    cleaned = clean_tpm(expr, gene_table=gt)
    cleaned.index = d["Symbol"].reindex(expr.index).values
    cleaned = cleaned[~cleaned.index.duplicated(keep="first")]
    return cleaned[col].to_dict()


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


def test_mismatch_repair_state_mapping_includes_pmmr_endometrial_classes():
    assert _mismatch_repair_state_for_code("UCEC_MSI") == "MSI"
    assert _mismatch_repair_state_for_code("UCEC_CNL") == "MSS"
    assert _mismatch_repair_state_for_code("UCEC_CNH") == "MSS"
    assert _mismatch_repair_state_for_code("UCEC_POLE") == ""


def test_mismatch_repair_release_context_gate_is_public_and_limited():
    assert mismatch_repair_context_group("COAD") == "CRC"
    assert mismatch_repair_context_group("READ_MSS") == "CRC"
    assert mismatch_repair_context_group("UCEC") == "UCEC"
    assert mismatch_repair_context_group("STAD") == "STAD"
    assert mismatch_repair_context_group("GBM") == ""
    assert mismatch_repair_context_group("PRAD") == ""


def test_hierarchical_votes_are_stage_scoped():
    votes = classify_expression_hierarchy(_bulk_sample("SKCM"), top_k=3)
    assert votes
    by_stage = {vote.stage: vote for vote in votes}
    assert {"compartment", "family", "entity", "mismatch_repair"} <= set(by_stage)
    assert by_stage["compartment"].label
    assert by_stage["entity"].label
    assert by_stage["compartment"].public_dict()["label_space"] == "learned_compartment"
    assert (
        by_stage["mismatch_repair"].public_dict()["label_space"]
        == "learned_mismatch_repair_binary"
    )


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("COAD_MSI", "MSI"),
        ("COAD_MSS", "MSS"),
        ("READ_MSI", "MSI"),
        ("READ_MSS", "MSS"),
    ],
)
def test_mismatch_repair_binary_classifier_recovers_representative_states(code, expected):
    vote = classify_mismatch_repair_expression(_representative_sample(code))

    assert vote is not None
    assert vote.label == expected
    assert {label for label, _probability in vote.predictions} <= {"MSI", "MSS"}
    assert vote.training_sample_count == 17
    assert {"COAD_MSI", "COAD_MSS", "READ_MSI", "READ_MSS"} <= set(
        vote.training_cohorts
    )


def test_mismatch_repair_release_classifier_requires_supported_context():
    sample = _representative_sample("COAD_MSI")

    assert classify_mismatch_repair_expression(sample, "GBM") is None
    vote = classify_mismatch_repair_expression(sample, "COAD")

    assert vote is not None
    assert vote.label_space == "learned_mismatch_repair_release_ensemble"
    assert vote.details["context_group"] == "CRC"
    assert "CRC" in vote.training_cohorts


@pytest.mark.parametrize(
    ("code", "context", "expected"),
    [
        ("COAD_MSI", "COAD", "MSI"),
        ("COAD_MSS", "COAD", "MSS"),
        ("READ_MSI", "READ", "MSI"),
        ("READ_MSS", "READ", "MSS"),
    ],
)
def test_mismatch_repair_sibling_vote_uses_contextual_msi_mss_labels(
    code,
    context,
    expected,
):
    vote = classify_mismatch_repair_sibling_expression(
        _representative_sample(code),
        context,
    )

    assert vote is not None
    assert vote.label == expected
    assert vote.label_space == "learned_mismatch_repair_sibling_entity"
