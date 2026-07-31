"""Cancer-signature noise handling in the production within-sample space.

Near-zero genes used to clear a zero-heavy cross-cohort distribution and create
false evidence for rare-marker cancers. The production scorer now uses
within-sample percentile, where near-zero genes naturally remain low ranked.
The historical HK clamp remains available only for explicit A/B evaluation.
"""

import pandas as pd
import pytest

import trufflepig.plot_embedding as pe
from trufflepig.plot_embedding import _compute_cancer_type_signature_stats
from trufflepig.reference import pan_cancer_expression


def _cohort_sample(code: str) -> pd.DataFrame:
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"].values,
            "gene_name": ref["Symbol"].values,
            "TPM": ref[f"{code}_TPM"].values,
        }
    )


def _scores(df, basis="within_sample") -> dict[str, float]:
    return {
        row["code"]: row["score"]
        for row in _compute_cancer_type_signature_stats(
            df,
            cohort_basis=basis,
        )
    }


def test_default_signature_does_not_depend_on_hk_detection_floor():
    df = _cohort_sample("PAAD")
    with_clamp = _scores(df)

    saved = pe._SIGNATURE_DETECTION_FLOOR_HK
    pe._SIGNATURE_DETECTION_FLOOR_HK = 0.0
    try:
        without_clamp = _scores(df)
    finally:
        pe._SIGNATURE_DETECTION_FLOOR_HK = saved

    assert with_clamp == without_clamp
    assert max(with_clamp, key=with_clamp.get) == "PAAD"


def test_default_signature_never_loads_hk_feature_space(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("default signature path attempted HK normalization")

    monkeypatch.setattr(pe, "_full_cohort_hk_reference", fail)
    monkeypatch.setattr(pe, "_sample_expression_by_symbol", fail)

    scores = _scores(_cohort_sample("PAAD"))
    assert max(scores, key=scores.get) == "PAAD"


def test_default_signature_uses_a_fixed_reference_rank_universe():
    complete = _cohort_sample("PAAD")
    omitted_zeros = complete[complete["TPM"].astype(float) > 0].copy()
    annotated_zeros = pd.concat(
        [
            omitted_zeros,
            pd.DataFrame(
                {
                    "ensembl_gene_id": ["EXTRA_ZERO_1", "EXTRA_ZERO_2"],
                    "gene_name": ["EXTRA_ZERO_1", "EXTRA_ZERO_2"],
                    "TPM": [0.0, 0.0],
                }
            ),
        ],
        ignore_index=True,
    )

    expected = _scores(complete)
    assert _scores(omitted_zeros) == pytest.approx(expected)
    assert _scores(annotated_zeros) == pytest.approx(expected)


def test_explicit_hk_ab_basis_still_clamps_noise():
    df = _cohort_sample("PAAD")
    with_clamp = _scores(df, "hk")
    saved = pe._SIGNATURE_DETECTION_FLOOR_HK
    pe._SIGNATURE_DETECTION_FLOOR_HK = 0.0
    try:
        without_clamp = _scores(df, "hk")
    finally:
        pe._SIGNATURE_DETECTION_FLOOR_HK = saved
    assert with_clamp["PCPG"] < without_clamp["PCPG"]


def test_expressed_marker_type_outranks_noise_in_default_space():
    scores = _scores(_cohort_sample("SARC"))
    assert scores["SARC"] > scores["PCPG"]
