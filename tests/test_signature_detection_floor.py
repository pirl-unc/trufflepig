"""Detection-floor fix for *zero-floor percentile inflation* in the cancer-type
candidate screen (``plot_embedding._compute_cancer_type_signature_stats``).

Named pathology: a *specific* marker is near-zero across non-target cancers, so
its cross-cancer reference distribution is a floor of zeros — and ANY nonzero
sample value, including quantifier noise, clears it and scores ~1.0 percentile,
false evidence of expression. This inflates rare-marker types (PCPG) and deflates
stromal/ubiquitous-marker types (SARC). The fix clamps a signature gene the sample
does not detectably express (sample_hk below ``_SIGNATURE_DETECTION_FLOOR_HK``) to
the neutral 0.5 so it cannot contribute false positive evidence.

These tests pin the fix directly (the regression it guards against: a future edit
silently deletes the clamp and the suite stays green).
"""

import pandas as pd

import trufflepig.plot_embedding as pe
from trufflepig.plot_embedding import _compute_cancer_type_signature_stats
from trufflepig.reference import pan_cancer_expression


def _cohort_sample(code: str) -> pd.DataFrame:
    """A synthetic sample = a cohort's own clean-TPM column, so it genuinely
    expresses that type's biology and NOT the markers of unrelated types."""
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"].values,
            "gene_name": ref["Symbol"].values,
            "TPM": ref[f"{code}_TPM"].values,
        }
    )


def _scores(df) -> dict[str, float]:
    return {r["code"]: r["score"] for r in _compute_cancer_type_signature_stats(df)}


def test_detection_floor_strips_noise_only_evidence():
    """An epithelial PAAD sample does not express PCPG's neuroendocrine markers;
    without the clamp those near-zero values clear the cross-cancer zero floor and
    inflate PCPG. The clamp strips that inflation, while a type the sample
    genuinely expresses (PAAD) is essentially unaffected."""
    df = _cohort_sample("PAAD")
    with_clamp = _scores(df)

    saved = pe._SIGNATURE_DETECTION_FLOOR_HK
    pe._SIGNATURE_DETECTION_FLOOR_HK = 0.0  # disable the clamp
    try:
        without_clamp = _scores(df)
    finally:
        pe._SIGNATURE_DETECTION_FLOOR_HK = saved

    # the noise-driven rare-marker type loses its inflated score ...
    assert with_clamp["PCPG"] < without_clamp["PCPG"]
    # ... while the genuinely-expressed type is largely preserved. The modern
    # PAAD panel still has a small number of low-tail genes, so the clamp can
    # trim a little support, but it should not erase the expressed lineage.
    assert with_clamp["PAAD"] >= 0.95 * without_clamp["PAAD"]


def test_detection_floor_lets_expressed_marker_type_outrank_noise():
    """On a SARC cohort sample, the genuinely-expressed mesenchymal signature must
    out-score a rare-marker type (PCPG) that would otherwise win on noise — the
    core of the alvin regression, at the screen level."""
    scores = _scores(_cohort_sample("SARC"))
    assert scores["SARC"] > scores["PCPG"]
