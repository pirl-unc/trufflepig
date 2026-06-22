"""Regression test for issue #160 — family-factor penalty miscalls
orphan-family cancer types.

Before the fix, the classifier's geomean multiplied a `family_factor`
that dropped orphan cancer types (BLCA, PAAD — no family grouping) to
~0.15 whenever a family-matched competitor sat near the top family.
That made the TCGA-median of BLCA itself classify as ESCA (squamous
family) despite BLCA having the highest signature, purity, and
lineage scores.

The fix has two guards: a raw-signal dominance override for clean cohort
medians, and a Step-0 TCGA/normal-tissue context rescue for admixed samples
where an orphan's direct cancer evidence is strong but family scoring would
otherwise prefer a broad-family competitor.
"""

import pandas as pd
import pytest

from trufflepig.reference import pan_cancer_expression
from trufflepig.tumor_purity import (
    TUMOR_PURITY_PARAMETERS,
    CANCER_TO_TISSUE,
    _CANCER_FAMILY_BY_CODE,
    _apply_coarse_tcga_orphan_rescue,
    _candidate_codes_from_family_panels,
    rank_cancer_type_candidates,
)


def _cohort_median_sample(code: str) -> pd.DataFrame:
    """Synthesize a pseudo-sample whose TPM column is the TCGA cohort
    median for ``code``. Equivalent to what the median battery uses."""
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{code}_TPM"].astype(float),
        }
    )


def test_family_panel_candidate_quota_skips_unmapped_families():
    """Unmapped local/rare family panels must not consume candidate slots.

    PFO002 StringTie had strong CRC family evidence, but higher-scoring
    unmapped panels consumed ``candidate_panel_top_n`` before CRC could add
    COAD/READ to the first-pass candidate pool.
    """
    family_params = dict(TUMOR_PURITY_PARAMETERS["family_scoring"])
    family_params["candidate_panel_top_n"] = 2
    family_params["candidate_panel_min_score"] = 0.05

    codes = _candidate_codes_from_family_panels(
        [
            ("SELLAR_EPITHELIAL", 2.4),
            ("NERVE_SHEATH", 0.47),
            ("MELANOCYTIC", 0.46),
            ("CRC", 0.42),
            ("GASTRIC", 0.35),
        ],
        family_params=family_params,
        soft_families=set(),
        top_family_score=0.46,
        top_signature_family=None,
    )

    assert codes == ["SKCM", "UVM", "COAD", "READ"]


@pytest.mark.parametrize("code", ["BLCA"])
def test_orphan_family_cohort_median_classifies_as_itself(code):
    """BLCA median → top candidate must be BLCA, not the family
    competitor that was winning before the fix (ESCA)."""
    df = _cohort_median_sample(code)
    ranked = rank_cancer_type_candidates(df)
    assert ranked, "classifier returned no candidates"
    top = ranked[0]
    assert top["code"] == code, (
        f"{code} median miscalled as {top['code']}. Rows: "
        + ", ".join(f"{r['code']}({r['support_geomean']:.3f})" for r in ranked[:5])
    )


def test_orphan_dominance_override_raises_orphan_family_factor():
    """When an orphan's raw signal dominates the family winner's by
    ≥ 1.5×, the override resets its family_factor to 1.0 and
    recomputes support_score / support_geomean."""
    df = _cohort_median_sample("BLCA")
    ranked = rank_cancer_type_candidates(df)
    blca_row = next((r for r in ranked if r["code"] == "BLCA"), None)
    assert blca_row is not None, "BLCA missing from candidate trace"
    # Post-override BLCA (orphan) should not be down-weighted.
    assert blca_row["family_factor"] >= 0.9, (
        f"BLCA family_factor={blca_row['family_factor']} — orphan "
        "dominance override did not fire; expected ≥ 0.9"
    )


def test_non_orphan_family_matched_candidates_keep_family_factor():
    """The override only fires for orphans. A family-matched candidate
    (e.g. COAD → CRC family) should NOT have its family_factor
    changed by this code path — the within-family boost/penalty logic
    is independent."""
    df = _cohort_median_sample("COAD")
    ranked = rank_cancer_type_candidates(df)
    coad_row = next((r for r in ranked if r["code"] == "COAD"), None)
    assert coad_row is not None
    # COAD has a family label — override shouldn't touch it.
    assert coad_row["family_label"] is not None
    # And classifier still picks COAD as top.
    assert ranked[0]["code"] == "COAD"


def test_override_rejects_orphan_with_tme_driven_raw_signal():
    """Guard: the override must NOT fire when an orphan's apparent raw
    signal is TME-driven rather than tumor-driven. On a COAD + lymph-
    node 30/70 mix, DLBC's raw_signal (sig × purity × lineage) can
    exceed COAD's because the lymph-node TME matches DLBC's panel —
    but DLBC's purity is ~0.33 and signature is ~0.60, below the
    override's signature + purity gates. Classifier should keep COAD
    (or READ) on top, not DLBC."""
    from trufflepig.reference import pan_cancer_expression

    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    coad = pd.Series(
        ref["COAD_TPM"].astype(float).values,
        index=ref["Ensembl_Gene_ID"].values,
    )
    lymph = pd.Series(
        ref["lymph_node_nTPM"].astype(float).values,
        index=ref["Ensembl_Gene_ID"].values,
    )
    mix = 0.3 * coad + 0.7 * lymph
    df_mix = pd.DataFrame(
        {
            "ensembl_gene_id": mix.index,
            "gene_symbol": ref.set_index("Ensembl_Gene_ID")
            .loc[mix.index, "Symbol"]
            .values,
            "TPM": mix.values,
        }
    )
    ranked = rank_cancer_type_candidates(
        df_mix,
        candidate_codes=["COAD", "READ", "DLBC", "THYM", "SARC", "STAD"],
        top_k=6,
    )
    top_code = ranked[0]["code"]
    assert top_code in {"COAD", "READ"}, (
        f"COAD+lymph mix miscalled as {top_code} — override fired "
        f"despite TME bleed-through. Rows: "
        + ", ".join(
            f"{r['code']}(sig={r['signature_score']:.2f},pur={r['purity_estimate']:.2f},"
            f"ff={r['family_factor']:.2f},gm={r['support_geomean']:.2f})"
            for r in ranked[:4]
        )
    )


def _rank_row(
    code,
    *,
    support_score,
    signature_score,
    purity_estimate,
    lineage_support_factor,
    family_label=None,
    family_factor=1.0,
    signature_stability=0.3,
):
    return {
        "code": code,
        "support_score": support_score,
        "support_geomean": support_score ** 0.2,
        "signature_score": signature_score,
        "purity_estimate": purity_estimate,
        "lineage_support_factor": lineage_support_factor,
        "family_label": family_label,
        "family_factor": family_factor,
        "signature_stability": signature_stability,
    }


class _TissueSignal:
    cancer_hint = "tumor-consistent"

    def __init__(self, top_tcga, top_normal):
        self.top_tcga_cohorts = top_tcga
        self.top_normal_tissues = top_normal


@pytest.mark.parametrize(
    "orphan_code,competitor_code,competitor_family",
    [
        ("BLCA", "ESCA", "ESCA_SQ"),
        ("PAAD", "STAD", "GASTRIC"),
        ("LIHC", "SARC", "MESENCHYMAL"),
        ("LUAD", "LUSC", "SQUAMOUS"),
    ],
)
def test_coarse_tcga_context_rescues_orphan_from_family_penalty(
    orphan_code, competitor_code, competitor_family
):
    """An orphan TCGA code can have stronger direct evidence and Step-0
    context, but still lose to a family-coded candidate after the orphan
    family penalty. The context rescue must be generic, not BLCA-specific."""

    rows = [
        _rank_row(
            competitor_code,
            support_score=0.0379,
            signature_score=0.741,
            purity_estimate=0.390,
            lineage_support_factor=0.488,
            family_label=competitor_family,
            family_factor=0.811,
        ),
        _rank_row(
            orphan_code,
            support_score=0.0348,
            signature_score=0.767,
            purity_estimate=0.741,
            lineage_support_factor=0.923,
            family_factor=0.150,
            signature_stability=0.442,
        ),
    ]
    tissue = CANCER_TO_TISSUE[orphan_code]
    signal = _TissueSignal(
        [(f"{orphan_code}_TPM", 0.84), (f"{competitor_code}_TPM", 0.82)],
        [(f"{tissue}_nTPM", 0.79)],
    )

    rescued = _apply_coarse_tcga_orphan_rescue(
        rows,
        TUMOR_PURITY_PARAMETERS["family_scoring"],
        tissue_signal=signal,
    )

    assert rescued[0]["code"] == orphan_code
    assert rescued[0]["family_factor"] == 1.0
    assert rescued[0]["support_override"]["kind"] == "coarse_tcga_orphan_context"
    assert rescued[0]["support_override"]["context_basis"] == "normal_tissue_match"


def test_all_tcga_codes_are_either_family_mapped_or_context_rescuable_orphans():
    """Pin the curation contract: family panels are intentionally partial.

    Codes outside ``_CANCER_FAMILY_BY_CODE`` must still have a primary tissue
    mapping so Step-0 tissue context can rescue them from the non-family
    penalty when appropriate.
    """

    codes = sorted(
        c.removesuffix("_TPM")
        for c in pan_cancer_expression().columns
        if c.endswith("_TPM")
    )
    orphans = [code for code in codes if code not in _CANCER_FAMILY_BY_CODE]
    assert orphans, "test setup should include orphan-family TCGA codes"
    missing_tissue = [code for code in orphans if code not in CANCER_TO_TISSUE]
    assert not missing_tissue


def test_coarse_tcga_context_rescue_allows_met_site_background_by_raw_dominance():
    """PFO017-liver shape: a BLCA liver-met sample may have liver as the top
    normal-tissue background. Strong BLCA raw evidence can still rescue the
    orphan BLCA row without claiming urinary-bladder host tissue matched."""

    rows = [
        _rank_row(
            "SARC",
            support_score=0.032,
            signature_score=0.62,
            purity_estimate=0.45,
            lineage_support_factor=0.60,
            family_label="MESENCHYMAL",
            family_factor=0.811,
        ),
        _rank_row(
            "BLCA",
            support_score=0.030,
            signature_score=0.77,
            purity_estimate=0.74,
            lineage_support_factor=0.92,
            family_factor=0.150,
            signature_stability=0.442,
        ),
    ]
    signal = _TissueSignal(
        [("BLCA_TPM", 0.84), ("SARC_TPM", 0.82)],
        [("liver_nTPM", 0.91)],
    )

    rescued = _apply_coarse_tcga_orphan_rescue(
        rows,
        TUMOR_PURITY_PARAMETERS["family_scoring"],
        tissue_signal=signal,
    )

    assert rescued[0]["code"] == "BLCA"
    assert rescued[0]["support_override"]["context_basis"] == "raw_signal_dominance"
    assert rescued[0]["support_override"]["expected_tissue"] == "urinary_bladder"
    assert rescued[0]["support_override"]["top_normal_tissue"] == "liver"


def test_coarse_tcga_context_does_not_rescue_family_coded_sarc_signal():
    """ASY shape: SARC can be the coarse TCGA/mesenchymal signal, but it is
    not an orphan penalized by family scoring, so this rescue must not fire."""

    rows = [
        _rank_row(
            "READ",
            support_score=0.0537,
            signature_score=0.545,
            purity_estimate=0.448,
            lineage_support_factor=0.961,
            family_label="CRC",
            family_factor=1.0,
        ),
        _rank_row(
            "SARC",
            support_score=0.0213,
            signature_score=0.518,
            purity_estimate=0.721,
            lineage_support_factor=0.900,
            family_label="MESENCHYMAL",
            family_factor=0.250,
        ),
    ]
    signal = _TissueSignal(
        [("SARC_TPM", 0.81)],
        [("smooth_muscle_nTPM", 0.86)],
    )

    rescued = _apply_coarse_tcga_orphan_rescue(
        rows,
        TUMOR_PURITY_PARAMETERS["family_scoring"],
        tissue_signal=signal,
    )

    assert rescued[0]["code"] == "READ"
    assert "support_override" not in rescued[1]
