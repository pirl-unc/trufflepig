"""Data-derived centroid + range-plausibility (the #83 coarse-lineage anchor).

Self-recovery tests: a cohort's own reference profile must match itself. These use
only the shipped reference matrices (no ~/data dependency), so they're
deterministic and CI-safe.
"""
import numpy as np
import pandas as pd
import pytest

from trufflepig.cancer_type_centroid import (
    centroid_correlations,
    coarse_lineage_scores,
    range_plausibility,
)
from trufflepig.reference import (
    pan_cancer_expression,
    subtype_deconvolved_expression,
    tcga_deconvolved_expression,
)


def _bulk_cohort_as_sample(code):
    """Build a {symbol: tpm} sample from a cohort's own bulk centroid column."""
    pan = pan_cancer_expression(technical_rna_normalize=True).drop_duplicates("Symbol")
    col = f"{code}_TPM"
    return dict(zip(pan["Symbol"].astype(str), pan[col].astype(float)))


def _ref_cohort_as_sample(code):
    """Build a {symbol: tpm} sample from a cohort's own column in the (expanded,
    subtype-aware) centroid reference — the self-recovery source that matches whatever
    reference :func:`centroid_correlations` actually uses."""
    from trufflepig.cancer_type_centroid import _bulk_centroids

    bulk, _ = _bulk_centroids()
    col = np.expm1(bulk[code])
    return {str(g): float(v) for g, v in col.items()}


def _a_sarcoma_cohort():
    """A sarcoma cohort code present in the expanded reference (SARC subtypes only —
    the bare broad SARC pseudo-cohort is intentionally dropped)."""
    from trufflepig.cancer_type_centroid import _bulk_centroids
    from pirlygenes.gene_sets_cancer import cancer_lineage_group

    bulk, _ = _bulk_centroids()
    for c in bulk.columns:
        if cancer_lineage_group(str(c)) == "Sarcoma":
            return str(c)
    return None


def _dec_cohort_as_sample(code):
    """Build a {symbol: tpm} sample from a cohort's deconvolved tumor-only median."""
    dec = tcga_deconvolved_expression(technical_rna_normalize=True)
    sub = dec[dec["cancer_code"].astype(str) == code]
    return dict(zip(sub["symbol"].astype(str), sub["tumor_tpm_median"].astype(float)))


@pytest.mark.parametrize("code", ["COAD", "PRAD", "BLCA", "BRCA", "BRCA_Basal"])
def test_cohort_centroid_matches_itself(code):
    """A cohort's own reference profile must correlate highest with its own centroid —
    self-recovery against the expanded, subtype-aware reference (incl. BRCA_Basal,
    which is the reference a basal/TNBC tumor should match, not the luminal-biased
    broad BRCA bulk centroid)."""
    corr = centroid_correlations(_ref_cohort_as_sample(code))
    assert not corr.empty
    assert corr.index[0] == code, f"{code}: top centroid was {corr.index[0]}, top3={list(corr.head(3).items())}"
    # self-correlation is (near) perfect
    assert corr.iloc[0] > 0.99


def test_sarc_centroid_coarse_lineage_is_sarcoma():
    """The mesenchymal cohort resolves to the Sarcoma coarse group, not Epithelial —
    this is the data-derived signal that fixes the stroma->SARC/HNSC mis-call."""
    lin = coarse_lineage_scores(_bulk_cohort_as_sample("SARC"))
    assert not lin.empty
    assert lin.index[0] == "Sarcoma"


def test_epithelial_centroid_coarse_lineage_is_epithelial():
    lin = coarse_lineage_scores(_bulk_cohort_as_sample("COAD"))
    assert not lin.empty
    assert lin.index[0] == "Epithelial"


def test_range_plausibility_high_for_own_cohort_low_for_wrong_lineage():
    """A cohort's own tumor-only profile is plausible for itself; a cross-lineage
    type's markers are NOT in range — this is the sanity veto's signal."""
    sample = _dec_cohort_as_sample("SARC")
    assert range_plausibility("SARC", sample) > 0.6
    # HNSC (squamous) markers are implausible on a sarcoma profile
    assert range_plausibility("HNSC", sample) < range_plausibility("SARC", sample)


def test_centroid_correlations_empty_on_empty_input():
    assert centroid_correlations({}).empty
    assert coarse_lineage_scores({}).empty


def test_range_plausibility_abstains_for_unknown_code():
    # a code with no deconvolved reference returns 1.0 (abstain, never invents)
    assert range_plausibility("NOT_A_REAL_CODE", {"TP53": 10.0}) == 1.0


def _deconv_cohort_as_sample(code):
    """Tumor-only {symbol: tpm} from a cohort's deconvolved tumor median."""
    for fn in (tcga_deconvolved_expression, subtype_deconvolved_expression):
        d = fn(technical_rna_normalize=True)
        sub = d[d["cancer_code"].astype(str) == code]
        if len(sub):
            return dict(zip(sub["symbol"].astype(str), sub["tumor_tpm_median"].astype(float)))
    return {}


@pytest.mark.parametrize("code", ["COAD", "SARC", "BRCA", "NUTM"])
def test_tumor_only_cohort_matches_itself(code):
    """A cohort's own deconvolved tumor profile matches its own tumor-only centroid —
    including NUTM, which is invisible to the 33-code bulk centroids."""
    from trufflepig.cancer_type_centroid import tumor_only_correlations

    corr = tumor_only_correlations(_deconv_cohort_as_sample(code))
    assert not corr.empty
    assert corr.index[0] == code, f"{code}: top was {corr.index[0]}"
    assert corr.iloc[0] > 0.99


def test_tumor_only_coverage_exceeds_bulk_and_includes_rare_types():
    """The tumor-only space covers the subtypes / rare types the bulk centroids miss."""
    from trufflepig.cancer_type_centroid import _deconvolved_centroids

    _, codes = _deconvolved_centroids()
    cs = set(codes)
    assert len(codes) > 50  # ~63, far beyond the 33 bulk cohorts
    for code in ("NUTM", "RB"):  # absent from the 33 TCGA bulk columns
        assert code in cs, f"{code} missing from tumor-only coverage"


def test_tumor_only_correlations_empty_on_empty_input():
    from trufflepig.cancer_type_centroid import tumor_only_correlations

    assert tumor_only_correlations({}).empty


def _cohort_sample_df(code):
    """A gene-expression frame built from a cohort's own bulk centroid column."""
    ref = pan_cancer_expression(technical_rna_normalize=True).drop_duplicates(
        subset="Ensembl_Gene_ID"
    )
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{code}_TPM"].astype(float),
        }
    )


def test_ranker_annotates_centroid_crosscheck_and_stays_additive():
    """The centroid cross-check wiring in rank_cancer_type_candidates must annotate
    every candidate (additive — new keys only) and must NOT change the call.

    A cohort's own median classifies as itself (the marker-panel ranking), and on a
    SARC profile the data-derived coarse lineage is Sarcoma and agrees with the
    SARC call.
    """
    from trufflepig.tumor_purity import rank_cancer_type_candidates

    rows = rank_cancer_type_candidates(_cohort_sample_df("SARC"), top_k=5)
    assert rows
    # additive: the cross-check adds these to every candidate
    for r in rows:
        assert "centroid_correlation" in r
        assert "range_plausibility" in r
        assert 0.0 <= float(r["range_plausibility"]) <= 1.0
    top = rows[0]
    # the marker-panel call (SARC on its own median) is unchanged by the cross-check
    assert top["code"] == "SARC"
    # data-derived coarse lineage is computed and agrees with the call
    assert top.get("centroid_coarse_lineage") == "Sarcoma"
    assert top.get("centroid_lineage_agreement") is True
    assert top.get("centroid_top_code")


def test_ranker_crosscheck_flags_cross_lineage_disagreement():
    """When the call's lineage disagrees with the data-derived one, the flag is
    False — the mis-call detector. A SARC-profile sample CONSTRAINED to call COAD
    (epithelial) must flag the disagreement (Sarcoma vs Epithelial)."""
    from trufflepig.tumor_purity import rank_cancer_type_candidates

    rows = rank_cancer_type_candidates(
        _cohort_sample_df("SARC"), candidate_codes=["COAD"], top_k=3
    )
    assert rows and rows[0]["code"] == "COAD"
    # the data still says Sarcoma; the epithelial call disagrees
    assert rows[0].get("centroid_coarse_lineage") == "Sarcoma"
    assert rows[0].get("centroid_lineage_agreement") is False
    # entirely-out-of-compartment constrained set must NOT be restricted to empty
    assert rows[0].get("centroid_compartment_restricted") is False
    assert rows[0].get("compartment_in_set") is False


# --------------------------------------------------------------------------- #
# Stage 1 — compartment_call + leaf restriction.
# --------------------------------------------------------------------------- #
def test_compartment_call_sarc_profile_is_sarcoma():
    """A sarcoma subtype's own profile resolves to the Sarcoma compartment. (Whether
    it clears the *confidence* margin is subtype-dependent — a melanocytic-adjacent
    subtype like PEComa sits near Melanoma — so we assert the compartment, not the
    margin; the structured fields must still be populated.)"""
    from trufflepig.cancer_type_centroid import compartment_call

    code = _a_sarcoma_cohort()
    assert code, "no sarcoma cohort in the expanded reference"
    call = compartment_call(_ref_cohort_as_sample(code))
    assert call["compartment"] == "Sarcoma"
    assert isinstance(call["confident"], bool)
    assert call["margin"] >= 0.0
    assert call["margin"] > 0


def test_compartment_call_coad_is_confident_epithelial():
    from trufflepig.cancer_type_centroid import compartment_call

    call = compartment_call(_bulk_cohort_as_sample("COAD"))
    assert call["compartment"] == "Epithelial"
    assert call["confident"] is True


def test_compartment_call_empty_input_abstains():
    from trufflepig.cancer_type_centroid import compartment_call

    call = compartment_call({})
    assert call["compartment"] is None
    assert call["confident"] is False


def test_compartment_call_reuses_precomputed_correlations():
    """Passing _corr must give the same compartment as the from-scratch call (and
    avoids a second centroid pass in the ranker)."""
    from trufflepig.cancer_type_centroid import centroid_correlations, compartment_call

    sample = _bulk_cohort_as_sample("BLCA")
    corr = centroid_correlations(sample)
    assert compartment_call(sample, _corr=corr)["compartment"] == (
        compartment_call(sample)["compartment"]
    )


def test_in_compartment_sarcoma_is_broad():
    from trufflepig.cancer_type_centroid import in_compartment

    # every sarcoma subtype is in the Sarcoma compartment...
    assert in_compartment("SARC", "Sarcoma")
    assert in_compartment("SARC_LMS", "Sarcoma")
    # ...and a carcinoma is not
    assert not in_compartment("COAD", "Sarcoma")
    assert in_compartment("COAD", "Epithelial")
    # unknown lineage / no compartment -> fail-open (never excluded)
    assert in_compartment("COAD", None)


def test_restrict_rows_floats_in_compartment_leaf_when_confident():
    """The core stage-1 behavior: a confident compartment call floats in-compartment
    leaves above out-of-compartment ones, preserving within-tier order (stable)."""
    from trufflepig.cancer_type_centroid import restrict_rows_to_compartment

    # marker-panel order put an epithelial leaf on top of a sarcoma leaf (saturation)
    rows = [
        {"code": "HNSC"},      # out (Epithelial)
        {"code": "SARC_LMS"},  # in  (Sarcoma)
        {"code": "COAD"},      # out (Epithelial)
        {"code": "SARC"},      # in  (Sarcoma)
    ]
    out, restricted = restrict_rows_to_compartment(rows, "Sarcoma", confident=True)
    assert restricted is True
    assert [r["code"] for r in out] == ["SARC_LMS", "SARC", "HNSC", "COAD"]
    assert out[0]["compartment_in_set"] is True
    assert out[-1]["compartment_in_set"] is False


def test_restrict_rows_abstains_when_not_confident():
    from trufflepig.cancer_type_centroid import restrict_rows_to_compartment

    rows = [{"code": "HNSC"}, {"code": "SARC_LMS"}]
    out, restricted = restrict_rows_to_compartment(rows, "Sarcoma", confident=False)
    assert restricted is False
    assert [r["code"] for r in out] == ["HNSC", "SARC_LMS"]  # order untouched
    # annotation still added even when not restricting
    assert out[0]["compartment_in_set"] is False
    assert out[1]["compartment_in_set"] is True


def test_restrict_rows_never_restricts_to_empty():
    """If every candidate is out-of-compartment, do not reorder (and never empty)."""
    from trufflepig.cancer_type_centroid import restrict_rows_to_compartment

    rows = [{"code": "COAD"}, {"code": "HNSC"}]
    out, restricted = restrict_rows_to_compartment(rows, "Sarcoma", confident=True)
    assert restricted is False
    assert [r["code"] for r in out] == ["COAD", "HNSC"]


# --------------------------------------------------------------------------- #
# Hallmark-gene veto.
# --------------------------------------------------------------------------- #
def test_hallmark_fit_high_for_own_cohort():
    """A cohort's own reference profile expresses (essentially) all of its own
    hallmark markers — fit ~1.0, never vetoed."""
    from trufflepig.cancer_type_centroid import hallmark_fit, hallmark_veto

    s = _ref_cohort_as_sample("SKCM")
    assert hallmark_fit("SKCM", s) > 0.8
    assert hallmark_veto("SKCM", s) is False


def test_hallmark_veto_drops_melanoma_on_carcinoma():
    """The canonical case: a carcinoma profile has MLANA/PMEL/TYR/SOX10 ~0, so a
    Melanoma/SKCM candidate is a horrible fit and is vetoed."""
    from trufflepig.cancer_type_centroid import hallmark_fit, hallmark_veto

    coad = _ref_cohort_as_sample("COAD")
    assert hallmark_fit("SKCM", coad) < 0.2
    assert hallmark_veto("SKCM", coad) is True
    # ...but the carcinoma's own type is not vetoed
    assert hallmark_veto("COAD", coad) is False


def test_hallmark_veto_abstains_for_unknown_code():
    from trufflepig.cancer_type_centroid import hallmark_veto

    assert hallmark_veto("NOT_A_REAL_CODE", {"TP53": 50.0}) is False
