"""Data-derived centroid + range-plausibility (the #83 coarse-lineage anchor).

Self-recovery tests: a cohort's own reference profile must match itself. These use
only the shipped reference matrices (no ~/data dependency), so they're
deterministic and CI-safe.
"""
import numpy as np
import pandas as pd
import pytest

from trufflepig.cancer_type_centroid import (
    _rankdata,
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


def test_rankdata_matches_average_rank_with_ties_and_nans():
    out = _rankdata(np.array([3.0, np.nan, 1.0, 3.0, 2.0]))

    assert out[0] == pytest.approx(3.5)
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(1.0)
    assert out[3] == pytest.approx(3.5)
    assert out[4] == pytest.approx(2.0)


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


def test_ranker_does_not_hallmark_veto_on_unconfident_compartment(monkeypatch):
    """A near-tie compartment call may annotate disagreement but must not delete
    cross-compartment leaves. This is the HCC1395-shaped failure mode: a basal/EMT
    BRCA profile can graze Sarcoma vs Epithelial by whole-profile rho, but the
    margin is below the action threshold, so BRCA must remain in the candidate
    trace even if the cross-compartment hallmark veto would otherwise fire."""
    import trufflepig.cancer_type_centroid as ctc
    from trufflepig.tumor_purity import rank_cancer_type_candidates

    monkeypatch.setattr(
        ctc,
        "compartment_call",
        lambda sample, _corr=None: {
            "compartment": "Sarcoma",
            "score": 0.50,
            "runner_up": "Epithelial",
            "margin": 0.001,
            "confident": False,
            "scores": pd.Series({"Sarcoma": 0.50, "Epithelial": 0.499}),
        },
    )
    monkeypatch.setattr(ctc, "hallmark_veto", lambda code, sample: code == "BRCA")

    rows = rank_cancer_type_candidates(
        _cohort_sample_df("BRCA"),
        candidate_codes=["BRCA", "ESCA"],
        top_k=4,
    )
    codes = [r["code"] for r in rows]
    assert "BRCA" in codes
    brca = next(r for r in rows if r["code"] == "BRCA")
    assert brca.get("compartment_in_set") is False
    assert rows[0].get("centroid_lineage_confident") is False
    assert rows[0].get("hallmark_vetoed") == []


def test_ranker_renormalizes_support_after_compartment_rerank(monkeypatch):
    """When a confident compartment call promotes a lower-raw-support row, the
    normalized support metric must follow the final rank order, not the demoted
    raw marker-panel winner."""
    import trufflepig.cancer_type_centroid as ctc
    from trufflepig.tumor_purity import rank_cancer_type_candidates

    monkeypatch.setattr(
        ctc,
        "compartment_call",
        lambda sample, _corr=None: {
            "compartment": "Epithelial",
            "score": 0.90,
            "runner_up": "Sarcoma",
            "margin": 0.20,
            "confident": True,
            "scores": pd.Series({"Epithelial": 0.90, "Sarcoma": 0.70}),
        },
    )
    # The restriction abstains when the centroid's single best cohort disagrees with the
    # (here forced) compartment, so stub the centroid to an Epithelial-topped series — a
    # self-consistent confident Epithelial call that legitimately floats COAD over SARC.
    monkeypatch.setattr(
        ctc, "centroid_correlations",
        lambda sample, restrict_to=None: pd.Series({"COAD": 0.90, "SARC": 0.70}),
    )
    monkeypatch.setattr(ctc, "hallmark_veto", lambda code, sample: False)

    rows = rank_cancer_type_candidates(
        _cohort_sample_df("SARC"),
        candidate_codes=["SARC", "COAD"],
        top_k=2,
    )
    by_code = {r["code"]: r for r in rows}
    assert by_code["SARC"]["support_score"] > by_code["COAD"]["support_score"]
    assert rows[0]["code"] == "COAD"
    assert rows[0]["support_fraction_of_top"] == pytest.approx(1.0)
    assert by_code["SARC"]["support_raw_fraction_of_max"] == pytest.approx(1.0)
    assert by_code["SARC"]["support_fraction_of_top"] < 1.0


def test_final_support_preserves_same_family_display_order_without_compartment():
    from trufflepig.tumor_purity import _finalize_candidate_rank_support

    rows = [
        {
            "code": "COAD",
            "family_label": "CRC",
            "support_score": 1.0,
            "signature_score": 1.0,
        },
        {
            "code": "READ",
            "family_label": "CRC",
            "support_score": 0.70,
            "signature_score": 0.70,
        },
        {
            "code": "STAD",
            "family_label": "GASTRIC",
            "support_score": 0.90,
            "signature_score": 0.90,
        },
    ]

    _finalize_candidate_rank_support(rows, compartment_restricted=False)

    assert [row["code"] for row in rows] == ["COAD", "READ", "STAD"]
    assert rows[0]["support_fraction_of_top"] == pytest.approx(1.0)
    assert rows[1]["support_fraction_of_top"] == pytest.approx(0.70)
    assert rows[2]["support_fraction_of_top"] == pytest.approx(0.90)


def test_final_support_compartment_tier_preserves_order_within_tier():
    from trufflepig.tumor_purity import _finalize_candidate_rank_support

    rows = [
        {
            "code": "SARC",
            "compartment_in_set": False,
            "support_score": 1.0,
            "signature_score": 1.0,
        },
        {
            "code": "COAD",
            "compartment_in_set": True,
            "support_score": 0.80,
            "signature_score": 0.80,
        },
        {
            "code": "READ",
            "compartment_in_set": True,
            "support_score": 0.70,
            "signature_score": 0.70,
        },
    ]

    _finalize_candidate_rank_support(rows, compartment_restricted=True)

    assert [row["code"] for row in rows] == ["COAD", "READ", "SARC"]
    assert rows[0]["support_fraction_of_top"] == pytest.approx(1.0)
    assert rows[1]["support_fraction_of_top"] < 1.0
    assert rows[2]["support_raw_fraction_of_max"] == pytest.approx(1.0)
    assert rows[2]["support_fraction_of_top"] < rows[1]["support_fraction_of_top"]


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


def test_in_compartment_honors_secondary_lineage():
    """A polyphenotypic tumor is in-compartment for its SECONDARY program too, so a
    confident compartment call that locks onto that program does not float a look-alike
    above it. HEPB (embryonal blastoma, hepatic/epithelial differentiation) reads as
    Epithelial -> it must NOT be demoted below LIHC."""
    from trufflepig.cancer_type_centroid import in_compartment

    # primary compartment still holds
    assert in_compartment("HEPB", "Embryonal")
    # secondary (epithelial/hepatic) program — the regression this fixes
    assert in_compartment("HEPB", "Epithelial")
    # synovial sarcoma is biphasic: in both Sarcoma and Epithelial
    assert in_compartment("SARC_SYN", "Sarcoma")
    assert in_compartment("SARC_SYN", "Epithelial")
    # a unilineage carcinoma gains no spurious sarcoma membership
    assert not in_compartment("COAD", "Sarcoma")


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
# Centroid-authoritative fine-subtype resolution (#98).
# --------------------------------------------------------------------------- #
def _sarc_corr(**rho):
    """A centroid Series over SARC children with the given rhos (others low)."""
    base = {
        "SARC_OS": 0.5, "SARC_LMS": 0.5, "SARC_UPS": 0.5, "SARC_DDLPS": 0.5,
        "SARC_LPS_UNSPEC": 0.5, "SARC_SYN": 0.5, "SARC_MPNST": 0.5,
    }
    base.update(rho)
    return pd.Series(base)


def test_resolve_fine_subtype_overrides_inferior_curated_label():
    """The SARC_UPS-medoid case: the centroid clearly prefers SARC_UPS over the curated
    SARC_LMS (>margin below), so the centroid overrides — even though UPS leads its
    runner-up by less than the margin (a lead gate would wrongly abstain)."""
    from trufflepig.cancer_type_centroid import resolve_fine_subtype

    cc = _sarc_corr(SARC_UPS=0.964, SARC_LPS_UNSPEC=0.961, SARC_LMS=0.943)
    assert resolve_fine_subtype("SARC", cc, current_subtype="SARC_LMS") == "SARC_UPS"


def test_resolve_fine_subtype_keeps_competitive_curated_label():
    """When the curated label is centroid-competitive with the best child (within margin),
    curation breaks the near-tie — the centroid does not overrule it."""
    from trufflepig.cancer_type_centroid import resolve_fine_subtype

    # SARC_DDLPS curated, centroid top SARC_OS only 0.004 higher -> keep DDLPS.
    cc = _sarc_corr(SARC_OS=0.960, SARC_DDLPS=0.956)
    assert resolve_fine_subtype("SARC", cc, current_subtype="SARC_DDLPS") == "SARC_DDLPS"


def test_resolve_fine_subtype_anchors_when_curated_label_wrong():
    """The pfo004 case: curated SARC_DDLPS but the whole-profile centroid is unmistakably
    SARC_OS — the centroid anchors the call onto SARC_OS."""
    from trufflepig.cancer_type_centroid import resolve_fine_subtype

    cc = _sarc_corr(SARC_OS=0.86, SARC_LMS=0.80, SARC_DDLPS=0.77)
    assert resolve_fine_subtype("SARC", cc, current_subtype="SARC_DDLPS") == "SARC_OS"


def test_resolve_fine_subtype_abstains_to_broad_without_curated_label():
    """No curated label and the centroid can't separate its top child from the field ->
    abstain to the broad call (None), never invent a fine label on noise."""
    from trufflepig.cancer_type_centroid import resolve_fine_subtype

    cc = _sarc_corr(SARC_OS=0.901, SARC_LMS=0.900)  # 0.001 lead, no curated label
    assert resolve_fine_subtype("SARC", cc, current_subtype=None) is None
    # but a clear lead does commit a fine label even with no curated anchor
    cc2 = _sarc_corr(SARC_OS=0.95, SARC_LMS=0.90)
    assert resolve_fine_subtype("SARC", cc2, current_subtype=None) == "SARC_OS"


def test_resolve_fine_subtype_no_children_keeps_current():
    """A broad call with no reference-present children leaves the label untouched."""
    from trufflepig.cancer_type_centroid import resolve_fine_subtype

    cc = pd.Series({"GBM": 0.9, "LGG": 0.8})
    assert resolve_fine_subtype("GBM", cc, current_subtype=None) is None


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
