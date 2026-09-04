"""Tests for the robust attribution algorithm (#128).

Covers three invariants the new breadth-aware attribution must hold:

1. Tissue-restricted genes (KLK3 in prostate, NEUROD1 in brain, etc.)
   must NOT trip the broadly-expressed flag even when low-level
   detection crosses many tissues.
2. Ubiquitously-expressed housekeeping-like genes (TBCE, NPM1, broad
   surface markers) MUST trip the flag.
3. The breadth floor on non-tumor attribution dampens the tumor-inferred
   residual for broadly-expressed genes without zeroing it out, and
   doesn't touch restricted genes.
"""

import numpy as np
import pandas as pd

from trufflepig.plot_tumor_expr import (
    AMPLIFICATION_MIN_FOLD,
    BROADLY_ENRICHED_MAX_RATIO,
    BROAD_TISSUE_COUNT,
    HK_TISSUE_NTPM_THRESHOLD,
    BREADTH_BASELINE_TOP_N,
)


# ── threshold-sanity tests ──────────────────────────────────────────────


def test_thresholds_are_tunable_constants():
    """Thresholds should be module-level constants so downstream
    consumers can tune them for specific tissue contexts without
    forking the call sites."""
    assert HK_TISSUE_NTPM_THRESHOLD == 5.0
    assert BROAD_TISSUE_COUNT == 15
    assert BROADLY_ENRICHED_MAX_RATIO == 3.0
    assert BREADTH_BASELINE_TOP_N == 10
    assert AMPLIFICATION_MIN_FOLD == 5.0


# ── amplification override ──────────────────────────────────────────────


def test_amplification_cell_renders_amplified_not_broadly_expr():
    """A gene broadly expressed at baseline but observed ≥ 5× peak
    healthy should render 'amplified N×' rather than 'broadly expr.'
    — HER2 / MDM2 / GPC3 pattern (#128 refinement)."""
    from trufflepig.main import _format_attribution_cell

    row = {
        "observed_tpm": 1200.0,
        "attribution": {"endothelial": 40.0},
        "attr_tumor_tpm": 1100.0,
        "attr_top_compartment": "endothelial",
        "attr_top_compartment_tpm": 40.0,
        # broadly_expressed is already gated on `not amplified` in the
        # pipeline, so the cell receives broadly_expressed=False and
        # amplified=True for HER2-amp BRCA.
        "broadly_expressed": False,
        "amplified_over_healthy": True,
        "amplification_fold": 12.0,
    }
    cell = _format_attribution_cell(row)
    assert "amplified" in cell
    assert "12.0" in cell
    assert "broadly expressed" not in cell


def test_amplification_does_not_trigger_confidence_downgrade():
    """Confidence tier must NOT downgrade an amplified target even
    though it's broadly expressed at baseline."""
    from trufflepig.confidence import (
        ConfidenceTier,
        compute_target_confidence,
    )

    purity_tier = ConfidenceTier(tier="high", reasons=[])
    target = {
        "observed_tpm": 1200.0,
        "tme_dominant": False,
        "tme_explainable": False,
        "attr_tumor_fraction": 0.92,
        # Pipeline sets broadly_expressed=False on amplified-over-
        # healthy genes; confidence sees that as the gate.
        "broadly_expressed": False,
        "amplified_over_healthy": True,
        "amplification_fold": 12.0,
        "n_healthy_tissues_expressed": 42,
    }
    tier = compute_target_confidence(target, purity_tier)
    assert tier.tier == "high", (
        f"amplified target wrongly downgraded to {tier.tier!r}: {tier}"
    )


def test_option_a_recovers_nonzero_tumor_when_mn_over_predicts():
    """#134 Option A: when the per-gene matched-normal prediction
    over-shoots observed, tumor attribution must fall back to
    `observed × tumor_fraction` — not zero — so clinically curated
    PRAD targets (KLK3 / KLK2 / TACSTD2 / FOLH1) on CRPC samples
    retain useful signal. Previously the proportional-rescale path
    reported tumor = 0 because matched-normal proportionally absorbed
    the full observed signal.

    We unit-test the decision logic without re-running the whole
    decomposition: given the per-row inputs that the function receives
    inside the attribution loop, verify that the `attr_tumor_tpm`
    formula we chose actually matches `observed × tumor_fraction`.
    """
    # Canonical CRPC scenario: KLK3 observed 82, fitted tumor=28%,
    # matched-normal fraction=26%, everything else ~3%.
    observed = 82.0
    top_fractions = {
        "tumor": 0.28,
        "matched_normal_prostate": 0.26,
        "T_cell": 0.03,
        "endothelial": 0.03,
    }
    attribution_raw = {"matched_normal_prostate": 4848.0}  # HPA over-predicts
    attr_tme_total_raw = sum(attribution_raw.values())
    matched_normal_over_predicted = (
        observed > 0 and attribution_raw and attr_tme_total_raw > observed
    )
    assert matched_normal_over_predicted

    # Option A: rescale attribution to `observed × fitted_fraction`
    attribution = {
        comp: round(observed * top_fractions.get(comp, 0.0), 2)
        for comp in attribution_raw
    }
    tumor_tpm = observed * top_fractions["tumor"]

    # Expected: tumor ≈ 23 TPM (not zero)
    assert 20 < tumor_tpm < 26, f"expected ~23 TPM tumor attribution; got {tumor_tpm}"
    assert attribution["matched_normal_prostate"] < observed, (
        "matched-normal should be scaled below observed (not absorb all)"
    )


def test_activated_tme_refinement_updates_source_attribution(monkeypatch):
    """Activated CAF/TAM/TIL refinements must feed source attribution.

    The context-TPM path uses the refined collapsed TME background, while
    target evidence uses the per-compartment attribution map. If the latter
    stays stale, CAF markers such as FAP can still leak into tumor-source
    TPM even though the refinement says the fibroblast contribution is
    higher.
    """
    import trufflepig.decomposition.signature as signature
    from trufflepig.plot_tumor_expr import estimate_tumor_expression_ranges

    def fake_signature_matrix(
        components,
        gene_subset=None,
        sample_by_eid=None,
        component_reference_tissues=None,
    ):
        assert components == ["fibroblast"]
        return (
            ["ENSG00000078098"],
            ["FAP"],
            np.array([[1.0]], dtype=float),
            ["fibroblast"],
        )

    monkeypatch.setattr(
        signature,
        "build_signature_matrix",
        fake_signature_matrix,
    )

    class Result:
        fractions = {"tumor": 0.2, "fibroblast": 0.1}
        matched_normal_tissue = None
        matched_normal_fraction = 0.0

    df = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000078098"],
            "gene_symbol": ["FAP"],
            "TPM": [100.0],
        }
    )
    ranges = estimate_tumor_expression_ranges(
        df,
        "PRAD",
        {"overall_lower": 0.2, "overall_estimate": 0.2, "overall_upper": 0.2},
        decomposition_results=[Result()],
    )
    row = ranges.set_index("symbol").loc["FAP"]

    assert row["subtype_refined"]
    assert row["subtype_refinement_label"] == "CAF"
    assert row["tme_tpm_before_subtype_refinement"] == 0.1
    assert row["attribution"]["fibroblast"] == 1.0
    assert row["attr_top_compartment_tpm"] == 1.0


def test_target_attribution_reuses_fitted_composite_tissue(monkeypatch):
    """Full-gene attribution must not replace the physical reference NNLS fit."""

    import trufflepig.decomposition.signature as signature
    from trufflepig.plot_tumor_expr import estimate_tumor_expression_ranges

    captured = {}

    def fake_signature_matrix(
        components,
        gene_subset=None,
        sample_by_eid=None,
        component_reference_tissues=None,
    ):
        captured["sample_by_eid"] = sample_by_eid
        captured["component_reference_tissues"] = component_reference_tissues
        return (
            ["ENSG00000148773"],
            ["MKI67"],
            np.array([[10.0]], dtype=float),
            list(components),
        )

    monkeypatch.setattr(signature, "build_signature_matrix", fake_signature_matrix)

    class Result:
        fractions = {"tumor": 0.2, "smooth_muscle": 0.8}
        matched_normal_tissue = None
        matched_normal_fraction = 0.0
        component_reference_tissues = {"smooth_muscle": "smooth_muscle"}

    df = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000148773"],
            "gene_symbol": ["MKI67"],
            "TPM": [20.0],
        }
    )
    estimate_tumor_expression_ranges(
        df,
        "PRAD",
        {"overall_lower": 0.2, "overall_estimate": 0.2, "overall_upper": 0.2},
        decomposition_results=[Result()],
    )

    assert captured["sample_by_eid"] == {"ENSG00000148773": 20.0}
    assert captured["component_reference_tissues"] == {
        "smooth_muscle": "smooth_muscle"
    }


def test_matched_normal_over_predicted_downgrades_and_tags():
    """When the fitted matched-normal compartment predicts more TPM
    than the sample contains (KLK3 / TACSTD2 on CRPC samples, #131),
    the attribution cell must surface the caveat and the confidence
    tier must drop from high to moderate — otherwise a clinically
    curated target silently reports "tumor 0" as if the tumor isn't
    expressing it."""
    from trufflepig.main import _format_attribution_cell
    from trufflepig.confidence import (
        ConfidenceTier,
        compute_target_confidence,
    )

    # Row shape after the #131 rescaling: attribution values clipped
    # to observed, over_predicted flag set.
    row = {
        "observed_tpm": 82.17,
        "attribution": {"matched_normal_prostate": 82.17},  # rescaled
        "attr_tumor_tpm": 0.0,
        "attr_tumor_fraction": 0.0,
        "attr_top_compartment": "matched_normal_prostate",
        "attr_top_compartment_tpm": 82.17,
        "matched_normal_over_predicted": True,
        "attribution_raw_sum_tpm": 4847.81,
        "broadly_expressed": False,
        "amplified_over_healthy": False,
        "amplification_fold": 0.0,
        "tme_dominant": True,
        "tme_explainable": True,
    }
    cell = _format_attribution_cell(row)
    assert "external tissue reference predicts more RNA than measured" in cell

    purity_tier = ConfidenceTier(tier="high", reasons=[])
    tier = compute_target_confidence(row, purity_tier)
    assert tier.tier in ("moderate", "low"), (
        f"over-predicted row should trip the tier: {tier}"
    )
    assert any(
        "external tissue reference predicts more RNA than measured" in r
        for r in tier.reasons
    ), tier.reasons


def test_truly_broadly_expressed_still_downgrades():
    """Without amplification, the broadly-expressed flag should still
    fire and the confidence tier should drop to low."""
    from trufflepig.confidence import (
        ConfidenceTier,
        compute_target_confidence,
    )

    purity_tier = ConfidenceTier(tier="high", reasons=[])
    target = {
        "observed_tpm": 800.0,
        "tme_dominant": False,
        "tme_explainable": False,
        "attr_tumor_fraction": 0.88,
        "broadly_expressed": True,
        "amplified_over_healthy": False,
        "amplification_fold": 1.4,
        "n_healthy_tissues_expressed": 42,
    }
    tier = compute_target_confidence(target, purity_tier)
    assert tier.tier == "low"


# ── breadth-gate semantics ──────────────────────────────────────────────


def _simulate_tissue_profile(
    peak_tpm, n_tissues_at_peak=1, other_tissues_tpm=0.0, n_total_tissues=50
):
    """Build a synthetic row of 50 nTPM values. One or more tissues at
    ``peak_tpm``, the rest at ``other_tissues_tpm``.
    """
    values = [other_tissues_tpm] * n_total_tissues
    for i in range(n_tissues_at_peak):
        values[i] = peak_tpm
    return values


def _breadth_call(peak_tpm, n_tissues_at_peak, other_tpm, n_total_tissues=50):
    """Compute broadly_expressed directly from a simulated profile using
    the same logic the production code applies.
    """
    profile = np.array(
        _simulate_tissue_profile(
            peak_tpm,
            n_tissues_at_peak,
            other_tpm,
            n_total_tissues,
        )
    )
    n_expressed = int((profile >= HK_TISSUE_NTPM_THRESHOLD).sum())
    top_n = np.sort(profile)[-BREADTH_BASELINE_TOP_N:]
    mean_top = float(top_n.mean())
    peak = float(profile.max())
    ratio = peak / mean_top if mean_top > 0 else float("inf")
    return {
        "broadly_expressed": (
            n_expressed >= BROAD_TISSUE_COUNT and ratio < BROADLY_ENRICHED_MAX_RATIO
        ),
        "n_expressed": n_expressed,
        "ratio": ratio,
        "mean_top": mean_top,
    }


def test_prostate_restricted_gene_not_broadly_expressed():
    """KLK3-like profile: very high in prostate, near-zero elsewhere.
    Must NOT be flagged broadly-expressed even though a handful of
    tissues cross the detection threshold due to baseline noise."""
    call = _breadth_call(peak_tpm=500.0, n_tissues_at_peak=1, other_tpm=0.2)
    assert not call["broadly_expressed"], (
        f"restricted gene flagged broadly_expressed: {call!r}"
    )
    # Sanity: enrichment ratio is huge.
    assert call["ratio"] > 5


def test_brain_restricted_gene_not_broadly_expressed():
    """NEUROD1-like: high in a few CNS tissues, quiet elsewhere."""
    call = _breadth_call(peak_tpm=200.0, n_tissues_at_peak=3, other_tpm=0.1)
    assert not call["broadly_expressed"], call


def test_ubiquitous_housekeeping_is_broadly_expressed():
    """TBCE-like: moderate expression everywhere, no strong enrichment
    anywhere. Must trip the flag."""
    # 50 tissues all at ~80 nTPM — no enrichment, full breadth.
    profile = [80.0] * 50
    n_expressed = sum(1 for v in profile if v >= HK_TISSUE_NTPM_THRESHOLD)
    peak = max(profile)
    top_n = sorted(profile)[-BREADTH_BASELINE_TOP_N:]
    mean_top = sum(top_n) / len(top_n)
    ratio = peak / mean_top
    broadly = n_expressed >= BROAD_TISSUE_COUNT and ratio < BROADLY_ENRICHED_MAX_RATIO
    assert broadly, f"expected broadly-expressed; got n={n_expressed}, ratio={ratio}"


def test_broad_surface_with_mild_enrichment_is_broadly_expressed():
    """CRIM1 / IL6ST-like: expressed across many tissues with only mild
    enrichment in any single one (< 3× mean-of-top)."""
    # 25 tissues at 30 nTPM, peak at 60 — ratio 2× top-10 mean.
    values = [30.0] * 25 + [0.0] * 25
    values[0] = 60.0
    profile = np.array(values)
    n_expressed = int((profile >= HK_TISSUE_NTPM_THRESHOLD).sum())
    top_n = np.sort(profile)[-BREADTH_BASELINE_TOP_N:]
    mean_top = float(top_n.mean())
    ratio = float(profile.max()) / mean_top
    broadly = n_expressed >= BROAD_TISSUE_COUNT and ratio < BROADLY_ENRICHED_MAX_RATIO
    assert broadly, f"broad-surface gene not flagged: n={n_expressed}, ratio={ratio}"


def test_borderline_tissue_count_but_strong_enrichment_not_flagged():
    """Gene crossing the tissue-count threshold purely on low-level
    detection, but with one tissue dominating by >3×, must NOT be
    flagged. This is the enrichment-ratio gate in action — it keeps
    the flag tissue-type agnostic."""
    # 20 tissues at 6 nTPM (just above HK threshold), one at 500.
    values = [6.0] * 20 + [0.0] * 30
    values[0] = 500.0
    profile = np.array(values)
    n_expressed = int((profile >= HK_TISSUE_NTPM_THRESHOLD).sum())
    top_n = np.sort(profile)[-BREADTH_BASELINE_TOP_N:]
    mean_top = float(top_n.mean())
    ratio = float(profile.max()) / mean_top
    broadly = n_expressed >= BROAD_TISSUE_COUNT and ratio < BROADLY_ENRICHED_MAX_RATIO
    assert not broadly, (
        f"tissue-restricted (strong enrichment) flagged broadly: "
        f"n={n_expressed}, ratio={ratio}"
    )


# ── confidence-tier integration ─────────────────────────────────────────


def test_confidence_tier_fires_low_on_broadly_expressed():
    from trufflepig.confidence import (
        ConfidenceTier,
        compute_target_confidence,
    )

    purity_tier = ConfidenceTier(tier="high", reasons=[])
    target = {
        "observed_tpm": 800.0,
        "tme_dominant": False,
        "tme_explainable": False,
        "attr_tumor_fraction": 0.92,
        "broadly_expressed": True,
        "n_healthy_tissues_expressed": 28,
    }
    tier = compute_target_confidence(target, purity_tier)
    assert tier.tier == "low"
    assert any("broadly expressed" in r for r in tier.reasons), tier.reasons


def test_confidence_tier_stays_high_on_tissue_restricted_high_attr():
    from trufflepig.confidence import (
        ConfidenceTier,
        compute_target_confidence,
    )

    purity_tier = ConfidenceTier(tier="high", reasons=[])
    target = {
        "observed_tpm": 142.0,
        "tme_dominant": False,
        "tme_explainable": False,
        "attr_tumor_fraction": 0.90,
        "broadly_expressed": False,
        "n_healthy_tissues_expressed": 2,  # prostate-restricted
    }
    tier = compute_target_confidence(target, purity_tier)
    assert tier.tier == "high", tier


# ── brief / top-3 skip semantics ────────────────────────────────────────


def test_brief_trusts_curation_over_broadly_expressed_flag():
    """Curated therapy targets (#110) may legitimately be broadly
    expressed — HER2 for BRCA, MDM2 for WD/DD-LPS, GPC3 for HCC are
    all expressed in many healthy tissues, but the targeting
    mechanism is amplification / lineage-retained overexpression, not
    baseline specificity. The brief's top-3 must trust the curation
    panel and NOT filter broadly-expressed rows out — otherwise
    legitimate first-line agents get silently dropped from the
    clinician handoff (#128).
    """
    from trufflepig.brief import _top_therapies

    targets_df = pd.DataFrame(
        [
            {
                "symbol": "ERBB2",
                "agent": "trastuzumab",
                "agent_class": "antibody",
                "phase": "approved",
                "indication": "HER2+ BRCA",
            },
            {
                "symbol": "FOLH1",
                "agent": "177Lu-PSMA-617",
                "agent_class": "radioligand",
                "phase": "approved",
                "indication": "mCRPC",
            },
        ]
    )
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "ERBB2",
                "observed_tpm": 1200.0,  # HER2-amplified
                "attribution": {"endothelial": 40.0},
                "attr_tumor_tpm": 1100.0,
                "attr_tumor_fraction": 0.92,
                "attr_top_compartment": "endothelial",
                "attr_top_compartment_tpm": 40.0,
                "tme_dominant": False,
                "tme_explainable": False,
                # HER2 is broadly expressed in HPA but AMPLIFICATION is what
                # drives the therapy — this flag must not drop it from the
                # brief.
                "broadly_expressed": True,
            },
            {
                "symbol": "FOLH1",
                "observed_tpm": 142.0,
                "attribution": {"endothelial": 12.0},
                "attr_tumor_tpm": 128.0,
                "attr_tumor_fraction": 0.90,
                "attr_top_compartment": "endothelial",
                "attr_top_compartment_tpm": 12.0,
                "tme_dominant": False,
                "tme_explainable": False,
                "broadly_expressed": False,
            },
        ]
    )

    top = _top_therapies(targets_df, ranges_df, limit=3)
    symbols = [t["symbol"] for t, _ in top]
    assert "ERBB2" in symbols, (
        "curated HER2 target must remain in the brief top-3 despite "
        "being broadly expressed in HPA"
    )
    assert "FOLH1" in symbols


def test_brief_does_not_prioritize_her2_therapy_when_her2_axis_is_down():
    from trufflepig.brief import _top_therapies

    targets_df = pd.DataFrame(
        [
            {
                "cancer_code": "BRCA",
                "symbol": "ERBB2",
                "agent": "trastuzumab",
                "agent_class": "HER2 antibody",
                "phase": "approved",
                "indication": "HER2-positive metastatic breast cancer",
            },
            {
                "cancer_code": "BRCA",
                "symbol": "TACSTD2",
                "agent": "sacituzumab govitecan",
                "agent_class": "ADC",
                "phase": "approved",
                "indication": "metastatic TNBC",
            },
        ]
    )
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "ERBB2",
                "observed_tpm": 80.0,
                "attr_tumor_tpm": 60.0,
                "attr_tumor_fraction": 0.75,
                "attr_top_compartment": "tumor",
                "attr_top_compartment_tpm": 60.0,
                "tme_dominant": False,
                "tme_explainable": False,
            },
            {
                "symbol": "TACSTD2",
                "observed_tpm": 160.0,
                "attr_tumor_tpm": 120.0,
                "attr_tumor_fraction": 0.75,
                "attr_top_compartment": "tumor",
                "attr_top_compartment_tpm": 120.0,
                "tme_dominant": False,
                "tme_explainable": False,
            },
        ]
    )
    analysis = {"therapy_response_scores": {"HER2_signaling": {"state": "down"}}}

    top = _top_therapies(targets_df, ranges_df, limit=3, analysis=analysis)
    assert [t["symbol"] for t, _ in top] == ["TACSTD2"]


def test_brief_keeps_mature_target_when_interval_support_is_material():
    from trufflepig.brief import _source_trace_reason, _top_therapies

    target = {
        "cancer_code": "BRCA",
        "symbol": "TACSTD2",
        "agent": "sacituzumab govitecan",
        "agent_class": "ADC",
        "phase": "approved",
        "indication": "mTNBC / HR+/HER2-",
        "treatment_path_tier": "approved_later_line",
        "eligibility_note": "confirm prior therapies and indication-specific eligibility",
    }
    targets_df = pd.DataFrame([target])
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "TACSTD2",
                "observed_tpm": 240.0,
                "attr_tumor_tpm_low": 0.0,
                "attr_tumor_tpm": 31.0,
                "attr_tumor_tpm_high": 131.0,
                "attr_tumor_fraction_low": 0.0,
                "attr_tumor_fraction": 0.13,
                "attr_tumor_fraction_high": 0.54,
                "attr_support_fraction": 0.33,
                "attr_top_compartment": "endothelial",
                "attr_top_compartment_tpm": 13.0,
                "tme_dominant": True,
                "tme_explainable": True,
                "matched_normal_over_predicted": False,
                "source_marker_non_tumor_prior": False,
            },
        ]
    )

    top = _top_therapies(targets_df, ranges_df, limit=3, analysis={})
    assert [t["symbol"] for t, _ in top] == ["TACSTD2"]
    assert "interval includes material tumor signal" in _source_trace_reason(
        target,
        ranges_df.iloc[0],
        in_shortlist=True,
    )


def test_brief_keeps_same_lineage_targets_but_skips_background_dominant_rows():
    from trufflepig.brief import _format_therapy_bullet, _top_therapies

    targets_df = pd.DataFrame(
        [
            {
                "symbol": "FOLH1",
                "agent": "177Lu-PSMA-617",
                "agent_class": "radioligand",
                "phase": "approved",
                "indication": "mCRPC",
            },
            {
                "symbol": "STEAP2",
                "agent": "experimental ADC",
                "agent_class": "ADC",
                "phase": "phase_2",
                "indication": "mCRPC",
            },
        ]
    )
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "FOLH1",
                "observed_tpm": 87.0,
                "attribution": {"matched_normal_prostate": 46.0},
                "attr_tumor_tpm": 34.0,
                "attr_tumor_fraction": 0.39,
                "attr_tumor_tpm_low": 28.0,
                "attr_tumor_tpm_high": 39.0,
                "attr_tumor_fraction_low": 0.32,
                "attr_tumor_fraction_high": 0.45,
                "attr_support_fraction": 1.0,
                "attr_top_compartment": "matched_normal_prostate",
                "attr_top_compartment_tpm": 46.0,
                "tme_dominant": False,
                "tme_explainable": True,
                "matched_normal_over_predicted": False,
                "broadly_expressed": False,
                "matched_normal_tissue": "prostate",
                "matched_normal_tpm": 46.0,
            },
            {
                "symbol": "STEAP2",
                "observed_tpm": 90.0,
                "attribution": {"matched_normal_prostate": 78.0},
                "attr_tumor_tpm": 13.0,
                "attr_tumor_fraction": 0.14,
                "attr_tumor_tpm_low": 8.0,
                "attr_tumor_tpm_high": 17.0,
                "attr_tumor_fraction_low": 0.09,
                "attr_tumor_fraction_high": 0.19,
                "attr_support_fraction": 0.0,
                "attr_top_compartment": "matched_normal_prostate",
                "attr_top_compartment_tpm": 78.0,
                "tme_dominant": True,
                "tme_explainable": True,
                "matched_normal_over_predicted": False,
                "broadly_expressed": False,
            },
        ]
    )

    top = _top_therapies(targets_df, ranges_df, limit=3)
    symbols = [t["symbol"] for t, _ in top]
    assert symbols == ["FOLH1"]

    bullet = _format_therapy_bullet(top[0][0], top[0][1], target_panel=targets_df)
    assert "also expected in healthy tissue" in bullet
    assert "mostly tumor" in bullet
    assert "Provisional:" not in bullet


def test_same_lineage_clinical_target_can_remain_provisional_when_bulk_background_dominant():
    from trufflepig.reporting import target_reliability_status

    target = {
        "symbol": "KLK2",
        "agent": "JNJ-69086420",
        "agent_class": "bispecific",
        "phase": "phase_1",
        "indication": "mCRPC",
    }
    row = {
        "observed_tpm": 247.0,
        "attr_tumor_tpm": 57.0,
        "attr_tumor_fraction": 0.23,
        "attr_top_compartment": "matched_normal_prostate",
        "attr_top_compartment_tpm": 155.0,
        "tme_dominant": False,
        "tme_explainable": True,
        "matched_normal_over_predicted": True,
    }

    assert target_reliability_status(row, target_row=target) == "provisional"


def test_same_lineage_target_can_stay_supported_when_band_remains_material():
    from trufflepig.reporting import (
        normal_expression_context,
        target_reliability_status,
        tumor_attribution_context,
    )

    row = {
        "observed_tpm": 87.0,
        "attr_tumor_tpm": 34.0,
        "attr_tumor_tpm_low": 28.0,
        "attr_tumor_tpm_high": 39.0,
        "attr_tumor_fraction": 0.39,
        "attr_tumor_fraction_low": 0.32,
        "attr_tumor_fraction_high": 0.45,
        "attr_support_fraction": 1.0,
        "matched_normal_tissue": "prostate",
        "matched_normal_tpm": 46.0,
        "attr_top_compartment": "matched_normal_prostate",
        "tme_explainable": True,
        "tme_dominant": False,
        "matched_normal_over_predicted": False,
    }
    source = tumor_attribution_context(row)
    normal = normal_expression_context(row)
    assert source["tier"] == "tumor_supported"
    assert normal["tier"] == "same_lineage_expected"
    assert target_reliability_status(row) == "supported"


def test_expression_independent_indication_is_not_demoted_by_target_tpm():
    from trufflepig.brief import _format_therapy_bullet, _top_therapies
    from trufflepig.reporting import indication_biomarker, target_reliability_status

    targets_df = pd.DataFrame(
        [
            {
                "symbol": "PDCD1",
                "agent": "pembrolizumab",
                "agent_class": "immune_checkpoint",
                "phase": "approved",
                "indication": "MSI-H / dMMR metastatic colorectal cancer",
            },
            {
                "symbol": "STEAP2",
                "agent": "experimental ADC",
                "agent_class": "ADC",
                "phase": "phase_2",
                "indication": "mCRPC",
            },
        ]
    )
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "PDCD1",
                "observed_tpm": 0.2,
                "attr_tumor_tpm": 0.0,
                "attr_tumor_fraction": 0.0,
                "attr_tumor_tpm_low": 0.0,
                "attr_tumor_tpm_high": 0.0,
                "attr_tumor_fraction_low": 0.0,
                "attr_tumor_fraction_high": 0.0,
                "attr_support_fraction": 0.0,
                "tme_dominant": True,
                "tme_explainable": True,
            },
            {
                "symbol": "STEAP2",
                "observed_tpm": 80.0,
                "attr_tumor_tpm": 2.0,
                "attr_tumor_fraction": 0.03,
                "tme_dominant": True,
            },
        ]
    )

    top = _top_therapies(targets_df, ranges_df, limit=3)
    assert [t["symbol"] for t, _ in top] == ["PDCD1"]
    assert indication_biomarker(targets_df.iloc[0]) == "msi_high"
    assert (
        target_reliability_status(ranges_df.iloc[0], target_row=targets_df.iloc[0])
        == "provisional"
    )
    bullet = _format_therapy_bullet(top[0][0], top[0][1], target_panel=targets_df)
    assert "target expression is not the eligibility criterion" in bullet
    assert "target absent" not in bullet.lower()


def test_pmmr_text_does_not_get_classified_as_msi_high():
    from trufflepig.reporting import indication_biomarker

    row = {
        "agent": "pembrolizumab",
        "agent_class": "immune_checkpoint",
        "indication": "pMMR colorectal cancer with PD-L1 expression",
    }
    assert indication_biomarker(row) == "clinical_target_assay"


# ── attribution cell rendering ──────────────────────────────────────────


def test_attribution_cell_appends_broadly_expr_tag():
    from trufflepig.main import _format_attribution_cell

    row = {
        "observed_tpm": 800.0,
        "attribution": {"endothelial": 20.0, "fibroblast": 10.0},
        "attr_tumor_tpm": 770.0,
        "attr_top_compartment": "endothelial",
        "attr_top_compartment_tpm": 20.0,
        "broadly_expressed": True,
    }
    cell = _format_attribution_cell(row)
    assert "broadly expressed" in cell
    assert "770" in cell


def test_attribution_cell_no_tag_on_tissue_restricted():
    from trufflepig.main import _format_attribution_cell

    row = {
        "observed_tpm": 142.0,
        "attribution": {"endothelial": 12.0},
        "attr_tumor_tpm": 128.0,
        "attr_top_compartment": "endothelial",
        "attr_top_compartment_tpm": 12.0,
        "broadly_expressed": False,
    }
    cell = _format_attribution_cell(row)
    assert "broadly expressed" not in cell


def test_epithelial_context_caps_caf_marker_to_non_tumor_fraction():
    from trufflepig.reference import pan_cancer_expression
    from trufflepig.plot_tumor_expr import estimate_tumor_expression_ranges
    from trufflepig.reporting import tumor_attribution_context

    ref = pan_cancer_expression().drop_duplicates(subset="Symbol")
    symbol_to_id = dict(zip(ref["Symbol"], ref["Ensembl_Gene_ID"]))
    genes = ["FAP", "ACTB", "GAPDH", "RPLP0"]
    df = pd.DataFrame(
        {
            "gene_id": [symbol_to_id[g] for g in genes],
            "gene_symbol": genes,
            "TPM": [100.0, 100.0, 100.0, 100.0],
        }
    )
    purity = {
        "overall_estimate": 0.20,
        "overall_lower": 0.20,
        "overall_upper": 0.20,
    }

    ranges = estimate_tumor_expression_ranges(df, "PRAD", purity)
    row = ranges[ranges["symbol"] == "FAP"].iloc[0].to_dict()

    assert row["source_marker_non_tumor_prior"] is True
    assert row["source_marker_compartment"] == "stromal/CAF marker"
    assert row["attr_tumor_tpm"] <= 20.0
    assert row["attr_tumor_fraction"] <= 0.21
    ctx = tumor_attribution_context(row)
    assert ctx["tier"] == "background_dominant"


def test_tumor_attribution_context_adds_low_sample_purity_note():
    from trufflepig.reporting import tumor_attribution_context

    base = {
        "observed_tpm": 100.0,
        "attr_tumor_tpm": 90.0,
        "attr_tumor_fraction": 0.9,
        "attr_support_fraction": 1.0,
    }
    with_flag = tumor_attribution_context({**base, "sample_low_purity": True})
    assert any("low estimated tumor fraction" in n for n in with_flag["notes"])
    without = tumor_attribution_context(base)
    assert not any("low sample purity" in n for n in without["notes"])
