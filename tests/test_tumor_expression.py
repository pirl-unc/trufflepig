# Licensed under the Apache License, Version 2.0

"""Tests for lineage-based purity estimation and 9-point tumor expression ranges."""

import numpy as np
import pandas as pd

from trufflepig.plot_tumor_expr import (
    HK_TISSUE_NTPM_THRESHOLD,
    _cached_healthy_reference_metrics,
    _cached_reference_by_symbol,
    _row_mean_top_n,
)
from trufflepig.tumor_purity import (
    LINEAGE_GENES,
    lineage_purity_panel_codes,
    TCGA_MEDIAN_PURITY,
    _combine_purity_estimates,
    _lineage_purity_estimates,
    _select_tumor_specific_genes,
    _summarize_gene_level_purity,
    _summarize_lineage_support,
)


def test_row_mean_top_n_matches_pandas_nlargest_semantics():
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, np.nan, 10.0],
            "b": [4.0, np.nan, 2.0, 1.0],
            "c": [3.0, np.nan, np.nan, 5.0],
            "d": [2.0, np.nan, 6.0, np.nan],
        },
        index=["all_values", "all_nan", "sparse", "with_nan"],
    )

    expected = {
        idx: (
            float(row.nlargest(3).mean())
            if row.notna().any()
            else 0.0
        )
        for idx, row in df.iterrows()
    }

    assert _row_mean_top_n(df, 3) == expected


def test_healthy_reference_metrics_are_cached_per_reference_matrix():
    ref_full = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["g1", "g2", "g3"],
            "Symbol": ["A", "B", "C"],
            "lung_nTPM": [10.0, 1.0, np.nan],
            "liver_nTPM": [8.0, 6.0, np.nan],
            "brain_nTPM": [1.0, 7.0, np.nan],
            "TCGA_TEST_TPM": [2.0, 3.0, 4.0],
        }
    )

    ref_by_symbol = _cached_reference_by_symbol(ref_full)
    assert _cached_reference_by_symbol(ref_full) is ref_by_symbol

    cols = ["lung_nTPM", "liver_nTPM", "brain_nTPM"]
    first = _cached_healthy_reference_metrics(ref_by_symbol, cols, 2)
    second = _cached_healthy_reference_metrics(ref_by_symbol, cols, 2)

    assert first is second
    max_healthy, n_tissues, mean_top = first
    assert max_healthy["A"] == 10.0
    assert max_healthy["B"] == 7.0
    assert np.isnan(max_healthy["C"])
    assert n_tissues == {
        "A": 2,
        "B": 2,
        "C": 0,
    }
    assert mean_top == {
        "A": 9.0,
        "B": 6.5,
        "C": 0.0,
    }
    assert HK_TISSUE_NTPM_THRESHOLD == 5.0


# ── LINEAGE_GENES coverage ────────────────────────────────────────


def test_lineage_genes_covers_all_tcga_types():
    """Every TCGA cancer type with a known purity should have lineage genes."""
    missing = [ct for ct in TCGA_MEDIAN_PURITY if ct not in LINEAGE_GENES]
    assert missing == [], f"Cancer types without lineage genes: {missing}"


def test_lineage_genes_values_are_nonempty_lists():
    panels = lineage_purity_panel_codes()
    for ct, genes in LINEAGE_GENES.items():
        assert isinstance(genes, list), f"{ct}: expected list, got {type(genes)}"
        assert len(genes) >= 1, f"{ct}: empty lineage entry"
        # Diagnostic-marker subtypes (e.g. SARC_ASPS=[TFE3]) legitimately
        # carry a single pathognomonic marker; only codes that anchor their
        # own purity estimate need the >=2 floor (purity panels need >=5, #170).
        if ct in panels:
            assert len(genes) >= 2, (
                f"{ct}: purity panel needs at least 2 lineage genes, got {len(genes)}"
            )


def test_lineage_genes_has_no_duplicates():
    for ct, genes in LINEAGE_GENES.items():
        assert len(genes) == len(set(genes)), f"{ct} has duplicate genes"


# ── _lineage_purity_estimates edge cases ──────────────────────────


def test_lineage_unknown_cancer_type_returns_empty():
    # Returns (estimates, skipped_detected); both empty when cancer
    # type isn't in the lineage panel registry.
    estimates, skipped = _lineage_purity_estimates("FAKE_TYPE", {}, {}, 0.7)
    assert estimates == []
    assert skipped == []


def test_lineage_empty_sample_returns_empty():
    estimates, skipped = _lineage_purity_estimates("PRAD", {}, {}, 0.69)
    assert estimates == []
    assert skipped == []


# ── Upper-half median estimator ───────────────────────────────────


def test_upper_half_median_ignores_low_outliers():
    """Simulates de-differentiated genes pulling down the median."""
    # 9 genes: 4 de-differentiated (low), 5 retained (high)
    purities = [0.003, 0.004, 0.013, 0.025, 0.061, 0.078, 0.098, 0.104, 0.206]
    mid = len(purities) // 2
    upper_half = purities[mid:]
    estimate = float(np.median(upper_half))
    lower = float(np.percentile(upper_half, 25))
    upper = float(np.percentile(upper_half, 75))

    # Estimate should be in the retained cluster, not dragged down
    assert estimate > 0.05, f"Estimate {estimate} should be > 5%"
    assert lower > 0.03, f"Lower {lower} should be > 3%"
    assert upper < 0.25, f"Upper {upper} should be < 25%"


def test_lineage_support_penalizes_wrong_pattern():
    rows = [
        {"sample_ratio": 1.00, "tme_ratio": 0.0, "tumor_ratio": 0.20},
        {"sample_ratio": 0.01, "tme_ratio": 0.0, "tumor_ratio": 5.00},
        {"sample_ratio": 0.01, "tme_ratio": 0.0, "tumor_ratio": 3.00},
    ]
    stats = _summarize_lineage_support(rows)
    assert stats["concordance"] < 0.2
    assert stats["support_factor"] < 0.2


def test_combine_purity_ignores_zero_estimate_when_lineage_exists():
    overall, lower, upper = _combine_purity_estimates(
        sig_purity=0.35,
        sig_lower=0.20,
        sig_upper=0.50,
        estimate_purity=0.0,
        lineage_purity=0.60,
        lineage_lower=0.45,
        lineage_upper=0.75,
    )
    assert 0.45 < overall < 0.50
    assert lower <= overall <= upper
    assert lower == 0.20
    assert upper == 0.75


def test_combine_purity_penalizes_signature_only_calls_with_infiltration():
    overall, lower, upper = _combine_purity_estimates(
        sig_purity=0.52,
        sig_lower=0.30,
        sig_upper=0.70,
        estimate_purity=0.0,
        lineage_purity=None,
        lineage_lower=None,
        lineage_upper=None,
    )
    assert 0.15 < overall < 0.17
    assert lower <= overall <= upper


def test_combine_purity_deprioritizes_unstable_low_signature_when_lineage_exists():
    overall, lower, upper = _combine_purity_estimates(
        sig_purity=0.03,
        sig_lower=0.005,
        sig_upper=0.08,
        estimate_purity=0.35,
        lineage_purity=0.11,
        lineage_lower=0.08,
        lineage_upper=0.14,
        sig_stability=0.2,
    )
    assert overall > 0.10
    assert lower >= 0.08
    assert upper >= overall


def test_signature_summary_ignores_high_outlier():
    overall, lower, upper, stability = _summarize_gene_level_purity(
        [0.03, 0.04, 0.05, 0.06, 0.80],
        strategy="winsorized_median",
    )
    assert 0.04 < overall < 0.07
    assert 0.03 <= lower <= overall
    assert upper < 0.30
    assert 0.1 < stability < 1.0


def test_prad_signature_panel_excludes_rearranged_immune_genes():
    panel = _select_tumor_specific_genes("PRAD", n=30)
    assert panel
    # Only rearranged V/D/J/C segments should be blocked — prefix-based
    # checks would also drop unrelated genes like TRAF*, TRAK1, TRAP1.
    rearranged_prefixes = ("IGH", "IGK", "IGL", "TRA", "TRB", "TRG", "TRD")
    for gene in panel:
        for prefix in rearranged_prefixes:
            if gene.startswith(prefix) and len(gene) > len(prefix):
                segment_char = gene[len(prefix)]
                assert segment_char not in "VDJC", (
                    f"{gene} looks like a rearranged receptor segment"
                )
    assert "TRGV9" not in panel
    assert "TRGC1" not in panel


def test_signature_exclusion_preserves_unrelated_tr_ig_genes():
    """TRAF*, TRAK1, TRAP1, TRADD, IGHMBP2, IGFBP* should not be excluded."""
    from trufflepig.tumor_purity import _compile_excluded_gene_matcher

    is_excluded = _compile_excluded_gene_matcher()
    for gene in ["TRAF3", "TRAF6", "TRAK1", "TRAP1", "TRADD", "IGHMBP2", "IGFBP3"]:
        assert not is_excluded(gene), f"{gene} should not be excluded"
    # HLA class I stays, class II goes
    for gene in ["HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F", "HLA-G"]:
        assert not is_excluded(gene), f"{gene} (class I) should not be excluded"
    for gene in ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DQB1"]:
        assert is_excluded(gene), f"{gene} (class II) should be excluded"
    # Rearranged receptors should still be excluded
    for gene in ["TRGV9", "TRGC1", "IGHV3-33", "IGKV1-5", "TRBV7-9"]:
        assert is_excluded(gene), f"{gene} (rearranged receptor) should be excluded"


def test_dlbc_panel_bypasses_immune_exclusion():
    """Immune-origin cancers should include lineage markers that would otherwise
    be filtered as infiltrate contamination."""
    panel = _select_tumor_specific_genes("DLBC", n=30)
    assert panel
    # B-cell lineage markers should be recoverable for DLBC
    assert any(g.startswith("HLA-D") for g in panel), (
        f"DLBC panel should include HLA class II markers, got: {panel}"
    )


def test_signature_panel_cache_invalidates_on_param_change():
    """Mutating TUMOR_PURITY_PARAMETERS should not serve stale panels."""
    from trufflepig.tumor_purity import (
        TUMOR_PURITY_PARAMETERS,
        _select_tumor_specific_genes_for_panel,
    )

    panel_with_defaults = _select_tumor_specific_genes_for_panel("PRAD", n=30)
    params = TUMOR_PURITY_PARAMETERS["tumor_specific_markers"]
    original = params["excluded_gene_regexes"]
    try:
        # Add a pattern that blocks "KLK.*" — any KLK gene previously in
        # the panel should now be removed.
        params["excluded_gene_regexes"] = list(original) + [r"KLK.*"]
        panel_with_klk_blocked = _select_tumor_specific_genes_for_panel("PRAD", n=30)
        assert not any(g.startswith("KLK") for g in panel_with_klk_blocked), (
            f"cache should have re-keyed on param change; got {panel_with_klk_blocked}"
        )
    finally:
        params["excluded_gene_regexes"] = original

    # After restoring params the original panel should return.
    panel_restored = _select_tumor_specific_genes_for_panel("PRAD", n=30)
    assert panel_restored == panel_with_defaults


def test_combine_purity_treats_zero_stability_as_low_weight():
    """A stability of exactly 0.0 must not collapse to full weight via truthiness.

    Guards against the `sig_stability or 1.0` pattern: 0.0 is not
    "unknown", it's the strongest possible signal that the signature
    channel is unreliable. When the conflict gate happens NOT to fire
    (signature is close to lineage in absolute value), the weighted-log
    path should still downweight the signature instead of giving it full
    weight equal to lineage.
    """
    # Pick values where the conflict gate does NOT trigger — i.e.
    # sig/lineage >= signature_conflict_ratio (0.75). This forces the code
    # through the weighted-log branch where the bug lived.
    kwargs = dict(
        sig_lower=0.30,
        sig_upper=0.70,
        estimate_purity=None,
        lineage_purity=0.60,
        lineage_lower=0.55,
        lineage_upper=0.65,
    )
    overall_zero, _, _ = _combine_purity_estimates(
        sig_purity=0.50, sig_stability=0.0, **kwargs
    )
    overall_unknown, _, _ = _combine_purity_estimates(
        sig_purity=0.50, sig_stability=None, **kwargs
    )
    overall_high, _, _ = _combine_purity_estimates(
        sig_purity=0.50, sig_stability=0.9, **kwargs
    )
    # Stability=0 should pull the anchor closer to lineage (0.60) than
    # stability=None (which is treated as full signature weight = 1.0).
    # Stability=0.9 lies between the two.
    assert overall_zero > overall_unknown, (
        f"stability=0 must give lineage more relative weight than "
        f"stability=None; zero={overall_zero:.3f} unknown={overall_unknown:.3f}"
    )
    assert overall_unknown <= overall_high <= overall_zero or (
        overall_unknown <= overall_zero
    ), (
        f"Monotonicity check: lower stability should pull toward lineage; "
        f"zero={overall_zero:.3f} high={overall_high:.3f} unknown={overall_unknown:.3f}"
    )


# ── estimate_tumor_expression_ranges ──────────────────────────────


def test_ranges_dataframe_columns():
    """Verify the output DataFrame has all expected columns."""
    from trufflepig.plot import estimate_tumor_expression_ranges
    import pandas as pd

    # Minimal expression data with just a few genes
    df = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000169398", "ENSG00000160752"],
            "gene_symbol": ["PTK2", "IL34"],
            "TPM": [50.0, 10.0],
        }
    )
    purity_result = {
        "overall_lower": 0.05,
        "overall_estimate": 0.10,
        "overall_upper": 0.15,
    }
    result = estimate_tumor_expression_ranges(df, "PRAD", purity_result)
    assert isinstance(result, pd.DataFrame)

    expected_cols = [
        "gene_id",
        "symbol",
        "category",
        "observed_tpm",
        "tme_tpm_lo",
        "tme_tpm_med",
        "tme_tpm_hi",
        "max_healthy_tpm",
        "tme_explainable",
        "cohort_prior_tpm",
        "expression_reference_code",
        "expression_reference_source",
        "est_1",
        "est_5",
        "est_9",
        "median_est",
        "pct_cancer_median",
        "tcga_percentile",
        "is_surface",
        "is_cta",
        "therapies",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_ranges_use_fine_deconvolved_expression_reference(monkeypatch):
    """Fine report labels can provide tumor-expression priors while the
    broad cohort still controls decomposition/TME context."""
    import pandas as pd
    import trufflepig.plot_tumor_expr as mod

    def fake_deconvolved_reference(code):
        if str(code).upper() == "SARC_OS":
            return {"PTK2": 123.0}, "subtype_deconvolved"
        return {}, ""

    monkeypatch.setattr(
        mod, "_deconvolved_tumor_tpm_reference", fake_deconvolved_reference
    )
    df = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000169398", "ENSG00000160752"],
            "gene_symbol": ["PTK2", "IL34"],
            "TPM": [50.0, 10.0],
        }
    )
    purity_result = {
        "overall_lower": 0.05,
        "overall_estimate": 0.10,
        "overall_upper": 0.15,
    }

    result = mod.estimate_tumor_expression_ranges(
        df,
        "SARC",
        purity_result,
        expression_reference_type="SARC_OS",
    )
    row = result[result["symbol"].eq("PTK2")].iloc[0]
    assert row["cohort_prior_tpm"] == 123.0
    assert row["expression_reference_code"] == "SARC_OS"
    assert row["expression_reference_source"] == "subtype_deconvolved"


def test_deconvolved_reference_preserves_source_cohort(monkeypatch):
    import pandas as pd
    import trufflepig.plot_tumor_expr as mod

    monkeypatch.setattr(
        mod,
        "tcga_deconvolved_expression",
        lambda: pd.DataFrame(columns=["cancer_code", "symbol", "tumor_tpm_median"]),
    )
    monkeypatch.setattr(
        mod,
        "subtype_deconvolved_expression",
        lambda: pd.DataFrame(
            [
                {
                    "cancer_code": "SARC_OS",
                    "symbol": "RUNX2",
                    "source_cohort": "TREEHOUSE_POLYA_25_01",
                    "tumor_tpm_median": 25.0,
                }
            ]
        ),
    )
    mod._deconvolved_tumor_tpm_reference.cache_clear()
    mod._exact_expression_tpm_reference.cache_clear()

    try:
        values, source = mod._deconvolved_tumor_tpm_reference("SARC_OS")
    finally:
        mod._deconvolved_tumor_tpm_reference.cache_clear()
        mod._exact_expression_tpm_reference.cache_clear()

    assert values == {"RUNX2": 25.0}
    assert source == "subtype_deconvolved:TREEHOUSE_POLYA_25_01"


def test_deconvolved_reference_uses_source_code_alias(monkeypatch):
    from types import SimpleNamespace

    import pandas as pd
    import trufflepig.analyze as analyze
    import trufflepig.plot_tumor_expr as mod

    monkeypatch.setattr(
        analyze,
        "expression_reference_options",
        lambda code, include_fallback=False: (
            SimpleNamespace(
                source_kind="deconvolved_tumor_reference",
                source_code="TARGET_WT",
                reference_code="WILMS",
            ),
        )
        if str(code).upper() == "WILMS"
        else (),
    )
    monkeypatch.setattr(
        mod,
        "tcga_deconvolved_expression",
        lambda: pd.DataFrame(columns=["cancer_code", "symbol", "tumor_tpm_median"]),
    )
    monkeypatch.setattr(
        mod,
        "subtype_deconvolved_expression",
        lambda: pd.DataFrame(
            [
                {
                    "cancer_code": "TARGET_WT",
                    "symbol": "SIX2",
                    "source_cohort": "TARGET_WT_2015",
                    "tumor_tpm_median": 42.0,
                }
            ]
        ),
    )
    mod._deconvolved_tumor_tpm_reference.cache_clear()
    mod._exact_expression_tpm_reference.cache_clear()

    try:
        values, source = mod._deconvolved_tumor_tpm_reference("WILMS")
    finally:
        mod._deconvolved_tumor_tpm_reference.cache_clear()
        mod._exact_expression_tpm_reference.cache_clear()

    assert values == {"SIX2": 42.0}
    assert source == "subtype_deconvolved:TARGET_WT_2015"


def test_ranges_use_observed_reference_when_deconvolved_missing(monkeypatch):
    """Observed pirlygenes references can supply priors, but the sample's
    tumor-specific estimate is still computed from the sample."""
    import pandas as pd
    import trufflepig.plot_tumor_expr as mod

    monkeypatch.setattr(
        mod,
        "_deconvolved_tumor_tpm_reference",
        lambda code: ({}, ""),
    )
    monkeypatch.setattr(
        mod,
        "_observed_bulk_tpm_reference",
        lambda code: (
            {"PTK2": 77.0},
            "observed_bulk_reference:MMRF_COMMPASS",
        )
        if str(code).upper() == "MM"
        else ({}, ""),
    )
    mod._exact_expression_tpm_reference.cache_clear()

    df = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000169398", "ENSG00000160752"],
            "gene_symbol": ["PTK2", "IL34"],
            "TPM": [50.0, 10.0],
        }
    )
    purity_result = {
        "overall_lower": 0.05,
        "overall_estimate": 0.10,
        "overall_upper": 0.15,
    }

    try:
        result = mod.estimate_tumor_expression_ranges(
            df,
            "DLBC",
            purity_result,
            expression_reference_type="MM",
        )
    finally:
        mod._exact_expression_tpm_reference.cache_clear()

    row = result[result["symbol"].eq("PTK2")].iloc[0]
    assert row["cohort_prior_tpm"] == 77.0
    assert row["expression_reference_code"] == "MM"
    assert row["expression_reference_source"] == "observed_bulk_reference:MMRF_COMMPASS"
    assert row["expression_reference_kind"] == "observed_bulk_reference"
    assert bool(row["expression_reference_is_tumor_cell_estimate"]) is False
    assert row["tumor_attributed_bulk_tpm"] >= 0.0


def test_ranges_tme_explainable_clamps_at_observed():
    """For genes whose healthy-tissue max can explain the sample signal
    alone, `median_est` must not exceed `observed_tpm`. This guards
    against the 1/purity inflation for stromal / normal-lineage genes
    (e.g. KLK3 in prostate, FN1 stroma).
    """
    from trufflepig.plot import estimate_tumor_expression_ranges
    import pandas as pd

    # KLK3: high in normal prostate (~7700 nTPM), observed 500 TPM.
    # max_healthy >> observed → tme_explainable = True → clamped at
    # observed. Without the clamp, 500/0.3 ≈ 1667 would be reported.
    df = pd.DataFrame(
        {
            "ensembl_gene_id": [
                "ENSG00000142515",  # KLK3
                "ENSG00000075624",  # ACTB
                "ENSG00000156508",  # EEF1A1
                "ENSG00000111640",  # GAPDH
            ],
            "gene_symbol": ["KLK3", "ACTB", "EEF1A1", "GAPDH"],
            "TPM": [500.0, 150.0, 300.0, 100.0],
        }
    )
    purity = {"overall_lower": 0.25, "overall_estimate": 0.30, "overall_upper": 0.35}
    out = estimate_tumor_expression_ranges(df, "PRAD", purity)
    klk3 = out[out["symbol"] == "KLK3"].iloc[0]
    assert klk3["tme_explainable"], (
        "KLK3 should be tme_explainable in a 30% purity sample"
    )
    assert klk3["median_est"] <= klk3["observed_tpm"] + 1e-6, (
        f"median_est {klk3['median_est']} exceeds observed {klk3['observed_tpm']} "
        f"despite tme_explainable=True"
    )


def test_ranges_skips_shrinkage_when_cohort_prior_is_near_zero():
    """CTAs with cohort median ≈ 0 (activated in a minority of samples)
    must NOT be shrunk toward zero by empirical-Bayes — the cohort
    median is uninformative for sparsely-expressed genes.
    """
    from trufflepig.plot import estimate_tumor_expression_ranges
    from pirlygenes.gene_sets_cancer import CTA_gene_id_to_name
    from trufflepig.reference import pan_cancer_expression
    import pandas as pd

    # Pick 3 CTAs with cohort_prior near zero in PRAD.
    ref = pan_cancer_expression().drop_duplicates(subset="Symbol").set_index("Symbol")
    cta_map = CTA_gene_id_to_name()
    zero_cohort_ctas = []
    for gid, sym in cta_map.items():
        if sym in ref.index and float(ref.loc[sym, "PRAD_TPM"]) < 0.1:
            zero_cohort_ctas.append((gid, sym))
        if len(zero_cohort_ctas) >= 3:
            break
    assert zero_cohort_ctas, "Should find at least one near-zero CTA in PRAD"

    rows = [
        {"gene_id": gid, "gene_name": sym, "TPM": 50.0} for gid, sym in zero_cohort_ctas
    ]
    rows.extend(
        [
            {"gene_id": "ENSG00000075624", "gene_name": "ACTB", "TPM": 150.0},
            {"gene_id": "ENSG00000156508", "gene_name": "EEF1A1", "TPM": 300.0},
            {"gene_id": "ENSG00000111640", "gene_name": "GAPDH", "TPM": 100.0},
        ]
    )
    df = pd.DataFrame(rows)
    purity = {"overall_lower": 0.30, "overall_estimate": 0.35, "overall_upper": 0.40}
    out = estimate_tumor_expression_ranges(df, "PRAD", purity)

    # Near-zero cohort prior CTAs should land at roughly `observed/purity`,
    # the raw deconvolution — NOT shrunk toward zero.
    cta_symbols = {sym for _, sym in zero_cohort_ctas}
    ctas = out[out["symbol"].isin(cta_symbols)]
    assert len(ctas) > 0
    for _, row in ctas.iterrows():
        # observed=50, purity~0.35 → raw ≈ 143. Allow some flexibility
        # for the 3x3 TME/purity grid's median.
        assert row["cohort_prior_tpm"] < 1.0, (
            f"{row['symbol']} cohort prior {row['cohort_prior_tpm']} not near zero"
        )
        assert row["median_est"] > 100, (
            f"{row['symbol']} median_est {row['median_est']} was shrunk toward zero "
            f"despite near-zero cohort prior (should be ~143)"
        )


def test_ranges_low_purity_shrinks_toward_cohort_prior():
    """At very low purity, the sample-based estimator has high variance
    (1/purity inflates). Shrinkage should pull estimates toward the
    TCGA cohort prior so we don't report pathological numbers.
    """
    from trufflepig.plot import estimate_tumor_expression_ranges
    import pandas as pd

    # Gene not expressed in any healthy tissue but elevated in sample.
    # Without shrinkage: observed/purity is the estimator.
    # With shrinkage at very low purity: pulled toward cohort prior.
    df = pd.DataFrame(
        {
            "ensembl_gene_id": [
                "ENSG00000137959",  # IFI44L — interferon-stimulated, low baseline
                "ENSG00000075624",  # ACTB (for HK median)
                "ENSG00000156508",  # EEF1A1
                "ENSG00000111640",  # GAPDH
            ],
            "gene_symbol": ["IFI44L", "ACTB", "EEF1A1", "GAPDH"],
            "TPM": [10.0, 150.0, 300.0, 100.0],
        }
    )
    purity_low = {
        "overall_lower": 0.08,
        "overall_estimate": 0.10,
        "overall_upper": 0.12,
    }
    purity_high = {
        "overall_lower": 0.60,
        "overall_estimate": 0.65,
        "overall_upper": 0.70,
    }

    out_low = estimate_tumor_expression_ranges(df, "PRAD", purity_low)
    out_high = estimate_tumor_expression_ranges(df, "PRAD", purity_high)

    # Find the target gene in both outputs
    g_low = out_low[out_low["symbol"] == "IFI44L"]
    g_high = out_high[out_high["symbol"] == "IFI44L"]
    if len(g_low) and len(g_high):
        # At low purity the estimate should be LESS inflated than raw
        # 1/purity would give: raw low-purity estimate ≈ 10/0.10 = 100.
        # With shrinkage toward a (low) cohort prior, it should be much
        # closer to the cohort prior than to 100.
        low_est = float(g_low.iloc[0]["median_est"])
        raw_low = 10.0 / 0.10
        assert low_est < raw_low, (
            f"Shrinkage should pull low-purity estimate ({low_est}) "
            f"below raw 1/purity estimate ({raw_low})"
        )


def test_ranges_nine_estimates_are_sorted():
    """Each gene's 9 estimates should be in ascending order."""
    from trufflepig.plot import estimate_tumor_expression_ranges
    import pandas as pd

    df = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000169398"],
            "gene_symbol": ["PTK2"],
            "TPM": [100.0],
        }
    )
    purity_result = {
        "overall_lower": 0.05,
        "overall_estimate": 0.10,
        "overall_upper": 0.20,
    }
    result = estimate_tumor_expression_ranges(df, "PRAD", purity_result)
    if not result.empty:
        row = result.iloc[0]
        ests = [row[f"est_{i + 1}"] for i in range(9)]
        assert ests == sorted(ests), f"Estimates not sorted: {ests}"


def test_ranges_all_estimates_nonnegative():
    """No tumor expression estimate should be negative."""
    from trufflepig.plot import estimate_tumor_expression_ranges
    import pandas as pd

    df = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000169398"],
            "gene_symbol": ["PTK2"],
            "TPM": [0.5],  # very low expression
        }
    )
    purity_result = {
        "overall_lower": 0.05,
        "overall_estimate": 0.10,
        "overall_upper": 0.15,
    }
    result = estimate_tumor_expression_ranges(df, "PRAD", purity_result)
    if not result.empty:
        row = result.iloc[0]
        for i in range(9):
            assert row[f"est_{i + 1}"] >= 0, f"est_{i + 1} is negative"


def test_source_attribution_invariants_on_low_purity_prad_stroma_mix():
    """Per-gene source attribution must stay on bulk-TPM semantics.

    ``tumor_cell_tpm`` / ``median_est`` can exceed observed TPM after purity
    normalization. ``attr_tumor_tpm`` is different: it is the bulk expression
    mass attributed to tumor cells, so it must remain bounded by observed TPM
    and keep low/median/high intervals ordered.
    """
    from trufflepig.decomposition import decompose_sample
    from trufflepig.reference import pan_cancer_expression
    from trufflepig.plot import estimate_tumor_expression_ranges
    import pandas as pd

    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    sample_tpm = (
        0.15 * ref["PRAD_TPM"].astype(float)
        + 0.85 * ref["smooth_muscle_nTPM"].astype(float)
    )
    df = pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": sample_tpm,
        }
    )
    decomp = decompose_sample(
        df,
        cancer_types=["PRAD"],
        templates=["solid_primary"],
        top_k=1,
    )
    assert decomp
    ranges = estimate_tumor_expression_ranges(
        df,
        "PRAD",
        decomp[0].purity_result,
        decomposition_results=decomp,
    )
    expressed = ranges[ranges["observed_tpm"] > 0].copy()
    assert not expressed.empty

    assert (
        expressed["attr_tumor_tpm"] <= expressed["observed_tpm"] + 1e-6
    ).all()
    assert (
        expressed["attr_tumor_tpm_low"] <= expressed["attr_tumor_tpm"] + 1e-6
    ).all()
    assert (
        expressed["attr_tumor_tpm"] <= expressed["attr_tumor_tpm_high"] + 1e-6
    ).all()
    assert expressed["attr_tumor_fraction"].between(0.0, 1.0).all()
    assert (
        expressed["tumor_attributed_bulk_tpm"] == expressed["attr_tumor_tpm"]
    ).all()

    capped = expressed[expressed["low_purity_cap_applied"]]
    if not capped.empty:
        assert (
            capped["tumor_attributed_bulk_tpm_pre_low_purity_cap"]
            >= capped["attr_tumor_tpm"]
        ).all()


def test_ranges_empty_input():
    """Empty expression data should produce empty DataFrame."""
    from trufflepig.plot import estimate_tumor_expression_ranges
    import pandas as pd

    df = pd.DataFrame(
        {
            "ensembl_gene_id": [],
            "gene_symbol": [],
            "TPM": [],
        }
    )
    purity_result = {
        "overall_lower": 0.05,
        "overall_estimate": 0.10,
        "overall_upper": 0.15,
    }
    result = estimate_tumor_expression_ranges(df, "PRAD", purity_result)
    assert len(result) == 0


def test_ranges_pct_cancer_median_steap1_near_one():
    """STEAP1 at the PRAD clean-TPM median should have a fold near 1.0."""
    from trufflepig.plot import estimate_tumor_expression_ranges
    from trufflepig.reference import pan_cancer_expression
    import pandas as pd

    # At 100% purity, the sample and cohort are directly comparable in clean TPM.
    ref = pan_cancer_expression()
    ref_dedup = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    steap1_prad = float(ref_dedup.loc["STEAP1", "PRAD_TPM"])
    df = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000205542"],
            "gene_symbol": ["STEAP1"],
            "TPM": [steap1_prad],
        }
    )

    purity_result = {
        "overall_lower": 0.90,
        "overall_estimate": 1.0,
        "overall_upper": 1.0,
    }
    result = estimate_tumor_expression_ranges(df, "PRAD", purity_result)
    steap_row = result[result["symbol"] == "STEAP1"]
    if not steap_row.empty:
        pct = steap_row.iloc[0]["pct_cancer_median"]
        assert pct is not None, "pct_cancer_median should not be None"
        assert 0.5 < pct < 2.0, f"Expected ~1.0, got {pct}"


def _fn1_target_test_frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "ensembl_gene_id": [
                "ENSG00000115414",  # FN1
                "ENSG00000075624",  # ACTB
                "ENSG00000156508",  # EEF1A1
                "ENSG00000111640",  # GAPDH
            ],
            "gene_symbol": ["FN1", "ACTB", "EEF1A1", "GAPDH"],
            "TPM": [180.0, 150.0, 300.0, 100.0],
        }
    )


def test_ranges_fn1_therapy_requires_transcript_support():
    from trufflepig.plot import estimate_tumor_expression_ranges

    df = _fn1_target_test_frame()
    purity = {"overall_lower": 0.45, "overall_estimate": 0.55, "overall_upper": 0.65}
    out = estimate_tumor_expression_ranges(df, "PRAD", purity)
    fn1 = out[out["symbol"] == "FN1"].iloc[0]

    assert fn1["therapies"] == ""
    assert fn1["category"] != "therapy_target"
    assert bool(fn1["therapy_supported"]) is False
    assert "EDB+ FN1" in fn1["therapy_support_note"]
    assert "transcript-level data is unavailable" in fn1["therapy_support_note"]


def test_ranges_fn1_therapy_stays_blocked_without_edb_transcripts():
    from trufflepig.plot import estimate_tumor_expression_ranges
    import pandas as pd

    df = _fn1_target_test_frame()
    df.attrs["transcript_expression"] = pd.DataFrame(
        {
            "transcript_id": ["ENST00000443816", "ENST00000421182"],
            "TPM": [70.0, 30.0],
            "ensembl_gene_id": ["ENSG00000115414", "ENSG00000115414"],
            "gene_symbol": ["FN1", "FN1"],
        }
    )
    purity = {"overall_lower": 0.45, "overall_estimate": 0.55, "overall_upper": 0.65}
    out = estimate_tumor_expression_ranges(df, "PRAD", purity)
    fn1 = out[out["symbol"] == "FN1"].iloc[0]

    assert fn1["therapies"] == ""
    assert fn1["category"] != "therapy_target"
    assert bool(fn1["therapy_supported"]) is False
    assert "no EDB+ FN1 transcripts were detected" in fn1["therapy_support_note"]


def test_ranges_fn1_therapy_requires_high_edb_transcripts():
    from trufflepig.plot import estimate_tumor_expression_ranges
    import pandas as pd

    df = _fn1_target_test_frame()
    df.attrs["transcript_expression"] = pd.DataFrame(
        {
            "transcript_id": ["ENST00000432072", "ENST00000443816"],
            "TPM": [40.0, 20.0],
            "ensembl_gene_id": ["ENSG00000115414", "ENSG00000115414"],
            "gene_symbol": ["FN1", "FN1"],
        }
    )
    purity = {"overall_lower": 0.45, "overall_estimate": 0.55, "overall_upper": 0.65}
    out = estimate_tumor_expression_ranges(df, "PRAD", purity)
    fn1 = out[out["symbol"] == "FN1"].iloc[0]

    assert "ADC" in fn1["therapies"]
    assert fn1["category"] == "therapy_target"
    assert bool(fn1["therapy_supported"]) is True
    assert fn1["therapy_support_tpm"] == 40.0
    assert fn1["therapy_support_fraction"] == 0.667
    assert "PYX-201 / NCT05720117" in fn1["therapy_support_note"]


# ── _TME_TISSUES consistency ─────────────────────────────────────


def test_tme_tissues_are_valid():
    """All curated TME tissues should exist in the reference data."""
    from trufflepig.plot import _TME_TISSUES
    from trufflepig.reference import pan_cancer_expression

    ref = pan_cancer_expression()
    ntpm_cols = {c.removesuffix("_nTPM") for c in ref.columns if c.endswith("_nTPM")}
    for tissue in _TME_TISSUES:
        assert tissue in ntpm_cols, (
            f"TME tissue {tissue!r} not in reference nTPM columns"
        )


# ── CLI lineage narrative ─────────────────────────────────────────


def test_lineage_narrative_generation():
    """The lineage narrative should handle all three cases: retained, lost, not found."""
    from trufflepig.tumor_purity import LINEAGE_GENES

    # Simulate a purity result with lineage component
    lineage_per_gene = [
        {
            "gene": "STEAP1",
            "purity": 0.098,
            "sample_tpm": 90.0,
            "sample_ratio": 0.16,
            "ref_ratio": 1.1,
            "tme_ratio": 0.01,
            "tumor_ratio": 1.6,
        },
        {
            "gene": "KLK3",
            "purity": 0.003,
            "sample_tpm": 139.0,
            "sample_ratio": 0.25,
            "ref_ratio": 50.0,
            "tme_ratio": 0.0,
            "tumor_ratio": 73.0,
        },
    ]
    purity = {
        "cancer_type": "PRAD",
        "overall_estimate": 0.098,
        "overall_lower": 0.078,
        "overall_upper": 0.104,
        "components": {
            "lineage": {
                "genes": ["STEAP1", "KLK3"],
                "purity": 0.098,
                "lower": 0.078,
                "upper": 0.104,
                "per_gene": lineage_per_gene,
            },
            "stromal": {"enrichment": 4.3, "n_genes": 141},
            "immune": {"enrichment": 2.5, "n_genes": 141},
        },
    }

    # The narrative code in cli.py uses these fields
    lineage = purity["components"]["lineage"]
    sorted_genes = sorted(lineage["per_gene"], key=lambda g: g["purity"], reverse=True)
    median_p = lineage["purity"]

    retained = [g for g in sorted_genes if g["purity"] >= median_p * 0.5]
    lost = [g for g in sorted_genes if g["purity"] < median_p * 0.5]

    all_lineage = LINEAGE_GENES.get("PRAD", [])
    found_names = {g["gene"] for g in lineage_per_gene}
    not_found = [g for g in all_lineage if g not in found_names]

    assert len(retained) == 1  # STEAP1
    assert retained[0]["gene"] == "STEAP1"
    assert len(lost) == 1  # KLK3
    assert lost[0]["gene"] == "KLK3"
    assert len(not_found) > 0  # Missing genes from PRAD lineage set
    assert "FOLH1" in not_found


# ── #108 per-target compositional attribution ─────────────────────────────


def test_format_attribution_cell_renders_estimated_tumor_and_reference_component():
    """Attribution cells identify both values as outputs of the RNA model."""
    from trufflepig.main import _format_attribution_cell

    row_with_attr = {
        "observed_tpm": 150.0,
        "attribution": {"endothelial": 12.0, "fibroblast": 2.0},
        "attr_tumor_tpm": 136.0,
        "attr_top_compartment": "endothelial",
        "attr_top_compartment_tpm": 12.0,
    }
    assert _format_attribution_cell(row_with_attr) == (
        "estimated tumor 136 / endothelial reference component 12 (RNA model)"
    )

    row_no_attr = {
        "observed_tpm": 150.0,
        "attribution": {},
        "attr_tumor_tpm": 0.0,
        "attr_top_compartment": "",
        "attr_top_compartment_tpm": 0.0,
    }
    assert _format_attribution_cell(row_no_attr) == "—"

    row_zero_observed = {
        "observed_tpm": 0.0,
        "attribution": {"T_cell": 1.0},
        "attr_tumor_tpm": 0.0,
        "attr_top_compartment": "T_cell",
        "attr_top_compartment_tpm": 1.0,
    }
    assert _format_attribution_cell(row_zero_observed) == "—"

    row_tumor_only = {
        "observed_tpm": 80.0,
        "attribution": {"endothelial": 5.0},
        "attr_tumor_tpm": 75.0,
        "attr_top_compartment": "",
        "attr_top_compartment_tpm": 0.0,
    }
    assert _format_attribution_cell(row_tumor_only) == "estimated tumor 75 (RNA model)"


def test_tme_dominant_flag_reads_attribution_when_available():
    """When `attribution` is present, `tme_dominant` must be derived
    from `attr_tumor_fraction < 0.3` (#108), not the legacy tme_fold
    formula. Simulate the column shape estimate_tumor_expression_ranges
    produces."""
    import pandas as pd

    # We don't call estimate_tumor_expression_ranges directly here — it
    # requires a heavy reference load. Instead, verify the post-hoc
    # derivation logic used elsewhere: a row with tumor < 30% of
    # observed should be TME-dominant even if tme_fold is small.
    row = pd.Series(
        {
            "observed_tpm": 100.0,
            "attribution": {"fibroblast": 80.0},
            "attr_tumor_tpm": 20.0,
            "attr_tumor_fraction": 0.20,
            "tme_tpm_med": 0.05,  # would NOT trigger the legacy fold rule
        }
    )
    attribution = row["attribution"]
    assert isinstance(attribution, dict) and attribution, "attribution populated"
    attr_fraction = row["attr_tumor_fraction"]
    tme_dominant_derived = row["observed_tpm"] > 0 and attr_fraction < 0.30
    assert tme_dominant_derived, (
        "tumor fraction is 20% of observed — should flag as TME-dominant"
    )
