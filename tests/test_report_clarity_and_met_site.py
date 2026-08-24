"""Tests for v4.1 report-clarity + met-site bundle (issues #13, #32, #33)."""

from types import SimpleNamespace

import pandas as pd
import pytest

from trufflepig.main import (
    _build_target_report,
    _cancer_type_vote_summary_markdown,
    _clamp_interval_to_cap,
    _constrain_purity_interval_with_decomposition,
    _effective_met_site_for_background,
    _finalize_fused_purity,
    _infer_likely_met_site_context,
    _next_best_support_gap,
)


def test_vote_summary_escapes_pipe_and_newline_in_detail_cells():
    """A free-text channel rationale carrying `|` or a newline must not break the
    Cancer-Type Evidence Votes table (same contract as the decision-trace table:
    both route the detail through reporting.md_table_cell)."""
    analysis = {
        "cancer_type_evidence": {
            "selected": {
                "cancer_type": "COAD",
                "selected_by": "pan_cancer_signature_ranker",
                "evidence_channels": [
                    {
                        "channel": "local_expression_reference",
                        "role": "supporting",
                        "support": 0.9,
                        "status": "supporting",
                        "details": {"panel": "lineage|program\nrationale"},
                        "selects_report_label": False,
                    }
                ],
            }
        },
    }
    md = _cancer_type_vote_summary_markdown(analysis)
    data_rows = [
        ln
        for ln in md.splitlines()
        if ln.startswith("| ") and "Signal" not in ln and "---" not in ln
    ]
    assert data_rows, md
    row = data_rows[0]
    # The pipe is escaped (kept as a literal `\|` in the cell) and the newline is
    # flattened, so the row still has exactly the 5-column / 6-delimiter structure.
    assert "\n" not in row
    assert "\\|" in row
    assert row.replace("\\|", "").count("|") == 6


def _write_target_report(ranges_df, analysis, prefix, cancer_type, purity_result):
    """Build the target report via the live ``_build_target_report`` and write it
    to ``{prefix}-targets.md`` (the contract the removed ``_generate_target_report``
    wrapper used to provide)."""
    md = _build_target_report(
        ranges_df,
        analysis,
        cancer_type=cancer_type,
        purity_result=purity_result,
        decomp_results=analysis.get("decomposition_results"),
    )
    with open(f"{prefix}-targets.md", "w") as fh:
        fh.write(md)
    return md


from trufflepig.plot import (
    MET_SITE_TISSUE_AUGMENTATION,
    estimate_tumor_expression_ranges,
)
from trufflepig.reference import pan_cancer_expression


# ── #32: qualitative score language ────────────────────────────────────


def test_next_best_support_gap_returns_ratio_for_two_candidates():
    trace = [
        {"code": "PRAD", "support_fraction_of_top": 1.00},
        {"code": "HNSC", "support_fraction_of_top": 0.40},
    ]
    code, ratio = _next_best_support_gap(trace)
    assert code == "HNSC"
    assert ratio == pytest.approx(2.5, rel=1e-3)


def test_next_best_support_gap_handles_edge_cases():
    # One candidate — no gap to measure
    assert _next_best_support_gap([{"code": "PRAD", "support_fraction_of_top": 1.0}]) == (
        None,
        None,
    )
    # Empty
    assert _next_best_support_gap([]) == (None, None)
    # Runner-up has zero support — can't divide
    code, ratio = _next_best_support_gap(
        [
            {"code": "PRAD", "support_fraction_of_top": 1.0},
            {"code": "HNSC", "support_fraction_of_top": 0.0},
        ]
    )
    assert code == "HNSC"
    assert ratio is None


# ── #13: met-site augmentation ─────────────────────────────────────────


def test_met_site_augmentation_map_covers_all_sites_mentioned_in_issue():
    """Issue #13 lists primary / lymph_node / liver / brain / lung / bone.
    Any new met site exposed via CLI must live in the augmentation map
    so downstream knows how to augment the TME reference."""
    expected = {"primary", "lymph_node", "liver", "brain", "lung", "bone"}
    assert expected.issubset(set(MET_SITE_TISSUE_AUGMENTATION.keys()))
    # `primary` intentionally adds nothing — tumor is assumed in situ.
    assert MET_SITE_TISSUE_AUGMENTATION["primary"] == set()


def test_met_site_liver_includes_liver_tissue_in_tme_reference():
    """Validates #13: a liver-met ``estimate_tumor_expression_ranges``
    call should use a TME reference set that includes ``liver``, while
    the default (met_site=None) does not."""
    # Small synthetic sample drawn from the bundled reference so the
    # function can actually run.
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    df = pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref["liver_nTPM"].astype(float),  # a liver-rich synthetic
        }
    )
    purity = {
        "overall_estimate": 0.6,
        "overall_lower": 0.4,
        "overall_upper": 0.8,
        "components": {"stromal": {"enrichment": 1.0}, "immune": {"enrichment": 1.0}},
    }

    baseline = estimate_tumor_expression_ranges(
        df_gene_expr=df,
        cancer_type="COAD",
        purity_result=purity,
        met_site=None,
    )
    augmented = estimate_tumor_expression_ranges(
        df_gene_expr=df,
        cancer_type="COAD",
        purity_result=purity,
        met_site="liver",
    )

    # TME medians for liver-expressed genes (ALB, APOA1, HP) should be
    # materially higher under the liver-augmented reference, since
    # adding the liver column to the TME tissue set raises the
    # percentile for liver-high genes.
    def _tme_med(df_, symbol):
        sub = df_[df_["symbol"] == symbol]
        if sub.empty:
            return None
        # estimate_tumor_expression_ranges reports the TME background
        # as a fold-over-HK median; a higher value means the reference
        # TME set picks up more of this gene's signal.
        return float(sub.iloc[0]["tme_tpm_med"])

    alb_baseline = _tme_med(baseline, "ALB")
    alb_augmented = _tme_med(augmented, "ALB")
    # Both runs should produce an ALB row (it's a reference gene).
    assert alb_baseline is not None
    assert alb_augmented is not None
    # Augmented reference must have higher or equal TME contribution
    # for the host-tissue gene.
    assert alb_augmented >= alb_baseline
    # And strictly higher when liver signal is the distinguishing
    # factor (ALB is nearly liver-only).
    assert alb_augmented > alb_baseline


def test_met_site_rejects_unknown_value():
    """Invalid met_site passed directly to the estimator is a no-op on
    the set-union rather than a crash (the CLI layer validates before
    reaching here). The function must not error on unknown strings."""
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    df = pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref["liver_nTPM"].astype(float),
        }
    )
    purity = {
        "overall_estimate": 0.6,
        "overall_lower": 0.4,
        "overall_upper": 0.8,
        "components": {"stromal": {"enrichment": 1.0}, "immune": {"enrichment": 1.0}},
    }
    # Must not crash on unknown met_site (augmentation map .get default
    # is an empty set). CLI validates up front in analyze().
    ranges = estimate_tumor_expression_ranges(
        df_gene_expr=df,
        cancer_type="COAD",
        purity_result=purity,
        met_site="moon",
    )
    assert len(ranges) > 0


def test_infers_likely_met_site_from_strong_off_primary_background():
    analysis = {
        "cancer_type": "BLCA",
        "sample_mode": "solid",
        "tissue_scores": [
            ("liver", 0.964, 20),
            ("gallbladder", 0.807, 20),
            ("urinary_bladder", 0.796, 20),
        ],
    }

    inferred = _infer_likely_met_site_context(analysis)

    assert inferred["site"] == "liver"
    assert inferred["tissue"] == "liver"
    assert inferred["primary_tissue"] == "urinary_bladder"
    assert _effective_met_site_for_background(
        {"analysis_constraints": {}, "inferred_site_context": inferred}
    ) == "liver"


def test_supported_decomposition_site_feeds_background_met_site():
    analysis = {
        "cancer_type": "COAD",
        "analysis_constraints": {},
        "decomposition_results": [
            SimpleNamespace(
                template="met_liver",
                site_evidence={"site_supported": True, "status": "site_supported"},
                warnings=[],
                template_site_factor=0.9,
                template_tissue_score=0.8,
            )
        ],
    }

    assert _effective_met_site_for_background(analysis) == "liver"


def test_fit_only_decomposition_site_does_not_feed_background_met_site():
    analysis = {
        "cancer_type": "COAD",
        "analysis_constraints": {},
        "decomposition_results": [
            SimpleNamespace(
                template="met_bone",
                site_evidence={"site_supported": False, "status": "fit_only"},
                warnings=[],
                template_site_factor=0.9,
                template_tissue_score=0.8,
            )
        ],
    }

    assert _effective_met_site_for_background(analysis) is None


def test_does_not_infer_met_site_when_background_matches_primary():
    analysis = {
        "cancer_type": "BLCA",
        "sample_mode": "solid",
        "tissue_scores": [
            ("urinary_bladder", 0.93, 20),
            ("liver", 0.86, 20),
        ],
    }

    assert _infer_likely_met_site_context(analysis) is None


def test_does_not_infer_liver_met_for_liver_primary():
    analysis = {
        "cancer_type": "LIHC",
        "sample_mode": "solid",
        "tissue_scores": [
            ("liver", 0.96, 20),
            ("gallbladder", 0.80, 20),
        ],
    }

    assert _infer_likely_met_site_context(analysis) is None


def test_decomposition_non_tumor_caps_purity_upper_bound():
    purity = {
        "overall_estimate": 0.685,
        "overall_lower": 0.33,
        "overall_upper": 1.0,
        "components": {},
    }
    decomp = SimpleNamespace(
        fractions={
            "tumor": 0.685,
            "endothelial": 0.109,
            "hepatocyte": 0.093,
            "T_cell": 0.041,
            "B_cell": 0.039,
            "fibroblast": 0.018,
            "myeloid": 0.015,
        },
        warnings=[],
    )

    changed = _constrain_purity_interval_with_decomposition(purity, decomp)

    assert changed is True
    assert purity["overall_upper"] == pytest.approx(0.8425)
    cap = purity["components"]["decomposition_interval_cap"]
    assert cap["original_upper"] == pytest.approx(1.0)
    assert cap["modeled_non_tumor_fraction"] == pytest.approx(0.315)


def test_decomposition_interval_cap_ignores_tiny_non_tumor_assignments():
    purity = {
        "overall_estimate": 0.96,
        "overall_lower": 0.90,
        "overall_upper": 1.0,
        "components": {},
    }
    decomp = SimpleNamespace(
        fractions={"tumor": 0.96, "endothelial": 0.02, "T_cell": 0.01},
        warnings=[],
    )

    changed = _constrain_purity_interval_with_decomposition(purity, decomp)

    assert changed is False
    assert purity["overall_upper"] == pytest.approx(1.0)


def test_fused_upper_never_exceeds_decomposition_cap():
    """The random-effects fusion pass must not widen the upper bound back above
    the physical decomposition cap. Reviewer scenario: a cap ~0.84 must survive a
    discordant fused upper ~0.97."""
    cap = 0.8425
    est, lo, up = _clamp_interval_to_cap(0.60, 0.31, 0.97, cap)
    assert up == pytest.approx(cap)
    # A point/lower already below the cap are left untouched.
    assert est == pytest.approx(0.60)
    assert lo == pytest.approx(0.31)


def test_clamp_interval_to_cap_pulls_point_and_lower_below_ceiling():
    """If the whole fused interval sits above the cap, the point and lower are
    pulled down so lower <= estimate <= upper stays coherent."""
    est, lo, up = _clamp_interval_to_cap(0.95, 0.90, 0.99, 0.84)
    assert up == pytest.approx(0.84)
    assert est == pytest.approx(0.84)
    assert lo == pytest.approx(0.84)


def test_clamp_interval_to_cap_is_noop_without_cap():
    """No decomposition cap (None / non-numeric) leaves the fused interval intact."""
    assert _clamp_interval_to_cap(0.7, 0.5, 0.9, None) == (0.7, 0.5, 0.9)
    assert _clamp_interval_to_cap(0.7, 0.5, 0.9, "n/a") == (0.7, 0.5, 0.9)


def _saturated_nut_like_analysis():
    """A ceiling-pinned (~100%) purity whose only physical signal (decomposition
    residual) says ~55% and whose signature is uncorroborating — the NUT-carcinoma
    saturation the anti-saturation guard exists to catch."""
    return {
        "purity": {
            "overall_estimate": 1.0,
            "overall_lower": 0.9,
            "overall_upper": 1.0,
            "components": {
                "signature": {"purity": 0.5, "stability": 0.8},
                "lineage": {"purity": 1.0},  # saturated -> dropped from the fusion
                "estimate_purity": 0.9,
                "decomposition": {"residual_fraction": 0.55},
            },
        },
        "decomposition": {"purity_stability": {"fragile": False}},
    }


def test_finalize_fused_purity_desaturates_ceiling_pinned_read():
    analysis = _saturated_nut_like_analysis()
    _finalize_fused_purity(analysis)
    p = analysis["purity"]
    assert p["pre_fusion_estimate"] == 1.0
    assert p["best_integration"]["point_source"] == "desaturated_fusion"
    assert p["overall_estimate"] < 0.85  # pulled off the spurious 100%
    assert p["overall_upper"] <= 1.0


def test_finalize_fused_purity_respects_decomposition_cap():
    analysis = _saturated_nut_like_analysis()
    analysis["purity"]["components"]["decomposition_interval_cap"] = {
        "constrained_upper": 0.60,
    }
    _finalize_fused_purity(analysis)
    p = analysis["purity"]
    assert p["overall_upper"] <= 0.60  # fused interval never re-widens above the cap
    assert p["overall_estimate"] <= 0.60


def test_finalize_fused_purity_is_safe_without_purity():
    analysis = {"decomposition": {}}
    _finalize_fused_purity(analysis)  # must not raise
    assert "purity" not in analysis


def test_summary_md_structure_for_report_clarity(tmp_path):
    """#33: summary.md should include the input filename, the source
    attribution (auto vs user-specified), the tissue-score caveat, and
    no raw ``score 0.019``-style composite number in the prose."""
    from trufflepig.main import _generate_text_reports

    analysis = {
        "cancer_type": "PRAD",
        "cancer_name": "Prostate Adenocarcinoma",
        "cancer_score": 0.019,  # realistic low composite
        "cancer_type_source": "auto-detected",
        "top_cancers": [("PRAD", 0.019)],
        "purity": {
            "overall_estimate": 0.11,
            "overall_lower": 0.01,
            "overall_upper": 0.11,
            "components": {
                "stromal": {"enrichment": 4.3},
                "immune": {"enrichment": 2.5},
            },
        },
        "tissue_scores": [
            ("prostate", 0.90, 20),
            ("smooth_muscle", 0.72, 18),
            ("rectum", 0.71, 17),
        ],
        "mhc1": {"HLA-A": 25, "HLA-B": 30, "HLA-C": 22, "B2M": 3000},
        "mhc2": {},
        "candidate_trace": [
            {"code": "PRAD", "support_fraction_of_top": 1.00, "signature_score": 0.67},
            {"code": "HNSC", "support_fraction_of_top": 0.40, "signature_score": 0.35},
        ],
        "family_summary": {},
        "fit_quality": {
            "label": "strong_separation",
            "message": "Top match is clearly separated from alternatives.",
        },
        "sample_mode": "solid",
        "analysis_constraints": {},
    }

    prefix = str(tmp_path / "sample")
    embedding_meta = {
        "method": "hierarchy",
        "feature_kind": "hierarchical_scores",
        "n_genes": 0,
        "n_features": 0,
        "n_types": 33,
        "families": [],
    }
    _generate_text_reports(
        analysis,
        embedding_meta,
        prefix,
        decomp_results=[],
        input_path="/data/sample_BG002.tsv",
    )

    # The retired free-form summary.md used to carry input-filename
    # header, source attribution, and the "2.5× HNSC" / "similarity"
    # clauses. Analysis.md now owns the input-filename + candidate
    # trace; the "similarity" tissue-score caveat lives in analysis.md
    # too. No try/except guard — if _generate_text_reports KeyErrors on
    # a minimal fixture that's a real test signal, not something to
    # swallow.
    analysis_md_path = tmp_path / "sample-analysis.md"
    assert analysis_md_path.exists(), (
        "_generate_text_reports must write analysis.md; missing file "
        "points at a test-fixture bug, not an expected branch"
    )
    analysis_md = analysis_md_path.read_text()
    # Input filename header — now under "## Input characterization".
    assert "/data/sample_BG002.tsv" in analysis_md
    # Source attribution (#33) — moved from summary.md to analysis.md.
    assert "auto-detected" in analysis_md
    # Raw composite score 0.019 must not appear in prose (#32).
    assert "0.019" not in analysis_md, (
        "Raw composite cancer_score must not be in report prose (#32)."
    )
    # Tissue-score caveat (#33).
    assert "similarity" in analysis_md.lower()


def test_analysis_renders_background_tissue_driver_genes(tmp_path):
    from types import SimpleNamespace
    from trufflepig.main import _generate_text_reports

    analysis = {
        "cancer_type": "SARC",
        "cancer_name": "Sarcoma",
        "cancer_score": 0.42,
        "cancer_type_source": "auto-detected",
        "top_cancers": [("SARC", 0.42)],
        "candidate_trace": [],
        "family_summary": {},
        "fit_quality": {},
        "sample_mode": "solid",
        "analysis_constraints": {},
        "purity": {
            "overall_estimate": 0.80,
            "overall_lower": 0.48,
            "overall_upper": 1.00,
            "components": {
                "stromal": {"enrichment": 3.15},
                "immune": {"enrichment": 1.28},
            },
        },
        "tissue_scores": [
            ("testis", 0.703, 20),
            ("retina", 0.667, 20),
        ],
        "tissue_score_details": [
            {
                "tissue": "testis",
                "score": 0.703,
                "n_genes": 20,
                "drivers": [
                    {"gene": "UBQLN3", "sample_tpm": 154.3, "percentile": 0.98},
                    {"gene": "CXorf51B", "sample_tpm": 145.8, "percentile": 0.98},
                ],
            },
            {
                "tissue": "retina",
                "score": 0.667,
                "n_genes": 20,
                "drivers": [
                    {"gene": "FSTL5", "sample_tpm": 91.9, "percentile": 0.96},
                    {"gene": "GNGT1", "sample_tpm": 1842.8, "percentile": 0.94},
                ],
            },
        ],
        "mhc1": {"HLA-A": 438, "HLA-B": 625, "HLA-C": 346, "B2M": 6027},
        "mhc2": {},
        "sample_context": SimpleNamespace(
            library_prep="exome_capture",
            library_prep_confidence=0.9,
            preservation="unknown",
            degradation_index=2.58,
            missing_mt=False,
            signals={},
            degradation_severity="none",
        ),
        "quality": {
            "degradation": {
                "level": "normal",
                "mt_fraction": 0.0,
                "rp_fraction": 0.067,
                "long_short_ratio": 2.75,
                "matched_tissue": "testis",
                "baseline_mt": 0.245,
                "baseline_rp": 0.083,
                "mt_fold": 0.0,
                "rp_fold": 0.8,
                "message": "Within normal range.",
            },
            "culture": {"level": "normal", "message": "No cell culture signal."},
        },
    }
    embedding_meta = {
        "method": "hierarchy",
        "feature_kind": "hierarchical_scores",
        "n_features": 12,
        "n_types": 2,
        "families": ["MESENCHYMAL"],
        "sites": ["bone", "lung"],
    }
    prefix = str(tmp_path / "sarcoma-bg")

    _generate_text_reports(analysis, embedding_meta, prefix, decomp_results=[])

    detailed = (tmp_path / "sarcoma-bg-analysis.md").read_text()
    assert "Top matching genes" in detailed
    assert "UBQLN3 (154 TPM)" in detailed
    assert "FSTL5 (92 TPM)" in detailed
    assert "bone marrow but not a standalone bone/osteoblast tissue row" in detailed
    assert "Matched tissue baseline" not in detailed
    assert "Fold over baseline" not in detailed
    assert "MT/RP baseline comparison is not emphasized here" in detailed


def test_detailed_report_uses_generic_lineage_caveat(tmp_path):
    from trufflepig.main import _generate_text_reports

    analysis = {
        "cancer_type": "HNSC",
        "cancer_name": "Head and Neck Squamous Cell Carcinoma",
        "cancer_score": 0.18,
        "top_cancers": [("HNSC", 0.18), ("LUSC", 0.12)],
        "candidate_trace": [
            {
                "code": "HNSC",
                "support_score": 0.18,
                "support_geomean": 0.18,
                "support_fraction_of_top": 1.0,
                "signature_score": 0.81,
                "purity_estimate": 0.42,
                "family_label": "SQUAMOUS",
                "lineage_purity": 0.44,
                "lineage_concordance": 0.89,
            },
            {
                "code": "LUSC",
                "support_score": 0.12,
                "support_geomean": 0.12,
                "support_fraction_of_top": 0.67,
                "signature_score": 0.72,
                "purity_estimate": 0.38,
                "family_label": "SQUAMOUS",
                "lineage_purity": 0.41,
                "lineage_concordance": 0.82,
            },
        ],
        "fit_quality": {
            "label": "ambiguous",
            "message": "Top squamous candidates remain close; treat the exact site as provisional.",
        },
        "family_summary": {
            "display": "squamous family (HNSC > LUSC)",
            "subtype_clause": "HNSC > LUSC",
        },
        "purity": {
            "overall_estimate": 0.42,
            "overall_lower": 0.25,
            "overall_upper": 0.58,
            "cancer_type": "HNSC",
            "components": {
                "stromal": {"enrichment": 2.1},
                "immune": {"enrichment": 1.7},
                "lineage": {
                    "per_gene": [
                        {"gene": "KRT14", "purity": 0.52},
                        {"gene": "KRT5", "purity": 0.47},
                        {"gene": "TP63", "purity": 0.24},
                    ],
                    "purity": 0.49,
                    "lower": 0.47,
                    "upper": 0.52,
                },
            },
        },
        "tissue_scores": [("tongue", 0.93, 20), ("esophagus", 0.84, 20)],
        "mhc1": {"HLA-A": 26, "HLA-B": 31, "B2M": 220},
        "mhc2": {},
        "sample_mode": "solid",
        "call_summary": {
            "label_options": ["HNSC", "LUSC"],
            "label_display": "HNSC or LUSC",
            "reported_context": "primary",
            "reported_site": "primary site",
            "site_indeterminate": False,
            "site_note": None,
            "hypothesis_display": ["HNSC / solid_primary", "LUSC / solid_primary"],
        },
    }
    embedding_meta = {
        "method": "hierarchy",
        "feature_kind": "hierarchical_scores",
        "n_features": 12,
        "n_types": 3,
        "families": ["SQUAMOUS"],
        "sites": ["tongue", "lung", "esophagus"],
    }
    prefix = str(tmp_path / "hnsccase")

    _generate_text_reports(analysis, embedding_meta, prefix, decomp_results=[])

    detailed = (tmp_path / "hnsccase-analysis.md").read_text()
    assert "Lineage caveat" in detailed
    assert "prostate-lineage" not in detailed
    assert (
        "do NOT by themselves distinguish tumor cells from benign cells of the same lineage"
        in detailed
    )


def test_compose_disease_state_detects_crpc_nepc_pattern():
    """#78: AR retained + AR targets collapsed + NE up + AR axis
    down must produce the castrate-resistant + emerging-NEPC narrative.
    """
    from trufflepig.main import compose_disease_state_narrative
    from trufflepig.therapy_response import TherapyAxisScore

    analysis = {
        "cancer_type": "PRAD",
        "purity": {
            "overall_estimate": 0.64,
            "components": {
                "lineage": {
                    "per_gene": [
                        {"gene": "AR", "purity": 0.51},
                        {"gene": "STEAP2", "purity": 0.16},
                        {"gene": "KLK3", "purity": 0.003},
                        {"gene": "KLK2", "purity": 0.015},
                        {"gene": "NKX3-1", "purity": 0.011},
                        {"gene": "HOXB13", "purity": 0.004},
                        {"gene": "FOLH1", "purity": 0.073},
                    ]
                }
            },
        },
        "therapy_response_scores": {
            "AR_signaling": TherapyAxisScore(
                therapy_class="AR_signaling",
                state="down",
                up_geomean_fold=0.33,
                down_geomean_fold=2.54,
            ),
            "NE_differentiation": TherapyAxisScore(
                therapy_class="NE_differentiation",
                state="up",
                up_geomean_fold=2.08,
            ),
            "EMT": TherapyAxisScore(
                therapy_class="EMT",
                state="up",
                up_geomean_fold=8.95,
            ),
            "hypoxia": TherapyAxisScore(
                therapy_class="hypoxia",
                state="up",
                up_geomean_fold=3.52,
            ),
            "IFN_response": TherapyAxisScore(
                therapy_class="IFN_response",
                state="up",
                up_geomean_fold=2.73,
            ),
        },
    }
    narrative = compose_disease_state_narrative(analysis)
    # Core clinical call must be present.
    assert "Castrate-resistant" in narrative
    assert "neuroendocrine" in narrative.lower()
    # Each collapsed AR target should be cited in evidence.
    for g in ("KLK3", "KLK2", "NKX3-1"):
        assert g in narrative
    # EMT + hypoxia cross-axis must land.
    assert "EMT" in narrative and "hypoxia" in narrative
    # IFN active must be flagged so users discount MHC-I fold changes.
    assert "IFN" in narrative


def test_compose_disease_state_empty_when_no_pattern():
    """Generic samples with nothing notable should not produce a
    disease-state narrative — callers skip the section when empty.
    """
    from trufflepig.main import compose_disease_state_narrative

    analysis = {
        "cancer_type": "PRAD",
        "purity": {"overall_estimate": 0.7, "components": {}},
        "therapy_response_scores": {},
    }
    assert compose_disease_state_narrative(analysis) == ""


def test_recommended_targets_skips_tme_dominant_rows():
    """#79: tme_dominant rows must not appear in the Recommended
    Targets Summary; they're called out as excluded."""
    import pandas as pd

    purity = {
        "overall_estimate": 0.6,
        "overall_lower": 0.5,
        "overall_upper": 0.7,
    }
    analysis = {
        "sample_mode": "solid",
        "cancer_type": "PRAD",
        "mhc1": {"HLA-A": 100, "HLA-B": 200, "HLA-C": 80, "B2M": 300},
    }
    ranges_df = pd.DataFrame(
        [
            # TME-dominant top row → must be filtered from the summary
            {
                "symbol": "CD74",
                "median_est": 1580,
                "observed_tpm": 1580,
                "est_1": 1156,
                "est_9": 1580,
                "pct_cancer_median": 1.5,
                "tcga_percentile": 0.94,
                "is_surface": True,
                "is_cta": False,
                "tme_explainable": True,
                "tme_dominant": True,
                "excluded_from_ranking": False,
                "therapies": "",
                "attr_tumor_tpm": 180.0,
                "attr_tumor_tpm_low": 80.0,
                "attr_tumor_tpm_high": 240.0,
                "attr_tumor_fraction": 0.11,
                "attr_tumor_fraction_low": 0.05,
                "attr_tumor_fraction_high": 0.16,
                "attr_support_fraction": 0.0,
                "attr_top_compartment": "myeloid",
                "attr_top_compartment_tpm": 1100.0,
                "max_healthy_tpm": 2000,
                "tme_tpm_lo": 0.1,
                "tme_tpm_med": 0.2,
                "tme_tpm_hi": 0.3,
                "cohort_prior_tpm": 1400,
                "tme_only_tpm": 1100,
                "matched_normal_tpm": 0,
                "matched_normal_tissue": "",
                "matched_normal_fraction": 0.0,
                "estimation_path": "clamped",
                "low_confidence_tumor": True,
                "category": "therapy_target",
                **{f"est_{i + 1}": 1156 + i * 50 for i in range(9)},
            },
            # Clean ADAM9 → SHOULD appear in the summary
            {
                "symbol": "ADAM9",
                "median_est": 998,
                "observed_tpm": 825,
                "est_1": 696,
                "est_9": 2179,
                "pct_cancer_median": 7.5,
                "tcga_percentile": 1.0,
                "is_surface": True,
                "is_cta": False,
                "tme_explainable": False,
                "tme_dominant": False,
                "excluded_from_ranking": False,
                "therapies": "ADC",
                "attr_tumor_tpm": 150.0,
                "attr_tumor_tpm_low": 130.0,
                "attr_tumor_tpm_high": 175.0,
                "attr_tumor_fraction": 0.62,
                "attr_tumor_fraction_low": 0.54,
                "attr_tumor_fraction_high": 0.69,
                "attr_support_fraction": 1.0,
                "attr_top_compartment": "tumor",
                "attr_top_compartment_tpm": 150.0,
                "max_healthy_tpm": 300,
                "tme_tpm_lo": 0.1,
                "tme_tpm_med": 0.2,
                "tme_tpm_hi": 0.3,
                "cohort_prior_tpm": 100,
                "tme_only_tpm": 150,
                "matched_normal_tpm": 0,
                "matched_normal_tissue": "",
                "matched_normal_fraction": 0.0,
                "estimation_path": "tme_only",
                "low_confidence_tumor": False,
                "category": "therapy_target",
                **{f"est_{i + 1}": 696 + i * 150 for i in range(9)},
            },
        ]
    )

    tmp_prefix = "/tmp/target_test"
    import os

    if os.path.exists(f"{tmp_prefix}-targets.md"):
        os.remove(f"{tmp_prefix}-targets.md")
    _write_target_report(
        ranges_df,
        analysis,
        tmp_prefix,
        cancer_type="PRAD",
        purity_result=purity,
    )
    with open(f"{tmp_prefix}-targets.md", encoding="utf-8") as target_file:
        targets = target_file.read()

    # The Recommended Targets section must not list CD74 as a best
    # surface target — it was low-confidence flagged.
    recs_block = targets.split("## Recommended Targets Summary")[-1]
    assert "**Best surface targets**" in recs_block
    # Clean ADAM9 should be there
    assert "ADAM9" in recs_block
    # CD74 is in the full targets table above but NOT in the
    # recommendations block
    assert "CD74" not in recs_block.split("**Best CTA targets**")[0]


def test_target_report_explains_blocked_fn1_pyx201_call():
    """FN1 should carry an explicit report caveat when the curated
    PYX-201 hook is withheld for lack of EDB+ transcript support."""

    purity = {
        "overall_estimate": 0.6,
        "overall_lower": 0.5,
        "overall_upper": 0.7,
    }
    analysis = {
        "sample_mode": "solid",
        "cancer_type": "PRAD",
        "mhc1": {"HLA-A": 100, "HLA-B": 200, "HLA-C": 80, "B2M": 300},
    }
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "ADAM9",
                "median_est": 998,
                "observed_tpm": 825,
                "est_1": 696,
                "est_9": 2179,
                "pct_cancer_median": 7.5,
                "tcga_percentile": 1.0,
                "is_surface": True,
                "is_cta": False,
                "tme_explainable": False,
                "tme_dominant": False,
                "excluded_from_ranking": False,
                "therapies": "ADC",
                "therapy_supported": True,
                "therapy_support_note": "",
                "max_healthy_tpm": 300,
                "tme_tpm_lo": 0.1,
                "tme_tpm_med": 0.2,
                "tme_tpm_hi": 0.3,
                "cohort_prior_tpm": 100,
                "tme_only_tpm": 150,
                "matched_normal_tpm": 0,
                "matched_normal_tissue": "",
                "matched_normal_fraction": 0.0,
                "estimation_path": "tme_only",
                "low_confidence_tumor": False,
                "category": "therapy_target",
                **{f"est_{i + 1}": 696 + i * 150 for i in range(9)},
            },
            {
                "symbol": "FN1",
                "median_est": 260,
                "observed_tpm": 180,
                "est_1": 180,
                "est_9": 340,
                "pct_cancer_median": 0.9,
                "tcga_percentile": 0.65,
                "is_surface": False,
                "is_cta": False,
                "tme_explainable": False,
                "tme_dominant": False,
                "excluded_from_ranking": False,
                "therapies": "",
                "therapy_supported": False,
                "therapy_support_note": (
                    "PYX-201 (NCT05720117) targets EDB+ FN1; bulk gene-level FN1 alone "
                    "is not sufficient evidence because transcript-level data is unavailable."
                ),
                "max_healthy_tpm": 500,
                "tme_tpm_lo": 0.1,
                "tme_tpm_med": 0.2,
                "tme_tpm_hi": 0.3,
                "cohort_prior_tpm": 120,
                "tme_only_tpm": 80,
                "matched_normal_tpm": 0,
                "matched_normal_tissue": "",
                "matched_normal_fraction": 0.0,
                "estimation_path": "tme_only",
                "low_confidence_tumor": False,
                "category": "other",
                **{f"est_{i + 1}": 180 + i * 20 for i in range(9)},
            },
        ]
    )

    tmp_prefix = "/tmp/target_test_fn1"
    import os

    if os.path.exists(f"{tmp_prefix}-targets.md"):
        os.remove(f"{tmp_prefix}-targets.md")
    _write_target_report(
        ranges_df,
        analysis,
        tmp_prefix,
        cancer_type="PRAD",
        purity_result=purity,
    )
    with open(f"{tmp_prefix}-targets.md", encoding="utf-8") as target_file:
        targets = target_file.read()

    assert "PYX-201 (NCT05720117) targets EDB+ FN1" in targets
    assert "Landscape cautions" in targets


def test_target_report_falls_back_to_mixed_source_surface_targets(tmp_path):
    """If no surface row stays tumor-supported, keep the best mixed-source
    options visible with explicit caveats."""

    purity = {
        "overall_estimate": 0.16,
        "overall_lower": 0.07,
        "overall_upper": 0.24,
    }
    analysis = {
        "sample_mode": "solid",
        "cancer_type": "PRAD",
        "mhc1": {"HLA-A": 100, "HLA-B": 200, "HLA-C": 80, "B2M": 300},
    }
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "FOLH1",
                "median_est": 87.0,
                "observed_tpm": 87.0,
                "est_1": 60.0,
                "est_9": 100.0,
                "pct_cancer_median": 6.5,
                "tcga_percentile": 1.0,
                "is_surface": True,
                "is_cta": False,
                "tme_explainable": True,
                "tme_dominant": False,
                "excluded_from_ranking": False,
                "therapies": "radioligand",
                "max_healthy_tpm": 46.0,
                "tme_tpm_lo": 0.1,
                "tme_tpm_med": 0.2,
                "tme_tpm_hi": 0.3,
                "cohort_prior_tpm": 10.0,
                "tme_only_tpm": 12.0,
                "matched_normal_tpm": 46.0,
                "matched_normal_tissue": "prostate",
                "matched_normal_fraction": 0.53,
                "estimation_path": "matched_normal_split",
                "low_confidence_tumor": False,
                "category": "therapy_target",
                "attr_tumor_tpm": 34.0,
                "attr_tumor_tpm_low": 18.0,
                "attr_tumor_tpm_high": 40.0,
                "attr_tumor_fraction": 0.39,
                "attr_tumor_fraction_low": 0.21,
                "attr_tumor_fraction_high": 0.45,
                "attr_support_fraction": 0.33,
                "attr_top_compartment": "matched_normal_prostate",
                "attr_top_compartment_tpm": 46.0,
                "matched_normal_over_predicted": False,
                "broadly_expressed": False,
                "n_healthy_tissues_expressed": 0,
                **{f"est_{i + 1}": 60.0 + i * 5.0 for i in range(9)},
            }
        ]
    )

    prefix = str(tmp_path / "sample")
    _write_target_report(
        ranges_df, analysis, prefix, cancer_type="PRAD", purity_result=purity
    )
    targets = (tmp_path / "sample-targets.md").read_text()
    recs_block = targets.split("## Recommended Targets Summary")[-1]

    assert "no surface target stayed tumor-supported" in targets
    assert "**Best surface targets**" in recs_block
    assert "FOLH1" in recs_block
    assert "mixed-source rather than tumor-supported" in recs_block


def test_background_dominant_curated_therapy_row_is_audit_only(tmp_path):
    """Issue #105: host-attributed disease-relevant genes should not read as active opportunities."""

    purity = {
        "overall_estimate": 0.64,
        "overall_lower": 0.55,
        "overall_upper": 0.72,
    }
    analysis = {
        "sample_mode": "solid",
        "cancer_type": "BLCA",
        "cancer_name": "Bladder Urothelial Carcinoma",
        "mhc1": {"HLA-A": 100, "HLA-B": 200, "HLA-C": 80, "B2M": 300},
        "call_summary": {
            "label_options": ["BLCA"],
            "label_display": "BLCA (Bladder Urothelial Carcinoma)",
            "reported_site": "liver-associated host context",
        },
    }
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "FGFR3",
                "gene_id": "ENSG00000068078",
                "category": "therapy_target",
                "observed_tpm": 10.6,
                "median_est": 11.0,
                "est_1": 0.0,
                "est_9": 11.0,
                "pct_cancer_median": 1.0,
                "tcga_percentile": 0.5,
                "is_surface": True,
                "is_cta": False,
                "tme_explainable": True,
                "tme_dominant": True,
                "low_confidence_tumor": True,
                "excluded_from_ranking": False,
                "therapies": "",
                "max_healthy_tpm": 16.0,
                "tme_tpm_lo": 0.1,
                "tme_tpm_med": 0.2,
                "tme_tpm_hi": 0.3,
                "cohort_prior_tpm": 0.0,
                "tme_only_tpm": 1.2,
                "matched_normal_tpm": 0.0,
                "matched_normal_tissue": "",
                "matched_normal_fraction": 0.0,
                "estimation_path": "clamped",
                "attr_tumor_tpm": 0.0,
                "attr_tumor_tpm_low": 0.0,
                "attr_tumor_tpm_high": 0.0,
                "attr_tumor_fraction": 0.0,
                "attr_tumor_fraction_low": 0.0,
                "attr_tumor_fraction_high": 0.0,
                "attr_support_fraction": 0.0,
                "attr_top_compartment": "hepatocyte",
                "attr_top_compartment_tpm": 1.2,
                "matched_normal_over_predicted": False,
                "broadly_expressed": True,
                "n_healthy_tissues_expressed": 20,
                **{f"est_{i + 1}": 0.0 for i in range(9)},
            }
        ]
    )

    prefix = str(tmp_path / "pfo017-liver")
    _write_target_report(
        ranges_df,
        analysis,
        prefix,
        cancer_type="BLCA",
        purity_result=purity,
    )
    targets = (tmp_path / "pfo017-liver-targets.md").read_text()

    unsupported_heading = "### Other curated rows — not supported by this sample"
    assert unsupported_heading in targets
    active_block = targets.split(unsupported_heading, 1)[0]
    audit_block = targets.split(unsupported_heading, 1)[1]
    assert "erdafitinib" not in active_block
    assert "FGFR3" in audit_block
    assert "erdafitinib" in audit_block
    assert "not sample-supported; negative/background evidence" in audit_block


def test_ci_confidence_tier_buckets():
    from trufflepig.main import _ci_confidence_tier

    assert _ci_confidence_tier(0.58, 0.70) == "high"  # span 0.12
    assert _ci_confidence_tier(0.40, 0.70) == "moderate"  # span 0.30
    assert _ci_confidence_tier(0.19, 1.00) == "low"  # span 0.81
    assert _ci_confidence_tier(None, 0.5) == "unknown"


def test_filter_quality_flags_rewrites_mt_warning_under_exome():
    """#77: MT 'Suspicious' warning must be rewritten as informational
    when the inferred library prep (exome capture / poly-A) already
    explains MT absence."""
    from trufflepig.main import _filter_quality_flags_against_context
    from trufflepig.sample_context import SampleContext

    flags = [
        "Suspicious MT fraction: 0.0% (n_mt_found=13/15) — mitochondrial "
        "genes appear filtered or renamed in the input",
        "Some other warning that must pass through",
    ]
    ctx = SampleContext(library_prep="exome_capture")
    out = _filter_quality_flags_against_context(flags, ctx)
    assert len(out) == 2
    assert "Suspicious MT fraction" not in out[0]
    assert "informational" in out[0]
    assert out[1] == "Some other warning that must pass through"

    # Under total_rna the warning should pass through unchanged.
    ctx2 = SampleContext(library_prep="total_rna")
    out2 = _filter_quality_flags_against_context(flags, ctx2)
    assert "Suspicious MT fraction" in out2[0]


def test_cli_analyze_rejects_invalid_met_site(monkeypatch, tmp_path):
    """CLI-level validation: analyze() should raise ValueError on
    an unknown --met-site value rather than silently ignoring it."""
    from trufflepig import main as cli_mod

    monkeypatch.setattr(
        cli_mod, "load_expression_data", lambda *a, **k: pd.DataFrame({"x": [1]})
    )
    out_dir = str(tmp_path / "out")
    with pytest.raises(ValueError):
        cli_mod.analyze(
            "input.csv",
            output_dir=out_dir,
            met_site="not_a_site",
        )


def _axis(up=None, down=None, state="down"):
    return SimpleNamespace(
        up_geomean_fold=up,
        down_geomean_fold=down,
        state=state,
        up_genes_measured=5,
        down_genes_measured=5,
        message="",
    )


def test_disease_state_text_shares_render_source_with_pathway_figure():
    """Belief-consistency (plan §2.4/§2.5): the disease-state text must not deny a
    pattern the therapy-pathway-state figure visibly shows. Both route through the
    single ``pathway_state_figure_axes`` predicate, so whenever the figure would
    render an axis the text points at the figure instead of asserting nothing
    passed thresholds."""
    from trufflepig.reporting import (
        pathway_state_figure_axes,
        report_disease_state_text,
    )

    # Axes with a measurable fold → the figure renders these rows.
    scores = {"IFN_response": _axis(up=0.35), "EMT": _axis(down=1.9, state="up")}
    axes = pathway_state_figure_axes(scores)
    assert axes == ["IFN_response", "EMT"]  # single source of "figure has content"

    text = report_disease_state_text("", {"therapy_response_scores": scores})
    # The figure shows the axes, so the text must NOT claim nothing passed.
    assert "passed reporting thresholds" not in text
    assert "therapy-pathway-state figure" in text

    # A non-empty synthesized narrative always wins verbatim.
    assert (
        report_disease_state_text("AR-active CRPC.", {"therapy_response_scores": scores})
        == "AR-active CRPC."
    )


def test_disease_state_text_keeps_bounded_no_pattern_when_figure_empty():
    """When no axis has a measurable fold the figure does not render, so the bounded
    'nothing passed thresholds' statement is truthful and is preserved."""
    from trufflepig.reporting import (
        pathway_state_figure_axes,
        report_disease_state_text,
    )

    scores = {"IFN_response": _axis(up=None, down=None)}
    assert pathway_state_figure_axes(scores) == []
    text = report_disease_state_text("", {"therapy_response_scores": scores})
    assert "No strong RNA-defined therapy-exposure" in text

    # With a separate active-pathway inference, the bounded statement defers to it.
    text2 = report_disease_state_text(
        "",
        {"therapy_response_scores": scores, "pathway_activity_inferences": [{"label": "MAPK"}]},
    )
    assert "active pathway evidence is summarized separately" in text2.lower()
