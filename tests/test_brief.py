"""Tests for the two-tier brief / actionable handoff (#111)."""

import pandas as pd

from trufflepig.brief import (
    build_actionable as _build_actionable,
    build_brief as _build_brief,
    build_summary as _build_summary,
    biomarker_expression_is_not_eligibility,
    _expression_independent_evidence_gap,
    _format_therapy_bullet,
    _empty_therapy_shortlist_message,
    _lineage_panel_evidence_line,
    _lineage_panel_subtype_reasoning_line,
    _format_cta_outlier_bullet,
    _notable_cta_outliers,
    _shortlist_omission_note,
    _top_therapies,
    mismatch_repair_summary_line,
)
from trufflepig.confidence import ConfidenceTier
from trufflepig.report_view import build_report_view


def _render_call(builder, analysis, *args, **kwargs):
    if "report_view" not in kwargs:
        finalized = dict(analysis)
        finalized.setdefault("cancer_type", kwargs.get("cancer_code") or "UNKNOWN")
        finalized.setdefault("sample_mode", "solid")
        finalized.setdefault("purity", {})
        kwargs["report_view"] = build_report_view(finalized)
    return builder(analysis, *args, **kwargs)


def build_summary(analysis, *args, **kwargs):
    return _render_call(_build_summary, analysis, *args, **kwargs)


def build_actionable(analysis, *args, **kwargs):
    return _render_call(_build_actionable, analysis, *args, **kwargs)


def build_brief(analysis, *args, **kwargs):
    return _render_call(_build_brief, analysis, *args, **kwargs)


def _lineage_panel_evidence(top_panel, *, promoted=False, code="", blockers=()):
    return {
        "lineage_panel_evidence": {
            "top_score": 0.90,
            "top_panel": top_panel,
            "top_panel_program_note": "biliary epithelial program",
            "promotion": {
                "promoted": promoted,
                "code": code,
                "blockers": list(blockers),
            },
        }
    }


def test_empty_shortlist_does_not_mislabel_every_present_target_as_nontumor():
    targets = pd.DataFrame([{"symbol": "CDK4", "agent": "palbociclib"}])
    ranges = pd.DataFrame(
        [
            {
                "symbol": "CDK4",
                "observed_tpm": 63.0,
                "attr_tumor_tpm": 53.0,
                "attr_tumor_fraction": 0.84,
            }
        ]
    )

    message = _empty_therapy_shortlist_message(targets, ranges)

    assert "did not meet the shortlist's" in message
    assert "non-tumor-supported" not in message


def test_notable_cta_summary_prioritizes_estimated_patient_tumor_signal():
    ranges = pd.DataFrame(
        [
            {
                "symbol": "BACKGROUND_CTA",
                "is_cta": True,
                "observed_tpm": 500.0,
                "attr_tumor_tpm": 5.0,
                "attr_tumor_tpm_low": 0.0,
                "attr_tumor_tpm_high": 20.0,
                "tcga_percentile": 0.99,
            },
            {
                "symbol": "TUMOR_CTA",
                "is_cta": True,
                "observed_tpm": 100.0,
                "attr_tumor_tpm": 60.0,
                "attr_tumor_tpm_low": 30.0,
                "attr_tumor_tpm_high": 100.0,
                "tcga_percentile": 0.95,
            },
        ]
    )

    rows = _notable_cta_outliers(ranges)
    assert [row["symbol"] for row in rows] == ["TUMOR_CTA", "BACKGROUND_CTA"]
    bullet = _format_cta_outlier_bullet(rows[0])
    assert "100 patient bulk TPM" in bullet
    assert "60 estimated patient tumor TPM" in bullet
    assert "RNA model interval 30-100" in bullet
    assert "check HLA" in bullet


def test_subtype_line_suppressed_when_panel_blocked_against_call():
    # A high-scoring CHOL panel that was held back under a PRAD call must not
    # render as the call's "subtype" — that contradicts the cancer call.
    analysis = _lineage_panel_evidence(
        "CHOL", blockers=["CHOL is not among the top-5 first-pass RNA candidates"]
    )
    assert _lineage_panel_subtype_reasoning_line(analysis, "PRAD") is None


def test_subtype_line_shown_when_panel_consistent_with_call():
    # Promoted-and-adopted, or confirmed-without-blockers, both render.
    promoted = _lineage_panel_evidence("CHOL", promoted=True, code="CHOL")
    assert "**Subtype:** CHOL" in (
        _lineage_panel_subtype_reasoning_line(promoted, "CHOL") or ""
    )
    confirmed = _lineage_panel_evidence("CHOL")
    assert "**Subtype:** CHOL" in (
        _lineage_panel_subtype_reasoning_line(confirmed, "CHOL") or ""
    )


def test_subtype_line_suppressed_when_panel_proposes_unadopted_label():
    # Panel proposed a different label that wasn't adopted as the report scope.
    analysis = _lineage_panel_evidence("CHOL", promoted=True, code="CHOL")
    assert _lineage_panel_subtype_reasoning_line(analysis, "PRAD") is None


def test_lineage_panel_evidence_marks_promoted_unadopted_label_as_competing():
    analysis = _lineage_panel_evidence("BRCA_BASAL", promoted=True, code="BRCA")
    line = _lineage_panel_evidence_line(analysis, "ADCC") or ""
    assert "competing BRCA lineage signal" in line
    assert "did not override the ADCC call" in line


def test_lineage_panel_evidence_distinguishes_tumor_residual_from_host():
    analysis = _lineage_panel_evidence("BLCA_LUMINAL", promoted=True, code="BLCA")
    analysis["lineage_panel_evidence"]["decomposition_attribution"] = {
        "status": "tumor_residual",
        "evaluated_marker_count": 7,
        "tumor_dominant_count": 7,
    }

    line = _lineage_panel_evidence_line(analysis, "BLCA") or ""

    assert "supports the BLCA call" in line
    assert "7/7 positive markers" in line
    assert (
        "estimated tumor residual rather than external stromal, immune, "
        "and tissue reference background" in line
    )


def test_lineage_panel_explains_background_resolved_low_marker_violation():
    analysis = _lineage_panel_evidence(
        "CRC",
        blockers=["top panel is not a complete positive/negative-marker program"],
    )
    analysis["lineage_panel_evidence"].update(
        {
            "top_rationale": (
                "6/6 required markers in cohort range; "
                "low-marker violations: DES=6039>10"
            ),
            "decomposition_attribution": {
                "status": "tumor_residual",
                "evaluated_marker_count": 6,
                "tumor_dominant_count": 6,
            },
        }
    )
    analysis["cancer_type_decision"] = {
        "status": "resolved",
        "supported_code": "CRC",
        "background_separation_confirmed": True,
        "background_attributed_genes": ["DES"],
    }

    line = _lineage_panel_evidence_line(analysis, "CRC") or ""

    assert "bulk panel remains incomplete because of DES" in line
    assert (
        "benign structural-tissue reference signal can explain the expected-low violation"
        in line
    )
    assert "complete panel and ontology programs agree" in line
    assert "not a claim that every measured DES transcript is non-tumor" in line
    assert "noted, did not change the call" not in line


def _make_analysis(
    purity_point=0.28,
    ci_low=0.19,
    ci_high=0.40,
    purity_tier_label="moderate",
    degradation="mild",
    library_prep="exome_capture",
    preservation="ffpe",
):
    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.library_prep = library_prep
    ctx.library_prep_confidence = 0.9
    ctx.preservation = preservation
    ctx.preservation_confidence = 0.85
    ctx.degradation_severity = degradation
    ctx.degradation_index = 0.6
    ctx.missing_mt = False
    ctx.signals = {}
    ctx.flags = []

    return {
        "cancer_type": "PRAD",
        "cancer_name": "Prostate adenocarcinoma",
        "sample_mode": "solid",
        "purity": {
            "overall_estimate": purity_point,
            "overall_lower": ci_low,
            "overall_upper": ci_high,
        },
        "purity_confidence": ConfidenceTier(
            tier=purity_tier_label,
            reasons=(
                ["moderate purity CI span (21 pp)", "low-purity regime (28%)"]
                if purity_tier_label in {"moderate", "low"}
                else []
            ),
        ),
        "sample_context": ctx,
        "therapy_response_scores": {},
    }


def _make_ranges_df():
    return pd.DataFrame(
        [
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
            },
            {
                "symbol": "STEAP1",
                "observed_tpm": 78.0,
                "attribution": {"fibroblast": 10.0, "endothelial": 6.0},
                "attr_tumor_tpm": 62.0,
                "attr_tumor_fraction": 0.79,
                "attr_top_compartment": "fibroblast",
                "attr_top_compartment_tpm": 10.0,
                "tme_dominant": False,
                "tme_explainable": False,
            },
            {
                "symbol": "DLL3",
                "observed_tpm": 0.5,
                "attribution": {},
                "attr_tumor_tpm": 0.0,
                "attr_tumor_fraction": 0.0,
                "attr_top_compartment": "",
                "attr_top_compartment_tpm": 0.0,
                "tme_dominant": False,
                "tme_explainable": False,
            },
            {
                "symbol": "AR",
                "observed_tpm": 50.0,
                "attribution": {"endothelial": 2.0},
                "attr_tumor_tpm": 48.0,
                "attr_tumor_fraction": 0.96,
                "attr_top_compartment": "endothelial",
                "attr_top_compartment_tpm": 2.0,
                "tme_dominant": False,
                "tme_explainable": False,
            },
        ]
    )


def test_summary_purity_reads_frozen_snapshot_not_stale_live_dict():
    """Summary purity comes only from the explicit frozen report."""
    from trufflepig.report_view import build_report_view

    analysis = _make_analysis(purity_point=0.78, ci_low=0.70, ci_high=0.85)
    report_view = build_report_view(
        {
            **analysis,
            "purity": {
                "overall_estimate": 0.10,
                "overall_lower": 0.06,
                "overall_upper": 0.16,
            }
        }
    )
    md = build_summary(
        analysis,
        _make_ranges_df(),
        cancer_code="PRAD",
        disease_state="",
        sample_id="sample_X",
        report_view=report_view,
    )
    assert "**Estimated tumor fraction (RNA model):** 10% (model interval 6%–16%" in md
    assert "78%" not in md  # the stale candidate purity never reaches the report


def test_actionable_purity_reads_frozen_snapshot_not_stale_live_dict():
    """The actionable report uses the same explicit frozen purity."""
    from trufflepig.report_view import build_report_view

    analysis = _make_analysis(purity_point=0.78, ci_low=0.70, ci_high=0.85)
    report_view = build_report_view(
        {
            **analysis,
            "purity": {
                "overall_estimate": 0.10,
                "overall_lower": 0.06,
                "overall_upper": 0.16,
            }
        }
    )
    md = build_actionable(
        analysis,
        _make_ranges_df(),
        cancer_code="PRAD",
        disease_state="",
        sample_id="sample_X",
        report_view=report_view,
    )
    assert "Purity point estimate: **10%** (model interval 6%–16%" in md
    assert "78%" not in md  # the stale candidate purity never reaches the actionable review


def test_actionable_purity_degrades_to_bare_point_when_interval_missing():
    """A purity estimate without bounds renders as a bare point."""
    from trufflepig.report_view import build_report_view

    analysis = _make_analysis()
    analysis["purity"] = {"overall_estimate": 0.30}  # point only, no CI bounds
    report_view = build_report_view(analysis)
    md = build_actionable(
        analysis,
        _make_ranges_df(),
        cancer_code="PRAD",
        disease_state="",
        sample_id="sample_X",
        report_view=report_view,
    )
    assert "Purity point estimate: **30%**." in md
    assert "model interval" not in md  # no interval clause when a bound is missing


def test_reports_present_discordant_purity_estimators_as_separate_scenarios():
    analysis = _make_analysis(
        purity_point=0.05,
        ci_low=0.01,
        ci_high=0.12,
        purity_tier_label="low",
    )
    analysis["purity"].update(
        {
            "quantitative_status": "discordant_estimators",
            "operational_estimate_only": True,
            "estimator_scenarios": [
                {
                    "source": "lineage_panel",
                    "estimate": 0.05,
                    "lower": 0.01,
                    "upper": 0.12,
                },
                {
                    "source": "signature",
                    "estimate": 0.43,
                    "lower": 0.32,
                    "upper": 0.55,
                },
            ],
        }
    )
    report_view = build_report_view(analysis)
    # Simulate later accidental mutation of the working analysis dictionary.
    # Every report field must still come from the one frozen conclusion.
    analysis["purity"].update(
        {
            "overall_estimate": 0.78,
            "overall_lower": 0.70,
            "overall_upper": 0.85,
            "quantitative_status": "resolved",
            "estimator_scenarios": [],
        }
    )

    summary = build_summary(
        analysis,
        _make_ranges_df(),
        cancer_code="PRAD",
        disease_state="",
        sample_id="sample_X",
        report_view=report_view,
    )
    actionable = build_actionable(
        analysis,
        _make_ranges_df(),
        cancer_code="PRAD",
        disease_state="",
        sample_id="sample_X",
        report_view=report_view,
    )

    assert "**Estimated tumor fraction (RNA model):** quantitatively unresolved" in summary
    assert "selected operational model uses 5% [1%–12%]" in summary
    assert "healthy-tissue lineage reference model: 5% [1%–12%]" in summary
    assert "upstream expression model: 43% [32%–55%]" in summary
    assert "Purity is **quantitatively unresolved**" in actionable
    assert "not a consensus tumor-purity estimate" in actionable
    assert "model interval 1%–43%" not in summary + actionable


def test_brief_is_compact():
    analysis = _make_analysis()
    ranges_df = _make_ranges_df()
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="Castrate-resistant pattern.",
        sample_id="sample_X",
    )
    lines = md.splitlines()
    # ≤ 40 lines is the contract.
    assert len(lines) <= 40, f"brief is {len(lines)} lines, must be ≤ 40:\n{md}"

    # Key structural elements present.
    # File was renamed brief → summary in 4.41.0; header tracks the name.
    assert "# Summary" in md
    assert "**Cancer call:**" in md
    assert "**Estimated tumor fraction (RNA model):**" in md
    assert "model interval" in md
    assert "(CI " not in md
    assert "**Disease state:**" in md
    assert "Top candidate therapies" in md


def test_summary_surfaces_rna_qc_and_prad_stromal_pitfall():
    analysis = _make_analysis()
    analysis["cancer_call_rescue"] = {
        "kind": "low_purity_prad_stromal_context",
        "message": "Prostate context with stromal SARC pitfall.",
    }
    analysis["rna_quant_qc"] = {
        "available": True,
        "summary": "Salmon mapping 33.5%; 12,612/35,037 genes >=1 TPM",
        "warnings": [
            "Salmon mapping rate is low (33.5%). Interpret RNA-derived calls cautiously."
        ],
    }

    md = build_summary(
        analysis,
        _make_ranges_df(),
        cancer_code="PRAD",
        disease_state="",
        sample_id="sample_X",
    )

    assert "**RNA quant QC:** Salmon mapping 33.5%" in md
    assert "**QC/call pitfall:** prostate tissue/context is present" in md
    assert "RNA-inferred PRAD context rescue" in md


def test_summary_explains_a_ceiling_purity_estimate():
    analysis = _make_analysis()
    analysis["purity"] = {
        "overall_estimate": 1.0,
        "overall_lower": 0.83,
        "overall_upper": 1.0,
    }

    md = build_summary(
        analysis,
        _make_ranges_df(),
        cancer_code="PRAD",
        disease_state="",
        sample_id="sample_X",
    )

    assert "**Estimated tumor fraction (RNA model):** 100%" in md
    assert "Do not interpret this as literal 100% tumor cellularity" in md


def test_summary_uses_generic_text_for_orphan_context_rescue():
    analysis = _make_analysis()
    analysis["cancer_type"] = "BLCA"
    analysis["cancer_name"] = "Bladder Urothelial Carcinoma"
    analysis["cancer_call_rescue"] = {
        "kind": "coarse_tcga_orphan_context",
        "recommended_code": "BLCA",
        "competing_code": "ESCA",
        "context_basis": "raw_signal_dominance",
        "message": (
            "Tissue composition screen and direct cancer evidence support BLCA; "
            "suspending the orphan family penalty for the auto-detected call."
        ),
    }

    md = build_summary(
        analysis,
        _make_ranges_df(),
        cancer_code="BLCA",
        disease_state="",
        sample_id="sample_X",
    )

    assert "RNA-inferred BLCA (Bladder Urothelial Carcinoma) context rescue" in md
    assert "orphan-family penalty" in md
    assert "PRAD context rescue" not in md
    assert "prostate tissue/context" not in md


def test_summary_surfaces_inferred_met_site_context():
    analysis = _make_analysis()
    analysis["cancer_type"] = "BLCA"
    analysis["cancer_name"] = "Bladder Urothelial Carcinoma"
    analysis["inferred_site_context"] = {
        "site": "liver",
        "tissue": "liver",
        "score": 0.964,
        "primary_tissue": "urinary_bladder",
        "primary_tissue_score": 0.796,
    }

    md = build_summary(
        analysis,
        _make_ranges_df(),
        cancer_code="BLCA",
        disease_state="",
        sample_id="sample_X",
    )

    assert "**Inferred site context:** likely liver metastatic host/background" in md
    assert "inferred from expression, not supplied as a user constraint" in md


def test_summary_marks_supplied_cancer_type_basis():
    analysis = _make_analysis()
    analysis["analysis_constraints"] = {"cancer_type": "PRAD"}
    analysis["cancer_type_source"] = "user-specified"
    ranges_df = _make_ranges_df()

    md = build_summary(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )

    assert "Cancer-type basis" in md
    assert "externally supplied PRAD (Prostate Adenocarcinoma) sets the report label" in md
    assert "RNA evidence is used downstream for confidence" in md
    assert "Clinical interpretation still needs external patient context" in md
    assert "RNA-inferred — treat it as a hypothesis" not in md


def test_summary_names_background_separated_cancer_type_basis():
    from trufflepig.healthy_vs_tumor import TissueCompositionSignal

    analysis = _make_analysis()
    analysis.update(
        {
            "cancer_type": "CRC",
            "cancer_type_source": "auto-detected",
            "report_scope_cancer_type": "CRC",
            "cancer_type_evidence": {
                "selected": {
                    "cancer_type": "CRC",
                    "selected_by": "entity_evidence_consensus",
                    "entity_evidence_consensus": {
                        "decomposition_decision_was_decisive": True,
                    },
                }
            },
            "cancer_type_decision": {
                "status": "resolved",
                "supported_code": "CRC",
                "background_separation_confirmed": True,
            },
            "cancer_type_decision_refit": {
                "accepted": True,
                "previous_cancer_type": "SARC_DDLPS",
            },
            "candidate_trace": [
                {"code": "SARC_DDLPS", "support_geomean": 0.55},
                {"code": "CRC", "support_geomean": 0.44},
            ],
            "healthy_vs_tumor": TissueCompositionSignal(
                top_normal_tissues=[("smooth_muscle_nTPM", 0.86)],
                top_tcga_cohorts=[("SARC_TPM", 0.81)],
                cancer_hint="possibly-tumor",
                structural_ambiguity=True,
            ),
        }
    )

    md = build_summary(
        analysis,
        _make_ranges_df(),
        cancer_code="CRC",
        disease_state="",
    )

    assert "Cancer-type basis" in md
    assert "bulk profile initially favored SARC_DDLPS" in md
    assert "tissue-composition screen was dominated by smooth muscle" in md
    assert "candidate-independent decomposition recovered" in md
    assert "complete and invariant CRC (Colorectal Adenocarcinoma)" in md
    assert "refitted for that final scope reproduced it" in md
    assert "remains only in the audit differential" in md
    assert "does not drive downstream interpretation" in md
    assert "**Retained RNA differential:**" not in md


def test_summary_keeps_parent_decision_at_parent_scope():
    analysis = _make_analysis()
    analysis.update(
        {
            "cancer_type": "READ",
            "cancer_type_source": "auto-detected",
            "report_scope_cancer_type": "READ",
            "cancer_type_evidence": {
                "selected": {
                    "cancer_type": "READ",
                    "selected_by": "entity_evidence_consensus",
                    "entity_evidence_consensus": {
                        "decomposition_decision_was_decisive": False,
                    },
                }
            },
            "cancer_type_decision": {
                "status": "resolved",
                "supported_code": "CRC",
                "background_separation_confirmed": True,
            },
            "cancer_type_decision_refit": {"accepted": True},
        }
    )

    md = build_summary(
        analysis,
        _make_ranges_df(),
        cancer_code="READ",
        disease_state="",
    )

    assert "CRC (Colorectal Adenocarcinoma) as the supported tumor type" in md
    assert "establishes the broader branch but does not by itself establish" in md
    assert "READ" in md
    assert "complete and invariant READ" not in md


def test_summary_marks_supplied_cancer_type_rna_concordance():
    analysis = _make_analysis()
    analysis["analysis_constraints"] = {"cancer_type": "PRAD"}
    analysis["cancer_type_source"] = "user-specified"
    analysis["candidate_trace"] = [
        {"code": "PRAD", "support_geomean": 0.82},
        {"code": "BLCA", "support_geomean": 0.31},
        {"code": "COAD", "support_geomean": 0.25},
    ]
    ranges_df = _make_ranges_df()

    md = build_summary(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )

    assert (
        "**RNA classifier check:** expression-reference context is concordant "
        "with supplied "
        "PRAD (Prostate Adenocarcinoma)"
    ) in md
    assert "nearest RNA alternatives: BLCA, COAD" in md


def test_summary_compares_registry_child_against_parent_reference():
    analysis = _make_analysis()
    analysis["cancer_type"] = "SARC_SYN"
    analysis["cancer_name"] = "Synovial Sarcoma"
    analysis["analysis_constraints"] = {"cancer_type": "Synovial Sarcoma"}
    analysis["cancer_type_source"] = "user-specified"
    analysis["report_scope_cancer_type"] = "SARC_SYN"
    analysis["report_scope_parent_cancer_type"] = "SARC"
    analysis["candidate_trace"] = [
        {"code": "SARC", "support_geomean": 0.82},
        {"code": "BLCA", "support_geomean": 0.31},
    ]
    ranges_df = _make_ranges_df()

    md = build_summary(
        analysis,
        ranges_df,
        cancer_code="SARC_SYN",
        disease_state="",
    )

    assert "expression-reference context is concordant at the parent level" in md
    assert "SARC (Sarcoma) is top" in md
    assert "refined report label remains SARC_SYN (Synovial Sarcoma)" in md
    assert "nearest RNA alternatives: BLCA" in md


def test_summary_replaces_stale_sibling_context_with_shared_parent():
    analysis = _make_analysis()
    analysis.update(
        {
            "cancer_type": "BRCA_Basal",
            "cancer_name": "Basal-like breast carcinoma",
            "report_scope_cancer_type": "BRCA_Basal",
            "report_scope_parent_cancer_type": "BRCA_HER2",
            "reference_cancer_type": "BRCA_HER2",
            "reference_cancer_name": "HER2-enriched",
            "expression_reference_cancer_type": "BRCA_HER2",
            "cancer_type_source": "auto-detected",
            "analysis_constraints": {},
            "candidate_trace": [
                {"code": "BRCA", "support_geomean": 0.82},
            ],
            "fine_report_scope_inference": {
                "reference_cancer_type": "BRCA_HER2",
                "metrics": {"fine_reference_support": 0.91},
            },
        }
    )

    md = build_summary(
        analysis,
        _make_ranges_df(),
        cancer_code="BRCA_Basal",
        disease_state="",
    )

    assert "BRCA_Basal (Basal-like) as the fine label" in md
    assert "BRCA (Breast Invasive Carcinoma) expression-reference context" in md
    assert "BRCA is analysis context only, not an alternative diagnosis" in md
    assert "no sibling subtype context is carried" in md
    assert "BRCA_HER2" not in md


def test_supplied_summary_reads_resolved_parent_not_stale_sibling_field():
    analysis = _make_analysis()
    analysis.update(
        {
            "cancer_type": "BRCA_Basal",
            "cancer_name": "Basal-like breast carcinoma",
            "report_scope_cancer_type": "BRCA_Basal",
            "report_scope_parent_cancer_type": "BRCA_HER2",
            "reference_cancer_type": "BRCA_HER2",
            "reference_cancer_name": "HER2-enriched",
            "expression_reference_cancer_type": "BRCA_HER2",
            "cancer_type_source": "user-specified",
            "analysis_constraints": {"cancer_type": "BRCA_Basal"},
            "candidate_trace": [
                {"code": "BRCA", "support_geomean": 0.82},
                {"code": "BLCA", "support_geomean": 0.31},
            ],
        }
    )

    md = build_summary(
        analysis,
        _make_ranges_df(),
        cancer_code="BRCA_Basal",
        disease_state="",
    )

    assert "BRCA (Breast Invasive Carcinoma) is used as the parent expression context" in md
    assert "concordant at the parent level: BRCA (Breast Invasive Carcinoma) is top" in md
    assert "refined report label remains BRCA_Basal (Basal-like)" in md
    assert "BRCA_HER2" not in md


def test_summary_marks_supplied_cancer_type_rna_discordance():
    analysis = _make_analysis()
    analysis["cancer_type"] = "COAD"
    analysis["cancer_name"] = "Colon Adenocarcinoma"
    analysis["analysis_constraints"] = {"cancer_type": "COAD"}
    analysis["cancer_type_source"] = "user-specified"
    analysis["candidate_trace"] = [
        {"code": "SARC", "support_geomean": 0.82},
        {"code": "COAD", "support_geomean": 0.35},
    ]
    analysis["fit_quality"] = {"label": "good"}
    ranges_df = _make_ranges_df()

    md = build_summary(
        analysis,
        ranges_df,
        cancer_code="COAD",
        disease_state="",
    )

    assert "fallback broad expression reference is discordant with supplied COAD" in md
    assert (
        "top broad-reference match is SARC (Sarcoma) while "
        "COAD (Colon Adenocarcinoma) is rank 2"
    ) in md
    assert "Keep the supplied label as the report label" in md


def test_summary_marks_supplied_cancer_type_rna_ambiguity():
    analysis = _make_analysis()
    analysis["cancer_type"] = "COAD"
    analysis["cancer_name"] = "Colon Adenocarcinoma"
    analysis["analysis_constraints"] = {"cancer_type": "COAD"}
    analysis["cancer_type_source"] = "user-specified"
    analysis["candidate_trace"] = [
        {"code": "SARC", "support_geomean": 0.48},
        {"code": "COAD", "support_geomean": 0.44},
    ]
    analysis["fit_quality"] = {"label": "ambiguous"}
    ranges_df = _make_ranges_df()

    md = build_summary(
        analysis,
        ranges_df,
        cancer_code="COAD",
        disease_state="",
    )

    assert "fallback broad expression reference is ambiguous against supplied COAD" in md
    assert (
        "top broad-reference match is SARC (Sarcoma) while "
        "COAD (Colon Adenocarcinoma) is rank 2"
    ) in md


def test_summary_treats_broad_sarc_as_compatible_with_supplied_osteosarcoma():
    analysis = _make_analysis()
    analysis["cancer_type"] = "SARC_OS"
    analysis["cancer_name"] = "Osteosarcoma"
    analysis["analysis_constraints"] = {"cancer_type": "SARC_OS"}
    analysis["cancer_type_source"] = "user-specified"
    analysis["report_scope_cancer_type"] = "SARC_OS"
    analysis["reference_cancer_type"] = "SARC"
    analysis["candidate_trace"] = [
        {"code": "SARC", "support_geomean": 0.58},
        {"code": "UCS", "support_geomean": 0.39},
        {"code": "BRCA", "support_geomean": 0.36},
    ]
    analysis["signature_top_cancers"] = [("KIRC", 0.70)]
    ranges_df = _make_ranges_df()

    md = build_summary(
        analysis,
        ranges_df,
        cancer_code="SARC_OS",
        disease_state="",
    )

    # SARC_OS is now a registry child of SARC (5.13), so the brief recognizes
    # the direct parent-child relationship ("concordant at the parent level")
    # rather than the old same-family broad-context heuristic.
    assert "externally supplied SARC_OS (Osteosarcoma) sets the fine/report label" in md
    assert "SARC (Sarcoma) is used as the parent expression context" in md
    assert "concordant at the parent level: SARC (Sarcoma) is top" in md
    assert "refined report label remains SARC_OS (Osteosarcoma)" in md
    assert "raw signature favors KIRC" not in md
    assert "confidence caveats" not in md


def test_summary_marks_rna_inferred_cancer_type_as_hypothesis():
    analysis = _make_analysis()
    analysis["analysis_constraints"] = {}
    analysis["cancer_type_source"] = "auto-detected"
    ranges_df = _make_ranges_df()

    md = build_summary(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )

    assert "Cancer-type basis" in md
    assert "RNA-inferred hypothesis" in md
    assert "Cancer type is RNA-inferred — treat it as a hypothesis" in md
    assert "Clinical interpretation still needs external patient context" in md


def test_summary_lists_rna_alternatives_for_inferred_non_rare_call():
    analysis = _make_analysis()
    analysis["analysis_constraints"] = {}
    analysis["cancer_type_source"] = "auto-detected"
    # raw-signature top is now derived from the SAME candidate_trace shown in the
    # table (highest signature_score), not a separate signature_top_cancers field —
    # so the report can't say "raw-signature top X" for an X absent from / ranked
    # differently in the candidate table.
    analysis["candidate_trace"] = [
        {"code": "PRAD", "support_geomean": 0.50, "signature_score": 0.40},
        {"code": "BLCA", "support_geomean": 0.40, "signature_score": 0.91},
        {"code": "COAD", "support_geomean": 0.25, "signature_score": 0.20},
    ]
    ranges_df = _make_ranges_df()

    md = build_summary(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )

    assert "**Retained RNA differential:** ordered RNA candidates PRAD (rank 1)" in md
    assert "BLCA (rank 2, 0.80x top support)" in md
    assert "COAD (rank 3, 0.50x top support)" in md
    assert "raw-signature top BLCA" in md


def test_summary_mmr_release_vote_overrides_conflicting_mss_subtype_text():
    analysis = _make_analysis()
    analysis.update(
        {
            "cancer_type": "READ",
            "cancer_name": "Rectum Adenocarcinoma",
            "analysis_constraints": {},
            "cancer_type_source": "auto-detected",
            "candidate_trace": [
                {
                    "code": "READ",
                    "support_geomean": 0.50,
                    "support_fraction_of_top": 0.93,
                    "signature_score": 0.54,
                    "winning_subtype": "READ_MSS",
                }
            ],
            "cancer_type_evidence": {
                "staged_evidence_graph": {
                    "channels": [
                        {
                            "candidate_code": "READ",
                            "role": "hierarchical_mismatch_repair_vote",
                            "code": "MSI",
                            "status": "admission_context",
                            "details": {
                                "label_space": (
                                    "learned_mismatch_repair_release_ensemble"
                                ),
                                "mismatch_repair": {
                                    "context_group": "CRC",
                                    "decision_threshold": 0.5,
                                    "msi_probability": 0.808925,
                                },
                            },
                        }
                    ]
                }
            },
        }
    )

    md = build_summary(
        analysis,
        _make_ranges_df(),
        cancer_code="READ",
        disease_state="",
    )

    assert "**Mismatch-repair RNA context:** CRC MMR ensemble favors MSI-like" in md
    assert "MSI-like probability 0.81" in md
    assert "conflicts with the candidate-trace subtype READ_MSS" in md
    assert "MSS Rectum Adenocarcinoma-consistent" not in md
    assert "RNA subtype signal is" not in md


def test_summary_mmr_vote_ignores_unrelated_retained_candidate():
    analysis = _make_analysis()
    analysis.update(
        {
            "cancer_type": "GBM",
            "cancer_name": "Glioblastoma",
            "analysis_constraints": {},
            "cancer_type_source": "auto-detected",
            "candidate_trace": [{"code": "GBM", "support_fraction_of_top": 1.0}],
            "cancer_type_evidence": {
                "staged_evidence_graph": {
                    "channels": [
                        {
                            "candidate_code": "COAD",
                            "role": "hierarchical_mismatch_repair_vote",
                            "code": "MSI",
                            "status": "admission_context",
                            "details": {
                                "label_space": (
                                    "learned_mismatch_repair_release_ensemble"
                                ),
                                "mismatch_repair": {
                                    "context_group": "CRC",
                                    "decision_threshold": 0.5,
                                    "msi_probability": 0.92,
                                },
                            },
                        }
                    ]
                }
            },
        }
    )

    md = build_summary(
        analysis,
        _make_ranges_df(),
        cancer_code="GBM",
        disease_state="",
    )

    assert "Mismatch-repair RNA context" not in md
    assert "CRC MMR ensemble" not in md


def test_summary_mmr_vote_can_use_explicit_crc_context_for_read_call():
    analysis = _make_analysis()
    analysis.update(
        {
            "cancer_type": "READ",
            "cancer_name": "Rectum Adenocarcinoma",
            "analysis_constraints": {},
            "cancer_type_source": "auto-detected",
            "candidate_trace": [{"code": "READ", "support_fraction_of_top": 1.0}],
            "cancer_type_evidence": {
                "staged_evidence_graph": {
                    "channels": [
                        {
                            "candidate_code": "COAD",
                            "role": "hierarchical_mismatch_repair_vote",
                            "code": "MSI",
                            "status": "admission_context",
                            "details": {
                                "label_space": (
                                    "learned_mismatch_repair_release_ensemble"
                                ),
                                "mismatch_repair": {
                                    "context_group": "CRC",
                                    "decision_threshold": 0.5,
                                    "msi_probability": 0.71,
                                },
                            },
                        }
                    ]
                }
            },
        }
    )

    md = build_summary(
        analysis,
        _make_ranges_df(),
        cancer_code="READ",
        disease_state="",
    )

    assert "**Mismatch-repair RNA context:** CRC MMR ensemble favors MSI-like" in md
    assert "MSI-like probability 0.71" in md


def _mmr_analysis(msi_probability, *, mlh1_expression=None, code="COAD"):
    mmr = {
        "context_group": "CRC",
        "decision_threshold": 0.5,
        "msi_probability": msi_probability,
    }
    if mlh1_expression is not None:
        mmr["mlh1_expression"] = mlh1_expression
    return {
        "cancer_type": code,
        "cancer_type_evidence": {
            "staged_evidence_graph": {
                "channels": [
                    {
                        "candidate_code": code,
                        "role": "hierarchical_mismatch_repair_vote",
                        "code": "MSI",
                        "status": "admission_context",
                        "details": {
                            "label_space": (
                                "learned_mismatch_repair_release_ensemble"
                            ),
                            "mismatch_repair": mmr,
                        },
                    }
                ]
            }
        },
    }


def test_mmr_summary_flags_mlh1_retained_msi_tension():
    line = mismatch_repair_summary_line(
        _mmr_analysis(0.81, mlh1_expression={"tpm": 18.0, "cohort_ratio": 1.01})
    )
    assert "favors MSI-like" in line
    assert "MLH1 mRNA is retained (18 TPM, 101% of the cohort-typical level)" in line
    assert "does not exclude MSI" in line


def test_mmr_summary_omits_tension_when_mlh1_silenced():
    line = mismatch_repair_summary_line(
        _mmr_analysis(0.81, mlh1_expression={"tpm": 4.0, "cohort_ratio": 0.24})
    )
    assert "favors MSI-like" in line
    assert "MLH1 mRNA is retained" not in line


def test_mmr_summary_omits_tension_when_no_cohort_ratio():
    # Classifier surfaced the raw TPM but no reference cohort was in scope, so retention
    # is unknown — the clause must not fire off the sample TPM alone.
    line = mismatch_repair_summary_line(
        _mmr_analysis(0.81, mlh1_expression={"tpm": 18.0})
    )
    assert "favors MSI-like" in line
    assert "MLH1 mRNA is retained" not in line


def test_mmr_summary_omits_tension_when_mss():
    line = mismatch_repair_summary_line(
        _mmr_analysis(0.20, mlh1_expression={"tpm": 18.0, "cohort_ratio": 1.01})
    )
    assert "favors MSS-like" in line
    assert "MLH1 mRNA is retained" not in line


def test_mmr_summary_tension_survives_flat_to_nested_renesting(monkeypatch):
    # Cross the real seam: the release MMR classifier emits FLAT vote details;
    # cancer_type_evidence enriches them (adds cohort_ratio) while still flat; the
    # channel builder re-nests the flat details verbatim under "mismatch_repair"
    # (cancer_type_evidence.py ~983); brief reads the nested channel. Assert the
    # tension clause survives that round-trip.
    import trufflepig.cancer_type_evidence as cte

    monkeypatch.setattr(cte, "_cohort_bulk_gene_median", lambda code, gene: 18.0)
    vote = {
        "label_space": "learned_mismatch_repair_release_ensemble",
        "details": {
            "context_group": "CRC",
            "decision_threshold": 0.5,
            "msi_probability": 0.81,
            "mlh1_expression": {"tpm": 18.0},
        },
    }
    enriched = cte._enrich_mmr_vote_mlh1_cohort_context(vote, "COAD")
    analysis = {
        "cancer_type": "COAD",
        "cancer_type_evidence": {
            "staged_evidence_graph": {
                "channels": [
                    {
                        "candidate_code": "COAD",
                        "role": "hierarchical_mismatch_repair_vote",
                        "code": "MSI",
                        "status": "admission_context",
                        "details": {
                            "label_space": (
                                "learned_mismatch_repair_release_ensemble"
                            ),
                            # Exactly how the channel builder nests it (line ~983).
                            "mismatch_repair": enriched["details"],
                        },
                    }
                ]
            }
        },
    }
    line = mismatch_repair_summary_line(analysis)
    assert "MLH1 mRNA is retained (18 TPM, 100% of the cohort-typical level)" in line
    assert "does not exclude MSI" in line


def test_summary_rna_alternatives_use_post_gate_support_fraction():
    analysis = _make_analysis()
    analysis["analysis_constraints"] = {}
    analysis["cancer_type_source"] = "auto-detected"
    # Compartment gating can promote a biologically compatible row above a raw
    # higher-support row. The summary must explain the final/post-gate ranking,
    # not report a rank-2 candidate as >1x the top call.
    analysis["candidate_trace"] = [
        {
            "code": "SARC",
            "support_geomean": 0.42,
            "support_fraction_of_top": 1.0,
            "signature_score": 0.64,
        },
        {
            "code": "UCS",
            "support_geomean": 0.44,
            "support_fraction_of_top": 0.51,
            "signature_score": 0.54,
        },
    ]
    ranges_df = _make_ranges_df()

    md = build_summary(
        analysis,
        ranges_df,
        cancer_code="SARC",
        disease_state="",
    )

    assert "UCS (rank 2, 0.51x top support)" in md
    assert "1.03x top support" not in md


def test_summary_does_not_list_rna_alternatives_for_supplied_label():
    analysis = _make_analysis()
    analysis["analysis_constraints"] = {"cancer_type": "PRAD"}
    analysis["cancer_type_source"] = "user-specified"
    analysis["candidate_trace"] = [
        {"code": "PRAD", "support_geomean": 0.50},
        {"code": "BLCA", "support_geomean": 0.40},
    ]
    ranges_df = _make_ranges_df()

    md = build_summary(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )

    assert "expression-reference context is concordant with supplied PRAD" in md
    assert "**Retained RNA differential:**" not in md


def test_low_confidence_call_punctuation_is_clean():
    analysis = _make_analysis()
    analysis["candidate_trace"] = [
        {
            "code": "PRAD",
            "support_geomean": 0.4,
            "signature_score": 0.4,
        }
    ]
    analysis["fit_quality"] = {"label": "weak", "message": "flat signature"}
    ranges_df = _make_ranges_df()

    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )
    cancer_line = next(
        line for line in md.splitlines() if line.startswith("**Cancer call:**")
    )
    assert "). —" not in cancer_line

    actionable = build_actionable(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )
    working_line = next(
        line for line in actionable.splitlines() if line.startswith("Working call:")
    )
    assert "). —" not in working_line


def test_fusion_scoped_low_confidence_call_remains_explicitly_provisional():
    analysis = _make_analysis()
    analysis.update(
        {
            "cancer_type": "NUTM",
            "cancer_name": "NUT Carcinoma",
            "candidate_trace": [
                {
                    "code": "NUTM",
                    "support_geomean": 0.4,
                    "signature_score": 0.4,
                }
            ],
            "fit_quality": {
                "label": "weak",
                "message": "Top subtype candidates remain close",
            },
            "fusion_report_scope_inference": {
                "cancer_type": "NUTM",
                "expected_pair": "BRD4--NUTM1",
            },
        }
    )

    summary = build_brief(
        analysis,
        _make_ranges_df(),
        cancer_code="NUTM",
        disease_state="",
    )
    cancer_line = next(
        line for line in summary.splitlines() if line.startswith("**Cancer call:**")
    )

    assert "**low confidence, provisional**" in cancer_line


def test_brief_excludes_absent_targets():
    analysis = _make_analysis()
    ranges_df = _make_ranges_df()
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )
    # DLL3 is absent (0.5 TPM) — must not appear in the top bullets.
    assert "DLL3" not in md, "brief should skip absent targets from the top list"


def test_brief_reports_tumor_attributed_for_present_targets():
    analysis = _make_analysis()
    ranges_df = _make_ranges_df()
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )
    # FOLH1 has tumor-attr 128; the bullet should mention it.
    assert "FOLH1" in md
    assert "128" in md or "**FOLH1**" in md


def test_brief_renders_no_pattern_disease_state_when_scores_exist():
    analysis = _make_analysis()
    analysis["therapy_response_scores"] = {"IFN_response": object()}
    ranges_df = _make_ranges_df()
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )
    assert "No strong RNA-defined therapy-exposure" in md


def test_brief_summarizes_active_mapk_pathway_inference():
    analysis = _make_analysis()
    analysis["pathway_activity_inferences"] = [
        {
            "label": "MAPK / ERK activity",
            "up_geomean_fold": 8.2,
            "support_genes": ["DUSP6 15.0x", "SPRY2 12.0x"],
            "candidate_sources": [
                {"label": "supplied EGFR kinase domain duplication"}
            ],
            "caveat": "MAPK/ERK RNA activity is a convergent downstream readout.",
        }
    ]
    ranges_df = _make_ranges_df()
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )
    assert "**Active pathway:** MAPK / ERK activity high" in md
    assert "supplied EGFR kinase domain duplication" in md


def test_brief_prioritizes_ar_path_and_flags_possible_current_therapy():
    from trufflepig.therapy_response import TherapyAxisScore

    analysis = _make_analysis()
    analysis["therapy_response_scores"] = {
        "AR_signaling": TherapyAxisScore(
            therapy_class="AR_signaling",
            state="down",
            up_geomean_fold=0.39,
            down_geomean_fold=3.24,
        )
    }
    ranges_df = _make_ranges_df()
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="**AR axis suppressed** — consistent with ADT exposure.",
    )
    assert "- **FOLH1**" not in md
    ar_line = next(line for line in md.splitlines() if line.startswith("- **AR**"))
    assert "guideline-standard approved pathway" in ar_line
    assert "current/prior ADT or ARPI" in ar_line


def test_brief_does_not_promote_breast_therapies_without_clinical_biomarkers():
    analysis = _make_analysis()
    analysis["cancer_type"] = "BRCA"
    analysis["cancer_name"] = "Breast invasive carcinoma"
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "TACSTD2",
                "observed_tpm": 260.0,
                "attr_tumor_tpm": 240.0,
                "attr_tumor_fraction": 0.92,
                "attr_top_compartment": "tumor",
                "attr_top_compartment_tpm": 240.0,
                "tme_dominant": False,
                "tme_explainable": False,
            },
            {
                "symbol": "ERBB2",
                "observed_tpm": 45.0,
                "attr_tumor_tpm": 40.0,
                "attr_tumor_fraction": 0.89,
                "attr_top_compartment": "tumor",
                "attr_top_compartment_tpm": 40.0,
                "tme_dominant": False,
                "tme_explainable": False,
            },
        ]
    )
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="BRCA",
        disease_state="",
    )
    assert "- **ERBB2**" not in md
    assert "- **TACSTD2**" not in md
    assert "## Top candidate therapies" in md


def test_expression_independent_therapy_without_eligibility_stays_out_of_shortlist():
    analysis = _make_analysis()
    analysis["cancer_type"] = "COAD"
    analysis["cancer_name"] = "Colon adenocarcinoma"
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "CD274",
                "observed_tpm": 0.0,
                "attr_tumor_tpm": 0.0,
                "attr_tumor_fraction": 0.0,
                "attr_top_compartment": "",
                "attr_top_compartment_tpm": 0.0,
                "tme_dominant": False,
                "tme_explainable": False,
            },
        ]
    )
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="COAD",
        disease_state="",
    )
    assert not any(
        line.startswith("- **CD274**") for line in md.splitlines()
    )
    assert "## Top candidate therapies" in md


def test_expression_independent_therapy_surfaces_missing_required_evidence():
    analysis = _make_analysis()
    target = pd.Series(
        {
            "symbol": "NTRK1",
            "agent": "larotrectinib",
            "agent_class": "small_molecule",
            "phase": "approved",
            "indication": "NTRK fusion-positive solid tumor",
            "rationale": "requires an NTRK fusion or rearrangement",
        }
    )
    expression = pd.Series(
        {
            "symbol": "NTRK1",
            "observed_tpm": 12.0,
            "attr_tumor_tpm": 10.0,
            "attr_tumor_fraction": 0.83,
            "attr_top_compartment": "",
            "attr_top_compartment_tpm": 0.0,
            "tme_dominant": False,
            "tme_explainable": False,
        }
    )

    line = _format_therapy_bullet(target, expression, analysis=analysis)

    assert "target expression is not the eligibility criterion" in line
    assert "target RNA is context only" in line
    assert "required eligibility evidence not supplied" in line
    assert "confirm mutation / fusion / amplification before treating as eligible" in line


def test_agent_only_sarcoma_therapies_are_shortlisted_without_nan_symbol():
    analysis = _make_analysis()
    analysis["cancer_type"] = "SARC"
    analysis["cancer_name"] = "Sarcoma"
    targets_df = pd.DataFrame(
        [
            {
                "cancer_code": "SARC",
                "subtype": "leiomyosarcoma",
                "symbol": float("nan"),
                "agent": "trabectedin",
                "agent_class": "small_molecule",
                "phase": "approved",
                "indication": "advanced LMS",
            },
            {
                "cancer_code": "SARC",
                "subtype": "leiomyosarcoma",
                "symbol": float("nan"),
                "agent": "doxorubicin",
                "agent_class": "small_molecule",
                "phase": "approved",
                "indication": "first-line STS",
            },
            {
                "cancer_code": "SARC",
                "subtype": "leiomyosarcoma",
                "symbol": float("nan"),
                "agent": "pazopanib",
                "agent_class": "small_molecule",
                "phase": "approved",
                "indication": "advanced non-adipocytic STS",
            },
        ]
    )
    ranges_df = pd.DataFrame(columns=["symbol", "observed_tpm"])

    top = _top_therapies(targets_df, ranges_df, analysis=analysis)

    assert [row["agent"] for row, _expr in top] == ["doxorubicin", "pazopanib"]
    line = _format_therapy_bullet(top[0][0], top[0][1], analysis=analysis)
    assert line.startswith("- **doxorubicin** — agent-only therapy")
    assert "target expression is not the eligibility criterion" in line
    assert "nan" not in line.lower()


def test_therapy_bullet_uses_agent_class_when_agent_is_missing():
    target = pd.Series(
        {
            "symbol": "PGR",
            "agent": float("nan"),
            "agent_class": "hormone",
            "phase": "approved",
            "indication": "ER+/HER2- BRCA",
        }
    )
    expression = pd.Series(
        {
            "symbol": "PGR",
            "observed_tpm": 30.0,
            "attr_tumor_tpm": 25.0,
            "attr_tumor_tpm_low": 10.0,
            "attr_tumor_tpm_high": 30.0,
            "attr_tumor_fraction": 0.83,
            "attr_tumor_fraction_high": 0.90,
            "attr_top_compartment": "",
            "attr_top_compartment_tpm": 0.0,
            "tme_dominant": False,
            "tme_explainable": False,
        }
    )

    line = _format_therapy_bullet(target, expression)

    assert "**PGR** — hormone therapy (Approved, ER+/HER2- BRCA)" in line
    assert "nan" not in line.lower()


def test_expression_independent_therapy_distinguishes_generic_fusion_file_from_supporting_call():
    analysis = _make_analysis()
    analysis["fusion_inputs_supplied"] = True
    target = pd.Series(
        {
            "symbol": "NTRK1",
            "agent": "larotrectinib",
            "agent_class": "small_molecule",
            "phase": "approved",
            "indication": "NTRK fusion-positive solid tumor",
        }
    )
    line = _format_therapy_bullet(target, None, analysis=analysis)

    assert "orthogonal mutation/fusion/CNV evidence was supplied" in line
    assert "no target-specific supporting call was recognized" in line


def test_mutation_only_rows_do_not_treat_supplied_fusions_as_exact_evidence():
    analysis = _make_analysis()
    analysis["fusion_inputs_supplied"] = True
    target = pd.Series(
        {
            "symbol": "ESR1",
            "agent": "elacestrant",
            "agent_class": "small_molecule",
            "phase": "approved",
            "indication": "ER+/HER2- ESR1-mut BRCA",
            "rationale": "requires ESR1 mutation",
        }
    )

    line = _format_therapy_bullet(target, None, analysis=analysis)

    assert "orthogonal mutation/fusion/CNV evidence was supplied" in line
    assert "no target-specific supporting call was recognized" in line
    assert "required eligibility evidence was supplied" not in line


def test_nutm_scope_level_rows_reference_report_scope_not_target_specific_mutation():
    analysis = _make_analysis()
    analysis.update(
        {
            "cancer_type": "NUTM",
            "fusion_report_scope_inference": {
                "cancer_type": "NUTM",
                "expected_pair": "BRD3--NUTM1",
            },
        }
    )
    target = pd.Series(
        {
            "cancer_code": "NUTM",
            "symbol": "BRD4",
            "agent": "molibresib",
            "agent_class": "small_molecule",
            "phase": "phase_1",
            "indication": "NUT carcinoma",
        }
    )

    line = _format_therapy_bullet(target, None, analysis=analysis)

    assert "scope-level fusion evidence supports the NUTM report label" in line
    assert "no target-specific supporting call" not in line


def test_nutm_therapy_does_not_treat_an_unrelated_brd4_fusion_as_eligibility():
    analysis = _make_analysis()
    analysis.update(
        {
            "cancer_type": "NUTM",
            "variant_inputs_supplied": True,
            "variant_records": [
                {
                    "gene": "BRD4",
                    "genes": ("BRD4", "LINC00486"),
                    "variant": "BRD4--LINC00486 fusion",
                    "variant_type": "fusion",
                }
            ],
        }
    )
    target = pd.Series(
        {
            "cancer_code": "NUTM",
            "symbol": "BRD4",
            "agent": "molibresib",
            "agent_class": "small_molecule",
            "phase": "phase_1",
            "indication": "NUT carcinoma",
        }
    )

    line = _format_therapy_bullet(target, None, analysis=analysis)

    assert "NUTM report label is RNA-inferred" in line
    assert "confirm NUTM1 fusion/IHC/FISH/pathology" in line
    assert "supplied variant evidence matches" not in line


def test_sarc_summary_uses_supplied_egfr_kdd_and_skips_unresolved_subtype_spillover():
    analysis = _make_analysis()
    analysis.update(
        {
            "cancer_type": "SARC",
            "cancer_name": "Sarcoma",
            "cancer_type_source": "user-specified",
            "analysis_constraints": {"cancer_type": "SARC"},
            "variant_inputs_supplied": True,
            "variant_records": [
                {
                    "gene": "EGFR",
                    "variant": "EGFR kinase domain duplication / KDD",
                    "variant_type": "kdd",
                }
            ],
        }
    )
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "EGFR",
                "observed_tpm": 583.0,
                "attr_tumor_tpm": 570.0,
                "attr_tumor_fraction": 0.98,
                "attr_top_compartment": "",
                "attr_top_compartment_tpm": 0.0,
                "tme_dominant": False,
                "tme_explainable": False,
            },
            {
                "symbol": "PDGFRA",
                "observed_tpm": 200.0,
                "attr_tumor_tpm": 180.0,
                "attr_tumor_fraction": 0.90,
                "attr_top_compartment": "",
                "attr_top_compartment_tpm": 0.0,
                "tme_dominant": False,
                "tme_explainable": False,
            },
            {
                "symbol": "NTRK1",
                "observed_tpm": 120.0,
                "attr_tumor_tpm": 110.0,
                "attr_tumor_fraction": 0.92,
                "attr_top_compartment": "",
                "attr_top_compartment_tpm": 0.0,
                "tme_dominant": False,
                "tme_explainable": False,
            },
        ]
    )

    md = build_summary(analysis, ranges_df, cancer_code="SARC", disease_state="")

    assert "**Variant evidence:** supplied EGFR kinase domain duplication" in md
    top_lines = [line for line in md.splitlines() if line.startswith("- **")]
    assert top_lines[0].startswith("- **EGFR**")
    assert "supplied variant evidence matches this therapy requirement" in md
    assert "- **PDGFRA**" not in md
    assert "- **NTRK1**" not in md
    assert "- **CDK4**" not in md
    assert "- **PRAME**" not in md
    assert "HLA typing needed" not in md


def test_summary_does_not_present_a_negative_variant_as_eligibility_evidence():
    analysis = _make_analysis()
    analysis.update(
        {
            "variant_inputs_supplied": True,
            "variant_records": [
                {
                    "gene": "EGFR",
                    "variant": "EGFR KDD not detected",
                    "variant_type": "kdd",
                }
            ],
        }
    )

    md = build_summary(analysis, pd.DataFrame(), cancer_code="SARC", disease_state="")

    assert "**Variant evidence:** supplied EGFR" not in md
    assert "no usable positive call was available" in md


def test_summary_prompts_for_hla_when_hla_gated_target_is_plausible():
    analysis = _make_analysis()
    analysis["cancer_type"] = "UVM"
    analysis["cancer_name"] = "Uveal Melanoma"
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "PMEL",
                "observed_tpm": 80.0,
                "attr_tumor_tpm": 70.0,
                "attr_tumor_fraction": 0.88,
                "attr_top_compartment": "",
                "attr_top_compartment_tpm": 0.0,
                "tme_dominant": False,
                "tme_explainable": False,
            },
        ]
    )

    md = build_summary(
        analysis,
        ranges_df,
        cancer_code="UVM",
        disease_state="",
    )

    assert "HLA typing needed for tebentafusp" in md
    assert "requires A*02:01" in md


def test_brief_downranks_er_dependent_brca_therapy_when_er_axis_low():
    from trufflepig.therapy_response import TherapyAxisScore

    analysis = _make_analysis()
    analysis["cancer_type"] = "BRCA"
    analysis["cancer_name"] = "Breast invasive carcinoma"
    analysis["therapy_response_scores"] = {
        "ER_signaling": TherapyAxisScore(
            therapy_class="ER_signaling",
            state="down",
            up_geomean_fold=0.31,
            down_geomean_fold=2.4,
        )
    }
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "ESR1",
                "observed_tpm": 80.0,
                "attr_tumor_tpm": 70.0,
                "attr_tumor_fraction": 0.88,
                "attr_top_compartment": "tumor",
                "attr_top_compartment_tpm": 70.0,
                "tme_dominant": False,
                "tme_explainable": False,
            },
            {
                "symbol": "TACSTD2",
                "observed_tpm": 100.0,
                "attr_tumor_tpm": 90.0,
                "attr_tumor_fraction": 0.90,
                "attr_top_compartment": "tumor",
                "attr_top_compartment_tpm": 90.0,
                "tme_dominant": False,
                "tme_explainable": False,
            },
        ]
    )
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="BRCA",
        disease_state="**ER-axis suppressed / ER-low pattern**.",
    )
    top_lines = [line for line in md.splitlines() if line.startswith("- **")]
    assert all(not line.startswith("- **TACSTD2**") for line in top_lines)
    assert all(not line.startswith("- **ESR1**") for line in top_lines)

    actionable = build_actionable(
        analysis,
        ranges_df,
        cancer_code="BRCA",
        disease_state="**ER-axis suppressed / ER-low pattern**.",
    )
    assert "RNA-context conflict: ER axis is suppressed/ER-low" in actionable
    assert "ER-low biology or current/prior endocrine therapy signal" in actionable


def test_disease_state_ifn_split_preserves_sentence_punctuation():
    from trufflepig.brief import _disease_state_summary_lines

    lines = _disease_state_summary_lines(
        "**ER-axis suppressed / ER-low pattern** (ESR1 low; classic ER targets "
        "collapsed). This can reflect ER-negative/basal-like biology or "
        "endocrine resistance/exposure depending on clinical context. "
        "**Active IFN response** — MHC-I / ISG surface fold-changes carry "
        "IFN-driven inflation."
    )

    assert lines[0].endswith("context.")
    assert lines[1].startswith("**Immune/IFN state:** Active IFN response")


def test_brief_explains_bulk_present_targets_that_fail_source_gate():
    analysis = _make_analysis()
    ranges_df = pd.concat(
        [
            _make_ranges_df(),
            pd.DataFrame(
                [
                    {
                        "symbol": "STEAP2",
                        "observed_tpm": 90.0,
                        "attribution": {"matched_normal_prostate": 78.0},
                        "attr_tumor_tpm": 13.0,
                        "attr_tumor_fraction": 0.14,
                        "attr_top_compartment": "matched_normal_prostate",
                        "attr_top_compartment_tpm": 78.0,
                        "tme_dominant": True,
                        "tme_explainable": True,
                    },
                    {
                            "symbol": "PSCA",
                        "observed_tpm": 247.0,
                        "attribution": {"matched_normal_prostate": 155.0},
                        "attr_tumor_tpm": 57.0,
                        "attr_tumor_fraction": 0.23,
                        "attr_top_compartment": "matched_normal_prostate",
                        "attr_top_compartment_tpm": 155.0,
                        "tme_dominant": False,
                        "tme_explainable": True,
                        "matched_normal_over_predicted": True,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )
    assert "Where target RNA signal appears to come from" in md
    assert "Estimated tumor TPM (RNA model)" in md
    assert "Top estimated background contribution" in md
    assert "STEAP2" in md
    assert "PSCA" in md
    assert "prostate lineage reference (external panel)" in md
    assert "phase 1 exploratory" in md
    assert len(md.splitlines()) <= 40


def test_source_trace_renders_when_top_trial_rows_are_mixed_source():
    target = pd.Series(
        {
            "symbol": "TARGET1",
            "phase": "phase_2",
            "agent": "trial drug",
            "agent_class": "antibody",
            "treatment_path_tier": "trial_follow_up",
            "eligibility_note": "not default standard",
        }
    )
    expr = pd.Series(
        {
            "symbol": "TARGET1",
            "observed_tpm": 20.0,
            "attr_tumor_tpm": 8.0,
            "attr_tumor_fraction": 0.40,
            "attr_top_compartment": "",
            "attr_top_compartment_tpm": 0.0,
            "tme_dominant": False,
            "tme_explainable": False,
        }
    )
    md = _shortlist_omission_note(
        pd.DataFrame([target]),
        pd.DataFrame([expr]),
        [(target, expr)],
    )
    assert "Where target RNA signal appears to come from" in md
    assert "none estimated" in md


def test_source_trace_starts_with_approved_shortlist_rows():
    approved = pd.Series(
        {
            "symbol": "EGFR",
            "phase": "approved",
            "agent": "cetuximab",
            "agent_class": "antibody",
            "treatment_path_tier": "approved_biomarker",
        }
    )
    approved_expr = pd.Series(
        {
            "symbol": "EGFR",
            "observed_tpm": 64.0,
            "attr_tumor_tpm": 39.0,
            "attr_tumor_fraction": 0.61,
            "attr_top_compartment": "mesothelial",
            "attr_top_compartment_tpm": 9.0,
            "tme_dominant": False,
            "tme_explainable": False,
        }
    )
    omitted = pd.Series(
        {
            "symbol": "ERBB2",
            "phase": "approved",
            "agent": "trastuzumab + tucatinib",
            "agent_class": "antibody",
        }
    )
    omitted_expr = pd.Series(
        {
            "symbol": "ERBB2",
            "observed_tpm": 56.0,
            "attr_tumor_tpm": 0.0,
            "attr_tumor_fraction": 0.0,
            "attr_top_compartment": "mesothelial",
            "attr_top_compartment_tpm": 12.0,
            "tme_dominant": True,
            "tme_explainable": True,
        }
    )

    md = _shortlist_omission_note(
        pd.DataFrame([approved, omitted]),
        pd.DataFrame([approved_expr, omitted_expr]),
        [(approved, approved_expr)],
    )

    data_lines = [
        line
        for line in md.splitlines()
        if line.startswith("| ") and not line.startswith("| Gene") and not line.startswith("|---")
    ]
    assert data_lines[0].startswith("| EGFR ")
    assert any(line.startswith("| ERBB2 ") for line in data_lines[1:])


def test_source_trace_does_not_call_non_lineage_component_lineage_background():
    top = pd.Series(
        {
            "symbol": "TOP",
            "phase": "phase_2",
            "agent": "trial drug",
            "agent_class": "antibody",
        }
    )
    top_expr = pd.Series(
        {
            "symbol": "TOP",
            "observed_tpm": 20.0,
            "attr_tumor_tpm": 12.0,
            "attr_tumor_fraction": 0.60,
            "attr_top_compartment": "osteoblast",
            "attr_top_compartment_tpm": 1.0,
            "tme_dominant": False,
            "tme_explainable": False,
        }
    )
    omitted = pd.Series(
        {
            "symbol": "ERBB2",
            "phase": "phase_2",
            "agent": "trial drug",
            "agent_class": "ADC",
        }
    )
    omitted_expr = pd.Series(
        {
            "symbol": "ERBB2",
            "observed_tpm": 25.0,
            "attr_tumor_tpm": 0.0,
            "attr_tumor_fraction": 0.0,
            "attr_top_compartment": "osteoblast",
            "attr_top_compartment_tpm": 8.0,
            "matched_normal_over_predicted": True,
            "tme_dominant": True,
            "tme_explainable": True,
        }
    )
    md = _shortlist_omission_note(
        pd.DataFrame([top, omitted]),
        pd.DataFrame([top_expr, omitted_expr]),
        [(top, top_expr)],
    )
    erbb2_line = next(line for line in md.splitlines() if line.startswith("| ERBB2 "))
    assert "osteoblast reference component over-predicts / non-tumor background" in erbb2_line
    assert "osteoblast over-predicts / lineage background" not in erbb2_line


def test_brief_no_internal_jargon():
    analysis = _make_analysis(purity_tier_label="low")
    ranges_df = _make_ranges_df()
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )
    # Forbidden jargon — internal variable names and pipeline terms.
    for token in [
        "NNLS",
        "Spearman",
        "x1.10",
        "×1.10",
        "tme_tpm_med",
        "_combine_purity_estimates",
        "overexplained_tpm",
        "sig_stability",
    ]:
        assert token not in md, f"jargon leak: {token}"


def test_brief_handles_uncurated_cancer_type():
    """The brief must gracefully handle any cancer code that isn't in
    the curated key-genes panel — not just TCGA codes. Uses a fake
    placeholder code so the test is independent of which TCGA codes
    we've curated (all 33 are curated as of #155; pick a non-existent
    code so the test remains valid as we expand)."""
    analysis = _make_analysis()
    analysis["cancer_type"] = "ZZUNCURATED"
    ranges_df = _make_ranges_df()
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="ZZUNCURATED",
        disease_state="",
    )
    assert "not yet in the curated key-genes panel" in md


def test_actionable_is_longer_but_structured():
    analysis = _make_analysis()
    ranges_df = _make_ranges_df()
    md = build_actionable(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="Castrate-resistant.",
        sample_id="sample_X",
    )
    # Actionable should be > 15 lines (more detail than the brief).
    assert len(md.splitlines()) > 15

    # Actionable remains available as an internal builder, but its
    # cross-links should now point to the consolidated evidence.md
    # appendix rather than a standalone targets.md file.
    for heading in [
        "Sample and confidence",
        "Cancer call and disease state",
        "Therapy Prioritization",
    ]:
        assert heading in md, f"missing heading: {heading}"
    assert "model interval" in md
    assert "(CI " not in md
    assert "*-evidence.md*" in md or "`*-evidence.md`" in md, (
        "actionable should link to evidence.md as the target-table source"
    )


def test_actionable_surfaces_offcontext_expressed_target():
    # CLDN18 is a curated target for gastric/PAAD (zolbetuximab), NOT prostate.
    # Highly expressed in a PRAD sample it should surface as an off-context lead
    # rather than being dropped because it's off this cancer's panel (#47).
    analysis = _make_analysis()
    ranges_df = _make_ranges_df()
    ranges_df = pd.concat(
        [
            ranges_df,
            pd.DataFrame([{
                "symbol": "CLDN18",
                "observed_tpm": 120.0,
                "attribution": {},
                "attr_tumor_tpm": 110.0,
                "attr_tumor_fraction": 0.92,
                "attr_top_compartment": "",
                "attr_top_compartment_tpm": 0.0,
                "tme_dominant": False,
                "tme_explainable": False,
            }]),
        ],
        ignore_index=True,
    )
    md = build_actionable(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
        sample_id="sample_X",
    )
    assert "Off-Context Expressed Targets" in md
    assert "CLDN18" in md
    # Names the off-context binder + that it's off-label here.
    assert "zolbetuximab" in md
    assert "not on this cancer's curated panel" in md


def test_brief_normalizes_path_like_sample_id():
    analysis = _make_analysis()
    ranges_df = _make_ranges_df()
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
        sample_id="/tmp/run-123/rs",
    )
    assert md.splitlines()[0] == "# Summary: rs"


def test_brief_does_not_promote_psma_rna_without_required_imaging():
    analysis = _make_analysis()
    ranges_df = _make_ranges_df()
    idx = ranges_df.index[ranges_df["symbol"] == "FOLH1"][0]
    ranges_df.at[idx, "attribution"] = {}
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )
    assert "tumor-specific decomposition was unavailable" not in md
    assert "- **FOLH1**" not in md


def test_actionable_renders_tumor_band_without_attribution_dict():
    analysis = _make_analysis()
    ranges_df = _make_ranges_df()
    idx = ranges_df.index[ranges_df["symbol"] == "FOLH1"][0]
    ranges_df.at[idx, "attribution"] = {}
    md = build_actionable(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )
    assert (
        "| **FOLH1** | 177Lu-PSMA-617 | radioligand | Approved | mCRPC | 142.0 | 128 (128-128) |"
        in md
    )


def test_actionable_background_dominant_molecular_therapy_is_eligibility_dependent():
    # FGFR3 RNA can be hepatocyte/background-attributed without ruling erdafitinib
    # in or out: the clinical gate is a susceptible FGFR3 genetic alteration.
    analysis = _make_analysis()
    analysis["cancer_type"] = "BLCA"
    analysis["cancer_name"] = "Bladder Urothelial Carcinoma"
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "FGFR3",
                "gene_id": "ENSG00000068078",
                "observed_tpm": 10.6,
                "attribution": {},
                "attr_tumor_tpm": 0.0,
                "attr_tumor_tpm_low": 0.0,
                "attr_tumor_tpm_high": 0.0,
                "attr_tumor_fraction": 0.0,
                "attr_tumor_fraction_low": 0.0,
                "attr_tumor_fraction_high": 0.0,
                "attr_support_fraction": 0.0,
                "attr_top_compartment": "hepatocyte",
                "attr_top_compartment_tpm": 1.2,
                "tme_dominant": True,
                "tme_explainable": True,
                "matched_normal_over_predicted": False,
            }
        ]
    )
    md = build_actionable(
        analysis,
        ranges_df,
        cancer_code="BLCA",
        disease_state="",
        sample_id="pfo017-liver",
    )
    assert "## Therapy Prioritization" in md
    therapy_section = md.split("## Therapy Prioritization", 1)[1]
    row = next(
        line for line in therapy_section.splitlines()
        if "| **FGFR3** | erdafitinib |" in line
    )
    assert "target expression is not the eligibility criterion" in row
    assert "required eligibility evidence not supplied" in row
    assert "not sample-supported; negative/background evidence" not in row


def test_actionable_canonicalizes_curated_antigen_symbols(monkeypatch):
    import trufflepig.brief as brief_mod

    analysis = _make_analysis()
    analysis["cancer_type"] = "SARC"
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "MAGEA4",
                "observed_tpm": 19.0,
                "attribution": {},
                "attr_tumor_tpm": 8.0,
                "attr_tumor_fraction": 0.42,
                "attr_top_compartment": "",
                "attr_top_compartment_tpm": 0.0,
                "tme_dominant": False,
                "tme_explainable": False,
            }
        ]
    )
    targets_df = pd.DataFrame(
        [
            {
                "symbol": "MAGE-A4",
                "agent": "afami-cel",
                "agent_class": "TCR-T",
                "phase": "approved",
                "indication": "MAGE-A4+ HLA-A*02+ synovial sarcoma",
            }
        ]
    )
    monkeypatch.setattr(
        brief_mod,
        "_curated_target_panel_for_sample",
        lambda *a, **k: ("SARC", None, targets_df),
    )

    md = build_actionable(
        analysis,
        ranges_df,
        cancer_code="SARC",
        disease_state="",
        sample_id="sample_X",
    )

    assert "| **MAGEA4** | afami-cel |" in md
    assert "| **MAGE-A4** |" not in md
    assert "gene symbol not present in input file" not in md


# --- Biomarker-outlier belief-consistency (plan §2.4) ---------------------------
#
# A biomarker whose clinical eligibility is gated by a DNA alteration (mutation /
# specific allele / wild-type status) must not be presented as an actionable
# EXPRESSION outlier — that contradicts the report's own "expression is not the
# eligibility criterion". The gene's mutation-vs-expression basis is derived from
# the pirlygenes-authored biomarker rationale, not a hard-coded gene list.


def test_biomarker_basis_flags_mutation_gated_rationales():
    # Mutation / allele / wild-type gated → expression is NOT the biomarker.
    for rationale in (
        "Prognostic / predictive; mutation common and linked to chemo response",
        "G12C targetable; other alleles remain a negative predictor for EGFR TKIs",
        "Must be wild-type for EGFR-antibody response; pan-RAS testing standard",
        "V600E mutation — dabrafenib+trametinib eligible",
        "Activating mutations (exon 19 del / L858R) gate TKI indication",
        "R132 mutation — ivosidenib eligibility",
    ):
        assert biomarker_expression_is_not_eligibility(rationale) is True, rationale


def test_biomarker_basis_does_not_flag_expression_readable_rationales():
    # Amplification / overexpression / IHC / FISH cue wins → expression IS readable,
    # so HER2/MDM2-style biomarkers are never flagged (avoids the false positive of
    # telling a clinician "expression isn't the biomarker" where it partly is).
    for rationale in (
        "HER2 — gating biomarker for HER2-targeted agents; IHC/ISH 3+ or FISH-amplified eligible",
        "HER2 amplification — approved indication for trastuzumab combos in RAS WT",
        "Ring-chromosome 12q amplification with MDM2 — pathognomonic for WDLPS/DDLPS",
        "Overexpression / mutation — cetuximab eligibility",
        "",
    ):
        assert biomarker_expression_is_not_eligibility(rationale) is False, rationale


def test_her2_rna_proxy_feeds_assay_gated_therapy_reasoning_without_eligibility():
    target_row = {
        "symbol": "ERBB2",
        "agent": "trastuzumab deruxtecan",
        "indication": "HER2-positive breast cancer",
        "indication_biomarker": "clinical_target_assay",
    }
    analysis = {
        "rna_biomarker_proxies": {
            "her2": {
                "status": "supported",
                "clinical_claim": "context_only",
                "eligibility_established": False,
            }
        }
    }

    gap = _expression_independent_evidence_gap(target_row, analysis)

    assert "required eligibility evidence not supplied" in gap
    assert "IHC/ISH confirmation" in gap
    assert "does not establish eligibility" in gap


def test_summary_flags_mutation_gated_biomarker_outlier_via_public_api():
    # End-to-end through build_summary (the public surface that renders "Notable
    # biomarker outliers"): a mutation-gated biomarker surfaced as a high-mRNA
    # outlier must carry the "expression is not the eligibility criterion" caveat,
    # so the block can't contradict the report's own eligibility principle.
    # COAD TP53 is mutation-gated ("mutation common ... chemo response"); render a
    # strongly-amplified, top-percentile TP53 row so it qualifies as an outlier.
    analysis = _make_analysis()
    analysis["cancer_type"] = "COAD"
    analysis["cancer_name"] = "Colon adenocarcinoma"
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "TP53",
                "gene_id": "ENSG00000141510",
                "observed_tpm": 180.0,
                "amplification_fold": 15.0,
                "tcga_percentile": 0.99,
                "attribution": {},
                "attr_tumor_tpm": 170.0,
                "attr_tumor_fraction": 0.94,
                "attr_top_compartment": "",
                "attr_top_compartment_tpm": 0.0,
                "tme_dominant": False,
                "tme_explainable": False,
            }
        ]
    )
    md = build_summary(
        analysis,
        ranges_df,
        cancer_code="COAD",
        disease_state="",
        sample_id="sample_X",
    )
    assert "## Notable biomarker outliers" in md
    assert "TP53" in md
    assert "expression is not the eligibility criterion" in md
    assert len(md.splitlines()) <= 40


def test_summary_low_purity_caveat_rides_on_tumor_source_tpm():
    # Low-purity sample: the attribution caveat rides inline with target TPMs,
    # sourced from the sample_low_purity flag that analyze_sample sets on ranges_df.
    analysis = _make_analysis(purity_tier_label="low")
    ranges_df = _make_ranges_df()
    ranges_df["sample_low_purity"] = True
    md = build_summary(analysis, ranges_df, cancer_code="PRAD", disease_state="")
    assert (
        "low estimated tumor fraction — the tumor expression estimate is less certain"
        in md
    )


def test_summary_omits_low_purity_caveat_when_not_flagged():
    analysis = _make_analysis(purity_tier_label="high")
    ranges_df = _make_ranges_df()
    ranges_df["sample_low_purity"] = False
    md = build_summary(analysis, ranges_df, cancer_code="PRAD", disease_state="")
    assert "low estimated tumor fraction — the tumor expression estimate" not in md


def test_summary_omits_support_pct_when_preservation_unknown():
    """An 'unknown' preservation call carries no confidence (defaults to 0.0),
    so appending '(support 0%)' next to it reads as a spurious quantified
    claim. The summary must omit the support parenthetical (#85 minor)."""
    analysis = _make_analysis(preservation="unknown")
    analysis["sample_context"].preservation_confidence = 0.0
    md = build_summary(analysis, _make_ranges_df(), cancer_code="PRAD", disease_state="")
    assert "preservation inferred as unknown from RNA QC." in md
    assert "preservation inferred as unknown from RNA QC (" not in md


def test_summary_keeps_support_pct_for_known_preservation():
    """A known preservation call keeps its support parenthetical."""
    analysis = _make_analysis(preservation="ffpe")
    md = build_summary(analysis, _make_ranges_df(), cancer_code="PRAD", disease_state="")
    assert "from RNA QC (" in md


def test_rna_alternatives_flags_lineage_incoherent_runner_up():
    """A runner-up whose histogenesis is incoherent with its own gene pattern
    (near-zero lineage concordance) is annotated so it does not read as
    'second-strongest' without caveat (#85.5)."""
    from trufflepig.brief import _rna_alternatives_line

    analysis = {
        "candidate_trace": [
            {
                "code": "LUAD",
                "support_geomean": 0.80,
                "signature_score": 0.5,
                "lineage_concordance": 0.9,
            },
            {
                "code": "LAML",
                "support_geomean": 0.60,
                "signature_score": 0.1,
                "lineage_concordance": 0.0,
            },
            {
                "code": "COAD",
                "support_geomean": 0.40,
                "signature_score": 0.2,
                "lineage_concordance": 0.85,
            },
        ],
    }
    line = _rna_alternatives_line(analysis, "LUAD")
    assert "LAML (rank 2" in line
    assert "lineage-incoherent" in line
    # Only the incoherent runner-up is flagged, not the concordant one.
    assert line.count("lineage-incoherent") == 1


def test_rna_alternatives_no_caveat_when_concordance_missing():
    """Missing lineage_concordance is treated as coherent (no caveat) — the
    trace simply didn't score it; 0.0 is the real incoherence signal."""
    from trufflepig.brief import _rna_alternatives_line

    analysis = {
        "candidate_trace": [
            {"code": "LUAD", "support_geomean": 0.80},
            {"code": "LUSC", "support_geomean": 0.60},
        ],
    }
    line = _rna_alternatives_line(analysis, "LUAD")
    assert "LUSC (rank 2" in line
    assert "lineage-incoherent" not in line
