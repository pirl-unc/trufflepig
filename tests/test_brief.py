"""Tests for the two-tier brief / actionable handoff (#111)."""

import pandas as pd

from trufflepig.brief import (
    build_actionable,
    build_brief,
    build_summary,
    _format_therapy_bullet,
    _shortlist_omission_note,
)
from trufflepig.confidence import ConfidenceTier


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
    assert "**Purity:**" in md
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
    assert "Patient-facing LLM interpretation needs external clinical context" in md
    assert "RNA-inferred — treat it as a hypothesis" not in md


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

    assert "expression-reference context is discordant with supplied COAD" in md
    assert (
        "top expression-reference match is SARC (Sarcoma) while "
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

    assert "expression-reference context is ambiguous against supplied COAD" in md
    assert (
        "top expression-reference match is SARC (Sarcoma) while "
        "COAD (Colon Adenocarcinoma) is rank 2"
    ) in md


def test_summary_treats_broad_sarc_as_compatible_with_supplied_osteosarcoma():
    analysis = _make_analysis()
    analysis["cancer_type"] = "OS"
    analysis["cancer_name"] = "Osteosarcoma"
    analysis["analysis_constraints"] = {"cancer_type": "OS"}
    analysis["cancer_type_source"] = "user-specified"
    analysis["report_scope_cancer_type"] = "OS"
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
        cancer_code="OS",
        disease_state="",
    )

    assert "externally supplied OS (Osteosarcoma) sets the fine/report label" in md
    assert "expression-reference context is SARC (Sarcoma)" in md
    assert "sarcoma-family broad-context support for supplied OS (Osteosarcoma)" in md
    assert "does not independently resolve the refined label" in md
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
    assert "Patient-facing LLM interpretation needs external clinical context" in md


def test_summary_lists_rna_alternatives_for_inferred_non_rare_call():
    analysis = _make_analysis()
    analysis["analysis_constraints"] = {}
    analysis["cancer_type_source"] = "auto-detected"
    analysis["candidate_trace"] = [
        {"code": "PRAD", "support_geomean": 0.50},
        {"code": "BLCA", "support_geomean": 0.40},
        {"code": "COAD", "support_geomean": 0.25},
    ]
    analysis["signature_top_cancers"] = [("BLCA", 0.91)]
    ranges_df = _make_ranges_df()

    md = build_summary(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )

    assert "**RNA alternatives:** ordered RNA candidates PRAD (rank 1)" in md
    assert "BLCA (rank 2, 0.80x top support)" in md
    assert "COAD (rank 3, 0.50x top support)" in md
    assert "raw-signature top BLCA" in md


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
    assert "**RNA alternatives:**" not in md


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
    assert md.index("**AR**") < md.index("**FOLH1**")
    ar_line = next(line for line in md.splitlines() if line.startswith("- **AR**"))
    assert "guideline-standard approved pathway" in ar_line
    assert "current/prior ADT or ARPI" in ar_line


def test_brief_uses_path_maturity_across_cancer_types_not_prad_special_case():
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
    assert md.index("**ERBB2**") < md.index("**TACSTD2**")
    erbb2_line = next(
        line for line in md.splitlines() if line.startswith("- **ERBB2**")
    )
    assert "guideline-standard approved pathway" in erbb2_line
    tacstd2_line = next(
        line for line in md.splitlines() if line.startswith("- **TACSTD2**")
    )
    assert "approved later-line pathway" in tacstd2_line


def test_expression_independent_therapy_summary_keeps_rna_contextual():
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
    pdcd1_line = next(
        line for line in md.splitlines() if line.startswith("- **CD274**")
    )
    assert "target expression is not the eligibility criterion" in pdcd1_line
    assert "target RNA is context only" in pdcd1_line
    assert "Clinical maturity: approved antibody" in pdcd1_line
    assert "; Clinical maturity" not in pdcd1_line
    assert "model interval" not in pdcd1_line


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


def test_sarc_summary_uses_supplied_egfr_kdd_and_skips_unresolved_subtype_spillover():
    analysis = _make_analysis()
    analysis.update(
        {
            "cancer_type": "SARC",
            "cancer_name": "Sarcoma",
            "cancer_type_source": "user-specified",
            "analysis_constraints": {"cancer_type": "SARC"},
            "alteration_inputs_supplied": True,
            "alteration_records": [
                {
                    "gene": "EGFR",
                    "alteration": "EGFR kinase domain duplication / KDD",
                    "alteration_type": "kdd",
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

    assert "**Alteration evidence:** supplied EGFR kinase domain duplication" in md
    top_lines = [line for line in md.splitlines() if line.startswith("- **")]
    assert top_lines[0].startswith("- **EGFR**")
    assert "supplied alteration evidence matches this therapy requirement" in md
    assert "- **PDGFRA**" not in md
    assert "- **NTRK1**" not in md
    assert "- **CDK4**" not in md
    assert "- **PRAME**" not in md
    assert "HLA typing needed" not in md


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
    assert any(line.startswith("- **TACSTD2**") for line in top_lines)
    assert all(not line.startswith("- **ESR1**") for line in top_lines)

    actionable = build_actionable(
        analysis,
        ranges_df,
        cancer_code="BRCA",
        disease_state="**ER-axis suppressed / ER-low pattern**.",
    )
    assert "RNA-context conflict: ER axis is suppressed/ER-low" in actionable
    assert "ER-low biology or current/prior endocrine therapy signal" in actionable


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
                        "symbol": "KLK2",
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
    assert "Tumor-source bulk TPM" in md
    assert "Top non-tumor attribution" in md
    assert "STEAP2" in md
    assert "KLK2" in md
    assert "matched-normal prostate" in md
    assert "phase 1 exploratory" in md


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
    assert "none modeled" in md


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
    assert "osteoblast over-predicts / non-tumor background" in erbb2_line
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
        "tme_fold_med",
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


def test_brief_uses_tumor_band_without_attribution_dict():
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
    assert "128 tumor-source bulk TPM (model interval 128-128" in md


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
