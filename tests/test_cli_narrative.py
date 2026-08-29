# Licensed under the Apache License, Version 2.0

"""Unit tests for CLI narrative and report helper functions.

Covers: compose_disease_state_narrative (BRCA ER/HER2, EMT, IFN),
annotate_surface_targets_with_cross_signals, _summarize_sample_call,
_candidate_label_options.
"""

from types import SimpleNamespace

from trufflepig.main import (
    compose_disease_state_narrative,
    annotate_surface_targets_with_cross_signals,
    _candidate_label_options,
    _effective_met_site_for_background,
    _format_purity_interval,
    _hypothesis_display_label,
    _infer_likely_met_site_context,
    _integrated_evidence_bullets,
    _primary_tissues_for_analysis,
    _prioritize_report_compatible_decomposition,
    _summarize_sample_call,
)


def _mock_therapy_scores(**axis_states):
    """Build a therapy_scores dict with named axes set to given states."""
    scores = {}
    for cls, state in axis_states.items():
        scores[cls] = SimpleNamespace(state=state, message=f"{cls} is {state}")
    return scores


def _base_analysis(**overrides):
    """Minimal analysis dict."""
    a = {
        "cancer_type": "PRAD",
        "cancer_name": "Prostate Adenocarcinoma",
        "therapy_response_scores": {},
        "purity": {
            "overall_estimate": 0.6,
            "components": {"lineage": {"per_gene": []}},
        },
        "candidate_trace": [{"code": "PRAD", "support_fraction_of_top": 1.0}],
        "fit_quality": {},
    }
    a.update(overrides)
    return a


# ── compose_disease_state_narrative: BRCA patterns ──────────────────────


def test_brca_er_down_pattern():
    analysis = _base_analysis(
        cancer_type="BRCA",
        therapy_response_scores=_mock_therapy_scores(ER_signaling="down"),
        purity={
            "overall_estimate": 0.6,
            "components": {
                "lineage": {
                    "per_gene": [
                        {"gene": "ESR1", "purity": 0.01},
                    ]
                }
            },
        },
    )
    narrative = compose_disease_state_narrative(analysis)
    assert "ER" in narrative
    assert "suppressed" in narrative.lower() or "endocrine" in narrative.lower()


def test_brca_her2_up_pattern():
    analysis = _base_analysis(
        cancer_type="BRCA",
        therapy_response_scores=_mock_therapy_scores(HER2_signaling="up"),
    )
    narrative = compose_disease_state_narrative(analysis)
    assert "HER2" in narrative


def test_brca_combined_er_down_her2_up():
    analysis = _base_analysis(
        cancer_type="BRCA",
        therapy_response_scores=_mock_therapy_scores(
            ER_signaling="down",
            HER2_signaling="up",
        ),
        purity={
            "overall_estimate": 0.6,
            "components": {
                "lineage": {
                    "per_gene": [
                        {"gene": "ESR1", "purity": 0.01},
                    ]
                }
            },
        },
    )
    narrative = compose_disease_state_narrative(analysis)
    assert "ER" in narrative
    assert "HER2" in narrative


# ── compose_disease_state_narrative: EMT / hypoxia ──────────────────────


def test_emt_plus_hypoxia_aggressive_pattern():
    analysis = _base_analysis(
        therapy_response_scores=_mock_therapy_scores(EMT="up", hypoxia="up"),
    )
    narrative = compose_disease_state_narrative(analysis)
    assert "EMT" in narrative
    assert "hypoxia" in narrative.lower()
    assert "aggressive" in narrative.lower()


def test_emt_only_mesenchymal_switch():
    analysis = _base_analysis(
        therapy_response_scores=_mock_therapy_scores(EMT="up"),
    )
    narrative = compose_disease_state_narrative(analysis)
    assert "EMT" in narrative
    assert "mesenchymal" in narrative.lower()


def test_hypoxia_only_no_mention():
    """Hypoxia alone without EMT doesn't trigger the combined pattern."""
    analysis = _base_analysis(
        therapy_response_scores=_mock_therapy_scores(hypoxia="up"),
    )
    narrative = compose_disease_state_narrative(analysis)
    assert "aggressive" not in narrative.lower()


# ── compose_disease_state_narrative: IFN ─────────────────────────────────


def test_ifn_up_mentions_inflation():
    analysis = _base_analysis(
        therapy_response_scores=_mock_therapy_scores(IFN_response="up"),
    )
    narrative = compose_disease_state_narrative(analysis)
    assert "IFN" in narrative
    assert "inflation" in narrative.lower() or "IFN-driven" in narrative


def test_ifn_down_no_mention():
    analysis = _base_analysis(
        therapy_response_scores=_mock_therapy_scores(IFN_response="down"),
    )
    narrative = compose_disease_state_narrative(analysis)
    assert "inflation" not in narrative.lower()


# ── annotate_surface_targets_with_cross_signals ──────────────────────────


def test_annotate_ifn_active_tags_core_isg():
    scores = {"IFN_response": SimpleNamespace(state="up")}
    result = annotate_surface_targets_with_cross_signals(None, scores)
    assert "HLA-A" in result
    assert result["HLA-A"] == "IFN-driven"
    assert "B2M" in result


def test_annotate_ifn_inactive_returns_empty():
    scores = {"IFN_response": SimpleNamespace(state="down")}
    result = annotate_surface_targets_with_cross_signals(None, scores)
    assert result == {}


def test_annotate_ifn_missing_returns_empty():
    result = annotate_surface_targets_with_cross_signals(None, {})
    assert result == {}


# ── _candidate_label_options ─────────────────────────────────────────────


def test_candidate_label_single_strong():
    analysis = {
        "candidate_trace": [{"code": "PRAD"}, {"code": "BRCA"}],
        "fit_quality": {"label": "strong"},
    }
    assert _candidate_label_options(analysis) == ["PRAD"]


def test_candidate_label_weak_shows_two():
    analysis = {
        "candidate_trace": [{"code": "COAD"}, {"code": "READ"}],
        "fit_quality": {"label": "weak"},
    }
    assert _candidate_label_options(analysis) == ["COAD", "READ"]


def test_candidate_label_ambiguous_shows_two():
    analysis = {
        "candidate_trace": [{"code": "LUAD"}, {"code": "LUSC"}],
        "fit_quality": {"label": "ambiguous"},
    }
    assert _candidate_label_options(analysis) == ["LUAD", "LUSC"]


def test_candidate_label_respects_forced_cancer_type():
    analysis = {
        "analysis_constraints": {"cancer_type": "COAD"},
        "candidate_trace": [{"code": "SARC"}, {"code": "COAD"}],
        "fit_quality": {"label": "ambiguous"},
    }
    assert _candidate_label_options(analysis) == ["COAD"]


def test_candidate_label_uses_selected_report_scope_when_candidates_miss_it():
    analysis = {
        "report_scope_cancer_type": "COAD",
        "cancer_type_evidence": {
            "selected": {
                "cancer_type": "COAD",
                "selected_by": "tumor_label_refinement",
            }
        },
        "candidate_trace": [{"code": "SARC"}, {"code": "STAD"}],
        "fit_quality": {"label": "weak"},
    }
    assert _candidate_label_options(analysis) == ["COAD"]


def test_integrated_evidence_calls_discordant_auto_top_rna_candidate():
    analysis = _base_analysis(
        cancer_type="COAD",
        cancer_type_source="auto-detected",
        report_scope_cancer_type="COAD",
        cancer_type_evidence={
            "selected": {
                "cancer_type": "COAD",
                "selected_by": "tumor_label_refinement",
            }
        },
        candidate_trace=[
            {
                "code": "SARC",
                "signature_score": 0.72,
                "support_fraction_of_top": 1.0,
                "lineage_concordance": 1.0,
            },
            {
                "code": "STAD",
                "signature_score": 0.65,
                "support_fraction_of_top": 0.95,
                "lineage_concordance": 0.8,
            },
        ],
        call_summary={"label_options": ["COAD"], "label_display": "COAD"},
    )

    bullets = _integrated_evidence_bullets(analysis)
    text = "\n".join(bullets)
    assert "**RNA classifier line**" in text
    assert "pan-cancer signature ranker does not independently set the report label" in text
    assert "integrated evidence selected COAD (Colon Adenocarcinoma)" in text
    assert "SARC (Sarcoma) is the leading label" not in text


def test_source_resolved_call_names_bulk_signal_as_host_context():
    analysis = _base_analysis(
        cancer_type="CRC",
        cancer_type_source="auto-detected",
        report_scope_cancer_type="CRC",
        cancer_type_evidence={
            "selected": {
                "cancer_type": "CRC",
                "selected_by": "entity_evidence_consensus",
                "entity_evidence_consensus": {
                    "source_resolved_identity_decisive": True,
                },
            }
        },
        residual_identity_evidence={
            "status": "corroborated",
            "candidate_code": "CRC",
            "source_resolved_identity": True,
        },
        post_residual_decomposition_refit={"accepted": True},
        candidate_trace=[
            {
                "code": "SARC_DDLPS",
                "signature_score": 0.74,
                "support_fraction_of_top": 1.0,
            },
            {
                "code": "SARC_PLEOLPS",
                "support_fraction_of_top": 0.98,
            },
        ],
        call_summary={"label_options": ["CRC"], "label_display": "CRC"},
    )

    text = "\n".join(_integrated_evidence_bullets(analysis))

    assert "**Tumor-versus-host identity line**" in text
    assert "bulk pan-cancer signature ranker favors SARC_DDLPS" in text
    assert "retained as host/background differential context" in text
    assert "final-scope decomposition refit reproduced" in text
    assert "ahead of SARC_PLEOLPS" not in text
    assert "; signature 0.74" not in text


def test_parent_residual_is_context_not_child_selection_basis():
    analysis = _base_analysis(
        cancer_type="READ",
        cancer_type_source="auto-detected",
        report_scope_cancer_type="READ",
        cancer_type_evidence={
            "selected": {
                "cancer_type": "READ",
                "selected_by": "entity_evidence_consensus",
                "entity_evidence_consensus": {
                    "source_resolved_identity_decisive": False,
                },
            }
        },
        residual_identity_evidence={
            "status": "corroborated",
            "candidate_code": "CRC",
            "source_resolved_identity": True,
        },
        post_residual_decomposition_refit={"accepted": True},
        candidate_trace=[
            {
                "code": "SARC_DDLPS",
                "signature_score": 0.74,
                "support_fraction_of_top": 1.0,
            },
            {
                "code": "READ",
                "signature_score": 0.70,
                "support_fraction_of_top": 0.95,
            },
        ],
        call_summary={"label_options": ["READ"], "label_display": "READ"},
    )

    text = "\n".join(_integrated_evidence_bullets(analysis))

    assert "**RNA classifier line**" in text
    assert "CRC (Colorectal Adenocarcinoma) residual identity" in text
    assert "establishes the broader branch but not the more specific READ" in text
    assert "active report label is RNA rank 2" in text
    assert "complete marker and ontology program for READ" not in text


def test_integrated_evidence_separates_top_rna_candidate_from_fallback_reference():
    analysis = _base_analysis(
        cancer_type="BRCA_Basal",
        cancer_type_source="user-specified",
        reference_cancer_type="BRCA",
        expression_reference_cancer_type="BRCA_Basal",
        report_scope_cancer_type="BRCA_Basal",
        candidate_trace=[
            {
                "code": "CESC",
                "signature_score": 0.38,
                "support_fraction_of_top": 1.0,
                "lineage_concordance": 0.87,
            },
            {
                "code": "LUSC",
                "support_fraction_of_top": 0.99,
            },
            {
                "code": "BRCA",
                "support_fraction_of_top": 0.98,
            },
        ],
        call_summary={"label_options": ["BRCA_Basal"], "label_display": "BRCA_Basal"},
    )

    text = "\n".join(_integrated_evidence_bullets(analysis))

    assert (
        "CESC (Cervical Squamous / Adenocarcinoma) is the leading "
        "pan-cancer signature-ranker candidate"
    ) in text
    assert (
        "BRCA (Breast Invasive Carcinoma) is the active fallback "
        "expression/reference context"
    ) in text
    assert "CESC (Cervical Squamous / Adenocarcinoma) is the active fallback" not in text
    assert "used for cohort-normalized downstream analyses" not in text


def test_integrated_evidence_names_descendant_decomposition_without_branch_conflict():
    analysis = _base_analysis(
        cancer_type="CRC",
        reference_cancer_type="CRC",
        report_scope_cancer_type="CRC",
        candidate_trace=[
            {"code": "CRC", "signature_score": 0.7, "support_fraction_of_top": 1.0}
        ],
        call_summary={"label_options": ["CRC"], "label_display": "CRC"},
    )
    decomposition = SimpleNamespace(
        cancer_type="READ",
        template="solid_primary",
        fractions={"tumor": 0.4, "T_cell": 0.2},
        warnings=[],
    )

    text = "\n".join(_integrated_evidence_bullets(analysis, [decomposition]))

    assert "descendant background model within the CRC" in text
    assert "does not refine the report label" in text
    assert "diverging from the report-label" not in text


def test_integrated_evidence_mmr_skips_unrelated_retained_candidate():
    analysis = _base_analysis(
        cancer_type="GBM",
        cancer_name="Glioblastoma",
        candidate_trace=[{"code": "GBM", "support_fraction_of_top": 1.0}],
        cancer_type_evidence={
            "staged_evidence_graph": {
                "channels": [
                    {
                        "candidate_code": "COAD",
                        "role": "hierarchical_mismatch_repair_vote",
                        "code": "MSI",
                        "status": "admission_context",
                        "details": {
                            "label_space": "learned_mismatch_repair_release_ensemble",
                            "mismatch_repair": {
                                "context_group": "CRC",
                                "decision_threshold": 0.5,
                                "msi_probability": 0.94,
                            },
                        },
                    }
                ]
            }
        },
    )

    text = "\n".join(_integrated_evidence_bullets(analysis))

    assert "Mismatch-repair RNA context" not in text
    assert "CRC MMR ensemble" not in text


def test_candidate_label_empty_trace():
    analysis = {"candidate_trace": [], "fit_quality": {}}
    assert _candidate_label_options(analysis) == []


# ── _summarize_sample_call ───────────────────────────────────────────────


def _mock_decomp_result(**kwargs):
    defaults = {
        "cancer_type": "COAD",
        "template": "solid_primary",
        "score": 0.9,
        "purity": 0.6,
        "warnings": [],
        "template_site_factor": 0.9,
        "template_tissue_score": 0.8,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_summarize_call_primary_template():
    analysis = _base_analysis(
        cancer_type="COAD",
        fit_quality={"label": "strong"},
        candidate_trace=[{"code": "COAD", "support_fraction_of_top": 1.0}],
    )
    best = _mock_decomp_result(template="solid_primary")
    result = _summarize_sample_call(analysis, [best], sample_mode="solid")
    assert result["reported_context"] == "primary"
    assert result["site_indeterminate"] is False


def test_registry_primary_tissue_blocks_false_met_site_inference():
    analysis = _base_analysis(
        cancer_type="HEPB",
        sample_mode="solid",
        tissue_scores=[("liver", 0.96, 42)],
    )

    assert _primary_tissues_for_analysis(analysis=analysis, cancer_code="HEPB") == [
        "liver"
    ]
    assert _infer_likely_met_site_context(analysis) is None


def test_primary_site_family_blocks_false_met_site_inference():
    analysis = _base_analysis(
        cancer_type="MBL",
        sample_mode="solid",
        tissue_scores=[
            ("choroid_plexus", 0.94, 39),
            ("cerebellum", 0.81, 31),
            ("cerebral_cortex", 0.78, 28),
        ],
    )

    assert "cerebellum" in _primary_tissues_for_analysis(
        analysis=analysis,
        cancer_code="MBL",
    )
    assert _infer_likely_met_site_context(analysis) is None
    best = _mock_decomp_result(
        cancer_type="MBL",
        template="met_brain",
        site_evidence={"site_supported": True, "status": "site_supported"},
        warnings=[],
    )
    assert _effective_met_site_for_background(
        analysis | {"decomposition_results": [best]}
    ) is None
    assert "primary-compatible brain context" in _hypothesis_display_label(
        best,
        primary_code="MBL",
        analysis=analysis,
    )


def test_summarize_call_met_template_with_good_site():
    analysis = _base_analysis(
        cancer_type="COAD",
        fit_quality={"label": "strong"},
    )
    best = _mock_decomp_result(
        template="met_liver",
        template_site_factor=0.9,
        template_tissue_score=0.8,
        warnings=[],
    )
    result = _summarize_sample_call(analysis, [best], sample_mode="solid")
    assert result["reported_context"] == "met"
    assert result["reported_site"] == "liver-associated host context"


def test_summarize_call_primary_compatible_met_template_is_not_indeterminate():
    analysis = _base_analysis(
        cancer_type="GBM",
        fit_quality={"label": "strong"},
        candidate_trace=[{"code": "GBM", "support_fraction_of_top": 1.0}],
    )
    best = _mock_decomp_result(
        cancer_type="GBM",
        template="met_brain",
        template_site_factor=0.9,
        template_tissue_score=0.8,
        site_evidence={"site_supported": True, "status": "site_supported"},
        warnings=[],
    )

    result = _summarize_sample_call(analysis, [best], sample_mode="solid")

    assert result["site_indeterminate"] is False
    assert result["reported_context"] == "primary"
    assert result["reported_site"] == "primary-compatible brain context"
    assert "not treated as evidence of metastasis" in result["site_note"]
    assert _effective_met_site_for_background(analysis | {"decomposition_results": [best]}) is None
    assert "site indeterminate" not in _hypothesis_display_label(
        best,
        primary_code="GBM",
        analysis=analysis,
    )
    assert "primary-compatible brain context" in _hypothesis_display_label(
        best,
        primary_code="GBM",
        analysis=analysis,
    )


def test_format_purity_interval_handles_unestimated_values():
    assert _format_purity_interval(None, None, None) == "not estimated"
    assert _format_purity_interval(0.42, None, None) == "42%"
    assert _format_purity_interval(0.42, 0.2, 0.8) == "42% [20%-80%]"


def test_known_label_prioritizes_report_compatible_decomposition():
    from types import SimpleNamespace

    from trufflepig.main import _prioritize_report_compatible_decomposition

    raw_top = SimpleNamespace(
        cancer_type="PCPG",
        template="solid_primary",
        score=0.9,
        warnings=[],
    )
    compatible = SimpleNamespace(
        cancer_type="SCLC",
        template="solid_primary",
        score=0.4,
        warnings=[],
    )

    ordered = _prioritize_report_compatible_decomposition(
        [raw_top, compatible],
        reference_code="SCLC",
        report_code="SCLC_YAP1",
        enabled=True,
    )

    assert ordered[0] is compatible
    assert ordered[1] is raw_top
    assert "raw best fit was PCPG/solid_primary" in compatible.warnings[-1]


def test_blind_report_call_prioritizes_compatible_background_decomposition():
    raw_best = SimpleNamespace(
        cancer_type="HNSC",
        template="met_brain",
        score=0.55,
        warnings=["Many genes are overexplained by the TME background"],
        site_evidence={"site_supported": False, "status": "fit_only"},
    )
    weak_report_met = SimpleNamespace(
        cancer_type="CESC",
        template="met_brain",
        score=0.18,
        warnings=[
            "Primary tissue support exceeds metastatic-site support",
            "Many genes are overexplained by the TME background",
        ],
        site_evidence={"site_supported": False, "status": "fit_only"},
        template_site_factor=0.57,
        template_tissue_score=0.52,
    )
    report_primary = SimpleNamespace(
        cancer_type="CESC",
        template="solid_primary",
        score=0.12,
        warnings=[],
        site_evidence={},
    )

    ordered = _prioritize_report_compatible_decomposition(
        [raw_best, weak_report_met, report_primary],
        reference_code="CESC",
        report_code="CESC",
        enabled=False,
        analysis={"cancer_type": "CESC"},
    )

    assert ordered[0] is report_primary
    assert ordered[1] is raw_best
    assert any(
        "Selected report-compatible decomposition" in warning
        for warning in report_primary.warnings
    )


def test_summarize_call_weak_fit_met_is_indeterminate():
    analysis = _base_analysis(
        cancer_type="COAD",
        fit_quality={"label": "weak"},
    )
    best = _mock_decomp_result(template="met_liver")
    result = _summarize_sample_call(analysis, [best], sample_mode="solid")
    assert result["site_indeterminate"] is True
    assert result["site_note"] is not None


def test_summarize_call_low_site_factor_is_indeterminate():
    analysis = _base_analysis(
        cancer_type="COAD",
        fit_quality={"label": "strong"},
    )
    best = _mock_decomp_result(
        template="met_liver",
        template_site_factor=0.5,  # below 0.75
        template_tissue_score=0.3,  # below 0.4
        warnings=[],
    )
    result = _summarize_sample_call(analysis, [best], sample_mode="solid")
    assert result["site_indeterminate"] is True


def test_summarize_call_fit_only_met_template_is_indeterminate():
    analysis = _base_analysis(
        cancer_type="COAD",
        fit_quality={"label": "strong"},
    )
    best = _mock_decomp_result(
        template="met_bone",
        template_site_factor=0.9,
        template_tissue_score=0.8,
        site_evidence={"site_supported": False, "status": "fit_only"},
        warnings=[],
    )
    result = _summarize_sample_call(analysis, [best], sample_mode="solid")
    assert result["site_indeterminate"] is True
    assert result["reported_site"] is None
    assert "fit is useful" in result["site_note"]


def test_summarize_call_divergent_overexplained_met_template_is_indeterminate():
    analysis = _base_analysis(
        cancer_type="NUTM",
        reference_cancer_type="ESCA",
        expression_reference_cancer_type="NUTM",
        cancer_type_context={
            "report_code": "NUTM",
            "reference_code": "ESCA",
            "expression_code": "NUTM",
        },
        fit_quality={"label": "strong"},
    )
    best = _mock_decomp_result(
        cancer_type="HNSC",
        template="met_bone",
        template_site_factor=0.9,
        template_tissue_score=0.8,
        site_evidence={"site_supported": True, "status": "site_supported"},
        warnings=["Many genes are overexplained by the TME background"],
    )
    result = _summarize_sample_call(analysis, [best], sample_mode="solid")
    assert result["site_indeterminate"] is True
    assert result["reported_site"] is None
    assert result["reported_context"] is None


def test_summarize_call_explicit_site_hint_survives_divergent_warning():
    analysis = _base_analysis(
        cancer_type="NUTM",
        reference_cancer_type="ESCA",
        expression_reference_cancer_type="NUTM",
        fit_quality={"label": "strong"},
    )
    best = _mock_decomp_result(
        cancer_type="HNSC",
        template="met_bone",
        template_site_factor=0.9,
        template_tissue_score=0.8,
        site_evidence={
            "site_supported": True,
            "status": "site_supported",
            "basis": "site_hint",
        },
        warnings=["Many genes are overexplained by the TME background"],
    )
    result = _summarize_sample_call(analysis, [best], sample_mode="solid")
    assert result["site_indeterminate"] is False
    assert result["reported_site"] == "bone-associated host context"


def test_summarize_call_explicit_met_site_survives_fit_warning():
    analysis = _base_analysis(
        cancer_type="BLCA",
        fit_quality={"label": "ambiguous"},
        analysis_constraints={"met_site": "liver"},
    )
    best = _mock_decomp_result(
        cancer_type="BLCA",
        template="met_liver",
        template_site_factor=0.9,
        template_tissue_score=0.8,
        site_evidence={"site_supported": True, "status": "site_supported"},
        warnings=["Many genes are overexplained by the TME background"],
    )

    result = _summarize_sample_call(analysis, [best], sample_mode="solid")

    assert result["site_indeterminate"] is False
    assert result["reported_context"] == "met"
    assert result["reported_site"] == "liver-associated host context"


def test_explicit_met_site_does_not_bless_a_different_template():
    analysis = _base_analysis(
        cancer_type="BLCA",
        fit_quality={"label": "strong"},
        analysis_constraints={"met_site": "liver"},
    )
    best = _mock_decomp_result(
        cancer_type="BLCA",
        template="met_bone",
        site_evidence={"site_supported": True, "status": "site_supported"},
        warnings=[],
    )

    result = _summarize_sample_call(analysis, [best], sample_mode="solid")

    assert result["site_indeterminate"] is True
    assert result["reported_site"] is None


def test_report_compatible_decomposition_skips_unsupported_met_template():
    analysis = _base_analysis(cancer_type="BRCA_Basal")
    raw_best = SimpleNamespace(
        cancer_type="LUSC",
        template="met_peritoneal",
        score=0.16,
        warnings=[],
        site_evidence={"site_supported": True},
    )
    unsupported_report_met = SimpleNamespace(
        cancer_type="BRCA",
        template="met_skin",
        score=0.07,
        warnings=[
            "Template-specific host component is effectively unused",
            "Metastatic-site evidence below template-specific threshold",
        ],
        site_evidence={"site_supported": False, "status": "fit_only"},
    )
    report_primary = SimpleNamespace(
        cancer_type="BRCA",
        template="solid_primary",
        score=0.05,
        warnings=[],
        site_evidence={},
    )

    prioritized = _prioritize_report_compatible_decomposition(
        [raw_best, unsupported_report_met, report_primary],
        reference_code="BRCA",
        report_code="BRCA_Basal",
        enabled=True,
        analysis=analysis,
    )

    assert prioritized[0] is report_primary
    assert any(
        "Selected fallback-reference decomposition" in warning
        for warning in report_primary.warnings
    )


def test_report_decomposition_requires_independent_context_for_met_site():
    """A host component fit cannot by itself assign an anatomic sample site."""
    analysis = _base_analysis(cancer_type="NUTM")
    met_fit = SimpleNamespace(
        cancer_type="NUTM",
        template="met_peritoneal",
        score=0.28,
        warnings=[],
        site_evidence={"site_supported": True, "status": "site_supported"},
    )
    primary_fit = SimpleNamespace(
        cancer_type="NUTM",
        template="solid_primary",
        score=0.25,
        warnings=[],
        site_evidence={},
    )

    ordered = _prioritize_report_compatible_decomposition(
        [met_fit, primary_fit],
        reference_code="NUTM",
        report_code="NUTM",
        enabled=True,
        analysis=analysis,
    )

    assert ordered[0] is primary_fit
    assert ordered[1] is met_fit


def test_report_decomposition_uses_independently_inferred_met_site():
    analysis = _base_analysis(cancer_type="BLCA")
    analysis["inferred_site_context"] = {
        "site": "liver",
        "basis": "strong_off_primary_host_tissue",
    }
    met_fit = SimpleNamespace(
        cancer_type="BLCA",
        template="met_liver",
        score=0.28,
        warnings=[],
        site_evidence={"site_supported": True, "status": "site_supported"},
    )
    primary_fit = SimpleNamespace(
        cancer_type="BLCA",
        template="solid_primary",
        score=0.25,
        warnings=[],
        site_evidence={},
    )

    ordered = _prioritize_report_compatible_decomposition(
        [met_fit, primary_fit],
        reference_code="BLCA",
        report_code="BLCA",
        enabled=True,
        analysis=analysis,
    )

    assert ordered[0] is met_fit
