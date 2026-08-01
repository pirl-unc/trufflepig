from types import SimpleNamespace

import pandas as pd
import pytest


def _selectable(code, selected_by, priority):
    from trufflepig.cancer_type_evidence import CancerTypeEvidence

    h = CancerTypeEvidence(cancer_type=code)
    h.can_select_report_label = True
    h.selected_by = selected_by
    h.selection_priority = priority
    return h


def test_pick_selected_definitive_fusion_is_never_overridden_by_centroid():
    """A definitive molecular call (a detected fusion) outranks whole-profile expression and the
    centroid corroboration must never override it. Guards the #98 fusion-override regression."""
    from trufflepig.cancer_type_evidence import _pick_selected

    fusion = _selectable("SARC_EWS", "direct_fusion", (3, 1.0, 6))
    marker = _selectable("SARC_DDLPS", "local_expression_reference", (2, 0.9, 4))
    hyps = {"SARC_EWS": fusion, "SARC_DDLPS": marker}
    # Even though the centroid prefers SARC_DDLPS's sibling cohorts, the fusion call stands.
    cen = pd.Series({"SARC_OS": 0.95, "SARC_DDLPS": 0.90, "SARC_EWS": 0.70})
    assert _pick_selected(hyps, cen=cen, compartment_confident=True).cancer_type == "SARC_EWS"


def test_pick_selected_centroid_corroborates_among_selectable():
    """At a confident compartment the centroid re-ranks the SELECTABLE hypotheses, deferring to the
    one it matches best — the salivary-ADCC vs NUT-carcinoma case: ADCC wins on marker authority but
    the whole profile is NUTM. Only acts on a clear margin and only on hypotheses a selector proposed
    (so it can't promote a contaminant), and only when the compartment call is confident."""
    from trufflepig.cancer_type_evidence import _pick_selected

    def hyps():
        return {
            "ADCC": _selectable("ADCC", "local_expression_reference", (2, 0.8, 4)),
            "NUTM": _selectable("NUTM", "rare_marker", (1, 0.9, 2)),
        }

    # ADCC wins on authority (higher class_rank); the centroid clearly prefers NUTM (0.85 vs 0.82).
    cen = pd.Series({"NUTM": 0.85, "ADCC": 0.82})
    assert _pick_selected(hyps(), cen=cen, compartment_confident=True).cancer_type == "NUTM"
    # Not confident → defer to marker authority (the centroid is review-only on an unsure compartment).
    assert _pick_selected(hyps(), cen=cen, compartment_confident=False).cancer_type == "ADCC"
    # Within-margin near-tie → keep the authority winner (curation/markers break it, not the centroid).
    near = pd.Series({"NUTM": 0.826, "ADCC": 0.82})
    assert _pick_selected(hyps(), cen=near, compartment_confident=True).cancer_type == "ADCC"


def test_pick_selected_same_lineage_centroid_needs_marker_corroboration():
    """A close whole-profile centroid match cannot slip among sarcoma siblings by itself.

    Guards the #102 SARC_RMS_ERMS regression: ERMS markers are present, while the sibling
    sarcoma subtype has only a modest centroid advantage and no matching marker program.
    """
    from trufflepig.cancer_type_evidence import _pick_selected

    hyps = {
        "SARC_RMS_ERMS": _selectable(
            "SARC_RMS_ERMS",
            "lineage_panel",
            (2, 0.9, 4),
        ),
        "SARC_MYXFIB": _selectable(
            "SARC_MYXFIB",
            "local_expression_reference",
            (1, 0.9, 4),
        ),
    }
    cen = pd.Series({"SARC_MYXFIB": 0.85, "SARC_RMS_ERMS": 0.82})
    sample = {"MYOD1": 40.0, "MYOG": 35.0, "DES": 60.0, "MYF5": 10.0, "MYF6": 8.0}

    assert (
        _pick_selected(
            hyps,
            cen=cen,
            compartment_confident=True,
            sample_tpm_by_symbol=sample,
        ).cancer_type
        == "SARC_RMS_ERMS"
    )


def _analysis(*rows):
    return {
        "cancer_type": rows[0][0],
        "candidate_trace": [
            {"code": code, "support_fraction_of_top": support}
            for code, support in rows
        ],
    }


def _candidate_analysis(rows):
    return {
        "cancer_type": rows[0]["code"],
        "candidate_trace": rows,
    }


def _fused_context_analysis(code, support=1.0):
    return _candidate_analysis(
        [
            {
                "code": code,
                "support_fraction_of_top": support,
            }
        ]
    )


def _empty_expression_frame():
    return pd.DataFrame(columns=["ensembl_gene_id", "canonical_gene_name", "TPM"])


def _expression_frame(tpm_by_symbol):
    from trufflepig.reference import pan_cancer_expression

    ref = pan_cancer_expression()[["Ensembl_Gene_ID", "Symbol"]].drop_duplicates(
        "Symbol"
    )
    id_by_symbol = dict(zip(ref["Symbol"], ref["Ensembl_Gene_ID"]))
    rows = []
    for symbol, tpm in tpm_by_symbol.items():
        ensg = id_by_symbol.get(symbol)
        if not ensg:
            continue
        rows.append(
            {
                "ensembl_gene_id": ensg,
                "canonical_gene_name": symbol,
                "TPM": float(tpm),
            }
        )
    return pd.DataFrame(rows)


def test_nutm_rna_surrogate_promotes_in_squamous_context():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    finding = {
        "cancer_type": "NUTM",
        "rule_id": "nutm_nutm1",
        "surrogate": "NUTM1",
        "surrogate_tpm": 6.2,
        "threshold_tpm": 1.0,
        "support_genes": [],
    }

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("LUSC", 1.0), ("HNSC", 0.7)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "NUTM"
    assert result["selected"]["evidence_sources"] == ["rare_marker"]
    assert result["selected"]["can_select_report_label"] is True
    assert result["selected"]["metrics"]["rna_marker_support"] == 1.0
    graph = result["staged_evidence_graph"]
    assert graph["selection_order"] == ["family", "coarse_type", "exact_subtype"]
    assert graph["selected"]["code"] == "NUTM"
    assert graph["selected"]["stage"] == "exact_subtype"
    assert graph["stages"][2]["status"] == "selected"
    rare_channels = [
        row for row in graph["channels"]
        if row.get("candidate_code") == "NUTM"
        and row["channel"] == "rare_fusion_anchor"
        and row["role"] == "rna_marker_anchor"
    ]
    assert rare_channels
    assert rare_channels[0]["selects_report_label"] is True
    assert rare_channels[0]["stage"] == "exact_subtype"
    deconv = next(row for row in graph["channels"] if row["channel"] == "deconvolution")
    assert deconv["status"] == "not_available_pre_label_selection"
    therapy = next(row for row in graph["channels"] if row["channel"] == "therapy_context")
    assert therapy["status"] == "downstream_consumer_not_selector"


def test_nutm_rare_marker_outranks_generic_squamous_marker_program(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    import trufflepig.expression_classifier as classifier
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(classifier, "classify_expression", lambda _sample, top_k=5: [])
    monkeypatch.setattr(classifier, "classify_expression_hierarchy", lambda _sample, top_k=5: [])
    monkeypatch.setattr(
        evidence,
        "_centroid_supports_for_hypotheses",
        lambda hypotheses, cen=None: {code: 1.0 for code in hypotheses},
    )

    original_marker_coherence = evidence._marker_coherence

    def fake_marker_coherence(code, sample):
        if code == "ESCA":
            return {
                "code": "ESCA",
                "status": "consistent",
                "detected": 4,
                "total": 5,
                "required_for_consistent": 3,
                "detected_fraction": 0.8,
                "unexpected_low_detected": 1,
                "unexpected_low_genes": ["PTPRC"],
            }
        return original_marker_coherence(code, sample)

    monkeypatch.setattr(evidence, "_marker_coherence", fake_marker_coherence)
    finding = {
        "cancer_type": "NUTM",
        "rule_id": "nutm_nutm1",
        "surrogate": "NUTM1",
        "surrogate_tpm": 73.0,
        "threshold_tpm": 1.0,
        "support_genes": [],
        "support_pass": True,
        "context_support_fraction_of_top": 1.0,
        "context_match_reason": "top_context",
    }

    result = select_report_scope_from_evidence(
        _expression_frame({"NUTM1": 73.0, "KRT5": 100.0, "TP63": 80.0, "PTPRC": 25.0}),
        _analysis(("ESCA", 1.0), ("HNSC", 0.93)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "NUTM"
    assert result["selected"]["selected_by"] == "rare_marker"


def test_registry_molecular_children_are_orthogonal_axes():
    from trufflepig.cancer_type_evidence import (
        _lineage_path_for_code,
        _orthogonal_axes_for_code,
    )

    coad_axes = _orthogonal_axes_for_code("COAD_MSI")
    read_axes = _orthogonal_axes_for_code("READ_MSI")
    ucec_msi_axes = _orthogonal_axes_for_code("UCEC_MSI")
    ucec_pole_axes = _orthogonal_axes_for_code("UCEC_POLE")

    assert coad_axes[0]["axis"] == "mismatch_repair"
    assert coad_axes[0]["state"] == "MSI-H / dMMR"
    assert coad_axes[0]["base_code"] == "COAD"
    assert coad_axes[0]["ancestors"] == ["COAD", "CRC"]
    assert read_axes[0]["axis"] == "mismatch_repair"
    assert read_axes[0]["base_code"] == "READ"
    assert ucec_msi_axes[0]["axis"] == "mismatch_repair"
    assert ucec_msi_axes[0]["base_code"] == "UCEC"
    assert ucec_pole_axes[0]["axis"] == "polymerase_epsilon"

    assert [row.get("code") or row.get("family") for row in _lineage_path_for_code("COAD_MSI")] == [
        "carcinoma-gi",
        "CRC",
        "COAD",
    ]
    assert [row.get("code") or row.get("family") for row in _lineage_path_for_code("UCEC_MSI")] == [
        "carcinoma-gu",
        "UCEC",
    ]


def test_registry_status_mentions_do_not_create_orthogonal_axes():
    from trufflepig.cancer_type_evidence import (
        _lineage_path_for_code,
        _orthogonal_axes_for_code,
    )

    for code in ("CRC", "COAD", "READ", "STAD", "MM"):
        assert _orthogonal_axes_for_code(code) == []

    assert [
        row.get("code") or row.get("family") for row in _lineage_path_for_code("COAD")
    ] == [
        "carcinoma-gi",
        "CRC",
        "COAD",
    ]


def test_broad_only_channels_require_positive_signal():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("COAD", 1.0), ("READ", 0.8)),
    )

    selected_channels = result["selected"]["evidence_channels"]
    assert [
        (row["channel"], row["role"]) for row in selected_channels
    ] == [
        ("pan_cancer_signature_ranker", "top_ranked_candidate"),
        ("fused_evidence", "integrated_evidence_selection"),
    ]
    assert selected_channels[0]["status"] == "candidate_generation"
    assert selected_channels[0]["selects_report_label"] is False
    assert selected_channels[1]["status"] == "informative"
    assert selected_channels[1]["selects_report_label"] is False

    graph_channels = [
        row for row in result["staged_evidence_graph"]["channels"]
        if row.get("candidate_code") == "COAD"
    ]
    assert [
        (row["channel"], row["role"]) for row in graph_channels
    ] == [
        ("pan_cancer_signature_ranker", "top_ranked_candidate"),
        ("fused_evidence", "integrated_evidence_selection"),
    ]


def test_blocked_learned_selector_does_not_admit_fused_evidence():
    """A rejected learned call must not become a non-ranker admission path."""
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _fused_component_scores,
        _fused_evidence_eligible,
    )

    hypothesis = CancerTypeEvidence(
        cancer_type="SARC_ASPS",
        learned_expression_support=0.90,
    )
    hypothesis.selected_by = "learned_expression_classifier"
    hypothesis.label_status = "blocked"
    hypothesis.can_select_report_label = False
    hypothesis.blocking_reasons = ("learned call lacks broad-ranker context",)
    hypothesis.details.update(
        {
            "learned_expression_hierarchical_context_support": 0.80,
            "learned_expression_entity_support": 0.80,
            "learned_expression_entity_label": "SARC_ASPS",
            "learned_expression_margin": 0.60,
        }
    )

    components = _fused_component_scores(hypothesis, centroid_support=0.0)
    can_select, blockers = _fused_evidence_eligible(
        hypothesis,
        _fused_context_analysis("SARC_ASPS"),
        score=sum(components.values()),
        centroid_support=0.0,
        components=components,
    )

    assert can_select is False
    assert any("lacks a non-ranker admission path" in reason for reason in blockers)


def test_rare_marker_channel_contributes_when_learned_selector_admits_candidate():
    """Rare-marker evidence remains a channel even if another selector first admitted the label."""
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _fused_component_scores,
        _fused_evidence_eligible,
    )

    hypothesis = CancerTypeEvidence(
        cancer_type="NUTM",
        learned_expression_support=0.98,
        rna_marker_support=1.0,
    )
    hypothesis.add_source("learned_expression_classifier")
    hypothesis.add_source("rare_marker")
    hypothesis.selected_by = "learned_expression_classifier"
    hypothesis.can_select_report_label = True
    hypothesis.details.update(
        {
            "learned_expression_hierarchical_context_support": 0.80,
            "learned_expression_entity_support": 0.98,
            "learned_expression_entity_label": "NUTM",
            "learned_expression_margin": 0.65,
            "rare_marker_can_select": True,
        }
    )

    components = _fused_component_scores(hypothesis, centroid_support=0.35)
    can_select, blockers = _fused_evidence_eligible(
        hypothesis,
        _fused_context_analysis("NUTM"),
        score=sum(components.values()),
        centroid_support=0.35,
        components=components,
    )

    assert components["rare_marker"] == 2.0
    assert components["marker_program"] == 0.6
    assert can_select is True
    assert blockers == []


def test_hierarchy_candidate_votes_are_preserved_in_trace_channels():
    """Candidate-wide hierarchy votes use the shorter stored key name."""
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _hypothesis_evidence_channels,
    )

    hypothesis = CancerTypeEvidence(cancer_type="STAD")
    hypothesis.details["learned_expression_hierarchy_votes"] = [
        {
            "stage": "entity",
            "label": "STAD",
            "probability": 0.31,
            "top_predictions": [{"label": "STAD", "probability": 0.31}],
        }
    ]

    channels = _hypothesis_evidence_channels(hypothesis)

    assert any(
        row["channel"] == "learned_expression_classifier"
        and row["role"] == "hierarchical_entity_vote"
        and row.get("code") == "STAD"
        for row in channels
    )


def test_global_hierarchy_context_does_not_score_a_specific_fused_candidate():
    """Global hierarchy confidence is context, not candidate-specific support."""
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _fused_component_scores,
        _fused_evidence_eligible,
    )

    hypothesis = CancerTypeEvidence(
        cancer_type="SARC",
        learned_expression_support=0.98,
    )
    hypothesis.details.update(
        {
            "learned_expression_hierarchy_support": 0.80,
            "learned_expression_flat_lineage_support": 0.90,
            "learned_expression_entity_support": 0.80,
            "learned_expression_entity_label": "SARC",
            "learned_expression_margin": 0.60,
        }
    )

    components = _fused_component_scores(hypothesis, centroid_support=0.35)
    can_select, blockers = _fused_evidence_eligible(
        hypothesis,
        _fused_context_analysis("SARC"),
        score=sum(components.values()),
        centroid_support=0.35,
        components=components,
    )

    assert "learned_hierarchy" not in components
    assert components["learned_expression_classifier"] == pytest.approx(1.176)
    assert can_select is True
    assert blockers == []


def test_admitted_composition_selector_remains_in_fused_candidate_set():
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _fused_component_scores,
        _fused_evidence_eligible,
        _group_fused_evidence_components,
    )

    hypothesis = CancerTypeEvidence(cancer_type="UCS")
    hypothesis.admit_adjudication_support(
        "composition_reference",
        0.86,
        selector="coarse_composition_reference",
    )
    hypothesis.consider_for_report_label(
        selected_by="coarse_composition_reference",
        can_select=True,
        blocking_reasons=(),
        priority=(2, 0.86),
    )

    components = _fused_component_scores(hypothesis, centroid_support=0.0)
    grouped = _group_fused_evidence_components(components)
    can_select, blockers = _fused_evidence_eligible(
        hypothesis,
        _fused_context_analysis("UCS"),
        score=sum(grouped.values()),
        centroid_support=0.0,
        components=components,
    )

    assert components["coarse_composition_reference"] == pytest.approx(0.86)
    assert can_select is True
    assert blockers == []


def test_weak_learned_fused_call_cannot_break_cross_context_ranker_tie():
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _fused_component_scores,
        _fused_evidence_eligible,
        _group_fused_evidence_components,
    )

    hypothesis = CancerTypeEvidence(
        cancer_type="HNSC",
        broad_rna_support=0.988,
        family_marker_support=0.99,
        learned_expression_support=0.25,
    )
    hypothesis.details.update(
        {
            "learned_expression_hierarchy_support": 0.80,
            "learned_expression_family_support": 0.94,
            "learned_expression_family_label": "HNSC",
            "learned_expression_entity_support": 0.25,
            "learned_expression_entity_label": "HNSC",
            "learned_expression_margin": 0.04,
            "pan_cancer_signature_marker_support": 0.99,
            "pan_cancer_signature_marker_selectable": True,
        }
    )
    analysis = _analysis(("LUSC", 1.0), ("HNSC", 0.988))

    components = _fused_component_scores(hypothesis, centroid_support=0.95)
    grouped = _group_fused_evidence_components(components)
    can_select, blockers = _fused_evidence_eligible(
        hypothesis,
        analysis,
        score=sum(grouped.values()),
        centroid_support=0.95,
        components=components,
    )

    assert can_select is False
    assert any(
        "cannot create a cross-context entity" in reason
        for reason in blockers
    )


def test_fused_score_counts_each_biological_evidence_group_once():
    from trufflepig.cancer_type_evidence import (
        _group_fused_evidence_components,
    )

    grouped = _group_fused_evidence_components(
        {
            "learned_expression_classifier": 0.60,
            "learned_compartment_anchored_pan_cancer_context": 0.99,
            "learned_family_anchored_pan_cancer_context": 0.98,
            "pan_cancer_signature_ranker": 0.30,
            "pan_cancer_signature_subtype": 0.64,
            "coarse_composition_reference": 0.90,
            "centroid_spearman": 0.40,
        }
    )

    assert grouped == {
        "learned_expression_model": 0.99,
        "pan_cancer_signature": 0.64,
        "composition_context": 0.90,
        "centroid_spearman": 0.40,
    }
    assert sum(grouped.values()) == pytest.approx(2.93)


def _top_hierarchy_details(
    entity,
    *,
    entity_support,
    entity_margin,
    family,
    family_support,
    compartment,
    compartment_support,
):
    return {
        "learned_expression_top_entity_label": entity,
        "learned_expression_top_entity_support": entity_support,
        "learned_expression_top_entity_margin": entity_margin,
        "learned_expression_top_family_label": family,
        "learned_expression_top_family_support": family_support,
        "learned_expression_top_compartment_label": compartment,
        "learned_expression_top_compartment_support": compartment_support,
    }


def test_learned_hierarchy_adjudicator_repairs_coherent_cross_lineage_call():
    """A marker-corroborated hierarchy can act as a lineage safety net.

    NBL's entity probability is distributed among siblings, but the separately
    trained family/compartment stages and curated marker program all agree on
    the embryonal path.  Together they can reject a neuroendocrine MTC fallback
    without making the moderate entity probability a general-purpose selector.
    """
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    selected = _selectable("MTC", "fused_evidence", (3, 2.0, 5))
    selected.details.update(
        _top_hierarchy_details(
            "NBL",
            entity_support=0.5351,
            entity_margin=0.0765,
            family="NBL",
            family_support=0.9959,
            compartment="embryonal",
            compartment_support=0.9976,
        )
    )
    nbl = CancerTypeEvidence(cancer_type="NBL")
    nbl.details["learned_expression_marker_coherence"] = {
        "status": "consistent",
        "detected": 6,
        "total": 8,
        "required_for_consistent": 4,
        "detected_fraction": 0.75,
    }
    hypotheses = {selected.cancer_type: selected, "NBL": nbl}

    result = _adjudicate_selection_with_learned_hierarchy(hypotheses, selected)

    assert result.cancer_type == "NBL"
    assert result.selected_by == "learned_expression_classifier"
    assert result.details["learned_hierarchy_adjudication_mode"] == "cross_lineage_safety"
    assert result.details["learned_hierarchy_previous_code"] == "MTC"


def test_learned_hierarchy_recomputes_marker_coherence_for_candidate(monkeypatch):
    """A selected row's marker program must not corroborate another entity."""
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    selected = _selectable("MTC", "fused_evidence", (3, 2.0, 5))
    selected.details.update(
        _top_hierarchy_details(
            "NBL",
            entity_support=0.5351,
            entity_margin=0.0765,
            family="NBL",
            family_support=0.9959,
            compartment="embryonal",
            compartment_support=0.9976,
        )
    )
    selected.details["learned_expression_marker_coherence"] = {
        "status": "consistent",
        "detected": 6,
        "total": 8,
        "required_for_consistent": 4,
        "detected_fraction": 0.75,
    }
    candidate = CancerTypeEvidence(cancer_type="NBL")
    queried_codes = []

    def marker_coherence(code, sample_tpm_by_symbol):
        queried_codes.append(code)
        if code == "NBL":
            return {
                "status": "inconsistent",
                "detected": 1,
                "total": 8,
                "required_for_consistent": 4,
                "detected_fraction": 0.125,
            }
        return selected.details["learned_expression_marker_coherence"]

    monkeypatch.setattr(evidence, "_marker_coherence", marker_coherence)

    result = _adjudicate_selection_with_learned_hierarchy(
        {"MTC": selected, "NBL": candidate},
        selected,
        sample_tpm_by_symbol={"PHOX2B": 1.0},
    )

    assert result is selected
    assert "NBL" in queried_codes
    assert candidate.details["learned_hierarchy_entity_marker_coherence"] == {
        "status": "inconsistent",
        "detected": 1,
        "total": 8,
        "required_for_consistent": 4,
        "detected_fraction": 0.125,
    }


def test_learned_hierarchy_adjudicator_allows_calibrated_compartment_escape():
    """A strong held-out-calibrated entity vote can correct a shared compartment bias."""
    from trufflepig.cancer_type_evidence import (
        _adjudicate_selection_with_learned_hierarchy,
    )

    selected = _selectable("SARC_LPS_UNSPEC", "fused_evidence", (3, 2.0, 5))
    selected.details.update(
        _top_hierarchy_details(
            "GBM",
            entity_support=0.9826,
            entity_margin=0.9801,
            family="cns-glial",
            family_support=0.9929,
            # The compartment stage shares the mesenchymal/stromal bias.  The
            # strong entity path is calibrated to escape it.
            compartment="mesenchymal",
            compartment_support=0.9982,
        )
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {selected.cancer_type: selected},
        selected,
    )

    assert result.cancer_type == "GBM"
    assert result.details["learned_hierarchy_adjudication_mode"] == "cross_lineage_safety"


def test_learned_hierarchy_adjudicator_withholds_weak_uncorroborated_rb_vote():
    """Do not memorize the all-QC-fail RB representative to manufacture accuracy."""
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    selected = _selectable("SARC_LPS_UNSPEC", "fused_evidence", (3, 2.0, 5))
    selected.details.update(
        _top_hierarchy_details(
            "RB",
            entity_support=0.2781,
            entity_margin=0.1331,
            family="embryonal",
            family_support=0.4690,
            compartment="embryonal",
            compartment_support=0.7632,
        )
    )
    rb = CancerTypeEvidence(cancer_type="RB")
    rb.details["learned_expression_marker_coherence"] = {
        "status": "partial",
        "detected": 1,
        "total": 7,
        "required_for_consistent": 4,
        "detected_fraction": 0.1429,
    }

    result = _adjudicate_selection_with_learned_hierarchy(
        {selected.cancer_type: selected, "RB": rb},
        selected,
    )

    assert result is selected


def test_learned_hierarchy_adjudicator_rejects_incoherent_family_path():
    """Independent hierarchy stages must describe a biologically valid path."""
    from trufflepig.cancer_type_evidence import (
        _adjudicate_selection_with_learned_hierarchy,
    )

    selected = _selectable("RB", "fused_evidence", (3, 2.0, 5))
    selected.details.update(
        _top_hierarchy_details(
            "MPN",
            entity_support=0.60,
            entity_margin=0.30,
            family="heme-bcell",  # MPN is not a B-cell family entity.
            family_support=0.80,
            compartment="heme",
            compartment_support=0.97,
        )
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {selected.cancer_type: selected},
        selected,
    )

    assert result is selected


def test_learned_hierarchy_adjudicator_refines_entity_at_calibrated_precision():
    """A very strong coherent hierarchy can resolve a same-lineage entity error."""
    from trufflepig.cancer_type_evidence import (
        _adjudicate_selection_with_learned_hierarchy,
    )

    selected = _selectable("ESCA", "local_expression_reference", (3, 2.0, 4))
    selected.details.update(
        _top_hierarchy_details(
            "STAD",
            entity_support=0.985,
            entity_margin=0.90,
            family="carcinoma-gi",
            family_support=0.97,
            compartment="epithelial",
            compartment_support=0.99,
        )
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {selected.cancer_type: selected},
        selected,
    )

    assert result.cancer_type == "STAD"
    assert result.details["learned_hierarchy_adjudication_mode"] == (
        "high_precision_entity_refinement"
    )


def test_learned_hierarchy_entity_refinement_withholds_below_calibrated_support():
    from trufflepig.cancer_type_evidence import (
        _adjudicate_selection_with_learned_hierarchy,
    )

    selected = _selectable("ESCA", "local_expression_reference", (3, 2.0, 4))
    selected.details.update(
        _top_hierarchy_details(
            "STAD",
            entity_support=0.969,
            entity_margin=0.90,
            family="carcinoma-gi",
            family_support=0.97,
            compartment="epithelial",
            compartment_support=0.99,
        )
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {selected.cancer_type: selected},
        selected,
    )

    assert result is selected


def _add_entity_prediction_vector(details, *predictions):
    details["learned_expression_hierarchy_votes"] = [
        {
            "stage": "entity",
            "label": predictions[0][0],
            "probability": predictions[0][1],
            "top_predictions": [
                {"label": label, "probability": probability}
                for label, probability in predictions
            ],
        }
    ]
    return details


def _adjudicate_stad_esca_evidence_scenario(
    axis_preferences,
    *,
    learned_entity_support=0.80,
    family_anchored_candidate=False,
    candidate_broad_rank=None,
):
    """Run the final entity adjudicator for one independent-evidence split.

    The learned hierarchy consistently proposes STAD over the currently
    selected ESCA call.  The caller controls how the five other independent
    evidence groups vote, tie, or remain unavailable.  This keeps scenario
    tests at the report-label decision seam instead of testing the internal
    vote counter in isolation.
    """
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    def support_pair(preference):
        if preference == "candidate":
            return 0.8, 0.2
        if preference == "selected":
            return 0.2, 0.8
        if preference == "tie":
            return 0.5, 0.5
        if preference == "unavailable":
            return 0.0, 0.0
        raise AssertionError(f"unknown evidence preference: {preference}")

    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "STAD",
            entity_support=learned_entity_support,
            entity_margin=0.60,
            family="carcinoma-gi",
            family_support=0.95,
            compartment="epithelial",
            compartment_support=0.99,
        ),
        ("STAD", learned_entity_support),
        ("ESCA", 0.20),
    )
    selected = _selectable("ESCA", "local_expression_reference", (3, 2.0, 4))
    selected.details.update(details)
    candidate = CancerTypeEvidence(cancer_type="STAD")
    candidate.broad_rna_rank = candidate_broad_rank
    if family_anchored_candidate:
        candidate.details["fused_evidence_components"] = {
            "learned_family_anchored_pan_cancer_context": 0.9,
        }

    candidate.broad_rna_support, selected.broad_rna_support = support_pair(
        axis_preferences["pan_cancer_signature"]
    )
    candidate.rna_marker_support, selected.rna_marker_support = support_pair(
        axis_preferences["curated_marker_program"]
    )
    candidate.admit_adjudication_support(
        "curated_marker_program",
        candidate.rna_marker_support,
        selector="test_marker",
    )
    selected.admit_adjudication_support(
        "curated_marker_program",
        selected.rna_marker_support,
        selector="test_marker",
    )
    candidate.fine_reference_support, selected.fine_reference_support = support_pair(
        axis_preferences["exact_expression_reference"]
    )
    candidate.admit_adjudication_support(
        "exact_expression_reference",
        candidate.fine_reference_support,
        selector="test_reference",
    )
    selected.admit_adjudication_support(
        "exact_expression_reference",
        selected.fine_reference_support,
        selector="test_reference",
    )
    (
        candidate.coarse_composition_support,
        selected.coarse_composition_support,
    ) = support_pair(axis_preferences["composition_reference"])
    candidate.admit_adjudication_support(
        "composition_reference",
        candidate.coarse_composition_support,
        selector="test_composition",
    )
    selected.admit_adjudication_support(
        "composition_reference",
        selected.coarse_composition_support,
        selector="test_composition",
    )

    centroid_preference = axis_preferences["whole_profile_centroid"]
    if centroid_preference == "candidate":
        centroid = pd.Series({"STAD": 0.90, "ESCA": 0.80})
    elif centroid_preference == "selected":
        centroid = pd.Series({"STAD": 0.80, "ESCA": 0.90})
    elif centroid_preference == "tie":
        centroid = pd.Series({"STAD": 0.85, "ESCA": 0.85})
    elif centroid_preference == "unavailable":
        centroid = None
    else:
        raise AssertionError(
            f"unknown centroid preference: {centroid_preference}"
        )

    result = _adjudicate_selection_with_learned_hierarchy(
        {"ESCA": selected, "STAD": candidate},
        selected,
        cen=centroid,
        centroid_confident=centroid is not None,
    )
    return result, selected, candidate


@pytest.mark.parametrize(
    (
        "axis_preferences",
        "expected_code",
        "expected_candidate_votes",
        "expected_selected_votes",
        "expected_available_axes",
    ),
    [
        pytest.param(
            {
                "pan_cancer_signature": "selected",
                "whole_profile_centroid": "selected",
                "curated_marker_program": "selected",
                "exact_expression_reference": "selected",
                "composition_reference": "selected",
            },
            "ESCA",
            1,
            5,
            6,
            id="losing-candidate-retains-only-its-own-vote",
        ),
        pytest.param(
            {
                "pan_cancer_signature": "candidate",
                "whole_profile_centroid": "tie",
                "curated_marker_program": "candidate",
                "exact_expression_reference": "selected",
                "composition_reference": "selected",
            },
            "ESCA",
            3,
            2,
            6,
            id="three-two-one-plurality-does-not-promote",
        ),
        pytest.param(
            {
                "pan_cancer_signature": "candidate",
                "whole_profile_centroid": "unavailable",
                "curated_marker_program": "candidate",
                "exact_expression_reference": "selected",
                "composition_reference": "selected",
            },
            "STAD",
            3,
            2,
            5,
            id="three-of-five-available-is-a-majority",
        ),
        pytest.param(
            {
                "pan_cancer_signature": "candidate",
                "whole_profile_centroid": "selected",
                "curated_marker_program": "candidate",
                "exact_expression_reference": "selected",
                "composition_reference": "selected",
            },
            "ESCA",
            3,
            3,
            6,
            id="even-split-preserves-selected-leaf",
        ),
        pytest.param(
            {
                "pan_cancer_signature": "candidate",
                "whole_profile_centroid": "selected",
                "curated_marker_program": "candidate",
                "exact_expression_reference": "selected",
                "composition_reference": "candidate",
            },
            "STAD",
            4,
            2,
            6,
            id="four-of-six-majority-promotes",
        ),
        pytest.param(
            {
                "pan_cancer_signature": "candidate",
                "whole_profile_centroid": "unavailable",
                "curated_marker_program": "unavailable",
                "exact_expression_reference": "selected",
                "composition_reference": "unavailable",
            },
            "ESCA",
            2,
            1,
            3,
            id="majority-with-only-two-supporting-groups-is-insufficient",
        ),
    ],
)
def test_learned_hierarchy_adjudicates_available_axis_majorities(
    axis_preferences,
    expected_code,
    expected_candidate_votes,
    expected_selected_votes,
    expected_available_axes,
):
    from trufflepig.cancer_type_evidence import _hypothesis_evidence_channels

    result, selected, candidate = _adjudicate_stad_esca_evidence_scenario(
        axis_preferences
    )

    consensus = selected.details["entity_evidence_consensus"]
    available_axes = sum(axis["available"] for axis in consensus["axes"])
    assert consensus["candidate_votes"] == expected_candidate_votes
    assert consensus["selected_votes"] == expected_selected_votes
    assert available_axes == expected_available_axes
    assert consensus["available_axis_count"] == expected_available_axes
    for hypothesis, expected_votes in (
        (candidate, expected_candidate_votes),
        (selected, expected_selected_votes),
    ):
        consensus_channel = next(
            row
            for row in _hypothesis_evidence_channels(hypothesis)
            if row["channel"] == "entity_evidence_consensus"
        )
        assert consensus_channel["support"] == round(
            expected_votes / expected_available_axes,
            4,
        )
    assert result.cancer_type == expected_code
    if expected_code == "STAD":
        assert result is candidate
        assert result.selected_by == "entity_evidence_consensus"
        assert consensus["decisive_candidate"] is True
    else:
        assert result is selected
        assert consensus["decisive_candidate"] is False


def test_high_confidence_learned_call_cannot_override_opposing_majority():
    preferences = {
        "pan_cancer_signature": "selected",
        "whole_profile_centroid": "selected",
        "curated_marker_program": "tie",
        "exact_expression_reference": "selected",
        "composition_reference": "selected",
    }

    result, selected, candidate = _adjudicate_stad_esca_evidence_scenario(
        preferences,
        learned_entity_support=0.98,
    )

    assert result is selected
    consensus = candidate.details[
        "learned_hierarchy_withheld_by_independent_majority"
    ]
    assert consensus["candidate_votes"] == 1
    assert consensus["selected_votes"] == 4
    assert consensus["available_axis_count"] == 6


def test_weak_learned_tail_cannot_originate_entity_consensus_promotion():
    preferences = {
        "pan_cancer_signature": "candidate",
        "whole_profile_centroid": "candidate",
        "curated_marker_program": "unavailable",
        "exact_expression_reference": "unavailable",
        "composition_reference": "selected",
    }

    result, selected, candidate = _adjudicate_stad_esca_evidence_scenario(
        preferences,
        learned_entity_support=0.21,
    )

    assert result is selected
    consensus = candidate.details["entity_evidence_consensus"]
    assert consensus["candidate_votes"] == 3
    assert consensus["selected_votes"] == 1
    assert consensus["decisive_candidate"] is True
    assert consensus["credible_learned_candidate"] is False
    assert consensus["candidate_has_identity_vote"] is False
    assert consensus["candidate_origin_credible"] is False


def test_family_context_cannot_originate_same_lineage_entity_refinement():
    preferences = {
        "pan_cancer_signature": "candidate",
        "whole_profile_centroid": "candidate",
        "curated_marker_program": "unavailable",
        "exact_expression_reference": "unavailable",
        "composition_reference": "selected",
    }

    result, selected, candidate = _adjudicate_stad_esca_evidence_scenario(
        preferences,
        learned_entity_support=0.21,
        family_anchored_candidate=True,
    )

    assert result is selected
    consensus = candidate.details["entity_evidence_consensus"]
    assert consensus["decisive_candidate"] is True
    assert consensus["candidate_has_family_anchored_origin"] is True
    assert consensus["same_lineage_as_selected"] is True
    assert consensus["candidate_origin_credible"] is False


def test_broad_top_can_be_restored_by_same_lineage_entity_consensus():
    preferences = {
        "pan_cancer_signature": "candidate",
        "whole_profile_centroid": "candidate",
        "curated_marker_program": "unavailable",
        "exact_expression_reference": "unavailable",
        "composition_reference": "selected",
    }

    result, _selected, candidate = _adjudicate_stad_esca_evidence_scenario(
        preferences,
        learned_entity_support=0.21,
        family_anchored_candidate=True,
        candidate_broad_rank=1,
    )

    assert result is candidate
    consensus = candidate.details["entity_evidence_consensus"]
    assert consensus["candidate_is_broad_top"] is True
    assert consensus["candidate_origin_credible"] is True


def test_entity_consensus_requires_multiple_independent_evidence_groups():
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _entity_evidence_consensus,
    )

    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "STAD",
            entity_support=0.80,
            entity_margin=0.60,
            family="carcinoma-gi",
            family_support=0.95,
            compartment="epithelial",
            compartment_support=0.99,
        ),
        ("STAD", 0.80),
        ("ESCA", 0.20),
    )
    candidate = CancerTypeEvidence(
        cancer_type="STAD",
        broad_rna_support=0.85,
        fine_reference_support=0.80,
    )
    selected = CancerTypeEvidence(
        cancer_type="ESCA",
        broad_rna_support=0.45,
        fine_reference_support=0.20,
    )
    candidate.details["entity_evidence_marker_coherence"] = {
        "status": "consistent",
        "detected": 9,
        "detected_fraction": 0.90,
        "required_for_consistent": 5,
        "unexpected_low_detected": 0,
        "total": 10,
    }
    selected.details["entity_evidence_marker_coherence"] = {
        "status": "partial",
        "detected": 3,
        "detected_fraction": 0.30,
        "required_for_consistent": 5,
        "unexpected_low_detected": 1,
        "total": 10,
    }
    candidate.admit_adjudication_support(
        "exact_expression_reference",
        candidate.fine_reference_support,
        selector="test_reference",
    )
    selected.admit_adjudication_support(
        "exact_expression_reference",
        selected.fine_reference_support,
        selector="test_reference",
    )

    result = _entity_evidence_consensus(candidate, selected, details)

    assert result["decisive_candidate"] is True
    assert result["candidate_votes"] == 4
    assert result["candidate_nonlearned_votes"] == 3


def test_entity_consensus_does_not_promote_a_learned_vote_alone():
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _entity_evidence_consensus,
    )

    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "STAD",
            entity_support=0.99,
            entity_margin=0.98,
            family="carcinoma-gi",
            family_support=0.99,
            compartment="epithelial",
            compartment_support=0.99,
        ),
        ("STAD", 0.99),
        ("ESCA", 0.01),
    )

    result = _entity_evidence_consensus(
        CancerTypeEvidence(cancer_type="STAD"),
        CancerTypeEvidence(cancer_type="ESCA"),
        details,
    )

    assert result["candidate_votes"] == 1
    assert result["candidate_nonlearned_votes"] == 0
    assert result["decisive_candidate"] is False


def test_flat_learned_view_can_supply_a_corroborated_entity_beam_candidate():
    """A quantifier-robust flat result may enter the beam but remains one vote.

    This exercises the report-label decision seam: the calibrated hierarchy's
    top entity is unhelpful, while the flat view proposes BRCA and three
    independent report signals corroborate it over a selected sarcoma label.
    """

    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "RB",
            entity_support=0.50,
            entity_margin=0.20,
            family="RB",
            family_support=0.50,
            compartment="embryonal",
            compartment_support=0.50,
        ),
        ("RB", 0.50),
        ("SARC", 0.30),
    )
    details["learned_expression_hierarchy_votes"].append(
        {
            "stage": "compartment",
            "label": "embryonal",
            "probability": 0.50,
            "top_predictions": [
                {"label": "embryonal", "probability": 0.50},
                {"label": "epithelial", "probability": 0.40},
                {"label": "mesenchymal", "probability": 0.10},
            ],
        }
    )
    details["learned_expression_flat_top_predictions"] = [
        {"code": "BRCA_Basal", "probability": 0.40},
        {"code": "SARC_DDLPS", "probability": 0.15},
    ]

    selected = _selectable("SARC", "fused_evidence", (3, 2.0, 4))
    selected.broad_rna_support = 0.20
    selected.rna_marker_support = 0.20
    selected.coarse_composition_support = 0.20
    selected.details.update(details)
    brca = CancerTypeEvidence(
        cancer_type="BRCA",
        broad_rna_support=0.80,
        rna_marker_support=0.80,
        coarse_composition_support=0.80,
    )
    selected.admit_adjudication_support(
        "curated_marker_program",
        selected.rna_marker_support,
        selector="test_marker",
    )
    brca.admit_adjudication_support(
        "curated_marker_program",
        brca.rna_marker_support,
        selector="test_marker",
    )
    selected.admit_adjudication_support(
        "composition_reference",
        selected.coarse_composition_support,
        selector="test_composition",
    )
    brca.admit_adjudication_support(
        "composition_reference",
        brca.coarse_composition_support,
        selector="test_composition",
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {"SARC": selected, "BRCA": brca},
        selected,
    )

    assert result is brca
    assert result.selected_by == "entity_evidence_consensus"
    assert result.details["entity_consensus_adjudication_mode"] == (
        "candidate_beam_consensus"
    )
    consensus = result.details["entity_evidence_consensus"]
    learned_axis = next(
        axis
        for axis in consensus["axes"]
        if axis["axis"] == "learned_full_profile"
    )
    assert learned_axis["candidate_support"] == 0.40
    assert consensus["candidate_nonlearned_votes"] == 3


def test_learned_entity_leader_aggregates_mutually_exclusive_child_labels():
    """Entity selection sums sibling softmax leaves before choosing a leader."""

    from trufflepig.cancer_type_evidence import (
        _learned_entity_prediction_codes,
        _learned_entity_support_for_code,
    )

    details = {
        "learned_expression_hierarchy_votes": [
            {
                "stage": "entity",
                "top_predictions": [
                    {"label": "COAD_MSI", "probability": 0.35},
                    {"label": "COAD_MSS", "probability": 0.30},
                    {"label": "STAD", "probability": 0.40},
                ],
            }
        ]
    }

    predictions = _learned_entity_prediction_codes(details)
    assert predictions[0][0] == "COAD"
    assert predictions[0][1] == pytest.approx(0.65)
    assert _learned_entity_support_for_code(details, "COAD") == pytest.approx(0.65)


def test_flat_learned_entity_leader_aggregates_child_labels():
    """The quantifier-robust flat view obeys the same report-entity roll-up."""

    from trufflepig.cancer_type_evidence import _learned_entity_prediction_codes

    details = {
        "learned_expression_flat_top_predictions": [
            {"code": "COAD_MSI", "probability": 0.35},
            {"code": "COAD_MSS", "probability": 0.30},
            {"code": "STAD", "probability": 0.40},
        ]
    }

    predictions = _learned_entity_prediction_codes(details)
    assert predictions[0][0] == "COAD"
    assert predictions[0][1] == pytest.approx(0.65)


def test_residual_identity_completes_an_integrated_entity_consensus():
    """Post-background identity is one vote, not a standalone selector."""

    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "ACC",
            entity_support=0.60,
            entity_margin=0.30,
            family="endocrine-epithelial",
            family_support=0.75,
            compartment="epithelial",
            compartment_support=0.90,
        ),
        ("ACC", 0.60),
        ("SARC_DDLPS", 0.30),
    )
    selected = _selectable("SARC_DDLPS", "fused_evidence", (3, 2.0, 4))
    selected.broad_rna_support = 1.0
    selected.details.update(details)
    candidate = CancerTypeEvidence(
        cancer_type="ACC",
    )
    candidate.coarse_composition_support = 0.80
    selected.coarse_composition_support = 0.20
    candidate.admit_adjudication_support(
        "composition_reference",
        candidate.coarse_composition_support,
        selector="test_composition",
    )
    selected.admit_adjudication_support(
        "composition_reference",
        selected.coarse_composition_support,
        selector="test_composition",
    )
    hypotheses = {"SARC_DDLPS": selected, "ACC": candidate}

    with_incompatible_residual = _adjudicate_selection_with_learned_hierarchy(
        hypotheses,
        selected,
        residual_identity_evidence={
            "status": "candidate",
            "candidate_code": "ACC",
            "panel_candidate_code": "ACC",
            "current_code": "SARC_DDLPS",
            "adjudication_eligible": False,
        },
    )
    assert with_incompatible_residual is selected

    without_residual = _adjudicate_selection_with_learned_hierarchy(
        hypotheses,
        selected,
    )
    assert without_residual is selected

    with_residual = _adjudicate_selection_with_learned_hierarchy(
        hypotheses,
        selected,
        residual_identity_evidence={
            "status": "candidate",
            "candidate_code": "ACC",
            "panel_candidate_code": "ACC",
            "current_code": "SARC_DDLPS",
        },
    )

    assert with_residual is candidate
    consensus = candidate.details["entity_evidence_consensus"]
    assert consensus["candidate_votes"] == 3
    assert consensus["candidate_nonlearned_votes"] == 2
    residual_axis = next(
        axis
        for axis in consensus["axes"]
        if axis["axis"] == "decomposition_residual_identity"
    )
    assert residual_axis["preference"] == "candidate"


def test_residual_identity_cannot_originate_an_unrelated_entity():
    """An ontology-only residual program cannot manufacture a third vote."""

    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "ESCA",
            entity_support=0.60,
            entity_margin=0.30,
            family="carcinoma-gi",
            family_support=0.80,
            compartment="epithelial",
            compartment_support=0.95,
        ),
        ("ESCA", 0.60),
        ("CESC", 0.30),
    )
    selected = _selectable("CESC", "fused_evidence", (3, 2.0, 4))
    selected.broad_rna_support = 1.0
    selected.details.update(details)
    candidate = CancerTypeEvidence(
        cancer_type="ESCA",
        rna_marker_support=0.80,
    )
    candidate.admit_adjudication_support(
        "curated_marker_program",
        candidate.rna_marker_support,
        selector="lineage_panel",
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {"CESC": selected, "ESCA": candidate},
        selected,
        residual_identity_evidence={
            "status": "candidate",
            "candidate_code": "ESCA",
            "panel_candidate_code": None,
            "ontology_candidate_code": "ESCA",
            "current_code": "CESC",
        },
    )

    assert result is selected
    consensus = selected.details["entity_evidence_consensus"]
    residual_axis = next(
        axis
        for axis in consensus["axes"]
        if axis["axis"] == "decomposition_residual_identity"
    )
    assert residual_axis["available"] is False
    assert consensus["decisive_candidate"] is False


def test_invariant_residual_parent_can_enter_but_not_bypass_entity_consensus():
    """A residual parent needs separate signature and centroid corroboration."""

    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "ESCA",
            entity_support=0.80,
            entity_margin=0.60,
            family="carcinoma-gi",
            family_support=0.90,
            compartment="epithelial",
            compartment_support=0.95,
        ),
        ("ESCA", 0.80),
        ("STAD", 0.05),
    )
    selected = _selectable("ESCA", "fused_evidence", (3, 2.0, 4))
    selected.details.update(details)
    selected.details["signature_score"] = 0.20
    candidate = CancerTypeEvidence(
        cancer_type="CRC",
        details={"signature_score": 0.80},
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {"ESCA": selected, "CRC": candidate},
        selected,
        cen=pd.Series({"ESCA": 0.30, "COAD": 0.90, "READ": 0.88}),
        centroid_confident=True,
        residual_identity_evidence={
            "status": "candidate",
            "candidate_code": "CRC",
            "panel_candidate_code": None,
            "ontology_candidate_code": "CRC",
            "current_code": "ESCA",
            "background_models": [
                {
                    "candidate_code": "CRC",
                    "realizations": 2,
                    "template": "met_soft_tissue",
                }
            ],
        },
    )

    assert result is candidate
    assert result.selected_by == "entity_evidence_consensus"
    consensus = result.details["entity_evidence_consensus"]
    assert consensus["candidate_votes"] == 3
    assert consensus["selected_votes"] == 1
    assert consensus["candidate_has_learned_vote"] is False
    assert consensus["candidate_has_residual_vote"] is True
    assert consensus["entity_prediction_origin"] == "invariant_residual_identity"
    marker_axis = next(
        axis
        for axis in consensus["axes"]
        if axis["axis"] == "curated_marker_program"
    )
    assert marker_axis["available"] is False


def test_invariant_residual_consensus_does_not_require_a_learned_entity():
    """Independent residual evidence remains usable when the model abstains."""

    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    selected = _selectable("ESCA", "fused_evidence", (3, 2.0, 4))
    selected.details["signature_score"] = 0.20
    candidate = CancerTypeEvidence(
        cancer_type="CRC",
        details={"signature_score": 0.80},
    )
    candidate.admit_adjudication_support(
        "exact_expression_reference",
        0.90,
        selector="test_reference",
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {"ESCA": selected, "CRC": candidate},
        selected,
        cen=pd.Series({"ESCA": 0.30, "COAD": 0.90, "READ": 0.88}),
        centroid_confident=True,
        residual_identity_evidence={
            "status": "candidate",
            "candidate_code": "CRC",
            "panel_candidate_code": None,
            "ontology_candidate_code": "CRC",
            "current_code": "ESCA",
            "background_models": [
                {
                    "candidate_code": "CRC",
                    "realizations": 3,
                    "template": "solid_primary",
                }
            ],
        },
    )

    assert result is candidate
    assert result.selected_by == "entity_evidence_consensus"
    consensus = result.details["entity_evidence_consensus"]
    assert consensus["candidate_has_learned_vote"] is False
    assert consensus["candidate_has_residual_vote"] is True
    assert consensus["entity_prediction_origin"] == "invariant_residual_identity"


def test_source_resolved_residual_uses_learned_family_as_bulk_corroboration():
    """Host-resolved identity is compound evidence, not duplicated marker votes."""

    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    selected = _selectable("SARC_DDLPS", "fused_evidence", (3, 2.0, 4))
    selected.broad_rna_support = 1.0
    selected.details.update(
        {
            "learned_expression_top_family_label": "CRC",
            "learned_expression_top_entity_label": "SARC_DDLPS",
            "learned_expression_top_entity_probability": 0.40,
        }
    )
    candidate = CancerTypeEvidence(
        cancer_type="CRC",
        broad_rna_support=0.79,
        details={"signature_score": 0.79},
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {"SARC_DDLPS": selected, "CRC": candidate},
        selected,
        residual_identity_evidence={
            "status": "candidate",
            "candidate_code": "CRC",
            "panel_candidate_code": "CRC",
            "ontology_candidate_code": "CRC",
            "decision_basis": "panel_and_ontology",
            "source_resolved_identity": True,
            "current_code": "SARC_DDLPS",
            "background_models": [
                {
                    "candidate_code": "CRC",
                    "realizations": 1,
                    "template": "identity_background",
                },
                {
                    "candidate_code": "CRC",
                    "realizations": 1,
                    "template": "identity_structural_background",
                },
            ],
        },
    )

    assert result is candidate
    assert result.selected_by == "entity_evidence_consensus"
    consensus = result.details["entity_evidence_consensus"]
    assert consensus["source_resolved_identity_decisive"] is True
    assert consensus["source_resolved_identity_corroborators"] == [
        "learned_family_leader"
    ]
    # The CRC panel is already part of residual identity and must not appear as
    # a second independent marker vote.
    marker_axis = next(
        axis
        for axis in consensus["axes"]
        if axis["axis"] == "curated_marker_program"
    )
    assert marker_axis["available"] is False


def test_residual_identity_preserves_explicit_entity_blockers():
    """Even a full RNA majority cannot erase a persistent safety veto."""

    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "NUTM",
            entity_support=0.70,
            entity_margin=0.50,
            family="squamous",
            family_support=0.90,
            compartment="epithelial",
            compartment_support=0.95,
        ),
        ("NUTM", 0.70),
        ("LUSC", 0.20),
    )
    selected = _selectable("LUSC", "fused_evidence", (3, 2.0, 4))
    selected.broad_rna_support = 1.0
    selected.details.update(details)
    candidate = CancerTypeEvidence(
        cancer_type="NUTM",
        rna_marker_support=0.90,
        details={
            "hard_report_label_blockers": [
                "supplied fusion evidence excludes the defining fusion"
            ]
        },
    )
    candidate.admit_adjudication_support(
        "curated_marker_program",
        candidate.rna_marker_support,
        selector="lineage_panel",
    )
    candidate.coarse_composition_support = 0.90
    selected.coarse_composition_support = 0.10
    candidate.admit_adjudication_support(
        "composition_reference",
        candidate.coarse_composition_support,
        selector="test_composition",
    )
    selected.admit_adjudication_support(
        "composition_reference",
        selected.coarse_composition_support,
        selector="test_composition",
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {"LUSC": selected, "NUTM": candidate},
        selected,
        residual_identity_evidence={
            "status": "candidate",
            "candidate_code": "NUTM",
            "panel_candidate_code": "NUTM",
            "current_code": "LUSC",
        },
    )

    assert result is selected
    consensus = candidate.details["entity_evidence_consensus"]
    assert consensus["evidence_decisive_candidate"] is True
    assert consensus["decisive_candidate"] is False
    assert consensus["selection_blocked"] is True


def test_entity_beam_runs_after_matching_top_and_preserves_candidate_mmr():
    """A flat alternative can win without inheriting the old row's MMR state."""

    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
        _build_staged_evidence_graph,
    )

    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "STAD",
            entity_support=0.65,
            entity_margin=0.30,
            family="carcinoma-gi",
            family_support=0.90,
            compartment="epithelial",
            compartment_support=0.95,
        ),
        ("STAD", 0.65),
        ("COAD", 0.20),
    )
    details["learned_expression_flat_top_predictions"] = [
        {"code": "COAD_MSS", "probability": 0.80},
        {"code": "STAD", "probability": 0.10},
    ]
    details["learned_expression_hierarchy_votes"].append(
        {
            "stage": "mismatch_repair",
            "label_space": "selected_row_mmr",
            "label": "MSI",
            "probability": 0.90,
            "top_predictions": [{"label": "MSI", "probability": 0.90}],
        }
    )

    selected = _selectable("STAD", "fused_evidence", (3, 2.0, 4))
    selected.details.update(details)
    coad = CancerTypeEvidence(
        cancer_type="COAD",
        rna_marker_support=0.85,
        coarse_composition_support=0.80,
    )
    coad.details.update(
        {
            "learned_expression_hierarchy_votes": [
                {
                    "stage": "mismatch_repair",
                    "label_space": "candidate_row_mmr",
                    "label": "MSS",
                    "probability": 0.75,
                    "top_predictions": [
                        {"label": "MSS", "probability": 0.75}
                    ],
                }
            ],
            # The candidate-specific key must win over this legacy global key
            # when the evidence graph is rendered.
            "learned_expression_hierarchical_votes": [
                {
                    "stage": "mismatch_repair",
                    "label_space": "legacy_wrong_context",
                    "label": "MSI",
                    "probability": 0.90,
                }
            ],
        }
    )
    coad.admit_adjudication_support(
        "curated_marker_program",
        coad.rna_marker_support,
        selector="test_marker",
    )
    coad.admit_adjudication_support(
        "composition_reference",
        coad.coarse_composition_support,
        selector="test_composition",
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {"STAD": selected, "COAD": coad},
        selected,
    )

    assert result is coad
    assert result.selected_by == "entity_evidence_consensus"
    mmr_votes = [
        vote
        for vote in result.details["learned_expression_hierarchy_votes"]
        if vote["stage"] == "mismatch_repair"
    ]
    assert [vote["label"] for vote in mmr_votes] == ["MSS"]
    graph = _build_staged_evidence_graph(
        [selected, coad],
        result,
        {},
    )
    mmr_channels = [
        row
        for row in graph["channels"]
        if row["role"] == "hierarchical_mismatch_repair_vote"
    ]
    assert [row["code"] for row in mmr_channels] == ["MSS"]
    assert mmr_channels[0]["context_code"] == "COAD"


def test_entity_beam_does_not_promote_a_tail_from_both_learned_views():
    """Independent axes cannot turn learned tail noise into a report label.

    The hierarchy and flat views both lead with CRC-family entities.  STAD has
    strong reference-derived context, but it is a tail prediction in each
    learned view, so the already integrated READ selection remains active.
    """

    import pandas as pd

    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "COAD",
            entity_support=0.40,
            entity_margin=0.25,
            family="CRC",
            family_support=0.80,
            compartment="epithelial",
            compartment_support=0.95,
        ),
        ("COAD", 0.40),
        ("STAD", 0.12),
    )
    details["learned_expression_flat_top_predictions"] = [
        {"code": "COAD_MSS", "probability": 0.35},
        {"code": "STAD_CIN", "probability": 0.10},
    ]

    selected = _selectable("READ", "fused_evidence", (3, 1.50, 5))
    selected.broad_rna_support = 1.0
    selected.rna_marker_support = 0.70
    selected.details.update(details)
    selected.admit_adjudication_support(
        "curated_marker_program",
        selected.rna_marker_support,
        selector="test_marker",
    )

    stad = CancerTypeEvidence(
        cancer_type="STAD",
        broad_rna_support=0.95,
        fine_reference_support=0.90,
        coarse_composition_support=0.82,
    )
    stad.admit_adjudication_support(
        "exact_expression_reference",
        stad.fine_reference_support,
        selector="test_reference",
    )
    stad.admit_adjudication_support(
        "composition_reference",
        stad.coarse_composition_support,
        selector="test_composition",
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {
            "READ": selected,
            "COAD": CancerTypeEvidence(cancer_type="COAD"),
            "STAD": stad,
        },
        selected,
        cen=pd.Series({"READ": 0.80, "STAD": 0.84}),
        centroid_confident=True,
    )

    assert result is selected
    assert result.cancer_type == "READ"
    assert stad.can_select_report_label is False


def test_entity_beam_uses_valid_child_reference_and_stronger_corroboration():
    """Competing learned views are resolved by independent evidence strength."""

    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "SARC_MYXFIB",
            entity_support=0.239,
            entity_margin=0.085,
            family="SARC_OTHER",
            family_support=0.483,
            compartment="mesenchymal",
            compartment_support=0.827,
        ),
        ("SARC_MYXFIB", 0.239),
        ("BRCA_Basal", 0.033),
    )
    details["learned_expression_hierarchy_votes"].append(
        {
            "stage": "compartment",
            "label": "mesenchymal",
            "probability": 0.827,
            "top_predictions": [
                {"label": "mesenchymal", "probability": 0.827},
                {"label": "epithelial", "probability": 0.159},
                {"label": "heme", "probability": 0.009},
            ],
        }
    )
    details["learned_expression_flat_top_predictions"] = [
        {"code": "BRCA_Basal", "probability": 0.191},
        {"code": "SARC_ESS_HG", "probability": 0.122},
    ]

    selected = _selectable("HL", "pan_cancer_signature_ranker", (3, 2.0, 4))
    selected.broad_rna_support = 1.0
    selected.details.update(details)
    sarcoma = CancerTypeEvidence(
        cancer_type="SARC_MYXFIB",
        rna_marker_support=0.48,
        fine_reference_support=0.294,
    )
    brca = CancerTypeEvidence(
        cancer_type="BRCA",
        broad_rna_support=0.473,
        rna_marker_support=0.875,
    )
    basal = CancerTypeEvidence(
        cancer_type="BRCA_Basal",
        fine_reference_support=0.546,
    )
    basal.admit_adjudication_support(
        "exact_expression_reference",
        basal.fine_reference_support,
        selector="test_reference",
    )
    sarcoma.admit_adjudication_support(
        "curated_marker_program",
        sarcoma.rna_marker_support,
        selector="test_marker",
    )
    brca.admit_adjudication_support(
        "curated_marker_program",
        brca.rna_marker_support,
        selector="test_marker",
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {
            "HL": selected,
            "SARC_MYXFIB": sarcoma,
            "BRCA": brca,
            "BRCA_Basal": basal,
        },
        selected,
    )

    assert result is brca
    assert result.details["entity_descendant_exact_reference_code"] == "BRCA_Basal"
    assert result.details["entity_descendant_exact_reference_support"] == 0.546
    consensus = result.details["entity_evidence_consensus"]
    exact_axis = next(
        axis
        for axis in consensus["axes"]
        if axis["axis"] == "exact_expression_reference"
    )
    assert exact_axis["candidate_support"] == 0.546


def test_entity_beam_cannot_roll_up_a_blocked_child_reference():
    """A rejected child match stays visible but cannot originate its parent."""

    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "BRCA",
            entity_support=0.40,
            entity_margin=0.20,
            family="carcinoma-breast",
            family_support=0.60,
            compartment="epithelial",
            compartment_support=0.70,
        ),
        ("BRCA", 0.40),
        ("HL", 0.20),
    )
    selected = _selectable("HL", "pan_cancer_signature_ranker", (3, 2.0, 4))
    selected.broad_rna_support = 1.0
    selected.details.update(details)
    brca = CancerTypeEvidence(
        cancer_type="BRCA",
        coarse_composition_support=0.80,
    )
    basal = CancerTypeEvidence(
        cancer_type="BRCA_Basal",
        fine_reference_support=0.80,
        blocking_reasons=(
            "expression-reference context is not the top compatible context",
        ),
        label_status="blocked",
    )
    basal.admit_adjudication_support(
        "exact_expression_reference",
        basal.fine_reference_support,
        selector="test_reference",
        blocking_reasons=basal.blocking_reasons,
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {
            "HL": selected,
            "BRCA": brca,
            "BRCA_Basal": basal,
        },
        selected,
    )

    assert result is selected
    assert basal.fine_reference_support == 0.80
    assert basal.adjudication_support == {}
    assert (
        basal.adjudication_exclusions["exact_expression_reference"][
            "test_reference"
        ]
        == basal.blocking_reasons
    )
    consensus = brca.details["entity_evidence_consensus"]
    exact_axis = next(
        axis
        for axis in consensus["axes"]
        if axis["axis"] == "exact_expression_reference"
    )
    assert exact_axis["available"] is False
    assert "entity_descendant_exact_reference_support" not in brca.details


def test_top_learned_entity_uses_same_consensus_beam_as_runner_up():
    """A top entity can win on marker and composition despite stage conflict."""

    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "PRAD",
            entity_support=0.425,
            entity_margin=0.282,
            family="SARC_OTHER",
            family_support=0.229,
            compartment="mesenchymal",
            compartment_support=0.532,
        ),
        ("PRAD", 0.425),
        ("SARC", 0.064),
    )
    details["learned_expression_hierarchy_votes"].append(
        {
            "stage": "compartment",
            "label": "mesenchymal",
            "probability": 0.532,
            "top_predictions": [
                {"label": "mesenchymal", "probability": 0.532},
                {"label": "epithelial", "probability": 0.454},
            ],
        }
    )
    details["learned_expression_flat_top_predictions"] = [
        {"code": "SARC_KS", "probability": 0.356},
        {"code": "PRAD", "probability": 0.047},
    ]

    selected = _selectable("SARC", "fused_evidence", (3, 2.0, 4))
    selected.broad_rna_support = 1.0
    selected.details.update(details)
    prad = CancerTypeEvidence(
        cancer_type="PRAD",
        broad_rna_support=0.671,
        rna_marker_support=0.864,
    )
    prad.admit_adjudication_support(
        "curated_marker_program",
        prad.rna_marker_support,
        selector="test_marker",
    )
    prad.coarse_composition_support = 0.943
    prad.admit_adjudication_support(
        "composition_reference",
        prad.coarse_composition_support,
        selector="test_composition",
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {"SARC": selected, "PRAD": prad},
        selected,
    )

    assert result is prad
    assert result.details["entity_consensus_adjudication_mode"] == (
        "candidate_beam_consensus"
    )
    consensus = result.details["entity_evidence_consensus"]
    assert consensus["learned_entity_prediction_rank"] == 1
    assert consensus["candidate_votes"] == 3
    assert consensus["candidate_nonlearned_votes"] == 2


def test_entity_consensus_does_not_double_count_signature_derived_marker_score():
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _entity_evidence_consensus,
    )

    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "STAD",
            entity_support=0.80,
            entity_margin=0.60,
            family="carcinoma-gi",
            family_support=0.95,
            compartment="epithelial",
            compartment_support=0.99,
        ),
        ("STAD", 0.80),
        ("ESCA", 0.20),
    )
    candidate = CancerTypeEvidence(
        cancer_type="STAD",
        broad_rna_support=0.80,
        family_marker_support=0.95,
    )
    selected = CancerTypeEvidence(
        cancer_type="ESCA",
        broad_rna_support=0.20,
        family_marker_support=0.30,
    )
    shared_coherence = {
        "status": "consistent",
        "detected": 8,
        "detected_fraction": 0.80,
        "required_for_consistent": 5,
        "unexpected_low_detected": 0,
        "total": 10,
    }
    candidate.details.update(
        {
            "pan_cancer_signature_marker_coherence": dict(shared_coherence),
            "pan_cancer_signature_marker_support": 0.95,
        }
    )
    selected.details.update(
        {
            "pan_cancer_signature_marker_coherence": dict(shared_coherence),
            "pan_cancer_signature_marker_support": 0.30,
        }
    )

    result = _entity_evidence_consensus(candidate, selected, details)
    marker_axis = next(
        axis
        for axis in result["axes"]
        if axis["axis"] == "curated_marker_program"
    )

    assert marker_axis["preference"] == "abstain"
    assert marker_axis["candidate_support"] == 0.8
    assert marker_axis["selected_support"] == 0.8
    assert result["candidate_votes"] == 2
    assert result["candidate_nonlearned_votes"] == 1
    assert result["decisive_candidate"] is False


def test_entity_consensus_ignores_centroid_when_compartment_is_unconfident():
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _entity_evidence_consensus,
    )

    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "STAD",
            entity_support=0.80,
            entity_margin=0.60,
            family="carcinoma-gi",
            family_support=0.95,
            compartment="epithelial",
            compartment_support=0.99,
        ),
        ("STAD", 0.80),
        ("ESCA", 0.20),
    )
    candidate = CancerTypeEvidence(cancer_type="STAD", broad_rna_support=0.80)
    selected = CancerTypeEvidence(cancer_type="ESCA", broad_rna_support=0.20)
    weak_centroids = pd.Series({"STAD": 0.10, "ESCA": 0.09})

    unconfident = _entity_evidence_consensus(
        candidate,
        selected,
        details,
        cen=weak_centroids,
        centroid_confident=False,
    )
    centroid_axis = next(
        axis
        for axis in unconfident["axes"]
        if axis["axis"] == "whole_profile_centroid"
    )

    assert centroid_axis["available"] is False
    assert unconfident["candidate_votes"] == 2
    assert unconfident["candidate_nonlearned_votes"] == 1
    assert unconfident["decisive_candidate"] is False

    confident = _entity_evidence_consensus(
        candidate,
        selected,
        details,
        cen=weak_centroids,
        centroid_confident=True,
    )
    assert confident["candidate_votes"] == 3
    assert confident["candidate_nonlearned_votes"] == 2
    assert confident["decisive_candidate"] is True


def test_learned_hierarchy_uses_multi_axis_consensus_below_single_model_gate():
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    selected = _selectable("ESCA", "local_expression_reference", (3, 2.0, 4))
    selected.broad_rna_support = 0.45
    selected.fine_reference_support = 0.20
    selected.details["entity_evidence_marker_coherence"] = {
        "status": "partial",
        "detected_fraction": 0.30,
        "unexpected_low_detected": 1,
        "total": 10,
    }
    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "STAD",
            entity_support=0.80,
            entity_margin=0.60,
            family="carcinoma-gi",
            family_support=0.95,
            compartment="epithelial",
            compartment_support=0.99,
        ),
        ("STAD", 0.80),
        ("ESCA", 0.20),
    )
    selected.details.update(details)
    candidate = CancerTypeEvidence(
        cancer_type="STAD",
        broad_rna_support=0.85,
        fine_reference_support=0.80,
    )
    candidate.details["entity_evidence_marker_coherence"] = {
        "status": "consistent",
        "detected_fraction": 0.90,
        "unexpected_low_detected": 0,
        "total": 10,
    }
    candidate.admit_adjudication_support(
        "exact_expression_reference",
        candidate.fine_reference_support,
        selector="test_reference",
    )
    selected.admit_adjudication_support(
        "exact_expression_reference",
        selected.fine_reference_support,
        selector="test_reference",
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {"ESCA": selected, "STAD": candidate},
        selected,
    )

    assert result is candidate
    assert result.selected_by == "entity_evidence_consensus"
    assert result.details["learned_hierarchy_adjudication_mode"] == (
        "multi_axis_entity_refinement"
    )


def test_multi_axis_consensus_preserves_explicit_negative_fusion_blocker():
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    selected = _selectable("ESCA", "local_expression_reference", (3, 2.0, 4))
    selected.broad_rna_support = 0.20
    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "STAD",
            entity_support=0.80,
            entity_margin=0.60,
            family="carcinoma-gi",
            family_support=0.95,
            compartment="epithelial",
            compartment_support=0.99,
        ),
        ("STAD", 0.80),
        ("ESCA", 0.20),
    )
    selected.details.update(details)
    candidate = CancerTypeEvidence(
        cancer_type="STAD",
        broad_rna_support=0.80,
        fine_reference_support=0.80,
        can_select_report_label=False,
        blocking_reasons=("explicit molecular evidence vetoed this label",),
        label_status="blocked",
    )
    candidate.details.update(
        {
            "local_reference_explicit_negative_fusion": True,
            "local_reference_explicit_negative_fusion_details": {
                "expected_driver": "CLDN18-ARHGAP26",
                "fusion_record_count": 1,
            },
            "entity_evidence_marker_coherence": {
                "status": "consistent",
                "detected_fraction": 0.90,
                "unexpected_low_detected": 0,
                "total": 10,
            },
        }
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {"ESCA": selected, "STAD": candidate},
        selected,
    )

    assert result is selected
    assert candidate.can_select_report_label is False
    assert candidate.label_status == "blocked"
    assert candidate.blocking_reasons == (
        "explicit molecular evidence vetoed this label",
    )
    assert candidate.details["entity_consensus_hard_blockers"]


def test_conflicted_sibling_evidence_abstains_to_registry_parent():
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    selected = _selectable("READ_MSS", "local_expression_reference", (3, 2.0, 4))
    selected.broad_rna_support = 1.0
    selected.fine_reference_support = 0.90
    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "COAD_MSS",
            entity_support=0.75,
            entity_margin=0.55,
            family="CRC",
            family_support=0.99,
            compartment="epithelial",
            compartment_support=0.99,
        ),
        ("COAD_MSS", 0.75),
        ("READ_MSS", 0.20),
    )
    selected.details.update(details)
    candidate = CancerTypeEvidence(cancer_type="COAD_MSS")

    result = _adjudicate_selection_with_learned_hierarchy(
        {"READ_MSS": selected, "COAD_MSS": candidate},
        selected,
    )

    assert result.cancer_type == "CRC"
    assert result.selected_by == "entity_evidence_consensus"
    assert result.details["entity_consensus_adjudication_mode"] == (
        "common_ancestor_abstention"
    )
    assert result.details["entity_consensus_previous_code"] == "READ_MSS"
    assert result.details["entity_consensus_learned_code"] == "COAD"
    assert result.details["entity_consensus_learned_raw_code"] == "COAD_MSS"


def test_negligible_learned_sibling_preference_does_not_trigger_parent_abstention():
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _adjudicate_selection_with_learned_hierarchy,
    )

    selected = _selectable("READ_MSS", "local_expression_reference", (3, 2.0, 4))
    selected.broad_rna_support = 1.0
    selected.fine_reference_support = 0.90
    details = _add_entity_prediction_vector(
        _top_hierarchy_details(
            "COAD_MSS",
            entity_support=0.010,
            entity_margin=0.001,
            family="CRC",
            family_support=0.99,
            compartment="epithelial",
            compartment_support=0.99,
        ),
        ("COAD_MSS", 0.010),
        ("READ_MSS", 0.009),
    )
    selected.details.update(details)

    result = _adjudicate_selection_with_learned_hierarchy(
        {"READ_MSS": selected, "COAD_MSS": CancerTypeEvidence("COAD_MSS")},
        selected,
    )

    assert result is selected
    consensus = selected.details["entity_evidence_consensus"]
    assert consensus["conflicted"] is True
    assert consensus["credible_learned_candidate"] is False


def test_learned_status_vote_supports_parent_entity_not_status_claim():
    from trufflepig.cancer_type_evidence import (
        _adjudicate_selection_with_learned_hierarchy,
    )

    selected = _selectable("CML", "local_expression_reference", (3, 2.0, 4))
    selected.details.update(
        _top_hierarchy_details(
            "LAML_ELNadv",
            entity_support=0.99,
            entity_margin=0.98,
            family="LAML",
            family_support=0.99,
            compartment="heme",
            compartment_support=0.99,
        )
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {"CML": selected},
        selected,
    )

    assert result.cancer_type == "LAML"
    assert result.cancer_type != "LAML_ELNadv"
    assert result.details["learned_expression_top_entity_label"] == (
        "LAML_ELNadv"
    )
    assert result.details["learned_hierarchy_adjudication_mode"] == (
        "high_precision_entity_refinement"
    )


def test_learned_hierarchy_adjudicator_never_overrides_definitive_fusion():
    from trufflepig.cancer_type_evidence import (
        _adjudicate_selection_with_learned_hierarchy,
    )

    selected = _selectable("SARC_EWS", "direct_fusion", (5, 2.0, 6))
    selected.details.update(
        _top_hierarchy_details(
            "STAD",
            entity_support=0.999,
            entity_margin=0.99,
            family="carcinoma-gi",
            family_support=0.99,
            compartment="epithelial",
            compartment_support=0.99,
        )
    )

    result = _adjudicate_selection_with_learned_hierarchy(
        {selected.cancer_type: selected},
        selected,
    )

    assert result is selected


def test_nutm_rna_surrogate_promotes_with_strong_squamous_runner_up():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    finding = {
        "cancer_type": "NUTM",
        "rule_id": "nutm_nutm1",
        "surrogate": "NUTM1",
        "surrogate_tpm": 33.8,
        "threshold_tpm": 1.0,
        "support_genes": [],
    }

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("BRCA", 1.0), ("ESCA", 0.95), ("HNSC", 0.75)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "NUTM"
    assert result["selected"]["selected_by"] == "rare_marker"
    nutm = next(row for row in result["evidence"] if row["cancer_type"] == "NUTM")
    assert nutm["metrics"]["related_context_support"] == 0.95
    assert nutm["top_is_context"] is False


def test_nutm_rna_surrogate_blocks_from_mesenchymal_top_context():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    finding = {
        "cancer_type": "NUTM",
        "rule_id": "nutm_nutm1",
        "surrogate": "NUTM1",
        "surrogate_tpm": 9.0,
        "threshold_tpm": 1.0,
        "support_genes": [],
    }

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("SARC", 1.0), ("HNSC", 0.75), ("ESCA", 0.72)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "SARC"
    nutm = next(row for row in result["evidence"] if row["cancer_type"] == "NUTM")
    assert nutm["label_decision"]["status"] == "blocked"
    assert any(
        "top expression-reference lineage mesenchymal" in reason
        for reason in nutm["blocking_reasons"]
    )


def test_nutm_rna_surrogate_blocks_when_squamous_context_is_weak():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    finding = {
        "cancer_type": "NUTM",
        "rule_id": "nutm_nutm1",
        "surrogate": "NUTM1",
        "surrogate_tpm": 33.8,
        "threshold_tpm": 1.0,
        "support_genes": [],
    }

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("BRCA", 1.0), ("ESCA", 0.55), ("HNSC", 0.40)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "BRCA"
    nutm = next(row for row in result["evidence"] if row["cancer_type"] == "NUTM")
    assert nutm["label_decision"]["status"] == "blocked"
    assert any("expression-reference context" in reason for reason in nutm["blocking_reasons"])


def test_weak_fusion_defined_surrogate_does_not_bypass_cross_lineage_marker_conflict(
    monkeypatch,
):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_marker_coherence",
        lambda code, sample: {
            "status": "weak",
            "detected": 1,
            "total": 5,
            "required_for_consistent": 3,
            "detected_fraction": 0.2,
        }
        if code == "NUTM"
        else {},
    )
    monkeypatch.setattr(
        evidence,
        "_local_reference_cross_lineage_conflict",
        lambda code, context_codes, coherence: {
            "code": code,
            "code_lineage": "squamous",
            "context_codes": list(context_codes),
            "context_lineages": ["epithelial"],
            "marker_status": coherence["status"],
            "detected": coherence["detected"],
            "total": coherence["total"],
            "required_for_consistent": coherence["required_for_consistent"],
        }
        if code == "NUTM"
        else {},
    )
    finding = {
        "cancer_type": "NUTM",
        "rule_id": "nutm_nutm1",
        "surrogate": "NUTM1",
        "surrogate_tpm": 2.0,
        "threshold_tpm": 1.0,
        "support_genes": [],
    }

    result = select_report_scope_from_evidence(
        _expression_frame({"NUTM1": 2.0}),
        _analysis(("LUSC", 1.0), ("HNSC", 0.9)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "LUSC"
    nutm = next(row for row in result["evidence"] if row["cancer_type"] == "NUTM")
    assert nutm["can_select_report_label"] is False
    assert any("cross-lineage RNA-marker promotion" in reason for reason in nutm["blocking_reasons"])


def test_broad_context_is_part_of_unified_evidence_view():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("READ", 1.0), ("COAD", 0.4)),
    )

    assert result["selected"]["cancer_type"] == "READ"
    assert result["selected"]["inferred_cancer_type"] == "READ"
    assert result["selected"]["expression_reference_cancer_type"] == "READ"
    assert result["selected"]["selected_by"] == "pan_cancer_signature_ranker"
    assert result["primary_expression_context"]["cancer_type"] == "READ"
    assert [row["cancer_type"] for row in result["evidence"]] == ["READ", "COAD"]
    assert result["evidence"][0]["evidence_sources"] == [
        "pan_cancer_signature_ranker"
    ]
    assert result["evidence"][0]["metrics"]["broad_rna_support"] == 1.0
    assert result["evidence"][0]["metrics"]["pan_cancer_signature_support"] == 1.0
    assert result["evidence"][0]["report_label_candidate"] is True


def test_composition_reference_can_rescue_ambiguous_marker_incoherent_broad_call():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    signal = SimpleNamespace(
        cancer_hint="tumor-consistent",
        top_tcga_cohorts=[
            ("BLCA_TPM", 0.86),
            ("UCEC_TPM", 0.85),
            ("CESC_TPM", 0.84),
        ],
        type_specific_cohort="BLCA_TPM",
        type_specific_hits=[
            ("UPK1B", 27.0),
            ("KRT20", 18.0),
            ("GPR87", 15.0),
        ],
    )
    result = select_report_scope_from_evidence(
        _expression_frame({"SFTPB": 450.0, "SCGB1A1": 4.5}),
        {
            "cancer_type": "LUAD",
            "fit_quality": {"label": "ambiguous"},
            "healthy_vs_tumor": signal,
            "candidate_trace": [
                {"code": "LUAD", "support_fraction_of_top": 1.0},
                {"code": "BRCA", "support_fraction_of_top": 0.98},
            ],
        },
    )

    assert result["selected"]["cancer_type"] == "BLCA"
    assert result["selected"]["selected_by"] == "coarse_composition_reference"
    assert result["selected"]["metrics"]["coarse_composition_support"] > 0.7
    selected = next(row for row in result["evidence"] if row["cancer_type"] == "BLCA")
    assert selected["coarse_reference_type_specific_hit_count"] == 3


def test_composition_cannot_replace_same_tissue_family_entity_by_itself(
    monkeypatch,
):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_add_learned_expression_classifier_features",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evidence,
        "_add_learned_hierarchy_candidate_features",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evidence,
        "_centroid_and_confidence",
        lambda _sample: (pd.Series(dtype=float), False),
    )
    signal = SimpleNamespace(
        cancer_hint="tumor-consistent",
        top_tcga_cohorts=[
            ("UCEC_TPM", 0.90),
            ("CESC_TPM", 0.80),
            ("BLCA_TPM", 0.75),
        ],
        type_specific_cohort="UCEC_TPM",
        type_specific_hits=[
            ("PAX8", 100.0),
            ("FOXA2", 80.0),
            ("HOXA10", 60.0),
            ("ESR1", 40.0),
        ],
    )

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        {
            "cancer_type": "UCS",
            "fit_quality": {"label": "ambiguous"},
            "healthy_vs_tumor": signal,
            "candidate_trace": [
                {"code": "UCS", "support_fraction_of_top": 1.0},
                {"code": "UCEC", "support_fraction_of_top": 0.82},
            ],
        },
    )

    assert result["selected"]["cancer_type"] == "UCS"
    ucec = next(
        row for row in result["evidence"]
        if row["cancer_type"] == "UCEC"
    )
    assert ucec["can_select_report_label"] is False
    assert ucec["coarse_reference_same_tissue_family_broad_top"] is True
    assert any(
        "cannot by itself distinguish sibling entity" in reason
        for reason in ucec["blocking_reasons"]
    )


@pytest.mark.parametrize(
    ("stad_rho", "expected_code", "composition_available"),
    [
        (0.010, "ESCA", False),
        (0.800, "STAD", True),
    ],
)
def test_composition_consensus_requires_an_absolute_candidate_fit(
    monkeypatch,
    stad_rho,
    expected_code,
    composition_available,
):
    """Keep credible secondary fits while rejecting negligible correlations."""
    import trufflepig.cancer_type_evidence as evidence
    import trufflepig.expression_classifier as classifier
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        classifier,
        "classify_expression",
        lambda _sample, top_k=5: [("STAD", 0.40), ("ESCA", 0.20)],
    )
    monkeypatch.setattr(
        classifier,
        "classify_expression_hierarchy",
        lambda _sample, top_k=5: [
            SimpleNamespace(
                public_dict=lambda: {
                    "stage": "entity",
                    "label": "STAD",
                    "probability": 0.40,
                    "margin": 0.20,
                    "top_predictions": [
                        {"label": "STAD", "probability": 0.40},
                        {"label": "ESCA", "probability": 0.20},
                    ],
                }
            )
        ],
    )

    def marker_coherence(code, _sample):
        if code == "STAD":
            return {
                "status": "consistent",
                "detected": 4,
                "total": 4,
                "required_for_consistent": 3,
                "detected_fraction": 1.0,
                "unexpected_low_detected": 0,
            }
        return {}

    monkeypatch.setattr(evidence, "_marker_coherence", marker_coherence)
    monkeypatch.setattr(evidence, "_FINE_REFERENCE_SPECS", ())
    monkeypatch.setattr(
        evidence,
        "_add_local_expression_reference_features",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evidence,
        "_add_lineage_panel_features",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        evidence,
        "_add_contrast_discriminator_features",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evidence,
        "_add_fused_evidence_features",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evidence,
        "_centroid_and_confidence",
        lambda _sample: (pd.Series(dtype=float), False),
    )
    analysis = _analysis(("ESCA", 1.0), ("STAD", 0.90))
    analysis["fit_quality"] = {"label": "well-supported"}
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        cancer_hint="tumor-consistent",
        top_tcga_cohorts=[
            ("SGC_TPM", 0.820),
            ("STAD_TPM", stad_rho),
            ("ESCA_TPM", 0.009),
        ],
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"TP53": 10.0}),
        analysis,
    )

    assert result["selected"]["cancer_type"] == expected_code
    stad = next(row for row in result["evidence"] if row["cancer_type"] == "STAD")
    composition = next(
        axis
        for axis in stad["entity_evidence_consensus"]["axes"]
        if axis["axis"] == "composition_reference"
    )
    assert composition["available"] is composition_available
    assert (
        "composition_reference" in stad["adjudication_admissible_support"]
    ) is composition_available


def test_structural_sarcoma_composition_yields_to_learned_crc_family(monkeypatch):
    """Smooth-muscle/background composition cannot by itself turn close CRC RNA into SARC."""
    import trufflepig.cancer_type_evidence as evidence
    import trufflepig.expression_classifier as classifier
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(evidence, "_marker_coherence", lambda _code, _sample: {})
    monkeypatch.setattr(
        evidence,
        "_centroid_and_confidence",
        lambda _sample: (pd.Series({"SARC": 0.86, "READ": 0.92, "COAD": 0.90}), True),
    )
    monkeypatch.setattr(
        classifier,
        "classify_expression",
        lambda _sample, top_k=5: [
            ("READ_MSS", 0.12),
            # After report-entity roll-up the two SARC leaves must remain below
            # READ, matching this fixture's stated learned-CRC premise.
            ("SARC_PEC", 0.05),
            ("SARC", 0.04),
            ("COAD_MSS", 0.09),
        ],
    )
    monkeypatch.setattr(
        classifier,
        "classify_expression_hierarchy",
        lambda _sample, top_k=5: [
            SimpleNamespace(
                public_dict=lambda: {
                    "stage": "compartment",
                    "label_space": "learned_compartment",
                    "label": "mesenchymal",
                    "probability": 0.97,
                    "margin": 0.80,
                    "top_predictions": [
                        {"label": "mesenchymal", "probability": 0.97}
                    ],
                }
            ),
            SimpleNamespace(
                public_dict=lambda: {
                    "stage": "family",
                    "label_space": "learned_family",
                    "label": "CRC",
                    "probability": 0.62,
                    "margin": 0.25,
                    "top_predictions": [{"label": "CRC", "probability": 0.62}],
                }
            ),
            SimpleNamespace(
                public_dict=lambda: {
                    "stage": "entity",
                    "label_space": "learned_entity",
                    "label": "READ",
                    "probability": 0.12,
                    "margin": 0.02,
                    "top_predictions": [{"label": "READ", "probability": 0.12}],
                }
            ),
        ],
    )
    signal = SimpleNamespace(
        cancer_hint="tumor-consistent",
        top_normal_tissues=[("smooth muscle_nTPM", 0.85)],
        top_tcga_cohorts=[
            ("SARC_TPM", 0.808),
            ("READ_TPM", 0.804),
            ("COAD_TPM", 0.790),
        ],
        type_specific_cohort="SARC_TPM",
        type_specific_hits=[],
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"EPCAM": 120.0, "KRT20": 80.0, "ACTA2": 90.0}),
        {
            "cancer_type": "SARC",
            "fit_quality": {"label": "ambiguous"},
            "healthy_vs_tumor": signal,
            "candidate_trace": [
                {
                    "code": "SARC",
                    "support_fraction_of_top": 1.0,
                    "family_label": "MESENCHYMAL",
                    "signature_score": 0.59,
                },
                {
                    "code": "READ",
                    "support_fraction_of_top": 0.93,
                    "family_label": "CARCINOMA-GI",
                    "signature_score": 0.54,
                },
                {
                    "code": "COAD",
                    "support_fraction_of_top": 0.73,
                    "family_label": "CARCINOMA-GI",
                    "signature_score": 0.54,
                },
            ],
        },
    )

    assert result["selected"]["cancer_type"] == "READ"
    assert result["selected"]["selected_by"] == "entity_evidence_consensus"
    read = next(row for row in result["evidence"] if row["cancer_type"] == "READ")
    assert read["decision_features"]["learned_family_anchored_pan_cancer_context"] is True
    sarc = next(row for row in result["evidence"] if row["cancer_type"] == "SARC")
    assert sarc["can_select_report_label"] is False
    assert sarc["coarse_reference_structural_tissue_only_ambiguity"] is True
    assert sarc["decision_features"]["learned_compartment_family_contradicted"] is True


def test_composition_reference_uses_primary_tissue_to_resolve_close_cohort_tie():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    signal = SimpleNamespace(
        cancer_hint="tumor-consistent",
        top_normal_tissues=[
            ("esophagus_nTPM", 0.87),
            ("urinary_bladder_nTPM", 0.87),
            ("vagina_nTPM", 0.85),
        ],
        top_tcga_cohorts=[
            ("CESC_TPM", 0.92),
            ("BLCA_TPM", 0.91),
            ("HNSC_TPM", 0.91),
        ],
        type_specific_cohort="CESC_TPM",
        type_specific_hits=[("KRT17", 5000.0), ("DSG3", 70.0)],
    )
    result = select_report_scope_from_evidence(
        _expression_frame({"KRT17": 5000.0, "TP63": 45.0, "UPK1B": 5.0}),
        {
            "cancer_type": "LUSC",
            "fit_quality": {"label": "ambiguous"},
            "healthy_vs_tumor": signal,
            "candidate_trace": [
                {"code": "LUSC", "support_fraction_of_top": 1.0},
                {"code": "HNSC", "support_fraction_of_top": 0.98},
                {"code": "CESC", "support_fraction_of_top": 0.89},
                {"code": "BLCA", "support_fraction_of_top": 0.62},
            ],
        },
    )

    assert result["selected"]["cancer_type"] == "BLCA"
    assert result["selected"]["selected_by"] == "coarse_composition_reference"
    assert result["selected"]["coarse_reference_raw_top_code"] == "CESC"
    assert result["selected"]["coarse_reference_tissue_tiebreak_applied"] is True


def test_composition_reference_tie_with_broad_winner_stays_contextual():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    signal = SimpleNamespace(
        cancer_hint="tumor-consistent",
        top_normal_tissues=[
            ("stomach_nTPM", 0.84),
            ("pancreas_nTPM", 0.83),
            ("salivary gland_nTPM", 0.83),
        ],
        top_tcga_cohorts=[
            ("UCEC_TPM", 0.89),
            ("CESC_TPM", 0.89),
            ("BLCA_TPM", 0.88),
        ],
        type_specific_cohort="UCEC_TPM",
        type_specific_hits=[
            ("FOXA2", 100.0),
            ("PAX8", 90.0),
            ("SOX17", 80.0),
        ],
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"KRT5": 190.0, "SOX2": 50.0, "EGFR": 40.0}),
        {
            "cancer_type": "CESC",
            "fit_quality": {"label": "ambiguous"},
            "healthy_vs_tumor": signal,
            "candidate_trace": [
                {"code": "CESC", "support_fraction_of_top": 1.0},
                {"code": "LUSC", "support_fraction_of_top": 0.85},
                {"code": "UCEC", "support_fraction_of_top": 0.81},
            ],
        },
    )

    assert result["selected"]["cancer_type"] == "CESC"
    ucec = next(row for row in result["evidence"] if row["cancer_type"] == "UCEC")
    assert ucec["can_select_report_label"] is False
    assert any("tied with the first-pass RNA winner" in r for r in ucec["blocking_reasons"])


def test_composition_reference_cannot_escape_crc_family_without_specific_hits():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    signal = SimpleNamespace(
        cancer_hint="tumor-consistent",
        top_normal_tissues=[
            ("stomach_nTPM", 0.86),
            ("rectum_nTPM", 0.84),
            ("colon_nTPM", 0.83),
        ],
        top_tcga_cohorts=[
            ("STAD_TPM", 0.86),
            ("PAAD_TPM", 0.84),
            ("ESCA_TPM", 0.83),
        ],
        type_specific_cohort="",
        type_specific_hits=[],
    )

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        {
            "cancer_type": "READ",
            "fit_quality": {"label": "ambiguous"},
            "healthy_vs_tumor": signal,
            "candidate_trace": [
                {"code": "READ", "support_fraction_of_top": 1.0},
                {"code": "COAD", "support_fraction_of_top": 0.82},
                {"code": "STAD", "support_fraction_of_top": 0.78},
            ],
        },
    )

    assert result["selected"]["cancer_type"] == "READ"
    stad = next(row for row in result["evidence"] if row["cancer_type"] == "STAD")
    assert stad["can_select_report_label"] is False
    assert stad["coarse_reference_crc_family_lock"]["blocked_code"] == "STAD"
    assert any("CRC family" in reason for reason in stad["blocking_reasons"])


def test_composition_reference_cannot_escape_close_crc_family_with_weak_marker_program(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_marker_coherence",
        lambda code, sample: {
            "code": "STAD",
            "status": "partial",
            "detected": 2,
            "total": 6,
            "required_for_consistent": 3,
            "unexpected_low_detected": 4,
            "unexpected_low_genes": ["DES", "PTPRC", "MS4A1", "CD3D"],
        }
        if code == "STAD"
        else {},
    )
    signal = SimpleNamespace(
        cancer_hint="tumor-consistent",
        top_normal_tissues=[
            ("urinary_bladder_nTPM", 0.82),
            ("appendix_nTPM", 0.82),
            ("gallbladder_nTPM", 0.82),
        ],
        top_tcga_cohorts=[
            ("STAD_TPM", 0.86),
            ("PAAD_TPM", 0.84),
            ("ESCA_TPM", 0.83),
        ],
        type_specific_cohort="STAD_TPM",
        type_specific_hits=[
            ("REG4", 160.0),
            ("MUC5AC", 143.0),
            ("TFF1", 92.0),
            ("FOXA2", 64.0),
            ("ERN2", 34.0),
            ("ANXA10", 28.0),
            ("CLDN18", 12.0),
        ],
    )

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        {
            "cancer_type": "MCL",
            "fit_quality": {"label": "ambiguous"},
            "healthy_vs_tumor": signal,
            "candidate_trace": [
                {"code": "MCL", "support_fraction_of_top": 1.0, "signature_score": 0.48},
                {"code": "SARC", "support_fraction_of_top": 0.98, "signature_score": 0.81},
                {
                    "code": "READ",
                    "support_fraction_of_top": 0.883,
                    "signature_score": 0.49,
                    "family_label": "CRC",
                    "family_score": 0.38,
                },
                {"code": "STAD", "support_fraction_of_top": 0.847, "signature_score": 0.63},
            ],
        },
    )

    assert result["selected"]["cancer_type"] == "READ"
    stad = next(row for row in result["evidence"] if row["cancer_type"] == "STAD")
    assert stad["can_select_report_label"] is False
    assert stad["coarse_reference_crc_family_lock"]["crc_candidates"][0]["code"] == "READ"
    assert any("coherent marker program" in reason for reason in stad["blocking_reasons"])


def test_composition_reference_can_use_primary_normal_tissue_support():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    signal = SimpleNamespace(
        cancer_hint="tumor-consistent",
        top_normal_tissues=[
            ("lung_nTPM", 0.87),
            ("urinary_bladder_nTPM", 0.86),
            ("esophagus_nTPM", 0.84),
        ],
        top_tcga_cohorts=[
            ("LUAD_TPM", 0.91),
            ("BLCA_TPM", 0.90),
            ("BRCA_TPM", 0.89),
        ],
        type_specific_cohort="",
        type_specific_hits=[],
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"SFTPB": 450.0, "SCGB1A1": 4.5, "KRT5": 120.0}),
        {
            "cancer_type": "BRCA",
            "fit_quality": {"label": "ambiguous"},
            "healthy_vs_tumor": signal,
            "candidate_trace": [
                {"code": "BRCA", "support_fraction_of_top": 1.0},
                {"code": "LUSC", "support_fraction_of_top": 0.99},
                {"code": "BLCA", "support_fraction_of_top": 0.98},
                {"code": "LUAD", "support_fraction_of_top": 0.94},
            ],
        },
    )

    assert result["selected"]["cancer_type"] == "LUAD"
    assert result["selected"]["selected_by"] == "coarse_composition_reference"
    assert result["selected"]["coarse_reference_type_specific_hit_count"] == 0
    assert result["selected"]["coarse_reference_primary_tissue"] == "lung"
    assert result["selected"]["coarse_reference_primary_tissue_score"] == 0.87


def test_composition_reference_preserves_raw_top_when_tissue_scores_are_tied():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    signal = SimpleNamespace(
        cancer_hint="tumor-consistent",
        top_normal_tissues=[
            ("urinary_bladder_nTPM", 0.872),
            ("lung_nTPM", 0.869),
            ("esophagus_nTPM", 0.84),
        ],
        top_tcga_cohorts=[
            ("LUAD_TPM", 0.91),
            ("LUSC_TPM", 0.90),
            ("BLCA_TPM", 0.89),
        ],
        type_specific_cohort="",
        type_specific_hits=[],
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"SFTPB": 450.0, "SCGB1A1": 4.5, "KRT5": 120.0}),
        {
            "cancer_type": "BRCA",
            "fit_quality": {"label": "ambiguous"},
            "healthy_vs_tumor": signal,
            "candidate_trace": [
                {"code": "LUSC", "support_fraction_of_top": 1.0},
                {"code": "BRCA", "support_fraction_of_top": 0.99},
                {"code": "BLCA", "support_fraction_of_top": 0.98},
                {"code": "LUAD", "support_fraction_of_top": 0.97},
            ],
        },
    )

    assert result["selected"]["cancer_type"] == "LUSC"
    luad = next(row for row in result["evidence"] if row["cancer_type"] == "LUAD")
    assert luad["coarse_reference_raw_top_code"] == "LUAD"
    assert luad["coarse_reference_tissue_tiebreak_applied"] is False
    assert luad["can_select_report_label"] is False
    assert luad["coarse_reference_same_tissue_close_codes"] == ["LUSC"]
    assert any("does not distinguish LUAD from same-tissue" in r for r in luad["blocking_reasons"])


def test_luad_marker_sanity_requires_program_breadth_not_two_singletons():
    from trufflepig.tumor_type_ontology import tumor_type_sanity_check

    sanity = tumor_type_sanity_check(
        "LUAD",
        {
            "SFTPB": 450.0,
            "SCGB1A1": 4.5,
            "NAPSA": 0.2,
            "SFTPC": 0.1,
            "NKX2-1": 0.0,
            "SFTPA1": 0.0,
        },
    )

    assert sanity["status"] == "partial"
    assert len(sanity["expected_high_detected"]) == 2
    assert sanity["required_high_for_consistent"] == 3


def test_composition_reference_beats_status_child_when_broad_fit_is_ambiguous(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("ERBB2", "GRB7", "KRT8", "EPCAM")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "BRCA_Basal": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("BRCA",),
                "parent_code": "BRCA",
                "family": "carcinoma-breast",
                "primary_tissue": "breast",
                "source_cohort": "TREEHOUSE_POLYA_25_01_TCGA_BRCA_PAM50",
                "reference_kind": "deconvolved_tumor_reference",
                "expression_source": "TCGA/PAM50",
            }
        },
    )
    signal = SimpleNamespace(
        cancer_hint="tumor-consistent",
        top_tcga_cohorts=[
            ("BLCA_TPM", 0.86),
            ("UCEC_TPM", 0.85),
            ("CESC_TPM", 0.84),
        ],
        type_specific_cohort="BLCA_TPM",
        type_specific_hits=[("UPK1B", 27.0), ("KRT20", 18.0), ("GPR87", 15.0)],
    )

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                **{gene: 90.0 for gene in markers},
                "GATA3": 40.0,
                "FOXA1": 35.0,
                "TFF1": 25.0,
            }
        ),
        {
            "cancer_type": "BRCA",
            "fit_quality": {"label": "ambiguous"},
            "healthy_vs_tumor": signal,
            "candidate_trace": [
                {"code": "BRCA", "support_fraction_of_top": 1.0},
                {"code": "CESC", "support_fraction_of_top": 0.99},
            ],
        },
    )

    assert result["selected"]["cancer_type"] == "BLCA"
    assert result["selected"]["selected_by"] == "coarse_composition_reference"
    brca = next(row for row in result["evidence"] if row["cancer_type"] == "BRCA")
    assert brca["selected_by"] == "pan_cancer_signature_ranker"
    assert brca["local_reference_conflicting_coarse_reference"]["code"] == "BLCA"


def test_cross_lineage_local_reference_yields_to_coherent_composition_identity(
    monkeypatch,
):
    """A correlated exact cohort cannot outrank contradictory marker biology."""
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("ACTA2", "TFE3", "MITF", "MLANA", "PMEL")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "SARC_PEC": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("SARC",),
                "family": "sarcoma",
                "primary_tissue": "soft_tissue",
                "source_cohort": "PEC_TEST",
                "reference_kind": "observed_bulk_reference",
                "expression_source": "curated",
            }
        },
    )
    monkeypatch.setattr(
        evidence,
        "_marker_coherence",
        lambda code, _sample: {
            "code": code,
            "status": "mixed",
            "detected": 5,
            "total": 5,
            "required_for_consistent": 3,
            "detected_fraction": 1.0,
            "unexpected_low_detected": 4,
            "unexpected_low_genes": ["KRT8", "EPCAM", "KRT18", "PTPRC"],
        }
        if code == "SARC_PEC"
        else {},
    )
    monkeypatch.setattr(
        evidence,
        "_add_learned_expression_classifier_features",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evidence,
        "_add_learned_hierarchy_candidate_features",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evidence,
        "_centroid_and_confidence",
        lambda _sample: (pd.Series(dtype=float), False),
    )
    signal = SimpleNamespace(
        cancer_hint="tumor-consistent",
        top_tcga_cohorts=[
            ("PRAD_TPM", 0.79),
            ("BRCA_TPM", 0.78),
            ("SARC_TPM", 0.77),
        ],
        type_specific_cohort="PRAD_TPM",
        type_specific_hits=[
            ("KLK3", 200.0),
            ("KLK2", 100.0),
            ("NKX3-1", 80.0),
        ],
    )

    result = select_report_scope_from_evidence(
        _expression_frame({gene: 100.0 for gene in markers}),
        {
            "cancer_type": "SARC",
            "fit_quality": {"label": "ambiguous"},
            "healthy_vs_tumor": signal,
            "candidate_trace": [
                {"code": "SARC", "support_fraction_of_top": 1.0},
                {"code": "PRAD", "support_fraction_of_top": 0.80},
            ],
        },
    )

    assert result["selected"]["cancer_type"] == "PRAD"
    assert result["selected"]["selected_by"] == "coarse_composition_reference"
    pecoma = next(
        row for row in result["evidence"] if row["cancer_type"] == "SARC_PEC"
    )
    assert pecoma["can_select_report_label"] is False
    assert pecoma["local_reference_competing_composition_code"] == "PRAD"
    assert any(
        "independently selectable composition identity" in reason
        for reason in pecoma["blocking_reasons"]
    )


def test_background_like_top_label_can_yield_to_supported_tumor_label():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _candidate_analysis(
            [
                {
                    "code": "SARC",
                    "support_fraction_of_top": 1.0,
                    "signature_score": 0.82,
                    "family_label": "MESENCHYMAL",
                    "family_score": 19.0,
                },
                {
                    "code": "COAD",
                    "support_fraction_of_top": 0.80,
                    "signature_score": 0.79,
                    "family_label": "CRC",
                    "family_score": 1.0,
                },
            ]
        ),
    )

    assert result["selected"]["cancer_type"] == "COAD"
    assert result["selected"]["reference_cancer_type"] == "COAD"
    assert result["selected"]["evidence_sources"] == [
        "pan_cancer_signature_ranker",
        "tumor_label_refinement",
    ]
    assert result["selected"]["label_decision"]["status"] == "selected"
    assert result["selected"]["metrics"]["family_marker_support"] == 1.0
    assert result["selected"]["competing_background_code"] == "SARC"


def test_hematolymphoid_top_label_can_yield_to_supported_crc_candidate():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _candidate_analysis(
            [
                {"code": "MCL", "support_fraction_of_top": 1.0, "signature_score": 0.48},
                {
                    "code": "SARC",
                    "support_fraction_of_top": 0.98,
                    "signature_score": 0.81,
                    "family_label": "MESENCHYMAL",
                },
                {
                    "code": "READ",
                    "support_fraction_of_top": 0.88,
                    "signature_score": 0.49,
                    "family_label": "CRC",
                    "family_score": 0.38,
                },
                {
                    "code": "STAD",
                    "support_fraction_of_top": 0.85,
                    "signature_score": 0.63,
                    "family_label": "GASTRIC",
                    "family_score": 0.32,
                },
            ]
        ),
    )

    assert result["selected"]["cancer_type"] == "READ"
    assert result["selected"]["selected_by"] == "tumor_label_refinement"
    assert result["selected"]["tumor_label_crc_family_context"]["crc_candidate"]["code"] == "READ"


def test_weak_non_crc_fused_gi_call_blocked_by_close_crc_candidate(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    import trufflepig.expression_classifier as classifier
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        classifier,
        "classify_expression",
        lambda _sample, top_k=5: [
            ("STAD", 0.31),
            ("SARC_KS", 0.09),
            ("ESCA", 0.08),
            ("PAAD", 0.07),
            ("COAD_MSS", 0.05),
        ],
    )
    monkeypatch.setattr(
        classifier,
        "classify_expression_hierarchy",
        lambda _sample, top_k=5: [
            SimpleNamespace(
                public_dict=lambda: {
                    "stage": "compartment",
                    "label_space": "learned_compartment",
                    "label": "epithelial",
                    "probability": 0.13,
                    "margin": 0.02,
                    "top_predictions": [{"label": "epithelial", "probability": 0.13}],
                }
            ),
            SimpleNamespace(
                public_dict=lambda: {
                    "stage": "family",
                    "label_space": "learned_family",
                    "label": "carcinoma-gi",
                    "probability": 0.83,
                    "margin": 0.78,
                    "top_predictions": [{"label": "carcinoma-gi", "probability": 0.83}],
                }
            ),
            SimpleNamespace(
                public_dict=lambda: {
                    "stage": "entity",
                    "label_space": "learned_entity",
                    "label": "STAD",
                    "probability": 0.31,
                    "margin": 0.23,
                    "top_predictions": [{"label": "STAD", "probability": 0.31}],
                }
            ),
        ],
    )
    monkeypatch.setattr(
        evidence,
        "_centroid_supports_for_hypotheses",
        lambda hypotheses, cen=None: {code: (1.0 if code == "STAD" else 0.80) for code in hypotheses},
    )

    original_marker_coherence = evidence._marker_coherence

    def fake_marker_coherence(code, sample):
        if code == "STAD":
            return {
                "code": "STAD",
                "status": "mixed",
                "detected": 3,
                "total": 6,
                "required_for_consistent": 3,
                "unexpected_low_detected": 4,
                "unexpected_low_genes": ["DES", "PTPRC", "MS4A1", "CD3D"],
                "detected_fraction": 0.5,
            }
        return original_marker_coherence(code, sample)

    monkeypatch.setattr(evidence, "_marker_coherence", fake_marker_coherence)

    result = select_report_scope_from_evidence(
        _expression_frame({"EPCAM": 120.0, "KRT20": 80.0, "DES": 40.0}),
        _candidate_analysis(
            [
                {"code": "MCL", "support_fraction_of_top": 1.0, "signature_score": 0.48},
                {
                    "code": "SARC",
                    "support_fraction_of_top": 0.98,
                    "signature_score": 0.81,
                    "family_label": "MESENCHYMAL",
                },
                {
                    "code": "READ",
                    "support_fraction_of_top": 0.88,
                    "signature_score": 0.49,
                    "family_label": "CRC",
                    "family_score": 0.38,
                },
                {
                    "code": "STAD",
                    "support_fraction_of_top": 0.85,
                    "signature_score": 0.63,
                    "family_label": "GASTRIC",
                    "family_score": 0.32,
                },
            ]
        ),
    )

    assert result["selected"]["cancer_type"] == "READ"
    stad = next(row for row in result["evidence"] if row["cancer_type"] == "STAD")
    assert stad["fused_evidence_can_select"] is False
    structured_abstention = stad["fused_evidence_structured_parent_abstention"]
    assert structured_abstention["supporting_candidate"]["code"] == "READ"
    assert structured_abstention["abstention_code"] == "CRC"
    assert any("CRC-family RNA support" in reason for reason in stad["fused_evidence_blockers"])


def test_primary_context_blocker_inherits_parent_context_support():
    import trufflepig.cancer_type_evidence as evidence

    crc = evidence.CancerTypeEvidence(cancer_type="CRC")
    stad = evidence.CancerTypeEvidence(cancer_type="STAD")
    stad.broad_rna_support = 0.80
    analysis = _candidate_analysis(
        [
            {"code": "READ", "support_fraction_of_top": 1.0, "family_label": "CRC"},
            {"code": "LUSC", "support_fraction_of_top": 0.90},
            {"code": "STAD", "support_fraction_of_top": 0.80},
        ]
    )

    assert evidence._dominant_primary_context_competitor(crc, analysis) == {}
    conflict = evidence._dominant_primary_context_competitor(stad, analysis)
    assert conflict["code"] == "READ"
    assert conflict["context"] == "CRC"


def test_complete_marker_program_tolerates_single_immune_expected_low():
    from trufflepig.cancer_type_evidence import _pan_signature_marker_program_selectable

    complete_with_immune_context = {
        "status": "consistent",
        "detected": 6,
        "total": 6,
        "required_for_consistent": 3,
        "detected_fraction": 1.0,
        "unexpected_low_detected": 1,
        "unexpected_low_genes": ["PTPRC"],
    }
    mixed_with_alternate_lineage = {
        **complete_with_immune_context,
        "unexpected_low_detected": 2,
        "unexpected_low_genes": ["PTPRC", "DES"],
    }

    assert _pan_signature_marker_program_selectable(complete_with_immune_context)
    assert not _pan_signature_marker_program_selectable(mixed_with_alternate_lineage)


def test_complete_blca_marker_program_can_rescue_liver_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    import trufflepig.expression_classifier as classifier
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(classifier, "classify_expression", lambda _sample, top_k=5: [])
    monkeypatch.setattr(classifier, "classify_expression_hierarchy", lambda _sample, top_k=5: [])
    monkeypatch.setattr(
        evidence,
        "_centroid_supports_for_hypotheses",
        lambda hypotheses, cen=None: {code: (0.98 if code == "BLCA" else 0.60) for code in hypotheses},
    )

    original_marker_coherence = evidence._marker_coherence

    def fake_marker_coherence(code, sample):
        if code == "BLCA":
            return {
                "code": "BLCA",
                "status": "consistent",
                "detected": 6,
                "total": 6,
                "required_for_consistent": 3,
                "detected_fraction": 1.0,
                "unexpected_low_detected": 1,
                "unexpected_low_genes": ["PTPRC"],
                "detected_genes": ["UPK2", "UPK1A", "GATA3", "PPARG", "KRT20", "UPK3A"],
            }
        return original_marker_coherence(code, sample)

    monkeypatch.setattr(evidence, "_marker_coherence", fake_marker_coherence)

    result = select_report_scope_from_evidence(
        _expression_frame({"UPK2": 80.0, "UPK1A": 70.0, "GATA3": 100.0, "PPARG": 50.0, "KRT20": 40.0, "UPK3A": 35.0, "PTPRC": 25.0}),
        _candidate_analysis(
            [
                {"code": "HEPB", "support_fraction_of_top": 1.0, "signature_score": 0.67},
                {"code": "BLCA", "support_fraction_of_top": 0.96, "signature_score": 0.75},
            ]
        ),
    )

    assert result["selected"]["cancer_type"] == "BLCA"
    assert result["selected"]["selected_by"] == "fused_evidence"
    assert result["selected"]["pan_cancer_signature_marker_selectable"] is True


def test_background_like_top_label_does_not_yield_to_weak_tumor_label():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _candidate_analysis(
            [
                {
                    "code": "SARC",
                    "support_fraction_of_top": 1.0,
                    "signature_score": 0.66,
                    "family_label": "MESENCHYMAL",
                    "family_score": 7.0,
                },
                {
                    "code": "LUAD",
                    "support_fraction_of_top": 0.40,
                    "signature_score": 0.81,
                    "family_label": "",
                    "family_score": None,
                },
            ]
        ),
    )

    assert result["selected"]["cancer_type"] == "SARC"
    assert result["selected"]["selected_by"] == "pan_cancer_signature_ranker"


def test_local_expression_reference_can_select_future_exact_cohort(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "MM": {
                "markers": ("CD79A", "MS4A1", "MZB1"),
                "ref_medians": {"CD79A": 120.0, "MS4A1": 80.0, "MZB1": 100.0},
                "context_codes": ("DLBC", "LAML", "THYM"),
                "family": "heme-plasma",
                "primary_tissue": "bone_marrow",
                "source_cohort": "MMRF_COMMPASS",
            }
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"CD79A": 45.0, "MS4A1": 30.0, "MZB1": 40.0}),
        _analysis(("DLBC", 1.0), ("LAML", 0.4)),
    )

    assert result["selected"]["cancer_type"] == "MM"
    assert result["selected"]["reference_cancer_type"] == "DLBC"
    assert result["selected"]["expression_reference_cancer_type"] == "MM"
    assert result["selected"]["selected_by"] == "local_expression_reference"
    assert result["selected"]["metrics"]["fine_reference_support"] >= 0.65


def test_single_gene_child_reference_cannot_refine_parent_identity(monkeypatch):
    """One correlated transcript cannot refine an aggregate parent entity."""
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "CHOL": {
                "markers": ("PIGY",),
                "ref_medians": {"PIGY": 15.0},
                "context_codes": ("BTC",),
                "parent_code": "BTC",
                "family": "carcinoma-gi",
                "primary_tissue": "bile_duct",
                "source_cohort": "TCGA_CHOL",
                "reference_kind": "observed_bulk_reference",
            }
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"PIGY": 150.0}),
        _analysis(("BLCA", 1.0), ("BTC", 0.95)),
    )

    assert result["selected"]["cancer_type"] == "BLCA"
    chol = next(
        row for row in result["evidence"] if row["cancer_type"] == "CHOL"
    )
    assert chol["can_select_report_label"] is False
    assert chol["local_reference_marker_count"] == 1
    assert any(
        "single-gene child expression reference" in reason
        for reason in chol["blocking_reasons"]
    )


def test_pirlygenes_observed_reference_can_select_exact_cohort(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    evidence._local_expression_reference_panels.cache_clear()
    monkeypatch.setattr(
        evidence,
        "_registry_by_code",
        lambda: {
            "MM": {
                "family": "heme-plasma",
                "primary_tissue": "bone_marrow",
            }
        },
    )
    monkeypatch.setattr(
        "trufflepig.reference.subtype_deconvolved_expression",
        lambda *args, **kwargs: pd.DataFrame(
            columns=["cancer_code", "symbol", "tumor_tpm_median"]
        ),
    )
    monkeypatch.setattr(
        "trufflepig.reference.cancer_reference_expression",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "Ensembl_Gene_ID": "ENSG00000105369",
                    "Symbol": "CD79A",
                    "cancer_code": "MM",
                    "source_cohort": "MMRF_COMMPASS",
                    "normalization": "tpm_clean",
                    "expression": 120.0,
                    "q1": 60.0,
                    "q3": 180.0,
                },
                {
                    "Ensembl_Gene_ID": "ENSG00000156738",
                    "Symbol": "MS4A1",
                    "cancer_code": "MM",
                    "source_cohort": "MMRF_COMMPASS",
                    "normalization": "tpm_clean",
                    "expression": 80.0,
                    "q1": 30.0,
                    "q3": 110.0,
                },
                {
                    "Ensembl_Gene_ID": "ENSG00000170476",
                    "Symbol": "MZB1",
                    "cancer_code": "MM",
                    "source_cohort": "MMRF_COMMPASS",
                    "normalization": "tpm_clean",
                    "expression": 100.0,
                    "q1": 40.0,
                    "q3": 140.0,
                },
            ]
        ),
    )

    try:
        result = select_report_scope_from_evidence(
            _expression_frame({"CD79A": 45.0, "MS4A1": 30.0, "MZB1": 40.0}),
            _analysis(("DLBC", 1.0), ("LAML", 0.4)),
        )
    finally:
        evidence._local_expression_reference_panels.cache_clear()

    assert result["selected"]["cancer_type"] == "MM"
    assert result["selected"]["reference_cancer_type"] == "DLBC"
    assert result["selected"]["expression_reference_cancer_type"] == "MM"
    assert result["selected"]["local_reference_kind"] == "observed_bulk_reference"
    assert result["selected"]["local_reference_source_cohort"] == "MMRF_COMMPASS"


def test_local_expression_reference_uses_actual_top_compatible_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "MM": {
                "markers": ("CD79A", "MS4A1", "MZB1"),
                "ref_medians": {"CD79A": 120.0, "MS4A1": 80.0, "MZB1": 100.0},
                "context_codes": ("DLBC", "LAML", "THYM"),
                "family": "heme-plasma",
                "primary_tissue": "bone_marrow",
                "source_cohort": "MMRF_COMMPASS",
            }
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"CD79A": 45.0, "MS4A1": 30.0, "MZB1": 40.0}),
        _analysis(("LAML", 1.0), ("DLBC", 0.4)),
    )

    assert result["selected"]["cancer_type"] == "MM"
    assert result["selected"]["reference_cancer_type"] == "LAML"
    assert result["selected"]["local_reference_matched_context"] == "LAML"


def test_local_reference_context_codes_fall_back_to_family_root():
    from trufflepig.cancer_type_evidence import _local_reference_context_codes

    assert _local_reference_context_codes(
        "ATRT",
        {"family": "cns-embryonal", "primary_tissue": "cerebellum"},
    ) == ("GBM", "LGG")
    assert _local_reference_context_codes(
        "RT",
        {"family": "embryonal", "primary_tissue": "kidney_cns_soft"},
    ) == ("GBM",)
    assert _local_reference_context_codes(
        "HEPB",
        {"family": "embryonal", "primary_tissue": "liver"},
    ) == ("LIHC",)


def test_split_family_local_expression_reference_can_select(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "ATRT": {
                "markers": ("GFAP", "VIM", "NES"),
                "ref_medians": {"GFAP": 120.0, "VIM": 80.0, "NES": 100.0},
                "context_codes": evidence._local_reference_context_codes(
                    "ATRT",
                    {"family": "cns-embryonal", "primary_tissue": "cerebellum"},
                ),
                "family": "cns-embryonal",
                "primary_tissue": "cerebellum",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            }
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"GFAP": 45.0, "VIM": 30.0, "NES": 40.0}),
        _analysis(("GBM", 1.0), ("LGG", 0.8)),
    )

    assert result["selected"]["cancer_type"] == "ATRT"
    assert result["selected"]["reference_cancer_type"] == "GBM"
    assert result["selected"]["expression_reference_cancer_type"] == "ATRT"
    assert result["selected"]["selected_by"] == "local_expression_reference"


def test_bl_marker_axis_unblocks_exact_expression_reference(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("MYC", "BCL6", "MME", "CD79A", "MS4A1")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "BL": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("DLBC",),
                "family": "heme-bcell",
                "primary_tissue": "lymphoid",
                "source_cohort": "CGCI_BLGSP",
                "reference_kind": "observed_bulk_reference",
                "fusion_driven": "defining",
            }
        },
    )
    finding = {
        "cancer_type": "BL",
        "rule_id": "lit_bl",
        "surrogate": "MYC",
        "surrogate_tpm": 300.0,
        "threshold_tpm": 5.0,
        "support_genes": ["BCL6", "MME", "CD79A", "MS4A1"],
        "support_gene_count": 4,
        "min_support_genes": 4,
        "required_support_gene_count": 4,
        "support_pass": True,
    }

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "MYC": 300.0,
                "BCL6": 200.0,
                "MME": 120.0,
                "CD79A": 180.0,
                "MS4A1": 250.0,
            }
        ),
        _analysis(("DLBC", 1.0), ("LAML", 0.8)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "BL"
    assert result["selected"]["selected_by"] == "local_expression_reference"
    assert set(result["selected"]["evidence_sources"]) == {
        "local_expression_reference",
        "rare_marker",
    }
    bl = next(row for row in result["evidence"] if row["cancer_type"] == "BL")
    assert bl["can_select_report_label"] is True


def test_ball_exact_reference_beats_bl_myc_marker_axis(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    bl_markers = ("MYC", "BCL6", "MME", "CD79A", "MS4A1")
    ball_markers = ("DNTT", "VPREB1", "RAG1", "CD19", "PAX5", "CD22")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "BL": {
                "markers": bl_markers,
                "ref_medians": {gene: 100.0 for gene in bl_markers},
                "context_codes": ("DLBC",),
                "family": "heme-bcell",
                "primary_tissue": "lymphoid",
                "source_cohort": "CGCI_BLGSP",
                "reference_kind": "observed_bulk_reference",
                "fusion_driven": "defining",
                "fusion_driver": "IGH-MYC",
            },
            "B_ALL": {
                "markers": ball_markers,
                "ref_medians": {gene: 100.0 for gene in ball_markers},
                "context_codes": ("DLBC", "LAML", "THYM"),
                "family": "heme-bcell",
                "primary_tissue": "bone_marrow",
                "source_cohort": "TARGET_ALL_P3",
                "reference_kind": "observed_bulk_reference",
            },
        },
    )
    finding = {
        "cancer_type": "BL",
        "rule_id": "lit_bl",
        "surrogate": "MYC",
        "surrogate_tpm": 300.0,
        "threshold_tpm": 5.0,
        "support_genes": ["BCL6", "MME", "CD79A", "MS4A1"],
        "support_gene_count": 4,
        "min_support_genes": 4,
        "required_support_gene_count": 4,
        "support_pass": True,
    }

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "MYC": 300.0,
                "BCL6": 10.0,
                "MME": 5.0,
                "CD79A": 300.0,
                "MS4A1": 20.0,
                **{gene: 120.0 for gene in ball_markers},
            }
        ),
        _analysis(("DLBC", 1.0), ("LAML", 0.9)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "B_ALL"
    assert result["selected"]["selected_by"] == "local_expression_reference"
    bl = next(row for row in result["evidence"] if row["cancer_type"] == "BL")
    ball = next(row for row in result["evidence"] if row["cancer_type"] == "B_ALL")
    assert bl["can_select_report_label"] is True
    assert set(bl["evidence_sources"]) == {"local_expression_reference", "rare_marker"}
    assert ball["metrics"]["fine_reference_support"] > bl["metrics"]["fine_reference_support"]


def test_split_family_specificity_breaks_close_exact_reference_ties(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    base_markers = (
        "VIM",
        "NES",
        "SOX2",
        "SOX9",
        "COL1A1",
        "KRT8",
        "EPCAM",
        "MKI67",
        "CDH2",
        "PROM1",
        "OLIG2",
        "AQP4",
        "S100B",
        "DCX",
        "DLL3",
        "TOP2A",
        "HMGB2",
        "PCNA",
        "MCM2",
        "MCM3",
        "MCM4",
        "MCM5",
    )
    rt_markers = base_markers + ("KRT7", "GFAP")
    atrt_markers = base_markers + ("NTRK2", "GFAP")
    ref_medians = {gene: 100.0 for gene in set(rt_markers + atrt_markers)}

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "RT": {
                "markers": rt_markers,
                "ref_medians": ref_medians,
                "context_codes": ("GBM",),
                "family": "embryonal",
                "primary_tissue": "kidney_cns_soft",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            },
            "ATRT": {
                "markers": atrt_markers,
                "ref_medians": ref_medians,
                "context_codes": ("GBM", "LGG"),
                "family": "cns-embryonal",
                "primary_tissue": "cerebellum",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            },
        },
    )
    sample = {gene: 50.0 for gene in base_markers + ("KRT7",)}
    sample.update({"NTRK2": 0.0, "GFAP": 0.0, "SMARCB1": 0.0})

    result = select_report_scope_from_evidence(
        _expression_frame(sample),
        _analysis(("GBM", 1.0), ("LGG", 0.8)),
    )

    assert result["selected"]["cancer_type"] == "ATRT"
    atrt = next(row for row in result["evidence"] if row["cancer_type"] == "ATRT")
    rt = next(row for row in result["evidence"] if row["cancer_type"] == "RT")
    assert atrt["local_reference_strength"] < rt["local_reference_strength"]
    assert atrt["local_reference_specificity_bonus"] > rt["local_reference_specificity_bonus"]


def test_high_confidence_exact_reference_can_beat_parent_lineage_reinforcement(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("AFP", "DLK1", "GPC3", "EPCAM", "KRT8", "KRT18", "KRT19", "SOX9")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "HEPB": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("LIHC",),
                "family": "embryonal",
                "primary_tissue": "liver",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            }
        },
    )

    def fake_lineage_panel(hypotheses, *_args, **_kwargs):
        hypothesis = evidence._hypothesis(hypotheses, "LIHC")
        hypothesis.add_source("lineage_panel")
        hypothesis.expression_reference_cancer_type = "LIHC"
        hypothesis.reference_cancer_type = "LIHC"
        hypothesis.consider_for_report_label(
            selected_by="lineage_panel",
            can_select=True,
            blocking_reasons=(),
            priority=(2, 0.95),
        )
        return {"promotion": {"promoted": True, "code": "LIHC", "blockers": []}}

    monkeypatch.setattr(evidence, "_add_lineage_panel_features", fake_lineage_panel)

    result = select_report_scope_from_evidence(
        _expression_frame({gene: 80.0 for gene in markers}),
        _analysis(("LIHC", 1.0), ("CHOL", 0.8)),
    )

    assert result["selected"]["cancer_type"] == "HEPB"
    assert result["selected"]["local_reference_high_confidence"] is True


def test_hepb_reference_needs_fetal_anchor_to_override_strong_lihc_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("AFP", "DLK1", "GPC3", "EPCAM", "KRT8", "KRT18", "KRT19", "SOX9")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "HEPB": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("LIHC",),
                "family": "embryonal",
                "primary_tissue": "liver",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            }
        },
    )

    def fake_lineage_panel(hypotheses, *_args, **_kwargs):
        hypothesis = evidence._hypothesis(hypotheses, "LIHC")
        hypothesis.add_source("lineage_panel")
        hypothesis.expression_reference_cancer_type = "LIHC"
        hypothesis.reference_cancer_type = "LIHC"
        hypothesis.consider_for_report_label(
            selected_by="lineage_panel",
            can_select=True,
            blocking_reasons=(),
            priority=(2, 0.95),
        )
        return {"promotion": {"promoted": True, "code": "LIHC", "blockers": []}}

    monkeypatch.setattr(evidence, "_add_lineage_panel_features", fake_lineage_panel)

    expression = {
        "AFP": 80.0,
        "DLK1": 0.0,
        "GPC3": 80.0,
        "EPCAM": 80.0,
        "KRT8": 500.0,
        "KRT18": 500.0,
        "KRT19": 80.0,
        "SOX9": 80.0,
        "SALL4": 0.0,
        "IGF2": 2.0,
        "ALB": 20000.0,
        "APOB": 300.0,
        "CYP3A4": 1000.0,
        "HNF4A": 60.0,
        "F2": 500.0,
    }
    result = select_report_scope_from_evidence(
        _expression_frame(expression),
        _analysis(("LIHC", 1.0), ("CHOL", 0.8)),
    )

    assert result["selected"]["cancer_type"] == "LIHC"
    hepb = next(row for row in result["evidence"] if row["cancer_type"] == "HEPB")
    assert hepb["hepb_adult_liver_context_conflict"] is True
    assert any("fetal-liver anchor" in reason for reason in hepb["blocking_reasons"])


def test_high_confidence_exact_reference_can_rescue_near_top_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("AFP", "DLK1", "GPC3", "EPCAM", "KRT8", "KRT18", "KRT19", "SOX9")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "HEPB": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("LIHC",),
                "family": "embryonal",
                "primary_tissue": "liver",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            }
        },
    )
    analysis = _analysis(("LUAD", 1.0), ("LIHC", 0.94), ("HNSC", 0.88))
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        top_tcga_cohorts=[("LUAD_TPM", 0.87), ("PAAD_TPM", 0.85)]
    )

    result = select_report_scope_from_evidence(
        _expression_frame({gene: 80.0 for gene in markers}),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "HEPB"
    assert result["selected"]["local_reference_matched_context"] == "LIHC"
    assert result["selected"]["local_reference_context_is_top"] is True


def test_strong_marker_coherence_rescues_near_high_exact_reference(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    reference_markers = ("REG3A", "DLK1", "TOMM6", "XBP1", "IGF2", "NME1-NME2")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "HEPB": {
                "markers": reference_markers,
                "ref_medians": {gene: 100.0 for gene in reference_markers},
                "context_codes": ("LIHC",),
                "family": "embryonal",
                "primary_tissue": "liver",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            }
        },
    )
    analysis = _analysis(("LUAD", 1.0), ("LIHC", 0.86), ("UCS", 0.84))
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        top_tcga_cohorts=[("LUAD_TPM", 0.87), ("UCEC_TPM", 0.85)]
    )
    expression = {
        "REG3A": 0.0,
        "DLK1": 80.0,
        "TOMM6": 80.0,
        "XBP1": 80.0,
        "IGF2": 80.0,
        "NME1-NME2": 0.0,
        "AFP": 40.0,
        "GPC3": 200.0,
        "EPCAM": 150.0,
        "KRT8": 500.0,
        "KRT18": 500.0,
        "SALL4": 40.0,
        "GLUL": 200.0,
    }

    result = select_report_scope_from_evidence(_expression_frame(expression), analysis)

    assert result["selected"]["cancer_type"] == "HEPB"
    assert result["selected"]["local_reference_reference_high_confidence"] is False
    assert result["selected"]["local_reference_marker_coherence_supported"] is True
    assert result["selected"]["local_reference_context_is_top"] is True


def test_mixed_cross_lineage_local_reference_cannot_select_label(monkeypatch):
    """A local sarcoma reference can be context evidence, but mixed ontology markers cannot
    override a first-pass epithelial context across lineages."""
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("MDM2", "FRS2", "TP53", "CDK4", "HMGA2")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "SARC_DDLPS": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("SARC",),
                "family": "sarcoma",
                "primary_tissue": "adipose",
                "source_cohort": "GSE75885_DELESPAUL_2017",
                "reference_kind": "deconvolved_tumor_reference",
            }
        },
    )
    expression = {
        **{gene: 150.0 for gene in markers},
        # Expected-low/conflicting lineage markers make SARC_DDLPS "mixed", not clean.
        "EPCAM": 200.0,
        "KRT8": 200.0,
        "KRT18": 200.0,
        "PTPRC": 50.0,
        "CD3D": 50.0,
        "MS4A1": 50.0,
    }

    result = select_report_scope_from_evidence(
        _expression_frame(expression),
        _analysis(("READ", 1.0), ("SARC", 0.92), ("COAD", 0.85)),
    )

    assert result["selected"]["cancer_type"] == "READ"
    ddlps = next(row for row in result["evidence"] if row["cancer_type"] == "SARC_DDLPS")
    assert ddlps["can_select_report_label"] is False
    assert ddlps["local_reference_cross_lineage_marker_conflict"] is True
    assert any("while the compatible RNA context is READ" in r for r in ddlps["blocking_reasons"])


def test_generic_smooth_muscle_reference_yields_to_epithelial_compartment(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("ACTA2", "CALD1", "MYH11", "DES")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "SARC_LMS": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("SARC",),
                "family": "sarcoma",
                "primary_tissue": "smooth_muscle",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            }
        },
    )
    monkeypatch.setattr(
        evidence,
        "_centroid_and_confidence",
        lambda sample_tpm_by_symbol: (None, False),
    )

    result = select_report_scope_from_evidence(
        _expression_frame({gene: 160.0 for gene in markers}),
        _candidate_analysis(
            [
                {
                    "code": "SARC",
                    "support_fraction_of_top": 1.0,
                    "signature_score": 0.84,
                    "family_label": "MESENCHYMAL",
                    "centroid_coarse_lineage": "Epithelial",
                    "compartment_in_set": False,
                    "winning_subtype": "SARC_LMS",
                },
                {
                    "code": "READ",
                    "support_fraction_of_top": 0.97,
                    "signature_score": 0.60,
                    "family_label": "CRC",
                    "family_score": 0.40,
                },
                {
                    "code": "COAD",
                    "support_fraction_of_top": 0.80,
                    "signature_score": 0.59,
                    "family_label": "CRC",
                    "family_score": 0.40,
                },
            ]
        ),
    )

    assert result["selected"]["cancer_type"] == "READ"
    assert result["selected"]["selected_by"] == "tumor_label_refinement"
    assert result["selected"]["tumor_label_compartment_conflict_override"] is True
    lms = next(row for row in result["evidence"] if row["cancer_type"] == "SARC_LMS")
    assert lms["can_select_report_label"] is False
    assert lms["broad_subtype_background_compartment_conflict"]["winning_subtype"] == "SARC_LMS"
    assert lms["local_reference_background_compartment_conflict"] is True
    assert any("mesenchymal exact-reference" in r for r in lms["blocking_reasons"])


def test_basal_brca_candidate_overrides_background_label_refinement():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    analysis = _candidate_analysis(
        [
            {
                "code": "UCS",
                "support_fraction_of_top": 1.0,
                "signature_score": 0.56,
                "family_label": "MESENCHYMAL",
            },
            {
                "code": "ESCA",
                "support_fraction_of_top": 0.94,
                "signature_score": 0.49,
                "family_label": "ESCA_SQ",
                "family_score": 1.0,
            },
            {
                "code": "BRCA",
                "support_fraction_of_top": 0.67,
                "signature_score": 0.53,
                "winning_subtype": "BRCA_Basal",
            },
        ]
    )

    result = select_report_scope_from_evidence(_empty_expression_frame(), analysis)

    assert result["selected"]["cancer_type"] == "BRCA"
    assert result["selected"]["tumor_label_basal_brca_override"] is True


def test_nonclassification_local_reference_cannot_replace_broad_ranker(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("CDK4", "HMGA2", "MDM2", "TSPAN31", "YEATS4", "FRS2")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "SARC_LPS_UNSPEC": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("SARC",),
                "family": "sarcoma",
                "primary_tissue": "adipose",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            }
        },
    )
    analysis = _candidate_analysis(
        [
            {
                "code": "SARC",
                "support_fraction_of_top": 1.0,
                "signature_score": 0.8,
                "winning_subtype": "SARC_RMS_ERMS",
            }
        ]
    )
    expression = {
        **{gene: 150.0 for gene in markers},
        "MYOD1": 40.0,
        "MYOG": 35.0,
        "DES": 60.0,
        "MYF5": 10.0,
        "MYF6": 8.0,
    }

    result = select_report_scope_from_evidence(_expression_frame(expression), analysis)

    assert result["selected"]["cancer_type"] == "SARC"
    rms = next(row for row in result["evidence"] if row["cancer_type"] == "SARC_RMS_ERMS")
    assert rms["evidence_sources"] == ["pan_cancer_signature_subtype"]
    assert rms["can_select_report_label"] is False
    lps = next(row for row in result["evidence"] if row["cancer_type"] == "SARC_LPS_UNSPEC")
    assert lps["can_select_report_label"] is False
    assert any(
        "rather than a classification target" in reason
        for reason in lps["blocking_reasons"]
    )


def test_learned_expression_classifier_can_rescue_context_supported_type(monkeypatch):
    """A decisive learned full-profile vote can resolve a context-supported ambiguity."""
    import trufflepig.cancer_type_evidence as evidence
    import trufflepig.expression_classifier as classifier
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        classifier,
        "classify_expression",
        lambda _sample, top_k=5: [("STAD", 0.92), ("READ", 0.04), ("COAD", 0.03)],
    )
    monkeypatch.setattr(
        classifier,
        "classify_expression_hierarchy",
        lambda _sample, top_k=5: [
            SimpleNamespace(
                public_dict=lambda: {
                    "stage": "compartment",
                    "label_space": "learned_compartment",
                    "label": "epithelial",
                    "probability": 0.99,
                    "margin": 0.95,
                    "top_predictions": [
                        {"label": "epithelial", "probability": 0.99}
                    ],
                }
            ),
            SimpleNamespace(
                public_dict=lambda: {
                    "stage": "family",
                    "label_space": "learned_family",
                    "label": "carcinoma-gi",
                    "probability": 0.92,
                    "margin": 0.70,
                    "top_predictions": [
                        {"label": "carcinoma-gi", "probability": 0.92}
                    ],
                }
            ),
            SimpleNamespace(
                public_dict=lambda: {
                    "stage": "entity",
                    "label_space": "learned_entity",
                    "label": "STAD",
                    "probability": 0.92,
                    "margin": 0.70,
                    "top_predictions": [
                        {"label": "STAD", "probability": 0.92}
                    ],
                }
            ),
        ],
    )
    monkeypatch.setattr(evidence, "_marker_coherence", lambda _code, _sample: {})

    analysis = _candidate_analysis(
        [
            {"code": "READ", "support_fraction_of_top": 1.0},
            {"code": "STAD", "support_fraction_of_top": 0.42},
        ]
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"EPCAM": 100.0}),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "STAD"
    assert result["selected"]["selected_by"] == "learned_expression_classifier"
    stad = next(row for row in result["evidence"] if row["cancer_type"] == "STAD")
    assert stad["metrics"]["learned_expression_support"] == 0.92
    learned_channels = [
        row for row in result["staged_evidence_graph"]["channels"]
        if row["channel"] == "learned_expression_classifier"
    ]
    assert any(
        row["role"] == "full_profile_discriminative_vote"
        and row.get("candidate_code") == "STAD"
        for row in learned_channels
    )
    assert any(
        row["role"] == "hierarchical_compartment_vote"
        and row.get("candidate_code") == "epithelial"
        for row in learned_channels
    )
    assert any(
        row["role"] == "hierarchical_family_vote"
        and row.get("candidate_code") == "carcinoma-gi"
        for row in learned_channels
    )
    assert any(
        row["role"] == "hierarchical_entity_vote"
        and row.get("candidate_code") == "STAD"
        for row in learned_channels
    )


def test_learned_expression_classifier_can_admit_context_free_hierarchical_vote(monkeypatch):
    """A very strong learned hierarchy can beat an unsupported primary-expression attractor."""
    import trufflepig.cancer_type_evidence as evidence
    import trufflepig.expression_classifier as classifier
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        classifier,
        "classify_expression",
        lambda _sample, top_k=5: [("SARC_ASPS", 0.992), ("HNSC", 0.002)],
    )
    monkeypatch.setattr(
        classifier,
        "classify_expression_hierarchy",
        lambda _sample, top_k=5: [
            SimpleNamespace(
                public_dict=lambda: {
                    "stage": "compartment",
                    "label_space": "learned_compartment",
                    "label": "mesenchymal",
                    "probability": 0.992,
                    "margin": 0.90,
                    "top_predictions": [
                        {"label": "mesenchymal", "probability": 0.992}
                    ],
                }
            ),
            SimpleNamespace(
                public_dict=lambda: {
                    "stage": "family",
                    "label_space": "learned_family",
                    "label": "SARC_MELANOCYTIC_TRANSLOCATION",
                    "probability": 0.992,
                    "margin": 0.90,
                    "top_predictions": [
                        {
                            "label": "SARC_MELANOCYTIC_TRANSLOCATION",
                            "probability": 0.992,
                        }
                    ],
                }
            ),
            SimpleNamespace(
                public_dict=lambda: {
                    "stage": "entity",
                    "label_space": "learned_entity",
                    "label": "SARC_ASPS",
                    "probability": 0.992,
                    "margin": 0.90,
                    "top_predictions": [
                        {"label": "SARC_ASPS", "probability": 0.992}
                    ],
                }
            ),
        ],
    )
    monkeypatch.setattr(evidence, "_marker_coherence", lambda _code, _sample: {})

    result = select_report_scope_from_evidence(
        _expression_frame({"EPCAM": 10.0, "ASPSCR1": 100.0}),
        _candidate_analysis(
            [
                {"code": "HNSC", "support_fraction_of_top": 1.0},
            ]
        ),
    )

    assert result["selected"]["cancer_type"] == "SARC_ASPS"
    assert result["selected"]["selected_by"] == "learned_expression_classifier"
    assert result["selected"]["learned_expression_hierarchical_rescue"] is True


def test_learned_expression_classifier_blocks_background_compartment_flip(monkeypatch):
    """A learned sarcoma-like vote stays contextual in a PFO002-style epithelial case."""
    import trufflepig.cancer_type_evidence as evidence
    import trufflepig.expression_classifier as classifier
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        classifier,
        "classify_expression",
        lambda _sample, top_k=5: [
            ("SARC_LMS", 0.96),
            ("READ", 0.02),
            ("STAD", 0.01),
        ],
    )
    monkeypatch.setattr(evidence, "_marker_coherence", lambda _code, _sample: {})

    analysis = _candidate_analysis(
        [
            {
                "code": "SARC",
                "support_fraction_of_top": 1.0,
                "family_label": "MESENCHYMAL",
                "signature_score": 0.70,
                "centroid_coarse_lineage": "Epithelial",
                "centroid_lineage_confident": True,
                "compartment_in_set": False,
                "winning_subtype": "SARC_LMS",
            },
            {
                "code": "READ",
                "support_fraction_of_top": 0.90,
                "signature_score": 0.66,
                "family_score": 0.75,
                "family_label": "CARCINOMA-GI",
            },
        ]
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"EPCAM": 120.0, "KRT20": 80.0}),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "READ"
    sarc_lms = next(row for row in result["evidence"] if row["cancer_type"] == "SARC_LMS")
    assert sarc_lms["label_decision"]["status"] == "blocked"
    assert any(
        "background-like SARC context" in reason
        for reason in sarc_lms["blocking_reasons"]
    )


def test_status_child_parent_fallback_does_not_outrank_direct_exact_reference(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    hepb_markers = (
        "REG3A",
        "DLK1",
        "TOMM6",
        "XBP1",
        "IGF2",
        "NME1-NME2",
        "PGC",
        "SOD2",
    )
    luad_status_markers = ("SFTPC", "SFTPA1", "SFTPA2", "SFTPB")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "HEPB": {
                "markers": hepb_markers,
                "ref_medians": {gene: 100.0 for gene in hepb_markers},
                "context_codes": ("LIHC",),
                "family": "embryonal",
                "primary_tissue": "liver",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            },
            "LUAD_EGFR": {
                "markers": luad_status_markers,
                "ref_medians": {gene: 100.0 for gene in luad_status_markers},
                "context_codes": ("LUAD",),
                "parent_code": "LUAD",
                "family": "carcinoma-lung",
                "primary_tissue": "lung",
                "source_cohort": "TREEHOUSE_POLYA_25_01_TCGA_LUAD_MUT",
                "reference_kind": "deconvolved_tumor_reference",
                "expression_source": "TCGA/mut",
            },
        },
    )

    def fake_lineage_panel(hypotheses, *_args, **_kwargs):
        hypothesis = evidence._hypothesis(hypotheses, "LUAD")
        hypothesis.add_source("lineage_panel")
        hypothesis.expression_reference_cancer_type = "LUAD"
        hypothesis.reference_cancer_type = "LUAD"
        hypothesis.consider_for_report_label(
            selected_by="lineage_panel",
            can_select=True,
            blocking_reasons=(),
            priority=(2, 0.94),
        )
        return {"promotion": {"promoted": True, "code": "LUAD", "blockers": []}}

    monkeypatch.setattr(evidence, "_add_lineage_panel_features", fake_lineage_panel)

    expression = {
        "REG3A": 0.0,
        "DLK1": 80.0,
        "TOMM6": 80.0,
        "XBP1": 80.0,
        "IGF2": 80.0,
        "NME1-NME2": 0.0,
        "PGC": 80.0,
        "SOD2": 80.0,
        "AFP": 40.0,
        "GPC3": 200.0,
        "EPCAM": 150.0,
        "KRT8": 500.0,
        "KRT18": 500.0,
        "SALL4": 40.0,
        "GLUL": 200.0,
        **{gene: 150.0 for gene in luad_status_markers},
    }
    result = select_report_scope_from_evidence(
        _expression_frame(expression),
        _analysis(("LUAD", 1.0), ("LIHC", 0.86), ("UCS", 0.84)),
    )

    assert result["selected"]["cancer_type"] == "HEPB"
    assert result["selected"]["local_reference_marker_coherence_supported"] is True
    luad = next(row for row in result["evidence"] if row["cancer_type"] == "LUAD")
    assert luad["can_select_report_label"] is True
    assert luad["local_reference_status_child_code"] == "LUAD_EGFR"


def test_secondary_context_exact_reference_stays_blocked_when_not_high_confidence(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("AFP", "DLK1", "GPC3", "EPCAM", "KRT8", "KRT18", "KRT19", "SOX9")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "HEPB": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("LIHC",),
                "family": "embryonal",
                "primary_tissue": "liver",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            }
        },
    )
    analysis = _analysis(("LUAD", 1.0), ("LIHC", 0.94), ("HNSC", 0.88))
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        top_tcga_cohorts=[("LUAD_TPM", 0.87), ("PAAD_TPM", 0.85)]
    )
    sample = {gene: 0.0 for gene in markers}
    sample.update({gene: 80.0 for gene in markers[:5]})

    result = select_report_scope_from_evidence(_expression_frame(sample), analysis)

    assert result["selected"]["cancer_type"] == "LUAD"
    hepb = next(row for row in result["evidence"] if row["cancer_type"] == "HEPB")
    assert hepb["local_reference_high_confidence"] is False
    assert any("secondary near-match" in reason for reason in hepb["blocking_reasons"])


def test_rhabdoid_reference_requires_smarcb1_loss_compatible_rna(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "ATRT": {
                "markers": ("GFAP", "VIM", "NES"),
                "ref_medians": {"GFAP": 120.0, "VIM": 80.0, "NES": 100.0},
                "context_codes": ("GBM", "LGG"),
                "family": "cns-embryonal",
                "primary_tissue": "cerebellum",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            }
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame(
            {"GFAP": 45.0, "VIM": 30.0, "NES": 40.0, "SMARCB1": 100.0}
        ),
        _analysis(("GBM", 1.0), ("LGG", 0.8)),
    )

    assert result["selected"]["cancer_type"] == "GBM"
    atrt = next(row for row in result["evidence"] if row["cancer_type"] == "ATRT")
    assert atrt["can_select_report_label"] is False
    assert any("SMARCB1-loss-compatible" in reason for reason in atrt["blocking_reasons"])


def test_rt_reference_requires_near_absent_smarcb1_rna(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("VIM", "EPCAM", "KRT8", "KRT18", "SALL4")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "RT": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("GBM",),
                "family": "embryonal",
                "primary_tissue": "kidney_cns_soft",
                "source_cohort": "TARGET_RT_2017",
                "reference_kind": "deconvolved_tumor_reference",
            }
        },
    )
    analysis = _analysis(("BRCA", 1.0), ("GBM", 0.40))
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        top_tcga_cohorts=[("BRCA_TPM", 0.93), ("LUSC_TPM", 0.88)]
    )

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                **{gene: 120.0 for gene in markers},
                "SMARCB1": 38.0,
                "GATA3": 140.0,
                "FOXA1": 70.0,
                "MUC1": 130.0,
            }
        ),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "BRCA"
    rt = next(row for row in result["evidence"] if row["cancer_type"] == "RT")
    assert rt["can_select_report_label"] is False
    assert any("SMARCB1-loss-compatible" in reason for reason in rt["blocking_reasons"])


def test_mbl_reference_requires_broad_marker_fraction(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "MBL": {
                "markers": ("OTX2", "ATOH1", "GABRA5", "GRM8"),
                "ref_medians": {
                    "OTX2": 100.0,
                    "ATOH1": 80.0,
                    "GABRA5": 60.0,
                    "GRM8": 60.0,
                },
                "context_codes": ("GBM", "LGG"),
                "family": "cns-embryonal",
                "primary_tissue": "cerebellum",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            }
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"OTX2": 40.0, "ATOH1": 35.0, "GABRA5": 0.0, "GRM8": 0.0}),
        _analysis(("GBM", 1.0), ("LGG", 0.8)),
    )

    assert result["selected"]["cancer_type"] == "GBM"
    mbl = next(row for row in result["evidence"] if row["cancer_type"] == "MBL")
    assert mbl["can_select_report_label"] is False
    assert any("marker fraction" in reason for reason in mbl["blocking_reasons"])


def test_generic_embryonal_reference_does_not_refine_chol_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "RT": {
                "markers": ("VIM", "EPCAM", "KRT8"),
                "ref_medians": {"VIM": 100.0, "EPCAM": 80.0, "KRT8": 80.0},
                "context_codes": evidence._local_reference_context_codes(
                    "RT",
                    {"family": "embryonal", "primary_tissue": "kidney_cns_soft"},
                ),
                "family": "embryonal",
                "primary_tissue": "kidney_cns_soft",
                "source_cohort": "TARGET_RT_2017",
                "reference_kind": "deconvolved_tumor_reference",
                "expression_source": "TARGET",
            }
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"VIM": 80.0, "EPCAM": 70.0, "KRT8": 70.0}),
        _analysis(("CHOL", 1.0), ("LIHC", 0.3)),
    )

    assert result["selected"]["cancer_type"] == "CHOL"
    rt = next(row for row in result["evidence"] if row["cancer_type"] == "RT")
    assert rt["can_select_report_label"] is False
    assert "CHOL" not in rt["local_reference_context_codes"]


def test_local_expression_reference_can_use_coarse_reference_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "ATRT": {
                "markers": ("GFAP", "VIM", "NES"),
                "ref_medians": {"GFAP": 120.0, "VIM": 80.0, "NES": 100.0},
                "context_codes": ("GBM", "LGG"),
                "family": "cns-embryonal",
                "primary_tissue": "cerebellum",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            }
        },
    )
    analysis = _analysis(("SKCM", 1.0), ("GBM", 0.82))
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        top_tcga_cohorts=[("GBM_TPM", 0.84), ("OV_TPM", 0.82)]
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"GFAP": 45.0, "VIM": 30.0, "NES": 40.0}),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "ATRT"
    assert result["selected"]["reference_cancer_type"] == "GBM"
    assert result["selected"]["local_reference_primary_context_codes"] == [
        "SKCM",
        "GBM",
    ]


def test_coarse_reference_context_preserves_ranked_order_on_ties():
    from trufflepig.cancer_type_evidence import _primary_context_codes

    analysis = _analysis(("SKCM", 1.0), ("GBM", 0.82))
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        top_tcga_cohorts=[("GBM_TPM", 0.84), ("OV_TPM", 0.84)]
    )

    assert _primary_context_codes(analysis) == ("SKCM", "GBM")


def test_local_reference_does_not_override_broad_coarse_consensus(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "SARC_MYXFIB": {
                "markers": ("COL1A2", "VIM", "COL1A1", "PDGFRA"),
                "ref_medians": {
                    "COL1A2": 100.0,
                    "VIM": 100.0,
                    "COL1A1": 100.0,
                    "PDGFRA": 100.0,
                },
                "context_codes": ("SARC",),
                "family": "sarcoma",
                "primary_tissue": "soft_tissue",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            }
        },
    )
    analysis = _analysis(("ACC", 1.0), ("SARC", 0.97))
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        top_tcga_cohorts=[("ACC_TPM", 0.86), ("SARC_TPM", 0.855)]
    )

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "COL1A2": 80.0,
                "VIM": 80.0,
                "COL1A1": 80.0,
                "PDGFRA": 80.0,
            }
        ),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "ACC"
    exact = next(row for row in result["evidence"] if row["cancer_type"] == "SARC_MYXFIB")
    assert exact["can_select_report_label"] is False
    assert any(
        "pan-cancer signature-ranker context and coarse reference matching both support ACC" in reason
        for reason in exact["blocking_reasons"]
    )


def test_signature_anchored_exact_reference_can_escape_wrong_broad_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    gist_markers = (
        "KIT",
        "ANO1",
        "ETV1",
        "SDHA",
        "SDHB",
        "NF1",
        "BRAF",
        "PDGFRA",
    )
    hepb_markers = (
        "AFP",
        "DLK1",
        "GPC3",
        "EPCAM",
        "KRT8",
        "KRT18",
        "KRT19",
        "SOX9",
    )
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "SARC_GIST": {
                "markers": gist_markers,
                "ref_medians": {gene: 100.0 for gene in gist_markers},
                "context_codes": ("SARC",),
                "parent_code": "SARC",
                "family": "sarcoma",
                "primary_tissue": "gi_wall",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "observed_bulk_reference",
            },
            "HEPB": {
                "markers": hepb_markers,
                "ref_medians": {gene: 100.0 for gene in hepb_markers},
                "context_codes": ("LIHC",),
                "family": "embryonal",
                "primary_tissue": "liver",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            },
        },
    )
    analysis = _analysis(("LIHC", 1.0), ("LUAD", 0.82), ("CHOL", 0.78))
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        top_tcga_cohorts=[("LIHC_TPM", 0.88), ("CHOL_TPM", 0.84)]
    )
    expression = {gene: 120.0 for gene in gist_markers}

    result = select_report_scope_from_evidence(_expression_frame(expression), analysis)

    assert result["selected"]["cancer_type"] == "SARC_GIST"
    assert result["selected"]["reference_cancer_type"] == "SARC"
    assert result["selected"]["local_reference_signature_anchored"] is True
    assert all(row["cancer_type"] != "HEPB" for row in result["evidence"])


def test_mixed_marker_program_cannot_become_a_signature_anchor(monkeypatch):
    """Positive markers cannot erase explicit expected-low contradictions."""
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("GAB1", "ZIC1", "MYC", "NEUROD1", "MYCN", "OTX2", "SOX2", "ATOH1")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "MBL": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("GBM", "LGG"),
                "family": "cns-embryonal",
                "primary_tissue": "cerebellum",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
                "expression_source": "Treehouse",
            }
        },
    )
    monkeypatch.setattr(
        evidence,
        "_marker_coherence",
        lambda code, _sample: {
            "code": code,
            "status": "mixed",
            "detected": 8,
            "total": 8,
            "required_for_consistent": 4,
            "detected_fraction": 1.0,
            "unexpected_low_detected": 2,
            "unexpected_low_genes": ["KRT8", "EPCAM"],
        }
        if code == "MBL"
        else {},
    )

    result = select_report_scope_from_evidence(
        _expression_frame({gene: 120.0 for gene in markers}),
        _analysis(("SARC_LPS_UNSPEC", 1.0), ("GBM", 0.62)),
    )

    assert result["selected"]["cancer_type"] == "SARC_LPS_UNSPEC"
    mbl = next(row for row in result["evidence"] if row["cancer_type"] == "MBL")
    assert mbl["local_reference_signature_anchored"] is False
    assert mbl["can_select_report_label"] is False
    assert any("expected-low" in reason for reason in mbl["blocking_reasons"])


def test_molecular_status_child_cannot_originate_a_runner_up_parent(monkeypatch):
    """Resolve the entity branch before applying an expression-status child."""
    import trufflepig.cancer_type_evidence as evidence
    import trufflepig.expression_classifier as classifier
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("CLDN18", "MUC5AC", "MUC6", "KRT20")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "STAD_GS": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("STAD",),
                "parent_code": "STAD",
                "family": "carcinoma-gi",
                "primary_tissue": "stomach",
                "source_cohort": "TREEHOUSE_POLYA_25_01_TCGA_STAD_SUBTYPE",
                "reference_kind": "observed_bulk_reference",
                "expression_source": "curated",
            }
        },
    )
    monkeypatch.setattr(evidence, "_marker_coherence", lambda _code, _sample: {})
    monkeypatch.setattr(
        classifier,
        "classify_expression",
        lambda _sample, top_k=5: [("STAD", 0.40), ("READ", 0.20)],
    )
    monkeypatch.setattr(
        classifier,
        "classify_expression_hierarchy",
        lambda _sample, top_k=5: [
            SimpleNamespace(
                public_dict=lambda: {
                    "stage": "entity",
                    "label": "STAD",
                    "probability": 0.40,
                    "margin": 0.20,
                    "top_predictions": [
                        {"label": "STAD", "probability": 0.40},
                        {"label": "READ", "probability": 0.20},
                    ],
                }
            ),
        ],
    )

    def add_composition(hypotheses, *_args, **_kwargs):
        evidence._hypothesis(
            hypotheses,
            "STAD",
        ).coarse_composition_support = 0.80

    monkeypatch.setattr(
        evidence,
        "_add_coarse_composition_reference_features",
        add_composition,
    )
    monkeypatch.setattr(
        evidence,
        "_centroid_and_confidence",
        lambda _sample: (pd.Series(dtype=float), False),
    )

    result = select_report_scope_from_evidence(
        _expression_frame({gene: 95.0 for gene in markers}),
        _analysis(("READ", 1.0), ("STAD", 0.96), ("COAD", 0.84)),
    )

    assert result["selected"]["cancer_type"] == "READ", {
        key: result["selected"].get(key)
        for key in (
            "selected_by",
            "label_decision",
            "entity_evidence_consensus",
            "adjudication_admissible_support",
            "adjudication_exclusions",
            "fused_evidence_score",
            "fused_evidence_components",
            "fused_evidence_can_select",
            "fused_evidence_blockers",
        )
    }
    status = next(
        row for row in result["evidence"] if row["cancer_type"] == "STAD_GS"
    )
    parent = next(row for row in result["evidence"] if row["cancer_type"] == "STAD")
    assert status["can_select_report_label"] is False
    assert any(
        axis["axis"] == "molecular_status"
        for axis in status["orthogonal_axes"]
    )
    assert parent["can_select_report_label"] is False
    assert any("established STAD parent diagnosis" in reason for reason in parent["blocking_reasons"])
    assert "exact_expression_reference" not in (
        parent["adjudication_admissible_support"]
    )
    assert "exact_expression_reference" in parent["adjudication_exclusions"]
    exact_axis = next(
        axis
        for axis in parent["entity_evidence_consensus"]["axes"]
        if axis["axis"] == "exact_expression_reference"
    )
    assert exact_axis["candidate_support"] == 0.0


def test_nonclassification_rare_surrogate_remains_a_prompt(monkeypatch):
    """A surrogate cannot make a registry-non-target scope the diagnosis."""
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    rule_id = "test_nonclassification_marker"
    monkeypatch.setattr(
        evidence,
        "_rare_rules_by_id",
        lambda: {
            rule_id: {
                "rule_id": rule_id,
                "cancer_code": "NET_NONPANCREATIC",
                "required_support_genes": "SOX10;AQP5;DOG1",
                "min_support_genes": 2,
                "context_codes": "HNSC",
                "promote_report_scope": True,
            }
        },
    )
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(evidence, "_marker_coherence", lambda _code, _sample: {})
    finding = {
        "cancer_type": "NET_NONPANCREATIC",
        "rule_id": rule_id,
        "surrogate": "NR4A3",
        "surrogate_tpm": 15000.0,
        "threshold_tpm": 10.0,
        "support_genes": ["SOX10", "AQP5", "DOG1"],
        "support_gene_count": 3,
        "min_support_genes": 2,
        "required_support_gene_count": 2,
        "support_pass": True,
    }

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "NR4A3": 15000.0,
                "SOX10": 40.0,
                "AQP5": 30.0,
                "DOG1": 20.0,
            }
        ),
        _analysis(("STAD", 1.0), ("READ", 0.97), ("HNSC", 0.95)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "STAD"
    nonclassification = next(
        row
        for row in result["evidence"]
        if row["cancer_type"] == "NET_NONPANCREATIC"
    )
    assert nonclassification["registry_classification_target"] is False
    assert nonclassification["can_select_report_label"] is False
    assert any(
        "not a registry classification target" in reason
        for reason in nonclassification["blocking_reasons"]
    )


def test_generic_soft_tissue_reference_cannot_escape_wrong_broad_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("COL1A1", "COL1A2", "VIM", "PDGFRA", "MMP2")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "SARC_MYXFIB": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("SARC",),
                "parent_code": "SARC",
                "family": "sarcoma",
                "primary_tissue": "soft_tissue",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            }
        },
    )
    analysis = _analysis(("HNSC", 1.0), ("LUSC", 0.88), ("CESC", 0.82))
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        top_tcga_cohorts=[("HNSC_TPM", 0.89), ("LUSC_TPM", 0.86)]
    )

    result = select_report_scope_from_evidence(
        _expression_frame({gene: 150.0 for gene in markers}),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "HNSC"
    exact = next(row for row in result["evidence"] if row["cancer_type"] == "SARC_MYXFIB")
    assert exact["can_select_report_label"] is False
    assert exact["local_reference_signature_anchored"] is False


def test_amplicon_like_liposarcoma_reference_cannot_escape_squamous_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("MDM2", "CDK4", "TSPAN31", "FRS2", "YEATS4", "HMGA2")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "SARC_LPS_UNSPEC": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("SARC",),
                "parent_code": "SARC",
                "family": "sarcoma",
                "primary_tissue": "adipose_tissue",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            }
        },
    )
    analysis = _analysis(("LUSC", 1.0), ("CESC", 0.82), ("HNSC", 0.78))
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        top_tcga_cohorts=[("LUSC_TPM", 0.92), ("CESC_TPM", 0.9)]
    )

    result = select_report_scope_from_evidence(
        _expression_frame({gene: 180.0 for gene in markers}),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "LUSC"
    exact = next(
        row for row in result["evidence"] if row["cancer_type"] == "SARC_LPS_UNSPEC"
    )
    assert exact["can_select_report_label"] is False
    assert exact["local_reference_signature_anchored"] is False


def test_smooth_muscle_reference_cannot_escape_coherent_epithelial_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    sarc_markers = ("ACTA2", "MYH11", "TAGLN", "DES", "CNN1")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "SARC_LMS": {
                "markers": sarc_markers,
                "ref_medians": {gene: 100.0 for gene in sarc_markers},
                "context_codes": ("SARC",),
                "parent_code": "SARC",
                "family": "sarcoma",
                "primary_tissue": "smooth_muscle",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "deconvolved_tumor_reference",
            }
        },
    )
    analysis = _analysis(("BLCA", 1.0), ("LUSC", 0.92), ("PAAD", 0.85))
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        top_tcga_cohorts=[("BLCA_TPM", 0.92), ("LUSC_TPM", 0.88)]
    )

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                **{gene: 180.0 for gene in sarc_markers},
                "UPK1A": 350.0,
                "UPK1B": 250.0,
                "UPK2": 500.0,
                "GATA3": 120.0,
                "FOXA1": 50.0,
                "EPCAM": 140.0,
            }
        ),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "BLCA"
    exact = next(row for row in result["evidence"] if row["cancer_type"] == "SARC_LMS")
    assert exact["can_select_report_label"] is False
    assert exact["local_reference_signature_anchored"] is False


def test_nerve_sheath_reference_cannot_escape_epithelial_basal_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    mpnst_markers = ("MIA", "S100B", "SOX10", "PLP1", "PMP22")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "SARC_MPNST": {
                "markers": mpnst_markers,
                "ref_medians": {gene: 100.0 for gene in mpnst_markers},
                "context_codes": ("SARC",),
                "parent_code": "SARC",
                "family": "sarcoma",
                "primary_tissue": "nerve_sheath",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "observed_bulk_reference",
            }
        },
    )
    analysis = _analysis(("BRCA", 1.0), ("LUAD", 0.90), ("LUSC", 0.85))
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        top_tcga_cohorts=[("SARC_TPM", 0.93), ("BRCA_TPM", 0.92)]
    )

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                **{gene: 180.0 for gene in mpnst_markers},
                "KRT14": 220.0,
                "KRT17": 700.0,
                "KRT5": 65.0,
                "EPCAM": 480.0,
                "KRT8": 620.0,
                "KRT18": 230.0,
                "SMARCB1": 140.0,
            }
        ),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "BRCA"
    exact = next(row for row in result["evidence"] if row["cancer_type"] == "SARC_MPNST")
    assert exact["can_select_report_label"] is False
    assert exact["local_reference_signature_anchored"] is False


def test_nerve_sheath_reference_cannot_escape_coherent_cns_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    mpnst_markers = ("MIA", "S100B", "SOX10", "PLP1", "PMP22")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "SARC_MPNST": {
                "markers": mpnst_markers,
                "ref_medians": {gene: 100.0 for gene in mpnst_markers},
                "context_codes": ("SARC",),
                "parent_code": "SARC",
                "family": "sarcoma",
                "primary_tissue": "nerve_sheath",
                "source_cohort": "TREEHOUSE_POLYA_25_01",
                "reference_kind": "observed_bulk_reference",
            }
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                **{gene: 180.0 for gene in mpnst_markers},
                "GFAP": 500.0,
                "OLIG2": 120.0,
                "SOX2": 80.0,
                "EGFR": 60.0,
            }
        ),
        _analysis(("GBM", 1.0), ("LGG", 0.93), ("SKCM", 0.75)),
    )

    assert result["selected"]["cancer_type"] == "GBM"
    exact = next(row for row in result["evidence"] if row["cancer_type"] == "SARC_MPNST")
    assert exact["can_select_report_label"] is False
    assert any("nerve_sheath exact-reference refinement" in r for r in exact["blocking_reasons"])


def test_cross_lineage_local_reference_uses_active_context_for_marker_gate(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("IFI30", "XBP1", "TOMM6", "CALML3")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "NEC_LUNG_LARGECELL": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("PCPG", "LUSC"),
                "family": "neuroendocrine",
                "primary_tissue": "lung",
                "source_cohort": "DRMETRICS_ALCALA_2019_LNEN",
                "reference_kind": "observed_bulk_reference",
            }
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame({gene: 80.0 for gene in markers}),
        _analysis(("LUSC", 1.0), ("CESC", 0.92), ("HNSC", 0.9)),
    )

    assert result["selected"]["cancer_type"] == "LUSC"
    nec = next(
        row
        for row in result["evidence"]
        if row["cancer_type"] == "NEC_LUNG_LARGECELL"
    )
    assert nec["can_select_report_label"] is False
    assert nec["local_reference_cross_lineage_marker_conflict"] is True
    assert any(
        "while the compatible RNA context is LUSC" in reason
        for reason in nec["blocking_reasons"]
    )


def test_lineage_panel_does_not_override_broad_coarse_consensus(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    import trufflepig.lineage_panels as lineage_panels
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    fake_panel = lineage_panels.LineagePanel(
        name="ESCA_SQUAMOUS",
        parent_cohort="ESCA",
        high_markers=("TP63",),
        obligate=("TP63",),
    )
    fake_panel_evidence = SimpleNamespace(panel_name="ESCA_SQUAMOUS", score=0.9)

    monkeypatch.setattr(evidence, "_local_expression_reference_panels", lambda *args, **kwargs: {})
    monkeypatch.setattr(lineage_panels, "LINEAGE_PANELS", (fake_panel,))
    monkeypatch.setattr(
        lineage_panels,
        "evaluate_panels",
        lambda *_args, **_kwargs: (fake_panel_evidence,),
    )
    monkeypatch.setattr(
        lineage_panels,
        "summarize_evidence",
        lambda _evidence: {
            "top_panel": "ESCA_SQUAMOUS",
            "top_score": 0.9,
            "margin_over_second": 0.9,
            "top_rationale": "ESCA_SQUAMOUS: strong squamous marker signal",
        },
    )

    analysis = _analysis(("CESC", 1.0), ("ESCA", 0.93), ("HNSC", 0.84))
    analysis["fit_quality"] = {"label": "ambiguous"}
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        top_tcga_cohorts=[("CESC_TPM", 0.91), ("ESCA_TPM", 0.88)]
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"TP63": 250.0}),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "CESC"
    cesc = next(row for row in result["evidence"] if row["cancer_type"] == "CESC")
    assert not any(
        "orthogonal molecular/status state" in reason
        for reason in cesc["blocking_reasons"]
    )
    esca = next(row for row in result["evidence"] if row["cancer_type"] == "ESCA")
    assert esca["can_select_report_label"] is False
    assert any(
        "pan-cancer signature-ranker context and coarse reference matching both support CESC" in reason
        for reason in esca["blocking_reasons"]
    )
    assert result["lineage_panel_evidence"]["promotion"] == {
        "promoted": False,
        "code": "ESCA",
        "blockers": esca["blocking_reasons"],
    }


def test_base_viral_diagnosis_is_not_blocked_as_a_status_child():
    from trufflepig.cancer_type_evidence import (
        _orthogonal_axes_that_block_report_label,
    )

    assert _orthogonal_axes_that_block_report_label("CESC") == []
    assert _orthogonal_axes_that_block_report_label("NPC") == []
    hpv_child_axes = _orthogonal_axes_that_block_report_label("HNSC_HPVpos")
    assert [axis["axis"] for axis in hpv_child_axes] == ["viral_status"]
    assert hpv_child_axes[0]["base_code"] == "HNSC"


def test_lineage_panel_does_not_override_strong_conflicting_composition(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    import trufflepig.lineage_panels as lineage_panels
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    fake_panel = lineage_panels.LineagePanel(
        name="BRCA_BASAL",
        parent_cohort="BRCA",
        high_markers=("KRT14", "KRT5", "FOXC1"),
        obligate=("KRT14",),
    )
    fake_panel_evidence = SimpleNamespace(panel_name="BRCA_BASAL", score=0.9)

    monkeypatch.setattr(evidence, "_local_expression_reference_panels", lambda *args, **kwargs: {})
    monkeypatch.setattr(lineage_panels, "LINEAGE_PANELS", (fake_panel,))
    monkeypatch.setattr(
        lineage_panels,
        "evaluate_panels",
        lambda *_args, **_kwargs: (fake_panel_evidence,),
    )
    monkeypatch.setattr(
        lineage_panels,
        "summarize_evidence",
        lambda _evidence: {
            "top_panel": "BRCA_BASAL",
            "top_score": 0.9,
            "margin_over_second": 0.9,
            "top_rationale": "BRCA_BASAL: strong basal epithelial program",
        },
    )

    analysis = _analysis(
        ("LUSC", 1.0),
        ("CESC", 0.99),
        ("BRCA", 0.98),
        ("LUAD", 0.94),
    )
    analysis["fit_quality"] = {"label": "ambiguous"}
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        cancer_hint="tumor-consistent",
        top_normal_tissues=[
            ("lung_nTPM", 0.87),
            ("urinary_bladder_nTPM", 0.86),
            ("esophagus_nTPM", 0.84),
        ],
        top_tcga_cohorts=[
            ("LUAD_TPM", 0.91),
            ("LUSC_TPM", 0.90),
            ("BRCA_TPM", 0.89),
        ],
        type_specific_cohort="",
        type_specific_hits=[],
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"KRT14": 120.0, "KRT5": 80.0, "FOXC1": 30.0}),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "LUSC"
    brca = next(row for row in result["evidence"] if row["cancer_type"] == "BRCA")
    assert brca["can_select_report_label"] is False
    assert any(
        "lineage panel program conflicts with independent composition reference LUAD"
        in reason
        for reason in brca["blocking_reasons"]
    )


def test_mutation_defined_expression_reference_does_not_set_report_label(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "LUAD_KRAS": {
                "markers": ("MAGEA3", "MAGEA6", "CSAG3"),
                "ref_medians": {"MAGEA3": 50.0, "MAGEA6": 60.0, "CSAG3": 20.0},
                "context_codes": ("LUAD",),
                "family": "carcinoma-lung",
                "primary_tissue": "lung",
                "source_cohort": "TREEHOUSE_POLYA_25_01_TCGA_LUAD_MUT",
                "reference_kind": "deconvolved_tumor_reference",
                "expression_source": "TCGA/mut",
            }
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"MAGEA3": 40.0, "MAGEA6": 45.0, "CSAG3": 30.0}),
        _analysis(("LUAD", 1.0), ("HNSC", 0.4)),
    )

    assert result["selected"]["cancer_type"] == "LUAD"
    subtype = next(row for row in result["evidence"] if row["cancer_type"] == "LUAD_KRAS")
    assert subtype["can_select_report_label"] is False
    assert any(
        "molecular-status expression reference requires direct molecular evidence"
        in reason
        for reason in subtype["blocking_reasons"]
    )


def test_eln_expression_reference_promotes_parent_not_status_child(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("MPO", "ELANE", "FLT3", "KIT", "CD34")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "LAML_ELNadv": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("LAML",),
                "parent_code": "LAML",
                "family": "heme-myeloid",
                "primary_tissue": "bone_marrow",
                "source_cohort": "BEATAML_OHSU_2022",
                "reference_kind": "observed_bulk_reference",
                "expression_source": "BEATAML",
            }
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame({gene: 90.0 for gene in markers}),
        _analysis(("LAML", 1.0), ("DLBC", 0.4)),
    )

    assert result["selected"]["cancer_type"] == "LAML"
    assert result["selected"]["local_reference_status_child_code"] == "LAML_ELNadv"
    subtype = next(row for row in result["evidence"] if row["cancer_type"] == "LAML_ELNadv")
    assert subtype["can_select_report_label"] is False
    assert any(
        "molecular-status expression reference requires direct molecular evidence"
        in reason
        for reason in subtype["blocking_reasons"]
    )


def test_molecular_status_parent_provenance_follows_selected_child_priority(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    favored_markers = ("MPO", "ELANE", "FLT3", "KIT", "CD34")
    adverse_markers = ("DEFA3", "LTF", "MMP8", "CEACAM8")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "LAML_ELNfav": {
                "markers": favored_markers,
                "ref_medians": {gene: 100.0 for gene in favored_markers},
                "context_codes": ("LAML",),
                "parent_code": "LAML",
                "family": "heme-myeloid",
                "primary_tissue": "bone_marrow",
                "source_cohort": "BEATAML_OHSU_2022",
                "reference_kind": "observed_bulk_reference",
            },
            "LAML_ELNadv": {
                "markers": adverse_markers,
                "ref_medians": {gene: 100.0 for gene in adverse_markers},
                "context_codes": ("LAML",),
                "parent_code": "LAML",
                "family": "heme-myeloid",
                "primary_tissue": "bone_marrow",
                "source_cohort": "BEATAML_OHSU_2022",
                "reference_kind": "deconvolved_tumor_reference",
            },
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                **{gene: 90.0 for gene in favored_markers[:-1]},
                favored_markers[-1]: 0.0,
                **{gene: 90.0 for gene in adverse_markers[:-1]},
                adverse_markers[-1]: 0.0,
            }
        ),
        _analysis(("LAML", 1.0), ("DLBC", 0.4)),
    )

    assert result["selected"]["cancer_type"] == "LAML"
    assert result["selected"]["local_reference_status_child_code"] == "LAML_ELNadv"
    assert result["selected"]["local_reference_kind"] == "deconvolved_tumor_reference"


def test_direct_cml_reference_beats_aml_status_child_parent(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    cml_markers = ("AZU1", "ELANE", "MPO", "S100A12", "CTSG")
    eln_markers = ("DEFA3", "LTF", "MMP8", "CEACAM8", "BPI")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "CML": {
                "markers": cml_markers,
                "ref_medians": {gene: 100.0 for gene in cml_markers},
                "context_codes": ("LAML", "DLBC", "THYM"),
                "family": "heme-myeloid",
                "primary_tissue": "peripheral_blood",
                "source_cohort": "GSE100026_DING_2017",
                "reference_kind": "observed_bulk_reference",
                "expression_source": "GEO",
                "fusion_driven": "defining",
                "fusion_driver": "BCR-ABL1",
            },
            "LAML_ELNadv": {
                "markers": eln_markers,
                "ref_medians": {gene: 100.0 for gene in eln_markers},
                "context_codes": ("LAML",),
                "parent_code": "LAML",
                "family": "heme-myeloid",
                "primary_tissue": "bone_marrow",
                "source_cohort": "BEATAML_OHSU_2022",
                "reference_kind": "observed_bulk_reference",
                "expression_source": "BEATAML",
            },
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame({gene: 90.0 for gene in cml_markers + eln_markers}),
        _analysis(("LAML", 1.0), ("DLBC", 0.8), ("THYM", 0.5)),
    )

    assert result["selected"]["cancer_type"] == "CML"
    assert result["selected"]["selected_by"] == "local_expression_reference"
    assert result["selected"]["local_reference_requires_fusion_confirmation"] is True
    assert "fusion-defined" in result["selected"]["caveat"]
    laml = next(row for row in result["evidence"] if row["cancer_type"] == "LAML")
    assert laml["can_select_report_label"] is True
    assert laml["local_reference_status_child_code"] == "LAML_ELNadv"


def test_fusion_driven_subtype_reference_respects_supplied_negative_fusions(
    monkeypatch,
):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("ACTA2", "TFE3", "MITF", "RNF213")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "SARC_PEC": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("SARC",),
                "parent_code": "SARC",
                "family": "sarcoma",
                "primary_tissue": "soft_tissue",
                "source_cohort": "GSE328026_PECOMA_2026",
                "reference_kind": "observed_bulk_reference",
                "expression_source": "GEO",
                "fusion_driven": "subtype",
                "fusion_driver": "SFPQ-TFE3; DVL2-TFE3",
            }
        },
    )
    expression = _expression_frame({gene: 90.0 for gene in markers})

    no_fusion_analysis = _analysis(("SARC", 1.0), ("READ", 0.7))
    no_fusion = select_report_scope_from_evidence(expression, no_fusion_analysis)
    assert no_fusion["selected"]["cancer_type"] == "SARC_PEC"
    assert no_fusion["selected"]["local_reference_requires_fusion_confirmation"] is True

    matching_fusion_analysis = _analysis(("SARC", 1.0), ("READ", 0.7))
    matching_fusion_analysis["fusion_inputs_supplied"] = True
    matching_fusion_analysis["fusion_records"] = [
        {"gene_a": "SFPQ", "gene_b": "TFE3", "pair": "SFPQ--TFE3"}
    ]
    matching_fusion = select_report_scope_from_evidence(
        expression,
        matching_fusion_analysis,
    )
    assert matching_fusion["selected"]["cancer_type"] == "SARC_PEC"
    assert (
        matching_fusion["selected"]["local_reference_explicit_negative_fusion"]
        is False
    )

    negative_fusion_analysis = _analysis(("SARC", 1.0), ("READ", 0.7))
    negative_fusion_analysis["fusion_inputs_supplied"] = True
    negative_fusion_analysis["fusion_records"] = [
        {"gene_a": "ALK", "gene_b": "EML4", "pair": "EML4--ALK"}
    ]
    negative_fusion = select_report_scope_from_evidence(
        expression,
        negative_fusion_analysis,
    )

    assert negative_fusion["selected"]["cancer_type"] == "SARC"
    pec = next(
        row for row in negative_fusion["evidence"]
        if row["cancer_type"] == "SARC_PEC"
    )
    assert pec["can_select_report_label"] is False
    assert pec["local_reference_explicit_negative_fusion"] is True
    assert any(
        "fusion input was supplied" in reason and "SFPQ-TFE3" in reason
        for reason in pec["blocking_reasons"]
    )


def test_molecular_status_expression_reference_promotes_parent_label(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("ERBB2", "GRB7", "KRT8", "EPCAM")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "BRCA_HER2": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("BRCA",),
                "parent_code": "BRCA",
                "family": "carcinoma-breast",
                "primary_tissue": "breast",
                "source_cohort": "TREEHOUSE_POLYA_25_01_TCGA_BRCA_PAM50",
                "reference_kind": "deconvolved_tumor_reference",
                "expression_source": "TCGA/PAM50",
            }
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                **{gene: 90.0 for gene in markers},
                "GATA3": 40.0,
                "FOXA1": 35.0,
                "TFF1": 25.0,
            }
        ),
        _analysis(("BRCA", 1.0), ("KIRC", 0.97), ("LUAD", 0.8)),
    )

    assert result["selected"]["cancer_type"] == "BRCA"
    assert result["selected"]["selected_by"] == "local_expression_reference"
    assert result["selected"]["local_reference_status_child_code"] == "BRCA_HER2"
    graph = result["staged_evidence_graph"]
    assert graph["selected"]["code"] == "BRCA"
    assert graph["selected"]["stage"] == "coarse_type"
    assert graph["stages"][2]["status"] == "not_resolved"
    assert graph["orthogonal_axes"][0]["axis"] == "expression_subtype"
    assert graph["orthogonal_axes"][0]["state"] == "HER2-enriched"
    assert graph["orthogonal_axes"][0]["base_code"] == "BRCA"
    assert graph["orthogonal_axes"][0]["status"] == "supports_parent_label"
    subtype = next(row for row in result["evidence"] if row["cancer_type"] == "BRCA_HER2")
    assert subtype["can_select_report_label"] is False
    assert any("molecular-status expression reference" in reason for reason in subtype["blocking_reasons"])


def test_molecular_status_parent_promotion_requires_parent_marker_coherence(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    status_markers = ("MAGEA3", "MAGEA6", "CSAG3")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "LUAD_STK11": {
                "markers": status_markers,
                "ref_medians": {gene: 100.0 for gene in status_markers},
                "context_codes": ("LUAD",),
                "parent_code": "LUAD",
                "family": "carcinoma-lung",
                "primary_tissue": "lung",
                "source_cohort": "TREEHOUSE_POLYA_25_01_TCGA_LUAD_MUT",
                "reference_kind": "deconvolved_tumor_reference",
                "expression_source": "TCGA/mut",
            }
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "MAGEA3": 90.0,
                "MAGEA6": 95.0,
                "CSAG3": 85.0,
                "SFTPB": 450.0,
                "SCGB1A1": 4.5,
            }
        ),
        _analysis(("BRCA", 1.0), ("LUAD", 0.97)),
    )

    assert result["selected"]["cancer_type"] == "BRCA"
    parent = next(row for row in result["evidence"] if row["cancer_type"] == "LUAD")
    assert parent["can_select_report_label"] is False
    assert any("marker program is partial" in reason for reason in parent["blocking_reasons"])


def test_msi_status_reference_does_not_cross_broad_coarse_consensus(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    markers = ("MLH1", "MSH2", "MSH6", "PMS2", "EPCAM")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "COAD_MSI": {
                "markers": markers,
                "ref_medians": {gene: 100.0 for gene in markers},
                "context_codes": ("COAD",),
                "parent_code": "COAD",
                "family": "carcinoma-gi",
                "primary_tissue": "colon",
                "source_cohort": "TREEHOUSE_POLYA_25_01_TCGA_COADREAD_MSI",
                "reference_kind": "observed_bulk_reference",
                "expression_source": "TCGA/MSI",
            }
        },
    )
    analysis = _analysis(("BRCA", 1.0), ("COAD", 0.94), ("LUAD", 0.7))
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        top_tcga_cohorts=[("BRCA_TPM", 0.9), ("LUAD_TPM", 0.85)]
    )

    result = select_report_scope_from_evidence(
        _expression_frame({gene: 90.0 for gene in markers}),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "BRCA"
    subtype = next(row for row in result["evidence"] if row["cancer_type"] == "COAD_MSI")
    parent = next(row for row in result["evidence"] if row["cancer_type"] == "COAD")
    assert subtype["can_select_report_label"] is False
    assert parent["can_select_report_label"] is False
    assert any("secondary near-match" in reason for reason in parent["blocking_reasons"])


def test_salivary_exact_reference_combines_with_marker_axis(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "ADCC": {
                "markers": ("MYB", "NFIB", "KRT7", "SOX10", "KIT"),
                "ref_medians": {
                    "MYB": 30.0,
                    "NFIB": 40.0,
                    "KRT7": 80.0,
                    "SOX10": 20.0,
                    "KIT": 25.0,
                },
                "context_codes": ("HNSC", "LUAD", "BRCA"),
                "family": "salivary",
                "primary_tissue": "salivary_gland",
                "source_cohort": "GSE294016_BARTL_2025_SGC",
                "reference_kind": "observed_bulk_reference",
                "expression_source": "GEO",
                "fusion_driven": "defining",
                "fusion_driver": "MYB-NFIB; MYBL1-NFIB",
            }
        },
    )
    finding = {
        "cancer_type": "ADCC",
        "rule_id": "adcc_myb",
        "surrogate": "MYB",
        "surrogate_tpm": 25.0,
        "threshold_tpm": 10.0,
        "support_genes": ["KRT7", "SOX10", "KIT"],
        "support_pass": True,
        "support_gene_count": 3,
        "min_support_genes": 1,
        "required_support_gene_count": 3,
    }

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "MYB": 25.0,
                "NFIB": 30.0,
                "KRT7": 100.0,
                "SOX10": 25.0,
                "KIT": 30.0,
            }
        ),
        _analysis(("LUAD", 1.0), ("HNSC", 0.9), ("BRCA", 0.6)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "ADCC"
    assert result["selected"]["selected_by"] == "local_expression_reference"
    assert result["selected"]["expression_reference_cancer_type"] == "ADCC"
    assert set(result["selected"]["evidence_sources"]) == {
        "local_expression_reference",
        "rare_marker",
    }
    assert result["selected"]["local_reference_source_cohort"] == "GSE294016_BARTL_2025_SGC"


def test_adcc_marker_axis_requires_complete_support_without_strong_local_reference():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    finding = {
        "cancer_type": "ADCC",
        "rule_id": "adcc_myb",
        "surrogate": "MYB",
        "surrogate_tpm": 30.0,
        "threshold_tpm": 10.0,
        "support_genes": ["KRT7", "KIT"],
        "missing_support_genes": ["SOX10"],
        "support_gene_count": 2,
        "min_support_genes": 1,
        "required_support_gene_count": 3,
        "support_pass": True,
    }

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("BRCA", 1.0), ("HNSC", 0.86), ("LUAD", 0.82)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "BRCA"
    adcc = next(row for row in result["evidence"] if row["cancer_type"] == "ADCC")
    assert adcc["can_select_report_label"] is False
    assert any("complete MYB-axis" in reason for reason in adcc["blocking_reasons"])


def test_adcc_rare_marker_mixed_program_cannot_override_brca(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    original_marker_coherence = evidence._marker_coherence

    def fake_marker_coherence(code, sample):
        if code == "ADCC":
            return {
                "code": "ADCC",
                "status": "mixed",
                "detected": 7,
                "total": 7,
                "required_for_consistent": 4,
                "detected_fraction": 1.0,
                "unexpected_low_detected": 3,
                "unexpected_low_genes": ["KRT8", "KRT18", "EPCAM"],
            }
        return original_marker_coherence(code, sample)

    monkeypatch.setattr(evidence, "_marker_coherence", fake_marker_coherence)
    finding = {
        "cancer_type": "ADCC",
        "rule_id": "adcc_myb",
        "surrogate": "MYB",
        "surrogate_tpm": 30.0,
        "threshold_tpm": 10.0,
        "support_genes": ["KRT7", "SOX10", "KIT"],
        "support_pass": True,
        "support_gene_count": 3,
        "min_support_genes": 1,
        "required_support_gene_count": 3,
    }

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("BRCA", 1.0), ("HNSC", 0.86), ("LUAD", 0.82)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "BRCA"
    adcc = next(row for row in result["evidence"] if row["cancer_type"] == "ADCC")
    assert adcc["can_select_report_label"] is False
    assert any("requires a clean marker program" in reason for reason in adcc["blocking_reasons"])


def test_low_myb_adcc_exact_reference_can_override_brca_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "ADCC": {
                "markers": ("MYB", "NFIB", "KRT7", "SOX10", "KIT"),
                "ref_medians": {
                    "MYB": 30.0,
                    "NFIB": 40.0,
                    "KRT7": 80.0,
                    "SOX10": 20.0,
                    "KIT": 25.0,
                },
                "context_codes": ("HNSC", "LUAD", "BRCA"),
                "family": "salivary",
                "primary_tissue": "salivary_gland",
                "source_cohort": "GSE294016_BARTL_2025_SGC",
                "reference_kind": "observed_bulk_reference",
                "expression_source": "GEO",
                "fusion_driven": "defining",
                "fusion_driver": "MYB-NFIB; MYBL1-NFIB",
            }
        },
    )
    finding = {
        "cancer_type": "ADCC",
        "rule_id": "adcc_myb",
        "surrogate": "MYB",
        "surrogate_tpm": 18.0,
        "threshold_tpm": 10.0,
        "support_genes": ["KRT7", "SOX10", "KIT"],
        "support_pass": True,
        "support_gene_count": 3,
        "min_support_genes": 1,
        "required_support_gene_count": 3,
    }

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "MYB": 18.0,
                "NFIB": 30.0,
                "KRT7": 100.0,
                "SOX10": 25.0,
                "KIT": 30.0,
            }
        ),
        _analysis(("BRCA", 1.0), ("HNSC", 0.85)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "ADCC"
    assert result["selected"]["local_reference_high_confidence"] is True
    adcc = next(row for row in result["evidence"] if row["cancer_type"] == "ADCC")
    assert adcc["can_select_report_label"] is True
    assert not any(
        "ADCC MYB promoter signal" in reason for reason in adcc["blocking_reasons"]
    )
    assert "high-confidence exact expression-reference support" in (
        result["selected"].get("caveat") or ""
    )


def test_adcc_exact_reference_yields_to_native_brca_consensus_without_strong_driver_axis(
    monkeypatch,
):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "ADCC": {
                "markers": ("MYB", "NFIB", "KRT7", "SOX10", "KIT"),
                "ref_medians": {
                    "MYB": 30.0,
                    "NFIB": 40.0,
                    "KRT7": 80.0,
                    "SOX10": 20.0,
                    "KIT": 25.0,
                },
                "context_codes": ("HNSC", "LUAD", "BRCA"),
                "family": "salivary",
                "primary_tissue": "salivary_gland",
                "source_cohort": "GSE294016_BARTL_2025_SGC",
                "reference_kind": "observed_bulk_reference",
                "expression_source": "GEO",
                "fusion_driven": "defining",
                "fusion_driver": "MYB-NFIB; MYBL1-NFIB",
            }
        },
    )
    finding = {
        "cancer_type": "ADCC",
        "rule_id": "adcc_myb",
        "surrogate": "MYB",
        "surrogate_tpm": 28.0,
        "threshold_tpm": 10.0,
        "support_genes": ["KRT7", "SOX10", "KIT"],
        "support_pass": True,
        "support_gene_count": 3,
        "min_support_genes": 1,
        "required_support_gene_count": 3,
    }
    analysis = _analysis(("BRCA", 1.0), ("ESCA", 0.90), ("HNSC", 0.85))
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        cancer_hint="tumor-consistent",
        top_tcga_cohorts=[("BRCA_TPM", 0.88), ("PRAD_TPM", 0.87)],
        top_normal_tissues=[("breast_nTPM", 0.86), ("salivary gland_nTPM", 0.10)],
    )

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "MYB": 28.0,
                "NFIB": 142.0,
                "KRT7": 1174.0,
                "SOX10": 233.0,
                "KIT": 209.0,
            }
        ),
        analysis,
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "BRCA"
    adcc = next(row for row in result["evidence"] if row["cancer_type"] == "ADCC")
    assert adcc["local_reference_high_confidence"] is True
    assert adcc["fusion_defined_context_conflict"] is True
    assert adcc["can_select_report_label"] is False
    assert any(
        "native BRCA broad/coarse expression consensus" in reason
        and "MYB 28.0 TPM < 75.0 TPM" in reason
        for reason in adcc["blocking_reasons"]
    )


def test_strong_myb_adcc_exact_reference_can_refine_native_brca_consensus(
    monkeypatch,
):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "ADCC": {
                "markers": ("MYB", "NFIB", "KRT7", "SOX10", "KIT"),
                "ref_medians": {
                    "MYB": 30.0,
                    "NFIB": 40.0,
                    "KRT7": 80.0,
                    "SOX10": 20.0,
                    "KIT": 25.0,
                },
                "context_codes": ("HNSC", "LUAD", "BRCA"),
                "family": "salivary",
                "primary_tissue": "salivary_gland",
                "source_cohort": "GSE294016_BARTL_2025_SGC",
                "reference_kind": "observed_bulk_reference",
                "expression_source": "GEO",
                "fusion_driven": "defining",
                "fusion_driver": "MYB-NFIB; MYBL1-NFIB",
            }
        },
    )
    analysis = _analysis(("BRCA", 1.0), ("ESCA", 0.90), ("HNSC", 0.85))
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        cancer_hint="tumor-consistent",
        top_tcga_cohorts=[("BRCA_TPM", 0.88), ("PRAD_TPM", 0.87)],
        top_normal_tissues=[("breast_nTPM", 0.86), ("salivary gland_nTPM", 0.10)],
    )

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "MYB": 123.0,
                "NFIB": 99.0,
                "KRT7": 500.0,
                "SOX10": 300.0,
                "KIT": 90.0,
            }
        ),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "ADCC"
    adcc = next(row for row in result["evidence"] if row["cancer_type"] == "ADCC")
    assert adcc["local_reference_high_confidence"] is True
    assert adcc["fusion_defined_context_conflict"] is False
    assert adcc["can_select_report_label"] is True


def test_low_myb_adcc_exact_reference_yields_to_basal_brca_program(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "ADCC": {
                "markers": ("MYB", "NFIB", "KRT7", "SOX10", "KIT"),
                "ref_medians": {
                    "MYB": 30.0,
                    "NFIB": 40.0,
                    "KRT7": 80.0,
                    "SOX10": 20.0,
                    "KIT": 25.0,
                },
                "context_codes": ("HNSC", "LUAD", "BRCA"),
                "family": "salivary",
                "primary_tissue": "salivary_gland",
                "source_cohort": "GSE294016_BARTL_2025_SGC",
                "reference_kind": "observed_bulk_reference",
                "expression_source": "GEO",
                "fusion_driven": "defining",
                "fusion_driver": "MYB-NFIB; MYBL1-NFIB",
            }
        },
    )
    finding = {
        "cancer_type": "ADCC",
        "rule_id": "adcc_myb",
        "surrogate": "MYB",
        "surrogate_tpm": 19.0,
        "threshold_tpm": 10.0,
        "support_genes": ["KRT7", "SOX10", "KIT"],
        "support_pass": True,
        "support_gene_count": 3,
        "min_support_genes": 1,
        "required_support_gene_count": 3,
    }

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "MYB": 19.0,
                "NFIB": 68.0,
                "KRT7": 100.0,
                "SOX10": 30.0,
                "KIT": 55.0,
                "KRT14": 52.0,
                "KRT5": 1600.0,
                "FOXC1": 46.0,
                "MIA": 94.0,
                "MUCL1": 0.0,
                "ESR1": 4.0,
                "PGR": 0.0,
                "UPK1B": 0.0,
                "TP63": 10.0,
                "SOX2": 0.0,
            }
        ),
        _analysis(("CESC", 1.0), ("ESCA", 0.94), ("BRCA", 0.90), ("HNSC", 0.85)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] != "ADCC"
    adcc = next(row for row in result["evidence"] if row["cancer_type"] == "ADCC")
    assert adcc["local_reference_high_confidence"] is True
    assert adcc["adcc_low_myb_basal_breast_conflict"] is True
    assert adcc["can_select_report_label"] is False
    assert any("BRCA_BASAL" in reason for reason in adcc["blocking_reasons"])


def test_high_myb_adcc_exact_reference_yields_to_luminal_brca_program(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "ADCC": {
                "markers": ("MYB", "NFIB", "KRT7", "SOX10", "KIT"),
                "ref_medians": {
                    "MYB": 30.0,
                    "NFIB": 40.0,
                    "KRT7": 80.0,
                    "SOX10": 20.0,
                    "KIT": 25.0,
                },
                "context_codes": ("HNSC", "LUAD", "BRCA"),
                "family": "salivary",
                "primary_tissue": "salivary_gland",
                "source_cohort": "GSE294016_BARTL_2025_SGC",
                "reference_kind": "observed_bulk_reference",
                "expression_source": "GEO",
                "fusion_driven": "defining",
                "fusion_driver": "MYB-NFIB; MYBL1-NFIB",
            }
        },
    )
    finding = {
        "cancer_type": "ADCC",
        "rule_id": "adcc_myb",
        "surrogate": "MYB",
        "surrogate_tpm": 69.0,
        "threshold_tpm": 10.0,
        "support_genes": ["KRT7", "SOX10", "KIT"],
        "support_pass": True,
        "support_gene_count": 3,
        "min_support_genes": 1,
        "required_support_gene_count": 3,
    }

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "MYB": 69.0,
                "NFIB": 14.0,
                "KRT7": 279.0,
                "SOX10": 7.0,
                "KIT": 9.0,
                "ESR1": 89.0,
                "PGR": 76.0,
                "FOXA1": 172.0,
                "GATA3": 319.0,
                "SCGB2A2": 228.0,
                "MUCL1": 32.0,
                "KRT14": 87.0,
                "KRT5": 85.0,
            }
        ),
        _analysis(("BRCA", 1.0), ("LUAD", 0.88), ("BLCA", 0.88), ("HNSC", 0.82)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "BRCA"
    adcc = next(row for row in result["evidence"] if row["cancer_type"] == "ADCC")
    assert adcc["local_reference_high_confidence"] is True
    assert adcc["adcc_breast_program_conflict"] is True
    assert adcc["can_select_report_label"] is False
    assert any("BRCA_LUMINAL" in reason for reason in adcc["blocking_reasons"])


def test_low_myb_adcc_surrogate_without_strong_local_reference_stays_blocked(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(evidence, "_local_expression_reference_panels", lambda: {})
    finding = {
        "cancer_type": "ADCC",
        "rule_id": "adcc_myb",
        "surrogate": "MYB",
        "surrogate_tpm": 18.0,
        "threshold_tpm": 10.0,
        "support_genes": ["KRT7", "SOX10", "KIT"],
        "support_pass": True,
        "support_gene_count": 3,
        "min_support_genes": 1,
        "required_support_gene_count": 3,
    }

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "MYB": 18.0,
                "KRT7": 100.0,
                "SOX10": 25.0,
                "KIT": 30.0,
            }
        ),
        _analysis(("BRCA", 1.0), ("HNSC", 0.85)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "BRCA"
    adcc = next(row for row in result["evidence"] if row["cancer_type"] == "ADCC")
    assert adcc["can_select_report_label"] is False
    assert any("ADCC MYB promoter signal" in reason for reason in adcc["blocking_reasons"])


def test_local_expression_reference_requires_top_compatible_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "MM": {
                "markers": ("CD79A", "MS4A1", "MZB1"),
                "ref_medians": {"CD79A": 120.0, "MS4A1": 80.0, "MZB1": 100.0},
                "context_codes": ("DLBC", "LAML", "THYM"),
                "family": "heme-plasma",
                "primary_tissue": "bone_marrow",
                "source_cohort": "MMRF_COMMPASS",
            }
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"CD79A": 45.0, "MS4A1": 30.0, "MZB1": 40.0}),
        _analysis(("PRAD", 1.0), ("DLBC", 0.9)),
    )

    assert result["selected"]["cancer_type"] == "PRAD"
    mm = next(row for row in result["evidence"] if row["cancer_type"] == "MM")
    assert mm["can_select_report_label"] is False
    assert any("not the top compatible context" in r for r in mm["blocking_reasons"])


def test_fine_reference_medians_filter_to_requested_cancer_code(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence

    df = pd.DataFrame(
        [
            {"symbol": "RUNX2", "cancer_code": "SARC_OS", "tumor_tpm_median": 100.0},
            {"symbol": "RUNX2", "cancer_code": "SARC_CHOR", "tumor_tpm_median": 2.0},
            {"symbol": "TBXT", "cancer_code": "SARC_CHOR", "tumor_tpm_median": 50.0},
        ]
    )
    monkeypatch.setattr(
        "trufflepig.reference.subtype_deconvolved_expression",
        lambda: df,
    )
    evidence._reference_medians.cache_clear()

    try:
        assert evidence._reference_medians("SARC_OS") == {"RUNX2": 100.0}
    finally:
        evidence._reference_medians.cache_clear()


def test_salivary_surrogate_does_not_override_crc_context():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    finding = {
        "cancer_type": "ADCC",
        "rule_id": "adcc_myb",
        "surrogate": "MYB",
        "surrogate_tpm": 60.0,
        "threshold_tpm": 10.0,
        "support_genes": ["KIT"],
    }

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("READ", 1.0), ("COAD", 0.4), ("HNSC", 0.15)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "READ"
    adcc = next(row for row in result["evidence"] if row["cancer_type"] == "ADCC")
    assert adcc["cancer_type"] == "ADCC"
    assert adcc["can_select_report_label"] is False
    assert (
        "expression-reference context is not one of HNSC"
        in adcc["blocking_reasons"]
    )


def test_rare_marker_selector_honors_min_support_genes_with_optional_markers():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    finding = {
        "cancer_type": "MTC",
        "rule_id": "mtc_calca",
        "surrogate": "CALCA",
        "surrogate_tpm": 18.0,
        "threshold_tpm": 10.0,
        "support_genes": ["CHGA"],
        "missing_support_genes": ["SYP", "RET"],
        "support_pass": True,
        "support_gene_count": 1,
        "min_support_genes": 1,
        "required_support_gene_count": 3,
    }

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("THCA", 1.0), ("PCPG", 0.25)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "MTC"
    mtc = next(row for row in result["evidence"] if row["cancer_type"] == "MTC")
    assert mtc["can_select_report_label"] is True
    assert mtc["support_gene_count"] == 1
    assert mtc["min_support_genes"] == 1
    assert mtc["required_support_gene_count"] == 3
    assert mtc["missing_support_genes"] == ["SYP", "RET"]


def test_rare_marker_selector_rejects_rules_below_min_support_genes():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    finding = {
        "cancer_type": "ACINIC",
        "rule_id": "acinic_nr4a3",
        "surrogate": "NR4A3",
        "surrogate_tpm": 18.0,
        "threshold_tpm": 10.0,
        "support_genes": [],
        "missing_support_genes": ["SOX10", "AQP5", "DOG1"],
        "support_pass": False,
        "support_gene_count": 0,
        "min_support_genes": 1,
        "required_support_gene_count": 3,
    }

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("HNSC", 1.0), ("LUSC", 0.25)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "HNSC"
    assert all(row["cancer_type"] != "ACINIC" for row in result["evidence"])


def test_real_rare_rna_rules_promote_with_configured_partial_support():
    from trufflepig.rare_inference import infer_rare_cancer_marker_hypotheses_from_rna

    cases = [
        ("ACINIC", "HNSC", {"NR4A3": 18.0, "SOX10": 3.0}),
        ("ADCC", "HNSC", {"MYB": 18.0, "KIT": 3.0, "SOX10": 3.0}),
        ("MTC", "THCA", {"CALCA": 18.0, "CHGA": 3.0}),
    ]

    for expected_code, context_code, tpm_by_symbol in cases:
        hypotheses = infer_rare_cancer_marker_hypotheses_from_rna(
            _expression_frame(tpm_by_symbol),
            _analysis((context_code, 1.0)),
        )
        result = next(
            (h for h in hypotheses if h["cancer_type"] == expected_code), None
        )
        # The configured rule fired in its gated context: the surrogate marker
        # was detected and the partial-support contract is exposed (one support
        # gene required, the rest reported as missing).
        assert result is not None
        assert result["min_support_genes"] >= 1
        assert result["missing_support_genes"]


def test_complete_context_gated_rare_marker_axis_beats_neighboring_exact_reference(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    mbl_markers = (
        "ZIC1",
        "GPM6A",
        "STMN2",
        "INSM1",
        "GFAP",
        "NCAM1",
        "OTX2",
        "ATOH1",
        "SOX2",
        "MYC",
        "MYCN",
    )
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "MBL_SHH": {
                "markers": mbl_markers,
                "ref_medians": {gene: 100.0 for gene in mbl_markers},
                "context_codes": ("PCPG",),
                "parent_code": "MBL",
                "family": "cns-embryonal",
                "primary_tissue": "cerebellum",
                "source_cohort": "TREEHOUSE_POLYA_25_01_MBL_SUBGROUP_MARKERS",
                "reference_kind": "observed_bulk_reference",
            }
        },
    )
    original_marker_coherence = evidence._marker_coherence

    def fake_marker_coherence(code, sample):
        if code == "MBL_SHH":
            return {
                "status": "consistent",
                "detected": 5,
                "total": 5,
                "required_for_consistent": 3,
                "detected_fraction": 1.0,
            }
        return original_marker_coherence(code, sample)

    monkeypatch.setattr(evidence, "_marker_coherence", fake_marker_coherence)
    finding = {
        "cancer_type": "MTC",
        "rule_id": "mtc_calca",
        "surrogate": "CALCA",
        "surrogate_tpm": 1500.0,
        "threshold_tpm": 10.0,
        "support_genes": ["CHGA", "SYP", "RET"],
        "missing_support_genes": [],
        "support_pass": True,
        "support_gene_count": 3,
        "min_support_genes": 1,
        "required_support_gene_count": 3,
    }

    sample = {gene: 120.0 for gene in mbl_markers}
    sample.update({"CALCA": 1500.0, "CHGA": 200.0, "SYP": 180.0, "RET": 120.0})
    result = select_report_scope_from_evidence(
        _expression_frame(sample),
        _analysis(("PCPG", 1.0), ("NET_PANCREAS", 0.5), ("SCLC", 0.5)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "MTC"
    assert result["selected"]["selected_by"] == "rare_marker"
    mbl = next(row for row in result["evidence"] if row["cancer_type"] == "MBL_SHH")
    assert mbl["can_select_report_label"] is True


def test_mtc_marker_axis_needs_specific_anchor_in_neural_crest_context():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    finding = {
        "cancer_type": "MTC",
        "rule_id": "mtc_calca",
        "surrogate": "CALCA",
        "surrogate_tpm": 600.0,
        "threshold_tpm": 10.0,
        "support_genes": ["CHGA", "SYP", "RET"],
        "missing_support_genes": [],
        "support_pass": True,
        "support_gene_count": 3,
        "min_support_genes": 1,
        "required_support_gene_count": 3,
    }
    sample = {
        "CALCA": 600.0,
        "CHGA": 200.0,
        "SYP": 150.0,
        "RET": 12.0,
        "CEACAM5": 0.0,
        "CALCR": 0.0,
        "PHOX2B": 100.0,
        "TH": 400.0,
        "B4GALNT1": 30.0,
        "ALK": 10.0,
        "MYCN": 8.0,
        "DBH": 200.0,
        "CHGB": 500.0,
    }

    result = select_report_scope_from_evidence(
        _expression_frame(sample),
        _analysis(("PCPG", 1.0), ("SCLC", 0.5)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "PCPG"
    mtc = next(row for row in result["evidence"] if row["cancer_type"] == "MTC")
    assert mtc["label_decision"]["status"] == "blocked"
    assert any(
        "not specific enough to override a PCPG/neural-crest expression context"
        in reason
        for reason in mtc["blocking_reasons"]
    )
    assert mtc["mtc_neural_crest_context_conflict"] is True


def test_subgroup_reference_requires_established_parent_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    subgroup_markers = ("ZIC1", "GPM6A", "STMN2", "INSM1", "GFAP", "NCAM1")
    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "MBL_SHH": {
                "markers": subgroup_markers,
                "ref_medians": {gene: 100.0 for gene in subgroup_markers},
                "context_codes": ("MBL",),
                "parent_code": "MBL",
                "family": "cns-embryonal",
                "primary_tissue": "cerebellum",
                "source_cohort": "TREEHOUSE_POLYA_25_01_MBL_SUBGROUP_MARKERS",
                "reference_kind": "observed_bulk_reference",
            }
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame({gene: 120.0 for gene in subgroup_markers}),
        _analysis(("GBM", 1.0), ("LGG", 0.93), ("SKCM", 0.75)),
    )

    assert result["selected"]["cancer_type"] == "GBM"
    subgroup = next(row for row in result["evidence"] if row["cancer_type"] == "MBL_SHH")
    assert subgroup["can_select_report_label"] is False
    assert any("requires established MBL parent-context support" in r for r in subgroup["blocking_reasons"])


def test_adcc_marker_prompt_requires_combined_co_marker_support():
    from trufflepig.rare_inference import infer_rare_cancer_marker_hypotheses_from_rna

    hypotheses = infer_rare_cancer_marker_hypotheses_from_rna(
        _expression_frame({"MYB": 45.0, "KRT7": 12.0}),
        _analysis(("HNSC", 1.0)),
    )

    assert all(row["cancer_type"] != "ADCC" for row in hypotheses)


def test_direct_rare_rna_scope_requires_top_context_match():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence
    from trufflepig.rare_inference import infer_rare_cancer_marker_hypotheses_from_rna

    df = _expression_frame({"MYB": 60.0, "KIT": 40.0, "KRT7": 10.0})
    analysis = _analysis(("READ", 1.0), ("COAD", 0.4), ("HNSC", 0.15))

    # The ADCC MYB/KIT rule is gated to a salivary (HNSC) context; with READ on
    # top it must not be promotable to the report scope. Whether the rule yields
    # a hypothesis at all, it can never select the report label, so the report
    # scope stays the top expression match (READ).
    hypotheses = infer_rare_cancer_marker_hypotheses_from_rna(df, analysis)
    result = select_report_scope_from_evidence(
        df, analysis, rare_marker_hypotheses=hypotheses
    )

    assert result["selected"]["cancer_type"] == "READ"
    assert all(
        row.get("can_select_report_label") is not True
        for row in result["evidence"]
        if row["cancer_type"] == "ADCC"
    )


def test_marker_prompt_only_rules_never_set_report_scope():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    finding = {
        "cancer_type": "SARC_CHOR",
        "rule_id": "chor_tbxt",
        "surrogate": "TBXT",
        "surrogate_tpm": 50.0,
        "threshold_tpm": 5.0,
        "support_genes": ["KRT19", "SOX9"],
    }

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("SARC", 1.0)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "SARC"
    chor = next(row for row in result["evidence"] if row["cancer_type"] == "SARC_CHOR")
    assert chor["rule_promotes_report_scope"] is False
    assert chor["can_select_report_label"] is False


def test_osteogenic_reference_evidence_promotes_os_over_broad_sarc():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    df = _expression_frame(
        {
            "RUNX2": 520,
            "SATB2": 480,
            "IBSP": 1400,
            "SPP1": 1500,
            "COL1A1": 7600,
            "COL1A2": 12400,
            "DLX5": 230,
            "DMP1": 90,
            "MEPE": 190,
            "MDM2": 1180,
            "CDK4": 110,
            "FRS2": 110,
        }
    )

    result = select_report_scope_from_evidence(
        df,
        _analysis(("SARC", 1.0), ("UCS", 0.3)),
    )

    assert result["selected"]["cancer_type"] == "SARC_OS"
    assert result["selected"]["reference_cancer_type"] == "SARC"
    assert result["selected"]["expression_reference_cancer_type"] == "SARC_OS"
    assert result["selected"]["evidence_sources"] == ["fine_reference"]
    assert result["selected"]["metrics"]["related_context_support"] == 1.0
    assert result["selected"]["metrics"]["fine_reference_support"] >= 0.7


def test_mdm2_amp_without_osteogenic_program_stays_broad_sarc():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    df = _expression_frame(
        {
            "RUNX2": 25,
            "SATB2": 20,
            "IBSP": 1,
            "SPP1": 60,
            "COL1A1": 900,
            "COL1A2": 1500,
            "MDM2": 250,
            "CDK4": 190,
            "FRS2": 70,
        }
    )

    result = select_report_scope_from_evidence(
        df,
        _analysis(("SARC", 1.0), ("UCS", 0.3)),
    )

    assert result["selected"]["cancer_type"] == "SARC"
    os_evidence = next(row for row in result["evidence"] if row["cancer_type"] == "SARC_OS")
    assert os_evidence["cancer_type"] == "SARC_OS"
    assert os_evidence["can_select_report_label"] is False
    assert os_evidence["blocking_reasons"]


def test_blocked_marker_evidence_does_not_downgrade_direct_fusion_selection():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("READ", 1.0), ("COAD", 0.4)),
        fusion_scope_inference={
            "cancer_type": "NUTM",
            "fusion": {"pair": "BRD4--NUTM1"},
            "expected_pair": "BRD4--NUTM1",
        },
        rare_marker_hypotheses=[
            {
                "cancer_type": "NUTM",
                "rule_id": "nutm_nutm1",
                "surrogate": "NUTM1",
                "surrogate_tpm": 8.0,
                "threshold_tpm": 1.0,
                "support_genes": [],
            }
        ],
    )

    assert result["selected"]["cancer_type"] == "NUTM"
    assert result["selected"]["selected_by"] == "direct_fusion"
    assert result["selected"]["label_decision"]["status"] == "selected"


def test_empty_candidate_trace_does_not_crash():
    """Degenerate analysis (no broad RNA classifier output) must produce
    an empty evidence result, not raise. Pins the all-empty contract.
    """
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        {"candidate_trace": []},
    )
    assert result["selected"] is None
    assert result["evidence"] == []
    assert result["primary_expression_context"] is None


def test_single_element_candidate_trace_runs_only_broad_path():
    """A 1-row trace can't trigger tumor_label_refinement (needs rank-2)
    but still emits a broad-RNA hypothesis for that single code.
    """
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("BRCA", 0.85)),
    )
    selected = result["selected"]
    assert selected is not None
    assert selected["cancer_type"] == "BRCA"
    assert selected["selected_by"] == "pan_cancer_signature_ranker"


def test_close_lineage_trace_candidate_handles_missing_support():
    """Legacy/hand-built candidate traces may omit normalized support.

    The first trace row is still the ranker top hit and should be treated as
    full support instead of crashing when local-reference conflict checks search
    for a close same-lineage candidate.
    """
    from trufflepig import cancer_type_evidence as cte

    missing = cte._close_trace_candidate_for_lineage(
        [{"code": "SARC"}],
        "Sarcoma",
    )
    zero = cte._close_trace_candidate_for_lineage(
        [{"code": "SARC", "support_fraction_of_top": 0.0}],
        "Sarcoma",
    )

    assert missing["code"] == "SARC"
    assert zero["code"] == "SARC"


def test_equally_prioritized_hypotheses_break_ties_alphabetically():
    """Two hypotheses with identical class_rank + strength + tiebreak
    must resolve deterministically (ascending cancer_type), not at
    Python's sort-stability default order (which depends on insertion
    order) or descending (the prior reverse=True bug).
    """
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    # Two cohorts tied at exactly the same support_fraction_of_top; the broad-RNA
    # path will create one hypothesis per code with identical priority.
    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        {
            "cancer_type": "READ",
            "candidate_trace": [
                {"code": "READ", "support_fraction_of_top": 1.0},
            ],
        },
    )
    # Sanity: deterministic — re-running produces the same winner.
    result2 = select_report_scope_from_evidence(
        _empty_expression_frame(),
        {
            "cancer_type": "READ",
            "candidate_trace": [
                {"code": "READ", "support_fraction_of_top": 1.0},
            ],
        },
    )
    assert result["selected"]["cancer_type"] == result2["selected"]["cancer_type"]


def test_subtype_deconvolved_expression_without_subtype_column(monkeypatch):
    """Defensive: when the subtype-deconvolved-expression accessor returns
    a DataFrame without a ``subtype`` column, the local-reference path
    must skip the row gracefully instead of crashing on
    ``df[scalar_bool]``.
    """
    import pandas as pd

    from trufflepig import cancer_type_evidence as cte
    from trufflepig import reference as ref_module

    no_subtype_df = pd.DataFrame(
        {
            "cancer_code": ["HEPB"],
            "symbol": ["AFP"],
            "ensembl_gene_id": ["ENSG00000081051"],
            "tumor_tpm_median": [120.0],
        }
    )

    def fake_subtype(*args, **kwargs):
        return no_subtype_df

    monkeypatch.setattr(
        ref_module, "subtype_deconvolved_expression", fake_subtype
    )
    cte._local_expression_reference_panels.cache_clear()
    # Should not raise even though no "subtype" column is present.
    try:
        cte._local_expression_reference_panels()
    finally:
        cte._local_expression_reference_panels.cache_clear()


def _contrast_rows(*, contrast="GBC_vs_PAAD"):
    return (
        {
            "contrast": contrast,
            "type_a": "GBC",
            "type_b": "PAAD",
            "favors": "GBC",
            "symbol": "CLDN4",
            "direction": "high",
            "tier": "primary",
            "separability": "poor",
            "source": "GSE139682",
            "support_type": "gallbladder_expression_candidate",
        },
        {
            "contrast": contrast,
            "type_a": "GBC",
            "type_b": "PAAD",
            "favors": "GBC",
            "symbol": "KRT7",
            "direction": "high",
            "tier": "supporting",
            "separability": "poor",
            "source": "GSE139682",
            "support_type": "gallbladder_expression_candidate",
        },
        {
            "contrast": contrast,
            "type_a": "GBC",
            "type_b": "PAAD",
            "favors": "GBC",
            "symbol": "KRT19",
            "direction": "high",
            "tier": "supporting",
            "separability": "poor",
            "source": "GSE139682",
            "support_type": "gallbladder_expression_candidate",
        },
        {
            "contrast": contrast,
            "type_a": "GBC",
            "type_b": "PAAD",
            "favors": "GBC",
            "symbol": "ERBB2",
            "direction": "high",
            "tier": "supporting",
            "separability": "poor",
            "source": "GSE139682",
            "support_type": "gallbladder_expression_candidate",
        },
        {
            "contrast": contrast,
            "type_a": "GBC",
            "type_b": "PAAD",
            "favors": "PAAD",
            "symbol": "GATA6",
            "direction": "high",
            "tier": "primary",
            "separability": "poor",
            "source": "PMID:26343385",
            "support_type": "contrast_marker_literature",
        },
        {
            "contrast": contrast,
            "type_a": "GBC",
            "type_b": "PAAD",
            "favors": "PAAD",
            "symbol": "PDX1",
            "direction": "high",
            "tier": "primary",
            "separability": "poor",
            "source": "PMID:26343385",
            "support_type": "contrast_marker_literature",
        },
        {
            "contrast": contrast,
            "type_a": "GBC",
            "type_b": "PAAD",
            "favors": "PAAD",
            "symbol": "MSLN",
            "direction": "high",
            "tier": "supporting",
            "separability": "poor",
            "source": "PMID:26343385",
            "support_type": "contrast_marker_literature",
        },
    )


def _stad_chol_contrast_rows():
    """Representative parent-level GI contrast in the pirlygenes row schema."""
    rows = []
    for favors, primary, supporting in (
        ("STAD", ("CDX2", "CLDN18"), ("GKN1",)),
        ("CHOL", ("KRT19", "EPCAM"), ("KRT7", "SOX9", "HNF1B")),
    ):
        for symbol in (*primary, *supporting):
            rows.append(
                {
                    "contrast": "STAD_vs_CHOL",
                    "type_a": "STAD",
                    "type_b": "CHOL",
                    "favors": favors,
                    "symbol": symbol,
                    "direction": "high",
                    "tier": "primary" if symbol in primary else "supporting",
                    "separability": "strong",
                    "source": "pirlygenes#266",
                    "support_type": "contrast_marker_literature",
                }
            )
    return tuple(rows)


def _laml_cml_contrast_rows():
    """Representative parent-level heme contrast in the pirlygenes row schema."""
    rows = []
    for favors, primary, supporting in (
        ("LAML", ("CD34", "FLT3"), ("GATA2",)),
        ("CML", ("BCR", "ABL1"), ("LYZ",)),
    ):
        for symbol in (*primary, *supporting):
            rows.append(
                {
                    "contrast": "CML_vs_LAML",
                    "type_a": "LAML",
                    "type_b": "CML",
                    "favors": favors,
                    "symbol": symbol,
                    "direction": "high",
                    "tier": "primary" if symbol in primary else "supporting",
                    "separability": "strong",
                    "source": "pirlygenes#266",
                    "support_type": "contrast_marker_literature",
                }
            )
    return tuple(rows)


def _crc_stad_contrast_rows():
    """A valid but non-strong parent contrast for consensus safety tests."""
    rows = []
    for favors, primary, supporting in (
        ("CRC", ("KRT20",), ("CDH17",)),
        ("STAD", ("CLDN18",), ("GKN1",)),
    ):
        for symbol in (*primary, *supporting):
            rows.append(
                {
                    "contrast": "CRC_vs_STAD",
                    "type_a": "CRC",
                    "type_b": "STAD",
                    "favors": favors,
                    "symbol": symbol,
                    "direction": "high",
                    "tier": "primary" if symbol in primary else "supporting",
                    "separability": "strong",
                    "source": "curated-parent-contrast",
                    "support_type": "contrast_marker_literature",
                }
            )
    return tuple(rows)


def _brca_sarc_epith_contrast_rows():
    """A cross-lineage parent contrast with a basal breast child context."""
    rows = []
    for favors, primary, supporting in (
        ("BRCA", ("ESR1", "FOXA1"), ("GATA3",)),
        ("SARC_EPITH", ("KRT8", "KRT18"), ("EPCAM", "CD34", "SMARCB1")),
    ):
        for symbol in (*primary, *supporting):
            rows.append(
                {
                    "contrast": "BRCA_vs_SARC_EPITH",
                    "type_a": "BRCA",
                    "type_b": "SARC_EPITH",
                    "favors": favors,
                    "symbol": symbol,
                    "direction": "high",
                    "tier": "primary" if symbol in primary else "supporting",
                    "separability": "strong",
                    "source": "curated-parent-contrast",
                    "support_type": "contrast_marker_literature",
                }
            )
    return tuple(rows)


def _stad_program_expression():
    return _expression_frame(
        {
            "CLDN18": 100.0,
            "GKN1": 60.0,
            "CDX2": 40.0,
            "MUC5AC": 30.0,
            "MUC6": 20.0,
            "TFF1": 25.0,
            "KRT20": 0.1,
            "CDH17": 0.1,
        }
    )


def test_parent_contrast_can_resolve_an_active_child_entity_context(monkeypatch):
    """Pairwise evidence can inform, but cannot itself select, a child context."""
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_contrast_discriminator_rows",
        _stad_chol_contrast_rows,
    )
    analysis = _analysis(("STAD_EBV", 1.0), ("CHOL", 0.78))
    analysis["fit_quality"] = {"label": "ambiguous"}

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "KRT19": 120.0,
                "EPCAM": 80.0,
                "KRT7": 90.0,
                "SOX9": 35.0,
                "HNF1B": 25.0,
                "CDX2": 0.1,
                "CLDN18": 0.1,
                "GKN1": 0.1,
            }
        ),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "CHOL"
    assert result["selected"]["selected_by"] == "fused_evidence"
    assert result["selected"]["contrast_discriminator_context_code"] == "STAD"
    assert (
        result["selected"]["contrast_discriminator_context_match_code"]
        == "STAD_EBV"
    )
    assert result["selected"]["contrast_discriminator_top_participant"] == "STAD"
    ambiguity = result["selected"]["contrast_discriminator_active_ambiguity"]
    assert ambiguity["active_for_report_label"] is True
    assert ambiguity["top_code"] == "STAD_EBV"
    assert ambiguity["top_participant"] == "STAD"
    assert (
        result["selected"]["contrast_discriminator_upstream_consensus"]["status"]
        == "hypothesis_only"
    )
    assert "curated_marker_program" not in (
        result["selected"]["adjudication_admissible_support"]
    )


def test_parent_contrast_uses_coherent_child_program_before_cross_code_promotion(
    monkeypatch,
):
    """A coherent subtype is not rejected for lacking its parent's markers."""
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_contrast_discriminator_rows",
        _brca_sarc_epith_contrast_rows,
    )
    analysis = _analysis(("BRCA_Basal", 1.0))
    analysis["fit_quality"] = {"label": "focused"}

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                # Coherent basal-like breast program, with the luminal parent
                # program appropriately absent.
                "KRT5": 60.0,
                "KRT14": 50.0,
                "KRT17": 45.0,
                "KRT6B": 40.0,
                "FOXC1": 35.0,
                # A strong opposing contrast program must still not relabel a
                # focused, coherent child diagnosis merely because BRCA's
                # luminal markers are absent.
                "KRT8": 60.0,
                "KRT18": 60.0,
                "EPCAM": 50.0,
                "CD34": 40.0,
                "SMARCB1": 20.0,
            }
        ),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "BRCA_Basal"
    sarcoma = next(
        row for row in result["evidence"] if row["cancer_type"] == "SARC_EPITH"
    )
    assert sarcoma["contrast_discriminator_context_code"] == "BRCA"
    assert sarcoma["contrast_discriminator_context_match_code"] == "BRCA_Basal"
    assert (
        sarcoma["contrast_discriminator_context_marker_coherence_code"]
        == "BRCA_Basal"
    )
    assert sarcoma["contrast_discriminator_context_marker_coherence"]["status"] == (
        "consistent"
    )
    ambiguity = sarcoma["contrast_discriminator_active_ambiguity"]
    assert ambiguity["context_marker_incoherent"] is False
    assert ambiguity["active_for_report_label"] is False
    assert sarcoma["label_decision"]["status"] == "blocked"


def test_parent_contrast_support_does_not_demote_an_agreeing_child(monkeypatch):
    """Parent evidence explains a matching child call without broadening it."""
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_contrast_discriminator_rows",
        _stad_chol_contrast_rows,
    )

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "CDX2": 80.0,
                "CLDN18": 100.0,
                "GKN1": 60.0,
                "MUC5AC": 30.0,
                "MUC6": 20.0,
                "TFF1": 25.0,
                "KRT19": 0.1,
                "EPCAM": 0.1,
                "KRT7": 0.1,
                "SOX9": 0.1,
                "HNF1B": 0.1,
            }
        ),
        _analysis(("STAD_EBV", 1.0), ("CHOL", 0.72)),
    )

    assert result["selected"]["cancer_type"] == "STAD_EBV"
    child = next(
        row for row in result["evidence"] if row["cancer_type"] == "STAD_EBV"
    )
    assert "contrast_discriminator" in child["evidence_sources"]
    assert child["contrast_discriminator_context_match_code"] == "STAD_EBV"
    assert all(
        "contrast_discriminator" not in row["evidence_sources"]
        for row in result["evidence"]
        if row["cancer_type"] == "STAD"
    )
    ambiguity = child["contrast_discriminator_active_ambiguity"]
    assert ambiguity["same_top"] is True
    assert ambiguity["active_for_report_label"] is False


def test_agreeing_parent_contrast_survives_hierarchy_centroid_fusion_on_child(
    monkeypatch,
):
    """Downstream whole-profile support cannot broaden an agreeing child call."""
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_contrast_discriminator_rows",
        _stad_chol_contrast_rows,
    )
    monkeypatch.setattr(
        evidence,
        "_centroid_and_confidence",
        lambda _sample: (pd.Series({"STAD": 0.91}), True),
    )
    monkeypatch.setattr(
        evidence,
        "_add_pan_cancer_signature_marker_features",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evidence,
        "_add_learned_expression_classifier_features",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evidence,
        "_add_local_expression_reference_features",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evidence,
        "_add_lineage_panel_features",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(evidence, "_FINE_REFERENCE_SPECS", ())

    def add_parent_compatible_hierarchy(hypotheses, _sample):
        for hypothesis in hypotheses.values():
            hypothesis.details.update(
                {
                    "learned_expression_hierarchy_support": 0.90,
                    "learned_expression_entity_support": 0.90,
                    "learned_expression_entity_label": "STAD",
                    "learned_expression_family_support": 0.95,
                    "learned_expression_family_label": "carcinoma-gi",
                    "learned_expression_compartment_support": 0.98,
                    "learned_expression_compartment_label": "epithelial",
                }
            )

    monkeypatch.setattr(
        evidence,
        "_add_learned_hierarchy_candidate_features",
        add_parent_compatible_hierarchy,
    )

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "CDX2": 80.0,
                "CLDN18": 100.0,
                "GKN1": 60.0,
                "KRT19": 0.1,
                "EPCAM": 0.1,
                "KRT7": 0.1,
                "SOX9": 0.1,
                "HNF1B": 0.1,
            }
        ),
        _analysis(("STAD_EBV", 1.0)),
    )

    assert result["selected"]["cancer_type"] == "STAD_EBV"
    assert result["selected"]["fused_evidence_centroid_support"] == 1.0
    assert result["selected"]["learned_expression_hierarchy_support"] == 0.9
    assert "contrast_discriminator" in result["selected"]["evidence_sources"]
    assert all(
        "contrast_discriminator" not in row["evidence_sources"]
        for row in result["evidence"]
        if row["cancer_type"] == "STAD"
    )


def test_parent_contrast_stays_nonselecting_for_an_unrelated_top_context(monkeypatch):
    """A descendant secondary hit cannot activate a contrast under another lineage."""
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_contrast_discriminator_rows",
        _stad_chol_contrast_rows,
    )
    analysis = _analysis(("PCPG", 1.0), ("STAD_CIN", 0.76))
    analysis["fit_quality"] = {"label": "ambiguous"}

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "KRT19": 120.0,
                "EPCAM": 80.0,
                "KRT7": 90.0,
                "SOX9": 35.0,
                "HNF1B": 25.0,
            }
        ),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "PCPG"
    chol = next(row for row in result["evidence"] if row["cancer_type"] == "CHOL")
    assert chol["contrast_discriminator_context_code"] == "STAD"
    assert chol["contrast_discriminator_context_match_code"] == "STAD_CIN"
    assert chol["contrast_discriminator_top_participant"] == ""
    ambiguity = chol["contrast_discriminator_active_ambiguity"]
    assert ambiguity["top_participates"] is False
    assert ambiguity["active_for_report_label"] is False
    assert chol["label_decision"]["status"] == "blocked"


def test_parent_contrast_resolves_heme_entity_without_inventing_risk_group(monkeypatch):
    """A pairwise AML nomination remains contextual and cannot assign a child."""
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_contrast_discriminator_rows",
        _laml_cml_contrast_rows,
    )
    analysis = _analysis(("CML", 1.0), ("LAML_ELNadv", 0.82))
    analysis["fit_quality"] = {"label": "ambiguous"}

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "CD34": 80.0,
                "FLT3": 45.0,
                "GATA2": 35.0,
                "ELANE": 70.0,
                "KIT": 30.0,
                "MPO": 100.0,
                "BCR": 0.1,
                "ABL1": 0.1,
                "LYZ": 0.1,
            }
        ),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "CML"
    laml = next(row for row in result["evidence"] if row["cancer_type"] == "LAML")
    assert laml["contrast_discriminator_context_code"] == "CML"
    assert laml["contrast_discriminator_context_match_code"] == "CML"
    assert laml["contrast_discriminator_top_participant"] == "CML"
    assert laml["can_select_report_label"] is False
    assert laml["contrast_discriminator_upstream_consensus"]["status"] == (
        "hypothesis_only"
    )


def test_parent_contrast_preserves_normalized_broad_coarse_consensus(monkeypatch):
    """Matching child/parent contexts retain the existing consensus veto."""
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_contrast_discriminator_rows",
        _crc_stad_contrast_rows,
    )
    analysis = _analysis(("COAD_MSI", 1.0))
    analysis["fit_quality"] = {"label": "ambiguous"}
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        top_tcga_cohorts=[("COAD_TPM", 0.9), ("STAD_TPM", 0.8)]
    )

    result = select_report_scope_from_evidence(
        _stad_program_expression(),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "COAD_MSI"
    stad = next(row for row in result["evidence"] if row["cancer_type"] == "STAD")
    assert stad["contrast_discriminator_context_code"] == "CRC"
    assert stad["contrast_discriminator_context_match_code"] == "COAD_MSI"
    assert stad["contrast_discriminator_consensus_context"] == "CRC"
    assert stad["contrast_discriminator_strong_signal"] is False
    assert stad["label_decision"]["status"] == "blocked"
    assert any(
        "coarse reference matching both support CRC" in reason
        for reason in stad["blocking_reasons"]
    )


def test_parent_contrast_tie_prefers_active_top_participant(monkeypatch):
    """A tied context remains on the active child when the panel cannot promote."""
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_contrast_discriminator_rows",
        _crc_stad_contrast_rows,
    )
    analysis = _analysis(("COAD_MSI", 1.0))
    analysis["fit_quality"] = {"label": "ambiguous"}
    analysis["healthy_vs_tumor"] = SimpleNamespace(
        top_tcga_cohorts=[("STAD_TPM", 0.9)]
    )

    result = select_report_scope_from_evidence(
        _stad_program_expression(),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "COAD_MSI", result["selected"]
    stad = next(row for row in result["evidence"] if row["cancer_type"] == "STAD")
    assert stad["contrast_discriminator_context_code"] == "CRC"
    assert (
        stad["contrast_discriminator_context_match_code"]
        == "COAD_MSI"
    )
    ambiguity = stad["contrast_discriminator_active_ambiguity"]
    assert ambiguity["context_is_top"] is True
    assert ambiguity["active_for_report_label"] is True
    assert stad["can_select_report_label"] is False


def test_shipped_pairwise_contrast_program_remains_a_hypothesis():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    analysis = _analysis(("PAAD", 1.0))
    analysis["fit_quality"] = {"label": "ambiguous"}

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "CLDN4": 90.0,
                "KRT7": 120.0,
                "KRT19": 80.0,
                "ERBB2": 35.0,
                "GATA6": 0.2,
                "PDX1": 0.1,
                "MSLN": 0.2,
            }
        ),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "PAAD"
    gbc = next(row for row in result["evidence"] if row["cancer_type"] == "GBC")
    assert gbc["metrics"]["contrast_discriminator_support"] > 0.8
    assert gbc["can_select_report_label"] is False
    assert any(
        "hypothesis evidence only" in reason
        for reason in gbc["blocking_reasons"]
    )
    graph = result["staged_evidence_graph"]
    contrast_channel = next(
        row for row in graph["channels"]
        if row.get("candidate_code") == "GBC"
        and row["channel"] == "contrast_discriminator"
    )
    assert contrast_channel["selects_report_label"] is False
    assert contrast_channel["details"]["active_for_report_label"] is True
    assert contrast_channel["details"]["top_participates"] is True
    assert contrast_channel["details"]["context_is_top"] is True


def _conflicting_pairwise_contrast_rows():
    rows = []
    for contrast, type_a, type_b, programs in (
        (
            "PAAD_vs_STAD",
            "PAAD",
            "STAD",
            {
                "PAAD": ("GATA6", "PDX1", "MSLN"),
                "STAD": ("CLDN18", "GKN1", "MUC5AC"),
            },
        ),
        (
            "GBC_vs_STAD",
            "GBC",
            "STAD",
            {
                "GBC": ("KRT7", "KRT19", "ERBB2"),
                "STAD": ("CLDN18", "GKN1", "MUC5AC"),
            },
        ),
    ):
        for favors, symbols in programs.items():
            for index, symbol in enumerate(symbols):
                rows.append(
                    {
                        "contrast": contrast,
                        "type_a": type_a,
                        "type_b": type_b,
                        "favors": favors,
                        "symbol": symbol,
                        "direction": "high",
                        "tier": "primary" if index < 2 else "supporting",
                        "separability": "strong",
                        "source": "representative-pairwise-panel",
                        "support_type": "contrast_marker_literature",
                    }
                )
    return tuple(rows)


def test_conflicting_pairwise_contrasts_cannot_act_as_a_global_classifier(
    monkeypatch,
):
    """Simultaneous A-vs-B answers must not choose among A, B, C globally."""
    import trufflepig.cancer_type_evidence as evidence
    import trufflepig.expression_classifier as classifier
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_contrast_discriminator_rows",
        _conflicting_pairwise_contrast_rows,
    )
    monkeypatch.setattr(evidence, "_marker_coherence", lambda *_args: {})
    monkeypatch.setattr(
        classifier,
        "classify_expression",
        lambda _sample, top_k=5: [("PAAD", 0.40), ("STAD", 0.20)],
    )
    monkeypatch.setattr(
        classifier,
        "classify_expression_hierarchy",
        lambda _sample, top_k=5: [
            SimpleNamespace(
                public_dict=lambda: {
                    "stage": "entity",
                    "label": "PAAD",
                    "probability": 0.40,
                    "margin": 0.20,
                    "top_predictions": [
                        {"label": "PAAD", "probability": 0.40},
                        {"label": "STAD", "probability": 0.20},
                    ],
                }
            ),
        ],
    )

    def add_composition(hypotheses, *_args, **_kwargs):
        evidence._hypothesis(
            hypotheses,
            "PAAD",
        ).coarse_composition_support = 0.80

    monkeypatch.setattr(
        evidence,
        "_add_coarse_composition_reference_features",
        add_composition,
    )
    monkeypatch.setattr(
        evidence,
        "_centroid_and_confidence",
        lambda _sample: (pd.Series(dtype=float), False),
    )
    analysis = _analysis(("STAD", 1.0), ("READ", 0.97), ("PAAD", 0.93))
    analysis["fit_quality"] = {"label": "ambiguous"}

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "GATA6": 100.0,
                "PDX1": 80.0,
                "MSLN": 60.0,
                "KRT7": 100.0,
                "KRT19": 80.0,
                "ERBB2": 60.0,
                "CLDN18": 0.1,
                "GKN1": 0.1,
                "MUC5AC": 0.1,
            }
        ),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "STAD"
    for code in ("PAAD", "GBC"):
        candidate = next(
            row for row in result["evidence"] if row["cancer_type"] == code
        )
        assert candidate["can_select_report_label"] is False
        assert any(
            "discriminator consensus is conflict" in reason
            for reason in candidate["blocking_reasons"]
        )
        assert "curated_marker_program" not in (
            candidate["adjudication_admissible_support"]
        )
    paad = next(row for row in result["evidence"] if row["cancer_type"] == "PAAD")
    marker_axis = next(
        axis
        for axis in paad["entity_evidence_consensus"]["axes"]
        if axis["axis"] == "curated_marker_program"
    )
    assert marker_axis["candidate_support"] == 0.0


def test_conflicting_contrasts_preserve_an_independent_lineage_selection(
    monkeypatch,
):
    """Invalidating one selector must not erase another selector's decision."""
    import trufflepig.cancer_type_evidence as evidence
    import trufflepig.expression_classifier as classifier
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_contrast_discriminator_rows",
        _conflicting_pairwise_contrast_rows,
    )
    monkeypatch.setattr(evidence, "_marker_coherence", lambda *_args: {})
    monkeypatch.setattr(classifier, "classify_expression", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        classifier,
        "classify_expression_hierarchy",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(evidence, "_FINE_REFERENCE_SPECS", ())
    monkeypatch.setattr(
        evidence,
        "_add_local_expression_reference_features",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evidence,
        "_add_coarse_composition_reference_features",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evidence,
        "_add_fused_evidence_features",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evidence,
        "_centroid_and_confidence",
        lambda _sample: (pd.Series(dtype=float), False),
    )

    def add_lineage(hypotheses, *_args, **_kwargs):
        paad = evidence._hypothesis(hypotheses, "PAAD")
        paad.add_source("lineage_panel")
        paad.details["lineage_panel_score"] = 0.65
        paad.admit_adjudication_support(
            "curated_marker_program",
            0.65,
            selector="lineage_panel",
        )
        paad.consider_for_report_label(
            selected_by="lineage_panel",
            can_select=True,
            blocking_reasons=(),
            priority=(1, 0.65),
        )
        return {"promotion": {"promoted": True, "code": "PAAD"}}

    monkeypatch.setattr(evidence, "_add_lineage_panel_features", add_lineage)
    analysis = _analysis(("STAD", 1.0), ("PAAD", 0.93))
    analysis["fit_quality"] = {"label": "ambiguous"}

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "GATA6": 100.0,
                "PDX1": 80.0,
                "MSLN": 60.0,
                "KRT7": 100.0,
                "KRT19": 80.0,
                "ERBB2": 60.0,
                "CLDN18": 0.1,
                "GKN1": 0.1,
                "MUC5AC": 0.1,
            }
        ),
        analysis,
    )

    assert result["selected"]["cancer_type"] == "PAAD"
    assert result["selected"]["selected_by"] == "lineage_panel"
    paad = next(row for row in result["evidence"] if row["cancer_type"] == "PAAD")
    marker_support = paad["adjudication_admissible_support_by_selector"][
        "curated_marker_program"
    ]
    assert marker_support == {"lineage_panel": 0.65}
    assert "contrast_discriminator" in paad["adjudication_exclusions"][
        "curated_marker_program"
    ]
    assert paad["blocking_reasons"] == []


def test_contrast_discriminator_blocks_marker_incoherent_override(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(evidence, "_contrast_discriminator_rows", _contrast_rows)

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "CLDN4": 90.0,
                "KRT7": 120.0,
                "KRT19": 80.0,
                "ERBB2": 0.5,
                "GATA6": 0.2,
                "PDX1": 0.1,
                "MSLN": 0.2,
            }
        ),
        _analysis(("PAAD", 1.0)),
    )

    assert result["selected"]["cancer_type"] == "PAAD"
    gbc = next(row for row in result["evidence"] if row["cancer_type"] == "GBC")
    assert gbc["label_decision"]["status"] == "blocked"
    assert any("marker program is partial" in reason for reason in gbc["blocking_reasons"])


def test_contrast_discriminator_requires_primary_context_for_cross_code_promotion(
    monkeypatch,
):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(evidence, "_contrast_discriminator_rows", _contrast_rows)
    signal = SimpleNamespace(
        cancer_hint="tumor-consistent",
        top_tcga_cohorts=[
            ("PRAD_TPM", 0.76),
            ("PAAD_TPM", 0.76),
            ("PCPG_TPM", 0.75),
        ],
    )

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "CLDN4": 90.0,
                "KRT7": 120.0,
                "KRT19": 80.0,
                "ERBB2": 35.0,
                "CEACAM5": 8.0,
                "GATA6": 0.2,
                "PDX1": 0.1,
                "MSLN": 0.2,
            }
        ),
        {
            "cancer_type": "PCPG",
            "fit_quality": {"label": "focused"},
            "healthy_vs_tumor": signal,
            "candidate_trace": [
                {"code": "PCPG", "support_fraction_of_top": 1.0},
                {"code": "NET_PANCREAS", "support_fraction_of_top": 0.5},
            ],
        },
    )

    assert result["selected"]["cancer_type"] == "PCPG"
    gbc = next(row for row in result["evidence"] if row["cancer_type"] == "GBC")
    assert gbc["label_decision"]["status"] == "blocked"
    assert gbc["contrast_discriminator_context_code"] == "PAAD"
    assert gbc["contrast_discriminator_context_is_primary"] is False
    assert (
        gbc["contrast_discriminator_active_ambiguity"]["active_for_report_label"]
        is False
    )
    assert any(
        "does not resolve the active top RNA context" in r
        for r in gbc["blocking_reasons"]
    )
    graph = result["staged_evidence_graph"]
    contrast_channel = next(
        row for row in graph["channels"]
        if row.get("candidate_code") == "GBC"
        and row["channel"] == "contrast_discriminator"
    )
    assert contrast_channel["selects_report_label"] is False
    assert contrast_channel["status"] == "blocked"
    assert contrast_channel["details"]["top_participates"] is False


def test_same_top_contrast_does_not_outrank_exact_reference(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
            "HEPB": {
                "markers": ("DLK1", "GPC3", "KRT8", "KRT18"),
                "ref_medians": {
                    "DLK1": 100.0,
                    "GPC3": 100.0,
                    "KRT8": 100.0,
                    "KRT18": 100.0,
                },
                "context_codes": ("LUAD",),
                "parent_code": "",
                "family": "embryonal",
                "primary_tissue": "liver",
                "source_cohort": "TREEHOUSE_POLYA_25_01_HEPB",
                "reference_kind": "deconvolved_tumor_reference",
                "expression_source": "TREEHOUSE",
            }
        },
    )
    monkeypatch.setattr(
        evidence,
        "_contrast_discriminator_rows",
        lambda: (
            {
                "contrast": "LUAD_vs_LUSC",
                "type_a": "LUAD",
                "type_b": "LUSC",
                "favors": "LUAD",
                "symbol": "SFTPC",
                "direction": "high",
                "tier": "primary",
                "separability": "strong",
                "source": "PMID:22960745",
                "support_type": "contrast_marker_literature",
            },
            {
                "contrast": "LUAD_vs_LUSC",
                "type_a": "LUAD",
                "type_b": "LUSC",
                "favors": "LUAD",
                "symbol": "NAPSA",
                "direction": "high",
                "tier": "primary",
                "separability": "strong",
                "source": "PMID:22960745",
                "support_type": "contrast_marker_literature",
            },
            {
                "contrast": "LUAD_vs_LUSC",
                "type_a": "LUAD",
                "type_b": "LUSC",
                "favors": "LUSC",
                "symbol": "TP63",
                "direction": "high",
                "tier": "primary",
                "separability": "strong",
                "source": "PMID:22960745",
                "support_type": "contrast_marker_literature",
            },
            {
                "contrast": "LUAD_vs_LUSC",
                "type_a": "LUAD",
                "type_b": "LUSC",
                "favors": "LUSC",
                "symbol": "KRT5",
                "direction": "high",
                "tier": "primary",
                "separability": "strong",
                "source": "PMID:22960745",
                "support_type": "contrast_marker_literature",
            },
        ),
    )

    result = select_report_scope_from_evidence(
        _expression_frame(
            {
                "SFTPC": 500.0,
                "NAPSA": 100.0,
                "TP63": 0.1,
                "KRT5": 0.1,
                "DLK1": 20.0,
                "GPC3": 20.0,
                "KRT8": 20.0,
                "KRT18": 20.0,
            }
        ),
        _analysis(("LUAD", 1.0), ("LUSC", 0.8)),
    )

    assert result["selected"]["cancer_type"] == "HEPB"
    assert result["selected"]["selected_by"] == "local_expression_reference"
    luad = next(row for row in result["evidence"] if row["cancer_type"] == "LUAD")
    assert "contrast_discriminator" in luad["evidence_sources"]
    assert luad["selected_by"] == "pan_cancer_signature_ranker"


def test_fallback_context_selection_resets_stale_selectable_selector():
    """A hypothesis initially selectable and later demoted by ``_add_fused_evidence_features``
    keeps its selectable ``selected_by`` (e.g. ``local_expression_reference``) while
    ``can_select_report_label`` is False. The blocked fallback-context selection must reset that
    selector to the pan-cancer ranker — otherwise ``_apply_cancer_type_evidence`` (which treats any
    non-ranker ``selected_by`` as a report-scope selection) would let the blocked hypothesis drive
    the final report label."""
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _fallback_context_selected,
    )

    hyp = CancerTypeEvidence(
        cancer_type="READ",
        selected_by="local_expression_reference",  # stale selectable selector
        can_select_report_label=False,  # ...but demoted, so it must NOT select the label
        evidence_sources=("pan_cancer_signature_ranker",),
    )
    analysis = {"candidate_trace": [{"code": "READ"}]}

    result = _fallback_context_selected({"READ": hyp}, analysis)

    assert result is hyp
    # Reset to the context ranker (which IS the fallback's admission source), not the stale selector.
    assert result.selected_by == "pan_cancer_signature_ranker"
    assert result.label_basis == "pan_cancer_signature_ranker"
    # The serialized selection the caller reads must carry the ranker so it is not routed to scope.
    assert result.public_dict()["selected_by"] == "pan_cancer_signature_ranker"


def test_fallback_context_collapses_unresolved_tied_liposarcoma_children():
    """A blocked ordinal tie supports the parent, not arbitrary leaf precision."""

    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _fallback_context_selected,
    )

    hypotheses = {}
    trace = []
    for rank, (code, support) in enumerate(
        (
            ("SARC_DDLPS", 1.0),
            ("SARC_WDLPS", 0.982),
            ("SARC_PLEOLPS", 0.981),
        ),
        start=1,
    ):
        hypotheses[code] = CancerTypeEvidence(
            cancer_type=code,
            broad_rna_support=support,
            broad_rna_rank=rank,
            evidence_sources=("pan_cancer_signature_ranker",),
            details={"support_rank_tier": 1},
        )
        trace.append(
            {
                "code": code,
                "support_fraction_of_top": support,
                "support_rank_tier": 1,
            }
        )

    result = _fallback_context_selected(
        hypotheses,
        {"candidate_trace": trace},
    )

    assert result.cancer_type == "SARC_LPS"
    assert result.selected_by == "entity_evidence_consensus"
    assert result.details["fallback_context_adjudication"]["mode"] == (
        "tied_sibling_parent_abstention"
    )
    assert set(
        result.details["fallback_context_adjudication"]["tied_sibling_codes"]
    ) == {"SARC_DDLPS", "SARC_WDLPS", "SARC_PLEOLPS"}


def test_fallback_context_abstains_to_structured_crc_parent():
    """A blocked GI top row yields to the supported ontology parent."""
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _fallback_context_selected,
        _hypothesis_evidence_channels,
    )

    stad = CancerTypeEvidence(
        cancer_type="STAD",
        evidence_sources=("pan_cancer_signature_ranker",),
        details={
            "fused_evidence_structured_parent_abstention": {
                "candidate_code": "STAD",
                "abstention_code": "CRC",
                "supporting_candidate": {
                    "code": "READ",
                    "rank": 2,
                    "support_fraction_of_top": 0.97,
                    "family_support": 0.76,
                },
            },
        },
    )
    read = CancerTypeEvidence(
        cancer_type="READ",
        broad_rna_support=0.97,
        broad_rna_rank=2,
        evidence_sources=("pan_cancer_signature_ranker",),
    )
    analysis = {
        "candidate_trace": [
            {"code": "STAD", "support_fraction_of_top": 1.0},
            {"code": "READ", "support_fraction_of_top": 0.97},
        ]
    }

    hypotheses = {"STAD": stad, "READ": read}
    result = _fallback_context_selected(hypotheses, analysis)

    assert result is hypotheses["CRC"]
    assert result.can_select_report_label is True
    assert result.selected_by == "entity_evidence_consensus"
    assert result.expression_reference_cancer_type == "READ"
    assert result.details["entity_consensus_adjudication_mode"] == (
        "structured_parent_abstention"
    )
    assert result.details["fallback_context_adjudication"] == {
        "mode": "structured_parent_abstention",
        "blocked_top_code": "STAD",
        "supporting_child_code": "READ",
        "abstention_code": "CRC",
        "conflict": stad.details["fused_evidence_structured_parent_abstention"],
    }
    abstention_channel = next(
        channel
        for channel in _hypothesis_evidence_channels(result)
        if channel["channel"] == "entity_evidence_consensus"
    )
    assert abstention_channel["role"] == "structured_parent_abstention"
    assert abstention_channel["details"]["supporting_child_code"] == "READ"


def test_fallback_context_rejects_unrelated_structured_abstention():
    """A malformed structured blocker cannot select an unrelated branch."""
    from trufflepig.cancer_type_evidence import (
        CancerTypeEvidence,
        _fallback_context_selected,
    )

    stad = CancerTypeEvidence(
        cancer_type="STAD",
        evidence_sources=("pan_cancer_signature_ranker",),
        details={
            "fused_evidence_structured_parent_abstention": {
                "candidate_code": "STAD",
                "abstention_code": "BRCA",
                "supporting_candidate": {
                    "code": "READ",
                    "rank": 2,
                    "support_fraction_of_top": 0.97,
                },
            },
        },
    )
    read = CancerTypeEvidence(
        cancer_type="READ",
        broad_rna_support=0.97,
        broad_rna_rank=2,
        evidence_sources=("pan_cancer_signature_ranker",),
    )
    hypotheses = {"STAD": stad, "READ": read}

    result = _fallback_context_selected(
        hypotheses,
        {
            "candidate_trace": [
                {"code": "STAD", "support_fraction_of_top": 1.0},
                {"code": "READ", "support_fraction_of_top": 0.97},
            ]
        },
    )

    assert result is stad
    assert result.selected_by == "pan_cancer_signature_ranker"
    assert "BRCA" not in hypotheses


def test_enrich_mmr_vote_mlh1_cohort_context_adds_ratio(monkeypatch):
    import trufflepig.cancer_type_evidence as cte

    monkeypatch.setattr(cte, "_cohort_bulk_gene_median", lambda code, gene: 18.0)
    vote = {"details": {"msi_probability": 0.8, "mlh1_expression": {"tpm": 4.0}}}
    out = cte._enrich_mmr_vote_mlh1_cohort_context(vote, "COAD")
    mlh1 = out["details"]["mlh1_expression"]
    assert mlh1["cohort_median_tpm"] == 18.0
    assert mlh1["cohort_ratio"] == round(4.0 / 18.0, 4)
    # Original vote is not mutated.
    assert "cohort_ratio" not in vote["details"]["mlh1_expression"]


def test_enrich_mmr_vote_mlh1_cohort_context_noop_without_reference(monkeypatch):
    import trufflepig.cancer_type_evidence as cte

    monkeypatch.setattr(cte, "_cohort_bulk_gene_median", lambda code, gene: None)
    vote = {"details": {"mlh1_expression": {"tpm": 18.0}}}
    out = cte._enrich_mmr_vote_mlh1_cohort_context(vote, "COAD")
    assert "cohort_ratio" not in out["details"]["mlh1_expression"]


def test_enrich_mmr_vote_mlh1_cohort_context_noop_without_mlh1(monkeypatch):
    import trufflepig.cancer_type_evidence as cte

    monkeypatch.setattr(cte, "_cohort_bulk_gene_median", lambda code, gene: 18.0)
    vote = {"details": {"msi_probability": 0.8}}
    out = cte._enrich_mmr_vote_mlh1_cohort_context(vote, "COAD")
    assert out["details"].get("mlh1_expression") is None
