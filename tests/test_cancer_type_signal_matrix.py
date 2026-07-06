from types import SimpleNamespace

from trufflepig.cancer_type_signal_matrix import (
    SIGNAL_MATRIX_COLUMNS,
    SIGNAL_SAMPLE_SUMMARY_COLUMNS,
    build_cancer_type_signal_matrix,
    build_signal_matrix_summary_markdown,
    build_signal_sample_summary,
)


def test_signal_matrix_surfaces_selector_ranker_learned_and_met_context():
    analysis = {
        "sample_id": "case-1",
        "cancer_type": "BLCA",
        "reference_cancer_type": "BLCA",
        "candidate_trace": [
            {
                "code": "HEPB",
                "support_fraction_of_top": 1.0,
                "support_geomean": 0.52,
                "signature_score": 0.67,
            },
            {
                "code": "BLCA",
                "support_fraction_of_top": 0.96,
                "support_geomean": 0.50,
                "signature_score": 0.75,
            },
        ],
        "inferred_site_context": {
            "site": "liver",
            "tissue": "liver",
            "score": 0.91,
            "message": "liver-associated host context",
        },
        "cancer_type_evidence": {
            "selected": {
                "cancer_type": "BLCA",
                "selected_by": "coarse_composition_reference",
                "reference_cancer_type": "BLCA",
            },
            "staged_evidence_graph": {
                "channels": [
                    {
                        "candidate_code": "BLCA",
                        "code": "BLCA",
                        "channel": "composition_reference",
                        "stage": "coarse_type",
                        "role": "independent_tissue_composition",
                        "status": "selected_report_label",
                        "support": 0.856,
                        "selects_report_label": True,
                        "details": {"rho": 0.823, "margin": 0.015},
                    },
                    {
                        "candidate_code": "BLCA",
                        "code": "BLCA",
                        "channel": "learned_expression_classifier",
                        "stage": "coarse_type",
                        "role": "hierarchical_entity_vote",
                        "status": "admission_context",
                        "support": 0.395,
                        "details": {
                            "learned_stage": "entity",
                            "top_predictions": [
                                {"code": "SARC_PEC", "probability": 0.40},
                                {"code": "BLCA", "probability": 0.06},
                            ],
                        },
                    },
                ]
            },
        },
    }
    decomp = SimpleNamespace(
        cancer_type="BLCA",
        template="met_liver",
        score=0.075,
        purity=0.64,
        reconstruction_error=0.2,
        template_tissue_score=0.8,
        template_site_factor=1.2,
        warnings=[],
        site_evidence={"site_supported": True},
    )

    matrix = build_cancer_type_signal_matrix(
        analysis,
        sample_id="case-1",
        decomp_results=[decomp],
    )

    assert list(matrix.columns) == SIGNAL_MATRIX_COLUMNS
    assert set(matrix["signal_source"]) >= {
        "pan_cancer_signature_ranker",
        "composition_reference",
        "learned_expression_classifier",
        "background_site_context",
        "expression_decomposition",
    }
    selected = matrix[matrix["selects_report_label"] == True]  # noqa: E712
    assert selected.iloc[0]["predicted_code"] == "BLCA"
    assert bool(selected.iloc[0]["entity_agrees_final"]) is True
    site = matrix[matrix["signal_source"] == "background_site_context"].iloc[0]
    assert site["ontology_layer"] == "context"
    assert bool(site["is_context_only"]) is True

    summary = build_signal_matrix_summary_markdown(matrix)
    assert "Final call" in summary
    assert "Composition Reference" in summary

    compact = build_signal_sample_summary(matrix)
    assert list(compact.columns) == SIGNAL_SAMPLE_SUMMARY_COLUMNS
    assert len(compact) == 1
    row = compact.iloc[0]
    assert row["sample"] == "case-1"
    assert row["final_call"] == "BLCA"
    assert row["signal_rows"] == len(matrix)
    assert row["pan_cancer_top"] == "HEPB"
    assert row["lineage_panel_top"] == ""
    assert row["background_site"] == "liver"
    assert row["decomposition_top"] == "BLCA"


def test_ranker_candidate_trace_does_not_infer_report_selection():
    analysis = {
        "sample_id": "case-ranker",
        "cancer_type": "READ",
        "candidate_trace": [
            {"code": "READ", "support_fraction_of_top": 1.0},
        ],
        "cancer_type_evidence": {
            "selected": {
                "cancer_type": "READ",
                "selected_by": "pan_cancer_signature_ranker",
                "reference_cancer_type": "READ",
            },
            "staged_evidence_graph": {
                "channels": [
                    {
                        "candidate_code": "READ",
                        "code": "READ",
                        "channel": "pan_cancer_signature_ranker",
                        "stage": "coarse_type",
                        "role": "top_ranked_candidate",
                        "status": "candidate_generation",
                        "support": 1.0,
                        "selects_report_label": False,
                    }
                ]
            },
        },
    }

    matrix = build_cancer_type_signal_matrix(analysis)

    ranker_rows = matrix[
        (matrix["signal_source"] == "pan_cancer_signature_ranker")
        & (matrix["predicted_code"] == "READ")
    ]
    assert not ranker_rows["selects_report_label"].any()
