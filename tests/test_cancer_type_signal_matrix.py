from types import SimpleNamespace

import pandas as pd

from trufflepig.cancer_type_signal_matrix import (
    SIGNAL_MATRIX_COLUMNS,
    SIGNAL_SAMPLE_SUMMARY_COLUMNS,
    _md_cell,
    _ontology_layer,
    build_cancer_type_signal_matrix,
    build_signal_matrix_summary_markdown,
    build_signal_sample_summary,
    compact_signal_plot_rows,
    status_parent_code,
    subtype_parent_code,
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
                    {
                        "candidate_code": "BLCA",
                        "code": "epithelial",
                        "channel": "learned_expression_classifier",
                        "stage": "family",
                        "role": "hierarchical_compartment_vote",
                        "status": "admission_context",
                        "support": 0.90,
                        "details": {"learned_stage": "compartment"},
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
    epithelial = matrix[matrix["predicted_code"] == "epithelial"].iloc[0]
    assert epithelial["predicted_lineage"] == "solid"
    assert bool(epithelial["lineage_agrees_final"]) is True

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


def test_compact_signal_plot_collapses_orthogonal_status_variants():
    def row(
        *,
        source,
        code,
        support,
        role="",
        status="",
        rank=None,
        selected=False,
    ):
        out = {column: "" for column in SIGNAL_MATRIX_COLUMNS}
        out.update(
            {
                "sample": "case-msi",
                "final_call": "READ",
                "final_lineage": "solid",
                "signal_source": source,
                "signal_label": source,
                "predicted_code": code,
                "role": role,
                "status": status,
                "support": support,
                "confidence": support,
                "rank": rank,
                "selects_report_label": selected,
                "entity_agrees_final": code.startswith("READ"),
                "lineage_agrees_final": True,
                "is_blocked": False,
                "is_context_only": False,
                "details": "{}",
            }
        )
        return out

    matrix = pd.DataFrame(
        [
            row(source="fused_evidence", code="READ", support=2.4, selected=True),
            row(source="fused_evidence", code="READ_MSS", support=1.0),
            row(source="fused_evidence", code="COAD_MSS", support=1.7),
            row(source="fused_evidence", code="COAD", support=1.2),
            row(source="pan_cancer_signature_ranker", code="COAD_MSI", support=1.0, rank=1),
            row(source="pan_cancer_signature_ranker", code="COAD", support=0.8, rank=2),
            row(source="exact_expression_reference", code="READ_MSS", support=0.93),
            row(source="exact_expression_reference", code="READ", support=0.94),
            row(source="exact_expression_reference", code="HNSC_HPV_pos", support=0.82),
            row(source="pan_cancer_signature_ranker", code="SARC_ASPS", support=0.91),
            row(source="pan_cancer_signature_ranker", code="SARC_DSRCT", support=0.90),
            row(source="rare_fusion_anchor", code="SARC_ASPS", support=0.89),
            row(source="rare_fusion_anchor", code="SARC_DSRCT", support=0.88),
        ]
    )

    compact = compact_signal_plot_rows(matrix, max_rows=20)
    labels = compact["display_label"].tolist()

    assert status_parent_code("COAD_MSI") == "COAD"
    assert status_parent_code("HNSC_HPV_pos") == "HNSC"
    assert subtype_parent_code("SARC_ASPS") == "SARC"
    assert labels.count("Fused evidence: READ") == 1
    assert not any(label == "Fused evidence: READ_MSS" for label in labels)
    assert "Fused evidence: COAD (status row COAD_MSS)" in labels
    assert "Fused evidence: COAD" not in labels
    assert "Pan-cancer signature ranker rank 1: COAD (status row COAD_MSI)" in labels
    assert "Exact expression reference: READ" in labels
    assert not any(
        label == "Exact expression reference: READ (status row READ_MSS)"
        for label in labels
    )
    assert "Pan-cancer signature ranker: SARC (subtype row SARC_ASPS)" in labels
    assert not any("SARC_DSRCT" in label for label in labels)
    assert "Rare fusion subtype anchor: SARC (subtype row SARC_ASPS)" in labels


def test_ontology_layer_keeps_base_entity_winner_an_entity():
    """A fused_evidence / learned_* selection reports the ``exact_subtype`` decision stage even
    when it wins a base entity. The layer of a base-entity code (no ``_`` suffix) must stay
    ``entity`` — only a real subtype/status code sits at the ``subtype`` layer."""
    # Fused headline COAD: exact_subtype STAGE, but COAD is a base entity → entity layer.
    assert _ontology_layer("exact_subtype", "fused_evidence", "COAD") == "entity"
    # A true fine subtype keeps the subtype layer.
    assert _ontology_layer("exact_subtype", "fused_evidence", "READ_MSI") == "subtype"
    # An orthogonal-status code also stays subtype.
    assert _ontology_layer("orthogonal_state", "fused_evidence", "COAD_MSI") == "subtype"
    # Role-based hierarchical entity stays entity regardless of stage.
    assert _ontology_layer("exact_subtype", "hierarchical_entity", "LUAD") == "entity"
    # Coarse-type stage is always an entity.
    assert _ontology_layer("coarse_type", "", "BRCA") == "entity"


def test_summary_markdown_survives_all_nan_selector_columns():
    """Regression: a freshly-regenerated single-sample matrix whose selected_by / final_call /
    reference_call columns are entirely empty must not crash the summary builder — the old
    ``.dropna().iloc[0]`` raised IndexError (single positional indexer out-of-bounds)."""
    single = pd.DataFrame([{c: None for c in SIGNAL_MATRIX_COLUMNS}])
    single["sample"] = "case-x"
    single["support"] = 0.0
    md = build_signal_matrix_summary_markdown(single)
    assert "Final call" in md  # renders the single-sample layout with em-dash placeholders

    multi = pd.DataFrame([{c: None for c in SIGNAL_MATRIX_COLUMNS} for _ in range(2)])
    multi["sample"] = ["a", "b"]
    multi["support"] = [0.0, 0.0]
    md_multi = build_signal_matrix_summary_markdown(multi)
    assert "| Sample |" in md_multi  # multi-sample table renders too


def test_md_cell_escapes_pipe_and_flattens_newlines():
    """A channel rationale carrying a literal ``|`` or newline must not split or break the
    markdown table it is interpolated into."""
    assert _md_cell("STAD | READ conflict") == "STAD \\| READ conflict"
    assert _md_cell("line one\nline two") == "line one line two"
    assert _md_cell("carriage\r\nreturn") == "carriage return"
    assert _md_cell(None) == ""
    assert _md_cell("") == ""


def test_summary_markdown_is_not_broken_by_pipe_in_rationale():
    """End-to-end: a details rationale with a pipe must not inject a stray table column."""
    import json

    matrix = pd.DataFrame(
        [
            {
                "sample": "case-x",
                "final_call": "COAD",
                "selected_by": "fused_evidence",
                "reference_call": "COAD",
                "signal_label": "Learned classifier",
                "signal_source": "learned_expression_classifier",
                "predicted_code": "STAD",
                "ontology_layer": "entity",
                "status": "blocked",
                "support": 0.42,
                "entity_agrees_final": False,
                "lineage_agrees_final": False,
                "selects_report_label": False,
                "is_context_only": False,
                "is_blocked": True,
                "details": json.dumps({"rationale": "conflicts with READ | CRC context"}),
            }
        ]
    )
    md = build_signal_matrix_summary_markdown(matrix)
    row_lines = [ln for ln in md.splitlines() if ln.startswith("| Learned classifier")]
    assert len(row_lines) == 1
    # After stripping the escaped ``\|`` the row has exactly 8 structural delimiters
    # (7 columns) — the rationale's pipe did not inject a stray column.
    structural = row_lines[0].replace("\\|", "")
    assert structural.count("|") == 8
    assert "READ \\| CRC context" in row_lines[0]
