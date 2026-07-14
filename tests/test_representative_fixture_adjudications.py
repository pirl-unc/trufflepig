"""Regression tests for the source-qualified representative accuracy gate."""

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_per_sample_confusion.py"
_SPEC = importlib.util.spec_from_file_location("eval_per_sample_confusion", _SCRIPT)
eval_confusion = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(eval_confusion)


def test_fixture_adjudications_are_traceable_and_source_grouped():
    rows = eval_confusion.load_fixture_adjudications()

    assert set(rows) == {"BRCA_rep02", "BRCA_Basal_rep02", "RB_rep03"}
    assert rows["BRCA_rep02"]["source_group_id"] == "TCGA-AC-A2QH-01"
    assert rows["BRCA_Basal_rep02"]["source_group_id"] == "TCGA-AC-A2QH-01"
    assert rows["BRCA_rep02"]["allowed_lineage_modes"] == (
        "solid",
        "mesenchymal",
    )
    assert rows["RB_rep03"]["benchmark_eligible"] is False
    assert all(str(row["source_issue"]).startswith("https://") for row in rows.values())


def test_match_level_uses_recursive_canonical_entity_compatibility():
    assert eval_confusion.match_level("COAD_MSI", "CRC") == "subtype"
    # Corresponding colon/rectal molecular states remain distinct entities in
    # the canonical ontology; this is a real GI sibling confusion, not a metric
    # shortcut.
    assert eval_confusion.match_level("COAD_MSS", "READ_MSS") == "lineage"


def test_qualified_gate_is_conservative_and_deduplicates_source_aliases():
    adjudications = {
        "BRCA_rep02": {
            "truth_code": "BRCA",
            "source_group_id": "metaplastic",
            "benchmark_eligible": True,
            "allowed_lineage_modes": ("solid", "mesenchymal"),
            "adjudication": "dual",
            "source_issue": "issue-a",
        },
        "BRCA_Basal_rep02": {
            "truth_code": "BRCA_Basal",
            "source_group_id": "metaplastic",
            "benchmark_eligible": True,
            "allowed_lineage_modes": ("solid", "mesenchymal"),
            "adjudication": "dual",
            "source_issue": "issue-a",
        },
        "RB_rep03": {
            "truth_code": "RB",
            "source_group_id": "rb-fail",
            "benchmark_eligible": False,
            "allowed_lineage_modes": ("embryonal",),
            "adjudication": "qc_fail",
            "source_issue": "issue-b",
        },
    }
    results = [
        ("BRCA", "BRCA_rep02", "SARC_EPITH", "miss"),
        ("BRCA_Basal", "BRCA_Basal_rep02", "SARC_EPITH", "miss"),
        ("RB", "RB_rep03", "SARC_LPS_UNSPEC", "miss"),
        ("KIRC", "KIRC_rep01", "KIRC", "exact"),
    ]

    summary = eval_confusion.qualified_gate_summary(results, adjudications)

    assert summary["n"] == 2
    assert summary["lineage_correct"] == 2
    # The metaplastic aliases are still entity-wrong and conservatively score
    # their one source group as wrong at the entity level.
    assert summary["entity_correct"] == 1
    assert summary["aliases_collapsed"] == 1
    assert [row["sample"] for row in summary["excluded"]] == ["RB_rep03"]


def test_qualified_gate_rejects_truth_drift():
    adjudications = {
        "sample": {
            "truth_code": "BRCA",
            "source_group_id": "group",
            "benchmark_eligible": True,
            "allowed_lineage_modes": ("solid",),
        }
    }

    try:
        eval_confusion.qualified_gate_summary(
            [("OV", "sample", "OV", "exact")], adjudications
        )
    except ValueError as exc:
        assert "truth mismatch" in str(exc)
    else:
        raise AssertionError("truth drift must fail loudly")


def test_qualified_gate_does_not_count_same_organ_cross_lineage_as_lineage_correct():
    summary = eval_confusion.qualified_gate_summary(
        [("UCEC", "sample", "UCS", "organ")], {}
    )

    assert summary["n"] == 1
    assert summary["lineage_correct"] == 0
