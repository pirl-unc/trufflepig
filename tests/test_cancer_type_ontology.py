"""Unit tests for the interpretable cancer-type ontology walk.

These exercise the reasoning layer directly via synthetic per-cohort score maps
(passed as ``ranked_rows``), so they are fast and do not load reference matrices
or invoke the signature engine.
"""

import pandas as pd
import pytest

from pirlygenes.gene_sets_cancer import cancer_type_registry
from trufflepig.cancer_type_ontology import (
    DEFAULT_MARGINS,
    broad_lineage,
    classify_cancer_type_ontology,
    ontology_path,
)


def _rows(score_map):
    return [
        {"code": code, "signature_score": score, "support_geomean": score}
        for code, score in score_map.items()
    ]


def _classify(score_map, **kw):
    return classify_cancer_type_ontology(ranked_rows=_rows(score_map), **kw)


def test_every_registry_code_has_a_broad_lineage():
    """Coverage contract: all ~145 pirlygenes codes map onto the ontology."""
    codes = cancer_type_registry()["code"].astype(str).tolist()
    uncovered = [c for c in codes if broad_lineage(c) == "other"]
    assert not uncovered, f"codes with no broad lineage: {uncovered}"


def test_carcinoma_path_descends_to_organ_and_leaf():
    assert ontology_path("READ") == ["root", "epithelial", "glandular", "GI", "READ"]
    assert ontology_path("BLCA") == ["root", "epithelial", "urothelial", "bladder", "BLCA"]


def test_non_carcinoma_path_is_broad_then_code():
    assert ontology_path("SARC") == ["root", "mesenchymal", "SARC"]


def test_clean_leaf_resolution():
    r = _classify({"PRAD": 0.62, "BRCA": 0.30, "LUAD": 0.25, "SARC": 0.12})
    assert r.candidates == ["PRAD"]
    assert r.resolved_level == "leaf"
    assert r.stopped_early is False


def test_leaf_tie_returns_neighbour_set_best_first():
    # READ barely ahead of COAD -> within the leaf margin -> report both.
    r = _classify({"READ": 0.55, "COAD": 0.53, "STAD": 0.30, "LUAD": 0.25})
    assert r.stopped_early is True
    assert r.resolved_level == "leaf"
    assert set(r.candidates) == {"READ", "COAD"}
    assert r.candidates[0] == "READ"  # most likely first


def test_leaf_resolves_when_margin_cleared():
    # READ ahead of COAD by 0.097 > leaf margin 0.04 -> single call.
    r = _classify({"READ": 0.557, "COAD": 0.460, "STAD": 0.30})
    assert r.candidates == ["READ"]
    assert r.stopped_early is False


def test_broad_tie_stops_at_lineage():
    r = _classify({"PRAD": 0.520, "SARC": 0.513, "DLBC": 0.30})
    assert r.stopped_early is True
    assert r.resolved_level == "broad"
    assert set(r.candidates) == {"PRAD", "SARC"}


def test_margins_are_tunable():
    score_map = {"READ": 0.557, "COAD": 0.460, "STAD": 0.30}
    # widen the leaf margin past the READ/COAD gap -> now a tie.
    r = _classify(score_map, margins={"leaf": 0.2})
    assert r.stopped_early is True
    assert set(r.candidates) == {"READ", "COAD"}


def test_trace_is_human_readable_and_per_step():
    r = _classify({"PRAD": 0.62, "BRCA": 0.30, "SARC": 0.12})
    assert r.trace and all(isinstance(line, str) for line in r.trace)
    assert len(r.steps) == len(r.trace)
    # the first decision is the broad-lineage split
    assert r.steps[0].level == "broad"


def test_empty_scores_yield_no_candidates():
    r = _classify({})
    assert r.candidates == []


def test_requires_input():
    with pytest.raises(ValueError):
        classify_cancer_type_ontology()


def test_default_margins_increase_with_level_breadth():
    assert DEFAULT_MARGINS["broad"] >= DEFAULT_MARGINS["leaf"]
