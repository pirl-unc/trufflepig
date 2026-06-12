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


def test_translation_table_covers_every_pirlygenes_lineage_group():
    """Guard: ``broad_lineage`` consumes pirlygenes' ``cancer_lineage_group`` and
    translates it (``_PIRLYGENES_GROUP_TO_BROAD``). If pirlygenes adds a NEW coarse
    group, that translation silently falls back to the local family map — so pin it
    here, turning a silent blind spot into a loud failure (like the new-code
    coverage tests)."""
    from pirlygenes.gene_sets_cancer import cancer_lineage_groups
    from trufflepig.cancer_type_ontology import _PIRLYGENES_GROUP_TO_BROAD

    upstream = {str(v).strip().lower() for v in cancer_lineage_groups().values()}
    missing = upstream - set(_PIRLYGENES_GROUP_TO_BROAD)
    assert not missing, (
        f"pirlygenes lineage groups not in the broad_lineage translation table: "
        f"{sorted(missing)} — add them to _PIRLYGENES_GROUP_TO_BROAD"
    )


def test_broad_lineage_follows_pirlygenes_grouping():
    """Pin the centralization itself (the coverage test passes under the fallback
    too, so it wouldn't catch a revert). Includes the NBL->Embryonal move."""
    assert broad_lineage("NBL") == "embryonal"  # pirlygenes moved NBL out of NE
    assert broad_lineage("SCLC") == "neuroendocrine"
    assert broad_lineage("SARC") == "mesenchymal"
    assert broad_lineage("COAD") == "epithelial"
    assert broad_lineage("DLBC") == "hematolymphoid"
    assert broad_lineage("SKCM") == "melanocytic"


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


def _cohort_median_df(code):
    """A TCGA cohort median as a pseudo-sample (battery-test pattern)."""
    from trufflepig.reference import pan_cancer_expression

    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{code}_TPM"].astype(float),
        }
    )


def test_df_path_auto_computes_marker_channels_from_df():
    """Integration: when df_gene_expr is supplied the marker channels (recall +
    epithelial exclusion) auto-compute FROM THE DF. Pass cheap pre-computed
    ranked_rows so the (separately battery-tested) signature engine is skipped —
    this isolates and covers the need_sample / marker_hk_median-from-df wiring
    without the ~30s full-engine scoring.
    """
    df = _cohort_median_df("COAD")  # real ensembl IDs + clean TPM for the markers
    rows = [
        {"code": "READ", "signature_score": 0.60, "support_geomean": 0.60},
        {"code": "SARC", "signature_score": 0.50, "support_geomean": 0.50},
        {"code": "COAD", "signature_score": 0.55, "support_geomean": 0.55},
    ]
    r = classify_cancer_type_ontology(df_gene_expr=df, ranked_rows=rows)
    # epithelial program present in COAD -> exclusion fires from the df, demoting
    # SARC out of the candidate set (proves df -> marker_hk_median -> evidence).
    assert any(t.startswith("[exclude]") for t in r.trace), r.trace
    assert "SARC" not in r.candidates, r.candidates
    assert {"COAD", "READ"} & set(r.candidates), r.candidates
    # a colon adenocarcinoma is not neuroendocrine -> recall stays silent
    assert not r.recall_notes


def test_df_path_requires_clean_input_columns():
    # guards that the df path actually reaches the loaders (KeyError on junk).
    bad = pd.DataFrame({"ensembl_gene_id": ["x"], "gene_symbol": ["Y"]})
    with pytest.raises(Exception):
        classify_cancer_type_ontology(df_gene_expr=bad)
