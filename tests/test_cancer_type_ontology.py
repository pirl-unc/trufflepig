"""Unit tests for the interpretable cancer-type ontology walk.

These exercise the reasoning layer directly via synthetic per-cohort score maps
(passed as ``ranked_rows``), so they are fast and do not load reference matrices
or invoke the signature engine.
"""

from pirlygenes.gene_sets_cancer import cancer_type_registry
from trufflepig.cancer_type_ontology import (
    broad_lineage,
    ontology_path,
)


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
