"""Unit tests for the interpretable cancer-type ontology walk.

These exercise the reasoning layer directly via synthetic per-cohort score maps
(passed as ``ranked_rows``), so they are fast and do not load reference matrices
or invoke the signature engine.
"""

from pirlygenes.gene_sets_cancer import cancer_type_registry
from trufflepig.cancer_type_ontology import (
    broad_lineage,
    compatible_compartments,
    lineage_compatibility,
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


def test_secondary_lineage_keys_are_registry_codes():
    """Every ``_SECONDARY_LINEAGES`` key must be a supported registry code. A typo'd
    code (the historical ``WT``/``RBL``/``SARC_SS``/``PBL`` bug) silently never applies
    its secondary lineages, re-exposing those tumors to the demotion/compartment gates
    the table exists to prevent — so pin the keys to real codes loudly."""
    from trufflepig.cancer_type_ontology import _SECONDARY_LINEAGES

    codes = set(cancer_type_registry()["code"].astype(str))
    unsupported = [k for k in _SECONDARY_LINEAGES if k not in codes]
    assert not unsupported, (
        f"_SECONDARY_LINEAGES keys are not registry codes: {unsupported} — "
        f"a typo'd code silently never applies its secondary lineages"
    )


def test_broad_to_compartment_map_is_bijective_over_registry():
    """``compatible_compartments`` crosses the broad-lineage vocabulary into the
    compartment (``cancer_lineage_group``) vocabulary via ``_BROAD_TO_COMPARTMENT``.
    That crossing is only sound if the map agrees with ``cancer_lineage_group`` on every
    primary — pin it so a vocabulary drift on either side fails loudly here."""
    from pirlygenes.gene_sets_cancer import cancer_lineage_group

    from trufflepig.cancer_type_ontology import _BROAD_TO_COMPARTMENT

    codes = cancer_type_registry()["code"].astype(str).tolist()
    mismatches = []
    for code in codes:
        clg = cancer_lineage_group(code)
        mapped = _BROAD_TO_COMPARTMENT.get(broad_lineage(code))
        if clg and mapped != clg:
            mismatches.append((code, broad_lineage(code), mapped, clg))
    assert not mismatches, (
        f"_BROAD_TO_COMPARTMENT disagrees with cancer_lineage_group: {mismatches[:10]}"
    )


def test_compatible_compartments_carries_secondary_programs():
    """A polyphenotypic tumor is in-compartment for BOTH its primary and secondary
    program; a unilineage tumor for only its primary. This is what stops a confident
    whole-profile compartment call on the secondary program (HEPB→Epithelial,
    SARC_SYN→Epithelial) from marking the correct candidate out-of-compartment."""
    # HEPB: embryonal blastoma with hepatic (epithelial) differentiation.
    assert compatible_compartments("HEPB") == frozenset({"Embryonal", "Epithelial"})
    # Synovial sarcoma: biphasic mesenchymal + epithelial.
    assert compatible_compartments("SARC_SYN") == frozenset({"Sarcoma", "Epithelial"})
    # Unilineage carcinoma: primary only.
    assert compatible_compartments("COAD") == frozenset({"Epithelial"})
    # Subtype codes inherit the parent's secondary set via the suffix fallback.
    assert "epithelial" in lineage_compatibility("SARC_SYN")
