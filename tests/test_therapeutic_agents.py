"""Tests for the trufflepig-owned therapeutic-agent registry (#52) and its
off-context surfacing (#47)."""

import pandas as pd

from trufflepig.common import ensembl_id_to_symbol_map
from trufflepig.therapeutic_agents import (
    MODALITIES,
    agents_for_target,
    all_target_genes,
    best_agent_for_target,
    druggable_target_genes,
    is_druggable_target,
    target_agent_summary,
    target_annotation,
    target_liability_note,
    therapeutic_agents,
    therapeutic_targets,
)
from trufflepig.reporting import cross_cancer_target_index, offcontext_known_targets


def test_registry_loads_with_required_columns():
    df = therapeutic_agents()
    assert len(df) >= 140
    for col in (
        "agent",
        "target_gene",
        "modality",
        "highest_phase",
        "fda_approved",
        "key_pmids",
    ):
        assert col in df.columns
    # Every row names an agent and a target gene.
    assert (df["agent"].str.strip() != "").all()
    assert (df["target_gene"].str.strip() != "").all()


def test_every_agent_uses_the_controlled_modality_vocab():
    df = therapeutic_agents()
    bad = sorted(set(df["modality"]) - set(MODALITIES))
    assert not bad, f"agents use uncontrolled modality values: {bad}"


def test_every_druggable_target_has_at_least_one_agent():
    for gene in druggable_target_genes():
        agents = agents_for_target(gene)
        assert agents, f"{gene} is druggable but has no agent rows"
        assert all(a.agent for a in agents)


def test_druggable_targets_are_known_gene_symbols_so_they_are_rankable():
    # "Reasoning-available": a target must resolve to a real gene symbol so the
    # engine can rank it by expression (#47 availability criterion). A target on
    # a byte-identical-protein locus resolves through its proteoform key (NY-ESO-1
    # "CTAG1B" -> "CTAG1A/B"), so fold before checking membership.
    from trufflepig.common import fold_panel_symbols

    known = set(ensembl_id_to_symbol_map().values())
    unresolved = sorted(
        g for g in druggable_target_genes() if fold_panel_symbols([g])[0] not in known
    )
    assert not unresolved, f"druggable targets not resolvable to a gene symbol: {unresolved}"


def test_flagship_targets_resolve_to_expected_agents():
    # PSMA radioligand, DLL3 T-cell engager, TROP2 ADC — the #47 flagships.
    assert is_druggable_target("FOLH1")
    assert any("RLT" == a.modality for a in agents_for_target("FOLH1"))

    dll3 = best_agent_for_target("DLL3")
    assert dll3 is not None and dll3.modality == "TCE"

    trop2 = agents_for_target("TACSTD2")
    assert any(a.modality == "ADC" for a in trop2)

    # A Tier-D "invisible" target (no pirlygenes key-genes row) is now reachable.
    assert is_druggable_target("ROR1")
    assert "ROR1" in target_agent_summary("ROR1") or agents_for_target("ROR1")


def test_approved_agents_carry_approval_metadata():
    df = therapeutic_agents()
    approved = df[df["fda_approved"].str.lower() == "yes"]
    assert len(approved) >= 20
    # At least the approval clause renders an FDA-approved status.
    a = best_agent_for_target("DLL3")
    assert a.fda_approved and "FDA-approved" in a.approval_clause()


def test_cross_cancer_index_unions_registry_targets():
    index = cross_cancer_target_index()
    # pirlygenes-curated flagship still present...
    assert "FOLH1" in index and any(e["cancer_code"] == "PRAD" for e in index["FOLH1"])
    # ...and a registry-only Tier-D target now appears via the registry union.
    assert "ROR1" in index
    assert any(e.get("source") == "registry" for e in index["ROR1"])
    # Registry entries carry modality + approval context.
    dll3 = index.get("DLL3", ())
    assert any(e.get("modality") for e in dll3)


def test_target_annotations_cover_all_targets_with_localization():
    df = therapeutic_targets()
    assert len(df) >= 65
    valid_loc = {"surface", "surface_and_secreted", "intracellular_HLA_presented"}
    assert set(df["localization"]) <= valid_loc
    # Every druggable target also carries a target-level annotation (so the
    # normal-tissue guardrail can fire wherever an agent surfaces).
    annotated = set(df["target_gene"])
    missing = sorted(g for g in druggable_target_genes() if g not in annotated)
    assert not missing, f"druggable targets lacking a target annotation: {missing}"


def test_all_target_genes_is_reasoning_available_superset():
    # The full curated target universe (incl. no-binder targets) is the
    # reasoning-available set; it is a superset of the druggable ones.
    universe = all_target_genes()
    assert druggable_target_genes() <= universe
    # A no-binder target (e.g. ALPPL2) is still reasoning-available.
    assert "ALPPL2" in universe


def test_target_liability_note_surfaces_normal_tissue_risk():
    annotation = target_annotation("CA9")
    assert annotation is not None and annotation.is_surface
    note = target_liability_note("CA9")
    assert note.startswith("normal-tissue caveat:")
    assert "GI" in note or "biliary" in note.lower() or "stomach" in note.lower()
    # A gene with no curated target annotation yields no caveat.
    assert target_liability_note("GAPDH") == ""


def test_offcontext_surfaces_registry_only_target_when_high():
    # ROR1 is not on a melanoma panel and is registry-only, but high expression
    # should surface it with its agent.
    ranges = pd.DataFrame([{"symbol": "ROR1", "attr_tumor_tpm": 40.0}])
    hits = offcontext_known_targets(ranges, panel_symbols={"PMEL"})
    symbols = [h["symbol"] for h in hits]
    assert "ROR1" in symbols
    hit = next(h for h in hits if h["symbol"] == "ROR1")
    assert any(e.get("agent") for e in hit["indications"])
