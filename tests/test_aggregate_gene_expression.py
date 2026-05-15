from types import SimpleNamespace

import pandas as pd

import trufflepig.aggregate_gene_expression as ag
# Implementation lives in pirlygenes since the 5.1 reorganization;
# monkey-patches that target the loaders for transcript→gene resolution
# need to land on the pirlygenes module, not the trufflepig wrapper.
import pirlygenes.expression.aggregate as pg_ag


def test_expanded_tx_map_adds_versionless_keys():
    out = ag._expanded_tx_map({"ENST1.1": "A", "ENST2": "B"})
    assert out["ENST1.1"] == "A"
    assert out["ENST1"] == "A"
    assert out["ENST2"] == "B"


def test_aggregate_gene_expression_with_resolution(monkeypatch):
    df = pd.DataFrame(
        {
            "transcript_id": ["tx1.1", "tx2", "tx3", "tx4"],
            "tpm": [1.0, 2.0, 3.0, 4.0],
        }
    )

    monkeypatch.setattr(
        pg_ag,
        "find_gene_name_from_ensembl_transcript_id",
        lambda tx, verbose=False: "GENE3" if tx == "tx3" else None,
    )
    monkeypatch.setattr(pg_ag, "extra_tx_mappings", {"tx4": "GENE4"})

    def fake_meta_lookup(gene_name):
        gene_id = {
            "GENE1": "ENSG1",
            "GENE2": "ENSG2",
            "GENE3": "ENSG3",
            "GENE4": "ENSG4",
        }[gene_name]
        genome = SimpleNamespace(release=112)
        gene = SimpleNamespace(id=gene_id)
        return (genome, gene)

    monkeypatch.setattr(pg_ag, "find_gene_and_ensembl_release_by_name", fake_meta_lookup)

    out = ag.aggregate_gene_expression(
        df,
        tx_to_gene_name={"tx1": "GENE1", "tx2": "GENE2"},
        verbose=False,
        progress=False,
    )

    assert set(out["gene"]) == {"GENE1", "GENE2", "GENE3", "GENE4"}
    assert set(out["gene_id"]) == {"ENSG1", "ENSG2", "ENSG3", "ENSG4"}
    assert set(out["ensembl_release"]) == {112}


def test_aggregate_gene_expression_with_unresolved_transcripts(monkeypatch):
    df = pd.DataFrame({"transcript": ["u1", "u2"], "tpm": [2.5, 0.5]})
    monkeypatch.setattr(
        pg_ag,
        "find_gene_name_from_ensembl_transcript_id",
        lambda tx, verbose=False: None,
    )
    monkeypatch.setattr(pg_ag, "extra_tx_mappings", {})
    monkeypatch.setattr(pg_ag, "find_gene_and_ensembl_release_by_name", lambda g: None)

    out = ag.aggregate_gene_expression(
        df, tx_to_gene_name={}, verbose=True, progress=False
    )
    assert list(out.columns) == ["gene", "TPM", "gene_id", "ensembl_release"]
    assert out.empty
