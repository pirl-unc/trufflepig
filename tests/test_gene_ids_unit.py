from types import SimpleNamespace

import pirlygenes.gene_ids as gi


class FakeTx:
    def __init__(self, is_protein_coding=True):
        self.is_protein_coding = is_protein_coding


class FakeGene:
    def __init__(self, gene_id, name, gene_name=None, num_pc=1):
        self.id = gene_id
        self.name = name
        self.gene_name = gene_name or name
        self.transcripts = [FakeTx(True) for _ in range(num_pc)]

    def __repr__(self):
        return f"FakeGene({self.id},{self.name})"


class FakeGenome:
    def __init__(self, release, gene_by_id_map=None, tx_by_id_map=None, by_name=None):
        self.release = release
        self._gene_by_id_map = gene_by_id_map or {}
        self._tx_by_id_map = tx_by_id_map or {}
        self._by_name = by_name or {}

    def gene_by_id(self, gene_id):
        if gene_id not in self._gene_by_id_map:
            raise KeyError(gene_id)
        return self._gene_by_id_map[gene_id]

    def transcript_by_id(self, tx_id):
        if tx_id not in self._tx_by_id_map:
            raise KeyError(tx_id)
        return self._tx_by_id_map[tx_id]

    def genes(self):
        return list(self._gene_by_id_map.values())

    def transcripts(self):
        return [
            SimpleNamespace(id=tid, gene_name=tx.gene_name)
            for tid, tx in self._tx_by_id_map.items()
        ]

    def genes_by_name(self, name):
        return self._by_name.get(name, [])


def test_pick_best_gene_prefers_more_protein_coding():
    g1 = FakeGene("ENSG1", "TP53", num_pc=1)
    g2 = FakeGene("ENSG2", "TP53", num_pc=3)
    assert gi.pick_best_gene([g1, g2]).id == "ENSG2"


def test_lookup_functions_with_fake_genomes(monkeypatch):
    gene_a = FakeGene("ENSGA", "GENEA")
    tx_a = SimpleNamespace(gene_name="GENEA")
    by_name_gene = FakeGene("ENSGX", "CD276", gene_name="CD276")
    genomes = [
        FakeGenome(
            release=112,
            gene_by_id_map={"ENSGA": gene_a},
            tx_by_id_map={"ENST1": tx_a},
            by_name={
                "CD276": [by_name_gene],
                "B7-H3": [by_name_gene],
                "cd276": [by_name_gene],
            },
        )
    ]
    monkeypatch.setattr(gi, "genomes", genomes)
    monkeypatch.setattr(gi, "_indexes_built", False)
    monkeypatch.setattr(gi, "_gene_id_to_name", {})
    monkeypatch.setattr(gi, "_transcript_id_to_gene_name", {})
    monkeypatch.setattr(gi, "_gene_id_miss_cache", set())
    monkeypatch.setattr(gi, "_transcript_id_miss_cache", set())

    assert gi.find_gene_name_from_ensembl_gene_id("ENSGA", verbose=False) == "GENEA"
    assert (
        gi.find_gene_name_from_ensembl_transcript_id("ENST1", verbose=False) == "GENEA"
    )

    genome, gene = gi.find_gene_and_ensembl_release_by_name("CD276", verbose=False)
    assert genome.release == 112
    assert gene.id == "ENSGX"

    assert gi.find_gene_by_name_from_ensembl("CD276", verbose=False).id == "ENSGX"
    assert gi.find_gene_id_by_name_from_ensembl("CD276", verbose=False) == "ENSGX"
    assert gi.find_canonical_gene_id_and_name("CD276") == ("ENSGX", "CD276")


def test_lookup_functions_fall_back_to_older_release_on_latest_miss(monkeypatch):
    older_gene = FakeGene("ENSGOLD", "OLDER")
    older_tx = SimpleNamespace(gene_name="OLDER")
    genomes = [
        FakeGenome(release=112),
        FakeGenome(
            release=111,
            gene_by_id_map={"ENSGOLD": older_gene},
            tx_by_id_map={"ENSTOLD": older_tx},
        ),
    ]
    monkeypatch.setattr(gi, "genomes", genomes)
    monkeypatch.setattr(gi, "_indexes_built", False)
    monkeypatch.setattr(gi, "_gene_id_to_name", {})
    monkeypatch.setattr(gi, "_transcript_id_to_gene_name", {})
    monkeypatch.setattr(gi, "_gene_id_miss_cache", set())
    monkeypatch.setattr(gi, "_transcript_id_miss_cache", set())

    assert gi.find_gene_name_from_ensembl_gene_id("ENSGOLD", verbose=False) == "OLDER"
    assert (
        gi.find_gene_name_from_ensembl_transcript_id("ENSTOLD", verbose=False)
        == "OLDER"
    )


def test_find_canonical_gene_ids_and_names(monkeypatch):
    mapping = {
        "A": ("ENSGA", "A"),
        "B": (None, None),
    }
    monkeypatch.setattr(
        gi,
        "find_canonical_gene_id_and_name",
        lambda name: mapping.get(name, (None, None)),
    )
    ids, names = gi.find_canonical_gene_ids_and_names(["A", "B"])
    assert ids == ["ENSGA", None]
    assert names == ["A", None]
