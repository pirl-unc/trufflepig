#!/usr/bin/env python3

from pyensembl.shell import collect_all_installed_ensembl_releases

from pirlygenes import load_all_dataframes
from pirlygenes.load_dataset import get_data

_human_genomes = sorted(
    [
        g
        for g in collect_all_installed_ensembl_releases()
        if g.species.latin_name == "homo_sapiens"
    ],
    reverse=True,
    key=lambda g: g.release,
)


def _load_ensembl_id_aliases():
    """Return {alt_haplotype_id: primary_contig_id} mapping from bundled data.

    Handles two categories of Ensembl gene IDs that don't resolve in pyensembl's
    default indexing:
    1. Alt-haplotype contigs (e.g. HLA-A on HSCHR6_MHC_DBB_CTG1 → primary HLA-A)
    2. Retired IDs with a documented successor
    """
    try:
        df = get_data("ensembl-id-aliases")
        return dict(zip(df["alt_haplotype_id"], df["primary_contig_id"]))
    except Exception:
        return {}


def _resolve_in_any_release(gene_id, aliases=None):
    """Check if gene_id (or its alias) resolves in any installed Ensembl release."""
    aliases = aliases or {}
    candidates = [gene_id]
    if gene_id in aliases:
        candidates.append(aliases[gene_id])
    for candidate in candidates:
        for genome in _human_genomes:
            try:
                gene = genome.gene_by_id(candidate)
                if gene is not None:
                    return True
            except Exception:
                pass
    return False


def _split_ids(value):
    if not isinstance(value, str):
        return []
    return [token.strip() for token in value.split(";") if token.strip()]


def _collect_ensembl_gene_ids(df):
    id_columns = [
        c for c in df.columns if ("Ensembl_Gene_ID" in c or c.endswith("_Gene_IDs"))
    ]
    result = set()
    for col in id_columns:
        for value in df[col]:
            for token in _split_ids(value):
                if token.startswith("ENSG"):
                    result.add(token)
    return result


# Ensembl retires gene IDs over time (~0.5% per release).  The test tolerates
# a small unresolvable rate per dataset since Ensembl updates are outside
# our control.  Datasets with a larger share of unresolvable IDs (e.g.
# surface-proteins includes alt-haplotype MHC/KIR genes that don't have
# primary-contig equivalents) can list per-dataset thresholds.
_MAX_UNRESOLVABLE_FRACTION = 0.02  # 2% default

_PER_DATASET_TOLERANCE = {
    # surface-proteins includes ~8 alt-haplotype-only genes (HLA-DRB3/4,
    # KIR family) that have no primary-contig equivalent plus ~16 truly
    # retired IDs with no Ensembl successor.  49/2700 ≈ 1.8% with aliases.
    "surface-proteins": 0.025,
}


def test_all_gene_ids_resolve_in_ensembl():
    """
    Validate that all Ensembl gene IDs in packaged CSVs resolve via pyensembl.

    Applies the bundled alt-haplotype/retirement alias map first (e.g.
    HLA-A on alt contig → HLA-A primary).  A small per-dataset fraction of
    unresolvable IDs is tolerated (retirements, alt-haplotype-only genes).

    Checks all installed human Ensembl releases, not just the default GRCh38.
    Recommend having at least Ensembl 110 installed for reasonable coverage
    of alt-haplotype genes that have been added/revised since earlier releases.
    """
    aliases = _load_ensembl_id_aliases()

    per_dataset_missing = {}
    per_dataset_total = {}
    for name, df in load_all_dataframes():
        stem = name.replace(".csv", "")
        # Don't self-check the aliases file
        if stem == "ensembl-id-aliases":
            continue
        dataset_ids = _collect_ensembl_gene_ids(df)
        if not dataset_ids:
            continue
        missing = sorted(
            gid
            for gid in dataset_ids
            if not _resolve_in_any_release(gid, aliases=aliases)
        )
        per_dataset_missing[stem] = missing
        per_dataset_total[stem] = len(dataset_ids)

    assert per_dataset_total, "No Ensembl gene IDs were found in packaged CSVs."

    failures = []
    for stem, missing in per_dataset_missing.items():
        total = per_dataset_total[stem]
        tolerance = _PER_DATASET_TOLERANCE.get(stem, _MAX_UNRESOLVABLE_FRACTION)
        fraction = len(missing) / total if total else 0.0
        if fraction > tolerance:
            failures.append(
                f"{stem}: {len(missing)}/{total} unresolvable ({fraction:.1%} > {tolerance:.1%}). "
                f"First few: {missing[:5]}"
            )

    assert not failures, "\n".join(failures)
