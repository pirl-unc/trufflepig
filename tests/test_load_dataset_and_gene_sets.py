from pathlib import Path

import pandas as pd
import pytest

from pirlygenes import load_dataset as ld
from pirlygenes import gene_sets_cancer as gsc


def test_get_data_from_provided_dict():
    fake = {"abc.csv": pd.DataFrame({"x": [1]})}
    df = ld.get_data("abc", _dataframes_dict=fake)
    assert list(df.columns) == ["x"]


def test_get_data_missing_raises():
    with pytest.raises(ValueError):
        ld.get_data("does-not-exist", _dataframes_dict={})


def test_get_data_returns_copy_not_cached_reference():
    # Mutating the returned DataFrame must not corrupt the cached copy that
    # subsequent callers will get (issue #29).
    first = ld.get_data("ADC-trials")
    first["_should_not_persist"] = 1
    first.drop(first.index, inplace=True)
    second = ld.get_data("ADC-trials")
    assert "_should_not_persist" not in second.columns
    assert len(second) > 0


def test_get_all_csv_paths_contains_core_dataset():
    paths = ld.get_all_csv_paths()
    assert any(Path(p).name == "ADC-trials.csv" for p in paths)


def test_expression_effect_rule_datasets_load():
    fusion_rules = gsc.fusion_expression_effect_rules_df()
    mutation_rules = gsc.mutation_expression_effect_rules_df()
    direct_fusion_rules = gsc.rare_cancer_fusion_rules_df()

    assert {"rule_id", "gene_a", "gene_b", "expected_up_genes"}.issubset(
        set(fusion_rules.columns)
    )
    assert {"rule_id", "alteration", "expected_up_genes"}.issubset(
        set(mutation_rules.columns)
    )
    assert "nutm1_rearranged_uncertain" in set(direct_fusion_rules["rule_id"])


def test_gene_set_field_lookup_variants(monkeypatch):
    # exercises plural/lower/upper/no-underscore candidate expansion
    fake_df = pd.DataFrame({"TUMORTARGETSYMBOLS": ["A;B"]})
    monkeypatch.setattr(gsc, "get_data", lambda name: fake_df)
    out = gsc.get_field_from_gene_set("x", ["Tumor_Target_Symbol"])
    assert out == {"A", "B"}


def test_all_gene_set_wrappers(monkeypatch):
    df_generic = pd.DataFrame(
        {
            "Symbol": ["GENE1;GENE2"],
            "Gene_ID": ["ENSG1;ENSG2"],
            "Tumor_Target_Symbols": ["GENE3"],
            "Tumor_Target_Ensembl_Gene_IDs": ["ENSG3"],
        }
    )

    def fake_get_data(name):
        return df_generic

    monkeypatch.setattr(gsc, "get_data", fake_get_data)

    # ADC
    assert gsc.therapy_target_gene_names("ADC-trials")
    assert gsc.therapy_target_gene_ids("ADC-trials")
    assert gsc.therapy_target_gene_names("ADC-approved")
    assert gsc.therapy_target_gene_ids("ADC-approved")
    assert gsc.therapy_target_gene_names("ADC")
    assert gsc.therapy_target_gene_ids("ADC")

    # TCR-T
    assert gsc.therapy_target_gene_names("TCR-T-trials")
    assert gsc.therapy_target_gene_ids("TCR-T-trials")
    assert gsc.therapy_target_gene_names("TCR-T")
    assert gsc.therapy_target_gene_ids("TCR-T")

    # CAR-T
    assert gsc.therapy_target_gene_names("CAR-T")
    assert gsc.therapy_target_gene_ids("CAR-T")

    # MuTE
    assert gsc.therapy_target_gene_names("multispecific-TCE")
    assert gsc.therapy_target_gene_ids("multispecific-TCE")

    # Bispecifics
    assert gsc.therapy_target_gene_names("bispecific-antibodies")
    assert gsc.therapy_target_gene_ids("bispecific-antibodies")
    assert gsc.therapy_target_gene_id_to_name("bispecific-antibodies-approved")
    assert gsc.therapy_target_gene_id_to_name("CAR-T-approved")

    # Radioligand + CTA
    assert gsc.therapy_target_gene_names("radioligand")
    assert gsc.therapy_target_gene_ids("radioligand")
    assert gsc.CTA_gene_names()
    assert gsc.CTA_gene_ids()


def test_cta_filtered_and_evidence():
    # CTA_gene_names() = filtered + expressed (excludes never_expressed)
    expressed_names = gsc.CTA_gene_names()
    expressed_ids = gsc.CTA_gene_ids()
    # CTA_filtered includes never_expressed
    filtered_names = gsc.CTA_filtered_gene_names()
    filtered_ids = gsc.CTA_filtered_gene_ids()
    # never_expressed = filtered - expressed
    never_expr_names = gsc.CTA_never_expressed_gene_names()
    never_expr_ids = gsc.CTA_never_expressed_gene_ids()
    # unfiltered = full superset
    all_names = gsc.CTA_unfiltered_gene_names()
    all_ids = gsc.CTA_unfiltered_gene_ids()
    # excluded = fail filter
    excluded_names = gsc.CTA_excluded_gene_names()

    assert expressed_names
    assert filtered_names
    assert all_names
    assert expressed_names < filtered_names  # expressed is strict subset of filtered
    assert filtered_names < all_names  # filtered is strict subset of unfiltered
    assert expressed_names & never_expr_names == set()  # no overlap
    assert expressed_names | never_expr_names == filtered_names  # partition
    assert filtered_names | excluded_names == all_names  # partition
    assert filtered_names & excluded_names == set()  # no overlap

    evidence_df = gsc.CTA_evidence()
    assert len(evidence_df) == len(all_names)
    expected_cols = [
        "protein_reproductive",
        "protein_thymus",
        "protein_reliability",
        "rna_reproductive",
        "rna_thymus",
        "protein_strict_expression",
        "rna_reproductive_frac",
        "rna_reproductive_and_thymus_frac",
        "rna_deflated_reproductive_frac",
        "rna_deflated_reproductive_and_thymus_frac",
        "rna_80_pct_filter",
        "rna_90_pct_filter",
        "rna_95_pct_filter",
        "rna_99_pct_filter",
        "filtered",
        "source_databases",
        "biotype",
        "Canonical_Transcript_ID",
        "rna_max_ntpm",
        "never_expressed",
    ]
    for col in expected_cols:
        assert col in evidence_df.columns, f"Missing column: {col}"


def test_cta_gene_id_to_name_preserves_row_pairing():
    mapping = gsc.CTA_gene_id_to_name()
    assert mapping["ENSG00000181323"] == "SPEM1"
    assert mapping["ENSG00000230594"] == "CT47A4"
    assert mapping["ENSG00000236126"] == "CT47A3"


def test_cta_partition():
    # gene_ids
    p = gsc.CTA_partition_gene_ids()
    assert isinstance(p, gsc.CTAPartitionSets)
    assert len(p.cta) > 200
    assert len(p.non_cta) > 15000
    assert p.cta & p.cta_never_expressed == set()
    assert p.cta & p.non_cta == set()
    assert p.cta_never_expressed & p.non_cta == set()

    # gene_names
    p2 = gsc.CTA_partition_gene_names()
    assert isinstance(p2, gsc.CTAPartitionSets)
    assert "MAGEA4" in p2.cta
    assert "TP53" in p2.non_cta

    # dataframes
    p3 = gsc.CTA_partition_dataframes()
    assert isinstance(p3, gsc.CTAPartitionDataFrames)
    assert "rna_deflated_reproductive_frac" in p3.cta.columns
    assert "Ensembl_Gene_ID" in p3.non_cta.columns

    # cta_excluded genes are in non_cta
    excluded_ids = gsc.CTA_excluded_gene_ids()
    assert excluded_ids.issubset(p.non_cta)


# ── Externalized gene sets (mitochondrial / culture / TME / degradation /
#    lineage / cancer-family) ─────────────────────────────────────────────


def test_mitochondrial_gene_loaders():
    df = gsc.mitochondrial_genes_df()
    # 13 protein-coding OXPHOS subunits + 2 rRNAs + 22 tRNAs = 37
    assert len(df) == 37
    assert set(df["Role"]) == {"protein_coding", "rRNA", "tRNA"}
    assert df["Ensembl_Gene_ID"].notna().all()
    assert "MT-CO1" in gsc.mitochondrial_gene_names()
    assert "MT-RNR1" in gsc.mitochondrial_gene_names(role="rRNA")
    assert "MT-CO1" not in gsc.mitochondrial_gene_names(role="rRNA")
    assert "MT-TA" in gsc.mitochondrial_gene_names(role="tRNA")
    assert len(gsc.mitochondrial_gene_names(role="protein_coding")) == 13
    assert len(gsc.mitochondrial_gene_names(role="rRNA")) == 2
    assert len(gsc.mitochondrial_gene_names(role="tRNA")) == 22
    assert len(gsc.mitochondrial_gene_ids()) == 37


def test_culture_stress_gene_loaders():
    df = gsc.culture_stress_genes_df()
    assert len(df) >= 20
    assert "HSPA1A" in gsc.culture_stress_gene_names()
    assert "HSPA1A" in gsc.culture_stress_gene_names(category="HSP")
    assert "LDHA" not in gsc.culture_stress_gene_names(category="HSP")


def test_tme_marker_loaders():
    df = gsc.tme_markers_df()
    assert len(df) >= 15
    expected_cell_types = {"T_cell", "B_cell", "myeloid", "fibroblast", "endothelial"}
    assert expected_cell_types.issubset(set(df["Cell_Type"]))
    assert "CD3D" in gsc.tme_marker_gene_names(cell_type="T_cell")
    assert "CD3D" not in gsc.tme_marker_gene_names(cell_type="fibroblast")


def test_degradation_gene_pairs_loader():
    pairs = gsc.degradation_gene_pairs()
    assert len(pairs) == 20
    short, long, ratio = pairs[0]
    assert isinstance(short, str) and isinstance(long, str)
    assert 0.5 < ratio < 1.5


def test_lineage_gene_loaders_cover_all_tcga_codes():
    from trufflepig.tumor_purity import TCGA_MEDIAN_PURITY

    by_code = gsc.lineage_genes_by_cancer_type()
    missing = [c for c in TCGA_MEDIAN_PURITY if c not in by_code]
    assert missing == [], f"Missing lineage genes for: {missing}"
    # PRAD sanity: classic prostate markers
    prad = set(gsc.lineage_gene_symbols("PRAD"))
    assert {"KLK3", "FOLH1", "TMPRSS2"}.issubset(prad)


def test_cancer_family_panel_loader():
    families = gsc.cancer_family_panels()
    # Stable core families that must always be present. The panel set grows as
    # new lineages are curated upstream (heme, neuroendocrine, embryonal, ...),
    # so assert the core is a subset rather than pinning an exact set — pinning
    # makes every upstream panel addition a spurious failure here.
    core = {
        "PROSTATE",
        "CRC",
        "GASTRIC",
        "SQUAMOUS",
        "ESCA_SQ",
        "MESENCHYMAL",
        "RENAL",
        "GLIAL",
        "MELANOCYTIC",
    }
    assert core <= set(families), f"missing core families: {core - set(families)}"
    # Every loaded panel must carry markers.
    for family, markers in families.items():
        assert markers, f"{family}: empty panel"
    # Behavioral checks: panels resolve to the right markers, not cross-contaminated.
    assert "KLK3" in gsc.cancer_family_panel("PROSTATE")
    assert "GFAP" not in gsc.cancer_family_panel("PROSTATE")


def test_gene_loaders_are_exported_from_package():
    """Core panel loaders should be reachable via `from pirlygenes import ...`."""
    import pirlygenes

    for name in [
        "mitochondrial_gene_ids",
        "culture_stress_gene_names",
        "tme_markers_df",
        "degradation_gene_pairs",
        "lineage_genes_by_cancer_type",
        "cancer_family_panels",
    ]:
        assert hasattr(pirlygenes, name), f"{name} not re-exported"
