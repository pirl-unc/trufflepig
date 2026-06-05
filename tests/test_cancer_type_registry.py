"""Tests for the expanded cancer-type registry.

The registry is a richer superset of TCGA — covers non-TCGA heme,
pediatric, NET, and rare entities, plus expression-based subtype rows
under TCGA umbrellas (BRCA × PAM50, LAML × ELN/APL, SARC × subtype,
LUAD × mutation class, SCLC × ASCL1/NEUROD1/POU2F3/YAP1, etc.).
"""

from pirlygenes.gene_sets_cancer import (
    cancer_type_registry,
    cancer_types_in_family,
    cancer_types_by_tissue,
    cancer_type_subtypes_of,
)

from trufflepig.reference import subtype_deconvolved_expression


def test_registry_has_required_columns():
    df = cancer_type_registry()
    required = {
        "code",
        "name",
        "family",
        "primary_tissue",
        "primary_template",
        "parent_code",
        "expression_source",
        "notes",
    }
    missing = required - set(df.columns)
    assert not missing, f"registry missing columns: {missing}"


def test_registry_codes_are_unique():
    df = cancer_type_registry()
    dupes = df["code"][df["code"].duplicated()].tolist()
    assert not dupes, f"duplicate codes in registry: {dupes}"


def test_registry_covers_all_33_tcga_codes():
    """Every TCGA code must appear in the registry or we'll lose
    compatibility with existing cancer-type detection code paths."""
    df = cancer_type_registry()
    tcga_codes = {
        "ACC",
        "BLCA",
        "BRCA",
        "CESC",
        "CHOL",
        "COAD",
        "DLBC",
        "ESCA",
        "GBM",
        "HNSC",
        "KICH",
        "KIRC",
        "KIRP",
        "LAML",
        "LGG",
        "LIHC",
        "LUAD",
        "LUSC",
        "MESO",
        "OV",
        "PAAD",
        "PCPG",
        "PRAD",
        "READ",
        "SARC",
        "SKCM",
        "STAD",
        "TGCT",
        "THCA",
        "THYM",
        "UCEC",
        "UCS",
        "UVM",
    }
    registry_codes = set(df["code"])
    missing = tcga_codes - registry_codes
    assert not missing, f"registry missing TCGA codes: {missing}"


def test_registry_includes_non_tcga_heme():
    df = cancer_type_registry()
    codes = set(df["code"])
    for need in ("CLL", "MM", "MCL", "FL", "HL", "BL", "CML", "MDS", "MPN", "HCL"):
        assert need in codes, f"missing heme code: {need}"


def test_registry_includes_pediatric():
    df = cancer_type_registry()
    codes = set(df["code"])
    for need in (
        "SARC_OS",
        "SARC_EWS",
        "SARC_RMS_ERMS",
        "SARC_RMS_ARMS",
        "NBL",
        "WILMS",
        "RT",
        "MBL",
        "ATRT",
        "RB",
        "HEPB",
    ):
        assert need in codes, f"missing pediatric code: {need}"


def test_registry_includes_net_axis():
    df = cancer_type_registry()
    codes = set(df["code"])
    for need in ("PANNET", "MID_NET", "LUNG_NET_LC", "SCLC", "MEC"):
        assert need in codes, f"missing NET code: {need}"


def test_registry_includes_rare_entities():
    df = cancer_type_registry()
    codes = set(df["code"])
    for need in ("NUTM", "ADCC", "MTC", "SARC_CHOR", "NPC"):
        assert need in codes, f"missing rare code: {need}"


def test_brca_pam50_subtypes_present():
    """BRCA's expression-based PAM50 tiles must be in the registry so
    the second-pass subtype classifier can route to them."""
    subs = cancer_type_subtypes_of("BRCA")
    assert set(subs) == {
        "BRCA_LumA",
        "BRCA_LumB",
        "BRCA_HER2",
        "BRCA_Basal",
        "BRCA_Normal",
    }


def test_sarc_subtypes_cover_main_entities():
    """SARC subtypes must at minimum include the Tranche B tiles
    plus the known tumor-biology subtypes (MPNST, angiosarcoma,
    UPS)."""
    subs = set(cancer_type_subtypes_of("SARC"))
    required = {
        "SARC_LMS",
        "SARC_DDLPS",
        "SARC_MYXLPS",
        "SARC_SYN",
        "SARC_DSRCT",
        "SARC_GIST",
        "SARC_MPNST",
        "SARC_ANGIO",
        "SARC_UPS",
    }
    missing = required - subs
    assert not missing, f"SARC subtypes missing: {missing}"


def test_laml_has_apl_and_eln_tiles():
    subs = set(cancer_type_subtypes_of("LAML"))
    assert "LAML_APL" in subs
    # ELN2017 is the modern risk-stratification that gates transplant
    # vs chemo; must be representable as a subtype tile.
    for eln in ("LAML_ELN_Fav", "LAML_ELN_Int", "LAML_ELN_Adv"):
        assert eln in subs


def test_bone_tissue_returns_osteosarcoma_and_ewing():
    """Site-aware hypothesis: any sample suspected of bone origin
    should be able to enumerate OS + Ewing as candidates."""
    bone_cancers = set(cancer_types_by_tissue("bone"))
    assert "SARC_OS" in bone_cancers
    assert "SARC_EWS" in bone_cancers


def test_heme_myeloid_family_contains_laml_and_related():
    """Family grouping must pull LAML + its tiles + MDS + MPN + CML
    together — they share the heme_marrow / heme_blood templates and
    need joint candidate enumeration when sample mode is heme."""
    myeloid = set(cancer_types_in_family("heme-myeloid"))
    for need in ("LAML", "MDS", "MPN", "CML"):
        assert need in myeloid


def test_net_family_contains_sclc_and_pannet():
    # pirlygenes 5.12 lineage-only ontology renamed the `net` family to
    # `neuroendocrine` (WHO Endocrine & Neuroendocrine 2022).
    net = set(cancer_types_in_family("neuroendocrine"))
    assert "SCLC" in net
    assert "PANNET" in net
    assert "MEC" in net  # Merkel cell carcinoma


def test_parent_codes_reference_registry_entries():
    """Every non-null parent_code must reference an existing code."""
    df = cancer_type_registry()
    codes = set(df["code"])
    parents = df["parent_code"].dropna().astype(str)
    orphan = [p for p in parents if p and p not in codes]
    assert not orphan, f"parent_codes not in registry: {set(orphan)}"


def test_registry_has_source_cohort_column():
    """Every curated row carries the cohort that produced its expression
    median — enables downstream tracking of which cohort + paper each
    reference value came from."""
    df = cancer_type_registry()
    assert "source_cohort" in df.columns
    assert "source_pmid" in df.columns


def test_source_cohort_values_are_canonical():
    """source_cohort should only take values from the canonical cohort
    vocabulary — rejects typos like 'TCGA_BRCA' vs 'TREEHOUSE_POLYA_25_01'.

    The canonical vocabulary is pirlygenes' own expression-reference manifest
    (``available_cancer_expression_references``) so the allowlist tracks the
    dependency instead of drifting as new cohorts are added. A handful of
    non-expression values (curated literature, the pan-cancer Xena matrix,
    blank) are not in that manifest and are allowed explicitly.
    """
    import pirlygenes as _pirlygenes

    df = cancer_type_registry()
    canonical = set(
        _pirlygenes.available_cancer_expression_references()["source_cohort"]
        .fillna("")
        .astype(str)
    )
    valid = canonical | {"", "LITERATURE_CURATED", "TCGA_XENA_TOIL"}
    present = set(df["source_cohort"].fillna("").astype(str).unique())
    # COMPUTED_* are pirlygenes' computed aggregate cohorts (e.g. the SARC
    # grand union COMPUTED_PAN_SARCOMA, Phase C.2) — a legitimate cohort
    # category that isn't a single deposited dataset in the manifest.
    unknown = {c for c in (present - valid) if not c.startswith("COMPUTED_")}
    assert not unknown, f"unknown source_cohort values: {unknown}"


def test_expanded_sarcomas_present():
    """The 19 sarcoma additions (WHO therapy-distinct entities) must
    be in the registry so the second-pass subtype classifier can
    route to them."""
    df = cancer_type_registry()
    codes = set(df["code"])
    required = {
        "SARC_EPITH",
        "SARC_DFSP",
        "SARC_ASPS",
        "SARC_CCS",
        "SARC_IFS",
        "SARC_EHE",
        "SARC_PEC",
        "SARC_KS",
        "SARC_MYXFIB",
        "SARC_SFT",
        "SARC_IMT",
        "SARC_GCTB",
        "SARC_ESS_LG",
        "SARC_ESS_HG",
        "SARC_LGFMS",
        "SARC_EMC",
        "SARC_PLEOLPS",
        "SARC_RMS_PRMS",
        "SARC_RMS_SSRMS",
    }
    missing = required - codes
    assert not missing, f"expanded-sarcoma codes missing: {missing}"


def test_subtype_key_maps_sarc_subtypes_to_key_genes_entries():
    """The subtype_key column must match the actual subtype values
    used in cancer-key-genes.csv, otherwise the cancers CLI
    subcommand will report bm=0 / tg=0 for curated subtypes."""
    from pirlygenes.gene_sets_cancer import (
        cancer_biomarker_genes,
        cancer_therapy_targets,
    )

    df = cancer_type_registry()
    mapped = df[df["subtype_key"].fillna("").astype(str).ne("")]
    assert len(mapped) >= 7, "expected at least 7 rows with subtype_key populated"
    for _, row in mapped.iterrows():
        parent = row["parent_code"]
        subtype = row["subtype_key"]
        bm = cancer_biomarker_genes(parent, subtype=subtype)
        tg = cancer_therapy_targets(parent, subtype=subtype)
        assert len(bm) > 0 or len(tg) > 0, (
            f"subtype_key {parent}/{subtype} (code {row['code']}) has "
            f"no key-genes rows — either the subtype_key is wrong or "
            f"cancer-key-genes.csv is missing the tile"
        )


def test_cancers_cli_uses_explicit_coverage_columns(capsys):
    from trufflepig.main import print_cancer_registry

    # RMS_ARMS moved from the retired `pediatric-soft` family to `sarcoma` (5.12).
    print_cancer_registry(family="sarcoma")
    out = capsys.readouterr().out

    assert "Clinical group: Sarcoma, bone, and soft-tissue tumors" in out
    assert "Expression ref" in out
    assert "Expression source" in out
    assert "Curation source" in out
    assert "Biomarkers" in out
    assert "Targets" in out
    assert "Lineage" in out
    assert "Normal" in out
    assert "Response" in out
    assert "SARC_RMS_ARMS" in out
    assert "Treehouse v25.01 PolyA" in out
    assert "| Code |" not in out
    assert "bm=" not in out
    assert "tg=" not in out
    assert "B5 T1" not in out
    assert "sub-child" not in out


def test_cancers_cli_counts_work_outside_repo_root(capsys, monkeypatch, tmp_path):
    from trufflepig.main import print_cancer_registry

    monkeypatch.chdir(tmp_path)
    print_cancer_registry(family="carcinoma-gu")
    out = capsys.readouterr().out

    assert "BLCA" in out
    assert "B9 T7 L6 N9 R0" not in out
    assert "Coverage audit:" in out
    assert "Lineage" in out
    assert "Matched normal" in out
    assert "Normal" in out


def test_cancers_cli_source_qualifies_expression_refs(capsys):
    from trufflepig.main import print_cancer_registry

    print_cancer_registry(family="sarcoma")
    out = capsys.readouterr().out

    assert "TCGA:SARC" in out
    assert "Treehouse:SARC_SYN" in out
    assert "GEO:SARC_CHON" in out
    assert "GEO:SARC_DDLPS" in out
    assert "GEO GSE299759" in out
    assert "GEO GSE75885" in out
    assert "sources:" not in out

    print_cancer_registry(family="carcinoma-breast")
    out = capsys.readouterr().out

    assert "TCGA/PAM50:BRCA_HER2" in out
    assert "no expr 0" in out

    # RB moved from the retired `pediatric-eye` family to `embryonal` (5.12).
    print_cancer_registry(family="embryonal")
    out = capsys.readouterr().out

    assert "Treehouse/RiboD:RB" in out
    assert "Treehouse v25.01 RiboD" in out


def test_gse75885_sarcoma_expression_refs_are_bundled():
    d = subtype_deconvolved_expression()
    assert d is not None

    expected = {
        "SARC_DDLPS": 19,
        "SARC_PLEOLPS": 4,
        "SARC_LGFMS": 2,
    }
    for code, n_samples in expected.items():
        rows = d[
            (d["cancer_code"] == code)
            & (d["source_cohort"] == "GSE75885_DELESPAUL_2017")
        ]
        assert not rows.empty
        assert int(rows["n_samples"].max()) == n_samples

    ddlps = d[d["cancer_code"] == "SARC_DDLPS"].set_index("symbol")
    assert ddlps.loc["MDM2", "tumor_tpm_median"] > 1000
    assert ddlps.loc["CDK4", "tumor_tpm_median"] > 500

    lgfms = d[d["cancer_code"] == "SARC_LGFMS"].set_index("symbol")
    assert lgfms.loc["MUC4", "tumor_tpm_median"] > 1000
    assert lgfms.loc["CREB3L2", "tumor_tpm_median"] > 100


def test_gse299759_chondrosarcoma_expression_ref_is_bundled():
    d = subtype_deconvolved_expression()
    assert d is not None

    rows = d[
        (d["cancer_code"] == "SARC_CHON")
        & (d["source_cohort"] == "GSE299759_MEIJER_2026")
    ]
    assert not rows.empty
    assert int(rows["n_samples"].max()) == 54

    chon = rows.set_index("symbol")
    assert chon.loc["COL2A1", "tumor_tpm_median"] > 1000
    assert chon.loc["ACAN", "tumor_tpm_median"] > 1000
    assert chon.loc["SOX9", "tumor_tpm_median"] > 100


def test_treehouse_ribod_sparse_expression_refs_are_bundled():
    d = subtype_deconvolved_expression()
    assert d is not None

    expected = {
        "RB": 15,
        "SARC_CHOR": 3,
    }
    for code, n_samples in expected.items():
        rows = d[
            (d["cancer_code"] == code)
            & (d["source_cohort"] == "TREEHOUSE_RIBOD_25_01")
        ]
        assert not rows.empty
        assert int(rows["n_samples"].max()) == n_samples

    rb = d[d["cancer_code"] == "RB"].set_index("symbol")
    assert rb.loc["CRX", "tumor_tpm_median"] > 25
    assert rb.loc["ARR3", "tumor_tpm_median"] > 5

    chor = d[d["cancer_code"] == "SARC_CHOR"].set_index("symbol")
    assert chor.loc["COL2A1", "tumor_tpm_median"] > 25
    assert chor.loc["ACAN", "tumor_tpm_median"] > 10


def test_nutm_has_actionable_curation():
    """NUT carcinoma gets the fusion-partner biomarkers (NUTM1,
    BRD4, BRD3, NSD3) plus BET-inhibitor therapy rows — these were
    added because pirlygenes is applied to NUTM1-rearranged carcinoma samples."""
    from pirlygenes.gene_sets_cancer import (
        cancer_biomarker_genes,
        cancer_therapy_targets,
    )

    bm = cancer_biomarker_genes("NUTM")
    for gene in ("NUTM1", "BRD4", "BRD3", "MYC", "TP63"):
        assert gene in bm, f"NUTM biomarker missing: {gene}"
    tg = cancer_therapy_targets("NUTM")
    agents = set(tg["agent"].astype(str).str.lower())
    # At least one BET inhibitor must be present.
    assert any(
        "bet" in row.lower()
        or "bromodomain" in row.lower()
        or "molibresib" in row.lower()
        or "birabresib" in row.lower()
        or "bms-986158" in row.lower()
        for row in list(agents) + list(tg["rationale"].astype(str).str.lower())
    )


def test_primary_templates_are_declared_or_planned():
    """Every row has a primary_template — either an implemented
    template name or a ``primary_<tissue>`` name documented as planned
    (osteosarcoma, chondrosarcoma, adipose, etc.). Catches typos like
    ``primary_bones`` or blanks."""
    df = cancer_type_registry()
    templates = df["primary_template"].dropna().unique()
    # Every template must match the convention.
    for t in templates:
        assert (
            t == "solid_primary" or t.startswith("primary_") or t.startswith("heme_")
        ), f"unknown primary_template convention: {t}"
