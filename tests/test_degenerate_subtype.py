"""Tests for the degenerate-subtype registry + fusion-surrogate catalog (#198).

Pins the OS/DDLPS bone-vs-soft-tissue tiebreaker, the Ewing/DSRCT/ARMS
fusion-surrogate voting, the NE-axis site disambiguator, and the
data-integrity contract on both CSV files.
"""

from pirlygenes.gene_sets_cancer import cancer_type_registry

from trufflepig.brief import build_summary
from trufflepig.degenerate_subtype import (
    degenerate_subtype_pairs,
    fusion_surrogate_expression,
    fusion_surrogate_genes_for,
    resolve_degenerate_subtype,
)
from trufflepig.report_view import build_report_view


def render_summary(analysis, *args, **kwargs):
    finalized = dict(analysis)
    finalized.setdefault("cancer_type", kwargs.get("cancer_code") or "UNKNOWN")
    finalized.setdefault("sample_mode", "solid")
    kwargs["report_view"] = build_report_view(finalized)
    return build_summary(analysis, *args, **kwargs)


def test_degenerate_pairs_csv_loads_and_parses():
    df = degenerate_subtype_pairs()
    assert len(df) >= 5
    for _, row in df.iterrows():
        assert isinstance(row["members"], list)
        assert len(row["members"]) >= 2
        assert isinstance(row["tiebreaker_mapping"], dict)
        assert isinstance(row["activation_signature"], dict)
        # Activation signature values must parse as floats.
        for gene, min_tpm in row["activation_signature"].items():
            assert isinstance(gene, str) and gene
            assert isinstance(min_tpm, float)
        assert row["tiebreaker_rule"] in {
            "site_template",
            "fusion_surrogate",
            "marker_combo",
        }


def test_nutm_pair_uses_nutm1_only_not_mage_or_prame():
    """MAGEA / PRAME / HORMAD1 / CTAG1B are all broadly expressed in
    LUSC and HNSC (PRAME median 24.5 TPM, MAGEA3 42 TPM in LUSC) —
    using them as NUTM surrogates would systematically mis-call
    squamous tumors as NUT carcinoma. The NUTM_vs_squamous pair must
    rely only on NUTM1 mRNA (near-silent across TCGA cohorts) for
    disambiguation. Pinned to prevent regression."""
    df = degenerate_subtype_pairs()
    nutm_row = df[df["pair_id"] == "NUTM_vs_squamous"].iloc[0]
    mapping_genes = set(nutm_row["tiebreaker_mapping"].keys())
    assert mapping_genes == {"NUTM1"}, (
        f"NUTM_vs_squamous tiebreaker mapping must contain only NUTM1, "
        f"got {sorted(mapping_genes)}. MAGE-A / PRAME / HORMAD1 are "
        f"NOT NUTM-specific (see pair notes)."
    )
    # Activation signature likewise requires NUTM1 expression.
    assert "NUTM1" in nutm_row["activation_signature"], (
        "NUTM_vs_squamous activation must require NUTM1 expression"
    )


def test_degenerate_pair_members_are_real_registry_codes():
    """Every member listed must exist in the registry."""
    reg_codes = set(cancer_type_registry()["code"])
    df = degenerate_subtype_pairs()
    # Some advanced pair entries reference subtype codes we have not
    # yet added to the registry (SARC_CIC_DUX4, SARC_BCOR, EPN_RELA,
    # MBL_SHH/WNT/MYC, LAML_APL). Allow those as a transitional
    # allowlist; shrink when the registry grows to include them.
    allow_not_yet_in_registry = {
        "SARC_CIC_DUX4",
        "SARC_BCOR",
        "EPN_RELA",
        "MBL_SHH",
        "MBL_WNT",
        "MBL_MYC",
    }
    unknown = set()
    for _, row in df.iterrows():
        for member in row["members"]:
            if member not in reg_codes and member not in allow_not_yet_in_registry:
                unknown.add(member)
    assert not unknown, (
        f"Pair members not in registry and not allowlisted: {sorted(unknown)}"
    )


def test_fusion_surrogate_csv_loads():
    df = fusion_surrogate_expression()
    assert len(df) >= 30
    required_cols = {
        "fusion_class",
        "surrogate_gene",
        "surrogate_role",
        "cancer_code",
        "rationale",
    }
    missing = required_cols - set(df.columns)
    assert not missing, f"fusion-surrogate schema missing: {missing}"


def test_fusion_surrogate_roles_are_valid():
    df = fusion_surrogate_expression()
    valid = {
        "activated",
        "activated_ectopic",
        "myogenic_lineage",
        "not_reliable_total_gene_surrogate",
    }
    bad = set(df["surrogate_role"]) - valid
    assert not bad, f"unknown surrogate_role values: {bad}"


# ── Resolver contract ──────────────────────────────────────────────


def test_resolver_returns_no_pair_for_unlisted_subtype():
    result = resolve_degenerate_subtype(
        winning_subtype="BRCA_LumA",
        site_template="solid_primary",
    )
    assert result["status"] == "no_pair"
    assert result["final_subtype"] == "BRCA_LumA"


def test_resolver_returns_no_pair_for_none_subtype():
    result = resolve_degenerate_subtype(winning_subtype=None)
    assert result["status"] == "no_pair"


def test_resolver_os_vs_ddlps_bone_site_swaps_to_os():
    """Sid Sijbrandij public-data shape: classifier picked DDLPS, site is
    bone, MDM2 strongly amplified — the pair activates and site-template
    tiebreaker swaps DDLPS → OS."""
    result = resolve_degenerate_subtype(
        winning_subtype="SARC_DDLPS",
        site_template="met_bone",
        tumor_tpm_by_symbol={"MDM2": 877.0, "CDK4": 73.0},
    )
    assert result["status"] == "corrected", result
    assert result["final_subtype"] == "SARC_OS"
    assert "site_template" in result["reason"]


def test_resolver_os_vs_ddlps_low_mdm2_is_pair_inactive():
    """A DDLPS call on a bone-site sample WITHOUT MDM2 amplification
    shouldn't be second-guessed by the OS/DDLPS pair — the 12q amp
    ambiguity simply isn't present. Classifier's DDLPS pick stands."""
    result = resolve_degenerate_subtype(
        winning_subtype="SARC_DDLPS",
        site_template="met_bone",
        tumor_tpm_by_symbol={"MDM2": 3.0, "CDK4": 5.0},
    )
    assert result["status"] == "pair_inactive", result
    assert result["final_subtype"] == "SARC_DDLPS"


def test_resolver_os_vs_ddlps_retroperitoneum_confirms_ddlps():
    """When classifier picks DDLPS, site is retroperitoneum, and MDM2
    is amplified, the pair activates and the tiebreaker confirms DDLPS."""
    result = resolve_degenerate_subtype(
        winning_subtype="SARC_DDLPS",
        site_template="primary_retroperitoneum",
        tumor_tpm_by_symbol={"MDM2": 500.0, "CDK4": 80.0},
    )
    assert result["status"] == "confirmed"
    assert result["final_subtype"] == "SARC_DDLPS"


def test_resolver_os_vs_ddlps_unknown_site_leaves_degenerate():
    """When site doesn't appear in the mapping but the pair activates,
    resolver returns degenerate so the markdown layer can emit the
    ambiguity."""
    result = resolve_degenerate_subtype(
        winning_subtype="SARC_DDLPS",
        site_template="solid_primary",  # not in OS/DDLPS mapping
        tumor_tpm_by_symbol={"MDM2": 500.0, "CDK4": 80.0},
    )
    assert result["status"] == "degenerate"
    assert result["final_subtype"] == "SARC_DDLPS"  # left unchanged
    assert "degenerate between" in result["reason"]


def test_resolver_ewing_fate1_nr0b1_confirms_ewing():
    """FATE1 + NR0B1 high with CD99 elevated → EWS-FLI1 signature
    active → confirms Ewing."""
    result = resolve_degenerate_subtype(
        winning_subtype="SARC_EWS",
        tumor_tpm_by_symbol={
            "CD99": 500.0,  # activation gate
            "FATE1": 80.0,
            "NR0B1": 50.0,
            "WT1": 0.0,
        },
    )
    assert result["status"] == "confirmed"
    assert result["final_subtype"] == "SARC_EWS"


def test_resolver_ewing_pair_inactive_without_cd99():
    """If CD99 is silent, the small-blue-round-cell ambiguity isn't
    present and the pair shouldn't fire regardless of other genes."""
    result = resolve_degenerate_subtype(
        winning_subtype="SARC_EWS",
        tumor_tpm_by_symbol={"CD99": 0.5, "FATE1": 80.0, "NR0B1": 50.0},
    )
    assert result["status"] == "pair_inactive"
    assert result["final_subtype"] == "SARC_EWS"


def test_resolver_dsrct_wt1_corrects_from_ewing():
    """If classifier picked EWS but WT1 is high and FATE1/NR0B1 are
    silent (and CD99 is high to activate the pair), DSRCT is the
    real call."""
    result = resolve_degenerate_subtype(
        winning_subtype="SARC_EWS",
        tumor_tpm_by_symbol={
            "CD99": 500.0,
            "FATE1": 0.0,
            "NR0B1": 0.0,
            "WT1": 200.0,
        },
    )
    assert result["status"] == "corrected"
    assert result["final_subtype"] == "SARC_DSRCT"


def test_resolver_pannet_vs_midnet_pancreas_site():
    """NE markers present (activates pair) + pancreatic site → NET_PANCREAS."""
    result = resolve_degenerate_subtype(
        winning_subtype="NET_MIDGUT",
        site_template="primary_pancreas",
        tumor_tpm_by_symbol={"CHGA": 1200.0, "SYP": 80.0, "ENO2": 150.0},
    )
    assert result["status"] == "corrected"
    assert result["final_subtype"] == "NET_PANCREAS"


def test_resolver_pannet_pair_inactive_without_ne_markers():
    """A NET_MIDGUT pick on a sample with silent NE markers (CHGA/SYP/ENO2
    all low) means the NE-axis ambiguity isn't present; skip the pair."""
    result = resolve_degenerate_subtype(
        winning_subtype="NET_MIDGUT",
        site_template="primary_pancreas",
        tumor_tpm_by_symbol={"CHGA": 1.0, "SYP": 2.0, "ENO2": 5.0},
    )
    assert result["status"] == "pair_inactive"


def test_resolver_squamous_axis_lung_corrects_hnsc_to_lusc():
    """Squamous markers present + primary_lung site → LUSC corrected."""
    result = resolve_degenerate_subtype(
        winning_subtype="HNSC",
        site_template="primary_lung",
        tumor_tpm_by_symbol={"KRT5": 300.0, "KRT6A": 200.0, "TP63": 50.0},
    )
    assert result["status"] == "corrected"
    assert result["final_subtype"] == "LUSC"


def test_resolver_mcl_ccnd1_corrects_from_cll():
    """CCND1 high drives MCL even if classifier picked CLL — but the
    pair only activates when B-cell markers are present."""
    result = resolve_degenerate_subtype(
        winning_subtype="CLL",
        tumor_tpm_by_symbol={
            "CD19": 40.0,
            "MS4A1": 80.0,
            "CD79A": 30.0,  # activation
            "CCND1": 200.0,
            "SOX11": 80.0,
            "BCL6": 0.0,
        },
    )
    assert result["status"] == "corrected"
    assert result["final_subtype"] == "MCL"


def test_resolver_no_fusion_panel_tpm_leaves_pair_inactive():
    """When no TPM dict is provided for a fusion-surrogate pair whose
    activation signature requires gene expression, the resolver can't
    confirm activation → pair_inactive, classifier's pick stands."""
    result = resolve_degenerate_subtype(
        winning_subtype="SARC_EWS",
        tumor_tpm_by_symbol=None,
    )
    assert result["status"] == "pair_inactive"
    assert result["final_subtype"] == "SARC_EWS"


# ── NUT carcinoma vs squamous — critical correctness case ──────────


def test_resolver_nutm_inactive_on_squamous_sample_with_silent_nutm1():
    """A LUSC sample with high PRAME + MAGEA3 (common in squamous!)
    but silent NUTM1 should NOT get pulled toward NUT carcinoma.
    Before the activation-gate fix, vote-based disambiguation would
    count PRAME+MAGE-A as NUTM evidence; now the NUTM_vs_squamous
    pair only activates when NUTM1 is expressed."""
    # With LUSC as winning_subtype, the LUSC_vs_HNSC_vs_CESC pair is
    # preferred because it has site context available and it's
    # applicable.
    result = resolve_degenerate_subtype(
        winning_subtype="LUSC",
        site_template="primary_lung",
        tumor_tpm_by_symbol={
            # Squamous activation signal (this pair fires)
            "KRT5": 400.0,
            "KRT6A": 300.0,
            "TP63": 80.0,
            # High PRAME + MAGE-A as is typical of LUSC
            "PRAME": 30.0,
            "MAGEA3": 60.0,
            "MAGEA1": 5.0,
            # But NUTM1 silent — NOT a NUT carcinoma
            "NUTM1": 0.1,
        },
    )
    assert result["status"] == "confirmed"
    assert result["final_subtype"] == "LUSC"


def test_resolver_nutm_corrects_squamous_when_nutm1_expressed():
    """A sample that classified as LUSC but actually expresses NUTM1
    at diagnostic levels should get flagged. With LUSC's primary pair
    being LUSC_vs_HNSC_vs_CESC (site-template) — it confirms LUSC on
    site — NUTM is not reached. The NUTM flag comes via the NUTM
    winning_subtype path instead: a sample classified as NUTM but
    without NUTM1 expression should be pair_inactive (stays NUTM) —
    the NUTM->squamous swap would require a reverse-mapping which is
    out of scope for this pair.

    This test pins that a NUTM winning_subtype WITH NUTM1 present is
    confirmed by the fusion_surrogate rule."""
    result = resolve_degenerate_subtype(
        winning_subtype="NUTM",
        tumor_tpm_by_symbol={"NUTM1": 15.0, "PRAME": 200.0, "MAGEA3": 20.0},
    )
    assert result["status"] == "confirmed"
    assert result["final_subtype"] == "NUTM"


def test_resolver_nutm_pair_inactive_when_nutm1_silent():
    """A NUTM winning_subtype on a sample where NUTM1 is silent is
    pair_inactive — the pair doesn't activate, so the classifier's
    (likely wrong) NUTM call stays, but the confidence / markdown
    layer can inspect ``status='pair_inactive'`` to tell the reader
    the diagnostic NUTM1 signal is absent."""
    result = resolve_degenerate_subtype(
        winning_subtype="NUTM",
        tumor_tpm_by_symbol={"NUTM1": 0.05, "PRAME": 20.0, "MAGEA3": 40.0},
    )
    assert result["status"] == "pair_inactive"
    assert result["final_subtype"] == "NUTM"


def test_resolver_pair_selection_prefers_applicable_pair():
    """HNSC is a member of both LUSC_vs_HNSC_vs_CESC (site_template)
    and NUTM_vs_squamous (fusion_surrogate). When site_template
    context is available, the site pair is preferred; when only TPM
    context is available, the fusion pair takes over."""
    # Site context → LUSC_vs_HNSC_vs_CESC applies
    result_site = resolve_degenerate_subtype(
        winning_subtype="HNSC",
        site_template="primary_head_neck",
        tumor_tpm_by_symbol={"KRT5": 400.0, "TP63": 80.0, "NUTM1": 0.0},
    )
    assert result_site["pair_id"] == "LUSC_vs_HNSC_vs_CESC"
    assert result_site["status"] == "confirmed"


# ── Fusion-surrogate genes by cancer ────────────────────────────────


def test_fusion_surrogate_genes_for_ewing():
    hits = fusion_surrogate_genes_for("SARC_EWS")
    genes = {h["gene"] for h in hits}
    # PHOX2B was dropped from the Ewing surrogate set in pirlygenes 5.22.107 (it is a
    # neuroblastoma marker, not Ewing-specific); FATE1/NR0B1 remain the Ewing
    # fusion-target surrogates.
    for required in ("FATE1", "NR0B1"):
        assert required in genes, f"EWS surrogate missing: {required}"


def test_fusion_surrogate_genes_for_nutm():
    hits = fusion_surrogate_genes_for("NUTM")
    genes = {h["gene"] for h in hits}
    # NUTM1 itself + PRAME are the critical surrogates — without them
    # the NUT-carcinoma diagnosis from bulk RNA is not reachable.
    for required in ("NUTM1", "PRAME", "MAGEA1"):
        assert required in genes, f"NUTM surrogate missing: {required}"


def test_fusion_surrogate_pan_cancer_applies_to_any_code():
    """NTRK/ROS1 fusion surrogates are marked pan_cancer and should
    return for any cancer code that receives them."""
    hits = fusion_surrogate_genes_for("LUAD")
    genes = {h["gene"] for h in hits}
    for required in ("NTRK1", "NTRK3", "ROS1", "ALK"):
        assert required in genes, f"LUAD fusion surrogate missing: {required}"


def test_brief_renders_corrected_subtype():
    """End-to-end pin: when the analysis dict carries a liposarcoma
    winning_subtype, the decomposition top template is met_bone, AND
    MDM2 is amplified (activating the 12q-amp pair), the summary.md
    should render osteosarcoma-consistent + a subtype note explaining
    the swap. Reproduces the Sid public osteosarcoma-data failure mode.

    This version exercises the tumor_tpm_by_symbol-from-ranges_df
    path — the production analyze call builds the TPM dict from
    ranges_df, not from an analysis key."""
    import pandas as pd

    from trufflepig.reporting import cancer_key_genes_lookup_for_analysis

    analysis = {
        "purity": {
            "overall_estimate": 0.74,
            "overall_lower": 0.42,
            "overall_upper": 1.0,
        },
        "purity_confidence": type("PT", (), {"tier": "low"})(),
        "sample_context": None,
        "cancer_name": "Sarcoma",
        "candidate_trace": [
            {"code": "SARC", "winning_subtype": "SARC_LPS_UNSPEC"},
        ],
        "decomposition": {
            "best_template": "met_bone",
            "best_cancer_type": "SARC",
            "hypotheses": [
                {"template": "met_bone", "cancer_type": "SARC", "score": 0.22},
            ],
        },
    }
    ranges_df = pd.DataFrame(
        [
            {"symbol": "MDM2", "attr_tumor_tpm": 877.0},
            {"symbol": "CDK4", "attr_tumor_tpm": 73.0},
            {"symbol": "FRS2", "attr_tumor_tpm": 137.0},
        ]
    )
    summary = render_summary(
        analysis,
        ranges_df=ranges_df,
        cancer_code="SARC",
        disease_state="",
        sample_id="synthetic-bone-mdm2",
    )
    assert "osteosarcoma-consistent" in summary, summary
    assert "Bone-associated context favors osteosarcoma over liposarcoma" in summary, (
        summary
    )
    assert cancer_key_genes_lookup_for_analysis("SARC", analysis, ranges_df) == (
        "SARC_OS",
        None,
    )
    assert "MDM2 / CDK4 / FRS2 amplification" in summary, summary
    assert "site_template tiebreaker swapped" not in summary


def test_brief_uses_site_hint_for_corrected_subtype_without_decomposition():
    """Explicit site constraints can resolve subtype even when decomposition
    did not produce a best template."""
    import pandas as pd


    analysis = {
        "cancer_type": "SARC",
        "cancer_name": "Sarcoma",
        "analysis_constraints": {
            "cancer_type": "SARC",
            "tumor_context": "met",
            "site_hint": "bone",
        },
        "cancer_type_source": "user-specified",
        "purity": {
            "overall_estimate": 0.74,
            "overall_lower": 0.42,
            "overall_upper": 1.0,
        },
        "purity_confidence": type("PT", (), {"tier": "low"})(),
        "sample_context": None,
        "candidate_trace": [
            {
                "code": "SARC",
                "winning_subtype": "SARC_LPS_UNSPEC",
                "support_geomean": 0.58,
            },
        ],
    }
    ranges_df = pd.DataFrame(
        [
            {"symbol": "MDM2", "attr_tumor_tpm": 877.0},
            {"symbol": "CDK4", "attr_tumor_tpm": 73.0},
            {"symbol": "FRS2", "attr_tumor_tpm": 137.0},
        ]
    )

    summary = render_summary(
        analysis,
        ranges_df=ranges_df,
        cancer_code="SARC",
        disease_state="",
        sample_id="synthetic-bone-mdm2",
    )

    assert "osteosarcoma-consistent" in summary, summary
    assert "Bone-associated context favors osteosarcoma over liposarcoma" in summary, (
        summary
    )


def test_key_genes_lookup_switches_to_direct_os_panel():
    """Degenerate resolution can land on a standalone cancer code.

    When that happens, report curation must use the resolved code's
    panel directly instead of the umbrella parent union.
    """
    import pandas as pd

    from trufflepig.reporting import cancer_key_genes_lookup_for_analysis

    analysis = {
        "candidate_trace": [
            {"code": "SARC", "winning_subtype": "SARC_LPS_UNSPEC"},
        ],
        "decomposition": {
            "best_template": "met_bone",
            "best_cancer_type": "SARC",
        },
    }
    ranges_df = pd.DataFrame(
        [
            {"symbol": "MDM2", "attr_tumor_tpm": 877.0},
            {"symbol": "CDK4", "attr_tumor_tpm": 73.0},
            {"symbol": "FRS2", "attr_tumor_tpm": 137.0},
        ]
    )
    assert cancer_key_genes_lookup_for_analysis(
        "SARC",
        analysis,
        ranges_df=ranges_df,
    ) == ("SARC_OS", None)


def test_key_genes_lookup_matches_uppercase_parent_subtype_rows():
    """Key-genes subtype matching should be case-tolerant.

    Registry codes like ``SARC_MPNST`` should recover the curated
    uppercase subtype value ``MPNST`` rather than falling back to the
    umbrella SARC union.
    """
    from trufflepig.reporting import cancer_key_genes_lookup_for_analysis

    analysis = {
        "candidate_trace": [
            {"code": "SARC", "winning_subtype": "SARC_MPNST"},
        ],
        "decomposition": {
            "best_template": "primary_nerve_sheath",
            "best_cancer_type": "SARC",
        },
    }
    assert cancer_key_genes_lookup_for_analysis(
        "SARC",
        analysis,
    ) == ("SARC", "MPNST")


def test_brief_uses_os_scope_after_corrected_subtype_without_stale_targets():
    """User-facing pin for the Sid public osteosarcoma-data failure mode.

    The summary should stop surfacing DDLPS-only therapies once the
    bone-site tiebreaker resolves the sample to osteosarcoma, while
    still filtering stale OS rows such as miscited ganitumab.
    """
    import pandas as pd


    analysis = {
        "purity": {
            "overall_estimate": 0.74,
            "overall_lower": 0.42,
            "overall_upper": 1.0,
        },
        "purity_confidence": type("PT", (), {"tier": "low"})(),
        "sample_context": None,
        "cancer_name": "Sarcoma",
        "candidate_trace": [
            {"code": "SARC", "winning_subtype": "SARC_LPS_UNSPEC"},
        ],
        "decomposition": {
            "best_template": "met_bone",
            "best_cancer_type": "SARC",
        },
    }
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "MDM2",
                "observed_tpm": 1180.4,
                "attr_tumor_tpm": 1176.0,
                "attr_tumor_fraction": 0.99,
                "attribution": "tumor",
            },
            {
                "symbol": "CDK4",
                "observed_tpm": 112.8,
                "attr_tumor_tpm": 90.0,
                "attr_tumor_fraction": 0.80,
                "attribution": "tumor",
            },
            {
                "symbol": "FRS2",
                "observed_tpm": 140.0,
                "attr_tumor_tpm": 137.0,
                "attr_tumor_fraction": 0.96,
                "attribution": "tumor",
            },
            {
                "symbol": "IGF1R",
                "observed_tpm": 125.0,
                "attr_tumor_tpm": 120.0,
                "attr_tumor_fraction": 0.96,
                "attribution": "tumor",
            },
        ]
    )
    summary = render_summary(
        analysis,
        ranges_df=ranges_df,
        cancer_code="SARC",
        disease_state="",
        sample_id="synthetic-bone-os-panel",
    )
    assert "Using osteosarcoma-specific therapy evidence" in summary, summary
    assert "ganitumab + chemo" not in summary, summary
    assert "brigimadlin" not in summary, summary


def test_forced_cancer_type_does_not_inherit_unrelated_subtype():
    """A constrained COAD report must not adopt a SARC subtype from the
    unconstrained classifier trace."""
    import pandas as pd

    from trufflepig.reporting import (
        cancer_key_genes_lookup_for_analysis,
        candidate_winning_subtype_for_analysis,
    )

    analysis = {
        "cancer_type": "COAD",
        "cancer_name": "Colon Adenocarcinoma",
        "analysis_constraints": {"cancer_type": "COAD"},
        "purity": {
            "overall_estimate": 0.30,
            "overall_lower": 0.20,
            "overall_upper": 0.45,
        },
        "purity_confidence": type("PT", (), {"tier": "moderate"})(),
        "sample_context": None,
        "candidate_trace": [
            {"code": "SARC", "winning_subtype": "SARC_LMS"},
            {"code": "COAD"},
        ],
        "decomposition": {
            "best_template": "solid_primary",
            "best_cancer_type": "COAD",
        },
    }
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "EGFR",
                "observed_tpm": 90.0,
                "attr_tumor_tpm": 70.0,
                "attr_tumor_fraction": 0.78,
                "attr_top_compartment": "",
                "attr_top_compartment_tpm": 0.0,
                "attribution": {},
                "tme_dominant": False,
                "tme_explainable": False,
            }
        ]
    )

    assert candidate_winning_subtype_for_analysis(analysis) is None
    assert cancer_key_genes_lookup_for_analysis("COAD", analysis) == ("COAD", None)

    summary = render_summary(
        analysis,
        ranges_df=ranges_df,
        cancer_code="COAD",
        disease_state="",
        sample_id="forced-coad",
    )
    assert "Cancer call:** COAD" in summary, summary
    assert "Subtype-resolved therapy curation" not in summary, summary
    assert "leiomyosarcoma" not in summary.lower(), summary


def test_report_subtype_can_refine_parent_but_cannot_jump_to_sibling():
    from trufflepig.reporting import candidate_winning_subtype_for_analysis

    parent = {
        "cancer_type": "BRCA",
        "candidate_trace": [
            {"code": "BRCA", "winning_subtype": "BRCA_Basal"},
        ],
    }
    sibling = {
        "cancer_type": "BRCA_Basal",
        "candidate_trace": [
            {"code": "BRCA_Basal", "winning_subtype": "BRCA_HER2"},
        ],
    }

    assert candidate_winning_subtype_for_analysis(parent) == "BRCA_Basal"
    assert candidate_winning_subtype_for_analysis(sibling) is None


def test_learned_compartment_parent_call_does_not_inherit_trace_subtype():
    from trufflepig.reporting import candidate_winning_subtype_for_analysis

    analysis = {
        "cancer_type": "SARC",
        "candidate_trace": [
            {"code": "SCLC"},
            {"code": "SARC", "winning_subtype": "SARC_PEC"},
        ],
        "cancer_type_evidence": {
            "selected": {
                "cancer_type": "SARC",
                "selected_by": "fused_evidence",
                "decision_features": {
                    "learned_compartment_anchored_pan_cancer_context": True,
                },
            }
        },
    }

    assert candidate_winning_subtype_for_analysis(analysis) is None


def test_integrated_evidence_blocked_child_does_not_reenter_parent_report():
    from trufflepig.reporting import candidate_winning_subtype_for_analysis

    analysis = {
        "cancer_type": "SARC",
        "candidate_trace": [
            {"code": "SARC", "winning_subtype": "SARC_PEC"},
        ],
        "cancer_type_evidence": {
            "selected": {
                "cancer_type": "SARC",
                "selected_by": "pan_cancer_signature_ranker",
            },
            "evidence": [
                {
                    "cancer_type": "SARC_PEC",
                    "can_select_report_label": False,
                    "blocking_reasons": [
                        "child reference contradicts an epithelial expected-low program"
                    ],
                }
            ],
        },
    }

    assert candidate_winning_subtype_for_analysis(analysis) is None


def test_consensus_parent_abstention_does_not_inherit_trace_subtype():
    from trufflepig.reporting import candidate_winning_subtype_for_analysis

    analysis = {
        "cancer_type": "CRC",
        "candidate_trace": [
            {"code": "CRC", "winning_subtype": "READ_MSS"},
            {"code": "COAD_MSS"},
        ],
        "cancer_type_evidence": {
            "selected": {
                "cancer_type": "CRC",
                "selected_by": "entity_evidence_consensus",
                "entity_consensus_adjudication_mode": (
                    "common_ancestor_abstention"
                ),
            }
        },
    }

    assert candidate_winning_subtype_for_analysis(analysis) is None


def test_consensus_entity_refinement_does_not_inherit_old_ranker_subtype():
    from trufflepig.reporting import candidate_winning_subtype_for_analysis

    analysis = {
        "cancer_type": "COAD",
        "candidate_trace": [
            {"code": "COAD", "winning_subtype": "READ"},
            {"code": "READ"},
        ],
        "cancer_type_evidence": {
            "selected": {
                "cancer_type": "COAD",
                "selected_by": "entity_evidence_consensus",
                "learned_hierarchy_adjudication_mode": (
                    "multi_axis_entity_refinement"
                ),
            }
        },
    }

    assert candidate_winning_subtype_for_analysis(analysis) is None


def test_hierarchy_entity_refinement_does_not_inherit_old_ranker_subtype():
    from trufflepig.reporting import candidate_winning_subtype_for_analysis

    analysis = {
        "cancer_type": "LAML",
        "candidate_trace": [
            {"code": "LAML", "winning_subtype": "CML"},
            {"code": "CML"},
        ],
        "cancer_type_evidence": {
            "selected": {
                "cancer_type": "LAML",
                "selected_by": "learned_expression_classifier",
                "learned_hierarchy_adjudicated": True,
                "learned_hierarchy_adjudication_mode": (
                    "high_precision_entity_refinement"
                ),
            }
        },
    }

    assert candidate_winning_subtype_for_analysis(analysis) is None


def test_brief_lusc_with_high_prame_mage_does_not_flag_nutm():
    """Pin the squamous-vs-NUTM correctness case: a LUSC sample with
    high PRAME + MAGEA3 (typical of LUSC) but silent NUTM1 should NOT
    be pulled into a NUT carcinoma flag. Before the activation-gate
    fix, vote-based disambiguation would have mis-called this as NUTM."""
    import pandas as pd


    analysis = {
        "purity": {
            "overall_estimate": 0.60,
            "overall_lower": 0.40,
            "overall_upper": 0.80,
        },
        "purity_confidence": type("PT", (), {"tier": "moderate"})(),
        "sample_context": None,
        "cancer_name": "Lung squamous cell carcinoma",
        "candidate_trace": [{"code": "LUSC", "winning_subtype": "LUSC"}],
        "decomposition": {
            "best_template": "primary_lung",
            "best_cancer_type": "LUSC",
        },
    }
    ranges_df = pd.DataFrame(
        [
            # Squamous lineage — this activates LUSC_vs_HNSC_vs_CESC
            {"symbol": "KRT5", "attr_tumor_tpm": 400.0},
            {"symbol": "KRT6A", "attr_tumor_tpm": 300.0},
            {"symbol": "TP63", "attr_tumor_tpm": 80.0},
            # High PRAME + MAGE-A as typical of LUSC (NOT NUTM-specific!)
            {"symbol": "PRAME", "attr_tumor_tpm": 30.0},
            {"symbol": "MAGEA3", "attr_tumor_tpm": 60.0},
            # But NUTM1 silent — NOT a NUT carcinoma
            {"symbol": "NUTM1", "attr_tumor_tpm": 0.1},
        ]
    )
    summary = render_summary(
        analysis,
        ranges_df=ranges_df,
        cancer_code="LUSC",
        disease_state="",
        sample_id="synthetic-lusc-high-cta",
    )
    assert "NUT" not in summary, (
        "LUSC with high PRAME/MAGEA but silent NUTM1 must NOT flag NUT carcinoma"
    )
    assert "degenerate" not in summary.lower() or "degenerate between" not in summary


def test_refs_populated_for_each_pair():
    """Every pair row should cite at least one reference (PMID or
    empty-cell for canonical/textbook cases). Enforces the curation
    bar from issue #198."""
    df = degenerate_subtype_pairs()
    # Allow at most one row with empty refs (the trivial COAD/READ
    # site case — canonical and doesn't need a citation).
    empty_refs = df[df["refs"].fillna("").astype(str).str.strip().eq("")]
    assert len(empty_refs) <= 1, (
        f"Too many degenerate-pair rows without a PMID citation: "
        f"{empty_refs['pair_id'].tolist()}"
    )
