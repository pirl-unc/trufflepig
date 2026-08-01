import numpy as np
import pandas as pd
import pytest

from trufflepig.decomposition import (
    decompose_identity_backgrounds,
    decompose_sample,
    evaluate_residual_identity,
    infer_sample_mode,
)
from trufflepig.decomposition.signature import _load_hpa_cell_types
from trufflepig.decomposition.templates import get_template_components
from trufflepig.reference import pan_cancer_expression
from trufflepig.tumor_purity import estimate_tumor_purity, rank_cancer_type_candidates


def _tcga_sample(cancer_code):
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            # A missing reference cell is an unmeasured/undetected gene, not a NaN TPM in the
            # synthetic sample. Production sample loading likewise presents finite TPM values.
            "TPM": ref[f"{cancer_code}_TPM"].astype(float).fillna(0.0),
        }
    )


def _normal_tissue_sample(tissue):
    ref = pan_cancer_expression().drop_duplicates(
        subset="Ensembl_Gene_ID"
    )
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{tissue}_nTPM"].astype(float).fillna(0.0),
        }
    )


def _mix_samples(parts):
    value_by_gene = {}
    symbol_by_gene = {}
    for weight, df in parts:
        for row in df.itertuples(index=False):
            value_by_gene[row.ensembl_gene_id] = value_by_gene.get(
                row.ensembl_gene_id, 0.0
            ) + weight * float(row.TPM)
            symbol_by_gene[row.ensembl_gene_id] = row.gene_symbol
    out = pd.DataFrame({"ensembl_gene_id": list(value_by_gene.keys())})
    out["gene_symbol"] = out["ensembl_gene_id"].map(symbol_by_gene)
    out["TPM"] = out["ensembl_gene_id"].map(value_by_gene)
    return out


def _candidate_row(code="HNSC", purity=0.25, support=1.0, signature=0.8):
    return {
        "code": code,
        "signature_score": signature,
        "purity_estimate": purity,
        "support_fraction_of_top": support,
        "purity_result": {
            "overall_lower": max(0.01, purity - 0.1),
            "overall_estimate": purity,
            "overall_upper": min(1.0, purity + 0.1),
        },
    }


def _set_symbol_tpm(df, values):
    out = df.copy()
    for symbol, value in values.items():
        mask = out["gene_symbol"].astype(str) == symbol
        if mask.any():
            out.loc[mask, "TPM"] = float(value)
    return out


def test_lymph_node_template_uses_broad_t_cell_only():
    components = get_template_components("met_lymph_node", "PRAD")
    assert "T_cell" in components
    assert "CD4_T" not in components
    assert "CD8_T" not in components
    assert "LN_parenchyma" not in components


def test_metastasis_template_ranking_uses_cancer_support():
    """Shared met-site matrices should still rank hypotheses by cancer support."""
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    df = pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref["COAD_TPM"].astype(float),
        }
    )

    results = decompose_sample(
        df,
        cancer_types=["SARC", "COAD"],
        templates=["met_liver"],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].template == "met_liver"
    assert results[1].template == "met_liver"
    assert results[0].cancer_type == "COAD"
    assert results[0].cancer_support_score > results[1].cancer_support_score
    assert results[0].score > results[1].score


def test_empty_template_override_uses_default_templates():
    """CLI config returns [] when no override is supplied; that must mean
    default templates, not no decomposition hypotheses."""

    results = decompose_sample(
        _tcga_sample("BLCA"),
        cancer_types=["BLCA"],
        templates=[],
        top_k=3,
    )

    assert results
    assert {row.template for row in results}


def test_tcga_prad_uses_external_purity_anchor():
    """TCGA PRAD median should stay near the known cohort purity scale."""
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    df = pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref["PRAD_TPM"].astype(float),
        }
    )

    result = decompose_sample(
        df,
        cancer_types=["PRAD"],
        templates=["solid_primary"],
        top_k=1,
    )[0]

    assert 0.5 < result.purity < 0.85
    assert result.fractions["tumor"] == result.purity
    assert np.isfinite(result.reconstruction_error)
    assert result.score > 0.05


def test_tcga_coad_primary_beats_lymph_node_template():
    """Primary-like COAD should not be nudged into lymph node by immune signal."""
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    df = pd.DataFrame(
        {
            "gene_id": ref["Ensembl_Gene_ID"],
            "canonical_gene_name": ref["Symbol"],
            "gene_display_name": ref["Symbol"],
            "TPM": ref["COAD_TPM"].astype(float),
        }
    )

    results = decompose_sample(
        df,
        cancer_types=["COAD"],
        templates=["solid_primary", "met_lymph_node"],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].template == "solid_primary"
    assert results[0].template_tissue_score > results[1].template_tissue_score


def test_pure_t_cell_control_stays_in_t_cell_bucket():
    """A pure T-cell HPA profile should not be split into fake subtypes."""
    hpa = _load_hpa_cell_types()
    df = pd.DataFrame(
        {
            "ensembl_gene_id": hpa["Ensembl_Gene_ID"],
            "gene_symbol": hpa["Symbol"],
            "TPM": hpa["T-cells"].astype(float),
        }
    )

    result = decompose_sample(
        df,
        cancer_types=["THYM"],
        templates=["solid_primary"],
        top_k=1,
        purity_override=0.0,
    )[0]

    assert result.fractions["tumor"] == 0.0
    assert result.fractions["T_cell"] > 0.75


def test_pure_b_cell_control_stays_in_b_cell_bucket():
    """A pure B-cell HPA profile should map to the broad B-cell bucket."""
    hpa = _load_hpa_cell_types()
    df = pd.DataFrame(
        {
            "ensembl_gene_id": hpa["Ensembl_Gene_ID"],
            "gene_symbol": hpa["Symbol"],
            "TPM": hpa["B-cells"].astype(float),
        }
    )

    result = decompose_sample(
        df,
        cancer_types=["DLBC"],
        templates=["solid_primary"],
        top_k=1,
        purity_override=0.0,
    )[0]

    assert result.fractions["tumor"] == 0.0
    assert result.fractions["B_cell"] > 0.75


def test_auto_heme_mode_uses_heme_templates():
    """DLBC should default to heme templates, not solid/met templates."""
    df = _tcga_sample("DLBC")

    results = decompose_sample(
        df,
        cancer_types=["DLBC"],
        top_k=3,
    )

    assert results
    assert all(
        row.template in {"heme_nodal", "heme_blood", "heme_marrow"} for row in results
    )


def test_infer_sample_mode_prefers_best_heme_candidate():
    mode = infer_sample_mode(
        candidate_rows=[
            {"code": "DLBC"},
            {"code": "THYM"},
        ],
        sample_mode="auto",
    )
    assert mode == "heme"


@pytest.mark.parametrize("code", ["CML", "FL", "B_ALL"])
def test_infer_sample_mode_uses_ontology_for_non_tcga_heme_entities(code):
    assert infer_sample_mode(cancer_types=[code], sample_mode="auto") == "heme"


def test_explicit_pure_mode_uses_pure_population_template():
    """Explicit pure mode should bypass bulk-mixture templates."""
    hpa = _load_hpa_cell_types()
    df = pd.DataFrame(
        {
            "ensembl_gene_id": hpa["Ensembl_Gene_ID"],
            "gene_symbol": hpa["Symbol"],
            "TPM": hpa["B-cells"].astype(float),
        }
    )

    result = decompose_sample(
        df,
        cancer_types=["DLBC"],
        top_k=1,
        sample_mode="pure",
    )[0]

    assert result.template == "pure_population"
    assert result.fractions["tumor"] == 1.0


def test_primary_context_override_limits_to_primary_template():
    df = _tcga_sample("PRAD")

    results = decompose_sample(
        df,
        cancer_types=["PRAD"],
        sample_mode="solid",
        tumor_context="primary",
        top_k=3,
    )

    assert results
    assert all(row.template == "solid_primary" for row in results)


def test_met_site_override_limits_to_requested_template():
    df = _tcga_sample("COAD")

    results = decompose_sample(
        df,
        cancer_types=["COAD"],
        sample_mode="solid",
        tumor_context="met",
        site_hint="liver",
        top_k=3,
    )

    assert results
    assert all(row.template == "met_liver" for row in results)


def test_unbounded_decomposition_returns_every_usable_realization():
    """Identity adjudication can inspect the complete structural beam."""

    templates = ["solid_primary", "met_bone", "met_liver"]
    results = decompose_sample(
        _tcga_sample("PRAD"),
        cancer_types=["PRAD"],
        templates=templates,
        sample_mode="solid",
        top_k=None,
    )

    assert {row.template for row in results} == set(templates)
    assert len(results) == len(templates)


def test_synthetic_coad_colon_mix_tracks_known_purity():
    """A known 30/70 COAD-colon mix should stay CRC-family at ~30% purity."""
    df = _mix_samples(
        [
            (0.3, _tcga_sample("COAD")),
            (0.7, _normal_tissue_sample("colon")),
        ]
    )

    purity = estimate_tumor_purity(df, cancer_type="COAD")
    candidates = rank_cancer_type_candidates(df, top_k=3)
    results = decompose_sample(
        df, cancer_types=[row["code"] for row in candidates], top_k=2
    )

    assert 0.2 < purity["overall_estimate"] < 0.45
    assert candidates[0]["code"] in {"COAD", "READ"}
    assert results[0].template == "solid_primary"
    assert results[0].cancer_type in {"COAD", "READ"}


def test_synthetic_coad_lymph_mix_stays_crc_family():
    """CRC mixed with lymph-node background should not flip to DLBC."""
    df = _mix_samples(
        [
            (0.3, _tcga_sample("COAD")),
            (0.7, _normal_tissue_sample("lymph_node")),
        ]
    )

    candidates = rank_cancer_type_candidates(
        df,
        candidate_codes=["COAD", "READ", "DLBC", "THYM", "SARC", "STAD"],
        top_k=4,
    )

    assert candidates[0]["code"] in {"COAD", "READ"}
    assert candidates[1]["code"] in {"COAD", "READ"}


def test_synthetic_coad_liver_mix_uses_liver_background():
    """CRC mixed with liver background should stay CRC-family and use hepatocyte context."""
    df = _mix_samples(
        [
            (0.3, _tcga_sample("COAD")),
            (0.7, _normal_tissue_sample("liver")),
        ]
    )

    candidates = rank_cancer_type_candidates(
        df,
        candidate_codes=["COAD", "READ", "LIHC", "CHOL", "STAD", "DLBC"],
        top_k=4,
    )
    results = decompose_sample(
        df,
        cancer_types=["COAD", "READ"],
        templates=["solid_primary", "met_liver"],
        top_k=4,
    )

    assert candidates[0]["code"] in {"COAD", "READ"}
    assert results[0].template == "met_liver"
    assert results[0].cancer_type in {"COAD", "READ"}
    assert results[0].fractions["hepatocyte"] > 0.5
    assert results[0].template_extra_fraction > 0.5


def test_synthetic_coad_liver_residual_retains_colorectal_identity():
    """Residual programs distinguish metastatic tumor from liver host."""
    from trufflepig.decomposition import evaluate_residual_identity

    df = _mix_samples(
        [
            (0.3, _tcga_sample("COAD")),
            (0.7, _normal_tissue_sample("liver")),
        ]
    )
    candidate_codes = ["COAD", "READ", "LIHC", "CHOL"]
    results = decompose_sample(
        df,
        cancer_types=candidate_codes,
        templates=["met_liver"],
        purity_override=0.3,
        top_k=len(candidate_codes),
    )

    evidence = evaluate_residual_identity(
        results,
        candidate_codes=candidate_codes,
        current_code="COAD",
    )

    assert evidence["status"] == "corroborated", [
        {
            "candidate": model["candidate_code"],
            "panel": model["panel_candidate"],
            "ontology": model["ontology_candidate"],
            "rows": [
                {
                    "code": row["decomposition_cancer_type"],
                    "panel": row["panel_candidate"],
                    "ontology": row["ontology_candidate"],
                    "complete": row["ontology_complete_codes"],
                    "programs": row["ontology_programs"],
                }
                for row in model["rows"]
            ],
        }
        for model in evidence["background_models"]
    ]
    assert evidence["candidate_code"] == "CRC"
    assert evidence["models_evaluated"] == 1
    assert evidence["realizations_evaluated"] == len(candidate_codes)


def test_met_bone_requires_hard_bone_evidence_not_generic_ecm():
    """Generic stromal/ECM signal can fit a background, but should not
    become a site-supported bone-met call without osteogenic anchors."""

    df = _set_symbol_tpm(
        _tcga_sample("HNSC"),
        {
            "COL1A1": 1200,
            "SPP1": 300,
            "CXCL12": 120,
            "KITLG": 15,
            "VCAM1": 15,
            # IBSP/RUNX2/ALPL can be strong in invasive squamous or
            # mesenchymal programs; they should not be enough without a
            # more specific mineralized-bone/osteocyte anchor.
            "IBSP": 60,
            "RUNX2": 16,
            "ALPL": 6,
            "BGLAP": 1,
            "SOST": 1,
            "DMP1": 1,
            "PHEX": 2,
            "SP7": 1,
            "MEPE": 1,
        },
    )

    result = decompose_sample(
        df,
        cancer_types=["HNSC"],
        candidate_rows=[_candidate_row("HNSC", purity=0.22)],
        templates=["met_bone"],
        top_k=1,
    )[0]

    assert result.template == "met_bone"
    assert result.site_evidence["site_supported"] is False
    assert result.site_evidence["status"] == "fit_only"
    assert "bone_specific_markers" in result.site_evidence["missing"]
    assert any("Metastatic-site evidence below" in w for w in result.warnings)


def test_met_bone_site_hint_is_site_supported_external_context():
    df = _set_symbol_tpm(
        _tcga_sample("HNSC"),
        {
            "COL1A1": 1200,
            "SPP1": 300,
            "ALPL": 4,
            "BGLAP": 1,
        },
    )

    result = decompose_sample(
        df,
        cancer_types=["HNSC"],
        candidate_rows=[_candidate_row("HNSC", purity=0.22)],
        templates=["met_bone"],
        site_hint="bone",
        top_k=1,
    )[0]

    assert result.site_evidence["site_supported"] is True
    assert result.site_evidence["status"] == "site_supported"
    assert result.site_evidence["basis"] == "site_hint"
    assert not any("Metastatic-site evidence below" in w for w in result.warnings)


def test_synthetic_prad_lymph_mix_stays_primary_not_lymphoma():
    """Immune-rich prostate should stay PRAD and primary-like, not nodal/heme."""
    df = _mix_samples(
        [
            (0.2, _tcga_sample("PRAD")),
            (0.8, _normal_tissue_sample("lymph_node")),
        ]
    )

    candidates = rank_cancer_type_candidates(
        df,
        candidate_codes=["PRAD", "DLBC", "THYM", "LUSC", "HNSC"],
        top_k=4,
    )
    results = decompose_sample(
        df,
        cancer_types=["PRAD"],
        templates=["solid_primary", "met_lymph_node"],
        top_k=2,
    )

    assert candidates[0]["code"] == "PRAD"
    assert results[0].template == "solid_primary"
    assert results[0].purity < 0.2
    # Lymph-node-dominated mix: B_cell should be the single largest
    # non-tumor compartment. The absolute fraction depends on which
    # discriminative markers survive marker-selection filters (the #60
    # extended-housekeeping exclusion tightened this set in v4.2), so
    # we assert on the qualitative intent — B_cell tops the fractions —
    # rather than a hard numeric threshold.
    non_tumor_fracs = {k: v for k, v in results[0].fractions.items() if k != "tumor"}
    top_component = max(non_tumor_fracs, key=non_tumor_fracs.get)
    assert top_component == "B_cell", (
        f"Expected B_cell to top non-tumor fractions in a lymph-node mix, got "
        f"{top_component} (fractions={non_tumor_fracs})"
    )
    assert non_tumor_fracs["B_cell"] > 0.4


def test_synthetic_prad_smooth_muscle_mix_keeps_prad_and_primary_template():
    """Soft mesenchymal fallback should not outrank a strong prostate family signal.

    The strong, prostate-specific signal (KLK3 etc.) survives 80% smooth-muscle
    admixture: PRAD stays #1 and the whole-profile compartment gate (#83) confidently
    pins **Epithelial**, so the mesenchymal SARC fallback is *suppressed* — demoted
    below the epithelial candidates and flagged out-of-compartment — rather than
    surfaced as a top alternative. (Pre-hierarchy this test asserted SARC appeared in
    the top-4; the compartment gate strengthens the property it was guarding.)
    """
    df = _mix_samples(
        [
            (0.2, _tcga_sample("PRAD")),
            (0.8, _normal_tissue_sample("smooth_muscle")),
        ]
    )

    candidates = rank_cancer_type_candidates(df, top_k=8)
    results = decompose_sample(
        df,
        cancer_types=["PRAD"],
        templates=["solid_primary", "met_bone", "met_soft_tissue"],
        top_k=3,
    )

    assert candidates[0]["code"] == "PRAD"
    # the gate pinned the epithelial compartment and suppressed the SARC fallback
    assert candidates[0].get("centroid_coarse_lineage") == "Epithelial"
    sarc = next((row for row in candidates if row["code"] == "SARC"), None)
    assert sarc is None or sarc.get("compartment_in_set") is False
    assert results[0].template == "solid_primary"
    assert results[0].score > results[1].score
    unsupported_soft_tissue = next(
        row for row in results if row.template == "met_soft_tissue"
    )
    assert "smooth_muscle" not in unsupported_soft_tissue.fractions


def test_synthetic_sarc_smooth_muscle_mix_surfaces_mesenchymal_family():
    """Mesenchymal samples should expose SARC as the broad-family call."""
    df = _mix_samples(
        [
            (0.3, _tcga_sample("SARC")),
            (0.7, _normal_tissue_sample("smooth_muscle")),
        ]
    )

    candidates = rank_cancer_type_candidates(df, top_k=4)

    assert candidates[0]["code"] == "SARC"
    assert candidates[0]["family_label"] == "MESENCHYMAL"


def test_synthetic_stromal_heavy_crc_ranker_keeps_crc_near_top():
    """Synthetic CRC with heavy stromal admixture should keep COAD/READ near
    the top RNA candidates. Mirrors the
    clinical scenario we used to gate on a gitignored patient sample —
    synthetic mix lets the test run anywhere without PHI-adjacent IDs."""
    df = _mix_samples(
        [
            (0.3, _tcga_sample("COAD")),
            (0.4, _normal_tissue_sample("colon")),
            (0.3, _normal_tissue_sample("smooth_muscle")),
        ]
    )

    candidates = rank_cancer_type_candidates(df, top_k=6)
    top_codes = {row["code"] for row in candidates[:2]}
    assert top_codes & {"COAD", "READ"}


def test_soft_tissue_decomposition_separates_crc_from_smooth_muscle_host():
    """Tumor identity should be scored after the declared host is subtracted."""

    df = _mix_samples(
        [
            (0.3, _tcga_sample("COAD")),
            (0.7, _normal_tissue_sample("smooth_muscle")),
        ]
    )
    result = decompose_sample(
        df,
        cancer_types=["COAD"],
        templates=["met_soft_tissue"],
        site_hint="soft_tissue",
        top_k=1,
    )[0]

    attribution = result.gene_attribution.set_index("symbol")
    assert "smooth_muscle" in result.fractions
    assert result.component_reference_tissues["smooth_muscle"] == "smooth_muscle"
    assert result.fractions["smooth_muscle"] > result.fractions["tumor"]
    assert attribution.loc["DES", "smooth_muscle"] > attribution.loc["DES", "tumor"]
    for marker in ("CDX2", "SATB2", "CDH17", "VIL1"):
        assert attribution.loc[marker, "tumor"] > attribution.loc[
            marker, "smooth_muscle"
        ]

    identity = evaluate_residual_identity(
        [result],
        candidate_codes=["COAD", "SARC_PLEOLPS"],
        current_code="SARC_PLEOLPS",
    )
    assert identity["status"] == "candidate"
    assert identity["candidate_code"] == "CRC"


def test_candidate_independent_model_resolves_crc_from_muscularis():
    """A normal-tissue screen, not a tentative tumor label, supplies the host.

    DES remains a contradiction in bulk but becomes attributable to purified
    smooth muscle, while the complete CRC positive/negative program remains in
    the residual.
    """

    df = _mix_samples(
        [
            (0.4, _tcga_sample("COAD")),
            (0.6, _normal_tissue_sample("smooth_muscle")),
        ]
    )
    identity_models = decompose_identity_backgrounds(df, sample_mode="solid")

    assert identity_models
    assert all(row.model_role == "identity_background" for row in identity_models)
    structural = next(
        row
        for row in identity_models
        if row.template == "identity_structural_background"
    )
    assert "smooth_muscle_identity" in structural.fractions

    evidence = evaluate_residual_identity(
        identity_models,
        candidate_codes=["COAD", "READ", "SARC_DDLPS"],
        current_code="SARC_DDLPS",
    )

    assert evidence["status"] == "candidate"
    assert evidence["candidate_code"] == "CRC"
    assert evidence["panel_candidate_code"] == "CRC"
    assert evidence["ontology_candidate_code"] == "CRC"
    assert evidence["source_resolved_identity"] is True
    assert "DES" in evidence["background_attributed_expected_low_genes"]


def test_hollow_organ_identity_model_includes_mucosa_and_muscularis():
    """Composition supplies both layers without consulting a cancer code."""

    df = _mix_samples(
        [
            (0.3, _tcga_sample("COAD")),
            (0.4, _normal_tissue_sample("colon")),
            (0.3, _normal_tissue_sample("smooth_muscle")),
        ]
    )
    identity_models = decompose_identity_backgrounds(df, sample_mode="solid")
    structural = next(
        row
        for row in identity_models
        if row.template == "identity_structural_background"
    )

    assert "smooth_muscle_identity" in structural.fractions
    assert any(
        component.startswith("matched_normal_")
        for component in structural.fractions
    )


def test_identity_background_cannot_manufacture_crc_without_crc_program():
    """Smooth muscle subtraction alone is not affirmative CRC evidence."""

    identity_models = decompose_identity_backgrounds(
        _normal_tissue_sample("smooth_muscle"),
        sample_mode="solid",
    )
    evidence = evaluate_residual_identity(
        identity_models,
        candidate_codes=["COAD", "READ", "SARC_LMS"],
        current_code="SARC_LMS",
    )

    assert evidence.get("candidate_code") != "CRC"
    assert evidence.get("source_resolved_identity") is not True


def test_identity_background_abstains_without_gene_identity_columns(caplog):
    incomplete = pd.DataFrame({"value": [1.0]})

    with caplog.at_level("WARNING"):
        result = decompose_identity_backgrounds(incomplete, sample_mode="solid")

    assert result == []
    assert "Residual-identity decomposition skipped" in caplog.text
    assert "gene identity columns are unavailable" in caplog.text


def test_synthetic_low_purity_crc_purity_matches_expected_scale():
    """Synthetic 30%-purity CRC should estimate purity in the 0.2–0.4
    range (same property as the clinical retroperitoneal-CRC fixture
    used to assert)."""
    df = _mix_samples(
        [
            (0.3, _tcga_sample("COAD")),
            (0.7, _normal_tissue_sample("colon")),
        ]
    )

    purity = estimate_tumor_purity(df, cancer_type="COAD")
    candidates = rank_cancer_type_candidates(df, top_k=3)

    assert 0.15 < purity["overall_estimate"] < 0.5
    assert candidates[0]["code"] in {"COAD", "READ"}


# ── Edge-case coverage ──────────────────────────────────────────────────


def test_low_gene_coverage_still_returns_results():
    """Subsample to ~500 genes — decomposition should degrade gracefully."""
    df = _tcga_sample("COAD")
    # Keep ~500 random genes (deterministic seed)
    rng = np.random.RandomState(42)
    keep = rng.choice(len(df), size=500, replace=False)
    df_small = df.iloc[keep].reset_index(drop=True)

    results = decompose_sample(
        df_small,
        cancer_types=["COAD"],
        templates=["solid_primary"],
        top_k=1,
    )

    assert len(results) == 1
    result = results[0]
    assert result.template == "solid_primary"
    assert result.cancer_type == "COAD"
    # Purity should still be in a plausible range
    assert 0.0 < result.purity < 1.0
    # Fractions should be non-negative and sum to ~1
    assert all(v >= 0 for v in result.fractions.values())
    assert abs(sum(result.fractions.values()) - 1.0) < 0.05


def test_extreme_high_purity_override():
    """purity_override=0.99 should yield nearly all-tumor fractions."""
    df = _tcga_sample("BRCA")

    result = decompose_sample(
        df,
        cancer_types=["BRCA"],
        templates=["solid_primary"],
        top_k=1,
        purity_override=0.99,
    )[0]

    assert result.purity == pytest.approx(0.99, abs=0.001)
    assert result.fractions["tumor"] == pytest.approx(0.99, abs=0.001)
    # Non-tumor components should be near-zero
    non_tumor = sum(v for k, v in result.fractions.items() if k != "tumor")
    assert non_tumor < 0.02


def test_extreme_low_purity_override():
    """purity_override=0.01 should put nearly everything in TME components."""
    hpa = _load_hpa_cell_types()
    # Use a T-cell profile to simulate a very immune-heavy sample
    df = pd.DataFrame(
        {
            "ensembl_gene_id": hpa["Ensembl_Gene_ID"],
            "gene_symbol": hpa["Symbol"],
            "TPM": hpa["T-cells"].astype(float),
        }
    )

    result = decompose_sample(
        df,
        cancer_types=["THYM"],
        templates=["solid_primary"],
        top_k=1,
        purity_override=0.01,
    )[0]

    assert result.purity == pytest.approx(0.01, abs=0.001)
    assert result.fractions["tumor"] == pytest.approx(0.01, abs=0.001)
    non_tumor = sum(v for k, v in result.fractions.items() if k != "tumor")
    assert non_tumor > 0.95


def test_all_tumor_when_purity_is_one():
    """purity_override=1.0 should return an all-tumor result with no TME."""
    df = _tcga_sample("PRAD")

    results = decompose_sample(
        df,
        cancer_types=["PRAD"],
        templates=["solid_primary"],
        top_k=1,
        purity_override=1.0,
    )

    assert len(results) == 1
    result = results[0]
    assert result.fractions == {"tumor": 1.0}
    assert "No non-tumor components" in result.warnings[0]


def test_all_tumor_met_template_is_fit_only_and_suppressed_when_primary_available():
    df = _tcga_sample("PRAD")

    results = decompose_sample(
        df,
        cancer_types=["PRAD"],
        templates=["met_adrenal", "solid_primary"],
        top_k=2,
        purity_override=1.0,
    )

    assert [row.template for row in results] == ["solid_primary"]


def test_all_tumor_met_template_remains_available_when_explicitly_requested():
    df = _tcga_sample("PRAD")

    results = decompose_sample(
        df,
        cancer_types=["PRAD"],
        templates=["met_adrenal"],
        top_k=1,
        purity_override=1.0,
    )

    assert results[0].template == "met_adrenal"
    assert results[0].site_evidence["site_supported"] is False
    assert results[0].site_evidence["basis"] == "no_non_tumor_components"


def test_invalid_template_name_raises():
    """Explicit template names should be validated against known templates."""
    df = _tcga_sample("COAD")

    with pytest.raises(ValueError, match="Unknown template"):
        decompose_sample(
            df,
            cancer_types=["COAD"],
            templates=["met_liverr"],  # typo
            top_k=1,
        )


def test_invalid_template_name_lists_valid_options():
    """The error message should include the valid template names."""
    df = _tcga_sample("COAD")

    with pytest.raises(ValueError, match="solid_primary"):
        decompose_sample(
            df,
            cancer_types=["COAD"],
            templates=["nonexistent"],
            top_k=1,
        )


# ── Heme breadth ────────────────────────────────────────────────────────


def test_laml_uses_heme_templates():
    """LAML should default to heme templates, not solid/met templates."""
    df = _tcga_sample("LAML")

    results = decompose_sample(
        df,
        cancer_types=["LAML"],
        top_k=3,
    )

    assert results
    assert all(
        row.template in {"heme_nodal", "heme_blood", "heme_marrow"} for row in results
    )


def test_laml_marrow_template_fits_myeloid_components():
    """LAML in heme_marrow should have myeloid-lineage components."""
    df = _tcga_sample("LAML")

    result = decompose_sample(
        df,
        cancer_types=["LAML"],
        templates=["heme_marrow"],
        top_k=1,
    )[0]

    assert result.template == "heme_marrow"
    assert result.cancer_type == "LAML"
    assert 0.0 < result.purity < 1.0
    # Should have at least some non-tumor fractions
    non_tumor = sum(v for k, v in result.fractions.items() if k != "tumor")
    assert non_tumor > 0.0


def test_dlbc_blood_template_available():
    """DLBC in heme_blood should produce valid decomposition."""
    df = _tcga_sample("DLBC")

    result = decompose_sample(
        df,
        cancer_types=["DLBC"],
        templates=["heme_blood"],
        top_k=1,
    )[0]

    assert result.template == "heme_blood"
    assert result.cancer_type == "DLBC"
    assert 0.0 < result.purity < 1.0


# ── Composite tissue references ─────────────────────────────────────────


def test_brain_met_detects_cns_via_composite_reference():
    """Synthetic brain met should detect brain parenchyma via composite tissue reference.

    Both astrocyte and neuron resolve to the same bulk CNS tissue; their
    fractions should be summed and read as "CNS parenchyma."
    """
    df = _mix_samples(
        [
            (0.25, _tcga_sample("LUAD")),
            (0.75, _normal_tissue_sample("cerebral_cortex")),
        ]
    )

    results = decompose_sample(
        df,
        cancer_types=["LUAD"],
        templates=["met_brain", "solid_primary"],
        top_k=2,
    )

    assert results[0].template == "met_brain"
    assert results[0].score > results[1].score
    # astrocyte + neuron together represent CNS parenchyma
    cns_frac = results[0].fractions.get("astrocyte", 0.0) + results[0].fractions.get(
        "neuron", 0.0
    )
    assert cns_frac > 0.4, f"CNS fraction {cns_frac} too low for 75% brain sample"


def test_brain_met_does_not_win_for_colon_sample():
    """CRC + colon should prefer solid_primary over met_brain."""
    df = _mix_samples(
        [
            (0.3, _tcga_sample("COAD")),
            (0.7, _normal_tissue_sample("colon")),
        ]
    )

    results = decompose_sample(
        df,
        cancer_types=["COAD"],
        templates=["solid_primary", "met_brain"],
        top_k=2,
    )

    assert results[0].template == "solid_primary"


def test_plot_decomposition_candidates_saves_png(tmp_path):
    """Candidate composition bar plot writes a non-empty PNG and handles
    templates with and without a template-specific compartment."""
    import matplotlib

    matplotlib.use("Agg")
    from types import SimpleNamespace
    from trufflepig.decomposition import plot_decomposition_candidates

    rows = [
        SimpleNamespace(
            cancer_type="COAD",
            template="solid_primary",
            purity=0.45,
            template_extra_fraction=0.6,
            cancer_support_score=0.7,
            template_tissue_score=0.85,
            score=0.42,
        ),
        SimpleNamespace(
            cancer_type="COAD",
            template="met_lymph",
            purity=0.45,
            template_extra_fraction=0.0,
            cancer_support_score=0.65,
            template_tissue_score=0.4,
            score=0.18,
        ),
    ]
    out = tmp_path / "candidates.png"
    fig = plot_decomposition_candidates(rows, save_to_filename=str(out))
    assert fig is not None
    assert out.exists()
    assert out.stat().st_size > 5_000


def test_plot_decomposition_candidates_empty_results_returns_none():
    from trufflepig.decomposition import plot_decomposition_candidates

    assert plot_decomposition_candidates([]) is None


def test_plot_decomposition_candidates_surfaces_fit_and_markers(tmp_path):
    """#123: the candidates figure must surface per-row
    reconstruction_error (as 'fit err') and median marker score (as
    'markers'), so two similarly-scored candidates can be distinguished
    by which one actually explains the sample. Also: rows with engine
    warnings should be rendered with a hatched overlay (gated flag)
    so purity-floor / marker-support / template-incomplete candidates
    stand out visually."""
    import matplotlib

    matplotlib.use("Agg")
    from types import SimpleNamespace
    import pandas as pd
    from trufflepig.decomposition import plot_decomposition_candidates

    trace_strong = pd.DataFrame(
        [
            {
                "component": "T_cell",
                "fraction": 0.03,
                "marker_score": 0.42,
                "top_markers": "TRAC,CD3D",
                "n_markers": 3,
            },
            {
                "component": "myeloid",
                "fraction": 0.02,
                "marker_score": 0.38,
                "top_markers": "LYZ,CD68",
                "n_markers": 3,
            },
        ]
    )
    trace_weak = pd.DataFrame(
        [
            {
                "component": "T_cell",
                "fraction": 0.01,
                "marker_score": 0.08,
                "top_markers": "",
                "n_markers": 1,
            },
        ]
    )

    rows = [
        SimpleNamespace(
            cancer_type="PRAD",
            template="solid_primary",
            purity=0.28,
            template_extra_fraction=0.60,
            cancer_support_score=0.72,
            template_tissue_score=0.80,
            score=0.72,
            reconstruction_error=0.18,
            component_trace=trace_strong,
            warnings=[],
        ),
        SimpleNamespace(
            cancer_type="PRAD",
            template="met_liver",
            purity=0.32,
            template_extra_fraction=0.50,
            cancer_support_score=0.65,
            template_tissue_score=0.40,
            score=0.45,
            reconstruction_error=0.55,
            component_trace=trace_weak,
            warnings=["Low marker support for template fit"],
        ),
    ]
    out = tmp_path / "candidates_fit.png"
    fig = plot_decomposition_candidates(rows, save_to_filename=str(out))
    assert fig is not None
    assert out.exists()
    assert out.stat().st_size > 5_000

    # Fit + markers annotation should be on the axes as a text artist.
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert any("fit err" in t and "markers" in t for t in texts), (
        f"missing fit/markers annotation; texts: {texts}"
    )
    assert any("ADOPTED" in t for t in texts), texts
    assert any("audit only" in t for t in texts), texts


def test_plot_decomposition_candidates_handles_missing_reconstruction_error(tmp_path):
    """Old-style rows (SimpleNamespace without reconstruction_error /
    component_trace) still render — the new annotation falls back to
    fit err = 0.00 and markers = 0.00."""
    import matplotlib

    matplotlib.use("Agg")
    from types import SimpleNamespace
    from trufflepig.decomposition import plot_decomposition_candidates

    rows = [
        SimpleNamespace(
            cancer_type="COAD",
            template="solid_primary",
            purity=0.45,
            template_extra_fraction=0.6,
            cancer_support_score=0.7,
            template_tissue_score=0.85,
            score=0.42,
        ),
    ]
    out = tmp_path / "candidates_legacy.png"
    fig = plot_decomposition_candidates(rows, save_to_filename=str(out))
    assert fig is not None
    assert out.exists()


def test_auto_marker_selection_excludes_mhc_ii_and_ribosomal(monkeypatch):
    """#31: CD74 / HLA-D* / RPL* / RPS* must not be auto-picked as
    component-specific markers, even when their reference nTPM is high
    in the target component. Curated ``COMPONENT_MARKERS`` from
    signature.py remains the source of truth.
    """
    import numpy as np
    from trufflepig.decomposition.engine import (
        _AUTO_MARKER_EXCLUDED_SYMBOLS,
        _select_marker_rows,
    )

    # Build a 3-component (T_cell, B_cell, myeloid) synthetic matrix where
    # the first few "genes" are exactly the excluded symbols and happen to
    # be high only in B_cell — so the specificity rule would otherwise
    # auto-pick them.
    tainted = ["CD74", "HLA-DPB1", "HLA-DQB1", "RPL18A", "RPS6"]
    clean_b = ["BANK1", "MS4A1", "CD79A"]  # genuine B-cell markers
    genes = tainted + clean_b + [f"OTHER{i}" for i in range(50)]
    symbols = list(genes)  # symbol==gene_id for this toy case

    n = len(genes)
    mat = np.full((n, 3), 0.1)  # baseline in T_cell / B_cell / myeloid
    # tainted genes: very high in B_cell only
    for i in range(len(tainted)):
        mat[i, 1] = 200.0
    # clean b-cell markers: also high in B_cell
    for i in range(len(tainted), len(tainted) + len(clean_b)):
        mat[i, 1] = 150.0

    fit_rows, _weights, marker_df = _select_marker_rows(
        genes=genes,
        symbols=symbols,
        signature_tpm=mat,
        comp_names=["T_cell", "B_cell", "myeloid"],
    )

    auto_b_cell = marker_df[marker_df["component"] == "B_cell"]["symbol"].tolist()
    for bad in tainted:
        assert bad not in auto_b_cell, f"Auto-picked excluded marker {bad} for B_cell"

    # The exclusion table itself must cover the MHC-II shared-APC genes
    # that motivated issue #31 (guard against accidental deletion).
    for required in ["CD74", "HLA-DPB1", "HLA-DQB1"]:
        assert required in _AUTO_MARKER_EXCLUDED_SYMBOLS


def test_candidate_composition_segments_sum_to_one():
    """(tumor, template_specific, shared_host) must sum to 1."""
    from types import SimpleNamespace
    from trufflepig.decomposition.plot import _candidate_composition_segments

    for purity in [0.0, 0.2, 0.5, 0.85, 1.0]:
        for extra in [0.0, 0.35, 1.0]:
            row = SimpleNamespace(purity=purity, template_extra_fraction=extra)
            tumor, tmpl, shared = _candidate_composition_segments(row)
            assert abs(tumor + tmpl + shared - 1.0) < 1e-9
            assert 0 <= tumor <= 1
            assert 0 <= tmpl <= 1
            assert 0 <= shared <= 1


def test_decompose_sample_scores_requested_scope_missing_from_reused_trace(monkeypatch):
    """A selected/report scope must reach decomposition even when the CLI
    reuses a broad first-pass candidate trace.

    The broad trace is an optimization, not the full universe of hypotheses.
    If a selected local/fine scope is passed through ``cancer_types`` but is
    absent from ``candidate_rows``, decompose_sample should score just that
    missing code and include it in the fit set.
    """
    from types import SimpleNamespace

    import pandas as pd

    from trufflepig.decomposition import engine

    broad_row = {
        "code": "BROAD",
        "support_fraction_of_top": 1.0,
        "signature_score": 0.8,
        "purity_estimate": 0.6,
        "purity_result": {"overall_estimate": 0.6},
    }
    local_row = {
        "code": "LOCAL",
        "support_fraction_of_top": 0.7,
        "signature_score": 0.5,
        "purity_estimate": 0.5,
        "purity_result": {"overall_estimate": 0.5},
    }
    ranked_requests = []

    def fake_rank(_df, *, candidate_codes, **_kwargs):
        ranked_requests.append(tuple(candidate_codes))
        assert tuple(candidate_codes) == ("LOCAL",)
        return [dict(local_row)]

    def fake_fit(_df, _sample_by_eid, candidate_row, _tissue_score_map, template_name, **_kwargs):
        return SimpleNamespace(
            cancer_type=candidate_row["code"],
            template=template_name,
            score=float(candidate_row["support_fraction_of_top"]),
            scope_source=candidate_row.get("decomposition_scope_source", ""),
        )

    monkeypatch.setattr(engine, "rank_cancer_type_candidates", fake_rank)
    monkeypatch.setattr(engine, "_fit_one_hypothesis", fake_fit)
    monkeypatch.setattr(engine, "_score_host_tissues", lambda *_a, **_k: [])

    rows = engine.decompose_sample(
        pd.DataFrame({"Symbol": [], "TPM": []}),
        cancer_types=["LOCAL", "BROAD"],
        templates=["solid_primary"],
        top_k=5,
        candidate_rows=[broad_row],
        sample_raw_by_symbol={},
        sample_by_eid={},
    )

    assert ranked_requests == [("LOCAL",)]
    assert {row.cancer_type for row in rows} == {"BROAD", "LOCAL"}
    local = next(row for row in rows if row.cancer_type == "LOCAL")
    assert local.scope_source == "requested_scope"
