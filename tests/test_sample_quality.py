import numpy as np
import pandas as pd

from trufflepig.sample_quality import (
    assess_sample_quality,
    _MT_GENES,
    _CULTURE_STRESS_UP,
    _TME_MARKERS,
)
from trufflepig.reference import pan_cancer_expression


def _tcga_sample(cancer_code):
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{cancer_code}_TPM_raw"].astype(float),
        }
    )


def _normal_tissue_sample(tissue):
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{tissue}_nTPM_raw"].astype(float),
        }
    )


def test_tcga_coad_has_no_quality_issues():
    """TCGA COAD median should not flag degradation or culture."""
    df = _tcga_sample("COAD")
    # Provide tissue match so tissue-matched baselines are used
    result = assess_sample_quality(df, tissue_scores=[("colon", 0.9, 20)])

    assert result["degradation"]["level"] == "normal"
    assert result["degradation"]["matched_tissue"] == "colon"
    assert result["culture"]["level"] == "normal"
    assert not result["has_issues"]


def test_tcga_brca_has_no_quality_issues():
    """TCGA BRCA median should also be clean."""
    df = _tcga_sample("BRCA")
    result = assess_sample_quality(df, tissue_scores=[("breast", 0.85, 20)])

    assert result["degradation"]["level"] == "normal"
    assert not result["has_issues"]


def test_all_tcga_types_pass_quality_with_tissue_match():
    """No TCGA cohort median should flag as degraded when tissue-matched."""
    from trufflepig.tumor_purity import CANCER_TO_TISSUE

    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    cohort_cols = [c for c in ref.columns if c.endswith("_TPM")]

    for col in cohort_cols:
        code = col.removesuffix("_TPM")
        df = pd.DataFrame(
            {
                "ensembl_gene_id": ref["Ensembl_Gene_ID"],
                "gene_symbol": ref["Symbol"],
                "TPM": ref[f"{code}_TPM_raw"].astype(float),
            }
        )
        tissue = CANCER_TO_TISSUE.get(code)
        tissue_scores = [(tissue, 0.9, 20)] if tissue else None
        result = assess_sample_quality(df, tissue_scores=tissue_scores)
        assert result["degradation"]["level"] == "normal", (
            f"{code} (tissue={tissue}): unexpected degradation={result['degradation']['level']} "
            f"MT fold={result['degradation'].get('mt_fold')}, "
            f"RP fold={result['degradation'].get('rp_fold')}"
        )


def test_synthetic_degraded_sample_flags_degradation():
    """Simulate FFPE-like degradation: inflate MT and RP genes."""
    df = _tcga_sample("COAD").copy()
    tpm = dict(zip(df["gene_symbol"], df["TPM"]))

    # Inflate mitochondrial genes 50× to simulate severe degradation
    for gene in _MT_GENES:
        if gene in tpm:
            tpm[gene] *= 50.0

    # Inflate ribosomal protein genes 10× (short transcripts survive)
    for gene in list(tpm):
        if gene.startswith("RPL") or gene.startswith("RPS"):
            tpm[gene] *= 10.0

    df["TPM"] = df["gene_symbol"].map(tpm).fillna(0.0)
    result = assess_sample_quality(df, tissue_scores=[("colon", 0.9, 20)])

    assert result["degradation"]["level"] in ("moderate", "severe"), (
        f"Expected moderate/severe, got {result['degradation']['level']}. "
        f"MT fold={result['degradation'].get('mt_fold')}, "
        f"RP fold={result['degradation'].get('rp_fold')}"
    )
    assert result["has_issues"]
    assert result["degradation"]["mt_fold"] > 2.0


def test_synthetic_cell_line_flags_culture():
    """Simulate cell line: remove TME markers, inflate stress genes."""
    df = _tcga_sample("COAD").copy()
    tpm = dict(zip(df["gene_symbol"], df["TPM"]))

    # Zero out all TME markers
    for gene in _TME_MARKERS:
        tpm[gene] = 0.0

    # Inflate culture stress genes 50×
    for gene in _CULTURE_STRESS_UP:
        if gene in tpm:
            tpm[gene] *= 50.0

    df["TPM"] = df["gene_symbol"].map(tpm).fillna(0.0)
    result = assess_sample_quality(df)

    assert result["culture"]["tme_absent"]
    assert result["culture"]["stress_score"] > 2.0
    assert result["culture"]["level"] in ("likely_cell_line", "possible_cell_line")
    assert result["has_issues"]


def test_quality_result_structure():
    """Result dict should have all expected keys."""
    df = _tcga_sample("PRAD")
    result = assess_sample_quality(df)

    assert "degradation" in result
    assert "culture" in result
    assert "flags" in result
    assert "has_issues" in result

    deg = result["degradation"]
    assert "mt_fraction" in deg
    assert "rp_fraction" in deg
    assert "level" in deg
    assert "message" in deg

    cul = result["culture"]
    assert "stress_score" in cul
    assert "tme_mean_tpm" in cul
    assert "tme_absent" in cul
    assert "top_stress_genes" in cul
    assert "level" in cul
    assert "message" in cul


def test_normal_tissue_is_not_flagged_as_culture():
    """Normal colon tissue should not be flagged as cell line."""
    df = _normal_tissue_sample("colon")
    result = assess_sample_quality(df)

    assert result["culture"]["level"] in ("normal", "stress_only")
    assert not result["culture"]["tme_absent"]


def test_exome_capture_library_prep_mutes_mt_warning():
    """#77: when library_prep=exome_capture and MT is near-zero, the MT
    fraction is expected to be ~0 and there's still a valid length-pair
    signal, so the "Suspicious MT fraction" flag must not fire and the
    degradation level must not be overwritten to "unknown"."""
    df = _tcga_sample("PRAD").copy()
    tpm = dict(zip(df["gene_symbol"], df["TPM"]))
    for gene in _MT_GENES:
        tpm[gene] = 0.0
    df["TPM"] = df["gene_symbol"].map(tpm).fillna(0.0)

    result = assess_sample_quality(
        df,
        tissue_scores=[("prostate", 0.9, 20)],
        library_prep="exome_capture",
    )
    assert not any("Suspicious MT fraction" in f for f in result["flags"]), (
        f"MT warning should be muted under exome_capture; got {result['flags']!r}"
    )
    assert result["degradation"]["level"] != "unknown", (
        "deg level was clobbered to 'unknown' despite valid length-pair signal"
    )
    assert any(
        "consistent with rna hybrid-capture" in f.lower() for f in result["flags"]
    ), f"Expected informational MT line; got {result['flags']!r}"


def test_quality_baseline_prefers_plausible_host_background_tissue():
    """QC baseline selection should prefer plausible host-background tissues
    over CTA-like tissues when both appear in the background ranking."""
    df = _tcga_sample("COAD")
    result = assess_sample_quality(
        df,
        tissue_scores=[("testis", 0.95, 20), ("thymus", 0.80, 20)],
        library_prep="ribo_depleted",
    )
    assert result["degradation"]["matched_tissue"] == "thymus"


def test_unknown_library_prep_still_warns_on_missing_mt():
    """Without a prep call, MT near zero should still flag as suspicious
    and the degradation level should still fall back to "unknown"."""
    df = _tcga_sample("PRAD").copy()
    tpm = dict(zip(df["gene_symbol"], df["TPM"]))
    for gene in _MT_GENES:
        tpm[gene] = 0.0
    df["TPM"] = df["gene_symbol"].map(tpm).fillna(0.0)

    result = assess_sample_quality(
        df,
        tissue_scores=[("prostate", 0.9, 20)],
        library_prep=None,
    )
    assert any("Suspicious MT fraction" in f for f in result["flags"])
    assert result["degradation"]["level"] == "unknown"


def test_nan_tpm_values_do_not_cause_errors():
    """Genes with NaN TPM should be handled gracefully."""
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    df = pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref["COAD_TPM_raw"].astype(float),  # has NaN values
        }
    )
    result = assess_sample_quality(df)

    # Should not have NaN in any numeric field
    assert not np.isnan(result["degradation"]["mt_fraction"])
    assert not np.isnan(result["degradation"]["rp_fraction"])
    assert not np.isnan(result["culture"]["stress_score"])
