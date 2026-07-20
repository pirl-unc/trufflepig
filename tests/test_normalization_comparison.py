"""Rigorous comparison of normalization × weighting strategies for NNLS decomposition.

Tests 12 combinations (4 normalizations × 3 weightings) across 6 samples with
known or strongly expected composition.  The goal is to find the strategy that
minimises proportional reconstruction error AND attribution leakage (e.g. Ig
genes misattributed to tumor because the NNLS residual on high-expression
genes dominates the objective).

Normalization options (applied to both A and b):
  hk         – divide each vector by its housekeeping-gene median (current)
  zscore     – per-gene z-score across the K reference components
  hk_zscore  – HK-normalise first, then z-score across components
  raw        – no normalisation (TPM as-is, marker selection only)

Weighting options (per-row weight in the NNLS system):
  uniform    – only the marker-specificity weight (no expression adjustment)
  inv_sqrt   – marker_weight / sqrt(max(b, 0.1))  (previous default)
  inv_b      – marker_weight / max(b, 0.1)         (current default)
"""

import numpy as np
import pandas as pd

from trufflepig.decomposition.engine import (
    _hk_normalize,
    _select_marker_rows,
    _weighted_constrained_nnls,
)
from trufflepig.decomposition.signature import (
    build_signature_matrix,
    _load_hpa_cell_types,
)
from trufflepig.decomposition.templates import get_template_components
from pirlygenes.gene_sets_cancer import housekeeping_gene_ids
from trufflepig.reference import pan_cancer_expression
from trufflepig.tumor_purity import estimate_tumor_purity


# ── Test fixtures ────────────────────────────────────────────────────────


def _tcga_sample(cancer_code):
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{cancer_code}_TPM"].astype(float),
        }
    )


def _normal_tissue_sample(tissue):
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{tissue}_nTPM"].astype(float),
        }
    )


def _hpa_cell_sample(cell_type):
    hpa = _load_hpa_cell_types()
    return pd.DataFrame(
        {
            "ensembl_gene_id": hpa["Ensembl_Gene_ID"],
            "gene_symbol": hpa["Symbol"],
            "TPM": hpa[cell_type].astype(float),
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


# ── Core fitting with configurable normalization/weighting ───────────────

NORMALIZATIONS = ["hk", "zscore", "hk_zscore", "raw"]
WEIGHTINGS = ["uniform", "inv_sqrt", "inv_b"]
IG_GENES = {
    "IGKC",
    "IGLC2",
    "IGHG1",
    "IGHG2",
    "IGHG3",
    "IGHG4",
    "IGHA1",
    "IGHA2",
    "IGLC1",
    "IGLC3",
    "IGLL5",
    "JCHAIN",
    "MZB1",
}


def _apply_normalization(sample_vec, sig_raw, genes, normalization):
    """Return (normalized_sample, normalized_sig, hk_median)."""
    hk_ids = housekeeping_gene_ids()

    if normalization == "raw":
        return sample_vec.copy(), sig_raw.copy(), 1.0

    if normalization == "hk":
        obs_hk, hk_med = _hk_normalize(sample_vec, genes, hk_ids)
        sig_hk = np.zeros_like(sig_raw, dtype=float)
        for k in range(sig_raw.shape[1]):
            sig_hk[:, k], _ = _hk_normalize(sig_raw[:, k], genes, hk_ids)
        return obs_hk, sig_hk, hk_med

    if normalization == "zscore":
        # Z-score each gene across the K reference components.
        # Genes with no variation across components are uninformative and
        # get a fallback weight of 0 (effectively filtered out by the
        # marker selection step).
        mu = sig_raw.mean(axis=1)
        sigma = sig_raw.std(axis=1)
        sigma = np.where(sigma > 1e-6, sigma, 1.0)  # floor to avoid div/0
        sig_z = (sig_raw - mu[:, None]) / sigma[:, None]
        obs_z = (sample_vec - mu) / sigma
        return obs_z, sig_z, 1.0

    if normalization == "hk_zscore":
        # Step 1: HK-normalise
        obs_hk, hk_med = _hk_normalize(sample_vec, genes, hk_ids)
        sig_hk = np.zeros_like(sig_raw, dtype=float)
        for k in range(sig_raw.shape[1]):
            sig_hk[:, k], _ = _hk_normalize(sig_raw[:, k], genes, hk_ids)
        # Step 2: Z-score across components on HK-normalised values
        mu = sig_hk.mean(axis=1)
        sigma = sig_hk.std(axis=1)
        sigma = np.where(sigma > 1e-6, sigma, 1.0)
        sig_z = (sig_hk - mu[:, None]) / sigma[:, None]
        obs_z = (obs_hk - mu) / sigma
        return obs_z, sig_z, hk_med

    raise ValueError(f"Unknown normalization: {normalization}")


def _apply_weighting(fit_weights, b, weighting):
    """Return combined weights for the NNLS rows."""
    if weighting == "uniform":
        return fit_weights.copy()
    if weighting == "inv_sqrt":
        return fit_weights / np.sqrt(np.maximum(np.abs(b), 0.1))
    if weighting == "inv_b":
        return fit_weights / np.maximum(np.abs(b), 0.1)
    raise ValueError(f"Unknown weighting: {weighting}")


def _fit_combo(df_sample, template_name, cancer_type, purity, normalization, weighting):
    """Run one normalization × weighting combo and return metrics.

    Returns dict with:
      fractions     – component fractions (including tumor)
      residual      – NNLS residual (on normalised scale)
      ig_tumor_tpm  – total Ig-gene TPM attributed to tumor
      ig_tme_tpm    – total Ig-gene TPM attributed to TME
      ig_obs_tpm    – total observed Ig-gene TPM
      prop_error    – median |predicted - observed| / observed over markers
    """
    hk_ids = housekeeping_gene_ids()
    components = get_template_components(template_name, cancer_type)
    comp_names = [c for c in components if c != "tumor"]
    tumor_fraction = float(purity)

    if not comp_names or tumor_fraction >= 0.999:
        return {
            "fractions": {"tumor": 1.0},
            "residual": 0.0,
            "ig_tumor_tpm": 0.0,
            "ig_tme_tpm": 0.0,
            "ig_obs_tpm": 0.0,
            "prop_error": 0.0,
        }

    # Build sample and signature
    sample_by_eid = dict(
        zip(
            df_sample["ensembl_gene_id"].astype(str),
            df_sample["TPM"].astype(float),
        )
    )
    gene_subset = set(sample_by_eid.keys())
    filt_genes, filt_symbols, sig_raw, _ = build_signature_matrix(
        comp_names, gene_subset=gene_subset
    )
    filt_sample_vec = np.array(
        [sample_by_eid.get(gid, 0.0) for gid in filt_genes], dtype=float
    )

    # Normalise
    obs_norm, sig_norm, hk_med = _apply_normalization(
        filt_sample_vec, sig_raw, filt_genes, normalization
    )

    # Select markers (always on HK space for consistency of marker selection)
    obs_hk_for_markers, _ = _hk_normalize(filt_sample_vec, filt_genes, hk_ids)
    sig_hk_for_markers = np.zeros_like(sig_raw, dtype=float)
    for k in range(sig_raw.shape[1]):
        sig_hk_for_markers[:, k], _ = _hk_normalize(sig_raw[:, k], filt_genes, hk_ids)
    fit_rows, fit_weights, _ = _select_marker_rows(
        filt_genes, filt_symbols, sig_hk_for_markers, comp_names
    )
    if not fit_rows:
        fit_rows = list(range(len(filt_genes)))
        fit_weights = np.ones(len(fit_rows), dtype=float)

    A = sig_norm[fit_rows]
    b = obs_norm[fit_rows]
    combined_weights = _apply_weighting(fit_weights, b, weighting)

    # For z-score normalizations, b can be negative.
    # NNLS requires non-negative target — handle by shifting.
    # But actually NNLS with negative b values still works in scipy (it
    # just finds the best non-negative x that minimises ||Ax - b||^2 even
    # when b has negative entries).  The solution can still be valid.
    comp_mix, residual = _weighted_constrained_nnls(A, b, weights=combined_weights)

    # Build fractions
    fractions = {"tumor": tumor_fraction}
    for idx, comp in enumerate(comp_names):
        fractions[comp] = float(comp_mix[idx] * max(0.0, 1.0 - tumor_fraction))

    # Compute proportional error on markers (in original TPM space)
    # Use the raw signature and sample for this, not the normalised versions,
    # since we want to measure how well the decomposition explains the actual
    # observed expression.
    A_raw = sig_raw[fit_rows]
    b_raw = filt_sample_vec[fit_rows]
    predicted_raw = A_raw @ comp_mix
    mask = b_raw > 0.1  # only genes with meaningful expression
    if mask.sum() > 0:
        prop_errors = np.abs(predicted_raw[mask] - b_raw[mask]) / b_raw[mask]
        prop_error = float(np.median(prop_errors))
    else:
        prop_error = float("nan")

    # Ig attribution: how much Ig TPM leaks into tumor?
    ig_indices = [i for i, s in enumerate(filt_symbols) if str(s) in IG_GENES]
    ig_obs_tpm = 0.0
    ig_tme_tpm = 0.0
    for idx in ig_indices:
        obs_tpm = float(filt_sample_vec[idx])
        ig_obs_tpm += obs_tpm
        tme_attr = 0.0
        for comp_idx, comp in enumerate(comp_names):
            tme_attr += (
                (1.0 - tumor_fraction)
                * float(comp_mix[comp_idx])
                * float(sig_raw[idx, comp_idx])
            )
        ig_tme_tpm += min(tme_attr, obs_tpm)  # cap at observed
    ig_tumor_tpm = max(0.0, ig_obs_tpm - ig_tme_tpm)

    return {
        "fractions": fractions,
        "residual": round(residual, 4),
        "ig_tumor_tpm": round(ig_tumor_tpm, 1),
        "ig_tme_tpm": round(ig_tme_tpm, 1),
        "ig_obs_tpm": round(ig_obs_tpm, 1),
        "prop_error": round(prop_error, 4),
    }


# ── Test samples ─────────────────────────────────────────────────────────

SAMPLES = {}


def _get_samples():
    """Lazily build test samples (expensive, cached)."""
    if SAMPLES:
        return SAMPLES

    # 1. TCGA COAD — expect ~65% purity, immune/stroma mix
    SAMPLES["TCGA_COAD"] = {
        "df": _tcga_sample("COAD"),
        "template": "solid_primary",
        "cancer_type": "COAD",
        "expected_purity_range": (0.45, 0.85),
        "description": "TCGA COAD median (solid primary)",
    }

    # 2. Synthetic 30% CRC + 70% colon — known purity, primary site
    SAMPLES["CRC_colon_30"] = {
        "df": _mix_samples(
            [
                (0.3, _tcga_sample("COAD")),
                (0.7, _normal_tissue_sample("colon")),
            ]
        ),
        "template": "solid_primary",
        "cancer_type": "COAD",
        "expected_purity_range": (0.15, 0.45),
        "description": "30% CRC + 70% colon normal",
    }

    # 3. Synthetic 30% CRC + 70% liver — should detect hepatocyte
    SAMPLES["CRC_liver_30"] = {
        "df": _mix_samples(
            [
                (0.3, _tcga_sample("COAD")),
                (0.7, _normal_tissue_sample("liver")),
            ]
        ),
        "template": "met_liver",
        "cancer_type": "COAD",
        "expected_purity_range": (0.15, 0.45),
        "description": "30% CRC + 70% liver (met_liver template)",
    }

    # 4. Pure T-cell — expect 0% tumor, >75% T_cell
    SAMPLES["pure_T_cell"] = {
        "df": _hpa_cell_sample("T-cells"),
        "template": "solid_primary",
        "cancer_type": "THYM",
        "expected_purity_range": (-0.01, 0.01),  # purity_override=0
        "purity_override": 0.0,
        "description": "Pure HPA T-cell profile (purity=0)",
    }

    # 5. Plasma-heavy sample — the immunoglobulin stress test
    #    20% COAD (non-Ig-producing epithelial tumor) + 80% plasma cells.
    #    ALL Ig expression comes from the plasma TME component, so any Ig
    #    attributed to "tumor" is pure leakage from the decomposition.
    SAMPLES["plasma_heavy"] = {
        "df": _mix_samples(
            [
                (0.2, _tcga_sample("COAD")),
                (0.8, _hpa_cell_sample("Plasma cells")),
            ]
        ),
        "template": "solid_primary",
        "cancer_type": "COAD",
        "expected_purity_range": (0.05, 0.35),
        "description": "20% COAD + 80% plasma cells (Ig leakage test)",
    }

    # 6. Brain met — 25% LUAD + 75% brain (astrocyte/neuron)
    SAMPLES["brain_met"] = {
        "df": _mix_samples(
            [
                (0.25, _tcga_sample("LUAD")),
                (0.75, _normal_tissue_sample("cerebral_cortex")),
            ]
        ),
        "template": "met_brain",
        "cancer_type": "LUAD",
        "expected_purity_range": (0.10, 0.40),
        "description": "25% LUAD + 75% brain (met_brain template)",
    }

    return SAMPLES


# ── Main comparison ──────────────────────────────────────────────────────


def run_comparison():
    """Run all normalization × weighting combos across all samples."""
    samples = _get_samples()
    rows = []

    for sample_name, spec in samples.items():
        df = spec["df"]
        template = spec["template"]
        cancer_type = spec["cancer_type"]

        # Get purity estimate (or use override)
        if "purity_override" in spec:
            purity = spec["purity_override"]
        else:
            purity_result = estimate_tumor_purity(df, cancer_type=cancer_type)
            purity = purity_result["overall_estimate"]

        for norm in NORMALIZATIONS:
            for wt in WEIGHTINGS:
                try:
                    result = _fit_combo(df, template, cancer_type, purity, norm, wt)
                    rows.append(
                        {
                            "sample": sample_name,
                            "normalization": norm,
                            "weighting": wt,
                            "purity": round(purity, 3),
                            **{
                                f"f_{k}": round(v, 4)
                                for k, v in result["fractions"].items()
                                if k != "tumor" and v > 0.005
                            },
                            "residual": result["residual"],
                            "prop_error": result["prop_error"],
                            "ig_obs_tpm": result["ig_obs_tpm"],
                            "ig_tumor_tpm": result["ig_tumor_tpm"],
                            "ig_tme_tpm": result["ig_tme_tpm"],
                        }
                    )
                except Exception as e:
                    rows.append(
                        {
                            "sample": sample_name,
                            "normalization": norm,
                            "weighting": wt,
                            "error": str(e)[:80],
                        }
                    )

    return pd.DataFrame(rows)


# ── Pytest entry points ──────────────────────────────────────────────────


def test_comparison_runs_without_errors():
    """All 72 normalization × weighting × sample combos should run."""
    df = run_comparison()
    if "error" in df.columns:
        errors = df[df["error"].notna()]
        if not errors.empty:
            print("\n=== ERRORS ===")
            print(errors.to_string(index=False))
        assert errors.empty, f"{len(errors)} combos failed"


def print_comparison_table_for_manual_review():
    """Print the full comparison table for human review."""
    df = run_comparison()

    # Print per-sample summary tables
    for sample_name in df["sample"].unique():
        sub = df[df["sample"] == sample_name].copy()
        desc = _get_samples()[sample_name]["description"]
        print(f"\n{'=' * 80}")
        print(f"  {sample_name}: {desc}")
        print(f"{'=' * 80}")

        # Show key metrics
        show_cols = [
            "normalization",
            "weighting",
            "residual",
            "prop_error",
            "ig_obs_tpm",
            "ig_tumor_tpm",
        ]
        # Add any fraction columns
        frac_cols = [c for c in sub.columns if c.startswith("f_")]
        show_cols.extend(sorted(frac_cols))
        show_cols = [c for c in show_cols if c in sub.columns]
        print(sub[show_cols].to_string(index=False))

    # Print ranking: which combo is best overall?
    print(f"\n{'=' * 80}")
    print("  AGGREGATE RANKING (lower is better)")
    print(f"{'=' * 80}")

    # For each combo, compute mean rank across samples on:
    # (1) prop_error  (2) ig_tumor_tpm
    combo_scores = []
    for norm in NORMALIZATIONS:
        for wt in WEIGHTINGS:
            mask = (df["normalization"] == norm) & (df["weighting"] == wt)
            sub = df[mask]
            if sub.empty or "error" in sub.columns and sub["error"].notna().any():
                continue

            # Mean proportional error across samples
            mean_prop = (
                sub["prop_error"].mean()
                if "prop_error" in sub.columns
                else float("nan")
            )
            # Mean Ig leakage into tumor
            mean_ig_leak = (
                sub["ig_tumor_tpm"].mean()
                if "ig_tumor_tpm" in sub.columns
                else float("nan")
            )
            # Mean residual
            mean_resid = (
                sub["residual"].mean() if "residual" in sub.columns else float("nan")
            )

            combo_scores.append(
                {
                    "normalization": norm,
                    "weighting": wt,
                    "mean_prop_error": round(mean_prop, 4),
                    "mean_ig_tumor_tpm": round(mean_ig_leak, 1),
                    "mean_residual": round(mean_resid, 4),
                }
            )

    ranking = pd.DataFrame(combo_scores).sort_values("mean_prop_error")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    print_comparison_table_for_manual_review()
