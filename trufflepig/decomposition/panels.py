# Licensed under the Apache License, Version 2.0

"""Tumor-biased / matched-normal-biased / shared-lineage gene panels.

Supports the matched-normal lineage-splitting work in issue #50. For each
epithelial cancer type with a defensible matched-normal tissue, we want
three gene categories:

- **tumor-biased**: genes meaningfully higher in the TCGA cancer cohort
  than in the matched-normal bulk tissue. These carry the signal that a
  three-component decomposition relies on to distinguish tumor from
  benign parent tissue.

- **matched-normal-biased**: genes meaningfully higher in the matched
  normal than in the cancer cohort. Useful for sanity-checking that a
  sample actually contains benign parent tissue and not just stroma.

- **shared-lineage**: genes expressed comparably in both, e.g. retained
  luminal / glandular markers. Explicitly excluded from drive-the-fit
  marker sets — these cannot discriminate tumor from matched normal.

These utilities operate on the bundled ``pan_cancer_expression`` table
(FPKM_<cancer> cancer cohort medians and nTPM_<tissue> bulk normal
references). They are currently provided for downstream calibration /
inspection and are NOT wired into the decomposition marker selection.

That wiring — using tumor-biased genes as anchoring markers so the NNLS
can reliably allocate between tumor and matched_normal compartments — is
the step that will let the matched-normal feature be turned on by
default. See issue #50 step 2–3.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from pirlygenes.gene_sets_cancer import lineage_gene_symbols

from trufflepig.reference import pan_cancer_expression
from ..tumor_purity import TCGA_MEDIAN_PURITY
from .signature import _load_hpa_cell_types, COMPONENT_TO_HPA
from .templates import EPITHELIAL_MATCHED_NORMAL_TISSUE

# Gene families we exclude from tumor-biased / matched-normal-biased
# panels because their TCGA bulk vs HPA normal comparison is dominated
# by technical artifacts rather than biology:
#   - MT-* mitochondrial transcripts (degradation-sensitive; absent or
#     depleted in many TCGA RNA-seq preps, so they falsely look
#     "matched-normal-biased").
#   - RPL* / RPS* ribosomal proteins (normalisation-sensitive).
#   - Rearranged B-cell / T-cell receptor V/D/J/C segments (sample-
#     specific clonal expansion, not a stable bias).
_PANEL_EXCLUDE_REGEXES = [
    re.compile(r"MT-.*"),
    re.compile(r"(RPL|RPS)\d.*"),
    re.compile(r"(IGH|IGK|IGL|TRA|TRB|TRG|TRD)[VDJC]\d.*"),
]


def _is_panel_excluded_symbol(symbol):
    s = str(symbol)
    for pattern in _PANEL_EXCLUDE_REGEXES:
        if pattern.fullmatch(s):
            return True
    return False


def _load_cohort_vs_tissue(cancer_code, tissue=None):
    """Return a DataFrame with per-gene tumor-cell and matched-normal estimates.

    TCGA cohort medians are bulk, not tumor-cell-only (median TCGA purity
    is 0.4–0.8 depending on cancer type). A raw ``FPKM_<cancer> /
    nTPM_<tissue>`` ratio therefore understates the tumor-vs-normal
    difference for genes where benign admixed parent tissue contributes
    meaningfully to the TCGA bulk average — the exact class of genes a
    matched-normal panel cares about (AMACR in PRAD, CDX2 in COAD, ...).

    This loader deconvolves the TCGA cohort median to a crude tumor-cell-
    only estimate using the published TCGA median purity and the matched-
    normal bulk as the non-tumor background:

        tumor_cell ≈ max(0, (FPKM_cancer - (1 - p) * nTPM_tissue) / p)

    The TCGA_MEDIAN_PURITY dict is the same calibration constant used in
    ``estimate_tumor_expression_ranges`` to build the per-gene cohort
    prior, so the two code paths use a consistent definition.

    Returns ``(df, tissue)`` where ``df`` has columns ``symbol``,
    ``tumor_fpkm`` (deconvolved tumor-cell estimate), ``normal_ntpm``,
    and ``tcga_bulk_fpkm`` (the raw cohort median, retained for
    inspection).
    """
    if tissue is None:
        tissue = EPITHELIAL_MATCHED_NORMAL_TISSUE.get(cancer_code)
        if tissue is None:
            raise ValueError(
                f"{cancer_code} has no default matched-normal tissue; "
                f"pass `tissue=` explicitly or extend "
                f"EPITHELIAL_MATCHED_NORMAL_TISSUE."
            )

    # Keep raw bundled references here. Shared-lineage panel construction
    # intentionally compares raw TCGA bulk magnitude to matched-normal bulk
    # magnitude; column-wise technical-RNA renormalization can move canonical
    # retained-lineage markers such as PRAD KLK3 across the tolerance boundary.
    ref = pan_cancer_expression().drop_duplicates(subset="Symbol")
    cancer_col = f"FPKM_{cancer_code}"
    tissue_col = f"nTPM_{tissue}"
    missing = [c for c in (cancer_col, tissue_col) if c not in ref.columns]
    if missing:
        raise KeyError(f"Missing reference columns: {missing}")

    cohort_bulk = ref[cancer_col].astype(float)
    normal_ntpm = ref[tissue_col].astype(float)
    purity = float(TCGA_MEDIAN_PURITY.get(cancer_code, 0.7))
    tumor_cell = np.maximum(
        0.0,
        (cohort_bulk - (1.0 - purity) * normal_ntpm) / purity,
    )

    out = pd.DataFrame(
        {
            "symbol": ref["Symbol"].astype(str),
            "tumor_fpkm": tumor_cell,
            "normal_ntpm": normal_ntpm,
            "tcga_bulk_fpkm": cohort_bulk,
        }
    )
    # Drop gene families where bulk-vs-bulk comparisons are technical-
    # artifact-dominated (see _PANEL_EXCLUDE_REGEXES).
    out = out[~out["symbol"].map(_is_panel_excluded_symbol)].reset_index(drop=True)
    return out, tissue


def _stromal_immune_background_by_symbol():
    """Max HPA single-cell nTPM across generic fibroblast / endothelial /
    broad-immune cell types, per gene symbol.

    Used to reject matched-normal-biased candidates whose elevated
    expression in the tissue bulk is really coming from the stromal or
    immune sub-population of that tissue (e.g. ``TAGLN``/``ACTA2`` in
    prostate are smooth-muscle stromal, not prostate-epithelial). If the
    matched-normal NNLS column has to compete with the generic fibroblast
    NNLS column for these markers, the fit destabilises — the two
    columns are nearly collinear at those rows.
    """
    hpa = _load_hpa_cell_types().drop_duplicates(subset="Symbol").copy()
    hpa_indexed = hpa.set_index(hpa["Symbol"].astype(str))
    stromal_cells = set()
    for comp in (
        "fibroblast",
        "endothelial",
        "T_cell",
        "B_cell",
        "plasma",
        "NK",
        "myeloid",
    ):
        stromal_cells.update(COMPONENT_TO_HPA.get(comp, []))
    present = [c for c in stromal_cells if c in hpa_indexed.columns]
    if not present:
        return {}
    max_series = hpa_indexed[present].astype(float).max(axis=1)
    return max_series.to_dict()


def _log2_ratio(a, b, floor=0.1):
    """Symmetric, floor-stabilized log2 ratio ``log2((a + floor) / (b + floor))``."""
    return np.log2(
        (np.asarray(a, dtype=float) + floor) / (np.asarray(b, dtype=float) + floor)
    )


def build_tumor_biased_panel(
    cancer_code,
    tissue=None,
    delta_log2=1.0,
    min_tumor_expression=1.0,
):
    """Genes meaningfully higher in the TCGA cancer cohort than in the
    matched-normal tissue.

    Parameters
    ----------
    cancer_code : str
        TCGA cancer code (e.g. ``"PRAD"``).
    tissue : str, optional
        Matched-normal tissue (e.g. ``"prostate"``). Defaults to
        :data:`EPITHELIAL_MATCHED_NORMAL_TISSUE`.
    delta_log2 : float
        Minimum log2(tumor / normal) ratio. ``1.0`` ≡ ≥2× tumor-biased.
    min_tumor_expression : float
        Require ``FPKM_<cancer> >= min_tumor_expression`` so we don't
        retain noisy ratios driven purely by the floor term.

    Returns
    -------
    DataFrame with columns ``symbol, tumor_fpkm, normal_ntpm,
    log2_ratio``, sorted by ``log2_ratio`` descending.
    """
    df, _tissue = _load_cohort_vs_tissue(cancer_code, tissue=tissue)
    df["log2_ratio"] = _log2_ratio(df["tumor_fpkm"], df["normal_ntpm"])
    keep = (df["tumor_fpkm"] >= min_tumor_expression) & (df["log2_ratio"] >= delta_log2)
    return df[keep].sort_values("log2_ratio", ascending=False).reset_index(drop=True)


def build_matched_normal_biased_panel(
    cancer_code,
    tissue=None,
    delta_log2=1.0,
    min_normal_expression=1.0,
    stromal_collinearity_ratio=0.5,
):
    """Genes meaningfully higher in the matched-normal tissue than in
    the TCGA cancer cohort **and** not collinear with generic stromal /
    immune cell-type references.

    The matched-normal column in the decomposition NNLS sits next to
    generic fibroblast / endothelial / broad-immune columns. Markers
    that light up in both the matched-normal tissue (because the tissue
    bulk contains stromal cells) and in the generic stromal references
    are effectively collinear — using them would destabilise the NNLS
    allocation. ``stromal_collinearity_ratio`` drops any candidate where
    ``max(stromal HPA nTPM) >= ratio * normal_ntpm``, defaulting to 0.5
    so a gene must be at least 2× more expressed in the tissue bulk than
    in any stromal cell-type reference.

    Parameters
    ----------
    delta_log2 : float
        Minimum log2(normal / tumor_cell) ratio. ``1.0`` ≡ ≥2× normal-biased.
    min_normal_expression : float
        Require ``nTPM_<tissue> >= min_normal_expression``.
    stromal_collinearity_ratio : float
        Reject genes where the max across HPA fibroblast / endothelial
        / broad-immune references exceeds this fraction of the matched-
        normal reference expression. Set to ``None`` or ``0`` to skip.
    """
    df, _tissue = _load_cohort_vs_tissue(cancer_code, tissue=tissue)
    df["log2_ratio"] = _log2_ratio(df["normal_ntpm"], df["tumor_fpkm"])
    keep = (df["normal_ntpm"] >= min_normal_expression) & (
        df["log2_ratio"] >= delta_log2
    )
    df = df[keep].copy()
    if stromal_collinearity_ratio:
        background = _stromal_immune_background_by_symbol()
        df["stromal_bg"] = df["symbol"].map(background).fillna(0.0).astype(float)
        df = df[df["stromal_bg"] < df["normal_ntpm"] * stromal_collinearity_ratio].drop(
            columns=["stromal_bg"]
        )
    return df.sort_values("log2_ratio", ascending=False).reset_index(drop=True)


def build_shared_lineage_panel(
    cancer_code,
    tissue=None,
    tolerance_log2=1.0,
    curated_lineage_tolerance_log2=2.0,
    min_expression=5.0,
):
    """Genes expressed at similar levels in tumor-cohort bulk and matched-normal bulk.

    Useful to flag retained-lineage markers (e.g. KLK3 in PRAD, SFTPB in
    LUAD) that cannot distinguish tumor from benign parent tissue.

    Note: the comparison uses **raw bulk** (``tcga_bulk_fpkm`` vs
    ``nTPM_tissue``), not the deconvolved tumor-cell estimate. A naive
    TCGA deconvolution subtracts the full normal contribution, which by
    construction pulls lineage-shared genes toward zero and would remove
    them from a "shared" panel — precisely the wrong answer. Raw bulk
    comparison captures the intended semantic: both cohorts express the
    gene at comparable magnitudes.

    Parameters
    ----------
    tolerance_log2 : float
        Max abs log2 ratio to count as "shared". ``1.0`` ≡ within ~2x.
    curated_lineage_tolerance_log2 : float
        Wider abs log2 ratio allowed for curated lineage genes already marked
        as load-bearing biology for this cancer type. This prevents a generic
        cliff-edge threshold from dropping retained-lineage markers such as
        PRAD ``KLK3`` when both tumor and matched normal express the gene.
    min_expression : float
        Require both cohort bulk and normal reference to exceed this
        floor so the ratio is meaningful.

    Returns
    -------
    DataFrame with columns ``symbol, tumor_fpkm, normal_ntpm,
    tcga_bulk_fpkm, abs_log2_ratio``.
    """
    df, _tissue = _load_cohort_vs_tissue(cancer_code, tissue=tissue)
    df["abs_log2_ratio"] = np.abs(_log2_ratio(df["tcga_bulk_fpkm"], df["normal_ntpm"]))
    ratio_keep = (
        (df["tcga_bulk_fpkm"] >= min_expression)
        & (df["normal_ntpm"] >= min_expression)
        & (df["abs_log2_ratio"] <= tolerance_log2)
    )
    curated_genes = set(lineage_gene_symbols(cancer_code))
    curated_keep = (
        df["symbol"].isin(curated_genes)
        & (df["tcga_bulk_fpkm"] >= min_expression)
        & (df["normal_ntpm"] >= min_expression)
        & (df["abs_log2_ratio"] <= curated_lineage_tolerance_log2)
    )
    out = df[ratio_keep | curated_keep].copy()
    out["shared_lineage_basis"] = np.where(
        ratio_keep.loc[out.index],
        "ratio",
        "curated_lineage",
    )
    return out.sort_values("abs_log2_ratio").reset_index(drop=True)


def estimate_lineage_tumor_fraction(
    sample_tpm_by_symbol,
    cancer_code,
    tissue=None,
    panel_size=30,
    delta_log2=1.0,
    min_tumor_expression=1.0,
):
    """Estimate tumor purity from per-gene reconciliation of the sample's
    tumor-biased-panel signal against the TCGA tumor-cell reference and
    matched-normal tissue reference.

    For each gene ``g`` in the tumor-biased panel we assume the sample's
    observed expression is a two-component mix:

        sample_g ≈ f_tumor · tumor_cell_g + (1 − f_tumor) · normal_ref_g

    Solving:

        f_tumor ≈ (sample_g − normal_ref_g) / (tumor_cell_g − normal_ref_g)

    ``tumor_cell_g`` comes from deconvolving the TCGA cohort median
    against ``TCGA_MEDIAN_PURITY`` (same definition as in
    ``estimate_tumor_expression_ranges``). ``normal_ref_g`` is
    ``nTPM_<tissue>`` for the matched-normal tissue.

    The robust summary (winsorized median + IQR stability) gives a
    lineage-specific purity estimate that does not rely on the
    signature-gene / ESTIMATE machinery. It is the purity signal used by
    ``decompose_sample`` as the ``tumor_fraction`` prior when a
    ``matched_normal_<tissue>`` compartment is active.

    Parameters
    ----------
    sample_tpm_by_symbol : mapping
        Sample expression, keyed by gene symbol, values in TPM.
    cancer_code : str
        TCGA cancer code.
    tissue : str, optional
        Matched-normal tissue. Defaults to
        :data:`EPITHELIAL_MATCHED_NORMAL_TISSUE`.
    panel_size : int
        Maximum number of top tumor-biased genes to use.
    delta_log2, min_tumor_expression : float
        Forwarded to :func:`build_tumor_biased_panel`.

    Returns
    -------
    dict or None
        ``{"estimate", "lower", "upper", "stability", "panel_size",
        "panel_genes_observed", "per_gene"}`` or ``None`` if the cancer
        type has no matched-normal mapping or the panel cannot be
        evaluated on the sample.
    """
    try:
        panel = build_tumor_biased_panel(
            cancer_code,
            tissue=tissue,
            delta_log2=delta_log2,
            min_tumor_expression=min_tumor_expression,
        )
    except (KeyError, ValueError):
        return None
    if panel.empty:
        return None
    panel = panel.head(panel_size)

    per_gene = []
    observed_in_sample = 0
    for row in panel.itertuples(index=False):
        symbol = str(row.symbol)
        sample_val = float(sample_tpm_by_symbol.get(symbol, 0.0))
        tumor_val = float(row.tumor_fpkm)
        normal_val = float(row.normal_ntpm)
        denom = tumor_val - normal_val
        if denom <= 0:
            # Safety: build_tumor_biased_panel should already guarantee
            # tumor > normal. Skip if somehow not.
            continue
        if sample_val > 0.1:
            observed_in_sample += 1
        f = (sample_val - normal_val) / denom
        f = float(np.clip(f, 0.0, 1.0))
        per_gene.append(
            {
                "gene": symbol,
                "sample_tpm": sample_val,
                "tumor_cell_tpm": tumor_val,
                "normal_tpm": normal_val,
                "fraction": f,
            }
        )

    # Require at least 5 panel genes to even attempt, and at least 5 of
    # them to have non-trivial sample expression. The second check
    # guards against the pathological case where the sample is so
    # subsampled / incomplete that most panel genes are missing — the
    # estimator would then confidently report 0.0 from a sea of zeros,
    # which is wrong (absence of evidence != evidence of absence).
    if len(per_gene) < 5 or observed_in_sample < 5:
        return None

    fractions = np.array([g["fraction"] for g in per_gene], dtype=float)
    # Winsorized median: drop top / bottom 10% before summarizing.
    # Stability is reported as (IQR / (|median| + 0.05)). Small values
    # mean per-gene agreement; a large IQR relative to the median
    # indicates the panel disagrees about what fraction best explains
    # the sample — treat those results as advisory.
    lo_pct, hi_pct = np.percentile(fractions, [10, 90])
    trimmed = fractions[(fractions >= lo_pct) & (fractions <= hi_pct)]
    if len(trimmed) < 3:
        trimmed = fractions
    median = float(np.median(trimmed))
    q1, q3 = np.percentile(trimmed, [25, 75])
    iqr = float(q3 - q1)
    stability = iqr / (abs(median) + 0.05)
    return {
        "estimate": median,
        "lower": float(q1),
        "upper": float(q3),
        "stability": stability,
        "panel_size": int(len(panel)),
        "panel_genes_observed": int(len(per_gene)),
        "per_gene": per_gene,
    }


def summarize_panels(cancer_code, tissue=None):
    """Convenience snapshot of panel sizes for a cancer type.

    Returns a dict with counts of tumor-biased, matched-normal-biased,
    and shared-lineage genes at the default thresholds. Useful for
    eyeballing whether a cancer type has enough discriminative signal
    to drive a three-component decomposition.
    """
    return {
        "cancer_code": cancer_code,
        "tissue": tissue or EPITHELIAL_MATCHED_NORMAL_TISSUE.get(cancer_code),
        "tumor_biased": len(build_tumor_biased_panel(cancer_code, tissue=tissue)),
        "matched_normal_biased": len(
            build_matched_normal_biased_panel(cancer_code, tissue=tissue)
        ),
        "shared_lineage": len(build_shared_lineage_panel(cancer_code, tissue=tissue)),
    }
