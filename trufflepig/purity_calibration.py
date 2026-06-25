"""Calibrate the lineage-routed decomposition's aneuploidy signal into an absolute purity (#96).

Bulk aneuploidy amplitude is purity-scaled: ``A_obs = A_ref(type) × purity``. ``A_ref`` (the
pure-tumor amplitude) is **type-characteristic** — near-diploid THCA vs CIN-high COAD — so it is
built per cancer type from that type's reference expression and its median purity::

    A_ref(type) = bulk_amplitude(type_reference) / TCGA_median_purity(type)

i.e. the cohort-median reference sits at ~median purity, so dividing extrapolates it to purity = 1.
Then::

    aneuploidy_purity(sample) = clip(bulk_amplitude(sample) / A_ref(type), 0, 1)

**Approximate, by design.** ``A_ref(type)`` is a *population prior* (within-type aneuploidy varies:
MSI/quiet vs CIN), and a true *per-sample* absolute CNA purity needs copy-number step-spacing
(ABSOLUTE/FACETS, allele/segment-level DNA). So this is a corroborator / interval, not a point fit.
The follow-up that upgrades it: regress ``A_obs`` against per-sample TCGA **ABSOLUTE** purity
(slope = ``A_ref``) and ship the result as a versioned offline table — see #96. This module is the
on-the-fly prototype of that table.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def _type_reference_sample(cancer_type):
    """``{symbol: clean-TPM}`` mean profile for a cancer type's reference cohort, or None."""
    try:
        from pirlygenes.expression.accessors import representative_cohort_samples
        from oncoref.normalization import clean_tpm
        import pandas as pd

        d = representative_cohort_samples(cancer_type).drop_duplicates("Ensembl_Gene_ID")
        cols = [c for c in d.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
        gt = pd.DataFrame({"Ensembl_Gene_ID": d["Ensembl_Gene_ID"].values, "Symbol": d["Symbol"].values})
        clean = clean_tpm(d.set_index("Ensembl_Gene_ID")[cols].astype(float), gene_table=gt.set_index(d.index))
        clean.index = d["Symbol"].values
        return clean.groupby(level=0).sum().mean(axis=1).to_dict()
    except Exception:  # noqa: BLE001 — reference cohort unavailable
        return None


@lru_cache(maxsize=None)
def aneuploidy_reference(cancer_type, median_purity=None):
    """``A_ref(type)`` — the pure-tumor (purity≈1) bulk aneuploidy amplitude for a cancer type.

    ``median_purity`` (the reference cohort's typical purity, e.g. TCGA ABSOLUTE median) is passed
    IN so this module stays decoupled from ``tumor_purity`` — the reference sits at ~median purity,
    so we divide to extrapolate to purity = 1; ``None`` ⇒ treat the reference as ~pure. Returns
    None when the type has no reference cohort or is near-diploid (no usable purity signal).
    """
    ref = _type_reference_sample(cancer_type)
    if ref is None:
        return None
    from .expression_decomposition import bulk_aneuploidy_amplitude

    amp = bulk_aneuploidy_amplitude(ref)
    if not amp:                                              # None or ~0 → near-diploid; no signal
        return None
    return round(amp / median_purity, 4) if median_purity else amp


def aneuploidy_purity(sample_tpm_by_symbol, cancer_type, median_purity=None, bulk_amplitude=None):
    """Calibrated purity from bulk aneuploidy: ``clip(A_obs / A_ref(type), 0, 1)``.

    Pass ``bulk_amplitude`` if already computed (avoids recomputing the sample's bulk aneuploidy).
    Returns None when the type is uncalibratable (no reference / near-diploid) or the sample's
    aneuploidy is unavailable — i.e. when aneuploidy carries no usable purity information.
    """
    a_ref = aneuploidy_reference(cancer_type, median_purity)
    if not a_ref:
        return None
    if bulk_amplitude is None:
        from .expression_decomposition import bulk_aneuploidy_amplitude
        bulk_amplitude = bulk_aneuploidy_amplitude(sample_tpm_by_symbol)
    if bulk_amplitude is None:
        return None
    return round(float(np.clip(bulk_amplitude / a_ref, 0.0, 1.0)), 3)
