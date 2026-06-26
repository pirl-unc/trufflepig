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

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

_MIN_MEDIAN_PURITY = 0.1   # floor the extrapolation denominator so a tiny median can't blow up A_ref

# NOTE (#96): A_ref here is built from the cohort-MEAN reference profile, whose per-arm dispersion is
# lower than a single tumor's (sporadic events average out; only recurrent arm events survive). So
# A_ref can under-estimate the true pure-sample amplitude → aneuploidy_purity can saturate. The
# proper fix is to regress single-sample amplitude vs per-sample TCGA ABSOLUTE purity (slope = A_ref)
# and ship that table; this on-the-fly version is the documented prototype.


def _reference_code_candidates(cancer_type):
    """The cancer code, then its registry parent — so a rare subtype without its own pan-cancer
    reference column (e.g. ``SARC_DSRCT``) falls back to the parent cohort's column (``SARC``)."""
    code = str(cancer_type or "").strip().upper()
    candidates = [code] if code else []
    try:
        from trufflepig.analyze.cancer_type_context import registry_parent_code

        parent = (registry_parent_code(code) or "").strip().upper()
        if parent and parent not in candidates:
            candidates.append(parent)
    except (ImportError, KeyError, ValueError):                # no registry / unknown code → no parent
        pass
    return candidates


@lru_cache(maxsize=None)
def _type_reference_sample(cancer_type):
    """``{symbol: clean-TPM}`` reference profile for a cancer type, or None.

    Uses pirlygenes' pan-cancer ``<CODE>_TPM_clean`` reference column — already clean-TPM scale (so it
    matches the analyzed sample) and a declared dependency, avoiding an undeclared oncoref import. A
    rare subtype without its own column falls back to its parent cohort's column (e.g. SARC_DSRCT →
    SARC), which closes the aneuploidy reference-gap for the sarcoma subtypes (the aneuploid types
    where this signal matters most). Types with no own *or* parent column (ATRT/HEPB/NUTM/heme) have
    no pan-cancer reference and stay uncalibratable here — and are typically near-diploid anyway.
    """
    try:
        from trufflepig.reference import pan_cancer_expression

        indexed = pan_cancer_expression(technical_rna_normalize=True).drop_duplicates("Symbol").set_index("Symbol")
        for code in _reference_code_candidates(cancer_type):
            col = f"{code}_TPM_clean"
            if col not in indexed.columns:
                continue
            s = indexed[col].astype(float)
            s = s[s > 0]
            if not s.empty:
                return s.to_dict()
        return None
    except (ImportError, KeyError, ValueError):                # pan-cancer reference unavailable
        logger.debug("aneuploidy reference unavailable for %s", cancer_type, exc_info=True)
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
    if median_purity is None:                               # no purity prior → treat reference as ~pure
        return round(amp, 4)
    return round(amp / max(median_purity, _MIN_MEDIAN_PURITY), 4)  # extrapolate cohort→purity 1 (floored denom)


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
    if not bulk_amplitude:                                   # None OR 0.0 — no detectable aneuploidy in
        return None                                         # this sample → aneuploidy is uninformative (not "0% pure")
    return round(float(np.clip(bulk_amplitude / a_ref, 0.0, 1.0)), 3)
