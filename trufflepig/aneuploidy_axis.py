"""Cheap aneuploidy axis — a "poor-man's inferCNV" scalar.

A malignancy signal orthogonal to lineage and to proliferation: malignant cells carry
chromosome-arm copy-number changes; normal stroma / CAFs / infiltrating immune are
diploid. We don't need inferCNV's HMM/smoothing because we want an *axis* (is this
aneuploid?), not per-region CNA *calls*.

Method (one pass):
  1. map genes -> chromosome arm (Ensembl GRCh37 positions + a static centromere table),
  2. per gene, ``log2((sample+1)/(diploid_reference+1))``,
  3. per-arm **median** of those log-ratios (averaging many co-located genes cancels
     gene-specific noise and leaves the shared copy-number component),
  4. ``score`` = robust dispersion (MAD) of the centered per-arm medians.

High score = several chromosome arms coherently shifted = aneuploid = malignant. The
reference is a pan-normal (diploid) profile, so per-arm gene-composition bias cancels.

Stacks with the proliferation hallmark: proliferation catches proliferative-but-diploid
tumors, aneuploidy catches aneuploid-but-indolent ones (CLL del13q/+12, many PRAD) that
proliferation misses.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import numpy as np
import pandas as pd

# Approximate GRCh37/hg19 centromere midpoints (bp); genes below -> p arm, above -> q arm.
_CENTROMERE = {
    "1": 125.0e6, "2": 93.3e6, "3": 91.0e6, "4": 50.4e6, "5": 48.4e6, "6": 61.0e6,
    "7": 59.9e6, "8": 45.6e6, "9": 49.0e6, "10": 40.2e6, "11": 53.7e6, "12": 35.8e6,
    "13": 17.9e6, "14": 17.6e6, "15": 19.0e6, "16": 36.6e6, "17": 24.0e6, "18": 17.2e6,
    "19": 26.5e6, "20": 27.5e6, "21": 13.2e6, "22": 14.7e6, "X": 60.6e6, "Y": 12.5e6,
}
_ENSEMBL_RELEASE = 75  # GRCh37, installed locally


@lru_cache(maxsize=1)
def gene_arm_map() -> dict:
    """``{HGNC symbol: chromosome-arm}`` e.g. ``{"TP53": "17p"}`` (Ensembl GRCh37)."""
    from pyensembl import EnsemblRelease

    data = EnsemblRelease(_ENSEMBL_RELEASE)
    out: dict[str, str] = {}
    for g in data.genes():
        chrom = str(g.contig)
        if chrom not in _CENTROMERE or not g.gene_name:
            continue
        start = float(g.start)
        arm = "p" if start < _CENTROMERE[chrom] else "q"
        out[g.gene_name] = f"{chrom}{arm}"
    return out


@lru_cache(maxsize=1)
def _diploid_reference() -> pd.Series:
    """Pan-normal diploid per-gene reference (mean of HPA normal cell types)."""
    from pirlygenes.expression.accessors import hpa_cell_type_expression

    hpa = hpa_cell_type_expression().drop_duplicates("Symbol").set_index("Symbol")
    cell_cols = [c for c in hpa.columns if c not in ("Ensembl_Gene_ID",)]
    return hpa[cell_cols].mean(axis=1).clip(lower=0)


def aneuploidy_score(sample_tpm_by_symbol: Mapping[str, float],
                     reference_tpm_by_symbol: Mapping[str, float] | None = None,
                     *, min_genes_per_arm: int = 25) -> dict:
    """Scalar aneuploidy signal for one sample (``{symbol: clean-TPM}``).

    Returns ``{score, n_arms, arm_profile, top_gained, top_lost}``. ``score`` is the MAD
    of centered per-arm log2-ratios vs the (diploid) reference; higher = more aneuploid.
    Pass a tumor-only residual to score the *tumor* compartment specifically.
    """
    arms = gene_arm_map()
    sample = pd.Series(dict(sample_tpm_by_symbol), dtype=float)
    ref = (pd.Series(dict(reference_tpm_by_symbol), dtype=float)
           if reference_tpm_by_symbol is not None else _diploid_reference())

    genes = [g for g in sample.index if g in arms and g in ref.index]
    if not genes:
        return {"score": float("nan"), "n_arms": 0, "arm_profile": {}, "top_gained": [], "top_lost": []}
    logratio = np.log2((sample.loc[genes] + 1.0) / (ref.loc[genes] + 1.0))
    by_arm = pd.DataFrame({"arm": [arms[g] for g in genes], "lr": logratio.values})
    counts = by_arm.groupby("arm")["lr"].size()
    arm_med = by_arm.groupby("arm")["lr"].median()[counts >= min_genes_per_arm]
    if arm_med.empty:
        return {"score": float("nan"), "n_arms": 0, "arm_profile": {}, "top_gained": [], "top_lost": []}
    centered = arm_med - arm_med.median()                      # remove global tumor/normal shift
    score = float(1.4826 * (centered - centered.median()).abs().median())  # MAD
    ordered = centered.sort_values()
    return {
        "score": round(score, 4),
        "n_arms": int(len(centered)),
        "arm_profile": {a: round(float(v), 3) for a, v in centered.items()},
        "top_gained": [a for a in ordered.index[::-1][:3]],
        "top_lost": [a for a in ordered.index[:3]],
    }
