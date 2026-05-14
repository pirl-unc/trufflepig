# Licensed under the Apache License, Version 2.0
"""Unified tumor-evidence scoring (#149 synthesis).

Every tumor-evidence channel Step-0 computes — CTA re-expression,
oncofetal reactivation, coordinated proliferation, Warburg /
glycolysis, hypoxia, cancer-type-specific tumor-up hits, DDR
activation — each produces a normalised 0..1 score. Scores are
summed into a combined ``aggregate_score`` that the cancer-hint
logic consumes as a single number instead of a chain of isolated
threshold checks.

The aggregate is deliberately additive (not thresholded-OR): a
sample with five channels at 0.3 each (total 1.5) is as confident
a tumor call as one with a single channel at 1.0. This handles
the low-purity case where no single channel crosses the "strong"
threshold but several soft signals co-occur.

Gene panels are imported from :mod:`pirlygenes.gene_sets_cancer`
so the definitions are in one public-API location and re-usable
downstream (for scoring / calibration / cross-cohort comparison).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pirlygenes.gene_sets_cancer import glycolysis_panel_gene_names


# Per-channel saturating thresholds — the point at which a channel
# contributes its maximum 1.0 score. Below the threshold, the score
# scales linearly from 0 toward 1 based on the channel-specific
# metric. Tuned against the 6-sample battery so real tumors produce
# total ≥ 1.0 while healthy tissues stay below 0.3.
_CTA_SATURATION_COUNT = 5  # ≥5 CTA hits at 3+ TPM → 1.0
_ONCOFETAL_SATURATION_COUNT = 2  # ≥2 oncofetal hits → 1.0
_TYPE_SPECIFIC_SATURATION = 2  # ≥2 type-specific hits → 1.0
_PROLIF_SATURATION_LOG2 = 5.0  # panel geomean ≥5.0 log2-TPM → 1.0
_PROLIF_BASELINE_LOG2 = 2.0  # panel geomean ≤2.0 log2-TPM → 0.0
_CA9_SATURATION_TPM = 50.0  # CA9 ≥50 TPM → 1.0 hypoxia
_GLYCOLYSIS_SATURATION = 3.0  # panel fold-over-median ≥3 → 1.0


@dataclass
class TumorEvidenceScore:
    """Per-channel + combined evidence scores in [0, 1].

    A channel score reports how much this channel thinks the sample
    is a tumor. ``aggregate_score`` is the sum (un-clamped) — > 1.0
    means multiple channels agree, and a single channel at ≥ 1.0 is
    already sufficient by itself.

    Channels carry a ``label`` for the summary narrative.
    """

    cta: float = 0.0
    oncofetal: float = 0.0
    type_specific: float = 0.0
    proliferation: float = 0.0
    hypoxia: float = 0.0
    glycolysis: float = 0.0

    # Per-channel hit counts / values retained for the summary
    cta_count: int = 0
    oncofetal_count: int = 0
    type_specific_count: int = 0
    prolif_log2: float = 0.0
    ca9_tpm: float = 0.0
    glycolysis_geomean_fold: float = 0.0

    @property
    def aggregate_score(self) -> float:
        """Sum of per-channel scores (un-clamped; > 1.0 = multiple channels)."""
        return (
            self.cta
            + self.oncofetal
            + self.type_specific
            + self.proliferation
            + self.hypoxia
            + self.glycolysis
        )

    @property
    def strong_channels(self) -> list[str]:
        """Channels with score ≥ 0.8 — single-channel strong evidence."""
        out = []
        for name, val in [
            ("CTA", self.cta),
            ("oncofetal", self.oncofetal),
            ("type-specific", self.type_specific),
            ("proliferation", self.proliferation),
            ("hypoxia (CA9)", self.hypoxia),
            ("glycolysis", self.glycolysis),
        ]:
            if val >= 0.8:
                out.append(name)
        return out

    @property
    def soft_channels(self) -> list[str]:
        """Channels with 0.2 ≤ score < 0.8 — soft evidence."""
        out = []
        for name, val in [
            ("CTA", self.cta),
            ("oncofetal", self.oncofetal),
            ("type-specific", self.type_specific),
            ("proliferation", self.proliferation),
            ("hypoxia (CA9)", self.hypoxia),
            ("glycolysis", self.glycolysis),
        ]:
            if 0.2 <= val < 0.8:
                out.append(name)
        return out

    def synthesis(self) -> str:
        """One-paragraph evidence synthesis suitable for a summary."""
        strong = self.strong_channels
        soft = self.soft_channels
        parts = [
            f"Aggregate tumor-evidence score: **{self.aggregate_score:.2f}** "
            f"(≥1.0 = tumor-consistent; ≥0.3 = some evidence; <0.3 = weak)."
        ]
        if strong:
            parts.append(f"Strong channels: {', '.join(strong)}.")
        if soft:
            parts.append(f"Soft channels: {', '.join(soft)}.")
        if not strong and not soft:
            parts.append("No tumor-evidence channels fired.")
        # Per-channel detail
        parts.append(
            f"CTAs {self.cta_count} hits; oncofetal {self.oncofetal_count}; "
            f"type-specific {self.type_specific_count}; proliferation panel "
            f"{self.prolif_log2:.1f} log2-TPM; CA9 {self.ca9_tpm:.0f} TPM; "
            f"glycolysis geomean {self.glycolysis_geomean_fold:.1f}×."
        )
        return " ".join(parts)


def _ramp(value: float, low: float, high: float) -> float:
    """Linear ramp from 0 at ``low`` to 1 at ``high``, clipped."""
    if high <= low:
        return 1.0 if value >= high else 0.0
    return float(max(0.0, min(1.0, (value - low) / (high - low))))


def score_cta(cta_count: int) -> float:
    return _ramp(float(cta_count), 0.0, float(_CTA_SATURATION_COUNT))


def score_oncofetal(oncofetal_count: int) -> float:
    return _ramp(float(oncofetal_count), 0.0, float(_ONCOFETAL_SATURATION_COUNT))


def score_type_specific(type_specific_count: int) -> float:
    return _ramp(
        float(type_specific_count),
        0.0,
        float(_TYPE_SPECIFIC_SATURATION),
    )


def score_proliferation(prolif_log2: float) -> float:
    return _ramp(prolif_log2, _PROLIF_BASELINE_LOG2, _PROLIF_SATURATION_LOG2)


def score_hypoxia_ca9(ca9_tpm: float) -> float:
    return _ramp(ca9_tpm, 5.0, _CA9_SATURATION_TPM)


def score_glycolysis(sample_by_symbol: dict[str, float]) -> tuple[float, float]:
    """Glycolysis fold over cohort-universal baseline.

    Returns ``(score, geomean_fold)``. Fold is computed as the
    geometric mean of panel-gene TPMs divided by a conservative
    cross-tissue baseline (50 TPM) — tumors push the geomean above
    200; normal tissue sits below 100 typically.
    """
    genes = glycolysis_panel_gene_names()
    tpms = [float(sample_by_symbol.get(g, 0.0)) for g in genes]
    tpms = [t for t in tpms if t > 0]
    if not tpms:
        return 0.0, 0.0
    geomean = float(np.exp(np.mean(np.log(np.array(tpms) + 1.0))))
    baseline = 50.0
    fold = geomean / baseline
    return _ramp(fold, 1.0, _GLYCOLYSIS_SATURATION), fold


def compute_tumor_evidence_score(
    sample_by_symbol: dict[str, float],
    cta_count: int,
    oncofetal_count: int,
    type_specific_count: int,
    proliferation_log2_mean: float,
) -> TumorEvidenceScore:
    """Combine per-channel metrics into a :class:`TumorEvidenceScore`.

    Callers pass the channel hit counts + proliferation log2-TPM
    mean (pre-computed elsewhere) and the sample-expression dict.
    Hypoxia (CA9) and glycolysis (panel geomean) are computed here
    directly from ``sample_by_symbol``.
    """
    ca9_tpm = float(sample_by_symbol.get("CA9", 0.0))
    glycolysis, glycolysis_fold = score_glycolysis(sample_by_symbol)
    return TumorEvidenceScore(
        cta=score_cta(cta_count),
        oncofetal=score_oncofetal(oncofetal_count),
        type_specific=score_type_specific(type_specific_count),
        proliferation=score_proliferation(proliferation_log2_mean),
        hypoxia=score_hypoxia_ca9(ca9_tpm),
        glycolysis=glycolysis,
        cta_count=cta_count,
        oncofetal_count=oncofetal_count,
        type_specific_count=type_specific_count,
        prolif_log2=proliferation_log2_mean,
        ca9_tpm=ca9_tpm,
        glycolysis_geomean_fold=glycolysis_fold,
    )
