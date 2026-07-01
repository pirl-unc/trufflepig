"""Tests for the cheap chromosome-arm aneuploidy axis."""
import numpy as np
import pytest

from trufflepig import aneuploidy_axis as ax
from trufflepig.aneuploidy_axis import aneuploidy_score, gene_arm_map


def test_gene_arm_map_known_loci():
    arms = gene_arm_map()
    assert len(arms) > 10000
    assert arms.get("TP53") == "17p"      # 17p13
    assert arms.get("MYCN") == "2p"       # 2p24
    assert arms.get("EGFR") == "7p"       # 7p11


def test_diploid_sample_scores_low_aneuploid_sample_scores_higher():
    ref = ax._diploid_reference()
    arms = gene_arm_map()
    baseline = aneuploidy_score(ref.to_dict())["score"]            # sample == reference → ~no arm deviation
    # The MAD score is (by design) robust to a single chromosome; it detects GENOME-WIDE
    # aneuploidy — so simulate gains/losses across many chromosomes.
    factors = {str(c): (2.0 if c % 2 else 0.5) for c in range(1, 13)}
    shifted = ref.copy()
    for g in ref.index:
        chrom = arms.get(g, "")[:-1]                               # "7p" -> "7"
        if chrom in factors:
            shifted.loc[g] = shifted.loc[g] * factors[chrom]
    out = aneuploidy_score(shifted.to_dict())
    gained = out["score"]
    assert baseline is not None and gained is not None
    assert gained > baseline + 0.1                                 # broad aneuploidy is detected
    assert not (set(out["top_gained"]) & set(out["top_lost"]))     # an arm is never both gained and lost


def test_returns_none_not_nan_when_empty():
    out = aneuploidy_score({"NOT_A_GENE": 1.0})
    assert out["score"] is None                                    # valid JSON null, not NaN


def test_fallback_safe_when_genome_unavailable(monkeypatch):
    def boom():
        raise ImportError("no pyensembl genome")
    monkeypatch.setattr(ax, "gene_arm_map", boom)
    out = aneuploidy_score({"TP53": 10.0, "MYCN": 5.0})
    assert out["score"] is None
    assert "unavailable" in out.get("note", "")                    # degrades gracefully, no raise
