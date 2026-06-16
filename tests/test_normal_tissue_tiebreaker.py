"""Unit tests for the normal-tissue parental-tissue tiebreaker.

When two cohort candidates sit within the close-call window on
support_score, the tiebreaker consults each candidate's primary
tissue match (CANCER_TO_TISSUE → HPA nTPM column) and boosts the
candidate whose primary tissue is present in the sample. The boost
is intentionally small (2%) so it flips rankings within the close
window but never overrides a clear winner.

These tests use synthetic ``rows`` and a stub for the host-tissue
scorer so we can exercise the tiebreaker logic without needing a
real expression dataframe.
"""

from __future__ import annotations

import pytest

import trufflepig.tumor_purity as tp_module
from trufflepig.tumor_purity import _apply_normal_tissue_tiebreaker


@pytest.fixture(autouse=True)
def _patch_host_tissue_scorer(monkeypatch):
    """Default stub: BRCA-favoring breast match."""

    def fake_scorer(sample_tpm, top_n=None, **kwargs):
        return [
            {"tissue": "breast", "score": 0.82, "n_genes": 12},
            {"tissue": "tongue", "score": 0.31, "n_genes": 9},
            {"tissue": "esophagus", "score": 0.28, "n_genes": 8},
            {"tissue": "lung", "score": 0.22, "n_genes": 7},
        ]

    monkeypatch.setattr(tp_module, "_score_host_tissue_details", fake_scorer)


def _row(code, support_score, signature_score=0.5):
    return {
        "code": code,
        "support_score": float(support_score),
        "support_geomean": float(support_score),  # support_score is itself the geomean now
        "signature_score": float(signature_score),
    }


def test_tiebreaker_promotes_breast_match_over_close_squamous_top():
    """The HCC1395-style scenario: ESCA tops by a hair, BRCA close, breast tissue present."""
    rows = [
        _row("ESCA", 1.00, signature_score=0.40),
        _row("BRCA", 0.99, signature_score=0.59),  # within the geomean-scale close window
        _row("LUSC", 0.85, signature_score=0.32),
    ]
    out = _apply_normal_tissue_tiebreaker(rows, sample_tpm_by_symbol={})
    assert out[0]["code"] == "BRCA"
    brca_row = next(r for r in out if r["code"] == "BRCA")
    assert brca_row["primary_tissue"] == "breast"
    assert brca_row["primary_tissue_match_score"] == pytest.approx(0.82)
    info = brca_row["normal_tissue_tiebreaker"]
    assert info["applied"] is True
    assert info["competing_top_code"] == "ESCA"
    assert "all_close_tissues" in info


def test_tiebreaker_is_a_no_op_when_top_already_best_on_tissue(monkeypatch):
    """Top candidate already matches its tissue — no flip needed."""

    def fake_scorer(sample_tpm, top_n=None, **kwargs):
        return [
            {"tissue": "breast", "score": 0.82},  # BRCA's tissue tops
            {"tissue": "esophagus", "score": 0.20},
        ]

    monkeypatch.setattr(tp_module, "_score_host_tissue_details", fake_scorer)
    rows = [
        _row("BRCA", 1.00, signature_score=0.59),
        _row("ESCA", 0.99, signature_score=0.40),  # within the geomean-scale close window
    ]
    out = _apply_normal_tissue_tiebreaker(rows, sample_tpm_by_symbol={})
    assert out[0]["code"] == "BRCA"
    # The top row gets a no-op tiebreaker note explaining why.
    info = out[0].get("normal_tissue_tiebreaker") or {}
    assert info.get("applied") is False


def test_tiebreaker_does_not_fire_outside_close_window():
    """A 20% gap shouldn't trigger the tiebreaker."""
    rows = [
        _row("ESCA", 1.00, signature_score=0.40),
        _row("BRCA", 0.79, signature_score=0.59),  # below 0.93 close_window
    ]
    out = _apply_normal_tissue_tiebreaker(rows, sample_tpm_by_symbol={})
    assert out[0]["code"] == "ESCA"  # unchanged
    assert "normal_tissue_tiebreaker" not in out[1]


def test_tiebreaker_does_nothing_when_candidates_share_tissue(monkeypatch):
    """LUAD + LUSC both map to 'lung' — tissue signal can't discriminate."""

    def fake_scorer(sample_tpm, top_n=None, **kwargs):
        return [{"tissue": "lung", "score": 0.65}]

    monkeypatch.setattr(tp_module, "_score_host_tissue_details", fake_scorer)
    rows = [
        _row("LUAD", 1.00, signature_score=0.40),
        _row("LUSC", 0.99, signature_score=0.42),  # within the geomean-scale close window
    ]
    out = _apply_normal_tissue_tiebreaker(rows, sample_tpm_by_symbol={})
    assert out[0]["code"] == "LUAD"
    # Order is unchanged — LUAD's tissue match is already best on its own,
    # so the top row gets a "no rerank needed" marker (and LUSC, which
    # also maps to lung, never gets promoted past it).
    info = out[0].get("normal_tissue_tiebreaker") or {}
    assert info.get("applied") is False


def test_tiebreaker_handles_missing_tissue_mapping():
    """Codes without a CANCER_TO_TISSUE entry are skipped gracefully."""
    rows = [
        _row("UNKNOWN_CODE", 1.00, signature_score=0.40),
        _row("BRCA", 0.96, signature_score=0.59),
    ]
    # Just verify it doesn't raise; behavior is sensible without crashing.
    out = _apply_normal_tissue_tiebreaker(rows, sample_tpm_by_symbol={})
    # With only BRCA having a known tissue, the tiebreaker has fewer
    # than 2 annotated candidates — should leave order intact.
    assert out[0]["code"] == "UNKNOWN_CODE"


def test_tiebreaker_handles_empty_rows():
    out = _apply_normal_tissue_tiebreaker([], sample_tpm_by_symbol={})
    assert out == []


def test_tiebreaker_handles_single_row():
    rows = [_row("BRCA", 1.0)]
    out = _apply_normal_tissue_tiebreaker(rows, sample_tpm_by_symbol={})
    assert out[0]["code"] == "BRCA"


def test_tiebreaker_skips_when_winning_tissue_score_too_low(monkeypatch):
    """A tied-but-no-tissue-signal case (cell-line-like) is a no-op."""

    def fake_scorer(sample_tpm, top_n=None, **kwargs):
        # Every tissue scores near zero — cell-line / dedifferentiated case.
        return [
            {"tissue": "breast", "score": 0.0},
            {"tissue": "esophagus", "score": 0.0},
        ]

    monkeypatch.setattr(tp_module, "_score_host_tissue_details", fake_scorer)
    rows = [
        _row("ESCA", 1.00, signature_score=0.40),
        _row("BRCA", 0.96, signature_score=0.59),
    ]
    out = _apply_normal_tissue_tiebreaker(rows, sample_tpm_by_symbol={})
    assert out[0]["code"] == "ESCA"  # no flip on zero scores
