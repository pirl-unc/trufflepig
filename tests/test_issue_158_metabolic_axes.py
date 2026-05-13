"""Regression tests for #158 — metabolic evidence surfaces as
therapy-axis rows in the report.

Step-0 already computes proliferation / hypoxia / glycolysis
channel scores on ``TumorEvidenceScore`` and ``ca9_tpm`` /
``glycolysis_geomean_fold`` / ``prolif_log2`` raw values. The
``_metabolic_axes_rows`` helper promotes these into actionable
report rows (CA9-directed ADCs, MCT inhibitors, CDK4/6 candidates)
when any channel crosses the display threshold.
"""

from trufflepig.main import _metabolic_axes_rows
from trufflepig.tumor_evidence import TumorEvidenceScore


def test_none_evidence_returns_empty():
    assert _metabolic_axes_rows(None) == []


def test_quiet_evidence_returns_empty():
    ev = TumorEvidenceScore(
        hypoxia=0.1,
        glycolysis=0.2,
        proliferation=0.3,
        ca9_tpm=2.0,
        glycolysis_geomean_fold=1.0,
        prolif_log2=2.5,
    )
    assert _metabolic_axes_rows(ev) == []


def test_hypoxia_surfaces_ca9_axis():
    ev = TumorEvidenceScore(hypoxia=0.8, ca9_tpm=60.0)
    rows = _metabolic_axes_rows(ev)
    assert len(rows) == 1
    axis, signal, therapy = rows[0]
    assert "Hypoxia" in axis
    assert "60" in signal
    assert "acetazolamide" in therapy.lower() or "ca9" in therapy.lower()


def test_hypoxia_triggers_on_high_ca9_even_if_score_low():
    """A single very high CA9 (≥ 20 TPM) fires the axis even if the
    aggregated hypoxia channel hasn't saturated — CA9 itself is the
    actionable readout here."""
    ev = TumorEvidenceScore(hypoxia=0.3, ca9_tpm=30.0)
    rows = _metabolic_axes_rows(ev)
    assert len(rows) == 1
    assert "Hypoxia" in rows[0][0]


def test_glycolysis_surfaces_mct_axis():
    ev = TumorEvidenceScore(glycolysis=0.7, glycolysis_geomean_fold=4.0)
    rows = _metabolic_axes_rows(ev)
    assert len(rows) == 1
    axis, signal, therapy = rows[0]
    assert "Glycolysis" in axis or "MCT" in axis
    assert "MCT" in therapy or "LDHA" in therapy


def test_proliferation_surfaces_cell_cycle_axis():
    ev = TumorEvidenceScore(proliferation=0.7, prolif_log2=4.8)
    rows = _metabolic_axes_rows(ev)
    assert len(rows) == 1
    axis, signal, therapy = rows[0]
    assert "Proliferation" in axis or "cell-cycle" in axis.lower()
    assert "CDK" in therapy or "palbociclib" in therapy.lower()


def test_all_three_axes_emit_in_order():
    ev = TumorEvidenceScore(
        hypoxia=0.9,
        ca9_tpm=80.0,
        glycolysis=0.7,
        glycolysis_geomean_fold=5.0,
        proliferation=0.8,
        prolif_log2=5.5,
    )
    rows = _metabolic_axes_rows(ev)
    assert len(rows) == 3
    # Order must be stable so the rendered report is deterministic.
    assert "Hypoxia" in rows[0][0]
    assert "Glycolysis" in rows[1][0] or "MCT" in rows[1][0]
    assert "Proliferation" in rows[2][0] or "cell-cycle" in rows[2][0].lower()
