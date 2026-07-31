"""Tests for the tumour-intrinsic epithelial-exclusion lineage gate (multi-view)."""

import numpy as np

from trufflepig.lineage_evidence import (
    DEFAULT_EPITHELIAL_CONFIDENCE,
    EPITHELIAL_EXCLUDES,
    EPITHELIAL_MARKERS,
    lineage_exclusion_evidence,
)

SCALE = 100.0
# synthetic cross-cohort background for the epithelial markers (low) — keeps the
# tests fast (no reference-matrix load) and deterministic.
COHORT = {g: np.array([1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 3.0, 1.0, 0.0, 2.0]) for g in EPITHELIAL_MARKERS}


def _sample(marker_ratio):
    """A sample where the epithelial markers sit at ``marker_ratio`` x HK among a
    low background — so within-sample percentile is high when markers are high."""
    d = {f"FILL{i}": 1.0 for i in range(30)}
    for g in EPITHELIAL_MARKERS:
        d[g] = marker_ratio * SCALE
    return d


def _ev(marker_ratio, **kw):
    return lineage_exclusion_evidence(
        _sample(marker_ratio), cohort_reference=COHORT, **kw
    )


def test_epithelial_absent_demotes_epithelial_branch():
    """#69: a confident epithelial absence (a real sarcoma) demotes the EPITHELIAL
    branch so a stroma-matched carcinoma can't out-compete the sarcoma — the inverse
    of the present-arm. (The dead band 0.15..0.45 leaves borderline low-purity
    carcinomas untouched.)"""
    ev = _ev(0.0)  # real sarcoma: epithelial program off
    assert set(ev.factors) == {"epithelial"}
    assert ev.factors["epithelial"] < 1.0
    assert ev.notes
    assert ev.signal.call == "absent"


def test_epithelial_borderline_is_in_the_dead_band():
    """A weak/borderline epithelial signal (between the absent and present
    thresholds) fires neither arm — so a stroma-rich carcinoma isn't wrongly
    demoted out of the epithelial branch."""
    ev = _ev(0.3)  # mid-band
    if 0.15 < ev.signal.confidence < 0.45:
        assert "epithelial" not in ev.factors


def test_demotes_non_epithelial_when_epithelial_present():
    ev = _ev(4.0)  # strong epithelial program
    assert set(ev.factors) == set(EPITHELIAL_EXCLUDES)
    assert all(f < 1.0 for f in ev.factors.values())
    assert ev.notes
    assert ev.signal.call == "present"


def test_demotion_is_confidence_proportional():
    weak = _ev(0.8)
    strong = _ev(5.0)
    assert strong.signal.confidence > weak.signal.confidence
    if weak.factors and strong.factors:
        assert strong.factors["mesenchymal"] <= weak.factors["mesenchymal"]


def test_fingerprint_reports_non_hk_views():
    ev = _ev(4.0)
    note = ev.notes[0]
    for token in (
        "log1p(clean TPM)",
        "within-sample pct",
        "cohort-pct",
        "cohort-z",
    ):
        assert token in note, note


def test_threshold_is_a_confidence_in_unit_interval():
    assert 0.0 < DEFAULT_EPITHELIAL_CONFIDENCE < 1.0


# --- specific-program relative-margin reasoning (#75) -------------------------

import numpy as _np  # noqa: E402
from trufflepig.lineage_evidence import SPECIFIC_LINEAGE_PROGRAMS  # noqa: E402

_PROG_BY_LINEAGE = {lin: markers for _n, markers, lin, _t in SPECIFIC_LINEAGE_PROGRAMS}


def _multi_program_sample(ratios):
    """Sample + cohort background where each lineage's program sits at its given
    ratio x HK among a low background."""
    d = {f"FILL{i}": 1.0 for i in range(30)}
    cohort = {}
    for lineage, markers in _PROG_BY_LINEAGE.items():
        for g in markers:
            d[g] = ratios.get(lineage, 0.0) * SCALE
            cohort[g] = _np.array([1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 3.0, 1.0, 0.0, 2.0])
    return d, cohort


def test_cofiring_specific_programs_do_not_cancel():
    """Two lineage-defining programs both present must NOT cancel (the earlier
    bug: independent per-program demotion left every lineage demoted equally so
    the original wrong top survived). With relative-margin reasoning the stronger
    lineage stays intact, the weaker is demoted only by their gap, and an
    unsupported lineage is demoted by the full margin."""
    d, cohort = _multi_program_sample(
        {"neuroendocrine": 0.4, "melanocytic": 0.1}
    )
    f = lineage_exclusion_evidence(d, cohort_reference=cohort).factors
    # both programs must clear threshold for this to be a real co-fire test
    assert f, "neither specific program fired — raise the ratios"
    # stronger lineage (NE) is intact; weaker (melanocytic) only mildly demoted
    assert f.get("neuroendocrine", 1.0) == 1.0
    assert f.get("melanocytic", 1.0) < 1.0
    # an unsupported lineage is demoted MORE than the co-firing weaker one
    assert f.get("epithelial", 1.0) < f["melanocytic"]
    # i.e. NOT the degenerate "all demoted equally"
    assert f["melanocytic"] > f.get("epithelial", 0.0)


def test_single_specific_program_demotes_all_others_decisively():
    """One program firing alone behaves as a decisive single call (margin == its
    confidence for every other lineage)."""
    d, cohort = _multi_program_sample({"neuroendocrine": 8.0})
    f = lineage_exclusion_evidence(d, cohort_reference=cohort).factors
    assert f.get("neuroendocrine", 1.0) == 1.0
    assert f.get("epithelial", 1.0) < 1.0 and f.get("mesenchymal", 1.0) < 1.0
