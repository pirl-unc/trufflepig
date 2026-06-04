"""Tests for off-context known-target surfacing (#47 "surface when high").

A gene that is a drugged target *in another indication* should be surfaced when
it is highly expressed in a sample whose curated panel doesn't include it,
instead of being silently dropped.
"""

import pandas as pd

from trufflepig.reporting import (
    cross_cancer_target_index,
    offcontext_known_targets,
)


_INDEX = {
    "MSLN": (
        {"cancer_code": "MESO", "agent": "anetumab ravtansine", "phase": "phase_2",
         "indication": "mesothelioma", "agent_class": "ADC"},
    ),
    "FOLH1": (
        {"cancer_code": "PRAD", "agent": "177Lu-PSMA-617", "phase": "approved",
         "indication": "mCRPC", "agent_class": "radioligand"},
    ),
}


def _ranges(rows):
    return pd.DataFrame(rows)


def test_surfaces_high_offpanel_known_target():
    ranges = _ranges([
        {"symbol": "MSLN", "observed_tpm": 80.0, "attr_tumor_tpm": 60.0},
    ])
    hits = offcontext_known_targets(ranges, panel_symbols={"ERBB2"}, index=_INDEX)
    assert [h["symbol"] for h in hits] == ["MSLN"]
    assert hits[0]["tumor_tpm"] == 60.0  # attr_tumor_tpm preferred
    assert hits[0]["indications"][0]["agent"] == "anetumab ravtansine"


def test_excludes_targets_already_on_panel():
    ranges = _ranges([{"symbol": "MSLN", "attr_tumor_tpm": 90.0}])
    assert offcontext_known_targets(ranges, panel_symbols={"MSLN"}, index=_INDEX) == []


def test_excludes_low_expression():
    ranges = _ranges([{"symbol": "FOLH1", "attr_tumor_tpm": 3.0}])
    assert offcontext_known_targets(ranges, panel_symbols=set(), index=_INDEX) == []
    # ...but clears at the threshold
    ranges_hi = _ranges([{"symbol": "FOLH1", "attr_tumor_tpm": 12.0}])
    assert len(offcontext_known_targets(ranges_hi, set(), index=_INDEX, min_tumor_tpm=10.0)) == 1


def test_excludes_genes_that_are_not_known_targets():
    ranges = _ranges([{"symbol": "GAPDH", "attr_tumor_tpm": 5000.0}])
    assert offcontext_known_targets(ranges, panel_symbols=set(), index=_INDEX) == []


def test_sorted_by_tumor_tpm_descending():
    ranges = _ranges([
        {"symbol": "FOLH1", "attr_tumor_tpm": 40.0},
        {"symbol": "MSLN", "attr_tumor_tpm": 120.0},
    ])
    hits = offcontext_known_targets(ranges, panel_symbols=set(), index=_INDEX)
    assert [h["symbol"] for h in hits] == ["MSLN", "FOLH1"]


def test_falls_back_to_observed_tpm_when_no_attribution():
    ranges = _ranges([{"symbol": "MSLN", "observed_tpm": 55.0}])
    hits = offcontext_known_targets(ranges, panel_symbols=set(), index=_INDEX)
    assert hits and hits[0]["tumor_tpm"] == 55.0


def test_real_cross_cancer_index_has_flagship_targets():
    index = cross_cancer_target_index()
    # PSMA/HER2/BCMA are curated targets; index should map them to real cancers.
    assert "FOLH1" in index and any(e["cancer_code"] == "PRAD" for e in index["FOLH1"])
    assert "ERBB2" in index and any(e["agent"] for e in index["ERBB2"])
    assert "TNFRSF17" in index  # BCMA
