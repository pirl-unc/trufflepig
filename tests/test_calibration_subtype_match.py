"""Subtype-aware code matching for the calibration script.

A model answer of "BRCA_LumB" against an expected label of "BRCA"
should count as correct (the call is MORE specific than the
expected parent cohort, not wrong). Symmetrically, "BRCA" against
expected "BRCA_LumB" is correct as a parent-class agreement.
"LUSC" against expected "BRCA" remains wrong (different cancer).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Load the script as a module (it isn't an installed package).
_spec = importlib.util.spec_from_file_location(
    "calibrate_decomposition",
    Path(__file__).resolve().parents[1] / "scripts" / "calibrate_decomposition.py",
)
calib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calib)


def test_exact_match_is_exact_kind():
    matched, kind = calib._codes_match("BRCA", "BRCA")
    assert matched is True
    assert kind == "exact"


def test_subtype_of_expected_counts_as_correct():
    """``BRCA_LumB`` vs expected ``BRCA`` → MORE specific, correct."""
    matched, kind = calib._codes_match("BRCA_LumB", "BRCA")
    assert matched is True
    assert kind == "subtype_of_expected"


def test_parent_of_expected_counts_as_correct():
    """``BRCA`` vs expected ``BRCA_LumB`` → less specific, correct family."""
    matched, kind = calib._codes_match("BRCA", "BRCA_LumB")
    assert matched is True
    assert kind == "parent_of_expected"


def test_unrelated_cohort_is_wrong():
    matched, kind = calib._codes_match("LUSC", "BRCA")
    assert matched is False
    assert kind == "none"


def test_sibling_subtype_is_wrong():
    """``BRCA_LumA`` vs expected ``BRCA_LumB`` → same parent but
    not the same subtype; counted as wrong because the model picked
    the wrong subtype of the right parent."""
    matched, kind = calib._codes_match("BRCA_LumA", "BRCA_LumB")
    assert matched is False, (
        "Sibling subtypes are different cancer types; classifier should "
        "not get credit for the wrong subtype just because the parent matches."
    )


def test_empty_inputs_are_wrong():
    assert calib._codes_match("", "BRCA") == (False, "none")
    assert calib._codes_match("BRCA", "") == (False, "none")


def test_top3_kind_picks_best_match_class():
    """If exact AND subtype both appear in top-3, ``kind`` should
    report the strongest agreement (exact > subtype > parent)."""
    matched, kind = calib._any_in_kin(["LUSC", "BRCA_LumB", "BRCA"], "BRCA")
    assert matched is True
    assert kind == "exact"


def test_top3_subtype_only_when_no_exact():
    matched, kind = calib._any_in_kin(["LUSC", "BRCA_LumB", "ESCA"], "BRCA")
    assert matched is True
    assert kind == "subtype_of_expected"


def test_top3_parent_only():
    matched, kind = calib._any_in_kin(["LUSC", "BRCA", "ESCA"], "BRCA_LumB")
    assert matched is True
    assert kind == "parent_of_expected"


def test_top3_no_kin():
    matched, kind = calib._any_in_kin(["LUSC", "ESCA", "HNSC"], "BRCA")
    assert matched is False
    assert kind == "none"
