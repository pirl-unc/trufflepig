"""MALAT1 / NEAT1 polyadenylation-bias QC classification + default drop.

Pins:
  - classify_gene_qc routes MALAT1 / NEAT1 to the
    ``polyadenylation_bias_lncrna`` group.
  - summarize_qc_class_shares surfaces ``lncrna_qc_artifact_fraction``
    and a per-gene share map so QC text/visualizations can report
    presence/absence before the drop.
  - is_rescue_feature returns True for them (they're in the default
    technical-RNA group set).
  - normalize_expression zeroes them and renormalizes by default.
  - technical_rna_component_phrase names MALAT1 / NEAT1 individually
    when present at material share.
"""

import pandas as pd
import pytest

from trufflepig.expression_qc import (
    classify_gene_qc,
    is_rescue_feature,
    raw_qc_profile,
    summarize_qc_class_shares,
    technical_rna_component_phrase,
)
from trufflepig.expression_normalize import normalize_expression


def test_malat1_and_neat1_classify_to_polyadenylation_bias_lncrna():
    assert classify_gene_qc("MALAT1").group == "polyadenylation_bias_lncrna"
    assert classify_gene_qc("NEAT1").group == "polyadenylation_bias_lncrna"
    # Other nuclear-retained lncRNAs (KCNQ1OT1, XIST, HOTAIR) are
    # intentionally NOT in the default panel — their biological signal
    # is stronger than their artifact signal.
    assert classify_gene_qc("KCNQ1OT1").group != "polyadenylation_bias_lncrna"
    assert classify_gene_qc("XIST").group != "polyadenylation_bias_lncrna"
    assert classify_gene_qc("HOTAIR").group != "polyadenylation_bias_lncrna"


def test_malat1_and_neat1_are_rescue_features():
    assert is_rescue_feature("MALAT1")
    assert is_rescue_feature("NEAT1")


def test_summarize_surfaces_lncrna_artifact_share():
    items = [
        ("ACTB", 100.0),
        ("GAPDH", 100.0),
        ("MALAT1", 600.0),
        ("NEAT1", 200.0),
    ]
    summary = summarize_qc_class_shares(items)
    assert summary["lncrna_qc_artifact_fraction"] == 0.8
    per_gene = summary["lncrna_qc_artifact_per_gene_share"]
    assert per_gene["MALAT1"] == 0.6
    assert per_gene["NEAT1"] == 0.2
    # MALAT1 sorts ahead of NEAT1 by TPM share.
    assert list(per_gene.keys()) == ["MALAT1", "NEAT1"]


def test_raw_qc_profile_carries_lncrna_share():
    items = [("ACTB", 100.0), ("MALAT1", 900.0)]
    prof = raw_qc_profile(items)
    assert prof["class_shares"]["lncrna_qc_artifact_fraction"] == 0.9


def test_normalize_expression_drops_malat1_and_neat1_by_default():
    df = pd.DataFrame(
        {
            "Symbol": ["ACTB", "GAPDH", "MALAT1", "NEAT1", "RNA5SP389", "MT-CO1"],
            "TPM": [100.0, 100.0, 600.0, 200.0, 50.0, 50.0],
        }
    )
    out, record = normalize_expression(df)
    assert record["applied"]
    s = out.set_index("Symbol")["TPM"]
    # All four artifact features zeroed.
    assert s["MALAT1"] == 0.0
    assert s["NEAT1"] == 0.0
    assert s["RNA5SP389"] == 0.0
    assert s["MT-CO1"] == 0.0
    # Remaining mass renormalized back to the original column total.
    assert out["TPM"].sum() == pytest.approx(1100.0)


def test_technical_rna_phrase_names_malat1_and_neat1_individually():
    items = [
        ("ACTB", 100.0),
        ("MALAT1", 400.0),
        ("NEAT1", 100.0),
    ]
    summary = summarize_qc_class_shares(items)
    phrase = technical_rna_component_phrase(summary)
    assert "MALAT1" in phrase
    assert "NEAT1" in phrase
    # MALAT1 share is ~67%; ensure the % rendering happens.
    assert "%" in phrase
