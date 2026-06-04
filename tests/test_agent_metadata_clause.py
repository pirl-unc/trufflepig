"""Tests for the leveled-up therapeutic-agent metadata rendering (#47).

``agent_metadata_clause`` renders the enriched ``therapeutic-agents`` schema
(modality / route / approval / efficacy / toxicity / combination). The hard
contract is **backward compatibility**: until pirlygenes ships those columns,
every current curated target row must render to ``""`` so existing reports are
byte-for-byte unchanged.
"""

import pandas as pd

from trufflepig.reporting import agent_metadata_clause


def test_current_style_row_renders_empty():
    # The columns cancer-key-genes carries today — none of the enriched fields.
    row = {
        "symbol": "ERBB2",
        "agent": "trastuzumab",
        "agent_class": "mAb",
        "phase": "approved",
        "indication": "HER2+ breast cancer",
        "eligibility_note": "",
        "rationale": "",
    }
    assert agent_metadata_clause(row) == ""


def test_empty_and_nan_fields_render_empty():
    assert agent_metadata_clause({}) == ""
    assert agent_metadata_clause({"modality": "", "route": None, "efficacy": "nan"}) == ""


def test_full_enriched_row_renders_all_parts():
    row = {
        "agent": "trastuzumab deruxtecan",
        "modality": "adc",
        "modality_detail": "DXd payload",
        "route": "IV",
        "approval_status": "approved",
        "approval_year": "2019",
        "approving_body": "FDA",
        "efficacy": "ORR 61%",
        "key_toxicities": "ILD/pneumonitis",
        "boxed_warning": "true",
        "combination_partners": "pertuzumab",
    }
    clause = agent_metadata_clause(row)
    assert "ADC (DXd payload)" in clause
    assert "IV" in clause
    assert "approved 2019, FDA" in clause
    assert "efficacy ORR 61%" in clause
    assert "key toxicities: ILD/pneumonitis" in clause
    assert "boxed warning" in clause
    assert "combination with pertuzumab" in clause


def test_modality_vocab_maps_known_and_passes_through_unknown():
    assert agent_metadata_clause({"modality": "rlt"}) == "radioligand (RLT)"
    assert agent_metadata_clause({"modality": "tce"}) == "T-cell engager"
    assert agent_metadata_clause({"modality": "car_t"}) == "CAR-T"
    # Unknown modality passes through verbatim (forward-compatible).
    assert agent_metadata_clause({"modality": "antibody-peptide-conjugate"}) == (
        "antibody-peptide-conjugate"
    )


def test_combination_only_flag():
    assert (
        agent_metadata_clause({"combination_only": "true", "combination_partners": "carboplatin"})
        == "combination-only (with carboplatin)"
    )
    assert agent_metadata_clause({"combination_only": "yes"}) == "combination-only"


def test_isotope_detail_without_modality():
    # RLT payload/isotope detail surfaces even if modality column is blank.
    assert agent_metadata_clause({"isotope": "225Ac"}) == "225Ac"


def test_accepts_pandas_series_rows():
    series = pd.Series({"modality": "degrader", "route": "oral"})
    assert agent_metadata_clause(series) == "degrader; oral"


def test_real_curated_targets_are_all_backward_compatible():
    # Every role=target row pirlygenes ships today must render empty, proving
    # the wiring leaves current reports untouched.
    from pirlygenes import get_data

    kg = get_data("cancer-key-genes")
    targets = kg[kg["role"] == "target"]
    nonempty = [
        str(r.get("symbol"))
        for _, r in targets.iterrows()
        if agent_metadata_clause(r) != ""
    ]
    assert not nonempty, f"enriched-schema fields already present for: {nonempty[:10]}"
