"""Unit tests for the per-candidate evidence tables in analysis.md.

The evidence-tables module is consumed by the analysis-markdown
builder; its public function ``build_candidate_evidence_block``
takes a ranked ``candidate_trace`` plus the sample TPM map and
returns markdown lines.

These tests pin three things that the original v1 got wrong in code
review on PR #30:

  1. The "candidate median" column actually uses
     ``subtype_deconvolved_expression`` when a winning subtype is
     present (not ``pan_cancer_expression`` aka the broad cohort
     median, which is luminal-dominated for BRCA — labeling its
     values "BRCA_Basal median" would be misleading).
  2. ``_rescue_evidence_table`` renders the TNBC/basal-BRCA rescue
     payload with all marker sections.
  3. ``_tiebreaker_evidence_table`` renders close-call tissue
     scores even when no boost fired (transparency).
"""

from __future__ import annotations


import pandas as pd

import trufflepig.evidence_tables as et_mod
from trufflepig.evidence_tables import (
    _per_subtype_evidence_table,
    _rescue_evidence_table,
    _tiebreaker_evidence_table,
    build_candidate_evidence_block,
)


# ---------------------------------------------------------------------------
# _per_subtype_evidence_table
# ---------------------------------------------------------------------------


def test_per_subtype_table_renders_with_competitor_and_dlog2():
    sample = {"KRT81": 600.0, "FOXC1": 30.0, "MIA": 180.0}
    cohort = {"KRT81": 200.0, "FOXC1": 25.0, "MIA": 150.0}
    competitor = {"KRT81": 1.0, "FOXC1": 3.0, "MIA": 1.0}

    lines = _per_subtype_evidence_table(
        "BRCA",
        "BRCA_Basal",
        sample,
        cohort,
        competitor,
        ["KRT81", "FOXC1", "MIA"],
        title_hint="Signature evidence for **BRCA** (winning subtype: `BRCA_Basal`)",
    )
    body = "\n".join(lines)
    assert "BRCA_Basal" in body
    assert "Δlog2" in body
    # Δlog2 row for KRT81 = log2(201 / 2) ≈ +6.65
    assert "+6.65" in body or "+6.6" in body
    # Caption explains the competitor convention.
    assert "Δlog2 = log2" in body


def test_per_subtype_table_drops_dlog2_without_competitor():
    sample = {"GENE": 10.0}
    cohort = {"GENE": 5.0}
    lines = _per_subtype_evidence_table(
        "ZZZZ",  # not in _COMPETING_COHORT_FOR_TABLE
        None,
        sample,
        cohort,
        competitor_medians=None,
        signature_genes=["GENE"],
    )
    body = "\n".join(lines)
    assert "Δlog2" not in body
    assert "| GENE |" in body
    assert "5.00" in body or "5" in body


def test_per_subtype_table_returns_empty_on_no_genes():
    assert _per_subtype_evidence_table("BRCA", None, {}, {}, None, []) == []


def test_per_subtype_table_does_not_invent_missing_reference_medians():
    body = "\n".join(
        _per_subtype_evidence_table(
            "SARC_MPLPS",
            None,
            {"KRT8": 42.0, "MDM2": 0.0},
            {},
            None,
            ["KRT8", "MDM2"],
        )
    )

    assert "No exact expression reference is available" in body
    assert "| Gene | Sample TPM |" in body
    assert "median |" not in body
    assert "| KRT8 | 42.0 |" in body


# ---------------------------------------------------------------------------
# _rescue_evidence_table
# ---------------------------------------------------------------------------


def test_rescue_evidence_table_renders_tnbc_basal_payload():
    payload = {
        "kind": "tnbc_basal_brca_misclassification",
        "keratin_tpm": {"KRT5": 506.0, "KRT14": 1122.0},
        "luminal_marker_tpm": {"ESR1": 0.14, "PGR": 0.01},
        "basal_positive_tpm": {"MIA": 181.0, "GABRP": 7.0},
        "squamous_program_tpm": {"TP63": 2.43, "SOX2": 0.07},
        "foxc1_tpm": 29.0,
        "urothelial_panel_sum_tpm": 1.96,
    }
    lines = _rescue_evidence_table(payload)
    body = "\n".join(lines)
    assert "Basal cytokeratin program" in body
    assert "Luminal program" in body
    assert "FOXC1" in body
    assert "Basal-mammary positive markers" in body
    assert "Squamous program" in body
    assert "Urothelial program" in body
    # Specific TPMs from each section.
    assert "1,122" in body  # KRT14 with thousand separator
    assert "0.14" in body
    assert "29.0" in body
    assert "Hoadley 2014" in body  # citation in the lead paragraph


def test_rescue_evidence_table_no_op_on_other_kinds():
    assert _rescue_evidence_table({"kind": "low_purity_prad_stromal_context"}) == []
    assert _rescue_evidence_table(None) == []
    assert _rescue_evidence_table({}) == []


# ---------------------------------------------------------------------------
# _tiebreaker_evidence_table
# ---------------------------------------------------------------------------


def test_tiebreaker_table_renders_when_boost_fired():
    rows = [
        {
            "code": "BRCA",
            "primary_tissue": "breast",
            "primary_tissue_match_score": 0.82,
            "normal_tissue_tiebreaker": {
                "applied": True,
                "boost_factor": 1.02,
                "close_window": 0.93,
                "tissue": "breast",
                "tissue_score": 0.82,
                "competing_top_code": "ESCA",
                "competing_top_tissue": "esophagus",
                "competing_top_tissue_score": 0.0,
            },
        },
        {
            "code": "ESCA",
            "primary_tissue": "esophagus",
            "primary_tissue_match_score": 0.0,
        },
    ]
    lines = _tiebreaker_evidence_table(rows)
    body = "\n".join(lines)
    assert "Normal-tissue tiebreaker" in body
    assert "BRCA" in body and "breast" in body
    assert "+2%" in body
    assert "0.820" in body  # BRCA tissue score


def test_tiebreaker_table_renders_when_no_boost_but_tissue_annotated():
    """Even without a boost, the tissue scores get surfaced for transparency."""
    rows = [
        {
            "code": "ESCA",
            "primary_tissue": "esophagus",
            "primary_tissue_match_score": 0.30,
        },
        {
            "code": "BRCA",
            "primary_tissue": "breast",
            "primary_tissue_match_score": 0.25,
        },
    ]
    lines = _tiebreaker_evidence_table(rows)
    body = "\n".join(lines)
    assert "Parental tissue scores" in body
    assert "ESCA" in body and "esophagus" in body
    assert "BRCA" in body and "breast" in body
    # No boost row anywhere.
    assert "+%" not in body
    assert "+1%" not in body


def test_tiebreaker_table_empty_without_annotations():
    assert _tiebreaker_evidence_table([]) == []
    assert _tiebreaker_evidence_table([{"code": "BRCA"}]) == []


# ---------------------------------------------------------------------------
# build_candidate_evidence_block — cohort-median lookup correctness
# (this is the test that would catch the v1 review bug)
# ---------------------------------------------------------------------------


def test_winning_subtype_uses_subtype_medians_not_broad_cohort(monkeypatch):
    """v1 of evidence_tables.py pulled "BRCA_Basal median" from BRCA_TPM,
    which is luminal-dominated. The header said "BRCA_Basal" but the
    values were the broad BRCA cohort. This regression test pins the fix.

    Construct a contrived BRCA cohort where the broad median for a basal
    marker is small (luminal-dominated) while the subtype median is
    large. The evidence table for a BRCA candidate with
    ``winning_subtype="BRCA_Basal"`` must show the subtype value.
    """
    # Stub pan_cancer_expression with a broad-BRCA value of 5.0 for KRT81.
    fake_pan = pd.DataFrame(
        {
            "Symbol": ["KRT81"],
            "Ensembl_Gene_ID": ["ENSG_KRT81"],
            "BRCA_TPM": [5.0],
            "HNSC_TPM": [0.5],
        }
    )

    def fake_pan_loader(**kwargs):
        return fake_pan

    import trufflepig.reference as ref_mod
    import trufflepig.plot_embedding as plot_emb_mod
    import trufflepig.subtype_signature as subsig_mod

    monkeypatch.setattr(ref_mod, "pan_cancer_expression", fake_pan_loader)
    monkeypatch.setattr(et_mod, "_subtype_medians_lookup", lambda: {
        ("BRCA", "BRCA_Basal"): {"KRT81": 600.0},
        ("BRCA", "BRCA_LumA"): {"KRT81": 1.5},
    })
    monkeypatch.setattr(
        subsig_mod,
        "subtype_signature_panels",
        lambda **k: {("BRCA", "BRCA_Basal"): ("KRT81",)},
    )
    monkeypatch.setattr(
        plot_emb_mod,
        "_get_cancer_type_signature_panels",
        lambda **k: {"BRCA": ("KRT81",)},
    )

    candidate_trace = [
        {
            "code": "BRCA",
            "winning_subtype": "BRCA_Basal",
            "signature_subtype_promoted": "BRCA_Basal",
        },
    ]
    lines = build_candidate_evidence_block(
        candidate_trace,
        sample_tpm_by_symbol={"KRT81": 624.0},
        top_k=1,
    )
    body = "\n".join(lines)
    # Subtype-median column must show 600 (BRCA_Basal), not 5 (broad).
    assert "600" in body
    # Verify the broad value did NOT slip into the column. Broad BRCA
    # value 5.0 appears nowhere as KRT81's "BRCA_Basal median".
    assert "| KRT81 |" in body
    # The row line must contain "600" between the second and third pipes.
    krt81_line = [ln for ln in lines if ln.startswith("| KRT81")][0]
    cells = [c.strip() for c in krt81_line.split("|")]
    # cells: ['', 'KRT81', '624.1', '600', '0.50', '+9.46', '']  (approx)
    # The third value cell (index 3) is the candidate median; assert it == 600.
    assert "600" in cells[3]


def test_build_block_handles_empty_trace():
    assert build_candidate_evidence_block(None, {}) == []
    assert build_candidate_evidence_block([], {}) == []


def test_build_block_surfaces_rescue_payload_when_present(monkeypatch):
    """When a candidate carries support_override, the rescue table renders."""
    import trufflepig.reference as ref_mod
    import trufflepig.plot_embedding as plot_emb_mod
    import trufflepig.subtype_signature as subsig_mod

    monkeypatch.setattr(et_mod, "_subtype_medians_lookup", lambda: {})
    monkeypatch.setattr(subsig_mod, "subtype_signature_panels", lambda **k: {})
    monkeypatch.setattr(
        plot_emb_mod,
        "_get_cancer_type_signature_panels",
        lambda **k: {"BRCA": ("FOXC1",)},
    )
    monkeypatch.setattr(
        ref_mod,
        "pan_cancer_expression",
        lambda **k: pd.DataFrame(
            {"Symbol": ["FOXC1"], "Ensembl_Gene_ID": ["ENSG_FOXC1"], "BRCA_TPM": [1.4]}
        ),
    )
    trace = [
        {
            "code": "BRCA",
            "support_override": {
                "kind": "tnbc_basal_brca_misclassification",
                "keratin_tpm": {"KRT14": 1122.0},
                "luminal_marker_tpm": {"ESR1": 0.14},
                "basal_positive_tpm": {"MIA": 181.0},
                "squamous_program_tpm": {"TP63": 2.4},
                "foxc1_tpm": 29.0,
                "urothelial_panel_sum_tpm": 1.96,
            },
        }
    ]
    body = "\n".join(build_candidate_evidence_block(trace, {"FOXC1": 30.0}))
    assert "TNBC / basal-BRCA misclassification rescue" in body
    assert "1,122" in body
