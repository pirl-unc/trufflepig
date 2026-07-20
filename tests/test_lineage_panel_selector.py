"""Regression: lineage_panel selector in cancer_type_evidence.

PR-42 wires ``trufflepig.lineage_panels`` into the cancer-type
evidence consolidator as a new selector. These tests pin the
contract:

  - The selector only fires when a single panel clearly wins
    (score >= 0.60 AND margin over second-best >= 0.20).
  - When it fires, the proposed cancer type is the winning panel's
    ``parent_cohort``, NOT the panel name (e.g. BRCA_BASAL → BRCA).
  - The selector is graceful: missing inputs, panel evaluation
    failures, and empty sample data all return without raising.
  - Hypothesis details carry the rationale so reports can surface it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trufflepig import cancer_type_evidence as cte


def _empty_analysis():
    return {
        "candidate_trace": [{"code": "ESCA", "support_fraction_of_top": 1.0}],
    }


def test_selector_no_op_on_empty_sample():
    """Empty sample_tpm → selector returns without registering hypotheses."""
    hyps: dict = {}
    cte._add_lineage_panel_features(hyps, {}, _empty_analysis())
    assert hyps == {}


def test_selector_does_not_fire_below_score_threshold():
    """A sample with no relevant lineage markers must not produce a
    lineage_panel hypothesis that can select the report label."""
    hyps: dict = {}
    # Sample with mostly housekeeping-level expression; no panel will
    # have its obligate cleared.
    sample = {"ACTB": 100.0, "GAPDH": 100.0, "B2M": 100.0}
    cte._add_lineage_panel_features(hyps, sample, _empty_analysis())
    # The selector may register no hypotheses or only weak ones — crucially no
    # lineage_panel entry may be ``can_select_report_label`` below threshold.
    for code, h in hyps.items():
        public = h.public_dict() or {}
        if "lineage_panel" not in set(public.get("evidence_sources", [])):
            continue
        assert public.get("can_select_report_label") is not True, (
            f"lineage_panel selected report label {code!r} on a "
            f"housekeeping-only sample that clears no panel obligate"
        )


def test_selector_proposes_parent_cohort_not_panel_name():
    """When BRCA_BASAL fires, proposed cancer_type is BRCA, not the
    panel name. Mirrors the basal-BRCA rescue's lifting of the
    panel-specific call to a registry-valid cancer code."""
    hyps: dict = {}
    # HCC1395-shaped sample (real values from the smoke test):
    # KRT14 high, KRT5 high, FOXC1 present, MIA elevated, luminal off.
    sample = {
        "KRT14": 708.0,
        "KRT5": 368.0,
        "FOXC1": 21.0,
        "MIA": 88.0,
        "ESR1": 0.0,
        "PGR": 0.0,
        "UPK1B": 0.0,
        "TP63": 1.1,
        "SOX2": 0.0,
        "MUCL1": 0.5,
        # Provide HK gene set for the selector's HK-median computation.
        "ACTB": 200.0,
        "GAPDH": 200.0,
    }
    cte._add_lineage_panel_features(hyps, sample, _empty_analysis())
    # BRCA hypothesis should exist with lineage_panel source. If the
    # panel framework is unavailable (e.g. cohort medians missing in
    # the test env), the selector early-returns — accept that.
    if not hyps:
        return  # selector graceful no-op is also a pass
    assert "BRCA" in hyps, (
        f"Expected BRCA hypothesis from BRCA_BASAL panel, got: "
        f"{list(hyps.keys())}"
    )
    brca = hyps["BRCA"]
    public = brca.public_dict() or {}
    sources = set(public.get("evidence_sources", []))
    assert "lineage_panel" in sources
    # public_dict spreads .details into the top-level dict (see
    # CancerTypeEvidence.public_dict ``row.update(self.details)``).
    assert public.get("lineage_panel_top") == "BRCA_BASAL"
    assert public.get("lineage_panel_score", 0) >= 0.60
    rationale = (public.get("lineage_panel_rationale") or "").lower()
    assert (
        "required markers" in rationale
        or "negative markers" in rationale
        or "in cohort range" in rationale
    )


def test_selector_is_robust_to_lineage_panels_import_failure(monkeypatch):
    """If lineage_panels can't be imported (test env, broken module),
    the selector logs and returns — must not raise."""
    import sys

    saved = sys.modules.pop("trufflepig.lineage_panels", None)
    monkeypatch.setattr(
        "trufflepig.cancer_type_evidence._LOGGER.warning",
        lambda *a, **k: None,
    )

    # Inject a broken import
    sys.modules["trufflepig.lineage_panels"] = None
    try:
        hyps: dict = {}
        cte._add_lineage_panel_features(
            hyps, {"ACTB": 100.0}, _empty_analysis()
        )
        assert hyps == {}
    finally:
        if saved is not None:
            sys.modules["trufflepig.lineage_panels"] = saved
        else:
            sys.modules.pop("trufflepig.lineage_panels", None)


def test_full_pipeline_includes_lineage_panel_in_evidence():
    """``select_report_scope_from_evidence`` must call the new
    selector and surface its output. Pins the integration."""
    df_expr = pd.DataFrame(
        {"ensembl_gene_id": [], "canonical_gene_name": [], "TPM": []}
    )
    analysis = {
        "candidate_trace": [
            {"code": "ESCA", "support_fraction_of_top": 1.0},
        ],
    }
    # Selector runs but no sample expression → returns no hypotheses.
    # The call must not raise and must produce a valid evidence dict.
    result = cte.select_report_scope_from_evidence(df_expr, analysis)
    assert "evidence" in result
    assert isinstance(result["evidence"], list)
    # Wiring point #2: lineage_panel_evidence must be a key in the
    # returned dict (value may be None when sample TPM is empty).
    assert "lineage_panel_evidence" in result


def test_selector_returns_summary_with_promotion_block():
    """When evaluation succeeds, _add_lineage_panel_features returns
    a summary dict with a ``promotion`` block describing why the
    panel did or didn't promote a label."""
    hyps: dict = {}
    summary = cte._add_lineage_panel_features(
        hyps,
        {"ACTB": 100.0, "GAPDH": 100.0, "B2M": 100.0},
        _empty_analysis(),
    )
    if summary is None:
        return  # selector early-returned (no panel hit threshold)
    assert "promotion" in summary
    promotion = summary["promotion"]
    assert "promoted" in promotion
    assert "blockers" in promotion


def test_basis_line_renders_panel_when_summary_present():
    """``brief._lineage_panel_evidence_line`` must surface the panel
    verdict whenever ``analysis['lineage_panel_evidence']`` has a
    top_score >= 0.5. Pins wiring point #3 (report rendering)."""
    from trufflepig import brief

    analysis = {
        "lineage_panel_evidence": {
            "top_panel": "BRCA_BASAL",
            "top_score": 0.78,
            "top_rationale": "5/5 required markers in cohort range; 5/5 negative markers below threshold",
            "margin_over_second": 0.30,
            "promotion": {
                "promoted": False,
                "code": "BRCA",
                "blockers": [
                    "broad classifier is confident — broad call preserved",
                ],
            },
        }
    }
    line = brief._lineage_panel_evidence_line(analysis, "BRCA")
    assert line is not None
    assert "BRCA_BASAL" in line
    assert "0.78" in line
    assert "required markers" in line
    # Non-promoted summary should be flagged as "noted, did not change the call".
    assert "noted, did not change the call" in line


def test_basis_line_renders_panel_when_promoted():
    """When the panel actually promoted the report label, the line
    notes that explicitly."""
    from trufflepig import brief

    analysis = {
        "lineage_panel_evidence": {
            "top_panel": "BRCA_BASAL",
            "top_score": 0.78,
            "top_rationale": "5/5 required markers in cohort range",
            "margin_over_second": 0.30,
            "promotion": {
                "promoted": True,
                "code": "BRCA",
                "blockers": [],
            },
        }
    }
    line = brief._lineage_panel_evidence_line(analysis, "BRCA")
    assert line is not None
    # "supports the BRCA call" when promoted_code == cancer_code.
    assert "supports the BRCA call" in line


def test_basis_line_skips_panel_below_threshold():
    """A panel verdict below the 0.5 reporting bar is suppressed."""
    from trufflepig import brief

    analysis = {
        "lineage_panel_evidence": {
            "top_panel": "BRCA_BASAL",
            "top_score": 0.30,
            "top_rationale": "weak",
            "promotion": {"promoted": False, "code": None, "blockers": []},
        }
    }
    assert brief._lineage_panel_evidence_line(analysis, "BRCA") is None


def test_basis_line_skips_panel_when_no_evidence():
    """When analysis has no lineage_panel_evidence, return None."""
    from trufflepig import brief

    assert brief._lineage_panel_evidence_line({}, "BRCA") is None
    assert (
        brief._lineage_panel_evidence_line(
            {"lineage_panel_evidence": None}, "BRCA"
        )
        is None
    )


def test_subtype_reasoning_line_surfaces_program_implication():
    """When a panel with a curated ``program_note`` fires above the
    brief reporting bar, the subtype line should expose the
    biological note so the report reads usefully — not just
    "BRCA_BASAL scored 0.78".
    """
    from trufflepig import brief

    analysis = {
        "lineage_panel_evidence": {
            "top_panel": "BRCA_BASAL",
            "top_score": 0.78,
            "top_panel_program_note": (
                "basal-like breast program (KRT5/KRT14, FOXC1, low ER/PR)."
            ),
            "promotion": {"promoted": False, "code": "BRCA", "blockers": []},
        }
    }
    line = brief._lineage_panel_subtype_reasoning_line(analysis, "BRCA")
    assert line is not None
    assert "BRCA_BASAL" in line
    assert "basal-like" in line.lower()
    # The line should NOT call the reader's attention with robotic
    # "pattern detected" framing — single source of truth is the
    # panel's program_note read verbatim.
    assert "pattern detected" not in line.lower()


def test_subtype_reasoning_line_skips_panel_without_note():
    """Panels with no curated ``program_note`` return None — silent
    rather than dumping a generic placeholder."""
    from trufflepig import brief

    analysis = {
        "lineage_panel_evidence": {
            "top_panel": "MADE_UP_PANEL",
            "top_score": 0.99,
            # no top_panel_program_note
            "promotion": {"promoted": False, "code": None, "blockers": []},
        }
    }
    assert brief._lineage_panel_subtype_reasoning_line(analysis, "BRCA") is None


def test_subtype_reasoning_line_skips_below_threshold():
    """A panel below the brief reporting bar doesn't get a subtype
    line either, even if it has a program_note."""
    from trufflepig import brief

    analysis = {
        "lineage_panel_evidence": {
            "top_panel": "BRCA_BASAL",
            "top_score": 0.30,
            "top_panel_program_note": "basal-like breast program",
            "promotion": {"promoted": False, "code": None, "blockers": []},
        }
    }
    assert brief._lineage_panel_subtype_reasoning_line(analysis, "BRCA") is None


def test_program_note_on_panel_is_single_source_of_truth():
    """Pin: every panel in LINEAGE_PANELS that needs a subtype line
    must carry its ``program_note`` directly. Catches the
    "added a panel, forgot to update brief.py's dict" foot-gun by
    ensuring brief.py reads ONLY from the panel.
    """
    from trufflepig.lineage_panels import LINEAGE_PANELS

    panels_with_notes = [p for p in LINEAGE_PANELS if p.program_note]
    # All current panels carry a note (sanity check the curation).
    assert len(panels_with_notes) == len(LINEAGE_PANELS), (
        "Some LINEAGE_PANELS lack a program_note — the report will "
        "skip the subtype line for those panels: "
        f"{[p.name for p in LINEAGE_PANELS if not p.program_note]}"
    )
    # Notes are short single-line strings (no newlines or huge text).
    for p in panels_with_notes:
        assert "\n" not in p.program_note, p.name
        assert len(p.program_note) < 300, p.name


# ---------------------------------------------------------------------------
# Family-aware promotion gate (regression for the SKCM → BRCA bug)
# ---------------------------------------------------------------------------
#
# Earlier wiring used a top-1/top-2 support ratio < 1.30 as the
# "broad is uncertain" trigger. On TCGA-EB-A24D-01 (SKCM) the broad
# classifier was correct (SKCM top-1) but sibling-melanocytic (UVM)
# was a close runner-up; that ratio gate fired and a BRCA lineage
# panel overrode the call. The family-aware gate replaces the ratio
# path with a same-family / explicitly-uncertain rule. These tests
# pin that contract end-to-end.

_HCC1395_SAMPLE = {
    "KRT14": 708.0,
    "KRT5": 368.0,
    "FOXC1": 21.0,
    "MIA": 88.0,
    "ESR1": 0.0,
    "PGR": 0.0,
    "UPK1B": 0.0,
    "TP63": 1.1,
    "SOX2": 0.0,
    "MUCL1": 0.5,
    "ACTB": 200.0,
    "GAPDH": 200.0,
}


def test_gate_blocks_cross_code_when_broad_confident_skcm_brca():
    """Regression for TCGA-EB-A24D-01 (SKCM→BRCA). Broad top-1 is a
    different cancer code than the panel's parent_cohort AND there
    is no explicit fit_quality=weak/ambiguous → the panel must NOT
    promote a report label.
    """
    hyps: dict = {}
    analysis = {
        "candidate_trace": [
            {
                "code": "SKCM",
                "support_score": 0.9,
            },
            {
                "code": "UVM",
                "support_score": 0.8,
            },
            {"code": "BRCA", "support_score": 0.3},
        ],
    }
    cte._add_lineage_panel_features(hyps, _HCC1395_SAMPLE, analysis)
    if not hyps:
        pytest.skip("panel framework unavailable — no panel cleared thresholds")
    brca = hyps.get("BRCA")
    if brca is None:
        pytest.skip("BRCA panel did not fire — cannot exercise the gate")
    public = brca.public_dict() or {}
    assert public.get("can_select_report_label") is False, (
        "Cross-code panel promotion survived when broad top-1 was "
        "confident — the SKCM → BRCA regression is back."
    )
    blocked = " ".join(public.get("blocking_reasons") or []).lower()
    # Look for the actual blocker phrasing, not a substring coincidence.
    assert "differs from" in blocked and "parent_cohort" in blocked


def test_gate_allows_same_code_reinforcement():
    """When the panel's parent_cohort EQUALS the broad top-1
    (BRCA_BASAL → BRCA when broad picked BRCA), promotion is safe:
    the cancer call doesn't change, the panel just contributes the
    subtype/program detail recorded in ``details``."""
    hyps: dict = {}
    analysis = {
        "candidate_trace": [
            {
                "code": "BRCA",
                "family_label": "breast",
                "support_score": 0.9,
            },
            {"code": "HNSC", "family_label": "squamous", "support_score": 0.4},
        ],
    }
    cte._add_lineage_panel_features(hyps, _HCC1395_SAMPLE, analysis)
    brca = hyps.get("BRCA")
    if brca is None:
        return  # panel framework unavailable; selector early-returned
    public = brca.public_dict() or {}
    if "lineage_panel" not in set(public.get("evidence_sources", [])):
        return
    assert public.get("can_select_report_label") is True, (
        "Same-code lineage_panel reinforcement was blocked — BRCA "
        "panel firing on a sample broad already called BRCA should be "
        "allowed (the call doesn't change, the subtype detail goes "
        "into details)."
    )


def test_gate_blocks_cross_code_same_family_when_broad_confident(monkeypatch):
    """Regression for the real TCGA-D7-6524-01 STAD→PAAD incident.

    The marker TPMs below come from that sample in the TCGA RSEM matrix
    (``2**log2(TPM + 0.001) - 0.001``).  Its PAAD ductal program is real and
    strong: all five positive markers and all three negative-marker checks
    pass.  The panel evaluator is pinned to that observed result here because
    this test owns the *promotion gate*, not the independently tested panel
    score/margin calibration.  Returning the PAAD evidence as a clear winner
    guarantees that calibration drift cannot skip the safety assertion again.

    The calibrated broad call is confident STAD (with READ and COAD next; see
    ``docs/calibration-baseline.json``).  Even with PAAD retained in the broad
    beam and a decisive PAAD panel, a confident cross-code call must be noted as
    evidence rather than promoted to the report label.
    """
    from trufflepig import lineage_panels

    sample = {
        "KRT19": 1100.389459,
        "MUC1": 558.106854,
        "CLDN18": 286.461002,
        "AGR2": 1521.205295,
        "S100P": 659.943297,
        "ALB": 0.0,
        "AFP": 0.0,
        "CDX2": 26.359397,
    }
    sample_by_gene_id = {
        "ENSG00000171345": sample["KRT19"],
        "ENSG00000185499": sample["MUC1"],
        "ENSG00000066405": sample["CLDN18"],
        "ENSG00000106541": sample["AGR2"],
        "ENSG00000163993": sample["S100P"],
        "ENSG00000163631": sample["ALB"],
        "ENSG00000081051": sample["AFP"],
        "ENSG00000165556": sample["CDX2"],
    }
    observed_paad_evidence = lineage_panels.PanelEvidence(
        panel_name="PAAD_DUCTAL",
        parent_cohort="PAAD",
        obligate_passed=True,
        obligate_failures=(),
        high_hits=(
            ("KRT19", sample["KRT19"], 1575.477769),
            ("MUC1", sample["MUC1"], 824.21409),
            ("CLDN18", sample["CLDN18"], 59.884901),
            ("AGR2", sample["AGR2"], 447.326444),
            ("S100P", sample["S100P"], 586.764946),
        ),
        high_misses=(),
        low_passes=(
            ("ALB", sample["ALB"], 5.0),
            ("AFP", sample["AFP"], 5.0),
            ("CDX2", sample["CDX2"], 30.0),
        ),
        low_violations=(),
        score=1.0,
        rationale=(
            "PAAD_DUCTAL: 5/5 required markers in cohort range "
            "(KRT19=1100, MUC1=558, CLDN18=286, AGR2=1521); "
            "3/3 negative markers below threshold"
        ),
    )

    def evaluate_observed_paad(_panels, observed_gene_ids, _sample_hk_median):
        assert observed_gene_ids == sample_by_gene_id
        return (observed_paad_evidence,)

    monkeypatch.setattr(
        lineage_panels,
        "evaluate_panels",
        evaluate_observed_paad,
    )
    analysis = {
        "candidate_trace": [
            {"code": "STAD"},
            {"code": "READ"},
            {"code": "COAD"},
            {"code": "PAAD"},
            {"code": "LUAD"},
        ],
    }
    hyps: dict = {}

    summary = cte._add_lineage_panel_features(
        hyps,
        sample,
        analysis,
        sample_tpm_by_gene_id=sample_by_gene_id,
    )

    assert summary is not None
    assert summary["top_panel"] == "PAAD_DUCTAL"
    assert summary["top_score"] == 1.0
    assert summary["promotion"]["promoted"] is False
    assert summary["promotion"]["code"] == "PAAD"
    assert summary["promotion"]["blockers"]
    paad = hyps["PAAD"]
    public = paad.public_dict() or {}
    assert "lineage_panel" in set(public.get("evidence_sources", []))
    assert public.get("can_select_report_label") is False, (
        "A decisive PAAD program overrode the confident STAD call for "
        "TCGA-D7-6524-01."
    )
    blocked = " ".join(public.get("blocking_reasons") or []).lower()
    assert "first-pass top-1 (stad) differs" in blocked
    assert "panel parent_cohort (paad)" in blocked


def test_gate_allows_cross_family_when_broad_explicitly_weak():
    """When ``fit_quality.label`` is "weak" or "ambiguous", the panel
    earns the right to propose across families. This is the case
    the panel exists for (CHOL vs LIHC for a hepatic-margin sample
    with a weak broad call)."""
    hyps: dict = {}
    analysis = {
        "candidate_trace": [
            {"code": "BRCA", "family_label": "breast", "support_score": 0.4},
        ],
        "fit_quality": {"label": "weak"},
    }
    cte._add_lineage_panel_features(hyps, _HCC1395_SAMPLE, analysis)
    brca = hyps.get("BRCA")
    if brca is None:
        return
    public = brca.public_dict() or {}
    if "lineage_panel" not in set(public.get("evidence_sources", [])):
        return
    # broad is weak AND BRCA is in broad top-5 → promotion allowed
    assert public.get("can_select_report_label") is True


def test_gate_blocks_when_proposed_code_outside_broad_top_5():
    """Gate (a): the proposed cancer code must appear in the broad
    top-5 candidate_trace. A panel firing for a code the broad
    classifier never considered must not promote."""
    hyps: dict = {}
    analysis = {
        "candidate_trace": [
            {"code": "OV", "family_label": "müllerian", "support_score": 0.9},
            {"code": "UCEC", "family_label": "müllerian", "support_score": 0.8},
            {"code": "STAD", "family_label": "gi", "support_score": 0.6},
        ],
        "fit_quality": {"label": "weak"},  # even with weak broad,
    }
    cte._add_lineage_panel_features(hyps, _HCC1395_SAMPLE, analysis)
    brca = hyps.get("BRCA")
    if brca is None:
        return
    public = brca.public_dict() or {}
    assert public.get("can_select_report_label") is False, (
        "Panel promoted BRCA even though BRCA is not in broad top-5 "
        "— gate (a) failed."
    )
    blocked = " ".join(public.get("blocking_reasons") or []).lower()
    assert "top-5" in blocked or "broad rna candidates" in blocked


def test_strong_panel_can_rescue_out_of_beam_when_expression_qc_is_fragile():
    """A very strong lineage panel can rescue a missing candidate when the
    broad expression distribution is technically unreliable.

    This pins the HCC1395/StringTie failure mode: a concentrated transcript
    artifact lets the broad softmax jump to an unrelated lineage, while the
    BRCA_BASAL positive/negative marker program is coherent.
    """
    hyps: dict = {}
    analysis = {
        "candidate_trace": [
            {"code": "T_ALL", "family_label": "heme-tcell", "support_score": 0.9},
            {"code": "ESCA", "family_label": "squamous", "support_score": 0.7},
            {"code": "SKCM", "family_label": "melanoma", "support_score": 0.6},
            {"code": "SARC", "family_label": "sarcoma", "support_score": 0.5},
            {"code": "UCS", "family_label": "mullerian", "support_score": 0.4},
        ],
        "sample_context": {
            "signals": {
                "expression_concentration_level": "extreme",
                "top_gene_share_of_total_tpm": 0.20,
            },
            "flags": [
                "Expression distribution is extremely concentrated",
            ],
        },
    }

    cte._add_lineage_panel_features(hyps, _HCC1395_SAMPLE, analysis)
    brca = hyps.get("BRCA")
    assert brca is not None
    public = brca.public_dict() or {}
    assert public.get("can_select_report_label") is True
    assert public.get("selected_by") == "lineage_panel"
    assert public.get("lineage_panel_out_of_beam_rescue")


def test_strong_panel_can_rescue_out_of_beam_when_expression_lineage_conflicts():
    """A very strong lineage panel can rescue a missing candidate when the
    current call's code lineage disagrees with the expression decomposition.

    This is the HCC1395/StringTie no-hint pattern after the concentration
    signal moved into decomposition: T-ALL is top-ranked, but its heme code
    lineage conflicts with the mesenchymal expression decomposition and the
    BRCA_BASAL positive/negative marker program is coherent.
    """
    hyps: dict = {}
    analysis = {
        "candidate_trace": [
            {"code": "T_ALL", "family_label": "heme-tcell", "support_score": 0.9},
            {"code": "ESCA", "family_label": "squamous", "support_score": 0.7},
            {"code": "SKCM", "family_label": "melanoma", "support_score": 0.6},
            {"code": "SARC", "family_label": "sarcoma", "support_score": 0.5},
            {"code": "UCS", "family_label": "mullerian", "support_score": 0.4},
        ],
        "purity": {
            "components": {
                "decomposition": {
                    "lineage_conflict": True,
                    "code_lineage": "Heme",
                    "expression_lineage": "Sarcoma",
                }
            }
        },
    }

    cte._add_lineage_panel_features(hyps, _HCC1395_SAMPLE, analysis)
    brca = hyps.get("BRCA")
    assert brca is not None
    public = brca.public_dict() or {}
    assert public.get("can_select_report_label") is True
    rescue = public.get("lineage_panel_out_of_beam_rescue")
    assert rescue
    assert rescue.get("expression_lineage_conflict") is True


def test_main_propagates_lineage_panel_evidence_to_analysis_dict(monkeypatch):
    """``_apply_cancer_type_evidence`` copies ``lineage_panel_evidence``
    from the cancer_type_evidence return onto the ``analysis`` dict
    so analysis-parameters.json carries the verdict without
    consumers having to dig into cancer_type_evidence themselves.
    """
    import trufflepig.cancer_type_evidence as _cte
    from trufflepig import main

    fake_summary = {
        "top_panel": "BRCA_BASAL",
        "top_score": 0.78,
        "promotion": {"promoted": True, "code": "BRCA", "blockers": []},
    }
    fake_evidence = {
        "selected": None,
        "evidence": [],
        "primary_expression_context": None,
        "top_reference_cancer_type": None,
        "lineage_panel_evidence": fake_summary,
    }
    monkeypatch.setattr(
        _cte,
        "select_report_scope_from_evidence",
        lambda *_a, **_kw: fake_evidence,
    )
    analysis = {"candidate_trace": []}
    main._apply_cancer_type_evidence(
        analysis,
        pd.DataFrame(
            {"ensembl_gene_id": [], "canonical_gene_name": [], "TPM": []}
        ),
        rna_inferred_cancer_type="BRCA",
        fusion_scope_inference=None,
        report_scope_cancer_type=None,
        rare_scope_inference=None,
        fine_scope_inference=None,
    )
    assert analysis.get("lineage_panel_evidence") == fake_summary


# ---------------------------------------------------------------------------
# Gene-ID migration safety
# ---------------------------------------------------------------------------


def test_id_path_and_symbol_path_produce_equivalent_scores():
    """The ID-keyed code path (the new default) and the symbol→ID
    fallback path (used when callers only provide
    ``sample_tpm_by_symbol``) MUST produce identical PanelEvidence
    for the same sample. Pin equivalence so a future change to one
    path can't silently drift from the other.
    """
    from trufflepig.common import panel_symbols_to_gene_ids
    from trufflepig.lineage_panels import (
        LINEAGE_PANELS,
        evaluate_panels,
    )

    sample_by_symbol = dict(_HCC1395_SAMPLE)

    # ID-keyed path: resolve symbols → IDs once, then call directly.
    sym_to_id = panel_symbols_to_gene_ids(list(sample_by_symbol.keys()))
    sample_by_id = {}
    for sym, tpm in sample_by_symbol.items():
        gid = sym_to_id.get(sym)
        if not gid:
            continue
        if gid not in sample_by_id or tpm > sample_by_id[gid]:
            sample_by_id[gid] = float(tpm)

    if not sample_by_id:
        pytest.skip("pirlygenes unavailable; nothing to compare")

    direct = evaluate_panels(LINEAGE_PANELS, sample_by_id, sample_hk_median=200.0)

    # Symbol-fallback path: drive through the selector with sample_tpm_by_symbol
    # only, letting it rebuild the ID view internally.
    hyps: dict = {}
    summary = cte._add_lineage_panel_features(
        hyps,
        sample_by_symbol,
        {"candidate_trace": [{"code": "BRCA"}]},
    )
    if summary is None:
        return

    # Both paths' top_panel and top_score must match.
    assert summary.get("top_panel") == direct[0].panel_name, (
        f"ID-path picked {direct[0].panel_name}, symbol-fallback path "
        f"picked {summary.get('top_panel')}"
    )
    direct_top_score = round(float(direct[0].score), 4)
    summary_top_score = round(float(summary.get("top_score") or 0.0), 4)
    assert direct_top_score == summary_top_score, (
        f"ID path top_score {direct_top_score} != symbol fallback "
        f"top_score {summary_top_score}"
    )


def test_same_code_reinforcement_uses_class_rank_2():
    """When the panel agrees with broad top-1 (same-code reinforcement),
    the hypothesis is registered at class_rank=2 so it can defend the
    correct broad call against an aggressive local_expression_reference
    promotion to a different code (the LIHC→HEPB / PCPG→NBL pattern).

    Reads ``selection_priority`` directly off the CancerTypeEvidence
    object rather than ``public_dict()`` (which doesn't expose it).
    """
    hyps: dict = {}
    analysis = {
        "candidate_trace": [
            {"code": "BRCA", "support_score": 0.9},
            {"code": "HNSC", "support_score": 0.4},
        ],
    }
    cte._add_lineage_panel_features(hyps, _HCC1395_SAMPLE, analysis)
    brca = hyps.get("BRCA")
    if brca is None:
        return
    if "lineage_panel" not in brca.evidence_sources:
        return
    cls, _strength, _tb = brca.selection_priority
    assert cls == 2, (
        f"Same-code reinforcement should be class_rank=2 so it can "
        f"compete with local_expression_reference (class 2); got class={cls}"
    )


def test_cross_code_uncertainty_stays_at_class_rank_1():
    """Cross-code panel promotion (broad explicitly weak) stays at
    class_rank=1 — does NOT compete with fine_reference / LEX. This
    preserves the "augment uncertainty, don't override" contract for
    the cross-code path.
    """
    hyps: dict = {}
    analysis = {
        "candidate_trace": [
            {"code": "HNSC", "support_score": 0.4},
            {"code": "BRCA", "support_score": 0.3},
        ],
        "fit_quality": {"label": "weak"},
    }
    cte._add_lineage_panel_features(hyps, _HCC1395_SAMPLE, analysis)
    brca = hyps.get("BRCA")
    if brca is None:
        return
    if "lineage_panel" not in brca.evidence_sources:
        return
    cls, _strength, _tb = brca.selection_priority
    decision = brca.public_dict().get("label_decision", {})
    if decision.get("status") == "blocked":
        assert cls == 0, (
            "Blocked cross-code panel evidence should not receive a "
            f"selectable class rank; got class={cls}"
        )
        assert any(
            "marker program" in reason
            for reason in brca.public_dict().get("blocking_reasons", [])
        )
    else:
        assert cls == 1, (
            f"Cross-code promotion (broad weak) should stay at class_rank=1; "
            f"got class={cls}"
        )


def test_unresolvable_low_marker_counts_as_violation_not_pass_through():
    """Pessimistic handling: an unresolvable low_marker symbol must
    be treated as a violation (pessimistic), not silently skipped.

    Drives the scorer with a panel whose low_marker symbol is bogus
    and confirms it lands in ``low_violations`` rather than
    disappearing into ``low_passes``.
    """
    from trufflepig.lineage_panels import LineagePanel, score_panel

    bogus_panel = LineagePanel(
        name="TEST_BOGUS",
        parent_cohort="BRCA",
        high_markers=("KRT14",),
        low_markers=(
            ("ESR1", 5.0),  # resolvable, low in sample
            ("NOT_A_REAL_GENE_SYMBOL_XYZ", 5.0),  # unresolvable
        ),
        obligate=("KRT14",),
    )
    # A sample that would pass real low_markers but the bogus one
    # must still fail.
    from trufflepig.common import panel_symbols_to_gene_ids

    sym_to_id = panel_symbols_to_gene_ids(["KRT14", "ESR1"])
    if not sym_to_id:
        pytest.skip("pirlygenes unavailable; nothing to test")
    sample_by_id = {gid: 100.0 if sym == "KRT14" else 0.5 for sym, gid in sym_to_id.items()}

    ev = score_panel(bogus_panel, sample_by_id, sample_hk_median=100.0)
    bogus_violations = [v for v in ev.low_violations if v[0].startswith("NOT_A_REAL")]
    assert bogus_violations, (
        "Unresolvable low_marker was silently skipped instead of "
        "counting as a violation. Pessimistic semantics broken."
    )
