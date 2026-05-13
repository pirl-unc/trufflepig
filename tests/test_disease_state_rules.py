"""Tests for the data-driven disease-state narrative engine (#202).

Pins the narratives previously produced by the hardcoded PRAD / BRCA
branches in ``cli._synthesize_disease_state``, plus the pan-cancer
EMT/hypoxia/IFN rules. Also exercises the rule engine primitives
(condition DSL, template rendering, priority/claims) so regressions
in the engine surface independently of the catalog content.
"""

import pandas as pd

from trufflepig.disease_state_rules import (
    _all_rules,
    _conditions_match,
    _render_narrative,
    narrative_gene_sets,
    synthesize_disease_state,
)


# ── Data loading ─────────────────────────────────────────────────────


def test_narrative_gene_sets_loads():
    sets = narrative_gene_sets()
    # AR_targets is the validation anchor for CRPC pattern detection
    # and must not drift silently. The 10-gene panel is the one the
    # hardcoded cli code used pre-#202.
    assert "AR_targets" in sets
    ar_targets = set(sets["AR_targets"])
    for g in (
        "KLK3",
        "KLK2",
        "NKX3-1",
        "HOXB13",
        "FOLH1",
        "TMPRSS2",
        "SLC45A3",
        "NDRG1",
        "PMEPA1",
        "FKBP5",
    ):
        assert g in ar_targets, f"AR_targets missing {g}"
    # HER2 amplicon panel used by the BRCA rule's narrative string.
    assert "HER2_amplicon" in sets
    assert "ERBB2" in sets["HER2_amplicon"]


def test_rule_csv_loads_and_priority_order():
    rules = _all_rules()
    assert len(rules) >= 6
    # Pan-cancer rules and cancer-specific rules coexist.
    cancers = {r.cancer_code for r in rules}
    assert "pan_cancer" in cancers
    assert "PRAD" in cancers
    assert "BRCA" in cancers


# ── Condition DSL ────────────────────────────────────────────────────


def test_axis_state_condition():
    assert _conditions_match(
        "axis:AR_signaling=down",
        {"AR_signaling": "down"},
        retained=set(),
        collapsed=set(),
    )
    assert not _conditions_match(
        "axis:AR_signaling=down",
        {"AR_signaling": "up"},
        retained=set(),
        collapsed=set(),
    )
    # Missing axis reads as None and fails.
    assert not _conditions_match(
        "axis:AR_signaling=down",
        {},
        retained=set(),
        collapsed=set(),
    )


def test_retained_and_collapsed_tokens():
    # retained:<gene> — single gene
    assert _conditions_match(
        "retained:AR",
        axis_states={},
        retained={"AR"},
        collapsed=set(),
    )
    # retained:<set_name> — any-of semantics
    assert _conditions_match(
        "retained:AR_targets",
        axis_states={},
        retained={"KLK3"},
        collapsed=set(),
    )
    # collapsed:<gene>
    assert _conditions_match(
        "collapsed:ESR1",
        axis_states={},
        retained=set(),
        collapsed={"ESR1"},
    )
    # collapsed_ge:N=SET
    assert _conditions_match(
        "collapsed_ge:3=AR_targets",
        axis_states={},
        retained=set(),
        collapsed={"KLK3", "KLK2", "FKBP5"},
    )
    assert not _conditions_match(
        "collapsed_ge:3=AR_targets",
        axis_states={},
        retained=set(),
        collapsed={"KLK3", "KLK2"},  # only 2
    )


def test_pipe_is_and_all_subconditions_must_match():
    """Pipe separator means AND; every subcondition must match."""
    condition = "axis:AR_signaling=down | retained:AR | collapsed_ge:3=AR_targets"
    axis = {"AR_signaling": "down"}
    # All three met
    assert _conditions_match(
        condition,
        axis,
        {"AR"},
        {"KLK3", "KLK2", "FKBP5", "TMPRSS2"},
    )
    # AR_signaling missing
    assert not _conditions_match(
        condition,
        {},
        {"AR"},
        {"KLK3", "KLK2", "FKBP5"},
    )
    # AR not retained
    assert not _conditions_match(
        condition,
        axis,
        set(),
        {"KLK3", "KLK2", "FKBP5"},
    )


def test_unknown_token_fails_closed():
    """Typos in CSVs must fail the match, not silently pass."""
    assert not _conditions_match(
        "unknown_op:AR",
        {},
        set(),
        set(),
    )


# ── Template rendering ───────────────────────────────────────────────


def test_collapsed_placeholder_expands_in_declared_order():
    """Placeholders render members in the gene-set's declared order
    (not set-iteration order, which is nondeterministic)."""
    collapsed = {"TMPRSS2", "KLK3", "KLK2"}
    out = _render_narrative(
        "targets ({collapsed:AR_targets}) are collapsed",
        retained=set(),
        collapsed=collapsed,
    )
    # CSV order is KLK3;KLK2;NKX3-1;HOXB13;FOLH1;TMPRSS2;... so the
    # rendered list should follow that order for the subset present.
    assert "KLK3, KLK2, TMPRSS2" in out


def test_retained_placeholder_expands():
    out = _render_narrative(
        "retained: {retained:AR_targets}",
        retained={"KLK3"},
        collapsed=set(),
    )
    assert "retained: KLK3" in out


def test_empty_placeholder_renders_empty_string():
    out = _render_narrative(
        "({collapsed:AR_targets})",
        retained=set(),
        collapsed=set(),
    )
    assert out == "()"


# ── Cancer-specific narrative synthesis ──────────────────────────────


def test_prad_castrate_resistant_narrative():
    """PRAD validation pattern: AR retained + AR-target panel
    collapsed + AR axis down → textbook castrate-resistant narrative."""
    result = synthesize_disease_state(
        cancer_code="PRAD",
        axis_states={"AR_signaling": "down", "NE_differentiation": None},
        retained={"AR"},
        collapsed={"KLK3", "KLK2", "FKBP5", "TMPRSS2", "NKX3-1"},
    )
    assert "Castrate-resistant pattern" in result
    assert "KLK3" in result  # collapsed targets rendered
    assert "prior ADT exposure" in result


def test_prad_castrate_resistant_with_ne_emerges():
    """Same as above + NE axis up → emerging-NEPC narrative."""
    result = synthesize_disease_state(
        cancer_code="PRAD",
        axis_states={"AR_signaling": "down", "NE_differentiation": "up"},
        retained={"AR"},
        collapsed={"KLK3", "KLK2", "FKBP5", "TMPRSS2", "NKX3-1"},
    )
    assert "emerging neuroendocrine differentiation" in result
    assert "NEPC" in result


def test_prad_ar_suppressed_fallback():
    """AR axis down but AR not retained → "AR axis suppressed"
    fallback narrative (the castrate-resistant rules don't match)."""
    result = synthesize_disease_state(
        cancer_code="PRAD",
        axis_states={"AR_signaling": "down"},
        retained=set(),  # AR NOT retained
        collapsed={"KLK3"},  # only one collapsed, fails the >=3 rule
    )
    assert "AR axis suppressed" in result
    # And the more-specific castrate-resistant narrative didn't fire
    assert "Castrate-resistant pattern" not in result


def test_prad_claims_prevents_double_emit():
    """Priority + claims: when castrate-resistant rule fires, the
    lower-priority AR-suppressed rule must NOT also emit — both claim
    ``AR_axis``."""
    result = synthesize_disease_state(
        cancer_code="PRAD",
        axis_states={"AR_signaling": "down"},
        retained={"AR"},
        collapsed={"KLK3", "KLK2", "FKBP5"},
    )
    assert "Castrate-resistant pattern" in result
    assert "AR axis suppressed" not in result


def test_brca_her2_amplification_pattern():
    result = synthesize_disease_state(
        cancer_code="BRCA",
        axis_states={"HER2_signaling": "up"},
        retained=set(),
        collapsed=set(),
    )
    assert "HER2-amplification pattern" in result


def test_brca_er_suppressed_pattern():
    result = synthesize_disease_state(
        cancer_code="BRCA",
        axis_states={"ER_signaling": "down"},
        retained=set(),
        collapsed={"ESR1"},
    )
    assert "ER-axis suppressed" in result
    assert "ESR1" in result


# ── Pan-cancer rules ─────────────────────────────────────────────────


def test_pan_emt_hypoxia_claims_emt_axis():
    """When both EMT + hypoxia are up, the combined narrative fires;
    the lower-priority EMT-only rule must NOT also fire."""
    result = synthesize_disease_state(
        cancer_code="SARC",
        axis_states={"EMT": "up", "hypoxia": "up"},
        retained=set(),
        collapsed=set(),
    )
    assert "EMT and hypoxia programs are both active" in result
    assert "mesenchymal switch" not in result


def test_pan_emt_only_fires_when_hypoxia_absent():
    result = synthesize_disease_state(
        cancer_code="SARC",
        axis_states={"EMT": "up", "hypoxia": None},
        retained=set(),
        collapsed=set(),
    )
    assert "EMT program is active (mesenchymal switch)" in result


def test_pan_ifn_fires_on_any_cancer():
    """IFN rule has no ``cancer_code`` restriction — applies to every
    sample whose IFN_response axis is up."""
    result = synthesize_disease_state(
        cancer_code="READ",
        axis_states={"IFN_response": "up"},
        retained=set(),
        collapsed=set(),
    )
    assert "Active IFN response" in result


def test_pan_rules_apply_even_without_cancer_code():
    """Pan-cancer rules must fire for the ``None`` cancer_code case
    (e.g. when the classifier couldn't commit to a code)."""
    result = synthesize_disease_state(
        cancer_code=None,
        axis_states={"IFN_response": "up"},
        retained=set(),
        collapsed=set(),
    )
    assert "Active IFN response" in result


# ── Uniformity / regression ──────────────────────────────────────────


def test_unknown_cancer_code_runs_only_pan_cancer_rules():
    """A cancer the catalog doesn't name should still get pan-cancer
    narratives — no silent skip."""
    result = synthesize_disease_state(
        cancer_code="NUTM",  # not referenced in disease-state-rules.csv
        axis_states={"EMT": "up"},
        retained=set(),
        collapsed=set(),
    )
    assert "EMT program is active" in result


def test_empty_axes_empty_result():
    result = synthesize_disease_state(
        cancer_code="PRAD",
        axis_states={},
        retained=set(),
        collapsed=set(),
    )
    assert result == ""


def test_every_rule_id_and_cancer_code_valid():
    """Catch malformed rows in the CSV."""
    rules = _all_rules()
    seen_ids = set()
    for r in rules:
        assert r.rule_id and r.rule_id not in seen_ids, (
            f"rule_id {r.rule_id!r} is empty or duplicate"
        )
        seen_ids.add(r.rule_id)
        assert r.cancer_code, f"rule {r.rule_id} has empty cancer_code"
        assert r.narrative, f"rule {r.rule_id} has empty narrative"


def test_no_prad_ar_target_genes_hardcoded_anymore():
    """Ensure the old frozenset ``_PRAD_AR_TARGET_GENES`` is gone.
    If someone reinstates it they bring back the asymmetry this PR
    was trying to remove."""
    from trufflepig import main as cli

    assert not hasattr(cli, "_PRAD_AR_TARGET_GENES"), (
        "Hardcoded PRAD AR target set re-appeared in cli; move it to "
        "narrative-gene-sets.csv per #202."
    )


def test_narrative_gene_sets_csv_headers():
    """Schema contract for the gene-sets CSV — no silent drift."""
    from trufflepig._data import DATA_DIR

    df = pd.read_csv(DATA_DIR / "narrative-gene-sets.csv")
    assert set(df.columns) >= {"set_name", "members"}


def test_disease_state_rules_csv_headers():
    from trufflepig._data import DATA_DIR

    df = pd.read_csv(DATA_DIR / "disease-state-rules.csv")
    assert set(df.columns) == {
        "rule_id",
        "cancer_code",
        "priority",
        "claims",
        "conditions",
        "narrative",
    }
