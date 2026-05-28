"""Tests for the tissue-composition reasoning rules as standalone pure functions.

Each rule in :mod:`pirlygenes.reasoning` takes a
``TissueCompositionSignal`` and a pre-computed :class:`DerivedFlags`,
and returns a :class:`RuleOutcome` or ``None``. The rules are testable
without running the full ``assess_tissue_composition`` pipeline —
these tests cover each rule's contract individually.
"""

from dataclasses import replace

from trufflepig.healthy_vs_tumor import TissueCompositionSignal
from trufflepig.reasoning import (
    aggregate_tumor_evidence,
    compute_derived_flags,
    confident_healthy_tissue,
    healthy_with_soft_tumor_signal,
    high_proliferation_panel,
    lymphoid_tissue_ambiguity,
    mesenchymal_tissue_ambiguity,
    near_exact_normal_reference_match,
    run_step0_rules,
    cancer_reference_dominant_correlation,
    tumor_marker_overrides_ambiguity,
    weak_healthy_lean,
)
from trufflepig.tumor_evidence import TumorEvidenceScore


def _make_signal(**kwargs):
    """Build a minimal ``TissueCompositionSignal`` for testing."""
    defaults = dict(
        top_normal_tissues=[],
        top_tcga_cohorts=[],
        proliferation_log2_mean=0.0,
        proliferation_genes_observed=0,
        cancer_hint="tumor-consistent",
        n_reference_genes=5000,
        verdict="",
        structural_ambiguity=False,
        cta_panel_sum_tpm=0.0,
        cta_count_above_1_tpm=0,
        cta_top_hits=[],
        oncofetal_count_above_threshold=0,
        oncofetal_top_hits=[],
        type_specific_cohort="",
        type_specific_hits=[],
        evidence=TumorEvidenceScore(),
    )
    defaults.update(kwargs)
    return TissueCompositionSignal(**defaults)


def _flags(signal, **overrides):
    """Compute flags for a signal, optionally overriding ambiguity fields."""
    f = compute_derived_flags(signal)
    if overrides:
        f = replace(f, **overrides)
    return f


# ---------- tumor_marker_overrides_ambiguity ----------


def test_tumor_marker_overrides_ambiguity_fires_on_strong_cta_in_lymphoid():
    s = _make_signal(cta_count_above_1_tpm=10, cta_panel_sum_tpm=500)
    f = _flags(s, lymphoid_ambiguity=True)
    out = tumor_marker_overrides_ambiguity(s, f)
    assert out is not None
    assert out.hint == "tumor-consistent"
    assert out.rule_name == "tumor-marker-overrides-ambiguity"


def test_tumor_marker_overrides_ambiguity_skips_without_ambiguity_flag():
    s = _make_signal(cta_count_above_1_tpm=10)
    f = _flags(s)
    assert tumor_marker_overrides_ambiguity(s, f) is None


def test_tumor_marker_overrides_ambiguity_skips_without_strong_signal():
    s = _make_signal(cta_count_above_1_tpm=1)  # below strong threshold
    f = _flags(s, lymphoid_ambiguity=True)
    assert tumor_marker_overrides_ambiguity(s, f) is None


# ---------- lymphoid / mesenchymal ambiguity ----------


def test_lymphoid_ambiguity_fires_and_marks_structural():
    s = _make_signal(
        top_normal_tissues=[("lymph_node_nTPM", 0.82)],
        top_tcga_cohorts=[("DLBC_TPM", 0.83)],
    )
    f = _flags(s, lymphoid_ambiguity=True)
    out = lymphoid_tissue_ambiguity(s, f)
    assert out is not None
    assert out.hint == "possibly-tumor"
    assert out.structural is True


def test_mesenchymal_ambiguity_fires_and_marks_structural():
    s = _make_signal(
        top_normal_tissues=[("smooth_muscle_nTPM", 0.85)],
        top_tcga_cohorts=[("SARC_TPM", 0.80)],
    )
    f = _flags(s, mesenchymal_ambiguity=True)
    out = mesenchymal_tissue_ambiguity(s, f)
    assert out is not None
    assert out.hint == "possibly-tumor"
    assert out.structural is True


def test_ambiguity_rules_skip_when_not_ambiguous():
    s = _make_signal()
    f = _flags(s)
    assert lymphoid_tissue_ambiguity(s, f) is None
    assert mesenchymal_tissue_ambiguity(s, f) is None


# ---------- aggregate_tumor_evidence ----------


def test_aggregate_tumor_evidence_fires_on_aggregate_above_1():
    evidence = TumorEvidenceScore(cta=0.5, oncofetal=0.3, proliferation=0.4)
    assert evidence.aggregate_score >= 1.0
    s = _make_signal(evidence=evidence)
    f = _flags(s)
    out = aggregate_tumor_evidence(s, f)
    assert out is not None
    assert out.hint == "tumor-consistent"
    assert "aggregate" in out.rationale


def test_aggregate_tumor_evidence_fires_on_strong_single_cta():
    s = _make_signal(cta_count_above_1_tpm=10)
    f = _flags(s)
    out = aggregate_tumor_evidence(s, f)
    assert out is not None
    assert "CTA_strong" in out.rationale


def test_single_high_cta_gene_is_not_strong_by_itself():
    s = _make_signal(cta_count_above_1_tpm=1, cta_panel_sum_tpm=100.0)
    f = _flags(s)
    assert f.cta_strong is False


def test_aggregate_tumor_evidence_skips_below_thresholds():
    s = _make_signal(
        cta_count_above_1_tpm=1,
        evidence=TumorEvidenceScore(cta=0.2),  # aggregate only 0.2
    )
    f = _flags(s)
    assert aggregate_tumor_evidence(s, f) is None


# ---------- near_exact_normal_reference_match ----------


def test_near_exact_normal_reference_match_beats_panel_noise():
    s = _make_signal(
        top_normal_tissues=[("colon_nTPM", 0.999)],
        top_tcga_cohorts=[("COAD_TPM", 0.86)],
        cta_count_above_1_tpm=5,
        oncofetal_count_above_threshold=2,
        type_specific_hits=[("A", 5.0), ("B", 4.0)],
        evidence=TumorEvidenceScore(
            cta=1.0,
            oncofetal=1.0,
            type_specific=1.0,
            glycolysis=0.5,
            prolif_log2=2.4,
        ),
    )
    f = _flags(s)
    out = near_exact_normal_reference_match(s, f)
    assert out is not None
    assert out.hint == "healthy-dominant"


def test_near_exact_normal_reference_match_with_high_proliferation_is_ambiguous():
    s = _make_signal(
        top_normal_tissues=[("rectum_nTPM", 0.999)],
        top_tcga_cohorts=[("READ_TPM", 0.85)],
        evidence=TumorEvidenceScore(prolif_log2=4.8),
    )
    f = _flags(s)
    out = near_exact_normal_reference_match(s, f)
    assert out is not None
    assert out.hint == "possibly-tumor"


# ---------- high_proliferation_panel ----------


def test_high_proliferation_fires_when_panel_elevated():
    s = _make_signal(evidence=TumorEvidenceScore(prolif_log2=5.0))
    f = _flags(s)
    out = high_proliferation_panel(s, f)
    assert out is not None
    assert out.hint == "tumor-consistent"


def test_high_proliferation_skips_when_quiet():
    s = _make_signal(evidence=TumorEvidenceScore(prolif_log2=2.0))
    f = _flags(s)
    assert high_proliferation_panel(s, f) is None


# ---------- confident_healthy_tissue / healthy_with_soft_tumor_signal ----------


def test_confident_healthy_fires_on_clean_healthy_profile():
    s = _make_signal(
        top_normal_tissues=[("cerebral_cortex_nTPM", 0.95)],
        top_tcga_cohorts=[("LGG_TPM", 0.85)],
        evidence=TumorEvidenceScore(prolif_log2=0.5),
    )
    f = _flags(s)
    out = confident_healthy_tissue(s, f)
    assert out is not None
    assert out.hint == "healthy-dominant"


def test_healthy_with_soft_tumor_signal_demotes_on_soft_cta():
    s = _make_signal(
        top_normal_tissues=[("cerebral_cortex_nTPM", 0.95)],
        top_tcga_cohorts=[("LGG_TPM", 0.85)],
        evidence=TumorEvidenceScore(prolif_log2=0.5),
        cta_count_above_1_tpm=3,  # soft CTA
    )
    f = _flags(s)
    out = healthy_with_soft_tumor_signal(s, f)
    assert out is not None
    assert out.hint == "possibly-tumor"
    assert "CTA_soft" in out.rationale


# ---------- weak_healthy_lean / cancer_reference_dominant_correlation ----------


def test_weak_healthy_lean_fires_on_weak_margin():
    s = _make_signal(
        top_normal_tissues=[("lung_nTPM", 0.82)],
        top_tcga_cohorts=[("LUAD_TPM", 0.79)],
    )
    f = _flags(s)
    out = weak_healthy_lean(s, f)
    assert out is not None
    assert out.hint == "possibly-tumor"


def test_cancer_reference_dominant_correlation_is_unconditional_default():
    s = _make_signal()
    f = _flags(s)
    out = cancer_reference_dominant_correlation(s, f)
    assert out is not None
    assert out.hint == "tumor-consistent"


# ---------- Rule runner ----------


def test_run_step0_rules_picks_first_match():
    """Rule runner returns the first rule that fires — ordering matters."""
    s = _make_signal(cta_count_above_1_tpm=10)
    f = _flags(s, lymphoid_ambiguity=True)
    outcome, trace = run_step0_rules(s, f)
    # tumor_marker_overrides_ambiguity is first and matches.
    assert outcome.rule_name == "tumor-marker-overrides-ambiguity"
    assert outcome.hint == "tumor-consistent"


def test_run_step0_rules_falls_to_default_when_nothing_else_matches():
    """Rule runner reaches the cancer-reference-dominant default when no earlier
    rule fires."""
    s = _make_signal()  # no tumor evidence, no ambiguity, no healthy margin
    f = _flags(s)
    outcome, trace = run_step0_rules(s, f)
    assert outcome.rule_name == "cancer-reference-dominant-correlation"
    assert outcome.hint == "tumor-consistent"


def test_run_step0_rules_accepts_custom_rule_order():
    """Custom rule order is honored."""
    s = _make_signal(cta_count_above_1_tpm=10)
    f = _flags(s, lymphoid_ambiguity=True)
    # Put lymphoid first — it would then fire before the single-channel
    # override.
    custom_order = [
        lymphoid_tissue_ambiguity,
        tumor_marker_overrides_ambiguity,
        cancer_reference_dominant_correlation,
    ]
    outcome, trace = run_step0_rules(s, f, rules=custom_order)
    assert outcome.rule_name == "lymphoid-tissue-tumor-indistinguishable"


def test_run_step0_rules_auto_computes_flags_when_omitted():
    """When ``flags`` is omitted the runner pre-computes them."""
    s = _make_signal(cta_count_above_1_tpm=10)
    outcome, trace = run_step0_rules(s)
    assert outcome.hint == "tumor-consistent"
    assert outcome.rule_name == "aggregate-tumor-evidence"
