"""Tests for the tissue-composition + cancer-hint gate (#149).

Uses the shipped HPA nTPM + TCGA TPM reference to construct synthetic
samples (pure-tissue + pure-tumor) and asserts the top-3 matches and
the coarse cancer_hint fall where expected.

Does NOT depend on external cohort files so CI can run offline.
"""

import pandas as pd

from trufflepig.reference import pan_cancer_expression
from trufflepig.healthy_vs_tumor import (
    TissueCompositionSignal,
    HealthyVsTumorResult,  # back-compat alias
    assess_tissue_composition,
    assess_healthy_vs_tumor,  # back-compat
    _PROLIFERATION_PANEL,
)


def _ref():
    return pan_cancer_expression().drop_duplicates(subset="Symbol").set_index("Symbol")


def _as_df(sample_dict: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"gene_symbol": list(sample_dict.keys()), "TPM": list(sample_dict.values())}
    )


def test_backcompat_alias_points_to_new_class():
    """Callers using the old name must still find the class."""
    assert HealthyVsTumorResult is TissueCompositionSignal


def test_backcompat_function_dispatches_to_new_impl():
    ref = _ref()
    sample = ref["BRCA_TPM"].astype(float).to_dict()
    for g in _PROLIFERATION_PANEL:
        sample[g] = 300.0
    r_old = assess_healthy_vs_tumor(_as_df(sample))
    r_new = assess_tissue_composition(_as_df(sample))
    assert r_old.cancer_hint == r_new.cancer_hint


def test_brain_sample_routes_to_healthy_dominant_with_brain_tissues_on_top():
    """A synthetic brain sample with quiet proliferation must produce
    a healthy-dominant hint AND the top HPA match must be a brain
    tissue (cerebral_cortex / spinal_cord / cerebellum). This is the
    coarse-to-fine signal that downstream steps read."""
    from trufflepig.healthy_vs_tumor import _ONCOFETAL_STRICT
    from pirlygenes.gene_sets_cancer import glycolysis_panel_gene_names

    ref = _ref()
    sample = ref["cerebral_cortex_nTPM"].astype(float).to_dict()
    for g in _PROLIFERATION_PANEL:
        sample[g] = 0.5
    # Zero the strict oncofetal panel too — HPA brain has trace
    # residual LIN28B / others that would spuriously trip the
    # oncofetal soft-signal and demote healthy-dominant to
    # possibly-tumor. The test targets the correlation-driven
    # healthy call, not the tumor-evidence overrides.
    for g in _ONCOFETAL_STRICT:
        sample[g] = 0.0
    # The test isolates tissue-composition behavior. Clean HPA brain
    # profiles carry high glycolysis-panel expression, which is a separate
    # tumor-evidence channel tested elsewhere.
    for g in glycolysis_panel_gene_names():
        sample[g] = 0.0
    r = assess_tissue_composition(_as_df(sample))
    assert r.cancer_hint == "healthy-dominant", (
        f"synthetic healthy brain should be healthy-dominant; got {r.cancer_hint}"
    )
    top_names = [t for t, _ in r.top_normal_tissues]
    assert any(
        "cerebral" in n or "spinal" in n or "cerebell" in n or "hippocampal" in n
        for n in top_names
    ), f"expected a brain tissue in top HPA matches; got {top_names}"


def test_high_proliferation_overrides_healthy_call():
    """Even when HPA correlation favours normal tissue, a loud
    proliferation panel must produce tumor-consistent. This is the
    primary guard against low-purity normal-like tumors being called
    healthy."""
    ref = _ref()
    sample = ref["liver_nTPM"].astype(float).to_dict()
    for g in _PROLIFERATION_PANEL:
        sample[g] = 500.0
    r = assess_tissue_composition(_as_df(sample))
    assert r.cancer_hint == "tumor-consistent"
    assert r.proliferation_log2_mean > 4.5


def test_top_matches_are_three_entries():
    """The signal must always surface the top-3 matches (when overlap
    is sufficient) so downstream reasoning can enumerate plausible
    tissues + cohorts rather than a single best-guess."""
    ref = _ref()
    sample = ref["breast_nTPM"].astype(float).to_dict()
    for g in _PROLIFERATION_PANEL:
        sample[g] = 1.0
    r = assess_tissue_composition(_as_df(sample))
    assert len(r.top_normal_tissues) == 3
    assert len(r.top_tcga_cohorts) == 3


def test_insufficient_reference_overlap_returns_neutral_signal():
    """If the sample barely overlaps the reference, return a neutral
    tumor-consistent signal (no top matches) so downstream doesn't
    act on noise."""
    sample = {"FAKE_GENE_1": 10.0, "FAKE_GENE_2": 20.0}
    r = assess_tissue_composition(_as_df(sample))
    assert r.cancer_hint == "tumor-consistent"
    assert r.top_normal_tissues == []
    assert r.top_tcga_cohorts == []
    assert "Insufficient" in r.verdict


def test_summary_line_includes_top_tissue_top_cohort_and_hint():
    """The one-liner must carry enough detail to propagate forward:
    top tissue, top cohort, proliferation, hint."""
    r = TissueCompositionSignal(
        top_normal_tissues=[
            ("prostate_nTPM", 0.88),
            ("seminal_vesicle_nTPM", 0.85),
            ("smooth_muscle_nTPM", 0.82),
        ],
        top_tcga_cohorts=[("PRAD_TPM", 0.78), ("BRCA_TPM", 0.75), ("OV_TPM", 0.74)],
        proliferation_log2_mean=2.1,
        proliferation_genes_observed=5,
        cancer_hint="possibly-tumor",
        n_reference_genes=5000,
        verdict="",
    )
    line = r.summary_line()
    assert "prostate" in line
    assert "PRAD" in line
    assert "possibly-tumor" in line
    assert "2.1" in line  # proliferation


def test_brief_banner_fires_for_healthy_and_ambiguous_hints():
    for hint, expect_fire in [
        ("tumor-consistent", False),
        ("possibly-tumor", True),
        ("healthy-dominant", True),
    ]:
        r = TissueCompositionSignal(
            top_normal_tissues=[
                ("liver_nTPM", 0.9),
                ("gallbladder_nTPM", 0.85),
                ("pancreas_nTPM", 0.8),
            ],
            top_tcga_cohorts=[
                ("LIHC_TPM", 0.78),
                ("CHOL_TPM", 0.74),
                ("PAAD_TPM", 0.72),
            ],
            proliferation_log2_mean=1.0,
            proliferation_genes_observed=5,
            cancer_hint=hint,
            n_reference_genes=5000,
            verdict="",
        )
        banner = r.brief_banner()
        if expect_fire:
            assert banner is not None, f"banner should fire for hint={hint}"
            assert "liver" in banner
        else:
            assert banner is None, f"banner should NOT fire for hint={hint}"


def test_structural_banner_respects_active_report_label():
    r = TissueCompositionSignal(
        top_normal_tissues=[("smooth_muscle_nTPM", 0.9)],
        top_tcga_cohorts=[("SARC_TPM", 0.82), ("READ_TPM", 0.81)],
        proliferation_log2_mean=4.0,
        proliferation_genes_observed=13,
        cancer_hint="possibly-tumor",
        structural_ambiguity=True,
        n_reference_genes=5000,
        verdict="",
    )

    banner = r.brief_banner(
        active_cancer_code="READ",
        active_cancer_label="READ (Rectum Adenocarcinoma)",
    )

    assert "retained as background/differential context" in banner
    assert "downstream report sections use READ (Rectum Adenocarcinoma)" in banner
    assert "SARC-specific downstream analysis" not in banner
