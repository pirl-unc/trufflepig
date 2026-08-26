"""Smoke tests for the provenance page (#106)."""

import pandas as pd

from trufflepig.provenance import (
    build_provenance_md as _build_provenance_md,
    plot_provenance_funnel as _plot_provenance_funnel,
)
from trufflepig.report_view import build_report_view


def _view(analysis, cancer_code="UNKNOWN"):
    finalized = dict(analysis)
    finalized.setdefault("cancer_type", cancer_code)
    finalized.setdefault("sample_mode", "solid")
    finalized.setdefault("purity", {})
    return build_report_view(finalized)


def build_provenance_md(analysis, *args, **kwargs):
    if "report_view" not in kwargs:
        kwargs["report_view"] = _view(
            analysis,
            kwargs.get("cancer_code") or "UNKNOWN",
        )
    return _build_provenance_md(analysis, *args, **kwargs)


def plot_provenance_funnel(analysis, *args, **kwargs):
    if "report_view" not in kwargs:
        kwargs["report_view"] = _view(analysis)
    return _plot_provenance_funnel(analysis, *args, **kwargs)


class _Ctx:
    def __init__(
        self,
        prep="exome_capture",
        preservation="ffpe",
        severity="mild",
        index=0.55,
        confidence=0.9,
    ):
        self.library_prep = prep
        self.library_prep_confidence = confidence
        self.preservation = preservation
        self.preservation_confidence = 0.85
        self.degradation_severity = severity
        self.degradation_index = index
        self.missing_mt = False
        self.signals = {}
        self.flags = []


class _Decomp:
    def __init__(self, purity=0.28):
        self.purity = purity
        self.fractions = {
            "tumor": purity,
            "T_cell": 0.03,
            "endothelial": 0.03,
            "fibroblast": 0.25,
            "myeloid": 0.15,
            "matched_normal_prostate": 0.26,
        }
        self.component_trace = pd.DataFrame(
            [
                {"component": "T_cell", "fraction": 0.03},
                {"component": "matched_normal_prostate", "fraction": 0.26},
            ]
        )


def _ranges_df():
    return pd.DataFrame(
        [
            {
                "symbol": "FOLH1",
                "observed_tpm": 142.0,
                "attribution": {"endothelial": 12.0},
                "attr_tumor_tpm": 128.0,
                "attr_tumor_fraction": 0.90,
            },
            {
                "symbol": "STEAP1",
                "observed_tpm": 78.0,
                "attribution": {"fibroblast": 10.0},
                "attr_tumor_tpm": 62.0,
                "attr_tumor_fraction": 0.79,
            },
        ]
    )


def test_provenance_md_walks_the_five_steps():
    analysis = {
        "sample_context": _Ctx(),
        "purity": {
            "overall_estimate": 0.28,
            "overall_lower": 0.19,
            "overall_upper": 0.40,
        },
    }
    md = build_provenance_md(
        analysis,
        _ranges_df(),
        [_Decomp()],
        cancer_code="PRAD",
        sample_id="sample_X",
    )
    # Each of the five attribution steps must render its section heading.
    for heading in [
        "## RNA Prep and Preservation",
        "### Preservation and Degradation",
        "## Tumor Purity and Coarse Composition",
        "## Subtype and Background Refinements",
        "## Tumor-Attributed Expression",
    ]:
        assert heading in md, f"missing step heading: {heading}"
    assert "RNA hybrid-capture" in md
    assert "FOLH1" in md or "tumor-linked" in md.lower()


def test_provenance_composition_uses_finalized_headline_not_decomp_purity():
    """The decomposition's own tumor fraction (0.28, frozen at fit time) can
    differ from the finalized/fused headline purity (0.60). The coarse
    composition must display the headline so 'Fitted fractions: tumor X%'
    cannot contradict the reported purity (#85.1)."""
    analysis = {
        "sample_context": _Ctx(),
        "purity": {
            "overall_estimate": 0.60,
            "overall_lower": 0.45,
            "overall_upper": 0.75,
        },
    }
    md = build_provenance_md(
        analysis,
        _ranges_df(),
        [_Decomp(purity=0.28)],
        cancer_code="PRAD",
        sample_id="sample_X",
    )
    # tumor% equals the finalized headline (60%), not the frozen decomp (28%).
    assert "**tumor 60%**" in md
    assert "tumor 28%" not in md
    # The Chain summary's non-tumor figure must equal 1 - headline (40%), not
    # 1 - decomp (72%), so the two purity-bearing lines agree on one page.
    assert "subtracts 40% as non-tumor" in md
    assert "subtracts 72% as non-tumor" not in md


def test_provenance_composition_falls_back_to_decomp_when_no_headline():
    """With no finalized purity available, the composition still renders using
    the decomposition's own tumor fraction (no crash, no rescale)."""
    analysis = {"sample_context": _Ctx(), "purity": {}}
    md = build_provenance_md(
        analysis,
        _ranges_df(),
        [_Decomp(purity=0.28)],
        cancer_code="PRAD",
    )
    assert "**tumor 28%**" in md


def test_provenance_abstains_from_consensus_composition_when_purity_is_unresolved(
    tmp_path,
):
    from trufflepig.report_view import build_report_view

    analysis = {
        "sample_context": _Ctx(),
        "cancer_type": "READ",
        "cancer_name": "Rectum Adenocarcinoma",
        "sample_mode": "solid",
        "purity": {
            "overall_estimate": 0.05,
            "overall_lower": 0.01,
            "overall_upper": 0.12,
            "quantitative_status": "discordant_estimators",
            "estimator_scenarios": [
                {
                    "source": "lineage_panel",
                    "estimate": 0.05,
                    "lower": 0.01,
                    "upper": 0.12,
                },
                {
                    "source": "signature",
                    "estimate": 0.43,
                    "lower": 0.32,
                    "upper": 0.55,
                },
            ],
        },
    }
    report_view = build_report_view(analysis)

    md = build_provenance_md(
        analysis,
        _ranges_df(),
        [_Decomp(purity=0.05)],
        cancer_code="READ",
        report_view=report_view,
    )

    assert "**Quantitative purity is unresolved.**" in md
    assert "matched-normal lineage model: 5% [1%–12%]" in md
    assert "upstream expression model: 43% [32%–55%]" in md
    assert "operational tumor model 5%" in md
    assert "not a resolved sample-composition measurement" in md
    assert "subtracts 95% as non-tumor" not in md
    assert "quantitative tumor/non-tumor split remains unresolved" in md

    out = tmp_path / "unresolved-provenance.png"
    plot_provenance_funnel(
        analysis,
        _ranges_df(),
        [_Decomp(purity=0.05)],
        save_to_filename=str(out),
        report_view=report_view,
    )
    assert out.exists() and out.stat().st_size > 1000


def test_provenance_handles_missing_decomposition():
    analysis = {
        "sample_context": _Ctx(),
        "purity": {
            "overall_estimate": 0.28,
            "overall_lower": 0.19,
            "overall_upper": 0.40,
        },
    }
    md = build_provenance_md(
        analysis,
        _ranges_df(),
        decomp_results=[],
        cancer_code="PRAD",
    )
    # Still renders — just with a no-decomposition note.
    assert "No decomposition result" in md


def test_provenance_distinguishes_tumor_supported_from_mixed_source_core():
    analysis = {
        "sample_context": _Ctx(),
        "purity": {
            "overall_estimate": 0.28,
            "overall_lower": 0.19,
            "overall_upper": 0.40,
        },
    }
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "SAFE",
                "observed_tpm": 142.0,
                "attribution": {"endothelial": 12.0},
                "attr_tumor_tpm": 128.0,
                "attr_tumor_fraction": 0.90,
                "tme_dominant": False,
                "matched_normal_over_predicted": False,
                "smooth_muscle_stromal_leakage": False,
                "broadly_expressed": False,
                "tme_explainable": False,
                "low_purity_cap_applied": False,
            },
            {
                "symbol": "MIXED",
                "observed_tpm": 78.0,
                "attribution": {"matched_normal_prostate": 48.0},
                "attr_tumor_tpm": 26.0,
                "attr_tumor_fraction": 0.33,
                "attr_tumor_tpm_low": 8.0,
                "attr_tumor_tpm_high": 36.0,
                "attr_tumor_fraction_low": 0.10,
                "attr_tumor_fraction_high": 0.46,
                "attr_support_fraction": 0.33,
                "tme_dominant": False,
                "matched_normal_over_predicted": True,
                "smooth_muscle_stromal_leakage": False,
                "broadly_expressed": False,
                "tme_explainable": True,
                "low_purity_cap_applied": False,
            },
        ]
    )
    md = build_provenance_md(
        analysis,
        ranges_df,
        [_Decomp()],
        cancer_code="PRAD",
        sample_id="sample_X",
    )
    assert (
        "**1 genes** retain ≥1 TPM of tumor-supported tumor-attributed expression."
        in md
    )
    assert "additional **1 genes** retain residual tumor-attributed TPM" in md
    assert "SAFE (128)" in md
    assert "MIXED (26)" not in md


def test_provenance_funnel_renders_png(tmp_path):
    analysis = {
        "sample_context": _Ctx(),
        "purity": {
            "overall_estimate": 0.28,
            "overall_lower": 0.19,
            "overall_upper": 0.40,
        },
    }
    out = tmp_path / "prov.png"
    result = plot_provenance_funnel(
        analysis,
        _ranges_df(),
        [_Decomp()],
        save_to_filename=str(out),
    )
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 1000
