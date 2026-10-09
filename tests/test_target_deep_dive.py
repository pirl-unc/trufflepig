# Licensed under the Apache License, Version 2.0

"""Tests for therapy target deep-dive and subtype signature plots."""

import pandas as pd

from trufflepig.reference import pan_cancer_expression
from trufflepig.plot_target_deep_dive import (
    actionable_surface_targets,
    plot_actionable_targets,
    plot_curated_target_evidence,
    plot_priority_target_context,
    plot_priority_targets,
    plot_tumor_attribution,
    plot_cta_deep_dive,
    _CANCER_SURFACE_TARGETS,
)
from trufflepig.plot_subtype_signature import (
    SUBTYPE_CONTRASTS,
    plot_subtype_signature,
)


def _tcga_sample(cancer_code):
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{cancer_code}_TPM"].astype(float),
        }
    )


# ── actionable_surface_targets ───────────────────────────────────────────


def test_prad_targets_include_psma():
    targets = actionable_surface_targets("PRAD")
    assert "FOLH1" in targets


def test_brca_targets_include_her2():
    targets = actionable_surface_targets("BRCA")
    assert "ERBB2" in targets


def test_unknown_type_uses_default():
    targets = actionable_surface_targets("ACC")
    assert len(targets) > 0


def test_curated_panels_cover_major_types():
    for code in ["PRAD", "BRCA", "LUAD", "COAD", "SKCM", "OV", "LIHC", "GBM", "BLCA"]:
        assert code in _CANCER_SURFACE_TARGETS
        assert len(_CANCER_SURFACE_TARGETS[code]) >= 5


# ── plot_actionable_targets ──────────────────────────────────────────────


def test_plot_actionable_targets_saves_png(tmp_path):
    df = _tcga_sample("PRAD")
    out = tmp_path / "targets.png"
    fig = plot_actionable_targets(
        df, "PRAD", purity_estimate=0.6, save_to_filename=str(out)
    )
    assert fig is not None
    assert out.exists()


def test_plot_actionable_targets_without_purity(tmp_path):
    df = _tcga_sample("BRCA")
    out = tmp_path / "targets.png"
    fig = plot_actionable_targets(df, "BRCA", save_to_filename=str(out))
    assert fig is not None


def test_plot_actionable_targets_offsets_observed_and_adjusted_markers():
    ref = pan_cancer_expression().drop_duplicates(subset="Symbol").set_index("Symbol")
    gid = ref.loc["FOLH1", "Ensembl_Gene_ID"]
    df = pd.DataFrame(
        {
            "ensembl_gene_id": [gid],
            "gene_symbol": ["FOLH1"],
            "TPM": [40.0],
        }
    )
    ranges = pd.DataFrame(
        [
            {
                "symbol": "FOLH1",
                "observed_tpm": 40.0,
                "attr_tumor_tpm": 20.0,
                "attr_tumor_tpm_low": 15.0,
                "attr_tumor_tpm_high": 25.0,
                "attr_tumor_fraction": 0.5,
            }
        ]
    )

    fig = plot_actionable_targets(
        df,
        "PRAD",
        purity_estimate=0.20,
        custom_genes=["FOLH1"],
        ranges_df=ranges,
    )

    ax = fig.axes[0]
    assert ax.get_xlim()[0] < 0  # Zero-valued estimates must not be clipped by the bars' sticky edge.
    assert ax.get_xlim()[1] > ax.dataLim.x1
    scatter_offsets = [
        collection.get_offsets()
        for collection in ax.collections
        if hasattr(collection, "get_offsets") and len(collection.get_offsets()) > 0
    ]
    y_values = {
        round(float(offset[1]), 2)
        for offsets in scatter_offsets
        for offset in offsets
        if len(offset) == 2
    }
    x_values = {
        round(float(offset[0]), 1)
        for offsets in scatter_offsets
        for offset in offsets
        if len(offset) == 2
    }
    assert {-0.11, 0.11}.issubset(y_values)
    assert {40.0, 100.0}.issubset(x_values)


# ── plot_tumor_attribution ───────────────────────────────────────────────


def test_plot_tumor_attribution_surface(tmp_path):
    df = _tcga_sample("PRAD")
    out = tmp_path / "attrib.png"
    fig = plot_tumor_attribution(
        df, "PRAD", purity_estimate=0.6, save_to_filename=str(out)
    )
    assert fig is not None
    assert out.exists()


def test_plot_tumor_attribution_cta(tmp_path):
    df = _tcga_sample("PRAD")
    out = tmp_path / "attrib_cta.png"
    fig = plot_tumor_attribution(
        df, "PRAD", purity_estimate=0.6, category="CTA", save_to_filename=str(out)
    )
    # May be None if no CTAs expressed
    if fig is not None:
        assert out.exists()


# ── plot_cta_deep_dive ───────────────────────────────────────────────────


def test_plot_cta_deep_dive_saves_png(tmp_path):
    df = _tcga_sample("SKCM")  # melanoma has high CTA expression
    out = tmp_path / "cta.png"
    fig = plot_cta_deep_dive(
        df, "SKCM", purity_estimate=0.65, save_to_filename=str(out)
    )
    assert fig is not None
    assert out.exists()


def test_plot_curated_target_evidence_saves_png(tmp_path):
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "FOLH1",
                "observed_tpm": 87.0,
                "attr_tumor_tpm": 34.0,
                "attr_tumor_tpm_low": 28.0,
                "attr_tumor_tpm_high": 39.0,
                "attr_tumor_fraction": 0.39,
                "attr_tumor_fraction_low": 0.32,
                "attr_tumor_fraction_high": 0.45,
                "attr_support_fraction": 1.0,
                "matched_normal_tissue": "prostate",
                "matched_normal_tpm": 46.0,
                "attr_top_compartment": "matched_normal_prostate",
                "tme_explainable": True,
                "tme_dominant": False,
                "matched_normal_over_predicted": False,
                "category": "therapy_target",
            },
            {
                "symbol": "STEAP2",
                "observed_tpm": 90.0,
                "attr_tumor_tpm": 13.0,
                "attr_tumor_tpm_low": 8.0,
                "attr_tumor_tpm_high": 17.0,
                "attr_tumor_fraction": 0.14,
                "attr_tumor_fraction_low": 0.09,
                "attr_tumor_fraction_high": 0.19,
                "attr_support_fraction": 0.0,
                "matched_normal_tissue": "prostate",
                "matched_normal_tpm": 78.0,
                "attr_top_compartment": "matched_normal_prostate",
                "tme_explainable": True,
                "tme_dominant": True,
                "matched_normal_over_predicted": False,
                "category": "therapy_target",
            },
        ]
    )
    target_panel = pd.DataFrame(
        [
            {
                "symbol": "FOLH1",
                "agent": "177Lu-PSMA-617",
                "agent_class": "radioligand",
                "phase": "approved",
                "indication": "mCRPC",
            },
            {
                "symbol": "STEAP2",
                "agent": "experimental ADC",
                "agent_class": "ADC",
                "phase": "phase_2",
                "indication": "mCRPC",
            },
        ]
    )
    out = tmp_path / "curated-evidence.png"
    fig = plot_curated_target_evidence(
        ranges_df,
        target_panel,
        "PRAD",
        save_to_filename=str(out),
    )
    assert fig is not None
    assert out.exists()


def test_plot_priority_targets_saves_png(tmp_path):
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "FOLH1",
                "observed_tpm": 87.0,
                "attr_tumor_tpm": 34.0,
                "attr_tumor_tpm_low": 18.0,
                "attr_tumor_tpm_high": 40.0,
                "attr_tumor_fraction": 0.39,
                "attr_tumor_fraction_low": 0.21,
                "attr_tumor_fraction_high": 0.45,
                "attr_support_fraction": 0.67,
                "matched_normal_tissue": "prostate",
                "matched_normal_tpm": 46.0,
                "attr_top_compartment": "matched_normal_prostate",
                "tme_explainable": True,
                "tme_dominant": False,
                "matched_normal_over_predicted": False,
                "therapy_supported": True,
                "therapies": "ADC, radioligand",
                "tcga_percentile": 0.97,
                "category": "therapy_target",
            },
            {
                "symbol": "FAP",
                "observed_tpm": 44.0,
                "attr_tumor_tpm": 41.0,
                "attr_tumor_tpm_low": 37.0,
                "attr_tumor_tpm_high": 44.0,
                "attr_tumor_fraction": 0.94,
                "attr_tumor_fraction_low": 0.83,
                "attr_tumor_fraction_high": 1.0,
                "attr_support_fraction": 1.0,
                "matched_normal_tissue": "",
                "matched_normal_tpm": 0.0,
                "attr_top_compartment": "endothelial",
                "tme_explainable": True,
                "tme_dominant": False,
                "matched_normal_over_predicted": False,
                "therapy_supported": True,
                "therapies": "ADC, radioligand",
                "tcga_percentile": 1.0,
                "category": "therapy_target",
            },
        ]
    )
    target_panel = pd.DataFrame(
        [
            {
                "symbol": "FOLH1",
                "agent": "177Lu-PSMA-617",
                "agent_class": "radioligand",
                "phase": "approved",
                "indication": "mCRPC",
            },
        ]
    )
    out = tmp_path / "priority-targets.png"
    fig = plot_priority_targets(
        ranges_df,
        "PRAD",
        target_panel=target_panel,
        save_to_filename=str(out),
    )
    assert fig is not None
    assert out.exists()
    ax = fig.axes[0]
    assert ax.get_title() == "Target Priority Ranking — PRAD"
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert ax.spines["left"].get_visible()
    assert ax.spines["bottom"].get_visible()
    texts = "\n".join(text.get_text() for ax in fig.axes for text in ax.texts)
    assert "Approved pathway / eligibility pending" in texts
    assert any(
        text.get_text() == "Eligibility / state fit"
        for legend in fig.legends + [ax.get_legend() for ax in fig.axes]
        if legend is not None
        for text in legend.get_texts()
    )
    fig.canvas.draw()
    legend_bbox = ax.get_legend().get_window_extent()
    assert not any(
        legend_bbox.overlaps(patch.get_window_extent())
        for patch in ax.patches
    )
    # Figure 2 now owns RNA-source/context details. No dense footer may expand
    # the ranking canvas and shrink the actual bars in the report.
    assert not fig.texts
    assert out.with_suffix(".pdf").exists()


def test_priority_targets_exclude_hla_mismatched_rows(tmp_path):
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "MAGEA4",
                "observed_tpm": 40.0,
                "attr_tumor_tpm": 35.0,
                "attr_tumor_tpm_low": 25.0,
                "attr_tumor_tpm_high": 44.0,
                "attr_tumor_fraction": 0.88,
                "attr_tumor_fraction_low": 0.70,
                "attr_tumor_fraction_high": 1.0,
                "attr_support_fraction": 1.0,
                "matched_normal_tissue": "",
                "matched_normal_tpm": 0.0,
                "therapy_supported": True,
                "therapies": "TCR-T",
                "tcga_percentile": 0.99,
                "category": "therapy_target",
            },
            {
                "symbol": "EGFR",
                "observed_tpm": 30.0,
                "attr_tumor_tpm": 25.0,
                "attr_tumor_tpm_low": 20.0,
                "attr_tumor_tpm_high": 35.0,
                "attr_tumor_fraction": 0.83,
                "attr_tumor_fraction_low": 0.65,
                "attr_tumor_fraction_high": 1.0,
                "attr_support_fraction": 1.0,
                "matched_normal_tissue": "",
                "matched_normal_tpm": 0.0,
                "therapy_supported": True,
                "therapies": "antibody",
                "tcga_percentile": 0.85,
                "category": "therapy_target",
            },
        ]
    )
    target_panel = pd.DataFrame(
        [
            {
                "symbol": "MAGEA4",
                "agent": "afamitresgene autoleucel",
                "agent_class": "TCR-T",
                "phase": "approved",
                "indication": "advanced synovial sarcoma; requires A*02:01",
            },
            {
                "symbol": "EGFR",
                "agent": "cetuximab",
                "agent_class": "antibody",
                "phase": "approved",
                "indication": "EGFR-expressing cancer",
            },
        ]
    )

    fig = plot_priority_targets(
        ranges_df,
        "SARC",
        target_panel=target_panel,
        analysis={"analysis_constraints": {"hla_types": ["A*01:01"]}},
        save_to_filename=str(tmp_path / "priority-targets.png"),
    )

    labels = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
    assert "MAGEA4" not in labels
    assert labels == ["EGFR"]


def test_plot_priority_target_context_saves_png(tmp_path):
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "FOLH1",
                "observed_tpm": 87.0,
                "attr_tumor_tpm": 34.0,
                "attr_tumor_tpm_low": 18.0,
                "attr_tumor_tpm_high": 40.0,
                "attr_tumor_fraction": 0.39,
                "attr_tumor_fraction_low": 0.21,
                "attr_tumor_fraction_high": 0.45,
                "attr_support_fraction": 0.67,
                "matched_normal_tissue": "prostate",
                "matched_normal_tpm": 46.0,
                "attr_top_compartment": "matched_normal_prostate",
                "tme_explainable": True,
                "tme_dominant": False,
                "matched_normal_over_predicted": False,
                "therapy_supported": True,
                "therapies": "ADC, radioligand",
                "tcga_percentile": 0.97,
                "category": "therapy_target",
            },
            {
                "symbol": "FAP",
                "observed_tpm": 44.0,
                "attr_tumor_tpm": 17.0,
                "attr_tumor_tpm_low": 10.0,
                "attr_tumor_tpm_high": 24.0,
                "attr_tumor_fraction": 0.39,
                "attr_tumor_fraction_low": 0.23,
                "attr_tumor_fraction_high": 0.55,
                "attr_support_fraction": 0.0,
                "matched_normal_tissue": "",
                "matched_normal_tpm": 0.0,
                "attr_top_compartment": "fibroblast",
                "tme_explainable": True,
                "tme_dominant": True,
                "matched_normal_over_predicted": False,
                "therapy_supported": True,
                "therapies": "radioligand",
                "tcga_percentile": 1.0,
                "category": "therapy_target",
            },
        ]
    )
    target_panel = pd.DataFrame(
        [
            {
                "symbol": "FOLH1",
                "agent": "177Lu-PSMA-617",
                "agent_class": "radioligand",
                "phase": "approved",
                "indication": "mCRPC",
            },
        ]
    )
    out = tmp_path / "priority-target-context.png"
    fig = plot_priority_target_context(
        ranges_df,
        "PRAD",
        target_panel=target_panel,
        save_to_filename=str(out),
    )
    assert fig is not None
    assert out.exists()
    ax_range = fig.axes[0]
    assert ax_range.get_xscale() == "symlog"
    assert ax_range.get_xlabel() == "Contribution to bulk RNA (TPM)"
    assert ax_range.get_title(loc="left") == "RNA amount"
    assert len(fig.axes) == 3
    assert fig.axes[1].get_title(loc="left") == "RNA source"
    assert fig.axes[2].get_title(loc="left") == "Healthy-tissue overlap"
    # The estimable target retains its modeled share. The stromal target gets
    # a source caveat rather than a precise share next to contradictory text.
    import pytest
    assert [bar.get_width() for bar in fig.axes[1].patches] == pytest.approx([1, 0.39])
    assert "Non-tumor source\nplausible" in [text.get_text() for text in fig.axes[1].texts]
    assert not fig.texts
    assert out.with_suffix(".pdf").exists()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for axis in fig.axes:
        for text in axis.texts:
            assert fig.bbox.contains(*text.get_window_extent(renderer).get_points()[0])


def test_priority_target_context_can_use_actionable_target_symbols(tmp_path):
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "EGFR",
                "observed_tpm": 80.0,
                "attr_tumor_tpm": 70.0,
                "attr_tumor_tpm_low": 55.0,
                "attr_tumor_tpm_high": 90.0,
                "attr_tumor_fraction": 0.88,
                "attr_tumor_fraction_low": 0.70,
                "attr_tumor_fraction_high": 1.0,
                "attr_support_fraction": 1.0,
                "matched_normal_tissue": "",
                "matched_normal_tpm": 0.0,
                "therapy_supported": True,
                "therapies": "antibody",
                "tcga_percentile": 0.90,
                "category": "therapy_target",
            },
            {
                "symbol": "FAP",
                "observed_tpm": 90.0,
                "attr_tumor_tpm": 85.0,
                "attr_tumor_tpm_low": 80.0,
                "attr_tumor_tpm_high": 92.0,
                "attr_tumor_fraction": 0.94,
                "attr_tumor_fraction_low": 0.88,
                "attr_tumor_fraction_high": 1.0,
                "attr_support_fraction": 1.0,
                "matched_normal_tissue": "",
                "matched_normal_tpm": 0.0,
                "therapy_supported": True,
                "therapies": "radioligand",
                "tcga_percentile": 1.0,
                "category": "therapy_target",
            },
        ]
    )
    target_panel = pd.DataFrame(
        [
            {
                "symbol": "EGFR",
                "agent": "cetuximab",
                "agent_class": "antibody",
                "phase": "approved",
                "indication": "CRC",
            },
        ]
    )

    fig = plot_priority_target_context(
        ranges_df,
        "READ",
        target_panel=target_panel,
        target_symbols=["EGFR"],
        save_to_filename=str(tmp_path / "priority-target-context.png"),
    )

    labels = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
    assert labels == ["EGFR"]


# ── subtype signatures ───────────────────────────────────────────────────


def test_subtype_contrasts_exist_for_prad():
    assert "PRAD" in SUBTYPE_CONTRASTS
    assert SUBTYPE_CONTRASTS["PRAD"][0]["axis_a"] == "AR_signaling"
    assert SUBTYPE_CONTRASTS["PRAD"][0]["axis_b"] == "NE_differentiation"


def test_subtype_contrasts_exist_for_brca():
    assert "BRCA" in SUBTYPE_CONTRASTS


def test_plot_subtype_signature_prad(tmp_path):
    df = _tcga_sample("PRAD")
    out = tmp_path / "subtype.png"
    fig = plot_subtype_signature(df, "PRAD", save_to_filename=str(out))
    assert fig is not None
    assert out.exists()


def test_plot_subtype_signature_brca(tmp_path):
    df = _tcga_sample("BRCA")
    out = tmp_path / "subtype.png"
    fig = plot_subtype_signature(df, "BRCA", save_to_filename=str(out))
    assert fig is not None
    assert out.exists()


def test_plot_subtype_signature_no_contrast_returns_none():
    df = _tcga_sample("ACC")
    fig = plot_subtype_signature(df, "ACC")
    assert fig is None


def test_priority_source_does_not_present_caps_or_zero_residual_as_measured_shares(monkeypatch):
    import trufflepig.plot_target_deep_dive as plots
    from trufflepig.reporting import tumor_attribution_context

    rows = []
    for symbol, fraction, capped in [('CAPPED', 0.2304, True), ('ZERO', 0, False)]:
        source = tumor_attribution_context({
            'observed_tpm': 100, 'attr_tumor_tpm': 100 * fraction,
            'attr_tumor_fraction': fraction, 'low_purity_cap_applied': capped,
        })
        rows.append(dict(symbol=symbol, observed=100, low=0, mid=100 * fraction,
                         high=60, source=source,
                         normal={'tier': 'same_lineage_expected', 'label': ''}))
    monkeypatch.setattr(plots, '_priority_target_rows', lambda *a, **kw: ('COAD', rows))
    fig = plots.plot_priority_target_context(pd.DataFrame(), 'COAD')
    labels = [text.get_text() for text in fig.axes[1].texts]
    assert labels == ['Tumor source\nuncertain', 'Non-tumor source\nplausible']
    assert not fig.axes[1].patches  # neither a 23% share nor a fully background bar
    assert len(fig.axes[0].collections) == 2  # measured points only, no unsupported tumor estimates


def test_uncertain_residual_does_not_add_tumor_priority_points():
    from trufflepig.plot_target_deep_dive import _priority_target_rows

    ranges = pd.DataFrame([{
        "symbol": "EGFR", "observed_tpm": 100, "attr_tumor_tpm": 90,
        "attr_tumor_fraction": 0.9, "attribution_low_purity": True,
        "therapies": "antibody", "tcga_percentile": 0.99,
    }])
    _, rows = _priority_target_rows(ranges, "COAD")
    assert len(rows) == 1
    assert rows[0]["source_points"] == 0
    assert rows[0]["strength_points"] == 0
