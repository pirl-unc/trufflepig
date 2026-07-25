import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import trufflepig.plot as plot_mod
import trufflepig.plot_scatter as plot_scatter_mod
import trufflepig.plot_strip as plot_strip_mod
import trufflepig.main as cli_mod
import trufflepig.tumor_purity as purity_mod
from trufflepig.decomposition.plot import (
    plot_decomposition_candidates,
    plot_decomposition_composition,
)
from trufflepig.tumor_purity import _summarize_candidate_family


def _write_target_report(ranges_df, analysis, prefix, cancer_type, purity_result):
    """Build the target report via the live ``_build_target_report`` and write it
    to ``{prefix}-targets.md`` (the contract the removed ``_generate_target_report``
    wrapper used to provide)."""
    md = cli_mod._build_target_report(
        ranges_df,
        analysis,
        cancer_type=cancer_type,
        purity_result=purity_result,
        decomp_results=analysis.get("decomposition_results"),
    )
    with open(f"{prefix}-targets.md", "w") as fh:
        fh.write(md)
    return md


def test_guess_gene_cols_and_pick_genes():
    df = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG2", "ENSG3"],
            "gene_display_name": ["A", "B", "C"],
            "TPM": [1.0, 3.0, 2.0],
            "category": ["x", "x", "y"],
        }
    )
    gid_col, gname_col = plot_mod._guess_gene_cols(df)
    assert gid_col == "gene_id"
    assert gname_col == "gene_display_name"

    selected = plot_mod.pick_genes_to_annotate(df, num_per_category=1)
    assert selected == {"ENSG2", "ENSG3"}

    with pytest.raises(KeyError):
        plot_mod._guess_gene_cols(pd.DataFrame({"TPM": [1.0]}))


def test_purity_ci_phrase_uses_text_not_warning_icon():
    phrase = cli_mod._purity_ci_phrase(
        {
            "overall_estimate": 0.50,
            "overall_lower": 0.10,
            "overall_upper": 1.00,
        }
    )
    assert "low confidence" in phrase
    assert "\u26a0" not in phrase


def test_tumor_tpm_map_uses_source_attributed_tpm_not_context_tpm():
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "FAP",
                "attr_tumor_tpm": 0.2,
                "tumor_cell_tpm": 125.0,
                "median_est": 80.0,
            },
            {
                "symbol": "EGFR",
                "attr_tumor_tpm": 42.0,
                "tumor_cell_tpm": 50.0,
                "median_est": 45.0,
            },
        ]
    )

    mapping = cli_mod._tumor_tpm_by_symbol_from_ranges(ranges_df)

    assert mapping["FAP"] == 0.2
    assert mapping["EGFR"] == 42.0


def test_resolve_always_label_gene_ids(monkeypatch):
    df = pd.DataFrame(
        {"gene_id": ["ENSG1", "ENSG2"], "gene_display_name": ["GENE1", "B7-H3"]}
    )
    monkeypatch.setattr(
        plot_strip_mod,
        "find_canonical_gene_ids_and_names",
        lambda tokens: (["ENSG2"], ["CD276"]),
    )
    out = plot_mod.resolve_always_label_gene_ids(df, {"GENE1", "CD276"})
    assert out == {"ENSG1", "ENSG2"}


def test_plot_gene_expression_smoke(monkeypatch, tmp_path):
    prepared = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG2"],
            "gene_display_name": ["GENE1", "GENE2"],
            "TPM": [0.05, 2.0],
            "category": ["A", "A"],
        }
    )
    monkeypatch.setattr(
        plot_strip_mod, "prepare_gene_expr_df", lambda *a, **k: prepared.copy()
    )
    monkeypatch.setattr(plot_strip_mod, "adjust_text", lambda *a, **k: None)

    class FakeAx:
        def __init__(self):
            self.text_calls = []
            self.collections = []

        def text(self, *args, **kwargs):
            self.text_calls.append((args, kwargs))
            return SimpleNamespace()

        def scatter(self, *args, **kwargs):
            pass

        def axhline(self, *args, **kwargs):
            pass

        def annotate(self, *args, **kwargs):
            pass

    class FakeFigure:
        def __init__(self):
            self.saved = None

        def savefig(self, *args, **kwargs):
            self.saved = (args, kwargs)

    class FakeCat:
        def __init__(self):
            self.ax = FakeAx()
            self.figure = FakeFigure()

    fake_cat = FakeCat()
    import trufflepig.plot_strip as _ps

    monkeypatch.setattr(_ps.sns, "catplot", lambda **kwargs: fake_cat)

    out_path = tmp_path / "plot.png"
    result = plot_mod.plot_gene_expression(
        pd.DataFrame(
            {
                "gene_id": ["ENSG1", "ENSG2"],
                "gene_display_name": ["GENE1", "GENE2"],
                "TPM": [0.05, 2.0],
            }
        ),
        gene_sets={"A": {"GENE1", "GENE2"}},
        save_to_filename=str(out_path),
        always_label_genes={"GENE1"},
        save_dpi=123,
    )
    assert result is fake_cat
    assert fake_cat.figure.saved is not None
    _, kwargs = fake_cat.figure.saved
    assert kwargs["dpi"] == 123


def test_tumor_expr_plot_suppresses_near_zero_median_fold():
    """#85.3: the tumor-expression figure's "vs reference" panel applies the same
    detection-floor guard as the markdown cell — a finite fold off a sub-1-TPM
    cohort median renders "ref X TPM", not a noise-amplified bar — so the plot
    and the table can't disagree."""
    import matplotlib.pyplot as plt

    import trufflepig.plot_tumor_expr as pte_mod

    df_ranges = pd.DataFrame(
        [
            {  # near-zero cohort median → the 137× fold is noise
                "category": "surface",
                "symbol": "NTRK1",
                "therapies": "",
                "median_est": 22.0,
                "pct_cancer_median": 137.4,
                "tcga_cohort_median_tpm": 0.16,
                **{f"est_{i + 1}": 22.0 for i in range(9)},
            },
            {  # detectable cohort median → a real fold, kept
                "category": "surface",
                "symbol": "ERBB2",
                "therapies": "",
                "median_est": 50.0,
                "pct_cancer_median": 3.2,
                "tcga_cohort_median_tpm": 45.0,
                **{f"est_{i + 1}": 50.0 for i in range(9)},
            },
        ]
    )
    purity = {"overall_lower": 0.3, "overall_estimate": 0.5, "overall_upper": 0.7}
    fig = pte_mod.plot_tumor_expression_ranges(
        df_ranges, purity, "BRCA", categories=["surface"]
    )
    try:
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    finally:
        plt.close(fig)
    # Near-zero-median gene shows the reference TPM, not the noise fold.
    assert "ref 0.16 TPM" in texts
    assert not any(t in ("137.4×", "137×") for t in texts)
    # A real fold off a detectable cohort median is preserved.
    assert "3.2×" in texts


def test_cli_plot_expression_and_main(monkeypatch, tmp_path):
    calls = []
    scatter_calls = []
    cancer_gene_calls = []
    pca_calls = []
    mds_calls = []
    neighborhood_calls = []
    tissue_calls = []
    monkeypatch.setattr(
        cli_mod, "load_expression_data", lambda *a, **k: pd.DataFrame({"x": [1]})
    )
    monkeypatch.setattr(
        cli_mod, "plot_gene_expression", lambda *a, **k: calls.append(k)
    )
    # PR-4 (§2.5): materialize the per-category curated panel scatter PNGs that the
    # real plot_sample_vs_cancer emits into its {prefix}-vs-cancer/ dir, so the
    # reader-vs-audit routing of those PNGs is exercised (asserted at the end).
    def _fake_plot_sample_vs_cancer(*a, **k):
        scatter_calls.append(k)
        save_to = k.get("save_to_filename")
        if save_to:
            sdir = Path(str(save_to)).with_suffix("")
            sdir.mkdir(parents=True, exist_ok=True)
            for _name in ("Oncogenes.png", "CTAs.png"):
                cli_mod.Image.new("RGB", (16, 16), "white").save(sdir / _name)

    monkeypatch.setattr(cli_mod, "plot_sample_vs_cancer", _fake_plot_sample_vs_cancer)

    # Record every Image.open path: the reader packet (all-figures.pdf) opens
    # png_files from their original locations; the audit packet (figure-audit.pdf)
    # opens the moved figures/ copies. This lets us prove the scatters route to audit.
    opened_paths = []
    _real_image_open = cli_mod.Image.open

    def _recording_open(fp, *a, **k):
        opened_paths.append(str(fp))
        return _real_image_open(fp, *a, **k)

    monkeypatch.setattr(cli_mod.Image, "open", _recording_open)
    monkeypatch.setattr(
        cli_mod, "plot_therapy_target_tissues", lambda *a, **k: tissue_calls.append(k)
    )
    # plot_cancer_type_genes / plot_cancer_type_disjoint_genes were
    # removed from the default plot set (polish/4.40.1); skip
    # monkeypatching them — they're no longer imported by cli.
    monkeypatch.setattr(
        cli_mod, "plot_cancer_type_mds", lambda *a, **k: mds_calls.append(k)
    )
    monkeypatch.setattr(
        cli_mod,
        "plot_cancer_type_neighborhood",
        lambda *a, **k: neighborhood_calls.append(k),
    )
    monkeypatch.setattr(
        cli_mod, "therapy_target_gene_id_to_name", lambda t: {"ENSG_MOCK": t}
    )
    monkeypatch.setattr(
        cli_mod, "pMHC_TCE_target_gene_id_to_name", lambda: {"ENSG_PMHC": "PMHC"}
    )
    monkeypatch.setattr(
        cli_mod, "surface_TCE_target_gene_id_to_name", lambda: {"ENSG_SURF": "SURF"}
    )
    mock_analysis = {
        "cancer_type": "PRAD",
        "cancer_name": "Prostate",
        "cancer_score": 0.9,
        "top_cancers": [("PRAD", 0.9)],
        "purity": {
            "overall_estimate": 0.1,
            "overall_lower": 0.05,
            "overall_upper": 0.15,
            "components": {
                "stromal": {"enrichment": 4.0},
                "immune": {"enrichment": 2.0},
            },
        },
        "tissue_scores": [("prostate", 0.9, 20)],
        "mhc1": {"HLA-A": 100, "HLA-B": 200, "HLA-C": 150, "B2M": 3000},
        "mhc2": {},
    }
    monkeypatch.setattr(cli_mod, "analyze_sample", lambda *a, **k: mock_analysis)
    monkeypatch.setattr(
        cli_mod,
        "assess_sample_quality",
        lambda *a, **k: {
            "degradation": {
                "mt_fraction": 0.05,
                "rp_fraction": 0.08,
                "long_short_ratio": 0.95,
                "n_mt_found": 13,
                "n_long_found": 18,
                "matched_tissue": "prostate",
                "baseline_mt": 0.29,
                "baseline_rp": 0.21,
                "mt_fold": 0.17,
                "rp_fold": 0.38,
                "level": "normal",
                "message": "No degradation",
            },
            "culture": {
                "stress_score": 0.5,
                "tme_mean_tpm": 50.0,
                "tme_absent": False,
                "top_stress_genes": [],
                "n_tme_found": 15,
                "level": "normal",
                "message": "No culture signal",
            },
            "flags": ["No quality concerns detected"],
            "has_issues": False,
        },
    )
    # PR-5 (§2.5): materialize the 4-panel composite PNG the real plot_sample_summary
    # writes, so its reader-vs-audit routing is exercised (asserted at the end).
    def _fake_plot_sample_summary(*a, **k):
        save_to = k.get("save_to_filename")
        if save_to:
            cli_mod.Image.new("RGB", (16, 16), "white").save(save_to)
        return (None, mock_analysis)

    monkeypatch.setattr(cli_mod, "plot_sample_summary", _fake_plot_sample_summary)
    monkeypatch.setattr(
        cli_mod, "plot_tumor_purity", lambda *a, **k: (None, mock_analysis["purity"])
    )
    decomp_kwargs = {}
    monkeypatch.setattr(
        cli_mod,
        "decompose_sample",
        lambda *a, **k: decomp_kwargs.update(k) or [],
    )
    monkeypatch.setattr(cli_mod, "plot_decomposition_composition", lambda *a, **k: None)
    monkeypatch.setattr(
        cli_mod, "plot_decomposition_component_breakdown", lambda *a, **k: None
    )
    monkeypatch.setattr(cli_mod, "plot_decomposition_candidates", lambda *a, **k: None)

    report_calls = []
    target_report_calls = []
    monkeypatch.setattr(
        cli_mod, "_generate_text_reports", lambda *a, **k: report_calls.append(True)
    )
    monkeypatch.setattr(
        cli_mod,
        "_build_target_report",
        lambda *a, **k: target_report_calls.append(True)
        or "# Therapeutic Target Analysis\n\nmock",
    )
    monkeypatch.setattr(
        cli_mod,
        "get_embedding_feature_metadata",
        lambda **k: {
            "method": "hierarchy",
            "feature_kind": "hierarchical_scores",
            "n_features": 5,
            "n_types": 2,
            "families": ["PROSTATE"],
        },
    )
    monkeypatch.setattr(
        cli_mod, "estimate_tumor_expression_ranges", lambda *a, **k: pd.DataFrame()
    )
    monkeypatch.setattr(cli_mod, "plot_tumor_expression_ranges", lambda *a, **k: None)

    out_dir = str(tmp_path / "test-output")
    cli_mod.analyze(
        "input.csv",
        transcripts="input.csv",
        output_dir=out_dir,
        output_image_prefix="out",
        label_genes="FAP,CD276",
        output_dpi=200,
        sample_mode="solid",
        tumor_context="met",
        site_hint="liver",
        decomposition_templates="met_liver",
        therapy_target_top_k=12,
        therapy_target_tpm_threshold=18,
    )
    # prefix becomes output_dir/output_image_prefix
    expected_prefix = str(tmp_path / "test-output" / "out")
    # v4.46.0: retired the immune / tumor / antigens overview strip
    # plots — they duplicated the 10 per-category curated strip plots
    # (Immune_checkpoints / Oncogenes / CTAs / ...). Only the
    # treatments modality strip plot remains.
    assert len(calls) == 1
    assert calls[0]["save_to_filename"] == f"{expected_prefix}-treatments.png"
    assert calls[0]["gene_sets"]["Radio"] == {"ENSG_MOCK": "radioligand"}
    assert calls[0]["always_label_genes"] == {"FAP", "CD276"}
    assert len(scatter_calls) == 1
    assert scatter_calls[0]["save_to_filename"] == f"{expected_prefix}-vs-cancer.pdf"
    # #83: scatter must use the resolved cancer type (PRAD from
    # analyze_sample), not None / the raw CLI arg.
    assert scatter_calls[0]["cancer_type"] == "PRAD"
    assert len(tissue_calls) == 1
    assert tissue_calls[0]["top_k"] == 12
    assert tissue_calls[0]["tpm_threshold"] == 18
    assert {"FAP", "CD276"}.issubset(tissue_calls[0]["extra_symbols"])
    # plot_cancer_type_genes / plot_cancer_type_disjoint_genes were
    # removed from the default plot set (polish/4.40.1).
    assert len(cancer_gene_calls) == 0
    # Pan-reference MDS plus a nearest-reference distance ranking are emitted now;
    # PCA and hierarchy-method plots have been removed from the default output
    # (see pirl-unc/pirlygenes#36).
    assert len(pca_calls) == 0
    assert len(mds_calls) == 1
    assert mds_calls[0]["method"] == "panref"
    assert mds_calls[0]["include_normals"] is True
    assert mds_calls[0]["include_subtypes"] is True
    assert mds_calls[0]["label_nearest_cancers"] == 5
    assert mds_calls[0]["label_nearest_normals"] == 5
    assert mds_calls[0]["label_all"] is False
    assert len(neighborhood_calls) == 1
    assert neighborhood_calls[0]["method"] == "panref"
    assert neighborhood_calls[0]["include_normals"] is True
    assert neighborhood_calls[0]["include_subtypes"] is True
    assert neighborhood_calls[0]["label_nearest_cancers"] == 5
    assert neighborhood_calls[0]["label_nearest_normals"] == 5
    assert neighborhood_calls[0]["label_all"] is False
    assert neighborhood_calls[0]["focus_nearest_cancers"] == 25
    assert neighborhood_calls[0]["focus_nearest_normals"] == 10
    assert len(report_calls) == 2
    assert len(target_report_calls) == 1
    assert (tmp_path / "test-output" / "out-summary.md").exists()
    assert (tmp_path / "test-output" / "out-evidence.md").exists()
    assert not (tmp_path / "test-output" / "out-actionable.md").exists()
    assert not (tmp_path / "test-output" / "out-targets.md").exists()
    assert not (tmp_path / "test-output" / "out-provenance.md").exists()
    assert not (tmp_path / "test-output" / "out-brief.md").exists()
    params = json.loads(
        (tmp_path / "test-output" / "out-analysis-parameters.json").read_text()
    )
    assert "tumor_purity" in params
    assert "decomposition" in params
    assert params["selected_sample_mode"] == "solid"
    assert params["embedding_methods"] == [
        "pan_reference_mds",
        "pan_reference_nearest_references",
    ]
    assert params["input"]["tumor_context"] == "met"
    assert params["input"]["site_hint"] == "liver"
    assert params["input"]["decomposition_templates"] == ["met_liver"]
    assert decomp_kwargs["tumor_context"] == "met"
    assert decomp_kwargs["site_hint"] == "liver"
    assert decomp_kwargs["templates"] == ["met_liver"]
    readme = (tmp_path / "test-output" / "README.md").read_text()
    assert readme.startswith("# Trufflepig Analysis Output")
    assert readme.index("## Start here") < readme.index("## Data and normalization")
    assert "Prefer the standalone decomposition figures" in readme
    assert "*-decomposition-composition.png" in readme
    assert "*-decomposition.png" not in readme

    # PR-4 (§2.5): the ~10 curated per-category panel scatters are audit-only. The
    # mock created out-vs-cancer/{Oncogenes,CTAs}.png; each must be relocated into
    # figures/ (retained, swept into figure-audit.pdf) but NEVER collected into the
    # reader packet (all-figures.pdf). Reader collection opens png_files from the
    # original vs-cancer/ location; audit collection opens the moved figures/ copy —
    # so the vs-cancer/ path must not appear among opened figures, the figures/ copy
    # must, and the file must survive in figures/.
    figures_dir = tmp_path / "test-output" / "figures"
    for _name in ("Oncogenes.png", "CTAs.png"):
        assert (figures_dir / _name).exists()  # retained in the audit set, not deleted
        assert not any(
            f"out-vs-cancer/{_name}" in op for op in opened_paths
        )  # never collected into the reader packet from its original location
        assert any(
            f"figures/{_name}" in op for op in opened_paths
        )  # present in the audit packet (opened from figures/)

    # PR-5 (§2.5): the 4-panel sample-summary.png composite is audit-only. The mock
    # wrote {prefix}-sample-summary.png; it must relocate into figures/ (audit) and
    # NOT be collected into the reader packet from its original location — the four
    # standalone panels it duplicates stay in the reader set (untouched here).
    assert (figures_dir / "out-sample-summary.png").exists()  # retained in audit
    assert (
        f"{expected_prefix}-sample-summary.png" not in opened_paths
    )  # composite never collected into the reader packet
    assert any(
        "figures/out-sample-summary.png" in op for op in opened_paths
    )  # composite present in the audit packet

    # After the migration, `python -m trufflepig.main` no longer ships
    # a CLI — it's a redirect-only entry point that prints a
    # "use trufflepig.cli" message and exits 2. The real CLI lives in
    # :mod:`trufflepig.cli`.
    import pytest

    with pytest.raises(SystemExit) as excinfo:
        cli_mod.main()
    assert excinfo.value.code == 2


def test_generate_text_reports_uses_family_and_background_language(tmp_path):
    analysis = {
        "cancer_type": "COAD",
        "cancer_name": "Colon Adenocarcinoma",
        "cancer_score": 0.24,
        "family_summary": {
            "display": "CRC-family (COAD > READ)",
            "subtype_clause": "COAD > READ",
        },
        "top_cancers": [("COAD", 0.24), ("READ", 0.22)],
        "candidate_trace": [
            {
                "code": "COAD",
                "support_score": 0.24,
                "signature_score": 0.84,
                "purity_estimate": 0.37,
                "family_label": "CRC",
                "lineage_purity": 0.50,
                "lineage_concordance": 0.78,
            },
            {
                "code": "READ",
                "support_score": 0.22,
                "signature_score": 0.78,
                "purity_estimate": 0.36,
                "family_label": "CRC",
                "lineage_purity": 0.45,
                "lineage_concordance": 0.77,
            },
        ],
        "fit_quality": {
            "label": "ambiguous",
            "message": "Top subtype candidates remain close; treat the leading label as provisional.",
        },
        "purity": {
            "overall_estimate": 0.37,
            "overall_lower": 0.13,
            "overall_upper": 0.81,
            "cancer_type": "COAD",
            "components": {
                "stromal": {"enrichment": 15.3},
                "immune": {"enrichment": 3.0},
                "lineage": {"per_gene": []},
            },
        },
        "tissue_scores": [("smooth_muscle", 0.91, 20), ("gallbladder", 0.86, 20)],
        "mhc1": {"HLA-A": 19, "HLA-B": 614, "B2M": 7089},
        "mhc2": {},
        "sample_mode": "solid",
        "call_summary": {
            "label_options": ["COAD", "READ"],
            "label_display": "COAD or READ",
            "reported_context": "primary",
            "reported_site": "primary site",
            "site_indeterminate": False,
            "site_note": None,
            "hypothesis_display": ["COAD / solid_primary", "READ / solid_primary"],
        },
    }
    embedding_meta = {
        "method": "hierarchy",
        "feature_kind": "hierarchical_scores",
        "n_features": 12,
        "n_types": 2,
        "families": ["CRC", "GASTRIC"],
    }
    prefix = str(tmp_path / "sample")

    cli_mod._generate_text_reports(analysis, embedding_meta, prefix, decomp_results=[])

    # The old free-form summary.md that carried family-call phrasing
    # (CRC-family, retained-labels, subtype-candidates clause) was
    # retired in 4.41.0 as ~80% redundant with analysis.md. The
    # content below is now only checked in analysis.md.
    detailed = (tmp_path / "sample-analysis.md").read_text()
    assert "not literal" in detailed  # tissue-score caveat
    assert "Retained alternatives" in detailed
    assert "Broad family context" in detailed
    assert "Fit quality" in detailed
    assert "Integrated evidence synthesis" in detailed
    assert "Cancer-Type Differential" in detailed
    assert "Raw decomposition audit" in detailed
    assert "Residual Tissue-like Programs" in detailed


def test_generate_text_reports_is_mode_aware_for_heme(tmp_path):
    analysis = {
        "cancer_type": "DLBC",
        "cancer_name": "Diffuse Large B-Cell Lymphoma",
        "cancer_score": 0.42,
        "family_summary": {"display": None, "subtype_clause": None},
        "top_cancers": [("DLBC", 0.42)],
        "candidate_trace": [],
        "purity": {
            "overall_estimate": 0.81,
            "overall_lower": 0.71,
            "overall_upper": 0.89,
            "cancer_type": "DLBC",
            "components": {
                "stromal": {"enrichment": 1.2},
                "immune": {"enrichment": 4.8},
                "lineage": {"per_gene": []},
            },
        },
        "tissue_scores": [("lymph_node", 0.93, 20), ("spleen", 0.78, 20)],
        "mhc1": {"HLA-A": 40, "HLA-B": 55, "B2M": 200},
        "mhc2": {},
        "sample_mode": "heme",
    }
    embedding_meta = {
        "method": "hierarchy",
        "feature_kind": "hierarchical_scores",
        "n_features": 10,
        "n_types": 2,
        "families": ["SQUAMOUS"],
    }
    prefix = str(tmp_path / "heme")

    cli_mod._generate_text_reports(analysis, embedding_meta, prefix, decomp_results=[])

    # Heme-mode "malignant-lineage fraction proxy" phrasing lived in
    # the retired summary.md paragraph; analysis.md still carries the
    # mode-aware "not a strict tumor-vs-immune split" caveat.
    detailed = (tmp_path / "heme-analysis.md").read_text()
    assert "not a strict tumor-vs-immune split" in detailed
    assert "Lineage / Background Context" in detailed


def test_generate_text_reports_handles_missing_lineage_summary(
    tmp_path,
    monkeypatch,
):
    analysis = {
        "cancer_type": "UCS",
        "cancer_name": "Uterine Carcinosarcoma",
        "cancer_score": 0.09,
        "family_summary": {
            "display": "mesenchymal / sarcoma-like family (UCS > SARC)",
            "subtype_clause": "UCS > SARC",
        },
        "top_cancers": [("UCS", 0.09), ("SARC", 0.05)],
        "candidate_trace": [
            {
                "code": "UCS",
                "support_score": 0.09,
                "signature_score": 0.94,
                "purity_estimate": 0.25,
                "family_label": "MESENCHYMAL",
                "lineage_purity": None,
                "lineage_concordance": 0.92,
            },
            {
                "code": "SARC",
                "support_score": 0.05,
                "signature_score": 0.82,
                "purity_estimate": 0.28,
                "family_label": "MESENCHYMAL",
                "lineage_purity": None,
                "lineage_concordance": 1.0,
            },
        ],
        "fit_quality": {
            "label": "weak",
            "message": "Subtype fit is weak: the sample sits in a flat TCGA signature landscape, so broad family interpretation is more trustworthy than the exact top label.",
        },
        "purity": {
            "overall_estimate": 0.25,
            "overall_lower": 0.09,
            "overall_upper": 0.26,
            "cancer_type": "UCS",
            "components": {
                "stromal": {"enrichment": 4.7},
                "immune": {"enrichment": 1.3},
                "lineage": {
                    "per_gene": [
                        {"gene": "COL3A1", "purity": 0.31},
                        {"gene": "DCN", "purity": 0.28},
                    ],
                    "purity": None,
                    "lower": None,
                    "upper": None,
                },
            },
        },
        "tissue_scores": [("smooth_muscle", 0.96, 20), ("skin", 0.87, 20)],
        "mhc1": {"HLA-A": 36, "HLA-B": 66, "B2M": 1997},
        "mhc2": {},
        "sample_mode": "solid",
        "call_summary": {
            "label_options": ["UCS", "SARC"],
            "label_display": "UCS or SARC",
            "reported_context": None,
            "reported_site": None,
            "site_indeterminate": True,
            "site_note": "Weak subtype fit prevents a reliable metastatic site call.",
            "hypothesis_display": ["UCS / met_bone", "SARC / met_bone"],
        },
    }
    embedding_meta = {
        "method": "hierarchy",
        "feature_kind": "hierarchical_scores",
        "n_features": 12,
        "n_types": 2,
        "families": ["MESENCHYMAL", "SQUAMOUS"],
    }
    prefix = str(tmp_path / "sarcoma-like")

    monkeypatch.setattr(
        "trufflepig.common.build_sample_tpm_by_symbol",
        lambda _df: {"ESR1": 0.003},
    )
    cli_mod._generate_text_reports(
        analysis,
        embedding_meta,
        prefix,
        decomp_results=[],
        df_expr=pd.DataFrame({"Ensembl_Gene_ID": [], "TPM": []}),
    )

    # "broad family interpretation is more trustworthy" + "site/template
    # assignment is indeterminate" lived in the retired summary.md
    # paragraph. Analysis.md still carries the Reliable cluster signal.
    detailed = (tmp_path / "sarcoma-like-analysis.md").read_text()
    assert "**Reliable cluster**: COL3A1, DCN." in detailed
    assert "Reported site/template call: **indeterminate**." in detailed
    assert "SARC (Sarcoma)" in detailed
    assert "UCS (Uterine Carcinosarcoma)" in detailed
    assert "top-level cancer-code hypothesis" in detailed
    assert (
        "Lineage** is a purity estimate derived only from the curated lineage genes"
        in detailed
    )
    assert "UCS / met_bone" not in detailed
    assert (
        "| ESR1 | — | detected <0.01 TPM — uninformative "
        "(not specific enough to this cancer type for purity calibration) |"
        in detailed
    )
    assert "| ESR1 | — | not detected |" not in detailed


def test_summarize_sample_call_keeps_primary_site_for_weak_primary_fit():
    analysis = {
        "cancer_type": "SARC",
        "candidate_trace": [
            {
                "code": "UCS",
                "support_score": 0.21,
                "signature_score": 0.94,
                "purity_estimate": 0.25,
                "family_label": "MESENCHYMAL",
                "lineage_purity": None,
                "lineage_concordance": 0.92,
            },
            {
                "code": "SARC",
                "support_score": 0.20,
                "signature_score": 0.82,
                "purity_estimate": 0.28,
                "family_label": "MESENCHYMAL",
                "lineage_purity": None,
                "lineage_concordance": 1.0,
            },
        ],
        "fit_quality": {"label": "weak"},
    }
    decomp_results = [
        SimpleNamespace(
            template="solid_primary",
            cancer_type="SARC",
            score=0.2,
            warnings=[],
            template_site_factor=0.9,
            template_tissue_score=0.8,
        ),
        SimpleNamespace(
            template="solid_primary",
            cancer_type="UCS",
            score=0.18,
            warnings=[],
            template_site_factor=0.88,
            template_tissue_score=0.77,
        ),
    ]

    summary = cli_mod._summarize_sample_call(
        analysis, decomp_results, sample_mode="solid"
    )

    assert summary["site_indeterminate"] is False
    assert summary["reported_context"] == "primary"
    assert summary["reported_site"] == "primary site"


def test_generate_text_reports_mentions_analysis_constraints(tmp_path):
    analysis = {
        "cancer_type": "SARC",
        "cancer_name": "Sarcoma",
        "cancer_score": 0.2,
        "top_cancers": [("SARC", 0.2), ("UCS", 0.18)],
        "candidate_trace": [
            {
                "code": "UCS",
                "support_score": 0.21,
                "signature_score": 0.94,
                "purity_estimate": 0.25,
                "family_label": "MESENCHYMAL",
                "lineage_purity": None,
                "lineage_concordance": 0.92,
            },
            {
                "code": "SARC",
                "support_score": 0.20,
                "signature_score": 0.82,
                "purity_estimate": 0.28,
                "family_label": "MESENCHYMAL",
                "lineage_purity": None,
                "lineage_concordance": 1.0,
            },
        ],
        "purity": {
            "overall_estimate": 0.28,
            "overall_lower": 0.06,
            "overall_upper": 0.49,
            "components": {
                "stromal": {"enrichment": 1.7},
                "immune": {"enrichment": 0.4},
                "lineage": {"per_gene": []},
            },
        },
        "family_summary": {
            "display": "mesenchymal / sarcoma-like family (UCS > SARC)",
            "subtype_clause": "UCS > SARC",
        },
        "fit_quality": {
            "label": "weak",
            "message": "Subtype fit is weak.",
        },
        "tissue_scores": [("liver", 0.96, 20)],
        "mhc1": {"HLA-A": 36, "HLA-B": 66, "B2M": 1997},
        "mhc2": {},
        "sample_mode": "solid",
        "analysis_constraints": {
            "cancer_type": "SARC",
            "sample_mode": "solid",
            "tumor_context": "primary",
        },
        "call_summary": {
            "label_options": ["UCS", "SARC"],
            "label_display": "UCS or SARC",
            "reported_context": "primary",
            "reported_site": "primary site",
            "site_indeterminate": False,
            "site_note": None,
            "hypothesis_display": ["UCS / solid_primary", "SARC / solid_primary"],
        },
    }
    embedding_meta = {
        "method": "hierarchy",
        "feature_kind": "hierarchical_scores",
        "n_features": 12,
        "n_types": 2,
        "families": ["MESENCHYMAL"],
    }
    prefix = str(tmp_path / "constrained")

    cli_mod._generate_text_reports(analysis, embedding_meta, prefix, decomp_results=[])

    # "constrained working subtype" + the one-line "Analysis constraints"
    # recap lived in the retired summary.md paragraph. Analysis.md
    # surfaces the constraint set in its own externally supplied /
    # Requested-context lines.
    detailed = (tmp_path / "constrained-analysis.md").read_text()
    assert "Externally supplied report label" in detailed
    assert "Requested tumor context" in detailed


def test_matched_normal_attribution_uses_decomposition_residual():
    import matplotlib

    matplotlib.use("Agg")

    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "CD74",
                "category": "surface",
                "median_est": 50.0,
                "observed_tpm": 100.0,
                "attr_tumor_tpm": 5.0,
                "matched_normal_tpm": 0.0,
                "tme_only_tpm": 0.0,
                "attribution": {"B_cell": 80.0, "macrophage": 15.0},
                "therapies": "",
                "tme_explainable": True,
            },
            {
                "symbol": "KLK3",
                "category": "surface",
                "median_est": 80.0,
                "observed_tpm": 80.0,
                "attr_tumor_tpm": 20.0,
                "matched_normal_tpm": 45.0,
                "tme_only_tpm": 15.0,
                "attribution": {
                    "matched_normal_prostate": 45.0,
                    "fibroblast": 15.0,
                },
                "therapies": "",
                "tme_explainable": False,
            },
        ]
    )

    fig = plot_mod.plot_matched_normal_attribution(
        ranges_df,
        cancer_type="PRAD",
        category="surface",
    )

    assert fig is not None
    ax = fig.axes[0]
    tumor_widths = [patch.get_width() for patch in ax.patches[:2]]
    assert tumor_widths[0] == pytest.approx(5.0)
    labels = "\n".join(label.get_text() for label in ax.get_yticklabels())
    assert "tissue-explainable" in labels
    assert "\u26a0" not in labels


def test_select_actionable_plot_genes_prefers_therapy_linked_surface_hits():
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "ITGB1",
                "observed_tpm": 2400.0,
                "attr_tumor_tpm": 2300.0,
                "category": "surface",
                "is_surface": True,
                "therapies": "",
                "therapy_supported": False,
            },
            {
                "symbol": "FAP",
                "observed_tpm": 44.0,
                "attr_tumor_tpm": 41.0,
                "category": "therapy_target",
                "is_surface": True,
                "therapies": "ADC, radioligand",
                "therapy_supported": True,
            },
        ]
    )

    genes = cli_mod._select_actionable_plot_genes(
        ranges_df,
        "SARC_OS",
        target_panel=None,
        max_genes=5,
    )

    assert "FAP" in genes
    assert "ITGB1" not in genes


def test_generate_target_report_is_mode_aware(tmp_path):
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "GENE1",
                "median_est": 12.0,
                "est_1": 8.0,
                "est_9": 15.0,
                "observed_tpm": 10.0,
                "pct_cancer_median": 1.3,
                "tcga_percentile": 0.82,
                "is_surface": True,
                "is_cta": False,
                "therapies": "CAR-T",
                "category": "therapy_target",
            }
        ]
    )
    purity = {"overall_lower": 0.9, "overall_estimate": 0.95, "overall_upper": 0.99}

    pure_prefix = str(tmp_path / "pure")
    _write_target_report(
        ranges_df,
        {
            "sample_mode": "pure",
            "mhc1": {"HLA-A": 100, "HLA-B": 100, "HLA-C": 100, "B2M": 500},
        },
        pure_prefix,
        "PRAD",
        purity,
    )
    pure_text = (tmp_path / "pure-targets.md").read_text()
    assert "Population-expression range" in pure_text
    assert "Context TPM (model)" in pure_text

    heme_prefix = str(tmp_path / "heme-targets")
    _write_target_report(
        ranges_df,
        {
            "sample_mode": "heme",
            "mhc1": {"HLA-A": 100, "HLA-B": 100, "HLA-C": 100, "B2M": 500},
        },
        heme_prefix,
        "DLBC",
        purity,
    )
    heme_text = (tmp_path / "heme-targets-targets.md").read_text()
    assert "Malignant-lineage expression range" in heme_text
    assert "Context TPM (model)" in heme_text


def test_generate_target_report_adds_tumor_context_and_landscape_summary(tmp_path):
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "MAGEA4",
                "median_est": 28.0,
                "est_1": 20.0,
                "est_9": 40.0,
                "observed_tpm": 19.0,
                "pct_cancer_median": 6.2,
                "tcga_percentile": 0.98,
                "is_surface": False,
                "is_cta": True,
                "therapies": "",
                "tme_explainable": False,
                "tme_dominant": False,
                "excluded_from_ranking": False,
                "category": "CTA",
                "matched_normal_tpm": 0.0,
                "matched_normal_tissue": "colon",
                "matched_normal_fraction": 0.18,
            },
            {
                "symbol": "CEACAM5",
                "median_est": 120.0,
                "est_1": 90.0,
                "est_9": 145.0,
                "observed_tpm": 88.0,
                "pct_cancer_median": 3.8,
                "tcga_percentile": 0.95,
                "is_surface": True,
                "is_cta": False,
                "therapies": "ADC",
                "tme_explainable": False,
                "tme_dominant": False,
                "excluded_from_ranking": False,
                "category": "therapy_target",
                "matched_normal_tpm": 14.0,
                "matched_normal_tissue": "colon",
                "matched_normal_fraction": 0.18,
            },
            {
                "symbol": "WT1",
                "median_est": 22.0,
                "est_1": 17.0,
                "est_9": 33.0,
                "observed_tpm": 16.0,
                "pct_cancer_median": 2.5,
                "tcga_percentile": 0.90,
                "is_surface": False,
                "is_cta": False,
                "therapies": "TCR-T",
                "tme_explainable": False,
                "tme_dominant": False,
                "excluded_from_ranking": False,
                "category": "therapy_target",
                "matched_normal_tpm": 0.0,
                "matched_normal_tissue": "colon",
                "matched_normal_fraction": 0.18,
            },
        ]
    )
    purity = {
        "overall_lower": 0.42,
        "overall_estimate": 0.51,
        "overall_upper": 0.63,
        "components": {
            "integration": {"signature_deprioritized": True},
        },
    }
    analysis = {
        "sample_mode": "solid",
        "cancer_type": "COAD",
        "family_summary": {"display": "CRC-family (COAD > READ)"},
        "fit_quality": {
            "label": "ambiguous",
            "message": "Top subtype candidates remain close; treat the exact site as provisional.",
        },
        "call_summary": {
            "label_options": ["COAD", "READ"],
            "label_display": "COAD or READ",
            "reported_context": "primary",
            "reported_site": "primary site",
            "site_indeterminate": False,
            "site_note": None,
            "hypothesis_display": ["COAD / solid_primary", "READ / solid_primary"],
        },
        "mhc1": {"HLA-A": 80, "HLA-B": 72, "HLA-C": 68, "B2M": 420},
    }

    prefix = str(tmp_path / "coad")
    _write_target_report(
        ranges_df,
        analysis,
        prefix,
        "COAD",
        purity,
    )

    text = (tmp_path / "coad-targets.md").read_text()
    assert "## Tumor context for interpretation" in text
    assert "## Therapy Prioritization at a Glance" in text
    assert "**Working label**: **COAD (Colon Adenocarcinoma)**" in text
    assert "Retained alternatives" in text
    assert "downstream target and biomarker interpretation below uses the working label" in text
    assert "colon-like matched-normal reference" in text
    assert "CEACAM5" in text
    assert "MAGEA4" in text
    assert "WT1" in text


def test_generate_target_report_canonicalizes_curated_antigen_symbols(
    tmp_path, monkeypatch
):
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "MAGEA4",
                "median_est": 28.0,
                "est_1": 20.0,
                "est_9": 40.0,
                "observed_tpm": 19.0,
                "pct_cancer_median": 6.2,
                "tcga_percentile": 0.98,
                "is_surface": False,
                "is_cta": True,
                "therapies": "TCR-T",
                "tme_explainable": False,
                "tme_dominant": False,
                "excluded_from_ranking": False,
                "category": "therapy_target",
                "matched_normal_tpm": 0.0,
                "matched_normal_tissue": "testis",
                "matched_normal_fraction": 0.0,
            }
        ]
    )
    targets_df = pd.DataFrame(
        [
            {
                "symbol": "MAGE-A4",
                "agent": "afami-cel",
                "agent_class": "TCR-T",
                "phase": "approved",
                "indication": "MAGE-A4+ HLA-A*02+ synovial sarcoma",
            }
        ]
    )
    monkeypatch.setattr(cli_mod, "cancer_therapy_targets", lambda *a, **k: targets_df)
    monkeypatch.setattr(cli_mod, "filter_current_therapy_targets", lambda df: df)

    purity = {
        "overall_lower": 0.42,
        "overall_estimate": 0.51,
        "overall_upper": 0.63,
        "components": {},
    }
    analysis = {
        "sample_mode": "solid",
        "cancer_type": "SARC",
        "call_summary": {"label_options": ["SARC"], "label_display": "SARC"},
        "mhc1": {"HLA-A": 80, "HLA-B": 72, "HLA-C": 68, "B2M": 420},
    }

    prefix = str(tmp_path / "sarc")
    _write_target_report(ranges_df, analysis, prefix, "SARC", purity)

    text = (tmp_path / "sarc-targets.md").read_text()
    assert "| **MAGEA4** |" in text
    assert "| **MAGE-A4** |" not in text
    assert "| MAGE-A4 | *not measured*" not in text
    assert "gene symbol not present in input file" not in text


def test_decomposition_plots_accept_reader_facing_titles_and_labels():
    best = SimpleNamespace(
        cancer_type="SARC",
        template="met_bone",
        fractions={
            "tumor": 0.80,
            "osteoblast": 0.09,
            "marrow_stroma": 0.09,
            "T_cell": 0.01,
            "endothelial": 0.01,
        },
        component_trace=pd.DataFrame(
            [
                {
                    "component": "osteoblast",
                    "fraction": 0.09,
                    "marker_score": 1.2,
                    "n_markers": 4,
                },
                {
                    "component": "marrow_stroma",
                    "fraction": 0.09,
                    "marker_score": 1.1,
                    "n_markers": 4,
                },
            ]
        ),
    )
    fig = plot_decomposition_composition(
        best,
        title="Sample composition — SARC (Sarcoma) (host context indeterminate)",
    )
    assert (
        fig.axes[0].get_title()
        == "Sample composition — SARC (Sarcoma) (host context indeterminate)"
    )

    row = SimpleNamespace(
        purity=0.80,
        template_extra_fraction=0.45,
        score=0.15,
        cancer_support_score=0.43,
        template_tissue_score=0.56,
        reconstruction_error=0.12,
        component_trace=pd.DataFrame(),
        warnings=[],
    )
    fig2 = plot_decomposition_candidates(
        [row],
        labels=["SARC (Sarcoma) bone-associated host context"],
    )
    assert (
        fig2.axes[0].get_yticklabels()[0].get_text()
        == "SARC (Sarcoma) bone-associated host context"
    )


def test_generate_target_report_filters_unreliable_rows_from_headlines(tmp_path):
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "BAD_TME",
                "median_est": 500.0,
                "est_1": 300.0,
                "est_9": 700.0,
                "observed_tpm": 200.0,
                "pct_cancer_median": 8.0,
                "tcga_percentile": 0.99,
                "is_surface": True,
                "is_cta": False,
                "therapies": "ADC",
                "tme_explainable": False,
                "tme_dominant": True,
                "excluded_from_ranking": False,
                "category": "therapy_target",
                "matched_normal_tpm": 0.0,
                "matched_normal_tissue": "colon",
                "matched_normal_fraction": 0.12,
                "attr_tumor_tpm": 15.0,
                "attr_tumor_fraction": 0.08,
                "attr_top_compartment": "fibroblast",
                "attr_top_compartment_tpm": 130.0,
                "attribution": {"fibroblast": 130.0},
                "broadly_expressed": False,
                "matched_normal_over_predicted": False,
                "smooth_muscle_stromal_leakage": False,
                "low_purity_cap_applied": False,
            },
            {
                "symbol": "BAD_BROAD",
                "median_est": 320.0,
                "est_1": 220.0,
                "est_9": 430.0,
                "observed_tpm": 180.0,
                "pct_cancer_median": 5.0,
                "tcga_percentile": 0.98,
                "is_surface": True,
                "is_cta": False,
                "therapies": "",
                "tme_explainable": False,
                "tme_dominant": False,
                "excluded_from_ranking": False,
                "category": "therapy_target",
                "matched_normal_tpm": 10.0,
                "matched_normal_tissue": "colon",
                "matched_normal_fraction": 0.12,
                "attr_tumor_tpm": 120.0,
                "attr_tumor_fraction": 0.67,
                "attr_top_compartment": "matched_normal_colon",
                "attr_top_compartment_tpm": 40.0,
                "attribution": {"matched_normal_colon": 40.0},
                "broadly_expressed": True,
                "matched_normal_over_predicted": False,
                "smooth_muscle_stromal_leakage": False,
                "low_purity_cap_applied": False,
            },
            {
                "symbol": "GOOD1",
                "median_est": 120.0,
                "est_1": 90.0,
                "est_9": 150.0,
                "observed_tpm": 95.0,
                "pct_cancer_median": 3.0,
                "tcga_percentile": 0.95,
                "is_surface": True,
                "is_cta": False,
                "therapies": "ADC",
                "tme_explainable": False,
                "tme_dominant": False,
                "excluded_from_ranking": False,
                "category": "therapy_target",
                "matched_normal_tpm": 6.0,
                "matched_normal_tissue": "colon",
                "matched_normal_fraction": 0.12,
                "attr_tumor_tpm": 80.0,
                "attr_tumor_fraction": 0.84,
                "attr_top_compartment": "matched_normal_colon",
                "attr_top_compartment_tpm": 6.0,
                "attribution": {"matched_normal_colon": 6.0},
                "broadly_expressed": False,
                "matched_normal_over_predicted": False,
                "smooth_muscle_stromal_leakage": False,
                "low_purity_cap_applied": False,
            },
        ]
    )
    analysis = {
        "sample_mode": "solid",
        "cancer_type": "COAD",
        "candidate_trace": [
            {
                "code": "COAD",
                "support_fraction_of_top": 1.0,
                "support_score": 0.4,
                "signature_score": 0.82,
                "lineage_concordance": 0.76,
            },
            {
                "code": "READ",
                "support_fraction_of_top": 0.7,
                "support_score": 0.28,
                "signature_score": 0.79,
                "lineage_concordance": 0.74,
            },
        ],
        "fit_quality": {
            "label": "ambiguous",
            "message": "Top subtype candidates remain close; treat the leading label as provisional.",
        },
        "call_summary": {
            "label_options": ["COAD", "READ"],
            "label_display": "COAD or READ",
            "reported_context": "primary",
            "reported_site": "primary site",
            "site_indeterminate": False,
            "site_note": None,
            "hypothesis_display": ["COAD / solid_primary", "READ / solid_primary"],
        },
        "purity": {
            "overall_lower": 0.32,
            "overall_estimate": 0.41,
            "overall_upper": 0.52,
            "components": {"integration": {}},
        },
        "mhc1": {"HLA-A": 80, "HLA-B": 75, "HLA-C": 70, "B2M": 400},
    }
    prefix = str(tmp_path / "filtered")
    _write_target_report(
        ranges_df,
        analysis,
        prefix,
        "COAD",
        analysis["purity"],
    )

    text = (tmp_path / "filtered-targets.md").read_text()
    context_section = text.split("## Therapy Prioritization at a Glance", 1)[1].split(
        "##", 1
    )[0]
    assert "GOOD1" in context_section
    assert "BAD_TME" not in context_section
    assert "BAD_BROAD" not in context_section
    assert "Integrated evidence synthesis" in text


def _tcga_sample(cancer_code):
    from trufflepig.reference import pan_cancer_expression

    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{cancer_code}_TPM"].astype(float),
        }
    )


def _normal_tissue_reference_sample(tissue):
    from trufflepig.reference import pan_cancer_expression

    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{tissue}_nTPM"].astype(float),
        }
    )


def test_hierarchy_embedding_keeps_coad_near_crc_family():
    df = _tcga_sample("COAD")
    matrix, labels = plot_mod._cancer_type_feature_matrix(df, method="hierarchy")

    sample = matrix[labels.index("SAMPLE")]
    coad = matrix[labels.index("COAD")]
    read = matrix[labels.index("READ")]
    prad = matrix[labels.index("PRAD")]
    dlbc = matrix[labels.index("DLBC")]

    assert np.linalg.norm(sample - coad) < np.linalg.norm(sample - prad)
    assert np.linalg.norm(sample - read) < np.linalg.norm(sample - dlbc)


def test_reference_family_matrix_scores_each_cohort_to_its_own_family():
    """Lock the reference/sample scorer unification across multiple cohorts.

    ``_reference_family_feature_matrix`` scores TCGA centroids with the same
    ENSG-keyed scorer as the sample side (one normalization basis, one
    vocabulary). Validate on several lineages so a regression in the shared path
    or the normalization basis can't pass: each cohort's own lineage family must
    score in the TOP tier of the reference family block.

    Not the strict single argmax: the pirlygenes CNS split added overlapping
    sub-lineage panels (EPENDYMAL overlaps GLIAL; some epithelial CNS panels
    carry pan-epithelial markers), which can edge the broad family on one
    max-normalized score. The embedding is robust to that because the sample
    side is scored identically, so the overlap cancels in the sample↔reference
    distance (see test_hierarchy_embedding_keeps_coad_near_crc_family) — what
    must hold here is that the own family is present in the top tier, not
    uniquely top or above a fixed absolute score.
    """
    candidate_codes, family_labels, _sites, _labels = (
        plot_mod._hierarchy_feature_labels()
    )
    matrix = plot_mod._reference_family_feature_matrix(candidate_codes, family_labels)
    code_row = {code: i for i, code in enumerate(candidate_codes)}
    fam_col = {fam: j for j, fam in enumerate(family_labels)}

    # (cohort code, family that must score in the top tier for it)
    expected = [
        ("COAD", "CRC"),
        ("READ", "CRC"),
        ("PRAD", "PROSTATE"),
        ("GBM", "GLIAL"),
        ("LGG", "GLIAL"),
        ("DLBC", "HEME_BCELL"),
    ]
    for code, family in expected:
        if code not in code_row or family not in fam_col:
            continue
        row = matrix[code_row[code]]
        own = row[fam_col[family]]
        rank = int((row > own).sum())  # 0 == argmax
        assert rank < 3 and own > 0.0, (
            f"{code}: family {family!r} scored {own:.3f} at rank {rank} "
            f"(max {row.max():.3f}) — expected present in top tier (rank<3)"
        )


def test_hierarchy_embedding_metadata_reports_feature_space():
    meta = plot_mod.get_embedding_feature_metadata(method="hierarchy")

    assert meta["method"] == "hierarchy"
    assert meta["feature_kind"] == "hierarchical_scores"
    assert meta["n_features"] > 0
    assert "CRC" in meta["families"]
    assert "sites" in meta
    assert "lymph_node" in meta["sites"]


def test_plot_tumor_purity_is_mode_aware(monkeypatch):
    mock_result = {
        "cancer_type": "DLBC",
        "tcga_median_purity": None,
        "overall_estimate": 0.81,
        "overall_lower": 0.71,
        "overall_upper": 0.89,
        "components": {
            "signature": {
                "per_gene": [{"gene": "CD19", "purity": 0.8}],
                "purity": 0.8,
                "lower": 0.7,
                "upper": 0.9,
                "genes": ["CD19"],
            },
            "stromal": {"n_genes": 5, "enrichment": 1.2},
            "immune": {"n_genes": 5, "enrichment": 4.8},
            "estimate_purity": 0.75,
        },
    }
    monkeypatch.setattr(
        purity_mod, "estimate_tumor_purity", lambda *a, **k: mock_result
    )

    fig, result = purity_mod.plot_tumor_purity(
        pd.DataFrame({"gene_id": ["ENSG1"], "gene_display_name": ["A"], "TPM": [1.0]}),
        cancer_type="DLBC",
        sample_mode="heme",
    )
    assert result["cancer_type"] == "DLBC"
    assert fig.axes[0].get_xlabel() == "Fraction estimate (%)"
    assert fig.axes[1].get_title() == "Fraction / context components"
    assert "TCGA median purity not available" in fig.axes[0].get_title()
    assert "Malignant-lineage fraction estimate" in fig._suptitle.get_text()


def test_plot_sample_summary_is_mode_aware(monkeypatch):
    mock_analysis = {
        "cancer_type": "DLBC",
        "cancer_name": "Diffuse Large B-Cell Lymphoma",
        "top_cancers": [("DLBC", 0.42)],
        "purity": {
            "overall_estimate": 0.81,
            "overall_lower": 0.71,
            "overall_upper": 0.89,
            "tcga_median_purity": 0.94,
            "components": {
                "stromal": {"enrichment": 1.2},
                "immune": {"enrichment": 4.8},
            },
        },
        "tissue_scores": [("lymph_node", 0.93, 20), ("spleen", 0.78, 20)],
        "mhc1": {"HLA-A": 40, "HLA-B": 55, "HLA-C": 30, "B2M": 200},
        "mhc2": {},
        "candidate_trace": [{"code": "DLBC"}],
    }
    monkeypatch.setattr(purity_mod, "analyze_sample", lambda *a, **k: mock_analysis)

    fig, analysis = purity_mod.plot_sample_summary(
        pd.DataFrame({"gene_id": ["ENSG1"], "gene_display_name": ["A"], "TPM": [1.0]}),
        cancer_type="DLBC",
        sample_mode="heme",
    )
    assert analysis["cancer_type"] == "DLBC"
    assert fig.axes[1].get_title() == "Heme Composition Context"
    assert fig.axes[2].get_title().startswith("Lineage / Background Context")
    assert "hematologic / lymphoid bulk" in fig._suptitle.get_text()


def test_plot_sample_summary_distinguishes_final_call_from_ranker_leader():
    mock_analysis = {
        "cancer_type": "BRCA",
        "cancer_name": "Breast Invasive Carcinoma",
        "top_cancers": [("HL", 1.0), ("BRCA", 0.47)],
        "fit_quality": {"label": "ambiguous"},
        "purity": {
            "overall_estimate": 0.81,
            "overall_lower": 0.71,
            "overall_upper": 0.89,
            "tcga_median_purity": 0.73,
            "components": {
                "stromal": {"enrichment": 1.2},
                "immune": {"enrichment": 0.8},
            },
        },
        "tissue_scores": [("breast", 0.93, 20)],
        "mhc1": {"HLA-A": 40, "HLA-B": 55, "HLA-C": 30, "B2M": 200},
        "mhc2": {},
        "candidate_trace": [{"code": "HL"}, {"code": "BRCA"}],
    }

    fig, _analysis = purity_mod.plot_sample_summary(
        pd.DataFrame(
            {"gene_id": ["ENSG1"], "gene_display_name": ["A"], "TPM": [1.0]}
        ),
        cancer_type="BRCA",
        sample_mode="solid",
        analysis=mock_analysis,
    )

    ranker_panel = fig.axes[0]
    assert ranker_panel.get_title() == (
        "Pre-adjudication cancer-type ranker (ambiguous fit)"
    )
    panel_text = "\n".join(text.get_text() for text in ranker_panel.texts)
    assert "Final report call: BRCA" in panel_text
    assert "Pre-adjudication ranker leader: HL" in panel_text
    assert "Lead label: HL" not in panel_text


def test_plot_sample_summary_allows_missing_reference_purity(monkeypatch):
    """A loadable report must not depend on optional cohort-purity metadata."""
    mock_analysis = {
        "cancer_type": "SARC_LPS_UNSPEC",
        "cancer_name": "Liposarcoma, unspecified",
        "top_cancers": [("SARC_LPS_UNSPEC", 0.42)],
        "purity": {
            "overall_estimate": 0.61,
            "overall_lower": 0.48,
            "overall_upper": 0.72,
            "tcga_median_purity": None,
            "components": {
                "stromal": {"enrichment": 1.2},
                "immune": {"enrichment": 0.8},
            },
        },
        "tissue_scores": [("adipose", 0.93, 20)],
        "mhc1": {"HLA-A": 40, "HLA-B": 55, "HLA-C": 30, "B2M": 200},
        "mhc2": {},
        "candidate_trace": [{"code": "SARC_LPS_UNSPEC"}],
    }
    monkeypatch.setattr(purity_mod, "analyze_sample", lambda *a, **k: mock_analysis)

    fig, _analysis = purity_mod.plot_sample_summary(
        pd.DataFrame({"gene_id": ["ENSG1"], "gene_display_name": ["A"], "TPM": [1.0]}),
        cancer_type="SARC_LPS_UNSPEC",
        sample_mode="solid",
    )

    purity_text = "\n".join(text.get_text() for text in fig.axes[1].texts)
    assert "median purity: not available" in purity_text


@pytest.mark.parametrize(
    ("cancer_type", "expected_label"),
    [
        (
            "SARC_LPS_UNSPEC",
            "parent cohort (SARC; requested SARC_LPS_UNSPEC)",
        ),
        (
            "NUTM",
            "Mean across available pan-cancer cohorts "
            "(no NUTM pan-cancer cohort)",
        ),
    ],
)
def test_scatter_uses_an_honest_available_reference_fallback(
    monkeypatch,
    cancer_type,
    expected_label,
):
    """Report plots remain loadable when a leaf has no pan-cancer column."""
    ref = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG1"],
            "Symbol": ["GENE1"],
            "SARC_TPM": [2.0],
            "LUAD_TPM": [4.0],
        }
    )
    monkeypatch.setattr(
        plot_scatter_mod,
        "pan_cancer_expression",
        lambda **_kwargs: ref.copy(),
    )
    monkeypatch.setattr(
        plot_scatter_mod,
        "housekeeping_gene_ids",
        lambda: {"ENSG1"},
    )

    _plot_df, _cats, _colors, _sample_label, cohort_label = (
        plot_scatter_mod._prepare_sample_vs_cancer_data(
            pd.DataFrame(
                {
                    "gene_id": ["ENSG1"],
                    "gene_display_name": ["GENE1"],
                    "TPM": [10.0],
                }
            ),
            {"markers": ["ENSG1"]},
            cancer_type,
        )
    )

    assert expected_label in cohort_label


def test_hierarchy_embedding_plot_adds_family_legend_and_neighbors(monkeypatch):
    import trufflepig.plot_embedding as _pe

    monkeypatch.setattr(_pe, "adjust_text", lambda *a, **k: None)
    coords = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [1.5, 1.3],
            [0.1, 0.05],
        ]
    )
    labels = ["COAD", "READ", "PRAD", "SAMPLE"]

    fig, ax = plot_mod._plot_embedding_with_labels(
        coords,
        labels,
        title="Test",
        xlabel="x",
        ylabel="y",
        method="hierarchy",
    )

    assert ax.get_legend() is not None
    assert ax.get_legend().get_title().get_text() == "Family"
    all_text = "\n".join(text.get_text() for text in ax.texts)
    assert "Nearest by plotted 2D distance" in all_text
    assert "COAD" in all_text


def test_embedding_plot_can_label_nearest_cancers_and_normals_only(monkeypatch):
    import trufflepig.plot_embedding as _pe

    monkeypatch.setattr(_pe, "adjust_text", lambda *a, **k: None)
    coords = np.array(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [0.1, 0.0],
            [3.0, 3.0],
            [0.05, 0.05],
        ]
    )
    labels = ["PRAD", "COAD", "normal:prostate", "normal:liver", "SAMPLE"]

    _fig, ax = plot_mod._plot_embedding_with_labels(
        coords,
        labels,
        title="Test",
        xlabel="x",
        ylabel="y",
        label_nearest_cancers=1,
        label_nearest_normals=1,
        label_all=False,
    )

    all_text = "\n".join(text.get_text() for text in ax.texts)
    assert "PRAD" in all_text
    assert "prostate" in all_text
    assert "COAD" not in all_text
    assert "liver" not in all_text
    assert "Nearest normal tissues" in all_text
    assert "Nearest by plotted 2D distance" in all_text


def test_embedding_plot_can_use_feature_distance_neighbors(monkeypatch):
    import trufflepig.plot_embedding as _pe

    monkeypatch.setattr(_pe, "adjust_text", lambda *a, **k: None)
    coords = np.array(
        [
            [10.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.0],
        ]
    )
    labels = ["COAD", "PRAD", "SAMPLE"]

    _fig, ax = plot_mod._plot_embedding_with_labels(
        coords,
        labels,
        title="Test",
        xlabel="x",
        ylabel="y",
        label_nearest_cancers=1,
        label_all=False,
        nearest_neighbors=[
            (0.2, "COAD", "cancer"),
            (9.0, "PRAD", "cancer"),
        ],
        nearest_basis="input feature distance",
    )

    all_text = "\n".join(text.get_text() for text in ax.texts)
    assert "COAD" in all_text
    assert "PRAD" not in all_text
    assert "Nearest by input feature distance" in all_text


def test_mds_can_focus_on_sample_neighborhood(monkeypatch):
    import trufflepig.plot_embedding as _pe

    captured = {}

    def fake_feature_matrix(*_args, **_kwargs):
        return np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [20.0, 20.0],
                [0.0, 1.0],
                [20.0, 21.0],
                [0.1, 0.1],
            ]
        ), ["COAD", "READ", "PRAD", "normal:colon", "normal:liver", "SAMPLE"]

    def fake_plot(coords, labels, **kwargs):
        captured["coords"] = coords
        captured["labels"] = labels
        captured["kwargs"] = kwargs
        return None, None

    monkeypatch.setattr(_pe, "_cancer_type_feature_matrix", fake_feature_matrix)
    monkeypatch.setattr(_pe, "_plot_embedding_with_labels", fake_plot)

    _pe.plot_cancer_type_mds(
        pd.DataFrame({"gene_symbol": ["A"], "ensembl_gene_id": ["ENSG1"], "TPM": [1]}),
        include_normals=True,
        include_subtypes=True,
        focus_nearest_cancers=1,
        focus_nearest_normals=1,
    )

    assert captured["labels"] == ["COAD", "normal:colon", "SAMPLE"]
    assert captured["coords"].shape[0] == 3
    assert captured["kwargs"]["nearest_basis"] == "input feature distance"
    assert captured["kwargs"]["nearest_neighbors"][0][1] == "COAD"


def test_reference_neighborhood_preserves_sample_distances(monkeypatch):
    import trufflepig.plot_embedding as _pe

    def fake_feature_matrix(*_args, **_kwargs):
        return np.array(
            [
                [1.0, 0.0],
                [4.0, 0.0],
                [0.0, 3.0],
                [0.0, 0.0],
            ]
        ), ["COAD", "READ", "normal:colon", "SAMPLE"]

    monkeypatch.setattr(_pe, "_cancer_type_feature_matrix", fake_feature_matrix)

    fig, ax = _pe.plot_cancer_type_neighborhood(
        pd.DataFrame({"gene_symbol": ["A"], "ensembl_gene_id": ["ENSG1"], "TPM": [1]}),
        focus_nearest_cancers=None,
        focus_nearest_normals=None,
    )

    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert labels == ["COAD", "READ", "colon"]
    assert "Nearest reference distances" in ax.get_title()
    assert "Input feature-space distance" in ax.get_xlabel()
    widths = [float(patch.get_width()) for patch in ax.patches]
    assert widths[0] < widths[1]
    assert widths[0] < widths[2]
    fig.clf()


def test_embedding_matrix_sanitizes_nonfinite_features():
    matrix = np.array(
        [
            [1.0, np.nan, 2.0, np.inf],
            [2.0, np.nan, 3.0, 4.0],
            [3.0, np.nan, 4.0, 5.0],
        ]
    )

    clean = plot_mod._sanitize_embedding_matrix(matrix)

    assert clean.shape == (3, 2)
    assert np.isfinite(clean).all()


def test_embedding_can_include_available_subtype_references():
    df = _tcga_sample("SARC")
    matrix, labels = plot_mod._cancer_type_feature_matrix(
        df,
        method="tme",
        include_subtypes=True,
    )

    assert matrix.shape[0] == len(labels)
    assert "SARC_LMS" in labels
    assert "SARC_OS" in labels
    assert np.isfinite(matrix).all()


def test_pan_reference_metadata_includes_normal_tissue_gene_sets():
    meta = plot_mod.get_embedding_feature_metadata(method="panref", n_genes=4)

    assert meta["method"] == "panref"
    assert meta["feature_kind"] == "pan_reference_genes"
    # The packaged cancer reference is intentionally additive; new qualified
    # cohorts must not break a normal-tissue metadata contract test.
    assert meta["n_types"] == sum(bool(genes) for genes in meta["per_type"].values())
    assert meta["n_types"] >= 33
    assert meta["n_normals"] >= 40
    assert meta["per_type"]["COAD"]
    assert meta["per_normal"]["colon"]
    assert "PTPRC" in meta["anchor_added"] or any(
        "PTPRC" in genes for genes in meta["per_normal"].values()
    )


def test_pan_reference_gene_set_api_exposes_selection_context():
    from trufflepig.plot_embedding import pan_reference_embedding_genes

    genes = pan_reference_embedding_genes(n_genes_per_type=4)

    assert {
        "Ensembl_Gene_ID",
        "Symbol",
        "selected_for_cancer_refs",
        "selected_for_normal_refs",
        "curated_anchor",
    }.issubset(genes.columns)
    assert genes["Symbol"].is_unique
    assert (genes["selected_for_cancer_refs"].str.contains("COAD")).any()
    assert (genes["selected_for_normal_refs"].str.contains("colon")).any()


def test_pan_reference_embedding_handles_cancer_and_normal_references():
    normal_colon = _normal_tissue_reference_sample("colon")
    matrix, labels = plot_mod._cancer_type_feature_matrix(
        normal_colon,
        method="panref",
        n_genes=4,
        include_normals=True,
        include_subtypes=False,
    )

    sample = matrix[labels.index("SAMPLE")]
    colon = matrix[labels.index("normal:colon")]
    liver = matrix[labels.index("normal:liver")]
    prad = matrix[labels.index("PRAD")]

    assert np.linalg.norm(sample - colon) < np.linalg.norm(sample - liver)
    assert np.linalg.norm(sample - colon) < np.linalg.norm(sample - prad)
    assert np.isfinite(matrix).all()


def test_singleton_family_is_not_rendered_as_family_call():
    summary = _summarize_candidate_family(
        [
            {"code": "PRAD", "family_label": "PROSTATE", "support_score": 0.4},
            {"code": "DLBC", "family_label": None, "support_score": 0.1},
        ]
    )
    assert summary["label"] == "PROSTATE"
    assert summary["display"] is None
    assert summary["subtype_clause"] is None


def test_collect_ranked_therapy_targets_tracks_multicategory_and_approval(monkeypatch):
    df = pd.DataFrame(
        {
            "gene_id": ["ENSG_A", "ENSG_B", "ENSG_C"],
            "gene_display_name": ["GENEA", "GENEB", "GENEC"],
            "TPM": [120.0, 90.0, 60.0],
        }
    )

    therapy_maps = {
        "ADC": {"ENSG_B": "GENEB"},
        "ADC-approved": {"ENSG_B": "GENEB"},
        "CAR-T": {"ENSG_A": "GENEA"},
        "CAR-T-approved": {},
        "TCR-T": {},
        "TCR-T-approved": {},
        "bispecific-antibodies": {"ENSG_A": "GENEA"},
        "bispecific-antibodies-approved": {},
        "radioligand": {"ENSG_C": "GENEC"},
    }

    import trufflepig.plot_therapy as _pt

    monkeypatch.setattr(
        _pt,
        "therapy_target_gene_id_to_name",
        lambda therapy: therapy_maps.get(therapy, {}),
    )
    monkeypatch.setattr(
        _pt,
        "get_data",
        lambda name: pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["ENSG_C"],
                "Status_Bucket": ["FDA_approved"],
            }
        )
        if name == "radioligand-targets"
        else pd.DataFrame(),
    )

    out = plot_mod._collect_ranked_therapy_targets(df, top_k=1, tpm_threshold=10)

    assert [row["gene_id"] for row in out] == ["ENSG_A", "ENSG_B", "ENSG_C"]
    assert out[0]["therapies"] == ("CAR-T", "bispecific-antibodies")
    assert out[0]["has_approved"] is True
    assert out[0]["approved_therapies"] == ("CAR-T",)
    assert out[1]["therapies"] == ("ADC",)
    assert out[1]["approved_therapies"] == ("ADC",)
    assert out[2]["therapies"] == ("radioligand",)
    assert out[2]["approved_therapies"] == ("radioligand",)


def test_collect_ranked_therapy_targets_force_includes_extra_symbols(monkeypatch):
    df = pd.DataFrame(
        {
            "gene_id": ["ENSG_A", "ENSG_B"],
            "gene_display_name": ["GENEA", "KLK2"],
            "TPM": [120.0, 2.0],
        }
    )

    import trufflepig.plot_therapy as _pt

    monkeypatch.setattr(
        _pt,
        "therapy_target_gene_id_to_name",
        lambda therapy: {"ENSG_A": "GENEA"} if therapy == "ADC" else {},
    )
    monkeypatch.setattr(
        _pt,
        "get_data",
        lambda name: pd.DataFrame(
            {"Ensembl_Gene_ID": [], "Status_Bucket": []}
        ),
    )

    out = plot_mod._collect_ranked_therapy_targets(
        df,
        top_k=1,
        tpm_threshold=30,
        extra_symbols={"KLK2"},
    )

    assert {row["symbol"] for row in out} == {"GENEA", "KLK2"}
    klk2 = next(row for row in out if row["symbol"] == "KLK2")
    assert klk2["sample_tpm"] == 2.0
    assert klk2["therapies"] == ()


def test_resolve_gene_set_symbols_by_category_name():
    """String input resolves via the Category column in data/gene-sets.csv."""
    syms = plot_mod._resolve_gene_set_symbols("Interferon_response")
    assert isinstance(syms, list) and len(syms) > 5
    # Case / whitespace insensitive
    syms2 = plot_mod._resolve_gene_set_symbols("interferon response")
    assert set(syms) == set(syms2)


def test_resolve_gene_set_symbols_iterable_passthrough():
    out = plot_mod._resolve_gene_set_symbols(["IRF1", "STAT1"])
    assert out == ["IRF1", "STAT1"]


def test_resolve_gene_set_symbols_unknown_raises():
    with pytest.raises(ValueError):
        plot_mod._resolve_gene_set_symbols("this_gene_set_does_not_exist")


def test_plot_geneset_vs_vital_tissues_saves_png(tmp_path):
    """The toxicity view renders a non-empty PNG from a minimal sample.

    Uses a handful of real symbols so the ref-expression lookup path
    exercises (they are in pan_cancer_expression).
    """
    import matplotlib

    matplotlib.use("Agg")

    sample = pd.DataFrame(
        {
            "gene_id": [
                "ENSG00000125347",  # IRF1
                "ENSG00000115415",  # STAT1
                "ENSG00000081059",  # TCF7 (not in IFN set; shouldn't show)
                "ENSG00000165949",  # IFI27
            ],
            "gene_name": ["IRF1", "STAT1", "TCF7", "IFI27"],
            "TPM": [85.0, 110.0, 12.0, 40.0],
        }
    )
    out = tmp_path / "ifn_vs_vitals.png"
    fig = plot_mod.plot_geneset_vs_vital_tissues(
        sample,
        gene_set=["IRF1", "STAT1", "IFI27"],
        title="IFN vs vital tissues (test)",
        save_to_filename=str(out),
    )
    assert fig is not None
    assert out.exists()
    assert out.stat().st_size > 5_000


def test_plot_geneset_vs_vital_tissues_empty_symbols_returns_none(capsys):
    """Empty gene list should return None and print, not raise."""
    sample = pd.DataFrame(
        {
            "gene_id": ["ENSG00000125347"],
            "gene_name": ["IRF1"],
            "TPM": [85.0],
        }
    )
    fig = plot_mod.plot_geneset_vs_vital_tissues(sample, gene_set=[])
    assert fig is None


def test_plot_geneset_vs_vital_tissues_all_absent_returns_none(capsys):
    """All genes absent from both sample and reference → None, not crash."""
    sample = pd.DataFrame(
        {
            "gene_id": ["ENSG00000125347"],
            "gene_name": ["IRF1"],
            "TPM": [85.0],
        }
    )
    fig = plot_mod.plot_geneset_vs_vital_tissues(
        sample, gene_set=["SOME_GENE_THAT_DOES_NOT_EXIST_ANYWHERE_XYZ"]
    )
    assert fig is None


def test_plot_geneset_vs_vital_tissues_unknown_tissue_raises():
    sample = pd.DataFrame(
        {
            "gene_id": ["ENSG00000125347"],
            "gene_name": ["IRF1"],
            "TPM": [85.0],
        }
    )
    with pytest.raises(ValueError):
        plot_mod.plot_geneset_vs_vital_tissues(
            sample, gene_set=["IRF1"], vital_tissues=["mars"]
        )


def test_plot_ctas_vs_cancer_type_detail_saves_png(tmp_path):
    """End-to-end render with real CTA Ensembl IDs → non-empty PNG."""
    import matplotlib

    matplotlib.use("Agg")
    from pirlygenes.gene_sets_cancer import CTA_gene_id_to_name

    cta_map = CTA_gene_id_to_name()
    assert cta_map, "Reference CTA set is unexpectedly empty"
    picks = list(cta_map.items())[:8]

    sample = pd.DataFrame(
        {
            "gene_id": [gid for gid, _ in picks],
            "gene_name": [sym for _, sym in picks],
            "TPM": [5.0, 25.0, 1.5, 80.0, 3.0, 12.0, 2.0, 60.0][: len(picks)],
        }
    )
    out = tmp_path / "prad_cta_detail.png"
    fig = plot_mod.plot_ctas_vs_cancer_type_detail(
        sample,
        cancer_type="PRAD",
        top_k=6,
        save_to_filename=str(out),
    )
    assert fig is not None
    assert out.exists()
    assert out.stat().st_size > 5_000


def test_plot_ctas_vs_cancer_type_detail_worst_vital_excludes_testis_and_thymus(
    monkeypatch,
):
    """CTA max-vital tissue excludes reproductive/immune-privileged tissues."""
    import matplotlib

    matplotlib.use("Agg")
    import trufflepig.plot_therapy as therapy_mod

    monkeypatch.setattr(
        therapy_mod,
        "CTA_gene_id_to_name",
        lambda: {"ENSGCTA": "CTA1"},
    )
    monkeypatch.setattr(
        therapy_mod,
        "pan_cancer_expression",
        lambda **_: pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["ENSGCTA"],
                "Symbol": ["CTA1"],
                "PRAD_TPM": [1.0],
                "prostate_nTPM": [0.5],
                "testis_nTPM": [300.0],
                "thymus_nTPM": [250.0],
                "heart_muscle_nTPM": [35.0],
                "liver_nTPM": [4.0],
            }
        ),
    )

    sample = pd.DataFrame(
        {
            "gene_id": ["ENSGCTA"],
            "gene_name": ["CTA1"],
            "TPM": [50.0],
        }
    )
    fig = therapy_mod.plot_ctas_vs_cancer_type_detail(
        sample,
        cancer_type="PRAD",
        min_sample_tpm=1.0,
    )

    text = "\n".join(t.get_text() for t in fig.axes[0].texts)
    assert "heart 35" in text
    assert "thymus" not in text
    assert "testis" not in text


def test_plot_ctas_vs_cancer_type_detail_min_sample_tpm_filters_rows():
    """Rows below `min_sample_tpm` are dropped; all-below → None."""
    import matplotlib

    matplotlib.use("Agg")
    from pirlygenes.gene_sets_cancer import CTA_gene_id_to_name

    cta_map = CTA_gene_id_to_name()
    picks = list(cta_map.items())[:3]
    sample = pd.DataFrame(
        {
            "gene_id": [gid for gid, _ in picks],
            "gene_name": [sym for _, sym in picks],
            "TPM": [0.1, 0.2, 0.3],
        }
    )
    fig = plot_mod.plot_ctas_vs_cancer_type_detail(
        sample,
        cancer_type="PRAD",
        min_sample_tpm=1.0,
    )
    assert fig is None


def test_plot_ctas_vs_cancer_type_detail_unknown_type_raises():
    sample = pd.DataFrame(
        {
            "gene_id": ["ENSG00000125347"],
            "gene_name": ["IRF1"],
            "TPM": [10.0],
        }
    )
    with pytest.raises((ValueError, KeyError)):
        plot_mod.plot_ctas_vs_cancer_type_detail(
            sample,
            cancer_type="NOT_A_REAL_CANCER_TYPE_XYZ",
        )
