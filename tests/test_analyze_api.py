"""Contracts for the structured analyze API boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from trufflepig.analyze import (
    AnalyzeConfig,
    build_analysis_parameters,
    build_analyze_comparison_markdown,
    build_analyze_paths,
    apply_sample_context_to_purity,
    resolve_analyze_inputs,
    should_adopt_decomposition_purity,
)
from trufflepig.sample_context import SampleContext


def test_resolve_analyze_inputs_auto_transcript():
    config = AnalyzeConfig(input_path="quant.sf")

    resolved = resolve_analyze_inputs(
        config,
        sniff_input_level=lambda path: "transcript",
    )

    assert resolved.gene_input == "quant.sf"
    assert resolved.transcript_input is None
    assert resolved.aggregate_gene_expression is True
    assert resolved.input_level == "transcript"
    assert resolved.notes


def test_resolve_analyze_inputs_explicit_pair_does_not_sniff():
    config = AnalyzeConfig(
        input_path="ignored.csv",
        genes="gene.csv",
        transcripts="transcript.sf",
    )

    def _unexpected_sniff(_path):
        raise AssertionError("explicit inputs should not be sniffed")

    resolved = resolve_analyze_inputs(config, sniff_input_level=_unexpected_sniff)

    assert resolved.gene_input == "gene.csv"
    assert resolved.transcript_input == "transcript.sf"
    assert resolved.aggregate_gene_expression is False
    assert resolved.input_level == "explicit"


def test_build_analyze_paths_centralizes_prefix(tmp_path: Path):
    config = AnalyzeConfig(
        input_path="/data/example/quant.sf",
        output_dir=str(tmp_path),
        output_image_prefix=None,
        sample_id_value="sample 1",
    )
    resolution = resolve_analyze_inputs(
        config,
        sniff_input_level=lambda _path: "transcript",
    )

    paths = build_analyze_paths(
        config,
        resolution,
        default_output_dir=lambda: "unused",
        derive_sample_display_id=lambda path, sample_id_value=None: "sample-1",
        sanitize_output_basename=lambda value: str(value).replace(" ", "-")
        if value
        else "",
    )

    assert paths.out_dir == tmp_path
    assert paths.sample_display_id == "sample-1"
    assert paths.prefix == str(tmp_path / "sample-1")
    assert paths.file("summary.md") == str(tmp_path / "sample-1-summary.md")


def test_build_analysis_parameters_records_cancer_type_source():
    quality = {
        "degradation": {"level": "unknown", "long_short_ratio": None},
        "culture": {"level": "unknown", "stress_score": None},
        "has_issues": False,
    }

    constrained = AnalyzeConfig(input_path="gene.tsv", cancer_type="COAD")
    constrained_resolution = resolve_analyze_inputs(
        constrained,
        sniff_input_level=lambda _path: "gene",
    )
    constrained_params = build_analysis_parameters(
        config=constrained,
        resolution=constrained_resolution,
        template_overrides=[],
        selected_sample_mode="bulk",
        quality=quality,
        tumor_purity_parameters={},
        decomposition_parameters={},
    )

    inferred = AnalyzeConfig(input_path="gene.tsv")
    inferred_resolution = resolve_analyze_inputs(
        inferred,
        sniff_input_level=lambda _path: "gene",
    )
    inferred_params = build_analysis_parameters(
        config=inferred,
        resolution=inferred_resolution,
        template_overrides=[],
        selected_sample_mode="bulk",
        quality=quality,
        tumor_purity_parameters={},
        decomposition_parameters={},
    )

    assert constrained_params["input"]["cancer_type_source"] == "user-specified"
    assert inferred_params["input"]["cancer_type_source"] == "auto-detected"


def test_build_analysis_parameters_records_hla_types():
    quality = {
        "degradation": {"level": "unknown", "long_short_ratio": None},
        "culture": {"level": "unknown", "stress_score": None},
        "has_issues": False,
    }
    config = AnalyzeConfig(input_path="gene.tsv", hla_types="HLA-A*02:01, A24:02")
    resolution = resolve_analyze_inputs(config, sniff_input_level=lambda _path: "gene")

    params = build_analysis_parameters(
        config=config,
        resolution=resolution,
        template_overrides=[],
        selected_sample_mode="bulk",
        quality=quality,
        tumor_purity_parameters={},
        decomposition_parameters={},
    )

    assert config.hla_type_list() == ["A*02:01", "A*24:02"]
    assert params["input"]["hla_types"] == ["A*02:01", "A*24:02"]


def test_build_analysis_parameters_records_fusion_paths():
    quality = {
        "degradation": {"level": "unknown", "long_short_ratio": None},
        "culture": {"level": "unknown", "stress_score": None},
        "has_issues": False,
    }
    config = AnalyzeConfig(input_path="gene.tsv", fusions="calls.tsv;extra.jsonl")
    resolution = resolve_analyze_inputs(config, sniff_input_level=lambda _path: "gene")

    params = build_analysis_parameters(
        config=config,
        resolution=resolution,
        template_overrides=[],
        selected_sample_mode="bulk",
        quality=quality,
        tumor_purity_parameters={},
        decomposition_parameters={},
    )

    assert config.fusion_path_list() == ["calls.tsv", "extra.jsonl"]
    assert params["input"]["fusions"] == ["calls.tsv", "extra.jsonl"]


def test_build_analysis_parameters_records_alteration_inputs():
    quality = {
        "degradation": {"level": "unknown", "long_short_ratio": None},
        "culture": {"level": "unknown", "stress_score": None},
        "has_issues": False,
    }
    config = AnalyzeConfig(input_path="gene.tsv", alterations="EGFR KDD;variants.tsv")
    resolution = resolve_analyze_inputs(config, sniff_input_level=lambda _path: "gene")

    params = build_analysis_parameters(
        config=config,
        resolution=resolution,
        template_overrides=[],
        selected_sample_mode="bulk",
        quality=quality,
        tumor_purity_parameters={},
        decomposition_parameters={},
    )

    assert config.alteration_input_list() == ["EGFR KDD", "variants.tsv"]
    assert params["input"]["alterations"] == ["EGFR KDD", "variants.tsv"]


def test_registry_only_cancer_label_becomes_report_scope():
    from trufflepig.main import _analysis_input_cancer_type

    composition_scope, report_scope = _analysis_input_cancer_type("NUT carcinoma")

    assert composition_scope is None
    assert report_scope == "NUTM"


def test_registry_child_cancer_label_constrains_parent_cohort():
    from trufflepig.main import _analysis_input_cancer_type

    composition_scope, report_scope = _analysis_input_cancer_type("SARC_SYN")

    assert composition_scope == "SARC"
    assert report_scope == "SARC_SYN"


def test_registry_child_cancer_name_constrains_parent_cohort():
    from trufflepig.main import _analysis_input_cancer_type

    composition_scope, report_scope = _analysis_input_cancer_type("Synovial Sarcoma")

    assert composition_scope == "SARC"
    assert report_scope == "SARC_SYN"


def test_local_reference_only_cancer_label_becomes_report_scope():
    from trufflepig.main import _analysis_input_cancer_type

    composition_scope, report_scope = _analysis_input_cancer_type("SARC_OS")

    # OS is now the sarcoma subtype SARC_OS with SARC as its coarse parent
    # (5.13 registry restructure), so composition scopes to SARC while the fine
    # label still becomes the report scope.
    assert composition_scope == "SARC"
    assert report_scope == "SARC_OS"


def test_nutm1_expression_can_infer_registry_only_report_scope():
    import pandas as pd

    from trufflepig.main import _infer_registry_report_scope_from_rna

    df = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000184507"],
            "canonical_gene_name": ["NUTM1"],
            "TPM": [6.2],
        }
    )
    analysis = {"candidate_trace": [{"code": "LUSC"}]}

    inference = _infer_registry_report_scope_from_rna(df, analysis)

    assert inference["cancer_type"] == "NUTM"
    assert inference["surrogate"] == "NUTM1"
    assert inference["top_reference_cancer_type"] == "LUSC"
    assert "hypothesis" in inference["caveat"].lower()


def test_rare_rna_surrogate_rules_are_data_backed_and_context_gated():
    import pandas as pd

    from pirlygenes.gene_sets_cancer import rare_cancer_rna_surrogate_rules_df
    from trufflepig.rare_inference import (
        infer_rare_cancer_marker_hypotheses_from_rna,
        infer_rare_cancer_report_scope_from_rna,
    )

    rules = rare_cancer_rna_surrogate_rules_df()
    assert {"NUTM", "SARC_CHOR", "ACINIC"}.issubset(set(rules["cancer_code"]))

    tbxt_only = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000164458"],
            "canonical_gene_name": ["TBXT"],
            "TPM": [12.0],
        }
    )
    assert (
        infer_rare_cancer_report_scope_from_rna(
            tbxt_only,
            {"candidate_trace": [{"code": "SARC"}]},
        )
        is None
    )
    marker_prompts = infer_rare_cancer_marker_hypotheses_from_rna(
        tbxt_only,
        {"candidate_trace": [{"code": "SARC"}]},
    )
    assert marker_prompts[0]["cancer_type"] == "SARC_CHOR"
    assert marker_prompts[0]["surrogate"] == "TBXT"
    assert "KRT19" in marker_prompts[0]["missing_support_genes"]

    germ_cell_context = infer_rare_cancer_report_scope_from_rna(
        tbxt_only,
        {"candidate_trace": [{"code": "TGCT"}]},
    )
    assert germ_cell_context is None

    nutm_like = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000184507"],
            "canonical_gene_name": ["NUTM1"],
            "TPM": [6.2],
        }
    )
    inference = infer_rare_cancer_report_scope_from_rna(
        nutm_like,
        {"candidate_trace": [{"code": "LUSC"}]},
    )
    assert inference["cancer_type"] == "NUTM"
    assert inference["surrogate"] == "NUTM1"


def test_fusion_parser_preserves_5prime_3prime_orientation(tmp_path: Path):
    from trufflepig.fusions import parse_fusion_file

    fusion_file = tmp_path / "fusions.csv"
    fusion_file.write_text(
        "\n".join(
            [
                "gene5,gene3,effect,total_support,fusion_caller,reportable",
                "BRD4,NUTM1,in-frame,288,STAR-Fusion,true",
            ]
        )
    )

    records = parse_fusion_file(fusion_file)

    assert len(records) == 1
    assert records[0].gene_a == "BRD4"
    assert records[0].gene_b == "NUTM1"
    assert records[0].orientation == "5prime_3prime"
    assert records[0].support_total == 288


def test_fusion_parser_rejects_missing_supplied_file(tmp_path: Path):
    from trufflepig.fusions import parse_fusion_file

    missing = tmp_path / "missing-fusions.tsv"

    with pytest.raises(FileNotFoundError, match="Fusion evidence file not found"):
        parse_fusion_file(missing)


def test_alteration_parser_accepts_inline_kdd():
    from trufflepig.alterations import parse_alteration_inputs

    records = parse_alteration_inputs("EGFR kinase domain duplication / KDD")

    assert len(records) == 1
    assert records[0].gene == "EGFR"
    assert records[0].alteration_type == "kdd"


def test_alteration_parser_reads_loose_table(tmp_path: Path):
    from trufflepig.alterations import parse_alteration_file

    alterations = tmp_path / "variants.tsv"
    alterations.write_text(
        "\n".join(
            [
                "Gene\tAlteration\tVAF",
                "EGFR\tKinase domain duplication\t0.42",
            ]
        )
    )

    records = parse_alteration_file(alterations)

    assert len(records) == 1
    assert records[0].gene == "EGFR"
    assert records[0].alteration_type == "kdd"
    assert records[0].support["VAF"] == 0.42


def test_alteration_parser_rejects_missing_supplied_file(tmp_path: Path):
    from trufflepig.alterations import parse_alteration_inputs

    missing = tmp_path / "missing-variants.tsv"

    with pytest.raises(FileNotFoundError, match="Alteration evidence file not found"):
        parse_alteration_inputs(str(missing))


def test_fusion_parser_reads_hash_prefixed_star_fusion_header(tmp_path: Path):
    from trufflepig.fusions import parse_fusion_file

    fusion_file = tmp_path / "star-fusion.fusion_predictions.tsv"
    fusion_file.write_text(
        "\n".join(
            [
                "#FusionName\tJunctionReadCount\tSpanningFragCount",
                "BRD3--NUTM1\t12\t8",
            ]
        )
    )

    records = parse_fusion_file(fusion_file)

    assert len(records) == 1
    assert records[0].gene_a == "BRD3"
    assert records[0].gene_b == "NUTM1"
    assert records[0].support_total == 20


def test_empty_supplied_fusion_file_reports_no_usable_calls():
    from trufflepig.main import _fusion_evidence_markdown

    md = _fusion_evidence_markdown(
        {
            "fusion_inputs_supplied": True,
            "fusion_input_paths": ["empty.tsv"],
            "fusion_records": [],
        }
    )

    assert "no usable fusion calls were parsed" in md


def test_nutm1_fusion_rules_distinguish_partner_specificity():
    from trufflepig.fusions import FusionRecord
    from trufflepig.rare_inference import (
        infer_rare_cancer_report_scope_from_fusions,
        match_rare_cancer_fusion_rules,
    )

    brd4 = FusionRecord(gene_a="BRD4", gene_b="NUTM1", support_total=20)
    brd4_scope = infer_rare_cancer_report_scope_from_fusions(
        [brd4],
        {"candidate_trace": [{"code": "LUSC"}]},
    )
    assert brd4_scope["cancer_type"] == "NUTM"
    assert brd4_scope["expected_pair"] == "BRD4--NUTM1"

    cic = FusionRecord(gene_a="CIC", gene_b="NUTM1", support_total=20)
    cic_scope = infer_rare_cancer_report_scope_from_fusions([cic])
    cic_hits = match_rare_cancer_fusion_rules([cic])

    assert cic_scope is None
    assert cic_hits
    assert cic_hits[0]["promote_report_scope"] is False
    assert "sarcoma" in cic_hits[0]["label"].lower()


def test_explicit_5prime_3prime_fusion_orientation_is_respected():
    from trufflepig.fusions import FusionRecord
    from trufflepig.rare_inference import infer_rare_cancer_report_scope_from_fusions

    reversed_oriented = FusionRecord(
        gene_a="NUTM1",
        gene_b="BRD4",
        support_total=20,
        orientation="5prime_3prime",
    )

    assert infer_rare_cancer_report_scope_from_fusions([reversed_oriented]) is None


def test_fusion_expression_effects_use_tumor_tpm_when_available():
    from trufflepig.fusions import FusionRecord
    from trufflepig.fusion_effects import match_fusion_expression_effects

    record = FusionRecord(gene_a="BRD4", gene_b="NUTM1", support_total=20)
    findings = match_fusion_expression_effects(
        [record],
        {"NUTM1": 8.0, "MYC": 50.0, "PRAME": 0.0, "SOX2": 0.0},
        tumor_tpm_by_symbol={"NUTM1": 6.0, "MYC": 60.0, "PRAME": 9.0},
    )

    assert findings
    assert findings[0]["status"] == "active"
    assert findings[0]["expression_source"] in {"tumor_inferred", "mixed"}
    assert {"MYC", "PRAME"}.issubset(set(findings[0]["observed_genes"]))
    myc = next(row for row in findings[0]["gene_evidence"] if row["gene"] == "MYC")
    assert myc["source"] == "tumor_inferred"
    assert myc["bulk_tpm"] == 50.0
    assert myc["tumor_tpm"] == 60.0


def test_mutation_expression_effects_are_hypotheses_not_calls():
    from trufflepig.alteration_effects import infer_mutation_expression_hypotheses

    findings = infer_mutation_expression_hypotheses(
        {"ESR1": 0.0, "PGR": 0.0, "EGFR": 30.0, "KRT5": 20.0, "KRT14": 12.0},
        tumor_tpm_by_symbol={
            "ESR1": 0.0,
            "PGR": 0.0,
            "EGFR": 25.0,
            "KRT5": 18.0,
            "KRT14": 8.0,
        },
        cancer_code="BRCA",
    )

    labels = {finding["label"] for finding in findings}
    assert "Basal-like TNBC EGFR/KRT program" in labels
    assert all(finding["promote_report_scope"] is False for finding in findings)


def test_unmeasured_low_markers_do_not_support_mutation_expression_hypotheses():
    from trufflepig.alteration_effects import infer_mutation_expression_hypotheses

    findings = infer_mutation_expression_hypotheses(
        {"EGFR": 30.0, "KRT5": 20.0, "KRT14": 12.0},
        cancer_code="BRCA",
    )

    assert "Basal-like TNBC EGFR/KRT program" not in {
        finding["label"] for finding in findings
    }


def test_build_analyze_comparison_markdown_from_summary_files(tmp_path: Path):
    first = tmp_path / "case-alpha-baseline"
    second = tmp_path / "case-alpha-followup"
    first.mkdir()
    second.mkdir()
    first.joinpath("sample-summary.md").write_text(
        "\n".join(
            [
                "# Summary: case alpha baseline",
                "",
                "**Cancer call:** BLCA (Bladder Urothelial Carcinoma).",
                "**Cancer-type basis:** externally supplied (BLCA), not RNA-inferred; RNA evidence is used.",
                "**Purity:** 74% (model interval 62%-83%, moderate confidence).",
                "**Disease state:** interferon-high.",
                "**Sample:** polyA RNA-seq, FFPE preservation.",
                "",
                "## Top candidate therapies",
                "",
                "- **ERBB2** -> trastuzumab deruxtecan.",
                "",
                "## Caveats",
                "- Patient-facing LLM interpretation needs external clinical context.",
            ]
        )
    )
    second.joinpath("sample-summary.md").write_text(
        "\n".join(
            [
                "# Summary: case alpha followup",
                "",
                "**Cancer call:** BLCA (Bladder Urothelial Carcinoma).",
                "**Cancer-type basis:** externally supplied (BLCA), not RNA-inferred; RNA evidence is used.",
                "**Purity:** 85% (model interval 80%-90%, high confidence).",
                "**Disease state:** no strong RNA-defined therapy-exposure pattern.",
                "**Sample:** polyA RNA-seq, FFPE preservation.",
                "",
                "## Top candidate therapies",
                "",
                "- **NECTIN4** -> enfortumab vedotin.",
            ]
        )
    )

    markdown = build_analyze_comparison_markdown(
        [first, second],
        title="Longitudinal Analyze Comparison",
    )

    assert "# Longitudinal Analyze Comparison" in markdown
    assert "| case alpha baseline |" in markdown
    assert "| case alpha followup |" in markdown
    assert "## Longitudinal Deltas" in markdown
    assert "gained NECTIN4; lost ERBB2" in markdown
    assert "## Biology And Response State" in markdown
    assert "## Therapy Shortlists" in markdown
    assert "Treat RNA-inferred cancer labels as hypotheses" in markdown
    assert "Patient-facing LLM use" in markdown


def test_apply_sample_context_to_purity_widens_ci():
    analysis = {
        "purity": {
            "overall_estimate": 0.5,
            "overall_lower": 0.4,
            "overall_upper": 0.6,
        }
    }
    context = SampleContext(
        preservation="ffpe",
        degradation_severity="severe",
        degradation_index=0.2,
    )

    changed = apply_sample_context_to_purity(analysis, context)

    assert changed is True
    assert analysis["purity"]["overall_lower"] == pytest.approx(0.34)
    assert analysis["purity"]["overall_upper"] == pytest.approx(0.66)
    assert analysis["purity"]["degradation_caveat"]["severity"] == "severe"


def test_should_adopt_decomposition_purity_contract():
    ok = SimpleNamespace(
        cancer_type="COAD",
        warnings=[],
        purity_result={"overall_estimate": 0.4},
    )
    mismatch = SimpleNamespace(
        cancer_type="BRCA",
        warnings=[],
        purity_result={"overall_estimate": 0.9},
    )
    no_tme = SimpleNamespace(
        cancer_type="COAD",
        warnings=["No non-tumor components in template"],
        purity_result={"overall_estimate": 1.0},
    )
    missing = SimpleNamespace(cancer_type="COAD", warnings=[], purity_result=None)

    assert should_adopt_decomposition_purity("COAD", ok)
    assert not should_adopt_decomposition_purity("COAD", mismatch)
    assert not should_adopt_decomposition_purity("COAD", no_tme)
    assert not should_adopt_decomposition_purity("COAD", missing)


def test_top_level_import_does_not_import_matplotlib_pyplot():
    code = "import sys; import pirlygenes; print('matplotlib.pyplot' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
