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


def test_observed_reference_parent_without_broad_tpm_can_anchor_child(monkeypatch):
    import trufflepig.main as main

    monkeypatch.setattr(
        main,
        "_pan_cancer_expression_cohort_codes",
        lambda: frozenset({"PCPG", "SCLC", "SARC"}),
    )
    monkeypatch.setattr(
        main,
        "_has_operational_analysis_reference",
        lambda code: str(code) == "NBL",
    )

    composition_scope, report_scope = main._analysis_input_cancer_type("NBL_MYCNamp")

    assert composition_scope == "NBL"
    assert report_scope == "NBL_MYCNamp"


def test_observed_reference_parent_without_operational_purity_stays_report_scope(
    monkeypatch,
):
    import trufflepig.main as main

    monkeypatch.setattr(
        main,
        "_pan_cancer_expression_cohort_codes",
        lambda: frozenset({"PCPG", "SCLC", "SARC"}),
    )
    monkeypatch.setattr(main, "_has_operational_analysis_reference", lambda code: False)

    composition_scope, report_scope = main._analysis_input_cancer_type("NBL_MYCNamp")

    assert composition_scope is None
    assert report_scope == "NBL_MYCNamp"


def test_sclc_subtype_uses_supported_parent_reference(monkeypatch):
    import trufflepig.main as main

    monkeypatch.setattr(
        main,
        "_pan_cancer_expression_cohort_codes",
        lambda: frozenset({"PCPG", "SARC"}),
    )
    monkeypatch.setattr(
        main,
        "_has_operational_analysis_reference",
        lambda code: str(code) == "SCLC",
    )

    composition_scope, report_scope = main._analysis_input_cancer_type("SCLC_YAP1")

    assert composition_scope == "SCLC"
    assert report_scope == "SCLC_YAP1"


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


def test_concrete_child_with_abstract_umbrella_parent_is_not_promoted():
    """A concrete expression cohort (COAD) whose registry parent is an *abstract
    grouping* with no cohort of its own (CRC, which exists only to bucket
    COAD/READ) must stay its own composition scope — never be promoted to the
    umbrella. Promoting COAD->CRC stripped it of its expression/purity reference
    and crashed the purity resolver on a missing CRC_TPM column."""
    from trufflepig.main import _analysis_input_cancer_type

    for concrete in ("COAD", "READ"):
        composition_scope, report_scope = _analysis_input_cancer_type(concrete)
        assert composition_scope == concrete, concrete
        assert report_scope is None, concrete


def test_nutm1_expression_can_infer_registry_only_report_scope(monkeypatch):
    import pandas as pd

    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.main import _infer_registry_report_scope_from_rna

    # This test owns rare-marker/report-scope integration. Whole-profile,
    # centroid, lineage, and exact-reference behavior is covered by their own
    # real-data tests; running those independent axes on a one-gene fixture
    # rebuilt all references and added ~3 minutes without changing this result.
    monkeypatch.setattr(
        evidence, "_add_learned_expression_classifier_features", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evidence, "_add_learned_hierarchy_candidate_features", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evidence, "_add_local_expression_reference_features", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evidence, "_add_lineage_panel_features", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evidence, "_centroid_and_confidence", lambda *a, **k: (None, False)
    )
    monkeypatch.setattr(evidence, "_FINE_REFERENCE_SPECS", ())

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


def test_context_roles_make_hint_outputs_consistent_for_nutm():
    from trufflepig.analyze import cancer_type_context_from_analysis
    from trufflepig.main import _apply_cancer_type_context_roles

    analysis = {
        "cancer_type": "NUTM",
        "report_scope_cancer_type": "NUTM",
        "report_scope_parent_cancer_type": "LUSC",
        "reference_cancer_type": "LUSC",
    }
    context = cancer_type_context_from_analysis(
        analysis,
        supplied_cancer_type="NUTM",
    )

    _apply_cancer_type_context_roles(analysis, context)

    assert analysis["reference_cancer_type"] == "LUSC"
    assert analysis["fallback_expression_reference_cancer_type"] == "LUSC"
    assert analysis["expression_reference_cancer_type"] == "NUTM"
    assert analysis["expression_reference_role"] == "report_label_exact"
    assert "report_scope_parent_cancer_type" not in analysis


def test_context_roles_never_propagate_a_sibling_subtype():
    from trufflepig.analyze import cancer_type_context_from_analysis
    from trufflepig.main import _apply_cancer_type_context_roles

    analysis = {
        "cancer_type": "BRCA_Basal",
        "report_scope_cancer_type": "BRCA_Basal",
        "report_scope_parent_cancer_type": "BRCA_HER2",
        "reference_cancer_type": "BRCA_HER2",
        "reference_cancer_name": "HER2-enriched",
        "expression_reference_cancer_type": "BRCA_HER2",
    }
    context = cancer_type_context_from_analysis(analysis)

    _apply_cancer_type_context_roles(analysis, context)

    assert analysis["reference_cancer_type"] == "BRCA"
    assert analysis["reference_cancer_name"] == "Breast Invasive Carcinoma"
    assert analysis["report_scope_parent_cancer_type"] == "BRCA"
    assert analysis["expression_reference_cancer_type"] == "BRCA_Basal"
    assert analysis["expression_reference_role"] == "report_label_exact"
    assert analysis["requested_reference_cancer_type"] == "BRCA_HER2"
    assert analysis["requested_expression_reference_cancer_type"] == "BRCA_HER2"
    assert analysis["excluded_sibling_cancer_type_contexts"] == ["BRCA_HER2"]
    assert analysis["cancer_type_tree_roles"] == {
        "report": {
            "code": "BRCA_Basal",
            "role": "diagnosis",
            "relationship": "same",
        },
        "reference": {
            "code": "BRCA",
            "role": "analysis_context",
            "relationship": "ancestor",
        },
        "expression": {
            "code": "BRCA_Basal",
            "role": "expression_reference",
            "relationship": "same",
        },
        "excluded_siblings": ["BRCA_HER2"],
    }
    final_context = cancer_type_context_from_analysis(analysis)
    assert final_context.excluded_sibling_codes == ("BRCA_HER2",)
    assert final_context.code_for("reference") == "BRCA"
    assert final_context.code_for("expression") == "BRCA_Basal"


def test_context_resynchronization_drops_the_new_report_from_old_exclusions():
    from trufflepig.main import _synchronize_cancer_type_context

    analysis = {
        "cancer_type": "READ",
        "report_scope_cancer_type": "READ",
        "reference_cancer_type": "READ",
        "expression_reference_cancer_type": "READ",
        # READ was excluded while COAD was active; after a sibling change it is
        # the diagnosis and can no longer remain an exclusion.
        "excluded_sibling_cancer_type_contexts": ["READ"],
    }

    context = _synchronize_cancer_type_context(
        analysis,
        supplied_cancer_type=None,
    )

    assert context.code_for("report") == "READ"
    assert context.excluded_sibling_codes == ()
    assert "excluded_sibling_cancer_type_contexts" not in analysis
    assert analysis["cancer_type_tree_roles"]["excluded_siblings"] == []
    assert analysis["cancer_type_context"]["excluded_sibling_codes"] == ()


def test_rare_rna_surrogate_rules_are_data_backed_and_context_gated():
    from pirlygenes.gene_sets_cancer import rare_cancer_rna_surrogate_rules_df

    rules = rare_cancer_rna_surrogate_rules_df()
    assert {"NUTM", "SARC_CHOR", "ACINIC"}.issubset(set(rules["cancer_code"]))


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


def test_rna_only_fusion_hypotheses_require_compatible_context():
    from trufflepig.fusion_effects import infer_fusion_expression_hypotheses

    sample = {"TFE3": 4.0, "ANGPTL2": 30.0, "VEGFA": 25.0}

    acc_findings = infer_fusion_expression_hypotheses(sample, cancer_code="ACC")
    assert all(
        row["rule_id"] != "aspscr1_tfe3_asps_program" for row in acc_findings
    )

    sarc_findings = infer_fusion_expression_hypotheses(sample, cancer_code="SARC")
    asps = next(
        row for row in sarc_findings if row["rule_id"] == "aspscr1_tfe3_asps_program"
    )
    assert asps["status"] == "active"
    assert {"ANGPTL2", "VEGFA"}.issubset(set(asps["observed_genes"]))


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

    widened = dict(analysis["purity"])
    assert apply_sample_context_to_purity(analysis, context) is False
    assert analysis["purity"] == widened

    # A downstream purity replacement or fusion can change the interval while
    # retaining metadata. The same context must widen that new final interval
    # exactly once.
    analysis["purity"]["overall_lower"] = 0.45
    analysis["purity"]["overall_upper"] = 0.55
    assert apply_sample_context_to_purity(analysis, context) is True
    assert analysis["purity"]["overall_lower"] == pytest.approx(0.42)
    assert analysis["purity"]["overall_upper"] == pytest.approx(0.58)
    re_widened = dict(analysis["purity"])
    assert apply_sample_context_to_purity(analysis, context) is False
    assert analysis["purity"] == re_widened


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
    crc_child = SimpleNamespace(
        cancer_type="READ",
        warnings=[],
        purity_result={"overall_estimate": 0.4},
    )
    child_to_parent = SimpleNamespace(
        cancer_type="CRC",
        warnings=[],
        purity_result={"overall_estimate": 0.4},
    )

    assert should_adopt_decomposition_purity("COAD", ok)
    assert should_adopt_decomposition_purity("CRC", crc_child)
    assert not should_adopt_decomposition_purity("COAD", child_to_parent)
    assert not should_adopt_decomposition_purity("COAD", mismatch)
    assert not should_adopt_decomposition_purity("COAD", no_tme)
    assert not should_adopt_decomposition_purity("COAD", missing)


def _decomp_hyp(cancer_type, purity, recon, template, warnings=()):
    return SimpleNamespace(
        cancer_type=cancer_type,
        purity=purity,
        reconstruction_error=recon,
        template=template,
        warnings=list(warnings),
    )


def test_decomposition_purity_stability_flags_fragile_and_stable():
    from trufflepig.analyze import decomposition_purity_stability

    # READ-like: top-by-score hypothesis is a low-purity over-subtracted fit while the same-cancer
    # templates span 5%–78% — the adopted point purity is one pick in a wide range.
    fragile = [
        _decomp_hyp("READ", 0.10, 16.75, "solid_primary",
                    ["Many genes are overexplained by the TME background"]),
        _decomp_hyp("READ", 0.78, 2.68, "met_liver"),
        _decomp_hyp("READ", 0.40, 3.0, "met_bone"),
        _decomp_hyp("READ", 0.05, 4.0, "met_skin"),
    ]
    out = decomposition_purity_stability(fragile)
    assert out["fragile"] is True
    assert out["tme_overexplained"] is True
    assert out["hypothesis_purity_spread"] == pytest.approx(0.73)
    assert len(out["top_hypotheses"]) == 4

    # Osteosarcoma-like: every same-cancer template agrees at ~99% purity → stable, no over-subtraction.
    stable = [
        _decomp_hyp("SARC", 0.994, 5.2, "solid_primary"),
        _decomp_hyp("SARC", 0.994, 2.6, "met_bone"),
        _decomp_hyp("SARC", 0.994, 4.7, "met_liver"),
    ]
    out2 = decomposition_purity_stability(stable)
    assert out2["fragile"] is False
    assert out2["tme_overexplained"] is False
    assert out2["hypothesis_purity_spread"] == pytest.approx(0.0)

    # Defensive: empty input → empty dict, no raise.
    assert decomposition_purity_stability([]) == {}


def test_reconcile_decomposition_purity_rejects_fragile_and_fully_inconsistent():
    from trufflepig.analyze import reconcile_decomposition_purity

    classifier = {
        "overall_estimate": 0.55,
        "overall_lower": 0.45,
        "overall_upper": 0.65,  # trustworthy: non-saturated with a real interval
        "components": {
            "signature": {"purity": 0.58},
            "lineage": {"purity": 0.62},
            "estimate_purity": 0.55,
        },
    }
    decomp = {"overall_estimate": 0.10, "overall_lower": 0.06, "overall_upper": 0.16}
    stability = {"fragile": True, "tme_overexplained": True,
                 "hypothesis_purity_lo": 0.03, "hypothesis_purity_hi": 0.78,
                 "hypothesis_purity_spread": 0.75}
    action, out = reconcile_decomposition_purity(classifier, decomp, stability)
    # 10% disagrees with every independent signal (58/62/55%) by > margin → reject.
    assert action == "reject"
    assert out["overall_estimate"] == 0.55  # classifier purity kept
    assert "decomposition_purity_rejected" in out


def test_reconcile_does_not_fall_back_to_saturated_signature():
    """The TME-overexplained low-purity case: the decomposition (9%) disagrees with a SATURATED
    signature (~100%) that is exactly what decomposition corrects. Falling back would revert to a
    wrong ~100%; the gate must widen instead of reject."""
    from trufflepig.analyze import reconcile_decomposition_purity

    saturated_classifier = {
        "overall_estimate": 1.0,
        "overall_lower": 1.0,
        "overall_upper": 1.0,  # saturated + zero-width → NOT a trustworthy fall-back
        "components": {
            "signature": {"purity": 1.0},
            "lineage": {"purity": 0.98},
            "estimate_purity": 0.95,
        },
    }
    decomp = {"overall_estimate": 0.09, "overall_lower": 0.03, "overall_upper": 0.20}
    stability = {"fragile": True, "tme_overexplained": True,
                 "hypothesis_purity_lo": 0.06, "hypothesis_purity_hi": 0.19,
                 "hypothesis_purity_spread": 0.13}
    action, out = reconcile_decomposition_purity(saturated_classifier, decomp, stability)
    assert action == "widen"  # NOT reject — keeps the honest low purity, widened
    assert out["overall_estimate"] == 0.09
    assert "purity_interval_widened_for_fragility" in out


def test_reconcile_decomposition_purity_widens_when_consistent_with_one_signal():
    from trufflepig.analyze import reconcile_decomposition_purity

    classifier = {
        "overall_estimate": 0.15,
        "components": {
            "signature": {"purity": 0.12},  # corroborates the ~10% decomposition
            "lineage": {"purity": 0.62},
            "estimate_purity": 0.55,
        },
    }
    decomp = {"overall_estimate": 0.10, "overall_lower": 0.06, "overall_upper": 0.16}
    stability = {"fragile": True, "tme_overexplained": True,
                 "hypothesis_purity_lo": 0.03, "hypothesis_purity_hi": 0.78,
                 "hypothesis_purity_spread": 0.75}
    action, out = reconcile_decomposition_purity(classifier, decomp, stability)
    # signature agrees within margin → keep the 10% point but widen to span the hypotheses.
    assert action == "widen"
    assert out["overall_estimate"] == 0.10  # point kept
    assert out["overall_lower"] == pytest.approx(0.03)
    assert out["overall_upper"] == pytest.approx(0.78)
    assert "purity_interval_widened_for_fragility" in out


def test_reconcile_decomposition_purity_adopts_stable_unchanged():
    from trufflepig.analyze import reconcile_decomposition_purity

    decomp = {"overall_estimate": 0.994, "overall_lower": 0.99, "overall_upper": 1.0}
    action, out = reconcile_decomposition_purity({}, decomp, {"fragile": False})
    assert action == "adopt"
    assert out is decomp  # returned unchanged


def test_wilms_deconvolved_reference_uses_expression_source_code_alias():
    import pandas as pd

    from trufflepig.tumor_purity import (
        _resolve_purity_reference,
        _subtype_tumor_tpm_lookup,
    )

    wilms_ref = _subtype_tumor_tpm_lookup("WILMS")
    target_wt_ref = _subtype_tumor_tpm_lookup("TARGET_WT")

    assert len(wilms_ref) > 1000
    assert wilms_ref == target_wt_ref
    resolved = _resolve_purity_reference("WILMS", pd.DataFrame())
    assert resolved["reference_cancer_code"] == "WILMS"
    assert resolved["reference_expression_source"] == "subtype_deconvolved"


def test_top_level_import_does_not_import_matplotlib_pyplot():
    code = "import sys; import pirlygenes; print('matplotlib.pyplot' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
