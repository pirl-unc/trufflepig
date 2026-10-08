"""One decision drives recommendations, blockers and deduplicated requests."""

import pandas as pd
import pytest

from trufflepig.brief import recommend_therapies
from trufflepig.report_content import assess_therapy
from trufflepig.reporting import cancer_therapy_panel_for_analysis, target_observation_state
from trufflepig.therapy_eligibility import collect_evidence_requests, evaluate_therapy_eligibility
from trufflepig.variants import normalize_protein_substitution


def mutation_context(gene, variant):
    return {
        "cancer_type": "COAD",
        "variant_inputs_supplied": True,
        "variant_records": [
            {
                "gene": gene,
                "variant": variant,
                "variant_type": "mutation",
                "status": "detected",
                "source": "clinical report",
            },
        ],
    }


@pytest.mark.parametrize(
    "gene, wrong, right, agent",
    [
        ("KRAS", "p.G12D", "p.Gly12Cys", "sotorasib"),
        ("BRAF", "p.D594G", "p.Val600Glu", "encorafenib"),
    ],
)
def test_real_colorectal_regimens_require_the_exact_mutation(gene, wrong, right, agent):
    for variant, expected in [(wrong, False), (right, True)]:
        analysis = mutation_context(gene, variant)
        _, subtype, panel = cancer_therapy_panel_for_analysis("COAD", analysis)
        row = next(row for row in panel.to_dict("records") if agent in row["agent"])
        decision = evaluate_therapy_eligibility(row, analysis, panel_subtype=subtype)
        assert decision.supplied_variant_supported is expected
        selected = recommend_therapies(pd.DataFrame([row]), pd.DataFrame(), analysis=analysis)
        assert bool(selected) is expected
        assessment = assess_therapy(row, analysis=analysis)
        assert assessment["eligibility"]["permits_review"] is expected


def test_unrelated_variant_file_never_satisfies_a_target_requirement():
    analysis = mutation_context("TP53", "p.R175H")
    row = {
        "symbol": "KRAS",
        "agent": "sotorasib + panitumumab",
        "cancer_code": "COAD",
        "phase": "approved",
        "indication": "KRAS G12C colorectal cancer",
        "requires_verified_alteration": True,
    }
    decision = evaluate_therapy_eligibility(row, analysis)
    assert not decision.permits_review
    assert not decision.direct_evidence_supported
    assert "G12C" in decision.requirements[-1].description


@pytest.mark.parametrize(
    "hla_types, status", [(["A*24:02"], "blocked"), (["A*02:05"], "blocked"), ([], "missing")]
)
def test_known_hla_exclusions_are_blockers_and_unknown_typing_is_a_request(hla_types, status):
    analysis = {"cancer_type": "SARC_SYN", "analysis_constraints": {"hla_types": hla_types}}
    _, subtype, panel = cancer_therapy_panel_for_analysis("SARC_SYN", analysis)
    row = next(row for row in panel.to_dict("records") if row["agent"] == "afami-cel (Tecelra)")
    assessment = assess_therapy(row, analysis=analysis, panel_subtype=subtype)
    requirement = next(r for r in assessment["eligibility"]["requirements"] if r["kind"] == "hla")
    assert requirement["status"] == status
    requests = collect_evidence_requests([assessment])
    assert any(r["kind"] == "hla" for r in requests) is (status == "missing")


def test_contraindicated_component_suppresses_other_testing_requests():
    analysis = {
        "cancer_type": "COAD",
        "treatment_history": [{"therapy": "panitumumab", "status": "contraindicated"}],
    }
    row = {
        "symbol": "KRAS",
        "agent": "sotorasib + panitumumab",
        "cancer_code": "COAD",
        "phase": "approved",
        "indication": "KRAS G12C colorectal cancer",
        "requires_verified_alteration": True,
    }
    assessment = assess_therapy(row, analysis=analysis)
    assert not assessment["eligibility"]["permits_review"]
    assert any(r["status"] == "blocked" for r in assessment["eligibility"]["requirements"])
    assert collect_evidence_requests([assessment]) == []


def test_shared_hla_request_retains_every_affected_agent_and_requirement():
    analysis = {"cancer_type": "SARC_SYN"}
    rows = [
        dict(
            symbol="MAGEA4",
            agent=agent,
            phase="phase_2",
            hla_restriction="A*02:01",
            hla_source="protocol",
        )
        for agent in ("afami-cel", "Example TCR therapy")
    ]
    assessments = [assess_therapy(row, analysis=analysis) for row in rows]
    requests = [r for r in collect_evidence_requests(assessments) if r["kind"] == "hla"]
    assert len(requests) == 1
    assert set(requests[0]["affects"]) == {"afami-cel", "Example TCR therapy"}
    assert requests[0]["accepted_inputs"]


def test_missing_observation_keeps_antigen_aliases_and_empty_coverage_distinct():
    frame = pd.DataFrame()
    assert target_observation_state("MAGE-A4", frame) == "unknown"
    frame.attrs["sample_input_symbols"] = set()
    assert target_observation_state("MAGEA4", frame) == "not_in_input"
    frame.attrs["sample_input_symbols"] = {"MAGE-A4"}
    assert target_observation_state("MAGEA4", frame) == "below_detection"
    assessment = assess_therapy({"symbol": "MAGEA4", "agent": "Example"}, ranges_df=frame)
    assert assessment["observation"] == {"state": "below_detection", "observed_tpm": None}
    text = " ".join(assessment["rationale"])
    assert "below detection" in text
    assert "not measured" not in text


@pytest.mark.parametrize(
    "value, expected",
    [
        ("p.Gly12Cys", "G12C"),
        ("KRAS p.G12C", "G12C"),
        ("p.(G12C)", "G12C"),
        ("p.(G12C", ""),
        ("G12C or G12D", ""),
        ("c.34G>T", ""),
        ("BRAF p.G12C", ""),
    ],
)
def test_protein_identity_does_not_guess_ambiguous_or_nucleotide_only_input(value, expected):
    assert normalize_protein_substitution(value, gene="KRAS") == expected


def test_prior_benefit_supports_review_without_claiming_a_confirmed_assay():
    row = {
        "symbol": "FAP",
        "agent": "177Lu-FAP-2286",
        "phase": "phase_2",
        "indication": "FAP-directed radioligand",
        "requires_verified_alteration": True,
        "eligibility_note": "requires target imaging",
    }
    analysis = {"treatment_history": [{"therapy": "177Lu-FAP-2286", "status": "major_benefit"}]}
    assessment = assess_therapy(row, analysis=analysis)
    assert assessment["eligibility"]["permits_review"]
    assert any(r["status"] == "missing" for r in assessment["eligibility"]["requirements"])
    assert collect_evidence_requests([assessment])
    assert recommend_therapies(pd.DataFrame([row]), pd.DataFrame(), analysis=analysis)


def test_prior_benefit_cannot_override_missing_hla_typing():
    row = {"symbol": "MAGEA4", "agent": "afami-cel", "phase": "approved"}
    analysis = {"treatment_history": [{"therapy": "afami-cel", "status": "major_benefit"}]}
    assert not evaluate_therapy_eligibility(row, analysis).permits_review


@pytest.mark.parametrize("status", ["VUS", "negative", "failed", "not reportable"])
def test_exact_allele_still_requires_a_usable_positive_call(status):
    analysis = mutation_context("KRAS", "p.G12C")
    analysis["variant_records"][0]["result_status"] = status
    _, subtype, panel = cancer_therapy_panel_for_analysis("COAD", analysis)
    row = next(row for row in panel.to_dict("records") if "sotorasib" in row["agent"])
    decision = evaluate_therapy_eligibility(row, analysis, panel_subtype=subtype)
    assert not decision.permits_review
    assert not recommend_therapies(pd.DataFrame([row]), pd.DataFrame(), analysis=analysis)


def test_unmapped_protein_accession_does_not_establish_a_gene_specific_allele():
    assert normalize_protein_substitution("NP_123456.1:p.G12C", gene="KRAS") == ""


def test_wrong_allele_reason_survives_markdown_json_and_pdf(tmp_path):
    from trufflepig.report_content import build_report_content, render_report_summary
    from trufflepig.report_document import write_report_document, load_report_document
    from trufflepig.report_pdf import build_interpretive_report_pdf
    from trufflepig.report_view import build_report_view
    from pypdf import PdfReader

    analysis = {**mutation_context("KRAS", "p.G12D"), "purity": {}, "sample_mode": "solid"}
    view = build_report_view(analysis, sample_id="synthetic-wrong-allele")
    content = build_report_content(analysis, pd.DataFrame(), "COAD", "", report_view=view)
    text = render_report_summary(content)
    assert "p.G12D" in text and "required KRAS G12C" in text
    write_report_document(tmp_path, "synthetic-wrong-allele", report_view=view, content=content)
    doc = load_report_document(tmp_path, "synthetic-wrong-allele")
    assessment = next(row for row in doc["therapy_assessments"] if "sotorasib" in row["agent"])
    requirement = next(
        r for r in assessment["eligibility"]["requirements"] if r["kind"] == "mutation"
    )
    assert requirement["status"] == "unresolved"
    assert requirement["evidence"]["supplied_variants"][0]["variant"] == "p.G12D"
    (tmp_path / "synthetic-wrong-allele-summary.md").write_text(text)
    pdf = PdfReader(build_interpretive_report_pdf(tmp_path))
    pdf_text = " ".join(" ".join(page.extract_text() for page in pdf.pages).split())
    assert "p.G12D" in pdf_text and "required KRAS G12C" in pdf_text


def test_prior_benefit_cannot_override_an_incompatible_supplied_allele():
    analysis = mutation_context("KRAS", "p.G12D")
    analysis["treatment_history"] = [{"therapy": "sotorasib", "status": "major_benefit"}]
    row = {
        "symbol": "KRAS",
        "agent": "sotorasib",
        "cancer_code": "COAD",
        "indication": "KRAS G12C colorectal cancer",
        "phase": "approved",
    }
    eligibility = evaluate_therapy_eligibility(row, analysis)
    assert eligibility.history_supported
    assert not eligibility.permits_review


def test_spindle_diagnostic_workup_occurs_only_in_the_information_section():
    from trufflepig.report_content import build_report_content, render_report_summary
    from trufflepig.report_view import build_report_view

    analysis = {"cancer_type": "SARC_IFS", "sample_mode": "mesenchymal", "purity": {}}
    content = build_report_content(
        analysis, pd.DataFrame(), "SARC_IFS", "", report_view=build_report_view(analysis)
    )
    summary = render_report_summary(content)
    workup = "Review renal versus soft-tissue site and pathology"
    assert summary.count(workup) == 1
    assert summary.index(workup) > summary.index("## Information needed")
    assert any(r["key"] == "spindle_diagnostic_context" for r in content.evidence_requests)


@pytest.mark.parametrize(
    "observed, expected",
    [
        (0.0, "below_detection"),
        (12.0, "measured"),
        (float("nan"), "invalid"),
        (float("inf"), "invalid"),
        (-1, "invalid"),
        ("bad", "invalid"),
        (None, "unknown"),
    ],
)
def test_shared_rna_observation_preserves_zero_missing_and_invalid(observed, expected):
    from trufflepig.reporting import target_rna_observation, expression_independent_rna_context

    expression = {"observed_tpm": observed}
    result = target_rna_observation(expression)
    assert result["state"] == expected
    assessment = assess_therapy({"symbol": "FAP", "agent": "Example"}, expression)
    assert assessment["observation"] == result
    narrative = expression_independent_rna_context(expression)
    if expected == "invalid":
        assert "invalid or unresolved" in narrative
        ranges = pd.DataFrame(
            [
                {
                    "symbol": "FAP",
                    "observed_tpm": observed,
                    "attr_tumor_tpm": 20,
                    "attr_tumor_fraction": 0.9,
                }
            ]
        )
        assert (
            recommend_therapies(
                pd.DataFrame([{"symbol": "FAP", "agent": "Example", "phase": "phase_2"}]), ranges
            )
            == []
        )
    if expected == "below_detection":
        assert "below detection" in narrative and "not measured" not in narrative


@pytest.mark.parametrize(
    "pairs",
    [
        [
            ("target:A", "Shared assay"),
            ("target:B", "Shared assay"),
            ("target:B", "Additional assay detail"),
        ],
        [("target:A", "First assay"), ("target:B", "Second assay"), ("target:A", "Second assay")],
    ],
)
def test_request_deduplication_merges_transitive_keys_and_questions(pairs):
    from trufflepig.therapy_eligibility import EvidenceRequirement

    assessments = [
        {
            "agent": f"Therapy {index}",
            "eligibility": {
                "requirements": [
                    EvidenceRequirement(
                        key,
                        "clinical_target_assay",
                        "missing",
                        "Assay unavailable",
                        question,
                        ("clinical report",),
                    ).public_dict()
                ]
            },
        }
        for index, (key, question) in enumerate(pairs)
    ]
    requests = collect_evidence_requests(assessments)
    assert len(requests) == 1
    assert set(requests[0]["keys"]) == {"target:A", "target:B"}
    assert len(requests[0]["affects"]) == 3
    assert set(requests[0]["requirements"]) == {question for _, question in pairs}


@pytest.mark.parametrize(
    "code, agent, phrase",
    [
        ("BLCA", "avelumab", "without progression after first-line platinum"),
        ("PRAD", "enzalutamide", "continued medical/surgical castration"),
        ("SARC_OS", "regorafenib", "recurrent/progressive disease after chemotherapy"),
    ],
)
def test_selected_therapy_clinical_criteria_survive_all_report_formats(tmp_path, code, agent, phrase):
    from trufflepig.report_content import build_report_content, render_report_summary
    from trufflepig.report_document import write_report_document, load_report_document
    from trufflepig.report_pdf import build_interpretive_report_pdf
    from trufflepig.report_view import build_report_view
    from pypdf import PdfReader

    analysis = {"cancer_type": code, "purity": {}, "sample_mode": "solid"}
    view = build_report_view(analysis, sample_id="synthetic-clinical-setting")
    content = build_report_content(analysis, pd.DataFrame(), code, "", report_view=view)
    assert any(a["agent"] == agent and a["selected"] for a in content.therapy_assessments)
    summary = render_report_summary(content)
    assert summary.count(phrase) == 1
    assert summary.index(phrase) > summary.index("## Information needed")
    request = next(r for r in content.evidence_requests if r["key"] == "clinical_setting")
    detail = next(d for d in request["details"] if phrase in d["question"])
    assert agent in detail["affects"]
    if code == "SARC_OS":
        assert set(detail["affects"]) == {"regorafenib", "cabozantinib"}
    if code == "BLCA":
        assert "maintenance after first line platinum" in summary
        assert detail["affects"] == ["avelumab"]
    if code == "PRAD":
        # A shared BRCA2 work item must not apply niraparib's mCSPC setting
        # to the other PARP-inhibitor combinations in the group.
        paragraph = next(
            block["text"] for section in content.sections if section["id"] == "information"
            for block in section["blocks"] if "label-specific mCSPC" in block["text"]
        )
        assert "niraparib" in paragraph
        assert "olaparib" not in paragraph
    write_report_document(tmp_path, "synthetic-clinical-setting", report_view=view, content=content)
    doc = load_report_document(tmp_path, "synthetic-clinical-setting")
    assert doc["evidence_requests"] == content.evidence_requests
    (tmp_path / "synthetic-clinical-setting-summary.md").write_text(summary)
    pdf = PdfReader(build_interpretive_report_pdf(tmp_path))
    pdf_text = " ".join(" ".join(page.extract_text() for page in pdf.pages).split())
    assert pdf_text.count(phrase) == 1


def test_parsed_variant_file_satisfies_exact_allele_gates(tmp_path):
    from trufflepig.variants import parse_variant_file

    path = tmp_path / "variants.csv"
    path.write_text("gene,variant\nKRAS,p.G12C\nBRAF,V600E\n")
    analysis = {
        "cancer_type": "COAD",
        "variant_inputs_supplied": True,
        "variant_records": [record.public_dict() for record in parse_variant_file(path)],
    }
    _, subtype, panel = cancer_therapy_panel_for_analysis("COAD", analysis)
    for agent, label in [("sotorasib", "KRAS G12C"), ("encorafenib", "BRAF V600E")]:
        row = next(row for row in panel.to_dict("records") if agent in row["agent"])
        decision = evaluate_therapy_eligibility(row, analysis, panel_subtype=subtype)
        requirement = next(r for r in decision.requirements if r.kind == "mutation")
        assert decision.permits_review and requirement.status == "satisfied"
        assert requirement.label == label
        assert requirement.source.startswith("https://")


def test_fusion_satisfies_an_altered_indication_but_amplification_not_a_named_mutation():
    from trufflepig.reporting import supplied_variant_supports_target_row

    fusion = {
        "cancer_type": "BLCA",
        "fusion_records": [{"gene_a": "FGFR3", "gene_b": "TACC3", "name": "FGFR3--TACC3"}],
    }
    _, _, panel = cancer_therapy_panel_for_analysis("BLCA", fusion)
    erdafitinib = next(row for row in panel.to_dict("records") if "erdafitinib" in row["agent"])
    assert supplied_variant_supports_target_row(erdafitinib, fusion)

    _, _, panel = cancer_therapy_panel_for_analysis("LUAD", {"cancer_type": "LUAD"})
    capmatinib = next(row for row in panel.to_dict("records") if "capmatinib" in row["agent"])
    for variant, variant_type, expected in [
        ("MET amplification", "amplification", False),
        ("MET loss", "loss", False),
        ("MET L1195V", "mutation", False),
        ("MET exon 13 skipping", "mutation", False),
        ("MET exon 14 deletion", "loss", False),
        ("MET exon 14 skipping", "mutation", True),
    ]:
        analysis = {
            "cancer_type": "LUAD",
            "variant_records": [{"gene": "MET", "variant": variant, "variant_type": variant_type}],
        }
        assert bool(supplied_variant_supports_target_row(capmatinib, analysis)) is expected


def test_named_mutation_event_is_required_for_eligibility_and_shortlisting():
    _, subtype, panel = cancer_therapy_panel_for_analysis("LUAD", {"cancer_type": "LUAD"})
    row = next(row for row in panel.to_dict("records") if "capmatinib" in row["agent"])
    for variant, permits_review in [
        ("MET L1195V", False),
        ("MET exon 14 skipping", True),
    ]:
        analysis = {
            "cancer_type": "LUAD",
            "variant_records": [
                {"gene": "MET", "variant": variant, "variant_type": "mutation"}
            ],
        }
        decision = evaluate_therapy_eligibility(row, analysis, panel_subtype=subtype)
        assert decision.permits_review is permits_review
        recommendations = recommend_therapies(
            pd.DataFrame([row]), pd.DataFrame(), analysis=analysis
        )
        assert bool(recommendations) is permits_review


def test_mutation_indication_without_evidence_is_a_request_not_a_candidate():
    analysis = {"cancer_type": "OV"}
    _, subtype, panel = cancer_therapy_panel_for_analysis("OV", analysis)
    row = next(row for row in panel.to_dict("records") if row["agent"] == "olaparib")
    decision = evaluate_therapy_eligibility(row, analysis, panel_subtype=subtype)
    assert not decision.permits_review
    assert [r.status for r in decision.requirements] == ["missing"]
    assert not recommend_therapies(pd.DataFrame([row]), pd.DataFrame(), analysis=analysis)
    requests = collect_evidence_requests([assess_therapy(row, analysis=analysis)])
    assert requests[0]["label"] == f"{row['symbol']} mutation / fusion / amplification"


def test_unestablished_subtype_is_requested_rather_than_listed_as_a_blocker():
    analysis = {"cancer_type": "SARC"}
    row = {
        "symbol": "KIT",
        "agent": "imatinib",
        "cancer_code": "SARC",
        "subtype": "SARC_GIST",
        "indication": "gastrointestinal stromal tumor",
        "phase": "approved",
    }
    decision = evaluate_therapy_eligibility(row, analysis)
    scope = next(r for r in decision.requirements if r.kind == "scope")
    assert scope.status == "missing" and not decision.permits_review
    assert not decision.has_known_blocker
    requests = collect_evidence_requests([assess_therapy(row, analysis=analysis)])
    assert any(request["label"] == "Disease subtype" for request in requests)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("G12C;", ("G12C",)),
        ("G12C ;; G12D", ("G12C", "G12D")),
        ("KRAS G12C", ("G12C",)),
        ("g12c", ("G12C",)),
        ("exon 14 skipping", ("exon 14 skipping",)),
    ],
)
def test_curated_protein_change_spellings_normalize_or_fail_closed(value, expected):
    from trufflepig.reporting import required_protein_changes_for_therapy

    row = {"symbol": "KRAS", "agent": "sotorasib", "required_protein_changes": value}
    assert required_protein_changes_for_therapy(row) == expected


@pytest.mark.parametrize(
    "agent, gene, code, expected",
    [
        ("sotorasib + panitumumab", "KRAS", "COAD", ("G12C",)),
        ("adagrasib", "KRAS", "LUAD", ("G12C",)),
        ("dabrafenib + trametinib", "BRAF", "SKCM", ("V600E", "V600K")),
        ("dabrafenib + trametinib", "BRAF", "THCA", ("V600E",)),
        ("encorafenib + binimetinib", "BRAF", "SKCM", ("V600E", "V600K")),
        ("vemurafenib", "BRAF", "SKCM", ("V600E",)),
        ("MRTX1133", "KRAS", "PAAD", ()),
    ],
)
def test_packaged_protein_change_table_keeps_label_scopes(agent, gene, code, expected):
    from trufflepig.reporting import required_protein_changes_for_therapy

    row = {"symbol": gene, "agent": agent, "cancer_code": code}
    assert required_protein_changes_for_therapy(row) == expected


def test_input_symbol_membership_never_scans_the_whole_input():
    class MembershipOnly(frozenset):
        def __iter__(self):
            raise AssertionError("the input symbol set must not be scanned")

    frame = pd.DataFrame()
    frame.attrs["sample_input_symbols"] = MembershipOnly({"EGFR", "MAGE-A4"})
    assert target_observation_state("EGFR", frame) == "below_detection"
    assert target_observation_state("MAGEA4", frame) == "below_detection"
    assert target_observation_state("ERBB2", frame) == "not_in_input"


@pytest.mark.parametrize("hla_types", [["A*02:01"], []])
def test_landscape_cell_states_each_hla_sentence_once_with_escaped_alleles(hla_types):
    from trufflepig.reporting import therapy_path_context
    from trufflepig.therapy_eligibility import requirement_descriptions

    row = {
        "symbol": "MAGEA4",
        "agent": "afamitresgene autoleucel",
        "cancer_code": "SARC",
        "indication": "synovial sarcoma",
        "treatment_path_tier": "approved_indication_matched",
        "eligibility_note": "requires HLA-A*02 typing and MAGE-A4 IHC",
    }
    analysis = {"cancer_type": "SARC", "analysis_constraints": {"hla_types": hla_types}}
    cell = therapy_path_context(row, analysis=analysis)
    decision = evaluate_therapy_eligibility(row, analysis)
    assert requirement_descriptions(decision, stated=cell) == []
    assert sum(cell.count(p) for p in ("HLA match:", "HLA typing is unavailable")) == 1
    assert "A\\*02" in cell and "A*02:0" not in cell
    assert "Curated eligibility note: requires HLA-A\\*02 typing" in cell


def test_report_clauses_and_guards_tolerate_sentences_and_missing_rows():
    from trufflepig.reporting import join_report_clauses, therapy_path_context, therapy_rationale_paragraphs

    assert join_report_clauses(["Sentence one.", "", "clause two;", "clause three"]) == (
        "Sentence one; clause two; clause three"
    )
    assert therapy_path_context(None) == ""
    assert therapy_rationale_paragraphs(None) == []


def test_request_labels_name_the_evidence_not_the_internal_kind():
    rows = [
        {"symbol": "MAGEA4", "agent": "afami-cel", "phase": "phase_2", "hla_restriction": "A*02:01"},
        {"symbol": "KRAS", "agent": "sotorasib", "cancer_code": "LUAD", "indication": "KRAS G12C LUAD"},
    ]
    assessments = [assess_therapy(row, analysis={"cancer_type": "LUAD"}) for row in rows]
    labels = {request["label"] for request in collect_evidence_requests(assessments)}
    assert {"HLA typing", "KRAS G12C"} <= labels
    assert not any(label in {"Hla", "Mutation", "Tmb high"} for label in labels)
