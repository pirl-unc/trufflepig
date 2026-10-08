import json

from scripts.analyze_reports import Compat, expected_codes
from scripts.audit_generated_reports import _sample_issues


def _compat():
    return Compat(
        {
            "COAD": "CRC",
            "READ": "CRC",
            "SARC_OS": "SARC",
        },
        lambda code: {
            "CRC": "solid",
            "COAD": "solid",
            "READ": "solid",
            "SARC": "mesenchymal",
            "SARC_OS": "mesenchymal",
        }.get(code, ""),
    )


def _write_report_tree(tmp_path, analysis_text, summary_call="READ"):
    sample = "sample"
    analysis = tmp_path / f"{sample}-analysis.md"
    summary = tmp_path / f"{sample}-summary.md"
    evidence = tmp_path / f"{sample}-evidence.md"
    decomposition = tmp_path / f"{sample}-decomposition-hypotheses.tsv"
    ranges = tmp_path / f"{sample}-tumor-expression-ranges.tsv"
    signal_matrix = tmp_path / f"{sample}-cancer-type-signal-matrix.tsv"
    analysis.write_text(analysis_text)
    summary.write_text(f"**Cancer call:** {summary_call}\n")
    evidence.write_text("evidence\n")
    decomposition.write_text("cancer_type\ttemplate\twarnings\n")
    ranges.write_text("gene\tsample_tpm\n")
    signal_matrix.write_text(
        "signal_source\trole\tpredicted_code\tcontext_code\tsupport\n"
    )
    return {
        "analysis": analysis,
        "summary": summary,
        "evidence": evidence,
        "decomposition": decomposition,
        "ranges": ranges,
        "signal_matrix": signal_matrix,
    }


def test_pipe_delimited_truth_is_shared_by_report_scorer_and_auditor(tmp_path):
    paths = _write_report_tree(
        tmp_path,
        "**Working cancer call**: READ (Rectum Adenocarcinoma).\n",
    )

    assert expected_codes("CRC|COAD|READ") == ["CRC", "COAD", "READ"]
    issues = _sample_issues(
        sample_id="sample",
        expected="CRC|COAD|READ",
        paths=paths,
        compat=_compat(),
    )

    assert not [
        issue for issue in issues
        if issue["category"] == "headline_incompatible_with_expected"
    ]


def test_full_report_audit_catches_detected_gene_narrated_as_absent(tmp_path):
    paths = _write_report_tree(
        tmp_path,
        "\n".join(
            [
                "**Working cancer call**: READ (Rectum Adenocarcinoma).",
                "| Expected high marker | TPM | Source |",
                "|---|---:|---|",
                "| MUC2 | 70.5 | lineage panel |",
                "",
                "**Not detected**: MUC2, KRT20.",
            ]
        ),
    )

    issues = _sample_issues(
        sample_id="sample",
        expected="READ",
        paths=paths,
        compat=_compat(),
    )

    contradiction = next(
        issue for issue in issues
        if issue["category"] == "detected_gene_narrated_absent"
    )
    assert "MUC2" in contradiction["detail"]
    assert "70.5 TPM" in contradiction["detail"]


def test_full_report_audit_rejects_fabricated_zero_reference_table(tmp_path):
    paths = _write_report_tree(
        tmp_path,
        "\n".join(
            [
                "**Working cancer call**: SARC_OS (Osteosarcoma).",
                "#### Signature evidence for **SARC_WDLPS**",
                "",
                "| Gene | Sample TPM | SARC_LPS_UNSPEC median |",
                "|---|---:|---:|",
                "| MDM2 | 953.4 | 0 |",
                "| CDK4 | 91.1 | 0 |",
                "",
            ]
        ),
        summary_call="SARC_OS",
    )

    issues = _sample_issues(
        sample_id="sample",
        expected="SARC_OS|SARC",
        paths=paths,
        compat=_compat(),
    )

    issue = next(
        issue for issue in issues
        if issue["category"] == "unusable_signature_reference_medians"
    )
    assert "all candidate reference medians are zero" in issue["detail"]


def test_full_report_audit_rejects_sample_vote_copied_to_every_candidate(tmp_path):
    paths = _write_report_tree(
        tmp_path,
        "**Working cancer call**: READ (Rectum Adenocarcinoma).\n",
    )
    paths["signal_matrix"].write_text(
        "\t".join(
            ["signal_source", "role", "predicted_code", "context_code", "support"]
        )
        + "\n"
        + "\n".join(
            "\t".join(
                [
                    "learned_expression_classifier",
                    "hierarchical_entity_vote",
                    "READ",
                    candidate,
                    "0.8",
                ]
            )
            for candidate in ("COAD", "READ", "STAD")
        )
        + "\n"
    )

    issues = _sample_issues(
        sample_id="sample",
        expected="READ",
        paths=paths,
        compat=_compat(),
    )

    issue = next(
        issue for issue in issues
        if issue["category"] == "duplicated_global_learned_vote"
    )
    assert "repeated across 3 candidate rows" in issue["detail"]


def test_full_report_audit_rejects_nearly_uninformative_purity_interval(tmp_path):
    paths = _write_report_tree(
        tmp_path,
        "**Working cancer call**: SARC_OS (Osteosarcoma).\n",
        summary_call="SARC_OS",
    )
    paths["summary"].write_text(
        "\n".join(
            [
                "**Cancer call:** SARC_OS",
                "**Purity:** 12% (model interval 2%–98%, low confidence).",
            ]
        )
    )

    issues = _sample_issues(
        sample_id="sample",
        expected="SARC_OS|SARC",
        paths=paths,
        compat=_compat(),
    )

    issue = next(
        issue
        for issue in issues
        if issue["category"] == "uninformative_purity_interval"
    )
    assert issue["severity"] == "error"
    assert "2%–98%" in issue["detail"]


def test_full_report_audit_rejects_zero_width_purity_interval(tmp_path):
    paths = _write_report_tree(
        tmp_path,
        "**Working cancer call**: READ (Rectum Adenocarcinoma).\n",
    )
    paths["summary"].write_text(
        "\n".join(
            [
                "**Cancer call:** READ",
                "**Purity:** 70% (model interval 70%–70%, degenerate confidence).",
            ]
        )
    )

    issues = _sample_issues(
        sample_id="sample",
        expected="READ|CRC",
        paths=paths,
        compat=_compat(),
    )

    issue = next(
        issue
        for issue in issues
        if issue["category"] == "degenerate_purity_interval"
    )
    assert issue["severity"] == "error"
    assert "70%–70%" in issue["detail"]


def test_full_report_audit_requires_matching_provisional_status(tmp_path):
    paths = _write_report_tree(
        tmp_path,
        "**Working cancer call**: READ (Rectum Adenocarcinoma).\n",
    )
    paths["summary"].write_text(
        "**Cancer call:** READ (Rectum Adenocarcinoma) — "
        "**low confidence, provisional**\n"
    )

    issues = _sample_issues(
        sample_id="sample",
        expected="READ|CRC",
        paths=paths,
        compat=_compat(),
    )

    issue = next(
        issue
        for issue in issues
        if issue["category"] == "cancer_call_provisional_status_mismatch"
    )
    assert issue["severity"] == "error"


def test_full_report_audit_requires_authored_content_to_match_summary(tmp_path):
    paths = _write_report_tree(
        tmp_path,
        "**Working cancer call**: READ (Rectum Adenocarcinoma).\n",
    )
    paths["summary"].write_text(
        """**Cancer call:** READ (Rectum Adenocarcinoma)

## Top candidate therapies

- **EGFR** — cetuximab (Approved, RAS-WT mCRC). mixed-source; 20 tumor-source bulk TPM (model interval 10-30); confirm RAS-WT status.
"""
    )
    report_path = tmp_path / "sample-report.json"
    report_path.write_text(json.dumps({"schema_version": 2, "sections": []}))
    paths["report"] = report_path

    issues = _sample_issues(
        sample_id="sample",
        expected="READ|CRC",
        paths=paths,
        compat=_compat(),
    )

    issue = next(
        issue
        for issue in issues
        if issue["category"] == "authored_summary_mismatch"
    )
    assert issue["severity"] == "error"


def _with_structured_report(paths, tmp_path, document, summary_text=None):
    report = tmp_path / "sample-report.json"
    report.write_text(json.dumps(document))
    if summary_text is not None:
        paths["summary"].write_text(summary_text, encoding="utf-8")
    return {**paths, "report": report}


def _audit(paths):
    return {
        issue["category"]
        for issue in _sample_issues(sample_id="sample", expected="READ", paths=paths, compat=_compat())
    }


def test_legacy_report_schema_is_flagged_instead_of_failing_every_summary(tmp_path):
    paths = _write_report_tree(tmp_path, "**Working cancer call**: READ (Rectum Adenocarcinoma).\n")
    paths = _with_structured_report(paths, tmp_path, {"schema_version": 1, "sample_id": "sample"})
    categories = _audit(paths)
    assert "legacy_structured_report" in categories
    assert "authored_summary_mismatch" not in categories


def test_audit_requires_selected_therapies_citations_and_requests_in_the_summary(tmp_path):
    from trufflepig.report_language import markdown_url, render_report_template

    paths = _write_report_tree(tmp_path, "**Working cancer call**: READ (Rectum Adenocarcinoma).\n")
    url = "https://example.org/label(2026).pdf"
    decisions = {
        "schema_version": 2,
        "sample_id": "sample",
        "therapy_assessments": [
            {
                "agent": "sotorasib + panitumumab",
                "selected": True,
                "source_url": url,
                "eligibility": {"permits_review": True},
            }
        ],
        "evidence_requests": [
            {"key": "msi_high", "keys": ["msi_high"], "question": "Supply the clinical MSI result."}
        ],
    }
    uncited = [{"id": "therapies", "title": "Therapies", "blocks": [{"kind": "paragraph", "text": "None."}]}]
    document = {**decisions, "sections": uncited}
    summary = render_report_template("report", sample_id="sample", sections=uncited)
    categories = _audit(_with_structured_report(paths, tmp_path, document, summary))
    assert "authored_summary_mismatch" not in categories
    assert {
        "selected_therapy_missing_from_summary",
        "therapy_source_missing_from_summary",
        "evidence_request_missing_from_summary",
    } <= categories

    cited = [
        {
            "id": "therapies",
            "title": "Therapies",
            "blocks": [
                {
                    "kind": "table",
                    "headers": ["Recommendation", "Evidence source"],
                    "rows": [["sotorasib \\+ panitumumab", f"[FDA label]({markdown_url(url)})"]],
                },
                {"kind": "paragraph", "text": "**MSI/MMR:** Supply the clinical MSI result."},
            ],
        }
    ]
    document = {**decisions, "sections": cited}
    summary = render_report_template("report", sample_id="sample", sections=cited)
    categories = _audit(_with_structured_report(paths, tmp_path, document, summary))
    assert not categories & {
        "authored_summary_mismatch",
        "selected_therapy_missing_from_summary",
        "therapy_source_missing_from_summary",
        "evidence_request_missing_from_summary",
    }
