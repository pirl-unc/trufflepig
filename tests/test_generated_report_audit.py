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
