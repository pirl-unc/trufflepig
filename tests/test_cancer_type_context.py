from trufflepig.analyze import cancer_type_context_from_analysis


def test_context_keeps_prad_as_single_active_label_when_no_refined_label_exists():
    context = cancer_type_context_from_analysis(
        {
            "cancer_type": "PRAD",
            "reference_cancer_type": "PRAD",
            "cancer_type_source": "auto-detected",
        }
    )

    assert context.code_for("report") == "PRAD"
    assert context.code_for("reference") == "PRAD"
    assert context.code_for("expression") == "PRAD"
    assert context.relationship == "same"
    assert not context.uses_distinct_reference


def test_context_exposes_refined_child_and_parent_reference():
    context = cancer_type_context_from_analysis(
        {
            "cancer_type": "SARC_SYN",
            "reference_cancer_type": "SARC",
            "report_scope_cancer_type": "SARC_SYN",
            "report_scope_parent_cancer_type": "SARC",
            "analysis_constraints": {"cancer_type": "SARC_SYN"},
            "cancer_type_source": "user-specified",
        }
    )

    assert context.code_for("therapy") == "SARC_SYN"
    assert context.code_for("cohort") == "SARC"
    assert context.code_for("parent") == "SARC"
    assert context.relationship == "fine_child_of_reference"
    assert context.uses_distinct_reference


def test_context_defaults_refined_label_to_registry_parent_reference():
    context = cancer_type_context_from_analysis({"cancer_type": "SARC_SYN"})

    assert context.code_for("report") == "SARC_SYN"
    assert context.code_for("reference") == "SARC"
    assert context.code_for("parent") == "SARC"
    assert context.relationship == "fine_child_of_reference"
    assert context.uses_distinct_reference


def test_context_prefers_fine_expression_reference_when_available():
    context = cancer_type_context_from_analysis(
        {
            "cancer_type": "OS",
            "reference_cancer_type": "SARC",
            "report_scope_cancer_type": "OS",
            "analysis_constraints": {"cancer_type": "OS"},
            "cancer_type_source": "user-specified",
        }
    )

    assert context.code_for("report") == "OS"
    assert context.code_for("reference") == "SARC"
    assert context.report_has_expression_ref
    assert context.code_for("expression") == "OS"
