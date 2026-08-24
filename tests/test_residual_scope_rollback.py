"""Report-scope state remains transactional across residual refit rejection."""

from trufflepig.main import (
    _restore_report_scope_metadata,
    _selected_report_scope_label,
)


def test_auto_detected_call_is_not_marked_explicit_after_residual_rollback():
    # This is the state immediately after the rollback propagation restores the
    # original ranker label but before optional report-scope metadata is reset.
    analysis = {
        "cancer_type": "SARC",
        "report_scope_cancer_type": "SARC",
        "report_scope_parent_cancer_type": "SARC_IMT",
        "cancer_type_evidence": {
            "selected": {
                "cancer_type": "SARC",
                "selected_by": "pan_cancer_signature_ranker",
            }
        },
    }

    _restore_report_scope_metadata(
        analysis,
        report_scope_cancer_type=None,
        report_scope_parent_cancer_type=None,
    )

    assert "report_scope_cancer_type" not in analysis
    assert "report_scope_parent_cancer_type" not in analysis
    assert _selected_report_scope_label(analysis) == ""


def test_explicit_preexisting_report_scope_survives_residual_rollback():
    analysis = {
        "cancer_type": "COAD",
        "report_scope_cancer_type": "READ",
        "report_scope_parent_cancer_type": "CRC",
    }

    _restore_report_scope_metadata(
        analysis,
        report_scope_cancer_type="COAD",
        report_scope_parent_cancer_type="CRC",
    )

    assert analysis["report_scope_cancer_type"] == "COAD"
    assert analysis["report_scope_parent_cancer_type"] == "CRC"
    assert _selected_report_scope_label(analysis) == "COAD"
