"""Identity constraints that expression similarity cannot establish.

Sarcoma cohorts share lineage programs with benign muscle and stroma. Multiple
transformations of that RNA signal, including residuals, are not independent
proof of sarcoma. Until a selector is validated against those backgrounds,
only supplied disease context or a defining molecular result can establish it.
"""

from __future__ import annotations

from collections.abc import Mapping


UNRESOLVED_CANCER_TYPE = "UNRESOLVED"
SARCOMA_IDENTITY_REASON = (
    "RNA similarity to sarcoma, muscle or stroma does not establish sarcoma. "
    "A supplied diagnosis or disease-defining molecular result is required; "
    "signature, centroid, learned-classifier and decomposition agreement "
    "cannot replace that evidence."
)


def requires_independent_sarcoma_identity(code: str) -> bool:
    """Cover the entire registry family, including future non-prefixed codes."""
    code = str(code or "").strip()
    if code == "SARC" or code.startswith("SARC_"):
        return True
    if not code or code == UNRESOLVED_CANCER_TYPE:
        return False
    from .cancer_ontology import cancer_family

    return cancer_family(code).lower() == "sarcoma"


def sarcoma_identity_is_supplied(analysis: Mapping, code: str) -> bool:
    """Accept scoped input context or the existing defining-fusion selector."""
    supplied = str((analysis.get("analysis_constraints") or {}).get("cancer_type") or "")
    if supplied:
        return supplied == code
    if analysis.get("cancer_type_source") == "user-specified":
        return str(analysis.get("cancer_type") or "") == code
    selected = (analysis.get("cancer_type_evidence") or {}).get("selected") or {}
    return bool(
        selected.get("cancer_type") == code
        and selected.get("selected_by") == "direct_fusion"
        and selected.get("can_select_report_label")
        and (selected.get("metrics") or {}).get("direct_fusion_support", 0) > 0
    )


def finalize_sarcoma_identity(analysis: dict) -> bool:
    """Abstain after exploratory decomposition, preserving its reference data.

    This also closes the legacy ranker fallback when no evidence selector can
    establish a label. Reference codes remain model context, never diagnosis.
    """
    code = str(analysis.get("cancer_type") or "")
    if not requires_independent_sarcoma_identity(code) or sarcoma_identity_is_supplied(analysis, code):
        return False
    analysis["cancer_type_abstention"] = {
        "status": "unresolved",
        "context_code": code,
        "reason": SARCOMA_IDENTITY_REASON,
    }
    analysis.setdefault("reference_cancer_type", code)
    analysis.setdefault("expression_reference_cancer_type", code)
    analysis["cancer_type"] = UNRESOLVED_CANCER_TYPE
    analysis["report_scope_cancer_type"] = UNRESOLVED_CANCER_TYPE
    analysis["inferred_cancer_type"] = UNRESOLVED_CANCER_TYPE
    analysis["cancer_name"] = "Unresolved cancer type"
    evidence = analysis.get("cancer_type_evidence")
    if isinstance(evidence, dict):
        evidence["selected"] = None
        graph = evidence.get("staged_evidence_graph")
        if isinstance(graph, dict):
            graph["selected"] = None
            graph["lineage_path"] = []
            graph["orthogonal_axes"] = []
            for stage in graph.get("stages") or []:
                stage.update(status="not_resolved", code="", basis="identity_abstention")
                if "family" in stage:
                    stage["family"] = ""
    for key in ("rare_report_scope_inference", "fine_report_scope_inference", "cancer_call_rescue"):
        if key in analysis:
            analysis["retained_" + key] = analysis.pop(key)
    return True


def mark_unresolved_identity_purity(analysis: dict) -> None:
    """A conditional decomposition fraction cannot become established purity."""
    if analysis.get("cancer_type") != UNRESOLVED_CANCER_TYPE:
        return
    purity = analysis["purity"]
    purity["quantitative_status"] = "discordant_estimators"
    purity["quantitative_unresolved_reason"] = "cancer_type_unresolved"


def reportable_rare_marker_hypotheses(analysis: Mapping) -> list:
    """Do not turn shared structural RNA into sarcoma-specific testing prompts."""
    return [
        finding for finding in (analysis.get("rare_marker_hypotheses") or [])
        if not requires_independent_sarcoma_identity(finding.get("cancer_type"))
        or sarcoma_identity_is_supplied(analysis, finding.get("cancer_type"))
    ]
