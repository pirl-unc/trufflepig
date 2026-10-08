"""Concise sample conclusions and therapeutic leads from structured evidence.

These are presentation decisions, not new diagnoses or eligibility decisions.
RNA follow-up leads remain distinct from the therapy shortlist.
"""

from __future__ import annotations

import math

from .report_language import report_literal


def finding(label, text, *figures):
    return {
        "kind": "bullet",
        "text": f"**{label}:** {text}",
        "figure_suffixes": list(figures),
    }


def sample_takeaways(analysis, ranges_df, report_view, disease_state=""):
    """Lead with the final identity, its evidence, and quantitative limits."""
    from . import brief
    from .decomposition import CancerTypeDecision

    code = report_view.cancer_type
    name = report_literal(report_view.cancer_type_name or code)
    confidence = report_view.call_confidence.tier
    supplied = bool(
        (analysis.get("analysis_constraints") or {}).get("cancer_type")
        or analysis.get("cancer_type_source") == "user-specified"
    )
    unresolved = bool(analysis.get("cancer_type_abstention")) or code == "UNRESOLVED"
    if unresolved:
        identity = (
            "UNRESOLVED. Muscle/stromal RNA similarity cannot establish sarcoma. "
            "Pathology or a disease-defining molecular result is needed before disease-specific therapy selection."
        )
    elif supplied:
        identity = f"{report_literal(code)} ({name}), supplied diagnosis. RNA provides supporting biological context."
    else:
        identity = (
            f"{report_literal(code)} ({name}), "
            + ("**low confidence, provisional**" if confidence == "low" else f"{confidence} confidence, RNA-inferred")
            + ". Confirm the RNA-inferred label with pathology."
        )
    blocks = [{"kind": "paragraph", "text": f"**Cancer call:** {identity}"}]
    decision = CancerTypeDecision.from_analysis(analysis, code)
    lineage = analysis.get("lineage_panel_evidence") or {}
    special_basis = any(analysis.get(key) for key in (
        "fusion_report_scope_inference", "rare_report_scope_inference",
        "fine_report_scope_inference", "cancer_call_rescue",
    ))
    if supplied or special_basis or (decision.refit_confirmed and decision.relationship != "same" and not decision.is_selection_basis):
        blocks.append({"kind": "bullet", "text": brief._cancer_type_basis_line(analysis, code)})
    elif not unresolved and not supplied and decision.is_selection_basis:
        blocks.append(finding(
            "Why this call",
            f"The {report_literal(decision.supported_code)} lineage program persists after background subtraction; "
            "the final refit agrees. Whole-profile similarity is background context.",
            "decomposition-composition.png",
        ))
    elif not unresolved and lineage.get("top_panel") == code and (lineage.get("top_score") or 0) >= 0.5:
        blocks.append(finding(
            "Supporting evidence",
            f"The {report_literal(code)} lineage panel supports this interpretation "
            f"(score {lineage['top_score']:.2f}).",
        ))
    crosscheck = brief._rna_crosscheck_line(analysis, code, call_tier=report_view.call_confidence)
    if crosscheck:
        blocks.append({"kind": "bullet", "text": crosscheck})
    elif not unresolved:
        alternatives = brief._rna_alternatives_line(analysis, code)
        if alternatives:
            codes = ", ".join(report_literal(candidate) for candidate, _ in report_view.cancer_type_alternatives[:3])
            if codes:
                blocks.append(finding(
                    "Remaining uncertainty",
                    f"RNA comparisons retain {codes} as hypotheses. Pathology must resolve the site and subtype; "
                    "the full candidate comparison is in the detailed evidence.",
                ))
            blocks.append({"kind": "bullet", "text": alternatives, "detail": True})

    tissue = analysis.get("healthy_vs_tumor")
    if getattr(tissue, "structural_ambiguity", False):
        normal = getattr(tissue, "top_normal_tissues", [])
        label = str(normal[0][0]).removesuffix("_nTPM").replace("_", " ") if normal else "normal tissue"
        blocks.append(finding(
            "Background matters",
            f"Strong {report_literal(label)} RNA makes bulk similarity nonspecific. "
            "Tumor and normal tissue share lineage programs; interpret targets after source attribution.",
            "decomposition-components.png", "priority-target-context.png",
        ))
    elif getattr(tissue, "cancer_hint", "") == "healthy-dominant":
        blocks.append(finding(
            "Tissue context",
            "The tissue-composition screen is healthy-dominant. Bulk RNA alone may not "
            "distinguish malignant cells from their normal tissue of origin.",
            "decomposition-composition.png",
        ))

    purity = report_view.purity
    if purity.status == "discordant_estimators":
        reason = {
            "cancer_type_unresolved": "tumor identity is unresolved; the modeled fraction can include benign muscle or stroma",
            "same_lineage_not_identifiable": "tumor and benign cells share the modeled lineage programs",
        }.get(purity.unresolved_reason, "the estimators disagree")
        text = f"quantitatively unresolved because {reason}."
        if purity.estimate is not None:
            interval = f" [{purity.lower:.0%}–{purity.upper:.0%}]" if purity.lower is not None and purity.upper is not None else ""
            text += f" The selected operational model uses {purity.estimate:.0%}{interval}; this is not a malignant-cell measurement."
        if purity.unresolved_reason not in {"cancer_type_unresolved", "same_lineage_not_identifiable"}:
            scenarios = brief.purity_estimator_scenario_text(purity.scenarios)
            if scenarios:
                text += " Scenarios: " + scenarios + "."
    elif purity.estimate is not None:
        interval = f" (model interval {purity.lower:.0%}–{purity.upper:.0%})" if purity.lower is not None and purity.upper is not None else ""
        text = f"{purity.estimate:.0%}{interval}; this is RNA fraction, not tumor cellularity."
        if purity.estimate < 0.25:
            text += " Low tumor fraction makes target attribution and negative findings less reliable."
        elif purity.estimate >= 0.995:
            text += " The model reached its ceiling; this does not establish 100% tumor cellularity."
    else:
        text = "Tumor fraction was not resolved; quantitative target attribution is limited."
    blocks.append(finding("Estimated tumor fraction (RNA model)", text, "purity-methods.png"))

    for text in (brief._fusion_evidence_line(analysis, code), brief._variant_evidence_line(analysis)):
        if text:
            blocks.append({"kind": "bullet", "text": text})
    from .infantile_spindle import infantile_spindle_guidance

    spindle = infantile_spindle_guidance(code, analysis)
    if spindle:
        blocks.append(finding("Spindle-cell context", spindle["interpretation"] + " " + "; ".join(spindle["findings"])))
    sarcoma_context = brief.sarcoma_subtype_guidance_markdown(code)
    if sarcoma_context and not unresolved:
        blocks.append({"kind": "bullet", "text": sarcoma_context})
    subtype = brief._lineage_panel_subtype_reasoning_line(analysis, code)
    if subtype and not unresolved:
        blocks.append({"kind": "bullet", "text": subtype, "figure_suffixes": ["subtype-signature.png"]})
    subtype_status = brief.resolved_subtype_summary_line(analysis, ranges_df)
    if subtype_status and not unresolved and not brief.mismatch_repair_rna_state(analysis):
        blocks.append({"kind": "bullet", "text": subtype_status, "figure_suffixes": ["subtype-signature.png"]})
    mmr_channel = brief.mismatch_repair_summary_context(analysis)
    mmr = (mmr_channel.get("details") or {}).get("mismatch_repair") or {}
    state = brief.mismatch_repair_rna_state(analysis)
    probability = mmr.get("msi_probability")
    if state and isinstance(probability, (int, float)) and math.isfinite(probability):
        text = f"{state} expression (MSI-like model score {probability:.2f}). "
        text += "Prioritize clinical MSI/MMR confirmation." if state == "MSI-like" else "This RNA result does not determine clinical MSI/MMR status."
        candidate = brief.candidate_winning_subtype_for_analysis(analysis)
        candidate_state = brief._mismatch_repair_state_from_code(candidate)
        if candidate_state and candidate_state != ("MSI" if state == "MSI-like" else "MSS"):
            text += f" The final MMR evidence conflicts with the candidate-trace subtype {candidate}; that subtype is not adopted."
        mlh1 = mmr.get("mlh1_expression") or {}
        ratio = mlh1.get("cohort_ratio")
        if state == "MSI-like" and isinstance(ratio, (int, float)) and ratio >= brief._MLH1_RETAINED_COHORT_RATIO:
            text += " Retained bulk MLH1 RNA does not exclude MSI or tumor-specific MLH1 loss."
        blocks.append(finding("Mismatch repair", text))
    pathway = brief._pathway_activity_line(analysis)
    if pathway:
        blocks.append({"kind": "bullet", "text": pathway, "figure_suffixes": ["therapy-pathway-state.png"]})
    for text in brief._disease_state_summary_lines(brief.report_disease_state_text(disease_state, analysis=analysis)):
        if text.strip():
            blocks.append({"kind": "bullet", "text": text, "detail": not bool(disease_state),
                           "figure_suffixes": ["therapy-pathway-state.png"]})
    her2 = brief.her2_proxy_summary_line(analysis)
    if her2:
        status = ((analysis.get("rna_biomarker_proxies") or {}).get("her2") or {}).get("status")
        blocks.append({"kind": "bullet", "text": her2, "detail": status not in {"supported", "discordant"}})
    site = brief._inferred_site_context_line(analysis)
    if site:
        blocks.append({"kind": "bullet", "text": site})
    context = analysis.get("sample_context")
    if context is not None:
        prep = brief.library_prep_clause(getattr(context, "library_prep", "unknown"))
        preservation = str(getattr(context, "preservation", "unknown")).replace("_", " ")
        support = "" if preservation == "unknown" else f" ({brief.heuristic_support_label(getattr(context, 'preservation_confidence', 0.0))})"
        blocks.append(finding("Sample quality", f"{prep}; preservation inferred as {preservation} from RNA QC{support}.", "sample-context.png", "degradation-index.png"))
    for text in (brief.rna_quant_qc_summary_line(analysis.get("rna_quant_qc")), brief._cancer_call_rescue_summary_line(analysis)):
        if text:
            blocks.append({"kind": "bullet", "text": text, "figure_suffixes": ["sample-context.png"],
                           "detail": not bool((analysis.get("rna_quant_qc") or {}).get("warnings") or analysis.get("cancer_call_rescue"))})
    for warning in (analysis.get("expression_scale_qc") or {}).get("warnings") or []:
        blocks.append(finding("Quality limitation", report_literal(warning), "sample-context.png"))
    if (analysis.get("expression_scale_qc") or {}).get("converted_from") == "log2_tpm_plus_one":
        blocks.append({
            **finding("Expression scale QC", "Input resembled log2(TPM+1); converted to linear TPM before interpretation."),
            "detail": True,
        })
    rescue = brief.expression_qc_rescue_summary_line(analysis.get("expression_qc_rescue"))
    if rescue:
        blocks.append({"kind": "bullet", "text": rescue, "figure_suffixes": ["sample-context.png"],
                       "detail": not bool((analysis.get("expression_qc_rescue") or {}).get("high_burden"))})
    return blocks


def therapeutic_lead_basis(assessment, analysis):
    """Describe sample support without turning unconfirmed RNA into eligibility."""
    from .brief import mismatch_repair_rna_state

    eligibility = assessment["eligibility"]
    requirements = eligibility["requirements"]
    if analysis.get("cancer_type_abstention") or any(r["status"] == "blocked" for r in requirements):
        return ""
    if eligibility.get("supplied_variant_supported"):
        return "Supplied molecular evidence matches the curated target requirement."
    if eligibility.get("history_supported"):
        return "Supplied treatment history records benefit."
    if eligibility.get("direct_evidence_supported"):
        return "Supplied clinical evidence supports the biomarker requirement."
    if any(r["kind"] == "msi_high" for r in requirements):
        if mismatch_repair_rna_state(analysis) == "MSI-like":
            return "MSI-like RNA prioritizes testing; clinical MSI-H/dMMR is unconfirmed."
        return ""
    if any(r["kind"] in {"mutation", "tmb_high", "histology_only"} for r in requirements):
        # Target abundance cannot supply a genotype, mutation burden or diagnosis.
        return ""
    if any(r["kind"] == "clinical_target_assay" for r in requirements):
        # The required assay can measure a different phenotype from the drug
        # target (for example, hormone receptors for a CDK4/6 inhibitor).
        # Only a matching, supported RNA proxy prioritizes that assay here.
        her2 = (analysis.get("rna_biomarker_proxies") or {}).get("her2") or {}
        if assessment["target"] != "ERBB2" or her2.get("status") != "supported":
            return ""
    source = assessment.get("rna_source") or {}
    observation = assessment.get("observation") or {}
    if source.get("tier") == "tumor_supported" and observation.get("state") == "measured":
        normal = assessment.get("normal_context") or {}
        return (
            f"{assessment['target']} RNA remains in the estimated tumor component "
            f"({assessment['tumor_band']} TPM). "
            + (f"Healthy-tissue context: {normal['label']}. " if normal.get("label") else "")
            + "Confirm the indication-specific biomarker requirements."
        )
    if assessment["selected"]:
        return "Disease-matched curated treatment pathway; selection does not establish individual treatment fitness."
    return ""
