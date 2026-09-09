"""Build the reader's four report sections directly from evidence decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from collections import Counter
from typing import Any
import hashlib
import json

from .report_language import render_report_paragraph, render_report_template
from .reporting import (
    canonical_target_symbol,
    clinical_maturity_summary,
    expression_independent_indication,
    expression_independent_rna_context,
    normal_expression_context,
    therapy_state_caution,
    tumor_attribution_context,
    tumor_band_available,
    tumor_band_cell,
)
from .therapeutic_agents import resolve_therapy_identity
from .therapy_eligibility import (
    EvidenceRequirement,
    clean_therapy_value,
    collect_evidence_requests,
    evaluate_therapy_review,
    msi_mmr_requirement,
)
from .treatment_history import treatment_history_summary_lines, treatment_records


@dataclass
class ReportContent:
    """Serializable prose and evidence, authored once before format rendering."""

    sample_id: str
    sections: list[dict[str, Any]]
    therapy_assessments: list[dict[str, Any]]
    evidence_requests: list[dict[str, Any]]
    therapy: dict[str, Any] | None
    treatment_history: list[dict[str, Any]]
    clinical_context: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)

    @property
    def specimen_blocks(self) -> list[dict[str, Any]]:
        """Reuse authored specimen context and provenance in detailed documents."""
        return [
            block for section in self.sections for block in section["blocks"]
            if block.get("id") in {"specimen_context", "specimen_provenance"}
        ]

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def paragraph(text: str) -> dict:
    return {"kind": "paragraph", "text": text}


def therapy_assessment_id(row) -> str:
    """Stable identity for one complete curated row, including disease scope."""
    facts = {str(key): clean_therapy_value(value) for key, value in row.items()}
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()[:16]


def assess_therapy(
    row,
    expr=None,
    *,
    analysis=None,
    panel_subtype=None,
    target_panel=None,
    ranges_df=None,
    disease_state="",
    selected=False,
) -> dict:
    """Explain one curated treatment with its identity, requirements and RNA evidence.

    This function does not select a treatment. Call ``recommend_therapies`` for
    ranking; use this assessment to inspect selected and excluded alternatives.
    """
    from .brief import _therapy_agent_label, _phase_label

    agent = _therapy_agent_label(row)
    identity = resolve_therapy_identity(row.get("agent"))
    gene = canonical_target_symbol(row.get("symbol"))
    review = evaluate_therapy_review(
        row, expr, analysis=analysis, panel_subtype=panel_subtype,
        ranges_df=ranges_df, disease_state=disease_state,
    )
    observation = review.observation
    state, observed = observation["state"], observation["observed_tpm"]
    eligibility = review.eligibility
    rationale = []
    rationale.extend(r.description for r in eligibility.requirements if r.kind in {"msi_high", "tmb_high"})
    if eligibility.supplied_variant_supported:
        from .reporting import supplied_variant_context_for_target_row

        rationale.append(supplied_variant_context_for_target_row(row, analysis).rstrip(". ") + ".")
    from .reporting import therapy_rationale_paragraphs

    rationale.extend(therapy_rationale_paragraphs(row, analysis=analysis))
    if expression_independent_indication(row):
        rationale.append(expression_independent_rna_context(expr, observation_state=state))
    if state in {"measured", "below_detection"} and expr is not None and tumor_band_available(expr):
        source = tumor_attribution_context(expr)
        normal = normal_expression_context(expr)
        notes = list(
            dict.fromkeys(list(source.get("notes") or []) + list(normal.get("details") or []))
        )
        rationale.append(
            render_report_paragraph(
                "therapy_rna_attribution",
                target=gene,
                source=source,
                normal=normal,
                notes=[note.rstrip(". ") for note in notes],
            )
        )
    elif not expression_independent_indication(row):
        rationale.append(
            render_report_paragraph(
                "rna_observation", state=state, observed_tpm=observed, context_only=False
            )
        )
    caution = therapy_state_caution(row, analysis=analysis, disease_state=disease_state)
    if caution:
        rationale.append("Current-treatment context: " + caution.rstrip(". ") + ".")
    return {
        "id": therapy_assessment_id(row),
        "identity": asdict(identity),
        "agent": agent,
        "canonical_agent": identity.canonical_name,
        "target": gene or "Clinical pathway",
        "selected": selected,
        "phase": _phase_label(clean_therapy_value(row.get("phase"))),
        "indication": clean_therapy_value(row.get("indication")),
        "maturity": clinical_maturity_summary(row, target_panel=target_panel),
        "eligibility": eligibility.public_dict(),
        "selection": {
            "status": review.status,
            "reason": review.reason,
            "permits_review": review.permits_review,
        },
        "rationale": rationale,
        "observation": observation,
        "tumor_band": tumor_band_cell(expr)
        if state in {"measured", "below_detection"}
        and expr is not None
        and tumor_band_available(expr)
        else "—",
        "curation": {
            key: clean_therapy_value(row.get(key))
            for key in (
                "eligibility_note", "clinical_setting_note", "line_of_therapy",
                "treatment_path_tier", "rationale",
            )
        },
        "source": clean_therapy_value(row.get("therapy_evidence_source")),
        "source_url": clean_therapy_value(row.get("therapy_evidence_url")),
    }


def empty_shortlist_summary(assessments: list[dict]) -> str:
    """Explain an empty shortlist from its recorded selection decisions.

    Counts concern curated therapy rows, which can share an agent or target.
    This formatter never reevaluates eligibility, reads RNA tables, or infers
    an exclusion from an observation alone.
    """
    if any(a["selected"] for a in assessments):
        raise ValueError("An empty-shortlist explanation requires no selected therapies")
    descriptions = {
        "clinical_blocker": "blocked by known clinical or disease-scope exclusions",
        "eligibility_pending": "awaiting clinical eligibility evidence",
        "inactive_disease_context": "not supported by the RNA disease-state context",
        "rna_unavailable": "missing usable target RNA for an RNA-dependent pathway",
        "rna_below_threshold": "below the target-RNA discovery threshold",
        "rna_source_unsupported": "without sufficient tumor-source RNA support",
        "reviewable": "reviewable but outside the reported shortlist",
    }
    counts = Counter(a["selection"]["status"] for a in assessments)
    groups = [
        {"count": counts[status], "description": description}
        for status, description in descriptions.items() if counts[status]
    ]
    return render_report_paragraph("empty_shortlist", total=len(assessments), groups=groups)


def build_report_content(
    analysis,
    ranges_df,
    cancer_code: str,
    disease_state: str,
    sample_id=None,
    *,
    report_view,
    prefix: str = "",
) -> ReportContent:
    """Create conclusion, therapies, information requests and detailed evidence.

    Clinical status comes from the same eligibility API used by recommendation
    selection. JSON retains exact identities, observation state and requirements;
    no clinical meaning is recovered from generated Markdown.
    """
    from .brief import (
        _caveats_from_purity_tier,
        _curated_target_panel_for_sample,
        _display_sample_id,
        _format_biomarker_outlier_bullet,
        _format_cta_outlier_bullet,
        _notable_biomarker_outliers,
        _notable_cta_outliers,
        _therapy_agent_label,
        recommend_therapies,
        summary_conclusion_paragraphs,
        source_attribution_rows,
        _format_trace_tpm,
    )
    from .common import ranges_by_symbol, ranges_by_gene_id, panel_symbols_to_gene_ids
    from .clinical_context import clinical_context_for_analysis, evaluate_msi_mmr, report_identity

    cancer_code = report_view.cancer_type
    analysis = {**analysis, "cancer_type": cancer_code, "cancer_name": report_view.cancer_type_name}
    sample_id = _display_sample_id(sample_id) or report_view.sample_id or ""
    prefix = prefix or sample_id or "report"
    clinical_context = clinical_context_for_analysis(analysis)
    identity = report_identity(
        clinical_context, **(analysis.get("report_input") or {}),
        output_prefix=prefix, fallback_label=sample_id,
    )
    research = identity["purpose"] == "research"
    conclusion = [
        paragraph(text)
        for text in summary_conclusion_paragraphs(
            analysis,
            ranges_df,
            cancer_code,
            disease_state,
            sample_id,
            report_view=report_view,
        )
        if text.strip()
    ]
    conclusion.insert(0, {
        **paragraph(render_report_paragraph("specimen_context", context=clinical_context, identity=identity)),
        "id": "specimen_context",
    })
    panel_code, panel_subtype, panel = _curated_target_panel_for_sample(
        cancer_code,
        analysis,
        ranges_df=ranges_df,
    )
    recommended = recommend_therapies(
        panel,
        ranges_df,
        analysis=analysis,
        disease_state=disease_state,
        panel_subtype=panel_subtype,
    )
    selected = {therapy_assessment_id(row.therapy) for row in recommended}
    expression = ranges_by_symbol(ranges_df) if ranges_df is not None else {}
    gene_expression = ranges_by_gene_id(ranges_df) if ranges_df is not None else {}
    assessments = []
    rows = panel.to_dict("records") if panel is not None else []
    gene_ids = panel_symbols_to_gene_ids(canonical_target_symbol(row.get("symbol")) for row in rows)
    for row in rows:
        agent = _therapy_agent_label(row)
        if not agent:
            continue
        gene = canonical_target_symbol(row.get("symbol"))
        expr = gene_expression.get(gene_ids.get(gene))
        if expr is None:
            expr = expression.get(gene)
        assessment = assess_therapy(
            row,
            expr,
            analysis=analysis,
            panel_subtype=panel_subtype,
            target_panel=panel,
            ranges_df=ranges_df,
            disease_state=disease_state,
            selected=therapy_assessment_id(row) in selected,
        )
        assessments.append(assessment)

    selected_assessments = []
    for recommendation in recommended:
        key = therapy_assessment_id(recommendation.therapy)
        match = next((a for a in assessments if a["id"] == key and a["selected"]), None)
        if match is not None and match not in selected_assessments:
            selected_assessments.append(match)

    # Report-level questions join therapy requirements before deduplication.
    from .brief import mismatch_repair_summary_context, mismatch_repair_rna_state
    clinical_mmr = evaluate_msi_mmr(clinical_context)
    rna_mmr = mismatch_repair_summary_context(analysis)
    rna_state = mismatch_repair_rna_state(analysis)
    if clinical_mmr.assays:
        conclusion.append(paragraph(
            "**Clinical MSI/MMR evidence:** " + msi_mmr_requirement(analysis).description
        ))
        if (clinical_mmr.status, rna_state) in {("positive", "MSS-like"), ("negative", "MSI-like")}:
            conclusion.append(paragraph(render_report_paragraph(
                "clinical_rna_discordance", rna_state=rna_state
            )))

    from .infantile_spindle import infantile_spindle_guidance

    spindle_guidance = infantile_spindle_guidance(cancer_code, analysis)
    report_requirements = []
    if spindle_guidance:
        report_requirements.append(
            EvidenceRequirement(
                "spindle_diagnostic_context",
                "diagnosis",
                "unresolved",
                "Overlapping kinase events do not establish the exact spindle-cell diagnosis.",
                spindle_guidance["workup"],
                ("pathology and primary-site report", "RNA/DNA structural-variant report"),
            )
        )
    constraints = analysis.get("analysis_constraints") or {}
    if (
        not constraints.get("cancer_type")
        and analysis.get("cancer_type_source") != "user-specified"
    ):
        report_requirements.append(
            EvidenceRequirement(
                "diagnosis",
                "diagnosis",
                "unresolved",
                "The disease call is based on RNA evidence and needs clinical reconciliation.",
                "Reconcile the RNA disease context with the sample's documented origin and disease-defining molecular evidence."
                if research else "Reconcile the proposed disease and subtype with pathology, clinical history and any disease-defining molecular assay.",
                ("sample provenance", "reference-material characterization", "disease-defining molecular result")
                if research else ("pathology report", "confirmed cancer type", "disease-defining molecular result"),
            )
        )
    if rna_mmr or clinical_mmr.assays:
        report_requirements.append(msi_mmr_requirement(analysis, rna_triage=rna_state == "MSI-like"))
    clinical_requirements = []
    if selected_assessments and not research:
        clinical_requirement = EvidenceRequirement(
            "clinical_setting",
            "clinical_setting",
            "unresolved",
            "A report candidate is not confirmation of individual treatment fitness.",
            "Reconcile the supplied clinical assay reports, current disease setting, treatment sequence, response, toxicity and organ function before choosing a treatment. For trial options, verify protocol criteria and current recruitment.",
            (
                "current oncology assessment",
                "treatment and toxicity history",
                "relevant laboratory and imaging results",
            ),
        )
        clinical_requirements = [
            {
                "agent": assessment["agent"],
                "eligibility": {"requirements": [clinical_requirement.public_dict()]},
            }
            for assessment in selected_assessments
        ]
        for assessment, clinical in zip(selected_assessments, clinical_requirements):
            note = (
                assessment["curation"]["clinical_setting_note"]
                or assessment["curation"]["eligibility_note"]
            )
            # A satisfied molecular gate does not establish treatment setting
            # or fitness. Preserve those curated criteria in the shared list.
            if note and not any(
                r["question"] == note and r["status"] in {"missing", "unresolved"}
                for r in assessment["eligibility"]["requirements"]
            ):
                clinical["eligibility"]["requirements"].append(
                    replace(clinical_requirement, question=note).public_dict()
                )
    requests = collect_evidence_requests(
        [
            *assessments,
            *clinical_requirements,
            *({
                "agent": "report conclusion",
                "eligibility": {"requirements": [requirement.public_dict()]},
            } for requirement in report_requirements),
        ]
    )

    therapies = [
        paragraph(render_report_paragraph(
            "therapy_scope", scope=panel_code or cancer_code, purpose=identity["purpose"],
        ))
    ] if assessments else []

    if spindle_guidance:
        therapies.append(paragraph(spindle_guidance["therapy"]))

    for index, assessment in enumerate(selected_assessments, 1):
        therapies.append(
            {"kind": "heading", "text": f"{index}. {assessment['agent']} · {assessment['phase']}"}
        )
        therapies.append(
            paragraph(
                render_report_paragraph(
                    "therapy_candidate",
                    indication=assessment["indication"],
                    maturity=assessment["maturity"],
                )
            )
        )
        therapies.extend(paragraph(text) for text in assessment["rationale"])
    if not selected_assessments:
        therapies.append(paragraph(empty_shortlist_summary(assessments)))
    for title, statuses in (
        ("Known blockers and current treatments", {"clinical_blocker"}),
        ("Eligibility pending", {"eligibility_pending"}),
        ("Sample support not established", {
            "inactive_disease_context", "rna_unavailable", "rna_below_threshold", "rna_source_unsupported",
        }),
    ):
        excluded = [a for a in assessments if a["selection"]["status"] in statuses]
        if excluded:
            therapies.append({"kind": "heading", "text": title})
            therapies.extend(
                {"kind": "bullet", "text": f"**{a['agent']}:** {a['selection']['reason']}"}
                for a in excluded
            )

    request_blocks = [paragraph(render_report_paragraph("research_information"))] if research else []
    for request in requests:
        label = {"msi_high": "MSI/MMR", "tmb_high": "TMB"}.get(
            request["kind"], request["kind"].replace("_", " ").capitalize()
        )
        request_blocks.append(
            paragraph(render_report_paragraph("evidence_request", label=label, **request))
        )
        # Different therapies can require distinct tests for one evidence kind.
        # Keep those specifications visible even when the request is deduplicated.
        for detail in request["details"]:
            if detail["question"] != request["question"]:
                request_blocks.append({
                    "kind": "bullet",
                    "text": render_report_paragraph("evidence_request_detail", **detail),
                })
    if not requests:
        request_blocks.append(
            paragraph(
                "No additional therapy-specific information requests were identified by the curated rules."
            )
        )

    detail = [
        paragraph(
            f"[Full interpreted analysis]({prefix}-analysis.md) · [Detailed evidence tables]({prefix}-evidence.md)"
        )
    ]
    detail.append({"kind": "heading", "text": "Specimen provenance"})
    detail.append({
        **paragraph(render_report_paragraph("specimen_provenance", context=clinical_context, identity=identity)),
        "id": "specimen_provenance",
    })
    source_rows = source_attribution_rows(panel, ranges_df, recommended)
    if source_rows:
        detail.append({"kind": "heading", "text": "Where target RNA signal appears to come from"})
        for row in source_rows:
            detail.append(
                paragraph(
                    render_report_paragraph(
                        "target_attribution",
                        symbol=row["symbol"],
                        bulk=_format_trace_tpm(row["bulk"]),
                        tumor=_format_trace_tpm(row["tumor"]),
                        fraction=f"{row['fraction']:.0%}",
                        component=row["component"] if row["component"] != "—" else "none estimated",
                        component_tpm=_format_trace_tpm(row["component_tpm"]),
                        reason=row["reason"],
                    )
                )
            )
    history = treatment_history_summary_lines(analysis)
    if history:
        detail.append({"kind": "heading", "text": "Supplied treatment history"})
        detail.extend({"kind": "bullet", "text": line.removeprefix("- ")} for line in history)
    if clinical_context.assays:
        from .clinical_context import clinical_assay_records

        detail.append({"kind": "heading", "text": "Supplied clinical assays"})
        detail.extend(
            paragraph(render_report_paragraph("clinical_assay_record", assay=assay))
            for assay in clinical_assay_records(clinical_context)
        )
    for title, items, formatter in (
        (
            "Notable biomarker outliers",
            _notable_biomarker_outliers(
                ranges_df,
                panel_code,
                panel_subtype,
                excluded_symbols={a["target"] for a in selected_assessments},
            ),
            _format_biomarker_outlier_bullet,
        ),
        ("Notable CTA RNA signals", _notable_cta_outliers(ranges_df), _format_cta_outlier_bullet),
    ):
        if items:
            detail.append({"kind": "heading", "text": title})
            detail.extend(
                {"kind": "bullet", "text": formatter(item).removeprefix("- ")} for item in items
            )
    caveats = _caveats_from_purity_tier(
        report_view.purity.confidence, analysis.get("sample_context"), analysis
    )
    from .report_language import report_plain_text

    conclusion_text = " ".join(report_plain_text(block["text"]) for block in conclusion).casefold()
    caveats = [
        text
        for text in caveats
        if report_plain_text(text).split(": ", 1)[-1].rstrip(". ").casefold() not in conclusion_text
    ]
    if caveats:
        detail.append({"kind": "heading", "text": "Interpretation limits"})
        detail.extend({"kind": "bullet", "text": text} for text in dict.fromkeys(caveats))
    therapy_table = None
    if selected_assessments:
        therapy_table = {
            "columns": [
                ["Target", 15],
                ["Research candidate" if research else "Recommendation", 38],
                ["Estimated tumor TPM (RNA model)", 25],
                ["Eligibility / RNA provenance", 39],
            ],
            "rows": [
                [
                    a["target"],
                    a["agent"] + " · " + a["phase"],
                    a["tumor_band"],
                    " ".join(a["rationale"]) + " Indication: " + a["indication"],
                ]
                for a in selected_assessments
            ],
            "sources": [
                {"label": a["source"], "url": a["source_url"]}
                for a in selected_assessments
                if a["source_url"]
            ],
        }
    return ReportContent(
        sample_id,
        [
            {
                "id": "conclusion",
                "title": "Conclusion and supporting evidence",
                "blocks": conclusion,
            },
            {"id": "therapies", "title": "Therapy rationale and blockers", "blocks": therapies},
            {"id": "information", "title": "Information needed", "blocks": request_blocks},
            {"id": "evidence", "title": "Detailed evidence and figures", "blocks": detail},
        ],
        assessments,
        requests,
        therapy_table,
        [record.public_dict() for record in treatment_records(analysis)],
        clinical_context.public_dict(),
        identity,
    )


def render_report_summary(content: ReportContent) -> str:
    """Render the same authored sections retained in the structured report."""
    return render_report_template(
        "report", sample_id=content.identity.get("title") or content.sample_id,
        sections=content.sections,
    )
