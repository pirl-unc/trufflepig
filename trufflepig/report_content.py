"""Build the reader's four report sections directly from evidence decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any
import hashlib
import json
import re

from . import brief
from .report_language import (
    markdown_url,
    render_report_paragraph,
    render_report_template,
    report_literal,
    report_plain_text,
)
from .reporting import (
    canonical_target_symbol,
    clinical_maturity_summary,
    expression_independent_indication,
    expression_independent_rna_context,
    normal_expression_context,
    target_rna_observation,
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
    evaluate_therapy_eligibility,
    msi_mmr_requirement,
)
from .treatment_history import treatment_history_summary_lines, treatment_records
from .report_takeaways import sample_takeaways, therapeutic_lead_basis


@dataclass
class ReportContent:
    """Serializable prose and evidence, authored once before format rendering."""

    sample_id: str
    sections: list[dict[str, Any]]
    therapy_assessments: list[dict[str, Any]]
    evidence_requests: list[dict[str, Any]]
    treatment_history: list[dict[str, Any]]
    clinical_context: dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def paragraph(text: str) -> dict:
    return {"kind": "paragraph", "text": text}


def therapy_assessment_id(row) -> str:
    """Stable identity for one complete curated row, including disease scope."""
    facts = {str(key): clean_therapy_value(value) for key, value in row.items()}
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()[:16]


def therapy_evidence_sources(row) -> list[dict[str, str]]:
    """Retain curated citations, linking only recognized complete identifiers."""
    label = clean_therapy_value(row.get("therapy_evidence_source"))
    url = clean_therapy_value(row.get("therapy_evidence_url"))
    if label or url:
        return [{"label": label or "Source", "url": url}]
    sources = []
    for item in clean_therapy_value(row.get("source")).split(";"):
        item = item.strip()
        if not item:
            continue
        pmid = re.fullmatch(r"PMID:\s*(\d+)", item, re.IGNORECASE)
        if pmid:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid.group(1)}/"
        elif re.fullmatch(r"NCT\d{8}", item, re.IGNORECASE):
            url = f"https://clinicaltrials.gov/study/{item.upper()}"
        elif item.startswith(("https://", "http://")):
            url = item
        else:
            url = ""
        source = {"label": item, "url": url}
        if source not in sources:
            sources.append(source)
    return sources


def _evidence_source_cell(assessment) -> str:
    return " · ".join(
        f"[{report_literal(source['label'])}]({markdown_url(source['url'])})"
        if source["url"] else report_literal(source["label"])
        for source in assessment["sources"]
    ) or "Not supplied"


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
    eligibility=None,
) -> dict:
    """Explain one curated treatment with its identity, requirements and RNA evidence.

    This function does not select a treatment. Call ``recommend_therapies`` for
    ranking; use this assessment to inspect selected and excluded alternatives.
    """
    agent = brief.therapy_agent_label(row)
    identity = resolve_therapy_identity(row.get("agent"))
    gene = canonical_target_symbol(row.get("symbol"))
    observation = target_rna_observation(expr, symbol=gene, ranges_df=ranges_df)
    state, observed = observation["state"], observation["observed_tpm"]
    if eligibility is None:
        eligibility = evaluate_therapy_eligibility(row, analysis, panel_subtype=panel_subtype)
    rationale = []
    source, normal = {}, {}
    rationale.extend(r.description for r in eligibility.requirements if r.kind == "msi_high")
    if eligibility.supplied_variant_supported:
        from .reporting import supplied_variant_context_for_target_row

        rationale.append(supplied_variant_context_for_target_row(row, analysis).rstrip(". ") + ".")
    from .reporting import therapy_rationale_paragraphs

    rationale.extend(therapy_rationale_paragraphs(row, analysis=analysis))
    reader_rationale = list(rationale)
    if expression_independent_indication(row):
        rationale.append(expression_independent_rna_context(expr, observation_state=state))
        reader_rationale.append("Target RNA is context only; target expression is not the eligibility criterion.")
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
        reader_rationale.append(
            f"{gene}: {source['band']}. Source: {source['label']}; "
            f"healthy-tissue context: {normal['label']}."
            + (" Attribution limits: " + "; ".join(notes) + "." if notes else "")
        )
    elif not expression_independent_indication(row):
        rationale.append(
            render_report_paragraph(
                "rna_observation", state=state, observed_tpm=observed, context_only=False
            )
        )
        reader_rationale.append(rationale[-1])
    caution = therapy_state_caution(row, analysis=analysis, disease_state=disease_state)
    if caution:
        rationale.append("Current-treatment context: " + caution.rstrip(". ") + ".")
        reader_rationale.append(rationale[-1])
    sources = therapy_evidence_sources(row)
    return {
        "id": therapy_assessment_id(row),
        "identity": asdict(identity),
        "agent": agent,
        "canonical_agent": identity.canonical_name,
        "target": gene or "Clinical pathway",
        "selected": selected,
        "phase": brief.phase_label(clean_therapy_value(row.get("phase"))),
        "indication": clean_therapy_value(row.get("indication")),
        "maturity": clinical_maturity_summary(row, target_panel=target_panel),
        "eligibility": eligibility.public_dict(),
        "rationale": rationale,
        "reader_rationale": reader_rationale,
        "observation": observation,
        "rna_source": source,
        "normal_context": normal,
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
        "sources": sources,
        "source": "; ".join(source["label"] for source in sources),
        "source_url": sources[0]["url"] if len(sources) == 1 else "",
    }


def therapy_source_markdown(assessment: dict) -> str:
    """Citation cell for one therapy assessment, linked when a source URL is curated."""
    if assessment.get("sources"):
        return _evidence_source_cell(assessment)
    label = report_literal(assessment.get("source") or "")
    url = markdown_url(assessment.get("source_url"))
    if url:
        return f"[{label or 'Evidence source'}]({url})"
    return label or "—"


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
    from .common import ranges_by_symbol, ranges_by_gene_id, panel_symbols_to_gene_ids

    cancer_code = report_view.cancer_type
    analysis = {**analysis, "cancer_type": cancer_code, "cancer_name": report_view.cancer_type_name}
    sample_id = brief.display_sample_id(sample_id) or report_view.sample_id or ""
    prefix = prefix or sample_id or "report"
    takeaways = sample_takeaways(analysis, ranges_df, report_view, disease_state)
    conclusion = [block for block in takeaways if not block.get("detail")]
    supporting_context = [block for block in takeaways if block.get("detail")]
    panel_code, panel_subtype, panel = brief.curated_target_panel_for_sample(
        cancer_code,
        analysis,
        ranges_df=ranges_df,
    )
    eligibility_by_row: dict[str, Any] = {}

    def eligibility_for(row):
        # Selection and assessment read the same rows; evaluate each row once.
        key = therapy_assessment_id(row)
        if key not in eligibility_by_row:
            eligibility_by_row[key] = evaluate_therapy_eligibility(
                row, analysis, panel_subtype=panel_subtype
            )
        return eligibility_by_row[key]

    recommended = brief.recommend_therapies(
        panel,
        ranges_df,
        analysis=analysis,
        disease_state=disease_state,
        panel_subtype=panel_subtype,
        eligibility_for=eligibility_for,
    )
    selected = {therapy_assessment_id(row.therapy) for row in recommended}
    expression = ranges_by_symbol(ranges_df) if ranges_df is not None else {}
    gene_expression = ranges_by_gene_id(ranges_df) if ranges_df is not None else {}
    assessments = []
    rows = panel.to_dict("records") if panel is not None else []
    gene_ids = panel_symbols_to_gene_ids(canonical_target_symbol(row.get("symbol")) for row in rows)
    for row in rows:
        agent = brief.therapy_agent_label(row)
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
            eligibility=eligibility_for(row),
        )
        assessments.append(assessment)

    selected_assessments = []
    for recommendation in recommended:
        key = therapy_assessment_id(recommendation.therapy)
        match = next((a for a in assessments if a["id"] == key and a["selected"]), None)
        if match is not None and match not in selected_assessments:
            selected_assessments.append(match)

    # Report-level questions join therapy requirements before deduplication.
    from .clinical_context import clinical_context_for_analysis, evaluate_msi_mmr

    clinical_context = clinical_context_for_analysis(analysis)
    clinical_mmr = evaluate_msi_mmr(clinical_context)
    rna_mmr = brief.mismatch_repair_summary_context(analysis)
    rna_state = brief.mismatch_repair_rna_state(analysis)
    if clinical_context.assays:
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
                label="Spindle-cell diagnosis",
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
                "Reconcile the proposed disease and subtype with pathology, clinical history and any disease-defining molecular assay.",
                ("pathology report", "confirmed cancer type", "disease-defining molecular result"),
                label="Diagnosis",
            )
        )
    if rna_mmr or clinical_context.assays:
        report_requirements.append(msi_mmr_requirement(analysis, rna_triage=rna_state == "MSI-like"))
    clinical_requirements = []
    if selected_assessments:
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
            label="Clinical setting",
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
        paragraph(render_report_paragraph("therapy_scope", scope=panel_code or cancer_code))
    ]
    identity_unresolved = bool(analysis.get("cancer_type_abstention"))
    if identity_unresolved:
        therapies = [paragraph(
            "Cancer type remains unresolved. Disease-specific therapies are withheld "
            "until pathology or a disease-defining molecular result establishes their scope."
        )]
    elif panel is None or len(panel) == 0:
        therapies = [
            paragraph(
                f"{cancer_code} is not yet in the curated key-genes panel; no disease-specific therapy shortlist is available."
            )
        ]

    if spindle_guidance:
        therapies.append(paragraph(spindle_guidance["therapy"]))

    leads = [
        (assessment, therapeutic_lead_basis(assessment, analysis))
        for assessment in assessments if not assessment["selected"]
    ]
    leads = [(assessment, basis) for assessment, basis in leads if basis]
    leads.sort(key=lambda item: not any(r["kind"] == "msi_high" for r in item[0]["eligibility"]["requirements"]))

    if selected_assessments:
        therapies.append({
            "kind": "table",
            "headers": [
                "Target",
                "Recommendation",
                "Estimated tumor TPM (RNA model)",
                "Evidence source",
            ],
            "rows": [
                [
                    report_literal(a["target"]),
                    report_literal(f"{a['agent']} · {a['phase']}"),
                    report_literal(a["tumor_band"]),
                    therapy_source_markdown(a),
                ]
                for a in selected_assessments
            ],
        })

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
        therapies.extend(paragraph(text) for text in assessment["reader_rationale"])
    if leads:
        therapies.append({"kind": "heading", "text": "Vulnerabilities worth investigating"})
        grouped = {}
        for assessment, basis in leads:
            vulnerability = "MSI/MMR" if any(r["kind"] == "msi_high" for r in assessment["eligibility"]["requirements"]) else assessment["target"]
            grouped.setdefault((vulnerability, basis), []).append(assessment)
        for (target, basis), group in grouped.items():
            options = "; ".join(
                f"{report_literal(a['agent'])} ({report_literal(a['phase'])}; {therapy_source_markdown(a)})"
                for a in group
            )
            requirements = list(dict.fromkeys(
                r["label"] or r["kind"].replace("_", " ")
                for a in group for r in a["eligibility"]["requirements"]
                if r["status"] in {"missing", "unresolved"}
            ))
            therapies.append({
                "kind": "bullet",
                "text": f"**{report_literal(target)}:** {basis} Options to investigate: {options}. "
                + ("Confirm: " + "; ".join(report_literal(r) for r in requirements) + "." if requirements else ""),
                "figure_suffixes": [] if target == "MSI/MMR" else ["priority-target-context.png"],
            })
    if assessments and not identity_unresolved:
        therapies.append({
            "kind": "paragraph",
            "text": "The target figures compare RNA support, estimated source and healthy-tissue expression. "
            "Their scores prioritize follow-up; they do not measure drug response or establish eligibility.",
            "figure_suffixes": ["priority-targets.png", "priority-target-context.png", "actionable-targets.png"],
        })
    if not selected_assessments and not identity_unresolved:
        unmet = any(
            r["status"] in {"blocked", "missing", "unresolved"}
            for a in assessments for r in a["eligibility"]["requirements"]
        )
        therapies.append(paragraph(
            "No therapy meets the current shortlisting criteria. The confirmation steps below may change that."
            if unmet else brief.empty_therapy_shortlist_message(panel, ranges_df)
        ))
    blocked = [
        a
        for a in assessments
        if any(r["status"] == "blocked" for r in a["eligibility"]["requirements"])
    ]
    if blocked:
        therapies.append({"kind": "heading", "text": "Known blockers and current treatments"})
        for assessment in blocked:
            descriptions = [
                r["description"]
                for r in assessment["eligibility"]["requirements"]
                if r["status"] == "blocked"
            ]
            therapies.append(
                {"kind": "bullet", "text": f"**{assessment['agent']}:** " + " ".join(descriptions)}
            )

    request_blocks = []
    citations_by_agent = {}
    for assessment in assessments:
        citations_by_agent.setdefault(assessment["agent"], [])
        citation = therapy_source_markdown(assessment)
        if citation != "—" and citation not in citations_by_agent[assessment["agent"]]:
            citations_by_agent[assessment["agent"]].append(citation)

    def affected_options(agents):
        return "; ".join(
            report_literal(agent)
            + (" (" + "; ".join(citations_by_agent.get(agent, [])) + ")" if citations_by_agent.get(agent) else "")
            for agent in agents
        )

    for request in requests:
        # Missing data can be summarized by the requested test; conflicting
        # supplied evidence must retain the actual mismatch, not just the test name.
        for description in dict.fromkeys(
            r["description"] for a in assessments for r in a["eligibility"]["requirements"]
            if r["key"] in request["keys"] and r.get("evidence")
            and (r["status"] == "unresolved" or r["kind"] == "hla")
        ):
            request_blocks.append(paragraph(description))
        request_blocks.append(paragraph(render_report_paragraph(
            "evidence_request", **request,
            affected_options=affected_options(request["details"][0]["affects"]),
        )))
        sources = list(dict.fromkeys(
            r["source"] for a in assessments for r in a["eligibility"]["requirements"]
            if r["key"] in request["keys"] and r.get("source", "").startswith(("https://", "http://"))
        ))
        if sources:
            request_blocks[-1]["text"] += " " + " · ".join(
                f"[Eligibility source]({markdown_url(source)})" for source in sources
            ) + "."
        # Different therapies can require distinct tests for one evidence kind.
        # Keep those specifications visible even when the request is deduplicated.
        for detail in request["details"]:
            if detail["question"] != request["question"]:
                request_blocks.append({
                    "kind": "bullet",
                    "text": render_report_paragraph(
                        "evidence_request_detail", **detail,
                        affected_options=affected_options(detail["affects"]),
                    ),
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
    if supporting_context:
        detail.append({"kind": "heading", "text": "Additional sample context"})
        detail.extend(supporting_context)
    source_rows = brief.source_attribution_rows(panel, ranges_df, recommended)
    if source_rows:
        detail.append({"kind": "heading", "text": "Where target RNA signal appears to come from"})
        for row in source_rows:
            detail.append(
                paragraph(
                    render_report_paragraph(
                        "target_attribution",
                        symbol=row["symbol"],
                        bulk=brief.format_trace_tpm(row["bulk"]),
                        tumor=brief.format_trace_tpm(row["tumor"]),
                        fraction=f"{row['fraction']:.0%}",
                        component=row["component"] if row["component"] != "—" else "none estimated",
                        component_tpm=brief.format_trace_tpm(row["component_tpm"]),
                        reason=row["reason"],
                    )
                )
            )
    history = treatment_history_summary_lines(analysis)
    if history:
        detail.append({"kind": "heading", "text": "Supplied treatment history"})
        detail.extend({"kind": "bullet", "text": line.removeprefix("- ")} for line in history)
    if clinical_context.assays:
        detail.append({"kind": "heading", "text": "Supplied clinical assays"})
        detail.extend(
            paragraph(render_report_paragraph("clinical_assay_record", assay=assay))
            for assay in clinical_mmr.assays
        )
    if identity_unresolved:
        detail.append(paragraph(
            "Tumor-attributed TPM and cohort-relative pathway results are conditional "
            "on an exploratory reference model. They do not establish tumor-cell origin; "
            "the observed bulk RNA measurements remain available for review."
        ))
    for title, items, formatter in (
        (
            "Notable biomarker outliers",
            brief.notable_biomarker_outliers(
                ranges_df,
                panel_code,
                panel_subtype,
                excluded_symbols={a["target"] for a in selected_assessments},
            ),
            brief.format_biomarker_outlier_bullet,
        ),
        ("Notable CTA RNA signals", brief.notable_cta_outliers(ranges_df), brief.format_cta_outlier_bullet),
    ):
        if items:
            detail.append({"kind": "heading", "text": title})
            detail.extend(
                {"kind": "bullet", "text": formatter(item).removeprefix("- ")} for item in items
            )
    caveats = brief.caveats_from_purity_tier(
        report_view.purity.confidence, analysis.get("sample_context"), analysis
    )
    conclusion_text = " ".join(report_plain_text(block["text"]) for block in conclusion).casefold()
    caveats = [
        text
        for text in caveats
        if report_plain_text(text).split(": ", 1)[-1].rstrip(". ").casefold() not in conclusion_text
    ]
    if caveats:
        detail.append({"kind": "heading", "text": "Interpretation limits"})
        detail.extend({"kind": "bullet", "text": text} for text in dict.fromkeys(caveats))
    return ReportContent(
        sample_id,
        [
            {
                "id": "conclusion",
                "title": "What we learned about this sample",
                "blocks": conclusion,
            },
            {"id": "therapies", "title": "Therapeutic directions", "blocks": therapies},
            {"id": "information", "title": "What would change the treatment options", "blocks": request_blocks},
            {"id": "evidence", "title": "Detailed evidence and figures", "blocks": detail},
        ],
        assessments,
        requests,
        [record.public_dict() for record in treatment_records(analysis)],
        clinical_context.public_dict(),
    )


def render_report_summary(content: ReportContent) -> str:
    """Render the same authored sections retained in the structured report."""
    return render_report_template("report", sample_id=content.sample_id, sections=content.sections)
