"""Build the reader's four report sections directly from evidence decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import hashlib
import json
import math

from .report_language import render_report_paragraph, render_report_template
from .reporting import (
    canonical_target_symbol,
    clinical_maturity_summary,
    expression_independent_indication,
    expression_independent_rna_context,
    normal_expression_context,
    target_observation_state,
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
    observed = None
    if expr is not None:
        value = expr.get("observed_tpm")
        if value is not None and math.isfinite(float(value)):
            observed = float(value)
    state = "measured" if observed is not None else target_observation_state(gene, ranges_df)
    eligibility = evaluate_therapy_eligibility(row, analysis, panel_subtype=panel_subtype)
    rationale = []
    if eligibility.supplied_variant_supported:
        from .reporting import supplied_variant_context_for_target_row

        rationale.append(supplied_variant_context_for_target_row(row, analysis).rstrip(". ") + ".")
    from .reporting import therapy_rationale_paragraphs

    rationale.extend(therapy_rationale_paragraphs(row, analysis=analysis))
    if expression_independent_indication(row):
        from .reporting import expression_independent_interpretation

        rationale.append(expression_independent_interpretation(row))
        rationale.append(expression_independent_rna_context(expr, observation_state=state))
    if expr is not None and tumor_band_available(expr):
        source = tumor_attribution_context(expr)
        normal = normal_expression_context(expr)
        rationale.append(
            "; ".join(
                part for part in (source["label"], source["band"], normal["label"]) if part
            ).rstrip(". ")
            + "."
        )
        notes = list(source.get("notes") or []) + list(normal.get("details") or [])
        if notes:
            rationale.append(notes[0].rstrip(". ") + ".")
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
        "rationale": rationale,
        "observation": {"state": state, "observed_tpm": observed},
        "tumor_band": tumor_band_cell(expr)
        if expr is not None and tumor_band_available(expr)
        else "—",
        "source": clean_therapy_value(row.get("therapy_evidence_source")),
        "source_url": clean_therapy_value(row.get("therapy_evidence_url")),
    }


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
        _empty_therapy_shortlist_message,
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

    cancer_code = report_view.cancer_type
    analysis = {**analysis, "cancer_type": cancer_code, "cancer_name": report_view.cancer_type_name}
    sample_id = _display_sample_id(sample_id) or report_view.sample_id or ""
    prefix = prefix or sample_id or "report"
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

    # Report-level questions join therapy requirements before deduplication.
    from .brief import mismatch_repair_summary_context

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
                "Reconcile the proposed disease and subtype with pathology, clinical history and any disease-defining molecular assay.",
                ("pathology report", "confirmed cancer type", "disease-defining molecular result"),
            )
        )
    if mismatch_repair_summary_context(analysis):
        report_requirements.append(
            EvidenceRequirement(
                "msi_high",
                "msi_high",
                "unresolved",
                "The MMR expression signal is an RNA surrogate, not a clinical MSI/MMR assay.",
                "Confirm MSI/MMR status using MSI-PCR, MMR IHC or validated clinical sequencing before using it for treatment eligibility.",
                ("MSI-PCR result", "MMR IHC report", "validated clinical sequencing result"),
            )
        )
    requests = collect_evidence_requests(
        [
            *assessments,
            {
                "agent": "report conclusion",
                "eligibility": {"requirements": [r.public_dict() for r in report_requirements]},
            },
        ]
    )
    selected_assessments = []
    for recommendation in recommended:
        key = therapy_assessment_id(recommendation.therapy)
        match = next((a for a in assessments if a["id"] == key and a["selected"]), None)
        if match is not None and match not in selected_assessments:
            selected_assessments.append(match)
    if selected_assessments:
        requests.append(
            {
                "id": "request-clinical-setting",
                "key": "clinical_setting",
                "kind": "clinical_setting",
                "status": "unresolved",
                "question": "Reconcile the current disease setting, treatment sequence, response, toxicity and organ function before choosing a treatment. For trial options, verify protocol criteria and current recruitment.",
                "accepted_inputs": [
                    "current oncology assessment",
                    "treatment and toxicity history",
                    "relevant laboratory and imaging results",
                ],
                "affects": [a["agent"] for a in selected_assessments],
                "reasons": [
                    "A report candidate is not confirmation of individual treatment fitness."
                ],
                "requirements": [],
            }
        )

    therapies = [
        paragraph(render_report_paragraph("therapy_scope", scope=panel_code or cancer_code))
    ]
    if panel is None or len(panel) == 0:
        therapies = [
            paragraph(
                f"{cancer_code} is not yet in the curated key-genes panel; no disease-specific therapy shortlist is available."
            )
        ]

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
                    selected=True,
                    indication=assessment["indication"],
                    maturity=assessment["maturity"],
                    request_labels=["see the consolidated Information needed section"],
                )
            )
        )
        therapies.extend(paragraph(text) for text in assessment["rationale"])
    if not selected_assessments:
        therapies.append(paragraph(_empty_therapy_shortlist_message(panel, ranges_df)))
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

    pending = [
        a
        for a in assessments
        if not a["selected"]
        and not any(r["status"] == "blocked" for r in a["eligibility"]["requirements"])
        and any(r["status"] in {"missing", "unresolved"} for r in a["eligibility"]["requirements"])
    ]
    if pending:
        therapies.append({"kind": "heading", "text": "Eligibility pending"})
        for assessment in pending:
            descriptions = list(
                dict.fromkeys(
                    r["description"]
                    for r in assessment["eligibility"]["requirements"]
                    if r["status"] in {"missing", "unresolved"}
                )
            )
            therapies.append(
                {"kind": "bullet", "text": f"**{assessment['agent']}:** " + " ".join(descriptions)}
            )

    request_blocks = []
    for request in requests:
        label = request["kind"].replace("_", " ").capitalize()
        request_blocks.append(
            paragraph(render_report_paragraph("evidence_request", label=label, **request))
        )
        # Different therapies can require distinct tests for one evidence kind.
        # Keep those specifications visible even when the request is deduplicated.
        for requirement in request["requirements"]:
            if requirement != request["question"]:
                request_blocks.append({"kind": "bullet", "text": requirement})
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
    if caveats:
        detail.append({"kind": "heading", "text": "Interpretation limits"})
        detail.extend({"kind": "bullet", "text": text} for text in dict.fromkeys(caveats))
    therapy_table = None
    if selected_assessments:
        therapy_table = {
            "columns": [
                ["Target", 15],
                ["Recommendation", 38],
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
    )


def render_report_summary(content: ReportContent) -> str:
    """Render the same authored sections retained in the structured report."""
    return render_report_template("report", sample_id=content.sample_id, sections=content.sections)
