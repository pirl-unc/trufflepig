"""Shared therapy requirements for ranking, explanations and information requests.

A requirement describes the evidence actually available to this run. A known
exclusion is retained as a blocker and cannot become a request for missing data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
from typing import Any

from .reporting import (
    canonical_target_symbol,
    direct_eligibility_evidence_supported,
    hla_eligibility_context,
    indication_biomarker,
    indication_biomarker_label,
    required_protein_changes_for_therapy,
    supplied_variant_context_for_target_row,
    supplied_variant_supports_target_row,
    target_hla_eligibility,
    therapy_row_requires_confirmed_eligibility,
)
from .treatment_history import (
    treatment_history_blocks_row,
    treatment_history_context,
    treatment_history_marks_current,
    treatment_history_supports_review,
)


@dataclass(frozen=True)
class EvidenceRequirement:
    """A specific requirement and its current evidence state."""

    key: str
    kind: str
    status: str  # satisfied, missing, unresolved, blocked
    description: str
    question: str = ""
    accepted_inputs: tuple[str, ...] = ()
    source: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    priority: str = "routine"

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TherapyEligibility:
    """One gate result, reused wherever a therapy is selected or described."""

    requirements: tuple[EvidenceRequirement, ...]
    history_supported: bool
    supplied_variant_supported: bool
    direct_evidence_supported: bool

    @property
    def permits_review(self) -> bool:
        for requirement in self.requirements:
            if requirement.status == "blocked":
                return False
            if requirement.status in {"missing", "unresolved"}:
                # Prior benefit supports review without inventing a confirmed
                # assay. Conflicting evidence, HLA and disease scope retain hard gates.
                if (
                    requirement.status == "unresolved"
                    or not self.history_supported
                    or requirement.kind in {"hla", "scope"}
                ):
                    return False
        return True

    @property
    def has_known_blocker(self) -> bool:
        return any(r.status == "blocked" for r in self.requirements)

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["permits_review"] = self.permits_review
        return result


def therapy_row_in_scope(target_row, analysis, panel_subtype=None) -> bool:
    """Require subtype context for subtype-only rows in an unresolved SARC panel."""
    if not analysis or str(analysis.get("cancer_type") or "").strip() != "SARC":
        return True
    subtype = clean_therapy_value(target_row.get("subtype"))
    if not subtype or subtype == clean_therapy_value(panel_subtype):
        return True
    indication = clean_therapy_value(target_row.get("indication"))
    if re.search(r"\bSTS\b|\bsoft[- ]tissue sarcomas?\b", indication, re.I):
        return True
    return bool(supplied_variant_supports_target_row(target_row, analysis))


def clean_therapy_value(value) -> str:
    """Treat pandas missing cells as absent values at the evidence boundary."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>", "nat"} else text


def msi_mmr_requirement(analysis, *, rna_triage: bool = False) -> EvidenceRequirement:
    """One clinical-assay decision shared by therapy gates and report requests."""
    from .clinical_context import clinical_context_for_analysis, evaluate_msi_mmr
    from .report_language import render_report_paragraph

    decision = evaluate_msi_mmr(clinical_context_for_analysis(analysis))
    status = {"positive": "satisfied", "negative": "blocked", "conflicting": "unresolved",
              "unresolved": "unresolved", "missing": "missing"}[decision.status]
    question = (
        "Supply the existing clinical MSI-PCR, MMR IHC or validated clinical sequencing result, with assay validity, reportability, source and specimen identity."
        if decision.status == "missing" else
        "Reconcile the supplied MSI/MMR reports and their result, validity, reportability and specimen limitations."
    )
    if decision.specimen_id:
        question += f" Required specimen: {decision.specimen_id}."
    return EvidenceRequirement(
        "msi_high", "msi_high", status,
        render_report_paragraph("clinical_assay_decision", decision=decision.public_dict()),
        question,
        ("--clinical-context JSON: assays", "MSI-PCR result", "MMR IHC report", "validated clinical sequencing result"),
        evidence=decision.public_dict(),
        priority="high" if rna_triage or decision.status == "conflicting" else "routine",
    )


def evaluate_therapy_eligibility(
    target_row, analysis=None, *, panel_subtype=None
) -> TherapyEligibility:
    """Evaluate supplied history, disease scope, HLA and molecular requirements.

    This is a report eligibility contract, not a complete treatment-fitness
    assessment. Organ function, disease setting and protocol criteria still need
    clinical review when they have not been supplied.
    """
    requirements = []
    supported_history = treatment_history_supports_review(target_row, analysis)
    variant_match = bool(supplied_variant_supports_target_row(target_row, analysis))
    biomarker = indication_biomarker(target_row)
    direct_match = direct_eligibility_evidence_supported(analysis, biomarker)
    history_text = treatment_history_context(target_row, analysis)
    if treatment_history_blocks_row(target_row, analysis) or treatment_history_marks_current(
        target_row, analysis
    ):
        requirements.append(EvidenceRequirement("history", "history", "blocked", history_text))
    elif history_text:
        requirements.append(EvidenceRequirement("history", "history", "satisfied", history_text))
    if not therapy_row_in_scope(target_row, analysis, panel_subtype):
        requirements.append(
            EvidenceRequirement(
                "disease_scope",
                "scope",
                "blocked",
                "The required disease subtype has not been established for this report.",
            )
        )

    hla = target_hla_eligibility(target_row, analysis=analysis)
    if hla["status"] != "not_hla_restricted":
        status = {
            "matched": "satisfied",
            "mismatched": "blocked",
            "excluded": "blocked",
            "insufficient_resolution": "unresolved",
            "unknown": "missing",
        }[hla["status"]]
        requirements.append(
            EvidenceRequirement(
                "hla_typing",
                "hla",
                status,
                hla_eligibility_context(target_row, analysis=analysis),
                "Supply or reconcile high-resolution HLA typing, including all reported allele fields and annotations.",
                ("HLA typing report", "--hla-types"),
                hla.get("source", ""),
                evidence=hla,
            )
        )

    if biomarker == "msi_high":
        requirements.append(msi_mmr_requirement(analysis))
    elif therapy_row_requires_confirmed_eligibility(target_row):
        gene = canonical_target_symbol(target_row.get("symbol"))
        label = indication_biomarker_label(target_row)
        alleles = required_protein_changes_for_therapy(target_row)
        if alleles:
            label = gene + " " + " or ".join(alleles)
        elif gene and biomarker in {"mutation", "wildtype", "clinical_target_assay"}:
            label = gene + " " + label
        note = clean_therapy_value(target_row.get("eligibility_note"))
        accepted = {
            "mutation": ("clinical variant report", "normalized variant table", "--variants"),
            "wildtype": ("clinical molecular report with assay coverage and negative results",),
            "msi_high": (
                "MSI-PCR result",
                "MMR IHC report",
                "validated clinical sequencing result",
            ),
            "tmb_high": ("clinical TMB result with absolute mutations/Mb and assay method",),
            "clinical_target_assay": ("validated protein or companion-assay report",),
            "imaging": ("required target-imaging report",),
            "histology_only": ("pathology report", "confirmed disease and treatment setting"),
        }.get(biomarker, ("indication-specific clinical report",))
        key = biomarker if biomarker in {"msi_high", "tmb_high"} else f"{biomarker}:{gene}"
        from .variants import variant_evidence_records

        supplied_gene_records = [
            record
            for record in variant_evidence_records(analysis)
            if record.get("gene") == gene or gene in (record.get("genes") or [])
        ]
        if variant_match:
            description = supplied_variant_context_for_target_row(target_row, analysis)
        elif direct_match:
            description = f"Supplied evidence supports the {label} requirement; verify indication-specific criteria."
        elif alleles and supplied_gene_records:
            supplied = "; ".join(
                str(record.get("variant") or "unspecified alteration")
                for record in supplied_gene_records
            )
            description = f"Supplied {gene} evidence ({supplied}) does not establish the required {label} result. The call may be incompatible, imprecise or unsuitable for eligibility; reconcile the clinical assay."
        elif supported_history:
            description = "Prior patient benefit supports review; current eligibility still needs reconciliation."
        else:
            description = (
                f"{label[:1].upper() + label[1:]} evidence has not been confirmed for this therapy."
            )
        requirements.append(
            EvidenceRequirement(
                key,
                biomarker,
                "satisfied"
                if variant_match or direct_match
                else "unresolved"
                if alleles and supplied_gene_records
                else "missing",
                description,
                note or f"Supply the required {label} result.",
                accepted,
                evidence={
                    "required_protein_changes": list(alleles),
                    "supplied_variants": supplied_gene_records,
                    "matched_variants": supplied_variant_supports_target_row(target_row, analysis),
                },
            )
        )
    return TherapyEligibility(tuple(requirements), supported_history, variant_match, direct_match)


def collect_evidence_requests(assessments: list[dict]) -> list[dict]:
    """Combine shared evidence requests, including transitive key/question matches.

    Already blocked treatments retain requirements in their audit assessment
    without generating testing tasks. Merging retains every affected treatment,
    distinct assay specification and reason.
    """
    groups = []
    list_fields = ("keys", "accepted_inputs", "affects", "reasons", "requirements", "details", "evidence")
    for assessment in assessments:
        requirements = assessment.get("eligibility", {}).get("requirements", [])
        if any(r["status"] == "blocked" for r in requirements):
            continue
        for requirement in requirements:
            if requirement["status"] not in {"missing", "unresolved"}:
                continue
            key = requirement["key"]
            signature = (
                requirement["kind"],
                requirement["question"].strip().casefold(),
                tuple(requirement["accepted_inputs"]),
            )
            matches = [
                group
                for group in groups
                if key in group["keys"] or signature in group["question_signatures"]
            ]
            if matches:
                request = matches[0]
                for other in matches[1:]:
                    for field in list_fields:
                        request[field].extend(
                            value for value in other[field] if value not in request[field]
                        )
                    request["question_signatures"].update(other["question_signatures"])
                    if other["status"] == "unresolved":
                        request["status"] = "unresolved"
                    if other["priority"] == "high":
                        request["priority"] = "high"
                    groups.remove(other)
            else:
                request = {
                    "id": "request-" + hashlib.sha256(key.encode()).hexdigest()[:10],
                    "key": key,
                    "kind": requirement["kind"],
                    "status": requirement["status"],
                    "question": requirement["question"],
                    "priority": requirement.get("priority", "routine"),
                    "question_signatures": set(),
                    **{field: [] for field in list_fields},
                }
                groups.append(request)
            request["question_signatures"].add(signature)
            values = {
                "keys": [key],
                "accepted_inputs": requirement["accepted_inputs"],
                "affects": [assessment["agent"]],
                "reasons": [requirement["description"]],
                "requirements": [requirement["question"]],
                "details": [{"question": requirement["question"], "agent": assessment["agent"]}],
                "evidence": [requirement["evidence"]] if requirement.get("evidence") else [],
            }
            for field in list_fields:
                request[field].extend(
                    value for value in values[field] if value and value not in request[field]
                )
            if requirement["status"] == "unresolved":
                request["status"] = "unresolved"
            if requirement.get("priority") == "high":
                request["priority"] = "high"
    for request in groups:
        del request["question_signatures"]
        details = request["details"]
        request["details"] = [
            {
                "question": question,
                "affects": list(dict.fromkeys(
                    detail["agent"] for detail in details if detail["question"] == question
                )),
            }
            for question in request["requirements"]
        ]
    return sorted(groups, key=lambda request: request["priority"] != "high")
