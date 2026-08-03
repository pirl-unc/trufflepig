"""Exact therapy panels for molecularly defined spindle-pattern sarcomas."""

from __future__ import annotations

import pandas as pd

from .molecular_therapy import NTRK_GENES, ntrk_fusion_therapy_targets, therapy_row


EXACT_SARCOMA_THERAPY_CODES = frozenset({"SARC_IMT", "SARC_DFSP", "SARC_PEC"})


def sarcoma_subtype_therapy_targets(cancer_code) -> pd.DataFrame:
    """Return a diagnosis-specific panel without borrowing sibling therapies."""
    code = str(cancer_code or "").strip().upper()
    subtype = "exact_sarcoma_molecular"
    rows: list[dict] = []

    if code == "SARC_IMT":
        rows.append(
            therapy_row(
                code,
                "ALK",
                "crizotinib",
                subtype=subtype,
                phase="approved",
                treatment_path_tier="approved_indication_matched",
                line_of_therapy="approved_biomarker_matched",
                indication_biomarker="mutation",
                requires_supplied_alteration=True,
                eligibility_note=(
                    "requires pathologically confirmed unresectable, recurrent, or "
                    "refractory IMT and verified ALK-positive disease; FDA label covers "
                    "adults and children age >=1 year"
                ),
                indication=(
                    "ALK-positive unresectable, recurrent, or refractory inflammatory "
                    "myofibroblastic tumor"
                ),
                rationale="FDA-approved ALK inhibitor for ALK-positive IMT",
                source="FDA Xalkori label 2023",
            )
        )
        rows.extend(
            ntrk_fusion_therapy_targets(
                code,
                NTRK_GENES,
                subtype=subtype,
            ).to_dict("records")
        )
    elif code == "SARC_DFSP":
        rows.append(
            therapy_row(
                code,
                "PDGFB",
                "imatinib",
                subtype=subtype,
                phase="approved",
                treatment_path_tier="approved_indication_matched",
                line_of_therapy="approved_biomarker_matched",
                indication_biomarker="mutation",
                requires_supplied_alteration=True,
                eligibility_note=(
                    "requires pathologically confirmed adult unresectable, recurrent, "
                    "or metastatic DFSP; confirm COL1A1-PDGFB or another activating "
                    "PDGFB rearrangement when molecular evidence is available"
                ),
                indication="adult unresectable, recurrent, or metastatic DFSP",
                rationale=(
                    "FDA-approved PDGFR inhibitor for advanced DFSP; the canonical "
                    "COL1A1-PDGFB fusion provides the molecular rationale"
                ),
                source="FDA Gleevec label 2024",
            )
        )
    elif code == "SARC_PEC":
        rows.append(
            therapy_row(
                code,
                "",
                "nab-sirolimus (Fyarro)",
                subtype=subtype,
                agent_class="mTOR_inhibitor",
                phase="approved",
                treatment_path_tier="approved_indication_matched",
                line_of_therapy="approved_histology_matched",
                requires_supplied_alteration=False,
                eligibility_note=(
                    "requires pathologically confirmed malignant PEComa in an adult "
                    "with locally advanced unresectable or metastatic disease; TSC1, "
                    "TSC2, or other mTOR-pathway alterations are contextual, not label "
                    "requirements"
                ),
                indication=(
                    "adult locally advanced unresectable or metastatic malignant PEComa"
                ),
                rationale="FDA-approved mTOR inhibitor for malignant PEComa",
                source="FDA Fyarro approval package 2021",
            )
        )

    return pd.DataFrame(rows)


def sarcoma_subtype_guidance_markdown(cancer_code) -> str:
    """Concise diagnostic context for the three exact therapy panels."""
    code = str(cancer_code or "").strip().upper()
    if code == "SARC_IMT":
        return (
            "**IMT molecular context:** Confirm ALK status with an orthogonal fusion "
            "assay before using the approved crizotinib pathway. If ALK-negative, "
            "complete RNA structural-variant testing should cover NTRK1/2/3 and other "
            "kinase fusions; target RNA abundance alone is not eligibility evidence."
        )
    if code == "SARC_DFSP":
        return (
            "**DFSP molecular context:** Pathology and the characteristic "
            "COL1A1-PDGFB or another activating PDGFB rearrangement support the "
            "imatinib pathway in adult unresectable, recurrent, or metastatic disease; "
            "PDGFB RNA abundance alone is not eligibility evidence."
        )
    if code == "SARC_PEC":
        return (
            "**PEComa treatment context:** Nab-sirolimus is an approved histology-based "
            "option for adults with pathologically confirmed locally advanced "
            "unresectable or metastatic malignant PEComa. TSC1/TSC2 and mTOR-pathway "
            "findings are biologically informative but are not required by the label."
        )
    return ""
