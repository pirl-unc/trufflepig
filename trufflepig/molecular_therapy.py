"""Reusable alteration-gated molecular therapy rows.

These helpers describe treatment possibilities after a cancer context has been
selected.  They never infer an alteration from RNA abundance.
"""

from __future__ import annotations

import pandas as pd


NTRK_GENES = ("NTRK1", "NTRK2", "NTRK3")


def therapy_row(cancer_code: str, gene: str, agent: str, **values) -> dict:
    """Build one normalized locally curated therapy row."""
    return {
        "cancer_code": cancer_code,
        "symbol": gene,
        "ensembl_gene_id": "",
        "role": "target",
        "agent": agent,
        "agent_class": "small_molecule",
        **values,
    }


def ntrk_fusion_therapy_targets(
    cancer_code: str,
    genes=NTRK_GENES,
    *,
    subtype: str,
    include_ifs_evidence: bool = False,
) -> pd.DataFrame:
    """Approved tumor-agnostic TRK options, gated on a supplied fusion."""
    larotrectinib_rationale = "Tumor-agnostic TRK inhibitor for a verified NTRK fusion"
    larotrectinib_source = "FDA label 2025"
    if include_ifs_evidence:
        larotrectinib_rationale += (
            "; prospective pediatric IFS study reported 94% objective response "
            "within six cycles"
        )
        larotrectinib_source += "; PMID:39652801; NCT03834961"
    rows: list[dict] = []
    for gene in genes:
        rows.extend(
            [
                therapy_row(
                    cancer_code,
                    gene,
                    "larotrectinib",
                    subtype=subtype,
                    phase="approved",
                    treatment_path_tier="approved_indication_matched",
                    line_of_therapy="approved_biomarker_matched",
                    requires_supplied_alteration=True,
                    eligibility_note=(
                        "requires a verified in-frame NTRK gene fusion and the "
                        "label-specific advanced/unresectable or morbidity criteria"
                    ),
                    indication="NTRK gene fusion-positive solid tumor",
                    rationale=larotrectinib_rationale,
                    source=larotrectinib_source,
                ),
                therapy_row(
                    cancer_code,
                    gene,
                    "entrectinib",
                    subtype=subtype,
                    phase="approved",
                    treatment_path_tier="approved_indication_matched",
                    line_of_therapy="approved_biomarker_matched",
                    requires_supplied_alteration=True,
                    eligibility_note=(
                        "requires a verified NTRK gene fusion; age >1 month; confirm "
                        "label-specific advanced/unresectable or morbidity criteria"
                    ),
                    indication="NTRK gene fusion-positive solid tumor",
                    rationale=(
                        "Tumor-agnostic TRK/ROS1 inhibitor with a pediatric formulation"
                    ),
                    source="FDA pediatric indication 2023; FDA label 2024",
                ),
                therapy_row(
                    cancer_code,
                    gene,
                    "repotrectinib",
                    subtype=subtype,
                    phase="approved",
                    treatment_path_tier="approved_indication_matched",
                    line_of_therapy="approved_biomarker_matched",
                    requires_supplied_alteration=True,
                    eligibility_note=(
                        "requires a verified NTRK gene fusion and age >=12 years; "
                        "confirm label criteria and prior TRK-inhibitor history"
                    ),
                    indication=(
                        "NTRK gene fusion-positive solid tumor age 12 years or older"
                    ),
                    rationale=(
                        "Next-generation TRK inhibitor with activity in TKI-naive and "
                        "TKI-pretreated disease; pediatric/young-adult study NCT04094610"
                    ),
                    source="FDA accelerated approval 2024; NCT04094610",
                ),
            ]
        )
    return pd.DataFrame(rows)
