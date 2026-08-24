"""Report guidance for molecularly defined spindle-pattern sarcomas."""

from __future__ import annotations

def sarcoma_subtype_guidance_markdown(cancer_code) -> str:
    """Concise diagnostic context for the three exact therapy panels."""
    code = str(cancer_code or "").strip().upper()
    if code == "SARC_IMT":
        return (
            "**IMT molecular context:** Confirm ALK-positive status with validated "
            "ALK IHC or an orthogonal molecular method such as FISH before using the "
            "approved crizotinib pathway. If ALK-negative, complete RNA structural-"
            "variant testing should cover NTRK1/2/3 and other kinase fusions; target "
            "RNA abundance alone is not eligibility evidence."
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
