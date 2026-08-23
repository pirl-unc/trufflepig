"""Molecular interpretation for infantile spindle-cell tumor contexts.

These entities overlap morphologically and molecularly.  The helpers in this
module do not classify a tumor from a driver or from RNA expression.  They add
diagnostic-workup context and alteration-gated therapy rows after the report
scope has been selected.
"""

from __future__ import annotations

from functools import lru_cache
import logging

import pandas as pd

from .alterations import (
    alteration_record_genes,
    classify_alteration_type,
    molecular_evidence_for_gene,
)
from .molecular_therapy import NTRK_GENES, ntrk_fusion_therapy_targets, therapy_row


_LOGGER = logging.getLogger(__name__)

INFANTILE_SPINDLE_CODES = frozenset({"SARC_IFS", "CMN", "SARC_NTRK_SPINDLE"})


def _clean(value) -> str:
    text = str(value or "").strip()
    return "" if text.lower() == "nan" else text


def _is_fusion(record) -> bool:
    text = " ".join(
        _clean(record.get(key))
        for key in ("alteration_type", "alteration", "raw_name")
    ).lower()
    return (
        _clean(record.get("alteration_type")).lower() == "fusion"
        or classify_alteration_type(text) == "fusion"
    )


def _is_egfr_kdd(record) -> bool:
    if "EGFR" not in alteration_record_genes(record):
        return False
    text = " ".join(
        _clean(record.get(key))
        for key in ("alteration_type", "alteration", "raw_name")
    ).lower()
    return _clean(record.get("alteration_type")).lower() in {
        "kdd",
        "internal_tandem_duplication",
    } or any(
        term in text
        for term in (
            "kdd",
            "kinase domain duplication",
            "kinase-domain duplication",
            "internal tandem duplication",
            "itd",
        )
    )


def _confirmed_ntrk_fusion_genes(analysis) -> tuple[str, ...]:
    genes: list[str] = []
    for gene in NTRK_GENES:
        if any(
            _is_fusion(record)
            for record in molecular_evidence_for_gene(analysis, gene)
        ):
            if gene not in genes:
                genes.append(gene)
    return tuple(genes)


def _has_confirmed_egfr_kdd(analysis) -> bool:
    return any(
        _is_egfr_kdd(record)
        for record in molecular_evidence_for_gene(analysis, "EGFR")
    )


@lru_cache(maxsize=8)
def _upstream_driver_spectrum(cancer_code: str) -> tuple[dict, ...]:
    """Use oncoref's structured spectrum when the installed release has it."""
    try:
        from oncoref import cancer_driver_spectrum

        result = cancer_driver_spectrum(cancer_code)
        if isinstance(result, pd.DataFrame):
            return tuple(result.to_dict("records"))
        if isinstance(result, (list, tuple)):
            return tuple(dict(row) for row in result if hasattr(row, "get"))
    except (ImportError, AttributeError):
        return ()
    except Exception:
        _LOGGER.warning(
            "Could not load oncoref driver spectrum for %s; using report-safe "
            "fallback guidance",
            cancer_code,
            exc_info=True,
        )
    return ()


def infantile_spindle_context_applies(cancer_code, analysis=None) -> bool:
    """Whether infantile spindle molecular guidance belongs in this report."""
    code = _clean(cancer_code).upper()
    if code in INFANTILE_SPINDLE_CODES:
        return True
    if code != "SARC":
        return False
    return bool(_confirmed_ntrk_fusion_genes(analysis)) or _has_confirmed_egfr_kdd(
        analysis
    )


def infantile_spindle_guidance(cancer_code, analysis=None) -> dict:
    """Return high-level diagnostic and therapeutic guidance for a report.

    Driver overlap is made explicit: ETV6-NTRK3 occurs in both IFS and cellular
    CMN, while EGFR KDD/ITD strongly raises CMN but still requires a renal
    primary and compatible pathology.  No statement here changes report scope.
    """
    code = _clean(cancer_code).upper()
    if not infantile_spindle_context_applies(code, analysis):
        return {}

    ntrk_genes = _confirmed_ntrk_fusion_genes(analysis)
    egfr_kdd = _has_confirmed_egfr_kdd(analysis)
    upstream = {
        candidate: _upstream_driver_spectrum(candidate)
        for candidate in ("SARC_IFS", "CMN")
    }

    if code == "CMN":
        interpretation = (
            "CMN is a kidney-site-qualified pathologic diagnosis. EGFR KDD/ITD is "
            "common in classic CMN and ETV6-NTRK3 is enriched in cellular CMN, "
            "but neither event alone establishes the diagnosis."
        )
    elif code == "SARC_IFS":
        interpretation = (
            "IFS is a heterogeneous infantile MAPK-rearranged spindle-cell tumor, "
            "not an ETV6-NTRK3-only entity. Site, age, morphology, and a complete "
            "molecular assay are needed to separate it from CMN and other spindle tumors."
        )
    elif code == "SARC_NTRK_SPINDLE":
        interpretation = (
            "NTRK-rearranged spindle-cell neoplasm is a molecularly defined diagnosis; "
            "confirm an in-frame NTRK1/2/3 fusion and integrate site and pathology."
        )
    else:
        interpretation = (
            "The supplied kinase event raises an infantile spindle/CMN differential, "
            "but a broad sarcoma RNA call and the event alone do not establish IFS or CMN."
        )

    findings: list[str] = []
    if ntrk_genes:
        findings.append(
            "confirmed supplied NTRK fusion involving " + ", ".join(ntrk_genes)
        )
    if egfr_kdd:
        findings.append("supplied EGFR KDD/ITD")

    return {
        "interpretation": interpretation,
        "findings": findings,
        "workup": (
            "Review renal versus soft-tissue site and pathology, and use RNA/DNA "
            "structural-variant testing that covers NTRK1/2/3 fusions, EGFR KDD/ITD, "
            "BRAF rearrangement/internal deletion, and other kinase fusions. Pan-TRK "
            "IHC can screen but does not replace molecular confirmation."
        ),
        "therapy": (
            "A confirmed NTRK fusion supports an FDA-approved TRK inhibitor; the "
            "strongest prospective IFS evidence is for larotrectinib (94% objective "
            "response within six cycles in COG ADVL1823). EGFR KDD/ITD "
            "has only case-level/off-label TKI evidence in CMN-like disease and should "
            "go to pediatric/rare-tumor molecular review."
        ),
        "upstream_driver_spectrum_available": {
            candidate: bool(rows) for candidate, rows in upstream.items()
        },
    }


def infantile_spindle_guidance_markdown(cancer_code, analysis=None) -> str:
    """Concise report paragraph, ordered from interpretation to action."""
    guidance = infantile_spindle_guidance(cancer_code, analysis)
    if not guidance:
        return ""
    findings = guidance.get("findings") or []
    finding_clause = f" Supplied evidence: {'; '.join(findings)}." if findings else ""
    return (
        "**Infantile spindle-cell molecular context:** "
        + guidance["interpretation"]
        + finding_clause
        + " **Next diagnostic step:** "
        + guidance["workup"]
        + " **Treatment relevance:** "
        + guidance["therapy"]
    )


def infantile_spindle_driver_spectrum_markdown(cancer_code, analysis=None) -> str:
    """Render oncoref's observed driver spectrum for the full report.

    A broad sarcoma carrying a relevant supplied event shows both IFS and CMN
    spectra because the comparison is the useful point.  Exact report scopes
    show only their own spectrum.  Missing older dependency data is an honest
    omission, not a reconstructed frequency table.
    """
    code = _clean(cancer_code).upper()
    if not infantile_spindle_context_applies(code, analysis):
        return ""
    candidates = (code,) if code in {"SARC_IFS", "CMN"} else ("SARC_IFS", "CMN")
    rows: list[dict] = []
    for candidate in candidates:
        for row in _upstream_driver_spectrum(candidate):
            event = _clean(row.get("driver_event"))
            frequency = _clean(row.get("frequency"))
            relationship = _clean(row.get("relationship")).replace("_", " ")
            source = _clean(row.get("evidence_source"))
            if event:
                rows.append(
                    {
                        "entity": candidate,
                        "event": event,
                        "frequency": frequency or "observed",
                        "relationship": relationship or "observed",
                        "source": source or "oncoref molecular provenance",
                    }
                )
    if not rows:
        return ""
    lines = [
        "## Observed driver spectrum",
        "",
        "These are cohort observations, not mutually exclusive treatment rates and "
        "not proof of diagnosis in this sample.",
        "",
        "| Entity | Driver event | Observed frequency | Relationship | Source |",
        "|---|---|---:|---|---|",
    ]
    lines.extend(
        f"| {row['entity']} | {row['event']} | {row['frequency']} | "
        f"{row['relationship']} | {row['source']} |"
        for row in rows
    )
    return "\n".join(lines)


def infantile_spindle_therapy_targets(cancer_code, analysis=None) -> pd.DataFrame:
    """Alteration-gated therapy rows for IFS/CMN/NTRK-spindle contexts.

    Exact entities receive the complete diagnostic-context panel so the full
    report can show what must be tested.  A broad SARC report receives NTRK
    rows only when a supplied NTRK fusion is present; this prevents high NTRK
    RNA from creating a treatment recommendation.
    """
    code = _clean(cancer_code).upper()
    if not infantile_spindle_context_applies(code, analysis):
        return pd.DataFrame()
    confirmed_ntrk = set(_confirmed_ntrk_fusion_genes(analysis))
    exact_context = code in INFANTILE_SPINDLE_CODES
    ntrk_genes = NTRK_GENES if exact_context else tuple(sorted(confirmed_ntrk))
    rows = ntrk_fusion_therapy_targets(
        code,
        ntrk_genes,
        subtype="infantile_spindle_molecular",
        include_ifs_evidence=True,
    ).to_dict("records")
    if code == "CMN":
        rows.append(
            therapy_row(
                code,
                "EGFR",
                "EGFR TKI review (afatinib; osimertinib secondary)",
                subtype="infantile_spindle_molecular",
                phase="off_label",
                treatment_path_tier="off_label",
                line_of_therapy="clinical_trial",
                requires_supplied_alteration=True,
                eligibility_note=(
                    "requires verified EGFR kinase-domain duplication/internal tandem "
                    "duplication; case-level evidence only; molecular tumor board review"
                ),
                indication="EGFR KDD/ITD-positive CMN-like tumor",
                rationale=(
                    "No established CMN TKI standard; limited case and cross-tumor "
                    "evidence supports afatinib as the lead expert-review option, "
                    "with osimertinib as secondary cross-tumor case evidence"
                ),
                source="PMID:41810180; PMID:38821532; PMID:40821607",
            )
        )
    return pd.DataFrame(rows)
