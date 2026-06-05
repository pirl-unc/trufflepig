"""Generate trufflepig's owned ``therapeutic-agents.csv`` from the curated
protein-target workbook (``protein_target_list.xlsx``).

trufflepig owns the agent/therapy *clinical* layer (#52): one row per
binder (agent), keyed on the target gene symbol, with modality / approval /
trial / provenance metadata. pirlygenes keeps the gene-set / expression
layer; the only shared key is the gene symbol (rename-tolerant).

Usage:
    python scripts/import_therapeutic_agents.py [path/to/protein_target_list.xlsx]

Writes trufflepig/data/therapeutic-agents.csv. Re-run when the workbook is
refreshed; commit the regenerated CSV (it is the source of truth, not the xlsx).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFAULT_XLSX = Path.home() / "Downloads" / "protein_target_list.xlsx"
OUT = Path(__file__).resolve().parent.parent / "trufflepig" / "data" / "therapeutic-agents.csv"

# Normalize the workbook modality vocabulary to trufflepig's controlled set.
_MODALITY_MAP = {
    "ADC": "ADC",
    "CAR-T": "CAR_T",
    "TCR-T": "TCR_T",
    "bispecific": "TCE",
    "fusion_protein": "fusion_protein",
    "mAb": "mAb",
    "peptide": "peptide",
    "radioligand": "RLT",
    "other": "other",
}

_COLUMNS = [
    "agent",
    "aliases",
    "target_gene",
    "modality",
    "modality_detail",
    "sponsor",
    "development_stage",
    "highest_phase",
    "fda_approved",
    "approval_year",
    "approved_indication",
    "brand_name",
    "indications",
    "num_trials",
    "key_trials",
    "key_pmids",
    "notes",
]


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _year(value) -> str:
    text = _clean(value)
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def main() -> None:
    xlsx = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XLSX
    binders = pd.ExcelFile(xlsx).parse("Binders")

    rows = []
    for _, b in binders.iterrows():
        agent = _clean(b.get("Binder Name"))
        target_raw = _clean(b.get("Target Gene Symbol"))
        # Skip the placeholder rows for targets with no curated binder yet.
        if not agent or agent == "(unaddressed)" or not target_raw:
            continue
        # Normalize a parenthetical variant annotation (e.g. "EGFR (variant III)")
        # to the base HGNC symbol so the gene stays expression-rankable; keep the
        # variant qualifier in the modality detail.
        target = target_raw
        variant_note = ""
        if "(" in target_raw:
            target = target_raw.split("(", 1)[0].strip()
            variant_note = target_raw.split("(", 1)[1].rstrip(") ").strip()
        modality_raw = _clean(b.get("Modality"))
        modality_detail = _clean(b.get("Modality Notes"))
        if variant_note:
            modality_detail = (
                f"{target_raw}-specific; {modality_detail}".rstrip("; ")
                if modality_detail
                else f"{target_raw}-specific"
            )
        rows.append(
            {
                "agent": agent,
                "aliases": _clean(b.get("Synonyms")),
                "target_gene": target,
                "modality": _MODALITY_MAP.get(modality_raw, modality_raw or "other"),
                "modality_detail": modality_detail,
                "sponsor": _clean(b.get("Sponsor")),
                "development_stage": _clean(b.get("Development Stage")),
                "highest_phase": _clean(b.get("Highest Phase")),
                "fda_approved": "yes"
                if _clean(b.get("FDA Approved?")).lower() == "yes"
                else "no",
                "approval_year": _year(b.get("Approval Year")),
                "approved_indication": _clean(b.get("Approved Indication")),
                "brand_name": _clean(b.get("Brand Name")),
                "indications": _clean(b.get("Indications")),
                "num_trials": _clean(b.get("# Key Trials")),
                "key_trials": _clean(b.get("Key Trials Summary")),
                "key_pmids": _clean(b.get("Key PMIDs")),
                "notes": _clean(b.get("Notes")),
            }
        )

    df = pd.DataFrame(rows, columns=_COLUMNS)
    df = df.sort_values(["target_gene", "highest_phase", "agent"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} agent rows across {df['target_gene'].nunique()} targets -> {OUT}")


if __name__ == "__main__":
    main()
