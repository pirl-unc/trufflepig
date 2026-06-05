"""Trufflepig-owned therapeutic-agent registry (#52).

trufflepig owns the agent/therapy *clinical* layer — one row per binder
(agent), keyed on the target gene symbol, with modality / approval / trial /
provenance metadata curated from ``protein_target_list.xlsx`` (see
``scripts/import_therapeutic_agents.py``). pirlygenes keeps the gene-set /
expression layer; the only shared key is the gene symbol, which makes the
join taxonomy-rename-tolerant.

This decouples the clinically-critical therapy layer (drug-approval cadence)
from pirlygenes' gene-set cadence and code-rename churn.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from ._data import TRUFFLEPIG_DATA_DIR

_CSV = TRUFFLEPIG_DATA_DIR / "therapeutic-agents.csv"

# Controlled modality vocabulary + reader-facing labels.
MODALITY_LABELS = {
    "ADC": "antibody-drug conjugate",
    "RLT": "radioligand therapy",
    "TCE": "T-cell engager / bispecific",
    "CAR_T": "CAR-T cell therapy",
    "CAR_NK": "CAR-NK cell therapy",
    "TCR_T": "TCR-T cell therapy",
    "vaccine": "therapeutic vaccine",
    "mAb": "monoclonal antibody",
    "small_molecule": "small molecule",
    "macrocycle": "macrocycle",
    "degrader": "targeted degrader",
    "immunotoxin": "immunotoxin",
    "oncolytic": "oncolytic",
    "peptide": "peptide",
    "fusion_protein": "fusion protein",
    "other": "other modality",
}

MODALITIES = frozenset(MODALITY_LABELS)


@dataclass(frozen=True)
class TherapeuticAgent:
    agent: str
    target_gene: str
    modality: str
    modality_detail: str
    aliases: str
    sponsor: str
    development_stage: str
    highest_phase: str
    fda_approved: bool
    approval_year: str
    approved_indication: str
    brand_name: str
    indications: str
    num_trials: str
    key_trials: str
    key_pmids: str
    notes: str

    @property
    def modality_label(self) -> str:
        return MODALITY_LABELS.get(self.modality, self.modality or "agent")

    def approval_clause(self) -> str:
        """Short reader-facing approval/stage clause."""
        if self.fda_approved:
            year = f" {self.approval_year}" if self.approval_year else ""
            brand = f" ({self.brand_name})" if self.brand_name else ""
            return f"FDA-approved{year}{brand}"
        stage = (self.highest_phase or self.development_stage or "").replace("_", " ")
        return stage or "investigational"


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


@lru_cache(maxsize=1)
def therapeutic_agents() -> pd.DataFrame:
    """The full therapeutic-agent registry as a DataFrame (string columns)."""
    df = pd.read_csv(_CSV, dtype=str).fillna("")
    df["target_gene"] = df["target_gene"].str.strip()
    df["agent"] = df["agent"].str.strip()
    return df


@lru_cache(maxsize=1)
def _agents_by_gene() -> dict[str, tuple[TherapeuticAgent, ...]]:
    out: dict[str, list[TherapeuticAgent]] = {}
    for _, row in therapeutic_agents().iterrows():
        gene = _clean(row.get("target_gene"))
        if not gene:
            continue
        agent = TherapeuticAgent(
            agent=_clean(row.get("agent")),
            target_gene=gene,
            modality=_clean(row.get("modality")) or "other",
            modality_detail=_clean(row.get("modality_detail")),
            aliases=_clean(row.get("aliases")),
            sponsor=_clean(row.get("sponsor")),
            development_stage=_clean(row.get("development_stage")),
            highest_phase=_clean(row.get("highest_phase")),
            fda_approved=_clean(row.get("fda_approved")).lower() == "yes",
            approval_year=_clean(row.get("approval_year")),
            approved_indication=_clean(row.get("approved_indication")),
            brand_name=_clean(row.get("brand_name")),
            indications=_clean(row.get("indications")),
            num_trials=_clean(row.get("num_trials")),
            key_trials=_clean(row.get("key_trials")),
            key_pmids=_clean(row.get("key_pmids")),
            notes=_clean(row.get("notes")),
        )
        out.setdefault(gene, []).append(agent)
    # Most-advanced agents first: approved, then by phase, then name.
    _phase_rank = {
        "approved": 0,
        "phase_3": 1,
        "phase_2": 2,
        "phase_1": 3,
        "preclinical": 4,
        "none": 5,
        "": 6,
    }
    return {
        gene: tuple(
            sorted(
                agents,
                key=lambda a: (
                    0 if a.fda_approved else 1,
                    _phase_rank.get(a.highest_phase, 6),
                    a.agent.lower(),
                ),
            )
        )
        for gene, agents in out.items()
    }


def druggable_target_genes() -> frozenset[str]:
    """Gene symbols with at least one curated binder/agent."""
    return frozenset(_agents_by_gene())


def is_druggable_target(symbol: str | None) -> bool:
    return bool(symbol) and str(symbol).strip() in _agents_by_gene()


def agents_for_target(symbol: str | None) -> tuple[TherapeuticAgent, ...]:
    """Curated agents for a target gene, most-advanced first."""
    if not symbol:
        return ()
    return _agents_by_gene().get(str(symbol).strip(), ())


def best_agent_for_target(symbol: str | None) -> TherapeuticAgent | None:
    agents = agents_for_target(symbol)
    return agents[0] if agents else None


def target_agent_summary(symbol: str | None) -> str:
    """One-line reader-facing summary for a druggable target, e.g.
    ``"DLL3: tarlatamab-dlle (T-cell engager / bispecific, FDA-approved 2024) +2 more"``.
    Empty string if the target has no curated binder."""
    agents = agents_for_target(symbol)
    if not agents:
        return ""
    best = agents[0]
    extra = f" +{len(agents) - 1} more" if len(agents) > 1 else ""
    return (
        f"{best.agent} ({best.modality_label}, {best.approval_clause()})"
        f"{extra}"
    )
