"""Structured patient treatment evidence for therapy prioritization.

The RNA model can nominate targets, but it cannot recover whether a patient
already benefited from, progressed on, or could not tolerate a treatment.
This module gives that clinical evidence a small, explicit contract and keeps
matching/ranking semantics in one place so every report surface can use the
same precedence rules.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping


TREATMENT_STATUSES = (
    "major_benefit",
    "benefit",
    "stable_disease",
    "current",
    "no_benefit",
    "progression",
    "intolerance",
    "contraindicated",
)

_STATUS_ALIASES = {
    "major response": "major_benefit",
    "major benefit": "major_benefit",
    "very effective": "major_benefit",
    "complete response": "major_benefit",
    "partial response": "benefit",
    "response": "benefit",
    "clinical benefit": "benefit",
    "effective": "benefit",
    "benefit": "benefit",
    "stable disease": "stable_disease",
    "disease control": "stable_disease",
    "current therapy": "current",
    "ongoing": "current",
    "current": "current",
    "no response": "no_benefit",
    "ineffective": "no_benefit",
    "no benefit": "no_benefit",
    "progressive disease": "progression",
    "progressed": "progression",
    "progression": "progression",
    "not tolerated": "intolerance",
    "toxicity": "intolerance",
    "intolerant": "intolerance",
    "intolerance": "intolerance",
    "contraindication": "contraindicated",
    "contraindicated": "contraindicated",
}

_MODALITY_ALIASES = {
    "adc": "ADC",
    "antibody drug conjugate": "ADC",
    "antibody-drug conjugate": "ADC",
    "rlt": "RLT",
    "rpt": "RLT",
    "radioligand": "RLT",
    "radioligand therapy": "RLT",
    "radiopharmaceutical": "RLT",
    "radiopharmaceutical therapy": "RLT",
    "tce": "TCE",
    "bispecific": "TCE",
    "t cell engager": "TCE",
    "t-cell engager": "TCE",
    "car t": "CAR_T",
    "car-t": "CAR_T",
    "car nk": "CAR_NK",
    "car-nk": "CAR_NK",
    "tcr t": "TCR_T",
    "tcr-t": "TCR_T",
    "mab": "mAb",
    "antibody": "mAb",
    "monoclonal antibody": "mAb",
    "small molecule": "small_molecule",
    "small_molecule": "small_molecule",
    "fusion protein": "fusion_protein",
    "fusion_protein": "fusion_protein",
}

_POSITIVE_RANK = {
    "major_benefit": 0,
    "benefit": 1,
    "stable_disease": 2,
    "current": 3,
}
_NEGATIVE_STATUSES = frozenset(
    {"no_benefit", "progression", "intolerance", "contraindicated"}
)

_BENEFIT_RANK = {
    "curative": 0,
    "durable_rfs": 1,
    "major_survival": 2,
    "high_response": 3,
    "meaningful_pfs": 4,
    "incremental": 5,
    "modest": 6,
    "unclear": 7,
}
_TOXICITY_RANK = {
    "minimal": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "very_high": 4,
    "unclear": 5,
}


def _clean(value: object) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _normalized_words(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _clean(value).casefold()))


def normalize_treatment_status(value: object) -> str:
    text = _clean(value).casefold().replace("-", " ").replace("_", " ")
    status = _STATUS_ALIASES.get(text, text.replace(" ", "_"))
    if status not in TREATMENT_STATUSES:
        raise ValueError(
            f"Unsupported treatment status {_clean(value)!r}; use one of "
            f"{', '.join(TREATMENT_STATUSES)}"
        )
    return status


def normalize_treatment_modality(value: object) -> str:
    raw = _clean(value)
    if not raw:
        return ""
    key = raw.casefold().replace("_", " ")
    return _MODALITY_ALIASES.get(key, raw)


@dataclass(frozen=True)
class TreatmentRecord:
    """One patient treatment-history assertion supplied to the run."""

    therapy: str
    status: str
    target: str = ""
    modality: str = ""
    note: str = ""
    source: str = ""
    source_path: str = ""
    row_index: int | None = None

    def __post_init__(self) -> None:
        therapy = _clean(self.therapy).replace("|", "/")
        target = _clean(self.target).upper()
        modality = normalize_treatment_modality(self.modality).replace("|", "/")
        status = normalize_treatment_status(self.status)
        if not therapy and not target:
            raise ValueError("Treatment history requires a therapy or target")
        if target and not re.fullmatch(r"[A-Z0-9][A-Z0-9-]*", target):
            raise ValueError(f"Invalid treatment target symbol {target!r}")
        if not therapy and target and not modality:
            raise ValueError(
                "Target-only treatment history also requires a modality so it "
                "does not apply one outcome to every therapy for that target"
            )
        object.__setattr__(self, "therapy", therapy)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "note", _clean(self.note))
        object.__setattr__(self, "source", _clean(self.source))
        object.__setattr__(self, "source_path", _clean(self.source_path))

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        source_path: str = "",
        row_index: int | None = None,
    ) -> "TreatmentRecord":
        normalized = {
            re.sub(r"[^a-z0-9]+", "_", str(key).strip().casefold()).strip("_"): val
            for key, val in value.items()
        }

        def first(*keys: str) -> object:
            for key in keys:
                if _clean(normalized.get(key)):
                    return normalized[key]
            return ""

        return cls(
            therapy=_clean(first("therapy", "agent", "drug", "treatment")),
            target=_clean(first("target", "target_gene", "target_symbol", "gene")),
            modality=_clean(first("modality", "agent_class", "therapy_class")),
            status=_clean(first("status", "outcome", "response")),
            note=_clean(first("note", "notes", "details")),
            source=_clean(first("source", "provenance")),
            source_path=source_path,
            row_index=row_index,
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "therapy": self.therapy,
            "target": self.target,
            "modality": self.modality,
            "status": self.status,
            "note": self.note,
            "source": self.source,
            "source_path": self.source_path,
            "row_index": self.row_index,
        }


def _mapping_records_from_path(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, Mapping):
            payload = payload.get("treatments", payload.get("treatment_history", payload))
        if isinstance(payload, Mapping):
            payload = [payload]
        if not isinstance(payload, list) or not all(
            isinstance(item, Mapping) for item in payload
        ):
            raise ValueError(
                "Treatment-history JSON must be a record, a list of records, or "
                "an object with a treatments list"
            )
        return list(payload)
    if suffix in {".jsonl", ".ndjson"}:
        records = []
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"Treatment-history JSONL line {line_no} is not an object"
                )
            records.append(value)
        return records
    delimiter = "\t" if suffix in {".tsv", ".txt"} else ","
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def parse_treatment_history(path_value: str | Path | None) -> list[TreatmentRecord]:
    """Read treatment history from CSV, TSV, JSON, or JSONL."""
    if path_value is None or not str(path_value).strip():
        return []
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Treatment-history file not found: {path}")
    records = []
    for row_index, mapping in enumerate(_mapping_records_from_path(path), start=1):
        try:
            records.append(
                TreatmentRecord.from_mapping(
                    mapping,
                    source_path=str(path),
                    row_index=row_index,
                )
            )
        except ValueError as exc:
            raise ValueError(f"Invalid treatment-history row {row_index}: {exc}") from exc
    return records


def treatment_records(analysis: object) -> list[TreatmentRecord]:
    if not isinstance(analysis, Mapping):
        return []
    records = []
    for row_index, value in enumerate(analysis.get("treatment_history") or [], start=1):
        if isinstance(value, TreatmentRecord):
            records.append(value)
        elif isinstance(value, Mapping):
            records.append(TreatmentRecord.from_mapping(value, row_index=row_index))
    return records


def _row_modality(target_row: Mapping[str, Any]) -> str:
    return normalize_treatment_modality(
        target_row.get("modality") or target_row.get("agent_class")
    )


def treatment_record_match_kind(
    record: TreatmentRecord, target_row: Mapping[str, Any]
) -> str:
    """Return ``exact_agent``, ``target_modality``, or an empty non-match."""
    row_agent = _normalized_words(target_row.get("agent"))
    record_agent = _normalized_words(record.therapy)
    if row_agent and record_agent and row_agent == record_agent:
        return "exact_agent"

    row_target = _clean(target_row.get("symbol") or target_row.get("target_gene")).upper()
    if not record.target or not row_target or record.target != row_target:
        return ""
    if record.modality and record.modality == _row_modality(target_row):
        return "target_modality"
    return ""


def treatment_history_matches(target_row, analysis) -> list[dict[str, Any]]:
    if not hasattr(target_row, "get"):
        return []
    matches = []
    for record in treatment_records(analysis):
        kind = treatment_record_match_kind(record, target_row)
        if kind:
            matches.append({"record": record, "match_kind": kind})
    return matches


def treatment_history_rank(target_row, analysis) -> int:
    """Patient-evidence rank; lower values outrank model-only nominations."""
    ranks = [
        _POSITIVE_RANK[match["record"].status]
        for match in treatment_history_matches(target_row, analysis)
        if match["record"].status in {"major_benefit", "benefit", "stable_disease"}
    ]
    return min(ranks) if ranks else 10


def treatment_history_supports_review(target_row, analysis) -> bool:
    return treatment_history_rank(target_row, analysis) < 3


def treatment_history_blocks_row(target_row, analysis) -> bool:
    """Whether supplied history argues against presenting this as a candidate.

    A named negative outcome blocks that same agent. A target-and-modality-only
    record intentionally applies to the whole specified class. A negative
    outcome never spills across unrelated modalities for the same target.
    """
    for match in treatment_history_matches(target_row, analysis):
        record = match["record"]
        if record.status not in _NEGATIVE_STATUSES:
            continue
        if match["match_kind"] == "exact_agent" or not record.therapy:
            return True
    return False


def treatment_history_marks_current(target_row, analysis) -> bool:
    return any(
        match["record"].status == "current"
        and (
            match["match_kind"] == "exact_agent"
            or not match["record"].therapy
        )
        for match in treatment_history_matches(target_row, analysis)
    )


def best_treatment_history_match(target_row, analysis) -> dict[str, Any] | None:
    matches = [
        match
        for match in treatment_history_matches(target_row, analysis)
        if match["record"].status in _POSITIVE_RANK
        or match["match_kind"] == "exact_agent"
        or not match["record"].therapy
    ]
    if not matches:
        return None
    status_rank = {
        "contraindicated": 0,
        "intolerance": 1,
        "progression": 2,
        "no_benefit": 3,
        "current": 4,
        "major_benefit": 10,
        "benefit": 11,
        "stable_disease": 12,
    }
    return min(
        matches,
        key=lambda match: (
            status_rank.get(match["record"].status, 99),
            0 if match["match_kind"] == "exact_agent" else 1,
        ),
    )


def treatment_history_context(target_row, analysis) -> str:
    match = best_treatment_history_match(target_row, analysis)
    if match is None:
        return ""
    record = match["record"]
    therapy = record.therapy or "this target and modality"
    status_text = {
        "major_benefit": "major prior benefit",
        "benefit": "prior benefit",
        "stable_disease": "prior disease control",
        "current": "current treatment",
        "no_benefit": "no prior benefit",
        "progression": "prior progression",
        "intolerance": "prior intolerance",
        "contraindicated": "a contraindication",
    }[record.status]
    relation = (
        "this agent" if match["match_kind"] == "exact_agent" else "this target and modality"
    )
    base = f"Supplied treatment history reports {status_text} with {therapy} ({relation})"
    if record.status in {"major_benefit", "benefit", "stable_disease"}:
        base += (
            "; this evidence from the patient outranks the RNA source estimate, "
            "but current suitability, resistance, toxicity, organ function, and "
            "eligibility still require clinical review"
        )
    elif record.status == "current":
        base += "; reconcile response and toxicity before presenting it as a new start"
    else:
        base += "; do not prioritize the same treatment without a specific clinical rationale"
    if record.note:
        base += f"; supplied note: {record.note.replace('|', '/')}"
    if record.source:
        base += f"; supplied source: {record.source.replace('|', '/')}"
    return base


def treatment_history_summary_lines(analysis) -> list[str]:
    lines = []
    for record in treatment_records(analysis):
        label = record.therapy or f"{record.target} {record.modality}"
        status = record.status.replace("_", " ")
        details = []
        if record.note:
            details.append(record.note.replace("|", "/"))
        if record.source:
            details.append(f"source: {record.source.replace('|', '/')}")
        detail = " — " + "; ".join(details) if details else ""
        lines.append(f"- **{label}**: {status}{detail}")
    return lines


def _registry_agents_for_record(record: TreatmentRecord):
    from .therapeutic_agents import agents_for_target, therapeutic_agents

    candidates = list(agents_for_target(record.target)) if record.target else []
    if not candidates and record.therapy:
        wanted = _normalized_words(record.therapy)
        registry = therapeutic_agents()
        for _, row in registry.iterrows():
            names = [row.get("agent"), *str(row.get("aliases") or "").split(";")]
            if wanted and wanted in {_normalized_words(name) for name in names}:
                candidates.extend(agents_for_target(row.get("target_gene")))
                break
    if record.modality:
        candidates = [agent for agent in candidates if agent.modality == record.modality]
    if record.therapy:
        wanted = _normalized_words(record.therapy)
        exact = [
            agent
            for agent in candidates
            if wanted
            in {
                _normalized_words(agent.agent),
                *(_normalized_words(name) for name in agent.aliases.split(";")),
            }
        ]
        if exact:
            return exact
    return candidates


def _record_names_registry_agent(record: TreatmentRecord, agent) -> bool:
    if not record.therapy or agent is None:
        return False
    wanted = _normalized_words(record.therapy)
    names = {
        _normalized_words(agent.agent),
        *(_normalized_words(name) for name in agent.aliases.split(";")),
    }
    return bool(wanted and wanted in names)


def treatment_history_supplement_rows(
    analysis: object, *, cancer_code: str, existing_rows: object = None
) -> list[dict[str, Any]]:
    """Create report rows when patient evidence is outside the disease panel."""
    if existing_rows is None:
        existing_records = []
    elif hasattr(existing_rows, "to_dict"):
        existing_records = existing_rows.to_dict("records")
    else:
        existing_records = list(existing_rows)
    rows = []
    for record in treatment_records(analysis):
        if any(
            treatment_record_match_kind(record, row)
            for row in existing_records
            if hasattr(row, "get")
        ):
            continue
        candidates = _registry_agents_for_record(record)
        candidate = candidates[0] if candidates else None
        use_registry_metadata = bool(
            candidate
            and (
                not record.therapy
                or _record_names_registry_agent(record, candidate)
            )
        )
        target = record.target or (candidate.target_gene if candidate else "")
        therapy = record.therapy or (candidate.agent if candidate else "")
        modality = record.modality or (candidate.modality if candidate else "")
        if not therapy and not target:
            continue
        phase = candidate.highest_phase if use_registry_metadata else "patient_history"
        indication = (
            candidate.indications
            if use_registry_metadata and candidate.indications
            else "patient supplied treatment history"
        )
        rows.append(
            {
                "cancer_code": cancer_code,
                "subtype": "",
                "symbol": target,
                "agent": therapy,
                "agent_class": modality,
                "modality": modality,
                "phase": phase or "patient_history",
                "treatment_path_tier": "patient_history",
                "line_of_therapy": "patient_history",
                "eligibility_note": (
                    "patient treatment history supplied; confirm current disease "
                    "state, prior exposure, resistance, toxicity, organ function, "
                    "and indication-specific eligibility"
                ),
                "indication": indication,
                "rationale": "prior treatment evidence from this patient",
                "eligibility_basis": "patient_history",
                "requires_verified_alteration": False,
                "_treatment_history_supplement": True,
            }
        )
    return rows


def _population_agent_key(value: object) -> str:
    # Brand names and explanatory parentheticals should not prevent a match
    # between e.g. ``afami-cel (Tecelra)`` and the same canonical agent.
    text = re.sub(r"\([^)]*\)", " ", _clean(value))
    return _normalized_words(text)


def add_population_therapy_evidence(targets_df, *, cancer_code: str, subtype: str = ""):
    """Join sourced benefit/toxicity facts onto disease-curated therapy rows.

    This is deliberately an agent-and-disease join. Target RNA, target class,
    and a different drug against the same target cannot inherit clinical
    outcomes from another therapy.
    """
    if targets_df is None or len(targets_df) == 0:
        return targets_df
    try:
        from oncoref import therapy_benefit_toxicity_evidence

        evidence = therapy_benefit_toxicity_evidence()
    except (ImportError, AttributeError):
        return targets_df
    if evidence is None or evidence.empty:
        return targets_df

    wanted_code = _clean(cancer_code).casefold()
    code_values = evidence["cancer_code"].fillna("").astype(str).str.strip().str.casefold()
    evidence = evidence.loc[code_values.eq(wanted_code)]
    if subtype:
        wanted_subtype = _clean(subtype).casefold()
        subtype_values = (
            evidence["subtype"].fillna("").astype(str).str.strip().str.casefold()
        )
        evidence = evidence.loc[subtype_values.isin(("", wanted_subtype))]
    if evidence.empty:
        return targets_df

    by_agent: dict[str, list[Mapping[str, Any]]] = {}
    for record in evidence.to_dict("records"):
        key = _population_agent_key(record.get("agent"))
        if key:
            by_agent.setdefault(key, []).append(record)

    frame = targets_df.copy()
    joined_columns = {
        "benefit_tier": [],
        "toxicity_tier": [],
        "benefit_endpoint": [],
        "major_toxicities": [],
        "therapy_evidence_source": [],
        "therapy_evidence_url": [],
        "therapy_evidence_transfer": [],
        "therapy_evidence_note": [],
    }
    for _, target_row in frame.iterrows():
        matches = by_agent.get(_population_agent_key(target_row.get("agent")), [])
        if matches:
            record = min(
                matches,
                key=lambda item: (
                    _BENEFIT_RANK.get(_clean(item.get("benefit_tier")), 99),
                    _TOXICITY_RANK.get(_clean(item.get("toxicity_tier")), 99),
                ),
            )
        else:
            record = {}
        endpoint = " ".join(
            part
            for part in (
                _clean(record.get("endpoint_type")),
                _clean(record.get("endpoint_value")),
            )
            if part
        )
        values = {
            "benefit_tier": _clean(record.get("benefit_tier"))
            or _clean(target_row.get("benefit_tier")),
            "toxicity_tier": _clean(record.get("toxicity_tier"))
            or _clean(target_row.get("toxicity_tier")),
            "benefit_endpoint": endpoint
            or _clean(target_row.get("benefit_endpoint")),
            "major_toxicities": _clean(record.get("major_toxicities"))
            or _clean(target_row.get("major_toxicities")),
            "therapy_evidence_source": _clean(record.get("source_token"))
            or _clean(target_row.get("therapy_evidence_source")),
            "therapy_evidence_url": _clean(record.get("source_url"))
            or _clean(target_row.get("therapy_evidence_url")),
            "therapy_evidence_transfer": _clean(record.get("evidence_transfer"))
            or _clean(target_row.get("therapy_evidence_transfer")),
            "therapy_evidence_note": _clean(record.get("evidence_notes"))
            or _clean(target_row.get("therapy_evidence_note")),
        }
        for column, value in values.items():
            joined_columns[column].append(value)
    for column, values in joined_columns.items():
        frame[column] = values
    return frame


def population_therapy_evidence_rank(target_row) -> tuple[int, int]:
    if not hasattr(target_row, "get"):
        return 99, 99
    return (
        _BENEFIT_RANK.get(_clean(target_row.get("benefit_tier")), 99),
        _TOXICITY_RANK.get(_clean(target_row.get("toxicity_tier")), 99),
    )


def population_therapy_evidence_context(target_row) -> str:
    """Reader-facing sourced outcome context, never inferred from RNA."""
    if not hasattr(target_row, "get"):
        return ""
    benefit = _clean(target_row.get("benefit_tier"))
    toxicity = _clean(target_row.get("toxicity_tier"))
    endpoint = _clean(target_row.get("benefit_endpoint"))
    major_toxicities = _clean(target_row.get("major_toxicities"))
    source = _clean(target_row.get("therapy_evidence_source"))
    if not any((benefit, toxicity, endpoint, major_toxicities)):
        return ""
    parts = ["sourced clinical outcome evidence"]
    if benefit:
        parts.append(f"benefit tier {benefit.replace('_', ' ')}")
    if endpoint:
        parts.append(endpoint)
    if toxicity:
        parts.append(f"toxicity tier {toxicity.replace('_', ' ')}")
    if major_toxicities:
        toxicities = ", ".join(
            item.strip() for item in major_toxicities.split(";") if item.strip()
        )
        if toxicities:
            parts.append(f"major toxicities include {toxicities}")
    if source:
        parts.append(f"source {source}")
    return "; ".join(parts)


__all__ = [
    "TREATMENT_STATUSES",
    "TreatmentRecord",
    "add_population_therapy_evidence",
    "best_treatment_history_match",
    "normalize_treatment_modality",
    "normalize_treatment_status",
    "parse_treatment_history",
    "population_therapy_evidence_context",
    "population_therapy_evidence_rank",
    "treatment_history_blocks_row",
    "treatment_history_context",
    "treatment_history_matches",
    "treatment_history_marks_current",
    "treatment_history_rank",
    "treatment_history_supplement_rows",
    "treatment_history_summary_lines",
    "treatment_history_supports_review",
    "treatment_record_match_kind",
    "treatment_records",
]
