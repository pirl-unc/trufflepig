"""Validated, specimen-scoped clinical input, independent of RNA measurements.

Assay assertions remain separate when they disagree. A result is usable only
when its source, clinical validity, reportability and current-specimen scope
are explicit. This module does not infer clinical results from expression.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import date
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping


ASSAY_RESULTS = {
    "msi": ("MSI-H", "MSI-L", "MSS", "indeterminate", "pending", "not_tested", "unknown"),
    "mmr": ("dMMR", "pMMR", "indeterminate", "pending", "not_tested", "unknown"),
    "tmb": ("measured", "indeterminate", "pending", "not_tested", "unknown"),
}
POSITIVE_RESULTS = frozenset({"MSI-H", "dMMR"})
NEGATIVE_RESULTS = frozenset({"MSI-L", "MSS", "pMMR"})


def validated_fields(cls, value: Mapping) -> dict:
    """Reject unknown fields instead of silently dropping a clinical assertion."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{cls.__name__} must be a JSON object")
    extra = set(value) - {f.name for f in fields(cls)}
    if extra:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(extra)}")
    return dict(value)


@dataclass(frozen=True)
class ClinicalSource:
    """Document provenance and human review, separate from assay validity."""

    title: str = ""
    reference: str = ""
    excerpt: str = ""
    review_status: str = "supplied"

    def __post_init__(self):
        if not all(isinstance(getattr(self, f.name), str) for f in fields(self)):
            raise ValueError("Clinical source fields must be strings")
        if self.review_status not in {"supplied", "confirmed", "proposed", "rejected"}:
            raise ValueError(f"Invalid source review status: {self.review_status!r}")


@dataclass(frozen=True)
class SpecimenMetadata:
    """Explicit display and purpose metadata for the context's bound specimen.

    These assertions never change sample selection or disease evidence. A cell
    line identifier, file name or display label does not establish purpose.
    """

    display_label: str = ""
    purpose: str = "unknown"
    cell_line_id: str = ""
    collected_at: str = ""
    site: str = ""
    source: ClinicalSource = field(default_factory=ClinicalSource)

    def __post_init__(self):
        for f in fields(self):
            if f.name != "source" and not isinstance(getattr(self, f.name), str):
                raise ValueError(f"Specimen {f.name} must be a string")
        if self.purpose not in {"clinical", "research", "unknown"}:
            raise ValueError(f"Invalid specimen purpose: {self.purpose!r}")
        if self.collected_at:
            try:
                valid_date = date.fromisoformat(self.collected_at).isoformat() == self.collected_at
            except ValueError:
                valid_date = False
            if not valid_date:
                raise ValueError("Specimen collection date must be an ISO date (YYYY-MM-DD)")
        if not isinstance(self.source, ClinicalSource):
            object.__setattr__(self, "source", ClinicalSource(**validated_fields(ClinicalSource, self.source)))

    @property
    def report_purpose(self) -> str:
        """Unreviewed assertions remain visible without establishing purpose."""
        return self.purpose if self.source.review_status in {"supplied", "confirmed"} else "unknown"


@dataclass(frozen=True)
class ClinicalMeasurement:
    """An absolute reported measurement, never a value derived from RNA.

    Units are retained even when unsupported for a particular requirement.
    Missing and measured zero are distinct; a qualitative result is not a value.
    """

    value: float | None = None
    unit: str = ""

    def __post_init__(self):
        if self.value is not None and (
            isinstance(self.value, bool) or not isinstance(self.value, (int, float))
            or not math.isfinite(self.value) or self.value < 0
        ):
            raise ValueError("Clinical measurement must be a finite nonnegative number or null")
        if not isinstance(self.unit, str):
            raise ValueError("Clinical measurement unit must be a string")
        unit = self.unit.strip()
        if unit.casefold() in {"mut/mb", "mutations/mb", "mutations/megabase"}:
            unit = "mut/Mb"
        object.__setattr__(self, "unit", unit)


@dataclass(frozen=True)
class ClinicalAssay:
    """One reported clinical assay; missing and negative are distinct results.

    ``scope=current`` explicitly binds the assay to the context's specimen.
    Older, unrelated and unresolved results are retained without opening a gate.
    Per-protein MMR findings are preserved without deriving an overall dMMR call.
    """

    kind: str
    result: str = "unknown"
    method: str = ""
    specimen_id: str = ""
    scope: str = "unknown"
    collected_at: str = ""
    reported_at: str = ""
    validity: str = "unknown"
    reportability: str = "unknown"
    source: ClinicalSource = field(default_factory=ClinicalSource)
    protein_results: dict[str, str] = field(default_factory=dict)
    id: str = ""
    measurement: ClinicalMeasurement | None = None
    test_id: str = ""
    specimen_type: str = ""

    def __post_init__(self):
        for f in fields(self):
            if f.name not in {"source", "protein_results", "measurement"} and not isinstance(
                getattr(self, f.name), str
            ):
                raise ValueError(f"Clinical assay {f.name} must be a string")
        if self.kind not in ASSAY_RESULTS:
            raise ValueError(f"Unsupported clinical assay: {self.kind!r}")
        # Canonicalize case and punctuation only, never extract a result from prose.
        canonical = {
            v.casefold().replace("-", "").replace("_", ""): v for v in ASSAY_RESULTS[self.kind]
        }
        result = canonical.get(self.result.strip().casefold().replace("-", "").replace("_", ""))
        if result is None:
            raise ValueError(f"Invalid {self.kind.upper()} result: {self.result!r}")
        object.__setattr__(self, "result", result)
        if self.scope not in {"current", "historical", "other_specimen", "unknown"}:
            raise ValueError(f"Invalid assay specimen scope: {self.scope!r}")
        if self.validity not in {"validated", "failed", "unverified", "unknown"}:
            raise ValueError(f"Invalid assay validity: {self.validity!r}")
        if self.reportability not in {"reportable", "unreportable", "unknown"}:
            raise ValueError(f"Invalid assay reportability: {self.reportability!r}")
        for value in (self.collected_at, self.reported_at):
            if value:
                try:
                    parsed = date.fromisoformat(value)
                except ValueError as exc:
                    raise ValueError("Clinical dates must be ISO dates (YYYY-MM-DD)") from exc
                if parsed.isoformat() != value:
                    raise ValueError("Clinical dates must be ISO dates (YYYY-MM-DD)")
        if self.collected_at and self.reported_at and self.reported_at < self.collected_at:
            raise ValueError("Assay report date precedes specimen collection")
        if not isinstance(self.source, ClinicalSource):
            object.__setattr__(
                self, "source", ClinicalSource(**validated_fields(ClinicalSource, self.source))
            )
        if not isinstance(self.protein_results, Mapping):
            raise ValueError("MMR protein results must be an object")
        if self.protein_results and self.kind != "mmr":
            raise ValueError("Per-protein IHC results belong to an MMR assay")
        for protein, state in self.protein_results.items():
            if protein not in {"MLH1", "MSH2", "MSH6", "PMS2"}:
                raise ValueError(f"Unknown MMR protein: {protein!r}")
            if state not in {"retained", "lost", "equivocal", "not_tested", "unknown"}:
                raise ValueError(f"Invalid MMR protein result: {state!r}")
        object.__setattr__(self, "protein_results", dict(self.protein_results))
        if self.measurement is not None:
            if self.kind != "tmb":
                raise ValueError("An absolute measurement currently belongs to a TMB assay")
            if not isinstance(self.measurement, ClinicalMeasurement):
                object.__setattr__(self, "measurement", ClinicalMeasurement(
                    **validated_fields(ClinicalMeasurement, self.measurement)
                ))
        if not self.id:
            payload = {k: v for k, v in self.public_dict().items() if k != "id"}
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
            object.__setattr__(self, "id", f"{self.kind}-{digest}")

    def limitations(self, specimen_id: str) -> tuple[str, ...]:
        """Reasons this assertion cannot establish a current clinical result."""
        limits = []
        if self.scope != "current":
            limits.append(f"specimen scope is {self.scope}")
        if not specimen_id or not self.specimen_id:
            limits.append("specimen identity is missing")
        elif self.specimen_id != specimen_id:
            limits.append("assay concerns a different specimen")
        if self.validity != "validated":
            limits.append(f"assay validity is {self.validity}")
        if self.reportability != "reportable":
            limits.append(f"reportability is {self.reportability}")
        methods = {"msi": {"PCR", "NGS"}, "mmr": {"IHC", "NGS"}, "tmb": {"NGS"}}
        if self.method not in methods[self.kind]:
            limits.append("a supported clinical assay method is required")
        if not self.source.title.strip() and not self.source.reference.strip():
            limits.append("clinical source is missing")
        if self.source.review_status not in {"supplied", "confirmed"}:
            limits.append(f"source assertion is {self.source.review_status}")
        usable_results = {"measured"} if self.kind == "tmb" else POSITIVE_RESULTS | NEGATIVE_RESULTS
        if self.result not in usable_results:
            limits.append(f"result is {self.result}")
        if self.kind == "tmb":
            if self.measurement is None or self.measurement.value is None:
                limits.append("absolute TMB measurement is missing")
            if self.measurement is None or self.measurement.unit != "mut/Mb":
                limits.append("absolute TMB requires mutations per megabase")
        # An explicit overall result cannot silently override contradictory IHC.
        if self.result == "pMMR" and "lost" in self.protein_results.values():
            limits.append("pMMR conflicts with a reported lost MMR protein")
        if (
            self.result == "dMMR"
            and len(self.protein_results) == 4
            and set(self.protein_results.values()) == {"retained"}
        ):
            limits.append("dMMR conflicts with retention of all four MMR proteins")
        return tuple(limits)

    def public_dict(self) -> dict:
        value = asdict(self)
        # Preserve v1 MSI/MMR records and their content-derived IDs exactly.
        for key in ("measurement", "test_id", "specimen_type"):
            if value[key] is None or value[key] == "":
                value.pop(key)
        return value


@dataclass(frozen=True)
class ClinicalContext:
    """Versioned clinical input explicitly associated with one run specimen.

    The caller supplies the specimen binding; filenames and RNA similarity do
    not establish cross-assay identity. Different assay sources are not merged.
    """

    schema_version: int = 1
    specimen_id: str = ""
    assays: tuple[ClinicalAssay, ...] = ()
    specimen: SpecimenMetadata = field(default_factory=SpecimenMetadata)

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(f"Unsupported clinical-context version: {self.schema_version!r}")
        if not isinstance(self.specimen_id, str):
            raise ValueError("Clinical specimen ID must be a string")
        if not isinstance(self.specimen, SpecimenMetadata):
            object.__setattr__(self, "specimen", SpecimenMetadata(**validated_fields(SpecimenMetadata, self.specimen)))
        if not isinstance(self.assays, (tuple, list)):
            raise ValueError("Clinical assays must be an array")
        assays = tuple(
            a
            if isinstance(a, ClinicalAssay)
            else ClinicalAssay(**validated_fields(ClinicalAssay, a))
            for a in self.assays
        )
        if len({a.id for a in assays}) != len(assays):
            raise ValueError(
                "Clinical assay IDs must be unique; preserve conflicting sources separately"
            )
        object.__setattr__(self, "assays", assays)

    def public_dict(self) -> dict:
        value = asdict(self)
        value["assays"] = [a.public_dict() for a in self.assays]
        return value


def load_clinical_context(value=None) -> ClinicalContext:
    """Load the same validated input from Python, a mapping or a JSON path."""
    if value is None:
        return ClinicalContext()
    if isinstance(value, ClinicalContext):
        return value
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    return ClinicalContext(**validated_fields(ClinicalContext, value))


def clinical_context_for_analysis(analysis) -> ClinicalContext:
    """Return the clinical contract carried by the ordinary analysis path."""
    return (
        load_clinical_context(analysis.get("clinical_context"))
        if isinstance(analysis, Mapping)
        else ClinicalContext()
    )


def report_identity(
    context: ClinicalContext, *, source_path: str = "", sample_selector: str = "",
    selector_column: str = "", output_prefix: str = "", fallback_label: str = "",
) -> dict:
    """Resolve a visible identity without interpreting an input filename.

    The document identifier distinguishes source/selector combinations; it is
    not a biological specimen identifier or a cross-assay identity claim.
    """
    origin = {
        "source_path": str(source_path),
        "sample_selector": str(sample_selector),
        "selector_column": str(selector_column),
        "output_prefix": str(output_prefix),
    }
    digest = hashlib.sha256(json.dumps(origin, sort_keys=True).encode()).hexdigest()[:12]
    document_id = "RPT-" + digest
    metadata = context.specimen
    label = metadata.display_label.strip() or context.specimen_id.strip() or metadata.cell_line_id.strip()
    title = label or (fallback_label.strip() or "RNA evidence report") + " · " + document_id
    return {
        **origin, "document_id": document_id, "title": title,
        "specimen_id": context.specimen_id, "purpose": metadata.report_purpose,
        "explicit_display_identity": bool(label),
    }


@dataclass(frozen=True)
class ClinicalAssayDecision:
    status: str  # positive, negative, conflicting, unresolved, missing
    specimen_id: str
    reason: str
    assays: tuple[dict, ...]

    @property
    def satisfied(self) -> bool:
        return self.status == "positive"

    def public_dict(self) -> dict:
        value = asdict(self)
        value["assays"] = list(self.assays)
        return value


def evaluate_msi_mmr(context: ClinicalContext) -> ClinicalAssayDecision:
    """Evaluate clinical MSI-H/dMMR evidence without consulting an RNA proxy.

    Discordant reportable results require reconciliation. Historical/failed/
    unreviewed assertions remain visible and cannot cancel a usable current
    result. A contradictory overall/per-protein result remains unresolved.
    """
    records = clinical_assay_records(context, kinds=("msi", "mmr"))
    usable = [r for r in records if not r["limitations"]]
    states = {"positive" if r["result"] in POSITIVE_RESULTS else "negative" for r in usable}
    if len(states) > 1:
        status, reason = (
            "conflicting",
            "Reportable current MSI/MMR results disagree; reconcile the clinical reports.",
        )
    elif any(
        r["limitations"] and all("conflicts with" in limit for limit in r["limitations"])
        for r in records
    ):
        status, reason = (
            "conflicting",
            "An overall MMR result conflicts with its per-protein findings; reconcile the clinical report.",
        )
    elif states:
        status = next(iter(states))
        reason = (
            "Validated, reportable clinical MSI-H/dMMR evidence satisfies the biomarker requirement."
            if status == "positive"
            else "The supplied clinical result does not satisfy an MSI-H/dMMR indication."
        )
    elif records:
        status, reason = (
            "unresolved",
            "Supplied MSI/MMR evidence has unresolved result, quality, source or specimen limitations.",
        )
    else:
        status, reason = "missing", "No clinical MSI/MMR assay result was supplied."
    return ClinicalAssayDecision(status, context.specimen_id, reason, records)


def clinical_assay_records(context: ClinicalContext, *, kinds=None) -> tuple[dict, ...]:
    """All supplied assertions with limitations, optionally restricted by kind."""
    return tuple(
        {**a.public_dict(), "limitations": list(a.limitations(context.specimen_id))}
        for a in context.assays if kinds is None or a.kind in kinds
    )


@dataclass(frozen=True)
class TmbCriterion:
    """A sourced treatment requirement, separate from a specimen measurement.

    A threshold is not universal across therapies, assays or specimen materials.
    Empty assay/material criteria are rejected rather than treated as wildcards.
    """

    minimum: float
    unit: str
    specimen_type: str
    accepted_test_ids: tuple[str, ...]
    source: str

    def __post_init__(self):
        measurement = ClinicalMeasurement(self.minimum, self.unit)
        if measurement.value is None or measurement.value <= 0 or measurement.unit != "mut/Mb":
            raise ValueError("TMB criterion requires a positive threshold in mut/Mb")
        object.__setattr__(self, "unit", measurement.unit)
        if not isinstance(self.specimen_type, str) or not self.specimen_type.strip():
            raise ValueError("TMB criterion requires specimen material")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("TMB criterion requires its clinical source")
        if not isinstance(self.accepted_test_ids, (list, tuple)) or not self.accepted_test_ids:
            raise ValueError("TMB criterion requires accepted assay identifiers")
        if any(not isinstance(v, str) or not v.strip() for v in self.accepted_test_ids):
            raise ValueError("TMB assay identifiers must be nonempty strings")
        object.__setattr__(self, "accepted_test_ids", tuple(self.accepted_test_ids))

    def public_dict(self) -> dict:
        value = asdict(self)
        value["accepted_test_ids"] = list(self.accepted_test_ids)
        return value


def evaluate_tmb(context: ClinicalContext, *, criterion: TmbCriterion | None) -> ClinicalAssayDecision:
    """Compare usable absolute TMB results against one sourced assay criterion.

    Clinical MSI/MMR, RNA scores, mutation counts and cohort percentiles never
    participate. Conflicting usable results remain unresolved; older or failed
    assays stay visible without overriding a usable current result.
    """
    records = clinical_assay_records(context, kinds=("tmb",))
    if criterion is None:
        return ClinicalAssayDecision(
            "unresolved", context.specimen_id,
            "The treatment's TMB threshold and accepted assay/material criteria need clinical curation.",
            records,
        )
    if not isinstance(criterion, TmbCriterion):
        raise ValueError("TMB evaluation requires a TmbCriterion")
    for record in records:
        if record.get("test_id") not in criterion.accepted_test_ids:
            record["limitations"].append("test identity is not established for this TMB criterion")
        if record.get("specimen_type") != criterion.specimen_type:
            record["limitations"].append(f"this criterion requires {criterion.specimen_type} specimen material")
    states = {
        "positive" if r["measurement"]["value"] >= criterion.minimum else "negative"
        for r in records if not r["limitations"]
    }
    threshold = f"{criterion.minimum:g} {criterion.unit}"
    if len(states) > 1:
        status, reason = "conflicting", "Reportable current TMB results fall on opposite sides of the required threshold; reconcile the clinical reports."
    elif states:
        status = next(iter(states))
        reason = (
            f"Validated, reportable clinical TMB meets the threshold of at least {threshold} using an accepted assay and specimen material."
            if status == "positive" else
            f"The supplied clinical TMB is below the required threshold of {threshold}."
        )
    elif records:
        status, reason = "unresolved", "Supplied TMB evidence has unresolved measurement, quality, source, test or specimen limitations."
    else:
        status, reason = "missing", "No clinical TMB measurement was supplied."
    return ClinicalAssayDecision(status, context.specimen_id, reason, records)
