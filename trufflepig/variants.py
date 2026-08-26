"""Normalize supplied exact or symbolic variants for analysis.

A :class:`VariantRecord` represents an exact or symbolic assertion such as a
small variant, ``EGFR KDD``, an amplification, or a fusion. Coordinate-bearing
records carry validated 1-based intervals and an explicit GRCh37 or GRCh38
build. Sample-level genomic states such as MSI-H are deliberately outside this
contract.

The permissive table/text reader is retained for compatibility. New,
source-specific adapters and coordinate provenance are tracked in issues #140
and #141; callers should not mistake filename acceptance here for a complete
VCF or MAF parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from pathlib import Path
import gzip
import json
import re
from typing import Any, Iterable

import pandas as pd


_GENE_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,10}(?:-[A-Z0-9]{1,6})?\b")
_FUSION_PAIR_RE = re.compile(
    r"\b([A-Z][A-Z0-9]{1,10})\s*(?:::|--|/|-)\s*"
    r"([A-Z][A-Z0-9]{1,10})\b"
)
_NON_POSITIVE_EVENT_PATTERN = (
    r"(?:not\s+detected|negative|absent|wild[- ]?type|not\s+present|"
    r"inconclusive|fail(?:ed|ure)?|qns|pending|indeterminate|equivocal|"
    r"insufficient|cancel(?:l?ed)?|not\s+(?:assessed|evaluable)|"
    r"not\s+reportable|non[- ]?reportable|no\s+call|invalid|vus|"
    r"(?:variant\s+of\s+)?uncertain\s+significance|"
    r"(?:likely\s+)?benign|low[- _]?(?:qual(?:ity)?|confidence))"
)
_NON_POSITIVE_RESULT_PATTERN = rf"(?:{_NON_POSITIVE_EVENT_PATTERN}|unknown)"
_NON_POSITIVE_EVENT_RE = re.compile(
    rf"\b{_NON_POSITIVE_EVENT_PATTERN}\b",
    re.IGNORECASE,
)
_NON_POSITIVE_STATUS_RE = re.compile(
    rf"(?:\b(?:false|no|neg|{_NON_POSITIVE_RESULT_PATTERN})\b|"
    r"(?<![\d.])0(?:\.0+)?(?![\d.]))",
    re.IGNORECASE,
)
_EXPLICIT_UNKNOWN_OUTCOME_RE = re.compile(
    r"\b(?:result|status|call)\b(?:\s+(?:is|was))?\s*"
    r"(?:[-:=]\s*)?unknown\b|"
    r"\b(?:fusion|rearrangement|translocation|mutation|variant|kdd|"
    r"kinase\s+domain\s+duplication)\s*(?:[-:=]\s*|(?:is|was)\s+)unknown\b",
    re.IGNORECASE,
)
_FUSION_EVENT_PATTERN = r"(?:fusions?|rearrang(?:e|ed|ements?|ing)|translocations?)"
_FUSION_EVENT_RE = re.compile(rf"\b{_FUSION_EVENT_PATTERN}\b", re.IGNORECASE)
_SAMPLE_GENOMIC_STATE_RE = re.compile(
    r"\b(?:msi[- ]?(?:h|high)|dmmr|mismatch\s+repair\s+deficien(?:t|cy)|"
    r"tmb[- ]?high|high\s+tumou?r\s+mutation\s+burden)\b",
    re.IGNORECASE,
)
_NON_GENE_TOKENS = {
    "A",
    "AA",
    "AMP",
    "CNV",
    "DNA",
    "EXON",
    "FUSION",
    "GENE",
    "HIGH",
    "HLA",
    "IN",
    "ITD",
    "KDD",
    "LOW",
    "LOSS",
    "MSI",
    "NGS",
    "OUT",
    "RNA",
    "SNV",
    "TRUE",
    "FALSE",
    "VARIANT",
}
_PATH_SUFFIX_RE = re.compile(
    r"\.(csv|tsv|txt|xlsx|xls|json|jsonl|maf|vcf)(?:\.gz)?$",
    re.IGNORECASE,
)
_GENOME_BUILD_ALIASES = {
    "grch37": "GRCh37",
    "hg19": "GRCh37",
    "b37": "GRCh37",
    "grch38": "GRCh38",
    "hg38": "GRCh38",
}


def normalize_genome_build(value: object) -> str:
    """Return one canonical genome-build label or an empty unknown value."""
    build = str(value or "").strip()
    if not build or build.lower() in {"nan", "<na>", "none"}:
        return ""
    compact = re.sub(r"[^a-z0-9]+", "", build.lower())
    if compact in _GENOME_BUILD_ALIASES:
        return _GENOME_BUILD_ALIASES[compact]
    raise ValueError(
        f"Unsupported genome build {build!r}; use GRCh37/hg19 or GRCh38/hg38"
    )


def normalize_variant_contig(value: object) -> str:
    """Canonicalize recognized GRCh primary-contig aliases for comparison.

    ``7`` and ``chr7`` (likewise X/Y and M/MT) identify the same locus. Unknown
    alternate contigs are preserved verbatim rather than guessed.
    """
    contig = str(value or "").strip()
    if not contig:
        return ""
    token = re.sub(r"^chr", "", contig, flags=re.IGNORECASE).upper()
    if token.isdigit() and 1 <= int(token) <= 22:
        return str(int(token))
    if token in {"X", "Y"}:
        return token
    if token in {"M", "MT"}:
        return "MT"
    return contig


@dataclass(frozen=True)
class VariantCoordinate:
    """One 1-based genomic interval carried by a :class:`VariantRecord`."""

    contig: str
    start: int
    end: int | None = None
    ref: str = ""
    alt: str = ""
    role: str = ""

    def __post_init__(self) -> None:
        contig = str(self.contig or "").strip()
        if not contig:
            raise ValueError("Variant coordinates require a contig")
        start = _positive_integer(self.start, field_name="Variant start")
        if start is None:
            raise ValueError("Variant coordinates require a start position")
        end = (
            start
            if self.end is None
            else _positive_integer(self.end, field_name="Variant end")
        )
        if end is None or end < start:
            raise ValueError(
                f"Invalid 1-based variant interval {contig}:{start}-{end}"
            )
        object.__setattr__(self, "contig", contig)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "ref", str(self.ref or "").strip().upper())
        object.__setattr__(self, "alt", str(self.alt or "").strip().upper())
        object.__setattr__(self, "role", str(self.role or "").strip())

    @property
    def key(self) -> tuple[object, ...]:
        return (self.contig, self.start, self.end, self.ref, self.alt, self.role)

    def public_dict(self) -> dict[str, Any]:
        return {
            "contig": self.contig,
            "start": self.start,
            "end": self.end,
            "ref": self.ref,
            "alt": self.alt,
            "role": self.role,
        }


@dataclass(frozen=True)
class VariantRecord:
    """One normalized exact or symbolic variant call.

    ``variant_type`` is deliberately coarse. It is meant for report-gating
    compatibility (mutation vs fusion vs amplification vs KDD), not for
    replacing a clinical variant annotation.
    """

    gene: str
    variant: str = ""
    variant_type: str = "unknown"
    source_path: str = ""
    row_index: int | None = None
    confidence: str = ""
    support: dict[str, float] = field(default_factory=dict)
    raw_name: str = ""
    result_status: str = ""
    filter_status: str = ""
    filter_semantics: str = "generic"
    genes: tuple[str, ...] = ()
    representation: str = ""
    source_format: str = "unknown"
    caller_version: str = ""
    genome_build: str = ""
    ensembl_release: int | None = None
    coordinates: tuple[VariantCoordinate, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        genome_build: object = "",
        ensembl_release: int | None = None,
        source_path: str = "",
        source_format: str = "",
    ) -> "VariantRecord":
        """Build one record from a mapping using the public variant contract.

        Coordinates, rather than incidental metadata columns, determine whether
        an assembly applies.  Symbolic records therefore ignore both row-level
        and requested assembly labels.  Coordinate records canonicalize and
        reconcile those labels here, so every file adapter follows the same
        rule instead of reimplementing it.
        """
        item = dict(value)
        item.setdefault("variant", item.get("alteration", ""))
        item.setdefault("variant_type", item.get("alteration_type", "unknown"))
        item.pop("alteration", None)
        item.pop("alteration_type", None)

        coordinates = item.get("coordinates") or ()
        if coordinates:
            requested_build = normalize_genome_build(genome_build)
            row_build = normalize_genome_build(item.get("genome_build"))
            if requested_build and row_build and requested_build != row_build:
                row_index = item.get("row_index")
                location = (
                    f"record {row_index}" if row_index is not None else "record"
                )
                raise ValueError(
                    f"Variant {location} declares {row_build}, which conflicts "
                    f"with the requested build {requested_build}"
                )
            item["genome_build"] = row_build or requested_build
        else:
            # A symbolic event has no locus to interpret in an assembly.  Do not
            # reject an irrelevant exporter metadata value such as T2T-CHM13.
            item["genome_build"] = ""

        if item.get("ensembl_release") is None and ensembl_release is not None:
            item["ensembl_release"] = ensembl_release
        if not item.get("source_path") and source_path:
            item["source_path"] = source_path
        if not item.get("source_format") and source_format:
            item["source_format"] = source_format

        gene = str(item.get("gene") or "").strip().upper()
        return cls(
            gene=gene,
            genes=item.get("genes") or (gene,),
            variant=str(item.get("variant") or ""),
            variant_type=str(item.get("variant_type") or "unknown"),
            source_path=str(item.get("source_path") or ""),
            row_index=item.get("row_index"),
            confidence=str(item.get("confidence") or ""),
            support=dict(item.get("support") or {}),
            raw_name=str(item.get("raw_name") or ""),
            result_status=str(item.get("result_status") or ""),
            filter_status=str(item.get("filter_status") or ""),
            filter_semantics=str(item.get("filter_semantics") or "generic"),
            representation=str(item.get("representation") or ""),
            source_format=str(item.get("source_format") or "unknown"),
            caller_version=str(item.get("caller_version") or ""),
            genome_build=str(item.get("genome_build") or ""),
            ensembl_release=item.get("ensembl_release"),
            coordinates=coordinates,
        )

    def __post_init__(self) -> None:
        primary = str(self.gene or "").strip().upper()
        supplied_genes = (
            (self.genes,) if isinstance(self.genes, str) else self.genes
        )
        genes = tuple(
            dict.fromkeys(
                str(gene or "").strip().upper()
                for gene in (supplied_genes or (primary,))
                if str(gene or "").strip()
            )
        )
        if primary and primary not in genes:
            genes = (primary, *genes)
        if self.variant_type == "fusion" and len(genes) < 2:
            event_text = f"{self.variant} {self.raw_name}".upper()
            match = _FUSION_PAIR_RE.search(event_text)
            if match:
                pair = match.groups()
                if _FUSION_PAIR_RE.fullmatch(primary):
                    genes = tuple(gene for gene in genes if gene != primary)
                    primary = pair[0]
                genes = tuple(dict.fromkeys((*genes, *pair)))
        supplied_coordinates = (
            (self.coordinates,)
            if isinstance(self.coordinates, VariantCoordinate)
            or hasattr(self.coordinates, "get")
            else (self.coordinates or ())
        )
        coordinates = tuple(
            coordinate
            if isinstance(coordinate, VariantCoordinate)
            else VariantCoordinate(**dict(coordinate))
            for coordinate in supplied_coordinates
        )
        representation = "coordinate" if coordinates else "symbolic"
        requested_representation = str(self.representation or representation).strip()
        if requested_representation not in {"symbolic", "coordinate"}:
            raise ValueError(
                "Variant representation must be 'symbolic' or 'coordinate'"
            )
        if requested_representation != representation:
            raise ValueError(
                f"Variant representation {requested_representation!r} conflicts "
                f"with {len(coordinates)} coordinate interval(s)"
            )
        release = (
            _positive_integer(self.ensembl_release, field_name="Ensembl release")
            if self.ensembl_release is not None
            else None
        )
        genome_build = (
            normalize_genome_build(self.genome_build) if coordinates else ""
        )
        object.__setattr__(self, "gene", primary)
        object.__setattr__(self, "genes", genes)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(
            self,
            "source_format",
            str(self.source_format or "unknown").strip().lower(),
        )
        object.__setattr__(self, "caller_version", str(self.caller_version or ""))
        object.__setattr__(self, "genome_build", genome_build)
        object.__setattr__(
            self,
            "ensembl_release",
            release,
        )

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.gene,
            self.variant_type,
            self.variant.lower(),
            self.source_path,
            self.row_index,
            self.genome_build,
            tuple(coordinate.key for coordinate in self.coordinates),
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "gene": self.gene,
            "variant": self.variant,
            "variant_type": self.variant_type,
            "source_path": self.source_path,
            "row_index": self.row_index,
            "confidence": self.confidence,
            "result_status": self.result_status,
            "filter_status": self.filter_status,
            "filter_semantics": self.filter_semantics,
            "genes": list(self.genes),
            "representation": self.representation,
            "source_format": self.source_format,
            "caller_version": self.caller_version,
            "genome_build": self.genome_build,
            "ensembl_release": self.ensembl_release,
            "coordinates": [
                coordinate.public_dict() for coordinate in self.coordinates
            ],
            "support": dict(self.support),
            "raw_name": self.raw_name,
        }

    # Narrow Python compatibility for callers of the pre-1.24 API. New code
    # and serialized output use only ``variant`` / ``variant_type``.
    @property
    def alteration(self) -> str:
        return self.variant

    @property
    def alteration_type(self) -> str:
        return self.variant_type


def normalize_variant_record(record: object) -> dict[str, Any] | None:
    """Return one JSON-safe record using the current public field names.

    Pre-1.24 ``alteration`` field names remain readable, but are never emitted.
    Coordinates are validated through :class:`VariantCoordinate` so callers do
    not need a second, private normalization contract.
    """
    if isinstance(record, Mapping):
        item = dict(record)
    else:
        public_dict = getattr(record, "public_dict", None)
        if not callable(public_dict):
            return None
        public = public_dict()
        if not isinstance(public, Mapping):
            return None
        item = dict(public)
    item.setdefault("variant", item.get("alteration", ""))
    item.setdefault("variant_type", item.get("alteration_type", "unknown"))
    item.pop("alteration", None)
    item.pop("alteration_type", None)

    normalized = VariantRecord.from_mapping(item).public_dict()
    for key, value in item.items():
        normalized.setdefault(key, value)
    return normalized


def validate_variant_genome_builds(
    records: Iterable[object],
    *,
    expected_build: object = "",
    require_coordinate_build: bool = True,
) -> dict[str, Any]:
    """Validate coordinate-bearing records and summarize build provenance.

    Symbolic calls such as ``EGFR KDD`` are assembly-neutral. Coordinate calls
    must declare one build by default, and a collection may not mix builds.
    This function validates provenance only; it never performs liftover.
    """
    expected = normalize_genome_build(expected_build)
    coordinate_records = 0
    assembly_neutral_records = 0
    builds: set[str] = set()
    missing_build_sources: list[str] = []
    for record in records:
        item = normalize_variant_record(record)
        if item is None:
            continue
        if not item["coordinates"]:
            assembly_neutral_records += 1
            continue
        coordinate_records += 1
        build = normalize_genome_build(item.get("genome_build"))
        if not build:
            source = str(item.get("source_path") or "<programmatic record>")
            missing_build_sources.append(source)
            continue
        builds.add(build)
    if require_coordinate_build and missing_build_sources:
        sources = ", ".join(sorted(set(missing_build_sources)))
        raise ValueError(
            "Coordinate variants require an explicit genome build; missing for "
            f"{sources}"
        )
    if len(builds) > 1:
        raise ValueError(
            "Variant inputs use contradictory genome builds: "
            + ", ".join(sorted(builds))
        )
    observed = next(iter(builds), "")
    if expected and observed and expected != observed:
        raise ValueError(
            f"Variant input build {observed} does not match expected build {expected}"
        )
    return {
        "genome_build": observed,
        "expected_genome_build": expected,
        "coordinate_records": coordinate_records,
        "assembly_neutral_records": assembly_neutral_records,
        "coordinate_records_without_build": len(missing_build_sources),
    }


def split_variant_inputs(value: object) -> list[str]:
    """Parse semicolon/newline-separated variant inputs.

    Commas are intentionally not split here because inline variants often use
    comma-delimited prose (for example ``EGFR KDD, EGFR amplified``). Users can
    pass multiple files with semicolons or repeated list values through the API.
    """
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        if not text:
            return []
        return [part.strip() for part in re.split(r"[;\n]+", text) if part.strip()]
    if isinstance(value, Iterable):
        inputs: list[str] = []
        for item in value:
            inputs.extend(split_variant_inputs(item))
        return inputs
    return [str(value)]


def _clean_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "<na>", "none"}:
        return ""
    return text


def _result_status_is_non_positive(value: object) -> bool:
    """Whether a structured status explicitly says the event is not usable.

    Generic ``Status`` columns also carry variant classifications such as
    ``Pathogenic`` and ``Somatic``. Those are neither positive nor negative
    assay outcomes, so only recognized non-positive vocabulary vetoes a call.
    """
    parts = [part for part in re.split(r"[;,|]", _text(value)) if part.strip()]
    return any(_NON_POSITIVE_STATUS_RE.search(part) for part in parts)


def _filter_status_is_failed(value: object, *, vcf_semantics: bool) -> bool:
    """Whether a filter field explicitly rejects the event.

    VCF reserves ``PASS`` for calls that pass every filter and ``.`` for calls
    where filters were not applied, so every named VCF filter is a rejection.
    Generic spreadsheet ``Filter`` columns often hold confidence labels such
    as ``HighConfidence`` or ``Tier 1``; those are rejected only when they use
    explicit non-positive outcome vocabulary.
    """
    parts = [part.strip().lower() for part in re.split(r"[;,|]", _text(value))]
    parts = [part for part in parts if part]
    if vcf_semantics:
        return any(part not in {".", "pass", "passed"} for part in parts)
    return any(_NON_POSITIVE_STATUS_RE.search(part) for part in parts)


def variant_record_passes_assay_filters(record: object) -> bool:
    """Whether a normalized call passes structured result and filter fields.

    This public gate is intentionally disease- and therapy-independent. Free-
    text event negation remains gene-specific and is handled by
    :func:`variant_record_gene_is_negated`.
    """
    item = normalize_variant_record(record)
    if item is None:
        return False
    if any(
        _result_status_is_non_positive(item.get(key))
        for key in ("result_status", "result", "status")
    ):
        return False

    filter_semantics = str(item.get("filter_semantics") or "").lower()
    stored_filter_is_vcf = filter_semantics == "vcf" or (
        not filter_semantics and bool(_text(item.get("filter_status")))
    )
    if _filter_status_is_failed(
        item.get("filter_status"),
        vcf_semantics=stored_filter_is_vcf,
    ):
        return False
    # Preserve sensible behavior for callers that pass unnormalized records.
    # Exact uppercase FILTER follows VCF semantics; a generic ``filter`` key
    # rejects only explicit failure vocabulary.
    if _filter_status_is_failed(item.get("FILTER"), vcf_semantics=True):
        return False
    if _filter_status_is_failed(item.get("filter"), vcf_semantics=False):
        return False
    return True


def _event_clause_is_non_positive(clause: object) -> bool:
    """Whether free text explicitly reports that its event is not positive."""
    text = _text(clause)
    return bool(
        _NON_POSITIVE_EVENT_RE.search(text)
        or _EXPLICIT_UNKNOWN_OUTCOME_RE.search(text)
    )


def _clean_gene(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text or text == "NAN":
        return ""
    text = re.sub(r"\(.*?\)", "", text)
    text = text.split(";")[0].split(",")[0].strip()
    matches = [m.group(0) for m in _GENE_RE.finditer(text)]
    for token in matches:
        if token not in _NON_GENE_TOKENS and not token.startswith("CHR"):
            return token
    return ""


def variant_record_gene_is_negated(record: object, gene: str) -> bool:
    """Whether prose explicitly negates this gene's molecular finding."""
    item = normalize_variant_record(record)
    if item is None:
        return False
    wanted = str(gene or "").strip().upper()
    if not wanted:
        return False
    if not variant_record_passes_assay_filters(item):
        return True
    event_texts = [
        str(item.get(key) or "")
        for key in ("variant", "alteration", "raw_name", "confidence")
        if str(item.get(key) or "").strip()
    ]
    escaped = re.escape(wanted)
    # A negative outcome applies to the complete variant clause, not only
    # to a bare gene or fusion token.  This catches ordinary exports such as
    # ``EGFR KDD not detected`` and ``NTRK3::ETV6 fusion not detected`` while
    # the clause boundary keeps ``ALK fusion, NTRK3 not detected`` from
    # negating the independent ALK event.
    negative_in_gene_clause = any(
        re.search(rf"\b{escaped}\b", clause, re.IGNORECASE)
        and _event_clause_is_non_positive(clause)
        for text in event_texts
        for clause in re.split(r"[,;\n]+", text)
    )
    return bool(
        negative_in_gene_clause
        or any(
            re.search(
                rf"\b(?:no|without)\s+(?:evidence\s+of\s+)?(?:an?\s+)?"
                rf"\b{escaped}\b",
                text,
                re.IGNORECASE,
            )
            for text in event_texts
        )
    )


def _explicit_fusion_pair(record: object) -> tuple[str, ...]:
    """Return one connected, non-negated fusion pair from the event fields."""
    for key in ("gene", "variant", "alteration", "raw_name"):
        text = str(record.get(key) or "").upper()
        for match in _FUSION_PAIR_RE.finditer(text):
            genes = (match.group(1), match.group(2))
            if not _valid_fusion_pair_genes(genes):
                continue
            clause_end = min(
                [
                    boundary
                    for boundary in (
                        text.find(",", match.end()),
                        text.find(";", match.end()),
                        text.find("\n", match.end()),
                    )
                    if boundary >= 0
                ]
                or [len(text)]
            )
            clause = text[match.start() : clause_end]
            if _event_clause_is_non_positive(clause):
                continue
            if any(variant_record_gene_is_negated(record, gene) for gene in genes):
                continue
            return genes
    return ()


def variant_record_genes(record: object) -> tuple[str, ...]:
    """Return the structured gene and any explicit fusion partner.

    ``VariantRecord.gene`` remains the primary gene for backwards
    compatibility, but a fusion is intrinsically a two-gene event.  Loose text
    inputs such as ``ETV6-NTRK3 fusion`` historically retained only ``ETV6``;
    target matching therefore missed the therapeutically relevant ``NTRK3``
    partner. Recover a partner only from a connected pair such as
    ``ETV6-NTRK3`` or ``ETV6::NTRK3``. Other genes mentioned in commentary are
    not part of the event, and explicitly negated findings are excluded.
    """
    item = normalize_variant_record(record)
    if item is None:
        return ()
    variant_type = str(
        item.get("variant_type") or item.get("alteration_type") or ""
    ).strip().lower()
    event_text = " ".join(
        str(item.get(key) or "")
        for key in ("variant", "alteration", "raw_name")
    ).lower()
    fusion_like = (
        variant_type == "fusion"
        or classify_variant_type(event_text) == "fusion"
    )
    supplied_genes = item.get("genes") or ()
    if isinstance(supplied_genes, str):
        supplied_genes = (supplied_genes,)
    structured_genes = tuple(
        dict.fromkeys(
            gene
            for gene in (_clean_gene(value) for value in supplied_genes)
            if gene
        )
    )
    if structured_genes:
        if not variant_record_passes_assay_filters(item):
            return ()
        if any(
            variant_record_gene_is_negated(item, gene)
            for gene in structured_genes
        ):
            return ()
        return structured_genes
    if fusion_like:
        pair = _explicit_fusion_pair(item)
        if pair:
            return pair
    primary_text = str(item.get("gene") or "").upper()
    primary = _clean_gene(primary_text)
    if not primary:
        return ()
    if fusion_like and _FUSION_PAIR_RE.fullmatch(primary_text.strip()):
        # A connected pair was present but rejected above (for example because
        # the event was explicitly reported as not detected).
        return ()
    if variant_record_gene_is_negated(item, primary):
        return ()
    return (primary,)


def _record_dict(record: object) -> dict[str, Any] | None:
    return normalize_variant_record(record)


def _fusion_record_as_variant(record: object) -> dict[str, Any] | None:
    """Normalize a dedicated fusion call onto the variant-record contract."""
    source = _record_dict(record)
    if source is None:
        return None
    gene_a = _clean_gene(source.get("gene_a"))
    gene_b = _clean_gene(source.get("gene_b"))
    if not gene_a or not gene_b or gene_a == gene_b:
        return None
    pair = _text(source.get("pair")) or f"{gene_a}--{gene_b}"
    raw_name = _text(source.get("raw_name")) or pair
    normalized = dict(source)
    normalized.update(
        {
            "gene": gene_a,
            "genes": [gene_a, gene_b],
            "pair": pair,
            "variant": f"{pair} fusion",
            "variant_type": "fusion",
            "raw_name": raw_name,
            "result_status": _text(source.get("result_status"))
            or _text(source.get("reportable")),
            "filter_status": _text(source.get("filter_status"))
            or _text(source.get("confidence")),
            "filter_semantics": _text(source.get("filter_semantics")) or "generic",
            "source_format": (
                _text(source.get("source_format"))
                if _text(source.get("source_format")) not in {"", "unknown"}
                else "fusion"
            ),
            "evidence_source_type": "fusion",
            "evidence_source_types": ["fusion"],
        }
    )
    return normalized


def _variant_record_identity(record: dict[str, Any]) -> tuple[object, ...]:
    variant_type = _text(record.get("variant_type")).lower() or "unknown"
    # A fusion is biologically identified by its participating pair across both
    # the generic variant and dedicated fusion interfaces. Coordinates remain
    # provenance, but must not prevent a positive and negative report for that
    # same pair from reconciling and failing closed.
    if variant_type == "fusion":
        for key in ("pair", "variant", "raw_name", "gene"):
            match = _FUSION_PAIR_RE.search(_text(record.get(key)).upper())
            if match:
                pair = (match.group(1), match.group(2))
                if _valid_fusion_pair_genes(pair):
                    return ("fusion", *sorted(pair))
        gene_a = _clean_gene(record.get("gene_a"))
        gene_b = _clean_gene(record.get("gene_b"))
        if gene_a and gene_b and gene_a != gene_b:
            return ("fusion", *sorted((gene_a, gene_b)))
        supplied_genes = record.get("genes") or ()
        if isinstance(supplied_genes, str):
            supplied_genes = (supplied_genes,)
        participants = tuple(
            dict.fromkeys(
                gene
                for gene in (_clean_gene(value) for value in supplied_genes)
                if gene
            )
        )
        if len(participants) >= 2:
            return ("fusion", *sorted(participants[:2]))
    coordinates = tuple(
        (
            normalize_variant_contig(coordinate.get("contig")),
            coordinate.get("start"),
            coordinate.get("end"),
            _text(coordinate.get("ref")).upper(),
            _text(coordinate.get("alt")).upper(),
            _text(coordinate.get("role")),
        )
        for coordinate in (record.get("coordinates") or ())
        if hasattr(coordinate, "get")
    )
    if coordinates:
        return (
            "coordinate",
            normalize_genome_build(record.get("genome_build")),
            variant_type,
            *coordinates,
        )
    gene = _clean_gene(record.get("gene"))
    if variant_type == "fusion" and gene:
        return ("fusion", gene)
    # These variant classes are the molecular eligibility event.  Repeated
    # exports may phrase the result differently (for example ``amplification``
    # versus ``amplification not detected``), but a negative result should
    # reconcile only with the same class—not erase a distinct event involving
    # the same gene such as an EGFR kinase-domain duplication.
    if variant_type in {
        "amplification",
        "internal_tandem_duplication",
        "kdd",
        "loss",
    }:
        return (variant_type, gene)
    variant = re.sub(
        r"\s+",
        " ",
        _text(record.get("variant") or record.get("raw_name")).lower(),
    )
    # Outcome wording is not part of the biological event identity.  Removing
    # it lets ``BRAF V600E`` and ``BRAF V600E not detected`` reconcile as one
    # conflicting assay event while distinct variants remain separate.
    variant = _NON_POSITIVE_EVENT_RE.sub(" ", variant)
    variant = _EXPLICIT_UNKNOWN_OUTCOME_RE.sub(" ", variant)
    variant = re.sub(r"[^a-z0-9]+", " ", variant).strip()
    return (variant_type, gene, variant)


def _record_supports_identity(record: dict[str, Any]) -> bool:
    return bool(variant_record_genes(record))


def _source_types(record: dict[str, Any]) -> set[str]:
    values = record.get("evidence_source_types") or ()
    if isinstance(values, str):
        values = (values,)
    return {str(value) for value in values if str(value)}


def variant_evidence_records(analysis: object) -> list[dict[str, Any]]:
    """Return one normalized view of supplied variant and fusion evidence.

    The dedicated fusion interface remains the authoritative source for fusion
    classification and fusion-specific reporting. This view lets generic
    molecular consumers—therapy eligibility and pathway-source reasoning—use
    the same record contract regardless of which CLI/API input supplied an
    event. The same biological event supplied through both interfaces is
    retained once and carries both source types, so it cannot become two votes.
    """
    if not isinstance(analysis, Mapping):
        return []
    normalized: list[dict[str, Any]] = []
    by_identity: dict[tuple[object, ...], dict[str, Any]] = {}
    variant_rows = (
        analysis.get("variant_records")
        if "variant_records" in analysis
        else analysis.get("alteration_records")
    )
    sources = (
        ("variant", variant_rows or ()),
        ("fusion", analysis.get("fusion_records") or ()),
    )
    for source_type, records in sources:
        for record in records:
            item = (
                _fusion_record_as_variant(record)
                if source_type == "fusion"
                else _record_dict(record)
            )
            if item is None:
                continue
            if source_type == "variant":
                item.setdefault("evidence_source_type", "variant")
                item.setdefault("evidence_source_types", ["variant"])
            identity = _variant_record_identity(item)
            existing = by_identity.get(identity)
            if existing is None:
                by_identity[identity] = item
                normalized.append(item)
                continue
            existing_sources = _source_types(existing)
            existing_sources.update(_source_types(item))
            existing["evidence_source_types"] = sorted(
                existing_sources,
                key=lambda value: ({"variant": 0, "fusion": 1}.get(value, 2), value),
            )
            if _record_supports_identity(existing) != _record_supports_identity(item):
                existing["result_status"] = (
                    "inconclusive: conflicting supplied records"
                )
                existing["evidence_conflict"] = True
    return normalized


def variant_evidence_for_gene(analysis: object, gene: str) -> list[dict[str, Any]]:
    """Return positive, non-conflicting supplied evidence for one gene.

    :func:`variant_evidence_records` reconciles positive and non-positive
    reports for the same variant identity. Distinct variants involving
    one gene remain independent: a negative EGFR KDD assay must not suppress a
    separately reported EGFR amplification, for example.
    """
    wanted = _clean_gene(gene)
    if not wanted:
        return []
    records = variant_evidence_records(analysis)
    return [
        record
        for record in records
        if wanted in variant_record_genes(record)
    ]


def _valid_fusion_pair_genes(genes: tuple[str, str]) -> bool:
    return all(len(gene) >= 3 for gene in genes) and not any(
        gene in _NON_GENE_TOKENS or gene.startswith("CHR") for gene in genes
    )


def _has_unambiguous_fusion_pair(value: object) -> bool:
    """Whether a gene cell uses pair syntax that cannot be an HGNC hyphen."""
    text = _text(value).upper()
    if not re.search(r"(?:::|--|/|\|)", text):
        return False
    return any(
        _valid_fusion_pair_genes((match.group(1), match.group(2)))
        for match in _FUSION_PAIR_RE.finditer(text)
    )


def _find_column(columns: Iterable[object], candidates: set[str]) -> object | None:
    for col in columns:
        if _clean_header(col) in candidates:
            return col
    return None


_GENE_COLUMNS = {
    "gene",
    "genesymbol",
    "symbol",
    "hugosymbol",
    "hugo",
    "alteredgene",
    "targetgene",
    "target",
    "reportedgene",
    "variantgene",
}
_VARIANT_COLUMNS = {
    "variant",
    # Common external table header; normalized output still uses ``variant``.
    "alteration",
    "varianttype",
    "variantname",
    "mutation",
    "proteinchange",
    "codingchange",
    "event",
    "call",
    "description",
    "classification",
    "type",
    "effect",
}
_CONFIDENCE_COLUMNS = {"confidence", "filter", "status", "classification", "tier"}
_RESULT_STATUS_COLUMNS = {
    "result",
    "results",
    "callresult",
    "callstatus",
    "assayresult",
    "testresult",
    "status",
}
_FILTER_STATUS_COLUMNS = {"filter", "filterstatus", "vcffilter"}
_CONTIG_COLUMNS = {"chromosome", "chrom", "chr", "contig"}
_START_COLUMNS = {"position", "pos", "start", "startposition"}
_END_COLUMNS = {"end", "stop", "endposition"}
_REF_COLUMNS = {"ref", "reference", "referenceallele", "refallele"}
_ALT_COLUMNS = {"alt", "alternate", "alternateallele", "tumorseqallele2"}
_GENOME_BUILD_COLUMNS = {"genomebuild", "assembly", "ncbibuild", "referencebuild"}
_ENSEMBL_RELEASE_COLUMNS = {"ensemblrelease", "generelease"}
_CALLER_VERSION_COLUMNS = {"callerversion", "toolversion", "softwareversion"}
_SUPPORT_COLUMNS = {
    "readcount",
    "reads",
    "altreads",
    "variantreads",
    "supportingreads",
    "vaf",
    "allelefraction",
    "copyratio",
    "copynumber",
    "cn",
}


def _numeric(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _positive_integer(value: object, *, field_name: str) -> int | None:
    text = _text(value)
    if not text:
        return None
    numeric = _numeric(text)
    if numeric is None or numeric < 1 or not numeric.is_integer():
        raise ValueError(f"{field_name} must be a positive integer, got {text!r}")
    return int(numeric)


def _source_format(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return "csv"
    if name.endswith(".tsv") or name.endswith(".tsv.gz"):
        return "tsv"
    if name.endswith((".xlsx", ".xls")):
        return "excel"
    if name.endswith((".json", ".json.gz", ".jsonl", ".jsonl.gz")):
        return "json"
    if name.endswith(".txt") or name.endswith(".txt.gz"):
        return "text"
    if name.endswith(".vcf") or name.endswith(".vcf.gz"):
        return "vcf"
    if name.endswith(".maf") or name.endswith(".maf.gz"):
        return "maf"
    return "table"


def _read_text(path: Path) -> str:
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, mode="rt", errors="ignore") as handle:
            return handle.read()
    return path.read_text(errors="ignore")


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    text = _read_text(path)
    if path.name.lower().endswith((".jsonl", ".jsonl.gz")):
        payload: object = [
            json.loads(line) for line in text.splitlines() if line.strip()
        ]
    else:
        payload = json.loads(text)
    if isinstance(payload, dict):
        for key in ("variants", "alterations", "mutations", "records", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Variant JSON must contain one record or a list of records")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Variant JSON record {index} must be an object")
        rows.append(dict(row))
    return rows


def _records_from_typed_json(
    rows: list[dict[str, Any]],
    *,
    source_path: str,
    genome_build: object = "",
    ensembl_release: int | None = None,
) -> list[VariantRecord] | None:
    typed_rows = [
        row for row in rows if "coordinates" in row or "representation" in row
    ]
    if not typed_rows:
        return None
    if len(typed_rows) != len(rows):
        raise ValueError(
            "Variant JSON cannot mix typed VariantRecord objects with generic rows"
        )
    records: list[VariantRecord] = []
    for row in rows:
        item = dict(row)
        records.append(
            VariantRecord.from_mapping(
                item,
                genome_build=genome_build,
                ensembl_release=ensembl_release,
                source_path=source_path,
                source_format="json",
            )
        )
    return records


def _coordinate_from_row(row) -> VariantCoordinate | None:
    contig_col = _find_column(row.index, _CONTIG_COLUMNS)
    start_col = _find_column(row.index, _START_COLUMNS)
    end_col = _find_column(row.index, _END_COLUMNS)
    ref_col = _find_column(row.index, _REF_COLUMNS)
    alt_col = _find_column(row.index, _ALT_COLUMNS)
    contig = _text(row.get(contig_col)) if contig_col is not None else ""
    start = (
        _positive_integer(row.get(start_col), field_name="Variant start")
        if start_col is not None
        else None
    )
    if not contig and start is None:
        return None
    if not contig or start is None:
        raise ValueError(
            "Coordinate variant rows require both chromosome/contig and position/start"
        )
    end = (
        _positive_integer(row.get(end_col), field_name="Variant end")
        if end_col is not None
        else None
    )
    return VariantCoordinate(
        contig=contig,
        start=start,
        end=end,
        ref=_text(row.get(ref_col)) if ref_col is not None else "",
        alt=_text(row.get(alt_col)) if alt_col is not None else "",
    )


def _support_from_row(row) -> dict[str, float]:
    support: dict[str, float] = {}
    for col, value in row.items():
        if _clean_header(col) not in _SUPPORT_COLUMNS:
            continue
        num = _numeric(value)
        if num is not None:
            support[str(col)] = num
    return support


def classify_variant_type(text: object) -> str:
    """Return a coarse variant class from loose text."""
    low = str(text or "").lower()
    if re.search(r"\b(kdd|kinase\s+domain\s+duplication)\b", low):
        return "kdd"
    if re.search(r"\b(itd|internal\s+tandem\s+duplication|tandem\s+duplication)\b", low):
        return "internal_tandem_duplication"
    if _FUSION_EVENT_RE.search(low):
        return "fusion"
    if re.search(r"\b(amplification|amplified|\bamp\b|copy\s*number\s*gain)\b", low):
        return "amplification"
    if re.search(r"\b(loss|deletion|deleted|homozygous\s+del|copy\s*number\s*loss)\b", low):
        return "loss"
    if re.search(r"\b(v600|g12|g13|q61|l858r|t790m|exon\s*\d+|mutat|snv|indel|variant)\b", low):
        return "mutation"
    return "unknown"


def _reject_sample_genomic_state(text: str, variant_type: str) -> None:
    if variant_type == "unknown" and _SAMPLE_GENOMIC_STATE_RE.search(text):
        raise ValueError(
            f"{text!r} is a sample-level genomic state, not one variant; "
            "do not supply it through --variants"
        )


def _confidence_from_row(row) -> str:
    col = _find_column(row.index, _CONFIDENCE_COLUMNS)
    return _text(row.get(col)) if col is not None else ""


def _result_status_from_row(row) -> str:
    """Preserve structured assay outcomes used to veto negative calls."""
    values: list[str] = []
    for col, value in row.items():
        if _clean_header(col) not in _RESULT_STATUS_COLUMNS:
            continue
        text = _text(value)
        if text and text not in values:
            values.append(text)
    return "; ".join(values)


def _filter_status_from_row(row, *, source_path: str) -> tuple[str, str]:
    """Return the filter value and whether it has strict VCF semantics."""
    values: list[str] = []
    source_is_vcf = str(source_path).lower().endswith((".vcf", ".vcf.gz"))
    vcf_semantics = source_is_vcf
    for col, value in row.items():
        if _clean_header(col) not in _FILTER_STATUS_COLUMNS:
            continue
        text = _text(value)
        if text and text not in values:
            values.append(text)
        raw_col = str(col).strip()
        if raw_col == "FILTER" or _clean_header(col) == "vcffilter":
            vcf_semantics = True
    return "; ".join(values), "vcf" if vcf_semantics else "generic"


def _record_from_text_line(
    line: str,
    *,
    source_path: str = "",
    source_format: str = "inline",
    row_index: int | None = None,
) -> VariantRecord | None:
    text = str(line or "").strip()
    if not text:
        return None
    gene = _clean_gene(text)
    if not gene:
        return None
    variant_type = classify_variant_type(text)
    _reject_sample_genomic_state(text, variant_type)
    variant = text
    return VariantRecord(
        gene=gene,
        variant=variant,
        variant_type=variant_type,
        source_path=source_path,
        source_format=source_format,
        row_index=row_index,
        raw_name=text,
    )


def _records_from_dataframe(
    df: pd.DataFrame,
    *,
    source_path: str,
    source_format: str,
    genome_build: object = "",
    ensembl_release: int | None = None,
) -> list[VariantRecord]:
    gene_col = _find_column(df.columns, _GENE_COLUMNS)
    variant_col = _find_column(df.columns, _VARIANT_COLUMNS)
    build_col = _find_column(df.columns, _GENOME_BUILD_COLUMNS)
    release_col = _find_column(df.columns, _ENSEMBL_RELEASE_COLUMNS)
    caller_version_col = _find_column(df.columns, _CALLER_VERSION_COLUMNS)
    records: list[VariantRecord] = []
    for idx, row in df.iterrows():
        raw_gene = _text(row.get(gene_col)) if gene_col is not None else ""
        gene = _clean_gene(raw_gene)
        variant = _text(row.get(variant_col)) if variant_col is not None else ""
        if not gene and variant:
            gene = _clean_gene(variant)
        if not gene:
            continue
        if not variant:
            variant = " ".join(
                part
                for part in (_text(value) for value in row.values)
                if part
            ).strip()
        # Keep the complete structured gene cell for connected fusion-pair
        # recovery while retaining ``gene`` as the normalized primary symbol.
        full_text = f"{raw_gene or gene} {variant}".strip()
        variant_type = classify_variant_type(full_text)
        _reject_sample_genomic_state(full_text, variant_type)
        if variant_type == "unknown" and _has_unambiguous_fusion_pair(raw_gene):
            variant_type = "fusion"
        filter_status, filter_semantics = _filter_status_from_row(
            row,
            source_path=source_path,
        )
        coordinate = _coordinate_from_row(row)
        row_release = (
            _positive_integer(
                row.get(release_col),
                field_name="Ensembl release",
            )
            if release_col is not None
            else None
        )
        active_release = row_release or ensembl_release
        records.append(
            VariantRecord.from_mapping(
                {
                    "gene": gene,
                    "variant": variant or full_text,
                    "variant_type": variant_type,
                    "source_path": source_path,
                    "row_index": int(idx),
                    "confidence": _confidence_from_row(row),
                    "result_status": _result_status_from_row(row),
                    "filter_status": filter_status,
                    "filter_semantics": filter_semantics,
                    "support": _support_from_row(row),
                    "raw_name": full_text,
                    "source_format": source_format,
                    "caller_version": (
                        _text(row.get(caller_version_col))
                        if caller_version_col is not None
                        else ""
                    ),
                    "genome_build": (
                        row.get(build_col) if build_col is not None else ""
                    ),
                    "ensembl_release": active_release,
                    "coordinates": (coordinate,) if coordinate else (),
                },
                genome_build=genome_build,
            )
        )
    return records


def _read_variant_table(path: Path) -> pd.DataFrame:
    source_format = _source_format(path)
    if source_format in {"vcf", "maf"}:
        raise ValueError(
            f"{source_format.upper()} input requires its source-specific adapter; "
            "the generic variant-table parser will not guess its semantics"
        )
    if source_format in {"tsv", "text"}:
        try:
            return pd.read_csv(path, sep="\t", low_memory=False, comment="#")
        except Exception:
            return pd.read_csv(path, sep=None, engine="python", comment="#")
    if source_format == "excel":
        return pd.read_excel(path)
    if source_format == "json":
        return pd.DataFrame(_read_json_rows(path))
    return pd.read_csv(path, low_memory=False)


def _records_from_text(
    text: str,
    *,
    source_path: str = "",
    source_format: str = "inline",
) -> list[VariantRecord]:
    records: list[VariantRecord] = []
    for idx, line in enumerate(str(text or "").splitlines()):
        record = _record_from_text_line(
            line,
            source_path=source_path,
            source_format=source_format,
            row_index=idx,
        )
        if record is not None:
            records.append(record)
    if not records:
        record = _record_from_text_line(
            text,
            source_path=source_path,
            source_format=source_format,
            row_index=None,
        )
        if record is not None:
            records.append(record)
    return records


def parse_variant_file(
    path: str | Path,
    *,
    genome_build: object = "",
    ensembl_release: int | None = None,
) -> list[VariantRecord]:
    """Parse a generic variant table or text file into normalized records.

    VCF and MAF are intentionally rejected until their dedicated adapters are
    implemented. Treating either standard as an arbitrary table silently loses
    semantics that are required for safe therapy evidence.
    """
    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"Variant evidence file not found: {target}")
    if not target.is_file():
        raise ValueError(f"Variant evidence path is not a file: {target}")
    source_format = _source_format(target)
    if source_format in {"vcf", "maf"}:
        raise ValueError(
            f"{source_format.upper()} input requires its source-specific adapter; "
            "the generic variant parser will not guess its semantics"
        )
    if source_format == "json":
        typed_records = _records_from_typed_json(
            _read_json_rows(target),
            source_path=str(target),
            genome_build=genome_build,
            ensembl_release=ensembl_release,
        )
        if typed_records is not None:
            validate_variant_genome_builds(typed_records)
            return typed_records
    table_error: Exception | None = None
    text_error: Exception | None = None
    try:
        df = _read_variant_table(target)
    except Exception as exc:  # noqa: BLE001
        table_error = exc
    else:
        records = _records_from_dataframe(
            df,
            source_path=str(target),
            source_format=source_format,
            genome_build=genome_build,
            ensembl_release=ensembl_release,
        )
        if records:
            validate_variant_genome_builds(records)
            return records
    if source_format not in {"text", "table"}:
        details = f": {table_error}" if table_error else ""
        raise ValueError(
            f"No recognizable variant records in {source_format.upper()} file "
            f"{target}{details}"
        )
    try:
        text_records = _records_from_text(
            _read_text(target),
            source_path=str(target),
            source_format="text",
        )
        if text_records:
            return text_records
        text_error = ValueError("no recognizable gene-bearing variant records")
    except Exception as exc:  # noqa: BLE001
        text_error = exc
    details = []
    if table_error:
        details.append(f"table parser: {table_error}")
    if text_error:
        details.append(f"text parser: {text_error}")
    raise ValueError(
        f"Could not read variant evidence file {target}: " + "; ".join(details)
    )


def _looks_like_path(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    if candidate.startswith(("~", ".", "/")):
        return True
    if _PATH_SUFFIX_RE.search(candidate):
        return True
    if re.search(r"\s", candidate):
        return False
    return "/" in candidate or "\\" in candidate


def parse_variant_inputs(
    inputs: object,
    *,
    genome_build: object = "",
    ensembl_release: int | None = None,
) -> list[VariantRecord]:
    """Parse files or inline variant strings, deduplicating records."""
    records: list[VariantRecord] = []
    seen: set[tuple[object, ...]] = set()
    for raw in split_variant_inputs(inputs):
        target = Path(str(raw)).expanduser()
        if target.exists() or _looks_like_path(str(raw)):
            parsed = parse_variant_file(
                target,
                genome_build=genome_build,
                ensembl_release=ensembl_release,
            )
        else:
            parsed = _records_from_text(
                str(raw),
                source_path="",
                source_format="inline",
            )
        for record in parsed:
            if record.key in seen:
                continue
            seen.add(record.key)
            records.append(record)
    validate_variant_genome_builds(records)
    return records
