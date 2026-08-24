"""Permissive alteration parsing for analyze.

This module accepts lightweight mutation/CNV/structural-variant evidence from
loose text, CSV/TSV/Excel, and JSON-like exports. Parsed records are evidence
objects; downstream report logic decides whether they support a therapy row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
    r"(?:likely\s+)?benign|low[- ]?qual(?:ity)?)"
)
_NON_POSITIVE_RESULT_PATTERN = rf"(?:{_NON_POSITIVE_EVENT_PATTERN}|unknown)"
_NON_POSITIVE_EVENT_RE = re.compile(
    rf"\b{_NON_POSITIVE_EVENT_PATTERN}\b",
    re.IGNORECASE,
)
_NEGATIVE_RESULT_RE = re.compile(
    rf"^\s*(?:(?:false|no|neg|{_NON_POSITIVE_RESULT_PATTERN})\b|"
    r"0(?:\.0+)?(?![\d.]))",
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


@dataclass(frozen=True)
class AlterationRecord:
    """One normalized non-expression alteration call.

    ``alteration_type`` is deliberately coarse. It is meant for report-gating
    compatibility (mutation vs fusion vs amplification vs KDD), not for
    replacing a clinical variant annotation.
    """

    gene: str
    alteration: str = ""
    alteration_type: str = "unknown"
    source_path: str = ""
    row_index: int | None = None
    confidence: str = ""
    support: dict[str, float] = field(default_factory=dict)
    raw_name: str = ""
    result_status: str = ""
    filter_status: str = ""
    filter_semantics: str = "generic"

    @property
    def key(self) -> tuple[str, str, str, str, int | None]:
        return (
            self.gene,
            self.alteration_type,
            self.alteration.lower(),
            self.source_path,
            self.row_index,
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "gene": self.gene,
            "alteration": self.alteration,
            "alteration_type": self.alteration_type,
            "source_path": self.source_path,
            "row_index": self.row_index,
            "confidence": self.confidence,
            "result_status": self.result_status,
            "filter_status": self.filter_status,
            "filter_semantics": self.filter_semantics,
            "support": dict(self.support),
            "raw_name": self.raw_name,
        }


def split_alteration_inputs(value: object) -> list[str]:
    """Parse semicolon/newline-separated alteration inputs.

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
            inputs.extend(split_alteration_inputs(item))
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
    return any(_NEGATIVE_RESULT_RE.match(part) for part in parts)


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
    return any(_NEGATIVE_RESULT_RE.match(part) for part in parts)


def alteration_record_passes_assay_filters(record: object) -> bool:
    """Whether a normalized call passes structured result and filter fields.

    This public gate is intentionally disease- and therapy-independent. Free-
    text event negation remains gene-specific and is handled by
    :func:`alteration_record_gene_is_negated`.
    """
    if not hasattr(record, "get"):
        return False
    if any(
        _result_status_is_non_positive(record.get(key))
        for key in ("result_status", "result", "status")
    ):
        return False

    filter_semantics = str(record.get("filter_semantics") or "").lower()
    stored_filter_is_vcf = filter_semantics == "vcf" or (
        not filter_semantics and bool(_text(record.get("filter_status")))
    )
    if _filter_status_is_failed(
        record.get("filter_status"),
        vcf_semantics=stored_filter_is_vcf,
    ):
        return False
    # Preserve sensible behavior for callers that pass unnormalized records.
    # Exact uppercase FILTER follows VCF semantics; a generic ``filter`` key
    # rejects only explicit failure vocabulary.
    if _filter_status_is_failed(record.get("FILTER"), vcf_semantics=True):
        return False
    if _filter_status_is_failed(record.get("filter"), vcf_semantics=False):
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


def alteration_record_gene_is_negated(record: object, gene: str) -> bool:
    """Whether prose explicitly negates this gene's molecular finding."""
    if not hasattr(record, "get"):
        return False
    wanted = str(gene or "").strip().upper()
    if not wanted:
        return False
    if not alteration_record_passes_assay_filters(record):
        return True
    event_texts = [
        str(record.get(key) or "")
        for key in ("alteration", "raw_name", "confidence")
        if str(record.get(key) or "").strip()
    ]
    escaped = re.escape(wanted)
    # A negative outcome applies to the complete alteration clause, not only
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
    for key in ("gene", "alteration", "raw_name"):
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
            if any(alteration_record_gene_is_negated(record, gene) for gene in genes):
                continue
            return genes
    return ()


def alteration_record_genes(record: object) -> tuple[str, ...]:
    """Return the structured gene and any explicit fusion partner.

    ``AlterationRecord.gene`` remains the primary gene for backwards
    compatibility, but a fusion is intrinsically a two-gene event.  Loose text
    inputs such as ``ETV6-NTRK3 fusion`` historically retained only ``ETV6``;
    target matching therefore missed the therapeutically relevant ``NTRK3``
    partner. Recover a partner only from a connected pair such as
    ``ETV6-NTRK3`` or ``ETV6::NTRK3``. Other genes mentioned in commentary are
    not part of the event, and explicitly negated findings are excluded.
    """
    if not hasattr(record, "get"):
        return ()
    pair = _explicit_fusion_pair(record)
    if pair:
        return pair
    alteration_type = str(record.get("alteration_type") or "").strip().lower()
    event_text = " ".join(
        str(record.get(key) or "")
        for key in ("alteration", "raw_name")
    ).lower()
    fusion_like = (
        alteration_type == "fusion"
        or classify_alteration_type(event_text) == "fusion"
    )
    primary_text = str(record.get("gene") or "").upper()
    primary = _clean_gene(primary_text)
    if not primary:
        return ()
    if fusion_like and _FUSION_PAIR_RE.fullmatch(primary_text.strip()):
        # A connected pair was present but rejected above (for example because
        # the event was explicitly reported as not detected).
        return ()
    if alteration_record_gene_is_negated(record, primary):
        return ()
    return (primary,)


def _record_dict(record: object) -> dict[str, Any] | None:
    if hasattr(record, "public_dict"):
        public = record.public_dict()
        return dict(public) if hasattr(public, "get") else None
    if hasattr(record, "get"):
        return dict(record)
    return None


def _fusion_record_as_alteration(record: object) -> dict[str, Any] | None:
    """Normalize a dedicated fusion call onto the alteration-record contract."""
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
            "pair": pair,
            "alteration": f"{pair} fusion",
            "alteration_type": "fusion",
            "raw_name": raw_name,
            "result_status": _text(source.get("result_status"))
            or _text(source.get("reportable")),
            "filter_status": _text(source.get("filter_status"))
            or _text(source.get("confidence")),
            "filter_semantics": _text(source.get("filter_semantics")) or "generic",
            "evidence_source_type": "fusion",
            "evidence_source_types": ["fusion"],
        }
    )
    return normalized


def _molecular_record_identity(record: dict[str, Any]) -> tuple[object, ...]:
    alteration_type = _text(record.get("alteration_type")).lower() or "unknown"
    if alteration_type == "fusion":
        for key in ("pair", "gene", "alteration", "raw_name"):
            match = _FUSION_PAIR_RE.search(_text(record.get(key)).upper())
            if match:
                pair = (match.group(1), match.group(2))
                if _valid_fusion_pair_genes(pair):
                    return ("fusion", *sorted(pair))
        gene_a = _clean_gene(record.get("gene_a"))
        gene_b = _clean_gene(record.get("gene_b"))
        if gene_a and gene_b and gene_a != gene_b:
            return ("fusion", *sorted((gene_a, gene_b)))
    gene = _clean_gene(record.get("gene"))
    if alteration_type == "fusion" and gene:
        return ("fusion", gene)
    # These alteration classes are the molecular eligibility event.  Repeated
    # exports may phrase the result differently (for example ``amplification``
    # versus ``amplification not detected``), but a negative result should
    # reconcile only with the same class—not erase a distinct event involving
    # the same gene such as an EGFR kinase-domain duplication.
    if alteration_type in {
        "amplification",
        "internal_tandem_duplication",
        "kdd",
        "loss",
        "msi_high",
    }:
        return (alteration_type, gene)
    alteration = re.sub(
        r"\s+",
        " ",
        _text(record.get("alteration") or record.get("raw_name")).lower(),
    )
    # Outcome wording is not part of the biological event identity.  Removing
    # it lets ``BRAF V600E`` and ``BRAF V600E not detected`` reconcile as one
    # conflicting assay event while distinct variants remain separate.
    alteration = _NON_POSITIVE_EVENT_RE.sub(" ", alteration)
    alteration = _EXPLICIT_UNKNOWN_OUTCOME_RE.sub(" ", alteration)
    alteration = re.sub(r"[^a-z0-9]+", " ", alteration).strip()
    return (alteration_type, gene, alteration)


def _record_supports_identity(
    record: dict[str, Any],
    identity: tuple[object, ...],
) -> bool:
    identity_genes = identity[1:] if identity[0] == "fusion" else identity[1:2]
    genes = [str(value) for value in identity_genes if str(value)]
    return bool(genes) and all(
        not alteration_record_gene_is_negated(record, gene) for gene in genes
    )


def _source_types(record: dict[str, Any]) -> set[str]:
    values = record.get("evidence_source_types") or ()
    if isinstance(values, str):
        values = (values,)
    return {str(value) for value in values if str(value)}


def molecular_evidence_records(analysis: object) -> list[dict[str, Any]]:
    """Return one normalized view of supplied alteration and fusion evidence.

    The dedicated fusion interface remains the authoritative source for fusion
    classification and fusion-specific reporting. This view lets generic
    molecular consumers—therapy eligibility and pathway-source reasoning—use
    the same record contract regardless of which CLI/API input supplied an
    event. The same biological event supplied through both interfaces is
    retained once and carries both source types, so it cannot become two votes.
    """
    if not isinstance(analysis, dict):
        return []
    normalized: list[dict[str, Any]] = []
    by_identity: dict[tuple[object, ...], dict[str, Any]] = {}
    sources = (
        ("alteration", analysis.get("alteration_records") or ()),
        ("fusion", analysis.get("fusion_records") or ()),
    )
    for source_type, records in sources:
        for record in records:
            item = (
                _fusion_record_as_alteration(record)
                if source_type == "fusion"
                else _record_dict(record)
            )
            if item is None:
                continue
            if source_type == "alteration":
                item.setdefault("evidence_source_type", "alteration")
                item.setdefault("evidence_source_types", ["alteration"])
            identity = _molecular_record_identity(item)
            existing = by_identity.get(identity)
            if existing is None:
                by_identity[identity] = item
                normalized.append(item)
                continue
            existing_sources = _source_types(existing)
            existing_sources.update(_source_types(item))
            existing["evidence_source_types"] = sorted(existing_sources)
            if _record_supports_identity(
                existing,
                identity,
            ) != _record_supports_identity(item, identity):
                existing["result_status"] = (
                    "inconclusive: conflicting supplied records"
                )
                existing["evidence_conflict"] = True
    return normalized


def molecular_evidence_for_gene(analysis: object, gene: str) -> list[dict[str, Any]]:
    """Return positive, non-conflicting supplied evidence for one gene.

    :func:`molecular_evidence_records` reconciles positive and non-positive
    reports for the same molecular-event identity. Distinct events involving
    one gene remain independent: a negative EGFR KDD assay must not suppress a
    separately reported EGFR amplification, for example.
    """
    wanted = _clean_gene(gene)
    if not wanted:
        return []
    records = molecular_evidence_records(analysis)
    return [
        record
        for record in records
        if wanted in alteration_record_genes(record)
    ]


def _valid_fusion_pair_genes(genes: tuple[str, str]) -> bool:
    return all(len(gene) >= 3 for gene in genes) and not any(
        gene in _NON_GENE_TOKENS or gene.startswith("CHR") for gene in genes
    )


def _has_connected_fusion_pair(value: object) -> bool:
    text = _text(value).upper()
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
_ALTERATION_COLUMNS = {
    "alteration",
    "variant",
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
    except Exception:
        return None
    if result != result:
        return None
    return result


def _support_from_row(row) -> dict[str, float]:
    support: dict[str, float] = {}
    for col, value in row.items():
        if _clean_header(col) not in _SUPPORT_COLUMNS:
            continue
        num = _numeric(value)
        if num is not None:
            support[str(col)] = num
    return support


def classify_alteration_type(text: object) -> str:
    """Return a coarse alteration class from loose text."""
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
    if re.search(r"\b(msi[- ]?h|dmmr|deficient\s+mmr)\b", low):
        return "msi_high"
    if re.search(r"\b(v600|g12|g13|q61|l858r|t790m|exon\s*\d+|mutat|snv|indel|variant)\b", low):
        return "mutation"
    return "unknown"


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
    row_index: int | None = None,
) -> AlterationRecord | None:
    text = str(line or "").strip()
    if not text:
        return None
    gene = _clean_gene(text)
    if not gene:
        return None
    alteration_type = classify_alteration_type(text)
    alteration = text
    return AlterationRecord(
        gene=gene,
        alteration=alteration,
        alteration_type=alteration_type,
        source_path=source_path,
        row_index=row_index,
        raw_name=text,
    )


def _records_from_dataframe(df: pd.DataFrame, *, source_path: str) -> list[AlterationRecord]:
    gene_col = _find_column(df.columns, _GENE_COLUMNS)
    alteration_col = _find_column(df.columns, _ALTERATION_COLUMNS)
    records: list[AlterationRecord] = []
    for idx, row in df.iterrows():
        raw_gene = _text(row.get(gene_col)) if gene_col is not None else ""
        gene = _clean_gene(raw_gene)
        alteration = _text(row.get(alteration_col)) if alteration_col is not None else ""
        raw_parts = [alteration]
        if not gene:
            raw_parts.extend(_text(value) for value in row.values)
            gene = _clean_gene(" ".join(raw_parts))
        if not alteration:
            alteration = " ".join(
                part
                for part in (_text(value) for value in row.values)
                if part
            ).strip()
        if not gene:
            continue
        # Keep the complete structured gene cell for connected fusion-pair
        # recovery while retaining ``gene`` as the normalized primary symbol.
        full_text = f"{raw_gene or gene} {alteration}".strip()
        alteration_type = classify_alteration_type(full_text)
        if alteration_type == "unknown" and _has_connected_fusion_pair(raw_gene):
            alteration_type = "fusion"
        filter_status, filter_semantics = _filter_status_from_row(
            row,
            source_path=source_path,
        )
        records.append(
            AlterationRecord(
                gene=gene,
                alteration=alteration or full_text,
                alteration_type=alteration_type,
                source_path=source_path,
                row_index=int(idx),
                confidence=_confidence_from_row(row),
                result_status=_result_status_from_row(row),
                filter_status=filter_status,
                filter_semantics=filter_semantics,
                support=_support_from_row(row),
                raw_name=full_text,
            )
        )
    return records


def _read_alteration_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt", ".maf", ".vcf"}:
        try:
            return pd.read_csv(path, sep="\t", low_memory=False, comment="#")
        except Exception:
            return pd.read_csv(path, sep=None, engine="python", comment="#")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".json", ".jsonl"}:
        text = path.read_text()
        if suffix == ".jsonl":
            return pd.DataFrame(json.loads(line) for line in text.splitlines() if line.strip())
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("alterations", "variants", "mutations", "records", "data"):
                if isinstance(payload.get(key), list):
                    return pd.DataFrame(payload[key])
            return pd.DataFrame([payload])
        return pd.DataFrame(payload)
    return pd.read_csv(path, low_memory=False)


def _records_from_text(text: str, *, source_path: str = "") -> list[AlterationRecord]:
    records: list[AlterationRecord] = []
    for idx, line in enumerate(str(text or "").splitlines()):
        record = _record_from_text_line(line, source_path=source_path, row_index=idx)
        if record is not None:
            records.append(record)
    if not records:
        record = _record_from_text_line(text, source_path=source_path, row_index=None)
        if record is not None:
            records.append(record)
    return records


def parse_alteration_file(path: str | Path) -> list[AlterationRecord]:
    """Parse one alteration evidence file into normalized records."""
    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"Alteration evidence file not found: {target}")
    if not target.is_file():
        raise ValueError(f"Alteration evidence path is not a file: {target}")
    table_error: Exception | None = None
    text_error: Exception | None = None
    try:
        df = _read_alteration_table(target)
        records = _records_from_dataframe(df, source_path=str(target))
        if records:
            return records
    except Exception as exc:  # noqa: BLE001
        table_error = exc
    try:
        return _records_from_text(target.read_text(errors="ignore"), source_path=str(target))
    except Exception as exc:  # noqa: BLE001
        text_error = exc
    details = []
    if table_error:
        details.append(f"table parser: {table_error}")
    if text_error:
        details.append(f"text parser: {text_error}")
    raise ValueError(
        f"Could not read alteration evidence file {target}: " + "; ".join(details)
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


def parse_alteration_inputs(inputs: object) -> list[AlterationRecord]:
    """Parse files or inline alteration strings, deduplicating records."""
    records: list[AlterationRecord] = []
    seen: set[tuple[str, str, str, str, int | None]] = set()
    for raw in split_alteration_inputs(inputs):
        target = Path(str(raw)).expanduser()
        if target.exists() or _looks_like_path(str(raw)):
            parsed = parse_alteration_file(target)
        else:
            parsed = _records_from_text(str(raw), source_path="")
        for record in parsed:
            if record.key in seen:
                continue
            seen.add(record.key)
            records.append(record)
    return records
