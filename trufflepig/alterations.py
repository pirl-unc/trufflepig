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
_NEGATIVE_CALL_RE = re.compile(
    r"\b(?:not\s+detected|negative|absent|wild[- ]?type|not\s+present)\b",
    re.IGNORECASE,
)
_NEGATIVE_RESULT_RE = re.compile(
    r"^\s*(?:(?:false|no|not\s+detected|negative|absent|"
    r"wild[- ]?type|not\s+present)\b|0(?:\.0+)?(?![\d.]))",
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


def _result_status_is_negative(value: object) -> bool:
    """Whether a structured assay result explicitly reports no event."""
    return any(
        _NEGATIVE_RESULT_RE.match(part)
        for part in re.split(r"[;,|]", _text(value))
        if part.strip()
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
    if any(
        _result_status_is_negative(record.get(key))
        for key in ("result_status", "result", "status")
    ):
        return True
    text = " ".join(
        str(record.get(key) or "")
        for key in ("alteration", "raw_name", "confidence")
    )
    escaped = re.escape(wanted)
    return bool(
        re.search(
            rf"\b{escaped}\b(?:\s+{_FUSION_EVENT_PATTERN})?"
            rf"\s*(?:[:=]|[-–—])?\s*(?:is\s+)?{_NEGATIVE_CALL_RE.pattern}",
            text,
            re.IGNORECASE,
        )
        or re.search(
            rf"\b(?:no|without)\s+(?:evidence\s+of\s+)?(?:an?\s+)?"
            rf"\b{escaped}\b",
            text,
            re.IGNORECASE,
        )
    )


def _explicit_fusion_pair(record: object) -> tuple[str, ...]:
    """Return one connected, non-negated fusion pair from the event fields."""
    for key in ("gene", "alteration", "raw_name"):
        text = str(record.get(key) or "").upper()
        for match in _FUSION_PAIR_RE.finditer(text):
            genes = (match.group(1), match.group(2))
            if not all(len(gene) >= 3 for gene in genes):
                continue
            if any(
                gene in _NON_GENE_TOKENS or gene.startswith("CHR") for gene in genes
            ):
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
            if _NEGATIVE_CALL_RE.search(clause):
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
    alteration_type = str(record.get("alteration_type") or "").strip().lower()
    event_text = " ".join(
        str(record.get(key) or "")
        for key in ("alteration", "raw_name")
    ).lower()
    fusion_like = (
        alteration_type == "fusion"
        or classify_alteration_type(event_text) == "fusion"
    )
    if fusion_like:
        pair = _explicit_fusion_pair(record)
        if pair:
            return pair

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
        records.append(
            AlterationRecord(
                gene=gene,
                alteration=alteration or full_text,
                alteration_type=classify_alteration_type(full_text),
                source_path=source_path,
                row_index=int(idx),
                confidence=_confidence_from_row(row),
                result_status=_result_status_from_row(row),
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
