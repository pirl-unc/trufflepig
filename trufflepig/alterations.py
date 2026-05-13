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
    text = str(value or "").strip()
    if text.lower() == "nan":
        return ""
    return text


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
    if re.search(r"\b(fusion|rearrang|translocation)\b", low):
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
        gene = _clean_gene(row.get(gene_col)) if gene_col is not None else ""
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
        full_text = f"{gene} {alteration}".strip()
        records.append(
            AlterationRecord(
                gene=gene,
                alteration=alteration or full_text,
                alteration_type=classify_alteration_type(full_text),
                source_path=source_path,
                row_index=int(idx),
                confidence=_confidence_from_row(row),
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
