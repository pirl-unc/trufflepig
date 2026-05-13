"""Permissive fusion-call parsing for analyze.

Fusion callers use wildly different headers. This module accepts common
tabular caller outputs (Tempus-like, Arriba, STAR-Fusion, FusionCatcher,
DRAGEN-style) and simple text lists such as ``BRD4--NUTM1``. Parsed
records are evidence, not diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from typing import Any, Iterable

import pandas as pd


_GENE_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,10}(?:-[A-Z0-9]{1,6})?\b")
_PAIR_RE = re.compile(
    r"\b(?P<a>[A-Z][A-Z0-9]{1,10}(?:-[A-Z0-9]{1,6})?)"
    r"\s*(?:::|--|[-_/|])\s*"
    r"(?P<b>[A-Z][A-Z0-9]{1,10}(?:-[A-Z0-9]{1,6})?)\b"
)
_NON_GENE_TOKENS = {
    "CHR",
    "DNA",
    "EXON",
    "FUSION",
    "GENE",
    "HLA",
    "IN",
    "OUT",
    "RNA",
    "TRUE",
    "FALSE",
}


@dataclass(frozen=True)
class FusionRecord:
    """One normalized fusion call from a loose input file.

    ``gene_a`` is the reported 5-prime/upstream partner when the caller
    provides orientation, and ``gene_b`` is the reported 3-prime/downstream
    partner. For plain text lists where orientation is not explicit, the
    reported left/right order is preserved and downstream rules may match
    either direction while still documenting the expected 5-prime/3-prime
    orientation.
    """

    gene_a: str
    gene_b: str
    source_path: str = ""
    row_index: int | None = None
    effect: str = ""
    frame: str = ""
    caller: str = ""
    confidence: str = ""
    reportable: str = ""
    support_total: float | None = None
    support: dict[str, float] = field(default_factory=dict)
    raw_name: str = ""
    orientation: str = ""

    @property
    def pair_key(self) -> str:
        return "::".join(sorted((self.gene_a, self.gene_b)))

    def public_dict(self) -> dict[str, Any]:
        return {
            "gene_a": self.gene_a,
            "gene_b": self.gene_b,
            "pair": f"{self.gene_a}--{self.gene_b}",
            "pair_key": self.pair_key,
            "source_path": self.source_path,
            "row_index": self.row_index,
            "effect": self.effect,
            "frame": self.frame,
            "caller": self.caller,
            "confidence": self.confidence,
            "reportable": self.reportable,
            "support_total": self.support_total,
            "support": dict(self.support),
            "raw_name": self.raw_name,
            "orientation": self.orientation,
        }


def split_fusion_paths(value: object) -> list[str]:
    """Parse comma/semicolon-separated fusion input paths."""
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        if not text:
            return []
        return [part.strip() for part in re.split(r"[,;]", text) if part.strip()]
    if isinstance(value, Iterable):
        paths: list[str] = []
        for item in value:
            paths.extend(split_fusion_paths(item))
        return paths
    return [str(value)]


def _clean_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


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


_GENE_A_COLUMNS = {
    "gene5",
    "5gene",
    "5primegene",
    "fiveprimegene",
    "gene5prime",
    "gene15fusionpartner",
    "gene15primefusionpartner",
    "gene1fiveprimefusionpartner",
    "fiveprime",
    "gene1",
    "gene1fusionpartner",
    "genea",
    "leftgene",
    "upstreamgene",
    "partner1",
    "fusiongene1",
    "breakend1gene",
    "gene1symbol",
    "geneasymbol",
}
_GENE_B_COLUMNS = {
    "gene3",
    "3gene",
    "3primegene",
    "threeprimegene",
    "gene3prime",
    "gene23fusionpartner",
    "gene23primefusionpartner",
    "gene2threeprimefusionpartner",
    "threeprime",
    "gene2",
    "gene2fusionpartner",
    "geneb",
    "rightgene",
    "downstreamgene",
    "partner2",
    "fusiongene2",
    "breakend2gene",
    "gene2symbol",
    "genebsymbol",
}
_PAIR_COLUMNS = {
    "fusion",
    "fusionname",
    "fusiongenes",
    "fusiongene",
    "fusionpair",
    "genepair",
    "name",
    "event",
}
_EFFECT_COLUMNS = {
    "effect",
    "predictedeffect",
    "predictedfusioneffect",
    "annotation",
    "annots",
}
_FRAME_COLUMNS = {"frame", "readingframe", "inframe", "frameshift"}
_CALLER_COLUMNS = {"fusioncaller", "caller", "tool", "program"}
_CONFIDENCE_COLUMNS = {"confidence", "filter", "classification", "status"}
_REPORTABLE_COLUMNS = {"reportable", "whitelist", "somatic", "oncogenic"}
_SUPPORT_COLUMNS = {
    "directsupport",
    "discordantsupport",
    "totalsupport",
    "splitreads",
    "splitread",
    "splitreads1",
    "splitreads2",
    "junctionreadcount",
    "spanningfragcount",
    "spanningfrags",
    "spanningreads",
    "spanningpairs",
    "totalspanningreads",
    "discordantmates",
    "readcount",
    "reads",
    "ffpm",
}
_TOTAL_SUPPORT_COLUMNS = {
    "totalsupport",
    "totalreads",
    "totalspanningreads",
    "readcount",
}


def _numeric(value: object) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    if result != result:
        return None
    return result


def _text(value: object) -> str:
    text = str(value or "").strip()
    if text.lower() == "nan":
        return ""
    return text


def _pair_from_text(value: object) -> tuple[str, str] | None:
    text = str(value or "").upper()
    for match in _PAIR_RE.finditer(text):
        a = _clean_gene(match.group("a"))
        b = _clean_gene(match.group("b"))
        if a and b and a != b:
            return a, b
    genes = [_clean_gene(match.group(0)) for match in _GENE_RE.finditer(text)]
    genes = [gene for gene in genes if gene]
    if len(genes) >= 2 and genes[0] != genes[1]:
        return genes[0], genes[1]
    return None


def _support_from_row(row) -> tuple[dict[str, float], float | None]:
    support: dict[str, float] = {}
    explicit_total: float | None = None
    for col, value in row.items():
        clean = _clean_header(col)
        if clean not in _SUPPORT_COLUMNS:
            continue
        num = _numeric(value)
        if num is None:
            continue
        support[str(col)] = num
        if clean in _TOTAL_SUPPORT_COLUMNS and explicit_total is None:
            explicit_total = num
    if explicit_total is not None:
        return support, explicit_total
    counted = [
        value
        for key, value in support.items()
        if "ffpm" not in _clean_header(key)
    ]
    if counted:
        return support, float(sum(counted))
    return support, None


def _row_value(row, candidates: set[str]) -> str:
    col = _find_column(row.index, candidates)
    return _text(row.get(col)) if col is not None else ""


def _records_from_dataframe(df: pd.DataFrame, *, source_path: str) -> list[FusionRecord]:
    gene_a_col = _find_column(df.columns, _GENE_A_COLUMNS)
    gene_b_col = _find_column(df.columns, _GENE_B_COLUMNS)
    pair_col = _find_column(df.columns, _PAIR_COLUMNS)
    base_orientation = (
        "5prime_3prime" if gene_a_col is not None and gene_b_col is not None else ""
    )
    records: list[FusionRecord] = []
    for idx, row in df.iterrows():
        orientation = base_orientation
        gene_a = _clean_gene(row.get(gene_a_col)) if gene_a_col is not None else ""
        gene_b = _clean_gene(row.get(gene_b_col)) if gene_b_col is not None else ""
        raw_name = ""
        if (not gene_a or not gene_b) and pair_col is not None:
            raw_name = _text(row.get(pair_col))
            pair = _pair_from_text(raw_name)
            if pair:
                gene_a, gene_b = pair
                if not orientation:
                    orientation = "reported_order"
        if not raw_name:
            raw_name = f"{gene_a}--{gene_b}" if gene_a and gene_b else ""
        if not gene_a or not gene_b or gene_a == gene_b:
            continue
        support, support_total = _support_from_row(row)
        records.append(
            FusionRecord(
                gene_a=gene_a,
                gene_b=gene_b,
                source_path=source_path,
                row_index=int(idx),
                effect=_row_value(row, _EFFECT_COLUMNS),
                frame=_row_value(row, _FRAME_COLUMNS),
                caller=_row_value(row, _CALLER_COLUMNS),
                confidence=_row_value(row, _CONFIDENCE_COLUMNS),
                reportable=_row_value(row, _REPORTABLE_COLUMNS),
                support_total=support_total,
                support=support,
                raw_name=raw_name,
                orientation=orientation or "reported_order",
            )
        )
    return records


def _read_fusion_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".sf", ".txt"}:
        try:
            return pd.read_csv(path, sep="\t", low_memory=False)
        except Exception:
            try:
                return pd.read_csv(path, sep="\t", comment="#", low_memory=False)
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
            for key in ("fusions", "fusion_calls", "records", "data"):
                if isinstance(payload.get(key), list):
                    return pd.DataFrame(payload[key])
            return pd.DataFrame([payload])
        return pd.DataFrame(payload)
    return pd.read_csv(path, low_memory=False)


def _records_from_text(path: Path) -> list[FusionRecord]:
    records: list[FusionRecord] = []
    for idx, line in enumerate(path.read_text(errors="ignore").splitlines()):
        pair = _pair_from_text(line)
        if not pair:
            continue
        records.append(
            FusionRecord(
                gene_a=pair[0],
                gene_b=pair[1],
                source_path=str(path),
                row_index=idx,
                raw_name=line.strip(),
                orientation="reported_order",
            )
        )
    return records


def parse_fusion_file(path: str | Path) -> list[FusionRecord]:
    """Parse one fusion evidence file into normalized records."""
    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"Fusion evidence file not found: {target}")
    if not target.is_file():
        raise ValueError(f"Fusion evidence path is not a file: {target}")
    table_error: Exception | None = None
    text_error: Exception | None = None
    try:
        df = _read_fusion_table(target)
        records = _records_from_dataframe(df, source_path=str(target))
        if records:
            return records
    except Exception as exc:  # noqa: BLE001
        table_error = exc
    try:
        return _records_from_text(target)
    except Exception as exc:  # noqa: BLE001
        text_error = exc
    if table_error or text_error:
        details = []
        if table_error:
            details.append(f"table parser: {table_error}")
        if text_error:
            details.append(f"text parser: {text_error}")
        raise ValueError(
            f"Could not read fusion evidence file {target}: " + "; ".join(details)
        )
    return []


def parse_fusion_files(paths: object) -> list[FusionRecord]:
    """Parse one or more fusion files, deduplicating by pair/source row."""
    records: list[FusionRecord] = []
    seen: set[tuple[str, str, int | None]] = set()
    for path in split_fusion_paths(paths):
        for record in parse_fusion_file(path):
            key = (record.pair_key, record.source_path, record.row_index)
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    return records
