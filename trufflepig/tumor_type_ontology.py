"""Tumor-type ontology and RNA marker sanity checks.

The cancer-type call can be finer than the closest available expression
reference. This module keeps those relationships explicit and attaches a
small, interpretable marker panel to every registry cancer type.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Mapping


_HIGH_TPM_FLOOR = 2.0
_LOW_TPM_CEILING = 10.0


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _norm_key(value: Any) -> str:
    return _clean(value).lower().replace("-", "_").replace(" ", "_")


def _gene_symbol(value: Any) -> str:
    text = _clean(value).upper()
    if not text or text.endswith("_SYN"):
        return ""
    if any(ch.isspace() for ch in text):
        return ""
    return text


def _unique_genes(genes: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for gene in genes:
        symbol = _gene_symbol(gene)
        if symbol and symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return tuple(out)


@dataclass(frozen=True)
class GeneExpectation:
    symbol: str
    direction: str
    source: str
    inherited_from: str = ""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TumorTypeOntologyEntry:
    code: str
    name: str
    family: str
    parent_code: str
    ancestors: tuple[str, ...]
    expression_reference_code: str
    expression_reference_source: str
    expression_reference_kind: str
    expression_reference_direct: bool
    expression_reference_reason: str = ""
    expected_high: tuple[GeneExpectation, ...] = ()
    expected_low: tuple[GeneExpectation, ...] = ()

    @property
    def expected_high_genes(self) -> tuple[str, ...]:
        return tuple(row.symbol for row in self.expected_high)

    @property
    def expected_low_genes(self) -> tuple[str, ...]:
        return tuple(row.symbol for row in self.expected_low)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["expected_high_genes"] = list(self.expected_high_genes)
        data["expected_low_genes"] = list(self.expected_low_genes)
        return data


_CURATED_HIGH: Mapping[str, tuple[str, ...]] = {
    "BRCA_LumA": ("ESR1", "PGR", "GATA3", "FOXA1", "TFF1", "TFF3"),
    "BRCA_LumB": ("ESR1", "PGR", "GATA3", "FOXA1", "MKI67"),
    "BRCA_HER2": ("ERBB2", "GRB7", "GATA3", "FOXA1"),
    "BRCA_Basal": ("KRT5", "KRT14", "KRT17", "EGFR", "TP63"),
    "BRCA_Normal": ("KRT5", "KRT14", "KRT17", "EPCAM", "KRT8"),
    "HNSC_HPV_pos": ("CDKN2A", "KRT5", "KRT14", "TP63", "SOX2"),
    "HNSC_HPV_neg": ("KRT5", "KRT14", "TP63", "SOX2", "EGFR"),
    "LUAD_EGFR": ("NKX2-1", "NAPSA", "SFTPB", "SFTPC", "SLC34A2", "EGFR"),
    "LUAD_KRAS": ("NKX2-1", "NAPSA", "SFTPB", "SFTPC", "SLC34A2", "KRAS"),
    "LUAD_STK11": ("NKX2-1", "NAPSA", "SFTPB", "SFTPC", "SLC34A2", "STK11"),
    "LAML_ELN_Fav": ("CD34", "ELANE", "MPO", "KIT", "FLT3"),
    "LAML_ELN_Int": ("CD34", "ELANE", "MPO", "KIT", "FLT3"),
    "LAML_ELN_Adv": ("CD34", "ELANE", "MPO", "KIT", "FLT3"),
    "NBL_MYCN_amp": ("MYCN", "PHOX2B", "TH", "B4GALNT1", "ALK"),
    "NBL_MYCN_nonamp": ("PHOX2B", "TH", "B4GALNT1", "ALK"),
    # Rectal NET (pirlygenes >=5.11): a top-level NET with no lineage panel
    # or key-gene coverage upstream, so seed it with the pan-NET core shared
    # by its midgut-NET sibling.
    "REC_NET": ("CHGA", "SYP", "INSM1", "ENO2", "SSTR2"),
    "MBL": ("OTX2", "ATOH1", "SOX2", "MYC", "MYCN"),
    "CHOR": ("TBXT", "KRT8", "KRT18", "EPCAM", "COL2A1"),
    "ATRT": ("VIM", "SOX2", "NES", "LIN28A", "EPCAM"),
    "RB": ("CRX", "VSX2", "RCVRN", "OTX2", "ARR3"),
    "HEPB": ("AFP", "DLK1", "GPC3", "EPCAM", "KRT8", "KRT18"),
    "CLL": ("MS4A1", "CD5", "FCER2", "LEF1", "CD79A"),
    "B_ALL": ("CD19", "CD22", "MS4A1", "PAX5", "CD79A"),
    "NUTM": ("NUTM1", "MYC", "TP63", "SOX2", "KRT5"),
    "WILMS": ("WT1", "SIX1", "SIX2", "PAX2", "PAX8", "IGF2"),
    "OS": ("RUNX2", "COL1A1", "ALPL", "SPP1", "IBSP"),
    "EWS": ("EWSR1", "FLI1", "NKX2-2", "CD99", "CAV1"),
    "CHON": ("COL2A1", "SOX9", "ACAN", "COMP", "COL11A1"),
    "RT": ("EPCAM", "KRT8", "KRT18", "VIM", "SALL4"),
    "RMS_ERMS": ("MYOD1", "MYOG", "DES", "MYF5", "MYF6"),
    "RMS_ARMS": ("PAX3", "PAX7", "FOXO1", "MYOG", "FGFR4"),
    "SARC_UPS": ("VIM", "COL1A1", "COL1A2", "CD44", "PDGFRA"),
    "SARC_MYXFIB": ("COL1A1", "COL1A2", "VIM", "PDGFRA", "MMP2"),
    "SARC_LGFMS": ("MUC4", "FUS", "CREB3L2", "COL1A1", "COL1A2"),
    "SARC_PLEOLPS": ("MDM2", "CDK4", "HMGA2", "PPARG", "VIM"),
    "RMS_PRMS": ("MYOD1", "MYOG", "DES", "MYF5", "MYF6"),
    "RMS_SSRMS": ("MYOD1", "MYOG", "DES", "MYF5", "MYF6"),
}

_CURATED_LOW: Mapping[str, tuple[str, ...]] = {
    "BRCA_LumA": ("KRT5", "KRT14", "ERBB2"),
    "BRCA_LumB": ("KRT5", "KRT14"),
    "BRCA_HER2": ("ESR1", "PGR", "KRT5", "KRT14"),
    "BRCA_Basal": ("ESR1", "PGR", "ERBB2"),
    "HNSC_HPV_neg": ("CDKN2A",),
    "NBL_MYCN_nonamp": ("MYCN",),
    "RT": ("SMARCB1",),
}

_FAMILY_LOW_GENES: Mapping[str, tuple[str, ...]] = {
    "carcinoma": ("PTPRC", "CD3D", "MS4A1", "MYOD1", "MYOG", "DES"),
    "endocrine": ("PTPRC", "CD3D", "MS4A1", "EPCAM", "KRT5", "KRT14"),
    "heme": ("EPCAM", "KRT8", "KRT18", "ACTA2", "DES", "MYOD1"),
    "net": ("PTPRC", "CD3D", "MS4A1", "MYOD1", "MYOG", "DES"),
    "pediatric-bone": ("EPCAM", "KRT8", "KRT18", "PTPRC", "CD3D", "MS4A1"),
    "pediatric-cns": ("EPCAM", "KRT8", "KRT18", "PTPRC", "CD3D", "MS4A1"),
    "pediatric-embryonal": ("PTPRC", "CD3D", "MS4A1", "MYOD1", "MYOG", "DES"),
    "pediatric-net": ("EPCAM", "KRT8", "KRT18", "PTPRC", "CD3D", "MS4A1"),
    "pediatric-soft": ("EPCAM", "KRT8", "KRT18", "PTPRC", "CD3D", "MS4A1"),
    "sarcoma": ("EPCAM", "KRT8", "KRT18", "PTPRC", "CD3D", "MS4A1"),
}

_DEFAULT_LOW_GENES = ("PTPRC", "CD3D", "MS4A1", "EPCAM", "KRT8", "KRT18")


@lru_cache(maxsize=1)
def _registry() -> dict[str, dict[str, Any]]:
    from pirlygenes.gene_sets_cancer import cancer_type_registry

    df = cancer_type_registry().fillna("")
    return {_clean(row["code"]): dict(row) for _, row in df.iterrows()}


@lru_cache(maxsize=1)
def _lineage_genes() -> dict[str, tuple[str, ...]]:
    from pirlygenes.gene_sets_cancer import lineage_genes_by_cancer_type

    return {
        _clean(code): _unique_genes(tuple(genes))
        for code, genes in lineage_genes_by_cancer_type().items()
    }


@lru_cache(maxsize=1)
def _key_gene_rows():
    from pirlygenes import get_data

    return get_data("cancer-key-genes").fillna("")


def _ancestor_codes(code: str) -> tuple[str, ...]:
    registry = _registry()
    out: list[str] = []
    current = _clean(code)
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        parent = _clean(registry.get(current, {}).get("parent_code"))
        if not parent:
            break
        out.append(parent)
        current = parent
    return tuple(out)


def _add_expectations(
    rows: list[GeneExpectation],
    *,
    genes: tuple[str, ...] | list[str],
    direction: str,
    source: str,
    inherited_from: str = "",
    rationale: str = "",
) -> None:
    seen = {row.symbol for row in rows}
    for gene in _unique_genes(tuple(genes)):
        if gene in seen:
            continue
        rows.append(
            GeneExpectation(
                symbol=gene,
                direction=direction,
                source=source,
                inherited_from=inherited_from,
                rationale=rationale,
            )
        )
        seen.add(gene)


def _key_biomarker_genes(code: str, subtype_key: str = "") -> tuple[str, ...]:
    df = _key_gene_rows()
    mask = df["cancer_code"].astype(str).eq(code) & df["role"].astype(str).eq(
        "biomarker"
    )
    if subtype_key:
        mask = mask & df["subtype"].map(_norm_key).eq(_norm_key(subtype_key))
    else:
        mask = mask & df["subtype"].map(_norm_key).eq("")
    return _unique_genes(tuple(df.loc[mask, "symbol"].astype(str)))


def _high_expectations_for_code(
    code: str,
    *,
    inherited_from: str = "",
    visited: frozenset[str] = frozenset(),
) -> tuple[GeneExpectation, ...]:
    code = _clean(code)
    if not code or code in visited:
        return ()
    visited = visited | {code}
    row = _registry().get(code, {})
    parent = _clean(row.get("parent_code"))
    subtype_key = _clean(row.get("subtype_key"))

    out: list[GeneExpectation] = []
    _add_expectations(
        out,
        genes=_lineage_genes().get(code, ()),
        direction="high",
        source="lineage panel",
        inherited_from=inherited_from,
    )
    _add_expectations(
        out,
        genes=_CURATED_HIGH.get(code, ()),
        direction="high",
        source="tumor-type ontology",
        inherited_from=inherited_from,
    )

    if subtype_key and parent:
        _add_expectations(
            out,
            genes=_key_biomarker_genes(parent, subtype_key),
            direction="high",
            source="subtype biomarker panel",
            inherited_from=inherited_from,
        )

    try:
        from .literature_signatures import literature_signature

        signature = literature_signature(code)
    except Exception:
        signature = None
    if signature is not None:
        _add_expectations(
            out,
            genes=signature.marker_genes,
            direction="high",
            source="literature signature",
            inherited_from=inherited_from,
            rationale=signature.rationale,
        )

    if len(out) < 3:
        _add_expectations(
            out,
            genes=_key_biomarker_genes(code),
            direction="high",
            source="cancer-key biomarker panel",
            inherited_from=inherited_from,
        )

    if len(out) < 3 and parent:
        inherited = _high_expectations_for_code(
            parent,
            inherited_from=parent,
            visited=visited,
        )
        for expectation in inherited:
            if expectation.symbol not in {row.symbol for row in out}:
                out.append(expectation)

    return tuple(out)


def _low_expectations_for_code(code: str) -> tuple[GeneExpectation, ...]:
    row = _registry().get(code, {})
    family = _clean(row.get("family"))
    family_root = family.split("-", 1)[0] if family else ""
    family_lows = _FAMILY_LOW_GENES.get(family, _FAMILY_LOW_GENES.get(family_root))
    genes = list(_CURATED_LOW.get(code, ()))
    genes.extend(family_lows or _DEFAULT_LOW_GENES)
    out: list[GeneExpectation] = []
    _add_expectations(
        out,
        genes=tuple(genes),
        direction="low",
        source="tumor-type ontology",
        rationale="genes that would argue for a competing lineage if strongly expressed",
    )
    return tuple(out)


@lru_cache(maxsize=1)
def tumor_type_ontology() -> dict[str, TumorTypeOntologyEntry]:
    from .analyze import effective_expression_reference

    out: dict[str, TumorTypeOntologyEntry] = {}
    for code, row in _registry().items():
        ref = effective_expression_reference(code)
        out[code] = TumorTypeOntologyEntry(
            code=code,
            name=_clean(row.get("name")) or code,
            family=_clean(row.get("family")),
            parent_code=_clean(row.get("parent_code")),
            ancestors=_ancestor_codes(code),
            expression_reference_code=_clean(getattr(ref, "reference_code", "")),
            expression_reference_source=_clean(getattr(ref, "source", "")),
            expression_reference_kind=_clean(getattr(ref, "source_kind", "")),
            expression_reference_direct=bool(getattr(ref, "direct", False)),
            expression_reference_reason=_clean(getattr(ref, "fallback_reason", "")),
            expected_high=_high_expectations_for_code(code),
            expected_low=_low_expectations_for_code(code),
        )
    return out


def tumor_type_ontology_entry(code: str | None) -> TumorTypeOntologyEntry | None:
    if not code:
        return None
    return tumor_type_ontology().get(_clean(code))


def _observed_rows(
    expectations: tuple[GeneExpectation, ...],
    sample_tpm_by_symbol: Mapping[str, float],
) -> list[dict[str, Any]]:
    sample = {_gene_symbol(k): float(v or 0.0) for k, v in sample_tpm_by_symbol.items()}
    rows: list[dict[str, Any]] = []
    for expectation in expectations:
        rows.append(
            {
                "gene": expectation.symbol,
                "tpm": float(sample.get(expectation.symbol, 0.0)),
                "source": expectation.source,
                "inherited_from": expectation.inherited_from,
                "rationale": expectation.rationale,
            }
        )
    return rows


def tumor_type_sanity_check(
    code: str | None,
    sample_tpm_by_symbol: Mapping[str, float],
    *,
    high_tpm_floor: float = _HIGH_TPM_FLOOR,
    low_tpm_ceiling: float = _LOW_TPM_CEILING,
) -> dict[str, Any]:
    entry = tumor_type_ontology_entry(code)
    if entry is None:
        return {}

    high_rows = _observed_rows(entry.expected_high, sample_tpm_by_symbol)
    low_rows = _observed_rows(entry.expected_low, sample_tpm_by_symbol)
    high_detected = [row for row in high_rows if row["tpm"] >= high_tpm_floor]
    low_present = [row for row in low_rows if row["tpm"] > low_tpm_ceiling]
    high_detected = sorted(high_detected, key=lambda row: row["tpm"], reverse=True)
    low_present = sorted(low_present, key=lambda row: row["tpm"], reverse=True)

    required_high = min(2, len(high_rows))
    if not high_rows:
        status = "not_evaluable"
    elif len(high_detected) >= required_high:
        status = "consistent"
    elif high_detected:
        status = "partial"
    else:
        status = "weak"
    if status == "consistent" and len(low_present) >= 3:
        status = "mixed"

    return {
        "code": entry.code,
        "name": entry.name,
        "family": entry.family,
        "parent_code": entry.parent_code,
        "ancestors": list(entry.ancestors),
        "expression_reference": {
            "code": entry.expression_reference_code,
            "source": entry.expression_reference_source,
            "kind": entry.expression_reference_kind,
            "direct": entry.expression_reference_direct,
            "reason": entry.expression_reference_reason,
        },
        "status": status,
        "high_tpm_floor": float(high_tpm_floor),
        "low_tpm_ceiling": float(low_tpm_ceiling),
        "expected_high": sorted(high_rows, key=lambda row: row["tpm"], reverse=True),
        "expected_high_detected": high_detected,
        "expected_low": sorted(low_rows, key=lambda row: row["tpm"], reverse=True),
        "expected_low_present": low_present,
        "summary": (
            f"{len(high_detected)}/{len(high_rows)} expected high markers "
            f"are >= {high_tpm_floor:g} TPM; "
            f"{len(low_present)}/{len(low_rows)} expected low markers "
            f"are > {low_tpm_ceiling:g} TPM."
        ),
    }
