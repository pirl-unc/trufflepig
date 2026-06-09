"""Cancer-type label context shared across analysis, reports, and plots.

The analyze pipeline often carries more than one cancer label:

- a report label, used for the clinical/curation surface;
- a broad reference label, used by coarse classifier and cohort math;
- sometimes a finer expression reference, when an exact subtype cohort exists.

This module keeps that relationship explicit so downstream code can choose the
right level without re-deriving parent/subtype rules locally.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Any, Mapping

log = logging.getLogger(__name__)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


@dataclass(frozen=True)
class ExpressionReferenceRecord:
    """One expression reference available for a registry cancer code."""

    requested_code: str
    reference_code: str
    source_kind: str
    source: str
    normalization: str = "clean_tpm"
    gene_key: str = "ensembl_symbol"
    source_code: str = ""
    direct: bool = True
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def _registry_records() -> dict[str, dict[str, Any]]:
    try:
        from pirlygenes.gene_sets_cancer import cancer_type_registry

        df = cancer_type_registry()
        return {
            _clean(row.get("code")): dict(row)
            for _, row in df.iterrows()
            if _clean(row.get("code"))
        }
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _direct_expression_reference_records() -> dict[str, tuple[ExpressionReferenceRecord, ...]]:
    records_by_code: dict[str, list[ExpressionReferenceRecord]] = {}
    registry = _registry_records()
    registry_codes = set(registry)
    aliases = {
        "BRCA_Her2": "BRCA_HER2",
        "LUAD_STK11_KEAP1": "LUAD_STK11",
        "LUAD_KRAS_STK11": "LUAD_STK11",
        "BEATAML_APL": "LAML_APL",
        "BEATAML_ELN_Adverse": "LAML_ELNadv",
        "BEATAML_ELN_Favorable": "LAML_ELNfav",
        "BEATAML_ELN_Intermediate": "LAML_ELNint",
        "TARGET_AML": "LAML",
        "TARGET_NBL": "NBL",
        "TARGET_RT": "RT",
        "TARGET_WT": "WILMS",
    }

    def canonical_code(value: Any) -> str:
        text = _clean(value)
        if not text:
            return ""
        if text in aliases:
            return aliases[text]
        try:
            from pirlygenes.gene_sets_cancer import resolve_cancer_type

            return _clean(resolve_cancer_type(text))
        except Exception:
            return text

    def add(
        code: Any,
        *,
        source_kind: str,
        source: str,
        gene_key: str,
        source_code: str | None = None,
    ) -> None:
        canonical = canonical_code(code)
        if not canonical or canonical not in registry_codes:
            return
        record = ExpressionReferenceRecord(
            requested_code=canonical,
            reference_code=canonical,
            source_kind=source_kind,
            source=source,
            gene_key=gene_key,
            source_code=_clean(source_code) or canonical,
            direct=True,
        )
        records_by_code.setdefault(canonical, []).append(record)

    try:
        from trufflepig.reference import pan_cancer_expression

        pan = pan_cancer_expression(technical_rna_normalize=True)
        for col in pan.columns:
            col_text = str(col)
            if not col_text.endswith("_TPM"):
                continue
            code = col_text.removesuffix("_TPM")
            row = registry.get(code, {})
            add(
                code,
                source_kind="observed_pan_cancer_reference",
                source=_clean(row.get("source_cohort")) or "pan_cancer",
                gene_key="ensembl_symbol",
            )
    except Exception as exc:
        log.warning("observed pan-cancer reference discovery failed: %s", exc)

    try:
        from trufflepig.reference import cancer_reference_expression

        observed = cancer_reference_expression(
            normalize="tpm_clean",
            format="long",
            include_provenance=True,
        )
        if observed is not None and "cancer_code" in observed.columns:
            for code, group in observed.groupby("cancer_code"):
                code_text = _clean(code)
                if not code_text:
                    continue
                if "source_cohort" in group.columns:
                    values = {
                        _clean(v)
                        for v in group["source_cohort"].dropna().unique()
                        if _clean(v)
                    }
                else:
                    values = set()
                for source in values or {"observed_bulk_reference"}:
                    add(
                        code_text,
                        source_kind="observed_bulk_reference",
                        source=source,
                        gene_key="ensembl_symbol",
                    )
    except Exception as exc:
        log.warning("observed-bulk reference discovery failed: %s", exc)

    try:
        from trufflepig.reference import tcga_deconvolved_expression

        tcga = tcga_deconvolved_expression()
        if tcga is not None and "cancer_code" in tcga.columns:
            for code, group in tcga.groupby("cancer_code"):
                code_text = _clean(code)
                if not code_text:
                    continue
                if "source_cohort" in group.columns:
                    values = {
                        _clean(v)
                        for v in group["source_cohort"].dropna().unique()
                        if _clean(v)
                    }
                else:
                    values = set()
                for source in values or {"TCGA"}:
                        add(
                            code_text,
                            source_kind="deconvolved_tumor_reference",
                            source=source,
                            gene_key="ensembl_symbol",
                            source_code=code_text,
                        )
    except Exception as exc:
        log.warning("TCGA deconvolved reference discovery failed: %s", exc)

    try:
        from trufflepig.reference import subtype_deconvolved_expression

        sub = subtype_deconvolved_expression()
        if sub is not None and "cancer_code" in sub.columns:
            for code, group in sub.groupby("cancer_code"):
                code_text = _clean(code)
                if not code_text:
                    continue
                if "source_cohort" in group.columns:
                    values = {
                        _clean(v)
                        for v in group["source_cohort"].dropna().unique()
                        if _clean(v)
                    }
                else:
                    values = set()
                subtype_values = (
                    group.get("subtype").fillna("").astype(str).map(_clean)
                    if "subtype" in group.columns
                    else []
                )
                if all(not _clean(v) for v in subtype_values):
                    for source in values or {"subtype_reference"}:
                        add(
                            code_text,
                            source_kind="deconvolved_tumor_reference",
                            source=source,
                            gene_key="symbol_only",
                            source_code=code_text,
                        )
            if "subtype" in sub.columns:
                for subtype, group in sub.dropna(subset=["subtype"]).groupby("subtype"):
                    subtype_text = _clean(subtype)
                    if not subtype_text:
                        continue
                    if "source_cohort" in group.columns:
                        values = {
                            _clean(v)
                            for v in group["source_cohort"].dropna().unique()
                            if _clean(v)
                        }
                    else:
                        values = set()
                    for source in values or {"subtype_reference"}:
                        add(
                            subtype_text,
                            source_kind="deconvolved_tumor_reference",
                            source=source,
                            gene_key="symbol_only",
                            source_code=subtype_text,
                        )
    except Exception as exc:
        log.warning("subtype deconvolved reference discovery failed: %s", exc)

    out: dict[str, tuple[ExpressionReferenceRecord, ...]] = {}
    for code, records in records_by_code.items():
        unique = {}
        for record in records:
            key = (
                record.reference_code,
                record.source_kind,
                record.source,
                record.normalization,
                record.gene_key,
                record.source_code,
                record.direct,
            )
            unique[key] = record
        out[code] = tuple(
            sorted(
                unique.values(),
                key=lambda record: (
                    record.source_kind != "deconvolved_tumor_reference",
                    record.source_kind,
                    record.source,
                ),
            )
        )
    return out


_REFERENCE_CODE_FALLBACKS: Mapping[str, tuple[str, ...]] = {
    "NBL": ("NBL_MYCNnonamp", "NBL_MYCNamp"),
    "PCN": ("MM",),
    # NCI-coverage expansion (pirlygenes 5.18): GI/CNS leaves whose family
    # fallback would be a poor lineage match (anal SCC is squamous, not the
    # GI-adeno bulk; gallbladder is biliary adeno; DIPG/EPN are glial, not
    # the embryonal MBL the cns-family fallback resolves first).
    "ANSC": ("CESC", "HNSC"),
    "GBC": ("CHOL", "PAAD"),
    "DIPG": ("GBM", "LGG"),
    "EPN": ("GBM", "LGG"),
    "CRANIO": ("LGG", "GBM"),
}

# Keyed on pirlygenes' lineage-only family ontology (5.12+): the old
# ``net`` / ``pediatric-*`` families collapsed into ``neuroendocrine`` /
# ``sarcoma`` / ``cns`` / ``embryonal``.
_REFERENCE_FAMILY_FALLBACKS: Mapping[str, tuple[str, ...]] = {
    "carcinoma-breast": ("BRCA",),
    "carcinoma-head-neck": ("HNSC",),
    "carcinoma-lung": ("LUAD", "LUSC"),
    # Keratinocyte carcinomas (BCC, cSCC): nearest bulk squamous reference.
    "carcinoma-skin": ("HNSC",),
    # HPV-associated GU squamous carcinomas (VSCC, VAGC, PENSCC, URETH):
    # cervical SCC is the closest cohort; urothelial as secondary.
    "carcinoma-gu": ("CESC", "BLCA"),
    "cns": ("MBL", "GBM", "LGG"),
    "endocrine": ("THCA", "PCPG", "ACC"),
    "heme-bcell": ("DLBC", "CLL", "B_ALL"),
    "heme-myeloid": ("LAML",),
    "heme-plasma": ("MM",),
    "heme-tcell": ("T_ALL",),
    "neuroendocrine": ("SCLC", "NET_PANCREAS", "LUAD"),
    "salivary": ("HNSC",),
    "sarcoma": ("SARC",),
}


def _fallback_candidates(code: str) -> tuple[tuple[str, str], ...]:
    code_text = _clean(code)
    row = _registry_records().get(code_text, {})
    candidates: list[tuple[str, str]] = []
    parent = _clean(row.get("parent_code"))
    if parent:
        candidates.append((parent, "registry parent"))
    for fallback in _REFERENCE_CODE_FALLBACKS.get(code_text, ()):
        candidates.append((fallback, "curated code fallback"))
    family = _clean(row.get("family")).lower()
    for fallback in _REFERENCE_FAMILY_FALLBACKS.get(family, ()):
        candidates.append((fallback, f"{family} family fallback"))

    seen = set()
    out = []
    for candidate, reason in candidates:
        candidate = _clean(candidate)
        if candidate and candidate != code_text and candidate not in seen:
            seen.add(candidate)
            out.append((candidate, reason))
    return tuple(out)


def expression_reference_options(
    code: str | None,
    *,
    include_fallback: bool = True,
) -> tuple[ExpressionReferenceRecord, ...]:
    code_text = _clean(code)
    if not code_text:
        return ()
    direct = _direct_expression_reference_records().get(code_text, ())
    if direct or not include_fallback:
        return direct
    return _expression_reference_options_with_fallback(code_text, visited=frozenset())


def _expression_reference_options_with_fallback(
    code_text: str,
    *,
    visited: frozenset[str],
) -> tuple[ExpressionReferenceRecord, ...]:
    direct = _direct_expression_reference_records().get(code_text, ())
    if direct:
        return direct
    if code_text in visited:
        return ()
    visited = visited | {code_text}
    for candidate, reason in _fallback_candidates(code_text):
        if candidate in visited:
            continue
        records = _direct_expression_reference_records().get(candidate, ())
        if not records:
            records = _expression_reference_options_with_fallback(
                candidate,
                visited=visited,
            )
        if records:
            return tuple(
                replace(
                    record,
                    requested_code=code_text,
                    direct=False,
                    fallback_reason=(
                        f"{reason}; {record.fallback_reason}"
                        if record.fallback_reason
                        else reason
                    ),
                )
                for record in records
            )
    return ()


def effective_expression_reference(code: str | None) -> ExpressionReferenceRecord | None:
    options = expression_reference_options(code, include_fallback=True)
    return options[0] if options else None


def registry_parent_code(code: str | None) -> str:
    row = _registry_records().get(_clean(code), {})
    return _clean(row.get("parent_code"))


def registry_display_name(code: str | None) -> str:
    code_text = _clean(code)
    if not code_text:
        return ""
    row = _registry_records().get(code_text, {})
    return _clean(row.get("name")) or code_text


def cancer_type_context_label(code: str | None) -> str:
    code_text = _clean(code)
    if not code_text:
        return ""
    name = registry_display_name(code_text)
    if name and name.lower() != code_text.lower():
        return f"{code_text} ({name})"
    return code_text


def expression_reference_sources(code: str | None) -> tuple[str, ...]:
    records = expression_reference_options(code, include_fallback=False)
    return tuple(sorted({record.source for record in records}))


def has_expression_reference(code: str | None) -> bool:
    return bool(expression_reference_options(code, include_fallback=True))


@dataclass(frozen=True)
class CancerTypeContext:
    report_code: str
    reference_code: str
    fine_code: str
    coarse_code: str
    best_expression_code: str
    parent_code: str = ""
    supplied_code: str = ""
    source: str = ""
    relationship: str = "same"
    report_has_expression_ref: bool = False
    reference_has_expression_ref: bool = False
    fine_expression_sources: tuple[str, ...] = ()
    reference_expression_sources: tuple[str, ...] = ()
    best_expression_source: str = ""
    best_expression_source_kind: str = ""
    best_expression_gene_key: str = ""
    best_expression_direct: bool = False
    best_expression_fallback_reason: str = ""

    def code_for(self, role: str) -> str:
        key = _clean(role).lower().replace("-", "_")
        if key in {"report", "report_scope", "fine", "curation", "therapy"}:
            return self.fine_code or self.report_code or self.coarse_code
        if key in {
            "reference",
            "coarse",
            "cohort",
            "classifier",
            "purity",
            "decomposition",
            "tcga",
            "broad",
        }:
            return self.coarse_code or self.reference_code or self.report_code
        if key in {"expression", "best_expression", "plot_expression"}:
            return (
                self.best_expression_code
                or self.fine_code
                or self.coarse_code
                or self.report_code
            )
        if key == "parent":
            return self.parent_code
        return self.report_code or self.reference_code

    def label_for(self, role: str) -> str:
        return cancer_type_context_label(self.code_for(role))

    @property
    def uses_distinct_reference(self) -> bool:
        return bool(self.report_code and self.reference_code) and (
            self.report_code != self.reference_code
        )

    @property
    def fine_expression_available(self) -> bool:
        return bool(self.fine_code and self.report_has_expression_ref)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["report_label"] = cancer_type_context_label(self.report_code)
        data["reference_label"] = cancer_type_context_label(self.reference_code)
        data["fine_label"] = cancer_type_context_label(self.fine_code)
        data["coarse_label"] = cancer_type_context_label(self.coarse_code)
        data["best_expression_label"] = cancer_type_context_label(self.best_expression_code)
        return data

    def markdown_lines(self) -> list[str]:
        if not self.report_code:
            return []
        lines = [f"- **Report label**: {cancer_type_context_label(self.report_code)}."]
        if self.uses_distinct_reference:
            lines.append(
                f"- **Broad reference context**: {cancer_type_context_label(self.reference_code)} "
                "is used when a step needs a coarse cohort reference."
            )
        else:
            lines.append(
                "- **Broad reference context**: same as report label; no finer "
                "registry label is active."
            )
        if self.parent_code:
            lines.append(
                f"- **Hierarchy**: {cancer_type_context_label(self.report_code)} is modeled "
                f"as a refined label under {cancer_type_context_label(self.parent_code)}."
            )
        if self.best_expression_code:
            if self.best_expression_code == self.report_code:
                if self.best_expression_direct:
                    lines.append(
                        "- **Best expression reference**: expression data are "
                        "available for the report label."
                    )
            else:
                fallback_reason = (
                    f" ({self.best_expression_fallback_reason})"
                    if self.best_expression_fallback_reason
                    else ""
                )
                lines.append(
                    f"- **Best expression reference**: falls back to "
                    f"{cancer_type_context_label(self.best_expression_code)}"
                    f"{fallback_reason} because an exact expression cohort is not "
                    "available for the report label."
                )
        if self.fine_expression_available and self.uses_distinct_reference:
            lines.append(
                "- **Context caveat**: subtype-aware modules may use the fine-grained "
                "reference; coarse pan-reference modules explicitly use the broad "
                "reference."
            )
        return lines


def cancer_type_context_from_analysis(
    analysis: Mapping[str, Any], supplied_cancer_type: str | None = None
) -> CancerTypeContext:
    report_code = _clean(
        analysis.get("report_scope_cancer_type") or analysis.get("cancer_type")
    )
    parent_code = _clean(analysis.get("report_scope_parent_cancer_type"))
    registry_parent = registry_parent_code(report_code)
    if not parent_code and registry_parent:
        parent_code = registry_parent
    reference_code = _clean(
        analysis.get("reference_cancer_type")
        or analysis.get("report_scope_parent_cancer_type")
    )
    if not reference_code:
        reference_code = parent_code or report_code

    report_direct_options = expression_reference_options(
        report_code, include_fallback=False
    )
    reference_direct_options = expression_reference_options(
        reference_code, include_fallback=False
    )
    report_sources = tuple(sorted({record.source for record in report_direct_options}))
    reference_sources = tuple(
        sorted({record.source for record in reference_direct_options})
    )
    report_effective = effective_expression_reference(report_code)
    reference_effective = effective_expression_reference(reference_code)
    best_expression_record = None
    if report_direct_options:
        best_expression_code = report_code
        best_expression_record = report_direct_options[0]
    elif report_effective is not None:
        best_expression_code = report_effective.reference_code
        best_expression_record = report_effective
    elif reference_direct_options:
        best_expression_code = reference_code
        best_expression_record = reference_direct_options[0]
    elif reference_effective is not None:
        best_expression_code = reference_effective.reference_code
        best_expression_record = reference_effective
    else:
        best_expression_code = reference_code or report_code

    if report_code and reference_code and report_code == reference_code:
        relationship = "same"
    elif parent_code and reference_code and parent_code == reference_code:
        relationship = "fine_child_of_reference"
    elif parent_code:
        relationship = "fine_child_with_independent_reference"
    elif report_code and reference_code:
        relationship = "report_scope_with_independent_reference"
    else:
        relationship = "unresolved"

    return CancerTypeContext(
        report_code=report_code,
        reference_code=reference_code,
        fine_code=report_code,
        coarse_code=reference_code or parent_code or report_code,
        best_expression_code=best_expression_code,
        parent_code=parent_code,
        supplied_code=_clean(
            supplied_cancer_type
            or (analysis.get("analysis_constraints") or {}).get("cancer_type")
        ),
        source=_clean(analysis.get("cancer_type_source")),
        relationship=relationship,
        report_has_expression_ref=bool(report_sources),
        reference_has_expression_ref=bool(reference_sources),
        fine_expression_sources=report_sources,
        reference_expression_sources=reference_sources,
        best_expression_source=best_expression_record.source
        if best_expression_record is not None
        else "",
        best_expression_source_kind=best_expression_record.source_kind
        if best_expression_record is not None
        else "",
        best_expression_gene_key=best_expression_record.gene_key
        if best_expression_record is not None
        else "",
        best_expression_direct=bool(
            best_expression_record.direct if best_expression_record is not None else False
        ),
        best_expression_fallback_reason=best_expression_record.fallback_reason
        if best_expression_record is not None
        else "",
    )


def cancer_type_context_code(analysis: Mapping[str, Any], role: str) -> str:
    return cancer_type_context_from_analysis(analysis).code_for(role)
