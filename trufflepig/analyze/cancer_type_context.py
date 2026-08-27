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


def registry_ancestor_codes(code: str | None) -> tuple[str, ...]:
    """Return registry ancestors from immediate parent to root."""
    code_text = _clean(code)
    records = _registry_records()
    if not code_text or code_text not in records:
        return ()
    ancestors: list[str] = []
    seen = {code_text}
    parent = _clean(records[code_text].get("parent_code"))
    while parent and parent not in seen:
        ancestors.append(parent)
        seen.add(parent)
        parent = _clean(records.get(parent, {}).get("parent_code"))
    return tuple(ancestors)


def cancer_type_tree_relationship(
    report_code: str | None,
    other_code: str | None,
) -> str:
    """Describe ``other_code`` relative to the active report node."""
    report = _clean(report_code)
    other = _clean(other_code)
    if not report or not other:
        return "unknown"
    if report == other:
        return "same"
    records = _registry_records()
    if report not in records or other not in records:
        return "unknown"
    report_ancestors = registry_ancestor_codes(report)
    other_ancestors = registry_ancestor_codes(other)
    if other in report_ancestors:
        return "ancestor"
    if report in other_ancestors:
        return "descendant"
    if set(report_ancestors).intersection(other_ancestors):
        return "sibling"
    return "independent"


def nearest_common_ancestor(
    left_code: str | None,
    right_code: str | None,
) -> str:
    """Return the nearest shared registry node, preferring ``left_code``'s path."""
    left = _clean(left_code)
    right = _clean(right_code)
    if not left or not right:
        return ""
    right_path = {right, *registry_ancestor_codes(right)}
    return next(
        (
            code
            for code in (left, *registry_ancestor_codes(left))
            if code in right_path
        ),
        "",
    )


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
        from trufflepig.reference import cancer_reference_manifest

        # Discovery only needs (code, cohort) provenance.  Loading the full
        # long expression table here made every xdist worker materialize ~7 GB
        # of object-heavy data before returning a ~137-row manifest.
        observed = cancer_reference_manifest()
        if observed is not None and "cancer_code" in observed.columns:
            for row in observed.to_dict("records"):
                code_text = _clean(row.get("cancer_code"))
                if not code_text:
                    continue
                add(
                    code_text,
                    source_kind="observed_bulk_reference",
                    source=_clean(row.get("source_cohort"))
                    or "observed_bulk_reference",
                    gene_key="ensembl_symbol",
                    source_code=code_text,
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

    # Some loadable grouping nodes are computed member unions rather than
    # physical cohorts, so the compact leaf manifest cannot name them directly.
    # Discover only unresolved union nodes through oncoref's lightweight
    # artifact index. This keeps the active code at the shared parent (for
    # example THYM_EPITHELIAL) instead of borrowing one child as if it were the
    # requested entity, and does not materialize expression values.
    member_union_codes = sorted(
        code
        for code, row in registry.items()
        if _clean(row.get("reference_source")) == "member_union"
        and not records_by_code.get(code)
    )
    if member_union_codes:
        try:
            import oncoref

            availability = oncoref.cancer_reference_expression_availability(
                cancer_types=member_union_codes,
                normalize="tpm_clean",
                reference_source="artifact",
                sample_qc="artifact",
            )
            if "available" in availability.columns:
                availability = availability[
                    availability["available"].fillna(False).astype(bool)
                ]
            for code, group in availability.groupby("requested_code"):
                code_text = _clean(code)
                if not code_text or records_by_code.get(code_text):
                    continue
                sources = sorted(
                    {
                        _clean(source)
                        for source in group.get("source_cohort", ())
                        if _clean(source)
                    }
                )
                add(
                    code_text,
                    source_kind="computed_member_union_reference",
                    source=" + ".join(sources) or "artifact member union",
                    gene_key="ensembl_symbol",
                    source_code=code_text,
                )
        except Exception as exc:
            log.warning("member-union reference discovery failed: %s", exc)

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
                    record.source
                    != _clean(
                        registry.get(record.reference_code, {}).get(
                            "source_cohort"
                        )
                    ),
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
    # GI adenocarcinomas (incl. the CRC colorectal grouping parent, which has no
    # own cohort) -> nearest GI cohorts.
    "carcinoma-gi": ("COAD", "STAD", "READ"),
    "carcinoma-head-neck": ("HNSC",),
    "carcinoma-lung": ("LUAD", "LUSC"),
    # Keratinocyte carcinomas (BCC, cSCC): nearest bulk squamous reference.
    "carcinoma-skin": ("HNSC",),
    # GU squamous carcinomas without a direct reference (VAGC, PENSCC, URETH):
    # cervical SCC is the closest cohort; urothelial as secondary. VSCC now has
    # a direct reference in oncoref 1.8.184 / pirlygenes 5.23.55 and therefore
    # resolves before this fallback is considered.
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


def _reference_backed_member_candidates(code_text: str) -> tuple[tuple[str, str], ...]:
    """Reference-backed descendants of a grouping/aggregate — a faithful in-lineage
    representative to fall back to before any cross-family fallback.

    A grouping with no direct reference of its own (NET, BTC, SGC, …) otherwise
    resolves *sideways* to a family stranger (NET→SCLC, BTC→COAD, SGC→HNSC) even
    though it has member cohorts that carry a real reference (a NET site atom, CHOL,
    ADCC). Prefer such a member. Only descendants that already carry a DIRECT
    expression reference are returned, so the resolver resolves immediately without
    recursing further. Deterministic (alphabetical) order — the *proper* fix is a
    pooled member-union reference produced upstream (oncoref#329); this is the
    defensive interim so a grouping is never characterized against a wrong-lineage
    cohort in the meantime.
    """
    try:
        from oncoref import cancer_type_descendants

        descendants = cancer_type_descendants(code_text)
    except Exception:  # noqa: BLE001 — ontology optional; degrade to family fallback
        return ()
    direct = _direct_expression_reference_records()
    members = sorted(
        {
            _clean(d)
            for d in descendants or ()
            if _clean(d) and direct.get(_clean(d))
        }
    )
    return tuple((member, "member cohort") for member in members)


def _fallback_candidates(code: str) -> tuple[tuple[str, str], ...]:
    code_text = _clean(code)
    row = _registry_records().get(code_text, {})
    candidates: list[tuple[str, str]] = []
    # A code's OWN reference-backed members (a NET site atom for NET, CHOL for BTC, ADCC
    # for SGC) are the most faithful in-lineage fallback, so they come FIRST — before the
    # registry parent. Walking UP to the parent and back DOWN risks landing in a
    # different-lineage sibling subtree: since oncoref 1.8.107 reparented NET under the new
    # NEN grouping, a parent-first walk resolved NET -> NEN -> NEC_LUNG_LARGECELL (a
    # poorly-differentiated neuroendocrine CARCINOMA) — the exact cross-lineage stranger
    # this descent exists to avoid. Leaf subtypes (COAD_MSI, …) have no reference-backed
    # members, so this list is empty for them and the registry parent below still wins,
    # keeping a parented subtype on its parent reference. The proper fix is a pooled
    # member-union reference upstream (oncoref#329); this is the defensive interim.
    for member, reason in _reference_backed_member_candidates(code_text):
        candidates.append((member, reason))
    parent = _clean(row.get("parent_code"))
    if parent:
        candidates.append((parent, "registry parent"))
    for fallback in _REFERENCE_CODE_FALLBACKS.get(code_text, ()):
        candidates.append((fallback, "curated code fallback"))
    family = _clean(row.get("family")).lower()
    fam_fallbacks = _REFERENCE_FAMILY_FALLBACKS.get(family)
    if not fam_fallbacks and "-" in family:
        # Fine sub-family (pirlygenes #359 split: ``cns`` -> ``cns-meningeal`` /
        # ``cns-choroid`` / ``cns-sellar`` / ``cns-glial`` / ``cns-ependymal`` /
        # ``cns-embryonal``; ``endocrine`` -> ``endocrine-epithelial`` /
        # ``endocrine-neuroendocrine``). Fall back via the family group root so a
        # cohort-less sub-family still resolves to the group's reference (e.g.
        # MENINGIOMA cns-meningeal -> the cns group -> MBL). Robust to future splits.
        root = family.split("-", 1)[0]
        fam_fallbacks = _REFERENCE_FAMILY_FALLBACKS.get(root)
    for fallback in fam_fallbacks or ():
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
    return _expression_reference_options_with_fallback(
        code_text,
        requested_code=code_text,
        visited=frozenset(),
    )


def _expression_reference_options_with_fallback(
    code_text: str,
    *,
    requested_code: str,
    visited: frozenset[str],
) -> tuple[ExpressionReferenceRecord, ...]:
    direct = _direct_expression_reference_records().get(code_text, ())
    if direct:
        return tuple(
            record
            for record in direct
            if cancer_type_tree_relationship(
                requested_code,
                record.reference_code,
            )
            != "sibling"
        )
    if code_text in visited:
        return ()
    visited = visited | {code_text}
    for candidate, reason in _fallback_candidates(code_text):
        if candidate in visited:
            continue
        records = _direct_expression_reference_records().get(candidate, ())
        if records:
            records = tuple(
                record
                for record in records
                if cancer_type_tree_relationship(
                    requested_code,
                    record.reference_code,
                )
                != "sibling"
            )
        if not records:
            records = _expression_reference_options_with_fallback(
                candidate,
                requested_code=requested_code,
                visited=visited,
            )
        if records:
            return tuple(
                replace(
                    record,
                    requested_code=requested_code,
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
    requested_reference_code: str = ""
    requested_expression_code: str = ""
    reference_relationship: str = "same"
    expression_relationship: str = "same"
    excluded_sibling_codes: tuple[str, ...] = ()
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
        data["expression_code"] = self.best_expression_code
        data["fallback_expression_code"] = self.reference_code
        data["best_expression_label"] = cancer_type_context_label(self.best_expression_code)
        data["fallback_expression_label"] = cancer_type_context_label(self.reference_code)
        data["report_role"] = "diagnosis"
        data["reference_role"] = f"{self.reference_relationship}_analysis_context"
        data["expression_role"] = f"{self.expression_relationship}_expression_reference"
        return data

    def markdown_lines(self) -> list[str]:
        if not self.report_code:
            return []
        report_label = cancer_type_context_label(self.report_code)
        lines = [f"- **Report label (diagnosis node)**: {report_label}."]
        if self.uses_distinct_reference:
            relation = (
                "ancestor"
                if self.reference_relationship == "ancestor"
                else "independent"
            )
            lines.append(
                f"- **Fallback expression/reference context**: "
                f"{cancer_type_context_label(self.reference_code)} is the {relation} "
                "analysis context used only when a step needs a broader cohort; "
                f"it does not replace the report diagnosis {report_label}."
            )
        else:
            if (
                self.best_expression_direct
                and self.best_expression_source_kind
                not in {"observed_pan_cancer_reference", ""}
            ):
                lines.append(
                    "- **Reference context**: same code as report label; a direct "
                    "expression reference is available."
                )
            else:
                lines.append(
                    "- **Broad reference context**: same as report label; no finer "
                    "registry label is active."
                )
        if self.parent_code:
            lines.append(
                f"- **Hierarchy**: {cancer_type_context_label(self.report_code)} is modeled "
                f"as a refined label under {cancer_type_context_label(self.parent_code)}; "
                "both codes are on the same registry branch."
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
                expression_label = cancer_type_context_label(
                    self.best_expression_code
                )
                if self.expression_relationship == "descendant":
                    lines.append(
                        f"- **Best expression reference (descendant only)**: "
                        f"{expression_label}{fallback_reason} supplies expression "
                        f"context but does not refine the report diagnosis {report_label}."
                    )
                else:
                    lines.append(
                        f"- **Best expression reference**: falls back to "
                        f"{expression_label}{fallback_reason} because an exact "
                        "expression cohort is not available for the report label; "
                        f"the diagnosis remains {report_label}."
                    )
        if self.excluded_sibling_codes:
            excluded = ", ".join(
                cancer_type_context_label(code)
                for code in self.excluded_sibling_codes
            )
            lines.append(
                f"- **Sibling isolation**: {excluded} was considered as competing "
                "evidence but is not used as a report, reference, or expression "
                f"context; active interpretation stays on the {report_label} branch."
            )
        if self.fine_expression_available and self.uses_distinct_reference:
            lines.append(
                "- **Context caveat**: subtype-aware modules prefer the fine-grained "
                "reference; coarse pan-reference modules explicitly use the broad "
                "fallback reference."
            )
        return lines


def cancer_type_context_from_analysis(
    analysis: Mapping[str, Any], supplied_cancer_type: str | None = None
) -> CancerTypeContext:
    report_code = _clean(
        analysis.get("report_scope_cancer_type") or analysis.get("cancer_type")
    )
    requested_parent_code = _clean(analysis.get("report_scope_parent_cancer_type"))
    registry_parent = registry_parent_code(report_code)
    parent_code = (
        requested_parent_code
        if cancer_type_tree_relationship(report_code, requested_parent_code)
        == "ancestor"
        else registry_parent
    )
    requested_reference_code = _clean(
        analysis.get("requested_reference_cancer_type")
        or analysis.get("reference_cancer_type")
        or analysis.get("report_scope_parent_cancer_type")
    )
    requested_expression_code = _clean(
        analysis.get("requested_expression_reference_cancer_type")
        or analysis.get("expression_reference_cancer_type")
    )
    # ``requested_*`` fields are audit history. Once synchronization has
    # replaced an invalid sibling or independent fallback, the active fields
    # must win on the second pass rather than resurrecting the old request.
    reference_code = _clean(analysis.get("reference_cancer_type"))
    if not reference_code:
        reference_code = requested_reference_code or parent_code or report_code
    expression_code = _clean(analysis.get("expression_reference_cancer_type"))
    if not expression_code:
        expression_code = requested_expression_code

    excluded_values = analysis.get("excluded_sibling_cancer_type_contexts") or ()
    if isinstance(excluded_values, str):
        excluded_values = (excluded_values,)
    excluded_sibling_codes = {
        _clean(code)
        for code in excluded_values
        if _clean(code)
    }
    reference_relationship = cancer_type_tree_relationship(
        report_code,
        reference_code,
    )
    if reference_relationship == "sibling":
        excluded_sibling_codes.add(reference_code)
        reference_code = (
            nearest_common_ancestor(report_code, reference_code)
            or parent_code
            or report_code
        )
    elif reference_relationship == "descendant":
        # A descendant can provide a fine expression cohort, but it cannot be
        # called the broad/coarse reference for its parent diagnosis.
        expression_code = expression_code or reference_code
        requested_expression_code = requested_expression_code or reference_code
        reference_code = report_code

    if (
        cancer_type_tree_relationship(report_code, expression_code)
        == "sibling"
    ):
        excluded_sibling_codes.add(expression_code)

    report_direct_options = expression_reference_options(
        report_code, include_fallback=False
    )
    # An unrelated broad fallback must not remain the active analysis context
    # when the resolved report entity has its own broad classifier reference.
    # This includes physical pan-cancer columns and computed member unions, but
    # not a fine observed-bulk cohort such as NUTM that still needs its LUSC
    # context for broad decomposition. The old code remains in
    # ``requested_reference_code`` and in the evidence differential, but no
    # longer controls decomposition or target ranges.
    if (
        cancer_type_tree_relationship(report_code, reference_code)
        == "independent"
        and any(
            record.source_kind
            in {
                "observed_pan_cancer_reference",
                "computed_member_union_reference",
            }
            for record in report_direct_options
        )
    ):
        reference_code = report_code
    reference_direct_options = expression_reference_options(
        reference_code, include_fallback=False
    )
    report_sources = tuple(sorted({record.source for record in report_direct_options}))
    reference_sources = tuple(
        sorted({record.source for record in reference_direct_options})
    )
    report_effective = effective_expression_reference(report_code)
    reference_effective = effective_expression_reference(reference_code)
    requested_expression_options = expression_reference_options(
        expression_code,
        include_fallback=False,
    )
    candidates = (
        list(report_direct_options)
        + list(requested_expression_options)
        + ([report_effective] if report_effective is not None else [])
        + list(reference_direct_options)
        + ([reference_effective] if reference_effective is not None else [])
    )
    best_expression_record = next(
        (
            record
            for record in candidates
            if cancer_type_tree_relationship(
                report_code,
                record.reference_code,
            )
            != "sibling"
        ),
        None,
    )
    for record in candidates:
        if (
            cancer_type_tree_relationship(report_code, record.reference_code)
            == "sibling"
        ):
            excluded_sibling_codes.add(record.reference_code)

    if report_direct_options:
        best_expression_code = report_code
    elif best_expression_record is not None:
        best_expression_code = best_expression_record.reference_code
    else:
        best_expression_code = reference_code or report_code

    reference_relationship = cancer_type_tree_relationship(
        report_code,
        reference_code,
    )
    expression_relationship = cancer_type_tree_relationship(
        report_code,
        best_expression_code,
    )
    if report_code and reference_code and report_code == reference_code:
        relationship = "same"
    elif reference_relationship == "ancestor":
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
        requested_reference_code=requested_reference_code,
        requested_expression_code=requested_expression_code,
        reference_relationship=reference_relationship,
        expression_relationship=expression_relationship,
        excluded_sibling_codes=tuple(sorted(excluded_sibling_codes)),
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
