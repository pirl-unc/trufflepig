"""Data-backed rare cancer hypotheses from RNA expression.

These rules are intentionally conservative. They promote a report scope
for rare non-TCGA cancer hypotheses, while keeping the TCGA classifier
result as expression context and requiring orthogonal diagnostic confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _split_semicolon(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    if result != result:
        return float(default)
    return result


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_bool(value: object, default: bool = False) -> bool:
    text = str(value if value is not None else "").strip().lower()
    if not text or text == "nan":
        return bool(default)
    return text in {"1", "true", "yes", "y"}


def _evaluate_rna_rule(rule, sample_tpm: dict[str, float], top_codes: list[str]):
    primary_gene = str(rule.get("primary_gene") or "").strip()
    if not primary_gene:
        return None
    primary_tpm = _safe_float(sample_tpm.get(primary_gene), 0.0)
    min_tpm = _safe_float(rule.get("min_tpm"), 0.0)
    if primary_tpm < min_tpm:
        return None

    context_codes = set(_split_semicolon(rule.get("context_codes")))
    context_top_k = _safe_int(rule.get("context_top_k"), 5)
    context_slice = set(top_codes[:context_top_k])
    context_match = not context_codes or bool(context_codes & context_slice)
    excluded_context_codes = set(_split_semicolon(rule.get("excluded_context_codes")))
    excluded_context_match = bool(excluded_context_codes & context_slice)
    if not context_match or excluded_context_match:
        return None

    support_min = _safe_float(rule.get("support_min_tpm"), 0.0)
    required_support = _split_semicolon(rule.get("required_support_genes"))
    support_genes = []
    missing_support = []
    for gene in required_support:
        if _safe_float(sample_tpm.get(gene), 0.0) >= support_min:
            support_genes.append(gene)
        else:
            missing_support.append(gene)
    min_support = _safe_int(rule.get("min_support_genes"), 0)
    support_pass = len(support_genes) >= min_support

    absent_max = _safe_float(rule.get("absent_max_tpm"), 0.0)
    absent_confirmed = []
    absent_unconfirmed = []
    for gene in _split_semicolon(rule.get("expected_absent_genes")):
        if _safe_float(sample_tpm.get(gene), 0.0) <= absent_max:
            absent_confirmed.append(gene)
        else:
            absent_unconfirmed.append(gene)

    exclusion_max = _safe_float(rule.get("exclusion_max_tpm"), 0.0)
    exclusion_observed = []
    for gene in _split_semicolon(rule.get("exclusion_genes")):
        if _safe_float(sample_tpm.get(gene), 0.0) > exclusion_max:
            exclusion_observed.append(gene)

    return {
        "primary_gene": primary_gene,
        "primary_tpm": primary_tpm,
        "min_tpm": min_tpm,
        "support_genes": support_genes,
        "missing_support_genes": missing_support,
        "support_pass": support_pass,
        "support_gene_count": len(support_genes),
        "required_support_gene_count": len(required_support),
        "min_support_genes": min_support,
        "absent_genes_confirmed": absent_confirmed,
        "absent_genes_unconfirmed": absent_unconfirmed,
        "exclusion_genes_observed": exclusion_observed,
        "context_codes": sorted(context_codes),
        "context_slice": [code for code in top_codes[:context_top_k] if code],
    }


@dataclass(frozen=True)
class RareCancerRnaInference:
    """One report-scope hypothesis promoted from RNA surrogate evidence."""

    cancer_type: str
    rule_id: str
    surrogate: str
    surrogate_tpm: float
    threshold_tpm: float
    top_reference_cancer_type: str
    confidence: str
    support_genes: tuple[str, ...]
    basis: str
    confirmatory_tests: str
    caveat: str
    source: str
    missing_support_genes: tuple[str, ...] = ()
    exclusion_genes_observed: tuple[str, ...] = ()
    absent_genes_confirmed: tuple[str, ...] = ()
    support_pass: bool = True
    min_support_genes: int = 0
    required_support_gene_count: int = 0
    promote_report_scope: bool = True

    def public_dict(self) -> dict[str, Any]:
        return {
            "cancer_type": self.cancer_type,
            "rule_id": self.rule_id,
            "surrogate": self.surrogate,
            "surrogate_tpm": round(self.surrogate_tpm, 3),
            "threshold_tpm": self.threshold_tpm,
            "top_reference_cancer_type": self.top_reference_cancer_type,
            "confidence": self.confidence,
            "support_genes": list(self.support_genes),
            "missing_support_genes": list(self.missing_support_genes),
            "support_pass": bool(self.support_pass),
            "support_gene_count": len(self.support_genes),
            "min_support_genes": int(self.min_support_genes),
            "required_support_gene_count": int(self.required_support_gene_count),
            "exclusion_genes_observed": list(self.exclusion_genes_observed),
            "absent_genes_confirmed": list(self.absent_genes_confirmed),
            "basis": self.basis,
            "confirmatory_tests": self.confirmatory_tests,
            "caveat": self.caveat,
            "source": self.source,
            "promote_report_scope": self.promote_report_scope,
        }


def rare_cancer_rna_surrogate_rules_df():
    """Return the curated rare-cancer RNA-surrogate rule table."""
    from pirlygenes.load_dataset import get_data

    base = get_data("rare-cancer-rna-surrogates")
    from .literature_signatures import literature_signature_rules_df

    overlay = literature_signature_rules_df()
    if overlay.empty:
        return base
    import pandas as pd

    return (
        pd.concat([base, overlay], ignore_index=True, sort=False)
        .drop_duplicates(subset=["rule_id"], keep="first")
        .reset_index(drop=True)
    )


def rare_cancer_fusion_rules_df():
    """Return the curated direct-fusion rare-cancer rule table."""
    from pirlygenes.load_dataset import get_data

    return get_data("rare-cancer-fusion-rules")


def _record_value(record, key: str, default=None):
    if hasattr(record, "get"):
        return record.get(key, default)
    return getattr(record, key, default)


def _rule_gene_set(value: object) -> set[str]:
    genes = set(_split_semicolon(value))
    return {gene.upper() for gene in genes if gene}


def _rule_gene_display(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return "?"
    return "/".join(_split_semicolon(text)) if ";" in text else text


def _fusion_record_key(record) -> tuple[str, str, int | None]:
    gene_a = str(_record_value(record, "gene_a", "") or "").strip().upper()
    gene_b = str(_record_value(record, "gene_b", "") or "").strip().upper()
    pair_key = "::".join(sorted((gene_a, gene_b)))
    source = str(_record_value(record, "source_path", "") or "")
    row_index = _record_value(record, "row_index", None)
    return pair_key, source, row_index


def _fusion_rule_match_detail(rule, record) -> dict[str, str] | None:
    gene_a = str(_record_value(record, "gene_a", "") or "").strip().upper()
    gene_b = str(_record_value(record, "gene_b", "") or "").strip().upper()
    if not gene_a or not gene_b:
        return None
    rule_a = _rule_gene_set(rule.get("gene_a"))
    rule_b = _rule_gene_set(rule.get("gene_b"))
    matching = str(rule.get("matching") or "oriented_or_unoriented").strip().lower()

    def _in(gene: str, genes: set[str]) -> bool:
        return "*" in genes or gene in genes

    direct = _in(gene_a, rule_a) and _in(gene_b, rule_b)
    reverse = _in(gene_b, rule_a) and _in(gene_a, rule_b)
    strict = matching in {"strict", "strict_5to3", "ordered", "direct", "5to3"}
    record_orientation = str(_record_value(record, "orientation", "") or "").lower()
    if direct:
        return {
            "matched_orientation": "as_reported",
            "orientation_note": "reported pair matches expected 5-prime/3-prime rule orientation",
        }
    if reverse and not strict and record_orientation != "5prime_3prime":
        return {
            "matched_orientation": "reverse_of_expected",
            "orientation_note": (
                "reported pair is reverse of the curated 5-prime/3-prime rule; "
                "confirm caller orientation"
            ),
        }
    return None


def _fusion_rule_specificity(rule) -> int:
    score = 0
    for column in ("gene_a", "gene_b"):
        genes = _rule_gene_set(rule.get(column))
        if genes and "*" not in genes:
            score += 1
    return score


def match_rare_cancer_fusion_rules(fusion_records) -> list[dict[str, Any]]:
    """Return curated rare-cancer fusion findings for supplied calls.

    The rule table is oriented as ``gene_a`` = expected 5-prime partner and
    ``gene_b`` = expected 3-prime partner. Matching is deliberately loose by
    default because caller outputs and manually supplied lists often omit or
    invert orientation. Findings retain both the reported pair and the expected
    5-prime/3-prime rule pair.
    """
    records = list(fusion_records or [])
    if not records:
        return []
    rules = rare_cancer_fusion_rules_df().fillna("")
    hits: list[tuple[tuple[float, int, float], dict[str, Any]]] = []
    confidence_rank = {"high": 2, "moderate": 1, "low": 0}
    for _, rule in rules.iterrows():
        min_support = _safe_float(rule.get("min_total_support"), 0.0)
        for record in records:
            match_detail = _fusion_rule_match_detail(rule, record)
            if not match_detail:
                continue
            support_total = _record_value(record, "support_total", None)
            if support_total is not None and _safe_float(support_total, 0.0) < min_support:
                continue
            confidence = str(rule.get("confidence") or "high").strip()
            promote = _safe_bool(rule.get("promote_report_scope"), default=True)
            specificity = _fusion_rule_specificity(rule)
            finding = {
                "cancer_type": str(rule.get("cancer_code") or "").strip(),
                "label": str(rule.get("label") or "").strip(),
                "rule_id": str(rule.get("rule_id") or "").strip(),
                "promote_report_scope": promote,
                "specificity": specificity,
                "expected_pair": (
                    f"{_rule_gene_display(rule.get('gene_a'))}"
                    f"--{_rule_gene_display(rule.get('gene_b'))}"
                ),
                "matched_orientation": match_detail["matched_orientation"],
                "orientation_note": match_detail["orientation_note"],
                "fusion": {
                    "gene_a": str(_record_value(record, "gene_a", "") or ""),
                    "gene_b": str(_record_value(record, "gene_b", "") or ""),
                    "pair": f"{_record_value(record, 'gene_a', '')}--{_record_value(record, 'gene_b', '')}",
                    "support_total": support_total,
                    "effect": str(_record_value(record, "effect", "") or ""),
                    "frame": str(_record_value(record, "frame", "") or ""),
                    "caller": str(_record_value(record, "caller", "") or ""),
                    "confidence": str(_record_value(record, "confidence", "") or ""),
                    "reportable": str(_record_value(record, "reportable", "") or ""),
                    "orientation": str(_record_value(record, "orientation", "") or ""),
                },
                "confidence": confidence,
                "basis": str(rule.get("basis") or "").strip(),
                "confirmatory_tests": str(rule.get("confirmatory_tests") or "").strip(),
                "caveat": str(rule.get("caveat") or "").strip(),
                "source": str(rule.get("source") or "").strip(),
            }
            support_score = _safe_float(support_total, min_support or 1.0)
            hits.append(
                (
                    (
                        1.0 if promote else 0.0,
                        confidence_rank.get(confidence.lower(), 0),
                        specificity,
                        support_score,
                    ),
                    finding,
                )
            )
    if not hits:
        return []
    hits.sort(key=lambda item: item[0], reverse=True)
    findings = [item[1] for item in hits]

    max_specificity_by_record: dict[tuple[str, str, int | None], int] = {}
    for finding in findings:
        key = _fusion_record_key(finding["fusion"])
        max_specificity_by_record[key] = max(
            max_specificity_by_record.get(key, 0),
            _safe_int(finding.get("specificity"), 0),
        )
    filtered: list[dict[str, Any]] = []
    for finding in findings:
        key = _fusion_record_key(finding["fusion"])
        if _safe_int(finding.get("specificity"), 0) < max_specificity_by_record.get(key, 0):
            continue
        filtered.append(finding)
    return filtered


def infer_rare_cancer_report_scope_from_fusions(fusion_records, analysis=None):
    """Return a rare-cancer report-scope hypothesis from direct fusions."""
    findings = match_rare_cancer_fusion_rules(fusion_records)
    if not findings:
        return None
    candidate_trace = (analysis or {}).get("candidate_trace") or []
    top_reference = (
        str(candidate_trace[0].get("code") or "").strip() if candidate_trace else ""
    )
    for finding in findings:
        if not finding.get("promote_report_scope"):
            continue
        if not finding.get("cancer_type"):
            continue
        result = dict(finding)
        result["top_reference_cancer_type"] = top_reference
        return result
    return None


def infer_rare_cancer_marker_hypotheses_from_rna(df_expr, analysis) -> list[dict[str, Any]]:
    """Return non-promoting rare-cancer RNA marker hypotheses.

    These markers are useful testing prompts but are not specific enough to
    replace the RNA classifier/report scope by themselves.
    """
    try:
        from .common import build_sample_tpm_by_symbol

        sample_tpm = build_sample_tpm_by_symbol(df_expr)
    except (ImportError, KeyError, ValueError, TypeError):
        return []

    candidate_trace = analysis.get("candidate_trace") or []
    top_reference = (
        str(candidate_trace[0].get("code") or "").strip() if candidate_trace else ""
    )
    top_codes = [
        str(row.get("code") or "").strip()
        for row in candidate_trace
        if str(row.get("code") or "").strip()
    ]

    rules = rare_cancer_rna_surrogate_rules_df().fillna("")
    findings: list[dict[str, Any]] = []
    for _, rule in rules.iterrows():
        evidence = _evaluate_rna_rule(rule, sample_tpm, top_codes)
        if evidence is None:
            continue
        findings.append(
            RareCancerRnaInference(
                cancer_type=str(rule.get("cancer_code") or "").strip(),
                rule_id=str(rule.get("rule_id") or "").strip(),
                surrogate=evidence["primary_gene"],
                surrogate_tpm=evidence["primary_tpm"],
                threshold_tpm=evidence["min_tpm"],
                top_reference_cancer_type=top_reference,
                confidence=str(rule.get("confidence") or "moderate").strip(),
                support_genes=tuple(evidence["support_genes"]),
                basis=str(rule.get("basis") or "").strip(),
                confirmatory_tests=str(rule.get("confirmatory_tests") or "").strip(),
                caveat=str(rule.get("caveat") or "").strip(),
                source=str(rule.get("source") or "").strip(),
                missing_support_genes=tuple(evidence["missing_support_genes"]),
                exclusion_genes_observed=tuple(evidence["exclusion_genes_observed"]),
                absent_genes_confirmed=tuple(evidence["absent_genes_confirmed"]),
                support_pass=bool(evidence["support_pass"]),
                min_support_genes=int(evidence["min_support_genes"]),
                required_support_gene_count=int(evidence["required_support_gene_count"]),
                promote_report_scope=False,
            ).public_dict()
        )
    findings.sort(
        key=lambda row: (
            len(row.get("support_genes") or []),
            float(row.get("surrogate_tpm") or 0.0)
            / max(float(row.get("threshold_tpm") or 1.0), 1e-9),
        ),
        reverse=True,
    )
    return findings
