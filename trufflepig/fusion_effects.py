"""Expression consequences of oncogenic fusions.

Direct fusion calls and downstream expression are separate evidence layers.
This module keeps the downstream layer data-backed: supplied fusion calls can
be checked against expected target-gene activation, while RNA-only signatures
can ask for fusion testing without pretending to be fusion calls.
"""

from __future__ import annotations

from typing import Any


def fusion_expression_effect_rules_df():
    """Return curated fusion -> expression-consequence rules."""
    from pirlygenes.load_dataset import get_data

    return get_data("fusion-expression-effects")


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


def _record_value(record, key: str, default=None):
    if hasattr(record, "get"):
        return record.get(key, default)
    return getattr(record, key, default)


def _rule_gene_set(value: object) -> set[str]:
    genes = set(_split_semicolon(value))
    return {gene.upper() for gene in genes if gene}


def _gene_thresholds(value: object, default_min_tpm: float) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for token in _split_semicolon(value):
        if ":" in token:
            gene, threshold = token.rsplit(":", 1)
            thresholds[gene.strip().upper()] = _safe_float(threshold, default_min_tpm)
        else:
            thresholds[token.strip().upper()] = default_min_tpm
    return {gene: tpm for gene, tpm in thresholds.items() if gene}


def _expression_value(
    sample_tpm_by_symbol: dict[str, Any] | None,
    gene: str,
    *,
    tumor_tpm_by_symbol: dict[str, Any] | None = None,
) -> tuple[float, str, float | None, float | None]:
    candidates = [gene, gene.upper(), gene.replace("-", ""), gene.upper().replace("-", "")]
    bulk_tpm: float | None = None
    tumor_tpm: float | None = None
    if tumor_tpm_by_symbol:
        for candidate in candidates:
            if candidate in tumor_tpm_by_symbol:
                tumor_tpm = _safe_float(tumor_tpm_by_symbol.get(candidate), 0.0)
                break
    if sample_tpm_by_symbol:
        for candidate in candidates:
            if candidate in sample_tpm_by_symbol:
                bulk_tpm = _safe_float(sample_tpm_by_symbol.get(candidate), 0.0)
                break
    if tumor_tpm is not None:
        return tumor_tpm, "tumor_inferred", bulk_tpm, tumor_tpm
    if bulk_tpm is not None:
        return bulk_tpm, "bulk", bulk_tpm, None
    return 0.0, "unavailable", None, None


def _fusion_rule_matches(rule, record) -> tuple[bool, str]:
    gene_a = str(_record_value(record, "gene_a", "") or "").strip().upper()
    gene_b = str(_record_value(record, "gene_b", "") or "").strip().upper()
    if not gene_a or not gene_b:
        return False, ""
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
        return True, "as_reported"
    if reverse and not strict and record_orientation != "5prime_3prime":
        return True, "reverse_of_expected"
    return False, ""


def _evaluate_rule_expression(
    rule,
    sample_tpm_by_symbol: dict[str, Any] | None,
    *,
    tumor_tpm_by_symbol: dict[str, Any] | None = None,
) -> dict:
    default_min = _safe_float(rule.get("min_gene_tpm"), 1.0)
    expected = _gene_thresholds(rule.get("expected_up_genes"), default_min)
    anchors = _gene_thresholds(rule.get("anchor_genes"), default_min)
    min_genes = _safe_int(rule.get("min_genes"), max(1, len(expected)))
    min_anchors = _safe_int(rule.get("min_anchor_genes"), len(anchors))

    gene_evidence = []
    observed_genes = []
    expression_sources = set()
    for gene, threshold in expected.items():
        tpm, source, bulk_tpm, tumor_tpm = _expression_value(
            sample_tpm_by_symbol,
            gene,
            tumor_tpm_by_symbol=tumor_tpm_by_symbol,
        )
        expression_sources.add(source)
        observed = tpm >= threshold
        if observed:
            observed_genes.append(gene)
        gene_evidence.append(
            {
                "gene": gene,
                "tpm": round(tpm, 3),
                "threshold_tpm": threshold,
                "observed": observed,
                "source": source,
                "bulk_tpm": round(bulk_tpm, 3) if bulk_tpm is not None else None,
                "tumor_tpm": round(tumor_tpm, 3) if tumor_tpm is not None else None,
            }
        )

    anchor_evidence = []
    observed_anchors = []
    for gene, threshold in anchors.items():
        tpm, source, bulk_tpm, tumor_tpm = _expression_value(
            sample_tpm_by_symbol,
            gene,
            tumor_tpm_by_symbol=tumor_tpm_by_symbol,
        )
        expression_sources.add(source)
        observed = tpm >= threshold
        if observed:
            observed_anchors.append(gene)
        anchor_evidence.append(
            {
                "gene": gene,
                "tpm": round(tpm, 3),
                "threshold_tpm": threshold,
                "observed": observed,
                "source": source,
                "bulk_tpm": round(bulk_tpm, 3) if bulk_tpm is not None else None,
                "tumor_tpm": round(tumor_tpm, 3) if tumor_tpm is not None else None,
            }
        )

    if not sample_tpm_by_symbol and not tumor_tpm_by_symbol:
        status = "not_evaluable"
    elif len(observed_anchors) < min_anchors:
        status = "anchor_absent"
    elif len(observed_genes) >= min_genes:
        status = "active"
    elif observed_genes:
        status = "partial"
    else:
        status = "not_evident"

    return {
        "status": status,
        "observed_genes": observed_genes,
        "observed_gene_count": len(observed_genes),
        "min_genes": min_genes,
        "expression_source": "mixed"
        if len(expression_sources - {"unavailable"}) > 1
        else next(iter(expression_sources - {"unavailable"}), "unavailable"),
        "gene_evidence": gene_evidence,
        "observed_anchors": observed_anchors,
        "anchor_evidence": anchor_evidence,
    }


def match_fusion_expression_effects(
    fusion_records,
    sample_tpm_by_symbol: dict[str, Any] | None,
    *,
    tumor_tpm_by_symbol: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Check supplied fusion calls against expected expression effects."""
    records = list(fusion_records or [])
    if not records:
        return []
    rules = fusion_expression_effect_rules_df().fillna("")
    findings: list[dict[str, Any]] = []
    for _, rule in rules.iterrows():
        for record in records:
            matched, orientation = _fusion_rule_matches(rule, record)
            if not matched:
                continue
            expression = _evaluate_rule_expression(
                rule,
                sample_tpm_by_symbol,
                tumor_tpm_by_symbol=tumor_tpm_by_symbol,
            )
            finding = {
                "rule_id": str(rule.get("rule_id") or "").strip(),
                "label": str(rule.get("label") or "").strip(),
                "fusion": {
                    "gene_a": str(_record_value(record, "gene_a", "") or ""),
                    "gene_b": str(_record_value(record, "gene_b", "") or ""),
                    "pair": f"{_record_value(record, 'gene_a', '')}--{_record_value(record, 'gene_b', '')}",
                    "support_total": _record_value(record, "support_total", None),
                    "orientation": str(_record_value(record, "orientation", "") or ""),
                },
                "expected_pair": f"{rule.get('gene_a')}--{rule.get('gene_b')}",
                "matched_orientation": orientation,
                "confidence": str(rule.get("confidence") or "moderate").strip(),
                "basis": str(rule.get("basis") or "").strip(),
                "caveat": str(rule.get("caveat") or "").strip(),
                "source": str(rule.get("source") or "").strip(),
                **expression,
            }
            findings.append(finding)
    return sorted(
        findings,
        key=lambda item: (
            item["status"] != "active",
            -item.get("observed_gene_count", 0),
            item["label"],
        ),
    )


def infer_fusion_expression_hypotheses(
    sample_tpm_by_symbol: dict[str, Any] | None,
    *,
    tumor_tpm_by_symbol: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return RNA-only fusion-effect hypotheses that should prompt testing."""
    if not sample_tpm_by_symbol:
        return []
    rules = fusion_expression_effect_rules_df().fillna("")
    hypotheses: list[dict[str, Any]] = []
    for _, rule in rules.iterrows():
        if not _safe_bool(rule.get("allow_expression_hypothesis"), default=False):
            continue
        expression = _evaluate_rule_expression(
            rule,
            sample_tpm_by_symbol,
            tumor_tpm_by_symbol=tumor_tpm_by_symbol,
        )
        if expression["status"] != "active":
            continue
        hypotheses.append(
            {
                "rule_id": str(rule.get("rule_id") or "").strip(),
                "label": str(rule.get("label") or "").strip(),
                "expected_pair": f"{rule.get('gene_a')}--{rule.get('gene_b')}",
                "confidence": str(rule.get("confidence") or "moderate").strip(),
                "basis": str(rule.get("basis") or "").strip(),
                "caveat": str(rule.get("caveat") or "").strip(),
                "source": str(rule.get("source") or "").strip(),
                "promote_report_scope": False,
                **expression,
            }
        )
    return sorted(
        hypotheses,
        key=lambda item: (-item.get("observed_gene_count", 0), item["label"]),
    )
