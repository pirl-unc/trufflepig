"""Expression effects of non-fusion alterations.

These rules are deliberately hypothesis-level. They use cancer-aware,
tumor-attributed expression when available to say that an expression pattern
is compatible with a mutation/copy-number class, or not compatible with a
supplied class. They do not call DNA mutations from RNA alone.
"""

from __future__ import annotations

from typing import Any


def mutation_expression_effect_rules_df():
    """Return curated mutation/CNV -> expression-consequence rules."""
    from pirlygenes.load_dataset import get_data

    return get_data("mutation-expression-effects")


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


def _rule_cancer_matches(rule, cancer_code: str | None) -> bool:
    codes = {code.upper() for code in _split_semicolon(rule.get("cancer_codes"))}
    if not codes or "*" in codes:
        return True
    code = str(cancer_code or "").strip().upper()
    return bool(code and code in codes)


def _evaluate_markers(
    rule,
    sample_tpm_by_symbol: dict[str, Any] | None,
    *,
    tumor_tpm_by_symbol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    default_min = _safe_float(rule.get("min_gene_tpm"), 1.0)
    up = _gene_thresholds(rule.get("expected_up_genes"), default_min)
    down = _gene_thresholds(rule.get("expected_low_genes"), default_min)
    anchors = _gene_thresholds(rule.get("anchor_genes"), default_min)
    min_up = _safe_int(rule.get("min_up_genes"), max(1, len(up)))
    min_low = _safe_int(rule.get("min_low_genes"), len(down))
    min_anchors = _safe_int(rule.get("min_anchor_genes"), len(anchors))

    evidence = []
    observed_up = []
    observed_low = []
    observed_anchors = []
    expression_sources = set()

    for gene, threshold in up.items():
        value, source, bulk_tpm, tumor_tpm = _expression_value(
            sample_tpm_by_symbol,
            gene,
            tumor_tpm_by_symbol=tumor_tpm_by_symbol,
        )
        expression_sources.add(source)
        observed = source != "unavailable" and value >= threshold
        if observed:
            observed_up.append(gene)
        evidence.append(
            {
                "gene": gene,
                "direction": "up",
                "tpm": round(value, 3),
                "threshold_tpm": threshold,
                "observed": observed,
                "source": source,
                "bulk_tpm": round(bulk_tpm, 3) if bulk_tpm is not None else None,
                "tumor_tpm": round(tumor_tpm, 3) if tumor_tpm is not None else None,
            }
        )

    for gene, threshold in down.items():
        value, source, bulk_tpm, tumor_tpm = _expression_value(
            sample_tpm_by_symbol,
            gene,
            tumor_tpm_by_symbol=tumor_tpm_by_symbol,
        )
        expression_sources.add(source)
        observed = source != "unavailable" and value <= threshold
        if observed:
            observed_low.append(gene)
        evidence.append(
            {
                "gene": gene,
                "direction": "low",
                "tpm": round(value, 3),
                "threshold_tpm": threshold,
                "observed": observed,
                "source": source,
                "bulk_tpm": round(bulk_tpm, 3) if bulk_tpm is not None else None,
                "tumor_tpm": round(tumor_tpm, 3) if tumor_tpm is not None else None,
            }
        )

    anchor_evidence = []
    for gene, threshold in anchors.items():
        value, source, bulk_tpm, tumor_tpm = _expression_value(
            sample_tpm_by_symbol,
            gene,
            tumor_tpm_by_symbol=tumor_tpm_by_symbol,
        )
        expression_sources.add(source)
        observed = source != "unavailable" and value >= threshold
        if observed:
            observed_anchors.append(gene)
        anchor_evidence.append(
            {
                "gene": gene,
                "tpm": round(value, 3),
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
    elif len(observed_up) >= min_up and len(observed_low) >= min_low:
        status = "compatible"
    elif observed_up or observed_low:
        status = "partial"
    else:
        status = "not_evident"

    return {
        "status": status,
        "observed_up_genes": observed_up,
        "observed_low_genes": observed_low,
        "observed_anchor_genes": observed_anchors,
        "observed_marker_count": len(observed_up) + len(observed_low),
        "min_up_genes": min_up,
        "min_low_genes": min_low,
        "marker_evidence": evidence,
        "anchor_evidence": anchor_evidence,
        "expression_source": "mixed"
        if len(expression_sources - {"unavailable"}) > 1
        else next(iter(expression_sources - {"unavailable"}), "unavailable"),
    }


def infer_mutation_expression_hypotheses(
    sample_tpm_by_symbol: dict[str, Any] | None,
    *,
    tumor_tpm_by_symbol: dict[str, Any] | None = None,
    cancer_code: str | None = None,
) -> list[dict[str, Any]]:
    """Return expression patterns compatible with curated alteration classes."""
    rules = mutation_expression_effect_rules_df().fillna("")
    hypotheses: list[dict[str, Any]] = []
    for _, rule in rules.iterrows():
        if not _safe_bool(rule.get("allow_expression_hypothesis"), default=True):
            continue
        if not _rule_cancer_matches(rule, cancer_code):
            continue
        evidence = _evaluate_markers(
            rule,
            sample_tpm_by_symbol,
            tumor_tpm_by_symbol=tumor_tpm_by_symbol,
        )
        if evidence["status"] != "compatible":
            continue
        hypotheses.append(
            {
                "rule_id": str(rule.get("rule_id") or "").strip(),
                "cancer_codes": _split_semicolon(rule.get("cancer_codes")),
                "alteration": str(rule.get("alteration") or "").strip(),
                "label": str(rule.get("label") or "").strip(),
                "confidence": str(rule.get("confidence") or "moderate").strip(),
                "basis": str(rule.get("basis") or "").strip(),
                "suggested_assay": str(rule.get("suggested_assay") or "").strip(),
                "caveat": str(rule.get("caveat") or "").strip(),
                "source": str(rule.get("source") or "").strip(),
                "promote_report_scope": False,
                **evidence,
            }
        )
    return sorted(
        hypotheses,
        key=lambda item: (-item.get("observed_marker_count", 0), item["label"]),
    )
