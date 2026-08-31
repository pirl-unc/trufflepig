"""Tabular cancer-type evidence traces.

The markdown decision trace is optimized for human reading inside one report.
This module normalizes the same signals into a stable TSV schema so local and
565-sample sweeps can compare all cancer-type evidence channels side by side.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .cancer_ontology import (
    cancer_family,
    cancer_lineage_group,
    molecular_status_parent_code as status_parent_code,
    registry_parent_code,
    subtype_display_parent_code as subtype_parent_code,
)
from .expression_decomposition import _group_to_mode
from .reporting import cancer_code_display_name, md_table_cell


SIGNAL_MATRIX_COLUMNS = [
    "sample",
    "final_call",
    "final_lineage",
    "reference_call",
    "selected_by",
    "signal_source",
    "signal_label",
    "ontology_layer",
    "predicted_code",
    "predicted_label",
    "predicted_lineage",
    "predicted_family",
    "predicted_parent",
    "stage",
    "role",
    "status",
    "selects_report_label",
    "support",
    "confidence",
    "rank",
    "support_metric",
    "context_code",
    "entity_agrees_final",
    "lineage_agrees_final",
    "family_agrees_final",
    "is_blocked",
    "is_context_only",
    "details",
]


SIGNAL_SAMPLE_SUMMARY_COLUMNS = [
    "sample",
    "final_call",
    "final_lineage",
    "reference_call",
    "selected_by",
    "signal_rows",
    "strong_conflict_rows",
    "pan_cancer_top",
    "pan_cancer_top_support",
    "pan_cancer_top_rank",
    "pan_cancer_runner_up",
    "pan_cancer_runner_up_support",
    "learned_entity_top",
    "learned_entity_support",
    "learned_entity_margin",
    "learned_family_top",
    "learned_family_support",
    "learned_compartment_top",
    "learned_compartment_support",
    "fused_selected",
    "fused_selected_score",
    "fused_top_score",
    "fused_top_blocked",
    "fused_top_blocker",
    "entity_consensus_candidate",
    "entity_consensus_previous",
    "entity_consensus_decision",
    "entity_consensus_candidate_votes",
    "entity_consensus_selected_votes",
    "lineage_panel_top",
    "lineage_panel_support",
    "lineage_panel_status",
    "rare_marker_top",
    "rare_marker_support",
    "rare_marker_status",
    "background_site",
    "background_site_support",
    "decomposition_top",
    "decomposition_score",
]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _lineage_mode(code: str) -> str:
    code = _clean(code)
    if not code:
        return ""
    lowered = code.lower()
    learned_lineage_map = {
        "carcinoma": "solid",
        "epithelial": "solid",
        "solid": "solid",
        "sarcoma": "mesenchymal",
        "mesenchymal": "mesenchymal",
        "hematolymphoid": "heme",
        "heme": "heme",
        "cns": "embryonal",
        "embryonal": "embryonal",
    }
    if lowered in learned_lineage_map:
        return learned_lineage_map[lowered]
    try:
        group = cancer_lineage_group(code)
    except (KeyError, ValueError):
        group = None
    return _group_to_mode(group) if group else ""


def _family(code: str) -> str:
    try:
        return cancer_family(code)
    except (KeyError, ValueError):
        return ""


def _parent(code: str) -> str:
    try:
        return registry_parent_code(code)
    except (KeyError, ValueError):
        return ""


def _ancestor_chain(code: str) -> list[str]:
    code = _clean(code)
    out: list[str] = []
    while code and code not in out:
        out.append(code)
        code = _parent(code)
    return out


def _entity_agrees(code: str, final_call: str) -> bool | None:
    code = _clean(code)
    final_call = _clean(final_call)
    if not code or not final_call:
        return None
    left = set(_ancestor_chain(code))
    right = set(_ancestor_chain(final_call))
    if not left or not right:
        return None
    return bool(left & right)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _details_json(details: Any) -> str:
    if not details:
        return ""
    return json.dumps(_jsonable(details), sort_keys=True, separators=(",", ":"))


def _parse_details(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    text = _clean(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _details_summary(details: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, label in (
        ("probability", "p"),
        ("margin", "margin"),
        ("context_support", "context"),
        ("rho", "rho"),
        ("holdout_top1_accuracy", "held-out top1"),
        ("holdout_medoid_top1_accuracy", "medoid top1"),
        ("oof_top3_recovery", "top3 recovery"),
    ):
        value = _safe_float(details.get(key))
        if value is not None:
            parts.append(f"{label}={value:.3f}")
    top = details.get("top_predictions") or []
    if isinstance(top, Iterable) and not isinstance(top, (str, bytes)):
        formatted = []
        for item in list(top)[:3]:
            if not isinstance(item, Mapping):
                continue
            label = _clean(item.get("code") or item.get("label"))
            prob = _safe_float(item.get("probability"))
            if label and prob is not None:
                formatted.append(f"{label} {prob:.2f}")
            elif label:
                formatted.append(label)
        if formatted:
            parts.append("top " + ", ".join(formatted))
    rationale = _clean(details.get("rationale"))
    if rationale:
        if len(rationale) > 120:
            rationale = rationale[:117].rstrip() + "…"
        parts.append(rationale)
    return "; ".join(parts)


def _ontology_layer(stage: str, role: str, code: str) -> str:
    stage = _clean(stage)
    role = _clean(role)
    if role.startswith("hierarchical_compartment"):
        return "lineage"
    if role.startswith("hierarchical_family") or stage == "family":
        return "family"
    if role.startswith("hierarchical_entity"):
        return "entity"
    # exact_subtype / orthogonal_state is the SELECTOR's decision stage, not a claim about the code:
    # fused_evidence / learned_* / fine_reference all report "exact_subtype" even when they win a
    # base entity (COAD). Only a code that actually carries a subtype/status suffix ("_" present)
    # sits at the subtype layer — a base entity winner (incl. the fused headline) stays an entity.
    if stage in {"exact_subtype", "orthogonal_state"}:
        return "subtype" if "_" in code else "entity"
    if stage == "coarse_type":
        return "entity"
    # A base entity that merely rolls up to a group parent (COAD -> CRC) is still an entity winner —
    # `_parent(code)` is truthy for it, so gating on the "_" subtype marker keeps entity headlines
    # from being mislabeled "subtype".
    if "_" in code and _parent(code):
        return "subtype"
    return "context"


def _status_flags(status: str, selects_report_label: bool) -> tuple[bool, bool]:
    text = _clean(status).lower()
    blocked = (
        "blocked" in text
        or "vetoed" in text
        or text in {"not_selectable", "rejected"}
    )
    context_only = "context" in text or (not selects_report_label and text == "admission_context")
    return blocked, context_only


def _row(
    *,
    sample_id: str,
    final_call: str,
    reference_call: str,
    selected_by: str,
    signal_source: str,
    signal_label: str,
    stage: str = "",
    role: str = "",
    status: str = "",
    predicted_code: str = "",
    support: float | None = None,
    confidence: float | None = None,
    rank: int | None = None,
    support_metric: str = "",
    context_code: str = "",
    selects_report_label: bool = False,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    details = dict(details or {})
    code = _clean(predicted_code)
    layer = _ontology_layer(stage, role, code)
    lineage = _lineage_mode(code)
    if layer == "lineage" and not lineage:
        lineage = code.lower()
    final_lineage = _lineage_mode(final_call)
    predicted_family = _family(code)
    final_family = _family(final_call)
    is_blocked, is_context_only = _status_flags(status, selects_report_label)
    entity_agrees = _entity_agrees(code, final_call)
    lineage_agrees = bool(lineage and final_lineage and lineage == final_lineage)
    family_agrees = bool(predicted_family and final_family and predicted_family == final_family)
    label = cancer_code_display_name(code, code) if code else ""
    return {
        "sample": sample_id,
        "final_call": final_call,
        "final_lineage": final_lineage,
        "reference_call": reference_call,
        "selected_by": selected_by,
        "signal_source": _clean(signal_source),
        "signal_label": _clean(signal_label) or _clean(signal_source),
        "ontology_layer": layer,
        "predicted_code": code,
        "predicted_label": label,
        "predicted_lineage": lineage,
        "predicted_family": predicted_family,
        "predicted_parent": _parent(code),
        "stage": _clean(stage),
        "role": _clean(role),
        "status": _clean(status),
        "selects_report_label": bool(selects_report_label),
        "support": support,
        "confidence": confidence if confidence is not None else support,
        "rank": rank,
        "support_metric": _clean(support_metric),
        "context_code": _clean(context_code),
        "entity_agrees_final": entity_agrees,
        "lineage_agrees_final": lineage_agrees if lineage and final_lineage else None,
        "family_agrees_final": family_agrees if predicted_family and final_family else None,
        "is_blocked": is_blocked,
        "is_context_only": is_context_only,
        "details": _details_json(details),
    }


def _selected_scope(analysis: Mapping[str, Any]) -> Mapping[str, Any]:
    evidence = analysis.get("cancer_type_evidence") or {}
    if not isinstance(evidence, Mapping):
        return {}
    selected = evidence.get("selected") or {}
    return selected if isinstance(selected, Mapping) else {}


def build_cancer_type_signal_matrix(
    analysis: Mapping[str, Any],
    *,
    sample_id: str | None = None,
    decomp_results: Iterable[Any] | None = None,
) -> pd.DataFrame:
    """Return one normalized evidence row per cancer-type signal."""
    sample = _clean(sample_id) or _clean(analysis.get("sample_id")) or "sample"
    final_call = _clean(analysis.get("cancer_type") or analysis.get("inferred_cancer_type"))
    selected = _selected_scope(analysis)
    selected_by = _clean(selected.get("selected_by"))
    reference_call = _clean(
        analysis.get("reference_cancer_type")
        or selected.get("reference_cancer_type")
        or final_call
    )
    rows: list[dict[str, Any]] = []

    for rank, candidate in enumerate(analysis.get("candidate_trace") or [], start=1):
        if not isinstance(candidate, Mapping):
            continue
        code = _clean(candidate.get("code"))
        if not code:
            continue
        details = {
            key: candidate.get(key)
            for key in (
                "signature_score",
                "support_geomean",
                "purity_estimate",
                "lineage_concordance",
                "family_label",
                "winning_subtype",
                "centroid_correlation",
                "centroid_coarse_lineage",
                "compartment_in_set",
            )
            if candidate.get(key) is not None
        }
        rows.append(
            _row(
                sample_id=sample,
                final_call=final_call,
                reference_call=reference_call,
                selected_by=selected_by,
                signal_source="pan_cancer_signature_ranker",
                signal_label="Pan-cancer signature ranker",
                stage="coarse_type",
                role="ranked_candidate",
                status="candidate_generation",
                predicted_code=code,
                support=_safe_float(candidate.get("support_fraction_of_top")),
                confidence=_safe_float(candidate.get("support_geomean")),
                rank=rank,
                support_metric="support_fraction_of_top",
                selects_report_label=False,
                details=details,
            )
        )

    evidence = analysis.get("cancer_type_evidence") or {}
    graph = evidence.get("staged_evidence_graph") if isinstance(evidence, Mapping) else {}
    for channel in (graph or {}).get("channels") or []:
        if not isinstance(channel, Mapping):
            continue
        details = channel.get("details") or {}
        if not isinstance(details, Mapping):
            details = {"details": details}
        code = _clean(channel.get("code") or channel.get("candidate_code"))
        source = _clean(channel.get("channel"))
        role = _clean(channel.get("role"))
        context_code = _clean(channel.get("context_code"))
        if (
            source == "learned_expression_classifier"
            and role.startswith("hierarchical_")
            and not _clean(details.get("label_space"))
        ):
            context_code = ""
        rows.append(
            _row(
                sample_id=sample,
                final_call=final_call,
                reference_call=reference_call,
                selected_by=selected_by,
                signal_source=source,
                signal_label=source.replace("_", " ").title(),
                stage=_clean(channel.get("stage")),
                role=role,
                status=_clean(channel.get("status")),
                predicted_code=code,
                support=_safe_float(channel.get("support")),
                confidence=_safe_float(channel.get("support")),
                rank=_safe_float(details.get("rank")),
                support_metric="channel_support",
                context_code=context_code,
                selects_report_label=bool(channel.get("selects_report_label")),
                details=details,
            )
        )

    site_context = analysis.get("inferred_site_context") or {}
    if isinstance(site_context, Mapping) and site_context:
        details = dict(site_context)
        rows.append(
            _row(
                sample_id=sample,
                final_call=final_call,
                reference_call=reference_call,
                selected_by=selected_by,
                signal_source="background_site_context",
                signal_label="Background / met-site context",
                stage="post_label_context",
                role="host_background_model",
                status="context_only",
                predicted_code=_clean(site_context.get("site") or site_context.get("tissue")),
                support=_safe_float(site_context.get("score")),
                confidence=_safe_float(site_context.get("score")),
                support_metric="site_context_score",
                context_code=_clean(site_context.get("tissue")),
                selects_report_label=False,
                details=details,
            )
        )

    cancer_type_decision = analysis.get("cancer_type_decision") or {}
    if isinstance(cancer_type_decision, Mapping) and cancer_type_decision:
        predicted_code = _clean(cancer_type_decision.get("supported_code"))
        status = _clean(cancer_type_decision.get("status")) or "not_evaluable"
        details = {
            "reason": cancer_type_decision.get("reason"),
            "current_code": cancer_type_decision.get("current_code"),
            "panel_code": cancer_type_decision.get("panel_code"),
            "ontology_code": cancer_type_decision.get("ontology_code"),
            "decision_basis": cancer_type_decision.get("decision_basis"),
            "background_separation_confirmed": cancer_type_decision.get(
                "background_separation_confirmed"
            ),
            "background_attributed_genes": cancer_type_decision.get(
                "background_attributed_genes"
            ),
            "selection_allowed": cancer_type_decision.get(
                "selection_allowed"
            ),
            "block_reason": cancer_type_decision.get("block_reason"),
            "sample_mode": cancer_type_decision.get("sample_mode"),
            "supported_code_mode": cancer_type_decision.get(
                "supported_code_mode"
            ),
            "models_evaluated": cancer_type_decision.get("models_evaluated"),
            "realizations_evaluated": cancer_type_decision.get(
                "realizations_evaluated"
            ),
            "background_models": [
                {
                    "template": row.get("template"),
                    "components": row.get("components"),
                    "model_role": row.get("model_role"),
                    "realizations": row.get("realizations"),
                    "candidate_code": row.get("candidate_code"),
                    "panel_candidate": row.get("panel_candidate"),
                    "complete_panel_or_background_candidate": row.get(
                        "complete_panel_or_background_candidate"
                    ),
                    "background_attributed_expected_low_genes": row.get(
                        "background_attributed_expected_low_genes"
                    ),
                    "ontology_candidate": row.get("ontology_candidate"),
                }
                for row in (cancer_type_decision.get("background_models") or ())
                if isinstance(row, Mapping)
            ],
        }
        rows.append(
            _row(
                sample_id=sample,
                final_call=final_call,
                reference_call=reference_call,
                selected_by=selected_by,
                signal_source="decomposition_cancer_type_decision",
                signal_label="Decomposition cancer-type decision",
                stage="post_label_context",
                role="background_separated_cancer_type",
                status=status,
                predicted_code=predicted_code,
                support=None,
                confidence=None,
                support_metric="structural_unanimity",
                context_code=_clean(cancer_type_decision.get("current_code")),
                selects_report_label=False,
                details=details,
            )
        )

    for rank, result in enumerate(list(decomp_results or [])[:8], start=1):
        code = _clean(getattr(result, "cancer_type", ""))
        template = _clean(getattr(result, "template", ""))
        score = _safe_float(getattr(result, "score", None))
        details = {
            "template": template,
            "purity": getattr(result, "purity", None),
            "reconstruction_error": getattr(result, "reconstruction_error", None),
            "template_tissue_score": getattr(result, "template_tissue_score", None),
            "template_site_factor": getattr(result, "template_site_factor", None),
            "warnings": list(getattr(result, "warnings", ()) or ()),
            "site_evidence": getattr(result, "site_evidence", None),
        }
        rows.append(
            _row(
                sample_id=sample,
                final_call=final_call,
                reference_call=reference_call,
                selected_by=selected_by,
                signal_source="expression_decomposition",
                signal_label="Expression decomposition",
                stage="post_label_context",
                role="template_fit",
                status="selected_context" if rank == 1 else "retained_context",
                predicted_code=code,
                support=score,
                confidence=score,
                rank=rank,
                support_metric="decomposition_fit_score",
                context_code=template,
                selects_report_label=False,
                details=details,
            )
        )

    df = pd.DataFrame(rows)
    for column in SIGNAL_MATRIX_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[SIGNAL_MATRIX_COLUMNS].drop_duplicates(ignore_index=True)


def _first_value(series: pd.Series) -> str:
    non_empty = [_clean(value) for value in series.dropna().tolist()]
    return next((value for value in non_empty if value), "")


def _top_supported_row(
    df: pd.DataFrame,
    mask: pd.Series,
    *,
    include_context_only: bool = False,
    include_blocked: bool = False,
) -> pd.Series | None:
    sub = df[mask].copy()
    if sub.empty:
        return None
    if not include_context_only and "is_context_only" in sub.columns:
        sub = sub[sub["is_context_only"] != True]  # noqa: E712
    if not include_blocked and "is_blocked" in sub.columns:
        sub = sub[sub["is_blocked"] != True]  # noqa: E712
    if sub.empty:
        return None
    sub["_support_sort"] = pd.to_numeric(sub["support"], errors="coerce").fillna(-1)
    sub["_rank_sort"] = pd.to_numeric(sub["rank"], errors="coerce").fillna(9999)
    sub = sub.sort_values(
        ["selects_report_label", "_support_sort", "_rank_sort"],
        ascending=[False, False, True],
        kind="stable",  # deterministic tie order for the 565-sweep TSV (quicksort is not stable)
    )
    return sub.iloc[0]


def _learned_summary_for_stage(df: pd.DataFrame, stage: str) -> tuple[str, float | None, float | None]:
    rows = df[
        (df["signal_source"] == "learned_expression_classifier")
        & (df["role"].astype(str).str.contains(stage, na=False))
    ].copy()
    if rows.empty:
        return "", None, None
    rows["_support_sort"] = pd.to_numeric(rows["support"], errors="coerce").fillna(-1)
    row = rows.sort_values("_support_sort", ascending=False, kind="stable").iloc[0]
    details = _parse_details(row.get("details"))
    label = _clean(row.get("predicted_code"))
    support = _safe_float(row.get("support"))
    margin = _safe_float(details.get("margin"))
    return label, support, margin


def build_signal_sample_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    """Return a compact one-row-per-sample view of a signal matrix."""

    if matrix is None or len(matrix) == 0:
        return pd.DataFrame(columns=SIGNAL_SAMPLE_SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    df = matrix.copy()
    if "support" in df.columns:
        df["support"] = pd.to_numeric(df["support"], errors="coerce")
    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    for sample, sub in df.groupby("sample", sort=True):
        final_call = _first_value(sub["final_call"])
        final_lineage = _first_value(sub["final_lineage"])
        reference_call = _first_value(sub["reference_call"])
        selected_by = _first_value(sub["selected_by"])
        strong_conflicts = sub[
            (pd.to_numeric(sub["support"], errors="coerce").fillna(0) >= 0.5)
            & (sub["entity_agrees_final"] == False)  # noqa: E712
            & (sub["is_context_only"] != True)  # noqa: E712
            & (sub["is_blocked"] != True)  # noqa: E712
        ]

        ranker_rows = sub[
            (sub["signal_source"] == "pan_cancer_signature_ranker")
            & (sub["role"] == "ranked_candidate")
        ].copy()
        ranker_rows["_rank_sort"] = pd.to_numeric(
            ranker_rows["rank"],
            errors="coerce",
        ).fillna(9999)
        ranker_rows["_support_sort"] = pd.to_numeric(
            ranker_rows["support"],
            errors="coerce",
        ).fillna(-1)
        ranker_rows = ranker_rows.sort_values(
            ["_rank_sort", "_support_sort"],
            ascending=[True, False],
            kind="stable",  # stable so drop_duplicates keeps a deterministic row on ties
        ).drop_duplicates("predicted_code")
        ranker_top = ranker_rows.iloc[0] if len(ranker_rows) else None
        ranker_runner_up = ranker_rows.iloc[1] if len(ranker_rows) > 1 else None

        learned_entity, learned_entity_support, learned_entity_margin = (
            _learned_summary_for_stage(sub, "entity")
        )
        learned_family, learned_family_support, _family_margin = (
            _learned_summary_for_stage(sub, "family")
        )
        learned_compartment, learned_compartment_support, _compartment_margin = (
            _learned_summary_for_stage(sub, "compartment")
        )

        fused_selected = _top_supported_row(
            sub,
            (sub["signal_source"] == "fused_evidence")
            & (sub["selects_report_label"] == True),  # noqa: E712
            include_blocked=True,
        )
        fused_top = _top_supported_row(
            sub,
            sub["signal_source"] == "fused_evidence",
            include_blocked=True,
        )
        fused_details = _parse_details(fused_top.get("details")) if fused_top is not None else {}
        fused_blockers = fused_details.get("blockers") or []
        if isinstance(fused_blockers, str):
            fused_blocker = fused_blockers
        elif isinstance(fused_blockers, list) and fused_blockers:
            fused_blocker = _clean(fused_blockers[0])
        else:
            fused_blocker = ""
        fused_top_blocked = bool(
            fused_top is not None
            and (bool(fused_top.get("is_blocked")) or bool(fused_blocker))
        )
        entity_consensus = _top_supported_row(
            sub,
            sub["signal_source"] == "entity_evidence_consensus",
            include_blocked=True,
        )
        entity_consensus_details = (
            _parse_details(entity_consensus.get("details"))
            if entity_consensus is not None
            else {}
        )

        lineage_panel = _top_supported_row(
            sub,
            sub["signal_source"] == "lineage_panel",
            include_blocked=True,
        )
        rare_marker = _top_supported_row(
            sub,
            sub["signal_source"].astype(str).str.contains("rare|fusion", case=False, na=False),
            include_blocked=True,
        )
        site_context = _top_supported_row(
            sub,
            sub["signal_source"] == "background_site_context",
            include_context_only=True,
            include_blocked=True,
        )
        decomposition = _top_supported_row(
            sub,
            sub["signal_source"] == "expression_decomposition",
            include_context_only=True,
            include_blocked=True,
        )

        rows.append(
            {
                "sample": sample,
                "final_call": final_call,
                "final_lineage": final_lineage,
                "reference_call": reference_call,
                "selected_by": selected_by,
                "signal_rows": int(len(sub)),
                "strong_conflict_rows": int(len(strong_conflicts)),
                "pan_cancer_top": _clean(ranker_top.get("predicted_code")) if ranker_top is not None else "",
                "pan_cancer_top_support": _safe_float(ranker_top.get("support")) if ranker_top is not None else None,
                "pan_cancer_top_rank": _safe_float(ranker_top.get("rank")) if ranker_top is not None else None,
                "pan_cancer_runner_up": _clean(ranker_runner_up.get("predicted_code")) if ranker_runner_up is not None else "",
                "pan_cancer_runner_up_support": _safe_float(ranker_runner_up.get("support")) if ranker_runner_up is not None else None,
                "learned_entity_top": learned_entity,
                "learned_entity_support": learned_entity_support,
                "learned_entity_margin": learned_entity_margin,
                "learned_family_top": learned_family,
                "learned_family_support": learned_family_support,
                "learned_compartment_top": learned_compartment,
                "learned_compartment_support": learned_compartment_support,
                "fused_selected": _clean(fused_selected.get("predicted_code")) if fused_selected is not None else "",
                "fused_selected_score": _safe_float(fused_selected.get("support")) if fused_selected is not None else None,
                "fused_top_score": _safe_float(fused_top.get("support")) if fused_top is not None else None,
                "fused_top_blocked": fused_top_blocked,
                "fused_top_blocker": fused_blocker,
                "entity_consensus_candidate": _clean(
                    entity_consensus_details.get("candidate_code")
                ),
                "entity_consensus_previous": _clean(
                    entity_consensus_details.get("selected_code")
                ),
                "entity_consensus_decision": _clean(
                    entity_consensus.get("role")
                )
                if entity_consensus is not None
                else "",
                "entity_consensus_candidate_votes": _safe_float(
                    entity_consensus_details.get("candidate_votes")
                ),
                "entity_consensus_selected_votes": _safe_float(
                    entity_consensus_details.get("selected_votes")
                ),
                "lineage_panel_top": _clean(lineage_panel.get("predicted_code")) if lineage_panel is not None else "",
                "lineage_panel_support": _safe_float(lineage_panel.get("support")) if lineage_panel is not None else None,
                "lineage_panel_status": _clean(lineage_panel.get("status")) if lineage_panel is not None else "",
                "rare_marker_top": _clean(rare_marker.get("predicted_code")) if rare_marker is not None else "",
                "rare_marker_support": _safe_float(rare_marker.get("support")) if rare_marker is not None else None,
                "rare_marker_status": _clean(rare_marker.get("status")) if rare_marker is not None else "",
                "background_site": _clean(site_context.get("predicted_code")) if site_context is not None else "",
                "background_site_support": _safe_float(site_context.get("support")) if site_context is not None else None,
                "decomposition_top": _clean(decomposition.get("predicted_code")) if decomposition is not None else "",
                "decomposition_score": _safe_float(decomposition.get("support")) if decomposition is not None else None,
            }
        )
    out = pd.DataFrame(rows)
    for column in SIGNAL_SAMPLE_SUMMARY_COLUMNS:
        if column not in out.columns:
            out[column] = None
    return out[SIGNAL_SAMPLE_SUMMARY_COLUMNS]


def _first_clean(series) -> str:
    """First non-null value of a column, cleaned; ``""`` if the column is empty/all-NaN.

    Guards the ``.dropna().iloc[0]`` out-of-bounds crash when a matrix (or a per-sample group) has an
    all-empty ``selected_by`` / ``final_call`` / ``reference_call`` column — e.g. a single-sample
    summary built before any selector populated those fields (seen regenerating a fresh report).
    """
    non_null = series.dropna().astype(str)
    return _clean(non_null.iloc[0]) if len(non_null) else ""


def build_signal_matrix_summary_markdown(
    matrix: pd.DataFrame,
    *,
    title: str = "Cancer-Type Signal Matrix Summary",
    max_rows: int = 12,
) -> str:
    """Compact markdown summary for one or many samples."""
    if matrix is None or len(matrix) == 0:
        return f"# {title}\n\nNo cancer-type signal rows were available.\n"
    df = matrix.copy()
    lines = [f"# {title}", ""]
    sample_count = int(df["sample"].nunique()) if "sample" in df.columns else 1
    lines.append(f"Rows: **{len(df)}** across **{sample_count}** sample(s).")
    lines.append("")
    if sample_count > 1:
        lines.append("| Sample | Final call | Selected by | Signals | Strong conflicts |")
        lines.append("|---|---|---|---:|---:|")
        for sample, sub in df.groupby("sample", sort=True):
            final_call = _first_clean(sub["final_call"])
            selected_by = _first_clean(sub["selected_by"])
            support = pd.to_numeric(sub["support"], errors="coerce").fillna(0.0)
            strong = sub[
                (support >= 0.5)
                & (sub["entity_agrees_final"] == False)  # noqa: E712
                & (sub["is_context_only"] != True)  # noqa: E712
                & (sub["is_blocked"] != True)  # noqa: E712
            ]
            lines.append(
                f"| {md_table_cell(sample) or '—'} | {md_table_cell(final_call) or '—'} | "
                f"`{md_table_cell(selected_by) or '—'}` | {len(sub)} | {len(strong)} |"
            )
        lines.append("")
        return "\n".join(lines) + "\n"

    final_call = _first_clean(df["final_call"])
    selected_by = _first_clean(df["selected_by"])
    reference = _first_clean(df["reference_call"])
    lines.append(f"- **Final call**: {final_call or '—'}.")
    if reference and reference != final_call:
        lines.append(f"- **Reference call**: {reference}.")
    if selected_by:
        lines.append(f"- **Selected by**: `{selected_by}`.")
    lines.append("")
    lines.append("| Signal | Top prediction | Layer | Status | Support | Agreement | Details |")
    lines.append("|---|---|---|---|---:|---|---|")
    ranked = (
        df.assign(
            _support=pd.to_numeric(df["support"], errors="coerce").fillna(-1.0)
        )
        .sort_values(["selects_report_label", "_support"], ascending=[False, False], kind="stable")
        .head(max_rows)
    )
    for _, row in ranked.iterrows():
        agree = (
            "entity"
            if row.get("entity_agrees_final") is True
            else ("lineage" if row.get("lineage_agrees_final") is True else "no")
        )
        support = row.get("support")
        support_text = f"{float(support):.3f}" if isinstance(support, (int, float)) else ""
        details = _parse_details(row.get("details"))
        detail_text = _details_summary(details)
        signal_name = md_table_cell(row.get("signal_label") or row.get("signal_source"))
        lines.append(
            f"| {signal_name or '—'} | "
            f"{md_table_cell(row.get('predicted_code')) or '—'} | {md_table_cell(row.get('ontology_layer')) or '—'} | "
            f"{md_table_cell(row.get('status')) or '—'} | {support_text} | {agree} | "
            f"{md_table_cell(detail_text) or '—'} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def compact_signal_plot_rows(matrix: pd.DataFrame, *, max_rows: int = 18) -> pd.DataFrame:
    """Return rows used by the compact cancer-type signal plot.

    This is intentionally smaller than the TSV: repeated learned MMR
    votes and orthogonal MSI/MSS/HPV/mutation-status labels are collapsed
    into one displayed row per interpreted signal so the plot supports
    the report text instead of replaying every trace row.
    """

    if matrix is None or len(matrix) == 0:
        return pd.DataFrame()

    df = matrix.copy()
    df["_support"] = pd.to_numeric(df["support"], errors="coerce").fillna(0.0)
    df["_support_low"] = df["_support"]
    df["_support_high"] = df["_support"]
    df["_selected"] = df["selects_report_label"].fillna(False).astype(bool)
    df["_details_map"] = df["details"].map(_parse_details)
    df["_label_space"] = df["_details_map"].map(
        lambda details: _clean(details.get("label_space"))
    )
    df["_mmr"] = (
        df["role"].fillna("").astype(str).str.contains("mismatch_repair", case=False)
        | df["signal_source"].fillna("").astype(str).str.contains("mismatch", case=False)
    )
    release_mmr = (
        df["_mmr"]
        & df["_label_space"].astype(str).str.contains("release_ensemble", na=False)
    )
    if release_mmr.any():
        df = df[
            (~df["_mmr"])
            | df["_label_space"].astype(str).str.contains("release_ensemble", na=False)
        ].copy()

    # The reader-facing chart is a decision summary, not the full audit trace.
    # Blocked counterfactuals (especially unsupported fusion/status subtypes)
    # remain in the TSV and evidence appendix, but letting their normalized
    # support bars fill the chart makes them look like live diagnoses.
    blocked = df["is_blocked"].fillna(False).astype(bool)
    df = df[(~blocked) | df["_selected"]].copy()
    if df.empty:
        return df

    def display_code(row: pd.Series) -> tuple[str, str, str]:
        pred = _clean(row.get("predicted_code"))
        status_parent = status_parent_code(pred)
        if status_parent and status_parent != pred:
            return status_parent, pred, "status"
        subtype_parent = subtype_parent_code(pred)
        if subtype_parent and subtype_parent != pred:
            return subtype_parent, pred, "subtype"
        return pred, "", ""

    def display_bucket(row: pd.Series) -> str:
        source = _clean(row.get("signal_source"))
        role = _clean(row.get("role"))
        pred = _clean(row.get("predicted_code"))
        context = _clean(row.get("context_code"))
        label_space = _clean(row.get("_label_space"))
        code, variant, _variant_kind = display_code(row)
        if row.get("_mmr"):
            if "release_ensemble" in label_space:
                return f"mmr:{label_space}:{pred}"
            return f"mmr:{label_space}:{context or code or pred}"
        if source in {"exact_expression_reference", "fused_evidence"}:
            return f"{source}:{code or pred}"
        if source == "learned_expression_classifier" and role.startswith("hierarchical_"):
            return f"{source}:{role}:{pred}"
        if source == "pan_cancer_signature_ranker":
            return f"{source}:{code or pred}"
        if source == "expression_decomposition":
            return f"{source}:{code or pred}:{context}:{row.get('status')}"
        if variant:
            return f"{source}:{role}:{code}:{context}:{row.get('status')}"
        return f"{source}:{role}:{pred}:{context}:{row.get('status')}"

    def display_label(row: pd.Series) -> str:
        source = _clean(row.get("signal_source"))
        role = _clean(row.get("role"))
        pred = _clean(row.get("predicted_code"))
        context = _clean(row.get("context_code"))
        code, variant, variant_kind = display_code(row)
        variant_note = f" ({variant_kind} row {variant})" if variant else ""
        if row.get("_mmr"):
            n = _safe_float(row.get("_mmr_n_votes"))
            votes = f" ({int(n)} candidate votes)" if n and n > 1 else ""
            return f"MMR release ensemble: {pred}-like{votes}"
        if source == "learned_expression_classifier":
            role_labels = {
                "hierarchical_compartment_vote": "Learned compartment",
                "hierarchical_family_vote": "Learned family",
                "hierarchical_entity_vote": "Learned entity",
                "hierarchical_subtype_axis_vote": "Learned subtype axis",
            }
            label = role_labels.get(role, "Learned expression")
            return f"{label}: {pred or context}"
        if source == "fused_evidence":
            return f"Fused evidence: {code or pred or role}{variant_note}"
        if source == "pan_cancer_signature_ranker":
            rank = _safe_float(row.get("rank"))
            rank_text = f" rank {int(rank)}" if rank is not None else ""
            return f"Pan-cancer signature ranker{rank_text}: {code or pred}{variant_note}"
        if source == "expression_decomposition":
            context_text = f" / {context}" if context else ""
            return f"Decomposition fit: {code or pred}{context_text}"
        if source == "background_site_context":
            return f"Background/site context: {pred or context}"
        if source == "exact_expression_reference":
            return f"Exact expression reference: {code or pred}{variant_note}"
        if source == "rare_fusion_anchor":
            return f"Rare fusion subtype anchor: {code or pred}{variant_note}"
        readable = source.replace("_", " ")
        return f"{readable}: {code or pred or role}{variant_note}"

    def display_color(row: pd.Series) -> str:
        if row.get("_mmr"):
            return "#7c3aed"
        if row.get("selects_report_label") is True:
            return "#059669"
        if row.get("is_blocked") is True:
            return "#dc2626"
        if row.get("is_context_only") is True:
            return "#6b7280"
        if row.get("entity_agrees_final") is True or row.get("lineage_agrees_final") is True:
            return "#2563eb"
        return "#f59e0b"

    df["_display_bucket"] = df.apply(display_bucket, axis=1)
    # Collapse the many near-duplicate per-candidate MMR MSI/MSS votes (one row
    # per contextualized status code) into a single MSS-vs-MSI summary bar that
    # carries the mean support and its min/max spread, instead of ~13 stacked
    # purple bars that all restate the same underlying MMR ensemble call.
    mmr_mask = df["_mmr"].astype(bool)
    mmr_summary = None
    if int(mmr_mask.sum()) > 1:
        supports = df.loc[mmr_mask, "_support"].astype(float)
        mmr_summary = {
            "support": float(supports.mean()),
            "low": float(supports.min()),
            "high": float(supports.max()),
            "n": int(mmr_mask.sum()),
        }
        df.loc[mmr_mask, "_display_bucket"] = "mmr:summary"
    ranked = (
        df.sort_values(["_selected", "_mmr", "_support"], ascending=[False, False, False], kind="stable")
        .drop_duplicates("_display_bucket", keep="first")
        .head(max_rows)
        .copy()
    )
    if ranked.empty:
        return ranked
    if mmr_summary is not None:
        # The dominant (highest-support) vote survives dedup and represents the
        # call; restate its bar as the vote mean with the spread as an error bar.
        sel = ranked["_display_bucket"] == "mmr:summary"
        ranked.loc[sel, "_support"] = mmr_summary["support"]
        ranked.loc[sel, "_support_low"] = mmr_summary["low"]
        ranked.loc[sel, "_support_high"] = mmr_summary["high"]
        ranked.loc[sel, "_mmr_n_votes"] = mmr_summary["n"]
    ranked["display_label"] = ranked.apply(display_label, axis=1)
    ranked["display_color"] = ranked.apply(display_color, axis=1)
    return ranked


def plot_cancer_type_signal_matrix(
    matrix: pd.DataFrame,
    out_path: str | Path,
    *,
    max_rows: int = 18,
    dpi: int = 180,
) -> str | None:
    """Plot the highest-support cancer-type evidence rows for one sample."""

    if matrix is None or len(matrix) == 0:
        return None
    import matplotlib.pyplot as plt

    ranked = compact_signal_plot_rows(matrix, max_rows=max_rows)
    if ranked.empty:
        return None

    # Fused-evidence support is an unbounded aggregate score (>1), while learned
    # probabilities, MMR calls, and support fractions live on 0-1. Sharing one
    # x-axis lets the big fused bars crush every probability bar into a sliver,
    # so draw the two scales on separate stacked panels.
    support = ranked["_support"].astype(float)
    scored = ranked[support > 1.0]
    prob = ranked[support <= 1.0]
    panels = [
        (sub, xlabel)
        for sub, xlabel in (
            (scored, "Fused evidence support (unbounded score)"),
            (prob, "Probability / support fraction (0-1)"),
        )
        if not sub.empty
    ]
    if not panels:
        return None

    def _draw(ax, sub: pd.DataFrame, xlabel: str) -> None:
        supp = sub["_support"].astype(float)
        y = list(range(len(sub)))[::-1]
        ax.barh(y, supp.tolist(), color=sub["display_color"].tolist(), alpha=0.86)
        # Draw spread whiskers only on rows that actually carry a spread (the MMR
        # vote summary), so zero-width caps don't clutter the point estimates.
        err_lo = (supp - sub["_support_low"].astype(float)).clip(lower=0.0)
        err_hi = (sub["_support_high"].astype(float) - supp).clip(lower=0.0)
        spread = (err_lo + err_hi) > 0
        if bool(spread.any()):
            keep = spread.tolist()
            ax.errorbar(
                [v for v, k in zip(supp.tolist(), keep) if k],
                [v for v, k in zip(y, keep) if k],
                xerr=[
                    [v for v, k in zip(err_lo.tolist(), keep) if k],
                    [v for v, k in zip(err_hi.tolist(), keep) if k],
                ],
                fmt="none",
                ecolor="#374151",
                elinewidth=1.0,
                capsize=3,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(sub["display_label"].tolist(), fontsize=8)
        ax.set_xlim(0, max(1.0, float(supp.max()) * 1.08))
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

    heights = [max(1, len(sub)) for sub, _ in panels]
    total = sum(heights)
    fig_h = max(4.8, 0.36 * total + 1.3 * len(panels))
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(10.5, fig_h),
        gridspec_kw={"height_ratios": heights},
        squeeze=False,
    )
    for ax, (sub, xlabel) in zip(axes[:, 0], panels):
        _draw(ax, sub, xlabel)
    final_calls = ranked["final_call"].dropna().astype(str)
    final_call = _clean(final_calls.iloc[0]) if not final_calls.empty else ""
    axes[0, 0].set_title(f"Cancer-type decision evidence - final call {final_call}")
    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def write_cancer_type_signal_artifacts(
    prefix: str,
    analysis: Mapping[str, Any],
    *,
    sample_id: str | None = None,
    decomp_results: Iterable[Any] | None = None,
) -> tuple[str, str, pd.DataFrame]:
    """Write ``*-cancer-type-signal-matrix.tsv`` and compact markdown summary."""
    matrix = build_cancer_type_signal_matrix(
        analysis,
        sample_id=sample_id,
        decomp_results=decomp_results,
    )
    base = _clean(prefix) or "sample"
    tsv_path = f"{base}-cancer-type-signal-matrix.tsv"
    md_path = f"{base}-cancer-type-signal-summary.md"
    Path(tsv_path).parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(tsv_path, sep="\t", index=False)
    Path(md_path).write_text(build_signal_matrix_summary_markdown(matrix))
    return tsv_path, md_path, matrix
