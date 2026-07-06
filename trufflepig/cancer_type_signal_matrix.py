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

from .cancer_ontology import cancer_family, cancer_lineage_group, registry_parent_code
from .expression_decomposition import _group_to_mode
from .reporting import cancer_code_display_name


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
    if lowered in {"solid", "mesenchymal", "heme", "embryonal"}:
        return lowered
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
        parts.append(rationale[:120])
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
    if stage in {"exact_subtype", "orthogonal_state"}:
        return "subtype"
    if stage == "coarse_type":
        return "entity"
    if _parent(code):
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
                status="ranked_candidate",
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
    row = rows.sort_values("_support_sort", ascending=False).iloc[0]
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
            final_call = _clean(sub["final_call"].dropna().astype(str).iloc[0]) if len(sub) else ""
            selected_by = _clean(sub["selected_by"].dropna().astype(str).iloc[0]) if len(sub) else ""
            strong = sub[
                (sub["support"].fillna(0).astype(float) >= 0.5)
                & (sub["entity_agrees_final"] == False)  # noqa: E712
                & (sub["is_context_only"] != True)  # noqa: E712
                & (sub["is_blocked"] != True)  # noqa: E712
            ]
            lines.append(
                f"| {sample} | {final_call or '—'} | `{selected_by or '—'}` | "
                f"{len(sub)} | {len(strong)} |"
            )
        lines.append("")
        return "\n".join(lines) + "\n"

    final_call = _clean(df["final_call"].dropna().astype(str).iloc[0])
    selected_by = _clean(df["selected_by"].dropna().astype(str).iloc[0])
    reference = _clean(df["reference_call"].dropna().astype(str).iloc[0])
    lines.append(f"- **Final call**: {final_call or '—'}.")
    if reference and reference != final_call:
        lines.append(f"- **Reference call**: {reference}.")
    if selected_by:
        lines.append(f"- **Selected by**: `{selected_by}`.")
    lines.append("")
    lines.append("| Signal | Top prediction | Layer | Status | Support | Agreement | Details |")
    lines.append("|---|---|---|---|---:|---|---|")
    ranked = (
        df.assign(_support=df["support"].fillna(-1).astype(float))
        .sort_values(["selects_report_label", "_support"], ascending=[False, False])
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
        detail_text = ""
        try:
            details = json.loads(row.get("details") or "{}")
        except json.JSONDecodeError:
            details = {}
        if isinstance(details, Mapping):
            detail_text = _details_summary(details)
        lines.append(
            f"| {row.get('signal_label') or row.get('signal_source') or '—'} | "
            f"{row.get('predicted_code') or '—'} | {row.get('ontology_layer') or '—'} | "
            f"{row.get('status') or '—'} | {support_text} | {agree} | "
            f"{detail_text or '—'} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


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
