"""Structured longitudinal deltas across completed analyze runs.

The Markdown comparison built by :func:`build_analyze_comparison_markdown`
is the human-facing view. This module computes the *machine* view:
typed observations (cancer-call shifts, purity drift, response-axis
movement, target gains/losses, assay-context changes) that an LLM,
clinician dashboard, or downstream tooling can consume directly.

Tracking issue: pirl-unc/pirlygenes#230 (model explicit longitudinal
deltas in analyze comparisons).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


# ---------- Per-sample structured state ----------


@dataclass(frozen=True)
class ResponseAxisState:
    """One row of the per-axis therapy-state context.

    Mirrors what trufflepig's analyze() prints to the evidence.md
    "Therapy-state context" section, e.g.::

        **MAPK EGFR signaling** — active.
        Active signaling: up-panel geomean 3.52× cohort

    Fields are normalized lower-snake-case so per-axis matching across
    samples doesn't depend on punctuation.
    """

    axis: str            # e.g. "MAPK_EGFR_signaling"
    state: str           # "active" / "suppressed" / "mixed" / "inconclusive" / ...
    up_fold: float | None
    down_fold: float | None
    note: str = ""


@dataclass(frozen=True)
class TargetShortlistEntry:
    """One row from the summary.md "Top candidate therapies" section.

    The summary line is:
        - **GENE** — drug (Approved/Trial, indication). tumor-supported;
          XXX tumor-source bulk TPM (...); ...
    """

    gene: str
    drug: str
    indication: str
    tpm: float | None
    tier: str            # "Approved", "Trial", "Approved, advanced ..." etc.
    raw: str = ""        # full line as rendered


def _norm_axis(text: str) -> str:
    """Canonicalize an axis label ('MAPK EGFR signaling') to a stable id."""
    cleaned = re.sub(r"\s+", "_", text.strip())
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", cleaned)
    return cleaned


_AXIS_HEADER_RE = re.compile(r"^\*\*([^*]+?)\*\*\s+[—–-]\s+(.+)$")
_AXIS_FOLD_RE = re.compile(r"up-panel\s+(?:geomean\s+)?([0-9.]+)\s*[x×]")
_AXIS_DOWN_FOLD_RE = re.compile(r"down-panel\s+(?:geomean\s+)?([0-9.]+)\s*[x×]")
_FIRST_SENTENCE_RE = re.compile(r"^\s*([^.]+?)\s*(?:\.|$)")


def parse_response_axes(evidence_lines: Iterable[str]) -> dict[str, ResponseAxisState]:
    """Parse the ``### Therapy-state context`` section of evidence.md.

    The trufflepig renderer emits each axis as either a single line::

        **MAPK EGFR signaling** — active. Active signaling: up-panel geomean 3.52x cohort

    or as ``**Axis** — state.`` on one line followed by the fold sentence
    on the next. We extract the state from the first sentence after the
    em-dash and scan the whole axis section for ``up-panel`` and
    ``down-panel`` folds.
    """
    lines = list(evidence_lines)
    out: dict[str, ResponseAxisState] = {}

    in_section = False
    cur_axis: str | None = None
    cur_state: str | None = None
    cur_up: float | None = None
    cur_down: float | None = None
    cur_note: str = ""

    def _flush():
        nonlocal cur_axis, cur_state, cur_up, cur_down, cur_note
        if cur_axis and cur_state is not None:
            out[cur_axis] = ResponseAxisState(
                axis=cur_axis,
                state=cur_state.strip().lower(),
                up_fold=cur_up,
                down_fold=cur_down,
                note=cur_note.strip(),
            )
        cur_axis = cur_state = None
        cur_up = cur_down = None
        cur_note = ""

    def _absorb_fold_text(text: str):
        nonlocal cur_up, cur_down
        m_up = _AXIS_FOLD_RE.search(text)
        if m_up and cur_up is None:
            cur_up = float(m_up.group(1))
        m_down = _AXIS_DOWN_FOLD_RE.search(text)
        if m_down and cur_down is None:
            cur_down = float(m_down.group(1))

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("### "):
            if in_section:
                _flush()
                in_section = False
            if "therapy-state context" in stripped.lower():
                in_section = True
            continue
        if not in_section:
            continue
        m = _AXIS_HEADER_RE.match(stripped)
        if m:
            _flush()
            cur_axis = _norm_axis(m.group(1))
            tail = m.group(2).strip()
            sentence = _FIRST_SENTENCE_RE.match(tail)
            cur_state = (sentence.group(1) if sentence else tail).strip()
            cur_note = tail
            _absorb_fold_text(tail)
            continue
        if cur_axis is None:
            continue
        if stripped.startswith("|") or stripped.startswith("Gene"):
            continue
        if not stripped:
            continue
        cur_note = (cur_note + " " + stripped).strip()
        _absorb_fold_text(stripped)
    if in_section:
        _flush()
    return out


_TARGET_LINE_RE = re.compile(
    r"^\*\*([A-Z0-9][A-Z0-9_/-]*)\*\*\s+—\s+(.+?)\s+\(([^)]+)\)\."
)
_TARGET_TPM_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s+tumor-source bulk TPM", re.I)


def parse_target_shortlist(summary_lines: Iterable[str]) -> list[TargetShortlistEntry]:
    """Parse the ``## Top candidate therapies`` section of summary.md."""
    out: list[TargetShortlistEntry] = []
    lines = list(summary_lines)
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "## Top candidate therapies":
            start = i + 1
            break
    if start is None:
        return out
    for line in lines[start:]:
        if line.startswith("## "):
            break
        stripped = line.strip()
        if not stripped.startswith("- **"):
            continue
        body = stripped[2:].strip()  # drop leading '- '
        m = _TARGET_LINE_RE.match(body)
        if not m:
            continue
        gene = m.group(1).strip()
        drug = m.group(2).strip()
        scope = m.group(3).strip()
        m_tpm = _TARGET_TPM_RE.search(body)
        tpm = float(m_tpm.group(1)) if m_tpm else None
        # Tier is the first comma-separated token of the scope (e.g. "Approved").
        tier = scope.split(",", 1)[0].strip()
        indication = scope.split(",", 1)[1].strip() if "," in scope else ""
        out.append(
            TargetShortlistEntry(
                gene=gene,
                drug=drug,
                indication=indication,
                tpm=tpm,
                tier=tier,
                raw=body,
            )
        )
    return out


# ---------- Cross-sample observations ----------


@dataclass(frozen=True)
class LongitudinalDelta:
    """One typed observation between two sequential samples.

    ``kind`` identifies the comparison axis; ``before`` / ``after`` are
    JSON-safe snapshots. ``magnitude`` is a unitless effect size where
    meaningful; ``direction`` is one of ``up``, ``down``, ``unchanged``,
    ``gained``, ``lost``, ``shifted``, ``new``, ``cleared``.
    """

    kind: str
    before_sample: str
    after_sample: str
    direction: str
    magnitude: float | None = None
    before: Any = None
    after: Any = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LongitudinalDeltaSet:
    before_sample: str
    after_sample: str
    deltas: list[LongitudinalDelta] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_sample": self.before_sample,
            "after_sample": self.after_sample,
            "deltas": [d.to_dict() for d in self.deltas],
        }


# Library-prep / preservation pairs that materially change TPM comparability.
_LIBRARY_BUCKETS = {
    "ribosomal depletion": "ribo",
    "rrna depletion": "ribo",
    "total rna": "ribo",
    "polya": "polya",
    "poly-a": "polya",
    "exon capture": "capture",
    "exome capture": "capture",
}


def _bucket_library(text: str) -> str:
    """Reduce a sample-context string to a comparability bucket."""
    if not text:
        return "unknown"
    text = text.lower()
    for needle, bucket in _LIBRARY_BUCKETS.items():
        if needle in text:
            return bucket
    return "other"


def _preservation_bucket(text: str) -> str:
    if not text:
        return "unknown"
    text = text.lower()
    if "ffpe" in text:
        return "ffpe"
    if "fresh" in text or "frozen" in text:
        return "fresh"
    return "unknown"


def _direction_from_signed(value: float | None, threshold: float = 0.1) -> str:
    if value is None:
        return "unknown"
    if value > threshold:
        return "up"
    if value < -threshold:
        return "down"
    return "unchanged"


def _safe_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def _safe_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a == 0:
        return None
    return b / a


def compute_pairwise_deltas(before, after) -> LongitudinalDeltaSet:
    """Compute typed deltas between two sequential ``AnalyzeSummaryRecord``s.

    The records expose response_axes and target_shortlist via the
    extended summary loader; older records fall back to the basic
    cancer-call / purity / stroma / immune comparison.
    """
    deltas: list[LongitudinalDelta] = []

    before_call = (before.cancer_call or "").split(" ", 1)[0]
    after_call = (after.cancer_call or "").split(" ", 1)[0]
    if before_call and after_call and before_call != after_call:
        deltas.append(
            LongitudinalDelta(
                kind="cancer_call",
                before_sample=before.sample_id,
                after_sample=after.sample_id,
                direction="shifted",
                before=before.cancer_call,
                after=after.cancer_call,
                note=(
                    "Cancer-call differs across samples. Confirm whether "
                    "this is true progression/transformation or an "
                    "RNA-context shift driven by purity/site/assay."
                ),
            )
        )

    purity_delta = _safe_diff(before.purity_pct, after.purity_pct)
    if purity_delta is not None:
        deltas.append(
            LongitudinalDelta(
                kind="purity",
                before_sample=before.sample_id,
                after_sample=after.sample_id,
                direction=_direction_from_signed(purity_delta, threshold=5.0),
                magnitude=purity_delta,
                before=before.purity_pct,
                after=after.purity_pct,
                note="Absolute purity-percentage delta.",
            )
        )

    stromal_log = None
    if before.stromal_fold and after.stromal_fold:
        stromal_log = (after.stromal_fold / before.stromal_fold) - 1.0
        deltas.append(
            LongitudinalDelta(
                kind="stroma",
                before_sample=before.sample_id,
                after_sample=after.sample_id,
                direction=_direction_from_signed(stromal_log, threshold=0.2),
                magnitude=stromal_log,
                before=before.stromal_fold,
                after=after.stromal_fold,
                note="Stromal-enrichment fold change relative to cohort.",
            )
        )

    immune_log = None
    if before.immune_fold and after.immune_fold:
        immune_log = (after.immune_fold / before.immune_fold) - 1.0
        deltas.append(
            LongitudinalDelta(
                kind="immune",
                before_sample=before.sample_id,
                after_sample=after.sample_id,
                direction=_direction_from_signed(immune_log, threshold=0.2),
                magnitude=immune_log,
                before=before.immune_fold,
                after=after.immune_fold,
                note="Immune-enrichment fold change relative to cohort.",
            )
        )

    hla_delta = _safe_diff(before.hla_mean_tpm, after.hla_mean_tpm)
    if hla_delta is not None:
        deltas.append(
            LongitudinalDelta(
                kind="mhc_i",
                before_sample=before.sample_id,
                after_sample=after.sample_id,
                direction=_direction_from_signed(hla_delta, threshold=50),
                magnitude=hla_delta,
                before=before.hla_mean_tpm,
                after=after.hla_mean_tpm,
                note="Mean HLA-A/B/C TPM delta — MHC-I antigen-presentation context.",
            )
        )

    # Response-axis deltas (MAPK/EMT/hypoxia/IFN/...).
    before_axes = getattr(before, "response_axes", {}) or {}
    after_axes = getattr(after, "response_axes", {}) or {}
    for axis in sorted(set(before_axes) | set(after_axes)):
        b = before_axes.get(axis)
        a = after_axes.get(axis)
        if b is None and a is None:
            continue
        if b is None:
            deltas.append(
                LongitudinalDelta(
                    kind="response_axis",
                    before_sample=before.sample_id,
                    after_sample=after.sample_id,
                    direction="new",
                    before=None,
                    after=asdict(a),
                    note=f"Axis {axis} only emerged in {after.sample_id}.",
                )
            )
            continue
        if a is None:
            deltas.append(
                LongitudinalDelta(
                    kind="response_axis",
                    before_sample=before.sample_id,
                    after_sample=after.sample_id,
                    direction="cleared",
                    before=asdict(b),
                    after=None,
                    note=f"Axis {axis} present in {before.sample_id} but not in {after.sample_id}.",
                )
            )
            continue
        magnitude = _safe_ratio(b.up_fold, a.up_fold)
        if magnitude is not None:
            log_delta = magnitude - 1.0
        else:
            log_delta = None
        if b.state != a.state:
            direction = "shifted"
        elif log_delta is not None:
            direction = _direction_from_signed(log_delta, threshold=0.3)
        else:
            direction = "unchanged"
        deltas.append(
            LongitudinalDelta(
                kind="response_axis",
                before_sample=before.sample_id,
                after_sample=after.sample_id,
                direction=direction,
                magnitude=log_delta,
                before=asdict(b),
                after=asdict(a),
                note=f"Axis {axis}: {b.state} ({b.up_fold}x) -> {a.state} ({a.up_fold}x).",
            )
        )

    # Target shortlist gains / losses.
    before_targets = {t.gene: t for t in (before.target_shortlist or [])}
    after_targets = {t.gene: t for t in (after.target_shortlist or [])}
    for gene in sorted(set(after_targets) - set(before_targets)):
        t = after_targets[gene]
        deltas.append(
            LongitudinalDelta(
                kind="target",
                before_sample=before.sample_id,
                after_sample=after.sample_id,
                direction="gained",
                magnitude=t.tpm,
                before=None,
                after=asdict(t),
                note=f"{gene} entered the top-target shortlist (TPM={t.tpm}).",
            )
        )
    for gene in sorted(set(before_targets) - set(after_targets)):
        t = before_targets[gene]
        deltas.append(
            LongitudinalDelta(
                kind="target",
                before_sample=before.sample_id,
                after_sample=after.sample_id,
                direction="lost",
                magnitude=t.tpm,
                before=asdict(t),
                after=None,
                note=f"{gene} dropped from the top-target shortlist (was TPM={t.tpm}).",
            )
        )
    for gene in sorted(set(before_targets) & set(after_targets)):
        bt = before_targets[gene]
        at = after_targets[gene]
        tpm_delta = _safe_diff(bt.tpm, at.tpm)
        if tpm_delta is None:
            continue
        relative = _safe_ratio(bt.tpm, at.tpm)
        rel_delta = (relative - 1.0) if relative is not None else None
        if relative is not None and abs(rel_delta or 0) < 0.2:
            continue
        deltas.append(
            LongitudinalDelta(
                kind="target_tpm",
                before_sample=before.sample_id,
                after_sample=after.sample_id,
                direction=_direction_from_signed(rel_delta, threshold=0.2),
                magnitude=rel_delta,
                before={"gene": gene, "tpm": bt.tpm, "tier": bt.tier},
                after={"gene": gene, "tpm": at.tpm, "tier": at.tier},
                note=f"{gene} target TPM moved {bt.tpm} -> {at.tpm}.",
            )
        )

    # Assay-comparability advisory.
    before_lib = _bucket_library(before.sample_context)
    after_lib = _bucket_library(after.sample_context)
    before_preserve = _preservation_bucket(before.sample_context)
    after_preserve = _preservation_bucket(after.sample_context)
    if before_lib != after_lib or before_preserve != after_preserve:
        deltas.append(
            LongitudinalDelta(
                kind="assay_comparability",
                before_sample=before.sample_id,
                after_sample=after.sample_id,
                direction="shifted",
                before={"library": before_lib, "preservation": before_preserve},
                after={"library": after_lib, "preservation": after_preserve},
                note=(
                    "Library prep and/or preservation differ across samples; "
                    "interpret absolute TPM deltas via source-attributed and "
                    "context TPM fields rather than raw values."
                ),
            )
        )

    return LongitudinalDeltaSet(
        before_sample=before.sample_id,
        after_sample=after.sample_id,
        deltas=deltas,
    )


def write_deltas_json(path: Path, sets: list[LongitudinalDeltaSet]) -> None:
    import json

    path.write_text(
        json.dumps([s.to_dict() for s in sets], indent=2, default=str)
    )


# ---------- Markdown sections for the comparison report ----------


def _direction_glyph(direction: str) -> str:
    return {
        "up": "↑",
        "down": "↓",
        "unchanged": "·",
        "gained": "+",
        "lost": "−",
        "shifted": "→",
        "new": "✦",
        "cleared": "○",
        "unknown": "?",
    }.get(direction, direction)


def response_axis_table(records) -> list[str]:
    """Markdown table of per-sample response-axis states.

    One column per sample, one row per axis seen across the cohort. The
    first column carries the canonical axis id; cells render as
    ``state (Xx)`` or blank when the axis is absent from that sample.
    """
    axes = sorted({a for rec in records for a in (rec.response_axes or {}).keys()})
    if not axes:
        return []
    header = "| Axis | " + " | ".join(rec.sample_id for rec in records) + " |"
    sep = "|---|" + "---|" * len(records)
    rows = [header, sep]
    for axis in axes:
        cells = [axis]
        for rec in records:
            state = (rec.response_axes or {}).get(axis)
            if state is None:
                cells.append("—")
                continue
            fold = f" ({state.up_fold:.2f}x)" if state.up_fold is not None else ""
            cells.append(f"{state.state}{fold}")
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def delta_summary_table(delta_sets: list[LongitudinalDeltaSet]) -> list[str]:
    """One row per delta across all pairwise comparisons."""
    header = "| Transition | Kind | Direction | Magnitude | Detail |"
    sep = "|---|---|---|---:|---|"
    rows = [header, sep]
    for ds in delta_sets:
        for d in ds.deltas:
            mag = f"{d.magnitude:+.2f}" if isinstance(d.magnitude, (int, float)) else ""
            note = (d.note or "").replace("|", "/").replace("\n", " ")
            rows.append(
                f"| {ds.before_sample} → {ds.after_sample} "
                f"| {d.kind} "
                f"| {_direction_glyph(d.direction)} {d.direction} "
                f"| {mag} "
                f"| {note} |"
            )
    return rows
