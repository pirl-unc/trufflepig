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
    """One row of the per-axis pathway and treatment-state signals.

    Mirrors what trufflepig's analyze() prints to the evidence.md
    "Pathway and Treatment-State Signals" section, e.g.::

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
_FIRST_SENTENCE_RE = re.compile(r"^\s*([^.;]+?)\s*(?:[.;]|$)")

# Controlled vocabulary for ResponseAxisState.state. The renderer
# emits state text in free form ("active", "Active signaling", "mixed
# signal; partial agreement", "suppressed", ...); we normalize to one
# of these tokens so cross-sample equality is stable. Anything that
# doesn't match falls through as the raw lowercased prefix.
_STATE_KEYWORDS = (
    ("suppressed", "suppressed"),
    ("active", "active"),
    ("elevated", "active"),
    ("mixed", "mixed"),
    ("partial", "mixed"),
    ("inconclusive", "inconclusive"),
    ("indeterminate", "inconclusive"),
    ("unknown", "inconclusive"),
)


def _normalize_axis_state(text: str | None) -> str:
    """Map a free-form axis-state phrase to a controlled token.

    Returns one of ``active``, ``suppressed``, ``mixed``, ``inconclusive``,
    or the first lowercased word of ``text`` when no keyword matches —
    so unanticipated renderer output is preserved but downstream
    equality checks aren't tripped by punctuation/qualifier drift.
    """
    if not text:
        return ""
    lowered = text.strip().lower()
    for needle, token in _STATE_KEYWORDS:
        if needle in lowered:
            return token
    # Fall back to the first word for anything unexpected.
    head = re.split(r"[\s.;]+", lowered, maxsplit=1)[0]
    return head or ""


def parse_response_axes(evidence_lines: Iterable[str]) -> dict[str, ResponseAxisState]:
    """Parse the ``### Pathway and Treatment-State Signals`` section of evidence.md.

    Older reports used ``Therapy Response Signals``, ``Therapy-State Evidence``,
    and ``Therapy-state context``; keep accepting those headings so longitudinal
    comparison can read saved workspaces.

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
                state=_normalize_axis_state(cur_state),
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
        # Any new heading (h1/h2/h3/h4/...) or horizontal rule ends the
        # therapy-state section, except when the heading itself names
        # the section we want to enter.
        if stripped.startswith("#") or stripped.startswith("---"):
            if in_section:
                _flush()
                in_section = False
            heading = stripped.lower()
            if stripped.startswith("#") and (
                "pathway and treatment-state signals" in heading
                or "therapy response signals" in heading
                or "therapy-state evidence" in heading
                or "therapy-state context" in heading
            ):
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
    r"^\*\*([A-Z0-9][A-Z0-9_/-]*)\*\*\s+[—–-]\s+(.+?)\s+"
    # scope group: allow one level of nested parens (e.g. "Approved (subset)")
    r"\(((?:[^()]|\([^()]*\))+)\)\s*\."
)
_TARGET_TPM_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s+(?:estimated\s+tumor\s+TPM|"
    r"patient\s+tumor-attributed\s+TPM|"
    r"tumor-source\s+bulk\s+TPM)",
    re.I,
)
# Cancer-call prefixes: leading TCGA-style code (≥2 chars of upper-case
# letters, digits, underscores, hyphens). The all-caps requirement
# rejects "Cancer-type unclear" → "" (would otherwise greedy-match just
# the leading "C") and accepts "BLCA", "SARC_DDLPS", "PFO017"-style codes.
_CANCER_CALL_PREFIX_RE = re.compile(r"^([A-Z][A-Z0-9_-]+)(?=\s|$|[—–-])")


def _cancer_call_prefix(call: str | None) -> str:
    if not call:
        return ""
    m = _CANCER_CALL_PREFIX_RE.match(call.strip())
    return m.group(1) if m else ""


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

    ``kind`` is the comparison axis (``cancer_call``, ``purity``,
    ``stroma``, ``immune``, ``mhc_i``, ``response_axis``, ``target``,
    ``target_tpm``, ``assay_comparability``). ``before`` / ``after`` are
    JSON-safe snapshots whose schema depends on ``kind``:

    =================== =====================================================
    kind                ``before`` / ``after`` schema
    =================== =====================================================
    cancer_call         string (e.g. ``"BLCA — moderate confidence"``)
    purity              float (percentage)
    stroma, immune      float (cohort-relative fold)
    mhc_i               float (TPM)
    response_axis       ``{axis, state, up_fold, down_fold, note}``
    target              ``{gene, drug, indication, tpm, tier, raw}``
    target_tpm          ``{gene, tpm, tier}``
    assay_comparability ``{library, preservation}``
    =================== =====================================================

    ``magnitude`` is a unitless effect size where meaningful: absolute
    delta for ``purity`` / ``mhc_i``, log-relative ``(after/before)-1``
    for fold metrics and TPMs. ``direction`` is one of ``up``, ``down``,
    ``unchanged``, ``gained``, ``lost``, ``shifted``, ``new``, ``cleared``,
    or ``unknown``. ``unit`` describes the magnitude scale so downstream
    tooling doesn't have to look it up from ``kind``.
    """

    kind: str
    before_sample: str
    after_sample: str
    direction: str
    magnitude: float | None = None
    unit: str = ""
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


def _is_finite(x: object) -> bool:
    """True only for a finite numeric value — rejects NaN, +inf, -inf."""
    import math

    try:
        return math.isfinite(float(x))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _safe_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if not (_is_finite(a) and _is_finite(b)):
        return None
    return b - a


def _safe_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a == 0:
        return None
    if not (_is_finite(a) and _is_finite(b)):
        return None
    return b / a


def compute_pairwise_deltas(before, after) -> LongitudinalDeltaSet:
    """Compute typed deltas between two sequential ``AnalyzeSummaryRecord``s.

    The records expose response_axes and target_shortlist via the
    extended summary loader; older records fall back to the basic
    cancer-call / purity / stroma / immune comparison.
    """
    deltas: list[LongitudinalDelta] = []

    before_call = _cancer_call_prefix(before.cancer_call)
    after_call = _cancer_call_prefix(after.cancer_call)
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
                unit="percentage_points",
                before=before.purity_pct,
                after=after.purity_pct,
                note="Absolute purity-percentage delta.",
            )
        )

    stromal_ratio = _safe_ratio(before.stromal_fold, after.stromal_fold)
    if stromal_ratio is not None:
        stromal_log = stromal_ratio - 1.0
        deltas.append(
            LongitudinalDelta(
                kind="stroma",
                before_sample=before.sample_id,
                after_sample=after.sample_id,
                direction=_direction_from_signed(stromal_log, threshold=0.2),
                magnitude=stromal_log,
                unit="log_relative",
                before=before.stromal_fold,
                after=after.stromal_fold,
                note="Stromal-enrichment fold change relative to cohort.",
            )
        )
    elif (before.stromal_fold == 0 and after.stromal_fold) or (
        after.stromal_fold == 0 and before.stromal_fold
    ):
        deltas.append(
            LongitudinalDelta(
                kind="stroma",
                before_sample=before.sample_id,
                after_sample=after.sample_id,
                direction="new" if before.stromal_fold == 0 else "cleared",
                before=before.stromal_fold,
                after=after.stromal_fold,
                note="Stromal signal appeared/cleared (zero baseline).",
            )
        )

    immune_ratio = _safe_ratio(before.immune_fold, after.immune_fold)
    if immune_ratio is not None:
        immune_log = immune_ratio - 1.0
        deltas.append(
            LongitudinalDelta(
                kind="immune",
                before_sample=before.sample_id,
                after_sample=after.sample_id,
                direction=_direction_from_signed(immune_log, threshold=0.2),
                magnitude=immune_log,
                unit="log_relative",
                before=before.immune_fold,
                after=after.immune_fold,
                note="Immune-enrichment fold change relative to cohort.",
            )
        )
    elif (before.immune_fold == 0 and after.immune_fold) or (
        after.immune_fold == 0 and before.immune_fold
    ):
        deltas.append(
            LongitudinalDelta(
                kind="immune",
                before_sample=before.sample_id,
                after_sample=after.sample_id,
                direction="new" if before.immune_fold == 0 else "cleared",
                before=before.immune_fold,
                after=after.immune_fold,
                note="Immune signal appeared/cleared (zero baseline).",
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
                unit="tpm",
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
        # State is normalized to a controlled vocabulary by parse_response_axes,
        # so equality is meaningful. When the state changes, suppress the
        # magnitude — up_fold semantics depend on the activation direction,
        # so a "magnitude" comparing active→suppressed is misleading (the
        # axis went down clinically, even if up_fold dropped numerically).
        magnitude = _safe_ratio(b.up_fold, a.up_fold)
        log_delta = (magnitude - 1.0) if magnitude is not None else None
        if b.state != a.state:
            direction = "shifted"
            log_delta = None  # suppress magnitude — semantics depend on state
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
                unit="log_relative" if log_delta is not None else "",
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
                unit="tpm" if t.tpm is not None else "",
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
                unit="tpm" if t.tpm is not None else "",
                before=asdict(t),
                after=None,
                note=f"{gene} dropped from the top-target shortlist (was TPM={t.tpm}).",
            )
        )
    # Target TPM movement: gate on both an absolute floor (max(before,after)
    # must be at least _TARGET_TPM_FLOOR so noise near the detection limit
    # doesn't masquerade as biology) and a 50% relative threshold, matching
    # the analyze-side fold-change conventions.
    _TARGET_TPM_FLOOR = 5.0
    _TARGET_TPM_REL_THRESHOLD = 0.5
    for gene in sorted(set(before_targets) & set(after_targets)):
        bt = before_targets[gene]
        at = after_targets[gene]
        if bt.tpm is None or at.tpm is None:
            continue
        if max(bt.tpm, at.tpm) < _TARGET_TPM_FLOOR:
            continue
        relative = _safe_ratio(bt.tpm, at.tpm)
        rel_delta = (relative - 1.0) if relative is not None else None
        if rel_delta is None or abs(rel_delta) < _TARGET_TPM_REL_THRESHOLD:
            continue
        deltas.append(
            LongitudinalDelta(
                kind="target_tpm",
                before_sample=before.sample_id,
                after_sample=after.sample_id,
                direction=_direction_from_signed(rel_delta, threshold=_TARGET_TPM_REL_THRESHOLD),
                magnitude=rel_delta,
                unit="log_relative",
                before={"gene": gene, "tpm": bt.tpm, "tier": bt.tier},
                after={"gene": gene, "tpm": at.tpm, "tier": at.tier},
                note=f"{gene} target TPM moved {bt.tpm} -> {at.tpm}.",
            )
        )

    # Assay-comparability advisory. Only fire when both buckets are known
    # AND they differ — emitting "shifted" because one side is "unknown"
    # would be misleading (the side just didn't report a library prep).
    before_lib = _bucket_library(before.sample_context)
    after_lib = _bucket_library(after.sample_context)
    before_preserve = _preservation_bucket(before.sample_context)
    after_preserve = _preservation_bucket(after.sample_context)
    _UNKNOWN = {"unknown", "other"}
    lib_known_and_differs = (
        before_lib not in _UNKNOWN
        and after_lib not in _UNKNOWN
        and before_lib != after_lib
    )
    preserve_known_and_differs = (
        before_preserve not in _UNKNOWN
        and after_preserve not in _UNKNOWN
        and before_preserve != after_preserve
    )
    # Asymmetric unknown — one side has the info, the other doesn't.
    # That's a comparability gap (the unknown side could be anything,
    # so absolute TPM deltas should be widened by an assay-uncertainty
    # allowance). Symmetric unknown (both sides lack info) emits nothing
    # — the two samples are in the same boat.
    lib_asymmetric_unknown = (
        (before_lib in _UNKNOWN) != (after_lib in _UNKNOWN)
    )
    preserve_asymmetric_unknown = (
        (before_preserve in _UNKNOWN) != (after_preserve in _UNKNOWN)
    )
    if lib_known_and_differs or preserve_known_and_differs:
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
    elif lib_asymmetric_unknown or preserve_asymmetric_unknown:
        deltas.append(
            LongitudinalDelta(
                kind="assay_comparability",
                before_sample=before.sample_id,
                after_sample=after.sample_id,
                direction="unknown",
                before={"library": before_lib, "preservation": before_preserve},
                after={"library": after_lib, "preservation": after_preserve},
                note=(
                    "Library prep or preservation is missing for one of the "
                    "samples — direct TPM deltas should be widened by an "
                    "assay-uncertainty allowance."
                ),
            )
        )

    return LongitudinalDeltaSet(
        before_sample=before.sample_id,
        after_sample=after.sample_id,
        deltas=deltas,
    )


DELTAS_JSON_SCHEMA_VERSION = 1


def write_deltas_json(path: Path, sets: list[LongitudinalDeltaSet]) -> None:
    """Atomically write ``deltas.json`` with a versioned envelope.

    The envelope shape ``{"schema_version": N, "deltas": [...]}``
    gives consumers a forward-compat gate. We deliberately reject NaN
    (``allow_nan=False``) so downstream JSON-strict parsers don't break
    on stray NaNs from upstream parsing — :func:`_safe_ratio` and
    :func:`_safe_diff` already filter non-finite inputs, this is the
    backstop. ``default=str`` is omitted on purpose: any unexpected
    object type should raise loudly rather than silently stringify.
    """
    import json

    payload = {
        "schema_version": DELTAS_JSON_SCHEMA_VERSION,
        "deltas": [s.to_dict() for s in sets],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    tmp.replace(path)


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
