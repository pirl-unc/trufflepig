# Licensed under the Apache License, Version 2.0

"""Shared feature frame for RNA-seq cancer-type calls.

The ranker used to mix several kinds of evidence in place: tumor signatures,
candidate-specific purity, normal-tissue context, subtype markers, and centroid
corroboration all lived as local adjustments in the caller. This module keeps
the per-sample facts and per-candidate identity checks together so ranking,
subtype resolution, and reporting can describe the same evidence.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Mapping, Sequence


# These helpers pull best-effort co-signals from bundled reference/ontology data. The two
# failure modes are handled DIFFERENTLY rather than caught together:
#   * a failed *import* of an internal module is a structural bug — it is NOT caught, so it
#     surfaces at the call site instead of silently degrading to "no evidence";
#   * *missing/degenerate reference data* (an absent bundled matrix, a degenerate sample) is a
#     legitimate runtime condition — only the data ACCESS is wrapped, in the narrow set of
#     errors those loads/compute steps actually raise, and degrades to empty.
# Imports stay function-local ONLY to break the cancer_call <-> tumor_purity import cycle
# (tumor_purity.py lazy-imports this module in turn), never to be swallowed.
_MISSING_REFERENCE_DATA_ERRORS = (FileNotFoundError, OSError, KeyError, ValueError, TypeError)


_NORMAL_ORIGIN_MIN_HOST_SCORE = 0.86
_NORMAL_ORIGIN_MAX_PROTECTIVE_IDENTITY = 0.55
_NORMAL_ORIGIN_MIN_ALTERNATE_SIGNATURE = 0.35
_NORMAL_ORIGIN_MAX_SIGNATURE_RATIO = 1.35
_NORMAL_ORIGIN_MAX_DEMOTION = 0.55
_DISTINCTIVE_REFERENCE_MIN_TPM = 5.0
_DISTINCTIVE_REFERENCE_FOLD = 1.8


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _sample_dict(sample_tpm_by_symbol: Mapping[str, Any] | None) -> dict[str, float]:
    return {
        str(symbol).upper(): float(value or 0.0)
        for symbol, value in (sample_tpm_by_symbol or {}).items()
        if str(symbol).strip()
    }


def _status_marker_support(status: str, high_fraction: float) -> float:
    status = _clean(status)
    high_fraction = max(0.0, min(1.0, float(high_fraction or 0.0)))
    if status == "consistent":
        return max(0.80, high_fraction)
    if status == "partial":
        return max(0.35, min(0.75, high_fraction))
    if status == "mixed":
        return min(0.55, max(0.35, high_fraction))
    if status == "not_evaluable":
        return 0.20
    return 0.0


def _tumor_up_support(hit_count: int) -> float:
    if hit_count >= 3:
        return 1.0
    if hit_count == 2:
        return 0.70
    if hit_count == 1:
        return 0.35
    return 0.0


@lru_cache(maxsize=512)
def _tumor_up_panel_symbols(cancer_code: str) -> tuple[str, ...]:
    code = _clean(cancer_code)
    if not code:
        return ()
    from .reference import heme_tumor_up_vs_matched_normal, tumor_up_vs_matched_normal

    try:
        panel = tumor_up_vs_matched_normal(cancer_code=code)
        if panel is None or panel.empty:
            panel = heme_tumor_up_vs_matched_normal(cancer_code=code)
    except _MISSING_REFERENCE_DATA_ERRORS:
        return ()
    if panel is None or panel.empty or "symbol" not in panel.columns:
        return ()
    return tuple(
        str(symbol).upper()
        for symbol in panel["symbol"].astype(str)
        if str(symbol).strip()
    )


def _candidate_reference_codes(code: str) -> tuple[str, ...]:
    code = _clean(code)
    if not code:
        return ()
    out: list[str] = [code]
    from .tumor_type_ontology import tumor_type_ontology_entry

    entry = tumor_type_ontology_entry(code)  # total: returns None for an unknown code
    if entry is not None:
        for candidate in (entry.parent_code, *entry.ancestors):
            candidate = _clean(candidate)
            if candidate and candidate not in out:
                out.append(candidate)
    return tuple(out)


def _candidate_primary_tissue(code: str) -> str:
    # Function-local import only to break the cancer_call <-> tumor_purity cycle; tumor_purity is
    # always importable by the time this runs, so an import failure here is a real bug, not
    # something to swallow.
    from .tumor_purity import CANCER_TO_TISSUE

    for candidate in _candidate_reference_codes(code):
        tissue = _clean(CANCER_TO_TISSUE.get(candidate))
        if tissue:
            return tissue
    return ""


@lru_cache(maxsize=1)
def _pan_reference_by_symbol():
    from .reference import pan_cancer_expression

    try:
        ref = pan_cancer_expression(technical_rna_normalize=True)
    except _MISSING_REFERENCE_DATA_ERRORS:
        return None
    return ref.drop_duplicates(subset="Symbol").set_index("Symbol")


@lru_cache(maxsize=8192)
def _reference_gene_tpm(code: str, symbol: str) -> float:
    code = _clean(code)
    symbol = str(symbol or "").upper()
    if not code or not symbol:
        return 0.0
    by_symbol = _pan_reference_by_symbol()
    if by_symbol is None:
        return 0.0
    col = f"{code}_TPM"
    if col not in by_symbol.columns or symbol not in by_symbol.index:
        return 0.0
    # A missing reference cell reads back as NaN, and `float(nan or 0.0)` is NaN
    # (bool(nan) is True), which would then pass the distinctiveness fold-change
    # guards as a fake "distinctive tumor marker". Map any non-finite cell to 0.0.
    try:
        value = float(by_symbol.loc[symbol, col])
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _max_reference_gene_tpm(code: str, symbol: str) -> float:
    # Each _reference_gene_tpm is finite (NaN mapped to 0.0 above), so a plain
    # max is safe here — no NaN-ordering hazard.
    values = [_reference_gene_tpm(candidate, symbol) for candidate in _candidate_reference_codes(code)]
    return max(values, default=0.0)


def _normal_tissue_match(a: str, b: str) -> bool:
    return _clean(a) == _clean(b)


@dataclass(frozen=True)
class CandidateIdentityEvidence:
    """Interpretable identity evidence for one candidate label."""

    code: str
    primary_tissue: str = ""
    dominant_normal_tissue: str = ""
    dominant_normal_score: float = 0.0
    primary_tissue_score: float = 0.0
    tumor_up_hit_count: int = 0
    tumor_up_hits: tuple[tuple[str, float], ...] = ()
    distinctive_tumor_up_hit_count: int = 0
    distinctive_tumor_up_hits: tuple[tuple[str, float], ...] = ()
    marker_status: str = ""
    expected_high_detected_fraction: float = 0.0
    expected_low_present_count: int = 0
    tumor_identity_support: float = 0.0
    normal_origin_dominance: float = 0.0
    normal_origin_applied: bool = False
    identity_factor: float = 1.0
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tumor_up_hits"] = [
            {"gene": gene, "tpm": round(float(tpm), 3)}
            for gene, tpm in self.tumor_up_hits
        ]
        data["distinctive_tumor_up_hits"] = [
            {"gene": gene, "tpm": round(float(tpm), 3)}
            for gene, tpm in self.distinctive_tumor_up_hits
        ]
        return data


@dataclass
class CancerCallFeatureFrame:
    """Sample-level RNA-seq features reused across candidate decisions."""

    sample_tpm_by_symbol: Mapping[str, float]
    host_tissue_details: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    dominant_normal_tissue: str = ""
    dominant_normal_score: float = 0.0

    @classmethod
    def from_sample(
        cls,
        sample_tpm_by_symbol: Mapping[str, Any] | None,
        *,
        host_tissue_details: Sequence[Mapping[str, Any]] | None = None,
    ) -> "CancerCallFeatureFrame":
        sample = _sample_dict(sample_tpm_by_symbol)
        details: Sequence[Mapping[str, Any]]
        if host_tissue_details is None:
            # Function-local import breaks the cancer_call <-> tumor_purity cycle; only the
            # scoring CALL (which can hit degenerate sample data) is guarded, not the import.
            from .tumor_purity import _score_host_tissue_details

            try:
                details = tuple(_score_host_tissue_details(sample, top_n=10))
            except _MISSING_REFERENCE_DATA_ERRORS:
                details = ()
        else:
            details = tuple(host_tissue_details)
        top = details[0] if details else {}
        return cls(
            sample_tpm_by_symbol=sample,
            host_tissue_details=details,
            dominant_normal_tissue=_clean(top.get("tissue")),
            dominant_normal_score=float(top.get("score") or 0.0),
        )

    def primary_tissue_score(self, tissue: str) -> float:
        tissue = _clean(tissue)
        if not tissue:
            return 0.0
        for row in self.host_tissue_details:
            if _normal_tissue_match(str(row.get("tissue") or ""), tissue):
                return float(row.get("score") or 0.0)
        return 0.0

    def tumor_up_hits(self, code: str, *, min_tpm: float = 3.0) -> tuple[tuple[str, float], ...]:
        hits: dict[str, float] = {}
        for reference_code in _candidate_reference_codes(code):
            for symbol in _tumor_up_panel_symbols(reference_code):
                value = float(self.sample_tpm_by_symbol.get(symbol, 0.0) or 0.0)
                if value >= min_tpm:
                    hits[symbol] = max(value, hits.get(symbol, 0.0))
            if hits:
                break
        return tuple(sorted(hits.items(), key=lambda item: (-item[1], item[0]))[:10])

    def distinctive_tumor_up_hits(
        self,
        code: str,
        hits: Sequence[tuple[str, float]],
        competing_codes: Sequence[str],
    ) -> tuple[tuple[str, float], ...]:
        distinctive: list[tuple[str, float]] = []
        competitors = tuple(_clean(other) for other in competing_codes if _clean(other))
        for symbol, value in hits:
            candidate_ref = _max_reference_gene_tpm(code, symbol)
            competitor_ref = max(
                (_max_reference_gene_tpm(other, symbol) for other in competitors),
                default=0.0,
            )
            if candidate_ref < _DISTINCTIVE_REFERENCE_MIN_TPM:
                continue
            if candidate_ref < _DISTINCTIVE_REFERENCE_FOLD * max(competitor_ref, 1.0):
                continue
            distinctive.append((symbol, value))
        return tuple(distinctive)

    def marker_sanity(self, code: str) -> dict[str, Any]:
        from .tumor_type_ontology import tumor_type_sanity_check

        try:
            return tumor_type_sanity_check(code, self.sample_tpm_by_symbol)
        except _MISSING_REFERENCE_DATA_ERRORS:
            return {}

    def evidence_for_candidate(
        self,
        code: str,
        *,
        best_alternate_signature: float = 0.0,
        candidate_signature: float = 0.0,
        competing_codes: Sequence[str] = (),
    ) -> CandidateIdentityEvidence:
        code = _clean(code)
        primary_tissue = _candidate_primary_tissue(code)
        primary_tissue_score = self.primary_tissue_score(primary_tissue)
        tumor_up_hits = self.tumor_up_hits(code)
        distinctive_hits = self.distinctive_tumor_up_hits(
            code,
            tumor_up_hits,
            competing_codes,
        )
        sanity = self.marker_sanity(code)
        marker_status = _clean(sanity.get("status"))
        high_fraction = float(sanity.get("expected_high_detected_fraction") or 0.0)
        low_present = sanity.get("expected_low_present") or ()
        marker_support = _status_marker_support(marker_status, high_fraction)
        reasons: list[str] = []
        identity_factor = 1.0
        normal_origin_dominance = 0.0
        applied = False
        same_as_dominant = (
            bool(primary_tissue)
            and _normal_tissue_match(primary_tissue, self.dominant_normal_tissue)
        )
        if (
            same_as_dominant
            and self.dominant_normal_score >= _NORMAL_ORIGIN_MIN_HOST_SCORE
        ):
            tumor_specific_support = max(
                _tumor_up_support(len(distinctive_hits)),
                0.25 * marker_support,
            )
        else:
            tumor_specific_support = max(
                _tumor_up_support(len(tumor_up_hits)),
                marker_support,
            )
        alternate_ok = (
            float(best_alternate_signature or 0.0)
            >= _NORMAL_ORIGIN_MIN_ALTERNATE_SIGNATURE
            and float(candidate_signature or 0.0)
            <= _NORMAL_ORIGIN_MAX_SIGNATURE_RATIO * float(best_alternate_signature or 0.0)
        )
        if (
            same_as_dominant
            and alternate_ok
            and self.dominant_normal_score >= _NORMAL_ORIGIN_MIN_HOST_SCORE
            and tumor_specific_support < _NORMAL_ORIGIN_MAX_PROTECTIVE_IDENTITY
        ):
            host_strength = (
                (self.dominant_normal_score - _NORMAL_ORIGIN_MIN_HOST_SCORE)
                / max(1e-6, 1.0 - _NORMAL_ORIGIN_MIN_HOST_SCORE)
            )
            identity_gap = (
                (_NORMAL_ORIGIN_MAX_PROTECTIVE_IDENTITY - tumor_specific_support)
                / _NORMAL_ORIGIN_MAX_PROTECTIVE_IDENTITY
            )
            normal_origin_dominance = max(0.0, min(1.0, host_strength * identity_gap))
            if normal_origin_dominance >= 0.10:
                identity_factor = max(
                    1.0 - _NORMAL_ORIGIN_MAX_DEMOTION,
                    1.0 - _NORMAL_ORIGIN_MAX_DEMOTION * normal_origin_dominance,
                )
                applied = identity_factor < 0.995
                if applied:
                    reasons.append(
                        "dominant normal tissue matches this candidate but tumor-specific markers do not corroborate it"
                    )

        return CandidateIdentityEvidence(
            code=code,
            primary_tissue=primary_tissue,
            dominant_normal_tissue=self.dominant_normal_tissue,
            dominant_normal_score=float(self.dominant_normal_score),
            primary_tissue_score=float(primary_tissue_score),
            tumor_up_hit_count=len(tumor_up_hits),
            tumor_up_hits=tumor_up_hits,
            distinctive_tumor_up_hit_count=len(distinctive_hits),
            distinctive_tumor_up_hits=distinctive_hits,
            marker_status=marker_status,
            expected_high_detected_fraction=high_fraction,
            expected_low_present_count=len(low_present),
            tumor_identity_support=float(tumor_specific_support),
            normal_origin_dominance=float(normal_origin_dominance),
            normal_origin_applied=bool(applied),
            identity_factor=float(identity_factor),
            reasons=tuple(reasons),
        )


def apply_staged_identity_evidence(
    rows: list[dict[str, Any]],
    frame: CancerCallFeatureFrame,
) -> list[dict[str, Any]]:
    """Attach staged identity evidence and apply normal-origin demotions."""

    if not rows:
        return rows
    primary_tissues = {
        str(row.get("code") or ""): _candidate_primary_tissue(str(row.get("code") or ""))
        for row in rows
    }
    for row in rows:
        code = str(row.get("code") or "")
        row_tissue = primary_tissues.get(code, "")
        best_alternate_signature = max(
            (
                float(other.get("signature_score") or 0.0)
                for other in rows
                if not _normal_tissue_match(
                    primary_tissues.get(str(other.get("code") or ""), ""),
                    row_tissue,
                )
            ),
            default=0.0,
        )
        evidence = frame.evidence_for_candidate(
            code,
            best_alternate_signature=best_alternate_signature,
            candidate_signature=float(row.get("signature_score") or 0.0),
            competing_codes=[
                str(other.get("code") or "")
                for other in rows
                if str(other.get("code") or "") != code
            ],
        )
        row["staged_identity_evidence"] = evidence.to_dict()
        row["staged_identity_factor"] = float(evidence.identity_factor)
        if evidence.normal_origin_applied:
            row.setdefault("pre_staged_identity_support_score", row.get("support_score"))
            row.setdefault(
                "pre_staged_identity_support_geomean",
                row.get("support_geomean"),
            )
            row["support_score"] = float(row.get("support_score") or 0.0) * evidence.identity_factor
            row["support_geomean"] = row["support_score"]
    rows.sort(
        key=lambda row: (
            -float(row.get("support_score") or 0.0),
            -float(row.get("signature_score") or 0.0),
            str(row.get("code") or ""),
        )
    )
    return rows
