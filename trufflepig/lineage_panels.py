# Licensed under the Apache License, Version 2.0

"""Lightweight lineage-discrimination panels.

Each panel evaluates a single hypothesis: "is this sample a member of
lineage X?" using a small, curated set of markers. The score is a
synthesis of two evidence streams kept separately because the
TPM-vs-HK experiment (see `docs/CANCER_CALL_DECISION_FLOW.md` known
gaps section) showed they have opposite optimal scales:

  - **High-direction markers** ("this gene should be ON") score better
    when normalized to housekeeping median, because HK normalization
    cancels library-depth differences across samples.
  - **Low-direction markers** ("this gene should be OFF") score better
    as absolute TPM thresholds, because the question is "is it
    essentially zero?" — HK ratios at near-zero are noise.

Evidence is preserved per marker, not collapsed prematurely. The
returned ``PanelEvidence`` is JSON-serializable so it flows into
``analysis-parameters.json`` and is consumable by downstream selectors,
the cancer_type_evidence consolidator, and report rendering.

Design constraints (deliberately small):
  - One ``LineagePanel`` dataclass (no per-marker dataclass).
  - One ``score_panel`` function (no nested helpers).
  - Cohort medians are looked up from
    ``cancer_reference_expression`` at runtime, not specified by hand.
  - Adding a new panel = 10-15 lines of data.

Wiring into the broader cancer-call pipeline is intentionally NOT
done in this file — the panels stand alone as an evidence layer that
``cancer_type_evidence.select_report_scope_from_evidence`` can opt
into via a new selector if it wants to. See the wiring contract
section at the bottom for the integration sketch.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Any, Mapping

import numpy as np


# ---------- core types ----------

@dataclass(frozen=True)
class LineagePanel:
    """A small curated panel evaluating membership in one lineage.

    All sample-level evaluation lives in ``score_panel`` — this is
    just data.
    """

    name: str
    parent_cohort: str
    """TCGA cohort code whose medians serve as the 'in range' reference
    for ``high_markers``. Looked up in cancer_reference_expression."""

    high_markers: tuple[str, ...]
    """Genes that should be ON in this lineage. Scored HK-normalized
    against the parent cohort's HK-normalized cohort median."""

    low_markers: tuple[tuple[str, float], ...] = ()
    """(symbol, max_tpm) — gene should be below the absolute TPM cap.
    Absolute TPM is more interpretable than HK ratios near zero."""

    obligate: tuple[str, ...] = ()
    """At least one high_marker per entry must clear 0.5× cohort HK ratio
    or the panel returns score=0 and the obligate-failed rationale."""

    description: str = ""
    references: tuple[str, ...] = ()

    program_note: str = ""
    """One short sentence describing the transcriptional program this
    panel captures and its biological / therapy context. Surfaced in
    report rendering as the "Subtype" follow-on line when the panel
    fires above the brief reporting bar. Leave empty when there's
    nothing useful to say — the report just omits the line."""


@dataclass(frozen=True)
class PanelEvidence:
    """All evidence for one panel against one sample.

    JSON-serializable via ``to_dict``. Score is one field; the rest is
    the supporting observations preserved for downstream reasoning,
    debugging, and report rendering.
    """

    panel_name: str
    parent_cohort: str
    obligate_passed: bool
    obligate_failures: tuple[tuple[str, float, float], ...]
    """(symbol, observed_hk_ratio, required_hk_ratio) for each failed
    obligate marker. Empty when obligate_passed."""

    high_hits: tuple[tuple[str, float, float], ...]
    """(symbol, observed_tpm, cohort_median_tpm) for high_markers that
    cleared the 0.5× cohort HK ratio threshold."""
    high_misses: tuple[tuple[str, float, float], ...]
    """Same shape as high_hits, for markers that did not clear."""

    low_passes: tuple[tuple[str, float, float], ...]
    """(symbol, observed_tpm, cap_tpm) for low_markers under the cap."""
    low_violations: tuple[tuple[str, float, float], ...]
    """Same shape as low_passes, for markers above the cap."""

    score: float
    """Synthesis: (high_fraction ** 0.6) × (low_fraction ** 0.4).
    Range [0, 1]. The exponents weight positive evidence more than
    negative compliance — a panel with all positives firing but one
    negative violation is still strong evidence, while a panel with
    only negative compliance and no positives is weak."""

    rationale: str
    """One human-readable sentence describing the panel verdict for
    reports. Mirrors the style of the existing basal-BRCA rescue text."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------- cohort lookup (cached) ----------
#
# Cohort-median tables are keyed by ``(cohort, gene_id)`` — internal
# lookups go through Ensembl IDs to match the rest of trufflepig's
# ID-first contract (see common.py / cancer_type_evidence.py for
# the consistent pattern). LineagePanel definitions remain
# symbol-friendly because curating by HGNC name is what biologists
# read; the symbol → ID resolution flows through the shared
# ``trufflepig.common.panel_symbols_to_gene_ids`` resolver so there's
# one Ensembl-lookup path in the whole codebase.


@lru_cache(maxsize=1)
def _cohort_medians_by_gene_id() -> dict[tuple[str, str], float]:
    """Map ``(cohort_code, versionless_gene_id) → median TPM_clean``.

    Cached because cancer_reference_expression is ~2M rows. The lookup
    table itself is ~70 cohorts × 20k genes ≈ 1.4M entries — a single
    GroupBy at first use, hot lookup after.
    """
    from pirlygenes import cancer_reference_expression
    from .common import _versionless_gene_id

    df = cancer_reference_expression()
    df = df[df["normalization"] == "TPM_clean"]
    out: dict[tuple[str, str], float] = {}
    for row in df.itertuples():
        gid = _versionless_gene_id(getattr(row, "Ensembl_Gene_ID", None))
        if not gid:
            continue
        out[(str(row.cancer_code), gid)] = float(row.expression)
    return out


@lru_cache(maxsize=1)
def _cohort_hk_medians() -> dict[str, float]:
    """Per-cohort housekeeping median, used as the HK denominator when
    normalizing a marker against its cohort distribution."""
    from pirlygenes import cancer_reference_expression, housekeeping_gene_ids

    df = cancer_reference_expression()
    hk_ids = set(housekeeping_gene_ids())
    mask = df["normalization"].eq("TPM_clean") & df["Ensembl_Gene_ID"].isin(hk_ids)
    hk = df.loc[mask, ["cancer_code", "expression"]]
    return {
        str(k): float(v)
        for k, v in hk.groupby("cancer_code")["expression"].median().items()
    }


def _cohort_median_by_id(cohort: str, gene_id: str) -> float | None:
    if not gene_id:
        return None
    return _cohort_medians_by_gene_id().get((cohort, gene_id))


def _cohort_hk(cohort: str) -> float:
    return _cohort_hk_medians().get(cohort, 1.0) or 1.0


@lru_cache(maxsize=64)
def _panel_marker_ids(panel: LineagePanel) -> dict[str, str]:
    """Cache the panel's full marker → gene_id resolution. Computed
    once per panel per process via the shared
    ``trufflepig.common.panel_symbols_to_gene_ids`` resolver — the
    only Ensembl-lookup path in the codebase.

    Returns ``{symbol: gene_id}``; symbols that don't resolve are
    omitted from the mapping. Callers should treat a missing symbol
    as "this marker couldn't be evaluated" (NOT as "marker passed").
    """
    from .common import panel_symbols_to_gene_ids

    syms: list[str] = []
    syms.extend(panel.obligate)
    syms.extend(panel.high_markers)
    syms.extend(sym for sym, _ in panel.low_markers)
    return panel_symbols_to_gene_ids(syms)


# Single-fire warning per (panel, symbol) when a panel marker
# fails to resolve — repeated firings would spam the log on every
# sample. The lru_cache makes the warning idempotent without extra
# bookkeeping.
@lru_cache(maxsize=None)
def _warn_unresolved_marker(panel_name: str, role: str, sym: str) -> None:
    import logging

    logging.getLogger(__name__).warning(
        "lineage_panels: %s panel marker %s (%s) could not be resolved "
        "to an Ensembl gene_id; the marker will be treated pessimistically",
        panel_name,
        sym,
        role,
    )


# ---------- the one scoring function ----------

# A positive lineage marker must be *actively expressed* in the sample to count
# as a hit, not merely detectable. 1 TPM is the detection floor and is too low to
# call a lineage-defining marker "present"; we use the same "primary marker"
# level as ``literature_signatures.min_primary_tpm`` (5 TPM). This also guards the
# ``0 >= 0.5 * 0`` edge case where a marker absent in both sample and cohort would
# otherwise be mis-counted as in-range (e.g. MUC5AC=0 inflating a CHOL panel).
_HIGH_MARKER_MIN_TPM = 5.0


def score_panel(
    panel: LineagePanel,
    sample_tpm_by_gene_id: Mapping[str, float],
    sample_hk_median: float,
) -> PanelEvidence:
    """Evaluate one panel against one sample.

    ``sample_tpm_by_gene_id`` is a ``{versionless_ensembl_id: TPM}``
    mapping; the panel's own marker symbols are resolved to gene IDs
    via the cached ``_panel_marker_ids`` helper and looked up against
    it. Rationale strings keep the HGNC symbol because that's what
    biologists read.

    Unresolvable symbols (pirlygenes unavailable, unmapped name) are
    handled pessimistically so a registry hiccup never makes a panel
    artificially STRONGER:

      - Unresolvable obligate → returns a no-score PanelEvidence with
        an obligate-not-evaluable rationale.
      - Unresolvable high_marker → recorded as a miss (zero contribution
        to the positive evidence count).
      - Unresolvable low_marker → recorded as a violation (the
        negative-marker check defaults to "presumed expressed" instead
        of silently skipping the check, which is the conservative
        choice for "this gene should be OFF" semantics).

    Each unresolvable marker is logged exactly once per process via
    ``_warn_unresolved_marker``.

    ``sample_hk_median`` is the sample's own housekeeping median
    (computed once per sample upstream and passed in — keeps this
    function pure).
    """

    cohort_hk = _cohort_hk(panel.parent_cohort)
    marker_ids = _panel_marker_ids(panel)

    def _sample(gid: str) -> float:
        return float(sample_tpm_by_gene_id.get(gid, 0.0))

    # Obligate check — required marker must clear cohort-normalized
    # half-of-cohort-HK-ratio bar, otherwise the panel returns 0.
    obligate_failures: list[tuple[str, float, float]] = []
    for sym in panel.obligate:
        gid = marker_ids.get(sym)
        if not gid:
            _warn_unresolved_marker(panel.name, "obligate", sym)
            return PanelEvidence(
                panel_name=panel.name,
                parent_cohort=panel.parent_cohort,
                obligate_passed=False,
                obligate_failures=((sym, 0.0, 0.0),),
                high_hits=(), high_misses=(),
                low_passes=(), low_violations=(),
                score=0.0,
                rationale=(
                    f"{panel.name}: obligate marker {sym} could not be "
                    "resolved to a gene_id — panel not scored"
                ),
            )
        cohort_val = _cohort_median_by_id(panel.parent_cohort, gid)
        if cohort_val is None:
            continue
        cohort_hk_ratio = cohort_val / cohort_hk
        sample_hk_ratio = _sample(gid) / sample_hk_median
        required = 0.5 * cohort_hk_ratio
        if sample_hk_ratio < required:
            obligate_failures.append((sym, sample_hk_ratio, required))

    if obligate_failures:
        sym, obs, req = obligate_failures[0]
        return PanelEvidence(
            panel_name=panel.name,
            parent_cohort=panel.parent_cohort,
            obligate_passed=False,
            obligate_failures=tuple(obligate_failures),
            high_hits=(), high_misses=(),
            low_passes=(), low_violations=(),
            score=0.0,
            rationale=(
                f"{panel.name}: required marker {sym} absent "
                f"(HK ratio {obs:.3f} vs threshold {req:.3f})"
            ),
        )

    # Positive markers — HK-normalized vs cohort
    high_hits: list[tuple[str, float, float]] = []
    high_misses: list[tuple[str, float, float]] = []
    for sym in panel.high_markers:
        gid = marker_ids.get(sym)
        if not gid:
            _warn_unresolved_marker(panel.name, "high", sym)
            # Pessimistic: count as a miss so an unresolved positive
            # marker can't inflate the panel's score.
            high_misses.append((sym, 0.0, 0.0))
            continue
        cohort_val = _cohort_median_by_id(panel.parent_cohort, gid)
        obs_tpm = _sample(gid)
        if cohort_val is None:
            high_misses.append((sym, obs_tpm, 0.0))
            continue
        sample_hk_ratio = obs_tpm / sample_hk_median
        cohort_hk_ratio = cohort_val / cohort_hk
        # A positive lineage marker can only count as a hit if it is actually
        # expressed in the sample. Without this floor, a marker that is ~0 in
        # both the sample and the cohort median trivially passes
        # ``0 >= 0.5 * 0`` and is mis-counted as "in cohort range" (e.g.
        # MUC5AC=0 inflating a CHOL panel).
        if obs_tpm >= _HIGH_MARKER_MIN_TPM and sample_hk_ratio >= 0.5 * cohort_hk_ratio:
            high_hits.append((sym, obs_tpm, cohort_val))
        else:
            high_misses.append((sym, obs_tpm, cohort_val))

    # Negative markers — absolute TPM threshold (interpretable near zero)
    low_passes: list[tuple[str, float, float]] = []
    low_violations: list[tuple[str, float, float]] = []
    for sym, cap in panel.low_markers:
        gid = marker_ids.get(sym)
        if not gid:
            _warn_unresolved_marker(panel.name, "low", sym)
            # Pessimistic: count as a violation so an unresolved
            # negative marker can't silently let a panel match a
            # sample where that marker is in fact expressed.
            low_violations.append((sym, float("nan"), cap))
            continue
        obs = _sample(gid)
        if obs < cap:
            low_passes.append((sym, obs, cap))
        else:
            low_violations.append((sym, obs, cap))

    # Synthesis — positives weighted harder than negatives
    high_total = max(1, len(panel.high_markers))
    low_total = max(1, len(panel.low_markers))
    high_frac = len(high_hits) / high_total
    low_frac = len(low_passes) / low_total
    score = (high_frac ** 0.6) * (low_frac ** 0.4)

    # Rationale — short, scannable, uses HGNC symbols since those are
    # what readers recognize. Markers with no Ensembl mapping show as
    # "(unresolved)" in the violation list so report readers can see
    # the panel didn't ignore the marker — it failed it pessimistically.
    hit_summary = ", ".join(f"{s}={t:.0f}" for s, t, _ in high_hits[:4])
    if not hit_summary:
        hit_summary = "(none)"
    violation_phrase = ""
    if low_violations:
        def _v_str(s, t, c):
            return f"{s}=unresolved" if t != t else f"{s}={t:.0f}>{c:.0f}"  # NaN check
        v = ", ".join(_v_str(s, t, c) for s, t, c in low_violations[:3])
        violation_phrase = f"; low-marker violations: {v}"
    rationale = (
        f"{panel.name}: {len(high_hits)}/{high_total} required markers in cohort range "
        f"({hit_summary}); {len(low_passes)}/{low_total} negative markers below threshold"
        f"{violation_phrase}"
    )

    return PanelEvidence(
        panel_name=panel.name,
        parent_cohort=panel.parent_cohort,
        obligate_passed=True,
        obligate_failures=(),
        high_hits=tuple(high_hits),
        high_misses=tuple(high_misses),
        low_passes=tuple(low_passes),
        low_violations=tuple(low_violations),
        score=score,
        rationale=rationale,
    )


def evaluate_panels(
    panels: tuple[LineagePanel, ...],
    sample_tpm_by_gene_id: Mapping[str, float],
    sample_hk_median: float,
) -> tuple[PanelEvidence, ...]:
    """Score every panel; return evidence sorted by score (highest first).

    Evidence for ALL panels is returned, not just the top — downstream
    reasoning (confidence, report rationale, conflicting-call detection)
    needs to see misses as well as hits.
    """
    evidence = tuple(
        score_panel(p, sample_tpm_by_gene_id, sample_hk_median) for p in panels
    )
    return tuple(sorted(evidence, key=lambda e: -e.score))


# ---------- curated panel set (start small, grow with evidence) ----------

# Each entry is ~15 lines including markers and rationale.
# Cohort medians are looked up at runtime — no manual q25/median/q75.

_BRCA_BASAL = LineagePanel(
    name="BRCA_BASAL",
    parent_cohort="BRCA",
    high_markers=("KRT14", "KRT5", "FOXC1", "MIA", "MUCL1"),
    low_markers=(
        ("ESR1", 5.0),
        ("PGR", 1.0),
        ("UPK1B", 1.0),    # discriminator vs basal-MIBC
        ("TP63", 30.0),    # incomplete squamous (TCGA-HNSC median ~50)
        ("SOX2", 5.0),     # incomplete squamous
    ),
    obligate=("KRT14",),
    description="Basal-like / TNBC mammary",
    references=("Hoadley 2014 Cell", "Lehmann 2011 JCI", "Damrauer 2014 PNAS"),
    program_note=(
        "basal-like breast program (KRT5/KRT14 cytokeratins, FOXC1, low ER/PR/HER2). "
        "Associated with triple-negative biology, BRCA1/HRD signatures, and PD-L1 candidacy"
    ),
)

_BRCA_LUMINAL = LineagePanel(
    name="BRCA_LUMINAL",
    parent_cohort="BRCA",
    high_markers=("ESR1", "PGR", "FOXA1", "GATA3", "MUCL1", "SCGB2A2"),
    low_markers=(
        ("KRT14", 50.0),   # basal cytokeratins off
        ("KRT5", 80.0),
        ("FOXC1", 5.0),    # basal TF off
    ),
    obligate=("FOXA1",),
    description="Luminal A/B/HER2 mammary",
    references=("Perou 2000 Nature", "Sorlie 2001 PNAS"),
    program_note=(
        "luminal breast program (ESR1, PGR, FOXA1, GATA3). "
        "Hormone-receptor-positive biology; anti-estrogen therapy context"
    ),
)

_ESCA_SQUAMOUS = LineagePanel(
    name="ESCA_SQUAMOUS",
    parent_cohort="ESCA",
    high_markers=("TP63", "SOX2", "KRT5", "KRT14", "AGR2", "MAL"),
    low_markers=(
        ("MUCL1", 1.0),    # not mammary
        ("FOXA1", 30.0),
        ("UPK1B", 1.0),
    ),
    obligate=("TP63",),
    description="Esophageal squamous (TCGA-ESCA is ~70% squamous)",
    program_note=(
        "squamous esophageal program (TP63, SOX2, keratins). "
        "Distinct from adenocarcinoma; chemoradiation-sensitive context"
    ),
)

_BLCA_LUMINAL = LineagePanel(
    name="BLCA_LUMINAL",
    parent_cohort="BLCA",
    high_markers=("UPK1A", "UPK1B", "UPK2", "UPK3A", "KRT20", "GATA3", "FOXA1"),
    low_markers=(
        ("MUCL1", 1.0),    # not mammary
        ("SCGB2A2", 1.0),
        ("CDX2", 5.0),     # not GI
    ),
    obligate=("UPK1B",),
    description="Urothelial luminal (MIBC luminal subtype)",
    references=("Damrauer 2014 PNAS", "Choi 2014 Cancer Cell"),
    program_note=(
        "luminal urothelial program (FOXA1, GATA3, PPARG, UPK1A/UPK2). "
        "Less chemo-sensitive than basal MIBC; FGFR3 alterations enriched"
    ),
)

_HNSC_PANEL = LineagePanel(
    name="HNSC",
    parent_cohort="HNSC",
    high_markers=("TP63", "SOX2", "KRT5", "KRT14", "KLK10", "SPRR2A"),
    low_markers=(
        ("MUCL1", 1.0),
        ("FOXA1", 30.0),
        ("UPK1B", 1.0),
        ("AGR2", 30.0),    # AGR2 high in ESCA, lower in HNSC
    ),
    obligate=("TP63",),
    description="Head & neck squamous",
    program_note=(
        "head-and-neck squamous program (keratins, p63, SOX2). "
        "HPV status independently informative"
    ),
)

_LIHC_PANEL = LineagePanel(
    name="LIHC",
    parent_cohort="LIHC",
    high_markers=("AFP", "ALB", "F2", "HNF4A", "HP", "APOB"),
    low_markers=(
        ("KRT19", 30.0),   # cholangio elevation
        ("MUC1", 50.0),
        ("MUC5AC", 5.0),
    ),
    obligate=("ALB",),
    description="Hepatocellular carcinoma",
    program_note=(
        "hepatocyte program (albumin synthesis, drug-metabolism enzymes). "
        "Be alert for drug-drug interactions from preserved hepatic clearance"
    ),
)

_PAAD_PANEL = LineagePanel(
    name="PAAD_DUCTAL",
    parent_cohort="PAAD",
    high_markers=("KRT19", "MUC1", "CLDN18", "AGR2", "S100P"),
    low_markers=(
        ("ALB", 5.0),      # not hepatocyte
        ("AFP", 5.0),
        ("CDX2", 30.0),    # not GI adeno
    ),
    obligate=("KRT19",),
    description="Pancreatic ductal adenocarcinoma",
    program_note=(
        "pancreatic ductal program (KRT19, MUC1, CDX2-negative). "
        "Stromal-dominant tumor microenvironment context"
    ),
)

# --- Panels added to close TCGA-160 top-3 misses (issue #42) ---

_CHOL_PANEL = LineagePanel(
    name="CHOL",
    parent_cohort="CHOL",
    high_markers=("KRT19", "MUC1", "MUC5AC", "EPCAM", "CDH1", "CFTR"),
    low_markers=(
        ("ALB", 30.0),     # not hepatocyte (otherwise → LIHC)
        ("AFP", 5.0),
        ("CDX2", 30.0),    # not colorectal (otherwise → COAD/READ)
        ("CDH17", 50.0),
        ("VIL1", 50.0),
    ),
    obligate=("KRT19",),
    description="Cholangiocarcinoma — biliary epithelium",
    program_note=(
        "biliary epithelial program (KRT19, CFTR, MUC5AC). "
        "Distinct from hepatocellular biology; FGFR2 fusions enriched"
    ),
)

_UCEC_PANEL = LineagePanel(
    name="UCEC",
    parent_cohort="UCEC",
    high_markers=("PAX8", "ESR1", "PGR", "FOXA2", "HOXA10", "WT1"),
    low_markers=(
        ("VIM", 200.0),    # not mesenchymal-dominant (UCS shows VIM high)
        ("KRT5", 50.0),    # not basal-squamous
        ("KRT14", 50.0),
        ("MUC16", 1.0),    # OV uses MUC16 (CA125); UCEC variable
    ),
    obligate=("PAX8",),
    description="Endometrial adenocarcinoma — Müllerian glandular",
    program_note=(
        "endometrial Müllerian program (PAX8, ESR1, FOXA2). "
        "Hormone-receptor context overlapping with luminal breast"
    ),
)

_MESO_PANEL = LineagePanel(
    name="MESO",
    parent_cohort="MESO",
    high_markers=("WT1", "MSLN", "CALB2", "UPK3B", "KRT5", "PDPN", "KRT8"),
    low_markers=(
        ("SFTPC", 5.0),    # not LUAD (alveolar surfactant)
        ("NAPSA", 10.0),
        ("CDX2", 5.0),     # not GI
        ("FOXA1", 30.0),
        ("MUCL1", 1.0),    # not mammary
    ),
    obligate=("MSLN",),    # mesothelin is the canonical mesothelial marker
    description="Mesothelioma — mesothelium",
    references=("Hassan 2014",),
    program_note=(
        "mesothelial program (MSLN, WT1, CALB2). "
        "Mesothelin is a therapy target; BAP1 loss enriched"
    ),
)

_ACC_PANEL = LineagePanel(
    name="ACC",
    parent_cohort="ACC",
    high_markers=("CYP11A1", "CYP17A1", "CYP21A2", "STAR", "NR5A1", "INHA", "MC2R"),
    low_markers=(
        ("KRT5", 30.0),    # not squamous
        ("KRT14", 30.0),
        ("TP63", 30.0),
        ("TG", 5.0),       # not thyroid (otherwise → THCA)
        ("PAX8", 10.0),    # PAX8 is gyn/thyroid; low in ACC
    ),
    # NR5A1 (SF1) is the master TF for adrenocortical lineage. Without
    # it, the sample isn't expressing the ACC program — dedifferentiated
    # variants will fail this obligate, leaving no lineage panel hypothesis
    # for ACC. That IS the correct honest-reporting outcome per issue #42.
    obligate=("NR5A1",),
    description="Adrenocortical carcinoma — cortical steroidogenic",
    references=("Roden 2024",),
    program_note=(
        "adrenocortical steroidogenic program (CYP11A1/CYP17A1, NR5A1/SF1, STAR). "
        "Active cortisol/aldosterone biology; mitotane sensitivity context"
    ),
)

_THYM_PANEL = LineagePanel(
    name="THYM",
    parent_cohort="THYM",
    high_markers=("AIRE", "FOXN1", "PSMB11", "PRSS16", "CCL25", "CHRNA1"),
    low_markers=(
        ("MUCL1", 1.0),    # not mammary
        ("UPK1B", 1.0),    # not urothelial
        ("FOXA1", 30.0),
    ),
    obligate=("FOXN1",),   # thymic epithelium master TF
    description="Thymoma — thymic epithelium",
    program_note=(
        "thymic-epithelial program (AIRE, FOXN1, PSMB11). "
        "Paraneoplastic autoimmune context; checkpoint-inhibitor caution"
    ),
)

_PCPG_PANEL = LineagePanel(
    name="PCPG",
    parent_cohort="PCPG",
    # Mature adrenal-medulla chromaffin / extra-adrenal paraganglion
    # program. CHGA / SYN1 are pan-neuroendocrine; PNMT (phenylethanolamine
    # N-methyltransferase) is what distinguishes adult adrenal medulla
    # (PCPG) from immature neural-crest tumors (NBL). TH / DBH / NPY
    # are catecholamine-pathway enzymes.
    high_markers=("CHGA", "SYN1", "PNMT", "TH", "DBH", "NPY", "CHGB"),
    low_markers=(
        ("ALB", 5.0),      # not hepatocyte (otherwise → LIHC override)
        ("KRT19", 30.0),   # not ductal
        ("MUC1", 30.0),
        ("CDX2", 5.0),     # not GI
        ("PHOX2B", 30.0),  # NBL master TF — should be LOW in mature PCPG
        ("LIN28B", 5.0),   # fetal marker, NBL-associated
        ("MYCN", 5.0),     # NBL amplification context
    ),
    # PNMT is the canonical mature-adrenal-medulla marker. Without it,
    # the sample isn't expressing the chromaffin program — extra-adrenal
    # paragangliomas may lack PNMT (they're noradrenergic), so we use
    # CHGA as the obligate to keep PGL within scope; PNMT still scores
    # as a positive marker when present.
    obligate=("CHGA",),
    description="Pheochromocytoma / paraganglioma — adrenal-medulla chromaffin",
    references=("Burnichon 2017 Nat Rev Endocrinol", "Fishbein 2017 Cancer Cell"),
    program_note=(
        "neuroendocrine chromaffin program (CHGA, SYN1, PNMT, TH, DBH). "
        "Adrenergic biology, catecholamine secretion context; SDHx/RET/VHL "
        "germline mutations enriched, especially in PGL"
    ),
)

_LUAD_PANEL = LineagePanel(
    name="LUAD",
    parent_cohort="LUAD",
    # Alveolar adenocarcinoma program. NKX2-1 (TTF1) is the master TF
    # for distal-lung epithelium and is the canonical lineage discriminator
    # vs squamous (LUSC), GI adeno (STAD/PAAD), and mesothelial (MESO).
    high_markers=("SFTPC", "SFTPA1", "SFTPA2", "SFTPB", "NAPSA", "NKX2-1"),
    low_markers=(
        ("KRT5", 30.0),    # not squamous
        ("KRT14", 30.0),
        ("TP63", 30.0),
        ("CDX2", 5.0),     # not GI
        ("MUC2", 5.0),     # not mucinous GI
        ("MUCL1", 1.0),    # not mammary
        ("MSLN", 30.0),    # not mesothelial (otherwise → MESO)
    ),
    obligate=("NKX2-1",),  # alveolar epithelial master TF
    description="Lung adenocarcinoma — alveolar epithelium",
    references=("Travis 2015 WHO classification", "Yatabe 2002 Am J Surg Pathol"),
    program_note=(
        "alveolar adenocarcinoma program (TTF1/NKX2-1, surfactant proteins, NAPSA). "
        "EGFR/KRAS/ALK-driven biology context; distinct from squamous (TP63-high) "
        "and mucinous (MUC2/CDX2) lung tumors"
    ),
)

_BLCA_BASAL_PANEL = LineagePanel(
    name="BLCA_BASAL",
    parent_cohort="BLCA",
    # Basal MIBC retains partial uroplakin expression but loses the
    # luminal urothelial differentiation; squamous-like keratins rise.
    high_markers=("KRT5", "KRT14", "KRT6A", "UPK1B", "UPK2", "S100P"),
    low_markers=(
        ("MUCL1", 1.0),    # not mammary (otherwise → BRCA_BASAL)
        ("SCGB2A2", 1.0),
        ("FOXA1", 200.0),  # luminal urothelial has FOXA1/GATA3 very high
        ("GATA3", 200.0),
    ),
    obligate=("UPK1B",),   # urothelial marker — distinguishes from BRCA_BASAL
    description="Basal-like muscle-invasive bladder cancer",
    references=("Damrauer 2014 PNAS", "Choi 2014 Cancer Cell"),
    program_note=(
        "basal-like muscle-invasive bladder program (KRT5/KRT14/KRT6A, S100P, low FOXA1/GATA3). "
        "More chemo-sensitive than luminal MIBC; squamous-like differentiation"
    ),
)


# Public registry. Tier-1 broad triage (squamous-vs-glandular-vs-...)
# is a separate concern — these are tier-3 within-tier discriminators.
LINEAGE_PANELS: tuple[LineagePanel, ...] = (
    _BRCA_BASAL,
    _BRCA_LUMINAL,
    _ESCA_SQUAMOUS,
    _BLCA_LUMINAL,
    _BLCA_BASAL_PANEL,
    _HNSC_PANEL,
    _LIHC_PANEL,
    _PAAD_PANEL,
    _CHOL_PANEL,
    _UCEC_PANEL,
    _MESO_PANEL,
    _ACC_PANEL,
    _THYM_PANEL,
    _PCPG_PANEL,
    _LUAD_PANEL,
)


# ---------- downstream-reasoning helpers (the "info preserved" part) ----------

def summarize_evidence(evidence: tuple[PanelEvidence, ...]) -> dict[str, Any]:
    """Aggregate evidence into a shape suitable for analysis output.

    The returned dict is the contract that downstream consumers
    (confidence, report rendering, cancer_type_evidence selectors)
    read — keep stable across versions.
    """
    if not evidence:
        return {
            "top_panel": None,
            "top_score": 0.0,
            "margin_over_second": 0.0,
            "panels": [],
        }
    top = evidence[0]
    second_score = evidence[1].score if len(evidence) > 1 else 0.0
    margin = top.score - second_score
    return {
        "top_panel": top.panel_name,
        "top_score": top.score,
        "top_rationale": top.rationale,
        "margin_over_second": margin,
        "panels": [e.to_dict() for e in evidence],
    }


def conflicting_calls(
    evidence: tuple[PanelEvidence, ...],
    tie_threshold: float = 0.10,
) -> tuple[str, ...]:
    """Identify panels within ``tie_threshold`` of the top score.

    Returned for downstream confidence assessment — a sample that
    fires BRCA_BASAL=0.85 and ESCA_SQUAMOUS=0.82 is a real ambiguity
    and the confidence badge should reflect it.
    """
    if len(evidence) < 2:
        return ()
    top_score = evidence[0].score
    return tuple(
        e.panel_name for e in evidence
        if e.score >= top_score - tie_threshold and e.panel_name != evidence[0].panel_name
    )


# ---------- wiring contract (NOT yet wired) ----------
#
# Downstream consumers integrate via three points, in order of how
# invasive they are:
#
#   1. NEW SELECTOR in cancer_type_evidence (~30 lines):
#      Add a ``lineage_panel`` selector that calls
#      ``evaluate_panels(LINEAGE_PANELS, sample_tpm, hk_median)``.
#      Promotes a hypothesis when ``summarize_evidence`` shows a clear
#      winner above some threshold (e.g. top score >= 0.6 AND margin
#      over second >= 0.2). Lives between ``rare_marker`` (priority 2)
#      and ``tumor_label_refinement`` (priority 3) in the selector
#      cascade. The selector output naturally feeds the existing
#      cancer_type_evidence consolidation — no other code path changes.
#
#   2. ANALYSIS DICT exposure (~5 lines in main.py):
#      Persist ``summarize_evidence(...)`` to
#      ``analysis["lineage_panel_evidence"]`` so it appears in
#      ``analysis-parameters.json`` for debugging / external consumers.
#
#   3. REPORT RENDERING (~15 lines in brief.py / reporting.py):
#      When summarize_evidence returns top_score >= 0.5, surface the
#      top rationale string in the report's "cancer-type basis" line.
#      This mirrors the existing basal-BRCA rescue text but generalizes
#      to every panel.
#
# No change to ``_score_cancer_family_panels``, no change to broad
# classification, no change to the rescue paths. The lineage-panel
# evidence is purely additive — a new line of evidence, not a
# replacement for any existing one.
