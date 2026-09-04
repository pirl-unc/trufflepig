# Licensed under the Apache License, Version 2.0

"""Lightweight lineage-discrimination panels.

Each panel evaluates a single hypothesis: "is this sample a member of
lineage X?" using a small, curated set of markers. The score is a
synthesis of two evidence streams kept separately because they answer
different biological questions:

  - **High-direction markers** ("this gene should be ON") score better
    by combining compressed absolute clean-TPM burden with the marker's
    percentile among cancer cohorts.
  - **Low-direction markers** ("this gene should be OFF") score better
    as absolute TPM thresholds, because the question is "is it
    essentially zero?"

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

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, asdict
from functools import lru_cache
from statistics import median
from typing import Any, Mapping



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
    """Genes that should be ON in this lineage. Scored by log1p(clean TPM)
    burden and cross-cancer cohort percentile."""

    reference_cohorts: tuple[str, ...] = ()
    """Optional loadable child cohorts used to construct a parent-level
    reference. This keeps the reported identity at the defensible parent while
    allowing aggregate registry nodes such as CRC to use their observed
    children (COAD and READ) as the expression reference."""

    low_markers: tuple[tuple[str, float], ...] = ()
    """(symbol, max_tpm) — gene should be below the absolute TPM cap.
    Absolute TPM is more interpretable than HK ratios near zero."""

    obligate: tuple[str, ...] = ()
    """Every listed marker must reach 0.5 integrated positive support or
    the panel returns score=0 and an obligate-failed rationale."""

    identity_marker_groups: tuple[tuple[str, ...], ...] = ()
    """Alternative marker groups that establish tissue identity.

    Every group must contribute at least one actively expressed marker. This
    represents biological alternatives without making one transcript a
    brittle veto: for example, urothelial plaques contain a UPK1-family
    subunit and a UPK2/3-family partner, but an individual tumor need not
    retain every uroplakin. These gates establish lineage identity; the
    ordinary ``high_markers`` still describe the phenotype within it.
    """

    background_attribution_markers: tuple[tuple[str, str], ...] = ()
    """Expected-low markers that can originate in a named normal component.

    These remain ordinary low-marker violations in bulk RNA. Background-separated
    analysis may clear one only when a candidate-independent decomposition
    identifies the named component as the dominant fitted background.  This
    keeps a host-tissue transcript from masquerading as tumor-cell evidence
    without weakening the bulk panel or adding a sample-specific threshold.
    """

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
    """(symbol, observed_support, required_support) for each failed obligate
    marker. Empty when obligate_passed."""

    high_hits: tuple[tuple[str, float, float], ...]
    """(symbol, observed_tpm, cohort_median_tpm) for high_markers that
    reached 0.5 integrated positive support."""
    high_misses: tuple[tuple[str, float, float], ...]
    """Same shape as high_hits, for markers that did not clear."""

    low_passes: tuple[tuple[str, float, float], ...]
    """(symbol, observed_tpm, cap_tpm) for low_markers under the cap."""
    low_violations: tuple[tuple[str, float, float], ...]
    """Same shape as low_passes, for markers above the cap."""

    score: float
    """Synthesis: (mean_high_support ** 0.6) × (low_fraction ** 0.4).
    Range [0, 1]. The exponents weight positive evidence more than
    negative compliance — a panel with all positives firing but one
    negative violation is still strong evidence, while a panel with
    only negative compliance and no positives is weak."""

    rationale: str
    """One human-readable sentence describing the panel verdict for
    reports. Mirrors the style of the existing basal-BRCA rescue text."""

    identity_marker_groups_passed: bool = True
    """Whether every configured identity-marker group had an expressed hit."""

    identity_marker_hits: tuple[tuple[str, float], ...] = ()
    """Flattened ``(symbol, TPM)`` hits across the identity-marker groups."""

    identity_marker_group_failures: tuple[tuple[str, ...], ...] = ()
    """Configured alternative groups for which no marker was expressed."""

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


def _marker_symbols_for_panels(
    panels: tuple[LineagePanel, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                symbol
                for panel in panels
                for symbol in (
                    *panel.high_markers,
                    *panel.obligate,
                    *(s for group in panel.identity_marker_groups for s in group),
                    *(s for s, _ in panel.low_markers),
                )
            }
        )
    )


def _reference_symbols_for_panel(panel: LineagePanel) -> tuple[str, ...]:
    """Use one shared key for built-ins and an exact key for custom panels."""

    built_in_panels = tuple(globals().get("LINEAGE_PANELS", ()))
    panels = built_in_panels if panel in built_in_panels else (panel,)
    return _marker_symbols_for_panels(panels)


@lru_cache(maxsize=64)
def _cohort_medians_by_gene_id(
    symbols: tuple[str, ...] = (),
) -> dict[tuple[str, str], float]:
    """Map ``(cohort_code, versionless_gene_id) → median TPM_clean``.

    The cache key is the marker universe being evaluated. Calls without an
    explicit universe retain the shared built-in-panel path; public custom
    panels use their own bounded symbol tuple.
    """

    from .common import _versionless_gene_id
    from .reference import cancer_reference_expression

    if not symbols:
        panels = tuple(globals().get("LINEAGE_PANELS", ()))
        symbols = _marker_symbols_for_panels(panels)
    if not symbols:
        return {}
    df = cancer_reference_expression(genes=list(symbols), normalize="tpm_clean")
    df = df[df["normalization"] == "TPM_clean"]
    df = df.assign(
        _gene_id=df["Ensembl_Gene_ID"].map(_versionless_gene_id),
        _expression=df["expression"].astype(float).clip(lower=0.0),
    )
    df = df[df["_gene_id"].astype(bool)]
    grouped = df.groupby(["cancer_code", "_gene_id"], sort=False)[
        "_expression"
    ].median()
    return {
        (str(cancer_code), str(gene_id)): float(expression)
        for (cancer_code, gene_id), expression in grouped.items()
    }


@lru_cache(maxsize=64)
def _cohort_values_by_gene_id(
    symbols: tuple[str, ...] = (),
) -> dict[str, tuple[float, ...]]:
    """Sorted clean-TPM medians across all reference cohorts, per gene."""
    values: dict[str, list[float]] = {}
    for (_, gene_id), value in _cohort_medians_by_gene_id(symbols).items():
        if math.isfinite(value):
            values.setdefault(gene_id, []).append(max(0.0, float(value)))
    return {
        gene_id: tuple(sorted(gene_values))
        for gene_id, gene_values in values.items()
    }


def _cohort_median_by_id(
    cohort: str,
    gene_id: str,
    reference_symbols: tuple[str, ...] = (),
) -> float | None:
    if not gene_id:
        return None
    return _cohort_medians_by_gene_id(reference_symbols).get((cohort, gene_id))


def _panel_reference_cohorts(panel: LineagePanel) -> tuple[str, ...]:
    return panel.reference_cohorts or (panel.parent_cohort,)


def _panel_cohort_median(
    panel: LineagePanel,
    gene_id: str,
    reference_symbols: tuple[str, ...] = (),
) -> float | None:
    reference_symbols = reference_symbols or _reference_symbols_for_panel(panel)
    values = [
        value
        for cohort in _panel_reference_cohorts(panel)
        if (
            value := _cohort_median_by_id(cohort, gene_id, reference_symbols)
        ) is not None
    ]
    return float(median(values)) if values else None


def _cohort_percentile(
    gene_id: str,
    value: float,
    reference_symbols: tuple[str, ...] = (),
) -> float:
    """Midrank percentile of ``value`` among clean-TPM cohort medians."""
    values = _cohort_values_by_gene_id(reference_symbols).get(gene_id, ())
    if not values:
        return 0.0
    left = bisect_left(values, value)
    right = bisect_right(values, value)
    return float((left + right) / (2.0 * len(values)))


def _positive_marker_support(
    panel: LineagePanel,
    gene_id: str,
    observed_tpm: float,
) -> tuple[float, float | None]:
    """Integrate absolute burden and cross-cancer specificity for one marker."""
    reference_symbols = _reference_symbols_for_panel(panel)
    reference_tpm = _panel_cohort_median(panel, gene_id, reference_symbols)
    if reference_tpm is None:
        return 0.0, None
    reference_tpm = max(0.0, float(reference_tpm))
    observed_tpm = max(0.0, float(observed_tpm))
    reference_log = math.log1p(reference_tpm)
    log_support = (
        min(1.0, math.log1p(observed_tpm) / reference_log)
        if reference_log > 0.0
        else float(observed_tpm > 0.0)
    )
    reference_percentile = _cohort_percentile(
        gene_id,
        reference_tpm,
        reference_symbols,
    )
    sample_percentile = _cohort_percentile(
        gene_id,
        observed_tpm,
        reference_symbols,
    )
    cohort_support = min(
        1.0,
        sample_percentile / reference_percentile,
    ) if reference_percentile > 0.0 else float(sample_percentile > 0.0)
    return float((log_support + cohort_support) / 2.0), reference_tpm


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
    for group in panel.identity_marker_groups:
        syms.extend(group)
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

    Both sample and reference values are clean TPM. No housekeeping
    normalization is used.
    """
    marker_ids = _panel_marker_ids(panel)

    def _sample(gid: str) -> float:
        return float(sample_tpm_by_gene_id.get(gid, 0.0))

    # Tissue-identity gates use active expression, not distance from the cohort
    # median. The latter is unstable in highly concentrated profiles (a basal
    # squamous tumor can devote most reads to keratins and thereby depress
    # every other gene's proportional abundance). Requiring one member from every
    # biologically interchangeable group preserves specificity without making
    # any single transcript an all-or-nothing veto.
    identity_hits: list[tuple[str, float]] = []
    identity_failures: list[tuple[str, ...]] = []
    for group in panel.identity_marker_groups:
        group_hits: list[tuple[str, float]] = []
        for sym in group:
            gid = marker_ids.get(sym)
            if not gid:
                _warn_unresolved_marker(panel.name, "identity", sym)
                continue
            observed = _sample(gid)
            if observed >= _HIGH_MARKER_MIN_TPM:
                group_hits.append((sym, observed))
        if group_hits:
            identity_hits.extend(group_hits)
        else:
            identity_failures.append(tuple(group))

    if identity_failures:
        missing = " or ".join(identity_failures[0])
        return PanelEvidence(
            panel_name=panel.name,
            parent_cohort=panel.parent_cohort,
            obligate_passed=True,
            obligate_failures=(),
            high_hits=(),
            high_misses=(),
            low_passes=(),
            low_violations=(),
            score=0.0,
            rationale=(
                f"{panel.name}: tissue-identity marker group absent "
                f"({missing}) — panel not scored"
            ),
            identity_marker_groups_passed=False,
            identity_marker_hits=tuple(identity_hits),
            identity_marker_group_failures=tuple(identity_failures),
        )

    # Obligate check — required markers must be both expressed and supported
    # by the parent-vs-cross-cancer comparison.
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
                identity_marker_hits=tuple(identity_hits),
            )
        observed = _sample(gid)
        support, cohort_val = _positive_marker_support(panel, gid, observed)
        if cohort_val is None:
            continue
        required = 0.5
        if observed < _HIGH_MARKER_MIN_TPM or support < required:
            obligate_failures.append((sym, support, required))

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
                f"(integrated clean-TPM support {obs:.3f} vs {req:.3f})"
            ),
            identity_marker_hits=tuple(identity_hits),
        )

    # Positive markers — log1p(clean TPM) burden + cross-cancer specificity.
    high_hits: list[tuple[str, float, float]] = []
    high_misses: list[tuple[str, float, float]] = []
    high_supports: list[float] = []
    for sym in panel.high_markers:
        gid = marker_ids.get(sym)
        if not gid:
            _warn_unresolved_marker(panel.name, "high", sym)
            # Pessimistic: count as a miss so an unresolved positive
            # marker can't inflate the panel's score.
            high_misses.append((sym, 0.0, 0.0))
            high_supports.append(0.0)
            continue
        obs_tpm = _sample(gid)
        support, cohort_val = _positive_marker_support(panel, gid, obs_tpm)
        if cohort_val is None:
            high_misses.append((sym, obs_tpm, 0.0))
            high_supports.append(0.0)
            continue
        if obs_tpm < _HIGH_MARKER_MIN_TPM:
            support = 0.0
        high_supports.append(support)
        if support >= 0.5:
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
    high_frac = sum(high_supports) / high_total
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
        identity_marker_hits=tuple(identity_hits),
    )


def evaluate_panels(
    panels: tuple[LineagePanel, ...],
    sample_tpm_by_gene_id: Mapping[str, float],
) -> tuple[PanelEvidence, ...]:
    """Score every panel; return evidence sorted by score (highest first).

    Evidence for ALL panels is returned, not just the top — downstream
    reasoning (confidence, report rationale, conflicting-call detection)
    needs to see misses as well as hits.
    """
    evidence = tuple(score_panel(p, sample_tpm_by_gene_id) for p in panels)
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
    identity_marker_groups=(
        ("UPK1A", "UPK1B"),
        ("UPK2", "UPK3A"),
    ),
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
    # The keratins define the basal/squamous phenotype. Urothelial identity is
    # established separately by the two complementary uroplakin groups below,
    # so loss of any one uroplakin cannot zero the whole program.
    high_markers=("KRT5", "KRT14", "KRT6A"),
    low_markers=(
        ("MUCL1", 1.0),    # not mammary (otherwise → BRCA_BASAL)
        ("SCGB2A2", 1.0),
        ("FOXA1", 200.0),  # luminal urothelial has FOXA1/GATA3 very high
        ("GATA3", 200.0),
    ),
    identity_marker_groups=(
        ("UPK1A", "UPK1B"),
        ("UPK2", "UPK3A"),
    ),
    description="Basal-like muscle-invasive bladder cancer",
    references=("Damrauer 2014 PNAS", "Choi 2014 Cancer Cell"),
    program_note=(
        "basal-like muscle-invasive bladder program (KRT5/KRT14/KRT6A, S100P, low FOXA1/GATA3). "
        "More chemo-sensitive than luminal MIBC; squamous-like differentiation"
    ),
)

_CRC_PANEL = LineagePanel(
    name="CRC",
    parent_cohort="CRC",
    reference_cohorts=("COAD", "READ"),
    high_markers=("VIL1", "CDH17", "SATB2", "CDX2", "KRT20", "MUC2"),
    low_markers=(
        ("ALB", 5.0),
        ("AFP", 5.0),
        ("UPK1B", 1.0),
        ("MYOD1", 5.0),
        ("MYOG", 5.0),
        ("DES", 10.0),
    ),
    obligate=("CDX2",),
    identity_marker_groups=(
        ("SATB2", "CDX2"),
        ("CDH17", "VIL1"),
    ),
    background_attribution_markers=(("DES", "smooth_muscle"),),
    description="Colorectal adenocarcinoma — shared colon/rectal identity",
    references=("WHO Classification of Tumours, Digestive System Tumours",),
    program_note=(
        "colorectal epithelial program (CDX2, SATB2, CDH17, VIL1, KRT20, MUC2). "
        "This parent-level program establishes CRC identity but does not by "
        "itself distinguish colon from rectal origin"
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
    _CRC_PANEL,
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


def complete_program_entity_decision(
    evidence: tuple[PanelEvidence, ...],
) -> dict[str, Any]:
    """Describe whether the top panel is an unopposed complete program.

    Panel scores are useful summaries, but they are not the whole evidentiary
    structure. A panel with every expected-high marker present, every
    expected-low marker compliant, and its obligate marker present is
    qualitatively stronger than an incomplete runner-up. This helper preserves
    that distinction without introducing another numeric cutoff.

    Competition is evaluated between parent cancer entities, not between two
    subtype panels that both map to the same parent. If a second entity also
    has a complete program, the result remains ambiguous.
    """
    if not evidence:
        return {
            "decisive": False,
            "reason": "no panel evidence",
            "top_parent_cohort": None,
            "competing_parent_cohort": None,
        }

    top = evidence[0]

    def _complete(row: PanelEvidence) -> bool:
        return bool(
            getattr(row, "obligate_passed", False)
            and getattr(row, "identity_marker_groups_passed", True)
            and getattr(row, "high_hits", ())
            and getattr(row, "low_passes", ())
            and not getattr(row, "high_misses", ())
            and not getattr(row, "low_violations", ())
        )

    top_parent = str(getattr(top, "parent_cohort", "") or "")
    competing_rows = tuple(
        row
        for row in evidence[1:]
        if str(getattr(row, "parent_cohort", "") or "") != top_parent
    )
    competing = competing_rows[0] if competing_rows else None
    top_complete = _complete(top)
    competing_complete = any(_complete(row) for row in competing_rows)
    decisive = bool(top_complete and not competing_complete)
    if not top_complete:
        reason = "top panel is not a complete positive/negative-marker program"
    elif competing_complete:
        reason = (
            "another cancer entity also has a complete positive/negative-marker "
            "program"
        )
    else:
        reason = (
            "top panel is complete and no competing cancer entity has a "
            "complete program"
        )
    return {
        "decisive": decisive,
        "reason": reason,
        "top_parent_cohort": top_parent or None,
        "top_panel": top.panel_name,
        "top_complete": top_complete,
        "top_identity_specific": bool(
            getattr(top, "identity_marker_hits", ())
        ),
        "competing_parent_cohort": (
            getattr(competing, "parent_cohort", None)
            if competing is not None
            else None
        ),
        "competing_panel": competing.panel_name if competing is not None else None,
        "competing_complete": competing_complete,
        "margin_over_competing_entity": (
            float(top.score - competing.score)
            if competing is not None
            else float(top.score)
        ),
    }


def attribute_panel_markers_to_decomposition(
    panel_evidence: Mapping[str, Any],
    gene_attribution: Any,
) -> dict[str, Any]:
    """Attribute a panel's positive markers to tumor or background.

    The decomposition table contains one expression contribution per modeled
    component. We call a marker tumor-residual only when ``tumor`` is its
    largest modeled contributor. This is a structural comparison—there is no
    new tumor-fraction threshold to tune.

    The result is post-selection corroboration. It must not be counted as an
    independent cancer-classifier vote because the decomposition candidates
    already reuse ranker support.
    """
    high_hits = list(panel_evidence.get("high_hits") or [])
    symbols = [str(row[0]).strip() for row in high_hits if row]
    empty_result = {
        "status": "not_evaluable",
        "role": "post_selection_corroboration",
        "panel": panel_evidence.get("panel_name"),
        "parent_cohort": panel_evidence.get("parent_cohort"),
        "evaluated_marker_count": 0,
        "tumor_dominant_count": 0,
        "ambiguous_marker_count": 0,
        "markers": [],
    }
    if (
        not symbols
        or gene_attribution is None
        or getattr(gene_attribution, "empty", True)
    ):
        return empty_result

    metadata_columns = {
        "gene_id",
        "symbol",
        "observed_tpm",
        "overexplained_tpm",
        "tumor_fraction_of_total",
    }
    component_columns = [
        str(column)
        for column in gene_attribution.columns
        if str(column) not in metadata_columns
    ]
    if "tumor" not in component_columns:
        return empty_result

    rows_by_symbol = {}
    symbol_set = set(symbols)
    for _, row in gene_attribution.iterrows():
        symbol = row.get("symbol")
        if isinstance(symbol, str) and symbol.strip() in symbol_set:
            rows_by_symbol[symbol.strip()] = row
    markers: list[dict[str, Any]] = []
    ambiguous_count = 0
    for symbol in symbols:
        row = rows_by_symbol.get(symbol)
        if row is None:
            continue
        contributions = {}
        for component in component_columns:
            try:
                value = float(row.get(component) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            contributions[component] = value if math.isfinite(value) else 0.0
        largest = max(contributions.values())
        dominant_components = sorted(
            component
            for component, value in contributions.items()
            if value == largest
        )
        unambiguous = len(dominant_components) == 1
        dominant_component = (
            dominant_components[0] if unambiguous else "ambiguous"
        )
        if not unambiguous:
            ambiguous_count += 1
        markers.append(
            {
                "symbol": symbol,
                "observed_tpm": round(float(row.get("observed_tpm") or 0.0), 4),
                "dominant_component": dominant_component,
                "dominant_components": dominant_components,
                "tumor_dominant": unambiguous and dominant_component == "tumor",
                "tumor_contribution": round(contributions["tumor"], 4),
            }
        )

    tumor_dominant_count = sum(row["tumor_dominant"] for row in markers)
    if not markers:
        status = "not_evaluable"
    elif tumor_dominant_count == len(markers):
        status = "tumor_residual"
    elif tumor_dominant_count or ambiguous_count:
        status = "mixed_tumor_and_background"
    else:
        status = "background_attributed"
    return {
        "status": status,
        "role": "post_selection_corroboration",
        "panel": panel_evidence.get("panel_name"),
        "parent_cohort": panel_evidence.get("parent_cohort"),
        "evaluated_marker_count": len(markers),
        "tumor_dominant_count": tumor_dominant_count,
        "ambiguous_marker_count": ambiguous_count,
        "markers": markers,
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
#      ``evaluate_panels(LINEAGE_PANELS, sample_tpm)``.
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
