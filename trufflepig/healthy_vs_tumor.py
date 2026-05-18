# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Step-0 tissue-composition + cancer-hint gate (#149, refined).

Runs BEFORE cancer-type classification. Produces the coarsest
possible reading of "what kind of tissue is this, and is there a
hint of cancer?" and propagates the result forward so later steps
can refine. Explicitly does NOT make a binary healthy-vs-cancer call
on the first pass — healthy and low-purity-tumor look similar here
and will be distinguished downstream once lineage markers, purity,
and therapy-axis signals have been read.

Output:

- ``top_normal_tissues``: the three HPA normal-tissue columns that
  best correlate with the sample's log-TPM profile, each with a
  Spearman rho.
- ``top_tcga_cohorts``: the three TCGA cohorts that best correlate.
- ``proliferation_log2_mean``: geomean on log-TPM of the public
  mitotic proliferation panel returned by
  ``proliferation_panel_gene_names()``.
- ``cancer_hint``: one of ``"tumor-consistent"``, ``"possibly-tumor"``,
  ``"healthy-dominant"`` — a coarse read, not a commitment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pirlygenes.gene_sets_cancer import (

    CTA_gene_names, proliferation_panel_gene_names, oncofetal_strict_gene_names,

)

from trufflepig.reference import (

    pan_cancer_expression, tumor_up_vs_matched_normal, heme_tumor_up_vs_matched_normal,

)
from .tumor_evidence import (
    TumorEvidenceScore,
    compute_tumor_evidence_score,
)
from .reasoning import DerivedFlags, compute_derived_flags, run_step0_rules


_MIN_REFERENCE_GENES = 2000

# Coordinate cell-cycle / mitosis panel — sourced from the public API
# in gene_sets_cancer so consumers can use the same panel for
# downstream scoring. Expanded in v4.28 from the 5-gene minimal set
# to the full 13-gene panel (CENPF median-fold 14×, FOXM1 6.3×,
# CDC20 6.1× were missing from the original set).
_PROLIFERATION_PANEL = tuple(proliferation_panel_gene_names())
_PROLIFERATION_HIGH_LOG2 = 4.5  # panel mean > this → "tumor-consistent"
_PROLIFERATION_QUIET_LOG2 = 3.5  # panel mean < this → proliferation quiet

# The margin by which the top HPA tissue must beat the top TCGA cohort
# correlation before we nudge toward "healthy-dominant". In combination
# with the proliferation panel.
_HPA_MARGIN_STRONG = 0.05
_HPA_MARGIN_WEAK = 0.02

# Lymphoid-tissue normals whose expression profile is structurally
# indistinguishable from lymphoid malignancy by bulk RNA — the TCGA
# DLBC reference is itself >90% lymph-node tissue + the malignant
# clone, so correlation alone can't separate them. When the top HPA
# match falls in this set AND the top TCGA match is a heme-lymphoid
# cancer, we always flag as "possibly-tumor" regardless of
# proliferation (germinal centers + bone marrow are normally
# cycling) and regardless of correlation margin.
_LYMPHOID_NORMAL_TISSUES = frozenset(
    {
        "lymph_node_nTPM",
        "spleen_nTPM",
        "thymus_nTPM",
        "bone_marrow_nTPM",
        "appendix_nTPM",
        "tonsil_nTPM",
    }
)
_HEME_LYMPHOID_TCGA_COHORTS = frozenset(
    {
        "DLBC_TPM",
        "LAML_TPM",
        "THYM_TPM",
    }
)

# Mesenchymal structural-ambiguity: the sarcoma analogue of the
# lymphoid case. Normal smooth muscle / adipose / skeletal muscle /
# endometrial myometrium share the same mesenchymal lineage as SARC
# tumors, so bulk-RNA correlation cannot distinguish them — a well-
# differentiated leiomyosarcoma looks like smooth muscle, a well-
# differentiated liposarcoma looks like adipose. The workflow prior
# is that users upload cancer samples (not random normal tissue), so
# when this override fires, downstream cancer-specific analysis
# proceeds — we just note the ambiguity so the clinician sees it.
_MESENCHYMAL_NORMAL_TISSUES = frozenset(
    {
        "smooth_muscle_nTPM",
        "adipose_tissue_nTPM",
        "skeletal_muscle_nTPM",
        "heart_muscle_nTPM",
        "endometrium_nTPM",  # uterine myometrium
        "cervix_nTPM",  # also smooth-muscle-rich
        "urinary_bladder_nTPM",  # smooth-muscle dominant
        "epididymis_nTPM",  # smooth muscle
    }
)
_MESENCHYMAL_SARC_TCGA_COHORTS = frozenset(
    {
        "SARC_TPM",
        "UCS_TPM",
    }
)

# CTA panel as independent tumor evidence. Cancer-testis antigens are
# epigenetically silenced in every somatic tissue except testis +
# placenta (+ in some cases ovary). Detection above threshold in a
# non-reproductive sample = positive tumor evidence that can override
# a correlation-based "healthy-dominant" call.
# The filtered-and-expressed CTA list (~257 genes) is shipped with
# pirlygenes; we load it once and cache.
_CTA_SYMBOLS_CACHE: frozenset[str] | None = None

# HPA tissues where CTA expression is physiological, not pathological.
# CTAs are defined by their testis-restricted normal expression, so
# ANY sample that matches testis / placenta / ovary is expected to
# express them — flagging CTA presence as tumor evidence in those
# regimes would false-positive on reproductive-tissue normals.
_CTA_NORMAL_TISSUES = frozenset(
    {
        "testis_nTPM",
        "placenta_nTPM",
        "ovary_nTPM",
    }
)

# Oncofetal / embryonic-stemness panel — complementary tumor evidence
# beyond the CTA panel. Every member of this panel must be near-zero
# in *any* adult somatic tissue (HPA check: median nTPM < 1 across
# non-reproductive tissues) so that detection above threshold in a
# non-reproductive sample is near-definitive tumor evidence.
#
# Explicitly excluded (initially included then removed after healthy-
# tissue false positives): SOX2 (normal neural progenitors — 46 TPM
# in GTEx brain), KLF4 (normal gut + smooth muscle — 67 TPM in GTEx
# smooth muscle), IGF2 (widely expressed — 45 TPM in smooth muscle),
# HMGA2 (adult expression in some tissues), HLA-G (placenta +
# immune-privileged niches). These expressed-in-normal markers fail
# the "near-zero in adult somatic" criterion and were dropping the
# panel's specificity.
_ONCOFETAL_STRICT = frozenset(oncofetal_strict_gene_names())
# Back-compat: keep _ONCOFETAL_LOOSE as an empty set so any caller
# that still references it (including the panel-union in
# _oncofetal_panel_signal below) doesn't break.
_ONCOFETAL_LOOSE: frozenset[str] = frozenset()

# HPA tissues where oncofetal expression is physiological.
_ONCOFETAL_NORMAL_TISSUES = frozenset(
    {
        "testis_nTPM",
        "placenta_nTPM",
        "ovary_nTPM",
        "liver_nTPM",  # AFP can be elevated in regenerating / fetal-pattern liver
    }
)

_ONCOFETAL_PER_GENE_MIN_TPM = 3.0
_ONCOFETAL_STRONG_COUNT = 2  # stricter than CTA since the panel is smaller

# Cancer-type-specific tumor-up-vs-matched-normal panel. Built from
# :func:`tumor_up_vs_matched_normal`. When Step-0's top TCGA match
# aligns with a sample expressing several of that cohort's private
# tumor-up markers, that's independent type-specific tumor evidence.
_TUMOR_UP_PANEL_PER_GENE_MIN_TPM = 3.0
_TUMOR_UP_PANEL_STRONG_COUNT = 2

# Thresholds for the CTA signal. Empirically tuned against the 6-
# sample battery: healthy tissues (GTEx brain / smooth muscle / spleen)
# show a few CTA hits at 1-5 TPM (PHF7, STKLD1, TEX14 — broadly
# expressed CTAs with relaxed tumor specificity). Tumors with real
# CTA re-expression (low-purity PRAD with DAZ3 ≈ 13 TPM, CRC with
# PIWIL1 ≈ 8, sarcoma with PAGE5 ≈ 1383) sit one or two orders of
# magnitude above the healthy-tissue noise floor.
# Per-gene 3 TPM + count ≥ 4 avoids the healthy-tissue CTA noise
# floor while preserving tumor detection at low purity.
_CTA_PER_GENE_MIN_TPM = 3.0
_CTA_STRONG_COUNT = 4
_CTA_STRONG_SUM_TPM = 30.0


@dataclass
class TissueCompositionSignal:
    """Step-0 coarse reading of tissue composition + cancer hint.

    ``cancer_hint`` is one of:

    - ``"tumor-consistent"`` — proliferation panel is clearly active
      OR TCGA cohort correlation exceeds HPA tissue correlation by
      more than the margin (typical primary tumor).
    - ``"possibly-tumor"`` — proliferation panel is not loud AND HPA
      correlation is close to TCGA; could be tumor or normal.
    - ``"healthy-dominant"`` — proliferation panel is quiet AND HPA
      correlation clearly exceeds TCGA.

    This is a coarse hint designed to be refined downstream, not a
    final classification.
    """

    top_normal_tissues: list[tuple[str, float]] = field(default_factory=list)
    top_tcga_cohorts: list[tuple[str, float]] = field(default_factory=list)
    proliferation_log2_mean: float = 0.0
    proliferation_genes_observed: int = 0
    cancer_hint: str = "tumor-consistent"
    n_reference_genes: int = 0
    verdict: str = ""
    # True when the top HPA / top TCGA pair is structurally
    # indistinguishable by bulk RNA (lymphoid normal vs heme-lymphoid
    # cancer). Downstream tumor-evidence gates should NOT suppress
    # the Step-0 banner in this case — the "purity" estimate on a
    # normal lymphoid sample is spurious anchor.
    structural_ambiguity: bool = False
    # CTA panel as positive tumor evidence (zero in non-reproductive
    # tissue; re-expression in cancer after epigenetic de-silencing).
    # ``cta_panel_sum_tpm`` is the total TPM across the curated
    # filtered-and-expressed CTA panel (~257 genes).
    cta_panel_sum_tpm: float = 0.0
    cta_count_above_1_tpm: int = 0
    cta_top_hits: list[tuple[str, float]] = field(default_factory=list)
    # Oncofetal / embryonic-stemness re-expression panel (AFP, LIN28A,
    # TPBG, PLAC1, NANOG, POU5F1, SOX2, KLF4, HLA-G, etc.). Independent
    # tumor evidence beyond CTAs; most informative for HCC / hepato-
    # blastoma / germ cell / SCLC / MBL / liposarcoma / neuroblastoma
    # where specific members pop ~100x vs normal.
    oncofetal_count_above_threshold: int = 0
    oncofetal_top_hits: list[tuple[str, float]] = field(default_factory=list)
    # Cancer-type-specific tumor-up-vs-matched-normal panel hits.
    # Keyed on the top Step-0 TCGA cohort — if that cohort has a
    # specific tumor-up panel (PRAD: OTOP1/ANKRD34C; OV: CLDN6;
    # KIRC: CD70, GAGE1; STAD: CT45A9/DAZ1) and the sample expresses
    # hits from it, that's cancer-type-specific positive evidence.
    type_specific_cohort: str = ""
    type_specific_hits: list[tuple[str, float]] = field(default_factory=list)
    # Unified tumor-evidence score across all channels. aggregate_score
    # ≥ 1.0 is tumor-consistent; < 0.3 + healthy correlation margin →
    # healthy-dominant; in-between → possibly-tumor. Holistic — no
    # single-threshold win/lose per channel; channels contribute
    # additively up to their saturation point.
    evidence: TumorEvidenceScore = field(default_factory=TumorEvidenceScore)
    # Ordered list of named rules that fired during the cancer-hint
    # decision. Human-readable; makes the reasoning auditable.
    reasoning_trace: list[str] = field(default_factory=list)

    def synthesis(self) -> str:
        """Single-spot narrative of all Step-0 evidence.

        Enumerates the top tissue / top cohort / all tumor-evidence
        channels + aggregate + the rule trace that drove the
        cancer_hint call. Intended for the analysis.md report + as
        a machine-readable reasoning audit.
        """
        parts = [self.summary_line()]
        if self.reasoning_trace:
            parts.append("Reasoning trace: " + " → ".join(self.reasoning_trace) + ".")
        parts.append(self.evidence.synthesis())
        return " ".join(parts)

    def summary_line(self) -> str:
        """One-sentence description suitable for a brief or summary.md."""
        if not self.top_normal_tissues:
            return (
                "Tissue-composition signal unavailable (reference overlap too small)."
            )

        def _fmt_tissue(name, rho):
            return f"{name.removesuffix('_nTPM').replace('_', ' ')} (ρ={rho:.2f})"

        def _fmt_cancer(name, rho):
            return f"{name.removesuffix('_TPM')} (ρ={rho:.2f})"

        tissues = ", ".join(_fmt_tissue(n, r) for n, r in self.top_normal_tissues[:3])
        cohorts = ", ".join(_fmt_cancer(n, r) for n, r in self.top_tcga_cohorts[:3])
        cta_clause = ""
        if self.cta_count_above_1_tpm > 0:
            top_cta = ", ".join(f"{s} {t:.0f}" for s, t in self.cta_top_hits[:3])
            cta_clause = (
                f" CTA panel: {self.cta_count_above_1_tpm} above 3 TPM "
                f"(sum {self.cta_panel_sum_tpm:.0f} TPM; top: {top_cta})."
            )
        oncofetal_clause = ""
        if self.oncofetal_count_above_threshold > 0:
            top_of = ", ".join(f"{s} {t:.0f}" for s, t in self.oncofetal_top_hits[:3])
            oncofetal_clause = (
                f" Oncofetal panel: {self.oncofetal_count_above_threshold} "
                f"hits (top: {top_of})."
            )
        return (
            f"Top normal-tissue matches: {tissues}. "
            f"Top TCGA cohorts: {cohorts}. "
            f"Proliferation panel {self.proliferation_log2_mean:.1f} log2-TPM "
            f"(of {self.proliferation_genes_observed}/{len(_PROLIFERATION_PANEL)} "
            f"genes observed).{cta_clause}{oncofetal_clause} "
            f"Hint: **{self.cancer_hint}**."
        )

    def brief_banner(
        self,
        purity: float | None = None,
        signature_score: float | None = None,
    ) -> str | None:
        """Terse banner for the brief — shown for every non-tumor-consistent hint.

        Propagates the Step-0 coarse reading forward so a clinician
        reading the brief sees the tissue context and the cancer-hint
        confidence up front rather than just the downstream TCGA label.

        When ``purity`` (Step 2) and/or ``signature_score`` (Step 1)
        are supplied, corroborating tumor evidence can suppress the
        banner: we don't want to shout "composition ambiguous" on a
        sample that has a solid lineage-matched TCGA cohort at moderate
        purity. The healthy-dominant banner has a higher bar to
        suppress than the possibly-tumor banner — it takes strong
        evidence (purity ≥ 0.5 AND signature ≥ 0.75) to override a
        confident Step-0 healthy call.
        """
        if self.cancer_hint == "tumor-consistent":
            return None
        if not self.top_normal_tissues:
            return None

        if (
            purity is not None or signature_score is not None
        ) and not self.structural_ambiguity:
            # Structural-ambiguity cases (lymphoid normal vs heme-
            # lymphoid cancer) always keep the banner — the purity
            # and signature estimates are spurious in that regime
            # because the reference itself is lymphoid tissue.
            p = float(purity) if purity is not None else 0.0
            s = float(signature_score) if signature_score is not None else 0.0
            if self.cancer_hint == "healthy-dominant":
                # Require strong corroborating tumor evidence to
                # override a confident healthy signal.
                if p >= 0.50 and s >= 0.75:
                    return None
            elif self.cancer_hint == "possibly-tumor":
                # Modest corroborating evidence suffices to suppress
                # the soft ambiguity banner — the downstream cancer
                # call is already well-supported.
                if p >= 0.30 or s >= 0.75:
                    return None

        tissue, rho = self.top_normal_tissues[0]
        tissue_name = tissue.removesuffix("_nTPM").replace("_", " ")
        cohort = (
            self.top_tcga_cohorts[0][0].removesuffix("_TPM")
            if self.top_tcga_cohorts
            else "—"
        )
        cohort_rho = self.top_tcga_cohorts[0][1] if self.top_tcga_cohorts else 0.0
        if self.cancer_hint == "healthy-dominant":
            return (
                f"**Step-0 hint: healthy tissue dominant.** Sample profile "
                f"correlates with normal **{tissue_name}** (ρ={rho:.2f}) more "
                f"than any TCGA cohort (best {cohort} ρ={cohort_rho:.2f}); "
                f"proliferation panel quiet "
                f"({self.proliferation_log2_mean:.1f} log2-TPM). Downstream "
                f"cancer-type call is soft-confidence."
            )
        # possibly-tumor
        if self.structural_ambiguity:
            # Distinguish lymphoid vs mesenchymal wording. The workflow
            # prior is that users upload cancer samples, so the banner
            # explicitly notes that downstream cancer-specific analysis
            # still proceeds — the ambiguity is context for the
            # clinician, not a blocker.
            tissue_lc = tissue.lower()
            lymphoid_tissues = (
                "lymph",
                "spleen",
                "thymus",
                "bone_marrow",
                "bone marrow",
                "tonsil",
                "appendix",
            )
            if any(tag in tissue_lc for tag in lymphoid_tissues):
                return (
                    f"**Step-0 hint: structural ambiguity (lymphoid).** Top "
                    f"normal match is **{tissue_name}** (ρ={rho:.2f}); top "
                    f"TCGA match is {cohort} (ρ={cohort_rho:.2f}). Normal "
                    f"lymphoid tissue and lymphoid malignancy are indist-"
                    f"inguishable by bulk-RNA correlation. Proceeding with "
                    f"the {cohort}-specific downstream analysis under the "
                    f"tumor-sample prior, but treat the cancer call and "
                    f"purity as soft-confidence — the purity estimate itself "
                    f"is unreliable in this regime."
                )
            return (
                f"**Step-0 hint: structural ambiguity (mesenchymal).** Top "
                f"normal match is **{tissue_name}** (ρ={rho:.2f}); top TCGA "
                f"match is {cohort} (ρ={cohort_rho:.2f}). Well-differentiated "
                f"sarcomas share a mesenchymal expression program with "
                f"normal smooth muscle / adipose / muscle / myometrium, so "
                f"bulk-RNA correlation cannot cleanly distinguish tumor "
                f"from tissue-of-origin. Proceeding with the {cohort}-"
                f"specific downstream analysis under the tumor-sample prior "
                f"— use the CTA / oncofetal / type-specific tumor-up panels "
                f"below as the primary tumor-evidence channels."
            )
        return (
            f"**Step-0 hint: composition ambiguous.** Top normal-tissue match "
            f"is **{tissue_name}** (ρ={rho:.2f}); top TCGA match is {cohort} "
            f"(ρ={cohort_rho:.2f}). Could be normal tissue or a low-purity "
            f"tumor — the downstream cancer call is soft-confidence pending "
            f"lineage / purity / therapy-axis evidence."
        )


# Back-compat alias: old name some callers used.
HealthyVsTumorResult = TissueCompositionSignal


def _extract_sample_tpm_by_symbol(df_expr: pd.DataFrame) -> dict[str, float]:
    tpm_col = next(
        (c for c in df_expr.columns if c.upper() == "TPM"),
        None,
    )
    if tpm_col is None:
        return {}
    gene_id_col = next(
        (
            c
            for c in df_expr.columns
            if c.lower()
            in ("gene_id", "ensembl_gene_id", "canonical_gene_id", "geneid")
        ),
        None,
    )
    gene_name_col = next(
        (
            c
            for c in df_expr.columns
            if c.lower()
            in (
                "gene_name",
                "canonical_gene_name",
                "gene_symbol",
                "symbol",
                "hugo_symbol",
            )
        ),
        None,
    )
    ref = pan_cancer_expression(technical_rna_normalize=True)[
        ["Ensembl_Gene_ID", "Symbol"]
    ].drop_duplicates(subset="Ensembl_Gene_ID")
    eid_to_symbol = dict(zip(ref["Ensembl_Gene_ID"], ref["Symbol"]))

    symbols = None
    if gene_id_col:
        bare_ids = df_expr[gene_id_col].astype(str).str.split(".", n=1).str[0]
        symbols = [eid_to_symbol.get(eid, "") for eid in bare_ids]
    if symbols is None or not any(symbols):
        if gene_name_col:
            symbols = df_expr[gene_name_col].astype(str).tolist()
    if symbols is None:
        return {}

    tpm_vals = df_expr[tpm_col].astype(float).tolist()
    out: dict[str, float] = {}
    for sym, tpm in zip(symbols, tpm_vals):
        if not sym or str(sym).strip() == "":
            continue
        out[str(sym)] = out.get(str(sym), 0.0) + float(tpm)
    return out


def _cta_symbols() -> frozenset[str]:
    """Cache the filtered-and-expressed CTA symbol set."""
    global _CTA_SYMBOLS_CACHE
    if _CTA_SYMBOLS_CACHE is None:
        _CTA_SYMBOLS_CACHE = frozenset(CTA_gene_names() or [])
    return _CTA_SYMBOLS_CACHE


def _oncofetal_panel_signal(
    sample_by_symbol: dict[str, float],
) -> tuple[int, list[tuple[str, float]]]:
    """Return (count above threshold, top-hits) for the oncofetal panel.

    Combines ``_ONCOFETAL_STRICT`` + ``_ONCOFETAL_LOOSE``. Count and
    hits are drawn from genes exceeding ``_ONCOFETAL_PER_GENE_MIN_TPM``.
    Strict-panel members score their own hit; loose-panel members
    also count but carry less specificity — the caller decides how
    strongly to weight them.
    """
    hits = []
    for sym in _ONCOFETAL_STRICT | _ONCOFETAL_LOOSE:
        tpm = float(sample_by_symbol.get(sym, 0.0))
        if tpm >= _ONCOFETAL_PER_GENE_MIN_TPM:
            hits.append((sym, tpm))
    hits.sort(key=lambda t: t[1], reverse=True)
    return len(hits), hits[:10]


def _type_specific_tumor_up_signal(
    sample_by_symbol: dict[str, float],
    top_tcga_code: str,
) -> list[tuple[str, float]]:
    """Return hits from the cancer-type-specific tumor-up panel.

    For the top-matching TCGA cohort, looks up the curated tumor-up
    vs matched-normal panel (genes ≥ 10-fold up in this cancer,
    low across all HPA normal tissues). Returns the sample's hits
    above ``_TUMOR_UP_PANEL_PER_GENE_MIN_TPM``.
    """
    if not top_tcga_code:
        return []
    panel = tumor_up_vs_matched_normal(cancer_code=top_tcga_code)
    if panel is None or panel.empty:
        # Fall back to the heme panel — DLBC / LAML are the covered
        # heme cohorts today.
        panel = heme_tumor_up_vs_matched_normal(cancer_code=top_tcga_code)
    if panel is None or panel.empty:
        return []
    hits = []
    for sym in panel["symbol"].astype(str):
        tpm = float(sample_by_symbol.get(sym, 0.0))
        if tpm >= _TUMOR_UP_PANEL_PER_GENE_MIN_TPM:
            hits.append((sym, tpm))
    hits.sort(key=lambda t: t[1], reverse=True)
    return hits[:10]


def _cta_panel_signal(
    sample_by_symbol: dict[str, float],
) -> tuple[float, int, list[tuple[str, float]]]:
    """Return (sum TPM across hits, count above threshold, top-hits) for CTAs.

    Only CTAs exceeding the per-gene threshold contribute to the sum —
    a handful of noise-level CTAs at 0.5-1 TPM shouldn't push the
    sample over the "strong-signal" line. The panel is the filtered-
    and-expressed CTA set (~257 genes): epigenetically silenced in
    somatic tissue, re-expressed in tumors.
    """
    cta_set = _cta_symbols()
    hits = []
    total = 0.0
    for sym in cta_set:
        tpm = float(sample_by_symbol.get(sym, 0.0))
        if tpm >= _CTA_PER_GENE_MIN_TPM:
            total += tpm
            hits.append((sym, tpm))
    hits.sort(key=lambda t: t[1], reverse=True)
    return total, len(hits), hits[:10]


def assess_tissue_composition(df_expr: pd.DataFrame) -> TissueCompositionSignal:
    """Race the sample against HPA-normal-tissue and TCGA-cohort columns.

    Returns the top-3 matches on each side + proliferation-panel
    geomean + a coarse cancer-hint call. Intended as the first step
    in a coarse-to-fine pipeline: it gives downstream steps a
    tissue-composition prior without committing to healthy / cancer.
    """
    sample_by_symbol = _extract_sample_tpm_by_symbol(df_expr)
    ref = (
        pan_cancer_expression(technical_rna_normalize=True)
        .drop_duplicates(subset="Symbol")
        .set_index("Symbol")
    )

    hpa_cols = [c for c in ref.columns if c.endswith("_nTPM")]
    tcga_cols = [c for c in ref.columns if c.endswith("_TPM")]

    shared_symbols = [s for s in sample_by_symbol if s in ref.index]
    n_overlap = len(shared_symbols)

    prolif_tpms = [
        sample_by_symbol.get(g, 0.0)
        for g in _PROLIFERATION_PANEL
        if g in sample_by_symbol
    ]
    if prolif_tpms:
        prolif_log2 = float(np.mean(np.log2(np.array(prolif_tpms) + 1.0)))
    else:
        prolif_log2 = 0.0
    n_prolif_obs = len(prolif_tpms)

    cta_sum_tpm, cta_count, cta_top_hits = _cta_panel_signal(sample_by_symbol)
    oncofetal_count, oncofetal_top_hits = _oncofetal_panel_signal(sample_by_symbol)
    # Compute top-TCGA code for the type-specific tumor-up lookup.
    # Done before the full top_matches pass because it only needs
    # the best match — cheaper than the full rank.
    _top_tcga_for_panel = ""
    if shared_symbols and len(shared_symbols) >= _MIN_REFERENCE_GENES:
        # We'll fill this properly below after computing top_tcga;
        # delay the type-specific lookup until after.
        pass

    if n_overlap < _MIN_REFERENCE_GENES:
        return TissueCompositionSignal(
            proliferation_log2_mean=prolif_log2,
            proliferation_genes_observed=n_prolif_obs,
            cancer_hint="tumor-consistent",
            n_reference_genes=n_overlap,
            verdict=(
                f"Insufficient reference overlap ({n_overlap} < "
                f"{_MIN_REFERENCE_GENES} genes); tissue-composition signal "
                f"unavailable — defer to downstream cancer-type inference."
            ),
            cta_panel_sum_tpm=cta_sum_tpm,
            cta_count_above_1_tpm=cta_count,
            cta_top_hits=cta_top_hits,
            oncofetal_count_above_threshold=oncofetal_count,
            oncofetal_top_hits=oncofetal_top_hits,
        )

    sample_log = np.log2(np.array([sample_by_symbol[s] + 1.0 for s in shared_symbols]))

    def _top_matches(cols: list[str], k: int = 3) -> list[tuple[str, float]]:
        scored = []
        for col in cols:
            ref_vals = ref.loc[shared_symbols, col].astype(float).to_numpy()
            if not np.isfinite(ref_vals).any() or float(np.nanmax(ref_vals)) <= 0:
                continue
            ref_log = np.log2(np.nan_to_num(ref_vals) + 1.0)
            rho = float(_spearman_rho(sample_log, ref_log))
            scored.append((col, rho))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    top_normal = _top_matches(hpa_cols, k=3)
    top_tcga = _top_matches(tcga_cols, k=3)

    # Cancer-type-specific tumor-up panel: check the top TCGA cohort's
    # tumor-vs-matched-normal private markers against the sample.
    top_tcga_code = top_tcga[0][0].removesuffix("_TPM") if top_tcga else ""
    type_specific_hits = _type_specific_tumor_up_signal(
        sample_by_symbol,
        top_tcga_code,
    )

    best_normal_rho = top_normal[0][1] if top_normal else 0.0
    best_tcga_rho = top_tcga[0][1] if top_tcga else 0.0
    margin = best_normal_rho - best_tcga_rho

    # Structural-ambiguity override: when the top HPA match is a
    # lymphoid normal AND the top TCGA match is a heme-lymphoid
    # cancer, bulk-RNA correlation cannot distinguish them. Always
    # flag as possibly-tumor so the downstream reader sees the
    # ambiguity rather than a false tumor-consistent call.
    top_hpa_name = top_normal[0][0] if top_normal else ""
    top_tcga_name = top_tcga[0][0] if top_tcga else ""
    lymphoid_ambiguity = (
        top_hpa_name in _LYMPHOID_NORMAL_TISSUES
        and top_tcga_name in _HEME_LYMPHOID_TCGA_COHORTS
    )
    mesenchymal_ambiguity = (
        top_hpa_name in _MESENCHYMAL_NORMAL_TISSUES
        and top_tcga_name in _MESENCHYMAL_SARC_TCGA_COHORTS
    )

    # CTA / oncofetal / type-specific strong + soft signals are now
    # evaluated inside the standalone rule functions
    # (pirlygenes.reasoning) so they can be tested in isolation. The
    # physiological-guard tissue sets (_CTA_NORMAL_TISSUES,
    # _ONCOFETAL_NORMAL_TISSUES) still inform the counts pre-computed
    # above — a sample whose top HPA match is testis / placenta /
    # ovary / liver has its CTA / oncofetal counts zeroed below
    # rather than flowing a spurious "tumor evidence" signal into the
    # rule runner.
    if top_hpa_name in _CTA_NORMAL_TISSUES:
        cta_count = 0
        cta_sum_tpm = 0.0
        cta_top_hits = []
    if top_hpa_name in _ONCOFETAL_NORMAL_TISSUES:
        oncofetal_count = 0
        oncofetal_top_hits = []

    # Unified tumor-evidence score across all channels.
    evidence = compute_tumor_evidence_score(
        sample_by_symbol=sample_by_symbol,
        cta_count=cta_count,
        oncofetal_count=oncofetal_count,
        type_specific_count=len(type_specific_hits),
        proliferation_log2_mean=prolif_log2,
    )

    # ---- Ordered rule-list reasoning ----
    # Build a lightweight signal carrier holding every field the
    # standalone rules need (rules are in pirlygenes.reasoning and
    # are testable in isolation — see docs/step0-reasoning.md).
    tmp = TissueCompositionSignal(
        top_normal_tissues=top_normal,
        top_tcga_cohorts=top_tcga,
        proliferation_log2_mean=prolif_log2,
        proliferation_genes_observed=n_prolif_obs,
        cancer_hint="tumor-consistent",
        n_reference_genes=n_overlap,
        verdict="",
        structural_ambiguity=False,
        cta_panel_sum_tpm=cta_sum_tpm,
        cta_count_above_1_tpm=cta_count,
        cta_top_hits=cta_top_hits,
        oncofetal_count_above_threshold=oncofetal_count,
        oncofetal_top_hits=oncofetal_top_hits,
        type_specific_cohort=top_tcga_code,
        type_specific_hits=type_specific_hits,
        evidence=evidence,
    )
    # Derive the Step-0 flags (lymphoid/mesenchymal ambiguity, strong/
    # soft tumor-evidence booleans, correlation margin) from the signal
    # fields + the ambiguity booleans that were computed upstream from
    # the raw tissue correlations.
    base_flags = compute_derived_flags(tmp)
    flags = DerivedFlags(
        lymphoid_ambiguity=lymphoid_ambiguity,
        mesenchymal_ambiguity=mesenchymal_ambiguity,
        cta_strong=base_flags.cta_strong,
        oncofetal_strong=base_flags.oncofetal_strong,
        type_specific_strong=base_flags.type_specific_strong,
        cta_soft=base_flags.cta_soft,
        oncofetal_soft=base_flags.oncofetal_soft,
        type_specific_soft=base_flags.type_specific_soft,
        correlation_margin=base_flags.correlation_margin,
    )

    outcome, reasoning_trace = run_step0_rules(tmp, flags)
    cancer_hint = outcome.hint
    structural_ambiguity = outcome.structural
    if outcome.rationale:
        reasoning_trace = [f"{outcome.rule_name}[{outcome.rationale}]"]

    if cancer_hint == "healthy-dominant":
        verdict = (
            f"Healthy-dominant: best HPA "
            f"{top_normal[0][0].removesuffix('_nTPM')} ρ={best_normal_rho:.3f} "
            f"beats best TCGA {top_tcga[0][0].removesuffix('_TPM') if top_tcga else '-'} "
            f"ρ={best_tcga_rho:.3f} by {margin:+.3f}; proliferation "
            f"{prolif_log2:.1f} log2-TPM quiet."
        )
    elif cancer_hint == "possibly-tumor":
        verdict = (
            f"Composition ambiguous: HPA ρ={best_normal_rho:.3f} vs "
            f"TCGA ρ={best_tcga_rho:.3f} (margin {margin:+.3f}); "
            f"proliferation {prolif_log2:.1f} log2-TPM moderate."
        )
    else:
        verdict = (
            f"Tumor-consistent: proliferation panel {prolif_log2:.1f} log2-TPM "
            f"or TCGA ρ={best_tcga_rho:.3f} dominant vs HPA ρ={best_normal_rho:.3f}."
        )

    return TissueCompositionSignal(
        top_normal_tissues=top_normal,
        top_tcga_cohorts=top_tcga,
        proliferation_log2_mean=prolif_log2,
        proliferation_genes_observed=n_prolif_obs,
        cancer_hint=cancer_hint,
        n_reference_genes=n_overlap,
        verdict=verdict,
        structural_ambiguity=structural_ambiguity,
        cta_panel_sum_tpm=cta_sum_tpm,
        cta_count_above_1_tpm=cta_count,
        cta_top_hits=cta_top_hits,
        oncofetal_count_above_threshold=oncofetal_count,
        oncofetal_top_hits=oncofetal_top_hits,
        type_specific_cohort=top_tcga_code,
        type_specific_hits=type_specific_hits,
        evidence=evidence,
        reasoning_trace=reasoning_trace,
    )


# Back-compat alias: callers wrote ``assess_healthy_vs_tumor`` before
# the rename. Keep the old entry point working.
def assess_healthy_vs_tumor(
    df_expr: pd.DataFrame,
    *_args,
    **_kwargs,
) -> TissueCompositionSignal:
    return assess_tissue_composition(df_expr)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(x, y, nan_policy="omit").statistic)
    except Exception:
        return _pearson_on_ranks(x, y)


def _pearson_on_ranks(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    rx = _rank(x[mask])
    ry = _rank(y[mask])
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(v) + 1)
    return ranks
