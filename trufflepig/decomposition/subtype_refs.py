# Licensed under the Apache License, Version 2.0

"""Post-hoc subtype-aware TME reference refinement (issue #56).

The decomposition engine anchors each non-tumor compartment to HPA
generic cell-type nTPM vectors:

- ``fibroblast`` → HPA "Fibroblasts" (primary-culture fibroblast)
- ``myeloid`` → HPA macrophage / monocyte / granulocyte mix

Both references under-represent the **tumor-activated** states that
dominate the TME in a real biopsy:

- Cancer-associated fibroblasts (CAFs) upregulate FAP, POSTN, S100A4,
  TNC, THY1, PDGFRA/B, matrix genes (COL1A1/2, DCN, SPARC) far above
  what a generic fibroblast reference expresses.
- Tumor-associated macrophages (TAMs — M2-polarized) upregulate CD163,
  MRC1 (CD206), LYVE1, STAB1, MARCO, VSIG4, TREM2 relative to
  infiltrating monocytes / generic macrophages.

Consequence in ``estimate_tumor_expression_ranges``: the per-gene TME
subtraction
``fibroblast_fraction × HPA_Fibroblasts[gene]`` *under-subtracts*
CAF-marker genes; the residual lands in ``tumor_tpm`` and flags
unambiguously-stromal genes as tumor-expressed therapy targets.

This module carries the curated marker panels + fold-over-baseline
values used by :func:`refine_tme_per_gene` to post-hoc swap the
per-gene reference for marker genes. The NNLS solve is left alone —
only marker-gene TME contributions are adjusted, and ``tumor_tpm`` is
recomputed from the refined contributions.

Gene marker panels are drawn from:

- CAF: Dominguez 2020 Cell *Single-Cell RNA Sequencing Reveals Stromal
  Evolution into LRRC15+ Myofibroblasts as a Determinant of Patient
  Response to Cancer Immunotherapy* + Kieffer 2020 (breast CAF atlas)
  + Elyada 2019 (PDAC CAF atlas).
- TAM: Cheng 2021 Cell *A pan-cancer single-cell transcriptional atlas
  of tumor infiltrating myeloid cells* + Azizi 2018 (breast).

Fold-over-baseline values are median across those cohorts, rounded to
one decimal. They are approximations — the point is directional
(swap the reference away from "generic" toward "tumor-activated") not
millipercent calibration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple


# ---------- CAF markers — canonical stromal reprogramming signals ----------

# Gene → expected multiplicative fold in tumor-associated state vs the
# HPA generic-fibroblast baseline. Derived as median log2-FC across
# Dominguez 2020 + Kieffer 2020 + Elyada 2019, exponentiated back to
# linear fold. Core ECM / desmoplasia markers; NOT exhaustive.
CAF_MARKER_FOLDS: Dict[str, float] = {
    "FAP": 10.0,  # Fibroblast activation protein — canonical CAF marker
    "POSTN": 8.0,  # Periostin — desmoplastic stroma
    "S100A4": 4.0,  # FSP1 — activated fibroblast
    "TNC": 5.0,  # Tenascin-C — desmoplastic ECM
    "THY1": 2.5,  # CD90 — CAF subtype marker
    "PDGFRA": 3.5,  # Myofibroblast lineage
    "PDGFRB": 4.0,  # Pericyte-like CAF
    "COL1A1": 5.0,  # Fibrotic ECM
    "COL1A2": 5.0,
    "COL3A1": 4.0,
    "DCN": 3.0,  # Decorin — small leucine-rich proteoglycan
    "SPARC": 4.0,  # Osteonectin — matricellular, CAF-elevated
    "A2M": 3.0,  # α2-macroglobulin — stromal abundance
    "VIM": 2.0,  # Vimentin — mesenchymal lineage marker
    "LUM": 3.0,  # Lumican — CAF ECM
}


# ---------- TAM markers — M2-polarized tumor-infiltrating macrophages ----------

TAM_MARKER_FOLDS: Dict[str, float] = {
    "CD163": 5.0,  # M2 scavenger receptor
    "MRC1": 7.0,  # CD206 — M2 mannose receptor
    "LYVE1": 6.0,  # Lymphatic/tissue-resident macrophage
    "STAB1": 4.0,  # Stabilin-1 — TAM scavenger
    "MARCO": 5.0,  # M2 scavenger
    "VSIG4": 4.0,  # CRIg — immunosuppressive TAM
    "TREM2": 6.0,  # Lipid-associated TAM
    # Selenoprotein P. HGNC renamed SEPP1 → SELENOP; the bundled
    # reference data uses the new symbol. Keep both so older quant
    # files that still emit SEPP1 don't silently no-op.
    "SELENOP": 3.0,  # Selenoprotein P — M2 polarization
    "SEPP1": 3.0,  # (legacy alias for SELENOP)
    "C1QA": 3.5,  # Complement — TAM-elevated
    "C1QB": 3.5,
    "C1QC": 3.5,
    "APOE": 3.0,  # TAM lipid-handling program
}


# ---------- Extended TME activated-state panels (#58) ----------

# These compartments live alongside CAF / TAM in the same refinement
# mechanism. Each panel targets a template compartment where the HPA
# generic reference under-represents the tumor-activated state.


# Exhausted / tumor-infiltrating T cells. Zheng 2021 pan-cancer TIL
# atlas; Oliveira 2021 TCR/T-cell atlas. Checkpoint-receptor biology —
# these are the genes ICI (anti-PD-1 / CTLA4 / LAG3) targets, and
# their over-attribution to tumor is a major UX failure on any
# lymphocyte-rich sample.
EXHAUSTED_T_MARKER_FOLDS: Dict[str, float] = {
    "PDCD1": 8.0,  # PD-1 — canonical exhaustion marker
    "CTLA4": 6.0,  # CTLA-4 — checkpoint
    "LAG3": 6.0,  # LAG-3 — co-expressed with PD-1
    "HAVCR2": 6.0,  # TIM-3
    "TIGIT": 5.0,  # TIGIT checkpoint
    "TOX": 5.0,  # Exhaustion transcription factor
    "ENTPD1": 4.0,  # CD39 — tissue-resident exhausted Treg
    "ITGAE": 4.0,  # CD103 — Trm / TIL marker
    "LAYN": 4.0,  # Layilin — terminal exhaustion
    "CXCL13": 5.0,  # Follicular / TLS-T cell chemokine
}


# Tumor endothelium — angiogenic / tip-cell / venule programs elevated
# in solid tumors. Goveia 2020 pan-cancer lung EC atlas; Kalucka 2020
# pan-tissue EC atlas. Anti-angiogenic ADC / TKI biology.
TUMOR_ENDOTHELIUM_MARKER_FOLDS: Dict[str, float] = {
    "DLL4": 8.0,  # Notch ligand — tip-cell / angiogenic EC
    "ESM1": 7.0,  # Endocan — tip-cell marker
    "ANGPT2": 6.0,  # Angiogenic signaling
    "PLVAP": 5.0,  # Fenestrated tumor venule
    "ACKR1": 4.0,  # Post-capillary venule (Duffy)
    "APLN": 5.0,  # Apelin — tumor-EC-specific ligand
    "CXCR4": 4.0,  # Tip-cell migration
    "INSR": 3.0,  # Metabolic tumor-EC program
    "KDR": 4.0,  # VEGFR2 — angiogenic
    "FLT1": 4.0,  # VEGFR1 — angiogenic
}


# Myeloid-derived suppressor cells — distinct from TAM by lineage
# (granulocytic / monocytic MDSC). Alshetaiwi 2020 MDSC atlas;
# Zilionis 2019 lung TAN/TAM atlas. Lives on the same ``myeloid``
# compartment as TAM; genes don't overlap meaningfully.
MDSC_MARKER_FOLDS: Dict[str, float] = {
    "ARG1": 8.0,  # Arginase-1 — immunosuppressive; PMN-MDSC
    "S100A8": 5.0,  # Calprotectin subunit — MDSC / neutrophil
    "S100A9": 5.0,  # Calprotectin subunit
    "CEBPB": 3.0,  # Transcription factor — MDSC polarization
    "FCGR3B": 4.0,  # CD16b — granulocytic MDSC / neutrophil
    "MPO": 4.0,  # Myeloperoxidase — neutrophil lineage
    "ELANE": 4.0,  # Neutrophil elastase
    "CAMP": 4.0,  # Cathelicidin — neutrophil
}


# Tertiary lymphoid structure B cells — germinal-center B cells
# organised into ectopic lymphoid follicles in the tumor. Meylan 2022
# RCC TLS; Patil 2022 pan-cancer TLS atlas. Positive ICI prognostic.
TLS_B_MARKER_FOLDS: Dict[str, float] = {
    "CXCR5": 5.0,  # Germinal-center homing
    "BCL6": 4.0,  # Germinal-center master TF
    "AICDA": 6.0,  # Activation-induced deaminase — class-switch
    "MS4A1": 3.0,  # CD20 — B-cell identity (scaled for TLS density)
    "RGS13": 4.0,  # GC-B signature
    "STMN1": 3.0,  # Proliferative GC-B
}


# Tumor-infiltrating plasma cells. Cheng 2021 pan-cancer B/plasma
# atlas. IgG class-switched patterns distinct from marrow plasma.
TI_PLASMA_MARKER_FOLDS: Dict[str, float] = {
    "MZB1": 4.0,  # Plasma cell maturation — elevated in TI plasma
    "JCHAIN": 3.0,  # Polymeric Ig joining chain
    "IGHG1": 5.0,  # IgG1 heavy chain (class-switched)
    "IGHG3": 4.0,  # IgG3 heavy chain
    "IGHA1": 4.0,  # IgA1 heavy chain
    "XBP1": 3.0,  # Plasma-cell UPR / ER stress
}


# ---------- Panel registry ----------

# Each row: (label, NNLS-compartment-name, marker-fold dict). Order
# matters for provenance (first panel whose compartment has signal on
# a gene wins), but canonical markers do not overlap across panels —
# a TAM-marker like CD163 doesn't appear in MDSC or exhausted-T.
# Add a new compartment by appending a row here + its ``_MARKER_FOLDS``
# dict above; no other code needs to change.
PANELS: List[Tuple[str, str, Dict[str, float]]] = [
    ("CAF", "fibroblast", CAF_MARKER_FOLDS),
    ("TAM", "myeloid", TAM_MARKER_FOLDS),
    ("MDSC", "myeloid", MDSC_MARKER_FOLDS),
    ("exhausted_T", "T_cell", EXHAUSTED_T_MARKER_FOLDS),
    ("tumor_endothelium", "endothelial", TUMOR_ENDOTHELIUM_MARKER_FOLDS),
    ("TLS_B", "B_cell", TLS_B_MARKER_FOLDS),
    ("TI_plasma", "plasma", TI_PLASMA_MARKER_FOLDS),
]


def caf_markers() -> List[str]:
    """Return the canonical CAF marker-gene list (#56)."""
    return list(CAF_MARKER_FOLDS.keys())


def tam_markers() -> List[str]:
    """Return the canonical TAM marker-gene list (#56)."""
    return list(TAM_MARKER_FOLDS.keys())


def panel_markers(label: str) -> List[str]:
    """Return the marker-gene list for any registered panel (#58)."""
    for panel_label, _compartment, folds in PANELS:
        if panel_label == label:
            return list(folds.keys())
    return []


def panel_labels() -> List[str]:
    """List every registered panel label (#58)."""
    return [label for label, _comp, _folds in PANELS]


def refine_tme_per_gene(
    tme_bg_tpm_by_symbol: Mapping[str, float],
    per_compartment_tpm_by_symbol: Mapping[str, Mapping[str, float]] | None,
    sample_tpm_by_symbol: Mapping[str, float],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """Post-hoc swap the per-gene TME reference for marker genes (#56).

    For every gene on the CAF (respectively TAM) marker panel that has
    a non-zero ``fibroblast`` (respectively ``myeloid``) compartment
    contribution in the existing decomposition, scale that
    contribution up to the tumor-activated fold-over-baseline. Clamp
    the refined TME contribution at the observed TPM so we never
    subtract more than the sample carries.

    The NNLS solve is *not* re-run — this is strictly a per-gene
    reference swap. Non-marker genes pass through byte-identically.

    Parameters
    ----------
    tme_bg_tpm_by_symbol
        Per-gene expected non-tumor TPM built from the NNLS fit
        (``estimate_tumor_expression_ranges`` line ~566).
    per_compartment_tpm_by_symbol
        Per-gene ``{compartment: tpm}`` breakdown used for #108
        attribution columns. ``None`` allowed — we then fall back to
        scaling the aggregate TME by the marker's fold, which is
        approximate but preserves directionality.
    sample_tpm_by_symbol
        Observed sample TPMs by gene symbol.

    Returns
    -------
    refined_tme_by_symbol
        New ``{symbol: refined_tme_tpm}`` dict. Identical to the input
        for genes not on a marker panel.
    provenance
        ``{symbol: {"before": tme_before, "after": tme_after,
        "subtype": "CAF"|"TAM", "fold": fold}}`` — only populated for
        genes actually refined. Consumers emit this into the TSV
        ``subtype_refined`` + ``tumor_tpm_before_subtype_refinement``
        columns and the ``subtype-attribution-*.png`` plots.
    """
    refined: Dict[str, float] = dict(tme_bg_tpm_by_symbol)
    # Provenance is heterogeneous: ``before`` / ``after`` / ``fold``
    # are floats, ``subtype`` is a string label (CAF / TAM / …). Any-
    # typed values let future callers add structured fields without
    # widening every downstream signature. Consumers today only read
    # the fields this module populates, via ``.get(name)`` on dicts.
    provenance: Dict[str, Dict[str, Any]] = {}

    # Iterate every registered panel (#58). Canonical markers don't
    # overlap across panels, but if a gene IS shared, first-match-wins
    # by registration order.
    for subtype_label, compartment, marker_folds in PANELS:
        for gene, fold in marker_folds.items():
            if gene not in refined:
                continue
            if gene in provenance:
                # First panel to refine this gene wins — keeps the
                # provenance single-source and avoids compounding folds
                # on genes that appear on multiple panels.
                continue
            tme_before = float(refined[gene])
            observed = float(sample_tpm_by_symbol.get(gene, 0.0))
            if observed <= 0:
                continue

            if per_compartment_tpm_by_symbol is not None:
                per_comp = per_compartment_tpm_by_symbol.get(gene, {}) or {}
                compartment_tpm = float(per_comp.get(compartment, 0.0))
                if compartment_tpm <= 0:
                    # This compartment didn't land any signal on this
                    # gene — nothing to refine.
                    continue
                # Replace this compartment's contribution with its
                # tumor-activated fold-over-baseline. The other
                # compartments' TME contributions pass through.
                scaled = compartment_tpm * fold
                tme_after = tme_before - compartment_tpm + scaled
            else:
                # Fallback: no per-compartment breakdown. Scale the
                # aggregate TME by the fold.
                compartment_tpm = tme_before
                tme_after = tme_before * fold

            # Clamp so the refined TME never exceeds what's observed —
            # over-refinement would push ``tumor_tpm`` negative, which
            # the caller silently clips to zero.
            tme_after = min(tme_after, observed)

            if tme_after > tme_before:
                if per_compartment_tpm_by_symbol is None:
                    compartment_after = tme_after
                else:
                    compartment_after = compartment_tpm + (tme_after - tme_before)
                refined[gene] = tme_after
                provenance[gene] = {
                    "before": tme_before,
                    "after": tme_after,
                    "subtype": subtype_label,
                    "compartment": compartment,
                    "compartment_before": compartment_tpm,
                    "compartment_after": compartment_after,
                    "fold": float(fold),
                }

    return refined, provenance


def partition_compartment(
    sample_tpm_by_symbol: Mapping[str, float],
    compartment_fraction: float,
    marker_folds: Mapping[str, float],
) -> Tuple[float, float]:
    """Display-only split of a compartment fraction into subtype + generic.

    .. note::
       Not yet consumed in-tree — intended for the composition-report
       display work planned under #58 (extended TME refinement). The
       scoring heuristic is provisional; calibration against
       ground-truth CAF / TAM fractions is deferred until a cohort
       with orthogonal (single-cell) compartment estimates is
       available.

    Scores the sample's relative activation of the compartment's
    subtype-biased markers and partitions the fitted compartment
    fraction into ``(subtype_fraction, generic_fraction)`` accordingly.
    NNLS is *not* re-run — this is for composition-report display only.

    Marker activation is a simple ``Σ log1p(sample_tpm * fold_weight)``
    so a gene with a high activation-fold contributes more evidence
    toward the subtype side. Genes absent from the sample contribute
    zero on both sides.

    Returns (subtype_fraction, generic_fraction) summing to
    ``compartment_fraction``.
    """
    import math

    subtype_score = 0.0
    generic_score = 0.0
    for gene, fold in marker_folds.items():
        sample_val = float(sample_tpm_by_symbol.get(gene, 0.0))
        if sample_val <= 0:
            continue
        # Subtype weight: marker present at any level → adds to the
        # activated score; fold amplifies.
        subtype_score += math.log1p(sample_val * (fold - 1.0))
        # Generic weight: marker present at any level at baseline
        # strength.
        generic_score += math.log1p(sample_val)

    total = subtype_score + generic_score
    if total <= 0:
        return (0.0, compartment_fraction)

    subtype_frac = compartment_fraction * (subtype_score / total)
    generic_frac = compartment_fraction - subtype_frac
    return (subtype_frac, generic_frac)
