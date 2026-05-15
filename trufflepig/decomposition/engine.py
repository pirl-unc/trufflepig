# Licensed under the Apache License, Version 2.0

"""Broad-compartment sample decomposition with external purity anchoring.

The decomposition deliberately avoids unsupported fine-grained immune splits.
Tumor purity is estimated separately, then the non-tumor fraction is
distributed across broad, reference-supported compartments using weighted
NNLS on component-enriched marker genes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .panels import estimate_lineage_tumor_fraction
from .signature import build_signature_matrix, get_component_markers
from .templates import (
    EPITHELIAL_MATCHED_NORMAL_TISSUE,
    TEMPLATES,
    _detect_optional_compartments,
    get_template_components,
    get_template_extra_components,
    get_template_host_tissues,
    matched_normal_component,
)
from pirlygenes.gene_sets_cancer import (
    housekeeping_gene_ids, is_extended_housekeeping_symbol,
)
from trufflepig.reference import (
    pan_cancer_expression,
)
from ..tumor_purity import rank_cancer_type_candidates, _score_host_tissues


# Marker symbols that must never be auto-selected by the specificity-based
# rule, even when their HPA nTPM in a target component is high (issue #31).
#
# MHC-II / shared APC genes are expressed on B cells *and* macrophages,
# dendritic cells, monocytes, and activated T cells. Auto-selecting them
# as "B_cell markers" lets the NNLS re-route MHC-II signal into T_cell
# and myeloid columns, distorting per-compartment fractions. The curated
# ``COMPONENT_MARKERS`` in signature.py remains the source of truth for
# B-cell-specific markers (MS4A1, CD79A, CD79B, CD19, BANK1).
_AUTO_MARKER_EXCLUDED_SYMBOLS = frozenset(
    {
        "CD74",
        "HLA-DRA",
        "HLA-DRB1",
        "HLA-DRB5",
        "HLA-DPA1",
        "HLA-DPB1",
        "HLA-DQA1",
        "HLA-DQB1",
        "HLA-DMA",
        "HLA-DMB",
        "HLA-DOA",
        "HLA-DOB",
    }
)


def _is_excluded_auto_marker(symbol: str) -> bool:
    """Return True when the symbol must not be auto-selected as a
    decomposition-component discriminator.

    Combines three exclusion sources:

    - The curated MHC-II / shared-APC blacklist above (#31).
    - The extended housekeeping panel from ``gene_sets_cancer``
      (``scope="markers"``; #60) — universal exclusion of ribosomal,
      mitochondrial, translation/splicing, rearranged IG/TR, proteasome
      and classic housekeeping symbols that can't discriminate cell
      types. This subsumes the old hard-coded RPL/RPS-prefix filter.
    """
    if not isinstance(symbol, str):
        return False
    if symbol in _AUTO_MARKER_EXCLUDED_SYMBOLS:
        return True
    return is_extended_housekeeping_symbol(symbol, scope="markers")


DECOMPOSITION_PARAMETERS = {
    # ── Sample mode routing ──────────────────────────────────────────────
    # Controls which template set is evaluated.
    "sample_mode": {
        "default": "auto",
        # TCGA codes that trigger heme-specific templates when mode is "auto"
        "heme_codes": ["DLBC", "LAML"],
        "tumor_contexts": ["auto", "primary", "met"],
        # Normalised site_hint → template mapping (aliases resolve common names)
        "site_hint_templates": {
            "adrenal": "met_adrenal",
            "adrenal_gland": "met_adrenal",
            "bone": "met_bone",
            "bone_marrow": "met_bone",
            "brain": "met_brain",
            "cerebral_cortex": "met_brain",
            "liver": "met_liver",
            "lung": "met_lung",
            "lymph": "met_lymph_node",
            "lymph_node": "met_lymph_node",
            "node": "met_lymph_node",
            "peritoneal": "met_peritoneal",
            "peritoneum": "met_peritoneal",
            "retroperitoneal": "met_soft_tissue",
            "skin": "met_skin",
            "soft_tissue": "met_soft_tissue",
            "soft-tissue": "met_soft_tissue",
            "smooth_muscle": "met_soft_tissue",
        },
        "template_sets": {
            "solid": [
                "solid_primary",
                "met_lymph_node",
                "met_soft_tissue",
                "met_liver",
                "met_lung",
                "met_bone",
                "met_brain",
                "met_peritoneal",
                "met_adrenal",
                "met_skin",
            ],
            "heme": ["heme_nodal", "heme_blood", "heme_marrow"],
            "pure": ["pure_population"],
        },
    },
    # ── NNLS solver ──────────────────────────────────────────────────────
    "nnls": {
        # Soft sum-to-one penalty added as an extra row in the augmented
        # system.  Higher values push component fractions toward summing to 1;
        # 8.0 is large enough to enforce the constraint tightly while still
        # allowing the solver room to fit genuine biological deviations.
        "sum_to_one_weight": 8.0,
        # Light ridge (L2) penalty on the solution vector.  Prevents any
        # single component from absorbing all expression when the signature
        # matrix is near-collinear (e.g. T_cell vs NK).
        "l2_penalty": 0.05,
    },
    # ── Marker gene selection ────────────────────────────────────────────
    "marker_selection": {
        # Max marker genes kept per component.  More markers stabilise the fit
        # but increase the chance of including ambiguous genes.
        "top_n_per_component": 12,
        # HK-normalised expression floor used when the primary marker
        # selection finds too few rows and falls back to all expressed genes.
        "fallback_expression_floor": 0.05,
    },
    # ── Matched-normal lineage override ────────────────────────────────
    "lineage_override": {
        # The matched-normal panel is allowed to correct an over-high
        # signature purity downward (e.g. pure normal prostate run as
        # PRAD), but large *upward* jumps are a red flag that the panel
        # got contaminated by non-lineage genes. Reject those jumps.
        "max_upward_delta": 0.25,
        "max_upward_ratio": 2.0,
        "ratio_floor": 0.05,
    },
    # ── Template / hypothesis scoring ────────────────────────────────────
    # Final score = fit_score × (fit_score_base + fit_score_gain × cancer_support)
    #               × template_factor
    # where template_factor = clip(site_factor × extra_component_factor).
    "template_scoring": {
        # Primary site factor = base + gain × origin_tissue_score.
        # Starts at 0.85 so even a cancer with no tissue reference
        # (origin_tissue_score=0) still competes reasonably.
        "primary_site_base": 0.85,
        "primary_site_gain": 0.15,
        # Factor applied when a template has no host-tissue reference at all
        # (e.g. heme templates without an explicit tissue column).
        "missing_host_factor": 0.75,
        # Met site factor = base + gain × sqrt(template_tissue_score).
        # sqrt compresses the tissue score so partial matches still contribute.
        "met_site_base": 0.30,
        "met_site_gain": 0.70,
        # When the primary-site tissue scores above this threshold on a met
        # template, a penalty is applied — the idea is that strong primary-site
        # expression evidence makes a metastatic template less plausible.
        "met_origin_preference_min": 0.5,
        # Penalty = clip(1 - gain × (origin_score - template_score), floor, 1)
        "met_origin_penalty_gain": 0.6,
        "met_origin_penalty_floor": 0.65,
        # Extra-component factor rewards met templates whose site-specific
        # host cell (e.g. hepatocyte, astrocyte) is actually detected.
        # factor = base + gain × clip(extra_fraction / full_fraction, 0, 1)
        "extra_component_base": 0.55,
        "extra_component_gain": 0.45,
        # The extra-component fraction at which the bonus is fully realised.
        # 3% of the total sample is a meaningful host-cell presence for a
        # metastatic site (e.g. hepatocytes in a liver met biopsy).
        "extra_component_full_fraction": 0.03,
        # Discount for met templates that lack a site-specific component
        # (e.g. met_lymph_node and met_soft_tissue share the standard
        # immune/stroma basis with no extra host cell).
        "met_no_extra_factor": 0.85,
        # How much the NNLS fit quality matters relative to cancer-type
        # support.  score = fit × (base + gain × cancer_support), so at
        # cancer_support=1.0 the fit matters fully; at 0.0 the fit is
        # discounted to 35% weight.
        "fit_score_base": 0.35,
        "fit_score_gain": 0.65,
        # Hard floor on the combined template factor to prevent a template
        # from being completely zeroed out by a poor tissue match.
        "min_template_factor": 0.05,
        # Minimum tumor fraction below which the hypothesis score is
        # heavily penalised.  A candidate at <2% purity is biologically
        # nonsensical (no sample is 98% TME) and represents the NNLS
        # solver routing almost all expression to host components.
        # Penalty: score *= max(purity, floor) / floor  → smooth ramp
        # that effectively zeroes out near-zero-purity candidates.
        "min_tumor_fraction": 0.02,
    },
}


def get_decomposition_parameters():
    """Return the current decomposition free parameters."""
    return DECOMPOSITION_PARAMETERS


def infer_sample_mode(candidate_rows=None, cancer_types=None, sample_mode="auto"):
    """Infer the broad sample regime used to choose decomposition templates."""
    if sample_mode != "auto":
        return sample_mode

    heme_codes = set(DECOMPOSITION_PARAMETERS["sample_mode"]["heme_codes"])
    codes = []
    if candidate_rows:
        codes.extend([row["code"] for row in candidate_rows[:2] if row.get("code")])
    if not codes and cancer_types:
        codes.extend([str(code) for code in cancer_types if code])

    if codes and codes[0] in heme_codes:
        return "heme"
    if codes and all(code in heme_codes for code in codes):
        return "heme"
    return "solid"


def _normalize_site_hint(site_hint):
    if site_hint is None:
        return None
    norm = str(site_hint).strip().lower().replace("-", "_").replace(" ", "_")
    return norm or None


def _resolve_templates(
    sample_mode,
    candidate_rows=None,
    cancer_types=None,
    templates=None,
    tumor_context="auto",
    site_hint=None,
):
    """Return the templates to evaluate for the selected sample mode."""
    if templates is not None:
        template_list = list(templates)
        if not template_list:
            templates = None
        else:
            unknown = [t for t in template_list if t not in TEMPLATES]
            if unknown:
                valid = sorted(TEMPLATES)
                raise ValueError(
                    f"Unknown template(s): {unknown}. Valid templates: {valid}"
                )
            return template_list, infer_sample_mode(
                candidate_rows=candidate_rows,
                cancer_types=cancer_types,
                sample_mode=sample_mode,
            )

    resolved_mode = infer_sample_mode(
        candidate_rows=candidate_rows,
        cancer_types=cancer_types,
        sample_mode=sample_mode,
    )
    template_sets = DECOMPOSITION_PARAMETERS["sample_mode"]["template_sets"]
    if resolved_mode not in template_sets:
        raise ValueError(
            f"Unknown sample_mode '{resolved_mode}'. "
            f"Valid modes: {sorted(template_sets)} and 'auto'."
        )

    resolved_templates = list(template_sets[resolved_mode])
    if resolved_mode != "solid":
        return resolved_templates, resolved_mode

    valid_contexts = set(DECOMPOSITION_PARAMETERS["sample_mode"]["tumor_contexts"])
    if tumor_context not in valid_contexts:
        raise ValueError(
            f"Unknown tumor_context '{tumor_context}'. "
            f"Valid contexts: {sorted(valid_contexts)}."
        )

    site_template = None
    site_hint_norm = _normalize_site_hint(site_hint)
    if site_hint_norm is not None:
        site_template = DECOMPOSITION_PARAMETERS["sample_mode"][
            "site_hint_templates"
        ].get(site_hint_norm)
        if site_template is None:
            valid_hints = sorted(
                DECOMPOSITION_PARAMETERS["sample_mode"]["site_hint_templates"]
            )
            raise ValueError(
                f"Unknown site_hint '{site_hint}'. Valid hints include: {valid_hints[:12]}"
            )

    if tumor_context == "primary":
        return ["solid_primary"], resolved_mode

    if tumor_context == "met":
        if site_template is not None:
            return [site_template], resolved_mode
        return [
            name for name in resolved_templates if name.startswith("met_")
        ], resolved_mode

    if site_template is not None:
        prioritized = [site_template, "solid_primary"]
        prioritized.extend(
            name for name in resolved_templates if name not in prioritized
        )
        return prioritized, resolved_mode

    return resolved_templates, resolved_mode


@dataclass
class DecompositionResult:
    """Result of decomposing a sample into broad components."""

    template: str
    cancer_type: str
    cancer_signature_score: float | None
    cancer_purity_score: float | None
    cancer_support_score: float | None
    template_tissue_score: float | None
    template_origin_tissue_score: float | None
    template_site_factor: float | None
    template_extra_fraction: float | None
    fractions: dict[str, float]
    purity: float
    purity_result: dict[str, Any] | None
    reconstruction_error: float
    component_trace: pd.DataFrame
    marker_trace: pd.DataFrame
    gene_attribution: pd.DataFrame
    tme_background_hk: dict[str, float]
    score: float
    description: str = ""
    warnings: list[str] = field(default_factory=list)
    matched_normal_tissue: str | None = None
    matched_normal_fraction: float = 0.0
    lineage_tumor_fraction: dict[str, Any] | None = None
    purity_source: str = "signature"
    n_measured_in_fit: int = 0


def _hk_normalize(values, genes, hk_gene_set):
    """Normalize an expression vector by its housekeeping-gene median."""
    hk_vals = [
        values[i] for i, g in enumerate(genes) if g in hk_gene_set and values[i] > 0
    ]
    hk_med = float(np.median(hk_vals)) if hk_vals else 1.0
    if hk_med <= 0:
        hk_med = 1.0
    return values / hk_med, hk_med


def _weighted_constrained_nnls(
    A,
    b,
    weights=None,
    sum_to_one_weight=DECOMPOSITION_PARAMETERS["nnls"]["sum_to_one_weight"],
    l2_penalty=DECOMPOSITION_PARAMETERS["nnls"]["l2_penalty"],
):
    """Weighted NNLS with a soft sum-to-one constraint and light ridge penalty."""
    from scipy.optimize import nnls

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.size == 0 or A.shape[1] == 0:
        return np.zeros(A.shape[1], dtype=float), float("inf")

    if weights is None:
        weights = np.ones(A.shape[0], dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = np.where(weights > 0, weights, 1.0)

    A_weighted = A * weights[:, None]
    b_weighted = b * weights

    aug_rows = [A_weighted]
    aug_targets = [b_weighted]

    if l2_penalty > 0:
        aug_rows.append(np.sqrt(l2_penalty) * np.eye(A.shape[1]))
        aug_targets.append(np.zeros(A.shape[1], dtype=float))

    aug_rows.append(np.full((1, A.shape[1]), sum_to_one_weight, dtype=float))
    aug_targets.append(np.array([sum_to_one_weight], dtype=float))

    A_aug = np.vstack(aug_rows)
    b_aug = np.concatenate(aug_targets)
    solution, _ = nnls(A_aug, b_aug)

    total = float(solution.sum())
    if total > 0:
        solution = solution / total

    residual = float(np.sqrt(np.mean(((A @ solution) - b) ** 2)))
    return solution, residual


def _select_marker_rows(
    genes,
    symbols,
    sig_matrix_hk,
    comp_names,
    cancer_type=None,
    sample_context=None,
    top_n_per_component=DECOMPOSITION_PARAMETERS["marker_selection"][
        "top_n_per_component"
    ],
):
    """Pick component-enriched marker rows for the weighted fit.

    Matched-normal (``matched_normal_<tissue>``) components are left out
    of marker selection. The generic specificity machinery would pick
    prostate-glandular / lung-alveolar / colon-enterocyte markers that
    are also strongly expressed in samples of the matched cancer type
    (retained-lineage genes), and using those as matched-normal markers
    destabilises the NNLS — we saw fit residuals more than double on a
    PRAD+smooth-muscle synthetic, flipping template selection (see
    ``ffa9325``). Panel-based marker anchoring was explored and produced
    the same destabilisation via a different route (smooth-muscle
    collinearity between prostate bulk and the generic fibroblast
    reference); dropped in favor of letting the NNLS allocate the
    matched-normal column as a free sink for parent-tissue signal, and
    using the tumor-biased / matched-normal-biased panels elsewhere
    (lineage-specific tumor-fraction estimator — see
    ``tumor_purity.estimate_lineage_tumor_fraction``).

    ``cancer_type`` is still threaded through so future marker-selection
    calibration (e.g. panel-gated anchoring with a stronger specificity
    test) can use it without another signature change.
    """
    symbol_to_rows = {}
    for idx, symbol in enumerate(symbols):
        symbol_to_rows.setdefault(str(symbol), []).append(idx)

    # #25: when the sample is FFPE / moderately-or-severely degraded,
    # markers drawn from known long-transcript genes (>6.9 kb coding —
    # the ``long`` column of ``data/degradation-gene-pairs.csv``) get
    # systematically suppressed TPM because long transcripts fragment
    # first. Downweight these markers so the NNLS isn't fooled into
    # reading the suppression as low component abundance.
    long_transcript_symbols: set[str] = set()
    context_weight_factor = 1.0
    if sample_context is not None and getattr(sample_context, "is_degraded", False):
        context_weight_factor = float(sample_context.long_transcript_weight_factor())
        if context_weight_factor < 1.0:
            from pirlygenes.gene_sets_cancer import degradation_gene_pairs

            long_transcript_symbols = {
                long_sym for _, long_sym, _ in degradation_gene_pairs()
            }

    marker_records = []
    fit_weight_by_row = {}

    for comp_idx, comp in enumerate(comp_names):
        if comp.startswith("matched_normal_"):
            continue
        comp_signal = sig_matrix_hk[:, comp_idx]
        if sig_matrix_hk.shape[1] > 1:
            other_mask = np.arange(sig_matrix_hk.shape[1]) != comp_idx
            other_max = sig_matrix_hk[:, other_mask].max(axis=1)
        else:
            other_max = np.zeros(sig_matrix_hk.shape[0], dtype=float)

        specificity = (comp_signal + 1e-6) / (other_max + 1e-6)
        score = comp_signal * np.log2(specificity + 1.0)
        keep = (comp_signal > 0.2) & (specificity > 1.5)

        chosen = []
        for marker_symbol in get_component_markers(comp):
            for idx in sorted(
                symbol_to_rows.get(marker_symbol, []),
                key=lambda row_idx: score[row_idx],
                reverse=True,
            ):
                if idx not in chosen:
                    chosen.append(idx)
                    break

        for idx in np.argsort(score)[::-1]:
            if not keep[idx]:
                continue
            if idx in chosen:
                continue
            # Block shared-APC (MHC-II) and ribosomal-protein leakage (#31).
            # Curated markers above already ran, so genuine B/T/myeloid
            # specifics are in; this only blocks the auto-pick residue.
            if _is_excluded_auto_marker(str(symbols[idx])):
                continue
            chosen.append(int(idx))
            if len(chosen) >= top_n_per_component:
                break

        if not chosen:
            chosen = [int(idx) for idx in np.argsort(score)[::-1][: min(5, len(score))]]

        for idx in chosen[:top_n_per_component]:
            marker_weight = float(max(0.5, np.log2(specificity[idx] + 1.0)))
            # #25: downweight long-transcript markers under FFPE degradation.
            if (
                context_weight_factor < 1.0
                and str(symbols[idx]) in long_transcript_symbols
            ):
                marker_weight = float(marker_weight * context_weight_factor)
            fit_weight_by_row[idx] = max(fit_weight_by_row.get(idx, 0.0), marker_weight)
            marker_records.append(
                {
                    "component": comp,
                    "gene_id": genes[idx],
                    "symbol": symbols[idx],
                    "specificity": float(specificity[idx]),
                    "reference_hk": float(comp_signal[idx]),
                    "fit_weight": marker_weight,
                }
            )

    fit_rows = sorted(fit_weight_by_row.keys())
    fit_weights = np.array([fit_weight_by_row[idx] for idx in fit_rows], dtype=float)
    marker_df = pd.DataFrame(marker_records)
    if not marker_df.empty:
        marker_df = marker_df.sort_values(
            ["component", "specificity", "reference_hk"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
    return fit_rows, fit_weights, marker_df


def _build_gene_attribution(
    genes,
    symbols,
    observed_hk,
    sample_hk_median,
    comp_names,
    comp_mix,
    tumor_fraction,
    sig_matrix_hk,
):
    """Build per-gene attribution on TPM scale."""
    tme_background_hk = (
        sig_matrix_hk @ comp_mix if len(comp_mix) else np.zeros(len(genes))
    )

    rows = []
    for idx, (gid, symbol) in enumerate(zip(genes, symbols)):
        obs_hk = float(observed_hk[idx])
        obs_tpm = obs_hk * sample_hk_median
        if obs_tpm < 0.01:
            continue

        row = {
            "gene_id": gid,
            "symbol": symbol,
            "observed_tpm": round(obs_tpm, 2),
        }
        tme_total_tpm = 0.0
        for comp_idx, comp in enumerate(comp_names):
            attr_hk = (
                (1.0 - tumor_fraction)
                * float(comp_mix[comp_idx])
                * float(sig_matrix_hk[idx, comp_idx])
            )
            attr_tpm = attr_hk * sample_hk_median
            row[comp] = round(attr_tpm, 2)
            tme_total_tpm += attr_tpm

        tumor_tpm = max(0.0, obs_tpm - tme_total_tpm)
        overexplained_tpm = max(0.0, tme_total_tpm - obs_tpm)
        row["tumor"] = round(tumor_tpm, 2)
        row["overexplained_tpm"] = round(overexplained_tpm, 2)
        row["tumor_fraction_of_total"] = round(
            tumor_tpm / obs_tpm if obs_tpm > 0 else 0.0, 4
        )
        rows.append(row)

    attr_df = pd.DataFrame(rows)
    if not attr_df.empty:
        attr_df = attr_df.sort_values("observed_tpm", ascending=False).reset_index(
            drop=True
        )
    tme_by_symbol = {
        str(symbol): float(value) for symbol, value in zip(symbols, tme_background_hk)
    }
    return attr_df, tme_by_symbol


def _build_component_trace(marker_trace, comp_names, comp_mix, tumor_fraction):
    """Summarize fitted fractions and dominant markers for each component."""
    rows = []
    non_tumor_fraction = max(0.0, 1.0 - tumor_fraction)
    for comp_idx, comp in enumerate(comp_names):
        sub = (
            marker_trace[marker_trace["component"] == comp]
            if not marker_trace.empty
            else marker_trace
        )
        if sub is not None and not sub.empty:
            marker_score = float(
                np.nanmedian(
                    sub["sample_to_ref_ratio"].replace([np.inf, -np.inf], np.nan)
                )
            )
            top_markers = ", ".join(
                sub.sort_values(
                    ["observed_tpm", "specificity"], ascending=[False, False]
                )["symbol"].head(4)
            )
            n_markers = int(len(sub))
        else:
            marker_score = 0.0
            top_markers = ""
            n_markers = 0

        rows.append(
            {
                "component": comp,
                "mix_within_tme": round(float(comp_mix[comp_idx]), 4),
                "fraction": round(float(comp_mix[comp_idx] * non_tumor_fraction), 4),
                "marker_score": round(marker_score, 4)
                if np.isfinite(marker_score)
                else None,
                "n_markers": n_markers,
                "top_markers": top_markers,
            }
        )

    component_df = pd.DataFrame(rows)
    if not component_df.empty:
        component_df = component_df.sort_values(
            "fraction", ascending=False
        ).reset_index(drop=True)
    return component_df


def _fit_one_hypothesis(
    df_gene_expr,
    sample_by_eid,
    candidate_row,
    tissue_score_map,
    template_name,
    purity_override=None,
    sample_raw_by_symbol=None,
    sample_context=None,
):
    """Fit one (cancer_type, template) broad-compartment hypothesis."""
    hk_ids = housekeeping_gene_ids()
    cancer_type = candidate_row["code"]
    purity_result = candidate_row["purity_result"]

    # Mixture-cohort subtype (#171) — when the classifier tagged a
    # winning subtype (e.g. SARC parent with winning SARC_LMS), use it
    # to route matched-normal selection (#51). Falls back silently to
    # parent-code lookup for non-mixture cohorts.
    winning_subtype = candidate_row.get("winning_subtype")

    # #59 items 2-4: optional-compartment detection. Adipocyte /
    # Schwann / erythroid compartments enter the NNLS only when the
    # sample carries enough marker signal to justify absorbing them,
    # and only on templates / cancer types where the biology makes
    # sense (see ``templates.OPTIONAL_COMPARTMENT_GATES``). With no
    # detections the component list is byte-identical to the
    # pre-#59 path.
    detected_compartments = _detect_optional_compartments(
        sample_raw_by_symbol,
        cancer_type=cancer_type,
        template_name=template_name,
    )
    components = get_template_components(
        template_name,
        cancer_type,
        winning_subtype=winning_subtype,
        detected_compartments=detected_compartments,
    )
    comp_names = [comp for comp in components if comp != "tumor"]
    matched_normal_name = (
        matched_normal_component(cancer_type, winning_subtype=winning_subtype)
        if template_name == "solid_primary"
        else None
    )

    # Lineage-specific tumor-fraction estimator (issue #54). When we
    # have a matched-normal compartment for this cancer type, prefer a
    # panel-based tumor fraction over the generic signature-gene
    # estimate: the panel is explicitly tumor-biased vs matched normal,
    # so it doesn't confuse retained-lineage genes for tumor-cell signal
    # the way the signature machinery can. Only overrides when the
    # caller didn't pin `purity_override`, the cancer has a panel, and
    # the per-gene agreement (``stability``) is good enough. Falls back
    # silently otherwise so non-epithelial paths stay on the existing
    # purity flow.
    lineage_fraction_info = None
    purity_source = "signature"
    warnings = []
    if (
        purity_override is None
        and matched_normal_name is not None
        and sample_raw_by_symbol is not None
    ):
        lineage_fraction_info = estimate_lineage_tumor_fraction(
            sample_raw_by_symbol,
            cancer_type,
        )

    if purity_override is None:
        if (
            lineage_fraction_info is not None
            and lineage_fraction_info["stability"] < 1.5
            and lineage_fraction_info["panel_genes_observed"] >= 10
        ):
            candidate_purity = float(purity_result.get("overall_estimate") or 0.5)
            lineage_estimate = float(lineage_fraction_info["estimate"])
            override_params = DECOMPOSITION_PARAMETERS["lineage_override"]
            upward_delta = lineage_estimate - candidate_purity
            upward_ratio = lineage_estimate / max(
                candidate_purity, override_params["ratio_floor"]
            )
            if (
                upward_delta > override_params["max_upward_delta"]
                and upward_ratio > override_params["max_upward_ratio"]
            ):
                tumor_fraction = candidate_purity
                purity_to_store = dict(purity_result or {})
                purity_to_store["lineage_tumor_fraction"] = lineage_fraction_info
                purity_to_store["purity_source"] = purity_source
                warnings.append(
                    "Lineage-panel purity conflicts with signature purity; "
                    "kept the conservative signature prior"
                )
            else:
                tumor_fraction = lineage_estimate
                purity_source = "lineage_panel"
                purity_to_store = dict(purity_result or {})
                purity_to_store["overall_estimate"] = tumor_fraction
                purity_to_store["overall_lower"] = float(lineage_fraction_info["lower"])
                purity_to_store["overall_upper"] = float(lineage_fraction_info["upper"])
                purity_to_store["lineage_tumor_fraction"] = lineage_fraction_info
                purity_to_store["purity_source"] = purity_source
        else:
            tumor_fraction = float(purity_result.get("overall_estimate") or 0.5)
            purity_to_store = purity_result
            if lineage_fraction_info is not None:
                purity_to_store = dict(purity_result or {})
                purity_to_store["lineage_tumor_fraction"] = lineage_fraction_info
    else:
        tumor_fraction = float(np.clip(purity_override, 0.0, 1.0))
        purity_to_store = {
            "overall_lower": tumor_fraction,
            "overall_estimate": tumor_fraction,
            "overall_upper": tumor_fraction,
            "purity_source": "override",
        }
        purity_source = "override"
    if not comp_names or tumor_fraction >= 0.999:
        return DecompositionResult(
            template=template_name,
            cancer_type=cancer_type,
            cancer_signature_score=float(candidate_row["signature_score"]),
            cancer_purity_score=float(candidate_row["purity_estimate"]),
            cancer_support_score=float(candidate_row["support_norm"]),
            template_tissue_score=1.0,
            template_origin_tissue_score=1.0,
            template_site_factor=1.0,
            template_extra_fraction=0.0,
            fractions={"tumor": 1.0},
            purity=1.0,
            purity_result=purity_to_store,
            reconstruction_error=0.0,
            component_trace=pd.DataFrame(),
            marker_trace=pd.DataFrame(),
            gene_attribution=pd.DataFrame(),
            tme_background_hk={},
            score=float(candidate_row["support_norm"]),
            description=f"{cancer_type} — {TEMPLATES.get(template_name, {}).get('description', template_name)}",
            warnings=["No non-tumor components in template"],
            matched_normal_tissue=(
                EPITHELIAL_MATCHED_NORMAL_TISSUE.get(cancer_type)
                if matched_normal_name
                else None
            ),
            matched_normal_fraction=0.0,
            lineage_tumor_fraction=lineage_fraction_info,
            purity_source=purity_source,
            n_measured_in_fit=0,
        )

    gene_subset = set(sample_by_eid.keys())
    filt_genes, filt_symbols, sig_raw, _ = build_signature_matrix(
        comp_names,
        gene_subset=gene_subset,
        sample_by_eid=sample_by_eid,
    )
    filt_sample_vec = np.array(
        [
            float(sample_by_eid[gene_id]) if gene_id in sample_by_eid else np.nan
            for gene_id in filt_genes
        ],
        dtype=float,
    )
    measured_mask = np.isfinite(filt_sample_vec)
    if not np.all(measured_mask):
        dropped = int((~measured_mask).sum())
        warnings.append(f"Dropped {dropped} unmeasured genes from decomposition fit")
        filt_genes = [gene for gene, keep in zip(filt_genes, measured_mask) if keep]
        filt_symbols = [
            symbol for symbol, keep in zip(filt_symbols, measured_mask) if keep
        ]
        sig_raw = sig_raw[measured_mask, :]
        filt_sample_vec = filt_sample_vec[measured_mask]
    observed_hk, sample_hk_median = _hk_normalize(filt_sample_vec, filt_genes, hk_ids)

    sig_hk = np.zeros_like(sig_raw, dtype=float)
    for comp_idx in range(sig_raw.shape[1]):
        sig_hk[:, comp_idx], _ = _hk_normalize(sig_raw[:, comp_idx], filt_genes, hk_ids)

    fit_rows, fit_weights, marker_trace = _select_marker_rows(
        filt_genes,
        filt_symbols,
        sig_hk,
        comp_names,
        cancer_type=cancer_type,
        sample_context=sample_context,
    )
    if len(fit_rows) < max(10, len(comp_names) * 2):
        warnings.append("Low marker support for template fit")

    if not fit_rows:
        floor = DECOMPOSITION_PARAMETERS["marker_selection"][
            "fallback_expression_floor"
        ]
        fit_rows = list(
            np.where((observed_hk > floor) | (sig_hk.max(axis=1) > floor))[0]
        )
        fit_weights = np.ones(len(fit_rows), dtype=float)
    n_measured_in_fit = int(len(fit_rows))

    A = sig_hk[fit_rows]
    b = observed_hk[fit_rows]
    # Weight by 1/observed to minimize proportional (not absolute) error.
    # Without this, genes spanning 10^-2 to 10^5 HK-units let high-expression
    # outliers (e.g. Ig genes in plasma-heavy samples) dominate the residual,
    # leaking hundreds of TPM of "unexplained" immunoglobulin into the tumor
    # attribution.  The floor at 0.1 prevents noisy low-expression genes from
    # being over-amplified.
    combined_weights = fit_weights / np.maximum(b, 0.1)
    comp_mix, residual = _weighted_constrained_nnls(A, b, weights=combined_weights)

    if not marker_trace.empty:
        marker_trace = marker_trace.copy()
        symbol_to_obs = {
            str(symbol): float(obs)
            for symbol, obs in zip(filt_symbols, filt_sample_vec)
        }
        symbol_to_obs_hk = {
            str(symbol): float(obs) for symbol, obs in zip(filt_symbols, observed_hk)
        }
        marker_trace["observed_tpm"] = (
            marker_trace["symbol"].map(symbol_to_obs).fillna(0.0)
        )
        marker_trace["sample_hk"] = (
            marker_trace["symbol"].map(symbol_to_obs_hk).fillna(0.0)
        )
        marker_trace["sample_to_ref_ratio"] = marker_trace["sample_hk"] / marker_trace[
            "reference_hk"
        ].replace(0, np.nan)

    component_trace = _build_component_trace(
        marker_trace, comp_names, comp_mix, tumor_fraction
    )
    gene_attr, tme_background_hk = _build_gene_attribution(
        filt_genes,
        filt_symbols,
        observed_hk,
        sample_hk_median,
        comp_names,
        comp_mix,
        tumor_fraction,
        sig_hk,
    )

    fractions = {"tumor": float(tumor_fraction)}
    for comp_idx, comp in enumerate(comp_names):
        fractions[comp] = float(comp_mix[comp_idx] * max(0.0, 1.0 - tumor_fraction))

    origin_tissues = get_template_host_tissues("solid_primary", cancer_type=cancer_type)
    origin_tissue_scores = [
        (tissue, float(tissue_score_map.get(tissue, 0.0))) for tissue in origin_tissues
    ]
    if origin_tissue_scores:
        _, origin_tissue_score = max(origin_tissue_scores, key=lambda item: item[1])
    else:
        origin_tissue_score = 0.0

    host_tissues = get_template_host_tissues(template_name, cancer_type=cancer_type)
    host_tissue_scores = [
        (tissue, float(tissue_score_map.get(tissue, 0.0))) for tissue in host_tissues
    ]
    if host_tissue_scores:
        host_tissue, template_tissue_score = max(
            host_tissue_scores, key=lambda item: item[1]
        )
    else:
        host_tissue, template_tissue_score = None, 0.0

    # Extra-component scoring rewards met templates whose site-specific host
    # cell is detected. Matched-normal epithelium is deliberately excluded:
    # it is a lineage-awareness addition to solid_primary, not a met host
    # cell, and including it here would re-balance the primary-vs-met
    # scoring (see `ffa9325` regression notes in issue #50).
    extra_components = {
        comp
        for comp in get_template_extra_components(template_name)
        if not comp.startswith("matched_normal_")
    }
    extra_fraction = float(
        sum(
            comp_mix[idx]
            for idx, comp in enumerate(comp_names)
            if comp in extra_components
        )
    )
    extra_sample_fraction = extra_fraction * max(0.0, 1.0 - tumor_fraction)

    matched_normal_mix = 0.0
    if matched_normal_name is not None and matched_normal_name in comp_names:
        mn_idx = comp_names.index(matched_normal_name)
        matched_normal_mix = float(comp_mix[mn_idx])
    matched_normal_fraction = matched_normal_mix * max(0.0, 1.0 - tumor_fraction)

    scoring = DECOMPOSITION_PARAMETERS["template_scoring"]
    if template_name == "solid_primary":
        template_site_factor = (
            scoring["primary_site_base"]
            + scoring["primary_site_gain"] * origin_tissue_score
        )
    elif host_tissue is None:
        template_site_factor = scoring["missing_host_factor"]
    else:
        template_site_factor = float(
            scoring["met_site_base"]
            + scoring["met_site_gain"] * np.sqrt(max(template_tissue_score, 0.0))
        )
        if origin_tissue_score >= scoring["met_origin_preference_min"]:
            origin_advantage = max(0.0, origin_tissue_score - template_tissue_score)
            origin_penalty = float(
                np.clip(
                    1.0 - scoring["met_origin_penalty_gain"] * origin_advantage,
                    scoring["met_origin_penalty_floor"],
                    1.0,
                )
            )
            template_site_factor *= origin_penalty

    if extra_components:
        extra_component_factor = float(
            scoring["extra_component_base"]
            + scoring["extra_component_gain"]
            * np.clip(
                extra_sample_fraction / scoring["extra_component_full_fraction"],
                0.0,
                1.0,
            )
        )
        if extra_sample_fraction < 0.01:
            warnings.append("Template-specific host component is effectively unused")
    elif template_name.startswith("met_"):
        # Met templates without an explicit host compartment need stronger site
        # evidence than primaries because immune/stromal infiltrates can occur
        # in either setting.
        extra_component_factor = scoring["met_no_extra_factor"]
    else:
        extra_component_factor = 1.0

    template_factor = float(
        np.clip(
            template_site_factor * extra_component_factor,
            scoring["min_template_factor"],
            1.0,
        )
    )

    if host_tissue is not None and template_tissue_score < 0.2:
        warnings.append(f"Weak host-tissue support for {host_tissue}")
    if (
        template_name.startswith("met_")
        and origin_tissue_score > template_tissue_score + 0.2
    ):
        warnings.append("Primary tissue support exceeds metastatic-site support")

    fit_score = 1.0 / (1.0 + residual)
    cancer_support = float(candidate_row["support_norm"])
    score = (
        fit_score
        * (scoring["fit_score_base"] + scoring["fit_score_gain"] * cancer_support)
        * template_factor
    )

    # Purity floor penalty (#98): candidates with biologically
    # implausible purity (<2%) get their score collapsed so they
    # don't clutter the candidate list as noise.
    min_tf = scoring.get("min_tumor_fraction", 0.02)
    if tumor_fraction < min_tf:
        penalty = tumor_fraction / max(min_tf, 1e-6)
        score *= penalty
        warnings.append(
            f"Purity {tumor_fraction:.1%} below {min_tf:.0%} floor "
            f"(score penalised ×{penalty:.2f})"
        )

    if not gene_attr.empty and gene_attr["overexplained_tpm"].gt(0).mean() > 0.2:
        warnings.append("Many genes are overexplained by the TME background")

    return DecompositionResult(
        template=template_name,
        cancer_type=cancer_type,
        cancer_signature_score=float(candidate_row["signature_score"]),
        cancer_purity_score=float(candidate_row["purity_estimate"]),
        cancer_support_score=cancer_support,
        template_tissue_score=template_tissue_score,
        template_origin_tissue_score=origin_tissue_score,
        template_site_factor=template_factor,
        template_extra_fraction=extra_sample_fraction,
        fractions=fractions,
        purity=float(tumor_fraction),
        purity_result=purity_to_store,
        reconstruction_error=float(residual),
        component_trace=component_trace,
        marker_trace=marker_trace,
        gene_attribution=gene_attr,
        tme_background_hk=tme_background_hk,
        score=float(score),
        description=f"{cancer_type} — {TEMPLATES.get(template_name, {}).get('description', template_name)}",
        warnings=warnings,
        matched_normal_tissue=(
            EPITHELIAL_MATCHED_NORMAL_TISSUE.get(cancer_type)
            if matched_normal_name
            else None
        ),
        matched_normal_fraction=matched_normal_fraction,
        lineage_tumor_fraction=lineage_fraction_info,
        purity_source=purity_source,
        n_measured_in_fit=n_measured_in_fit,
    )


def decompose_sample(
    df_gene_expr,
    cancer_types=None,
    templates=None,
    top_k=3,
    purity_override=None,
    sample_mode="auto",
    tumor_context="auto",
    site_hint=None,
    sample_context=None,
    candidate_rows=None,
):
    """Decompose a sample across multiple cancer-type and template hypotheses.

    Epithelial primaries whose cancer type is in
    :data:`pirlygenes.decomposition.templates.EPITHELIAL_MATCHED_NORMAL_TISSUE`
    get an additional ``matched_normal_<tissue>`` compartment in the
    ``solid_primary`` template, so admixed benign parent tissue (benign
    prostate glands, adjacent normal colon mucosa, etc.) is absorbed as
    non-tumor signal rather than attributed to tumor cells (issue #50).
    Purity for those same cases comes from a lineage-specific tumor-
    fraction estimator (:func:`panels.estimate_lineage_tumor_fraction`)
    when the per-gene agreement is stable enough; falls back to the
    signature-gene estimator otherwise. Non-epithelial primaries
    (SARC, heme, glioma, etc.) retain the existing behavior unchanged.
    """
    from ..plot import _sample_expression_by_symbol

    sample_raw_by_symbol, _ = _sample_expression_by_symbol(df_gene_expr)
    # Decomposition templates were tuned against native-scale references.
    # Opt out of renormalize_to_million pending the #27 calibration sweep.
    ref = (
        pan_cancer_expression(
            technical_rna_normalize=True,
            renormalize_to_million=False,
        )
        .drop_duplicates(subset="Symbol")
        .set_index("Symbol")
    )
    sym_to_eid = ref["Ensembl_Gene_ID"].to_dict()
    sample_by_eid = {}
    for symbol, tpm in sample_raw_by_symbol.items():
        eid = sym_to_eid.get(symbol)
        if eid:
            sample_by_eid[eid] = float(tpm)

    # Reuse pre-ranked candidates when the caller has them (#85). The
    # CLI's analyze() already computes the full ranking trace —
    # including the expensive per-candidate ``estimate_tumor_purity``
    # pass — so re-ranking here doubled that cost for every run.
    if candidate_rows is None:
        candidate_rows = rank_cancer_type_candidates(
            df_gene_expr,
            candidate_codes=cancer_types,
            top_k=len(cancer_types) if cancer_types is not None else 6,
        )
    else:
        if cancer_types is not None:
            requested = set(cancer_types)
            candidate_rows = [row for row in candidate_rows if row["code"] in requested]
    if not candidate_rows:
        return []

    templates, _resolved_mode = _resolve_templates(
        sample_mode=sample_mode,
        candidate_rows=candidate_rows,
        cancer_types=cancer_types,
        templates=templates,
        tumor_context=tumor_context,
        site_hint=site_hint,
    )

    tissue_score_map = {
        tissue: score
        for tissue, score, _ in _score_host_tissues(sample_raw_by_symbol, top_n=None)
    }

    results = []
    for candidate_row in candidate_rows:
        for template_name in templates:
            result = _fit_one_hypothesis(
                df_gene_expr,
                sample_by_eid,
                candidate_row,
                tissue_score_map,
                template_name,
                purity_override=purity_override,
                sample_raw_by_symbol=sample_raw_by_symbol,
                sample_context=sample_context,
            )
            results.append(result)

    results.sort(key=lambda row: (-row.score, row.cancer_type, row.template))
    return results[:top_k]
