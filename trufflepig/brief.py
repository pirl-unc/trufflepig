# Licensed under the Apache License, Version 2.0

"""Two-tier markdown handoff documents (#111).

Audience distinction:

- ``*-summary.md`` — one-page summary, ≤ 40 lines. For a clinician
  skimming before a tumor board, or an LLM asked for a 3-sentence
  referral-note paragraph. Strict structure; no internal jargon.
  (Named ``brief.md`` through 4.40; the earlier free-form
  ``summary.md`` paragraph was retired as redundant with analysis.md.)
- ``*-actionable.md`` — longer treatment-review document. For an
  oncologist preparing a treatment discussion, reading carefully.

Both consume:

- ``analysis`` (the shared dict produced by ``analyze_sample`` and
  enriched by the CLI pipeline), including ``purity_confidence`` from
  :mod:`pirlygenes.confidence`.
- ``ranges_df`` (per-gene tumor-expression + #108 attribution).
- The disease-state narrative from ``compose_disease_state_narrative``.
- The curated cancer-key-genes panels from ``gene_sets_cancer``.

No JSON is produced — consumers read markdown. (See
``feedback_markdown_not_json`` in the memory: the audience test for a
new output is "who reads this and when?", not "is it machine-
parseable?".)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from .reporting import (
    agent_metadata_clause,
    offcontext_known_targets,
    analysis_site_template_for_subtype,
    cancer_code_display_name,
    cancer_key_genes_lookup_for_analysis,
    candidate_winning_subtype_for_analysis,
    clinical_maturity_summary,
    context_expression_band_cell,
    indication_biomarker,
    indication_biomarker_label,
    expression_independent_indication,
    expression_independent_interpretation,
    expression_independent_rna_context,
    filter_current_therapy_targets,
    format_missing_observation_cell,
    format_missing_observation_interp,
    hla_restrictions_for_target_row,
    hla_restricted_target_supported,
    normal_expression_context,
    report_disease_state_text,
    target_observation_state,
    same_lineage_material_target_candidate,
    supplied_alteration_context_for_target_row,
    supplied_alteration_supports_target_row,
    subtype_curation_scope_note,
    therapy_path_context,
    therapy_path_rank,
    therapy_state_caution,
    tumor_band_available,
    tumor_band_cell,
    target_reliability_status,
    tpm_semantics_note,
    tumor_attribution_context,
)
from .confidence import concise_confidence_reasons
from .analyze import cancer_type_context_from_analysis, cancer_type_context_label
from .rna_qc import rna_quant_qc_summary_line
from trufflepig.expression_qc import expression_qc_rescue_summary_line
from .sample_context import (
    heuristic_support_label,
    library_prep_clause,
    library_prep_display_label,
)

logger = logging.getLogger(__name__)


def _display_sample_id(sample_id: Optional[str]) -> Optional[str]:
    if sample_id is None:
        return None
    text = str(sample_id).strip()
    if not text:
        return None
    if "/" in text or "\\" in text:
        text = Path(text).name.strip()
    return text or None


def _display_subtype_code(code: Optional[str]) -> str:
    text = str(code or "").strip()
    if not text:
        return "the alternate subtype"
    try:
        from pirlygenes.gene_sets_cancer import cancer_type_registry

        reg = cancer_type_registry()
        match = reg[reg["code"] == text]
        if not match.empty:
            row = match.iloc[0]
            subtype_key = row.get("subtype_key")
            if (
                isinstance(subtype_key, str)
                and subtype_key
                and subtype_key.lower() != "nan"
            ):
                return subtype_key.replace("_", " ")
            name = row.get("name")
            if isinstance(name, str) and name:
                return name.split("(")[0].strip().lower()
    except Exception:
        logger.debug("subtype display lookup failed", exc_info=True)
    return text.replace("_", " ").lower()


def _registry_row_for_code(code: Optional[str]):
    text = str(code or "").strip()
    if not text:
        return None
    try:
        from pirlygenes.gene_sets_cancer import cancer_type_registry

        reg = cancer_type_registry()
        match = reg[reg["code"] == text]
        if not match.empty:
            return match.iloc[0]
    except Exception:
        logger.debug("failed to load cancer-type registry row", exc_info=True)
    return None


def _cancer_type_context_label(code: Optional[str]) -> str:
    return cancer_type_context_label(code)


def _cancer_call_rescue_kind(analysis) -> str:
    rescue = analysis.get("cancer_call_rescue") or {}
    return str(rescue.get("kind") or "").strip()


def _cancer_call_rescue_basis_line(analysis, cancer_code: str) -> Optional[str]:
    call_rescue = analysis.get("cancer_call_rescue") or {}
    if not call_rescue:
        return None

    kind = str(call_rescue.get("kind") or "").strip()
    if kind == "low_purity_prad_stromal_context":
        return (
            "**Cancer-type basis:** RNA-inferred PRAD context rescue: prostate "
            "tissue/marker evidence and raw PRAD signature are present, while the "
            "epithelial PRAD lineage program is attenuated and a stromal/"
            "smooth-muscle SARC signal dominates. Confirm pathology, tumor "
            "cellularity, preservation/RIN, and treatment state before using the "
            "therapy shortlist."
        )
    if kind == "coarse_tcga_orphan_context":
        recommended = str(
            call_rescue.get("recommended_code") or cancer_code or ""
        ).strip()
        competitor = str(call_rescue.get("competing_code") or "").strip()
        basis = str(call_rescue.get("context_basis") or "").strip()
        label = _cancer_type_context_label(recommended)
        if basis == "normal_tissue_match":
            evidence = (
                "Tissue composition screen and expected normal-tissue context"
            )
        else:
            evidence = "Tissue composition screen and direct cancer evidence"
        competitor_clause = (
            f" over {_cancer_type_context_label(competitor)}" if competitor else ""
        )
        return (
            f"**Cancer-type basis:** RNA-inferred {label} context rescue: "
            f"{evidence} support {recommended}, so the classifier suspended the "
            f"orphan-family penalty{competitor_clause}. Confirm pathology or "
            "clinical diagnosis before using the therapy shortlist."
        )
    message = str(call_rescue.get("message") or "").strip()
    if message:
        return (
            "**Cancer-type basis:** RNA-inferred context rescue: "
            f"{message.rstrip('.')}. Confirm pathology or clinical diagnosis "
            "before using the therapy shortlist."
        )
    return (
        "**Cancer-type basis:** RNA-inferred context rescue adjusted the provisional "
        "report label; confirm pathology or clinical diagnosis before using the "
        "therapy shortlist."
    )


def _cancer_call_rescue_summary_line(analysis) -> Optional[str]:
    if _cancer_call_rescue_kind(analysis) != "low_purity_prad_stromal_context":
        return None
    return (
        "**QC/call pitfall:** prostate tissue/context is present, but the "
        "PRAD epithelial lineage program is attenuated and stromal/"
        "smooth-muscle RNA can mimic SARC; check tumor cellularity, "
        "preservation/RIN, and treatment state before trusting an "
        "expression-only sarcoma call."
    )


def _cancer_call_rescue_actionable_line(analysis) -> Optional[str]:
    if _cancer_call_rescue_kind(analysis) != "low_purity_prad_stromal_context":
        return None
    return (
        "\nQC/call pitfall: prostate tissue/context is present, but the "
        "PRAD epithelial lineage program is attenuated and stromal/"
        "smooth-muscle RNA can mimic SARC. Treat a standalone sarcoma call "
        "as unsupported unless pathology agrees."
    )


def _inferred_site_context_line(analysis) -> Optional[str]:
    inferred = analysis.get("inferred_site_context") or {}
    if not inferred:
        return None
    site = str(inferred.get("site") or "").replace("_", " ")
    tissue = str(inferred.get("tissue") or site).replace("_", " ")
    score = float(inferred.get("score") or 0.0)
    primary = str(inferred.get("primary_tissue") or "").replace("_", " ")
    primary_score = inferred.get("primary_tissue_score")
    primary_clause = ""
    if primary and isinstance(primary_score, (int, float)):
        primary_clause = f", above {primary} {float(primary_score):.2f}"
    return (
        f"**Inferred site context:** likely {site} metastatic host/background "
        f"signal ({tissue} score {score:.2f}{primary_clause}); inferred from "
        "expression, not supplied as a user constraint."
    )


def _registry_family(code: Optional[str]) -> str:
    row = _registry_row_for_code(code)
    if row is None:
        return ""
    family = str(row.get("family") or "").strip()
    return "" if family.lower() == "nan" else family


def _sarcoma_lineage_codes() -> frozenset:
    # Derive sarcoma membership from pirlygenes' canonical, prefix-agnostic API
    # rather than a hardcoded family set, so it survives the ongoing taxonomy /
    # registry restructure (pirlygenes Phase C) without a trufflepig edit.
    from pirlygenes.gene_sets_cancer import sarcoma_lineage_codes

    return frozenset(sarcoma_lineage_codes())


def _clinical_supergroup(code: Optional[str]) -> str:
    code_text = str(code or "").strip()
    if code_text and code_text in _sarcoma_lineage_codes():
        return "sarcoma/bone/soft-tissue"
    family = _registry_family(code)
    if family.startswith("carcinoma-"):
        return family
    return family


def _broad_context_compatibility(top_code: str, supplied_code: str) -> str:
    """Reader-facing relationship between expression context and report scope."""

    top = str(top_code or "").strip()
    supplied = str(supplied_code or "").strip()
    if not top or not supplied or top == supplied:
        return ""
    top_group = _clinical_supergroup(top)
    supplied_group = _clinical_supergroup(supplied)
    if top_group and supplied_group and top_group == supplied_group:
        if top_group == "sarcoma/bone/soft-tissue":
            return "sarcoma-family first-pass support"
        return "same-family first-pass support"
    return ""


def _call_confidence_suffix(
    call_tier,
    *,
    concise: bool = True,
    include_reasons: bool = True,
) -> str:
    """Render cancer-call confidence consistently across Markdown reports."""
    if call_tier.tier not in {"low", "moderate"} or not call_tier.reasons:
        return ""
    tier_text = f"{call_tier.tier} confidence"
    if call_tier.tier == "low":
        tier_text += ", provisional"
    if not include_reasons:
        return f" — **{tier_text}**"
    note = concise_confidence_reasons(call_tier) if concise else call_tier.inline_note
    return f" — **{tier_text}** ({note})" if note else f" — **{tier_text}**"


def _confidence_caveat_clause(call_tier) -> str:
    if getattr(call_tier, "tier", "") not in {"low", "moderate"}:
        return ""
    note = concise_confidence_reasons(call_tier)
    return f"; confidence caveats: {note}" if note else ""


def _site_template_note_label(template_name: Optional[str]) -> str:
    mapping = {
        "met_adrenal": "adrenal-associated",
        "met_bone": "bone-associated",
        "met_brain": "brain-associated",
        "met_liver": "liver-associated",
        "met_lung": "lung-associated",
        "met_lymph_node": "lymph-node-associated",
        "met_peritoneal": "peritoneal-associated",
        "met_skin": "skin-associated",
        "met_soft_tissue": "soft-tissue-associated",
        "solid_primary": "primary-site-compatible",
    }
    text = str(template_name or "").strip()
    if not text:
        return "site"
    return mapping.get(text, text.replace("_", " "))


def _shared_signature_note(shared_signature: Optional[str]) -> str:
    text = str(shared_signature or "").strip()
    if not text:
        return "shared lineage pattern"
    text = text.replace("_", " ")
    text = text.replace("+", " / ")
    text = text.replace(" amp", " amplification")
    return text


def _render_subtype_note(
    resolution: dict,
    *,
    original_subtype: Optional[str],
    site_template: Optional[str],
) -> str:
    if not resolution:
        return ""
    status = str(resolution.get("status") or "")
    rule = str(resolution.get("rule") or "")
    shared_signature = _shared_signature_note(resolution.get("shared_signature"))
    final_label = _display_subtype_code(resolution.get("final_subtype"))
    original_label = _display_subtype_code(original_subtype)
    alternatives = [
        _display_subtype_code(code) for code in (resolution.get("alternatives") or [])
    ]
    deduped_alternatives = []
    seen_labels = {final_label.lower()}
    for label in alternatives:
        key = label.lower()
        if key in seen_labels:
            continue
        seen_labels.add(key)
        deduped_alternatives.append(label)
    alternatives = deduped_alternatives

    if status == "corrected":
        if rule == "site_template":
            site_label = _site_template_note_label(site_template)
            return (
                f"{site_label.capitalize()} context favors {final_label} over {original_label}; "
                f"both can share the {shared_signature}."
            )
        if rule == "fusion_surrogate":
            return (
                f"Fusion-surrogate expression favors {final_label} over {original_label}; "
                f"both can share the {shared_signature}."
            )
        if rule == "marker_combo":
            return (
                f"The marker combination is more consistent with {final_label} than {original_label}; "
                f"both can share the {shared_signature}."
            )
        return (
            f"Additional subtype evidence favors {final_label} over {original_label}; "
            f"both can share the {shared_signature}."
        )

    if status == "degenerate":
        option_text = (
            " vs ".join([final_label] + alternatives) if alternatives else final_label
        )
        if rule == "site_template":
            return (
                f"Subtype remains unresolved between {option_text}; the available site context does not break the tie, "
                f"and these options can share the {shared_signature}."
            )
        if rule == "fusion_surrogate":
            return (
                f"Subtype remains unresolved between {option_text}; the available fusion-surrogate expression does not break the tie, "
                f"and these options can share the {shared_signature}."
            )
        if rule == "marker_combo":
            return (
                f"Subtype remains unresolved between {option_text}; the available marker combination does not break the tie, "
                f"and these options can share the {shared_signature}."
            )
        return (
            f"Subtype remains unresolved between {option_text}; the available evidence does not break the tie, "
            f"and these options can share the {shared_signature}."
        )

    return str(resolution.get("reason") or "").strip()


def _phase_label(phase: str) -> str:
    return {
        "approved": "Approved",
        "phase_3": "Phase 3",
        "phase_2": "Phase 2",
        "phase_1": "Phase 1",
        "preclinical": "Preclinical",
        "off_label": "Off-label / transfer rationale",
    }.get(phase, phase)


def _top_candidate_signature_score(analysis) -> float | None:
    """Return the top-ranked cancer candidate's signature match score.

    Used by the tissue-composition banner to suppress noise: a confident
    cancer-reference signature match is evidence of tumor and nudges the banner to
    stay silent on soft "composition-ambiguous" cases.
    """
    candidates = (
        analysis.get("cancer_candidates") or analysis.get("candidate_trace") or []
    )
    if not candidates:
        return None
    top = candidates[0]
    # Different code paths store this under slightly different keys.
    for key in ("signature_score", "support_fraction_of_top", "geomean", "normalized"):
        if key in top and top[key] is not None:
            try:
                return float(top[key])
            except (TypeError, ValueError):
                continue
    return None


def _subtype_specific_row_out_of_scope(target_row, analysis) -> bool:
    """Suppress subtype-specific SARC rows when SARC is unresolved.

    The full target tables can list subtype-specific rows as context. The brief
    top-3 and summary HLA prompts should not imply Ewing/GIST/synovial-specific
    eligibility for a broad SARC call unless that subtype was actually selected
    or direct alteration evidence supports the row.
    """
    if not analysis:
        return False
    active_code = str(analysis.get("cancer_type") or "").strip()
    if active_code != "SARC":
        return False
    target_subtype = str(target_row.get("subtype") or "").strip()
    if not target_subtype:
        return False
    active_panel_subtype = str(analysis.get("_target_panel_subtype") or "").strip()
    if active_panel_subtype and target_subtype == active_panel_subtype:
        return False
    return not bool(supplied_alteration_supports_target_row(target_row, analysis))


def _row_context_text(target_row) -> str:
    if not hasattr(target_row, "get"):
        return ""
    return " ".join(
        str(target_row.get(key) or "")
        for key in ("indication", "rationale", "agent", "agent_class", "eligibility_note")
    ).lower()


def _therapy_axis_state_for_brief(analysis, axis: str) -> str:
    if not isinstance(analysis, dict):
        return ""
    scores = analysis.get("therapy_response_scores") or {}
    score = scores.get(axis) if hasattr(scores, "get") else None
    if score is None:
        return ""
    if hasattr(score, "get"):
        return str(score.get("state") or "").strip().lower()
    return str(getattr(score, "state", "") or "").strip().lower()


def _brca_er_dependent_row_inactive(target_row, analysis) -> bool:
    """Whether an ER/HR-dependent BRCA row conflicts with ER-low RNA state."""
    if not isinstance(analysis, dict) or not hasattr(target_row, "get"):
        return False
    cancer_code = str(target_row.get("cancer_code") or analysis.get("cancer_type") or "")
    if cancer_code.strip().upper() != "BRCA":
        return False
    if _therapy_axis_state_for_brief(analysis, "ER_signaling") != "down":
        return False
    sym = str(target_row.get("symbol") or "").strip().upper()
    if sym not in {"ESR1", "PGR", "CDK4", "CDK6"}:
        return False
    text = _row_context_text(target_row)
    return any(token in text for token in ("er+", "hr+", "endocrine", "esr1-mut"))


def _has_direct_eligibility_input(analysis, biomarker: str) -> bool:
    """Best-effort check for orthogonal eligibility evidence supplied to this run."""
    if not isinstance(analysis, dict):
        return False
    constraints = analysis.get("analysis_constraints") or {}
    if biomarker == "mutation":
        return any(
            bool(analysis.get(key))
            for key in (
                "fusion_inputs_supplied",
                "alteration_inputs_supplied",
                "variant_inputs_supplied",
                "mutation_inputs_supplied",
                "cnv_inputs_supplied",
            )
        ) or any(
            bool(constraints.get(key))
            for key in (
                "fusions",
                "fusion_file",
                "variants",
                "mutations",
                "cnvs",
                "alterations",
            )
        )
    if biomarker == "msi_high":
        return any(
            bool(constraints.get(key))
            for key in ("msi_status", "mmr_status", "msi", "mmr")
        )
    if biomarker == "tmb_high":
        return any(bool(constraints.get(key)) for key in ("tmb", "tmb_status"))
    if biomarker == "histology_only":
        return bool(constraints.get("cancer_type")) or str(
            analysis.get("cancer_type_source") or ""
        ).strip() == "user-specified"
    return False


def _scope_level_eligibility_context(target_row, analysis) -> str:
    """Context for rows gated by the report diagnosis rather than target RNA."""
    if not isinstance(analysis, dict) or not hasattr(target_row, "get"):
        return ""
    row_code = str(target_row.get("cancer_code") or "").strip().upper()
    report_code = str(analysis.get("cancer_type") or "").strip().upper()
    if row_code != "NUTM" or report_code != "NUTM":
        return ""
    if analysis.get("fusion_report_scope_inference"):
        return (
            "scope-level fusion evidence supports the NUTM report label; "
            "verify NUT carcinoma diagnosis/fusion status before treating "
            "as eligible"
        )
    if analysis.get("rare_report_scope_inference"):
        return (
            "NUTM report label is RNA-inferred; confirm NUTM1 fusion/IHC/"
            "FISH/pathology before treating as eligible"
        )
    if str(analysis.get("cancer_type_source") or "").strip() == "user-specified":
        return (
            "externally supplied NUTM label supports report scope; verify "
            "clinical diagnosis/fusion status before treating as eligible"
        )
    return ""


def _expression_independent_evidence_gap(target_row, analysis) -> str:
    """Surface when non-expression eligibility evidence was not provided."""
    if not expression_independent_indication(target_row):
        return ""
    supplied_context = supplied_alteration_context_for_target_row(
        target_row,
        analysis,
    )
    if supplied_context:
        return supplied_context
    scope_context = _scope_level_eligibility_context(target_row, analysis)
    if scope_context:
        return scope_context
    biomarker = indication_biomarker(target_row)
    if biomarker == "histology_only":
        if _has_direct_eligibility_input(analysis, biomarker):
            return ""
        return (
            "eligibility evidence not supplied to this run: confirm diagnosis/"
            "histology before treating as eligible"
        )
    label = indication_biomarker_label(target_row)
    if biomarker == "mutation" and _has_direct_eligibility_input(analysis, biomarker):
        return (
            "orthogonal mutation/fusion/CNV evidence was supplied, but no "
            "target-specific supporting call was recognized for this row; "
            f"confirm {label} before treating as eligible"
        )
    if biomarker != "mutation" and _has_direct_eligibility_input(analysis, biomarker):
        return (
            f"required eligibility evidence was supplied to this run; verify the "
            f"{label} call matches the indication"
        )
    return (
        f"required eligibility evidence not supplied to this run: confirm {label} "
        "before treating as eligible"
    )


def _format_therapy_bullet(
    target_row,
    expression_row,
    target_panel=None,
    *,
    analysis=None,
    disease_state=None,
    ranges_df=None,
) -> str:
    """One standardized therapy bullet for the brief."""
    sym = str(target_row.get("symbol") or "")
    agent = str(target_row.get("agent") or "—")
    phase = _phase_label(str(target_row.get("phase") or ""))
    indication = str(target_row.get("indication") or "")
    indication_clause = f", {indication}" if indication else ""
    expr_independent = expression_independent_indication(target_row)
    path_context = therapy_path_context(
        target_row,
        analysis=analysis,
        disease_state=disease_state,
    )
    path_prefix = f"{path_context}. " if path_context else ""
    state_caution = therapy_state_caution(
        target_row,
        analysis=analysis,
        disease_state=disease_state,
    )
    caution_suffix = (
        f" Current-therapy check: {state_caution}." if state_caution else ""
    )
    maturity = clinical_maturity_summary(target_row, target_panel=target_panel)

    def _eligibility_evidence_gap() -> str:
        return _expression_independent_evidence_gap(target_row, analysis)

    def _sentence(parts, *, maturity: str | None = None) -> str:
        body = "; ".join(part for part in parts if part)
        if maturity:
            return f"{body}. Clinical maturity: {maturity}."
        return f"{body}."

    if expression_row is None:
        if expr_independent:
            parts = [
                expression_independent_interpretation(target_row),
                expression_independent_rna_context(None),
                _eligibility_evidence_gap(),
            ]
            if path_context:
                parts.append(path_context)
            return (
                f"- **{sym}** — {agent} ({phase}{indication_clause}). "
                f"{_sentence(parts, maturity=maturity)}{caution_suffix}"
            )
        state = target_observation_state(sym, ranges_df)
        if state == "below_detection":
            missing_phrase = "Target **below detection** in this sample (bulk TPM ≈ 0)"
        elif state == "not_in_input":
            missing_phrase = (
                "Target gene symbol **not present in input file** "
                "(coverage gap, not a biological negative)"
            )
        else:
            missing_phrase = "Target **not measured** in this sample"
        return (
            f"- **{sym}** — {agent} ({phase}{indication_clause}). "
            f"{path_prefix}{missing_phrase}.{caution_suffix}"
        )
    observed = float(expression_row.get("observed_tpm") or 0.0)
    if observed < 1.0:
        if expr_independent:
            parts = [
                expression_independent_interpretation(target_row),
                expression_independent_rna_context(expression_row),
                _eligibility_evidence_gap(),
            ]
            if path_context:
                parts.append(path_context)
            return (
                f"- **{sym}** — {agent} ({phase}{indication_clause}). "
                f"{_sentence(parts, maturity=maturity)}{caution_suffix}"
            )
        return (
            f"- **{sym}** — {agent} ({phase}{indication_clause}). "
            f"{path_prefix}Bulk target RNA {observed:.1f} TPM — "
            f"**target absent** in this sample.{caution_suffix}"
        )
    if not tumor_band_available(expression_row):
        parts = [
            f"Bulk TPM {observed:.0f}",
            "tumor-inferred model interval unavailable",
        ]
        if path_context:
            parts.append(path_context)
        return (
            f"- **{sym}** — {agent} ({phase}{indication_clause}). "
            f"{_sentence(parts, maturity=maturity)}{caution_suffix}"
        )
    source = tumor_attribution_context(expression_row)
    normal = normal_expression_context(expression_row)
    if expr_independent:
        interpretation_parts = [
            expression_independent_interpretation(target_row),
            expression_independent_rna_context(expression_row),
            _eligibility_evidence_gap(),
        ]
    else:
        interpretation_parts = [source["label"], source["band"], normal["label"]]
    notes = list(source.get("notes") or []) + list(normal.get("details") or [])
    if notes and not expr_independent:
        interpretation_parts.append(notes[0])
    if path_context:
        interpretation_parts.append(path_context)
    agent_meta = agent_metadata_clause(target_row)
    if agent_meta:
        interpretation_parts.append(agent_meta)
    interpretation = "; ".join(part for part in interpretation_parts if part)
    maturity_sentence = f" Clinical maturity: {maturity}." if maturity else ""
    return (
        f"- **{sym}** — {agent} ({phase}{indication_clause}). "
        f"{interpretation}.{maturity_sentence}{caution_suffix}"
    )


def _top_therapies(
    targets_df,
    ranges_df,
    limit=3,
    *,
    analysis=None,
    disease_state=None,
):
    """Pick the top therapies to show in the brief.

    Priority: approved agents with present targets first, ranked by
    tumor-attributed TPM; then trial phases. Non-measured / absent
    targets are skipped from the brief (they still show in the full
    landscape in ``actionable.md`` / ``targets.md``).
    """
    if targets_df is None or len(targets_df) == 0 or ranges_df is None:
        return []
    from .common import (
        ranges_by_gene_id,
        ranges_by_symbol,
        panel_symbols_to_gene_ids,
    )
    # Resolve target symbols → Ensembl IDs at the boundary; internal
    # lookups go through ID-keyed view. Symbol-keyed view is kept
    # as a fallback for legacy ranges_df frames without gene_id.
    target_records = targets_df.to_dict("records")
    sym_to_id = panel_symbols_to_gene_ids(
        str(t.get("symbol") or "").strip() for t in target_records
    )
    id_to_row = ranges_by_gene_id(ranges_df)
    sym_to_row = ranges_by_symbol(ranges_df)

    phase_priority = {
        "approved": 0,
        "phase_3": 1,
        "phase_2": 2,
        "phase_1": 3,
        "preclinical": 4,
        "off_label": 5,
    }

    scored = []
    for t in target_records:
        sym = str(t.get("symbol") or "")
        gene_id = sym_to_id.get(sym.strip())
        expr = id_to_row.get(gene_id) if gene_id else None
        if expr is None:
            expr = sym_to_row.get(sym)
        expr_independent = expression_independent_indication(t)
        if _subtype_specific_row_out_of_scope(t, analysis):
            continue
        if _brca_er_dependent_row_inactive(t, analysis):
            continue
        supplied_alteration_rank = (
            0 if supplied_alteration_supports_target_row(t, analysis) else 1
        )
        if expr is None:
            if not hla_restricted_target_supported(t, analysis=analysis):
                continue
            if expr_independent:
                phase = str(t.get("phase") or "")
                scored.append(
                    (
                        (
                            supplied_alteration_rank,
                            therapy_path_rank(
                                t,
                                analysis=analysis,
                                disease_state=disease_state,
                            ),
                            phase_priority.get(phase, 99),
                            1,
                            1,
                            0.0,
                            sym,
                        ),
                        t,
                        None,
                    )
                )
            continue
        if not hla_restricted_target_supported(t, analysis=analysis):
            continue
        observed = float(expr.get("observed_tpm") or 0.0)
        if observed < 1.0 and not expr_independent:
            # Target absent — the brief reports presence, not absence.
            # The full landscape in targets.md has the absence noted.
            continue
        attr_tumor = float(expr.get("attr_tumor_tpm") or 0.0)
        attr_fraction = float(expr.get("attr_tumor_fraction") or 1.0)
        lineage_material = same_lineage_material_target_candidate(
            expr,
            target_row=t,
        )
        # Drop rows that are mostly non-tumor from the top-3 — they
        # don't belong in the clinician handoff per #79 semantics.
        # Same-lineage clinical targets are a special case: a prostate
        # lineage marker assigned partly to matched-normal prostate is
        # source-ambiguous, not equivalent to an immune/stromal target.
        if attr_fraction < 0.30 and not expr_independent and not lineage_material:
            continue
        reliability_status = target_reliability_status(expr, target_row=t)
        if reliability_status == "unsupported":
            continue
        # Note (#128): we deliberately do NOT filter on
        # ``broadly_expressed`` here. The caller's ``targets_df`` is
        # the **curated** cancer-key-genes panel (#110) — every row
        # in it has been evaluated by hand as a clinician-relevant
        # target, often because the targeting mechanism is
        # amplification or lineage-retained overexpression rather
        # than baseline expression breadth (ERBB2 for HER2+ BRCA,
        # MDM2 for WD/DD-LPS, GPC3 for HCC). Trust curation. The
        # broadly-expressed flag is enforced in the generic Surface
        # / Intracellular target tables where ranking is by raw
        # expression, not curation.
        phase = str(t.get("phase") or "")
        reliability_rank = 0 if reliability_status == "supported" else 1
        expression_rank = 1 if expr_independent else 0
        sort_key = (
            supplied_alteration_rank,
            therapy_path_rank(
                t,
                analysis=analysis,
                disease_state=disease_state,
            ),
            phase_priority.get(phase, 99),
            expression_rank,
            reliability_rank,
            -attr_tumor,
            sym,
        )
        scored.append((sort_key, t, expr))

    # Sort by key only — avoid pandas Series comparison in tie-break.
    scored.sort(key=lambda item: item[0])

    deduped = []
    seen_symbols = set()
    for sort_key, t, expr in scored:
        sym = str(t.get("symbol") or "")
        if sym in seen_symbols:
            continue
        seen_symbols.add(sym)
        deduped.append((t, expr))
        if len(deduped) >= limit:
            break
    return deduped


def _brief_float(value, default=0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    if result != result:
        return float(default)
    return result


def _brief_truthy(value) -> bool:
    if value is None:
        return False
    try:
        if value != value:
            return False
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _format_trace_tpm(value) -> str:
    value = _brief_float(value, 0.0)
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_component_label(value) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return "—"
    text = text.replace("_", " ")
    if text.startswith("matched normal "):
        text = text.replace("matched normal ", "matched-normal ", 1)
    return text


def _top_non_tumor_attribution(expression_row) -> tuple[str, float]:
    label = _format_component_label(expression_row.get("attr_top_compartment"))
    value = _brief_float(expression_row.get("attr_top_compartment_tpm"), 0.0)
    if label != "—" and label.lower() != "tumor" and value > 0:
        return label, value

    attribution = expression_row.get("attribution")
    if isinstance(attribution, str):
        try:
            import ast

            attribution = ast.literal_eval(attribution)
        except Exception:
            attribution = None
    if isinstance(attribution, dict):
        candidates = []
        for comp, comp_value in attribution.items():
            comp_label = _format_component_label(comp)
            if comp_label == "—" or comp_label.lower() == "tumor":
                continue
            comp_tpm = _brief_float(comp_value, 0.0)
            if comp_tpm > 0:
                candidates.append((comp_label, comp_tpm))
        if candidates:
            return max(candidates, key=lambda item: item[1])
    return "—", 0.0


def _trace_phase_label(target_row) -> str:
    phase = str(target_row.get("phase") or "")
    return {
        "approved": "approved",
        "phase_3": "phase 3",
        "phase_2": "phase 2",
        "phase_1": "phase 1 exploratory",
        "preclinical": "preclinical",
    }.get(phase, phase.replace("_", " ") or "curated")


def _source_trace_reason(target_row, expression_row, *, in_shortlist: bool) -> str:
    source = tumor_attribution_context(expression_row)
    reliability = target_reliability_status(expression_row, target_row=target_row)
    phase = _trace_phase_label(target_row)
    attr_fraction = _brief_float(expression_row.get("attr_tumor_fraction"), 0.0)
    comp_label, _comp_tpm = _top_non_tumor_attribution(expression_row)
    lineage_material = same_lineage_material_target_candidate(
        expression_row,
        target_row=target_row,
    )

    parts = []
    if phase and phase != "approved":
        parts.append(phase)

    if expression_independent_indication(target_row):
        parts.append("RNA is context only; eligibility does not depend on target expression")
    elif in_shortlist:
        if source["tier"] == "tumor_supported":
            parts.append("mostly tumor signal")
        elif lineage_material:
            parts.append("same-lineage marker; tumor origin uncertain")
        elif source["tier"] == "mixed_source":
            parts.append("mixed tumor/background signal")
        elif source["tier"] == "background_dominant":
            parts.append("mostly background signal")
        else:
            parts.append(source["summary"])
    elif lineage_material:
        parts.append("same-lineage marker; tumor origin uncertain")
    elif _brief_truthy(expression_row.get("matched_normal_over_predicted")):
        if comp_label != "—":
            background = (
                "lineage background"
                if comp_label.lower().startswith("matched-normal")
                else "non-tumor background"
            )
            parts.append(f"{comp_label} over-predicts / {background}")
        else:
            parts.append("matched-normal over-predicts / lineage background")
    elif reliability == "unsupported" and attr_fraction < 0.30:
        if comp_label != "—":
            parts.append(f"{attr_fraction:.0%} tumor; mostly {comp_label}")
        else:
            parts.append(f"{attr_fraction:.0%} tumor fraction")
    elif reliability == "unsupported":
        parts.append(
            f"mostly {comp_label}/background"
            if comp_label != "—"
            else "background-dominant"
        )
    elif reliability == "provisional":
        parts.append(source["label"])
    else:
        parts.append("ranked below top list")

    deduped = []
    for part in parts:
        if part and part not in deduped:
            deduped.append(part)
    return "; ".join(deduped)


def _shortlist_omission_note(targets_df, ranges_df, top_rows) -> str:
    """Trace source attribution for non-clean clinical target decisions."""
    if targets_df is None or ranges_df is None or not top_rows:
        return ""
    top_symbols = {str(t.get("symbol") or "") for t, _ in top_rows}
    from .common import (
        ranges_by_gene_id,
        ranges_by_symbol,
        panel_symbols_to_gene_ids,
    )
    target_records = targets_df.to_dict("records")
    sym_to_id = panel_symbols_to_gene_ids(
        str(t.get("symbol") or "").strip() for t in target_records
    )
    id_to_row = ranges_by_gene_id(ranges_df)
    sym_to_row = ranges_by_symbol(ranges_df)  # fallback for ID-less frames
    omitted = []
    seen = set()
    for target in target_records:
        sym = str(target.get("symbol") or "").strip()
        if not sym or sym.lower() == "nan" or sym in top_symbols or sym in seen:
            continue
        seen.add(sym)
        gene_id = sym_to_id.get(sym)
        expr = id_to_row.get(gene_id) if gene_id else None
        if expr is None:
            expr = sym_to_row.get(sym)
        if expr is None or _brief_float(expr.get("observed_tpm"), 0.0) < 1.0:
            continue
        attr_fraction = _brief_float(expr.get("attr_tumor_fraction"), 1.0)
        reliability_status = target_reliability_status(expr, target_row=target)
        phase = str(target.get("phase") or "")
        if reliability_status != "supported" or phase in {"phase_1", "preclinical"}:
            comp_label, comp_tpm = _top_non_tumor_attribution(expr)
            omitted.append(
                {
                    "symbol": sym,
                    "bulk": _brief_float(expr.get("observed_tpm"), 0.0),
                    "tumor": _brief_float(expr.get("attr_tumor_tpm"), 0.0),
                    "fraction": attr_fraction,
                    "component": comp_label,
                    "component_tpm": comp_tpm,
                    "reason": _source_trace_reason(target, expr, in_shortlist=False),
                }
            )
        if len(omitted) >= 4:
            break
    shortlist_context = []
    for target, expr in top_rows:
        if expr is None:
            continue
        phase = str(target.get("phase") or "")
        source = tumor_attribution_context(expr)
        if phase == "approved" and source["tier"] == "tumor_supported":
            continue
        if phase == "approved":
            continue
        comp_label, comp_tpm = _top_non_tumor_attribution(expr)
        shortlist_context.append(
            {
                "symbol": str(target.get("symbol") or ""),
                "bulk": _brief_float(expr.get("observed_tpm"), 0.0),
                "tumor": _brief_float(expr.get("attr_tumor_tpm"), 0.0),
                "fraction": _brief_float(expr.get("attr_tumor_fraction"), 0.0),
                "component": comp_label,
                "component_tpm": comp_tpm,
                "reason": _source_trace_reason(target, expr, in_shortlist=True),
            }
        )
        if len(shortlist_context) >= 2:
            break

    rows = shortlist_context + omitted
    if not rows:
        return ""
    lines = [
        "**Where target RNA signal appears to come from**",
        "",
        "| Gene | Bulk TPM | Tumor-source bulk TPM | Tumor fraction | Top non-tumor attribution | Component TPM | Main reason |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        component = row["component"] if row["component"] != "—" else "none modeled"
        lines.append(
            f"| {row['symbol']} | {_format_trace_tpm(row['bulk'])} | "
            f"{_format_trace_tpm(row['tumor'])} | {row['fraction']:.0%} | "
            f"{component} | {_format_trace_tpm(row['component_tpm'])} | "
            f"{row['reason']} |"
        )
    lines.append(
        "*Source attribution is a caveat, not an automatic exclusion; "
        "clinical maturity and eligibility still set the shortlist order.*"
    )
    return "\n".join(lines)


def _disease_state_summary_lines(disease_state_display):
    """Return summary lines without packing unrelated state clauses together.

    Handles every on-the-wire form of the IFN marker we've seen:
    ``**Active IFN response**``, ``** Active IFN response**`` (stray
    space from upstream composition, which used to strand the leading
    ``**`` on the disease-state line), and the unbolded ``Active IFN
    response``.  Strips dangling bold / punctuation from the boundary
    between the two halves so neither line ends up with unbalanced
    bold markers or a doubled em-dash.  Emits only the immune line
    when there's no real disease-state content before the marker.
    """
    text = str(disease_state_display or "").strip()
    if not text:
        return []
    import re as _re

    pattern = _re.compile(r"\*{0,2}\s*Active IFN response\*{0,2}")
    match = pattern.search(text)
    if match is None:
        return [f"**Disease state:** {text}"]

    before = text[: match.start()]
    after = text[match.end():]
    before = _re.sub(r"[\s;,.\*]+$", "", before).strip()
    after = _re.sub(r"^[\s\*]+", "", after)
    after = _re.sub(r"^[—–-]\s*", "", after).strip()

    lines: list[str] = []
    if before:
        lines.append(f"**Disease state:** {before}")
    ifn = "Active IFN response"
    if after:
        ifn = f"{ifn} — {after}"
    lines.append(f"**Immune/IFN state:** {ifn}")
    return lines


def _panel_display_label(panel_code, panel_subtype=None):
    if panel_subtype:
        return f"{panel_code} ({str(panel_subtype).replace('_', ' ')})"
    return panel_code


def _curated_target_panel_for_sample(cancer_code, analysis, ranges_df=None):
    from pirlygenes.gene_sets_cancer import cancer_therapy_targets

    panel_code, panel_subtype = cancer_key_genes_lookup_for_analysis(
        cancer_code,
        analysis,
        ranges_df=ranges_df,
    )
    if panel_subtype:
        targets_df = cancer_therapy_targets(panel_code, subtype=panel_subtype)
    else:
        targets_df = cancer_therapy_targets(panel_code)
    targets_df = filter_current_therapy_targets(targets_df)
    return panel_code, panel_subtype, targets_df.reset_index(drop=True)


def _caveats_from_purity_tier(
    purity_tier,
    sample_context,
    analysis=None,
) -> List[str]:
    """User-facing caveat lines for the brief.

    Converts the ``purity_tier.reasons`` (which are short internal
    strings) into full sentences without internal jargon.
    """
    if purity_tier is None:
        return []
    reasons = getattr(purity_tier, "reasons", []) or []
    out = []
    for reason in reasons:
        r = str(reason)
        if "wide purity CI" in r:
            out.append(
                "Purity range is wide — target TPMs "
                "could be over- or under-stated depending on the true "
                "purity."
            )
        elif "low-purity regime" in r:
            out.append(
                "Sample is in a low-purity regime — raw target TPMs "
                "tend to overstate tumor presence. Prefer the tumor-"
                "attributed values."
            )
        elif "severe RNA degradation" in r:
            out.append(
                "RNA is severely degraded — long-transcript targets "
                "are systematically under-counted; interpret "
                "negative results cautiously."
            )
        elif "moderate RNA degradation" in r:
            out.append(
                "RNA is partially degraded — long-transcript targets "
                "are under-counted to a moderate degree."
            )
        elif "targeted-panel" in r:
            out.append(
                "Input appears to be a targeted panel rather than "
                "whole-transcriptome — relative expression estimates "
                "should be interpreted within the panel only."
            )
    # Library prep / preservation note from sample_context.
    if sample_context is not None:
        prep = getattr(sample_context, "library_prep", None)
        preservation = getattr(sample_context, "preservation", None)
        prep_label = library_prep_display_label(prep) if prep else None
        if prep_label and preservation == "ffpe":
            out.append(
                f"FFPE preservation with {prep_label} library prep — "
                "short transcripts favored; some classes of targets "
                "are artificially depressed."
            )
        elif prep == "exome_capture":
            out.append(
                "RNA hybrid-capture / RNA-exome prep — rRNA and many "
                "non-polyadenylated RNAs are under-sampled by design; "
                "low MT fraction is expected, but measurable MT mRNAs "
                "should not be treated as filtered out."
            )
    scale_qc = (analysis or {}).get("expression_scale_qc") or {}
    rescue_line = expression_qc_rescue_summary_line(
        (analysis or {}).get("expression_qc_rescue")
    )
    if rescue_line:
        out.append(rescue_line.replace("**", ""))
    if scale_qc.get("warnings"):
        out.append("Expression scale QC: " + str(scale_qc["warnings"][0]) + ".")
    rna_qc = (analysis or {}).get("rna_quant_qc") or {}
    if rna_qc.get("warnings"):
        out.append("RNA quantification QC: " + str(rna_qc["warnings"][0]))
    return out


def _cancer_type_basis_line(analysis, cancer_code: str) -> str:
    constraints = analysis.get("analysis_constraints") or {}
    constrained_code = str(constraints.get("cancer_type") or "").strip()
    source = str(analysis.get("cancer_type_source") or "").strip()
    cancer_type_context = cancer_type_context_from_analysis(analysis)
    report_context_code = cancer_type_context.code_for("report")
    parent_context_code = cancer_type_context.code_for("parent")
    fusion_inference = analysis.get("fusion_report_scope_inference") or {}
    if fusion_inference and not constrained_code and source != "user-specified":
        fusion = fusion_inference.get("fusion") or {}
        pair = str(fusion.get("pair") or fusion_inference.get("expected_pair") or "")
        label = str(
            fusion_inference.get("label")
            or fusion_inference.get("cancer_type")
            or "rare cancer"
        ).strip()
        confirm = str(
            fusion_inference.get("confirmatory_tests")
            or "orthogonal clinical testing"
        ).strip()
        return (
            f"**Cancer-type basis:** fusion-supported {label} hypothesis from "
            f"{pair} sets a provisional report label; confirm with {confirm} or "
            "clinical diagnosis before using the therapy shortlist."
        )
    rare_inference = analysis.get("rare_report_scope_inference") or {}
    if rare_inference and not constrained_code and source != "user-specified":
        surrogate = str(rare_inference.get("surrogate") or "RNA surrogate")
        tpm = rare_inference.get("surrogate_tpm")
        tpm_clause = f" ({tpm:g} TPM)" if isinstance(tpm, (int, float)) else ""
        confirm = str(
            rare_inference.get("confirmatory_tests")
            or "orthogonal clinical testing"
        ).strip()
        return (
            f"**Cancer-type basis:** RNA-inferred rare-cancer hypothesis from "
            f"{surrogate}{tpm_clause} sets a provisional report label; confirm "
            f"with {confirm} or clinical diagnosis before using the therapy shortlist."
        )
    fine_inference = analysis.get("fine_report_scope_inference") or {}
    if fine_inference and not constrained_code and source != "user-specified":
        reference = str(
            fine_inference.get("reference_cancer_type")
            or analysis.get("reference_cancer_type")
            or "the broad reference"
        ).strip()
        metrics = fine_inference.get("metrics") or {}
        score = metrics.get("fine_reference_support") or fine_inference.get(
            "fine_reference_strength"
        )
        score_clause = (
            f" (fine-reference support {score:.2f})"
            if isinstance(score, (int, float))
            else ""
        )
        return (
            f"**Cancer-type basis:** RNA evidence supports "
            f"{_cancer_type_context_label(cancer_code)} as the fine label over "
            f"the {_cancer_type_context_label(reference)} expression-reference context"
            f"{score_clause}; modules that need a coarse expression reference still "
            f"use {reference}."
        )
    call_rescue = analysis.get("cancer_call_rescue") or {}
    if call_rescue and not constrained_code and source != "user-specified":
        return _cancer_call_rescue_basis_line(analysis, cancer_code)
    if constrained_code or source == "user-specified":
        supplied = report_context_code or constrained_code or str(cancer_code or "").strip()
        supplied_label = cancer_type_context.label_for("report") or _cancer_type_context_label(supplied)
        if cancer_type_context.uses_distinct_reference:
            reference_label = cancer_type_context.label_for("reference")
            hierarchy = (
                "parent expression context"
                if parent_context_code
                else "broad expression context"
            )
            fine_clause = (
                " Exact fine-grained expression references are available for "
                "subtype-aware modules."
                if cancer_type_context.fine_expression_available
                else ""
            )
            return (
                f"**Cancer-type basis:** externally supplied {supplied_label} "
                f"sets the fine/report label; {reference_label} is used as the "
                f"{hierarchy} where downstream steps need a coarse cohort."
                f"{fine_clause}"
            )
        return (
            f"**Cancer-type basis:** externally supplied {supplied_label} sets "
            "the report label; RNA evidence is used downstream for confidence, "
            "purity/decomposition, target attribution, and expression-context checks."
        )
    return (
        "**Cancer-type basis:** RNA-inferred hypothesis sets a provisional report "
        "label; confirm with pathology or clinical diagnosis before using the "
        "therapy shortlist."
    )


# Lineage panels below this score don't earn a brief.md line. Mirrors
# the cancer_type_evidence selector's _LINEAGE_PANEL_MIN_SCORE (0.60)
# but stays lower so high-confidence-but-below-promotion panels still
# get reported as evidence.
_LINEAGE_PANEL_BRIEF_MIN_SCORE = 0.5


def _lineage_panel_evidence_line(analysis, cancer_code: str) -> Optional[str]:
    """Render the lineage-panel verdict as a follow-on context line
    after the cancer-type basis. The panel is a tie-breaker, not a
    primary determination, so this is additive: it appears whenever
    ``summarize_evidence.top_score`` reaches a reportable bar
    (``_LINEAGE_PANEL_BRIEF_MIN_SCORE``), even if the panel didn't
    promote a label.

    Wiring point #3 from ``trufflepig.lineage_panels`` (see contract
    block in lineage_panels.py). The line surfaces the panel name,
    score, rationale, and — when applicable — whether the panel
    promoted the report label or was held back by the broad
    classifier.
    """
    summary = analysis.get("lineage_panel_evidence") or {}
    if not summary:
        return None
    try:
        top_score = float(summary.get("top_score") or 0.0)
    except (TypeError, ValueError):
        return None
    if top_score < _LINEAGE_PANEL_BRIEF_MIN_SCORE:
        return None
    top_panel = str(summary.get("top_panel") or "").strip()
    if not top_panel:
        return None
    rationale = str(summary.get("top_rationale") or "").strip()
    promotion = summary.get("promotion") or {}
    promoted = bool(promotion.get("promoted"))
    promoted_code = str(promotion.get("code") or "").strip()
    blockers = [str(b) for b in (promotion.get("blockers") or []) if b]
    if promoted and promoted_code:
        promoted_clause = (
            f" — supports the {promoted_code} call"
            if promoted_code == str(cancer_code or "").strip()
            else f" — proposes {promoted_code}"
        )
    elif blockers:
        promoted_clause = f" — noted, did not change the call ({blockers[0]})"
    else:
        promoted_clause = ""
    rationale_clause = f": {rationale}" if rationale else ""
    return (
        f"**Lineage panel:** {top_panel} score "
        f"{top_score:.2f}{rationale_clause}{promoted_clause}."
    )


def _lineage_panel_subtype_reasoning_line(analysis, cancer_code: str) -> Optional[str]:
    """Surface the lineage panel's transcriptional-program note as a
    follow-on "**Subtype:**" line, separate from the raw evidence
    line. The evidence line says "what the panel scored"; this line
    says "what that biological program implies".

    The program note is read from
    ``analysis['lineage_panel_evidence']['top_panel_program_note']``
    — single source of truth, populated by the cancer_type_evidence
    selector from the winning ``LineagePanel.program_note`` field.
    Panels without a curated note simply skip this line.
    """
    summary = analysis.get("lineage_panel_evidence") or {}
    if not summary:
        return None
    try:
        top_score = float(summary.get("top_score") or 0.0)
    except (TypeError, ValueError):
        return None
    if top_score < _LINEAGE_PANEL_BRIEF_MIN_SCORE:
        return None
    top_panel = str(summary.get("top_panel") or "").strip()
    if not top_panel:
        return None
    note = str(summary.get("top_panel_program_note") or "").strip()
    if not note:
        return None
    # Only render a subtype program when the panel is consistent with the actual
    # cancer call. A high-scoring panel that was held back by the broad
    # classifier (e.g. a CHOL biliary panel under a PRAD call) must NOT be
    # presented as the call's "subtype" — that reads as a contradiction. The
    # evidence line still reports it as "noted, did not change the call".
    promotion = summary.get("promotion") or {}
    promoted_code = str(promotion.get("code") or "").strip()
    call = str(cancer_code or "").strip()
    if promotion.get("blockers"):
        return None
    if promotion.get("promoted") and promoted_code and promoted_code != call:
        return None
    return f"**Subtype:** {top_panel} — {note}."


def _fusion_pair_display(finding: dict) -> str:
    fusion = finding.get("fusion") or {}
    pair = str(fusion.get("pair") or "").strip()
    if pair:
        return pair
    return str(finding.get("expected_pair") or "fusion").strip()


def _fusion_evidence_line(analysis, cancer_code: str) -> str:
    fusion_inference = analysis.get("fusion_report_scope_inference") or {}
    findings = analysis.get("fusion_findings") or []
    if fusion_inference:
        pair = _fusion_pair_display(fusion_inference)
        label = str(
            fusion_inference.get("label")
            or fusion_inference.get("cancer_type")
            or cancer_code
        ).strip()
        expected = str(fusion_inference.get("expected_pair") or "").strip()
        expected_clause = f"; expected 5-prime/3-prime rule {expected}" if expected else ""
        note = str(fusion_inference.get("orientation_note") or "").strip()
        note_clause = f"; {note}" if note else ""
        return (
            f"**Fusion evidence:** {pair} supports {label}{expected_clause}"
            f"{note_clause}."
        )
    if findings:
        top = findings[0]
        pair = _fusion_pair_display(top)
        label = str(top.get("label") or "fusion finding").strip()
        caveat = str(top.get("caveat") or "").strip()
        caveat_clause = f" {caveat}" if caveat else ""
        return (
            f"**Fusion evidence:** {pair} matches curated {label} evidence, "
            f"but does not by itself assign the report cancer type.{caveat_clause}"
        )
    rare_inference = analysis.get("rare_report_scope_inference") or {}
    if rare_inference and not analysis.get("fusion_inputs_supplied"):
        confirm = str(
            rare_inference.get("confirmatory_tests") or "fusion testing"
        ).strip()
        surrogate = str(rare_inference.get("surrogate") or "RNA marker").strip()
        return (
            f"**Fusion evidence needed:** no fusion file was supplied; because "
            f"{surrogate} RNA supports this rare-cancer hypothesis, ask whether "
            f"{confirm} data are available."
        )
    return ""


def _alteration_evidence_line(analysis) -> str:
    records = analysis.get("alteration_records") or []
    if records:
        labels = []
        for record in records[:4]:
            if not hasattr(record, "get"):
                continue
            gene = str(record.get("gene") or "").strip()
            alteration = str(
                record.get("alteration") or record.get("alteration_type") or ""
            ).strip()
            if gene:
                if alteration.upper().startswith(gene.upper()):
                    labels.append(alteration)
                else:
                    labels.append(f"{gene} {alteration}".strip())
        if labels:
            suffix = "" if len(records) <= 4 else f" (+{len(records) - 4} more)"
            return (
                "**Alteration evidence:** supplied "
                + ", ".join(labels)
                + suffix
                + "; used as driver/eligibility context, not inferred from RNA."
            )
    if analysis.get("alteration_inputs_supplied"):
        return (
            "**Alteration evidence:** alteration input was supplied, but no usable "
            "calls were parsed; verify the file format before using therapies "
            "that require alteration evidence."
        )
    return ""


def _pathway_activity_line(analysis) -> str:
    inferences = analysis.get("pathway_activity_inferences") or []
    if not inferences:
        return ""
    finding = inferences[0]
    label = str(finding.get("label") or "Pathway activity").strip()
    fold = finding.get("up_geomean_fold")
    support = ", ".join((finding.get("support_genes") or [])[:4])
    source_labels = [
        str(source.get("label") or "").strip()
        for source in (finding.get("candidate_sources") or [])[:3]
        if str(source.get("label") or "").strip()
    ]
    unresolved = ", ".join((finding.get("unresolved_sources") or [])[:2])
    fold_clause = (
        f" high ({fold:.2f}x context)" if isinstance(fold, (int, float)) else " high"
    )
    support_clause = f"; support genes: {support}" if support else ""
    if source_labels:
        source_clause = "; plausible sources: " + "; ".join(source_labels)
    elif unresolved:
        source_clause = "; possible sources to test: " + unresolved
    else:
        source_clause = ""
    caveat = str(finding.get("caveat") or "").strip()
    caveat_clause = f" {caveat}" if caveat else ""
    return (
        f"**Active pathway:** {label}{fold_clause}{support_clause}"
        f"{source_clause}.{caveat_clause}"
    )


def _candidate_trace_rank(
    candidate_trace: list[dict],
    cancer_code: str,
) -> tuple[int | None, dict | None]:
    code = str(cancer_code or "").strip()
    if not code:
        return None, None
    for idx, row in enumerate(candidate_trace, start=1):
        if str(row.get("code") or "").strip() == code:
            return idx, row
    return None, None


def _candidate_support_score(row: dict | None) -> float | None:
    if not row:
        return None
    for key in ("support_geomean", "support_score", "support_fraction_of_top"):
        if row.get(key) is not None:
            try:
                return float(row.get(key))
            except (TypeError, ValueError):
                continue
    return None


def _candidate_code_list(rows: list[dict], *, exclude: set[str], limit: int) -> str:
    codes: list[str] = []
    for row in rows:
        code = str(row.get("code") or "").strip()
        if not code or code in exclude:
            continue
        codes.append(code)
        if len(codes) >= limit:
            break
    return ", ".join(codes)


def _rna_crosscheck_line(analysis, cancer_code: str, call_tier=None) -> str:
    constraints = analysis.get("analysis_constraints") or {}
    constrained_code = str(constraints.get("cancer_type") or "").strip()
    source = str(analysis.get("cancer_type_source") or "").strip()
    rare_inference = analysis.get("rare_report_scope_inference") or {}
    if not (constrained_code or source == "user-specified"):
        if rare_inference:
            top_code = str(rare_inference.get("top_reference_cancer_type") or "")
            candidate_trace = analysis.get("candidate_trace") or []
            alternatives = _candidate_code_list(
                candidate_trace,
                exclude={top_code},
                limit=2,
            )
            alt_clause = (
                f"; nearby alternatives include {alternatives}" if alternatives else ""
            )
            return (
                f"**RNA classifier check:** {cancer_code} is an RNA-inferred "
                f"rare-cancer hypothesis; closest expression reference is "
                f"{top_code or 'unresolved'}{alt_clause}. Use the expression "
                "reference for cohort comparisons, not as the diagnosis."
            )
        fine_inference = analysis.get("fine_report_scope_inference") or {}
        if fine_inference:
            reference = str(
                fine_inference.get("reference_cancer_type")
                or analysis.get("reference_cancer_type")
                or ""
            ).strip()
            return (
                f"**RNA classifier check:** expression-reference ranking supports "
                f"{reference or 'the parent context'}; fine-label evidence supports "
                f"{cancer_code}. Use the expression reference for cohort math and "
                "the fine label for report interpretation."
            )
        return ""

    cancer_type_context = cancer_type_context_from_analysis(analysis)
    report_context_code = cancer_type_context.code_for("report")
    parent_context_code = cancer_type_context.code_for("parent")
    supplied_code = (
        report_context_code or constrained_code or str(cancer_code or "").strip()
    )
    comparison_code = parent_context_code or constrained_code or supplied_code
    supplied_label = _cancer_type_context_label(supplied_code)
    comparison_label = _cancer_type_context_label(comparison_code)
    candidate_trace = analysis.get("candidate_trace") or []
    if not comparison_code or not candidate_trace:
        return "**RNA classifier check:** no cancer-type candidate trace available."

    top = candidate_trace[0]
    top_code = str(top.get("code") or "").strip()
    top_label = _cancer_type_context_label(top_code)
    supplied_rank, supplied_row = _candidate_trace_rank(
        candidate_trace, comparison_code
    )
    if top_code == comparison_code:
        alternatives = _candidate_code_list(
            candidate_trace,
            exclude={comparison_code},
            limit=2,
        )
        suffix = f"; nearest RNA alternatives: {alternatives}" if alternatives else ""
        suffix += _confidence_caveat_clause(call_tier)
        if report_context_code and parent_context_code:
            return (
                f"**RNA classifier check:** expression-reference context is concordant "
                f"at the parent level: {comparison_label} is top; refined report "
                f"label remains {supplied_label}{suffix}."
            )
        return (
            f"**RNA classifier check:** expression-reference context is concordant with supplied "
            f"{supplied_label}{suffix}."
        )

    fit_quality = str((analysis.get("fit_quality") or {}).get("label") or "").strip()
    top_score = _candidate_support_score(top)
    supplied_score = _candidate_support_score(supplied_row)
    near_tie = (
        top_score is not None
        and supplied_score is not None
        and abs(top_score - supplied_score) <= 0.10
    )
    status = (
        "ambiguous against"
        if fit_quality in {"weak", "ambiguous"} or near_tie
        else "discordant with"
    )
    rank_clause = (
        f"rank {supplied_rank}"
        if supplied_rank is not None
        else "not in the RNA top candidates"
    )
    caveat_clause = _confidence_caveat_clause(call_tier)
    compatibility = _broad_context_compatibility(top_code, supplied_code)
    if compatibility:
        alternatives = _candidate_code_list(
            candidate_trace,
            exclude={top_code, comparison_code},
            limit=2,
        )
        alt_clause = (
            f"; nearest expression-reference alternatives: {alternatives}"
            if alternatives
            else ""
        )
        return (
            f"**RNA classifier check:** expression-reference context is {top_label}, giving "
            f"{compatibility} for supplied {supplied_label}; the expression-reference classifier "
            f"does not independently resolve the refined label{alt_clause}. "
            f"Keep {supplied_label} as the report label."
        )
    return (
        f"**RNA classifier check:** expression-reference context is {status} supplied "
        f"{supplied_label}; top expression-reference match is {top_label or 'unresolved'} "
        f"while {comparison_label} is {rank_clause}. "
        "Keep the supplied label as the report label and review pathology/subtype context"
        f"{caveat_clause}."
    )


def _signature_top_code(analysis) -> str:
    candidates = analysis.get("signature_top_cancers") or []
    if not candidates:
        return ""
    top = candidates[0]
    if isinstance(top, dict):
        return str(top.get("code") or top.get("cancer_type") or "").strip()
    if isinstance(top, (list, tuple)) and top:
        return str(top[0] or "").strip()
    return ""


def _rna_alternatives_line(analysis, cancer_code: str) -> str:
    """Concise ordered alternatives for RNA-inferred, non-rare report scopes."""
    constraints = analysis.get("analysis_constraints") or {}
    source = str(analysis.get("cancer_type_source") or "").strip()
    if constraints.get("cancer_type") or source == "user-specified":
        return ""
    if analysis.get("rare_report_scope_inference") or analysis.get(
        "fusion_report_scope_inference"
    ):
        return ""
    candidate_trace = analysis.get("candidate_trace") or []
    if not candidate_trace:
        return ""

    top_score = _candidate_support_score(candidate_trace[0])
    chunks: list[str] = []
    for idx, row in enumerate(candidate_trace[:3], start=1):
        code = str(row.get("code") or "").strip()
        if not code:
            continue
        score = _candidate_support_score(row)
        ratio = ""
        if idx > 1 and top_score and score is not None:
            ratio = f", {score / top_score:.2f}x top support"
        chunks.append(f"{code} (rank {idx}{ratio})")
    if not chunks:
        return ""

    sig_top = _signature_top_code(analysis)
    sig_clause = ""
    if sig_top and sig_top != str(cancer_code or "").strip():
        sig_clause = f"; raw-signature top {sig_top}"
    return (
        f"**RNA alternatives:** ordered RNA candidates {', '.join(chunks)}"
        f"{sig_clause}. Treat these as hypotheses until pathology/clinical "
        "context resolves them."
    )


def _subtype_status_line(
    *,
    winning_subtype: Optional[str],
    degenerate_status: Optional[str],
    degenerate_resolution: Optional[dict],
    original_winning_subtype: Optional[str],
    analysis: dict,
) -> str:
    if degenerate_status in ("corrected", "degenerate"):
        subtype_note = _render_subtype_note(
            degenerate_resolution or {},
            original_subtype=original_winning_subtype,
            site_template=analysis_site_template_for_subtype(analysis),
        ).strip()
        if subtype_note:
            if degenerate_status == "corrected":
                final_label = _display_subtype_code(
                    (degenerate_resolution or {}).get("final_subtype")
                )
                return (
                    f"**Subtype status:** {final_label}-consistent after context "
                    f"check; {subtype_note}"
                )
            return f"**Subtype status:** {subtype_note}"
    if winning_subtype:
        label = _display_subtype_code(winning_subtype)
        return (
            f"**Subtype status:** RNA subtype signal is {label}-consistent; "
            "use as expression context unless clinically confirmed."
        )
    return ""


def _clinical_context_caveats(analysis) -> List[str]:
    constraints = analysis.get("analysis_constraints") or {}
    source = str(analysis.get("cancer_type_source") or "").strip()
    caveats: List[str] = []
    if not constraints.get("cancer_type") and source != "user-specified":
        caveats.append(
            "Cancer type is RNA-inferred — treat it as a hypothesis, not a diagnosis."
        )
    caveats.append(
        "Patient-facing LLM interpretation needs external clinical context: "
        "diagnosis, stage, prior lines, current medications, MSI/MMR/TMB, "
        "mutations/fusions/CNVs, relevant imaging such as HER2/PSMA, and trial availability."
    )
    return caveats


def _missing_hla_prompts(targets_df, ranges_df, analysis, limit: int = 3) -> List[str]:
    constraints = analysis.get("analysis_constraints") or {}
    if constraints.get("hla_types") or targets_df is None or ranges_df is None:
        return []
    # ID-based lookup: convert target symbols → Ensembl IDs once, then
    # match against the ID-keyed view of ranges_df. Avoids symbol-alias
    # ambiguity AND avoids the Series.__finalize__ deepcopy hot path.
    # Falls back to symbol-keyed lookup when a ranges_df row lacks a
    # gene_id (legacy test fixtures, synthetic frames).
    from .common import (
        ranges_by_gene_id,
        ranges_by_symbol,
        panel_symbols_to_gene_ids,
    )
    target_symbols = {
        str(t.get("symbol") or "").strip()
        for t in targets_df.to_dict("records")
    }
    target_symbols.discard("")
    sym_to_id = panel_symbols_to_gene_ids(target_symbols)
    id_to_row = ranges_by_gene_id(ranges_df)
    sym_to_row = ranges_by_symbol(ranges_df)
    prompts: List[str] = []
    seen = set()
    for target in targets_df.to_dict("records"):
        if _subtype_specific_row_out_of_scope(target, analysis):
            continue
        required = hla_restrictions_for_target_row(target)
        if not required:
            continue
        sym = str(target.get("symbol") or "").strip()
        if not sym or sym.lower() == "nan":
            continue
        gene_id = sym_to_id.get(sym)
        expr = id_to_row.get(gene_id) if gene_id else None
        if expr is None:
            # Fallback to symbol lookup for legacy ranges_df frames
            # that don't carry the gene_id column.
            expr = sym_to_row.get(sym) or sym_to_row.get(sym.replace("-", ""))
        if expr is None:
            continue
        observed = _brief_float(expr.get("observed_tpm"), 0.0)
        tumor_tpm = _brief_float(expr.get("attr_tumor_tpm"), observed)
        if max(observed, tumor_tpm) < 1.0:
            continue
        agent = str(target.get("agent") or "the HLA-restricted therapy").strip()
        key = (sym, agent)
        if key in seen:
            continue
        seen.add(key)
        prompts.append(
            f"HLA typing needed for {agent} ({sym}): requires "
            f"{'/'.join(required)}; if compatible, review eligibility alongside "
            "target expression, diagnosis, and trial/label criteria."
        )
        if len(prompts) >= limit:
            break
    return prompts


# Thresholds for the "Notable expression outliers" summary block.
# Amplification fold = observed / max-healthy-tpm (see
# ``plot_tumor_expr.estimate_tumor_expression_ranges``). 10× is the
# clinically-meaningful threshold for "this is amplified vs everywhere
# the gene normally lives". TPM ≥ 50 prevents low-magnitude noise (a
# gene at 0.6 TPM with 12× over peak-healthy = 0.05 TPM is mathematically
# amplified but clinically irrelevant).
_OUTLIER_MIN_AMPLIFICATION_FOLD = 10.0
_OUTLIER_MIN_OBSERVED_TPM = 50.0
_OUTLIER_HIGH_PERCENTILE = 0.95


def _notable_biomarker_outliers(
    ranges_df,
    panel_code: Optional[str],
    panel_subtype: Optional[str],
    *,
    excluded_symbols: Optional[set] = None,
    top_n: int = 4,
):
    """Surface biomarker-panel genes that are amplified or top-percentile.

    Mirrors the gating in the curated key-genes biomarker panel rendered
    by the analysis report but lifts the qualifying genes into the
    summary so the headline reflects them (MDM2 ~38× in OS, NUTM1 ~44×
    in NUTM, CDK4/RUNX2 amplicons, etc.). Returns a list of dicts
    ``{symbol, observed_tpm, amplification_fold, tcga_percentile}``.

    ``excluded_symbols`` lets the caller suppress genes that are already
    listed in the top-therapy block so the summary doesn't repeat the
    same finding under two headers.
    """
    if ranges_df is None or len(ranges_df) == 0 or not panel_code:
        return []
    try:
        from pirlygenes.gene_sets_cancer import cancer_biomarker_genes

        biomarker_syms = (
            cancer_biomarker_genes(panel_code, subtype=panel_subtype)
            if panel_subtype
            else cancer_biomarker_genes(panel_code)
        )
    except (ImportError, KeyError, ValueError, TypeError):
        return []
    if not biomarker_syms:
        return []
    excluded = {str(s) for s in (excluded_symbols or set())}
    biomarker_set = {str(s) for s in biomarker_syms}
    # Boundary conversion: panel comes in as symbols; resolve to
    # canonical Ensembl IDs once, then do all internal matching by ID.
    # This eliminates the HLA-A/HLAA / synonym-collision ambiguity that
    # symbol-set membership tests inherit.
    from .common import (
        ranges_records,
        panel_symbols_to_gene_ids,
        _versionless_gene_id,
    )
    panel_ids = panel_symbols_to_gene_ids(biomarker_set - excluded)
    panel_id_set = set(panel_ids.values())
    panel_sym_fallback = biomarker_set - excluded  # for rows w/o gene_id
    candidates: list[dict] = []
    for row in ranges_records(ranges_df):
        gene_id = _versionless_gene_id(row.get("gene_id"))
        row_sym = str(row.get("symbol") or "")
        if gene_id:
            if gene_id not in panel_id_set:
                continue
        elif row_sym not in panel_sym_fallback:
            continue
        observed = float(row.get("observed_tpm") or 0.0)
        amp = float(row.get("amplification_fold") or 0.0)
        pct = float(row.get("tcga_percentile") or 0.0)
        if observed < _OUTLIER_MIN_OBSERVED_TPM:
            continue
        if (
            amp < _OUTLIER_MIN_AMPLIFICATION_FOLD
            and pct < _OUTLIER_HIGH_PERCENTILE
        ):
            continue
        candidates.append(
            {
                "symbol": str(row.get("symbol") or ""),
                "gene_id": gene_id,
                "observed_tpm": observed,
                "amplification_fold": amp,
                "tcga_percentile": pct,
            }
        )
    # Rank: prefer the most amplified, break ties by TPM. Both signals
    # are clinically meaningful — amplification flags drug-targetability
    # (MDM2 inhibitors, CDK4/6 inhibitors), absolute TPM ties together
    # the eligibility-band reading.
    candidates.sort(
        key=lambda c: (c["amplification_fold"], c["observed_tpm"]),
        reverse=True,
    )
    return candidates[:top_n]


def _format_biomarker_outlier_bullet(row: dict) -> str:
    sym = row["symbol"]
    obs = row["observed_tpm"]
    amp = row["amplification_fold"]
    pct = row["tcga_percentile"]
    parts: list[str] = [f"{obs:.0f} TPM"]
    if amp >= _OUTLIER_MIN_AMPLIFICATION_FOLD:
        parts.append(f"amplified {amp:.1f}× over peak healthy tissue")
    if pct >= _OUTLIER_HIGH_PERCENTILE:
        parts.append(f"TCGA cohort {pct * 100:.0f}th percentile")
    return f"- **{sym}** — " + "; ".join(parts) + " (biomarker panel)"


# CTA threshold for the "Notable CTAs" summary block. Matches the
# clinical convention that ≥ 10 TPM is the lower bound for vaccine /
# TCR-T / engineered-cell consideration; ≥ 100 TPM is the band where
# CTAs are commonly trial-eligible.
_CTA_MIN_OBSERVED_TPM = 10.0


def _notable_cta_outliers(ranges_df, *, top_n: int = 3):
    """Surface top CTAs by observed TPM (vaccine / TCR-T-relevant).

    CTAs are flagged on each ``ranges_df`` row via ``is_cta`` from
    ``estimate_tumor_expression_ranges``; the row already incorporates
    a tumor-attribution context. Selection is intentionally simple —
    threshold by TPM and rank by TPM — because the read-out the reader
    cares about ("is this CTA actually highly expressed?") is direct.
    """
    if ranges_df is None or len(ranges_df) == 0:
        return []
    rows: list[dict] = []
    from .common import ranges_records
    for row in ranges_records(ranges_df):
        if not bool(row.get("is_cta")):
            continue
        observed = float(row.get("observed_tpm") or 0.0)
        if observed < _CTA_MIN_OBSERVED_TPM:
            continue
        rows.append(
            {
                "symbol": str(row.get("symbol") or ""),
                "observed_tpm": observed,
                "tcga_percentile": float(row.get("tcga_percentile") or 0.0),
            }
        )
    rows.sort(key=lambda r: r["observed_tpm"], reverse=True)
    return rows[:top_n]


def _format_cta_outlier_bullet(row: dict) -> str:
    sym = row["symbol"]
    obs = row["observed_tpm"]
    pct = row["tcga_percentile"]
    parts: list[str] = [f"{obs:.0f} TPM"]
    if pct >= _OUTLIER_HIGH_PERCENTILE:
        parts.append(f"TCGA cohort {pct * 100:.0f}th percentile")
    return f"- **{sym}** — " + "; ".join(parts) + " (CTA — vaccine / TCR-T)"


def _empty_therapy_shortlist_message(targets_df, ranges_df) -> str:
    """Differentiated message when the top-therapy block is empty.

    The original single line ("No approved or trialed agents with a
    measured, tumor-supported target") collapsed three clinically
    distinct situations into one wording:

      (a) most curated targets are in the input but expression-suppressed
          (real biological negative — e.g. ERBB2 = 0 TPM on a TNBC line);
      (b) curated targets are not present in the input file at all
          (RNA-seq coverage gap — symbol-mapping / pipeline issue, not
          biology);
      (c) curated targets are HLA-restricted or subtype-locked out.

    Surface the actual distribution so the clinician knows which kind
    of "no shortlist" they're reading.
    """
    if (
        targets_df is None
        or len(targets_df) == 0
        or ranges_df is None
    ):
        return (
            "*No curated agents available for this cancer type — see the "
            "full Therapy Landscape table for raw expression rankings.*\n"
        )
    from .common import (
        ranges_by_gene_id,
        ranges_by_symbol,
        panel_symbols_to_gene_ids,
    )
    target_records = targets_df.to_dict("records")
    sym_to_id = panel_symbols_to_gene_ids(
        str(t.get("symbol") or "").strip() for t in target_records
    )
    id_to_row = ranges_by_gene_id(ranges_df)
    sym_to_row = ranges_by_symbol(ranges_df)  # fallback for ID-less frames
    input_syms = getattr(ranges_df, "attrs", {}).get(
        "sample_input_symbols"
    ) or set()
    n_total = 0
    n_in_input_low = 0
    n_in_input_present = 0
    n_not_in_input = 0
    for t in target_records:
        sym = str(t.get("symbol") or "")
        if not sym or sym == "—":
            continue
        n_total += 1
        gene_id = sym_to_id.get(sym.strip())
        expr = id_to_row.get(gene_id) if gene_id else None
        if expr is None:
            expr = sym_to_row.get(sym)
        if expr is not None:
            observed = float(expr.get("observed_tpm") or 0.0)
            if observed >= 1.0:
                n_in_input_present += 1
            else:
                n_in_input_low += 1
        elif input_syms and sym in input_syms:
            n_in_input_low += 1
        elif input_syms:
            n_not_in_input += 1
        else:
            # Legacy ranges_df with no input-symbol attrs — can't
            # disambiguate (a) vs (b).
            n_in_input_low += 1
    if n_total == 0:
        return (
            "*No curated agents available for this cancer type — see the "
            "full Therapy Landscape table for raw expression rankings.*\n"
        )
    parts: list[str] = []
    if n_in_input_present:
        parts.append(
            f"{n_in_input_present} measured and present but filtered as "
            "non-tumor-supported"
        )
    if n_in_input_low:
        parts.append(
            f"{n_in_input_low} measured below detection (real RNA-level "
            "negative for the target)"
        )
    if n_not_in_input:
        parts.append(
            f"{n_not_in_input} not present in input file (coverage gap, "
            "investigate symbol mapping)"
        )
    body = "; ".join(parts) if parts else "no qualifying rows"
    return (
        f"*Therapy shortlist is empty: of {n_total} curated agents, "
        f"{body}. See the full Therapy Landscape table for details.*\n"
    )


def build_summary(
    analysis,
    ranges_df,
    cancer_code: str,
    disease_state: str,
    sample_id: Optional[str] = None,
) -> str:
    """Return the one-page ``*-summary.md`` content (≤ 40 lines).

    Audience: clinician skimming before a tumor board; LLM asked for a
    short referral-note paragraph. Strict structure; no internal
    jargon. (Named ``build_brief`` through 4.40; the legacy name is
    still exported as an alias.)
    """
    from pirlygenes.gene_sets_cancer import cancer_key_genes_cancer_types

    purity = analysis.get("purity") or {}
    purity_tier = analysis.get("purity_confidence")
    sample_context = analysis.get("sample_context")
    cancer_name = analysis.get("cancer_name") or cancer_code

    lines: List[str] = []
    sample_id = _display_sample_id(sample_id)
    header_id = f": {sample_id}" if sample_id else ""
    lines.append(f"# Summary{header_id}\n")

    # #149: tissue-composition banner. Above the cancer call so
    # the reader sees the caveat before anchoring on the cancer label.
    # Banner decision reads downstream tumor evidence (purity from
    # tumor purity and signature score so a confident cancer call
    # doesn't trigger a spurious tissue-composition warning.
    hvt = analysis.get("healthy_vs_tumor")
    if hvt is not None:
        banner = hvt.brief_banner(
            purity=purity.get("overall_estimate") if purity else None,
            signature_score=_top_candidate_signature_score(analysis),
        )
        if banner:
            lines.append(banner)
            lines.append("")

    # Cancer call — annotated with #169 contested-call confidence when
    # orthogonal signals (lineage concordance, runner-up gap, tissue-composition
    # top-ρ cohort) disagree with the classifier's pick.
    from .confidence import compute_call_confidence

    call_tier = compute_call_confidence(analysis)
    suffix = _call_confidence_suffix(
        call_tier,
        concise=True,
        include_reasons=False,
    )
    rare_scope = analysis.get("rare_report_scope_inference") or {}
    fusion_scope = analysis.get("fusion_report_scope_inference") or {}
    if rare_scope or fusion_scope:
        tier = getattr(call_tier, "tier", "unknown")
        suffix = f" — **{tier} confidence**"

    # #171/#198: resolve subtype evidence separately from the report-scope
    # cancer label. The subtype signal is useful context, but rendering it
    # inside the cancer-call parenthetical made clinical labels, RNA labels,
    # and confidence caveats look like one decision.
    winning_subtype = candidate_winning_subtype_for_analysis(analysis)

    # #198: before rendering, consult the degenerate-subtype registry.
    # Several within-family subtypes share a gene signature (OS vs DDLPS
    # both carry 12q13-15 amplicon; Ewing vs DSRCT vs ARMS all CD99+)
    # and need a site or fusion-surrogate tiebreaker. The resolver
    # reasons over the full sample context — decomposition site
    # template AND the complete tumor-attributed TPM dict — so that
    # decisions aren't made from single-gene lookups in isolation.
    # Activation-signature gating ensures pairs only fire when the
    # shared signature is actually present; high-confidence clear-
    # winner calls bypass the resolver entirely.
    degenerate_status = None
    degenerate_resolution = None
    original_winning_subtype = winning_subtype
    if winning_subtype:
        try:
            from .degenerate_subtype import resolve_degenerate_subtype

            site_template = analysis_site_template_for_subtype(analysis)
            tumor_tpm_by_symbol = analysis.get("tumor_tpm_by_symbol")
            if not tumor_tpm_by_symbol and ranges_df is not None:
                # Build from ``ranges_df`` — the per-gene attribution
                # stage's output. Propagates the full tumor-attributed
                # TPM context so activation signatures and multi-gene
                # tiebreakers get the full evidence, not a single-gene
                # slice.
                try:
                    import pandas as pd

                    if (
                        isinstance(ranges_df, pd.DataFrame)
                        and "symbol" in ranges_df.columns
                        and "attr_tumor_tpm" in ranges_df.columns
                    ):
                        tumor_tpm_by_symbol = dict(
                            zip(
                                ranges_df["symbol"].astype(str),
                                pd.to_numeric(
                                    ranges_df["attr_tumor_tpm"],
                                    errors="coerce",
                                )
                                .fillna(0.0)
                                .astype(float),
                            )
                        )
                except Exception:
                    logger.debug(
                        "degenerate-subtype: failed to build tumor_tpm_by_symbol from ranges_df",
                        exc_info=True,
                    )
                    tumor_tpm_by_symbol = None
            resolution = resolve_degenerate_subtype(
                winning_subtype,
                site_template=site_template,
                tumor_tpm_by_symbol=tumor_tpm_by_symbol,
            )
            degenerate_resolution = resolution
            if resolution["status"] == "corrected":
                winning_subtype = resolution["final_subtype"]
            degenerate_status = resolution["status"]
        except Exception:
            logger.debug(
                "degenerate-subtype resolution failed; keeping classifier pick",
                exc_info=True,
            )

    call_punctuation = suffix or "."
    lines.append(f"**Cancer call:** {cancer_code} ({cancer_name}){call_punctuation}")
    lines.append(_cancer_type_basis_line(analysis, cancer_code))
    panel_line = _lineage_panel_evidence_line(analysis, cancer_code)
    if panel_line:
        lines.append(panel_line)
    subtype_line = _lineage_panel_subtype_reasoning_line(analysis, cancer_code)
    if subtype_line:
        lines.append(subtype_line)
    rna_crosscheck = _rna_crosscheck_line(analysis, cancer_code, call_tier=call_tier)
    if rna_crosscheck:
        lines.append(rna_crosscheck)
    else:
        rna_alternatives = _rna_alternatives_line(analysis, cancer_code)
        if rna_alternatives:
            lines.append(rna_alternatives)
    fusion_line = _fusion_evidence_line(analysis, cancer_code)
    if fusion_line:
        lines.append(fusion_line)
    alteration_line = _alteration_evidence_line(analysis)
    if alteration_line:
        lines.append(alteration_line)
    lateral_rare_prompts_are_summary_level = not (
        fusion_line
        or alteration_line
        or analysis.get("rare_report_scope_inference")
        or analysis.get("cancer_type_source") == "user-specified"
    )
    rare_marker_hypotheses = (
        [
            finding
            for finding in (analysis.get("rare_marker_hypotheses") or [])
            if str(finding.get("cancer_type") or "").strip()
            != str(cancer_code).strip()
        ]
        if lateral_rare_prompts_are_summary_level
        else []
    )
    if rare_marker_hypotheses:
        finding = rare_marker_hypotheses[0]
        surrogate = str(finding.get("surrogate") or "marker").strip()
        tpm = finding.get("surrogate_tpm")
        tpm_clause = f" {tpm:g} TPM" if isinstance(tpm, (int, float)) else ""
        label = finding.get("cancer_type") or "rare cancer"
        support = ", ".join(finding.get("support_genes") or [])
        missing = ", ".join(finding.get("missing_support_genes") or [])
        evidence_bits = []
        if support:
            evidence_bits.append(f"supporting co-markers: {support}")
        if missing:
            evidence_bits.append(f"missing/low expected co-markers: {missing}")
        evidence_clause = "; " + "; ".join(evidence_bits) if evidence_bits else ""
        top_ref = str(finding.get("top_reference_cancer_type") or "").strip()
        context_clause = f" in {top_ref} RNA context" if top_ref else ""
        lines.append(
            f"**Rare-marker prompt:** {surrogate}{tpm_clause}{context_clause} raises {label} as a "
            f"testing prompt, not the report scope{evidence_clause}."
        )
    subtype_line = _subtype_status_line(
        winning_subtype=winning_subtype,
        degenerate_status=degenerate_status,
        degenerate_resolution=degenerate_resolution,
        original_winning_subtype=original_winning_subtype,
        analysis=analysis,
    )
    if subtype_line:
        lines.append(subtype_line)

    # Purity
    overall = purity.get("overall_estimate")
    lower = purity.get("overall_lower")
    upper = purity.get("overall_upper")
    if overall is not None and lower is not None and upper is not None:
        tier_label = (
            getattr(purity_tier, "tier", "unknown") if purity_tier else "unknown"
        )
        lines.append(
            f"**Purity:** {overall:.0%} (model interval {lower:.0%}–{upper:.0%}, "
            f"{tier_label} confidence)."
        )
    rescue_line = _cancer_call_rescue_summary_line(analysis)
    if rescue_line:
        lines.append(rescue_line)
    inferred_site_line = _inferred_site_context_line(analysis)
    if inferred_site_line:
        lines.append(inferred_site_line)

    # Sample context
    if sample_context is not None:
        prep_label = library_prep_clause(
            getattr(sample_context, "library_prep", "unknown")
        )
        pres_label = str(getattr(sample_context, "preservation", "unknown")).replace(
            "_", " "
        )
        pres_conf = getattr(sample_context, "preservation_confidence", 0.0)
        if pres_label == "fresh frozen":
            pres_label = "fresh/frozen-like"
        lines.append(
            f"**Sample:** {prep_label}; preservation inferred as {pres_label} "
            f"from RNA QC ({heuristic_support_label(pres_conf)})."
        )
    rna_qc_line = rna_quant_qc_summary_line(analysis.get("rna_quant_qc"))
    if rna_qc_line:
        lines.append(rna_qc_line)
    rescue_line = expression_qc_rescue_summary_line(
        analysis.get("expression_qc_rescue")
    )
    if rescue_line:
        lines.append(rescue_line)
    scale_qc = analysis.get("expression_scale_qc") or {}
    if scale_qc.get("converted_from") == "log2_tpm_plus_one":
        post_sum = scale_qc.get("post_conversion_sum_tpm") or scale_qc.get("sum_tpm")
        sum_clause = f"; post-conversion sum {post_sum/1_000_000:.2f}M" if post_sum else ""
        lines.append(
            "**Expression scale QC:** input resembled log2(TPM+1); converted to "
            f"linear TPM before interpretation{sum_clause}."
        )
    elif scale_qc.get("warnings"):
        lines.append(
            "**Expression scale QC:** "
            + str((scale_qc.get("warnings") or ["check expression scale"])[0])
            + "."
        )

    # Disease/pathway state comes after assay/QC framing so readers know how
    # much weight to put on RNA-derived biology before acting on it.
    disease_state_display = report_disease_state_text(disease_state, analysis=analysis)
    lines.extend(_disease_state_summary_lines(disease_state_display))
    pathway_activity = _pathway_activity_line(analysis)
    if pathway_activity:
        lines.append(pathway_activity)

    lines.append("")

    # Top therapies — subtype/direct-code resolved when the umbrella
    # cancer call narrows onto a more specific curated panel.
    panel_code, panel_subtype, targets_df = _curated_target_panel_for_sample(
        cancer_code,
        analysis,
        ranges_df=ranges_df,
    )
    hla_prompts = _missing_hla_prompts(targets_df, ranges_df, analysis)
    if panel_code in cancer_key_genes_cancer_types():
        therapy_analysis = dict(analysis)
        if panel_subtype:
            therapy_analysis["_target_panel_subtype"] = panel_subtype
        top = _top_therapies(
            targets_df,
            ranges_df,
            limit=3,
            analysis=therapy_analysis,
            disease_state=disease_state_display,
        )
        lines.append("## Top candidate therapies\n")
        lines.append(
            "*Static curation, not a live NCCN or trial-matching engine; ranked by "
            "treatment-path maturity first, then tumor-source support. Verify current "
            "NCCN/trial status and current therapy before acting on any row.*\n"
        )
        if panel_code != cancer_code or panel_subtype:
            lines.append(
                "*Subtype-resolved therapy curation:* "
                + subtype_curation_scope_note(
                    panel_code,
                    panel_subtype=panel_subtype,
                    base_code=cancer_code,
                    base_name=analysis.get("cancer_name") or cancer_code,
                    noun="therapy evidence",
                )
                + "\n"
            )
        if top:
            for target_row, expression_row in top:
                lines.append(
                    _format_therapy_bullet(
                        target_row,
                        expression_row,
                        target_panel=targets_df,
                        analysis=therapy_analysis,
                        disease_state=disease_state_display,
                        ranges_df=ranges_df,
                    )
                )
            omission_note = _shortlist_omission_note(targets_df, ranges_df, top)
            if omission_note:
                lines.append("")
                lines.append(omission_note)
            lines.append("")
        else:
            lines.append(
                _empty_therapy_shortlist_message(targets_df, ranges_df)
            )
    else:
        lines.append(
            f"## Top candidate therapies\n"
            f"*Cancer type {cancer_code} is not yet in the curated "
            "key-genes panel — see the full tables below for a raw "
            "expression ranking.*\n"
        )

    # Notable biomarker-panel outliers (amplified / top-percentile).
    # Surfaces curated biomarker-panel genes that the therapy-shortlist
    # filter excludes (no registered agent, or filtered as non-tumor-
    # supported). These are clinically important signals the headline
    # should not bury: MDM2 ~38× in OS, CDK4 amplicons, NUTM1 ~44× in
    # NUT carcinoma, etc.
    if panel_code in cancer_key_genes_cancer_types():
        top_therapy_symbols = {
            str((target_row.get("symbol") or "")).strip()
            for target_row, _ in (top or [])
        }
        outliers = _notable_biomarker_outliers(
            ranges_df,
            panel_code,
            panel_subtype,
            excluded_symbols=top_therapy_symbols,
        )
        if outliers:
            lines.append("## Notable biomarker outliers\n")
            lines.append(
                "*Curated biomarker-panel genes outside the therapy "
                "shortlist that are amplified vs peak healthy tissue or "
                "in the top 5% of TCGA cohort expression. Driver / "
                "lineage / amplicon signals — see the analysis report "
                "for full biomarker-panel context.*\n"
            )
            for row in outliers:
                lines.append(_format_biomarker_outlier_bullet(row))
            lines.append("")

    # Notable CTAs (cancer-testis antigens). Vaccine / TCR-T-relevant
    # surface signals that are independent of the curated therapy
    # registry. PAGE5 in OS, HORMAD1 in NUTM, PRAME in melanoma are
    # the canonical examples; the registry-gated shortlist often
    # omits them because no FDA-approved CTA-targeting agent exists
    # for that exact cancer type, even though TCR-T trials do.
    cta_outliers = _notable_cta_outliers(ranges_df)
    if cta_outliers:
        lines.append("## Notable CTAs\n")
        lines.append(
            "*Cancer-testis antigens expressed above the vaccine / TCR-T "
            "consideration threshold (≥ 10 TPM). Independent of the "
            "approved-therapy registry — see the CTA table in the "
            "analysis report for HLA / immunogenicity context.*\n"
        )
        for row in cta_outliers:
            lines.append(_format_cta_outlier_bullet(row))
        lines.append("")

    # Caveats
    caveats = _caveats_from_purity_tier(
        purity_tier,
        sample_context,
        analysis,
    ) + hla_prompts + _clinical_context_caveats(analysis)
    if caveats:
        lines.append("## Caveats")
        for c in caveats:
            lines.append(f"- {c}")
        lines.append("")

    lines.append(
        "*Full detail: see the accompanying `*-analysis.md` and `*-evidence.md`.*"
    )

    return "\n".join(lines)


# Back-compat alias — ``build_brief`` was the public name through 4.40;
# removed in 5.0. External importers should migrate to ``build_summary``.
build_brief = build_summary


def build_actionable(
    analysis,
    ranges_df,
    cancer_code: str,
    disease_state: str,
    sample_id: Optional[str] = None,
) -> str:
    """Return the longer ``*-actionable.md`` content (~2-3 pages).

    Audience: oncologist preparing a treatment discussion, or LLM
    asked to draft a clinical summary. Full structure with the
    biomarker panel and therapy landscape inline; no pipeline-
    internal jargon.
    """
    from pirlygenes.gene_sets_cancer import cancer_key_genes_cancer_types

    purity = analysis.get("purity") or {}
    purity_tier = analysis.get("purity_confidence")
    sample_context = analysis.get("sample_context")
    cancer_name = analysis.get("cancer_name") or cancer_code

    lines: List[str] = []
    sample_id = _display_sample_id(sample_id)
    header_id = f" — {sample_id}" if sample_id else ""
    lines.append(f"# Actionable review{header_id}\n")
    lines.append(
        "<!-- Audience: oncologist preparing a treatment discussion; "
        "molecular tumor board member reading carefully. -->"
    )
    lines.append("")

    # Sample + confidence paragraph
    lines.append("## Sample and confidence\n")
    prep_label = (
        library_prep_clause(getattr(sample_context, "library_prep", "unknown"))
        if sample_context
        else "unknown"
    )
    pres_label = (
        str(getattr(sample_context, "preservation", "unknown")).replace("_", " ")
        if sample_context
        else "unknown"
    )
    if sample_context:
        lines.append(
            f"Input: **{prep_label}**, **{pres_label}** "
            "preservation. " + _preservation_clinical_clause(sample_context)
        )
    rna_qc_line = rna_quant_qc_summary_line(analysis.get("rna_quant_qc"))
    if rna_qc_line:
        lines.append("\n" + rna_qc_line)
    rescue_line = expression_qc_rescue_summary_line(
        analysis.get("expression_qc_rescue")
    )
    if rescue_line:
        lines.append("\n" + rescue_line)

    overall = purity.get("overall_estimate")
    lower = purity.get("overall_lower")
    upper = purity.get("overall_upper")
    tier_label = getattr(purity_tier, "tier", "unknown") if purity_tier else "unknown"
    tier_reasons = getattr(purity_tier, "reasons", []) if purity_tier else []
    if overall is not None:
        confidence_clause = f"**{tier_label}** confidence"
        if tier_reasons and tier_label in {"low", "moderate"}:
            confidence_clause += " (" + "; ".join(tier_reasons) + ")"
        lines.append(
            f"\nPurity point estimate: **{overall:.0%}** "
            f"(model interval {lower:.0%}–{upper:.0%}). {confidence_clause.capitalize()}."
        )

    lines.append("")

    # Cancer call + disease state
    lines.append("## Cancer call and disease state\n")
    from .confidence import compute_call_confidence

    call_tier = compute_call_confidence(analysis)
    call_suffix = _call_confidence_suffix(call_tier, concise=True)
    call_punctuation = call_suffix or "."
    lines.append(f"Working call: **{cancer_code}** ({cancer_name}){call_punctuation}")
    rescue_line = _cancer_call_rescue_actionable_line(analysis)
    if rescue_line:
        lines.append(rescue_line)
    inferred_site_line = _inferred_site_context_line(analysis)
    if inferred_site_line:
        lines.append("\n" + inferred_site_line)
    basis_line = _cancer_type_basis_line(analysis, cancer_code)
    rna_crosscheck = _rna_crosscheck_line(analysis, cancer_code)
    panel_line = _lineage_panel_evidence_line(analysis, cancer_code)
    subtype_line = _lineage_panel_subtype_reasoning_line(analysis, cancer_code)
    if basis_line:
        lines.append(f"\n{basis_line}")
    if panel_line:
        lines.append(f"\n{panel_line}")
    if subtype_line:
        lines.append(f"\n{subtype_line}")
    if rna_crosscheck:
        lines.append(f"\n{rna_crosscheck}")
    fusion_line = _fusion_evidence_line(analysis, cancer_code)
    if fusion_line:
        lines.append(f"\n{fusion_line}")
    # Tissue-composition banner (if non-tumor-consistent) so
    # an actionable reader sees the caveat attached to the
    # working call, not buried in the summary. Same evidence-gated
    # logic as the brief — strong tumor signal suppresses the banner.
    hvt = analysis.get("healthy_vs_tumor")
    if hvt is not None:
        banner = hvt.brief_banner(
            purity=purity.get("overall_estimate") if purity else None,
            signature_score=_top_candidate_signature_score(analysis),
        )
        if banner:
            lines.append(f"\n{banner}")
    disease_state_display = report_disease_state_text(disease_state, analysis=analysis)
    if disease_state_display:
        lines.append(f"\n{disease_state_display}")
    lines.append("")

    # Therapy prioritization.
    panel_code, panel_subtype, targets_df = _curated_target_panel_for_sample(
        cancer_code,
        analysis,
        ranges_df=ranges_df,
    )
    hla_prompts = _missing_hla_prompts(targets_df, ranges_df, analysis)
    panel_label = _panel_display_label(panel_code, panel_subtype)
    if panel_code in cancer_key_genes_cancer_types():
        from .common import ranges_by_symbol
        sym_to_row = ranges_by_symbol(ranges_df)

        if len(targets_df):
            lines.append("## Therapy Prioritization\n")
            if panel_code != cancer_code or panel_subtype:
                lines.append(
                    "*Subtype-resolved therapy curation:* "
                    + subtype_curation_scope_note(
                        panel_code,
                        panel_subtype=panel_subtype,
                        base_code=cancer_code,
                        base_name=cancer_name or cancer_code,
                        noun="therapy evidence",
                    )
                    + "\n"
                )
            lines.append(
                "Agents with an approved or trialed indication for "
                f"{cancer_code_display_name(panel_code, panel_label)}, cross-referenced to this sample. "
                "Approved agents listed first. Interpretation separates "
                "tumor-source support from normal-expression context so "
                "lineage markers are not confused with tumor-exclusive "
                "targets. Treatment-path context flags standard options, "
                "later-line requirements, trial follow-ups, and possible "
                "current/prior therapy exposure."
            )
            lines.append(tpm_semantics_note())
            lines.append("")
            lines.append(
                "| Target | Agent | Class | Phase | Indication | "
                "Bulk TPM (measured) | Tumor-source bulk TPM (model) | Context TPM (model) | Interpretation |"
            )
            lines.append(
                "|--------|-------|-------|-------|------------|"
                "----------|-------------------------------|---------------------|----------------|"
            )
            phase_order = {
                "approved": 0,
                "phase_3": 1,
                "phase_2": 2,
                "phase_1": 3,
                "preclinical": 4,
            }
            _target_records = targets_df.to_dict("records")
            sorted_df = targets_df.assign(
                _inactive_key=[
                    1 if _brca_er_dependent_row_inactive(t, analysis) else 0
                    for t in _target_records
                ],
                _path_key=[
                    therapy_path_rank(
                        t,
                        analysis=analysis,
                        disease_state=disease_state_display,
                    )
                    for t in _target_records
                ],
                _po=targets_df["phase"].map(lambda p: phase_order.get(str(p), 99)),
            ).sort_values(["_inactive_key", "_path_key", "_po", "symbol", "agent"])

            def _cell(value):
                """Render a cell, turning NaN / blank / 'nan' into em-dash."""
                if value is None:
                    return "—"
                s = str(value).strip()
                if s == "" or s.lower() == "nan":
                    return "—"
                return s

            for t in sorted_df.to_dict("records"):
                raw_sym = t.get("symbol")
                sym = _cell(raw_sym)
                # Agent-only rows (no gene target — e.g. doxorubicin, pazopanib,
                # trabectedin for sarcoma) have a blank ``symbol``; sym_to_row
                # keying by "nan" would always miss, so skip the lookup and
                # mark as not measurable rather than reporting the TPM of a
                # nonexistent gene.
                if sym == "—":
                    obs_cell = "*not measured*"
                    tumor_source_cell = "—"
                    context_cell = "—"
                    interp_cell = "agent-only / no direct gene target"
                else:
                    expr = sym_to_row.get(sym)
                    if expr is None:
                        obs_state = target_observation_state(sym, ranges_df)
                        obs_cell = format_missing_observation_cell(obs_state)
                        tumor_source_cell = "—"
                        context_cell = "—"
                        if expression_independent_indication(t):
                            interp_cell = (
                                expression_independent_interpretation(t)
                                + "; "
                                + expression_independent_rna_context(None)
                            )
                            gap = _expression_independent_evidence_gap(t, analysis)
                            if gap:
                                interp_cell += "; " + gap
                        else:
                            interp_cell = format_missing_observation_interp(obs_state)
                        path_context = therapy_path_context(
                            t,
                            analysis=analysis,
                            disease_state=disease_state_display,
                        )
                        state_caution = therapy_state_caution(
                            t,
                            analysis=analysis,
                            disease_state=disease_state_display,
                        )
                        extra_parts = []
                        if path_context:
                            extra_parts.append(path_context)
                        if state_caution:
                            extra_parts.append(
                                f"current-therapy check: {state_caution}"
                            )
                        if extra_parts:
                            interp_cell += "; " + "; ".join(extra_parts)
                        if _brca_er_dependent_row_inactive(t, analysis):
                            interp_cell += (
                                "; RNA-context conflict: ER axis is suppressed/"
                                "ER-low; verify ER-positive disease and "
                                "indication-specific eligibility before acting"
                            )
                    else:
                        obs_cell = f"{float(expr.get('observed_tpm') or 0):.1f}"
                        tumor_source_cell = tumor_band_cell(expr)
                        context_cell = context_expression_band_cell(expr)
                        source = tumor_attribution_context(expr)
                        normal = normal_expression_context(expr)
                        expr_independent = expression_independent_indication(t)
                        if expr_independent:
                            interp_parts = [
                                expression_independent_interpretation(t),
                                expression_independent_rna_context(expr),
                                _expression_independent_evidence_gap(t, analysis),
                            ]
                        else:
                            interp_parts = [source["label"], normal["label"]]
                        if _brca_er_dependent_row_inactive(t, analysis):
                            interp_parts.append(
                                "RNA-context conflict: ER axis is suppressed/"
                                "ER-low; verify ER-positive disease and "
                                "indication-specific eligibility before acting"
                            )
                        notes = list(source.get("notes") or []) + list(
                            normal.get("details") or []
                        )
                        if notes and not expr_independent:
                            interp_parts.append(notes[0])
                        path_context = therapy_path_context(
                            t,
                            analysis=analysis,
                            disease_state=disease_state_display,
                        )
                        if path_context:
                            interp_parts.append(path_context)
                        state_caution = therapy_state_caution(
                            t,
                            analysis=analysis,
                            disease_state=disease_state_display,
                        )
                        if state_caution:
                            interp_parts.append(
                                f"current-therapy check: {state_caution}"
                            )
                        agent_meta = agent_metadata_clause(t)
                        if agent_meta:
                            interp_parts.append(agent_meta)
                        interp_parts.append(
                            clinical_maturity_summary(t, target_panel=targets_df)
                        )
                        interp_cell = "; ".join(part for part in interp_parts if part)
                phase = _phase_label(str(t.get("phase") or ""))
                bold = "**" if phase == "Approved" and sym != "—" else ""
                lines.append(
                    f"| {bold}{sym}{bold} | {_cell(t.get('agent'))} | "
                    f"{_cell(t.get('agent_class'))} | {phase} | "
                    f"{_cell(t.get('indication'))} | {obs_cell} | "
                    f"{tumor_source_cell} | {context_cell} | {interp_cell} |"
                )
            lines.append("")
        else:
            lines.append(
                "## Therapy Prioritization\n"
                "*No curated therapy targets are available for this resolved panel.*\n"
            )
    else:
        lines.append(
            "## Therapy Prioritization\n"
            f"*Cancer type {cancer_code} is not yet in the curated "
            "key-genes panel — see `evidence.md` for the generic "
            "expression-ranked tables.*\n"
        )

    # Off-context expressed targets (#47): genes highly expressed in this
    # sample that are validated drug targets in *other* indications but absent
    # from this cancer's curated panel. Surfaced as expression-only leads so a
    # known binder isn't silently dropped just because it's off-label here.
    panel_symbols = set()
    if targets_df is not None and "symbol" in getattr(targets_df, "columns", []):
        panel_symbols = {str(s) for s in targets_df["symbol"]}
    offcontext = offcontext_known_targets(ranges_df, panel_symbols)
    if offcontext:
        lines.append("## Off-Context Expressed Targets\n")
        lines.append(
            "Highly expressed here and a validated drug target in *another* "
            "indication, but not on this cancer's curated panel — "
            "expression-only leads, **not** on-label recommendations; confirm "
            "biology and eligibility before acting.\n"
        )
        for hit in offcontext[:8]:
            primary = hit["indications"][0]
            agent = primary.get("agent") or "binder"
            phase = _phase_label(str(primary.get("phase") or ""))
            other_codes = ", ".join(
                sorted({e["cancer_code"] for e in hit["indications"] if e["cancer_code"]})
            )
            where = other_codes or primary.get("indication") or "another indication"
            lines.append(
                f"- **{hit['symbol']}** — tumor-attributed {hit['tumor_tpm']:.0f} TPM; "
                f"{agent} ({phase}) in {where}."
            )
        lines.append("")

    # Caveats
    caveats = (
        _caveats_from_purity_tier(purity_tier, sample_context, analysis)
        + hla_prompts
    )
    if caveats:
        lines.append("## Caveats\n")
        for c in caveats:
            lines.append(f"- {c}")
        lines.append("")

    lines.append(
        "*See also: `*-analysis.md` (full integrated interpretation) "
        "and `*-evidence.md` (stepwise deduction chain + full target tables).*"
    )
    return "\n".join(lines)


def _preservation_clinical_clause(sample_context) -> str:
    """One-sentence clinician-facing framing of preservation."""
    prep = getattr(sample_context, "library_prep", None)
    preservation = getattr(sample_context, "preservation", None)
    severity = getattr(sample_context, "degradation_severity", "none")
    if preservation == "ffpe" and severity in ("moderate", "severe"):
        return (
            f"FFPE with {severity} degradation — long-transcript "
            "quantification is biased; negative results for long genes "
            "should be interpreted cautiously."
        )
    if prep == "exome_capture":
        return (
            "RNA hybrid-capture / RNA-exome prep selectively enriches "
            "targeted transcripts; rRNA and many non-polyadenylated RNAs "
            "are under-sampled, while low-level MT mRNAs may still be real."
        )
    return ""
