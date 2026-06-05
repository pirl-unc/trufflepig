"""Shared report-facing helpers for evidence and attribution reliability.

These helpers are intentionally narrow: they do not change the raw
attribution math, only how downstream markdown decides whether a row is
safe to summarize as credible tumor-linked signal. The goal is to keep
the headline markdown blocks aligned with the caveats already present in
the tables and TSVs.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import re

from .hla import (
    extract_hla_types_from_text,
    hla_types_compatibility_status,
    parse_hla_types,
)


def _truthy(value) -> bool:
    if value is None:
        return False
    try:
        if value != value:
            return False
    except Exception:
        pass
    try:
        return bool(value)
    except Exception:
        return False


def _clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _safe_float(value, default=0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    if result != result:
        return float(default)
    return result


def _safe_int(value, default=0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


_THERAPY_FILTER_RULES = (
    {
        "cancer_code": "BLCA",
        "symbol": "TACSTD2",
        "agent_contains": "sacituzumab govitecan",
        "indication_contains": ("urothelial", "bladder"),
        "note": (
            "withdrawn urothelial carcinoma indication; do not shortlist as a "
            "current BLCA option"
        ),
    },
    {
        "cancer_code": "SARC_OS",
        "symbol": "IGF1R",
        "agent_contains": "ganitumab",
        "indication_contains": ("os", "osteosarcoma"),
        "note": (
            "suppressed osteosarcoma row; cited ganitumab evidence is not a "
            "current osteosarcoma-specific recommendation"
        ),
    },
)


def therapy_filter_note(target_row) -> str:
    """Return why a stale therapy row should be hidden from reports.

    This is deliberately report-layer filtering. Pirlygenes owns the source
    curation table, but trufflepig should not present known-withdrawn or
    miscited disease-specific rows as active recommendations while the upstream
    release catches up.
    """
    if target_row is None or not hasattr(target_row, "get"):
        return ""
    cancer_code = _clean_text(target_row.get("cancer_code")).upper()
    symbol = _clean_text(target_row.get("symbol")).upper()
    agent = _clean_text(target_row.get("agent")).lower()
    indication = _clean_text(target_row.get("indication")).lower()
    for rule in _THERAPY_FILTER_RULES:
        if cancer_code != rule["cancer_code"]:
            continue
        if symbol != rule["symbol"]:
            continue
        if rule["agent_contains"] not in agent:
            continue
        if not any(term in indication for term in rule["indication_contains"]):
            continue
        return rule["note"]
    return ""


def therapy_withdrawal_note(target_row) -> str:
    """Return only current-market withdrawal notes for stale therapy rows."""
    note = therapy_filter_note(target_row)
    return note if "withdrawn" in note else ""


def filter_current_therapy_targets(targets_df):
    """Drop therapy rows that should not be surfaced as current options."""
    if targets_df is None:
        return None
    try:
        if len(targets_df) == 0:
            return targets_df.reset_index(drop=True)
        keep = [
            not therapy_filter_note(row)
            for row in targets_df.to_dict("records")
        ]
        return targets_df.loc[keep].reset_index(drop=True)
    except Exception:
        return targets_df


@lru_cache(maxsize=1)
def _cancer_registry_display_names():
    try:
        from pirlygenes.gene_sets_cancer import cancer_type_registry

        df = cancer_type_registry()
    except Exception:
        return {}

    mapping = {}
    for _, row in df.iterrows():
        code = _clean_text(row.get("code"))
        if not code:
            continue
        name = _clean_text(row.get("name"))
        subtype_key = _clean_text(row.get("subtype_key"))
        if name:
            mapping[code] = name.split("(")[0].strip()
        elif subtype_key:
            mapping[code] = subtype_key.replace("_", " ").strip()
    return mapping


def cancer_code_display_name(code, fallback=None):
    text = _clean_text(code)
    if not text:
        return _clean_text(fallback) or "this cancer type"
    reg_name = _cancer_registry_display_names().get(text)
    if reg_name:
        return reg_name
    fallback_text = _clean_text(fallback)
    if fallback_text:
        return fallback_text
    return text.replace("_", " ").strip()


def subtype_curation_scope_note(
    panel_code,
    *,
    panel_subtype=None,
    base_code=None,
    base_name=None,
    noun="therapy evidence",
):
    panel_name = cancer_code_display_name(panel_code).lower()
    base_label = cancer_code_display_name(base_code, fallback=base_name).lower()
    subtype_label = _clean_text(panel_subtype).replace("_", " ").lower()
    if subtype_label:
        focus = (
            subtype_label
            if subtype_label.endswith(panel_name)
            else f"{subtype_label} {panel_name}".strip()
        )
    else:
        focus = panel_name
    if not base_label or focus == base_label:
        return f"Using {focus}-specific {noun}."
    return f"Using {focus}-specific {noun} rather than the broader {base_label} list."


def tumor_attribution_context(row):
    """Summarize how securely a row stays tumor-attributed."""
    observed = _safe_float(row.get("observed_tpm"), 0.0)
    mid_tpm = _safe_float(row.get("attr_tumor_tpm"), 0.0)
    low_tpm = _safe_float(row.get("attr_tumor_tpm_low"), mid_tpm)
    high_tpm = _safe_float(row.get("attr_tumor_tpm_high"), mid_tpm)
    mid_frac = _safe_float(row.get("attr_tumor_fraction"), 0.0)
    low_frac = _safe_float(row.get("attr_tumor_fraction_low"), mid_frac)
    high_frac = _safe_float(row.get("attr_tumor_fraction_high"), mid_frac)
    support_fraction = _safe_float(
        row.get("attr_support_fraction"),
        1.0 if (mid_tpm >= 1.0 and mid_frac >= 0.30) else 0.0,
    )
    low_tpm, mid_tpm, high_tpm = sorted(
        [max(0.0, low_tpm), max(0.0, mid_tpm), max(0.0, high_tpm)]
    )
    low_frac, mid_frac, high_frac = sorted(
        [
            max(0.0, min(1.0, low_frac)),
            max(0.0, min(1.0, mid_frac)),
            max(0.0, min(1.0, high_frac)),
        ]
    )
    support_fraction = max(0.0, min(1.0, support_fraction))

    notes = []
    if _truthy(row.get("matched_normal_over_predicted")):
        notes.append("matched-normal reference overshoots the sample")
    if _truthy(row.get("low_purity_cap_applied")):
        notes.append("low-purity cap is active")
    source_marker = _truthy(row.get("source_marker_non_tumor_prior"))
    source_marker_compartment = _clean_text(row.get("source_marker_compartment"))
    if source_marker:
        notes.append(
            f"{source_marker_compartment or 'non-tumor lineage'} prior is active"
        )
    if _truthy(row.get("tme_explainable")) and support_fraction < 1.0:
        notes.append("non-tumor tissue explanations remain plausible")
    if _truthy(row.get("tme_dominant")):
        notes.append("most fitted signal remains non-tumor")

    if high_tpm < 1.0 or high_frac < 0.30 or _truthy(row.get("tme_dominant")):
        tier = "background_dominant"
        label = "background-dominant"
        summary = "non-tumor compartments remain the simpler explanation"
    elif (
        low_tpm >= 1.0
        and low_frac >= 0.30
        and support_fraction >= 0.67
        and not _truthy(row.get("matched_normal_over_predicted"))
        and not source_marker
    ):
        tier = "tumor_supported"
        label = "tumor-supported"
        summary = "tumor-attributed signal stays material across the range"
    else:
        tier = "mixed_source"
        label = "mixed-source"
        summary = "both tumor and benign/background sources remain plausible"

    if observed > 0:
        band = (
            f"{mid_tpm:.0f} tumor-source bulk TPM "
            f"(model interval {low_tpm:.0f}-{high_tpm:.0f}; "
            f"{mid_frac:.0%} tumor, {low_frac:.0%}-{high_frac:.0%} interval)"
        )
    else:
        band = f"{mid_tpm:.0f} tumor-source bulk TPM"

    return {
        "tier": tier,
        "label": label,
        "summary": summary,
        "notes": notes,
        "band": band,
        "observed_tpm": observed,
        "attr_tumor_tpm": mid_tpm,
        "attr_tumor_tpm_low": low_tpm,
        "attr_tumor_tpm_high": high_tpm,
        "attr_tumor_fraction": mid_frac,
        "attr_tumor_fraction_low": low_frac,
        "attr_tumor_fraction_high": high_frac,
        "attr_support_fraction": support_fraction,
    }


def context_expression_context(row):
    """Return the broader tumor-context expression range.

    This is intentionally separate from ``tumor_attribution_context``:
    ``tumor_cell_tpm`` / ``median_est`` can remain useful for cohort-context
    and pathway reasoning, but it is not proof that the observed RNA came
    from tumor cells when the source attribution says background-dominant.
    """
    mid_tpm = _safe_float(
        row.get("tumor_cell_tpm"),
        _safe_float(row.get("median_est"), _safe_float(row.get("observed_tpm"), 0.0)),
    )
    low_tpm = _safe_float(
        row.get("tumor_cell_tpm_low"),
        _safe_float(row.get("est_1"), mid_tpm),
    )
    high_tpm = _safe_float(
        row.get("tumor_cell_tpm_high"),
        _safe_float(row.get("est_9"), mid_tpm),
    )
    low_tpm, mid_tpm, high_tpm = sorted(
        [max(0.0, low_tpm), max(0.0, mid_tpm), max(0.0, high_tpm)]
    )
    return {
        "context_tpm": mid_tpm,
        "context_tpm_low": low_tpm,
        "context_tpm_high": high_tpm,
    }


def context_expression_band_cell(row):
    ctx = context_expression_context(row)
    if abs(ctx["context_tpm_low"] - ctx["context_tpm_high"]) < 0.5:
        return f"{ctx['context_tpm']:.0f}"
    return (
        f"{ctx['context_tpm']:.0f} "
        f"({ctx['context_tpm_low']:.0f}-{ctx['context_tpm_high']:.0f})"
    )


def tpm_semantics_note() -> str:
    """One reader-facing explanation of bulk vs modeled TPM columns."""
    return (
        "**TPM semantics:** Bulk TPM is the measured RNA abundance in the mixed "
        "specimen. Tumor-attributed bulk TPM is the share of that bulk signal "
        "assigned to tumor rather than matched-normal/TME compartments. Context "
        "TPM is a broader per-tumor-cell/cohort-context estimate used for cancer "
        "context, pathway biology, and 'is this gene high in this specimen?' "
        "reasoning; it can collapse to a single value when the range model has "
        "no meaningful spread. Do not treat context TPM as tumor-source evidence "
        "when the tumor-attribution row says background-dominant. For immune/stromal "
        "markers, agent-only rows, and therapies whose eligibility does not depend "
        "on target expression, bulk RNA is contextual and the clinical biomarker must come from the "
        "indicated assay."
    )


_IMMUNE_CHECKPOINT_AGENTS = re.compile(
    r"\b("
    r"pembrolizumab|nivolumab|ipilimumab|dostarlimab|relatlimab|"
    r"atezolizumab|avelumab|durvalumab"
    r")\b",
    re.IGNORECASE,
)
_TARGET_EXPRESSION_INDICATION = re.compile(
    r"\b(pd[- ]?l1|pd[- ]?1|cps|tps|ihc|overexpress|expression|expressing)\b",
    re.IGNORECASE,
)
_MUTATION_INDICATION = re.compile(
    r"\b("
    r"mutation|mutant|v600|[a-z]\d{2,4}[a-z]|exon\s*\d+|fusion|rearrangement|"
    r"amplification|amplified|amp\b|kdd|kinase\s+domain\s+duplication|"
    r"internal\s+tandem\s+duplication|itd\b|her2\+|brca[- ]?(mut|mutation)"
    r")\b",
    re.IGNORECASE,
)
_MSI_HIGH_INDICATION = re.compile(
    r"\b(msi[- ]?h|msi[- ]?high|dmmr|deficient\s+mmr|mismatch\s+repair\s+deficien)",
    re.IGNORECASE,
)
_MMR_PROFICIENT = re.compile(
    r"\b(pmmr|mmr[- ]?proficient|mismatch\s+repair\s+proficient|mss|msi[- ]?stable)\b",
    re.IGNORECASE,
)


def indication_biomarker(target_row) -> str:
    """Return the typed biomarker that gates a curated therapy row.

    The CSV may eventually carry an explicit ``indication_biomarker`` column.
    Until then this central inference prevents report renderers from treating
    histology/MSI/TMB/mutation-gated therapies as if target RNA expression were
    the approval criterion.
    """
    explicit = _clean_text(
        target_row.get("indication_biomarker") if hasattr(target_row, "get") else None
    ).lower()
    if explicit:
        return explicit

    text = " ".join(
        _clean_text(target_row.get(key))
        for key in ("indication", "rationale", "agent", "agent_class")
        if hasattr(target_row, "get")
    )
    low = text.lower()
    cancer_code = _clean_text(
        target_row.get("cancer_code") if hasattr(target_row, "get") else ""
    ).upper()
    agent_class = _agent_class_text(target_row)
    agent = _clean_text(target_row.get("agent") if hasattr(target_row, "get") else "")
    agent_low = agent.lower()
    if cancer_code == "NUTM" and "small_molecule" in agent_class:
        return "mutation"
    if cancer_code == "ADCC" and "lenvatinib" in agent_low:
        return "histology_only"
    if _MSI_HIGH_INDICATION.search(low) and not _MMR_PROFICIENT.search(low):
        return "msi_high"
    if "tmb" in low or "tumor mutational burden" in low:
        return "tmb_high"
    if _MUTATION_INDICATION.search(text):
        return "mutation"

    if _IMMUNE_CHECKPOINT_AGENTS.search(
        agent
    ) and not _TARGET_EXPRESSION_INDICATION.search(text):
        return "histology_only"

    return "target_expression"


def indication_biomarker_label(target_row) -> str:
    biomarker = indication_biomarker(target_row)
    return {
        "target_expression": "target expression",
        "msi_high": "MSI-H / dMMR",
        "tmb_high": "TMB-high",
        "mutation": "mutation / fusion / amplification",
        "histology_only": "histology indication",
    }.get(biomarker, biomarker.replace("_", " "))


def expression_independent_indication(target_row) -> bool:
    return indication_biomarker(target_row) != "target_expression"


def expression_independent_interpretation(target_row) -> str:
    label = indication_biomarker_label(target_row)
    if indication_biomarker(target_row) == "histology_only":
        return "target expression is not the eligibility criterion — confirm clinical eligibility"
    return f"target expression is not the eligibility criterion — confirm {label} status"


def expression_independent_rna_context(expression_row) -> str:
    """Explain RNA values when eligibility does not depend on target expression."""
    if expression_row is None:
        return "target RNA not measured; eligibility is not inferred from expression"
    observed = _safe_float(expression_row.get("observed_tpm"), 0.0)
    return (
        f"target RNA is context only (bulk {observed:.1f} TPM; "
        "it does not establish eligibility)"
    )


def target_observation_state(sym, ranges_df) -> str:
    """Three-state observation classifier for a target symbol.

    Returns one of:
      ``"below_detection"`` — symbol is in the input file but produced no
          row in ``ranges_df`` (TPM below the ``< 0.01`` filter in
          ``estimate_tumor_expression_ranges``). Clinically: a real negative.
      ``"not_in_input"``   — symbol is *not* in the input file at all.
          Clinically: a coverage gap, not biology.
      ``"unknown"``        — input-symbol set was not attached to
          ``ranges_df`` (legacy callers, malformed DataFrame). Conservative
          fall-through that preserves the historical "not measured" label.
    """
    if not isinstance(sym, str) or sym in ("", "—"):
        return "unknown"
    input_syms = getattr(ranges_df, "attrs", {}).get("sample_input_symbols")
    if not input_syms:
        return "unknown"
    return "below_detection" if sym in input_syms else "not_in_input"


def format_missing_observation_cell(state: str) -> str:
    """Render the bulk-TPM cell for a target with no ``ranges_df`` row.

    Mirrors the three states from :func:`target_observation_state`.
    """
    if state == "below_detection":
        return "0.0 (below detection)"
    if state == "not_in_input":
        return "*not in input*"
    return "*not measured*"


def format_missing_observation_interp(state: str) -> str:
    """Render the interpretation-cell phrase for a target with no row."""
    if state == "below_detection":
        return "below detection (bulk TPM ≈ 0)"
    if state == "not_in_input":
        return "gene symbol not present in input file (coverage gap)"
    return "not measured"


def supplied_alterations_for_gene(analysis, gene: str) -> list[dict]:
    """Return supplied alteration records matching a target gene symbol."""
    if not isinstance(analysis, dict):
        return []
    wanted = _clean_text(gene).upper()
    if not wanted:
        return []
    records = analysis.get("alteration_records") or []
    matches: list[dict] = []
    for record in records:
        if not hasattr(record, "get"):
            continue
        observed = _clean_text(record.get("gene")).upper()
        if observed == wanted:
            matches.append(dict(record))
    return matches


def supplied_alteration_supports_target_row(target_row, analysis) -> list[dict]:
    """Return supplied alteration records compatible with a therapy row.

    This is intentionally conservative: a row that requires EGFR KDD is not
    supported by generic EGFR expression or a vague EGFR variant. For broader
    mutation/fusion/amplification rows, a same-gene supplied alteration of a
    compatible coarse class is enough to mark the eligibility evidence as
    present, still with clinical verification language in the reports.
    """
    sym = _clean_text(target_row.get("symbol") if hasattr(target_row, "get") else "")
    records = supplied_alterations_for_gene(analysis, sym)
    if not records:
        return []
    text = " ".join(
        _clean_text(target_row.get(key))
        for key in ("indication", "rationale", "eligibility_note")
        if hasattr(target_row, "get")
    ).lower()
    required_types: set[str] = set()
    if re.search(r"\b(kdd|kinase\s+domain\s+duplication)\b", text):
        required_types.update({"kdd", "internal_tandem_duplication"})
    elif re.search(r"\b(itd|internal\s+tandem\s+duplication)\b", text):
        required_types.add("internal_tandem_duplication")
    elif re.search(r"\b(fusion|rearrang|translocation)\b", text):
        required_types.add("fusion")
    elif re.search(r"\b(amplification|amplified|\bamp\b|copy\s*number\s*gain)\b", text):
        required_types.add("amplification")
    elif indication_biomarker(target_row) == "mutation":
        required_types.update(
            {
                "mutation",
                "fusion",
                "amplification",
                "loss",
                "kdd",
                "internal_tandem_duplication",
            }
        )
    if not required_types:
        return []
    supported: list[dict] = []
    for record in records:
        observed_type = _clean_text(record.get("alteration_type")).lower()
        observed_text = " ".join(
            _clean_text(record.get(key))
            for key in ("alteration", "raw_name", "alteration_type")
        ).lower()
        if observed_type in required_types:
            supported.append(record)
            continue
        if "kdd" in required_types and "kinase domain duplication" in observed_text:
            supported.append(record)
            continue
        if "amplification" in required_types and "amplif" in observed_text:
            supported.append(record)
    return supported


def supplied_alteration_context_for_target_row(target_row, analysis) -> str:
    """Reader-facing summary of supplied alteration evidence for a target row."""
    supported = supplied_alteration_supports_target_row(target_row, analysis)
    if not supported:
        return ""
    labels: list[str] = []
    for record in supported[:3]:
        gene = _clean_text(record.get("gene"))
        alteration = _clean_text(record.get("alteration")) or _clean_text(
            record.get("alteration_type")
        )
        if gene and alteration.upper().startswith(gene.upper()):
            labels.append(alteration)
        else:
            labels.append(f"{gene} {alteration}".strip())
    suffix = "" if len(supported) <= 3 else f" (+{len(supported) - 3} more)"
    return (
        "supplied alteration evidence matches this therapy requirement: "
        + ", ".join(labels)
        + suffix
        + "; verify against the clinical assay report"
    )


def report_disease_state_text(disease_state: str | None, analysis=None) -> str:
    """Consistent disease-state sentence for Markdown reports.

    ``compose_disease_state_narrative`` intentionally returns an empty
    string when no positive state rule fires. In clinician-facing
    Markdown, silence reads like missing propagation rather than a
    negative finding, so reports render a bounded no-pattern statement
    when therapy-state panels were actually evaluated.
    """
    text = _clean_text(disease_state)
    if text:
        return text
    scores = (
        analysis.get("therapy_response_scores") if isinstance(analysis, dict) else None
    )
    if scores:
        if isinstance(analysis, dict) and analysis.get("pathway_activity_inferences"):
            return (
                "No strong RNA-defined therapy-exposure pattern passed reporting "
                "thresholds; active pathway evidence is summarized separately."
            )
        return "No strong RNA-defined therapy-exposure or pathway-state pattern passed reporting thresholds."
    return ""


def tumor_attribution_band_text(row):
    return tumor_attribution_context(row)["band"]


def tumor_band_available(row):
    """Whether a row carries a usable tumor-attributed range."""
    for key in (
        "attr_tumor_tpm",
        "attr_tumor_tpm_low",
        "attr_tumor_tpm_high",
        "attr_tumor_fraction",
        "attr_tumor_fraction_high",
    ):
        value = row.get(key)
        if value is None:
            continue
        text = _clean_text(value)
        if text:
            return True
    return False


def tumor_band_cell(row):
    """Compact source-attributed ``mid (low-high)`` cell for markdown tables."""
    if not tumor_band_available(row):
        return "—"
    ctx = tumor_attribution_context(row)
    return (
        f"{ctx['attr_tumor_tpm']:.0f} "
        f"({ctx['attr_tumor_tpm_low']:.0f}-{ctx['attr_tumor_tpm_high']:.0f})"
    )


def target_reliability_reasons(row, *, category=None):
    """Return ordered reader-facing caveats for a target/expression row."""
    source = tumor_attribution_context(row)
    reasons = []
    if source["tier"] == "background_dominant":
        reasons.append("background-dominant")
    elif source["tier"] == "mixed_source":
        reasons.append("mixed-source")
    if _truthy(row.get("matched_normal_over_predicted")):
        reasons.append("matched-normal over-predicted")
    if _truthy(row.get("smooth_muscle_stromal_leakage")):
        reasons.append("possible smooth-muscle stromal leakage")
    if _truthy(row.get("tme_explainable")) and source["tier"] != "tumor_supported":
        reasons.append("healthy-tissue-explainable")
    if _truthy(row.get("source_marker_non_tumor_prior")):
        reasons.append("non-tumor lineage marker")
    if _truthy(row.get("low_purity_cap_applied")):
        reasons.append("low-purity capped")
    return reasons


def same_lineage_material_target_candidate(
    row,
    target_row=None,
    *,
    min_tumor_tpm=10.0,
) -> bool:
    """Whether a same-lineage clinical target should remain reviewable.

    Matched-normal attribution for a lineage marker is different from
    unrelated immune/stromal attribution: it is a source and specificity
    caveat, not automatic evidence that the target is clinically irrelevant.
    Keep such rows provisional when there is a named clinical agent and a
    material tumor-inferred signal.
    """
    if target_row is None or expression_independent_indication(target_row):
        return False
    phase = _clean_text(target_row.get("phase"))
    if phase not in {"approved", "phase_3", "phase_2", "phase_1"}:
        return False
    if not _clean_text(target_row.get("agent")):
        return False
    if _truthy(row.get("tme_dominant")) and not _truthy(
        row.get("matched_normal_over_predicted")
    ):
        return False
    if _safe_float(row.get("attr_tumor_tpm"), 0.0) < float(min_tumor_tpm):
        return False
    normal = normal_expression_context(row)
    if normal.get("tier") != "same_lineage_expected":
        return False
    top_compartment = _clean_text(row.get("attr_top_compartment")).replace("_", " ")
    return (
        top_compartment.startswith("matched normal ")
        or _truthy(row.get("matched_normal_over_predicted"))
        or _safe_float(row.get("matched_normal_tpm"), 0.0) > 0.0
    )


def target_reliability_status(row, *, category=None, target_row=None):
    """Classify a row as ``supported``, ``provisional``, or ``unsupported``."""
    if target_row is not None and expression_independent_indication(target_row):
        return "provisional"
    source = tumor_attribution_context(row)
    if source["tier"] == "background_dominant":
        if same_lineage_material_target_candidate(row, target_row=target_row):
            return "provisional"
        return "unsupported"
    if _truthy(row.get("matched_normal_over_predicted")):
        return "provisional"
    if _truthy(row.get("smooth_muscle_stromal_leakage")):
        return "provisional"
    if source["tier"] == "mixed_source":
        return "provisional"
    return "supported"


_ESSENTIAL_TISSUE_COLS = {
    "brain": [
        "cerebral_cortex_nTPM",
        "cerebellum_nTPM",
        "basal_ganglia_nTPM",
        "hippocampal_formation_nTPM",
        "amygdala_nTPM",
        "midbrain_nTPM",
        "hypothalamus_nTPM",
        "spinal_cord_nTPM",
        "choroid_plexus_nTPM",
    ],
    "heart": ["heart_muscle_nTPM"],
    "liver": ["liver_nTPM"],
    "lung": ["lung_nTPM"],
    "kidney": ["kidney_nTPM"],
    "bone_marrow": ["bone_marrow_nTPM"],
    "spleen": ["spleen_nTPM"],
    "pancreas": ["pancreas_nTPM"],
    "colon": ["colon_nTPM", "rectum_nTPM", "duodenum_nTPM", "small_intestine_nTPM"],
    "stomach": ["stomach_nTPM"],
}


@lru_cache(maxsize=1)
def _pan_cancer_normal_expression_index():
    try:
        from trufflepig.reference import pan_cancer_expression
        df = pan_cancer_expression(technical_rna_normalize=True).copy()
        if "Ensembl_Gene_ID" not in df.columns:
            return None
        df["Ensembl_Gene_ID"] = (
            df["Ensembl_Gene_ID"].astype(str).str.split(".", n=1).str[0]
        )
        df = df.drop_duplicates(subset="Ensembl_Gene_ID").set_index("Ensembl_Gene_ID")
        return df
    except Exception:
        return None


def _healthy_expression_profile(gene_id):
    gene_id = _clean_text(gene_id).split(".", 1)[0]
    if not gene_id:
        return {}
    ref = _pan_cancer_normal_expression_index()
    if ref is None or gene_id not in ref.index:
        return {}
    row = ref.loc[gene_id]
    profile = {}
    for tissue, columns in _ESSENTIAL_TISSUE_COLS.items():
        values = []
        for column in columns:
            if column not in row.index:
                continue
            try:
                value = float(row[column])
            except Exception:
                continue
            if value == value:
                values.append(value)
        if values:
            profile[tissue] = max(values)
    return profile


def normal_expression_context(row):
    """Return a coarse reader-facing normal-expression context label."""
    category_label = str(row.get("category") or "").upper()
    if _truthy(row.get("is_cta")) or category_label == "CTA":
        return {
            "tier": "cta_restricted",
            "label": "CTA / immune-privileged",
            "summary": "CTA-style restricted normal expression",
            "details": [],
        }

    matched_tissue = _clean_text(row.get("matched_normal_tissue")).replace("_", " ")
    matched_normal_tpm = _safe_float(row.get("matched_normal_tpm"), 0.0)
    tumor_tpm = _safe_float(row.get("attr_tumor_tpm"), 0.0)
    attr_top = _clean_text(row.get("attr_top_compartment")).replace("_", " ")
    if not matched_tissue and attr_top.startswith("matched normal "):
        matched_tissue = attr_top.replace("matched normal ", "", 1)
    details = []
    same_lineage = False
    if matched_tissue:
        same_lineage = (
            attr_top == f"matched normal {matched_tissue}"
            or _truthy(row.get("matched_normal_over_predicted"))
            or matched_normal_tpm >= max(1.0, tumor_tpm * 0.5)
        )

    healthy_profile = _healthy_expression_profile(row.get("gene_id"))
    vital_detail = ""
    if healthy_profile:
        top_tissue, top_value = max(healthy_profile.items(), key=lambda item: item[1])
        if top_value >= 10.0:
            vital_detail = f"{top_tissue.replace('_', ' ')} expression is appreciable"

    if (
        _truthy(row.get("broadly_expressed"))
        or _safe_int(row.get("n_healthy_tissues_expressed"), 0) >= 15
    ):
        details.append(
            "broader healthy-tissue signal is present outside the likely tissue of origin"
        )
    if vital_detail:
        details.append(vital_detail)

    if same_lineage:
        return {
            "tier": "same_lineage_expected",
            "label": "same-lineage expected",
            "summary": f"benign {matched_tissue} lineage expression is expected",
            "details": details,
        }

    if vital_detail:
        return {
            "tier": "vital_tissue_concern",
            "label": "vital-tissue concern",
            "summary": vital_detail,
            "details": details[1:] if details else [],
        }

    if details:
        return {
            "tier": "broad_healthy_expression",
            "label": "broad healthy expression",
            "summary": details[0],
            "details": details[1:],
        }

    return {
        "tier": "restricted_outside_lineage",
        "label": "restricted outside likely tissue of origin",
        "summary": "limited healthy-tissue signal outside the likely tissue of origin",
        "details": [],
    }


def clinical_maturity_info(target_row, target_panel=None):
    """Summarize maturity from the current row and its curated siblings."""

    def _display_agent_class(value):
        text = _clean_text(value).replace("_", " ")
        if not text:
            return ""
        acronym_map = {
            "adc": "ADC",
            "tce": "TCE",
            "car t": "CAR-T",
            "cart": "CAR-T",
            "tcr t": "TCR-T",
            "tcrt": "TCR-T",
            "pmhc": "pMHC",
            "mhc": "MHC",
        }
        lowered = text.lower()
        if lowered in acronym_map:
            return acronym_map[lowered]
        return re.sub(
            r"\b(adc|tce|car-t|tcr-t|pmhc|mhc)\b",
            lambda match: acronym_map.get(
                match.group(1).lower(), match.group(1).upper()
            ),
            text,
            flags=re.IGNORECASE,
        )

    phase = _clean_text(target_row.get("phase"))
    phase_label = {
        "approved": "approved",
        "phase_3": "late clinical",
        "phase_2": "mid-clinical",
        "phase_1": "early clinical",
        "preclinical": "preclinical",
        "off_label": "off-label / transfer rationale",
    }.get(phase, phase or "curated")
    agent_class = _display_agent_class(target_row.get("agent_class"))
    summary = phase_label
    if agent_class:
        summary += f" {agent_class}"

    if target_panel is None or len(target_panel) == 0:
        return {
            "tier": phase_label,
            "summary": summary,
            "n_agents": 0,
            "n_modalities": 0,
        }

    symbol = _clean_text(target_row.get("symbol"))
    if not symbol or "symbol" not in target_panel.columns:
        return {
            "tier": phase_label,
            "summary": summary,
            "n_agents": 0,
            "n_modalities": 0,
        }

    try:
        sub = target_panel[target_panel["symbol"].astype(str) == symbol]
    except Exception:
        sub = None

    if sub is None or len(sub) <= 1:
        return {
            "tier": phase_label,
            "summary": summary,
            "n_agents": 1 if sub is not None and len(sub) == 1 else 0,
            "n_modalities": 1 if sub is not None and len(sub) == 1 else 0,
        }

    n_agents = (
        sub["agent"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("nan", "")
        .loc[lambda s: s.ne("")]
        .nunique()
        if "agent" in sub.columns
        else 0
    )
    n_modalities = (
        sub["agent_class"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("nan", "")
        .loc[lambda s: s.ne("")]
        .nunique()
        if "agent_class" in sub.columns
        else 0
    )
    extras = []
    if n_agents > 1:
        extras.append(f"{n_agents} agents")
    if n_modalities > 1:
        extras.append(f"{n_modalities} modalities")
    if extras:
        summary += f" ({', '.join(extras)})"
    return {
        "tier": phase_label,
        "summary": summary,
        "n_agents": int(n_agents),
        "n_modalities": int(n_modalities),
    }


def clinical_maturity_summary(target_row, target_panel=None):
    return clinical_maturity_info(target_row, target_panel=target_panel)["summary"]


# Controlled-vocab display labels for the leveled-up agent ``modality`` column
# (issue #47). Permissive: any value not listed passes through verbatim so a new
# pirlygenes modality renders sensibly without a trufflepig change.
_AGENT_MODALITY_LABELS = {
    "adc": "ADC",
    "rlt": "radioligand (RLT)",
    "radioligand": "radioligand (RLT)",
    "tce": "T-cell engager",
    "bispecific": "T-cell engager",
    "car_t": "CAR-T",
    "car-t": "CAR-T",
    "car_nk": "CAR-NK",
    "tcr_t": "TCR-T",
    "tcr-t": "TCR-T",
    "vaccine": "cancer vaccine",
    "mab": "mAb",
    "small_molecule": "small molecule",
    "macrocycle": "macrocycle",
    "degrader": "degrader",
    "immunotoxin": "immunotoxin",
    "oncolytic": "oncolytic virus",
    "peptide": "peptide",
}


def _agent_field(target_row, *keys):
    """First non-blank value among ``keys`` in a dict/Series-like target row."""
    getter = getattr(target_row, "get", None)
    for key in keys:
        value = getter(key, "") if getter else (
            target_row[key] if key in target_row else ""
        )
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none"}:
            return text
    return ""


def _is_truthy_flag(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def agent_metadata_clause(target_row) -> str:
    """Render the leveled-up therapeutic-agent metadata into one compact clause.

    Covers modality (+ payload/isotope/effector detail), route, approval
    status/year/body, efficacy, key toxicities / boxed warning, and combination
    use — the enriched ``therapeutic-agents`` schema planned in #47.

    Every field is optional. Before pirlygenes ships those columns the curated
    rows carry none of them, so this returns ``""`` and every caller's output is
    byte-for-byte unchanged; the metadata surfaces automatically once the
    columns exist.
    """
    bits: list[str] = []

    modality = _AGENT_MODALITY_LABELS.get(
        _agent_field(target_row, "modality", "modality_type").lower(),
        _agent_field(target_row, "modality", "modality_type"),
    )
    detail = _agent_field(
        target_row, "modality_detail", "payload", "isotope", "engaged_effector"
    )
    if modality:
        bits.append(f"{modality} ({detail})" if detail else modality)
    elif detail:
        bits.append(detail)

    route = _agent_field(target_row, "route", "route_of_administration")
    if route:
        bits.append(route)

    status = _agent_field(target_row, "approval_status")
    if status:
        approval = status.replace("_", " ")
        year = _agent_field(target_row, "approval_year")
        body = _agent_field(target_row, "approving_body")
        if year:
            approval += f" {year}"
        if body:
            approval += f", {body}"
        bits.append(approval)

    efficacy = _agent_field(target_row, "efficacy", "response_rate", "orr")
    if efficacy:
        bits.append(f"efficacy {efficacy}")

    toxicity = _agent_field(target_row, "key_toxicities", "toxicity")
    if toxicity:
        bits.append(f"key toxicities: {toxicity}")
    if _is_truthy_flag(_agent_field(target_row, "boxed_warning")):
        bits.append("boxed warning")

    partners = _agent_field(target_row, "combination_partners", "combination")
    if _is_truthy_flag(_agent_field(target_row, "combination_only")):
        bits.append(
            f"combination-only (with {partners})" if partners else "combination-only"
        )
    elif partners:
        bits.append(f"combination with {partners}")

    return "; ".join(bits)


@lru_cache(maxsize=1)
def cross_cancer_target_index() -> dict[str, tuple[dict, ...]]:
    """Map ``symbol -> curated target rows across *all* cancer codes`` (#47).

    The per-sample recommendation only pulls the sample cancer's curated panel,
    so a gene that is a known, drugged target *in another indication* is invisible
    when it lights up in a different tumor. This index unions two sources so an
    off-context binder is surfaced rather than dropped:

    1. pirlygenes ``cancer-key-genes`` ``role=target`` rows — carry the specific
       cancer_code an agent is curated against.
    2. the trufflepig-owned therapeutic-agent registry (#52) — the full binder
       universe (modality / approval / phase), including targets pirlygenes does
       not curate against any cancer code (the Tier-D "invisible" genes from #47).
    """
    from pirlygenes import get_data
    from .therapeutic_agents import agents_for_target, druggable_target_genes

    rows = get_data("cancer-key-genes")
    rows = rows[rows["role"].astype(str) == "target"]
    out: dict[str, list[dict]] = {}
    seen: dict[str, set[str]] = {}
    for _, row in rows.iterrows():
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        agent = str(row.get("agent") or "").strip()
        out.setdefault(symbol, []).append(
            {
                "cancer_code": str(row.get("cancer_code") or "").strip(),
                "agent": agent,
                "agent_class": str(row.get("agent_class") or "").strip(),
                "phase": str(row.get("phase") or "").strip(),
                "indication": str(row.get("indication") or "").strip(),
                "modality": "",
                "approval": "",
                "source": "key_genes",
            }
        )
        seen.setdefault(symbol, set()).add(agent.lower())

    # Union in the trufflepig therapeutic-agent registry. Each registry agent
    # is keyed on its target gene symbol (cancer-agnostic), so it surfaces
    # wherever the gene is expressed; dedupe against agents already named by
    # the cancer-key-genes rows.
    for gene in druggable_target_genes():
        symbol = gene.strip().upper()
        for agent in agents_for_target(gene):
            if agent.agent.lower() in seen.get(symbol, set()):
                continue
            out.setdefault(symbol, []).append(
                {
                    "cancer_code": "",
                    "agent": agent.agent,
                    "agent_class": agent.modality_label,
                    "phase": agent.highest_phase,
                    "indication": agent.approved_indication or agent.indications,
                    "modality": agent.modality,
                    "approval": agent.approval_clause(),
                    "source": "registry",
                }
            )
            seen.setdefault(symbol, set()).add(agent.agent.lower())
    return {sym: tuple(entries) for sym, entries in out.items()}


def _row_tumor_tpm(record) -> float:
    getter = getattr(record, "get", None)

    def field(key):
        return getter(key) if getter else record[key] if key in record else None

    for key in ("attr_tumor_tpm", "observed_tpm"):
        value = field(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def offcontext_known_targets(
    ranges_df,
    panel_symbols,
    *,
    index=None,
    min_tumor_tpm: float = 10.0,
):
    """Highly-expressed genes that are drugged targets in *another* indication
    but absent from this sample's curated panel (#47 "surface when high").

    Returns a list of ``{symbol, tumor_tpm, indications}`` sorted by tumor TPM,
    where ``indications`` is the cross-cancer curated rows (agent / phase /
    indication / cancer_code). ``panel_symbols`` are the symbols already on the
    sample's curated panel (excluded). ``index`` defaults to
    :func:`cross_cancer_target_index` and is injectable for testing.
    """
    if index is None:
        index = cross_cancer_target_index()
    panel = {str(s).strip().upper() for s in (panel_symbols or [])}

    if hasattr(ranges_df, "to_dict"):
        records = ranges_df.to_dict("records")
    else:
        records = list(ranges_df or [])

    hits = []
    seen: set[str] = set()
    for record in records:
        getter = getattr(record, "get", None)
        symbol = str(
            (getter("symbol") if getter else record.get("symbol")) or ""
        ).strip().upper()
        if not symbol or symbol in panel or symbol in seen:
            continue
        entries = index.get(symbol)
        if not entries:
            continue
        tumor_tpm = _row_tumor_tpm(record)
        if tumor_tpm < min_tumor_tpm:
            continue
        seen.add(symbol)
        hits.append(
            {"symbol": symbol, "tumor_tpm": tumor_tpm, "indications": entries}
        )
    hits.sort(key=lambda hit: hit["tumor_tpm"], reverse=True)
    return hits


_STANDARD_PATH_TEXT = re.compile(
    r"\b("
    r"standard(?:\s+of\s+care)?|standard\s+backbone|backbone|"
    r"first[- ]line|frontline|newly\s+diagnosed|1l\b|"
    r"adjuvant|neoadjuvant|maintenance|combined\s+with\s+chemo"
    r")\b",
    re.IGNORECASE,
)
_LATER_LINE_TEXT = re.compile(
    r"\b("
    r"pretreated|previously\s+treated|relapsed[/ -]?refractory|"
    r"refractory|post[- ]|after\s+.+\btherapy|after\s+.+\btreatment|"
    r"second[- ]line|third[- ]line|subsequent\s+lines?|"
    r"2l\b|3l\b|r/r|resistance\s+setting|prior\s+(?:line|lines|therapy|"
    r"therapies|arpi|taxane)|at\s+least\s+one\s+prior|>=\s*2|≥\s*2"
    r")\b",
    re.IGNORECASE,
)
_AR_PATHWAY_AGENTS = re.compile(
    r"\b(enzalutamide|apalutamide|darolutamide|abiraterone)\b",
    re.IGNORECASE,
)
_ER_ENDOCRINE_AGENTS = re.compile(
    r"\b("
    r"elacestrant|fulvestrant|tamoxifen|letrozole|anastrozole|exemestane|"
    r"palbociclib|ribociclib|abemaciclib|aromatase\s+inhibitor|serd|"
    r"endocrine\s+therapy"
    r")\b",
    re.IGNORECASE,
)
_HER2_DIRECTED_AGENTS = re.compile(
    r"\b("
    r"trastuzumab|pertuzumab|tucatinib|lapatinib|neratinib|margetuximab|"
    r"trastuzumab\s+deruxtecan|t-dxd"
    r")\b",
    re.IGNORECASE,
)
_MAPK_RTK_AGENTS = re.compile(
    r"\b("
    r"osimertinib|erlotinib|gefitinib|afatinib|dacomitinib|amivantamab|"
    r"alectinib|brigatinib|ceritinib|crizotinib|lorlatinib|entrectinib|"
    r"larotrectinib|selpercatinib|pralsetinib|capmatinib|tepotinib|"
    r"sotorasib|adagrasib|dabrafenib|trametinib|encorafenib|binimetinib|"
    r"vemurafenib|cetuximab|panitumumab|erdafitinib"
    r")\b",
    re.IGNORECASE,
)
THERAPY_PATH_TIERS = frozenset(
    {
        "approved_standard",
        "approved_indication_matched",
        "approved_later_line",
        "late_clinical",
        "investigational_biomarker_matched",
        "trial_follow_up",
        "preclinical",
        "off_label",
    }
)
_THERAPY_PATH_RANK = {
    "approved_standard": 0,
    "approved_indication_matched": 1,
    "approved_later_line": 2,
    "late_clinical": 3,
    # Biomarker-matched investigational (e.g. HER2+ gallbladder trastuzumab,
    # BRAF V600E craniopharyngioma): a phase-2 trial selected by a molecular
    # match — preferred over a generic trial follow-up but still investigational.
    "investigational_biomarker_matched": 4,
    "trial_follow_up": 5,
    "preclinical": 6,
    "off_label": 7,
}
_THERAPY_PATH_DEFAULT_NOTE = {
    "approved_standard": "confirm indication and line of therapy",
    "approved_indication_matched": "confirm clinical eligibility",
    "approved_later_line": "confirm prior therapies and indication-specific eligibility",
    "late_clinical": "not default standard",
    "investigational_biomarker_matched": "confirm biomarker and trial eligibility",
    "trial_follow_up": "not default standard",
    "preclinical": "not a clinical recommendation",
    "off_label": "confirm rationale and alternatives",
}
_THERAPY_EXPOSURE_RULES = (
    {
        "axis": "AR_signaling",
        "state": "down",
        "symbols": {"AR"},
        "agent_re": _AR_PATHWAY_AGENTS,
        "class_re": re.compile(r"\bandrogen\b", re.IGNORECASE),
        "disease_re": re.compile(
            r"\b(ar[- ]?axis\s+suppressed|ar\s+signaling\s+suppressed|"
            r"adt\s+exposure|androgen\s+deprivation|arpi)\b",
            re.IGNORECASE,
        ),
        "axis_label": "AR-axis",
        "exposure_label": "ADT or ARPI",
    },
    {
        "axis": "ER_signaling",
        "state": "down",
        "symbols": {"ESR1", "PGR"},
        "agent_re": _ER_ENDOCRINE_AGENTS,
        "class_re": re.compile(r"\bhormone\b", re.IGNORECASE),
        "disease_re": re.compile(
            r"\b(er[- ]?axis\s+suppressed|endocrine[- ]?exposed|"
            r"endocrine\s+therapy|aromatase\s+inhibitor|serd)\b",
            re.IGNORECASE,
        ),
        "axis_label": "ER-axis",
        "exposure_label": "endocrine therapy",
        "suppressed_explanation": "ER-low biology or current/prior endocrine therapy signal",
    },
    {
        "axis": "HER2_signaling",
        "state": "down",
        "symbols": {"ERBB2", "HER2"},
        "agent_re": _HER2_DIRECTED_AGENTS,
        "class_re": re.compile(r"\bher2\b", re.IGNORECASE),
        "disease_re": re.compile(
            r"\b(her2[- ]?axis\s+suppressed|anti[- ]?her2|"
            r"her2[- ]?directed)\b",
            re.IGNORECASE,
        ),
        "axis_label": "HER2-axis",
        "exposure_label": "HER2-directed therapy",
    },
    {
        "axis": "MAPK_EGFR_signaling",
        "state": "down",
        "symbols": {
            "EGFR",
            "BRAF",
            "KRAS",
            "NRAS",
            "MAP2K1",
            "MAP2K2",
            "MEK1",
            "MEK2",
            "ALK",
            "ROS1",
            "RET",
            "MET",
            "NTRK1",
            "NTRK2",
            "NTRK3",
            "FGFR1",
            "FGFR2",
            "FGFR3",
        },
        "agent_re": _MAPK_RTK_AGENTS,
        "class_re": re.compile(r"\b(tki|kinase)\b", re.IGNORECASE),
        "disease_re": re.compile(
            r"\b(egfr[- ]?tki|mapk[-/ ]?pathway\s+suppressed|"
            r"targeted\s+kinase\s+inhibitor)\b",
            re.IGNORECASE,
        ),
        "axis_label": "MAPK/RTK pathway",
        "exposure_label": "targeted kinase inhibitor",
    },
)


def _target_symbol(target_row) -> str:
    return _clean_text(
        target_row.get("symbol") if hasattr(target_row, "get") else ""
    ).upper()


def _agent_text(target_row) -> str:
    return _clean_text(target_row.get("agent") if hasattr(target_row, "get") else "")


def _agent_class_text(target_row) -> str:
    return _clean_text(
        target_row.get("agent_class") if hasattr(target_row, "get") else ""
    ).lower()


def _phase_text(target_row) -> str:
    return _clean_text(
        target_row.get("phase") if hasattr(target_row, "get") else ""
    ).lower()


def _therapy_row_text(target_row) -> str:
    if not hasattr(target_row, "get"):
        return ""
    return " ".join(
        _clean_text(target_row.get(key))
        for key in ("agent", "agent_class", "indication", "rationale")
    )


def _therapy_row_matches_exposure_rule(target_row, rule: dict) -> bool:
    symbol = _target_symbol(target_row)
    if symbol and symbol in rule.get("symbols", set()):
        return True
    text = _therapy_row_text(target_row)
    agent_re = rule.get("agent_re")
    if agent_re is not None and agent_re.search(text):
        return True
    class_re = rule.get("class_re")
    if class_re is not None and class_re.search(_agent_class_text(target_row)):
        return True
    return False


def _therapy_path_context_for_tier(target_row, tier: str, note: str = "") -> str:
    agent_class = _agent_class_text(target_row)
    if tier == "approved_standard":
        prefix = "guideline-standard approved pathway"
    elif tier == "approved_indication_matched":
        prefix = "approved biomarker/indication-matched pathway"
    elif tier == "approved_later_line":
        prefix = (
            "approved radioligand pathway"
            if "radioligand" in agent_class
            else "approved later-line pathway"
        )
    elif tier == "late_clinical":
        prefix = "late-clinical follow-up"
    elif tier == "investigational_biomarker_matched":
        prefix = "biomarker-matched investigational pathway"
    elif tier == "trial_follow_up":
        prefix = "clinical-trial follow-up"
    elif tier == "preclinical":
        prefix = "preclinical follow-up"
    elif tier == "off_label":
        prefix = "off-label follow-up"
    else:
        return ""

    suffix = _clean_text(note) or _THERAPY_PATH_DEFAULT_NOTE.get(tier, "")
    suffix_lc = suffix.lower()
    prefix_lc = prefix.lower()
    if suffix_lc == prefix_lc:
        suffix = ""
    elif suffix_lc.startswith(prefix_lc + ";"):
        suffix = suffix[len(prefix) :].lstrip(" ;")
    elif suffix_lc.startswith(prefix_lc + ","):
        suffix = suffix[len(prefix) :].lstrip(" ,")
    elif suffix_lc.startswith(prefix_lc + " -"):
        suffix = suffix[len(prefix) :].lstrip(" -")
    if suffix:
        return f"{prefix}; {suffix}"
    return prefix


def _explicit_therapy_path_info(target_row) -> dict | None:
    tier = _clean_text(
        target_row.get("treatment_path_tier") if hasattr(target_row, "get") else ""
    ).lower()
    if not tier:
        return None
    if tier not in THERAPY_PATH_TIERS:
        return None
    note = _clean_text(
        target_row.get("eligibility_note") if hasattr(target_row, "get") else ""
    )
    return {
        "tier": tier,
        "rank": _THERAPY_PATH_RANK.get(tier, 99),
        "context": _therapy_path_context_for_tier(target_row, tier, note),
        "source": "curated",
    }


def _inferred_therapy_path_info(target_row) -> dict:
    phase = _phase_text(target_row)
    agent_class = _agent_class_text(target_row)
    text = _therapy_row_text(target_row)
    is_standard = _STANDARD_PATH_TEXT.search(text) is not None
    is_later_line = _LATER_LINE_TEXT.search(text) is not None

    if phase == "approved":
        if "radioligand" in agent_class:
            return {
                "tier": "approved_later_line",
                "rank": 2,
                "context": (
                    "approved radioligand pathway; confirm imaging/eligibility "
                    "and prior-line requirements"
                ),
                "source": "inferred",
            }
        if is_standard:
            return {
                "tier": "approved_standard",
                "rank": 0,
                "context": (
                    "guideline-standard approved pathway; confirm the indication "
                    "and line of therapy"
                ),
                "source": "inferred",
            }
        if is_later_line:
            return {
                "tier": "approved_later_line",
                "rank": 2,
                "context": (
                    "approved later-line pathway; confirm prior therapies and "
                    "indication-specific eligibility"
                ),
                "source": "inferred",
            }
        return {
            "tier": "approved_indication_matched",
            "rank": 1,
            "context": (
                "approved biomarker/indication-matched pathway; confirm clinical "
                "eligibility"
            ),
            "source": "inferred",
        }

    phase_context = {
        "phase_3": (
            "late_clinical",
            3,
            "late-clinical follow-up, not default standard",
        ),
        "phase_2": (
            "trial_follow_up",
            4,
            "clinical-trial follow-up, not default standard",
        ),
        "phase_1": (
            "trial_follow_up",
            5,
            "clinical-trial follow-up, not default standard",
        ),
        "preclinical": (
            "preclinical",
            6,
            "preclinical follow-up, not a clinical recommendation",
        ),
        "off_label": (
            "off_label",
            7,
            "off-label follow-up; confirm rationale and alternatives",
        ),
    }
    tier, rank, context = phase_context.get(phase, ("unknown", 99, ""))
    return {"tier": tier, "rank": rank, "context": context, "source": "inferred"}


def _therapy_path_info(target_row) -> dict:
    return _explicit_therapy_path_info(target_row) or _inferred_therapy_path_info(
        target_row
    )


def therapy_path_tier(target_row) -> str:
    """Data-driven treatment-path tier used by report sorting/wording."""
    return _therapy_path_info(target_row)["tier"]


def _exposure_rule_active(rule: dict, *, analysis=None, disease_state=None) -> bool:
    axis_down = (
        _therapy_axis_state(analysis, str(rule.get("axis") or ""))
        == str(rule.get("state") or "").lower()
    )
    disease_re = rule.get("disease_re")
    disease_match = (
        disease_re.search(_clean_text(disease_state)) is not None
        if disease_re is not None
        else False
    )
    return bool(axis_down or disease_match)


def _therapy_axis_state(analysis, axis: str) -> str:
    if not isinstance(analysis, dict):
        return ""
    scores = analysis.get("therapy_response_scores") or {}
    score = scores.get(axis) if hasattr(scores, "get") else None
    if score is None:
        return ""
    if hasattr(score, "get"):
        return _clean_text(score.get("state")).lower()
    return _clean_text(getattr(score, "state", "")).lower()


def therapy_state_caution(target_row, *, analysis=None, disease_state=None) -> str:
    """Warn when a candidate therapy matches an exposure pattern already seen.

    This is intentionally phrased as a medication-reconciliation prompt, not
    a clinical contraindication. Single-sample RNA can suggest exposure but
    cannot determine whether the drug is current, prior, tolerated, or failed.
    """
    for rule in _THERAPY_EXPOSURE_RULES:
        if not _therapy_row_matches_exposure_rule(target_row, rule):
            continue
        if not _exposure_rule_active(
            rule, analysis=analysis, disease_state=disease_state
        ):
            continue
        suppressed_explanation = rule.get(
            "suppressed_explanation",
            f"current/prior {rule['exposure_label']} signal",
        )
        return (
            f"{rule['axis_label']} RNA/signaling is already suppressed "
            f"({suppressed_explanation}); verify the medication list before "
            "treating this as new-start therapy"
        )
    return ""


def hla_restrictions_for_target_row(target_row) -> list[str]:
    """Return class-I HLA restrictions encoded in a therapy row."""
    if not hasattr(target_row, "get"):
        return []
    explicit = []
    for key in ("hla_restriction", "HLA_Restriction", "hla", "HLA"):
        value = target_row.get(key)
        if _clean_text(value):
            explicit.extend(parse_hla_types(value))
    if explicit:
        return sorted(set(explicit))

    text = " ".join(
        _clean_text(target_row.get(key))
        for key in ("indication", "rationale", "eligibility_note")
    )
    return extract_hla_types_from_text(text)


def target_hla_eligibility(target_row, *, analysis=None) -> dict:
    """Classify supplied HLA types against a row's HLA restriction."""
    restrictions = hla_restrictions_for_target_row(target_row)
    supplied = []
    if isinstance(analysis, dict):
        constraints = analysis.get("analysis_constraints") or {}
        supplied = parse_hla_types(constraints.get("hla_types"))
    if not restrictions:
        return {
            "status": "not_hla_restricted",
            "required": [],
            "supplied": supplied,
            "matched_supplied": None,
            "matched_required": None,
        }
    if not supplied:
        return {
            "status": "unknown",
            "required": restrictions,
            "supplied": [],
            "matched_supplied": None,
            "matched_required": None,
        }
    status, matched_supplied, matched_required = hla_types_compatibility_status(
        supplied, restrictions
    )
    return {
        "status": status,
        "required": restrictions,
        "supplied": supplied,
        "matched_supplied": matched_supplied,
        "matched_required": matched_required,
    }


def hla_eligibility_context(target_row, *, analysis=None) -> str:
    """Reader-facing HLA note for TCR/pMHC-gated rows."""
    eligibility = target_hla_eligibility(target_row, analysis=analysis)
    status = eligibility["status"]
    if status == "not_hla_restricted":
        return ""
    required = "/".join(eligibility["required"])
    supplied = "/".join(eligibility["supplied"])
    if status == "matched":
        return (
            f"HLA match: supplied {eligibility['matched_supplied']} is compatible "
            f"with required {eligibility['matched_required']}"
        )
    if status == "insufficient_resolution":
        return (
            f"HLA unresolved: supplied {eligibility['matched_supplied']} is lower "
            f"resolution than required {eligibility['matched_required']}; provide "
            "high-resolution HLA typing to assess eligibility"
        )
    if status == "mismatched":
        return f"HLA mismatch: supplied {supplied} does not match required {required}"
    agent = _clean_text(target_row.get("agent")) if hasattr(target_row, "get") else ""
    agent_clause = f" for {agent}" if agent else ""
    return (
        f"HLA requirement{agent_clause}: requires {required}; supply germline/tumor "
        "HLA type to assess eligibility"
    )


def hla_restricted_target_supported(target_row, *, analysis=None) -> bool:
    """Whether an HLA-restricted therapy can be shortlisted."""
    return target_hla_eligibility(target_row, analysis=analysis)["status"] != "mismatched"


def therapy_path_context(target_row, *, analysis=None, disease_state=None) -> str:
    """Brief reader-facing treatment-path context for curated therapy rows."""
    parts = [_therapy_path_info(target_row)["context"]]
    hla_context = hla_eligibility_context(target_row, analysis=analysis)
    if hla_context:
        parts.append(hla_context)
    return "; ".join(part for part in parts if part)


def therapy_path_rank(target_row, *, analysis=None, disease_state=None) -> int:
    """Sort standard paths ahead of exploratory rows in concise reports."""
    return int(_therapy_path_info(target_row)["rank"])


def target_interpretation_summary(
    target_row,
    expression_row,
    target_panel=None,
    *,
    analysis=None,
    disease_state=None,
):
    """Return a compact integrated summary for a curated target row."""
    source = tumor_attribution_context(expression_row)
    normal = normal_expression_context(expression_row)
    maturity = (
        clinical_maturity_summary(target_row, target_panel=target_panel)
        if target_row is not None
        else ""
    )
    expr_independent = target_row is not None and expression_independent_indication(
        target_row
    )
    if expr_independent:
        parts = [
            expression_independent_interpretation(target_row),
            expression_independent_rna_context(expression_row),
        ]
    else:
        parts = [source["label"], source["band"]]
        parts.append(normal["label"])
    details = list(normal.get("details") or [])
    if details and not expr_independent:
        parts.append(details[0])
    path_context = (
        therapy_path_context(target_row, analysis=analysis, disease_state=disease_state)
        if target_row is not None
        else ""
    )
    if path_context:
        parts.append(path_context)
    caution = (
        therapy_state_caution(
            target_row, analysis=analysis, disease_state=disease_state
        )
        if target_row is not None
        else ""
    )
    if caution:
        parts.append(f"current-therapy check: {caution}")
    if maturity:
        parts.append(maturity)
    return "; ".join(part for part in parts if part)


def partition_tumor_core_rows(ranges_df, min_tumor_tpm=1.0):
    """Split expression rows by report-facing tumor-inferred reliability."""
    if (
        ranges_df is None
        or len(ranges_df) == 0
        or "attr_tumor_tpm" not in ranges_df.columns
    ):
        empty = ranges_df.iloc[0:0] if ranges_df is not None else None
        return empty, empty, empty

    eligible = ranges_df[
        ranges_df["attr_tumor_tpm"].astype(float) >= float(min_tumor_tpm)
    ].copy()
    if eligible.empty:
        empty = eligible.iloc[0:0]
        return empty, empty, empty

    statuses = [target_reliability_status(row) for row in eligible.to_dict("records")]
    eligible["_report_reliability"] = statuses
    supported = eligible[eligible["_report_reliability"] == "supported"].copy()
    provisional = eligible[eligible["_report_reliability"] == "provisional"].copy()
    unsupported = eligible[eligible["_report_reliability"] == "unsupported"].copy()
    return supported, provisional, unsupported


def summarize_reliability_reasons(rows, top_n=3):
    """Summarize the most common reliability caveats in a row set."""
    if rows is None or len(rows) == 0:
        return ""
    counter = Counter()
    for row in rows.to_dict("records"):
        counter.update(target_reliability_reasons(row))
    if not counter:
        return ""
    return ", ".join(reason for reason, _ in counter.most_common(top_n))


def resolved_subtype_code_for_analysis(analysis, ranges_df=None):
    """Return the final subtype/cancer code implied by the analysis."""
    try:
        from .analyze import cancer_type_context_from_analysis

        cancer_type_context = cancer_type_context_from_analysis(analysis)
        if cancer_type_context.uses_distinct_reference and cancer_type_context.report_code:
            return cancer_type_context.report_code
    except Exception:
        pass

    winning_subtype = candidate_winning_subtype_for_analysis(analysis)
    if not winning_subtype:
        return None

    tumor_tpm_by_symbol = analysis.get("tumor_tpm_by_symbol")
    if not tumor_tpm_by_symbol and ranges_df is not None:
        try:
            from .common import ranges_records
            tumor_tpm_by_symbol = {
                str(row["symbol"]): float(row.get("attr_tumor_tpm") or 0.0)
                for row in ranges_records(ranges_df)
                if str(row.get("symbol") or "")
            }
        except Exception:
            tumor_tpm_by_symbol = None

    try:
        from .degenerate_subtype import resolve_degenerate_subtype

        resolution = resolve_degenerate_subtype(
            winning_subtype,
            site_template=analysis_site_template_for_subtype(analysis),
            tumor_tpm_by_symbol=tumor_tpm_by_symbol,
        )
        winning_subtype = (
            _clean_text(resolution.get("final_subtype")) or winning_subtype
        )
    except Exception:
        pass
    return winning_subtype or None


def analysis_site_template_for_subtype(analysis):
    """Return the site template available to subtype disambiguation.

    Decomposition is the strongest source because it fits the sample
    expression against template panels. When that stage has no usable
    best template, fall back to explicit constraints such as
    ``--site-hint bone`` or ``--met-site bone`` so subtype labels can
    still use supplied site context.
    """
    decomposition = analysis.get("decomposition") or {}
    best_template = _clean_text(decomposition.get("best_template"))
    if best_template:
        return best_template

    constraints = analysis.get("analysis_constraints") or {}
    templates = constraints.get("decomposition_templates") or []
    if isinstance(templates, str):
        templates = [templates]
    templates = [_clean_text(item) for item in templates if _clean_text(item)]
    if len(templates) == 1:
        return templates[0]

    site_hint = _clean_text(constraints.get("site_hint")) or _clean_text(
        constraints.get("met_site")
    )
    if not site_hint:
        return None

    norm = site_hint.lower().replace("-", "_").replace(" ", "_")
    tumor_context = _clean_text(constraints.get("tumor_context")).lower()
    if tumor_context == "primary":
        primary_map = {
            "bone": "primary_bone",
            "bone_marrow": "primary_bone",
            "retroperitoneal": "primary_retroperitoneum",
            "retroperitoneum": "primary_retroperitoneum",
            "abdomen": "primary_abdomen",
            "abdominal": "primary_abdomen",
            "lung": "primary_lung",
            "pancreas": "primary_pancreas",
            "head_neck": "primary_head_neck",
            "head_and_neck": "primary_head_neck",
        }
        if norm in primary_map:
            return primary_map[norm]

    try:
        from .decomposition.engine import DECOMPOSITION_PARAMETERS

        return DECOMPOSITION_PARAMETERS["sample_mode"]["site_hint_templates"].get(norm)
    except Exception:
        fallback_map = {
            "adrenal": "met_adrenal",
            "bone": "met_bone",
            "bone_marrow": "met_bone",
            "brain": "met_brain",
            "liver": "met_liver",
            "lung": "met_lung",
            "lymph": "met_lymph_node",
            "lymph_node": "met_lymph_node",
            "peritoneal": "met_peritoneal",
            "peritoneum": "met_peritoneal",
            "retroperitoneal": "met_soft_tissue",
            "skin": "met_skin",
            "soft_tissue": "met_soft_tissue",
        }
        return fallback_map.get(norm)


def candidate_winning_subtype_for_analysis(analysis):
    """Return the raw winning subtype for the analysis' active cancer code.

    ``candidate_trace[0]`` is the unconstrained expression classifier pick.
    When the user supplied ``--cancer-type``, the final report is explicitly
    scoped to ``analysis["cancer_type"]`` and must not inherit a subtype from
    an unrelated top classifier row (for example COAD report text adopting a
    SARC subtype).  If the active cancer code is present in the trace, use that
    row's subtype; otherwise leave the report un-subtyped.
    """
    candidate_trace = analysis.get("candidate_trace") or []
    if not candidate_trace:
        return None

    active_code = _clean_text(analysis.get("cancer_type"))
    row = None
    if active_code:
        for candidate in candidate_trace:
            if _clean_text(candidate.get("code")) == active_code:
                row = candidate
                break
        if row is None:
            return None
    else:
        row = candidate_trace[0]

    return _clean_text(row.get("winning_subtype")) or None


def _match_curated_subtype(parent_code, *candidates):
    """Return the exact curated subtype string matching any candidate."""
    if not parent_code:
        return None
    try:
        from pirlygenes.gene_sets_cancer import cancer_key_genes_subtypes

        curated_subtypes = [
            _clean_text(item) for item in cancer_key_genes_subtypes(parent_code)
        ]
    except Exception:
        return None

    curated_subtypes = [item for item in curated_subtypes if item]
    if not curated_subtypes:
        return None

    exact_lookup = {item: item for item in curated_subtypes}
    lower_lookup = {item.lower(): item for item in curated_subtypes}

    for candidate in candidates:
        raw = _clean_text(candidate)
        if not raw:
            continue
        normalized = raw.replace("-", "_")
        variants = (
            raw,
            normalized,
            raw.lower(),
            normalized.lower(),
            raw.upper(),
            normalized.upper(),
        )
        for variant in variants:
            if variant in exact_lookup:
                return exact_lookup[variant]
        for variant in variants:
            match = lower_lookup.get(str(variant).lower())
            if match:
                return match
    return None


def cancer_key_genes_lookup_for_analysis(cancer_code, analysis, ranges_df=None):
    """Return ``(panel_code, panel_subtype)`` for curated report lookups.

    Most samples use their top-level cancer code directly. Some need a
    subtype-filtered panel (for example ``SARC`` + ``leiomyosarcoma``),
    while others resolve onto a different curated code entirely (for
    example a ``SARC`` umbrella call that degeneracy-resolution upgrades
    to ``OS``).
    """
    default_code = _clean_text(cancer_code)
    resolved_code = resolved_subtype_code_for_analysis(
        analysis,
        ranges_df=ranges_df,
    )

    try:
        from pirlygenes.gene_sets_cancer import (
            cancer_key_genes_cancer_types, cancer_type_registry,
        )

        curated_codes = {_clean_text(code) for code in cancer_key_genes_cancer_types()}
        if resolved_code and resolved_code in curated_codes:
            return resolved_code, None

        if resolved_code:
            reg = cancer_type_registry()
            match = reg[reg["code"] == resolved_code]
            if not match.empty:
                row = match.iloc[0]
                parent_code = _clean_text(row.get("parent_code"))
                subtype_key = _clean_text(row.get("subtype_key"))
                suffix = ""
                if parent_code and resolved_code.startswith(parent_code + "_"):
                    suffix = resolved_code[len(parent_code) + 1 :]
                matched_subtype = _match_curated_subtype(
                    parent_code,
                    subtype_key,
                    suffix,
                )
                if parent_code and matched_subtype:
                    return parent_code, matched_subtype

        if default_code and default_code in curated_codes:
            return default_code, None
    except Exception:
        pass

    return default_code or resolved_code, None


def subtype_key_for_analysis(analysis, ranges_df=None):
    """Return the curated subtype key implied by the current analysis."""
    winning_subtype = resolved_subtype_code_for_analysis(
        analysis,
        ranges_df=ranges_df,
    )
    if not winning_subtype:
        return None

    try:
        from pirlygenes.gene_sets_cancer import cancer_type_registry

        reg = cancer_type_registry()
        match = reg[reg["code"] == winning_subtype]
        if match.empty:
            return None
        row = match.iloc[0]
        subtype_key = str(row.get("subtype_key") or "").strip()
        if subtype_key and subtype_key.lower() != "nan":
            return subtype_key
        parent_code = str(row.get("parent_code") or "").strip()
        code = str(row.get("code") or "").strip()
        if parent_code and code.startswith(parent_code + "_"):
            suffix = code[len(parent_code) + 1 :].strip().lower()
            return suffix or None
    except Exception:
        return None
    return None
