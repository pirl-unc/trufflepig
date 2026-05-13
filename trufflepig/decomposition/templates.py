# Licensed under the Apache License, Version 2.0

"""Sample decomposition templates.

Templates define broad non-tumor compartments that are plausible for a sample.
They deliberately avoid unsupported fine-grained immune splits and avoid
including an "origin normal epithelium" column that steals retained lineage
signal away from tumor cells in mixed samples.
"""

from ..tumor_purity import CANCER_TO_TISSUE

# ── Hierarchical tissue categories ──────────────────────────────────────
# Site-specific host components are scored against a *category* of bulk
# tissues rather than a single HPA cell type.  For the NNLS, the best-
# matching member tissue is selected as the reference column.
#
# Why: HPA single-cell profiles for astrocytes and neurons have
# housekeeping-gene medians of ~20 nTPM (vs ~350 for immune/stroma),
# likely from harsh brain-tissue dissociation.  HK-normalisation inflates
# these references ~17×, making the NNLS think a 75%-brain sample has
# <1% brain signal.  Bulk tissue references (cerebral_cortex etc.) have
# normal HK medians (~380) and avoid this distortion entirely.
#
# The best-match selection within each category captures within-category
# variation (e.g. a cerebellar met matches cerebellum while a cortical
# met matches cerebral_cortex).  Correlation is computed only on
# category-enriched genes (>2× cross-tissue median) to prevent broadly
# expressed genes from dominating the match.

TISSUE_CATEGORIES = {
    "CNS": [
        "cerebral_cortex",
        "cerebellum",
        "hippocampal_formation",
        "amygdala",
        "basal_ganglia",
        "hypothalamus",
        "midbrain",
        "choroid_plexus",
        "spinal_cord",
    ],
    "liver": ["liver"],
    "lung": ["lung"],
    "kidney": ["kidney"],
    "GI_lower": ["colon", "rectum", "appendix", "small_intestine"],
    "GI_upper": ["stomach", "esophagus", "duodenum"],
    "bone_marrow": ["bone_marrow"],
    "skin": ["skin"],
    "adrenal": ["adrenal_gland"],
    "muscle": ["heart_muscle", "skeletal_muscle", "smooth_muscle"],
    "immune_lymphoid": ["lymph_node", "spleen", "thymus", "tonsil"],
}

# Map from decomposition component name → tissue category.
# Components listed here use bulk tissue composite references (best-match
# selection) instead of HPA single-cell profiles.
# Only components whose HPA single-cell reference has a distorted HK
# median are routed through the composite approach.  Components with
# normal HK medians (osteoblast=428, marrow_stroma=428, melanocyte=393)
# keep their HPA profiles — using the broad bulk tissue reference would
# let them absorb unrelated signal (e.g. bone_marrow absorbs immune).
COMPONENT_TO_CATEGORY = {
    "hepatocyte": "liver",
    "pneumocyte": "lung",
    "astrocyte": "CNS",
    "neuron": "CNS",
    "adrenal_cortical": "adrenal",
    "keratinocyte": "skin",
    # Optional compartments (#59 items 2-4) are registered only in
    # ``signature.COMPONENT_TO_HPA`` — they resolve directly to HPA
    # single-cell rows (Adipocytes / Schwann cells / Erythroid cells)
    # and don't use the COMPONENT_TO_CATEGORY composite-tissue path.
}


# ---------- Optional-compartment gating (#59 items 2-4) ----------

# Each entry describes a compartment that should be *conditionally*
# added to a template when the sample carries enough marker signal to
# justify absorbing it. The compartment never fires by default; the
# detection hook below is the only writer of the allowlist.
#
# Schema:
#   compartment_name → {
#     "markers": [gene symbols, summed],
#     "min_tpm_sum": gate threshold on the sum,
#     "templates": {template names where this compartment is allowed},
#     "cancer_types": optional set — when present, further restrict
#         to specific cancer codes (e.g. Schwann only for cancers with
#         known perineural-invasion rates). ``None`` means any cancer.
#   }
#
# Adding a compartment:
#   1. Populate this dict entry.
#   2. Register a reference vector in
#      ``signature.COMPONENT_TO_HPA`` (or ``COMPONENT_MARKERS``).
#   3. Register a row in ``COMPONENT_TO_CATEGORY`` if downstream
#      bulk-tissue lookups need it.
#
# That's it — the engine's detection + template-expansion path
# discovers the new entry automatically.
OPTIONAL_COMPARTMENT_GATES = {
    "adipocyte": {
        "markers": ["ADIPOQ", "FABP4", "PLIN1", "LEP", "CIDEA", "CIDEC"],
        "min_tpm_sum": 50.0,
        "templates": {"solid_primary", "met_soft_tissue"},
        "cancer_types": {"BRCA", "SARC"},  # breast + retroperitoneal LPS
    },
    "schwann": {
        # Canonical peripheral-nerve / Schwann markers. MPZ + PMP22
        # are myelin-specific; S100B is pan-neural support.
        "markers": ["MPZ", "PMP22", "S100B", "GFAP"],
        "min_tpm_sum": 30.0,
        "templates": {"solid_primary"},
        # Cancers with well-documented high perineural-invasion rates.
        "cancer_types": {"PRAD", "PAAD", "HNSC", "CHOL"},
    },
    "erythroid_solid": {
        # Contamination gate — poorly-flushed solid-tumor samples carry
        # hemoglobin transcripts. Surfaces as the existing ``erythroid``
        # compartment (already in signature.COMPONENT_TO_HPA). Listed
        # under a distinct key so the gate config stays self-contained
        # and the engine can translate ``erythroid_solid`` → the
        # existing ``erythroid`` component when appending.
        "markers": ["HBA1", "HBA2", "HBB", "ALAS2"],
        "min_tpm_sum": 100.0,
        "templates": {
            "solid_primary",
            "met_liver",
            "met_brain",
            "met_lung",
            "met_bone",
            "met_peritoneal",
            "met_adrenal",
            "met_skin",
            "met_soft_tissue",
            "met_lymph_node",
        },
        "cancer_types": None,  # any solid cancer
    },
}

# Translate gate keys that don't match their real compartment name.
# ``erythroid_solid`` reuses the existing ``erythroid`` NNLS component.
_GATE_KEY_TO_COMPONENT = {
    "erythroid_solid": "erythroid",
}


def _detect_optional_compartments(
    sample_tpm_by_symbol,
    cancer_type=None,
    template_name=None,
):
    """Return the list of optional NNLS compartments to append.

    For each entry in :data:`OPTIONAL_COMPARTMENT_GATES`, this checks
    that the sample carries enough marker-gene signal *and* that the
    requested ``template_name`` / ``cancer_type`` are on the entry's
    allowlist. Compartments that pass all gates are appended to the
    template (see :func:`get_template_components`).

    Contract: with an empty or mismatched sample, the return value is
    ``[]`` and the engine's decomposition path stays byte-identical.
    """
    if not sample_tpm_by_symbol:
        return []
    detected = []
    for gate_key, gate in OPTIONAL_COMPARTMENT_GATES.items():
        if template_name is not None and gate.get("templates") is not None:
            if template_name not in gate["templates"]:
                continue
        if cancer_type is not None and gate.get("cancer_types") is not None:
            if cancer_type not in gate["cancer_types"]:
                continue
        marker_sum = sum(
            float(sample_tpm_by_symbol.get(g, 0.0) or 0.0) for g in gate["markers"]
        )
        if marker_sum >= gate["min_tpm_sum"]:
            component = _GATE_KEY_TO_COMPONENT.get(gate_key, gate_key)
            detected.append(component)
    return detected


# Shared immune components used across most solid tumor templates
_SOLID_IMMUNE = [
    "T_cell",
    "B_cell",
    "plasma",
    "NK",
    "myeloid",
]

_SOLID_STROMA = ["fibroblast", "endothelial"]

TEMPLATES = {
    "pure_population": {
        "components": [],  # just tumor; filled dynamically
        "description": "Cell line or sorted population",
    },
    "solid_primary": {
        "components": _SOLID_IMMUNE + _SOLID_STROMA,
        "description": "Solid tumor primary resection",
    },
    "met_lymph_node": {
        "components": _SOLID_IMMUNE + _SOLID_STROMA,
        "host_tissue": "lymph_node",
        "description": "Metastasis in lymph node",
    },
    "met_liver": {
        "components": _SOLID_IMMUNE + _SOLID_STROMA + ["hepatocyte"],
        "host_tissue": "liver",
        "description": "Metastasis in liver",
    },
    "met_brain": {
        # Both astrocyte and neuron resolve to the same bulk tissue via
        # COMPONENT_TO_CATEGORY → "CNS".  The NNLS sees two identical
        # columns and splits the weight evenly — report their fractions
        # summed as "CNS parenchyma" rather than reading them literally.
        "components": [
            "T_cell",
            "myeloid",
            "fibroblast",
            "endothelial",
            "astrocyte",
            "neuron",
        ],
        "host_tissue": "cerebral_cortex",
        "description": "Metastasis in brain (reduced immune diversity)",
    },
    "met_lung": {
        "components": _SOLID_IMMUNE + _SOLID_STROMA + ["pneumocyte"],
        "host_tissue": "lung",
        "description": "Metastasis in lung",
    },
    "met_bone": {
        "components": _SOLID_IMMUNE + ["endothelial", "osteoblast", "marrow_stroma"],
        "host_tissue": "bone_marrow",
        "description": "Metastasis in bone",
    },
    "met_peritoneal": {
        "components": _SOLID_IMMUNE + _SOLID_STROMA + ["mesothelial"],
        "host_tissue": "appendix",
        "description": "Peritoneal metastasis",
    },
    "met_adrenal": {
        "components": [
            "T_cell",
            "myeloid",
            "fibroblast",
            "endothelial",
            "adrenal_cortical",
        ],
        "host_tissue": "adrenal_gland",
        "description": "Metastasis in adrenal gland",
    },
    "met_skin": {
        "components": [
            "T_cell",
            "myeloid",
            "NK",
            "fibroblast",
            "endothelial",
            "keratinocyte",
            "melanocyte",
        ],
        "host_tissue": "skin",
        "description": "Metastasis in skin",
    },
    "met_soft_tissue": {
        "components": _SOLID_IMMUNE + _SOLID_STROMA,
        "host_tissue": "smooth_muscle",
        "description": "Metastasis in soft tissue / retroperitoneum",
    },
    "heme_marrow": {
        "components": [
            "normal_myeloid",
            "normal_lymphoid",
            "erythroid",
            "endothelial",
            "marrow_stroma",
        ],
        "description": "Heme malignancy in bone marrow",
    },
    "heme_nodal": {
        "components": [
            "T_cell",
            "B_cell",
            "plasma",
            "myeloid",
            "fibroblast",
            "endothelial",
        ],
        "host_tissue": "lymph_node",
        "description": "Heme malignancy in lymph node",
    },
    "heme_blood": {
        "components": ["T_cell", "B_cell", "NK", "normal_myeloid", "erythroid"],
        "description": "Heme malignancy in peripheral blood",
    },
}

_PRIMARY_HOST_TISSUES = {
    # Lower-GI primaries often look colon/rectum/appendix-like in bulk and
    # should not be forced into a single exact bowel segment.
    "COAD": ["colon", "rectum", "appendix"],
    "READ": ["rectum", "colon", "appendix"],
}

# Cancer codes with a defensible matched-normal parent tissue — enables an
# optional ``matched_normal_<tissue>`` compartment in solid_primary so that
# admixed benign parent tissue (adjacent benign prostate in a PRAD resection,
# adjacent mucosa in a COAD biopsy, uterine smooth muscle in a LMS) is
# absorbed as non-tumor signal rather than attributed to tumor cells.
#
# Scope note: mesenchymal / glial / melanocytic / heme cancers lack a single
# stable benign analog at the **parent** level. For SARC the analog depends
# on subtype (LMS → smooth muscle; liposarcoma → adipose; MPNST → Schwann),
# so the parent SARC code is deliberately absent and the classifier's
# ``winning_subtype`` (#171) routes per-subtype. UPS / MFS / UPS-like /
# synovial sarcoma / angiosarcoma have no defensible benign counterpart and
# stay on the unassigned path. TGCT germ-cell origin and MESO serosal
# mesothelium are also excluded for reference-quality reasons.
MATCHED_NORMAL_TISSUE = {
    # Epithelial primaries
    "BLCA": "urinary_bladder",
    "BRCA": "breast",
    "CESC": "cervix",
    "CHOL": "gallbladder",
    "COAD": "colon",
    "ESCA": "esophagus",
    "HNSC": "tongue",
    "KICH": "kidney",
    "KIRC": "kidney",
    "KIRP": "kidney",
    "LIHC": "liver",
    "LUAD": "lung",
    "LUSC": "lung",
    "OV": "ovary",
    "PAAD": "pancreas",
    "PRAD": "prostate",
    "READ": "rectum",
    "STAD": "stomach",
    "THCA": "thyroid_gland",
    "UCEC": "endometrium",
    # SARC subtypes (issue #51, enabled by mixture-cohort winning_subtype
    # from #171). LMS → smooth muscle; all liposarcoma flavors → mature
    # adipose. MPNST → Schwann would require a non-HPA reference and is
    # deferred. UPS / MFS / synovial / angiosarcoma stay unassigned.
    "SARC_LMS": "smooth_muscle",
    "SARC_DDLPS": "adipose_tissue",
    "SARC_WDLPS": "adipose_tissue",
    "SARC_MYXLPS": "adipose_tissue",
    "SARC_LPS_UNSPEC": "adipose_tissue",
}

# Backwards-compatibility alias kept so external importers that used
# ``EPITHELIAL_MATCHED_NORMAL_TISSUE`` keep working. Reads the merged map
# and filters out the sarcoma subtypes at access time; prefer the new name.
EPITHELIAL_MATCHED_NORMAL_TISSUE = {
    code: tissue
    for code, tissue in MATCHED_NORMAL_TISSUE.items()
    if not code.startswith("SARC_")
}


def matched_normal_component(cancer_type, winning_subtype=None):
    """Return matched-normal component name, or ``None`` when unavailable.

    ``winning_subtype`` is the mixture-cohort classifier's per-subtype
    call (#171). When set, it takes precedence over the parent code so
    a SARC parent with ``winning_subtype=SARC_LMS`` picks up
    ``matched_normal_smooth_muscle`` instead of falling to the unassigned
    default.
    """
    code = winning_subtype or cancer_type
    if code is None:
        return None
    tissue = MATCHED_NORMAL_TISSUE.get(code)
    if tissue is None:
        return None
    return f"matched_normal_{tissue}"


def epithelial_matched_normal_component(cancer_type):
    """Deprecated alias for :func:`matched_normal_component`.

    Kept so external importers keep working. Does not consult the
    mixture-cohort ``winning_subtype`` — callers that need the SARC
    subtype-aware path should use :func:`matched_normal_component`.
    """
    return matched_normal_component(cancer_type)


_TEMPLATE_HOST_TISSUES = {
    # Retroperitoneal/deep soft tissue biopsies can look like a mix of muscle
    # and adipose rather than one exact reference tissue.
    "met_soft_tissue": ["smooth_muscle", "skeletal_muscle", "adipose_tissue"],
}

# Tumor cell-of-origin → which constraint categories to relax
# (these genes become lineage markers instead of constraints)
TUMOR_ORIGIN_TYPE = {
    # Mesenchymal tumors: relax fibroblast constraints
    "SARC": "fibroblast",
    # B-cell tumors: relax IG/B-cell constraints
    "DLBC": "B_cell",
    "CLL": "B_cell",
    # T-cell tumors: relax TCR/T-cell constraints
    # (future: T-cell lymphoma types)
    # Myeloid tumors: relax myeloid constraints
    "LAML": "normal_myeloid",
    # All others: epithelial origin, no standard constraints affected
}


def get_template_components(
    template_name,
    cancer_type=None,
    winning_subtype=None,
    detected_compartments=None,
):
    """Get the full component list for a template + cancer type.

    For ``template_name == "solid_primary"`` and a cancer code with a
    defensible benign parent tissue in :data:`MATCHED_NORMAL_TISSUE`, a
    ``matched_normal_<tissue>`` component is appended so admixed benign
    parent tissue is absorbed as non-tumor signal rather than attributed
    to tumor cells (issue #50).

    ``winning_subtype`` is the mixture-cohort classifier's per-subtype
    hypothesis (#171). When set it takes precedence over ``cancer_type``
    for matched-normal lookup — the SARC subtype-aware path (#51)
    routes SARC_LMS → smooth muscle, liposarcoma flavors → adipose,
    etc., while leaving UPS / MFS / synovial / angiosarcoma unassigned.

    ``detected_compartments`` is the output of
    :func:`_detect_optional_compartments` (#59 items 2-4). Each entry is
    appended to the component list *after* matched-normal; duplicates
    are filtered so a compartment that's already in the template (e.g.
    ``erythroid`` in a heme template, but also detected on a solid
    sample that somehow lands on that template) only appears once.

    Returns
    -------
    components : list of str
        Component names including "tumor" as the first entry.
    """
    tmpl = TEMPLATES[template_name]
    components = ["tumor"] + list(tmpl["components"])
    if template_name == "solid_primary":
        matched = matched_normal_component(
            cancer_type,
            winning_subtype=winning_subtype,
        )
        if matched is not None:
            components.append(matched)
    if detected_compartments:
        already = set(components)
        for comp in detected_compartments:
            if comp not in already:
                components.append(comp)
                already.add(comp)
    return components


def get_template_extra_components(template_name):
    """Return components beyond the shared solid immune/stroma basis."""
    return [
        comp
        for comp in TEMPLATES[template_name]["components"]
        if comp not in _SOLID_IMMUNE + _SOLID_STROMA
    ]


def get_template_host_tissues(template_name, cancer_type=None):
    """Return one or more host tissues used to score a template."""
    if template_name == "solid_primary" and cancer_type is not None:
        tissues = _PRIMARY_HOST_TISSUES.get(cancer_type)
        if tissues:
            return list(tissues)
        tissue = CANCER_TO_TISSUE.get(cancer_type)
        return [tissue] if tissue else []

    tissues = _TEMPLATE_HOST_TISSUES.get(template_name)
    if tissues:
        return list(tissues)

    tissue = TEMPLATES[template_name].get("host_tissue")
    return [tissue] if tissue else []


def get_template_host_tissue(template_name, cancer_type=None):
    """Return the matched host tissue used to score a template, if any."""
    tissues = get_template_host_tissues(template_name, cancer_type=cancer_type)
    return tissues[0] if tissues else None


def list_templates():
    """Return list of (name, description) for all templates."""
    return [(name, t["description"]) for name, t in TEMPLATES.items()]
