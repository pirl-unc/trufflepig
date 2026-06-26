"""Interpretable cancer-type reasoning ontology.

This module is a *reasoning layer* on top of the validated production signature
engine (:func:`trufflepig.tumor_purity.rank_cancer_type_candidates`). The engine
scores every TCGA reference cohort for a sample; this module organises those
per-cohort scores into a clinically-meaningful ontology and walks it top-down,
**refining the call only as far as the evidence supports** and otherwise
reporting the near-tied neighbours as a candidate set.

Why a layer and not a new scorer
---------------------------------
Earlier prototypes that scored lineage from within-sample marker percentiles
kept failing on real patients: a marker's rank inside one sample's transcriptome
is contaminated by tumour-microenvironment infiltrate and by the long tail of
near-zero genes (so a carcinoma with immune infiltrate scores "hematolymphoid",
and any expressed keratin clears an "epithelial" floor in a sarcoma). The
production engine avoids this with cross-cohort signature panels, purity
anchoring and family-factor correction — it calls all 11 local clinical samples
correctly. We therefore *consume* its scores rather than re-deriving lineage.

The ontology
------------
Each scored cohort sits on a path::

    root -> broad lineage -> (carcinoma) differentiation -> organ -> cohort

A node's score is the best (max) evidence among its descendant cohorts. Walking
down, at each node we either:

* **commit** to the leading child when its score beats the runner-up by at least
  the node's margin, or
* **stop** and return every child within that margin as the candidate set
  (e.g. ``{COAD, READ}`` for a colorectal sample, or ``{glandular, squamous}``
  for an ambiguous epithelial-EMT line).

Every step appends a plain-language line to a trace. The signals are RNA
transcript-abundance signatures — they are weakly correlated with protein IHC
and must not be described as staining; this output complements, and does not
replace, histopathology.

Coverage
--------
:func:`ontology_path` places **every** pirlygenes registry code on the tree
(broad lineage at minimum). Cohorts with a TCGA reference signature (~33 codes)
can refine to a leaf; rarer codes (fusion-defined or pediatric entities without
a TCGA cohort) are scored by their nearest reference neighbours and the walk
honestly stops higher up rather than guessing a specific subtype.
"""

from __future__ import annotations

import functools

from trufflepig.cancer_ontology import cancer_type_registry

# --------------------------------------------------------------------------
# Ontology definition
# --------------------------------------------------------------------------

# FALLBACK ONLY (since 5.22.77 broad_lineage consumes pirlygenes'
# ``cancer_lineage_group`` first — see ``_pirlygenes_broad_lineage``). Used when
# a code is ungrouped upstream, a test injects ``_registry``, or the API is
# unavailable. Salivary / thymic / endocrine carcinomas are epithelial; embryonal
# (small-round-blue-cell pediatric) tumours get their own broad node because they
# are poly-phenotypic and resolved by context + defining alterations.
_FAMILY_BROAD = {
    "sarcoma": "mesenchymal",
    "melanoma": "melanocytic",
    "germ-cell": "germ",
    "neuroendocrine": "neuroendocrine",
    "embryonal": "embryonal",
    # ``cns`` and ``endocrine`` are no longer single families — pirlygenes #359
    # split them into ``cns-glial/ependymal/meningeal/sellar/choroid/embryonal``
    # and ``endocrine-epithelial/-neuroendocrine``. Handled by prefix in
    # ``broad_lineage`` so future sub-splits don't re-break this.
}

# Carcinoma cohorts -> (differentiation program, organ-of-origin).
# Differentiation is the RNA-lineage split within carcinoma; organ-of-origin is
# the Tier-2 follow-up asked only once the glandular/squamous branch is reached.
_CARCINOMA_DIFFERENTIATION: dict[str, tuple[str, str]] = {
    "COAD": ("glandular", "GI"),
    "READ": ("glandular", "GI"),
    "STAD": ("glandular", "GI"),
    "ESCA": ("squamous", "GI"),
    "LUAD": ("glandular", "lung"),
    "LUSC": ("squamous", "lung"),
    "BRCA": ("glandular", "breast"),
    "PRAD": ("glandular", "prostate"),
    "KIRC": ("glandular", "kidney"),
    "KIRP": ("glandular", "kidney"),
    "KICH": ("glandular", "kidney"),
    "THCA": ("glandular", "thyroid"),
    "PAAD": ("glandular", "panc_bil"),
    "CHOL": ("glandular", "panc_bil"),
    "OV": ("glandular", "gyne"),
    "UCEC": ("glandular", "gyne"),
    "CESC": ("squamous", "gyne"),
    "HNSC": ("squamous", "head_neck"),
    "BLCA": ("urothelial", "bladder"),
    "LIHC": ("hepatocellular", "liver"),
    "MESO": ("mesothelial", "pleura"),
}

# Per-level margins for the "is the leader clearly ahead?" gate. Broader splits
# get a wider margin (a wrong broad call is the most expensive); leaf splits
# (COAD vs READ) get a narrow one because the siblings are genuinely close and
# we *want* to surface both when they tie.
DEFAULT_MARGINS = {
    "broad": 0.06,
    "differentiation": 0.05,
    "organ": 0.05,
    "leaf": 0.04,
}
_LEVEL_SEQUENCE = ["broad", "differentiation", "organ", "leaf"]

_EPITHELIAL_CODES_WITHOUT_TCGA = ("NUTM", "ADCC", "ACINIC", "NPC", "THYM", "MESO")


# pirlygenes' canonical coarse histogenesis grouping (``cancer_lineage_group``)
# is the single source of truth; translate its labels to trufflepig's broad-lineage
# vocabulary. Consuming it means new codes and taxonomy moves (NBL->Embryonal,
# NEC/NET split, THYMCA, ...) are picked up automatically instead of re-breaking a
# hardcoded family map on every pirlygenes bump.
_PIRLYGENES_GROUP_TO_BROAD = {
    "epithelial": "epithelial",
    "sarcoma": "mesenchymal",
    "heme": "hematolymphoid",
    "melanoma": "melanocytic",
    "neuroendocrine": "neuroendocrine",
    "germ cell": "germ",
    "embryonal": "embryonal",
    "cns": "neural",
}


@functools.lru_cache(maxsize=None)
def _pirlygenes_broad_lineage(code: str) -> str | None:
    """Cached translation of pirlygenes' coarse lineage group to trufflepig's
    vocabulary (``cancer_lineage_group`` is uncached upstream, and ``broad_lineage``
    runs in per-candidate hot loops). ``None`` when the code is ungrouped or the
    API is unavailable, so the caller falls back to the local family map.
    """
    try:
        from trufflepig.cancer_ontology import cancer_lineage_group

        group = cancer_lineage_group(code)
    except Exception:  # noqa: BLE001 — caller falls back to the local family map
        return None
    return _PIRLYGENES_GROUP_TO_BROAD.get(str(group).strip().lower()) if group else None


def broad_lineage(code: str, _registry=None) -> str:
    """Return the broad-lineage node for a registry code.

    Primary source is pirlygenes' canonical coarse lineage grouping
    (``cancer_lineage_group``), translated to trufflepig's vocabulary. Falls back
    to the local family map (below) only for codes pirlygenes doesn't group, an
    injected test ``_registry``, or if the API is unavailable.
    """
    if _registry is None:
        mapped = _pirlygenes_broad_lineage(code)
        if mapped:
            return mapped
    reg = _registry if _registry is not None else _registry_families()
    family = str(reg.get(code, "")) if reg is not None else ""
    if family in _FAMILY_BROAD:
        return _FAMILY_BROAD[family]
    if family.startswith("heme"):
        return "hematolymphoid"
    # CNS sub-families (pirlygenes #359): all CNS/neural at the broad level,
    # except the embryonal class (MBL/ATRT) which keeps its own poly-phenotypic
    # node. Covers the legacy bare ``cns`` too.
    if family == "cns" or family.startswith("cns-"):
        return "embryonal" if "embryonal" in family else "neural"
    # Endocrine split (pirlygenes #359 follow-up): neuroendocrine vs epithelial.
    if family.startswith("endocrine"):
        return "neuroendocrine" if "neuroendocrine" in family else "epithelial"
    if (
        family.startswith("carcinoma")
        or family in ("salivary", "thymic")
        or code in _EPITHELIAL_CODES_WITHOUT_TCGA
    ):
        return "epithelial"
    return "other"


def ontology_path(code: str) -> list[str]:
    """Return the root->leaf node path a cohort code occupies on the ontology."""
    broad = broad_lineage(code)
    if broad == "epithelial" and code in _CARCINOMA_DIFFERENTIATION:
        differentiation, organ = _CARCINOMA_DIFFERENTIATION[code]
        return ["root", broad, differentiation, organ, code]
    return ["root", broad, code]


# Registry lookup (family by code), loaded lazily on first use so importing this
# module stays cheap.
@functools.lru_cache(maxsize=1)
def _registry_families() -> dict[str, str]:
    reg = cancer_type_registry()
    return dict(zip(reg["code"].astype(str), reg["family"].astype(str)))


