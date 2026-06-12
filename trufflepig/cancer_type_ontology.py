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
import math
from dataclasses import dataclass, field

from pirlygenes.gene_sets_cancer import cancer_type_registry

# --------------------------------------------------------------------------
# Ontology definition
# --------------------------------------------------------------------------

# Broad lineage for every registry family. Salivary / thymic / endocrine
# carcinomas are epithelial; embryonal (small-round-blue-cell pediatric) tumours
# get their own broad node because they are poly-phenotypic and resolved by
# context + defining alterations, not by a single differentiation program.
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


def broad_lineage(code: str, _registry=None) -> str:
    """Return the broad-lineage node for a registry code.

    Primary source is pirlygenes' canonical coarse lineage grouping
    (``cancer_lineage_group``), translated to trufflepig's vocabulary. Falls back
    to the local family map only for codes pirlygenes doesn't group, an injected
    test ``_registry``, or if the API is unavailable.
    """
    if _registry is None:
        try:
            from pirlygenes.gene_sets_cancer import cancer_lineage_group

            group = cancer_lineage_group(code)
            mapped = _PIRLYGENES_GROUP_TO_BROAD.get(str(group).strip().lower()) if group else None
            if mapped:
                return mapped
        except Exception:  # noqa: BLE001 — fall back to the local family map
            pass
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


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass
class OntologyStep:
    """One node visited during the walk."""

    level: str
    ranked: list[tuple[str, float]]
    descended_into: str | None
    candidates: list[str]
    line: str


@dataclass
class CancerTypeOntologyResult:
    """Outcome of the ontology walk.

    Attributes
    ----------
    candidates:
        Cohort codes, best-first. A single code when the walk reached a leaf;
        the near-tied neighbours (best-first) when it stopped early.
    resolved_level:
        The level at which the walk stopped (``broad`` / ``differentiation`` /
        ``organ`` / ``leaf``).
    stopped_early:
        ``True`` when a margin gate halted the walk above a leaf.
    trace:
        Readable RNA-lineage reasoning lines (one per step).
    steps:
        Structured per-node decisions backing ``trace``.
    scores:
        The per-cohort signature scores the walk consumed.
    """

    candidates: list[str]
    resolved_level: str
    stopped_early: bool
    trace: list[str] = field(default_factory=list)
    steps: list[OntologyStep] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    recall_notes: list[str] = field(default_factory=list)

    @property
    def top(self) -> str | None:
        """Most likely cohort code, or ``None`` if nothing scored."""
        return self.candidates[0] if self.candidates else None


# --------------------------------------------------------------------------
# Tree construction + walk
# --------------------------------------------------------------------------


def _build_tree(score_map: dict[str, float]) -> dict:
    """Nested {name: node} tree; each node carries its max-descendant score."""
    root = {"score": 0.0, "children": {}, "codes": set()}
    for code, score in score_map.items():
        node = root
        for name in ontology_path(code):
            if name == "root":
                node["score"] = max(node["score"], score)
                node["codes"].add(code)
                continue
            child = node["children"].setdefault(
                name, {"score": 0.0, "children": {}, "codes": set()}
            )
            child["score"] = max(child["score"], score)
            child["codes"].add(code)
            node = child
    return root


def _format_ranked(ranked: list[tuple[str, float]], limit: int = 4) -> str:
    return ", ".join(f"{name} {score:.3f}" for name, score in ranked[:limit])


def _walk(
    score_map: dict[str, float], margins: dict[str, float]
) -> CancerTypeOntologyResult:
    root = _build_tree(score_map)
    steps: list[OntologyStep] = []
    node = root
    depth = 0
    while node["children"]:
        level = _LEVEL_SEQUENCE[min(depth, len(_LEVEL_SEQUENCE) - 1)]
        ranked = sorted(
            ((name, child["score"]) for name, child in node["children"].items()),
            key=lambda kv: (-kv[1], kv[0]),
        )
        if len(ranked) == 1:
            name = ranked[0][0]
            steps.append(
                OntologyStep(
                    level=level,
                    ranked=ranked,
                    descended_into=name,
                    candidates=[name],
                    line=f"{level}: {name} is the only candidate ({ranked[0][1]:.3f}) — refine.",
                )
            )
            node = node["children"][name]
            depth += 1
            continue
        top_name, top_score = ranked[0]
        margin = margins.get(level, 0.05)
        lead = top_score - ranked[1][1]
        if lead >= margin:
            steps.append(
                OntologyStep(
                    level=level,
                    ranked=ranked,
                    descended_into=top_name,
                    candidates=[top_name],
                    line=(
                        f"{level}: {top_name} leads by {lead:.3f} "
                        f"(≥ {margin:.2f}) over [{_format_ranked(ranked)}] — refine."
                    ),
                )
            )
            node = node["children"][top_name]
            depth += 1
            continue
        tied = [name for name, score in ranked if top_score - score < margin]
        steps.append(
            OntologyStep(
                level=level,
                ranked=ranked,
                descended_into=None,
                candidates=tied,
                line=(
                    f"{level}: {top_name} and {len(tied) - 1} neighbour(s) within "
                    f"{margin:.2f} ([{_format_ranked(ranked)}]) — stop and report the set."
                ),
            )
        )
        # collect cohort codes under the tied children, best-first by score
        tied_codes = {c for name in tied for c in node["children"][name]["codes"]}
        ordered = sorted(tied_codes, key=lambda c: (-score_map.get(c, 0.0), c))
        return CancerTypeOntologyResult(
            candidates=ordered,
            resolved_level=level,
            stopped_early=True,
            trace=[s.line for s in steps],
            steps=steps,
            scores=score_map,
        )

    # reached a leaf node
    leaf_codes = sorted(node["codes"], key=lambda c: (-score_map.get(c, 0.0), c))
    return CancerTypeOntologyResult(
        candidates=leaf_codes,
        resolved_level="leaf",
        stopped_early=False,
        trace=[s.line for s in steps],
        steps=steps,
        scores=score_map,
    )


# --------------------------------------------------------------------------
# Lineage-marker recall injection
# --------------------------------------------------------------------------


def _recall_score(program_score: float) -> float:
    """Map a recall HK-ratio program score onto the signature score scale.

    Signature support_geomean runs ~0.4–0.7; well-differentiated NE (huge granin
    HK-ratios) should land high, high-grade SCLC (modest ratios) competitively.
    """
    return float(min(0.95, max(0.5, 0.5 + 0.09 * math.log1p(max(program_score, 0.0)))))


# Floor for a recall-injected score when no signature candidate competes (a pure
# no-reference NE sample whose signatures all scatter low) — keeps the NE entity
# visible. Matches the floor of :func:`_recall_score`.
_RECALL_FLOOR = 0.5


def _inject_recall(score_map: dict[str, float], proposals) -> list[str]:
    """Add recall-proposed entities to the score map; return trace notes.

    Additive — only raises a code's score (via ``max``), never lowers an existing
    screen candidate. The injected score is capped at the strongest existing
    signature score (floored at :data:`_RECALL_FLOOR`), so recall can at most
    *tie* a confident signature — surfacing both lineages on a genuine overlap
    (e.g. small-cell lung: epithelial + neuroendocrine) rather than letting a
    marker fire override a well-supported call. The strict core-granin-obligate
    gate remains the primary false-positive guard; this bounds the blast radius
    if it ever mis-fires.
    """
    notes = []
    if not proposals:
        return notes
    ceiling = max(max(score_map.values(), default=0.0), _RECALL_FLOOR)
    for proposal in proposals:
        base = min(_recall_score(proposal.program_score), ceiling)
        for entity in proposal.entities:
            bump = 0.02 if entity == proposal.subtype_hint else 0.0
            score_map[entity] = max(score_map.get(entity, 0.0), min(base + bump, ceiling))
        notes.append(proposal.rationale)
    return notes


def _apply_lineage_exclusion(score_map: dict[str, float], evidence) -> list[str]:
    """Down-weight non-epithelial candidates when epithelial markers are present.

    Multiplies each code's score by its broad lineage's demotion factor (1.0 for
    lineages not excluded). Additive-only — never raises a score.
    """
    if not evidence or not evidence.factors:
        return []
    for code in list(score_map):
        factor = evidence.factors.get(broad_lineage(code), 1.0)
        if factor != 1.0:
            score_map[code] *= factor
    return list(evidence.notes)


def classify_cancer_type_ontology(
    df_gene_expr=None,
    *,
    ranked_rows=None,
    recall_proposals=None,
    use_recall: bool = True,
    use_lineage_exclusion: bool = True,
    lineage_evidence=None,
    margins: dict[str, float] | None = None,
    top_k: int = 40,
) -> CancerTypeOntologyResult:
    """Classify a sample by walking the cancer-type ontology.

    Parameters
    ----------
    df_gene_expr:
        Clean per-sample gene-expression frame (``gene_symbol`` / ``TPM`` style),
        as consumed by :func:`rank_cancer_type_candidates`. Used for the signature
        screen (when ``ranked_rows`` is not supplied) **and** for the tumour-
        intrinsic marker channels (recall + lineage exclusion) — so it is still
        consumed for those even when ``ranked_rows`` is provided.
    ranked_rows:
        Pre-computed output of :func:`rank_cancer_type_candidates` (avoids
        re-scoring when the caller already ran it). The marker channels still run
        from ``df_gene_expr`` if it is also supplied.
    margins:
        Optional per-level margin overrides (see :data:`DEFAULT_MARGINS`).
    top_k:
        Number of candidate rows to request from the engine when scoring here.

    Returns
    -------
    CancerTypeOntologyResult
    """
    if ranked_rows is None:
        if df_gene_expr is None:
            raise ValueError("provide df_gene_expr or ranked_rows")
        from .tumor_purity import rank_cancer_type_candidates

        ranked_rows = rank_cancer_type_candidates(df_gene_expr, top_k=top_k)
    score_map = {
        row["code"]: float(row.get("support_geomean") or row["signature_score"])
        for row in ranked_rows
    }

    # Tumour-intrinsic marker evidence (recall + lineage exclusion) is computed
    # from the sample once and shared. Recall surfaces no-reference NE entities;
    # the exclusion gate down-weights non-epithelial candidates when the
    # epithelial program is present (admixture / stromal-confound correction).
    sample_tpm = hk_median = None
    need_sample = df_gene_expr is not None and (
        (use_recall and recall_proposals is None)
        or (use_lineage_exclusion and lineage_evidence is None)
    )
    if need_sample:
        from .lineage_marker_recall import marker_hk_median
        from .tumor_purity import _build_sample_tpm_by_symbol

        sample_tpm = _build_sample_tpm_by_symbol(df_gene_expr)
        # Ribosomal-free HK median: the gate denominator invariant to v4 clean-TPM.
        hk_median = marker_hk_median(sample_tpm)

    if use_recall and recall_proposals is None and sample_tpm is not None:
        from .lineage_marker_recall import recall_candidates

        recall_proposals = recall_candidates(sample_tpm, hk_median)
    recall_notes = _inject_recall(score_map, recall_proposals) if use_recall else []

    exclusion_notes = []
    if use_lineage_exclusion:
        if lineage_evidence is None and sample_tpm is not None:
            from .lineage_evidence import lineage_exclusion_evidence

            lineage_evidence = lineage_exclusion_evidence(sample_tpm, hk_median)
        exclusion_notes = _apply_lineage_exclusion(score_map, lineage_evidence)

    resolved_margins = {**DEFAULT_MARGINS, **(margins or {})}
    result = _walk(score_map, resolved_margins)
    result.recall_notes = recall_notes
    prefix = [f"[exclude] {n}" for n in exclusion_notes] + [f"[recall] {n}" for n in recall_notes]
    if prefix:
        result.trace = prefix + result.trace
    return result
