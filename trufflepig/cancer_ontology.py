"""Canonical cancer-type ontology for trufflepig.

trufflepig consumes its cancer-type ontology from **oncoref** (`oncoref.cancer_ontology` /
`oncoref.cancer_types`) — the canonical, richer source (full lineage_group → family → type graph,
per-type ``normal_tissue_code``, ancestors/descendants/path). pirlygenes keeps the *gene↔biology*
panels (lineage genes, housekeeping, therapy targets, biomarkers); those are imported from
``pirlygenes.gene_sets_cancer`` directly and are NOT re-exported here.

A thin **pirlygenes fallback** covers the handful of codes oncoref does not yet carry (currently only
``ASTB`` — tracked at oncoref#221), so the migration is behaviour-neutral: every code that resolves
today keeps resolving.

It also surfaces the richer per-type structure oncoref provides — ``cancer_family`` and
``normal_tissue_code`` — which is the basis for category-informed (family-level, e.g. heme
B/T/myeloid) decomposition backgrounds.
"""
from __future__ import annotations

from functools import lru_cache

import oncoref.cancer_ontology as _onc_ont
import oncoref.cancer_types as _onc_types


def _pirlygenes():
    """The pirlygenes ontology module, used only as a fallback for codes oncoref lacks."""
    from pirlygenes import gene_sets_cancer

    return gene_sets_cancer


def cancer_lineage_group(code):
    """Lineage group (Epithelial/Sarcoma/Heme/Melanoma/Neuroendocrine/Germ cell/Embryonal/CNS) for a
    cancer code — oncoref-canonical."""
    return _onc_ont.cancer_lineage_group(code)


@lru_cache(maxsize=None)
def cancer_type_registry():
    """The oncoref-canonical cancer-type registry DataFrame.

    Classification support is described by the registry's typed reference fields. In particular,
    ``reference_source == "member_union"`` makes grouping codes such as NET and CRC_MSI valid
    expression-reference targets even though they do not have a single per-sample source cohort.
    """
    return _onc_ont.cancer_type_registry()


def resolve_cancer_type(*args, **kwargs):
    """Normalize a free-text / alias cancer label to a **trufflepig-supported** registry code.

    Resolution is validated against :func:`cancer_type_registry`:

    - oncoref resolving to a SUPPORTED code → returned (the common path).
    - oncoref unable to resolve at all → a pirlygenes fallback covers any label that only
      pirlygenes' alias table maps (e.g. curated display aliases oncoref doesn't carry).

    Genuinely-unknown labels still raise (the original oncoref error).
    """
    supported = set(cancer_type_registry()["code"].astype(str))
    onc_err: Exception
    try:
        resolved = _onc_ont.resolve_cancer_type(*args, **kwargs)
    except (KeyError, ValueError) as exc:
        onc_err = exc
    else:
        if str(resolved) in supported:
            return resolved
        onc_err = ValueError(
            f"{resolved!r} resolved by oncoref but is not a trufflepig-supported code "
            f"(the canonical registry has no matching row)"
        )
    try:
        pirly_resolved = _pirlygenes().resolve_cancer_type(*args, **kwargs)
    except (KeyError, ValueError, ImportError, AttributeError):
        raise onc_err
    if str(pirly_resolved) in supported:
        return pirly_resolved
    raise onc_err


@lru_cache(maxsize=None)
def _children_by_parent() -> dict[str, list[str]]:
    """``{parent_code: [direct child codes]}`` from the (oncoref) registry."""
    reg = cancer_type_registry()
    out: dict[str, list[str]] = {}
    for code, parent in zip(reg["code"].astype(str), reg["parent_code"].fillna("").astype(str)):
        p = parent.strip()
        if p and p.lower() not in ("nan", "none"):
            out.setdefault(p, []).append(code)
    return out


def cancer_type_subtypes_of(parent_code, *, recursive: bool = True):
    """Registry subtype codes of a (mixture) parent.

    Since oncoref nests some subtypes under intermediate tiers (a *more correct* 2-level hierarchy:
    ``SARC_LPS`` -> ``SARC_DDLPS``/``SARC_MYXLPS``/…, ``SARC_RMS`` -> the RMS leaves, ``SARC_ESS`` ->
    its leaves), this returns the **transitive** set of all descendant tiles by default — the flat
    leaf set trufflepig's single-level mixture / centroid refinement enumerates (``SARC`` -> every
    ``SARC_*``). Pass ``recursive=False`` for direct children only.
    """
    kids = _children_by_parent()
    if not recursive:
        return list(kids.get(str(parent_code), []))
    out: list[str] = []
    seen: set[str] = set()
    stack = list(kids.get(str(parent_code), []))
    while stack:
        code = stack.pop()
        if code in seen:
            continue
        seen.add(code)
        out.append(code)
        stack.extend(kids.get(code, []))
    return out


def is_mixture_cohort(code):
    """Whether a code's reference cohort is a multi-subtype mixture — oncoref-canonical."""
    return _onc_types.is_mixture_cohort(code)


@lru_cache(maxsize=None)
def _records_by_code():
    import pandas as pd

    records = _onc_ont.cancer_type_records()
    df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    by_code = {str(r["code"]): r for r in df.to_dict("records")}
    # Fallback-only supported codes (e.g. ASTB; oncoref#221) are in the merged cancer_type_registry
    # but NOT in oncoref's records — add their registry row so cancer_family / registry_parent_code /
    # normal_tissue_code don't return empty metadata for a code resolve_cancer_type accepts. The
    # registry carries fewer columns than the oncoref record (e.g. no normal_tissue_code), so those
    # fields degrade to '' via the helpers' .get() — still strictly better than no record at all.
    for r in cancer_type_registry().to_dict("records"):
        code = str(r["code"])
        if code not in by_code:
            by_code[code] = r
    return by_code


def registry_parent_code(code):
    """Direct registry parent of a code (e.g. COAD→CRC, SARC_OS→SARC), or '' if top-level."""
    rec = _records_by_code().get(str(code))
    parent = str((rec or {}).get("parent_code") or "").strip()
    return "" if parent.lower() in ("", "nan", "none") else parent


MOLECULAR_STATUS_SUFFIXES = (
    "_MSI",
    "_MSIH",
    "_DMMR",
    "_MMRD",
    "_MSS",
    "_PMMR",
    "_MMRP",
    "_CIN",
    "_CNL",
    "_CNH",
    "_HPV_pos",
    "_HPV_neg",
    "_HPVpos",
    "_HPVneg",
    "_EGFR",
    "_ALK",
    "_KRAS",
    "_BRAF",
    "_ERBB2",
    "_HER2",
    "_ROS1",
    "_RET",
    "_MET",
    "_NTRK",
    "_IDHmut",
    "_IDHwt",
)


def molecular_status_parent_code(code):
    """Cancer entity for an orthogonal molecular/status code, or ''.

    Examples: ``READ_MSI`` → ``READ``, ``HNSC_HPV_pos`` → ``HNSC``.
    Ordinary entity codes are intentionally not rolled up here: ``READ``
    stays ``''`` rather than becoming its registry parent ``CRC``.
    """

    text = str(code or "").strip()
    if not text:
        return ""
    for suffix in MOLECULAR_STATUS_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return ""


def subtype_display_parent_code(code):
    """Display parent for a true subtype code in compact evidence views.

    This helper is deliberately for display/nomenclature. It should not be
    used to decide whether two top-level entities are biologically equivalent.
    """

    text = str(code or "").strip()
    if "_" not in text:
        return ""
    parent = registry_parent_code(text)
    if parent and parent != text:
        return parent
    if text.startswith("SARC_"):
        return "SARC"
    return ""


def _entity_base_code(code):
    text = str(code or "").strip()
    return molecular_status_parent_code(text) or text


def _registry_ancestors(code):
    text = _entity_base_code(code)
    ancestors = []
    seen = set()
    while text and text not in seen:
        ancestors.append(text)
        seen.add(text)
        text = registry_parent_code(text)
    return tuple(ancestors)


def cancer_codes_entity_compatible(left, right):
    """Whether two codes refer to the same report entity or parent/subtype chain.

    This is stricter than lineage/family compatibility. ``READ`` and ``COAD``
    are not entity-compatible here even though both are CRC-family tumors;
    callers with a model that is explicitly trained at a broader molecular
    context layer can add that context-specific rule separately.
    """

    left_base = _entity_base_code(left)
    right_base = _entity_base_code(right)
    if not left_base or not right_base:
        return False
    if left_base == right_base:
        return True
    left_ancestors = set(_registry_ancestors(left_base))
    right_ancestors = set(_registry_ancestors(right_base))
    return left_base in right_ancestors or right_base in left_ancestors


def cancer_codes_context_compatible(left, right, *, context_code=None):
    """Whether two codes are compatible for an explicitly broader context.

    Use this for model/report contexts that are intentionally broader than a
    single registry entity, such as CRC-level mismatch-repair RNA context for
    COAD and READ. Without ``context_code`` this is just entity compatibility.
    """

    if cancer_codes_entity_compatible(left, right):
        return True
    context = str(context_code or "").strip()
    if not context:
        return False
    left_base = _entity_base_code(left)
    right_base = _entity_base_code(right)
    if not left_base or not right_base:
        return False
    left_ancestors = set(_registry_ancestors(left_base))
    right_ancestors = set(_registry_ancestors(right_base))
    return context in left_ancestors and context in right_ancestors


def cancer_family(code):
    """The registry *family* of a code (e.g. heme-bcell, heme-tcell, heme-myeloid, carcinoma-gi,
    sarcoma) — the middle layer between lineage group and entity, used for category-informed
    decomposition backgrounds."""
    rec = _records_by_code().get(str(code))
    family = str((rec or {}).get("family") or "").strip()
    return "" if family.lower() in ("", "nan", "none") else family


def normal_tissue_code(code):
    """The normal tissue whose program should be subtracted as this tumor's background (e.g.
    DLBC→lymph_node, LAML→bone_marrow, CTCL→skin) — the enabler for category-informed (family-level,
    incl. heme B/T/myeloid) decomposition. '' when oncoref has no normal-tissue mapping for the code."""
    rec = _records_by_code().get(str(code))
    tissue = str((rec or {}).get("normal_tissue_code") or "").strip()
    return "" if tissue.lower() in ("", "nan", "none") else tissue


def cancer_type_path(code):
    """Full lineage_group → family → cancer_type path (oncoref) — the graph view of the ontology.

    Fallback-only codes (e.g. ASTB) aren't in oncoref's graph, so the direct call raises; degrade
    to a best-effort path from the merged-registry metadata (lineage group → family → code) rather
    than crash a caller that legitimately resolved the code."""
    try:
        return _onc_ont.cancer_type_path(code)
    except (KeyError, ValueError):
        import pandas as pd

        grp = cancer_lineage_group(code) or ""
        fam = cancer_family(code) or ""
        rows = [("lineage_group", grp), ("family", fam), ("cancer_type", str(code))]
        return pd.DataFrame(
            [{"level": lvl, "code": val} for lvl, val in rows if val]
        )
