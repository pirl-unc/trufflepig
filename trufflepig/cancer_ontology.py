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


# Aggregate codes oncoref carries as first-class registry rows but which trufflepig does NOT classify
# per-sample: it represents them per-organ (CRC_MSI → COAD_MSI/READ_MSI) or per-site (NET → the
# individual NET_* cohorts). Dropping them keeps ``resolve_cancer_type`` failing loudly rather than
# emitting a vague whole-group call that has no dedicated markers/reference of its own.
_UNSUPPORTED_AGGREGATES = frozenset({"NET", "CRC_MSI"})


@lru_cache(maxsize=None)
def cancer_type_registry():
    """The cancer-type registry DataFrame — oncoref-canonical, minus the aggregate codes trufflepig
    represents per-organ (:data:`_UNSUPPORTED_AGGREGATES`).

    Since pirlygenes now re-exports oncoref's registry, the two code sets are identical, so this is
    just oncoref's registry with the per-organ aggregates filtered out. (The historical pirlygenes
    merge — extra pirlygenes-only codes like ASTB, minus oncoref-only aggregates — is obsolete now
    that oncoref carries every code, incl. ASTB.)
    """
    reg = _onc_ont.cancer_type_registry()
    return reg[~reg["code"].astype(str).isin(_UNSUPPORTED_AGGREGATES)].reset_index(drop=True)


def resolve_cancer_type(*args, **kwargs):
    """Normalize a free-text / alias cancer label to a **trufflepig-supported** registry code.

    Both the oncoref resolution and the pirlygenes fallback are validated against the merged
    :func:`cancer_type_registry`, which is trufflepig's supported code set (oncoref ∩ pirlygenes
    + pirlygenes-only codes like ASTB − oncoref-only aggregates like NET / CRC_MSI that have no
    markers or reference column here):

    - oncoref resolving to a SUPPORTED code → returned (the common path).
    - oncoref resolving to a DROPPED aggregate (NET, CRC_MSI) → NOT returned: it would only fail
      later in purity/reference lookup. Try the pirlygenes mapping; if that also doesn't land on a
      supported code, raise — so an unsupported code fails loudly at resolve time.
    - oncoref unable to resolve at all → pirlygenes fallback for the pirlygenes-only codes (ASTB).

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
            f"(an oncoref-only aggregate dropped from the merged registry)"
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
