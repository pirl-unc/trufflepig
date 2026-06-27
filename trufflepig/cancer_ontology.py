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

import logging
from functools import lru_cache

import oncoref.cancer_ontology as _onc_ont
import oncoref.cancer_types as _onc_types

logger = logging.getLogger(__name__)


def _pirlygenes():
    """The pirlygenes ontology module, used only as a fallback for codes oncoref lacks."""
    from pirlygenes import gene_sets_cancer

    return gene_sets_cancer


def cancer_lineage_group(code):
    """Lineage group (Epithelial/Sarcoma/Heme/Melanoma/Neuroendocrine/Germ cell/Embryonal/CNS) for a
    cancer code — oncoref-canonical, with a pirlygenes fallback for codes oncoref lacks (e.g. ASTB)."""
    group = _onc_ont.cancer_lineage_group(code)
    if group:
        return group
    try:
        return _pirlygenes().cancer_lineage_group(code)
    except (KeyError, ValueError, ImportError):
        return None


@lru_cache(maxsize=None)
def cancer_type_registry():
    """The cancer-type registry DataFrame — oncoref-sourced metadata, constrained to the code set
    trufflepig supports (behaviour-neutral switch).

    = oncoref's rows for every code trufflepig already recognised (the pirlygenes code set),
    **plus** the pirlygenes-only codes oncoref doesn't yet carry (currently ASTB; oncoref#221),
    **minus** oncoref-only *aggregate* codes (CRC_MSI, NET) that trufflepig represents per-organ
    (COAD_MSI/READ_MSI, per-site NETs) and has no markers for. Adopting those aggregates is a
    separate, validated step — they already have curated ontology markers waiting.
    """
    onc = _onc_ont.cancer_type_registry()
    try:
        pirly = _pirlygenes().cancer_type_registry()
    except (ImportError, AttributeError):
        return onc
    pcodes = set(pirly["code"].astype(str))
    onc_codes = set(onc["code"].astype(str))
    kept = onc[onc["code"].astype(str).isin(pcodes)]                 # oncoref rows, trufflepig-known codes
    pirly_only = sorted(pcodes - onc_codes)                          # e.g. ASTB
    dropped = sorted(onc_codes - pcodes)                             # e.g. CRC_MSI, NET
    if not pirly_only:
        if dropped:
            logger.debug("cancer_type_registry: dropping %d oncoref-only aggregate codes (%s)", len(dropped), dropped)
        return kept
    import pandas as pd

    extra = pirly[pirly["code"].astype(str).isin(pirly_only)].reindex(columns=onc.columns)
    logger.debug("cancer_type_registry: +%s (pirlygenes-only), -%s (oncoref-only aggregates)", pirly_only, dropped)
    return pd.concat([kept, extra], ignore_index=True)


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


def cancer_type_subtypes_of(parent_code):
    """Child subtype codes of a (mixture) parent.

    Kept on **pirlygenes** deliberately: it's tied to which subtypes carry a curated lineage-gene
    panel (``lineage-genes.csv``), and oncoref nests RMS subtypes under ``SARC_RMS`` (a *more correct*
    2-level hierarchy) whereas trufflepig's single-level mixture refinement expects them flat under
    ``SARC``. Migrating this requires making ``_mixture_cohort_lineage_summary`` recurse first — a
    separate, validated change — so for a behaviour-neutral ontology switch this stays pirlygenes.
    """
    return _pirlygenes().cancer_type_subtypes_of(parent_code)


def is_mixture_cohort(code):
    """Whether a code's reference cohort is a multi-subtype mixture — oncoref-canonical."""
    return _onc_types.is_mixture_cohort(code)


@lru_cache(maxsize=None)
def _records_by_code():
    import pandas as pd

    records = _onc_ont.cancer_type_records()
    df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    return {str(r["code"]): r for r in df.to_dict("records")}


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
    """Full lineage_group → family → cancer_type path (oncoref) — the graph view of the ontology."""
    return _onc_ont.cancer_type_path(code)
