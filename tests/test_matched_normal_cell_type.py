"""Tests for the cell-type-resolved matched-normal reference map (#214)
and its proxy extension (#216)."""

from trufflepig.decomposition.signature import (
    MATCHED_NORMAL_CELL_TYPE,
    MATCHED_NORMAL_CELL_TYPE_PROXY,
    get_matched_normal_cell_type,
    _load_hpa_cell_types,
)


def test_direct_mappings_hit_real_hpa_cell_types():
    """Every cell type listed in the direct map must exist in
    hpa-cell-type-expression.csv — prevents silent typos."""
    hpa_columns = set(_load_hpa_cell_types().columns)
    for tissue, cell_types in MATCHED_NORMAL_CELL_TYPE.items():
        for cell_type in cell_types:
            assert cell_type in hpa_columns, (
                f"{tissue!r} → {cell_type!r} not in HPA atlas"
            )


def test_proxy_mappings_hit_real_hpa_cell_types():
    hpa_columns = set(_load_hpa_cell_types().columns)
    for tissue, entry in MATCHED_NORMAL_CELL_TYPE_PROXY.items():
        for cell_type in entry["cell_types"]:
            assert cell_type in hpa_columns, (
                f"proxy {tissue!r} → {cell_type!r} not in HPA atlas"
            )


def test_proxy_confidences_in_valid_range():
    """Proxies must declare confidence in (0, 1); direct entries are
    implicit 1.0. Anything below 0.3 should have been rejected during
    curation (use bulk fallback instead)."""
    for tissue, entry in MATCHED_NORMAL_CELL_TYPE_PROXY.items():
        confidence = entry["confidence"]
        assert 0.3 <= confidence < 1.0, (
            f"proxy {tissue!r} confidence {confidence} out of valid range"
        )


def test_proxy_entries_have_rationale():
    for tissue, entry in MATCHED_NORMAL_CELL_TYPE_PROXY.items():
        rationale = entry.get("rationale", "")
        assert rationale and len(rationale) > 20, f"proxy {tissue!r} needs a rationale"


def test_get_matched_normal_cell_type_direct():
    cell_types, confidence, source = get_matched_normal_cell_type("prostate")
    assert cell_types == ["Prostatic glandular cells"]
    assert confidence == 1.0
    assert source == "direct"


def test_get_matched_normal_cell_type_proxy():
    cell_types, confidence, source = get_matched_normal_cell_type("urinary_bladder")
    assert "Basal squamous epithelial cells" in cell_types
    assert 0.3 <= confidence < 1.0
    assert source == "proxy"


def test_get_matched_normal_cell_type_unknown_tissue():
    """Tissues with neither direct nor proxy map return empty +
    source='bulk_fallback' — callers must fall through to the bulk
    nTPM column."""
    cell_types, confidence, source = get_matched_normal_cell_type("xyz_unknown_tissue")
    assert cell_types == []
    assert confidence == 0.0
    assert source == "bulk_fallback"


def test_direct_and_proxy_are_disjoint():
    """A tissue should be in one map or the other, not both —
    otherwise the ``get_matched_normal_cell_type`` direct-before-proxy
    precedence hides the proxy."""
    overlap = set(MATCHED_NORMAL_CELL_TYPE) & set(MATCHED_NORMAL_CELL_TYPE_PROXY)
    assert not overlap, f"tissues in both maps: {overlap}"


def test_all_leaf_tissues_either_covered_or_documented():
    """Every leaf-cancer ``primary_tissue`` should either (a) have
    a direct cell-type match, (b) have a proxy entry, or (c) be
    explicitly known to fall back to bulk. Catches new registry
    tissues that silently regress to bulk without justification."""
    from pirlygenes.gene_sets_cancer import cancer_type_registry

    reg = cancer_type_registry()
    leaf = reg[reg["parent_code"].fillna("").astype(str).eq("")]
    all_tissues = set(leaf["primary_tissue"].dropna().astype(str))

    covered = set(MATCHED_NORMAL_CELL_TYPE) | set(MATCHED_NORMAL_CELL_TYPE_PROXY)

    # Tissues that deliberately fall back to bulk (matched-normal
    # isn't expected to have a single-cell reference — e.g. heme
    # tissues are handled by dedicated heme templates, colon/rectum
    # left on bulk by choice during v4.50.5 roll-out, etc).
    _ACKNOWLEDGED_BULK_FALLBACK = {
        "peripheral_blood",  # blood tumors: heme template uses bulk lymph
        "lymph_node",
        "spleen",
        "spleen_marrow",
        "bone_marrow",
        "soft_tissue",  # SARC umbrella — dispatched via subtype
        "",
        "nan",
        "NaN",  # registry junk
        # The following are in MATCHED_NORMAL_CELL_TYPE_BLEND=0 v4.50.5
        # list — left on bulk until their synthetic tests are updated:
        "colon",
        "rectum",
        "esophagus",
        "cervix",
        "tongue",
        "skin",
        "small_intestine",
        # Head-neck region not yet mapped — HNSC uses tongue (already
        # acknowledged-bulk), NPC → proxy above, oral/laryngeal
        # composites not yet broken out in registry:
        "head_neck",
        "oral",
        "oropharynx",
        "larynx",
        # Cerebellum and CNS subtypes covered by "cerebellum" /
        # "cerebrum" in the direct map; other CNS subtissues fall
        # back transparently.
        # Kidney composite for pediatric:
        "kidney_cns_soft",
        # NCI-coverage expansion (pirlygenes 5.18) — new skin / GU / GI /
        # CNS leaf tissues on bulk fallback until cell-type decomposition
        # panels are curated (#46 follow-on).
        "epidermis",
        "vulva",
        "vagina",
        "penis",
        "urethra",
        "anal_canal",
        "ependyma",
        "sellar_suprasellar",
        "pons_midline",
        "pituitary",
    }
    uncovered = all_tissues - covered - _ACKNOWLEDGED_BULK_FALLBACK
    assert not uncovered, (
        f"registry tissues with no cell-type mapping and not in the "
        f"acknowledged-bulk-fallback allowlist: {sorted(uncovered)}"
    )
