"""Regression test for issue #170 — purity lineage panels must have ≥ 5 genes.

A small panel is fragile. Removing a single noisy gene drops a 4-gene
panel below the 2-gene fallback floor of the specificity filter; a
3-gene panel produces no upper-half median so the estimator can't
anchor a stable purity. The #170 audit expanded every panel with at
least 5 curated, home-cohort-expressed markers.

Scope: the floor applies to codes whose **own** lineage panel anchors
their purity estimate (``lineage_purity_panel_codes()``). Fusion-defined
sarcoma subtypes (SARC_ASPS=[TFE3], SARC_SFT=[STAT6], ...) fall back to
the SARC parent for purity, so their single pathognomonic *diagnostic*
marker is never used as a purity panel — it drives subtype identification
via the ontology/recall path instead. Holding a diagnostic marker to the
purity floor would be a category error; the guard below instead pins that
every floor-exempt code is a genuine fallback subtype (so a top-level
purity target can never silently drop below the floor and slip through).
"""

import pytest

from pirlygenes.gene_sets_cancer import cancer_type_registry

from trufflepig.tumor_purity import (
    LINEAGE_GENES,
    _broad_purity_fallback_code,
    lineage_purity_panel_codes,
)


LINEAGE_PANEL_MIN_GENES = 5


@pytest.mark.parametrize("code", sorted(lineage_purity_panel_codes()))
def test_purity_panel_has_at_least_five_genes(code):
    panel = LINEAGE_GENES[code]
    assert len(panel) >= LINEAGE_PANEL_MIN_GENES, (
        f"{code} purity lineage panel has only {len(panel)} genes "
        f"({panel}) — expand via pirlygenes/data/lineage-genes.csv "
        "or tag this cohort as mixture-cohort per #171"
    )


def test_floor_exempt_codes_are_fallback_subtypes():
    """Every code exempt from the purity-panel floor must be a genuine
    subtype that falls back to a parent cohort for purity — i.e. its own
    panel is never consulted as a purity anchor. This guards against a
    top-level purity target silently dropping below the floor and being
    waved through as "diagnostic-only"."""
    reg = cancer_type_registry().set_index("code")
    panels = lineage_purity_panel_codes()
    exempt = [c for c in LINEAGE_GENES if c not in panels]
    assert exempt, "expected at least the subtype-marker batch to be exempt"
    for code in exempt:
        assert LINEAGE_GENES[code], f"{code}: empty lineage entry"
        # (a) actually falls back for purity (own panel unused) ...
        assert _broad_purity_fallback_code(code), (
            f"{code} is floor-exempt but does NOT fall back for purity — it "
            "would use its own sub-floor panel as a purity anchor"
        )
        # ... and (b) is a real subtype, not a top-level cohort.
        parent = str(reg.loc[code, "parent_code"]) if code in reg.index else ""
        assert parent and parent.lower() != "nan", (
            f"{code} is floor-exempt but is not a subtype (parent_code="
            f"{parent!r}) — a top-level purity target must meet the floor"
        )


def test_lineage_panel_sizes_distribution():
    """Smoke-check: purity panels should be 5–20 genes. Very large panels
    (>20) are suspicious — panels should be curated, not dumped from HPA."""
    sizes = [len(LINEAGE_GENES[c]) for c in lineage_purity_panel_codes()]
    assert sizes, "expected at least one purity panel"
    assert min(sizes) >= LINEAGE_PANEL_MIN_GENES
    assert max(sizes) <= 20, (
        "Some purity lineage panel is >20 genes — is this curated or a "
        "raw HPA dump? Curation beats volume."
    )
