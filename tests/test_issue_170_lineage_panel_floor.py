"""Regression test for issue #170 — lineage panels must have ≥ 5 genes.

A small panel is fragile. Removing a single noisy gene drops a 4-gene
panel below the 2-gene fallback floor of the specificity filter; a
3-gene panel produces no upper-half median so the estimator can't
anchor a stable purity. The #170 audit expanded every panel with at
least 5 curated, home-cohort-expressed markers.

This test pins the floor so a future edit to ``lineage-genes.csv``
can't silently drop us back to fragile panels.
"""

import pytest

from trufflepig.tumor_purity import LINEAGE_GENES


LINEAGE_PANEL_MIN_GENES = 5


@pytest.mark.parametrize("code", sorted(LINEAGE_GENES.keys()))
def test_lineage_panel_has_at_least_five_genes(code):
    panel = LINEAGE_GENES[code]
    assert len(panel) >= LINEAGE_PANEL_MIN_GENES, (
        f"{code} lineage panel has only {len(panel)} genes "
        f"({panel}) — expand via pirlygenes/data/lineage-genes.csv "
        "or tag this cohort as mixture-cohort per #171"
    )


def test_lineage_panel_sizes_distribution():
    """Smoke-check: we expect most panels between 5 and 10 genes.
    Very large panels (>15) are suspicious — panels should be
    curated, not dumped from HPA."""
    sizes = [len(v) for v in LINEAGE_GENES.values()]
    assert min(sizes) >= LINEAGE_PANEL_MIN_GENES
    assert max(sizes) <= 20, (
        "Some lineage panel is >20 genes — is this curated or a "
        "raw HPA dump? Curation beats volume."
    )
