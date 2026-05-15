"""Cross-package contract: every pirlygenes gene-family must map to a QC class.

If pirlygenes adds a new family panel (a new ``<family>-genes.csv``
under ``pirlygenes/data/``), and the QC classifier (now also in
pirlygenes) doesn't list that family in ``_FAMILY_TO_QC``, the
unknown-family branch silently falls through to
``("protein-coding/other", "other")`` and the panel is excluded from
the default normalization drop set. That silent drift is exactly the
kind of bug that's painful to notice months later when a downstream
report stops mentioning a family.

The classifier moved into ``pirlygenes.expression.qc`` in pirlygenes
5.1; the contract is still load-bearing for trufflepig (the analysis
layer assumes every classified family becomes a QC group), so this
test stays here as the downstream guard.
"""

from __future__ import annotations

import pytest

from pirlygenes.expression.qc import _FAMILY_TO_QC


# Families that pirlygenes ships but trufflepig has deliberately
# chosen NOT to drop by default. Each entry must come with a comment
# explaining why — the friction is the point.
_INTENTIONALLY_UNMAPPED: frozenset[str] = frozenset({
    # (none today — every pirlygenes gene family is currently mapped
    # to a QC group. Add a family name here with a comment if/when
    # trufflepig consciously declines to map it.)
})


def _pirlygenes_family_names() -> list[str]:
    try:
        from pirlygenes.gene_families import gene_family_names
    except ImportError:
        pytest.skip("pirlygenes not installed; cross-package contract not testable")
    return gene_family_names()


def test_every_pirlygenes_family_is_classified_or_explicitly_unmapped():
    """Every gene family pirlygenes exposes must either be in
    ``pirlygenes.expression.qc._FAMILY_TO_QC`` or in this module's
    ``_INTENTIONALLY_UNMAPPED`` set. New panels in pirlygenes must
    not silently fall through to ``"other"``."""
    pg_families = set(_pirlygenes_family_names())
    mapped = set(_FAMILY_TO_QC.keys())
    unmapped_but_intentional = _INTENTIONALLY_UNMAPPED
    drift = pg_families - mapped - unmapped_but_intentional
    assert not drift, (
        "pirlygenes exposes gene-family names that "
        "pirlygenes.expression.qc._FAMILY_TO_QC doesn't classify:\n  "
        f"{sorted(drift)}\n"
        "Either map each one to a QC group in pirlygenes, or list it "
        "in tests/test_family_qc_contract.py::_INTENTIONALLY_UNMAPPED "
        "with a comment explaining why."
    )


def test_family_qc_map_has_no_stale_entries():
    """The reverse direction: _FAMILY_TO_QC should not list family
    names that pirlygenes doesn't expose. Stale entries are a hint
    that pirlygenes renamed or removed a family without the QC layer
    catching up."""
    pg_families = set(_pirlygenes_family_names())
    mapped = set(_FAMILY_TO_QC.keys())
    stale = mapped - pg_families
    assert not stale, (
        "pirlygenes.expression.qc._FAMILY_TO_QC references family "
        "names not exposed by pirlygenes.gene_families.gene_family_names(): "
        f"{sorted(stale)}"
    )


def test_unknown_family_silently_maps_to_other():
    """Defensive — the ``_family_to_qc_class`` fallback for an unknown
    family name returns ``("protein-coding/other", "other")``. This
    test pins the contract so a future change to a stricter raise
    doesn't silently break callers."""
    from pirlygenes.expression.qc import _family_to_qc_class

    qc = _family_to_qc_class("definitely_not_a_real_family")
    assert qc.group == "other"
    assert qc.label == "protein-coding/other"


def test_classify_gene_qc_does_not_silently_swallow_runtime_errors(monkeypatch):
    """Regression: ``classify_gene_qc`` previously had a bare
    ``except Exception`` around the pirlygenes import which would
    swallow ANY error (including a broken pirlygenes data file at
    runtime) and silently degrade ENSG-only inputs to ``"other"``.
    The fix narrowed it to ``except ImportError`` only — runtime
    errors must now propagate."""
    import pirlygenes.gene_families as gf_mod
    from trufflepig.expression_qc import classify_gene_qc

    def _boom(_ensg):
        raise RuntimeError("simulated data-file corruption")

    monkeypatch.setattr(gf_mod, "gene_family_for_ensembl_id", _boom)
    with pytest.raises(RuntimeError, match="simulated data-file corruption"):
        classify_gene_qc(ensembl_id="ENSG00000251562")
