"""Shared pytest fixtures + cross-test cache isolation.

``ensembl_id_to_symbol_map()`` is a process-global ``lru_cache`` built from the
pan-cancer reference. A test that monkeypatches ``pan_cancer_expression`` to a
small fake (e.g. ``test_build_sample_tpm_by_symbol_does_not_deepcopy_attrs``)
populates this cache with a tiny map; without isolation that map leaks into
later tests sharing the same xdist worker process and degrades their
symbol-mapping — the long-standing "passes alone, fails in suite" flake noted
in CLAUDE.md.

Clearing this cache after every test is cheap: it only drops a dict; the
expensive pan-cancer *matrix* stays cached in ``trufflepig.reference``, so the
map is rebuilt lazily in ~milliseconds (a single ``dict(zip(...))``) only for
tests that actually need it. This removes the worker-affinity flake without the
memory/CPU cost of re-loading reference matrices.
"""

import pytest


@pytest.fixture
def clinical_hla_context():
    """Explicit synthetic clinical evidence for tests of HLA compatibility gates."""
    from trufflepig.clinical_context import ClinicalAssay, ClinicalContext, ClinicalSource

    def context(alleles):
        assays = () if not alleles else (ClinicalAssay(
            kind="hla", result="typed", alleles=tuple(alleles), complete_loci=("A",),
            method="NGS", specimen_id="synthetic-specimen", scope="current",
            validity="validated", reportability="reportable",
            source=ClinicalSource(title="Synthetic clinical HLA report"),
        ),)
        return ClinicalContext(specimen_id="synthetic-specimen", assays=assays).public_dict()

    return context


@pytest.fixture
def clinical_magea4_assay():
    """Explicit positive companion result when isolating other afami gates."""
    from trufflepig.clinical_context import ClinicalAssay, ClinicalSource

    def assay(specimen_id="synthetic-specimen"):
        return ClinicalAssay(
            kind="ihc", analyte="MAGEA4", result="positive", method="IHC",
            test_id="FDA:P230016", specimen_type="tissue", specimen_id=specimen_id,
            scope="current", validity="validated", reportability="reportable",
            source=ClinicalSource(title="Synthetic MAGE-A4 companion report"),
        )

    return assay


@pytest.fixture(autouse=True)
def _isolate_reference_discovery_caches(request):
    yield
    try:
        from trufflepig.common import ensembl_id_to_symbol_map

        ensembl_id_to_symbol_map.cache_clear()
    except Exception:
        # Never let cache teardown mask a real test result.
        pass
    if "monkeypatch" in getattr(request, "fixturenames", ()):
        try:
            from trufflepig.analyze import cancer_type_context

            cancer_type_context._direct_expression_reference_records.cache_clear()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close any matplotlib figures a test left open.

    The plotting helpers (e.g. ``plot_priority_targets``) ``return fig`` for the
    caller to own; the main pipeline closes per-sample, but tests that just
    assert-and-discard don't, so pyplot's global registry accumulates figures
    across tests in a worker (the ``More than 20 figures have been opened``
    RuntimeWarning + a slow memory creep). Only act if matplotlib is already
    imported, so non-plotting tests don't pay the import.
    """
    yield
    import sys

    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        try:
            plt.close("all")
        except Exception:
            pass
