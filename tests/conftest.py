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


@pytest.fixture(autouse=True)
def _isolate_symbol_map_cache():
    yield
    try:
        from trufflepig.common import ensembl_id_to_symbol_map

        ensembl_id_to_symbol_map.cache_clear()
    except Exception:
        # Never let cache teardown mask a real test result.
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
