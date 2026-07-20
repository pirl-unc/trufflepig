"""Regression: ``--no-figures`` must not leak no-op plot bindings
across analyze() invocations.

The PR-41 implementation rebinds ~150 plot_* functions in main.py
globals + several plot module attributes to no-ops for the duration
of a no-figures run. Without explicit restoration, a second call to
analyze() in the same process inherits the no-op state — silently
skipping figure generation when the caller wanted figures.

This test pins the contract that ``_NOOP_PLOT_RESTORE_QUEUE`` drains
on exit and the original callables are reattached.
"""

from __future__ import annotations

import trufflepig.main as main_mod


def test_noop_plot_restore_queue_drains_after_analyze_exit():
    """After ``_restore_noop_plot_patches`` runs, the queue is empty
    and any callable that was a no-op is back to its original."""
    queue = main_mod._NOOP_PLOT_RESTORE_QUEUE
    # Pre-condition: queue is empty (nothing patched yet).
    queue.clear()

    # Simulate the patching that happens inside _analyze_body when
    # ``--no-figures`` is set: pick a real plot_* function from
    # main.py's globals and save it.
    original = main_mod.plot_sample_summary

    def sentinel(*_args, **_kwargs):
        return None

    queue.append((main_mod.__dict__, "plot_sample_summary", original))
    main_mod.__dict__["plot_sample_summary"] = sentinel
    assert main_mod.plot_sample_summary is sentinel

    # Run the restoration the same way analyze()'s finally does.
    main_mod._restore_noop_plot_patches()

    # Post-conditions: queue drained, original rebinding restored.
    assert queue == []
    assert main_mod.plot_sample_summary is original


def test_restore_is_idempotent():
    """A second restore call on an empty queue is a no-op."""
    queue = main_mod._NOOP_PLOT_RESTORE_QUEUE
    queue.clear()
    main_mod._restore_noop_plot_patches()  # no-op
    main_mod._restore_noop_plot_patches()  # no-op
    assert queue == []


def test_restore_tolerates_failed_setattr():
    """If a target was unloaded between patch and restore, the rest
    of the queue still drains. Pins the broad-except in
    ``_restore_noop_plot_patches``."""
    queue = main_mod._NOOP_PLOT_RESTORE_QUEUE
    queue.clear()

    class _NoSetAttr:
        __slots__ = ()  # makes setattr raise AttributeError

    bad_target = _NoSetAttr()
    good_target = {}
    good_target["live"] = "patched"

    queue.append((bad_target, "anything", "original_value"))
    queue.append((good_target, "live", "original"))

    main_mod._restore_noop_plot_patches()

    # Both entries removed despite the first failing.
    assert queue == []
    # The good restoration still applied.
    assert good_target["live"] == "original"
