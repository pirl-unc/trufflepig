"""Plotting context object — bundle figure params + on/off switch.

Replaces six positional kwargs (``output_image_prefix``, ``output_dpi``,
``plot_height``, ``plot_aspect``, output_dir, deprecated_figures) that
were threaded through every plot callsite, and gives a single early-out
for ``--no-figures`` markdown-only runs.

Plot functions accept ``ctx: PlottingContext`` and exit early when
``ctx.enabled`` is false, so neither matplotlib figure construction nor
disk writes happen. The orchestrator builds the context once in
:func:`_analyze_body` and passes it everywhere.

This module is intentionally dependency-light — no matplotlib imports,
no pandas — so the disabled path stays cheap even on cold starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class PlottingContext:
    """Bundled figure parameters with an enabled switch.

    Fields:

    - ``enabled``: when False, every plot function early-outs without
      constructing a figure or writing to disk. ``figure_path()`` and
      ``save()`` return ``None``.
    - ``output_dir``: directory figures are written to (typically
      ``<workspace>/analyze``).
    - ``prefix``: filename prefix derived from the sample id (e.g.
      ``PFO017-bladder-2023``). May be ``None`` for un-prefixed runs.
    - ``dpi`` / ``height`` / ``aspect``: matplotlib sizing params.
    - ``deprecated_figures``: opt-in for the legacy extra figures
      directory; kept on the context so plotting modules don't import
      ``AnalyzeConfig``.

    Use :meth:`disabled` for tests or markdown-only runs.
    """

    enabled: bool = True
    output_dir: Path = field(default_factory=lambda: Path("."))
    prefix: Optional[str] = None
    dpi: int = 300
    height: float = 14.0
    aspect: float = 1.4
    deprecated_figures: bool = False

    def figure_path(self, suffix: str) -> Optional[Path]:
        """Path a figure with this suffix would be written to, or None when disabled."""
        if not self.enabled:
            return None
        stem = f"{self.prefix}-{suffix}" if self.prefix else suffix
        return self.output_dir / stem

    def save(
        self,
        fig: "Figure",
        suffix: str,
        *,
        log: Optional[str] = None,
        **savefig_kwargs: Any,
    ) -> Optional[Path]:
        """Save the figure to ``figure_path(suffix)``; no-op when disabled.

        Defaults ``dpi`` from the context but lets callers override.
        """
        path = self.figure_path(suffix)
        if path is None:
            return None
        savefig_kwargs.setdefault("dpi", self.dpi)
        fig.savefig(path, **savefig_kwargs)
        if log:
            print(f"[plot] {log} -> {path}")
        return path

    def with_prefix(self, prefix: Optional[str]) -> "PlottingContext":
        """Return a copy with a different filename prefix."""
        return replace(self, prefix=prefix)

    @classmethod
    def disabled(cls) -> "PlottingContext":
        """Build a no-op context for tests or ``--no-figures`` runs."""
        return cls(enabled=False)


__all__ = ["PlottingContext"]
