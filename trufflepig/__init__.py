"""trufflepig — step-by-step RNA tumor analysis.

Top-level exports keep the package surface minimal; the real work lives
in the per-stage modules under :mod:`trufflepig.stages`. The CLI composes
them into a pipeline, and each stage reads and writes a serializable
record so a website frontend can stream incremental results.
"""

from .version import __version__

__all__ = ["__version__"]
