"""trufflepig — RNA tumor analysis built on the pirlygenes data package.

Public entry points:

* :func:`trufflepig.main.analyze` — single-sample analyze (was
  ``pirlygenes analyze`` before the migration).
* :func:`trufflepig.main.compare_analyze` — multi-sample longitudinal
  comparison (was ``pirlygenes compare-analyze``).
* :mod:`trufflepig.cli` — argparse entry point exposed as the
  ``trufflepig`` console script.

The stage DAG in :mod:`trufflepig.pipeline` is the seam for the future
per-stage extraction (trufflepig#2..#14) so a web UI can run and stream
single stages independently.
"""

from .version import __version__

__all__ = ["__version__"]
