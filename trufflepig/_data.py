"""Where the gene-set / reference CSVs live.

After the analysis migration, all curated data still ships from
``pirlygenes`` — the package was deliberately reduced to a data-only
shape. trufflepig modules that need to read a CSV directly (rather
than through :func:`pirlygenes.load_dataset.get_data`) should import
:data:`DATA_DIR` from here.
"""

from __future__ import annotations

from pathlib import Path

import pirlygenes

DATA_DIR: Path = Path(pirlygenes.__file__).resolve().parent / "data"
