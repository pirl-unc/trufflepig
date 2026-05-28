"""Where bundled reference CSVs live."""

from __future__ import annotations

from pathlib import Path

import pirlygenes

PIRLYGENES_DATA_DIR: Path = Path(pirlygenes.__file__).resolve().parent / "data"
TRUFFLEPIG_DATA_DIR: Path = Path(__file__).resolve().parent / "data"

# Back-compat for modules that still read pirlygenes-owned curated tables
# directly. Trufflepig-owned generated artifacts should use
# TRUFFLEPIG_DATA_DIR.
DATA_DIR: Path = PIRLYGENES_DATA_DIR
