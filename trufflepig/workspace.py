"""Workspace root and metadata for the production run/compare commands.

Analysis reports, evidence tables, and figures live under ``analyze/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class Workspace:
    root: Path

    @classmethod
    def open(cls, path) -> "Workspace":
        root = Path(path).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return cls(root=root)

    def write_meta(self, meta: Dict[str, Any]) -> None:
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2))
