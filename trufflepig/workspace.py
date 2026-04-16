"""Workspace layout + on-disk record format for the pipeline.

Each stage produces a small, human-readable record that subsequent
stages can read back. The chosen format is **JSON** for dict-valued
records and **TSV** for tabular outputs — both are trivially inspected
with standard tools and cheap to stream from a website backend.

Markdown documents (for clinician consumption) live in the same
workspace but are not "records" — they're rendered from records.

Workspace layout for ``trufflepig run --workspace WS``:

    WS/
      meta.json                  # run metadata (sample id, args, version)
      records/
        sample_context.json
        analysis.json
        decomposition.json
        ranges.tsv
        confidence.json
        provenance.md            # rendered pages live alongside records
        brief.md
        actionable.md
        targets.md
        summary.md
      figures/
        sample_context.png
        provenance.png
        ...

A stage is idempotent: rerunning it should overwrite its records without
touching downstream records that it doesn't invalidate. The ``trufflepig
run`` command orchestrates a dependency DAG; sub-commands let a user (or
the website) invoke a single stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Workspace:
    root: Path

    @classmethod
    def open(cls, path) -> "Workspace":
        root = Path(path).resolve()
        root.mkdir(parents=True, exist_ok=True)
        (root / "records").mkdir(exist_ok=True)
        (root / "figures").mkdir(exist_ok=True)
        return cls(root=root)

    def record_path(self, name: str) -> Path:
        return self.root / "records" / name

    def figure_path(self, name: str) -> Path:
        return self.root / "figures" / name

    def write_meta(self, meta: Dict[str, Any]) -> None:
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2))

    def read_meta(self) -> Dict[str, Any]:
        path = self.root / "meta.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def write_record(self, name: str, payload: Dict[str, Any]) -> Path:
        path = self.record_path(name)
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=_json_default))
        return path

    def read_record(self, name: str) -> Optional[Dict[str, Any]]:
        path = self.record_path(name)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def write_markdown(self, name: str, content: str) -> Path:
        path = self.record_path(name)
        path.parent.mkdir(exist_ok=True)
        path.write_text(content)
        return path


def _json_default(obj):
    """JSON fallback for objects the stages produce (numpy scalars, sets)."""
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, set):
        return sorted(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)
