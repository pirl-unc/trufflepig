"""Pipeline DAG declaration (stage name → upstream dependencies).

Used by the CLI's ``trufflepig run`` command to invoke stages in order
and by ``trufflepig run --stage <name>`` to resolve the minimum
upstream set that must exist in the workspace before a given stage can
run. A website frontend can consume this dict to render a progress
view.
"""

from __future__ import annotations

from typing import Dict, List

STAGE_ORDER: List[str] = [
    "load_expression",
    "sample_context",
    "analyze",
    "decompose",
    "ranges",
    "confidence",
    "render_targets",
    "render_summary",
    "render_analysis",
    "render_provenance",
    "render_brief",
    "bundle",
]

# Each stage's direct upstream dependencies (which records it reads).
STAGE_DEPS: Dict[str, List[str]] = {
    "load_expression": [],
    "sample_context": ["load_expression"],
    "analyze": ["load_expression"],
    "decompose": ["analyze", "sample_context"],
    "ranges": ["analyze", "decompose"],
    "confidence": ["analyze"],
    "render_targets": ["ranges", "confidence"],
    "render_summary": ["analyze", "confidence", "sample_context"],
    "render_analysis": ["analyze", "decompose", "sample_context", "confidence"],
    "render_provenance": ["sample_context", "decompose", "ranges"],
    "render_brief": ["analyze", "ranges", "confidence"],
    "bundle": [
        "render_targets",
        "render_summary",
        "render_analysis",
        "render_provenance",
        "render_brief",
    ],
}


def required_upstream(stage: str) -> List[str]:
    """Return the minimal set of upstream stage records required for
    ``stage`` to run, as a topologically ordered list.
    """
    if stage not in STAGE_DEPS:
        raise ValueError(f"Unknown stage: {stage}")
    visited = set()
    ordered: List[str] = []

    def _walk(s):
        for d in STAGE_DEPS[s]:
            if d in visited:
                continue
            visited.add(d)
            _walk(d)
            ordered.append(d)

    _walk(stage)
    return ordered
