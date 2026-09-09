# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serialize the authored report, finalized headline and emitted figure manifest.

The report model is built from evidence decisions before rendering. Markdown and
PDF present its sections; JSON retains identities, eligibility and provenance.
No report content is recovered by parsing generated prose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .report_view import ReportView

SCHEMA_VERSION = 2

# Reader-facing figure manifest: (filename-suffix, title, interpretation sentence).
# The interpretation is what the figure *means* for the decision — it replaces
# captioning a figure with its PNG filename. Figures are gated on existence at
# emit time (a belief that never fired never wrote its plot), so a manifest entry's
# ``present`` flag is belief-gated. Ordering mirrors the patient PDF: compact QC,
# final-call composition/purity, selected biological analyses, then recommendations.
#
# Preliminary ranker leaders, alternative decomposition candidates, raw reference
# maps, and redundant technical plots are intentionally absent. Individual PNGs
# and the evidence tables retain them for technical review.
FIGURE_REGISTRY = [
    (
        "sample-context.png",
        "Sample quality context",
        "Compact library, preservation, and expression-concentration checks used "
        "to judge whether the sample is suitable for interpretation.",
    ),
    (
        "degradation-index.png",
        "RNA degradation check",
        "Long-to-short transcript ratios show the sample-specific degradation signal "
        "that informs uncertainty in downstream estimates.",
    ),
    (
        "decomposition-composition.png",
        "Estimated RNA composition",
        "Conditional RNA-mixture model weights for the estimated tumor "
        "contribution and fitted external normal, immune, and stromal references. "
        "These are not per-gene subtraction percentages.",
    ),
    (
        "decomposition-components.png",
        "Tumor microenvironment components",
        "The selected final-call model partitions its non-tumor RNA weight among "
        "external stromal and immune references; each gene can have a different "
        "source attribution.",
    ),
    (
        "purity-methods.png",
        "Purity method agreement",
        "Several RNA-derived tumor-fraction estimates and their agreement. Some methods "
        "share inputs, so this is a consistency check rather than an independent "
        "measurement; the reported interval widens when they disagree.",
    ),
    (
        "therapy-pathway-state.png",
        "Therapy pathway state",
        "Expression state of therapy-relevant pathways provides biological context "
        "for the candidate recommendations. These cohort-relative RNA panels do not "
        "establish prior treatment, drug sensitivity, or resistance.",
    ),
    (
        "subtype-signature.png",
        "Subtype RNA context",
        "Cohort-relative expression of the displayed subtype-panel genes provides "
        "biological context. It does not establish a clinical subtype, treatment "
        "history or drug eligibility, and does not change the finalized report call.",
    ),
]


# --------------------------------------------------------------------------- #
# Artifact discovery.
# --------------------------------------------------------------------------- #
def find_prefix(analyze_dir: Path) -> str:
    summaries = sorted(
        path
        for path in analyze_dir.glob("*-summary.md")
        if not path.name.endswith("-cancer-type-signal-summary.md")
    )
    if summaries:
        latest = max(summaries, key=lambda path: path.stat().st_mtime)
        return latest.name.removesuffix("-summary.md")
    analyses = sorted(analyze_dir.glob("*-analysis.md"))
    if analyses:
        latest = max(analyses, key=lambda path: path.stat().st_mtime)
        return latest.name.removesuffix("-analysis.md")
    raise FileNotFoundError(f"No *-summary.md or *-analysis.md found in {analyze_dir}")


def find_figure(analyze_dir: Path, prefix: str, suffix: str) -> Optional[Path]:
    name = f"{prefix}-{suffix}"
    candidates = [
        analyze_dir / name,
        analyze_dir / "figures" / name,
    ]
    candidates.extend(analyze_dir.glob(f"**/{name}"))
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


# --------------------------------------------------------------------------- #
# Document assembly.
# --------------------------------------------------------------------------- #
def build_figure_manifest(
    analyze_dir: Path,
    prefix: str,
    *,
    purity_status: str = "resolved",
    purity_unresolved_reason: Optional[str] = None,
) -> List[dict]:
    """The belief-gated reader-figure manifest: every registry figure, each with a
    ``present`` flag (True iff the pipeline actually emitted the plot — which it
    only does when the underlying belief passed threshold) and its resolved path."""
    manifest: List[dict] = []
    unresolved_captions = {
        "decomposition-composition.png": (
            "Selected operational tumor/background model used for attribution; "
            "not a resolved sample-composition measurement."
        ),
        "decomposition-components.png": (
            "Components from the selected operational model; not a resolved "
            "sample-composition measurement."
        ),
        "purity-methods.png": (
            "Independent purity estimators support incompatible scenarios; the "
            "operational value is not a fused consensus estimate."
        ),
    }
    if purity_unresolved_reason == "same_lineage_not_identifiable":
        unresolved_captions = {
            "decomposition-composition.png": (
                "Selected operating model for target attribution. Tumor and benign "
                "same-lineage cells cannot be cleanly separated from this bulk RNA."
            ),
            "decomposition-components.png": (
                "External reference weights from the selected operating model; not "
                "a resolved malignant-versus-benign cell composition."
            ),
            "purity-methods.png": (
                "RNA tumor-fraction methods are structurally limited because tumor "
                "and benign same-lineage cells share the modeled programs."
            ),
        }
    for suffix, title, caption in FIGURE_REGISTRY:
        if purity_status == "discordant_estimators":
            caption = unresolved_captions.get(suffix, caption)
        figure = find_figure(analyze_dir, prefix, suffix)
        present = figure is not None
        manifest.append(
            {
                "suffix": suffix,
                "title": title,
                "caption": caption,
                "present": present,
                "path": figure.name if present else None,
            }
        )
    return manifest


def build_report_document(
    analyze_dir: Path,
    prefix: Optional[str] = None,
    *,
    report_view: "ReportView",
    content,
) -> dict:
    """Assemble the structured report document for one analyze directory.

    ``report_view`` is the authoritative headline. ``content`` is the authored
    report model shared with Markdown; the builder never parses generated prose.
    """
    analyze_dir = Path(analyze_dir)
    if prefix is None:
        prefix = find_prefix(analyze_dir)
    headline = report_view.public_dict()
    document = {
        "schema_version": SCHEMA_VERSION,
        "prefix": prefix,
        "sample_id": report_view.sample_id,
        "headline": headline,
        **content.public_dict(),
        "figures": build_figure_manifest(
            analyze_dir,
            prefix,
            purity_status=report_view.purity.status,
            purity_unresolved_reason=report_view.purity.unresolved_reason,
        ),
    }
    detail = next(section for section in document["sections"] if section["id"] == "evidence")
    for figure in document["figures"]:
        if figure["present"]:
            detail["blocks"].append({"kind": "figure", **figure})
    return document


def write_report_document(
    analyze_dir: Path,
    prefix: str,
    *,
    report_view: "ReportView",
    content,
) -> Path:
    """Build and write ``<prefix>-report.json`` into *analyze_dir*; return its path."""
    analyze_dir = Path(analyze_dir)
    document = build_report_document(
        analyze_dir,
        prefix,
        report_view=report_view,
        content=content,
    )
    path = analyze_dir / f"{prefix}-report.json"
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False))
    return path


def load_report_document(analyze_dir: Path, prefix: Optional[str] = None) -> dict:
    """Load the structured ``<prefix>-report.json`` written by the pipeline."""
    analyze_dir = Path(analyze_dir)
    if prefix is None:
        prefix = find_prefix(analyze_dir)
    path = analyze_dir / f"{prefix}-report.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No structured report document found at {path}; rerun analysis to create it"
        )
    return json.loads(path.read_text(errors="replace"))
