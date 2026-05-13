# Licensed under the Apache License, Version 2.0

"""Sample-provenance page — one synthesized 'what is this sample' doc (#106).

The existing reports (summary, analysis, targets, brief, actionable)
each surface different parts of the 5-step attribution chain:

    library prep -> preservation -> coarse TME -> fine subtypes -> tumor-inferred expression

Readers have to reassemble the chain mentally. The provenance page is
the assembled view in one ~30-line document plus a simple stacked-bar
figure, cross-linked from the other reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .reporting import (
    partition_tumor_core_rows,
    summarize_reliability_reasons,
)
from .sample_context import (
    heuristic_support_label,
    length_pair_display_label,
    library_prep_display_label,
)


def _compartment_label(comp: str) -> str:
    return comp.replace("matched_normal_", "matched-normal ").replace("_", " ")


def _display_sample_id(sample_id: Optional[str]) -> Optional[str]:
    if sample_id is None:
        return None
    text = str(sample_id).strip()
    if not text:
        return None
    if "/" in text or "\\" in text:
        text = Path(text).name.strip()
    return text or None


def build_provenance_md(
    analysis,
    ranges_df,
    decomp_results,
    cancer_code: str,
    sample_id: Optional[str] = None,
) -> str:
    """Render ``*-provenance.md`` — the 5-step attribution chain.

    Each step states its input and what it deducts from the naive
    "all signal is tumor" interpretation so the reader can follow the
    chain from raw TPMs to conservative tumor-inferred expression.
    """
    sample_context = analysis.get("sample_context")
    purity = analysis.get("purity") or {}
    lines: List[str] = []
    sample_id = _display_sample_id(sample_id)
    header_id = f": {sample_id}" if sample_id else ""
    lines.append(f"# Sample provenance{header_id}\n")
    lines.append(
        "<!-- What is in this sample, step by step. Each step "
        "explains what it deducts before the next one. -->"
    )
    lines.append("")

    # Step 0b — tissue composition screen (#149). Runs before the
    # lineage-aware steps so the reader sees "what kind of tissue is
    # this, and is there any hint of cancer" in the first breath.
    hvt = analysis.get("healthy_vs_tumor")
    if hvt is not None and hvt.top_normal_tissues:
        lines.append("## 0. Tissue composition screen\n")
        lines.append(hvt.summary_line())
        if hvt.cancer_hint != "tumor-consistent":
            lines.append(
                "\nThis Step-0 signal carries forward: the downstream "
                "cancer call is treated more cautiously, and the per-gene "
                "expression ranges are widened to reflect the ambiguity."
            )
        lines.append("")

    # Step 1 — library prep
    lines.append("## 1. Library prep\n")
    if sample_context is not None:
        prep = getattr(sample_context, "library_prep", "unknown")
        confidence = float(
            getattr(sample_context, "library_prep_confidence", 0.0) or 0.0
        )
        prep_label = library_prep_display_label(prep)
        lines.append(
            f"Inferred: **{prep_label}** ({heuristic_support_label(confidence)}). "
        )
        if prep == "exome_capture":
            lines.append(
                "Implication: rRNA and many non-polyadenylated transcripts "
                "are under-sampled by the capture design. Low MT fraction is "
                "expected, but measured MT protein-coding transcripts can "
                "still be real low-level signal."
            )
        elif prep == "poly_a":
            lines.append(
                "Implication: rRNA and non-polyadenylated transcripts are "
                "absent by design; mitochondrial is depressed but present."
            )
        elif prep == "ribo_depleted":
            lines.append(
                "Implication: ribosomal RNAs are reduced but a small residual "
                "remains. Mitochondrial transcripts are retained and can rise "
                "with degradation."
            )
        elif prep == "total_rna":
            lines.append(
                "Implication: rRNA dominates the library unless normalized. "
                "Treat raw TPM signals with caution until rRNA-corrected."
            )
        else:
            lines.append(
                "Implication: prep could not be inferred with high confidence; "
                "the downstream expectation bands fall back to tolerant defaults."
            )
    else:
        lines.append("*Library prep could not be inferred from this input.*")
    lines.append("")

    # Step 2 — preservation / degradation
    lines.append("## 2. Preservation\n")
    if sample_context is not None:
        pres = getattr(sample_context, "preservation", "unknown").replace("_", " ")
        sev = getattr(sample_context, "degradation_severity", "none")
        lines.append(
            f"Inferred: **{pres}**, degradation severity **{sev}**; "
            f"length-pair check **{length_pair_display_label(sample_context)}**."
        )
        if sev in ("moderate", "severe"):
            lines.append(
                "\nImplication: long-transcript quantification is biased "
                "downward — the purity range has been widened and long-gene "
                "therapy targets are de-emphasized in the ranking."
            )
        elif pres == "ffpe":
            lines.append(
                "\nImplication: FFPE preservation is accounted for but no "
                "heavy degradation was detected."
            )
    lines.append("")

    # Step 3 — coarse composition
    lines.append("## 3. Coarse composition\n")
    best = decomp_results[0] if decomp_results else None
    if best is not None:
        tumor_frac = float(getattr(best, "purity", 0.0) or 0.0)
        fractions = dict(getattr(best, "fractions", {}) or {})
        non_tumor = sorted(
            ((c, f) for c, f in fractions.items() if c != "tumor" and f > 0),
            key=lambda kv: -kv[1],
        )
        top = non_tumor[:5]
        parts = [f"**tumor {tumor_frac:.0%}**"]
        for comp, frac in top:
            if frac >= 0.005:
                parts.append(f"{_compartment_label(comp)} {frac:.0%}")
        rest_frac = sum(f for _, f in non_tumor if f < 0.005)
        if rest_frac > 0:
            parts.append(f"other {rest_frac:.0%}")
        lines.append("Fitted fractions: " + ", ".join(parts) + ".")
        lines.append(
            "\nEach non-tumor component is subtracted from the observed "
            "TPM per gene (#108). A target whose signal is mostly assigned "
            "to a non-tumor compartment is flagged TME-dominant in the "
            "target tables."
        )
    else:
        lines.append("*No decomposition result available for this sample.*")
    lines.append("")

    # Step 4 — activated subtypes
    lines.append("## 4. Subtype refinements\n")
    if best is not None:
        trace = getattr(best, "component_trace", None)
        subtype_notes = []
        if trace is not None and not trace.empty:
            for _, row in trace.iterrows():
                comp = str(row.get("component", ""))
                frac = float(row.get("fraction") or 0.0)
                if frac < 0.01:
                    continue
                if comp.startswith("matched_normal_"):
                    subtype_notes.append(
                        f"Matched-normal {comp.replace('matched_normal_', '')} "
                        f"compartment present at {frac:.0%} — subtracted "
                        "before target-expression ranking."
                    )
                elif any(k in comp.lower() for k in ("caf", "tam", "mdsc", "treg")):
                    subtype_notes.append(
                        f"Activated subtype **{comp}** contributing {frac:.0%}."
                    )
        if subtype_notes:
            for n in subtype_notes:
                lines.append(f"- {n}")
        else:
            lines.append(
                "No activated-subtype refinements flagged above the 1% "
                "threshold. Tumor-subtracted TPMs use the coarse compartment "
                "fit only."
            )
    lines.append("")

    # Step 5 — tumor-linked expression
    lines.append("## 5. Tumor-linked expression\n")
    if ranges_df is not None and len(ranges_df):
        if "attribution" in ranges_df.columns:
            supported_core, provisional_core, _ = partition_tumor_core_rows(
                ranges_df,
                min_tumor_tpm=1.0,
            )
            n_core = int(len(supported_core))
            lines.append(
                f"After subtracting the fitted non-tumor compartments, "
                f"**{n_core} genes** retain ≥1 TPM of tumor-supported "
                "tumor-attributed expression."
            )
            if len(provisional_core):
                reason_summary = summarize_reliability_reasons(provisional_core)
                lines.append(
                    f"\nAn additional **{len(provisional_core)} genes** retain residual "
                    "tumor-attributed TPM but remain mixed-source in the markdown layer"
                    + (f" ({reason_summary})." if reason_summary else ".")
                )
            # Top 5 supported tumor-linked genes.
            top = supported_core.sort_values("attr_tumor_tpm", ascending=False).head(5)
            if len(top):
                names = ", ".join(
                    f"{str(r['symbol'])} ({float(r['attr_tumor_tpm']):.0f})"
                    for _, r in top.iterrows()
                )
                lines.append(f"\nTop tumor-linked genes (symbol, tumor TPM): {names}.")
            elif len(provisional_core):
                lines.append(
                    "\nNo gene cleared the current tumor-supported filter; "
                    "use the mixed-source evidence tables and the TSV for manual review."
                )
    else:
        lines.append("*No target-expression ranges available.*")
    lines.append("")

    # Chain summary + cross-links
    overall = purity.get("overall_estimate")
    if overall is not None:
        lines.append(
            f"**Chain summary:** observed expression → library-prep-aware "
            f"artifact expectations → preservation-adjusted quantification → "
            f"decomposition subtracts {1 - float(overall):.0%} as non-tumor "
            "compartments → residual is the tumor-linked signal used for "
            "therapy-target ranking."
        )
    lines.append("")
    lines.append("*See also: `*-summary.md`, `*-analysis.md`, and `*-evidence.md`.*")
    return "\n".join(lines)


def plot_provenance_funnel(
    analysis,
    ranges_df,
    decomp_results,
    save_to_filename: str,
    save_dpi: int = 150,
):
    """Render ``*-provenance.png`` — horizontal stacked bar showing the
    compartment fractions with tumor-linked signal on the right and non-tumor
    compartments to its left, one simple figure per sample.

    Returns the filename on success, ``None`` when the inputs don't
    support a meaningful plot (e.g. no decomposition).
    """
    import matplotlib.pyplot as plt

    best = decomp_results[0] if decomp_results else None
    if best is None:
        return None
    fractions = dict(getattr(best, "fractions", {}) or {})
    if not fractions:
        return None

    tumor_frac = float(fractions.pop("tumor", 0.0))
    non_tumor = sorted(
        ((c, f) for c, f in fractions.items() if f > 0.005),
        key=lambda kv: -kv[1],
    )
    rest_frac = sum(f for c, f in fractions.items() if f <= 0.005 and f > 0)

    labels = ["tumor-linked"] + [_compartment_label(c) for c, _ in non_tumor]
    values = [tumor_frac] + [f for _, f in non_tumor]
    if rest_frac > 0:
        labels.append("other")
        values.append(rest_frac)
    values = np.array(values, dtype=float)

    colors = ["#e74c3c"] + [
        plt.cm.tab20.colors[i % len(plt.cm.tab20.colors)]
        for i in range(len(values) - 1)
    ]

    fig, ax = plt.subplots(figsize=(10, 2.6))
    left = 0.0
    for label, val, color in zip(labels, values, colors):
        ax.barh([0], [val], left=[left], color=color, edgecolor="white", label=label)
        if val > 0.03:
            ax.text(
                left + val / 2,
                0,
                f"{label}\n{val:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        left += val

    ax.set_xlim(0, max(1.0, left))
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_title("Sample composition", fontsize=11, fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=min(4, len(labels)),
        fontsize=8,
        frameon=False,
    )

    plt.tight_layout()
    fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    return save_to_filename
