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

from .report_view import ReportView
from .reporting import (
    component_display_label,
    partition_tumor_core_rows,
    summarize_reliability_reasons,
)
from .sample_context import (
    heuristic_support_label,
    length_pair_display_label,
    library_prep_display_label,
)


def _compartment_label(comp: str) -> str:
    return component_display_label(comp, include_model_role=True)


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
    *,
    report_view: ReportView,
) -> str:
    """Render ``*-provenance.md`` — the 5-step attribution chain.

    Each step states its input and what it deducts from the naive
    "all signal is tumor" interpretation so the reader can follow the
    chain from raw TPMs to conservative tumor-inferred expression.
    """
    sample_context = analysis.get("sample_context")
    lines: List[str] = []
    sample_id = _display_sample_id(sample_id)
    header_id = f": {sample_id}" if sample_id else ""
    lines.append(f"# Sample provenance{header_id}\n")
    lines.append(
        "<!-- What is in this sample, step by step. Each step "
        "explains what it deducts before the next one. -->"
    )
    lines.append("")

    # Tissue composition screen (#149). Runs before the lineage-aware
    # stages so the reader sees "what kind of tissue is this, and is
    # there any hint of cancer" in the first breath.
    hvt = analysis.get("healthy_vs_tumor")
    if hvt is not None and hvt.top_normal_tissues:
        lines.append("## Tissue Composition Screen\n")
        lines.append(hvt.summary_line())
        if hvt.cancer_hint != "tumor-consistent":
            lines.append(
                "\nThis tissue-composition signal carries forward: the downstream "
                "cancer call is treated more cautiously, and the per-gene "
                "expression ranges are widened to reflect the ambiguity."
            )
        lines.append("")

    # RNA prep and preservation.
    lines.append("## RNA Prep and Preservation\n")
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

    lines.append("### Preservation and Degradation\n")
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

    # Coarse composition.
    lines.append("## Estimated Tumor Fraction and Coarse Composition\n")
    best = decomp_results[0] if decomp_results else None
    if best is not None:
        fractions = dict(getattr(best, "fractions", {}) or {})
        conclusion = report_view.purity
        purity_status, purity_scenarios = conclusion.status, conclusion.scenarios
        purity_is_unresolved = purity_status == "discordant_estimators"
        # Display the FINALIZED headline purity as tumor%, not the
        # decomposition's own pre-fusion tumor fraction. best.purity is frozen
        # at fit time from the pre-fusion purity, but _finalize_fused_purity
        # later re-fuses (blending in the residual fraction + anti-saturation),
        # moving the headline. Reading best.purity here made "Fitted fractions:
        # tumor X%" contradict the reported purity (#85.1). Rescale the
        # non-tumor components so they sum to 1 - tumor%, preserving their
        # relative proportions, so the coarse composition equals the headline
        # by construction.
        headline_tumor = conclusion.estimate
        decomp_tumor = float(getattr(best, "purity", 0.0) or 0.0)
        tumor_frac = decomp_tumor if headline_tumor is None else float(headline_tumor)
        tumor_frac = min(max(tumor_frac, 0.0), 1.0)
        non_tumor = sorted(
            ((c, f) for c, f in fractions.items() if c != "tumor" and f > 0),
            key=lambda kv: -kv[1],
        )
        non_tumor_mass = sum(f for _, f in non_tumor)
        if non_tumor_mass > 0 and not purity_is_unresolved:
            rescale = (1.0 - tumor_frac) / non_tumor_mass
            non_tumor = [(c, f * rescale) for c, f in non_tumor]
        top = non_tumor[:5]
        if purity_is_unresolved:
            if conclusion.unresolved_reason == "same_lineage_not_identifiable":
                lines.append(
                    "**Estimated tumor fraction is unresolved.** Tumor and benign "
                    "same-lineage cells share the RNA programs used for subtraction, "
                    "so this page does not assign a resolved malignant-versus-benign "
                    "percentage."
                )
            else:
                lines.append(
                    "**Quantitative purity is unresolved.** Independent estimators "
                    "support incompatible scenarios, so this page does not assign a "
                    "consensus tumor/non-tumor percentage."
                )
            source_labels = {
                "background_residual": "background-residual decomposition",
                "lineage_panel": "healthy-tissue lineage reference model",
                "signature": "upstream expression model",
            }
            scenario_text = []
            for source, estimate, lower, upper in purity_scenarios:
                if estimate is None:
                    continue
                value = f"{estimate:.0%}"
                if lower is not None and upper is not None:
                    value += f" [{lower:.0%}–{upper:.0%}]"
                scenario_text.append(
                    f"{source_labels.get(source, source.replace('_', ' '))}: {value}"
                )
            if scenario_text:
                lines.append("\nEstimator scenarios: " + "; ".join(scenario_text) + ".")
            parts = [f"operational estimated tumor fraction {tumor_frac:.0%}"]
            fraction_prefix = "Selected operational decomposition fractions"
        else:
            parts = [
                f"**estimated tumor fraction {tumor_frac:.0%} (RNA model)**"
            ]
            fraction_prefix = "Estimated fractions from the selected RNA model"
        for comp, frac in top:
            if frac >= 0.005:
                parts.append(f"{_compartment_label(comp)} {frac:.0%}")
        rest_frac = sum(f for _, f in non_tumor if f < 0.005)
        if rest_frac > 0:
            parts.append(f"other {rest_frac:.0%}")
        lines.append(f"\n{fraction_prefix}: " + ", ".join(parts) + ".")
        if purity_is_unresolved:
            lines.append(
                "These fractions belong to the selected operational model and "
                "are used for target attribution only; they are not a resolved "
                "sample-composition measurement."
            )
        lines.append(
            "\nEach non-tumor component is subtracted from the observed "
            "TPM per gene (#108). A target whose signal is mostly assigned "
            "to an external non-tumor reference component is flagged "
            "as mostly background in the target tables. All fractions in this "
            "section are estimates from the RNA model, not direct cell-count measurements."
        )
    else:
        lines.append("*No decomposition result available for this sample.*")
    lines.append("")

    # Activated subtypes.
    lines.append("## Subtype and Background Refinements\n")
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
                        f"{_compartment_label(comp).capitalize()} present at "
                        f"{frac:.0%} — this comes from an external healthy-tissue "
                        "reference panel, not a separate normal sample from this patient, and is "
                        "subtracted before target-expression ranking."
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
                "threshold. Estimated tumor TPMs use the coarse compartment "
                "fit only."
            )
    lines.append("")

    # Tumor-linked expression.
    lines.append("## Estimated Tumor Expression (RNA Model)\n")
    if ranges_df is not None and len(ranges_df):
        if "attribution" in ranges_df.columns:
            supported_core, provisional_core, _ = partition_tumor_core_rows(
                ranges_df,
                min_tumor_tpm=1.0,
            )
            n_core = int(len(supported_core))
            lines.append(
                f"After the RNA model subtracts fitted external-reference background, "
                f"**{n_core} genes** retain ≥1 estimated tumor TPM."
            )
            if len(provisional_core):
                reason_summary = summarize_reliability_reasons(provisional_core)
                lines.append(
                    f"\nAn additional **{len(provisional_core)} genes** retain residual "
                    "estimated tumor TPM but remain mixed source "
                    "in the markdown layer"
                    + (f" ({reason_summary})." if reason_summary else ".")
                )
            # Top 5 supported tumor-linked genes.
            top = supported_core.sort_values("attr_tumor_tpm", ascending=False).head(5)
            if len(top):
                names = ", ".join(
                    f"{str(r['symbol'])} ({float(r['attr_tumor_tpm']):.0f})"
                    for _, r in top.iterrows()
                )
                lines.append(
                    f"\nTop genes by estimated tumor TPM "
                    f"(symbol, estimated TPM): {names}."
                )
            elif len(provisional_core):
                lines.append(
                    "\nNo gene cleared the current tumor support filter; "
                    "use the mixed tumor/background evidence tables and the TSV for manual review."
                )
    else:
        lines.append("*No target-expression ranges available.*")
    lines.append("")

    # Chain summary + cross-links. Read the FINALIZED headline (not the live
    # purity dict) so the "subtracts X% as non-tumor" figure equals the
    # 1 - tumor% of the coarse-composition section above by construction — the
    # two must not disagree within one provenance page (#85.1).
    conclusion = report_view.purity
    overall = conclusion.estimate
    purity_status = conclusion.status
    if purity_status == "discordant_estimators":
        lines.append(
            "**Chain summary:** observed expression → library-prep-aware "
            "artifact expectations → preservation-adjusted quantification → "
            "the selected decomposition supplies an operational background "
            "model for target attribution, while the quantitative tumor/non-tumor "
            "split remains unresolved."
        )
    elif overall is not None:
        tumor_pct = min(max(float(overall), 0.0), 1.0)
        lines.append(
            f"**Chain summary:** observed expression → library-prep-aware "
            f"artifact expectations → preservation-adjusted quantification → "
            f"decomposition subtracts {1 - tumor_pct:.0%} as non-tumor "
            "reference components → the RNA model's estimated tumor signal is used for "
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
    *,
    report_view: ReportView,
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

    purity_is_unresolved = (
        report_view.purity.status == "discordant_estimators"
    )

    if purity_is_unresolved:
        fig, ax = plt.subplots(figsize=(10, 2.6))
        ax.barh(
            [0],
            [1.0],
            color="#9e9e9e",
            edgecolor="white",
            hatch="//",
            label="RNA model could not resolve composition",
        )
        ax.text(
            0.5,
            0,
            "Estimated patient tumor / non-tumor\ncomposition unresolved",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
        )
        ax.set_xlim(0, 1.0)
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_title(
            "Estimated RNA composition — unresolved",
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            fontsize=8,
            frameon=False,
        )
        fig.subplots_adjust(left=0.04, right=0.98, top=0.78, bottom=0.32)
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        plt.close(fig)
        return save_to_filename

    tumor_frac = float(fractions.pop("tumor", 0.0))
    non_tumor = sorted(
        ((c, f) for c, f in fractions.items() if f > 0.005),
        key=lambda kv: -kv[1],
    )
    rest_frac = sum(f for c, f in fractions.items() if f <= 0.005 and f > 0)

    labels = ["Patient tumor contribution (estimated by RNA model)"] + [
        _compartment_label(c) for c, _ in non_tumor
    ]
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
                f"{val:.0%}",
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
    ax.set_title("Estimated RNA composition", fontsize=11, fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=min(4, len(labels)),
        fontsize=8,
        frameon=False,
    )

    fig.subplots_adjust(left=0.04, right=0.98, top=0.78, bottom=0.34)
    fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    return save_to_filename
