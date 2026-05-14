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

from collections import defaultdict
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np

from .common import _guess_gene_cols
from .plot_data_helpers import _strip_ensembl_version
from pirlygenes.gene_sets_cancer import (
    therapy_target_gene_id_to_name, CTA_gene_id_to_name,
)
from trufflepig.reference import (
    pan_cancer_expression,
)
from pirlygenes.load_dataset import get_data
from .plot_scatter import resolve_cancer_type


_FN1_GENE_ID = "ENSG00000115414"
_FN1_EDB_MIN_TPM = 5.0
_FN1_EDB_MIN_FRACTION = 0.10
_FN1_EDB_TRANSCRIPT_IDS = frozenset(
    {
        "ENST00000323926",  # FN1-201
        "ENST00000354785",  # FN1-203
        "ENST00000432072",  # FN1-209
        "ENST00000456923",  # FN1-213
    }
)
_FN1_EDB_TRANSCRIPT_NAMES = frozenset({"FN1-201", "FN1-203", "FN1-209", "FN1-213"})
_FN1_EDB_EXON_IDS = frozenset({"ENSE00000965897", "ENSE00001744777"})
_FN1_EDB_EXON_INTERVALS = frozenset(
    {
        (215392931, 215393203),
        (215392931, 215393147),
    }
)


@lru_cache(maxsize=1)
def _fn1_edb_transcript_ids():
    """Return versionless FN1 transcript IDs carrying the EDB cassette exon.

    The curated FN1 ADC hook in this repo is PYX-201 / NCT05720117,
    which is specific to EDB+ fibronectin rather than total FN1.
    Resolve the EDB+ transcript set once from local Ensembl releases,
    with a hardcoded fallback so the gate still works if pyensembl
    metadata is unavailable at runtime.
    """
    ids = set(_FN1_EDB_TRANSCRIPT_IDS)
    try:
        from pirlygenes.gene_ids import genomes
    except Exception:
        return frozenset(ids)

    # ``genomes`` is sorted newest-release-first. FN1-EDB transcript IDs
    # are stable Ensembl accessions, so once the first release with an
    # intact ``exon`` schema yields exon data we have the authoritative
    # answer — stop rather than querying older releases whose schemas
    # may be broken (see #192: release 54's ``exon`` table is missing
    # ``exon_id``). Each release is wrapped in try/except so a broken
    # release is skipped, not fatal.
    for genome in genomes:
        try:
            genes = genome.genes_by_name("FN1")
        except Exception:
            continue
        if not genes:
            continue
        gene = genes[0]
        try:
            transcript_ids = genome.transcript_ids_of_gene_id(gene.gene_id)
        except Exception:
            continue
        release_saw_exons = False
        for transcript_id in transcript_ids:
            try:
                transcript = genome.transcript_by_id(transcript_id)
            except Exception:
                continue
            if getattr(transcript, "biotype", None) != "protein_coding":
                continue
            try:
                exons = list(getattr(transcript, "exons", []) or [])
            except Exception:
                continue
            if exons:
                release_saw_exons = True
            try:
                exon_ids = {
                    str(exon.exon_id)
                    for exon in exons
                    if getattr(exon, "exon_id", None)
                }
            except Exception:
                exon_ids = set()
            try:
                exon_intervals = {
                    (int(exon.start), int(exon.end))
                    for exon in exons
                    if getattr(exon, "start", None) is not None
                    and getattr(exon, "end", None) is not None
                }
            except Exception:
                exon_intervals = set()
            if (
                getattr(transcript, "transcript_name", None)
                in _FN1_EDB_TRANSCRIPT_NAMES
                or exon_ids & _FN1_EDB_EXON_IDS
                or exon_intervals & _FN1_EDB_EXON_INTERVALS
            ):
                ids.add(str(transcript.transcript_id).split(".", 1)[0])
        if release_saw_exons:
            break
    return frozenset(ids)


def _summarize_fn1_edb_transcript_support(df_gene_expr):
    """Summarize whether transcript-level input supports EDB+ FN1 targeting."""
    import pandas as pd

    prefix = (
        "PYX-201 (NCT05720117) targets EDB+ FN1; bulk gene-level FN1 alone "
        "is not sufficient evidence"
    )
    empty = {
        "supported": False,
        "note": f"{prefix} because transcript-level data is unavailable.",
        "edb_tpm": None,
        "edb_fraction": None,
        "supporting_transcripts": "",
    }

    tx_df = getattr(df_gene_expr, "attrs", {}).get("transcript_expression")
    if not isinstance(tx_df, pd.DataFrame) or tx_df.empty:
        return empty
    if "transcript_id" not in tx_df.columns or "TPM" not in tx_df.columns:
        return empty

    tx_ids = tx_df["transcript_id"].astype(str).str.split(".", n=1).str[0]
    fn1_mask = pd.Series(False, index=tx_df.index, dtype=bool)
    if "ensembl_gene_id" in tx_df.columns:
        fn1_mask |= (
            tx_df["ensembl_gene_id"]
            .astype(str)
            .str.split(".", n=1)
            .str[0]
            .eq(_FN1_GENE_ID)
        )
    if "gene_symbol" in tx_df.columns:
        fn1_mask |= tx_df["gene_symbol"].fillna("").astype(str).str.upper().eq("FN1")
    if not fn1_mask.any():
        return {
            **empty,
            "note": f"{prefix} because no FN1 transcripts were retained in the transcript-level input.",
        }

    tx_tpm = pd.to_numeric(tx_df["TPM"], errors="coerce").fillna(0.0)
    total_fn1_tpm = float(tx_tpm[fn1_mask].sum())
    edb_mask = fn1_mask & tx_ids.isin(_fn1_edb_transcript_ids())
    edb_tpm = float(tx_tpm[edb_mask].sum())
    edb_fraction = (edb_tpm / total_fn1_tpm) if total_fn1_tpm > 0 else 0.0

    support_by_tx = (
        pd.DataFrame(
            {
                "transcript_id": tx_ids[edb_mask].values,
                "TPM": tx_tpm[edb_mask].values,
            }
        )
        .groupby("transcript_id", sort=False)["TPM"]
        .sum()
        .sort_values(ascending=False)
    )
    supporting_transcripts = ", ".join(
        f"{transcript_id}:{value:.1f}"
        for transcript_id, value in support_by_tx.head(4).items()
    )

    if edb_tpm <= 0:
        note = f"{prefix} because no EDB+ FN1 transcripts were detected."
        supported = False
    elif edb_tpm >= _FN1_EDB_MIN_TPM and edb_fraction >= _FN1_EDB_MIN_FRACTION:
        note = (
            f"EDB+ FN1 transcript support present for PYX-201 / NCT05720117: "
            f"{edb_tpm:.1f} TPM ({edb_fraction:.0%} of FN1 transcript signal)."
        )
        supported = True
    else:
        note = (
            f"{prefix} because EDB+ FN1 transcripts reached only {edb_tpm:.1f} TPM "
            f"({edb_fraction:.0%} of FN1 transcript signal), below the current gate."
        )
        supported = False

    return {
        "supported": supported,
        "note": note,
        "edb_tpm": edb_tpm,
        "edb_fraction": edb_fraction,
        "supporting_transcripts": supporting_transcripts,
    }


def _apply_therapy_support_gate(symbol, therapies, fn1_support):
    """Return gated therapies plus structured support metadata."""
    therapies = set(therapies or ())
    if not therapies:
        return therapies, None, "", None, None, ""
    if symbol != "FN1":
        return therapies, True, "", None, None, ""
    if fn1_support.get("supported"):
        return (
            therapies,
            True,
            fn1_support.get("note", ""),
            fn1_support.get("edb_tpm"),
            fn1_support.get("edb_fraction"),
            fn1_support.get("supporting_transcripts", ""),
        )
    return (
        set(),
        False,
        fn1_support.get("note", ""),
        fn1_support.get("edb_tpm"),
        fn1_support.get("edb_fraction"),
        fn1_support.get("supporting_transcripts", ""),
    )


ESSENTIAL_TISSUES = [
    "brain",
    "heart",
    "liver",
    "lung",
    "kidney",
    "bone_marrow",
    "spleen",
    "pancreas",
    "colon",
    "stomach",
]

_THERAPY_PLOT_ORDER = [
    "ADC",
    "CAR-T",
    "TCR-T",
    "bispecific-antibodies",
    "radioligand",
]
_THERAPY_PLOT_LABELS = {
    "ADC": "ADC",
    "CAR-T": "CAR-T",
    "TCR-T": "TCR-T",
    "bispecific-antibodies": "Bispecific",
    "radioligand": "Radio",
}

# Map essential tissue labels to nTPM column names
_ESSENTIAL_TISSUE_COLS = {
    "brain": [
        "nTPM_cerebral_cortex",
        "nTPM_cerebellum",
        "nTPM_basal_ganglia",
        "nTPM_hippocampal_formation",
        "nTPM_amygdala",
        "nTPM_midbrain",
        "nTPM_hypothalamus",
        "nTPM_spinal_cord",
        "nTPM_choroid_plexus",
    ],
    "heart": ["nTPM_heart_muscle"],
    "liver": ["nTPM_liver"],
    "lung": ["nTPM_lung"],
    "kidney": ["nTPM_kidney"],
    "bone_marrow": ["nTPM_bone_marrow"],
    "spleen": ["nTPM_spleen"],
    "pancreas": ["nTPM_pancreas"],
    "colon": ["nTPM_colon"],
    "stomach": ["nTPM_stomach"],
}


def _ordered_therapy_tuple(therapies):
    therapies = set(therapies)
    return tuple(sorted(therapies, key=_THERAPY_PLOT_ORDER.index))


def _therapy_combo_label(therapies):
    if not therapies:
        return "Other"
    return " + ".join(_THERAPY_PLOT_LABELS[t] for t in therapies)


def _therapy_combo_sort_key(therapies):
    return (len(therapies), [_THERAPY_PLOT_ORDER.index(t) for t in therapies])


def _therapy_base_colors():
    import seaborn as sns

    return dict(
        zip(_THERAPY_PLOT_ORDER, sns.color_palette("Set2", len(_THERAPY_PLOT_ORDER)))
    )


def _therapy_combo_colors(therapy_combos):
    base_palette = _therapy_base_colors()
    combo_to_color = {}
    for combo in sorted(set(therapy_combos), key=_therapy_combo_sort_key):
        if len(combo) == 0:
            combo_to_color[combo] = "gray"
        elif len(combo) == 1:
            combo_to_color[combo] = base_palette[combo[0]]
        else:
            rgb = np.array([base_palette[t] for t in combo]).mean(axis=0)
            combo_to_color[combo] = tuple(rgb.clip(0, 1))
    return combo_to_color


def _draw_therapy_marker(
    ax,
    x,
    y,
    therapies,
    marker="o",
    size=80,
    alpha=0.85,
    zorder=3,
):
    base_palette = _therapy_base_colors()
    ordered = _ordered_therapy_tuple(therapies)
    fill_color = base_palette.get(ordered[0], "gray") if ordered else "gray"

    ax.scatter(
        x,
        y,
        s=size,
        color=fill_color,
        marker=marker,
        alpha=alpha,
        edgecolors="white",
        linewidths=0.6,
        zorder=zorder,
    )

    ring_size = size + 34
    for therapy in ordered[1:]:
        ax.scatter(
            x,
            y,
            s=ring_size,
            facecolors="none",
            edgecolors=base_palette.get(therapy, "gray"),
            marker=marker,
            alpha=0.95,
            linewidths=1.8,
            zorder=zorder - 0.1,
        )
        ring_size += 34

    return fill_color


def _approved_radioligand_gene_ids():
    df = get_data("radioligand-targets")
    if "Status_Bucket" not in df.columns:
        return set()
    mask = df["Status_Bucket"].astype(str).str.startswith("FDA_approved", na=False)
    return set(df.loc[mask, "Ensembl_Gene_ID"].astype(str))


def _collect_ranked_therapy_targets(
    df_gene_expr, top_k=10, tpm_threshold=30, extra_symbols=None
):
    gene_id_col, gene_name_col = _guess_gene_cols(df_gene_expr)

    df = df_gene_expr.copy()
    df[gene_id_col] = df[gene_id_col].astype(str).map(_strip_ensembl_version)

    tpm_col = (
        "TPM"
        if "TPM" in df.columns
        else next((c for c in df.columns if c.lower() == "tpm"), None)
    )
    if tpm_col is None:
        raise KeyError(f"No TPM column found. Columns: {list(df.columns)}")

    sample_tpm = dict(zip(df[gene_id_col].astype(str), df[tpm_col].astype(float)))
    sample_name = dict(zip(df[gene_id_col].astype(str), df[gene_name_col].astype(str)))

    therapy_to_targets = {}
    gene_to_therapies = defaultdict(set)
    gene_name_from_sets = {}
    for therapy in _THERAPY_PLOT_ORDER:
        targets = {}
        for gid, sym in therapy_target_gene_id_to_name(therapy).items():
            gid_clean = _strip_ensembl_version(str(gid))
            targets[gid_clean] = sym
            gene_to_therapies[gid_clean].add(therapy)
            gene_name_from_sets.setdefault(gid_clean, sym)
        therapy_to_targets[therapy] = targets

    approved_sources = {
        "ADC": "ADC-approved",
        "CAR-T": "CAR-T",
        "TCR-T": "TCR-T-approved",
        "bispecific-antibodies": "bispecific-antibodies-approved",
    }
    gene_to_approved_therapies = defaultdict(set)
    for therapy, dataset in approved_sources.items():
        for gid in therapy_target_gene_id_to_name(dataset):
            gid_clean = _strip_ensembl_version(str(gid))
            gene_to_approved_therapies[gid_clean].add(therapy)
    for gid in _approved_radioligand_gene_ids():
        gene_to_approved_therapies[_strip_ensembl_version(str(gid))].add("radioligand")

    selected_gene_ids = set()
    for therapy, targets in therapy_to_targets.items():
        scored = []
        for gid in targets:
            tpm = sample_tpm.get(gid, 0.0)
            if tpm >= tpm_threshold:
                scored.append((gid, tpm))
        scored.sort(
            key=lambda item: (
                -item[1],
                sample_name.get(item[0], gene_name_from_sets.get(item[0], item[0])),
            )
        )
        selected_gene_ids.update(gid for gid, _ in scored[:top_k])

    extra_symbol_set = {
        str(sym).strip().upper()
        for sym in (extra_symbols or [])
        if str(sym).strip() and str(sym).strip().lower() != "nan"
    }
    if extra_symbol_set:
        sample_symbol_to_ids = defaultdict(list)
        for gid, sym in sample_name.items():
            sample_symbol_to_ids[str(sym).strip().upper()].append(gid)
        for sym_upper in sorted(extra_symbol_set):
            for gid in sample_symbol_to_ids.get(sym_upper, []):
                selected_gene_ids.add(gid)
                gene_name_from_sets.setdefault(gid, sample_name.get(gid, sym_upper))

        missing_symbols = extra_symbol_set - set(sample_symbol_to_ids)
        if missing_symbols:
            ref_symbols = pan_cancer_expression(technical_rna_normalize=True)[
                ["Ensembl_Gene_ID", "Symbol"]
            ].drop_duplicates()
            for row in ref_symbols.itertuples(index=False):
                sym = str(row.Symbol).strip()
                if sym.upper() not in missing_symbols:
                    continue
                gid = _strip_ensembl_version(str(row.Ensembl_Gene_ID))
                selected_gene_ids.add(gid)
                sample_tpm.setdefault(gid, 0.0)
                gene_name_from_sets.setdefault(gid, sym)

    records = []
    for gid in sorted(
        selected_gene_ids,
        key=lambda gene_id: (
            -sample_tpm.get(gene_id, 0.0),
            sample_name.get(gene_id, gene_name_from_sets.get(gene_id, gene_id)),
        ),
    ):
        therapies = _ordered_therapy_tuple(gene_to_therapies.get(gid, ()))
        approved_therapies = _ordered_therapy_tuple(
            gene_to_approved_therapies.get(gid, ())
        )
        display_name = sample_name.get(gid) or gene_name_from_sets.get(gid) or gid
        records.append(
            {
                "gene_id": gid,
                "symbol": display_name,
                "sample_tpm": sample_tpm.get(gid, 0.0),
                "therapies": therapies,
                "therapy_label": _therapy_combo_label(therapies),
                "approved_therapies": approved_therapies,
                "approved_label": _therapy_combo_label(approved_therapies)
                if approved_therapies
                else "",
                "has_approved": bool(approved_therapies),
            }
        )
    return records


def plot_therapy_target_tissues(
    df_gene_expr,
    top_k=10,
    tpm_threshold=30,
    extra_symbols=None,
    save_to_filename=None,
    save_dpi=300,
):
    """For top expressed therapy targets, show healthy tissue expression vs sample.

    For each therapy category, takes the top K genes above a TPM threshold
    and creates a sorted bar plot of HPA normal tissue expression alongside
    the sample TPM value.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Patient expression data.
    top_k : int
        Number of top genes per therapy category.
    tpm_threshold : float
        Minimum sample TPM to include a gene.
    save_to_filename : str or None
        Output path (PDF with one page per gene, or PNG directory).
    """
    from pathlib import Path
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.backends.backend_pdf import PdfPages

    # Load normal tissue expression
    ref = pan_cancer_expression(technical_rna_normalize=True)
    ntpm_cols = sorted([c for c in ref.columns if c.startswith("nTPM_")])
    tissue_labels = [c.replace("nTPM_", "").replace("_", " ") for c in ntpm_cols]
    ref_by_id = ref.drop_duplicates(subset="Ensembl_Gene_ID").set_index(
        "Ensembl_Gene_ID"
    )

    records = _collect_ranked_therapy_targets(
        df_gene_expr,
        top_k=top_k,
        tpm_threshold=tpm_threshold,
        extra_symbols=extra_symbols,
    )
    if not records:
        print(f"No therapy targets above {tpm_threshold} TPM")
        return None
    combo_to_color = _therapy_combo_colors([record["therapies"] for record in records])

    figures = {}
    out = Path(save_to_filename) if save_to_filename else None
    pdf = PdfPages(out) if out is not None and out.suffix.lower() == ".pdf" else None
    if out is not None and out.suffix.lower() != ".pdf":
        out_dir = out.parent / out.stem
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    # Generate one page per gene
    for record in records:
        gid = record["gene_id"]
        sym = record["symbol"]
        s_tpm = record["sample_tpm"]
        combo_color = combo_to_color.get(record["therapies"], "#d62728")

        # Get tissue expression
        if gid in ref_by_id.index:
            tissue_vals = [
                float(ref_by_id.loc[gid, c]) if c in ref_by_id.columns else 0
                for c in ntpm_cols
            ]
        else:
            tissue_vals = [0] * len(ntpm_cols)

        # Sort tissues by expression
        sorted_pairs = sorted(zip(tissue_labels, tissue_vals), key=lambda x: -x[1])
        t_labels = [p[0] for p in sorted_pairs]
        t_vals = [p[1] for p in sorted_pairs]

        fig, ax = plt.subplots(figsize=(10, 8))
        y = np.arange(len(t_labels))
        ax.barh(y, t_vals, color="#aec7e8", edgecolor="none", height=0.7)

        # Sample TPM as a vertical line
        ax.axvline(x=s_tpm, color=combo_color, linewidth=2, linestyle="-", alpha=0.85)

        ax.set_yticks(y)
        ax.set_yticklabels(t_labels, fontsize=7)
        ax.set_xlabel("Expression (nTPM / TPM)", fontsize=10)
        subtitle = (
            f"Sample TPM: {s_tpm:.1f} | therapy categories: {record['therapy_label']}"
        )
        if record["has_approved"]:
            subtitle += f" | approved: {record['approved_label']}"
            ax.scatter(
                [0.97],
                [0.96],
                transform=ax.transAxes,
                marker="^",
                s=70,
                color=combo_color,
                edgecolors="black",
                linewidths=0.5,
                clip_on=False,
                zorder=6,
            )
            ax.text(
                0.93,
                0.96,
                "approved",
                transform=ax.transAxes,
                fontsize=8,
                ha="right",
                va="center",
                alpha=0.85,
            )
        ax.set_title(
            f"{sym} — healthy tissue vs sample\n{subtitle}",
            fontsize=11,
        )
        legend_handles = [
            Patch(facecolor="#aec7e8", edgecolor="none", label="Normal tissue (nTPM)"),
            Line2D(
                [],
                [],
                color=combo_color,
                linewidth=2,
                label=f"Sample TPM = {s_tpm:.1f}",
            ),
        ]
        if record["has_approved"]:
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    marker="^",
                    linestyle="None",
                    color=combo_color,
                    markeredgecolor="black",
                    label=f"Approved target ({record['approved_label']})",
                )
            )
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=False)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        if pdf is not None:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        elif out_dir is not None:
            fig.savefig(out_dir / f"{sym}.png", dpi=save_dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            figures[sym] = fig

    if pdf is not None:
        pdf.close()
        print(f"Saved {out} ({len(records)} pages)")
    elif out_dir is not None:
        print(f"Saved {len(records)} PNGs to {out_dir}/")

    return figures


def plot_therapy_target_safety(
    df_gene_expr,
    top_k=10,
    tpm_threshold=30,
    extra_symbols=None,
    save_to_filename=None,
    save_dpi=300,
    figsize=(12, 10),
):
    """Scatter: sample TPM vs max essential-tissue expression for therapy targets.

    X-axis is sample expression, Y-axis is max expression in essential
    tissues (brain, heart, liver, lung, kidney). Genes in the lower-right
    (high sample, low essential tissue) are the safest therapy targets.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Patient expression data.
    top_k : int
        Number of top genes per therapy category to include.
    tpm_threshold : float
        Minimum sample TPM to include a gene.
    save_to_filename : str or None
        Output path.
    """
    from adjustText import adjust_text
    from matplotlib.lines import Line2D

    ref = pan_cancer_expression(technical_rna_normalize=True)
    ref_by_id = ref.drop_duplicates(subset="Ensembl_Gene_ID").set_index(
        "Ensembl_Gene_ID"
    )

    # Essential tissue columns
    essential_cols = []
    for tissue, cols in _ESSENTIAL_TISSUE_COLS.items():
        essential_cols.extend([c for c in cols if c in ref_by_id.columns])

    records = _collect_ranked_therapy_targets(
        df_gene_expr,
        top_k=top_k,
        tpm_threshold=tpm_threshold,
        extra_symbols=extra_symbols,
    )
    if not records:
        print(f"No therapy targets above {tpm_threshold} TPM")
        return None, None
    base_palette = _therapy_base_colors()

    fig, ax = plt.subplots(figsize=figsize)
    texts = []
    point_x = []
    point_y = []

    for record in records:
        gid = record["gene_id"]
        sym = record["symbol"]
        tpm = record["sample_tpm"]
        max_ess = 0
        if gid in ref_by_id.index:
            vals = [
                float(ref_by_id.loc[gid, c])
                for c in essential_cols
                if c in ref_by_id.columns
            ]
            max_ess = max(vals) if vals else 0
        y_value = max_ess + 0.1
        _draw_therapy_marker(
            ax,
            tpm,
            y_value,
            record["therapies"],
            marker="o",
            size=80,
            alpha=0.85,
            zorder=3,
        )
        point_x.append(tpm)
        point_y.append(y_value)
        texts.append(
            ax.text(
                tpm * 1.03,
                y_value * 1.03,
                sym,
                fontsize=8,
                va="bottom",
                ha="left",
                alpha=0.85,
                zorder=4,
            )
        )

    therapy_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markerfacecolor=base_palette[therapy],
            markeredgecolor="white",
            color=base_palette[therapy],
            label=_THERAPY_PLOT_LABELS[therapy],
            markersize=8,
        )
        for therapy in _THERAPY_PLOT_ORDER
    ]
    legend = ax.legend(
        handles=therapy_handles,
        loc="upper left",
        fontsize=8,
        frameon=False,
        title="Therapy categories\n(extra categories = outer rings)",
        title_fontsize=9,
    )
    ax.add_artist(legend)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample TPM", fontsize=11)
    ax.set_ylabel(
        "Max expression in selected essential tissues (nTPM)",
        fontsize=10,
    )
    ax.set_title(
        "Therapy target safety: sample expression vs essential tissue expression\n"
        "(brain, heart, liver, lung, kidney, bone marrow, spleen, pancreas, colon, stomach)",
        fontsize=11,
    )
    try:
        from .common import build_sample_tpm_by_symbol
        from .plot_reference_lines import add_p90_reference_line

        add_p90_reference_line(
            ax,
            build_sample_tpm_by_symbol(df_gene_expr),
            orientation="vertical",
        )
    except Exception:
        pass
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    adjust_text(
        texts,
        x=point_x,
        y=point_y,
        ax=ax,
        expand=(1.15, 1.45),
        force_text=(0.15, 0.3),
        force_static=(0.12, 0.25),
        force_pull=(0.01, 0.02),
        arrowprops=dict(arrowstyle="-", color="#666666", alpha=0.35, lw=0.6),
        min_arrow_len=6,
        ensure_inside_axes=False,
    )

    # Diagonal reference
    lims = [
        max(ax.get_xlim()[0], ax.get_ylim()[0]),
        min(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    if lims[1] > lims[0]:
        ax.plot(lims, lims, ":", color="#cccccc", linewidth=0.8, alpha=0.5, zorder=0)

    fig.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


# -------------------- gene-set vs essential-tissue toxicity view -----------


def _resolve_gene_set_symbols(gene_set):
    """Return a list of gene symbols from a flexible gene-set specifier.

    Accepts:
        - an iterable of symbols: ("IFI6", "ISG15", ...) used directly
        - a str referring to a Category in `data/gene-sets.csv`
          (case-insensitive; the set `"Interferon response"` or
          `"interferon_response"` would match). Loaded via get_data.

    Raises ValueError if a string reference doesn't resolve.
    """
    if isinstance(gene_set, str):
        df = get_data("gene-sets")
        target = gene_set.strip().lower().replace(" ", "_")
        categories_available = {
            str(c).strip().lower().replace(" ", "_"): c
            for c in df["Category"].dropna().unique()
        }
        matched = categories_available.get(target)
        if matched is None:
            raise ValueError(
                f"Gene set {gene_set!r} not found in data/gene-sets.csv. "
                f"Available: {sorted(categories_available.values())}"
            )
        symbols = df.loc[df["Category"] == matched, "Symbol"].astype(str).tolist()
        return [s for s in symbols if s]
    return [str(s) for s in gene_set]


def plot_geneset_vs_vital_tissues(
    df_gene_expr,
    gene_set,
    title=None,
    toxicity_tpm_threshold=10.0,
    vital_tissues=None,
    save_to_filename=None,
    save_dpi=300,
    figsize=None,
):
    """Plot a sample's expression of a gene set against vital-tissue baselines.

    For each gene in the set, renders a horizontal strip on a log TPM
    axis with:
        - the sample's TPM (blue filled marker)
        - each vital tissue's nTPM (small tissue-colored dots, flipped
          to red when the tissue value exceeds `toxicity_tpm_threshold`
          — a visual flag that targeting this gene therapeutically
          would affect a tissue that can't tolerate damage / regenerate)

    Per-gene reading: "is the sample high AND are vital tissues also
    high on this gene?" Genes where only the sample is high are safer
    therapeutic targets; genes where one or more vital tissues are
    above the threshold are toxicity risks.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Sample expression (same schema expected by `_guess_gene_cols`).
    gene_set : str or iterable of str
        Either a gene-set Category name in `data/gene-sets.csv`
        (e.g. "Interferon response", "MHC1_presentation") or an explicit
        list of gene symbols.
    title : str or None
        Figure title.
    toxicity_tpm_threshold : float
        nTPM above which a vital-tissue dot is colored red. Default 10.0.
    vital_tissues : list of str or None
        Keys into `_ESSENTIAL_TISSUE_COLS` to display. Default: all.
    save_to_filename, save_dpi : plot save options.
    figsize : tuple or None
        Explicit size. Default scales with gene count.

    Returns
    -------
    matplotlib.figure.Figure
    """
    symbols = _resolve_gene_set_symbols(gene_set)
    if not symbols:
        print(f"Gene set {gene_set!r} resolved to zero genes.")
        return None

    if vital_tissues is None:
        vital_tissues = list(_ESSENTIAL_TISSUE_COLS.keys())
    else:
        missing = [t for t in vital_tissues if t not in _ESSENTIAL_TISSUE_COLS]
        if missing:
            raise ValueError(
                f"Unknown vital_tissues: {missing}. "
                f"Available: {list(_ESSENTIAL_TISSUE_COLS.keys())}"
            )

    gene_id_col, gene_name_col = _guess_gene_cols(df_gene_expr)
    df = df_gene_expr.copy()
    df[gene_id_col] = df[gene_id_col].astype(str).map(_strip_ensembl_version)
    tpm_col = (
        "TPM"
        if "TPM" in df.columns
        else next((c for c in df.columns if c.lower() == "tpm"), None)
    )
    if tpm_col is None:
        raise KeyError(f"No TPM column in sample. Columns: {list(df.columns)}")

    ref = pan_cancer_expression(technical_rna_normalize=True)
    id_to_sym = dict(zip(ref["Ensembl_Gene_ID"], ref["Symbol"]))
    sample_by_symbol = {}
    for _, row in df.iterrows():
        gid = str(row[gene_id_col])
        sym = id_to_sym.get(gid)
        if sym is None:
            sym = str(row.get(gene_name_col, ""))
        if not sym:
            continue
        tpm = float(row[tpm_col])
        if sym not in sample_by_symbol or tpm > sample_by_symbol[sym]:
            sample_by_symbol[sym] = tpm

    # For tissue groups like "brain" that span several nTPM_* columns,
    # use the MAX — any single CNS region with high expression is a
    # toxicity concern, no need to average it out.
    ref_by_sym = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    tissue_columns_resolved = {
        tissue: [c for c in _ESSENTIAL_TISSUE_COLS[tissue] if c in ref_by_sym.columns]
        for tissue in vital_tissues
    }

    def _tissue_value(sym, tissue):
        cols = tissue_columns_resolved.get(tissue, [])
        if sym not in ref_by_sym.index or not cols:
            return None
        return float(ref_by_sym.loc[sym, cols].astype(float).max())

    rows = []
    for sym in symbols:
        sample_tpm = sample_by_symbol.get(sym)
        tissue_vals = {t: _tissue_value(sym, t) for t in vital_tissues}
        has_any = (sample_tpm is not None and sample_tpm > 0) or any(
            v is not None and v > 0 for v in tissue_vals.values()
        )
        if has_any:
            rows.append((sym, sample_tpm or 0.0, tissue_vals))

    if not rows:
        print(f"Gene set {gene_set!r} had no expression in sample or reference.")
        return None

    n = len(rows)
    if figsize is None:
        figsize = (11, 0.32 * n + 2.2)

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(n)

    tissue_colors = {
        "brain": "#6a3d9a",
        "heart": "#e31a1c",
        "liver": "#ff7f00",
        "lung": "#1f78b4",
        "kidney": "#33a02c",
        "bone_marrow": "#b15928",
        "spleen": "#a6cee3",
        "pancreas": "#fb9a99",
        "colon": "#b2df8a",
        "stomach": "#fdbf6f",
    }

    # Vertical jitter spread for tissue dots — overlapping tissues at
    # similar TPM become distinguishable. Spread is symmetric around the
    # gene's row index, inside ±0.35 so rows don't visually collide.
    n_tissues = max(1, len(vital_tissues))
    jitter_map = {
        tissue: (idx / max(1, n_tissues - 1) - 0.5) * 0.7
        for idx, tissue in enumerate(vital_tissues)
    }

    for i, (sym, sample_tpm, tissue_vals) in enumerate(rows):
        x_sample = max(sample_tpm, 0.05)
        ax.scatter(
            [x_sample],
            [i],
            s=110,
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
            label=None,
        )
        ax.text(
            x_sample * 1.2,
            i,
            f"{sample_tpm:.0f}",
            fontsize=7.5,
            va="center",
            color="#1f77b4",
            fontweight="bold",
            zorder=6,
        )
        for tissue, val in tissue_vals.items():
            if val is None:
                continue
            base = tissue_colors.get(tissue, "#888888")
            if val >= toxicity_tpm_threshold:
                color = "#d62728"
                edge = "#7a0000"
                lw = 0.9
                size = 55
            else:
                color = base
                edge = "white"
                lw = 0.4
                size = 32
            x = max(val, 0.05)
            y_pos = i + jitter_map.get(tissue, 0.0)
            ax.scatter(
                [x],
                [y_pos],
                s=size,
                color=color,
                edgecolor=edge,
                linewidth=lw,
                zorder=4 if color == "#d62728" else 3,
                alpha=0.9,
            )

    ax.set_xscale("log")
    ax.set_xlim(0.05, 10_000)
    ax.axvline(
        toxicity_tpm_threshold,
        linestyle="--",
        color="#d62728",
        linewidth=0.8,
        alpha=0.5,
        zorder=1,
    )
    try:
        from .common import build_sample_tpm_by_symbol
        from .plot_reference_lines import add_p90_reference_line

        add_p90_reference_line(
            ax,
            build_sample_tpm_by_symbol(df_gene_expr),
            orientation="vertical",
        )
    except Exception:
        pass
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(
        "TPM (log scale) — sample in blue, vital tissues colored by tissue; red = above threshold"
    )
    if title is None:
        title = (
            f"{gene_set} vs vital tissues"
            if isinstance(gene_set, str)
            else "Gene set vs vital tissues"
        )
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D

    legend_entries = [
        Line2D(
            [],
            [],
            marker="o",
            color="#1f77b4",
            markeredgecolor="black",
            markersize=10,
            linestyle="",
            label="sample",
        ),
    ]
    for tissue in vital_tissues:
        legend_entries.append(
            Line2D(
                [],
                [],
                marker="o",
                color=tissue_colors.get(tissue, "#888888"),
                markersize=6,
                linestyle="",
                label=tissue.replace("_", " "),
            )
        )
    legend_entries.append(
        Line2D(
            [],
            [],
            marker="o",
            color="#d62728",
            markeredgecolor="#7a0000",
            markersize=8,
            linestyle="",
            label=f"tissue > {toxicity_tpm_threshold:g} TPM",
        )
    )
    ax.legend(
        handles=legend_entries,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(len(legend_entries), 6),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig


# -------------------- per-cancer-type CTA detail view ----------------------


def plot_ctas_vs_cancer_type_detail(
    df_gene_expr,
    cancer_type,
    top_k=30,
    min_sample_tpm=1.0,
    save_to_filename=None,
    save_dpi=300,
    figsize=None,
):
    """Per-sample CTA expression zoomed to one cancer type.

    Each row is one CTA gene. Each row plots, on a single log TPM axis:

        - sample TPM ................ large blue filled circle + label
        - TCGA cohort median ........ orange diamond (cancer_type's FPKM_*
                                      column, treated as TPM-equivalent)
        - tissue-of-origin normal ... green square (nTPM for the normal
                                      tissue mapped from cancer_type via
                                      CANCER_TO_TISSUE)
        - testis .................... purple triangle (expected baseline
                                      for CTAs — the tissue they're
                                      *defined* against)
        - max non-testis vital ...... red X when its value crosses the
                                      10-TPM toxicity threshold, else a
                                      small gray X. The offending tissue
                                      name is annotated to the right.

    Reads per gene:

        - sample ≫ cohort → outlier up-regulation in this patient
        - sample ≈ cohort → this CTA is typical for the cancer type
        - sample ≪ cohort → cohort expresses it but this patient doesn't
        - high red X → toxicity risk (a non-testis vital tissue also
          expresses this CTA, regardless of tumor/cohort signal)

    The data is cohort-level (one median per cancer type), so "per-
    patient variation within PRAD" is not available. This plot shows
    what IS in the dataset: how a single sample compares to the per-
    cohort median + normal-tissue baselines.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Sample expression with gene ID, name, TPM columns.
    cancer_type : str
        TCGA cancer code or alias (e.g. "PRAD", "prostate").
    top_k : int
        Render at most this many CTAs, ranked by sample TPM.
    min_sample_tpm : float
        Skip CTAs where sample TPM is below this; keeps the plot focused
        on genes that are actually expressed in this patient.
    save_to_filename, save_dpi : plot save options.
    figsize : tuple or None
        Explicit size. Default scales with the number of rendered rows.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.lines import Line2D

    from .tumor_purity import CANCER_TO_TISSUE

    cancer_code = resolve_cancer_type(cancer_type)
    ref = pan_cancer_expression(technical_rna_normalize=True)
    cancer_col = f"FPKM_{cancer_code}"
    if cancer_col not in ref.columns:
        raise ValueError(
            f"Unknown cancer_type {cancer_type!r} → code {cancer_code!r} "
            f"has no FPKM column in the reference."
        )

    # CTAs as Ensembl IDs + display symbols.
    cta_id_to_name = CTA_gene_id_to_name()
    if not cta_id_to_name:
        print("No CTAs available in reference.")
        return None

    # Build a row per CTA: sample TPM, cancer-cohort TPM-equivalent,
    # tissue-of-origin nTPM, testis nTPM, max non-testis vital nTPM.
    gene_id_col, gene_name_col = _guess_gene_cols(df_gene_expr)
    sample = df_gene_expr.copy()
    sample[gene_id_col] = sample[gene_id_col].astype(str).map(_strip_ensembl_version)
    tpm_col = (
        "TPM"
        if "TPM" in sample.columns
        else next((c for c in sample.columns if c.lower() == "tpm"), None)
    )
    if tpm_col is None:
        raise KeyError(f"No TPM column in sample. Columns: {list(sample.columns)}")
    sample_tpm = dict(
        zip(
            sample[gene_id_col].astype(str),
            sample[tpm_col].astype(float),
        )
    )

    ref_by_id = ref.drop_duplicates(subset="Ensembl_Gene_ID").set_index(
        "Ensembl_Gene_ID"
    )
    origin_tissue = CANCER_TO_TISSUE.get(cancer_code)
    origin_col = f"nTPM_{origin_tissue}" if origin_tissue else None
    testis_col = "nTPM_testis" if "nTPM_testis" in ref_by_id.columns else None

    # Vital-tissue columns minus testis (testis is treated separately as
    # the "expected" CTA baseline, not a toxicity concern).
    vital_tissue_cols = []
    for tissue in ESSENTIAL_TISSUES:
        for col in _ESSENTIAL_TISSUE_COLS.get(tissue, []):
            if col in ref_by_id.columns and col != testis_col:
                vital_tissue_cols.append((tissue, col))

    rows = []
    for gid, sym in cta_id_to_name.items():
        gid_clean = _strip_ensembl_version(str(gid))
        s_tpm = sample_tpm.get(gid_clean, 0.0)
        if s_tpm < min_sample_tpm:
            continue
        if gid_clean not in ref_by_id.index:
            continue
        ref_row = ref_by_id.loc[gid_clean]
        cohort_val = float(ref_row[cancer_col])
        origin_val = (
            float(ref_row[origin_col]) if origin_col and origin_col in ref_row else None
        )
        testis_val = float(ref_row[testis_col]) if testis_col else None

        worst_vital = None
        for tissue, col in vital_tissue_cols:
            val = float(ref_row[col]) if col in ref_row else 0.0
            if worst_vital is None or val > worst_vital[1]:
                worst_vital = (tissue, val, col)

        rows.append(
            {
                "symbol": sym,
                "sample": s_tpm,
                "cohort": cohort_val,
                "origin": origin_val,
                "testis": testis_val,
                "worst_vital": worst_vital,
            }
        )

    if not rows:
        print(f"No CTAs with sample TPM ≥ {min_sample_tpm} found for {cancer_code}.")
        return None

    # Rank by sample TPM descending; truncate to top_k.
    rows.sort(key=lambda r: -r["sample"])
    rows = rows[:top_k]
    n = len(rows)
    if figsize is None:
        figsize = (13, 0.34 * n + 2.2)

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(n)

    toxicity_threshold = 10.0

    for i, r in enumerate(rows):
        # Sample — large blue circle, TPM labeled inline to the right of
        # the marker.
        x_sample = max(r["sample"], 0.05)
        ax.scatter(
            [x_sample],
            [i],
            s=120,
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.8,
            zorder=6,
        )
        ax.text(
            x_sample * 1.2,
            i,
            f"{r['sample']:.0f}",
            fontsize=7.5,
            va="center",
            color="#1f77b4",
            fontweight="bold",
            zorder=7,
        )

        # Cohort median — orange diamond.
        x_cohort = max(r["cohort"], 0.05)
        ax.scatter(
            [x_cohort],
            [i],
            s=70,
            marker="D",
            color="#ff7f0e",
            edgecolor="#704000",
            linewidth=0.6,
            zorder=5,
            alpha=0.95,
        )

        # Tissue-of-origin normal — green square.
        if r["origin"] is not None:
            x_origin = max(r["origin"], 0.05)
            ax.scatter(
                [x_origin],
                [i],
                s=65,
                marker="s",
                color="#2ca02c",
                edgecolor="#175417",
                linewidth=0.6,
                zorder=5,
                alpha=0.95,
            )

        # Testis — purple triangle (CTAs' expected reference).
        if r["testis"] is not None:
            x_testis = max(r["testis"], 0.05)
            ax.scatter(
                [x_testis],
                [i],
                s=65,
                marker="^",
                color="#9467bd",
                edgecolor="#4a285e",
                linewidth=0.6,
                zorder=5,
                alpha=0.95,
            )

        # Worst non-testis vital tissue — only prominent if it crosses
        # the toxicity threshold; otherwise faint gray X.
        if r["worst_vital"] is not None:
            tissue, val, _col = r["worst_vital"]
            x_vital = max(val, 0.05)
            if val >= toxicity_threshold:
                ax.scatter(
                    [x_vital],
                    [i],
                    s=95,
                    marker="X",
                    color="#d62728",
                    edgecolor="#7a0000",
                    linewidth=0.9,
                    zorder=5.5,
                )
                ax.text(
                    x_vital * 1.15,
                    i,
                    f"  {tissue} {val:.0f}",
                    fontsize=7,
                    va="center",
                    color="#d62728",
                    zorder=6,
                )
            else:
                ax.scatter(
                    [x_vital],
                    [i],
                    s=45,
                    marker="X",
                    color="#999999",
                    edgecolor="none",
                    zorder=4,
                    alpha=0.7,
                )

    ax.axvline(
        toxicity_threshold,
        linestyle="--",
        color="#d62728",
        linewidth=0.8,
        alpha=0.45,
        zorder=1,
    )

    ax.set_xscale("log")
    ax.set_xlim(0.05, max(1000.0, max(r["sample"] for r in rows) * 3))
    ax.set_yticks(y)
    ax.set_yticklabels([r["symbol"] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("TPM (log scale)")
    origin_display = (
        origin_tissue.replace("_", " ") if origin_tissue else "origin tissue"
    )
    ax.set_title(
        f"CTA detail — {cancer_code} (sample vs cohort / {origin_display} / testis / worst vital tissue)",
        fontweight="bold",
    )
    try:
        from .plot_reference_lines import add_p90_reference_line

        add_p90_reference_line(ax, sample_tpm, orientation="vertical")
    except Exception:
        pass
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = [
        Line2D(
            [],
            [],
            marker="o",
            color="#1f77b4",
            markeredgecolor="black",
            markersize=10,
            linestyle="",
            label="sample",
        ),
        Line2D(
            [],
            [],
            marker="D",
            color="#ff7f0e",
            markeredgecolor="#704000",
            markersize=8,
            linestyle="",
            label=f"{cancer_code} cohort",
        ),
    ]
    if origin_tissue:
        legend.append(
            Line2D(
                [],
                [],
                marker="s",
                color="#2ca02c",
                markeredgecolor="#175417",
                markersize=8,
                linestyle="",
                label=f"normal {origin_display}",
            )
        )
    legend.extend(
        [
            Line2D(
                [],
                [],
                marker="^",
                color="#9467bd",
                markeredgecolor="#4a285e",
                markersize=8,
                linestyle="",
                label="testis",
            ),
            Line2D(
                [],
                [],
                marker="X",
                color="#d62728",
                markeredgecolor="#7a0000",
                markersize=9,
                linestyle="",
                label=f"vital tissue > {toxicity_threshold:g} TPM",
            ),
            Line2D(
                [],
                [],
                marker="X",
                color="#999999",
                markersize=6,
                linestyle="",
                label="worst vital tissue (below threshold)",
            ),
        ]
    )
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(len(legend), 3),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig


# ── Therapy-pathway state plot (#136) ───────────────────────────────────


_AXIS_STATE_COLOR = {
    "up": "#2ca25f",  # active
    "down": "#d6604d",  # suppressed
    "mixed": "#c9a227",  # mixed / indeterminate
    "indeterminate": "#8888a0",
}


def _format_axis_label(therapy_class: str) -> str:
    """Human labels for the therapy-response axis names — the raw
    ``therapy_class`` strings (e.g. ``AR_signaling``) read fine in a
    TSV but are noisy in a figure title."""
    mapping = {
        "AR_signaling": "AR signaling",
        "ER_signaling": "ER signaling",
        "HER2_signaling": "HER2 signaling",
        "MAPK_EGFR": "MAPK / EGFR",
        "MAPK_EGFR_signaling": "MAPK / ERK activity",
        "NE_differentiation": "NE differentiation",
        "EMT": "EMT",
        "hypoxia": "Hypoxia",
        "IFN_response": "IFN response",
    }
    return mapping.get(therapy_class, therapy_class.replace("_", " "))


def plot_therapy_pathway_state(
    therapy_response_scores,
    cancer_code: str = "",
    disease_state_caption: str = "",
    save_to_filename=None,
    save_dpi: int = 300,
    figsize=None,
):
    """One-figure therapy-pathway state readout (#136).

    Renders the disease-state narrative visually: one row per therapy-
    response axis (AR / NE / EMT / hypoxia / IFN / ER / HER2 / MAPK-ERK
    where applicable), dumbbell showing up-panel vs down-panel fold-vs-
    cohort, with state label + color. A caption underneath restates
    the disease-state sentence so the figure is self-contained for
    tumor-board review.

    Parameters
    ----------
    therapy_response_scores : dict
        ``analysis["therapy_response_scores"]`` — per-axis
        :class:`pirlygenes.therapy_response.TherapyAxisScore`.
    cancer_code : str
        Displayed in the title for context.
    disease_state_caption : str
        Text to display under the plot (typically the output of
        :func:`compose_disease_state_narrative`). Empty-caption calls
        just skip the caption row.
    save_to_filename : str, optional
        Write PNG here (and print the saved-path line for the CLI
        progress log).
    """
    # Materialize and order axes: active / suppressed axes first (more
    # informative), baseline-ish axes last. Within each bucket keep the
    # input order so cancer-type-specific ordering (AR first for PRAD,
    # ER first for BRCA etc.) is preserved.
    items = []
    for cls, score in (therapy_response_scores or {}).items():
        state = getattr(score, "state", "indeterminate")
        up_fold = getattr(score, "up_geomean_fold", None)
        down_fold = getattr(score, "down_geomean_fold", None)
        if up_fold is None and down_fold is None:
            continue
        items.append(
            {
                "cls": cls,
                "label": _format_axis_label(cls),
                "state": state,
                "up_fold": up_fold,
                "down_fold": down_fold,
                "up_n": getattr(score, "up_genes_measured", 0),
                "down_n": getattr(score, "down_genes_measured", 0),
                "message": getattr(score, "message", "") or "",
            }
        )
    if not items:
        return None

    bucket_priority = {"up": 0, "down": 0, "mixed": 1, "indeterminate": 2}
    items.sort(key=lambda r: bucket_priority.get(r["state"], 2))

    plot_rows = []
    for axis_idx, item in enumerate(items):
        panels = []
        if item["up_fold"] is not None:
            panels.append(
                {
                    "panel": "up",
                    "fold": item["up_fold"],
                    "n": item["up_n"],
                    "marker": "o",
                    "arrow": "\u2191",
                    "panel_label": "expected-up genes",
                    "legend_label": "genes expected up when pathway active",
                }
            )
        if item["down_fold"] is not None:
            panels.append(
                {
                    "panel": "down",
                    "fold": item["down_fold"],
                    "n": item["down_n"],
                    "marker": "s",
                    "arrow": "\u2193",
                    "panel_label": "expected-down genes",
                    "legend_label": "genes expected down when pathway active",
                }
            )
        for panel_idx, panel in enumerate(panels):
            plot_rows.append(
                {
                    **item,
                    **panel,
                    "axis_index": axis_idx,
                    "axis_first_row": panel_idx == 0,
                    "axis_last_row": panel_idx == len(panels) - 1,
                    "axis_panel_count": len(panels),
                }
            )

    n_rows = len(plot_rows)
    if figsize is None:
        # Reserve vertical room for the caption wrap. Width accommodates
        # fold labels without the legend overflowing. Up- and down-panel
        # folds are separate rows because they are opposite evidence streams.
        figsize = (12, max(4.0, 0.55 * n_rows + 2.8))

    fig = plt.figure(figsize=figsize)
    # Top area for the dumbbells, bottom for the narrative caption.
    if disease_state_caption:
        ax = fig.add_axes([0.07, 0.30, 0.88, 0.60])
        ax_caption = fig.add_axes([0.07, 0.02, 0.88, 0.22])
        ax_caption.axis("off")
    else:
        ax = fig.add_axes([0.07, 0.10, 0.88, 0.80])
        ax_caption = None

    # --- Dumbbell plot ---
    y_positions = np.arange(n_rows)
    legend_seen = set()
    for i, row in enumerate(plot_rows):
        color = _AXIS_STATE_COLOR.get(row["state"], "#555555")

        fold = row["fold"]
        if fold is None or fold <= 0:
            continue
        # Anchor each panel to baseline. Do not connect up- and down-panel
        # folds: for suppressed pathways they move in opposite directions and
        # connecting them implies a single continuum that is not biological.
        ax.plot(
            [min(1.0, fold), max(1.0, fold)],
            [i, i],
            color=color,
            linewidth=3,
            alpha=0.24,
            solid_capstyle="round",
            zorder=2,
        )
        legend_label = None
        if row["panel"] not in legend_seen:
            legend_label = row["legend_label"]
            legend_seen.add(row["panel"])
        ax.plot(
            [fold],
            [i],
            marker=row["marker"],
            markersize=10.5,
            color=color,
            markeredgecolor="white",
            markeredgewidth=1.4,
            linestyle="",
            zorder=3,
            label=legend_label,
        )
        text_x = fold * (1.08 if fold >= 1.0 else 0.92)
        ha = "left" if fold >= 1.0 else "right"
        ax.text(
            text_x,
            i,
            f"{row['arrow']} {fold:.2f}\u00d7",
            ha=ha,
            va="center",
            fontsize=8,
            color=color,
            fontweight="bold",
        )

        if row["axis_last_row"] and i < n_rows - 1:
            ax.axhline(i + 0.5, color="#eeeeee", linewidth=0.8, zorder=0)

    ax.axvline(1.0, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)
    ax.set_xscale("log")
    x_vals = []
    for r in plot_rows:
        f = r["fold"]
        if f is not None and f > 0:
            x_vals.append(f)
    if x_vals:
        lo = min(0.1, min(x_vals) * 0.7)
        hi = max(10.0, max(x_vals) * 1.4)
        ax.set_xlim(lo, hi)
    ax.set_xlabel(
        "Fold vs TCGA cohort median  (log scale; 1.0 = baseline)", fontsize=10
    )

    # Y-axis: label + state tag (color-coded)
    ax.set_yticks(y_positions)
    labels = []
    for row in plot_rows:
        state_tag = {
            "up": "active",
            "down": "suppressed",
            "mixed": "mixed",
            "indeterminate": "near baseline",
        }.get(row["state"], row["state"])
        n_info = f" ({row['n']})" if row["n"] else ""
        panel_text = f"{row['panel_label']}{n_info}"
        if row["axis_first_row"]:
            labels.append(f"{row['label']}  \u2014  {state_tag}\n  {panel_text}")
        else:
            labels.append(f"  \u21b3 {panel_text}")
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    title = "Therapy-pathway state"
    if cancer_code:
        title += f" \u2014 {cancer_code}"
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.legend(
        loc="lower right", fontsize=8, frameon=False, markerscale=0.9, handletextpad=0.4
    )

    # --- Caption ---
    if ax_caption is not None and disease_state_caption:
        # Single wrapped paragraph underneath. Narrow to ~130 chars per
        # line so wrapping matches the figure width.
        import textwrap

        wrapped = "\n".join(
            textwrap.wrap(
                disease_state_caption,
                width=130,
                break_long_words=False,
                break_on_hyphens=True,
            )
        )
        ax_caption.text(
            0.0,
            1.0,
            wrapped,
            ha="left",
            va="top",
            fontsize=9,
            color="#222222",
            wrap=True,
        )

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig
