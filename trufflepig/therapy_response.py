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

"""Therapy-response gene signatures (issue #57).

Curated per-axis panels that let a single RNA-seq sample be scored for
the signaling state of each axis (AR, ER, HER2, MAPK/ERK, NE
differentiation, EMT, hypoxia, IFN response) and — crucially —
annotated against the cancer-type cohort median so suppressed states
surface as divergence below cohort rather than just raw low expression.

Motivation: expression alone carries strong signal about what therapy
a tumor has been exposed to. A PRAD sample with KLK2/KLK3/TMPRSS2/NKX3-1
suppressed and FOLH1 (PSMA) elevated + NE markers rising is almost
certainly ADT-treated, possibly with early lineage plasticity. The
scoring module surfaces that chain explicitly, so readers see

    **AR signaling suppressed** (z=-2.1 vs cohort; KLK3 0.12x cohort,
    FOLH1 4.3x cohort, NKX3-1 0.08x cohort) — pattern consistent with
    ADT exposure

rather than having to reconstruct it from per-gene TPM tables.

Attribution flow placement: this sits after RNA prep and preservation, tumor
purity / coarse composition, and subtype/background refinement. Therapy-response
explains *why* a tumor gene is high or low independent of TME
mis-attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Optional

from pirlygenes.load_dataset import get_data


MAPK_ACTIVITY_AXIS = "MAPK_EGFR_signaling"
MAPK_ACTIVITY_LABEL = "MAPK / ERK activity"
MAPK_ACTIVITY_SCORE_NAME = "MPAS-like downstream transcriptional score"
MAPK_ACTIVITY_GENES = frozenset(
    {
        "PHLDA1",
        "SPRY2",
        "SPRY4",
        "DUSP4",
        "DUSP6",
        "CCND1",
        "EPHA2",
        "EPHA4",
        "ETV4",
        "ETV5",
    }
)

_MAPK_DRIVER_SOURCE_GENES = frozenset(
    {
        "ABL1",
        "ALK",
        "ARAF",
        "BRAF",
        "EGFR",
        "ERBB2",
        "ERBB3",
        "ERBB4",
        "FGFR1",
        "FGFR2",
        "FGFR3",
        "FGFR4",
        "FLT3",
        "HRAS",
        "KIT",
        "KRAS",
        "MAP2K1",
        "MAP2K2",
        "MET",
        "NF1",
        "NTRK1",
        "NTRK2",
        "NTRK3",
        "NRAS",
        "PDGFRA",
        "PDGFRB",
        "RAF1",
        "RET",
        "ROS1",
    }
)

_MAPK_RNA_SOURCE_GENES = frozenset(
    {
        "ALK",
        "EGFR",
        "ERBB2",
        "ERBB3",
        "ERBB4",
        "FGFR1",
        "FGFR2",
        "FGFR3",
        "FGFR4",
        "KIT",
        "MET",
        "NTRK1",
        "NTRK2",
        "NTRK3",
        "PDGFRA",
        "PDGFRB",
        "RET",
        "ROS1",
    }
)

_MAPK_UNRESOLVED_SOURCE_PROMPTS = (
    "RAS/RAF/MEK hotspot mutation or NF1 loss",
    "RTK amplification or activating kinase-domain duplication",
    "ALK/RET/NTRK/ROS1/FGFR/RAF-family fusion",
    "ligand-driven or microenvironmental MAPK activation",
)


# ── Data access ───────────────────────────────────────────────────────────


def load_therapy_signatures():
    """Return ``{therapy_class: {"up": [Gene, ...], "down": [Gene, ...]}}``.

    Each ``Gene`` is a ``dict`` carrying symbol, ensembl_gene_id,
    mechanism, cancer_context (semicolon-separated TCGA codes or
    ``pan_cancer``), strength (``canonical``/``supported``/``emerging``),
    and refs string.  Callers use the mechanism / refs fields to
    annotate therapy targets in the report.
    """
    df = get_data("therapy-response-signatures")
    out: dict[str, dict[str, list[dict]]] = {}
    for _, row in df.iterrows():
        cls = str(row["therapy_class"])
        direction = str(row["direction"]).strip()
        rec = {
            "symbol": str(row["symbol"]),
            "ensembl_gene_id": str(row["ensembl_gene_id"]),
            "mechanism": str(row.get("mechanism", "")),
            "cancer_context": str(row.get("cancer_context", "")),
            "strength": str(row.get("strength", "")),
            "refs": str(row.get("refs", "")),
        }
        cls_entry = out.setdefault(cls, {"up": [], "down": []})
        cls_entry.setdefault(direction, []).append(rec)
    return out


def _sigs_for_cancer(all_sigs, cancer_code):
    """Return only the therapy classes where at least one gene applies
    to ``cancer_code`` (either ``pan_cancer`` or an explicit match).
    """
    result = {}
    for cls, directions in all_sigs.items():
        kept_up = [g for g in directions.get("up", []) if _applies(g, cancer_code)]
        kept_down = [g for g in directions.get("down", []) if _applies(g, cancer_code)]
        if kept_up or kept_down:
            result[cls] = {"up": kept_up, "down": kept_down}
    return result


def _applies(gene_rec, cancer_code) -> bool:
    ctx = str(gene_rec.get("cancer_context") or "").strip()
    if not ctx or ctx == "pan_cancer":
        return True
    codes = {part.strip().upper() for part in ctx.split(";") if part.strip()}
    return str(cancer_code).upper() in codes


# ── Per-axis scoring ──────────────────────────────────────────────────────


@dataclass
class TherapyAxisScore:
    """Single-sample signaling-state readout for one therapy axis."""

    therapy_class: str
    state: str  # "up" / "down" / "mixed" / "indeterminate"
    up_geomean_fold: Optional[float] = None
    down_geomean_fold: Optional[float] = None
    up_genes_measured: int = 0
    down_genes_measured: int = 0
    per_gene: list[dict] = field(default_factory=list)
    message: str = ""


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _record_value(record: Any, key: str, default: Any = "") -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _clean_symbol(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text or text == "NAN":
        return ""
    return text


def _short_source_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.rsplit("/", 1)[-1]


def _support_genes_for_score(score: TherapyAxisScore, *, max_genes: int = 5) -> list[str]:
    rows = [
        row
        for row in (getattr(score, "per_gene", None) or [])
        if str(row.get("direction") or "") == "up"
        and _as_float(row.get("fold_vs_cohort")) is not None
    ]
    rows.sort(key=lambda row: _as_float(row.get("fold_vs_cohort")) or 0.0, reverse=True)
    out = []
    for row in rows:
        symbol = str(row.get("symbol") or "").strip()
        fold = _as_float(row.get("fold_vs_cohort"))
        if not symbol or fold is None:
            continue
        out.append(f"{symbol} {fold:.1f}x")
        if len(out) >= max_genes:
            break
    return out


def _candidate_source_from_alteration(record: Any) -> dict[str, Any] | None:
    gene = _clean_symbol(_record_value(record, "gene"))
    if gene not in _MAPK_DRIVER_SOURCE_GENES:
        return None
    alteration_type = str(_record_value(record, "alteration_type") or "").strip()
    alteration = str(_record_value(record, "alteration") or "").strip()
    raw_name = str(_record_value(record, "raw_name") or "").strip()
    evidence_text = " ".join([alteration_type, alteration, raw_name]).lower()
    if gene == "NF1" and any(
        token in evidence_text for token in ("loss", "delet", "lof", "truncat")
    ):
        mechanism = "negative-regulator loss can raise RAS/MAPK signaling"
    elif any(
        token in evidence_text
        for token in (
            "kdd",
            "kinase domain duplication",
            "activating",
            "hotspot",
            "amplification",
            "amplified",
            "fusion",
            "rearrang",
            "itd",
            "internal tandem duplication",
        )
    ):
        mechanism = "supplied activating MAPK-source alteration"
    elif alteration_type in {
        "kdd",
        "amplification",
        "fusion",
        "internal_tandem_duplication",
        "mutation",
        "unknown",
    }:
        mechanism = "supplied alteration in a MAPK-source gene"
    else:
        return None

    detail = alteration or raw_name or alteration_type or "alteration"
    source_path = _short_source_path(_record_value(record, "source_path"))
    if str(detail).strip().upper().startswith(gene):
        label = f"supplied {detail}".strip()
    else:
        label = f"supplied {gene} {detail}".strip()
    if source_path:
        label += f" ({source_path})"
    return {
        "kind": "supplied_alteration",
        "gene": gene,
        "label": label,
        "mechanism": mechanism,
        "confidence": str(_record_value(record, "confidence") or "supplied").strip()
        or "supplied",
    }


def _candidate_source_from_fusion(record: Any) -> dict[str, Any] | None:
    gene_a = _clean_symbol(_record_value(record, "gene_a"))
    gene_b = _clean_symbol(_record_value(record, "gene_b"))
    genes = [gene for gene in (gene_a, gene_b) if gene in _MAPK_DRIVER_SOURCE_GENES]
    if not genes:
        return None
    pair = str(_record_value(record, "pair") or "").strip()
    if not pair and gene_a and gene_b:
        pair = f"{gene_a}--{gene_b}"
    source_path = _short_source_path(_record_value(record, "source_path"))
    label = f"supplied fusion involving {', '.join(genes)}"
    if pair:
        label += f" ({pair})"
    if source_path:
        label += f" from {source_path}"
    return {
        "kind": "supplied_fusion",
        "gene": genes[0],
        "label": label,
        "mechanism": "kinase or MAPK-pathway fusion can drive downstream ERK output",
        "confidence": str(_record_value(record, "confidence") or "supplied").strip()
        or "supplied",
    }


def _candidate_sources_from_rna(ranges_df, *, max_sources: int = 4) -> list[dict[str, Any]]:
    if ranges_df is None or not hasattr(ranges_df, "columns") or "symbol" not in ranges_df.columns:
        return []
    value_col = "attr_tumor_tpm" if "attr_tumor_tpm" in ranges_df.columns else "observed_tpm"
    if value_col not in ranges_df.columns:
        return []

    rows: list[dict[str, Any]] = []
    for _, row in ranges_df.iterrows():
        symbol = _clean_symbol(row.get("symbol"))
        if symbol not in _MAPK_RNA_SOURCE_GENES:
            continue
        tpm = _as_float(row.get(value_col))
        observed = _as_float(row.get("observed_tpm"))
        fold = _as_float(row.get("pct_cancer_median"))
        percentile = _as_float(row.get("tcga_percentile"))
        if tpm is None or tpm < 10.0:
            continue
        if fold is None and percentile is None:
            continue
        if (fold is not None and fold < 2.0) and (
            percentile is None or percentile < 0.90
        ):
            continue
        rows.append(
            {
                "symbol": symbol,
                "tpm": tpm,
                "observed_tpm": observed,
                "fold": fold,
                "percentile": percentile,
            }
        )

    rows.sort(
        key=lambda row: (
            row["fold"] if row["fold"] is not None else 0.0,
            row["tpm"],
        ),
        reverse=True,
    )

    out = []
    for row in rows[:max_sources]:
        fold_clause = (
            f"; {row['fold']:.1f}x cancer median" if row["fold"] is not None else ""
        )
        percentile_clause = (
            f"; reference percentile {row['percentile']:.0%}"
            if row["percentile"] is not None
            else ""
        )
        out.append(
            {
                "kind": "rna_high_source_gene",
                "gene": row["symbol"],
                "label": (
                    f"RNA-high {row['symbol']} ({row['tpm']:.0f} tumor-source TPM"
                    f"{fold_clause}{percentile_clause})"
                ),
                "mechanism": "high receptor/kinase RNA can be compatible with RTK-driven MAPK signaling but is not alteration proof",
                "confidence": "expression-only",
            }
        )
    return out


def infer_mapk_activity_sources(
    analysis: dict[str, Any],
    *,
    ranges_df=None,
    min_activity_fold: float = 2.0,
) -> list[dict[str, Any]]:
    """Attach source hypotheses for high pan-cancer MAPK/ERK activity.

    The MAPK score is a downstream transcriptional state. It can support
    pathway activation, but it cannot by itself identify whether the driver is
    EGFR KDD, RAS/RAF mutation, NF1 loss, RTK amplification, a kinase fusion, or
    ligand/background signaling. This helper keeps that uncertainty explicit.
    """
    if not isinstance(analysis, dict):
        return []
    score = (analysis.get("therapy_response_scores") or {}).get(MAPK_ACTIVITY_AXIS)
    if score is None or getattr(score, "state", None) != "up":
        return []
    fold = _as_float(getattr(score, "up_geomean_fold", None))
    if fold is None or fold < min_activity_fold:
        return []

    candidate_sources: list[dict[str, Any]] = []
    for record in analysis.get("alteration_records") or []:
        source = _candidate_source_from_alteration(record)
        if source:
            candidate_sources.append(source)
    for record in analysis.get("fusion_records") or []:
        source = _candidate_source_from_fusion(record)
        if source:
            candidate_sources.append(source)
    candidate_sources.extend(_candidate_sources_from_rna(ranges_df))

    deduped_sources = []
    seen = set()
    for source in candidate_sources:
        key = (source.get("kind"), source.get("gene"), source.get("label"))
        if key in seen:
            continue
        seen.add(key)
        deduped_sources.append(source)

    unresolved = list(_MAPK_UNRESOLVED_SOURCE_PROMPTS)
    if deduped_sources:
        source_genes = {str(source.get("gene") or "") for source in deduped_sources}
        if source_genes & {"EGFR", "ERBB2", "ERBB3", "ERBB4", "MET", "FGFR1", "FGFR2", "FGFR3", "FGFR4"}:
            unresolved = [
                prompt
                for prompt in unresolved
                if "RTK amplification" not in prompt
            ] + ["other RTK/RAS/RAF/NF1 or fusion drivers still possible"]

    return [
        {
            "axis": MAPK_ACTIVITY_AXIS,
            "label": MAPK_ACTIVITY_LABEL,
            "score_name": MAPK_ACTIVITY_SCORE_NAME,
            "state": "up",
            "up_geomean_fold": round(fold, 3),
            "support_genes": _support_genes_for_score(score),
            "candidate_sources": deduped_sources[:6],
            "unresolved_sources": unresolved,
            "caveat": (
                "MAPK/ERK RNA activity is a convergent downstream readout; it "
                "supports active biology but is not source-specific."
            ),
        }
    ]


def _cohort_median_for_symbol(symbol, cancer_code, ref_flat):
    """Return the TCGA cohort median TPM for a symbol in a cancer
    type, or None if the cohort column is missing or the symbol is
    absent from the reference universe."""
    col = f"{cancer_code}_TPM"
    if col not in ref_flat.columns:
        return None
    sub = ref_flat[ref_flat["Symbol"] == symbol]
    if sub.empty:
        return None
    val = float(sub.iloc[0][col])
    if val != val:  # NaN check without numpy
        return None
    return val


def score_therapy_signatures(
    sample_tpm_by_symbol,
    cancer_type,
):
    """Score each applicable therapy axis for a sample.

    For every axis relevant to ``cancer_type``:

    - For each ``up`` gene, compute sample/cohort fold change
      (1.0 ⇒ equals cohort; >1 ⇒ elevated; <1 ⇒ suppressed).
    - Geomean the fold changes across ``up`` genes → ``up_geomean_fold``.
    - Same for ``down`` genes → ``down_geomean_fold``.
    - State is ``up``, ``down``, ``mixed`` or ``indeterminate`` based on
      which sides diverge ≥2× from cohort.

    Returns ``{therapy_class: TherapyAxisScore}``.

    ``sample_tpm_by_symbol`` is the dict produced by
    ``tumor_purity._build_sample_tpm_by_symbol`` or the
    ``sample_context._build_tpm_by_symbol`` fallback.
    """
    import numpy as np

    from trufflepig.reference import pan_cancer_expression
    all_sigs = load_therapy_signatures()
    applicable = _sigs_for_cancer(all_sigs, cancer_type)
    if not applicable:
        return {}

    ref_flat = pan_cancer_expression(technical_rna_normalize=True).drop_duplicates(
        subset="Symbol"
    )

    out = {}
    for cls, directions in applicable.items():
        per_gene = []
        up_folds = []
        down_folds = []
        for direction, genes in (
            ("up", directions["up"]),
            ("down", directions["down"]),
        ):
            for g in genes:
                sym = g["symbol"]
                sample_tpm = sample_tpm_by_symbol.get(sym)
                cohort_med = _cohort_median_for_symbol(sym, cancer_type, ref_flat)
                if sample_tpm is None or cohort_med is None:
                    continue
                # Cohort-referenced fold change. Pseudocount of 0.5 TPM
                # keeps the ratio well-defined when either side is zero.
                fold = (float(sample_tpm) + 0.5) / (float(cohort_med) + 0.5)
                per_gene.append(
                    {
                        "symbol": sym,
                        "direction": direction,
                        "sample_tpm": round(float(sample_tpm), 2),
                        "cohort_median": round(float(cohort_med), 2),
                        "fold_vs_cohort": round(fold, 3),
                        "mechanism": g["mechanism"],
                        "strength": g["strength"],
                    }
                )
                if direction == "up":
                    up_folds.append(fold)
                else:
                    down_folds.append(fold)

        up_geo = float(np.exp(np.mean(np.log(up_folds)))) if up_folds else None
        down_geo = float(np.exp(np.mean(np.log(down_folds)))) if down_folds else None

        # Interpretive state. Thresholds:
        #   >2× cohort on the "up" side  → axis is *active*
        #   <0.5× cohort on the "up" side → axis is *suppressed*
        # (the "down" genes are already expected to move opposite to
        # signaling, so they confirm rather than drive the call.)
        if up_geo is None and down_geo is None:
            state = "indeterminate"
            message = "No cohort-resolvable genes on either side"
        elif (
            up_geo is not None
            and up_geo >= 2.0
            and (down_geo is None or down_geo < 1.5)
        ):
            state = "up"
            message = f"Active signaling: up-panel geomean {up_geo:.2f}× cohort" + (
                f", down-panel {down_geo:.2f}× cohort" if down_geo else ""
            )
        elif (
            up_geo is not None
            and up_geo <= 0.5
            and (down_geo is None or down_geo >= 1.5)
        ):
            state = "down"
            message = f"Suppressed: up-panel geomean {up_geo:.2f}× cohort" + (
                f", down-panel {down_geo:.2f}× cohort (therapy-response genes elevated)"
                if down_geo and down_geo >= 1.5
                else ""
            )
        elif (
            up_geo is not None
            and up_geo > 1.5
            and down_geo is not None
            and down_geo > 1.5
        ):
            state = "mixed"
            message = (
                f"Mixed: up-panel {up_geo:.2f}× and down-panel {down_geo:.2f}× "
                "both elevated"
            )
        elif up_geo is not None and 0.5 < up_geo < 2.0:
            state = "indeterminate"
            message = f"Near-cohort baseline: up-panel {up_geo:.2f}× cohort" + (
                f", down-panel {down_geo:.2f}× cohort" if down_geo else ""
            )
        else:
            state = "indeterminate"
            message = "Mixed / weak signal"

        out[cls] = TherapyAxisScore(
            therapy_class=cls,
            state=state,
            up_geomean_fold=round(up_geo, 3) if up_geo is not None else None,
            down_geomean_fold=round(down_geo, 3) if down_geo is not None else None,
            up_genes_measured=len(up_folds),
            down_genes_measured=len(down_folds),
            per_gene=per_gene,
            message=message,
        )
    return out


def symbol_therapy_annotations(all_sigs, cancer_code):
    """Return ``{symbol: [mechanism_string, ...]}`` for every gene
    applicable to ``cancer_code``, used to annotate therapy targets in
    the report so the reader sees *why* a gene's observed expression
    might be where it is.
    """
    out: dict[str, list[str]] = {}
    for cls, directions in all_sigs.items():
        for direction, genes in directions.items():
            for g in genes:
                if not _applies(g, cancer_code):
                    continue
                ann = f"{cls.replace('_', ' ')} {direction}: {g['mechanism']}"
                out.setdefault(g["symbol"], []).append(ann)
    return out
