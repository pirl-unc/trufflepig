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

"""Sample-context inference — library prep, preservation, degradation.

First step of the unified attribution flow (see README "Attribution
flow"). Runs before cancer-type inference and produces a ``SampleContext``
that every downstream step consumes as a **base-layer of expression
expectations**: which genes are expected to be over- or under-represented
for artifactual reasons, and how to compensate.

The detection uses signals that are tissue-independent, so this module
can run on a raw expression table without knowing what cancer or normal
tissue the sample is from:

- **Library prep** — replication-dependent histone fraction and MT
  ribosomal-RNA presence. Histone mRNAs and MT rRNAs are *not*
  polyadenylated and therefore disappear under poly-A capture but
  survive in total-RNA / ribosomal-depletion libraries.
- **Preservation (FFPE vs fresh)** — long/short matched-gene-pair ratio
  (from ``data/degradation-gene-pairs.csv``). Long transcripts degrade
  faster than short ones in FFPE.
- **Missing-MT** — complete absence of mitochondrial genes is an
  artifact (pipeline filter, or exome-capture) rather than biology;
  degrades confidence in all MT-based signals.

What downstream steps use the context for:

- Marker selection: downweight long-transcript markers under FFPE; skip
  the extended-housekeeping exclusion universe from tumor-specific
  marker computations.
- Purity estimator: widen confidence intervals proportional to
  degradation severity.
- Decomposition: adjust fit weights for sample_context-biased genes.
- Tumor-value adjustment: subtract TME-explainable expression from
  therapy-target columns (prevents FN1/COL1A1 low-purity inflation).
- Reporting: show the context chain so users see *why* a number was
  adjusted, not just the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .common import without_dataframe_attrs
from trufflepig.expression_qc import classify_gene_qc, summarize_qc_class_shares
from typing import Optional


# ── Reference gene patterns ──────────────────────────────────────────────
#
# Replication-dependent histones (H1-*, H2AC*, H2BC*, H3C*, H4C*) are the
# canonical family of non-polyadenylated mRNAs. Their fraction collapses
# under poly-A selection and remains meaningful under total-RNA /
# ribo-depletion, making them the primary library-prep indicator when
# combined with MT rRNAs.
#
# We match by symbol prefix to cover both HGNC 2020+ names (H2AC1,
# H2BC1, H3C1, H4C1) and the older HIST* synonyms that some annotations
# still emit (HIST1H2BC, etc.). A sample's histone signal is then
# ``sum(TPM for histone symbols) / sum(TPM for all expressed symbols)``.
_HISTONE_SYMBOL_PREFIXES: tuple[str, ...] = (
    "H1-",
    "H2AC",
    "H2BC",
    "H3C",
    "H4C",
    "HIST1H",
    "HIST2H",
    "HIST3H",
    "HIST4H",
)
# MT ribosomal RNAs — non-polyadenylated. Present in total / ribo-dep,
# absent in poly-A and typically in exome capture. Combined with the
# histone fraction, this disambiguates all four common libraries.
_MT_RRNA_SYMBOLS: frozenset[str] = frozenset({"MT-RNR1", "MT-RNR2"})

# MT protein-coding mRNAs — polyadenylated. Present in every library
# that retains mitochondrial transcripts (total, ribo-dep, poly-A).
# Absence alongside absence of MT rRNAs indicates either exome capture
# (no MT coverage) or an upstream filter that stripped MT contigs.
_MT_MRNA_SYMBOLS: frozenset[str] = frozenset(
    {
        "MT-CO1",
        "MT-CO2",
        "MT-CO3",
        "MT-ND1",
        "MT-ND2",
        "MT-ND3",
        "MT-ND4",
        "MT-ND4L",
        "MT-ND5",
        "MT-ND6",
        "MT-CYB",
        "MT-ATP6",
        "MT-ATP8",
    }
)


def library_prep_display_label(prep: str | None, *, title_case: bool = False) -> str:
    """Human label for the inferred RNA library-prep class."""
    text = str(prep or "unknown").strip()
    label = {
        "poly_a": "poly-A capture",
        "ribo_depleted": "ribosomal depletion",
        "total_rna": "total RNA",
        "exome_capture": "RNA hybrid-capture / RNA-exome capture",
        "unknown": "unknown library prep",
    }.get(text, text.replace("_", " "))
    if title_case:
        return label[:1].upper() + label[1:]
    return label


def library_prep_clause(prep: str | None, *, title_case: bool = False) -> str:
    """Human phrase for report prose, including ``library`` when helpful."""
    label = library_prep_display_label(prep, title_case=title_case)
    low = label.lower()
    if "library" in low or "prep" in low:
        return label
    return f"{label} library"


def heuristic_support_label(value: float | int | None) -> str:
    """Reader-facing label for heuristic support scores.

    These are not assay-measured percentages. They summarize how strongly a
    rule set supports an inferred state, so report prose should avoid calling
    them measured probabilities.
    """

    try:
        return f"support {float(value or 0.0):.0%}"
    except Exception:
        return "support unknown"


def length_pair_display_label(sample_context: "SampleContext") -> str:
    index = sample_context.degradation_index
    severity = str(sample_context.degradation_severity or "none")
    if index is not None and index > _THRESHOLDS["degradation_pair_biased_upper"]:
        label = "capture-biased"
        if severity and severity != "none":
            label += f"; orthogonal {severity}"
    elif severity == "none":
        label = "no degradation signal"
    else:
        label = severity
    if index is not None:
        label += f" ({index:.2f})"
    return label


# ── Thresholds ────────────────────────────────────────────────────────────
# Calibrated from ENCODE total-RNA vs poly-A replicates and a TCGA
# poly-A-capture snapshot. Values are conservative — the inference
# returns ``unknown`` rather than guess when signals conflict.

_THRESHOLDS = {
    # Fraction of total expressed TPM that is replication-dependent
    # histone mRNA.  Poly-A libraries: <<0.1%. Total/ribo-dep: 0.3–2%.
    "histone_fraction_polyA_ceiling": 0.002,  # below: likely poly-A
    "histone_fraction_total_floor": 0.005,  # above: likely total / ribo-dep
    # MT rRNA fraction — fraction of all MT TPM attributed to MT-RNR1/2.
    # Total RNA: MT rRNAs dominate MT signal (>70% of MT TPM).
    # Ribo-depleted: rRNAs removed; MT rRNAs usually <10% of MT TPM.
    # Poly-A: rRNAs absent.
    "mt_rrna_fraction_of_mt_depleted_ceiling": 0.10,
    "mt_rrna_fraction_of_mt_total_floor": 0.40,
    # Overall MT expression — expressed as fraction of all sample TPM.
    # <0.1% flagged as "MT missing" (either filtered or exome capture).
    "mt_fraction_suspicious_floor": 0.001,
    # Tempus xT / Twist RNA Exome and similar *hybrid capture / exome-
    # capture RNA* protocols drop MT probes entirely, so MT-total is
    # effectively zero (<0.05%) and MT-rRNA absolutely zero — much
    # stronger than the ``mt_fraction_suspicious_floor``. Used to route
    # to ``exome_capture`` even when histones are in a "limbo" band
    # (0.1-0.5% — not pure poly-A, not true total RNA).
    "mt_fraction_exome_ceiling": 0.0005,
    # Gene-length degradation index (long/short observed/expected
    # median ratio). Calibrated in sample_quality.QUALITY_THRESHOLDS;
    # kept consistent here.
    "degradation_pair_moderate": 0.30,
    "degradation_pair_severe": 0.20,
    # Upper bound on the index: values >> 1.0 indicate systematic
    # over-representation of long transcripts — a dead giveaway for
    # exon capture enrichment (long genes have more probes), NOT fresh
    # RNA. Without this bound the naive "anything > 0.55 is fresh"
    # path mislabels capture-enriched FFPE samples as fresh_frozen.
    "degradation_pair_biased_upper": 1.40,
}


# ── Data class ────────────────────────────────────────────────────────────


@dataclass
class SampleContext:
    """Base-layer of expression expectations for a sample.

    Every downstream step reads this to know which genes will be over-
    or under-represented for *artifactual* reasons, independent of
    biology.  The attribution flow is:

        raw TPM  →  [SampleContext inference]  →  sample_context +
                    expression_priors  →  coarse TME decomposition  →
                    fine TME subtype dissection  →  tumor-value
                    adjustment  →  conservative tumor-specific core

    Fields are all additive: none replace or override existing
    functionality; they *inform* it.  A ``None`` or ``unknown`` field is
    a signal that the context was inconclusive and downstream should
    behave as if the sample has default expectations.
    """

    # Library preparation method inferred from histone + MT rRNA signals.
    # One of: "poly_a", "ribo_depleted", "total_rna",
    # "exome_capture", "unknown". ``exome_capture`` means RNA
    # hybrid-capture / RNA-exome style enrichment, not DNA exome
    # sequencing; MT mRNAs can still be measured if probes/reference
    # retain them, so downstream wording must not collapse absent MT-rRNA
    # into "all MT stripped".
    library_prep: str = "unknown"
    library_prep_confidence: float = 0.0  # 0.0 (guess) to 1.0 (strong)

    # Preservation / degradation status. ``fresh_frozen`` is the default
    # expectation for a non-degraded sample; ``ffpe`` flags formalin-
    # fixed paraffin-embedded RNA; ``degraded`` catches partial
    # degradation from slow processing or thaw cycles.
    preservation: str = "unknown"  # "fresh_frozen" | "ffpe" | "degraded" | "unknown"
    preservation_confidence: float = 0.0

    # Severity of degradation from the gene-length pair index.
    degradation_severity: str = "none"  # "none" | "mild" | "moderate" | "severe"
    degradation_index: Optional[float] = None  # long/short observed/expected median

    # Whether mitochondrial genes are missing from the quant table.
    # If True, degradation signal cannot be assessed from MT fold and
    # is unreliable.
    missing_mt: bool = False

    # Diagnostic signals used by the inference, retained so the report
    # can explain *why* a call was made. Do not rely on these keys from
    # external code; they are reporting-only.
    signals: dict = field(default_factory=dict)

    # User-readable context flags for console output / summary.md.
    flags: list[str] = field(default_factory=list)

    # ── Computed downstream adjustments ──────────────────────────────

    @property
    def is_ffpe(self) -> bool:
        return self.preservation == "ffpe"

    @property
    def is_degraded(self) -> bool:
        return self.degradation_severity in ("moderate", "severe")

    def long_transcript_weight_factor(self) -> float:
        """Multiplier applied to long-transcript marker weights (#25).

        FFPE samples systematically under-represent long transcripts,
        which biases any decomposition marker with long-transcript-
        dominant reference signal.  Returns 1.0 when no degradation is
        detected and halves progressively as severity rises.
        """
        return {
            "severe": 0.3,
            "moderate": 0.5,
            "mild": 0.75,
            "none": 1.0,
        }.get(self.degradation_severity, 1.0)

    def purity_ci_widening_factor(self) -> float:
        """Multiplicative widening of purity confidence intervals (#26).

        A degraded sample yields noisier purity estimates because the
        upstream gene-expression measurements are noisier.  Returns a
        factor ≥ 1 that downstream CI construction multiplies into its
        half-width.
        """
        return {
            "severe": 1.6,
            "moderate": 1.3,
            "mild": 1.1,
            "none": 1.0,
        }.get(self.degradation_severity, 1.0)

    def summary_line(self) -> str:
        """One-line human summary for reports and stdout."""
        prep = library_prep_display_label(self.library_prep)
        pres = {
            "fresh_frozen": "fresh / frozen",
            "ffpe": "FFPE",
            "degraded": "partially degraded",
            "unknown": "unknown preservation",
        }.get(self.preservation, self.preservation)
        deg = (
            ""
            if self.degradation_severity == "none"
            else f", {self.degradation_severity} degradation"
        )
        return f"{prep}, {pres}{deg}"


# ── Inference ─────────────────────────────────────────────────────────────


def _sum_tpm_for_symbols(tpm_by_symbol, symbol_set) -> float:
    total = 0.0
    for sym in symbol_set:
        v = tpm_by_symbol.get(sym)
        if v is not None and not (isinstance(v, float) and math.isnan(v)) and v > 0:
            total += float(v)
    return total


def _sum_tpm_for_prefixes(tpm_by_symbol, prefixes) -> float:
    total = 0.0
    for sym, v in tpm_by_symbol.items():
        if not isinstance(sym, str):
            continue
        if v is None or (isinstance(v, float) and math.isnan(v)) or v <= 0:
            continue
        if sym.startswith(prefixes):
            total += float(v)
    return total


def _infer_library_prep(tpm_by_symbol, signals):
    """Infer library prep from histone + MT rRNA signals. Records
    evidence into ``signals`` and returns ``(library_prep, confidence)``.
    """
    total_tpm = sum(
        v
        for v in tpm_by_symbol.values()
        if v is not None and not (isinstance(v, float) and math.isnan(v)) and v > 0
    )
    if total_tpm <= 0:
        return "unknown", 0.0

    histone_tpm = _sum_tpm_for_prefixes(tpm_by_symbol, _HISTONE_SYMBOL_PREFIXES)
    histone_frac = histone_tpm / total_tpm

    mt_rrna_tpm = _sum_tpm_for_symbols(tpm_by_symbol, _MT_RRNA_SYMBOLS)
    mt_mrna_tpm = _sum_tpm_for_symbols(tpm_by_symbol, _MT_MRNA_SYMBOLS)
    mt_total_tpm = mt_rrna_tpm + mt_mrna_tpm
    mt_fraction = mt_total_tpm / total_tpm
    mt_rrna_fraction_of_mt = mt_rrna_tpm / mt_total_tpm if mt_total_tpm > 0 else None

    signals["histone_fraction"] = round(histone_frac, 5)
    signals["mt_fraction"] = round(mt_fraction, 5)
    signals["mt_rrna_fraction_of_mt"] = (
        round(mt_rrna_fraction_of_mt, 4) if mt_rrna_fraction_of_mt is not None else None
    )

    # RNA hybrid-capture / RNA-exome style enrichment. The strong signal
    # is absent MT-rRNA plus very low total chrM fraction, not proof that
    # every MT transcript was removed. Clinical capture references may
    # still quantify MT protein-coding transcripts at low TPM.
    if mt_fraction < _THRESHOLDS["mt_fraction_exome_ceiling"] and (mt_rrna_tpm == 0.0):
        # Very high confidence when MT-rRNA is exactly zero and total chrM
        # is very low. This is a capture-enrichment signature, not a
        # statement that MT mRNAs are unmeasurable.
        return "exome_capture", 0.9

    # Broader low-MT signal; call RNA capture at lower confidence only
    # when histones are also poly-A-like.
    if (
        mt_fraction < _THRESHOLDS["mt_fraction_suspicious_floor"]
        and histone_frac < _THRESHOLDS["histone_fraction_polyA_ceiling"]
    ):
        return "exome_capture", 0.6

    # Total RNA: histones high AND MT rRNAs dominate MT signal
    if (
        histone_frac >= _THRESHOLDS["histone_fraction_total_floor"]
        and mt_rrna_fraction_of_mt is not None
        and mt_rrna_fraction_of_mt >= _THRESHOLDS["mt_rrna_fraction_of_mt_total_floor"]
    ):
        return "total_rna", 0.9

    # Ribo-depleted: histones high AND MT rRNAs depleted
    if (
        histone_frac >= _THRESHOLDS["histone_fraction_total_floor"]
        and mt_rrna_fraction_of_mt is not None
        and mt_rrna_fraction_of_mt
        <= _THRESHOLDS["mt_rrna_fraction_of_mt_depleted_ceiling"]
    ):
        return "ribo_depleted", 0.85

    # Poly-A: histones absent AND MT rRNAs absent
    if (
        histone_frac <= _THRESHOLDS["histone_fraction_polyA_ceiling"]
        and (
            mt_rrna_fraction_of_mt is None
            or mt_rrna_fraction_of_mt
            <= _THRESHOLDS["mt_rrna_fraction_of_mt_depleted_ceiling"]
        )
        and mt_fraction >= _THRESHOLDS["mt_fraction_suspicious_floor"]
    ):
        return "poly_a", 0.8

    # Ambiguous: histones low, MT signal mixed. Most clinical cohorts
    # are poly-A so default there at low confidence.
    if histone_frac <= _THRESHOLDS["histone_fraction_polyA_ceiling"]:
        return "poly_a", 0.5

    return "unknown", 0.2


def compute_isoform_length_bias(transcript_df, signals=None):
    """Within-gene transcript-isoform length bias — capture-bias immune.

    For each gene with ≥ 2 detected isoforms (TPM > 0), compute the
    TPM-weighted mean transcript length and compare it to the geometric
    mean of the gene's max isoform length and the gene's min isoform
    length. Median across qualifying genes:

        index = 1.0  → no preferential isoform loss (fresh-like)
        index < 0.7  → systematic shorter-isoform preference (FFPE-like)
        index > 1.3  → unusual longer-isoform preference (capture-only,
                       not FFPE)

    Because both isoforms of a gene share the same capture probes,
    this signal is NOT confounded by exon-capture probe density bias.
    Returns ``(index, n_genes_used)``; (None, 0) when transcript_df is
    None / empty / has no multi-isoform genes.
    """
    if transcript_df is None or transcript_df.empty:
        return None, 0
    needed = {"transcript_id", "length", "TPM"}
    if not needed.issubset(transcript_df.columns):
        return None, 0
    # Accept either an ensembl_gene_id or gene_symbol column as the
    # gene grouping key. Raw salmon quant.sf inputs come through with
    # gene_symbol (resolved from transcript ID at load time) rather
    # than gene_id.
    group_col = None
    if "ensembl_gene_id" in transcript_df.columns:
        group_col = "ensembl_gene_id"
    elif "gene_symbol" in transcript_df.columns:
        group_col = "gene_symbol"
    else:
        return None, 0

    import numpy as np

    df = transcript_df.copy()
    df["length"] = df["length"].astype(float)
    df["TPM"] = df["TPM"].astype(float)
    df = df[df["TPM"] > 0]
    df = df[df["length"] > 0]
    # Strip trailing version on ENSG gene IDs; skip empty / NaN symbols.
    if group_col == "ensembl_gene_id":
        df = df[df["ensembl_gene_id"].astype(str).str.startswith("ENS")]
        df["_gene_key"] = df["ensembl_gene_id"].astype(str).str.split(".", n=1).str[0]
    else:
        df = df[df["gene_symbol"].notna()]
        df = df[df["gene_symbol"].astype(str).str.strip() != ""]
        df["_gene_key"] = df["gene_symbol"].astype(str)
    if df.empty:
        return None, 0

    ratios = []
    for _, group in df.groupby("_gene_key"):
        if len(group) < 2:
            continue
        lo = float(group["length"].min())
        hi = float(group["length"].max())
        if lo <= 0 or hi <= lo * 1.2:
            # Too narrow length range — not informative for length bias.
            continue
        # TPM-weighted mean length, then expressed as a position
        # between min length (= 0) and max length (= 1).
        tpm = group["TPM"].values
        lens = group["length"].values
        if tpm.sum() <= 0:
            continue
        mean_len = float(np.average(lens, weights=tpm))
        position = (mean_len - lo) / (hi - lo)
        # Convert to an "index" — 1.0 = same as length-weighted balanced
        # usage, 0 = all signal at the shortest isoform.
        ratios.append(position * 2)  # scale 0..1 → 0..2 so 1.0 ≈ balanced

    if not ratios:
        return None, 0
    index = float(np.median(ratios))
    if signals is not None:
        signals["isoform_length_bias_index"] = round(index, 3)
        signals["isoform_length_bias_n_genes"] = int(len(ratios))
    return index, int(len(ratios))


def compute_ffpe_marker_score(tpm_by_symbol, signals=None):
    """Score the FFPE-sensitive marker panel.

    Pseudocounted geomean of stable-in-FFPE genes divided by geomean of
    drops-in-FFPE genes. A value materially above the *expected fresh
    ratio* indicates FFPE: the long / unstable transcripts are
    disproportionately suppressed relative to short stable references.

    Returns ``(score, n_genes_used)`` or ``(None, 0)`` when the panel
    can't be evaluated. Threshold interpretation lives in the
    preservation routing logic so the threshold can be tuned without
    re-walking the math.
    """
    import numpy as np

    from pirlygenes.gene_sets_cancer import ffpe_sensitive_markers_df

    panel = ffpe_sensitive_markers_df()
    if panel is None or panel.empty:
        return None, 0

    drops_syms = panel[panel["direction"] == "drops_in_ffpe"]["symbol"].tolist()
    stable_syms = panel[panel["direction"] == "stable_in_ffpe"]["symbol"].tolist()

    drops_vals = [
        float(tpm_by_symbol.get(s, 0.0)) for s in drops_syms if s in tpm_by_symbol
    ]
    stable_vals = [
        float(tpm_by_symbol.get(s, 0.0)) for s in stable_syms if s in tpm_by_symbol
    ]
    if not drops_vals or not stable_vals:
        return None, 0

    drops_geomean = float(np.exp(np.mean(np.log(np.array(drops_vals) + 0.5))))
    stable_geomean = float(np.exp(np.mean(np.log(np.array(stable_vals) + 0.5))))
    if drops_geomean <= 0:
        return None, 0
    score = stable_geomean / drops_geomean
    if signals is not None:
        signals["ffpe_marker_score"] = round(score, 3)
        signals["ffpe_marker_drops_n"] = int(len(drops_vals))
        signals["ffpe_marker_stable_n"] = int(len(stable_vals))
    return score, int(len(drops_vals) + len(stable_vals))


def _infer_preservation_and_degradation(tpm_by_symbol, signals):
    """Infer preservation + degradation severity using the gene-length
    pair index from ``data/degradation-gene-pairs.csv``.

    Returns ``(preservation, preservation_confidence, severity, index)``.
    """
    from pirlygenes.gene_sets_cancer import degradation_gene_pairs

    ratios = []
    n_short_expressed = 0
    for short_gene, long_gene, expected in degradation_gene_pairs():
        short_tpm = tpm_by_symbol.get(short_gene)
        long_tpm = tpm_by_symbol.get(long_gene)
        if (
            short_tpm is None
            or (isinstance(short_tpm, float) and math.isnan(short_tpm))
            or short_tpm <= 1
        ):
            continue
        if (
            long_tpm is None
            or (isinstance(long_tpm, float) and math.isnan(long_tpm))
            or expected <= 0
        ):
            continue
        n_short_expressed += 1
        observed = float(long_tpm) / float(short_tpm)
        ratios.append(observed / float(expected))

    signals["n_degradation_pairs"] = n_short_expressed

    if not ratios:
        return "unknown", 0.0, "none", None

    import numpy as np

    index = float(np.median(ratios))
    signals["degradation_index"] = round(index, 3)

    if index < _THRESHOLDS["degradation_pair_severe"]:
        return "ffpe", 0.85, "severe", index
    if index < _THRESHOLDS["degradation_pair_moderate"]:
        return "ffpe", 0.7, "moderate", index
    if index < 0.55:
        return "degraded", 0.6, "mild", index
    # Upper bound (bug 2026-04-14): index >> 1.0 indicates systematic
    # over-representation of long transcripts — exon-capture enrichment,
    # NOT fresh RNA. The length-pair signal can't report preservation
    # in that regime because the reference calibration (fresh tissue,
    # uncapture-enriched) doesn't apply. Return ``unknown`` so
    # downstream consumers don't treat a capture-biased sample as
    # confidently fresh.
    if index > _THRESHOLDS["degradation_pair_biased_upper"]:
        return "unknown", 0.0, "none", index
    return "fresh_frozen", 0.7, "none", index


def _build_tpm_by_symbol(df_gene_expr):
    """Return {symbol: max_TPM}, working whether the input frame has
    a symbol column directly or only a gene_id column.

    Prefers a direct symbol column (``gene_symbol``/``symbol``/``Symbol``)
    so that synthetic test frames with just ``(gene_symbol, TPM)`` work
    without requiring a gene-ID column. Falls back to
    ``common.build_sample_tpm_by_symbol`` (which maps gene IDs
    through the HPA/pan-cancer reference) for frames without a symbol
    column.
    """
    sym_col = next(
        (
            c
            for c in ("gene_symbol", "gene", "Symbol", "symbol", "GeneName")
            if c in df_gene_expr.columns
        ),
        None,
    )
    tpm_col = (
        "TPM"
        if "TPM" in df_gene_expr.columns
        else next((c for c in df_gene_expr.columns if c.lower() == "tpm"), None)
    )
    if sym_col is not None and tpm_col is not None:
        import pandas as pd

        with without_dataframe_attrs(df_gene_expr):
            syms = df_gene_expr[sym_col].astype(str)
            tpms = pd.to_numeric(df_gene_expr[tpm_col], errors="coerce")
            valid = syms.ne("") & syms.ne("nan") & syms.ne("None") & tpms.notna()
            return dict(
                pd.DataFrame({"sym": syms[valid], "tpm": tpms[valid]})
                .groupby("sym")["tpm"]
                .max()
            )
    # Fall back to the gene-ID path (canonical home: common.py).
    from .common import build_sample_tpm_by_symbol

    return build_sample_tpm_by_symbol(df_gene_expr)


def _summarise_expression_distribution(tpm_by_symbol, signals):
    """Compute shape of the full expression distribution.

    The whole range of TPM values carries library-prep and depth
    information that is independent of the histone / MT / length-pair
    probes above: detection breadth (how many genes are seen at all),
    dynamic range, tail heaviness, and median expressed level.

    Adds fields to ``signals``:

    - ``genes_detected_above_0p5_tpm`` / ``above_1_tpm`` / ``above_10_tpm``
    - ``log2_tpm_median`` (over expressed genes only) and IQR
    - ``high_expression_tail_share`` — fraction of total TPM captured by
      the top 1% of genes (platforms with aggressive poly-A capture and
      shallow depth concentrate more mass in fewer genes)
    """
    import numpy as np

    values = np.array(
        [float(v) for v in tpm_by_symbol.values() if v > 0],
        dtype=float,
    )
    n_genes = int(values.size)
    signals["n_genes_with_any_expression"] = n_genes
    if n_genes == 0:
        return
    signals["genes_detected_above_0p5_tpm"] = int((values > 0.5).sum())
    signals["genes_detected_above_1_tpm"] = int((values > 1.0).sum())
    signals["genes_detected_above_10_tpm"] = int((values > 10.0).sum())

    expressed = values[values > 1.0]
    if expressed.size:
        log2_expr = np.log2(expressed + 1.0)
        signals["log2_tpm_median"] = round(float(np.median(log2_expr)), 3)
        signals["log2_tpm_iqr"] = round(
            float(np.subtract(*np.percentile(log2_expr, [75, 25]))), 3
        )
        signals["log2_tpm_p95"] = round(float(np.percentile(log2_expr, 95)), 3)

    total = float(values.sum())
    if total > 0:
        sorted_desc = np.sort(values)[::-1]
        top_n = max(1, int(round(0.01 * n_genes)))
        signals["top_gene_share_of_total_tpm"] = round(float(sorted_desc[0]) / total, 4)
        signals["top_10_share_of_total_tpm"] = (
            round(float(sorted_desc[:10].sum()) / total, 4) if n_genes >= 10 else None
        )
        signals["top_1pct_share_of_total_tpm"] = round(
            float(sorted_desc[:top_n].sum()) / total, 4
        )
        # Panel-vs-whole-transcriptome heuristic (#68): whole-
        # transcriptome samples carry ~10–15% of total TPM in the top
        # 50 genes; targeted panels concentrate much more sharply.
        signals["top_50_share_of_total_tpm"] = (
            round(float(sorted_desc[:50].sum()) / total, 4) if n_genes >= 50 else None
        )
        signals["top_2000_share_of_total_tpm"] = (
            round(float(sorted_desc[:2000].sum()) / total, 4)
            if n_genes >= 2000
            else None
        )
        # Likely a targeted panel when BOTH signals fire (few detected
        # genes AND most TPM concentrated in a narrow top). Using AND
        # avoids flagging normal-tissue nTPM references (which are
        # strongly concentrated at the top but still cover >10k genes)
        # while still catching sparse real panels (few genes, heavy
        # top-2000 share).
        #
        # When the sample has fewer than 2000 detected genes total,
        # top-2000 share is trivially 1.0 — treat it that way rather
        # than ``None`` so the heuristic fires correctly on very small
        # panels.
        n_det_1 = signals.get("genes_detected_above_1_tpm", 0) or 0
        top_2000 = signals.get("top_2000_share_of_total_tpm")
        effective_top_2000 = top_2000 if top_2000 is not None else 1.0
        signals["likely_targeted_panel"] = bool(
            n_det_1 < 4000 and effective_top_2000 >= 0.95
        )
        top_items = sorted(
            tpm_by_symbol.items(),
            key=lambda item: (-float(item[1]), item[0]),
        )[:10]
        signals["dominant_expression_genes"] = [
            {
                "gene": gene,
                "tpm": round(float(value), 3),
                "share": round(float(value) / total, 4),
                "qc_class": classify_gene_qc(gene).label,
                "qc_group": classify_gene_qc(gene).group,
            }
            for gene, value in top_items
        ]
        signals["qc_class_shares"] = summarize_qc_class_shares(
            (gene, float(value)) for gene, value in tpm_by_symbol.items()
        )
        top_gene = signals.get("top_gene_share_of_total_tpm") or 0.0
        top_10 = signals.get("top_10_share_of_total_tpm") or 0.0
        top_50 = signals.get("top_50_share_of_total_tpm") or 0.0
        if top_gene >= 0.20 or top_10 >= 0.60 or top_50 >= 0.75:
            signals["expression_concentration_level"] = "extreme"
        elif top_10 >= 0.35 or top_50 >= 0.50:
            signals["expression_concentration_level"] = "high"
        else:
            signals["expression_concentration_level"] = "typical"


def _refine_preservation_with_orthogonal_signals(
    preservation,
    severity,
    index,
    tpm_by_symbol,
    transcript_df,
    signals,
    flags,
):
    """Second-pass preservation call using the FFPE marker panel and
    transcript-level isoform-length-bias index — both immune to
    capture-enrichment probe-density bias.

    Only fires when the primary length-pair test was inconclusive
    (``preservation == "unknown"`` because the index sat above the
    fresh-reference upper bound). Returns possibly-updated
    ``(preservation, severity)`` plus appended flags. Never overrides
    a confident FFPE / fresh call.
    """
    if preservation != "unknown":
        return preservation, severity

    ffpe_score, _ = compute_ffpe_marker_score(tpm_by_symbol, signals=signals)
    iso_index, n_iso_genes = compute_isoform_length_bias(transcript_df, signals=signals)

    # Calibration: in fresh / frozen samples the FFPE marker score
    # (geomean(stable) / geomean(drops)) sits roughly 1–8 because the
    # "drops_in_ffpe" panel is dominated by lowly-expressed muscle /
    # neuronal long transcripts. In FFPE it climbs well above 30.
    # Conservative: flag FFPE only when the score is > 30.
    ffpe_strong = ffpe_score is not None and ffpe_score > 30.0
    ffpe_moderate = ffpe_score is not None and ffpe_score > 15.0

    # Isoform length bias (within-gene, capture-immune):
    # ~1.0 fresh, < 0.6 systematic short-isoform preference (FFPE).
    iso_strong = iso_index is not None and n_iso_genes >= 50 and iso_index < 0.6
    iso_moderate = iso_index is not None and n_iso_genes >= 50 and iso_index < 0.85

    if ffpe_strong and iso_strong:
        flags.append(
            f"FFPE detected via orthogonal signals: marker score "
            f"{ffpe_score:.1f} (long-transcript panel suppressed) + "
            f"isoform bias {iso_index:.2f} (within-gene short isoforms "
            "dominate). Length-pair index was masked by capture enrichment."
        )
        return "ffpe", "severe"
    if ffpe_strong or (ffpe_moderate and iso_moderate):
        flags.append(
            "FFPE likely (orthogonal signals): "
            + (f"marker score {ffpe_score:.1f}" if ffpe_score else "")
            + (
                f"; isoform bias {iso_index:.2f} (n={n_iso_genes})"
                if iso_index is not None
                else ""
            )
            + ". Length-pair index masked by capture enrichment."
        )
        return "ffpe", "moderate"
    if ffpe_moderate or iso_moderate:
        flags.append(
            "Possible FFPE: orthogonal signals weakly support degradation "
            + (f"(marker score {ffpe_score:.1f}" if ffpe_score else "(")
            + (f", isoform bias {iso_index:.2f})" if iso_index is not None else ")")
        )
        return "degraded", "mild"
    return preservation, severity


def infer_sample_context(df_gene_expr) -> SampleContext:
    """Infer a ``SampleContext`` from a TPM expression table.

    This is the FIRST step of the unified attribution flow — it runs
    before cancer-type inference, decomposition, and any downstream
    analysis that might be biased by FFPE / library-prep artifacts.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Expression frame with at minimum a ``gene_symbol`` (or
        ``Symbol``/``symbol``) column and a ``TPM`` column. Accepts
        the full pirlygenes expression format as well — any frame that
        works with the existing purity pipeline works here.

    Returns
    -------
    SampleContext
        ``library_prep``, ``preservation``, ``degradation_severity``,
        ``missing_mt``, diagnostic ``signals``, and human-readable
        ``flags``.
    """
    try:
        tpm_by_symbol = _build_tpm_by_symbol(df_gene_expr)
    except (KeyError, ValueError, TypeError):
        # Malformed / test-harness frames without gene-id or symbol
        # columns cannot yield a context. Return a conservative unknown
        # context so downstream code still has a valid object to read.
        return SampleContext(
            library_prep="unknown",
            preservation="unknown",
            degradation_severity="none",
            missing_mt=True,
            flags=["Sample context inference skipped — input frame has no gene column"],
        )
    # Drop NaN
    tpm_by_symbol = {
        s: v
        for s, v in tpm_by_symbol.items()
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    }

    signals = {}

    print("[context]   expression distribution...")
    _summarise_expression_distribution(tpm_by_symbol, signals)
    print("[context]   library prep inference...")
    library_prep, lp_conf = _infer_library_prep(tpm_by_symbol, signals)
    print("[context]   preservation / degradation...")
    preservation, pr_conf, severity, index = _infer_preservation_and_degradation(
        tpm_by_symbol, signals
    )

    print("[context]   orthogonal-signal refinement...")
    transcript_df = (
        df_gene_expr.attrs.get("transcript_expression")
        if hasattr(df_gene_expr, "attrs")
        else None
    )
    refined_flags: list[str] = []
    preservation, severity = _refine_preservation_with_orthogonal_signals(
        preservation,
        severity,
        index,
        tpm_by_symbol,
        transcript_df,
        signals,
        refined_flags,
    )
    # If refinement updated the call, raise the confidence above 0.
    if refined_flags:
        pr_conf = max(pr_conf, 0.65)

    # Missing-MT detection (#61). We check the count of MT symbols that
    # map to any TPM > 0 in the sample. ``missing_mt`` is a distinct
    # signal from preservation — it's about the quant table, not the RNA.
    mt_found = sum(
        1
        for s in (_MT_RRNA_SYMBOLS | _MT_MRNA_SYMBOLS)
        if s in tpm_by_symbol and tpm_by_symbol[s] > 0
    )
    signals["mt_genes_detected"] = mt_found
    signals["mt_genes_total"] = len(_MT_RRNA_SYMBOLS) + len(_MT_MRNA_SYMBOLS)
    missing_mt = mt_found <= 1

    flags = []
    concentration_level = str(
        signals.get("expression_concentration_level") or ""
    ).strip()
    if concentration_level in {"high", "extreme"}:
        top_gene = signals.get("top_gene_share_of_total_tpm")
        top_10 = signals.get("top_10_share_of_total_tpm")
        top_50 = signals.get("top_50_share_of_total_tpm")
        dominant = signals.get("dominant_expression_genes") or []
        dominant_label = ""
        if dominant:
            class_label = str(dominant[0].get("qc_class") or "").strip()
            class_suffix = (
                f" ({class_label})"
                if class_label and class_label != "protein-coding/other"
                else ""
            )
            dominant_label = f"; top gene {dominant[0]['gene']}{class_suffix}"
        top_gene_label = f"{top_gene:.0%}" if isinstance(top_gene, (int, float)) else "n/a"
        top_10_label = f"{top_10:.0%}" if isinstance(top_10, (int, float)) else "n/a"
        top_50_label = f"{top_50:.0%}" if isinstance(top_50, (int, float)) else "n/a"
        flags.append(
            f"Expression distribution is {concentration_level}ly concentrated "
            f"(top gene {top_gene_label}, top 10 {top_10_label}, top 50 {top_50_label}"
            f"{dominant_label}) — check for rRNA/pseudogene/contaminant dominance, "
            "low library complexity, or assay/input issues before over-interpreting "
            "expression-derived calls"
        )
    if missing_mt:
        flags.append(
            "Mitochondrial genes missing from quant table — degradation "
            "signal from MT fraction is unreliable; FFPE detection falls "
            "back to gene-length pair index only"
        )
    if library_prep == "unknown":
        flags.append(
            "Library prep inconclusive — histone + MT-rRNA signals did not match any profile"
        )
    else:
        flags.append(
            f"Library prep: {library_prep_display_label(library_prep)} "
            f"({heuristic_support_label(lp_conf)})"
        )
    if preservation == "ffpe":
        # If the FFPE call came from the orthogonal-signal refinement
        # path, the explanatory flag was already added there. Only emit
        # the length-pair-based message when the primary test was the
        # one that triggered FFPE.
        if not refined_flags:
            flags.append(
                f"Preservation: FFPE / heavily degraded (length-pair index {index:.2f})"
            )
    elif preservation == "degraded":
        if not refined_flags:
            flags.append(
                f"Preservation: partial degradation (length-pair index {index:.2f})"
            )
    elif preservation == "fresh_frozen":
        flags.append(f"Preservation: fresh / frozen (length-pair index {index:.2f})")
    elif preservation == "unknown" and index is not None and index > 1.0:
        # Capture-enriched libraries (exome / hybrid) inflate the long/
        # short ratio because long genes have more probes. Even after
        # the orthogonal-signal pass, the call remained inconclusive —
        # surface that and explain why so the user knows what's missing.
        flags.append(
            f"Preservation unknown — length-pair index {index:.2f} is "
            f"above fresh-reference range (>{_THRESHOLDS['degradation_pair_biased_upper']:.1f}), "
            "consistent with exon-capture enrichment (long transcripts "
            "over-represented). FFPE marker panel and isoform length "
            "bias did not strongly support FFPE either; orthogonal "
            "BAM-level signals (read-length distribution, 3′ bias) "
            "needed for a definitive FFPE call."
        )

    # Append refinement-step flags last so they read in narrative order.
    flags.extend(refined_flags)

    return SampleContext(
        library_prep=library_prep,
        library_prep_confidence=lp_conf,
        preservation=preservation,
        preservation_confidence=pr_conf,
        degradation_severity=severity,
        degradation_index=index,
        missing_mt=missing_mt,
        signals=signals,
        flags=flags,
    )


# ── Plotting ──────────────────────────────────────────────────────────────


_GENE_CLASS_TO_SIGNAL = {
    "histone_fraction": "histone_cluster",
    "mt_fraction": "mitochondrial",
    "mt_rrna_fraction_of_mt": "mitochondrial",  # same gene class
}


def _load_artifact_expectations():
    """Return (library_prep, preservation, gene_class) -> (lo, hi, interp).

    Loaded from ``data/artifact-expectations.csv`` (#107). ``preservation``
    is either an explicit value or ``"*"``, which matches any preservation.
    Returns an empty dict when the CSV is missing so downstream code
    degrades gracefully.
    """
    try:
        from pirlygenes.load_dataset import get_data

        df = get_data("artifact-expectations")
    except Exception:
        return {}
    out = {}
    for _, row in df.iterrows():
        key = (
            str(row["library_prep"]).strip(),
            str(row["preservation"]).strip(),
            str(row["gene_class"]).strip(),
        )
        out[key] = (
            float(row["expected_lo"]),
            float(row["expected_hi"]),
            str(row.get("interpretation", "")).strip(),
        )
    return out


def _expectation_for(expectations, library_prep, preservation, gene_class):
    """Look up the expected range for a (prep, preservation, class) combo.

    Exact preservation match preferred; ``preservation="*"`` used as
    the fallback. Returns ``None`` when no rule applies (unknown prep
    or gene class).
    """
    for pres in (preservation, "*"):
        key = (library_prep, pres, gene_class)
        if key in expectations:
            return expectations[key]
    return None


def _sample_context_concentration_line(sample_context: SampleContext) -> str:
    """Short bridge from sample-context inference to expression QC."""

    signals = sample_context.signals or {}
    level = str(signals.get("expression_concentration_level") or "").strip()
    top_10_share = signals.get("top_10_share_of_total_tpm")
    if not level and top_10_share is None:
        return ""
    prefix = "TPM concentration"
    if level:
        prefix += f": {level}"
    parts = [prefix]
    if top_10_share is not None:
        parts.append(f"top 10 {float(top_10_share):.0%}")
    return f"{parts[0]} ({', '.join(parts[1:])})" if len(parts) > 1 else parts[0]


def plot_sample_context(
    sample_context: SampleContext, save_to_filename: str, save_dpi: int = 150
) -> Optional[str]:
    """Standalone PNG summarising ``sample_context`` as the first panel
    a user sees in an analyze report.

    Two-row layout (single PNG — plot-crowding preference):

    - **Top**: text block with library prep, preservation, degradation
      severity, missing-MT flag, and confidence.
    - **Bottom**: three horizontal bars showing the diagnostic fractions
      (histone mRNA fraction of total; MT rRNA fraction of MT; overall
      MT fraction of total) with the thresholds used for the call.
    """
    import matplotlib.pyplot as plt

    fig, (ax_text, ax_bars) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 5.5),
        gridspec_kw={"height_ratios": [1, 1.5]},
    )
    ax_text.axis("off")

    prep_label = library_prep_display_label(
        sample_context.library_prep,
        title_case=True,
    )

    pres_label = {
        "fresh_frozen": "Fresh / frozen",
        "ffpe": "FFPE",
        "degraded": "Partial degradation",
        "unknown": "Unknown",
    }.get(sample_context.preservation, sample_context.preservation)

    severity_color = {
        "none": "#2e8b57",
        "mild": "#daa520",
        "moderate": "#d2691e",
        "severe": "#b22222",
    }.get(sample_context.degradation_severity, "#666666")

    header_lines = [
        (
            f"Library:       {prep_label} "
            f"({heuristic_support_label(sample_context.library_prep_confidence)})"
        ),
        f"Preservation:  {pres_label}",
        f"Length-pair:   {length_pair_display_label(sample_context)}",
    ]
    if sample_context.missing_mt:
        header_lines.append("MT genes missing from quant table")

    concentration_line = _sample_context_concentration_line(sample_context)
    ax_text.text(
        0.02,
        0.95,
        "Sample context",
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    for i, line in enumerate(header_lines):
        ax_text.text(
            0.02,
            0.70 - 0.22 * i,
            line,
            fontsize=11,
            va="top",
            family="monospace",
            color=severity_color if i == 2 else "black",
        )
    if concentration_line:
        ax_text.text(
            0.02,
            0.04,
            concentration_line,
            fontsize=9,
            va="bottom",
            color="#7a2e1f",
        )

    # Bottom panel: diagnostic bars
    signals = sample_context.signals
    hist_frac = signals.get("histone_fraction", 0.0) or 0.0
    mt_frac = signals.get("mt_fraction", 0.0) or 0.0
    mt_rrna_frac = signals.get("mt_rrna_fraction_of_mt") or 0.0

    bars = []
    signal_keys = []

    def _append_bar(label, value, threshold, threshold_name, signal_key):
        bars.append((label, value, threshold, threshold_name))
        signal_keys.append(signal_key)

    _append_bar(
        "Histone fraction\n(replication-dependent mRNAs)",
        hist_frac,
        _THRESHOLDS["histone_fraction_total_floor"],
        "Total/ribo-dep floor",
        "histone_fraction",
    )
    _append_bar(
        "MT fraction\n(of total sample TPM)",
        mt_frac,
        _THRESHOLDS["mt_fraction_suspicious_floor"],
        "Suspicious-floor",
        "mt_fraction",
    )
    _append_bar(
        "MT-rRNA / MT-total",
        mt_rrna_frac,
        _THRESHOLDS["mt_rrna_fraction_of_mt_total_floor"],
        "Total-RNA floor",
        "mt_rrna_fraction_of_mt",
    )
    top_10_share = signals.get("top_10_share_of_total_tpm")
    concentration_level = str(signals.get("expression_concentration_level") or "")
    qc_shares = signals.get("qc_class_shares") or {}
    rrna_pseudogene_fraction = float(
        qc_shares.get("rrna_pseudogene_fraction") or 0.0
    )
    rrna_plus_mt_fraction = float(qc_shares.get("rrna_plus_mt_fraction") or 0.0)
    nuclear_rrna_like_fraction = float(
        qc_shares.get("nuclear_rrna_like_fraction") or 0.0
    )
    if top_10_share is not None:
        _append_bar(
            "Top-10 TPM share\n(expression concentration QC)",
            float(top_10_share),
            0.60,
            "QC flag",
            "top_10_share_of_total_tpm",
        )
    if rrna_pseudogene_fraction >= 0.005:
        _append_bar(
            "rRNA pseudogene\n(of total sample TPM)",
            rrna_pseudogene_fraction,
            0.01,
            "QC flag",
            "rrna_pseudogene_fraction",
        )
    elif nuclear_rrna_like_fraction >= 0.005:
        _append_bar(
            "Nuclear rRNA-like\n(of total sample TPM)",
            nuclear_rrna_like_fraction,
            0.01,
            "QC flag",
            "nuclear_rrna_like_fraction",
        )
    if rrna_plus_mt_fraction >= 0.10:
        _append_bar(
            "mtDNA + rRNA-like\n(technical-RNA burden)",
            rrna_plus_mt_fraction,
            0.10,
            "QC flag",
            "rrna_plus_mt_fraction",
        )
    y = [i for i in range(len(bars))]
    values = [b[1] for b in bars]
    labels = [b[0] for b in bars]
    thresholds = [(b[2], b[3]) for b in bars]

    # #107: expected-range bands for the inferred library prep +
    # preservation. Shading a band behind each bar lets the reader see
    # whether the observed value is inside the "normal for this prep"
    # window instead of chasing an absolute threshold that doesn't
    # apply to every prep.
    expectations = _load_artifact_expectations()
    bar_bands = []
    in_band_count = 0
    for yi, sig_key in zip(y, signal_keys):
        gene_class = _GENE_CLASS_TO_SIGNAL.get(sig_key)
        band = (
            _expectation_for(
                expectations,
                sample_context.library_prep,
                sample_context.preservation,
                gene_class,
            )
            if gene_class
            else None
        )
        bar_bands.append(band)
        if band is not None:
            lo, hi, _ = band
            ax_bars.axhspan(
                yi - 0.35,
                yi + 0.35,
                xmin=0.0,
                xmax=1.0,
                facecolor="none",
                edgecolor="none",  # no-op placeholder
            )
            # Use axvspan-shape-per-row: draw a narrow patch via bar
            # on a secondary layer to keep implementation simple.
            ax_bars.barh(
                [yi],
                [hi - lo],
                left=[lo],
                color="#bbd8a3",
                alpha=0.4,
                edgecolor="none",
                height=0.75,
                zorder=0,
            )
            val = (
                values[yi]
                if sig_key
                in ("histone_fraction", "mt_fraction", "mt_rrna_fraction_of_mt")
                else None
            )
            if val is not None and lo <= val <= hi:
                in_band_count += 1

    ax_bars.barh(y, values, color="#4682b4", alpha=0.85)
    ax_bars.set_yticks(y)
    label_fontsize = 8 if len(bars) > 4 else 9
    ax_bars.set_yticklabels(labels, fontsize=label_fontsize)
    ax_bars.invert_yaxis()
    # #102: with exome-capture samples every value is ~0 and a raw axis
    # (xlim = 1.0) produces a chart that looks empty. Give a minimum axis
    # width tied to the thresholds so bars remain readable, and annotate
    # the "all near zero" case with the plausible library-prep explanation.
    max_value = max(values) if values else 0.0
    max_threshold = max((b[2] for b in bars), default=0.0)
    max_band_hi = max((band[1] for band in bar_bands if band is not None), default=0.0)
    if max_value < 0.01:
        axis_floor = max(max_threshold * 4.0, max_band_hi * 1.1, 0.05)
    else:
        axis_floor = max(max_threshold * 1.15, max_band_hi * 1.1, 0.05)
    ax_bars.set_xlim(0, min(1.0, max(max_value * 1.12, axis_floor)))
    ax_bars.set_xlabel("Fraction", fontsize=10)
    for yi, (thr, thr_name) in zip(y, thresholds):
        ax_bars.axvline(thr, color="#888888", linestyle="--", linewidth=0.8)
        ax_bars.text(
            thr,
            yi - 0.4,
            thr_name,
            fontsize=7,
            color="#666666",
            ha="left",
        )
    for yi, val, band in zip(y, values, bar_bands):
        if band is not None:
            lo, hi, _ = band
            status = (
                "ok"
                if lo <= val <= hi
                else ("out" if val < lo * 0.5 or val > hi * 2 else "~")
            )
        else:
            status = ""
        text_color = (
            "#7a2e1f"
            if (
                (
                    signal_keys[yi] == "top_10_share_of_total_tpm"
                    and concentration_level in {"high", "extreme"}
                )
                or signal_keys[yi]
                in {
                    "rrna_pseudogene_fraction",
                    "nuclear_rrna_like_fraction",
                    "rrna_plus_mt_fraction",
                }
            )
            else "black"
        )
        ax_bars.text(
            val, yi, f"  {val:.3f} {status}", va="center", fontsize=9, color=text_color
        )

    if max_value < 0.01:
        # Show the user what "all near zero" means rather than leaving a
        # visually empty plot. Exome / hybrid-capture library prep strips
        # non-polyadenylated RNAs (histones, mitochondrial), so near-zero
        # values are the expected pattern — not a quality problem.
        explanation = (
            "All diagnostic signals are near zero — consistent with\n"
            "exome / hybrid-capture library prep (non-polyadenylated\n"
            "RNAs are filtered). Interpret MT / histone fractions\n"
            "relative to the expected floor for this prep, not as\n"
            "degradation evidence."
        )
        ax_bars.text(
            0.98,
            0.05,
            explanation,
            transform=ax_bars.transAxes,
            fontsize=8,
            color="#444444",
            ha="right",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#f5f5dc",
                edgecolor="#bbbbbb",
                linewidth=0.6,
            ),
        )

    ax_bars.spines["top"].set_visible(False)
    ax_bars.spines["right"].set_visible(False)
    ax_bars.set_title("Library-prep and degradation signals", fontsize=10, loc="left")

    fig.tight_layout()
    fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    return save_to_filename


def _qc_fraction_from_items(items) -> dict[str, float]:
    summary = summarize_qc_class_shares(items)
    return {
        "mt_dna_fraction": float(summary.get("mt_dna_fraction") or 0.0),
        "mitochondrial_rrna_fraction": float(
            summary.get("mitochondrial_rrna_fraction") or 0.0
        ),
        "mt_non_rrna_fraction": float(summary.get("mt_non_rrna_fraction") or 0.0),
        "rrna_like_fraction": float(summary.get("rrna_like_fraction") or 0.0),
        "nuclear_rrna_like_fraction": float(
            summary.get("nuclear_rrna_like_fraction") or 0.0
        ),
        "rrna_pseudogene_fraction": float(
            summary.get("rrna_pseudogene_fraction") or 0.0
        ),
        "rrna_plus_mt_fraction": float(summary.get("rrna_plus_mt_fraction") or 0.0),
    }


def _sample_qc_fractions(df_gene_expr) -> dict[str, float]:
    return _qc_fraction_from_items(_build_tpm_by_symbol(df_gene_expr).items())


def _reference_qc_fraction_rows():
    """Return QC fractions for bundled TCGA/HPA reference columns.

    The package ships TCGA cohort medians and HPA tissue/cell-type reference
    columns, not the full raw per-patient/per-donor matrices. These rows are
    therefore reference-column distributions, used as a visual QC context.
    """

    from trufflepig.reference import pan_cancer_expression
    ref = pan_cancer_expression().drop_duplicates(subset="Symbol")
    rows = []
    symbol_values = ref["Symbol"].fillna("").astype(str).tolist()

    def qc_value_columns(alias_suffix: str, raw_suffix: str):
        alias_cols = [str(c) for c in ref.columns if str(c).endswith(alias_suffix)]
        if alias_cols:
            return [
                (
                    col.removesuffix(alias_suffix),
                    f"{col}_raw" if f"{col}_raw" in ref.columns else col,
                )
                for col in alias_cols
            ]
        return [
            (str(col).removesuffix(raw_suffix), str(col))
            for col in ref.columns
            if str(col).endswith(raw_suffix)
        ]

    for label, col in qc_value_columns("_TPM", "_TPM_raw"):
        fractions = _qc_fraction_from_items(
            zip(symbol_values, ref[col].fillna(0.0).astype(float))
        )
        rows.append(
            {
                "source": "TCGA cohort medians",
                "label": label,
                **fractions,
            }
        )
    for label, col in qc_value_columns("_nTPM", "_nTPM_raw"):
        fractions = _qc_fraction_from_items(
            zip(symbol_values, ref[col].fillna(0.0).astype(float))
        )
        rows.append(
            {
                "source": "HPA normal tissues",
                "label": label.replace("_", " "),
                **fractions,
            }
        )
    return rows


def _expression_concentration_plot_data(df_gene_expr, top_n: int = 20):
    import numpy as np

    tpm_by_symbol = _build_tpm_by_symbol(df_gene_expr)
    items = [
        (gene, float(value))
        for gene, value in tpm_by_symbol.items()
        if value is not None and float(value) > 0
    ]
    if not items:
        return None
    total = sum(value for _, value in items)
    if total <= 0:
        return None
    items = sorted(items, key=lambda item: (-item[1], item[0]))

    def display_class(gene: str) -> str:
        qc = classify_gene_qc(gene)
        if qc.group == "rrna_like" and "pseudogene" in qc.label:
            return "rRNA pseudogene"
        if qc.group == "rrna_like":
            return "rRNA-like"
        if qc.label == "mitochondrial rRNA":
            return "mitochondrial rRNA"
        if qc.group == "mt_dna":
            return "other mtDNA"
        if qc.group == "ribosomal_protein_pseudogene":
            return "RP pseudogene"
        if qc.group == "ribosomal_protein":
            return "ribosomal protein"
        if qc.group == "small_ncrna":
            return "small ncRNA"
        if qc.group == "histone":
            return "histone"
        if qc.group == "immune_receptor":
            return "immune receptor"
        if qc.group == "hemoglobin":
            return "hemoglobin"
        return "other"

    top = items[:top_n]
    sorted_values = np.array([value for _gene, value in items], dtype=float)
    return {
        "items": items,
        "top": top,
        "total": total,
        "shares": [value / total for _, value in top],
        "labels": [
            gene if len(gene) <= 24 else gene[:21] + "..."
            for gene, _value in top
        ],
        "classes": [display_class(gene) for gene, _value in top],
        "cumulative": np.cumsum(sorted_values) / total,
    }


def _expression_concentration_colors() -> dict[str, str]:
    return {
        "rRNA pseudogene": "#b64b3c",
        "rRNA-like": "#df7f5f",
        "mitochondrial rRNA": "#7a5195",
        "other mtDNA": "#b189c6",
        "ribosomal protein": "#2f7ed8",
        "RP pseudogene": "#6aaed6",
        "small ncRNA": "#d99b2b",
        "histone": "#5f8f3f",
        "immune receptor": "#c44e52",
        "hemoglobin": "#8c564b",
        "other": "#6c757d",
    }


def _format_feature_share(share: float) -> str:
    """Readable percent label for top-feature QC plots."""
    if share <= 0:
        return "0%"
    if share < 0.005:
        return "<0.5%"
    return f"{share:.0%}"


def _add_expression_concentration_legend(ax, classes, colors) -> None:
    import matplotlib.pyplot as plt

    legend_handles = []
    for group, color in colors.items():
        if group in classes:
            legend_handles.append(
                plt.Line2D([0], [0], marker="s", linestyle="", color=color, label=group)
            )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8)


def plot_expression_concentration_top_features_qc(
    df_gene_expr,
    save_to_filename: str,
    save_dpi: int = 150,
    top_n: int = 20,
) -> Optional[str]:
    """Plot dominant genes/features for expression-concentration QC."""

    import matplotlib.pyplot as plt
    import numpy as np

    data = _expression_concentration_plot_data(df_gene_expr, top_n=top_n)
    if data is None:
        return None
    top = data["top"]
    shares = data["shares"]
    labels = data["labels"]
    classes = data["classes"]
    colors = _expression_concentration_colors()
    bar_colors = [colors.get(cls, "#6c757d") for cls in classes]

    fig, ax_bar = plt.subplots(figsize=(8.8, 6.2))
    y = np.arange(len(top))
    ax_bar.barh(y, shares, color=bar_colors, alpha=0.92)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Share of total TPM")
    ax_bar.set_title("Dominant expression features", loc="left", fontsize=12)
    # Whole-percent labels collapse every tick to ``0%`` when the dominant
    # feature is below 1%. Keep one decimal in that range so low-concentration
    # samples remain interpretable, while preserving compact integer labels for
    # high-concentration technical-RNA failures.
    max_share = float(max(shares, default=0.0))
    decimals = 1 if max_share < 0.02 else 0
    ax_bar.xaxis.set_major_formatter(
        lambda x, _pos: f"{x:.{decimals}%}"
    )
    for yi, share, (gene, value) in zip(y, shares, top):
        qc = classify_gene_qc(gene)
        annotation = _format_feature_share(float(share))
        if qc.group != "other" and share >= 0.005:
            annotation += f"  {qc.label}"
        ax_bar.text(share, yi, "  " + annotation, va="center", fontsize=8)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    _add_expression_concentration_legend(ax_bar, classes, colors)
    fig.suptitle("Expression concentration QC: top features", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    return save_to_filename


def plot_expression_concentration_curve_qc(
    df_gene_expr,
    save_to_filename: str,
    save_dpi: int = 150,
    top_n: int = 20,
) -> Optional[str]:
    """Plot cumulative TPM concentration curve."""

    import matplotlib.pyplot as plt
    import numpy as np

    data = _expression_concentration_plot_data(df_gene_expr, top_n=top_n)
    if data is None:
        return None
    cumulative = data["cumulative"]
    ranks = np.arange(1, len(cumulative) + 1)

    fig, ax_curve = plt.subplots(figsize=(7.2, 5.8))
    ax_curve.plot(ranks, cumulative, color="#333333", linewidth=2)
    for n in (1, 10, 50, 200, 2000):
        if len(cumulative) >= n:
            ax_curve.scatter([n], [cumulative[n - 1]], color="#b64b3c", zorder=3)
            ax_curve.text(
                n,
                cumulative[n - 1],
                f" top {n}: {cumulative[n - 1]:.0%}",
                fontsize=8,
                va="bottom",
            )
    ax_curve.set_xscale("log")
    ax_curve.set_ylim(0, 1.02)
    ax_curve.set_xlabel("Gene rank by TPM")
    ax_curve.set_ylabel("Cumulative share of total TPM")
    ax_curve.set_title("TPM concentration curve", loc="left", fontsize=12)
    ax_curve.yaxis.set_major_formatter(lambda x, _pos: f"{x:.0%}")
    ax_curve.grid(True, axis="y", alpha=0.25)
    ax_curve.spines["top"].set_visible(False)
    ax_curve.spines["right"].set_visible(False)
    fig.suptitle("Expression concentration QC: cumulative curve", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    return save_to_filename


def _plot_reference_technical_rna_metric_qc(
    df_gene_expr,
    *,
    metric: str,
    title: str,
    sample_label: str,
    save_to_filename: str,
    save_dpi: int = 150,
) -> Optional[str]:
    """Plot one technical-RNA fraction against bundled references."""

    import matplotlib.pyplot as plt
    import pandas as pd

    rows = _reference_qc_fraction_rows()
    if not rows:
        return None
    ref = pd.DataFrame(rows)
    sample = _sample_qc_fractions(df_gene_expr)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = {
        "TCGA cohort medians": "#3f6fb5",
        "HPA normal tissues": "#3f8f5f",
    }
    sample_val = float(sample.get(metric) or 0.0)
    all_ref_vals = ref[metric].astype(float)
    ref_min = float(all_ref_vals.min()) if len(all_ref_vals) else 0.0
    ref_max = float(all_ref_vals.max()) if len(all_ref_vals) else 0.0
    degenerate_ref = bool((ref_max - ref_min) <= 1e-6)
    if degenerate_ref:
        ref_val = ref_min
        ax.axvline(
            ref_val,
            color="#777777",
            linestyle="-",
            linewidth=5,
            alpha=0.65,
            label=f"all bundled references ({ref_val:.0%})",
        )
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.text(
            0.02,
            0.78,
            "Bundled reference columns are degenerate for this metric.\n"
            "This is usually an annotation-limit signal: TCGA cohort medians\n"
            "and HPA normal tissue columns mostly omit nuclear rRNA /\n"
            "rRNA-pseudogene features, so a histogram would collapse to one bin.",
            transform=ax.transAxes,
            fontsize=9,
            color="#444444",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#f7f3e8",
                "edgecolor": "#d4c8a8",
                "linewidth": 0.7,
            },
        )
    else:
        for source, group in ref.groupby("source"):
            vals = group[metric].astype(float)
            ax.hist(
                vals,
                bins=22,
                alpha=0.55,
                color=colors.get(source, "#777777"),
                label=f"{source} (n={len(vals)})",
            )
    ax.axvline(
        sample_val,
        color="#111111",
        linestyle=(0, (6, 3)),
        linewidth=3,
        label=f"sample ({sample_val:.0%})",
    )
    ax.set_xlabel("Fraction of total TPM")
    ax.set_ylabel("Reference columns" if not degenerate_ref else "")
    ax.xaxis.set_major_formatter(lambda x, _pos: f"{x:.0%}")
    xmax = max(ref_max, sample_val, 0.01)
    ax.set_xlim(0, min(1.0, xmax * 1.15 + 0.01))
    ax.set_title(title, loc="left", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "Reference context uses TCGA cohort medians and HPA normal tissue columns; "
        f"sample marker uses {sample_label} present in the input table.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    return save_to_filename


def plot_reference_mtdna_fraction_qc(
    df_gene_expr,
    save_to_filename: str,
    save_dpi: int = 150,
) -> Optional[str]:
    """Plot sample mtDNA fraction against bundled references."""

    return _plot_reference_technical_rna_metric_qc(
        df_gene_expr,
        metric="mt_dna_fraction",
        title="mtDNA fraction vs bundled reference columns",
        sample_label="all MT-* genes",
        save_to_filename=save_to_filename,
        save_dpi=save_dpi,
    )


def plot_reference_nuclear_rrna_fraction_qc(
    df_gene_expr,
    save_to_filename: str,
    save_dpi: int = 150,
) -> Optional[str]:
    """Plot sample nuclear rRNA-like fraction against bundled references."""

    return _plot_reference_technical_rna_metric_qc(
        df_gene_expr,
        metric="nuclear_rrna_like_fraction",
        title="Nuclear rRNA-like fraction vs bundled reference columns",
        sample_label="nuclear rRNA-like non-pseudogene features",
        save_to_filename=save_to_filename,
        save_dpi=save_dpi,
    )


def plot_reference_rrna_pseudogene_fraction_qc(
    df_gene_expr,
    save_to_filename: str,
    save_dpi: int = 150,
) -> Optional[str]:
    """Plot sample rRNA-pseudogene fraction against bundled references."""

    return _plot_reference_technical_rna_metric_qc(
        df_gene_expr,
        metric="rrna_pseudogene_fraction",
        title="rRNA-pseudogene fraction vs bundled reference columns",
        sample_label="rRNA-pseudogene features",
        save_to_filename=save_to_filename,
        save_dpi=save_dpi,
    )


def plot_reference_technical_rna_burden_qc(
    df_gene_expr,
    save_to_filename: str,
    save_dpi: int = 150,
) -> Optional[str]:
    """Plot combined mtDNA+rRNA-like fraction against bundled references."""

    import matplotlib.pyplot as plt
    import pandas as pd

    rows = _reference_qc_fraction_rows()
    if not rows:
        return None
    ref = pd.DataFrame(rows)
    sample = _sample_qc_fractions(df_gene_expr)
    metric = "rrna_plus_mt_fraction"
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = {
        "TCGA cohort medians": "#3f6fb5",
        "HPA normal tissues": "#3f8f5f",
    }
    for source, group in ref.groupby("source"):
        vals = group[metric].astype(float)
        ax.hist(
            vals,
            bins=22,
            alpha=0.55,
            color=colors.get(source, "#777777"),
            label=f"{source} (n={len(vals)})",
        )
    sample_val = float(sample.get(metric) or 0.0)
    ax.axvline(
        sample_val,
        color="#111111",
        linestyle=(0, (6, 3)),
        linewidth=3,
        label=f"sample ({sample_val:.0%})",
    )
    ax.set_xlabel("mtDNA + rRNA-like fraction of total TPM")
    ax.set_ylabel("Reference columns")
    ax.xaxis.set_major_formatter(lambda x, _pos: f"{x:.0%}")
    xmax = max(float(ref[metric].max()), sample_val, 0.01)
    ax.set_xlim(0, min(1.0, xmax * 1.12 + 0.01))
    ax.set_title("Combined technical-RNA burden", loc="left", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "Reference context uses TCGA cohort medians and HPA normal tissue columns; "
        "the sample marker uses every mtDNA/rRNA-like feature present in the input table.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    return save_to_filename


def plot_degradation_index(
    df_gene_expr,
    sample_context: SampleContext,
    save_to_filename: str,
    save_dpi: int = 150,
) -> Optional[str]:
    """Scatter of expected vs observed long/short pair ratios (#27).

    One point per gene pair from ``data/degradation-gene-pairs.csv``:
    x = expected (fresh-tissue calibrated) ratio, y = observed in this
    sample. Diagonal = no degradation. Points below the diagonal flag
    preferential loss of long transcripts. The plot annotates the
    median observed/expected ratio and the severity call from the
    sample context, so users can see whether the call is driven by a
    systematic shift across pairs or a few outliers.

    Returns the filename on success, ``None`` when no pair has the
    short-gene expressed (``s_tpm > 1``) so the plot would be empty.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from pirlygenes.gene_sets_cancer import degradation_gene_pairs

    tpm_by_symbol = _build_tpm_by_symbol(df_gene_expr)

    expected_vals = []
    observed_vals = []
    labels = []
    for short_sym, long_sym, expected in degradation_gene_pairs():
        s = tpm_by_symbol.get(short_sym)
        long_tpm = tpm_by_symbol.get(long_sym)
        if s is None or not (s > 1):
            continue
        if long_tpm is None or expected <= 0:
            continue
        expected_vals.append(float(expected))
        observed_vals.append(float(long_tpm) / float(s))
        labels.append(f"{short_sym}/{long_sym}")

    if not expected_vals:
        return None

    expected_arr = np.array(expected_vals)
    observed_arr = np.array(observed_vals)
    deviation = observed_arr / np.maximum(expected_arr, 1e-6)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Diagonal guide.
    diag_lo = 0.0
    diag_hi = max(expected_arr.max(), observed_arr.max()) * 1.1
    ax.plot(
        [diag_lo, diag_hi],
        [diag_lo, diag_hi],
        linestyle="--",
        color="#888888",
        linewidth=0.8,
        label="expected = observed (no degradation)",
    )

    sc = ax.scatter(
        expected_arr,
        observed_arr,
        c=np.log2(np.maximum(deviation, 1e-3)),
        cmap="RdYlGn",
        edgecolors="black",
        linewidth=0.3,
        s=60,
    )
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("log2(observed / expected)", fontsize=9)

    ax.set_xlabel("Expected long/short ratio (fresh-tissue calibration)", fontsize=10)
    ax.set_ylabel("Observed long/short ratio (this sample)", fontsize=10)
    ax.set_xlim(diag_lo, diag_hi)
    ax.set_ylim(diag_lo, diag_hi)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    title_parts = [
        f"Length-pair ratio check — {length_pair_display_label(sample_context)}"
    ]
    title_parts.append(f"{len(expected_vals)} pairs")
    ax.set_title(" · ".join(title_parts), fontsize=11, loc="left")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    return save_to_filename
