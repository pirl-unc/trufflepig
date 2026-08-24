"""Non-HK views of a biological signal, with reasoning over the views.

Any signal (a lineage marker panel, a pathway, a background/cluster score) is
represented in four complementary clean-TPM views. The decision uses two
independent biological questions: absolute program burden (``log1p``) and
specificity versus other cancers (``cohort_pct``). The remaining views diagnose
why those signals disagree without manufacturing extra votes.

The four views
--------------
1. ``within_pct`` — median(within-sample percentile rank of the marker).
                    "How dominant within this transcriptome." Purity/dominance;
                    TME-contaminated for non-tumour-intrinsic panels.
2. ``log1p``      — median(log1p(clean TPM)). "Absolute expression, compressed."
                    Tightest within-class spread → best for a threshold gate.
3. ``cohort_pct`` — median(fraction of reference cohorts below this sample).
                    "How high vs other cancer types." Specificity; bounded [0,1].
4. ``cohort_z``   — median((value − cohort mean) / cohort sd). "How many SDs above
                    the cross-cohort background." Outlier-ness vs background.

Reasoning
---------
Each decision view is mapped to a 0–1 presence score. Their mean is the program
confidence and their agreement is its concordance. Diagnostic views are exposed
and can explain admixture, but do not independently change the lineage call.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np

VIEW_NAMES = ("within_pct", "log1p", "cohort_pct", "cohort_z")
DECISION_VIEW_NAMES = ("log1p", "cohort_pct")
PRESENT_CONFIDENCE = 0.6
ABSENT_CONFIDENCE = 0.35


@functools.lru_cache(maxsize=1)
def _cohort_reference():
    """{gene_symbol: np.array of per-cohort TCGA values} — the cross-cohort
    background distribution used by the cohort_pct / cohort_z views."""
    from .reference import pan_cancer_expression

    ref = pan_cancer_expression().drop_duplicates(subset="Symbol")
    cohort_cols = [c for c in ref.columns if c.endswith("_TPM")]
    mat = ref[cohort_cols].to_numpy(dtype=float)
    return {str(s): mat[i] for i, s in enumerate(ref["Symbol"])}


def _median(values):
    arr = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(arr)) if arr else float("nan")


@dataclass
class SignalViews:
    """The four normalized views of one signal, plus the reasoning verdict."""

    name: str
    views: dict[str, float]
    presence: dict[str, float]          # each view mapped to 0–1 presence
    concordance: float                  # mean pairwise agreement of the presence votes
    call: str                           # "present" / "ambiguous" / "absent"
    confidence: float                   # concordance-weighted mean presence
    flags: list[str] = field(default_factory=list)

    def as_row(self):
        return {n: self.views.get(n, float("nan")) for n in VIEW_NAMES}


def compute_views(
    name,
    genes,
    sample_tpm_by_symbol,
    *,
    within_sample_pct: dict | None = None,
    cohort_reference: dict | None = None,
) -> dict:
    """Compute the four non-HK views for one gene panel."""
    ref = cohort_reference if cohort_reference is not None else _cohort_reference()
    if within_sample_pct is None:
        syms = list(sample_tpm_by_symbol)
        vals = np.asarray([sample_tpm_by_symbol[s] for s in syms], float)
        from scipy.stats import rankdata

        within_sample_pct = dict(zip(syms, rankdata(vals, method="average") / len(vals)))

    def g_tpm(g):
        return float(sample_tpm_by_symbol.get(g, 0.0))

    wp_v = _median([within_sample_pct.get(g, 0.0) for g in genes])
    lg_v = _median([np.log1p(g_tpm(g)) for g in genes])
    cp_v, cz_v = [], []
    for g in genes:
        dist = ref.get(g)
        if dist is None or len(dist) == 0:
            continue
        cp_v.append(float((dist < g_tpm(g)).mean()))
        sd = dist.std()
        cz_v.append(float((g_tpm(g) - dist.mean()) / sd) if sd > 0 else 0.0)
    return {
        "within_pct": wp_v,
        "log1p": lg_v,
        "cohort_pct": _median(cp_v),
        "cohort_z": _median(cz_v),
    }


# View-specific calibration to a 0–1 "presence". Anchors are robust midpoints
# observed across the 109-rep + 18-local sweeps (carcinoma/sarcoma, NE/non-NE):
# each maps its "clearly present" level to ~0.8 and "clearly absent" to ~0.2.
_PRESENCE_ANCHORS = {
    "within_pct": (0.55, 0.92),  # within-sample rank
    "log1p": (1.50, 5.50),       # log1p(TPM)
    "cohort_pct": (0.30, 0.85),  # cross-cohort percentile
    "cohort_z": (0.00, 2.00),    # SDs above cohort background
}


def _to_presence(view, value):
    if value is None or not np.isfinite(value):
        return None
    lo, hi = _PRESENCE_ANCHORS[view]
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def reason_over_views(name, views) -> SignalViews:
    """Integrate the two decision views; retain the others as diagnostics."""
    presence = {v: _to_presence(v, views.get(v)) for v in VIEW_NAMES}
    votes = [presence[v] for v in DECISION_VIEW_NAMES if presence[v] is not None]
    confidence = float(np.mean(votes)) if votes else 0.0
    # concordance = 1 - mean pairwise absolute disagreement
    if len(votes) >= 2:
        diffs = [abs(a - b) for i, a in enumerate(votes) for b in votes[i + 1 :]]
        concordance = float(1.0 - np.mean(diffs))
    else:
        concordance = 1.0
    call = (
        "present"
        if confidence >= PRESENT_CONFIDENCE
        else ("absent" if confidence <= ABSENT_CONFIDENCE else "ambiguous")
    )

    flags = []
    cz, wp = presence.get("cohort_z"), presence.get("within_pct")
    cp = presence.get("cohort_pct")
    if cz is not None and wp is not None and cz - wp >= 0.4:
        flags.append("high vs background but not dominant in-sample — admixture / low purity")
    if wp is not None and cp is not None and wp - cp >= 0.4:
        flags.append("dominant in-sample but not cohort-specific — possible background-high genes")
    if concordance < 0.6:
        flags.append("views disagree — treat the call as provisional")
    return SignalViews(
        name=name, views=views, presence=presence,
        concordance=concordance, call=call, confidence=confidence, flags=flags,
    )


def signal_report(name, genes, sample_tpm_by_symbol, **kw) -> SignalViews:
    """Convenience: compute the four views for a panel and reason over them."""
    views = compute_views(name, genes, sample_tpm_by_symbol, **kw)
    return reason_over_views(name, views)
