"""Lineage-routed tumor expression decomposition.

A single ``tumor + stroma + immune`` split is *unidentifiable* when the tumor's own
lineage IS one of the backgrounds: a lymphoma is collinear with the immune template,
a sarcoma with the stromal template (measured cosines 0.67 / 0.64 over signature-gene
space). So we run lineage-routed *modes*, each subtracting only the backgrounds that
are genuinely non-tumor for that lineage:

==============  =========================  ================================
mode            tumor lineage              subtract (non-tumor backgrounds)
==============  =========================  ================================
``solid``       epithelial / other solid   immune + stromal
``mesenchymal`` mesenchymal (sarcoma)      immune + epithelial-normal
``heme``        hematopoietic              epithelial-normal + stromal
==============  =========================  ================================

**Routing.** Picking the mode from the decomposition alone is ~as hard as classifying
the tumor (it *is* the identifiability problem), so we route with the global cancer
call instead:

* an explicit ``cancer`` hint (family / type / group / molecular-subtype) →
  :func:`resolve_mode` (100% on the validation set), or
* trufflepig's :func:`~trufflepig.cancer_type_centroid.compartment_call`
  (52/54 = 0.96 overall; 50/50 = 1.00 on *confident* calls). When the compartment call
  is not confident, we run only the top + runner-up candidate modes and resolve within
  them by the residual lineage signal (the two non-confident misses on the validation
  set both had the correct mode as the runner-up).

Background subtraction is NNLS over HPA normal cell-type templates; the **tumor-specific
residual** is then scored for a malignancy hallmark orthogonal to lineage —
proliferation — which separates tumor from same-lineage normal where the lineage
signature cannot (it is *not* a mode-selector, only a within-mode content metric).

Signature scoring defaults to within-sample **percentile rank** (empirically the most
discrimination- and purity-robust space; ssGSEA and z-score also supported). Inputs are
``{symbol: clean-TPM}`` — the same contract as ``compartment_call``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import numpy as np
import pandas as pd

# --- malignancy / lineage gene sets (symbol space) ---
PROLIFERATION = ("MKI67","CCNB1","CCNB2","CDK1","TOP2A","BUB1","AURKA","AURKB","PLK1","CENPA",
                 "CENPE","FOXM1","UBE2C","BIRC5","TYMS","RRM2","CDC20","KIF11","NUSAP1","TPX2")
EPITHELIAL = ("EPCAM","KRT8","KRT18","KRT19","KRT7","CDH1","CLDN3","CLDN4","CLDN7","ELF3","GRHL2","ESRP1","KRT5","KRT14")
_STROMAL_CT = ("Fibroblasts","Endothelial cells","Smooth muscle cells","Adipocytes","Mesothelial cells")
_IMMUNE_CT = ("T-cells","B-cells","NK-cells","Macrophages","monocytes","dendritic cells","granulocytes","Plasma cells")
_EPITHELIAL_CT = ("Basal keratinocytes","Suprabasal keratinocytes","Squamous epithelial cells","Glandular and luminal cells",
                  "Breast glandular cells","Prostatic glandular cells","Alveolar cells type 1","Alveolar cells type 2",
                  "Club cells","Hepatocytes","Cholangiocytes","Distal enterocytes","Proximal enterocytes",
                  "Secretory cells","Exocrine glandular cells")

# which backgrounds each mode subtracts (the rest, incl. the tumor lineage, stays in the residual)
MODE_SUBTRACT = {"solid": ("immune", "stromal"),
                 "mesenchymal": ("immune", "epithelial"),
                 "heme": ("epithelial", "stromal")}
# the tumor-lineage signature expected to SURVIVE subtraction in each mode (used to resolve ambiguity)
MODE_TUMOR_SIG = {"solid": "epithelial", "mesenchymal": "stromal", "heme": "immune"}

_MOL_SUBTYPE_NOTE = {
    "MSI": "MSI / dMMR — high immune infiltrate is real biology, not contamination; expect an elevated immune fraction.",
    "POLE": "POLE-ultramutated — immune-hot; treat high immune as expected, not contamination.",
    "HPVPOS": "HPV-positive — typically immune-infiltrated.",
}


@lru_cache(maxsize=1)
def _refs():
    """Symbol-keyed normal cell-type templates + signature symbol sets (cached)."""
    from pirlygenes.expression.accessors import hpa_cell_type_expression, estimate_signatures

    hpa = hpa_cell_type_expression().drop_duplicates("Symbol").set_index("Symbol")

    def template(cell_types):
        cols = [c for c in cell_types if c in hpa.columns]
        v = hpa[cols].mean(axis=1).clip(lower=0)
        total = float(v.sum()) or 1.0
        return v / total * 1e6

    templates = {"immune": template(_IMMUNE_CT),
                 "stromal": template(_STROMAL_CT),
                 "epithelial": template(_EPITHELIAL_CT)}
    est = estimate_signatures()
    signatures = {
        "immune": frozenset(est.loc[est["Category"] == "Immune", "Symbol"]),
        "stromal": frozenset(est.loc[est["Category"] == "Stromal", "Symbol"]),
        "epithelial": frozenset(EPITHELIAL),
        "proliferation": frozenset(PROLIFERATION),
    }
    return templates, signatures


def _group_to_mode(group: str) -> str:
    """Map a lineage group / compartment label to a decomposition mode."""
    g = (group or "").lower()
    if any(k in g for k in ("heme", "hema", "lymph", "myel", "leuk")):
        return "heme"
    if "sarcoma" in g or "mesench" in g:
        return "mesenchymal"
    return "solid"  # Epithelial, Melanoma, CNS, Neuroendocrine, Embryonal, Germ cell, …


def _candidates(code: str):
    parts = code.split("_")
    for k in range(len(parts), 0, -1):
        yield "_".join(parts[:k])


def resolve_mode(cancer):
    """Resolve a ``cancer`` hint (family / type / group / molecular-subtype) to a mode.

    Returns ``(mode_or_None, routing_explanation)``. Molecular-subtype suffixes
    (``CRC_MSI``) are progressively stripped until the lineage group resolves.
    """
    if not cancer:
        return None, "no hint"
    text = str(cancer).strip()
    fam = text.lower()
    if fam in {"solid", "epithelial", "carcinoma", "melanoma", "cns", "neuroendocrine", "embryonal", "germ cell"}:
        return "solid", f"family={text}"
    if fam in {"mesenchymal", "sarcoma"}:
        return "mesenchymal", f"family={text}"
    if fam in {"heme", "hematologic", "hematopoietic", "blood", "lymphoid", "myeloid", "leukemia", "lymphoma"}:
        return "heme", f"family={text}"
    try:
        from pirlygenes.gene_sets_cancer import cancer_lineage_group
    except Exception:
        return None, f"{text!r}: no lineage resolver"
    for cand in _candidates(text):
        try:
            group = cancer_lineage_group(cand)
        except Exception:
            group = None
        if group:
            return _group_to_mode(group), f"{text} → lineage_group={group} (via {cand})"
    return None, f"{text!r} unresolved"


def _safe(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _score(sample: pd.Series, gene_symbols, space: str, ref=None) -> float:
    """Enrichment of a gene set in the chosen space (percentile / ssgsea / zscore)."""
    present = [g for g in gene_symbols if g in sample.index]
    if not present:
        return float("nan")
    if space == "percentile":
        return float(sample.rank(pct=True).loc[present].mean() * 100.0)
    if space == "ssgsea":  # Barbie 2009 single-sample GSEA enrichment (rank-weighted)
        x = sample.values
        order = np.argsort(x)[::-1]
        in_set = sample.index.isin(set(present))[order]
        w = np.power(np.abs(x[order]) + 1e-9, 0.25) * in_set
        s = w.sum()
        if s == 0:
            return 0.0
        p_in = np.cumsum(w) / s
        p_out = np.cumsum(~in_set) / max(int((~in_set).sum()) or 1, 1)
        return float(np.sum(p_in - p_out))
    if space == "zscore" and ref is not None:
        z = (np.log1p(sample) - ref["mean"]).div(ref["std"].replace(0, 1))
        return float(_safe(z).loc[present].mean())
    return float(np.log1p(sample).loc[present].mean())  # fallback


def _collinearity(templates, genes) -> float:
    """Max pairwise |cosine| of the subtracted templates over ``genes`` (identifiability)."""
    if len(templates) < 2:
        return 0.0
    m = np.log1p(np.column_stack([t.reindex(genes).fillna(0.0).values for t in templates]))
    m = m - m.mean(axis=0)
    norms = np.linalg.norm(m, axis=0)
    norms[norms == 0] = 1.0
    corr = (m.T @ m) / np.outer(norms, norms)
    return float(np.abs(corr[np.triu_indices(len(templates), 1)]).max())


def _run_mode(mode, sample, templates, signatures, space, ref):
    subs = MODE_SUBTRACT[mode]
    fit = sorted(set().union(*[signatures[s] for s in subs]) & set(sample.index))
    if not fit:
        return {"purity": float("nan"), "note": "no fit genes"}
    from scipy.optimize import nnls

    a = np.column_stack([templates[s].reindex(fit).fillna(0.0).values for s in subs])
    f, _ = nnls(a, sample.reindex(fit).fillna(0.0).values)
    recon = sum(f[i] * templates[subs[i]] for i in range(len(subs)))
    residual = (sample - recon.reindex(sample.index).fillna(0.0)).clip(lower=0)
    total = float(sample.sum()) or 1.0
    return {
        "purity": round(float(residual.sum() / total), 3),
        "subtracted": {subs[i]: round(float(f[i] * templates[subs[i]].sum() / 1e6), 3) for i in range(len(subs))},
        "residual_proliferation": round(_score(residual, signatures["proliferation"], space, ref), 2),
        "tumor_lineage_in_residual": round(_score(residual, signatures[MODE_TUMOR_SIG[mode]], space, ref), 2),
        "immune_score": round(_score(sample, signatures["immune"], space, ref), 2),
        "stromal_score": round(_score(sample, signatures["stromal"], space, ref), 2),
        "template_collinearity": round(_collinearity([templates[s] for s in subs], fit), 3),
    }


def decompose_expression(sample_tpm_by_symbol: Mapping[str, float], cancer=None, *,
                         space: str = "percentile", route_via_classifier: bool = True,
                         run_all: bool = False, ref=None) -> dict:
    """Lineage-routed tumor/background decomposition of one sample.

    Parameters
    ----------
    sample_tpm_by_symbol
        ``{HGNC symbol: clean-TPM}`` for the sample (the ``compartment_call`` contract).
    cancer
        Optional lineage prior — family (``"epithelial"`` / ``"sarcoma"`` / ``"heme"``),
        a type (``"PRAD"``), a group (``"CRC"``), or a molecular subtype (``"CRC_MSI"``).
        When omitted and ``route_via_classifier`` is set, the mode is routed by
        :func:`~trufflepig.cancer_type_centroid.compartment_call`.
    space
        Signature-scoring space: ``"percentile"`` (default), ``"ssgsea"`` or ``"zscore"``
        (the last needs ``ref={"mean":Series,"std":Series}`` over ``log1p`` expression).

    Returns a dict with the selected mode, routing/confidence, the classifier call (if
    used), a molecular-subtype interpretation note, and per-mode metrics (purity,
    subtracted fractions, residual proliferation, tumor-lineage-in-residual,
    template collinearity).
    """
    templates, signatures = _refs()
    sample = _safe(pd.Series(dict(sample_tpm_by_symbol), dtype=float))

    mode, routing = resolve_mode(cancer)
    classifier = None
    candidates = None
    if mode is None and route_via_classifier:
        try:
            from .cancer_type_centroid import compartment_call

            call = compartment_call(dict(sample_tpm_by_symbol))
            classifier = {"compartment": call.get("compartment"), "confident": bool(call.get("confident")),
                          "margin": round(float(call.get("margin", 0.0)), 4), "runner_up": call.get("runner_up")}
            top = _group_to_mode(call.get("compartment", ""))
            if call.get("confident"):
                mode = top
                routing = f"compartment_call={call['compartment']} (confident, margin {classifier['margin']}) → {top}"
            else:  # narrow to top + runner-up, resolve within them by residual lineage signal
                run = _group_to_mode(call.get("runner_up", "")) if call.get("runner_up") else None
                candidates = list(dict.fromkeys([m for m in (top, run) if m]))
        except Exception as exc:  # noqa: BLE001
            classifier = {"error": str(exc)[:160]}

    if mode is not None and not run_all:
        modes = [mode]
    elif candidates and not run_all:
        modes = candidates
    else:
        modes = ["solid", "mesenchymal", "heme"]

    out = {m: _run_mode(m, sample, templates, signatures, space, ref) for m in modes}

    if mode is not None:
        selected, confidence = mode, "routed: " + routing
    elif candidates:
        selected = max(candidates, key=lambda m: out[m].get("tumor_lineage_in_residual", float("-inf")))
        confidence = (f"ambiguous compartment ({classifier['compartment']} vs {classifier['runner_up']}, "
                      f"margin {classifier['margin']}) → {selected} by residual lineage signal — verify")
    elif out:
        selected = max(out, key=lambda m: out[m].get("tumor_lineage_in_residual", float("-inf")))
        confidence = "low (data-only fallback; no hint and classifier unavailable)"
    else:
        selected, confidence = None, "no decomposition produced"

    note = next((v for k, v in _MOL_SUBTYPE_NOTE.items() if str(cancer or "").upper().endswith(k)), None)
    return {"selected_mode": selected, "routing": routing, "confidence": confidence,
            "classifier": classifier, "space": space, "subtype_note": note, "modes": out}
