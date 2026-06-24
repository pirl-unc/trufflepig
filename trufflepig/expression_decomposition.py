"""Lineage-routed tumor expression decomposition + tumor sub-population characterization.

A single ``tumor + stroma + immune`` split is *unidentifiable* when the tumor's own
lineage IS one of the backgrounds (a lymphoma is collinear with the immune template, a
sarcoma with the stromal template). So we run lineage-routed *modes*, each subtracting
only the backgrounds that are genuinely non-tumor for that lineage, then characterize
the tumor-specific residual with malignancy signals that are orthogonal to lineage.

=================================  ==============================================================
WHAT THE PIPELINE DOES             SUMMARY
=================================  ==============================================================
**Input**                          ``{HGNC symbol: clean-TPM}`` for one sample (the
                                   ``compartment_call`` contract). Optional: a ``cancer`` lineage
                                   prior (family / type / group / molecular-subtype) and a
                                   ``met_sites`` list (off by default).

**Templates** (HPA normal cells)   immune (+ sub-lineages b_cell / t_cell / myeloid / plasma),
                                   stromal, epithelial-normal, and met-site organs
                                   (liver / lung / brain / marrow — used only when evidenced).

**Routing decision**               explicit ``cancer`` hint → :func:`resolve_mode`; else the global
                                   ``compartment_call``. Confident → one mode; not-confident → run
                                   top+runner-up candidate modes and resolve by residual lineage
                                   signal. Picks one of 4 modes:

                                     ``solid``       subtract immune + stromal
                                     ``mesenchymal`` subtract immune + epithelial
                                     ``heme``        subtract epithelial + stromal + *healthy*
                                                     immune sub-lineages (keep the malignant one)
                                     ``embryonal``   subtract immune only (tumor is polyphenotypic)

**Background subtraction**         NNLS over the mode's templates → tumor-specific residual +
                                   per-population fractions → ``decomposition_purity`` = residual
                                   mass fraction (the lineage-aware purity that fixes ESTIMATE's
                                   heme/sarcoma blind spots).

**Signals on the residual**        proliferation (cell-cycle), aneuploidy
                                   (:mod:`trufflepig.aneuploidy_axis`, with the gained/lost arms),
                                   and — for heme — the malignant-vs-healthy immune sub-lineage
                                   split. These are malignancy hallmarks *orthogonal to lineage*,
                                   so they separate tumor from same-lineage normal where the
                                   lineage signature cannot.

**Output**                         ``selected_mode`` + routing/confidence, a ``purity`` block,
                                   a ``tumor_characteristics`` block (proliferative? aneuploid?
                                   which arms? heme sub-lineage?), per-mode metrics, and an
                                   identifiability gate (template collinearity).
=================================  ==============================================================

**Interaction with purity estimation.** Three purity views triangulate:
  * ``decomposition_purity`` (residual fraction) — lineage-aware; works where ESTIMATE breaks.
  * the bulk ESTIMATE/lineage purity (``tumor_purity.estimate_tumor_purity``) — cross-check;
    large disagreement flags a heme/mesenchymal sample ESTIMATE mis-handles.
  * **aneuploidy amplitude is itself purity-scaled** — an arm gain in a 50%-pure sample shows a
    half-strength shift, so the aneuploidy signal is an orthogonal, CNA-based purity corroborator
    (strong aneuploidy ⇒ real tumor content; aneuploid-but-low-purity ⇒ possible over-subtraction).
  Crucially, proliferation and aneuploidy are measured on the **residual** (purity-corrected),
  not the bulk — the decomposition is what makes the tumor-intrinsic values readable.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# --- malignancy / lineage signatures (symbol space) ---
PROLIFERATION = ("MKI67","CCNB1","CCNB2","CDK1","TOP2A","BUB1","AURKA","AURKB","PLK1","CENPA",
                 "CENPE","FOXM1","UBE2C","BIRC5","TYMS","RRM2","CDC20","KIF11","NUSAP1","TPX2")
EPITHELIAL = ("EPCAM","KRT8","KRT18","KRT19","KRT7","CDH1","CLDN3","CLDN4","CLDN7","ELF3","GRHL2","ESRP1","KRT5","KRT14")
_IMMUNE_SUBLINEAGE_MARKERS = {
    "b_cell": ("CD19","MS4A1","CD79A","CD79B","PAX5","CD22"),
    "t_cell": ("CD3D","CD3E","CD3G","CD2","TRAC","CD8A","CD4","NKG7","KLRD1"),
    "myeloid": ("CD14","CD68","LYZ","CSF1R","ITGAM","FCGR3A","MPO","ELANE"),
    "plasma": ("SDC1","MZB1","XBP1","PRDM1","TNFRSF17","DERL3"),
}
_MET_SITE_MARKERS = {
    "liver": ("ALB","APOB","SERPINA1","TF","APOA1","FGA","CYP2E1"),
    "lung": ("SFTPB","SFTPC","SFTPA1","NAPSA","SCGB1A1","AGER"),
    "brain": ("GFAP","MBP","SNAP25","SYT1","NEFL","PLP1"),
    "marrow": ("HBB","HBA1","GYPA","ALAS2","HBA2"),
}
# HPA cell-type columns backing each expression template
_CT = {
    "immune": ("T-cells","B-cells","NK-cells","Macrophages","monocytes","dendritic cells","granulocytes","Plasma cells"),
    "b_cell": ("B-cells",),
    "t_cell": ("T-cells","NK-cells"),
    "myeloid": ("Macrophages","monocytes","dendritic cells","granulocytes"),
    "plasma": ("Plasma cells",),
    "stromal": ("Fibroblasts","Endothelial cells","Smooth muscle cells","Adipocytes","Mesothelial cells"),
    "epithelial": ("Basal keratinocytes","Suprabasal keratinocytes","Squamous epithelial cells","Glandular and luminal cells",
                   "Breast glandular cells","Prostatic glandular cells","Alveolar cells type 1","Alveolar cells type 2",
                   "Club cells","Hepatocytes","Cholangiocytes","Distal enterocytes","Proximal enterocytes",
                   "Secretory cells","Exocrine glandular cells"),
    "liver": ("Hepatocytes","Cholangiocytes"),
    "lung": ("Alveolar cells type 1","Alveolar cells type 2","Club cells","Ciliated cells"),
    "brain": ("Excitatory neurons","Inhibitory neurons","Astrocytes","Oligodendrocytes"),
    "marrow": ("Erythroid cells",),
}
_SUBLINEAGES = ("b_cell", "t_cell", "myeloid", "plasma")
# malignant immune sub-lineage by heme type (when the type is known)
HEME_MALIGNANT = {
    "DLBC": "b_cell", "FL": "b_cell", "MCL": "b_cell", "BL": "b_cell", "CLL": "b_cell", "HL": "b_cell",
    "MM": "plasma", "CTCL": "t_cell",
    "LAML": "myeloid", "CML": "myeloid", "MDS": "myeloid", "MPN": "myeloid",
}
MODE_SUBTRACT = {"solid": ("immune", "stromal"),
                 "mesenchymal": ("immune", "epithelial"),
                 "heme": ("epithelial", "stromal"),       # + healthy immune sub-lineages, added per-sample
                 "embryonal": ("immune",)}
MODE_TUMOR_SIG = {"solid": "epithelial", "mesenchymal": "stromal", "heme": "immune", "embryonal": "proliferation"}
_MET_PRIMARY_LINEAGE = {"liver": {"LIHC", "HEPB", "CHOL"}, "lung": {"LUAD", "LUSC", "SCLC", "MESO"},
                        "brain": {"GBM", "LGG"}, "marrow": set()}  # marrow guarded by heme mode
_MOL_SUBTYPE_NOTE = {"MSI": "MSI/dMMR — high immune infiltrate is real biology, not contamination.",
                     "POLE": "POLE-ultramutated — immune-hot; treat high immune as expected.",
                     "HPVPOS": "HPV-positive — typically immune-infiltrated."}
_ANEUPLOIDY_STRONG = 0.20
_PROLIFERATIVE_PCTL = 60.0
_ANEUPLOIDY_NOISE_FLOOR = 0.10  # bulk-aneuploidy MAD floor; subtract before reading purity off it

# Curated positive tumor-lineage signatures — compartment-aware residual validation, so a
# melanoma/NE/embryonal tumor is confirmed against ITS markers, not the generic mode signature.
MELANOMA = ("MLANA", "PMEL", "TYR", "MITF", "DCT", "TYRP1", "SOX10")
NEUROENDOCRINE = ("CHGA", "CHGB", "SYP", "INSM1", "NCAM1", "ASCL1")
EMBRYONAL = ("LIN28A", "LIN28B", "SALL4", "SOX2", "POU5F1", "DPPA4", "MYCN", "UTF1")
CNS_GLIAL = ("GFAP", "OLIG1", "OLIG2", "SOX2", "MBP", "S100B")
GERM = ("DDX4", "SALL4", "POU5F1", "NANOG", "KIT", "DPPA4")
COMPARTMENT_SIG = {"Epithelial": "epithelial", "Melanoma": "melanoma", "Neuroendocrine": "neuroendocrine",
                   "Embryonal": "embryonal", "CNS": "cns", "Germ cell": "germ", "Heme": "immune", "Sarcoma": "stromal"}
_FAMILY_COMPARTMENT = {"solid": "Epithelial", "epithelial": "Epithelial", "carcinoma": "Epithelial",
                       "melanoma": "Melanoma", "cns": "CNS", "neuroendocrine": "Neuroendocrine", "germ cell": "Germ cell",
                       "mesenchymal": "Sarcoma", "sarcoma": "Sarcoma", "heme": "Heme", "hematologic": "Heme",
                       "lymphoid": "Heme", "myeloid": "Heme", "leukemia": "Heme", "lymphoma": "Heme",
                       "embryonal": "Embryonal", "rhabdoid": "Embryonal"}


@lru_cache(maxsize=1)
def _refs():
    """Symbol-keyed templates + signature symbol sets (cached)."""
    from pirlygenes.expression.accessors import hpa_cell_type_expression, estimate_signatures

    hpa = hpa_cell_type_expression().drop_duplicates("Symbol").set_index("Symbol")

    def template(cols):
        present = [c for c in cols if c in hpa.columns]
        v = hpa[present].mean(axis=1).clip(lower=0)
        return v / (float(v.sum()) or 1.0) * 1e6

    templates = {k: template(cols) for k, cols in _CT.items()}
    est = estimate_signatures()
    signatures = {"immune": frozenset(est.loc[est["Category"] == "Immune", "Symbol"]),
                  "stromal": frozenset(est.loc[est["Category"] == "Stromal", "Symbol"]),
                  "epithelial": frozenset(EPITHELIAL), "proliferation": frozenset(PROLIFERATION)}
    for k, m in _IMMUNE_SUBLINEAGE_MARKERS.items():
        signatures[k] = frozenset(m)
    for k, m in _MET_SITE_MARKERS.items():
        signatures[k] = frozenset(m)
    for k, m in {"melanoma": MELANOMA, "neuroendocrine": NEUROENDOCRINE, "embryonal": EMBRYONAL,
                 "cns": CNS_GLIAL, "germ": GERM}.items():
        signatures[k] = frozenset(m)
    # germline/CTA-restricted genes: ~off in all somatic normal cells -> bulk expression ∝ purity.
    germline = ("Early spermatids", "Late spermatids", "Spermatocytes", "Spermatogonia", "Oocytes")
    somatic = [c for c in hpa.columns if c not in germline and c != "Ensembl_Gene_ID"]
    signatures["restricted"] = frozenset(hpa.index[hpa[somatic].max(axis=1) < 5.0])
    return templates, signatures


def _group_to_mode(group: str) -> str:
    g = (group or "").lower()
    if any(k in g for k in ("heme", "hema", "lymph", "myel", "leuk")):
        return "heme"
    if "embryonal" in g or "rhabdoid" in g:
        return "embryonal"
    if "sarcoma" in g or "mesench" in g:
        return "mesenchymal"
    return "solid"


def _candidates(code: str):
    parts = code.split("_")
    for k in range(len(parts), 0, -1):
        yield "_".join(parts[:k])


def resolve_mode(cancer):
    """Resolve a ``cancer`` hint to ``(mode_or_None, routing, type_code_or_None)``."""
    if not cancer:
        return None, "no hint", None
    text = str(cancer).strip()
    fam = text.lower()
    families = {"solid": "solid", "epithelial": "solid", "carcinoma": "solid", "melanoma": "solid",
                "cns": "solid", "neuroendocrine": "solid", "germ cell": "solid",
                "mesenchymal": "mesenchymal", "sarcoma": "mesenchymal",
                "heme": "heme", "hematologic": "heme", "lymphoid": "heme", "myeloid": "heme",
                "leukemia": "heme", "lymphoma": "heme", "embryonal": "embryonal", "rhabdoid": "embryonal"}
    if fam in families:
        return families[fam], f"family={text}", None
    try:
        from pirlygenes.gene_sets_cancer import cancer_lineage_group
    except Exception:
        return None, f"{text!r}: no resolver", None
    for cand in _candidates(text):
        try:
            group = cancer_lineage_group(cand)
        except Exception:
            group = None
        if group:
            return _group_to_mode(group), f"{text} → lineage_group={group} (via {cand})", cand
    return None, f"{text!r} unresolved", None


def _safe(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def signature_score(sample: pd.Series, gene_symbols, space: str = "percentile") -> float:
    """Enrichment of a gene set in a sample, in ``percentile`` (default) or ``ssgsea`` space."""
    present = [g for g in gene_symbols if g in sample.index]
    if not present:
        return float("nan")
    if space == "ssgsea":
        x = sample.values
        order = np.argsort(x)[::-1]
        in_set = sample.index.isin(set(present))[order]
        w = np.power(np.abs(x[order]) + 1e-9, 0.25) * in_set
        s = w.sum()
        if s == 0:
            return 0.0
        return float(np.sum(np.cumsum(w) / s - np.cumsum(~in_set) / max(int((~in_set).sum()) or 1, 1)))
    return float(sample.rank(pct=True).loc[present].mean() * 100.0)  # percentile default


def _collinearity(templates, genes) -> float:
    if len(templates) < 2:
        return 0.0
    m = np.log1p(np.column_stack([t.reindex(genes).fillna(0.0).values for t in templates]))
    m = m - m.mean(axis=0)
    norms = np.linalg.norm(m, axis=0)
    norms[norms == 0] = 1.0
    corr = (m.T @ m) / np.outer(norms, norms)
    return float(np.abs(corr[np.triu_indices(len(templates), 1)]).max())


def _heme_immune(sample, signatures, space, type_code):
    """Determine the malignant immune sub-lineage and the healthy ones to subtract."""
    scores = {sl: signature_score(sample, signatures[sl], space) for sl in _SUBLINEAGES}
    if type_code and type_code in HEME_MALIGNANT:
        malignant, source = HEME_MALIGNANT[type_code], "type-map"
    else:
        malignant = max(scores, key=lambda s: (scores[s] if not np.isnan(scores[s]) else -1)); source = "dominant-marker"
    healthy = [sl for sl in _SUBLINEAGES if sl != malignant]
    return {"malignant_sublineage": malignant, "source": source,
            "sublineage_scores": {k: round(v, 1) for k, v in scores.items()}}, healthy


def _subtract_keys(mode, sample, signatures, space, type_code, met_sites):
    info = {}
    keys = list(MODE_SUBTRACT[mode])
    if mode == "heme":
        info, healthy = _heme_immune(sample, signatures, space, type_code)
        keys += healthy                                   # keep the malignant sub-lineage as tumor
    applied_mets, skipped_mets = [], []
    for site in (met_sites or []):
        if site not in _CT:
            continue
        if type_code and type_code in _MET_PRIMARY_LINEAGE.get(site, set()):
            skipped_mets.append(site)                     # don't subtract a primary's own organ
            continue
        keys.append(site); applied_mets.append(site)
    if met_sites:
        info = {**info, "met_sites_subtracted": applied_mets, "met_sites_skipped_as_primary": skipped_mets}
    return keys, info


def decompose_mode(mode, sample, templates, signatures, space, type_code, met_sites):
    """Run one lineage mode's NNLS background subtraction → (metrics, residual, info)."""
    from scipy.optimize import nnls

    keys, info = _subtract_keys(mode, sample, signatures, space, type_code, met_sites)
    fit = sorted(set().union(*[signatures.get(k, frozenset()) for k in keys]) & set(sample.index))
    if not fit:
        return {"purity": float("nan"), "note": "no fit genes"}, sample, info
    a = np.column_stack([templates[k].reindex(fit).fillna(0.0).values for k in keys])
    f, _ = nnls(a, sample.reindex(fit).fillna(0.0).values)
    recon = sum(f[i] * templates[keys[i]] for i in range(len(keys)))
    residual = (sample - recon.reindex(sample.index).fillna(0.0)).clip(lower=0)
    total = float(sample.sum()) or 1.0
    metrics = {"purity": round(float(residual.sum() / total), 3),
               "subtracted": {keys[i]: round(float(f[i] * templates[keys[i]].sum() / 1e6), 3) for i in range(len(keys))},
               "tumor_lineage_in_residual": round(signature_score(residual, signatures[MODE_TUMOR_SIG[mode]], space), 2),
               "template_collinearity": round(_collinearity([templates[k] for k in keys], fit), 3)}
    return metrics, residual, info


def characterize_residual(residual, *, mode: str = "solid", space: str = "percentile",
                          signatures=None, heme_info=None) -> dict:
    """Tumor sub-population characteristics for a residual (``{symbol: TPM}`` or Series).

    Proliferation (cell-cycle) + aneuploidy (chromosome-arm coherence) + heme sub-lineage.
    Aneuploidy needs :mod:`trufflepig.aneuploidy_axis` (pyensembl); if it is unavailable it
    degrades gracefully (``aneuploidy_score=None``) rather than raising.
    """
    if signatures is None:
        _, signatures = _refs()
    if not isinstance(residual, pd.Series):
        residual = _safe(pd.Series(dict(residual), dtype=float))
    prolif = signature_score(residual, signatures["proliferation"], space)
    try:
        from .aneuploidy_axis import aneuploidy_score
        aneu = aneuploidy_score(residual.to_dict())
    except Exception:  # noqa: BLE001 — aneuploidy axis (pyensembl/genome) optional; degrade gracefully
        aneu = {"score": float("nan"), "top_gained": [], "top_lost": []}
    proliferative = bool(not np.isnan(prolif) and prolif >= _PROLIFERATIVE_PCTL)
    aneuploid = bool(not np.isnan(aneu["score"]) and aneu["score"] >= _ANEUPLOIDY_STRONG)
    support = ([s for s, ok in (("proliferation", proliferative), ("aneuploidy", aneuploid)) if ok] or ["weak"])
    char = {"proliferation_percentile": round(prolif, 1) if not np.isnan(prolif) else None,
            "proliferative": proliferative,
            "aneuploidy_score": aneu["score"], "aneuploid": aneuploid,
            "aneuploidy_arms_gained": aneu["top_gained"], "aneuploidy_arms_lost": aneu["top_lost"],
            "malignancy_support": "+".join(support)}
    if mode == "heme" and heme_info:
        char["heme_sublineage"] = {k: heme_info[k] for k in ("malignant_sublineage", "source", "sublineage_scores") if k in heme_info}
    return char


def _compartment_for(type_code, fam, classifier):
    """Fine lineage compartment (Epithelial/Melanoma/Heme/Embryonal/…) for tumor-signature choice."""
    if classifier and classifier.get("compartment"):
        return classifier["compartment"]
    if type_code:
        try:
            from pirlygenes.gene_sets_cancer import cancer_lineage_group
            for cand in _candidates(type_code):
                g = cancer_lineage_group(cand)
                if g:
                    return g
        except Exception:  # noqa: BLE001
            pass
    return _FAMILY_COMPARTMENT.get((fam or "").lower())


def bulk_aneuploidy_amplitude(sample) -> float | None:
    """Noise-floor-subtracted bulk aneuploidy amplitude — a purity-scaled (∝ purity) signal."""
    if not isinstance(sample, pd.Series):
        sample = _safe(pd.Series(dict(sample), dtype=float))
    try:
        from .aneuploidy_axis import aneuploidy_score
        s = aneuploidy_score(sample.to_dict()).get("score")
    except Exception:  # noqa: BLE001
        return None
    return None if (s is None or np.isnan(s)) else round(max(0.0, s - _ANEUPLOIDY_NOISE_FLOOR), 4)


def restricted_marker_burden(sample, signatures=None) -> dict:
    """Burden of germline/CTA-restricted genes (off in normal soma) — a purity floor + lineage flag."""
    if signatures is None:
        _, signatures = _refs()
    if not isinstance(sample, pd.Series):
        sample = _safe(pd.Series(dict(sample), dtype=float))
    g = [x for x in signatures.get("restricted", ()) if x in sample.index]
    if not g:
        return {"n_expressed": 0, "max_percentile": None}
    pct = sample.rank(pct=True).loc[g] * 100
    return {"n_expressed": int((pct > 95).sum()), "max_percentile": round(float(pct.max()), 1)}


def decompose_expression(sample_tpm_by_symbol: Mapping[str, float], cancer=None, *,
                         space: str = "percentile", route_via_classifier: bool = True,
                         run_all: bool = False, met_sites: Sequence[str] | None = None) -> dict:
    """Lineage-routed decomposition + tumor characterization for one sample.

    ``met_sites`` is **off by default**; pass e.g. ``["liver"]`` only when met evidence
    exists. Met organs matching the primary's own lineage are skipped (no self-subtraction).
    """
    templates, signatures = _refs()
    sample = _safe(pd.Series(dict(sample_tpm_by_symbol), dtype=float))

    mode, routing, type_code = resolve_mode(cancer)
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
            else:
                run = _group_to_mode(call.get("runner_up", "")) if call.get("runner_up") else None
                candidates = list(dict.fromkeys([m for m in (top, run) if m]))
        except Exception as exc:  # noqa: BLE001
            classifier = {"error": str(exc)[:160]}

    modes = [mode] if (mode and not run_all) else (candidates if (candidates and not run_all)
                                                   else ["solid", "mesenchymal", "heme", "embryonal"])
    out, residuals, heme_infos = {}, {}, {}
    for m in modes:
        metrics, residual, info = decompose_mode(m, sample, templates, signatures, space, type_code, met_sites)
        out[m], residuals[m], heme_infos[m] = metrics, residual, info

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
        return {"selected_mode": None, "routing": routing, "confidence": "no decomposition produced"}

    sel = out[selected]
    residual = residuals[selected]
    compartment = _compartment_for(type_code, str(cancer or ""), classifier)
    characteristics = characterize_residual(residual, mode=selected, space=space,
                                            signatures=signatures, heme_info=heme_infos.get(selected))
    # compartment-aware tumor-identity confirmation on the residual (melanoma/NE/embryonal/CNS markers, not the generic mode sig)
    sig_key = COMPARTMENT_SIG.get(compartment) or MODE_TUMOR_SIG.get(selected)
    characteristics["tumor_lineage_signature"] = sig_key
    characteristics["tumor_lineage_signature_score"] = (
        round(signature_score(residual, signatures.get(sig_key, frozenset()), space), 1) if sig_key else None)

    note = next((v for k, v in _MOL_SUBTYPE_NOTE.items() if str(cancer or "").upper().endswith(k)), None)
    dp = sel.get("purity")
    bulk_aneu = bulk_aneuploidy_amplitude(sample)            # purity-scaled structural corroborator (bulk)
    restricted = restricted_marker_burden(sample, signatures)
    flags = []
    if bulk_aneu is not None and bulk_aneu < 0.02 and not characteristics["proliferative"]:
        flags.append("no aneuploidy & not proliferative — near-diploid indolent OR low-purity (ambiguous)")
    if restricted["n_expressed"] >= 1 and dp is not None and dp < 0.10:
        flags.append("restricted/CTA markers expressed but decomposition purity ~0 — possible over-subtraction")
    purity = {
        "primary": dp, "decomposition_purity": dp, "subtracted_fractions": sel.get("subtracted", {}),
        "corroborators": {"aneuploidy_amplitude_bulk": bulk_aneu, "restricted_marker_burden": restricted},
        "consistency_flags": flags,
        "note": "primary = lineage-routed decomposition_purity, computed under the SELECTED mode — never an "
                "epithelial estimate applied to another lineage. Corroborators are purity-aware but gated: bulk "
                "aneuploidy amplitude is purity-scaled (∝ purity, noise-floor-subtracted; noisier, used as a check / "
                "where deconvolution is ambiguous — not equal-weight); restricted-gene burden sets a purity floor. "
                "Proliferation & aneuploidy CHARACTERIZATION are on the purity-corrected residual.",
    }
    lineage = {"compartment": compartment, "mode": selected, "type_code": type_code,
               "routing": routing, "confidence": confidence}
    return {"selected_mode": selected, "lineage": lineage, "routing": routing, "confidence": confidence,
            "classifier": classifier, "space": space, "subtype_note": note, "purity": purity,
            "tumor_characteristics": characteristics, "modes": out}
