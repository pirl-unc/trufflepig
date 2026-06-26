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
                                   per-population fractions → ``residual_fraction`` = residual mass
                                   fraction (lineage-aware; fixes ESTIMATE's heme/sarcoma blind
                                   spots — monotone with purity, not yet calibrated to absolute).

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

**Interaction with purity estimation.** ``residual_fraction`` is the lineage-aware primary
purity signal (works where ESTIMATE breaks on heme/sarcoma). It is *corroborated*, not naively
fused, by two orthogonal signals (equal-weight fusion was rejected — it was empirically worse):
  * **bulk aneuploidy amplitude** ∝ purity (an arm gain in a 50%-pure sample is half-strength) —
    a lineage-agnostic, CNA-based corroborator. Measured on the BULK (not the residual).
  * **restricted/CTA-gene burden** — off in normal soma, so its bulk expression sets a purity floor.
Distinct from the above, proliferation and *residual* aneuploidy are tumor CHARACTERIZATION,
measured on the purity-corrected **residual** — the decomposition is what makes them readable.
(Wiring + the absolute-purity calibration live in ``tumor_purity.estimate_tumor_purity``; see #96.)
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Mapping, Sequence, TypedDict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── output contract (documents the result shape; structural, not enforced at runtime) ──
class ModeMetrics(TypedDict, total=False):
    residual_fraction: float | None          # residual mass fraction under this mode
    subtracted: dict[str, float]             # background → subtracted fraction
    tumor_lineage_in_residual: float | None  # mode tumor-signature score on the residual (routing tiebreak)
    lineage_fit: float                       # mode-comparable goodness-of-fit (tumor survives − background leakage)
    template_collinearity: float             # identifiability: max |cosine| of subtracted templates
    met_sites_subtracted: list[str]          # met organs whose normal template was subtracted
    met_sites_skipped_as_primary: list[str]  # met organs skipped (would self-subtract the primary)
    met_sites_unknown: list[str]             # requested met organs with no template (typo/unsupported)
    note: str


class Corroborators(TypedDict):
    aneuploidy_amplitude_bulk: float | None  # purity-scaled (∝ purity), noise-floor-subtracted; bulk
    restricted_marker_burden: dict           # germline/CTA burden → purity floor


class PurityBlock(TypedDict):
    residual_fraction: float | None          # primary (lineage-routed; not yet absolute purity)
    subtracted_fractions: dict[str, float]
    corroborators: Corroborators
    consistency_flags: list[str]
    note: str


class TumorCharacteristics(TypedDict, total=False):
    proliferation_percentile: float | None
    proliferative: bool
    aneuploidy_score: float | None           # RESIDUAL aneuploidy (characterization, not purity)
    aneuploid: bool
    aneuploidy_arms_gained: list[str]
    aneuploidy_arms_lost: list[str]
    malignancy_support: str                  # "proliferation+aneuploidy" / … / "weak"
    tumor_lineage_signature: str
    tumor_lineage_signature_score: float | None
    heme_sublineage: dict


class LineageInfo(TypedDict):
    compartment: str | None
    mode: str
    type_code: str | None
    routing: str
    confidence: str


class DecompositionResult(TypedDict, total=False):
    selected_mode: str | None
    lineage: LineageInfo
    routing: str
    confidence: str
    classifier: dict | None
    space: str
    subtype_note: str | None
    purity: PurityBlock
    tumor_characteristics: TumorCharacteristics
    modes: dict[str, ModeMetrics]

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
# Each decomposition MODE: the non-tumor backgrounds it subtracts (NNLS templates), and the
# signature its residual is validated against by default (a finer compartment signature overrides
# it — see _COMPARTMENTS). `heme` additionally subtracts the *healthy* immune sub-lineages,
# determined per sample, keeping the malignant one as tumor.
_MODES = {
    "solid":       {"subtract": ("immune", "stromal"),     "tumor_sig": "epithelial"},
    "mesenchymal": {"subtract": ("immune", "epithelial"),  "tumor_sig": "stromal"},
    "heme":        {"subtract": ("epithelial", "stromal"), "tumor_sig": "immune"},
    "embryonal":   {"subtract": ("immune",),               "tumor_sig": "proliferation"},
}
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
# Each fine lineage COMPARTMENT (as returned by compartment_call / cancer_lineage_group): the
# decomposition mode it routes to, and the positive marker signature its tumor residual is
# confirmed against. This is the single source for compartment → (mode, signature).
_COMPARTMENTS = {
    "Epithelial":     {"mode": "solid",       "tumor_sig": "epithelial"},
    "Melanoma":       {"mode": "solid",       "tumor_sig": "melanoma"},
    "Neuroendocrine": {"mode": "solid",       "tumor_sig": "neuroendocrine"},
    "CNS":            {"mode": "solid",       "tumor_sig": "cns"},
    "Germ cell":      {"mode": "solid",       "tumor_sig": "germ"},
    "Embryonal":      {"mode": "embryonal",   "tumor_sig": "embryonal"},
    "Heme":           {"mode": "heme",        "tumor_sig": "immune"},
    "Sarcoma":        {"mode": "mesenchymal", "tumor_sig": "stromal"},
}
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
    """Compartment label → decomposition mode: exact via _COMPARTMENTS, fuzzy fallback for variants."""
    if group in _COMPARTMENTS:
        return _COMPARTMENTS[group]["mode"]
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


def _code_candidates(type_code):
    """Yield a type_code and its parent codes, upper-cased — e.g. ``'LAML_APL'`` → ``'LAML_APL','LAML'``.

    Lets subtype hints (``LAML_APL``, ``LUAD_EGFR``) match the parent-keyed registries
    (:data:`HEME_MALIGNANT`, :data:`_MET_PRIMARY_LINEAGE`) instead of silently falling through.
    """
    if type_code:
        yield from _candidates(str(type_code).upper())


def resolve_mode(cancer):
    """Resolve a ``cancer`` hint to ``(mode_or_None, routing, type_code_or_None)``."""
    if not cancer:
        return None, "no hint", None
    text = str(cancer).strip()
    fam = text.lower()
    if fam in _FAMILY_COMPARTMENT:                            # family keyword → mode via its compartment (single source)
        return _group_to_mode(_FAMILY_COMPARTMENT[fam]), f"family={text}", None
    try:
        from pirlygenes.gene_sets_cancer import cancer_lineage_group
    except ImportError:
        return None, f"{text!r}: no resolver", None
    for cand in _candidates(text):
        try:
            group = cancer_lineage_group(cand)
        except (KeyError, ValueError):                       # unknown candidate code — expected, try the next
            group = None
        if group:
            return _group_to_mode(group), f"{text} → lineage_group={group} (via {cand})", cand
    return None, f"{text!r} unresolved", None


def _safe(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _as_series(sample) -> pd.Series:
    """Coerce a ``{symbol: value}`` mapping (or Series) to a finite float Series."""
    return sample if isinstance(sample, pd.Series) else _safe(pd.Series(dict(sample), dtype=float))


def _round_or_none(v, n):
    """Round a score for output; NaN/None → None (valid JSON null, never NaN)."""
    return round(v, n) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else None


def _rank_score(v):
    """Ranking key for residual-lineage scores: missing/NaN sorts last (never wins a max())."""
    return v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else float("-inf")


def signature_score(sample: pd.Series, gene_symbols, space: str = "percentile") -> float | None:
    """Enrichment of a gene set in a sample, in ``percentile`` (default) or ``ssgsea`` space.

    Returns **None** when not computable (no signature gene present in the sample) — an explicit
    "not available" at the scoring boundary, so callers never have to test for NaN.
    """
    present = [g for g in gene_symbols if g in sample.index]
    if not present:
        return None
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


@lru_cache(maxsize=None)
def _mode_tumor_lineages(mode):
    """Tumor-lineage signatures a mode treats as TUMOR (kept, not subtracted) — derived from
    _COMPARTMENTS (e.g. solid → epithelial/melanoma/neuroendocrine/cns/germ; heme → immune)."""
    sigs = tuple(sorted({c["tumor_sig"] for c in _COMPARTMENTS.values() if c["mode"] == mode}))
    return sigs or (_MODES[mode]["tumor_sig"],)


_FIT_PROLIF_WEIGHT = 0.3  # small bonus: a real tumor's proliferation survives the right decomposition


def lineage_fit_score(mode, residual, *, signatures, space: str = "percentile") -> float:
    """Mode-comparable goodness-of-fit: how coherently a real tumor of ``mode``'s lineage survives
    in its residual.

    ``= max(tumor-lineage signal in residual) − max(subtracted-background leakage in residual)``
    ``  + small proliferation bonus``.

    Every term is a within-sample percentile, so fits ARE comparable across modes — unlike the raw
    residual fraction. The RIGHT mode keeps its tumor lineage (high) and cleanly removed its
    backgrounds (low → little leakage); a WRONG mode either subtracted the tumor itself (the tumor
    signal collapses) or left its backgrounds behind (leakage high). This is the goodness-of-fit a
    reconstruction-error criterion can't give you — ESTIMATE fails with *low* reconstruction error
    precisely when it absorbs a collinear tumor into a background.
    """
    residual = _as_series(residual)
    tumor = max((signature_score(residual, signatures.get(s, frozenset()), space) or 0.0) for s in _mode_tumor_lineages(mode))
    leakage = max((signature_score(residual, signatures.get(s, frozenset()), space) or 0.0) for s in _MODES[mode]["subtract"])
    prolif = signature_score(residual, signatures.get("proliferation", frozenset()), space) or 0.0
    return round((tumor - leakage) + _FIT_PROLIF_WEIGHT * (prolif - 50.0), 1)


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
    # subtype hints (LAML_APL) must resolve to the parent's malignant sub-lineage before marker fallback
    type_mapped = next((HEME_MALIGNANT[c] for c in _code_candidates(type_code) if c in HEME_MALIGNANT), None)
    if type_mapped is not None:
        malignant, source = type_mapped, "type-map"
    elif any(v is not None for v in scores.values()):
        malignant = max(scores, key=lambda s: _rank_score(scores[s])); source = "dominant-marker"
    else:                                                   # no sub-lineage markers present → can't tell
        malignant, source = None, "indeterminate"
    # when indeterminate, subtract NO sub-lineage (keep all immune as tumor) rather than guessing one
    healthy = [sl for sl in _SUBLINEAGES if sl != malignant] if malignant else []
    return {"malignant_sublineage": malignant, "source": source,
            "sublineage_scores": {k: _round_or_none(v, 1) for k, v in scores.items()}}, healthy


def _subtract_keys(mode, sample, signatures, space, type_code, met_sites):
    info = {}
    keys = list(_MODES[mode]["subtract"])
    if mode == "heme":
        info, healthy = _heme_immune(sample, signatures, space, type_code)
        keys += healthy                                   # keep the malignant sub-lineage as tumor
    applied_mets, skipped_mets, unknown_mets = [], [], []
    for site in (met_sites or []):
        if site not in _MET_SITE_MARKERS:                # only true met ORGANS — not 'immune'/'stromal'/etc.
            unknown_mets.append(site)                     # typo / unsupported / non-organ template key — surfaced, not silent
            continue
        primary = _MET_PRIMARY_LINEAGE.get(site, set())  # match subtype hints too (LUAD_EGFR → LUAD)
        if any(c in primary for c in _code_candidates(type_code)):
            skipped_mets.append(site)                     # don't subtract a primary's own organ
            continue
        keys.append(site); applied_mets.append(site)
    if met_sites:
        info = {**info, "met_sites_subtracted": applied_mets, "met_sites_skipped_as_primary": skipped_mets,
                "met_sites_unknown": unknown_mets}
    return keys, info


def decompose_mode(mode, sample, templates, signatures, space, type_code, met_sites):
    """Run one lineage mode's NNLS background subtraction → (metrics, residual, info)."""
    from scipy.optimize import nnls

    keys, info = _subtract_keys(mode, sample, signatures, space, type_code, met_sites)
    # surface the met-site bookkeeping in the output (which organs were subtracted / skipped-as-primary
    # / unknown) — it lives in `info` but the heme part of `info` is consumed separately downstream.
    met = {k: info[k] for k in ("met_sites_subtracted", "met_sites_skipped_as_primary", "met_sites_unknown") if k in info}
    fit = sorted(set().union(*[signatures.get(k, frozenset()) for k in keys]) & set(sample.index))
    if not fit:
        return {"residual_fraction": None, "note": "no fit genes", **met}, sample, info
    a = np.column_stack([templates[k].reindex(fit).fillna(0.0).values for k in keys])
    f, _ = nnls(a, sample.reindex(fit).fillna(0.0).values)
    recon = sum(f[i] * templates[keys[i]] for i in range(len(keys)))
    residual = (sample - recon.reindex(sample.index).fillna(0.0)).clip(lower=0)
    total = float(sample.sum()) or 1.0
    metrics = {"residual_fraction": round(float(residual.sum() / total), 3),
               "subtracted": {keys[i]: round(float(f[i] * templates[keys[i]].sum() / 1e6), 3) for i in range(len(keys))},
               "tumor_lineage_in_residual": _round_or_none(signature_score(residual, signatures[_MODES[mode]["tumor_sig"]], space), 2),
               "lineage_fit": lineage_fit_score(mode, residual, signatures=signatures, space=space),
               "template_collinearity": round(_collinearity([templates[k] for k in keys], fit), 3), **met}
    return metrics, residual, info


def characterize_residual(residual, *, mode: str = "solid", space: str = "percentile",
                          signatures=None, heme_info=None) -> TumorCharacteristics:
    """Tumor sub-population characteristics for a residual (``{symbol: TPM}`` or Series).

    Proliferation (cell-cycle) + aneuploidy (chromosome-arm coherence) + heme sub-lineage.
    Aneuploidy needs :mod:`trufflepig.aneuploidy_axis` (pyensembl); if it is unavailable it
    degrades gracefully (``aneuploidy_score=None``) rather than raising.
    """
    if signatures is None:
        _, signatures = _refs()
    residual = _as_series(residual)
    prolif = signature_score(residual, signatures["proliferation"], space)
    try:
        from .aneuploidy_axis import aneuploidy_score
        aneu = aneuploidy_score(residual.to_dict())
    except Exception:  # noqa: BLE001 — aneuploidy axis (pyensembl/genome) is optional; degrade, don't crash
        logger.debug("aneuploidy_score failed in characterize_residual; degrading to None", exc_info=True)
        aneu = {"score": None, "top_gained": [], "top_lost": []}
    aneu_score = aneu["score"]                                  # None when aneuploidy unavailable (valid JSON null)
    proliferative = bool(prolif is not None and prolif >= _PROLIFERATIVE_PCTL)
    aneuploid = bool(aneu_score is not None and aneu_score >= _ANEUPLOIDY_STRONG)
    support = ([s for s, ok in (("proliferation", proliferative), ("aneuploidy", aneuploid)) if ok] or ["weak"])
    char = {"proliferation_percentile": _round_or_none(prolif, 1),
            "proliferative": proliferative,
            "aneuploidy_score": aneu_score, "aneuploid": aneuploid,
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
        except ImportError:
            cancer_lineage_group = None
        for cand in (_candidates(type_code) if cancer_lineage_group else ()):
            try:
                g = cancer_lineage_group(cand)
            except (KeyError, ValueError):                   # unknown candidate code — try the next
                continue
            if g:
                return g
    return _FAMILY_COMPARTMENT.get((fam or "").lower())


def bulk_aneuploidy_amplitude(sample) -> float | None:
    """Noise-floor-subtracted bulk aneuploidy amplitude — a purity-scaled (∝ purity) signal."""
    sample = _as_series(sample)
    try:
        from .aneuploidy_axis import aneuploidy_score
        s = aneuploidy_score(sample.to_dict()).get("score")
    except Exception:  # noqa: BLE001 — aneuploidy axis (pyensembl/genome) is optional
        logger.debug("bulk aneuploidy amplitude unavailable", exc_info=True)
        return None
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    return round(max(0.0, s - _ANEUPLOIDY_NOISE_FLOOR), 4)


def restricted_marker_burden(sample, signatures=None) -> dict:
    """Burden of germline/CTA-restricted genes (off in normal soma) — a purity floor + lineage flag."""
    if signatures is None:
        _, signatures = _refs()
    sample = _as_series(sample)
    g = [x for x in signatures.get("restricted", ()) if x in sample.index]
    if not g:
        return {"n_expressed": 0, "max_percentile": None}
    pct = sample.rank(pct=True).loc[g] * 100
    return {"n_expressed": int((pct > 95).sum()), "max_percentile": round(float(pct.max()), 1)}


_PURITY_NOTE = (
    "residual_fraction = lineage-routed residual MASS FRACTION (computed under the SELECTED mode — "
    "never an epithelial estimate applied to another lineage). It is MONOTONE with purity but NOT yet "
    "calibrated to absolute purity (≈0.6 at a pure medoid); calibration is a follow-up. Corroborators "
    "are purity-aware but gated: BULK aneuploidy amplitude is purity-scaled (noise-floor-subtracted; "
    "noisier — a check / for where deconvolution is ambiguous, not equal-weight); restricted-gene "
    "burden sets a purity floor. Proliferation & RESIDUAL aneuploidy are tumor CHARACTERIZATION on the "
    "purity-corrected residual — distinct from the bulk-aneuploidy purity signal."
)
_NO_ANEUPLOIDY_FLOOR = 0.02   # below this bulk amplitude is indistinguishable from diploid noise
_OVER_SUBTRACTION_RF = 0.10   # residual_fraction this low with restricted markers present ⇒ likely over-subtracted


def _build_purity_block(sel, sample, signatures, proliferative) -> PurityBlock:
    """Purity block: ``residual_fraction`` primary + gated purity-aware corroborators + consistency flags."""
    rf = sel.get("residual_fraction")
    bulk_aneu = bulk_aneuploidy_amplitude(sample)            # purity-scaled corroborator (BULK, not residual)
    restricted = restricted_marker_burden(sample, signatures)
    flags = []
    if bulk_aneu is not None and bulk_aneu < _NO_ANEUPLOIDY_FLOOR and not proliferative:
        flags.append("no aneuploidy & not proliferative — near-diploid indolent OR low-purity (ambiguous)")
    if restricted["n_expressed"] >= 1 and rf is not None and rf < _OVER_SUBTRACTION_RF:
        flags.append("restricted/CTA markers expressed but residual_fraction ~0 — possible over-subtraction")
    return {"residual_fraction": rf, "subtracted_fractions": sel.get("subtracted", {}),
            "corroborators": {"aneuploidy_amplitude_bulk": bulk_aneu, "restricted_marker_burden": restricted},
            "consistency_flags": flags, "note": _PURITY_NOTE}


def decompose_expression(sample_tpm_by_symbol: Mapping[str, float], cancer=None, *,
                         space: str = "percentile", route_via_classifier: bool = True,
                         run_all: bool = False, met_sites: Sequence[str] | None = None) -> DecompositionResult:
    """Lineage-routed decomposition + tumor characterization for one sample.

    ``met_sites`` is **off by default**; pass e.g. ``["liver"]`` only when met evidence
    exists. Met organs matching the primary's own lineage are skipped (no self-subtraction).
    """
    templates, signatures = _refs()
    sample = _as_series(sample_tpm_by_symbol)

    mode, routing, type_code = resolve_mode(cancer)
    classifier = None
    candidates = None
    candidate_compartment = {}                                # mode -> compartment for the ambiguous candidates
    if mode is None and route_via_classifier:
        try:
            from .cancer_type_centroid import compartment_call

            call = compartment_call(dict(sample_tpm_by_symbol))
            classifier = {"compartment": call.get("compartment"), "confident": bool(call.get("confident")),
                          "margin": round(float(call.get("margin", 0.0)), 4), "runner_up": call.get("runner_up")}
            compartment = call.get("compartment")
            # A missing compartment means compartment_call ABSTAINED (e.g. sparse input) — it must NOT
            # map to solid, or candidates collapse to ['solid'] and the all-mode fallback never runs.
            top = _group_to_mode(compartment) if compartment else None
            if compartment and call.get("confident"):
                mode = top
                routing = f"compartment_call={compartment} (confident, margin {classifier['margin']}) → {top}"
            else:
                run = _group_to_mode(call.get("runner_up")) if call.get("runner_up") else None
                candidates = list(dict.fromkeys([m for m in (top, run) if m])) or None  # None → all-mode fallback
                if top:
                    candidate_compartment[top] = compartment
                if run:
                    candidate_compartment.setdefault(run, call.get("runner_up"))
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
        selected = max(candidates, key=lambda m: _rank_score(out[m].get("lineage_fit")))
        confidence = (f"ambiguous compartment ({classifier['compartment']} vs {classifier['runner_up']}, "
                      f"margin {classifier['margin']}) → {selected} by lineage fit — verify")
    elif out:
        selected = max(out, key=lambda m: _rank_score(out[m].get("lineage_fit")))
        confidence = "low (data-only fallback; no hint and classifier unavailable)"
    else:
        return {"selected_mode": None, "routing": routing, "confidence": "no decomposition produced"}

    sel = out[selected]
    residual = residuals[selected]
    # compartment of the SELECTED mode (when ambiguous routing picked the runner-up, use ITS compartment)
    compartment = candidate_compartment.get(selected) or _compartment_for(type_code, str(cancer or ""), classifier)
    characteristics = characterize_residual(residual, mode=selected, space=space,
                                            signatures=signatures, heme_info=heme_infos.get(selected))
    # compartment-aware tumor-identity confirmation on the residual: prefer the fine compartment's
    # markers (melanoma/NE/embryonal/CNS/germ) and fall back to the selected mode's signature
    # (every mode has one, so sig_key always resolves to a real, present signature).
    sig_key = (_COMPARTMENTS.get(compartment) or {}).get("tumor_sig") or _MODES[selected]["tumor_sig"]
    sig_genes = signatures.get(sig_key)
    characteristics["tumor_lineage_signature"] = sig_key
    characteristics["tumor_lineage_signature_score"] = (
        _round_or_none(signature_score(residual, sig_genes, space), 1) if sig_genes else None)

    note = next((v for k, v in _MOL_SUBTYPE_NOTE.items() if str(cancer or "").upper().endswith(k)), None)
    purity = _build_purity_block(sel, sample, signatures, characteristics["proliferative"])
    lineage = {"compartment": compartment, "mode": selected, "type_code": type_code,
               "routing": routing, "confidence": confidence}
    return {"selected_mode": selected, "lineage": lineage, "routing": routing, "confidence": confidence,
            "classifier": classifier, "space": space, "subtype_note": note, "purity": purity,
            "tumor_characteristics": characteristics, "modes": out}
