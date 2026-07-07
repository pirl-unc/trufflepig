"""Optional LEARNED cancer-type co-signal — a multinomial logistic regression over the per-gene
z-score feature space, trained on the representative-cohort samples.

This is a deliberately guarded second opinion. The default analysis pipeline can fuse it through the
evidence graph, but the hierarchical compartment→leaf system (curated panels, vetoes, family logic,
interpretability, rare/zero-sample types, structured-contamination handling) remains the primary path.
This module exists so callers can add cheap, purity-robust learned votes as explicit evidence rows.

Measured (scripts/zscore_classifier_ab.py, 266 samples / 54 primary types, leakage-free 5-fold CV):
  - LR (z-score) 0.865 clean, > nearest-centroid 0.835;
  - perfectly purity-robust to a GENERIC (pan-cancer-mean) contaminant — 0.865 flat to 30% purity,
    where nearest-centroid collapses to ~0.16.
Honest caveats it does NOT fix: (1) STRUCTURED contamination (liver->LIHC, immune-rich->heme) still
misleads any linear model; (2) rare types with ≤5 (or zero) training samples can't get a reliable
boundary — class_weight='balanced' helps but does not invent data; (3) it is not interpretable. Use
it as a co-signal, never as the sole call.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import numpy as np
import pandas as pd

# Informative-gene subset size (top cross-sample variance) — matches the validation harness.
_N_GENES = 2000
_MIN_SHARED_GENES = 200
_EXPECTED_OPTIONAL_MODEL_ERRORS = (
    ImportError,
    OSError,
    KeyError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True)
class LearnedExpressionVote:
    """One stage-scoped learned expression vote."""

    stage: str
    label_space: str
    predictions: tuple[tuple[str, float], ...]
    training_split_policy: str = "all_representative_samples"
    holdout_top1_accuracy: float | None = None
    holdout_medoid_top1_accuracy: float | None = None
    oof_precision_at_threshold: float | None = None
    oof_top3_recovery: float | None = None

    @property
    def label(self) -> str:
        return self.predictions[0][0] if self.predictions else ""

    @property
    def probability(self) -> float:
        return float(self.predictions[0][1]) if self.predictions else 0.0

    @property
    def margin(self) -> float:
        if not self.predictions:
            return 0.0
        second = float(self.predictions[1][1]) if len(self.predictions) > 1 else 0.0
        return float(self.predictions[0][1]) - second

    def public_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "label_space": self.label_space,
            "label": self.label,
            "probability": round(self.probability, 6),
            "margin": round(self.margin, 6),
            "top_predictions": [
                {"label": label, "probability": round(float(probability), 6)}
                for label, probability in self.predictions
            ],
            "training_split_policy": self.training_split_policy,
            "holdout_top1_accuracy": self.holdout_top1_accuracy,
            "holdout_medoid_top1_accuracy": self.holdout_medoid_top1_accuracy,
            "oof_precision_at_threshold": self.oof_precision_at_threshold,
            "oof_top3_recovery": self.oof_top3_recovery,
        }


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _learned_compartment_for_code(code: str) -> str:
    try:
        from pirlygenes.gene_sets_cancer import cancer_lineage_group
    except ImportError:
        return ""
    try:
        group = _clean(cancer_lineage_group(code) if code else "")
    except (KeyError, TypeError, ValueError):
        return ""
    return {
        "Epithelial": "epithelial",
        "Sarcoma": "mesenchymal",
        "Hematolymphoid": "hematolymphoid",
        "Melanoma": "melanocytic",
        "CNS": "cns",
        "Embryonal": "embryonal",
        "Neuroendocrine": "neuroendocrine",
        "Germ cell": "germ_cell",
    }.get(group, group.lower().replace(" ", "_"))


def _learned_family_for_code(code: str) -> str:
    code = _clean(code)
    if not code:
        return ""
    rollup_roots = {
        "CRC",
        "BRCA",
        "LUAD",
        "HNSC",
        "SCLC",
        "NBL",
        "MBL",
        "LAML",
        "SARC_RMS",
    }
    try:
        from pirlygenes.gene_sets_cancer import cancer_type_registry

        registry = cancer_type_registry().set_index("code").to_dict("index")
    except _EXPECTED_OPTIONAL_MODEL_ERRORS:
        registry = {}
    row = registry.get(code, {})
    family = _clean(row.get("family"))
    parent = _clean(row.get("parent_code"))
    current = code
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        if current in rollup_roots:
            return current
        current = _clean(registry.get(current, {}).get("parent_code"))
    if code.startswith("SARC_LPS") or code in {
        "SARC_DDLPS",
        "SARC_WDLPS",
        "SARC_MYXLPS",
        "SARC_PLEOLPS",
    }:
        return "SARC_LPS"
    if code.startswith("SARC_RMS"):
        return "SARC_RMS"
    if code in {"SARC_GIST"}:
        return "SARC_GIST"
    if code in {"SARC_ASPS", "SARC_CCS", "SARC_PEC"}:
        return "SARC_MELANOCYTIC_TRANSLOCATION"
    if code in {"SARC_ANGIO", "SARC_EHE", "SARC_KS"}:
        return "SARC_VASCULAR"
    if code in {"SARC_MPNST"}:
        return "SARC_NERVE_SHEATH"
    if code.startswith("SARC_"):
        return "SARC_OTHER"
    if code.startswith("NET_") or code.startswith("NEC_") or code in {"MTC", "PCPG"}:
        return "NEUROENDOCRINE_OTHER"
    if code.startswith("NBL"):
        return "NBL"
    if code.startswith("MBL"):
        return "MBL"
    if code.startswith("LAML"):
        return "LAML"
    return family or parent or code


def _learned_entity_for_code(code: str) -> str:
    code = _clean(code)
    try:
        from pirlygenes.gene_sets_cancer import cancer_type_registry

        row = cancer_type_registry().set_index("code").to_dict("index").get(code, {})
    except _EXPECTED_OPTIONAL_MODEL_ERRORS:
        row = {}
    parent = _clean(row.get("parent_code"))
    subtype_key = _clean(row.get("subtype_key"))
    if subtype_key and parent:
        return parent
    return code


def _learned_subtype_axis_for_code(code: str) -> str:
    code = _clean(code)
    try:
        from pirlygenes.gene_sets_cancer import cancer_type_registry

        row = cancer_type_registry().set_index("code").to_dict("index").get(code, {})
    except _EXPECTED_OPTIONAL_MODEL_ERRORS:
        row = {}
    parent = _clean(row.get("parent_code"))
    subtype_key = _clean(row.get("subtype_key"))
    return parent if subtype_key and parent else ""


def _collapse_predictions(
    predictions: list[tuple[str, float]] | tuple[tuple[str, float], ...],
    mapper,
    *,
    top_k: int,
) -> tuple[tuple[str, float], ...]:
    scores: dict[str, float] = {}
    for code, probability in predictions:
        label = _clean(mapper(code))
        if not label:
            continue
        scores[label] = scores.get(label, 0.0) + float(probability)
    return tuple(
        sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    )


@lru_cache(maxsize=1)
def _training_matrix():
    try:
        from oncoref.normalization import clean_tpm
        from pirlygenes.expression.accessors import (
            available_representative_cohorts,
            representative_cohort_samples,
        )
    except ImportError:
        return None

    try:
        frames, labels, base = [], [], None
        for code in sorted(available_representative_cohorts()):
            d = (
                representative_cohort_samples(code)
                .drop_duplicates("Ensembl_Gene_ID")
                .set_index("Ensembl_Gene_ID")
            )
            cols = [c for c in d.columns if c != "Symbol"]
            if base is None:
                base = d[["Symbol"]].copy()
            frames.append(d[cols])
            labels += [code] * len(cols)
        expr = pd.concat(frames, axis=1).dropna(how="any")
        gt = pd.DataFrame(
            {"Ensembl_Gene_ID": expr.index, "Symbol": base["Symbol"].reindex(expr.index).values}
        )
        clean = clean_tpm(expr.astype(float), gene_table=gt)
        clean.index = base["Symbol"].reindex(expr.index).values
        clean = clean[~clean.index.duplicated(keep="first")]
        logc = np.log1p(clean.clip(lower=0))
        var = logc.var(axis=1)
        genes = list(var.sort_values(ascending=False).index[:_N_GENES])
        X = np.nan_to_num(logc.loc[genes].T.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.asarray(labels)
        return X, y, genes
    except _EXPECTED_OPTIONAL_MODEL_ERRORS:
        return None


def _fit_lr_model(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="newton-cg",
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
        ),
    )
    pipe.fit(X, y)
    return pipe


@lru_cache(maxsize=1)
def _trained_model():
    """``(pipeline, gene_symbols, classes)`` — lazily trained once, cached.

    Features are log1p(clean-TPM) over a fixed high-variance SYMBOL index; the pipeline z-scores
    (StandardScaler) then fits a balanced multinomial logistic regression. Returns ``None`` if the
    reference samples or sklearn are unavailable (callers then skip the co-signal)."""
    try:
        training = _training_matrix()
        if training is None:
            return None
        X, y, genes = training
        pipe = _fit_lr_model(X, y)
        return pipe, genes, list(pipe.classes_)
    except _EXPECTED_OPTIONAL_MODEL_ERRORS:
        return None


@lru_cache(maxsize=1)
def _trained_hierarchy_models():
    try:
        training = _training_matrix()
        if training is None:
            return None
        X, y, genes = training
        labelers = {
            "compartment": _learned_compartment_for_code,
            "family": _learned_family_for_code,
            "entity": _learned_entity_for_code,
        }
        models = {}
        for stage, labeler in labelers.items():
            stage_y = np.asarray([labeler(code) for code in y])
            keep = np.asarray([bool(label) for label in stage_y])
            if keep.sum() < 2 or len(set(stage_y[keep])) < 2:
                continue
            pipe = _fit_lr_model(X[keep], stage_y[keep])
            models[stage] = (pipe, list(pipe.classes_))
        return genes, models
    except _EXPECTED_OPTIONAL_MODEL_ERRORS:
        return None


def _sample_vector(
    sample_tpm_by_symbol: Mapping[str, float],
    genes: list[str],
) -> np.ndarray | None:
    if not sample_tpm_by_symbol:
        return None
    present = sum(1 for gene in genes if gene in sample_tpm_by_symbol)
    if present < _MIN_SHARED_GENES:
        return None
    return np.log1p(
        np.asarray(
            [
                max(0.0, float(sample_tpm_by_symbol.get(gene, 0.0) or 0.0))
                for gene in genes
            ]
        )
    )


def _predict_with_model(
    pipe,
    classes: list[str],
    vec: np.ndarray,
    top_k: int,
) -> tuple[tuple[str, float], ...]:
    proba = pipe.predict_proba(vec.reshape(1, -1))[0]
    order = np.argsort(proba)[::-1][:top_k]
    return tuple((str(classes[i]), float(proba[i])) for i in order)


def classify_expression(sample_tpm_by_symbol, top_k=5):
    """Optional learned co-signal: ``[(code, probability), …]`` top-``top_k``, sorted descending.

    ``sample_tpm_by_symbol`` is clean-TPM keyed by gene symbol (e.g. ``_build_sample_tpm_by_symbol``).
    Returns ``[]`` when the model can't be trained or the sample shares too few genes with the
    feature index. This is a SECOND OPINION — fuse it, don't trust it alone (see module docstring).
    """
    if not sample_tpm_by_symbol:
        return []
    if len(sample_tpm_by_symbol) < _MIN_SHARED_GENES:
        return []
    trained = _trained_model()
    if trained is None:
        return []
    pipe, genes, classes = trained
    vec = _sample_vector(sample_tpm_by_symbol, genes)
    if vec is None:
        return []
    return list(_predict_with_model(pipe, classes, vec, top_k))


def classify_expression_hierarchy(
    sample_tpm_by_symbol: Mapping[str, float],
    *,
    top_k: int = 5,
) -> list[LearnedExpressionVote]:
    """Return stage-scoped learned expression votes.

    Compartment, family, and entity votes are trained as separate logistic
    models over the same feature space. The subtype-axis vote is derived from
    the flat classifier because those axes are sparse and parent-scoped. These
    votes are evidence rows for candidate admission/corroboration; they are not
    a separate cancer-call oracle.
    """

    trained = _trained_hierarchy_models()
    flat = tuple(classify_expression(sample_tpm_by_symbol, top_k=1000))
    if trained is None or not flat:
        return []
    genes, models = trained
    vec = _sample_vector(sample_tpm_by_symbol, genes)
    if vec is None:
        return []
    # Seed-0 3-train/2-test harness values from
    # scripts/eval_learned_classifier_holdout.py. They document calibration
    # context for evidence review; selectors do not consume them.
    stage_holdout_top1 = {
        "compartment": 219 / 223,
        "family": 193 / 223,
        "entity": 152 / 223,
    }
    stage_medoid_top1 = {
        "compartment": 116 / 117,
        "family": 104 / 117,
        "entity": 83 / 117,
    }
    stage_top3_recovery = {
        "entity": 218 / 223,
    }
    votes: list[LearnedExpressionVote] = []
    for stage, label_space in (
        ("compartment", "learned_compartment"),
        ("family", "learned_family"),
        ("entity", "learned_entity"),
    ):
        if stage not in models:
            continue
        pipe, classes = models[stage]
        predictions = _predict_with_model(pipe, classes, vec, top_k)
        if predictions:
            votes.append(
                LearnedExpressionVote(
                    stage=stage,
                    label_space=label_space,
                    predictions=predictions,
                    holdout_top1_accuracy=stage_holdout_top1.get(stage),
                    holdout_medoid_top1_accuracy=stage_medoid_top1.get(stage),
                    oof_top3_recovery=stage_top3_recovery.get(stage),
                )
            )
    subtype_predictions = _collapse_predictions(
        flat,
        lambda code: f"{_learned_subtype_axis_for_code(code)}:{code}"
        if _learned_subtype_axis_for_code(code)
        else "",
        top_k=top_k,
    )
    if subtype_predictions:
        votes.append(
            LearnedExpressionVote(
                stage="subtype_axis",
                label_space="learned_subtype_axis",
                predictions=subtype_predictions,
            )
        )
    return votes


__all__ = [
    "LearnedExpressionVote",
    "classify_expression",
    "classify_expression_hierarchy",
]
