"""Optional learned cancer-type co-signals trained on representative cohorts.

At a high level, the flat classifier supplies a quantifier-robust second
opinion and the staged hierarchy supplies calibrated compartment, family, and
entity context. The evidence graph treats those related predictions as one
learned full-profile evidence group and requires independent corroboration
before changing a report label.

The flat model uses within-sample expression percentiles, which limit the
leverage of isolated abundance differences between quantifiers. The hierarchy
uses calibrated log-clean-TPM because its stage probabilities and lineage
contexts depend on absolute reference-space abundance. Neither view is an
oracle: structured contamination can mislead both, rare classes remain
data-limited, and neither replaces interpretable marker or reference evidence.

The module also exposes orthogonal learned molecular-state votes when the
representative labels support them. The MSI-vs-MSS classifier is reported as
mismatch-repair RNA context, not as clinical MSI/MMR status.

Historical benchmark results for an earlier z-score flat model are retained in
``docs/cancer-type-hierarchical-classifier.md`` and should not be read as
current-bundle performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

# Informative-gene subset size (top cross-sample variance) — matches the validation harness.
_N_GENES = 2000
_MIN_SHARED_GENES = 200
_MMR_RELEASE_MODEL = (
    Path(__file__).resolve().parent
    / "data"
    / "mismatch-repair-expression-ensemble.csv"
)
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
    training_sample_count: int | None = None
    training_cohorts: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

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
            "training_sample_count": self.training_sample_count,
            "training_cohorts": list(self.training_cohorts),
            "details": self.details,
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


def _mismatch_repair_state_for_code(code: str) -> str:
    """Map registry subtype codes to the binary mismatch-repair state label."""

    parts = [part for part in _clean(code).upper().split("_") if part]
    if not parts:
        return ""
    suffix = parts[-1]
    if suffix in {"MSI", "MSIH", "DMMR", "MMRD"}:
        return "MSI"
    if suffix in {"MSS", "PMMR", "MMRP", "CNL", "CNH"}:
        return "MSS"
    return ""


def _mismatch_repair_base_code(code: str) -> str:
    parts = [part for part in _clean(code).upper().split("_") if part]
    if not parts:
        return ""
    if _mismatch_repair_state_for_code("_".join(parts)):
        parts = parts[:-1]
    return "_".join(parts)


def mismatch_repair_context_group(cancer_type: str) -> str:
    """Return the supported tissue/entity context for the release MMR model."""

    code = _mismatch_repair_base_code(cancer_type)
    if not code:
        return ""
    family = _learned_family_for_code(code)
    if family == "CRC" or code in {"CRC", "COAD", "READ"}:
        return "CRC"
    if code == "UCEC" or code.startswith("UCEC_"):
        return "UCEC"
    if code == "STAD" or code.startswith("STAD_"):
        return "STAD"
    return ""


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
def _training_matrices():
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
        raw_variance = logc.var(axis=1)
        raw_genes = list(
            raw_variance.sort_values(ascending=False).index[:_N_GENES]
        )
        raw_x = np.nan_to_num(
            logc.loc[raw_genes].T.to_numpy(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        # A second, quantifier-robust view for the flat classifier. Per-sample
        # percentile ranks cap the leverage of a few gene-level abundance
        # disagreements (for example, alternative quantifiers assigning very
        # different TPM to one paralog) while retaining the within-profile
        # lineage program. Hierarchical models keep the calibrated log-TPM view.
        ranked = clean.rank(axis=0, pct=True, method="average")
        rank_variance = ranked.var(axis=1)
        rank_genes = list(
            rank_variance.sort_values(ascending=False).index[:_N_GENES]
        )
        rank_x = np.nan_to_num(
            ranked.loc[rank_genes].T.to_numpy(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        rank_universe = list(ranked.index)
        y = np.asarray(labels)
        return (raw_x, y, raw_genes), (
            rank_x,
            y,
            rank_genes,
            rank_universe,
        )
    except _EXPECTED_OPTIONAL_MODEL_ERRORS:
        return None


@lru_cache(maxsize=1)
def _training_matrix():
    matrices = _training_matrices()
    return matrices[0] if matrices is not None else None


@lru_cache(maxsize=1)
def _rank_training_matrix():
    matrices = _training_matrices()
    return matrices[1] if matrices is not None else None


@lru_cache(maxsize=1)
def _mismatch_repair_training_matrix():
    """Binary MSI/MSS matrix from all packaged representative subtype samples."""

    try:
        from oncoref.normalization import clean_tpm
        from pirlygenes.expression.accessors import (
            available_representative_cohorts,
            representative_cohort_samples,
        )
    except ImportError:
        return None

    try:
        frames, labels, cohort_labels, base = [], [], [], None
        cohorts_by_state: dict[str, list[str]] = {"MSI": [], "MSS": []}
        for code in sorted(available_representative_cohorts()):
            state = _mismatch_repair_state_for_code(code)
            if state not in cohorts_by_state:
                continue
            d = (
                representative_cohort_samples(code)
                .drop_duplicates("Ensembl_Gene_ID")
                .set_index("Ensembl_Gene_ID")
            )
            cols = [c for c in d.columns if c != "Symbol"]
            if not cols:
                continue
            if base is None:
                base = d[["Symbol"]].copy()
            frames.append(d[cols])
            labels += [state] * len(cols)
            cohort_labels += [code] * len(cols)
            cohorts_by_state[state].append(code)
        if not frames or len(set(labels)) < 2 or base is None:
            return None
        expr = pd.concat(frames, axis=1).dropna(how="any")
        if expr.empty:
            return None
        gt = pd.DataFrame(
            {
                "Ensembl_Gene_ID": expr.index,
                "Symbol": base["Symbol"].reindex(expr.index).values,
            }
        )
        clean = clean_tpm(expr.astype(float), gene_table=gt)
        clean.index = base["Symbol"].reindex(expr.index).values
        clean = clean[~clean.index.duplicated(keep="first")]
        logc = np.log1p(clean.clip(lower=0))
        var = logc.var(axis=1)
        genes = list(var.sort_values(ascending=False).index[:_N_GENES])
        X = np.nan_to_num(
            logc.loc[genes].T.to_numpy(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        y = np.asarray(labels)
        cohorts = tuple(
            sorted({code for codes in cohorts_by_state.values() for code in codes})
        )
        return X, y, genes, cohorts, tuple(cohort_labels)
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


def _leave_one_out_top1_accuracy(X, y) -> float | None:
    try:
        from sklearn.model_selection import LeaveOneOut
    except ImportError:
        return None

    try:
        correct = 0
        evaluated = 0
        for train_idx, test_idx in LeaveOneOut().split(X):
            y_train = y[train_idx]
            if len(set(y_train)) < 2:
                continue
            pipe = _fit_lr_model(X[train_idx], y_train)
            pred = str(pipe.predict(X[test_idx])[0])
            correct += int(pred == str(y[test_idx][0]))
            evaluated += 1
        return (correct / evaluated) if evaluated else None
    except _EXPECTED_OPTIONAL_MODEL_ERRORS:
        return None


@lru_cache(maxsize=1)
def _trained_model():
    """``(pipeline, features, classes, rank_universe)`` — trained and cached.

    Features are within-sample percentile ranks over a fixed high-variance
    SYMBOL index. Percentiles are computed over the full stored training gene
    universe before selecting those features, exactly as they were during
    training. The pipeline z-scores them and fits a balanced multinomial
    logistic regression. This flat view is deliberately robust to isolated
    quantifier-specific TPM spikes and to quantifiers that omit zero rows.
    Returns ``None`` if references or sklearn are unavailable.
    """
    try:
        training = _rank_training_matrix()
        if training is None:
            return None
        X, y, genes, rank_universe = training
        pipe = _fit_lr_model(X, y)
        return pipe, genes, list(pipe.classes_), rank_universe
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


@lru_cache(maxsize=1)
def _trained_mismatch_repair_model():
    """``(pipeline, genes, classes, metrics)`` for the binary MSI/MSS vote."""

    try:
        training = _mismatch_repair_training_matrix()
        if training is None:
            return None
        X, y, genes, cohorts, cohort_labels = training
        if len(set(y)) < 2:
            return None
        pipe = _fit_lr_model(X, y)
        metrics = {
            "training_sample_count": int(len(y)),
            "training_cohorts": cohorts,
            "training_cohort_sample_count": {
                str(code): int(sum(1 for item in cohort_labels if item == code))
                for code in cohorts
            },
            "holdout_top1_accuracy": _leave_one_out_top1_accuracy(X, y),
        }
        return pipe, genes, list(pipe.classes_), metrics
    except _EXPECTED_OPTIONAL_MODEL_ERRORS:
        return None


@lru_cache(maxsize=1)
def _release_mismatch_repair_ensemble():
    """Load the packaged CRC/UCEC/STAD MMR expression ensemble."""

    try:
        df = pd.read_csv(_MMR_RELEASE_MODEL)
    except _EXPECTED_OPTIONAL_MODEL_ERRORS:
        return None
    if df.empty:
        return None
    try:
        members = []
        for member_index, group in df.groupby("member_index", sort=True):
            ordered = group.sort_values("feature_index").copy()
            members.append(
                {
                    "member_index": int(member_index),
                    "member": _clean(ordered["member"].iloc[0]),
                    "feature_space": _clean(ordered["feature_space"].iloc[0]),
                    "feature_selection": _clean(ordered["feature_selection"].iloc[0]),
                    "n_genes": int(ordered["n_genes"].iloc[0]),
                    "decision_threshold": float(
                        ordered["decision_threshold"].iloc[0]
                    ),
                    "feature_type": tuple(str(item) for item in ordered["feature_type"]),
                    "feature_id": tuple(str(item) for item in ordered["feature_id"]),
                    "symbol": tuple(_clean(item) for item in ordered["symbol"]),
                    "tissue_category": tuple(
                        _clean(item) for item in ordered["tissue_category"]
                    ),
                    "scaler_mean": ordered["scaler_mean"].astype(float).to_numpy(),
                    "scaler_scale": ordered["scaler_scale"].astype(float).to_numpy(),
                    "coef_msi": ordered["coef_msi"].astype(float).to_numpy(),
                    "intercept_msi": float(ordered["intercept_msi"].iloc[0]),
                }
            )
        if not members:
            return None
        threshold = float(members[0]["decision_threshold"])
        return {
            "members": tuple(members),
            "decision_threshold": threshold,
            "training_sample_count": 1386,
            "training_cohorts": ("CRC", "STAD", "UCEC"),
            "validation": {
                "pooled_5fold_accuracy": 0.9632,
                "leave_tissue_out_accuracy": 0.9343,
                "within_tissue_accuracy": 0.9524,
                "leave_tissue_out_crc_accuracy": 0.9570,
                "leave_tissue_out_stad_accuracy": 0.9566,
                "leave_tissue_out_ucec_accuracy": 0.8886,
            },
        }
    except (KeyError, TypeError, ValueError):
        return None


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = np.exp(-value)
        return float(1.0 / (1.0 + z))
    z = np.exp(value)
    return float(z / (1.0 + z))


def _release_member_msi_probability(
    sample_tpm_by_symbol: Mapping[str, float],
    context_group: str,
    member: Mapping[str, Any],
) -> float | None:
    feature_type = tuple(member.get("feature_type") or ())
    feature_id = tuple(member.get("feature_id") or ())
    gene_symbols = [
        feature_id[i]
        for i, kind in enumerate(feature_type)
        if kind == "sample_gene_z"
    ]
    if not gene_symbols:
        return None
    present = sum(1 for gene in gene_symbols if gene in sample_tpm_by_symbol)
    min_present = min(_MIN_SHARED_GENES, max(50, int(0.8 * len(gene_symbols))))
    if present < min_present:
        return None
    log_values = np.log1p(
        np.asarray(
            [
                max(0.0, float(sample_tpm_by_symbol.get(gene, 0.0) or 0.0))
                for gene in gene_symbols
            ],
            dtype=float,
        )
    )
    gene_features = (log_values - log_values.mean()) / (log_values.std() + 1e-8)
    features: list[float] = []
    gene_i = 0
    for i, kind in enumerate(feature_type):
        if kind == "sample_gene_z":
            features.append(float(gene_features[gene_i]))
            gene_i += 1
        elif kind == "tissue_onehot":
            category = _clean((member.get("tissue_category") or ("",))[i])
            features.append(1.0 if category == context_group else 0.0)
        else:
            return None
    arr = np.asarray(features, dtype=float)
    scale = np.asarray(member.get("scaler_scale"), dtype=float)
    scale = np.where(scale == 0.0, 1.0, scale)
    scaled = (arr - np.asarray(member.get("scaler_mean"), dtype=float)) / scale
    logit = float(member.get("intercept_msi", 0.0)) + float(
        np.dot(scaled, np.asarray(member.get("coef_msi"), dtype=float))
    )
    return _sigmoid(logit)


def _mlh1_retention_context(
    sample_tpm_by_symbol: Mapping[str, float],
) -> dict[str, float] | None:
    """The sample's raw MLH1 clean-TPM (retention is judged cohort-relative downstream).

    MLH1 promoter hypermethylation — the dominant sporadic-MSI mechanism — silences
    MLH1 to a small fraction of its normal level, so retained MLH1 argues *against* that
    mechanism (MSI driven by MSH2/MSH6/PMS2 loss or POLE proofreading mutation leaves
    MLH1 expressed). MLH1 is a moderately-expressed gene, so *within-sample* rank cannot
    see the silencing (a silenced MLH1 still sits above the sample median); whether it is
    retained or silenced is only visible relative to the cohort-typical MLH1, which is
    added where the reference cohort is in scope (``cancer_type_evidence``). Here we only
    surface the sample fact. Returns ``None`` when MLH1 is absent from the sample.
    """
    raw = sample_tpm_by_symbol.get("MLH1")
    if not isinstance(raw, (int, float)) or float(raw) != float(raw):
        return None
    return {"tpm": round(max(0.0, float(raw)), 3)}


def _classify_release_mismatch_repair_expression(
    sample_tpm_by_symbol: Mapping[str, float],
    cancer_type: str,
    *,
    top_k: int = 2,
) -> LearnedExpressionVote | None:
    context_group = mismatch_repair_context_group(cancer_type)
    if not context_group:
        return None
    ensemble = _release_mismatch_repair_ensemble()
    if ensemble is None:
        return None
    member_probs: list[dict[str, Any]] = []
    for member in ensemble["members"]:
        probability = _release_member_msi_probability(
            sample_tpm_by_symbol,
            context_group,
            member,
        )
        if probability is None:
            return None
        member_probs.append(
            {
                "member": member.get("member"),
                "msi_probability": round(float(probability), 6),
            }
        )
    if not member_probs:
        return None
    p_msi = float(np.mean([row["msi_probability"] for row in member_probs]))
    predictions = (("MSI", p_msi), ("MSS", 1.0 - p_msi))
    if p_msi < ensemble["decision_threshold"]:
        predictions = (("MSS", 1.0 - p_msi), ("MSI", p_msi))
    return LearnedExpressionVote(
        stage="mismatch_repair",
        label_space="learned_mismatch_repair_release_ensemble",
        predictions=tuple(predictions[:top_k]),
        training_split_policy=(
            "cbioportal_crc_ucec_stad_sample_gene_z_three_member_ensemble"
        ),
        holdout_top1_accuracy=ensemble["validation"].get(
            "leave_tissue_out_accuracy"
        ),
        training_sample_count=ensemble.get("training_sample_count"),
        training_cohorts=tuple(ensemble.get("training_cohorts") or ()),
        details={
            "context_group": context_group,
            "cancer_type_context": _clean(cancer_type),
            "msi_probability": round(p_msi, 6),
            "decision_threshold": ensemble["decision_threshold"],
            "mlh1_expression": _mlh1_retention_context(sample_tpm_by_symbol),
            "member_probabilities": member_probs,
            "validation": ensemble.get("validation") or {},
            "interpretation": (
                "RNA expression context only; confirm MSI/MMR status with "
                "MSI-PCR, MMR IHC, or validated DNA/RNA clinical testing."
            ),
        },
    )


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


def _sample_rank_vector(
    sample_tpm_by_symbol: Mapping[str, float],
    genes: list[str],
    rank_universe: list[str],
) -> np.ndarray | None:
    """Return flat-model percentiles over its fixed training gene universe.

    Missing training genes are zero, matching an explicit zero-expression row;
    input-only genes are ignored. Thus equivalent profiles cannot change merely
    because a quantifier omits zeros or an annotation supplies extra genes.
    """

    if not sample_tpm_by_symbol:
        return None
    values_by_gene: dict[str, float] = {}
    expressed_features = 0
    feature_set = set(genes)
    for gene in rank_universe:
        try:
            value = float(sample_tpm_by_symbol.get(gene, 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        value = value if np.isfinite(value) and value > 0 else 0.0
        values_by_gene[gene] = value
        if gene in feature_set and value > 0:
            expressed_features += 1
    if expressed_features < _MIN_SHARED_GENES:
        return None
    values = pd.Series(values_by_gene, dtype=float)
    ranks = values.rank(pct=True, method="average")
    return np.asarray([float(ranks.get(gene, 0.0)) for gene in genes])


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
    trained = _trained_model()
    if trained is None:
        return []
    pipe, genes, classes, rank_universe = trained
    vec = _sample_rank_vector(sample_tpm_by_symbol, genes, rank_universe)
    if vec is None:
        return []
    return list(_predict_with_model(pipe, classes, vec, top_k))


def classify_mismatch_repair_expression(
    sample_tpm_by_symbol: Mapping[str, float],
    cancer_type: str | None = None,
    *,
    top_k: int = 2,
) -> LearnedExpressionVote | None:
    """Binary learned RNA vote for MSI-like vs MSS-like expression state.

    This is an orthogonal molecular-state co-signal, not clinical MSI/MMR
    testing. When ``cancer_type`` is a supported context (CRC/COAD/READ, UCEC,
    or STAD), use the packaged cBioPortal TCGA ensemble. Without a supported
    context, fall back to the older representative-cohort vote so direct callers
    can still inspect available paired ``_MSI``/``_MSS`` representative data.
    """

    if not sample_tpm_by_symbol or len(sample_tpm_by_symbol) < _MIN_SHARED_GENES:
        return None
    if cancer_type is not None:
        context_group = mismatch_repair_context_group(cancer_type)
        if not context_group:
            return None
        release_vote = _classify_release_mismatch_repair_expression(
            sample_tpm_by_symbol,
            cancer_type,
            top_k=top_k,
        )
        if release_vote is not None:
            return release_vote
    trained = _trained_mismatch_repair_model()
    if trained is None:
        return None
    pipe, genes, classes, metrics = trained
    vec = _sample_vector(sample_tpm_by_symbol, genes)
    if vec is None:
        return None
    predictions = _predict_with_model(pipe, classes, vec, top_k)
    if not predictions:
        return None
    return LearnedExpressionVote(
        stage="mismatch_repair",
        label_space="learned_mismatch_repair_binary",
        predictions=predictions,
        training_split_policy="leave_one_out_representative_msi_mss_samples",
        holdout_top1_accuracy=metrics.get("holdout_top1_accuracy"),
        training_sample_count=metrics.get("training_sample_count"),
        training_cohorts=tuple(metrics.get("training_cohorts") or ()),
    )


def _mismatch_repair_codes_for_context(
    cancer_type: str,
    classes: list[str],
) -> tuple[tuple[str, ...], str]:
    code = _clean(cancer_type)
    if not code:
        return (), ""
    target_base = _mismatch_repair_base_code(code)
    target_family = _learned_family_for_code(code)

    def with_both_states(candidates: list[str]) -> tuple[str, ...]:
        states = {
            _mismatch_repair_state_for_code(candidate) for candidate in candidates
        }
        states.discard("")
        return tuple(candidates) if {"MSI", "MSS"} <= states else ()

    entity_matches = with_both_states(
        [
            candidate
            for candidate in classes
            if _mismatch_repair_state_for_code(candidate)
            and _mismatch_repair_base_code(candidate) == target_base
        ]
    )
    if entity_matches:
        return entity_matches, "entity"

    family_matches = with_both_states(
        [
            candidate
            for candidate in classes
            if _mismatch_repair_state_for_code(candidate)
            and target_family
            and _learned_family_for_code(candidate) == target_family
        ]
    )
    if family_matches:
        return family_matches, "family"
    return (), ""


def classify_mismatch_repair_sibling_expression(
    sample_tpm_by_symbol: Mapping[str, float],
    cancer_type: str,
    *,
    top_k: int = 2,
) -> LearnedExpressionVote | None:
    """MMR-state vote from fine-grained ``_MSI``/``_MSS`` sibling classes.

    Unlike :func:`classify_mismatch_repair_expression`, this does not train a
    new binary model. It reads the flat learned cancer-type probabilities,
    keeps only paired MSI/MSS siblings for the supplied entity, and falls back
    to the cancer family only when the entity lacks both states.
    """

    if not sample_tpm_by_symbol:
        return None
    trained = _trained_model()
    if trained is None:
        return None
    _pipe, _genes, classes, _rank_universe = trained
    flat = classify_expression(sample_tpm_by_symbol, top_k=max(len(classes), top_k))
    if not flat:
        return None
    return mismatch_repair_sibling_vote_from_predictions(
        flat,
        cancer_type,
        top_k=top_k,
    )


def mismatch_repair_sibling_vote_from_predictions(
    flat_predictions: list[tuple[str, float]] | tuple[tuple[str, float], ...],
    cancer_type: str,
    *,
    top_k: int = 2,
) -> LearnedExpressionVote | None:
    """Contextual MMR vote from an already-computed flat prediction vector."""

    classes = [str(code) for code, _probability in flat_predictions]
    sibling_codes, scope = _mismatch_repair_codes_for_context(cancer_type, classes)
    if not sibling_codes:
        return None
    sibling_set = set(sibling_codes)
    scores = {"MSI": 0.0, "MSS": 0.0}
    for code, probability in flat_predictions:
        if code not in sibling_set:
            continue
        state = _mismatch_repair_state_for_code(code)
        if state in scores:
            scores[state] += float(probability)
    total = sum(scores.values())
    if total <= 0:
        return None
    predictions = tuple(
        sorted(
            ((state, value / total) for state, value in scores.items()),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
    )
    return LearnedExpressionVote(
        stage="mismatch_repair",
        label_space=f"learned_mismatch_repair_sibling_{scope}",
        predictions=predictions,
        training_split_policy="flat_classifier_sibling_probability_renormalization",
    )


def classify_expression_hierarchy(
    sample_tpm_by_symbol: Mapping[str, float],
    *,
    top_k: int = 5,
) -> list[LearnedExpressionVote]:
    """Return stage-scoped learned expression votes.

    Compartment, family, entity, and binary mismatch-repair votes are trained
    as separate logistic models over the same feature space. The subtype-axis
    vote is derived from the flat classifier because those axes are sparse and
    parent-scoped. These votes are evidence rows for candidate admission or
    corroboration; they are not a separate cancer-call oracle.
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
    mismatch_repair_vote = classify_mismatch_repair_expression(
        sample_tpm_by_symbol,
        top_k=2,
    )
    if mismatch_repair_vote is not None:
        votes.append(mismatch_repair_vote)
    return votes


__all__ = [
    "LearnedExpressionVote",
    "classify_expression",
    "classify_expression_hierarchy",
    "classify_mismatch_repair_expression",
    "classify_mismatch_repair_sibling_expression",
    "mismatch_repair_sibling_vote_from_predictions",
    "mismatch_repair_context_group",
]
