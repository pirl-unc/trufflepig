#!/usr/bin/env python
"""Evaluate a broader RNA MSI/dMMR vs MSS/pMMR classifier.

This script is intentionally experimental. It builds a labeled sample matrix
from the locally cached Treehouse 25.01 TCGA per-sample TPM parquets and
cBioPortal molecular-status annotations:

* COAD/READ: MSIsensor score from coadread_tcga_pan_can_atlas_2018.
* UCEC: TCGA molecular subtype from ucec_tcga_pan_can_atlas_2018.
* STAD: TCGA molecular subtype from stad_tcga_pan_can_atlas_2018.

Labels are binary for the mismatch-repair axis:

* MSI: MSI-H, dMMR/MMRd, TCGA MSI subtype.
* MSS: MSS/pMMR/non-MSI classes such as UCEC CN_LOW/CN_HIGH and STAD CIN/GS.

POLE-ultramutated and EBV gastric samples are excluded by default and written
to the sample audit table as confounders rather than folded into either class.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ID_COLS = {"Feature_ID", "Ensembl_Gene_ID", "Symbol", "Hugo_Symbol", "Entrez_Gene_Id"}
DEFAULT_DERIVED = (
    Path.home()
    / ".cache"
    / "pirlygenes"
    / "expression"
    / "treehouse-polya-25-01"
    / "derived"
)
DEFAULT_CBIOPORTAL_CACHE = Path.home() / ".cache" / "trufflepig" / "cbioportal-mmr"
CBIOPORTAL_MEDIA_ROOT = (
    "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public"
)
CBIOPORTAL_RSEM_FILES = {
    "CRC": (
        "coadread_tcga_pan_can_atlas_2018",
        "data_mrna_seq_v2_rsem.txt",
    ),
    "UCEC": (
        "ucec_tcga_pan_can_atlas_2018",
        "data_mrna_seq_v2_rsem.txt",
    ),
    "STAD": (
        "stad_tcga_pan_can_atlas_2018",
        "data_mrna_seq_v2_rsem.txt",
    ),
}


@dataclass(frozen=True)
class LabelRecord:
    sample_id: str
    patient_id: str
    tissue: str
    label: str
    source_value: str
    label_basis: str
    include: bool = True
    exclude_reason: str = ""


@dataclass(frozen=True)
class FeatureConfig:
    n_genes: int
    feature_space: str
    feature_selection: str
    tissue_penalty: float
    include_tissue_onehot: bool = False
    include_tissue_interactions: bool = False


@dataclass(frozen=True)
class ExpressionDataset:
    logc: pd.DataFrame
    sample_cols: list[str]
    labels: np.ndarray
    tissues: np.ndarray
    symbol_by_feature: pd.Series


def _case_id(sample_id: str) -> str:
    return "-".join(str(sample_id).split("-")[:3])


def _sample_cols(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if str(c) not in ID_COLS]


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return ["Feature_ID", "Symbol"] if "Feature_ID" in df.columns else ["Ensembl_Gene_ID", "Symbol"]


def _normalize_expression_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Hugo_Symbol" in out.columns and "Symbol" not in out.columns:
        out = out.rename(columns={"Hugo_Symbol": "Symbol"})
    if "Symbol" not in out.columns:
        raise ValueError("expression matrix needs Symbol or Hugo_Symbol column")
    if "Ensembl_Gene_ID" in out.columns:
        out["Feature_ID"] = out["Ensembl_Gene_ID"].astype(str)
    else:
        out["Feature_ID"] = out["Symbol"].astype(str)
    sample_cols = _sample_cols(out)
    out[sample_cols] = out[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    out = out[out["Symbol"].notna() & out["Feature_ID"].notna()]
    return (
        out[["Feature_ID", "Symbol", *sample_cols]]
        .copy()
        .groupby(["Feature_ID", "Symbol"], as_index=False, sort=False)[sample_cols]
        .sum()
    )


def _download_cbioportal_expression(
    study: str,
    file_name: str,
    cache_dir: Path,
) -> Path:
    out = cache_dir / study / file_name
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{CBIOPORTAL_MEDIA_ROOT}/{study}/{file_name}"
    with urllib.request.urlopen(url, timeout=120) as response, out.open("wb") as handle:
        handle.write(response.read())
    return out


def _load_cbioportal_expression(
    tissue: str,
    cache_dir: Path,
) -> pd.DataFrame:
    study, file_name = CBIOPORTAL_RSEM_FILES[tissue]
    path = _download_cbioportal_expression(study, file_name, cache_dir)
    raw = pd.read_csv(path, sep="\t", low_memory=False)
    return _normalize_expression_frame(raw)


def _fetch_clinical_data(
    study: str,
    attribute_id: str,
    clinical_data_type: str,
    *,
    cache_path: Path,
) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    url = (
        f"https://www.cbioportal.org/api/studies/{study}/clinical-data"
        f"?clinicalDataType={clinical_data_type}"
        f"&attributeId={attribute_id}&projection=SUMMARY&pageSize=10000"
    )
    with urllib.request.urlopen(url, timeout=60) as handle:
        data = json.load(handle)
    df = pd.DataFrame(data)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def _numeric_value(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _crc_score_labels(
    derived_dir: Path,
    *,
    strict: bool,
) -> dict[str, LabelRecord]:
    path = derived_dir / "cbioportal_coadread_msi.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = _fetch_clinical_data(
            "coadread_tcga_pan_can_atlas_2018",
            "MSI_SENSOR_SCORE",
            "SAMPLE",
            cache_path=path,
        )
        if "sampleId" in df.columns and "value" in df.columns:
            df = df[["sampleId", "value"]].rename(columns={"value": "score"})
            df.to_csv(path, index=False)
    out: dict[str, LabelRecord] = {}
    negative_cutoff = 4.0 if strict else 10.0
    for _, row in df.iterrows():
        sample_id = str(row.get("sampleId") or "")
        score = _numeric_value(row.get("score", row.get("value")))
        if not sample_id or score is None:
            continue
        patient_id = _case_id(sample_id)
        if score >= 10.0:
            label = "MSI"
            include = True
            reason = ""
        elif score < negative_cutoff:
            label = "MSS"
            include = True
            reason = ""
        else:
            label = ""
            include = False
            reason = f"indeterminate MSIsensor {score:g}"
        out[patient_id] = LabelRecord(
            sample_id=sample_id,
            patient_id=patient_id,
            tissue="CRC",
            label=label,
            source_value=f"{score:g}",
            label_basis=(
                "cBioPortal COADREAD MSIsensor score; MSI>=10; "
                f"MSS<{negative_cutoff:g}"
            ),
            include=include,
            exclude_reason=reason,
        )
    return out


def _subtype_labels(
    derived_dir: Path,
    *,
    study: str,
    tissue: str,
    cache_name: str,
    positive: set[str],
    negative: set[str],
    excluded: dict[str, str],
) -> dict[str, LabelRecord]:
    path = derived_dir / cache_name
    if path.exists():
        df = pd.read_csv(path)
        patient_col = "patientId"
        value_col = next(
            c for c in ("ucec_subtype", "stad_subtype", "value") if c in df.columns
        )
    else:
        df = _fetch_clinical_data(
            study,
            "SUBTYPE",
            "PATIENT",
            cache_path=path,
        )
        patient_col = "patientId"
        value_col = "value"
        df[[patient_col, value_col]].to_csv(path, index=False)
    out: dict[str, LabelRecord] = {}
    for _, row in df.iterrows():
        patient_id = str(row.get(patient_col) or "")
        subtype = str(row.get(value_col) or "")
        if not patient_id or not subtype:
            continue
        if subtype in positive:
            label = "MSI"
            include = True
            reason = ""
        elif subtype in negative:
            label = "MSS"
            include = True
            reason = ""
        else:
            label = ""
            include = False
            reason = excluded.get(subtype, f"unmapped subtype {subtype}")
        out[patient_id] = LabelRecord(
            sample_id=patient_id,
            patient_id=patient_id,
            tissue=tissue,
            label=label,
            source_value=subtype,
            label_basis=f"cBioPortal {study} SUBTYPE",
            include=include,
            exclude_reason=reason,
        )
    return out


def _load_labeled_samples(
    derived_dir: Path,
    *,
    strict_crc: bool,
    expression_source: str,
    cbioportal_cache: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_maps = {
        "COAD": _crc_score_labels(derived_dir, strict=strict_crc),
        "READ": _crc_score_labels(derived_dir, strict=strict_crc),
        "UCEC": _subtype_labels(
            derived_dir,
            study="ucec_tcga_pan_can_atlas_2018",
            tissue="UCEC",
            cache_name="cbioportal_ucec_subtype.csv",
            positive={"UCEC_MSI"},
            negative={"UCEC_CN_LOW", "UCEC_CN_HIGH", "UCEC_CNL", "UCEC_CNH"},
            excluded={"UCEC_POLE": "POLE hypermutated confounder"},
        ),
        "STAD": _subtype_labels(
            derived_dir,
            study="stad_tcga_pan_can_atlas_2018",
            tissue="STAD",
            cache_name="cbioportal_stad_subtype.csv",
            positive={"STAD_MSI"},
            negative={"STAD_CIN", "STAD_GS"},
            excluded={
                "STAD_EBV": "EBV immune-hot confounder",
                "STAD_POLE": "POLE hypermutated confounder",
            },
        ),
    }
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    if expression_source == "treehouse":
        expression_frames = {}
        for tissue, filename in {
            "COAD": "tcga_coad_per_sample_tpm.parquet",
            "READ": "tcga_read_per_sample_tpm.parquet",
            "UCEC": "tcga_ucec_per_sample_tpm.parquet",
            "STAD": "tcga_stad_per_sample_tpm.parquet",
        }.items():
            path = derived_dir / filename
            if path.exists():
                expression_frames[tissue] = _normalize_expression_frame(
                    pd.read_parquet(path)
                )
    elif expression_source == "cbioportal-rsem":
        expression_frames = {
            tissue: _load_cbioportal_expression(tissue, cbioportal_cache)
            for tissue in ("CRC", "UCEC", "STAD")
        }
        label_maps = dict(label_maps)
        label_maps["CRC"] = label_maps["COAD"]
    else:
        raise ValueError(f"unsupported expression source: {expression_source}")

    for tissue, df in expression_frames.items():
        labels = label_maps[tissue]
        keep_cols: list[str] = []
        rename: dict[str, str] = {}
        for col in _sample_cols(df):
            patient_id = _case_id(col)
            rec = labels.get(patient_id)
            if rec is None:
                audit_rows.append(
                    {
                        "sample_id": col,
                        "patient_id": patient_id,
                        "tissue": tissue,
                        "label": "",
                        "include": False,
                        "source_value": "",
                        "label_basis": "",
                        "exclude_reason": "no matching cBioPortal label",
                    }
                )
                continue
            audit_rows.append(
                {
                    "sample_id": col,
                    "patient_id": patient_id,
                    "tissue": tissue,
                    "label": rec.label,
                    "include": rec.include,
                    "source_value": rec.source_value,
                    "label_basis": rec.label_basis,
                    "exclude_reason": rec.exclude_reason,
                }
            )
            if rec.include:
                keep_cols.append(col)
                rename[col] = f"{tissue}:{col}:{rec.label}"
        if keep_cols:
            subset = df[[*_feature_cols(df), *keep_cols]].rename(columns=rename)
            frames.append(subset)
    if not frames:
        return pd.DataFrame(), pd.DataFrame(audit_rows)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["Feature_ID", "Symbol"], how="inner")
    return merged, pd.DataFrame(audit_rows)


def _dataset_from_expression(expr: pd.DataFrame) -> ExpressionDataset:
    sample_cols = _sample_cols(expr)
    labels = np.asarray([col.rsplit(":", 1)[1] for col in sample_cols])
    tissues = np.asarray([col.split(":", 1)[0] for col in sample_cols])
    clean = expr.set_index("Feature_ID")[sample_cols].astype(float)
    logc = np.log1p(clean.clip(lower=0.0))
    symbol_by_feature = expr.drop_duplicates("Feature_ID").set_index("Feature_ID")[
        "Symbol"
    ]
    return ExpressionDataset(
        logc=logc,
        sample_cols=sample_cols,
        labels=labels,
        tissues=tissues,
        symbol_by_feature=symbol_by_feature,
    )


def _tissue_residualized(
    logc: pd.DataFrame,
    tissues: np.ndarray,
    sample_idx: np.ndarray,
) -> pd.DataFrame:
    """Center expression by tissue medians using only the supplied samples."""

    cols = [logc.columns[i] for i in sample_idx]
    sub = logc[cols].copy()
    for tissue in sorted(set(tissues[sample_idx])):
        tissue_cols = [
            logc.columns[i] for i in sample_idx if str(tissues[i]) == str(tissue)
        ]
        if not tissue_cols:
            continue
        median = sub[tissue_cols].median(axis=1)
        sub[tissue_cols] = sub[tissue_cols].sub(median, axis=0)
    return sub


def _standardized_binary_effect(
    values: pd.DataFrame,
    labels: np.ndarray,
) -> pd.Series:
    msi_cols = [col for col, label in zip(values.columns, labels) if label == "MSI"]
    mss_cols = [col for col, label in zip(values.columns, labels) if label == "MSS"]
    if not msi_cols or not mss_cols:
        return pd.Series(0.0, index=values.index)
    pooled_sd = values.std(axis=1).replace(0.0, np.nan)
    effect = (values[msi_cols].mean(axis=1) - values[mss_cols].mean(axis=1)).abs()
    return (effect / (pooled_sd + 1e-8)).fillna(0.0)


def _tissue_eta_squared(
    values: pd.DataFrame,
    tissues: np.ndarray,
) -> pd.Series:
    if len(set(tissues)) < 2:
        return pd.Series(0.0, index=values.index)
    overall = values.mean(axis=1)
    total = ((values.sub(overall, axis=0)) ** 2).sum(axis=1)
    between = pd.Series(0.0, index=values.index)
    for tissue in sorted(set(tissues)):
        cols = [col for col, value in zip(values.columns, tissues) if value == tissue]
        if not cols:
            continue
        delta = values[cols].mean(axis=1) - overall
        between = between + len(cols) * (delta**2)
    return (between / (total + 1e-8)).clip(lower=0.0, upper=1.0).fillna(0.0)


def _consistent_cross_tissue_mmr(
    values: pd.DataFrame,
    labels: np.ndarray,
    tissues: np.ndarray,
    *,
    tissue_penalty: float,
) -> pd.DataFrame:
    """Score genes whose MSI/MSS contrast is stable and tissue-independent.

    Good genes have a large same-direction MSI-vs-MSS contrast within each
    tissue, little tissue-to-tissue drift in that contrast, and similar
    MSI/MSS midpoints across tissues.
    """

    contrasts: list[pd.Series] = []
    midpoints: list[pd.Series] = []
    weights: list[float] = []
    label_by_col = dict(zip(values.columns, labels))
    for tissue in sorted(set(tissues)):
        tissue_cols = [col for col, value in zip(values.columns, tissues) if value == tissue]
        msi_cols = [col for col in tissue_cols if label_by_col.get(col) == "MSI"]
        mss_cols = [col for col in tissue_cols if label_by_col.get(col) == "MSS"]
        if not msi_cols or not mss_cols:
            continue
        msi_mean = values[msi_cols].mean(axis=1)
        mss_mean = values[mss_cols].mean(axis=1)
        contrasts.append((msi_mean - mss_mean).rename(str(tissue)))
        midpoints.append(((msi_mean + mss_mean) / 2.0).rename(str(tissue)))
        weights.append(math.sqrt((len(msi_cols) * len(mss_cols)) / (len(msi_cols) + len(mss_cols))))

    index = values.index
    if not contrasts:
        return pd.DataFrame(
            {
                "cross_tissue_mmr_strength": 0.0,
                "cross_tissue_direction_consistency": 0.0,
                "cross_tissue_effect_stability": 0.0,
                "cross_tissue_baseline_stability": 0.0,
                "cross_tissue_signed_contrast": 0.0,
                "cross_tissue_higher_state": "",
                "cross_tissue_mmr_score": 0.0,
            },
            index=index,
        )

    contrast_df = pd.concat(contrasts, axis=1)
    midpoint_df = pd.concat(midpoints, axis=1)
    weight_arr = np.asarray(weights, dtype=float)
    weight_sum = float(weight_arr.sum())
    pooled_sd = values.std(axis=1).replace(0.0, np.nan)
    pooled_var = values.var(axis=1).replace(0.0, np.nan)

    weighted_contrast = contrast_df.mul(weight_arr, axis=1).sum(axis=1) / weight_sum
    weighted_abs_contrast = contrast_df.abs().mul(weight_arr, axis=1).sum(axis=1) / weight_sum
    direction_consistency = (
        weighted_contrast.abs() / (weighted_abs_contrast + 1e-8)
    ).clip(lower=0.0, upper=1.0)
    contrast_var = (
        contrast_df.sub(weighted_contrast, axis=0).pow(2).mul(weight_arr, axis=1).sum(axis=1)
        / weight_sum
    )
    effect_stability = 1.0 / (1.0 + contrast_var / (weighted_abs_contrast.pow(2) + 1e-8))

    weighted_midpoint = midpoint_df.mul(weight_arr, axis=1).sum(axis=1) / weight_sum
    midpoint_var = (
        midpoint_df.sub(weighted_midpoint, axis=0).pow(2).mul(weight_arr, axis=1).sum(axis=1)
        / weight_sum
    )
    baseline_stability = 1.0 / (1.0 + tissue_penalty * midpoint_var / (pooled_var + 1e-8))
    strength = weighted_contrast.abs() / (pooled_sd + 1e-8)
    score = strength * direction_consistency * effect_stability * baseline_stability
    return pd.DataFrame(
        {
            "cross_tissue_mmr_strength": strength.fillna(0.0),
            "cross_tissue_direction_consistency": direction_consistency.fillna(0.0),
            "cross_tissue_effect_stability": effect_stability.fillna(0.0),
            "cross_tissue_baseline_stability": baseline_stability.fillna(0.0),
            "cross_tissue_signed_contrast": weighted_contrast.fillna(0.0),
            "cross_tissue_higher_state": np.where(weighted_contrast >= 0.0, "MSI", "MSS"),
            "cross_tissue_mmr_score": score.fillna(0.0),
        },
        index=index,
    )


def _feature_score_table(
    dataset: ExpressionDataset,
    train_idx: np.ndarray,
    config: FeatureConfig,
) -> pd.DataFrame:
    train_cols = [dataset.sample_cols[i] for i in train_idx]
    train_logc = dataset.logc[train_cols]
    train_labels = dataset.labels[train_idx]
    train_tissues = dataset.tissues[train_idx]
    variance = train_logc.var(axis=1).fillna(0.0)
    raw_effect = _standardized_binary_effect(train_logc, train_labels)
    residual = _tissue_residualized(dataset.logc, dataset.tissues, train_idx)
    residual_effect = _standardized_binary_effect(residual, train_labels)
    tissue_effect = _tissue_eta_squared(train_logc, train_tissues)
    cross_tissue = _consistent_cross_tissue_mmr(
        train_logc,
        train_labels,
        train_tissues,
        tissue_penalty=config.tissue_penalty,
    )
    if config.feature_selection == "variance":
        score = variance
    elif config.feature_selection == "mmr_effect":
        score = raw_effect
    elif config.feature_selection == "residual_mmr_effect":
        score = residual_effect
    elif config.feature_selection == "tissue_independent_mmr":
        score = residual_effect / (1.0 + config.tissue_penalty * tissue_effect)
    elif config.feature_selection == "consistent_cross_tissue_mmr":
        score = cross_tissue["cross_tissue_mmr_score"]
    else:
        raise ValueError(f"unsupported feature selection: {config.feature_selection}")
    return pd.DataFrame(
        {
            "Feature_ID": train_logc.index,
            "Symbol": dataset.symbol_by_feature.reindex(train_logc.index).fillna(""),
            "selection_score": score.reindex(train_logc.index).fillna(0.0).values,
            "variance": variance.reindex(train_logc.index).fillna(0.0).values,
            "mmr_effect": raw_effect.reindex(train_logc.index).fillna(0.0).values,
            "residual_mmr_effect": residual_effect.reindex(train_logc.index)
            .fillna(0.0)
            .values,
            "tissue_eta_squared": tissue_effect.reindex(train_logc.index)
            .fillna(0.0)
            .values,
            "cross_tissue_mmr_strength": cross_tissue[
                "cross_tissue_mmr_strength"
            ].values,
            "cross_tissue_direction_consistency": cross_tissue[
                "cross_tissue_direction_consistency"
            ].values,
            "cross_tissue_effect_stability": cross_tissue[
                "cross_tissue_effect_stability"
            ].values,
            "cross_tissue_baseline_stability": cross_tissue[
                "cross_tissue_baseline_stability"
            ].values,
            "cross_tissue_signed_contrast": cross_tissue[
                "cross_tissue_signed_contrast"
            ].values,
            "cross_tissue_higher_state": cross_tissue[
                "cross_tissue_higher_state"
            ].values,
            "cross_tissue_mmr_score": cross_tissue["cross_tissue_mmr_score"].values,
        }
    ).sort_values("selection_score", ascending=False)


def _select_genes(
    dataset: ExpressionDataset,
    train_idx: np.ndarray,
    config: FeatureConfig,
) -> tuple[list[str], pd.DataFrame]:
    scores = _feature_score_table(dataset, train_idx, config)
    genes = [str(gene) for gene in scores["Feature_ID"].head(config.n_genes)]
    return genes, scores


def _residual_features(
    dataset: ExpressionDataset,
    genes: list[str],
    sample_idx: np.ndarray,
    train_idx: np.ndarray,
) -> np.ndarray:
    train_cols = [dataset.sample_cols[i] for i in train_idx]
    train_logc = dataset.logc.loc[genes, train_cols]
    global_median = train_logc.median(axis=1)
    median_by_tissue = {
        str(tissue): train_logc[
            [col for col, i in zip(train_cols, train_idx) if dataset.tissues[i] == tissue]
        ].median(axis=1)
        for tissue in sorted(set(dataset.tissues[train_idx]))
    }
    columns = []
    for i in sample_idx:
        col = dataset.sample_cols[i]
        tissue = str(dataset.tissues[i])
        baseline = median_by_tissue.get(tissue, global_median)
        columns.append(dataset.logc.loc[genes, col].sub(baseline).rename(col))
    return pd.concat(columns, axis=1).T.to_numpy()


def _one_hot_features(
    dataset: ExpressionDataset,
    sample_idx: np.ndarray,
) -> np.ndarray:
    categories = sorted(str(tissue) for tissue in set(dataset.tissues))
    rows = np.zeros((len(sample_idx), len(categories)), dtype=float)
    lookup = {category: i for i, category in enumerate(categories)}
    for row, sample_i in enumerate(sample_idx):
        rows[row, lookup[str(dataset.tissues[sample_i])]] = 1.0
    return rows


def _build_features(
    dataset: ExpressionDataset,
    genes: list[str],
    sample_idx: np.ndarray,
    train_idx: np.ndarray,
    config: FeatureConfig,
) -> np.ndarray:
    sample_cols = [dataset.sample_cols[i] for i in sample_idx]
    logc = dataset.logc.loc[genes, sample_cols]
    if config.feature_space == "raw_log":
        features = logc.T.to_numpy()
    elif config.feature_space == "within_sample_rank":
        features = logc.rank(axis=0, pct=True).T.to_numpy()
    elif config.feature_space == "sample_gene_z":
        arr = logc.T.to_numpy()
        features = (arr - arr.mean(axis=1, keepdims=True)) / (
            arr.std(axis=1, keepdims=True) + 1e-8
        )
    elif config.feature_space == "tissue_residual":
        features = _residual_features(dataset, genes, sample_idx, train_idx)
    elif config.feature_space == "rank_plus_tissue_residual":
        ranks = logc.rank(axis=0, pct=True).T.to_numpy()
        residual = _residual_features(dataset, genes, sample_idx, train_idx)
        features = np.concatenate([ranks, residual], axis=1)
    else:
        raise ValueError(f"unsupported feature space: {config.feature_space}")
    additions: list[np.ndarray] = []
    if config.include_tissue_interactions:
        onehot = _one_hot_features(dataset, sample_idx)
        additions.append(
            np.concatenate([features * onehot[:, [i]] for i in range(onehot.shape[1])], axis=1)
        )
    if config.include_tissue_onehot:
        additions.append(_one_hot_features(dataset, sample_idx))
    if additions:
        features = np.concatenate([features, *additions], axis=1)
    X = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _fit_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="newton-cg",
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
        ),
    )


def _eval_predictions(
    *,
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tissues: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = ["MSI", "MSS"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows.append(
        {
            "evaluation": name,
            "subset": "overall",
            "n": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "msi_as_msi": int(cm[0, 0]),
            "msi_as_mss": int(cm[0, 1]),
            "mss_as_msi": int(cm[1, 0]),
            "mss_as_mss": int(cm[1, 1]),
        }
    )
    for tissue in sorted(set(tissues)):
        mask = tissues == tissue
        if not mask.any():
            continue
        cm_t = confusion_matrix(y_true[mask], y_pred[mask], labels=labels)
        rows.append(
            {
                "evaluation": name,
                "subset": str(tissue),
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
                "msi_as_msi": int(cm_t[0, 0]),
                "msi_as_mss": int(cm_t[0, 1]),
                "mss_as_msi": int(cm_t[1, 0]),
                "mss_as_mss": int(cm_t[1, 1]),
            }
        )
    return rows


def _fit_predict_fold(
    dataset: ExpressionDataset,
    config: FeatureConfig,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    genes, _scores = _select_genes(dataset, train_idx, config)
    preds, _prob = _fit_predict_fold_with_genes(
        dataset,
        config,
        train_idx,
        test_idx,
        genes,
    )
    return preds


def _fit_predict_fold_with_genes(
    dataset: ExpressionDataset,
    config: FeatureConfig,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    genes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    X_train = _build_features(dataset, genes, train_idx, train_idx, config)
    X_test = _build_features(dataset, genes, test_idx, train_idx, config)
    model = _fit_model()
    model.fit(X_train, dataset.labels[train_idx])
    preds = model.predict(X_test).astype(str)
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if "MSI" in classes:
            msi_idx = classes.index("MSI")
            prob = model.predict_proba(X_test)[:, msi_idx]
        else:
            prob = np.zeros(len(test_idx), dtype=float)
    else:
        prob = (preds == "MSI").astype(float)
    return preds, np.asarray(prob, dtype=float)


def _cross_validated_predictions(
    dataset: ExpressionDataset,
    config: FeatureConfig,
    *,
    splits: int,
) -> np.ndarray:
    y = dataset.labels
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)
    preds = np.empty(len(y), dtype=object)
    for train_idx, test_idx in cv.split(np.zeros(len(y)), y):
        preds[test_idx] = _fit_predict_fold(dataset, config, train_idx, test_idx)
    return preds.astype(str)


def _leave_tissue_out_predictions(
    dataset: ExpressionDataset,
    config: FeatureConfig,
) -> np.ndarray:
    y = dataset.labels
    tissues = dataset.tissues
    preds = np.empty(len(y), dtype=object)
    for train_idx, test_idx in GroupKFold(n_splits=len(set(tissues))).split(
        np.zeros(len(y)),
        y,
        groups=tissues,
    ):
        preds[test_idx] = _fit_predict_fold(dataset, config, train_idx, test_idx)
    return preds.astype(str)


def _stratified_splits(
    dataset: ExpressionDataset,
    *,
    splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = dataset.labels
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)
    return [
        (np.asarray(train_idx), np.asarray(test_idx))
        for train_idx, test_idx in cv.split(np.zeros(len(y)), y)
    ]


def _leave_tissue_out_splits(
    dataset: ExpressionDataset,
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = dataset.labels
    tissues = dataset.tissues
    return [
        (np.asarray(train_idx), np.asarray(test_idx))
        for train_idx, test_idx in GroupKFold(n_splits=len(set(tissues))).split(
            np.zeros(len(y)),
            y,
            groups=tissues,
        )
    ]


def _within_tissue_splits(
    dataset: ExpressionDataset,
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = dataset.labels
    tissues = dataset.tissues
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for tissue in sorted(set(tissues)):
        idx = np.where(tissues == tissue)[0]
        counts = pd.Series(y[idx]).value_counts()
        if len(counts) < 2 or int(counts.min()) < 2:
            continue
        splits = min(5, int(counts.min()))
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)
        for train_local, test_local in cv.split(np.zeros(len(idx)), y[idx]):
            out.append((idx[train_local], idx[test_local]))
    return out


def _grid_predictions_for_splits(
    dataset: ExpressionDataset,
    config: FeatureConfig,
    splits: list[tuple[np.ndarray, np.ndarray]],
    gene_counts: list[int],
) -> dict[int, np.ndarray]:
    if not splits:
        return {}
    max_genes = max(gene_counts)
    ranking_config = replace(config, n_genes=max_genes)
    preds_by_count = {
        n_genes: np.full(len(dataset.labels), "", dtype=object)
        for n_genes in gene_counts
    }
    for train_idx, test_idx in splits:
        genes, _scores = _select_genes(dataset, train_idx, ranking_config)
        for n_genes in gene_counts:
            preds, _prob = _fit_predict_fold_with_genes(
                dataset,
                replace(config, n_genes=n_genes),
                train_idx,
                test_idx,
                genes[:n_genes],
            )
            preds_by_count[n_genes][test_idx] = preds
    return {n_genes: preds.astype(str) for n_genes, preds in preds_by_count.items()}


def _ensemble_predictions_for_splits(
    dataset: ExpressionDataset,
    base_config: FeatureConfig,
    splits: list[tuple[np.ndarray, np.ndarray]],
    members: list[tuple[str, int]],
) -> dict[str, np.ndarray]:
    hard_votes_by_sample = np.zeros(len(dataset.labels), dtype=float)
    prob_sum_by_sample = np.zeros(len(dataset.labels), dtype=float)
    vote_count_by_sample = np.zeros(len(dataset.labels), dtype=float)
    for train_idx, test_idx in splits:
        fold_probs: list[np.ndarray] = []
        fold_preds: list[np.ndarray] = []
        for feature_selection, n_genes in members:
            config = replace(
                base_config,
                feature_selection=feature_selection,
                n_genes=n_genes,
            )
            genes, _scores = _select_genes(dataset, train_idx, config)
            preds, prob = _fit_predict_fold_with_genes(
                dataset,
                config,
                train_idx,
                test_idx,
                genes,
            )
            fold_preds.append(preds)
            fold_probs.append(prob)
        probs = np.vstack(fold_probs)
        hard_votes = np.vstack([(preds == "MSI").astype(float) for preds in fold_preds])
        prob_sum_by_sample[test_idx] = probs.sum(axis=0)
        hard_votes_by_sample[test_idx] = hard_votes.sum(axis=0)
        vote_count_by_sample[test_idx] = len(members)

    mean_prob = np.divide(
        prob_sum_by_sample,
        vote_count_by_sample,
        out=np.zeros_like(prob_sum_by_sample),
        where=vote_count_by_sample > 0,
    )
    half = vote_count_by_sample / 2.0
    return {
        "ensemble_mean_prob_0.50": np.where(mean_prob >= 0.50, "MSI", "MSS"),
        "ensemble_mean_prob_0.40": np.where(mean_prob >= 0.40, "MSI", "MSS"),
        "ensemble_mean_prob_0.60": np.where(mean_prob >= 0.60, "MSI", "MSS"),
        "ensemble_majority": np.where(hard_votes_by_sample > half, "MSI", "MSS"),
        "ensemble_any_msi": np.where(hard_votes_by_sample >= 1.0, "MSI", "MSS"),
        "ensemble_all_msi": np.where(hard_votes_by_sample >= vote_count_by_sample, "MSI", "MSS"),
    }


def _parse_csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_gene_counts(value: str) -> list[int]:
    counts = [int(item) for item in _parse_csv_items(value)]
    if not counts:
        raise ValueError("gene-count grid is empty")
    if min(counts) <= 0:
        raise ValueError("gene counts must be positive")
    return sorted(set(counts))


def _tagged_eval_rows(
    *,
    config: FeatureConfig,
    gene_count: int,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tagged = []
    for row in rows:
        tagged.append(
            {
                "feature_space": config.feature_space,
                "feature_selection": config.feature_selection,
                "n_genes": gene_count,
                "include_tissue_onehot": config.include_tissue_onehot,
                "include_tissue_interactions": config.include_tissue_interactions,
                "tissue_penalty": config.tissue_penalty,
                **row,
            }
        )
    return tagged


def _run_gene_selection_grid(
    *,
    dataset: ExpressionDataset,
    config: FeatureConfig,
    feature_selections: list[str],
    gene_counts: list[int],
    out_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_count = min(5, int(np.min(np.bincount(pd.Categorical(dataset.labels).codes))))
    split_sets: list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]] = []
    if split_count >= 2:
        split_sets.append(
            (
                f"{split_count}-fold_stratified",
                _stratified_splits(dataset, splits=split_count),
            )
        )
    split_sets.append(("leave_tissue_out", _leave_tissue_out_splits(dataset)))
    split_sets.append(("within_tissue_cv", _within_tissue_splits(dataset)))

    for feature_selection in feature_selections:
        selection_config = replace(
            config,
            feature_selection=feature_selection,
            n_genes=max(gene_counts),
        )
        _genes, scores = _select_genes(
            dataset,
            np.arange(len(dataset.labels)),
            selection_config,
        )
        scores.head(max(gene_counts)).to_csv(
            out_dir / f"mmr_feature_genes_{feature_selection}.tsv",
            sep="\t",
            index=False,
        )
        for eval_name, splits in split_sets:
            preds_by_count = _grid_predictions_for_splits(
                dataset,
                selection_config,
                splits,
                gene_counts,
            )
            for gene_count, preds in preds_by_count.items():
                scored = preds != ""
                if not scored.any():
                    continue
                eval_rows = _eval_predictions(
                    name=eval_name,
                    y_true=dataset.labels[scored],
                    y_pred=preds[scored],
                    tissues=dataset.tissues[scored],
                )
                rows.extend(
                    _tagged_eval_rows(
                        config=selection_config,
                        gene_count=gene_count,
                        rows=eval_rows,
                    )
                )
    return pd.DataFrame(rows)


def _feature_ids_for_config(
    dataset: ExpressionDataset,
    genes: list[str],
    config: FeatureConfig,
) -> list[dict[str, str]]:
    if config.include_tissue_interactions:
        raise ValueError("model export does not support tissue interactions")
    if config.feature_space == "rank_plus_tissue_residual":
        rows = [
            {
                "feature_type": "within_sample_rank",
                "feature_id": gene,
                "symbol": str(dataset.symbol_by_feature.get(gene, "")),
                "tissue_category": "",
            }
            for gene in genes
        ]
        rows.extend(
            {
                "feature_type": "tissue_residual",
                "feature_id": gene,
                "symbol": str(dataset.symbol_by_feature.get(gene, "")),
                "tissue_category": "",
            }
            for gene in genes
        )
    else:
        rows = [
            {
                "feature_type": config.feature_space,
                "feature_id": gene,
                "symbol": str(dataset.symbol_by_feature.get(gene, "")),
                "tissue_category": "",
            }
            for gene in genes
        ]
    if config.include_tissue_onehot:
        rows.extend(
            {
                "feature_type": "tissue_onehot",
                "feature_id": f"tissue:{tissue}",
                "symbol": "",
                "tissue_category": str(tissue),
            }
            for tissue in sorted(str(tissue) for tissue in set(dataset.tissues))
        )
    return rows


def _coef_for_msi(model) -> tuple[np.ndarray, float]:
    lr = model.named_steps["logisticregression"]
    classes = [str(item) for item in lr.classes_]
    coef = lr.coef_[0].astype(float)
    intercept = float(lr.intercept_[0])
    if len(classes) == 2 and classes[1] == "MSS":
        return -coef, -intercept
    if len(classes) == 2 and classes[1] == "MSI":
        return coef, intercept
    raise ValueError(f"expected binary MSI/MSS classes, got {classes}")


def _export_practical_ensemble_model(
    *,
    dataset: ExpressionDataset,
    config: FeatureConfig,
    model_path: Path,
    metadata_path: Path,
) -> None:
    members = [
        ("mmr_effect", 150),
        ("tissue_independent_mmr", 250),
        ("consistent_cross_tissue_mmr", 500),
    ]
    all_idx = np.arange(len(dataset.labels))
    rows: list[dict[str, Any]] = []
    for member_index, (feature_selection, n_genes) in enumerate(members):
        member_config = replace(
            config,
            feature_space="sample_gene_z",
            feature_selection=feature_selection,
            n_genes=n_genes,
            include_tissue_onehot=True,
            include_tissue_interactions=False,
        )
        genes, _scores = _select_genes(dataset, all_idx, member_config)
        X = _build_features(dataset, genes, all_idx, all_idx, member_config)
        model = _fit_model()
        model.fit(X, dataset.labels)
        scaler = model.named_steps["standardscaler"]
        coef_msi, intercept_msi = _coef_for_msi(model)
        feature_rows = _feature_ids_for_config(dataset, genes, member_config)
        if len(feature_rows) != len(coef_msi):
            raise ValueError(
                f"feature/coef mismatch for {feature_selection}@{n_genes}: "
                f"{len(feature_rows)} != {len(coef_msi)}"
            )
        member_name = f"{feature_selection}@{n_genes}"
        for feature_index, feature in enumerate(feature_rows):
            rows.append(
                {
                    "artifact_version": 1,
                    "ensemble_name": "mmr_expression_practical_ensemble",
                    "member_index": member_index,
                    "member": member_name,
                    "feature_space": member_config.feature_space,
                    "feature_selection": feature_selection,
                    "n_genes": n_genes,
                    "decision_threshold": 0.50,
                    "feature_index": feature_index,
                    **feature,
                    "scaler_mean": float(scaler.mean_[feature_index]),
                    "scaler_scale": float(scaler.scale_[feature_index]),
                    "coef_msi": float(coef_msi[feature_index]),
                    "intercept_msi": intercept_msi,
                }
            )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(model_path, index=False)
    label_counts = pd.Series(dataset.labels).value_counts().to_dict()
    metadata_rows = [
        {
            "key": "artifact_version",
            "value": "1",
        },
        {
            "key": "model",
            "value": "mean_probability_ensemble",
        },
        {
            "key": "members",
            "value": ",".join(f"{name}@{count}" for name, count in members),
        },
        {
            "key": "feature_space",
            "value": "sample_gene_z",
        },
        {
            "key": "decision_threshold",
            "value": "0.50",
        },
        {
            "key": "training_samples",
            "value": str(len(dataset.labels)),
        },
        {
            "key": "training_msi",
            "value": str(int(label_counts.get("MSI", 0))),
        },
        {
            "key": "training_mss",
            "value": str(int(label_counts.get("MSS", 0))),
        },
        {
            "key": "training_tissues",
            "value": ",".join(sorted(str(tissue) for tissue in set(dataset.tissues))),
        },
        {
            "key": "training_source",
            "value": "cBioPortal TCGA PanCancer Atlas RSEM; CRC MSIsensor, UCEC/STAD molecular subtype labels",
        },
    ]
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)


def _within_tissue_predictions(
    dataset: ExpressionDataset,
    config: FeatureConfig,
) -> np.ndarray:
    y = dataset.labels
    tissues = dataset.tissues
    preds = np.empty(len(y), dtype=object)
    for tissue in sorted(set(tissues)):
        mask = tissues == tissue
        idx = np.where(mask)[0]
        counts = pd.Series(y[idx]).value_counts()
        if len(counts) < 2 or int(counts.min()) < 2:
            preds[idx] = ""
            continue
        splits = min(5, int(counts.min()))
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)
        for train_local, test_local in cv.split(np.zeros(len(idx)), y[idx]):
            train_idx = idx[train_local]
            test_idx = idx[test_local]
            preds[test_idx] = _fit_predict_fold(dataset, config, train_idx, test_idx)
    return preds.astype(str)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-dir", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--cbioportal-cache", type=Path, default=DEFAULT_CBIOPORTAL_CACHE)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/trufflepig-mmr-classifier"))
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument(
        "--feature-space",
        choices=(
            "raw_log",
            "within_sample_rank",
            "sample_gene_z",
            "tissue_residual",
            "rank_plus_tissue_residual",
        ),
        default="raw_log",
    )
    parser.add_argument(
        "--feature-selection",
        choices=(
            "variance",
            "mmr_effect",
            "residual_mmr_effect",
            "tissue_independent_mmr",
            "consistent_cross_tissue_mmr",
        ),
        default="variance",
        help=(
            "variance is unsupervised; mmr_effect is a supervised MSI/MSS "
            "effect size; residual_mmr_effect uses within-training-tissue "
            "centering; tissue_independent_mmr penalizes genes whose expression "
            "is dominated by tissue identity; consistent_cross_tissue_mmr "
            "rewards same-direction MSI/MSS contrast and stable tissue midpoints."
        ),
    )
    parser.add_argument(
        "--grid-n-genes",
        default="",
        help="Comma-separated gene counts to evaluate in one fold-safe grid.",
    )
    parser.add_argument(
        "--grid-feature-selections",
        default="",
        help=(
            "Comma-separated feature-selection procedures to evaluate in one "
            "grid. Defaults to --feature-selection when omitted."
        ),
    )
    parser.add_argument(
        "--evaluate-practical-ensemble",
        action="store_true",
        help=(
            "Evaluate the current three-member practical MSI/MSS ensemble: "
            "mmr_effect@150, tissue_independent_mmr@250, and "
            "consistent_cross_tissue_mmr@500."
        ),
    )
    parser.add_argument(
        "--export-practical-ensemble-model",
        type=Path,
        default=None,
        help="Write the final practical MSI/MSS ensemble model CSV artifact.",
    )
    parser.add_argument(
        "--tissue-penalty",
        type=float,
        default=4.0,
        help="Penalty multiplier for tissue_eta_squared in tissue_independent_mmr.",
    )
    parser.add_argument(
        "--include-tissue-onehot",
        action="store_true",
        help="Append tissue one-hot features as calibration/intercept context.",
    )
    parser.add_argument(
        "--include-tissue-interactions",
        action="store_true",
        help=(
            "Append expression-by-tissue interaction blocks, retaining the "
            "pooled expression features as fallback."
        ),
    )
    parser.add_argument(
        "--expression-source",
        choices=("treehouse", "cbioportal-rsem"),
        default="treehouse",
    )
    parser.add_argument(
        "--crc-policy",
        choices=("strict", "permissive"),
        default="strict",
        help=(
            "strict uses CRC MSIsensor MSI>=10 and MSS<4; permissive uses "
            "MSI>=10 and MSS<10, matching the existing representative split."
        ),
    )
    args = parser.parse_args()

    expr, audit = _load_labeled_samples(
        args.derived_dir,
        strict_crc=args.crc_policy == "strict",
        expression_source=args.expression_source,
        cbioportal_cache=args.cbioportal_cache,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.out_dir / "mmr_sample_label_audit.tsv", sep="\t", index=False)
    if expr.empty:
        raise SystemExit(f"no labeled expression samples found in {args.derived_dir}")

    config = FeatureConfig(
        n_genes=args.n_genes,
        feature_space=args.feature_space,
        feature_selection=args.feature_selection,
        tissue_penalty=args.tissue_penalty,
        include_tissue_onehot=args.include_tissue_onehot,
        include_tissue_interactions=args.include_tissue_interactions,
    )
    dataset = _dataset_from_expression(expr)
    all_idx = np.arange(len(dataset.labels))
    _genes, scores = _select_genes(dataset, all_idx, config)
    y = dataset.labels
    tissues = dataset.tissues
    sample_cols = dataset.sample_cols
    included = pd.DataFrame(
        {
            "sample": sample_cols,
            "tissue": tissues,
            "label": y,
        }
    )
    included.to_csv(args.out_dir / "mmr_training_samples.tsv", sep="\t", index=False)
    scores.head(args.n_genes).to_csv(
        args.out_dir / "mmr_feature_genes.tsv",
        sep="\t",
        index=False,
    )

    if args.export_practical_ensemble_model is not None:
        model_path = args.export_practical_ensemble_model
        metadata_path = model_path.with_name(model_path.stem + "-metadata.csv")
        _export_practical_ensemble_model(
            dataset=dataset,
            config=config,
            model_path=model_path,
            metadata_path=metadata_path,
        )
        print(f"wrote model artifact: {model_path}")
        print(f"wrote model metadata: {metadata_path}")

    if args.grid_n_genes:
        gene_counts = _parse_gene_counts(args.grid_n_genes)
        feature_selections = (
            _parse_csv_items(args.grid_feature_selections)
            if args.grid_feature_selections
            else [args.feature_selection]
        )
        allowed_feature_selections = {
            "variance",
            "mmr_effect",
            "residual_mmr_effect",
            "tissue_independent_mmr",
            "consistent_cross_tissue_mmr",
        }
        unknown = sorted(set(feature_selections) - allowed_feature_selections)
        if unknown:
            raise SystemExit(f"unknown feature selections: {', '.join(unknown)}")
        grid_df = _run_gene_selection_grid(
            dataset=dataset,
            config=config,
            feature_selections=feature_selections,
            gene_counts=gene_counts,
            out_dir=args.out_dir,
        )
        grid_df.to_csv(
            args.out_dir / "mmr_classifier_grid_evaluation.tsv",
            sep="\t",
            index=False,
        )
        print(f"wrote {args.out_dir}")
        print("included samples:")
        print(included.groupby(["tissue", "label"]).size().unstack(fill_value=0))
        print("\ngrid evaluation:")
        overall = grid_df[
            (grid_df["subset"] == "overall")
            & grid_df["evaluation"].isin(
                ["5-fold_stratified", "within_tissue_cv", "leave_tissue_out"]
            )
        ]
        print(
            overall.pivot_table(
                index=["feature_selection", "n_genes"],
                columns="evaluation",
                values="accuracy",
                aggfunc="first",
            ).to_string(float_format=lambda value: f"{value:.4f}")
        )
        return 0

    if args.evaluate_practical_ensemble:
        members = [
            ("mmr_effect", 150),
            ("tissue_independent_mmr", 250),
            ("consistent_cross_tissue_mmr", 500),
        ]
        rows: list[dict[str, Any]] = []
        split_count = min(5, int(np.min(np.bincount(pd.Categorical(y).codes))))
        split_sets: list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]] = []
        if split_count >= 2:
            split_sets.append(
                (
                    f"{split_count}-fold_stratified",
                    _stratified_splits(dataset, splits=split_count),
                )
            )
        split_sets.append(("leave_tissue_out", _leave_tissue_out_splits(dataset)))
        split_sets.append(("within_tissue_cv", _within_tissue_splits(dataset)))
        for eval_name, splits in split_sets:
            for ensemble_name, preds in _ensemble_predictions_for_splits(
                dataset,
                config,
                splits,
                members,
            ).items():
                rows.extend(
                    {
                        "ensemble": ensemble_name,
                        "members": ",".join(f"{name}@{count}" for name, count in members),
                        **row,
                    }
                    for row in _eval_predictions(
                        name=eval_name,
                        y_true=y,
                        y_pred=preds,
                        tissues=tissues,
                    )
                )
        ensemble_df = pd.DataFrame(rows)
        ensemble_df.to_csv(
            args.out_dir / "mmr_classifier_practical_ensemble.tsv",
            sep="\t",
            index=False,
        )
        print(f"wrote {args.out_dir}")
        print("included samples:")
        print(included.groupby(["tissue", "label"]).size().unstack(fill_value=0))
        print("\nensemble evaluation:")
        overall = ensemble_df[
            (ensemble_df["subset"] == "overall")
            & ensemble_df["evaluation"].isin(
                ["5-fold_stratified", "within_tissue_cv", "leave_tissue_out"]
            )
        ]
        print(
            overall.pivot_table(
                index="ensemble",
                columns="evaluation",
                values="accuracy",
                aggfunc="first",
            ).to_string(float_format=lambda value: f"{value:.4f}")
        )
        return 0

    rows: list[dict[str, Any]] = []
    split_count = min(5, int(np.min(np.bincount(pd.Categorical(y).codes))))
    if split_count >= 2:
        pred = _cross_validated_predictions(dataset, config, splits=split_count)
        rows.extend(
            _eval_predictions(
                name=f"{split_count}-fold_stratified",
                y_true=y,
                y_pred=pred,
                tissues=tissues,
            )
        )
    pred_lto = _leave_tissue_out_predictions(dataset, config)
    rows.extend(
        _eval_predictions(
            name="leave_tissue_out",
            y_true=y,
            y_pred=pred_lto,
            tissues=tissues,
        )
    )
    pred_within = _within_tissue_predictions(dataset, config)
    scored = pred_within != ""
    if scored.any():
        rows.extend(
            _eval_predictions(
                name="within_tissue_cv",
                y_true=y[scored],
                y_pred=pred_within[scored],
                tissues=tissues[scored],
            )
        )
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(args.out_dir / "mmr_classifier_evaluation.tsv", sep="\t", index=False)

    print(f"wrote {args.out_dir}")
    print("included samples:")
    print(included.groupby(["tissue", "label"]).size().unstack(fill_value=0))
    print("\nexcluded/confounder samples:")
    excluded = audit[~audit["include"].astype(bool)]
    if excluded.empty:
        print("none")
    else:
        print(
            excluded.groupby(["tissue", "exclude_reason"])
            .size()
            .sort_values(ascending=False)
            .head(20)
        )
    print("\nevaluation:")
    print(eval_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
