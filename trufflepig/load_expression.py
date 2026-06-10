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

import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .aggregate_gene_expression import aggregate_gene_expression as tx2gene
from pirlygenes.gene_ids import (
    find_canonical_gene_ids_and_names,
    find_gene_name_from_ensembl_gene_id,
)
from pirlygenes.gene_names import display_name, short_gene_name

import re

from trufflepig.expression_qc import (
    classify_gene_qc,
    dominant_class_phrase,
    is_rescue_feature,
    summarize_qc_class_shares,
)
from trufflepig.expression_normalize import normalize_expression
from trufflepig.clean_tpm import assert_clean_tpm, technical_rna_mask

# Patterns for fuzzy column guessing (tried in order, first match wins).
# Each is matched case-insensitively against the full column name.
_GENE_NAME_PATTERNS = [
    r"^gene[_\s]?sym",  # gene_symbol, gene symbol, genesym...
    r"^gene[_\s]?name",  # gene_name, gene name
    r"^symbol$",
    r"^gene$",
    r"^name$",
    r"^hgnc[_\s]?symbol",
    r"^hugo[_\s]?symbol",
]
_GENE_ID_PATTERNS = [
    r"ensembl.*gene.*id",  # ensembl_gene_id, Ensembl Gene ID, ...
    r"^gene[_\s]?id$",
    r"^ensg$",
]
_TPM_PATTERNS = [
    r"^TPM$",
    r"^gene[_\s]?tpm",
    r"(^|[_\s])tpm($|[_\s])",
]

# Strict FPKM column patterns — match raw FPKM columns only, not derived
# columns like log2_fpkm, FPKM_zscore, FPKM_adjusted, or legacy
# per-cohort columns like FPKM_COAD.  A full column-name match is required.
_RAW_FPKM_PATTERNS = [
    r"^FPKM$",
    r"^fpkm$",
    r"^gene[_\s]?fpkm$",
    r"^rna[_\s]?fpkm$",
    r"^mrna[_\s]?fpkm$",
]

_MAX_UNRESOLVED_TRANSCRIPT_TPM_FRACTION = 0.05
_TPM_SUM_EXPECTED = 1_000_000.0
_TPM_SUM_MIN_REASONABLE = 500_000.0
_TPM_SUM_MAX_REASONABLE = 2_000_000.0
_LOG2_TPM_MAX_CANDIDATE = 30.0
_LOG2_TPM_SUM_MAX_CANDIDATE = 250_000.0


def _guess_col(columns, patterns):
    """Return the first column that matches any pattern, or None."""
    for pat in patterns:
        for col in columns:
            if re.search(pat, col, re.IGNORECASE):
                return col
    return None


def _select_sample_rows(
    df,
    sample_id_col=None,
    sample_id_value=None,
    gene_name_col=None,
    gene_id_col=None,
    verbose=True,
):
    if sample_id_col is None and sample_id_value is None:
        return df
    if sample_id_col is None and sample_id_value is not None:
        sample_col = str(sample_id_value)
        if sample_col not in df.columns:
            available = sorted(str(col) for col in df.columns)
            if len(available) > 8:
                available_preview = available[:8] + ["..."]
            else:
                available_preview = available
            raise ValueError(
                f"Sample value {sample_id_value!r} was provided without "
                f"sample_id_col, but no matching expression column was found; "
                f"available columns include: {available_preview}"
            )
        explicit_gene_cols = {
            col
            for col in (gene_name_col, gene_id_col)
            if col is not None and str(col).strip()
        }
        keep = [
            col
            for col in df.columns
            if col == sample_col
            or col in explicit_gene_cols
            or re.search(
                r"(^|[_\s])gene([_\s]?id|[_\s]?name)?$", str(col), re.IGNORECASE
            )
            or str(col).lower() in {"gene", "symbol", "ensembl_gene_id"}
        ]
        out = df.loc[:, keep].copy()
        out = out.rename(columns={sample_col: "TPM"})
        if verbose:
            print(f"[load] Selected wide expression column {sample_col!r} as TPM")
        return out
    if not sample_id_col or sample_id_value is None:
        raise ValueError(
            "Both sample_id_col and sample_id_value must be provided to select a sample"
        )
    if sample_id_col not in df.columns:
        raise ValueError(
            f"Sample column '{sample_id_col}' not found, available columns: {sorted(df.columns)}"
        )

    mask = df[sample_id_col].astype(str) == str(sample_id_value)
    if not mask.any():
        available = sorted(df[sample_id_col].dropna().astype(str).unique().tolist())
        if len(available) > 8:
            available_preview = available[:8] + ["..."]
        else:
            available_preview = available
        raise ValueError(
            f"No rows matched {sample_id_col}={sample_id_value!r}; "
            f"available values include: {available_preview}"
        )

    out = df.loc[mask].copy()
    if verbose:
        print(
            f"[load] Selected {len(out)} rows where {sample_id_col}={sample_id_value!r}"
        )
    return out


def get_canonical_gene_name_from_gene_ids_string(gene_ids_string):
    if pd.isna(gene_ids_string):
        return ""
    gene_ids = str(gene_ids_string).split(";")
    gene_names = [find_gene_name_from_ensembl_gene_id(gene_id) for gene_id in gene_ids]
    not_none_gene_names = [name for name in gene_names if name is not None]
    return ";".join(not_none_gene_names)


_MIN_RECOMMENDED_ENSEMBL_RELEASE = 110
_ensembl_release_check_done = False


def _check_installed_ensembl_releases():
    """Warn once if no Ensembl release >= 110 is installed.

    The alt-haplotype alias mapping canonicalises to primary-contig IDs
    that exist on the primary assembly.  Many of these IDs were added or
    updated in Ensembl 110+, so older installed releases may fail to
    resolve them to gene symbols downstream, leading to missing names in
    reports and plots.
    """
    global _ensembl_release_check_done
    if _ensembl_release_check_done:
        return
    _ensembl_release_check_done = True
    try:
        from pyensembl.shell import collect_all_installed_ensembl_releases
    except ImportError:
        return
    try:
        installed = [
            g
            for g in collect_all_installed_ensembl_releases()
            if g.species.latin_name == "homo_sapiens"
        ]
    except Exception:
        return
    if not installed:
        warnings.warn(
            "No human Ensembl releases are installed via pyensembl. "
            "Gene symbol resolution will be incomplete. Install with: "
            f"pyensembl install --release {_MIN_RECOMMENDED_ENSEMBL_RELEASE} "
            "--species homo_sapiens",
            UserWarning,
            stacklevel=3,
        )
        return
    latest = max(g.release for g in installed)
    if latest < _MIN_RECOMMENDED_ENSEMBL_RELEASE:
        warnings.warn(
            f"Installed Ensembl release {latest} is older than the "
            f"recommended minimum ({_MIN_RECOMMENDED_ENSEMBL_RELEASE}). "
            "Recently-added alt-haplotype MHC/KIR genes may fail to "
            f"resolve to symbols. Install with: pyensembl install "
            f"--release {_MIN_RECOMMENDED_ENSEMBL_RELEASE} --species homo_sapiens",
            UserWarning,
            stacklevel=3,
        )


def _load_ensembl_id_aliases():
    """Return {alt_haplotype_id: primary_contig_id} mapping from bundled data.

    The aliases cover alt-haplotype genes (MHC, KIR, olfactory receptors on
    HSCHR6_MHC_*/HSCHR19KIR_* contigs) that have a primary-chromosome
    equivalent, plus a few retired IDs with documented successors.  Alt-
    haplotype IDs represent alleles of the same gene as the primary-contig
    ID, so their expression should be summed when both are present.

    Chains (A→B→C) are resolved transitively: if the bundled data contains
    both entries, callers get A→C directly.  Cycles and over-long chains
    are detected and raise (indicates bad input data).
    """
    from .plot_data_helpers import _strip_ensembl_version

    try:
        from pirlygenes.load_dataset import get_data

        df = get_data("ensembl-id-aliases")
    except Exception:
        return {}
    raw = dict(
        zip(
            df["alt_haplotype_id"].astype(str).map(_strip_ensembl_version),
            df["primary_contig_id"].astype(str).map(_strip_ensembl_version),
        )
    )
    # Transitively resolve chains so the returned dict has no A→B where B
    # is itself a key.  Cap iterations to catch cycles.
    resolved = {}
    for src in raw:
        dst = raw[src]
        seen = {src}
        for _ in range(16):
            if dst not in raw:
                break
            if dst in seen:
                raise ValueError(
                    f"Cycle in ensembl-id-aliases starting from {src} (loop at {dst})"
                )
            seen.add(dst)
            dst = raw[dst]
        else:
            raise ValueError(f"Chain from {src} exceeded 16 hops in ensembl-id-aliases")
        resolved[src] = dst
    return resolved


def _detect_and_convert_to_tpm(df, verbose=True):
    """Convert FPKM column to TPM if the input looks like FPKM.

    TPM_i = 1e6 * FPKM_i / sum_g(FPKM_g).  This is a within-sample
    rescaling that makes values sum to 1M; it does not correct gene-length
    bias or protocol differences, but it harmonizes the units so
    downstream comparisons (e.g. sample vs. TCGA cohort) are on the same
    scale.

    Only raw FPKM columns are converted (full-name match against
    _RAW_FPKM_PATTERNS).  Derived columns like ``log2_fpkm``,
    ``FPKM_zscore``, or legacy per-cohort columns like ``FPKM_COAD``
    are deliberately NOT treated as raw FPKM — those contain values that
    have already been transformed or belong to a different use case.

    Emits a UserWarning when conversion happens, since this modifies the
    values in the input DataFrame.  The warning surfaces even when
    ``verbose=False`` so callers always see that a conversion occurred.
    """
    # Detect raw FPKM column by strict pattern; derived columns are ignored
    fpkm_col = _guess_col(df.columns, _RAW_FPKM_PATTERNS)
    tpm_col = next(
        (c for c in df.columns if c.upper() == "TPM" or c.lower().endswith("_tpm")),
        None,
    )
    if fpkm_col and not tpm_col:
        total = df[fpkm_col].astype(float).sum()
        if total > 0:
            df = df.copy()
            df[fpkm_col] = 1e6 * df[fpkm_col].astype(float) / total
            df = df.rename(columns={fpkm_col: "TPM"})
            warnings.warn(
                f"FPKM column '{fpkm_col}' was converted to TPM in-place "
                f"(original sum={total:.0f}; new sum=1e6). This is a "
                "within-sample rescaling; gene-length bias is not corrected.",
                UserWarning,
                stacklevel=2,
            )
            if verbose:
                print(
                    f"[load] Converted FPKM column '{fpkm_col}' to TPM (sum was {total:.0f})"
                )
    return df


def _detect_kallisto_gene_abundance_tpm(df, verbose=True):
    """Normalize kallisto/BostonGene gene_abundance.tsv TPM naming.

    Some kallisto-derived gene-level tables call the TPM-like value
    ``abundance`` and carry ``counts`` + ``length`` beside it. Treat that
    exact shape as a TPM input, while avoiding broad "abundance" guesses in
    unrelated tables.
    """
    if "TPM" in set(df.columns) or "abundance" not in set(df.columns):
        return df
    lower_cols = {str(col).lower() for col in df.columns}
    if not {"counts", "length"}.issubset(lower_cols):
        return df
    if verbose:
        print("[load] Auto-detected kallisto gene-level 'abundance' column as TPM")
    return df.rename(columns={"abundance": "TPM"})


def _expression_scale_qc(df) -> dict:
    if "TPM" not in set(df.columns):
        return {
            "sum_tpm": 0.0,
            "max_tpm": 0.0,
            "genes_above_10_tpm": 0,
            "warnings": [],
        }
    tpm = pd.to_numeric(df["TPM"], errors="coerce").fillna(0.0)
    qc = {
        "sum_tpm": float(tpm.sum()),
        "max_tpm": float(tpm.max()) if len(tpm) else 0.0,
        "genes_above_10_tpm": int((tpm > 10.0).sum()),
        "warnings": [],
    }
    total = qc["sum_tpm"]
    if total > 0 and len(tpm):
        sorted_tpm = tpm.sort_values(ascending=False)
        qc["top_gene_share_of_total_tpm"] = float(sorted_tpm.iloc[0] / total)
        qc["top_10_share_of_total_tpm"] = float(sorted_tpm.head(10).sum() / total)
        qc["top_50_share_of_total_tpm"] = float(sorted_tpm.head(50).sum() / total)
        qc["top_200_share_of_total_tpm"] = float(sorted_tpm.head(200).sum() / total)
        qc["top_2000_share_of_total_tpm"] = float(
            sorted_tpm.head(2000).sum() / total
        )
        label_col = next(
            (
                col
                for col in (
                    "gene_display_name",
                    "canonical_gene_name",
                    "gene",
                    "gene_symbol",
                    "symbol",
                )
                if col in df.columns
            ),
            None,
        )
        if label_col is not None:
            dominant = []
            for idx, value in sorted_tpm.head(10).items():
                label = str(df.loc[idx, label_col] or "").strip()
                if not label or label.lower() == "nan":
                    label = str(idx)
                qc_class = classify_gene_qc(label)
                dominant.append(
                    {
                        "gene": label,
                        "tpm": float(value),
                        "share": float(value / total),
                        "qc_class": qc_class.label,
                        "qc_group": qc_class.group,
                    }
                )
            qc["dominant_genes"] = dominant
        if label_col is not None:
            qc.update(
                summarize_qc_class_shares(
                    (
                        (str(df.loc[idx, label_col] or "").strip(), float(value))
                        for idx, value in tpm.items()
                    )
                )
            )
    if "gene" in df.columns:
        try:
            from pirlygenes.gene_sets_cancer import housekeeping_gene_names

            symbols = df["gene"].astype(str).str.upper()
            hk = {gene.upper() for gene in housekeeping_gene_names(core_only=True)}
            hk_tpm = tpm[symbols.isin(hk)]
            if len(hk_tpm):
                qc["housekeeping_median_tpm"] = float(hk_tpm.median())
                qc["housekeeping_genes_observed"] = int((hk_tpm > 1.0).sum())
        except Exception:
            pass
    if total and not (_TPM_SUM_MIN_REASONABLE <= total <= _TPM_SUM_MAX_REASONABLE):
        qc["warnings"].append(
            f"TPM-like values sum to {total:.0f}, outside the expected ~1,000,000 range"
        )
    if qc["max_tpm"] < 100.0 and len(tpm) >= 1000:
        qc["warnings"].append(
            f"maximum TPM-like value is only {qc['max_tpm']:.1f}; check whether input is log-transformed"
        )
    hk_median = qc.get("housekeeping_median_tpm")
    if hk_median is not None and hk_median < 5.0:
        qc["warnings"].append(
            f"core housekeeping median is only {hk_median:.1f} TPM"
        )
    top_gene_share = qc.get("top_gene_share_of_total_tpm")
    top_10_share = qc.get("top_10_share_of_total_tpm")
    top_50_share = qc.get("top_50_share_of_total_tpm")
    if top_gene_share is not None and top_gene_share >= 0.20:
        dominant = qc.get("dominant_genes") or []
        gene = dominant_class_phrase(dominant)
        gene = f" ({gene})" if gene else ""
        qc["warnings"].append(
            f"one gene{gene} accounts for {top_gene_share:.0%} of total TPM; "
            "expression is dominated by a tiny transcript set"
        )
    if top_10_share is not None and top_10_share >= 0.60:
        qc["warnings"].append(
            f"top 10 genes account for {top_10_share:.0%} of total TPM; "
            "check for rRNA/pseudogene/contaminant dominance or severe library complexity loss"
        )
    elif top_50_share is not None and top_50_share >= 0.75:
        qc["warnings"].append(
            f"top 50 genes account for {top_50_share:.0%} of total TPM; "
            "expression distribution is unusually concentrated"
        )
    return qc


def apply_expression_qc_rescue(
    df,
    mode: str = "auto",
    *,
    remove_noncoding: bool = False,
):
    """Remove mtDNA/rRNA-like TPM artifacts and re-normalize expression.

    Parameters
    ----------
    df : pandas.DataFrame
        Gene-level TPM table.
    mode : {"auto", "always"}
        ``auto`` applies the same technical-RNA normalization used for
        references whenever removable features are present, but marks the row as
        high-burden only when mtDNA/rRNA-like features dominate raw TPM.
        ``always`` forces the same normalization. Dirty TPM is not a supported
        analysis mode; raw TPM remains available through the pre-normalization
        QC record.
    remove_noncoding : bool
        If True, also removes noncoding-biotype rows when a biotype column is
        present, while retaining protein-coding, immunoglobulin, and TCR genes.

    Returns
    -------
    (DataFrame, dict)
        The original or rescued frame plus a structured decision record.
    """

    mode_norm = str(mode or "auto").strip().lower().replace("-", "_")
    if mode_norm in {"true", "1", "yes", "on", "force", "forced"}:
        mode_norm = "always"
    elif mode_norm not in {"auto", "always"}:
        raise ValueError(
            "--expression-qc-rescue must be one of auto or always"
        )

    if "TPM" not in set(df.columns):
        return df, {
            "mode": mode_norm,
            "applied": False,
            "reason": "not applied: no TPM column",
            "input_sum_tpm": 0.0,
            "output_sum_tpm": 0.0,
            "removed_tpm": 0.0,
            "removed_fraction": 0.0,
            "removed_gene_count": 0,
            "top_removed_genes": [],
            "remove_noncoding": bool(remove_noncoding),
            "warnings": [],
        }
    tpm = pd.to_numeric(df["TPM"], errors="coerce").fillna(0.0)
    raw_sum = float(tpm.sum())
    label_col = next(
        (
            col
            for col in (
                "gene_display_name",
                "canonical_gene_name",
                "gene",
                "gene_symbol",
                "symbol",
            )
            if col in df.columns
        ),
        None,
    )
    id_col = next(
        (
            col
            for col in (
                "canonical_gene_id",
                "ensembl_gene_id",
                "gene_id",
                "Ensembl_Gene_ID",
            )
            if col in df.columns
        ),
        None,
    )
    record = {
        "mode": mode_norm,
        "applied": False,
        "high_burden": False,
        "input_sum_tpm": raw_sum,
        "output_sum_tpm": raw_sum,
        "removed_tpm": 0.0,
        "removed_fraction": 0.0,
        "removed_gene_count": 0,
        "top_removed_genes": [],
        "remove_noncoding": bool(remove_noncoding),
        "reason": "",
        "warnings": [],
    }
    if (label_col is None and id_col is None) or raw_sum <= 0:
        if label_col is None and id_col is None:
            record["reason"] = "not applied: no gene label or Ensembl ID column"
        return df, record

    labels = (
        df[label_col].fillna("").astype(str).str.strip()
        if label_col is not None
        else pd.Series([""] * len(df), index=df.index)
    )
    ids = (
        df[id_col].fillna("").astype(str).str.strip()
        if id_col is not None
        else pd.Series([""] * len(df), index=df.index)
    )
    removable = technical_rna_mask(df, label_col=label_col, id_col=id_col)
    removed_tpm = float(tpm[removable].sum())
    remaining_tpm = raw_sum - removed_tpm
    removed_fraction = removed_tpm / raw_sum if raw_sum > 0 else 0.0
    record.update(
        {
            "removed_tpm": removed_tpm,
            "removed_fraction": removed_fraction,
            "removed_gene_count": int(removable.sum()),
        }
    )

    removed_rows = []
    for idx, value in tpm[removable].sort_values(ascending=False).head(12).items():
        gene = str(labels.loc[idx] or "").strip()
        ensg = str(ids.loc[idx] or "").strip()
        qc = classify_gene_qc(gene, ensembl_id=ensg)
        removed_rows.append(
            {
                "gene": gene or ensg,
                "tpm": float(value),
                "share": float(value / raw_sum) if raw_sum > 0 else 0.0,
                "qc_class": qc.label,
                "qc_group": qc.group,
            }
        )
    record["top_removed_genes"] = removed_rows

    class_summary = summarize_qc_class_shares(
        (str(label), float(value)) for label, value in zip(labels, tpm)
    )
    record["qc_class_shares"] = class_summary

    scale_qc = _expression_scale_qc(df)
    top_gene_share = float(scale_qc.get("top_gene_share_of_total_tpm") or 0.0)
    top_10_share = float(scale_qc.get("top_10_share_of_total_tpm") or 0.0)
    dominant = scale_qc.get("dominant_genes") or []
    top_gene_rescue_class = bool(
        dominant and is_rescue_feature(dominant[0].get("gene"))
    )
    auto_trigger = bool(
        removed_fraction >= 0.50
        or (top_gene_share >= 0.20 and top_gene_rescue_class)
        or (
            top_10_share >= 0.60
            and (
                float(class_summary.get("rrna_plus_mt_fraction") or 0.0) >= 0.35
            )
        )
    )
    record["high_burden"] = bool(auto_trigger)
    if mode_norm == "always":
        should_apply = raw_sum > 0 and (removed_tpm > 0 or remove_noncoding)
        record["reason"] = "forced by --expression-qc-rescue"
    else:
        should_apply = raw_sum > 0 and (removed_tpm > 0 or remove_noncoding)
        record["reason"] = (
            "auto-rescue: mtDNA/rRNA-like features dominate raw TPM"
            if auto_trigger and should_apply
            else (
                "auto-normalize: expression artifacts zeroed for comparability"
                if should_apply
                else "not applied: no removable expression-artifact burden"
            )
        )

    if not should_apply:
        return df, record

    norm_input = df
    norm_label_col = label_col
    if norm_label_col is None:
        norm_label_col = "__trufflepig_qc_label"
        norm_input = df.copy()
        norm_input[norm_label_col] = ""
    out, normalization_record = normalize_expression(
        norm_input,
        label_col=norm_label_col,
        id_col=id_col,
        value_cols=["TPM"],
        remove_noncoding=remove_noncoding,
    )
    if not normalization_record.get("applied"):
        record["reason"] = normalization_record.get("reason") or record["reason"]
        return df, record
    out = out.copy()
    if norm_label_col == "__trufflepig_qc_label" and norm_label_col in out.columns:
        out = out.drop(columns=[norm_label_col])
    out["TPM_raw_before_qc_rescue"] = tpm
    tpm_record = (normalization_record.get("columns") or {}).get("TPM") or {}
    removed_tpm = float(tpm_record.get("removed_tpm") or removed_tpm)
    removed_fraction = float(tpm_record.get("removed_fraction") or removed_fraction)
    remaining_tpm = raw_sum - removed_tpm
    scale = raw_sum / remaining_tpm if remaining_tpm > 0 else 1.0
    final_qc = _expression_scale_qc(out)
    final_qc["rescued_from_qc_artifact"] = True
    assert_clean_tpm(
        out,
        value_cols=["TPM"],
        label_col=label_col,
        id_col=id_col,
        context="sample expression after technical-RNA normalization",
    )
    out.attrs = dict(getattr(df, "attrs", {}))
    out.attrs["raw_expression_scale_qc"] = scale_qc
    out.attrs["expression_qc_rescue"] = record | {
        "applied": True,
        "output_sum_tpm": float(pd.to_numeric(out["TPM"], errors="coerce").sum()),
        "removed_tpm": removed_tpm,
        "removed_fraction": removed_fraction,
        "removed_gene_count": int(tpm_record.get("removed_gene_count") or 0),
        "removed_technical_gene_count": int(
            tpm_record.get("removed_technical_gene_count") or 0
        ),
        "removed_noncoding_gene_count": int(
            tpm_record.get("removed_noncoding_gene_count") or 0
        ),
        "renormalization_factor": float(scale),
        "reference_normalization_record": normalization_record,
    }
    out.attrs["expression_scale_qc"] = final_qc
    return out, out.attrs["expression_qc_rescue"]


def _detect_and_convert_log_tpm(df, verbose=True):
    if "TPM" not in set(df.columns):
        return df
    tpm = pd.to_numeric(df["TPM"], errors="coerce")
    values = tpm.dropna()
    if len(values) < 1000:
        df.attrs["expression_scale_qc"] = _expression_scale_qc(df)
        return df
    original_sum = float(values.sum())
    original_max = float(values.max())
    if (
        original_sum < _LOG2_TPM_SUM_MAX_CANDIDATE
        and original_max <= _LOG2_TPM_MAX_CANDIDATE
    ):
        converted = (2.0**values - 1.0).clip(lower=0.0)
        converted_sum = float(converted.sum())
        if _TPM_SUM_MIN_REASONABLE <= converted_sum <= _TPM_SUM_MAX_REASONABLE:
            out = df.copy()
            out["TPM"] = (2.0**tpm.fillna(0.0) - 1.0).clip(lower=0.0)
            qc = _expression_scale_qc(out)
            qc.update(
                {
                    "converted_from": "log2_tpm_plus_one",
                    "input_sum_tpm_like": original_sum,
                    "input_max_tpm_like": original_max,
                    "post_conversion_sum_tpm": converted_sum,
                }
            )
            qc["warnings"].append(
                "input values resembled log2(TPM+1) and were converted to linear TPM"
            )
            out.attrs["expression_scale_qc"] = qc
            warnings.warn(
                "TPM column values resembled log2(TPM+1); converted to linear TPM "
                f"(sum {original_sum:.0f} -> {converted_sum:.0f}).",
                UserWarning,
                stacklevel=2,
            )
            if verbose:
                print(
                    "[load] Converted log2(TPM+1)-like values to linear TPM "
                    f"(sum {original_sum:.0f} -> {converted_sum:.0f})"
                )
            return out
    df.attrs["expression_scale_qc"] = _expression_scale_qc(df)
    return df


def _normalize_bostongene_gene_columns(df, verbose=True):
    """Normalize BostonGene/kallisto gene-level column naming.

    Their ``gene_abundance.tsv`` has ``gene_name`` = symbol and ``gene`` =
    Ensembl gene ID. The generic aliases otherwise treat ``gene`` as the
    symbol and lose the Ensembl IDs.
    """
    if not {"gene_name", "gene"}.issubset(set(df.columns)):
        return df
    if "ensembl_gene_id" in set(df.columns):
        return df
    gene_values = df["gene"].dropna().astype(str)
    if gene_values.empty:
        return df
    ensg_fraction = gene_values.str.startswith("ENSG").mean()
    if ensg_fraction < 0.5:
        return df
    if verbose:
        print("[load] Auto-detected BostonGene gene_name/gene columns")
    return df.rename(columns={"gene": "ensembl_gene_id", "gene_name": "gene"})


def _first_non_empty(series):
    """Return the first non-empty, non-null value in a series; fall back to
    first value.  Used to merge string columns (symbol, display name)
    across rows that share a canonical ID: prefer a populated value over
    an empty one regardless of row order."""
    for val in series:
        if pd.notna(val) and str(val).strip() != "":
            return val
    return series.iloc[0] if len(series) else None


def _raise_if_transcript_aggregation_incomplete(df, input_path):
    """Fail fast when transcript aggregation lost too much nonzero signal.

    Downstream purity / ranking code interprets a missing symbol as 0 TPM
    via ``dict.get(symbol, 0.0)``. If transcript->gene aggregation drops a
    substantial chunk of TPM mass, continuing would produce overconfident
    calls from a materially incomplete expression vector.
    """
    stats = df.attrs.get("transcript_aggregation_stats") or {}
    unknown_fraction = float(stats.get("unknown_fraction") or 0.0)
    if unknown_fraction <= _MAX_UNRESOLVED_TRANSCRIPT_TPM_FRACTION:
        return

    unknown_tpm = float(stats.get("unknown_tpm") or 0.0)
    known_tpm = float(stats.get("known_tpm") or 0.0)
    unresolved_unique = int(stats.get("unresolved_unique_count") or 0)
    preview = stats.get("unresolved_high_tpm") or []
    preview_text = ", ".join(
        f"{row['tx']} ({row['TPM']:.1f} TPM)" for row in preview[:5]
    )
    if preview_text:
        preview_text = f" Top unresolved transcripts: {preview_text}."
    raise ValueError(
        "Transcript-level aggregation left too much signal unresolved for "
        f"{input_path}: {unknown_tpm:.1f} TPM ({unknown_fraction:.1%}) across "
        f"{unresolved_unique} unique transcript IDs remained unmapped "
        f"(known TPM={known_tpm:.1f}). This usually means the sample was "
        "quantified against an older or mismatched Ensembl annotation. "
        "Refusing to continue because downstream purity/classification code "
        "would silently treat those genes as 0 TPM." + preview_text
    )


def _apply_id_aliases_and_sum(df, verbose=True):
    """Map alt-haplotype Ensembl IDs to their primary-contig equivalents
    and sum TPM values (alt-haplotype IDs represent alleles of the same gene).

    Operates by ID only.  Runs AFTER symbol-based consolidation in the
    load pipeline: symbol consolidation handles inputs where alt-haplotype
    rows share a gene symbol with the primary, and this step catches
    alt-haplotype rows that lack a symbol (e.g. a sample exported with
    only Ensembl IDs).

    Sums TPM across rows that collapse onto the same canonical ID
    (alt-haplotype alleles contribute to the same gene's total expression).
    Non-TPM string columns are merged by preferring the first non-empty
    value, so a remapped alt-haplotype row with an empty symbol inherits
    the primary row's symbol.
    """
    if "ensembl_gene_id" not in df.columns:
        return df
    aliases = _load_ensembl_id_aliases()
    if not aliases:
        return df

    from .plot_data_helpers import _strip_ensembl_version

    df = df.copy()
    original_ids = df["ensembl_gene_id"].astype(str).map(_strip_ensembl_version)
    canonical_ids = original_ids.map(lambda gid: aliases.get(gid, gid))
    n_remapped = int((canonical_ids != original_ids).sum())
    if n_remapped == 0:
        return df
    df["ensembl_gene_id"] = canonical_ids

    # Sum TPM across rows that now share the same canonical ID; for other
    # columns, prefer the first non-empty value (avoids inheriting an
    # empty symbol from an alt-haplotype row just because it came first).
    tpm_col = next((c for c in df.columns if c.upper() == "TPM"), None)
    if tpm_col and df["ensembl_gene_id"].duplicated().any():
        agg_map = {tpm_col: "sum"}
        for col in df.columns:
            if col != "ensembl_gene_id" and col != tpm_col:
                agg_map[col] = _first_non_empty
        df = df.groupby("ensembl_gene_id", as_index=False).agg(agg_map)

    if verbose:
        print(
            f"[load] Remapped {n_remapped} alt-haplotype Ensembl IDs to primary contig and summed TPM"
        )
    return df


def _consolidate_gene_ids(df, verbose=True):
    """Collapse alt-haplotype / retired Ensembl IDs to one ID per gene symbol.

    GENCODE transcriptomes include genes on alternate haplotype contigs
    (e.g., B2M has ENSG00000166710 on primary + ENSG00000273686 on alt).
    Salmon distributes reads across both, splitting the expression.

    For each gene symbol with multiple IDs, pick one canonical ID
    (prefer the one in our pan-cancer reference) and SUM TPM across
    all alt-haplotype copies (reads are split by the aligner, so
    summing recovers the true total).
    """
    if "ensembl_gene_id" not in df.columns:
        return df

    from .plot_data_helpers import _strip_ensembl_version
    from trufflepig.reference import pan_cancer_expression
    df = df.copy()
    df["_id_stripped"] = df["ensembl_gene_id"].astype(str).map(_strip_ensembl_version)

    # Find the symbol column (canonical_gene_name or gene)
    sym_col = "canonical_gene_name" if "canonical_gene_name" in df.columns else "gene"
    if sym_col not in df.columns:
        return df

    symbols = df[sym_col].fillna("").astype(str)
    # Only consolidate genes with actual symbols (skip empty/nan)
    has_symbol = symbols.str.strip().ne("") & symbols.str.upper().ne("NAN")

    # Count IDs per symbol — only process multi-ID symbols
    id_counts = df.loc[has_symbol].groupby(sym_col)["_id_stripped"].nunique()
    multi_id_symbols = set(id_counts[id_counts > 1].index)

    if not multi_id_symbols:
        df.drop(columns=["_id_stripped"], inplace=True)
        return df

    # Build set of IDs in our pan-cancer reference for canonical preference
    ref_ids = set(pan_cancer_expression()["Ensembl_Gene_ID"])

    tpm_col = next((c for c in df.columns if c.upper() == "TPM"), None)

    n_consolidated = 0
    rows_to_drop = []
    id_remap = {}  # old_id → canonical_id

    for sym in multi_id_symbols:
        mask = has_symbol & (symbols == sym)
        sub = df[mask]
        ids = sub["_id_stripped"].unique()

        # Pick canonical: prefer ID in reference, then highest TPM
        in_ref = [i for i in ids if i in ref_ids]
        if in_ref:
            canonical = in_ref[0]
        elif tpm_col:
            canonical = sub.loc[sub[tpm_col].astype(float).idxmax(), "_id_stripped"]
        else:
            canonical = ids[0]

        # Map all alt IDs to canonical
        for alt_id in ids:
            if alt_id != canonical:
                id_remap[alt_id] = canonical

        # Sum TPM across alt-haplotype copies (reads are split by aligner)
        # and keep one row with the summed value
        if tpm_col and len(sub) > 1:
            total_tpm = sub[tpm_col].astype(float).sum()
            best_idx = sub[tpm_col].astype(float).idxmax()
            drop_idx = sub.index.difference([best_idx])
            rows_to_drop.extend(drop_idx)
            df.loc[best_idx, tpm_col] = total_tpm
            df.loc[best_idx, "_id_stripped"] = canonical
            n_consolidated += 1

    if rows_to_drop:
        df.drop(index=rows_to_drop, inplace=True)

    # Apply ID remapping
    df["ensembl_gene_id"] = df["_id_stripped"]
    df.drop(columns=["_id_stripped"], inplace=True)

    if verbose and n_consolidated:
        print(
            f"[load] Consolidated {n_consolidated} genes with multiple "
            f"Ensembl IDs (alt haplotypes / retired IDs)"
        )
    return df


def _attach_gene_sidecar_if_present(input_path, df, verbose=True):
    """Hydrate one-column TPM vectors using a sibling Gene.csv sidecar.

    Some sample bundles store gene names in ``Gene.csv`` and one or more
    parallel TPM vectors in separate one-column CSVs. When such a vector is
    loaded directly, attach the gene-name sidecar so the rest of the loader can
    treat it as ordinary gene-level expression data.
    """
    if list(df.columns) != ["TPM"]:
        return df, False

    sidecar = Path(input_path).with_name("Gene.csv")
    if not sidecar.exists():
        return df, False

    gene_df = pd.read_csv(sidecar)
    if len(gene_df.columns) != 1 or len(gene_df) != len(df):
        return df, False

    out = pd.DataFrame(
        {
            "gene": gene_df.iloc[:, 0].astype(str),
            "TPM": pd.to_numeric(df["TPM"], errors="coerce").fillna(0.0),
        }
    )
    if verbose:
        print(f"[load] Attached gene names from sidecar: {sidecar}")
    return out, True


def _resolve_unknown_transcripts_for_raw_frame(df, verbose=False, progress=True):
    """Build the ``transcript_id → gene_symbol`` map for a raw
    transcript-level frame in one pass (#81).

    Returns a dict that includes (a) the static ``extra_tx_mappings``
    hits, (b) their versionless equivalents, and (c) the pyensembl
    resolution for the remaining unknowns. Downstream consumers
    (``_build_transcript_expression_frame``, ``tx2gene``) read from
    this map instead of resolving the same IDs again. The map
    preserves full presence/absence semantics — zero-TPM transcripts
    are resolved too, so a gene whose every transcript is zero still
    makes it into the aggregated gene-level output with TPM=0
    (important for FFPE-panel and other present-vs-absent checks).
    """
    from .aggregate_gene_expression import _expanded_tx_map
    from .transcript_to_gene import extra_tx_mappings

    tx_col = None
    for candidate in (
        "ensembl_transcript_id",
        "transcript_id",
        "Name",
        "name",
        "transcript",
        "target",
        "target_id",
    ):
        if candidate in df.columns:
            tx_col = candidate
            break
    if tx_col is None:
        return {}

    tx_raw = df[tx_col].astype(str)
    tx0 = tx_raw.str.split(".", n=1).str[0]
    static_map = _expanded_tx_map(extra_tx_mappings or {})

    # pandas.Series.map with a dict: missing keys yield NaN.
    gene_series = tx0.map(static_map)
    # #122: count how many unique transcripts were resolved via the
    # auxiliary CSV vs pyensembl, so a curator can see the aux map
    # actually getting used (otherwise it silently goes stale).
    aux_used = set(tx0[~gene_series.isna()].dropna().unique()) & set(static_map.keys())
    unknown_mask = gene_series.isna()
    resolved = {k: v for k, v in static_map.items() if v is not None}
    pyensembl_used = set()
    if unknown_mask.any():
        unknown_unique = pd.Index(tx0[unknown_mask].dropna().unique())
        pyensembl_hits = _resolve_unknown_transcripts_to_genes(
            unknown_unique, verbose=verbose, progress=progress
        )
        # Drop unresolved (None-valued) entries from the shared map
        # so downstream `gene_series = tx0.map(map)` treats them as
        # unknown (which is correct) rather than mapping them to
        # Python None (which would quietly coerce to NaN but clutter
        # the map for debuggers).
        for k, v in pyensembl_hits.items():
            if v is not None:
                resolved[k] = v
                pyensembl_used.add(k)
    if verbose and (aux_used or pyensembl_used):
        print(
            f"[load] Transcript resolution: "
            f"{len(aux_used)} via extra-tx-mappings.csv (#122), "
            f"{len(pyensembl_used)} via pyensembl cascade"
        )
    return resolved


def _resolve_unknown_transcripts_to_genes(tx0_unique, verbose=False, progress=True):
    """One authoritative pyensembl pass over unique versionless
    transcript IDs. Returns ``dict[str, Optional[str]]`` — a ``None``
    value means pyensembl couldn't resolve it across any installed
    release. Shared by ``_build_transcript_expression_frame`` and
    ``tx2gene`` so the load path resolves the transcript set exactly
    once per file (#81).
    """
    from pirlygenes.gene_ids import find_gene_name_from_ensembl_transcript_id, _build_indexes

    # Build the Ensembl index *before* the progress bar starts so the
    # user doesn't see a 0% bar stuck while the index loads.
    _build_indexes()
    if verbose:
        print(f"[load] Resolving {len(tx0_unique)} unique transcripts via Ensembl")
    iterator = tqdm(
        tx0_unique,
        desc="Resolving transcript IDs",
        disable=not progress,
    )
    return {
        t: find_gene_name_from_ensembl_transcript_id(t, verbose=False) for t in iterator
    }


def _build_transcript_expression_frame(
    df,
    verbose=False,
    progress=True,
    tx_to_gene_name=None,
):
    """Return a normalized transcript-level frame extracted from ``df``.

    The input must look like a salmon ``quant.sf`` (Name + Length + TPM)
    or a previously-rebuilt rich transcript table
    (ensembl_transcript_id + length + TPM). Returns columns:

    - ``transcript_id``
    - ``length`` (transcript length in nt; needed for length-bias)
    - ``TPM``
    - ``ensembl_gene_id`` (when available)
    - ``gene_symbol`` (when available)

    Returns ``None`` if no usable transcript columns are found — the
    caller treats that as "transcript-level signals unavailable".
    """
    cols = {c.lower(): c for c in df.columns}
    tx_col = next(
        (
            cols[c]
            for c in (
                "ensembl_transcript_id",
                "transcript_id",
                "name",
                "transcript",
                "target",
                "target_id",
                "transcriptid",
                "targetid",
            )
            if c in cols
        ),
        None,
    )
    len_col = next(
        (cols[c] for c in ("length", "transcript_length") if c in cols),
        None,
    )
    tpm_col = next(
        (cols[c] for c in ("tpm",) if c in cols),
        None,
    )
    if tx_col is None or len_col is None or tpm_col is None:
        return None

    out = pd.DataFrame(
        {
            "transcript_id": df[tx_col].astype(str),
            "length": pd.to_numeric(df[len_col], errors="coerce"),
            "TPM": pd.to_numeric(df[tpm_col], errors="coerce").fillna(0.0),
        }
    )
    for src, dst in (
        ("ensembl_gene_id", "ensembl_gene_id"),
        ("gene_id", "ensembl_gene_id"),
        ("gene_symbol", "gene_symbol"),
        ("gene", "gene_symbol"),
        ("symbol", "gene_symbol"),
    ):
        if src in cols and dst not in out.columns:
            out[dst] = df[cols[src]].astype(str)
    out = out.dropna(subset=["length"])

    # Resolve transcript → gene when the upstream frame doesn't carry
    # gene identifiers (raw salmon quant.sf has Name/Length/TPM only).
    # Downstream signals (isoform length bias) need a gene grouping
    # key; without this they silently disable on the most common input.
    #
    # The caller can pass a precomputed ``tx_to_gene_name`` map to
    # skip the pyensembl pass entirely (#81) — this is how
    # ``load_expression_data`` shares one resolution across both the
    # transcript-level frame and ``tx2gene``.
    if "ensembl_gene_id" not in out.columns and "gene_symbol" not in out.columns:
        from .aggregate_gene_expression import _expanded_tx_map
        from .transcript_to_gene import extra_tx_mappings

        # Resolve every transcript, not just expressed ones. This keeps
        # the frame a faithful representation of the input quant so
        # future presence/absence signals on isoforms aren't forced to
        # distinguish "isoform absent from quant" from "isoform present
        # at TPM=0". The only current consumer
        # (``compute_isoform_length_bias``) applies its own ``TPM > 0``
        # filter internally — that's the correct layer for it.
        tx0 = out["transcript_id"].astype(str).str.split(".", n=1).str[0]
        static_map = _expanded_tx_map(extra_tx_mappings or {})
        gene_syms = tx0.map(static_map)
        unresolved = gene_syms.isna()
        if unresolved.any():
            uniq = pd.Index(tx0[unresolved].unique())
            if tx_to_gene_name is not None:
                resolved = {t: tx_to_gene_name.get(t) for t in uniq}
            else:
                resolved = _resolve_unknown_transcripts_to_genes(
                    uniq, verbose=verbose, progress=progress
                )
            gene_syms.loc[unresolved] = tx0[unresolved].map(resolved)
        out["gene_symbol"] = gene_syms.astype(object)
        out = out[out["gene_symbol"].notna()].copy()

    if verbose:
        print(
            f"[load] Retained {len(out)} transcript-level rows for downstream signals"
            + (
                " (no gene-id column; gene_symbol used for grouping)"
                if "ensembl_gene_id" not in out.columns
                else ""
            )
        )
    return out


def _try_load_sibling_transcript_frame(input_path, verbose=False, progress=True):
    """When the user passed a gene-level table, look for a sibling
    transcript-level file in the same directory (or common subdirs)
    and load it.

    Recognises standard quantifier outputs:
    - ``quant.sf`` — salmon (same dir, or one level up)
    - ``abundance.tsv`` — kallisto
    - ``transcript_expression_salmon.tsv`` — rich format with gene +
      length + TPM (legacy / custom pipelines)

    Returns the normalized transcript frame or ``None`` if no sibling
    found. The caller can bypass this search entirely by passing
    ``--transcripts`` to the CLI.
    """
    try:
        parent = Path(input_path).resolve().parent
    except (OSError, RuntimeError):
        return None
    candidates = [
        parent / "quant.sf",
        parent / "abundance.tsv",
        parent / "transcript_expression_salmon.tsv",
    ]
    # One level up — common when the gene file lives in a subdirectory
    # alongside the quantifier output (e.g. ``results/gene_tpm.csv``
    # next to ``results/../quant.sf``).
    grandparent = parent.parent
    if grandparent != parent:
        candidates.extend(
            [
                grandparent / "quant.sf",
                grandparent / "abundance.tsv",
            ]
        )
    for c in candidates:
        if c.exists():
            try:
                sep = "\t" if c.suffix in (".sf", ".tsv") else ","
                raw = pd.read_csv(str(c), sep=sep)
                tx = _build_transcript_expression_frame(
                    raw, verbose=verbose, progress=progress
                )
                if tx is not None and not tx.empty:
                    if verbose:
                        print(f"[load] Picked up sibling transcript file: {c}")
                    return tx
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"[load] Sibling transcript file load failed for {c}: {exc}")
    return None


def _validate_expression_input(df, input_path):
    """Fail loud and early on empty / unusable expression input.

    Without this an empty file or a pre-aggregated salmon gene table only
    surfaces much later as a confusing ``AttributeError: canonical_gene_name`` or
    ``Unable to find a column for transcript ID`` deep in the pipeline — see the
    alvin ``gene_tpm.tsv`` (header-only) case. Catch the obvious-garbage inputs
    at the load boundary with an actionable message instead.
    """
    if len(df) == 0:
        raise ValueError(
            f"Expression input {input_path!r} has no data rows — the file is "
            "empty or header-only. Check that the upstream quantification step "
            "actually produced output before re-running."
        )
    cols = {str(c).lower() for c in df.columns}
    # Salmon hallmark columns (NumReads + EffectiveLength) but only a gene id and
    # no transcript Name / gene symbol => a *pre-aggregated salmon gene* file.
    # trufflepig does its own transcript->gene aggregation (with clean-TPM), so it
    # needs the transcript-level quant.sf (a ``Name`` column) or a gene table that
    # carries symbols — not a gene file collapsed upstream by a different rule.
    salmon_like = "numreads" in cols and "effectivelength" in cols
    has_transcript = any(
        c in cols for c in ("name", "transcript_id", "transcript", "target_id")
    )
    has_symbol = any(
        c in cols
        for c in (
            "gene", "gene_name", "gene_symbol", "symbol",
            "canonical_gene_name", "gene_display_name",
        )
    )
    has_gene_id = any(c in cols for c in ("gene_id", "ensembl_gene_id"))
    if salmon_like and has_gene_id and not has_transcript and not has_symbol:
        raise ValueError(
            f"Expression input {input_path!r} looks like a pre-aggregated salmon "
            "gene-level file (gene_id + NumReads + EffectiveLength, no transcript "
            "IDs or gene symbols). trufflepig aggregates transcripts to genes "
            "itself, so provide the transcript-level quant.sf (a 'Name' column) "
            "or a gene table that includes a gene symbol column."
        )
    # A whole-transcriptome quant carries ~20k+ genes; a few hundred rows means a
    # truncated file or a targeted panel that trufflepig can't reference-compare.
    # Warn (not error) so intentionally-tiny unit fixtures still load.
    _MIN_EXPECTED_FEATURES = 1000
    if len(df) < _MIN_EXPECTED_FEATURES:
        warnings.warn(
            f"Expression input {input_path!r} has only {len(df)} rows — far below "
            f"the ~20k+ genes expected from whole-transcriptome RNA-seq. The file "
            "may be truncated or a targeted panel; reference-based calls will be "
            "unreliable.",
            stacklevel=2,
        )


def load_expression_data(
    input_path,
    aggregate_gene_expression=False,
    save_aggregated_gene_expression=True,
    aggregated_output_path=None,
    gene_name_col=None,
    gene_id_col=None,
    sample_id_col=None,
    sample_id_value=None,
    transcript_path=None,
    verbose=True,
    progress=True,
):
    _check_installed_ensembl_releases()
    if verbose:
        print(f"[load] Loading expression data from: {input_path}")

    if ".csv" in input_path:
        df = pd.read_csv(input_path)
    elif ".tsv" in input_path or ".sf" in input_path or ".txt" in input_path:
        df = pd.read_csv(input_path, sep="\t")
    elif ".xlsx" in input_path:
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unrecognized file format for {input_path}")
    if verbose:
        print(f"[load] Loaded {len(df)} rows and {len(df.columns)} columns")

    # Fail loud and early on empty / pre-aggregated-salmon-gene input rather than
    # letting it surface as a confusing error deep in the pipeline.
    _validate_expression_input(df, input_path)

    df, used_sidecar = _attach_gene_sidecar_if_present(input_path, df, verbose=verbose)
    df = _select_sample_rows(
        df,
        sample_id_col=sample_id_col,
        sample_id_value=sample_id_value,
        gene_name_col=gene_name_col,
        gene_id_col=gene_id_col,
        verbose=verbose,
    )
    df = _normalize_bostongene_gene_columns(df, verbose=verbose)
    # FPKM→TPM: convert BEFORE any aggregation or ID remapping so downstream
    # consolidation (which sums values) operates on consistent units.
    df = _detect_and_convert_to_tpm(df, verbose=verbose)
    df = _detect_kallisto_gene_abundance_tpm(df, verbose=verbose)
    if used_sidecar and aggregate_gene_expression:
        if verbose:
            print(
                "[load] Input is already gene-level via Gene.csv sidecar; skipping transcript aggregation"
            )
        aggregate_gene_expression = False

    # Preserve the transcript-level frame (if present in the input) so
    # downstream steps can use it for capture-bias-immune signals
    # (isoform length bias, APA-3'UTR usage). Computed here *before*
    # aggregation / consolidation because those pandas groupby ops
    # strip ``df.attrs``; the retained frame is re-attached to the
    # final returned DataFrame right before return.
    _retained_transcript_df = None
    if aggregate_gene_expression:
        if verbose:
            print("[load] Aggregating transcript-level TPM values to gene-level TPM")

        # #81: resolve tx → gene once, then hand the shared map to
        # both ``_build_transcript_expression_frame`` and ``tx2gene``.
        # Previously each step did its own pyensembl pass over the
        # same IDs, doubling the wall-clock cost of the load on a
        # ~177k-transcript salmon quant.
        shared_tx_map = _resolve_unknown_transcripts_for_raw_frame(
            df, verbose=verbose, progress=progress
        )
        _retained_transcript_df = _build_transcript_expression_frame(
            df,
            verbose=verbose,
            progress=progress,
            tx_to_gene_name=shared_tx_map,
        )
        df = tx2gene(
            df,
            verbose=verbose,
            progress=progress,
            tx_to_gene_name=shared_tx_map,
        )
        _raise_if_transcript_aggregation_incomplete(df, input_path)

        if save_aggregated_gene_expression:
            if aggregated_output_path:
                output_path = Path(aggregated_output_path)
            else:
                input_file = Path(input_path)
                output_path = input_file.with_name(f"{input_file.stem}.gene_tpm.csv")
            df.to_csv(output_path, index=False)
            if verbose:
                print(f"[load] Saved aggregated gene-level TPM CSV to: {output_path}")

    # Apply explicit overrides first, then fall back to auto-rename
    if gene_name_col and gene_name_col in df.columns and gene_name_col != "gene":
        df = df.rename(columns={gene_name_col: "gene"})
    if gene_id_col and gene_id_col in df.columns and gene_id_col != "ensembl_gene_id":
        df = df.rename(columns={gene_id_col: "ensembl_gene_id"})

    df = df.rename(
        columns={
            "Gene Symbol": "gene",
            "gene_symbol": "gene",
            "Gene": "gene",
            "Gene Name": "gene",
            "Gene_Name": "gene",
            "symbol": "gene",
            "Symbol": "gene",
            "Gene ID": "ensembl_gene_id",
            "Gene_ID": "ensembl_gene_id",
            "Ensembl Gene ID": "ensembl_gene_id",
            "Ensembl_Gene_ID": "ensembl_gene_id",
            "ensembl_gene": "ensembl_gene_id",
            "gene_id": "ensembl_gene_id",
            "canonical_gene_id": "ensembl_gene_id",
            "GeneID": "ensembl_gene_id",
            "tpm": "TPM",
            "gene_tpm": "TPM",
            "gene_tpm_cognizant_corrector": "TPM",
        }
    )

    # Fuzzy fallback: case-insensitive guess from column names
    if "gene" not in set(df.columns):
        col = _guess_col(df.columns, _GENE_NAME_PATTERNS)
        if col:
            if verbose:
                print(f"[load] Auto-detected gene name column: '{col}'")
            df = df.rename(columns={col: "gene"})
    if "ensembl_gene_id" not in set(df.columns):
        col = _guess_col(df.columns, _GENE_ID_PATTERNS)
        if col:
            if verbose:
                print(f"[load] Auto-detected gene ID column: '{col}'")
            df = df.rename(columns={col: "ensembl_gene_id"})
    if "TPM" not in set(df.columns):
        col = _guess_col(df.columns, _TPM_PATTERNS)
        if col:
            if verbose:
                print(f"[load] Auto-detected TPM column: '{col}'")
            df = df.rename(columns={col: "TPM"})

    if "gene" not in set(df.columns) and "ensembl_gene_id" not in set(df.columns):
        raise ValueError(
            f"Gene column not found in {input_path}, available columns: {sorted(set(df.columns))}"
        )
    if "TPM" not in set(df.columns):
        raise ValueError(
            f"TPM column not found in {input_path}, available columns: {sorted(set(df.columns))}"
        )
    if "gene" in set(df.columns):
        df["gene"] = df["gene"].fillna("").astype(str).apply(short_gene_name)

    if "ensembl_gene_id" not in set(df.columns):
        if verbose:
            print("[load] Resolving Ensembl gene IDs from gene symbols")
        gene_ids, canonical_gene_names = find_canonical_gene_ids_and_names(df.gene)
        if not gene_ids:
            raise ValueError(
                f"Unable to find Ensembl gene IDs for any of the genes in {input_path}"
            )
        if len(gene_ids) != len(df):
            raise ValueError(
                f"Number of Ensembl gene IDs ({len(gene_ids)}) does not match number of rows in {input_path} ({len(df)})"
            )

        df["ensembl_gene_id"] = gene_ids

        if not canonical_gene_names:
            raise ValueError(
                f"Unable to find canonical gene names for any of the genes in {input_path}"
            )
        if len(canonical_gene_names) != len(df):
            raise ValueError(
                f"Number of canonical gene names ({len(canonical_gene_names)}) does not match number of rows in {input_path} ({len(df)})"
            )
        if "canonical_gene_name" in set(df.columns):
            raise ValueError(
                f"Column 'canonical_gene_name' already exists in {input_path}, please rename it before loading."
            )
        else:
            df["canonical_gene_name"] = [
                (
                    gs
                    if type(gs) is str
                    else (
                        ""
                        if gs is None
                        else ";".join(gs)
                        if type(gs) in (list, tuple)
                        else "?"
                    )
                )
                for gs in canonical_gene_names
            ]
        if verbose:
            print("[load] Finished resolving Ensembl gene IDs")

    if "canonical_gene_name" not in set(df.columns):
        if (
            "ensembl_gene_id" in set(df.columns)
            and df["ensembl_gene_id"].astype(str).str.strip().ne("").any()
        ):
            if verbose:
                print("[load] Resolving canonical gene names from Ensembl gene IDs")
            # Pre-build the Ensembl indexes before starting tqdm so the
            # progress bar measures actual row resolution work rather
            # than sitting at 0% during the one-time index build.
            from pirlygenes.gene_ids import _build_indexes

            _build_indexes()
            iterator = df["ensembl_gene_id"]
            if progress:
                iterator = tqdm(
                    iterator, total=len(df), desc="Resolving canonical gene names"
                )
            df["canonical_gene_name"] = [
                get_canonical_gene_name_from_gene_ids_string(gene_ids_string)
                for gene_ids_string in iterator
            ]
        elif (
            "gene" in set(df.columns)
            and df["gene"].astype(str).str.strip().ne("").any()
        ):
            if verbose:
                print("[load] Using gene symbols as canonical gene names")
            df["canonical_gene_name"] = df["gene"].fillna("").astype(str)

    if "gene" not in set(df.columns):
        if verbose:
            print("[load] Using canonical gene names as gene symbols")
        df["gene"] = (
            df["canonical_gene_name"].fillna("").astype(str).apply(short_gene_name)
        )
    elif "canonical_gene_name" in set(df.columns):
        missing_gene = df["gene"].astype(str).str.strip().eq("")
        if missing_gene.any():
            df.loc[missing_gene, "gene"] = (
                df.loc[missing_gene, "canonical_gene_name"]
                .fillna("")
                .astype(str)
                .apply(short_gene_name)
            )

    if "gene_display_name" not in set(df.columns):
        if verbose:
            print("[load] Computing display labels for genes")
        iterator = df.canonical_gene_name
        if progress:
            iterator = tqdm(iterator, total=len(df), desc="Formatting display names")
        df["gene_display_name"] = [
            ";".join([display_name(gene_name) for gene_name in gene_names.split(";")])
            for gene_names in iterator
        ]
    df = _detect_and_convert_log_tpm(df, verbose=verbose)
    scale_qc = dict(df.attrs.get("expression_scale_qc") or {})
    # Consolidate alternate-locus / retired Ensembl IDs that map to the
    # same gene symbol.  Pick the canonical ID (the one in our pan-cancer
    # reference, or highest-expressed) and aggregate TPM via max.
    df = _consolidate_gene_ids(df, verbose=verbose)

    # Apply ID-based aliases (alt-haplotype MHC/KIR → primary contig).
    # Runs after symbol consolidation so either approach can consolidate
    # whatever the other missed.  Sums TPM across alt-haplotype alleles.
    df = _apply_id_aliases_and_sum(df, verbose=verbose)
    final_scale_qc = _expression_scale_qc(df)
    if scale_qc.get("converted_from"):
        final_scale_qc.update(
            {
                "converted_from": scale_qc.get("converted_from"),
                "input_sum_tpm_like": scale_qc.get("input_sum_tpm_like"),
                "input_max_tpm_like": scale_qc.get("input_max_tpm_like"),
                "post_conversion_sum_tpm": final_scale_qc.get("sum_tpm"),
            }
        )
        conversion_warning = (
            "input values resembled log2(TPM+1) and were converted to linear TPM"
        )
        final_scale_qc["warnings"] = [
            conversion_warning,
            *[
                warning
                for warning in final_scale_qc.get("warnings", [])
                if warning != conversion_warning
            ],
        ]
    df.attrs["expression_scale_qc"] = final_scale_qc

    # Attach a transcript-level frame on ``df.attrs`` so downstream
    # signals (isoform length bias, APA-3'UTR usage, etc.) can use
    # capture-bias-immune within-gene comparisons.
    #
    # Priority:
    # 1. Frame retained from this file's own transcript-level rows
    #    (available when the input was transcript-level + ``-a``).
    # 2. Explicit ``--transcripts <path>`` provided by the caller.
    # 3. Sibling auto-discovery: look for a standard quantifier
    #    output (salmon quant.sf, kallisto abundance.tsv) alongside
    #    the gene-level input.
    if _retained_transcript_df is not None and not _retained_transcript_df.empty:
        df.attrs["transcript_expression"] = _retained_transcript_df
    elif transcript_path is not None:
        try:
            tp = Path(transcript_path)
            sep = "\t" if tp.suffix in (".sf", ".tsv") else ","
            raw = pd.read_csv(str(tp), sep=sep)
            tx = _build_transcript_expression_frame(
                raw, verbose=verbose, progress=progress
            )
            if tx is not None and not tx.empty:
                df.attrs["transcript_expression"] = tx
                if verbose:
                    print(f"[load] Loaded transcript frame from --transcripts: {tp}")
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[load] Failed to load --transcripts {transcript_path}: {exc}")
    else:
        sibling_tx = _try_load_sibling_transcript_frame(
            input_path, verbose=verbose, progress=progress
        )
        if sibling_tx is not None:
            df.attrs["transcript_expression"] = sibling_tx

    if verbose:
        print(
            f"[load] Expression data ready: {len(df)} rows, columns={list(df.columns)}"
        )
    return df
