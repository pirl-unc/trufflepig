"""Regenerate trufflepig's tumor-up-vs-matched-normal panels.

The inputs are trufflepig-owned deconvolved references plus pirlygenes'
ID-keyed HPA/pan-cancer matrix exposed through ``trufflepig.reference``.
This deliberately does not read deconvolved data from pirlygenes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pirlygenes.gene_sets_cancer import (
    housekeeping_gene_names,
    mitochondrial_gene_names,
)

from trufflepig.reference import (
    pan_cancer_expression,
    subtype_deconvolved_expression,
    tcga_deconvolved_expression,
)


_SOLID_MATCHED_NORMAL_TISSUE = {
    "ACC": "adrenal_gland",
    "ATRT": "cerebellum",
    "BLCA": "urinary_bladder",
    "BRCA": "breast",
    "CESC": "cervix",
    "CHOL": "gallbladder",
    "CHON": "bone_marrow",
    "COAD": "colon",
    "ESCA": "esophagus",
    "EWS": "bone_marrow",
    "GBM": "cerebral_cortex",
    "HEPB": "liver",
    "HNSC": "tongue",
    "KICH": "kidney",
    "KIRC": "kidney",
    "KIRP": "kidney",
    "LGG": "cerebral_cortex",
    "LIHC": "liver",
    "LUAD": "lung",
    "LUSC": "lung",
    "MBL": "cerebellum",
    "MESO": "lung",
    "NBL": "adrenal_gland",
    "OS": "bone_marrow",
    "OV": "ovary",
    "PAAD": "pancreas",
    "PANNET": "pancreas",
    "PCPG": "adrenal_gland",
    "PRAD": "prostate",
    "RB": "retina",
    "READ": "rectum",
    "RMS_ARMS": "skeletal_muscle",
    "RMS_ERMS": "skeletal_muscle",
    "RMS_SSRMS": "skeletal_muscle",
    "RT": "kidney",
    "SCLC": "lung",
    "SKCM": "skin",
    "STAD": "stomach",
    "TGCT": "testis",
    "THCA": "thyroid_gland",
    "THYM": "thymus",
    "UCEC": "endometrium",
    "UCS": "endometrium",
    "UVM": "retina",
    "WILMS": "kidney",
}
_HEME_MATCHED_NORMAL_TISSUE = {
    "DLBC": "lymph_node",
    "LAML": "bone_marrow",
}

_EXCLUDED_SYMBOLS = {
    "A1BG",
    "C19orf81",
    "C4A",
    "CCDC163",
    "CCNI2",
    "DCAF8L1",
    "ECE2",
    "FUNDC2",
    "GCG",
    "H2BC12L",
    "H3Y1",
    "H4C11",
    "HERC3",
    "INS",
    "ITGAE",
    "KCTD2",
    "LGALS13",
    "LYZL6",
    "MKKS",
    "MTCP1",
    "RAMACL",
    "SLC39A4",
    "SMN2",
    "SRRM3",
    "TBL3",
    "TMC4",
    "TMEM151A",
    "TMEM265",
    "WASH6P",
    "ZACN",
    "ZDHHC19",
    "ZFTRAF1",
    "ZNF716",
}
_EXCLUDED_PREFIXES = (
    "RPS",
    "RPL",
    "RPLP",
    "MRPS",
    "MRPL",
    "IGH",
    "IGK",
    "IGL",
    "TRBV",
    "TRAV",
    "TRGV",
    "TRDV",
    "HIST1",
    "HIST2",
    "HIST3",
    "HIST4",
    "OR",
    "TAS2R",
    "TAS1R",
)
_IMMUNE_TISSUES = ("bone_marrow", "spleen", "lymph_node", "tonsil", "appendix")
_HEME_LYMPHOID_TISSUES = _IMMUNE_TISSUES + ("thymus",)
_MUSCLE_TISSUES = ("smooth_muscle", "skeletal_muscle", "heart_muscle")
_MAX_MATCHED_NORMAL_NTPM = 3.0
_SOURCE_ALIASES = {
    "NBL": "TARGET_NBL",
    "WILMS": "TARGET_WT",
}


def _panel_maps() -> tuple[dict[str, str], dict[str, str]]:
    return dict(_SOLID_MATCHED_NORMAL_TISSUE), dict(_HEME_MATCHED_NORMAL_TISSUE)


def _exclusion_set(symbols) -> set[str]:
    exclude = set(housekeeping_gene_names()) | set(mitochondrial_gene_names())
    exclude |= _EXCLUDED_SYMBOLS
    for sym in symbols:
        if not isinstance(sym, str):
            continue
        if sym.startswith(_EXCLUDED_PREFIXES) or "-" in sym:
            exclude.add(sym)
    return exclude


def _tumor_series(
    code: str,
    *,
    tcga: pd.DataFrame,
    subtype: pd.DataFrame,
) -> pd.Series:
    if code in set(tcga["cancer_code"].dropna().astype(str)):
        rows = tcga[tcga["cancer_code"].astype(str) == code]
    else:
        source = _SOURCE_ALIASES.get(code, code)
        rows = subtype[subtype["cancer_code"].astype(str) == source]
    if rows.empty:
        return pd.Series(dtype=float, name="tumor_tpm")
    return rows.groupby("symbol")["tumor_tpm_median"].median().rename("tumor_tpm")


def _safe_id(ref: pd.DataFrame, symbol: str) -> str:
    value = ref.loc[symbol, "Ensembl_Gene_ID"] if symbol in ref.index else ""
    return "" if pd.isna(value) else str(value)


def _solid_rows_for(
    code: str,
    tissue: str,
    *,
    ref: pd.DataFrame,
    tcga: pd.DataFrame,
    subtype: pd.DataFrame,
    exclude: set[str],
    top_n: int,
) -> list[dict]:
    hpa_col = f"{tissue}_nTPM"
    if hpa_col not in ref.columns:
        return []
    immune_cols = [
        f"{t}_nTPM"
        for t in _IMMUNE_TISSUES
        if f"{t}_nTPM" in ref.columns and f"{t}_nTPM" != hpa_col
    ]
    muscle_cols = [
        f"{t}_nTPM"
        for t in _MUSCLE_TISSUES
        if f"{t}_nTPM" in ref.columns and f"{t}_nTPM" != hpa_col
    ]
    fat_col = "adipose_tissue_nTPM" if hpa_col != "adipose_tissue_nTPM" else None
    cols = [hpa_col] + immune_cols + muscle_cols + ([fat_col] if fat_col else [])

    joined = _tumor_series(code, tcga=tcga, subtype=subtype).to_frame().join(
        ref[cols],
        how="inner",
    )
    if joined.empty:
        return []
    joined["matched_normal_ntpm"] = joined[hpa_col].fillna(0).astype(float)
    joined["max_immune_ntpm"] = (
        joined[immune_cols].max(axis=1).fillna(0) if immune_cols else 0.0
    )
    joined["max_muscle_ntpm"] = (
        joined[muscle_cols].max(axis=1).fillna(0) if muscle_cols else 0.0
    )
    joined["max_fat_ntpm"] = joined[fat_col].fillna(0) if fat_col else 0.0
    joined["fold_change_vs_matched_normal"] = (
        joined["tumor_tpm"] / (joined["matched_normal_ntpm"] + 1.0)
    )
    max_other = joined[["max_immune_ntpm", "max_muscle_ntpm", "max_fat_ntpm"]].max(axis=1)
    picks = joined[
        (joined["fold_change_vs_matched_normal"] >= 10)
        & (joined["tumor_tpm"] >= 5)
        & (joined["matched_normal_ntpm"] < _MAX_MATCHED_NORMAL_NTPM)
        & (max_other < joined["tumor_tpm"])
        & (~joined.index.isin(exclude))
    ].sort_values("fold_change_vs_matched_normal", ascending=False).head(top_n)

    rows = []
    for sym, r in picks.iterrows():
        rows.append(
            {
                "cancer_code": code,
                "matched_normal_tissue": tissue,
                "symbol": sym,
                "ensembl_gene_id": _safe_id(ref, sym),
                "fold_change_vs_matched_normal": round(
                    float(r["fold_change_vs_matched_normal"]), 1
                ),
                "tumor_tpm": round(float(r["tumor_tpm"]), 2),
                "matched_normal_ntpm": round(float(r["matched_normal_ntpm"]), 2),
                "max_immune_ntpm": round(float(r["max_immune_ntpm"]), 2),
                "max_muscle_ntpm": round(float(r["max_muscle_ntpm"]), 2),
                "max_fat_ntpm": round(float(r["max_fat_ntpm"]), 2),
            }
        )
    return rows


def _heme_rows_for(
    code: str,
    tissue: str,
    *,
    ref: pd.DataFrame,
    tcga: pd.DataFrame,
    subtype: pd.DataFrame,
    exclude: set[str],
    top_n: int,
) -> list[dict]:
    hpa_col = f"{tissue}_nTPM"
    non_lymphoid_cols = [
        c
        for c in ref.columns
        if c.endswith("_nTPM")
        and c not in {f"{t}_nTPM" for t in _HEME_LYMPHOID_TISSUES}
    ]
    joined = _tumor_series(code, tcga=tcga, subtype=subtype).to_frame().join(
        ref[[hpa_col] + non_lymphoid_cols],
        how="inner",
    )
    if joined.empty:
        return []
    joined["matched_normal_ntpm"] = joined[hpa_col].fillna(0).astype(float)
    joined["max_non_lymphoid_ntpm"] = joined[non_lymphoid_cols].max(axis=1).fillna(0)
    joined["fold_change_vs_matched_normal"] = (
        joined["tumor_tpm"] / (joined["matched_normal_ntpm"] + 1.0)
    )
    picks = joined[
        (joined["fold_change_vs_matched_normal"] >= 10)
        & (joined["tumor_tpm"] >= 5)
        & (joined["matched_normal_ntpm"] < _MAX_MATCHED_NORMAL_NTPM)
        & (joined["max_non_lymphoid_ntpm"] < joined["tumor_tpm"])
        & (~joined.index.isin(exclude))
    ].sort_values("fold_change_vs_matched_normal", ascending=False).head(top_n)

    rows = []
    for sym, r in picks.iterrows():
        rows.append(
            {
                "cancer_code": code,
                "matched_normal_tissue": tissue,
                "symbol": sym,
                "ensembl_gene_id": _safe_id(ref, sym),
                "fold_change_vs_matched_normal": round(
                    float(r["fold_change_vs_matched_normal"]), 1
                ),
                "tumor_tpm": round(float(r["tumor_tpm"]), 2),
                "matched_normal_ntpm": round(float(r["matched_normal_ntpm"]), 2),
                "max_non_lymphoid_ntpm": round(float(r["max_non_lymphoid_ntpm"]), 2),
            }
        )
    return rows


def regenerate(
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    solid_map, heme_map = _panel_maps()
    ref = (
        pan_cancer_expression(technical_rna_normalize=True)
        .drop_duplicates(subset="Symbol")
        .set_index("Symbol")
    )
    tcga = tcga_deconvolved_expression()
    subtype = subtype_deconvolved_expression()
    exclude = _exclusion_set(ref.index)

    solid_rows = []
    for code, tissue in sorted(solid_map.items()):
        solid_rows.extend(
            _solid_rows_for(
                code,
                tissue,
                ref=ref,
                tcga=tcga,
                subtype=subtype,
                exclude=exclude,
                top_n=top_n,
            )
        )
    heme_rows = []
    for code, tissue in sorted(heme_map.items()):
        heme_rows.extend(
            _heme_rows_for(
                code,
                tissue,
                ref=ref,
                tcga=tcga,
                subtype=subtype,
                exclude=exclude,
                top_n=top_n,
            )
        )
    solid = pd.DataFrame(solid_rows)
    heme = pd.DataFrame(heme_rows)
    solid_codes = set(solid["cancer_code"]) if "cancer_code" in solid.columns else set()
    heme_codes = set(heme["cancer_code"]) if "cancer_code" in heme.columns else set()
    missing_solid = sorted(set(solid_map) - solid_codes)
    missing_heme = sorted(set(heme_map) - heme_codes)
    return solid, heme, missing_solid, missing_heme


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top", type=int, default=10)
    p.add_argument(
        "--solid-output",
        default="trufflepig/data/tumor-up-vs-matched-normal.csv",
    )
    p.add_argument(
        "--heme-output",
        default="trufflepig/data/heme-tumor-up-vs-matched-normal.csv",
    )
    args = p.parse_args()

    solid, heme, missing_solid, missing_heme = regenerate(top_n=args.top)
    for frame, output in ((solid, args.solid_output), (heme, args.heme_output)):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"wrote {len(frame)} rows to {path}")
    if missing_solid:
        print(f"no regenerated solid rows for: {', '.join(missing_solid)}")
    if missing_heme:
        print(f"no regenerated heme rows for: {', '.join(missing_heme)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
