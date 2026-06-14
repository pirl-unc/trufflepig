#!/usr/bin/env python3
"""Bake-off: bulk screen vs supervised vs blind-program decomposition.

Empirical comparison (the architecture question raised in
docs/cancer-type-residual-matching-findings.md) of strategies for calling the
tumour's **broad lineage** on real local samples, scored against known truth:

  S1 production  — the bulk cross-cohort screen's top call (the 5/5 incumbent).
  S2 supervised  — decompose constrained to each top-K bulk candidate; the
                   best-fitting candidate's lineage (reconstruction error +
                   cancer_signature_score). Reuses decompose_sample.
  S3 blind       — NNLS the sample against the HPA cell-type *lineage-program*
                   basis with NO designated tumour; classify each program
                   tumour vs infiltrate/stroma using identifiable anchors
                   (immune/endothelial = TME); the dominant tumour-candidate
                   program is the lineage. Reuses hpa_cell_type_expression.

All three reuse existing machinery — no new analysis modules.

NOTE: exploratory harness, not a test. ``SAMPLES`` hardcodes machine-specific
absolute paths to local truth-set data (not in the repo); edit them to re-run
elsewhere. Not collected by pytest.

Run:  scripts/decomposition_bakeoff.py            # uses the embedded truth set
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# (truth broad-lineage, input expression path[, column]) — accessible local
# samples with known ground truth (the findings-doc set + a few more). The
# optional third element picks one column out of a multi-sample matrix file.
SAMPLES = [
    ("mesenchymal", "/Users/iskander/data/pathfinder/pfo004/analysis/gene-expression.csv"),  # pfo004 osteosarcoma
    ("epithelial", "/Users/iskander/data/pathfinder/pfo002/Personalis-PSNLDx20240402135/Processed/RNA_Pipeline/Expression_Reports/tsv/RNA_PSNLDx20240402135_tumor_rna_gene_expression_report.tsv"),  # pfo002 CRC
    ("epithelial", "/Users/iskander/data/Kat-DL/tempus/tempus-rna/salmon_rich_quant/quant.gene_tpm.csv"),  # tempus PRAD
    ("epithelial", "/Users/iskander/data/asy/salmon-output-asy/quant.gene_tpm.csv"),  # asy CRC
    ("mesenchymal", "/Users/iskander/data/alvin/RNA/2025-10-31_salmon/quant.gene_tpm.csv"),  # alvin SARC
    ("epithelial", "/Users/iskander/data/hcc1395/rnaseq/kallisto_expression/gene_abundance.tsv"),  # hcc1395 BRCA
    ("epithelial", "/Users/iskander/data/pathfinder/pfo017/salmon.merged.gene_tpm.tsv", "PFO017-bladder-2025"),  # pfo017 bladder
]

# HPA cell-type -> lineage program (keyword match; unmatched epithelial-default,
# since most HPA cell types are epithelial parenchyma). immune/endothelial are
# the identifiable TME anchors that are never the tumour (outside heme/angio).
_PROGRAM_KEYWORDS = [
    ("immune", ("b-cell", "t-cell", "nk-", "nk cell", "macrophage", "dendritic",
                "granulocyte", "plasma cell", "kupffer", "langerhans", "microglia",
                "erythroid", "monocyte", "neutrophil", "mast cell", "lymph")),
    ("endothelial", ("endothelial",)),
    ("mesenchymal", ("fibroblast", "adipocyte", "smooth muscle", "chondrocyte",
                     "stromal", "myofibroblast", "peritubular", "mesenchym")),
    ("neural", ("astrocyte", "neuron", "oligodendrocyte", "schwann", "bipolar",
                "müller", "muller", "photoreceptor", "glial", "horizontal cell")),
    ("melanocytic", ("melanocyte",)),
    ("neuroendocrine", ("enteroendocrine", "neuroendocrine", "endocrine",
                        "alpha cell", "beta cell", "islet")),
    ("germ", ("spermat", "oocyte", "trophoblast")),
]


def _program_of(cell_type: str) -> str:
    name = cell_type.lower()
    for program, keys in _PROGRAM_KEYWORDS:
        if any(k in name for k in keys):
            return program
    return "epithelial"


def _load_program_basis():
    """Lineage-program reference profiles = mean of each program's HPA cell types."""
    from trufflepig.reference import hpa_cell_type_expression

    hpa = hpa_cell_type_expression().drop_duplicates(subset="Symbol").set_index("Symbol")
    cell_cols = [c for c in hpa.columns if c not in ("Ensembl_Gene_ID",)]
    by_prog: dict[str, list[str]] = {}
    for c in cell_cols:
        by_prog.setdefault(_program_of(c), []).append(c)
    basis = pd.DataFrame(
        {prog: hpa[cols].mean(axis=1) for prog, cols in by_prog.items()}
    )
    return basis  # index=Symbol, columns=programs


def _load_sample(path: str, column: str | None = None):
    """Load + conform a sample to the reference clean-TPM space; return the
    symbol->TPM dict the classifier/decomposition consume, and the loaded df.

    ``column`` selects one sample out of a multi-sample matrix file (which
    ``load_expression_data`` rejects because it can't auto-pick a TPM column)."""
    from trufflepig.load_expression import load_expression_data
    from trufflepig.clean_tpm import normalize_to_reference_space, resolve_gene_columns
    from trufflepig.tumor_purity import _build_sample_tpm_by_symbol

    if column is not None:
        sep = "\t" if path.endswith((".tsv", ".txt")) else ","
        raw = pd.read_csv(path, sep=sep)
        gene = next(c for c in ("gene_name", "gene", "Symbol") if c in raw.columns)
        idc = next(c for c in ("gene_id", "ensembl_gene_id", "Ensembl_Gene_ID") if c in raw.columns)
        df = raw[[gene, idc, column]].rename(
            columns={gene: "gene_name", idc: "ensembl_gene_id", column: "TPM"}
        )
    else:
        df = load_expression_data(path)
    label, idc = resolve_gene_columns(df)
    df = normalize_to_reference_space(df, value_cols=["TPM"], label_col=label, id_col=idc)
    return _build_sample_tpm_by_symbol(df), df


def strategy_production(df) -> str:
    from trufflepig.tumor_purity import rank_cancer_type_candidates
    from trufflepig.cancer_type_ontology import broad_lineage

    rows = rank_cancer_type_candidates(df, top_k=10)
    return broad_lineage(rows[0]["code"]) if rows else "?", (rows[0]["code"] if rows else "?")


def strategy_supervised(df, candidate_codes):
    """Decompose constrained to the top-K bulk candidates; return the lineage of
    the candidate whose tumour template best explains the sample (decompose_sample
    picks it by reconstruction error + signature score). Reuses the engine."""
    from trufflepig.decomposition.engine import decompose_sample
    from trufflepig.cancer_type_ontology import broad_lineage

    res = decompose_sample(df, cancer_types=candidate_codes, top_k=len(candidate_codes))
    if isinstance(res, list):  # ranked list of DecompositionResult; best is first
        res = res[0] if res else None
    code = getattr(res, "cancer_type", None) if res is not None else None
    return (broad_lineage(code) if code else "?"), (code or "?")


def strategy_blind_program(sample_tpm: dict, basis: pd.DataFrame):
    """NNLS the sample against the lineage-program basis; report the dominant
    tumour-candidate program (immune/endothelial are TME, never the tumour)."""
    from scipy.optimize import nnls

    genes = [g for g in basis.index if g in sample_tpm]
    if len(genes) < 200:
        return "?", {}
    B = np.log1p(basis.loc[genes].to_numpy())
    y = np.log1p(np.array([sample_tpm[g] for g in genes]))
    w, _ = nnls(B, y)
    weights = dict(zip(basis.columns, w))
    total = sum(weights.values()) or 1.0
    frac = {k: v / total for k, v in weights.items()}
    # tumour candidate = dominant program that is not a pure-TME anchor
    tme = {"immune", "endothelial"}
    tumour_cands = {k: v for k, v in frac.items() if k not in tme}
    call = max(tumour_cands, key=tumour_cands.get) if tumour_cands else "?"
    return call, frac


def main(argv=None):
    from trufflepig.tumor_purity import rank_cancer_type_candidates
    from trufflepig.cancer_type_ontology import broad_lineage

    basis = _load_program_basis()
    print(f"program basis: {list(basis.columns)} ({len(basis)} genes)\n")
    hdr = f"{'truth':12s} {'S1 prod':16s} {'S2 superv':16s} {'S3 blind':22s} {'S1':3s}{'S2':3s}{'S3'}"
    print(hdr)
    print("-" * len(hdr))
    s1_ok = s2_ok = s3_ok = n = 0
    for entry in SAMPLES:
        truth, path = entry[0], entry[1]
        column = entry[2] if len(entry) > 2 else None
        try:
            sample_tpm, df = _load_sample(path, column)
            rows = rank_cancer_type_candidates(df, top_k=10)
            cands = [r["code"] for r in rows[:5]]
            s1_lin = broad_lineage(cands[0]) if cands else "?"
            s1_code = cands[0] if cands else "?"
            s2_lin, s2_code = strategy_supervised(df, cands)
            s3_lin, frac = strategy_blind_program(sample_tpm, basis)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"{truth:12s} ERROR: {str(e)[:60]}")
            continue
        n += 1
        s1_ok += s1_lin == truth
        s2_ok += s2_lin == truth
        s3_ok += s3_lin == truth
        top2 = ",".join(f"{k}={v:.2f}" for k, v in sorted(frac.items(), key=lambda x: -x[1])[:2])
        print(f"{truth:12s} {s1_lin+'('+s1_code+')':16s} {s2_lin+'('+s2_code+')':16s} "
              f"{s3_lin+'['+top2+']':22s} "
              f"{'✓' if s1_lin==truth else '✗':3s}{'✓' if s2_lin==truth else '✗':3s}"
              f"{'✓' if s3_lin==truth else '✗'}")
    if n:
        print(f"\nS1 production: {s1_ok}/{n}   S2 supervised-decomp: {s2_ok}/{n}   "
              f"S3 blind-program: {s3_ok}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
