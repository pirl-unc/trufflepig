#!/usr/bin/env python3
"""Curation aid for request (a): propose candidate lineage-marker panels for the SARC subtypes that
have a deconvolved tumor profile but NO panel yet, so the mixture pick can expand past LMS/SYN/LPS.

These are DATA-DERIVED suggestions (most subtype-specific highly-expressed genes vs the other SARC
subtypes) — a starting point for biological curation in pirlygenes' lineage-genes.csv, NOT validated
panels. For each gap subtype we print the top specific markers + whether each is already a known
lineage gene elsewhere.

Run:  python3 scripts/derive_sarc_subtype_panels.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from trufflepig.reference import subtype_deconvolved_expression
from trufflepig.tumor_purity import _lineage_genes_map, _subtype_tumor_tpm_lookup
from trufflepig.cancer_ontology import cancer_type_subtypes_of

N_MARKERS = 12
MIN_TPM = 20.0          # the subtype must actually express it
MIN_LOG2_SPEC = 1.5     # >= ~3x the median of the other SARC subtypes


def main():
    lg = _lineage_genes_map()
    subs = cancer_type_subtypes_of("SARC") or []
    profiled = [s for s in subs if _subtype_tumor_tpm_lookup(s)]
    gap = [s for s in profiled if not lg.get(s)]
    have = [s for s in profiled if lg.get(s)]
    print(f"SARC: {len(profiled)} profiled subtypes — {len(have)} with panels {have}, "
          f"{len(gap)} WITHOUT panels (curatable):", file=sys.stderr)

    dec = subtype_deconvolved_expression()
    med = dec.pivot_table(index="symbol", columns="cancer_code", values="tumor_tpm_median", aggfunc="median")
    sarc_cols = [c for c in med.columns if str(c).startswith("SARC_")]
    med = med[sarc_cols]
    logmed = np.log2(med + 1.0)
    known_lineage = {g for genes in lg.values() for g in genes}

    for sub in gap:
        if sub not in logmed.columns:
            continue
        others = logmed.drop(columns=[sub]).median(axis=1)
        spec = (logmed[sub] - others)
        cand = spec[(med[sub] >= MIN_TPM) & (spec >= MIN_LOG2_SPEC)].sort_values(ascending=False)
        markers = list(cand.head(N_MARKERS).index)
        print(f"\n{sub}  (proposed {len(markers)} markers):")
        for g in markers:
            tag = "  [known-lineage-gene]" if g in known_lineage else ""
            print(f"    {g:14s} tpm={med.at[g, sub]:8.1f}  log2-spec={spec[g]:+.2f}{tag}")
        if not markers:
            print("    (no gene cleared the specificity/expression floor — needs manual curation)")


if __name__ == "__main__":
    main()
