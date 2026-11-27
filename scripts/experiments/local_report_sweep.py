#!/usr/bin/env python
"""Sweep the local real-sample quant files → full trufflepig markdown reports, print the key sections.

Generates each report with --no-figures (fast) under /tmp/tp_sweep/<name>, then prints the cancer-type
call, lineage, purity, and decomposition lines from the markdown so the run can be eye-checked against
the curated truth. Use to validate ranker/normalization changes on real samples end-to-end.
"""
import re
import subprocess
import sys
from pathlib import Path

D = "/Users/iskander/data"
REPORTS = [
    ("alvin-sarcoma", f"{D}/alvin/RNA/2025-10-31_salmon/quant.gene_tpm.csv", "mesenchymal"),
    ("hcc1395-kallisto", f"{D}/hcc1395/rnaseq/kallisto_expression/gene_abundance.tsv", "solid"),
    ("hcc1395-stringtie", f"{D}/hcc1395/rnaseq/stringtie_expression/stringtie_gene_expression.tsv", "solid"),
    ("pfo002-colon", f"{D}/pathfinder/pfo002/WashU/mcdb032-BG002179-2022-05-colon/mcdb-workflow_results/gene_abundance.tsv", "solid"),
    ("pfo004-osteosarc", f"{D}/pathfinder/pfo004/analysis/gene-expression.csv", "mesenchymal"),
    ("pfo019-sinonasal", f"{D}/pathfinder/pfo019/BostonGene-BG011335-2024-03-20-nasal/Processed/final_results/final_results/rnaseq/kallisto_expression/gene_abundance.tsv", "solid"),
    ("tempus-nutm1", f"{D}/tempus-unc-nutm1/data_backfill/Data/Group_Level_Molecular/normalized_rna.csv", "solid"),
]
OUT = Path("/tmp/tp_sweep")
KEY = re.compile(r"cancer.?type|lineage|compartment|purity|decompos|residual|aneuploid|tumor.?content|"
                 r"primary|histolog|confidence|centroid", re.I)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, path, truth in REPORTS:
        if only and only not in name:
            continue
        if not Path(path).exists():
            print(f"\n##### {name}: INPUT MISSING ({path})")
            continue
        odir = OUT / name
        print(f"\n{'='*90}\n##### {name}   (truth lineage: {truth})\n{'='*90}")
        r = subprocess.run(
            ["python", "-m", "trufflepig.cli", "run", "--sample", path, "--workspace", str(odir),
             "--no-figures", "--force"],
            capture_output=True, text=True, timeout=900,
        )
        if r.returncode != 0:
            print(f"  ANALYZE FAILED (rc={r.returncode}):\n  {r.stderr[-600:]}")
            continue
        # `run` writes reports under <workspace>/analyze/. Prefer the summary (headline call).
        mds = sorted(odir.rglob("*summary.md")) + sorted(odir.rglob("*analysis.md"))
        if not mds:
            mds = sorted(odir.rglob("*.md"))
        md = mds[0] if mds else None
        if md is None:
            print("  (no markdown produced)")
            continue
        lines = md.read_text().splitlines()
        printed = 0
        for ln in lines:
            s = ln.strip()
            if s and (s.startswith("#") or KEY.search(s)) and not s.startswith("|--"):
                print("  " + s[:160])
                printed += 1
                if printed > 40:
                    print("  ...(truncated)")
                    break
        print(f"  [full report: {md}]")


if __name__ == "__main__":
    main()
