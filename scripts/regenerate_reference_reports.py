#!/usr/bin/env python3
"""Generate trufflepig reports for every per-code *representative* expression profile.

A validation sweep: pirlygenes ships one parquet per cancer code under
``cancer-reference-expression-representatives/`` (columns = representative
samples for that code). This explodes each column into a TPM sample, runs it
through ``trufflepig.main.analyze``, and writes a report workspace per sample so
``scripts/analyze_reports.py`` can score the calls against the source code (the
ground truth is the parquet/code name — pass ``--infer-expected`` to the scorer).

Parametric in source + destination so it never writes inside the repo by
default and can target any representative set / output root:

    scripts/regenerate_reference_reports.py \
        --rep-dir ~/code/pirlygenes/pirlygenes/data/cancer-reference-expression-representatives \
        --out ~/trufflepig-reference-reports/$(date +%Y%m%d) --workers 6

Pair with the scorer to measure lift between two runs:

    scripts/analyze_reports.py --reports <new-out> --baseline <old-out> --infer-expected
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_DEFAULT_REP_DIR = Path(
    "/Users/iskander/code/pirlygenes/pirlygenes/data/"
    "cancer-reference-expression-representatives"
)


def _build_jobs(rep_dir: Path, samp_root: Path):
    import pandas as pd

    jobs = []
    for pq in sorted(glob.glob(f"{rep_dir}/*.parquet")):
        code = os.path.basename(pq)[: -len(".parquet")]
        df = pd.read_parquet(pq)
        rep_cols = [c for c in df.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
        (samp_root / code).mkdir(parents=True, exist_ok=True)
        for rc in rep_cols:
            s = df[["Ensembl_Gene_ID", "Symbol", rc]].rename(columns={rc: "TPM"})
            s = s[s["TPM"].notna()]
            tsv = samp_root / code / f"{rc}.tsv"
            s.to_csv(tsv, sep="\t", index=False)
            jobs.append((code, rc, str(tsv)))
    return jobs


def _run(job):
    warnings.filterwarnings("ignore")
    code, rc, tsv, out_root = job
    outdir = Path(out_root) / code / rc
    outdir.mkdir(parents=True, exist_ok=True)
    from trufflepig.main import analyze

    try:
        analyze(input_path=tsv, output_dir=str(outdir), no_figures=True)
        return (code, rc, "ok")
    except Exception as e:  # noqa: BLE001 — record, don't abort the sweep
        return (code, rc, f"FAIL: {type(e).__name__}: {e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rep-dir", type=Path, default=_DEFAULT_REP_DIR,
                    help="dir of pirlygenes per-code representative parquets")
    ap.add_argument("--out", type=Path, required=True, help="report output root")
    ap.add_argument("--samples", type=Path, default=None,
                    help="where to stage the exploded sample TSVs (default: <out>/_samples)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-codes", nargs="*", default=None,
                    help="only these codes (parquet stems)")
    args = ap.parse_args(argv)

    if not args.rep_dir.is_dir():
        ap.error(f"--rep-dir {args.rep_dir} is not a directory")
    samp_root = args.samples or (args.out / "_samples")
    args.out.mkdir(parents=True, exist_ok=True)
    samp_root.mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(args.rep_dir, samp_root)
    if args.limit_codes:
        keep = set(args.limit_codes)
        jobs = [j for j in jobs if j[0] in keep]
    jobs = [(c, rc, tsv, str(args.out)) for (c, rc, tsv) in jobs]
    ncodes = len({j[0] for j in jobs})
    print(f"total rep samples: {len(jobs)} across {ncodes} codes -> {args.out}", flush=True)

    t0 = time.time()
    done = 0
    fails = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run, j) for j in jobs]
        for fut in as_completed(futs):
            code, rc, status = fut.result()
            done += 1
            if status != "ok":
                fails.append((code, rc, status))
            if done % 25 == 0:
                rate = done / (time.time() - t0)
                eta = (len(jobs) - done) / rate if rate else 0
                print(f"  {done}/{len(jobs)} ({time.time() - t0:.0f}s, eta {eta:.0f}s, fails {len(fails)})", flush=True)
    print(f"DONE {done} in {time.time() - t0:.0f}s; fails: {len(fails)}", flush=True)
    for f in fails[:40]:
        print("  FAIL", f, flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
