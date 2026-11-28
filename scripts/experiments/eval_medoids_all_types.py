#!/usr/bin/env python3
"""Run the lineage-decomposition + purity signals over every cancer type's representative (medoid)
sample, and score them against the type's known lineage.

Per type:
  - NO-HINT: compartment_call mode + lineage_fit argmax (does expression alone pick the lineage?)
  - HINTED:  estimate_tumor_purity(cancer_type=type) → overall, ESTIMATE+gating, decomposition
             (mode / residual_fraction / aneuploidy_purity / lineage_conflict), purity_consistency

Truth = _group_to_mode(cancer_lineage_group(type)). The medoid is the clean-TPM mean of the type's
representative cohort (clean_tpm here only — it is a dev/eval tool, not the runtime path).

Run:  python3 scripts/eval_medoids_all_types.py
"""
import sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd

from pirlygenes.expression.accessors import available_representative_cohorts, representative_cohort_samples
from pirlygenes.gene_sets_cancer import cancer_lineage_group
from trufflepig.expression_decomposition import decompose_expression, _group_to_mode
from trufflepig.tumor_purity import estimate_tumor_purity

MODES = ["solid", "mesenchymal", "heme", "embryonal"]


def _medoid(type_code):
    """(df_gene_expr, {symbol: clean-TPM}) for a type's representative cohort mean."""
    from oncoref.normalization import clean_tpm
    d = representative_cohort_samples(type_code).drop_duplicates("Ensembl_Gene_ID")
    cols = [c for c in d.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    gt = pd.DataFrame({"Ensembl_Gene_ID": d["Ensembl_Gene_ID"].values, "Symbol": d["Symbol"].values})
    clean = clean_tpm(d.set_index("Ensembl_Gene_ID")[cols].astype(float), gene_table=gt.set_index(d.index))
    tpm = clean.mean(axis=1)
    df = pd.DataFrame({"ensembl_gene_id": d["Ensembl_Gene_ID"].values,
                       "gene_symbol": d["Symbol"].values, "TPM": tpm.values})
    sym = df.groupby("gene_symbol")["TPM"].sum().to_dict()
    return df, sym


def evaluate(types):
    rows = []
    for t in types:
        truth = _group_to_mode(cancer_lineage_group(t) or "") if cancer_lineage_group(t) else None
        row = {"type": t, "truth": truth, "consistency": []}
        try:
            df, sym = _medoid(t)
        except Exception as e:  # noqa: BLE001
            row["error"] = str(e)[:70]; rows.append(row); continue
        try:                                                                  # no-hint lineage (works for any type)
            nh = decompose_expression(sym, cancer=None, run_all=True)
            row["fits"] = {m: nh["modes"].get(m, {}).get("lineage_fit") for m in MODES}
            row["clf_mode"] = nh["selected_mode"]
            row["fit_argmax"] = max(row["fits"], key=lambda m: (row["fits"][m] if row["fits"][m] is not None else -1e9))
        except Exception as e:  # noqa: BLE001
            row["nh_error"] = str(e)[:50]
        try:                                                                  # AUTO-detect end-to-end (non-circular)
            pur = estimate_tumor_purity(df)                                    # no hint: ranking picks the type
            comp = pur.get("components", {}); dc = comp.get("decomposition") or {}
            auto_code = pur.get("cancer_type")
            auto_grp = cancer_lineage_group(auto_code) if auto_code else None
            row.update({"auto_code": auto_code,
                        "auto_lineage": _group_to_mode(auto_grp) if auto_grp else None,
                        "decomp_mode": dc.get("mode"), "conflict": dc.get("lineage_conflict"),
                        "overall": pur.get("overall_estimate"), "estimate": comp.get("estimate_purity"),
                        "gated": comp.get("estimate_gated_for_lineage"),
                        "residual_fraction": dc.get("residual_fraction"), "aneuploidy_purity": dc.get("aneuploidy_purity"),
                        "consistency": pur.get("purity_consistency") or []})
        except Exception as e:  # noqa: BLE001
            row["purity_error"] = str(e)[:50]
        rows.append(row)
    return rows


def main():
    types = sorted(available_representative_cohorts())
    rows = evaluate(types)
    load_err = [r for r in rows if "error" in r]

    def f(v, w=6):
        return ("%*.2f" % (w, v)) if isinstance(v, (int, float)) else f"{str(v):>{w}s}"

    print(f"\n{'type':14s} {'truth':6s} {'clf':6s} {'fitamx':7s} {'decomp':6s} {'cflt':4s} | "
          f"{'overall':>7s} {'gated':>5s} {'resid':>6s} {'aneu':>6s} flags")
    print("-" * 110)
    for r in rows:
        if "error" in r:
            continue
        purity = "no-pan-cancer-ref" if "purity_error" in r else (
            f"{f(r.get('overall'),7)} {str(r.get('gated')):>5s} {f(r.get('residual_fraction'))} "
            f"{f(r.get('aneuploidy_purity'))} {'RECON' if r.get('consistency') else ''}")
        print(f"{r['type']:14s} {str(r['truth']):6s} {str(r.get('clf_mode'))[:6]:6s} {str(r.get('fit_argmax'))[:6]:7s} "
              f"{str(r.get('decomp_mode'))[:6]:6s} {('Y' if r.get('conflict') else ''):4s} | {purity}")
    print("-" * 110)

    nh = [r for r in rows if r.get("clf_mode") and r.get("truth")]              # no-hint scored
    auto = [r for r in rows if r.get("decomp_mode") and r.get("truth")]         # auto-detect end-to-end
    al = [r for r in rows if r.get("auto_lineage") and r.get("truth")]
    gated_rows = [r for r in rows if "gated" in r and r.get("truth")]
    print(f"\nTypes: {len(rows)} total, {len(load_err)} failed to load, "
          f"{sum('purity_error' in r for r in rows)} errored in auto purity.  (ALL metrics below are "
          f"NON-circular: no cancer-type hint given.)")
    print(f"  NO-HINT lineage == truth ({len(nh)} types): compartment_call "
          f"{sum(r['clf_mode']==r['truth'] for r in nh)}/{len(nh)}   "
          f"lineage_fit-argmax {sum(r['fit_argmax']==r['truth'] for r in nh)}/{len(nh)}")
    print(f"  AUTO cancer-type-call lineage == truth ({len(al)} types): "
          f"{sum(r['auto_lineage']==r['truth'] for r in al)}/{len(al)}  (rank_cancer_type_candidates)")
    print(f"  AUTO end-to-end decomp mode == truth ({len(auto)} types): "
          f"{sum(r['decomp_mode']==r['truth'] for r in auto)}/{len(auto)}  (auto code → decomposition + lineage_fit)")
    hs = [r for r in gated_rows if r["truth"] in ("heme", "mesenchymal")]
    ep = [r for r in gated_rows if r["truth"] == "solid"]
    print(f"  ESTIMATE gated on heme/sarcoma: {sum(bool(r['gated']) for r in hs)}/{len(hs)}   "
          f"wrongly gated on solid: {sum(bool(r['gated']) for r in ep)}/{len(ep)}")
    print(f"  lineage conflicts (auto code vs expression): {sum(bool(r.get('conflict')) for r in auto)}/{len(auto)}")
    print(f"  reconciliation flags fired: {sum(bool(r.get('consistency')) for r in rows)}/{len(rows)}")
    print("\nno-hint misses (compartment_call != truth) — auto decomp shown for contrast:")
    for r in nh:
        if r["clf_mode"] != r["truth"]:
            fv = " ".join(f"{m[:4]}={(r['fits'][m] if r['fits'][m] is not None else float('nan')):.0f}" for m in MODES)
            print(f"  {r['type']:14s} truth={r['truth']:11s} clf={r['clf_mode']:11s} "
                  f"decomp={str(r.get('decomp_mode')):11s} [{fv}]")
    if load_err:
        print("\nload errors:", [(r['type'], r['error']) for r in load_err][:10])
    return rows


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
