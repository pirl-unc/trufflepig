#!/usr/bin/env python3
"""Full per-sample signal dump for the 8 cross-compartment misses: EVERY signal the pipeline
generates and exactly which one determined the final call.

Run:  python3 scripts/diagnose_full.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from trufflepig.cancer_type_centroid import compartment_call
from trufflepig.cancer_type_evidence import select_report_scope_from_evidence
from trufflepig.expression_decomposition import decompose_expression
from trufflepig.tumor_purity import _build_sample_tpm_by_symbol, analyze_sample

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_per_sample_confusion import _clean_cohort, name  # noqa: E402

# (truth_type, sample_index, observed_miss_call)
SAMPLES = [
    ("ADCC", 3, "SARC_MYXFIB"), ("ATRT", 3, "SARC_MYXFIB"), ("BRCA_HER2", 2, "DLBC"),
    ("CHOL", 2, "CLL"), ("KICH", 4, "SARC_LPS_UNSPEC"), ("LIHC", 1, "HEPB"),
    ("LUAD_KRAS", 3, "SARC_LPS_UNSPEC"), ("LUAD_STK11", 4, "SARC_LPS_UNSPEC"),
]
MODES = ["solid", "mesenchymal", "heme", "embryonal"]


def _j(obj, n=1400):
    return json.dumps(obj, default=str)[:n]


def dump(truth, idx, miss_call):
    ens, sym_names, cleaned, cols = _clean_cohort(truth)
    df = pd.DataFrame({"ensembl_gene_id": ens, "gene_symbol": sym_names, "TPM": cleaned[cols[idx]].values})
    sym = _build_sample_tpm_by_symbol(df)
    a = analyze_sample(df)
    cc = compartment_call(sym)
    dec = decompose_expression(sym, cancer=None, run_all=True)
    sc = select_report_scope_from_evidence(df, a)
    seg = sc.get("staged_evidence_graph") or {}

    print("=" * 100)
    print(f"{name(truth)} [{truth}] sample#{idx}  ==>  CALLED {name(miss_call)} [{miss_call}]")
    print("=" * 100)

    print("\n-- (1) BULK CLASSIFIER (analyze_sample) --")
    print(f"   cancer_type={a.get('cancer_type')} score={a.get('cancer_score')} "
          f"fit_quality={a.get('fit_quality')}")
    print(f"   top_cancers={_j(a.get('top_cancers'), 300)}")
    print("   candidate_trace (code | score | concordance | detn | winning_subtype):")
    for c in (a.get("candidate_trace") or [])[:8]:
        print(f"      {c.get('code'):16s} score={c.get('score')} conc={c.get('concordance')} "
              f"detn={c.get('detection_fraction')} subtype={c.get('winning_subtype')}")
    print(f"   cancer_call_rescue={_j(a.get('cancer_call_rescue'), 300)}")

    print("\n-- (2) TISSUE COMPOSITION (the 'other tissue present') --")
    ts = a.get("tissue_scores") or {}
    top_t = sorted(ts.items(), key=lambda kv: -float(kv[1] or 0))[:6] if isinstance(ts, dict) else ts
    print(f"   tissue_scores(top)={_j(top_t, 400)}")

    print("\n-- (3) COMPARTMENT_CALL (lineage classifier) --")
    print(f"   {_j(cc, 400)}")

    print("\n-- (4) DECOMPOSITION (decompose_expression run_all) --")
    print(f"   selected_mode={dec.get('selected_mode')} routing={_j(dec.get('routing'), 200)}")
    for m in MODES:
        md = (dec.get("modes") or {}).get(m) or {}
        print(f"      {m:11s} lineage_fit={md.get('lineage_fit')} residual_fraction={md.get('residual_fraction')} "
              f"signal={md.get('tumor_lineage_signal')} leak={md.get('subtracted_leakage')} "
              f"aneu={(md.get('characterization') or {}).get('aneuploidy_score')}")

    print("\n-- (5) PURITY (estimate_tumor_purity components) --")
    pur = a.get("purity") or {}
    comp = pur.get("components") or {}
    print(f"   overall={pur.get('overall_estimate')} integration={comp.get('integration')} "
          f"reconciliation={pur.get('purity_consistency')}")
    for k in ("signature", "lineage"):
        cm = comp.get(k) or {}
        print(f"      {k}: purity={cm.get('purity')} winning_subtype={cm.get('winning_subtype')} "
              f"n_genes={len(cm.get('genes', []))} detn={cm.get('detection_fraction')}")
    print(f"      estimate_purity={comp.get('estimate_purity')} gated={comp.get('estimate_gated_for_lineage')} "
          f"decomp_mode={(comp.get('decomposition') or {}).get('mode')}")

    print("\n-- (6) EVIDENCE SELECTION (what DETERMINED the call) --")
    sel = sc.get("selected") or {}
    print(f"   selected_by={sel.get('selected_by')} selection_method={sel.get('selection_method')}")
    print(f"   label_decision={_j(sel.get('label_decision'), 300)}")
    print(f"   metrics={_j(sel.get('metrics'), 500)}")
    print(f"   blocking_reasons={_j(sel.get('blocking_reasons'), 200)}")
    print(f"   top_reference_cancer_type={sc.get('top_reference_cancer_type')}")
    print(f"   lineage_panel_evidence={_j(sc.get('lineage_panel_evidence'), 400)}")

    print("\n-- (7) STAGED EVIDENCE GRAPH (every channel + selection order) --")
    print(f"   selection_order={_j(seg.get('selection_order'), 300)}")
    print(f"   lineage_path={_j(seg.get('lineage_path'), 300)}")
    chans = seg.get("channels") or {}
    items = chans.items() if isinstance(chans, dict) else enumerate(chans)
    for key, ch in items:
        if not isinstance(ch, dict):
            continue
        print(f"      [{key}] call={ch.get('cancer_type') or ch.get('call') or ch.get('code')} "
              f"status={ch.get('status')} score={ch.get('score') or ch.get('support')} "
              f"detail={_j({k: v for k, v in ch.items() if k not in ('cancer_type','call','code','status','score','support')}, 280)}")
    print()


def main():
    for truth, idx, miss in SAMPLES:
        try:
            dump(truth, idx, miss)
        except Exception as exc:  # noqa: BLE001
            print(f"!! {truth}#{idx} dump failed: {exc!r}\n")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
