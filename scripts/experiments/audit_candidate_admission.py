#!/usr/bin/env python3
"""How many cancer types does the PANEL-signature admission gate lock out, despite a strong centroid?

The candidate set's dominant admission gate is the marker-panel signature top-8
(``_compute_cancer_type_signature_stats``); the whole-profile CENTROID gets no admission vote (it
only re-weights admitted candidates). This audits the scope directly and cheaply: for every
representative cohort's own medoid, compare where the type ranks by the admission signal (panel
signature) vs the robust signal (centroid).

  signature_rank  — rank of the true type by the marker-panel signature (admission ~ top 8)
  centroid_rank   — rank of the true type by whole-profile centroid correlation (its own medoid)

The gap class: ``centroid_rank is excellent (<=3) but signature_rank > 8`` — the centroid nails the
type on its own profile, but the panel-only admission gate would never let it into the candidate set.
(Family/subtype panels admit a few extra codes on top of the top-8, so this measures the dominant
signature gate; a type flagged here may still squeak in via a family panel.)

Run:  python3 scripts/audit_candidate_admission.py
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from pirlygenes.expression.accessors import (
    available_representative_cohorts,
    representative_cohort_samples,
)
from oncoref.normalization import clean_tpm
from trufflepig.cancer_ontology import registry_parent_code
from trufflepig.cancer_type_centroid import centroid_correlations
from trufflepig.plot_embedding import _compute_cancer_type_signature_stats
from trufflepig.tumor_purity import _build_sample_tpm_by_symbol

ADMISSION_TOP_N = 8  # the signature top-8 admitted by rank_cancer_type_candidates (tumor_purity.py)


def _medoid_df(code):
    d = representative_cohort_samples(code).drop_duplicates("Ensembl_Gene_ID")
    cols = [c for c in d.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    gt = pd.DataFrame({"Ensembl_Gene_ID": d["Ensembl_Gene_ID"].values, "Symbol": d["Symbol"].values})
    cl = clean_tpm(d.set_index("Ensembl_Gene_ID")[cols].astype(float), gene_table=gt.set_index(d.index))
    return pd.DataFrame({"ensembl_gene_id": d["Ensembl_Gene_ID"].values,
                         "gene_symbol": d["Symbol"].values, "TPM": cl.mean(axis=1).values})


def _rank_of(code, ordered_codes):
    return (ordered_codes.index(code) + 1) if code in ordered_codes else None


def main():
    codes = sorted(available_representative_cohorts())
    rows = []
    for i, code in enumerate(codes, 1):
        try:
            df = _medoid_df(code)
            stats = _compute_cancer_type_signature_stats(df)
            sig_order = [r["code"] for r in stats]
            cc = centroid_correlations(_build_sample_tpm_by_symbol(df))
            parent = registry_parent_code(code)
            # admitted by the signature gate if the type OR its parent (subtype -> broad) is top-N
            admitted_codes = set(sig_order[:ADMISSION_TOP_N])
            sig_admitted = code in admitted_codes or (parent and parent in admitted_codes)
            rows.append((code, _rank_of(code, sig_order), _rank_of(code, list(cc.index)), bool(sig_admitted)))
        except Exception as exc:  # noqa: BLE001
            rows.append((code, None, None, None))
            print(f"  !! {code}: {str(exc)[:50]}", file=sys.stderr)
        print(f"  [{i}/{len(codes)}] {code}", flush=True)

    print(f"\n{'type':18s} {'sig#':>5s} {'cen#':>5s} {'admitted':>9s}")
    gap = []
    for code, sig, cen, adm in rows:
        flag = ""
        if cen is not None and cen <= 3 and adm is False:
            flag = "  <-- GAP (centroid knows, panel locks out)"
            gap.append((code, sig, cen))
        print(f"{code:18s} {str(sig):>5s} {str(cen):>5s} {str(adm):>9s}{flag}")

    print("\n" + "=" * 70)
    print(f"types where centroid self-rank <= 3 but the panel-signature gate excludes them: {len(gap)}")
    for code, sig, cen in sorted(gap, key=lambda r: r[1] or 999, reverse=True):
        print(f"   {code:18s} signature rank #{sig:<3} vs centroid rank #{cen}")
    return rows


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
