#!/usr/bin/env python3
"""Per-SAMPLE no-hint cancer-type calls across every representative cohort (~565 individual samples),
and the confusion between the truth type and the FINAL call — i.e. what we ACTUALLY call each sample
after the full pipeline (bulk classifier → evidence selector → lineage veto), reported at the ENTITY
level (not just lineage).

Match levels per sample:
  exact     — final call == truth type
  subtype   — parent↔child (e.g. SCLC vs SCLC_ASCL1, COAD vs CRC)
  family    — same registry family (e.g. COAD vs READ)
  lineage   — same lineage mode only (solid/mesenchymal/heme/embryonal), different entity
  miss      — different lineage

Run:  python3 scripts/eval_per_sample_confusion.py
"""
import collections
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from oncoref.normalization import clean_tpm
from pirlygenes.expression.accessors import (
    available_representative_cohorts,
    representative_cohort_samples,
)
from pirlygenes.gene_sets_cancer import cancer_lineage_group, cancer_type_registry
from trufflepig.cancer_type_evidence import _primary_tissue_key_for_code
from trufflepig.expression_decomposition import _group_to_mode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_nohint_validation import full_granularity_call  # noqa: E402

_REGISTRY = {r["code"]: r for r in cancer_type_registry().to_dict("records")}
_NAME = {code: str(row.get("name") or code) for code, row in _REGISTRY.items()}
_REG_UPPER = {code.upper(): code for code in _REGISTRY}


def name(code):
    """Full cancer-type name for a code (e.g. ACC → Adrenocortical Carcinoma)."""
    return _NAME.get(_canon(code), code)


def _canon(code):
    """Resolve a call to its registry code: case-insensitive, and for a molecular subtype not in the
    registry (e.g. LUAD_KRAS_STK11 from the subtype-signature layer) fall back to the longest
    registry-code prefix (→ LUAD). Returns the code unchanged if nothing matches."""
    if not code:
        return ""
    upper = str(code).upper()
    if upper in _REG_UPPER:
        return _REG_UPPER[upper]
    parts = upper.split("_")
    for i in range(len(parts) - 1, 0, -1):
        candidate = "_".join(parts[:i])
        if candidate in _REG_UPPER:
            return _REG_UPPER[candidate]
    return code


def _lineage(code):
    group = cancer_lineage_group(code) if code else None
    return _group_to_mode(group) if group else None


def _parent(code):
    value = str(_REGISTRY.get(code, {}).get("parent_code") or "").strip().upper()
    return "" if value in ("", "NAN", "NONE") else value


def _organ(code):
    try:
        return _primary_tissue_key_for_code(code) or ""
    except (KeyError, ValueError):
        return ""


def match_level(call, truth):
    """exact > subtype (parent↔child) > sibling (shared parent, e.g. COAD/READ → CRC) >
    lineage (same compartment: solid/mesenchymal/heme/embryonal, different entity, e.g. SKCM→GBM) >
    organ (SAME primary site but DIFFERENT compartment — a co-located cross-lineage confusion, e.g. a
    uterine sarcoma vs endometrial carcinoma; the 'less bad' cross-compartment mixup) > miss.

    NOT using the registry ``family`` (too broad: 'carcinoma-gi' lumps colon+stomach), so COAD→READ
    scores 'sibling' while COAD→STAD scores 'lineage'. Note: generic SARC_* codes have no organ, so a
    sarcoma↔carcinoma mixup only scores 'organ' when the sarcoma code is site-specific (e.g. UCS)."""
    if not call:
        return "miss"
    cc, ct = _canon(call), _canon(truth)
    if cc.upper() == ct.upper():                              # same registry entity (case-insensitive)
        return "exact" if str(call).upper() == str(truth).upper() else "subtype"
    if _parent(cc) == ct.upper() or _parent(ct) == cc.upper():
        return "subtype"
    pc, pt = _parent(cc), _parent(ct)
    if pc and pc == pt:
        return "sibling"
    if _lineage(cc) and _lineage(cc) == _lineage(ct):
        return "lineage"
    oc, ot = _organ(cc), _organ(ct)
    if oc and oc == ot:
        return "organ"
    return "miss"


# data/format failures we tolerate per sample (anything else propagates so genuine bugs surface)
EXPECTED_SAMPLE_ERRORS = (ValueError, KeyError, FloatingPointError)


def _clean_cohort(type_code):
    """Return (ensembl_ids, symbols, cleaned_dataframe[sample_columns]) for a cohort."""
    raw = representative_cohort_samples(type_code).drop_duplicates("Ensembl_Gene_ID")
    sample_cols = [c for c in raw.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    gene_table = pd.DataFrame(
        {"Ensembl_Gene_ID": raw["Ensembl_Gene_ID"].values, "Symbol": raw["Symbol"].values}
    )
    cleaned = clean_tpm(
        raw.set_index("Ensembl_Gene_ID")[sample_cols].astype(float),
        gene_table=gene_table.set_index(raw.index),
    )
    return raw["Ensembl_Gene_ID"].values, raw["Symbol"].values, cleaned, sample_cols


def call_each_sample(type_code):
    """Yield (sample_column, final_call) for every individual sample in a cohort."""
    ensembl, symbols, cleaned, sample_cols = _clean_cohort(type_code)
    for col in sample_cols:
        df = pd.DataFrame(
            {"ensembl_gene_id": ensembl, "gene_symbol": symbols, "TPM": cleaned[col].values}
        )
        try:
            _bulk, final_call = full_granularity_call(df)
        except EXPECTED_SAMPLE_ERRORS as exc:
            final_call = f"ERROR:{str(exc)[:30]}"
        yield col, final_call


def main():
    types = sorted(available_representative_cohorts())
    results = []  # (truth_type, sample, final_call, match_level)
    for i, truth in enumerate(types, 1):
        calls = list(call_each_sample(truth))
        for sample, final_call in calls:
            results.append((truth, sample, final_call, match_level(final_call, truth)))
        print(f"  [{i}/{len(types)}] {truth:16s} -> {[c for _, c in calls]}", flush=True)

    levels = collections.Counter(r[3] for r in results)
    n = len(results)
    compatible = levels["exact"] + levels["subtype"] + levels["sibling"]
    print(f"\n==== {n} samples across {len(types)} types ====")
    for level in ("exact", "subtype", "sibling", "lineage", "miss"):
        print(f"  {level:8s}: {levels[level]:4d}  ({100*levels[level]/n:.0f}%)")
    print(f"  entity-correct (exact+subtype+sibling): {compatible}/{n} ({100*compatible/n:.0f}%)")
    print(f"  lineage-correct (all but miss): {n-levels['miss']}/{n} ({100*(n-levels['miss'])/n:.0f}%)")

    # confusion: per truth type, the distribution of final calls (only types with any non-exact call)
    by_type = collections.defaultdict(collections.Counter)
    for truth, _s, call, _lvl in results:
        by_type[truth][call] += 1
    print("\n==== per-type calls (types with ANY non-exact call) ====")
    print(f"{'truth':16s} {'n':>2s}  calls (count)")
    for truth in types:
        dist = by_type[truth]
        if list(dist) == [truth]:
            continue  # all exact, skip
        total = sum(dist.values())
        summary = "  ".join(f"{c}×{k}" for c, k in
                            sorted(((v, k) for k, v in dist.items()), reverse=True))
        print(f"{truth:16s} {total:>2d}  {summary}")

    # the mixup pairs (truth -> wrong call at miss/lineage level), most frequent first
    mixups = collections.Counter()
    for truth, _s, call, lvl in results:
        if lvl in ("lineage", "miss") and not str(call).startswith("ERROR"):
            mixups[(truth, call, lvl)] += 1
    print("\n==== cross-entity mixups (lineage-only or miss), most frequent ====")
    for (truth, call, lvl), cnt in mixups.most_common(30):
        print(f"  {cnt}×  {truth:16s} -> {call:16s} [{lvl}]")
    return results


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
