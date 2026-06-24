"""Tests for lineage-routed tumor expression decomposition + characterization."""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from trufflepig import expression_decomposition as ed
from trufflepig.expression_decomposition import (
    decompose_expression,
    resolve_mode,
    signature_score,
    restricted_marker_burden,
    bulk_aneuploidy_amplitude,
    _group_to_mode,
)


@lru_cache(maxsize=None)
def _sample(type_code):
    """{symbol: clean-TPM} mean profile for a representative cohort."""
    from pirlygenes.expression.accessors import representative_cohort_samples
    from oncoref.normalization import clean_tpm

    d = representative_cohort_samples(type_code).drop_duplicates("Ensembl_Gene_ID")
    cols = [c for c in d.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    gt = pd.DataFrame({"Ensembl_Gene_ID": d["Ensembl_Gene_ID"].values, "Symbol": d["Symbol"].values})
    clean = clean_tpm(d.set_index("Ensembl_Gene_ID")[cols].astype(float), gene_table=gt.set_index(d.index))
    clean.index = d["Symbol"].values
    return clean.groupby(level=0).sum().mean(axis=1).to_dict()


# ── routing (no heavy data) ─────────────────────────────────────────────

@pytest.mark.parametrize("hint,mode", [
    ("PRAD", "solid"), ("UCEC", "solid"), ("DLBC", "heme"), ("LAML", "heme"),
    ("SARC", "mesenchymal"), ("ATRT", "embryonal"),
    ("CRC", "solid"), ("CRC_MSI", "solid"), ("COAD_MSI", "solid"),   # group + molecular-subtype suffix strip
    ("mesenchymal", "mesenchymal"), ("heme", "heme"), ("melanoma", "solid"),
])
def test_resolve_mode_routes_family_type_group_subtype(hint, mode):
    assert resolve_mode(hint)[0] == mode


def test_group_to_mode():
    assert _group_to_mode("Heme") == "heme"
    assert _group_to_mode("Sarcoma") == "mesenchymal"
    assert _group_to_mode("Embryonal") == "embryonal"
    assert _group_to_mode("Epithelial") == "solid"
    assert _group_to_mode("Melanoma") == "solid"          # solid mode; compartment-specific sig applied separately


def test_signature_score_high_for_enriched_set():
    s = pd.Series({f"BG{i}": 1.0 for i in range(100)} | {"A": 900.0, "B": 800.0, "C": 700.0})
    assert signature_score(s, ["A", "B", "C"]) > 95
    assert np.isnan(signature_score(s, ["NOPE"]))


def test_restricted_marker_burden_flags_expressed_cta():
    sample = {f"BG{i}": 2.0 for i in range(300)}
    sample["MAGEA1"] = 1000.0                              # a germline-restricted CTA, expressed high
    out = restricted_marker_burden(sample)
    assert out["n_expressed"] >= 1
    assert out["max_percentile"] is not None and out["max_percentile"] > 95


# ── decomposition / characterization (real data) ────────────────────────

def test_routed_output_shape_and_residual_fraction():
    r = decompose_expression(_sample("COAD"), cancer="COAD")
    assert r["selected_mode"] == "solid"
    for k in ("lineage", "purity", "tumor_characteristics", "modes"):
        assert k in r
    rf = r["purity"]["residual_fraction"]
    assert 0.0 < rf <= 1.0
    assert r["lineage"]["compartment"] == "Epithelial"


def test_heme_sublineage_identified():
    r = decompose_expression(_sample("DLBC"), cancer="DLBC")
    assert r["selected_mode"] == "heme"
    assert r["tumor_characteristics"]["heme_sublineage"]["malignant_sublineage"] == "b_cell"


def test_routed_purity_beats_solid_assumption_for_heme():
    # the CLL/DLBC catastrophe: estimating purity under the solid assumption subtracts the tumor
    r = decompose_expression(_sample("DLBC"), cancer="DLBC", run_all=True)
    assert r["modes"]["heme"]["residual_fraction"] > r["modes"]["solid"]["residual_fraction"] + 0.2


def test_embryonal_mode_subtracts_immune_only():
    r = decompose_expression(_sample("ATRT"), cancer="ATRT")
    assert r["selected_mode"] == "embryonal"
    subtracted = set(r["modes"]["embryonal"]["subtracted"])
    assert "epithelial" not in subtracted and "stromal" not in subtracted   # kept (polyphenotypic tumor)


def test_compartment_aware_tumor_signature():
    assert decompose_expression(_sample("SKCM"), cancer="SKCM")["tumor_characteristics"]["tumor_lineage_signature"] == "melanoma"
    assert decompose_expression(_sample("GBM"), cancer="GBM")["tumor_characteristics"]["tumor_lineage_signature"] == "cns"


def test_residual_fraction_declines_with_contaminant_dilution():
    base = pd.Series(_sample("COAD"))
    from pirlygenes.expression.accessors import hpa_cell_type_expression
    hpa = hpa_cell_type_expression().drop_duplicates("Symbol").set_index("Symbol")
    immune = hpa[["T-cells", "B-cells", "Macrophages"]].mean(1)
    immune = (immune / immune.sum() * 1e6).reindex(base.index).fillna(0)
    pure = decompose_expression(base.to_dict(), cancer="COAD")["purity"]["residual_fraction"]
    diluted = ((base * 0.4 + immune * 0.6))
    diluted = (diluted / diluted.sum() * 1e6).to_dict()
    low = decompose_expression(diluted, cancer="COAD")["purity"]["residual_fraction"]
    assert low < pure                                      # residual_fraction is monotone with purity


# ── met templates: off by default, gated ────────────────────────────────

def test_met_templates_off_by_default_and_never_self_subtract_primary():
    coad_default = decompose_expression(_sample("COAD"), cancer="COAD")
    assert "liver" not in coad_default["modes"]["solid"]["subtracted"]            # off by default
    coad_met = decompose_expression(_sample("COAD"), cancer="COAD", met_sites=["liver"])
    assert "liver" in coad_met["modes"]["solid"]["subtracted"]                    # subtracted for a liver-met of colon ca
    lihc_met = decompose_expression(_sample("LIHC"), cancer="LIHC", met_sites=["liver"])
    assert "liver" not in lihc_met["modes"]["solid"]["subtracted"]                # skipped — liver IS the primary organ


# ── bulk vs residual aneuploidy + fallback safety + JSON validity ───────

def test_bulk_and_residual_aneuploidy_are_distinct_signals():
    r = decompose_expression(_sample("COAD"), cancer="COAD")
    bulk = r["purity"]["corroborators"]["aneuploidy_amplitude_bulk"]              # purity corroborator (bulk)
    residual = r["tumor_characteristics"]["aneuploidy_score"]                      # tumor characterization (residual)
    assert bulk is not None and residual is not None
    assert bulk != residual                                                       # computed on different inputs


def test_decompose_is_fallback_safe_when_aneuploidy_unavailable(monkeypatch):
    monkeypatch.setattr("trufflepig.aneuploidy_axis.gene_arm_map",
                        lambda: (_ for _ in ()).throw(ImportError("no genome")))
    r = decompose_expression(_sample("COAD"), cancer="COAD")                      # must NOT raise
    assert r["purity"]["corroborators"]["aneuploidy_amplitude_bulk"] is None
    assert r["tumor_characteristics"]["aneuploidy_score"] is None


def test_output_is_valid_json_no_nan():
    r = decompose_expression(_sample("DLBC"), cancer="DLBC")
    json.dumps(r["purity"], allow_nan=False)                                      # raises if any NaN
    json.dumps(r["tumor_characteristics"], allow_nan=False)


# ── wiring into estimate_tumor_purity ───────────────────────────────────

def _df(type_code):
    from pirlygenes.expression.accessors import representative_cohort_samples
    d = representative_cohort_samples(type_code).drop_duplicates("Ensembl_Gene_ID")
    cols = [c for c in d.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    return pd.DataFrame({"ensembl_gene_id": d["Ensembl_Gene_ID"].values,
                         "gene_symbol": d["Symbol"].values, "TPM": d[cols].mean(axis=1).values})


def test_estimate_tumor_purity_gates_estimate_for_heme_and_reports_decomposition():
    from trufflepig.tumor_purity import estimate_tumor_purity
    solid = estimate_tumor_purity(_df("COAD"), cancer_type="COAD")["components"]
    heme = estimate_tumor_purity(_df("DLBC"), cancer_type="DLBC")["components"]
    # ESTIMATE gated off where the tumor lineage IS a background (heme/sarcoma), kept for epithelial.
    assert solid["lineage_compartment"] == "Epithelial" and solid["estimate_gated_for_lineage"] is False
    assert heme["lineage_compartment"] == "Heme" and heme["estimate_gated_for_lineage"] is True
    # lineage-routed decomposition reported as a component (not fused into overall yet).
    for comp, mode in ((solid, "solid"), (heme, "heme")):
        dc = comp["decomposition"]
        assert dc["mode"] == mode
        assert 0.0 < dc["residual_fraction"] <= 1.0
