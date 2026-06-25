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
    """{symbol: TPM} mean profile for a representative cohort (raw; no oncoref). Covers rare types
    (ATRT/RT/…) absent from the pan-cancer table — routing/lineage tests are rank-based so raw is fine."""
    from pirlygenes.expression.accessors import representative_cohort_samples

    d = representative_cohort_samples(type_code).drop_duplicates("Symbol").set_index("Symbol")
    cols = [c for c in d.columns if c.startswith(type_code)]
    return d[cols].mean(axis=1).to_dict()


# ── routing (no heavy data) ─────────────────────────────────────────────

@pytest.mark.parametrize("hint,mode", [
    ("PRAD", "solid"), ("UCEC", "solid"), ("DLBC", "heme"), ("LAML", "heme"),
    ("SARC", "mesenchymal"), ("ATRT", "embryonal"),
    ("CRC", "solid"), ("CRC_MSI", "solid"), ("COAD_MSI", "solid"),   # group + molecular-subtype suffix strip
    ("mesenchymal", "mesenchymal"), ("heme", "heme"), ("melanoma", "solid"),
])
def test_resolve_mode_routes_family_type_group_subtype(hint, mode):
    assert resolve_mode(hint)[0] == mode


def test_lineage_fit_score_rewards_surviving_tumor_penalizes_leakage():
    # synthetic: residual rich in the mode's tumor lineage + poor in its backgrounds → high fit;
    # residual where the backgrounds leaked through (not subtracted) → low fit.
    from trufflepig.expression_decomposition import lineage_fit_score, _refs
    _, signatures = _refs()
    epi = list(signatures["epithelial"])[:5]
    imm = list(signatures["immune"])[:5]
    bg = {f"BG{i}": 10.0 for i in range(60)}
    good = pd.Series({**bg, **{g: 900.0 for g in epi}, **{g: 1.0 for g in imm}})   # epithelial tumor survived
    bad = pd.Series({**bg, **{g: 1.0 for g in epi}, **{g: 900.0 for g in imm}})    # immune leaked through
    assert lineage_fit_score("solid", good, signatures=signatures) > lineage_fit_score("solid", bad, signatures=signatures)


def test_lineage_fit_ranks_true_mode_solid_and_heme():
    # mode-comparable GoF: heme tumor must NOT fit solid (the ESTIMATE failure mode), epithelial must.
    heme = decompose_expression(_sample("DLBC"), cancer="DLBC", run_all=True)["modes"]
    assert heme["heme"]["lineage_fit"] > heme["solid"]["lineage_fit"]
    solid = decompose_expression(_sample("COAD"), cancer="COAD", run_all=True)["modes"]
    assert solid["solid"]["lineage_fit"] >= max(solid[m]["lineage_fit"] for m in ("heme", "mesenchymal", "embryonal"))


def test_config_self_consistency():
    # The mode/compartment/template/signature tables must stay mutually consistent (catches drift
    # when biology config is edited) — the "declarative table + validation" Codex asked for.
    from trufflepig.expression_decomposition import _MODES, _COMPARTMENTS, _SUBLINEAGES, _refs
    templates, signatures = _refs()
    for comp, c in _COMPARTMENTS.items():
        assert c["mode"] in _MODES, f"{comp} routes to unknown mode {c['mode']}"
        assert c["tumor_sig"] in signatures, f"{comp} tumor_sig {c['tumor_sig']} not a signature"
    for mode, m in _MODES.items():
        assert m["tumor_sig"] in signatures, f"{mode} tumor_sig {m['tumor_sig']} not a signature"
        for k in m["subtract"]:
            assert k in templates, f"{mode} subtracts {k} but no template exists"
    for sl in _SUBLINEAGES:                              # heme keeps the malignant sub-lineage, subtracts the rest
        assert sl in templates and sl in signatures, f"sub-lineage {sl} missing template/signature"


def test_group_to_mode():
    assert _group_to_mode("Heme") == "heme"
    assert _group_to_mode("Sarcoma") == "mesenchymal"
    assert _group_to_mode("Embryonal") == "embryonal"
    assert _group_to_mode("Epithelial") == "solid"
    assert _group_to_mode("Melanoma") == "solid"          # solid mode; compartment-specific sig applied separately


def test_signature_score_high_for_enriched_set():
    s = pd.Series({f"BG{i}": 1.0 for i in range(100)} | {"A": 900.0, "B": 800.0, "C": 700.0})
    assert signature_score(s, ["A", "B", "C"]) > 95
    assert signature_score(s, ["NOPE"]) is None              # not computable → None (never NaN), at the boundary


def test_decompose_mode_nnls_subtracts_only_non_tumor_synthetic(monkeypatch):
    # Fully synthetic templates/signatures — no real data. `solid` mode subtracts immune+stromal;
    # the epithelial tumor + proliferation must survive in the residual.
    import trufflepig.expression_decomposition as ed_mod
    genes = ["EPI1", "IMM1", "STR1", "PRO1"]

    def tmpl(on):
        s = pd.Series(0.0, index=genes); s[on] = 1.0
        return s / s.sum() * 1e6

    templates = {"immune": tmpl("IMM1"), "stromal": tmpl("STR1"), "epithelial": tmpl("EPI1")}
    signatures = {"immune": frozenset(["IMM1"]), "stromal": frozenset(["STR1"]),
                  "epithelial": frozenset(["EPI1"]), "proliferation": frozenset(["PRO1"]), "restricted": frozenset()}
    monkeypatch.setattr(ed_mod, "_refs", lambda: (templates, signatures))
    sample = pd.Series({"EPI1": 600.0, "IMM1": 200.0, "STR1": 200.0, "PRO1": 60.0})
    metrics, residual, _ = ed_mod.decompose_mode("solid", sample, templates, signatures, "percentile", None, None)
    assert metrics["residual_fraction"] > 0.0
    assert residual["EPI1"] > 0.0 and residual["PRO1"] > 0.0   # tumor + proliferation retained
    assert residual["IMM1"] < sample["IMM1"] and residual["STR1"] < sample["STR1"]  # backgrounds subtracted


def test_result_schema_keys():
    r = decompose_expression(_sample("COAD"), cancer="COAD")
    assert set(r) >= {"selected_mode", "lineage", "purity", "tumor_characteristics", "modes"}
    assert set(r["purity"]) >= {"residual_fraction", "corroborators", "consistency_flags"}
    assert set(r["lineage"]) >= {"compartment", "mode", "type_code"}


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


def test_sparse_panel_input_no_nan_and_none_scores():
    # P2: background fit genes present but NO tumor-signature genes — scores must be None (valid
    # JSON null), never NaN, and a missing score must not win the routing max().
    sparse = {f"FIB{i}": 50.0 for i in range(40)}
    sparse.update({"COL1A1": 900.0, "PTPRC": 800.0, "CD3D": 700.0})    # stromal + immune only
    r = decompose_expression(sparse, cancer="SARC")
    json.dumps(r, allow_nan=False)                                     # raises on any NaN
    assert r["tumor_characteristics"]["tumor_lineage_signature_score"] is None
    for m in r["modes"].values():
        v = m.get("tumor_lineage_in_residual")
        assert v is None or not (isinstance(v, float) and np.isnan(v))


def test_ambiguous_routing_uses_selected_mode_compartment(monkeypatch):
    # P2: when compartment_call is unconfident and the runner-up mode wins, lineage.compartment
    # must match the SELECTED mode — not the losing top compartment.
    monkeypatch.setattr("trufflepig.cancer_type_centroid.compartment_call",
                        lambda _: {"compartment": "Epithelial", "runner_up": "Heme",
                                   "confident": False, "margin": 0.001})
    r = decompose_expression(_sample("DLBC"))                          # no hint → ambiguous solid vs heme
    expected = {"heme": "Heme", "solid": "Epithelial"}[r["selected_mode"]]
    assert r["lineage"]["compartment"] == expected


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
        assert "aneuploidy_purity" in dc                                  # #96 calibrated signal present


def test_estimate_gated_by_expression_lineage_when_code_is_wrong():
    # On real samples the cancer-type CALL is often wrong at the fine level (a sarcoma miscalled BRCA).
    # ESTIMATE must still be gated off via the EXPRESSION lineage (compartment_call), not the code.
    from trufflepig.tumor_purity import estimate_tumor_purity
    comp = estimate_tumor_purity(_df("SARC"), cancer_type="BRCA")["components"]
    assert comp["lineage_compartment"] == "Epithelial"           # the (wrong) cancer-code lineage
    assert comp["expression_lineage_compartment"] == "Sarcoma"   # the right expression lineage
    assert comp["estimate_gated_for_lineage"] is True            # gated despite the epithelial code


def test_decomposition_lineage_override_only_for_auto_codes():
    # A sarcoma carrying a wrong epithelial code:
    #  - AUTO-detected code (allow_lineage_override=True) → expression+lineage_fit override to
    #    mesenchymal, conflict flagged, cross-lineage aneuploidy A_ref withheld.
    #  - EXPLICIT caller hint (allow_lineage_override=False) → trust the hint, NO override (so a correct
    #    explicit type — e.g. hepatoblastoma's hepatic expression — is never flipped).
    from trufflepig.tumor_purity import _decomposition_purity_component
    sym = _sample("SARC")
    auto = _decomposition_purity_component(sym, "BRCA", allow_lineage_override=True)
    assert auto["lineage_conflict"] is True and auto["mode"] == "mesenchymal"
    assert auto["code_lineage"] == "Epithelial" and auto["expression_lineage"] == "Sarcoma"
    assert auto["aneuploidy_purity"] is None         # withheld on conflict (per-type A_ref is cross-lineage)
    explicit = _decomposition_purity_component(sym, "BRCA", allow_lineage_override=False)
    assert explicit["mode"] == "solid" and explicit["lineage_conflict"] is False   # trusts the explicit code


def test_purity_reconciliation_guardrail():
    # #2: flag when the decomposition + ESTIMATE strongly disagree with overall_estimate (e.g. a cell
    # line read as ~1% pure); stay silent when they agree.
    from trufflepig.tumor_purity import _purity_reconciliation
    under = _purity_reconciliation(0.05, {"residual_fraction": 0.8}, 0.9)
    assert under and "UNDERESTIMATE" in under[0]
    over = _purity_reconciliation(0.9, {"residual_fraction": 0.2}, 0.2)
    assert over and "OVERESTIMATE" in over[0]
    assert _purity_reconciliation(0.6, {"residual_fraction": 0.6}, 0.6) == []   # agreement → no flag


def test_include_decomposition_flag_opts_out():
    from trufflepig.tumor_purity import estimate_tumor_purity
    on = estimate_tumor_purity(_df("COAD"), cancer_type="COAD")["components"]["decomposition"]
    off = estimate_tumor_purity(_df("COAD"), cancer_type="COAD", include_decomposition=False)["components"]["decomposition"]
    assert isinstance(on, dict) and on["mode"] == "solid"
    assert off is None                                                    # ranking loop uses this to skip the cost


def test_analyze_sample_computes_decomposition_once_for_winner():
    # candidate ranking opts out of the decomposition; the winner still gets it (computed once, not N×).
    import trufflepig.tumor_purity as tp
    orig, calls = tp._decomposition_purity_component, {"n": 0}
    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    tp._decomposition_purity_component = counting
    try:
        res = tp.analyze_sample(_df("COAD"), cancer_type="COAD")
    finally:
        tp._decomposition_purity_component = orig
    assert calls["n"] == 1                                                # once for the winner, not per candidate
    dc = (res["purity"].get("components") or {}).get("decomposition")
    assert isinstance(dc, dict) and dc.get("mode") == "solid"


def test_aneuploidy_purity_calibration():
    from trufflepig.purity_calibration import aneuploidy_reference, aneuploidy_purity
    aref = aneuploidy_reference("COAD")
    assert aref is not None and aref > 0                                  # aneuploid type → positive A_ref
    assert aneuploidy_reference("NOT_A_TYPE") is None                     # uncalibratable → None
    p = aneuploidy_purity(_sample("COAD"), "COAD")
    assert p is None or (0.0 <= p <= 1.0)                                 # calibrated purity in [0,1]
    assert aneuploidy_purity(_sample("COAD"), "NOT_A_TYPE") is None       # no reference → None (no signal)


def test_aneuploidy_purity_none_when_sample_has_no_aneuploidy():
    # bulk_amplitude 0.0 (near-diploid sample) → None, not a fake 0.0 ("0% pure") — aneuploidy uninformative.
    from trufflepig.purity_calibration import aneuploidy_purity
    assert aneuploidy_purity({}, "COAD", median_purity=0.6, bulk_amplitude=0.0) is None


def test_heme_immune_indeterminate_when_no_markers():
    # No b/t/myeloid/plasma markers present → can't tell which sub-lineage is malignant; subtract NONE
    # (never guess one and subtract the genuinely-tumor compartment).
    from trufflepig.expression_decomposition import _heme_immune, _refs
    _, signatures = _refs()
    info, healthy = _heme_immune(pd.Series({"SOMEGENE": 100.0}), signatures, "percentile", type_code=None)
    assert info["malignant_sublineage"] is None and info["source"] == "indeterminate"
    assert healthy == []


def test_unknown_met_organ_is_surfaced_not_silent():
    from trufflepig.expression_decomposition import _subtract_keys, _refs
    _, signatures = _refs()
    keys, info = _subtract_keys("solid", pd.Series({"EPCAM": 100.0}), signatures, "percentile", "COAD",
                                ["liver", "bone"])
    assert "bone" in info["met_sites_unknown"]               # unsupported organ surfaced, not dropped
    assert "liver" in info["met_sites_subtracted"]           # COAD is not a liver primary → subtracted


def test_heme_subtype_resolves_to_parent_malignant_sublineage():
    # LAML_APL subtype must map to LAML's myeloid sub-lineage (parent walk), NOT fall to marker
    # dominance and risk subtracting the myeloid (tumor) compartment as healthy background.
    from trufflepig.expression_decomposition import _heme_immune, _refs
    _, signatures = _refs()
    info, healthy = _heme_immune(pd.Series({"GENE": 1.0}), signatures, "percentile", type_code="LAML_APL")
    assert info["malignant_sublineage"] == "myeloid" and info["source"] == "type-map"
    assert "myeloid" not in healthy                          # the tumor sub-lineage is kept, not subtracted


def test_met_primary_subtype_skips_self_organ():
    # LUAD_EGFR primary must not self-subtract the lung template (parent LUAD ∈ lung primaries).
    from trufflepig.expression_decomposition import _subtract_keys, _refs
    _, signatures = _refs()
    keys, info = _subtract_keys("solid", pd.Series({"EPCAM": 1.0}), signatures, "percentile", "LUAD_EGFR",
                                ["lung"])
    assert "lung" in info["met_sites_skipped_as_primary"]
    assert "lung" not in keys


def test_met_sites_rejects_non_organ_template_keys():
    # met_sites must accept only true met ORGANS — a non-organ template key like 'immune' must NOT be
    # treated as a met (it would duplicate the mode's own subtraction key and corrupt the fit).
    from trufflepig.expression_decomposition import _subtract_keys, _refs
    _, signatures = _refs()
    keys, info = _subtract_keys("solid", pd.Series({"EPCAM": 1.0}), signatures, "percentile", "COAD",
                                ["immune", "liver"])
    assert "immune" in info["met_sites_unknown"]      # rejected as a non-organ, not subtracted as a met
    assert "liver" in info["met_sites_subtracted"]
    assert keys.count("immune") == 1                  # the mode's own immune key, not duplicated


def test_met_bookkeeping_surfaced_in_decompose_output():
    # the met info must be visible in the decompose_expression result, not just internal to _subtract_keys
    r = decompose_expression(_sample("LUAD"), cancer="LUAD_EGFR", met_sites=["lung", "bogus"])
    m = r["modes"]["solid"]
    assert m["met_sites_skipped_as_primary"] == ["lung"]      # LUAD_EGFR subtype → lung self-subtract guard
    assert m["met_sites_unknown"] == ["bogus"]


def test_signature_score_ssgsea_space():
    s = pd.Series({f"BG{i}": 1.0 for i in range(100)} | {"A": 900.0, "B": 800.0, "C": 700.0})
    sc = signature_score(s, ["A", "B", "C"], space="ssgsea")
    assert sc is not None and isinstance(sc, float)
    assert signature_score(s, ["NOPE"], space="ssgsea") is None   # not computable → None in every space


def test_aneuploidy_purity_declines_with_dilution():
    # Recovery: mixing a tumor toward the diploid baseline (simulated contamination) lowers bulk
    # aneuploidy → lower calibrated purity. Pins direction, not just shape.
    from trufflepig.purity_calibration import aneuploidy_purity
    from trufflepig.aneuploidy_axis import _diploid_reference
    base = _sample("COAD")
    ref = _diploid_reference()
    diluted = {k: 0.4 * v + 0.6 * float(ref.get(k, 0.0)) for k, v in base.items()}
    p_pure = aneuploidy_purity(base, "COAD", median_purity=0.6)
    p_dil = aneuploidy_purity(diluted, "COAD", median_purity=0.6)
    assert p_pure is not None and p_dil is not None
    assert p_dil < p_pure
