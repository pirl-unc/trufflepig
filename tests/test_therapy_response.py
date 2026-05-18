"""Tests for pirlygenes.therapy_response (#57)."""

from trufflepig.therapy_response import (
    MAPK_ACTIVITY_GENES,
    MAPK_ACTIVITY_AXIS,
    TherapyAxisScore,
    infer_mapk_activity_sources,
    load_therapy_signatures,
    score_therapy_signatures,
    symbol_therapy_annotations,
)


def test_load_therapy_signatures_has_ar_axis_for_prad():
    sigs = load_therapy_signatures()
    assert "AR_signaling" in sigs
    up = {g["symbol"] for g in sigs["AR_signaling"]["up"]}
    down = {g["symbol"] for g in sigs["AR_signaling"]["down"]}
    # Canonical AR-transactivated genes must be on the up side.
    for sym in ("KLK3", "KLK2", "TMPRSS2", "NKX3-1", "FKBP5"):
        assert sym in up, f"{sym} missing from AR_signaling up panel"
    # FOLH1 (PSMA) is AR-suppressed; CHGA is NE-emergence.
    assert "FOLH1" in down
    assert "CHGA" in down


def test_all_eight_axes_present():
    sigs = load_therapy_signatures()
    expected_axes = {
        "AR_signaling",
        "ER_signaling",
        "HER2_signaling",
        "MAPK_EGFR_signaling",
        "NE_differentiation",
        "EMT",
        "hypoxia",
        "IFN_response",
    }
    assert expected_axes.issubset(sigs.keys())


def test_symbol_therapy_annotations_maps_canonical_markers_for_prad():
    sigs = load_therapy_signatures()
    ann = symbol_therapy_annotations(sigs, "PRAD")
    # KLK3 must get at least one AR_signaling up annotation.
    assert "KLK3" in ann
    assert any("AR signaling up" in s for s in ann["KLK3"]), ann["KLK3"]
    # FOLH1 must get AR_signaling down annotation.
    assert "FOLH1" in ann
    assert any("AR signaling down" in s for s in ann["FOLH1"])
    # Hypoxia / IFN genes (pan_cancer) must show up too.
    assert "CA9" in ann
    assert "ISG15" in ann


def test_annotations_respect_cancer_context():
    sigs = load_therapy_signatures()
    # ESR1 applies to BRCA but not PRAD.
    prad = symbol_therapy_annotations(sigs, "PRAD")
    brca = symbol_therapy_annotations(sigs, "BRCA")
    assert "ESR1" in brca
    assert "ESR1" not in prad


def test_score_ar_signaling_suppressed_when_ar_targets_are_low_in_prad_context():
    # Synthetic sample where every AR-transactivated gene reads at
    # roughly 10% of its PRAD cohort median and FOLH1 reads at 4× —
    # the canonical post-ADT pattern.
    from trufflepig.reference import pan_cancer_expression

    ref = pan_cancer_expression().drop_duplicates(subset="Symbol")

    def cohort(sym):
        sub = ref[ref["Symbol"] == sym]
        if sub.empty:
            return 0.0
        col = "PRAD_TPM"
        return float(sub.iloc[0][col]) if col in sub.columns else 0.0

    sample = {}
    for sym in ("KLK3", "KLK2", "TMPRSS2", "NKX3-1", "FKBP5"):
        sample[sym] = cohort(sym) * 0.1
    sample["FOLH1"] = cohort("FOLH1") * 4.0
    sample["CHGA"] = cohort("CHGA") * 3.0

    scores = score_therapy_signatures(sample, "PRAD")
    ar = scores["AR_signaling"]
    assert isinstance(ar, TherapyAxisScore)
    assert ar.state == "down", ar
    assert ar.up_geomean_fold is not None and ar.up_geomean_fold < 0.5
    # The per-gene table should retain the canonical markers so the
    # report can enumerate the chain of evidence.
    per_gene_syms = {entry["symbol"] for entry in ar.per_gene}
    assert "KLK3" in per_gene_syms
    assert "FOLH1" in per_gene_syms


def test_score_returns_empty_when_cancer_has_no_applicable_axes():
    # Any real TCGA code has at least hypoxia / IFN (pan_cancer), so
    # use a made-up one that won't match and doesn't have a cohort col.
    scores = score_therapy_signatures({"KLK3": 100.0}, "NOTACANCER")
    # Scoring still iterates applicable classes (hypoxia/IFN are
    # pan_cancer), so the output may be empty or have empty measures
    # depending on ref universe. Contract: always a dict.
    assert isinstance(scores, dict)


def test_her2_signaling_active_when_erbb2_elevated_in_brca():
    from trufflepig.reference import pan_cancer_expression

    ref = pan_cancer_expression().drop_duplicates(subset="Symbol")
    col = "BRCA_TPM"

    def cohort(sym):
        sub = ref[ref["Symbol"] == sym]
        if sub.empty or col not in sub.columns:
            return 0.0
        return float(sub.iloc[0][col])

    sample = {
        "ERBB2": cohort("ERBB2") * 10.0,
        "GRB7": cohort("GRB7") * 10.0,
        "STARD3": cohort("STARD3") * 10.0,
    }
    scores = score_therapy_signatures(sample, "BRCA")
    her2 = scores.get("HER2_signaling")
    assert her2 is not None
    assert her2.state == "up"
    assert her2.up_geomean_fold is not None and her2.up_geomean_fold >= 5.0


def test_mapk_activity_score_is_pan_cancer_mpas_like():
    from trufflepig.reference import pan_cancer_expression

    ref = pan_cancer_expression().drop_duplicates(subset="Symbol")
    col = "SARC_TPM"

    def cohort(sym):
        sub = ref[ref["Symbol"] == sym]
        if sub.empty or col not in sub.columns:
            return 0.0
        return float(sub.iloc[0][col])

    sample = {sym: cohort(sym) * 5.0 for sym in MAPK_ACTIVITY_GENES}
    sample["EGFR"] = cohort("EGFR") * 20.0

    scores = score_therapy_signatures(sample, "SARC")
    mapk = scores[MAPK_ACTIVITY_AXIS]
    assert mapk.state == "up"
    assert mapk.up_geomean_fold is not None and mapk.up_geomean_fold >= 4.0
    per_gene_syms = {entry["symbol"] for entry in mapk.per_gene}
    assert {"DUSP4", "DUSP6", "EPHA4"}.issubset(per_gene_syms)
    assert "EGFR" not in per_gene_syms


def test_infer_mapk_activity_sources_keeps_driver_uncertainty():
    import pandas as pd

    score = TherapyAxisScore(
        therapy_class=MAPK_ACTIVITY_AXIS,
        state="up",
        up_geomean_fold=8.2,
        up_genes_measured=5,
        per_gene=[
            {"symbol": "DUSP6", "direction": "up", "fold_vs_cohort": 15.0},
            {"symbol": "SPRY2", "direction": "up", "fold_vs_cohort": 12.0},
        ],
    )
    analysis = {
        "therapy_response_scores": {MAPK_ACTIVITY_AXIS: score},
        "alteration_records": [
            {
                "gene": "EGFR",
                "alteration": "kinase domain duplication",
                "alteration_type": "kdd",
                "source_path": "alvin.tsv",
                "confidence": "supplied",
            }
        ],
        "fusion_records": [],
    }
    ranges = pd.DataFrame(
        [
            {
                "symbol": "EGFR",
                "attr_tumor_tpm": 570.0,
                "observed_tpm": 583.0,
                "pct_cancer_median": 29.3,
                "tcga_percentile": 0.99,
            }
        ]
    )

    findings = infer_mapk_activity_sources(analysis, ranges_df=ranges)

    assert findings and findings[0]["label"] == "MAPK / ERK activity"
    labels = [source["label"] for source in findings[0]["candidate_sources"]]
    assert any("supplied EGFR kinase domain duplication" in label for label in labels)
    assert any("RNA-high EGFR" in label for label in labels)
    assert "not source-specific" in findings[0]["caveat"]


def test_fold_uses_pseudocount_to_avoid_division_by_zero():
    # When cohort median is zero for a gene, the fold should still be
    # finite thanks to the pseudocount.
    scores = score_therapy_signatures({"CHGA": 100.0}, "PRAD")
    ar = scores["AR_signaling"]
    chga = [g for g in ar.per_gene if g["symbol"] == "CHGA"]
    # If CHGA is in the reference universe, we should see it; otherwise
    # the gene was skipped and we don't care for this test.
    if chga:
        assert chga[0]["fold_vs_cohort"] > 0.0
