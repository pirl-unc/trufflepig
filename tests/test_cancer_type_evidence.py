import pandas as pd


def _analysis(*rows):
    return {
        "cancer_type": rows[0][0],
        "candidate_trace": [
            {"code": code, "support_norm": support}
            for code, support in rows
        ],
    }


def _candidate_analysis(rows):
    return {
        "cancer_type": rows[0]["code"],
        "candidate_trace": rows,
    }


def _empty_expression_frame():
    return pd.DataFrame(columns=["ensembl_gene_id", "canonical_gene_name", "TPM"])


def _expression_frame(tpm_by_symbol):
    from trufflepig.reference import pan_cancer_expression

    ref = pan_cancer_expression()[["Ensembl_Gene_ID", "Symbol"]].drop_duplicates(
        "Symbol"
    )
    id_by_symbol = dict(zip(ref["Symbol"], ref["Ensembl_Gene_ID"]))
    rows = []
    for symbol, tpm in tpm_by_symbol.items():
        ensg = id_by_symbol.get(symbol)
        if not ensg:
            continue
        rows.append(
            {
                "ensembl_gene_id": ensg,
                "canonical_gene_name": symbol,
                "TPM": float(tpm),
            }
        )
    return pd.DataFrame(rows)


def test_nutm_rna_surrogate_promotes_in_squamous_context():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    finding = {
        "cancer_type": "NUTM",
        "rule_id": "nutm_nutm1",
        "surrogate": "NUTM1",
        "surrogate_tpm": 6.2,
        "threshold_tpm": 1.0,
        "support_genes": [],
    }

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("LUSC", 1.0), ("HNSC", 0.7)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "NUTM"
    assert result["selected"]["evidence_sources"] == ["rare_marker"]
    assert result["selected"]["can_select_report_label"] is True
    assert result["selected"]["metrics"]["rna_marker_support"] == 1.0


def test_broad_context_is_part_of_unified_evidence_view():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("READ", 1.0), ("COAD", 0.4)),
    )

    assert result["selected"] is None
    assert [row["cancer_type"] for row in result["evidence"]] == ["READ", "COAD"]
    assert result["evidence"][0]["evidence_sources"] == ["broad_rna"]
    assert result["evidence"][0]["metrics"]["broad_rna_support"] == 1.0
    assert result["evidence"][0]["report_label_candidate"] is False


def test_background_like_top_label_can_yield_to_supported_tumor_label():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _candidate_analysis(
            [
                {
                    "code": "SARC",
                    "support_norm": 1.0,
                    "signature_score": 0.82,
                    "family_label": "MESENCHYMAL",
                    "family_score": 19.0,
                },
                {
                    "code": "COAD",
                    "support_norm": 0.80,
                    "signature_score": 0.79,
                    "family_label": "CRC",
                    "family_score": 1.0,
                },
            ]
        ),
    )

    assert result["selected"]["cancer_type"] == "COAD"
    assert result["selected"]["reference_cancer_type"] == "COAD"
    assert result["selected"]["evidence_sources"] == [
        "broad_rna",
        "tumor_label_refinement",
    ]
    assert result["selected"]["label_decision"]["status"] == "selected"
    assert result["selected"]["metrics"]["family_marker_support"] == 1.0
    assert result["selected"]["competing_background_code"] == "SARC"


def test_background_like_top_label_does_not_yield_to_weak_tumor_label():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _candidate_analysis(
            [
                {
                    "code": "SARC",
                    "support_norm": 1.0,
                    "signature_score": 0.66,
                    "family_label": "MESENCHYMAL",
                    "family_score": 7.0,
                },
                {
                    "code": "LUAD",
                    "support_norm": 0.40,
                    "signature_score": 0.81,
                    "family_label": "",
                    "family_score": None,
                },
            ]
        ),
    )

    assert result["selected"] is None


def test_local_expression_reference_can_select_future_exact_cohort(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda: {
            "MM": {
                "markers": ("CD79A", "MS4A1", "MZB1"),
                "ref_medians": {"CD79A": 120.0, "MS4A1": 80.0, "MZB1": 100.0},
                "context_codes": ("DLBC", "LAML", "THYM"),
                "family": "heme-plasma",
                "primary_tissue": "bone_marrow",
                "source_cohort": "MMRF_COMMPASS",
            }
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"CD79A": 45.0, "MS4A1": 30.0, "MZB1": 40.0}),
        _analysis(("DLBC", 1.0), ("LAML", 0.4)),
    )

    assert result["selected"]["cancer_type"] == "MM"
    assert result["selected"]["reference_cancer_type"] == "DLBC"
    assert result["selected"]["selected_by"] == "local_expression_reference"
    assert result["selected"]["metrics"]["fine_reference_support"] >= 0.65


def test_local_expression_reference_requires_top_compatible_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda: {
            "MM": {
                "markers": ("CD79A", "MS4A1", "MZB1"),
                "ref_medians": {"CD79A": 120.0, "MS4A1": 80.0, "MZB1": 100.0},
                "context_codes": ("DLBC", "LAML", "THYM"),
                "family": "heme-plasma",
                "primary_tissue": "bone_marrow",
                "source_cohort": "MMRF_COMMPASS",
            }
        },
    )

    result = select_report_scope_from_evidence(
        _expression_frame({"CD79A": 45.0, "MS4A1": 30.0, "MZB1": 40.0}),
        _analysis(("PRAD", 1.0), ("DLBC", 0.9)),
    )

    assert result["selected"] is None
    mm = next(row for row in result["evidence"] if row["cancer_type"] == "MM")
    assert mm["can_select_report_label"] is False
    assert any("not the top compatible context" in r for r in mm["blocking_reasons"])


def test_fine_reference_medians_filter_to_requested_cancer_code(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence

    df = pd.DataFrame(
        [
            {"symbol": "RUNX2", "cancer_code": "OS", "tumor_tpm_median": 100.0},
            {"symbol": "RUNX2", "cancer_code": "CHOR", "tumor_tpm_median": 2.0},
            {"symbol": "TBXT", "cancer_code": "CHOR", "tumor_tpm_median": 50.0},
        ]
    )
    monkeypatch.setattr(
        "trufflepig.reference.subtype_deconvolved_expression",
        lambda: df,
    )
    evidence._reference_medians.cache_clear()

    try:
        assert evidence._reference_medians("OS") == {"RUNX2": 100.0}
    finally:
        evidence._reference_medians.cache_clear()


def test_salivary_surrogate_does_not_override_crc_context():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    finding = {
        "cancer_type": "ADCC",
        "rule_id": "adcc_myb",
        "surrogate": "MYB",
        "surrogate_tpm": 60.0,
        "threshold_tpm": 10.0,
        "support_genes": ["KIT"],
    }

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("READ", 1.0), ("COAD", 0.4), ("HNSC", 0.15)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"] is None
    adcc = result["evidence"][0]
    assert adcc["cancer_type"] == "ADCC"
    assert adcc["can_select_report_label"] is False
    assert "broad RNA context is not one of HNSC" in adcc["blocking_reasons"]


def test_direct_rare_rna_scope_requires_top_context_match():
    from trufflepig.rare_inference import infer_rare_cancer_report_scope_from_rna

    df = _expression_frame({"MYB": 60.0, "KIT": 40.0, "KRT7": 10.0})

    result = infer_rare_cancer_report_scope_from_rna(
        df,
        _analysis(("READ", 1.0), ("COAD", 0.4), ("HNSC", 0.15)),
    )

    assert result is None


def test_marker_prompt_only_rules_never_set_report_scope():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    finding = {
        "cancer_type": "CHOR",
        "rule_id": "chor_tbxt",
        "surrogate": "TBXT",
        "surrogate_tpm": 50.0,
        "threshold_tpm": 5.0,
        "support_genes": ["KRT19", "SOX9"],
    }

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("SARC", 1.0)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"] is None
    chor = next(row for row in result["evidence"] if row["cancer_type"] == "CHOR")
    assert chor["rule_promotes_report_scope"] is False
    assert chor["can_select_report_label"] is False


def test_osteogenic_reference_evidence_promotes_os_over_broad_sarc():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    df = _expression_frame(
        {
            "RUNX2": 520,
            "SATB2": 480,
            "IBSP": 1400,
            "SPP1": 1500,
            "COL1A1": 7600,
            "COL1A2": 12400,
            "DLX5": 230,
            "DMP1": 90,
            "MEPE": 190,
            "MDM2": 1180,
            "CDK4": 110,
            "FRS2": 110,
        }
    )

    result = select_report_scope_from_evidence(
        df,
        _analysis(("SARC", 1.0), ("UCS", 0.3)),
    )

    assert result["selected"]["cancer_type"] == "OS"
    assert result["selected"]["reference_cancer_type"] == "SARC"
    assert result["selected"]["evidence_sources"] == ["fine_reference"]
    assert result["selected"]["metrics"]["related_context_support"] == 1.0
    assert result["selected"]["metrics"]["fine_reference_support"] >= 0.7


def test_mdm2_amp_without_osteogenic_program_stays_broad_sarc():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    df = _expression_frame(
        {
            "RUNX2": 25,
            "SATB2": 20,
            "IBSP": 1,
            "SPP1": 60,
            "COL1A1": 900,
            "COL1A2": 1500,
            "MDM2": 250,
            "CDK4": 190,
            "FRS2": 70,
        }
    )

    result = select_report_scope_from_evidence(
        df,
        _analysis(("SARC", 1.0), ("UCS", 0.3)),
    )

    assert result["selected"] is None
    os_evidence = result["evidence"][0]
    assert os_evidence["cancer_type"] == "OS"
    assert os_evidence["can_select_report_label"] is False
    assert os_evidence["blocking_reasons"]
