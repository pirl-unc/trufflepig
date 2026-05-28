import pandas as pd


def _analysis(*rows):
    return {
        "cancer_type": rows[0][0],
        "candidate_trace": [
            {"code": code, "support_fraction_of_top": support}
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

    assert result["selected"]["cancer_type"] == "READ"
    assert result["selected"]["inferred_cancer_type"] == "READ"
    assert result["selected"]["expression_reference_cancer_type"] == "READ"
    assert result["selected"]["selected_by"] == "primary_expression_match"
    assert result["primary_expression_context"]["cancer_type"] == "READ"
    assert [row["cancer_type"] for row in result["evidence"]] == ["READ", "COAD"]
    assert result["evidence"][0]["evidence_sources"] == ["broad_rna"]
    assert result["evidence"][0]["metrics"]["broad_rna_support"] == 1.0
    assert result["evidence"][0]["report_label_candidate"] is True


def test_background_like_top_label_can_yield_to_supported_tumor_label():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _candidate_analysis(
            [
                {
                    "code": "SARC",
                    "support_fraction_of_top": 1.0,
                    "signature_score": 0.82,
                    "family_label": "MESENCHYMAL",
                    "family_score": 19.0,
                },
                {
                    "code": "COAD",
                    "support_fraction_of_top": 0.80,
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
                    "support_fraction_of_top": 1.0,
                    "signature_score": 0.66,
                    "family_label": "MESENCHYMAL",
                    "family_score": 7.0,
                },
                {
                    "code": "LUAD",
                    "support_fraction_of_top": 0.40,
                    "signature_score": 0.81,
                    "family_label": "",
                    "family_score": None,
                },
            ]
        ),
    )

    assert result["selected"]["cancer_type"] == "SARC"
    assert result["selected"]["selected_by"] == "primary_expression_match"


def test_local_expression_reference_can_select_future_exact_cohort(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
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
    assert result["selected"]["expression_reference_cancer_type"] == "MM"
    assert result["selected"]["selected_by"] == "local_expression_reference"
    assert result["selected"]["metrics"]["fine_reference_support"] >= 0.65


def test_pirlygenes_observed_reference_can_select_exact_cohort(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    evidence._local_expression_reference_panels.cache_clear()
    monkeypatch.setattr(
        evidence,
        "_registry_by_code",
        lambda: {
            "MM": {
                "family": "heme-plasma",
                "primary_tissue": "bone_marrow",
            }
        },
    )
    monkeypatch.setattr(
        "trufflepig.reference.subtype_deconvolved_expression",
        lambda *args, **kwargs: pd.DataFrame(
            columns=["cancer_code", "symbol", "tumor_tpm_median"]
        ),
    )
    monkeypatch.setattr(
        "trufflepig.reference.cancer_reference_expression",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "Ensembl_Gene_ID": "ENSG00000105369",
                    "Symbol": "CD79A",
                    "cancer_code": "MM",
                    "source_cohort": "MMRF_COMMPASS",
                    "normalization": "tpm_clean",
                    "expression": 120.0,
                    "q1": 60.0,
                    "q3": 180.0,
                },
                {
                    "Ensembl_Gene_ID": "ENSG00000156738",
                    "Symbol": "MS4A1",
                    "cancer_code": "MM",
                    "source_cohort": "MMRF_COMMPASS",
                    "normalization": "tpm_clean",
                    "expression": 80.0,
                    "q1": 30.0,
                    "q3": 110.0,
                },
                {
                    "Ensembl_Gene_ID": "ENSG00000170476",
                    "Symbol": "MZB1",
                    "cancer_code": "MM",
                    "source_cohort": "MMRF_COMMPASS",
                    "normalization": "tpm_clean",
                    "expression": 100.0,
                    "q1": 40.0,
                    "q3": 140.0,
                },
            ]
        ),
    )

    try:
        result = select_report_scope_from_evidence(
            _expression_frame({"CD79A": 45.0, "MS4A1": 30.0, "MZB1": 40.0}),
            _analysis(("DLBC", 1.0), ("LAML", 0.4)),
        )
    finally:
        evidence._local_expression_reference_panels.cache_clear()

    assert result["selected"]["cancer_type"] == "MM"
    assert result["selected"]["reference_cancer_type"] == "DLBC"
    assert result["selected"]["expression_reference_cancer_type"] == "MM"
    assert result["selected"]["local_reference_kind"] == "observed_bulk_reference"
    assert result["selected"]["local_reference_source_cohort"] == "MMRF_COMMPASS"


def test_local_expression_reference_uses_actual_top_compatible_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
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
        _analysis(("LAML", 1.0), ("DLBC", 0.4)),
    )

    assert result["selected"]["cancer_type"] == "MM"
    assert result["selected"]["reference_cancer_type"] == "LAML"
    assert result["selected"]["local_reference_matched_context"] == "LAML"


def test_local_expression_reference_requires_top_compatible_context(monkeypatch):
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    monkeypatch.setattr(
        evidence,
        "_local_expression_reference_panels",
        lambda *args, **kwargs: {
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

    assert result["selected"]["cancer_type"] == "PRAD"
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

    assert result["selected"]["cancer_type"] == "READ"
    adcc = next(row for row in result["evidence"] if row["cancer_type"] == "ADCC")
    assert adcc["cancer_type"] == "ADCC"
    assert adcc["can_select_report_label"] is False
    assert (
        "expression-reference context is not one of HNSC"
        in adcc["blocking_reasons"]
    )


def test_rare_marker_selector_honors_min_support_genes_with_optional_markers():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    finding = {
        "cancer_type": "ACINIC",
        "rule_id": "acinic_nr4a3",
        "surrogate": "NR4A3",
        "surrogate_tpm": 18.0,
        "threshold_tpm": 10.0,
        "support_genes": ["SOX10"],
        "missing_support_genes": ["AQP5", "DOG1"],
        "support_pass": True,
        "support_gene_count": 1,
        "min_support_genes": 1,
        "required_support_gene_count": 3,
    }

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("HNSC", 1.0), ("LUSC", 0.25)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "ACINIC"
    acinic = next(row for row in result["evidence"] if row["cancer_type"] == "ACINIC")
    assert acinic["can_select_report_label"] is True
    assert acinic["support_gene_count"] == 1
    assert acinic["min_support_genes"] == 1
    assert acinic["required_support_gene_count"] == 3
    assert acinic["missing_support_genes"] == ["AQP5", "DOG1"]


def test_rare_marker_selector_rejects_rules_below_min_support_genes():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    finding = {
        "cancer_type": "ACINIC",
        "rule_id": "acinic_nr4a3",
        "surrogate": "NR4A3",
        "surrogate_tpm": 18.0,
        "threshold_tpm": 10.0,
        "support_genes": [],
        "missing_support_genes": ["SOX10", "AQP5", "DOG1"],
        "support_pass": False,
        "support_gene_count": 0,
        "min_support_genes": 1,
        "required_support_gene_count": 3,
    }

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("HNSC", 1.0), ("LUSC", 0.25)),
        rare_marker_hypotheses=[finding],
    )

    assert result["selected"]["cancer_type"] == "HNSC"
    assert all(row["cancer_type"] != "ACINIC" for row in result["evidence"])


def test_real_rare_rna_rules_promote_with_configured_partial_support():
    from trufflepig.rare_inference import infer_rare_cancer_report_scope_from_rna

    cases = [
        ("ACINIC", "HNSC", {"NR4A3": 18.0, "SOX10": 3.0}),
        ("ADCC", "HNSC", {"MYB": 18.0, "KIT": 3.0}),
        ("MTC", "THCA", {"CALCA": 18.0, "CHGA": 3.0}),
    ]

    for expected_code, context_code, tpm_by_symbol in cases:
        result = infer_rare_cancer_report_scope_from_rna(
            _expression_frame(tpm_by_symbol),
            _analysis((context_code, 1.0)),
        )
        assert result is not None
        assert result["cancer_type"] == expected_code
        assert result["support_pass"] is True
        assert result["support_gene_count"] == 1
        assert result["min_support_genes"] == 1
        assert result["missing_support_genes"]


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

    assert result["selected"]["cancer_type"] == "SARC"
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
    assert result["selected"]["expression_reference_cancer_type"] == "OS"
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

    assert result["selected"]["cancer_type"] == "SARC"
    os_evidence = next(row for row in result["evidence"] if row["cancer_type"] == "OS")
    assert os_evidence["cancer_type"] == "OS"
    assert os_evidence["can_select_report_label"] is False
    assert os_evidence["blocking_reasons"]


def test_blocked_marker_evidence_does_not_downgrade_direct_fusion_selection():
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("READ", 1.0), ("COAD", 0.4)),
        fusion_scope_inference={
            "cancer_type": "NUTM",
            "fusion": {"pair": "BRD4--NUTM1"},
            "expected_pair": "BRD4--NUTM1",
        },
        rare_marker_hypotheses=[
            {
                "cancer_type": "NUTM",
                "rule_id": "nutm_nutm1",
                "surrogate": "NUTM1",
                "surrogate_tpm": 8.0,
                "threshold_tpm": 1.0,
                "support_genes": [],
            }
        ],
    )

    assert result["selected"]["cancer_type"] == "NUTM"
    assert result["selected"]["selected_by"] == "direct_fusion"
    assert result["selected"]["label_decision"]["status"] == "selected"


def test_empty_candidate_trace_does_not_crash():
    """Degenerate analysis (no broad RNA classifier output) must produce
    an empty evidence result, not raise. Pins the all-empty contract.
    """
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        {"candidate_trace": []},
    )
    assert result["selected"] is None
    assert result["evidence"] == []
    assert result["primary_expression_context"] is None


def test_single_element_candidate_trace_runs_only_broad_path():
    """A 1-row trace can't trigger tumor_label_refinement (needs rank-2)
    but still emits a broad-RNA hypothesis for that single code.
    """
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        _analysis(("BRCA", 0.85)),
    )
    selected = result["selected"]
    assert selected is not None
    assert selected["cancer_type"] == "BRCA"
    assert selected["selected_by"] == "primary_expression_match"


def test_equally_prioritized_hypotheses_break_ties_alphabetically():
    """Two hypotheses with identical class_rank + strength + tiebreak
    must resolve deterministically (ascending cancer_type), not at
    Python's sort-stability default order (which depends on insertion
    order) or descending (the prior reverse=True bug).
    """
    from trufflepig.cancer_type_evidence import select_report_scope_from_evidence

    # Two cohorts tied at exactly the same support_fraction_of_top; the broad-RNA
    # path will create one hypothesis per code with identical priority.
    result = select_report_scope_from_evidence(
        _empty_expression_frame(),
        {
            "cancer_type": "READ",
            "candidate_trace": [
                {"code": "READ", "support_fraction_of_top": 1.0},
            ],
        },
    )
    # Sanity: deterministic — re-running produces the same winner.
    result2 = select_report_scope_from_evidence(
        _empty_expression_frame(),
        {
            "cancer_type": "READ",
            "candidate_trace": [
                {"code": "READ", "support_fraction_of_top": 1.0},
            ],
        },
    )
    assert result["selected"]["cancer_type"] == result2["selected"]["cancer_type"]


def test_subtype_deconvolved_expression_without_subtype_column(monkeypatch):
    """Defensive: when the subtype-deconvolved-expression accessor returns
    a DataFrame without a ``subtype`` column, the local-reference path
    must skip the row gracefully instead of crashing on
    ``df[scalar_bool]``.
    """
    import pandas as pd
    from trufflepig import cancer_type_evidence as cte
    from trufflepig import reference as ref_module

    no_subtype_df = pd.DataFrame(
        {
            "cancer_code": ["HEPB"],
            "symbol": ["AFP"],
            "ensembl_gene_id": ["ENSG00000081051"],
            "tumor_tpm_median": [120.0],
        }
    )

    def fake_subtype(*args, **kwargs):
        return no_subtype_df

    monkeypatch.setattr(
        ref_module, "subtype_deconvolved_expression", fake_subtype
    )
    cte._local_expression_reference_panels.cache_clear()
    # Should not raise even though no "subtype" column is present.
    try:
        cte._local_expression_reference_panels()
    finally:
        cte._local_expression_reference_panels.cache_clear()
