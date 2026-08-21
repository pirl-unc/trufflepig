"""Integration tests for the ``_apply_cancer_type_evidence`` wiring in main.

The wiring contract pinned here was previously inlined in
``_analyze_body``. Extracting it into a named helper made the unit
boundary small enough to test directly without spinning up an
end-to-end analyze run — these tests are the regression guard for
``trufflepig/main.py:_apply_cancer_type_evidence``.

The earlier PR #41 review noted there was no integration coverage for
the wiring (only unit tests of ``select_report_scope_from_evidence`` in
isolation). These tests fill that gap.
"""

from __future__ import annotations

import pandas as pd

from trufflepig.decomposition import (
    scope_residual_identity_to_decomposition_mode,
)
from trufflepig.main import (
    _apply_cancer_type_evidence,
    _fit_report_scope_decompositions,
    _propagate_report_scope_selection,
    _selected_report_scope_basis_label,
)


def _empty_expression_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ensembl_gene_id", "canonical_gene_name", "TPM"])


def _analysis(*rows) -> dict:
    return {
        "cancer_type": rows[0][0],
        "candidate_trace": [
            {"code": code, "support_fraction_of_top": support}
            for code, support in rows
        ],
    }


def test_selected_report_scope_basis_label_names_integrated_evidence_source():
    assert (
        _selected_report_scope_basis_label(
            {
                "cancer_type_evidence": {
                    "selected": {"selected_by": "local_expression_reference"}
                }
            }
        )
        == "exact local expression-reference evidence"
    )
    assert (
        _selected_report_scope_basis_label(
            {"cancer_type_evidence": {"selected": {"selected_by": "rare_marker"}}}
        )
        == "rare RNA-marker and expression-context evidence"
    )


def test_pan_cancer_signature_ranker_populates_analysis_dict_without_promoting():
    """When the top pan-cancer signature-ranker support is the strongest evidence, the
    helper records cancer_type_evidence in analysis but does NOT promote
    report_scope_cancer_type (the caller's auto-detected code stays in
    play for the rest of the pipeline)."""
    analysis = _analysis(("PRAD", 1.0), ("BRCA", 0.4))

    (
        cancer_type_evidence,
        selected_scope,
        report_scope_cancer_type,
        rare_scope_inference,
        fine_scope_inference,
    ) = _apply_cancer_type_evidence(
        analysis,
        _empty_expression_frame(),
        rna_inferred_cancer_type="PRAD",
        fusion_scope_inference=None,
        report_scope_cancer_type=None,
        rare_scope_inference=None,
        fine_scope_inference=None,
    )

    # The full evidence record lands in analysis.
    assert "cancer_type_evidence" in analysis
    assert cancer_type_evidence is analysis["cancer_type_evidence"]

    # Selection happened, but selected_by is pan_cancer_signature_ranker -> no
    # report-scope promotion. The caller keeps the auto-detected code.
    assert selected_scope is not None
    assert selected_scope["cancer_type"] == "PRAD"
    assert selected_scope["selected_by"] == "pan_cancer_signature_ranker"
    assert report_scope_cancer_type is None
    assert rare_scope_inference is None
    assert fine_scope_inference is None

    # Analysis side-effects: inferred_cancer_type + expression_reference
    # come from the selection, even when no override fires.
    assert analysis["inferred_cancer_type"] == "PRAD"
    assert analysis["expression_reference_cancer_type"] == "PRAD"
    assert analysis["primary_expression_context"]["cancer_type"] == "PRAD"


def test_direct_fusion_promotes_report_scope_and_records_inference():
    """A direct fusion match should promote report_scope_cancer_type AND
    route the selected scope into rare_scope_inference / fine_scope_inference
    only via the corresponding evidence_sources — here, neither (the
    fusion path is its own bucket)."""
    fusion_scope = {
        "cancer_type": "NUTM",
        "fusion": "BRD4-NUTM1",
        "expected_pair": "BRD4 :: NUTM1",
        "basis": "BRD4-NUTM1 fusion is the defining lesion of NUT carcinoma",
        "confirmatory_tests": "FISH for NUTM1 break-apart",
        "source": "rule",
    }
    analysis = _analysis(("LUSC", 1.0), ("HNSC", 0.7))

    (
        _evidence,
        selected_scope,
        report_scope_cancer_type,
        rare_scope_inference,
        fine_scope_inference,
    ) = _apply_cancer_type_evidence(
        analysis,
        _empty_expression_frame(),
        rna_inferred_cancer_type="LUSC",
        fusion_scope_inference=fusion_scope,
        report_scope_cancer_type=None,
        rare_scope_inference=None,
        fine_scope_inference=None,
    )

    assert selected_scope is not None
    assert selected_scope["cancer_type"] == "NUTM"
    assert selected_scope["selected_by"] == "direct_fusion"
    assert report_scope_cancer_type == "NUTM"
    # Direct fusion is neither rare_marker nor fine_reference / local
    # — so neither inference slot is populated.
    assert rare_scope_inference is None
    assert fine_scope_inference is None


def test_rare_marker_promotion_routes_into_rare_scope_inference():
    """A rare-marker hypothesis that passes the policy gate should
    populate rare_scope_inference (not fine_scope_inference)."""
    analysis = _analysis(("LUSC", 1.0), ("HNSC", 0.7))
    analysis["rare_marker_hypotheses"] = [
        {
            "cancer_type": "NUTM",
            "rule_id": "nutm_nutm1",
            "surrogate": "NUTM1",
            "surrogate_tpm": 6.2,
            "threshold_tpm": 1.0,
            "support_genes": [],
        }
    ]

    (
        _evidence,
        selected_scope,
        report_scope_cancer_type,
        rare_scope_inference,
        fine_scope_inference,
    ) = _apply_cancer_type_evidence(
        analysis,
        _empty_expression_frame(),
        rna_inferred_cancer_type="LUSC",
        fusion_scope_inference=None,
        report_scope_cancer_type=None,
        rare_scope_inference=None,
        fine_scope_inference=None,
    )

    assert selected_scope is not None
    assert selected_scope["cancer_type"] == "NUTM"
    assert "rare_marker" in selected_scope["evidence_sources"]
    assert report_scope_cancer_type == "NUTM"
    assert rare_scope_inference is selected_scope
    assert fine_scope_inference is None


def test_fine_reference_promotion_routes_into_fine_scope_inference(monkeypatch):
    """An OS osteogenic-program match should populate fine_scope_inference."""
    import trufflepig.cancer_type_evidence as evidence
    from trufflepig.cancer_type_evidence import _FINE_REFERENCE_SPECS

    # Sanity: there is a configured fine-reference spec for OS so the path
    # is exercised end-to-end (not just stubbed).
    assert any(s.cancer_type == "SARC_OS" for s in _FINE_REFERENCE_SPECS)

    # Keep this integration test on the public main -> selector -> fine-scope
    # route, but do not initialize independent evidence systems. Their own
    # integration tests load the real learned, lineage, centroid, composition,
    # contrast, and exact-reference data.
    for helper in (
        "_add_learned_expression_classifier_features",
        "_add_learned_hierarchy_candidate_features",
        "_add_pan_cancer_signature_marker_features",
        "_add_coarse_composition_reference_features",
        "_add_local_expression_reference_features",
        "_add_lineage_panel_features",
    ):
        monkeypatch.setattr(evidence, helper, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evidence,
        "_centroid_and_confidence",
        lambda *args, **kwargs: (None, False),
    )
    monkeypatch.setattr(evidence, "_contrast_discriminator_rows", lambda: ())

    analysis = _analysis(("SARC", 1.0), ("UCS", 0.3))

    # Build a sample frame with strong osteogenic markers — same shape as
    # tests/test_cancer_type_evidence.py uses for the standalone module.
    from trufflepig.reference import pan_cancer_expression

    ref = pan_cancer_expression()[["Ensembl_Gene_ID", "Symbol"]].drop_duplicates(
        "Symbol"
    )
    id_by_symbol = dict(zip(ref["Symbol"], ref["Ensembl_Gene_ID"]))
    tpm_by_symbol = {
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
    rows = [
        {
            "ensembl_gene_id": id_by_symbol[symbol],
            "canonical_gene_name": symbol,
            "TPM": float(tpm),
        }
        for symbol, tpm in tpm_by_symbol.items()
        if symbol in id_by_symbol
    ]
    df_expr = pd.DataFrame(rows)

    (
        _evidence,
        selected_scope,
        report_scope_cancer_type,
        rare_scope_inference,
        fine_scope_inference,
    ) = _apply_cancer_type_evidence(
        analysis,
        df_expr,
        rna_inferred_cancer_type="SARC",
        fusion_scope_inference=None,
        report_scope_cancer_type=None,
        rare_scope_inference=None,
        fine_scope_inference=None,
    )

    assert selected_scope is not None
    assert selected_scope["cancer_type"] == "SARC_OS"
    assert "fine_reference" in selected_scope["evidence_sources"]
    assert report_scope_cancer_type == "SARC_OS"
    assert rare_scope_inference is None
    assert fine_scope_inference is selected_scope


def test_routing_uses_selected_by_not_set_membership(monkeypatch):
    """When a hypothesis has multiple selectors in evidence_sources, the
    routing must use ``selected_by`` (the winner), not naive set
    membership.

    Regression: prior code at main.py:4690-4696 checked ``"rare_marker"
    in evidence_sources`` first, so a hypothesis selected by
    ``fine_reference`` but also touched by a rare-marker selector would
    incorrectly route into ``rare_scope_inference``.
    """
    from trufflepig import cancer_type_evidence as cte_module

    fake_selected = {
        "cancer_type": "SARC_OS",
        "selected_by": "fine_reference",
        "evidence_sources": ["fine_reference", "rare_marker"],
        "expression_reference_cancer_type": "SARC_OS",
    }
    fake_evidence = {
        "selected": fake_selected,
        "primary_expression_context": "SARC",
    }
    monkeypatch.setattr(
        cte_module,
        "select_report_scope_from_evidence",
        lambda *a, **k: fake_evidence,
    )
    analysis = {"candidate_trace": [{"code": "SARC", "support_fraction_of_top": 1.0}]}
    (
        _evidence,
        selected_scope,
        report_scope_cancer_type,
        rare_scope_inference,
        fine_scope_inference,
    ) = _apply_cancer_type_evidence(
        analysis,
        pd.DataFrame(),
        rna_inferred_cancer_type="SARC",
        fusion_scope_inference=None,
        report_scope_cancer_type=None,
        rare_scope_inference=None,
        fine_scope_inference=None,
    )
    assert selected_scope is fake_selected
    assert report_scope_cancer_type == "SARC_OS"
    assert rare_scope_inference is None
    assert fine_scope_inference is fake_selected


def test_helper_is_robust_to_import_or_load_failures(monkeypatch):
    """If the cancer_type_evidence selector raises something the helper
    catches (KeyError/ValueError/TypeError), the helper returns None for
    cancer_type_evidence and leaves the caller's state untouched. This
    pins the contract that a single internal failure does not blow up
    the broader analyze pipeline."""
    import trufflepig.main as main_mod

    def raises(*args, **kwargs):
        raise KeyError("simulated registry failure")

    monkeypatch.setattr(
        "trufflepig.cancer_type_evidence.select_report_scope_from_evidence",
        raises,
    )

    analysis = _analysis(("PRAD", 1.0))
    (
        cancer_type_evidence,
        selected_scope,
        report_scope_cancer_type,
        rare_scope_inference,
        fine_scope_inference,
    ) = main_mod._apply_cancer_type_evidence(
        analysis,
        _empty_expression_frame(),
        rna_inferred_cancer_type="PRAD",
        fusion_scope_inference=None,
        report_scope_cancer_type=None,
        rare_scope_inference=None,
        fine_scope_inference=None,
    )
    assert cancer_type_evidence is None
    assert selected_scope is None
    assert report_scope_cancer_type is None
    assert rare_scope_inference is None
    assert fine_scope_inference is None
    # The helper must not have written cancer_type_evidence on failure —
    # the caller's analysis dict stays clean.
    assert "cancer_type_evidence" not in analysis


def test_helper_forwards_post_decomposition_identity_evidence(monkeypatch):
    """The second selector pass receives the independent residual audit."""

    import trufflepig.cancer_type_evidence as cte_module

    captured = {}
    fake_selected = {
        "cancer_type": "ACC",
        "selected_by": "entity_evidence_consensus",
        "evidence_sources": ["learned_expression_classifier"],
        "expression_reference_cancer_type": "ACC",
    }

    def fake_select(*args, **kwargs):
        captured.update(kwargs)
        return {
            "selected": fake_selected,
            "primary_expression_context": {"cancer_type": "SARC_DDLPS"},
            "residual_identity_evidence": kwargs.get(
                "residual_identity_evidence"
            ),
        }

    monkeypatch.setattr(
        cte_module,
        "select_report_scope_from_evidence",
        fake_select,
    )
    residual = {
        "status": "candidate",
        "candidate_code": "ACC",
        "panel_candidate_code": "ACC",
        "current_code": "SARC_DDLPS",
    }
    analysis = _analysis(("SARC_DDLPS", 1.0), ("ACC", 0.8))

    evidence, selected, report_code, *_ = _apply_cancer_type_evidence(
        analysis,
        _empty_expression_frame(),
        rna_inferred_cancer_type="SARC_DDLPS",
        fusion_scope_inference=None,
        report_scope_cancer_type=None,
        rare_scope_inference=None,
        fine_scope_inference=None,
        residual_identity_evidence=residual,
    )

    assert captured["residual_identity_evidence"] is residual
    assert evidence["residual_identity_evidence"] is residual
    assert selected is fake_selected
    assert report_code == "ACC"


def test_residual_selection_clears_superseded_report_basis(monkeypatch):
    """A second-pass winner cannot inherit the old rare/fine narrative."""

    import trufflepig.cancer_type_evidence as cte_module
    from trufflepig.brief import _cancer_type_basis_line

    selected = {
        "cancer_type": "BLCA",
        "reference_cancer_type": "BLCA",
        "expression_reference_cancer_type": "BLCA",
        "selected_by": "entity_evidence_consensus",
        "evidence_sources": ["residual_identity", "lineage_panel"],
    }
    monkeypatch.setattr(
        cte_module,
        "select_report_scope_from_evidence",
        lambda *args, **kwargs: {
            "selected": selected,
            "primary_expression_context": {"cancer_type": "BLCA"},
        },
    )
    stale_rare = {"cancer_type": "NUTM", "surrogate": "NUTM1"}
    stale_fine = {
        "cancer_type": "CHOL",
        "reference_cancer_type": "CHOL",
    }
    analysis = _analysis(("CHOL", 1.0), ("BLCA", 0.95))
    analysis.update(
        {
            "rare_report_scope_inference": stale_rare,
            "fine_report_scope_inference": stale_fine,
            "cancer_type_source": "auto-detected",
        }
    )

    (
        _evidence,
        returned_selection,
        report_code,
        rare_inference,
        fine_inference,
    ) = _apply_cancer_type_evidence(
        analysis,
        _empty_expression_frame(),
        rna_inferred_cancer_type="CHOL",
        fusion_scope_inference=None,
        report_scope_cancer_type="CHOL",
        rare_scope_inference=stale_rare,
        fine_scope_inference=stale_fine,
        residual_identity_evidence={"status": "candidate", "candidate_code": "BLCA"},
    )

    assert rare_inference is None
    assert fine_inference is None
    assert "rare_report_scope_inference" not in analysis
    assert "fine_report_scope_inference" not in analysis
    _propagate_report_scope_selection(
        analysis,
        _empty_expression_frame(),
        report_scope_cancer_type=report_code,
        selected_scope=returned_selection,
        rna_inferred_cancer_type="CHOL",
        rare_scope_inference=rare_inference,
        fine_scope_inference=fine_inference,
    )

    assert "rare_report_scope_inference" not in analysis
    assert "fine_report_scope_inference" not in analysis
    basis = _cancer_type_basis_line(analysis, "BLCA")
    assert "RNA-inferred hypothesis" in basis
    assert "rare-cancer hypothesis" not in basis
    assert "fine label" not in basis


def test_report_scope_decomposition_refit_uses_final_purity_and_site(monkeypatch):
    """The shared fit entry point consumes synchronized post-selection state."""

    from types import SimpleNamespace

    import trufflepig.main as main_module

    stale_purity = {"cancer_type": "CHOL", "overall_estimate": 0.25}
    final_purity = {"cancer_type": "BLCA", "overall_estimate": 0.60}
    analysis = {
        "cancer_type": "BLCA",
        "expression_reference_cancer_type": "BLCA",
        "purity": final_purity,
        "candidate_trace": [
            {"code": "CHOL", "purity_result": stale_purity},
            {"code": "BLCA", "purity_result": final_purity},
        ],
    }
    captured = {}

    def fake_decompose(_df_expr, **kwargs):
        captured.update(kwargs)
        final_row = next(
            row for row in kwargs["candidate_rows"] if row["code"] == "BLCA"
        )
        return [
            SimpleNamespace(
                cancer_type="BLCA",
                purity_result=final_row["purity_result"],
                fractions={"tumor": 0.60, "hepatocyte": 0.40},
            )
        ]

    monkeypatch.setattr(main_module, "decompose_sample", fake_decompose)
    monkeypatch.setattr(
        main_module,
        "_has_operational_analysis_reference",
        lambda _code: True,
    )

    results, candidate_codes, context = _fit_report_scope_decompositions(
        _empty_expression_frame(),
        analysis,
        report_code="BLCA",
        reference_code="BLCA",
        sample_mode="solid",
        tumor_context="auto",
        explicit_site_hint=None,
        inferred_site_context={"site": "liver"},
        template_overrides=(),
        sample_context={"degradation_severity": "none"},
    )

    assert results[0].purity_result is final_purity
    assert candidate_codes == ["BLCA", "CHOL"]
    assert captured["cancer_types"] == ["BLCA", "CHOL"]
    assert captured["candidate_rows"] is analysis["candidate_trace"]
    assert captured["site_hint"] == "liver"
    assert captured["tumor_context"] == "met"
    assert context == {
        "site_hint": "liver",
        "tumor_context": "met",
        "identity_background_models": 0,
    }


def test_residual_identity_stays_within_the_decomposition_regime():
    solid_residual = {
        "status": "candidate",
        "candidate_code": "CRC",
        "current_code": "SARC_PLEOLPS",
    }
    compatible = scope_residual_identity_to_decomposition_mode(
        solid_residual,
        sample_mode="solid",
    )
    assert compatible["adjudication_eligible"] is True
    assert compatible["candidate_sample_mode"] == "solid"

    heme_residual_from_solid_fit = {
        "status": "candidate",
        "candidate_code": "DLBC",
        "current_code": "SARC_PLEOLPS",
    }
    incompatible = scope_residual_identity_to_decomposition_mode(
        heme_residual_from_solid_fit,
        sample_mode="solid",
    )
    assert incompatible["adjudication_eligible"] is False
    assert incompatible["candidate_sample_mode"] == "heme"
    assert "requires heme decomposition" in incompatible["adjudication_blocker"]


def test_final_scope_refit_must_reproduce_compatible_residual_identity():
    from trufflepig.main import _residual_identity_confirms_report_scope

    assert _residual_identity_confirms_report_scope(
        {
            "status": "corroborated",
            "candidate_code": "CRC",
            "adjudication_eligible": True,
        },
        "READ",
    )
    assert not _residual_identity_confirms_report_scope(
        {"status": "ambiguous", "candidate_code": "CRC"},
        "READ",
    )
    assert not _residual_identity_confirms_report_scope(
        {
            "status": "corroborated",
            "candidate_code": "CRC",
            "adjudication_eligible": False,
        },
        "READ",
    )
    assert not _residual_identity_confirms_report_scope(
        {"status": "corroborated", "candidate_code": "CRC"},
        "SARC_DDLPS",
    )
