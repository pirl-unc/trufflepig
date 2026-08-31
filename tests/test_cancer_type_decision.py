from types import SimpleNamespace

import pandas as pd

from trufflepig.common import panel_symbols_to_gene_ids
from trufflepig.decomposition import (
    CancerTypeDecision,
    decide_cancer_type_from_decomposition,
)


def _result(
    cancer_type,
    *,
    template="met_liver",
    tumor_values,
    site_supported=True,
    purity=0.4,
):
    symbols = set(tumor_values) | {
        "UPK1A",
        "UPK1B",
        "UPK2",
        "UPK3A",
        "KRT20",
        "GATA3",
        "FOXA1",
        "PPARG",
        "MUCL1",
        "SCGB2A2",
        "CDX2",
        "KRT19",
        "MUC1",
        "MUC5AC",
        "EPCAM",
        "CDH1",
        "CFTR",
        "ALB",
        "AFP",
        "CDH17",
        "VIL1",
        "PTPRC",
        "CD3D",
        "MS4A1",
        "MYOD1",
        "MYOG",
        "DES",
    }
    gene_ids = panel_symbols_to_gene_ids(sorted(symbols))
    attribution = pd.DataFrame(
        [
            {
                "gene_id": gene_ids.get(symbol, f"unmapped-{symbol}"),
                "symbol": symbol,
                "observed_tpm": float(tumor_values.get(symbol, 0.0)),
                "tumor": float(tumor_values.get(symbol, 0.0)),
                "hepatocyte": 0.0,
                "overexplained_tpm": 0.0,
                "tumor_fraction_of_total": (
                    1.0 if float(tumor_values.get(symbol, 0.0)) > 0 else 0.0
                ),
            }
            for symbol in sorted(symbols)
        ]
    )
    return SimpleNamespace(
        cancer_type=cancer_type,
        template=template,
        gene_attribution=attribution,
        warnings=[],
        site_evidence={"site_supported": site_supported},
        purity=purity,
        reconstruction_error=12.0,
    )


def _blca_residual(scale=1.0):
    return {
        symbol: value * scale
        for symbol, value in {
            "UPK1A": 60,
            "UPK1B": 170,
            "UPK2": 85,
            "UPK3A": 16,
            "KRT20": 25,
            "GATA3": 30,
            "FOXA1": 16,
            "PPARG": 20,
            "KRT19": 12,
            "MUC1": 8,
            "EPCAM": 12,
            "CDH1": 12,
            # CHOL remains incomplete because MUC5AC and CFTR are absent.
            "MUC5AC": 0,
            "CFTR": 0,
        }.items()
    }


def _chol_residual():
    return {
        "KRT19": 80,
        "MUC1": 60,
        "MUC5AC": 40,
        "EPCAM": 50,
        "CDH1": 45,
        "CFTR": 35,
    }


def _crc_residual():
    return {
        "VIL1": 200,
        "CDH17": 200,
        "SATB2": 200,
        "CDX2": 200,
        "KRT20": 200,
        "MUC2": 200,
        "ALB": 0,
        "AFP": 0,
        "UPK1B": 0,
        "MYOD1": 0,
        "MYOG": 0,
        # A large absolute residual is still background-resolved when the
        # fitted smooth-muscle component explains most of the gene.
        "DES": 40,
    }


def test_cancer_type_decision_requires_invariance_across_candidate_realizations():
    """The same tumor program survives different candidate purity priors."""

    evidence = decide_cancer_type_from_decomposition(
        [
            _result("BLCA", tumor_values=_blca_residual(), purity=0.65),
            _result("CHOL", tumor_values=_blca_residual(0.8), purity=0.32),
            _result("HEPB", tumor_values=_blca_residual(1.1), purity=0.53),
        ],
        candidate_codes=["BLCA", "CHOL", "HEPB"],
        current_code="BLCA",
    )

    assert evidence.status == "resolved"
    assert evidence.supported_code == "BLCA"
    assert evidence.models_evaluated == 1
    assert evidence.realizations_evaluated == 3
    assert len(evidence.background_models) == 1
    model = evidence.background_models[0]
    assert model["template"] == "met_liver"
    assert model["panel_candidate"] == "BLCA"
    assert model["realizations"] == 3


def test_cancer_type_decision_preserves_conflicting_panel_votes(monkeypatch):
    """A contradictory axis cannot disappear behind a unanimous second axis."""
    import trufflepig.decomposition.cancer_type_decision as decision_module
    import trufflepig.lineage_panels as lineage_panels

    decisions = iter(
        [
            {
                "decisive": True,
                "top_parent_cohort": "BLCA",
                "reason": "complete BLCA program",
            },
            {
                "decisive": True,
                "top_parent_cohort": "CHOL",
                "reason": "complete CHOL program",
            },
        ]
    )
    monkeypatch.setattr(
        lineage_panels,
        "evaluate_panels",
        lambda *_args, **_kwargs: (SimpleNamespace(parent_cohort="BLCA"),),
    )
    monkeypatch.setattr(
        lineage_panels,
        "complete_program_entity_decision",
        lambda _rows: next(decisions),
    )

    def ontology_sanity(code, _residual):
        if code == "BLCA":
            return {
                "status": "coherent",
                "expected_high": ["UPK2"],
                "expected_high_detected": ["UPK2"],
                "expected_low": ["ALB"],
                "expected_low_present": [],
            }
        return {
            "status": "incomplete",
            "expected_high": ["MUC5AC"],
            "expected_high_detected": [],
            "expected_low": [],
            "expected_low_present": [],
        }

    monkeypatch.setattr(
        decision_module,
        "tumor_type_sanity_check",
        ontology_sanity,
    )

    evidence = decide_cancer_type_from_decomposition(
        [
            _result("BLCA", tumor_values=_blca_residual()),
            _result("CHOL", tumor_values=_blca_residual()),
        ],
        candidate_codes=["BLCA", "CHOL"],
        current_code="BLCA",
    )

    assert evidence.status == "ambiguous"
    assert not evidence.supported_code
    model = evidence.background_models[0]
    assert model["candidate_code"] is None
    assert model["panel_conflicting_candidates"] == ["BLCA", "CHOL"]
    assert model["ontology_candidate"] == "BLCA"
    assert model["ontology_conflicting_candidates"] == []


def test_cancer_type_decision_preserves_conflicting_ontology_votes(monkeypatch):
    import trufflepig.decomposition.cancer_type_decision as decision_module
    import trufflepig.lineage_panels as lineage_panels

    monkeypatch.setattr(
        lineage_panels,
        "evaluate_panels",
        lambda *_args, **_kwargs: (SimpleNamespace(parent_cohort="BLCA"),),
    )
    monkeypatch.setattr(
        lineage_panels,
        "complete_program_entity_decision",
        lambda _rows: {
            "decisive": True,
            "top_parent_cohort": "BLCA",
            "reason": "complete BLCA program",
        },
    )

    def ontology_sanity(code, residual):
        first_realization = float(residual.get("UPK1A", 0.0)) > 50.0
        selected = "BLCA" if first_realization else "CHOL"
        complete = code == selected
        return {
            "status": "coherent" if complete else "incomplete",
            "expected_high": ["identity-marker"],
            "expected_high_detected": ["identity-marker"] if complete else [],
            "expected_low": ["excluded-marker"],
            "expected_low_present": [],
        }

    monkeypatch.setattr(
        decision_module,
        "tumor_type_sanity_check",
        ontology_sanity,
    )

    evidence = decide_cancer_type_from_decomposition(
        [
            _result("BLCA", tumor_values=_blca_residual()),
            _result("CHOL", tumor_values=_blca_residual(0.8)),
        ],
        candidate_codes=["BLCA", "CHOL"],
        current_code="BLCA",
    )

    assert evidence.status == "ambiguous"
    assert not evidence.supported_code
    model = evidence.background_models[0]
    assert model["candidate_code"] is None
    assert model["panel_candidate"] == "BLCA"
    assert model["panel_conflicting_candidates"] == []
    assert model["ontology_conflicting_candidates"] == ["BLCA", "CHOL"]


def test_cancer_type_decision_abstains_when_background_models_disagree():
    """Competing host models must not be collapsed into a plurality."""

    evidence = decide_cancer_type_from_decomposition(
        [
            _result("BLCA", tumor_values=_blca_residual(), template="met_liver"),
            _result("CHOL", tumor_values=_chol_residual(), template="solid_primary"),
        ],
        candidate_codes=["BLCA", "CHOL"],
        current_code="BLCA",
    )

    assert evidence.status == "ambiguous"
    assert not evidence.supported_code
    assert {
        row["candidate_code"] for row in evidence.background_models
    } == {"BLCA", "CHOL"}


def test_cancer_type_decision_rejects_unsupported_metastatic_backgrounds():
    evidence = decide_cancer_type_from_decomposition(
        [
            _result(
                "BLCA",
                tumor_values=_blca_residual(),
                site_supported=False,
            )
        ],
        candidate_codes=["BLCA", "CHOL"],
        current_code="CHOL",
    )

    assert evidence.status == "not_evaluable"
    assert evidence.models_evaluated == 0
    assert evidence.realizations_evaluated == 0


def test_cancer_type_decision_does_not_create_a_candidate_from_zero_programs():
    """Failed identity gates and absent high markers are an abstention."""

    evidence = decide_cancer_type_from_decomposition(
        [_result("BLCA", tumor_values={})],
        candidate_codes=["BLCA", "CHOL"],
        current_code="CHOL",
    )

    assert not evidence.supported_code
    assert evidence.status in {"ambiguous", "not_evaluable"}
    assert all(
        row["panel_candidate"] == ""
        for model in evidence.background_models
        for row in model["rows"]
    )


def test_identity_background_requires_complete_panel_not_pareto_only(monkeypatch):
    """Candidate-independent subtraction cannot originate an incomplete vote."""
    import trufflepig.decomposition.cancer_type_decision as decision_module
    import trufflepig.lineage_panels as lineage_panels

    panel_rows = (
        SimpleNamespace(
            panel_name="BLCA partial",
            parent_cohort="BLCA",
            obligate_passed=True,
            identity_marker_groups_passed=True,
            high_hits=(("UPK2", 20.0),),
            high_misses=(("UPK1A", 0.0),),
            low_passes=(("ALB", 0.0),),
            low_violations=(),
            score=0.6,
        ),
        SimpleNamespace(
            panel_name="CHOL absent",
            parent_cohort="CHOL",
            obligate_passed=True,
            identity_marker_groups_passed=True,
            high_hits=(("MUC1", 6.0),),
            high_misses=(("MUC5AC", 0.0), ("CFTR", 0.0)),
            low_passes=(),
            low_violations=(("UPK2", 20.0),),
            score=0.1,
        ),
    )
    monkeypatch.setattr(
        lineage_panels,
        "evaluate_panels",
        lambda *_args, **_kwargs: panel_rows,
    )

    def ontology_sanity(code, _residual):
        return {
            "status": "coherent" if code == "BLCA" else "incomplete",
            "expected_high": ["identity-marker"],
            "expected_high_detected": ["identity-marker"] if code == "BLCA" else [],
            "expected_low": ["excluded-marker"],
            "expected_low_present": [],
        }

    monkeypatch.setattr(
        decision_module,
        "tumor_type_sanity_check",
        ontology_sanity,
    )
    result = _result("BLCA", tumor_values={"UPK2": 20.0})
    result.model_role = "identity_background"

    evidence = decide_cancer_type_from_decomposition(
        [result],
        candidate_codes=["BLCA", "CHOL"],
        current_code="CHOL",
    )

    assert evidence.status == "ambiguous"
    assert not evidence.supported_code
    model = evidence.background_models[0]
    assert model["panel_candidate"] == "BLCA"
    assert model["complete_panel_candidate"] is None
    assert model["candidate_code"] is None
    assert model["rows"][0]["panel_decision_basis"] == "pareto_dominance"


def test_cancer_type_decision_reports_shared_crc_parent_not_arbitrary_sibling():
    results = [
        _result(
            code,
            template="met_soft_tissue",
            tumor_values=_crc_residual(),
        )
        for code in ("READ", "SARC_PLEOLPS")
    ]
    for result in results:
        result.gene_attribution["smooth_muscle"] = 0.0
        is_des = result.gene_attribution["symbol"].eq("DES")
        result.gene_attribution.loc[is_des, "smooth_muscle"] = 600.0
        result.gene_attribution.loc[is_des, "observed_tpm"] = 640.0

    evidence = decide_cancer_type_from_decomposition(
        results,
        candidate_codes=["READ", "SARC_PLEOLPS"],
        current_code="SARC_PLEOLPS",
    )

    assert evidence.status == "resolved"
    assert evidence.supported_code == "CRC"
    assert evidence.panel_code == "CRC"
    assert evidence.ontology_code == "CRC"
    assert {
        row["raw_ontology_candidate"]
        for row in evidence.background_models[0]["rows"]
    } == {"READ"}


def test_background_separated_parent_broadens_an_unsupported_child():
    decision = CancerTypeDecision.from_dict(
        {
            "status": "resolved",
            "supported_code": "CRC",
            "current_code": "READ",
            "panel_code": "CRC",
            "ontology_code": "CRC",
            "background_separation_confirmed": True,
            "background_models": [
                {"candidate_code": "CRC", "realizations": 2}
            ],
        }
    )

    assert decision.relationship == "ancestor"
    assert decision.proposed_code == "CRC"


def test_missing_decomposition_mode_blocks_selection():
    decision = CancerTypeDecision.from_dict(
        {
            "status": "resolved",
            "supported_code": "BLCA",
            "current_code": "CHOL",
            "panel_code": "BLCA",
        }
    ).for_sample_mode("")

    assert decision.selection_allowed is False
    assert decision.proposed_code == ""
    assert decision.block_reason == "unsupported decomposition mode: missing"


def test_final_refit_must_reproduce_required_background_separation():
    ordinary_refit = CancerTypeDecision.from_dict(
        {
            "status": "resolved",
            "supported_code": "READ",
            "current_code": "CRC",
            "panel_code": "READ",
            "background_separation_confirmed": False,
        }
    )

    assert ordinary_refit.confirms("CRC") is True
    assert (
        ordinary_refit.confirms(
            "CRC",
            require_background_separation=True,
        )
        is False
    )


def test_descendant_refit_is_recorded_as_a_broad_selection_basis():
    analysis = {
        "cancer_type": "CRC",
        "cancer_type_decision": {
            "status": "resolved",
            "supported_code": "READ",
            "current_code": "CRC",
            "panel_code": "READ",
            "ontology_code": "READ",
            "background_separation_confirmed": True,
            "background_models": [
                {"candidate_code": "READ", "realizations": 2}
            ],
        },
        "cancer_type_decision_refit": {"accepted": True},
        "cancer_type_evidence": {
            "selected": {
                "selected_by": "entity_evidence_consensus",
                "entity_evidence_consensus": {
                    "decomposition_decision_was_decisive": True,
                },
            }
        },
    }

    decision = CancerTypeDecision.from_analysis(analysis)

    assert decision.relationship == "descendant"
    assert decision.is_selection_basis is True
