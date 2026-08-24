"""Decision-vector coverage for learned report-entity aggregation."""

import pytest


def test_flat_entity_leader_uses_complete_vector_not_public_top_ten(monkeypatch):
    """Subtype mass outside the audit preview still participates in decisions."""
    import trufflepig.cancer_type_evidence as evidence
    import trufflepig.expression_classifier as classifier
    from trufflepig.cancer_type_evidence import CancerTypeEvidence

    hierarchy_votes = [
        {
            "stage": "entity",
            "label": "STAD",
            "probability": 0.40,
            "margin": 0.10,
            "top_predictions": [
                {"label": "STAD", "probability": 0.40},
                {"label": "BRCA", "probability": 0.30},
            ],
        }
    ]
    monkeypatch.setattr(
        evidence,
        "_learned_hierarchical_votes",
        lambda _sample: hierarchy_votes,
    )
    monkeypatch.setattr(
        evidence,
        "_candidate_mismatch_repair_votes",
        lambda *_args, **_kwargs: [],
    )

    # STAD plus nine distinct leaves fill the public top-ten audit preview.
    # Five lower-probability BRCA subtypes collectively form the actual flat
    # report-entity leader.
    flat_predictions = [
        ("STAD", 0.18),
        *[
            (code, 0.06)
            for code in (
                "PRAD",
                "BLCA",
                "LIHC",
                "CHOL",
                "PAAD",
                "ACC",
                "THCA",
                "KIRC",
                "SKCM",
            )
        ],
        *[
            (code, 0.055)
            for code in (
                "BRCA_Normal",
                "BRCA_Basal",
                "BRCA_HER2",
                "BRCA_LumB",
                "BRCA_LumA",
            )
        ],
    ]
    monkeypatch.setattr(
        classifier,
        "classify_expression",
        lambda _sample, top_k=5: flat_predictions[:top_k],
    )

    hypotheses = {
        "STAD": CancerTypeEvidence(cancer_type="STAD"),
        "BRCA": CancerTypeEvidence(cancer_type="BRCA"),
    }
    evidence._add_learned_hierarchy_candidate_features(
        hypotheses,
        {"MKI67": 10.0},
    )

    details = hypotheses["STAD"].details
    assert len(details["learned_expression_flat_top_predictions"]) == 10
    assert not any(
        row["code"].startswith("BRCA_")
        for row in details["learned_expression_flat_top_predictions"]
    )
    assert details["learned_expression_flat_entity_supports"]["BRCA"] == (
        pytest.approx(0.275)
    )
    leaders = evidence._learned_entity_prediction_codes(details)
    assert leaders[0][0] == "STAD"  # hierarchy view
    assert leaders[1][0] == "BRCA"  # complete flat view
