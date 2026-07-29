from types import SimpleNamespace

import pandas as pd

from trufflepig.common import panel_symbols_to_gene_ids
from trufflepig.decomposition import evaluate_residual_identity


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


def test_residual_identity_requires_invariance_across_candidate_realizations():
    """The same tumor program survives different candidate purity priors."""

    evidence = evaluate_residual_identity(
        [
            _result("BLCA", tumor_values=_blca_residual(), purity=0.65),
            _result("CHOL", tumor_values=_blca_residual(0.8), purity=0.32),
            _result("HEPB", tumor_values=_blca_residual(1.1), purity=0.53),
        ],
        candidate_codes=["BLCA", "CHOL", "HEPB"],
        current_code="BLCA",
    )

    assert evidence["status"] == "corroborated"
    assert evidence["candidate_code"] == "BLCA"
    assert evidence["models_evaluated"] == 1
    assert evidence["realizations_evaluated"] == 3
    assert len(evidence["background_models"]) == 1
    model = evidence["background_models"][0]
    assert model["template"] == "met_liver"
    assert model["panel_candidate"] == "BLCA"
    assert model["realizations"] == 3


def test_residual_identity_abstains_when_background_models_disagree():
    """Competing host models must not be collapsed into a plurality."""

    evidence = evaluate_residual_identity(
        [
            _result("BLCA", tumor_values=_blca_residual(), template="met_liver"),
            _result("CHOL", tumor_values=_chol_residual(), template="solid_primary"),
        ],
        candidate_codes=["BLCA", "CHOL"],
        current_code="BLCA",
    )

    assert evidence["status"] == "ambiguous"
    assert evidence["candidate_code"] is None
    assert {
        row["candidate_code"] for row in evidence["background_models"]
    } == {"BLCA", "CHOL"}


def test_residual_identity_rejects_unsupported_metastatic_backgrounds():
    evidence = evaluate_residual_identity(
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

    assert evidence["status"] == "not_evaluable"
    assert evidence["models_evaluated"] == 0
    assert evidence["realizations_evaluated"] == 0


def test_residual_identity_reports_shared_crc_parent_not_arbitrary_sibling():
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

    evidence = evaluate_residual_identity(
        results,
        candidate_codes=["READ", "SARC_PLEOLPS"],
        current_code="SARC_PLEOLPS",
    )

    assert evidence["status"] == "candidate"
    assert evidence["candidate_code"] == "CRC"
    assert evidence["panel_candidate_code"] == "CRC"
    assert evidence["ontology_candidate_code"] == "CRC"
    assert {
        row["raw_ontology_candidate"]
        for row in evidence["background_models"][0]["rows"]
    } == {"READ"}
