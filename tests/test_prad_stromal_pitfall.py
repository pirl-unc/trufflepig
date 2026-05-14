from trufflepig.tumor_purity import _detect_low_purity_prad_stromal_pitfall


def test_detects_prad_stromal_sarc_pitfall():
    rows = [
        {
            "code": "SARC",
            "signature_score": 0.57,
            "lineage_detection_fraction": 1.0,
            "winning_subtype": "SARC_LMS",
        },
        {
            "code": "PRAD",
            "signature_score": 0.76,
            "lineage_purity": 0.10,
            "lineage_detection_fraction": 0.22,
        },
    ]
    sample_tpm = {
        "KLK2": 42.0,
        "KLK3": 24.0,
        "ACP3": 33.0,
        "KLK4": 13.0,
        "FOXA1": 8.2,
    }
    host = [{"tissue": "prostate", "score": 0.93, "n_genes": 20}]

    pitfall = _detect_low_purity_prad_stromal_pitfall(
        rows,
        sample_tpm,
        host_tissue_details=host,
    )

    assert pitfall
    assert pitfall["recommended_code"] == "PRAD"
    assert "smooth-muscle" in pitfall["message"]
    assert pitfall["prostate_marker_tpm"]["KLK2"] == 42.0


def test_prad_stromal_pitfall_requires_prostate_context():
    rows = [
        {
            "code": "SARC",
            "signature_score": 0.57,
            "lineage_detection_fraction": 1.0,
        },
        {
            "code": "PRAD",
            "signature_score": 0.76,
            "lineage_purity": 0.10,
            "lineage_detection_fraction": 0.22,
        },
    ]
    sample_tpm = {"KLK2": 42.0, "KLK3": 24.0, "ACP3": 33.0, "KLK4": 13.0}
    host = [{"tissue": "smooth_muscle", "score": 0.93, "n_genes": 20}]

    assert (
        _detect_low_purity_prad_stromal_pitfall(
            rows,
            sample_tpm,
            host_tissue_details=host,
        )
        is None
    )
