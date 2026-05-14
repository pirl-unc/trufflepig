"""Unit tests for the TNBC / basal-BRCA misclassification rescue.

Background: HCC1395-style TNBC samples carry a basal cytokeratin program
(KRT5/6A/6B/14 high), suppressed luminal markers (ESR1, PGR), and the
basal-mammary TF FOXC1 — but their broad RNA signature overlaps with
squamous cohorts (ESCA, LUSC, HNSC) because TCGA-BRCA is luminal-
dominated. Without ``--cancer-type BRCA`` they used to silently land on
ESCA. The rescue in :mod:`trufflepig.tumor_purity` fires only when:

  (1) top candidate is squamous-family
  (2) basal cytokeratin program is dominant
  (3) luminal program is off
  (4) FOXC1 is up (basal-mammary TF; cleanly separates from squamous)
  (5) squamous program (TP63 + SOX2) is absent

These tests pin the discriminator profile so a future "loosening" of any
single threshold can't re-enable the false-positive on TCGA-HNSC/LUSC
medians.
"""

from trufflepig.tumor_purity import _detect_tnbc_basal_brca_pattern


def _hcc1395_tpm():
    """Real HCC1395 TPM values (TNBC cell line) for the markers we check."""
    return {
        "KRT5": 502.0,
        "KRT6A": 79.0,
        "KRT6B": 326.0,
        "KRT14": 1111.0,
        "ESR1": 0.14,
        "PGR": 0.01,
        "FOXC1": 29.0,
        "EGFR": 35.0,
        "TP63": 2.4,
        "SOX2": 0.06,
    }


def _hnsc_median_tpm():
    """TCGA-HNSC median TPM values for the same markers (pan_cancer_expression)."""
    return {
        "KRT5": 9519.0,
        "KRT6A": 9068.0,
        "KRT6B": 1460.0,
        "KRT14": 13172.0,
        "ESR1": 0.3,
        "PGR": 0.0,
        "FOXC1": 2.9,
        "EGFR": 42.0,
        "TP63": 122.0,
        "SOX2": 13.4,
    }


def _lusc_median_tpm():
    """TCGA-LUSC median TPM values."""
    return {
        "KRT5": 2614.0,
        "KRT6A": 2591.0,
        "KRT6B": 151.0,
        "KRT14": 175.0,
        "ESR1": 0.1,
        "PGR": 0.0,
        "FOXC1": 1.3,
        "EGFR": 27.0,
        "TP63": 167.0,
        "SOX2": 88.9,
    }


def _cesc_median_tpm():
    """TCGA-CESC median TPM values."""
    return {
        "KRT5": 3678.0,
        "KRT6A": 1832.0,
        "KRT6B": 63.0,
        "KRT14": 971.0,
        "ESR1": 0.2,
        "PGR": 0.0,
        "FOXC1": 0.0,  # CESC median is NaN/missing — treat as 0
        "EGFR": 17.5,
        "TP63": 52.6,
        "SOX2": 10.1,
    }


def test_hcc1395_tnbc_pattern_fires_when_top_is_squamous():
    rows = [{"code": "ESCA"}, {"code": "LUSC"}]
    hint = _detect_tnbc_basal_brca_pattern(rows, _hcc1395_tpm())
    assert hint is not None
    assert hint["recommended_code"] == "BRCA"
    assert hint["recommended_subtype"] == "BRCA_Basal"
    assert "KRT5" in hint["high_basal_keratins"]
    assert "KRT14" in hint["high_basal_keratins"]


def test_hcc1395_pattern_does_not_fire_when_top_is_non_squamous():
    # If the classifier already lands on a non-squamous cohort, the
    # rescue must stay quiet — promoting BRCA over BRCA, COAD, etc.
    # makes no sense.
    rows = [{"code": "BRCA"}]
    assert _detect_tnbc_basal_brca_pattern(rows, _hcc1395_tpm()) is None
    rows = [{"code": "COAD"}]
    assert _detect_tnbc_basal_brca_pattern(rows, _hcc1395_tpm()) is None
    rows = [{"code": "SARC"}]
    assert _detect_tnbc_basal_brca_pattern(rows, _hcc1395_tpm()) is None


def test_hnsc_median_does_not_trip_rescue():
    """HNSC median has basal keratins + luminal off — only FOXC1 saves it."""
    rows = [{"code": "HNSC"}]
    assert _detect_tnbc_basal_brca_pattern(rows, _hnsc_median_tpm()) is None


def test_lusc_median_does_not_trip_rescue():
    rows = [{"code": "LUSC"}]
    assert _detect_tnbc_basal_brca_pattern(rows, _lusc_median_tpm()) is None


def test_cesc_median_does_not_trip_rescue():
    rows = [{"code": "CESC"}]
    assert _detect_tnbc_basal_brca_pattern(rows, _cesc_median_tpm()) is None


def test_high_esr1_blocks_rescue():
    """A luminal-BRCA sample (ESR1 high) must not trigger the basal rescue."""
    tpm = _hcc1395_tpm()
    tpm["ESR1"] = 50.0
    rows = [{"code": "ESCA"}]
    assert _detect_tnbc_basal_brca_pattern(rows, tpm) is None


def test_high_pgr_blocks_rescue():
    tpm = _hcc1395_tpm()
    tpm["PGR"] = 5.0
    rows = [{"code": "ESCA"}]
    assert _detect_tnbc_basal_brca_pattern(rows, tpm) is None


def test_low_foxc1_blocks_rescue():
    """FOXC1 is the cleanest discriminator from broad squamous medians."""
    tpm = _hcc1395_tpm()
    tpm["FOXC1"] = 2.0
    rows = [{"code": "ESCA"}]
    assert _detect_tnbc_basal_brca_pattern(rows, tpm) is None


def test_high_squamous_program_blocks_rescue():
    """A sample with both TP63 and SOX2 strongly up is squamous, not basal-BRCA."""
    tpm = _hcc1395_tpm()
    tpm["TP63"] = 100.0
    tpm["SOX2"] = 50.0
    rows = [{"code": "ESCA"}]
    assert _detect_tnbc_basal_brca_pattern(rows, tpm) is None


def test_only_one_basal_keratin_high_blocks_rescue():
    """KRT14 alone isn't enough — need at least two of the basal program up."""
    tpm = _hcc1395_tpm()
    tpm["KRT5"] = 5.0
    tpm["KRT6A"] = 5.0
    tpm["KRT6B"] = 5.0
    rows = [{"code": "ESCA"}]
    assert _detect_tnbc_basal_brca_pattern(rows, tpm) is None
