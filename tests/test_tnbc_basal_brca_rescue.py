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
        "MIA": 181.0,
        "GABRP": 6.92,
        "UPK1A": 0.0,
        "UPK1B": 0.0,
        "UPK2": 0.0,
        "UPK3A": 0.0,
        "UPK3B": 0.0,
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
        "MIA": 0.7,
        "GABRP": 1.0,
        "UPK1A": 0.2,
        "UPK1B": 0.3,
        "UPK2": 0.4,
        "UPK3A": 0.0,
        "UPK3B": 3.7,
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
        "MIA": 0.3,
        "GABRP": 0.8,
        "UPK1A": 0.2,
        "UPK1B": 21.7,
        "UPK2": 0.6,
        "UPK3A": 0.1,
        "UPK3B": 1.2,
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
        "MIA": 0.6,
        "GABRP": 0.4,
        "UPK1A": 0.0,
        "UPK1B": 0.0,
        "UPK2": 0.4,
        "UPK3A": 0.0,
        "UPK3B": 1.5,
    }


def _blca_basal_like_tpm():
    """Synthesized basal-MIBC profile (KRT5/6/14 high, urothelial up).

    TCGA-BLCA basal-subtype samples carry the keratin program but also
    retain urothelial differentiation markers (UPK1A/1B/2/3A/3B) at
    cohort median 28-92 TPM each. Mammary-positive markers (MIA/GABRP)
    are near zero. This pseudo-sample tests the urothelial-panel guard.
    """
    return {
        "KRT5": 1500.0,
        "KRT6A": 1200.0,
        "KRT6B": 200.0,
        "KRT14": 800.0,
        "ESR1": 0.5,
        "PGR": 0.0,
        "FOXC1": 12.0,  # bypass FOXC1 gate — defense in depth needed
        "EGFR": 40.0,
        "TP63": 25.0,  # below squamous-strong threshold
        "SOX2": 3.0,
        "MIA": 3.0,  # bypass mammary-positive gate
        "GABRP": 0.0,
        "UPK1A": 46.7,
        "UPK1B": 62.1,
        "UPK2": 92.7,
        "UPK3A": 28.6,
        "UPK3B": 50.8,
    }


def _hypothetical_foxc1_high_hnsc():
    """Adversarial: HNSC-like keratin profile with FOXC1 forced high.

    The point of MIA/GABRP positive-confirmation: even if FOXC1 is
    elevated in some atypical squamous variant, the absence of mammary-
    positive markers should still prevent a misfire.
    """
    return {
        "KRT5": 5000.0,
        "KRT6A": 5000.0,
        "KRT6B": 500.0,
        "KRT14": 6000.0,
        "ESR1": 0.2,
        "PGR": 0.0,
        "FOXC1": 30.0,  # forced high
        "EGFR": 40.0,
        "TP63": 25.0,  # tuned below the 30/5 squamous double-gate
        "SOX2": 3.0,
        # Squamous: MIA and GABRP near zero — the load-bearing
        # positive-confirmation discriminator from this gate.
        "MIA": 0.5,
        "GABRP": 0.5,
        "UPK1A": 0.0,
        "UPK1B": 0.0,
        "UPK2": 0.0,
        "UPK3A": 0.0,
        "UPK3B": 0.0,
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


def test_basal_mammary_positive_markers_required():
    """Without MIA or GABRP confirmation, the rescue must stay quiet —
    defense against squamous variants that happen to have FOXC1 elevated."""
    rows = [{"code": "HNSC"}]
    assert _detect_tnbc_basal_brca_pattern(rows, _hypothetical_foxc1_high_hnsc()) is None
    # Sanity: re-enable mammary positive markers and the rescue should fire.
    tpm = _hypothetical_foxc1_high_hnsc()
    tpm["MIA"] = 5.0
    # And turn off the very-high keratin / very-low squamous-TF to leave
    # the other gates passable; HNSC-like high keratins do pass gate 2.
    rows = [{"code": "HNSC"}]
    hint = _detect_tnbc_basal_brca_pattern(rows, tpm)
    assert hint is not None, "fixture intended to fire once MIA is up"


def test_either_mia_or_gabrp_satisfies_positive_gate():
    """MIA alone satisfies the gate; GABRP alone also satisfies."""
    rows = [{"code": "ESCA"}]
    tpm_mia_only = _hcc1395_tpm()
    tpm_mia_only["GABRP"] = 0.0
    assert _detect_tnbc_basal_brca_pattern(rows, tpm_mia_only) is not None
    tpm_gabrp_only = _hcc1395_tpm()
    tpm_gabrp_only["MIA"] = 0.0
    tpm_gabrp_only["GABRP"] = 5.0
    assert _detect_tnbc_basal_brca_pattern(rows, tpm_gabrp_only) is not None


def test_urothelial_panel_blocks_rescue():
    """Basal-MIBC samples carry the keratin program but also urothelial
    differentiation. They must not be promoted to BRCA — TCGA-BLCA basal
    is its own cohort with distinct treatment context."""
    rows = [{"code": "ESCA"}]
    assert _detect_tnbc_basal_brca_pattern(rows, _blca_basal_like_tpm()) is None
    # Confirm the gate is what's blocking it: turn off urothelial panel
    # and the sample (already passing other gates) becomes basal-BRCA-like.
    tpm = _blca_basal_like_tpm()
    for sym in ("UPK1A", "UPK1B", "UPK2", "UPK3A"):
        tpm[sym] = 0.0
    # The synthesized BLCA fixture has MIA=3 / GABRP=0 already passing
    # the basal-positive gate, so with urothelial off the rescue can fire.
    assert _detect_tnbc_basal_brca_pattern(rows, tpm) is not None


def test_hint_payload_carries_diagnostic_fields():
    """Reporting needs the full marker readout, not just the verdict."""
    rows = [{"code": "ESCA"}]
    hint = _detect_tnbc_basal_brca_pattern(rows, _hcc1395_tpm())
    assert hint is not None
    assert hint["competing_top_code"] == "ESCA"
    assert hint["foxc1_tpm"] == 29.0
    assert hint["basal_positive_tpm"]["MIA"] == 181.0
    assert hint["basal_positive_tpm"]["GABRP"] == 6.92
    assert hint["urothelial_panel_sum_tpm"] == 0.0
    assert hint["keratin_tpm"]["KRT14"] == 1111.0
    # Message includes the load-bearing values so reports stay
    # interpretable even when downstream only forwards the message.
    assert "FOXC1" in hint["message"]
    assert "MIA" in hint["message"]
    assert "ESR1" in hint["message"]
