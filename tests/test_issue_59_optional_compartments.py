"""Regression tests for #59 items 2-4 — gated optional compartments.

New compartments added to the NNLS template conditionally based on
sample signal:

- ``adipocyte`` — breast / retroperitoneal lipo samples (BRCA +
  SARC solid_primary / met_soft_tissue)
- ``schwann`` — PRAD / PAAD / HNSC / CHOL with perineural invasion
- ``erythroid`` — any solid template when hemoglobin signal is high

Contract: with no gate-firing signal, behavior is byte-identical to
the pre-#59 path.
"""

from trufflepig.decomposition.templates import (
    OPTIONAL_COMPARTMENT_GATES,
    _detect_optional_compartments,
    get_template_components,
)


# ── Gate schema ──────────────────────────────────────────────────────


def test_gate_config_covers_all_expected_compartments():
    expected = {"adipocyte", "schwann", "erythroid_solid"}
    assert set(OPTIONAL_COMPARTMENT_GATES.keys()) >= expected


def test_every_gate_has_required_fields():
    for name, gate in OPTIONAL_COMPARTMENT_GATES.items():
        assert "markers" in gate and gate["markers"], f"{name} lacks markers"
        assert gate["min_tpm_sum"] > 0, f"{name} threshold must be positive"
        assert "templates" in gate, f"{name} lacks templates allowlist"
        assert isinstance(gate["templates"], set), f"{name} templates must be a set"


def test_every_gate_compartment_has_hpa_mapping():
    """Gated compartment → HPA-row mapping must exist in
    ``signature.COMPONENT_TO_HPA`` so the NNLS can build its
    reference column."""
    from trufflepig.decomposition.signature import COMPONENT_TO_HPA
    from trufflepig.decomposition.templates import _GATE_KEY_TO_COMPONENT

    for gate_key in OPTIONAL_COMPARTMENT_GATES:
        compartment = _GATE_KEY_TO_COMPONENT.get(gate_key, gate_key)
        assert compartment in COMPONENT_TO_HPA, (
            f"Gate {gate_key} maps to compartment '{compartment}' but "
            "no HPA reference is registered"
        )


# ── Detection ────────────────────────────────────────────────────────


def test_empty_sample_returns_empty_list():
    assert (
        _detect_optional_compartments(
            {}, cancer_type="BRCA", template_name="solid_primary"
        )
        == []
    )


def test_sample_without_gate_genes_returns_empty():
    sample = {"TP53": 100.0, "MYC": 200.0}
    assert (
        _detect_optional_compartments(
            sample,
            cancer_type="BRCA",
            template_name="solid_primary",
        )
        == []
    )


def test_adipocyte_fires_for_brca_solid_with_high_signal():
    sample = {"ADIPOQ": 60.0, "FABP4": 40.0, "PLIN1": 20.0}  # sum=120 >> 50
    detected = _detect_optional_compartments(
        sample,
        cancer_type="BRCA",
        template_name="solid_primary",
    )
    assert "adipocyte" in detected


def test_adipocyte_does_not_fire_below_threshold():
    sample = {"ADIPOQ": 10.0}  # well below 50
    detected = _detect_optional_compartments(
        sample,
        cancer_type="BRCA",
        template_name="solid_primary",
    )
    assert "adipocyte" not in detected


def test_adipocyte_does_not_fire_on_wrong_cancer_type():
    """ADIPOQ etc. present but cancer_type isn't on the allowlist."""
    sample = {"ADIPOQ": 100.0, "FABP4": 100.0, "PLIN1": 100.0}
    detected = _detect_optional_compartments(
        sample,
        cancer_type="COAD",
        template_name="solid_primary",
    )
    assert "adipocyte" not in detected


def test_schwann_fires_for_prad_with_perineural_signal():
    sample = {"MPZ": 20.0, "PMP22": 15.0, "S100B": 10.0}  # sum=45 > 30
    detected = _detect_optional_compartments(
        sample,
        cancer_type="PRAD",
        template_name="solid_primary",
    )
    assert "schwann" in detected


def test_schwann_does_not_fire_for_luad():
    """LUAD isn't on the Schwann allowlist — perineural invasion is
    not characteristic of lung adenocarcinoma."""
    sample = {"MPZ": 100.0, "PMP22": 100.0}
    detected = _detect_optional_compartments(
        sample,
        cancer_type="LUAD",
        template_name="solid_primary",
    )
    assert "schwann" not in detected


def test_erythroid_fires_on_any_solid_cancer_with_hemoglobin():
    sample = {"HBA1": 40.0, "HBA2": 40.0, "HBB": 50.0, "ALAS2": 10.0}  # sum=140 > 100
    # Any solid cancer type works — no cancer allowlist for erythroid.
    for cancer in ("COAD", "LUAD", "BRCA", "PAAD"):
        detected = _detect_optional_compartments(
            sample,
            cancer_type=cancer,
            template_name="solid_primary",
        )
        assert "erythroid" in detected, f"{cancer} should detect erythroid"


def test_erythroid_does_not_fire_on_heme_template():
    """Heme templates already carry erythroid as a first-class
    compartment; the gate targets solid templates only."""
    sample = {"HBA1": 500.0, "HBA2": 500.0, "HBB": 500.0}
    detected = _detect_optional_compartments(
        sample,
        cancer_type="LAML",
        template_name="heme_marrow",
    )
    assert "erythroid" not in detected


# ── Template-expansion contract ──────────────────────────────────────


def test_default_template_components_unchanged_when_no_detection():
    """With ``detected_compartments=None`` or empty, the component
    list is identical to the pre-#59 path."""
    before = get_template_components("solid_primary", "BRCA")
    with_none = get_template_components(
        "solid_primary",
        "BRCA",
        detected_compartments=None,
    )
    with_empty = get_template_components(
        "solid_primary",
        "BRCA",
        detected_compartments=[],
    )
    assert before == with_none == with_empty


def test_detected_compartments_appended_after_matched_normal():
    components = get_template_components(
        "solid_primary",
        "BRCA",
        detected_compartments=["adipocyte"],
    )
    assert "adipocyte" in components
    # Matched-normal lands before the detected compartment.
    assert components.index("matched_normal_breast") < components.index("adipocyte")


def test_duplicate_compartment_not_appended_twice():
    """If a template already carries a compartment and it's also
    detected, it must appear only once."""
    components = get_template_components(
        "heme_marrow",
        None,
        detected_compartments=["erythroid"],
    )
    assert components.count("erythroid") == 1


# ── End-to-end through the engine ────────────────────────────────────


def test_detection_signature_path_survives_through_engine_api():
    """Smoke test — the detection hook doesn't blow up when called
    on a real sample through the decomposition engine."""
    import pandas as pd
    from trufflepig.reference import pan_cancer_expression

    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    sample_tpm = dict(
        zip(ref["Symbol"].astype(str), ref["nTPM_prostate"].astype(float))
    )
    # Inject Schwann markers so the gate should fire on PRAD.
    sample_tpm["MPZ"] = 50.0
    sample_tpm["PMP22"] = 40.0
    detected = _detect_optional_compartments(
        sample_tpm,
        cancer_type="PRAD",
        template_name="solid_primary",
    )
    assert "schwann" in detected
    # Engine's get_template_components path must accept it and
    # include the compartment.
    components = get_template_components(
        "solid_primary",
        "PRAD",
        detected_compartments=detected,
    )
    assert "schwann" in components

    _ = pd  # keep import noise at bay when the smoke test is the only user
