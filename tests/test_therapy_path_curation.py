"""Audit structured therapy-path curation for cancer-key-genes rows."""

import pandas as pd

from trufflepig._data import DATA_DIR as _DATA_DIR
from trufflepig.reporting import (
    KNOWN_UPSTREAM_NONCONFORMING_THERAPY_ROWS,
    THERAPY_PATH_TIERS,
    expression_independent_indication,
    indication_biomarker,
    hla_eligibility_context,
    hla_restricted_target_supported,
    subtype_curation_scope_note,
    target_hla_eligibility,
    filter_current_therapy_targets,
    therapy_filter_note,
    therapy_withdrawal_note,
    therapy_path_context,
    therapy_path_rank,
    therapy_path_tier,
)


_PHASE_BY_TIER = {
    "approved_standard": {"approved"},
    "approved_indication_matched": {"approved"},
    "approved_later_line": {"approved"},
    "late_clinical": {"phase_3"},
    "investigational_biomarker_matched": {"phase_1", "phase_2"},
    "trial_follow_up": {"phase_1", "phase_2"},
    "preclinical": {"preclinical"},
    "off_label": {"off_label"},
}


def _target_rows():
    df = pd.read_csv(_DATA_DIR / "cancer-key-genes.csv").fillna("")
    return df[df["role"].astype(str).str.strip() == "target"].copy()


def _is_quarantined(row) -> bool:
    return (
        str(row.cancer_code),
        str(row.symbol),
    ) in KNOWN_UPSTREAM_NONCONFORMING_THERAPY_ROWS


def _row_nonconformance(row) -> str | None:
    """Return a reason string if a target row violates the controlled
    tier/phase vocabulary, else None."""
    tier = str(row.treatment_path_tier)
    if tier not in THERAPY_PATH_TIERS:
        return f"tier '{tier}' not in controlled vocabulary"
    phase = str(row.phase)
    if phase not in _PHASE_BY_TIER.get(tier, set()):
        return f"phase '{phase}' incompatible with tier '{tier}'"
    return None


def test_target_rows_have_structured_treatment_path_curation():
    targets = _target_rows()
    assert len(targets) >= 300
    for column in ("treatment_path_tier", "line_of_therapy", "eligibility_note"):
        missing = targets[targets[column].astype(str).str.strip().eq("")]
        assert missing.empty, f"{column} missing for target rows: " + ", ".join(
            f"{row.cancer_code}:{row.symbol}:{row.agent}"
            for row in missing.head(10).itertuples()
        )

    # Quarantined upstream rows (see KNOWN_UPSTREAM_NONCONFORMING_THERAPY_ROWS)
    # are exempt; the contract still holds for every row trufflepig governs.
    invalid = sorted(
        {
            str(row.treatment_path_tier)
            for row in targets.itertuples()
            if not _is_quarantined(row)
            and str(row.treatment_path_tier) not in THERAPY_PATH_TIERS
        }
    )
    assert not invalid, f"invalid treatment_path_tier values: {', '.join(invalid)}"


def test_treatment_path_tier_is_phase_compatible():
    targets = _target_rows()
    bad = []
    for row in targets.itertuples():
        if _is_quarantined(row):
            continue
        phase = str(row.phase)
        tier = str(row.treatment_path_tier)
        if phase not in _PHASE_BY_TIER.get(tier, set()):
            bad.append(f"{row.cancer_code}:{row.symbol}:{row.agent}:{phase}->{tier}")
    assert not bad, "phase/tier mismatch: " + ", ".join(bad[:20])


def test_upstream_nonconformance_quarantine_is_current():
    """The quarantine must list exactly the rows that are genuinely non-
    conforming upstream — no stale entries (a row pirlygenes has since fixed,
    or one no longer present) silently suppressing the contract."""
    targets = _target_rows()
    present = {
        (str(row.cancer_code), str(row.symbol)): row for row in targets.itertuples()
    }
    for code, sym in KNOWN_UPSTREAM_NONCONFORMING_THERAPY_ROWS:
        row = present.get((code, sym))
        assert row is not None, (
            f"stale quarantine: {code}:{sym} is no longer a target row — "
            "remove it from KNOWN_UPSTREAM_NONCONFORMING_THERAPY_ROWS"
        )
        assert _row_nonconformance(row) is not None, (
            f"stale quarantine: {code}:{sym} now conforms upstream — "
            "remove it from KNOWN_UPSTREAM_NONCONFORMING_THERAPY_ROWS"
        )


def test_target_row_sources_are_present_for_most_curation_rows():
    targets = _target_rows()
    sources = targets["source"].astype(str).str.strip()
    assert sources.ne("").mean() >= 0.95
    assert sources[sources.ne("")].str.contains("PMID:", regex=False).all()


def test_withdrawn_disease_specific_rows_are_filtered_from_reports():
    targets = pd.DataFrame(
        [
            {
                "cancer_code": "BLCA",
                "symbol": "TACSTD2",
                "agent": "sacituzumab govitecan",
                "indication": "advanced urothelial cancer",
            },
            {
                "cancer_code": "BLCA",
                "symbol": "NECTIN4",
                "agent": "enfortumab vedotin",
                "indication": "advanced urothelial cancer",
            },
            {
                "cancer_code": "BRCA",
                "symbol": "TACSTD2",
                "agent": "sacituzumab govitecan",
                "indication": "metastatic HR-positive HER2-negative breast cancer",
            },
        ]
    )

    note = therapy_withdrawal_note(targets.iloc[0])
    filtered = filter_current_therapy_targets(targets)

    assert "withdrawn urothelial" in note
    assert list(filtered["agent"]) == ["enfortumab vedotin", "sacituzumab govitecan"]
    assert list(filtered["cancer_code"]) == ["BLCA", "BRCA"]


def test_miscited_osteosarcoma_ganitumab_row_is_filtered_from_reports():
    targets = pd.DataFrame(
        [
            {
                "cancer_code": "SARC_OS",
                "symbol": "IGF1R",
                "agent": "ganitumab + chemo",
                "indication": "metastatic OS",
            },
            {
                "cancer_code": "SARC_OS",
                "symbol": "VEGFA",
                "agent": "cabozantinib",
                "indication": "R/R OS",
            },
        ]
    )

    assert "osteosarcoma" in therapy_filter_note(targets.iloc[0])
    filtered = filter_current_therapy_targets(targets)
    assert list(filtered["symbol"]) == ["VEGFA"]


def test_adcc_lenvatinib_is_not_fgf2_expression_gated():
    row = {
        "cancer_code": "ADCC",
        "symbol": "FGF2",
        "agent": "lenvatinib",
        "agent_class": "small_molecule",
        "phase": "phase_2",
        "indication": "advanced ADCC",
        "rationale": "Multikinase TKI - modest activity in advanced ADCC",
    }

    assert indication_biomarker(row) == "histology_only"
    assert expression_independent_indication(row) is True


def test_reports_prefer_explicit_treatment_path_tier_over_rationale_text():
    later_line_row = {
        "symbol": "TEST1",
        "agent": "example ADC",
        "agent_class": "ADC",
        "phase": "approved",
        "indication": "example cancer",
        "rationale": "frontline standard backbone wording should not win",
        "treatment_path_tier": "approved_later_line",
        "eligibility_note": "confirm prior therapy",
    }
    assert therapy_path_tier(later_line_row) == "approved_later_line"
    assert therapy_path_rank(later_line_row) == 2
    assert "approved later-line pathway" in therapy_path_context(later_line_row)
    assert "guideline-standard" not in therapy_path_context(later_line_row)

    standard_row = {
        "symbol": "TEST2",
        "agent": "example antibody",
        "agent_class": "antibody",
        "phase": "approved",
        "indication": "example cancer",
        "rationale": "",
        "treatment_path_tier": "approved_standard",
        "eligibility_note": "confirm first-line eligibility",
    }
    assert therapy_path_tier(standard_row) == "approved_standard"
    assert therapy_path_rank(standard_row) == 0
    assert "guideline-standard approved pathway" in therapy_path_context(standard_row)


def test_treatment_path_context_dedupes_curated_note_prefix():
    row = {
        "symbol": "TEST3",
        "agent": "example TCE",
        "agent_class": "TCE",
        "phase": "phase_2",
        "indication": "example cancer",
        "rationale": "",
        "treatment_path_tier": "trial_follow_up",
        "eligibility_note": "clinical-trial follow-up; not default standard",
    }
    context = therapy_path_context(row)
    assert context == "clinical-trial follow-up; not default standard"
    assert "clinical-trial follow-up; clinical-trial follow-up" not in context


def test_hla_restricted_therapy_rows_use_supplied_hla_gate():
    row = {
        "symbol": "TEST4",
        "agent": "example TCR-T",
        "agent_class": "TCR-T",
        "phase": "phase_1",
        "indication": "HLA-A*02+ target-positive solid tumors",
        "rationale": "",
        "treatment_path_tier": "trial_follow_up",
        "eligibility_note": "clinical-trial follow-up; not default standard",
    }
    matched = {"analysis_constraints": {"hla_types": ["A*02:01"]}}
    mismatched = {"analysis_constraints": {"hla_types": ["A*24:02"]}}

    assert "HLA match" in hla_eligibility_context(row, analysis=matched)
    assert hla_restricted_target_supported(row, analysis=matched) is True
    assert "HLA mismatch" in hla_eligibility_context(row, analysis=mismatched)
    assert hla_restricted_target_supported(row, analysis=mismatched) is False


def test_low_resolution_hla_does_not_match_exact_allele_requirement():
    row = {
        "symbol": "TEST5",
        "agent": "example pMHC bispecific",
        "agent_class": "bispecific",
        "phase": "approved",
        "indication": "HLA-A*02:01 target-positive tumors",
        "rationale": "",
        "treatment_path_tier": "approved_biomarker_matched",
        "eligibility_note": "confirm biomarker/indication-specific eligibility",
    }
    low_resolution = {"analysis_constraints": {"hla_types": ["A*02"]}}

    eligibility = target_hla_eligibility(row, analysis=low_resolution)
    context = hla_eligibility_context(row, analysis=low_resolution)

    assert eligibility["status"] == "insufficient_resolution"
    assert eligibility["matched_supplied"] == "A*02"
    assert eligibility["matched_required"] == "A*02:01"
    assert "HLA unresolved" in context
    assert "HLA match" not in context
    assert hla_restricted_target_supported(row, analysis=low_resolution) is True


def test_low_resolution_hla_matches_broad_requirement_only():
    row = {
        "symbol": "TEST6",
        "agent": "example broad TCR-T",
        "agent_class": "TCR-T",
        "phase": "phase_1",
        "indication": "HLA-A*02+ target-positive solid tumors",
        "rationale": "",
        "treatment_path_tier": "trial_follow_up",
        "eligibility_note": "clinical-trial follow-up; not default standard",
    }
    low_resolution = {"analysis_constraints": {"hla_types": ["A*02"]}}

    assert target_hla_eligibility(row, analysis=low_resolution)["status"] == "matched"
    assert "HLA match" in hla_eligibility_context(row, analysis=low_resolution)


def test_trial_ids_are_not_misread_as_hla_restrictions():
    row = {
        "symbol": "IGF1R",
        "agent": "ganitumab + chemo",
        "agent_class": "antibody",
        "phase": "phase_2",
        "indication": "metastatic OS",
        "rationale": "IGF1R inhibitor - SARC021 + AEWS1221 trials",
        "treatment_path_tier": "trial_follow_up",
        "eligibility_note": "clinical-trial follow-up; not default standard",
    }

    assert target_hla_eligibility(row)["status"] == "not_hla_restricted"
    assert hla_eligibility_context(row) == ""


def test_subtype_scope_note_avoids_duplicate_parent_label():
    note = subtype_curation_scope_note(
        "SARC",
        panel_subtype="synovial_sarcoma",
        base_code="SARC",
        noun="therapy evidence",
    )
    assert "synovial sarcoma-specific therapy evidence" in note
    assert "synovial sarcoma sarcoma" not in note
