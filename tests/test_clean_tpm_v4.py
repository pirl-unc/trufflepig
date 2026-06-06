"""Clean-TPM v4 two-compartment adoption.

pirlygenes clean_tpm_v4 pins the technical-RNA + ribosomal compartment to a
fixed fraction (25%) of the 1e6 budget instead of zeroing it. ``assert_clean_tpm``
must accept that (``technical_fraction`` mode) while still rejecting genuinely
un-normalized, technical-dominant input — and every registry code's v4 reference
must load again (the regression these guard against: a v4 reference frame was
rejected by the old "technical ≈ 0" assertion, silently dropping the whole
observed-bulk layer).
"""

import pandas as pd
import pytest

from trufflepig.clean_tpm import assert_clean_tpm


def _frame(rows):
    # rows: list of (symbol, ensembl_id, value)
    return pd.DataFrame(
        [{"Symbol": s, "Ensembl_Gene_ID": e, "TPM": v} for s, e, v in rows]
    )


def test_legacy_mode_rejects_nonzero_technical():
    # MT-CO2 carries TPM and we assert the legacy zeroed contract -> raises.
    df = _frame([("MT-CO2", "ENSG00000198712", 250_000.0), ("EPCAM", "ENSG00000119888", 750_000.0)])
    with pytest.raises(ValueError, match="nonzero technical-RNA"):
        assert_clean_tpm(df, value_cols=["TPM"], context="legacy")


def test_v4_mode_accepts_technical_at_the_fixed_fraction():
    # Technical at ~25% is exactly v4 -> must NOT raise under fixed-fraction mode.
    df = _frame([("MT-CO2", "ENSG00000198712", 250_000.0), ("EPCAM", "ENSG00000119888", 750_000.0)])
    assert_clean_tpm(df, value_cols=["TPM"], context="v4", technical_fraction=0.25)


def test_v4_mode_still_rejects_technical_dominant_raw_input():
    # 60% technical exceeds 0.25 + 0.15 slack -> raises even in v4 mode.
    df = _frame([("MT-CO2", "ENSG00000198712", 600_000.0), ("EPCAM", "ENSG00000119888", 400_000.0)])
    with pytest.raises(ValueError, match="exceeds the v4 bound"):
        assert_clean_tpm(df, value_cols=["TPM"], context="v4", technical_fraction=0.25)


@pytest.mark.parametrize(
    "code,expected_source",
    [
        ("MM", "MMRF_COMMPASS"),
        ("ADCC", "GSE294016_BARTL_2025_SGC"),
        ("MTC", "GSE32662_PRINGLE_2012_MTC"),
        ("NET_LUNG", "DRMETRICS_ALCALA_2019_LNEN"),
        ("SARC_ASPS", "TREEHOUSE_POLYA_25_01"),
        ("CLL", "CLLMAP_2022"),
    ],
)
def test_v4_observed_bulk_references_load_directly(code, expected_source):
    # Regression: these resolved to None / a parent fallback when the v4 frame
    # was rejected by the old assertion. They must self-reference again.
    from trufflepig.analyze import effective_expression_reference

    ref = effective_expression_reference(code)
    assert ref is not None, f"{code} lost its expression reference under v4"
    assert ref.reference_code == code
    assert ref.source == expected_source
