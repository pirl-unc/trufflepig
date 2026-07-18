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
    # Legacy strict-zero is now opt-in via technical_fraction=None; the default is
    # the v4 fixed-fraction bound consumed from pirlygenes.
    with pytest.raises(ValueError, match="nonzero technical-RNA"):
        assert_clean_tpm(df, value_cols=["TPM"], context="legacy", technical_fraction=None)


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


def test_normalize_to_reference_space_conforms_v4_to_fixed_fraction():
    """A sample is conformed by deferring to pirlygenes' clean-TPM transform: the
    technical compartment is PINNED to its fixed fraction (not zeroed), the budget
    is 1e6, and biological rank order is preserved (#74). trufflepig asserts no
    specific fraction here — that's pirlygenes' contract."""
    from trufflepig.clean_tpm import (
        normalize_to_reference_space,
        resolve_gene_columns,
        technical_rna_mask,
    )

    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [
                "ENSG00000210082",  # MT-RNR2 (mt-DNA, technical)
                "ENSG00000089157",  # RPLP0 (ribosomal-protein compartment)
                "ENSG00000119888",  # EPCAM
                "ENSG00000108821",  # COL1A1
                "ENSG00000075624",  # ACTB
            ],
            "Symbol": ["MT-RNR2", "RPLP0", "EPCAM", "COL1A1", "ACTB"],
            "TPM": [150000.0, 100000.0, 50000.0, 200000.0, 500000.0],
        }
    )
    label, idc = resolve_gene_columns(df)
    out = normalize_to_reference_space(df, value_cols=["TPM"], label_col=label, id_col=idc)

    mask = technical_rna_mask(out, label_col=label, id_col=idc)
    assert float(out.loc[mask, "TPM"].sum()) > 0.0  # PINNED, not zeroed (new contract)
    assert float(out["TPM"].sum()) == pytest.approx(1e6)  # renormalized to the budget
    # biological ordering preserved (ACTB > COL1A1 > EPCAM)
    bio = out.loc[~mask].set_index("Symbol")["TPM"]
    assert bio["ACTB"] > bio["COL1A1"] > bio["EPCAM"]


def test_normalize_to_reference_space_noop_without_technical_compartment():
    """No technical genes -> no-op (don't force-rescale a partial/synthetic frame)."""
    from trufflepig.clean_tpm import normalize_to_reference_space

    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000119888", "ENSG00000108821"],
            "Symbol": ["EPCAM", "COL1A1"],
            "TPM": [10.0, 40.0],  # deliberately not summing to 1e6
        }
    )
    out = normalize_to_reference_space(df, value_cols=["TPM"])
    assert list(out["TPM"]) == [10.0, 40.0]
