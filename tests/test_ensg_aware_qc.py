"""ENSG-first classification + id_col plumbing through normalize_expression.

Pins:
  - classify_gene_qc accepts ensembl_id and prefers the pirlygenes
    static table over the symbol regex.
  - Versioned IDs (ENSG00000251562.5) are stripped before lookup.
  - Historic ENSG IDs for the same symbol (e.g. ENSG00000278217 for
    MALAT1 in releases 77-87) still classify correctly.
  - normalize_expression honors id_col, passing ENSG through to the
    classifier even when the symbol is missing or unconventional.
  - tpm_to_housekeeping_normalized matches the HK panel by ENSG when
    id_col is present (more robust than symbol matching).
"""

import pandas as pd

from trufflepig.expression_qc import classify_gene_qc
from trufflepig.expression_normalize import (
    normalize_expression,
    tpm_to_housekeeping_normalized,
)


def test_classify_gene_qc_ensembl_id_lookup():
    qc = classify_gene_qc(ensembl_id="ENSG00000251562")
    assert qc.group == "polyadenylation_bias_lncrna"


def test_classify_gene_qc_strips_version_suffix():
    qc = classify_gene_qc(ensembl_id="ENSG00000251562.5")
    assert qc.group == "polyadenylation_bias_lncrna"


def test_classify_gene_qc_historic_ensembl_id():
    # MALAT1 had ENSG00000278217 in releases 77-87 (biotype misc_RNA).
    # ENSG-first lookup must still classify it correctly even though
    # this ID is gone from current Ensembl.
    qc = classify_gene_qc(ensembl_id="ENSG00000278217")
    assert qc.group == "polyadenylation_bias_lncrna"


def test_classify_gene_qc_mt_dna_by_ensembl_id():
    qc = classify_gene_qc(ensembl_id="ENSG00000198804")  # MT-CO1
    assert qc.group == "mt_dna"


def test_classify_gene_qc_unknown_ensg_falls_back_to_symbol_regex():
    qc = classify_gene_qc("MT-CO1", ensembl_id="ENSG99999999999")
    assert qc.group == "mt_dna"


def test_normalize_expression_uses_id_col_when_present():
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [
                "ENSG00000251562",   # MALAT1
                "ENSG00000198804",   # MT-CO1
                "ENSG00000075624",   # ACTB
                "ENSG00000111640",   # GAPDH
            ],
            # Deliberately wrong symbols so symbol-regex would miss the
            # technical-RNA features. ENSG path should still catch them.
            "Symbol": ["FOO1", "FOO2", "ACTB", "GAPDH"],
            "TPM": [500.0, 500.0, 100.0, 100.0],
        }
    )
    out, record = normalize_expression(df)
    assert record["applied"]
    s = out.set_index("Ensembl_Gene_ID")["TPM"]
    assert s["ENSG00000251562"] == 0.0
    assert s["ENSG00000198804"] == 0.0
    # ACTB / GAPDH absorbed the rescaled mass.
    assert out["TPM"].sum() == 1200.0


def test_normalize_expression_without_id_col_falls_back_to_symbol():
    df = pd.DataFrame(
        {
            "Symbol": ["MALAT1", "MT-CO1", "ACTB", "GAPDH"],
            "TPM": [500.0, 500.0, 100.0, 100.0],
        }
    )
    out, record = normalize_expression(df)
    assert record["applied"]
    s = out.set_index("Symbol")["TPM"]
    assert s["MALAT1"] == 0.0
    assert s["MT-CO1"] == 0.0


def test_tpm_to_housekeeping_normalized_matches_via_ensembl_id():
    # ACTB ENSG00000075624, GAPDH ENSG00000111640, MYC ENSG00000136997
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [
                "ENSG00000075624",
                "ENSG00000111640",
                "ENSG00000136997",
            ],
            # Symbols deliberately wrong; ENSG path should still match
            # the HK panel by stable ID.
            "Symbol": ["NOT_ACTB", "NOT_GAPDH", "MYC"],
            "S_TPM": [100.0, 100.0, 200.0],
        }
    )
    out, record = tpm_to_housekeeping_normalized(df)
    assert record["match_mode"] == "ensembl_id"
    assert record["panel_present_in_table"] >= 2
    assert record["applied"]
