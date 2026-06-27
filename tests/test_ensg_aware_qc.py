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
import pytest

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
    """The HK normalizer (pirlygenes back-compat path) must match its housekeeping panel by
    stable Ensembl ID, not by symbol. Build a sample from real pirlygenes HK panel genes but
    give them WRONG symbols — if matching fell back to symbols, nothing would match and the
    non-HK gene wouldn't be normalized by the HK geometric mean."""
    from pirlygenes.gene_sets_cancer import housekeeping_gene_ids

    hk_ids = sorted(housekeeping_gene_ids())[:5]  # deterministic; clears the min-HK-genes guard
    myc = "ENSG00000136997"
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": hk_ids + [myc],
            "Symbol": [f"WRONG_{i}" for i in range(len(hk_ids))] + ["ALSO_WRONG"],
            "S_TPM": [100.0] * len(hk_ids) + [400.0],
        }
    )
    out, record = tpm_to_housekeeping_normalized(df)
    assert record["applied"], record.get("reason")
    # HK geomean of the five 100-TPM genes is ~100 → the non-HK gene (400) normalizes to ~4×
    # the HK genes' normalized value. This only happens if the HK genes were found via their
    # Ensembl IDs despite the wrong symbols (a symbol fallback would match nothing).
    hk_norm = out.loc[out["Ensembl_Gene_ID"] == hk_ids[0], "S_TPM"].item()
    myc_norm = out.loc[out["Ensembl_Gene_ID"] == myc, "S_TPM"].item()
    assert myc_norm / hk_norm == pytest.approx(4.0, rel=1e-2)
