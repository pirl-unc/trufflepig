import pandas as pd

from trufflepig.common import build_sample_tpm_by_symbol

# Backward-compat: the underscore alias is still importable from tumor_purity
from trufflepig.tumor_purity import _build_sample_tpm_by_symbol


class _NoDeepcopy:
    def __deepcopy__(self, memo):
        raise AssertionError("unexpected deepcopy of DataFrame.attrs")


def test_build_sample_tpm_by_symbol_does_not_deepcopy_attrs(monkeypatch):
    df = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000000001.5", "ENSG00000000002"],
            "gene_display_name": ["GENE1", "GENE2"],
            "TPM": [1.0, 2.5],
        }
    )
    df.attrs["transcript_expression"] = _NoDeepcopy()

    ref = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000000001", "ENSG00000000002"],
            "Symbol": ["GENE1", "GENE2"],
        }
    )
    # Patch at the source module — common.py lazy-imports from trufflepig.reference
    # (expression accessors moved off pirlygenes in trufflepig#23).
    monkeypatch.setattr(
        "trufflepig.reference.pan_cancer_expression", lambda: ref
    )

    out = build_sample_tpm_by_symbol(df)
    assert out == {"GENE1": 1.0, "GENE2": 2.5}

    # The tumor_purity delegate should produce the same result
    out2 = _build_sample_tpm_by_symbol(df)
    assert out2 == {"GENE1": 1.0, "GENE2": 2.5}
