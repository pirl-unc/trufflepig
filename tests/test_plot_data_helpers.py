import pandas as pd
import pytest

import trufflepig.plot_data_helpers as pdh


class _NoDeepcopy:
    def __deepcopy__(self, memo):
        raise AssertionError("unexpected deepcopy of DataFrame.attrs")


def _fake_find_canonical(tokens):
    mapping = {
        "GENE1": ("ENSG00000000001", "GENE1"),
        "GENE2": ("ENSG00000000002", "GENE2"),
        "GENE3": ("ENSG00000000003", "GENE3"),
        "BAD": (None, None),
        "ENSG00000000001": ("ENSG00000000001", "GENE1"),
        "ENSG00000000002": ("ENSG00000000002", "GENE2"),
    }
    ids = []
    names = []
    for t in tokens:
        gid, gname = mapping.get(str(t), (None, None))
        ids.append(gid)
        names.append(gname)
    return ids, names


def test_small_helpers():
    assert pdh._clean_token(None) is None
    assert pdh._clean_token(" nan ") is None
    assert pdh._clean_token("A") == "A"
    assert pdh._strip_ensembl_version("ENSG00000000001.5") == "ENSG00000000001"
    assert pdh._strip_ensembl_version("X") == "X"


def test_check_gene_ids_in_gene_sets_raises_missing_col():
    with pytest.raises(KeyError):
        pdh.check_gene_ids_in_gene_sets(pd.DataFrame({"x": [1]}), {"A": ["ENSG1"]})


def test_normalize_gene_sets_and_strict(monkeypatch):
    monkeypatch.setattr(pdh, "find_canonical_gene_ids_and_names", _fake_find_canonical)

    cat_ids, id_to_name = pdh.normalize_gene_sets(
        {"A": ["GENE1", "BAD"], "B": ["GENE1", "GENE2"]},
        priority_category="A",
        strict=False,
        verbose=False,
    )
    assert cat_ids["A"] == ["ENSG00000000001"]
    assert cat_ids["B"] == ["ENSG00000000002"]
    assert id_to_name["ENSG00000000001"] == "GENE1"

    with pytest.raises(ValueError):
        pdh.normalize_gene_sets({"A": ["BAD"]}, strict=True, verbose=False)


def test_prepare_gene_expr_df_with_and_without_name_col(monkeypatch):
    monkeypatch.setattr(pdh, "find_canonical_gene_ids_and_names", _fake_find_canonical)

    df = pd.DataFrame(
        {
            "canonical_gene_id": ["ENSG00000000001.1", "ENSG00000000002"],
            "canonical_gene_name": ["GENE1", "GENE2"],
            "TPM": [1.0, 0.2],
        }
    )
    out = pdh.prepare_gene_expr_df(
        df,
        gene_sets={"Set1": ["GENE1"], "Set2": ["GENE2"]},
        priority_category=None,
        place_other_first=True,
        strict_gene_sets=False,
    )
    assert {"gene_id", "category", "TPM", "log_TPM", "gene_display_name"} <= set(
        out.columns
    )
    assert "other" not in set(out["category"].astype(str))

    # exercise fallback name resolution path when gene_name_col missing
    df2 = pd.DataFrame(
        {
            "canonical_gene_id": ["ENSG00000000001", "ENSG00000000002"],
            "TPM": [1.0, 0.2],
        }
    )
    out2 = pdh.prepare_gene_expr_df(
        df2,
        gene_sets={"Set1": ["GENE1"]},
        gene_name_col=None,
        place_other_first=False,
    )
    assert len(out2) == 2


def test_prepare_gene_expr_df_does_not_deepcopy_attrs(monkeypatch):
    monkeypatch.setattr(pdh, "find_canonical_gene_ids_and_names", _fake_find_canonical)

    df = pd.DataFrame(
        {
            "canonical_gene_id": ["ENSG00000000001", "ENSG00000000002"],
            "canonical_gene_name": ["GENE1", "GENE2"],
            "TPM": [1.0, 0.2],
        }
    )
    df.attrs["transcript_expression"] = _NoDeepcopy()

    out = pdh.prepare_gene_expr_df(
        df,
        gene_sets={"Set1": ["GENE1"]},
        place_other_first=True,
        strict_gene_sets=False,
        verbose=False,
    )
    assert len(out) == 2
