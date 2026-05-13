import pandas as pd
import pytest

from trufflepig.common import find_column
from pirlygenes.gene_names import (
    display_name,
    get_alias_as_list,
    get_reverse_alias_as_list,
    short_gene_name,
)


def test_find_column_case_insensitive():
    df = pd.DataFrame(columns=["Transcript_ID", "TPM"])
    assert find_column(df, ["transcript_id"], "transcript") == "Transcript_ID"


def test_find_column_raises_when_missing():
    df = pd.DataFrame(columns=["A", "B"])
    with pytest.raises(ValueError):
        find_column(df, ["missing"], "missing")


def test_gene_name_alias_helpers():
    assert get_alias_as_list("CD276") == ["B7-H3"]
    assert get_alias_as_list("UNKNOWN") == []
    assert "CD276" in get_reverse_alias_as_list("B7-H3")
    assert display_name("TACSTD2") == "TROP2"
    assert display_name("NO_ALIAS") == "NO_ALIAS"


def test_short_gene_name_normalization():
    assert short_gene_name("B7-H3") == "CD276"
    assert short_gene_name("hla-a") == "HLA-A"
