import weakref

import pandas as pd

from trufflepig.common import (
    _RANGES_BY_GENE_ID_CACHE,
    _RANGES_BY_SYMBOL_CACHE,
    _RANGES_RECORDS_CACHE,
    ranges_by_gene_id,
    ranges_by_symbol,
    ranges_records,
)


def test_ranges_records_cache_rejects_stale_id_entry():
    old = pd.DataFrame({"symbol": ["OLD"], "gene_id": ["ENSGOLD"]})
    current = pd.DataFrame(
        {"symbol": ["A", "B"], "gene_id": ["ENSG000001", "ENSG000002"]}
    )
    cache_key = id(current)
    stale_entry = (weakref.ref(old), len(current), tuple(current.columns), [{"symbol": "OLD"}])
    _RANGES_RECORDS_CACHE[cache_key] = stale_entry
    _RANGES_BY_SYMBOL_CACHE[cache_key] = (
        weakref.ref(old),
        len(current),
        tuple(current.columns),
        {"OLD": {"symbol": "OLD"}},
    )
    _RANGES_BY_GENE_ID_CACHE[cache_key] = (
        weakref.ref(old),
        len(current),
        tuple(current.columns),
        {"ENSGOLD": {"gene_id": "ENSGOLD"}},
    )

    records = ranges_records(current)
    by_symbol = ranges_by_symbol(current)
    by_gene = ranges_by_gene_id(current)

    assert [row["symbol"] for row in records] == ["A", "B"]
    assert sorted(by_symbol) == ["A", "B"]
    assert sorted(by_gene) == ["ENSG000001", "ENSG000002"]
