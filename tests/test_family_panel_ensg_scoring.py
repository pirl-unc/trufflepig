"""Family-panel scoring matches by Ensembl gene ID, not HGNC symbol.

The cancer-family panels are ENSG-curated. Matching them by symbol made any
symbol/alias drift between the panel's curation vocabulary and trufflepig's
reference symbols silently resolve to ``.get(symbol, 0.0)`` — a real, expressed
marker read as "not expressed". Scoring by ID eliminates that class.
"""

import pandas as pd
from pirlygenes.gene_sets_cancer import cancer_family_panels_df, housekeeping_gene_ids

from trufflepig.tumor_purity import (
    _CANCER_FAMILY_PANELS,
    _CANCER_FAMILY_PANELS_BY_ID,
    _build_sample_tpm_by_gene_id,
    _score_cancer_family_panels,
)


def test_id_panels_cover_same_families_with_unversioned_ensgs():
    assert set(_CANCER_FAMILY_PANELS_BY_ID) == set(_CANCER_FAMILY_PANELS)
    for family, ids in _CANCER_FAMILY_PANELS_BY_ID.items():
        assert ids, f"{family}: empty ENSG panel"
        for gid in ids:
            assert gid.startswith("ENSG") and "." not in gid, f"{family}: bad ENSG {gid!r}"


def test_scoring_matches_by_ensg_even_with_wrong_symbol():
    """Markers present under their ENSG but a deliberately wrong symbol must
    still score the family — proving ID-matching, immune to alias drift.
    (Under the old symbol-keyed scoring these would silently read as 0.)"""
    panels = cancer_family_panels_df()
    ne = panels[panels["Family"] == "NEUROENDOCRINE"]
    assert not ne.empty

    rows = [
        # real NE markers at high TPM, but labelled with a bogus alias symbol
        {"gene_id": row["Ensembl_Gene_ID"], "symbol": "WRONG_ALIAS", "TPM": 500.0}
        for _, row in ne.iterrows()
    ]
    # housekeeping anchors (so the hk median denominator is > 0)
    rows += [
        {"gene_id": gid, "symbol": "HK", "TPM": 50.0}
        for gid in list(housekeeping_gene_ids())[:8]
    ]
    sample = pd.DataFrame(rows)

    by_id = _build_sample_tpm_by_gene_id(sample)
    scores = _score_cancer_family_panels(by_id)

    assert scores["NEUROENDOCRINE"] > 0, (
        "NE markers (correct ENSG, wrong symbol) scored 0 — scoring is not "
        "matching by Ensembl ID"
    )
    # nothing else was supplied, so the rest stay at zero
    assert all(
        v == 0 for fam, v in scores.items() if fam != "NEUROENDOCRINE"
    ), scores
