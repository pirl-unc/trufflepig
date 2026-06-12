"""Guard: the lineage gate must fire through the LIVE ranking path.

Regression guard for the disconnect found in #75: the lineage_evidence gate
(epithelial-present demotes mesenchymal / hematolymphoid; a specific
neuroendocrine / melanocytic program demotes the other lineages) lived only in
``classify_cancer_type_ontology`` — which ``analyze()`` never calls — so it was
dead in production while every gate *unit* test stayed green (they exercised it
through that parallel entry point). These tests assert the gate actually runs
inside ``rank_cancer_type_candidates``, the function that produces the Working
cancer call, so it can't silently disconnect again.
"""

import pandas as pd

from trufflepig.reference import pan_cancer_expression
from trufflepig.tumor_purity import rank_cancer_type_candidates


def _tcga_sample(cancer_code):
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{cancer_code}_TPM"].astype(float),
        }
    )


def test_specific_program_demotes_other_lineages_in_live_ranking():
    """A melanocytic sample (lineage-defining MLANA/PMEL/TYR/DCT program present)
    must demote its non-melanocytic candidates *in the live ranking output* —
    proving the gate runs inside rank_cancer_type_candidates, not just in the
    test-only ontology walk."""
    from trufflepig.cancer_type_ontology import broad_lineage

    rows = rank_cancer_type_candidates(_tcga_sample("SKCM"), top_k=40)
    demoted = [r for r in rows if float(r.get("lineage_exclusion_factor", 1.0)) < 1.0]
    assert demoted, (
        "lineage gate did not fire in rank_cancer_type_candidates — it is "
        "disconnected from the live path again (see #75)"
    )
    # The melanocytic program demotes every *other* lineage; nothing melanocytic.
    assert all(broad_lineage(r["code"]) != "melanocytic" for r in demoted)


def test_specific_program_top_call_stays_in_its_own_lineage():
    """Sanity: the gate promotes the asserted lineage, not displaces it."""
    from trufflepig.cancer_type_ontology import broad_lineage

    for code in ("SKCM", "UVM"):
        rows = rank_cancer_type_candidates(_tcga_sample(code), top_k=10)
        assert rows, f"no candidates produced for {code}"
        assert broad_lineage(rows[0]["code"]) == "melanocytic"


def test_carcinoma_is_not_flipped_to_a_non_epithelial_lineage():
    """Wiring the gate must not flip a clean carcinoma to a sarcoma/heme/NE."""
    from trufflepig.cancer_type_ontology import broad_lineage

    rows = rank_cancer_type_candidates(_tcga_sample("COAD"), top_k=10)
    assert rows and broad_lineage(rows[0]["code"]) == "epithelial"
