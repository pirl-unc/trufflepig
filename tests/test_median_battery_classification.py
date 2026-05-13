"""Battery contract: every TCGA cohort median must classify as itself.

Running ``rank_cancer_type_candidates`` on the FPKM_<code> median
vector (as a pseudo-sample) is the easiest possible classification
task — if the classifier can't call the cohort median correctly, any
real-sample call against that cohort is suspect.

This battery caught the original BLCA→ESCA and PAAD→STAD miscalls
(#160, #162) and now pins the post-fix contract. Running at the
ranker level is fast (~30s for 33 cohorts); the CLI-level battery in
``/tmp/battery_median`` is the slower end-to-end variant.
"""

import pandas as pd
import pytest

from pirlygenes.gene_sets_cancer import pan_cancer_expression
from trufflepig.tumor_purity import rank_cancer_type_candidates


def _all_tcga_codes():
    ref = pan_cancer_expression()
    return sorted(c.replace("FPKM_", "") for c in ref.columns if c.startswith("FPKM_"))


def _cohort_median_sample(code: str) -> pd.DataFrame:
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"FPKM_{code}"].astype(float),
        }
    )


# Parametrize over every TCGA code so pytest reports each cohort
# independently (one failure doesn't stop the others from running).
@pytest.mark.parametrize("code", _all_tcga_codes())
def test_tcga_cohort_median_classifies_as_itself(code):
    df = _cohort_median_sample(code)
    ranked = rank_cancer_type_candidates(df)
    assert ranked, f"{code}: ranker returned no candidates"
    top_code = ranked[0]["code"]
    assert top_code == code, (
        f"{code} median miscalled as {top_code}. Top 3: "
        + ", ".join(f"{r['code']}(gm={r['support_geomean']:.3f})" for r in ranked[:3])
    )
