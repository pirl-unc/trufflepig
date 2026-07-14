"""Battery contract: every TCGA cohort median must classify as itself or a
declared near-indistinguishable sibling.

Running ``rank_cancer_type_candidates`` on the ``<code>_TPM`` median
vector (as a pseudo-sample) is the easiest possible classification
task — if the classifier can't call the cohort median correctly, any
real-sample call against that cohort is suspect. COAD/READ is the accepted
exception because the report layer treats colorectal siblings as compatible.

This battery caught the original BLCA→ESCA and PAAD→STAD miscalls
(#160, #162) and now pins the post-fix contract. Running at the
ranker level is fast (~30s for 33 cohorts); the CLI-level battery in
``/tmp/battery_median`` is the slower end-to-end variant.
"""

import pandas as pd
import pytest

from trufflepig.cancer_ontology import cancer_type_registry, cancer_type_subtypes_of
from trufflepig.reference import pan_cancer_expression
from trufflepig.tumor_purity import rank_cancer_type_candidates

_ACCEPTED_SIBLING_TOP1 = {
    "COAD": {"READ"},
    "READ": {"COAD"},
}


def _accepted_top1(code: str) -> set[str]:
    """Entity-compatible calls for an expression-reference median.

    A ``member_union`` column is a computed group reference rather than a truth
    sample from one entity. Resolving that median to one of its declared member
    entities is therefore correct hierarchical behavior.
    """
    accepted = set(_ACCEPTED_SIBLING_TOP1.get(code, set()))
    registry = cancer_type_registry().set_index("code")
    if (
        code in registry.index
        and str(registry.loc[code].get("reference_source") or "") == "member_union"
    ):
        accepted.update(cancer_type_subtypes_of(code))
    return accepted


def _all_tcga_codes():
    ref = pan_cancer_expression()
    return sorted(c.removesuffix("_TPM") for c in ref.columns if c.endswith("_TPM"))


def _cohort_median_sample(code: str) -> pd.DataFrame:
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"{code}_TPM"].astype(float),
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
    assert top_code == code or top_code in _accepted_top1(code), (
        f"{code} median miscalled as {top_code}. Top 3: "
        + ", ".join(f"{r['code']}(gm={r['support_geomean']:.3f})" for r in ranked[:3])
    )
