"""Regression tests for #171 — mixture-cohort modeling for SARC.

A mixture cohort is a parent code (``SARC``) whose TCGA median is a
biological union of lineage-distinct subtypes. The classifier should:

1. Expose ``is_mixture_cohort("SARC") is True`` in the registry API.
2. Evaluate each subtype's lineage panel against subtype-deconvolved
   tumor-only TPM instead of the diluted parent median.
3. Surface the winning subtype on the ranked row as
   ``winning_subtype`` when any subtype outscores the parent panel.
"""

import pandas as pd
import pytest

from pirlygenes.gene_sets_cancer import (
    is_mixture_cohort,
    mixture_cohort_codes,
)
from trufflepig.reference import (
    pan_cancer_expression,
    subtype_deconvolved_expression,
)
from trufflepig.tumor_purity import (
    LINEAGE_GENES,
    rank_cancer_type_candidates,
)


def test_sarc_is_flagged_mixture_cohort():
    assert is_mixture_cohort("SARC")
    assert "SARC" in mixture_cohort_codes()


def test_non_mixture_cohorts_not_flagged():
    for code in ("BRCA", "LUAD", "PRAD", "COAD"):
        assert not is_mixture_cohort(code), code


def test_sarc_subtype_panels_present():
    """The three mixture-evaluable SARC subtypes all have curated
    lineage panels in ``lineage-genes.csv``."""
    for code in ("SARC_LMS", "SARC_LPS_UNSPEC", "SARC_SYN"):
        panel = LINEAGE_GENES.get(code, [])
        assert len(panel) >= 5, (
            f"{code} mixture-subtype panel has only {len(panel)} genes — "
            f"expand via lineage-genes.csv"
        )


def test_sarc_lms_panel_has_smooth_muscle_core():
    panel = set(LINEAGE_GENES.get("SARC_LMS", []))
    for gene in ("MYH11", "ACTA2", "TAGLN"):
        assert gene in panel, f"{gene} missing from SARC_LMS panel"


def _smooth_muscle_pseudo_sample() -> pd.DataFrame:
    """Build a pseudo-sample whose expression matches the
    subtype-deconvolved SARC_LMS tumor profile — the canonical signal
    a real leiomyosarcoma sample would produce."""
    sub = subtype_deconvolved_expression()
    lms = sub[sub["cancer_code"] == "SARC_LMS"][["symbol", "tumor_tpm_median"]]
    pan = pan_cancer_expression().drop_duplicates(subset="Symbol")
    merged = pan.merge(lms, left_on="Symbol", right_on="symbol", how="left")
    merged["tpm"] = merged["tumor_tpm_median"].fillna(merged["FPKM_SARC"])
    return pd.DataFrame(
        {
            "ensembl_gene_id": merged["Ensembl_Gene_ID"],
            "gene_symbol": merged["Symbol"],
            "TPM": merged["tpm"].astype(float),
        }
    )


def test_smooth_muscle_sample_picks_sarc_lms_subtype():
    """A pseudo-sample built from the SARC_LMS tumor profile should
    rank SARC at top and carry ``winning_subtype=SARC_LMS``."""
    sample = _smooth_muscle_pseudo_sample()
    ranked = rank_cancer_type_candidates(sample)
    assert ranked, "ranker returned no candidates"
    top = ranked[0]
    # SARC should be in the top 3 (exact position can shift with other
    # family penalties; the subtype identification is the main assertion).
    top_codes = [r["code"] for r in ranked[:3]]
    assert "SARC" in top_codes, (
        f"SARC not in top 3 — got {top_codes} (top gm={top['support_geomean']:.3f})"
    )
    sarc_row = next(r for r in ranked if r["code"] == "SARC")
    assert sarc_row.get("winning_subtype") == "SARC_LMS", (
        f"expected winning_subtype=SARC_LMS, got {sarc_row.get('winning_subtype')}"
    )


def test_winning_subtype_none_for_non_mixture():
    """Non-mixture cohorts must always report ``winning_subtype=None``
    — the subtype-aware path must not fire outside mixture parents."""
    pan = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    sample = pd.DataFrame(
        {
            "ensembl_gene_id": pan["Ensembl_Gene_ID"],
            "gene_symbol": pan["Symbol"],
            "TPM": pan["FPKM_PRAD"].astype(float),
        }
    )
    ranked = rank_cancer_type_candidates(sample)
    for row in ranked:
        if row["code"] == "PRAD":
            assert row.get("winning_subtype") is None
            break
    else:
        pytest.skip("PRAD not in ranked output")
