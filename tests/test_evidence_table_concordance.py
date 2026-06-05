"""Regression test for the per-candidate signature-evidence table (audit bug E).

The Δlog2 column compares the two *reference* medians (subtype vs competitor),
not the sample. A gene that is high in the subtype reference but absent in the
sample therefore shows a positive Δlog2 and used to read as misleading
"support". The "Sample supports?" column marks whether the sample's own
expression is actually closer to the subtype than to the competitor.
"""

from trufflepig.evidence_tables import (
    _COMPETING_COHORT_FOR_TABLE,
    _per_subtype_evidence_table,
)


def test_zero_expression_gene_is_not_marked_as_support():
    code = next(iter(_COMPETING_COHORT_FOR_TABLE))  # a code with a competitor
    lines = _per_subtype_evidence_table(
        code,
        "TESTSUB",
        sample_tpm_by_symbol={"GENE_ABSENT": 0.0, "GENE_PRESENT": 120.0},
        cohort_medians={"GENE_ABSENT": 200.0, "GENE_PRESENT": 150.0},
        competitor_medians={"GENE_ABSENT": 1.0, "GENE_PRESENT": 2.0},
        signature_genes=["GENE_ABSENT", "GENE_PRESENT"],
    )
    table = "\n".join(lines)
    assert "Sample supports?" in table  # the new column exists
    # Gene high in the subtype reference but ~0 in the sample: positive Δlog2 but
    # NOT supporting (sample is closer to the competitor).
    absent_row = next(l for l in lines if l.startswith("| GENE_ABSENT "))
    assert "+" in absent_row and absent_row.rstrip().endswith("✗ |")
    # Gene the sample actually expresses concordantly with the subtype: supports.
    present_row = next(l for l in lines if l.startswith("| GENE_PRESENT "))
    assert present_row.rstrip().endswith("✓ |")
