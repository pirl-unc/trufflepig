"""Compatibility aliases for the pre-1.24 variant API.

New code should import :mod:`trufflepig.variants` and use ``variant``
terminology. This module deliberately contains no parsing logic.
"""

from __future__ import annotations

from .variants import (
    VariantRecord,
    classify_variant_type,
    parse_variant_file,
    parse_variant_inputs,
    split_variant_inputs,
    variant_evidence_for_gene,
    variant_evidence_records,
    variant_record_gene_is_negated,
    variant_record_genes,
    variant_record_passes_assay_filters,
)


def AlterationRecord(  # noqa: N802
    gene: str,
    alteration: str = "",
    alteration_type: str = "unknown",
    **kwargs,
) -> VariantRecord:
    """Construct a :class:`VariantRecord` using legacy keyword names."""
    return VariantRecord(
        gene=gene,
        variant=alteration,
        variant_type=alteration_type,
        **kwargs,
    )


split_alteration_inputs = split_variant_inputs
alteration_record_passes_assay_filters = variant_record_passes_assay_filters
alteration_record_gene_is_negated = variant_record_gene_is_negated
alteration_record_genes = variant_record_genes
classify_alteration_type = classify_variant_type
parse_alteration_file = parse_variant_file
parse_alteration_inputs = parse_variant_inputs
molecular_evidence_records = variant_evidence_records
molecular_evidence_for_gene = variant_evidence_for_gene


__all__ = [
    "AlterationRecord",
    "alteration_record_gene_is_negated",
    "alteration_record_genes",
    "alteration_record_passes_assay_filters",
    "classify_alteration_type",
    "molecular_evidence_for_gene",
    "molecular_evidence_records",
    "parse_alteration_file",
    "parse_alteration_inputs",
    "split_alteration_inputs",
]
