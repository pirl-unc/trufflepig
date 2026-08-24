"""Compatibility import for the pre-1.24 variant-effects module name."""

from .variant_effects import (
    infer_variant_expression_hypotheses as infer_mutation_expression_hypotheses,
    variant_expression_effect_rules_df as mutation_expression_effect_rules_df,
)

__all__ = [
    "infer_mutation_expression_hypotheses",
    "mutation_expression_effect_rules_df",
]
