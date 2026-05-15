"""Expression-backed reference data for the analysis engine.

Thin pass-through to the canonical pirlygenes 5.1.1+ accessors. The
CSVs, the rescaling primitives, and the wide kwarg surface
(``technical_rna_normalize``, ``remove_noncoding``,
``renormalize_to_million``) now all live in pirlygenes. This module
exists for back-compat: existing analysis imports keep working, and
the trufflepig surface flips the ``renormalize_to_million`` default
to ``True`` (so user RNA-seq input and the bundled cohort columns
share the TPM-1e6 footing) while leaving pirlygenes' own default at
``False``.

Boundary (post-pirlygenes#246/#247/#248):

    pirlygenes ─ CSVs + accessors + rescaling primitives
    trufflepig ─ analysis composition: defaults tuned for the user-
                 sample analysis pipeline; per-sample QC narration,
                 decomposition, signature scoring (all elsewhere).

Decomposition still opts out of ``renormalize_to_million=True`` at
its three call-sites (engine, panels, signature) pending the full
recalibration sweep tracked in #27.
"""

from __future__ import annotations

import pandas as pd

import pirlygenes as _pirlygenes
from pirlygenes.load_dataset import get_data as get_reference_data


def tcga_deconvolved_expression() -> pd.DataFrame:
    """Per-(symbol, TCGA code) tumor-only TPM medians from offline deconv."""
    return _pirlygenes.tcga_deconvolved_expression()


def subtype_deconvolved_expression(
    technical_rna_normalize: bool = False,
    remove_noncoding: bool = False,
    renormalize_to_million: bool = True,
) -> pd.DataFrame:
    """Per-(cancer_code, subtype, symbol) tumor-only TPM medians.

    Trufflepig surface flips ``renormalize_to_million`` to ``True`` so
    each (cancer_code, subtype) group is on the standard TPM-1e6
    footing. Pass ``renormalize_to_million=False`` for the native
    bundled scale.
    """
    return _pirlygenes.subtype_deconvolved_expression(
        technical_rna_normalize=technical_rna_normalize,
        remove_noncoding=remove_noncoding,
        renormalize_to_million=renormalize_to_million,
    )


def pan_cancer_expression(
    genes=None,
    normalize=None,
    log_transform: bool = False,
    technical_rna_normalize: bool = False,
    remove_noncoding: bool = False,
    renormalize_to_million: bool = True,
) -> pd.DataFrame:
    """Expression across 50 normal tissues (nTPM) and 33 TCGA cancer types.

    Wraps :func:`pirlygenes.pan_cancer_expression` with the trufflepig
    default of ``renormalize_to_million=True`` so user RNA-seq input
    and bundled cohort columns share the TPM-1e6 footing. Pass
    ``renormalize_to_million=False`` for the native bundled scale —
    decomposition callsites explicitly opt out pending #27.
    """
    return _pirlygenes.pan_cancer_expression(
        genes=genes,
        normalize=normalize,
        log_transform=log_transform,
        technical_rna_normalize=technical_rna_normalize,
        remove_noncoding=remove_noncoding,
        renormalize_to_million=renormalize_to_million,
    )


def cancer_types() -> list[str]:
    """Return the list of available TCGA cancer type codes (FPKM_ columns)."""
    df = get_reference_data("pan-cancer-expression")
    return sorted(c.replace("FPKM_", "") for c in df.columns if c.startswith("FPKM_"))


def cancer_expression(cancer_type, genes=None) -> pd.DataFrame:
    """Expression for a single cancer type as a simple gene-level DataFrame.

    Columns: ``Ensembl_Gene_ID``, ``Symbol``, ``expression``
    (housekeeping-normalized, technical-RNA-normalized).
    """
    return _pirlygenes.cancer_expression(cancer_type, genes=genes)


def cancer_enriched_genes(cancer_type, min_fold: float = 3.0, min_expression: float = 0.01):
    """Genes enriched in one cancer type vs the median of all others."""
    return _pirlygenes.cancer_enriched_genes(
        cancer_type, min_fold=min_fold, min_expression=min_expression
    )


def top_enriched_per_cancer_type(top_n: int = 20, min_fold: float = 3.0, min_expression: float = 0.01):
    """Top-N enriched genes per TCGA code (dict keyed by code)."""
    out: dict[str, pd.DataFrame] = {}
    for code in cancer_types():
        enriched = cancer_enriched_genes(
            code,
            min_fold=min_fold,
            min_expression=min_expression,
        )
        out[code] = enriched.head(top_n)
    return out


def tumor_up_vs_matched_normal(cancer_code: str | None = None) -> pd.DataFrame:
    """Solid-tumor genes dramatically up vs matched normal tissue."""
    return _pirlygenes.tumor_up_vs_matched_normal(cancer_code=cancer_code)


def heme_tumor_up_vs_matched_normal(cancer_code: str | None = None) -> pd.DataFrame:
    """Heme analogue of :func:`tumor_up_vs_matched_normal`."""
    return _pirlygenes.heme_tumor_up_vs_matched_normal(cancer_code=cancer_code)


def hpa_cell_type_expression() -> pd.DataFrame:
    """Per-(gene, cell-type) nTPM from the HPA single-cell consensus."""
    return _pirlygenes.hpa_cell_type_expression()


def estimate_signatures() -> pd.DataFrame:
    """ESTIMATE stromal / immune signature panels (Yoshihara 2013)."""
    return _pirlygenes.estimate_signatures()
