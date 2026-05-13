# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
import pandas as pd
from typing import Iterator, Optional


def find_column(
    df: pd.DataFrame, candidates: list[str], column_name: str
) -> Optional[str]:
    result = None
    for col in df.columns:
        if col.lower() in candidates:
            result = col
            break
    if result is None:
        raise ValueError(
            "Unable to find a column for %s in expression data, available columns: %s"
            % (
                column_name,
                list(
                    df.columns,
                ),
            )
        )
    return result


@contextmanager
def without_dataframe_attrs(df: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Temporarily clear ``DataFrame.attrs`` during pandas-heavy helpers.

    Pandas deep-copies ``attrs`` in many column/subset/finalize paths.
    When callers attach a large object there (for example the retained
    transcript-level frame under ``attrs["transcript_expression"]``),
    otherwise-cheap helpers can become minute-scale. Clear attrs for the
    duration of the helper, then restore them.
    """
    saved_attrs = dict(getattr(df, "attrs", {}))
    if saved_attrs:
        df.attrs = {}
    try:
        yield df
    finally:
        if saved_attrs:
            df.attrs = saved_attrs


# -------------------- gene-column helpers --------------------


def guess_gene_cols(df):
    """Best-effort guess for gene ID and name columns in df_gene_expr."""
    id_candidates = ["gene_id", "ensembl_gene_id", "canonical_gene_id", "GeneID"]
    name_candidates = [
        "gene_display_name",
        "gene_name",
        "canonical_gene_name",
        "gene_symbol",
        "symbol",
        "GeneName",
    ]
    gene_id_col = next((c for c in id_candidates if c in df.columns), None)
    gene_name_col = next((c for c in name_candidates if c in df.columns), None)
    if gene_id_col is None:
        raise KeyError(
            "Could not find a gene ID column in df_gene_expr. "
            "Tried: %s" % (id_candidates,)
        )
    if gene_name_col is None:
        raise KeyError(
            "Could not find a gene name column in df_gene_expr. "
            "Tried: %s" % (name_candidates,)
        )
    return gene_id_col, gene_name_col


# Backward-compatible alias — internal callers historically imported
# ``_guess_gene_cols`` from plot.py; the underscore form is kept so
# those imports still resolve after plot.py re-exports it.
_guess_gene_cols = guess_gene_cols


# -------------------- TPM-by-symbol --------------------


def build_sample_tpm_by_symbol(df_gene_expr):
    """Return ``{symbol: max_TPM}`` from expression data (no normalization).

    Maps Ensembl gene IDs to HGNC symbols via the bundled pan-cancer
    reference, then groups by symbol keeping the maximum TPM per gene.
    """
    from .plot_data_helpers import _strip_ensembl_version
    from pirlygenes.gene_sets_cancer import pan_cancer_expression

    with without_dataframe_attrs(df_gene_expr):
        gene_id_col, _gene_name_col = guess_gene_cols(df_gene_expr)
        gene_ids = df_gene_expr[gene_id_col].astype(str).map(_strip_ensembl_version)

        tpm_col = (
            "TPM"
            if "TPM" in df_gene_expr.columns
            else next((c for c in df_gene_expr.columns if c.lower() == "tpm"), None)
        )
        if tpm_col is None:
            raise KeyError(
                f"No TPM column found. Columns: {list(df_gene_expr.columns)}"
            )

        ref = pan_cancer_expression()
        id_to_sym = dict(zip(ref["Ensembl_Gene_ID"], ref["Symbol"]))

        syms = gene_ids.map(id_to_sym)
        tpms = pd.to_numeric(df_gene_expr[tpm_col], errors="coerce")
        valid = syms.notna() & tpms.notna()
        return dict(
            pd.DataFrame({"sym": syms[valid], "tpm": tpms[valid]})
            .groupby("sym")["tpm"]
            .max()
        )


# Underscore alias for backward compatibility with internal callers.
_build_sample_tpm_by_symbol = build_sample_tpm_by_symbol
