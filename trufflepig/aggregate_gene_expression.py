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

# -----------------------------------------------------------
# Aggregate gene expression data from transcript level expression
# to gene level expression regardless of which exact transcript
# reference was used.

from functools import lru_cache

import pandas as pd
from tqdm import tqdm

from .common import find_column
from pirlygenes.gene_ids import (
    _build_indexes,
    find_gene_and_ensembl_release_by_name,
    find_gene_name_from_ensembl_transcript_id,
)
from .transcript_to_gene import extra_tx_mappings


def _expanded_tx_map(tx_to_gene_name: dict[str, str]) -> dict[str, str]:
    """
    Expand a transcript->gene dict to include versionless keys.
    If both versioned and versionless exist, keep the first-seen value.
    """
    out = {}
    for k, v in tx_to_gene_name.items():
        if k not in out:
            out[k] = v
        k0 = k.split(".", 1)[0]
        out.setdefault(k0, v)
    return out


def aggregate_gene_expression(
    df: pd.DataFrame,
    tx_to_gene_name: dict[str, str] = extra_tx_mappings,
    transcript_id_column_candidates: list[str] = (
        "transcript transcript_id transcriptid target target_id targetid name".split()
    ),
    tpm_column_candidates: list[str] = ("tpm",),
    verbose: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Aggregate transcript-level TPM values to gene-level TPM values.

    Returns a DataFrame with:
      - gene
      - TPM
      - gene_id
      - ensembl_release
    """
    if verbose:
        print(f"[aggregate] Starting transcript->gene aggregation for {len(df)} rows")

    transcript_id_column = find_column(
        df, transcript_id_column_candidates, "transcript ID"
    )
    tpm_column = find_column(df, tpm_column_candidates, "TPM")
    if verbose:
        print(
            f"[aggregate] Using columns transcript='{transcript_id_column}', tpm='{tpm_column}'"
        )

    tx_raw = df[transcript_id_column].astype(str)
    tpm = pd.to_numeric(df[tpm_column], errors="coerce").fillna(0.0)
    tx0 = tx_raw.str.split(".", n=1).str[0]

    tx_map = _expanded_tx_map(tx_to_gene_name or {})
    gene_series = tx0.map(tx_map)

    unknown_mask = gene_series.isna()
    if unknown_mask.any():
        # Resolve the full unknown set — including zero-TPM rows.
        # Dropping zero-TPM transcripts here is unsafe: a gene whose
        # every transcript is zero would fail to resolve, get dropped
        # from the aggregate entirely, and flip from "present at 0 TPM"
        # to "missing from quant" — which changes downstream
        # presence/absence decisions (e.g. the FFPE-sensitive panel
        # geomean in sample_context.py uses ``if s in tpm_by_symbol``).
        # The module-level ``@lru_cache`` on
        # ``find_gene_name_from_ensembl_transcript_id`` (gene_ids.py)
        # means duplicated calls from other passes on the same IDs
        # within one process are free — so we eat the cost of the
        # first resolution but don't pay for the subsequent
        # ``_build_transcript_expression_frame`` pass.
        unknown_unique = pd.Index(tx0[unknown_mask].dropna().unique())
        if verbose:
            print(
                f"[aggregate] Resolving {len(unknown_unique)} unique transcripts via Ensembl lookup"
            )

        # Build the Ensembl index *before* the progress bar starts so the
        # user doesn't see a 0% bar stuck while the index loads.
        _build_indexes()

        @lru_cache(maxsize=None)
        def _resolve_tx(t: str):
            g = find_gene_name_from_ensembl_transcript_id(t, verbose=False)
            if g:
                return g
            return extra_tx_mappings.get(t)

        resolved = {}
        iterator = tqdm(
            unknown_unique,
            desc="Resolving transcript IDs",
            disable=not progress,
        )
        for t in iterator:
            resolved[t] = _resolve_tx(t)
        resolved_series = tx0[unknown_mask].map(resolved)
        gene_series = gene_series.astype(object)
        gene_series.loc[unknown_mask] = resolved_series

    unknown_mask = gene_series.isna()
    unknown_genes_tpm = float(tpm[unknown_mask].sum())
    unresolved_tx_count = int(unknown_mask.sum())
    unresolved_unique_count = int(tx0[unknown_mask].nunique())
    unresolved_high_tpm = []

    if verbose and unknown_mask.any():
        high_tpm_unknown = (
            pd.DataFrame({"tx": tx0[unknown_mask], "TPM": tpm[unknown_mask]})
            .groupby("tx", as_index=False)["TPM"]
            .sum()
            .sort_values("TPM", ascending=False)
        )
        high_tpm_unknown = high_tpm_unknown[high_tpm_unknown["TPM"] > 1]
        unresolved_high_tpm = [
            {"tx": str(row.tx), "TPM": float(row.TPM)}
            for _, row in high_tpm_unknown.iterrows()
        ]
        if len(high_tpm_unknown):
            print(
                f"[aggregate] {len(high_tpm_unknown)} unresolved transcript IDs with TPM>1 (showing up to 20):"
            )
            for _, row in high_tpm_unknown.head(20).iterrows():
                print(f"[aggregate] unresolved tx={row.tx} TPM={row.TPM:.4f}")

    known = pd.DataFrame(
        {"gene": gene_series[~unknown_mask], "TPM": tpm[~unknown_mask]}
    )
    df_gene_expr = known.groupby("gene", as_index=False, sort=False)["TPM"].sum()
    df_gene_expr = df_gene_expr.sort_values("TPM")

    known_genes_tpm = float(df_gene_expr["TPM"].sum())
    denom = known_genes_tpm + unknown_genes_tpm
    pct_known = (known_genes_tpm * 100.0 / denom) if denom > 0 else 0.0
    if verbose:
        print(
            f"[aggregate] Assigned {known_genes_tpm:.2f} TPM to known genes, "
            f"{unknown_genes_tpm:.2f} to unknown gene names; {pct_known:.4f}% known"
        )

    @lru_cache(maxsize=None)
    def _lookup_gene_meta(gene_name: str):
        pair = find_gene_and_ensembl_release_by_name(gene_name)
        if pair is None:
            return (None, -1)
        ensembl_genome, gene = pair
        return (gene.id, ensembl_genome.release)

    genes = list(df_gene_expr["gene"])
    metas = []
    iterator = tqdm(
        genes,
        desc="Resolving Ensembl gene IDs",
        disable=not progress,
    )
    for gene_name in iterator:
        metas.append(_lookup_gene_meta(gene_name))

    if metas:
        gene_ids, releases = zip(*metas)
    else:
        gene_ids, releases = (), ()
    df_gene_expr["gene_id"] = list(gene_ids)
    df_gene_expr["ensembl_release"] = (
        pd.Series(releases, index=df_gene_expr.index).fillna(-1).astype(int)
    )

    if verbose:
        print(f"[aggregate] Completed aggregation with {len(df_gene_expr)} genes")
    df_gene_expr.attrs["transcript_aggregation_stats"] = {
        "known_tpm": known_genes_tpm,
        "unknown_tpm": unknown_genes_tpm,
        "known_fraction": float(pct_known / 100.0),
        "unknown_fraction": float(unknown_genes_tpm / denom) if denom > 0 else 0.0,
        "unresolved_row_count": unresolved_tx_count,
        "unresolved_unique_count": unresolved_unique_count,
        "unresolved_high_tpm": unresolved_high_tpm,
    }

    return df_gene_expr
