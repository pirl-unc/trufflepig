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


from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple
import re
import numpy as np
import pandas as pd

from .common import without_dataframe_attrs
from pirlygenes.gene_names import aliases
from pirlygenes.gene_ids import find_canonical_gene_ids_and_names

# --------------------------- small helpers -----------------------------------

_ENSG_WITH_VER = re.compile(r"^(ENSG\d{11})(?:\.\d+)?$", re.I)


def _clean_token(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s and s.lower() not in {"none", "nan"} else None


def _strip_ensembl_version(gid: str) -> str:
    m = _ENSG_WITH_VER.match(gid)
    return m.group(1) if m else gid


# --------------------------- validation / normalization ----------------------


def _remap_retired_gene_ids(
    cat_to_gene_id_list: Dict[str, List[str]],
    gene_id_to_name: Dict[str, str],
    df_gene_expr: pd.DataFrame,
    gene_id_col: str,
    gene_name_col: Optional[str],
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    For gene IDs not found in df_gene_expr, try to find the current ID
    by matching on gene name. Handles retired Ensembl IDs that have been
    replaced in newer annotations.
    """
    expr_ids = set(df_gene_expr[gene_id_col].astype(str))

    # Build upper-cased name → ID mapping from expression data.
    # Check ALL name columns (gene_display_name, canonical_gene_name, gene)
    # so we can remap even when pyensembl doesn't know the new ID.
    name_to_expr_id: Dict[str, str] = {}
    name_cols = [gene_name_col, "canonical_gene_name", "gene"]
    name_cols = [c for c in dict.fromkeys(name_cols) if c and c in df_gene_expr.columns]
    if not name_cols:
        return cat_to_gene_id_list, gene_id_to_name

    gids = df_gene_expr[gene_id_col].astype(str).map(_strip_ensembl_version)
    for col in name_cols:
        keys = df_gene_expr[col].astype(str).str.strip().str.upper()
        valid = keys.ne("") & keys.ne("NAN") & keys.ne("NONE")
        for key, gid in zip(keys[valid], gids[valid]):
            name_to_expr_id[key] = gid

    new_cat_to_ids: Dict[str, List[str]] = {}
    new_id_to_name = dict(gene_id_to_name)

    for cat, gene_ids in cat_to_gene_id_list.items():
        new_ids: List[str] = []
        for gid in gene_ids:
            if gid in expr_ids:
                new_ids.append(gid)
                continue
            name = gene_id_to_name.get(gid)
            if name:
                new_gid = name_to_expr_id.get(name.upper())
                if new_gid:
                    if verbose:
                        print(
                            f"[info] Remapped retired gene ID {gid} ({name}) -> {new_gid}"
                        )
                    new_ids.append(new_gid)
                    new_id_to_name[new_gid] = name
                    new_id_to_name.pop(gid, None)
                    continue
            new_ids.append(gid)  # keep as-is; will warn later
        new_cat_to_ids[cat] = new_ids

    return new_cat_to_ids, new_id_to_name


def check_gene_ids_in_gene_sets(
    df_gene_expr: pd.DataFrame,
    cat_to_gene_id_list: Dict[str, List[str]],
    *,
    gene_id_col: str = "gene_id",
    gene_id_to_name: Optional[Dict[str, str]] = None,
    source_file: Optional[str] = None,
) -> None:
    if gene_id_col not in df_gene_expr.columns:
        raise KeyError(
            f"Column '{gene_id_col}' not found in expression DataFrame; "
            f"available: {list(df_gene_expr.columns)}"
        )
    src = f" in {source_file}" if source_file else ""
    ids_in_expr = set(df_gene_expr[gene_id_col].astype(str))
    for cat, gene_ids in cat_to_gene_id_list.items():
        for gid in gene_ids:
            if gid not in ids_in_expr:
                expected = (gene_id_to_name or {}).get(gid)
                if expected:
                    print(
                        f"[warn] Gene ID {gid} ({expected}) (category='{cat}') not found{src}"
                    )
                else:
                    print(f"[warn] Gene ID {gid} (category='{cat}') not found{src}")


def normalize_gene_sets(
    gene_sets: Dict[str, Iterable],
    priority_category: Optional[str] = None,
    *,
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Map mixed IDs/names in each category to canonical Ensembl gene IDs.

    Values in gene_sets can be:
      - list of gene symbols (resolved via pyensembl — slow)
      - list of (ensembl_id, symbol) tuples (used directly — fast)
      - dict mapping ensembl_id -> symbol (used directly — fast)

    Returns:
      - cat_to_gene_id_list : category -> sorted list of canonical gene IDs (strings)
      - gene_id_to_name     : canonical gene ID -> canonical gene name
    """
    cat_to_gene_ids: Dict[str, set[str]] = {}
    gene_id_to_name: Dict[str, str] = {}
    unresolved: Dict[str, List[str]] = defaultdict(list)

    for cat, genes in list(gene_sets.items()):
        # Fast path: dict of {ensembl_id: symbol}
        if isinstance(genes, dict):
            keep = []
            for gid, gname in genes.items():
                gid = _strip_ensembl_version(str(gid))
                gene_id_to_name[gid] = gname
                keep.append(gid)
            cat_to_gene_ids[cat] = set(keep)
            continue

        genes_list = list(genes)

        # Fast path: list of (ensembl_id, symbol) tuples
        if (
            genes_list
            and isinstance(genes_list[0], (tuple, list))
            and len(genes_list[0]) == 2
        ):
            keep = []
            for gid, gname in genes_list:
                gid = _strip_ensembl_version(str(gid))
                gene_id_to_name[gid] = str(gname)
                keep.append(gid)
            cat_to_gene_ids[cat] = set(keep)
            continue

        # Slow path: list of symbols — resolve via pyensembl
        tokens = [_clean_token(g) for g in genes_list]
        tokens = [t for t in tokens if t is not None]
        if not tokens:
            cat_to_gene_ids[cat] = set()
            continue

        gene_ids, gene_names = find_canonical_gene_ids_and_names(tokens)
        if len(gene_ids) != len(gene_names):
            raise ValueError(
                "find_canonical_gene_ids_and_names returned mismatched lengths."
            )

        keep: List[str] = []
        for tok, gid, gname in zip(tokens, gene_ids, gene_names):
            if gid is None or gname is None:
                unresolved[cat].append(tok)
                continue
            gid = _strip_ensembl_version(str(gid))
            gene_id_to_name[gid] = gname
            keep.append(gid)

        cat_to_gene_ids[cat] = set(keep)

    # de-duplicate with priority if requested
    if priority_category and priority_category in cat_to_gene_ids:
        prio = cat_to_gene_ids[priority_category]
        for cat, ids in list(cat_to_gene_ids.items()):
            if cat != priority_category:
                cat_to_gene_ids[cat] = ids.difference(prio)

    if any(unresolved.values()):
        if strict:
            msg = "\n".join(
                f"{cat}: {', '.join(v)}" for cat, v in unresolved.items() if v
            )
            raise ValueError("Unresolved identifiers in gene_sets:\n" + msg)
        if verbose:
            for cat, v in unresolved.items():
                if v:
                    print("[warn]", f"{cat}: {', '.join(v)}")

    cat_to_gene_id_list = {cat: sorted(ids) for cat, ids in cat_to_gene_ids.items()}
    if verbose:
        n_total = sum(len(ids) for ids in cat_to_gene_id_list.values())
        print(
            f"[plot] {len(cat_to_gene_id_list)} gene set categories, {n_total} genes total"
        )
    return cat_to_gene_id_list, gene_id_to_name


def _create_gene_to_category_list_mapping(
    cat_to_gene_id_list: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    gene_to_categories = defaultdict(list)
    for category, gene_ids in cat_to_gene_id_list.items():
        for gid in gene_ids:
            gene_to_categories[gid].append(category)
    return gene_to_categories


# --------------------------- main constructor --------------------------------


def prepare_gene_expr_df(
    df_gene_expr: pd.DataFrame,
    gene_sets: Dict[str, Iterable[str]],
    *,
    priority_category: Optional[str] = None,
    TPM_offset: float = 10.0**-2,
    gene_id_col: str = "canonical_gene_id",  # ID column in df_gene_expr
    gene_name_col: Optional[
        str
    ] = "canonical_gene_name",  # optional name column in df_gene_expr
    tpm_col: str = "TPM",
    other_category_name: str = "Other",
    place_other_first: bool = True,
    strip_version: bool = True,
    strict_gene_sets: bool = False,
    verbose: bool = True,
    source_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns a long-format dataframe with:
      - gene               : Ensembl gene ID (canonical, no version)
      - category           : category name (includes 'Other' for genes in no set)
      - TPM, log_TPM       : expression values
      - gene_display_name  : human-friendly name (aliases applied)

    Notes:
      * Plots should label with 'gene_display_name' (not 'gene').
      * 'Other' appears as the left-most category if place_other_first=True.
    """
    # basic checks
    for col in [gene_id_col, tpm_col]:
        if col not in df_gene_expr.columns:
            raise KeyError(
                f"Column '{col}' not found; available: {list(df_gene_expr.columns)}"
            )

    # Avoid deep-copying DataFrame.attrs here: load_expression attaches the
    # retained transcript-level frame under ``attrs["transcript_expression"]``,
    # and even a simple column subset can deepcopy that blob via pandas'
    # finalize path.
    with without_dataframe_attrs(df_gene_expr):
        needed_cols = [gene_id_col, tpm_col]
        for extra in (gene_name_col, "canonical_gene_name", "gene"):
            if extra and extra in df_gene_expr.columns and extra not in needed_cols:
                needed_cols.append(extra)
        df = df_gene_expr[needed_cols].copy(deep=False)
    expr_gene_ids = df[gene_id_col].astype(str)
    if strip_version:
        expr_gene_ids = expr_gene_ids.map(_strip_ensembl_version)
        df[gene_id_col] = expr_gene_ids

    # normalize sets -> IDs, and validate presence in DF (with name-aware warnings)
    cat_to_gene_id_list, gene_id_to_name_from_sets = normalize_gene_sets(
        gene_sets, priority_category=priority_category, strict=strict_gene_sets
    )
    # remap any retired Ensembl IDs to their current replacements
    cat_to_gene_id_list, gene_id_to_name_from_sets = _remap_retired_gene_ids(
        cat_to_gene_id_list,
        gene_id_to_name_from_sets,
        df,
        gene_id_col=gene_id_col,
        gene_name_col=gene_name_col,
        verbose=verbose,
    )
    check_gene_ids_in_gene_sets(
        df,
        cat_to_gene_id_list,
        gene_id_col=gene_id_col,
        gene_id_to_name=gene_id_to_name_from_sets,
        source_file=source_file,
    )

    # map ID -> name from DF if provided, else resolve names for all expressed IDs
    id_to_name_from_df: Dict[str, str] = {}
    if gene_name_col and gene_name_col in df.columns:
        # build from DF
        id_to_name_from_df = dict(zip(expr_gene_ids, df[gene_name_col].astype(str)))
    else:
        # fallback: resolve names for all expressed IDs (one call). Slow
        # on full transcriptomes — announce so the CLI progress log
        # doesn't go silent for tens of seconds here.
        uniq_ids = list(dict.fromkeys(expr_gene_ids))  # preserves order, de-dups
        if verbose:
            print(f"[plot] resolving gene names for {len(uniq_ids)} IDs...")
        resolved_ids, resolved_names = find_canonical_gene_ids_and_names(uniq_ids)
        for gid, gname in zip(resolved_ids, resolved_names):
            if gid and gname:
                id_to_name_from_df[_strip_ensembl_version(str(gid))] = gname

    # expression maps
    tpm_values = df[tpm_col].astype(float).copy()
    if TPM_offset:
        tpm_values = tpm_values + float(TPM_offset)
    gene_to_tpm = dict(zip(expr_gene_ids, tpm_values))
    gene_to_log_tpm = dict(zip(expr_gene_ids, np.log2(1 + tpm_values)))

    # categories for each ID (from sets)
    gene_to_categories = _create_gene_to_category_list_mapping(cat_to_gene_id_list)

    # build rows; add 'Other' for genes not in any set
    new_gene_ids: List[str] = []
    new_cats: List[str] = []
    for gid in expr_gene_ids:
        cats = gene_to_categories.get(gid, [])
        if cats:
            for cat in cats:
                new_gene_ids.append(gid)
                new_cats.append(cat)
        else:
            new_gene_ids.append(gid)
            new_cats.append(other_category_name)

    # display names: prefer DF name, fall back to set-derived name, then ID; apply alias
    def _display_name(gid: str) -> str:
        base = id_to_name_from_df.get(gid) or gene_id_to_name_from_sets.get(gid)
        return aliases.get(base, base) if base else gid

    display_names = [_display_name(gid) for gid in new_gene_ids]

    # category ordering with 'Other' first if requested
    cat_order = [other_category_name] + [
        c for c in cat_to_gene_id_list.keys() if c != other_category_name
    ]
    if not place_other_first:
        cat_order = list(cat_to_gene_id_list.keys()) + [other_category_name]
    category_c = pd.Categorical(new_cats, categories=cat_order, ordered=True)

    out = pd.DataFrame(
        dict(
            gene_id=new_gene_ids,  # Ensembl IDs for joins/aggregation
            category=category_c,  # includes 'Other'
            TPM=[gene_to_tpm[g] for g in new_gene_ids],
            log_TPM=[gene_to_log_tpm[g] for g in new_gene_ids],
            gene_display_name=display_names,  # <- use this in plots
        )
    )
    return out
