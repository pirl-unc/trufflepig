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
from functools import lru_cache
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


@lru_cache(maxsize=1)
def ensembl_id_to_symbol_map() -> dict[str, str]:
    """Return the reference Ensembl ID -> HGNC symbol map.

    Built from the (proteoform-collapsed) pan-cancer reference, so a folded
    group's canonical Ensembl id maps to its **proteoform id** symbol
    (``ENSG…184033 -> "CTAG1A/B"``) — see :func:`collapse_proteoform_loci`. The
    member loci that left the reference key space are re-added as **aliases** onto
    the same proteoform id, so a caller that resolves a raw member ENSG without
    folding first still lands on the proteoform label instead of dropping the
    gene. (Members appear only as keys; ``.values()`` is unchanged.)
    """
    from trufflepig.reference import pan_cancer_expression

    ref = pan_cancer_expression()
    out = dict(zip(ref["Ensembl_Gene_ID"].astype(str), ref["Symbol"].astype(str)))
    canon2sym = _proteoform_canonical_id_to_symbol()
    for member_id, canon_id in _proteoform_member_to_canonical_id().items():
        proteoform = canon2sym.get(canon_id)
        if proteoform:
            out.setdefault(member_id, proteoform)
    return out


# -------------------- proteoform key space --------------------
#
# A protein-abundance proxy must key on the *protein*, not the gene locus. Two
# distinct Ensembl loci that encode a byte-identical protein (segmental-dup
# paralogs, histone clusters, the CT47A / NY-ESO-1 CTA clusters, …) each receive
# only a *fraction* of the reads a quantifier would assign to a single gene, so
# read as per-locus they make the protein look under-expressed and distort any
# per-gene threshold (CTA "ON" counting, marker panels). pirlygenes derives the
# byte-identical groups (``protein-identical-gene-groups``) and provides the
# collapse; trufflepig *consumes* it consistently at every seam where a sample,
# a reference matrix, or a curated panel enters the scoring space.
#
# Identifier contract (held everywhere a collapse runs):
#   * unique gene -> protein (not folded): key = the gene's real versionless
#     ENSG; symbol = its HGNC symbol. Untouched.
#   * folded proteoform (>=2 loci summed in LINEAR space): the member ENSGs LEAVE
#     the key space; the merged row is keyed by the group's **canonical member
#     ENSG** (a real ``ENSG…``, so the ENSG-keyed guard + family-panel joins keep
#     working) and *named* by the **proteoform id** (``CTAG1A/B``) so the symbol
#     space shows exactly what was summed. ``display_name`` maps it to a label
#     (``CTAG1A/B`` -> ``NY-ESO-1``).
# So ENSG space carries an ENSG xor a canonical-member ENSG; symbol space carries
# a symbol xor a proteoform id. Sample, reference, and panels all fold the same
# way, so a lookup by either is consistent.


@lru_cache(maxsize=1)
def _proteoform_member_to_canonical_id() -> dict[str, str]:
    """``{member_versionless_ENSG: group_canonical_ENSG}`` for byte-identical
    protein groups (identity for ungrouped genes is *not* stored — callers
    ``.get(id, id)``). Consumes pirlygenes' public protein-identical map."""
    from pirlygenes.expression.protein_groups import member_to_canonical

    return dict(member_to_canonical(kind="protein"))


@lru_cache(maxsize=1)
def _proteoform_canonical_id_to_symbol() -> dict[str, str]:
    """``{group_canonical_ENSG: proteoform_id}`` (``ENSG…184033 -> "CTAG1A/B"``).
    Consumes pirlygenes' public protein-identical map."""
    from pirlygenes.expression.protein_groups import canonical_to_symbol

    return dict(canonical_to_symbol(kind="protein"))


def fold_panel_symbols(symbols):
    """Fold a curated symbol panel onto the proteoform key space so it matches a
    collapsed matrix / sample (``["CTAG1B"] -> ["CTAG1A/B"]``; ungrouped symbols
    unchanged). Order-preserving + de-duplicated. Thin wrapper over pirlygenes'
    :func:`fold_symbols_to_canonical` — use wherever a curated gene list is
    looked up against folded expression."""
    from pirlygenes.expression.protein_groups import fold_symbols_to_canonical

    return fold_symbols_to_canonical(symbols)


def relabel_proteoform_symbols_long(
    df, *, symbol_col="symbol", id_col=None, dedup_keys, value_col=None
):
    """Relabel byte-identical-protein member symbols to their proteoform id in a
    **long summary-statistic** table (the deconvolved median/quantile artifacts),
    so a folded panel or folded sample joins consistently instead of silently
    missing the gene.

    This does **not** re-sum the values: those artifacts store medians/quantiles,
    and protein-level summation of order statistics is invalid — it must happen at
    the upstream generator (filed as follow-up). Where relabeling collides member
    rows within a logical row (``dedup_keys``), the row with the largest
    ``value_col`` is kept (the dominant locus as the proteoform proxy). Folded
    rows' ``id_col`` (if given) is moved to the canonical member ENSG too.

    Only the **relabeled** rows are deduplicated, and every other row is left
    untouched and in its original position: a non-protein-identical symbol that
    happens to have duplicate ``dedup_keys`` rows (e.g. MATR3, PINX1 — same symbol,
    distinct non-identical ENSGs) is never folded, so it is never merged, and the
    global row order is preserved (no whole-frame sort).
    """
    from pirlygenes.expression.protein_groups import canonical_symbol_map
    from .plot_data_helpers import _strip_ensembl_version

    sym_map = canonical_symbol_map()  # {member_symbol_upper: proteoform_id}
    up = df[symbol_col].astype(str).str.upper()
    changed = up.isin(sym_map)
    if not changed.any():
        return df
    out = df.copy()
    out.loc[changed, symbol_col] = up[changed].map(sym_map).values
    if id_col and id_col in out.columns:
        id2canon = _proteoform_member_to_canonical_id()
        ids = out.loc[changed, id_col].astype(str).map(_strip_ensembl_version)
        out.loc[changed, id_col] = ids.map(lambda g: id2canon.get(g, g)).values
    # Dedup ONLY the relabeled rows (a proteoform id never collides with an
    # unfolded row — no real symbol contains "/"). Keep the largest-value row per
    # collision; drop the losers; leave the rest of the frame in place.
    subset = [c for c in dedup_keys if c in out.columns]
    relabeled = out[changed]
    if value_col and value_col in relabeled.columns:
        relabeled = relabeled.sort_values(value_col, ascending=False, kind="stable")
    keep_idx = relabeled.drop_duplicates(subset=subset, keep="first").index
    drop_idx = out.index[changed].difference(keep_idx)
    return out.drop(index=drop_idx).reset_index(drop=True)


def collapse_proteoform_loci(df, *, id_col="Ensembl_Gene_ID", symbol_col="Symbol", value_cols):
    """Sum byte-identical-protein Ensembl loci in a **linear-space** wide matrix
    and relabel folded rows to the proteoform identifier contract.

    Wraps pirlygenes' :func:`collapse_protein_identical_loci` (the summation,
    ``min_count=1``: absent members ignored) and then normalizes every folded
    row so its ``id_col`` is the group's **canonical member ENSG** and its
    ``symbol_col`` is the **proteoform id** — robust to the wide collapse keeping
    a non-canonical rep when the canonical member is absent. Single-locus genes
    pass through unchanged. Must run in linear space, before any log transform.
    """
    from pirlygenes.expression.protein_groups import collapse_protein_identical_loci
    from .plot_data_helpers import _strip_ensembl_version

    out = collapse_protein_identical_loci(df, id_col=id_col, value_cols=value_cols)
    id2canon = _proteoform_member_to_canonical_id()
    canon2sym = _proteoform_canonical_id_to_symbol()
    ids = out[id_col].astype(str).map(_strip_ensembl_version)
    canon = ids.map(id2canon)
    folded = canon.notna()
    if folded.any():
        out.loc[folded, id_col] = canon[folded].values
        if symbol_col and symbol_col in out.columns:
            out.loc[folded, symbol_col] = canon[folded].map(canon2sym).values
    # Conservation sanity check: a proteoform collapse is a pure within-group SUM
    # in linear space — it MUST preserve each column's total exactly (no TPM is
    # ever lost, double-counted, or dropped). Fail loudly if it doesn't.
    for col in value_cols:
        before = float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
        after = float(pd.to_numeric(out[col], errors="coerce").fillna(0.0).sum())
        if before > 0.0 and abs(after - before) > 1e-6 * before:
            raise AssertionError(
                f"proteoform collapse did not conserve {col!r}: "
                f"{before:.6g} -> {after:.6g} (Δ={after - before:.3g})"
            )
    return out


@lru_cache(maxsize=1)
def _versionless_id_to_symbol_map() -> dict[str, str]:
    """Reference Ensembl ID -> Symbol map, keyed by versionless ID.

    Inherits the byte-identical-protein **member** aliases from
    :func:`ensembl_id_to_symbol_map` (member ENSG -> proteoform id), so resolving a
    raw member ENSG never drops the gene.
    """
    from .plot_data_helpers import _strip_ensembl_version

    return {
        _strip_ensembl_version(str(gid)): sym
        for gid, sym in ensembl_id_to_symbol_map().items()
    }


def canonical_reference_symbols(symbols, gene_ids):
    """Translate curated panel symbols to the reference's symbol vocabulary by
    routing each through its (unambiguous) Ensembl gene ID.

    Curated panels and the expression reference sometimes disagree on a gene's
    HGNC symbol — alias drift: a panel curated as ``CD20`` where the reference
    column says ``MS4A1``. The symbol-keyed sample inherits the reference's
    symbol vocabulary (it is built by mapping sample Ensembl IDs through the
    reference's ID->Symbol map), so a drifted panel symbol silently misses and
    the gene drops out of scoring entirely. Resolving each panel entry through
    its Ensembl ID to the reference's own symbol removes the drift while keeping
    the join in symbol space for the consumers that need it — the ENSG is the
    unambiguous join key, the symbol is just the reference-canonical label.

    Returns the reference's canonical symbol when the panel's gene ID is present
    in the reference, else the original symbol unchanged (no reference symbol to
    translate to). Length- and order-preserving.
    """
    from .plot_data_helpers import _strip_ensembl_version

    vmap = _versionless_id_to_symbol_map()
    out = []
    for sym, gid in zip(symbols, gene_ids):
        gid_s = str(gid)
        canon = (
            vmap.get(_strip_ensembl_version(gid_s))
            if gid_s and gid_s.lower() != "nan"
            else None
        )
        out.append(canon if canon else str(sym))
    return out


def canonicalize_symbols_to_reference(symbols):
    """Canonicalize curated symbols to the reference's vocabulary when the panel
    carries *no* Ensembl column, by resolving each missing symbol through
    pirlygenes' alias resolver to an Ensembl ID and then to the reference symbol.

    For panels that already carry Ensembl IDs prefer :func:`canonical_reference_symbols`
    (a pure dict lookup, no pyensembl walk). This variant exists for symbol-only
    sources like ``cancer_biomarker_genes`` where a clinical alias (``TROP2`` for
    ``TACSTD2``, ``CD20`` for ``MS4A1``, ``VEGFR2`` for ``KDR``) would otherwise
    silently miss the reference-vocabulary sample. Symbols already in the
    reference are passed through untouched (no resolver cost); symbols that don't
    resolve (or when pyensembl is unavailable) are returned unchanged — so this
    only ever *recovers* genes, never drops one. Order-preserving.
    """
    syms = [str(s) for s in symbols]
    vmap = _versionless_id_to_symbol_map()
    ref_syms = set(vmap.values())
    missing = [s for s in syms if s not in ref_syms]
    id_map = panel_symbols_to_gene_ids(missing) if missing else {}
    out = []
    for s in syms:
        if s in ref_syms:
            out.append(s)
            continue
        canon = vmap.get(id_map.get(s, ""))
        out.append(canon if canon else s)
    return out


@lru_cache(maxsize=1)
def lineage_genes_by_cancer_type_canonical() -> dict:
    """``{TCGA_code: [Symbol, ...]}`` lineage panels, symbols canonicalized to the
    reference vocabulary via Ensembl ID.

    The single source of canonical lineage panels: both the purity estimator
    (``tumor_purity.LINEAGE_GENES``) and the tumor-type ontology consume this so
    they cannot diverge on which symbol vocabulary they speak. Same shape and
    membership as pirlygenes' ``lineage_genes_by_cancer_type()`` (a groupby over
    the ``lineage-genes`` table); the only change is alias-drift immunity (e.g.
    the DLBC panel's ``CD20`` becomes the reference's ``MS4A1``).
    """
    from pirlygenes.gene_sets_cancer import lineage_genes_df

    df = lineage_genes_df()
    canon = canonical_reference_symbols(
        df["Symbol"].tolist(), df["Ensembl_Gene_ID"].tolist()
    )
    result: dict = {}
    seen: dict = {}
    for code, sym in zip(df["Cancer_Type"].astype(str).tolist(), canon):
        bucket = result.setdefault(code, [])
        marks = seen.setdefault(code, set())
        # Dedup only collisions introduced by canonicalization (two aliases of
        # one gene), matching the reference's own drop_duplicates(subset=Symbol).
        if sym not in marks:
            marks.add(sym)
            bucket.append(sym)
    return result


def build_sample_tpm_by_symbol(df_gene_expr):
    """Return ``{symbol: max_TPM}`` from already-clean sample expression.

    Maps Ensembl gene IDs to HGNC symbols via the bundled pan-cancer
    reference, then groups by symbol keeping the maximum TPM per gene.
    """
    from .plot_data_helpers import _strip_ensembl_version
    from trufflepig.clean_tpm import assert_clean_tpm

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
        assert_clean_tpm(
            df_gene_expr,
            value_cols=[tpm_col],
            label_col=_gene_name_col,
            id_col=gene_id_col,
            context="analysis sample expression",
        )

        # Fold byte-identical-protein loci to their canonical ENSG and SUM their
        # TPM (linear space) so a protein split across paralog loci reads as one
        # gene — the proteoform key space (see collapse_proteoform_loci). Members
        # leave the key space; the canonical id then maps to the proteoform-id
        # symbol via the (collapsed) reference map. groupby keeps the per-gene
        # MAX over any *residual* same-symbol-different-ENSG collisions (the few
        # non-protein-identical symbol clashes), matching the reference dedup.
        id2canon = _proteoform_member_to_canonical_id()
        gene_ids = gene_ids.map(lambda g: id2canon.get(g, g))
        tpms = pd.to_numeric(df_gene_expr[tpm_col], errors="coerce")
        valid = gene_ids.notna() & tpms.notna()
        folded_tpm = (
            pd.DataFrame({"gid": gene_ids[valid], "tpm": tpms[valid]})
            .groupby("gid")["tpm"]
            .sum()
        )
        # Conservation: folding member loci to canonical ENSG is a pure regroup +
        # SUM — the total over all valid genes must be unchanged (no read lost).
        _before = float(tpms[valid].sum())
        _after = float(folded_tpm.sum())
        if _before > 0.0 and abs(_after - _before) > 1e-6 * _before:
            raise AssertionError(
                f"proteoform fold lost TPM in build_sample_tpm_by_symbol: "
                f"{_before:.6g} -> {_after:.6g}"
            )
        id_to_sym = ensembl_id_to_symbol_map()
        syms = folded_tpm.index.map(id_to_sym)
        res = pd.DataFrame({"sym": syms, "tpm": folded_tpm.values})
        res = res[res["sym"].notna()]
        return dict(res.groupby("sym")["tpm"].max())


# Underscore alias for backward compatibility with internal callers.
_build_sample_tpm_by_symbol = build_sample_tpm_by_symbol


def assert_tpm_keyed_by_gene_id(sample_tpm, *, context=""):
    """Guard that a sample-TPM dict is keyed by versionless Ensembl gene ID.

    The ENSG-vs-symbol crossing is a *silent* failure class: hand a
    symbol-keyed dict (``{"KLK3": ...}``) to a consumer that looks markers up
    by Ensembl ID and every ``.get(ensg, 0.0)`` misses → the whole panel reads
    as zero with no exception (the trufflepig #65 family-panel regression: a
    second caller kept passing the symbol-keyed sample after the function moved
    to ID matching, silently degrading the hierarchy embedding).

    ENSG keys are recognizable where HGNC symbols are not — no human symbol
    starts with ``ENSG`` — so the wrong vocabulary is cheaply detectable. This
    converts "audit every call site forever" into "fail loudly at the boundary":
    ID-keyed consumers call this on entry, so a miswired caller raises instead
    of returning quietly-wrong numbers. Build the input with
    :func:`build_sample_tpm_by_gene_id`, never :func:`build_sample_tpm_by_symbol`.

    No-op on an empty dict (nothing to misread).
    """
    if not sample_tpm:
        return
    all_keys = list(sample_tpm.keys())
    # Strided sample across the whole key set, not the head: keys often arrive
    # groupby-sorted, so a head slice could be biased by a pathological prefix
    # (e.g. non-ENSG spike-ins that sort ahead of ENSG…). A stride is representative.
    stride = max(1, len(all_keys) // 256)
    sampled = all_keys[::stride][:256]
    ensg = sum(1 for k in sampled if str(k).startswith("ENSG"))
    if ensg < 0.5 * len(sampled):
        examples = [str(k) for k in all_keys[:5]]
        raise TypeError(
            f"{context or 'sample TPM dict'} must be keyed by versionless Ensembl "
            f"gene ID (ENSG…); got keys like {examples!r}. Either a symbol-keyed "
            "dict was passed (build it with build_sample_tpm_by_gene_id(), not "
            "build_sample_tpm_by_symbol()), or the source's gene-ID column is not "
            "Ensembl-keyed. Family panels join in ENSG space, so a non-ENSG key "
            "silently misses every marker."
        )


def build_sample_tpm_by_gene_id(df_gene_expr):
    """Return ``{versionless_ensembl_id: max_TPM}`` from already-clean
    sample expression.

    Mirrors :func:`build_sample_tpm_by_symbol` but keeps the canonical
    Ensembl ID as the dict key. Internal lookups should prefer this
    over the symbol-keyed variant — the symbol-keyed form remains for
    surfaces that already curate by HGNC name.
    """
    from .plot_data_helpers import _strip_ensembl_version
    from trufflepig.clean_tpm import assert_clean_tpm

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
        assert_clean_tpm(
            df_gene_expr,
            value_cols=[tpm_col],
            label_col=_gene_name_col,
            id_col=gene_id_col,
            context="analysis sample expression",
        )
        tpms = pd.to_numeric(df_gene_expr[tpm_col], errors="coerce")
        valid = (
            gene_ids.notna()
            & gene_ids.str.strip().str.len().gt(0)
            & tpms.notna()
        )
        # Fold byte-identical-protein loci to the canonical member ENSG and SUM
        # (linear space) — the ENSG-keyed half of the proteoform key space, so
        # ENSG family-panel joins see the whole protein. The key stays a real
        # ENSG (assert_tpm_keyed_by_gene_id holds). Residual duplicate ids take
        # MAX as before.
        id2canon = _proteoform_member_to_canonical_id()
        folded_ids = gene_ids[valid].map(lambda g: id2canon.get(g, g))
        summed = (
            pd.DataFrame({"gene_id": folded_ids, "tpm": tpms[valid]})
            .groupby("gene_id")["tpm"]
            .sum()
        )
        # Conservation: the fold is a pure regroup + SUM, so no read is lost.
        _before = float(tpms[valid].sum())
        _after = float(summed.sum())
        if _before > 0.0 and abs(_after - _before) > 1e-6 * _before:
            raise AssertionError(
                f"proteoform fold lost TPM in build_sample_tpm_by_gene_id: "
                f"{_before:.6g} -> {_after:.6g}"
            )
        return dict(summed)


# -------------------- ranges_df accessors --------------------
#
# Profiling identified ``for _, row in ranges_df.iterrows()`` as the
# single largest analyze-time bottleneck — ~95% of total time in a
# typical sample. Pandas' iterrows materializes a Series per row, and
# subsequent ``row.get(...)`` calls trigger ``Series.__finalize__``
# which deep-copies the attrs dict. With ~30k rows and ~10 different
# rendering passes each scanning the frame, this compounds to ~2 hours
# per sample.
#
# CRITICAL: the caches are keyed by id(ranges_df) in a module-level
# dict, NOT on ranges_df.attrs. Stashing the records list on
# ``ranges_df.attrs["_records_cache"]`` would make every remaining
# ``Series.__finalize__`` deep-copy the cache itself — *amplifying*
# the bug the cache was supposed to fix.

_RANGES_RECORDS_CACHE: dict[int, list[dict]] = {}
_RANGES_BY_SYMBOL_CACHE: dict[int, dict[str, dict]] = {}
_RANGES_BY_GENE_ID_CACHE: dict[int, dict[str, dict]] = {}
_PANEL_SYMBOL_TO_ID_CACHE: dict[frozenset[str], dict[str, str]] = {}


def _cap_cache(cache: dict, max_size: int = 4) -> None:
    """Bound the per-id cache so long-running processes (notebooks,
    upcoming service mode) don't accumulate stale frames."""
    while len(cache) > max_size:
        cache.pop(next(iter(cache)))


def ranges_records(ranges_df) -> list[dict]:
    """Return ranges_df as a list of plain dicts (cached by id).

    Use this instead of ``ranges_df.iterrows()`` in any rendering
    helper. See module-level comment for the perf rationale.
    """
    if ranges_df is None or len(ranges_df) == 0:
        return []
    key = id(ranges_df)
    cached = _RANGES_RECORDS_CACHE.get(key)
    if cached is not None:
        return cached
    records = ranges_df.to_dict("records")
    _RANGES_RECORDS_CACHE[key] = records
    _cap_cache(_RANGES_RECORDS_CACHE)
    return records


def ranges_by_symbol(ranges_df) -> dict[str, dict]:
    """Return ``{symbol -> record_dict}`` keyed by both the original
    symbol and a hyphen-stripped fallback (matches HLA-A → HLAA style
    sloppy lookups some target tables use).

    Cached by id alongside ``ranges_records``.

    Prefer :func:`ranges_by_gene_id` for new code — Ensembl IDs are
    stable across HGNC renames and don't have hyphen-collision
    ambiguity. Symbol-keyed lookups remain here for compatibility
    with the symbol-only ``cancer_biomarker_genes()`` panel API.
    """
    if ranges_df is None or len(ranges_df) == 0:
        return {}
    key = id(ranges_df)
    cached = _RANGES_BY_SYMBOL_CACHE.get(key)
    if cached is not None:
        return cached
    out: dict[str, dict] = {}
    for rec in ranges_records(ranges_df):
        sym = str(rec.get("symbol") or "").strip()
        if not sym:
            continue
        out[sym] = rec
        alt = sym.replace("-", "")
        if alt != sym:
            out.setdefault(alt, rec)
    _RANGES_BY_SYMBOL_CACHE[key] = out
    _cap_cache(_RANGES_BY_SYMBOL_CACHE)
    return out


def _versionless_gene_id(value) -> str:
    """Strip ``.N`` version suffix from an Ensembl gene id."""
    return str(value or "").split(".", 1)[0]


def ranges_by_gene_id(ranges_df) -> dict[str, dict]:
    """Return ``{ensembl_gene_id (versionless) -> record_dict}``.

    Ensembl IDs are the canonical key for per-gene lookups inside
    trufflepig — they're stable across HGNC symbol renames, don't
    suffer the HLA-A/HLAA hyphen ambiguity that symbols do, and they
    match the ``gene_id`` column trufflepig already stores in
    ``ranges_df``. Use this in any new lookup-by-gene code.

    Cached by id alongside the other ranges accessors.
    """
    if ranges_df is None or len(ranges_df) == 0:
        return {}
    key = id(ranges_df)
    cached = _RANGES_BY_GENE_ID_CACHE.get(key)
    if cached is not None:
        return cached
    out: dict[str, dict] = {}
    for rec in ranges_records(ranges_df):
        gene_id = _versionless_gene_id(rec.get("gene_id"))
        if not gene_id:
            continue
        out[gene_id] = rec
    _RANGES_BY_GENE_ID_CACHE[key] = out
    _cap_cache(_RANGES_BY_GENE_ID_CACHE)
    return out


def panel_symbols_to_gene_ids(symbols) -> dict[str, str]:
    """Resolve a set of HGNC symbols to canonical Ensembl gene IDs.

    Used to convert symbol-only panel APIs (e.g.
    ``cancer_biomarker_genes(code)``) into the ID space used by
    :func:`ranges_by_gene_id`. Cached on the symbol set so a
    rendering pass scanning the same panel multiple times pays the
    Ensembl lookup cost once.

    Returns ``{symbol -> versionless_gene_id}``. Symbols that can't
    be resolved are omitted from the mapping.
    """
    if not symbols:
        return {}
    sym_set = frozenset(str(s).strip() for s in symbols if str(s).strip())
    if not sym_set:
        return {}
    cached = _PANEL_SYMBOL_TO_ID_CACHE.get(sym_set)
    if cached is not None:
        return cached
    out: dict[str, str] = {}
    try:
        from pirlygenes.gene_ids import find_gene_id_by_name_from_ensembl
    except ImportError:
        _PANEL_SYMBOL_TO_ID_CACHE[sym_set] = out
        _cap_cache(_PANEL_SYMBOL_TO_ID_CACHE)
        return out
    for sym in sym_set:
        try:
            gid = find_gene_id_by_name_from_ensembl(sym)
        except Exception:  # noqa: BLE001
            continue
        if gid:
            out[sym] = _versionless_gene_id(gid)
    _PANEL_SYMBOL_TO_ID_CACHE[sym_set] = out
    _cap_cache(_PANEL_SYMBOL_TO_ID_CACHE)
    return out
