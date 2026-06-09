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
    """Return the reference Ensembl ID -> HGNC symbol map."""
    from trufflepig.reference import pan_cancer_expression

    ref = pan_cancer_expression()
    return dict(zip(ref["Ensembl_Gene_ID"].astype(str), ref["Symbol"].astype(str)))


@lru_cache(maxsize=1)
def _versionless_id_to_symbol_map() -> dict[str, str]:
    """Reference Ensembl ID -> Symbol map, keyed by versionless ID."""
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

        id_to_sym = ensembl_id_to_symbol_map()

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
        return dict(
            pd.DataFrame({"gene_id": gene_ids[valid], "tpm": tpms[valid]})
            .groupby("gene_id")["tpm"]
            .max()
        )


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
