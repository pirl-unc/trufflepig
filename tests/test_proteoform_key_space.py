"""Proteoform key space: byte-identical-protein loci fold consistently.

trufflepig consumes pirlygenes' protein-identical gene groups so a protein split
across paralog loci (NY-ESO-1 = CTAG1A/B, α-globin = HBA1/2, amylase = AMY1A/B/C,
…) reads as a single proteoform key everywhere — reference matrix, live sample,
and curated panel — instead of a per-locus fraction. See
``trufflepig.common.collapse_proteoform_loci`` for the identifier contract.
"""

import re

import pandas as pd

from trufflepig.common import (
    build_sample_tpm_by_gene_id,
    build_sample_tpm_by_symbol,
    collapse_proteoform_loci,
    fold_panel_symbols,
)


# A folded group with two distinct member ENSGs (NY-ESO-1).
CTAG1A_ID = "ENSG00000268651"
CTAG1B_ID = "ENSG00000184033"  # the group's canonical member


def _sample_df():
    """Minimal clean sample: two NY-ESO-1 loci + GAPDH (a singleton control)."""
    return pd.DataFrame(
        {
            "ensembl_gene_id": [CTAG1A_ID, CTAG1B_ID, "ENSG00000111640"],
            "gene_name": ["CTAG1A", "CTAG1B", "GAPDH"],
            "TPM": [3.0, 5.0, 100.0],
        }
    )


# ---- reference matrices fold to proteoform keys ---------------------------

def test_pan_cancer_folds_members_to_proteoform_id():
    from trufflepig.reference import pan_cancer_expression

    ref = pan_cancer_expression()
    syms = set(ref["Symbol"].astype(str))
    # members left the key space; the proteoform id carries the row
    assert "CTAG1A" not in syms and "CTAG1B" not in syms
    assert "CTAG1A/B" in syms
    assert "HBA1" not in syms and "HBA1/2" in syms
    assert "AMY1A/B/C" in syms
    # CD99's two loci share the symbol, so the proteoform id IS "CD99"
    assert "CD99" in syms
    # the folded row is keyed by the canonical member ENSG (a real ENSG…, one of
    # the group's members), so the ENSG-keyed guard + family-panel joins resolve
    row = ref[ref["Symbol"] == "CTAG1A/B"]
    assert row["Ensembl_Gene_ID"].iloc[0] in {CTAG1A_ID, CTAG1B_ID}


def test_hpa_folds_members():
    from trufflepig.reference import hpa_cell_type_expression

    syms = set(hpa_cell_type_expression()["Symbol"].astype(str))
    assert "HBA1/2" in syms and "HBA1" not in syms


def test_tcga_deconvolved_relabels_members():
    from trufflepig.reference import tcga_deconvolved_expression

    syms = set(tcga_deconvolved_expression()["symbol"].astype(str))
    assert "CTAG1A/B" in syms
    assert "CTAG1A" not in syms and "CTAG1B" not in syms


def test_tcga_deconvolved_preserves_tpm_1e6_after_folding():
    """Relabel runs pre-renormalize, so folding out member rows must NOT break the
    per-cancer TPM-1e6 invariant (regression guard for the bug where relabel ran
    after renormalization)."""
    from trufflepig.reference import tcga_deconvolved_expression

    sums = tcga_deconvolved_expression().groupby("cancer_code")["tumor_tpm_median"].sum()
    assert ((sums - 1_000_000.0).abs() < 1.0).all(), sums[(sums - 1_000_000.0).abs() >= 1.0]


def test_non_protein_identical_dup_symbols_are_not_merged():
    """A symbol with duplicate (symbol, cancer) rows from *non*-identical loci
    (MATR3, PINX1) must survive un-merged — only protein-identical members fold."""
    from trufflepig.reference import tcga_deconvolved_expression

    df = tcga_deconvolved_expression()
    for sym in ("MATR3", "PINX1"):
        per_cancer = df[df["symbol"] == sym].groupby("cancer_code").size()
        if len(per_cancer):
            assert per_cancer.max() > 1, f"{sym} was wrongly merged to one row/cancer"


# ---- sample conform sums protein-identical loci --------------------------

def test_sample_by_symbol_sums_members():
    d = build_sample_tpm_by_symbol(_sample_df())
    assert d.get("CTAG1A/B") == 8.0  # 3 + 5, summed in linear space
    assert "CTAG1A" not in d and "CTAG1B" not in d
    assert d.get("GAPDH") == 100.0


def test_sample_by_gene_id_sums_members_and_keys_by_canonical_ensg():
    d = build_sample_tpm_by_gene_id(_sample_df())
    assert d.get(CTAG1B_ID) == 8.0  # summed onto the canonical member ENSG
    assert CTAG1A_ID not in d  # member left the key space
    # still ENSG-keyed (no proteoform ids leak into the ENSG space)
    assert all(str(k).startswith("ENSG") for k in d)


# ---- panel folding -------------------------------------------------------

def test_fold_panel_symbols():
    assert fold_panel_symbols(["CTAG1B"]) == ["CTAG1A/B"]
    assert fold_panel_symbols(["HBA1", "HBA2"]) == ["HBA1/2"]  # dedup to one key
    assert fold_panel_symbols(["GAPDH"]) == ["GAPDH"]  # singleton untouched


def test_collapse_is_a_noop_for_ungrouped_matrix():
    df = pd.DataFrame(
        {"Ensembl_Gene_ID": ["ENSG00000111640"], "Symbol": ["GAPDH"], "v": [1.0]}
    )
    out = collapse_proteoform_loci(df, value_cols=["v"])
    assert list(out["Symbol"]) == ["GAPDH"] and list(out["v"]) == [1.0]


# ---- guard: no curated panel references an UNfolded member symbol --------

# Modules holding curated gene-symbol panels. The AST scan below covers EVERY
# panel in these (nested dicts, tuples, the OPTIONAL_COMPARTMENT_GATES markers,
# lineage programs, …) without enumerating each structure — so a panel added to
# any of them is checked automatically.
_PANEL_MODULES = (
    "family_extensions.py",
    "tumor_type_ontology.py",
    "literature_signatures.py",
    "lineage_evidence.py",
    "subtype_signature.py",
    "decomposition/signature.py",
    "decomposition/templates.py",
)

# A bare gene-symbol-shaped string literal (incl. proteoform ids like CTAG1A/B).
_SYMBOL_RE = re.compile(r"[A-Z][A-Z0-9./-]{1,14}")


def _panel_symbol_constants():
    """Gene-symbol-shaped *string constants* in the panel modules, via AST.

    Parsing (not grepping) means comments are absent and docstrings are single
    long Constants that never equal a bare symbol — so this has none of the
    false positives a source scan would, and ignores nothing a panel uses."""
    import ast
    import pathlib

    import trufflepig

    root = pathlib.Path(trufflepig.__file__).parent
    out = set()
    for rel in _PANEL_MODULES:
        tree = ast.parse((root / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value.strip()
                if _SYMBOL_RE.fullmatch(s):
                    out.add(s)
    return out


def test_no_curated_panel_references_an_unfolded_member():
    """A symbol that folds to something *other than itself* is an unfolded member
    of a byte-identical-protein group — it would silently miss the collapsed
    matrix/sample. Catches future panel additions in any module above (fail
    loudly, as with assert_tpm_keyed_by_gene_id)."""
    offenders = {
        s: fold_panel_symbols([s])[0]
        for s in _panel_symbol_constants()
        if fold_panel_symbols([s]) != [s]
    }
    assert not offenders, (
        "curated panels reference unfolded protein-identical members; use the "
        f"proteoform id instead: {offenders}"
    )
