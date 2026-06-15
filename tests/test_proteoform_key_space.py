"""Proteoform key space: byte-identical-protein loci fold consistently.

trufflepig consumes pirlygenes' protein-identical gene groups so a protein split
across paralog loci (NY-ESO-1 = CTAG1A/B, α-globin = HBA1/2, amylase = AMY1A/B/C,
…) reads as a single proteoform key everywhere — reference matrix, live sample,
and curated panel — instead of a per-locus fraction. See
``trufflepig.common.collapse_proteoform_loci`` for the identifier contract.
"""

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


# ---- sample conform chokepoint folds once + conserves TPM ----------------

def test_conform_folds_sample_and_conserves_tpm():
    """normalize_to_reference_space (the one sample chokepoint) folds member loci
    to the proteoform key AND loses no TPM — so every downstream consumer, even
    ad-hoc ones, sees a proteoform-native sample."""
    from trufflepig.clean_tpm import normalize_to_reference_space

    df = _sample_df()  # CTAG1A=3, CTAG1B=5, GAPDH=100
    before = df["TPM"].sum()
    out = normalize_to_reference_space(
        df, value_cols=["TPM"], label_col="gene_name", id_col="ensembl_gene_id"
    )
    # member loci folded into one proteoform row, summed
    assert "CTAG1A/B" in set(out["gene_name"].astype(str))
    assert not {"CTAG1A", "CTAG1B"} & set(out["gene_name"].astype(str))
    assert out.loc[out["gene_name"] == "CTAG1A/B", "TPM"].iloc[0] == 8.0
    # no TPM lost (no technical-RNA here, so no rescale — pure fold)
    assert abs(out["TPM"].sum() - before) < 1e-9


def test_member_ensg_resolves_via_reference_map_no_drop():
    """A raw member ENSG resolves to the proteoform id through the reference map,
    so an ad-hoc consumer that skips folding does not drop the gene."""
    from trufflepig.common import ensembl_id_to_symbol_map, _versionless_id_to_symbol_map

    for m in (ensembl_id_to_symbol_map(), _versionless_id_to_symbol_map()):
        assert m.get(CTAG1A_ID) == "CTAG1A/B"  # member alias, not dropped
        assert m.get(CTAG1B_ID) == "CTAG1A/B"


def test_collapse_conservation_guard_fires_on_loss(monkeypatch):
    """The conservation guard inside collapse_proteoform_loci raises if the
    underlying collapse ever fails to preserve a column total (regression guard
    against a future change that drops or double-counts TPM)."""
    import pytest

    from trufflepig.common import collapse_proteoform_loci

    # Patch the underlying pirlygenes collapse to drop a row -> total not conserved.
    import pirlygenes.expression.protein_groups as pgp

    real = pgp.collapse_protein_identical_loci
    monkeypatch.setattr(pgp, "collapse_protein_identical_loci", lambda d, **kw: real(d, **kw).iloc[1:])

    with pytest.raises(AssertionError, match="did not conserve"):
        collapse_proteoform_loci(
            _sample_df(), id_col="ensembl_gene_id", symbol_col="gene_name", value_cols=["TPM"]
        )


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


# ---- panel ACCESSORS fold member-symbol curation at lookup -----------------
#
# Curated panels are written in natural member symbols (HBA1/HBA2, CTAG1B, …) and
# their sanctioned accessors fold to the proteoform key at lookup — so the
# curation auto-adapts to whatever pirlygenes currently groups, instead of baking
# a folded label that silently misses if the group changes. These tests pin that
# each live accessor actually folds.

def test_component_markers_accessor_folds():
    from trufflepig.decomposition.signature import COMPONENT_MARKERS, get_component_markers

    # curation is in member symbols ...
    assert "HBA1" in COMPONENT_MARKERS["erythroid"]
    # ... but the accessor returns the folded proteoform key.
    folded = get_component_markers("erythroid")
    assert "HBA1/2" in folded and "HBA1" not in folded and "HBA2" not in folded


def test_literature_signature_rules_fold():
    from trufflepig.literature_signatures import _SIGNATURE_ROWS, literature_signature_rules_df

    # the curated row holds the member symbol ...
    myx = next(r for r in _SIGNATURE_ROWS if r.cancer_code == "SARC_MYXLPS")
    assert "CTAG1B" in myx.marker_genes
    # ... and the consumed rule folds it to the proteoform key.
    rules = literature_signature_rules_df()
    support = str(rules[rules["cancer_code"] == "SARC_MYXLPS"].iloc[0]["required_support_genes"])
    assert "CTAG1A/B" in support and "CTAG1B" not in support.replace("CTAG1A/B", "")


def test_optional_compartment_gate_folds_at_detection():
    """The compartment-gate detection folds its member-symbol markers, so a sample
    carrying the proteoform key (HBA1/2, as a conformed sample does) fires the gate
    even though the gate is curated as HBA1/HBA2."""
    from trufflepig.decomposition.templates import OPTIONAL_COMPARTMENT_GATES, _detect_optional_compartments

    assert "HBA1" in OPTIONAL_COMPARTMENT_GATES["erythroid_solid"]["markers"]
    sample = {"HBA1/2": 1000.0, "HBB": 1000.0, "ALAS2": 1000.0}
    detected = _detect_optional_compartments(
        sample, cancer_type="COAD", template_name="solid_primary"
    )
    assert "erythroid" in detected
