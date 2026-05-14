"""Tests for the auxiliary transcript → gene mapping table (#122)."""

import pandas as pd

from trufflepig.transcript_to_gene import (
    extra_tx_mappings,
    load_extra_tx_mapping_records,
)


def test_known_unresolved_transcripts_have_aux_entries():
    """The three canonical cases from #122 (discovered on the rs
    quant.sf after the pyensembl cascade couldn't resolve them) must
    resolve through the aux map: HLA-A / TRMT9B / CTSB patch-haplotype
    transcripts."""
    assert extra_tx_mappings.get("ENST00000639891") == "HLA-A"
    assert extra_tx_mappings.get("ENST00000647452") == "TRMT9B"
    assert extra_tx_mappings.get("ENST00000646105") == "CTSB"


def test_record_form_exposes_gene_id_and_biotype():
    """``load_extra_tx_mapping_records`` returns the richer per-row
    metadata so downstream consumers (future trufflepig / annotation
    tooling) can cross-reference the resolved gene IDs and filter by
    biotype. The plain ``extra_tx_mappings`` stays symbol-only for
    back-compat with the aggregator."""
    records = load_extra_tx_mapping_records()
    by_id = {r["transcript_id"]: r for r in records}
    r = by_id.get("ENST00000639891")
    assert r is not None
    assert r["gene_symbol"] == "HLA-A"
    assert r["ensembl_gene_id"] == "ENSG00000206505"
    assert r["biotype"] == "protein_coding"
    assert "#122" in r["source_notes"]


def test_symbols_are_plain_gene_symbols_not_transcript_names():
    """Transcript names from Ensembl REST look like 'HLA-A-247' or
    'TRMT9B-214'. The aux map must store the plain gene symbol only —
    otherwise downstream tx → gene aggregation creates a separate
    entry per transcript."""
    records = load_extra_tx_mapping_records()
    for r in records:
        symbol = r["gene_symbol"]
        # Last '-N' suffix where N is all digits is a red flag.
        if "-" in symbol:
            last = symbol.rsplit("-", 1)[-1]
            assert not last.isdigit(), (
                f"'{r['transcript_id']}' has transcript-number suffix "
                f"in symbol '{symbol}' — should be stripped to bare "
                "gene symbol (HLA-A, not HLA-A-247)"
            )


def test_resolver_uses_aux_map(tmp_path, capsys):
    """End-to-end: a quant-like DataFrame with transcripts that ONLY
    the aux map knows should get resolved through _resolve_unknown_
    transcripts_for_raw_frame without hitting pyensembl for them."""
    from trufflepig.load_expression import _resolve_unknown_transcripts_for_raw_frame

    # Only include transcripts that are in the aux map but NOT in
    # pyensembl's fast-path index (the HLA-A patch haplotype is the
    # canonical example).
    df = pd.DataFrame(
        {
            "Name": ["ENST00000639891.1", "ENST00000647452.1", "ENST00000646105.1"],
            "TPM": [9.5, 3.1, 7847.0],
        }
    )
    resolved = _resolve_unknown_transcripts_for_raw_frame(
        df,
        verbose=True,
        progress=False,
    )
    # Aux-map entries made it into the resolved dict.
    assert resolved.get("ENST00000639891") == "HLA-A"
    assert resolved.get("ENST00000647452") == "TRMT9B"
    assert resolved.get("ENST00000646105") == "CTSB"

    # Telemetry line printed so curators see the aux map is used.
    captured = capsys.readouterr()
    assert "extra-tx-mappings" in captured.out
    assert "#122" in captured.out


def test_no_duplicate_transcript_ids_in_csv():
    """A duplicate entry would mean one row silently wins; tests as a
    data-hygiene guard."""
    records = load_extra_tx_mapping_records()
    ids = [r["transcript_id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate transcript_id in extra-tx-mappings.csv"
