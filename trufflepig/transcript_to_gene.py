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

import csv

from ._data import DATA_DIR


def _load_extra_tx_mappings() -> dict:
    """Return ``{transcript_id: gene_symbol}`` for back-compat.

    Consumed by :mod:`pirlygenes.aggregate_gene_expression` and
    :mod:`pirlygenes.load_expression` as a fallback before the
    pyensembl cascade — a transcript that pyensembl doesn't know
    about (GENCODE alt-haplotype contigs, transcripts retired
    between Ensembl releases, etc.) can still be resolved to a gene
    symbol here. Richer per-row metadata (gene_id, biotype, source
    notes) lives in the CSV too and is exposed via
    :func:`load_extra_tx_mapping_records` for consumers that need it
    (#122).
    """
    p = DATA_DIR / "extra-tx-mappings.csv"
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            out[row["transcript_id"]] = row["gene_symbol"]
    return out


def load_extra_tx_mapping_records() -> list[dict]:
    """Return the full record for each row in
    ``data/extra-tx-mappings.csv``.

    Columns: ``transcript_id, gene_symbol, ensembl_gene_id (may be
    empty), biotype (may be empty), source_notes (may be empty)``.

    Use this when you need the gene_id / biotype annotation (#122) —
    the plain :func:`_load_extra_tx_mappings` strips everything except
    the transcript → symbol mapping for aggregator back-compat.
    """
    p = DATA_DIR / "extra-tx-mappings.csv"
    records = []
    with open(p) as f:
        for row in csv.DictReader(f):
            records.append(
                {
                    "transcript_id": row["transcript_id"],
                    "gene_symbol": row["gene_symbol"],
                    "ensembl_gene_id": row.get("ensembl_gene_id") or "",
                    "biotype": row.get("biotype") or "",
                    "source_notes": row.get("source_notes") or "",
                }
            )
    return records


extra_tx_mappings = _load_extra_tx_mappings()
