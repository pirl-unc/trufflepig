# Variant inputs and coordinate provenance

Trufflepig uses one term—**variant**—for exact and imprecise somatic changes,
including small variants, copy-number changes, kinase-domain duplications, and
structural variants such as fusions. `VariantRecord` is the common Python and
serialized representation. Sample-wide states such as MSI-H, dMMR, and TMB-high
are not individual variants and are kept outside this record type.

## What is guaranteed now

A record has one of two representations:

- `symbolic`: an assembly-neutral assertion such as `EGFR KDD` or
  `ALK rearrangement`;
- `coordinate`: one or more validated, 1-based genomic intervals accompanied
  by a declared `GRCh37` or `GRCh38` build.

Every record can retain its source path and format, caller version, Ensembl
release, participating genes, and coordinates. `normalize_variant_record()`
provides the public JSON-safe form. `validate_variant_genome_builds()` rejects
missing or contradictory builds for coordinate records and can compare the
observed build with an expected build. It does not perform liftover.

`parse_variant_file()` currently supports the compatibility-oriented generic
CSV, TSV, Excel, JSON, and text readers. Common columns such as `Chromosome`,
`Position`, `Ref`, `Alt`, `NCBI_Build`, `Ensembl_Release`, and
`Caller_Version` are retained. A caller can provide a missing build explicitly
through the Python API:

```python
from trufflepig.variants import parse_variant_file

records = parse_variant_file("variants.tsv", genome_build="GRCh38")
```

The explicit build fills only coordinate records. Symbolic records remain
assembly-neutral.

## What is deliberately not inferred

VCF and MAF are standards with semantics that a generic dataframe reader cannot
safely preserve. They are therefore rejected with an adapter-required error;
renaming one to `.tsv` is not supported. Source-specific VCF, MAF, and fusion
caller adapters, explicit table column maps, per-input format selection, and a
typed command-line surface are tracked in
[issue #140](https://github.com/pirl-unc/trufflepig/issues/140).

Coordinate compatibility checks and explicit, auditable opt-in liftover are
tracked separately in
[issue #141](https://github.com/pirl-unc/trufflepig/issues/141). Trufflepig does
not silently transform coordinates.
