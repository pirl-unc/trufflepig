# Repository notes for Claude

## Running tests

**Always use `./test.sh`, not raw `pytest`.**

`pyproject.toml` sets `addopts = "-n auto"`, which makes pytest-xdist spawn one
worker per logical CPU (~10 on the dev machine). Each worker re-imports
trufflepig + pirlygenes and loads the reference matrices (`pan-cancer-expression`,
`tcga-deconvolved-expression`, `hpa-cell-type-expression`, `subtype-deconvolved-expression`),
so peak RSS lands around 1.5 GB per worker — ~15 GB total. On a 32 GB Mac that's
fine alone, but OOM-bait when other pytest suites or fat IDEs are running.

`./test.sh` computes `min(cpu_count, available_RAM / 1.5GB)` and passes it as
`-n` (xdist resolves duplicate `-n` flags to the last one), so it stays under
memory pressure.

Pass extra pytest args after the script name:

    ./test.sh -q                              # quiet, full suite
    ./test.sh tests/test_expression_qc_lncrna.py  # one file
    ./test.sh -x -v                           # stop on first failure, verbose

Pytest-xdist also produces *intermittent* test-isolation failures on tests that
share module-level cache state (e.g. `_PAN_CANCER_CACHE`). If a test passes
alone but fails in the suite, that's almost always xdist worker-affinity flake,
not a real regression — re-run the failing file alone to confirm.

## Gene-identifier key space (proteoform contract)

Expression is read as a **protein-abundance proxy**, so the key is the *protein*,
not the gene locus. Two distinct Ensembl loci encoding a byte-identical protein
(NY-ESO-1 = `CTAG1A`+`CTAG1B`, α-globin = `HBA1`+`HBA2`, amylase = `AMY1A/B/C`,
histone/CT47A clusters, segmental-dup paralogs — 172 groups / ~236 genes) each
get only a *fraction* of the reads, so per-locus they under-count the protein.
pirlygenes derives the groups (`protein-identical-gene-groups`); trufflepig
consumes the collapse consistently at every seam.

**Contract** (see `common.collapse_proteoform_loci`):

- *Unique gene → protein*: key = real versionless **ENSG**; symbol = HGNC symbol.
- *Folded proteoform (≥2 loci, SUMmed in linear space)*: members leave the key
  space. The row is keyed in **ENSG space** by the group's **canonical member
  ENSG** (still a real `ENSG…`, so `assert_tpm_keyed_by_gene_id` + ENSG family
  panels keep working) and named in **symbol space** by the **proteoform id**
  (`CTAG1A/B`; `display_name` → `NY-ESO-1`).

Applied at: pan-cancer + HPA matrices (`collapse_proteoform_loci`, SUM, linear
space, before any normalize); the **sample conform chokepoint**
(`clean_tpm.normalize_to_reference_space`, which every analyzed sample passes
through — folds the whole sample once so *every* consumer, canonical or ad-hoc,
sees a proteoform-native frame and no member-locus read is ever dropped);
deconvolved artifacts (symbol **relabel** + keep max-median, *not* summed —
order-statistic summation must move upstream to the generator, filed as
follow-up). `ensembl_id_to_symbol_map` re-adds member ENSGs as aliases onto the
proteoform id, so any direct caller resolving a raw member ENSG never drops it.

**TPM is conserved**: `collapse_proteoform_loci` and both `build_sample_tpm_by_*`
assert each column total is preserved across the fold (a pure within-group SUM) —
a regression that drops or double-counts a read fails loudly.

**Curated panels must reference the proteoform id**, never a member symbol —
`tests/test_proteoform_key_space.py` AST-scans every panel module and fails
loudly if one references an unfolded member. Fold a panel at a lookup with
`common.fold_panel_symbols`.
