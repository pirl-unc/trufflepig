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
