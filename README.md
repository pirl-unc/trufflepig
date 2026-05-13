# trufflepig

> RNA tumor analysis driven by [`pirlygenes`](https://github.com/pirl-unc/pirlygenes) gene sets.

## What this is

`trufflepig` is the analysis, plotting, reporting, and CLI layer for RNA
tumor analysis. It loads curated gene sets and reference expression data
from the [`pirlygenes`](https://github.com/pirl-unc/pirlygenes) package,
which is now data-only.

The legacy `pirlygenes analyze` CLI has been **fully migrated** into
this repo as `trufflepig run`. Multi-sample longitudinal comparison
(`pirlygenes compare-analyze`) is `trufflepig compare`. Per-stage
extraction of the analyze pipeline (so a web UI can stream incremental
results) is the next track.

## Install

```
pip install -e .
```

Pulls `pirlygenes>=5.0.0` for the curated gene sets and reference data.

## Usage

### Single-sample analysis

```
trufflepig run \
    --sample path/to/quant.sf \
    --workspace out/patient_X_baseline \
    --cancer-type BLCA
```

Output layout:

```
out/patient_X_baseline/
  meta.json            # trufflepig run metadata (versions + args)
  analyze/             # full analyze output (figures + markdown)
  records/             # per-stage records (filled as stages migrate)
  figures/             # rendered figures
```

Common pass-through flags: `--hla-types`, `--fusions`, `--alterations`,
`--alignment-qc`, `--sample-mode`, `--tumor-context`, `--site-hint`,
`--met-site`, `--decomposition-templates`, `--output-image-prefix`,
`--sample-id-col`, `--sample-id-value`, `--gene-id-col`, `--gene-name-col`,
`--label-genes`, `--genes`, `--transcripts`,
`--aggregate-gene-expression`, `--expression-qc-rescue`,
`--therapy-target-top-k`, `--therapy-target-tpm-threshold`, `--force`.
All have the same meaning as in the old `pirlygenes analyze`.

### Multi-sample (longitudinal)

```
trufflepig compare \
    --workspace out/patient_X_longitudinal \
    --inputs out/patient_X_baseline,out/patient_X_relapse \
    --title "Patient X — baseline vs relapse"
```

`--inputs` accepts both trufflepig workspaces (auto-descends to
`analyze/`) and legacy pirlygenes output directories.

### Reference / cohort introspection

```
trufflepig data            # list bundled gene-set CSVs and TCGA cohorts
trufflepig cancers         # browse the cancer-type registry
trufflepig cancers --family sarcoma --details
trufflepig plot-cancer-cohorts --output-prefix /tmp/cohort
```

### Pipeline DAG

```
trufflepig list-stages
```

The DAG is the post-migration target for `trufflepig stage <name>`. The
top-level `trufflepig run` already runs the full pipeline; stage-level
execution is wired in as stages are extracted from the migrated
codebase.

## Layout

```
trufflepig/
  cli.py            # argparse entry exposed as the `trufflepig` console script
  main.py           # migrated analyze/compare_analyze + report assembly
  workspace.py      # workspace layout (meta.json + records/ + figures/)
  pipeline.py       # stage DAG (name -> upstream dependencies)
  analyze/          # data contracts shared with the migrated pipeline
  decomposition/    # compartment-fit engine + panels + plot helpers
  stages/           # one module per stage (post-extraction)
  load_expression.py, sample_context.py, tumor_purity.py,
  decomposition/, plot*.py, brief.py, confidence.py, ...   # the analysis code
```

## Roadmap

### Phase 1 — Subsume pirlygenes analyze ✅

- [x] Wire `trufflepig run` as a thin bridge to `pirlygenes.cli.analyze`
      (trufflepig#19)
- [x] Wire `trufflepig compare` as a thin bridge to
      `pirlygenes.cli.compare_analyze`
- [x] **Mass-move analysis modules** from pirlygenes to trufflepig
      (trufflepig#1). pirlygenes now ships data only.
- [x] Native `trufflepig run` / `trufflepig compare` dispatch — no bridge

### Phase 2 — Per-stage extraction

Break the migrated `analyze` function into the stage DAG so a web UI
can run and stream single stages:

- [ ] `load_expression` — parse sample TPM TSV/CSV into a canonical
      frame ([#2](https://github.com/pirl-unc/trufflepig/issues/2))
- [ ] `sample_context` — infer library prep, preservation, degradation
      ([#3](https://github.com/pirl-unc/trufflepig/issues/3))
- [ ] `analyze` — cancer-type call + purity
      ([#4](https://github.com/pirl-unc/trufflepig/issues/4))
- [ ] `decompose` — compartment-level decomposition fit
      ([#5](https://github.com/pirl-unc/trufflepig/issues/5))
- [ ] `ranges` — per-target tumor-expression ranges + attribution
      ([#6](https://github.com/pirl-unc/trufflepig/issues/6))
- [ ] `confidence` — purity + per-target confidence tiers
      ([#7](https://github.com/pirl-unc/trufflepig/issues/7))
- [ ] `render_targets`, `render_summary`, `render_analysis`,
      `render_provenance`, `render_brief`
      ([#8](https://github.com/pirl-unc/trufflepig/issues/8)–[#12](https://github.com/pirl-unc/trufflepig/issues/12))
- [ ] `bundle` — figures into PDF + finalize `meta.json`
      ([#13](https://github.com/pirl-unc/trufflepig/issues/13))
- [ ] Per-stage record schema documentation
      ([#14](https://github.com/pirl-unc/trufflepig/issues/14))

### Phase 3 — Multi-sample / longitudinal

`trufflepig compare` runs today; the richer layer:

- [ ] Explicit delta tables — cancer-call shifts, purity drift, target
      gains/losses, MHC/HLA changes, immune / IFN / hypoxia / EMT /
      therapy-response axis movement, assay/library differences that
      limit comparability (extension of
      [pirlygenes#230](https://github.com/pirl-unc/pirlygenes/issues/230))
- [ ] Cohort-level comparisons (browse N samples with the same cancer
      type; surface outlier targets)
- [ ] Patient-level provenance graph linking baseline → progression
      samples

### Phase 4 — Web UI

A single-page web frontend so a user can drop in a TPM or salmon quant,
watch each stage stream back, and download the rendered markdown / PDF.

- [ ] Architecture: serverless analysis endpoint + browser client
      ([#16](https://github.com/pirl-unc/trufflepig/issues/16))
- [ ] Streaming progress + per-stage output hooks
      ([#15](https://github.com/pirl-unc/trufflepig/issues/15))
- [ ] Reference-data layout for lazy-load from R2/S3 with browser cache
      ([#18](https://github.com/pirl-unc/trufflepig/issues/18))
- [ ] Pyensembl-free gene resolution (HGNC CSV dict lookup) for fast
      cold-start in serverless / browser contexts
      ([#17](https://github.com/pirl-unc/trufflepig/issues/17))
- [ ] Auth + workspace persistence so a user can return to a prior run

## Non-goals

- No JSON mirror of the markdown reports — the rendered markdown has
  named human audiences; a JSON mirror would have no real consumer.
- No change to the gene-set data in `pirlygenes`.

## Local-report regeneration

Researcher workflow: replay a private manifest of analyses on local
samples and write outputs **outside the repo**:

```
python scripts/regenerate_local_reports.py \
    --source /path/to/pirlygenes/local_reports/<run>/manifest.json \
    --root ~/trufflepig-local-reports/<stamp>
```

The script refuses to write inside the repo. The default `--root` is
`$HOME/trufflepig-local-reports/<timestamp>/`.

## License

Apache 2.0 — see [LICENSE](LICENSE).
