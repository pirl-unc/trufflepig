# trufflepig

> RNA tumor analysis driven by [`pirlygenes`](https://github.com/pirl-unc/pirlygenes) gene sets.

## What this is

`trufflepig` is the analysis, plotting, reporting, and CLI layer for RNA
tumor analysis. It loads curated gene sets and reference expression data
from the [`pirlygenes`](https://github.com/pirl-unc/pirlygenes) package,
which is now data-only.

The legacy `pirlygenes analyze` CLI has been **fully migrated** into
this repo as `trufflepig run`. Multi-sample longitudinal comparison
(`pirlygenes compare-analyze`) is `trufflepig compare`. The CLI and web UI
use the same production analysis and reporting path.

## Documentation

Start with the [documentation map](docs/README.md). It points first to the
end-to-end production workflow, then to classifier design records, report
consistency work, calibration, and performance notes.

## How the analysis works

The pipeline makes and finalizes one cancer call before it interprets purity,
tumor-attributed expression, or therapy relevance:

1. **Expression QC** loads the input expression file, maps gene identifiers,
   checks TPM scale, and removes technical RNA from clean TPM used downstream.
   Outputs: clean expression table and QC warnings.
2. **RNA Prep and Preservation** infers library prep, preservation,
   degradation, and assay caveats that affect confidence and expression
   interpretation. Outputs: prep/preservation calls, degradation flags, and
   widened uncertainty when needed.
3. **Tissue Composition Screen** compares the sample with normal tissues and
   cancer-expression references before the cancer-type call; it also adds
   tumor-evidence signals such as proliferation, CTA/oncofetal markers, and
   tumor-up markers. Outputs: healthy/tumor hint, top normal matches, and top
   cancer-reference matches.
4. **Cancer-Type Evidence** combines expression-reference matching,
   rare-marker/fusion evidence, exact local references, and registry
   relationships into one cancer-type call. Outputs: inferred cancer type,
   expression reference used for cohort math, and alternate hypotheses.
5. **Tumor Purity and Coarse Composition** estimates tumor fraction and broad
   non-tumor compartments such as immune, stromal, epithelial matched normal,
   and other background components. Outputs: purity interval and fitted
   compartment fractions.
6. **Subtype and Background Refinements** refines the coarse composition with
   activated background states such as CAF/TAM/Treg/MDSC and matched-normal
   compartments. Outputs: immune/stromal infiltration, subtype/background
   adjustments, and matched-normal splits used before target ranking.
7. **Tumor-Attributed Expression** subtracts fitted non-tumor signal and
   estimates how much observed expression is likely tumor-cell derived.
   Outputs: tumor-source TPM ranges, attribution flags, and confidence tiers.
8. **Therapy Prioritization** ranks actionable targets and pathway states
   using supplied patient treatment outcomes first, then sourced clinical
   benefit/toxicity evidence, indication curation, antigen-presentation status,
   tumor-attributed expression, immune/background attribution, and
   pathway/treatment-state signals.
   Outputs: therapy shortlist, target tables, pathway/treatment-state evidence,
   and caveats.

## Install

```
pip install -e .
```

Pulls `pirlygenes>=6.0.2` for the curated gene sets and reference data.

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
  analyze/             # full analyze output: figures, markdown reports, TSVs
```

Every analysis artifact lives under `analyze/`. Start with the automatically
generated `*-interpretive-report.pdf`: it includes the clinical summary, full
treatment rationales and requirements, and figures supporting the final call.

Common pass-through flags: `--hla-types`, `--fusions`, `--variants`,
`--treatment-history`, `--clinical-context`,
`--alignment-qc`, `--sample-mode`, `--tumor-context`, `--site-hint`,
`--met-site`, `--decomposition-templates`, `--output-image-prefix`,
`--sample-id-col`, `--sample-id-value`, `--gene-id-col`, `--gene-name-col`,
`--label-genes`, `--genes`, `--transcripts`,
`--aggregate-gene-expression`, `--expression-qc-rescue`,
`--therapy-target-top-k`, `--therapy-target-tpm-threshold`, `--force`.
See the input contracts below for the accepted evidence and its interpretation.

`--clinical-context` accepts a versioned JSON input with specimen-scoped clinical
MSI/MMR assays, results, quality and provenance. A usable positive assay can
satisfy the MSI-H/dMMR biomarker requirement; an RNA proxy cannot. Python and the
web form use the same [clinical-context contract](docs/clinical-context.md).

HLA inputs use mhcgnomes nomenclature and retain typing resolution and
annotations. Registered therapy requirements include explicit exclusions;
unresolved typing does not establish a match. See [HLA inputs and therapy
requirements](docs/hla-inputs.md).

`--variants` accepts a variant table or an explicit symbolic call such as
`"EGFR KDD"`. Fusions supplied through `--fusions` enter the same normalized
variant evidence stream. MSI-H/TMB-like sample states are not variants and are
kept separate. The old `--alterations` spelling remains a hidden compatibility
alias. VCF and MAF fail closed until their standards-aware adapters land; see
[variant input and coordinate provenance](docs/variant-inputs.md) and issues
[#140](https://github.com/pirl-unc/trufflepig/issues/140) and
[#141](https://github.com/pirl-unc/trufflepig/issues/141).

`--treatment-history` accepts CSV, TSV, JSON, or JSONL with `therapy`,
`status`, and optional `target`, `modality`, `note`, and `source` fields.
Patient outcomes are considered before the RNA model's target support: prior
benefit can keep a treatment path visible even when expression is assigned to
background, while prior progression, lack of benefit, intolerance, or a
contraindication keeps the same treatment out of the shortlist. This input is
clinical context; it does not establish current eligibility or make retreatment
appropriate.
See [treatment history input](docs/treatment-history.md).

Python callers use `trufflepig.brief.recommend_therapies`, the same selection API
as the production summary. It returns `TherapyRecommendation` records with
named `therapy` and `expression` fields; see the
[Python API example](docs/treatment-history.md#python-recommendation-api).

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

Expression references use one contract internally:

1. All analysis references are clean TPM. Raw TPM is only used in the
   early expression-QC stage.
2. Direct references keep their gene key explicit: pirlygenes observed
   cohorts and pan-cancer references are keyed by Ensembl ID + symbol;
   trufflepig subtype-deconvolved references are symbol-only because the
   source deconvolution artifacts are symbol-level.
3. Cancer-type context distinguishes the cancer label from the expression
   reference. If a registry code has no exact expression cohort,
   trufflepig records the compatible parent, curated, or family fallback
   used for cohort math.
4. The registry-completeness tests require every cancer type to have an
   effective expression reference and verify the normalization/gene-key
   contract for those references.
5. Cancer types without a direct expression cohort also have a compact
   literature-backed RNA signature tied to that related reference context.
   These signatures can add marker evidence, but they are not treated as
   replacement expression cohorts.
6. Every registry tumor type is placed in a small ontology record with
   its parent/family, effective expression reference, expected high RNA
   markers, and expected low contrast markers. Reports use those markers
   as a sanity check on the inferred cancer type; expected-low genes are
   review prompts, not standalone exclusions, because high values can
   come from immune, stromal, or mixed-lineage background.

### Web UI

```
pip install 'pirl-trufflepig[web]'
trufflepig serve --port 8000
# open http://127.0.0.1:8000
```

Upload a TPM file or salmon quant in the browser, watch each pipeline
stage stream back, and read the rendered `summary.md` / `analysis.md` /
`brief.md` inline. Comparison runs work the same way — pick prior runs
by ID. Each run writes a self-contained workspace under
`$TRUFFLEPIG_WEB_ROOT` (default `$HOME/trufflepig-web-runs`).

## Layout

```
trufflepig/
  cli.py            # argparse entry exposed as the `trufflepig` console script
  main.py           # migrated analyze/compare_analyze + report assembly
  workspace.py      # workspace root and run metadata
  report_pdf.py     # reader PDF from the finalized report document
  analyze/          # data contracts shared with the migrated pipeline
  decomposition/    # compartment-fit engine + panels + plot helpers
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

### Reporting

The production path runs the complete analysis and finalizes Markdown, JSON,
and the reader PDF together. Unimplemented stage commands and their unused
record directories have been removed.

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

- [x] FastAPI app + browser UI (`trufflepig serve`) with file upload,
      background analyze, server-sent-events progress stream, inline
      rendered reports, and longitudinal comparison launcher
      ([#16](https://github.com/pirl-unc/trufflepig/issues/16))
- [x] Streaming progress + per-stage output hooks (SSE stream of
      analyze stdout)
      ([#15](https://github.com/pirl-unc/trufflepig/issues/15))
- [ ] Reference-data layout for lazy-load from R2/S3 with browser cache
      ([#18](https://github.com/pirl-unc/trufflepig/issues/18))
- [ ] Pyensembl-free gene resolution (HGNC CSV dict lookup) for fast
      cold-start in serverless / browser contexts
      ([#17](https://github.com/pirl-unc/trufflepig/issues/17))
- [ ] Auth + workspace persistence so a user can return to a prior run
- [ ] Production deploy target (serverless) replacing the local
      subprocess runner with a remote-job submission

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

### Report interpretation and Python API

The summary and interpretive PDF follow four sections: conclusion with evidence;
therapy rationale and blockers; one consolidated information list; then detailed
evidence and figures. Their shared content is retained directly in report JSON
schema 2. The PDF contains searchable text and clickable source citations.

Use `trufflepig.report_content.assess_therapy` to inspect an individual therapy's
identity, RNA observation and eligibility requirements, including treatments that
were excluded. Use `trufflepig.therapy_eligibility.evaluate_therapy_eligibility`
for the shared requirement decision. A supplied file is not proof of a matching
mutation, and unknown or conflicting evidence does not establish eligibility.
See [report content and rendering](docs/report-language.md).
