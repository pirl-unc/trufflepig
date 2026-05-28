# Decomposition calibration notes

## Why this document exists

PR #41 removed decomposition's `renormalize_to_million=False` opt-out
that previously let the decomposition engine see "native-scale"
pirlygenes reference matrices. Templates were recalibrated against the
new TPM-1e6 footing:

| Parameter | Old | New | Justification |
| --- | --- | --- | --- |
| `met_origin_penalty_gain` | 0.6 | 1.0 | Sharper penalty on metastatic origin mismatch under the new scale. |
| `met_origin_penalty_floor` | 0.65 | 0.35 | Looser floor — origin mismatches can collapse further when site evidence is strong. |
| `fit_score_base` | 0.35 | 0.30 | Slightly lower minimum fit weight when cancer support is absent. |
| `fit_score_gain` | 0.65 | 0.70 | Slightly larger linear coefficient on cancer support. |
| `met_site_dominance_min_delta` | — (new) | 0.10 | Site-dominance boost: minimum site-vs-origin score delta. |
| `met_site_dominance_min_extra_fraction` | — (new) | 0.25 | Site-dominance boost: minimum host-compartment fraction. |
| `met_site_dominance_gain` | — (new) | 12.0 | Site-dominance boost multiplier coefficient. |
| `met_site_dominance_cap` | — (new) | 3.0 | Site-dominance boost cap. |

The new `met_site_dominance_*` boost compensates for the fact that, on
TPM-renormalized references, liver bulk no longer dominates absolute
magnitude — hepatocyte-heavy samples should still route to met-liver
templates by host-site evidence rather than generic immune/stromal fit.

## How to re-run the calibration

The headline numbers below were captured at `--samples-per-cohort 5`
(`n=160`); 2-per-cohort is fast enough for quick smoke checks.

```bash
# Full sweep (recommended after touching decomposition scoring).
PYTHONPATH=$HOME/code/pirlygenes:$PYTHONPATH \
    python scripts/calibrate_decomposition.py \
        --eval-tpm $HOME/code/pirlygenes/eval/eval_tpm.parquet \
        --eval-samples $HOME/code/pirlygenes/eval/eval_samples.pkl \
        --samples-per-cohort 5 \
        --include-local-no-override \
        --out-json calibration-report.json \
        --out-tsv calibration-report.tsv

# Diff against a prior committed baseline.
PYTHONPATH=$HOME/code/pirlygenes:$PYTHONPATH \
    python scripts/calibrate_decomposition.py \
        --samples-per-cohort 5 \
        --include-local-no-override \
        --baseline-json docs/calibration-baseline.json \
        --out-json calibration-report.json

# Optional: full TCGA matrix (~10k samples, multiprocessing).
PYTHONPATH=$HOME/code/pirlygenes:$PYTHONPATH \
    python scripts/calibrate_decomposition.py \
        --use-full-tcga-matrix \
        --workers 6 \
        --include-normals \
        --skip-hpa \
        --out-json calibration-full-tcga.json
```

The script runs five tracks (any combination):

1. **TCGA per-sample classification.** Each TCGA sample from
   `pirlygenes/eval/eval_tpm.parquet` is fed through
   `rank_cancer_type_candidates` and the top call is compared to the
   known cohort label. The `eval_tpm` matrix stores `log2(TPM+0.001)`
   so the script back-transforms and TPM-1e6 renormalizes per sample.
   Reports per-cohort top-1 / top-3 and a 95% bootstrap CI on the
   global accuracy (2000 resamples, fixed seed).
2. **HPA normal-tissue baseline.** Each HPA `*_nTPM` column is treated
   as a synthetic sample and passed to `assess_tissue_composition`.
   Lymphoid, mesenchymal, and CTA-rich tissues are *expected* to land
   in the structural-ambiguity bucket; the rest should be
   `healthy-dominant`.
3. **Full TCGA matrix** (`--use-full-tcga-matrix`, opt-in). Replaces
   the 160-sample eval-track with the full
   `~/code/pirlygenes/eval/tcga_RSEM_gene_tpm.gz` matrix (~10,500
   labelled samples). The first invocation materialises a float32
   parquet cache under `~/.cache/trufflepig/` (~2 min, ~1.5 GB);
   subsequent invocations load in seconds. Each worker process loads
   the parquet once via `multiprocessing` initializer, so memory
   scales as `workers × ~3 GB` rather than per-task. Use
   `--workers N` to parallelise (recommended: `--workers 4–8` on a
   16 GB+ machine), and `--max-tcga-samples N` to cap. Ground-truth
   labels come from `barcode_to_project.pkl`; sample-type suffix `01`
   is primary tumor, and `--include-normals` also evaluates `11`
   (matched normals) under a `NORMAL:` prefix so the classifier's
   healthy-call discrimination is testable at scale. **This is not
   the canonical headline metric** — the eval-set 5/cohort run is —
   but it's the right tool for ad-hoc deep dives, e.g. confirming a
   selector regression that only shows up at scale.
4. **Local 17-sample no-override replay** (`--include-local-no-override`).
   Reads the pirlygenes-style
   `local_reports/<run>/manifest.json`, runs each input through the
   broad RNA classifier + `cancer_type_evidence` selector with **no
   `--cancer-type` override**, and compares the inferred label against
   the per-sample claims in PR #41's description (PFO004→OS,
   PFO017→BLCA, PFO019→NUTM, etc.). The expected-labels table is
   bundled inside the script under `_LOCAL_REPLAY_EXPECTED`. Reports
   per-sample `code_match` (inferred label is in the accepted set)
   and `full_match` (label and `selected_by` both match).
5. **Legacy local report scan** (`--local-data-root`). Walks an
   existing `trufflepig-local-reports` tree for `analysis.json`
   files and optionally diffs against an `--expected-labels` TSV.

### Regression-guard mode

Pass `--baseline-json path/to/previous-calibration.json` and the
report grows a `vs_baseline` section with per-cohort and per-sample
deltas plus an explicit `tcga_sample_regressions` /
`local_sample_regressions` list of samples that USED to match and
now don't. This is the recommended path when changing decomposition
scoring constants — bisect bug → run with `--baseline-json` → look
at the regression list.

## Headline numbers (recorded on the PR #41 calibration pass)

These were captured by running the script above on
`pirlygenes==5.2.7`, `trufflepig` at the head of
`codex/consolidate-cancer-type-evidence`, 2 samples per TCGA cohort,
all HPA tissues.

### TCGA per-sample classification

Layer-1 results, broad RNA classifier only (`rank_cancer_type_candidates`):

| | n | top-1 (95% CI) | top-3 (95% CI) |
| --- | ---: | ---: | ---: |
| 2 samples / cohort | 64 | 0.812 (0.703–0.906) | 0.938 (0.875–0.984) |
| 5 samples / cohort | 160 | 0.794 (0.731–0.856) | 0.931 (0.888–0.969) |

95% CIs come from a fixed-seed 2000-resample bootstrap over the
per-sample correctness vector. The 5/cohort headline is the canonical
regression baseline; 2/cohort is the fast smoke check.

Per-cohort breakdown (5 samples / cohort) — only cohorts below 0.80
top-1 are listed so the table stays readable; full per-cohort numbers
are in the JSON artifact:

| Cohort | top-1 | top-3 | Where the misses go |
| --- | ---: | ---: | --- |
| ESCA | 0.00 | 1.00 | HNSC / STAD / LUSC (foregut-squamous) |
| MESO | 0.20 | 0.40 | LUAD / DLBC / SARC (lung pleural / mesenchymal overlap) |
| CESC | 0.40 | 1.00 | HNSC / LUSC / BRCA (squamous) |
| COAD | 0.40 | 1.00 | READ (colon/rectum continuum) |
| SARC | 0.40 | 1.00 | UCS / BRCA / THYM (mesenchymal/mixed) |
| CHOL | 0.60 | 0.60 | COAD / READ (bile-duct → CRC space) |
| LGG | 0.60 | 1.00 | GBM (glioma grade continuum) |
| STAD | 0.60 | 1.00 | foregut neighbors |
| UCEC | 0.60 | 0.60 | UCS / OV (gyn neighbors) |

All other cohorts hold ≥ 0.80 top-1 / ≥ 0.80 top-3. The two failure
modes — biological-neighbor confusion (ESCA↔HNSC, COAD↔READ,
CHOL↔COAD) and mesenchymal-mixed (MESO, SARC) — are well-known and
both rescued by top-3 in nearly every case.

### Per-step layer breakdown

Every TCGA-track per-sample row records two outputs in parallel —
the broad RNA classifier (`rank_cancer_type_candidates`) and the
consolidated call (`select_report_scope_from_evidence`) — plus the
selector that promoted the consolidated call. The summary reports
both top-1 numbers, the *delta* between them, a flip table, and
per-selector accuracy on the consolidated layer.

Headline numbers on the 5/cohort eval set (`n=160`):

| Layer | top-1 | top-3 |
| --- | ---: | ---: |
| Broad RNA classifier (`rank_cancer_type_candidates`) | **0.794** (CI 0.731–0.856) | **0.931** (CI 0.888–0.969) |
| Consolidated (`select_report_scope_from_evidence`)   | **0.725** (CI 0.656–0.794) | — |
| Δ (consolidated − broad), top-1                       | **−0.069**                | — |

Flip table:

| | n |
| --- | ---: |
| both_correct                              | 116 |
| both_wrong                                |  33 |
| broad_correct_consolidated_wrong (regression candidates) | **11** |
| broad_wrong_consolidated_correct (lift candidates)       |   0 |

Per-selector accuracy on the consolidated layer (TCGA-160):

| Selector | n promoted | top-1 |
| --- | ---: | ---: |
| `primary_expression_match`      | 148 | 0.784 |
| `local_expression_reference`    |  10 | **0.000** |
| `rare_marker`                   |   2 | **0.000** |
| `fine_reference`                |   0 | — |
| `tumor_label_refinement`        |   0 | — |
| `direct_fusion`                 |   0 | — |

**How to use this for regression isolation.** When a future
calibration shows `consolidated_top1_accuracy` dropping:

| `broad_top1` | `consolidated_top1` | Where to look |
| --- | --- | --- |
| stable | dropped | Cancer-type-evidence layer regressed (selector gates, tie-break, or fine/local-reference panels). |
| dropped | dropped | Broad RNA classifier regressed (`rank_cancer_type_candidates`, family scoring, signature stats). |
| stable | improved | Cancer-type-evidence layer is doing more lifting — usually good, but check the `per_selector` rows for any selector hitting suspiciously high `n`. |

The `broad_correct_consolidated_wrong` count plus the per-selector
top-1 accuracies pinpoint which selector is misfiring. The
companion `tcga_sample_regressions` in the `--baseline-json` diff
lists the exact sample IDs.

**Live finding (informational, not a regression introduced by PR #41
— surfaced *by* PR #41's per-step calibration, so the workflow is
working as intended).** On adult TCGA bulk samples, `local_expression_reference`
and `rare_marker` over-promote whenever they fire:

- `local_expression_reference` promoted on **10 / 160** TCGA samples
  and got **0** right — every one of these is a sample whose broad
  RNA top-1 was correct, then got flipped to a pediatric/refined
  registry label by the local-marker-panel gate (e.g. `LIHC` →
  `HEPB`, `PCPG` → `PANNET`).
- `rare_marker` promoted on **2 / 160** TCGA samples, both wrong.
- These 12 promotions account for **all 11 of the `broad_correct_
  consolidated_wrong` flips** plus a small selector-attribution
  difference.

The promotions are *internally consistent* — the samples do express
the refined-entity marker panel — but *contextually wrong* for adult
TCGA bulk. Compare this to the local-19 replay below where the same
selectors fire correctly every time on samples that genuinely *are*
the refined entity. A follow-up gate (e.g. require broad-RNA support
for the refined cohort *and* age/lineage disambiguation before
promoting away from the broad call) would close this without
penalising the rare-entity inputs the selectors were built for.
The full per-cohort 5/cohort breakdown is in the headline-numbers
section above.

### Local 17-sample no-override replay

Re-ran on 2026-05-23 with the bundled `_LOCAL_REPLAY_EXPECTED`
table against
`pirlygenes/local_reports/rnaseq-20260506-132208/manifest.json`:

| | n | code-match | full-match |
| --- | ---: | ---: | ---: |
| Overall | 19 | 1.000 (19/19) | 1.000 (19/19) |

Per-step layer breakdown on the local replay:

| Layer | Code-match rate | Lifted by consolidation |
| --- | ---: | ---: |
| Broad RNA classifier only | **0.632 (12/19)** | — |
| Consolidated (cancer_type_evidence) | **1.000 (19/19)** | **+7 samples** (Δ = +0.368) |

Per-selector on local-19 (consolidated):

| Selector | n | code-match rate |
| --- | ---: | ---: |
| `primary_expression_match` | 12 | 1.000 |
| `rare_marker`              |  5 | 1.000 (all NUTM1) |
| `fine_reference`           |  2 | 1.000 (both PFO004 OS) |

The local replay is the cleanest signal for whether the
cancer_type_evidence layer is actually *adding* lift over the broad
RNA classifier: every PFO004 (OS), PFO019 / Tempus NUTM1 (NUTM)
sample relies on the consolidated layer to flip out of the broad
SARC / LUSC call. The `n_lifted_by_consolidation` field in the
summary is the count of samples where the broad classifier alone
would have been wrong but the consolidation got it right.

The **contrast with TCGA-160 is the point**: the *same* `rare_marker`
and `local_expression_reference` selectors that hit 0% top-1 on
canonical adult TCGA bulk hit **100%** on the local replay because
the local samples genuinely *are* refined entities. The selectors
aren't broken — their gating is over-eager. Any future tightening
should be measured by re-running both tracks: TCGA-160's
`broad_correct_consolidated_wrong` flip count should drop while the
local-19 `n_lifted_by_consolidation` count should stay at 7.

Per-sample expected label vs. observed selected scope:

| Sample | Expected | Observed `selected_by` | Inferred | Pass |
| --- | --- | --- | --- | :---: |
| sid-pfo004-osteosarcoma-gene-expression | OS / fine_reference | fine_reference | OS | ✓ |
| sid-pfo004-osteosarcoma-salmon-gene-tpm | OS / fine_reference | fine_reference | OS | ✓ |
| pfo002-personalis-gene-expression | COAD\|READ | primary_expression_match | READ | ✓ |
| pfo002-washu-kallisto-gene-abundance | COAD\|READ | primary_expression_match | READ | ✓ |
| pfo002-washu-stringtie-gene-expression | COAD\|READ | primary_expression_match | READ | ✓ |
| rs-tempus-salmon-rich-gene-tpm | PRAD | primary_expression_match | PRAD | ✓ |
| asy-salmon-gene-tpm | COAD\|READ | primary_expression_match | READ | ✓ |
| alvin-salmon-gene-tpm | SARC | primary_expression_match | SARC | ✓ |
| hcc1395-kallisto-gene-abundance | BRCA | primary_expression_match | BRCA | ✓ |
| hcc1395-stringtie-gene-expression | BRCA | primary_expression_match | BRCA | ✓ |
| pfo019-kallisto-gene-abundance | NUTM / rare_marker | rare_marker | NUTM | ✓ |
| pfo019-stringtie-gene-expression | NUTM / rare_marker | rare_marker | NUTM | ✓ |
| pfo017-pfo017-bladder-2023 | BLCA | primary_expression_match | BLCA | ✓ |
| pfo017-pfo017-bladder-2025 | BLCA | primary_expression_match | BLCA | ✓ |
| pfo017-pfo017-liver-2023 | BLCA | primary_expression_match | BLCA | ✓ |
| tempus-nutm1-unc_0026-tl-21-ks8t3eyh | NUTM / rare_marker | rare_marker | NUTM | ✓ |
| tempus-nutm1-unc_0027-tl-21-tayehe9b | NUTM / rare_marker | rare_marker | NUTM | ✓ |
| tempus-nutm1-unc_0028-tl-20-10a3fc | NUTM / rare_marker | rare_marker | NUTM | ✓ |
| low-purity-prad-cegat-salmon-gene-tpm | PRAD | primary_expression_match | PRAD | ✓ |

Notable behaviours empirically confirmed:

- **OS via fine-reference inside SARC context.** Both PFO004 inputs
  have a top broad call of SARC; the `fine_reference` selector
  promotes OS because the osteogenic marker panel (RUNX2, SATB2,
  IBSP, SPP1, COL1A1/A2, DLX5, DMP1, MEPE) lights up plus the
  amplicon panel (MDM2, CDK4, FRS2).
- **NUTM via rare_marker inside LUSC context.** All five NUTM cases
  (PFO019 ×2, Tempus NUTM1 ×3) have a top broad call of LUSC; the
  NUTM1 RNA-surrogate rule promotes NUTM under the squamous-context
  gate.
- **PFO017 liver met → BLCA primary.** Liver-tissue HPA signature
  does not override the bladder-primary classification — the BLCA
  expression signature is intact even in metastatic liver tissue.
- **CRC continuum.** Every PFO002 / ASY sample returns READ rather
  than COAD; both are accepted per the expected-labels rubric since
  the colon/rectum continuum is biologically the same call.

### HPA normal baseline

Every HPA tissue in the reference is biologically **healthy normal
tissue**. The healthy-vs-tumor classifier's job on this synthetic
sample set is therefore to label all 50 as `healthy`. The fraction
it gets wrong (`tumor-like` on a known-normal sample) is the
classifier's **false-positive rate on normal tissues**, *not* an
expected positive count:

| Bucket | n | Read it as |
| --- | ---: | --- |
| healthy | 37 | True negatives — correctly identified as normal tissue |
| tumor-like | 13 | **False positives** — known bulk-RNA blind spots; see below |

The 13 false-positive tissues group into three structural ceilings
that bulk RNA expression alone cannot break:

| Tissue | False-positive driver | Why bulk-RNA can't resolve |
| --- | --- | --- |
| bone_marrow, lymph_node, spleen, thymus, tonsil, appendix | `lymphoid-tissue-tumor-indistinguishable` rule | Lymphoid normals ARE the tissues where heme malignancies originate. A real DLBC biopsy is ~lymph-node tissue + a malignant clone; the clone is a small TPM fraction in bulk. The discriminating signals — clonal Ig/TCR rearrangement, κ:λ light-chain restriction, somatic VAF — require WGS/WES or a clonality call, not bulk RNA. |
| smooth_muscle, skeletal_muscle, heart_muscle, adipose_tissue, endometrium | `mesenchymal-ambiguous` rule | A well-differentiated leiomyosarcoma looks like smooth muscle; SARC's `tumor_up_vs_matched_normal` panel is **empty** in pirlygenes because the tumor markers ARE the lineage markers. |
| testis | CTA panel saturation | CTAs are *defined* as testis-expressed; the CTA tumor-evidence channel correctly fires on testis. The CTA-normal-tissue guard zeros the count, but other panels still match. |
| rectum | Proliferation panel | High-turnover epithelium scores like tumor on the proliferation channel. |

These are documented as **expected limitations** rather than
calibration targets to drive down at this layer. The structural
ceiling is well-understood; the fixes require orthogonal data
sources that `assess_tissue_composition` does not currently consume:

- **Lymphoid**: clonal Ig/TCR repertoire concentration (a single
  rearrangement dominating reads), κ:λ light-chain restriction, or
  DNA-side somatic VAF context — none accessed by this module.
- **Mesenchymal**: somatic mutation context, fusion calls, or a
  matched-normal mesenchymal-specific RNA panel that doesn't yet
  exist in pirlygenes.

If a future PR adds any of these signals, the HPA false-positive
count can drop — and that drop becomes the regression metric. Until
then, **13/50 is the floor**, not a number to tune.

We empirically tried adding a "z-score escape" override (heme-tumor-
up panel vs HPA normal-lymphoid baseline) to drop the FP count via
bulk RNA alone. With a strict threshold (z ≥ 2.0, fraction ≥ 0.5)
it never fires on any HPA normal — preserves the 13 baseline, adds
~50 LOC of inert hook code. With a loose threshold (z ≥ 1.5) it
drops one FP (spleen) but via a flaky mechanism: spleen escapes
because of mild elevation on a handful of housekeeping-like genes
that happened to be in the DLBC tumor-up panel, not because of any
real tumor signal. We chose to keep the boolean intersection (no
z-score override) as the most honest reflection of the bulk-RNA
ceiling.

## How to use the report as a regression guard

After a change to decomposition scoring constants, rerun the script
and `diff` the JSON output. Expect cohort top-1 to stay ≥ 0.75 and
top-3 to stay ≥ 0.90 on a 2-sample-per-cohort sweep; deviations
below those bars warrant scoring rollback or a documented note here.

The calibration script intentionally depends on
`pirlygenes/eval/eval_tpm.parquet` which is *not* bundled inside
trufflepig — it lives in the pirlygenes development checkout. That
keeps trufflepig's test suite cheap (no 161-sample classification
sweep on every test run), but means the calibration is a manual
regression artifact, not a CI gate. The shape of the wiring it
exercises is covered by `tests/test_apply_cancer_type_evidence.py`.
