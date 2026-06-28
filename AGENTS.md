# Repository notes for Codex

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

## Pipeline: evidence → cancer-type call → decomposition → aneuploidy

`analyze_sample` → `rank_cancer_type_candidates` → (winner) `decompose_expression`. Whole-profile
signals are computed ONCE per sample and shared across stages.

- **signature_score** (`plot_embedding._compute_cancer_type_signature_stats`): mean over a ~20-gene
  panel of `cohort_pct × within_sample_pct`, per type; cohort reference = the 170-cohort HK-bridged
  matrix (118 cancer + 50 normal). Cached as `stats`.
- **centroid correlation** (`cancer_type_centroid.centroid_correlations`): whole-transcriptome Spearman
  to ~116 cohort centroids; `compartment_call` → coarse group + confidence. Computed once
  (`_ranker_cen_corr`), reused everywhere.
- **purity** (`estimate_tumor_purity`, per candidate): signature/ESTIMATE/lineage; reuses
  `_ranker_expr_lineage` (no centroid recompute).

**Cancer-TYPE call** = rank by `support_score` = GEOMETRIC MEAN of `signature × purity ×
lineage_support × signature_stability × family_factor × centroid_factor` (`_candidate_support_score`).
`centroid_factor` (`_centroid_support_factor`) = the candidate's centroid correlation RELATIVE to the
best, raised to `_CENTROID_FACTOR_POWER` — demotes whole-profile-mismatched candidates the ~20-gene
signature can't catch (eval: `scripts/classification_strategy_eval.py`). Gated by
`TRUFFLEPIG_CENTROID_IN_SUPPORT` (default on).

**Decomposition** (`decompose_expression`, winner only): FEATURE SPACE = within-sample **percentile**
(`space="percentile"`). Lineage-routed by `compartment_call` → one of 4 modes (solid / mesenchymal /
heme / embryonal). Each mode: NNLS background subtraction over stroma/immune/normal templates →
**residual** = `sample − reconstructed_background` (clipped ≥0). `residual_fraction` = PRIMARY purity.

**Aneuploidy**: **bulk** (`bulk_aneuploidy_amplitude`, on BULK, ∝ purity) is a purity CORROBORATOR —
`aneuploidy_purity = clip(A_obs / A_ref(type), 0, 1)`, `A_ref` from `purity_calibration` (subtype →
parent fallback). **residual** aneuploidy (on the purity-corrected residual) is CHARACTERIZATION (with
proliferation), orthogonal to lineage. Downstream: type call picks the cohort → decomposition routes by
its compartment → residual_fraction (primary purity) + bulk-aneuploidy corroborator + residual
characterization.
