# Repository notes for Codex

## Running tests

**Always use `./test.sh`, not raw `pytest`.**

Direct and IDE-driven pytest launches default to serial. A repository-root
pytest guard rejects `-n auto` or any unbudgeted request for more than one
worker before xdist starts its pool. Each worker re-imports trufflepig +
pirlygenes and loads the reference matrices (`pan-cancer-expression`,
`tcga-deconvolved-expression`, `hpa-cell-type-expression`,
`subtype-deconvolved-expression`). With the current reference bundle, the panic
snapshot caught workers at **6.3–7.4 GB RSS**, and a subsequent serial full-suite
run reached **~9.6 GB**. Never assume the historical ~1.5 GB figure is valid.

`./test.sh` reserves at least 8 GB for the OS/apps, budgets at least 12 GB per
worker, and passes the resulting cap as `-n`. Both the wrapper and root pytest
guard use one per-user lock and refuse to start a second local suite while one
is active; two pools racing on the same free-RAM reading caused a >100 GB
allocation and macOS watchdog panic in July 2026. Do not bypass the lock unless
an external scheduler is enforcing a combined memory budget.

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
lineage_support × signature_stability × family_factor` (`_candidate_support_score`). The whole-profile
centroid is NOT a support-score factor (folding it in was a documented negative result); it drives the
call via the `compartment_call` leaf-restriction (#83) and fine-subtype resolution/veto (#98) instead.

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
