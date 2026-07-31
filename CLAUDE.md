# Repository notes for Claude

## Running tests

**Always use `./test.sh`, not raw `pytest`.**

Direct and IDE-driven pytest launches default to serial. A repository-root
pytest guard rejects `-n auto` or any unbudgeted request for more than one
worker before xdist starts its pool. Each worker re-imports trufflepig +
pirlygenes and loads the reference matrices (`pan-cancer-expression`,
`tcga-deconvolved-expression`, `hpa-cell-type-expression`,
`subtype-deconvolved-expression`). The panic snapshot caught workers at
6.3–7.4 GB RSS, and a later serial full-suite run reached ~9.6 GB.

`./test.sh` reserves at least 8 GB for the OS/apps, budgets at least 12 GB per
worker, and passes the resulting safe cap as `-n`. Both the wrapper and root
pytest guard use one per-user lock, so accidental overlapping suites fail
before a second worker pool starts.

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

**Curated panels stay in natural member symbols** (`HBA1`/`HBA2`, `CTAG1B`, …);
their **accessors fold** to the proteoform key at lookup via
`common.fold_panel_symbols`, so curation auto-adapts to whatever pirlygenes
currently groups (no baked `HBA1/2` label to silently miss if the group changes).
Live fold points: `signature.get_component_markers`, the
`templates.OPTIONAL_COMPARTMENT_GATES` detection loop, and the literature-signature
consumers (`literature_signature_rules_df`, `tumor_type_ontology` expectations).
A consumer that reads a panel **raw** (bypassing its folding accessor) silently
misses a folded member — route every panel→expression lookup through the accessor
or `fold_panel_symbols`. `tests/test_proteoform_key_space.py` pins that each live
accessor folds.

## Pipeline: evidence → cancer-type call → decomposition → aneuploidy

`analyze_sample` → `rank_cancer_type_candidates` → (winner) `decompose_expression`. The expensive
whole-profile signals are computed ONCE per sample and shared across stages — the same evidence is
visible to type-calling, decomposition, and purity.

**Three evidence signals (each computed once):**
- **signature_score** (`plot_embedding._compute_cancer_type_signature_stats`) — mean over a curated
  ~20-gene panel of within-sample clean-TPM percentile, per cancer type. Cross-cohort clean-TPM
  contrasts derive the panels, while the sample ranks over a fixed reference gene universe. Panels
  exist for ~93 types including the missing base types. Cached as `stats`.
- **centroid correlation** (`cancer_type_centroid.centroid_correlations`) — Spearman of the WHOLE
  transcriptome to ~116 cohort centroids; `compartment_call` aggregates it to a coarse histogenesis
  group + confidence. Computed once as `_ranker_cen_corr`, reused everywhere.
- **purity** (`estimate_tumor_purity`, per candidate) — signature/ESTIMATE/lineage; its compartment
  gate reuses `_ranker_expr_lineage`, so it does NOT recompute the centroid.

**Cancer-TYPE call** = rank by `support_score`, the GEOMETRIC MEAN of:
  `signature × purity × lineage_support × signature_stability × family_factor`
`_candidate_support_score` (tumor_purity.py). The whole-profile centroid is NOT a support-score factor
(folding it in was a documented negative result — see the note in tumor_purity). Instead the centroid
drives the call two ways it's stronger at: the `compartment_call` leaf-restriction (coarse,
confidence-gated, #83) and centroid-authoritative fine-subtype resolution + veto (`resolve_fine_subtype`
/ `_apply_centroid_fine_subtype_veto`, #98).

**Decomposition** (`decompose_expression`) runs ONCE for the winner. FEATURE SPACE =
**housekeeping-normalized clean TPM**, retained because it has the lowest component-attribution
error across the controlled known-mixture benchmark; percentile, log1p, TPM-fraction, z-score, and
raw clean TPM are regression-tested alternatives. It is lineage-ROUTED by `compartment_call` →
one of 4 modes: **solid / mesenchymal / heme / embryonal** (not confident → run top+runner-up, resolve
by residual `lineage_fit`). Each mode runs an **NNLS background subtraction** over that mode's
stroma/immune/normal-tissue templates → tumor-specific **residual** = `sample − reconstructed_background`
(clipped ≥0). `residual_fraction` (residual's mass fraction) is the PRIMARY purity signal.

**Aneuploidy — two distinct readouts:**
- **bulk** (`bulk_aneuploidy_amplitude`) — chromosome-arm coherence on the BULK sample (noise-floor
  subtracted), ∝ purity → a purity CORROBORATOR. `aneuploidy_purity = clip(A_obs / A_ref(type), 0, 1)`,
  where `A_ref` (`purity_calibration.aneuploidy_reference`) extrapolates the cohort reference amplitude
  to purity≈1 at the type's median purity. (Mixed-case subtype codes fall back to the parent reference.)
- **residual** (`aneuploidy_score(residual)`) — same arm-coherence on the purity-corrected RESIDUAL;
  tumor CHARACTERIZATION (with proliferation), orthogonal to lineage.

**Downstream fit:** type call selects the cohort → decomposition routes by that cohort's compartment
and yields the residual + `residual_fraction` (primary purity) → bulk aneuploidy corroborates purity
(gated low when ambiguous) → residual aneuploidy + proliferation characterize the tumor on the
purity-corrected residual.
