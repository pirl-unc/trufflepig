# Changelog

This file records user-facing CLI and library changes that are not
inferable from a diff. Internal refactors and bug-fixes go in commit
messages, not here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
but this project does not version-tag releases yet — entries are grouped
under the merging PR title and the date the breaking change landed on
`main`.

## Unreleased — hierarchical compartment→leaf cancer-type call (PR #86)

### Behavior change

- The cancer-type ranker now runs a **hierarchical** call. A locked
  stage-1 *compartment* classifier
  (`cancer_type_centroid.compartment_call`) decides the histogenesis
  compartment (Epithelial / Sarcoma / Hematolymphoid / Melanoma /
  Neuroendocrine / CNS / Germ cell / Embryonal) by whole-profile
  correlation against the bulk TCGA centroids — a signal immune to the
  marker-panel saturation that mis-routes stroma-heavy / squamous-
  contaminated tumors to SARC/HNSC (15/15 on the local blind truth set).
- When that compartment call is **confident** (clear rho margin),
  `rank_cancer_type_candidates` restricts the leaves considered at
  stage 2 to that compartment: a stable two-tier re-rank floats
  in-compartment leaves above out-of-compartment ones, so the top call
  and the candidate set are drawn from the pinned compartment. Below the
  margin it abstains (no restriction). **Sarcoma stays broad** — every
  `SARC_*` subtype is in-compartment and no single sarcoma leaf is ever
  promoted by the gate. Stage-2 leaf ranking remains the existing
  geomean support score, now operating on the reduced candidate set.

### New surface

- `cancer_type_centroid.compartment_call`, `in_compartment`, and
  `restrict_rows_to_compartment` — the stage-1 API.
- Candidate rows gain `compartment_in_set`; the top row gains
  `centroid_coarse_lineage`, `centroid_lineage_margin`,
  `centroid_lineage_confident`, and `centroid_compartment_restricted`.
  The cancer-candidates TSV surfaces `compartment_in_set`,
  `centroid_coarse_lineage`, and `centroid_compartment_restricted`.
- Architecture + stage-2 bake-off recorded in
  `docs/cancer-type-hierarchical-classifier.md`.

## Unreleased — lineage panels wiring (PR #44)

### New surface

- **`trufflepig.lineage_panels.LINEAGE_PANELS`** is wired into the
  cancer-type evidence consolidator as a new selector
  (`_add_lineage_panel_features`). The panel set covers 15 cohorts
  (BRCA_BASAL, BRCA_LUMINAL, ESCA_SQUAMOUS, BLCA_LUMINAL,
  BLCA_BASAL, HNSC, LIHC, PAAD, CHOL, UCEC, MESO, ACC, THYM,
  PCPG, LUAD) with HGNC-symbol-curated `obligate` + `high_markers` +
  `low_markers` triples. Internal lookups go through Ensembl
  gene_ids; symbol → ID resolution flows through
  `trufflepig.common.panel_symbols_to_gene_ids` (the single
  Ensembl-lookup path).
- **`LineagePanel.program_note`** (new field) carries a one-line
  transcriptional-program note per panel — surfaced in the report
  as the **`Subtype:`** follow-on line. Single source of truth:
  add or edit notes on the panel itself, not in `brief.py`.
- **`analysis["lineage_panel_evidence"]`** now appears on the
  analysis dict when the panel selector ran. Carries
  `top_panel`, `top_score`, `top_rationale`,
  `top_panel_program_note`, `margin_over_second`, and a
  `promotion` block describing whether the panel actually promoted
  a report label or was held back. Lands in
  `*-analysis-parameters.json`.
- **Two new evidence lines in `*-brief.md` / `*-summary.md`**:
  - **`Lineage panel:`** — the raw evidence reading (panel name,
    score, rationale, "supports the X call" / "noted, did not
    change the call (REASON)").
  - **`Subtype:`** — the panel's `program_note` (basal-like
    breast program, hepatocyte program, etc.).

### Calibration metric rename (back-compat aliases shipped)

- **`broad_*` → `first_pass_*`**, **`consolidated_*` → `final_call_*`**
  across the calibration script's JSON / TSV output and the
  bundled `docs/calibration-baseline.json`. The names now narrate
  the pipeline: `first_pass` = the RNA classifier's initial guess,
  `final_call` = the call after the evidence-weighting selectors
  have weighed in. Affected keys (writer + baseline both):
  `broad_top_code` → `first_pass_code`,
  `broad_top_support` → `first_pass_support`,
  `broad_top1_accuracy` → `first_pass_top1_accuracy`,
  `broad_top3_accuracy` → `first_pass_top3_accuracy`,
  `consolidated_cancer_type` → `final_call_cancer_type`,
  `consolidated_top1_accuracy` → `final_call_top1_accuracy`,
  `delta_consolidated_vs_broad_top1` →
  `delta_final_call_vs_first_pass_top1`,
  `broad_correct_consolidated_wrong` →
  `first_pass_right_final_call_wrong`,
  `broad_only_code_match_rate` → `first_pass_only_code_match_rate`,
  `n_lifted_by_consolidation` → `n_lifted_by_final_call`,
  and the per-sample matching variants.
  *Migration:* the old keys are still emitted for one release
  cycle via `_add_back_compat_aliases`. After the next release,
  drop the back-compat helper. The blocker / rationale prose in
  `cancer_type_evidence` and `brief.py` was also updated:
  "broad RNA candidates" → "first-pass RNA candidates", etc.

### Report quality / clinical readability

- **Pessimistic handling for unresolvable panel markers.** A panel
  marker symbol that can't be resolved to an Ensembl gene_id no
  longer silently passes the check. Unresolvable obligate →
  panel returns no-score with explicit rationale. Unresolvable
  high_marker → high_misses. Unresolvable low_marker →
  low_violations (the negative-marker check defaults to "presumed
  expressed", the conservative choice). Each unresolvable marker
  is logged exactly once per process.
- **AI-slop wording audit.** The PanelEvidence rationale and the
  lineage_panel blocker strings dropped robotic phrasing
  (`obligate gate failed` → `required marker {sym} absent`,
  `low-markers compliant` → `negative markers below threshold`,
  `lineage-panel discriminator` → `lineage panel reading`,
  `promoted X as the report label` → `supports the X call` /
  `proposes X`, `recorded as evidence only` →
  `noted, did not change the call`,
  `**Subtype signal:** ... pattern detected` →
  `**Subtype:** {panel} — {note}`).

### Calibration

Calibration on TCGA-160 + HPA-50 + Local-19 against
`docs/calibration-baseline.json`:

| Track | Baseline | This PR |
|---|---|---|
| TCGA-160 `first_pass_top1` | 79.37% | 79.4% (unchanged) |
| TCGA-160 `first_pass_top3` | 93.13% | 93.1% (unchanged) |
| **TCGA-160 `final_call_top1`** | **72.5%** | **76.9%** (+4.4 pp) |
| `first_pass_right_final_call_wrong` | 11 | 4 |
| **`lineage_panel` firings (TCGA)** | — | **13, 100% correct** |
| HPA-50 healthy / tumor-like | 37 / 13 | 37 / 13 |
| Local-19 code match | 100% | 100% |

7 of 11 previously-misconsolidated samples now keep the correct
first-pass call (4× LIHC, 1× CHOL, 2× PCPG) — the lineage_panel
selector defends the first-pass call when its `parent_cohort`
matches the first-pass top-1 (`class_rank=2`,
"same-code reinforcement"). Cross-code promotion stays at
`class_rank=1` so it cannot override a confident first-pass call.


## Unreleased — cancer-type evidence consolidation (PR #41)

### Breaking changes

- **`--expression-qc-rescue` mode parser narrowed** (`trufflepig run`):
  the only accepted values are now `auto` and `always`. The previously
  accepted disabling aliases (`off`, `never`, `none`, `false`, `0`, `no`)
  have been removed. Dirty TPM is no longer a supported analysis mode —
  the pre-rescue raw values are still preserved on the per-sample
  expression-QC record so downstream consumers can inspect them.
  *Migration:* if a wrapper script was passing `--expression-qc-rescue off`,
  drop the flag (or pass `auto`); the analysis pipeline now always
  technical-RNA-normalizes when removable features are present.

- **`support_norm` renamed to `support_fraction_of_top`** (breaking
  JSON/TSV schema change). The field is per-candidate
  `support_score / max(support_score)` — the top candidate is always
  `1.0`, runners-up read as "fraction of the leader's RNA support".
  It is not a probability, not a sum-to-one share, and not a raw
  score. The old `_norm` suffix conflated "normalized in math" with
  "normalized to the max"; the new name makes both jobs explicit.
  Affected outputs: `*-cancer-candidates.tsv`,
  `*-analysis-parameters.json` (`candidate_trace[].support_norm` →
  `candidate_trace[].support_fraction_of_top`), and all
  `*_support_norm` variants in `cancer_type_evidence.details`
  (`competing_background_support_norm`, `tumor_label_support_norm`,
  `parent_support_norm`, `context_support_norm`, etc.).
  *Migration:* `s/support_norm/support_fraction_of_top/g` in any
  consumer code or pinned baseline JSON. No back-compat alias is
  shipped.

- **`renormalize_to_million=False` removed from reference accessors**:
  `trufflepig.reference.pan_cancer_expression`,
  `tcga_deconvolved_expression`, and `subtype_deconvolved_expression` no
  longer accept the `renormalize_to_million` keyword. All three always
  return TPM-1e6-normalized data. Callers that previously opted out for
  decomposition templates now share the same normalized scale; the
  decomposition scoring parameters were recalibrated to match — see
  `scripts/calibrate_decomposition.py` and `docs/CALIBRATION.md`.
  *Migration:* drop the keyword from any direct callers. If you need
  the native-scale data, regenerate it from the underlying pirlygenes
  artifacts directly.

### Performance

- **`_local_expression_reference_panels` cache no longer thrashes.**
  Previously the call site in
  `_add_local_expression_reference_features` keyed the `lru_cache` on
  `tuple(sorted(support_by_code))`, which varied per sample. Each
  analyzed sample rebuilt the full panel set (loading
  `cancer_reference_expression` + `subtype_deconvolved_expression` +
  iterating registry rows). The call site now passes the unfiltered
  default; the consumer filters per-panel against the live
  `support_by_code` so the semantic outcome is unchanged but the
  cache stays hot. Surfaced by per-step calibration where the
  per-sample wall time dropped from ~20 s to ~5 s.

### New surface

- `trufflepig.cancer_type_evidence.select_report_scope_from_evidence`
  is the single entry point for picking the report-label cancer type from
  broad RNA support, related-context support, RNA markers, fine
  references, local expression references, fusions, and tumor-label
  refinement. See `tests/test_cancer_type_evidence.py` for the shape of
  the returned dict.
- `analysis["cancer_type_evidence"]` is now populated whenever no
  user-supplied `cancer_type` and no registry-derived `report_scope` is
  in play. See `_apply_cancer_type_evidence` in `trufflepig/main.py`.

### Report quality / clinical readability

- **"Not measured" cell now distinguishes three states.** Therapy-table
  and biomarker-table cells previously rendered `*not measured*` for any
  target without a row in `ranges_df`, collapsing two clinically distinct
  conditions: (a) gene was measured-as-zero in the input
  (`TPM < 0.01`, real RNA-level negative), and (b) gene symbol was not in
  the input file at all (coverage gap / pipeline issue). Cells now render
  `0.0 (below detection)` and `*not in input*` respectively, with matching
  interpretation-cell text. See `target_observation_state` /
  `format_missing_observation_cell` in `trufflepig/reporting.py`. The
  three-state classifier reads the new
  `ranges_df.attrs["sample_input_symbols"]` set populated by
  `estimate_tumor_expression_ranges`.
- **Confidence badge respects 3-way candidate ties.** Top-3 broad-RNA
  candidates within 15% relative geomean of the top now drop the call
  confidence to `low` / provisional. Previously a sample like
  PFO017-liver (SARC 1.000 / ESCA 0.891 / BLCA 0.867) showed a
  `moderate confidence` badge despite the body explicitly calling the
  fit `ambiguous`. The runner-up threshold also widened from 10% to 15%
  to match the same band. See `compute_call_confidence` in
  `trufflepig/confidence.py`.
- **Summary now surfaces biomarker outliers.** A new
  `## Notable biomarker outliers` section appears in `*-summary.md` for
  curated-key-gene panels: any biomarker-panel gene with TPM ≥ 50 *and*
  (≥ 10× amplified vs peak healthy tissue OR ≥ 95th-percentile in TCGA
  cohort) is listed, ranked by amplification fold. This lifts canonical
  drivers like MDM2 ~38× in OS, RUNX2 ~50×, CDK4 ~14× — previously only
  visible in the analysis-report biomarker panel — into the headline.
  Excludes genes already in the top-therapy block to avoid duplication.
- **Summary now surfaces CTAs.** A new `## Notable CTAs` section lists
  cancer-testis antigens with TPM ≥ 10 (PAGE family in OS, HORMAD1 in
  NUTM, PRAME, etc.) — vaccine / TCR-T-relevant signals that the
  approved-therapy registry typically omits because no FDA agent exists
  for that exact cancer type.
- **Empty therapy shortlist now explains why.** The generic
  `No approved or trialed agents with a measured, tumor-supported target`
  line is replaced by a breakdown: `of N curated agents, X measured
  and present but filtered as non-tumor-supported; Y measured below
  detection (real RNA-level negative); Z not present in input file
  (coverage gap, investigate symbol mapping)`. Distinguishes biology
  from coverage-gap artifacts.

### Inference cleanups

- **Selector-cascade audit fixes** (`trufflepig/cancer_type_evidence.py`):
  - Subtype-deconvolved-expression accessor no longer crashes when the
    DataFrame is missing a `subtype` column (was indexing with a scalar
    `True`, raising `KeyError`).
  - Tie-break in `select_report_scope_from_evidence` is now fully
    deterministic: highest priority first, then *ascending* cancer-type
    alphabetical. The prior `reverse=True` flipped both fields,
    producing Z-before-A on ties.
  - `tumor_label_refinement` no longer defaults missing `support_fraction_of_top`
    to 1.0 (would have silently treated a missing-field background top
    candidate as fully supported) — now defaults to 0.0.
  - `_selection_method_label` returns distinct display strings per
    selector (`local_expression_reference` and `fine_reference`
    previously both rendered as `exact_expression_reference`, making
    JSON traces ambiguous).
- **Selector routing now uses `selected_by`, not evidence-source
  membership** (`trufflepig/main.py` `_apply_cancer_type_evidence`).
  When multiple selectors fire on the same hypothesis, the routing
  into `rare_scope_inference` vs `fine_scope_inference` is now keyed
  off the winning selector, not naive set membership. Prior code
  would route a `fine_reference`-selected hypothesis into
  `rare_scope_inference` purely because `rare_marker` also appeared
  in the source list.
- **`rare_inference.build_sample_tpm_by_symbol` exception narrowed**
  from bare `except Exception` to `(ImportError, KeyError, ValueError,
  TypeError)` to avoid masking unrelated errors.
- **Code-review followups** (`trufflepig/cancer_type_evidence.py`,
  `trufflepig/healthy_vs_tumor.py`):
  - `RareRnaPolicy` gained an explicit `top_context_promotes_to_full`
    field (default `True`, preserves current NUTM / salivary / default
    behavior) and a named `effective_context_support` method. The
    prior `effective_context = max(context_support, float(top_is_context))`
    was load-bearing but anonymous; the new naming makes "broad
    top is one of my compatible contexts → treat context as fully
    supported" inspectable in the policy table.
  - `related_context_is_top` no longer OR-accumulates across panels —
    a passing-but-weaker panel's `True` no longer leaks into a
    stronger panel's `False`. The flag now reflects the *winning*
    panel (highest `local_support` for this hypothesis).
  - Unified `related_context_code` delimiter to `,` across both
    `local_expression_reference` and `rare_marker` selectors
    (previously `,` vs `;` — downstream consumers had to know which
    selector fired to choose the split character).
  - Named `_LOCAL_REFERENCE_BURDEN_SCORE_ANCHOR = 0.35` (was magic
    number two lines below `_LOCAL_REFERENCE_MIN_BURDEN_RATIO = 0.10`,
    confusing on first read).
  - `_public_evidence_sort_key` uses the same explicit-negation
    pattern as the selectable sort — both lists tie-break on
    *ascending* cancer_type, consistently.
  - `healthy_vs_tumor.py:629` wraps `np.log2` input in `np.nan_to_num`
    to suppress `RuntimeWarning: invalid value encountered in log2`
    on missing-symbol NaN values.

### Test additions

- `tests/test_cancer_type_evidence.py`:
  - `test_empty_candidate_trace_does_not_crash` — pins the all-empty
    contract (degenerate analysis with no broad-RNA output).
  - `test_single_element_candidate_trace_runs_only_broad_path` —
    `tumor_label_refinement` requires rank-2 evidence; single-row
    trace must still emit a broad-RNA hypothesis.
  - `test_equally_prioritized_hypotheses_break_ties_alphabetically` —
    deterministic tie-break on identical priority.
  - `test_subtype_deconvolved_expression_without_subtype_column` —
    defensive against the prior `sub[scalar_bool]` crash bug when the
    DataFrame is missing a `subtype` column.
- `tests/test_apply_cancer_type_evidence.py`:
  - `test_routing_uses_selected_by_not_set_membership` — routing into
    `rare_scope_inference` vs `fine_scope_inference` now keys off the
    winning selector, not naive set membership.
