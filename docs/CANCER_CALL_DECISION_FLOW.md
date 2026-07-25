# Cancer-call decision workflow

This document is the canonical map of how trufflepig goes from a sample
TPM table to a final cancer-type call, decomposition, therapy
shortlist, and report.

## At a glance

1. Load and normalize the expression table, retaining the raw view for QC.
2. Compute reusable whole-sample evidence: tissue composition, signature
   ranking, centroid/compartment context, learned votes, and molecular evidence.
3. Admit plausible candidates and compare them at the appropriate ontology
   level: compartment, family, entity, then subtype.
4. Apply hard molecular blockers and require independent evidence before a
   candidate can replace the current call.
5. Finalize one cancer call and one reference scope before decomposition,
   tumor-attributed expression, therapy ranking, and report rendering.

The central rule is that a broad context may guide a finer comparison, but it
must not masquerade as independent evidence for that fine label. The finalized
call owns downstream interpretation; alternatives remain explicitly contextual
or blocked.

## Why this document exists

It exists because:

1. The flow has 12 stages with non-trivial conditional branching;
   debugging requires knowing which stage produced a value.
2. Several stages have **special rescue paths** (e.g. basal-BRCA
   misclassification rescue, met-site auto-detection, rare-marker
   promotion) that fire only under specific gates — these are not
   visible from the call graph alone.
3. PR-41 consolidated six previously-separate selectors into
   `cancer_type_evidence.select_report_scope_from_evidence`; this doc
   records where that selector sits and what feeds it.

## Current implementation note

The high-level flow below remains canonical. The 2026-07 staged-identity
refactor, described in
[the redesign note](./rnaseq-cancer-call-redesign.md), adds these safeguards:

- candidate rows carry staged identity evidence;
- dominant normal tissue can demote an unsupported same-origin cancer
  candidate;
- same-lineage centroid subtype changes and cross-lineage local-reference
  changes are marker-gated;
- flat percentile and hierarchical log-clean-TPM predictions are retained as
  two views of one learned evidence group; entity changes require a true
  majority of available groups and at least two non-learned groups;
- learned leaf predictions are normalized to the entity being adjudicated,
  while the original child remains explicit in audit fields and may be applied
  later on its subtype/status axis; and
- expected-low marker conflicts and definitive molecular blockers remain
  disqualifying even when other channels agree.

---

## Pipeline overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  INPUT                                                               │
│    Sample TPM table (+ optional fusions, HLA, --cancer-type hint,    │
│    --site-hint, --met-site)                                          │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 0 — Load & QC                                                 │
│    Resolve Ensembl IDs → symbols (Ensembl 114)                       │
│    Technical RNA normalization (drop mtDNA/rRNA-like features,       │
│      renormalize remaining genes to 1e6 TPM)                         │
│    Sample-mode inference: solid / heme / cell-line (cell-line        │
│      detector currently UNDER-CALIBRATED — see Known Gaps)           │
│    Library-prep / preservation inference                             │
│    Gene-detection breadth + RN7SK-dominance check                    │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Tissue composition screen ("Step-0")                      │
│    Two scoring passes against HPA normal-tissue references:          │
│      a. _score_normal_tissues (cohort-wide percentile rank)          │
│      b. _score_host_tissue_details (strict, lineage-specific)        │
│    Outputs:                                                          │
│      tissue_scores (top-10), tissue_score_details (top-10)           │
│      Hint = tumor-consistent | healthy-dominant | mixed              │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Broad lineage triage / family-panel admission             │
│    Current: flat scoring against family panels plus staged identity   │
│    evidence. Proposed future: score against ~7 broad-lineage panels   │
│    first:                                                            │
│      EPITHELIAL_GLANDULAR, EPITHELIAL_SQUAMOUS, MESENCHYMAL,         │
│      NEURAL_GLIAL, MELANOCYTIC, HEMATOPOIETIC, GERM_CELL,            │
│      NEUROENDOCRINE                                                  │
│    Gates Tier-2 child panels by which tier-1 panels scored above     │
│    threshold. Also gated by sample_mode (skip HEME on solid).        │
│    Output: 1-2 surviving tier-1 lineages.                            │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — Within-lineage child cohort scoring                       │
│    For each surviving tier-1 lineage, score the child cohort         │
│    panels (e.g. within EPITHELIAL_GLANDULAR: MAMMARY_LUMINAL,        │
│    GI_ADENO, HEPATOBILIARY, PANCREATIC_DUCTAL, LUNG_ADENO,           │
│    GYNECOLOGIC_GLANDULAR, UROTHELIAL_LUMINAL, etc.)                  │
│    Combine with pan-cancer signature-ranker scores and any           │
│    independent learned / centroid / reference evidence.              │
│    Output: ranked candidate_trace, family_label per row.             │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 4 — Lineage discrimination, staged identity, and rescues       │
│    Fires only when tier-2 winner has multiple TCGA cohorts.          │
│                                                                      │
│    Existing rescues (already wired):                                 │
│      basal-BRCA-vs-squamous (_detect_tnbc_basal_brca_pattern):       │
│        7-gate discriminator promoting BRCA when classifier lands     │
│        on ESCA/LUSC/HNSC/CESC but basal cytokeratin + FOXC1 +        │
│        MIA pattern is present.                                       │
│      normal-origin dominance: same-origin candidates are demoted      │
│        when dominant normal tissue explains their support and         │
│        candidate-distinctive tumor markers do not corroborate them.   │
│                                                                      │
│    PROPOSED LINEAGE_DISCRIMINATION_PANELS (data exists,              │
│    not yet wired):                                                   │
│      Within SQUAMOUS: SCGB2A2/MUCL1 (mammary), AGR2/TFF (ESCA),      │
│        KLK10/MUC21 (HNSC), KRT4/PAX8 (CESC), UPK family (BLCA).      │
│      Within GI_ADENO: CDX2/CDH17 (CRC), MUC5AC/GKN (STAD).           │
│      Within HEPATOBILIARY: AFP/ALB (LIHC), KRT19/MUC1 (CHOL).        │
│      Within NEUROENDOCRINE: ASCL1+INSM1 (SCLC), KRT20 (Merkel),      │
│        PDX1+INS (PANNET), CDX2 (gut NET), CALCA (MTC), TH/DBH (PCPG).│
│      Within MESENCHYMAL: see _FINE_REFERENCE_SPECS (already wired    │
│        for OS-within-SARC; pattern generalizes).                     │
│                                                                      │
│    Output: refined top candidate(s).                                 │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 5 — Met-site inference                                        │
│    User-supplied --site-hint / --met-site take priority.             │
│    Auto-detector (_infer_likely_met_site_from_analysis):             │
│      Fires when:                                                     │
│        (a) sample_mode is solid/auto                                 │
│        (b) no explicit hint supplied                                 │
│        (c) tissue_score_details[0] score >= 0.90                     │
│        (d) top tissue not in primary tissues for the called cancer   │
│        (e) top tissue >= 0.10 ahead of primary                       │
│      Maps liver → met_liver, brain → met_brain, etc.                 │
│      9 met templates: liver, brain, lung, bone, adrenal, skin,       │
│        peritoneal, lymph_node, soft_tissue.                          │
│                                                                      │
│    Output: site_hint, met_site, decomposition template selection.    │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 6 — Decomposition                                             │
│    NNLS fit against template compartments (immune, stroma,           │
│    optional host-tissue compartment from met template).              │
│    Optional compartment gates: adipocyte (BRCA/SARC), Schwann        │
│      (PRAD/PAAD/HNSC/CHOL), erythroid (all solid).                   │
│    Outputs: per-gene tumor fraction, per-compartment TPM share.      │
│    Site-specific met templates are favored when both host-tissue      │
│      score and fitted site-specific component fraction are strong.    │
│    Used by all downstream therapy attribution.                       │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 7 — Purity & subtype signature                                │
│    Purity combination: weighted ensemble of NNLS, marker, lineage,   │
│      signature-based estimates.                                      │
│    Subtype signature: within the chosen primary, score subtype       │
│      panels (BRCA: luminal-A/B/HER2/basal/normal-like; HNSC:         │
│      HPV+/HPV-; LUAD: EGFR/KRAS/STK11 — see _CURATED_HIGH in         │
│      tumor_type_ontology).                                           │
│    Output: purity range (low/estimate/high), subtype label.          │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 8 — Cancer-type evidence consolidation (PR-41 core)           │
│    cancer_type_evidence.select_report_scope_from_evidence            │
│    Runs selector channels with explicit priority and tie-break rules: │
│      1. direct_fusion           (priority 1)                         │
│      2. rare_marker             (priority 2)                         │
│      3. pan_cancer_signature_ranker (candidate/context only)         │
│      4. tumor_label_refinement  (priority 3)                         │
│      5. fine_reference          (priority 3)                         │
│      6. local_expression_reference  (priority 3)                     │
│      7. learned_expression_classifier / learned hierarchy features    │
│      8. fused_evidence          (requires non-ranker admission path) │
│    Each selector produces hypotheses; consolidator picks one         │
│      winner (selected_by) and reports the full evidence chain.       │
│    Curated exact/local references and pan-cancer marker programs      │
│      are evidence channels, not escape hatches: expected-low marker   │
│      conflicts block them unless independent learned/centroid/fusion/ │
│      lineage-panel evidence supports the same lineage.               │
│    Promotes report_scope_cancer_type only when a selector other than │
│      pan_cancer_signature_ranker / primary_expression_match wins.    │
│    When no selector can promote a label, the top ranker row may be    │
│      retained as fallback RNA context; the evidence appendix labels   │
│      this as context, not as an independently selected report label.  │
│    Centroid corroboration can choose among selectable hypotheses,     │
│      but same-lineage subtype swaps require subtype marker support    │
│      unless the centroid margin is decisive.                          │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 9 — Confidence badge                                          │
│    Inputs: candidate_trace spread, signature_stability, subtype      │
│      agreement, supplied-hint presence.                              │
│    Rules:                                                            │
│      - 3-way tie within 15% relative geomean → low                   │
│      - Top-2 spread < 15% → moderate-low / "provisional"             │
│      - Strong signal + lineage concordance > 0.9 → high              │
│      - Special: low / provisional always set for RNA-inferred        │
│        rare-cancer hypotheses (NUTM, MEC, etc.)                      │
│    Output: high | moderate | low | provisional                       │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 10 — Report scope selection                                   │
│    If --cancer-type supplied → use it (report still surfaces RNA     │
│      disagreement transparently).                                    │
│    Else → use cancer_type_evidence.selected.cancer_type only when     │
│      the selected evidence row actually selects a report label.       │
│      Nonselecting pan-cancer ranker rows remain fallback RNA context. │
│    Sets up downstream therapy registry queries, biomarker panels,    │
│    expression-reference cohort selection.                            │
│    local_expression_reference cross-lineage flips are vetoed when     │
│      the bulk call and confident compartment call agree against them. │
│    Blocked/contextual alternatives may appear in evidence text, but   │
│      they do not drive decomposition, biomarker scope, or therapy.    │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 11 — Therapy & biomarker rendering                            │
│    Cancer-type-specific curated registry → ranked therapy            │
│      shortlist.                                                      │
│    Per-gene tumor-source attribution (from decomposition).           │
│    3-state observation classifier for missing genes:                 │
│      - 0.0 (below detection)  — measured but TPM < 0.01              │
│      - *not in input*         — symbol absent from input file        │
│      - *not measured*         — fallback                             │
│    Empty therapy shortlist breakdown: non-tumor-supported vs         │
│      below-detection vs not-in-input counts.                         │
│    Outlier surfacing:                                                │
│      Notable biomarker outliers (≥10× amp OR ≥95th pct, TPM ≥ 50)   │
│      Notable CTAs (TPM ≥ 10)                                         │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 12 — Output                                                   │
│    *-summary.md       brief clinical summary                         │
│    *-analysis.md      detailed analysis with tables                  │
│    *-evidence.md      stepwise attribution chain                     │
│    *-cancer-candidates.tsv  ranked candidate trace                   │
│    *-decomposition-*.tsv    component fits                           │
│    *-tumor-expression-ranges.tsv  per-gene purity-adjusted TPM       │
│    *-analysis-parameters.json  full pipeline config                  │
│    (with --no-figures: only the above; without: + ~40 PNG/PDF)       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Where each input flows

| User input | Stages affected |
|---|---|
| Sample TPM file | All stages (root) |
| `--cancer-type` | Stage 10 (report scope override). Note: does NOT short-circuit Stages 2-9 — broad classification still runs and any disagreement is surfaced in the report. |
| `--site-hint` / `--met-site` | Stage 5 (overrides auto-detector), Stage 6 (selects met template) |
| `--hla-types` | Stage 11 (eligibility on HLA-restricted agents) |
| `--fusions` | Stage 8 selector #1 (direct_fusion takes priority 1 — fusion always wins if matched) |
| `--no-figures` | Stage 12 only (skips ~40 PNG/PDF outputs) |
| `--tumor-context` | Stage 5 (suppresses met auto-detection when `primary`) |

---

## Conditional rescues and overrides (the non-linear bits)

These fire **across stages** based on gates — easy to miss when reading the call graph linearly.

| Rescue / override | Where it fires | What it overrides | Gate |
|---|---|---|---|
| basal-BRCA → squamous misclassification | After Stage 4 candidate ranking | Promotes BRCA when top is ESCA/LUSC/HNSC/CESC | 7-gate: basal keratin program + ESR1/PGR off + FOXC1 ≥ 10 + (MIA or GABRP) ≥ 2 + squamous TFs incomplete + UPK panel off |
| Context rescue (general) | Stage 8 selector | Promotes a non-top broad-RNA candidate when secondary signals support it | Per-selector logic in `cancer_type_evidence` |
| Met-site auto-detection | Stage 5 | Switches decomposition template from solid_primary → met_<site> | tissue_score ≥ 0.90 + ≥ 0.10 ahead of primary + no explicit hint |
| Rare-marker promotion | Stage 8 selector #2 | Promotes a rare cancer (NUTM, salivary, etc.) when marker gene + context match | Marker TPM ≥ threshold + (top context promotes to full XOR top_context_weight applied) |
| Fine reference promotion | Stage 8 selector #4 | Promotes a fine label (OS-within-SARC) | All metrics in `FineReferenceSpec.minimum_metrics` met, support ≥ 0.70 |
| Learned expression classifier | Stage 8 selector | Adds selectable/contextual flat and hierarchical full-profile votes | Flat probability/margin pass, marker sanity is coherent, broad context or hierarchical context supports the label, and compartment/background checks do not contradict it |
| QC/decomposition-guarded lineage-panel out-of-beam rescue | Stage 8 selector | Lets a strong lineage panel admit a code omitted by the broad top-5 | Expression concentration warning or expression/code lineage conflict + lineage-panel score ≥ 0.85 + margin ≥ 0.25 + no independent composition conflict |
| User --cancer-type override | Stage 10 | Forces report scope but does NOT silence broad-classifier disagreement | Always when supplied |

---

## Known gaps & open work

| Gap | Status | Fix path |
|---|---|---|
| Flat panel scoring vs. tiered triage | Open | This document's Tiers 1-2-3 redesign. Implementation = refactor `_score_cancer_family_panels`. |
| BRCA / BLCA / LUAD / OV / UCEC / etc. have no family panel | Partial | `family_extensions.py` adds panels (data) but wiring is currently *too eager* — see family_specificity regression. Pair with tiered scoring. |
| family_specificity regresses 17× when ~26 new panels added flat | Open | Side-effect of un-tiered scoring. Fixed by Tier 1/2 gating. |
| LINEAGE_DISCRIMINATION_PANELS designed but not wired | Open | Wire into Stage 4 after tiered scoring lands. |
| SECONDARY_FAMILY_MEMBERSHIPS designed but not wired | Open | Allows BRCA→SQUAMOUS at 0.5×, BLCA→SQUAMOUS at 0.4× etc. for codes spanning two lineages. |
| Cell-line detection (`culture_level`) | Under-calibrated — returns "normal" on HCC1395 (a famous cell line) | Recalibrate `sample_quality._detect_culture_pattern` against more cell-line training data |
| Ensembl index ~2s startup per process | Open | pirlygenes issue #263 — pickle cache to ~/.cache/pirlygenes |
| Reference matrix 5-8s load per process | Open | pirlygenes issue #264 — Feather siblings + mmap |
| ESCA-adeno vs PAAD vs CHOL discrimination | Open | Tier 3 LINEAGE_DISCRIMINATION_PANELS under HEPATOBILIARY / GI_ADENO |
| OV vs UCEC discrimination | Open | Tier 3 LINEAGE_DISCRIMINATION_PANELS under GYNECOLOGIC_GLANDULAR |

---

## Summary

| Stage | Input | Core logic | Output |
|---|---|---|---|
| 0 | Sample TPM file | Ensembl resolution, tech-RNA norm, mode inference | Cleaned TPM table |
| 1 | Cleaned TPM | Score against HPA normal tissues (2 passes) | `tissue_scores`, `tissue_score_details` |
| 2 | Cleaned TPM | **PROPOSED**: score 7 broad-lineage panels | Top 1-2 tier-1 lineages |
| 3 | Surviving tier-1 lineages | Score child cohort panels + raw RNA signature | `candidate_trace`, `family_label` |
| 4 | candidate_trace | Tier-3 lineage discrimination + special rescues (basal-BRCA) | Refined top candidate |
| 5 | candidate_trace + tissue_scores | Met-site inference (auto or from user) | Decomposition template choice |
| 6 | Cleaned TPM + template | NNLS fit, optional compartment gates | Per-gene tumor fraction |
| 7 | Decomposition output | Purity ensemble + subtype scoring | Purity range, subtype label |
| 8 | All prior | Evidence consolidation across fusion, marker, pan-cancer signature-ranker context, refinement, reference, learned-classifier, centroid, and fused channels | `selected_by`, `report_scope_cancer_type` |
| 9 | candidate_trace + subtype | Tie checks, spread checks | Confidence badge |
| 10 | Stage 8 output + user hint | Choose report scope | Report cancer type |
| 11 | Report scope + ranges_df | Therapy registry + 3-state observation classifier | Therapy shortlist, outliers, CTAs |
| 12 | All prior | Markdown + TSV (+ figures unless `--no-figures`) | Output files |

**The five key concepts a debugger should know:**

1. **Ranker context and report scope are decoupled.** Stages 2-3 do RNA-inferred candidate generation; Stage 10 chooses what label the report wears. A user-supplied `--cancer-type` overrides 10 but does not silence 2-3, so disagreement is always reported.
2. **Selectors are stage-8's atomic units.** Each selector (direct_fusion, rare_marker, lineage_panel, tumor_label_refinement, fine_reference, local_expression_reference, learned_expression_classifier, contrast_discriminator, coarse_composition_reference, fused_evidence, pan_cancer_signature_ranker) is independently scoreable. `pan_cancer_signature_ranker` is retained as non-promoting context/candidate generation; `fused_evidence` can select only when at least one non-ranker admission path supports the call.
3. **Special rescues are conditional, not pipeline stages.** basal-BRCA rescue, met-site auto-detection, rare-marker promotion are gated overrides that fire under specific signatures — they're not always-on.
4. **Decomposition is downstream of classification, not upstream.** Stage 6 fits non-tumor compartments using the template chosen in Stage 5 — which itself depends on what cancer was called in Stages 2-3.
5. **Confidence is independent of correctness.** A high-confidence call can still be wrong (HCC1395 with `--cancer-type BRCA` would show "moderate confidence" even though the pan-cancer signature ranker disagrees); a low-confidence call can still be right (HCC1395 unhinted picks BRCA via rescue with "provisional" badge).
