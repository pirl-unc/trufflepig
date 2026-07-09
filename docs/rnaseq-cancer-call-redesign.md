# RNA-seq cancer-call information flow and redesign plan

This note summarizes how trufflepig currently determines cancer type,
subtype, molecular-expression context, purity, and decomposition from
RNA-seq, then proposes a simpler confluent redesign.

The short version: the current system already has the right ingredients,
but they are applied in several phases with different authority rules. The
broad ranker, centroid compartment gate, evidence selector, local-reference
vetoes, rare-marker promotions, and purity/decomposition rerouting can each
change the interpretation. That makes the pipeline powerful but hard to
reason about, and it creates failure modes where a signal is used at a
decision level where it is not valid.

## Implementation status in this refactor

This PR implements the first production slice of the redesign without waiting
for a full replacement classifier:

- `trufflepig.cancer_call.CancerCallFeatureFrame` centralizes reusable
  per-sample/candidate identity facts: dominant host tissue, tumor-up hits,
  candidate-distinctive tumor-up hits, ontology marker sanity, and
  normal-origin dominance.
- `rank_cancer_type_candidates` now attaches staged identity evidence to each
  candidate row and demotes same-origin candidates whose apparent support is
  better explained by a dominant normal tissue. This closes the COAD+liver
  issue by treating LIHC/liver as background unless LIHC-distinctive tumor
  markers corroborate it.
- Decomposition met-site scoring now lets strong host-site evidence plus a
  large fitted site-specific component overcome generic solid-primary residual
  fit. The applied boost is recorded in `site_evidence`.
- Centroid report-scope corroboration is marker-gated for same-lineage subtype
  swaps. It remains useful for broad/selectable rare entities, but it cannot
  slide SARC_RMS_ERMS to a sibling sarcoma subtype on a small whole-profile
  margin without subtype marker support.
- The pan-cancer signature ranker's own `winning_subtype` is represented as
  `pan_cancer_signature_subtype` evidence. This gives subtype evidence from the
  primary candidate trace a first-class row in the trace, while still requiring
  independent fused support before it can select a report label.
- Cross-lineage local-expression references now require coherent marker
  support against both the active local-reference context and the first-pass
  RNA context. A `mixed` marker sanity result is recorded but is not enough to
  relabel across lineages.
- Basal-BRCA background-label refinement is allowed to use a BRCA candidate's
  `winning_subtype=BRCA_Basal` as the subtype-aware tumor label when the top
  expression label is background-like.
- The main report-scope veto now covers any cross-lineage
  `local_expression_reference` flip against an agreeing bulk classifier and
  confident compartment call, not only epithelial-to-sarcoma attractors.
- A guarded `learned_expression_classifier` selector adds a full-profile
  discriminative vote. It can rescue a context-supported type or subtype, but
  only when probability, margin, marker sanity, broad-ranker context, and
  compartment/background guards agree. It is not allowed to turn a
  background-like whole-profile match into a cross-lineage report label by
  itself.
- The fused evidence layer now carries flat learned predictions plus staged
  learned compartment/family/entity votes as first-class decision features.
  A strong learned entity call must agree with its entity label, hierarchy
  support, flat-lineage support, and margin before it can overrule weak or
  blocked local-reference markers.
- Curated local-expression references and pan-cancer marker programs retain
  positive marker evidence in the trace, but expected-low conflicts make them
  contextual unless independent learned, centroid, fusion, lineage-panel, or
  marker-coherent evidence supports the same lineage. This is the rule that
  keeps MBL-like threshold hits from overriding sarcoma evidence.
- Strong lineage-marker panels can rescue a candidate omitted by the broad
  ranker only when the panel score/margin are very high and the current broad
  call is flagged as technically fragile by expression concentration or by an
  expression/code lineage conflict from decomposition. This keeps the rescue
  feature-level and inspectable rather than BRCA-specific.
- Background-like mesenchymal or gastric hypotheses are now retained as
  blocked/contextual evidence when an epithelial compartment plus close
  colorectal candidate better explains the sample. The selected report label
  owns decomposition, expression ranges, biomarker scope, and therapy
  attribution; blocked alternatives may appear only as uncertainty/context in
  the evidence text.
- Report wording now distinguishes the selected report label from RNA-family
  context. For example, a colorectal call with sarcoma-like contextual signal
  should read as a READ/COAD report with sarcoma retained as non-driving
  evidence, not as a sarcoma therapeutic or decomposition scope.
- Nonselecting ranker fallbacks are rendered as "Fallback RNA context" in the
  evidence appendix, not as an independently selected report label.

## Validation status

Validation through 2026-07-05 shows a large improvement, including removal of
the severe cross-lineage failures that motivated this refactor:

- Focused selector/report regression subset:
  `109 passed` (`./test.sh tests/test_cancer_type_evidence.py -q`).
- 565-sample baseline before the learned selector:
  entity-compatible 450/565 (80%), lineage-compatible 534/565 (95%).
- 565-sample result after fused learned hierarchy/reference gating:
  exact 495, subtype-compatible 48, sibling-compatible 21, lineage-only 1,
  miss 0; entity-compatible 564/565 (100% rounded), lineage-compatible
  565/565 (100%).
- The only lineage-only residual is the close renal pair `KIRC -> KICH`.
- #101 synthetic COAD+liver now passes without an xfail; liver is treated as
  structured host/background instead of letting LIHC win the type ranker.
- #102 is fixed on the available RMS set in the 565 harness:
  SARC_RMS_ERMS 5/5 exact, SARC_RMS_ARMS 5/5 exact, and
  SARC_RMS_SSRMS 5/5 exact.
- Local report replay completed for 19 available local samples with 0 failures
  and 2 skipped missing-input samples. Report scoring on the manifest-truth
  subset is 8/8 compatible; generated-report audit found 0 issues. The reviewed
  reports consistently use the selected label for biomarker/therapy scope and
  decomposition; contextual alternatives remain evidence/context only.
- Blind no-hint validation is entity- and lineage-compatible on 14/14 local
  truth samples and 14/14 medoids. HCC1395/StringTie still has a T_ALL
  pan-cancer ranker artifact, but the strong BRCA_BASAL panel plus
  expression/code lineage conflict now selects BRCA.

Learned-expression classifier leakage audit (2026-07-02):

- Added `scripts/eval_learned_classifier_holdout.py` to train the learned
  classifier on up to 3 representative samples per cohort and test on up to 2
  held-out samples per cohort. The scaler, high-variance gene selection, and
  logistic model are fit on the training split only.
- Five shuffled 3/2 splits use 342 training samples and 223 held-out samples.
  Held-out sample means: exact 66.9%, entity-compatible 88.2% (range
  86.5-91.9%), lineage-compatible 96.9% (range 96.0-98.7%), top-3
  entity-compatible 96.6%.
- Held-out medoid means: exact 69.1%, entity-compatible 90.6%,
  lineage-compatible 97.3%, top-3 entity-compatible 99.0%. All-sample medoids
  remain near-perfect, but that view includes training samples and is not the
  honest estimate.
- On the current 19 cross-lineage miss targets, held-out observations across
  seeds were 12 exact, 2 subtype-compatible, 1 sibling-compatible, and 12 miss.
  The full-trained p~0.99 truth votes are therefore partly training leakage,
  but the holdout signal is still strong enough to justify using the learned
  classifier as a guarded co-signal rather than an oracle.

Local report audit highlights from blind report regeneration:

- PFO002 reports select READ/colorectal scope where a concrete child is
  supported; weak non-CRC GI fused/composition calls are blocked when they
  compete with a close CRC-family ranker row carrying explicit CRC
  family-marker support and lack a coherent non-CRC marker program.
  decomposition, biomarker panel, and therapy target landscape use READ/CRC
  context. The evidence appendix labels the pan-cancer ranker row as fallback
  RNA context rather than an independent selector. Capture-like/OOD inputs may
  degrade to unresolved CRC-favored, but should not select STAD without stronger
  independent corroboration.
- PFO017 bladder/liver reports select BLCA; the liver sample uses liver host
  context in decomposition without changing the report label or therapy scope.
- Local sarcoma reports select SARC or SARC_OS as appropriate; biomarker and
  therapy sections remain sarcoma-scoped. Alvin reports SARC via fused evidence
  with EGFR KDD carried as orthogonal eligibility context.
- NUTM fusion-supported reports select NUTM; strong NUTM1 RNA-surrogate reports
  can select NUTM when squamous/epithelial context is compatible and carry
  NUTM-targeted therapy caveats that require orthogonal confirmation.
- HCC1395 Kallisto/StringTie both report BRCA after lineage-panel rescue,
  although the StringTie input still carries a T_ALL bulk-ranker artifact in
  the alternatives/decomposition context.

## Current decision process

### 1. Input, cleanup, and sample context

The CLI loads a gene-level TPM table, resolves gene identifiers, applies
technical RNA cleanup, and records sample-level context such as degradation,
culture-like expression, site hints, fusions, alterations, and HLA inputs.

The pipeline order is declared as:

```text
load_expression -> sample_context -> analyze -> decompose -> ranges ->
confidence -> report rendering
```

`sample_context` modifies confidence intervals and downstream caveats, but
the first cancer-type decision is made in `analyze`.

### 2. Step-0 tissue composition

`healthy_vs_tumor.assess_tissue_composition` is an early, coarse screen. It
compares the sample against HPA normal tissues and cancer references, reports
top normal tissues and top cancer cohorts, and adds tumor-evidence channels
such as proliferation, CTA re-expression, oncofetal markers, and
tumor-up-vs-matched-normal hits.

This stage is intentionally not a final healthy-vs-cancer classifier. It
emits context such as `tumor-consistent`, `possibly-tumor`, or
`healthy-dominant`, plus tissue/cancer-reference matches that later selectors
may use.

### 3. Pan-cancer signature-ranker scoring

`plot_embedding._compute_cancer_type_signature_stats` scores each candidate
against curated cancer-type signature panels. For each signature gene, it
computes:

```text
score_gene = cohort_percentile_HK_reference * within_sample_percentile
```

The cohort-percentile leg ranks the sample's HK-normalized expression against
the full approximately 170-column reference set: broad cancer cohorts plus
HPA normal tissues. The within-sample percentile leg makes the signal more
robust to dilution. The cancer-type score is the mean over the panel.

This is now explicitly the `pan_cancer_signature_ranker`, not an authoritative
`primary_expression_match`. Its HK cohort-percentile basis is legacy and useful
as candidate-generation/corroborating evidence, but it cannot select a report
label by itself. The A/B harness
`scripts/eval_pan_cancer_signature_ranker_ab.py` compares the HK basis against
raw clean-TPM/nTPM cohort percentiles, within-sample percentiles, the learned
classifier, centroid Spearman, and the fused report selector.

Molecular subtype cohorts are mostly kept out of the broad candidate space so
they do not leak into primary cancer-type routing. Separate subtype scoring
exists for selected broad types such as BRCA, HNSC, and LUAD.

### 4. Whole-profile centroid and compartment call

`cancer_type_centroid.centroid_correlations` computes whole-profile Spearman
correlations between the sample and subtype-aware cohort centroids.

`compartment_call` aggregates those correlations into histogenesis
compartments such as Epithelial, Sarcoma, Hematolymphoid, Melanoma,
Neuroendocrine, CNS, Germ cell, and Embryonal. Each compartment is scored as
the mean of its top 3 cohort correlations. A compartment is actionable only
when the top-vs-runner-up margin is at least 0.025.

This signal is deliberately not part of the current support-score geomean. It
is used later as:

- a confidence-gated leaf restriction across compartments,
- a hallmark-veto guard for impossible cross-compartment candidates,
- a fine-subtype resolver,
- and a final evidence-selector corroborator.

### 5. Candidate admission and broad ranker

`tumor_purity.rank_cancer_type_candidates` starts with the top signature
candidates, adds candidates suggested by family panels, guarantees admission
for some subtype-aware parent cohorts, and has a basal-BRCA admission rescue
when a squamous-looking top candidate coexists with a basal mammary pattern.

For each candidate it calls:

```text
estimate_tumor_purity(..., cancer_type=code, include_decomposition=False)
```

That produces candidate-specific signature purity, ESTIMATE stromal/immune
purity, lineage support, and signature stability. During ranking, lineage
decomposition is skipped for speed.

The support score is the geometric mean of:

```text
signature_score
purity_estimate
lineage_support_factor
signature_stability
family_factor
```

The purity term is explicitly documented as a known imperfect but
load-bearing factor: removing it alone previously regressed local truth
sets.

After the raw geomean rank, several mutation steps can change the order:

- orphan-dominance promotion for ungrouped but strong direct signals,
- coarse TCGA orphan rescue,
- low-purity PRAD stromal-context rescue,
- basal-BRCA over squamous rescue,
- normal-tissue tiebreaker for close candidates,
- tumor-intrinsic lineage exclusion,
- same-family promotion,
- confident centroid compartment restriction,
- hallmark veto,
- centroid-driven fine-subtype resolution,
- final normalization to `support_fraction_of_top`.

### 6. Purity and decomposition

`estimate_tumor_purity` combines:

- candidate-specific tumor signature purity,
- ESTIMATE stromal/immune purity, gated off for heme/sarcoma-like lineages,
- lineage gene purity and support,
- optional mixture-cohort subtype lineage panels,
- and, only outside hot ranking loops, lineage-routed decomposition.

`expression_decomposition.decompose_expression` routes the sample to one of
four modes: solid, mesenchymal, heme, or embryonal. Routing comes from an
explicit cancer hint when present, otherwise from `compartment_call`; if the
compartment is ambiguous, candidate modes are compared by `lineage_fit`.

Decomposition fits background templates by NNLS, subtracts immune/stroma/normal
host components, and reports:

- `residual_fraction` as the primary monotone purity signal,
- bulk aneuploidy amplitude as a purity-scaled corroborator,
- residual aneuploidy/proliferation as tumor characterization.

The residual fraction is reported, but it is not currently used as a general
pre-label cancer-type ranking term.

**Honest purity fusion + anti-saturation guard** (`purity_integration.best_purity_estimate`,
wired as the final purity pass in `analyze_sample` before the ReportView freeze).
After every adoption/override has settled `analysis["purity"]`, the per-method
estimates — signature, lineage, ESTIMATE, and the decomposition **residual
fraction** — are fused with a random-effects (DerSimonian–Laird) model in logit
space. Two evidence-backed effects, grounded in an HCC1395×HPA in-silico mixture
benchmark (signature can be flat/broken on atypical inputs; ESTIMATE is monotone
but biased high and saturating; lineage is tissue identity, not purity):

- *Honest interval.* The pooled interval widens automatically when methods
  disagree (τ² between-method heterogeneity), and `method_agreement` = 1 − I²
  drives the confidence tier. Conflicting methods can no longer report a narrow,
  overconfident CI.
- *Anti-saturation guard.* A read pinned at the 1.0 ceiling is never a real
  measurement (a biopsy is ~never 100% tumor); it is replaced by the physical
  fused consensus — chiefly the residual fraction. Below the ceiling but still
  high, the point is replaced only when no reliable method corroborates it. The
  combiner point is otherwise preserved (it is the most accurate single point in
  the benchmark). This is what pulls a rare type like NUT carcinoma — whose
  lineage-identity genes (`KRT5`/`KRT6A`/`NUTM1`) and ESTIMATE both saturate to
  100% while the residual fraction says ~55% — off a spurious 100% (validated:
  nutm1 samples 100%→53–57%, low confidence; genuinely-corroborated high-purity
  samples such as an 84%-pure cell line and an 88% bladder tumor are preserved).

### 7. Evidence selector and final report scope

After `analyze_sample` makes a working RNA call, `main._apply_cancer_type_evidence`
calls `cancer_type_evidence.select_report_scope_from_evidence`.

That selector accumulates hypotheses from multiple channels:

- pan-cancer signature-ranker candidates and subtype suggestions,
- coarse composition reference,
- tumor-label refinement when the top broad label looks background-like,
- fine reference panels,
- local expression references,
- lineage panels,
- contrast discriminators,
- learned full-profile expression classifier,
- learned mismatch-repair RNA state context,
- rare RNA marker rules,
- direct fusions.

The fused evidence layer records component scores for the signature ranker,
learned classifiers, centroid support, local/fine references, lineage panels,
composition references, contrast markers, rare markers, and fusions. A
signature-ranker component can improve a fused score, but it is not an
admission path: fused selection requires at least one non-ranker route such as
strong learned/hierarchical support, a centroid-backed or marker-specific exact
reference, a lineage panel, rare marker, contrast discriminator, composition
reference, or tumor-label refinement.

Each hypothesis can still call `consider_for_report_label`, which sets a
priority class, strength, and selector tie-break. `_pick_selected` first
respects definitive fusions, then chooses the highest-priority selectable
hypothesis. If the compartment call is confident, whole-profile centroid
correlation can replace that authority winner with another selectable
hypothesis that leads by at least 0.015 Spearman rho. A ranker-only top
candidate is retained as non-promoting context (`pan_cancer_signature_ranker`)
so downstream text and evidence traces keep continuity with the working
`analyze_sample` call.

If this final report-scope label differs from the working `analyze_sample`
label, `main._reroute_decomposition_to_call` recomputes purity/decomposition
so the final report is internally consistent. A separate lineage veto can
undo any cross-lineage exact-reference flip when the bulk classifier and
confident compartment call agree against it. Downstream modules then use the
selected report label only; blocked or contextual alternatives can be rendered
as evidence but cannot drive decomposition or therapy selection.

## Current failure modes

### Non-confluent decisions

The same biological signals are read in multiple places with different
authority:

- centroid restricts the broad ranker and later corroborates the evidence
  selector;
- purity ranks candidates before decomposition is available, then is
  recomputed after evidence refinement;
- local expression references can select labels, then a separate main-level
  veto can undo a subset of their choices;
- subtype information can be a candidate label, a row annotation, or a
  downstream degenerate-subtype resolution.

The system is therefore not a single decision DAG. It is a sequence of
partially overlapping selectors with repair steps.

### Issue #101: structured contaminant beats tumor

In the 70% normal liver / 30% COAD synthetic mix, COAD has the stronger cancer
signature than LIHC, but LIHC wins because the per-candidate purity term
rewards the dominant normal liver compartment. Whole-profile centroid and
tissue composition also prefer liver/LIHC because most RNA really is liver.

The correct signal is component-aware: subtract the structured liver
background, then score the residual tumor identity. The current pipeline does
that too late, only after the label has already been chosen.

The existing normal-tissue tiebreaker only boosts a candidate whose primary
normal tissue matches the sample. It does not demote a candidate whose apparent
tumor evidence is explainable as its normal tissue of origin.

This refactor adds that missing demotion as staged identity evidence and also
lets decomposition rank the evidenced liver host template over a generic
solid-primary template when hepatocytes dominate the fitted non-tumor fraction.

### Issue #102: centroid corroboration flips RMS subtype

After centroid corroboration, SARC_RMS_ERMS samples are selected as sibling
sarcoma subtypes. Lineage remains correct, but exact subtype accuracy regresses.

The likely cause is that `_pick_selected` allows whole-profile centroid
corroboration to decide among selectable same-lineage hypotheses using a
single global 0.015 rho margin. Dense sarcoma subtypes are too close in
whole-profile space for that rule to be a reliable exact-subtype authority.

The centroid signal is valuable for compartment and broad rare-type
corroboration, but it should not freely override a same-lineage subtype call
unless subtype-specific markers or a family-specific larger margin agree.

This refactor implements that rule in `_pick_selected`: same-lineage subtype
swaps require matching ontology marker sanity unless the centroid margin is
decisive for that dense lineage.

## Redesign goal

Replace the multi-pass selector stack with one staged analytical algorithm:

```text
feature extraction -> staged evidence frame -> monotone hierarchical decision -> report rendering
```

The key invariant: each signal may only act at the biological resolution where
it is validated.

- Whole-profile centroid: strong for compartment and some distinctive rare
  broad entities; weak for dense sibling subtypes.
- Normal-tissue profile: strong for contaminant/background modeling; unsafe as
  positive cancer identity by itself.
- Purity: describes how much tumor-like signal exists; it should not choose
  which cancer type when the sample is dominated by structured normal tissue.
- Decomposition residual: strong for mixture/background separation; noisy for
  whole-profile residual nearest-neighbor matching, so it should be scored on
  curated identity panels rather than cosine-like whole-profile similarity.
- Fusion and direct molecular evidence: authoritative for entities where the
  alteration is defining.

## Required design clarifications

### Lineage vetoes still exist

The redesign should keep lineage vetoes from implausible marker patterns. What
changes is where they live.

Current code already has three useful forms:

- tumor-intrinsic lineage exclusion, e.g. epithelial program present demotes
  mesenchymal/heme candidates, epithelial program confidently absent demotes
  epithelial candidates, and specific NE/melanocytic programs demote competing
  lineages;
- hallmark absence, e.g. a candidate whose defining high markers are near absent
  is a poor fit even if centroid/background similarity is high;
- expected-low marker violations, e.g. subtype/lineage panels where a marker
  that should be OFF is strongly expressed.

Those should become explicit stage-level evidence terms:

```text
required_positive_program_present
required_positive_program_absent
forbidden_or_expected_low_program_present
lineage_specific_program_conflict
```

The rule should be asymmetric and evidence-aware. Absence of a marker panel is
strong only when the panel should be detectable at the current tumor fraction
and expression/reference quality; over-expression of an expected-low marker is
strong only when that marker is specific for a competing lineage or subtype.
Low-purity samples should therefore get softened penalties or abstentions, not
automatic hard vetoes.

### Decomposition participates during exploration

Yes: the redesigned caller should evaluate multiple decompositions while
exploring lineages and structured backgrounds. The current pipeline computes
candidate purity cheaply during ranking and then computes decomposition for the
winner; #101 shows that is too late for structured tissue contaminants.

The feature frame should include, at minimum:

- all four lineage modes: solid, mesenchymal, heme, embryonal;
- likely host/background decompositions for dominant normal tissues such as
  liver, brain, lung, lymph node, smooth muscle, adipose, and bone when those
  dominate the bulk profile;
- residual identity-panel scores for plausible candidate families/entities;
- `lineage_fit`, residual fraction, subtraction fractions, and consistency
  flags for each attempted mode/background.

Decomposition should guide decisions at these levels:

- compartment and family: strong guidance, because mode fit and residual
  lineage signal answer "what tumor program survived background subtraction?";
- structured contaminant handling: strong guidance, because subtracting the
  host tissue distinguishes tumor from dominant normal background;
- exact subtype: cautious guidance only, because whole-profile residual nearest
  matching was a documented negative result. Residual evidence should be scored
  on curated subtype/entity panels, not cosine-like residual similarity.

The search does not need a full separate decomposition for every molecular
subtype when the subtype shares the parent's background model. It does need
candidate residual scores under the relevant parent/family decomposition.

### Subtypes need a coherent ontology

The redesigned caller needs a single subtype ontology, not scattered local
rules. Each registry label should have:

- `parent_code` and ancestors,
- whether it is an entity, histologic subtype, molecular subtype, expression
  state, therapy/status axis, or defining-alteration entity,
- compatible compartments/secondary lineages,
- direct expression-reference availability,
- fallback expression reference and fallback reason,
- expected-high marker program,
- expected-low/conflicting marker program,
- whether the marker panel is strong enough for selection, only descriptive, or
  not evaluable.

This ontology should be the only source used by ranking, evidence rendering,
therapy context, expression-reference selection, and subtype display. Existing
pieces already point in this direction (`cancer_type_context`,
`tumor_type_ontology`, registry parent codes, expression-reference discovery),
but the redesign should make the combined object authoritative.

### Subtype expression references are not all equivalent

Use subtype expression references when they are available, but with role and
quality metadata:

```text
direct subtype reference
direct parent reference
curated sibling/family fallback
status/molecular expression cohort
no expression reference
```

A direct subtype reference can support subtype identity if its marker program is
coherent in the sample. A status/molecular expression cohort can support a
state, but should not by itself relabel the parent entity without orthogonal
support. A fallback parent reference is suitable for purity/decomposition and
broad expression ranges, but it is not proof of the child subtype.

Subtypes should therefore carry separate decisions:

```text
subtype_reference_available
subtype_marker_panel_strength
sample_supports_subtype_markers
sample_violates_expected_low_markers
fallback_parent_used_for_expression
```

This prevents the two common mistakes:

- treating every subtype with a reference as if it has a decisive marker panel;
- treating a subtype without a direct reference as unevaluable when it may have
  strong defining positive/negative markers.

### Learned models should be hierarchical evidence, not a separate oracle

The flat learned expression classifier is useful, but it asks too many
biological questions in one softmax: compartment, family, exact entity,
histologic subtype, molecular state, and fusion-defined rare entity. The
redesign should split learned evidence into stage-specific co-signals:

- **compartment learned vote**: epithelial, mesenchymal/sarcoma,
  hematolymphoid, melanocytic, CNS, embryonal, neuroendocrine, germ-cell;
- **family learned vote within compartment**: e.g. GI/hepatobiliary/breast/
  squamous/renal/gynecologic/lung/urothelial for epithelial, and
  liposarcoma/RMS/GIST/vascular/nerve-sheath/undifferentiated/
  melanocytic-translocation-like groups for sarcoma;
- **entity learned vote within family**: exact labels only after the relevant
  family has been admitted;
- **subtype or molecular-axis learned vote**: BRCA basal/luminal/HER2,
  LUAD EGFR/KRAS/STK11, SCLC ASCL1/NEUROD1/POU2F3/YAP1, MBL groups, and
  similar axes after the parent is accepted.

Each learned vote should have an out-of-fold calibration record and a declared
role:

```text
learned_vote.stage
learned_vote.label_space
learned_vote.top_predictions
learned_vote.probability_or_score
learned_vote.margin
learned_vote.oof_precision_at_threshold
learned_vote.oof_top3_recovery
learned_vote.training_split_policy
learned_vote.admits_candidate
learned_vote.selects_candidate
learned_vote.blocking_or_caution_reasons
```

The learned signal should generally admit candidates before it selects them.
Selection requires agreement with the appropriate stage evidence: compartment,
marker sanity, fusion status, expression-reference support, decomposition
residual, or direct molecular evidence. A high flat probability without such
corroboration is context, not authority.

All learned evidence must be serialized into the same evidence trace as the
non-learned signals. The detailed markdown report should show these votes
beside centroid, marker, decomposition, local-reference, fusion, tumor-up, and
normal-background evidence so a reviewer can reason over the whole decision
rather than seeing only the winning selector.

The report now also emits machine-readable signal artifacts:

- `*-cancer-type-signal-matrix.tsv`: one row per `sample × signal/source ×
  predicted layer`, including candidate/ranker rows, staged evidence-graph
  channels, learned hierarchy votes, retained context-only rows, and post-label
  decomposition/met-site background context.
- `*-cancer-type-signal-summary.md`: a compact reader-facing summary of the
  same rows.
- `*-cancer-type-signal-matrix.png`: the decision-aligned plot version of the
  same evidence frame. This is the main cancer-call plot; raw MDS/neighborhood
  reference maps are audit/context-only because they bypass the fused evidence
  graph.

Batch runners concatenate the per-sample TSVs into the same schema for local
report sweeps and, when requested, the 565 representative-sample harness. The
selector is therefore no longer the only visible provenance field; every row
records `role`, `status`, `support`, `selects_report_label`, and agreement with
the final entity/lineage.

### Current PR implementation status

This PR implements the first usable slice of the learned-evidence redesign
without replacing the primary ranker.

- `expression_classifier.classify_expression` remains the flat all-entity
  full-profile logistic-regression vote.
- `expression_classifier.classify_expression_hierarchy` now emits explicit
  stage-scoped votes:
  - `learned_compartment`: a separate compartment model,
  - `learned_family`: a separate coarse-family model,
  - `learned_entity`: a separate parent/entity model,
  - `learned_subtype_axis`: a sparse parent-scoped rollup from the flat model,
  - `learned_mismatch_repair_binary`: an orthogonal RNA-state vote; production
    report use now comes from the release ensemble below when the accepted
    cancer context is one where MMR/MSI is clinically meaningful.
- `expression_classifier.classify_mismatch_repair_sibling_expression` is the
  parent-aware companion to the pooled binary model. Given a working entity
  such as `READ` or `COAD`, it reads the flat learned classifier probabilities,
  keeps only paired `_MSI`/`_MSS` siblings for that entity, and falls back to the
  cancer family only if the entity lacks both states. The resulting
  `learned_mismatch_repair_sibling_entity` or
  `learned_mismatch_repair_sibling_family` vote is reported as orthogonal
  mismatch-repair context.
- `cancer_type_evidence` preserves those votes in every learned hypothesis as
  `learned_expression_hierarchy_votes` and renders them as
  `learned_expression_classifier` channels in the staged evidence graph.
- The learned selector still prefers broad-ranker context. A context-free
  learned rescue is only selectable when the flat probability is at least
  `0.97`, the flat margin is at least `0.50`, either the learned entity or
  learned compartment support is at least `0.70`, and marker sanity does not
  block the candidate.
- Cross-lineage learned calls still need strong probability and either
  broad-ranker support or the hierarchical rescue condition. Otherwise they are
  retained as context/blocked evidence, not used as the report label.
- Strong lineage-marker panels can now rescue a candidate that the broad ranker
  omitted only when the broad expression distribution carries a technical
  concentration warning or decomposition reports an expression/code lineage
  conflict, the panel score is at least `0.85`, the panel margin is at least
  `0.25`, and no independent composition conflict blocks it. This preserves the
  old top-5 guard for normal inputs while fixing the local HCC1395/StringTie
  case where a BRCA_BASAL positive/negative marker program was previously noted
  but unable to correct a T_ALL artifact call.
- The detailed evidence markdown now includes a `Cancer-type decision evidence
  trace` table before the target evidence. It shows selected, blocked, and
  context-only channels with staged learned votes beside centroid, marker,
  local-reference, fusion, composition, and tumor-up signals.
- Hierarchical learned-vote rows carry the training policy plus held-out
  calibration fields where available (`holdout_top1_accuracy`,
  `holdout_medoid_top1_accuracy`, and `oof_top3_recovery`). These values are
  rendered for interpretation only; the selector thresholds use the live
  probability/margin/context fields.
- The MMR RNA-state votes are intentionally not primary report-label selectors
  and do not establish immunotherapy eligibility by themselves. The live report
  path is context-gated through `mismatch_repair_context_group`: CRC-family,
  UCEC, and STAD contexts can receive an MMR/MSI expression vote; unrelated
  contexts such as GBM or PRAD do not emit an MSI/MSS statement. The release
  model is saved as explicit CSV coefficients in
  `trufflepig/data/mismatch-repair-expression-ensemble.csv` with companion
  metadata in `trufflepig/data/mismatch-repair-expression-ensemble-metadata.csv`;
  no pickle/runtime training artifact is required for release use.
- `scripts/eval_mismatch_repair_classifier.py` starts the broader-sample
  expansion path. It can build a labeled experimental matrix from cached
  Treehouse TCGA per-sample TPMs or full cBioPortal TCGA RSEM matrices for
  CRC, UCEC, and STAD. The current cBioPortal RSEM run yields `1,386` labeled
  samples: CRC `78` MSI / `504` MSS, STAD `73` MSI / `273` MSS, and UCEC
  `148` MSI / `310` pMMR-like MSS. `UCEC_CNL` and `UCEC_CNH` are treated as
  MSS-like pMMR negatives; `UCEC_POLE`, `STAD_POLE`, and `STAD_EBV` are
  excluded/confounder rows, not binary labels.
- The evaluator now builds features inside each train/test fold so supervised
  gene selection and tissue-residual medians do not see held-out labels or
  held-out expression. A 500-gene variance baseline in
  `rank_plus_tissue_residual` space is `0.915` 5-fold pooled CV, `0.933`
  within-tissue CV, and `0.742` leave-tissue-out. Narrowing to genes with
  within-training-tissue MSI/MSS effect and low tissue eta-squared is a clear
  improvement: 500 genes gives `0.956` pooled, `0.960` within-tissue, and
  `0.885` leave-tissue-out.
- The best compact tissue-independent core so far is a 250-gene panel with
  tissue one-hot calibration: `0.952` pooled, `0.951` within-tissue, and
  `0.898` leave-tissue-out, with held-out UCEC at `0.819`. Tissue-by-expression
  interactions improve known-tissue pooled CV to `0.961` and overall
  leave-tissue-out to `0.908`, but weaken UCEC transfer relative to the
  one-hot-only core. Production use should therefore be a compact pooled
  tissue-independent MMRd-vs-pMMR core plus explicit tissue/entity calibration
  heads, with interaction heads reported separately from the transferable core.
- A more interpretable selector, `consistent_cross_tissue_mmr`, now scores each
  gene by same-direction MSI/MSS contrast, contrast stability across tissues,
  and stability of the MSI/MSS midpoint across tissues. At 250 genes with
  tissue one-hot it is close but not better overall (`0.948` pooled, `0.951`
  within-tissue, `0.895` leave-tissue-out; held-out UCEC `0.823`). At 500 genes
  with tissue one-hot it matches pooled/within-tissue performance and improves
  transfer (`0.956` pooled, `0.960` within-tissue, `0.905` leave-tissue-out;
  held-out UCEC `0.812`). As before, tissue-by-expression interactions raise
  pooled CV (`0.963`) but weaken UCEC transfer (`0.731`), so interactions remain
  a known-tissue calibration head rather than the transferable core.
- A fixed-selector feature-space check used `consistent_cross_tissue_mmr`, 500
  genes, tissue one-hot, and cBioPortal RSEM. `sample_gene_z` gives the best
  transfer (`0.912` leave-tissue-out; held-out UCEC `0.847`) despite slightly
  lower pooled CV (`0.955`). `tissue_residual` gives the best pooled CV
  (`0.961`) but weaker transfer (`0.904`; UCEC `0.814`). `raw_log`,
  `within_sample_rank`, and `rank_plus_tissue_residual` are similar but do not
  beat sample-level z-score on leave-tissue-out. This argues for treating
  sample-level z-score as the leading transferable MMR feature space and
  tissue-residual/log spaces as known-tissue corroboration candidates.
- A gene-count/procedure grid with `sample_gene_z` and tissue one-hot compared
  `50..500` genes in steps of `50` across `variance`, `mmr_effect`,
  `residual_mmr_effect`, `tissue_independent_mmr`, and
  `consistent_cross_tissue_mmr`. Variance selection is clearly unsuitable
  (`0.716` leave-tissue-out at 500 genes). Best overall leave-tissue-out is
  `mmr_effect` at 150 genes (`0.916`) and `tissue_independent_mmr` at 250 genes
  (`0.916`). They trade sensitivity and specificity: `mmr_effect` catches more
  MSI samples but makes more MSS false positives, while `tissue_independent_mmr`
  is more conservative. `consistent_cross_tissue_mmr` at 500 genes remains a
  strong interpretable specificity-oriented candidate (`0.912` leave-tissue-out;
  UCEC `83/148` MSI and `305/310` MSS correct).
- The selected release model is a mean-probability ensemble of the three
  practical candidates: `mmr_effect@150`, `tissue_independent_mmr@250`, and
  `consistent_cross_tissue_mmr@500`, all in `sample_gene_z` space with tissue
  one-hot calibration. At decision threshold `0.50`, the ensemble scores
  `0.963` pooled five-fold CV, `0.952` within-tissue CV, and `0.934`
  leave-tissue-out. Leave-tissue-out by context is CRC `0.957` (`70/78` MSI and
  `487/504` MSS correct), STAD `0.957` (`59/73` MSI and `272/273` MSS correct),
  and UCEC `0.889` (`109/148` MSI and `298/310` MSS correct). A `0.40`
  threshold is available in the evaluation output as a more sensitive screening
  point, but the release artifact uses `0.50` to limit false-positive MSS calls.
- Reports surface the release MMR vote in the integrated evidence bullets, the
  staged evidence graph, the signal matrix TSV/markdown, and the
  `cancer-type-signal-matrix.png` plot. The text explicitly says this is RNA
  context only and requires MSI-PCR, MMR IHC, or validated clinical sequencing
  before immunotherapy eligibility is inferred.

The 3-training/2-testing holdout harness now evaluates both flat and staged
learned models. Across five random seeds, the staged compartment model is the
most reliable new signal; the staged entity model is not better than the flat
entity classifier as a stand-alone oracle. That supports using the hierarchy
for admission and corroboration rather than replacing the final selector:

- flat held-out entity-compatible mean: `0.882`;
- flat held-out lineage-compatible mean: `0.969`;
- staged-entity held-out entity-compatible mean: `0.882`;
- staged-entity held-out lineage-compatible mean: `0.966`;
- staged compartment top-1 on seed 0: `219/223` held-out samples and
  `116/117` held-out medoids.

End-to-end 565-sample behavior for the current PR:

- original broad/evidence baseline: `450/565` entity-compatible and `534/565`
  lineage-compatible;
- previous flat learned-channel selector: `518/565` entity-compatible and
  `546/565` lineage-compatible;
- current fused learned-hierarchy selector: exact `500`, subtype-compatible
  `43`, sibling-compatible `21`, lineage-only `1`, miss `0`; `564/565`
  entity-compatible and `565/565` lineage-compatible.

Notable resolved misses include the prior NBL-to-SCLC slips, SARC_ASPS to
SKCM, SARC_ANGIO to HNSC, SARC_CCS to UVM, SARC_DSRCT to ESCA, and the
SARC_RMS_ERMS subtype regression from #102. The only remaining
non-entity-compatible row is the lineage-only close renal pair `KIRC -> KICH`.

## Proposed confluent algorithm

### Stage 0: compute a single reusable feature frame

Create one `CancerCallFeatureFrame` per sample. It should cache all expensive
sample-level signals exactly once:

- broad signature scores and gene-level details,
- subtype signature scores,
- whole-profile centroid correlations,
- compartment scores and confidence,
- HPA normal tissue scores and host-tissue details,
- cancer-reference composition/tumor-up evidence,
- family/lineage panel evidence,
- rare-marker and fusion evidence,
- per-mode decomposition results for solid, mesenchymal, heme, and embryonal,
- top-host residuals for structured tissue backgrounds,
- subtype ontology entries and expression-reference options for every admitted
  candidate,
- positive/negative marker sanity checks for every admitted entity/subtype,
- hierarchical learned-expression votes at compartment, family, entity, and
  subtype/molecular-axis resolution, including out-of-fold calibration metadata,
- bulk aneuploidy and residual characterization.

This frame is read-only. Later stages consume it and write a decision trace;
they do not recompute or reinterpret source signals ad hoc.

### Stage 1: tumor-present and contaminant/background model

Separate these questions before cancer typing:

1. Is there tumor evidence beyond normal tissue?
2. What normal or stromal host components dominate the bulk RNA?
3. For each candidate, is its apparent evidence explainable by its matched
   normal tissue?

Add a general normal-tissue-of-origin dominance score:

```text
normal_origin_dominance(candidate) =
  host_match(candidate.primary_normal_tissue)
  - tumor_specific_evidence(candidate)
```

For a candidate such as LIHC in a liver-dominant sample, high liver-normal
match without sufficient LIHC tumor-up/oncofetal/aneuploidy/residual support
should demote the LIHC identity interpretation. For genuine LIHC, tumor-up
markers, proliferation/oncofetal support, aneuploidy, and residual LIHC
signature should prevent demotion.

This addresses #101 as a general rule rather than a COAD/liver patch.

### Stage 2: compartment lock

Use `compartment_call` as the primary compartment prior, but make the output an
explicit stage decision:

```text
compartment = confident centroid compartment
            OR consensus(lineage panels, residual lineage_fit, broad signatures)
            OR calibrated learned compartment vote
            OR abstain
```

When the compartment locks, later candidates outside the compatible
compartment are not allowed to win except by definitive molecular evidence.
When it abstains, the downstream decision carries multiple compartments with
penalties instead of silently defaulting to solid.

This preserves the existing #83 win while making the restriction part of the
main hierarchy instead of a post-rank reorder.

### Stage 3: family and organ-context resolution

Within the locked compartment, decide the coarse family/tissue context before
exact entity:

- GI adenocarcinoma versus hepatobiliary versus pancreaticobiliary,
- squamous versus mammary basal versus urothelial,
- sarcoma broad classes,
- neuroendocrine classes,
- CNS and embryonal classes.

Use contrast panels and tumor-intrinsic lineage panels here. Existing rescues
become ordinary discriminators with weights and blockers, not post-hoc
overrides.

Use the learned family vote as a candidate-admission and corroboration signal,
not a replacement for family evidence. A learned family can keep a family in the
beam even when the flat broad ranker misses it, but exact entity selection still
requires the Stage 4 evidence model.

The broad ranker should no longer mix family factor, purity, and exact label
in one geomean. It should produce stage-specific support:

```text
family_identity_support
normal_background_penalty
compartment_consistency
family_margin
```

### Stage 4: exact entity selection

For candidates admitted by the prior stages, compute an interpretable entity
score. A workable first version is a log-evidence model with named terms:

```text
entity_score =
  + signature_identity
  + tumor_up_vs_matched_normal
  + hallmark_presence
  + lineage_panel_fit
  + residual_signature_identity
  + direct_molecular_evidence
  + centroid_broad_corroboration
  + learned_entity_corroboration
  - normal_origin_dominance
  - compartment_incompatibility
  - hallmark_absence
```

The important change is not the exact arithmetic; it is the separation of
identity evidence from purity/background evidence. Purity can scale confidence
or explain weak expression, but it should not be a positive type-identity
factor.

For #101, LIHC loses because its bulk support is explained by normal liver
background, while COAD/READ retain residual and signature identity after liver
subtraction.

### Stage 5: subtype and molecular-axis resolution

Subtypes should be resolved only after the parent entity is chosen. They should
not be allowed to change the parent unless a defining molecular event requires
it.

Rules:

- centroid can anchor a subtype only inside the selected parent/family;
- learned subtype/molecular-axis votes can refine only inside the accepted
  parent/family unless a defining molecular event upgrades the child to an
  entity;
- same-lineage subtype flips require subtype-specific marker/hallmark support
  or a family-specific decisive centroid margin;
- sarcoma subtype margins should be stricter than epithelial parent margins;
- molecular expression states such as LUAD_KRAS_STK11 or BRCA_Basal are
  orthogonal axes unless the registry treats them as reportable entities.
- subtype references should be used as expression references only at the role
  they support: direct subtype ranges for subtype expression/ranges, parent
  fallback for purity/decomposition, and molecular-status cohorts as state
  evidence unless direct molecular support upgrades them.

For #102, SARC_RMS_ERMS would remain in the RMS subtype unless the alternative
sarcoma subtype has both a clear centroid advantage and matching subtype
markers. Whole-profile centroid alone would be corroboration, not authority.

### Stage 6: one final decision object

The pipeline should produce one `CancerCallDecision`:

```text
tumor_status
dominant_backgrounds
compartment_decision
family_decision
entity_decision
subtype_decision
orthogonal_axes
purity_and_decomposition
evidence_trace
learned_evidence_trace
abstentions_and_conflicts
```

Reports, therapy selection, ranges, provenance, and confidence should read
that object. `cancer_type_evidence` can remain as a renderer/serializer of the
trace, but it should stop being a second selector that can require a later
purity/decomposition reroute.

The rendered `*-evidence.md` and detailed `*-analysis.md` files should preserve
the complete decision evidence table: selected and blocked candidates, learned
votes at every stage, calibration caveats, marker positive/negative axes,
centroid/compartment support, decomposition/residual signals, fusions, and
normal-background explanations. The report should make clear which signals
admitted a hypothesis, which signals blocked it, and which signals actually
selected the final label.

## Implementation plan

### Phase 1: instrumentation with no behavior change

Add the feature frame and decision trace alongside the current outputs.
Populate it from existing functions and emit it in calibration artifacts.

Acceptance:

- no behavior changes,
- no duplicate centroid/decomposition/purity recomputation in hot paths,
- 565 harness records per-sample feature terms and stage decisions,
- local and 565 validation runs can emit:
  `cancer_type_signal_matrix.tsv` for the rich one-row-per-signal trace,
  `cancer_type_signal_sample_summary.tsv` or `--signal-summary-out` for the
  compact one-row-per-sample validation view, and a compact markdown summary,
- evidence markdown includes a complete trace of learned and non-learned
  co-signals for selected and blocked hypotheses.

### Phase 2: candidate universe and stage separation

Refactor candidate admission so "considered" and "ranked winner" are separate.
Always keep the true parent/family in the candidate universe when any upstream
stage has plausible support.

Move family panels, lineage exclusion, hallmark veto, and normal-tissue
evidence into stage-specific columns rather than post-rank sort mutations.
Add learned compartment/family/entity votes as admission evidence with explicit
out-of-fold calibration metadata; do not let admission imply selection.

Acceptance:

- existing top-1/entity/lineage metrics unchanged or improved,
- candidate trace exposes why a candidate was admitted, blocked, or demoted.

### Phase 3: remove purity from type identity

Remove `purity_estimate` from `_candidate_support_score` only after replacing
it with:

- tumor-present confidence,
- normal-origin dominance penalty,
- residual signature/tumor-up support for structured host backgrounds,
- compartment consistency.

Add a direct regression for #101:

```text
70% normal liver + 30% COAD -> COAD/READ entity, liver as dominant background
```

Also add genuine LIHC/KIRC organ-confined controls so the dominance penalty
cannot simply demote all organ-like tumors.

### Phase 4: fold report-scope selection into the main decision

Recast `select_report_scope_from_evidence` selectors as feature providers or
stage-specific discriminators. Direct fusions remain authoritative; rare
markers and local references become exact-entity/subtype evidence with explicit
context gates.

Delete the need for `_reroute_decomposition_to_call` by computing purity and
decomposition after the final entity decision, or by storing all candidate
decomposition modes in the feature frame and selecting one view at the end.

Replace `_veto_local_reference_lineage_flip` with a general rule: a lower-stage
exact-reference channel cannot cross a locked compartment unless it carries
definitive molecular evidence or the compartment stage abstained.

### Phase 5: subtype policy and #102 fix

Make centroid corroboration stage-aware:

- compartment/broad entity: current margin can remain a starting point;
- same-parent subtype: require subtype-marker agreement or a larger
  family-specific margin;
- sarcoma subtypes: default to marker/local-reference authority on near-ties.

Add a direct regression for the five SARC_RMS_ERMS samples:

```text
SARC_RMS_ERMS -> SARC_RMS_ERMS exact, not SARC_LPS_UNSPEC or SARC_MYXFIB
```

Track exact, parent, sibling, entity, lineage, and miss categories so a subtype
fix cannot hide broader regressions.

### Phase 6: 565 optimization and guardrails

Use `scripts/eval_per_sample_confusion.py` as the primary A/B harness, extended
to record:

- stage outputs,
- selected feature terms,
- the first stage where the truth candidate lost,
- every selector/demotion that changed the winner,
- and ablation deltas for centroid, residual, normal-origin dominance, and
  subtype policy,
- out-of-fold learned predictions for flat and hierarchical learned models.

Add a hierarchical learned-model harness:

```text
compartment model -> family-within-compartment model
                  -> entity-within-family model
                  -> subtype/molecular-axis model
```

Compare flat 118-way top-1/top-3, hierarchical top-1/top-3, and hybrid
candidate-admission performance. Thresholds used by production selectors must
come from out-of-fold or held-out estimates, not from all-trained probabilities.

Targets:

- #101 passes,
- #102 returns to exact for RMS,
- no new lineage misses,
- entity-correct improves over the current 80% issue baseline,
- exact/subtype gains are accepted only when they do not trade away lineage or
  entity accuracy.

The aspirational target is 100% on the 565 available cancer samples. Treat that
as an optimization goal, not a license to overfit: if the remaining misses are
biologically indistinguishable siblings, the correct outcome may be a
documented abstention or parent/family call rather than a forced exact label.

## What should disappear

After the redesign, these should become unnecessary or much smaller:

- support-score purity crutch for type ranking,
- post-rank rescues that directly mutate `support_score`,
- separate evidence-selector centroid corroboration,
- local-reference lineage veto in `main.py`,
- purity/decomposition rerouting after a report-scope flip,
- subtype labels acting sometimes as entities and sometimes as annotations.

The replacement is not fewer signals. It is fewer places where signals can
change the answer.
