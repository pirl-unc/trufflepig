# Report belief-consistency + friendliness redesign plan

Companion to `docs/rnaseq-cancer-call-redesign.md`. That doc redesigns *how the
cancer type is decided*; this one redesigns *how every per-sample conclusion is
represented, finalized, and rendered* so figures, markdown, and the PDF can
never disagree — and then makes the rendered report legible and actionable for a
mixed audience (patient, therapy-navigation expert, oncologist).

Two tiers, in order:

1. **Tier 1 — correctness + information flow.** One immutable, uncertainty-
   carrying decision object, finalized before anything renders, read by every
   renderer. Extends the redesign doc's Stage 6 (`one final decision object`)
   and Phase 4 (`delete _reroute_decomposition_to_call by computing purity and
   decomposition after the final entity decision`).
2. **Tier 2 — friendlier report.** Rebuild the markdown + PDF on top of the
   frozen object: one at-a-glance decision block, therapy/consequences promoted,
   duplicated/contradictory figures removed, expression→consequence reasoning
   made explicit.

---

## 0. Diagnosis: this is a belief-consistency bug, not a purity bug

`analysis` is a mutable dict that many phases write and many renderers read, with
no point at which it is declared final. The purity field alone is swapped or
mutated at **six** stages during `main.py::analyze` (line anchors from the seam
map):

```
T0   analyze_sample builds analysis["purity"]                tumor_purity.py:4647
T4   _reroute_decomposition_to_call → _set_analysis_purity   main.py:2379 / 2843-caller-2 5248
T6   apply_sample_context_to_purity widens CI (in place)     flow.py:120-132
--- FIGURE #1: plot_sample_summary reads purity here (T7) --- main.py:2620
T9   decomposition results attached to analysis             main.py:2792-2818
T10  should_adopt_decomposition_purity → _set_analysis_purity main.py:2838-2843  (swaps whole purity obj)
T11  lineage-panel override rewrites overall_estimate/lo/hi  main.py:2866-2887
T12  decomposition interval cap lowers overall_upper         main.py:2896 / 5058-5061
--- FIGURE #2/#3: plot_tumor_purity, plot_purity_method_comparison (T12) --- main.py:3111, 3135
--- MARKDOWN: _generate_text_reports / build_summary (T12) --- main.py:3364, 3734
     (also the ONLY writer of analysis["purity_confidence"], main.py:8050 —
      after every figure, so no figure ever sees confidence)
```

`plot_sample_summary` renders purity at **T7** (78% — the pre-adoption candidate
value, identical to READ's `purity_estimate` in the candidate table); every other
artifact renders at **T12** (10%). Both are "correct" reads of a moving object.

The same disease affects every belief that is produced in stages and read by
multiple consumers: cancer type (`inferred_cancer_type` / `reference_cancer_type`
/ `report_scope_cancer_type` / `cancer_type` are mutated at T1/T2/T3/T4/T7),
composition, MSI/MMR, site. Purity is just the one that produced a visible 8×
contradiction.

Root causes:

- **No finalize barrier.** Renderers run interleaved with mutations.
- **No single source of truth.** `_set_analysis_purity` (2 callers) is *almost*
  a chokepoint, but T0/T6/T11/T12 bypass it and mutate `analysis["purity"]`
  directly.
- **Uncertainty is dropped in transit.** Intervals (`estimate/lower/upper`)
  exist only at the aggregate purity level; the moment purity is projected into
  a candidate row (`purity_estimate`, bare scalar) or into decomposition
  (`residual_fraction`, bare scalar) the interval is lost — so a point value can
  travel forward and be re-adopted with no memory of its own uncertainty.
- **Nothing is typed to prevent mix-ups.** Candidate-level purity and adopted
  purity are both "a float in a dict named purity," so a renderer reaching for
  one silently gets the other.
- **Two different `DecompositionResult` types** (`decomposition/engine.py:411`
  vs `expression_decomposition.py:119`) with different shapes — a latent
  conflation hazard.

---

## Tier 1 — correctness and information flow

Design principle (user's words): *reasoning should consider all available
information at each step, come to uncertainty-encoding conclusions (ranges, top
category vs worth-considering alternative), and pass those forward in a
hard-to-mix-up way.*

### 1.1 One uncertainty-carrying value type: `Estimate` / `Call`

Today the only first-class "top + alternatives + support + provenance + veto"
structure is `CancerTypeEvidence` / `select_report_scope_from_evidence`
(`cancer_type_evidence.py`). Everything else (purity, MSI, site, subtype,
confidence) is a bespoke dict. Lift the pattern into two small frozen generics
(new module, e.g. `trufflepig/belief.py`):

```python
@dataclass(frozen=True)
class Estimate:                      # a scalar quantity with uncertainty
    point: float
    lower: float | None
    upper: float | None
    unit: str                        # "fraction" | "tpm" | "prob" ...
    adopted_method: str              # which method won
    method_table: tuple[MethodEstimate, ...]   # every method + its value + quality + why kept/downweighted
    flags: tuple[str, ...] = ()      # reconciliation notes ("signature deprioritized", "capped by non-tumor mass")
    confidence: str | None = None    # low/moderate/high tier

@dataclass(frozen=True)
class MethodEstimate:
    name: str                        # "signature" | "lineage_panel" | "ESTIMATE_combined" | "decomposition_residual"
    point: float
    lower: float | None
    upper: float | None
    quality: float | None            # the method's own goodness/stability score
    role: str                        # "adopted" | "corroborating" | "deprioritized" | "gated_off"
    reason: str | None               # why it was downweighted / gated

@dataclass(frozen=True)
class Call:                          # a categorical quantity with alternatives
    top: str
    top_label: str
    alternatives: tuple[Alternative, ...]   # retained, ranked, with margin + why-not-selected
    selected_by: str
    margin: float | None
    is_candidate_set: bool           # True when we abstain to {COAD, READ}-style set
    evidence: tuple[EvidenceChannel, ...]   # reuse cancer_type_evidence channels
    confidence: str | None = None
```

`Estimate` subsumes the existing `(overall_estimate, overall_lower,
overall_upper)` triple *plus* the `components`/`integration` method table that
`plot_purity_method_comparison` already reconstructs by hand
(`tumor_purity.py:2531-2544`). `Call` subsumes `candidate_trace` +
`select_report_scope_from_evidence` (which becomes a *builder/serializer* of a
`Call`, never a second selector — per redesign doc Stage 6).

Key property: **`Estimate` and `Call` are frozen.** Once built they can't be
mutated; a "new" belief is a new object. This structurally kills T6/T11/T12-style
in-place edits.

### 1.2 One finalized decision object: `SampleDecision`

Extend the redesign doc's Stage-6 `CancerCallDecision` to carry *all* beliefs:

```python
@dataclass(frozen=True)
class SampleDecision:
    cancer_type: Call
    subtype: Call | None
    compartment: Call
    purity: Estimate
    composition: CompositionBelief          # tumor + named non-tumor components, each an Estimate/fraction
    mmr_state: Call | None                  # MSI-like vs MSS, prob + confirm-with-assay caveat
    site_context: Call | None
    aneuploidy: Estimate | None
    pathway_state: tuple[PathwayAxis, ...]  # each axis: fold vs cohort + direction + n_genes
    targets: tuple[TargetRow, ...]          # ONE ranked target list (see Tier 2)
    quality: SampleQuality                  # library, preservation, QC
    provenance: dict                        # inputs, versions, constraints
```

Contract: **`SampleDecision` is built by exactly one `finalize()` call, after
every belief-producing step has run, and is the only thing renderers receive.**
No figure, no markdown writer, no PDF builder reads the mutable `analysis` dict.

### 1.3 One purity reconciliation site (kills the swap chain)

Replace the T0→T12 chain with a single reconciler that sees every method and its
quality at once and emits one `Estimate`:

```python
def reconcile_purity(methods: list[MethodEstimate],
                     cohort_prior: float | None,
                     sample_quality: SampleQuality) -> Estimate:
    # signature, lineage_panel, ESTIMATE stromal/immune/combined, decomposition_residual
    # gate methods invalid for the lineage (ESTIMATE for heme/mesenchymal — see the
    #   ESTIMATE-purity memo), downweight low-quality/low-stability signatures,
    # choose adopted point + interval, widen for degradation, cap by modeled non-tumor mass,
    # return Estimate(point, lo, hi, adopted_method, method_table, flags, confidence).
```

This is the literal implementation of "consider all available information, come
to an uncertainty-encoding conclusion." The current logic already lives across
`_combine_purity_estimates` (`tumor_purity.py:1434+`), `_integrate_purity_signals`,
`should_adopt_decomposition_purity` (`flow.py:136`), the lineage-panel override
(`main.py:2860-2887`), and `_constrain_purity_interval_with_decomposition`
(`main.py:5023`) — this consolidates them into one pure function with one output.

### 1.4 Hard-to-mix-up typing

- Candidate rows keep `purity_estimate: float` **for ranking only**, and it is a
  distinct type/name from the sample's adopted `purity: Estimate`. Ranking code
  takes the scalar; renderers take the `Estimate`. A renderer literally cannot be
  handed the candidate scalar — different type at the boundary.
- Rename one of the two `DecompositionResult`s (e.g. `CompartmentFit` for the
  NNLS engine, keep `DecompositionResult` for the lineage-routed model) so the
  conflation is impossible.
- `residual_fraction` becomes a `MethodEstimate` (carry its monotone-not-absolute
  caveat as `role`/`reason`), not a bare float that can masquerade as purity.

### 1.5 Phasing (mirror the redesign doc's Phase numbering where possible)

**Phase 0 — immediate hotfix (hours, no refactor). ✅ DONE (2026-07-07).**
Stop the visible contradiction now: finalize purity (run T9–T12) *before*
`plot_sample_summary`, or move the three figure renders below `main.py:2896`.
Add the golden invariant test in 1.6. This is safe to land independently of
everything below and should go first so no shipped report contradicts itself
while the refactor lands.

*Landed:* moved the `plot_sample_summary` + standalone-panel render block in
`_analyze_body` from before decomposition to after the purity-finalization block
(decomposition adoption → lineage-panel override → interval cap), outside the
`if decomp_results:` guard so it still renders when decomposition is empty.
Verified end-to-end on the READ/caris sample: the headline figure now reads
"Tumor purity 10% [6–16%] / Tumor (10%)", matching purity-methods,
decomposition-composition, and the markdown. Guard test:
`tests/test_purity_render_ordering.py` (interim source-order contract, to be
replaced by the behavioral cross-artifact invariant once the ReportView barrier
lands — see 1.6). The behavioral invariant test in 1.6 is still TODO and is the
durable replacement.

**Phase 1 — instrument, no behavior change.** Build `SampleDecision` +
`Estimate`/`Call` alongside the current dict; populate them from existing values
at the end of `analyze()`. Add the cross-artifact assertion harness (1.6) and run
it over the local 20-report sweep + a 565 slice. Rename the second
`DecompositionResult`. Acceptance: byte-identical reports, harness green.

**Phase 2 — single reconciliation.** Introduce `reconcile_purity`; delete the
T6/T11/T12 in-place mutations and fold T4/T10 into it. Per redesign-doc Phase 4,
compute purity/decomposition **after** the final entity decision and delete
`_reroute_decomposition_to_call`. Acceptance: purity numbers unchanged on the
sweep except where the old order was demonstrably wrong; harness green.

**Phase 3 — freeze + retype.** Make `SampleDecision` frozen and the sole renderer
input; port `plot_sample_summary`, `plot_purity_method_comparison`,
`build_summary`, `_generate_text_reports`, `build_actionable`, and
`build_interpretive_report_pdf` to read it. Retype candidate vs adopted purity.
Renderers stop importing `analysis`.

**Phase 4 — generalize.** Roll composition, MMR, site, subtype, aneuploidy onto
`Estimate`/`Call`. `cancer_type_evidence` becomes a `Call` builder/serializer.

### 1.6 Test strategy

- **Golden cross-artifact invariant (add first).** One test that runs a fixed
  sample end-to-end and asserts the purity/composition/cancer-call *rendered into
  every artifact are identical*: parse the number out of `sample-summary`
  (or read the `SampleDecision` the figure was built from), `purity-methods`,
  `summary.md`, `analysis.md`. This test fails on today's tree (78 vs 10) and
  passes after Phase 0. It is the regression lock for the whole class of bug.
- **Immutability property test.** `SampleDecision`/`Estimate`/`Call` reject
  mutation.
- **Reconciliation unit tests.** `reconcile_purity` on constructed method sets:
  low-purity decomposition-dominated case (the READ sample: signature 26%,
  lineage 90%, ESTIMATE 55–58%, decomposition 10% → adopted 10% with method_table
  intact); heme case (ESTIMATE gated off); degraded case (CI widened).
- **Regression:** local 20-report sweep (`scripts/regenerate_local_reports.py
  --with-figures`) + the 565 harness; diff `SampleDecision` JSON per sample.

---

## Tier 2 — friendlier, non-contradictory, actionable report

All of this renders from the frozen `SampleDecision`, so figure/text agreement is
structural, not hand-maintained. Findings below are anchored to the reviewed
READ report.

### 2.1 summary.md → a real at-a-glance decision block

Current top-of-file is a stack of bold key:value prose lines
(`brief.py:2372-2420`). Replace with a compact, audience-layered header built
from `SampleDecision`:

- **Line 1 — the call:** `cancer_type.top_label` + confidence + (if
  `is_candidate_set`) the retained alternative and the one-line reason it's not
  resolved (COAD vs READ, margin below leaf threshold).
- **Line 2 — how sure / how pure:** `purity.point` + interval + the single master
  caveat when low ("at 10% purity, raw TPMs overstate tumor signal — prefer
  tumor-attributed values"). This caveat then propagates to every TPM the report
  shows, not just here.
- **Line 3 — what to do next:** the top 1–3 targets with their eligibility gate
  (what assay confirms them), not just their names.
- **MMR/immuno one-liner** only when it changes an action, with the confirm-with-
  assay caveat inline.

Keep it ≤ ~15 lines. Everything else is one click down.

### 2.2 analysis.md → decision-first ordering

Current order buries the actionable content (therapy is dead last at
`main.py:8331`, after a 90-line static gene-panel dump). Reorder:

1. Decision + confidence (cancer type Call with alternatives + what selected it).
2. Purity/composition (the `Estimate` method_table renders directly as the
   "why 10% and not 78%" table — the reconciliation is now visible, not hidden).
3. Targets + therapy (promoted up).
4. Expression consequences (2.4).
5. Evidence/provenance trace (the per-candidate Δlog2 tables — keep, they're
   genuinely good).
6. QC + sample quality (demoted below the call).
7. **Move the Embedding-Features gene dump** (genes-per-cancer-type ×33,
   genes-per-tissue ×50; `main.py:7944-8031`, ~90 lines) out of the main report
   into `evidence.md` or an appendix — it's static reference-panel documentation,
   not sample-specific.

### 2.3 One target ranking, one purity, everywhere

- The markdown therapy shortlist (maturity-ordered: BRAF, PDCD1, EGFR) and
  `priority-targets.png` (expression-priority-ordered: ERBB2, EGFR) currently
  disagree on the "top targets." Emit **one** `targets` list on `SampleDecision`
  with an explicit primary sort and a secondary column, and render both the table
  and the figure from it. If two orderings are genuinely wanted, label them
  ("guideline-maturity order" vs "expression-priority order") and show the same
  rows in both.
- Purity/composition: one number from `SampleDecision.purity`
  /`.composition` everywhere (fixes 78 vs 10 by construction).

### 2.4 Make expression *consequences* explicit

The pipeline computes the signals; the report states facts without the "so what."
Add a consequence layer keyed off `SampleDecision`:

- **MLH1 613 TPM (100th pct, retained-high) vs MMR ensemble "MSI-like 0.67":**
  high MLH1 expression argues *against* MLH1-silencing MSI — surface the tension
  explicitly instead of listing both facts 3 lines apart.
- **Suppressed antigen presentation + suppressed IFN (`therapy-pathway-state`)
  next to the PDCD1/pembrolizumab row:** connect the pathway state to the
  checkpoint-therapy discussion; right now the figure is an orphan and the text
  says "no pathway pattern passed thresholds" (a direct contradiction — the
  threshold logic and the figure must share one source).
- **Biomarker-outlier framing:** TP53/KRAS/NRAS are surfaced as "notable
  outliers" by mRNA level (`summary.md`), but these are mutation-driven
  biomarkers where high mRNA is not actionable — which contradicts the report's
  own repeated "target expression is not the eligibility criterion." Either drop
  expression-only mutation-driver outliers or explicitly annotate "expression is
  not the biomarker here."
- **Low-purity master caveat inline** on every tumor-source TPM, sourced from
  `purity.confidence`.

### 2.5 Figure set: dedup and de-contradict

- **One target figure.** `priority-targets.png` (score decomposition — answers
  "why is this ranked here") is the keeper; fold the one decision-relevant TPM cue
  (tumor-source vs max-vital-tissue safety band) into it. Drop
  `priority-target-context.png` and `actionable-targets.png` from the reader PDF
  (near-duplicate log-TPM dumbbell plots over the same ~18 genes); keep in the
  audit PDF.
- **Kill composite/standalone duplication.** `sample-summary.png` is a 4-panel
  composite whose panels are *also* emitted standalone (`cancer-hypotheses`,
  `background-tissues`, `mhc-expression`; `main.py:2628-2634`). Pick one.
- **Curated panel scatters** (`Growth_receptors`, `Oncogenes`, `CTAs`,
  `Immune_checkpoints`, `DNA_repair`, …, ~10) overlap each other and the target
  figures with no cross-explanation → audit/appendix only.
- **`cancer-type-signal-matrix.png`**: collapse the ~13 near-identical per-fold
  MSS/MSI classifier bars into one MSS-vs-MSI summary with spread; stop co-plotting
  probabilities (0–1) and fused-evidence support (→2.45) on one "Evidence support
  / probability" axis.
- **Pathway-state figure** must be gated by the same threshold logic the text
  uses, so it can't show content the text denies.
- Where two figures legitimately show the same genes for different reasons, add a
  one-line "how this differs from X" caption.

### 2.6 PDF: render the decision, don't scrape the markdown

`scripts/build_interpretive_report_pdf.py` currently scrapes `summary.md`, drops
every table row and header (`_highlight_lines`, line 66), and truncates at 28
prose lines — discarding the therapy table and target-attribution table (the most
actionable content) and captioning figures with the PNG filename (line 201).

- Render from `SampleDecision`: first page = the at-a-glance block + the therapy
  table + top-3 targets with tumor-source TPM and safety context.
- Caption each figure with its interpretation sentence, not its filename.
- Derive the figure manifest from `SampleDecision` (only include a figure whose
  underlying belief exists / passed threshold) so the PDF can't ship a figure the
  text contradicts — replaces the hard-coded `FIGURE_SPECS` list.
- Because both the PDF and the markdown now render from one `SampleDecision`,
  "markdown and PDF overlap on one report" is guaranteed with identical numbers,
  not best-effort.

---

## What should disappear

- `_reroute_decomposition_to_call` (redesign-doc Phase 4).
- The T6/T11/T12 in-place purity mutations and the second `_set_analysis_purity`
  swap — replaced by one `reconcile_purity` + one `finalize()`.
- Renderers reading the mutable `analysis` dict.
- The hard-coded PDF `FIGURE_SPECS`; the composite-vs-standalone figure
  duplication; `priority-target-context.png` + `actionable-targets.png` from the
  reader PDF.
- The mid-report Embedding-Features gene dump (moves to appendix/evidence.md).

## Handoff notes for the next session

- Land **Phase 0 + the golden invariant test first** — it's independent of the
  in-flight cancer-call work and stops shipped reports from contradicting
  themselves immediately.
- Tier-1 Phases 1–4 should be sequenced *after* codex's current
  cancer-call-evidence changes settle, since `Call` wraps
  `select_report_scope_from_evidence` and they touch the same files.
- Run everything through `./test.sh` (not raw pytest) and the local report sweep
  with `--with-figures`; the memory notes and CLAUDE.md apply.
