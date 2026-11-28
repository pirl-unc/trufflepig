# Trufflepig documentation

Start with the production workflow, then use the narrower design and operations
notes only when you need their implementation detail.

## Start here

1. [Cancer-call decision workflow](./CANCER_CALL_DECISION_FLOW.md) — the
   high-level path from an expression table to the final cancer call,
   decomposition, therapy shortlist, and report.
2. [Infantile spindle-cell molecular interpretation](./infantile-spindle-molecular-interpretation.md)
   — IFS, CMN, NTRK-spindle driver overlap, variant-gated therapies, and
   current trial context.
3. [Exact spindle-sarcoma therapy panels](./spindle-sarcoma-therapy-panels.md)
   — diagnosis-specific IMT, DFSP, PEComa, and GIST therapy scope without
   sibling-panel leakage.
4. [Variant inputs and coordinate provenance](./variant-inputs.md) — the typed
   exact/symbolic variant contract, source identity, and build checks.

## Classifier evidence and design records

- [Hierarchical cancer-type classifier](./cancer-type-hierarchical-classifier.md)
  explains the compartment-to-entity architecture and its validation history.
- [Cancer-type ontology](./cancer-type-ontology.md) explains ancestry,
  abstention, and how broad and fine labels relate.
- [Residual-matching findings](./cancer-type-residual-matching-findings.md)
  records a negative experiment so whole-profile residual matching is not
  repeated without new evidence.

These documents contain dated experiment snapshots. Treat their measurements as
historical evidence unless a section explicitly identifies the current release
and evaluation corpus.

## Internal planning notes

Working redesign plans and calibration snapshots live under `notes/`:

- [RNA-seq cancer-call redesign](../notes/rnaseq-cancer-call-redesign.md) — why
  the decision logic was consolidated and the staged architecture.
- [Report belief consistency and friendliness](../notes/report-belief-consistency-and-friendliness-plan.md)
  — how finalized conclusions and uncertainty flow into every report artifact
  without contradiction.
- [Calibration baseline JSON](../notes/calibration-baseline.json) — recorded
  baseline snapshot for `scripts/calibrate_decomposition.py`.

## Operations

- [Decomposition calibration](./CALIBRATION.md) describes the calibration
  harness, reproducible commands, and recorded regression baselines.
- [Service and batch performance](./SERVICE_PERFORMANCE.md) describes runtime,
  memory, and deployment-oriented performance work.
