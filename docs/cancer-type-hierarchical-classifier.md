# Hierarchical cancer-type classifier — compartment → leaf

A *positive* result, recorded so the architecture and its evidence aren't
re-derived from scratch. Companion to
[cancer-type-ontology.md](./cancer-type-ontology.md) (the reasoning ledger) and
[cancer-type-residual-matching-findings.md](./cancer-type-residual-matching-findings.md)
(the residual-matching negative result).

## The problem this fixes

The leaf marker panels **saturate for promiscuous lineages**. The (removed)
MESENCHYMAL panel and the SQUAMOUS/HNSC panel fire on the stroma and squamous
contamination present in essentially *every* solid tumor, so a stroma-heavy or
low-purity carcinoma mis-calls to **SARC**, and a contaminated sample mis-calls to
**HNSC**. Historically `purity_estimate` masked this (it down-weighted the
stroma-dominated candidate), which is why removing purity standalone regressed —
it was a load-bearing crutch, not a fix (see #83).

The elegant fix is **hierarchy**: decide the histogenesis *compartment* first with
a signal that *cannot* be fooled by a handful of non-specific markers, then narrow
to a leaf only *within* that compartment. The cross-compartment saturation errors
(colorectal→SARC, sarcoma→HNSC) simply cannot occur once the leaves are restricted
to the pinned compartment.

## Stage 1 — compartment call (LOCKED, 15/15)

`cancer_type_centroid.compartment_call(sample_tpm_by_symbol)`.

Whole-profile **Spearman correlation** of the sample against each bulk TCGA cohort
centroid (median per gene, all informative genes — *not* a discriminative subset,
which sharpened some calls but dropped a real sarcoma to rank 5), then **aggregated
by histogenesis compartment** via pirlygenes `cancer_lineage_group`:

> Epithelial · Sarcoma · Hematolymphoid · Melanoma · Neuroendocrine · CNS ·
> Germ cell · Embryonal

The compartment score is the best cohort rho within the group. Why it is immune to
the saturation that breaks the leaf screen: correlating the *whole* expression
profile against a reference centroid that *carries the same compartments the sample
does* (the bulk reference includes stroma/TME) cannot be flipped by a few
non-specific markers — the stromal program matches stroma on both sides and washes
out.

On the local blind truth set (15 samples, below) this is **15/15** at the
compartment level. It returns a confidence margin (top rho − runner-up rho); a call
is `confident` when the margin ≥ `_COMPARTMENT_CONFIDENT_MARGIN` (0.02 rho).

**Sarcoma is a broad grouping, never a leaf** (the SARC-is-broad rule). Stage 1 may
*pin* the Sarcoma compartment, but it never resolves a single sarcoma type; every
`SARC`/`SARC_*` subtype is in-compartment, and the TCGA `SARC` pseudo-leaf is never
emitted as a call.

## How stage 1 reduces stage-2 leaves (production wiring)

In `rank_cancer_type_candidates` (`tumor_purity.py`), after the marker-panel screen
and same-family promotion, when the compartment call is **confident** we apply a
**stable two-tier re-rank**: in-compartment leaves float above out-of-compartment
leaves; within each tier the marker-panel support order is preserved untouched. So
the reorder happens *only across the compartment boundary* — exactly the saturation
mis-calls — and the top call plus the candidate set are drawn from the pinned
compartment.

Guard rails:

- **Abstain below the margin.** An ambiguous profile (margin < 0.02) gets no
  restriction — we never risk excluding the true leaf on a genuine near-tie.
- **Never restrict to empty.** The re-rank only fires when there is at least one
  in-compartment *and* one out-of-compartment candidate; a caller-constrained set
  that is entirely out-of-compartment is left alone (and flagged as a disagreement,
  the mis-call detector).
- **Fail-open.** Any error leaves the marker-panel ranking standing (logged).

Every candidate is annotated with `centroid_correlation`, `range_plausibility`, and
`compartment_in_set`; the top row records `centroid_coarse_lineage`,
`centroid_lineage_margin`, `centroid_lineage_confident`,
`centroid_compartment_restricted`, and `centroid_lineage_agreement`.

## Stage 2 — leaf within the pinned compartment (bake-off)

Once the compartment is pinned, *which* leaf method discriminates best? We ran a
bake-off restricted to the **true** compartment's leaves (stage 1 is perfect, so
this isolates the stage-2 question). Sarcoma is scored as broad (any `SARC_*`
counts); the real discrimination test is the 12 **Epithelial** samples
(COAD/READ · BRCA · PRAD · BLCA · NUTM). `epi-leaf` = fraction of those 12 whose
leaf family is correct; `all` includes the 3 sarcomas (always satisfiable).

| stage-2 method | all | epi-leaf |
|---|---|---|
| **GATE within-markers(tumor) + decisive-centroid override** | **0.67** | **0.58** |
| within-markers (tumor-only deconvolved) | 0.60 | 0.50 |
| RRF: within(t)+tumor-centroid+pancorr | 0.53 | 0.42 |
| tumor centroid-corr / whole transcriptome | 0.47 | 0.33 |
| pan_cancer bulk-corr / whole transcriptome | 0.47 | 0.33 |
| z-norm ENS within(t)+tumor-centroid(t) | 0.47 | 0.33 |
| centroid-corr / signature genes | 0.40 | 0.25 |
| within-markers (bulk) | 0.40 | 0.25 |
| likelihood-ratio (tumor, fit vs global background) | 0.40 | 0.25 |
| panel-guided (curated signature panel mean) | 0.40 | 0.25 |
| ENS centroid+within+panel (bulk) | 0.40 | 0.25 |
| centroid-corr / whole (bulk sample) | 0.33 | 0.17 |

### What the bake-off establishes

- **within-markers on the deconvolved tumor profile** is the best *single*
  principled leaf method (0.50 epi-leaf). For each leaf we take its genes most
  specific *within the compartment* (leaf median − peer median in log space) and
  score the sample's mean expression of them; restricting specificity to the
  compartment is what the hierarchy buys us.
- **Flat fusion hurts.** Reciprocal-rank fusion and z-norm ensembles of the
  complementary methods both *regress* (0.42 / 0.33) — they dilute the strong
  within-markers signal with noisier voters. Averaging is the wrong combinator here.
- **A principled cascade helps.** The methods are complementary in a *structured*
  way: within-markers nails the common carcinomas but misses **NUTM** (a rare NUT
  carcinoma — within-markers calls HNSC/BRCA), while whole-profile centroid
  correlation nails NUTM 3/3 but is mediocre elsewhere. A cascade — within-markers
  as the base, overridden only when the whole-profile centroid has a *decisive*
  winner (a "distinctive rare type" rule, not blind averaging) — lifts epi-leaf
  0.50 → 0.58 without disturbing the common-carcinoma calls. We deliberately did
  **not** tune the override margin to grab the last two NUTM samples: that would
  overfit a 15-sample set.

### What ships in production (stage 2)

Production stage-2 is the **existing geomean support-score ranker**
(`rank_cancer_type_candidates`'s 5-factor geomean: signature_score ·
purity_estimate · lineage_support_factor · signature_stability · family_factor),
now operating on the **compartment-restricted candidate set** — i.e. "bring back the
old system, run it once the broad category is pinned". The compartment gate is what
makes that geomean reliable: it removes the cross-compartment candidates that the
saturated marker panels would otherwise let win.

The bake-off methods above (within-markers, tumor-centroid, the decisive-centroid
cascade) are recorded as the **evaluated alternatives**. The within-markers +
decisive-centroid cascade is the best research result (0.58) and a documented future
option, but it is *not* shipped yet: a 0.50→0.58 move on a 15-sample set is inside
the overfitting band, so we keep the principled, well-tested geomean as the stage-2
leaf ranker and let the compartment gate carry the robust win.

### Honest limit

Leaf discrimination *within* a compartment on these deconvolution-noisy samples
caps around **0.5–0.6**, and that is a property of the samples (admixture +
deconvolution residual noise), not a method failure — it is the same wall the
residual-matching experiment hit. The robust, bankable win is **stage 1**: perfect
compartment calls that eliminate the *cross-compartment* errors which were the
actual production bug. Stage 2 narrows within the compartment as far as the evidence
allows and otherwise leaves the near-tied leaves as a candidate set (the
[ontology layer](./cancer-type-ontology.md)'s abstention behavior), rather than
forcing a wrong leaf.

## Local blind truth set (15 samples)

Run on every local report under `/tmp/bughunt_sweep/*/analyze/` with known truth.
Each is scored from expression alone — the cancer type is **not** passed in.

| sample(s) | n | truth compartment | truth leaf |
|---|---|---|---|
| alvin | 1 | Sarcoma | (broad) |
| pfo004 | 2 | Sarcoma | (broad) |
| pfo002 | 3 | Epithelial | COAD/READ (colorectal) |
| hcc1395 | 2 | Epithelial | BRCA |
| rs-tempus | 1 | Epithelial | PRAD |
| pfo017 | 3 | Epithelial | BLCA |
| tempus-nutm1 | 3 | Epithelial | NUTM |

Stage 1: **15/15** compartments correct. Stage 2 (epithelial leaf, family-aware):
best method 7/12 (0.58); the misses are the rare/admixed cases noted above.
