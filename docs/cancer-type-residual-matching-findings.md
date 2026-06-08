# Findings: deconvolution-residual matching does not resolve cancer type

A negative result, recorded so the line isn't re-explored from scratch. Companion
to [cancer-type-ontology.md](./cancer-type-ontology.md).

## The idea (and why it was appealing)

The signature screen can be fooled by admixture: a colorectal sample with heavy
stroma (pfo002-washu) scores SARC on its raw bulk because the *stromal*
mesenchymal program matches the (also-stromal) SARC reference. The proposal was:

1. **screen** → top-K candidate hypotheses,
2. estimate **purity**,
3. **resolve** by deconvolution once purity is known — subtract the TME and match
   the tumour *residual* (`gene_attribution['tumor']` = observed − TME) against
   the references, so the stroma cancels.

Two refinements were tested: (a) decompose the **references too**, so it's
tumour-residual vs tumour-residual (stroma cancelled on *both* sides), and
(b) **sweep the normalization** of the residual vectors.

## What we tested

- 16–20 TCGA reference cohorts decomposed to tumour residuals (cached).
- 5 samples with known truth: pfo002-washu (CRC), asy (CRC), hcc1395 (BRCA-EMT),
  alvin (SARC), tempus (PRAD).
- 4 matching methods × 5 normalizations:
  - methods: plain cosine, per-gene centered (deviation from the average tumour),
    centered + top-1000 / top-3000 highest-cross-cohort-variance genes.
  - normalizations: raw, log1p, sqrt, percentile-rank, z-score.

## Result: it does not work, under any configuration

| sample | truth | resolved? | only by |
|--------|-------|:---------:|---------|
| pfo002-washu | CRC | ✗ never | (always BRCA / ESCA / STAD) |
| hcc1395 | BRCA | ✓ | cosine·log1p/sqrt/zscore; var3000·zscore |
| alvin | SARC | ✓ | centered_var1000·zscore *only* |
| tempus | PRAD | ✓ | centered·raw |
| asy | CRC | ✓ | cosine·pct; centered_var·pct |

**No single (method, normalization) reaches 2/5.** The production cross-cohort
percentile screen on *raw bulk* gets **5/5** on these same samples — so residual
matching **loses information** and actively breaks controls the screen gets for
free (alvin → LGG/BRCA, tempus → BRCA/STAD).

The case the idea was meant to fix — pfo002-washu's stromal-SARC confound — is
**not** fixed: the residual matches BRCA/ESCA, never COAD/READ.

## Why

- **Plain cosine collapses to the most central reference (BRCA).** Whole-profile
  similarity is dominated by the shared proliferation / tumour-core program every
  residual carries, so everything looks most like the "average" tumour.
- **Centering removes that bias but then decomposition *noise* dominates** — the
  residual carries reconstruction error in every gene, so centered cosine scatters
  to STAD/LGG/PRAD.
- **Normalization matters a lot but no choice rescues it.** The same residual pair
  flips cohort with normalization (tempus = PRAD under raw, STAD under sqrt, LGG
  under pct), confirming identification is normalization-dependent — but there is
  no universal winner because the limiter is the *matching*, not the scale.

The production screen wins because it does **not** do whole-profile similarity: it
ranks the sample within the cross-cohort distribution on **curated TME-low
signature panels**. That feature selection is exactly what cosine throws away.

## What did move the needle

Tumour-**intrinsic lineage-marker panels** (NE: CHGA/SYP/INSM1/CHGB separate
neuroendocrine tumours from non-NE by 100–1000×, clean because these genes are
absent from stroma/immune background — unlike the TME-contaminated heme/epithelial
panels). That is the lever for the no-reference entities (SCLC/NET/Merkel/NBL),
implemented as the lineage-marker recall layer
([lineage_marker_recall.py](../trufflepig/lineage_marker_recall.py)), **not**
residual cosine.

## Status

Residual-matching line **explored and parked.** If ever revisited, the only
variant with a chance is applying the production cross-cohort-percentile machinery
to the residual (rank residual-within-reference-residuals on curated panels)
rather than cosine — but given the raw-bulk screen already hits 5/5 here, the
expected upside is small.
