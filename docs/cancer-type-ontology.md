# Cancer-type reasoning ontology — design + inference ledger

`trufflepig/cancer_type_ontology.py` is an **interpretable reasoning layer** over
the production signature engine
(`trufflepig.tumor_purity.rank_cancer_type_candidates`). It refines a sample's
cancer-type call only as far as the evidence supports, and otherwise reports the
near-tied neighbours as a candidate set (e.g. *COAD or READ*).

This document is the **inference ledger**: the reasoning each decision is built
from, and the running ground-truth check, so we can see whether changes drift
relative to the (small) set of samples we actually know the answer for.

## Why a layer, not a new scorer

A repeatedly-confirmed lesson: scoring lineage from **within-sample marker
percentiles fails on real patients**. A marker's rank inside one sample's
transcriptome is contaminated by

- tumour-microenvironment infiltrate — a carcinoma with brisk TILs scores high
  on the hematolymphoid panel (PTPRC/CD3E), so a naive heme gate mis-fires
  (observed: pfo002, pfo019, pfo017 all mis-called "heme"); and
- the long tail of near-zero genes — any expressed keratin lands at ~p0.6 in a
  sarcoma, so an "epithelial present" floor leaks (observed: 93/93 sarcoma reps
  mis-routed).

The production engine avoids this with cross-cohort signature panels (rank the
sample's value within the 33-cohort reference distribution), purity anchoring and
family-factor correction. It calls **all 11 local clinical samples correctly**.
So we *consume* its per-cohort scores rather than re-deriving lineage. The
ontology adds the structure the flat engine lacks: exclusion gating, triggered
organ follow-up, abstention, and a readable trace.

## The ontology

Each scored cohort occupies a path

```
root → broad lineage → (carcinoma) differentiation → organ-of-origin → cohort
```

A node's score is the best (max) evidence among its descendant cohorts. The walk
descends, and at each node either **commits** to the leading child (its lead over
the runner-up clears the level's margin) or **stops** and returns every child
within that margin as the candidate set.

Per-level margins (`DEFAULT_MARGINS`) widen with the breadth of the split — a
wrong broad-lineage call is the most expensive, COAD-vs-READ the least:

| level | margin | rationale |
|-------|--------|-----------|
| broad | 0.06 | epithelial ⟂ mesenchymal/heme; a wrong call is costly |
| differentiation | 0.05 | glandular/squamous/urothelial/hepatocellular/mesothelial |
| organ | 0.05 | GI/lung/breast/prostate/kidney/thyroid/panc-bil/gyne/… |
| leaf | 0.04 | sibling cohorts (COAD/READ) are *meant* to tie and both surface |

The signals are **RNA transcript-abundance signatures** — weakly correlated with
protein IHC, **not** staining. The output complements, it does not replace,
histopathology.

## Coverage

`ontology_path()` places **all 145** pirlygenes registry codes on the tree
(broad lineage at minimum):

| broad node | # codes | notes |
|------------|--------:|-------|
| epithelial | 51 | carcinoma-* + salivary/thymic/endocrine; NUTM/ADCC/ACINIC/NPC/THYM/MESO |
| mesenchymal | 41 | all sarcoma subtypes |
| hematolymphoid | 21 | heme-bcell/tcell/myeloid/plasma |
| neuroendocrine | 14 | |
| neural | 11 | cns |
| embryonal | 4 | poly-phenotypic pediatric; resolved by context + defining alteration |
| melanocytic | 2 | SKCM/UVM |
| germ | 1 | TGCT |

### Lineage-marker recall for no-reference entities

The cross-cohort screen can only propose types with a TCGA reference cohort. Whole
neuroendocrine entities (SCLC, NETs, Merkel, neuroblastoma) have none — only PCPG
is a TCGA cohort — so the screen scatters them across LUAD/DLBC and they never
reach the walk. [`lineage_marker_recall.py`](../trufflepig/lineage_marker_recall.py)
adds them back using **tumour-intrinsic** markers: an absolute HK-ratio presence
test on the NE program (CHGA/CHGB/INSM1/ASCL1/SYP/SCG2), gated by an obligate core
granin/INSM1 and a 2-marker minimum. It's safe because these markers are absent
from stroma/immune background (100–1000× separation) — unlike the TME-contaminated
heme/epithelial panels. `classify_cancer_type_ontology` auto-runs it, injects the
proposed entities into the score map (additively — screen candidates are never
lowered), and the NE branch then competes in the walk. Validated against pirlygenes' expanded representative cohorts (73 → 109, which
now include the neuroendocrine family): the recall fires on **10/11** NE reps
(SCLC, NET_PANCREAS/MIDGUT/RECTAL, NEC_LUNG_LARGECELL, NBL ±MYCN, PCPG, MTC) and
stays silent on **22/22** new heme / sarcoma / embryonal reps (perfect
specificity) and all 11 local clinical samples. A single dominant core granin
(≥2.0×HK) fires on its own (catches well-differentiated NETs like NET_MIDGUT that
present one granin); NET_LUNG is the one honest miss (that rep's NE markers are
all <0.15×HK — atypical / low-purity).

Cohorts with a TCGA reference signature (~33) refine to a leaf; rarer codes are
scored by their nearest reference neighbours and the walk **honestly stops higher
up** rather than guessing. Examples from the rep sweep:

- **ATRT** (no TCGA cohort) → neighbours `{GBM, LGG}` → broad lineage *neural* ✓,
  subtype abstained (correct: ATRT is decided by SMARCB1 loss, not lineage).
- **NUTM** → `LUSC` → epithelial/squamous ✓; NUT carcinoma *is* squamous by
  expression and is distinguished from LUSC only by the NUTM1 fusion — a
  defining-alteration flag, not a lineage call.

## Holistic integration: signature + recall + lineage exclusion

`classify_cancer_type_ontology(df)` integrates three tumour-intrinsic evidence
channels, then walks the ontology:

1. **Signature backbone** — the production cross-cohort screen (per-cohort
   `support_geomean`). Validated, carries the clean cases.
2. **Lineage-marker recall** ([`lineage_marker_recall.py`](../trufflepig/lineage_marker_recall.py))
   — adds no-reference NE entities the screen can't propose (additive).
3. **Epithelial-exclusion gate** ([`lineage_evidence.py`](../trufflepig/lineage_evidence.py))
   — when the tumour-intrinsic epithelial program (EPCAM/keratins) is present,
   down-weights the mesenchymal & hematolymphoid branches, **confidence-
   proportional** to the epithelial signal's **multi-view confidence**
   ([`signal_views.py`](../trufflepig/signal_views.py)). This is the admixture /
   stromal-confound fix: a carcinoma with heavy stroma scores a mesenchymal cohort
   high off the *stromal* program, but its tumour cells are epithelial, so the gate
   demotes the spurious sarcoma. Real sarcomas (epithelial-absent) are untouched.

### Multi-view signal normalization

Every signal is expressed under **five normalizations**, each answering a different
question, and the gate reasons over their concordance ([`signal_views.py`](../trufflepig/signal_views.py)):

| view | question |
|------|----------|
| `hk` | how high vs this sample's housekeeping baseline (scale-invariant) |
| `within_pct` | how dominant within this transcriptome (purity/dominance) |
| `log1p` | absolute expression, compressed — **tightest within-class separator** |
| `cohort_pct` | how high vs other cancer types (specificity) |
| `cohort_z` | how many SDs above the cross-cohort background |

No single one is best: an empirical gate sweep on 334 reps + 18 locals gave
within-carcinoma CV of **1.02 (HK) vs 0.29 (log1p)** and carcinoma/sarcoma
separation **0.45 (HK) vs 1.46 (log1p)** — so the gate is driven by the log1p-led
multi-view confidence, not a single HK threshold. Concordance across views = the
call's confidence; *which* views disagree is itself diagnostic and is surfaced in
the trace (`high cohort_z + low within_pct ⇒ admixture / low purity`;
`high within_pct + low cohort_pct ⇒ dominant but not cohort-specific`). The full
five-view fingerprint + flags appear on the `[exclude]` trace lines.

All marker gates (recall + exclusion) normalise against a **ribosomal-free
housekeeping median** (`lineage_marker_recall.marker_hk_median`). The default
30-gene housekeeping set has 7 ribosomal-protein genes (RPL*/RPS*) which are
part of pirlygenes v4 clean-TPM's *pinned* ribosomal compartment — including them
couples the HK median to the normalization (~2.4× inflation on v4-clean input vs
~1.4× on legacy), shifting every HK-ratio threshold. Dropping them makes the gate
denominator invariant to the clean-TPM handling; thresholds then transfer across
normalizations (recalibrated: NE core ≥0.3, epithelial ≥0.25 — both in wide gaps).

This is what the deconvolution-residual line was reaching for and failed to
achieve (the decomposition has no stromal compartment — see
[cancer-type-residual-matching-findings.md](./cancer-type-residual-matching-findings.md)).
Marker exclusion does it cleanly: measured epithelial HK-ratio is 0.22–5.1× in
carcinomas vs 0.01–0.07× in real sarcomas.

With all three channels, the local clinical set reaches **11/11 broad and 11/11
leaf-in-candidates** (vs 11/11 broad, 10/11 leaf for the signature-only walk) —
the pfo002-washu stromal-SARC confound now resolves cleanly to READ.

## Ground-truth ledger — 11 local clinical samples

Truth labels are the curated local set (cegat = PRAD/Kat-DL, pfo004 = osteosarcoma,
asy = CRC, pfo019 = NUT carcinoma, pfo017 = bladder ± liver met, hcc1395 = BRCA
EMT cell line). "leaf-in-candidates" counts a hit when the truth cohort is in the
returned candidate set (COADREAD := {COAD, READ}).

| sample | truth | result | broad | leaf | reasoning |
|--------|-------|--------|:----:|:----:|-----------|
| tempus | PRAD | PRAD | ✓ | ✓ | epithelial→glandular→prostate→PRAD, clean descent |
| cegat | PRAD | stop @broad {epithelial, mesenchymal}; PRAD in set | ✓ | ✓ | low-purity/EMT prostate — honest abstention |
| alvin | SARC | SARC | ✓ | ✓ | mesenchymal leads, single sarcoma leaf |
| asy | COAD/READ | READ (COAD runner-up 0.46) | ✓ | ✓ | GI lead 0.24; near-neighbour surfaced |
| hcc1395 | BRCA | stop @diff {glandular, squamous}; BRCA in set | ✓ | ✓ | EMT breast line — genuinely ambiguous |
| pfo004 | SARC | SARC | ✓ | ✓ | mesenchymal leads |
| pfo002 | COAD/READ | READ (COAD runner-up 0.52) | ✓ | ✓ | strong glandular→GI |
| pfo019 | NUTM | LUSC | ✓ | (fusion) | squamous-lung by expression; NUTM1 fusion decides |
| pfo017-bl23 | BLCA | BLCA | ✓ | ✓ | epithelial 0.75→urothelial→bladder |
| pfo017-bl25 | BLCA | BLCA | ✓ | ✓ | urothelial leads |
| pfo017-li23 | BLCA | BLCA | ✓ | ✓ | urothelial, liver met still reads bladder |

**Broad lineage: 11/11. Leaf-in-candidates: 10/11** (the one is NUT carcinoma,
which expression legitimately reads as squamous — see coverage note above).

The two early stops are the intended behaviour: cegat and hcc1395 are genuinely
ambiguous at the level they stop at, and in both cases the true cohort is present
in the surfaced candidate set.

## What this does NOT do (kept honest)

- It does not call subtypes that lack a TCGA reference signature; it stops at the
  broad/differentiation level and names the near neighbours.
- It does not use defining alterations (NUTM1, SMARCB1, fusion surrogates) — those
  remain a separate evidence line that the report integrates *alongside* this.
- It is not IHC; RNA lineage signatures are weakly correlated with protein stains.
