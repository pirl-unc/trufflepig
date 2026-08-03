# Infantile spindle-cell molecular interpretation

## What the report should say first

Infantile fibrosarcoma (IFS), congenital mesoblastic nephroma (CMN), and
NTRK-rearranged spindle-cell neoplasms overlap in morphology and MAPK-pathway
drivers. RNA expression can support a spindle/fibroblastic and MAPK-active
program, but it cannot prove a fusion, a kinase-domain duplication, or the
anatomic site. Trufflepig therefore separates three questions:

1. **Tumor identity:** site, age, morphology, and RNA context establish the
   differential.
2. **Driver confirmation:** an orthogonal RNA/DNA structural-variant assay
   establishes an NTRK fusion, EGFR KDD/ITD, BRAF rearrangement/internal
   deletion, or another kinase event.
3. **Treatment relevance:** a therapy row is shortlisted only when the supplied
   alteration matches its biomarker requirement. High NTRK or EGFR RNA alone is
   context, not eligibility.

## How the entities differ

| Entity | What anchors the diagnosis | Important molecular spectrum |
|---|---|---|
| `SARC_IFS` | Infantile soft-tissue spindle-cell tumor with compatible pathology | ETV6–NTRK3 is common but not universal; EML4–NTRK3, BRAF internal deletions, and unresolved cases occur |
| `CMN` | A kidney primary with compatible nephroma pathology | EGFR kinase-domain ITD/KDD is common, especially in classic CMN; ETV6–NTRK3 is enriched in cellular CMN; rarer NTRK1 and BRAF events occur |
| `SARC_NTRK_SPINDLE` | Compatible spindle-cell pathology plus a confirmed in-frame NTRK1/2/3 fusion | Multiple NTRK fusion partners; expression of an NTRK gene is insufficient |

ETV6–NTRK3 does not by itself distinguish IFS from cellular CMN. EGFR KDD/ITD
strongly raises CMN in an infantile spindle tumor, but it does not manufacture a
kidney primary when site and pathology disagree. The observed frequencies used
in full reports come from oncoref's structured driver-spectrum table, rather
than being copied into selection thresholds.

## Treatment interpretation

### Confirmed NTRK fusion

- **Larotrectinib** is the best-supported first molecular option for IFS. In the
  prospective COG ADVL1823 study, 17 of 18 IFS patients responded within six
  cycles (94%); two-year event-free and overall survival were 82.2% and 93.8%.
  Local control and surgical planning remain part of care, and the FDA label's
  advanced/unresectable or surgery-morbidity criteria still apply.
  [JCO/COG study](https://pubmed.ncbi.nlm.nih.gov/39652801/),
  [FDA label](https://www.accessdata.fda.gov/drugsatfda_docs/label/2025/211710s010lbl.pdf)
- **Entrectinib** is another FDA-approved tumor-agnostic option for confirmed
  NTRK-fusion solid tumors, including patients older than one month; verify the
  label criteria and pediatric formulation.
  [FDA pediatric indication](https://www.fda.gov/drugs/resources-information-approved-drugs/fda-expands-pediatric-indication-entrectinib-and-approves-new-pellet-formulation)
- **Repotrectinib** is FDA-approved from age 12 for NTRK-fusion solid tumors and
  has activity in both TKI-naive and TKI-pretreated disease. The pediatric and
  young-adult study NCT04094610 is the most directly relevant recruiting trial
  found in the 2026-08-02 review.
  [FDA approval](https://www.fda.gov/drugs/resources-information-approved-drugs/fda-grants-accelerated-approval-repotrectinib-adult-and-pediatric-patients-ntrk-gene-fusion-positive),
  [NCT04094610](https://clinicaltrials.gov/study/NCT04094610)
- At progression on a first-generation TRK inhibitor, repeat molecular profiling
  should look for on-target resistance before choosing another TRK inhibitor.
  Selitrectinib remains an investigational resistance-directed strategy.
  [NCT03215511](https://clinicaltrials.gov/study/NCT03215511)

The IFS neoadjuvant trial [NCT03834961](https://clinicaltrials.gov/study/NCT03834961)
and the pediatric SCOUT study [NCT02637687](https://clinicaltrials.gov/study/NCT02637687)
were active but not recruiting in the same review. Trufflepig names trial IDs as
evidence/context; it does not claim a site is recruiting without a current trial
lookup.

### Confirmed EGFR KDD/ITD

EGFR is the primary target to review. **Afatinib** has the clearest published
KDD response precedent and is the lead drug for a rare-tumor molecular-board
discussion. **Osimertinib** has secondary cross-tumor case evidence, including a
reported response after afatinib, but neither is an established CMN or spindle
sarcoma standard. The small CMN literature includes a relapsed case with stable
disease on afatinib plus PD-1 blockade; that is not enough evidence to recommend
the combination. Trufflepig consequently emits an *EGFR TKI review (afatinib;
osimertinib secondary)* as off-label/case-level, not an approved recommendation.

- [CMN genomic/pathology series](https://pubmed.ncbi.nlm.nih.gov/32590884/)
- [Recurrent EGFR KDD in NTRK3-negative CMN](https://pubmed.ncbi.nlm.nih.gov/32490123/)
- [Recent small CMN treatment series](https://pubmed.ncbi.nlm.nih.gov/41810180/)
- [Afatinib then osimertinib KDD case](https://pubmed.ncbi.nlm.nih.gov/38821532/)
- [Afatinib KDD response and resistance case](https://pubmed.ncbi.nlm.nih.gov/40821607/)

### Other or unresolved drivers

BRAF internal deletions/rearrangements and other kinase fusions should trigger
complete molecular review and a genotype-matched basket-trial search. They do
not currently generate a named default drug in Trufflepig. A molecularly
unresolved case should prompt broader structural-variant testing rather than an
RNA-expression-based rescue therapy.

## Implementation contract

- The full therapy landscape shows the alteration-gated possibilities and their
  missing evidence.
- The concise shortlist shows one only when a supplied alteration matches the
  target and alteration class.
- Fusion matching checks both partners, so `ETV6-NTRK3 fusion` correctly gates
  an NTRK3 therapy row even when the input parser's primary label is ETV6.
- Exact IFS/CMN/NTRK-spindle reports receive their molecular panel without
  inheriting unrelated sarcoma-subtype therapies.
- A broad `SARC` plus EGFR KDD retains the broad diagnosis, raises the CMN/IFS
  differential, and uses the existing EGFR-KDD sarcoma row. It does not relabel
  the sample as CMN without renal-site/pathology evidence.
- Oncoref 1.8.172 or newer supplies the observed driver-spectrum table. Older
  supported installations retain the conservative narrative but omit the
  frequency table rather than guessing.
