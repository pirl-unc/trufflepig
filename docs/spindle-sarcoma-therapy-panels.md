# Exact therapy panels for spindle-pattern sarcomas

## Why these are separate panels

“Spindle cell” describes morphology, not one disease or one treatment pathway.
Inflammatory myofibroblastic tumor (IMT), dermatofibrosarcoma protuberans
(DFSP), PEComa, GIST, and infantile spindle tumors have different defining
biology and eligibility rules. An exact child diagnosis must therefore never
inherit every therapy assigned to unrelated siblings in the broad sarcoma
table.

Trufflepig uses diagnosis-specific panels and keeps molecular therapies gated
on supplied variant evidence. Target RNA abundance provides context but
does not prove a fusion or create drug eligibility.

## Inflammatory myofibroblastic tumor (`SARC_IMT`)

- **ALK-positive IMT:** crizotinib is FDA-approved for adult and pediatric
  patients age one year or older with unresectable, recurrent, or refractory
  ALK-positive IMT. Trufflepig requires a supplied ALK event before placing it
  on the concise shortlist.
- **ALK-negative IMT:** complete RNA structural-variant testing should cover
  NTRK1/2/3 and other kinase fusions. A verified NTRK fusion can use the
  tumor-agnostic larotrectinib, entrectinib, or age-eligible repotrectinib
  pathways; NTRK expression alone cannot.

[FDA Xalkori label](https://www.accessdata.fda.gov/drugsatfda_docs/label/2023/217581s000lbl.pdf)

## Dermatofibrosarcoma protuberans (`SARC_DFSP`)

Imatinib is FDA-approved for adults with unresectable, recurrent, or metastatic
DFSP. The characteristic COL1A1-PDGFB fusion, or another activating PDGFB
rearrangement, supplies the molecular rationale. Because an RNA-based working
entity call is not equivalent to pathologic confirmation, Trufflepig requires
supplied rearrangement evidence before placing imatinib on the concise
shortlist.

[FDA Gleevec label](https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/021588Orig1s063lbl.pdf)

## Malignant PEComa (`SARC_PEC`)

Nab-sirolimus (Fyarro) is FDA-approved for adults with pathologically confirmed
locally advanced unresectable or metastatic malignant PEComa. This is a
histology-based indication: TSC1, TSC2, and other mTOR-pathway findings are
biologically informative but are not label requirements. The report therefore
does not gate the drug on MTOR RNA abundance or manufacture a mutation
requirement.

[FDA Fyarro approval package](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2022/213312Orig1s000Approv.pdf)

## Reporting contract

- GIST retains its existing dedicated KIT/PDGFRA therapy panel.
- Exact IMT, DFSP, and PEComa calls use only their own panel.
- An unmapped sarcoma child can inherit genuinely parent-wide therapies, but
  not rows labeled for other sarcoma subtypes.
- Every molecularly gated shortlist row requires supplied variant evidence;
  expression alone is never treated as a structural variant.
