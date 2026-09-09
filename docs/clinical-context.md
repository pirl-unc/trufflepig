# Clinical and specimen input

Supply existing clinical MSI/MMR or absolute TMB results through the same versioned
`ClinicalContext` contract in Python, the CLI or the web upload form:

```sh
trufflepig run --sample gene_tpm.tsv --workspace output/specimen-A \
  --clinical-context clinical-context.json
```

```json
{
  "schema_version": 1,
    "specimen_id": "specimen-A",
  "assays": [
    {
      "kind": "msi",
      "result": "MSI-H",
      "method": "PCR",
      "specimen_id": "specimen-A",
      "scope": "current",
      "collected_at": "2026-07-29",
      "reported_at": "2026-08-01",
      "validity": "validated",
      "reportability": "reportable",
      "source": {
        "title": "Molecular pathology report",
        "reference": "laboratory-accession-123",
        "review_status": "supplied"
      }
    }
  ]
}
```

This is a synthetic example. Supply the actual result and its quality fields;
the software does not validate an assay merely because these fields are present.

`specimen_id` at the top level explicitly associates the clinical input with the
specimen being analyzed. Each assay retains its own specimen ID and scope.
Different IDs, historical results and unknown scope remain in the report with
their limitations. The software cannot establish cross-assay specimen identity
from filenames or expression similarity. An old collection date alone does not
make a result historical: an older assay on the same archived specimen can still
have `scope: "current"`. Dates are optional ISO dates; unknown dates stay unknown.

| Field | Accepted values or meaning |
|---|---|
| `kind` | `msi` or `mmr` |
| MSI `result` | `MSI-H`, `MSI-L`, `MSS`, `indeterminate`, `pending`, `not_tested`, `unknown` |
| MMR `result` | `dMMR`, `pMMR`, `indeterminate`, `pending`, `not_tested`, `unknown` |
| `method` | Clinical MSI: `PCR` or `NGS`; clinical MMR: `IHC` or `NGS`. Other/unknown methods are retained as unresolved. RNA expression and RNA-read evidence cannot satisfy this gate. |
| `scope` | `current`, `historical`, `other_specimen`, `unknown` |
| `validity` | `validated`, `failed`, `unverified`, `unknown` |
| `reportability` | `reportable`, `unreportable`, `unknown` |
| `source` | `title`, `reference`, optional verbatim `excerpt`, and `review_status` |
| `source.review_status` | `supplied`, `confirmed`, `proposed`, `rejected`. Proposed/rejected assertions cannot satisfy a gate. |
| `protein_results` | Optional MMR mapping for `MLH1`, `MSH2`, `MSH6`, `PMS2`: `retained`, `lost`, `equivocal`, `not_tested`, `unknown` |
| `id` | Optional stable assertion ID; otherwise generated from its content and provenance |

MSI and MMR remain separate assay records. Validated, reportable MSI-H or dMMR
evidence for the bound specimen can satisfy the MSI-H/dMMR biomarker requirement.
MSS, MSI-L and pMMR do not satisfy that indication. Conflicting current results
require reconciliation; later dates do not silently supersede a conflicting
report. Failed, historical and unreviewed records do not override a usable current
result. Per-protein findings are retained; the software does not derive an overall
dMMR result from a partial protein panel, and contradictory overall/protein
findings remain unresolved.

The supplied result satisfies only the biomarker requirement. Diagnosis,
disease setting, prior therapy, contraindications and other treatment criteria
still apply. For example, the [KEYTRUDA label, section 1.9](https://www.accessdata.fda.gov/drugsatfda_docs/label/2026/125514s194lbl.pdf)
specifies unresectable or metastatic MSI-H/dMMR colorectal cancer identified by
an FDA-authorized test. The report retains the remaining clinical-setting review.

The MMR expression proxy remains RNA context. An MSI-like proxy prioritizes an
unresolved clinical-assay request without opening eligibility. A usable clinical
result resolves that request. If the clinical result and RNA proxy differ, the
report shows the disagreement and retains the clinical decision.

The RNA paragraph preserves measured MLH1 bulk TPM and its cohort comparison
without assigning protein retention, promoter methylation or a mutation
mechanism. POLE-associated hypermutation can occur in microsatellite-stable
tumors ([primary sequencing study](https://pubmed.ncbi.nlm.nih.gov/31857678/));
a POLE variant, absolute TMB and clinical MSI/MMR are distinct evidence.
Supplying a POLE variant alone does not satisfy the MSI-H/dMMR requirement.

## Python and report contracts

```python
from trufflepig.clinical_context import load_clinical_context, evaluate_msi_mmr
from trufflepig.main import analyze

context = load_clinical_context("clinical-context.json")
decision = evaluate_msi_mmr(context)
analyze("gene_tpm.tsv", output_dir="output/specimen-A", clinical_context=context)
```

`ClinicalContext`, `ClinicalAssay`, `ClinicalMeasurement` and `ClinicalSource` also accept direct Python
construction. `load_clinical_context` accepts these contexts, JSON mappings or
JSON paths and rejects unknown fields. `AnalyzeConfig` validates and captures the
input once; manifests contain the normalized values rather than relying on a
mutable source file. The normal report JSON carries `clinical_context`, assay
evidence on eligibility requirements, and consolidated `evidence_requests` with
priority and retained evidence. Markdown and PDF render the same authored facts.

The web form accepts the same JSON as “Clinical assay results and specimen context”. It validates
the upload before launching a run. Uploads are limited to 1 MB; a JSON scalar
cannot be interpreted as a server-side file path. No clinical data is sent to an
external model by this input path.

RNA-only operation remains complete without this input. Other clinical facts,
integration of existing variant/HLA/history inputs into this context and
optional model extraction/editing remain in #163. Existing variant, fusion, HLA
and treatment-history inputs continue to use their current contracts.

## Absolute TMB measurements

TMB uses the same `assays` array, source and specimen rules:

```json
{
  "schema_version": 1,
  "specimen_id": "specimen-A",
  "assays": [{
    "kind": "tmb",
    "result": "measured",
    "measurement": {"value": 18, "unit": "mut/Mb"},
    "method": "NGS",
    "test_id": "FDA:P170019",
    "specimen_type": "tissue",
    "specimen_id": "specimen-A",
    "scope": "current",
    "reported_at": "2026-08-01",
    "validity": "validated",
    "reportability": "reportable",
    "source": {"title": "Synthetic molecular pathology report"}
  }]
}
```

`measurement.value` is a finite nonnegative number or null. Zero is measured
zero; null is missing. The result states are `measured`, `indeterminate`,
`pending`, `not_tested` and `unknown`. A high/low designation depends on the
treatment criterion and is not inferred from a qualitative claim. Invalid
numbers are rejected. Unsupported units, missing measurements, proposed
assertions, historical results and quality limitations remain visible without
satisfying a requirement. `mutations/Mb` and `mutations/megabase` normalize to
`mut/Mb`; percentiles and raw mutation counts are never converted to TMB.

`TmbCriterion` records a treatment's minimum, units, specimen material, accepted
test identifiers and source. `evaluate_tmb(context, criterion=...)` evaluates
only TMB assays, while `evaluate_msi_mmr` evaluates only MSI/MMR assays.
`clinical_assay_records(context)` preserves all assay kinds for detailed reporting.
Concordant usable results resolve the threshold comparison; usable results on
opposite sides stay conflicting. Older, failed or unreviewed assertions do not
override a usable current assay. Existing MSI/MMR record shapes and generated
IDs are preserved when the optional measurement/test/material fields are absent.

The public `tmb_criterion_for_therapy(row)` and `tmb_requirement(row, analysis)`
APIs use one decision for recommendation eligibility, rationale and the
deduplicated information list. A curated row may carry an explicit
`tmb_criterion` mapping. Otherwise, the covered pembrolizumab TMB-specific path
uses the [FDA FoundationOne CDx criterion](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpma/pma.cfm?id=P170019S016):
at least 10 mut/Mb from tissue using test ID `FDA:P170019`. This is the tissue
test, not FoundationOne Liquid CDx. The
[FDA companion-diagnostic table](https://www.fda.gov/medical-devices/in-vitro-diagnostics/list-fda-authorized-companion-diagnostic-devices-in-vitro-and-imaging-tools)
is the reference for test/biomarker/therapy scope. Uncurated treatments, combined
regimens or missing criterion metadata remain unresolved instead of borrowing
that threshold. This input does not create a new therapy row outside the report's
curated panel.

The [FDA TMB indication](https://www.fda.gov/drugs/drug-approvals-and-databases/fda-approves-pembrolizumab-adults-and-children-tmb-h-solid-tumors)
also specifies unresectable/metastatic disease, prior progression and no
satisfactory alternatives. Satisfying the assay requirement does not establish
those conditions; the separate clinical-setting request remains. Negative or
conflicting results and supplied treatment contraindications retain their
blockers. RNA scores, POLE variants and cohort TMB reference values cannot
substitute for the specimen's absolute clinical measurement.

## Specimen identity and purpose

The same context accepts optional `specimen` metadata for its top-level
`specimen_id`. This does not select a row or column from the expression table:
keep using `--sample-id-col` and `--sample-id-value` for that purpose.

```json
{
  "schema_version": 1,
  "specimen_id": "reference-001",
  "specimen": {
    "display_label": "Reference culture α — replicate 1",
    "purpose": "research",
    "cell_line_id": "synthetic-reference-line",
    "collected_at": "2026-07-29",
    "site": "supplied tissue of origin",
    "source": {
      "title": "Research sample manifest",
      "reference": "manifest-row-1",
      "review_status": "supplied"
    }
  }
}
```

`purpose` is `clinical`, `research` or `unknown` (the default). Filenames, cell-line
labels, RNA similarities and missing clinical metadata do not establish purpose.
Proposed or rejected source assertions remain in the context; their purpose is
not applied until a supplied or confirmed assertion replaces them. Collection
date is optional and must use `YYYY-MM-DD`; site and dates are never inferred.
Python callers can construct `SpecimenMetadata` directly.

Research reports prominently identify their purpose and describe therapy
candidates as research hypotheses and reference assay context. Molecular facts,
curated clinical requirements, assay status and ranked candidate identities stay
the same. The information section retains characterization questions and omits
the personal treatment-fitness request. Clinical treatment-setting criteria stay
in each assessment's curation for interpreting the reference pathway.

Display labels and specimen identity are preserved in the normal report JSON,
Markdown and PDF. The opening states the specimen context concisely; complete
metadata sources and excerpts remain in the detailed evidence section. The public `report_identity` API keeps the original source path
and sample selector separate from the display title. Each document has a stable
`RPT-…` identifier based on its source, selector and output prefix. The PDF footer
uses this concise identifier on every page; the complete Unicode title wraps at
the start. When no explicit identity is supplied, the title includes that
document identifier, so files with the same basename remain distinguishable.
This identifier is not a verified biological specimen identity.
