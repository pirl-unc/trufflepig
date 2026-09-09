# Clinical assay input

Supply an existing clinical MSI/MMR result through the same versioned
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

## Python and report contracts

```python
from trufflepig.clinical_context import load_clinical_context, evaluate_msi_mmr
from trufflepig.main import analyze

context = load_clinical_context("clinical-context.json")
decision = evaluate_msi_mmr(context)
analyze("gene_tpm.tsv", output_dir="output/specimen-A", clinical_context=context)
```

`ClinicalContext`, `ClinicalAssay` and `ClinicalSource` also accept direct Python
construction. `load_clinical_context` accepts these contexts, JSON mappings or
JSON paths and rejects unknown fields. `AnalyzeConfig` validates and captures the
input once; manifests contain the normalized values rather than relying on a
mutable source file. The normal report JSON carries `clinical_context`, assay
evidence on eligibility requirements, and consolidated `evidence_requests` with
priority and retained evidence. Markdown and PDF render the same authored facts.

The web form accepts the same JSON as “Clinical assay results”. It validates
the upload before launching a run. Uploads are limited to 1 MB; a JSON scalar
cannot be interpreted as a server-side file path. No clinical data is sent to an
external model by this input path.

RNA-only operation remains complete without this input. Other clinical facts,
TMB, integration of existing variant/HLA/history inputs into this context and
optional model extraction/editing remain in #163. Existing variant, fusion, HLA
and treatment-history inputs continue to use their current contracts.
