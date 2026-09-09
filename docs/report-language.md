# Report content and rendering

`build_report_content` authors one report from the finalized `ReportView`, curated
therapy panel and supplied evidence. Markdown, JSON and the native-text PDF use
its sections in the same order:

1. **Conclusion and supporting evidence:** disease call, confidence, competing
   evidence, tumor-fraction uncertainty and specimen context.
2. **Therapy rationale and blockers:** ranked candidates with patient and
   population evidence; known exclusions and treatments awaiting eligibility.
3. **Information needed:** one list grouped by evidence requirement, retaining
   every affected therapy and the specific results needed.
4. **Detailed evidence and figures:** supplied history, attribution explanations,
   limitations, links to full tables and the emitted figure manifest.

The main APIs are:

| API | Purpose |
|---|---|
| `brief.recommend_therapies` | Rank candidates using clinical requirements and RNA support |
| `therapy_eligibility.evaluate_therapy_eligibility` | Evaluate history, disease scope, HLA and molecular requirements |
| `therapy_eligibility.evaluate_therapy_review` | Apply clinical and sample-support criteria before ranking; retain the exclusion reason |
| `clinical_context.evaluate_msi_mmr` | Evaluate specimen-scoped clinical assay results and preserve conflicts |
| `reporting.target_rna_observation` | Preserve measured, below-detection, missing and invalid RNA states |
| `report_content.assess_therapy` | Explain one selected or excluded therapy without reranking it |
| `report_content.empty_shortlist_summary` | Explain an empty shortlist from recorded selection decisions |
| `report_content.build_report_content` | Author the report and deduplicate information requests |
| `report_language.render_report_paragraph` | Render a named paragraph from explicit facts |
| `report_language.render_report_template` | Render the complete Markdown layout |
| `report_document.write_report_document` | Serialize authored content, headline and figure provenance |
| `report_pdf.build_interpretive_report_pdf` | Render the serialized report as a searchable PDF |

Python owns matching, status, ranking and request deduplication. Packaged Jinja
`StrictUndefined` templates own paragraph wording and layout. Missing template
facts raise an error. Supplied text is a value; it is never executed as template
code. The PDF escapes HTML and preserves HLA tokens using the same mhcgnomes
nomenclature boundary as other HLA operations.

Each therapy assessment retains a `selection` status, reason and `permits_review`
flag from the same review decision used by ranking. Clinical eligibility remains
separate from RNA observation. Empty-shortlist counts describe actual clinical
exclusions, pending evidence, disease-state context or RNA discovery/support
limits; missing RNA on an expression-independent pathway is not an exclusion.
The `empty_shortlist` template consumes those recorded decisions without another
eligibility evaluation. Counts refer to curated rows, which may share an agent.

Eligibility requirements distinguish `satisfied`, `missing`, `unresolved` and
`blocked`. Prior benefit can support review while an assay remains missing; it
does not confirm that assay or override conflicting molecular evidence. A known HLA mismatch or exclusion remains a blocker; it does not become
a request for new typing. An explicitly contraindicated component blocks its
containing regimen. Requests on an already excluded treatment remain in the audit
assessment without generating a new testing task. Sharing a request does not
merge different treatments or erase different assay requirements. Overlapping
target keys and shared assay questions are merged transitively, so a later
requirement cannot recreate a duplicate request. Clinical-setting questions
use this same collector.

RNA abundance does not establish a mutation. Exact protein requirements on the
covered KRAS/BRAF drugs reject another amino-acid change, imprecise nomenclature,
a nucleotide-only assertion and an unrelated variant file. These rules do not
constitute an exhaustive molecular eligibility database. MSI/MMR/TMB inference
remains context for confirmation. [Clinical MSI/MMR and absolute TMB input](clinical-context.md)
can satisfy the corresponding assay criterion; RNA alone cannot. TMB thresholds,
test identity and specimen material are treatment-specific. VCF/MAF adapters remain tracked in #140/#141. The
current accepted variant input is a normalized table or an explicit symbolic
call; requests can also name the clinical report needed for reconciliation.

Report JSON schema 2 retains complete authored paragraphs, therapy assessments,
requirements, treatment history and figures. It does not recover clinical meaning
from generated Markdown. Older schema-1 reports must be regenerated before using
the new PDF renderer. PDFs use native text, clickable source links, automatic
pagination and packaged Unicode fonts; no fixed line count truncates a rationale.

The detailed analysis and evidence tables retain broader curation and source
attribution. They refer to the summary's consolidated information list. Figures
must describe measured or modeled RNA patterns and their uncertainty; they must
not assert treatment exposure, receptor-assay status or a mutation from expression.

An optional LLM language editor remains a separate feature. It may eventually
improve phrasing over these fixed facts, with review and deterministic output
available. It must preserve decisions, negation, uncertainty, identifiers, values
and citations. Clinical-background extraction and the validated structured input
contract belong to #163; language editing cannot repair an eligibility decision.
