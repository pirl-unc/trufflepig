# Report language

Clinical decisions and wording have separate owners. Python evaluates supplied
evidence, eligibility, blockers and uncertainty. Named Jinja templates describe
those decisions in complete paragraphs. The emitted paragraph is shared by the
Markdown report and its structured JSON/PDF projection.

`trufflepig.report_language.render_report_paragraph(name, **facts)` loads only
packaged templates from `trufflepig/report_templates/`. `StrictUndefined` makes
missing required facts an error instead of silently omitting an explanation.
Input text is a template value; it is never evaluated as template code. HTML
escaping is performed by the HTML renderer, not this plain-Markdown layer.

Treatment-history interpretation is the first migrated paragraph. Its template
owns the status names, relationship phrases, explanation and source wording.
`treatment_history_context` obtains the shared history match and passes an
explicit action to the template. It does not assemble sentence fragments.

Further migration should follow the same contract: produce one evidence decision,
then use it for selection and presentation. Do not move clinical criteria into
Jinja, create separate summaries that reinterpret the evidence, or use an LLM to
repair contradictory eligibility decisions. HLA, RNA observation state and
clinical evidence requests are the next paragraphs to migrate as their shared
decision contracts are consolidated.

The report should read in a consistent order:

1. The conclusion and the evidence supporting it, including uncertainty in the
   disease call and the specimen context.
2. The therapeutic candidates, each with its evidence basis, current blockers
   and a reference to any information needed to resolve them.
3. One list of missing or conflicting information, with each request naming the
   affected decisions and the accepted structured inputs.
4. Detailed evidence, source records and figures for audit.

Deduplicate information requests by their meaning before rendering. For example,
several therapies may depend on the same clinical MSI/MMR result; their entries
should link to one request rather than each repeat instructions for obtaining it.
A known HLA mismatch is a blocker, not a missing-typing request. An unspecified
mutation is missing allele evidence, not a confirmed drug-specific biomarker.

Use a small number of templates for coherent report sections and repeatable
paragraphs. Introduce a shared macro only when several templates truly reuse the
same presentation. Keep display conditionals such as an optional source citation
in templates; compute clinical status, ranking and request deduplication in
Python. Do not recreate the current sentence-fragment assembly with many tiny
template files.

The treatment-history paragraph migration is implemented. Consolidating other
decisions and replacing the current Markdown-to-JSON parsing are subsequent
steps. The intended main path is evidence → decisions → structured report →
Markdown / JSON / PDF. JSON must retain typed evidence and complete rationale;
it should not recover clinical meaning from generated prose. Format renderers
may change layout, but must share the same decisions and authored paragraphs.

Keep clinical labels and numerical evidence intact. Tests should check whether
important blockers and source facts survive, rather than freeze every adjective.
An optional language editor can later improve phrasing against this fixed
content, with deterministic template output retained as the fallback. It must
preserve decisions, negation, uncertainty, identifiers, values and citations.
