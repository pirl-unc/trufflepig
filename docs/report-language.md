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

Keep clinical labels and numerical evidence intact. Tests should check whether
important blockers and source facts survive, rather than freeze every adjective.
