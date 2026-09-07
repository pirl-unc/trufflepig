# Treatment history input

Single-sample RNA can nominate targets and therapies, but it cannot determine
what a patient previously received or how they responded. Supply that context
with `--treatment-history` so the recommendation flow does not rank an RNA
inference above an observed patient outcome.

CSV or TSV example:

```tsv
therapy\ttarget\tmodality\tstatus\tnote\tsource
FAP-targeted radioligand therapy\tFAP\tRLT\tmajor_benefit\tVery effective\tclinical history
doxorubicin\t\tsmall_molecule\tprogression\tProgressed after six cycles\tclinical history
```

JSON uses the same fields, either as a list or under a `treatments` key:

```json
{
  "treatments": [
    {
      "therapy": "FAP-targeted radioligand therapy",
      "target": "FAP",
      "modality": "RLT",
      "status": "major_benefit",
      "note": "Very effective",
      "source": "clinical history"
    }
  ]
}
```

Allowed status values are `major_benefit`, `benefit`, `stable_disease`,
`current`, `no_benefit`, `progression`, `intolerance`, and `contraindicated`.
Plain-language equivalents such as `very effective`, `partial response`, and
`progressed` are normalized to those values.

Provide a therapy name whenever possible. A target without a therapy also
requires a modality so that, for example, an outcome from one FAP radioligand
does not silently apply to every FAP-directed modality. Negative outcomes apply
only to the named agent, or to the explicitly supplied target and modality when
no agent is named.
Registered aliases (for example `FAP-2286` and `177Lu-FAP-2286`) refer to the
same treatment. Canonical names, registered brands, and display labels such as
`afami-cel (Tecelra)` resolve to the same agent when both names identify that
registered treatment. Unknown parenthetical annotations and drug combinations
remain distinct. Target aliases such as `MAGE-A4` and `MAGEA4` share the same
gene identity for history matching and supplementation. A different drug
against the same target does not inherit a named negative outcome or
current-treatment status; a target-only record intentionally covers all agents
in its specified modality.

The report uses this evidence as follows:

- Prior patient benefit takes precedence over the RNA model's estimate of
  whether the target signal came from tumor or background.
- Prior progression, no benefit, intolerance, or a contraindication removes
  the matching treatment from the concise shortlist while retaining it in the
  detailed audit table.
- Current treatment is reported for medication reconciliation and is not
  presented as a new start.
- Current disease state, resistance, toxicity, organ function, dosing, and
  indication-specific eligibility still require clinical review.

Population-level benefit and toxicity facts come separately from oncoref and
are joined only when the agent and disease context match. The software never
infers benefit or toxicity from RNA expression.
Subtype-specific outcomes require a matching resolved subtype; an unresolved
parent diagnosis does not inherit every child's outcomes. Direct child codes
use the same evidence scope as their parent/subtype representation.

Every run writes an `*-interpretive-report.pdf` from the finalized report
document, including runs with `--no-figures`. It preserves complete shortlisted
treatment rationales, clinical requirements, evidence sources, and supplied
history. The Markdown section **Clinical evidence to reconcile** identifies
missing clinical inputs for other curated paths without treating them as
eligible recommendations.
The production run emits this single reader PDF; technical review uses the
individual PNGs and evidence tables rather than duplicate PDF bundles.

## Python recommendation API

`trufflepig.brief.recommend_therapies` is the selection API used by the production
summary. Pass a curated panel, the sample's expression-range DataFrame, and
the same patient context used to resolve the panel:

```python
from trufflepig.brief import recommend_therapies
from trufflepig.reporting import cancer_therapy_panel_for_analysis

analysis = {
    "cancer_type": "SARC_SYN",
    "analysis_constraints": {"hla_types": ["A*02:01"]},
    "treatment_history": [
        {"therapy": "afami-cel", "status": "contraindicated"},
    ],
}
_, panel_subtype, panel = cancer_therapy_panel_for_analysis(
    analysis["cancer_type"], analysis,
)
recommendations = recommend_therapies(
    panel, ranges_df, analysis=analysis, panel_subtype=panel_subtype, limit=3,
)
for recommendation in recommendations:
    print(recommendation.therapy["agent"], recommendation.expression)
```

`ranges_df` is the expression-range table from the sample analysis. Each result
is a `TherapyRecommendation` with a `therapy` record and an `expression` record
(or `None` when target RNA is unmeasured). An empty DataFrame allows review of
expression-independent therapies. The API returns at most one row per canonical
target, or per named agent for agent-only therapies, and does not modify input
frames. `limit=0` returns an empty list; negative limits raise `ValueError`.
The example excludes afami-cel even with supported MAGEA4 RNA and matching HLA;
other agents retain their own eligibility and history checks.
