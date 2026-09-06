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
