# HLA inputs and therapy requirements

Supply co-present human class-I alleles through `--hla-types`, separated by
commas, or through `AnalyzeConfig.hla_types` in Python. Examples:

```sh
trufflepig run --sample expression.tsv --workspace report \
  --hla-types 'HLA-A*02:01:01:01,A*24:02,B*07:02'
```

Trufflepig uses **mhcgnomes** to parse and format HLA names. It retains all
supplied allele fields and annotations, including P/G group notation and
expression suffixes. Analysis constraints and parameter provenance retain the
original input in `hla_types_raw` alongside canonical `hla_types`.
Invalid or ambiguous inputs raise an input diagnostic; they are never silently
dropped. Use commas for co-present alleles. A slash
ambiguity such as `A*02:01/A*02:05` does not establish which alleles are present
and must be resolved before submitting it as a genotype.

Nomenclature normalization does not validate the typing assay or establish
that a syntactically valid allele exists in a particular nomenclature release.
Keep the original clinical typing report and assay/specimen provenance.

## Public decision API

```python
from trufflepig.hla import evaluate_hla_eligibility

decision = evaluate_hla_eligibility(
    ['A*02:01', 'A*02:05'],
    ['A*02:01P', 'A*02:02P', 'A*02:03P', 'A*02:06P'],
    excluded=['A*02:05P'],
)
assert decision.status == 'excluded'
evidence = decision.public_dict()
```

The result carries supplied typing, required and excluded groups, the decisive
match, an explanation, and the nomenclature-reference version. The public
`target_hla_eligibility` adapter in `trufflepig.reporting` obtains the therapy's
registered requirements and attaches the clinical source and review date.
Shortlist and report prose use that same decision.

Possible states are `not_hla_restricted`, `matched`, `excluded`, `mismatched`,
`insufficient_resolution`, and `unknown`. HLA-restricted therapies require a
match before entering the shortlist. Unknown or unresolved typing remains a
clinical evidence requirement. An exclusion takes precedence over a positive
match on another supplied allele. A low-resolution type cannot clear an
exclusion it might contain.

Null, secreted, or cytoplasmic annotations do not satisfy an ordinary surface-HLA
requirement. They remain visible in the supplied typing. Other unresolved
annotations and G-group membership are not erased to force a match. A known
match on another allele may satisfy a positive requirement only after any
applicable exclusion has been resolved.

## Registered clinical requirements

The agent registry has `hla_allowed`, `hla_excluded`, `hla_source`, and
`hla_reviewed_at` fields. Generic, brand, and registered development names use
the same policy. Registered requirements take precedence over broad shorthand
in the curated display text.

Afamitresgene autoleucel permits A*02:01P, A*02:02P, A*02:03P, and A*02:06P,
and excludes A*02:05P. The exclusion applies even when another allowed group
is present. This is one indication requirement: appropriate disease setting,
prior chemotherapy, and the indicated MAGE-A4 companion assay remain necessary.
[FDA TECELRA label, June 2026](https://www.fda.gov/media/180565/download?attachment=).

## Protein-group provenance

P groups are resolved using the complete, **unmodified**
`hla_nom_p.txt` from IPD-IMGT/HLA **3.65.0**, dated July 14, 2026. The package
stores the original bytes in `trufflepig/data/hla_nom_p.txt.gz`; compression
does not alter the source content. SHA-256 of the decompressed source:

```text
f27deb430304d70100ffc2d91816d2e1b1fd5ab9bca26152eded8a41f2c8df71
```

Membership is read from this reference rather than guessed from an allele-name
prefix. For example, the reference places A*02:09 in A*02:01P. Missing or
ambiguous membership remains unresolved. No reference download occurs during
analysis.

Source: [IPD-IMGT/HLA WMDA protein groups](https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/wmda/hla_nom_p.txt).
The original attribution and license are included in
`trufflepig/data/LICENCE-IMGT-HLA.md`.

Please cite Barker DJ et al., *The IPD-IMGT/HLA Database: recent developments
in sequence submission*, Nucleic Acids Research (2026), 54(D1): D1152–D1158;
Robinson J, Barker D, Marsh SGE, *25 years of the IPD-IMGT/HLA Database*, HLA
(2024), 103(6): e15549; and Robinson J et al., *IMGT/HLA — a sequence database
for the human major histocompatibility complex*, Tissue Antigens (2000),
55:280–287.

## Rendering

The report uses an inline Markdown parser with a nomenclature-aware literal
asterisk rule. HLA identifiers remain intact inside ordinary text, emphasis,
links, and code spans. Report formatting cannot change an allele's identity.
