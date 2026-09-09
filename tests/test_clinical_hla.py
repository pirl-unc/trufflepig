"""Clinical typing retains source conflicts and feeds the main report contract."""
import json

import pandas as pd
import pytest

from trufflepig.clinical_context import (
    ClinicalAssay, ClinicalContext, ClinicalSource, clinical_context_for_analysis,
    evaluate_clinical_hla, evaluate_msi_mmr, load_clinical_context,
    normalize_clinical_inputs,
)
from trufflepig.hla import hla_typings_conflict


def typing(alleles=("A*02:01",), **kwargs):
    fields = dict(
        kind="hla", result="typed", alleles=alleles, complete_loci=("A",),
        method="NGS", specimen_id="specimen-A", scope="current",
        validity="validated", reportability="reportable", reported_at="2026-08-01",
        source=ClinicalSource(title="Synthetic HLA report", reference="report-one",
                              excerpt=json.dumps(alleles) if alleles else "Typing result unavailable"),
    )
    fields.update(kwargs)
    return ClinicalAssay(**fields)


def context(*assays):
    return ClinicalContext(specimen_id="specimen-A", assays=assays)


def decision(*assays):
    return evaluate_clinical_hla(context(*assays), required=("A*02:01P",), excluded=("A*02:05P",))


@pytest.mark.parametrize("alleles,status", [
    (("A*02:01:01:01",), "matched"),
    (("A*02:05",), "excluded"),
    (("A*02:01", "A*02:05"), "excluded"),
    (("A*24:02",), "mismatched"),
    (("A*02",), "insufficient_resolution"),
    (("A*02:01:01:02N",), "mismatched"),
])
def test_usable_clinical_typing_uses_the_existing_nomenclature_policy(alleles, status):
    result = decision(typing(alleles))
    assert result.status == status
    assert result.assays[0]['alleles'] == list(alleles)
    assert result.assays[0]['limitations'] == []
    assert result.assays[0]['source']['reference'] == 'report-one'
    assert result.nomenclature_version == 'IPD-IMGT/HLA 3.65.0'


@pytest.mark.parametrize("kwargs,reason", [
    ({"validity": "unverified"}, "unverified"),
    ({"validity": "failed"}, "failed"),
    ({"reportability": "unreportable"}, "unreportable"),
    ({"scope": "historical"}, "historical"),
    ({"scope": "other_specimen"}, "other_specimen"),
    ({"specimen_id": "specimen-B"}, "different specimen"),
    ({"source": ClinicalSource(title="Proposed", review_status="proposed")}, "proposed"),
    ({"source": ClinicalSource()}, "source is missing"),
    ({"complete_loci": ()}, "complete typing"),
    ({"method": ""}, "method"),
    ({"result": "pending"}, "pending"),
    ({"result": "not_tested"}, "not_tested"),
    ({"result": "unknown"}, "unknown"),
])
def test_unusable_or_incomplete_typing_never_becomes_a_positive_gate(kwargs, reason):
    result = decision(typing(**kwargs))
    assert result.status == 'unresolved'
    assert reason in result.reason
    assert result.assays[0]['alleles'] == ['A*02:01']


def test_missing_and_unavailable_typing_are_different():
    assert decision().status == 'unknown'
    result = decision(typing((), result='not_tested', complete_loci=()))
    assert result.status == 'unresolved' and 'not_tested' in result.reason


def test_complete_locus_is_required_to_infer_mismatch_or_clear_exclusion():
    assert decision(typing(("A*24:02",), complete_loci=())).status == 'unresolved'
    assert decision(typing(("A*02:01",), complete_loci=())).status == 'unresolved'
    assert decision(typing(("A*02:05",), complete_loci=())).status == 'excluded'
    assert evaluate_clinical_hla(context(typing(complete_loci=())), required=['A*02']).status == 'matched'


@pytest.mark.parametrize("left,right,left_complete,right_complete,expected", [
    (("A*02",), ("A*02:01:01:01",), ("A",), ("A",), False),
    (("A*02:01",), ("A*24:02",), ("A",), ("A",), True),
    (("A*02:01",), ("A*24:02",), (), (), False),
    (("A*02:01", "A*24:02"), ("A*02:01", "A*03:01"), ("A",), ("A",), True),
    (("A*02:01P",), ("A*02:09",), ("A",), ("A",), False),
    (("A*02:01P",), ("A*24:02",), ("A",), ("A",), True),
    (("A*02:01N",), ("A*02:01",), ("A",), ("A",), True),
])
def test_conflict_matching_preserves_resolution_groups_and_annotations(left, right, left_complete, right_complete, expected):
    assert hla_typings_conflict(left, right, left_complete=left_complete, right_complete=right_complete) is expected


def test_discordant_usable_reports_are_not_merged_into_a_positive_genotype():
    first = typing()
    second = typing(("A*24:02",), source=ClinicalSource(title="Second HLA report"))
    result = decision(first, second)
    assert result.status == 'conflicting'
    assert len(result.assays) == 2
    assert {r['compatibility']['status'] for r in result.assays} == {'matched', 'mismatched'}
    assert evaluate_msi_mmr(context(first, second)).status == 'missing'
    # Both reports can pass a broad A*02 gate yet disagree on the other allele.
    result = decision(typing(("A*02:01", "A*24:02")), typing(("A*02:01", "A*03:01")))
    assert result.status == 'conflicting'


@pytest.mark.parametrize('kwargs', [
    {'scope': 'historical'}, {'validity': 'failed'},
    {'source': ClinicalSource(title='Unreviewed', review_status='proposed')},
])
def test_unusable_discordant_report_is_retained_without_overriding_current_typing(kwargs):
    result = decision(typing(), typing(("A*24:02",), **kwargs))
    assert result.status == 'matched'
    assert len(result.assays) == 2 and result.assays[1]['limitations']


def test_legacy_typing_normalizes_once_without_inventing_clinical_quality():
    from trufflepig.analyze.models import AnalyzeConfig

    config = AnalyzeConfig(input_path='synthetic.tsv', clinical_context=context(), hla_types='HLA-A0201')
    ctx = config.clinical_context
    assert ctx.assays[0].alleles == ('A*02:01',)
    assert ctx.assays[0].validity == 'unverified'
    assert ctx.assays[0].reportability == 'unknown'
    assert ctx.assays[0].complete_loci == ()
    assert ctx.assays[0].source.excerpt == 'HLA-A0201'
    assert normalize_clinical_inputs(ctx, hla_types='A*02:01') == ctx
    assert clinical_context_for_analysis({'clinical_context': ctx.public_dict(), 'analysis_constraints': {'hla_types': ['A*02:01']}}) == ctx
    assert load_clinical_context(json.loads(json.dumps(ctx.public_dict()))) == ctx
    assert decision(ctx.assays[0]).status == 'unresolved'


def test_legacy_normalization_rejects_conflicting_ids_instead_of_losing_typing():
    ctx = normalize_clinical_inputs(context(), hla_types='A*02:01')
    collision = ClinicalAssay(**{**ctx.assays[0].public_dict(), 'alleles': ['A*24:02']})
    with pytest.raises(ValueError, match='conflicts with an existing assay ID'):
        normalize_clinical_inputs(context(collision), hla_types='A*02:01')


def test_explicit_typing_and_legacy_input_both_survive_normalization():
    ctx = normalize_clinical_inputs(context(typing()), hla_types=['A*24:02'])
    assert len(ctx.assays) == 2
    # The legacy assertion's quality is unresolved; it cannot cancel the usable report.
    assert evaluate_clinical_hla(ctx, required=['A*02']).status == 'matched'
    assert ctx.assays[1].alleles == ('A*24:02',)


@pytest.mark.parametrize('kwargs', [
    {'alleles': ['A*02:01/A*02:05']}, {'alleles': [None]}, {'alleles': 'A*02:01'},
    {'complete_loci': ['DQB1']}, {'complete_loci': [['A']]}, {'complete_loci': ['B']},
])
def test_invalid_or_ambiguous_hla_assertions_are_rejected(kwargs):
    with pytest.raises(ValueError):
        typing(**kwargs)


def synthetic_analysis(ctx):
    return {
        'sample_mode': 'mesenchymal',
        'cancer_type': 'SARC_SYN', 'cancer_type_source': 'user-specified',
        'cancer_type_confidence': 'moderate', 'purity': {},
        'purity_lo': 0.5, 'purity_hi': 0.9, 'purity_method': 'synthetic',
        'purity_confidence': 'moderate', 'clinical_context': ctx.public_dict(),
    }


@pytest.mark.parametrize('case,expected', [
    ('matched', 'reviewable'), ('mismatched', 'clinical_blocker'),
    ('conflicting', 'eligibility_pending'), ('pending', 'eligibility_pending'),
])
def test_answering_hla_request_updates_shared_report_and_preserves_sources(tmp_path, case, expected):
    from pypdf import PdfReader
    from trufflepig.analyze.models import AnalyzeConfig, AnalyzePaths, AnalyzeRun, InputResolution
    from trufflepig.analyze.flow import write_analysis_output_records
    from trufflepig.report_content import build_report_content, render_report_summary
    from trufflepig.report_view import build_report_view

    ranges = pd.DataFrame([dict(
        symbol='MAGEA4', observed_tpm=100.0, attr_tumor_tpm=90.0,
        attr_tumor_tpm_low=80.0, attr_tumor_tpm_high=100.0, attr_tumor_fraction=0.9,
        attr_tumor_fraction_low=0.8, attr_tumor_fraction_high=1.0,
        attr_support_fraction=1.0, attr_top_compartment='tumor',
        tme_dominant=False, tme_explainable=False,
    )])
    analysis = synthetic_analysis(context())
    view = build_report_view(analysis, sample_id='synthetic-hla')
    initial = build_report_content(analysis, ranges, 'SARC_SYN', '', report_view=view)
    request = next(r for r in initial.evidence_requests if r['kind'] == 'hla')
    assays = {
        'matched': (typing(),),
        'mismatched': (typing(('A*24:02',)),),
        'conflicting': (typing(), typing(('A*24:02',), source=ClinicalSource(title='Second HLA report [amended] *source*'))),
        'pending': (typing((), result='pending', complete_loci=()),),
    }[case]
    ctx = context(*assays)
    analysis['clinical_context'] = ctx.public_dict()
    content = build_report_content(analysis, ranges, 'SARC_SYN', '', report_view=view)
    afami = next(a for a in content.therapy_assessments if a['agent'] == 'afami-cel (Tecelra)')
    assert afami['selection']['status'] == expected
    followup = [r for r in content.evidence_requests if r['kind'] == 'hla']
    assert bool(followup) is (case in {'conflicting', 'pending'})
    if followup:
        assert followup[0]['id'] == request['id']
        assert followup[0]['evidence'][0]['assays']
    assert not any(r['kind'] == 'msi_high' for r in content.evidence_requests)
    prefix = 'synthetic-hla-' + case
    summary = render_report_summary(content)
    (tmp_path / f'{prefix}-summary.md').write_text(summary)
    run = AnalyzeRun(AnalyzeConfig(input_path='synthetic.tsv', clinical_context=ctx),
                     InputResolution('synthetic.tsv', None, False, 'gene'),
                     AnalyzePaths(tmp_path, prefix, prefix))
    write_analysis_output_records(run, view, content=content)
    doc = json.loads((tmp_path / f'{prefix}-report.json').read_text())
    manifest = json.loads((tmp_path / f'{prefix}-manifest.json').read_text())
    assert doc['clinical_context'] == manifest['config']['clinical_context'] == ctx.public_dict()
    pdf = PdfReader(tmp_path / f'{prefix}-interpretive-report.pdf')
    text = ' '.join(' '.join(page.extract_text() for page in pdf.pages).split())
    for phrase in ('Synthetic HLA report', 'specimen-A', '2026-08-01'):
        assert phrase in summary and phrase in text
    for assay in assays:
        assert assay.source.title in text
        for allele in assay.alleles:
            assert allele in summary and allele in text
    if case == 'conflicting':
        assert 'reports conflict' in summary and 'reports conflict' in text
    elif case == 'mismatched':
        assert 'HLA mismatch' in summary and 'HLA mismatch' in text
