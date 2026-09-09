"""Companion-assay answers drive the same clinical requirements and documents."""
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from trufflepig.clinical_context import (
    ClinicalAssay, ClinicalAssayCriterion, ClinicalContext, ClinicalSource,
    evaluate_clinical_assay, evaluate_msi_mmr, load_clinical_context,
)
from trufflepig.therapy_eligibility import (
    clinical_assay_requirements, collect_evidence_requests, evaluate_therapy_eligibility,
)


def criterion(analyte='PTEN'):
    return ClinicalAssayCriterion(
        kind='ihc', analyte=analyte,
        label='PTEN deficiency by IHC' if analyte == 'PTEN' else 'MAGE-A4 IHC',
        positive_results=('lost', 'deficient') if analyte == 'PTEN' else ('positive',),
        negative_results=('retained', 'not_deficient') if analyte == 'PTEN' else ('negative',),
        accepted_test_ids=('FDA:P250031',) if analyte == 'PTEN' else ('FDA:P230016',),
        specimen_type='tissue', source='https://example.test/clinical-criterion',
    )


def assay(result='deficient', analyte='PTEN', **kwargs):
    values = dict(
        kind='ihc', analyte=analyte, result=result, method='IHC',
        test_id='FDA:P250031' if analyte == 'PTEN' else 'FDA:P230016',
        specimen_type='tissue', specimen_id='specimen-A', scope='current',
        validity='validated', reportability='reportable', reported_at='2026-08-01',
        source=ClinicalSource(title='Synthetic companion assay', reference='page 2',
                              excerpt=f'{analyte}: {result}; reported clinical result'),
    )
    values.update(kwargs)
    return ClinicalAssay(**values)


def context(*assays):
    return ClinicalContext(specimen_id='specimen-A', assays=assays)


def analysis(ctx):
    return dict(cancer_type='PRAD', cancer_type_source='user-specified',
                analysis_constraints={'cancer_type': 'PRAD'}, sample_mode='solid',
                purity={}, clinical_context=ctx.public_dict())


def row():
    return dict(cancer_code='PRAD', symbol='AKT1', agent='Synthetic companion pathway',
                indication_biomarker='clinical_target_assay', requires_supplied_variant=True,
                clinical_assay_criteria=[criterion().public_dict()])


@pytest.mark.parametrize('result,state', [
    ('deficient', 'positive'), ('lost', 'positive'), ('retained', 'negative'),
    ('not_deficient', 'negative'), ('positive', 'unresolved'), ('negative', 'unresolved'),
    ('pending', 'unresolved'), ('not_tested', 'unresolved'), ('unknown', 'unresolved'),
])
def test_reported_pten_deficiency_is_a_distinct_assay_endpoint(result, state):
    ctx = context(assay(result))
    decision = evaluate_clinical_assay(ctx, criterion=criterion())
    assert decision.status == state
    assert decision.assays[0]['result'] == result
    assert evaluate_msi_mmr(ctx).status == 'missing'
    assert load_clinical_context(json.loads(json.dumps(ctx.public_dict()))) == ctx


@pytest.mark.parametrize('changes,limit', [
    ({'validity': 'failed'}, 'failed'), ({'validity': 'unverified'}, 'unverified'),
    ({'reportability': 'unknown'}, 'reportability'), ({'scope': 'historical'}, 'historical'),
    ({'specimen_id': 'specimen-B'}, 'different specimen'),
    ({'test_id': 'Unrelated antibody'}, 'test identity'),
    ({'specimen_type': 'plasma'}, 'tissue'), ({'method': 'RNA'}, 'method'),
    ({'source': ClinicalSource()}, 'source is missing'),
    ({'source': ClinicalSource(title='Proposed', review_status='proposed')}, 'proposed'),
])
def test_quality_scope_and_assay_limits_remain_unresolved(changes, limit):
    result = evaluate_clinical_assay(context(assay(**changes)), criterion=criterion())
    assert result.status == 'unresolved'
    assert any(limit in text for text in result.assays[0]['limitations'])


def test_current_source_conflicts_are_not_overwritten_by_recency():
    first, second = assay(), assay('retained', reported_at='2026-09-01')
    result = evaluate_clinical_assay(context(first, second), criterion=criterion())
    assert result.status == 'conflicting' and len(result.assays) == 2
    result = evaluate_clinical_assay(context(first, replace(second, scope='historical')), criterion=criterion())
    assert result.status == 'positive' and result.assays[1]['limitations']


@pytest.mark.parametrize('her2', ['ERBB2', 'HER2', 'her2'])
def test_other_analytes_and_assay_kinds_do_not_supply_the_requested_result(her2):
    mage = assay('positive', analyte='MAGE-A4')
    assert mage.analyte == 'MAGEA4'
    assert evaluate_clinical_assay(context(mage), criterion=criterion('MAGEA4')).status == 'positive'
    assert evaluate_clinical_assay(context(mage), criterion=criterion()).status == 'missing'
    source = ClinicalSource(title='Synthetic ISH report', excerpt=f'{her2}: amplified')
    ish = ClinicalAssay(kind='ish', analyte=her2, result='amplified', method='FISH',
                        test_id='synthetic-ISH', specimen_type='tissue', specimen_id='specimen-A',
                        scope='current', validity='validated', reportability='reportable',
                        source=source)
    assert ish.analyte == 'ERBB2' and ish.source.excerpt == f'{her2}: amplified'
    target = ClinicalAssayCriterion('ish', 'ERBB2', 'HER2 ISH', ('amplified',), ('not_amplified',),
                                   ('synthetic-ISH',), 'tissue', 'synthetic protocol')
    assert evaluate_clinical_assay(context(ish), criterion=target).status == 'positive'
    assert evaluate_clinical_assay(context(ish), criterion=criterion()).status == 'missing'


@pytest.mark.parametrize('supplied,canonical', [
    ('mage-a4', 'MAGEA4'), (' MAGEA4 ', 'MAGEA4'), ('her2', 'ERBB2'),
    ('erbb2', 'ERBB2'), ('pten', 'PTEN'), (None, ''), (pd.NA, ''), (float('nan'), ''),
])
def test_clinical_and_therapy_targets_use_one_public_normalizer(supplied, canonical):
    from trufflepig import reporting, therapeutic_agents

    assert reporting.canonical_target_symbol is therapeutic_agents.canonical_target_symbol
    assert therapeutic_agents.canonical_target_symbol(supplied) == canonical


@pytest.mark.parametrize('changes', [
    {'positive_results': ['pending']}, {'positive_results': ['retained']},
    {'negative_results': []}, {'accepted_test_ids': []}, {'source': ''}, {'analyte': ''},
    {'kind': 'msi'}, {'kind': []}, {'analyte': 'NaN'}, {'positive_results': 'deficient'},
])
def test_invalid_or_ambiguous_criteria_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(criterion(), **changes)


def test_two_declared_assays_remain_independent_of_the_drug_target():
    target = row()
    target['clinical_assay_criteria'].append(criterion('MAGEA4').public_dict())
    ctx = context(assay())
    decision = evaluate_therapy_eligibility(target, analysis(ctx))
    assert [(r.key, r.status) for r in decision.requirements] == [('ihc:PTEN', 'satisfied'), ('ihc:MAGEA4', 'missing')]
    assert not decision.permits_review
    requests = collect_evidence_requests([{'agent': target['agent'], 'eligibility': decision.public_dict()}])
    assert len(requests) == 1 and requests[0]['label'] == 'MAGE-A4 IHC'
    assert requests[0]['key'] == 'ihc:MAGEA4'
    answered = context(*ctx.assays, assay('positive', analyte='MAGEA4'))
    assert evaluate_therapy_eligibility(target, analysis(answered)).permits_review


def test_a_companion_result_does_not_satisfy_an_independent_wildtype_requirement():
    target = row()
    target.update(indication_biomarker='wildtype', symbol='ERBB2')
    decision = evaluate_therapy_eligibility(target, analysis(context(assay())))
    assert next(r for r in decision.requirements if r.kind == 'clinical_target_assay').status == 'satisfied'
    assert next(r for r in decision.requirements if r.kind == 'wildtype').status == 'missing'
    assert not decision.permits_review


def test_rna_and_a_supplied_pten_variant_do_not_replace_the_protein_assay():
    value = analysis(context())
    value.update(pten_loss=True, protein_status='deficient',
                 variant_records=[{'gene': 'PTEN', 'variant': 'PTEN loss', 'variant_type': 'deletion'}])
    value['analysis_constraints']['pten_deficient'] = True
    requirement = clinical_assay_requirements(row(), value)[0]
    assert requirement.status == 'missing'
    assert not evaluate_therapy_eligibility(row(), value).permits_review


def test_assay_requirement_cannot_be_reinterpreted_from_variant_prose():
    from trufflepig.reporting import supplied_variant_supports_target_row
    from trufflepig.variants import VariantRecord

    target = row()
    target.update(symbol='PTEN', rationale='PTEN amplification is additional context')
    value = analysis(context())
    value['variant_records'] = [VariantRecord(
        gene='PTEN', variant='PTEN amplification', variant_type='amplification',
        result_status='positive', source_path='synthetic.tsv',
    ).public_dict()]
    assert supplied_variant_supports_target_row(target, value) == []
    decision = evaluate_therapy_eligibility(target, value)
    assert not decision.supplied_variant_supported and not decision.permits_review


def test_curation_errors_cannot_silently_discard_clinical_corrections(monkeypatch):
    from trufflepig import reporting

    def invalid_criterion(row):
        raise ValueError('Invalid companion criterion')

    monkeypatch.setattr(reporting, '_current_therapy_row_overrides', invalid_criterion)
    with pytest.raises(ValueError, match='Invalid companion criterion'):
        reporting.filter_current_therapy_targets(pd.DataFrame([row()]))
    assert reporting.filter_current_therapy_targets(pd.DataFrame()).empty
    assert reporting.filter_current_therapy_targets(None) is None


def test_cli_and_web_preserve_companion_assay_results(tmp_path, monkeypatch):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from trufflepig import cli, main
    from trufflepig.analyze.models import AnalyzeConfig
    from trufflepig.web import WebSettings, create_app

    ish = ClinicalAssay(
        kind='ish', analyte='ERBB2', result='not_amplified', method='FISH',
        specimen_id='specimen-A', scope='current', validity='validated',
        reportability='reportable', source=ClinicalSource(title='Synthetic ISH report'),
    )
    ctx = context(assay('retained'), ish)
    path = tmp_path / 'clinical.json'
    path.write_text(json.dumps(ctx.public_dict()))
    captured = {}

    def capture_analysis(**kwargs):
        captured.update(AnalyzeConfig(**kwargs).public_dict())

    monkeypatch.setattr(main, 'analyze', capture_analysis)
    assert cli.main(['run', '--sample', 'synthetic.tsv', '--workspace', str(tmp_path / 'out'),
                     '--clinical-context', str(path)]) == 0
    assert captured['clinical_context'] == ctx.public_dict()

    def capture_run(cmd, log_path, status_path):
        captured['cmd'] = cmd

    monkeypatch.setattr('trufflepig.web.runs._spawn', capture_run)
    client = TestClient(create_app(WebSettings(runs_root=tmp_path / 'runs', uploads_root=tmp_path / 'uploads')))
    response = client.post('/api/run', files={
        'sample': ('sample.tsv', b'gene\tTPM\n', 'text/plain'),
        'clinical_context': ('context.json', json.dumps(ctx.public_dict()), 'application/json'),
    })
    assert response.status_code == 200, response.text
    cmd = captured['cmd']
    assert load_clinical_context(Path(cmd[cmd.index('--clinical-context') + 1])) == ctx


@pytest.mark.parametrize('result,expected', [(None, 'missing'), ('positive', 'satisfied'), ('negative', 'blocked')])
def test_afami_requires_the_companion_assay_independently_of_hla(clinical_hla_context, result, expected):
    from trufflepig.reporting import cancer_therapy_panel_for_analysis
    from trufflepig.therapy_eligibility import evaluate_therapy_review

    ctx = load_clinical_context(clinical_hla_context(['A*02:01']))
    if result is not None:
        finding = assay(result, analyte='MAGEA4', specimen_id=ctx.specimen_id)
        ctx = replace(ctx, assays=ctx.assays + (finding,))
    value = analysis(ctx)
    value.update(cancer_type='SARC_SYN', analysis_constraints={'cancer_type': 'SARC_SYN'})
    _, subtype, panel = cancer_therapy_panel_for_analysis('SARC_SYN', value)
    target = panel.loc[panel.agent.eq('afami-cel (Tecelra)')].iloc[0]
    decision = evaluate_therapy_review(target, analysis=value, panel_subtype=subtype)
    assert next(r for r in decision.eligibility.requirements if r.kind == 'hla').status == 'satisfied'
    assert next(r for r in decision.eligibility.requirements if r.key == 'ihc:MAGEA4').status == expected
    assert decision.permits_review is (expected == 'satisfied')
    if result == 'positive':
        from trufflepig.report_content import build_report_content
        from trufflepig.report_view import build_report_view

        view = build_report_view(value, sample_id='synthetic-companion')
        content = build_report_content(value, pd.DataFrame(), 'SARC_SYN', '', report_view=view)
        setting = next(r for r in content.evidence_requests if r['key'] == 'clinical_setting')
        setting_text = json.dumps(setting)
        assert 'prior chemotherapy' in setting_text and 'synovial-sarcoma' in setting_text
        assert 'MAGE-A4' not in setting_text and 'HLA' not in setting_text


@pytest.mark.parametrize('history', [
    {'therapy': 'abiraterone', 'status': 'contraindicated'},
    {'therapy': 'capivasertib + abiraterone + prednisone', 'status': 'progression'},
    {'therapy': 'capivasertib + abiraterone + prednisone', 'status': 'current'},
])
def test_positive_companion_result_retains_component_and_regimen_restrictions(history):
    from trufflepig.reporting import cancer_therapy_panel_for_analysis
    from trufflepig.therapy_eligibility import evaluate_therapy_review

    value = analysis(context(assay()))
    _, subtype, panel = cancer_therapy_panel_for_analysis('PRAD', value)
    target = panel.loc[panel.agent.eq('capivasertib + abiraterone + prednisone')].iloc[0]
    assert evaluate_therapy_review(target, analysis=value, panel_subtype=subtype).permits_review
    value['treatment_history'] = [history]
    decision = evaluate_therapy_review(target, analysis=value, panel_subtype=subtype)
    assert next(r for r in decision.eligibility.requirements if r.key == 'ihc:PTEN').status == 'satisfied'
    assert decision.status == 'clinical_blocker' and not decision.permits_review


@pytest.mark.parametrize('result,selected', [('deficient', True), ('retained', False), ('pending', False)])
def test_answering_pten_request_updates_the_real_panel_and_shared_documents(tmp_path, result, selected):
    from pypdf import PdfReader
    from trufflepig.analyze.flow import write_analysis_output_records
    from trufflepig.analyze.models import AnalyzeConfig, AnalyzePaths, AnalyzeRun, InputResolution
    from trufflepig.report_content import build_report_content, render_report_summary
    from trufflepig.report_view import build_report_view

    initial = analysis(context())
    view = build_report_view(initial, sample_id='synthetic-companion')
    content = build_report_content(initial, pd.DataFrame(), 'PRAD', '', report_view=view)
    request = next(r for r in content.evidence_requests if r['key'] == 'ihc:PTEN')
    assert request['label'] == 'PTEN deficiency by IHC'
    ctx = context(assay(result))
    initial['clinical_context'] = ctx.public_dict()
    content = build_report_content(initial, pd.DataFrame(), 'PRAD', '', report_view=view)
    assessed = next(a for a in content.therapy_assessments if a['agent'] == 'capivasertib + abiraterone + prednisone')
    assert assessed['selected'] is selected
    requirement = next(r for r in assessed['eligibility']['requirements'] if r['key'] == 'ihc:PTEN')
    assert assessed['rationale'].count(requirement['description']) == 1
    followup = [r for r in content.evidence_requests if r['key'] == 'ihc:PTEN']
    assert bool(followup) is (result == 'pending')
    if followup:
        assert followup[0]['id'] == request['id']
        assert 'pending' in followup[0]['question']
    assert not any(r['kind'] == 'msi_high' for r in content.evidence_requests)
    summary = render_report_summary(content)
    prefix = 'synthetic-companion-' + result
    (tmp_path / f'{prefix}-summary.md').write_text(summary)
    config = AnalyzeConfig(input_path='synthetic.tsv', clinical_context=ctx)
    run = AnalyzeRun(config, InputResolution('synthetic.tsv', None, False, 'gene'), AnalyzePaths(tmp_path, prefix, prefix))
    write_analysis_output_records(run, view, content=content)
    doc = json.loads((tmp_path / f'{prefix}-report.json').read_text())
    manifest = json.loads((tmp_path / f'{prefix}-manifest.json').read_text())
    assert doc['clinical_context'] == manifest['config']['clinical_context'] == ctx.public_dict()
    text = ' '.join(' '.join(p.extract_text() for p in PdfReader(tmp_path / f'{prefix}-interpretive-report.pdf').pages).split())
    for phrase in ('PTEN', 'FDA:P250031', 'Synthetic companion assay', 'specimen-A', '2026-08-01', result):
        assert phrase in summary and phrase in text
    if selected:
        assert 'hormone-sensitive' in summary
        setting = next(r for r in content.evidence_requests if r['key'] == 'clinical_setting')
        setting_text = json.dumps(setting)
        assert 'hormone-sensitive' in setting_text
        assert 'PTEN' not in setting_text and 'SP218' not in setting_text
    assert f'“PTEN: {result}; reported clinical result”. Reported:' in summary
    assert f'“PTEN: {result}; reported clinical result”. Reported:' in text
