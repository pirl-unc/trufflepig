"""Patient restrictions use explicit drug and regimen identities."""

import json
from collections import defaultdict

import pandas as pd
import pytest

from trufflepig.brief import recommend_therapies
from trufflepig.reporting import cancer_therapy_panel_for_analysis
from trufflepig.therapeutic_agents import (
    agent_identity, agents_for_name, registered_agents, resolve_therapy_identity,
)
from trufflepig.treatment_history import (
    TreatmentRecord, treatment_history_blocks_row, treatment_history_context,
    treatment_history_supplement_rows,
)


@pytest.mark.parametrize('brand,generic', [
    ('Xtandi', 'enzalutamide'), ('Keytruda', 'pembrolizumab'),
    ('Padcev', 'enfortumab vedotin'), ('Bavencio', 'avelumab'),
    ('Stivarga', 'regorafenib'), ('Tagrisso', 'osimertinib'),
    ('Gleevec', 'imatinib'), ('Braftovi', 'encorafenib'),
    ('Lumakras', 'sotorasib'), ('Adriamycin', 'doxorubicin'),
    ('MGC018', 'vobramitamab duocarmazine'),
])
def test_curated_names_resolve_to_the_same_identity(brand, generic):
    identity = resolve_therapy_identity(brand)
    assert identity.registered
    assert identity.key == agent_identity(generic)


@pytest.mark.parametrize('first,second', [
    ('MGC018', 'DS-7300'), ('Doxil', 'Adriamycin'),
    ('Abraxane', 'paclitaxel'), ('Keytruda Qlex', 'Keytruda'),
    ('Cometriq', 'Cabometyx'), ('Yonsa', 'Zytiga'),
    ('89Zr-girentuximab', 'girentuximab'),
    ('177Lu-PSMA-617', '225Ac-PSMA-617'),
    ('177Lu-FAP-2286', '225Ac-FAPI-46'),
])
def test_distinct_products_are_not_equated(first, second):
    assert agent_identity(first) != agent_identity(second)


@pytest.mark.parametrize('name', ['Xtandi', 'enzalutamide', 'enzalutamide (Xtandi)'])
def test_brand_history_excludes_the_real_prostate_panel(name):
    analysis = {'cancer_type': 'PRAD', 'cancer_type_source': 'user-specified',
                'treatment_history': [{'therapy': name, 'status': 'contraindicated'}]}
    _, subtype, panel = cancer_therapy_panel_for_analysis('PRAD', analysis)
    selected = recommend_therapies(panel, pd.DataFrame(), analysis=analysis, panel_subtype=subtype)
    assert 'enzalutamide' not in [r.therapy['agent'] for r in selected]


@pytest.mark.parametrize('name', [
    'pembrolizumab', 'Keytruda', 'enfortumab vedotin', 'Padcev',
    'enfortumab vedotin + pembrolizumab', 'Keytruda + Padcev',
])
def test_component_contraindication_excludes_the_real_bladder_combination(name):
    history = TreatmentRecord(therapy=name, status='contraindicated',
                              note='Supplied restriction', source='Oncology note')
    analysis = {'cancer_type': 'BLCA', 'cancer_type_source': 'user-specified',
                'treatment_history': [history.public_dict()]}
    _, subtype, panel = cancer_therapy_panel_for_analysis('BLCA', analysis)
    selected = recommend_therapies(panel, pd.DataFrame(), analysis=analysis, panel_subtype=subtype)
    assert 'enfortumab vedotin + pembrolizumab' not in [r.therapy['agent'] for r in selected]
    row = panel.loc[panel.agent.eq('enfortumab vedotin + pembrolizumab')].iloc[0]
    context = treatment_history_context(row, analysis)
    assert 'excluded because of supplied treatment history' in context
    assert 'Oncology note' in context and 'Supplied restriction' in context
    assert not treatment_history_blocks_row({'agent': 'avelumab'}, analysis)


@pytest.mark.parametrize('status', ['progression', 'no_benefit', 'intolerance', 'current'])
def test_other_component_statuses_do_not_exclude_a_different_combination(status):
    analysis = {'treatment_history': [{'therapy': 'Keytruda', 'status': status}]}
    assert not treatment_history_blocks_row({'agent': 'enfortumab vedotin + pembrolizumab'}, analysis)
    if status != 'current':
        assert treatment_history_blocks_row({'agent': 'pembrolizumab'}, analysis)


def test_regimen_event_of_unknown_cause_does_not_become_component_contraindication():
    analysis = {'treatment_history': [{'therapy': 'Keytruda + Padcev', 'status': 'contraindicated'}]}
    assert not treatment_history_blocks_row({'agent': 'pembrolizumab'}, analysis)
    assert not treatment_history_blocks_row({'agent': 'enfortumab vedotin'}, analysis)
    assert treatment_history_blocks_row({'agent': 'pembrolizumab + enfortumab vedotin'}, analysis)


def test_alternative_drugs_are_not_treated_as_a_combination():
    identity = resolve_therapy_identity('EGFR TKI (afatinib / osimertinib)')
    assert identity.kind == 'alternatives'
    assert identity.key != agent_identity('afatinib + osimertinib')


def test_original_name_identity_and_file_provenance_survive_history_audit():
    record = TreatmentRecord(therapy='Keytruda', status='contraindicated',
                             source_path='clinical-history.tsv', row_index=4)
    payload = json.loads(json.dumps(record.public_dict()))
    assert payload['therapy'] == 'Keytruda'
    assert payload['therapy_identity']['canonical_name'] == 'pembrolizumab'
    restored = TreatmentRecord.from_mapping(payload)
    assert restored.source_path == record.source_path and restored.row_index == 4
    analysis = {'treatment_history': [payload]}
    combination = {'agent': 'enfortumab vedotin + pembrolizumab'}
    supplements = treatment_history_supplement_rows(analysis, cancer_code='BLCA', existing_rows=[combination])
    assert any(row['agent'] == 'Keytruda' for row in supplements)
    context = treatment_history_context(combination, analysis)
    assert 'Keytruda' in context and 'pembrolizumab' in context
    assert 'a component of this treatment' in context and 'clinical-history.tsv' in context


def test_all_registered_names_are_unambiguous_and_components_resolve():
    identities = defaultdict(set)
    for agent in registered_agents():
        for name in [agent.agent, *agent.aliases.split(';'), *agent.brand_name.split(';')]:
            if name.strip():
                records = agents_for_name(name)
                assert records, f'Unresolved registered name: {name}'
                identities[name.casefold().strip()].update(r.agent for r in records)
        for component in agent.components.split(';'):
            if component.strip():
                assert resolve_therapy_identity(component).registered, component
    assert all(len(names) == 1 for names in identities.values())


def test_every_upstream_curated_therapy_has_an_explicit_identity_kind():
    from pirlygenes.gene_sets_cancer import cancer_key_genes_df
    panel = cancer_key_genes_df()
    labels = panel.loc[panel.role.eq('target'), 'agent'].dropna().unique()
    unresolved = [name for name in labels if str(name).strip() and not resolve_therapy_identity(name).registered]
    assert not unresolved, unresolved


@pytest.mark.parametrize('brand,generic,other', [
    ('Cabometyx', 'cabozantinib', 'Cometriq'),
    ('Yonsa', 'abiraterone', 'Zytiga'),
])
def test_formulation_restriction_requires_reconciliation_when_product_is_unspecified(brand, generic, other):
    analysis = {'treatment_history': [{'therapy': brand, 'status': 'contraindicated'}]}
    assert treatment_history_blocks_row({'agent': generic}, analysis)
    assert not treatment_history_blocks_row({'agent': other}, analysis)
    context = treatment_history_context({'agent': generic}, analysis)
    assert 'formulation is unspecified' in context and 'Reconcile the exact product' in context
    generic_history = {'treatment_history': [{'therapy': generic, 'status': 'contraindicated'}]}
    assert treatment_history_blocks_row({'agent': brand}, generic_history)
