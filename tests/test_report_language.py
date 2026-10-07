"""Template facts must not disappear or execute as template instructions."""

from types import SimpleNamespace

import pytest
from jinja2 import UndefinedError

from trufflepig.report_language import render_report_paragraph


def history_facts(**overrides):
    facts = dict(
        record=SimpleNamespace(status='contraindicated', note='Reaction {{ 1 + 1 }}',
                               source='Clinical review', source_path='history.tsv'),
        therapy='Keytruda', identity=SimpleNamespace(canonical_name='pembrolizumab'),
        action='contraindicated', match_kind='contraindicated_component',
        show_identity=True, unresolved_product=False,
    )
    facts.update(overrides)
    return facts


def test_template_retains_restriction_identity_and_source_without_evaluating_user_text():
    text = render_report_paragraph('treatment_history', **history_facts())
    for required in ('contraindication', 'Keytruda', 'pembrolizumab',
                     'component', 'excluded', 'Clinical review', '{{ 1 + 1 }}'):
        assert required in text
    assert '\n' not in text


def test_missing_template_facts_are_errors():
    facts = history_facts()
    del facts['action']
    with pytest.raises(UndefinedError):
        render_report_paragraph('treatment_history', **facts)


def test_template_names_cannot_select_arbitrary_files():
    with pytest.raises(ValueError):
        render_report_paragraph('../report', **history_facts())
