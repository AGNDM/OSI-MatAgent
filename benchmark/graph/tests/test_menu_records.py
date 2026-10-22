"""Exact public-menu binding and deterministic transport, never model scores."""
from copy import deepcopy
import json

import pytest

from app.models import ActionProposal, OperationOptions, VerificationReport
from graph_local.backends import LocalHFRoles
from graph_local.menu_records import prepare_menu_request, procedure_envelope
from graph_local.public_context import restore_json
from test_backends import FakeHF


def payload():
    actions = [
        {'kind': 'invoke', 'instrument_id': 'public-device', 'capability_id': 'power.set', 'arguments': {'enabled': True}},
        {'kind': 'wait', 'dt_s': 30},
    ]
    return {'target': {'goal': 'Reach public target'},
        'observations': {'inspection': {'id': 'obs-7', 'revision': 7}, 'measurement': None},
        'benchmark_public_context': {'wrapper_revision': 7,
            'current_public_view': {'selected_instrument': 'public-device',
                'observation': {'temperature': {'value': 295., 'quality': 'estimated', 'sample_time_s': 4.}}},
            'complete_native_history': [],
            'complete_offered_commands_and_presets': [
                {'command': {'text': action['kind']}, 'argument_presets': [
                    {'text': 'Literal public preset', 'action': action}]} for action in actions]}}


def generated(menu_id='m0'):
    return {'assessment': 'Compare the public goal and current reading.', 'candidates': [
        {'menu_action_id': menu_id, 'rationale': 'Use the offered operation.',
         'postcondition': 'Observe the changed public state.', 'evidence_ids': ['obs-7']}]}


def test_public_ids_bind_to_exact_arguments_and_preserve_source():
    source = payload()
    before = deepcopy(source)
    request = prepare_menu_request('diagnostic', source, OperationOptions)
    wire = restore_json(request.payload)
    assert source == before
    assert wire['benchmark_public_context']['current_public_view'] == source['benchmark_public_context']['current_public_view']
    record = OperationOptions.model_validate(request.bind(request.schema.model_validate(generated()).model_dump()))
    expected = source['benchmark_public_context']['complete_offered_commands_and_presets'][0]['argument_presets'][0]['action']
    assert record.candidates[0].intent.arguments == {'action': expected}
    assert record.candidates[0].intent.capability_id == 'benchmark.invoke'
    assert record.candidates[0].candidate_id == 'm0'
    assert record.candidates[0].evidence_ids == ['obs-7']


def test_unknown_ids_citations_and_duplicates_are_rejected_without_replacement():
    request = prepare_menu_request('diagnostic', payload(), OperationOptions)
    with pytest.raises(ValueError):
        request.schema.model_validate(generated('fabricated'))
    wrong = generated()
    wrong['candidates'][0]['evidence_ids'] = ['private-or-invented']
    with pytest.raises(ValueError):
        request.schema.model_validate(wrong)
    duplicate = generated()
    duplicate['candidates'] *= 2
    with pytest.raises(ValueError, match='distinct'):
        request.bind(duplicate)


def test_procedure_copies_selected_action_and_fresh_transport_fields():
    source = {'intent': {'capability_id': 'benchmark.wait', 'arguments': {'action': {'kind': 'wait', 'dt_s': 30}},
                        'postcondition': 'Read after the offered wait.'},
              'sample_id': 'sample-a', 'expected_revision': 7, 'request_id': 'fresh-request-7'}
    before = deepcopy(source)
    result = procedure_envelope(source, ActionProposal)
    assert result['arguments'] == source['intent']['arguments']
    assert result['request_id'] == 'fresh-request-7' and result['expected_revision'] == 7
    assert source == before


def test_menu_role_reaches_original_app_contract_and_accounts_actual_generation():
    fake = FakeHF(raw=json.dumps(generated()), generated=55)
    roles = LocalHFRoles({'model_path': '/fixture/not-loaded', 'record_protocol': 'menu-ids-v1'}, backend=fake)
    record = roles.invoke('diagnostic', 'Original app prompt', payload(), OperationOptions)
    assert OperationOptions.model_validate(record).candidates[0].candidate_id == 'm0'
    assert len(fake.calls) == 1
    assert roles.last_call['usage']['completion_tokens'] == 55
    assert roles.last_call['menu_binding']['menu_action_count'] == 2
    assert roles.last_call['bound_application_record'] == record
    procedure = {'intent': record['candidates'][0]['intent'], 'sample_id': 'sample-a',
                 'expected_revision': 7, 'request_id': 'request-7'}
    result = roles.invoke('procedure', 'Copy selected operation.', procedure, ActionProposal)
    assert len(fake.calls) == 1
    assert result['arguments'] == record['candidates'][0]['intent']['arguments']
    assert roles.last_call['request_sent'] is False and roles.last_call['usage'] is None


def test_verification_keeps_native_quality_timestamp_history_and_app_record():
    source = payload()
    request = prepare_menu_request('verification', source, VerificationReport)
    assert restore_json(request.payload) == source
    assert request.schema is not VerificationReport
    result = {'status': 'unresolved', 'evidence_ids': ['obs-7'], 'missing_evidence': ['A fresh native reading']}
    wire = request.schema.model_validate(result).model_dump(mode='json')
    assert VerificationReport.model_validate(request.bind(wire)).model_dump(mode='json') == result
    assert request.audit['evidence_ids'] == ['obs-7']


@pytest.mark.parametrize('citations', [[], ['obs-7']])
@pytest.mark.parametrize('status', ['supported', 'unresolved', 'contradicted'])
def test_verification_schema_permits_all_verdicts_and_empty_or_current_citations(status, citations):
    request = prepare_menu_request('verification', payload(), VerificationReport)
    value = {'status': status, 'evidence_ids': citations, 'missing_evidence': []}
    validated = request.schema.model_validate(value).model_dump(mode='json')
    assert VerificationReport.model_validate(request.bind(validated)).status == status


@pytest.mark.parametrize('citations', [['shared_49'], ['obs-6'], ['private-or-invented'], ['obs-7', 'shared_49']])
def test_verification_rejects_shared_refs_unknown_and_stale_ids(citations):
    request = prepare_menu_request('verification', payload(), VerificationReport)
    with pytest.raises(ValueError):
        request.schema.model_validate({'status': 'supported', 'evidence_ids': citations, 'missing_evidence': []})


def test_verification_wire_bounds_explanation_length_without_changing_original_app_schema():
    request = prepare_menu_request('verification', payload(), VerificationReport)
    boundary = {'status': 'unresolved', 'evidence_ids': [], 'missing_evidence': ['x' * 240] * 4}
    result = request.bind(request.schema.model_validate(boundary).model_dump(mode='json'))
    assert VerificationReport.model_validate(result).missing_evidence == boundary['missing_evidence']
    for missing in (['x' * 241], ['brief gap'] * 5):
        value = {**boundary, 'missing_evidence': missing}
        with pytest.raises(ValueError):
            request.schema.model_validate(value)
        # Explanation limits are exclusive to this opt-in wire protocol.
        assert VerificationReport.model_validate(value).missing_evidence == missing
    assert request.audit['missing_evidence_limits'] == {'max_items': 4, 'max_characters_per_item': 240}


def test_verification_without_observations_omits_wire_citations_and_binds_empty_list():
    source = payload()
    source['observations'] = {'inspection': None, 'measurement': None}
    before = deepcopy(source)
    request = prepare_menu_request('verification', source, VerificationReport)
    assert 'evidence_ids' not in request.schema.model_json_schema()['properties']
    value = {'status': 'unresolved', 'missing_evidence': ['No current public reading']}
    result = request.bind(request.schema.model_validate(value).model_dump(mode='json'))
    assert VerificationReport.model_validate(result).evidence_ids == []
    assert source == before and restore_json(request.payload) == source
    with pytest.raises(ValueError):
        request.schema.model_validate({**value, 'evidence_ids': ['shared_49']})
    with pytest.raises(ValueError, match='must omit'):
        request.bind({**value, 'evidence_ids': []})


def test_menu_verifier_invocation_rejects_bad_citation_after_actual_wire_validation():
    value = {'status': 'supported', 'evidence_ids': ['shared_49'], 'missing_evidence': []}
    fake = FakeHF(raw=json.dumps(value))
    roles = LocalHFRoles({'model_path': '/fixture/not-loaded', 'record_protocol': 'menu-ids-v1'}, backend=fake)
    with pytest.raises(ValueError):
        roles.invoke('verification', 'Original verification prompt', payload(), VerificationReport)
    assert len(fake.calls) == 1
    assert roles.last_call['request_sent'] is True
    assert roles.last_call['usage']['completion_tokens'] == fake.generated


def test_menu_verifier_without_observations_reaches_original_schema_with_empty_citations():
    source = payload()
    source['observations'] = {}
    fake = FakeHF(raw=json.dumps({'status': 'unresolved', 'missing_evidence': ['Need a public reading']}))
    roles = LocalHFRoles({'model_path': '/fixture/not-loaded', 'record_protocol': 'menu-ids-v1'}, backend=fake)
    result = roles.invoke('verification', 'Original verification prompt', source, VerificationReport)
    assert VerificationReport.model_validate(result).evidence_ids == []
    assert roles.last_call['bound_application_record'] == result


def test_default_verifier_uses_unchanged_application_schema():
    value = {'status': 'unresolved', 'evidence_ids': ['shared_49'], 'missing_evidence': []}
    fake = FakeHF(raw=json.dumps(value))
    roles = LocalHFRoles({'model_path': '/fixture/not-loaded'}, backend=fake)
    assert roles.invoke('verification', 'Original verification prompt', payload(), VerificationReport) == value
    assert 'menu_binding' not in roles.last_call


def test_no_public_menu_cannot_be_replaced_by_an_inferred_action():
    source = payload()
    source['benchmark_public_context']['complete_offered_commands_and_presets'] = []
    with pytest.raises(ValueError, match='No public actions'):
        prepare_menu_request('diagnostic', source, OperationOptions)


def test_preparation_error_does_not_reuse_a_previous_role_call_usage():
    fake = FakeHF(raw=json.dumps(generated()), generated=55)
    roles = LocalHFRoles({'model_path': '/fixture/not-loaded', 'record_protocol': 'menu-ids-v1'}, backend=fake)
    roles.invoke('diagnostic', 'Propose candidates.', payload(), OperationOptions)
    assert roles.last_call['request_sent'] and roles.last_call['usage']['completion_tokens'] == 55
    bad = payload()
    bad['benchmark_public_context']['complete_offered_commands_and_presets'] = []
    with pytest.raises(ValueError, match='No public actions'):
        roles.invoke('diagnostic', 'Propose candidates.', bad, OperationOptions)
    assert roles.last_call['request_sent'] is False and roles.last_call['usage'] is None
    assert roles.last_call['response_classification'] == 'input_preparation'
    assert roles.last_call['error']


def test_preparation_error_does_not_reuse_a_previous_chooser_usage():
    from graph_local.backends import LocalDecisionBackend
    from test_backends import FakeChooser
    chooser = LocalDecisionBackend({'kind': 'hf', 'model_path': '/fixture/not-loaded',
        'context_protocol': 'decision-view-v1', 'context_max_bytes': 5000}, backend=FakeChooser())
    chooser.decide({'has_intent': False}, ['diagnose', 'verify'])
    assert chooser.last_call['request_sent'] and chooser.last_call['usage'] is not None
    with pytest.raises(ValueError):
        chooser.decide({'last_error': {'message': 'x' * 6000}}, ['diagnose', 'verify'])
    assert chooser.last_call['request_sent'] is False and chooser.last_call['usage'] is None
    assert chooser.last_call['error']
