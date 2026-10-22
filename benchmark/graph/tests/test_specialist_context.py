"""Readable specialist protocol invariants; execute only in the Slurm check job."""
from copy import deepcopy
import json

import pytest

from app.models import OperationOptions, VerificationReport
from graph_local.backends import LocalHFRoles
from graph_local.menu_records import prepare_menu_request
from test_backends import FakeHF


def payload():
    cached = {'program.running': {'value': False, 'unit': '1', 'quality': 'estimated',
        'sample_time_s': 0.0, 'sample_window_s': None}}
    read = {'kind': 'read', 'instrument_id': 'furnace', 'state_ids': ['program.running']}
    start = {'kind': 'invoke', 'instrument_id': 'furnace', 'capability_id': 'thermal.start_program', 'arguments': {}}
    busy = {'code': 'BUSY', 'message': 'Stop the active program first.', 'retryable': False}
    def receipt(action, response):
        return {'proposed_action': deepcopy(action), 'executed_action': deepcopy(action),
                'corrected': False, 'response': response}
    actions = [read, start, {'kind': 'wait', 'dt_s': 30},
        {'kind': 'invoke', 'instrument_id': 'furnace', 'capability_id': 'power.set', 'arguments': {'enabled': False}}]
    return {'target': {'goal': 'Complete the three-segment public program'},
        'observations': {'inspection': {'id': 'wrapper-3', 'revision': 3},
                         'measurement': {'id': 'wrapper-3', 'revision': 3}},
        'last_error': busy, 'previous_hypotheses': ['OBSOLETE_MODEL_PROPOSAL'],
        'benchmark_public_context': {'wrapper_revision': 3,
            'public_task': {'initial_context': {'goal': 'OBSOLETE_INITIAL_GOAL', 'private': 'SECRET_ORACLE'}},
            'current_public_view': {'goal': 'Complete the three-segment public program',
                'selected_instrument': 'furnace', 'observation': {'clock': {'time_s': 0.0}, 'observations': cached},
                'manual': ['Explicit reads observe; waits advance simulation.'],
                'contract': {'capabilities': {'thermal.start_program': {
                    'description': 'Start the configured program', 'preconditions': ['Program is not running'],
                    'input_schema': {'type': 'object', 'properties': {}},
                    'output_schema': {'description': 'OMITTED_OUTPUT_SCHEMA'}}},
                    'state_slots': {'program.running': {'unit': '1', 'origin': 'derived', 'access': 'read_only',
                        'value_schema': {'description': 'OMITTED_VALUE_SCHEMA'}}}},
                'elapsed_sim_time_s': 0.0, 'budget': {'decisions_used': 3, 'max_decisions': 32}},
            'complete_native_history': [
                receipt(read, {'status': 'succeeded', 'clock': {'time_s': 0.0}, 'observations': deepcopy(cached)}),
                receipt(start, {'status': 'succeeded', 'clock': {'time_s': 0.0},
                    'result': {'running': True}, 'observations': {}, 'error': None}),
                receipt(start, {'status': 'rejected', 'clock': {'time_s': 0.0},
                    'result': {}, 'observations': {}, 'error': deepcopy(busy)})],
            'complete_offered_commands_and_presets': [{'command': {'text': 'Command ' + str(index)},
                'argument_presets': [{'text': 'Literal preset ' + str(index), 'action': action}]}
                for index, action in enumerate(actions)]}}


def request(source=None, role='diagnostic'):
    return prepare_menu_request(role, source or payload(),
        OperationOptions if role == 'diagnostic' else VerificationReport, protocol='menu-ids-v2')


def generated(identifier='m0'):
    return {'assessment': 'Use public readings and newer native receipts.', 'candidates': [
        {'menu_action_id': identifier, 'rationale': 'Compare the available evidence.',
         'postcondition': 'Check the actual native response.', 'evidence_ids': ['wrapper-3']}]}


def test_readable_view_preserves_cached_values_and_distinct_newer_native_receipts():
    source = payload()
    before = deepcopy(source)
    value = request(source).payload
    assert value['goal'] == source['benchmark_public_context']['current_public_view']['goal']
    assert value['cached_native_observation'] == source['benchmark_public_context']['current_public_view']['observation']
    assert value['cached_native_observation']['observations']['program.running']['value'] is False
    receipts = value['native_receipts_in_execution_order']
    assert receipts[1]['response']['result']['running'] is True
    assert receipts[2]['response']['error']['code'] == 'BUSY'
    assert value['latest_native_receipt']['response']['error']['code'] == 'BUSY'
    assert value['latest_successful_native_receipt']['response']['result']['running'] is True
    assert [item['history_index'] for item in receipts] == [0, 1, 2]
    assert value['cached_reading_provenance']['latest_native_receipt_with_observations_index'] == 0
    assert value['cached_reading_provenance']['receipts_after_that_reading'] == 2
    assert value['cached_reading_provenance']['cached_values_equal_that_receipt'] is True
    assert value['graph_last_error'] == value['last_native_response_error'] == source['last_error']
    assert source == before


def test_every_offered_action_id_and_literal_preset_survives_without_encoding_or_ranking():
    source = payload()
    readable = request(source)
    old = prepare_menu_request('diagnostic', source, OperationOptions)
    assert readable.actions == old.actions
    menu = readable.payload['offered_menu_actions']
    for index, item in enumerate(menu):
        expected = source['benchmark_public_context']['complete_offered_commands_and_presets'][index]['argument_presets'][0]
        assert item == {'menu_action_id': 'm' + str(index), 'action': expected['action'], 'preset_text': expected['text']}
        bound = readable.bind(readable.schema.model_validate(generated(item['menu_action_id'])).model_dump())
        assert bound['candidates'][0]['intent']['arguments']['action'] == expected['action']
    text = json.dumps(readable.payload)
    assert all(marker not in text for marker in ('$public_ref', '$public_table', 'presence_bitmask'))
    # Even the unhelpful power-off action remains offered; no preferred action is added.
    assert menu[-1]['action']['arguments'] == {'enabled': False}
    assert readable.audit['ranked_or_filtered_actions'] is False


def test_explicit_projection_omits_initial_oracle_and_old_proposals_but_keeps_manual_units():
    value = request()
    text = json.dumps(value.payload)
    assert all(marker not in text for marker in ('SECRET_ORACLE', 'OBSOLETE_INITIAL_GOAL',
        'OBSOLETE_MODEL_PROPOSAL', 'OMITTED_OUTPUT_SCHEMA', 'OMITTED_VALUE_SCHEMA'))
    assert value.payload['current_public_contract']['state_slots']['program.running']['unit'] == '1'
    assert value.payload['public_manual'] == payload()['benchmark_public_context']['current_public_view']['manual']
    assert value.audit['source_payload_sha256']
    source = payload()
    source['benchmark_public_context']['current_public_view']['observation']['oracle'] = {'correct_action': 'wait'}
    with pytest.raises(ValueError, match='Non-public'):
        request(source)


def test_history_growth_keeps_recent_receipts_latest_read_and_failure_with_public_counts():
    source = payload()
    history = source['benchmark_public_context']['complete_native_history']
    history[:] = [history[0]] + [deepcopy(history[1]) for _ in range(999)]
    history[17] = deepcopy(payload()['benchmark_public_context']['complete_native_history'][2])
    result = request(source)
    assert result.audit['retained_history_indices'] == [0, 17, 994, 995, 996, 997, 998, 999]
    assert result.payload['public_history_progress']['native_decision_count'] == 1000
    assert result.payload['public_history_progress']['omitted_older_receipt_count'] == 992
    assert sum(item['count'] for item in result.payload['public_history_progress']['identical_executed_action_counts']) == 1000
    assert result.payload['cached_native_observation']['observations']['program.running']['sample_time_s'] == 0.0


def test_v2_short_wire_prose_and_citation_membership_are_explicit_without_changing_v1():
    newer = request()
    older = prepare_menu_request('diagnostic', payload(), OperationOptions)
    value = generated()
    value['candidates'][0]['rationale'] = 'x' * 121
    with pytest.raises(ValueError):
        newer.schema.model_validate(value)
    assert older.schema.model_validate(value)
    value = generated()
    value['candidates'][0]['evidence_ids'] = ['shared_49']
    with pytest.raises(ValueError):
        newer.schema.model_validate(value)
    verifier = request(role='verification')
    assert 'offered_menu_actions' not in verifier.payload
    assert verifier.payload['cached_native_observation'] == newer.payload['cached_native_observation']
    assert verifier.schema.model_validate({'status': 'unresolved', 'evidence_ids': ['wrapper-3'],
        'missing_evidence': ['x' * 160] * 4})
    with pytest.raises(ValueError):
        verifier.schema.model_validate({'status': 'unresolved', 'evidence_ids': [], 'missing_evidence': ['x' * 161]})


def test_v2_invocation_records_actual_readable_wire_and_preserves_original_binding():
    fake = FakeHF(raw=json.dumps(generated()), generated=50)
    source = payload()
    roles = LocalHFRoles({'record_protocol': 'menu-ids-v2'}, backend=fake)
    result = roles.invoke('diagnostic', 'Original prompt', source, OperationOptions)
    assert OperationOptions.model_validate(result).candidates[0].intent.arguments['action'] == request(source).actions['m0']
    wire = json.loads(fake.calls[0]['messages'][1]['content'])
    assert wire['protocol'] == 'menu-ids-v2'
    assert wire['native_receipts_in_execution_order'][-1]['response']['error']['code'] == 'BUSY'
    assert roles.last_call['record_protocol'] == 'menu-ids-v2'
    assert roles.last_call['source_payload_sha256'] == roles.last_call['menu_binding']['source_payload_sha256']
    assert roles.last_call['usage']['completion_tokens'] == 50
    text = fake.calls[0]['messages'][1]['content']
    assert text.index('"latest_native_receipt"') < text.index('"cached_native_observation"')


@pytest.mark.parametrize('protocol', ['menu-ids-v1', 'menu-ids-v2'])
def test_both_opt_in_record_versions_request_finite_whitespace(monkeypatch, protocol):
    import graph_local.guided_json as guidance
    captured = {}
    class Prefix:
        def receipt(self):
            return {'test_fixture_only': True}
    class FakeGuidance:
        def __init__(self, tokenizer, **kwargs):
            captured.update(kwargs)
        def for_schema(self, schema):
            return Prefix()
    monkeypatch.setattr(guidance, 'JsonSchemaGuidance', FakeGuidance)
    fake = FakeHF(raw=json.dumps(generated()))
    fake.tokenizer = object()
    roles = LocalHFRoles({'record_protocol': protocol, 'structured_output': 'json_schema_lmfe_v4'}, backend=fake)
    roles.invoke('diagnostic', 'Original prompt', payload(), OperationOptions)
    assert captured == {'max_consecutive_whitespaces': 16}
